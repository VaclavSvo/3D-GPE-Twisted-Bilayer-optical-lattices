"""bilayer.ground_state: the dt ladder + preconditioned polish, batch
consistency, renormalisation, initial states, and the residual criterion."""

import math

import pytest
import torch

from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import dt_ladder, initial_state, relax, renormalise
from twisted_bilayer.engine.lattice import harmonic_trap, lattice_1d, toy_bilayer_1d
from twisted_bilayer.engine.physics import BilayerGPE

CPU = torch.device("cpu")
F64 = torch.float64


def test_ladder_and_polish_reach_1e_10():
    """1D lattice V0 = 8 on 4 sites, g = 0.2, one atom per site: tol 1e-10
    within 2000 steps in total (ladder stages + polish).

    Seeded with a uniform state, which keeps the lattice periodicity: with
    random phases (inter-site phonons, E_exc ~ 0.01) the same call stalls
    near 1e-6 after the 2000 polish steps, see the report on the stall
    criterion."""
    g = Grid((128,), (4.0,), device=CPU)
    eng = BilayerGPE(g, lattice_1d(g, 8.0), g=0.2, n_components=1)
    sched = dt_ladder(eng, initial_state(eng, 4.0))
    assert sched[0] == pytest.approx(1.0 / (8.0 + 0.2 * 1.0), rel=1e-12)  # 1 / E_loc
    assert sched[-1] == pytest.approx(0.5 / (eng.kappa * g.k_sq_max), rel=1e-12)
    res = relax(eng, initial_state(eng, 4.0), 4.0, tol=1e-10, polish_steps=2000)
    assert bool(res.converged.all()), res.report()
    assert float(res.residual[0]) < 1e-10
    assert res.steps <= 2000
    assert float(eng.norm(res.psi)[0]) == pytest.approx(4.0, rel=1e-13)
    # the state is stationary: residual recomputed from scratch agrees
    assert float(eng.residual(res.psi)[0]) == pytest.approx(float(res.residual[0]), rel=1e-6)


def _toy(gv, Om):
    ns = 6
    g = Grid((ns * 32,), (float(ns),), device=CPU)
    return BilayerGPE(g, toy_bilayer_1d(g, 8.0, ns), g=gv, Omega_R=Om, n_components=2), float(ns)


def test_batched_relax_equals_separate_runs():
    """Two members (different g and Omega_R) in one call vs two calls.
    Fixed schedule (no adaptive stopping): identical arithmetic per member,
    so the fields agree to 1e-12. Adaptive ladder + polish: each member
    stops at a different step, so the fields agree to ~tol, the energies
    (second order) to 1e-12."""
    gs, Om = [0.1, 0.3], [0.05, 0.2]
    engB, N = _toy(torch.tensor(gs, dtype=F64), torch.tensor(Om, dtype=F64))
    psi0 = initial_state(engB, N, batch=2, fractions=(0.4, 0.6))
    # (initial_state already starts component 2 at relative phase pi)
    fixed = dict(tol=0.0, dt_schedule=[0.02, 0.005], max_steps_per_stage=200, stall_ratio=10.0, polish_steps=0)
    rB = relax(engB, psi0, N, **fixed)
    rA = relax(engB, psi0, N, tol=1e-9, polish_steps=2000)
    assert bool(rA.converged.all()), rA.report()
    for i in range(2):
        eng, _ = _toy(gs[i], Om[i])
        r = relax(eng, psi0[i : i + 1].clone(), N, **fixed)
        assert r.steps == rB.steps
        assert float((r.psi - rB.psi[i : i + 1]).abs().max()) < 1e-12
        assert float(r.energy[0]) == pytest.approx(float(rB.energy[i]), abs=1e-12)
        a = relax(eng, psi0[i : i + 1].clone(), N, tol=1e-9, polish_steps=2000)
        assert bool(a.converged.all())
        assert float(a.energy[0]) == pytest.approx(float(rA.energy[i]), abs=1e-12)
        assert float((a.psi - rA.psi[i : i + 1]).abs().max()) < 1e-7


def test_renormalise_sets_norm():
    g = Grid((16, 16), (2.0, 3.0), device=CPU)
    eng = BilayerGPE(g, None, g=0.0, n_components=2)
    gen = torch.Generator().manual_seed(3)
    psi = torch.randn((3, 2, 16, 16), dtype=torch.complex128, generator=gen)
    out = renormalise(eng, psi, 7.5)
    assert torch.allclose(eng.norm(out), torch.full((3,), 7.5, dtype=F64), rtol=1e-14)
    target = torch.tensor([1.0, 2.0, 50.0], dtype=F64)
    out = renormalise(eng, psi, target)
    assert torch.allclose(eng.norm(out), target, rtol=1e-14)
    # a pure rescaling: the shape and the component ratio are untouched
    ratio_in = eng.component_norms(psi)[:, 0] / eng.norm(psi)
    ratio_out = eng.component_norms(out)[:, 0] / eng.norm(out)
    assert torch.allclose(ratio_in, ratio_out, rtol=1e-13)


@pytest.mark.parametrize("kind", ["uniform", "gaussian", "thomas_fermi"])
def test_initial_state_fractions_and_norm(kind):
    g = Grid((32, 32), (4.0, 4.0), device=CPU)
    V = harmonic_trap(g, (1.0, 1.0)).expand(1, 2, 32, 32).contiguous()
    eng = BilayerGPE(g, V, g=0.1, Omega_R=0.2, n_components=2)
    psi = initial_state(eng, 10.0, batch=3, kind=kind, fractions=(0.3, 0.7))
    assert psi.shape == (3, 2, 32, 32) and psi.dtype == torch.complex128
    assert torch.allclose(eng.norm(psi), torch.full((3,), 10.0, dtype=F64), rtol=1e-13)
    frac = eng.component_norms(psi) / eng.norm(psi).reshape(-1, 1)
    assert torch.allclose(frac, torch.tensor([[0.3, 0.7]] * 3, dtype=F64), atol=1e-13)
    if kind == "uniform":
        n = eng.density(psi)
        assert float(n.max() - n.min()) < 1e-13
    # phase noise changes phases, not densities
    noisy = initial_state(eng, 10.0, batch=3, kind=kind, fractions=(0.3, 0.7), phase_noise=0.5, seed=1)
    assert torch.allclose(noisy.abs(), psi.abs(), atol=1e-13)
    assert float((noisy - psi).abs().max()) > 1e-2


def test_initial_state_single_component():
    g = Grid((64,), (8.0,), device=CPU)
    eng = BilayerGPE(g, lattice_1d(g, 4.0), g=0.1, n_components=1)
    psi = initial_state(eng, 3.0, kind="thomas_fermi")
    assert psi.shape == (1, 1, 64)
    assert float(eng.norm(psi)[0]) == pytest.approx(3.0, rel=1e-13)


def test_residual_of_exact_eigenstate():
    """Uniform field, V = None, g > 0: H psi = g n psi = mu psi exactly, so
    the residual is zero to rounding (< 1e-12). For C = 2 the equal-sign
    uniform state is the sigma_x = +1 eigenstate of the Rabi term, also
    exact; mu = g n + Omega_R. Batched over g."""
    g = Grid((16, 12), (3.0, 2.0), device=CPU)
    gv = torch.tensor([0.0, 0.1, 2.0], dtype=F64)
    eng1 = BilayerGPE(g, None, g=gv, n_components=1)
    psi = initial_state(eng1, 5.0, batch=3)
    assert float(eng1.residual(psi).max()) < 1e-12
    nbar = 5.0 / g.volume
    assert torch.allclose(eng1.energy_terms(psi)["mu"], gv * nbar, atol=1e-13)
    eng2 = BilayerGPE(g, None, g=gv, Omega_R=0.3, n_components=2)
    psi2 = initial_state(eng2, 5.0, batch=3)
    assert float(eng2.residual(psi2).max()) < 1e-12
    assert torch.allclose(eng2.energy_terms(psi2)["mu"], gv * nbar - 0.3, atol=1e-13)  # lower dressed state
    # and a non-eigenstate (two plane waves of different |k|) does not pass
    k = 2 * math.pi / g.L[0]
    x = g.coord(0)
    bent = psi * (torch.exp(1j * k * x) + 0.3 * torch.exp(2j * k * x)).to(torch.complex128)
    assert float(eng1.residual(bent).min()) > 1e-2


def test_stop_reason_and_stall_window():
    """relax records why it ended: "tol" when converged; a run capped by polish_steps ends on
    "max_steps" (or "stall"). The polish stall test asks for 0.5 % over the last stall_windows check
    windows: a steady 0.3 % per window descent is a stall over one window but goes on over four; 0.1 %
    per window is a stall either way. The default is four windows for complex128 and one (the old test)
    for complex64, whose solves stop at the rounding floor."""
    from twisted_bilayer.engine.ground_state import _stalled, precondition_polish

    for rate, w1, w4 in ((0.003, True, False), (0.001, True, True), (0.01, False, False)):
        hist = [(1 - rate) ** k for k in range(6)]
        assert _stalled(hist, 1) is w1 and _stalled(hist, 4) is w4, rate
    assert not _stalled([1.0, 0.999], 4)  # fewer windows than the span: never a stall

    g = Grid((128,), (4.0,), device=CPU)
    eng = BilayerGPE(g, lattice_1d(g, 8.0), g=0.2, n_components=1)
    res = relax(eng, initial_state(eng, 4.0), 4.0, tol=1e-10, polish_steps=2000)
    assert res.stop_reason == "tol" and "stopped on tol" in res.report()
    capped = relax(eng, initial_state(eng, 4.0), 4.0, tol=1e-30, dt_schedule=[], polish_steps=100)
    assert capped.stop_reason in ("max_steps", "stall", "tau_floor") and not bool(capped.converged.all())
    e32 = eng.to_dtype(torch.complex64)
    psi = initial_state(e32, 4.0)
    out = {}
    for w in (None, 1, 4):
        info = {}
        p, r, n = precondition_polish(e32, psi, 4.0, tol=0.0, max_steps=20000, stall_windows=w, info=info)
        assert info["stop_reason"] in ("stall", "tau_floor") and n < 20000
        out[w] = (n, p)
    assert out[None][0] == out[1][0] and torch.equal(out[None][1], out[1][1])  # complex64 default: one window
    assert out[4][0] >= out[1][0]


def _lattice_1d_engine():
    g = Grid((128,), (4.0,), device=CPU)
    return BilayerGPE(g, lattice_1d(g, 8.0), g=0.2, n_components=1)


def test_complex128_default_is_four_stall_windows():
    """The complex128 default (stall_windows=None) is the four-window stall test: on a residual falling by
    0.3 % per window (synthetic, check windows of 5 steps) it runs as stall_windows=4 does, bit for bit, and
    goes on to tol (it can reach it in the budget, so it is not "slow"); the one-window test stops it at the
    first window."""
    from twisted_bilayer.engine.ground_state import precondition_polish

    seq = [1e-6 * 0.997**k for k in range(2000)]
    out = {}
    for w in (None, 1, 4):
        eng = _lattice_1d_engine()
        _synthetic_residual(eng, seq)
        info = {}
        p, r, n = precondition_polish(
            eng, initial_state(eng, 4.0), 4.0, tol=1e-7, max_steps=40000, check_every=5, stall_windows=w, info=info
        )
        out[w] = (n, p, info["stop_reason"])
    assert out[None][2] == out[4][2] == "tol" and out[None][0] == out[4][0] and torch.equal(out[None][1], out[4][1])
    assert out[1][2] == "stall" and out[1][0] == 5


def test_too_slow_rule():
    """_too_slow: a descent slower than 0.5 % per window (averaged over the stall windows) that cannot
    reach tol in the windows left; never for a faster descent, a rising residual, or a short history."""
    from twisted_bilayer.engine.ground_state import _too_slow

    slow = [1e-6 * 0.997**k for k in range(6)]  # 0.3 % per window
    assert _too_slow(slow, 4, 100, 1e-9)  # 1e-6 * 0.997^100 ~ 7e-7 > tol
    assert not _too_slow(slow, 4, 10000, 1e-9)  # 1e-6 * 0.997^10000 < tol: it can still get there
    fast = [1e-6 * 0.99**k for k in range(6)]  # 1 % per window: always goes on
    assert not _too_slow(fast, 4, 1, 1e-9)
    assert not _too_slow([1e-6, 1e-6, 1.01e-6, 1.02e-6, 1.03e-6, 1.04e-6], 4, 1, 1e-9)  # rising: the stall test's
    assert not _too_slow(slow[:4], 4, 1, 1e-9)  # not enough windows yet


def _synthetic_residual(eng, seq):
    """Replace eng.residual by the sequence seq (one value per call, as a (1,) tensor)."""
    it = iter(seq)
    eng.residual = lambda psi: torch.tensor([next(it)], dtype=torch.float64)


def test_slow_descent_stops_when_tol_is_out_of_reach():
    """A residual falling by 0.3 % per window: with tol out of reach in the step budget the complex128
    polish stops after the stall windows ("slow"); with tol within reach it goes on and ends on tol."""
    from twisted_bilayer.engine.ground_state import precondition_polish

    seq = [1e-6 * 0.997**k for k in range(400)]
    eng = _lattice_1d_engine()
    _synthetic_residual(eng, seq)
    info = {}
    _, _, n = precondition_polish(eng, initial_state(eng, 4.0), 4.0, tol=5e-7, max_steps=4000, info=info)
    assert info["stop_reason"] == "slow" and n == 4 * 50  # as soon as there are four windows to measure the rate
    eng = _lattice_1d_engine()
    _synthetic_residual(eng, seq)
    info = {}
    _, r, n = precondition_polish(eng, initial_state(eng, 4.0), 4.0, tol=8e-7, max_steps=4000, info=info)
    assert info["stop_reason"] == "tol" and float(r.max()) < 8e-7 and 3000 < n < 4000


def _bumped_residual(eng, bumps):
    """eng.residual with call k multiplied by bumps[k] (a dict; call 1 is the start)."""
    orig, calls = eng.residual, [0]

    def residual(psi):
        calls[0] += 1
        return orig(psi) * bumps.get(calls[0], 1.0)

    eng.residual = residual


def test_one_residual_bump_with_falling_energy_is_not_an_overshoot():
    """complex128: one window whose residual rises while E falls does not restart the polish (the run is
    the same as without the bump, bit for bit); two rises in a row do (tau is halved: a different run).
    complex64 keeps the residual-only test: one rise restarts it."""
    from twisted_bilayer.engine.ground_state import precondition_polish

    def run(dtype, bumps):
        eng = _lattice_1d_engine().to_dtype(dtype)
        _bumped_residual(eng, bumps)
        info = {}
        p, r, n = precondition_polish(eng, initial_state(eng, 4.0), 4.0, tol=1e-12, max_steps=4000, info=info)
        return p, n, info["stop_reason"]

    p0, n0, s0 = run(torch.complex128, {})
    p1, n1, _ = run(torch.complex128, {4: 100.0})  # the descent falls ~10x per window: a 100x bump is a rise
    assert s0 == "tol" and n1 == n0 and torch.equal(p1, p0)
    p2, n2, _ = run(torch.complex128, {4: 100.0, 5: 1e4})  # the residual rises twice running
    assert not (n2 == n0 and torch.equal(p2, p0))
    q0, m0, _ = run(torch.complex64, {})
    q1, m1, _ = run(torch.complex64, {4: 100.0})
    assert not (m1 == m0 and torch.equal(q1, q0))


def test_bump_late_in_a_descent_is_not_a_stall():
    """complex128: a 10 % bump of the residual in the 13th window of a 2 %-per-window descent (synthetic) is
    let through and left out of the stall test: the polish goes on to tol (it used to restart and halve tau)."""
    from twisted_bilayer.engine.ground_state import precondition_polish

    seq = [1e-6 * 0.98**k * (1.1 if k == 12 else 1.0) for k in range(400)]
    eng = _lattice_1d_engine()
    _synthetic_residual(eng, seq)
    info = {}
    _, r, n = precondition_polish(eng, initial_state(eng, 4.0), 4.0, tol=1e-7, max_steps=4000, check_every=5, info=info)
    assert info["stop_reason"] == "tol" and float(r.max()) < 1e-7


def test_energy_rose_rounding_level():
    """The rounding level is 1e3 eps times the sum of the magnitudes of the terms, per batch member."""
    from twisted_bilayer.engine.ground_state import _energy_rose

    eps = torch.finfo(torch.float64).eps
    E = torch.tensor([100.0, 200.0], dtype=torch.float64)
    S = torch.tensor([300.0, 500.0], dtype=torch.float64)  # e.g. kinetic + potential + interaction, unsigned
    assert not _energy_rose((E + 10 * eps * 500, S), (E, S), eps)  # rounding noise
    assert _energy_rose((E + torch.tensor([0.0, 1e-9], dtype=torch.float64), S), (E, S), eps)  # one member rises
    assert not _energy_rose((E - 1e-6, S), (E, S), eps)
    zero = torch.zeros(2, dtype=torch.float64)  # a total that cancels to 0 keeps the terms' scale
    assert not _energy_rose((zero + 1e-14, S), (zero, S), eps)
