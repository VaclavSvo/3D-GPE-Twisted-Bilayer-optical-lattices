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
