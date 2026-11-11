"""The engine's inputs and their edges: the start phase per batch member, the read-only
geometry, the construction checks, and fast CPU tests at the input boundaries (Omega_R = 0 inside a batch,
zero-density regions, a one-member batch, complex64 at the stop rules' limits)."""

import math

import pytest
import torch

from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, precondition_polish, relax
from twisted_bilayer.engine.lattice import harmonic_trap, lattice_1d, toy_bilayer_1d
from twisted_bilayer.engine.npse import NPSE2D
from twisted_bilayer.engine.physics import BilayerGPE

CPU = torch.device("cpu")


def _toy(Omega_R, g=0.2, dtype=torch.complex128, **kw):
    grid = Grid((96,), (6.0,), device=CPU)
    return BilayerGPE(grid, toy_bilayer_1d(grid, 6.0, 6), g=g, Omega_R=Omega_R, dtype=dtype, **kw)


def _shared(Omega_R, g=0.2):
    """Both components in one lattice: the layer swap is an exact symmetry, so a symmetric start stays on the
    upper dressed branch (the case of the bilayer's mirror symmetry, in 1D)."""
    grid = Grid((96,), (6.0,), device=CPU)
    return BilayerGPE(grid, lattice_1d(grid, 6.0), g=g, Omega_R=Omega_R)


def _relative_phase(psi):
    """Phase of component 2 relative to component 1 per member, at the density peak of component 1."""
    i = psi[:, 0].abs().reshape(psi.shape[0], -1).argmax(dim=1)
    p1 = psi[:, 0].reshape(psi.shape[0], -1).gather(1, i[:, None])[:, 0]
    p2 = psi[:, 1].reshape(psi.shape[0], -1).gather(1, i[:, None])[:, 0]
    return torch.angle(p2 / p1)


# -- the start phase per batch member ----------------------------------------------------------------


def test_mixed_sign_batch_starts_each_member_on_its_lower_branch():
    """Omega_R = (-0.2, 0.2): member 0 starts symmetric (phase 0), member 1 antisymmetric (pi); both relax to
    their lower dressed branch, with E/N equal by the gauge copy psi_2 -> -psi_2, and equal to a one-member run
    at +0.2. A phase of pi for both (the earlier default) starts member 0 on its upper branch: Rabi energy
    +|Omega_R| per atom instead of -|Omega_R| (a state that imaginary time leaves only through rounding)."""
    eng = _shared(torch.tensor([-0.2, 0.2], dtype=torch.float64))
    psi0 = initial_state(eng, 6.0, batch=2)
    ph = _relative_phase(psi0)
    assert abs(float(ph[0])) < 1e-12 and abs(abs(float(ph[1])) - math.pi) < 1e-12
    r = relax(eng, psi0, 6.0, tol=1e-10)
    assert bool(r.converged.all())
    e = r.energy / 6.0
    single = relax(_shared(0.2), initial_state(_shared(0.2), 6.0), 6.0, tol=1e-10)
    assert abs(float(e[0] - e[1])) < 1e-10 and abs(float(e[1] - single.energy[0] / 6.0)) < 1e-10
    rabi = eng.energy_terms(psi0)["E_rabi"] / 6.0
    old = eng.energy_terms(initial_state(eng, 6.0, batch=2, relative_phase=math.pi))["E_rabi"] / 6.0
    assert torch.allclose(rabi, torch.tensor([-0.2, -0.2], dtype=torch.float64), atol=1e-12)
    assert abs(float(old[0]) - 0.2) < 1e-12 and abs(float(old[1]) + 0.2) < 1e-12


def test_same_sign_batch_start_is_unchanged():
    """Without mixed signs the start is unchanged, bit for bit: pi for every member when any Omega_R > 0
    (a member at Omega_R = 0 included), 0 when none is."""
    eng = _toy(torch.tensor([0.0, 0.24, 0.5], dtype=torch.float64))
    assert torch.equal(initial_state(eng, 6.0, batch=3), initial_state(eng, 6.0, batch=3, relative_phase=math.pi))
    neg = _toy(torch.tensor([0.0, -0.3], dtype=torch.float64))
    assert torch.equal(initial_state(neg, 6.0, batch=2), initial_state(neg, 6.0, batch=2, relative_phase=0.0))
    assert torch.equal(initial_state(_toy(0.2), 6.0), initial_state(_toy(0.2), 6.0, relative_phase=math.pi))


def test_mixed_sign_batch_needs_one_start_per_member():
    eng = _toy(torch.tensor([-0.2, 0.2], dtype=torch.float64))
    with pytest.raises(ValueError, match="mixes signs"):
        initial_state(eng, 6.0)  # batch=1 cannot carry two phases
    psi = initial_state(eng, 6.0, relative_phase=0.0)  # an explicit phase is the caller's choice
    assert psi.shape[0] == 1


# -- kappa and grid are read-only ----------------------------------------------------------------------


def test_kappa_and_grid_are_read_only():
    """The cached kinetic operators depend on kappa and the grid: assigning either raises, so the cache cannot
    go stale; to_dtype still makes a copy on the converted grid."""
    eng = _toy(0.2)
    with pytest.raises(AttributeError):
        eng.kappa = 2.0
    with pytest.raises(AttributeError):
        eng.grid = Grid((48,), (6.0,), device=CPU)
    e32 = eng.to_dtype(torch.complex64)
    assert e32.grid.real_dtype == torch.float32 and eng.grid.real_dtype == torch.float64 and e32.kappa == eng.kappa


# -- construction checks -------------------------------------------------------------------------------


def test_construction_rejects_bad_inputs():
    grid = Grid((32,), (4.0,), device=CPU)
    V = toy_bilayer_1d(grid, 4.0, 4)
    bad_V = V.clone()
    bad_V[0, 0, 3] = float("nan")
    cases = [
        dict(V=bad_V),
        dict(V=V, g=float("inf")),
        dict(V=V, Omega_R=torch.tensor([0.1, float("nan")], dtype=torch.float64)),
        dict(V=V, delta=float("-inf")),
        dict(V=V, kappa=0.0),
        dict(V=V, kappa=-1.0),
        dict(V=V, kappa=float("nan")),
        dict(V=V, g=-0.1),
        dict(V=V, g=torch.tensor([0.1, -0.1], dtype=torch.float64)),
        dict(V=V, Omega_R=(0.1, 0.2, 0.3), g=(0.1, 0.2)),  # batched lengths differ
        dict(V=V.expand(2, 2, 32), Omega_R=(0.1, 0.2, 0.3)),  # potential batch against parameter batch
        dict(V=V, twist=[float("nan")]),
        dict(V=V, dtype=torch.float64),
    ]
    for kw in cases:
        with pytest.raises(ValueError):
            BilayerGPE(grid, **kw)


def test_construction_accepts_the_shipped_inputs():
    """Attraction runs when asked; length-1 and matching batches, a component-less trap, a batched twist."""
    grid = Grid((32,), (4.0,), device=CPU)
    V = toy_bilayer_1d(grid, 4.0, 4)
    assert BilayerGPE(grid, V, g=-0.1, allow_attractive=True).g == -0.1
    BilayerGPE(grid, V, g=(0.1, 0.2), Omega_R=(0.0, 0.3), twist=torch.zeros(2, 1, dtype=torch.float64))
    BilayerGPE(grid, V, g=(0.1,), Omega_R=(0.0, 0.3))
    BilayerGPE(grid, harmonic_trap(grid, (1.0,)), g=0.0, n_components=1)
    BilayerGPE(grid, None, interaction=NPSE2D(0.5, 2.0))
    eng = BilayerGPE(grid, V, g=0.1)
    with pytest.raises(ValueError):
        eng.set_twist([float("inf")])


# -- the input boundaries ------------------------------------------------------------------------------


def test_zero_coupling_inside_a_coupled_batch():
    """Omega_R = 0 as one member of a batch with Omega_R > 0: the member relaxes to the uncoupled ground state
    (E/N of a one-member run at Omega_R = 0, whose start phase is 0 while the batch gives it pi: at Omega_R = 0
    the relative phase does not enter E)."""
    eng = _toy(torch.tensor([0.0, 0.24], dtype=torch.float64))
    r = relax(eng, initial_state(eng, 6.0, batch=2), 6.0, tol=1e-10)
    alone = relax(_toy(0.0), initial_state(_toy(0.0), 6.0), 6.0, tol=1e-10)
    assert bool(r.converged.all()) and abs(float(r.energy[0] - alone.energy[0])) / 6.0 < 1e-10


@pytest.mark.parametrize("npse", [False, True])
def test_zero_density_region(npse):
    """A Thomas-Fermi start in a trap is exactly zero outside its radius: energy terms, residual and the polish
    stay finite there (fixed g and the NPSE functional, whose width solve starts at n = 0), and the cloud
    converges."""
    grid = Grid((64, 64), (12.0, 12.0), device=CPU)
    V = harmonic_trap(grid, (0.6, 0.6))
    kw = dict(g=0.0, interaction=NPSE2D(0.3, 5.0)) if npse else dict(g=0.3)
    eng = BilayerGPE(grid, V, Omega_R=0.2, **kw)
    psi0 = initial_state(eng, 20.0, kind="thomas_fermi")
    assert float(eng.density(psi0).min()) == 0.0
    terms = eng.energy_terms(psi0)
    assert all(bool(torch.isfinite(v).all()) for v in terms.values())
    assert bool(torch.isfinite(eng.residual(psi0)).all())
    r = relax(eng, psi0, 20.0, tol=1e-9, dt_schedule=[], polish_steps=8000)
    assert r.stop_reason == "tol" and bool(torch.isfinite(r.psi).all())


def test_one_member_batch_equals_the_scalar_run():
    """Omega_R as a one-member tensor gives the run of the scalar Omega_R (same steps, E to rounding)."""
    a = relax(_toy(0.24), initial_state(_toy(0.24), 6.0), 6.0, tol=1e-10)
    eng = _toy(torch.tensor([0.24], dtype=torch.float64))
    b = relax(eng, initial_state(eng, 6.0), 6.0, tol=1e-10)
    assert a.steps == b.steps and abs(float(a.energy[0] - b.energy[0])) < 1e-11 and a.stop_reason == b.stop_reason


def test_complex64_at_the_stop_rules_limits():
    """complex64 keeps the residual-only rules: a tol above its rounding floor is reached ("tol"); a tol below
    it ends on the floor ("stall" or "tau_floor") well inside the budget, under the trapped runs' COMPLEX64_FLOOR,
    which `trapped.checks` then counts as converged to the floor."""
    from twisted_bilayer.engine.trapped import COMPLEX64_FLOOR, checks

    e32 = _toy(0.24).to_dtype(torch.complex64)
    psi = initial_state(e32, 6.0)
    info = {}
    _, r, n = precondition_polish(e32, psi, 6.0, tol=1e-4, max_steps=20000, info=info)
    assert info["stop_reason"] == "tol" and float(r.max()) < 1e-4
    info = {}
    _, r, n = precondition_polish(e32, psi, 6.0, tol=1e-12, max_steps=20000, info=info)
    assert info["stop_reason"] in ("stall", "tau_floor") and n < 20000 and float(r.max()) < COMPLEX64_FLOOR
    c = checks(0.0, False, residual=float(r.max()), dtype="complex64", stop_reason=info["stop_reason"])
    assert c["at_precision_floor"] and c["ok"]
