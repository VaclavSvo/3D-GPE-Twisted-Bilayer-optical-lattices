"""Lattice geometry and potentials (engine.lattice), and the engine's band
structure by twisted boundaries checked against the independent plane-wave
and matrix-free references of engine.bands.

Sources of the pinned numbers: tan(theta/2) = n/m and L = sqrt(m^2 + n^2) a
(engine.lattice docstring), L_M = a / (2 sin(theta/2)); the 1D bands
and hoppings are computed, not quoted (bands.bands_1d / hopping_1d).
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer.engine import bands
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, relax
from twisted_bilayer.engine.lattice import bilayer_potential, lattice_1d, moire_geometry, square_lattice
from twisted_bilayer.engine.physics import BilayerGPE

CPU = torch.device("cpu")


def check_periodicity(grid: Grid, V: torch.Tensor) -> float:
    """The discrete seam of a sampled potential: the finite difference across the
    periodic boundary against the interior one, max over axes (~1 for a seamless box)."""
    worst = 0.0
    for ax in grid.sdims:
        d_int = (V.roll(-1, dims=ax) - V).abs()
        interior = d_int.narrow(ax, 0, grid.N[ax] - 1).max()
        seam = d_int.narrow(ax, grid.N[ax] - 1, 1).max()
        worst = max(worst, float(seam / interior) if interior > 0 else 0.0)
    return worst


def _supercell(m, n, ppa):
    geo = moire_geometry(m, n)
    N = geo.grid_size(ppa)
    return geo, Grid((N, N), (geo.L_box, geo.L_box), device=CPU)


def _V_numpy(x, y, V0, alpha):
    """Paper Supplement Eq. 5, written out independently of lattice.py:
    lattice 1 is U(R(+alpha) r), lattice 2 the same with alpha -> -alpha."""
    c, s = math.cos(alpha), math.sin(alpha)
    V1 = V0 * (np.sin(np.pi * (x * c - y * s)) ** 2 + np.sin(np.pi * (y * c + x * s)) ** 2)
    V2 = V0 * (np.sin(np.pi * (x * c + y * s)) ** 2 + np.sin(np.pi * (y * c - x * s)) ** 2)
    return V1, V2


# -- geometry -------------------------------------------------------------------------


def test_moire_geometry_22_1():
    geo = moire_geometry(22, 1)
    # 2 atan(1/22) = 5.20512 deg (twisted_bilayer.config: 5.2051)
    assert geo.theta_deg == pytest.approx(math.degrees(2 * math.atan(1 / 22)), rel=1e-14)
    assert round(geo.theta_deg, 4) == 5.2051
    assert geo.L_box == pytest.approx(math.sqrt(485), rel=1e-15)
    assert geo.L_moire == pytest.approx(geo.L_box / 2, rel=1e-14)
    assert geo.primitive  # (m + n) odd: one primitive supercell
    assert geo.grid_size(16) == 360 and geo.grid_size(8) == 180
    geo4 = moire_geometry(4, 1)
    assert geo4.L_box == pytest.approx(math.sqrt(17)) and geo4.grid_size(8) == 36


# -- potentials -------------------------------------------------------------------------


@pytest.mark.parametrize("m,n,ppa", [(22, 1, 8), (22, 1, 16), (4, 1, 8)])
def test_bilayer_potential_seam(m, n, ppa):
    """Discrete seam check: the jump across the periodic boundary is like
    the interior ones (ratio ~ 1, within 5 %)."""
    geo, g = _supercell(m, n, ppa)
    V = bilayer_potential(g, 4.0, geometry=geo)
    assert V.shape == (1, 2) + g.N
    assert abs(check_periodicity(g, V) - 1.0) < 0.05


@pytest.mark.parametrize("m,n", [(22, 1), (4, 1), (7, 2)])
def test_bilayer_potential_analytic_periodicity(m, n):
    """The formula itself is periodic on the box: V(r + L e_x) = V(r + L e_y)
    = V(r) at random (off-grid) points, to 1e-12. The numpy formula is
    first checked against bilayer_potential on the grid, so this is a
    statement about the code's potential."""
    geo, g = _supercell(m, n, 8)
    V0 = 4.0
    V = bilayer_potential(g, V0, geometry=geo)[0].numpy()
    X, Y = np.meshgrid(g.x64[0].numpy(), g.x64[1].numpy(), indexing="ij")
    V1, V2 = _V_numpy(X, Y, V0, geo.alpha)
    assert np.abs(V[0] - V1).max() < 1e-12 and np.abs(V[1] - V2).max() < 1e-12
    rng = np.random.default_rng(0)
    L = geo.L_box
    x, y = rng.uniform(-L / 2, L / 2, (2, 500))
    for dx, dy in ((L, 0.0), (0.0, L), (L, L), (-L, 2 * L)):
        a1, a2 = _V_numpy(x, y, V0, geo.alpha)
        b1, b2 = _V_numpy(x + dx, y + dy, V0, geo.alpha)
        assert np.abs(a1 - b1).max() < 1e-12 and np.abs(a2 - b2).max() < 1e-12
    # and NOT periodic on a slightly wrong box (the check has teeth)
    b1, _ = _V_numpy(x + 1.01 * L, y, V0, geo.alpha)
    assert np.abs(_V_numpy(x, y, V0, geo.alpha)[0] - b1).max() > 1e-2


def test_AA_point_at_origin_and_mirror_symmetry():
    geo, g = _supercell(22, 1, 8)
    V = bilayer_potential(g, 6.0, geometry=geo)[0]
    c = g.N[0] // 2
    assert float(V[0, c, c]) == 0.0 and float(V[1, c, c]) == 0.0  # both minima at r = 0
    assert float(V.min()) >= 0.0 and float(V.max()) <= 12.0 + 1e-12
    # V2(x, y) = V1(x, -y): y -> -y is index j -> (N - j) mod N
    V1_mirror = torch.roll(torch.flip(V[0], dims=[1]), 1, dims=1)
    assert float((V[1] - V1_mirror).abs().max()) < 1e-12
    # each layer is C4 symmetric about the AA point: V1(-y, x) = V1(x, y)
    X, Y = np.meshgrid(g.x64[0].numpy(), g.x64[1].numpy(), indexing="ij")
    R1, _ = _V_numpy(-Y, X, 6.0, geo.alpha)
    assert np.abs(V[0].numpy() - R1).max() < 1e-12
    # zero twist: the two lattices coincide and square_lattice is V0[sin^2 + sin^2]
    V00 = bilayer_potential(g, 6.0, theta=0.0)[0]
    assert float((V00[0] - V00[1]).abs().max()) == 0.0
    assert torch.allclose(
        square_lattice(g, 6.0)[0, 0],
        6.0 * (torch.sin(math.pi * g.x64[0])[:, None] ** 2 + torch.sin(math.pi * g.x64[1])[None, :] ** 2),
        atol=1e-12,
    )


# -- bands by twist ----------------------------------------------------------------------


def test_1d_band_by_twist():
    """One site, twist theta = q a (L = a): the relaxed energy per particle
    is the lowest Bloch band E_0(q); 16 twists in one batched relaxation.
    Reference: plane-wave diagonalisation (bands.bands_1d). J1 from the 16
    energies vs bands.hopping_1d (64 q points)."""
    V0 = 8.0
    g = Grid((32,), (1.0,), device=CPU)
    q, E_ref = bands.bands_1d(V0, n_q=16)
    eng = BilayerGPE(
        g, lattice_1d(g, V0), g=0.0, n_components=1, twist=torch.tensor(q, dtype=torch.float64).reshape(-1, 1)
    )
    res = relax(eng, initial_state(eng, 1.0, batch=16), 1.0, tol=1e-10, polish_steps=2000)
    assert bool(res.converged.all()), res.report()
    E = (res.energy / eng.norm(res.psi)).numpy()
    assert np.abs(E - E_ref[:, 0]).max() < 1e-6
    J1 = -np.mean(E * np.cos(q))
    assert J1 == pytest.approx(bands.hopping_1d(V0)[0], abs=1e-5)
    assert round(J1, 4) == 0.0308  # tabulated J1 at V0 = 8
    # the band is symmetric and its minimum is at q = 0
    assert np.argmin(E) == np.argmin(np.abs(q))


def test_2d_band_bottom_separable_omega0():
    """Omega_R = 0, g = 0 on the m = 4 supercell: each layer is a rotated,
    separable square lattice, so the ground-state energy per particle is
    twice the 1D band bottom E_0(q = 0) (both layers alike)."""
    V0 = 4.0
    geo, g = _supercell(4, 1, 8)
    eng = BilayerGPE(g, bilayer_potential(g, V0, geometry=geo), g=0.0, Omega_R=0.0, n_components=2)
    res = relax(eng, initial_state(eng, 1.0), 1.0, tol=1e-9, polish_steps=2000)
    assert bool(res.converged.all()), res.report()
    q, E1d = bands.bands_1d(V0, n_q=2)  # q = 0 and pi
    E0 = E1d[np.argmin(np.abs(q)), 0]
    assert float(res.energy[0] / eng.norm(res.psi)[0]) == pytest.approx(2 * E0, abs=1e-6)


@pytest.mark.parametrize("Omega_R", [0.24, 1.0])
def test_two_component_band_bottom_vs_matrix_free(Omega_R):
    """Omega_R > 0, g = 0, m = 4, 8 points per a: the relaxed energy equals
    the lowest eigenvalue of the same Hamiltonian on the same grid from
    bands.moire_bands_q0 (independent numpy FFT + LOBPCG). The start has
    psi_2 = -psi_1 (the ground state's sign structure for +Omega_R sigma_x)."""
    V0, ppa = 4.0, 8
    geo, g = _supercell(4, 1, ppa)
    eng = BilayerGPE(g, bilayer_potential(g, V0, geometry=geo), g=0.0, Omega_R=Omega_R, n_components=2)
    psi0 = initial_state(eng, 1.0)
    # (initial_state already starts component 2 at relative phase pi)
    res = relax(eng, psi0, 1.0, tol=1e-9, polish_steps=4000)
    assert bool(res.converged.all()), res.report()
    mb = bands.moire_bands_q0(V0, Omega_R, m=4, n=1, points_per_a=ppa, n_bands=2)
    assert mb.N == g.N[0]
    E = float(res.energy[0] / eng.norm(res.psi)[0])
    assert E == pytest.approx(mb.energies[0], abs=1e-6)
    # same state: equal component weights (delta = 0, mirror symmetry) and
    # overlap 1 with the eigenvector
    u = torch.from_numpy(mb.vectors[0]).reshape((1, 2) + g.N).to(torch.complex128)
    psi = res.psi / torch.sqrt((res.psi.abs() ** 2).sum())
    assert abs(complex((psi.conj() * u).sum())) == pytest.approx(1.0, abs=1e-8)
    assert np.allclose(mb.weights[0], [0.5, 0.5], atol=1e-8)
