"""engine.bands: the single-particle references.

Quoted numbers are tabulated values of an independent semi-analytic band
calculation: J1, J2/J1 and band width / 4 J1; the Wannier overlap s(d) at
d = 0.5 a (and s(a) = 0); I4, U at g2d = 0.0158 and U/J1. Some are also
tabulated to one more digit.
Each is compared at the precision it is quoted to (half a unit of the last
quoted digit) unless stated. Everything else is analytic or a cross-check
between two independent routes.
"""

import math

import numpy as np
import pytest

from twisted_bilayer.engine import bands
from twisted_bilayer.engine.units import KAPPA, lattice_units


def _half_unit(quoted: str) -> float:
    """Half a unit in the last digit of a quoted decimal string."""
    dec = len(quoted.split(".")[1]) if "." in quoted else 0
    return 0.5 * 10.0 ** (-dec)


def _close_to_quoted(value: float, quoted: str, slack: float = 1.0) -> bool:
    return abs(value - float(quoted)) <= slack * _half_unit(quoted) + 1e-15


# -- 1D bands ---------------------------------------------------------------------------


def test_bz_grid():
    q = bands.bz_grid(16)
    assert q.shape == (16,)
    assert q[-1] == pytest.approx(math.pi, abs=1e-15) and q[0] > -math.pi
    assert np.any(np.abs(q) < 1e-15)
    assert np.allclose(np.diff(q), 2 * math.pi / 16)


def test_free_limit_and_first_order_gap():
    """V0 = 0: the folded free parabola, E_0 = kappa q^2, E_1 = kappa (|q| - 2 pi)^2.
    Small V0: E = V0/2 + kappa q^2 shift, gap V0/2 at the zone edge to first
    order (nearly-free-particle perturbation theory)."""
    q, E = bands.bands_1d(0.0, n_q=32, n_bands=2)
    assert np.allclose(E[:, 0], KAPPA * q**2, atol=1e-13)
    assert np.allclose(E[:, 1], KAPPA * (np.abs(q) - 2 * math.pi) ** 2, atol=1e-13)
    V0 = 0.05
    q, E = bands.bands_1d(V0, n_q=32, n_bands=2)
    edge = np.argmax(q)  # q = pi
    assert E[edge, 1] - E[edge, 0] == pytest.approx(V0 / 2, rel=1e-3)
    assert np.all(np.diff(E, axis=1) > 0)


@pytest.mark.parametrize(
    "V0,J1,J2J1",
    [
        (4, "0.0855", "-0.072"),
        (8, "0.0308", "-0.021"),
        (12, "0.01225", "-0.007"),
        (16, "0.00533", None),
        (24, "0.00123", None),
    ],
)
def test_hopping_vs_table(V0, J1, J2J1):
    J = bands.hopping_1d(V0, d_max=3)
    assert _close_to_quoted(J[0], J1), J
    if J2J1 is not None:
        assert _close_to_quoted(J[1] / J[0], J2J1), J
    # the same J1 to one more digit: 0.08549 / 0.03080 / 0.01225
    quoted = {4: "0.08549", 8: "0.03080", 12: "0.01225"}
    if V0 in quoted:
        assert _close_to_quoted(J[0], quoted[V0])


@pytest.mark.parametrize("V0,ratio", [(4, "1.009"), (8, "1.001"), (12, "1.000"), (24, "1.000")])
def test_band_width_over_4J1(V0, ratio):
    q, E = bands.bands_1d(V0, n_q=64, n_bands=1)
    width = E[:, 0].max() - E[:, 0].min()
    assert _close_to_quoted(width / (4 * bands.hopping_1d(V0)[0]), ratio)


def test_hopping_reconstructs_band():
    """E_0(q) = e_0 - 2 sum_d J(d) cos(q d) with d up to 12 reproduces the
    band to 1e-12 at V0 = 8 (the Fourier series converges exponentially)."""
    V0 = 8.0
    q, E = bands.bands_1d(V0, n_q=64, n_bands=1)
    J = bands.hopping_1d(V0, d_max=12)
    recon = E[:, 0].mean() - 2 * sum(J[d - 1] * np.cos(q * d) for d in range(1, 13))
    assert np.abs(recon - E[:, 0]).max() < 1e-12


# -- Wannier function, overlaps, U ------------------------------------------------------


@pytest.mark.parametrize("V0", [4, 12])
def test_wannier_function_properties(V0):
    n_x = 64
    x, w = bands.wannier_1d(V0, n_x=n_x)
    dx = 1.0 / n_x
    assert np.sum(w * w) * dx == pytest.approx(1.0, abs=1e-12)
    c = np.argmin(np.abs(x))
    assert x[c] == 0.0 and np.argmax(np.abs(w)) == c and w[c] > 0
    assert np.allclose(w[c + 1 : c + 200], w[c - 1 : c - 200 : -1], atol=1e-12)  # even
    # orthogonal to its neighbours (Wannier functions of one band)
    for d in (1, 2, 3):
        assert abs(np.sum(w * np.roll(w, d * n_x)) * dx) < 1e-12
    # localised: |w| beyond 3 sites is < 1 % of the peak
    assert np.abs(w[np.abs(x) > 3]).max() < 1e-2 * w[c]
    # window: the same function on 3 sites around 0
    xs, ws = bands.wannier_1d(V0, n_x=n_x, n_sites=3)
    assert xs.size == 3 * n_x and xs[0] == pytest.approx(-1.5)
    assert np.allclose(ws, w[(x >= -1.5) & (x < 1.5)])


@pytest.mark.parametrize("V0,s_half", [(4, "0.447"), (8, "0.286"), (12, "0.196"), (16, "0.142"), (24, "0.083")])
def test_overlap_vs_table(V0, s_half):
    """s(a/2) from the tabulated overlaps to 3 decimals (V0 = 24: 0.0825 vs
    0.083, so 2-3 digits: 1e-3), and s(0) = 1, s(a) = s(2a) = 0 exactly."""
    assert abs(bands.overlap_1d(V0, 0.5) - float(s_half)) < 1e-3
    assert bands.overlap_1d(V0, 0.0) == pytest.approx(1.0, abs=1e-13)
    assert abs(bands.overlap_1d(V0, 1.0)) < 1e-13
    assert abs(bands.overlap_1d(V0, 2.0)) < 1e-13


@pytest.mark.parametrize("V0", [4, 8, 24])
def test_overlap_formula_equals_real_space_integral(V0):
    """The plane-wave formula for s(d) vs direct quadrature of
    int w(x) w(x - d) dx on the Wannier function (d on the grid)."""
    n_x = 80
    x, w = bands.wannier_1d(V0, n_x=n_x)
    for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        direct = np.sum(w * np.roll(w, int(round(d * n_x)))) / n_x
        assert bands.overlap_1d(V0, d) == pytest.approx(direct, abs=1e-12)
    s = bands.overlap_1d(V0, np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert s.shape == (2, 2) and np.all(np.diff(s.ravel()) < 0)


@pytest.mark.parametrize(
    "V0,I4,U",
    [
        (4, "1.493", "0.035"),
        (8, "1.895", "0.057"),
        (12, "2.153", "0.073"),
        (16, "2.346", "0.087"),
        (24, "2.634", "0.109"),
    ],
)
def test_I4_and_U_vs_table(V0, I4, U):
    """I4 to 3 decimals (2.3455 at 16 sits on the rounding edge: 1e-3);
    U at g2d = 0.0158 (the tables' rounded value) to 1e-3, and at the
    unrounded g2d of units.lattice_units() (0.01576) to half a unit."""
    assert abs(bands.I4_1d(V0) - float(I4)) < 1e-3
    assert abs(bands.onsite_U(V0, 0.0158) - float(U)) < 1e-3
    g2d = lattice_units().g2d
    assert g2d == pytest.approx(0.0158, abs=5e-5)
    assert _close_to_quoted(bands.onsite_U(V0, g2d), U)


@pytest.mark.parametrize("V0,ratio", [(4, "0.41"), (8, "1.8"), (12, "6.0"), (16, "16.3"), (24, "89")])
def test_U_over_J1(V0, ratio):
    r = bands.onsite_U(V0, lattice_units().g2d) / bands.hopping_1d(V0)[0]
    assert _close_to_quoted(r, ratio)


def test_I4_harmonic_limit():
    """Deep lattice: w -> ground state of the harmonic well,
    int w^4 = 1 / (sqrt(2 pi) sigma), sigma = (1/pi) V0^(-1/4); the exact
    value lies below (tails), the gap closing with depth."""
    rel = []
    for V0 in (24, 100, 400):
        sigma = V0**-0.25 / math.pi
        harm = 1.0 / (math.sqrt(2 * math.pi) * sigma)
        rel.append(1 - bands.I4_1d(V0) / harm)
    assert all(r > 0 for r in rel) and rel[0] > rel[1] > rel[2] and rel[2] < 0.03


# -- moire supercell ---------------------------------------------------------------------


def _dense_lowest(V0, Om, q, delta=0.0, k=8):
    H, _, _ = bands.moire_hamiltonian(V0, Om, 4, 1, 8.0, delta, q)
    dim = H.shape[0]
    Hd = H.matmat(np.eye(dim, dtype=H.dtype))
    assert np.abs(Hd - Hd.conj().T).max() < 1e-12  # Hermitian
    return np.linalg.eigvalsh(0.5 * (Hd + Hd.conj().T))[:k]


@pytest.mark.parametrize("q,delta", [((0.0, 0.0), 0.1), ((math.pi / math.sqrt(17), math.pi / math.sqrt(17)), 0.0)])
def test_moire_bands_vs_dense(q, delta):
    """m = 4 (N = 36, dimension 2592): the matrix-free solvers against dense
    diagonalisation of the same operator. The M point has degenerate
    pairs that a single plain Lanczos run can miss; both methods must return
    the full multiplicity."""
    V0, Om = 4.0, 0.24
    ref = _dense_lowest(V0, Om, q, delta)
    for method in ("lobpcg", "eigsh"):
        mb = bands.moire_bands_q0(V0, Om, m=4, n=1, n_bands=8, q=q, delta=delta, method=method)
        assert np.abs(mb.energies - ref).max() < 1e-9, (method, mb.energies, ref)
        assert mb.vectors.shape == (8, 2, 36, 36)
        assert np.allclose(mb.weights.sum(axis=1), 1.0, atol=1e-12)
        assert np.all((mb.participation > 0) & (mb.participation <= 1))
        assert mb.residuals.max() < 1e-6


def test_moire_free_particles_and_rabi_shift():
    """V0 = 0: H = kappa (k + q)^2 + Omega_R sigma_x, so the lowest level at
    Gamma, X, M is -Omega_R + kappa (0, 1, 2) (pi/L)^2 and the width is
    2 kappa (pi/L)^2: checks the twist trick and the sign of the coupling."""
    Om = 0.3
    w = bands.moire_band_width(0.0, Om, m=4, n=1)
    e = KAPPA * (math.pi / w["L"]) ** 2
    assert w["Gamma"][0] == pytest.approx(-Om, abs=1e-10)
    assert w["X"][0] == pytest.approx(e - Om, abs=1e-10)
    assert w["M"][0] == pytest.approx(2 * e - Om, abs=1e-10)
    assert w["width"] == pytest.approx(2 * e, abs=1e-10)


def test_moire_band_width_separable_at_omega0():
    """Omega_R = 0, V0 = 4: each layer is a rotated separable lattice, so the
    lowest level at supercell quasi-momentum q is
    min_G [E_1D(k_u) + E_1D(k_v)] over k = q + G, G = (2 pi/L)(i, j), with
    (k_u, k_v) the components in the layer's rotated frame."""
    V0, m, n = 4.0, 4, 1
    w = bands.moire_band_width(V0, 0.0, m=m, n=n)
    L = w["L"]
    a = math.atan2(n, m)
    c, s = math.cos(a), math.sin(a)

    def E1(k):
        k = np.mod(k + math.pi, 2 * math.pi) - math.pi
        E, _, _ = bands._bloch_1d(V0, k.ravel(), 31)
        return E[:, 0].reshape(k.shape)

    I, J = np.meshgrid(np.arange(-6, 7), np.arange(-6, 7), indexing="ij")
    for name, q in (("Gamma", (0.0, 0.0)), ("X", (math.pi / L, 0.0)), ("M", (math.pi / L, math.pi / L))):
        kx, ky = q[0] + 2 * math.pi * I / L, q[1] + 2 * math.pi * J / L
        expect = (E1(kx * c - ky * s) + E1(ky * c + kx * s)).min()
        assert w[name][0] == pytest.approx(expect, abs=1e-8), name
        assert w[name][1] == pytest.approx(expect, abs=1e-8)  # the other layer, same energy


# -- higher-band Wannier functions and their overlaps --------------------------------------


@pytest.mark.parametrize("V0", [8.0, 12.0, 16.0])
def test_band_wannier_gauge_orthogonality_parity_and_completeness(V0):
    """`wannier_coefficients_band` / `overlap_bands_1d`: band 0 reproduces `overlap_1d` (the same
    real symmetric Wannier function); s_ab(0) = delta_ab and s_ab(integer) = 0 (orthonormal Wannier
    functions); s_ab(-d) = (-1)^(a+b) s_ab(d) (parity (-1)^b); the band centres are ordered; and the
    gauge-free band weights sum to 1 over all n_pw bands (completeness in the plane-wave space),
    while 6 bands hold > 99.9 % of a half-site-displaced lowest-band function."""
    d = np.linspace(-2.0, 2.0, 17)
    assert np.allclose(bands.overlap_bands_1d(V0, d, 0, 0), bands.overlap_1d(V0, d), atol=1e-14)
    for a in range(4):
        for b in range(4):
            assert bands.overlap_bands_1d(V0, 0.0, a, b) == pytest.approx(float(a == b), abs=1e-12)
            assert bands.overlap_bands_1d(V0, 1.0, a, b) == pytest.approx(0.0, abs=1e-12)
            assert bands.overlap_bands_1d(V0, 2.0, a, b) == pytest.approx(0.0, abs=1e-12)
            sp, sm = bands.overlap_bands_1d(V0, 0.37, a, b), bands.overlap_bands_1d(V0, -0.37, a, b)
            assert sm == pytest.approx((-1) ** (a + b) * sp, abs=1e-12)
    e = bands.band_centres_1d(V0, 6)
    assert np.all(np.diff(e) > 0) and e[1] - e[0] == pytest.approx(2.0 * math.sqrt(V0), rel=0.25)
    for dd in (0.0, 0.25, 0.5, 0.9):
        w = np.array([bands.band_weight_1d(V0, dd, b) for b in range(31)])
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(w >= -1e-14)
        n = np.arange(-32, 32)
        for b in range(5):  # the gauge-fixed overlaps sum to the same weight
            assert (bands.overlap_bands_1d(V0, dd + n, 0, b) ** 2).sum() == pytest.approx(w[b], abs=1e-12)
    w = np.array([bands.band_weight_1d(V0, 0.5, b) for b in range(8)])
    assert w[0] == pytest.approx((bands.overlap_1d(V0, 0.5 + np.arange(-32, 32)) ** 2).sum(), abs=1e-12)
    assert w[:6].sum() > 0.999 and w[:4].sum() > 0.95
    assert bands._ho_sigma(V0) == pytest.approx((math.pi**2 * math.sqrt(V0)) ** -0.5, rel=1e-12)
