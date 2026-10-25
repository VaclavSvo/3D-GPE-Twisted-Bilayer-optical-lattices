"""engine.bdg: Bogoliubov-de Gennes modes about a real ground state.

Checks: the operators against a finite-difference linearisation of the
engine's own H psi (contact and NPSE, two components, Delta, Omega_R, Bloch
q); the uniform-gas spectra (single component, and the density/spin branches
of the SU(2) two-component gas with Rabi coupling, at q = 0 and off Gamma);
the sound-speed identity c^2 = f_s n dmu/dn / m on a 1D lattice (superfluid
hydrodynamics); Hugenholtz-Pines (the projected-out zero mode); the matrix-free
solver against dense diagonalisation on a two-component moire supercell; and
the m = 4 supercell at V0 = 8, Omega_R = 0.24, filling 1 (realness,
positivity, normalisation; numbers printed with -s).
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer.engine.bdg import BdG, bdg_bands, sound_speed
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, relax
from twisted_bilayer.engine.lattice import bilayer_potential, lattice_1d, moire_geometry, toy_bilayer_1d
from twisted_bilayer.engine.npse import NPSE2D
from twisted_bilayer.engine.observables import superfluid_fraction_current
from twisted_bilayer.engine.physics import BilayerGPE
from twisted_bilayer.engine.units import KAPPA, lattice_units

CPU = torch.device("cpu")
C128 = torch.complex128


def _bogoliubov(eps, gn):
    return np.sqrt(eps * (eps + 2.0 * gn))


# -- operators ------------------------------------------------------------------------


@pytest.mark.parametrize("vertical", ["contact", "npse"])
def test_operators_are_the_linearised_engine(vertical):
    """For any real psi0 and complex phi, the engine with twist q L gives
    d/de H_q[psi0 + e phi] = (H_q + mu_int(n)-terms) phi + g' psi0 psi0.(phi + phi^*),
    i.e. [H_q(psi0 + e phi) - H_q(psi0 - e phi)] / 2e - mu phi = L phi + M phi^*.
    Central difference with e = 1e-4: O(e^2) ~ 1e-8 relative. Exercises the
    Bloch-shifted kinetic term, V, Delta, Omega_R, the mean field and the
    exchange kernel g' = d mu_int/dn (NPSE: the analytic dmu_int)."""
    gen = torch.Generator().manual_seed(0)
    g = Grid((24, 20), (3.0, 2.5), device=CPU)
    V = torch.rand((1, 2, 24, 20), generator=gen, dtype=torch.float64) * 3.0
    u = lattice_units()
    if vertical == "npse":
        kw = dict(g=0.0, interaction=NPSE2D(u.g2d, u.omega_z))
    else:
        kw = dict(g=0.7)
    q = (0.37, -0.21)
    eng = BilayerGPE(g, V, Omega_R=0.31, delta=0.17, n_components=2, **kw)
    eng_q = BilayerGPE(g, V, Omega_R=0.31, delta=0.17, n_components=2, twist=(q[0] * g.L[0], q[1] * g.L[1]), **kw)
    psi0 = (1.0 + torch.rand((1, 2, 24, 20), generator=gen, dtype=torch.float64)) * (3.0 if vertical == "npse" else 1.0)
    mu = 1.234
    bd = BdG(eng, psi0.to(C128), q=q, mu=mu)
    p0 = bd.psi0.to(C128)
    phi = torch.complex(
        torch.randn((1, 2, 24, 20), generator=gen, dtype=torch.float64),
        torch.randn((1, 2, 24, 20), generator=gen, dtype=torch.float64),
    )
    e = 1e-4
    fd = (eng_q.apply_H(p0 + e * phi) - eng_q.apply_H(p0 - e * phi)) / (2 * e) - mu * phi
    lin = bd.apply_L(phi) + bd.apply_M(phi.conj())
    assert float((fd - lin).abs().max() / lin.abs().max()) < 1e-7
    # A = L + M, B = L - M, and B = H_GP - mu: Hermitian
    x = torch.randn((2, 2, 24, 20), generator=gen, dtype=torch.float64).to(C128)
    for op in (bd.apply_A, bd.apply_B):
        G = x.reshape(2, -1).conj() @ op(x).reshape(2, -1).T
        assert torch.allclose(G, G.conj().T, atol=1e-10 * float(G.abs().max()))
    if vertical == "npse":
        f = eng.interaction
        n = torch.tensor([0.1, 1.0, 10.0, 100.0], dtype=torch.float64)
        h = 1e-5 * n
        assert torch.allclose(f.dmu_int(n), (f.mu_int(n + h) - f.mu_int(n - h)) / (2 * h), rtol=1e-7)


# -- uniform gases ----------------------------------------------------------------------


def test_uniform_single_component_1d_and_2d():
    """n_components = 1, V = 0, g n = 0.5: w(k) = sqrt(eps (eps + 2 g n)),
    eps = kappa k^2, k on the box's reciprocal lattice without k = 0 (the zero
    mode, projected out), to 1e-6. The dense solver sees the zero mode at
    w ~ 0. 2D box 6 x 4.5 a so the cut falls between degenerate shells."""
    gn = 0.5
    for N, L, n_modes in (((64,), (10.0,), 8), ((24, 18), (6.0, 4.5), 10)):
        g = Grid(N, L, device=CPU)
        eng = BilayerGPE(g, None, g=gn, n_components=1)
        psi0 = initial_state(eng, g.volume)  # n = 1
        ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n, d=l / n) for n, l in zip(N, L)], indexing="ij")
        eps = KAPPA * sum(k**2 for k in ks).ravel()
        exact = np.sort(_bogoliubov(eps[eps > 0], gn))[:n_modes]
        bd = BdG(eng, psi0)
        r = bd.solve(n_modes, tol=1e-9)
        assert np.allclose(r.omega.numpy(), exact, rtol=1e-6, atol=0)
        assert torch.allclose(r.norm_check, torch.ones(n_modes, dtype=torch.float64), atol=1e-10)
        if g.d == 1:
            rd = bd.solve(n_modes, method="dense")
            assert np.allclose(rd.omega.numpy(), exact, rtol=1e-10)
            assert abs(rd.zero_mode_omega) < 1e-6
            # Bogoliubov amplitudes: depletion |v_k|^2 = (eps + gn - w)/(2 w) per mode
            v2 = (eps[eps > 0] + gn - _bogoliubov(eps[eps > 0], gn)) / (2 * _bogoliubov(eps[eps > 0], gn))
            assert np.allclose(rd.depletion.numpy(), np.sort(v2)[::-1][:n_modes], rtol=1e-8)


def test_uniform_two_component_density_and_spin_branches():
    """SU(2) g, equal densities, Omega_R = 0.3 > 0 (lower dressed state
    antisymmetric), Delta = 0: density branch w_d = sqrt(eps (eps + 2 g n)),
    spin branch w_s = eps + 2 Omega_R (g = g12; the gap is 2 Omega_R because
    Omega_R is the off-diagonal element), each to 1e-6, with spin weights 0 and 1.
    At Bloch q the momenta are k_j + q (the twist), and q and q + 2 pi/L give
    the same spectrum (mini-zone reduction)."""
    gn, Om = 0.4, 0.3
    g = Grid((48,), (8.0,), device=CPU)
    eng = BilayerGPE(g, None, g=gn, Omega_R=Om, n_components=2)
    psi0 = initial_state(eng, g.volume)  # n = 1, relative phase pi
    assert float(eng.residual(psi0)[0]) < 1e-12
    kj = 2 * np.pi * np.fft.fftfreq(48, d=8.0 / 48)
    for q, n_modes in ((0.0, 7), (0.3, 8), (0.3 + 2 * np.pi / 8.0, 8)):
        eps = KAPPA * (kj + q) ** 2
        dens = [(w, 0.0) for w, e in zip(_bogoliubov(eps, gn), eps) if e > 1e-14]
        spin = [(w, 1.0) for w in eps + 2 * Om]
        exact = sorted(dens + spin)[:n_modes]
        r = BdG(eng, psi0, q=(q,)).solve(n_modes, tol=1e-9)
        assert np.allclose(r.omega.numpy(), [w for w, _ in exact], rtol=1e-6, atol=0)
        assert np.allclose(r.spin_weight.numpy(), [s for _, s in exact], atol=1e-6)
        assert np.allclose(r.local_spin_weight.numpy(), [s for _, s in exact], atol=1e-6)
        assert torch.allclose(r.norm_check, torch.ones(n_modes, dtype=torch.float64), atol=1e-10)
        # the spin modes are pure particles (v = 0) in the SU(2) case
        sp = r.spin_weight > 0.5
        assert float(r.depletion[sp].abs().max()) < 1e-10


def test_uniform_two_component_without_coupling():
    """Omega_R = 0: the relative phase is a second Goldstone mode at q = 0
    (h = (psi1, -psi2)), projected out with the first; the SU(2) spin branch
    is then free particles, w_s = eps, and the density branch is unchanged.
    The in-phase ground state (s = +1) gives the same spin weights 0 and 1."""
    gn = 0.4
    g = Grid((48,), (8.0,), device=CPU)
    eng = BilayerGPE(g, None, g=gn, Omega_R=0.0, n_components=2)
    psi0 = initial_state(eng, g.volume)  # relative phase 0
    kj = 2 * np.pi * np.fft.fftfreq(48, d=8.0 / 48)
    for q, n_modes in ((0.0, 6), (0.3, 8)):
        eps = KAPPA * (kj + q) ** 2
        eps = eps[eps > 1e-14]
        exact = sorted([(w, 0.0) for w in _bogoliubov(eps, gn)] + [(e, 1.0) for e in eps])[:n_modes]
        bd = BdG(eng, psi0, q=(q,))
        if q == 0.0:
            assert bd.zero_modes()[0].shape[0] == 2
        r = bd.solve(n_modes, tol=1e-9)
        assert np.allclose(r.omega.numpy(), [w for w, _ in exact], rtol=1e-6, atol=0)
        assert np.allclose(r.spin_weight.numpy(), [s for _, s in exact], atol=1e-6)


def test_su2_spin_rotation_is_a_kernel_vector():
    """Omega_R = 0, equal potentials, Delta = 0, spinor at a generic angle alpha (psi0 = sqrt(n) (cos, sin)):
    the spin rotation chi = (-psi2, psi1) is in ker A and ker B, so A^{-1} of the relative-phase vector
    (cos 2a psi0 - sin 2a chi) does not exist. It is detected (|A chi| / |chi| = the GP residual) and the
    constraints are built from chi and A^+ psi0: both solvers give the exact spectrum, the density branch
    sqrt(eps (eps + 2 g n)) and the free spin branch eps, k != 0, with no CG warning (the spin weight is
    measured in the layer basis, so it is (1 + sin 2a) / 2 here, the same in both solvers). With unequal
    potentials the kernel is not detected (the generic path); with one component empty (a = 0) it is
    detected when the empty component's potential is the same and not when it is 1e-8 higher (then A chi
    = 1e-8 chi: a soft mode, not a kernel)."""
    import warnings

    gn, a = 0.4, 0.3
    g = Grid((48,), (8.0,), device=CPU)
    eng = BilayerGPE(g, None, g=gn, Omega_R=0.0, n_components=2)
    psi0 = torch.zeros((1, 2, 48), dtype=C128)
    psi0[:, 0], psi0[:, 1] = math.cos(a), math.sin(a)
    assert float(eng.residual(psi0)[0]) < 1e-12
    kj = 2 * np.pi * np.fft.fftfreq(48, d=8.0 / 48)
    eps = KAPPA * kj**2
    eps = eps[eps > 1e-14]
    exact = np.sort(np.concatenate([_bogoliubov(eps, gn), eps]))[:8]
    bd = BdG(eng, psi0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bd.spin_kernel() is not None and bd.zero_modes()[0].shape[0] == 2
        rd = bd.solve(8, method="dense")
        r = bd.solve(8, tol=1e-9)
    assert np.allclose(rd.omega.numpy(), exact, rtol=1e-10)
    assert abs(rd.zero_mode_omega) < 1e-6
    assert np.allclose(r.omega.numpy(), exact, rtol=1e-6, atol=0)
    assert np.allclose(r.spin_weight.numpy(), rd.spin_weight.numpy(), atol=1e-6)
    V = torch.zeros((1, 2, 48), dtype=torch.float64)
    V[:, 1] = 0.05
    assert BdG(BilayerGPE(g, V, g=gn, Omega_R=0.0, n_components=2), psi0).spin_kernel() is None
    polarised = torch.zeros((1, 2, 48), dtype=C128)
    polarised[:, 0] = 1.0
    for d, found in ((0.0, True), (1e-8, False)):
        V = torch.zeros((1, 2, 48), dtype=torch.float64)
        V[:, 1] = d
        e = BilayerGPE(g, V, g=gn, Omega_R=0.0, n_components=2)
        assert float(e.residual(polarised)[0]) < 1e-12
        assert (BdG(e, polarised).spin_kernel() is not None) == found


def test_su2_lattice_spectrum_is_density_plus_spin_branch():
    """The SU(2) gas on a 1D lattice (both components in the same V0 = 8 lattice, Omega_R = 0, Delta = 0),
    psi0 = phi (cos a, sin a) with phi the one-component ground state of the total density: the spectrum is
    the one-component BdG spectrum of phi (density) and the positive eigenvalues of H_GP - mu (spin, the
    free particles in the mean field). Here psi0 is not uniform, so the preconditioner does not keep the
    complement of chi: the projected CG of A^+ psi0 is what makes the lobpcg converge to it."""
    import warnings

    eng1, r1 = _lattice_1d(4, ppa=12)
    g = eng1.grid
    V = lattice_1d(g, 8.0)
    eng2 = BilayerGPE(g, torch.cat([V, V], dim=1), g=0.2, Omega_R=0.0, n_components=2)
    a = 0.3
    phi = BdG(eng1, r1.psi).psi0  # real
    psi0 = torch.cat([math.cos(a) * phi, math.sin(a) * phi], dim=1).to(C128)
    b1 = BdG(eng1, r1.psi)
    w_dens = b1.solve(6, method="dense").omega.numpy()
    D = b1._dense_matrix(b1.apply_D, g.numel, torch.float64)
    e_spin = torch.linalg.eigvalsh(0.5 * (D + D.T)).numpy()
    e_spin = e_spin[e_spin > 1e-8]
    exact = np.sort(np.concatenate([w_dens, e_spin]))[:6]
    bd = BdG(eng2, psi0)
    assert bd.spin_kernel() is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rd = bd.solve(6, method="dense")
        r = bd.solve(6, tol=1e-9)
    assert np.allclose(rd.omega.numpy(), exact, rtol=1e-9)
    assert np.allclose(r.omega.numpy(), exact, rtol=1e-6)


def test_gauge_fix():
    """A global phase on psi0 changes nothing (the state is made real again
    to rounding); a psi0 that is not real up to a global phase warns."""
    eng, r = _lattice_1d(4, tol=1e-10)
    w_ref = BdG(eng, r.psi).solve(4).omega
    bd = BdG(eng, r.psi * complex(math.cos(0.7), math.sin(0.7)))
    assert bd.imag_fraction < 1e-12
    assert torch.allclose(bd.solve(4).omega, w_ref, rtol=1e-9)
    x = eng.grid.coord(0)
    with pytest.warns(UserWarning, match="not real"):
        BdG(eng, r.psi * torch.exp(0.3j * torch.sin(2 * math.pi * x / 4.0)))


# -- lattices -----------------------------------------------------------------------------


def _lattice_1d(n_sites, ppa=16, V0=8.0, gv=0.2, filling=1.0, twist=None, psi_start=None, tol=1e-11):
    g = Grid((n_sites * ppa,), (float(n_sites),), device=CPU)
    eng = BilayerGPE(g, lattice_1d(g, V0), g=gv, n_components=1, twist=twist)
    N = filling * n_sites
    start = initial_state(eng, N) if psi_start is None else psi_start
    r = relax(eng, start, N, tol=tol, polish_steps=20000)
    assert bool(r.converged.all()), r.report()
    return eng, r


def test_sound_speed_equals_fs_compressibility_1d_lattice():
    """Sound speed from superfluid fraction and compressibility: c^2 = f_s n dmu/dn / m,
    m = pi^2/2, i.e. c^2 = 2 kappa f_s n dmu/dn, on a 1D lattice V0 = 8,
    g = 0.2, one atom per site, 16 pts/a. f_s from the twist current
    (theta = 0.2 on 8 sites), dmu/dn by central difference (N +- 2 %).
    BdG: the phonons of the 8-site box at k = 2 pi/8 and 4 pi/8, computed as
    Bloch modes of one primitive cell (twist) and checked against the Gamma
    box modes of the 8-site box; c from the fit w/q = c + b q^2 through the
    two points: agreement to 1 % (it is 0.48 %, all of it the curvature beyond
    q^2 at q = pi/2; at q = 2 pi j/64 the two routes agree to 4e-5)."""
    e8, r8 = _lattice_1d(8)
    theta = 0.2
    etw, rtw = _lattice_1d(8, twist=(theta,), psi_start=r8.psi.clone())
    fs = float(superfluid_fraction_current(etw, rtw.psi, 0, theta)[0])
    mus = [float(_lattice_1d(8, filling=f, psi_start=r8.psi.clone() * math.sqrt(f))[1].mu[0]) for f in (0.98, 1.02)]
    nbar = 1.0
    dmu_dn = (mus[1] - mus[0]) / (0.04 * nbar)
    c_fs = math.sqrt(2.0 * KAPPA * fs * nbar * dmu_dn)

    e1, r1 = _lattice_1d(1, tol=1e-12)
    ss = sound_speed(e1, r1.psi, L=8.0, n_points=2)
    assert np.allclose(ss["q"], [2 * np.pi / 8, 4 * np.pi / 8])
    assert ss["used"].all()
    print(
        f"\n1D lattice: f_s = {fs:.6f}, dmu/dn = {dmu_dn:.6f}, c(f_s) = {c_fs:.6f}, "
        f"c(BdG, q = 2pi/8, 4pi/8) = {ss['c']:.6f} ({(ss['c'] / c_fs - 1) * 100:+.3f} %)"
    )
    assert abs(ss["c"] / c_fs - 1.0) < 0.01
    small = sound_speed(e1, r1.psi, L=64.0, n_points=3, method="dense")
    print(f"            c(BdG, q = 2pi j/64) = {small['c']:.6f} ({(small['c'] / c_fs - 1) * 100:+.4f} %)")
    assert abs(small["c"] / c_fs - 1.0) < 2e-4
    # the same phonons are the lowest Gamma modes of the 8-site box (doubly degenerate)
    r8g = BdG(e8, r8.psi).solve(4)
    assert np.allclose(r8g.omega.numpy(), np.repeat(ss["omega"], 2), rtol=1e-8)


def test_hugenholtz_pines_zero_mode():
    """At q = 0 the Goldstone mode (u, v) = (psi0, -psi0) has w = 0 up to the
    ground-state residual: B psi0 = H psi0 - mu psi0 is the GP residual, and
    the dense spectrum's projected-out eigenvalue obeys |w0|^2 < 10 x residual
    (w^2 of the Jordan-block zero mode moves at first order in the
    perturbation). After the projection the lowest mode is positive and equals
    the dense spectrum's next eigenvalue: the constraints f . psi0 = 0,
    (A^-1 psi0) . h = 0 remove exactly the zero mode. 1D lattice V0 = 8,
    g = 0.2 on 8 sites, and the two-component toy bilayer."""
    e8, r8 = _lattice_1d(8)
    bd = BdG(e8, r8.psi)
    assert bd.gp_residual == pytest.approx(float(e8.residual(r8.psi)[0]), rel=1e-3, abs=1e-14)
    rd = bd.solve(6, method="dense")
    rl = bd.solve(6)
    assert rd.zero_mode_omega**2 < 10.0 * bd.gp_residual
    assert float(rl.omega[0]) > 0.0
    assert np.allclose(rl.omega.numpy(), rd.omega.numpy(), rtol=1e-9)

    ns = 6
    g = Grid((ns * 16,), (float(ns),), device=CPU)
    eng = BilayerGPE(g, toy_bilayer_1d(g, 8.0, ns), g=0.2, Omega_R=0.05, n_components=2)
    r = relax(eng, initial_state(eng, float(ns)), float(ns), tol=1e-11, polish_steps=20000)
    assert bool(r.converged.all())
    bd2 = BdG(eng, r.psi)
    rd2, rl2 = bd2.solve(8, method="dense"), bd2.solve(8)
    assert rd2.zero_mode_omega**2 < 10.0 * bd2.gp_residual
    assert float(rl2.omega[0]) > 0.0
    assert np.allclose(rl2.omega.numpy(), rd2.omega.numpy(), rtol=1e-9)


def _supercell(points_per_a, V0=8.0, Omega_R=0.24, filling=1.0, tol=1e-10):
    u = lattice_units()
    geo = moire_geometry(4, 1)
    N = geo.grid_size(points_per_a)
    g = Grid((N, N), (geo.L_box, geo.L_box), device=CPU)
    eng = BilayerGPE(g, bilayer_potential(g, V0, geo), g=u.g2d, Omega_R=Omega_R, n_components=2)
    Np = filling * g.volume
    r = relax(eng, initial_state(eng, Np), Np, tol=tol, polish_steps=20000)
    assert bool(r.converged.all()), r.report()
    return geo, eng, r


def test_matrix_free_solver_matches_dense_on_supercell():
    """Two-component moire supercell (m = 4, V0 = 8, Omega_R = 0.24, filling 1)
    at 8 pts/a (2312 unknowns): the matrix-free solver and dense
    diagonalisation agree to 1e-9 at Gamma and at X; the full dense spectrum
    at Gamma (zero mode aside) is positive: B >= 0, A > 0, so every w is real."""
    geo, eng, r = _supercell(8.0)
    for q in ((0.0, 0.0), (math.pi / geo.L_box, 0.0)):
        bd = BdG(eng, r.psi, q=q)
        rd, rl = bd.solve(8, method="dense"), bd.solve(8)
        assert np.allclose(rl.omega.numpy(), rd.omega.numpy(), rtol=1e-9)
        assert np.allclose(rl.spin_weight.numpy(), rd.spin_weight.numpy(), atol=1e-6)
        assert float(rl.residual.max()) < 1e-8


def test_m4_supercell_spectrum():
    """m = 4 supercell, V0 = 8, Omega_R = 0.24, filling 1, 12 pts/a, contact
    g_2D: the BdG spectrum at Gamma (16 modes), X = (pi/L, 0) and
    M = (pi/L, pi/L) (8 modes) is real and positive with int (|u|^2 - |v|^2)
    = 1. Reported (not pinned): the lowest X mode against c pi/L with c from
    the small-q slope, and the lowest spin-like mode at Gamma against
    2 Omega_R x the layer overlap S = int psi1 psi2 / sqrt(N1 N2)."""
    geo, eng, r = _supercell(12.0)
    L = geo.L_box
    rg = BdG(eng, r.psi).solve(16)
    bands = bdg_bands(eng, r.psi, [(math.pi / L, 0.0), (math.pi / L, math.pi / L)], n_modes=8)
    results = [rg] + bands["results"]
    for res in results:
        E = res.omega.numpy()
        assert np.isfinite(E).all() and (E > 0).all()
        assert float(res.residual.max()) < 1e-8
        assert torch.allclose(res.norm_check, torch.ones(len(E), dtype=torch.float64), atol=1e-10)
        assert (res.participation_ratio > 0).all() and (res.participation_ratio <= 1 + 1e-12).all()
    ss = sound_speed(eng, r.psi, n_points=2, n_modes=4)
    S = float(
        torch.abs((r.psi[0, 0].conj() * r.psi[0, 1]).sum())
        / torch.sqrt((r.psi[0, 0].abs() ** 2).sum() * (r.psi[0, 1].abs() ** 2).sum())
    )
    print(f"\nm = 4 supercell, V0 = 8, Omega_R = 0.24, filling 1 (GP residual {rg.info['gp_residual']:.1e})")
    for nm, res in zip(("Gamma", "X", "M"), results):
        print(f"  {nm:5s} w    = {np.array2string(res.omega.numpy(), precision=5, max_line_width=200)}")
        print(f"        spin = {np.array2string(res.spin_weight.numpy(), precision=3, max_line_width=200)}")
    print(
        f"  sound speed c = {ss['c']:.5f} a E_R/hbar; c pi/L = {ss['c'] * math.pi / L:.5f} "
        f"vs lowest X mode {float(results[1].omega[0]):.5f}"
    )
    sp = torch.nonzero(rg.spin_weight > 0.5).reshape(-1)
    spin_txt = f"{float(rg.omega[int(sp[0])]):.5f} (mode {int(sp[0])})" if len(sp) else "none"
    print(
        f"  layer overlap S = {S:.4f}, 2 Omega_R S = {2 * 0.24 * S:.4f}; lowest spin-like mode "
        f"(spin weight > 0.5) at Gamma among 16: {spin_txt}"
    )
    assert ss["c"] > 0


def test_one_empty_component_has_one_goldstone_mode():
    """Omega_R = 0 with component 1 empty (psi0 = (0, sqrt n), the ground state at Delta > 0): the
    relative-phase vector (psi1, -psi2) = -psi0 is not a second Goldstone mode, so only one direction is
    projected out. Exact spectrum: the Bogoliubov branch of component 2, sqrt(eps (eps + 2 g n)) for
    k != 0, and the free component-1 particles in its field, eps + Delta for every k (k = 0 included).
    Both solvers (the dense one used to drop a valid mode, lobpcg raised on the singular Gram matrix)."""
    gn, dl = 0.4, 0.3
    g = Grid((48,), (8.0,), device=CPU)
    eng = BilayerGPE(g, None, g=gn, Omega_R=0.0, delta=dl, n_components=2)
    psi0 = torch.zeros((1, 2, 48), dtype=C128)
    psi0[:, 1] = 1.0
    assert float(eng.residual(psi0)[0]) < 1e-12
    bd = BdG(eng, psi0)
    assert bd.zero_modes()[0].shape[0] == 1
    kj = 2 * np.pi * np.fft.fftfreq(48, d=8.0 / 48)
    eps = KAPPA * kj**2
    exact = np.sort(np.concatenate([_bogoliubov(eps[eps > 1e-14], gn), eps + dl]))[:8]
    assert np.allclose(bd.solve(8, method="dense").omega.numpy(), exact, rtol=1e-10)
    assert np.allclose(bd.solve(8, tol=1e-9).omega.numpy(), exact, rtol=1e-6)


def test_mu_does_not_inherit_single_precision():
    """With a complex64 engine the BdG mu is <psi0|H_GP|psi0> from the operator's own float64 blocks, not
    the engine's single-precision energy_terms: it equals the complex128 mu of the same (float32-rounded)
    potential to 1e-12, while the complex64 energy_terms is off by more than 1e-9."""
    g = Grid((96,), (6.0,), device=CPU)
    e128 = BilayerGPE(g, lattice_1d(g, 8.0), g=0.2, n_components=1)
    r = relax(e128, initial_state(e128, 6.0), 6.0, tol=1e-11, polish_steps=20000)
    e64 = e128.to_dtype(torch.complex64)
    e64_in_128 = e64.to_dtype(torch.complex128)  # the rounded V and parameters, evaluated in double
    mu_ref = float(e64_in_128.energy_terms(r.psi)["mu"][0])
    mu64_engine = float(e64.energy_terms(r.psi.to(torch.complex64))["mu"][0])
    assert abs(mu64_engine - mu_ref) > 1e-9  # the engine's single precision is visible
    assert BdG(e64, r.psi).mu == pytest.approx(mu_ref, rel=1e-12)
    assert BdG(e64_in_128, r.psi).mu == pytest.approx(mu_ref, rel=1e-14)


def test_cg_keeps_an_exactly_converged_column_finite():
    """_cg on a block whose second column has a part in the operator's kernel (it never converges):
    the first column converges exactly (r = 0) within a few iterations and stays the solution while the
    second one runs on, instead of turning into 0/0 = NaN."""
    import warnings

    from twisted_bilayer.engine import bdg

    d = torch.tensor([0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=torch.float64).to(torch.complex128)
    b = torch.zeros(2, 7, dtype=torch.complex128)
    b[0, 1:] = torch.tensor([1.0, 2.0, 0.5, -1.0, 0.3, 0.7], dtype=torch.float64)
    b[1, :] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, it = bdg._cg(lambda X: X * d, b, lambda r: r, tol=1e-12, maxiter=60)
    assert it == 60 and bool(torch.isfinite(x[0]).all())
    assert float((x[0] * d - b[0]).abs().max()) < 1e-14


def test_batched_potential_is_refused():
    """A potential of shape (B, C, *N) with B > 1 is caught (engine.batch_hint looks at g, Omega_R, delta and
    the twist only, so it used to pass and the blocks broadcast wrongly); a single potential is accepted."""
    grid = Grid((16, 16), (6.0, 6.0), real_dtype=torch.float64)
    V = torch.zeros(2, 2, 16, 16, dtype=torch.float64)
    V[1] += 0.5
    psi = torch.ones(1, 2, 16, 16, dtype=C128) / 16
    eng = BilayerGPE(grid, V=V, g=0.1, Omega_R=0.0, delta=0.0, n_components=2, dtype=C128)
    assert eng.batch_hint == 1
    with pytest.raises(ValueError, match="V batch 2"):
        BdG(eng, psi)
    BdG(BilayerGPE(grid, V=V[:1], g=0.1, Omega_R=0.1, delta=0.0, n_components=2, dtype=C128), psi)
