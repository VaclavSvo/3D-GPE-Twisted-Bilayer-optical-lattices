"""Analytic references and pinned anchors, in code units (hbar = m = 1, a = 395 nm,
E_R = pi^2/2 = 4.934802) unless a name says otherwise. Every number here is either
exact (closed form computed on the spot) or a pinned numerical value; its source string says
which codes reproduced it.

Nothing here depends on an engine. Engine drivers call these to build gates.
"""

from __future__ import annotations

import math

import numpy as np

from twisted_bilayer import parameters as P

E_R = P.E_R_CODE
OMEGA_Z = P.OMEGA_Z

# ----------------------------------------------------------------------------- exact


def ho_ground_energy(omega: float, dims: int = 1) -> float:
    """E0 = dims * omega / 2 (hbar = m = 1)."""
    return 0.5 * dims * omega


def ho_width(omega: float) -> float:
    """rms width sqrt(<x^2>) = 1/sqrt(2 omega) of the HO ground state."""
    return 1.0 / math.sqrt(2.0 * omega)


def ho_breathing_period(omega: float) -> float:
    """A non-interacting cloud squeezed in a harmonic trap breathes <x^2>(t) at 2 omega."""
    return 2.0 * math.pi / (2.0 * omega)


def free_gaussian_sigma(sigma0: float, t: float) -> float:
    """rms width of a free Gaussian wavepacket: sigma(t) = sigma0 sqrt(1 + (t / (2 sigma0^2))^2)."""
    return sigma0 * math.sqrt(1.0 + (t / (2.0 * sigma0**2)) ** 2)


def lattice_band_bottom_1d(V0_ER: float, n_pw: int = 41) -> float:
    """Exact band bottom (q = 0) of V0 sin^2(pi x / a) in code units, by plane waves.
    V0 sin^2 = V0/2 - (V0/4)(e^{2ikx} + c.c.), k = pi/a: H_{mn} = (2 pi n)^2/2 delta_mn
    + V0/2 delta_mn - V0/4 delta_{m,n+-1}.  At 4 E_R: 1.54486 E_R (Mathieu)."""
    V0 = V0_ER * E_R
    n = np.arange(-(n_pw // 2), n_pw // 2 + 1)
    H = np.diag(0.5 * (2.0 * np.pi * n) ** 2 + 0.5 * V0)
    off = -0.25 * V0
    H += np.diag(np.full(n_pw - 1, off), 1) + np.diag(np.full(n_pw - 1, off), -1)
    return float(np.linalg.eigvalsh(H)[0])


def lattice_bloch_weights_1d(V0_ER: float, n_pw: int = 41, n_keep: int = 3) -> np.ndarray:
    """|c_n|^2 of the 1D q = 0 Bloch ground state, summed over +-n: at 4 E_R
    0.90583 (n = 0), 0.09382 (|n| = 1, i.e. 0.04691 per side), 3.5e-4 (|n| = 2).
    The 2D separable weights 0.8205 / 0.0425 / 0.0022 are |c_0|^4, |c_0|^2 |c_1|^2 and
    |c_1|^4 of these."""
    V0 = V0_ER * E_R
    n = np.arange(-(n_pw // 2), n_pw // 2 + 1)
    H = np.diag(0.5 * (2.0 * np.pi * n) ** 2 + 0.5 * V0)
    off = -0.25 * V0
    H += np.diag(np.full(n_pw - 1, off), 1) + np.diag(np.full(n_pw - 1, off), -1)
    w, v = np.linalg.eigh(H)
    c = np.abs(v[:, 0]) ** 2
    out = []
    for m in range(n_keep):
        if m == 0:
            out.append(c[n == 0].sum())
        else:
            out.append(c[np.abs(n) == m].sum())
    return np.array(out)


def rabi_population(J_perp: float, t, delta: float = 0.0) -> np.ndarray:
    """Two-level, start in |1>: N2/N = (J^2 / W^2) sin^2(W t), W = sqrt(J^2 + delta^2/4).
    J_perp is the off-diagonal matrix element (the convention of every engine here)."""
    W = math.sqrt(J_perp**2 + 0.25 * delta**2)
    t = np.asarray(t, dtype=float)
    return (J_perp**2 / W**2) * np.sin(W * t) ** 2


def kinetic_phase_per_step(dx: float, dt: float, dims_dx: tuple = None) -> float:
    """Largest kinetic phase per split step, k_max^2 dt / 2, k_max = pi/dx per axis.
    dims_dx: tuple of grid spacings; k_max^2 = sum (pi/dx_i)^2. Must stay < pi (the split step
    is unstable near pi); production runs keep <= 2.5."""
    if dims_dx is None:
        dims_dx = (dx, dx)
    k2 = sum((math.pi / d) ** 2 for d in dims_dx)
    return 0.5 * k2 * dt


# --------------------------------------------------------------------------- anchors
# Pinned values, most reproduced by two or more independent codes. Code units unless the name says
# otherwise. In a source string, "this engine" is the PyTorch engine of this package; the other codes
# are an independent quasi-2D split-step solver and an independent 3D solver.
ANCHORS = {
    # (name): (value, rel_tol, where it comes from)
    "band_bottom_1d_4ER_ER": (1.54486, 2e-5, "Mathieu / plane waves"),
    "supercell_EN_4ER_g0_J0_ER": (
        3.089723,
        2e-6,
        "2D supercell E/N, V0 = 4, g = J = 0 (this engine, quasi-2D split-step solver, 2 x 1.54486)",
    ),
    "supercell_EN_4ER_g0_J0_code": (15.247171, 2e-6, "same in code units"),
    "dressed_band_bottom_J0.2428_code": (
        14.17705565,
        1e-8,
        "E0(J_perp = 0.24277), g = 0, 176^2 (this engine, quasi-2D and 3D solvers, all digits)",
    ),
    "dressed_band_bottom_J0.1214_code": (14.72939727, 1e-8, "E0(J_perp = 0.12138), g = 0, 176^2"),
    "mu_J0.243_n1_NPSE_lam1_ER": (
        2.908439,
        5e-6,
        "quasi-2D mu at (0.243, nbar = 1), NPSE lambda = 1 (quasi-2D split-step solver 2.908439)",
    ),
    "mu_J0.243_n1_fixedg_ER": (
        2.909104,
        1e-5,
        "same with fixed g2D (this engine and the quasi-2D split-step solver to 6 digits)",
    ),
    "mu3d_minus_Ez0_J0.243_n3_code": (14.5636065, 1e-7, "3D pancake, 176^2 x 32, independent solver to 2e-7"),
    "E_z0_code": (
        0.6707839391,
        1e-9,
        "hbar omega_z / 2 = parameters.OMEGA_Z / 2",
    ),
    "fs_twist_J0.243_n1_fixedg": (
        0.6128,
        3e-4,
        "superfluid fraction by twist, this engine and the quasi-2D split-step solver",
    ),
    "fs_g0_4ER_m_over_mstar": (0.652, 2e-3, "g -> 0 limit = exact band curvature m/m* at 4 E_R"),
    "franck_condon_4ER": (0.8205, 5e-3, "<F> = |c_0|^4; three codes 0.813-0.819 in dynamics"),
    "C_psf_J0.243_n1_fixedg": (0.731, 4e-3, "converged fixed-g in-situ contrast after PSF"),
    "C_psf_J0.243_n1_NPSE_lam1": (0.7456, 4e-3, "quasi-2D split-step solver, NPSE lambda = 1"),
    "toy1d_fs_V8_gn0.2_energy": (0.3150, 2e-3, "1D toy, single component, V0 = 8, g nbar = 0.2"),
    # This engine (E_R units): the band bottom in its own units, and its NPSE mu with its own g2D from
    # a_s = 100 a0 (see engine.units EXCEPTIONS); 7e-6 E_R above the 5.29 nm value.
    "dressed_band_bottom_J0.2428_ER": (2.872872, 3e-7, "this engine, = 14.17705565 code / E_R_CODE"),
    "mu_J0.243_n1_NPSE_lam1_engine_native_ER": (
        2.908446,
        5e-6,
        "this engine, 16 pts/a, g2D from a_s = 100 a0",
    ),
    # trapped cloud N = 2000, J_perp = 1.19802 code, NPSE lambda = 1, 20 Hz, 3 x 3 cells at 8 pts/a
    # (an independent quasi-2D split-step solver). Tolerance 2e-3: absolute for the contrasts,
    # relative for R_rms.
    "trapped_N2000_J0.243_lam1_C_raw": (
        0.5855390,
        2e-3,
        "quasi-2D split-step solver in-situ C_raw_AAAB (|k| < pi/a envelope of |2>)",
    ),
    "trapped_N2000_J0.243_lam1_C_coh": (
        0.5748301,
        2e-3,
        "quasi-2D split-step solver in-situ C_coh_AAAB, coherent imaging s = 0.01",
    ),
    "trapped_N2000_J0.243_lam1_R_rms_a": (
        15.365620,
        2e-3,
        "quasi-2D split-step solver cloud R_rms = 6.069 um (relative tol)",
    ),
    # The lattice model (dimensionless): the n = 1 superfluid-Mott transition of the plain square
    # lattice (z = 4) in U/J. Single-site mean field: the lobe tip U/zJ = 3 + 2 sqrt 2 (Fisher et al.,
    # Phys. Rev. B 40, 546 (1989); Sheshadri et al., Europhys. Lett. 22, 257 (1993)), exact to rounding.
    # Worm quantum Monte Carlo: (J/U)_c = 0.05974(3), i.e. (U/J)_c = 16.739(8), Capogrosso-Sansone, Soyler,
    # Prokof'ev and Svistunov, Phys. Rev. A 77, 015602 (2008), arXiv:0710.2703 (the 2D paper; Phys. Rev. B
    # 75, 134302 (2007) by the same group is the 3D cubic lattice, (U/J)_c = 29.34(2)).
    "square_lattice_UJ_c_gutzwiller": (4.0 * (3.0 + 2.0 * math.sqrt(2.0)), 1e-12, "z (3 + 2 sqrt 2) = 23.3137"),
    "square_lattice_UJ_c_qmc": (16.74, 1e-3, "Capogrosso-Sansone et al. 2008, worm QMC: 1/0.05974(3)"),
}


# Anchors that no gate reads: documented values of the same calculations, kept for comparison with other codes
# and by hand (the gates read their siblings: the band bottom through plane waves on the spot, the supercell E/N
# in E_R, the 2D mu and f_s). `tests/test_gates.py` checks that this list and the suite agree.
UNREAD_BY_GATES = (
    "band_bottom_1d_4ER_ER",
    "supercell_EN_4ER_g0_J0_code",
    "mu3d_minus_Ez0_J0.243_n3_code",
    "fs_g0_4ER_m_over_mstar",
    "toy1d_fs_V8_gn0.2_energy",
)


def anchor(name: str) -> tuple[float, float, str]:
    v, tol, src = ANCHORS[name]
    return v, tol, src
