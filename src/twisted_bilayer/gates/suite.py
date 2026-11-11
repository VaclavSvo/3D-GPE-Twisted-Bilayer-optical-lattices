"""Physics gates for the engine (`twisted_bilayer.engine`: the PyTorch two-component GPE in lattice units
a = 1, E_R = 1, hbar = 1), run on the real engine.

    twisted-bilayer gates [--out DIR] [--full] [--trapped] [--only A1,B9,...] [--device cpu|cuda]

The scenarios are defined in code units (hbar = m = a = 1), the units of the anchors in references.py,
and converted at the boundary with `twisted_bilayer.parameters` (energies and rates / E_R_CODE, times x E_R_CODE;
lengths and densities identical):

  A  exact single-particle references (g = 0): 2D harmonic oscillator by imaginary time (A1),
     breathing at 2 omega (A2), free Gaussian expansion (A3), 1D / separable 2D / sqrt(485)-supercell
     band bottom of V0 sin^2 at 4 E_R (A4; 3.089723 E_R anchor), Bloch weights |c0|^4, |c0 c1|^2,
     |c1|^4 = Franck-Condon 0.8205 (A5), two-level Rabi with and without detuning plus the detuning
     sign (A6);
  B  exact identities on the sqrt(485) twisted cell: gauge copy H(-J) = sz H(J) sz (B7), N2/N
     conserved at J = 0 (B8), C4 isotropy of the twist superfluid fraction with zero ground-state
     momentum, the energy = current routes and the 0.6128 anchor (B9, 176^2 in both modes),
     reversibility dt -> -dt (B10);
  C  anchors: dressed band bottom E0(J = 0.24277 E_R) = 14.17705565 code = 2.872872 E_R (C11, 176^2);
     mu at (0.243, nbar = 1) for NPSE lambda = 1 and fixed g (C12) and C_psf (C13). g2D =
     parameters.G2D_ER (a_s = 5.29 nm), the value behind the anchors; the engine's own g2D (a_s = 100 a0,
     units.EXCEPTIONS) is run next to it and gated against its pinned 2.908446, and the
     difference is recorded in meta. quick: C12/C13 at 4 pts/a with loose tolerances; --full: 176^2
     with the pinned ones;
  E  (only with --trapped or --only E; ~10 min CPU) the trapped cloud through engine.trapped:
     N = 2000, J = 0.24277 E_R, NPSE lambda = 1, 20 Hz, 3 x 3 cells at 8 pts/a, g2D = parameters.G2D_ER;
     C_raw, C_coh (s = 0.01) and R_rms vs the anchors of an independent quasi-2D split-step solver to 2e-3,
     the polish converged, N and N2/N conserved, the cloud inside the box;
  D  pre-flight: the NPSE parameter kappa = g2D n_peak / hbar omega_z at nbar = 1. The nbar = 5 value
     (expected 1.31 > 0.8, a model-validity flag) is recorded in meta as an expected-FAIL demonstration, not
     counted. The kinetic phase per step of the engine's own choose_dt (order 2, and order 4 with the
     `rabi_quick` scenario's multiplier 20 -- outer step and largest Yoshida sub-step), the identity
     kappa k2_max dt (E_R units) = k2_max dt / 2 (code units) and the lattice resolution are recorded in
     values["D"] and checked by tests/test_gates.py: they follow from choose_dt's constants.

Device: `twisted-bilayer gates` and `python -m twisted_bilayer.gates` (both through cli.py) set
TWISTED_BILAYER_FORCE_CPU=1 before the engine is imported unless --device cuda is given, so they run on the CPU
by default; --device cuda needs a CUDA build of torch. `python -m twisted_bilayer.gates.suite` imports the engine
before it reads the command line and runs on the device the engine was imported on unless --device names one
(a mismatch is an error; set TWISTED_BILAYER_FORCE_CPU=1 to run it on the CPU of a GPU machine).
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import platform
import sys
import time

import numpy as np
import torch

from twisted_bilayer import parameters as P
from twisted_bilayer.engine import backend, npse, units
from twisted_bilayer.engine import ground_state as gs
from twisted_bilayer.engine import lattice as lat
from twisted_bilayer.engine import observables as obs
from twisted_bilayer.engine import physics as phys
from twisted_bilayer.engine.grid import Grid

from . import core as H
from . import references as R
from .core import GateResult, GateSuite, value_gate

ER = P.E_R_CODE  # code energy of 1 E_R
G2D_ER = P.G2D_ER  # 0.0157585 E_R a^2 (a_s = 5.29 nm): the anchors' coupling
WZ_ER = P.HBAR_OMEGA_Z_ER  # hbar omega_z = 0.271858 E_R
J_PROD_CODE = P.J_PERP_CODE[0.243]  # 1.19802 code = 0.24277 E_R: the paper's Omega_R = 2 pi x 893 Hz
L485 = P.L_SUPERCELL
ALL_IDS = ("A1", "A2", "A3", "A4", "A5", "A6", "B7", "B8", "B9", "B10", "C11", "C12", "C13", "D")
TRAPPED_IDS = ("E",)  # run only on request (--trapped / --only E)


# ======================================================== reference situations
# Plain parameter dicts in code units (hbar = m = a = 1); the scenarios below convert at the boundary
# with twisted_bilayer.parameters.


def ho_case(
    omega: float = 1.0,
    L: float = 16.0,
    N: int = 64,
    squeeze: float = 0.8,
    dt: float = 5e-4,
    n_periods: float = 1.0,
    start_sigma_factor: float = 1.3,
) -> dict:
    """2D isotropic harmonic trap V = omega^2 r^2 / 2 on an L x L periodic box of N^2 points
    (box edge at 11 sigma). Ground state: E0 = omega (2D), rms width per axis sigma0 = 1/sqrt(2 omega).
    Breathing: the Gaussian of width `squeeze` x sigma0 evolved for n_periods x pi/omega."""
    sigma0 = R.ho_width(omega)
    return dict(
        omega=omega,
        L=L,
        N=N,
        sigma0=sigma0,
        E0=R.ho_ground_energy(omega, dims=2),
        start_sigma=start_sigma_factor * sigma0,
        squeeze=squeeze,
        dt=dt,
        t_end=n_periods * R.ho_breathing_period(omega),
        period=R.ho_breathing_period(omega),
        n_frames=200,
    )


def free_gaussian_case(
    sigma0: float = 0.5, L: float = 16.0, N: int = 64, t_end: float = 1.0, dt: float = 0.01, n_frames: int = 10
) -> dict:
    """V = 0: a Gaussian of rms width sigma0 per axis expands to sigma0 sqrt(1 + (t/2 sigma0^2)^2)."""
    return dict(
        sigma0=sigma0, L=L, N=N, t_end=t_end, dt=dt, n_frames=n_frames, sigma_end=R.free_gaussian_sigma(sigma0, t_end)
    )


def lattice_1d_case(V0_ER: float = 4.0, n_cells: int = 4, ppa: int = 8) -> dict:
    """Single lattice V0 sin^2(pi x) on n_cells x a at ppa points per a (q = 0 band bottom); the
    separable 2D version has E/N = 2 x the 1D value and Bloch weights |c0|^4, |c0 c1|^2, |c1|^4."""
    w = R.lattice_bloch_weights_1d(V0_ER)
    c0_2, c1_2 = float(w[0]), float(w[1]) / 2.0
    return dict(
        V0_ER=V0_ER,
        V0=V0_ER * ER,
        n_cells=n_cells,
        ppa=ppa,
        Nx=n_cells * ppa,
        E_1d=R.lattice_band_bottom_1d(V0_ER),
        weights_2d=dict(w0=c0_2**2, w10=c0_2 * c1_2, w11=c1_2**2),
    )


def rabi_case(
    J: float = 0.1, delta: float = 0.0, t_end: float = 40.0, dt: float = 0.05, every: int = 10, N: int = 4
) -> dict:
    """Uniform two-level gas, V = g = 0, start in |1>: N2/N = (J^2/W^2) sin^2(W t), W = sqrt(J^2 + delta^2/4)."""
    return dict(J=J, delta=delta, t_end=t_end, dt=dt, every=every, N=N, L=1.0)


def twisted_cell_case(ppa: int = 4, V0_ER: float = 4.0, J: float = J_PROD_CODE, nbar: float = 1.0) -> dict:
    """The sqrt(485) a twisted supercell (2 x 2 moire cells, both lattices periodic) at ppa points per a:
    N = 22 ppa; nbar = atoms per lattice site."""
    return dict(L=L485, N=22 * ppa, ppa=ppa, V0_ER=V0_ER, V0=V0_ER * ER, J=J, nbar=nbar, N_atoms=nbar * L485**2)


def lattice1_reciprocal() -> np.ndarray:
    """Primitive reciprocal vectors of lattice 1 (paper labels; rows, 1/a): (2 pi/L)(m, -n), (n, m)."""
    m, n = P.TWIST_M, P.TWIST_N
    return (2.0 * math.pi / L485) * np.array([[m, -n], [n, m]], dtype=float)


er = P.code_to_er  # an energy, rate or coupling: code units -> E_R
t_er = P.t_code_to_er  # a time: code units -> hbar/E_R
res_tol = P.code_to_er  # a residual tolerance in code energy units -> E_R


def supercell_grid(N: int) -> tuple[Grid, lat.MoireGeometry]:
    geo = lat.moire_geometry(22, 1)
    assert abs(geo.L_box - L485) < 1e-12
    return Grid((N, N), (geo.L_box, geo.L_box)), geo


def make_engine(
    grid,
    V,
    J: float = 0.0,
    g: float = 0.0,
    vertical: str = "fixed",
    twist=None,
    C: int = 2,
    g_npse: float = G2D_ER,
    delta: float = 0.0,
) -> phys.BilayerGPE:
    inter = npse.NPSE2D(g_npse, WZ_ER) if vertical == "npse" else None
    return phys.BilayerGPE(
        grid,
        V=V,
        g=0.0 if inter is not None else g,
        Omega_R=J,
        delta=delta,
        n_components=C,
        twist=twist,
        dtype=torch.complex128,
        interaction=inter,
    )


def field_from(grid, amp1: np.ndarray, amp2: np.ndarray | None = None) -> torch.Tensor:
    """(1, C, *N) complex128 tensor from per-component numpy amplitudes in grid ('ij') order."""
    comps = [amp1] if amp2 is None else [amp1, amp2]
    arr = np.stack(comps)[None].astype(np.complex128)
    return torch.as_tensor(arr, device=grid.device)


def mesh(grid) -> tuple[np.ndarray, ...]:
    """Coordinate arrays in the engine's ('ij') order, numpy float64."""
    xs = [x.numpy() for x in grid.x64]
    return np.meshgrid(*xs, indexing="ij")


def gaussian(grid, sigma: float) -> np.ndarray:
    """Unit-norm amplitude exp(-r^2 / 4 sigma^2): rms width sigma per axis of the density."""
    X = mesh(grid)
    f = np.exp(-sum(x**2 for x in X) / (4.0 * sigma**2))
    return f / math.sqrt((f**2).sum() * grid.dV)


def momentum_per_atom(eng, psi) -> list[float]:
    """P_i / N = int j_i / (2 kappa) / N  (hbar / a: the same number in code units)."""
    js = eng.current(psi)
    N = float(eng.norm(psi)[0])
    return [float(eng.grid.integrate(j)[0]) / (2.0 * eng.kappa) / N for j in js]


def highk_mask(grid, frac: float = 0.75) -> np.ndarray:
    """|k| > frac k_nyquist on the bare FFT grid: the high-k monitor of a history."""
    ks = np.meshgrid(*[k.numpy() for k in grid.k64], indexing="ij")
    k_nyq = min(math.pi / dx for dx in grid.dx)
    return sum(k**2 for k in ks) > (frac * k_nyq) ** 2


new_history, tagged, fit_period = H.new_history, H.tagged, H.fit_period


def record(hist: dict, eng, psi, t: float, mask: np.ndarray) -> dict:
    terms = eng.energy_terms(psi)
    p = backend.to_numpy(psi)[0]
    pk = np.abs(np.fft.fftn(p, axes=tuple(range(1, p.ndim)))) ** 2
    cn = backend.to_numpy(eng.component_norms(psi))[0]
    hist["t"].append(float(t))
    hist["norm"].append(float(terms["norm"][0]))
    hist["E_total"].append(float(terms["E_total"][0]))
    hist["N2_over_N"].append(float(cn[1] / cn.sum()) if len(cn) > 1 else 0.0)
    hist["highk"].append(float(pk[:, mask].sum() / pk.sum()))
    return terms


def evolve_recording(
    eng, psi, dt: float, nsteps: int, every: int, hist: dict, callback=None, order: int = 2
) -> torch.Tensor:
    """nsteps real-time steps of the engine's run_block, a history frame (and callback(t, psi)) every
    `every` steps; t in hbar/E_R."""
    mask = highk_mask(eng.grid)
    record(hist, eng, psi, 0.0, mask)
    if callback is not None:
        callback(0.0, psi)
    done = 0
    while done < nsteps:
        n = min(every, nsteps - done)
        psi = eng.run_block(psi, n, dt, order=order)
        done += n
        record(hist, eng, psi, done * dt, mask)
        if callback is not None:
            callback(done * dt, psi)
    return psi


def residual_gate(tag: str, residual: float, code_target: float) -> list[GateResult]:
    return tagged(H.imaginary_time_gates({"residual": [residual]}, residual_target=res_tol(code_target)), tag)


def rms_widths(eng, psi) -> list[float]:
    n = backend.to_numpy(eng.density(psi))[0, 0]
    X = mesh(eng.grid)
    N = n.sum()
    return [math.sqrt((n * x**2).sum() / N) for x in X]


class Context:
    """Shared engines and cached ground states of one run."""

    def __init__(self, quick: bool, verbose: bool):
        self.quick, self.verbose = quick, verbose
        self._states: dict = {}
        self.timings: dict = {}
        self.values: dict = {}

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  .. {msg}", file=sys.stderr, flush=True)

    def supercell(
        self, N: int, J: float, g: float, vertical: str, twist=None, g_npse: float = G2D_ER
    ) -> phys.BilayerGPE:
        grid, geo = supercell_grid(N)
        V = lat.bilayer_potential(grid, 4.0, geo)
        return make_engine(grid, V, J=J, g=g, vertical=vertical, twist=twist, g_npse=g_npse)

    def ground(
        self, N: int, J: float, nbar: float, vertical: str, g: float = G2D_ER, code_tol: float = 1e-9
    ) -> tuple[phys.BilayerGPE, gs.RelaxResult]:
        """Polished ground state on the sqrt(485) cell (V0 = 4 E_R) from the engine's own start
        (uniform, component 2 at phase pi for J > 0), relax = ladder + heavy-ball polish."""
        key = (N, J, nbar, vertical, g, code_tol)
        if key not in self._states:
            eng = self.supercell(N, J, g, vertical, g_npse=g)
            Nat = nbar * L485**2
            t0 = time.perf_counter()
            r = gs.relax(eng, gs.initial_state(eng, Nat), Nat, tol=res_tol(code_tol))
            dtm = time.perf_counter() - t0
            self.timings[f"ground N={N} J={J:.5g} nbar={nbar:g} {vertical} g={g:.7g}"] = dtm
            self.log(
                f"ground N={N} J={J:.5g} nbar={nbar:g} {vertical} g={g:.7g}: mu={float(r.mu[0]):.9f} E_R "
                f"res={float(r.residual[0]):.1e} steps={r.steps} ({dtm:.1f} s)"
            )
            self._states[key] = (eng, r)
        return self._states[key]


# ===================================================================== scenarios
def gate_A1_ho_ground_state(suite: GateSuite, ctx: Context) -> dict:
    """[analytic] 2D HO ground state by imaginary time: E0 = hbar omega, rms width 1/sqrt(2 m omega)."""
    c = ho_case()
    grid = Grid((c["N"], c["N"]), (c["L"], c["L"]))
    w = er(c["omega"])
    eng = make_engine(grid, lat.harmonic_trap(grid, (w, w)), C=1)
    psi0 = field_from(grid, gaussian(grid, c["start_sigma"]))
    r = gs.relax(eng, psi0, 1.0, tol=res_tol(1e-12))
    E = float(r.energy[0])
    sx, sy = rms_widths(eng, r.psi)
    tag = "A1 HO ground state"
    suite.add(value_gate(f"{tag}: E0 vs hbar omega (2D)", E, w, 1e-4))
    suite.add(value_gate(f"{tag}: rms x width vs 1/sqrt(2 m omega)", sx, c["sigma0"], 1e-4))
    suite.add(value_gate(f"{tag}: rms y width vs 1/sqrt(2 m omega)", sy, c["sigma0"], 1e-4))
    for g in residual_gate(tag, float(r.residual[0]), 1e-10):
        suite.add(g)
    return dict(E=E, E_ref=w, sx=sx)


def gate_A2_ho_breathing(suite: GateSuite, ctx: Context) -> dict:
    """[analytic] squeezed HO ground state breathes <r^2>(t) at exactly 2 omega."""
    c = ho_case()
    grid = Grid((c["N"], c["N"]), (c["L"], c["L"]))
    w = er(c["omega"])
    eng = make_engine(grid, lat.harmonic_trap(grid, (w, w)), C=1)
    psi = field_from(grid, gaussian(grid, c["squeeze"] * c["sigma0"]))
    dt = t_er(c["dt"])
    nsteps = int(round(c["t_end"] / c["dt"]))
    every = max(1, nsteps // c["n_frames"])
    X, Y = mesh(grid)
    ts, r2 = [], []

    def cb(t, p):
        n = backend.to_numpy(eng.density(p))[0, 0]
        ts.append(t)
        r2.append(float((n * (X**2 + Y**2)).sum() / n.sum()))

    hist = new_history()
    evolve_recording(eng, psi, dt, nsteps, every, hist, callback=cb)
    period = fit_period(ts, r2, 2.0 * w * 1.02)
    tag = "A2 HO breathing"
    suite.add(value_gate(f"{tag}: period of <r^2>(t) vs 2 pi / 2 omega", period, t_er(c["period"]), 1e-3))
    # one component: no N2/N gate (N2_over_N is written as 0.0 every frame, a gate on it cannot fail)
    for g in tagged(H.conservation_gates(hist, tol_norm=1e-9, tol_energy=1e-7), tag):
        suite.add(g)
    return dict(period=period, period_ref=t_er(c["period"]))


def gate_A3_free_expansion(suite: GateSuite, ctx: Context) -> dict:
    """[analytic] free Gaussian: sigma(t) = sigma0 sqrt(1 + (t / 2 m sigma0^2)^2), sigma doubles."""
    c = free_gaussian_case()
    grid = Grid((c["N"], c["N"]), (c["L"], c["L"]))
    eng = make_engine(grid, None, C=1)
    psi = field_from(grid, gaussian(grid, c["sigma0"]))
    nsteps = int(round(c["t_end"] / c["dt"]))
    every = max(1, nsteps // c["n_frames"])
    errs, sig = [], []

    def cb(t, p):
        sx, sy = rms_widths(eng, p)
        ref = R.free_gaussian_sigma(c["sigma0"], P.t_er_to_code(t))
        errs.append(max(abs(sx / ref - 1), abs(sy / ref - 1)))
        sig.append(sx)

    hist = new_history()
    evolve_recording(eng, psi, t_er(c["dt"]), nsteps, every, hist, callback=cb)
    tag = "A3 free expansion"
    suite.add(
        GateResult(
            f"{tag}: sigma(t) vs free_gaussian_sigma",
            max(errs) < 1e-4,
            f"max rel err {max(errs):.3e} over {len(errs)} frames (sigma {sig[0]:.4f} -> {sig[-1]:.4f}, "
            f"x{sig[-1] / sig[0]:.3f}; tol 1e-4)",
            "analytic",
            max(errs),
            0.0,
            1e-4,
        )
    )
    suite.add(
        GateResult(
            f"{tag}: growth >= 2x",
            sig[-1] >= 2.0 * sig[0],
            f"sigma(t_end)/sigma0 = {sig[-1] / sig[0]:.4f} (reference {c['sigma_end'] / c['sigma0']:.4f})",
            "analytic",
            sig[-1] / sig[0],
            c["sigma_end"] / c["sigma0"],
            None,
        )
    )
    # one component: no N2/N gate (N2_over_N is written as 0.0 every frame, a gate on it cannot fail)
    for g in tagged(H.conservation_gates(hist, tol_norm=1e-9, tol_energy=1e-7), tag):
        suite.add(g)
    return dict(max_err=max(errs))


def _single_lattice_ground(grid, V, code_tol: float = 1e-12):
    eng = make_engine(grid, V, C=1)
    r = gs.relax(eng, gs.initial_state(eng, 1.0), 1.0, tol=res_tol(code_tol))
    terms = eng.energy_terms(r.psi)
    return eng, r, float(terms["E_total"][0] / terms["norm"][0]), float(terms["mu"][0])


def gate_A4_band_bottom(suite: GateSuite, ctx: Context) -> dict:
    """[analytic] q = 0 band bottom of V0 sin^2 at 4 E_R: 1D, separable 2D (= 2 x 1D, 3.089723 E_R)
    and the sqrt(485) supercell with lattice 1 only at 8 pts/a vs the anchor."""
    c = lattice_1d_case()
    E1 = c["E_1d"] / ER  # E_R
    tag = "A4 band bottom"
    g1 = Grid((c["Nx"],), (float(c["n_cells"]),))
    _, r1, EN1, mu1 = _single_lattice_ground(g1, lat.lattice_1d(g1, c["V0_ER"]))
    suite.add(value_gate(f"{tag}: 1D V0 sin^2, E/N vs plane waves", EN1, E1, 1e-6, unit=" E_R"))
    suite.add(value_gate(f"{tag}: 1D mu = E/N (linear)", mu1, EN1, 1e-10))
    for g in residual_gate(f"{tag} 1D", float(r1.residual[0]), 1e-10):
        suite.add(g)
    g2 = Grid((c["Nx"], c["Nx"]), (float(c["n_cells"]),) * 2)
    eng2, r2, EN2, _ = _single_lattice_ground(g2, lat.square_lattice(g2, c["V0_ER"], 0.0))
    suite.add(value_gate(f"{tag}: 2D square lattice, E/N vs 2 x 1D", EN2, 2.0 * E1, 2e-6))
    v, tol, _ = R.anchor("supercell_EN_4ER_g0_J0_ER")
    suite.add(value_gate(f"{tag}: 2D square lattice, E/N vs 3.089723 anchor", EN2, v, tol, kind="anchor", unit=" E_R"))
    for g in residual_gate(f"{tag} 2D", float(r2.residual[0]), 1e-10):
        suite.add(g)
    gs, geo = supercell_grid(176)
    t0 = time.perf_counter()
    engs, rs, ENs, _ = _single_lattice_ground(gs, lat.square_lattice(gs, c["V0_ER"], geo.alpha), code_tol=1e-11)
    ctx.timings["A4 supercell 176"] = time.perf_counter() - t0
    suite.add(
        value_gate(
            f"{tag}: sqrt(485) supercell lattice 1, 176^2, E/N vs anchor", ENs, v, tol, kind="anchor", unit=" E_R"
        )
    )
    suite.add(
        value_gate(f"{tag}: sqrt(485) supercell lattice 1, 176^2, E/N vs 2 x 1D plane waves", ENs, 2.0 * E1, 2e-6)
    )
    ctx.values.update(A4_E_1d=EN1, A4_E_2d=EN2, A4_E_supercell=ENs, A4_res_supercell=float(rs.residual[0]))
    return dict(case=c, sep2d=(eng2, r2.psi), supercell=(engs, rs.psi), E_1d=EN1, E_2d=EN2, E_sup=ENs)


def _k_bin_weight(eng, psi, kvecs) -> np.ndarray:
    """Fraction of the atoms of component 0 in the single k bins at the given (kx, ky)."""
    nk = backend.to_numpy(eng.momentum_distribution(psi))[0, 0]  # fftshifted, sum * dk^2 = N
    dk = [2.0 * math.pi / L for L in eng.grid.L]
    N = float(nk.sum() * dk[0] * dk[1])
    out = []
    for kx, ky in kvecs:
        ix = eng.grid.N[0] // 2 + int(round(kx / dk[0]))
        iy = eng.grid.N[1] // 2 + int(round(ky / dk[1]))
        out.append(nk[ix, iy] * dk[0] * dk[1] / N)
    return np.array(out)


def gate_A5_bloch_weights(suite: GateSuite, ctx: Context, a4: dict | None = None) -> dict:
    """[analytic] plane-wave weights of the 2D q = 0 Bloch state: |c0|^4 = 0.8205, |c0 c1|^2 = 0.0425,
    |c1|^4 = 0.0022; on the separable lattice and on the rotated sqrt(485) supercell."""
    if a4 is None:
        a4 = gate_A4_band_bottom(GateSuite("engine", "scratch"), ctx)
    w = a4["case"]["weights_2d"]
    tag = "A5 Bloch weights"
    eng2, psi2 = a4["sep2d"]
    q = 2.0 * math.pi
    w0 = _k_bin_weight(eng2, psi2, [(0.0, 0.0)])[0]
    w10 = _k_bin_weight(eng2, psi2, [(q, 0), (-q, 0), (0, q), (0, -q)])
    w11 = _k_bin_weight(eng2, psi2, [(q, q), (-q, q), (q, -q), (-q, -q)])
    suite.add(value_gate(f"{tag}: square lattice k = 0 weight vs |c0|^4", w0, w["w0"], 1e-3))
    suite.add(
        value_gate(
            f"{tag}: square lattice primary peak (worst of 4) vs |c0 c1|^2",
            w10[np.argmax(np.abs(w10 - w["w10"]))],
            w["w10"],
            1e-3,
        )
    )
    suite.add(
        value_gate(
            f"{tag}: square lattice diagonal peak (worst of 4) vs |c1|^4",
            w11[np.argmax(np.abs(w11 - w["w11"]))],
            w["w11"],
            1e-3,
        )
    )
    v, tol, _ = R.anchor("franck_condon_4ER")
    suite.add(value_gate(f"{tag}: Franck-Condon <F> = |c0|^4 vs 0.8205 anchor", w0, v, tol, kind="anchor"))
    engs, psis = a4["supercell"]
    G1 = lattice1_reciprocal()  # lattice 1 (paper labels), 1/a
    ws0 = _k_bin_weight(engs, psis, [(0.0, 0.0)])[0]
    ws1 = _k_bin_weight(engs, psis, [tuple(g) for g in np.vstack([G1, -G1])])
    suite.add(value_gate(f"{tag}: supercell 176^2 k = 0 weight vs |c0|^4", ws0, w["w0"], 1e-3))
    suite.add(
        value_gate(
            f"{tag}: supercell 176^2 lattice-1 primary peak (worst of 4) vs |c0 c1|^2",
            ws1[np.argmax(np.abs(ws1 - w["w10"]))],
            w["w10"],
            1e-3,
        )
    )
    ctx.values.update(A5_w0=float(w0), A5_w10=[float(x) for x in w10], A5_w0_supercell=float(ws0))
    return dict(w0=w0, w10=w10, w11=w11)


def gate_A6_rabi(suite: GateSuite, ctx: Context) -> dict:
    """[analytic] two-level Rabi, V = g = 0, uniform: N2/N = (J^2/W^2) sin^2(W t), at delta = 0 and
    delta != 0; and the detuning sign: |1> sits at +delta/2 (E/N = +delta/2 at t = 0)."""
    out = {}
    for delta in (0.0, 0.15):
        c = rabi_case(delta=delta)
        grid = Grid((c["N"], c["N"]), (c["L"], c["L"]))
        eng = make_engine(grid, None, J=er(c["J"]), delta=er(c["delta"]))
        psi = field_from(grid, np.full(grid.N, 1.0 / math.sqrt(grid.volume)), np.zeros(grid.N))
        nsteps = int(round(c["t_end"] / c["dt"]))
        hist = new_history()
        evolve_recording(eng, psi, t_er(c["dt"]), nsteps, c["every"], hist)
        t_code = [P.t_er_to_code(t) for t in hist["t"]]
        ref = R.rabi_population(c["J"], t_code, delta=c["delta"])
        err = float(np.max(np.abs(np.asarray(hist["N2_over_N"]) - ref)))
        tag = f"A6 Rabi J={c['J']:g} delta={delta:g} (code)"
        suite.add(
            GateResult(
                f"{tag}: N2/N(t) vs rabi_population",
                err < 1e-8,
                f"max abs err {err:.3e} over {len(ref)} frames (tol 1e-8; max N2/N {max(hist['N2_over_N']):.6f})",
                "analytic",
                err,
                0.0,
                1e-8,
            )
        )
        if delta == 0.0:
            hist.pop("E_total")  # E = 0 for |1> at delta = 0: no relative drift gate
        else:
            E0 = hist["E_total"][0]
            suite.add(
                value_gate(
                    f"{tag}: E/N at t = 0 vs +delta/2 (|1> at +delta/2)",
                    E0,
                    0.5 * er(c["delta"]),
                    1e-12,
                    kind="identity",
                )
            )
        for g in tagged(H.conservation_gates(hist, tol_norm=1e-12, tol_energy=1e-10), tag):
            suite.add(g)
        out[delta] = err
    ctx.values["A6_max_err"] = {str(k): v for k, v in out.items()}
    return out


def gate_B7_gauge_copy(suite: GateSuite, ctx: Context) -> dict:
    """[identity] H(-J) = sz H(J) sz: the engine's own +J and -J starts are sz copies (relative phase
    pi / 0); the relaxed states have the same energy and total density (fixed g, nbar = 1, 88^2)."""
    c = twisted_cell_case(ppa=4)
    J = er(c["J"])
    eng_p, r_p = ctx.ground(c["N"], J, c["nbar"], "fixed")
    eng_m, r_m = ctx.ground(c["N"], -J, c["nbar"], "fixed")
    tp, tm = eng_p.energy_terms(r_p.psi), eng_m.energy_terms(r_m.psi)
    na, nb = backend.to_numpy(eng_p.density(r_p.psi))[0, 0], backend.to_numpy(eng_m.density(r_m.psi))[0, 0]
    dn = float(np.abs(na - nb).max() / na.max())
    tag = "B7 gauge copy H(-J) = sz H(J) sz"
    suite.add(
        value_gate(
            f"{tag}: E_total(-J) vs E_total(+J)",
            float(tm["E_total"][0]),
            float(tp["E_total"][0]),
            1e-8,
            kind="identity",
        )
    )
    suite.add(
        GateResult(f"{tag}: max |n(-J) - n(+J)| / n_max", dn < 1e-6, f"{dn:.3e} (tol 1e-6)", "identity", dn, 0.0, 1e-6)
    )
    suite.add(
        value_gate(
            f"{tag}: E_rabi(-J) = E_rabi(+J) (sign absorbed by psi_2 -> -psi_2)",
            float(tm["E_rabi"][0]),
            float(tp["E_rabi"][0]),
            1e-8,
            kind="identity",
        )
    )
    for g in residual_gate(f"{tag} (+J)", float(r_p.residual[0]), 1e-8) + residual_gate(
        f"{tag} (-J)", float(r_m.residual[0]), 1e-8
    ):
        suite.add(g)
    ctx.values.update(B7_dn=dn)
    return dict(dn=dn)


def _perturbed_ground(ctx: Context, c: dict, vertical: str = "npse", phase_amp: float = 0.3):
    """The polished ground state with a long-wavelength phase imprint exp(i phase_amp cos(2 pi x / L))."""
    eng, r = ctx.ground(c["N"], er(c["J"]), c["nbar"], vertical)
    X, _ = mesh(eng.grid)
    ph = torch.as_tensor(np.exp(1j * phase_amp * np.cos(2.0 * math.pi * X / L485)))
    return r.psi * ph.to(r.psi.device)


def gate_B8_N2_conservation(suite: GateSuite, ctx: Context) -> dict:
    """[identity] J = 0, SU(2) NPSE: N2/N of a non-stationary state is exactly conserved."""
    c = twisted_cell_case(ppa=4)
    psi = _perturbed_ground(ctx, c)
    eng0 = ctx.supercell(c["N"], 0.0, 0.0, "npse")
    dt = t_er(2e-3)
    hist = new_history()
    evolve_recording(eng0, psi, dt, 400, 20, hist)
    tag = "B8 J = 0 real time (NPSE, 88^2)"
    for g in tagged(H.conservation_gates(hist, tol_norm=1e-9, tol_energy=1e-5, J_perp=0.0, tol_N2=1e-10), tag):
        suite.add(g)
    ph = eng0.kappa * eng0.grid.k_sq_max * dt
    suite.add(
        GateResult(
            f"{tag}: kinetic phase per step", ph < math.pi, f"{ph:.3f} rad (< pi)", "preflight", ph, math.pi, math.pi
        )
    )
    return dict(history=hist)


def gate_B9_c4_isotropy(suite: GateSuite, ctx: Context) -> dict:
    """[identity] C4 isotropy of the twist superfluid fraction on the pinned 176^2 grid (fixed g,
    nbar = 1, theta = 0.2): f_s^xx = f_s^yy, energy route = current route, 0 < f_s <= 1, the 0.6128
    anchor, and zero ground-state momentum (the 4 pts/a Nyquist artefact gives f_s = -0.22)."""
    N, theta = 176, 0.2
    c = twisted_cell_case(ppa=8)
    J = er(c["J"])
    eng0, r0 = ctx.ground(N, J, c["nbar"], "fixed")
    E0 = float(r0.energy[0])
    Nat = c["N_atoms"]
    P0 = math.hypot(*momentum_per_atom(eng0, r0.psi))
    fsE, fsI, res = {}, {}, {}
    for ax, name in ((0, "x"), (1, "y")):
        tw = [0.0, 0.0]
        tw[ax] = theta
        eng = ctx.supercell(N, J, G2D_ER, "fixed", twist=tw)
        t0 = time.perf_counter()
        r = gs.relax(eng, r0.psi.clone(), Nat, tol=res_tol(1e-9), dt_schedule=[])
        ctx.timings[f"B9 twist {name}"] = time.perf_counter() - t0
        fsE[name] = float(
            obs.superfluid_fraction_energy(r.energy, r0.energy, eng.norm(r.psi), L485, theta, eng.kappa)[0]
        )
        fsI[name] = float(obs.superfluid_fraction_current(eng, r.psi, ax, theta)[0])
        res[name] = float(r.residual[0])
        ctx.log(f"twist {name}: f_s energy {fsE[name]:.6f} current {fsI[name]:.6f} r={res[name]:.1e}")
    tag = f"B9 twist f_s ({N}^2, theta = {theta:g}, fixed g, nbar = 1)"
    suite.add(
        GateResult(
            f"{tag}: ground state carries no momentum",
            P0 < 1e-6,
            f"|P|/N = {P0:.3e} hbar/a (tol 1e-6)",
            "preflight",
            P0,
            0.0,
            1e-6,
        )
    )
    suite.add(value_gate(f"{tag}: C4 isotropy f_s^xx vs f_s^yy (energy)", fsE["x"], fsE["y"], 1e-4, kind="identity"))
    suite.add(value_gate(f"{tag}: energy route vs current route (x)", fsE["x"], fsI["x"], 2e-4, kind="identity"))
    f_mean = 0.5 * (fsE["x"] + fsE["y"])
    suite.add(
        GateResult(
            f"{tag}: 0 < f_s <= 1", 0.0 < f_mean <= 1.0 + 1e-12, f"f_s = {f_mean:.6f}", "identity", f_mean, None, None
        )
    )
    v, tol, _ = R.anchor("fs_twist_J0.243_n1_fixedg")
    suite.add(value_gate(f"{tag}: f_s vs 0.6128 anchor", f_mean, v, tol, kind="anchor"))
    suite.add(
        GateResult(
            f"{tag}: twisted states polished",
            max(res.values()) < res_tol(1e-8),
            f"residuals x {res['x']:.1e}, y {res['y']:.1e} E_R (target {res_tol(1e-8):.1e})",
            "convergence",
            max(res.values()),
            0.0,
            res_tol(1e-8),
        )
    )
    for g in residual_gate(f"{tag} untwisted", float(r0.residual[0]), 1e-8):
        suite.add(g)
    ctx.values.update(B9_fs_energy=fsE, B9_fs_current=fsI, B9_P0=P0, B9_E0_per_N=E0 / Nat)
    return dict(fsE=fsE, fsI=fsI, P0=P0)


def gate_B10_reversibility(suite: GateSuite, ctx: Context) -> dict:
    """[identity] exact SU(2) step: n steps of dt then n of -dt return the start to rounding."""
    c = twisted_cell_case(ppa=4)
    eng = ctx.supercell(c["N"], er(c["J"]), 0.0, "npse")
    psi0 = _perturbed_ground(ctx, c)
    dt, nsteps = t_er(2e-3), 400
    hist = new_history()
    psi_f = evolve_recording(eng, psi0.clone(), dt, nsteps, 50, hist)
    psi_b = eng.run_block(psi_f.clone(), nsteps, -dt)
    a, b, f = backend.to_numpy(psi0), backend.to_numpy(psi_b), backend.to_numpy(psi_f)
    dpsi = float(np.max(np.abs(a - b)) / np.max(np.abs(a)))
    moved = float(np.max(np.abs(a - f)) / np.max(np.abs(a)))
    tag = "B10 reversibility dt -> -dt (NPSE, 88^2)"
    suite.add(
        GateResult(
            f"{tag}: max |psi_back - psi_0| / max |psi_0|",
            dpsi < 1e-9,
            f"{dpsi:.3e} after {nsteps} + {nsteps} steps (tol 1e-9; forward state moved by {moved:.3f})",
            "identity",
            dpsi,
            0.0,
            1e-9,
        )
    )
    suite.add(
        GateResult(
            f"{tag}: forward evolution non-trivial",
            moved > 1e-2,
            f"max |psi(T) - psi_0| / max |psi_0| = {moved:.3f} (> 1e-2)",
            "identity",
            moved,
            0.0,
            None,
        )
    )
    for g in tagged(H.conservation_gates(hist, tol_norm=1e-9, tol_energy=1e-5), tag):
        suite.add(g)
    ctx.values["B10_dpsi"] = dpsi
    return dict(dpsi=dpsi)


def gate_C11_band_bottom(suite: GateSuite, ctx: Context) -> dict:
    """[anchor] dressed band bottom E0(J) at g = 0 on the pinned 176^2 grid: 14.17705565 code =
    2.872872 E_R at J = 1.19802 code (0.12138 E_R in --full too), one anchor gate and the residual per J.
    parameters.E0_BAND_BOTTOM holds the same literals and the E_R anchor is the code anchor over E_R_CODE;
    tests/test_gates.py checks both (as gates they compared one number with itself)."""
    out = {}
    Js = [("dressed_band_bottom_J0.2428_code", J_PROD_CODE)]
    if not ctx.quick:
        Js.append(("dressed_band_bottom_J0.1214_code", P.J_PERP_CODE[0.121]))
    for name, Jc in Js:
        t0 = time.perf_counter()
        eng, r = ctx.ground(176, er(Jc), 1.0, "fixed", g=0.0, code_tol=1e-11)
        ctx.timings[f"C11 J={Jc:g}"] = time.perf_counter() - t0
        E0 = float(r.mu[0])
        tag = f"C11 dressed band bottom J={Jc:g} code, g = 0, 176^2"
        v, tol, _ = R.anchor(name)
        suite.add(value_gate(f"{tag}: E0 x E_R_CODE vs anchor", E0 * ER, v, tol, kind="anchor"))
        for g in residual_gate(tag, float(r.residual[0]), 1e-10):
            suite.add(g)
        out[Jc] = E0
    ctx.values["C11_E0_ER"] = {str(k): v for k, v in out.items()}
    return out


def _contrast(eng, psi) -> dict:
    """C_psf / C_raw of the engine's state (imaging.standard_observables, in-focus pupil)."""
    from twisted_bilayer.engine import imaging

    n = backend.to_numpy(psi[0].abs() ** 2)
    return imaging.standard_observables(n[0], n[1], eng.grid.dx[0], imaging="pupil")


def gate_C12_C13_mu_contrast(suite: GateSuite, ctx: Context) -> dict:
    """[anchor] mu and C_psf at (J = 0.243 E_R, nbar = 1), NPSE lambda = 1 and fixed g, with the
    anchors' g2D (5.29 nm); NPSE also with the engine's own g2D (100 a0) against its pinned 2.908446.
    quick: 88^2 with loose tolerances (mu 5e-4 rel, C_psf 1e-2 abs); --full: 176^2 pinned."""
    N = 88 if ctx.quick else 176
    c = twisted_cell_case(ppa=N // 22)
    J = er(c["J"])
    note = "" if N == 176 else " [4 pts/a; the pinned grid is 8 pts/a = 176^2, run with --full]"
    g_native = units.lattice_units().g2d
    out = {}
    cases = (
        ("npse", G2D_ER, "mu_J0.243_n1_NPSE_lam1_ER", "C_psf_J0.243_n1_NPSE_lam1"),
        ("fixed", G2D_ER, "mu_J0.243_n1_fixedg_ER", "C_psf_J0.243_n1_fixedg"),
        ("npse", g_native, "mu_J0.243_n1_NPSE_lam1_engine_native_ER", None),
    )
    for vertical, g, mu_anchor, c_anchor in cases:
        eng, r = ctx.ground(N, J, c["nbar"], vertical, g=g)
        mu = float(r.mu[0])
        which = "engine g2D (a_s = 100 a0)" if g == g_native else "g2D 5.29 nm"
        tag = f"C12 mu (0.243, nbar = 1) {vertical}{' lambda = 1' if vertical == 'npse' else ' g'}, {which}, {N}^2"
        v, tol, _ = R.anchor(mu_anchor)
        gm = value_gate(f"{tag}: mu vs anchor", mu, v, tol if N == 176 else 5e-4, kind="anchor", unit=" E_R")
        gm.detail += note
        suite.add(gm)
        suite.add(
            GateResult(
                f"{tag}: ladder converged and polished",
                bool(r.converged[0]) and float(r.residual[0]) < res_tol(1e-8),
                f"converged={bool(r.converged[0])}, residual {float(r.residual[0]):.1e} E_R "
                f"(target {res_tol(1e-8):.1e})",
                "convergence",
                float(r.residual[0]),
                0.0,
                res_tol(1e-8),
            )
        )
        con = _contrast(eng, r.psi)
        if c_anchor is not None:
            v, tol, _ = R.anchor(c_anchor)
            gc = value_gate(
                f"C13 C_psf (0.243, nbar = 1) {vertical} {N}^2: C_psf vs anchor",
                con["C_psf"],
                v,
                tol if N == 176 else 1e-2,
                relative=False,
                kind="anchor",
            )
            gc.detail += note
            suite.add(gc)
        out[f"{vertical}_{'native' if g == g_native else 'parameters'}"] = dict(
            mu_ER=mu, C_psf=con["C_psf"], C_raw=con["C_raw"], N=N, g=g
        )
    d_mu = out["npse_native"]["mu_ER"] - out["npse_parameters"]["mu_ER"]
    ctx.values["C12_C13"] = out
    ctx.values["C12_dmu_a_s"] = dict(
        dmu_ER=d_mu, g_ratio=g_native / G2D_ER, note="mu(NPSE, a_s = 100 a0) - mu(NPSE, a_s = 5.29 nm) at the same grid"
    )
    ctx.log(f"a_s effect on mu (NPSE, {N}^2): {d_mu:+.3e} E_R (g ratio {g_native / G2D_ER:.6f})")
    return out


def choose_dt_values(eng, psi) -> dict:
    """The real-time step that `choose_dt` picks for `eng` at order 2 and at order 4 with the `rabi`
    multiplier 20, and what it implies: dt (E_R), the kinetic phase per step kappa k2_max dt in E_R units and
    k2_max dt / 2 in code units (`references.kinetic_phase_per_step`; the same number), the largest Yoshida
    sub-step theta_Y dt of order 4, and the grid's Nyquist wavenumber against twice the lattice's 2 pi.
    These follow from choose_dt's constants (the outer phase is capped at 1.5 rad), not from a run, so
    tests/test_gates.py checks them (phase and sub-step < 2.6 rad, the identity to 1e-12, k_Nyq > 4 pi) and
    gate D only records them."""
    dx = eng.grid.dx
    theta_y = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    out = dict(k_nyquist=min(math.pi / d for d in dx), k_lattice=2.0 * math.pi)
    for order, mult in ((2, 1.0), (4, 20.0)):
        dt = phys.choose_dt(eng, psi, order=order, order4_multiplier=mult)
        ph = eng.kappa * eng.grid.k_sq_max * dt
        out[f"order{order}_dt_ER"], out[f"order{order}_phase"] = dt, ph
        out[f"order{order}_phase_code"] = R.kinetic_phase_per_step(None, P.t_er_to_code(dt), dx)
        if order == 4:
            out["order4_substep_phase"] = theta_y * ph
    return out


def gate_D_preflight(suite: GateSuite, ctx: Context) -> dict:
    """[preflight] The NPSE parameter kappa = g2D n_peak / hbar omega_z at nbar = 1 (gate) and nbar = 5
    (expected-FAIL demonstration in meta), and, recorded in values only, the engine's own real-time step on
    the production supercell (176^2): `choose_dt_values` at order 2 and at order 4 with the `rabi` multiplier
    20. They test choose_dt's constants, not the run, so they are unit tests (tests/test_gates.py), not
    gates."""
    c = twisted_cell_case(ppa=8)
    eng, r = ctx.ground(176, er(c["J"]), 1.0, "fixed")
    dx = eng.grid.dx
    out = choose_dt_values(eng, r.psi)
    demo = {}
    for nbar in (1.0, 5.0):
        e, rn = ctx.ground(88, er(c["J"]), nbar, "npse")
        n_peak = float(backend.to_numpy(e.density(rn.psi)).max())
        kappa = G2D_ER * n_peak / WZ_ER
        g = H.preflight_gates(dx, 1.0, kappa=kappa)[-1]
        g.name = f"D NPSE validity nbar = {nbar:g}, J = {c['J']:g} code: {g.name}"
        g.detail += f"; n_peak = {n_peak:.3f} atoms/a^2 (88^2, lambda = 1)"
        out[f"kappa_nbar{nbar:g}"], out[f"n_peak_nbar{nbar:g}"] = kappa, n_peak
        if nbar == 1.0:
            suite.add(g)
        else:
            demo = dict(
                name=g.name,
                passed=g.passed,
                detail=g.detail + " -- expected FAIL: model-validity flag (NPSE outside its validated range)",
            )
    suite.meta["preflight_demo_expected_FAIL"] = demo
    ctx.values["D"] = out
    return out


def gate_E_trapped(suite: GateSuite, ctx: Context) -> dict:
    """[anchor] trapped cloud: N = 2000, J = 0.24277 E_R, NPSE lambda = 1, 20 Hz, 3 x 3 cells at
    8 pts/a, g2D = parameters.G2D_ER: C_raw, C_coh and R_rms vs the anchors from an independent
    quasi-2D split-step solver (its mu and E/N are recorded in meta)."""
    from twisted_bilayer.engine import trapped as TR

    spec = TR.TrappedSpec(g2d=G2D_ER, N_atoms=2000.0, nu_r_hz=20.0, box_cells=3, cell_points=176)
    t0 = time.perf_counter()
    run = TR.trapped_ground_state(spec, verbose=ctx.verbose)
    res = TR.summary(run, spec, s_sat=0.01)
    ctx.timings["E trapped"] = time.perf_counter() - t0
    tag = "E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2"
    ins, cl, gr = res["insitu"], res["cloud"], res["ground"]
    for key, val, rel in (
        ("C_raw", ins["C_raw"], False),
        ("C_coh", ins["C_coh"], False),
        ("R_rms_a", cl["R_rms"], True),
    ):
        v, tol, _ = R.anchor(f"trapped_N2000_J0.243_lam1_{key}")
        suite.add(value_gate(f"{tag}: {key} vs anchor", val, v, tol, relative=rel, kind="anchor"))
    suite.add(
        GateResult(
            f"{tag}: polish converged",
            gr["converged"] and gr["residual"] < res_tol(1e-9),
            f"residual {gr['residual']:.1e} E_R (target {res_tol(1e-9):.1e}), {gr['steps']} steps",
            "convergence",
            gr["residual"],
            0.0,
            res_tol(1e-9),
        )
    )
    suite.add(value_gate(f"{tag}: N", cl["N"], 2000.0, 1e-10, kind="conservation"))
    suite.add(
        value_gate(f"{tag}: N2/N = 1/2 (SU(2), delta = 0)", cl["N2_over_N"], 0.5, 1e-8, relative=False, kind="identity")
    )
    suite.add(
        GateResult(
            f"{tag}: cloud inside the box",
            cl["edge_density_rel"] < 1e-3,
            f"edge density / peak = {cl['edge_density_rel']:.1e} (< 1e-3)",
            "preflight",
            cl["edge_density_rel"],
            0.0,
            1e-3,
        )
    )
    # the two keys below: the same cloud's mu and E/N from the independent quasi-2D split-step solver, for comparison
    ctx.values["E_trapped"] = dict(
        insitu=ins,
        cloud={k: v for k, v in cl.items() if k != "atoms_per_tile"},
        ground=gr,
        lda={k: v for k, v in res["lda"].items() if k in ("mu_lda", "f_centre", "R_lda", "E_per_N", "mu")},
        reference_mu_ER=P.code_to_er(14.459569980887892),
        reference_E_per_N_ER=P.code_to_er(14.37977056728955),
    )
    return res


# ======================================================================== driver
SCENARIOS = {
    "A1": gate_A1_ho_ground_state,
    "A2": gate_A2_ho_breathing,
    "A3": gate_A3_free_expansion,
    "A4": gate_A4_band_bottom,
    "A5": gate_A5_bloch_weights,
    "A6": gate_A6_rabi,
    "B7": gate_B7_gauge_copy,
    "B8": gate_B8_N2_conservation,
    "B9": gate_B9_c4_isotropy,
    "B10": gate_B10_reversibility,
    "C11": gate_C11_band_bottom,
    "C12": gate_C12_C13_mu_contrast,
    "C13": gate_C12_C13_mu_contrast,
    "D": gate_D_preflight,
    "E": gate_E_trapped,
}


def run_all(
    out_dir=None,
    quick: bool = True,
    only=None,
    verbose: bool = True,
    device: str | None = None,
    threads: int | None = None,
    trapped: bool = False,
) -> GateSuite:
    """Run the gate scenarios (all, or the ids in `only`; the trapped scenario E only when trapped=True
    or named in `only`) on the engine; writes <out_dir>/gates[_cuda].json when out_dir is given.
    Returns the GateSuite."""
    if device is None:  # the device the engine was imported on
        device = "cuda" if backend.HAS_GPU else "cpu"
    if (device == "cuda") != backend.HAS_GPU:
        raise RuntimeError(
            f"device {device!r} requested, but the engine was imported on {backend.DEVICE} "
            "(TWISTED_BILAYER_FORCE_CPU, or no CUDA build of torch)"
        )
    if threads:
        torch.set_num_threads(int(threads))
    every = ALL_IDS + TRAPPED_IDS  # E (trapped) on request
    if only is not None:
        unknown = sorted(set(only) - set(every))
        if unknown or not only:
            what = f"unknown scenario id(s) {', '.join(unknown)}" if unknown else "no scenario id"
            raise ValueError(f"--only: {what}; valid ids: {', '.join(every)}")
    ids = (
        (list(ALL_IDS) + (list(TRAPPED_IDS) if trapped else []))
        if only is None
        else [i for i in every if i in set(only)]
    )
    suite = GateSuite(
        "engine", "physics gates " + ("(quick: C12/C13 at 4 pts/a)" if quick else "(full: 8 pts/a anchors)")
    )
    suite.meta.update(
        device=backend.describe_device(),
        quick=quick,
        scenarios=ids,
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        platform=platform.platform(),
        python=platform.python_version(),
        g2D_ER=G2D_ER,
        hbar_omega_z_ER=WZ_ER,
        J_prod_ER=er(J_PROD_CODE),
        units_exceptions={k: v[3] for k, v in units.EXCEPTIONS.items()},
    )
    ctx = Context(quick, verbose)
    results, done = {}, set()
    for sid in ids:
        fn = SCENARIOS[sid]
        if fn in done:
            continue
        done.add(fn)
        t0 = time.perf_counter()
        if verbose:
            print(f"-- scenario {sid}: {fn.__doc__.splitlines()[0].strip()}", file=sys.stderr, flush=True)
        if sid == "A5" and "A4" in results:
            results[sid] = fn(suite, ctx, results["A4"])
        else:
            results[sid] = fn(suite, ctx)
        ctx.timings[f"scenario {sid}"] = time.perf_counter() - t0
    suite.meta["timings_s"] = {k: round(v, 2) for k, v in ctx.timings.items()}
    suite.meta["values"] = ctx.values
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suite.to_json(str(out_dir / f"gates{'' if device == 'cpu' else '_' + device}.json"))
    return suite


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Physics gates of the twisted-bilayer engine")
    ap.add_argument("--out", default=None, help="directory for gates.json")
    ap.add_argument("--full", action="store_true", help="C12/C13 anchors on the pinned 176^2 grid")
    ap.add_argument("--trapped", action="store_true", help="add scenario E, the N = 2000 trapped cloud (~10 min CPU)")
    ap.add_argument("--only", default=None, help="comma-separated scenario ids, e.g. A1,A3,B8")
    ap.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="default: the device the engine was imported on (`twisted-bilayer gates` imports it on the CPU "
        "unless --device cuda)",
    )
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    only = None if a.only is None else {s.strip() for s in a.only.split(",") if s.strip()}
    device = a.device or ("cuda" if backend.HAS_GPU else "cpu")
    t0 = time.perf_counter()
    try:
        suite = run_all(
            a.out,
            quick=not a.full,
            only=only,
            verbose=not a.quiet,
            device=device,
            threads=a.threads or os.cpu_count(),
            trapped=a.trapped,
        )
    except ValueError as e:  # an unknown --only id: a usage error (exit code 2), not an empty pass
        ap.error(str(e))
    print(suite.report())
    demo = suite.meta.get("preflight_demo_expected_FAIL")
    if demo:
        print(
            f"-- demonstration (not counted): [{'PASS' if demo['passed'] else 'FAIL'}] {demo['name']}: {demo['detail']}"
        )
    print(f"({len(suite.results)} gates, {time.perf_counter() - t0:.0f} s, {suite.meta['device']})")
    if a.out:
        print(f"wrote {pathlib.Path(a.out) / ('gates' + ('' if device == 'cpu' else '_' + device) + '.json')}")
    return 0 if suite.passed else 1


if __name__ == "__main__":
    sys.exit(main())
