"""Trapped clouds ("geometry B"): a harmonic trap on a box of `box_cells` x `box_cells` sqrt(485)
supercells, the LDA start built from the engine's own supercell scan, the polished trapped ground
state, and the cloud numbers. An independent quasi-2D split-step solver with the same method gives the
trapped clouds behind a simulated Fig. 4 of Meng et al., Nature 615, 231 (2023):

  supercell_scan        the supercell ground states at the scan fillings (NPSE lambda = 1), computed here
  LDA                   nbar(mu) by PCHIP through the scan, 0 below E0, linear above
  lda_cloud             the scan cells tiled under the LDA envelope, phase-aligned (q = 0 of component 1
                        real positive), in the engine's [ix, iy] layout
  add_noise             seeded 1 % real amplitude noise of 2 a correlation length
  containment           the Thomas-Fermi radius of the LDA cloud in the infinite plane against the box
  resolve_box           refuse (default), enlarge or only warn when the cloud is wider than its box
  trapped_ground_state  LDA start -> heavy-ball polish only (a Strang ladder would first move away from it)
  cloud_numbers         R_rms, central-cell nbar, atoms per tile, edge density
  insitu_observables    C_raw, C_psf, C_coh at s = 0 and s = 0.01, n_AA, n_AB (`imaging`; the central
                        cell is the AA cell at the origin)
  checks                contained (edge density / peak < CONTAIN_EDGE) and converged, for every summary

Containment. The box is periodic: a cloud wider than the box overlaps its own periodic images, and every
observable then describes a lattice of overlapping clouds. Before the grid is allocated, `containment`
solves the LDA in the infinite plane, N = (2 pi / (m omega_r^2)) int_E0^mu f(u) du, and compares the
Thomas-Fermi radius R_TF = sqrt(2 (mu - E0) / m) / omega_r with the half-width of the box (the distance to
the nearest edge). Above R_TF = CONTAIN_REFUSE x half-width the cloud's tail reaches the edge at more than
~1e-3 of the peak; `resolve_box` then refuses the run (spec.containment = "refuse"), enlarges the box to the
smallest odd size with R_TF <= CONTAIN_TARGET x half-width at the same N and nu_r ("enlarge"), or runs it with
a warning ("warn"). CONTAIN_REFUSE = 0.88 is calibrated on the experiment's cloud (N = 2000, 20 Hz, m = 22):
R_TF / half-width 0.847 gives edge 2.7e-4 (the validated 3 x 3 reference), 0.870 gives 1.4e-4, 0.912
gives 7.9e-3, 0.937 gives 1.3e-2 and 1.0 gives 0.13. These are clouds of different nu_r and nu_z, not one
cloud in growing boxes: the edge density depends on the shape of the tail as well as on the ratio, so the
list is not monotone below the line (0.870 sits under 0.847); above ~0.9 every one of them fails. A small
cloud (R_TF of a few a) has a relatively longer tail and can fail the edge test below that line; after the
solve, `checks` applies the edge criterion of the trapped-cloud gate (edge density / peak < 1e-3) to every
model, which catches it.

Units: lattice units (a = 1, E_R = 1, hbar = 1); omega_r from Hz with units.lattice_units(). Fields are
(B, C, Nx, Ny) torch tensors / (C, Nx, Ny) numpy arrays in grid order [ix, iy].
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from twisted_bilayer import parameters as _P

from . import imaging, units
from .backend import REAL_OF, complex_dtype, to_numpy
from .grid import Grid
from .ground_state import initial_state, relax, renormalise
from .lattice import MoireGeometry, bilayer_potential, harmonic_trap, moire_geometry
from .npse import NPSE2D
from .physics import BilayerGPE

FILLINGS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)  # scan fillings (atoms per a^2)
J_PROD = units.code_to_er(_P.J_PERP_CODE[0.243])  # 0.24277 E_R, the "0.243" reading of twisted_bilayer.config

CONTAIN_EDGE = 1e-3  # edge density / peak at or above which a solved cloud counts as not contained
CONTAIN_REFUSE = 0.88  # R_TF / box half-width above which resolve_box refuses (or enlarges): see the docstring
CONTAIN_TARGET = 0.8  # R_TF / box half-width that an enlarged box (and the suggested box) satisfies
CONTAINMENT_POLICIES = ("refuse", "enlarge", "warn")


class UncontainedCloud(ValueError):
    """The LDA cloud is wider than its periodic box (see the module docstring)."""


@dataclass
class TrappedSpec:
    """One trapped-cloud run (`trapped_ground_state`). Lengths in a, energies in E_R, frequencies in Hz.

    V0: depth of each layer's square lattice (E_R). Omega_R: interlayer coupling (E_R). N_atoms: atoms in the
    cloud (both components). nu_r_hz: radial trap frequency (Hz; isotropic in the plane). box_cells: the
    periodic box in supercells per side (odd, so that an AA point sits at the origin). cell_points: grid
    points per supercell side (176 = 8 per a for the sqrt(485) cell). vertical: "npse" (density-dependent
    width, `engine.npse`) or "fixed" (g_2D n). g2d: the 2D coupling (E_R a^2); None derives it from
    omega_z_hz and a_s = 100 a0 (`engine.units`). omega_z_hz: vertical trap frequency (Hz). fillings: atoms
    per a^2 of the supercell scan that builds the LDA start. noise, seed: amplitude and seed of the random
    perturbation added to the start (`add_noise`). tol: residual at which the polish stops (E_R). scan_tol:
    the same for the supercell scan. polish_steps: step budget of the polish (raise it per row for a slow
    cloud: a descent that cannot reach tol in it stops "slow"). m, n: the commensurate twist,
    tan(theta / 2) = n / m. dtype: "complex128" or "complex64" for the trapped solve (the scan is always
    complex128). containment: "refuse" | "enlarge" | "warn", what `resolve_box` does when the cloud is wider
    than the box."""

    V0: float = 4.0
    Omega_R: float = J_PROD
    N_atoms: float = 2000.0
    nu_r_hz: float = 20.0
    box_cells: int = 3  # odd: the central tile keeps the AA point at the origin
    cell_points: int = 176  # points per sqrt(485) supercell (176 = 8 pts/a, the anchors' grid)
    vertical: str = "npse"  # NPSE lambda = 1 (the default everywhere) or "fixed"
    g2d: float | None = None  # None: units.lattice_units().g2d (the engine's a_s = 100 a0)
    omega_z_hz: float = 1000.0
    fillings: tuple = FILLINGS
    noise: float = 0.01
    seed: int = 0
    tol: float = 2.0e-10  # E_R; = 1e-9 in code units
    scan_tol: float = 2.0e-10
    polish_steps: int = 20000
    m: int = 22
    n: int = 1
    dtype: str = "complex128"  # of the trapped solve; the supercell scan always runs in complex128
    containment: str = "refuse"  # "refuse" | "enlarge" | "warn": what resolve_box does with a too-wide cloud

    def couplings(self) -> tuple[float, float]:
        u = units.lattice_units(omega_z_hz=self.omega_z_hz)
        return (u.g2d if self.g2d is None else float(self.g2d)), u.omega_z

    def omega_r(self) -> float:
        return units.lattice_units().rad_s_to_ER(2.0 * math.pi * self.nu_r_hz)


def _engine(
    grid: Grid, V: torch.Tensor, spec: TrappedSpec, g: float | None = None, dtype: str | None = None
) -> BilayerGPE:
    g2d, wz = spec.couplings()
    g = g2d if g is None else g
    J = spec.Omega_R
    dt = complex_dtype(spec.dtype if dtype is None else dtype)
    grid = grid.to(REAL_OF[dt])
    if spec.vertical == "npse" and g > 0:
        return BilayerGPE(grid, V=V, g=0.0, Omega_R=J, dtype=dt, interaction=NPSE2D(g, wz))
    if spec.vertical not in ("npse", "fixed"):
        raise ValueError("vertical must be 'npse' or 'fixed'")
    return BilayerGPE(grid, V=V, g=g, Omega_R=J, dtype=dt)


def supercell_grid(spec: TrappedSpec) -> tuple[Grid, MoireGeometry]:
    geo = moire_geometry(spec.m, spec.n)
    return Grid((spec.cell_points,) * 2, (geo.L_box,) * 2), geo


# ----------------------------------------------------------------------------- the supercell scan
def supercell_scan(spec: TrappedSpec, verbose: bool = False) -> dict:
    """Polished supercell ground states at spec.fillings (and the dressed band bottom E0 at g = 0)
    on the trapped run's grid spacing: the input of the LDA start. Returns dict(cells = list of
    (2, N, N) complex128 [ix, iy], fillings, mus (E_R), E0 (E_R), residuals, steps, seconds)."""
    grid, geo = supercell_grid(spec)
    V = bilayer_potential(grid, spec.V0, geo)
    A = geo.L_box**2
    t0 = time.perf_counter()
    e0 = _engine(grid, V, spec, g=0.0, dtype="complex128")
    r0 = relax(e0, initial_state(e0, A), A, tol=spec.scan_tol)
    out = dict(
        fillings=[], mus=[], cells=[], residuals=[], steps=[], E0=float(r0.mu[0]), E0_residual=float(r0.residual[0])
    )
    eng = _engine(grid, V, spec, dtype="complex128")
    for f in spec.fillings:
        r = relax(eng, initial_state(eng, f * A), f * A, tol=spec.scan_tol)
        out["fillings"].append(float(f))
        out["mus"].append(float(r.mu[0]))
        out["cells"].append(to_numpy(r.psi[0]).astype(np.complex128))
        out["residuals"].append(float(r.residual[0]))
        out["steps"].append(int(r.steps))
        if verbose:
            print(
                f"  scan nbar={f:g}: mu={float(r.mu[0]):.9f} E_R res={float(r.residual[0]):.1e} steps={r.steps}",
                flush=True,
            )
    out["seconds"] = time.perf_counter() - t0
    if verbose:
        print(f"  band bottom E0 = {out['E0']:.9f} E_R; scan {out['seconds']:.0f} s", flush=True)
    return out


class LDA:
    """f(mu_loc): local filling of the homogeneous ground state at chemical potential mu_loc
    (inverse of the scan's mu(nbar), PCHIP; f = 0 below E0, linear extrapolation above the last
    scan point). Energies in E_R."""

    def __init__(self, E0: float, fillings, mus):
        m = np.concatenate([[E0], np.asarray(mus, float)])
        f = np.concatenate([[0.0], np.asarray(fillings, float)])
        if np.any(np.diff(m) <= 0):
            raise ValueError("mu(nbar) of the scan must increase with the filling")
        self.m, self.f = m, f
        self.ip = PchipInterpolator(m, f, extrapolate=False)
        self.slope = (f[-1] - f[-2]) / (m[-1] - m[-2])

    def __call__(self, mu_loc):
        mu_loc = np.asarray(mu_loc, float)
        out = np.zeros_like(mu_loc)
        inside = (mu_loc > self.m[0]) & (mu_loc <= self.m[-1])
        out[inside] = self.ip(mu_loc[inside])
        hi = mu_loc > self.m[-1]
        out[hi] = self.f[-1] + self.slope * (mu_loc[hi] - self.m[-1])
        return np.clip(out, 0.0, None)

    def solve_mu(self, VT: np.ndarray, dA: float, N_atoms: float) -> float:
        """mu with sum f(mu - VT) dA = N_atoms on the box grid, by bisection from [E0, E0 + 10]; the
        upper end is doubled (from E0) until it holds N_atoms. Raises ValueError if 60 doublings do not
        reach N_atoms (a box too small for the cloud at any mu)."""
        lo, hi = self.m[0], self.m[0] + 10.0
        for _ in range(60):
            if self(hi - VT).sum() * dA >= N_atoms:
                break
            hi = lo + 2.0 * (hi - lo)
        else:
            raise ValueError(f"LDA: N = {N_atoms:g} atoms not reached on this box for mu up to {hi:.3g} E_R")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self(mid - VT).sum() * dA < N_atoms:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def integral(self, mu: float) -> float:
        """int_E0^mu f(u) du (atoms per a^2 times E_R): PCHIP inside the scan, linear above."""
        m0, m1 = self.m[0], self.m[-1]
        if mu <= m0:
            return 0.0
        out = float(self.ip.integrate(m0, min(mu, m1)))
        if mu > m1:
            d = mu - m1
            out += self.f[-1] * d + 0.5 * self.slope * d * d
        return out

    def atoms_in_plane(self, mu: float, omega_r: float) -> float:
        """N(mu) of the LDA cloud in the infinite plane (no box): with u = mu - (1/2) m omega_r^2 r^2,
        N = int 2 pi r f(u) dr = (2 pi / (m omega_r^2)) int_E0^mu f(u) du."""
        return 2.0 * math.pi / (units.M_LATTICE * omega_r**2) * self.integral(mu)

    def solve_mu_plane(self, N_atoms: float, omega_r: float) -> float:
        """mu with atoms_in_plane(mu) = N_atoms (bisection; the bracket is doubled from E0 as needed)."""
        lo, hi = self.m[0], self.m[0] + 10.0
        for _ in range(60):
            if self.atoms_in_plane(hi, omega_r) >= N_atoms:
                break
            hi = lo + 2.0 * (hi - lo)
        else:
            raise ValueError(f"LDA: N = {N_atoms:g} atoms not reached for mu up to {hi:.3g} E_R")
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.atoms_in_plane(mid, omega_r) < N_atoms:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


def lda_cloud(cells, fillings, f_env: np.ndarray, n_tiles: int) -> np.ndarray:
    """The LDA start on the engine's layout: each (C, Nc, Nc) cell [ix, iy] is
    phase-aligned (q = 0 component of component 1 real positive), tiled n_tiles x n_tiles (odd:
    the box grid x_j = (j - N/2) dx then continues the cell grid), and
        psi(r) = sqrt(f) [(1 - w) psi_k / sqrt(f_k) + w psi_k+1 / sqrt(f_k+1)](r),
    f = f_env (the LDA filling on the box grid), f_k <= f < f_k+1, w linear, constant outside."""
    if n_tiles % 2 != 1:
        raise ValueError("n_tiles must be odd (an even number of tiles needs an L/2 roll, not implemented)")
    cs = []
    for c in cells:
        c = np.asarray(c, complex)
        ph = c[0].sum()
        cs.append(c * (np.conj(ph) / abs(ph) if abs(ph) > 0 else 1.0))
    f_k = np.asarray(fillings, float)
    f = np.clip(np.asarray(f_env, float), 0.0, None)
    tiles = [np.tile(c / math.sqrt(fk), (1, n_tiles, n_tiles)) for c, fk in zip(cs, f_k)]
    if len(f_k) == 1:
        return tiles[0] * np.sqrt(f)[None]
    k = np.clip(np.searchsorted(f_k, f, side="right") - 1, 0, len(f_k) - 2)
    w = np.clip((f - f_k[k]) / (f_k[k + 1] - f_k[k]), 0.0, 1.0)
    out = np.zeros((cs[0].shape[0],) + f.shape, complex)
    for j in range(len(f_k) - 1):
        sel = k == j
        if sel.any():
            out[:, sel] = (1 - w[sel]) * tiles[j][:, sel] + w[sel] * tiles[j + 1][:, sel]
    return out * np.sqrt(f)[None]


def add_noise(psi: np.ndarray, grid: Grid, noise: float, seed: int) -> np.ndarray:
    """Seeded real amplitude noise of relative rms `noise`, correlation length 2 a (the filter
    exp(-k^2)), on each component: what the polish has to remove."""
    if noise <= 0:
        return psi
    rng = np.random.default_rng(seed)
    kx, ky = [k.numpy() for k in grid.k64]
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    filt = np.exp(-0.25 * k2 * 4.0)
    out = psi.copy()
    for c in range(out.shape[0]):
        w = np.fft.ifft2(np.fft.fft2(rng.standard_normal(grid.N)) * filt).real
        out[c] = out[c] * (1.0 + noise * w / w.std())
    return out


# ----------------------------------------------------------------------------- the trapped state
def trap_grid(spec: TrappedSpec) -> tuple[Grid, MoireGeometry]:
    geo = moire_geometry(spec.m, spec.n)
    Nb = spec.box_cells * spec.cell_points
    Lb = spec.box_cells * geo.L_box
    return Grid((Nb, Nb), (Lb, Lb)), geo


def box_half_width(spec: TrappedSpec) -> float:
    """Distance from the cloud centre to the nearest edge of the periodic box (a)."""
    return 0.5 * spec.box_cells * moire_geometry(spec.m, spec.n).L_box


def containment(spec: TrappedSpec, scan: dict) -> dict:
    """The LDA cloud of `spec` in the infinite plane against its box: mu_plane, R_TF (a), the box
    half-width, ratio = R_TF / half-width, fits (ratio <= CONTAIN_REFUSE) and box_needed, the smallest odd
    box_cells with ratio <= CONTAIN_TARGET at the same N_atoms and nu_r_hz."""
    lda = LDA(scan["E0"], scan["fillings"], scan["mus"])
    w = spec.omega_r()
    mu = lda.solve_mu_plane(spec.N_atoms, w)
    R = math.sqrt(2.0 * max(mu - scan["E0"], 0.0) / units.M_LATTICE) / w
    half = box_half_width(spec)
    L = moire_geometry(spec.m, spec.n).L_box
    need = max(1, math.ceil(2.0 * R / (CONTAIN_TARGET * L)))
    need += 1 - need % 2  # odd
    return dict(
        mu_plane=float(mu),
        R_TF=float(R),
        half_width=float(half),
        ratio=float(R / half),
        fits=bool(R <= CONTAIN_REFUSE * half),
        box_cells=int(spec.box_cells),
        box_needed=int(need),
    )


def resolve_box(spec: TrappedSpec, scan: dict) -> tuple[TrappedSpec, dict]:
    """(spec, containment) with spec.containment applied: a cloud that fits is returned unchanged;
    otherwise "refuse" raises UncontainedCloud, "enlarge" returns the spec on box_needed (same N_atoms
    and nu_r_hz: the same physical cloud), "warn" warns and returns it unchanged."""
    if spec.containment not in CONTAINMENT_POLICIES:
        raise ValueError(f"containment must be one of {CONTAINMENT_POLICIES}, got {spec.containment!r}")
    info = containment(spec, scan)
    if info["fits"]:
        return spec, dict(info, action="none")
    msg = (
        f"the LDA cloud (R_TF = {info['R_TF']:.1f} a) is too wide for its {spec.box_cells}x{spec.box_cells} box "
        f"(half-width {info['half_width']:.1f} a, ratio {info['ratio']:.2f} > {CONTAIN_REFUSE:g}): its tail reaches "
        f"the periodic images. box_cells >= {info['box_needed']} holds it (R_TF <= {CONTAIN_TARGET:g} x half-width)"
    )
    if spec.containment == "enlarge":
        big = replace(spec, box_cells=info["box_needed"])
        return big, dict(containment(big, scan), action=f"enlarged from {spec.box_cells}", box_requested=spec.box_cells)
    if spec.containment == "warn":
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return spec, dict(info, action="warned")
    raise UncontainedCloud(msg + '; or set containment="enlarge" (or "warn" to run it anyway)')


ENLARGE_RETRIES = 2  # "enlarge": further +2 boxes when the solved cloud still fails the edge check


def trapped_ground_state(spec: TrappedSpec, scan: dict | None = None, verbose: bool = False) -> dict:
    """LDA start (+ seeded noise) -> heavy-ball polish (ground_state.precondition_polish via relax
    with an empty dt ladder) to spec.tol. The box is first checked against the cloud (`resolve_box`);
    with containment="enlarge" a solved cloud whose edge density is still >= CONTAIN_EDGE (a cloud too small
    for the Thomas-Fermi estimate, R_TF of a few oscillator lengths) is solved again two boxes larger, up to
    ENLARGE_RETRIES times. Returns dict(engine, psi (1, 2, Nb, Nb) tensor, result (RelaxResult), lda (mu_lda,
    f_centre, R_lda, E_per_N and mu of the start), scan, seconds, spec (the spec actually run: box_cells may
    have been enlarged), containment)."""
    t0 = time.perf_counter()
    if scan is None:
        scan = supercell_scan(spec, verbose=verbose)
    requested = spec.box_cells
    spec, cont = resolve_box(spec, scan)
    if verbose:
        print(
            f"  containment: R_TF = {cont['R_TF']:.2f} a in the plane, box half-width {cont['half_width']:.2f} a "
            f"(ratio {cont['ratio']:.3f}; {cont['action']})",
            flush=True,
        )
    for attempt in range(ENLARGE_RETRIES + 1):
        run = _solve_on_box(spec, scan, verbose)
        edge = cloud_numbers(run["engine"], run["psi"], run["geometry"].L_box)["edge_density_rel"]
        if spec.containment != "enlarge" or edge < CONTAIN_EDGE or attempt == ENLARGE_RETRIES:
            break
        if verbose:
            print(
                f"  edge density {edge:.1e} >= {CONTAIN_EDGE:g} on box {spec.box_cells}: enlarging to "
                f"{spec.box_cells + 2}",
                flush=True,
            )
        spec = replace(spec, box_cells=spec.box_cells + 2)
        cont = dict(containment(spec, scan), action=f"enlarged from {requested} (edge check)", box_requested=requested)
    run.update(seconds=time.perf_counter() - t0, spec=spec, containment=cont)
    return run


def _solve_on_box(spec: TrappedSpec, scan: dict, verbose: bool) -> dict:
    """The LDA start and the polish of `trapped_ground_state` on spec's box."""
    grid, geo = trap_grid(spec)
    w = spec.omega_r()
    VT = harmonic_trap(grid, (w, w))
    V = bilayer_potential(grid, spec.V0, geo) + VT
    eng = _engine(grid, V, spec)
    lda = LDA(scan["E0"], scan["fillings"], scan["mus"])
    VT_np = to_numpy(VT[0, 0]).astype(float)
    mu_lda = lda.solve_mu(VT_np, grid.dV, spec.N_atoms)
    f_env = lda(mu_lda - VT_np)
    psi_np = lda_cloud(scan["cells"], scan["fillings"], f_env, spec.box_cells)
    psi_lda = torch.as_tensor(psi_np[None], dtype=eng.dtype, device=grid.device)
    t_lda = eng.energy_terms(psi_lda)
    lda_info = dict(
        mu_lda=mu_lda,
        f_centre=float(lda(np.array([mu_lda]))[0]),
        R_lda=math.sqrt(2.0 * max(mu_lda - scan["E0"], 0.0) / units.M_LATTICE) / w,
        E_per_N=float(t_lda["E_total"][0] / t_lda["norm"][0]),
        mu=float(t_lda["mu"][0]),
        norm=float(t_lda["norm"][0]),
    )
    if verbose:
        print(
            f"  LDA start: mu_LDA={mu_lda:.6f} f_centre={lda_info['f_centre']:.3f} R_LDA={lda_info['R_lda']:.1f} a; "
            f"E/N={lda_info['E_per_N']:.8f} mu={lda_info['mu']:.8f} E_R",
            flush=True,
        )
    psi0 = torch.as_tensor(add_noise(psi_np, grid, spec.noise, spec.seed)[None], dtype=eng.dtype, device=grid.device)
    tp = time.perf_counter()
    r = relax(eng, psi0, spec.N_atoms, tol=spec.tol, dt_schedule=[], polish_steps=spec.polish_steps, verbose=verbose)
    if verbose:
        print(
            f"  polish: mu={float(r.mu[0]):.10f} E_R res={float(r.residual[0]):.1e} steps={r.steps} "
            f"({time.perf_counter() - tp:.0f} s)",
            flush=True,
        )
    return dict(
        engine=eng,
        psi=r.psi,
        result=r,
        lda=lda_info,
        scan=scan,
        grid=grid,
        geometry=geo,
        polish_seconds=time.perf_counter() - tp,
        psi_lda=psi_lda,
    )


# ----------------------------------------------------------------------------- observables
def cloud_numbers(eng: BilayerGPE, psi: torch.Tensor, L_cell: float) -> dict:
    """N, N2/N, R_rms (a and um), the mean total filling of the central moire cell |x|, |y| <= L/4
    (the AA cell at the origin), the atoms per supercell tile, the edge density relative to the peak."""
    p = to_numpy(psi[0])
    return cloud_numbers_from_density(np.abs(p[0]) ** 2, np.abs(p[1]) ** 2, eng.grid, L_cell)


def _cell_weights(x: np.ndarray, half: float, dx: float) -> np.ndarray:
    """Weights of the grid points x for the interval [-half, half]: 1 inside, 1/2 on a point that lies
    on an end (within 1e-6 dx), 0 outside. Symmetric, and the weights sum to 2 half / dx whenever the
    ends are grid points, so the result does not depend on how x_j = (j - N/2) dx rounds at the ends."""
    d = np.abs(x) - half
    tol = 1e-6 * dx
    return np.where(d < -tol, 1.0, np.where(np.abs(d) <= tol, 0.5, 0.0))


def centre_cell_mean(n: np.ndarray, grid: Grid, L_cell: float) -> float:
    """Mean of n (Nx, Ny) [ix, iy] over the central moire cell |x|, |y| <= L_cell / 4 (one moire cell
    of a supercell holding 2 x 2 of them), with the boundary rows at half weight (`_cell_weights`)."""
    x, y = (g.numpy() for g in grid.x64[:2])
    wx = _cell_weights(x, L_cell / 4, grid.dx[0])
    wy = _cell_weights(y, L_cell / 4, grid.dx[1])
    w = np.outer(wx, wy)
    return float((w * n).sum() / w.sum())


def cloud_numbers_from_density(n1: np.ndarray, n2: np.ndarray, grid: Grid, L_cell: float) -> dict:
    """`cloud_numbers` of two (Nx, Ny) component densities [ix, iy] on a 2D grid (or on the (x, y)
    axes of a 3D grid: a column density)."""
    n1, n2 = np.asarray(n1, np.float64), np.asarray(n2, np.float64)
    n = n1 + n2
    X, Y = np.meshgrid(*[x.numpy() for x in grid.x64[:2]], indexing="ij")
    dA = grid.dx[0] * grid.dx[1]
    N = float(n.sum() * dA)
    R = math.sqrt(float((n * (X**2 + Y**2)).sum() / n.sum()))
    Nb = n.shape[0]
    nt = Nb // round(grid.L[0] / L_cell)
    tiles = n.reshape(Nb // nt, nt, Nb // nt, nt).sum(axis=(1, 3)) * dA
    edge = max(n[0].max(), n[-1].max(), n[:, 0].max(), n[:, -1].max()) / n.max()
    a_um = units.lattice_units().a_m * 1e6
    return dict(
        N=N,
        N2_over_N=float(n2.sum() * dA) / N,
        R_rms=R,
        R_rms_um=R * a_um,
        nbar_centre_cell=centre_cell_mean(n, grid, L_cell),
        n_peak=float(n.max()),
        atoms_per_tile=tiles.tolist(),
        edge_density_rel=float(edge),
    )


def insitu_observables(eng: BilayerGPE, psi: torch.Tensor, s_sat: float = 0.01, mn: tuple = (22, 1)) -> dict:
    """The in-situ contrasts of |2> (`imaging.standard_observables`): C_raw (|k| < pi/a envelope,
    AA/AB), C_raw_central, C_psf / C_img_central (legacy Gaussian PSF), C_coh / C_coh_img (coherent
    imaging at s_sat), C_coh_infocus (s = 0), n_AA, n_AB, C_smooth (the ringing-free envelope) and
    raw_ringing (True when the raw envelope is negative at AA or AB, so C_raw is not a contrast)."""
    n = to_numpy(psi[0].real ** 2 + psi[0].imag ** 2).astype(np.float64)
    return insitu_from_density(n[0], n[1], eng.grid.dx[0], s_sat=s_sat, mn=mn)


def insitu_from_density(n1: np.ndarray, n2: np.ndarray, dx: float, s_sat: float = 0.01, mn: tuple = (22, 1)) -> dict:
    """`insitu_observables` of two (Nx, Ny) component densities [ix, iy] (or column densities)."""
    n1, n2 = np.asarray(n1, np.float64), np.asarray(n2, np.float64)
    coh = imaging.standard_observables(n1, n2, dx, imaging="coherent", s_sat=s_sat, mn=mn)
    foc = imaging.standard_observables(n1, n2, dx, imaging="pupil", mn=mn)
    return dict(
        C_raw=coh["C_raw"],
        C_raw_central=coh.get("C_raw_central"),
        C_psf=coh["C_psf"],
        C_img_central_psf=coh["C_img_central"],
        C_coh=coh["C_coh"],
        C_coh_img=coh["C_coh_img"],
        C_coh_infocus=foc["C_coh"],
        C_coh_infocus_img=foc["C_coh_img"],
        s_sat=s_sat,
        imaging_model=coh["imaging_model"],
        n_AA=coh["n_AA"],
        n_AB=coh["n_AB"],
        sign_AA_minus_AB=float(np.sign(coh["S_raw_AA"] - coh["S_raw_AB"])),
        C_smooth=coh["C_smooth"],
        raw_ringing=bool(coh["raw_ringing"]),
        S_raw_AB=coh["S_raw_AB"],
    )


COMPLEX64_FLOOR = 1e-4  # E_R: a single-precision solve that stops below this residual is at its floor, not stuck


def checks(
    edge_density_rel: float,
    converged: bool,
    residual: float | None = None,
    dtype: str | None = None,
    stop_reason: str | None = None,
) -> dict:
    """The two conditions under which a trapped run describes one converged cloud: contained (edge
    density / peak < CONTAIN_EDGE, the trapped-cloud gate's criterion) and converged (residual below the
    run's tol). A complex64 solve cannot reach a tol below its rounding floor (it stops at 1e-5 ... 6e-5 E_R
    on these clouds): with dtype "complex64", a residual below COMPLEX64_FLOOR and a solve that ended on a
    stall (stop_reason "stall" or "tau_floor", not "max_steps" while still falling) it counts as converged to
    that floor (`at_precision_floor`, noted, not a failure). ok = contained and (converged or at the floor);
    warnings lists what failed, for reports."""
    contained = bool(edge_density_rel < CONTAIN_EDGE)
    floor = bool(
        not converged
        and dtype is not None
        and "64" in str(dtype)
        and "128" not in str(dtype)
        and residual is not None
        and residual < COMPLEX64_FLOOR
        and stop_reason in ("stall", "tau_floor")
    )
    ok = contained and (bool(converged) or floor)
    out = dict(contained=contained, converged=bool(converged), at_precision_floor=floor, ok=ok, warnings=[])
    if not contained:
        out["warnings"].append(
            f"NOT CONTAINED: edge density / peak = {edge_density_rel:.1e} >= {CONTAIN_EDGE:g}; the cloud overlaps its "
            "periodic images and its observables do not describe one trapped cloud"
        )
    if not converged and not floor:
        res = "" if residual is None else f" (residual {residual:.1e})"
        out["warnings"].append(f"NOT CONVERGED{res}: the solve stopped above its tol")
    return out


def summary(run: dict, spec: TrappedSpec, s_sat: float = 0.01) -> dict:
    """Everything a trapped run reports (JSON-ready): spec (the one actually run, `run["spec"]`), scan
    numbers, containment, LDA start, polish, cloud numbers, in-situ contrasts, checks."""
    spec = run.get("spec", spec)
    eng, psi, r = run["engine"], run["psi"], run["result"]
    geo = run["geometry"]
    terms = eng.energy_terms(psi)
    sc = run["scan"]
    out = dict(
        spec=asdict(spec),
        g2d=spec.couplings()[0],
        omega_z=spec.couplings()[1],
        omega_r=spec.omega_r(),
        grid=dict(N=list(eng.grid.N), L=list(eng.grid.L), dx=list(eng.grid.dx)),
        scan=dict(
            fillings=sc["fillings"], mus=sc["mus"], E0=sc["E0"], residuals=sc["residuals"], seconds=sc.get("seconds")
        ),
        lda=run["lda"],
        ground=dict(
            mu=float(r.mu[0]),
            E_per_N=float(terms["E_total"][0] / terms["norm"][0]),
            residual=float(r.residual[0]),
            converged=bool(r.converged[0]),
            steps=int(r.steps),
            stop_reason=r.stop_reason,
            polish_seconds=run["polish_seconds"],
            total_seconds=run["seconds"],
        ),
        cloud=cloud_numbers(eng, psi, geo.L_box),
        containment=run.get("containment"),
    )
    mn = (spec.m, spec.n)
    out["insitu"] = insitu_observables(eng, psi, s_sat=s_sat, mn=mn)
    lda_psi = run["psi_lda"]
    out["lda"]["insitu"] = insitu_observables(eng, renormalise(eng, lda_psi, spec.N_atoms), s_sat=s_sat, mn=mn)
    out["lda"]["cloud"] = cloud_numbers(eng, lda_psi, geo.L_box)
    g = out["ground"]
    out["checks"] = checks(
        out["cloud"]["edge_density_rel"], g["converged"], g["residual"], spec.dtype, g["stop_reason"]
    )
    return out
