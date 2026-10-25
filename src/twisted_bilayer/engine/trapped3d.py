"""Trapped clouds in full 3D against the quasi-2D reductions, sized to the device.

The trapped cloud of `engine.trapped` (a harmonic trap on box_cells x box_cells sqrt(m^2 + n^2)
supercells, LDA start from the supercell scan) is solved three ways on the same (x, y) grid:

  npse   quasi-2D with the density-dependent Gaussian width (engine.npse; the default everywhere)
  fixed  quasi-2D with g_2D = g_3D / (sqrt(2 pi) l_z)
  3d     the full 3D problem: V_lattice(x, y) + (1/2) m [omega_r^2 (x^2 + y^2) + omega_z^2 z^2] and g_3D,
         on N_z planes of a z box of +-z_extent l_z

The 3D start is the NPSE cloud times the NPSE's local Gaussian, psi(x, y) phi_sigma(x, y)(z) with
sigma = s(n) l_z, so the column density of the start is the NPSE density and the polish only has to
move what the reduction misses. Compared (`compare_models`): mu (3D minus hbar omega_z / 2, the
zero-point energy every 2D model leaves out), E/N, the column density (largest and integrated
deviation), the local width <z^2> / (l_z^2 / 2) against the NPSE's s^2, and the cloud numbers and
in-situ contrasts of `engine.trapped`.

Cloud scaling. A larger box holds the same cloud scaled at fixed peak density, so the local physics
is unchanged and only the number of moire cells grows: nu_r = nu_ref box_ref / box and
N = N_ref (box / box_ref)^2 (in 2D Thomas-Fermi, R ~ sqrt(mu) / omega_r and N ~ mu R^2). At
box_cells = box_ref = 3 the cloud is the `trapped` scenario (N = 2000, 20 Hz).

Sizing (`probe`, `plan`). The 3D polish dominates both memory and time. `probe` runs a few polish
steps of the 3D engine at box_cells = 1 for two N_z and fits peak memory and time per step as
a + b P (P = 3D grid points; FFT work is P log P, linear to a few per cent over the range).
`plan` then takes the largest odd box_cells that fits the budget at N_z >= Nz_min, and spends what is
left on the largest N_z (a multiple of 8, <= Nz_max). The budget is device memory for the
largest-system run and wall time for the precision run.

Units: lattice units (a = 1, E_R = 1, hbar = 1); fields (B, C, Nx, Ny[, Nz]) in grid order [ix, iy, iz].
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, replace

import numpy as np
import torch

from . import trapped as TR
from .backend import HAS_GPU, REAL_OF, complex_dtype, synchronize, to_numpy
from .grid import Grid
from .ground_state import precondition_polish, relax, renormalise
from .lattice import bilayer_potential, harmonic_trap
from .npse import NPSE2D
from .physics import BilayerGPE

BOX_REF = 3  # box_cells at which a row's N_atoms and nu_r_hz are given
MODELS = ("npse", "fixed", "3d")


@dataclass
class Vertical:
    """The z direction of the 3D solve."""

    Nz: int = 32
    z_extent_lz: float = 5.0  # z box +-z_extent l_z
    polish_steps: int = 20000


def scaled(spec: TR.TrappedSpec, box_cells: int, box_ref: int = BOX_REF) -> TR.TrappedSpec:
    """The row's cloud (given at box_ref) scaled to box_cells at fixed peak density."""
    if box_cells < 1 or box_cells % 2 != 1:
        raise ValueError(f"box_cells must be odd and >= 1, got {box_cells}")
    r = box_cells / box_ref
    return replace(spec, box_cells=int(box_cells), nu_r_hz=spec.nu_r_hz / r, N_atoms=spec.N_atoms * r * r)


def grid_3d(spec: TR.TrappedSpec, vert: Vertical) -> Grid:
    g2, _ = TR.trap_grid(spec)
    lz = TR.units.lattice_units(omega_z_hz=spec.omega_z_hz).l_z
    return Grid((*g2.N, int(vert.Nz)), (*g2.L, 2.0 * vert.z_extent_lz * lz))


def couplings_3d(spec: TR.TrappedSpec) -> tuple[float, float, float]:
    """(g_3D, omega_z, l_z), with g_3D = g_2D sqrt(2 pi) l_z so that a row's g2d override carries over."""
    g2d, wz = spec.couplings()
    lz = TR.units.lattice_units(omega_z_hz=spec.omega_z_hz).l_z
    return g2d * math.sqrt(2.0 * math.pi) * lz, wz, lz


def engine_3d(spec: TR.TrappedSpec, grid: Grid, dtype: str | torch.dtype | None = None) -> BilayerGPE:
    dt = complex_dtype(spec.dtype if dtype is None else dtype)
    g3d, wz, _ = couplings_3d(spec)
    _, geo = TR.trap_grid(spec)
    w = spec.omega_r()
    V = bilayer_potential(grid, spec.V0, geo) + harmonic_trap(grid, (w, w, wz))
    return BilayerGPE(grid.to(REAL_OF[dt]), V=V, g=g3d, Omega_R=spec.Omega_R, dtype=dt)


def start_3d(psi2: torch.Tensor, npse: NPSE2D, grid: Grid, l_z: float, dtype: torch.dtype) -> torch.Tensor:
    """psi(x, y) phi_sigma(z), sigma = s(n) l_z: the NPSE's own vertical profile under each column."""
    n = (psi2.real**2 + psi2.imag**2).sum(dim=1, keepdim=True).to(torch.float64)
    sigma = (npse.width(n) * l_z).unsqueeze(-1)  # (B, 1, Nx, Ny, 1)
    z = torch.as_tensor(grid.x64[2], dtype=torch.float64, device=psi2.device).reshape(1, 1, 1, 1, -1)
    phi = (math.pi * sigma * sigma) ** -0.25 * torch.exp(-0.5 * (z / sigma) ** 2)
    return (psi2.to(torch.complex128).unsqueeze(-1) * phi).to(dtype)


def column_moments(psi: torch.Tensor, grid: Grid, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Per-component column densities (C, Nx, Ny) and the local <z^2>(x, y) of the total density, in
    float64 numpy, computed in x slabs so that no second full 3D field is allocated."""
    dz = grid.dx[2]
    z2 = torch.as_tensor(grid.x64[2] ** 2, device=psi.device)
    cols, mom = [], []
    for i in range(0, psi.shape[2], chunk):
        sl = psi[0, :, i : i + chunk]
        d = (sl.real**2 + sl.imag**2).to(torch.float64)  # (C, nx, Ny, Nz)
        cols.append(to_numpy(d.sum(dim=-1) * dz))
        mom.append(to_numpy((d.sum(dim=0) * z2).sum(dim=-1) * dz))
    col = np.concatenate(cols, axis=1)
    n = col.sum(axis=0)
    return col, np.concatenate(mom, axis=0) / np.maximum(n, 1e-300)


def _peak_gib() -> float | None:
    return torch.cuda.max_memory_allocated() / 2**30 if HAS_GPU else None


def _reset_peak() -> None:
    if HAS_GPU:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ----------------------------------------------------------------------------- the three solves
def _no_enlarge(spec: TR.TrappedSpec) -> TR.TrappedSpec:
    """containment="enlarge" becomes "refuse" for a 3D run: its box and N_z are sized (and memory-checked)
    before the supercell scan exists, so an enlarged box would bypass that sizing; pin box_cells to the box
    the refusal names instead. Note also that `scaled` keeps R_TF / half-width fixed: a larger box at fixed
    peak density never contains a cloud that box 3 does not; the same N and nu_r in a larger box need the
    row's N_atoms and nu_r_hz divided by (box / 3)^2 and multiplied by box / 3."""
    if spec.containment == "enlarge":
        warnings.warn(
            'containment="enlarge" is not available for 3D runs (the box is sized before the scan); using "refuse"',
            RuntimeWarning,
            stacklevel=3,
        )
        return replace(spec, containment="refuse")
    return spec


def solve_models(
    spec: TR.TrappedSpec,
    vert: Vertical,
    dtype: str = "complex128",
    scan: dict | None = None,
    verbose: bool = False,
    dtype_2d: str | None = None,
) -> dict:
    """The NPSE, fixed-g and 3D ground states of one trapped cloud: the 3D solve in `dtype`, the two
    quasi-2D solves in `dtype_2d` (default: `dtype`). Returns dict(scan, models={name: dict(n (C, Nx, Ny)
    column density, mu, E_per_N, residual, steps, seconds, peak_gib, ...)}, width_3d, width_npse (both
    <z^2> / (l_z^2 / 2)), grid2, grid3)."""
    dtype_2d = dtype if dtype_2d is None else dtype_2d
    spec = replace(_no_enlarge(spec), dtype=str(dtype_2d), vertical="npse")
    g3d, wz, lz = couplings_3d(spec)
    if scan is None:
        scan = TR.supercell_scan(spec, verbose=verbose)
    out = dict(scan=scan, models={}, dtype=str(dtype), dtype_2d=str(dtype_2d))

    # -- quasi-2D, NPSE: LDA start -> polish ------------------------------------------------------
    _reset_peak()
    run = TR.trapped_ground_state(spec, scan=scan, verbose=verbose)
    synchronize()
    spec = run["spec"]  # resolve_box may have enlarged the box: the fixed-g and 3D solves use the same one
    out["containment"] = run["containment"]
    eng_n, psi_n, r = run["engine"], run["psi"], run["result"]
    out["models"]["npse"] = _numbers_2d(eng_n, psi_n, r, run["polish_seconds"], wz=0.0)
    grid2 = run["grid"]
    if verbose:
        _say("npse", out["models"]["npse"])

    # -- quasi-2D, fixed g_2D: from the NPSE cloud -------------------------------------------------
    _reset_peak()
    t0 = time.perf_counter()
    fixed = replace(spec, vertical="fixed")
    eng_f = TR._engine(grid2, eng_n.V, fixed)  # the same two lattices + trap
    rf = relax(eng_f, psi_n, spec.N_atoms, tol=spec.tol, dt_schedule=[], polish_steps=spec.polish_steps)
    synchronize()
    out["models"]["fixed"] = _numbers_2d(eng_f, rf.psi, rf, time.perf_counter() - t0, wz=0.0)
    if verbose:
        _say("fixed", out["models"]["fixed"])
    del eng_f, rf

    # -- full 3D: from the NPSE cloud times its Gaussian ------------------------------------------
    npse = eng_n.interaction
    n_np = to_numpy((psi_n.real**2 + psi_n.imag**2)[0]).astype(np.float64)
    s_np = to_numpy(npse.width(torch.as_tensor(n_np.sum(axis=0))))
    out["width_npse"] = s_np**2
    out["models"]["npse"]["n"] = n_np
    psi_start = psi_n.detach()
    del run, eng_n, psi_n, r
    out["models"]["3d"], out["width_3d"], grid3 = solve_3d(spec, vert, dtype, psi_start, npse)
    if verbose:
        _say("3d", out["models"]["3d"])
    out.update(grid2=grid2, grid3=grid3, spec=spec, vertical=vert, g3d=g3d, omega_z=wz, l_z=lz)
    return out


def solve_3d(
    spec: TR.TrappedSpec, vert: Vertical, dtype, psi_start: torch.Tensor, npse: NPSE2D
) -> tuple[dict, np.ndarray, Grid]:
    """The 3D ground state from a quasi-2D cloud psi_start (1, 2, Nx, Ny) times the NPSE Gaussian.
    Returns (numbers as in `solve_models`, the width map <z^2> / (l_z^2 / 2), the 3D grid)."""
    _, wz, lz = couplings_3d(spec)
    _reset_peak()
    t0 = time.perf_counter()
    grid3 = grid_3d(spec, vert)
    e3 = engine_3d(spec, grid3, dtype)
    psi3 = start_3d(psi_start, npse, e3.grid, lz, e3.dtype)
    r3 = relax(e3, psi3, spec.N_atoms, tol=spec.tol, dt_schedule=[], polish_steps=vert.polish_steps)
    del psi3
    synchronize()
    sec3 = time.perf_counter() - t0
    terms = e3.energy_terms(r3.psi)
    col, z2 = column_moments(r3.psi, grid3)
    numbers = dict(
        n=col,
        mu=float(r3.mu[0]) - 0.5 * wz,
        E_per_N=float(terms["E_total"][0] / terms["norm"][0]) - 0.5 * wz,
        residual=float(r3.residual[0]),
        converged=bool(r3.converged[0]),
        steps=int(r3.steps),
        stop_reason=r3.stop_reason,
        seconds=sec3,
        peak_gib=_peak_gib(),
        dtype=str(e3.dtype),
    )
    del e3, r3
    _reset_peak()
    return numbers, z2 / (0.5 * lz * lz), grid3


# ----------------------------------------------------------------------------- z resolution
def nz_convergence(
    spec: TR.TrappedSpec,
    Nz_list,
    z_extent_lz: float = 5.0,
    dtype: str = "complex64",
    dtype_2d: str = "complex128",
    polish_steps: int = 20000,
    verbose: bool = False,
) -> dict:
    """The 3D cloud at several N_z (same z box, so dz = 2 z_extent l_z / N_z), each from the same NPSE
    start, against the finest N_z and against the NPSE. Returns dict(rows = [dict(Nz, dz_lz, numbers,
    observables, width_max, width_mean)], npse = observables of the NPSE cloud, finest = index of the
    finest N_z, spec)."""
    Nz_list = sorted(int(n) for n in Nz_list)
    spec2 = replace(_no_enlarge(spec), dtype=str(dtype_2d), vertical="npse")
    _, wz, lz = couplings_3d(spec2)
    scan = TR.supercell_scan(spec2, verbose=verbose)
    run = TR.trapped_ground_state(spec2, scan=scan, verbose=verbose)
    spec2 = run["spec"]
    containment = run["containment"]
    eng_n, psi_n = run["engine"], run["psi"]
    npse = eng_n.interaction
    grid2 = run["grid"]
    n_np = to_numpy((psi_n.real**2 + psi_n.imag**2)[0]).astype(np.float64)
    base = dict(spec=spec2, grid2=grid2, models={"npse": _numbers_2d(eng_n, psi_n, run["result"], 0.0, 0.0)})
    base["models"]["npse"]["n"] = n_np
    psi_start = psi_n.detach()
    del run, eng_n, psi_n
    rows = []
    for Nz in Nz_list:
        vert = Vertical(Nz=Nz, z_extent_lz=z_extent_lz, polish_steps=polish_steps)
        numbers, width, grid3 = solve_3d(spec2, vert, dtype, psi_start, npse)
        if verbose:
            _say(f"Nz={Nz}", numbers)
        n = numbers["n"].sum(axis=0)
        occ = n > 1e-3 * n.max()
        rows.append(
            dict(
                Nz=Nz,
                dz_lz=grid3.dx[2] / lz,
                numbers=numbers,
                width_max=float(width[occ].max()),
                width_mean=float((width * n)[occ].sum() / n[occ].sum()),
            )
        )
    obs = model_observables(
        dict(base, models={"npse": base["models"]["npse"], **{r["Nz"]: r["numbers"] for r in rows}})
    )
    for r in rows:
        r["observables"] = obs[r["Nz"]]
    return dict(
        rows=rows,
        npse=obs["npse"],
        finest=len(rows) - 1,
        spec=spec2,
        grid2=grid2,
        z_extent_lz=z_extent_lz,
        containment=containment,
    )


NZ_KEYS = ("mu", "E_per_N", "C_raw", "C_coh", "n_AA", "n_AB", "R_rms", "n_peak")


def report_nz(conv: dict) -> str:
    """Each N_z against the finest one, and the 3D - NPSE gap it resolves."""
    rows, fin = conv["rows"], conv["rows"][conv["finest"]]
    dA = conv["grid2"].dx[0] * conv["grid2"].dx[1]
    head = f"  {'N_z':>5s}{'dz/l_z':>8s}" + "".join(f"{k:>13s}" for k in ("mu", "C_raw", "n_AA", "w_max"))
    head += f"{'max|dn|/n':>11s}{'residual':>10s}{'steps':>7s}{'s':>6s}"
    lines = [head]
    for r in rows:
        o = r["observables"]
        dev = density_deviation(r["numbers"]["n"], fin["numbers"]["n"], dA)["max_rel"]
        lines.append(
            f"  {r['Nz']:>5d}{r['dz_lz']:>8.3f}{o['mu']:>13.8f}{o['C_raw']:>13.6f}{o['n_AA']:>13.5f}"
            f"{r['width_max']:>13.5f}{dev:>11.1e}{o['residual']:>10.1e}{o['steps']:>7d}{o['seconds']:>6.0f}"
        )
    lines.append("  differences from the finest N_z, and the 3D - NPSE gap at the finest N_z:")
    lines.append(f"  {'N_z':>5s}" + "".join(f"{k:>12s}" for k in NZ_KEYS))
    for r in rows[:-1]:
        lines.append(
            f"  {r['Nz']:>5d}" + "".join(f"{r['observables'][k] - fin['observables'][k]:>+12.2e}" for k in NZ_KEYS)
        )
    lines.append(f"  {'gap':>5s}" + "".join(f"{fin['observables'][k] - conv['npse'][k]:>+12.2e}" for k in NZ_KEYS))
    worst = (
        max(
            abs(rows[-2]["observables"][k] - fin["observables"][k])
            / max(abs(fin["observables"][k] - conv["npse"][k]), 1e-300)
            for k in NZ_KEYS
        )
        if len(rows) > 1
        else float("nan")
    )
    lines.append(
        f"  the second-finest N_z = {rows[-2]['Nz'] if len(rows) > 1 else '-'} is off from the finest by at most "
        f"{worst:.1%} of the 3D - NPSE gap"
    )
    return "\n".join(lines)


def _numbers_2d(eng, psi, r, seconds, wz) -> dict:
    terms = eng.energy_terms(psi)
    n = to_numpy((psi.real**2 + psi.imag**2)[0]).astype(np.float64)
    return dict(
        n=n,
        mu=float(r.mu[0]) - 0.5 * wz,
        E_per_N=float(terms["E_total"][0] / terms["norm"][0]),
        residual=float(r.residual[0]),
        converged=bool(r.converged[0]),
        steps=int(r.steps),
        stop_reason=r.stop_reason,
        seconds=float(seconds),
        peak_gib=_peak_gib(),
        dtype=str(eng.dtype),
    )


def _say(name, m) -> None:
    mem = f", peak {m['peak_gib']:.2f} GiB" if m.get("peak_gib") is not None else ""
    print(
        f"  {name:>5s}: mu = {m['mu']:.8f} E_R, residual {m['residual']:.1e}, {m['steps']} steps, "
        f"{m['seconds']:.0f} s{mem}",
        flush=True,
    )


# ----------------------------------------------------------------------------- observables
def model_observables(res: dict, s_sat: float = 0.01) -> dict:
    """Cloud numbers and in-situ contrasts of each model's (column) density, plus mu, E/N, residual."""
    spec, grid2 = res["spec"], res["grid2"]
    _, geo = TR.trap_grid(spec)
    mn = (spec.m, spec.n)
    obs = {}
    for name, m in res["models"].items():
        n = m["n"]
        cloud = TR.cloud_numbers_from_density(n[0], n[1], grid2, geo.L_box)
        ins = TR.insitu_from_density(n[0], n[1], grid2.dx[0], s_sat=s_sat, mn=mn)
        obs[name] = dict(
            mu=m["mu"],
            E_per_N=m["E_per_N"],
            C_raw=ins["C_raw"],
            C_coh=ins["C_coh"],
            n_AA=ins["n_AA"],
            n_AB=ins["n_AB"],
            R_rms=cloud["R_rms"],
            N2_over_N=cloud["N2_over_N"],
            n_peak=cloud["n_peak"],
            nbar_centre_cell=cloud["nbar_centre_cell"],
            edge_density_rel=cloud["edge_density_rel"],
            residual=m["residual"],
            converged=m["converged"],
            steps=m["steps"],
            stop_reason=m.get("stop_reason"),
            dtype=m.get("dtype"),
            seconds=m["seconds"],
            peak_gib=m["peak_gib"],
            C_smooth=ins["C_smooth"],
            raw_ringing=ins["raw_ringing"],
        )
        obs[name]["checks"] = TR.checks(
            cloud["edge_density_rel"], m["converged"], m["residual"], m.get("dtype"), m.get("stop_reason")
        )
    return obs


def run_checks(obs: dict) -> dict:
    """The checks of every model in one place: per model contained / converged, ok = all of them, and the
    warnings (prefixed with the model) for the report."""
    per = {name: o["checks"] for name, o in obs.items() if isinstance(o, dict) and "checks" in o}
    warns = [f"{name}: {w}" for name, c in per.items() for w in c["warnings"]]
    return dict(
        models=per,
        contained=all(c["contained"] for c in per.values()),
        converged=all(c["converged"] for c in per.values()),
        at_precision_floor=[name for name, c in per.items() if c.get("at_precision_floor")],
        ok=all(c["ok"] for c in per.values()),
        warnings=warns,
    )


def describe_checks(chk: dict) -> str:
    """One line for a report: ok, contained, converged, and which models stopped at the complex64 floor."""
    floor = chk.get("at_precision_floor") or []
    note = f"; at the complex64 residual floor: {', '.join(str(f) for f in floor)}" if floor else ""
    return f"checks: ok {chk['ok']} (contained {chk['contained']}, converged {chk['converged']}{note})"


COMPARED = ("mu", "E_per_N", "C_raw", "C_coh", "n_AA", "n_AB", "R_rms", "N2_over_N", "n_peak")


def density_deviation(a: np.ndarray, b: np.ndarray, dA: float) -> dict:
    """Total densities of two (C, Nx, Ny) fields: max |a - b| / max b and int |a - b| / N_b."""
    na, nb = a.sum(axis=0), b.sum(axis=0)
    d = np.abs(na - nb)
    return dict(max_rel=float(d.max() / nb.max()), l1_rel=float(d.sum() / nb.sum()))


def compare_models(res: dict, obs: dict) -> dict:
    """3D against each quasi-2D model: observable differences, column-density deviation, widths."""
    dA = res["grid2"].dx[0] * res["grid2"].dx[1]
    out = {}
    for ref in ("npse", "fixed"):
        out[f"3d_vs_{ref}"] = dict(
            diff={k: obs["3d"][k] - obs[ref][k] for k in COMPARED},
            density=density_deviation(res["models"]["3d"]["n"], res["models"][ref]["n"], dA),
        )
    w3, wn = res["width_3d"], res["width_npse"]
    n = res["models"]["3d"]["n"].sum(axis=0)
    occ = n > 1e-3 * n.max()
    out["width"] = dict(
        width_3d_max=float(w3[occ].max()),
        width_npse_max=float(wn[occ].max()),
        width_3d_mean=float((w3 * n)[occ].sum() / n[occ].sum()),
        width_npse_mean=float((wn * n)[occ].sum() / n[occ].sum()),
        max_abs_diff=float(np.abs(w3 - wn)[occ].max()),
    )
    return out


# ----------------------------------------------------------------------------- sizing
@dataclass
class Cost:
    """cost(P) = a + b P for one engine and precision (bytes or seconds per polish step)."""

    a: float
    b: float

    def __call__(self, points: float) -> float:
        return self.a + self.b * points


def probe(
    spec: TR.TrappedSpec, vert: Vertical, dtype: str, Nz_pair: tuple[int, int] = (16, 32), steps: int = 10
) -> dict:
    """Peak memory (bytes; GPU only) and seconds per polish step of the 3D engine at box_cells = 1 for
    two N_z, fitted as a + b P. The probe state is a Gaussian, not a ground state: the cost of a
    polish step does not depend on the state."""
    s1 = scaled(spec, 1)
    mem, sec, pts = [], [], []
    for Nz in Nz_pair:
        _reset_peak()
        base = torch.cuda.memory_allocated() if HAS_GPU else 0
        grid = grid_3d(s1, replace(vert, Nz=Nz))
        eng = engine_3d(s1, grid, dtype)
        g = eng.grid
        r2 = sum(c**2 for c in (g.coord(0), g.coord(1)))
        z = g.coord(2)
        psi = torch.exp(-r2 / 50.0 - z * z).to(eng.dtype).expand(1, 2, *g.N).contiguous()
        psi = renormalise(eng, psi, s1.N_atoms)
        precondition_polish(eng, psi, s1.N_atoms, tol=0.0, max_steps=2, check_every=2)  # FFT plans, warm-up
        synchronize()
        t0 = time.perf_counter()
        precondition_polish(eng, psi, s1.N_atoms, tol=0.0, max_steps=steps, check_every=steps)
        synchronize()
        sec.append((time.perf_counter() - t0) / steps)
        mem.append((torch.cuda.max_memory_allocated() - base) if HAS_GPU else 0.0)
        pts.append(g.numel)
        del eng, psi, r2, z
    _reset_peak()

    def fit(y):
        b = (y[1] - y[0]) / (pts[1] - pts[0])
        if b <= 0.0:  # timing noise at small sizes: fall back to cost proportional to the points
            return Cost(a=0.0, b=max(y[1], 1e-300) / pts[1])
        return Cost(a=max(y[0] - b * pts[0], 0.0), b=b)

    m = fit(mem) if HAS_GPU else None
    if m is not None:
        m = Cost(a=PROBE_MEMORY_FACTOR * m.a, b=PROBE_MEMORY_FACTOR * m.b)
    return dict(memory=m, step_seconds=fit(sec), points=pts, dtype=str(dtype))


def plan(
    cost,
    budget: float,
    cell_points: int,
    Nz_min: int = 24,
    Nz_max: int = 128,
    box_max: int = 41,
    box_fixed: int | None = None,
    Nz_fixed: int | None = None,
) -> tuple[int, int] | None:
    """The largest odd box_cells with cost(box, Nz_min) <= budget, then the largest N_z (a multiple of
    8, Nz_min <= N_z <= Nz_max) at that box. cost(box, Nz) is any callable; None if box 1 does not fit.
    box_fixed / Nz_fixed pin one of the two and size the other."""
    nz0 = Nz_min if Nz_fixed is None else Nz_fixed
    if box_fixed is not None:
        best = box_fixed if cost(box_fixed, nz0) <= budget else None
    else:
        best = None
        for box in range(1, box_max + 1, 2):
            if cost(box, nz0) <= budget:
                best = box
            else:
                break
    if best is None:
        return None
    if Nz_fixed is not None:
        return best, Nz_fixed
    Nz = Nz_min
    for cand in range(Nz_min, Nz_max + 1):
        if cand % 8 == 0 and cost(best, cand) <= budget:
            Nz = cand
    return best, Nz


def points_3d(box: int, Nz: int, cell_points: int) -> int:
    return (box * cell_points) ** 2 * Nz


def memory_budget(fraction: float = 0.85) -> float | None:
    """Bytes the 3D solve may allocate: `fraction` of the device memory free now (None on the CPU)."""
    if not HAS_GPU:
        return None
    free, _total = torch.cuda.mem_get_info()
    return fraction * free + torch.cuda.memory_allocated()


# The wall-time prediction (nothing else uses these). Polish steps in complex128 to tol 1e-9 on the m = 22
# cloud at box 3, N_z = 48 (a 6 GB GPU): 1600 (3D from the NPSE start) and 1650 / 1550 (NPSE / fixed g from
# the LDA start); complex64 stops at its residual floor (~1e-5) after 400-1200. The same run took 0.447 s per
# 3D step against the probe's 0.303 s (the probe extrapolates from box 1): PROBE_TIME_FACTOR.
STEPS_3D = 1600
STEPS_2D = 1600
PROBE_TIME_FACTOR = 1.5
# The full run peaks above the probe's polish peak (the 3D start and the observables): 4.53 GiB against
# the probe's 4.08 GiB at box 5, N_z = 32, complex64 (same GPU). The memory plan uses the probe x 1.15.
PROBE_MEMORY_FACTOR = 1.15


def predicted_seconds(step3: Cost, box: int, Nz: int, cell_points: int, steps_3d=STEPS_3D, steps_2d=STEPS_2D):
    """Wall time of one `solve_models` call: the 3D polish plus two 2D polishes, each 2D step costing
    about a 3D step of one z plane (the 2D engine does the same work per point)."""
    p3 = points_3d(box, Nz, cell_points)
    t3 = steps_3d * step3(p3)
    t2 = 2 * steps_2d * step3(p3 / Nz)
    return PROBE_TIME_FACTOR * (t3 + t2)


def describe_plan(spec: TR.TrappedSpec, vert: Vertical) -> str:
    g3 = grid_3d(spec, vert)
    lz = TR.units.lattice_units(omega_z_hz=spec.omega_z_hz).l_z
    um = TR.units.lattice_units().a_m * 1e6
    return (
        f"box {spec.box_cells}x{spec.box_cells} supercells = {g3.L[0]:.1f} a = {g3.L[0] * um:.1f} um "
        f"({g3.N[0]}^2 points, dx = {g3.dx[0]:.4f} a), N_z = {g3.N[2]} (dz = {g3.dx[2] / lz:.3f} l_z, "
        f"z box +-{vert.z_extent_lz:g} l_z): {g3.numel:,} 3D points; cloud N = {spec.N_atoms:.0f} atoms, "
        f"nu_r = {spec.nu_r_hz:.3g} Hz"
    )


# ----------------------------------------------------------------------------- reports and files
_FMT = dict(mu="{:.8f}", E_per_N="{:.8f}", C_raw="{:.5f}", C_coh="{:.5f}", n_AA="{:.4f}", n_AB="{:.4f}")


def report_models(obs: dict, cmp: dict) -> str:
    """The three models side by side and the 3D - 2D differences."""
    keys = ("mu", "E_per_N", "C_raw", "C_coh", "n_AA", "n_AB", "R_rms", "n_peak")
    lines = [f"  {'':>10s}" + "".join(f"{m:>14s}" for m in MODELS) + f"{'3d - npse':>14s}{'3d - fixed':>14s}"]
    for k in keys:
        row = f"  {k:>10s}" + "".join(f"{_FMT.get(k, '{:.5f}').format(obs[m][k]):>14s}" for m in MODELS)
        row += f"{cmp['3d_vs_npse']['diff'][k]:>+14.2e}{cmp['3d_vs_fixed']['diff'][k]:>+14.2e}"
        lines.append(row)
    lines.append(
        f"  {'residual':>10s}" + "".join(f"{obs[m]['residual']:>14.1e}" for m in MODELS) + "   (E_R; mu and E/N of"
        " 3d without hbar omega_z / 2)"
    )
    lines.append(f"  {'steps':>10s}" + "".join(f"{obs[m]['steps']:>14d}" for m in MODELS))
    lines.append(f"  {'seconds':>10s}" + "".join(f"{obs[m]['seconds']:>14.0f}" for m in MODELS))
    if obs["3d"].get("peak_gib") is not None:
        lines.append(f"  {'peak GiB':>10s}" + "".join(f"{obs[m]['peak_gib']:>14.2f}" for m in MODELS))
    for ref in ("npse", "fixed"):
        d = cmp[f"3d_vs_{ref}"]["density"]
        lines.append(
            f"  column density, 3d vs {ref}: max |dn| / n_max = {d['max_rel']:.2e}, int |dn| / N = {d['l1_rel']:.2e}"
        )
    w = cmp["width"]
    lines.append(
        f"  vertical width <z^2> / (l_z^2/2): 3d mean {w['width_3d_mean']:.4f} max {w['width_3d_max']:.4f}; "
        f"npse s^2 mean {w['width_npse_mean']:.4f} max {w['width_npse_max']:.4f}; max |3d - npse| {w['max_abs_diff']:.3f}"
    )
    return "\n".join(lines)


def precision_table(obs64: dict, obs128: dict, cmp128: dict) -> tuple[str, dict]:
    """complex64 against complex128 for each model, and the precision error against the 3D - NPSE gap
    (complex128): complex64 'resolves' an observable when its error is below 10 % of that gap."""
    lines = [
        f"  {'':>10s}{'model':>7s}{'complex128':>16s}{'complex64':>16s}{'|64 - 128|':>12s}"
        f"{'|3d - npse|':>13s}{'ratio':>9s}"
    ]
    numbers = {}
    resolved, total = 0, 0
    for k in COMPARED:
        gap = abs(cmp128["3d_vs_npse"]["diff"][k])
        for m in MODELS:
            a, b = obs128[m][k], obs64[m][k]
            err = abs(b - a)
            ratio = err / gap if gap > 0 else float("inf")
            numbers.setdefault(k, {})[m] = dict(complex128=a, complex64=b, abs_err=err, gap_3d_npse=gap, ratio=ratio)
            if k == "N2_over_N":
                continue  # 1/2 by symmetry at Delta = 0: no gap to resolve
            if m == "3d" or m == "npse":
                total += 1
                resolved += ratio < 0.1
            fmt = _FMT.get(k, "{:.6f}")
            lines.append(
                f"  {k:>10s}{m:>7s}{fmt.format(a):>16s}{fmt.format(b):>16s}{err:>12.1e}{gap:>13.1e}{ratio:>9.2g}"
            )
    for m in MODELS:
        t64, t128 = obs64[m]["seconds"], obs128[m]["seconds"]
        lines.append(
            f"  {m:>5s}: complex128 {t128:.0f} s, {obs128[m]['steps']} steps, residual {obs128[m]['residual']:.1e}; "
            f"complex64 {t64:.0f} s, {obs64[m]['steps']} steps, residual {obs64[m]['residual']:.1e}; "
            f"time ratio {t128 / max(t64, 1e-9):.1f}x"
        )
    verdict = (
        f"  complex64 resolves the 3D - NPSE difference (error < 10 % of it) in {resolved} of {total} "
        "observable/model pairs (3d and npse)"
    )
    lines.append(verdict)
    return "\n".join(lines), dict(table=numbers, resolved=resolved, pairs=total)


def save_fields(res: dict, path: str) -> None:
    """Column densities of the three models, the two width maps and the axes (float32, compressed)."""
    g = res["grid2"]
    np.savez_compressed(
        path,
        x=g.x64[0].numpy(),
        y=g.x64[1].numpy(),
        **{f"n_{m}": res["models"][m]["n"].astype(np.float32) for m in MODELS},
        width_3d=res["width_3d"].astype(np.float32),
        width_npse=res["width_npse"].astype(np.float32),
    )


def plot_comparison(res: dict, path: str, title: str = "") -> None:
    """Column density (3D), its deviation from each 2D model, the two width maps and a cut along y = 0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = res["grid2"]
    x = g.x64[0].numpy()
    ext = (x[0], x[-1], x[0], x[-1])
    n3 = res["models"]["3d"]["n"].sum(axis=0)
    nn = res["models"]["npse"]["n"].sum(axis=0)
    nf = res["models"]["fixed"]["n"].sum(axis=0)
    step = max(1, n3.shape[0] // 800)  # keep the image near 800 px

    def show(ax, a, ttl, cmap="viridis", sym=False):
        a = a[::step, ::step].T
        v = float(np.abs(a).max()) if sym else None
        im = ax.imshow(a, origin="lower", extent=ext, cmap=cmap, vmin=-v if sym else None, vmax=v)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("x (a)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    show(axs[0, 0], n3, "3D column density n (atoms / a$^2$)")
    show(axs[0, 1], (n3 - nn) / nn.max(), "(3D - NPSE) / max n", "RdBu_r", sym=True)
    show(axs[0, 2], (n3 - nf) / nf.max(), "(3D - fixed g) / max n", "RdBu_r", sym=True)
    occ = n3 > 1e-3 * n3.max()
    show(axs[1, 0], np.where(occ, res["width_3d"], np.nan), r"3D $\langle z^2\rangle / (l_z^2/2)$", "magma")
    show(axs[1, 1], np.where(occ, res["width_npse"], np.nan), "NPSE $s^2$", "magma")
    ax = axs[1, 2]
    j = n3.shape[1] // 2
    for a, lab in ((n3, "3D"), (nn, "NPSE"), (nf, "fixed $g_{2D}$")):
        ax.plot(x, a[:, j], lw=1, label=lab)
    ax.set_xlabel("x (a), y = 0")
    ax.set_ylabel("n")
    ax.legend(fontsize=8)
    ax.set_title("cut through the AA point", fontsize=9)
    import textwrap

    fig.suptitle(textwrap.fill(title, 140), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
