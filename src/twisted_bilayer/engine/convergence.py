"""In-plane resolution: how many grid points per lattice constant the 2D problem needs.

Two systems, each solved at several in-plane resolutions and compared with the finest one:

  supercell   the homogeneous supercell (`sweep.coupling_scan`) at points_per_a in a list, for several
              lattice depths and couplings: mu, E/N, the superfluid fraction by twist energy and by current,
              the raw moire contrast, the participation ratio, N2/N
  trapped     the trapped cloud (`trapped.trapped_ground_state`, 2D NPSE) at cell_points per supercell in a
              list: mu, E/N, the in-situ contrasts C_raw and C_coh, n_AA, n_AB, R_rms, n_peak

Both run in complex128 so that the single-precision floor (~1e-5 in the residual) does not hide the
discretisation error. The grid represents the potential exactly (V1 and V2 are band-limited), so the error
comes from the wavefunction's harmonics beyond the grid's Nyquist wavenumber pi/dx; they decay
exponentially, faster in shallow lattices. The superfluid fraction is the most sensitive observable: it is
an energy difference of order theta^2 / L^2 (see `engine.observables`).

`needed` reports, per observable, the coarsest resolution whose difference from the finest is below a
tolerance (absolute, in the observable's units), and checks that every finer resolution also is. Only
converged rows count: a row whose solve stopped above its residual tolerance (`converged` False) ends the
walk, because an unconverged state can land near the finest value by chance; if the finest row itself is
unconverged nothing is reported as needed. Rows without a `converged` entry (records written before it
existed) count as converged.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from . import trapped as TR
from .backend import synchronize

SUPERCELL_KEYS = (
    "mu",
    "E_per_N",
    "fs_xx_energy",
    "fs_xx_current",
    "contrast_raw",
    "participation_ratio",
    "N2_frac",
)
TRAPPED_KEYS = ("mu", "E_per_N", "C_raw", "C_coh", "n_AA", "n_AB", "R_rms", "n_peak")

# default tolerances (absolute): energies in E_R, contrasts and fractions dimensionless, R_rms in a
TOLERANCES = dict(
    mu=1e-6,
    E_per_N=1e-6,
    fs_xx_energy=1e-4,
    fs_xx_current=1e-4,
    contrast_raw=1e-4,
    participation_ratio=1e-4,
    N2_frac=1e-6,
    C_raw=1e-4,
    C_coh=1e-4,
    n_AA=1e-4,
    n_AB=1e-4,
    R_rms=1e-4,
    n_peak=1e-4,
)


def supercell_convergence(
    V0_list,
    Omega_R,
    ppa_list,
    m: int = 22,
    n: int = 1,
    filling: float = 1.0,
    vertical: str = "npse",
    tol: float = 1e-9,
    theta: float = 0.3,
    verbose: bool = False,
) -> dict:
    """The homogeneous supercell at each points_per_a (the grid rounds up to an FFT-friendly size, so the
    actual points per a are reported). Returns dict(V0 -> list of rows), each row dict(ppa, N, ppa_actual,
    seconds, values = {key: (n_Omega,) array}), plus Omega_R."""
    from .sweep import SweepSpec, coupling_scan

    out = {}
    for V0 in V0_list:
        rows = []
        for ppa in sorted(ppa_list):
            spec = SweepSpec(
                V0=float(V0),
                Omega_R=tuple(Omega_R),
                m=m,
                n=n,
                filling=filling,
                points_per_a=float(ppa),
                vertical=vertical,
                theta=theta,
                tol=tol,
                dtype_relax="complex128",
                dtype_polish="complex128",
                save_fields=False,
            )
            t0 = time.perf_counter()
            res = coupling_scan(spec, verbose=False)
            synchronize()
            sec = time.perf_counter() - t0
            row = dict(
                ppa=float(ppa),
                N=int(res.N_grid),
                ppa_actual=res.N_grid / res.geometry.L_box,
                seconds=sec,
                residual=float(np.max(res.columns["residual_0"])),
                converged=bool(np.all(res.columns["converged"])),
                values={k: np.asarray(res.columns[k], dtype=float) for k in SUPERCELL_KEYS},
            )
            rows.append(row)
            if verbose:
                print(
                    f"  V0 = {V0:g}, {ppa:g} points per a (N = {row['N']}): residual {row['residual']:.1e}, "
                    f"{sec:.0f} s",
                    flush=True,
                )
        out[float(V0)] = rows
    return dict(
        rows=out, Omega_R=tuple(float(o) for o in Omega_R), m=m, n=n, filling=filling, vertical=vertical, tol=tol
    )


def trapped_convergence(spec: TR.TrappedSpec, cell_points_list, verbose: bool = False) -> dict:
    """The trapped cloud (2D NPSE, complex128) at each cell_points (points per supercell side). Returns
    dict(rows = list of dict(cell_points, ppa, N, seconds, residual, steps, values = {key: float}))."""
    _, geo = TR.supercell_grid(spec)
    rows = []
    for cp in sorted(int(c) for c in cell_points_list):
        s = replace(spec, cell_points=cp, dtype="complex128", vertical="npse")
        t0 = time.perf_counter()
        run = TR.trapped_ground_state(s, verbose=False)
        synchronize()
        sec = time.perf_counter() - t0
        eng, psi, r = run["engine"], run["psi"], run["result"]
        terms = eng.energy_terms(psi)
        cloud = TR.cloud_numbers(eng, psi, geo.L_box)
        ins = TR.insitu_observables(eng, psi, mn=(s.m, s.n))
        values = dict(
            mu=float(r.mu[0]),
            E_per_N=float(terms["E_total"][0] / terms["norm"][0]),
            C_raw=ins["C_raw"],
            C_coh=ins["C_coh"],
            n_AA=ins["n_AA"],
            n_AB=ins["n_AB"],
            R_rms=cloud["R_rms"],
            n_peak=cloud["n_peak"],
        )
        rows.append(
            dict(
                cell_points=cp,
                ppa=cp / geo.L_box,
                N=int(run["grid"].N[0]),
                seconds=sec,
                residual=float(r.residual[0]),
                converged=bool(r.converged[0]),
                steps=int(r.steps),
                box_cells=int(run["spec"].box_cells),
                edge_density_rel=float(cloud["edge_density_rel"]),
                values=values,
            )
        )
        if verbose:
            print(
                f"  trapped, {cp} points per supercell ({cp / geo.L_box:.2f} per a, {run['grid'].N[0]}^2): "
                f"mu = {values['mu']:.10f}, C_raw = {values['C_raw']:.6f}, residual {float(r.residual[0]):.1e}, "
                f"{sec:.0f} s",
                flush=True,
            )
        del run, eng, psi, r
    return dict(rows=rows, spec=spec)


_W = max(len(k) for k in SUPERCELL_KEYS) + 2  # column width of the supercell table


def _maxabs(x) -> float:
    return float(np.max(np.abs(np.asarray(x, dtype=float))))


def differences(rows, keys) -> list[dict]:
    """Per row: max over couplings of |value - value at the finest resolution| for each key."""
    fin = rows[-1]["values"]
    return [{k: _maxabs(np.asarray(r["values"][k]) - np.asarray(fin[k])) for k in keys} for r in rows]


def _converged(row) -> bool:
    return bool(row.get("converged", True))


def needed(rows, keys, res_key: str, tolerances: dict | None = None) -> dict:
    """Per key: the coarsest resolution (rows[i][res_key]) from which every finer one is converged and
    within the tolerance of the finest; None if only the finest itself qualifies (not converged within the
    scan) or the finest row is not converged."""
    tol = dict(TOLERANCES, **(tolerances or {}))
    diffs = differences(rows, keys)
    out = {}
    for k in keys:
        ok_from = None
        if _converged(rows[-1]):
            for i in range(len(rows) - 2, -1, -1):  # walk down from the second finest
                if _converged(rows[i]) and diffs[i][k] <= tol[k]:
                    ok_from = rows[i][res_key]
                else:
                    break
        out[k] = ok_from
    return out


def unconverged(rows, res_key: str) -> list:
    """The resolutions of the rows whose solve stopped above its residual tolerance."""
    return [r[res_key] for r in rows if not _converged(r)]


def _needed_text(rows, nd, k, res_key: str, fmt=str) -> str:
    """'<resolution>' if found; otherwise why not: '> <second finest>' when an observable is not converged in
    resolution, 'blocked by an unconverged row at <r>' when the walk stopped at an unconverged row."""
    if nd[k] is not None:
        return fmt(nd[k])
    if not _converged(rows[-1]):
        return f"none (the finest row, {fmt(rows[-1][res_key])}, is not converged)"
    if len(rows) > 1 and not _converged(rows[-2]):
        return f"blocked by an unconverged row at {fmt(rows[-2][res_key])}"
    return ">" + fmt(rows[-2][res_key]) if len(rows) > 1 else "-"


def to_record(conv_sc: dict | None, conv_tr: dict | None, tolerances: dict | None = None) -> dict:
    """Everything of both scans for a JSON file: the rows with their values, and per scan the differences
    from the finest resolution and the resolution each observable needs."""
    tol = dict(TOLERANCES, **(tolerances or {}))
    out = dict(tolerances=tol, supercell=None, trapped=None)
    if conv_sc:
        out["supercell"] = dict(
            m=conv_sc["m"],
            n=conv_sc["n"],
            filling=conv_sc["filling"],
            vertical=conv_sc["vertical"],
            tol=conv_sc.get("tol"),
            Omega_R=list(conv_sc["Omega_R"]),
            by_V0={
                f"{V0:g}": dict(
                    rows=[dict(r, values={k: np.asarray(v).tolist() for k, v in r["values"].items()}) for r in rows],
                    diff_from_finest=differences(rows, SUPERCELL_KEYS),
                    needed_ppa=needed(rows, SUPERCELL_KEYS, "ppa", tolerances),
                    unconverged_ppa=unconverged(rows, "ppa"),
                )
                for V0, rows in conv_sc["rows"].items()
            },
        )
    if conv_tr:
        rows = conv_tr["rows"]
        out["trapped"] = dict(
            spec=conv_tr["spec"],
            rows=rows,
            diff_from_finest=differences(rows, TRAPPED_KEYS),
            needed_cell_points=needed(rows, TRAPPED_KEYS, "cell_points", tolerances),
            unconverged_cell_points=unconverged(rows, "cell_points"),
        )
    return out


def table_rows(conv_sc: dict | None, conv_tr: dict | None) -> tuple[list[dict], list[dict]]:
    """Flat tables for CSV files: one row per (V0, points per a, Omega_R) of the supercell scan, and one
    row per cell_points of the trapped scan."""
    sc, tr = [], []
    if conv_sc:
        for V0, rows in conv_sc["rows"].items():
            for r in rows:
                for j, om in enumerate(conv_sc["Omega_R"]):
                    sc.append(
                        dict(V0=V0, Omega_R=om, ppa=r["ppa"], ppa_actual=r["ppa_actual"], N=r["N"])
                        | {k: float(np.asarray(r["values"][k])[j]) for k in SUPERCELL_KEYS}
                        | dict(residual=r["residual"], seconds=r["seconds"])
                    )
    if conv_tr:
        for r in conv_tr["rows"]:
            tr.append(
                dict(cell_points=r["cell_points"], ppa=r["ppa"], N=r["N"])
                | dict(r["values"])
                | dict(residual=r["residual"], steps=r["steps"], seconds=r["seconds"])
            )
    return sc, tr


def report(conv_sc: dict | None, conv_tr: dict | None, tolerances: dict | None = None) -> str:
    tol = dict(TOLERANCES, **(tolerances or {}))
    lines = []
    if conv_sc:
        lines.append(
            f"supercell m = {conv_sc['m']}, n = {conv_sc['n']}, filling {conv_sc['filling']:g}, {conv_sc['vertical']}; "
            f"Omega_R = {conv_sc['Omega_R']}; max over Omega_R of |value - finest|"
        )
        for V0, rows in conv_sc["rows"].items():
            diffs = differences(rows, SUPERCELL_KEYS)
            lines.append(f"  V0 = {V0:g} E_R")
            lines.append(f"  {'pts/a':>7s}{'N':>6s}" + "".join(f"{k:>{_W}s}" for k in SUPERCELL_KEYS) + f"{'s':>6s}")
            for r, d in zip(rows, diffs):
                lines.append(
                    f"  {r['ppa_actual']:>7.2f}{r['N']:>6d}"
                    + "".join(f"{d[k]:>{_W}.1e}" for k in SUPERCELL_KEYS)
                    + f"{r['seconds']:>6.0f}"
                    + ("" if _converged(r) else f"  NOT CONVERGED (residual {r['residual']:.1e})")
                )
            nd = needed(rows, SUPERCELL_KEYS, "ppa", tolerances)
            if not _converged(rows[-1]):
                lines.append("  the finest row is not converged: no resolution is reported as needed")
            lines.append(
                "  needed (pts/a, nominal): "
                + ", ".join(f"{k} {_needed_text(rows, nd, k, 'ppa', lambda v: format(v, 'g'))}" for k in SUPERCELL_KEYS)
            )
    if conv_tr:
        rows = conv_tr["rows"]
        diffs = differences(rows, TRAPPED_KEYS)
        s = conv_tr["spec"]
        b = rows[-1].get("box_cells", s.box_cells)
        lines.append(f"trapped cloud: box {b}x{b}, N = {s.N_atoms:g}, nu_r = {s.nu_r_hz:g} Hz, NPSE; |value - finest|")
        lines.append(
            f"  {'cell':>6s}{'pts/a':>7s}{'N':>6s}" + "".join(f"{k:>11s}" for k in TRAPPED_KEYS) + f"{'s':>6s}"
        )
        for r, d in zip(rows, diffs):
            lines.append(
                f"  {r['cell_points']:>6d}{r['ppa']:>7.2f}{r['N']:>6d}"
                + "".join(f"{d[k]:>11.1e}" for k in TRAPPED_KEYS)
                + f"{r['seconds']:>6.0f}"
                + ("" if _converged(r) else f"  NOT CONVERGED (residual {r['residual']:.1e})")
            )
        if not _converged(rows[-1]):
            lines.append("  the finest row is not converged: no resolution is reported as needed")
        fin = rows[-1]["values"]
        lines.append("  finest: " + ", ".join(f"{k} {fin[k]:.8g}" for k in TRAPPED_KEYS))
        nd = needed(rows, TRAPPED_KEYS, "cell_points", tolerances)
        lines.append(
            "  needed (points per supercell): "
            + ", ".join(f"{k} {_needed_text(rows, nd, k, 'cell_points')}" for k in TRAPPED_KEYS)
        )
    lines.append("  tolerances: " + ", ".join(f"{k} {v:g}" for k, v in tol.items()))
    return "\n".join(lines)
