"""The scenario runners behind `twisted-bilayer run <scenario>`.

Each scenario kind is one function `run_<kind>(name, row)` that runs a row of `twisted_bilayer.scenarios`,
prints a pre-flight and a summary, and writes its files to `<out>/<name>/`. Adding a kind: one function
here, one entry in `KINDS`, and rows in a scenarios table.

Importing this module imports the engine, which fixes the device: `twisted_bilayer.cli` sets
TWISTED_BILAYER_FORCE_CPU before it imports this module when a run must stay on the CPU.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from twisted_bilayer.engine import bands, evolve
from twisted_bilayer.engine.backend import check_fft_grid_size, describe_device, synchronize
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, relax
from twisted_bilayer.engine.lattice import bilayer_potential, lattice_1d, moire_geometry, toy_bilayer_1d
from twisted_bilayer.engine.observables import (
    leggett_bounds,
    superfluid_fraction_current,
    superfluid_fraction_energy,
)
from twisted_bilayer.engine.physics import BilayerGPE, choose_dt
from twisted_bilayer.engine.units import KAPPA, lattice_units
from twisted_bilayer.records import write_json

VERBOSE = True  # progress output of the long runs
TOL_GATES = 0.01  # pass/fail tolerance on norm and energy drift of a real-time run
_OUT_ROOT = "results"  # set by run_scenario


def _section(title: str) -> None:
    print(f"\n-- {title} --")


def _outdir(name: str) -> str:
    d = os.path.join(_OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def _memory_note(numel: int, batch: int, components: int, dtype: str) -> str:
    bytes_per = 16 if dtype == "complex128" else 8
    field = numel * components * batch * bytes_per
    # the solver keeps ~6 fields alive at the polish peak (psi, v, psi_best, H psi, work)
    gib = 6 * field / 2**30
    note = f"~{gib:.2f} GiB at the solver's peak (estimate; one field = {field / 2**20:.0f} MiB)"
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if gib > 0.75 * total:
            note += f"  <-- WARNING: near the {total:.0f} GiB of this GPU; split the batch"
    return note


# ── coupling scan ────────────────────────────────────────────────────────────


def run_coupling_scan(name: str, row: dict) -> None:
    from twisted_bilayer.engine.sweep import SweepSpec, coupling_scan, plot_coupling_scan

    keys = {k: row[k] for k in row if k != "kind"}
    spec = SweepSpec(**keys)
    geo = moire_geometry(spec.m, spec.n)
    N = geo.grid_size(spec.points_per_a)
    _section("pre-flight")
    print(describe_device())
    print(geo.describe())
    w = check_fft_grid_size(N)
    if w:
        print(w)
    print(f"grid {N}x{N}, {N / geo.L_box:.2f} points per a; batch of {len(spec.Omega_R)} couplings x 3 twists")
    print(_memory_note(N * N, len(spec.Omega_R), 2, spec.dtype_polish))
    if spec.V0 >= 12 and N / geo.L_box < 16:
        print("WARNING: fewer than 16 points per a in a deep lattice (Wannier functions under-resolved)")
    _section("run")
    res = coupling_scan(spec, verbose=VERBOSE)
    out = _outdir(name)
    res.save(out)
    plot_coupling_scan(res, os.path.join(out, "coupling_scan.png"))
    _section("result")
    unconverged = [float(o) for o, c in zip(res.columns["Omega_R"], res.columns["converged"]) if not c]
    if unconverged:
        print(f"WARNING: not converged to tol at Omega_R = {unconverged}; residuals in the table")
    print(f"wrote {out}/coupling_scan.{{png,csv,npz,txt}} and run_metadata.json; total {res.timing['total_s']:.1f} s")


# ── the bridge: lattice mean field against the continuum GPE ───────────────────


def run_bridge(name: str, row: dict) -> None:
    """The discrete GPE of the two-layer Bose-Hubbard model (variant C, with and without the interband
    shift) against the continuum GPE on the same supercell at fixed g2d (manybody.bridge): mu, E/N from
    each model's band bottom, f_s, the AA/AB density ratio, N2/N and the participation ratio over V0 and
    Omega_R, as a table, a json and a figure of the differences. One gate: mu - E_bottom at V0 = 16,
    Omega_R = 0 within 5 %."""
    from twisted_bilayer.manybody.bridge import BridgeSpec, bridge, format_bridge, save_bridge

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    spec = BridgeSpec(**{k: row[k] for k in row if k != "kind"})
    geo = moire_geometry(spec.m, spec.n)
    N = geo.grid_size(spec.points_per_a)
    _section("pre-flight")
    print(describe_device())
    print(geo.describe())
    print(
        f"continuum: grid {N}x{N} ({N / geo.L_box:.2f} points per a), {len(spec.Omega_R)} couplings x 3 twists per V0, "
        f"{spec.dtype_relax} / {spec.dtype_polish}; lattice: {2 * (spec.m**2 + spec.n**2)} sites, complex128"
    )
    print(f"V0 = {spec.V0}, Omega_R = {spec.Omega_R}, filling {spec.filling}, nu_z = {spec.nu_z_hz:g} Hz")
    _section("run")
    res = bridge(spec, verbose=VERBOSE)
    out = _outdir(name)
    save_bridge(res, out)
    _section("result")
    print(format_bridge(res))
    print(f"wrote {out}/bridge.{{txt,json,png}}")


# ── thin-z check ───────────────────────────────────────────────────────────────


def run_thin_z(name: str, row: dict) -> None:
    from twisted_bilayer.engine.thin_z import thin_z_check

    keys = {k: row[k] for k in row if k != "kind"}
    _section("pre-flight")
    print(describe_device())
    geo = moire_geometry(keys["m"], keys["n"])
    N = geo.grid_size(keys["points_per_a"])
    print(f"3D grid {N}x{N}x{keys['Nz']}: " + _memory_note(N * N * keys["Nz"], 2, 2, "complex128"))
    _section("run")
    res = thin_z_check(verbose=VERBOSE, **keys)
    out = _outdir(name)
    np.savez_compressed(
        os.path.join(out, "thin_z.npz"),
        column_density=res.column_density,
        density_2d=res.density_2d,
        width_ratio=res.width_map,
    )
    with open(os.path.join(out, "thin_z.json"), "w") as f:
        json.dump(dict(numbers=res.numbers, timing=res.timing), f, indent=2)
    with open(os.path.join(out, "thin_z.txt"), "w") as f:
        f.write(res.report() + "\n")
    _section("result")
    print(res.report())
    print(f"wrote {out}/thin_z.{{txt,json,npz}}")


# ── Rabi dynamics ──────────────────────────────────────────────────────────────


def run_rabi(name: str, row: dict) -> None:
    """All atoms in |1> in the lattice-1 ground state; switch Omega_R on and
    watch N_2(t). The bare oscillation is sin^2(Omega_R t) (frequency
    2 Omega_R / 2 pi in E_R); in the lattice it is slower by the Wannier
    overlap and dephases across the moire cell. The paper measures 893 Hz at
    V0 = 4 E_R (Fig. 3c)."""
    from twisted_bilayer.gates.core import conservation_gates, report

    units = lattice_units()
    geo = moire_geometry(row["m"], row["n"])
    N = geo.grid_size(row["points_per_a"])
    dtype = torch.complex128 if row["dtype"] == "complex128" else torch.complex64
    grid = Grid(
        (N, N), (geo.L_box, geo.L_box), real_dtype=torch.float64 if dtype == torch.complex128 else torch.float32
    )
    V = bilayer_potential(grid, row["V0"], geo)
    n_particles = row["filling"] * grid.volume
    _section("pre-flight")
    print(describe_device())
    print(geo.describe())
    print(
        f"grid {N}x{N}; V0 = {row['V0']}, Omega_R = {row['Omega_R']} E_R = {units.ER_to_hz(row['Omega_R']):.0f} Hz, "
        f"t_total = {row['t_total']} E_R^-1 = {units.lattice_time_to_s(row['t_total']) * 1e3:.2f} ms"
    )
    _section("ground state (Omega_R = 0, all atoms in |1>)")
    eng0 = BilayerGPE(grid, V=V, g=units.g2d, Omega_R=0.0, delta=row["delta"], n_components=2, dtype=dtype)
    psi0 = initial_state(eng0, n_particles, fractions=(1.0, 0.0))
    r = relax(eng0, psi0, n_particles, tol=1e-8)
    print(r.report())
    _section("real-time evolution")
    eng = BilayerGPE(grid, V=V, g=units.g2d, Omega_R=row["Omega_R"], delta=row["delta"], n_components=2, dtype=dtype)
    order = row["splitstep_order"]
    dt = choose_dt(eng, r.psi, order=order, order4_multiplier=row["order4_dt_multiplier"] if order == 4 else 1.0)
    n_frames = row["n_frames"]
    steps_per_frame = max(1, int(math.ceil(row["t_total"] / n_frames / dt)))
    dt = row["t_total"] / (n_frames * steps_per_frame)
    print(f"dt = {dt:.3e} ({steps_per_frame} steps per frame, {n_frames} frames, order {order})")
    t0 = time.time()
    history, psi = evolve.run_and_record(eng, r.psi, dt, order, n_frames, steps_per_frame, progress=VERBOSE)
    synchronize()
    print(f"evolution: {time.time() - t0:.1f} s")
    t = np.array(history["t"])
    frac2 = np.array(history["N2"]) / np.array(history["norm"])
    # Effective Rabi frequency from the FIRST maximum of N2(t) (sin^2(Omega_eff t)
    # peaks at t = pi/(2 Omega_eff)); an FFT over a few periods has a bin width
    # ~1/t_total, too coarse here. Expected: Omega_eff/Omega_R = Franck-Condon
    # factor |c_00|^2 of the q = 0 Bloch state (0.8205 at V0 = 4 E_R), the k = 0
    # overlap of the two rotated lattices' band-bottom states.
    i = int(np.argmax(frac2[: max(3, len(frac2) // 2)]))
    t_max = t[i]
    if 0 < i < len(frac2) - 1:
        y0, y1, y2 = frac2[i - 1], frac2[i], frac2[i + 1]
        curv = y0 - 2 * y1 + y2
        if curv < -1e-9:  # parabolic refinement of the maximum
            t_max = t[i] + 0.5 * (t[1] - t[0]) * (y0 - y2) / curv
    om_eff = math.pi / (2.0 * t_max)
    f_peak, f_bare = om_eff / math.pi, row["Omega_R"] / math.pi
    print(
        f"N2/N first maximum at t = {t_max:.3f} ({units.lattice_time_to_s(t_max) * 1e3:.3f} ms), N2/N = {frac2[i]:.3f}: "
        f"Omega_eff = {om_eff:.4f} E_R = {units.ER_to_hz(om_eff):.0f} Hz; Omega_eff/Omega_R = {om_eff / row['Omega_R']:.3f} "
        f"(Franck-Condon |c_00|^2 = {bands._bloch_1d(row['V0'], np.array([0.0]), 31)[1][0, 15, 0] ** 4:.4f})"
    )
    results = conservation_gates(history, tol_norm=TOL_GATES, tol_energy=TOL_GATES)
    print(report(results, header=f"engine / {name}"))
    out = _outdir(name)
    with open(os.path.join(out, "rabi.csv"), "w") as f:
        f.write("t,t_ms,N2_frac,E_total,norm\n")
        for i in range(len(t)):
            f.write(
                f"{t[i]},{units.lattice_time_to_s(t[i]) * 1e3},{frac2[i]},{history['E_total'][i]},{history['norm'][i]}\n"
            )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.plot(units.lattice_time_to_s(t) * 1e3, frac2, color="#0072B2", lw=1.5, label="lattice, GPE")
        ax.plot(
            units.lattice_time_to_s(t) * 1e3,
            np.sin(row["Omega_R"] * t) ** 2,
            color="#999999",
            lw=1.0,
            ls="--",
            label=r"bare $\sin^2(\Omega_R t)$",
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(r"$N_2/N$")
        ax.set_title(
            f"Rabi transfer in the bilayer lattice, $V_0={row['V0']}E_R$, "
            f"$\\Omega_R={row['Omega_R']}E_R$ ({units.ER_to_hz(row['Omega_R']):.0f} Hz)",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "rabi.png"), dpi=150)
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})")
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(
            dict(row=row, dt=dt, f_peak=f_peak, f_bare=f_bare, gates=[g.__dict__ for g in results]),
            f,
            indent=2,
            default=str,
        )
    print(f"wrote {out}/rabi.{{csv,png}}")


# ── 1D toy regression ──────────────────────────────────────────────────────────


def run_toy1d(name: str, row: dict) -> None:
    """The 1D reference numbers: exact J1, the single-component
    f_s at g nbar = 0.2 (energy = current = Leggett identity), and the
    two-lattice toy's f_s versus Omega_R."""
    V0, gn, th = row["V0"], row["g_nbar"], row["theta"]
    _section("1D band by twist")
    g1 = Grid((int(row["points_per_a"]),), (1.0,))
    qs = torch.linspace(-math.pi, math.pi, 33)[:-1]
    eng = BilayerGPE(g1, V=lattice_1d(g1, V0), n_components=1, twist=qs.reshape(-1, 1))
    r = relax(eng, initial_state(eng, 1.0, batch=len(qs)), 1.0, tol=row["tol"])
    J1 = -(r.energy * torch.cos(qs)).mean()
    print(f"J1 = {float(J1):.6f} E_R (bands.hopping_1d: {bands.hopping_1d(V0, 1)[0]:.6f}; 0.0308 at V0 = 8)")
    _section(f"single component, g nbar = {gn:g}")
    eng = BilayerGPE(g1, V=lattice_1d(g1, V0), g=gn, n_components=1, twist=torch.tensor([[0.0], [th]]))
    r = relax(eng, initial_state(eng, 1.0, batch=2), 1.0, tol=row["tol"])
    fsE = float(superfluid_fraction_energy(r.energy[1], r.energy[0], torch.tensor(1.0), 1.0, th, KAPPA))
    fsI = float(superfluid_fraction_current(eng, r.psi, 0, th)[1])
    lo, _ = leggett_bounds(eng.density(r.psi[0:1]), g1, 0)
    print(
        f"f_s: energy {fsE:.5f}, current {fsI:.5f}, Leggett identity {float(lo):.5f} "
        f"(0.2977 / 0.2975 / 0.2975 is the toy at gn = 0.1); m/m* exact band {bands.effective_mass_ratio(V0):.5f}"
    )
    _section(f"two-lattice toy, {row['n_sites']} sites, Omega_R scan")
    ns = row["n_sites"]
    gt = Grid((int(row["points_per_a"]) * ns,), (float(ns),))
    period2 = float(ns) / (ns - 1)
    V0_2 = bands.depth_for_equal_hopping(V0, period2)  # equal J in both lattices
    delta = bands.band_bottom_1d(V0_2, period2) - bands.band_bottom_1d(V0)  # resonant s-bands
    print(f"lattice 2: period {period2:.4f} a, depth {V0_2:.4f} E_R for equal J; detuning Delta = {delta:+.5f} E_R")
    V = toy_bilayer_1d(gt, V0, ns, V0_2=V0_2)
    omegas = (
        torch.tensor([0.0, 0.5, 1.0, 3.0, 10.0, 35.0]) * float(J1) / 2.0
    )  # toy's Omega/J (Omega = 2 Omega_R) -> Omega_R
    eng = BilayerGPE(gt, V=V, g=gn, Omega_R=omegas, delta=delta, n_components=2)
    r0 = relax(eng, initial_state(eng, float(ns), batch=len(omegas)), float(ns), tol=row["tol"])
    engt = BilayerGPE(
        gt,
        V=V,
        g=gn,
        Omega_R=omegas,
        delta=delta,
        n_components=2,
        twist=torch.full((len(omegas), 1), th, dtype=torch.float64),
    )
    rt = relax(engt, r0.psi, float(ns), tol=row["tol"])
    fsE = superfluid_fraction_energy(rt.energy, r0.energy, r0.energy.new_tensor(float(ns)), float(ns), th, KAPPA)
    fsI = superfluid_fraction_current(engt, rt.psi, 0, th)
    lo, _ = leggett_bounds(eng.density(r0.psi), gt, 0)
    print(
        f"{'Omega/J':>14s} {'Omega_R':>9s} {'E/N':>10s} {'mu':>10s} {'fs_energy':>10s} {'fs_current':>11s} {'Leggett':>9s}"
    )
    rows = []
    for i, om in enumerate(omegas):
        rows.append(
            (
                2 * float(om) / float(J1),
                float(om),
                float(r0.energy[i]) / ns,
                float(r0.mu[i]),
                float(fsE[i]),
                float(fsI[i]),
                float(lo[i]),
            )
        )
        print(
            f"{rows[-1][0]:14.2f} {rows[-1][1]:9.5f} {rows[-1][2]:10.5f} {rows[-1][3]:10.5f} "
            f"{rows[-1][4]:10.5f} {rows[-1][5]:11.5f} {rows[-1][6]:9.5f}"
        )
    out = _outdir(name)
    with open(os.path.join(out, "toy1d.csv"), "w") as f:
        f.write("Omega_over_J1,Omega_R,E_per_N,mu,fs_energy,fs_current,leggett\n")
        for rr in rows:
            f.write(",".join(f"{x}" for x in rr) + "\n")
    print(f"wrote {out}/toy1d.csv")


# ── two-layer tight binding ───────────────────────────────────────────

MODEL_C_VARIANTS = {
    "A": ("one neighbour: S cut at |d| < 1, lattice-1 frame, no interlayer interaction", {}),
    "B": ("S cut at |d| < 2", {"cutoff": 2.0}),
    "C": ("S cut at |d| < 2 + interlayer interaction V_ij", {"cutoff": 2.0, "interlayer_interaction": True}),
}


def run_tight_binding(name: str, row: dict) -> None:
    """The two-layer tight-binding model (engine.lattice_model) on the supercell over Omega_R: for each
    coupling the mean-field (discrete GPE) ground state, E/N and mu (relative to the on-site energy 2 eps_1D), f_s^xx by twist
    energy and by current, the PR over sites, the AA/AB density ratio, the layer-2 fraction, residuals
    and the lowest constrained Hessian eigenvalue (> 0: a minimum). Variants:
      A  the one-neighbour model (S cut at |d| < 1, lattice-1 frame, no interlayer interaction)
         -- the reference table;
      B  S cut at |d| < 2 (removes the cusp the |d| = 1 cutoff puts into the k = 0 coupling next to AA);
      C  B plus the interlayer density-density interaction V_ij (a12 = a11, as in the continuum model).
    Also the single-particle k = 0 levels against the continuum (bands.moire_bands_q0) at compare_m."""
    from twisted_bilayer.engine import lattice_model as lm

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    t_all = time.time()
    V0, m, n, filling, theta = row["V0"], row["m"], row["n"], row["filling"], row["theta"]
    g2d, g2d_source = lm.resolve_g2d(row.get("g2d"), row.get("nu_z_hz"))
    g2d_note = {
        "explicit": "given by the row",
        "nu_z": f"from nu_z = {row.get('nu_z_hz')} Hz (lattice_model.g2d_from_nu_z)",
        "default": "the shipped lattice_model.G2D_DEFAULT",
    }[g2d_source]
    J = bands.hopping_1d(V0, 2)
    U = bands.onsite_U(V0, g2d)
    geo = lm.supercell_sites(m, n).geometry
    head = [
        f"Two-layer tight binding, V0 = {V0} E_R, filling {filling} atoms/a^2 (both layers), "
        f"g2d = {g2d:.7f} E_R a^2 ({g2d_note})",
        f"  {geo.describe()}; {2 * (m**2 + n**2)} sites",
        f"  J1 = {J[0]:.6f}, J2 = {J[1]:.3e}, U = {U:.6f} E_R (U/J1 = {U / J[0]:.2f}); "
        f"twist theta = {theta} across the box",
        f"  E/N and mu relative to the on-site energy 2 eps_1D = {lm.onsite_energy(V0):.6f} E_R; "
        f"f_s by energy (E) and current (I); PR over the sites of both layers;",
        "  AA/AB = total density in a 1 a disc at the AA points / at the AB points; hess_min > 0: local minimum",
        "",
    ]
    cols = [
        "Omega_R",
        "Omega_R/J1",
        "E/N",
        "mu",
        "fs_xx_E",
        "fs_xx_I",
        "PR",
        "AA/AB",
        "N2/N",
        "res_0",
        "res_theta",
        "hess_min",
        "time_s",
    ]
    fmt = [
        "{:7.3f}",
        "{:10.3f}",
        "{:10.6f}",
        "{:10.6f}",
        "{:9.6f}",
        "{:9.6f}",
        "{:7.4f}",
        "{:10.4g}",
        "{:7.5f}",
        "{:8.1e}",
        "{:9.1e}",
        "{:9.2e}",
        "{:6.1f}",
    ]
    width = [7, 10, 10, 10, 9, 9, 7, 10, 7, 8, 9, 9, 6]
    lines = list(head)
    for key in row["variants"]:
        t0 = time.time()
        title, kw = MODEL_C_VARIANTS[key]
        block = [f"Model {key}: {title}", "  " + " ".join(f"{c:>{w}s}" for c, w in zip(cols, width))]
        for Om in row["Omega_R"]:
            t1 = time.time()
            r = lm.superfluid_fraction(V0, Om, m, n, filling, g2d=g2d, theta=theta, **kw)
            g = r["gs0"]
            vals = [
                Om,
                Om / J[0],
                g.E_per_N,
                g.mu,
                r["fs_energy"],
                r["fs_current"],
                g.participation_ratio,
                g.AA_AB_ratio,
                g.layer_fractions[1],
                g.residual,
                r["gs_theta"].residual,
                g.hessian_min,
                time.time() - t1,
            ]
            block.append("  " + " ".join(f.format(v) for f, v in zip(fmt, vals)))
        lines += block + [f"  ({time.time() - t0:.0f} s)", ""]
        print("\n".join(block), flush=True)
    out = _outdir(name)
    lines.append(f"total {time.time() - t_all:.0f} s on CPU ({torch.get_num_threads()} threads)")
    with open(os.path.join(out, "table.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    cm = row.get("compare_m", 0)
    if cm:
        sp = [
            f"Single-particle k = 0 levels, tight binding vs continuum (moire_bands_q0, 8 pts/a), "
            f"V0 = {V0}, m = {cm}, n = {n}; energies in E_R",
            "  TB absolute = TB + on-site energy 2 eps_1D; 'shift' = E_0 - (decoupled band bottom)",
        ]
        np.set_printoptions(precision=5, suppress=True, linewidth=160)
        for cutoff in (1.0, 2.0):
            for Om in (0.24, 1.0):
                t0 = time.time()
                r = lm.compare_with_continuum(V0, Om, cm, n, n_bands=10, cutoff=cutoff)
                sp += [
                    f"  Omega_R = {Om}, S cutoff {cutoff:g} ({time.time() - t0:.0f} s)",
                    f"    spacing TB   {r['spacing_tb']}",
                    f"    spacing cont {r['spacing_cont']}",
                    f"    max |spacing difference| {r['spacing_maxdiff']:.2e}; "
                    f"shift TB {r['shift_tb']:.6f} vs cont {r['shift_cont']:.6f}; "
                    f"max |absolute difference| {r['abs_maxdiff']:.2e}",
                ]
        print("\n".join(sp))
        with open(os.path.join(out, f"single_particle_m{cm}.txt"), "w") as fh:
            fh.write("\n".join(sp) + "\n")
    print(f"wrote {out}/table.txt ({time.time() - t_all:.0f} s total)")


# ── Bogoliubov-de Gennes ───────────────────────────────────────────────────────


def run_bdg(name: str, row: dict) -> None:
    """For each Omega_R: the ground state, then the lowest `modes` BdG modes at Gamma, X = (pi/L, 0),
    M = (pi/L, pi/L) with participation ratio, spin weight and depletion; the spin branch (lowest mode
    with spin weight > 0.5 among `spin_modes` at Gamma) against 2 Omega_R x the layer overlap
    S = int psi1 psi2 / sqrt(N1 N2); the sound speed along x from the lowest density branch at
    q = 2 pi j / (12 L), j = 1..3 (fit w/q = c + b q^2); and the check c^2 = 2 kappa f_s n dmu/dn with
    f_s^xx from the twist current (theta across the box) and dmu/dn from relaxations at N (1 +- 2 %)."""
    from twisted_bilayer.engine.bdg import BdG, bdg_bands, sound_speed
    from twisted_bilayer.engine.npse import NPSE2D

    torch.set_num_threads(2)
    t_all = time.time()
    u = lattice_units()
    geo = moire_geometry(row["m"], 1)
    N = geo.grid_size(row["points_per_a"])
    L = geo.L_box
    grid = Grid((N, N), (L, L), device=torch.device("cpu"))
    V = bilayer_potential(grid, row["V0"], geo)
    Np = row["filling"] * grid.volume
    nbar = row["filling"]
    inter = NPSE2D(u.g2d, u.omega_z) if row["vertical"] == "npse" else None
    print(f"BdG: {geo.describe()}")
    print(
        f"  {grid.describe()}, V0 = {row['V0']} E_R, filling {row['filling']} (N = {Np:.2f}), "
        f"{inter.describe() if inter else f'contact g2d = {u.g2d:.5f}'}"
    )

    def engine(Om, twist=None):
        return BilayerGPE(
            grid, V, g=0.0 if inter else u.g2d, Omega_R=Om, n_components=2, twist=twist, interaction=inter
        )

    def ground(eng, n_atoms, start):
        r = relax(eng, start, n_atoms, tol=row["tol"], polish_steps=20000)
        if not bool(r.converged.all()):
            print(f"  warning: relax stopped at residual {float(r.residual.max()):.1e}")
        return r

    for Om in row["Omega_R"]:
        t0 = time.time()
        eng = engine(Om)
        r0 = ground(eng, Np, initial_state(eng, Np))
        psi = r0.psi
        p1, p2 = psi[0, 0], psi[0, 1]
        S = float(torch.abs((p1.conj() * p2).sum()) / torch.sqrt((p1.abs() ** 2).sum() * (p2.abs() ** 2).sum()))
        print(
            f"\n=== Omega_R = {Om} E_R: mu = {float(r0.mu[0]):.6f}, residual {float(r0.residual[0]):.1e}, "
            f"layer overlap S = {S:.4f} ({time.time() - t0:.1f} s)"
        )
        t1 = time.time()
        qs = {"Gamma": (0.0, 0.0), "X": (math.pi / L, 0.0), "M": (math.pi / L, math.pi / L)}
        b = bdg_bands(eng, psi, list(qs.values()), n_modes=row["modes"])
        print(
            f"  BdG at Gamma, X, M: iterations {[int(i) for i in b['iterations']]}, max residual "
            f"{b['residual'].max():.1e} ({time.time() - t1:.1f} s)"
        )
        for i, qname in enumerate(qs):
            print(f"  {qname}:")
            print("    " + b["results"][i].table().replace("\n", "\n    "))
        t1 = time.time()
        rs = BdG(eng, psi).solve(row["spin_modes"])
        sel = torch.nonzero(rs.spin_weight > 0.5).reshape(-1)
        if len(sel):
            i = int(sel[0])
            print(
                f"  spin branch at Gamma: mode {i} of {row['spin_modes']}, w = {float(rs.omega[i]):.5f} "
                f"(spin weight {float(rs.spin_weight[i]):.3f}, local {float(rs.local_spin_weight[i]):.3f}, "
                f"PR {float(rs.participation_ratio[i]):.3f}); "
                f"2 Omega_R = {2 * Om:.4f}, 2 Omega_R S = {2 * Om * S:.4f} ({time.time() - t1:.1f} s)"
            )
        else:
            print(
                f"  spin branch at Gamma: above w = {float(rs.omega[-1]):.5f} (lowest {row['spin_modes']} "
                f"modes are density-like); 2 Omega_R S = {2 * Om * S:.4f}"
            )
        t1 = time.time()
        ss = sound_speed(eng, psi, L=12.0 * L, n_points=3, n_modes=4)
        etw = engine(Om, twist=(row["theta"], 0.0))
        rtw = ground(etw, Np, psi.clone())
        fs = float(superfluid_fraction_current(etw, rtw.psi, 0, row["theta"])[0])
        mus = []
        for f in (0.98, 1.02):
            mus.append(float(ground(eng, Np * f, psi.clone() * math.sqrt(f)).mu[0]))
        dmu_dn = (mus[1] - mus[0]) / (0.04 * nbar)
        c_fs = math.sqrt(2.0 * KAPPA * fs * nbar * dmu_dn)
        print(
            f"  sound speed: q = {np.array2string(ss['q'], precision=4)}, w = "
            f"{np.array2string(ss['omega'], precision=6)} -> c_BdG = {ss['c']:.5f} a E_R/hbar"
        )
        print(
            f"  f_s^xx = {fs:.5f}, dmu/dn = {dmu_dn:.5f}, c = sqrt(2 kappa f_s n dmu/dn) = {c_fs:.5f}: "
            f"c_BdG / c_fs - 1 = {(ss['c'] / c_fs - 1) * 100:+.3f} %"
        )
        print(
            f"  lowest X mode {b['energies'][1][0]:.5f} vs c pi/L = {ss['c'] * math.pi / L:.5f} "
            f"({time.time() - t1:.1f} s); total for this Omega_R {time.time() - t0:.1f} s"
        )
    print(f"\ndone in {time.time() - t_all:.1f} s")


# ── trapped clouds ─────────────────────────────────────────────────────────────


def run_trapped(name: str, row: dict) -> None:
    """LDA start from the supercell coupling scan -> polished trapped ground state; cloud numbers and
    in-situ contrasts (imaging.standard_observables, via trapped.summary); the field to fields/state.npz."""
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine.observables import export_state

    keys = {k: v for k, v in row.items() if k not in ("kind", "s_sat")}
    spec = TR.TrappedSpec(**keys)
    grid, geo = TR.trap_grid(spec)
    _section("pre-flight")
    print(describe_device())
    print(
        f"supercell m = {spec.m}: {spec.cell_points}^2 per cell (dx = {geo.L_box / spec.cell_points:.4f} a); "
        f"box {spec.box_cells}x{spec.box_cells} cells = {grid.N[0]}^2, L = {grid.L[0]:.3f} a"
    )
    print("trapped grid: " + _memory_note(grid.N[0] * grid.N[1], 1, 2, spec.dtype))
    print(
        f"N = {spec.N_atoms:g}, omega_r = 2 pi x {spec.nu_r_hz:g} Hz = {spec.omega_r():.6f} E_R, Omega_R = "
        f"{spec.Omega_R:.6f} E_R, vertical = {spec.vertical}, g2D = {spec.couplings()[0]:.7f} E_R a^2"
    )
    _section("run")
    run = TR.trapped_ground_state(spec, verbose=VERBOSE)
    res = TR.summary(run, spec, s_sat=row.get("s_sat", 0.01))
    out = _outdir(name)
    os.makedirs(os.path.join(out, "fields"), exist_ok=True)
    export_state(
        run["engine"],
        run["psi"],
        os.path.join(out, "fields", "state.npz"),
        Omega_R=spec.Omega_R,
        V0=spec.V0,
        N_atoms=spec.N_atoms,
        nu_r_hz=spec.nu_r_hz,
        vertical=spec.vertical,
        labels="paper",
    )
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    c, ins, gr = res["cloud"], res["insitu"], res["ground"]
    lines = [
        f"trapped {name}: N = {c['N']:.1f}, R_rms = {c['R_rms']:.4f} a = {c['R_rms_um']:.4f} um, "
        f"central-cell nbar = {c['nbar_centre_cell']:.4f}, N2/N = {c['N2_over_N']:.6f}, edge {c['edge_density_rel']:.1e}",
        f"  mu = {gr['mu']:.10f} E_R, E/N = {gr['E_per_N']:.10f} E_R, residual {gr['residual']:.1e}, "
        f"{gr['steps']} polish steps, {gr['polish_seconds']:.0f} s (total {gr['total_seconds']:.0f} s)",
        f"  in situ |2>: C_raw {ins['C_raw']:.5f}, C_coh(s={ins['s_sat']:g}) {ins['C_coh']:.5f}, "
        f"C_coh(in focus) {ins['C_coh_infocus']:.5f}, C_psf {ins['C_psf']:.5f}, n_AA {ins['n_AA']:.4f}, n_AB {ins['n_AB']:.4f}",
        f"  LDA start: C_raw {res['lda']['insitu']['C_raw']:.5f}, nbar_centre {res['lda']['cloud']['nbar_centre_cell']:.4f}, "
        f"R_rms {res['lda']['cloud']['R_rms']:.4f} a, E/N {res['lda']['E_per_N']:.8f}",
    ]
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    _section("result")
    print("\n".join(lines))
    print(f"wrote {out}/summary.{{json,txt}}, fields/state.npz")


# ── GPU certification ──────────────────────────────────────────────────────────

GPU_KEYS = [
    "E_per_N",
    "mu",
    "fs_xx_energy",
    "fs_xx_current",
    "fs_yy_energy",
    "leggett_x_lower",
    "leggett_x_upper",
    "participation_ratio",
    "contrast_blur",
]
GPU_TOL = {"fs_xx_energy": 1e-4, "fs_xx_current": 1e-4, "fs_yy_energy": 1e-4}


def run_gpu_check(name: str, row: dict) -> None:
    """The quick coupling-scan pipeline (m = 4, 12 pts/a, Omega_R = 0, 0.24, 1, complex128) on the GPU and in
    a CPU subprocess (TWISTED_BILAYER_FORCE_CPU=1), every number compared: 1e-6 on energies, 1e-4 on the
    superfluid fractions (an energy difference ~1e-3 E_R against ~80 E_R, so the last digits differ
    between cuFFT and pocketfft). Then the m = 22 production supercell is timed."""
    from twisted_bilayer.engine.backend import HAS_GPU

    print("GPU:", describe_device())
    if not HAS_GPU:
        raise SystemExit("no CUDA device visible to torch; nothing to certify")
    out = _outdir(name)

    def run_quick(outdir: str, force_cpu: bool) -> dict:
        env = dict(os.environ)
        if force_cpu:
            env["TWISTED_BILAYER_FORCE_CPU"] = "1"
        code = (
            "import torch, numpy as np; from twisted_bilayer.engine.sweep import SweepSpec, coupling_scan; "
            "spec = SweepSpec(V0=8.0, Omega_R=(0.0, 0.24, 1.0), m=4, points_per_a=12, tol=1e-9, "
            "dtype_relax='complex128', dtype_polish='complex128'); "
            "res = coupling_scan(spec, verbose=False); "
            f"res.save({outdir!r}); print(res.device)"
        )
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            raise SystemExit("quick run failed")
        print("  device:", p.stdout.strip().splitlines()[-1])
        return dict(np.load(os.path.join(outdir, "coupling_scan.npz")))

    print("quick pipeline on the GPU ...")
    gpu = run_quick(os.path.abspath(os.path.join(out, "gpu")), force_cpu=False)
    print("quick pipeline on the CPU ...")
    cpu = run_quick(os.path.abspath(os.path.join(out, "cpu")), force_cpu=True)
    ok = True
    for k in GPU_KEYS:
        d = float(np.max(np.abs(gpu[k] - cpu[k])))
        tol = GPU_TOL.get(k, 1e-6)
        ok &= d < tol
        print(f"  [{'OK ' if d < tol else 'FAIL'}] {k:>22s}: max |GPU - CPU| = {d:.2e} (tol {tol:g})")
    print("GPU == CPU within tolerance" if ok else "GPU/CPU MISMATCH -- do not trust GPU numbers")
    if not row.get("time_production", True):
        return
    print("\ntiming the production supercell (m = 22, 16 pts/a, 4 couplings, complex64 -> complex128) ...")
    from twisted_bilayer.engine.sweep import SweepSpec, coupling_scan

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = coupling_scan(
        SweepSpec(V0=8.0, Omega_R=(0.0, 0.1, 0.24, 0.5), m=22, points_per_a=16, tol=1e-8), verbose=False
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"  {dt:.1f} s for 4 couplings x 3 relaxations; peak GPU memory {peak:.2f} GiB; "
        f"residual max {float(np.max(res.columns['residual_0'])):.1e}"
    )
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(
        f"  => a 10-coupling sweep ~ {dt * 10 / 4:.0f} s; this {total:.0f} GiB GPU holds "
        f"~{int(0.75 * total / max(peak, 1e-3) * 4)} couplings per batch"
    )
    with open(os.path.join(out, "timing.json"), "w") as f:
        json.dump(dict(seconds=dt, peak_gib=peak, device=describe_device(), timing=res.timing), f, indent=2)


# ── full 3D against quasi-2D, sized to the device ─────────────────────────────


def _trapped3d_row(row: dict):
    """The TrappedSpec of a row (its cloud given at box_cells = 3), the Vertical, and the sizing keys."""
    import dataclasses

    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3

    spec_keys = {f.name for f in dataclasses.fields(TR.TrappedSpec)} - {"box_cells", "dtype", "vertical"}
    spec = TR.TrappedSpec(**{k: v for k, v in row.items() if k in spec_keys})
    vert = T3.Vertical(z_extent_lz=row.get("z_extent_lz", 5.0), polish_steps=row.get("polish_steps_3d", 20000))
    size = dict(
        box_cells=row.get("box_cells", "auto"),
        Nz=row.get("Nz", "auto"),
        Nz_min=row.get("Nz_min", 16),
        Nz_max=row.get("Nz_max", 128),
        cpu_box_cells=row.get("cpu_box_cells", 1),
        cpu_Nz=row.get("cpu_Nz", 24),
    )
    return spec, vert, size


def _fixed(size: dict) -> tuple[int | None, int | None]:
    b, z = size["box_cells"], size["Nz"]
    return (None if b == "auto" else int(b)), (None if z == "auto" else int(z))


def _cpu_size(size: dict) -> tuple[int, int]:
    b, z = _fixed(size)
    b = size["cpu_box_cells"] if b is None else b
    z = size["cpu_Nz"] if z is None else z
    print(f"no GPU: automatic sizing needs one; running box {b}, N_z = {z} (cpu_box_cells / cpu_Nz of the row)")
    return b, z


def _write_3d_outputs(out: str, tag: str, res: dict, obs: dict, cmp: dict, title: str) -> None:
    from twisted_bilayer.engine import trapped3d as T3

    os.makedirs(os.path.join(out, "fields"), exist_ok=True)
    T3.save_fields(res, os.path.join(out, "fields", f"columns_{tag}.npz"))
    T3.plot_comparison(res, os.path.join(out, f"comparison_{tag}.png"), title=title)


def run_trapped_3d(name: str, row: dict) -> None:
    """The trapped cloud in full 3D and in the two quasi-2D models (engine.trapped3d), as large as the
    device holds: the largest odd box_cells at N_z >= Nz_min within memory_fraction of the free GPU
    memory, then the largest N_z. box_cells / Nz as numbers pin the size."""
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.engine.backend import HAS_GPU

    spec0, vert, size = _trapped3d_row(row)
    dtype = row.get("dtype", "complex128")
    dtype_2d = row.get("dtype_2d", dtype)
    cp = spec0.cell_points
    _section("pre-flight")
    print(describe_device())
    plan_info = {}
    box, Nz = _fixed(size)
    if box is None or Nz is None:
        if not HAS_GPU:
            box, Nz = _cpu_size(size)
        else:
            pr = T3.probe(spec0, vert, dtype)
            budget = T3.memory_budget(row.get("memory_fraction", 0.85))
            mem = pr["memory"]
            p = T3.plan(
                lambda b, z: mem(T3.points_3d(b, z, cp)),
                budget,
                cp,
                size["Nz_min"],
                size["Nz_max"],
                box_fixed=box,
                Nz_fixed=Nz,
            )
            if p is None:
                raise SystemExit(f"box 1 at N_z = {size['Nz_min']} does not fit in {budget / 2**30:.2f} GiB")
            box, Nz = p
            pts = T3.points_3d(box, Nz, cp)
            plan_info = dict(
                budget_gib=budget / 2**30,
                bytes_per_point=mem.b,
                predicted_peak_gib=mem(pts) / 2**30,
                step_seconds=pr["step_seconds"](pts),
                predicted_seconds=T3.predicted_seconds(pr["step_seconds"], box, Nz, cp),
            )
            print(
                f"probe ({dtype}): {mem.b:.0f} bytes per 3D point at the polish peak, "
                f"{pr['step_seconds'].b * 1e9:.1f} ns per point per step; budget {plan_info['budget_gib']:.2f} GiB"
            )
            print(
                f"plan: peak ~{plan_info['predicted_peak_gib']:.2f} GiB, {plan_info['step_seconds']:.3f} s per 3D step, "
                f"~{plan_info['predicted_seconds'] / 60:.0f} min in all (from {T3.STEPS_3D} 3D and 2 x {T3.STEPS_2D} 2D "
                "polish steps; the real count depends on the cloud)"
            )
    spec = T3.scaled(spec0, box)
    vert.Nz = int(Nz)
    print(T3.describe_plan(spec, vert) + f", 3D in {dtype}, 2D in {dtype_2d}")
    _section("run")
    res = T3.solve_models(spec, vert, dtype, verbose=VERBOSE, dtype_2d=dtype_2d)
    obs = T3.model_observables(res, s_sat=row.get("s_sat", 0.01))
    cmp = T3.compare_models(res, obs)
    out = _outdir(name)
    report = T3.report_models(obs, cmp)
    head = T3.describe_plan(spec, vert) + f", 3D in {dtype}, 2D in {dtype_2d}"
    if HAS_GPU and obs["3d"]["peak_gib"] is not None:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        head += f"\n3D solve peak memory {obs['3d']['peak_gib']:.2f} GiB of {total:.1f} GiB"
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(
            dict(
                spec=dataclasses_asdict(spec),
                Nz=vert.Nz,
                z_extent_lz=vert.z_extent_lz,
                dtype=dtype,
                dtype_2d=dtype_2d,
                plan=plan_info,
                observables=obs,
                comparison=cmp,
                g3d=res["g3d"],
                omega_z=res["omega_z"],
                l_z=res["l_z"],
            ),
            f,
            indent=1,
            default=float,
        )
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(head + "\n" + report + "\n")
    _write_3d_outputs(out, dtype, res, obs, cmp, head.splitlines()[0])
    _section("result")
    print(head)
    print(report)
    print(f"wrote {out}/summary.json, report.txt, comparison_{dtype}.png, fields/columns_{dtype}.npz")


def run_trapped_3d_precision(name: str, row: dict) -> None:
    """The same comparison in complex128 and complex64, sized so that both together take about
    `minutes` on the device: is single precision good enough to resolve the 3D - 2D difference?"""
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.engine.backend import HAS_GPU

    spec0, vert, size = _trapped3d_row(row)
    minutes = float(row.get("minutes", 10.0))
    dtypes = ("complex128", "complex64")
    cp = spec0.cell_points
    _section("pre-flight")
    print(describe_device())
    plan_info = {}
    box, Nz = _fixed(size)
    if box is None or Nz is None:
        if not HAS_GPU:
            box, Nz = _cpu_size(size)
        else:
            prs = {d: T3.probe(spec0, vert, d) for d in dtypes}
            budget = T3.memory_budget(row.get("memory_fraction", 0.85))
            mem = prs["complex128"]["memory"]

            def cost(b, z):
                if mem(T3.points_3d(b, z, cp)) > budget:
                    return math.inf
                return sum(T3.predicted_seconds(prs[d]["step_seconds"], b, z, cp) for d in dtypes)

            target = 0.9 * 60.0 * minutes  # the supercell scan and the I/O take the rest
            p = T3.plan(cost, target, cp, size["Nz_min"], size["Nz_max"], box_fixed=box, Nz_fixed=Nz)
            if p is None:
                p = (box or 1, Nz or size["Nz_min"])
                print(f"even box {p[0]} at N_z = {p[1]} is predicted above {minutes:g} min; running it anyway")
            box, Nz = p
            plan_info = dict(
                minutes=minutes,
                predicted_seconds=cost(box, Nz),
                step_seconds={d: prs[d]["step_seconds"](T3.points_3d(box, Nz, cp)) for d in dtypes},
            )
            print(
                f"plan: ~{plan_info['predicted_seconds'] / 60:.1f} min predicted; 3D step "
                + ", ".join(f"{d} {s:.3f} s" for d, s in plan_info["step_seconds"].items())
            )
    spec = T3.scaled(spec0, box)
    vert.Nz = int(Nz)
    print(T3.describe_plan(spec, vert))
    _section("run")
    t0 = time.time()
    scan = TR.supercell_scan(spec, verbose=VERBOSE)
    runs = {}
    for d in dtypes:
        print(f"\n{d}:", flush=True)
        res = T3.solve_models(spec, vert, d, scan=scan, verbose=VERBOSE)
        obs = T3.model_observables(res, s_sat=row.get("s_sat", 0.01))
        runs[d] = (res, obs, T3.compare_models(res, obs))
    elapsed = time.time() - t0
    (r128, o128, c128), (r64, o64, c64) = runs["complex128"], runs["complex64"]
    table, numbers = T3.precision_table(o64, o128, c128)
    dA = r128["grid2"].dx[0] * r128["grid2"].dx[1]
    dens = {m: T3.density_deviation(r64["models"][m]["n"], r128["models"][m]["n"], dA) for m in T3.MODELS}
    lines = [
        T3.describe_plan(spec, vert),
        f"wall time {elapsed / 60:.1f} min (target {minutes:g} min)",
        "",
        "complex128:",
        T3.report_models(o128, c128),
        "",
        "complex64:",
        T3.report_models(o64, c64),
        "",
        "complex64 against complex128 (the 3D - NPSE gap from complex128):",
        table,
        "  column density, complex64 vs complex128: "
        + "; ".join(f"{m} max {v['max_rel']:.1e}, int {v['l1_rel']:.1e}" for m, v in dens.items()),
    ]
    out = _outdir(name)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(
            dict(
                spec=dataclasses_asdict(spec),
                Nz=vert.Nz,
                z_extent_lz=vert.z_extent_lz,
                plan=plan_info,
                elapsed_seconds=elapsed,
                observables={d: runs[d][1] for d in dtypes},
                comparison={d: runs[d][2] for d in dtypes},
                precision=numbers,
                density_precision=dens,
            ),
            f,
            indent=1,
            default=float,
        )
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for d in dtypes:
        _write_3d_outputs(out, d, runs[d][0], runs[d][1], runs[d][2], f"{T3.describe_plan(spec, vert)}, {d}")
    _section("result")
    print("\n".join(lines))
    print(f"wrote {out}/summary.json, report.txt, comparison_<dtype>.png, fields/columns_<dtype>.npz")


def run_trapped_3d_nz(name: str, row: dict) -> None:
    """The 3D cloud at several N_z over the same z box (engine.trapped3d.nz_convergence): is the vertical
    resolution converged, and is its error small against the 3D - NPSE difference? On a GPU, N_z values
    whose 3D solve would not fit in memory_fraction of the free memory are skipped."""
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.engine.backend import HAS_GPU

    spec0, vert, size = _trapped3d_row(row)
    box = int(row.get("box_cells", 3))
    spec = T3.scaled(spec0, box)
    dtype = row.get("dtype", "complex64")
    dtype_2d = row.get("dtype_2d", "complex128")
    Nz_list = sorted(int(n) for n in row["Nz_list"])
    _section("pre-flight")
    print(describe_device())
    if HAS_GPU:
        pr = T3.probe(spec0, vert, dtype)
        budget = T3.memory_budget(row.get("memory_fraction", 0.85))
        fits = [n for n in Nz_list if pr["memory"](T3.points_3d(box, n, spec.cell_points)) <= budget]
        if len(fits) < len(Nz_list):
            print(f"skipping N_z = {sorted(set(Nz_list) - set(fits))}: above {budget / 2**30:.2f} GiB")
        Nz_list = fits
        if len(Nz_list) < 2:
            raise SystemExit("fewer than two N_z fit on this device; lower Nz_list or box_cells")
        for n in Nz_list:
            pts = T3.points_3d(box, n, spec.cell_points)
            print(f"  N_z = {n}: ~{pr['memory'](pts) / 2**30:.2f} GiB, ~{pr['step_seconds'](pts):.3f} s per step")
    vert.Nz = Nz_list[-1]
    print(T3.describe_plan(spec, vert) + f" (finest); N_z = {Nz_list}; 3D in {dtype}, 2D in {dtype_2d}")
    _section("run")
    conv = T3.nz_convergence(
        spec,
        Nz_list,
        z_extent_lz=vert.z_extent_lz,
        dtype=dtype,
        dtype_2d=dtype_2d,
        polish_steps=vert.polish_steps,
        verbose=VERBOSE,
    )
    report = T3.report_nz(conv)
    out = _outdir(name)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(
            dict(
                spec=dataclasses_asdict(conv["spec"]),
                z_extent_lz=conv["z_extent_lz"],
                dtype=dtype,
                dtype_2d=dtype_2d,
                npse=conv["npse"],
                rows=[{k: v for k, v in r.items() if k != "numbers"} for r in conv["rows"]],
            ),
            f,
            indent=1,
            default=float,
        )
    head = T3.describe_plan(spec, vert) + f" (finest); 3D in {dtype}, 2D in {dtype_2d}"
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(head + "\n" + report + "\n")
    _section("result")
    print(head)
    print(report)
    print(f"wrote {out}/summary.json, report.txt")


def run_xy_convergence(name: str, row: dict) -> None:
    """In-plane resolution (engine.convergence): the homogeneous supercell at several points per a, and the
    trapped cloud at several points per supercell, each against the finest. Sub-dicts `supercell` and
    `trapped` of the row; either may be absent. complex128 throughout."""
    from twisted_bilayer.engine import convergence as CV
    from twisted_bilayer.engine import trapped as TR

    _section("pre-flight")
    print(describe_device())
    sc, tr = row.get("supercell"), row.get("trapped")
    tolerances = row.get("tolerances")
    if sc:
        print(
            f"supercell: V0 = {sc['V0_list']}, Omega_R = {sc['Omega_R']}, points per a = {sc['ppa_list']}, "
            f"m = {sc.get('m', 22)}"
        )
    if tr:
        print(f"trapped: points per supercell = {tr['cell_points_list']}, box {tr.get('box_cells', 3)}")
    _section("run")
    conv_sc = conv_tr = None
    if sc:
        conv_sc = CV.supercell_convergence(
            sc["V0_list"],
            sc["Omega_R"],
            sc["ppa_list"],
            m=sc.get("m", 22),
            n=sc.get("n", 1),
            filling=sc.get("filling", 1.0),
            vertical=sc.get("vertical", "npse"),
            tol=sc.get("tol", 1e-9),
            verbose=VERBOSE,
        )
    if tr:
        keys = {k: v for k, v in tr.items() if k not in ("cell_points_list", "kind", "s_sat")}
        spec = TR.TrappedSpec(**keys)
        conv_tr = CV.trapped_convergence(spec, tr["cell_points_list"], verbose=VERBOSE)
    text = CV.report(conv_sc, conv_tr, tolerances)
    out = _outdir(name)
    write_json(os.path.join(out, "summary.json"), CV.to_record(conv_sc, conv_tr, tolerances))
    sc_rows, tr_rows = CV.table_rows(conv_sc, conv_tr)
    for fname, rows in (("supercell.csv", sc_rows), ("trapped.csv", tr_rows)):
        if rows:
            with open(os.path.join(out, fname), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(os.path.join(out, "report.txt"), "w") as f:
        f.write(text + "\n")
    _section("result")
    print(text)
    print(f"wrote {out}/summary.json, report.txt, supercell.csv, trapped.csv")


def dataclasses_asdict(obj) -> dict:
    import dataclasses

    return dataclasses.asdict(obj)


KINDS = {
    "coupling_scan": run_coupling_scan,
    "thin_z": run_thin_z,
    "rabi": run_rabi,
    "toy1d": run_toy1d,
    "tight_binding": run_tight_binding,
    "bridge": run_bridge,
    "bdg": run_bdg,
    "trapped": run_trapped,
    "gpu_check": run_gpu_check,
    "trapped_3d": run_trapped_3d,
    "trapped_3d_precision": run_trapped_3d_precision,
    "trapped_3d_nz": run_trapped_3d_nz,
    "xy_convergence": run_xy_convergence,
}


def _provenance() -> dict:
    """Package version, git commit (when run from a checkout), library versions and machine."""
    from twisted_bilayer import __version__

    here = os.path.dirname(os.path.abspath(__file__))
    commit, dirty = None, None

    def git(*args):
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=5).stdout.strip()

    try:  # only when the package runs from a checkout (an editable install), not from site-packages
        top = git("rev-parse", "--show-toplevel")
        if top and os.path.samefile(os.path.join(top, "src", "twisted_bilayer"), here):
            commit = git("rev-parse", "--short", "HEAD") or None
            dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(
        twisted_bilayer=__version__,
        git_commit=commit,
        git_dirty=dirty,
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        device=describe_device(),
        platform=platform.platform(),
    )


# The precision keys of each kind. `twisted-bilayer run NAME --128` sets them all to complex128 (publication
# runs); without it a row runs in the precision it names. Kinds not listed run in complex128 throughout,
# and `trapped_3d_precision` runs both precisions by design.
PRECISION_KEYS = {
    "coupling_scan": ("dtype_relax", "dtype_polish"),
    "bridge": ("dtype_relax", "dtype_polish"),
    "thin_z": ("dtype_relax", "dtype_polish"),
    "rabi": ("dtype",),
    "trapped": ("dtype",),
    "trapped_3d": ("dtype", "dtype_2d"),
    "trapped_3d_nz": ("dtype", "dtype_2d"),
}


def double_precision(row: dict) -> dict:
    """A copy of the row with every precision key of its kind set to complex128."""
    row = dict(row)
    for k in PRECISION_KEYS.get(row["kind"], ()):
        row[k] = "complex128"
    return row


def run_scenario(name: str, row: dict, out_root: str = "results", double: bool = False) -> None:
    """Run one scenario row; files go to <out_root>/<name>/ (provenance.json included). double=True
    (`--128` on the command line) runs it in complex128 throughout."""
    global _OUT_ROOT
    _OUT_ROOT = out_root
    if row["kind"] not in KINDS:
        raise SystemExit(f"scenario {name!r}: unknown kind {row['kind']!r}; known kinds: {sorted(KINDS)}")
    if double:
        row = double_precision(row)
    print(f"scenario {name} ({row['kind']}) on {describe_device()}")
    keys = PRECISION_KEYS.get(row["kind"], ())
    if keys:
        print(
            "precision: " + ", ".join(f"{k} = {row.get(k, 'default')}" for k in keys) + (" (--128)" if double else "")
        )
    t0 = time.time()
    KINDS[row["kind"]](name, row)
    out = _outdir(name)
    with open(os.path.join(out, "provenance.json"), "w") as f:
        json.dump(
            dict(scenario=name, row=row, seconds=round(time.time() - t0, 1), **_provenance()), f, indent=2, default=str
        )
    print(f"\ndone in {time.time() - t0:.1f} s")
