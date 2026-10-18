"""The coupling scan: superfluid fraction, Leggett brackets, chemical
potential and pocket formation versus the interlayer coupling Omega_R on the
commensurate supercell.

Pipeline (all batched over Omega_R, so one GPU call per stage):
  1. ground state at every Omega_R, twist 0, from a phase-locked start
     (uniform density, component 2 at relative phase pi);
  2. from each of those, the twisted ground states theta_x = theta e_x and
     theta_y = theta e_y;
  3. observables: E/N, mu, f_s^xx and f_s^yy by energy and by current, and
     their twist-free limit f_s = 2 f_energy - f_current (columns fs_xx, fs_yy),
     Leggett lower/upper brackets along x and y from the twist-0 density,
     participation ratio, moire contrast (raw and blurred over the primary
     lattice), N_2/N, residuals.
Relaxation runs in `dtype_relax` (complex64 by default: fast on a consumer GPU)
and is polished to `tol` in `dtype_polish` (complex128).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import bands
from .backend import DEVICE, complex_dtype, describe_device, synchronize, to_numpy
from .grid import Grid
from .ground_state import RelaxResult, initial_state, relax
from .lattice import MoireGeometry, bilayer_potential, moire_geometry
from .observables import (
    component_fractions,
    contrast,
    export_state,
    leggett_bounds,
    participation_ratio,
    pocket_populations,
    superfluid_fraction_current,
    superfluid_fraction_energy,
)
from .physics import BilayerGPE
from .units import KAPPA, lattice_units


@dataclass
class SweepSpec:
    V0: float = 8.0
    Omega_R: tuple = tuple(np.round(np.linspace(0.0, 1.0, 11), 4))
    filling: float = 1.0  # atoms per a^2 (total, both components)
    m: int = 22
    n: int = 1
    points_per_a: float = 16.0
    delta: float = 0.0
    g2d: float | None = None  # None -> from units (omega_z = 1 kHz)
    omega_z_hz: float = 1000.0
    vertical: str = "npse"  # "fixed": g2d n; "npse": density-dependent width (engine.npse)
    theta: float = 0.3  # twist (rad) across the box for f_s
    tol: float = 1e-8
    dtype_relax: str = "complex64"
    dtype_polish: str = "complex128"
    max_polish_steps: int = 20000
    blur_sigma: float = 1.0  # a, for the moire contrast
    phase_noise: float = 0.0  # rad; > 0 for an independent-seed check
    seed: int = 0
    save_fields: bool = True  # one exchange npz per Omega_R (psi1, psi2, n, mu, ...)


@dataclass
class SweepResult:
    spec: SweepSpec
    geometry: MoireGeometry
    N_grid: int
    J1: float
    columns: dict = field(default_factory=dict)  # name -> np.ndarray over Omega_R
    densities: dict = field(default_factory=dict)  # "n", "n1", "n2": (n_Omega, N, N) float32
    timing: dict = field(default_factory=dict)
    device: str = ""

    def table(self) -> str:
        keys = [
            "Omega_R",
            "Omega_over_J",
            "E_per_N",
            "mu",
            "fs_xx",
            "fs_xx_energy",
            "fs_xx_current",
            "fs_yy_energy",
            "leggett_x_lower",
            "leggett_x_upper",
            "participation_ratio",
            "contrast_blur",
            "N_AA",
            "N_B1",
            "N_P",
            "AA_over_choke",
            "residual_0",
            "converged",
        ]
        rows = [" ".join(f"{k:>14s}" for k in keys)]
        n = len(self.columns["Omega_R"])
        for i in range(n):
            cells = []
            for k in keys:
                v = self.columns[k][i]
                cells.append(f"{v:>14.5g}" if isinstance(v, (float, np.floating)) else f"{str(v):>14s}")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    fields: list = field(default_factory=list)  # (Omega_R, psi (1,2,*N) cpu tensor) when save_fields
    engine_for_export: object = None

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        if self.fields and self.engine_for_export is not None:
            os.makedirs(os.path.join(outdir, "fields"), exist_ok=True)
            for om, psi in self.fields:
                eng = self.engine_for_export(om)
                export_state(
                    eng,
                    psi.to(eng.device),
                    os.path.join(outdir, "fields", f"state_OmegaR_{om:.4f}.npz"),
                    Omega_R=om,
                    V0=self.spec.V0,
                    filling=self.spec.filling,
                    m=self.geometry.m,
                    n=self.geometry.n,
                    delta=self.spec.delta,
                    vertical=self.spec.vertical,
                )
        np.savez_compressed(
            os.path.join(outdir, "coupling_scan.npz"),
            **self.columns,
            **{f"density_{k}": v for k, v in self.densities.items()},
        )
        with open(os.path.join(outdir, "coupling_scan.csv"), "w") as f:
            keys = list(self.columns.keys())
            f.write(",".join(keys) + "\n")
            for i in range(len(self.columns["Omega_R"])):
                f.write(",".join(str(self.columns[k][i]) for k in keys) + "\n")
        meta = dict(
            spec=asdict(self.spec),
            geometry=dict(
                m=self.geometry.m,
                n=self.geometry.n,
                theta_deg=self.geometry.theta_deg,
                L_box=self.geometry.L_box,
                L_moire=self.geometry.L_moire,
            ),
            N_grid=self.N_grid,
            J1=self.J1,
            timing=self.timing,
            device=self.device,
        )
        with open(os.path.join(outdir, "run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        with open(os.path.join(outdir, "coupling_scan.txt"), "w") as f:
            f.write(self.table() + "\n")


def _relax_two_stage(
    engine_relax: BilayerGPE, engine_polish: BilayerGPE, psi0, n_particles, spec: SweepSpec, verbose: bool
) -> RelaxResult:
    """Ladder + polish in the fast dtype, then polish to tol in the precise one."""
    if engine_polish is engine_relax:
        return relax(engine_relax, psi0, n_particles, tol=spec.tol, polish_steps=spec.max_polish_steps, verbose=verbose)
    r = relax(
        engine_relax, psi0, n_particles, tol=max(spec.tol, 1e-5), polish_steps=spec.max_polish_steps, verbose=verbose
    )
    r2 = relax(
        engine_polish,
        r.psi.to(engine_polish.dtype),
        n_particles,
        tol=spec.tol,
        dt_schedule=[],
        polish_steps=spec.max_polish_steps,
        verbose=verbose,
    )
    r2.steps += r.steps
    r2.stages = r.stages + r2.stages
    return r2


def coupling_scan(spec: SweepSpec, verbose: bool = True, device: torch.device | None = None) -> SweepResult:
    device = DEVICE if device is None else device
    t_start = time.time()
    units = lattice_units(omega_z_hz=spec.omega_z_hz)
    g2d = units.g2d if spec.g2d is None else spec.g2d
    geo = moire_geometry(spec.m, spec.n)
    N = geo.grid_size(spec.points_per_a)
    dt_relax, dt_polish = complex_dtype(spec.dtype_relax), complex_dtype(spec.dtype_polish)
    grid = Grid((N, N), (geo.L_box, geo.L_box), device=device, real_dtype=torch.float64)
    V = bilayer_potential(grid, spec.V0, geo)
    omegas = torch.tensor(list(spec.Omega_R), dtype=torch.float64)
    B = len(omegas)
    n_particles = spec.filling * grid.volume
    if spec.vertical == "npse":
        from .npse import NPSE2D

        interaction = NPSE2D(g2d, units.rad_s_to_ER(2.0 * math.pi * spec.omega_z_hz))
    elif spec.vertical == "fixed":
        interaction = None
    else:
        raise ValueError(f"vertical must be 'fixed' or 'npse', got {spec.vertical!r}")

    J1 = bands.hopping_1d(spec.V0, d_max=1)[0]
    if verbose:
        print(f"coupling scan on {describe_device()}")
        print(f"  {geo.describe()}")
        print(f"  {grid.describe()} ({N / geo.L_box:.2f} points per a)")
        print(
            f"  V0 = {spec.V0} E_R (J1 = {J1:.5f} E_R), g2d = {g2d:.5f} E_R a^2, filling {spec.filling}, "
            f"N = {n_particles:.1f} atoms, Delta = {spec.delta}"
        )
        print(f"  Omega_R = {[float(o) for o in omegas]} E_R  (Omega_R/J1 up to {float(omegas.max()) / J1:.1f})")
        print(f"  vertical reduction: {spec.vertical} ({interaction.describe() if interaction else 'g2d n'})")
        print(f"  relax in {spec.dtype_relax}, polish in {spec.dtype_polish} to residual {spec.tol:g}")

    def make(twist, dtype):
        return BilayerGPE(
            grid.to(torch.float32 if dtype == torch.complex64 else torch.float64),
            V=V,
            g=0.0 if interaction is not None else g2d,
            Omega_R=omegas,
            delta=spec.delta,
            n_components=2,
            twist=twist,
            dtype=dtype,
            interaction=interaction,
        )

    timing = {}
    # 1. twist-0 ground states
    t0 = time.time()
    e_relax, e_pol = make(None, dt_relax), make(None, dt_polish)
    if e_relax.dtype == e_pol.dtype:
        e_relax = e_pol
    psi0 = initial_state(e_relax, n_particles, batch=B, phase_noise=spec.phase_noise, seed=spec.seed)
    r0 = _relax_two_stage(e_relax, e_pol, psi0, n_particles, spec, verbose=False)
    synchronize()
    timing["ground_states_s"] = time.time() - t0
    if verbose:
        print(
            f"  twist 0: {r0.steps} steps, residual max {float(r0.residual.max()):.1e}, "
            f"{timing['ground_states_s']:.1f} s"
        )
    terms0 = e_pol.energy_terms(r0.psi)

    # 2. twisted ground states along x and y
    results_tw = {}
    for ax, name in ((0, "x"), (1, "y")):
        t0 = time.time()
        tw = torch.zeros(B, 2, dtype=torch.float64)
        tw[:, ax] = spec.theta
        er, ep = make(tw, dt_relax), make(tw, dt_polish)
        if er.dtype == ep.dtype:
            er = ep
        r = _relax_two_stage(er, ep, r0.psi.to(er.dtype), n_particles, spec, verbose=False)
        synchronize()
        timing[f"twist_{name}_s"] = time.time() - t0
        results_tw[name] = (ep, r)
        if verbose:
            print(
                f"  twist {name}: {r.steps} steps, residual max {float(r.residual.max()):.1e}, "
                f"{timing[f'twist_{name}_s']:.1f} s"
            )

    # 3. observables
    Nvec = terms0["norm"]
    dens0 = e_pol.density(r0.psi)
    cols = {
        "Omega_R": to_numpy(omegas),
        "Omega_over_J": to_numpy(omegas) / J1,
        "E_per_N": to_numpy(terms0["E_total"] / Nvec),
        "mu": to_numpy(terms0["mu"]),
        "E_kin_per_N": to_numpy(terms0["E_kin"] / Nvec),
        "E_int_per_N": to_numpy(terms0["E_int"] / Nvec),
        "E_rabi_per_N": to_numpy(terms0["E_rabi"] / Nvec),
    }
    for ax, name in ((0, "x"), (1, "y")):
        ep, r = results_tw[name]
        fsE = superfluid_fraction_energy(r.energy, terms0["E_total"], Nvec, geo.L_box, spec.theta, KAPPA)
        fsI = superfluid_fraction_current(ep, r.psi, ax, spec.theta)
        lo, up = leggett_bounds(dens0, grid, ax)
        cols[f"fs_{name}{name}_energy"] = to_numpy(fsE)
        cols[f"fs_{name}{name}_current"] = to_numpy(fsI)
        # The two routes carry the finite-twist term with slopes b and 2b in theta^2 (E(theta) - E(0) is
        # quartic in theta beyond the f_s term), so 2 f_E - f_C removes it: the theta -> 0 limit to
        # O(theta^4). Measured 17 Oct on m = 22 at theta = 0.1 / 0.2 / 0.3: slope ratio 2.00-2.04, the
        # limit to <= 3e-7 (docs: results 9.3.3). The current route is first order in the state error,
        # so this needs a tight tol (<= 1e-10) in deep lattices.
        cols[f"fs_{name}{name}"] = to_numpy(2.0 * fsE - fsI)
        cols[f"leggett_{name}_lower"] = to_numpy(lo)
        cols[f"leggett_{name}_upper"] = to_numpy(up)
        cols[f"residual_{name}"] = to_numpy(r.residual)
        cols[f"E_twist_{name}_per_N"] = to_numpy(r.energy / Nvec)
    cols["participation_ratio"] = to_numpy(participation_ratio(dens0, grid))
    cols["contrast_raw"] = to_numpy(contrast(dens0, grid, 0.0))
    cols["contrast_blur"] = to_numpy(contrast(dens0, grid, spec.blur_sigma))
    cols["N2_frac"] = to_numpy(component_fractions(e_pol, r0.psi)[:, 1])
    for k, v in pocket_populations(dens0, grid).items():
        cols[k] = to_numpy(v)
    cols["residual_0"] = to_numpy(r0.residual)
    cols["converged"] = to_numpy(r0.converged & results_tw["x"][1].converged & results_tw["y"][1].converged)
    cols["single_particle_mobility"] = np.full(B, bands.effective_mass_ratio(spec.V0))
    d = to_numpy(r0.psi.real**2 + r0.psi.imag**2).astype(np.float32)
    densities = {"n": d.sum(axis=1), "n1": d[:, 0], "n2": d[:, 1]}
    timing["total_s"] = time.time() - t_start
    res = SweepResult(
        spec=spec,
        geometry=geo,
        N_grid=N,
        J1=J1,
        columns=cols,
        densities=densities,
        timing=timing,
        device=describe_device(),
    )
    if spec.save_fields:
        for i, om in enumerate(to_numpy(omegas)):
            res.fields.append((float(om), r0.psi[i : i + 1].detach().cpu()))
        # export_state needs the per-member Omega_R: give save() an engine factory
        res.engine_for_export = (
            lambda om, _V=V, _g=(0.0 if interaction is not None else g2d), _d=spec.delta, _dt=dt_polish, _int=interaction: (
                BilayerGPE(grid, V=_V, g=_g, Omega_R=om, delta=_d, n_components=2, dtype=_dt, interaction=_int)
            )
        )
    if verbose:
        print(res.table())
        print(f"  total {timing['total_s']:.1f} s")
    return res


# -- plotting -------------------------------------------------------------------


def plot_coupling_scan(res: SweepResult, path: str, show_densities: int = 3) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Okabe-Ito: colour-blind safe, fixed order
    C = {
        "energy": "#0072B2",
        "current": "#D55E00",
        "leggett": "#009E73",
        "sp": "#999999",
        "mu": "#0072B2",
        "E": "#CC79A7",
        "pr": "#0072B2",
        "con": "#E69F00",
    }
    c = res.columns
    x = c["Omega_over_J"]
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, c["fs_xx_energy"], "o-", color=C["energy"], ms=4, lw=1.5, label=r"$f_s^{xx}$ (twist energy)")
    ax1.plot(x, c["fs_xx_current"], "s--", color=C["current"], ms=4, lw=1.2, label=r"$f_s^{xx}$ (current)")
    ax1.plot(x, c["fs_yy_energy"], "^:", color=C["energy"], ms=3, lw=1.0, alpha=0.6, label=r"$f_s^{yy}$ (energy)")
    ax1.fill_between(
        x, c["leggett_x_lower"], c["leggett_x_upper"], color=C["leggett"], alpha=0.25, label="Leggett brackets (x)"
    )
    ax1.axhline(
        float(res.columns["single_particle_mobility"][0]),
        color=C["sp"],
        lw=1.0,
        ls="--",
        label=r"$m/m^*$ (exact band, $\Omega_R=0$)",
    )
    ax1.set_xlabel(r"$\Omega_R / J_1$")
    ax1.set_ylabel("superfluid fraction")
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=7.5, loc="best")
    ax1.set_title(f"$V_0={res.spec.V0}\\,E_R$, filling {res.spec.filling}, $m={res.geometry.m}$", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, c["mu"], "o-", color=C["mu"], ms=4, lw=1.5, label=r"$\mu$")
    ax2.plot(x, c["E_per_N"], "s--", color=C["E"], ms=4, lw=1.2, label=r"$E/N$")
    ax2.set_xlabel(r"$\Omega_R / J_1$")
    ax2.set_ylabel(r"energy per atom ($E_R$)")
    ax2.legend(fontsize=8)
    ax2.set_title("chemical potential and energy", fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(x, c["participation_ratio"], "o-", color=C["pr"], ms=4, lw=1.5, label="participation ratio")
    ax3.plot(
        x,
        c["contrast_blur"],
        "s--",
        color=C["con"],
        ms=4,
        lw=1.2,
        label=f"moiré contrast (blur {res.spec.blur_sigma:g}a)",
    )
    ax3.set_xlabel(r"$\Omega_R / J_1$")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.set_title("pocket formation", fontsize=10)

    n = res.densities["n"]
    idx = np.unique(np.linspace(0, len(x) - 1, show_densities).round().astype(int))
    L = res.geometry.L_box
    for j, i in enumerate(idx[:3]):
        ax = fig.add_subplot(gs[1, j])
        im = ax.imshow(
            n[i].T, origin="lower", extent=(-L / 2, L / 2, -L / 2, L / 2), cmap="cividis", vmin=0, vmax=float(n.max())
        )
        ax.set_title(
            rf"$n(\mathbf{{r}})$ at $\Omega_R={c['Omega_R'][i]:.3g}\,E_R$ ($\Omega_R/J_1={x[i]:.1f}$)", fontsize=9
        )
        ax.set_xlabel("x (a)")
        if j == 0:
            ax.set_ylabel("y (a)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"atoms / $a^2$")
    fig.suptitle("Coupling scan on the commensurate supercell", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
