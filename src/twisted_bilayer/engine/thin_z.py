"""The thin-z 3D check: the same two-component problem in 3D with a thin harmonic z
direction, to measure what the quasi-2D reduction g_2D = g_3D/(sqrt(2 pi) l_z)
misses at a given filling and coupling.

The 3D run uses g_3D, V(x, y) + (1/2) m omega_z^2 z^2 and a z box of
+-z_extent l_z with N_z planes. Comparison with the 2D run on the same (x, y)
grid, both relaxed to the same residual:

  * mu_3D - hbar omega_z / 2  vs  mu_2D         (the zero-point energy is the
                                                  only exact difference)
  * column density n(x, y) = int n dz  vs  n_2D   (max relative deviation,
                                                  participation ratio, contrast)
  * local vertical width <z^2>(x, y)  vs  l_z^2/2 (how far from the Gaussian
                                                  ground state the pockets push)
  * f_s^xx by twist in both                       (optional, doubles the cost)

Expected: deviations of order (mu - V_min) / hbar omega_z, i.e. a few percent
at filling 1, growing with filling and with the pocket density at large
Omega_R. If the width grows by more than ~10 % anywhere, a density-dependent
reduction (the NPSE, engine.npse) is needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .backend import DEVICE, complex_dtype, describe_device, synchronize, to_numpy
from .grid import Grid
from .ground_state import initial_state, relax
from .lattice import bilayer_potential, harmonic_trap, moire_geometry
from .observables import contrast, participation_ratio, superfluid_fraction_energy
from .physics import BilayerGPE
from .units import KAPPA, M_LATTICE, lattice_units


@dataclass
class ThinZResult:
    numbers: dict = field(default_factory=dict)
    column_density: np.ndarray | None = None
    density_2d: np.ndarray | None = None
    width_map: np.ndarray | None = None
    timing: dict = field(default_factory=dict)

    def report(self) -> str:
        n = self.numbers
        lines = [
            f"  2D:  mu = {n['mu_2d']:.6f}  E/N = {n['E_per_N_2d']:.6f}  PR = {n['pr_2d']:.4f}  "
            f"contrast = {n['contrast_2d']:.4f}  n_peak = {n['n_peak_2d']:.3f}",
            f"  3D:  mu - w_z/2 = {n['mu_3d_minus_zpe']:.6f}  E/N - w_z/2 = {n['E_per_N_3d_minus_zpe']:.6f}  "
            f"PR = {n['pr_3d']:.4f}  contrast = {n['contrast_3d']:.4f}  n_peak = {n['n_peak_3d']:.3f}",
            f"  mu difference: {n['mu_rel_diff']:+.3%} of (mu - V_min);  column density max rel dev: "
            f"{n['density_max_rel_dev']:.3%};  <z^2>/(l_z^2/2): min {n['width_ratio_min']:.4f}, max {n['width_ratio_max']:.4f}",
        ]
        if "mu_2d_npse" in n:
            lines.append(
                f"  2D NPSE: mu = {n['mu_2d_npse']:.6f}  E/N = {n['E_per_N_2d_npse']:.6f}  PR = {n['pr_2d_npse']:.4f}  "
                f"contrast = {n['contrast_2d_npse']:.4f}  n_peak = {n['n_peak_2d_npse']:.3f}; column density max rel dev "
                f"vs 3D: {n['density_max_rel_dev_npse']:.3%}; max width ratio {n['width_ratio_npse_max']:.4f}"
            )
        if "fs_2d" in n:
            fs_line = f"  f_s^xx: 2D {n['fs_2d']:.5f}, 3D {n['fs_3d']:.5f}"
            if "fs_2d_npse" in n:
                fs_line += f", 2D NPSE {n['fs_2d_npse']:.5f}"
            lines.append(fs_line)
        return "\n".join(lines)


def thin_z_check(
    V0: float = 8.0,
    Omega_R: float = 0.24,
    filling: float = 1.0,
    m: int = 4,
    n: int = 1,
    points_per_a: float = 12.0,
    Nz: int = 32,
    z_extent_lz: float = 4.0,
    omega_z_hz: float = 1000.0,
    delta: float = 0.0,
    tol: float = 1e-7,
    with_fs: bool = False,
    theta: float = 0.3,
    dtype_relax: str = "complex64",
    dtype_polish: str = "complex128",
    verbose: bool = True,
    device: torch.device | None = None,
    npse: bool = True,
) -> ThinZResult:
    device = DEVICE if device is None else device
    units = lattice_units(omega_z_hz=omega_z_hz)
    geo = moire_geometry(m, n)
    N = geo.grid_size(points_per_a)
    L = geo.L_box
    lz, wz = units.l_z, units.omega_z
    Lz = 2.0 * z_extent_lz * lz
    n_particles = filling * L * L
    dtr, dtp = complex_dtype(dtype_relax), complex_dtype(dtype_polish)
    timing = {}
    if verbose:
        print(f"thin-z check on {describe_device()}: {geo.describe()}")
        print(
            f"  V0 = {V0}, Omega_R = {Omega_R}, filling {filling}, omega_z = {wz:.4f} E_R "
            f"(l_z = {lz:.3f} a), g3d = {units.g3d:.5f}, g2d = {units.g2d:.5f}"
        )

    def run(grid, V, g, twist, dtype_r, dtype_p, psi_start=None, interaction=None):
        kw = dict(V=V, g=g, Omega_R=Omega_R, delta=delta, n_components=2, twist=twist, interaction=interaction)
        er = BilayerGPE(grid.to(torch.float32 if dtype_r == torch.complex64 else torch.float64), dtype=dtype_r, **kw)
        ep = er if dtype_r == dtype_p else BilayerGPE(grid, dtype=dtype_p, **kw)
        psi0 = initial_state(er, n_particles) if psi_start is None else psi_start.to(er.dtype)
        r = relax(er, psi0, n_particles, tol=max(tol, 1e-5))
        if ep is not er:
            r = relax(ep, r.psi.to(ep.dtype), n_particles, tol=tol, dt_schedule=[])
        return ep, r

    # -- 2D ----------------------------------------------------------------
    t0 = time.time()
    g2 = Grid((N, N), (L, L), device=device)
    V2 = bilayer_potential(g2, V0, geo)
    e2, r2 = run(g2, V2, units.g2d, None, dtr, dtp)
    synchronize()
    timing["2d_s"] = time.time() - t0
    d2 = e2.density(r2.psi)
    t2 = e2.energy_terms(r2.psi)
    if verbose:
        print(f"  2D ground state: {r2.steps} steps, residual {float(r2.residual.max()):.1e}, {timing['2d_s']:.1f} s")

    # -- 2D with the density-dependent width (NPSE) ---------------------------
    if npse:
        from .npse import NPSE2D

        t0 = time.time()
        en, rn = run(g2, V2, 0.0, None, dtr, dtp, interaction=NPSE2D(units.g2d, wz))
        synchronize()
        timing["2d_npse_s"] = time.time() - t0
        dn = en.density(rn.psi)
        tn = en.energy_terms(rn.psi)
        if verbose:
            print(
                f"  2D NPSE ground state: {rn.steps} steps, residual {float(rn.residual.max()):.1e}, {timing['2d_npse_s']:.1f} s"
            )

    # -- 3D ----------------------------------------------------------------
    t0 = time.time()
    g3 = Grid((N, N, Nz), (L, L, Lz), device=device)
    V3 = bilayer_potential(g3, V0, geo) + harmonic_trap(g3, (0.0, 0.0, wz), m=M_LATTICE)
    e3, r3 = run(g3, V3, units.g3d, None, dtr, dtp)
    synchronize()
    timing["3d_s"] = time.time() - t0
    d3 = e3.density(r3.psi)  # (1, 1, N, N, Nz)
    t3 = e3.energy_terms(r3.psi)
    if verbose:
        print(
            f"  3D ground state ({N}x{N}x{Nz}): {r3.steps} steps, residual {float(r3.residual.max()):.1e}, "
            f"{timing['3d_s']:.1f} s"
        )

    # -- comparison ---------------------------------------------------------
    dz = g3.dx[2]
    col = d3.sum(dim=-1) * dz  # (1, 1, N, N) column density
    z = g3.coord(2)  # (1,1,1,1,Nz)
    z2 = ((d3 * z * z).sum(dim=-1) * dz) / col.clamp_min(1e-30)
    width_ratio = z2 / (0.5 * lz * lz)
    Vmin = float(V2.min())
    mu2, mu3 = float(t2["mu"]), float(t3["mu"]) - 0.5 * wz
    rel = (col - d2).abs().max() / d2.max()
    numbers = dict(
        V0=V0,
        Omega_R=Omega_R,
        filling=filling,
        m=m,
        n=n,
        N=N,
        Nz=Nz,
        Lz=Lz,
        omega_z=wz,
        l_z=lz,
        g2d=units.g2d,
        g3d=units.g3d,
        mu_2d=mu2,
        E_per_N_2d=float(t2["E_total"] / t2["norm"]),
        mu_3d_minus_zpe=mu3,
        E_per_N_3d_minus_zpe=float(t3["E_total"] / t3["norm"]) - 0.5 * wz,
        mu_rel_diff=(mu3 - mu2) / max(mu2 - Vmin, 1e-30),
        pr_2d=float(participation_ratio(d2, g2)),
        pr_3d=float(participation_ratio(col, g2)),
        contrast_2d=float(contrast(d2, g2, 1.0)),
        contrast_3d=float(contrast(col, g2, 1.0)),
        n_peak_2d=float(d2.max()),
        n_peak_3d=float(col.max()),
        density_max_rel_dev=float(rel),
        width_ratio_min=float(width_ratio.min()),
        width_ratio_max=float(width_ratio.max()),
        residual_2d=float(r2.residual.max()),
        residual_3d=float(r3.residual.max()),
    )
    if npse:
        numbers.update(
            mu_2d_npse=float(tn["mu"]),
            E_per_N_2d_npse=float(tn["E_total"] / tn["norm"]),
            pr_2d_npse=float(participation_ratio(dn, g2)),
            n_peak_2d_npse=float(dn.max()),
            contrast_2d_npse=float(contrast(dn, g2, 1.0)),
            density_max_rel_dev_npse=float((col - dn).abs().max() / dn.max()),
            width_ratio_npse_max=float(en.interaction.width(dn).max() ** 2),
            residual_2d_npse=float(rn.residual.max()),
        )
    if with_fs:
        t0 = time.time()
        tw2 = torch.tensor([[theta, 0.0]])
        tw3 = torch.tensor([[theta, 0.0, 0.0]])
        e2t, r2t = run(g2, V2, units.g2d, tw2, dtr, dtp, psi_start=r2.psi)
        e3t, r3t = run(g3, V3, units.g3d, tw3, dtr, dtp, psi_start=r3.psi)
        numbers["fs_2d"] = float(superfluid_fraction_energy(r2t.energy, t2["E_total"], t2["norm"], L, theta, KAPPA))
        numbers["fs_3d"] = float(superfluid_fraction_energy(r3t.energy, t3["E_total"], t3["norm"], L, theta, KAPPA))
        if npse:
            from .npse import NPSE2D

            ent, rnt = run(g2, V2, 0.0, tw2, dtr, dtp, psi_start=rn.psi, interaction=NPSE2D(units.g2d, wz))
            numbers["fs_2d_npse"] = float(
                superfluid_fraction_energy(rnt.energy, tn["E_total"], tn["norm"], L, theta, KAPPA)
            )
        synchronize()
        timing["fs_s"] = time.time() - t0
    res = ThinZResult(
        numbers=numbers,
        column_density=to_numpy(col[0, 0]).astype(np.float32),
        density_2d=to_numpy(d2[0, 0]).astype(np.float32),
        width_map=to_numpy(width_ratio[0, 0]).astype(np.float32),
        timing=timing,
    )
    if verbose:
        print(res.report())
    return res
