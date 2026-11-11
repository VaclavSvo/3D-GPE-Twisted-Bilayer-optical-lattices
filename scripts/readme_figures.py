"""Draw the README figure, docs/figures/twisted_bilayer.png (about 5 minutes on 2 CPU cores).

The experiment's supercell (tan(theta/2) = 1/22, theta = 5.2051 deg, side sqrt(485) a = 2 x 2 moire cells) at
V0 = 8 E_R per lattice and the microwave coupling Omega_R = J_perp = 0.150 E_R, one atom per lattice site, NPSE
vertical reduction, complex128:
  (a) the lattice of state |2> and the moire pattern of the two twisted lattices, V1 + V2;
  (b) the ground-state density of |2>;
  (c) the momentum distribution |psi_2(k)|^2 of that state, the far-field (long time-of-flight) image without
      interactions during the flight: the Bragg peaks of both lattices, rotated by +-theta/2, and the moire
      satellites around them.

    python scripts/readme_figures.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from twisted_bilayer.engine import (
    NPSE2D,
    BilayerGPE,
    Grid,
    bilayer_potential,
    initial_state,
    lattice_units,
    moire_geometry,
    relax,
)
from twisted_bilayer.scenarios import J_PERP_ER

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "twisted_bilayer.png"
V0, OMEGA_R = 8.0, J_PERP_ER[0.150]


def main() -> None:
    geo = moire_geometry(22, 1)
    N = geo.grid_size(8)  # 8 points per a: converged for supercell ground states
    grid = Grid((N, N), (geo.L_box, geo.L_box))
    lu = lattice_units()
    V = bilayer_potential(grid, V0, geo)
    eng = BilayerGPE(grid, V=V, Omega_R=OMEGA_R, interaction=NPSE2D(lu.g2d, lu.omega_z))
    n_atoms = geo.L_box**2
    r = relax(eng, initial_state(eng, n_atoms), n_atoms, tol=1e-9)
    print(f"mu = {float(r.mu[0]):.6f} E_R, residual {float(r.residual[0]):.1e}, {r.steps} steps")

    psi2 = r.psi[0, 1].cpu()
    n2 = (psi2.abs() ** 2).numpy()  # layout [ix, iy]
    nk = torch.fft.fftshift(torch.fft.fft2(psi2).abs() ** 2).numpy()
    nk /= nk.max()
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=geo.L_box / N))  # rad / a
    k_rec = 2 * np.pi  # the reciprocal lattice constant 2 pi / a

    half = geo.L_box / 2
    ext = [-half, half, -half, half]
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)
    im = ax[0].imshow(V[0].sum(0).cpu().numpy().T, origin="lower", extent=ext, cmap="viridis")
    ax[0].set_title(r"(a) the two lattices, $V_1 + V_2$")
    fig.colorbar(im, ax=ax[0], shrink=0.85, label=r"$E_R$")
    im = ax[1].imshow(n2.T, origin="lower", extent=ext, cmap="magma")
    ax[1].set_title(r"(b) density of $|2\rangle$")
    fig.colorbar(im, ax=ax[1], shrink=0.85, label=r"atoms / $a^2$")
    for a in ax[:2]:
        a.set_xlabel("x / a")
        a.set_ylabel("y / a")
    kk = k / k_rec
    sel = np.abs(kk) <= 1.2
    im = ax[2].imshow(
        nk[np.ix_(sel, sel)].T,
        origin="lower",
        extent=[kk[sel][0], kk[sel][-1], kk[sel][0], kk[sel][-1]],
        cmap="inferno",
        norm=LogNorm(vmin=1e-5, vmax=1),
        interpolation="nearest",
    )
    ax[2].set_title(r"(c) $|2\rangle$ after time of flight, $|\tilde\psi_2(k)|^2$")
    ax[2].set_xlabel(r"$k_x \, a / 2\pi$")
    ax[2].set_ylabel(r"$k_y \, a / 2\pi$")
    fig.colorbar(im, ax=ax[2], shrink=0.85, label="relative weight")
    fig.suptitle(
        rf"Two square lattices twisted by $\theta = 5.2^\circ$: $V_0 = {V0:g}\,E_R$, $\Omega_R = {OMEGA_R:.3f}\,E_R$, "
        r"one atom per site, on the periodic $\sqrt{485}\,a$ supercell (NPSE)"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"-> {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
