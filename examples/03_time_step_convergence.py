"""Convergence of the real-time integrator: Strang (order 2) against Yoshida (order 4).

All atoms start in |1> in the lattice ground state; at t = 0 the microwave coupling is switched on and the
atoms Rabi-oscillate into |2> while interacting. The local 2 x 2 step is exact, so the only error is the
operator splitting: the final state's error against a fine-step reference falls as dt^2 and dt^4. Norm and
energy are conserved along the way. About 15 s on a laptop CPU.

    python examples/03_time_step_convergence.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from twisted_bilayer.engine import (
    BilayerGPE,
    Grid,
    bilayer_potential,
    initial_state,
    lattice_units,
    moire_geometry,
    relax,
)

OUT = pathlib.Path(__file__).with_name("output")
OUT.mkdir(exist_ok=True)

geo = moire_geometry(4, 1)
N = geo.grid_size(8)
grid = Grid((N, N), (geo.L_box, geo.L_box))
V = bilayer_potential(grid, 4.0, geo)
g2d = lattice_units().g2d
n_atoms = grid.volume

ground = BilayerGPE(grid, V=V, g=g2d, Omega_R=0.0)
psi0 = relax(ground, initial_state(ground, n_atoms, fractions=(1.0, 0.0)), n_atoms, tol=1e-10).psi

eng = BilayerGPE(grid, V=V, g=g2d, Omega_R=0.243)
T = 1.0  # hbar / E_R: N2/N reaches ~6 %


def evolve(order: int, n_steps: int) -> torch.Tensor:
    return eng.run_block(psi0.clone(), n_steps, T / n_steps, order=order)


ref = evolve(4, 4096)  # order 4 at dt = 2.4e-4: error ~1e-13, below every point
E_start = float(eng.energy_terms(psi0)["E_total"][0])
steps = np.array([32, 64, 128, 256, 512, 1024])
errors = {2: [], 4: []}
for order in (2, 4):
    for n in steps:
        psi = evolve(order, int(n))
        errors[order].append(float((psi - ref).norm() / ref.norm()))
    psi = evolve(order, 256)
    t = eng.energy_terms(psi)
    print(
        f"order {order}: error {errors[order][0]:.1e} at dt = {T / steps[0]:.3f} -> {errors[order][-1]:.1e} at "
        f"dt = {T / steps[-1]:.4f};  256 steps: |dN/N| = {abs(float(t['norm'][0]) / n_atoms - 1):.1e}, "
        f"|dE/E| = {abs(float(t['E_total'][0]) / E_start - 1):.1e}"
    )
    slope = np.polyfit(np.log(T / steps), np.log(errors[order]), 1)[0]
    print(f"         fitted slope {slope:.2f}")

dts = T / steps
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
for order, mark in ((2, "o-"), (4, "s-")):
    ax.loglog(dts, errors[order], mark, label=f"order {order}")
    ax.loglog(dts, errors[order][0] * (dts / dts[0]) ** order, ":", color="0.5")
ax.set_xlabel(r"time step $dt$ ($\hbar/E_R$)")
ax.set_ylabel(r"$\|\psi - \psi_\mathrm{ref}\| / \|\psi_\mathrm{ref}\|$")
ax.set_title(r"split-step error at $t = \hbar/E_R$ (dotted: $dt^2$, $dt^4$)", fontsize=9)
ax.legend(frameon=False)
fig.savefig(OUT / "03_time_step_convergence.png", dpi=150)
print(f"wrote {OUT / '03_time_step_convergence.png'}")
