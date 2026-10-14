"""Superfluid fraction against the interlayer coupling, two ways, with Leggett's bounds.

A phase twist theta across the periodic box costs kinetic energy in proportion to the superfluid density:
f_s = (E(theta) - E(0)) / (N kappa theta^2 / L^2). The same f_s follows from the current the twist drives.
Both must lie inside Leggett's rigorous bounds computed from the untwisted density. The couplings are one
batch: Omega_R is a tensor, and every engine call works on all of them at once.

The small m = 4 cell (a 28 deg twist) keeps this to about 10 s on a laptop CPU.

    python examples/02_superfluid_fraction.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from twisted_bilayer.engine import (
    KAPPA,
    BilayerGPE,
    Grid,
    bilayer_potential,
    initial_state,
    lattice_units,
    moire_geometry,
    relax,
)
from twisted_bilayer.engine.bands import hopping_1d
from twisted_bilayer.engine.observables import (
    leggett_bounds,
    superfluid_fraction_current,
    superfluid_fraction_energy,
)

OUT = pathlib.Path(__file__).with_name("output")
OUT.mkdir(exist_ok=True)

V0, theta = 8.0, 0.3
omegas = torch.tensor([0.0, 0.05, 0.1, 0.24, 0.5, 1.0], dtype=torch.float64)  # E_R, one batch
geo = moire_geometry(4, 1)
N = geo.grid_size(12)
grid = Grid((N, N), (geo.L_box, geo.L_box))
V = bilayer_potential(grid, V0, geo)
g2d = lattice_units().g2d
n_atoms = grid.volume  # one atom per a^2


def engine(twist=None):
    return BilayerGPE(grid, V=V, g=g2d, Omega_R=omegas, twist=twist)


eng0 = engine()
r0 = relax(eng0, initial_state(eng0, n_atoms, batch=len(omegas)), n_atoms, tol=1e-9)
E0 = eng0.energy_terms(r0.psi)

tw = torch.zeros(len(omegas), 2, dtype=torch.float64)
tw[:, 0] = theta  # twist along x for every coupling
eng_t = engine(tw)
rt = relax(eng_t, r0.psi, n_atoms, tol=1e-9)

fs_energy = superfluid_fraction_energy(rt.energy, E0["E_total"], E0["norm"], geo.L_box, theta, KAPPA)
fs_current = superfluid_fraction_current(eng_t, rt.psi, 0, theta)
lower, upper = leggett_bounds(eng0.density(r0.psi), grid, 0)

J1 = hopping_1d(V0, d_max=1)[0]
print(f"{'Omega_R/J1':>10s} {'f_s(energy)':>12s} {'f_s(current)':>13s} {'Leggett bounds':>18s}")
for i, om in enumerate(omegas.tolist()):
    print(
        f"{om / J1:10.2f} {float(fs_energy[i]):12.5f} {float(fs_current[i]):13.5f}"
        f"   [{float(lower[i]):.4f}, {float(upper[i]):.4f}]"
    )

x = omegas.numpy() / J1
fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
ax.fill_between(x, lower.cpu().numpy(), upper.cpu().numpy(), color="0.85", label="Leggett bounds")
ax.plot(x, fs_energy.cpu().numpy(), "o-", label="twist energy")
ax.plot(x, fs_current.cpu().numpy(), "x--", label="twist current")
ax.set_xscale("symlog", linthresh=1.0)
ax.set_xlabel(r"$\Omega_R / J_1$")
ax.set_ylabel(r"superfluid fraction $f_s^{xx}$")
ax.set_ylim(0, 1)
ax.legend(frameon=False)
fig.savefig(OUT / "02_superfluid_fraction.png", dpi=150)
print(f"wrote {OUT / '02_superfluid_fraction.png'}")
