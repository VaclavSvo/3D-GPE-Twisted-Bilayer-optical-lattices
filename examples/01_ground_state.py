"""Ground state of the condensate on the experiment's moire supercell.

Two square lattices of depth V0 = 4 E_R, twisted by theta = 2 atan(1/22) = 5.2051 deg, one per hyperfine
state, coupled by a microwave field Omega_R. The periodic box holds 2 x 2 moire cells (side sqrt(485) a).
The vertical confinement is folded into 2D with the NPSE. About 15 s on a laptop CPU.

    python examples/01_ground_state.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

OUT = pathlib.Path(__file__).with_name("output")
OUT.mkdir(exist_ok=True)

geo = moire_geometry(22, 1)  # tan(theta/2) = 1/22: theta = 5.2051 deg, box side L = sqrt(485) a
N = geo.grid_size(8)  # 8 grid points per lattice constant
grid = Grid((N, N), (geo.L_box, geo.L_box))
lu = lattice_units()  # Rb-87, a = 395 nm, 1 kHz vertical trap
V = bilayer_potential(grid, 4.0, geo)  # V0 = 4 E_R per lattice, shape (1, 2, N, N)
eng = BilayerGPE(grid, V=V, Omega_R=0.24277, interaction=NPSE2D(lu.g2d, lu.omega_z))

n_atoms = geo.L_box**2  # one atom per lattice site
r = relax(eng, initial_state(eng, n_atoms), n_atoms, tol=1e-9)
print(f"mu = {float(r.mu[0]):.6f} E_R  (residual {float(r.residual[0]):.0e}, {r.steps} steps)")

n = (r.psi[0].abs() ** 2).cpu().numpy()  # (2, N, N): the densities of the two states, layout [ix, iy]

ext = [-geo.L_box / 2, geo.L_box / 2, -geo.L_box / 2, geo.L_box / 2]
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
im0 = ax[0].imshow(V[0].sum(0).cpu().numpy().T, origin="lower", extent=ext, cmap="viridis")
ax[0].set_title(r"$V_1 + V_2$ ($E_R$)")
fig.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(n[1].T, origin="lower", extent=ext, cmap="magma")
ax[1].set_title(r"density of $|2\rangle$ (atoms / $a^2$)")
fig.colorbar(im1, ax=ax[1], shrink=0.8)
for a in ax:
    a.set_xlabel("x / a")
    a.set_ylabel("y / a")
fig.savefig(OUT / "01_ground_state.png", dpi=150)
print(f"wrote {OUT / '01_ground_state.png'}")
