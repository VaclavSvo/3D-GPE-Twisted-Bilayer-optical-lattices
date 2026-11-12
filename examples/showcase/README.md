# Showcase: one trapped cloud, three models

`showcase.ipynb` computes the same trapped cloud of the twisted bilayer (Meng et al., Nature 615, 231 (2023);
V₀ = 4 E_R, N = 2000, ν_r = 20 Hz) with three models:

- full 3D;
- the NPSE (quasi-2D);
- fixed g (quasi-2D).

The comparison with the lattice (Bose–Hubbard) models of the same cloud is in the lattice repository,
[Twisted-Bilayer-Lattice-models](https://github.com/VaclavSvo/Twisted-Bilayer-Lattice-models).

It runs at three couplings, in 3 × 3 and 5 × 5 supercell boxes, and shows the images as a camera records
them, cuts, the observables and each model's deviation from full 3D. All physics comes from the package
(`twisted_bilayer`); `showcase_helpers.py` only wires it together and plots.

## Requirements

The package installed (`pip install -e ".[dev]"` from the repository root; the notebook also finds `src/` by
walking up from this folder) and Jupyter in the same environment: `pip install notebook ipykernel`. The
notebook is stored without outputs; running it draws everything.

## Run

1. Open `showcase.ipynb`, set `QUICK = True` and run all cells: a one-minute smoke test whose numbers are not
   physical.
2. Set `QUICK = False` and run all cells again. From scratch, on a GPU with 6 GiB of memory, this takes about
   40–60 minutes: about 3–5 min (3 × 3) and 10–15 min (5 × 5) per coupling. The 5 × 5 3D solve peaks at about
   2.5 GiB in complex64.
3. Results are cached in `showcase_results/` (git-ignored), so a rerun with the same parameters reloads in
   seconds, and changing one parameter (N, ν_r, a coupling, a box) computes only the new pieces.

## Parameters (first code cell)

| parameter | meaning |
| --- | --- |
| `COUPLINGS` | J⊥ in E_R. 0.121, 0.150 and 0.243 use the exact values from `scenarios.J_PERP_ER`; any other number is taken as E_R |
| `BOXES` | odd numbers of √485 a supercells |
| `MODELS` | which of the three to run |
| `N_ATOMS`, `NU_R_HZ` | the cloud (atom number and radial trap frequency) |
| `V0_ER`, `NU_Z_HZ` | the lattice depth and the pancake frequency |
| `CELL_POINTS`, `NZ`, `DTYPE_3D` | resolution and precision; the defaults are converged settings |
| `S_SAT` | imaging saturation used for C_coh |
| `SHOW_J`, `SHOW_BOX` | which cloud the image panels show |
| `FORCE` | recompute even if cached |

## Notes

- **What to expect.** The NPSE sits within a few 10⁻³ of 3D in C_coh, and fixed g a few 10⁻² below it.
