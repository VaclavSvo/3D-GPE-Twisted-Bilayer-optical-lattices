# Showcase: one trapped cloud, five models

`showcase.ipynb` computes the same trapped cloud of the twisted bilayer (Meng et al. 2023; V₀ = 4 E_R, N = 2000, ν_r = 20 Hz) in:
- full 3D;
- the NPSE;
- fixed g;
- the two-layer lattice GPE;
- Gutzwiller.

It runs at the three couplings, in the 3×3 and 5×5 boxes, and shows images, cuts, observables and the deviations from 3D. All physics comes from the repository (`twisted_bilayer`); `showcase_helpers.py` only wires it together and plots.

## Where it lives

It is in the repository under `examples/showcase/`. The notebook imports `twisted_bilayer` from the installed package, or finds `src/` by walking up from this folder. You also need Jupyter in the same environment: `pip install notebook ipykernel`.

The cached results in `showcase_results/` are git-ignored, so they stay on the machine that computed them. To show results on another machine, copy that folder next to the notebook, or rerun.

## Run

1. Open `showcase.ipynb`. Set `QUICK = True` and run all cells once. That is a one-minute smoke test; its numbers are not physical.
2. Set `QUICK = False` and run all cells again.
   - **On the GTX 1660 Ti (6 GB):** expect roughly 50–80 min from scratch.
     - Continuum models: about 3–5 min for 3×3 and 10–15 min for 5×5, per coupling.
     - Lattice models, on the CPU: about 5 min per coupling.
   - **Memory:** the 5×5 3D solve peaks at about 2.5 GB in complex64.
3. **Do the full run before the meeting.** Results are cached in `showcase_results\`, so the same parameters reload in seconds. Changing a parameter (N, ν_r, a coupling, a box) computes only the new pieces.
   - With the cache in place, a 3×3 change takes a few minutes live.
   - Copy `showcase_results\` along with the notebook if you compute on another machine.

## Parameters (first code cell)

| parameter | meaning |
| --- | --- |
| `COUPLINGS` | J⊥ in E_R. 0.121, 0.150 and 0.243 use the exact values from `scenarios.J_PERP_ER`; any other number is taken as E_R |
| `BOXES` | odd numbers of √485 a supercells |
| `MODELS` | which of the five to run |
| `N_ATOMS`, `NU_R_HZ` | the cloud (not published by the paper) |
| `V0_ER`, `NU_Z_HZ` | the lattice depth and the pancake frequency |
| `CELL_POINTS`, `NZ`, `DTYPE_3D` | resolution and precision; the defaults are the converged settings |
| `S_SAT` | imaging saturation used for C_coh |
| `SHOW_J`, `SHOW_BOX` | which cloud the image panels show |
| `FORCE` | recompute even if cached |

## Notes

- **Lattice models and the box.** The lattice models run in the 3×3 box and are padded with empty sites for 5×5. A 5×5 lattice model needs more than 6 GB of RAM for its site tables, and the cloud fits in 3×3.
- **The lattice models' settings.**
  - Variant C: the interlayer overlap cut at |d′| < 2, plus the interlayer interaction.
  - The interband shift, site part only.
  - U from the 1 kHz pancake.
  - The same harmonic trap as the continuum, applied on the sites.
  - Gutzwiller is solved at the lattice GPE's μ, then one Newton step brings it to the same N.
- **What to expect.** At 4 E_R the lattice models are expected to be the furthest from 3D: the tight-binding description is marginal that shallow (U/J₁ = 0.41, J₂/J₁ = −7 %). The NPSE should sit within a few 10⁻³ of 3D in C_coh, and fixed g a few 10⁻² below it.
