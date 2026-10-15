# Changelog

All notable changes to this project are listed here. Versions follow [semantic versioning](https://semver.org).

## 0.1.0 (unreleased)

First public version.

- Two-component Gross-Pitaevskii engine in PyTorch (CPU and CUDA): split-step Fourier with an exact local
  2 x 2 step, Strang (order 2) and Yoshida (order 4) composition, batched over parameter sweeps, twisted
  boundary conditions.
- Ground states by imaginary time with a residual criterion and a preconditioned polish; real-time evolution
  with recorded diagnostics.
- Observables: superfluid fraction from the twist energy and from the current, Leggett bounds, participation
  ratio, moire contrasts as imaged (coherent absorption imaging).
- Quasi-2D (NPSE) reduction and a full 3D thin-z check; trapped clouds from a local-density start;
  Bogoliubov-de Gennes modes; a two-layer tight-binding (Bose-Hubbard) mean field.
- Trapped clouds in full 3D against the NPSE and fixed-g quasi-2D models (`engine.trapped3d`), sized to the
  device by a memory and timing probe: scenarios `gpu_3d_vs_2d` (as large as the GPU holds), `gpu_precision`
  (complex128 against complex64, about 10 minutes) and `trapped_3d_quick`. Trapped runs take a `dtype`.
- Vertical-resolution convergence of the 3D cloud: `engine.trapped3d.nz_convergence`, scenarios `gpu_3d_nz` and
  `trapped_3d_nz_quick`.
- `docs/roadmap-beyond-mean-field.md`: the plan for the lattice model beyond mean field (parameter audit,
  Gutzwiller, cluster Gutzwiller, exact diagonalisation, quantum Monte Carlo). The 3D memory plan carries a
  measured 15 % margin over the probe.
- `twisted-bilayer run NAME --128`: complex128 throughout, overriding the precision a row names. 3D solves
  default to complex64 (validated by `gpu_precision`: errors below 2 % of the 3D - NPSE difference, 16x
  faster on a GTX 1660 Ti); 2D runs stay in complex128.
- Time-of-flight and absorption-imaging pipeline.
- Physics gates against analytic limits and independently reproduced anchors.
- Command line `twisted-bilayer` and examples.
