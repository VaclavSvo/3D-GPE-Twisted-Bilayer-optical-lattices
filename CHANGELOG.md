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
- Time-of-flight and absorption-imaging pipeline.
- Physics gates against analytic limits and independently reproduced anchors.
- Command line `twisted-bilayer` and examples.
