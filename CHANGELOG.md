# Changelog

User-facing changes of the package, newest first. Versions follow [semantic versioning](https://semver.org).

## Unreleased

- The showcase notebook runs with `QUICK = True`: in 0.2.0 and 0.3.0 it stopped with `UncontainedCloud`; the quick
  cloud is now trapped at 120 Hz instead of 80 Hz and fits its one-supercell box.

## 0.3.0 (November 2026)

- The lattice models moved to their own package, [Twisted-Bilayer-Lattice-models](https://github.com/VaclavSvo/Twisted-Bilayer-Lattice-models)
  (`twisted_bilayer_lattice`): `engine.lattice_model`, the whole `manybody` subpackage, the scenario kinds
  `tight_binding` and `phase_diagram`, their tests, `docs/manybody.md` and the two GPU scripts of the batched
  eigensolver. The scenario kind `bridge` and `manybody.parameters.occupation_dependent_U` are no longer part of
  this package. This package is the Gross-Pitaevskii code only; every continuum result of 0.2.0 is unchanged.
- The showcase example compares the three continuum models; the five-model comparison is in the lattice
  repository.
- `parameters` and `engine.units` import their siblings relatively (no change in behaviour).
- The two reference values of the square-lattice Mott transition (`square_lattice_UJ_c_gutzwiller`,
  `square_lattice_UJ_c_qmc`) moved from `gates/references.py` to `twisted_bilayer_lattice.references`.

## 0.2.0 (November 2026)

The first tagged version: the state of the code before it is split into a continuum (Gross-Pitaevskii) package and a
lattice-model package.

The tag `v0.2.0` of this repository does not contain `manybody/bridge.py` and its test, so `twisted_bilayer.manybody`
cannot be imported at that tag; the lattice models of 0.2.0 run from the lattice repository (0.3.0).

### Continuum engine

- Two-component Gross-Pitaevskii engine in PyTorch (CPU and CUDA): split-step Fourier with an exact local 2 x 2 step,
  Strang (order 2) and Yoshida (order 4) composition, batched over parameter sweeps, twisted boundary conditions.
- Ground states by imaginary time with a residual criterion and a preconditioned polish; `RelaxResult.stop_reason`
  says why a solve ended (`tol`, `stall`, `slow`, `max_steps`, `tau_floor`, `ladder`); a complex128 polish stops a
  descent that cannot reach `tol` within its step budget, complex64 solves stop at their rounding floor.
- Real-time evolution with recorded diagnostics (norm, energy, high-k weight).
- Quasi-2D NPSE reduction, a thin-z full-3D check, and trapped clouds in full 3D against the NPSE and fixed-g quasi-2D
  models (`engine.trapped3d`), sized to the device by a memory and timing probe; vertical (`nz_convergence`) and
  in-plane (`engine.convergence`) resolution scans.
- Trapped clouds must fit their box: every trapped run compares the infinite-plane Thomas-Fermi radius with the box
  and refuses a cloud that does not fit (`UncontainedCloud`; `containment = "enlarge"` or `"warn"` instead), and
  every summary carries `checks` (`contained`, `converged`, `at_precision_floor`, `ok`).
- `twisted-bilayer run NAME --128` forces complex128 throughout.
- Observables: superfluid fraction from the twist energy and from the current, and their combination `fs_xx`,
  `fs_yy` (the twist-free value to O(theta^4)); Leggett bounds; participation ratio; moire contrasts as imaged
  (coherent absorption imaging), `C_smooth` (no ringing) and the `raw_ringing` flag.
- Bogoliubov-de Gennes modes (`engine.bdg`), including the SU(2) spin-rotation zero mode and one-component limits.
- Time-of-flight and absorption-imaging pipeline (`tof`), with the release width of an NPSE state (`s0="npse"`), the
  Raman coupling kept on through the whole flight, and a guard on the high-k weight of the mean-field tail.

### Lattice models

- A two-layer tight-binding (Bose-Hubbard) model of the twisted bilayer (`engine.lattice_model`): discrete-GPE ground
  states with a trust-region Newton polish, the second-order interband shift (`interband=True`), superfluid fraction
  with error bounds and order checks; the paper's one-layer model (`manybody.paper_model`).
- Bose-Hubbard parameters from the continuum bands (`manybody.parameters`) and the `bridge` scenario kind: the lattice
  mean field against the continuum GPE on the same cell.
- Single-site Gutzwiller (`manybody.gutzwiller`; twisted, batched; `solve_filling`, `compressibility`, fixed-N
  superfluid fraction), cluster mean field (`manybody.cluster`), observables and phase labels
  (`manybody.observables`), and resumable phase maps over (V0, Omega_R) (`manybody.phase_diagram`, the scenario kind
  `phase_diagram`).
- `solve_batch(..., eig="fast")` with `manybody.fast_kernels`: a certified tridiagonal eigensolver for the site
  problem; the default `eig="exact"` is unchanged.
- On CUDA the batched site eigenproblem is cut into chunks (`EIGH_BATCH_MAX`), halved and retried after a cuSOLVER or
  out-of-memory failure.

### Running and checking

- Command line `twisted-bilayer` (`list`, `run`, `gates`, `panel`) and examples.
- Physics gates against analytic limits and independently reproduced anchors (`twisted-bilayer gates`); thresholds
  and sources in `gates/references.py`; NaN-safe drift gates; an empty or unknown selection is a usage error (exit 2).
- Check panels `twisted-bilayer panel quick` and `panel broad`; `check.py` (lint, format, tests; `--all` adds the
  slow tests).
- Run records (`twisted_bilayer.records`): every run kept in `results/<kind>/<date_time>/` with `latest.txt` and
  `history.csv`.
- Exit codes of `twisted-bilayer run`: 0 success, 1 a failed physics check (`rabi`, `gpu_check`), 2 a usage error,
  **3 a trapped run that finished but whose cloud is not contained** (edge density at or above 1e-3 of the peak; its
  files are written first, `checks.contained` is false).
- Documentation: `docs/methods.md` (the methods), `docs/validation.md` (the gate table), `docs/manybody.md` (the
  lattice models).

### Inputs and solver behaviour

- Ground states: in complex128 a window in which the residual rises while the energy falls by more than its rounding
  level is treated as descent, not as an overshoot, also right after a restart; deep-lattice solves that used to
  halve their step down to `tau_floor` on such a plateau now go on to the tolerance. complex64 solves are unchanged.
- `initial_state`: the default relative phase of the second component is chosen per batch member from the sign of
  Omega_R (each member starts on its lower dressed branch); a batch that mixes signs needs `batch=len(Omega_R)`
  (`ValueError` otherwise). Batches without mixed signs start exactly as before.
- `BilayerGPE.kappa` and `BilayerGPE.grid` are read-only.
- `BilayerGPE` checks its inputs at construction and raises `ValueError` for a non-finite potential or parameter,
  kappa <= 0, g < 0, batched parameters of different lengths, or a non-complex dtype; the new keyword
  `allow_attractive=True` admits g < 0. `set_twist` refuses a non-finite twist.
- Docstrings with units and meaning for every parameter of `BilayerGPE`, `relax`, `precondition_polish`,
  `initial_state`, `TrappedSpec`, `Vertical`, `SweepSpec`, `coupling_scan`, `solve_models` and `thin_z_check`;
  a parameter reference in `docs/methods.md`. Return type hints on the public functions.

### Fixed (since the development versions)

- Every saved `fields/state_OmegaR_*.npz` of a coupling scan stores the total density as `n` again (it was overwritten
  by a geometry index; the indices are now `geometry_m`, `geometry_n`).
- `nbar_centre_cell` gives the boundary rows of the central moire cell half weight, so it does not depend on the box.
- The interband bond corrections of layer 1 include the bonds across the box boundary (both layers now identical).
- `bdg`: a dependent zero-mode candidate is dropped; the CG solver keeps an exactly converged column finite; a
  batched potential is refused.
- `convergence.needed` ignores unconverged rows and rows whose cloud is not contained; repeated grid sizes run once.
- `rabi_quick` runs at the shipped coupling `J_PERP_ER[0.243]`; its conservation gates use the suite's tolerances.

## 0.1.0 (October 2026, development)

Untagged development versions on the main branch; their history is in the git log.
