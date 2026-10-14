# twisted-bilayer: Gross–Pitaevskii simulations of twisted-bilayer optical lattices

[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![python: 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`twisted-bilayer` solves the two-component Gross–Pitaevskii equation for a Bose–Einstein condensate held in
two square optical lattices twisted against each other, one lattice per hyperfine state, coupled by a
microwave field. It is written in PyTorch and runs on CPU and CUDA. It covers the exactly periodic moiré
supercell, trapped clouds, real-time dynamics, and time of flight with absorption imaging. Every run can be
checked against a catalogue of physics gates.

The model follows the experiment of Meng *et al.*,
[*Atomic Bose–Einstein condensate in twisted-bilayer optical lattices*, Nature **615**, 231 (2023)](https://doi.org/10.1038/s41586-023-05695-4):
⁸⁷Rb in |F=1, m=1⟩ and |F=2, m=0⟩, lattice constant a = 395 nm, twist θ = 5.2°, and a pancake-shaped
vertical trap folded into 2D. The scope is mean-field: ground states, real-time evolution, Bogoliubov
excitations, and a tight-binding cross-check.

<p align="center"><img src="docs/figures/ground_state.png" alt="Moiré potential of the two twisted lattices and the ground-state density of one component" width="760"></p>

*Left: the sum of the two lattice potentials on the periodic supercell (θ = 5.2051°, side √485 a, 2 × 2
moiré cells). Right: the density of |2⟩ in the ground state at V₀ = 4 E_R and Ω_R = 0.243 E_R.
`python examples/01_ground_state.py` makes this figure in about 15 seconds.*

## Highlights

- **Exact local step.** The split-step Fourier solver exponentiates the local 2 × 2 Hamiltonian exactly, so
  operator splitting is the only time-step error. The Strang (order 2) and Yoshida (order 4) compositions
  converge with fitted slopes 2.00 and 4.00, and the norm is conserved to 10⁻¹³
  ([example 3](examples/03_time_step_convergence.py)).
- **Batched parameter sweeps.** The coupling Ω_R, the detuning, the interaction and the phase twist can be
  tensors with a batch axis, so a whole scan relaxes in one set of FFT calls on the GPU.
- **Superfluid fraction two ways.** It comes from the energy of a phase twist and, independently, from the
  current that twist drives. The two routes agree to 4 × 10⁻⁵ and are checked against Leggett's rigorous
  bounds.
- **Exactly periodic moiré cells.** Commensurate twists tan(θ/2) = n/m give a periodic square box. The
  experiment's angle is n/m = 1/22.
- **More than one model.** The thin vertical direction is handled by the NPSE reduction and checked against
  the full 3D problem. The package also covers trapped clouds from a local-density start,
  Bogoliubov–de Gennes modes, a two-layer Bose–Hubbard mean field, and the time-of-flight and coherent
  absorption-imaging pipeline.
- **Physics gates.** 89 gates check the engine against analytic results, exact identities and numbers
  reproduced by independent solvers that are not part of this package
  ([docs/validation.md](docs/validation.md)).

## Validation

A selection from `twisted-bilayer gates --full --trapped` (89 gates, all passing; CPU, float64). The full table is in
[docs/validation.md](docs/validation.md).

| check | error | reference |
| --- | --- | --- |
| harmonic oscillator: ground-state energy and widths (2D) | 1.2 × 10⁻¹³ | exact |
| free expansion of a Gaussian, σ(t) | 1.8 × 10⁻¹³ | exact |
| detuned two-level Rabi oscillation, N₂/N(t) | 1.1 × 10⁻¹⁵ | exact |
| band bottom of V₀ sin²(πx) at 4 E_R, on the √485 supercell | 6.2 × 10⁻¹² | plane-wave diagonalisation |
| Bloch weight \|c₀\|⁴ at 4 E_R | 4.9 × 10⁻⁹ | plane-wave diagonalisation |
| gauge symmetry H(−Ω) = σ_z H(Ω) σ_z | 1.6 × 10⁻¹⁶ | exact identity |
| time reversal dt → −dt after 400 steps | 7.3 × 10⁻¹⁴ | exact identity |
| superfluid fraction, twist energy against twist current | 4.1 × 10⁻⁵ | two independent routes |
| dressed band bottom at Ω_R = 0.243 E_R | 2.2 × 10⁻¹⁰ | independent quasi-2D and 3D solvers |
| chemical potential at Ω_R = 0.243 E_R, one atom per site | 1.8 × 10⁻⁷ | independent quasi-2D solver |

<p align="center"><img src="docs/figures/time_step_convergence.png" alt="Error of the split-step integrator against the time step for orders 2 and 4" width="440"></p>

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/VaclavSvo/3D-GPE-Twisted-Bilayer-optical-lattices.git
cd 3D-GPE-Twisted-Bilayer-optical-lattices
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only (see below)
pip install -e ".[dev]"                       # numpy, scipy, matplotlib, torch + pytest, ruff
```

On Linux, a plain `pip install torch` pulls the CUDA build (an environment of about 6 GB). The line above installs the
CPU build instead. For a GPU, take the command for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/) instead. `pip install .` (not editable) works as
well.

## Quick start

```bash
twisted-bilayer list                          # the scenarios
twisted-bilayer run coupling_scan_quick       # ~15 s; files in results/coupling_scan_quick/
twisted-bilayer run rabi_quick --out runs     # files in runs/rabi_quick/
twisted-bilayer gates                         # the physics gates, ~45 s
python check.py                               # lint + the test suite, ~5 min
```

`coupling_scan_quick` finds the ground state at five couplings in one batch. For each, it measures the
superfluid fraction, the Leggett bounds and how the atoms gather into the moiré pockets. It writes a
six-panel figure, a table, the full `.csv`, the wave functions, and a `provenance.json` with the package
version, git commit, library versions and device:

```
 Omega_R  Omega_over_J  E_per_N      mu  fs_xx_energy  fs_xx_current  leggett_x_lower  leggett_x_upper
       0             0   4.9885  5.0045       0.28253        0.28244          0.26673          0.99882
     0.1        3.2468   4.9272   4.944       0.30064        0.30055          0.26609          0.99537
    0.24        7.7924   4.8347  4.8541       0.31178        0.31168          0.25726          0.96279
     0.5        16.234   4.6457  4.6729       0.30305        0.30292          0.22821          0.85176
       1        32.468   4.2367  4.2855       0.24014        0.23994          0.16413          0.59207
```

The quick scenarios use the small m = 4 cell, a 28° twist, so they finish in seconds. Use them to check
that everything works; they do not reproduce the experiment's geometry.

## Using the engine from Python

```python
from twisted_bilayer.engine import NPSE2D, BilayerGPE, Grid, bilayer_potential, initial_state, lattice_units
from twisted_bilayer.engine import moire_geometry, relax

geo = moire_geometry(22, 1)  # tan(theta/2) = 1/22: theta = 5.2051 deg, box side sqrt(485) a
N = geo.grid_size(8)  # 8 grid points per lattice constant
grid = Grid((N, N), (geo.L_box, geo.L_box))
lu = lattice_units()  # Rb-87, a = 395 nm, 1 kHz vertical trap
eng = BilayerGPE(grid, V=bilayer_potential(grid, 4.0, geo), Omega_R=0.24277, interaction=NPSE2D(lu.g2d, lu.omega_z))

n_atoms = geo.L_box**2  # one atom per lattice site
r = relax(eng, initial_state(eng, n_atoms), n_atoms, tol=1e-9)
print(f"mu = {float(r.mu[0]):.6f} E_R, residual {float(r.residual[0]):.0e}")
n = (r.psi[0].abs() ** 2).cpu().numpy()  # (2, N, N): the densities of the two states
```

```
mu = 2.908446 E_R, residual 7e-10
```

Quantities are in lattice units: lengths in a, energies in the recoil energy E_R = h × 3678 Hz, ħ = 1.
`lattice_units()` converts to SI.

## Examples

| script | what it shows | time |
| --- | --- | --- |
| [`01_ground_state.py`](examples/01_ground_state.py) | ground state on the experiment's supercell and the figure above | 15 s |
| [`02_superfluid_fraction.py`](examples/02_superfluid_fraction.py) | superfluid fraction against the coupling, by twist energy and current, with Leggett's bounds; one batched solve | 10 s |
| [`03_time_step_convergence.py`](examples/03_time_step_convergence.py) | order-2 and order-4 convergence of the real-time integrator, norm and energy conservation | 15 s |
| [`my_scenarios.py`](examples/my_scenarios.py) | your own scenario rows for the command line: `twisted-bilayer run scan_v6_quick --scenarios examples/my_scenarios.py` | 15 s |

Figures go to `examples/output/`. Times are for 2 CPU cores.

## Scenarios

`twisted-bilayer run <name>` runs a named row of [`scenarios.py`](src/twisted_bilayer/scenarios.py), or of
your own file with `--scenarios FILE`. Times are for 2 CPU cores.

| name | what it computes | time |
| --- | --- | --- |
| `coupling_scan_quick` | superfluid fraction, Leggett bounds, μ and pocket formation against the coupling (m = 4 cell) | 15 s |
| `trapped_quick` | a small trapped cloud: local-density start, ground state, contrasts as imaged | 10 s |
| `rabi_quick` | real-time Rabi transfer between the two states inside the lattice | 15 s |
| `toy1d` | the one-dimensional two-lattice model: exact hopping, superfluid fraction by twist | 10 s |
| `thin_z_quick` | the full 3D problem with a thin vertical direction against the quasi-2D reduction | 20 s |
| `bdg` | Bogoliubov–de Gennes modes: sound speed and the spin branch | 1.5 min |
| `tight_binding` | the two-layer Bose–Hubbard mean field on the experiment's supercell, three variants | 1.5 min |
| `coupling_scan_v4` | coupling scan at V₀ = 4 E_R on the experiment's supercell (θ = 5.2°) | 6 min |
| `trapped` | a cloud of N = 2000 atoms in a 20 Hz trap on 3 × 3 supercells, and its contrast as imaged | 6 min |
| `gpu_check` | the quick scan on the GPU and on the CPU, every number compared; then a production timing | GPU |

## How it works, in one paragraph

The two components evolve under the Hamiltonian of the paper's Supplement (Eqs. 4, 5, 10, 11): kinetic
energy, one twisted lattice per component, a detuning ±Δ/2, an SU(2)-symmetric contact interaction g n with
n = |ψ₁|² + |ψ₂|², and the coupling Ω_R as the off-diagonal element. Time stepping is split-step Fourier.
The kinetic part is diagonal in momentum. The local part is a 2 × 2 matrix whose exponential is exact,
because n is conserved by it. Twisted boundary conditions ψ(r + L) = e^{iθ} ψ(r) are carried as a momentum
shift of a periodic field. Ground states come from imaginary time with an adaptive step ladder and a
residual criterion ‖Hψ − μψ‖/‖ψ‖, then a preconditioned polish. The vertical direction is folded in by the
NPSE, a density-dependent Gaussian width, and checked against a 3D grid. Imaging applies the coherent
transfer function of the experiment's objective, and time of flight propagates each Bragg order separately
after a mean-field stage. The modules and their equations are documented in their docstrings.

## Repository layout

```
src/twisted_bilayer/
  engine/        the solver: physics.py (BilayerGPE), ground_state.py, evolve.py, grid.py, lattice.py,
                 npse.py, observables.py, imaging.py, sweep.py, trapped.py, thin_z.py, bdg.py, bands.py,
                 lattice_model.py, units.py, backend.py
  tof/           time of flight and absorption imaging
  gates/         physics gates: core.py, references.py (every reference number), suite.py
  config.py      the experiment's physical inputs
  parameters.py  derived constants and unit conversions
  scenarios.py   the named runs of the command line
  runners.py     what each kind of run does
  cli.py         the `twisted-bilayer` command
examples/        runnable examples
tests/           pytest; CPU only, two threads
scripts/         validation_report.py (writes docs/validation.md), readme_figures.py
docs/            validation.md and the README figures
check.py         lint + tests in one command
```

## Tests

```bash
python check.py                 # ruff + pytest (183 tests), ~5 min on 2 cores
python check.py --all           # + the quick physics-gate suite as one slow test, ~1 min more
pytest tests/test_kinetic.py    # one file
```

The tests run on the CPU in float64.

## Running on a GPU

Install a CUDA build of PyTorch before the package. The engine uses the GPU when one is visible; set
`TWISTED_BILAYER_FORCE_CPU=1` or pass `--cpu` to stay on the CPU. Start with `twisted-bilayer run gpu_check`:
it runs the quick scan on both devices and compares every number (10⁻⁶ on energies, 10⁻⁴ on superfluid
fractions) before it times a production-size cell. Supercell ground states run in complex128. Large grids
can relax in complex64 and polish in complex128.

## Limitations

- Mean field only. Quantum depletion and finite temperature are not included, beyond the Bogoliubov modes.
- One s-wave scattering length for all three channels (SU(2)-symmetric interaction), as in the paper.
- Commensurate twist angles only (tan(θ/2) = n/m). The measured angle 5.21(5)° is represented by 1/22.
- The quasi-2D reduction is checked against 3D on small cells. Full 3D production runs need a GPU.
- The CPU in float64 is the reference. The GPU path runs the same code and is checked against the CPU by
  `gpu_check`.

## How this was built

The physics, the model choices and the validation design are the author's. The implementation was
AI-assisted, written under the author's direction and reviewed by him. What the author stands behind is
the validation: the physics gates, the test suite and the pinned reference numbers make any unexplained
change in a result fail the build.

## Citing

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff), and the experiment it models.

## Licence and author

MIT licence; see [`LICENSE`](LICENSE).

Václav Svoboda, Mathematical Physics MSc, Wolfgang Pauli Institute, Vienna.
