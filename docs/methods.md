# Methods

How `twisted-bilayer` works and how to use it: the engine, the scenarios, the checks, GPU runs, the 3D
reference and the quasi-2D models, the repository layout. The [README](../README.md) is the short version;
[`validation.md`](validation.md) lists every physics gate; [`manybody.md`](manybody.md) describes the lattice
models (Bose-Hubbard, Gutzwiller, the bridge to the continuum).

## What the engine does

- **Exact local step.** The split-step Fourier solver exponentiates the local 2 × 2 Hamiltonian exactly, so
  operator splitting is the only time-step error. The Strang (order 2) and Yoshida (order 4) compositions
  converge with fitted slopes 2.00 and 4.00, and the norm is conserved to 10⁻¹³
  ([example 3](../examples/03_time_step_convergence.py)).
- **Batched parameter sweeps.** The coupling Ω_R, the detuning, the interaction and the phase twist can be
  tensors with a batch axis, so a whole scan relaxes in one set of FFT calls on the GPU.
- **Superfluid fraction with bounds.** A phase twist Φ across the box, the same on both components, gives
  f_s from the twist energy and from the twist current. Both carry an O(Φ²) error with slopes in the ratio
  1 : 2, so the combination 2 f_energy − f_current (the `fs_xx` and `fs_yy` columns) is the Φ → 0 limit,
  the quantity a winding-number Monte Carlo measures; at Φ = 0.3 it agrees with an extrapolation in Φ to
  ≤ 10⁻⁷. Leggett's upper and lower bounds, computed from the untwisted density, bracket the result.
- **One engine in 1D, 2D and 3D.** The grid, the split-step propagator, the ground-state solver and the
  energy, current and superfluid-fraction observables work on any number of spatial axes, so the full 3D
  problem with the vertical trap runs on the same code as the 2D reduction and the 1D toy models. The
  imaging observables are 2D (column densities). The tests include 3D real-time runs and the NPSE checked
  against full 3D.
- **Exactly periodic moiré cells.** Commensurate twists tan(θ/2) = n/m give a periodic square box. The
  experiment's angle is n/m = 1/22.
- **More than one model.** The vertical direction can be folded into 2D by the NPSE, a reduction that is
  checked against the full 3D solution. The package also covers trapped clouds from a local-density start,
  Bogoliubov–de Gennes modes, a two-layer Bose–Hubbard mean field, and the time-of-flight and coherent
  absorption-imaging pipeline.
- **Physics gates.** 77 gates check the engine against analytic results, exact identities, and numbers
  reproduced by the author's earlier, separately written quasi-2D and 3D solvers. Those solvers are not
  distributed; their values and tolerances are pinned in
  [`gates/references.py`](../src/twisted_bilayer/gates/references.py).

## How it works

The two components evolve under the Hamiltonian above, from the paper's Supplement (Eqs. 4, 5, 10, 11).
Each component has its own lattice, V₁ and V₂, rotated by ±θ/2. The contact interaction is SU(2)-symmetric
and acts on the total density n. Time stepping is split-step Fourier: the kinetic part is diagonal in
momentum, and the local part is a 2 × 2 matrix whose exponential is exact because it conserves n. A phase
twist ψ(r + L x̂) = e^{iΦ} ψ(r) is carried as a momentum shift of a periodic field. Ground states come from
imaginary time with an adaptive step ladder and a residual criterion ‖Hψ − μψ‖/‖ψ‖, then a preconditioned
polish. The vertical direction is folded in by the NPSE, which replaces g n by the interaction of a Gaussian
of density-dependent width, and is checked against a 3D grid. Imaging applies the coherent transfer
function of the experiment's objective, and time of flight propagates each Bragg order separately after a
mean-field stage. The modules and their equations are documented in their docstrings.

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

### Units

Lengths are in the lattice constant a = 395 nm, energies in the recoil energy E_R = h × 3678.38 Hz, and
ħ = 1, so m = π²/2 and the time unit is ħ/E_R = 43.27 µs. The engine uses the s-wave scattering length
a_s = 100 a₀ for all three channels; the gate anchors use 5.29 nm, and the 3 × 10⁻⁴ difference in g is
listed in `engine.units.EXCEPTIONS` and pinned by the tests. `lattice_units()` converts to SI.

## Parameter reference

Lattice units throughout: lengths in a, energies in E_R, ħ = 1, time in ħ/E_R; frequencies in Hz. Fields have
shape (B, C, *N) (batch member, component, grid axes) with |ψ|² in atoms per a^d. The docstrings carry the full
text; this is the summary.

### `engine.physics.BilayerGPE(grid, V=None, g=0.0, Omega_R=0.0, delta=0.0, kappa=1/π², n_components=2, twist=None, dtype=complex128, interaction=None, allow_attractive=False)`

Each of `g`, `Omega_R`, `delta` is a scalar or a (B,) sequence; batched inputs must all have length B or 1.

| parameter | unit | meaning |
| --- | --- | --- |
| `grid` | a | the periodic `Grid`; its device sets the engine's. Read-only afterwards |
| `V` | E_R | the potential: (*N), (1 or B, 1, *N) shared by both components, or (1 or B, C, *N) per component. Finite |
| `g` | E_R a^d | contact coupling, mean field g n with n = \|ψ₁\|² + \|ψ₂\|² (SU(2) symmetric); in 2D the quasi-2D g_2D. ≥ 0 unless `allow_attractive` |
| `Omega_R` | E_R | interlayer coupling, +Ω_R σ_x; its sign selects the lower dressed branch |
| `delta` | E_R | detuning, +δ/2 on component 1, −δ/2 on component 2 |
| `kappa` | E_R a² | kinetic prefactor, H_kin = −κ∇²; > 0; read-only afterwards |
| `n_components` | — | 1 or 2 (Ω_R and δ act only for 2) |
| `twist` | rad | phase twists θ_i across the box, ψ(r + L_i e_i) = e^{iθ_i} ψ(r); (d,) or (B, d); later `set_twist` |
| `dtype` | — | `torch.complex128` (default) or `torch.complex64` |
| `interaction` | — | a density functional replacing g n (e.g. `engine.npse.NPSE2D`); then pass g = 0. Not batchable |
| `allow_attractive` | — | accept g < 0 (default False: refused as a likely sign error) |

Construction raises `ValueError` for a non-finite input, κ ≤ 0, g < 0 without `allow_attractive`, batched inputs of
different lengths or a non-complex dtype. No physics range is imposed.

### `engine.ground_state.relax(engine, psi0, n_particles, tol=1e-7, dt_schedule=None, ..., polish_steps=20000, ...)`

| parameter | unit | meaning |
| --- | --- | --- |
| `engine` | — | the `BilayerGPE`; its dtype sets the precision |
| `psi0` | atoms^½ a^(−d/2) | the start (B, C, *N), renormalised to `n_particles` |
| `n_particles` | atoms | per member (scalar or (B,)), fixed throughout |
| `tol` | E_R | residual ‖Hψ − μψ‖/‖ψ‖ at which a member counts as converged |
| `dt_schedule` | ħ/E_R | the imaginary-time ladder; None builds it from `dt_start`, `dt_min`; [] skips it |
| `max_steps_per_stage`, `check_every`, `stall_ratio` | steps, steps, — | the ladder's cap per stage, steps between residual checks, the required fall per check |
| `polish_steps` | steps | the polish's budget (0: no polish) |
| `polish_tau`, `polish_beta`, `polish_alpha`, `polish_stall_windows` | —, —, E_R, windows | passed to `precondition_polish` |

Returns a `RelaxResult`: ψ, residual, μ, energy and converged per member, the steps, the stages and `stop_reason`.

### `engine.ground_state.precondition_polish(engine, psi, n_particles, tol=1e-8, max_steps=2000, tau=0.5, beta=0.9, alpha=None, check_every=50, stall_windows=None, info=None)`

Preconditioned heavy-ball descent at fixed N: v ← βv − τP(Hψ − μψ), P = (κk² + α)⁻¹. `tol` (E_R) the residual to
reach; `max_steps` the budget; `tau` and `beta` the step and momentum (dimensionless); `alpha` (E_R) the
preconditioner's shift (default the local energy scale |V| + gn + Ω_R); `check_every` steps per window;
`stall_windows` the span of the stall test (default 4 in complex128, 1 in complex64). A residual that rises
while the energy falls by more than its rounding level is a descent, not an overshoot (complex128). `info`
receives `stop_reason` (`tol`, `stall`, `slow`, `max_steps`, `tau_floor`). Returns (ψ_best, residual_best (B,),
steps). A deep lattice's soft mode can need 10⁵ steps or more: give such a run its budget
(`TrappedSpec.polish_steps`, `Vertical.polish_steps`, `SweepSpec.max_polish_steps`).

### `engine.ground_state.initial_state(engine, n_particles, batch=1, kind="uniform", fractions=(0.5, 0.5), seed=None, width=None, phase_noise=0.0, relative_phase=None)`

`kind`: `"uniform"` (the start on a periodic supercell), `"gaussian"` (`width` in a, default L_min/8; for traps)
or `"thomas_fermi"` (needs V). `fractions`: the share of atoms per component. `relative_phase` (rad): the phase
of component 2 against 1; default per member π for Ω_R > 0 and 0 for Ω_R < 0 (the lower dressed branch; a batch
mixing signs needs `batch = len(Omega_R)`). `phase_noise` (rad) and `seed`: a random phase field on top.

### `engine.trapped.TrappedSpec`

| field | unit | meaning |
| --- | --- | --- |
| `V0`, `Omega_R` | E_R | lattice depth per layer; interlayer coupling |
| `N_atoms` | atoms | both components |
| `nu_r_hz`, `omega_z_hz` | Hz | radial (isotropic) and vertical trap frequencies |
| `box_cells` | supercells | the periodic box per side (odd: an AA point at the origin) |
| `cell_points` | points | per supercell side (176 = 8 per a on the √485 cell) |
| `vertical` | — | `"npse"` or `"fixed"` (g_2D n) |
| `g2d` | E_R a² | the 2D coupling; None derives it from `omega_z_hz` and a_s = 100 a₀ |
| `fillings` | atoms / a² | the supercell scan that builds the LDA start |
| `noise`, `seed` | — | the random perturbation of the start |
| `tol`, `scan_tol` | E_R | residual targets of the trapped solve and of the scan |
| `polish_steps` | steps | the polish's budget (raise it per row for a slow cloud) |
| `m`, `n` | — | the commensurate twist, tan(θ/2) = n/m |
| `dtype` | — | `"complex128"` or `"complex64"` for the trapped solve (the scan is complex128) |
| `containment` | — | `"refuse"`, `"enlarge"` or `"warn"` for a cloud wider than the box |

### `engine.trapped3d.Vertical(Nz=32, z_extent_lz=5.0, polish_steps=20000)` and `solve_models(spec, vert, dtype="complex128", scan=None, dtype_2d=None)`

`Vertical`: `Nz` points over z ∈ [−z_extent_lz l_z, +z_extent_lz l_z] (l_z the vertical oscillator length, in
a) and the 3D polish's budget. `solve_models` computes the NPSE (from the LDA start), fixed-g and 3D (from the
NPSE cloud) ground states of one `TrappedSpec` cloud, the 3D solve in `dtype`, the 2D ones in `dtype_2d`
(default `dtype`); `scan` reuses a supercell scan. μ and E/N of the 3D solve are given without ħω_z/2. Returns
the models (column density in atoms/a², μ, E/N, residual, steps, `stop_reason`, time, peak memory) and the
widths ⟨z²⟩/(l_z²/2).

### `engine.sweep.SweepSpec` and `coupling_scan(spec, verbose=True, device=None)`

| field | unit | meaning |
| --- | --- | --- |
| `V0` | E_R | lattice depth per layer |
| `Omega_R` | E_R | the couplings scanned, one batch member each |
| `filling` | atoms / a² | both components |
| `m`, `n` | — | the twist (22, 1: the experiment's √485 cell) |
| `points_per_a` | points / a | ≥ 8 for ground states, ≥ 16 at V₀ ≥ 12 |
| `delta` | E_R | detuning |
| `g2d`, `omega_z_hz`, `vertical` | E_R a², Hz, — | the quasi-2D interaction (`"npse"` or `"fixed"`) |
| `theta` | rad | the phase twist for the superfluid fraction |
| `tol` | E_R | residual target |
| `dtype_relax`, `dtype_polish` | — | precisions of the relaxation and of the polish |
| `max_polish_steps` | steps | the polish's budget (twisted solves at large V₀ need far more) |
| `blur_sigma` | a | Gaussian blur of the moiré contrast |
| `phase_noise`, `seed` | rad, — | a random start phase for independent-seed checks |
| `save_fields` | — | one field file per Ω_R |

`coupling_scan` returns a `SweepResult`: per Ω_R E/N and μ (E_R), `fs_xx`, `fs_yy` and their energy and current
routes, the Leggett brackets, contrasts, pocket populations, residuals, `converged`; the densities and timings.

### `engine.thin_z.thin_z_check(V0=8.0, Omega_R=0.24, filling=1.0, m=4, n=1, points_per_a=12.0, Nz=32, z_extent_lz=4.0, omega_z_hz=1000.0, delta=0.0, tol=1e-7, with_fs=False, theta=0.3, dtype_relax="complex64", dtype_polish="complex128", npse=True)`

The quasi-2D reductions against a full 3D solve on the homogeneous supercell; parameters as in `SweepSpec`, plus
`Nz`, `z_extent_lz` for the z grid, `with_fs`/`theta` for the superfluid fraction, `npse` to compare the NPSE as
well as fixed g. Returns a `ThinZResult` (μ, E/N, participation ratio, contrast, peak density per model; f_s;
the column densities and the width map).

## Examples

Times are for 2 CPU cores. Figures go to `examples/output/`.

| script | what it shows | time |
| --- | --- | --- |
| [`01_ground_state.py`](../examples/01_ground_state.py) | ground state on the experiment's supercell, with a figure of its density | 15 s |
| [`02_superfluid_fraction.py`](../examples/02_superfluid_fraction.py) | superfluid fraction against the coupling, by twist energy and current, with Leggett's bounds; one batched solve | 10 s |
| [`03_time_step_convergence.py`](../examples/03_time_step_convergence.py) | order-2 and order-4 convergence of the real-time integrator, norm and energy conservation | 15 s |
| [`my_scenarios.py`](../examples/my_scenarios.py) | your own scenario rows for the command line: `twisted-bilayer run scan_v6_quick --scenarios examples/my_scenarios.py` | 15 s |

## Scenarios

`twisted-bilayer run <name>` runs a named row of [`scenarios.py`](../src/twisted_bilayer/scenarios.py), or of
your own file with `--scenarios FILE`. Times are for 2 CPU cores.

| name | what it computes | time |
| --- | --- | --- |
| `coupling_scan_quick` | superfluid fraction, Leggett bounds, μ and pocket formation against the coupling (tan(θ/2) = 1/4 cell) | 15 s |
| `trapped_quick` | a small trapped cloud: local-density start, ground state, contrasts as imaged | 10 s |
| `rabi_quick` | real-time Rabi transfer between the two states inside the lattice | 15 s |
| `toy1d` | the one-dimensional two-lattice model: exact hopping, superfluid fraction by twist | 10 s |
| `thin_z_quick` | the full 3D problem with a thin vertical direction against the quasi-2D reduction | 20 s |
| `trapped_3d_quick` | a small trapped cloud in full 3D against the NPSE and fixed-g quasi-2D models | 20 s |
| `trapped_3d_nz_quick` | convergence of that 3D cloud in the vertical resolution (N_z = 8 … 32) | 45 s |
| `xy_convergence_quick` | in-plane resolution on the small cell: supercell at 6 … 16 points per a, trapped cloud | 1.5 min |
| `bdg` | Bogoliubov–de Gennes modes: sound speed and the spin branch (always on the CPU) | 1.5 min |
| `tight_binding` | the two-layer Bose–Hubbard mean field on the experiment's supercell, three variants (always on the CPU) | 1.5 min |
| `bridge_quick` | the lattice mean field against the continuum GPE on the tan(θ/2) = 1/4 cell, with and without the interband shift | 1 min |
| `phase_diagram_quick` | single-site Gutzwiller phase map (SF / SF-II / I / MI) over V₀ and Ω_R on the tan(θ/2) = 1/4 cell; for speed and plotting | minutes |
| `phase_diagram_paper_quick` | the same for the paper's one-layer model | minutes |
| `phase_diagram_cluster_quick` | the same with interlayer-dimer clusters (cluster Gutzwiller) | ~10 min |
| `coupling_scan_v4` | coupling scan at V₀ = 4 E_R on the experiment's supercell (θ = 5.2°) | 6 min |
| `trapped` | a cloud of N = 2000 atoms in a 20 Hz trap on 3 × 3 supercells, and its contrast as imaged | 6 min |
| `bridge` | the same comparison on the experiment's supercell (12 continuum ground states at 16 points per a) | hours on a CPU, minutes on a GPU |
| `phase_diagram`, `phase_diagram_noshift` | single-site Gutzwiller phase map on the experiment's supercell, 21 × 13 points, with and without the interband shift | ~2 h on one core per map |
| `phase_diagram_paper`, `phase_diagram_paper_mu095` | the paper's model on the experiment's supercell, at μ/U = 1 and 0.95 (the reading that reproduces the paper's Fig. S4c) | ~1 h |
| `phase_diagram_cluster` | interlayer-dimer cluster map on the experiment's supercell, 11 × 7 points | ~2 h |
| `gpu_check` | the quick scan on the GPU and on the CPU, every number compared; then a production timing | GPU |
| `gpu_3d_vs_2d` | the trapped cloud in full 3D and in both quasi-2D models, as large as the GPU holds | GPU |
| `gpu_precision` | the same comparison in complex128 and complex64, sized to take about 10 minutes | GPU, 10 min |
| `gpu_3d_nz` | the experiment's cloud in 3D at N_z = 16 … 96 over the same z box, against the finest | GPU, ~10 min |
| `gpu_xy_convergence` | in-plane resolution: supercell (V₀ = 4, 8, 16) at 4 … 24 points per a, trapped cloud at 4 … 16 | GPU, tens of min |

The phase maps also run on their own, with more options (time budget, resume, U correction table):
`python -m twisted_bilayer.manybody.phase_diagram <row> --out <folder> [--device cuda] [--threads 1]
[--max-hours H] [--redo-unconverged] [--force-resume]`, where `<row>` is any entry of
`manybody.phase_diagram.ROWS`. A map is written after every V₀ column and a rerun into the same folder continues
it; a folder made with another row specification is refused.

## Exit codes

`twisted-bilayer run` exits with 0 when the run finished, 1 when a physics check of the run failed (the
conservation gates of `rabi`, the GPU/CPU comparison of `gpu_check`) or on any error (a cloud refused before
the solve because it does not fit its box ends here too), 2 on a usage error (an unknown scenario), and 3 when
a trapped run finished but its cloud is not contained in its box (edge density at or above 10⁻³ of the peak;
`checks.contained` is false in `summary.json`). An exit code 3 run has written all its files: read them as
describing overlapping periodic copies of the cloud, not one cloud, and rerun in a larger box. The gates and
the panels exit 0 only if everything they selected ran and passed, and 2 on a usage error.

## Checks and check panels

```bash
twisted-bilayer list                          # the scenarios
twisted-bilayer run coupling_scan_quick       # ~15 s; files in results/coupling_scan_quick/
twisted-bilayer run rabi_quick --out runs     # files in runs/rabi_quick/
twisted-bilayer gates                         # the quick physics gates, ~45 s
twisted-bilayer gates --full --trapped        # all 77 gates (the table of validation.md), ~10 min on 2 cores
python check.py                               # lint + the tests (run kept in results/check/)
python check.py --all                         # + the slow tests (the quick gates as one test among them)
pytest tests/test_kinetic.py                  # one test file
twisted-bilayer panel quick                   # every quick check with one verdict, ~6 min on a CPU
twisted-bilayer panel broad                   # + the GPU checks on the experiment's cloud, ~30-60 min
```

`check.py` takes about 5 minutes on 2 CPU cores (`--all` several minutes more). The tests run on the CPU in
complex128, with one thread; tests that take longer than about 20 s are marked `slow` and run only with
`--all` (or `pytest -m slow`).


`twisted-bilayer panel quick` is the check to run before every commit; `twisted-bilayer panel broad` after
every larger change. Each step passes or fails, and the panel exits 0 only if all of them ran and passed.

Every run is kept, so a result can be checked later without rerunning it. A run goes to
`results/panel_<name>/<date_time>/` in the checkout (git-ignored, wherever the panel is started from):

| file | what |
| --- | --- |
| `summary.json`, `report.txt`, `log.txt` | the verdict, git commit and versions, every check of every step; everything printed |
| `<step>/step.json`, `<step>/output.txt` | the step's checks with their numbers; what it printed (a crash keeps its traceback) |
| `tests/tests.csv`, `tests/junit.xml` | every test with its outcome and time |
| `gates_*/gates*.json` | every physics gate |
| `quick_3d/`, `experiment_3d/` | `data.json` (all observables of the three models, both precisions), `report.txt`, column densities (`.npz`), comparison figures |
| `*_nz/`, `*_xy/` | `data.json` (every resolution row, the differences from the finest, the resolution each observable needs), `report.txt`, `nz.csv` or `supercell.csv` and `trapped.csv` |
| `planner/` | the probe, the plan and the largest box at N_z = 16-64 with predicted times |

`results/panel_<name>/latest.txt` names the newest run and `history.csv` has one line per run (verdict,
failed steps, minutes, device, commit). `summary.json` is rewritten after each step, so an interrupted run
keeps what it finished. `python check.py` keeps its runs in the same way in `results/check/`.

| panel | steps | where, how long |
| --- | --- | --- |
| `quick` | lint and format; the test suite; the quick physics gates; on a small trapped cloud: 3D against both quasi-2D models against pinned numbers, vertical resolution, in-plane resolution | CPU or GPU, about 6 minutes on 2 cores |
| `broad` | `quick`, plus: the slow tests; the full physics gates with the trapped cloud (on the GPU); GPU against CPU; on the experiment's cloud: 3D against quasi-2D in complex64 and complex128 against pinned numbers, vertical and in-plane resolution; the 3D size planner | needs a GPU; about 30-60 minutes on a GPU with 12 GiB of memory |

`--list` shows the steps, `--only` and `--skip` select them. The pinned numbers and their tolerances are in
`gates/panel_references.py`; a step's pass criterion is in its docstring (`twisted_bilayer/panel.py`).

`coupling_scan_quick` finds the ground state at five couplings in one batch. For each, it measures the
superfluid fraction, the Leggett bounds and how the atoms gather into the moiré pockets. It writes a
six-panel figure, a table, the full `.csv`, the wave functions, and a `provenance.json` with the package
version, git commit, library versions and device:

```
 Omega_R  Omega_over_J  E_per_N      mu    fs_xx  fs_xx_energy  fs_xx_current  leggett_x_lower  leggett_x_upper
       0             0   4.9885  5.0045  0.28262       0.28253        0.28244          0.26673          0.99882
     0.1        3.2468   4.9272   4.944  0.30072       0.30064        0.30055          0.26609          0.99537
    0.24        7.7924   4.8347  4.8541  0.31188       0.31178        0.31168          0.25726          0.96279
     0.5        16.234   4.6457  4.6729  0.30319       0.30305        0.30292          0.22821          0.85176
       1        32.468   4.2367  4.2855  0.24034       0.24014        0.23994          0.16413          0.59207
```

(A selection of the printed columns.) Energies are in E_R. `Omega_over_J` is Ω_R/J₁, where J₁ = 0.0308 E_R
is the nearest-neighbour hopping of one lattice at V₀ = 8 E_R. `fs_xx` is the superfluid fraction along the
box x axis at Φ → 0 (2 f_energy − f_current); the two routes at Φ = 0.3 are printed beside it.

The quick scenarios use the small tan(θ/2) = 1/4 cell (θ = 28°), so they finish in seconds. Use them to
check that everything works; they do not reproduce the experiment's geometry.

## Running on a GPU

Install a CUDA build of PyTorch before the package (see Installation). The engine uses the GPU when one is
visible. To stay on the CPU, pass `--cpu` to `twisted-bilayer run`, or set the environment variable
`TWISTED_BILAYER_FORCE_CPU=1` (PowerShell: `$env:TWISTED_BILAYER_FORCE_CPU = "1"`). The physics gates run
on the CPU unless you pass `--device cuda`, and `bdg` and `tight_binding` always run on the CPU.

Start with `twisted-bilayer run gpu_check`. It runs the quick scan on both devices and compares every
number (10⁻⁶ on energies, 10⁻⁴ on superfluid fractions) before it times a production-size cell. On a CUDA
GPU, `coupling_scan_quick` reproduces the CPU table to every printed digit.
Supercell ground states run in complex128. Large grids can relax in complex64 and polish in complex128.
Each scenario row names its precision; `twisted-bilayer run NAME --128` overrides it with complex128
throughout (see below for what single precision costs).

### Full 3D against quasi-2D, sized to the GPU

`gpu_3d_vs_2d` solves the trapped cloud three ways on the same (x, y) grid: in full 3D (the vertical trap on
a z grid), with the NPSE, and with the fixed g₂D. It then compares μ, the column density, the local vertical
width and the contrasts as imaged. The cloud of the `trapped` row (N = 2000, 20 Hz on 3 × 3 supercells) is
scaled to the box at fixed peak density (ν_r ∝ 1/box, N ∝ box²), so a larger box holds more moiré cells
with the same local physics. Before the run, a short probe measures the memory and time per step of the 3D
solve on this GPU. The run then takes the largest odd box that fits in 85 % of the free memory at N_z ≥ 16,
and spends the rest on N_z. The same command fills a GPU with 6, 12 or 24 GiB of memory. To pin the size instead, set
`box_cells` and `Nz` in a row of your own.

`gpu_precision` runs the same comparison in complex128 and complex64 and reports, for every observable,
the single-precision error against the 3D − NPSE difference it has to resolve. complex64 stops at a
residual of about 10⁻⁵ E_R: on the experiment's cloud its error is 1–6 × 10⁻⁵ relative, 1–3 % of that
difference (up to 6 % in larger clouds), and the 3D solve is 4–18 times faster in half the memory. The
shipped 3D rows run in complex64, which is enough for scans; pass `--128` for numbers you quote. 2D runs
are cheap (the experiment's cloud takes 0.1 GiB and half a minute) and stay in complex128, as do
superfluid fractions and real-time runs.

`gpu_3d_nz` checks the vertical resolution: the same cloud at several N_z over the same z box, each against
the finest. In complex128 (`--128`) on the experiment's cloud, N_z = 24 over ±5 l_z (dz = 0.42 l_z) agrees
with N_z = 64 to 4 × 10⁻¹⁰ and N_z = 16 to 6 × 10⁻⁶ (relative, in the peak density); a ±6 or ±7 l_z box
changes nothing at 10⁻⁹. The shipped row runs in complex64, whose residual floor (~10⁻⁵ E_R) is far above
these differences: its N_z table shows noise at the 10⁻⁷ level in μ, not the N_z dependence.

**Containment.** The trapped box is periodic, so a cloud wider than the box overlaps its own images and
every observable then describes overlapping clouds. Before allocating the grid, every trapped run solves
the LDA in the infinite plane and compares its Thomas–Fermi radius with the box: a cloud whose radius
exceeds 0.88 of the half-width (above that the experiment's cloud has more than ~10⁻³ of its peak density at
the edge) is refused with the box size that holds it (`containment="enlarge"` in a 2D row takes that box at
the same N and ν_r, `"warn"` runs anyway; 3D rows pin their box). A larger 3D box at fixed peak density
(`gpu_3d_vs_2d`) scales the cloud with the box, so it never contains a cloud that 3 × 3 does not. After the solve each summary records `checks`:
`contained` (edge density below 10⁻³ of the peak), `converged`, and WARNING lines in the report when either
fails. Deep lattices and strong coupling widen the experiment's cloud a lot (R_TF ≈ 58 a in the infinite plane
at V₀ = 12 E_R, Ω_R = 1 E_R, against a 3 × 3 half-width of 33 a: it takes a 7 × 7 box).

`gpu_xy_convergence` does the same in the plane: the homogeneous supercell at V₀ = 4, 8, 16 E_R from 4 to 24
points per a, and the trapped cloud from 4 to 16 points per a (88 to 352 per supercell), each against the
finest, in complex128. It reports, per observable, the coarsest resolution within a tolerance (1e-6 E_R for
μ and E/N, 1e-4 for f_s and the contrasts): 6, 8 and 12 points per a at V₀ = 4, 8 and 16 E_R. Values read
at single grid points (the contrast extrema, the AA density) depend on where the grid samples them, so a
resolution scan should include a grid that does not share points with the others.

**Settings for a converged 3D cloud** (checked one source at a time, in complex128, on the experiment's
cloud; each source changes every observable by ≤ 10⁻⁵ relative):

| setting | value | cost on a GPU with 12 GiB of memory |
| --- | --- | --- |
| in-plane points | 176 per supercell (8 per a) at 4 E_R; 12 per a at 8–12 E_R, 16 per a beyond | |
| vertical | N_z = 24 over ±5 l_z (16 if memory forces it) | |
| tolerance | residual 10⁻⁹ E_R (tighter changes nothing above 10⁻⁸) | |
| box | the cloud's edge density below 10⁻³ of its peak: 5 × 5 supercells for N = 2000 at 20 Hz and 4 E_R. A 3 × 3 box moves absolute values (R_rms by 5 × 10⁻⁴ a) but not the 3D − quasi-2D differences, for this cloud; deeper lattices or stronger coupling need larger boxes (see Containment) | 5 × 5 at N_z 24: 6.8 GiB, 6 min; 7 × 7 at N_z 16: 8.9 GiB, 10 min |

The complex128 3D solve costs about 12 ns per grid point per step and 394 bytes per point; from the NPSE
start it needs about 1600 steps for a 20 Hz trap, and more in weaker traps (5600 at 8.6 Hz). Leave about
1.5 GB of the card free for the CUDA context. Deep lattices relax slowly: check the residual each run
reports.

```bash
twisted-bilayer run gpu_precision    # ~10 min
twisted-bilayer run gpu_3d_nz        # vertical resolution
twisted-bilayer run gpu_xy_convergence   # in-plane resolution
twisted-bilayer run gpu_3d_vs_2d     # the largest cloud this GPU holds
```

## Repository layout

```
src/twisted_bilayer/
  engine/        the solver: physics.py (BilayerGPE), ground_state.py, evolve.py, grid.py, lattice.py,
                 npse.py, observables.py, imaging.py, sweep.py, trapped.py, trapped3d.py, thin_z.py, bdg.py,
                 bands.py, lattice_model.py, units.py, backend.py
  manybody/      the lattice model beyond mean field: parameters.py (Bose-Hubbard parameters, the parameter
                 and sign audits, the occupation-dependent U), paper_model.py (the paper's one-layer model),
                 bridge.py (the lattice mean field against the continuum GPE), gutzwiller.py (single-site
                 Gutzwiller with twisted boundaries, batches and several starts), cluster.py (cluster
                 Gutzwiller), fast_kernels.py (the certified tridiagonal site solver of
                 `solve_batch(eig="fast")`), observables.py (momentum distribution, visibilities, phase
                 labels), phase_diagram.py (phase maps over V0 and Omega_R)
  tof/           time of flight and absorption imaging
  gates/         physics gates: core.py, references.py (every reference number), suite.py
  config.py      the experiment's physical inputs
  parameters.py  derived constants and unit conversions
  scenarios.py   the named runs of the command line
  runners.py     what each kind of run does
  cli.py         the `twisted-bilayer` command
  panel.py       the check panels
  records.py     run records (results/<kind>/<date_time>/, latest.txt, history.csv)
examples/        runnable examples
tests/           pytest; CPU only, two threads
scripts/         readme_figures.py (the README figure), validation_report.py (docs/validation.md from a gate
                 run), bisect_eigh_batch.py and gpu_column_check.py (the batched eigensolver on a GPU)
docs/            methods.md (this page), validation.md, manybody.md, figures/
check.py         lint + tests in one command (the panels: `twisted-bilayer panel quick | broad`)
```

## Limitations

- Mean field only in the continuum. Quantum depletion is included only through the Bogoliubov modes and,
  on the lattice, single-site and cluster Gutzwiller; finite temperature is not included.
- One s-wave scattering length for all three channels (SU(2)-symmetric interaction), as in the paper.
- Commensurate twist angles only (tan(θ/2) = n/m). The measured angle 5.21(5)° is represented by 1/22.
- The shipped 3D scenario (`thin_z_quick`) uses the small tan(θ/2) = 1/4 cell. Full 3D runs on the
  experiment's √485 supercell work with the same engine but are slow on a CPU; use a GPU for them.
- The CPU in complex128 is the reference. The GPU path runs the same code and is checked against the CPU
  by `gpu_check`.

## References

- Z. Meng *et al.*, Atomic Bose–Einstein condensate in twisted-bilayer optical lattices,
  [Nature **615**, 231 (2023)](https://doi.org/10.1038/s41586-023-05695-4). The experiment and the model.
- L. Salasnich, A. Parola and L. Reatto, Effective wave equations for the dynamics of cigar-shaped and
  disk-shaped Bose condensates, [Phys. Rev. A **65**, 043614 (2002)](https://doi.org/10.1103/PhysRevA.65.043614).
  The NPSE.
- H. Yoshida, Construction of higher order symplectic integrators, Phys. Lett. A **150**, 262 (1990). The
  order-4 composition.
- A. J. Leggett, Can a solid be "superfluid"?, [Phys. Rev. Lett. **25**, 1543 (1970)](https://doi.org/10.1103/PhysRevLett.25.1543);
  On the superfluid fraction of an arbitrary many-body system at T = 0, J. Stat. Phys. **93**, 927 (1998).
  The bounds.
- G. Chauveau *et al.*, Superfluid fraction in an interacting spatially modulated Bose–Einstein condensate,
  [Phys. Rev. Lett. **130**, 226003 (2023)](https://doi.org/10.1103/PhysRevLett.130.226003). The bounds in a
  modulated condensate.
