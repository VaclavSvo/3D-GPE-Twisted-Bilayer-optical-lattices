"""Named scenarios: one row per run of `twisted-bilayer run <name>`.

A row is a dict whose `kind` names the runner in `twisted_bilayer.runners`; the other keys are that
runner's parameters. Energies are in E_R and lengths in lattice constants a, unless a key says _hz.

Your own rows live in a separate Python file that defines a dict `SCENARIOS`; pass it with
`twisted-bilayer run <name> --scenarios my_rows.py`. It can build on the defaults here:

    from twisted_bilayer.scenarios import SCAN_DEFAULTS
    SCENARIOS = {"scan_v6": dict(SCAN_DEFAULTS, V0=6.0, filling=1.0, Omega_R=(0.0, 0.24, 0.5))}

Keys of a coupling-scan row (`kind="coupling_scan"`):
    V0            lattice depth;  m, n  the commensurate twist tan(theta/2) = n/m
    Omega_R       J_perp, one value or a tuple (run as one batch)
    filling       mean atoms per a^2, both components together
    points_per_a  grid resolution: >= 8 for supercell ground states, >= 16 for V0 >= 12; twist superfluid
                  fractions need >= 8 (4 aliases a Bloch harmonic)
    theta         phase twist across the box for the superfluid fraction (rad); keep << pi
    tol           ground-state residual ||H psi - mu psi|| / ||psi||
    dtype_relax / dtype_polish   precision of the two relaxation stages
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

from twisted_bilayer.config import J_PERP_CODE, N_ATOMS, NU_TRAP_HZ, NU_Z_HZ, S_SAT, TWIST_M, TWIST_N, V0_ER

J_PERP_ER = {k: v / (math.pi**2 / 2.0) for k, v in J_PERP_CODE.items()}  # 0.12138 / 0.15004 / 0.24277 E_R

# Defaults of a coupling scan on the experiment's supercell (theta = 5.2051 deg, L = sqrt(485) a).
SCAN_DEFAULTS = dict(
    kind="coupling_scan",
    m=TWIST_M,
    n=TWIST_N,
    points_per_a=16.0,
    delta=0.0,
    theta=0.3,
    tol=1e-8,
    omega_z_hz=NU_Z_HZ,
    dtype_relax="complex64",
    dtype_polish="complex128",
)

# Defaults of a trapped cloud: LDA start from a supercell scan, polished on box_cells^2 supercells.
TRAPPED_DEFAULTS = dict(
    kind="trapped",
    V0=V0_ER,
    Omega_R=J_PERP_ER[0.243],
    N_atoms=N_ATOMS,
    nu_r_hz=NU_TRAP_HZ,
    box_cells=3,
    vertical="npse",
    g2d=None,
    omega_z_hz=NU_Z_HZ,
    fillings=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    noise=0.01,
    seed=0,
    tol=2e-10,
    s_sat=S_SAT,
    dtype="complex128",
)

# Defaults of the full-3D comparison (engine.trapped3d): the trapped cloud above, given at box_cells = 3 and
# scaled to the box at fixed peak density; sizes "auto" fill the device (85 % of the free GPU memory).
#
# Precision. 3D solves run in complex64 by default: on the experiment's cloud (V0 = 4 E_R, box 3, N_z = 48)
# every observable of `gpu_precision` agrees with complex128 to < 2 % of the 3D - NPSE difference (mu to
# 1e-7 E_R, contrasts to 4e-6), 16x faster on a consumer GPU, in half the memory. That is the only cloud where
# it was measured: other clouds have smaller 3D - NPSE differences (down to a few 1e-3 a in R_rms), and
# there the complex64 floor (residual ~1e-5 E_R) is not known to be negligible -- run a `trapped_3d_precision`
# row on that cloud, or `--128`. 2D solves are cheap (0.1 GiB and half a minute for the experiment's cloud)
# and stay in complex128, as do superfluid fractions (energy differences ~1e-3 E_R on ~80 E_R) and real-time
# runs. `--128` runs complex128 throughout.
TRAPPED_3D_DEFAULTS = dict(
    {k: v for k, v in TRAPPED_DEFAULTS.items() if k not in ("kind", "box_cells", "vertical", "g2d")},
    cell_points=176,
    tol=1e-9,
    z_extent_lz=5.0,
    dtype="complex64",  # the 3D solve
    dtype_2d="complex128",  # the two quasi-2D solves
    box_cells="auto",
    Nz="auto",
    Nz_min=16,  # gpu_3d_nz: N_z = 16 ... 96 agree to the complex64 floor on the experiment's cloud
    Nz_max=128,
    memory_fraction=0.85,
)

SCENARIOS = {
    # ── quick runs: the small m = 4 cell (a 28 deg twist), seconds to minutes on a laptop CPU ──
    # superfluid fraction (twist energy and current), Leggett bounds, mu and pocket formation vs coupling
    "coupling_scan_quick": dict(
        SCAN_DEFAULTS,
        V0=8.0,
        m=4,
        n=1,
        points_per_a=12.0,
        filling=1.0,
        Omega_R=(0.0, 0.1, 0.24, 0.5, 1.0),
        dtype_relax="complex128",
    ),
    # a small trapped cloud: LDA start, ground state, in-situ contrasts as imaged
    "trapped_quick": dict(
        TRAPPED_DEFAULTS, N_atoms=60.0, nu_r_hz=200.0, cell_points=32, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0)
    ),
    # real-time Rabi transfer in the lattice from an all-|1> ground state (order-4 split step)
    "rabi_quick": dict(
        kind="rabi",
        V0=V0_ER,
        Omega_R=J_PERP_ER[0.243],  # 0.24277 E_R, as every other row
        filling=1.0,
        m=4,
        n=1,
        points_per_a=10.0,
        delta=0.0,
        t_total=40.0,
        n_frames=100,
        splitstep_order=4,
        order4_dt_multiplier=20.0,
        dtype="complex128",
    ),
    # the 1D two-lattice toy model: exact J_1, superfluid fraction by twist, energy = current = Leggett
    "toy1d": dict(kind="toy1d", V0=8.0, g_nbar=0.2, n_sites=12, points_per_a=32, theta=0.1, tol=1e-11),
    # the full 3D problem with a thin vertical direction against the quasi-2D (NPSE) reduction
    "thin_z_quick": dict(
        kind="thin_z",
        V0=8.0,
        Omega_R=0.24,
        filling=1.0,
        m=4,
        n=1,
        points_per_a=10.0,
        Nz=24,
        z_extent_lz=4.0,
        omega_z_hz=NU_Z_HZ,
        tol=1e-7,
        with_fs=True,
    ),
    # Bogoliubov-de Gennes modes: sound speed and the spin branch (contact g2D, ~20 s per coupling)
    "bdg": dict(
        kind="bdg",
        V0=8.0,
        filling=1.0,
        m=4,
        points_per_a=12.0,
        Omega_R=(0.1, 0.24, 0.5),
        modes=8,
        spin_modes=24,
        theta=0.3,
        tol=1e-10,
        vertical="fixed",
    ),
    # the two-layer tight-binding (Bose-Hubbard) mean field, variants A/B/C, on the experiment's supercell;
    # compare_m: single-particle tight binding against the continuum on the m = 4 cell (0: skip).
    # The interaction: g2d if given, else from nu_z_hz (engine.lattice_model.g2d_from_nu_z), else the shipped
    # lattice_model.G2D_DEFAULT (both None, as here)
    "tight_binding": dict(
        kind="tight_binding",
        V0=8.0,
        m=TWIST_M,
        n=TWIST_N,
        filling=1.0,
        g2d=None,
        nu_z_hz=None,
        theta=0.3,
        Omega_R=(0.0, 0.1, 0.24, 0.5, 1.0),
        variants="ABC",
        compare_m=4,
    ),
    # the bridge (manybody.bridge): the lattice mean field (variant C, with and without the interband shift)
    # against the continuum GPE at fixed g2d on the same cell; mu, E/N, f_s, AA/AB, N2/N, PR over V0 and Omega_R.
    # Always complex128 (energy differences of 1e-3 E_R). The m = 4 row takes about a minute on a CPU
    "bridge_quick": dict(
        kind="bridge",
        V0=(8.0, 12.0, 16.0),
        Omega_R=(0.0, 0.24, 0.6, 1.0),
        m=4,
        n=1,
        filling=1.0,
        nu_z_hz=NU_Z_HZ,
        points_per_a=16.0,
        theta=0.3,
        tol=1e-8,
        dtype_relax="complex128",
        dtype_polish="complex128",
        cutoff=2.0,
        frame="layer1",
        interband_n_bands=6,
        tb_tol=1e-11,
    ),
    # ── the experiment's geometry (theta = 5.2 deg) ──
    # the same on the experiment's supercell (12 continuum ground states x 3 twists at 16 points per a: minutes
    # on a GPU, hours on 2 CPU cores)
    "bridge": dict(
        kind="bridge",
        V0=(8.0, 12.0, 16.0),
        Omega_R=(0.0, 0.24, 0.6, 1.0),
        m=TWIST_M,
        n=TWIST_N,
        filling=1.0,
        nu_z_hz=NU_Z_HZ,
        points_per_a=16.0,
        theta=0.3,
        tol=1e-8,
        dtype_relax="complex128",
        dtype_polish="complex128",
        cutoff=2.0,
        frame="layer1",
        interband_n_bands=6,
        tb_tol=1e-11,
    ),
    # phase maps over (V0, Omega_R) with single-site / cluster Gutzwiller (manybody.phase_diagram; the parameters
    # live in its ROWS, named by "base"). Resumable: a rerun continues results/<name>/maps.npz. The m = 4 rows take
    # minutes on a CPU and are for speed and plotting only; the m = 22 rows take hours (one process per map)
    "phase_diagram_quick": dict(kind="phase_diagram", base="phase_diagram_quick"),
    "phase_diagram_paper_quick": dict(kind="phase_diagram", base="phase_diagram_paper_quick"),
    "phase_diagram_cluster_quick": dict(kind="phase_diagram", base="phase_diagram_cluster_quick"),
    "phase_diagram": dict(kind="phase_diagram", base="phase_diagram"),
    "phase_diagram_noshift": dict(kind="phase_diagram", base="phase_diagram_noshift"),
    "phase_diagram_paper": dict(kind="phase_diagram", base="phase_diagram_paper"),
    "phase_diagram_paper_mu095": dict(kind="phase_diagram", base="phase_diagram_paper_mu095"),
    "phase_diagram_cluster": dict(kind="phase_diagram", base="phase_diagram_cluster"),
    # coupling scan at V0 = 4 E_R on the sqrt(485) supercell, 12 points per a (~6 min on 2 CPU cores)
    "coupling_scan_v4": dict(
        SCAN_DEFAULTS, V0=4.0, filling=1.0, points_per_a=12.0, Omega_R=(0.0, 0.12, 0.24, 0.5, 1.0)
    ),
    # a trapped cloud of N = 2000 atoms in a 20 Hz trap, 3 x 3 supercells at 8 points per a (~6 min on 2 CPU cores)
    "trapped": dict(TRAPPED_DEFAULTS, cell_points=176),
    # the trapped cloud in full 3D against the NPSE and fixed-g quasi-2D models on the small m = 4 cell (~20 s)
    "trapped_3d_quick": dict(
        TRAPPED_3D_DEFAULTS,
        kind="trapped_3d",
        N_atoms=60.0,
        nu_r_hz=200.0,
        cell_points=32,
        m=4,
        n=1,
        fillings=(0.5, 1.0, 2.0, 3.0),
        box_cells=3,
        Nz=16,
    ),
    # ── GPU ──
    # the quick scan on the GPU and in a CPU subprocess, every number compared; then a production timing
    "gpu_check": dict(kind="gpu_check", time_production=True),
    # the trapped cloud in full 3D and in the two quasi-2D models, as large as the GPU holds (3D in complex64,
    # 2D in complex128; --128 for complex128 throughout, about half the 3D points)
    "gpu_3d_vs_2d": dict(TRAPPED_3D_DEFAULTS, kind="trapped_3d"),
    # the same comparison in complex128 and complex64, sized to take about `minutes` on the GPU
    "gpu_precision": dict(TRAPPED_3D_DEFAULTS, kind="trapped_3d_precision", minutes=10.0),
    # convergence in the vertical resolution: the 3 x 3 cloud in 3D at N_z = 16 ... 96 over the same z box
    # (dz = 0.62 ... 0.10 l_z), against the finest; N_z that do not fit the GPU are skipped (~10 min, 6 GB)
    "gpu_3d_nz": dict(TRAPPED_3D_DEFAULTS, kind="trapped_3d_nz", box_cells=3, Nz_list=(16, 24, 32, 48, 64, 96)),
    # in-plane resolution: the homogeneous supercell (V0 = 4, 8, 16; f_s is the most sensitive quantity) at
    # 4 ... 24 points per a, and the trapped cloud at 88 ... 352 points per supercell (4 ... 16 per a), each
    # against the finest, complex128 (GPU: tens of minutes)
    "gpu_xy_convergence": dict(
        kind="xy_convergence",
        supercell=dict(V0_list=(4.0, 8.0, 16.0), Omega_R=(0.0, 0.24, 1.0), ppa_list=(4, 6, 8, 12, 16, 24), m=22),
        trapped=dict(
            {k: v for k, v in TRAPPED_DEFAULTS.items() if k not in ("kind", "s_sat", "dtype")},
            cell_points_list=(88, 132, 176, 264, 352),
            tol=1e-9,
        ),
    ),
    # the same on the small m = 4 cell (~2 min on a CPU)
    "xy_convergence_quick": dict(
        kind="xy_convergence",
        supercell=dict(V0_list=(4.0, 8.0), Omega_R=(0.0, 0.24, 1.0), ppa_list=(6, 8, 12, 16), m=4),
        trapped=dict(
            V0=V0_ER,
            Omega_R=J_PERP_ER[0.243],
            N_atoms=60.0,
            nu_r_hz=200.0,
            m=4,
            n=1,
            fillings=(0.5, 1.0, 2.0, 3.0),
            box_cells=3,
            tol=1e-9,
            cell_points_list=(24, 32, 48, 64),
        ),
    ),
    # the same on the small m = 4 cloud (~1 min on a CPU)
    "trapped_3d_nz_quick": dict(
        TRAPPED_3D_DEFAULTS,
        kind="trapped_3d_nz",
        N_atoms=60.0,
        nu_r_hz=200.0,
        cell_points=32,
        m=4,
        n=1,
        fillings=(0.5, 1.0, 2.0, 3.0),
        box_cells=3,
        Nz_list=(8, 12, 16, 24, 32),
    ),
}


def load_scenarios(path: str | None = None) -> dict:
    """The rows above, plus the `SCENARIOS` dict of the Python file at `path` (its rows win on a name clash)."""
    rows = dict(SCENARIOS)
    if path:
        file = pathlib.Path(path).expanduser().resolve()
        if not file.is_file():
            raise FileNotFoundError(f"scenario file {file} not found")
        spec = importlib.util.spec_from_file_location(f"_scenarios_{file.stem}", file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        extra = getattr(mod, "SCENARIOS", None)
        if not isinstance(extra, dict):
            raise ValueError(f"{file} must define a dict SCENARIOS")
        rows.update(extra)
    return rows
