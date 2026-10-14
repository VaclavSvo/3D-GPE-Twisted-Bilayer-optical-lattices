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
        Omega_R=0.243,
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
    # compare_m: single-particle tight binding against the continuum on the m = 4 cell (0: skip)
    "tight_binding": dict(
        kind="tight_binding",
        V0=8.0,
        m=TWIST_M,
        n=TWIST_N,
        filling=1.0,
        g2d=None,
        theta=0.3,
        Omega_R=(0.0, 0.1, 0.24, 0.5, 1.0),
        variants="ABC",
        compare_m=4,
    ),
    # ── the experiment's geometry (theta = 5.2 deg) ──
    # coupling scan at V0 = 4 E_R on the sqrt(485) supercell, 12 points per a (~6 min on 2 CPU cores)
    "coupling_scan_v4": dict(
        SCAN_DEFAULTS, V0=4.0, filling=1.0, points_per_a=12.0, Omega_R=(0.0, 0.12, 0.24, 0.5, 1.0)
    ),
    # a trapped cloud of N = 2000 atoms in a 20 Hz trap, 3 x 3 supercells at 8 points per a (~6 min on 2 CPU cores)
    "trapped": dict(TRAPPED_DEFAULTS, cell_points=176),
    # ── GPU ──
    # the quick scan on the GPU and in a CPU subprocess, every number compared; then a production timing
    "gpu_check": dict(kind="gpu_check", time_production=True),
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
