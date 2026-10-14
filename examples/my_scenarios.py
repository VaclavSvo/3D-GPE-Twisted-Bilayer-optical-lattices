"""Your own scenario rows, run with the command line:

    twisted-bilayer list --scenarios examples/my_scenarios.py
    twisted-bilayer run scan_v6_quick --scenarios examples/my_scenarios.py

Each row is a dict with a `kind` and that runner's parameters (see twisted_bilayer.scenarios). Start from
the defaults and override what you need.
"""

from twisted_bilayer.scenarios import SCAN_DEFAULTS

SCENARIOS = {
    # a coupling scan at V0 = 6 E_R on the small m = 4 cell, three couplings (~15 s on a laptop CPU)
    "scan_v6_quick": dict(
        SCAN_DEFAULTS,
        V0=6.0,
        m=4,
        n=1,
        points_per_a=12.0,
        filling=1.0,
        Omega_R=(0.0, 0.24, 0.5),
        dtype_relax="complex128",
    ),
}
