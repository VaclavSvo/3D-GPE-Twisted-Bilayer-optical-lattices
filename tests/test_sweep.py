"""The coupling scan (engine.sweep) on the small m = 4 cell (seconds on a CPU).

The twist-free superfluid fraction fs_xx = 2 f_energy - f_current: both routes carry the finite-twist
term, with slopes b (energy) and 2b (current) in theta^2, so the combination is theta-independent to
O(theta^4) while each route alone moves by b theta^2.
"""

import pytest

from twisted_bilayer.engine.sweep import SweepSpec, coupling_scan


def _scan(theta):
    spec = SweepSpec(
        V0=8.0,
        Omega_R=(1.0,),
        filling=1.0,
        m=4,
        n=1,
        points_per_a=8.0,
        theta=theta,
        tol=1e-11,
        dtype_relax="complex128",
        dtype_polish="complex128",
        save_fields=False,
    )
    return coupling_scan(spec, verbose=False).columns


def test_twist_free_superfluid_fraction():
    a, b = _scan(0.4), _scan(0.2)
    for k in range(1):
        e1, c1, f1 = a["fs_xx_energy"][k], a["fs_xx_current"][k], a["fs_xx"][k]
        e2, c2, f2 = b["fs_xx_energy"][k], b["fs_xx_current"][k], b["fs_xx"][k]
        assert f1 == pytest.approx(2 * e1 - c1, abs=1e-15)
        # each route moves with theta^2, the current route twice as fast as the energy route ...
        assert (c1 - c2) / (e1 - e2) == pytest.approx(2.0, rel=0.05)
        # ... and the combination does not move (to O(theta^4))
        assert abs(f1 - f2) < 0.05 * abs(e1 - e2)
        assert a["fs_yy"][k] == pytest.approx(f1, abs=1e-6)  # C4 symmetry
