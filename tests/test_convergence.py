"""In-plane resolution (engine.convergence): the bookkeeping, and spectral convergence on a small cell."""

import numpy as np
import pytest

from twisted_bilayer.engine import convergence as CV


def test_needed_walks_down_from_the_finest():
    rows = [dict(r=r, values=dict(a=np.array([v]))) for r, v in ((4, 1.0), (6, 0.5 + 1e-3), (8, 0.5 + 1e-7), (12, 0.5))]
    assert CV.needed(rows, ("a",), "r", dict(a=1e-6)) == dict(a=8)
    assert CV.needed(rows, ("a",), "r", dict(a=1e-2)) == dict(a=6)
    assert CV.needed(rows, ("a",), "r", dict(a=1e-9)) == dict(a=None)


def test_needed_skips_unconverged_rows():
    """An unconverged row that happens to land near the finest value ends the walk (it does not count as
    converged); an unconverged finest row gives no needed resolution; rows without the flag count."""
    vals = ((4, 0.5 + 1e-7), (6, 0.5 + 1e-3), (8, 0.5 + 1e-7), (12, 0.5))
    rows = [dict(r=r, values=dict(a=np.array([v])), converged=True) for r, v in vals]
    assert CV.needed(rows, ("a",), "r", dict(a=1e-6)) == dict(a=8)
    rows[2]["converged"] = False  # the 8-point row stopped above tol but sits within 1e-7 of the finest
    assert CV.needed(rows, ("a",), "r", dict(a=1e-6)) == dict(a=None)
    assert CV.unconverged(rows, "r") == [8]
    assert (
        CV._needed_text(rows, CV.needed(rows, ("a",), "r", dict(a=1e-6)), "a", "r")
        == "blocked by an unconverged row at 8"
    )
    rows[2]["converged"] = True
    rows[-1]["converged"] = False
    assert CV.needed(rows, ("a",), "r", dict(a=1.0)) == dict(a=None)
    old = [{k: v for k, v in r.items() if k != "converged"} for r in rows]
    assert CV.needed(old, ("a",), "r", dict(a=1e-6)) == dict(a=8)


def test_supercell_converges_spectrally():
    """m = 4 cell at V0 = 8: the error against 12 points per a drops by orders of magnitude from 6 to 8."""
    conv = CV.supercell_convergence((8.0,), (0.0, 0.24), (6, 8, 12), m=4, tol=1e-10)
    rows = conv["rows"][8.0]
    d = CV.differences(rows, CV.SUPERCELL_KEYS)
    for k in ("mu", "E_per_N", "fs_xx_energy"):
        assert d[1][k] < 1e-2 * d[0][k], k
    assert d[1]["fs_xx_energy"] < 1e-5
    assert "needed" in CV.report(conv, None)
    rec = CV.to_record(conv, None)
    assert rec["supercell"]["by_V0"]["8"]["needed_ppa"] == CV.needed(rows, CV.SUPERCELL_KEYS, "ppa")
    assert len(rec["supercell"]["by_V0"]["8"]["rows"]) == 3
    assert all(r["converged"] for r in rows) and rec["supercell"]["by_V0"]["8"]["unconverged_ppa"] == []
    sc_rows, tr_rows = CV.table_rows(conv, None)
    assert len(sc_rows) == 3 * 2 and tr_rows == []
    assert sc_rows[-1]["mu"] == pytest.approx(float(rows[-1]["values"]["mu"][1]))


@pytest.mark.slow
def test_trapped_converges():
    from twisted_bilayer.engine import trapped as TR

    spec = TR.TrappedSpec(N_atoms=60.0, nu_r_hz=200.0, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0), tol=1e-9)
    conv = CV.trapped_convergence(spec, (24, 32, 48))
    d = CV.differences(conv["rows"], CV.TRAPPED_KEYS)
    assert d[1]["C_raw"] < d[0]["C_raw"] and d[1]["C_raw"] < 1e-5
