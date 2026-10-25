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


def test_uncontained_row_ends_the_walk():
    """A trapped row whose cloud is not contained (checks.contained False) counts as not converged."""
    vals = ((24, 0.5 + 1e-7), (32, 0.5 + 1e-7), (48, 0.5))
    rows = [dict(r=r, values=dict(a=np.array([v])), converged=True, checks=dict(contained=True)) for r, v in vals]
    assert CV.needed(rows, ("a",), "r", dict(a=1e-6)) == dict(a=24)
    rows[1]["checks"]["contained"] = False
    assert CV.needed(rows, ("a",), "r", dict(a=1e-6)) == dict(a=None) and CV.unconverged(rows, "r") == [32]


def test_trapped_rows_share_one_box_and_carry_checks():
    """The trapped scan records each row's containment, checks and stop reason (also in the CSV rows), and
    reports the one box they share (quick cloud, cells 16 and 20)."""
    from twisted_bilayer.engine import trapped as TR

    spec = TR.TrappedSpec(N_atoms=60.0, nu_r_hz=200.0, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0))
    conv = CV.trapped_convergence(spec, (16, 20))
    rows = conv["rows"]
    assert {r["box_cells"] for r in rows} == {conv["box_cells"]} == {spec.box_cells}
    for r in rows:
        assert r["checks"]["contained"] and r["checks"]["ok"] and r["stop_reason"] == "tol"
        assert r["containment"]["fits"]
    assert f"box {spec.box_cells}x{spec.box_cells}," in CV.report(None, conv)
    _, tr_rows = CV.table_rows(None, conv)
    assert tr_rows[0]["contained"] is True and tr_rows[0]["box_cells"] == spec.box_cells


def test_trapped_rows_are_pinned_to_the_first_box(monkeypatch):
    """The coarsest row settles the box under the spec's policy; every finer row runs on that box, with
    containment "refuse" ("warn" stays "warn"). trapped_ground_state is replaced by a stub that enlarges the
    first box (as "enlarge" would) and records the specs it is given."""
    from twisted_bilayer.engine import trapped as TR

    real = TR.trapped_ground_state
    seen, solved = [], {}

    def stub(s, verbose=False):
        seen.append((s.cell_points, s.box_cells, s.containment))
        grown = s if seen[1:] else TR.replace(s, box_cells=s.box_cells + 2)  # the first call "enlarges"
        key = (grown.cell_points, grown.box_cells)
        if key not in solved:  # the second policy reuses the solves of the first
            solved[key] = real(TR.replace(grown, containment="warn"), verbose=verbose)
        return solved[key]

    monkeypatch.setattr(TR, "trapped_ground_state", stub)
    base = dict(N_atoms=60.0, nu_r_hz=200.0, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0))
    for policy, later in (("enlarge", "refuse"), ("warn", "warn")):
        seen.clear()
        conv = CV.trapped_convergence(TR.TrappedSpec(**base, containment=policy), (16, 20))
        assert seen == [(16, 3, policy), (20, 5, later)] and {r["box_cells"] for r in conv["rows"]} == {5}


def test_supercell_scan_runs_each_grid_once(monkeypatch):
    """Two nominal resolutions that round to the same FFT-friendly N (10 and 10.1 points per a on the m = 4
    cell: N = 42) ran twice, giving an identical pair of rows; the second is now skipped and recorded."""
    from twisted_bilayer.engine import sweep
    from twisted_bilayer.engine.lattice import moire_geometry

    geo = moire_geometry(4, 1)
    assert geo.grid_size(10.0) == geo.grid_size(10.1) != geo.grid_size(12.0)
    seen = []

    class Res:
        def __init__(self, spec):
            self.N_grid, self.geometry = geo.grid_size(spec.points_per_a), geo
            self.columns = {k: np.zeros(1) for k in CV.SUPERCELL_KEYS} | dict(residual_0=np.zeros(1))
            self.columns["converged"] = np.ones(1, bool)

    def fake_scan(spec, verbose=False):
        seen.append(spec.points_per_a)
        return Res(spec)

    monkeypatch.setattr(sweep, "coupling_scan", fake_scan)
    conv = CV.supercell_convergence((8.0,), (0.0,), (12, 10.1, 10), m=4)
    assert seen == [10.0, 12.0] and [r["N"] for r in conv["rows"][8.0]] == [42, 50]
    assert conv["skipped_ppa"] == [dict(ppa=10.1, N=42, same_as_ppa=10.0)]
