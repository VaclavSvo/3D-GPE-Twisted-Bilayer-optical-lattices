"""The command line: argument parsing and the --128 precision override."""

from twisted_bilayer import cli
from twisted_bilayer.runners import KINDS, PRECISION_KEYS, double_precision
from twisted_bilayer.scenarios import SCENARIOS


def test_double_precision_sets_every_precision_key():
    for name, row in SCENARIOS.items():
        new = double_precision(row)
        for k in PRECISION_KEYS.get(row["kind"], ()):
            assert new[k] == "complex128", (name, k)
        assert row is not new and {k: v for k, v in new.items() if k not in PRECISION_KEYS.get(row["kind"], ())} == {
            k: v for k, v in row.items() if k not in PRECISION_KEYS.get(row["kind"], ())
        }
    assert set(PRECISION_KEYS) <= set(KINDS)


def test_run_accepts_128(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "twisted_bilayer.runners.run_scenario", lambda name, row, out_root, double: seen.update(d=double)
    )
    assert cli.main(["run", "toy1d", "--128"]) == 0
    assert seen["d"] is True
    assert cli.main(["run", "toy1d"]) == 0
    assert seen["d"] is False


def test_panels_are_consistent(capsys):
    from twisted_bilayer import panel

    quick = [s.name for s in panel.QUICK]
    broad = [s.name for s in panel.BROAD]
    assert quick == broad[: len(quick)]  # broad is quick plus more
    assert len(set(broad)) == len(broad)
    assert cli.main(["panel", "broad", "--list"]) == 0
    assert "experiment_3d" in capsys.readouterr().out


def test_panel_references_have_tolerances():
    from twisted_bilayer.gates import panel_references as R

    for model, values in R.QUICK_3D.items():
        assert set(values) <= set(R.QUICK_3D_TOL), model
    for model, values in R.EXPERIMENT_3D.items():
        assert set(values) <= set(R.EXPERIMENT_3D_TOL[model]), model
