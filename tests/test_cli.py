"""The command line: argument parsing and the --128 precision override."""

import pytest

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


def test_panel_unknown_step_is_an_error_and_writes_nothing(tmp_path, capsys):
    """`panel quick --only nosuchstep` used to run zero steps, print PASS (0 passed) and append a PASS row to
    history.csv; now it is a usage error (exit code 2) before anything is written, and no step, no check and
    no run is ever a pass."""
    from twisted_bilayer import panel

    assert cli.main(["panel", "quick", "--only", "nosuchstep", "--out", str(tmp_path)]) == 2
    assert "unknown step(s) nosuchstep" in capsys.readouterr().err
    assert cli.main(["panel", "quick", "--skip", "lint,nosuch", "--out", str(tmp_path)]) == 2
    everything = ",".join(s.name for s in panel.QUICK)
    assert cli.main(["panel", "quick", "--skip", everything, "--out", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="unknown step"):
        panel.run_panel("quick", out_root=str(tmp_path), only=["nosuchstep"])
    assert list(tmp_path.iterdir()) == []
    assert panel._outcome([]).ok is False
    assert panel._verdict([])[0] != "PASS"
    assert [s.name for s in panel.select_steps("quick", only=["lint"])] == ["lint"]


def test_panel_nz_checks_never_pass_on_nothing():
    """`_nz_checks` skips observables whose 3D - NPSE gap is below 1e-6; when every gap is that small the step
    used to return no checks, which `_outcome` read as a pass. Now it is one failing check that says why."""
    from twisted_bilayer import panel
    from twisted_bilayer.engine import trapped3d as T3

    def conv(gap):
        fin = {k: 1.0 + gap for k in T3.NZ_KEYS}
        return dict(
            rows=[dict(Nz=16, observables=dict(fin)), dict(Nz=24, observables=dict(fin))],
            npse={k: 1.0 for k in T3.NZ_KEYS},
        )

    tiny = panel._nz_checks(conv(1e-8), 0.02, "t")
    assert len(tiny) == 1 and tiny[0][1] is False and "no observable" in tiny[0][2]
    assert panel._outcome(tiny).ok is False
    real = panel._nz_checks(conv(1e-2), 0.02, "t")
    assert len(real) == len(T3.NZ_KEYS) and panel._outcome(real).ok is True
