"""Run records (twisted_bilayer.records) and the panel's on-disk layout: every run keeps its numbers."""

import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from twisted_bilayer import panel
from twisted_bilayer import records as REC


@dataclass
class _Spec:
    a: float = 1.0
    b: tuple = (1, 2)


def test_to_jsonable_converts_arrays_dataclasses_and_nonfinite():
    obj = dict(
        spec=_Spec(),
        x=np.arange(3.0),
        t=torch.tensor([1.0, 2.0]),
        c=np.array([3 + 4j]),
        scalar=np.float64(2.5),
        big=np.zeros((300, 300)),
        bad=float("nan"),
        keys={1.5: "v"},
        other=object,
    )
    out = REC.to_jsonable(obj)
    json.dumps(out)  # serialisable
    assert out["spec"] == dict(a=1.0, b=[1, 2])
    assert out["x"] == [0.0, 1.0, 2.0] and out["t"] == [1.0, 2.0] and out["c"] == [5.0]
    assert out["scalar"] == 2.5 and out["bad"] == "nan" and out["keys"] == {"1.5": "v"}
    assert out["big"]["array_shape"] == [300, 300] and out["big"]["max"] == 0.0


def test_run_dirs_are_unique_and_history_appends(tmp_path):
    p1, s1 = REC.run_dir(tmp_path, "check", "2026-01-01_000000")
    p2, s2 = REC.run_dir(tmp_path, "check", "2026-01-01_000000")
    assert p1 != p2 and os.path.isdir(p1) and os.path.isdir(p2) and s2.endswith("_2")
    REC.append_history(tmp_path, "check", dict(run=s1, verdict="PASS"))
    REC.append_history(tmp_path, "check", dict(run=s2, verdict="FAIL", extra="ignored"))
    with open(tmp_path / "check" / "history.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["verdict"] for r in rows] == ["PASS", "FAIL"]
    REC.mark_latest(tmp_path, "check", s2)
    assert (tmp_path / "check" / "latest.txt").read_text().strip() == s2


def test_junit_to_csv(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text(
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_a" name="test_ok" time="0.5"/>'
        '<testcase classname="tests.test_a" name="test_bad" time="1.0"><failure message="assert 1 == 2">tb</failure>'
        "</testcase>"
        '<testcase classname="tests.test_b" name="test_skip" time="0"><skipped message="no GPU"/></testcase>'
        "</testsuite></testsuites>"
    )
    counts = REC.junit_to_csv(str(xml), str(tmp_path / "t.csv"))
    assert counts == dict(passed=1, failed=1, error=0, skipped=1)
    with open(tmp_path / "t.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["outcome"] == "failed" and rows[1]["message"] == "assert 1 == 2"
    assert rows[0]["test"] == "tests.test_a::test_ok" and float(rows[0]["seconds"]) == 0.5


def test_tee_copies_output(tmp_path, capsys):
    with REC.Tee(str(tmp_path / "o.txt")):
        print("hello")
    assert "hello" in capsys.readouterr().out
    assert (tmp_path / "o.txt").read_text() == "hello\n"


def _step_ok(ctx):
    print("measuring")
    REC.write_json(os.path.join(ctx["dir"], "data.json"), dict(value=np.array([1.0, math.pi])))
    checks = []
    panel._check(checks, "value", True, "pi within tolerance")
    return panel._outcome(checks, dict(value=math.pi))


def _step_crash(ctx):
    raise RuntimeError("boom")


def test_panel_keeps_every_run(tmp_path, monkeypatch):
    steps = [panel.Step("good", "a passing step", _step_ok), panel.Step("bad", "a crashing step", _step_crash)]
    monkeypatch.setitem(panel.PANELS, "test", steps)
    monkeypatch.setattr(panel, "_provenance", lambda: dict(git_commit="abc1234", git_dirty=False))
    assert panel.run_panel("test", out_root=str(tmp_path)) == 1
    assert panel.run_panel("test", out_root=str(tmp_path), only=["good"]) == 0
    base = tmp_path / "panel_test"
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    assert len(runs) == 2
    first = runs[0]
    summary = json.loads((first / "summary.json").read_text())
    assert summary["verdict"] == "FAIL" and summary["state"] == "finished"
    assert [s["step"] for s in summary["steps"]] == ["good", "bad"]
    good = json.loads((first / "good" / "step.json").read_text())
    assert good["ok"] is True and good["checks"][0]["label"] == "value" and "data.json" in good["files"]
    assert json.loads((first / "good" / "data.json").read_text())["value"][1] == math.pi
    assert "measuring" in (first / "good" / "output.txt").read_text()
    assert "RuntimeError: boom" in (first / "bad" / "output.txt").read_text()  # the traceback is kept
    assert "measuring" in (first / "log.txt").read_text()
    assert "FAIL bad" in (first / "report.txt").read_text() or "FAIL" in (first / "report.txt").read_text()
    assert (base / "latest.txt").read_text().strip() == runs[1].name
    with open(base / "history.csv") as f:
        hist = list(csv.DictReader(f))
    assert [h["verdict"] for h in hist] == ["FAIL", "PASS"] and hist[0]["failed_steps"] == "bad"
    assert hist[0]["git_commit"] == "abc1234"


def test_history_header_grows_with_a_new_column(tmp_path):
    """A row with a column the history file lacks used to lose it (extrasaction="ignore"); now the header
    is extended, the old rows keep their values with an empty cell, and nothing is dropped."""
    REC.append_history(tmp_path, "check", dict(run="a", verdict="PASS"))
    REC.append_history(tmp_path, "check", dict(run="b", verdict="FAIL", git_dirty="True"))
    REC.append_history(tmp_path, "check", dict(run="c", verdict="PASS"))
    with open(tmp_path / "check" / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["run"], r["verdict"], r["git_dirty"]) for r in rows] == [
        ("a", "PASS", ""),
        ("b", "FAIL", "True"),
        ("c", "PASS", ""),
    ]
    assert sorted(os.listdir(tmp_path / "check")) == ["history.csv"]
