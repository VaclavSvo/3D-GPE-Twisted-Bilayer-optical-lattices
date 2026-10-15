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
