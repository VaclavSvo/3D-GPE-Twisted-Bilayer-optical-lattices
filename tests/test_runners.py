"""runners: every scenario row has a runner, and the trapped runner writes the containment and checks of
its cloud into summary.json and summary.txt."""

import json
import math

import pytest

from twisted_bilayer import runners
from twisted_bilayer.scenarios import SCENARIOS


def test_every_scenario_kind_has_a_runner():
    missing = {name: row.get("kind") for name, row in SCENARIOS.items() if row.get("kind") not in runners.KINDS}
    assert not missing, missing


def test_tight_binding_variants_and_the_shipped_row():
    """Variant D is C plus the interband shift of the site energies only (same bonds, same interlayer pairs);
    the shipped row keeps the variants A, B and C."""
    import numpy as np

    from twisted_bilayer.engine import lattice_model as lm

    assert SCENARIOS["tight_binding"]["variants"] == "ABC"
    assert set(runners.MODEL_C_VARIANTS) == set("ABCD")
    c = lm.build_model(12.0, 0.5, 4, 1, **runners.MODEL_C_VARIANTS["C"][1])
    d = lm.build_model(12.0, 0.5, 4, 1, **runners.MODEL_C_VARIANTS["D"][1])
    np.testing.assert_array_equal(c.bonds.rows, d.bonds.rows)
    np.testing.assert_array_equal(c.bonds.amp, d.bonds.amp)
    np.testing.assert_array_equal(c.V_pairs[2], d.V_pairs[2])
    shift = np.asarray(d.onsite) - np.asarray(c.onsite)
    assert np.abs(shift).max() > 1e-4 and shift.max() <= 1e-12  # second-order shifts lower the sites


def test_trapped_runner_records_containment_and_checks(tmp_path):
    verbose, out_root = runners.VERBOSE, runners._OUT_ROOT
    runners.VERBOSE = False
    try:
        runners.run_scenario("tq", dict(SCENARIOS["trapped_quick"]), out_root=str(tmp_path))
    finally:
        runners.VERBOSE, runners._OUT_ROOT = verbose, out_root
    out = tmp_path / "tq"
    s = json.loads((out / "summary.json").read_text())
    c, k = s["containment"], s["checks"]
    assert c["fits"] and c["ratio"] < 0.88 and c["box_cells"] == s["spec"]["box_cells"]
    assert k["contained"] and k["converged"] and k["ok"] and k["warnings"] == []
    assert s["ground"]["stop_reason"] == "tol"
    text = (out / "summary.txt").read_text()
    assert "WARNING" not in text and "checks: contained True" in text


def _tiny_rabi(tmp_path, name="rq"):
    row = dict(SCENARIOS["rabi_quick"], points_per_a=6.0, t_total=2.0, n_frames=4)
    verbose, out_root = runners.VERBOSE, runners._OUT_ROOT
    runners.VERBOSE = False
    try:
        runners.run_scenario(name, row, out_root=str(tmp_path))
    finally:
        runners.VERBOSE, runners._OUT_ROOT = verbose, out_root
    return json.loads((tmp_path / name / "run_metadata.json").read_text())


def test_physics_failures_end_the_run_with_exit_code_1(tmp_path, monkeypatch):
    """run_rabi with failing conservation gates, and run_gpu_check's comparison with a mismatch, raise
    SystemExit (`twisted-bilayer run` returned 0 after printing the failure). run_rabi writes its files first."""
    import numpy as np

    from twisted_bilayer.gates import core

    def failing(history, **kw):
        return [core.GateResult("norm", False, "forced failure")]

    monkeypatch.setattr(core, "conservation_gates", failing)
    with pytest.raises(SystemExit, match="conservation gates failed"):
        _tiny_rabi(tmp_path)
    assert (tmp_path / "rq" / "rabi.csv").exists() and (tmp_path / "rq" / "run_metadata.json").exists()
    cols = {k: np.linspace(0.0, 1.0, 3) for k in runners.GPU_KEYS}
    ok, lines, table = runners.compare_gpu_cpu(cols, {k: v.copy() for k, v in cols.items()})
    assert ok and len(lines) == len(runners.GPU_KEYS)
    k0 = runners.GPU_KEYS[0]
    assert not runners.compare_gpu_cpu(cols, dict(cols, **{k0: cols[k0] + 1e-3}))[0]  # beyond 1e-6
    assert not runners.compare_gpu_cpu(cols, {k: v for k, v in cols.items() if k != k0})[0]  # a missing column
    assert not runners.compare_gpu_cpu(cols, dict(cols, **{k0: cols[k0] * np.nan}))[0]  # NaN


def test_rabi_quick_uses_the_dressed_coupling_and_gate_Ds_multiplier():
    """rabi_quick had Omega_R = 0.243 literally; every other row uses J_PERP_ER[0.243] = 0.24277 E_R (0.1 %
    apart), and gate D times choose_dt with this row's order-4 multiplier."""
    from twisted_bilayer.scenarios import J_PERP_ER

    row = SCENARIOS["rabi_quick"]
    assert row["Omega_R"] == J_PERP_ER[0.243] == pytest.approx(0.24277, abs=1e-5)
    near = {n: r["Omega_R"] for n, r in SCENARIOS.items() if isinstance(r.get("Omega_R"), float)}
    assert all(v == J_PERP_ER[0.243] for v in near.values() if abs(v - 0.243) < 0.003), near
    assert row["order4_dt_multiplier"] == 20.0 and row["splitstep_order"] == 4


def test_rabi_gates_use_the_suites_tolerances_and_label_the_frequencies(tmp_path):
    """run_rabi gates norm and energy at 1e-9 and 1e-7 (the gate suite's values for a unitary split step; it
    was 1e-2 on both) and writes the frequencies with their units and in Hz."""
    meta = _tiny_rabi(tmp_path)
    gates = {g["name"]: g for g in meta["gates"]}
    assert gates["norm"]["tol"] == 1e-9 and gates["energy"]["tol"] == 1e-7
    assert all(g["passed"] for g in meta["gates"])
    hz = 1.0 / runners.lattice_units().lattice_time_to_s(1.0)
    assert meta["f_peak_hz"] == pytest.approx(meta["f_peak"] * hz) and meta["f_bare_hz"] == pytest.approx(
        meta["f_bare"] * hz
    )
    assert "not Hz" in meta["f_units"] and meta["Omega_eff_ER"] == pytest.approx(meta["f_peak"] * math.pi)
