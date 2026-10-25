"""Tests of the physics-gate driver (twisted_bilayer.gates.suite).

(i)   unit conversions: the driver's E_R <-> code conversions are those of twisted_bilayer.parameters, the
      engine's E_R anchors are the code-unit anchors divided by E_R_CODE, the engine's units module carries
      the two named exceptions (a_s = 100 a0, the 86.909180 u mass) at their recorded offsets;
(ii)  export_state on a real (small) engine state: grid order, densities, norm, the E_R scalars;
(iii) the smoke subset of run_all (A1, A3, A6, B8, B10) on the real engine passes;
(iv)  [slow] the whole quick suite passes, and the one expected-FAIL demonstration (kappa at
      nbar = 5) is recorded with its pinned value and not counted.
Run from the repository root: python -m pytest -q tests/test_gates.py [-m "not slow"].
"""

import math

import numpy as np
import pytest

from twisted_bilayer import parameters as P
from twisted_bilayer.gates import references as R
from twisted_bilayer.gates import suite as EA


# ------------------------------------------------------- (i) unit conversions
def test_conversions_come_from_parameters():
    assert EA.ER == P.E_R_CODE == math.pi**2 / 2
    assert EA.er(1.19802) == pytest.approx(0.242770, abs=2e-6)  # J_perp production value
    assert EA.t_er(1.0) == pytest.approx(P.E_R_CODE)  # 1 code time = pi^2/2 hbar/E_R
    assert EA.res_tol(1e-9) == pytest.approx(1e-9 / P.E_R_CODE)
    assert EA.G2D_ER == P.G2D_ER and EA.WZ_ER == P.HBAR_OMEGA_Z_ER


def test_engine_anchors_are_code_anchors_over_E_R_CODE():
    v_code, tol_code, _ = R.anchor("dressed_band_bottom_J0.2428_code")
    v_er, tol_er, _ = R.anchor("dressed_band_bottom_J0.2428_ER")
    assert abs(v_code / P.E_R_CODE / v_er - 1) < tol_er  # 14.17705565 / (pi^2/2) = 2.872872
    v_b, tol_b, _ = R.anchor("mu_J0.243_n1_NPSE_lam1_ER")  # parameters' a_s = 5.29 nm
    v_a, tol_a, _ = R.anchor("mu_J0.243_n1_NPSE_lam1_engine_native_ER")  # the engine's own a_s = 100 a0
    assert 0 < v_a - v_b < 1e-5  # 7e-6 E_R apart


def test_engine_units_exceptions():
    ex = EA.units.EXCEPTIONS
    assert set(ex) == {"RB87_SCATTERING_LENGTH_M", "RB87_MASS_U"}
    assert EA.units.lattice_units().g2d / P.G2D_ER - 1 == pytest.approx(3.350e-4, abs=2e-7)
    assert EA.units.E_R_CODE == P.E_R_CODE and EA.units.HBAR == P.HBAR_SI


# --------------------------------------------------------- (ii) export_state round trip
def test_export_state_round_trip(tmp_path):
    grid = EA.Grid((32, 24), (4.0, 3.0))  # dx = 0.125, non-square box
    eng = EA.make_engine(grid, EA.lat.square_lattice(grid, 4.0, 0.0), J=0.1, g=0.01)
    X, Y = EA.mesh(grid)  # 'ij' order: [ix, iy]
    a1 = 1.0 + 0.3 * np.cos(2 * np.pi * X / 4.0)  # varies along x only
    a2 = (1.0 + 0.2 * np.cos(2 * np.pi * Y / 3.0)) * np.exp(0.4j)  # varies along y only
    psi = EA.field_from(grid, a1, a2)
    path = tmp_path / "state.npz"
    EA.obs.export_state(eng, psi, str(path), Omega_R=0.1, V0=4.0)
    with np.load(path) as d:
        assert d["psi1"].shape == (32, 24)  # grid order [ix, iy]
        assert np.allclose(np.abs(d["psi1"]) ** 2, np.abs(a1) ** 2, rtol=1e-6)
        assert np.allclose(np.abs(d["psi2"]) ** 2, np.abs(a2) ** 2, rtol=1e-6)
        assert np.allclose(d["dx"], [0.125, 0.125]) and list(d["N"]) == [32, 24]
        assert float(d["norm"]) == pytest.approx(float(eng.norm(psi)[0]), rel=1e-12)
        assert float(d["mu"]) == pytest.approx(float(eng.energy_terms(psi)["mu"][0]))
        assert float(d["E_R_in_code_units"]) == P.E_R_CODE and float(d["Omega_R"]) == 0.1
        assert str(d["labels"]) == "paper"
        assert np.allclose(d["n"], np.abs(a1) ** 2 + np.abs(a2) ** 2, rtol=1e-6)  # the total density
    # metadata never overwrites a record (the coupling scan used to write the geometry index n over it)
    with pytest.raises(ValueError, match="overwrite"):
        EA.obs.export_state(eng, psi, str(path), n=1)


# ------------------------------------------------------ (iii) engine smoke subset
SMOKE = {"A1", "A3", "A6", "B8", "B10"}


def test_run_all_smoke_subset_passes():
    suite = EA.run_all(quick=True, only=SMOKE, verbose=False, threads=2)
    failed = [r for r in suite.results if not r.passed]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
    names = " ".join(r.name for r in suite.results)
    for sid in ("A1 ", "A3 ", "A6 ", "B8 ", "B10 "):
        assert sid in names
    kinds = {r.kind for r in suite.results}
    assert {"analytic", "identity", "conservation", "convergence"} <= kinds
    assert suite.meta["device"].lower().startswith("cpu")
    assert '"passed": true' in suite.to_json()


# ------------------------------------------------------------- (iv) full quick run
@pytest.mark.slow
def test_run_all_quick_full_suite(tmp_path):
    suite = EA.run_all(out_dir=tmp_path, quick=True, verbose=False, threads=2)
    assert (tmp_path / "gates.json").exists()
    failed = [r for r in suite.results if not r.passed]
    assert not failed, "\n".join(f"{r.name}: {r.detail}" for r in failed)
    ids = " ".join(r.name.split(" ")[0] for r in suite.results)
    for sid in EA.ALL_IDS:
        if sid != "C13":  # C13 gates carry the C13 prefix
            assert sid in ids.split(), sid
    assert any(r.name.startswith("C13 ") for r in suite.results)
    demo = suite.meta["preflight_demo_expected_FAIL"]  # kappa at nbar = 5, not counted
    assert demo and not demo["passed"]
    assert abs(suite.meta["values"]["D"]["kappa_nbar5"] - 1.31) < 0.05
    assert len(suite.results) >= 60


# ------------------------------------------------- (v) the gate primitives cannot pass on nothing or on NaN
def test_drift_gates_fail_on_a_non_finite_frame():
    """A run that turns NaN or inf after frame 0 fails every drift gate (Python's max keeps its first operand on
    a NaN comparison, so the drift used to be that of the finite frames only)."""
    from twisted_bilayer.gates import core as H

    nan, inf = float("nan"), float("inf")
    for bad in (nan, inf, -inf):
        assert not H.rel_drift_gate("n", [1.0, 1.0, bad], 1e-6).passed
        assert not H.abs_drift_gate("n", [0.0, 0.0, bad], 1.0, 1e-6).passed
    assert "frame 2" in H.rel_drift_gate("n", [1.0, 1.0, nan], 1e-6).detail
    hist = dict(t=[0, 1, 2], norm=[1, 1, nan], E_total=[2, 2, nan], N2_over_N=[0, 0, nan], highk=[1e-7, 1e-7, nan])
    gates = {g.name: g.passed for g in H.conservation_gates(hist, J_perp=0.0)}
    assert gates == {"norm": False, "energy": False, "N2_over_N (J_perp = 0)": False, "high-k weight": False}
    mu = H.imaginary_time_gates(dict(mu=[1.0] * 10 + [nan] + [1.0] * 9))[0]
    assert mu.name == "mu plateau" and not mu.passed
    # finite histories are judged as before
    assert all(g.passed for g in H.conservation_gates(dict(t=[0, 1], norm=[1, 1], E_total=[2, 2], highk=[1, 1])))


def test_unknown_only_id_is_an_error_and_no_gates_is_not_a_pass(capsys):
    from twisted_bilayer.gates import core as H

    with pytest.raises(ValueError, match="Z9.*valid ids: A1"):
        EA.run_all(only={"A1", "Z9"}, verbose=False)
    with pytest.raises(ValueError, match="no scenario id"):
        EA.run_all(only=set(), verbose=False)
    with pytest.raises(SystemExit) as e:  # the command line: a usage error, exit code 2
        EA.main(["--only", "Z9", "--quiet"])
    assert e.value.code == 2 and "unknown scenario id(s) Z9" in capsys.readouterr().err
    empty = H.GateSuite("engine", "nothing")
    assert not empty.passed and "NO GATES RUN" in empty.report() and '"passed": false' in empty.to_json()


def test_E_z0_anchor_is_parameters_omega_z_over_2():
    v, tol, _ = R.anchor("E_z0_code")
    assert abs(v - P.OMEGA_Z / 2) / v < tol  # 0.6707839384 was 1.07e-9 off at tol 1e-9


def test_suite_module_device_defaults_to_the_imported_device(monkeypatch):
    """`python -m twisted_bilayer.gates.suite` without --device runs on the device the engine was imported on
    (it used to default to cpu and raise on a GPU machine); `twisted-bilayer gates` forces the CPU first."""
    seen = {}

    def fake_run_all(out, **kw):
        seen.update(kw)
        s = EA.GateSuite("engine", "fake")
        s.add(EA.GateResult("x", True, "ok"))
        s.meta["device"] = "fake"
        return s

    monkeypatch.setattr(EA, "run_all", fake_run_all)
    assert EA.main(["--quiet"]) == 0
    assert seen["device"] == ("cuda" if EA.backend.HAS_GPU else "cpu")
    assert EA.main(["--quiet", "--device", "cpu"]) == 0 and seen["device"] == "cpu"


# ------------------------------------------- (vi) checks that were gates which could not fail (17 Oct audit)
def test_one_component_runs_have_no_N2_gate():
    """A2 and A3 are one-component runs whose N2/N is written as 0.0 every frame, so a gate on it could not
    fail; B8 (two components at J = 0) keeps its N2/N gate."""
    suite = EA.run_all(quick=True, only={"A2", "A3", "B8"}, verbose=False, threads=2)
    assert suite.passed
    n2 = [r.name.split(" ")[0] for r in suite.results if "N2_over_N" in r.name]
    assert n2 == ["B8"]


def test_choose_dt_values_obey_its_caps():
    """What gate D used to gate, as a test of choose_dt: the kinetic phase per step of order 2 and of order 4
    with the multiplier 20 (outer step and largest Yoshida sub-step) below 2.6 rad, the same phase in E_R and
    code units, and a grid that resolves twice the lattice wavenumber."""
    c = EA.twisted_cell_case(ppa=8)
    eng, r = EA.Context(True, False).ground(176, EA.er(c["J"]), 1.0, "fixed")  # the production cell, 8 pts/a
    v = EA.choose_dt_values(eng, r.psi)
    for o in (2, 4):
        assert v[f"order{o}_phase"] < 2.6, v
        assert abs(v[f"order{o}_phase"] / v[f"order{o}_phase_code"] - 1) < 1e-12, v
    assert v["order4_substep_phase"] < 2.6 and v["order4_dt_ER"] > v["order2_dt_ER"]
    assert v["k_nyquist"] > 2.0 * v["k_lattice"]


def test_band_bottom_literals_agree():
    """C11 compared E0 with the anchor, with parameters.E0_BAND_BOTTOM (the same literal) and with the E_R
    anchor (the same number at 30x the tolerance); the literals are checked here once."""
    for name, J in (("dressed_band_bottom_J0.2428_code", 1.19802), ("dressed_band_bottom_J0.1214_code", 0.599)):
        assert P.E0_BAND_BOTTOM[J] == R.anchor(name)[0]
    assert R.anchor("dressed_band_bottom_J0.2428_ER")[1] > R.anchor("dressed_band_bottom_J0.2428_code")[1]
