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
