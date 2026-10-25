"""The full-3D trapped cloud against the quasi-2D models (engine.trapped3d): sizing, the 3D start, and the
physics of a tiny cloud on the m = 4 cell."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from twisted_bilayer.engine import trapped as TR
from twisted_bilayer.engine import trapped3d as T3
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.npse import NPSE2D

TINY = TR.TrappedSpec(
    V0=4.0, N_atoms=60.0, nu_r_hz=200.0, cell_points=24, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0), tol=1e-9
)


def test_scaled_cloud_keeps_the_peak_density():
    s5 = T3.scaled(TR.TrappedSpec(), 5)
    assert s5.box_cells == 5
    assert s5.nu_r_hz == pytest.approx(20.0 * 3 / 5)
    assert s5.N_atoms == pytest.approx(2000.0 * 25 / 9)
    assert T3.scaled(TR.TrappedSpec(), 3) == TR.TrappedSpec()  # box_ref: the `trapped` cloud itself
    with pytest.raises(ValueError):
        T3.scaled(TR.TrappedSpec(), 4)


def test_plan_takes_the_largest_odd_box_then_the_largest_Nz():
    cost = lambda b, z: (b * 100) ** 2 * z  # a pure points count
    budget = (500**2) * 40  # box 5 fits at N_z = 40
    assert T3.plan(cost, budget, 100, Nz_min=24, Nz_max=128) == (5, 40)
    assert T3.plan(cost, budget, 100, Nz_min=24, Nz_max=32) == (5, 32)
    assert T3.plan(cost, budget, 100, box_fixed=3) == (3, 104)
    assert T3.plan(cost, budget, 100, Nz_fixed=24) == (5, 24)
    assert T3.plan(cost, 1.3 * budget, 100, Nz_fixed=24) == (7, 24)
    assert T3.plan(cost, 1.0, 100) is None
    assert T3.plan(lambda b, z: math.inf, 1e30, 100) is None


def test_start_3d_has_the_npse_column_and_width():
    g2 = Grid((16, 16), (4.0, 4.0))
    lz = 0.8
    g3 = Grid((16, 16, 64), (4.0, 4.0, 20.0 * lz))  # +-10 l_z: the widest column has s = 2
    x = g2.coord(0) ** 2 + g2.coord(1) ** 2
    psi2 = torch.cat([torch.exp(-x / 2), 0.5 * torch.exp(-x / 3)], dim=1).to(torch.complex128) * 3.0
    npse = NPSE2D(0.2, 0.27)
    psi3 = T3.start_3d(psi2, npse, g3, lz, torch.complex128)
    col, z2 = T3.column_moments(psi3, g3, chunk=5)
    n2 = (psi2.abs() ** 2)[0].numpy()
    assert np.allclose(col, n2, rtol=1e-9, atol=1e-12)
    s = npse.width(torch.as_tensor(n2.sum(axis=0))).numpy()
    occ = n2.sum(axis=0) > 1e-6
    assert np.allclose(z2[occ], 0.5 * (s * lz)[occ] ** 2, rtol=1e-7)


def test_probe_fits_a_positive_cost_per_point():
    pr = T3.probe(TINY, T3.Vertical(), "complex64", Nz_pair=(4, 8), steps=2)
    assert pr["step_seconds"].b > 0
    assert pr["points"][1] == 2 * pr["points"][0]


@pytest.fixture(scope="module")
def tiny_run():
    res = T3.solve_models(TINY, T3.Vertical(Nz=12), "complex128")
    obs = T3.model_observables(res)
    return res, obs, T3.compare_models(res, obs)


def test_three_models_converge(tiny_run):
    """Each model converges and conserves N; its observables keep the solve's stop reason and precision
    (the 3D / precision / N_z summaries are written from them)."""
    res, obs, _ = tiny_run
    for m in T3.MODELS:
        assert obs[m]["residual"] < 1e-7, m
        assert res["models"][m]["n"].sum() * res["grid2"].dV == pytest.approx(TINY.N_atoms, rel=1e-9)
        assert obs[m]["stop_reason"] == res["models"][m]["stop_reason"] and obs[m]["stop_reason"], m
        assert obs[m]["dtype"] == "torch.complex128", m


def test_fixed_g_bounds_the_3d_energy_from_above(tiny_run):
    """The fixed-g_2D state psi(x, y) phi_0(z) is a 3D trial state: E_3D - hbar omega_z/2 <= E_fixed."""
    _, obs, _ = tiny_run
    assert obs["3d"]["E_per_N"] <= obs["fixed"]["E_per_N"] + 1e-9


def test_npse_is_closer_to_3d_than_fixed_g(tiny_run):
    _, _, cmp = tiny_run
    for k in ("mu", "E_per_N", "C_raw", "n_peak"):
        assert abs(cmp["3d_vs_npse"]["diff"][k]) < abs(cmp["3d_vs_fixed"]["diff"][k]), k
    assert cmp["3d_vs_npse"]["density"]["max_rel"] < 0.5 * cmp["3d_vs_fixed"]["density"]["max_rel"]


def test_pockets_widen_the_cloud_vertically(tiny_run):
    _, _, cmp = tiny_run
    w = cmp["width"]
    assert 1.0 < w["width_3d_mean"] < w["width_3d_max"]
    assert w["width_3d_mean"] == pytest.approx(w["width_npse_mean"], rel=0.02)


def test_reports_build(tiny_run, tmp_path):
    res, obs, cmp = tiny_run
    text = T3.report_models(obs, cmp)
    assert "3d - npse" in text and "vertical width" in text
    obs64 = {m: dict(v, mu=v["mu"] + 1e-7) for m, v in obs.items()}
    table, numbers = T3.precision_table(obs64, obs, cmp)
    assert numbers["table"]["mu"]["3d"]["abs_err"] == pytest.approx(1e-7, rel=1e-3)
    assert "resolves" in table
    T3.save_fields(res, tmp_path / "f.npz")
    assert set(np.load(tmp_path / "f.npz")) >= {"n_3d", "n_npse", "n_fixed", "width_3d"}


def test_gpu_rows_describe_the_trapped_cloud():
    from twisted_bilayer.runners import _trapped3d_row
    from twisted_bilayer.scenarios import SCENARIOS

    for name in ("gpu_3d_vs_2d", "gpu_precision"):
        spec, vert, size = _trapped3d_row(SCENARIOS[name])
        assert (spec.N_atoms, spec.nu_r_hz, spec.cell_points, spec.m) == (2000.0, 20.0, 176, 22), name
        assert size["box_cells"] == "auto" and size["Nz"] == "auto"
        assert vert.z_extent_lz == 5.0


def test_precision_runner_writes_its_files(tmp_path):
    from twisted_bilayer import runners
    from twisted_bilayer.scenarios import SCENARIOS

    row = dict(SCENARIOS["trapped_3d_quick"], kind="trapped_3d_precision", cell_points=24, Nz=8, minutes=1.0)
    verbose, runners.VERBOSE = runners.VERBOSE, False
    try:
        runners.run_scenario("prec", row, out_root=str(tmp_path))
    finally:
        runners.VERBOSE = verbose
    out = tmp_path / "prec"
    for f in ("summary.json", "report.txt", "comparison_complex64.png", "fields/columns_complex128.npz"):
        assert (out / f).is_file(), f
    assert "resolves the 3D - NPSE difference" in (out / "report.txt").read_text()


def test_nz_convergence_is_spectral():
    """The 3D cloud converges fast in the vertical resolution: N_z = 12 and 16 over +-5 l_z agree far
    better than either agrees with the NPSE."""
    conv = T3.nz_convergence(TINY, (12, 16), z_extent_lz=5.0, dtype="complex128", dtype_2d="complex128")
    a, b = (r["observables"] for r in conv["rows"])
    gap = abs(b["mu"] - conv["npse"]["mu"])
    assert abs(a["mu"] - b["mu"]) < 0.05 * gap
    assert abs(a["C_raw"] - b["C_raw"]) < 0.05 * abs(b["C_raw"] - conv["npse"]["C_raw"])
    assert "second-finest" in T3.report_nz(conv)


def test_3d_runs_refuse_instead_of_enlarging():
    """A 3D run's box and N_z are sized before the scan, so containment="enlarge" is turned into "refuse"
    (with a warning): a cloud too wide for the box stops before any solve instead of growing the 3D grid."""
    wide = replace(TINY, nu_r_hz=40.0, containment="enlarge")
    with pytest.warns(RuntimeWarning, match="not available for 3D"):
        with pytest.raises(TR.UncontainedCloud):
            T3.solve_models(wide, T3.Vertical(Nz=8), "complex128")
