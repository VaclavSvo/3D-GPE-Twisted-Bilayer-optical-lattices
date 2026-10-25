"""Trapped clouds (engine.trapped).

(i)   LDA: the inverse of mu(nbar) through the scan points, 0 below E0, linear above; solve_mu
      returns the chemical potential that holds N atoms;
(ii)  lda_cloud: a uniform envelope reproduces the tiled, phase-aligned cell exactly; the box grid
      continues the cell grid (odd number of tiles); even tiles are refused;
(iii) the pipeline on the m = 4 cell (seconds): scan, LDA start, polish to tol, N and N2/N
      conserved, the polished state is stationary and close to the LDA start, trapped.summary
      returns the in-situ contrasts, a different noise seed gives the same ground state;
(iv)  pure trap (V0 = 0, g = 0, J only): the polished state is the 2D HO ground state of the lower
      dressed level, E/N = hbar omega - J;
(v)   the central-cell mean (nbar_centre_cell) of a density given as a function of the position is the
      same in every box size: the boundary rows at |x| = L/4 carry half weight on both sides, so the
      result no longer depends on how the grid coordinates round there.
The N = 2000 cloud (anchors in twisted_bilayer.gates.references) is a run (`twisted-bilayer run
trapped`, ~10 min CPU) gated by `twisted-bilayer gates --only E --trapped`.
"""

import numpy as np
import pytest
import torch

from twisted_bilayer.engine import trapped as TR
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import relax
from twisted_bilayer.engine.lattice import harmonic_trap
from twisted_bilayer.engine.physics import BilayerGPE


def test_lda_inverse_and_solve_mu():
    lda = TR.LDA(1.0, [0.5, 1.0, 2.0], [1.1, 1.2, 1.4])
    assert np.allclose(lda(np.array([1.1, 1.2, 1.4])), [0.5, 1.0, 2.0], atol=1e-12)
    assert lda(np.array([0.9]))[0] == 0.0 and lda(np.array([1.0]))[0] == 0.0
    assert lda(np.array([1.5]))[0] == pytest.approx(2.0 + (2.0 - 1.0) / (1.4 - 1.2) * 0.1)  # linear above
    VT = np.linspace(0.0, 0.5, 101) ** 2
    mu = lda.solve_mu(VT, 0.1, 5.0)
    assert lda(mu - VT).sum() * 0.1 == pytest.approx(5.0, rel=1e-9)
    with pytest.raises(ValueError):
        TR.LDA(1.0, [0.5, 1.0], [1.2, 1.1])


def test_solve_mu_beyond_the_initial_bracket():
    """solve_mu doubles its bracket until it holds N: a flat potential with N far above what E0 + 10
    holds (it used to return E0 + 10 silently); inside [E0, E0 + 10] the result is the plain bisection's."""
    lda = TR.LDA(1.0, [0.5, 1.0, 2.0], [1.1, 1.2, 1.4])
    VT = np.zeros(100)
    N = 1e6
    mu = lda.solve_mu(VT, 1.0, N)
    assert mu > 1.0 + 10.0 and lda(mu - VT).sum() == pytest.approx(N, rel=1e-9)
    VT = np.linspace(0.0, 0.5, 101) ** 2
    lo, hi = 1.0, 11.0  # the bisection as it was before the bracket could grow
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lda(mid - VT).sum() * 0.1 < 5.0 else (lo, mid)
    assert lda.solve_mu(VT, 0.1, 5.0) == 0.5 * (lo + hi)


def test_lda_in_the_plane_closed_form():
    """atoms_in_plane against the closed form for a linear f(mu) = c (mu - E0) (scan points on a line):
    N = (2 pi / (m w^2)) c (mu - E0)^2 / 2, R_TF = sqrt(2 (mu - E0) / m) / w; and against a box sum on a
    grid that holds the whole cloud."""
    E0, c = 1.0, 3.0
    lda = TR.LDA(E0, [0.3, 0.6, 1.2], [E0 + 0.1, E0 + 0.2, E0 + 0.4])
    w, mu = 0.05, E0 + 0.7  # above the last scan point: the linear extrapolation
    m = TR.units.M_LATTICE
    assert lda.atoms_in_plane(mu, w) == pytest.approx(2 * np.pi / (m * w * w) * c * 0.7**2 / 2, rel=1e-12)
    N = 500.0
    mu_p = lda.solve_mu_plane(N, w)
    assert lda.atoms_in_plane(mu_p, w) == pytest.approx(N, rel=1e-10)
    R = np.sqrt(2 * (mu_p - E0) / m) / w
    x = np.linspace(-1.5 * R, 1.5 * R, 1201)
    X, Y = np.meshgrid(x, x, indexing="ij")
    VT = 0.5 * m * w * w * (X**2 + Y**2)
    dA = (x[1] - x[0]) ** 2
    assert lda.solve_mu(VT, dA, N) == pytest.approx(mu_p, rel=1e-4)  # same cloud: the box holds it


def test_containment_policies():
    """resolve_box on a cloud wider than its box (the quick cell at a weak trap): refuse (default)
    raises and names the box that holds it; enlarge runs it on that box at the same N and nu_r; warn
    returns it unchanged with a warning. The quick cloud itself fits. On the quick cell at 16 points (4 per a):
    the test is about the box logic, and the enlarged solves (two boxes) stay at ~15 s."""
    Q = {**QUICK, "cell_points": 16}
    spec = TR.TrappedSpec(**Q)
    scan = TR.supercell_scan(spec)
    ok = TR.containment(spec, scan)
    assert ok["fits"] and ok["ratio"] < TR.CONTAIN_REFUSE and ok["box_needed"] <= spec.box_cells
    wide = TR.TrappedSpec(**{**Q, "nu_r_hz": 40.0})
    info = TR.containment(wide, scan)
    assert not info["fits"] and info["box_needed"] > wide.box_cells and info["box_needed"] % 2 == 1
    assert info["R_TF"] <= TR.CONTAIN_TARGET * TR.box_half_width(
        TR.TrappedSpec(**{**Q, "box_cells": info["box_needed"]})
    )
    with pytest.raises(TR.UncontainedCloud, match=f"box_cells >= {info['box_needed']}"):
        TR.trapped_ground_state(wide, scan=scan)
    big, cont = TR.resolve_box(TR.TrappedSpec(**{**Q, "nu_r_hz": 40.0, "containment": "enlarge"}), scan)
    assert big.box_cells == info["box_needed"] and big.N_atoms == wide.N_atoms and big.nu_r_hz == wide.nu_r_hz
    assert cont["fits"] and cont["action"].startswith("enlarged")
    # the quick cloud is small (R_TF ~ 2 oscillator lengths): the Thomas-Fermi box is not enough, and "enlarge"
    # grows the box again on the solved cloud's edge density until it is contained
    run = TR.trapped_ground_state(TR.TrappedSpec(**{**Q, "nu_r_hz": 40.0, "containment": "enlarge"}), scan=scan)
    b = run["spec"].box_cells
    assert b >= big.box_cells and run["grid"].N[0] == b * Q["cell_points"]
    s = TR.summary(run, wide)  # the summary describes the box that ran, not the requested one
    assert s["spec"]["box_cells"] == b and s["checks"]["contained"]
    assert b == big.box_cells or "edge check" in s["containment"]["action"]
    with pytest.warns(RuntimeWarning, match="too wide for its"):
        same, _ = TR.resolve_box(TR.TrappedSpec(**{**Q, "nu_r_hz": 40.0, "containment": "warn"}), scan)
    assert same.box_cells == wide.box_cells
    with pytest.raises(ValueError):
        TR.resolve_box(TR.TrappedSpec(**{**Q, "containment": "maybe"}), scan)


def test_checks():
    c = TR.checks(1e-5, True, 1e-11)
    assert c["ok"] and c["contained"] and c["converged"] and c["warnings"] == []
    c = TR.checks(0.3, False, 1e-3)
    assert not c["ok"] and not c["contained"] and not c["converged"] and len(c["warnings"]) == 2
    assert TR.checks(TR.CONTAIN_EDGE, True)["contained"] is False  # the edge criterion is strict
    # a complex64 solve at its rounding floor is not a failure; above the floor, or in complex128, it is
    f = TR.checks(1e-5, False, 6e-5, "torch.complex64", "stall")
    assert f["ok"] and f["at_precision_floor"] and not f["converged"] and f["warnings"] == []
    assert not TR.checks(1e-5, False, 6e-5, "complex128", "stall")["ok"]
    assert not TR.checks(1e-5, False, 3e-4, "complex64", "stall")["ok"]
    assert not TR.checks(1e-5, False, 6e-5, "complex64", "max_steps")["ok"]  # still falling when it was cut


def test_lda_cloud_uniform_envelope_and_tiling():
    rng = np.random.default_rng(1)
    c = rng.standard_normal((2, 8, 8)) + 1j * rng.standard_normal((2, 8, 8)) + 3.0
    f1 = float(np.mean(np.abs(c) ** 2))
    out = TR.lda_cloud([c], [f1], np.full((24, 24), f1), 3)
    ph = c[0].sum()
    ref = np.tile(c * np.conj(ph) / abs(ph), (1, 3, 3))
    assert np.allclose(out, ref, atol=1e-12)
    assert abs(np.angle(out[0, :8, :8].sum())) < 1e-12  # q = 0 of component 1 real positive
    two = TR.lda_cloud([c, 2 * c], [f1, 4 * f1], np.full((24, 24), 2.5 * f1), 3)
    assert np.allclose(np.abs(two) ** 2, 2.5 * np.abs(np.tile(c, (1, 3, 3))) ** 2, rtol=1e-12)  # w = 1/2
    with pytest.raises(ValueError):
        TR.lda_cloud([c], [f1], np.full((16, 16), f1), 2)


QUICK = dict(m=4, n=1, cell_points=32, box_cells=3, N_atoms=60.0, nu_r_hz=200.0, fillings=(0.5, 1.0, 2.0, 3.0))


@pytest.fixture(scope="module")
def quick_run():
    torch.set_num_threads(2)
    spec = TR.TrappedSpec(**QUICK)
    scan = TR.supercell_scan(spec)
    return spec, scan, TR.trapped_ground_state(spec, scan=scan)


def test_quick_pipeline_converges_and_conserves(quick_run):
    spec, scan, run = quick_run
    assert np.all(np.diff(scan["mus"]) > 0) and scan["E0"] < scan["mus"][0]
    eng, psi, r = run["engine"], run["psi"], run["result"]
    assert bool(r.converged[0]) and float(r.residual[0]) < spec.tol
    assert float(eng.norm(psi)[0]) == pytest.approx(spec.N_atoms, rel=1e-12)
    cn = eng.component_norms(psi)[0]
    assert float(cn[1] / cn.sum()) == pytest.approx(0.5, abs=1e-9)
    c = TR.cloud_numbers(eng, psi, run["geometry"].L_box)
    assert c["edge_density_rel"] < 1e-3 and sum(map(sum, c["atoms_per_tile"])) == pytest.approx(60.0, rel=1e-9)
    # the LDA start is already close to the ground state (at N = 2000: central filling 0.4 %, R_rms 1 %)
    assert abs(run["lda"]["E_per_N"] / float(r.energy[0] / spec.N_atoms) - 1) < 0.05
    assert float(r.mu[0]) > scan["E0"]


def test_quick_pipeline_insitu_observables(quick_run):
    """trapped.summary: finite in-situ contrasts (imaging.standard_observables, coherent model)."""
    spec, scan, run = quick_run
    s = TR.summary(run, spec)
    ins = s["insitu"]
    for k in ("C_raw", "C_coh", "C_coh_infocus", "C_psf", "n_AA", "n_AB"):
        assert np.isfinite(ins[k]), k
    assert ins["imaging_model"].startswith("coherent")
    assert s["ground"]["converged"] and s["cloud"]["N"] == pytest.approx(60.0)
    assert s["checks"]["ok"] and s["containment"]["fits"] and s["containment"]["action"] == "none"
    assert abs(ins["C_smooth"]) <= 1.0 and isinstance(ins["raw_ringing"], bool)


def test_quick_pipeline_seed_independent(quick_run):
    spec, scan, run = quick_run
    spec2 = TR.TrappedSpec(**{**QUICK, "seed": 7})
    run2 = TR.trapped_ground_state(spec2, scan=scan)
    e1 = float(run["result"].energy[0])
    e2 = float(run2["result"].energy[0])
    assert abs(e1 / e2 - 1) < 1e-10
    n1 = (run["psi"].abs() ** 2).sum(1)
    n2 = (run2["psi"].abs() ** 2).sum(1)
    assert float((n1 - n2).abs().max() / n1.max()) < 1e-6


def test_trap_only_ho_ground_state():
    """V0 = 0, g = 0: the trapped ground state is the 2D HO ground state of the lower dressed
    level, E/N = hbar omega - J (the trap term of the trapped scenario, alone)."""
    grid = Grid((64, 64), (24.0, 24.0))
    w, J = 0.05, 0.1  # l_HO = 2.0 a, 5 points per l_HO
    eng = BilayerGPE(grid, V=harmonic_trap(grid, (w, w)), g=0.0, Omega_R=J, dtype=torch.complex128)
    x = [t.numpy() for t in grid.x64]
    X, Y = np.meshgrid(*x, indexing="ij")
    amp = np.exp(-(X**2 + Y**2) / 8.0)
    psi0 = torch.as_tensor(np.stack([amp, -amp])[None].astype(np.complex128))
    r = relax(eng, psi0, 1.0, tol=1e-11, dt_schedule=[])
    assert float(r.energy[0]) == pytest.approx(w - J, rel=1e-9)


def test_centre_cell_mean_does_not_depend_on_the_box():
    def density(X, Y):
        return 1.0 + 0.3 * X + 0.05 * X**2 + 0.02 * X * Y + 0.04 * Y**2  # not symmetric, not periodic

    means = []
    for box in (3, 5, 7, 9):
        grid, geo = TR.trap_grid(TR.TrappedSpec(box_cells=box))
        X, Y = np.meshgrid(*[x.numpy() for x in grid.x64[:2]], indexing="ij")
        means.append(TR.centre_cell_mean(density(X, Y), grid, geo.L_box))
        if box == 3:
            assert TR.centre_cell_mean(np.ones_like(X), grid, geo.L_box) == pytest.approx(1.0, abs=1e-15)
    assert np.ptp(means) < 1e-12, means  # 88 / 89 boundary rows before the fix: 1e-3 here
    # the odd terms cancel on the symmetric cell; for x^2 the trapezoid mean over [-h, h] with spacing dx
    # (end points at half weight) is exactly h^2/3 + dx^2/6
    h, dx = geo.L_box / 4, grid.dx[0]
    assert means[0] == pytest.approx(1.0 + 0.09 * (h * h / 3 + dx * dx / 6), abs=1e-12)
