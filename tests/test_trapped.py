"""Trapped clouds (engine.trapped).

(i)   LDA: the inverse of mu(nbar) through the scan points, 0 below E0, linear above; solve_mu
      returns the chemical potential that holds N atoms;
(ii)  lda_cloud: a uniform envelope reproduces the tiled, phase-aligned cell exactly; the box grid
      continues the cell grid (odd number of tiles); even tiles are refused;
(iii) the pipeline on the m = 4 cell (seconds): scan, LDA start, polish to tol, N and N2/N
      conserved, the polished state is stationary and close to the LDA start, trapped.summary
      returns the in-situ contrasts, a different noise seed gives the same ground state;
(iv)  pure trap (V0 = 0, g = 0, J only): the polished state is the 2D HO ground state of the lower
      dressed level, E/N = hbar omega - J.
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
