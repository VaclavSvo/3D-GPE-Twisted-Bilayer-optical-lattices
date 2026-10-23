"""Imaging observables (engine.imaging) and the Leggett bounds of a density (engine.observables).

(i)   a pure moire cosine: C_raw, C_psf (the Gaussian MTF), the line contrast, the Fourier weight and
      the pixel image at AA are the analytic values; a radial transfer H scales the contrast by H(q_M);
      the coherent transfer is 1 at q_M in focus, 0 at the lattice, ~0.97 at s = 0.01;
(ii)  standard_observables: a uniform density has no contrast and unit Leggett bounds in every frame;
      an AA-bright (dark) modulation gives C_raw = +eps (-eps) and the signs of every contrast;
(iii) Leggett bounds: a density varying along one box axis gives 1/(<n><1/n>) and 1 (box frame and
      both lattice frames); a density separable in the lattice-1 axes is exact in that frame; the
      q = 0 Bloch state of one lattice gives m/m* = 0.65194 in its frame and [0.43, 1.00] in the box.
Densities are numpy arrays in the engine's grid order [ix, iy] unless a test says otherwise.
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer import parameters as P
from twisted_bilayer.engine import imaging as im
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.observables import leggett_bounds, leggett_bounds_aligned

L = P.L_SUPERCELL


def _coords(N):
    dx = L / N
    x = -(N // 2) * dx + dx * np.arange(N)
    X, Y = np.meshgrid(x, x, indexing="ij")  # [ix, iy]
    return dx, X, Y


def _lattice1_axes(X, Y):
    return X * P.C_ROT - Y * P.S_ROT, Y * P.C_ROT + X * P.S_ROT


def _synthetic(N, eps, V0=4.0):
    """Each component in its own rotated lattice (density ~ exp(-V/E_R)) times the moire modulation
    1 + eps (cos 2 q_M x + cos 2 q_M y)/2: +eps at AA, -eps at AB. Mean density 1 per component."""
    dx, X, Y = _coords(N)
    c, s = P.C_ROT, P.S_ROT
    V1 = V0 * (np.sin(np.pi * (X * c - Y * s)) ** 2 + np.sin(np.pi * (Y * c + X * s)) ** 2)
    V2 = V0 * (np.sin(np.pi * (X * c + Y * s)) ** 2 + np.sin(np.pi * (Y * c - X * s)) ** 2)
    q = 2 * np.pi / L
    mod = 1.0 + eps * 0.5 * (np.cos(2 * q * X) + np.cos(2 * q * Y))
    n1, n2 = np.exp(-V1) * mod, np.exp(-V2) * mod
    return n1 / n1.mean(), n2 / n2.mean(), dx


# ------------------------------------------------------------------- (i) analytic patterns
def test_contrasts_of_a_moire_cosine():
    box = im.Box(176, 176, L / 176)
    x = box.x0 + box.dx * np.arange(176)
    X, Y = np.meshgrid(x, x, indexing="xy")  # [iy, ix]: the module's internal order
    G, e = 4 * np.pi / L, 0.1
    n = 1 + e * (np.cos(G * X) + np.cos(G * Y))
    c = im.moire_contrasts(n, box)
    assert c["C_raw_AAAB"] == pytest.approx(2 * e, abs=1e-10)  # AA 1 + 2e, AB 1 - 2e
    mtf = math.exp(-0.5 * (G * im.PSF_SIGMA) ** 2)
    assert c["C_psf_AAAB"] == pytest.approx(2 * e * mtf, abs=1e-10)
    assert c["C_raw_line_x"] == pytest.approx(e / (1 + e), abs=1e-6)  # AA vs the saddle
    assert im.moire_fourier_weights(n, box)["F_moire_10"] == pytest.approx(e / 2, abs=1e-12)
    xc, yc, img = im.image(n, box, sigma=0.0, pixel=im.PIXEL)
    sinc = np.sinc(G * im.PIXEL / (2 * np.pi))
    assert img[np.argmin(np.abs(yc)), np.argmin(np.abs(xc))] == pytest.approx(1 + 2 * e * sinc, abs=1e-10)


def test_transfer_contrasts_scale_with_H():
    box = im.Box(88, 88, L / 88)
    x = box.x0 + box.dx * np.arange(88)
    X, _ = np.meshgrid(x, x, indexing="xy")
    eps = 0.3
    n2 = 1 + eps * np.cos(2 * np.pi * 2 * X / L)
    assert im.transfer_contrasts(n2, box, np.ones_like)["C_AAAB"] == pytest.approx(eps, abs=1e-12)
    c = im.transfer_contrasts(n2, box, lambda k: np.where(k > 0, 0.6, 1.0))
    assert c["C_AAAB"] == pytest.approx(0.6 * eps, abs=1e-12)


def test_coherent_transfer_values():
    qM = 2 * np.pi / P.L_MOIRE
    H0, H1 = im.coherent_transfer(0.0), im.coherent_transfer(0.01)
    assert float(H0(qM)) == 1.0 and float(H0(2 * np.pi)) == 0.0
    assert 0.96 < float(H1(qM)) < 0.985
    assert float(im.pupil_transfer(qM)) == 1.0 and float(im.pupil_transfer(2 * np.pi)) == 0.0


# --------------------------------------------------------------- (ii) standard_observables
def test_uniform_density():
    N = 88
    n = np.full((N, N), 1.3)
    out = im.standard_observables(n, 2 * n, L / N, imaging="pupil")
    for k in ("C_raw", "C_psf", "C_coh", "C_img_central", "C_coh_img", "C_cell_n", "C_raw_total"):
        assert abs(out[k]) < 1e-12, k
    assert out["n_AA"] == pytest.approx(3.9, abs=1e-12) and out["n_AB"] == pytest.approx(3.9, abs=1e-12)
    for k in ("fs_lower_x", "fs_upper_x", "fs_lower_y_lat1", "fs_upper_x_lat2"):
        assert out[k] == pytest.approx(1.0, abs=1e-12), k
    assert out["N2_over_N"] == pytest.approx(2 / 3)
    assert all(isinstance(v, (float, str)) for v in out.values())
    off_cell = im.standard_observables(np.ones((64, 64)), np.ones((64, 64)), 0.25, imaging="pupil")
    assert np.isnan(off_cell["fs_lower_x_lat1"])


@pytest.mark.parametrize("eps", [0.2, -0.15])
def test_modulation_sign(eps):
    n1, n2, dx = _synthetic(88, eps)
    out = im.standard_observables(n1, n2, dx, imaging="pupil")
    assert out["C_raw"] == pytest.approx(eps, abs=2e-3)  # the |k| < pi/a envelope is nbar (1 + eps m)
    assert out["C_coh"] == pytest.approx(eps, abs=2e-3)  # in-focus pupil = C_raw to 2e-4
    for k in ("C_psf", "C_cell_n", "C_paper_signed"):
        assert np.sign(out[k]) == np.sign(eps), k
    assert out["F_moire_10"] == pytest.approx(abs(eps) / 4, abs=2e-3)
    assert out["fs_lower_x_lat1"] == pytest.approx(out["fs_lower_x_lat2"], rel=1e-9)  # mirror-symmetric


# ----------------------------------------------------------------- (iii) Leggett bounds
def test_box_axis_modulation_all_frames():
    dx, X, _ = _coords(176)
    n = 1.0 + 0.5 * np.cos(2 * np.pi * X / L)
    lo, up = leggett_bounds(torch.as_tensor(n)[None], Grid((176, 176), (L, L)), 0)
    assert float(lo[0]) == pytest.approx(math.sqrt(0.75), abs=1e-10)
    assert float(up[0]) == pytest.approx(math.sqrt(0.75), abs=1e-10)
    lo, up = leggett_bounds(torch.as_tensor(n)[None], Grid((176, 176), (L, L)), 1)
    assert float(lo[0]) == pytest.approx(1.0, abs=1e-12) and float(up[0]) == pytest.approx(1.0, abs=1e-12)
    for lat in (1, 2):
        lo, up = leggett_bounds_aligned(n, dx, "x", lattice=lat)
        assert lo == pytest.approx(math.sqrt(0.75), abs=1e-10) and up == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("N", [88, 176])
def test_separable_density_is_exact_in_its_frame(N):
    dx, X, Y = _coords(N)
    u, v = _lattice1_axes(X, Y)
    n = (1.0 + 0.6 * np.cos(2 * np.pi * u)) * (1.0 + 0.6 * np.cos(2 * np.pi * v))
    for axis in ("x", "y"):
        lo, up = leggett_bounds_aligned(n, dx, axis, lattice=1)
        assert lo == pytest.approx(0.8, abs=1e-7) and up == pytest.approx(0.8, abs=1e-7)
    lo2, up2 = leggett_bounds_aligned(n, dx, "x", lattice=2)
    assert lo2 <= 0.8 + 1e-9 <= up2 + 2e-9  # every frame brackets the same f_s


def _bloch_q0_density_1d(V0_ER=4.0, nmax=12):
    """|phi(s)|^2 of the 1D q = 0 Bloch state of V0 sin^2(pi s) and its exact m/m* = 1/(<f><1/f>)."""
    V0 = V0_ER * math.pi**2 / 2
    ns = np.arange(-nmax, nmax + 1)
    H = np.diag(0.5 * (2 * np.pi * ns) ** 2 + V0 / 2)
    H += np.diag(np.full(2 * nmax, -V0 / 4), 1) + np.diag(np.full(2 * nmax, -V0 / 4), -1)
    c = np.linalg.eigh(H)[1][:, 0]

    def f(s):
        return np.abs(sum(cn * np.exp(2j * np.pi * n * s) for cn, n in zip(c, ns))) ** 2

    fs = f(np.arange(4096) / 4096)
    return f, 1.0 / (fs.mean() * (1.0 / fs).mean())


@pytest.mark.parametrize("N,tol", [(176, 5e-5), (352, 1e-6)])  # 8 pts/a aliases the 5th harmonic
def test_single_lattice_bloch_state(N, tol):
    f, m_ratio = _bloch_q0_density_1d()
    assert m_ratio == pytest.approx(0.65194, abs=1e-5)
    dx, X, Y = _coords(N)
    u, v = _lattice1_axes(X, Y)
    n = f(u) * f(v)
    lo, up = leggett_bounds_aligned(n, dx, "x", lattice=1)
    assert lo == pytest.approx(m_ratio, abs=tol) and up == pytest.approx(m_ratio, abs=tol)
    blo, bup = leggett_bounds(torch.as_tensor(n)[None], Grid((N, N), (L, L)), 0)
    assert 0.40 < float(blo[0]) < 0.45 and float(bup[0]) > 0.99  # the box frame is loose for a rotated lattice


def test_aligned_bounds_need_the_supercell():
    with pytest.raises(ValueError):
        leggett_bounds_aligned(np.ones((64, 64)), 0.25)
    with pytest.raises(ValueError):
        leggett_bounds_aligned(np.ones((88, 88)), L / 88, "z")


# ------------------------------------------------------------------- ringing of the raw envelope
def _pockets(N, width):
    """Narrow Gaussian pockets on the AA points of the supercell and nothing else: the pocket regime,
    where the hard |k| < pi/a cutoff rings below zero at the empty AB points. [iy, ix] order."""
    box = im.Box(N, N, L / N)
    x = box.x0 + box.dx * np.arange(N)
    X, Y = np.meshgrid(x, x, indexing="xy")
    n = np.zeros_like(X)
    for px, py in im.moire_points(L, box)["AA"]:
        dx_ = X - px - L * np.round((X - px) / L)
        dy_ = Y - py - L * np.round((Y - py) / L)
        n += np.exp(-(dx_**2 + dy_**2) / (2 * width**2))
    return n, box


def test_smooth_envelope_has_no_ringing():
    """The smooth envelope's transfer is W(0) = 1, W(pi/a) = 0, and its kernel is non-negative, so on
    sharp AA pockets (where the raw envelope rings below zero at AB and C_raw > 1) the smooth envelope
    stays >= 0 everywhere and C_smooth <= 1; raw_ringing flags the raw case."""
    assert im.smooth_transfer(0.0) == pytest.approx(1.0) and im.smooth_transfer(np.pi) == pytest.approx(0.0, abs=1e-15)
    k = np.linspace(0, np.pi, 50)
    assert np.all(np.diff(im.smooth_transfer(k)) <= 0)
    n, box = _pockets(176, 0.35)
    c = im.moire_contrasts(n, box)
    assert c["S_raw_AB"] < 0 and c["C_raw_AAAB"] > 1.0 and c["raw_ringing"]  # the failure mode
    assert c["S_smooth_AB"] >= 0.0 and 0.0 < c["C_smooth_AAAB"] <= 1.0
    KX, KY = np.meshgrid(box.kx0, box.ky0, indexing="xy")
    env = np.fft.ifft2(np.fft.fft2(n) * im.smooth_transfer(np.sqrt(KX**2 + KY**2))).real
    assert env.min() > -1e-12 * env.max()
    # a smooth moire cosine has no ringing: the flag is off and C_raw is the analytic value
    x = box.x0 + box.dx * np.arange(176)
    X, Y = np.meshgrid(x, x, indexing="xy")
    G = 4 * np.pi / L
    c2 = im.moire_contrasts(1 + 0.1 * (np.cos(G * X) + np.cos(G * Y)), box)
    assert not c2["raw_ringing"] and c2["C_smooth_AAAB"] == pytest.approx(0.2 * float(im.smooth_transfer(G)), rel=1e-9)
    so = im.standard_observables(n.T, n.T, box.dx)
    assert so["raw_ringing"] and so["C_smooth"] == pytest.approx(c["C_smooth_AAAB"])


def test_evaluate_nyquist_mode_is_symmetric():
    """A pure cos(pi x/dx) cos(pi y/dy) mode on an even grid: +-1 on the grid points, 0 half-way between
    them (the symmetric interpolant, as in tof's pixel binning; the one-sided exponential gave -1)."""
    N = 16
    box = im.Box(N, N, 0.5)
    x = box.x0 + box.dx * np.arange(N)
    X, Y = np.meshgrid(x, x, indexing="xy")
    f = np.cos(np.pi * (X - box.x0) / box.dx) * np.cos(np.pi * (Y - box.y0) / box.dx)
    Fk = np.fft.fft2(f)
    on = im.evaluate(Fk, box, x[:4], x[:4])
    assert np.allclose(on, f[:4, :4], atol=1e-12)
    half = im.evaluate(Fk, box, x[:4] + 0.5 * box.dx, x[:4] + 0.5 * box.dx)
    assert np.abs(half).max() < 1e-12
    # a single-axis Nyquist mode times a smooth one is unchanged by the symmetric form
    g = np.cos(np.pi * (X - box.x0) / box.dx) * np.cos(2 * np.pi * (Y - box.y0) / (N * box.dx))
    s = 0.3 * box.dx
    val = im.evaluate(np.fft.fft2(g), box, [x[2] + s], [x[5]])[0, 0]
    assert val == pytest.approx(
        np.cos(np.pi * (2 * box.dx + s) / box.dx) * np.cos(2 * np.pi * 5 * box.dx / (N * box.dx)), abs=1e-12
    )
