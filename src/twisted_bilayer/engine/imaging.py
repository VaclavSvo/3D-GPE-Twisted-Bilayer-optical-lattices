"""The moire contrast as the experiment images it, and the other observables of a 2D column density
in the paper's conventions.

`standard_observables(n1, n2, dx)` takes the two component densities of an engine state (numpy,
grid order [ix, iy], paper labels: component 2 is the imaged |2>) and returns every number below as
a dict of floats. Inside this module arrays are indexed [iy, ix] (rows = y) on the periodic box
x_j = x0 + j dx with the AA point at the origin; lengths in a, densities in atoms per a^2.

Contrast C = (S_AA - S_AB) / (S_AA + S_AB) of |2>, signed (+ = AA-bright), for these imaging models:
  raw     the moire envelope, n2 low-passed at |k| < pi/a (removes the atomic lattice) -- an ideal imager.
          The hard cutoff rings: its real-space kernel (an Airy amplitude) has negative lobes, so next to
          sharp AA pockets the envelope can go below zero at nearly empty AB points and C_raw > 1.
          `raw_ringing` flags S_raw_AB < 0 (or S_raw_AA < 0);
  smooth  the same band limit |k| < pi/a with the incoherent optical transfer function of a circular pupil,
          W(k) = (2/pi) [arccos(k/k_c) - (k/k_c) sqrt(1 - (k/k_c)^2)], k_c = pi/a (the autocorrelation of a
          disc of radius k_c/2): its kernel is the Airy intensity |2 J1(u)/u|^2 >= 0, so the envelope of a
          non-negative density stays non-negative and |C_smooth| <= 1. It weights the moire harmonics below
          pi/a by W(k) < 1, so C_smooth is not C_raw: it is the raw envelope without the ringing;
  coh     coherent absorption imaging: the amplitude pupil NA = 0.69 with the probe push at saturation
          s (tof.coherent_transfer_k). Compare this one, or C_raw where raw_ringing is False, with data;
  psf     the legacy incoherent Gaussian PSF (sigma = 0.625 a) on 0.32 um pixels; kept for comparison.
S_AA, S_AB are averages over the AA / AB points of a supercell box, or the central AA point and its
four AB corners on a larger (trapped-cloud) box. Every evaluation is exact in Fourier space.

An independent quasi-2D split-step solver using the same definitions gives identical numbers.
"""

from __future__ import annotations

import numpy as np
import torch

from twisted_bilayer import parameters as P
from twisted_bilayer import tof

from .grid import Grid
from .observables import leggett_bounds, leggett_bounds_aligned

L_SUPERCELL = P.L_SUPERCELL
PIXEL = P.PIXEL_INSITU_UM / P.A_UM  # 0.8101 a
PSF_SIGMA = P.PSF_SIGMA_A  # 0.625 a


class Box:
    """Periodic box of Nx x Ny square pixels of side dx, index [0, 0] at (x0, y0)."""

    def __init__(self, Nx: int, Ny: int, dx: float, x0: float | None = None, y0: float | None = None):
        self.Nx, self.Ny, self.dx = int(Nx), int(Ny), float(dx)
        self.Lx, self.Ly = self.Nx * self.dx, self.Ny * self.dx
        self.x0 = -0.5 * self.Lx if x0 is None else float(x0)
        self.y0 = -0.5 * self.Ly if y0 is None else float(y0)
        self.kx0 = 2.0 * np.pi * np.fft.fftfreq(self.Nx, d=self.dx)
        self.ky0 = 2.0 * np.pi * np.fft.fftfreq(self.Ny, d=self.dx)

    def is_supercell(self, L: float) -> bool:
        return abs(self.Lx - L) < 1e-9 and abs(self.Ly - L) < 1e-9


def _contrast(a) -> float:
    """(max - min) / (max + min)."""
    a = np.asarray(a)
    mx, mn = float(a.max()), float(a.min())
    return (mx - mn) / (mx + mn) if (mx + mn) != 0 else float("nan")


# ------------------------------------------------------------------ Fourier-space tools
def _k_mesh(box: Box):
    return np.meshgrid(box.kx0, box.ky0, indexing="xy")


def _interp_matrix(c: np.ndarray, origin: float, k: np.ndarray) -> np.ndarray:
    """exp(i k (c - origin)) for every point and wavenumber; on an even grid the Nyquist column is
    cos(k_N (c - origin)), the symmetric (real) interpolant of the unpaired Nyquist coefficient."""
    E = np.exp(1j * np.outer(c - origin, k))
    if k.size % 2 == 0:
        E[:, k.size // 2] = np.cos((c - origin) * k[k.size // 2])
    return E


def evaluate(field_k: np.ndarray, box: Box, xs, ys) -> np.ndarray:
    """Band-limited interpolation of a real periodic field at the points xs x ys, from its FFT
    `field_k`. Returns (len(ys), len(xs)). The Nyquist modes of an even grid enter with a cosine factor
    per axis (the symmetric interpolant), as in tof's pixel binning: a pure cos(pi x/dx) cos(pi y/dy)
    mode then reads 0 half-way between grid points, not -1."""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    Ex = _interp_matrix(xs, box.x0, box.kx0)
    Ey = _interp_matrix(ys, box.y0, box.ky0)
    return (Ey @ field_k @ Ex.T).real / (box.Nx * box.Ny)


def evaluate_points(field_k: np.ndarray, box: Box, pts) -> np.ndarray:
    """Band-limited interpolation at the points pts[i] = (x, y)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    return np.array([evaluate(field_k, box, [x], [y])[0, 0] for x, y in pts])


def moire_points(L: float, box: Box) -> dict:
    """AA = (L/2)(i, j) and AB = (L/4)(odd, odd) points inside the box."""

    def pts(step, offset):
        def axis(o, length):
            v = offset + step * np.arange(np.floor((o - offset) / step) - 1, np.ceil((o + length - offset) / step) + 1)
            return v[(v >= o - 1e-9) & (v < o + length - 1e-9)]

        xs, ys = axis(box.x0, box.Lx), axis(box.y0, box.Ly)
        return np.array([(x, y) for y in ys for x in xs])

    return {"AA": pts(L / 2, 0.0), "AB": pts(L / 2, L / 4)}


def _central_AB(L: float) -> np.ndarray:
    return np.array([[L / 4, L / 4], [-L / 4, L / 4], [L / 4, -L / 4], [-L / 4, -L / 4]])


def _aa_ab_means(Fk: np.ndarray, box: Box, L: float) -> tuple[float, float]:
    if box.is_supercell(L):
        pts = moire_points(L, box)
        return evaluate_points(Fk, box, pts["AA"]).mean(), evaluate_points(Fk, box, pts["AB"]).mean()
    return evaluate_points(Fk, box, [[0.0, 0.0]])[0], evaluate_points(Fk, box, _central_AB(L)).mean()


def _pixel_centres(origin: float, lo: float, hi: float, pixel: float) -> np.ndarray:
    i0 = int(np.ceil((lo - origin) / pixel - 1e-9))
    i1 = int(np.floor((hi - origin) / pixel - 1e-9))
    return origin + pixel * np.arange(i0, i1 + 1)


# ------------------------------------------------------------------------------ imaging
def image(n2: np.ndarray, box: Box, sigma: float = PSF_SIGMA, pixel: float = PIXEL, origin=(0.0, 0.0), extent=None):
    """In-situ image of a column density through the legacy Gaussian PSF, averaged over square
    pixels centred at origin + pixel (i, j) (one pixel on the AA point), over `extent` =
    (xmin, xmax, ymin, ymax) (default: the box). Returns (xc, yc, img), img in atoms/a^2."""
    KX, KY = _k_mesh(box)
    T = np.exp(-0.5 * sigma**2 * (KX**2 + KY**2)) if sigma > 0 else 1.0
    if pixel > 0:
        T = T * np.sinc(KX * pixel / (2 * np.pi)) * np.sinc(KY * pixel / (2 * np.pi))
    Fk = np.fft.fft2(n2) * T
    xmin, xmax, ymin, ymax = extent or (box.x0, box.x0 + box.Lx, box.y0, box.y0 + box.Ly)
    xc = _pixel_centres(origin[0], xmin, xmax, pixel)
    yc = _pixel_centres(origin[1], ymin, ymax, pixel)
    return xc, yc, evaluate(Fk, box, xc, yc)


def moire_contrasts(
    n2: np.ndarray,
    box: Box,
    L: float = L_SUPERCELL,
    sigma: float = PSF_SIGMA,
    pixel: float = PIXEL,
    k_cut: float = np.pi,
    n_line: int = 401,
) -> dict:
    """Contrasts of the raw envelope and the Gaussian-PSF image: C_{raw,psf}_AAAB (with S_AA, S_AB),
    along x through AA (_line_x), along the diagonal (_diag), within the central moire cell
    (_central), over the box (_global, supercell only); C_img_central / C_img_global on the
    pixelated PSF image (the paper's (max - min)/(max + min)). For a trapped cloud only the AA/AB
    and the central values are meaningful (the global ones see the trap envelope). Also the smooth
    envelope's C_smooth_AAAB with S_smooth_AA/AB (`smooth_transfer`, the same k_cut), and raw_ringing
    (S_raw_AB < 0 or S_raw_AA < 0: C_raw is then outside [-1, 1] or unreliable near it)."""
    KX, KY = _k_mesh(box)
    F = np.fft.fft2(n2)
    out: dict = {}
    for tag, Fk in (("raw", F * (KX**2 + KY**2 < k_cut**2)), ("psf", F * np.exp(-0.5 * sigma**2 * (KX**2 + KY**2)))):
        sAA, sAB = _aa_ab_means(Fk, box, L)
        if box.is_supercell(L):
            out[f"C_{tag}_global"] = _contrast(np.fft.ifft2(Fk).real)
        out[f"S_{tag}_AA"], out[f"S_{tag}_AB"] = float(sAA), float(sAB)
        out[f"C_{tag}_AAAB"] = float((sAA - sAB) / (sAA + sAB))
        s = np.linspace(-L / 4, L / 4, n_line)
        out[f"C_{tag}_line_x"] = _contrast(evaluate(Fk, box, s, [0.0])[0])
        out[f"C_{tag}_diag"] = _contrast(
            np.array([evaluate(Fk, box, [t], [t])[0, 0] for t in np.linspace(-L / 4, L / 4, 101)])
        )
        cell = evaluate(Fk, box, np.linspace(-L / 4, L / 4, 89), np.linspace(-L / 4, L / 4, 89))
        out[f"C_{tag}_central"] = _contrast(cell)
    sAA, sAB = _aa_ab_means(F * smooth_transfer(np.sqrt(KX**2 + KY**2), k_cut), box, L)
    out["S_smooth_AA"], out["S_smooth_AB"] = float(sAA), float(sAB)
    out["C_smooth_AAAB"] = float((sAA - sAB) / (sAA + sAB))
    out["raw_ringing"] = bool(out["S_raw_AB"] < 0.0 or out["S_raw_AA"] < 0.0)
    h = L / 4 + pixel / 2  # the central cell plus half a pixel: the AB corners are inside
    out["C_img_central"] = _contrast(image(n2, box, sigma, pixel, extent=(-h, h, -h, h))[2])
    if box.is_supercell(L):
        out["C_img_global"] = _contrast(image(n2, box, sigma, pixel)[2])
    return out


def smooth_transfer(k, k_cut: float = np.pi) -> np.ndarray:
    """W(k) = (2/pi) [arccos(q) - q sqrt(1 - q^2)], q = k / k_cut < 1, else 0: the incoherent transfer
    function of a circular pupil of cutoff k_cut. W(0) = 1, band-limited at k_cut like the raw envelope,
    and its 2D kernel is non-negative (the Airy intensity), so it preserves n >= 0."""
    q = np.clip(np.asarray(k, float) / float(k_cut), 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(q) - q * np.sqrt(1.0 - q * q))


def transfer_contrasts(n2: np.ndarray, box: Box, transfer, L: float = L_SUPERCELL, pixel: float = PIXEL) -> dict:
    """Contrasts of n2 through a radial imaging transfer H(|k|) (|k| in rad/a): the signed C_AAAB
    with S_AA, S_AB, and C_img_central, the (max - min)/(max + min) of the filtered image on
    `pixel` pixels over the central moire cell."""
    KX, KY = _k_mesh(box)
    Fk = np.fft.fft2(n2) * np.asarray(transfer(np.sqrt(KX**2 + KY**2)), float)
    sAA, sAB = _aa_ab_means(Fk, box, L)
    out = {"S_AA": float(sAA), "S_AB": float(sAB), "C_AAAB": float((sAA - sAB) / (sAA + sAB))}
    T = np.sinc(KX * pixel / (2 * np.pi)) * np.sinc(KY * pixel / (2 * np.pi)) if pixel > 0 else 1.0
    h = L / 4 + pixel / 2
    xc = _pixel_centres(0.0, -h, h, pixel)
    out["C_img_central"] = _contrast(evaluate(Fk * T, box, xc, xc))
    return out


def coherent_transfer(s_sat: float):
    """H(|k|) of coherent absorption imaging at saturation s_sat (see tof.coherent_transfer_k)."""
    return lambda k: tof.coherent_transfer_k(k, s=s_sat)


def pupil_transfer(k):
    """The in-focus amplitude pupil alone: H = 1 below NA / lambda_probe (2.20 rad/a), else 0."""
    k_c = 2.0 * np.pi * (P.NA_IMG / P.LAMBDA_PROBE_UM) * P.A_UM
    return (np.asarray(k, float) < k_c).astype(float)


def cell_averages(n: np.ndarray, box: Box, L: float = L_SUPERCELL, side: float = 1.0) -> dict:
    """n_AA, n_AB: the average of n over an axis-aligned side x side square centred on the AA / AB
    points (one primary cell for side = a), exact in Fourier space; C_cell_n their contrast."""
    KX, KY = _k_mesh(box)
    F = np.fft.fft2(n) * np.sinc(KX * side / (2 * np.pi)) * np.sinc(KY * side / (2 * np.pi))
    nAA, nAB = (float(v) for v in _aa_ab_means(F, box, L))
    return {"n_AA": nAA, "n_AB": nAB, "C_cell_n": (nAA - nAB) / (nAA + nAB)}


def moire_fourier_weights(n2: np.ndarray, box: Box, L: float = L_SUPERCELL) -> dict:
    """|n2~(G)| / n2~(0) at the moire harmonics (2 pi/L)(+-2, 0), (+-2, +-2), (+-4, 0): eps for
    n = nbar (1 + 2 eps cos(G x)). Exact on a box that is a multiple of L."""
    F = np.fft.fft2(n2)
    F0 = abs(F[0, 0])
    q = 2 * np.pi / L

    def amp(vecs):
        return float(
            np.mean(
                [
                    abs(
                        F[
                            int(np.rint(n * q * box.Ly / (2 * np.pi))) % box.Ny,
                            int(np.rint(m * q * box.Lx / (2 * np.pi))) % box.Nx,
                        ]
                    )
                    / F0
                    for m, n in vecs
                ]
            )
        )

    return {
        "F_moire_10": amp([(2, 0), (-2, 0), (0, 2), (0, -2)]),
        "F_moire_11": amp([(2, 2), (-2, 2), (2, -2), (-2, -2)]),
        "F_moire_20": amp([(4, 0), (-4, 0), (0, 4), (0, -4)]),
    }


# ------------------------------------------------------------------------------ bundle
def standard_observables(
    n1: np.ndarray,
    n2: np.ndarray,
    dx: float,
    imaging: str = "coherent",
    s_sat: float | None = None,
    mn: tuple = (P.TWIST_M, P.TWIST_N),
    side: float = 1.0,
) -> dict:
    """Every density observable of this module for two component densities in the engine's grid
    order [ix, iy] (x_j = (j - N//2) dx, paper labels) on the commensurate geometry tan(theta/2) =
    n/m: the box is the supercell of side L = sqrt(m^2 + n^2) a, or a multiple of it (a trapped cloud;
    then the central cell is used).

    On |2>: C_raw, C_psf, C_smooth with S_{raw,psf,smooth}_{AA,AB}, raw_ringing (1.0 or 0.0), C_img_central,
    C_paper_signed and the diagnostics of `moire_contrasts`; C_coh, S_coh_AA/AB, C_coh_img through the imaging transfer ('coherent' at
    s_sat, default config S_SAT = 0.01; 'pupil' = the in-focus pupil, s = 0); F_moire_10/11/20.
    On the total density: C_raw_total; n_AA, n_AB, C_cell_n; the Leggett bounds of f_s in the box
    frame (fs_lower/upper_x/y: loose while the primary lattice dominates, the box axes are rotated by
    theta/2 from both lattices) and along the axes of lattice 1 / 2 (fs_*_lat1/2, supercell box only,
    NaN otherwise). The bounds are diagnostics; f_s itself comes from the twist. Also N1, N2, N2_over_N."""
    n1x, n2x = np.asarray(n1, float), np.asarray(n2, float)
    Nx, Ny = n2x.shape
    L = float(np.sqrt(mn[0] ** 2 + mn[1] ** 2))
    box = Box(Nx, Ny, dx, -(Nx // 2) * dx, -(Ny // 2) * dx)
    n2y, ny = n2x.T, (n1x + n2x).T  # [iy, ix]
    out: dict = {"dx": float(dx), "Lx": box.Lx, "Ly": box.Ly}
    c = moire_contrasts(n2y, box, L)
    out["C_raw"], out["C_psf"], out["C_smooth"] = c["C_raw_AAAB"], c["C_psf_AAAB"], c["C_smooth_AAAB"]
    out["raw_ringing"] = float(c["raw_ringing"])  # 1.0 / 0.0: the bundle returns floats only
    for k in (
        "S_smooth_AA",
        "S_smooth_AB",
        "S_raw_AA",
        "S_raw_AB",
        "S_psf_AA",
        "S_psf_AB",
        "C_img_central",
        "C_raw_central",
        "C_psf_line_x",
        "C_psf_diag",
        "C_psf_central",
        "C_raw_global",
        "C_psf_global",
        "C_img_global",
    ):
        if k in c:
            out[k] = c[k]
    out["C_paper_signed"] = float(np.sign(c["S_psf_AA"] - c["S_psf_AB"]) * c["C_img_central"])
    s = P.S_SAT if s_sat is None else float(s_sat)
    if imaging == "coherent":
        H, model = coherent_transfer(s), f"coherent s={s:g}"
    elif imaging == "pupil":
        H, model = pupil_transfer, "pupil only (s = 0)"
    else:
        raise ValueError("imaging must be 'coherent' or 'pupil'")
    t = transfer_contrasts(n2y, box, H, L)
    out.update(
        C_coh=t["C_AAAB"], S_coh_AA=t["S_AA"], S_coh_AB=t["S_AB"], C_coh_img=t["C_img_central"], imaging_model=model
    )
    out.update(moire_fourier_weights(n2y, box, L))
    out["C_raw_total"] = moire_contrasts(ny, box, L)["C_raw_AAAB"]
    out.update(cell_averages(ny, box, L, side))
    n_t = torch.as_tensor(n1x + n2x)[None]
    grid = Grid((Nx, Ny), (box.Lx, box.Ly))
    for i, ax in enumerate("xy"):
        lo, up = leggett_bounds(n_t, grid, i)
        out[f"fs_lower_{ax}"], out[f"fs_upper_{ax}"] = float(lo[0]), float(up[0])
    on_cell = Nx == Ny and abs(box.Lx - L) < 1e-9 * L
    for lat in (1, 2):
        for ax in "xy":
            lo, up = (
                leggett_bounds_aligned(n1x + n2x, dx, ax, lattice=lat, L=L, pq=mn)
                if on_cell
                else (float("nan"), float("nan"))
            )
            out[f"fs_lower_{ax}_lat{lat}"], out[f"fs_upper_{ax}_lat{lat}"] = lo, up
    dA = dx * dx
    N1, N2 = float(n1x.sum() * dA), float(n2x.sum() * dA)
    out.update(N1=N1, N2=N2, N2_over_N=N2 / (N1 + N2) if N1 + N2 > 0 else float("nan"))
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in out.items()}
