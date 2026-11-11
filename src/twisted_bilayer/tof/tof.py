"""Time-of-flight (TOF) and in-situ absorption images of a 2D two-component condensate released from
the twisted-bilayer lattice, by Bloch-order demodulation.

Units: hbar = m = 1, length a = 395 nm (lattice constant), time unit 213.517 us (18 ms TOF = 84.302);
constants from twisted_bilayer.parameters (inputs in twisted_bilayer.config).

Pipeline (`tof_pipeline` chains the four steps):

    psi (fine grid, in situ)
      -> meanfield_stage(...)      2D GPE with g(t) = g2D/sqrt(1+omega_z^2 t^2), lattice off,
                                   the full fine field, 0 <= t <= t1
      -> demodulate(...)           phi_G(r) = LPF[psi(r) e^{-iG.r}] on coarse grids
      -> propagate_orders(...)     exact free propagation of every envelope in its own co-moving box
                                   (displacement G t as a q-space phase), with an optional per-order
                                   mean-field tail for t > t1
      -> image(...)                n_TOF = sum_G |chi_G|^2 through the coherent absorption-imaging
                                   transfer (amplitude pupil, pulse push at saturation s_sat, depth of
                                   field), exact pixel integrals on 1.455 um pixels
    Metrics: satellite_contrast (C_sat, non-overlapping disks), satellite_contrast_profile (C_prof,
    fringe-averaged), visibility (the paper's line-cut V).
    The mean-field stage takes dt='auto' at <= 2 rad kinetic phase per step and refuses an explicit dt
    above 3 rad (the split step is unstable near pi).

Representation (lab frame): psi(r, t) = sum_G exp(i G.r) chi_G(r, t), with chi_G stored on its own coarse
grid (Envelope). All coarse grids share one lattice of points x = x_ref + integer * dx, so their densities
add exactly. Arrays may be numpy arrays or (for the propagation kernels) torch tensors.

Labels: tof uses the mirror labels -- lattice_G(2) is the paper's lattice 1 and lattice_G(1) the paper's
lattice 2, so for a state in the paper's labels the imaged |2> orders are lattice_G(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft
import torch as _torch

from twisted_bilayer import parameters as _P

# ----------------------------------------------------------------------------
# constants (code units; values from twisted_bilayer.parameters)
# ----------------------------------------------------------------------------

A_NM = _P.A_NM
T_UNIT_MS = _P.T_UNIT_MS  # code time unit in ms (0.2135172)
T_TOF = _P.T_TOF_MS / T_UNIT_MS  # the time of flight in code units (84.302 at 18 ms)
OMEGA_Z = _P.OMEGA_Z  # 2 pi x 1 kHz = 1.341568
G2D = _P.G2D  # g_3D/(sqrt(2 pi) l_z) = 0.0777651
PIXEL_TOF = _P.PIXEL_TOF_UM / _P.A_UM  # 3.6835 a (1.455 um TOF pixel)
L_SUPER = _P.L_SUPERCELL  # supercell, 22.0227 a
L_MOIRE = _P.L_MOIRE  # moire period, 11.0114 a
C_TW, S_TW = _P.C_ROT, _P.S_ROT  # cos/sin(theta/2), tan(theta/2) = n/m = 1/22 (twisted_bilayer.config)
_WORKERS = -1
A_UM = A_NM * 1e-3  # lattice constant in um
L_Z = 1.0 / math.sqrt(OMEGA_Z)  # oscillator length of the pancake, 0.863363 a = 341 nm

# split-step stability of the mean-field stage: the largest kinetic phase per step
# 0.5 dt k2_max (k2_max = kx_Nyq^2 + ky_Nyq^2 of the actual grid)
MF_PHASE_TARGET = _P.MF_PHASE_TARGET  # dt='auto' -> phase per step <= this (2.0 rad)
MF_PHASE_MAX = _P.MF_PHASE_MAX  # an explicit dt above this raises (3.0 rad, below pi)
MF_DT_CAP = _P.MF_DT_CAP  # dt='auto' never exceeds 0.01

# absorption imaging: Rb D2 probe, Meng et al. Suppl. I
LAMBDA_IMG_UM = _P.LAMBDA_PROBE_UM  # probe wavelength (um)
NA_IMG = _P.NA_IMG  # numerical aperture of the in-situ objective
GAMMA_D2 = _P.GAMMA_D2  # natural linewidth (1/s)
V_RECOIL = _P.V_RECOIL  # recoil velocity (m/s), 5.8845e-3
T_PULSE = _P.T_PULSE_S  # imaging pulse (s)
OMEGA_Z_SI = 2 * math.pi * _P.NU_Z_HZ  # pancake trap frequency (1/s)
L_Z_UM = L_Z * A_UM  # 0.341 um


def ms_to_code(t_ms) -> float:
    return t_ms / T_UNIT_MS


def talbot_time(period) -> float:
    """Talbot time of a field periodic with `period` (hbar=m=1): T_T = d^2/pi."""
    return period**2 / math.pi


def g_of_t(t, g2D, omega_z, s0=1.0) -> np.ndarray:
    """The in-plane coupling of a pancake released at t = 0 from a Gaussian of width s0 l_z
    (g2D = g3D / (sqrt(2 pi) l_z), the coupling of the bare oscillator width): the free Gaussian
    widens as sigma(t)^2 = sigma0^2 (1 + (omega_z t / s0^2)^2), so
        g(t) = g2D / (s0 sqrt(1 + (omega_z t / s0^2)^2)).
    s0 = 1 (default) is the release from the oscillator ground state, g2D / sqrt(1 + omega_z^2 t^2); an
    NPSE in-situ state has s0 = s(n) > 1 (`npse_release_width`). The interaction-driven part of the
    vertical expansion is neglected either way."""
    s0 = float(s0)
    return g2D / (s0 * np.sqrt(1.0 + (omega_z * np.asarray(t, float) / (s0 * s0)) ** 2))


def g_integral(t, g2D, omega_z, s0=1.0) -> float | np.ndarray:
    """int_0^t g(t') dt' of `g_of_t` = g2D s0 asinh(omega_z t / s0^2) / omega_z."""
    t = np.asarray(t, float)
    s0 = float(s0)
    if omega_z == 0:
        return g2D * t / s0
    return g2D * s0 * np.arcsinh(omega_z * t / (s0 * s0)) / omega_z


def npse_release_width(n, g2D, omega_z) -> float:
    """The NPSE width s(n) (root of s^4 - x s - 1 = 0, x = g2D n / omega_z; see engine.npse) of a
    density n (atoms per a^2, any shape), averaged with the interaction weight n^2: the s0 of `g_of_t`
    for releasing that state with one cloud-wide width. Units: any with hbar = 1 (x is dimensionless)."""
    n = np.asarray(n, float)
    x = (float(g2D) / float(omega_z)) * n
    s = np.maximum(np.abs(x) ** (1.0 / 3.0), 1.0)
    for _ in range(8):
        s = s - (s**4 - x * s - 1.0) / (4.0 * s**3 - x)
    w = n * n
    return float((s * w).sum() / w.sum()) if w.sum() > 0 else 1.0


def kinetic_phase_per_step(dt, dx, dy, Nx, Ny) -> float:
    """Largest kinetic phase per split-step, 0.5 dt (kx_max^2 + ky_max^2), on the FFT
    grid (Nx, Ny, dx, dy) (kx_max = pi/dx for even Nx; the 2D corner of the grid).
    The split-step NLS is parametrically unstable when this approaches pi."""
    kx = np.abs(_kvec(int(Nx), dx)).max()
    ky = np.abs(_kvec(int(Ny), dy)).max()
    return 0.5 * float(dt) * float(kx * kx + ky * ky)


def stable_dt(dx, dy, Nx, Ny, phase=MF_PHASE_TARGET, dt_cap=MF_DT_CAP) -> float:
    """Largest dt <= dt_cap with kinetic phase per step <= `phase` (rad)."""
    p1 = kinetic_phase_per_step(1.0, dx, dy, Nx, Ny)
    return min(float(dt_cap), phase / p1)


def lattice_G(which=2, nmax=1, c=C_TW, s=S_TW, diagonals=True) -> np.ndarray:
    """Reciprocal vectors of lattice 1 or 2 in the symmetric frame.

    V2 = V0[sin^2(pi(xc - ys)) + sin^2(pi(xs + yc))] has primitive reciprocal
    vectors b1 = 2pi(c, -s), b2 = 2pi(s, c) (lattice 1: s -> -s).
    Returns the (2nmax+1)^2 vectors m b1 + n b2, |m|,|n| <= nmax (diagonals
    m,n != 0 optionally dropped), zero first.
    """
    sg = -1.0 if which == 2 else 1.0
    b1 = 2 * np.pi * np.array([c, sg * s])
    b2 = 2 * np.pi * np.array([-sg * s, c])
    out = [np.zeros(2)]
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            if m == 0 and n == 0:
                continue
            if not diagonals and m != 0 and n != 0:
                continue
            out.append(m * b1 + n * b2)
    return np.array(out)


def square_G(nmax=1, diagonals=True) -> np.ndarray:
    """Axis-aligned square-lattice reciprocal vectors 2pi(m, n)."""
    return lattice_G(which=2, nmax=nmax, c=1.0, s=0.0, diagonals=diagonals)


# ----------------------------------------------------------------------------
# small backend shim (numpy / torch) for the propagation kernels
# ----------------------------------------------------------------------------
class _NumpyBK:
    name = "numpy"

    @staticmethod
    def fft2(a):
        return sfft.fft2(a, axes=(-2, -1), workers=_WORKERS)

    @staticmethod
    def ifft2(a):
        return sfft.ifft2(a, axes=(-2, -1), workers=_WORKERS)

    @staticmethod
    def asarray(a, like):
        return np.asarray(a, dtype=like.dtype)

    @staticmethod
    def abs2(a):
        return a.real**2 + a.imag**2

    @staticmethod
    def expi(phase):
        """exp(i*phase) for a real backend array; complex dtype follows phase."""
        return np.exp(1j * phase)

    @staticmethod
    def to_numpy(a):
        return a

    @staticmethod
    def real_dtype(a):
        return np.float32 if a.dtype == np.complex64 else np.float64

    @staticmethod
    def zeros(shape, like):
        return np.zeros(shape, dtype=like.dtype)

    @staticmethod
    def as_real(a, like):
        return np.asarray(a, dtype=_NumpyBK.real_dtype(like))

    @staticmethod
    def sum(a):
        return float(a.sum())

    @staticmethod
    def copy(a):
        return a.copy()


class _TorchBK:
    name = "torch"

    def __init__(self, device):
        self.device = device

    @staticmethod
    def fft2(a):
        return _torch.fft.fft2(a, dim=(-2, -1))

    @staticmethod
    def ifft2(a):
        return _torch.fft.ifft2(a, dim=(-2, -1))

    def asarray(self, a, like):
        return _torch.as_tensor(np.asarray(a), device=like.device).to(like.dtype)

    @staticmethod
    def abs2(a):
        return a.real**2 + a.imag**2

    @staticmethod
    def expi(phase):
        return _torch.polar(_torch.ones_like(phase), phase)

    @staticmethod
    def to_numpy(a):
        return a.detach().cpu().numpy()

    @staticmethod
    def real_dtype(a):
        return _torch.float32 if a.dtype == _torch.complex64 else _torch.float64

    @staticmethod
    def zeros(shape, like):
        return _torch.zeros(shape, dtype=like.dtype, device=like.device)

    def as_real(self, a, like):
        return _torch.as_tensor(np.asarray(a), device=like.device).to(self.real_dtype(like))

    @staticmethod
    def sum(a):
        return float(a.sum())

    @staticmethod
    def copy(a):
        return a.clone()


def _bk(a):
    if isinstance(a, _torch.Tensor):
        return _TorchBK(a.device)
    return _NumpyBK


def _to_numpy(a):
    if isinstance(a, _torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _pair(v):
    if np.ndim(v) == 0:
        return float(v), float(v)
    return float(v[0]), float(v[1])


def _as_stack(psi):
    """(Ny,Nx) -> (1,Ny,Nx); (nc,Ny,Nx) unchanged. Returns (stack, was_2d)."""
    if psi.ndim == 2:
        return psi[None], True
    if psi.ndim == 3:
        return psi, False
    raise ValueError("psi must be (Ny,Nx) or (ncomp,Ny,Nx)")


def _fast_even(n):
    n = int(max(n, 2))
    while True:
        m = sfft.next_fast_len(n, real=False)
        if m % 2 == 0:
            return m
        n = m + 1


def _kvec(n, d):
    return 2 * np.pi * np.fft.fftfreq(n, d)


# ----------------------------------------------------------------------------
# data containers
# ----------------------------------------------------------------------------
@dataclass
class Envelope:
    """One Bloch order: psi_G(r) = exp(i G.r) phi(r) on its own coarse grid.

    phi : (ncomp, Ny, Nx) complex (numpy or torch); phi[c, j, i] is the
          lab-frame envelope at x = x0 + i dx, y = y0 + j dy.
    """

    G: np.ndarray
    phi: object
    x0: float
    y0: float
    dx: float
    dy: float

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.phi.shape[-2:])

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.shape[1])

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.dy * np.arange(self.shape[0])

    def density(self, component=None) -> np.ndarray:
        p = _to_numpy(self.phi)
        d = p.real**2 + p.imag**2
        if component is None:
            return d.sum(0)
        return d[component]

    def norm(self, component=None) -> float:
        return float(self.density(component).sum() * self.dx * self.dy)


@dataclass
class Orders:
    """Set of Bloch-order envelopes at time t (time since release)."""

    envs: list
    t: float = 0.0
    dropped: float = 0.0  # fraction of input norm outside all filters
    norm_in: float = 0.0  # norm of the demodulated field
    info: dict = field(default_factory=dict)

    def norms(self, component=None) -> np.ndarray:
        return np.array([e.norm(component) for e in self.envs])

    def weights(self, component=None) -> np.ndarray:
        """Norm fractions of the orders (empty for no orders); a zero total norm is a ValueError."""
        n = self.norms(component)
        if len(n) and not n.sum() > 0:
            raise ValueError(f"Orders.weights: the total norm of the {len(n)} orders is {n.sum()}; no weights")
        return n / n.sum() if len(n) else n


@dataclass
class MFResult:
    psi: object  # (Ny,Nx) or (nc,Ny,Nx), same type as input
    dx: float
    dy: float
    origin: tuple
    t: float
    diag: dict = field(default_factory=dict)


@dataclass
class TOFImage:
    img: np.ndarray  # atoms per pixel (Ny_pix, Nx_pix)
    xc: np.ndarray  # pixel-centre coordinates (a)
    yc: np.ndarray
    pixel: float
    per_order: list = None  # per-order images (same grid) if requested
    cross_bound: float = 0.0  # sum_pix 2 sqrt(I_k I_l) / sum_pix I (k<l)
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# grid helpers
# ----------------------------------------------------------------------------
def fourier_resample(psi, dx, factor_or_n) -> tuple[np.ndarray, tuple[float, float], float]:
    """Band-limited resampling of a (stack of) 2D field(s) on the same box by
    FFT truncation / zero padding. `factor_or_n` = (Nx_new, Ny_new).
    Returns (psi_new, (dx_new, dy_new), dropped_weight_fraction)."""
    p, was2d = _as_stack(_to_numpy(psi))
    dx, dy = _pair(dx)
    nc, Ny, Nx = p.shape
    Nxn, Nyn = factor_or_n
    pk = sfft.fft2(p, axes=(-2, -1), workers=_WORKERS)
    mx = np.fft.fftfreq(Nxn, 1.0 / Nxn).astype(int)
    my = np.fft.fftfreq(Nyn, 1.0 / Nyn).astype(int)
    out = np.zeros((nc, Nyn, Nxn), dtype=pk.dtype)
    okx = np.isin(mx, np.fft.fftfreq(Nx, 1.0 / Nx).astype(int))
    oky = np.isin(my, np.fft.fftfreq(Ny, 1.0 / Ny).astype(int))
    # drop the (ambiguous) Nyquist rows of the target when truncating
    if Nxn < Nx:
        okx &= mx != -Nxn // 2
    if Nyn < Ny:
        oky &= my != -Nyn // 2
    ix_t, iy_t = np.nonzero(okx)[0], np.nonzero(oky)[0]
    ix_s, iy_s = mx[ix_t] % Nx, my[iy_t] % Ny
    out[:, iy_t[:, None], ix_t[None, :]] = pk[:, iy_s[:, None], ix_s[None, :]]
    tot = np.sum(np.abs(pk) ** 2)
    kept = np.sum(np.abs(out) ** 2)
    # zero padding of an even grid: the source Nyquist coefficient (-N/2) stands for cos(pi x/dx), the sum of
    # the +N/2 and -N/2 modes of the finer grid; split it between them so a real field stays real
    if Nxn > Nx and Nx % 2 == 0:
        out[:, :, Nx // 2] = 0.5 * out[:, :, Nxn - Nx // 2]
        out[:, :, Nxn - Nx // 2] *= 0.5
    if Nyn > Ny and Ny % 2 == 0:
        out[:, Ny // 2, :] = 0.5 * out[:, Nyn - Ny // 2, :]
        out[:, Nyn - Ny // 2, :] *= 0.5
    out *= (Nxn * Nyn) / (Nx * Ny)
    res = sfft.ifft2(out, axes=(-2, -1), workers=_WORKERS)
    if was2d:
        res = res[0]
    return res, (Nx * dx / Nxn, Ny * dy / Nyn), float(1 - kept / tot) if tot > 0 else 0.0


def _rebox(phi, x0, y0, dx, dy, new_shape, center):
    """Copy (nc,Ny,Nx) field into a new zero box of `new_shape`=(Ny',Nx')
    centred near `center` with origin on the same point lattice.
    Returns (phi_new, x0', y0', lost_weight_fraction)."""
    B = _bk(phi)
    nc, Ny, Nx = phi.shape
    Nyn, Nxn = new_shape
    ix = int(round((center[0] - 0.5 * Nxn * dx - x0) / dx))
    iy = int(round((center[1] - 0.5 * Nyn * dy - y0) / dy))
    out = B.zeros((nc, Nyn, Nxn), like=phi)
    sx0, sx1 = max(0, ix), min(Nx, ix + Nxn)
    sy0, sy1 = max(0, iy), min(Ny, iy + Nyn)
    lost = 0.0
    if sx1 > sx0 and sy1 > sy0:
        out[:, sy0 - iy : sy1 - iy, sx0 - ix : sx1 - ix] = phi[:, sy0:sy1, sx0:sx1]
    tot = B.sum(B.abs2(phi))
    if tot > 0:
        lost = 1.0 - B.sum(B.abs2(out)) / tot
    return out, x0 + ix * dx, y0 + iy * dy, float(max(lost, 0.0))


def _support(phi_np, dx, dy, eps_real, eps_q):
    """Real-space extent (|phi|^2 > eps_real max) and q half-widths
    (|phi~|^2 > eps_q max). Returns (ix_lo, ix_hi, iy_lo, iy_hi, kx_sup, ky_sup)."""
    d = (phi_np.real**2 + phi_np.imag**2).sum(0)
    m = d.max()
    if not (np.isfinite(m) and m > 0):
        raise ValueError(f"_support: the field is zero or not finite everywhere (max |phi|^2 = {m}); no support")
    cols = np.nonzero(d.max(axis=0) > eps_real * m)[0]
    rows = np.nonzero(d.max(axis=1) > eps_real * m)[0]
    S = np.abs(sfft.fft2(phi_np, axes=(-2, -1), workers=_WORKERS)) ** 2
    S = S.sum(0)
    Sm = S.max()
    kx = np.abs(_kvec(phi_np.shape[-1], dx))
    ky = np.abs(_kvec(phi_np.shape[-2], dy))
    kxs = kx[S.max(axis=0) > eps_q * Sm].max()
    kys = ky[S.max(axis=1) > eps_q * Sm].max()
    return cols[0], cols[-1], rows[0], rows[-1], float(kxs), float(kys)


# ----------------------------------------------------------------------------
# 1. demodulation
# ----------------------------------------------------------------------------
def demodulate(psi, dx, G_list, q_cut=np.pi, dx_coarse=0.5, origin=None, filt="disc", t=0.0) -> Orders:
    """Split psi into Bloch-order envelopes, psi(r) = sum_G e^{iG.r} phi_G(r).

    phi_G = LPF[psi e^{-iG.r}] with the filter |q| < q_cut ('disc', default
    q_cut = pi = half the lattice reciprocal vector) or the Wigner-Seitz cell
    of the vectors in G_list ('ws'; an exact partition of q-space).
    Each phi_G is returned on a coarse grid (same box, spacing ~dx_coarse,
    obtained by FFT truncation; the coarse Nyquist pi/dx_c must exceed q_cut).

    psi    : (Ny,Nx) or (ncomp,Ny,Nx) complex (numpy or torch)
    dx     : grid spacing (scalar or (dx,dy)); box L = N dx (periodic FFT box)
    G_list : (n,2) reciprocal vectors
    origin : lab coordinates of psi[...,0,0]; default (-Lx/2, -Ly/2)
    t      : time label stored in the result (time since release)
    Returns Orders.
    """
    p, _ = _as_stack(_to_numpy(psi).astype(complex))
    dx, dy = _pair(dx)
    nc, Ny, Nx = p.shape
    Lx, Ly = Nx * dx, Ny * dy
    x0, y0 = origin if origin is not None else (-0.5 * Lx, -0.5 * Ly)
    dxc_t, dyc_t = _pair(dx_coarse)
    Ncx = min(Nx, 2 * int(math.ceil(Lx / dxc_t / 2)))
    Ncy = min(Ny, 2 * int(math.ceil(Ly / dyc_t / 2)))
    dxc, dyc = Lx / Ncx, Ly / Ncy
    G_list = np.atleast_2d(np.asarray(G_list, float))
    if filt == "disc" and (q_cut >= np.pi / dxc or q_cut >= np.pi / dyc):
        raise ValueError("coarse grid too coarse: need pi/dx_coarse > q_cut")

    pk = sfft.fft2(p, axes=(-2, -1), workers=_WORKERS)
    norm_in = float(np.sum(np.abs(p) ** 2) * dx * dy)
    # coarse frequencies (integers m, k = 2 pi m / L), drop coarse Nyquist
    mxc = np.fft.fftfreq(Ncx, 1.0 / Ncx).astype(int)
    myc = np.fft.fftfreq(Ncy, 1.0 / Ncy).astype(int)
    qx, qy = 2 * np.pi * mxc / Lx, 2 * np.pi * myc / Ly
    QX, QY = np.meshgrid(qx, qy)
    if filt == "disc":
        mask = QX**2 + QY**2 < q_cut**2
    elif filt == "ws":
        mask = np.ones_like(QX, dtype=bool)
        for b in G_list:
            nb = b @ b
            if nb < 1e-12:
                continue
            # tie-break on the boundary: keep the half with b "positive"
            pos = (b[0] > 1e-12) or (abs(b[0]) <= 1e-12 and b[1] > 0)
            lhs = QX * b[0] + QY * b[1]
            mask &= (lhs <= 0.5 * nb + 1e-9 * nb) if pos else (lhs < 0.5 * nb - 1e-9 * nb)
    else:
        raise ValueError("filt must be 'disc' or 'ws'")
    mask[myc == -Ncy // 2, :] = False
    mask[:, mxc == -Ncx // 2] = False
    xg = x0 + dx * np.arange(Nx)
    yg = y0 + dy * np.arange(Ny)
    envs = []
    kept = 0.0
    scale = (Ncx * Ncy) / (Nx * Ny)
    for G in G_list:
        Mx, My = G[0] * Lx / (2 * np.pi), G[1] * Ly / (2 * np.pi)
        on_grid = abs(Mx - round(Mx)) < 1e-9 and abs(My - round(My)) < 1e-9
        if on_grid:
            # (psi e^{-iG.r})~[m] = e^{-iG.r0} psi~[m + M]
            sx = (mxc + int(round(Mx))) % Nx
            sy = (myc + int(round(My))) % Ny
            spec = pk[:, sy[:, None], sx[None, :]] * np.exp(-1j * (G[0] * x0 + G[1] * y0))
        else:
            ph = np.exp(-1j * G[1] * yg)[:, None] * np.exp(-1j * G[0] * xg)[None, :]
            full = sfft.fft2(p * ph, axes=(-2, -1), workers=_WORKERS)
            spec = full[:, (myc % Ny)[:, None], (mxc % Nx)[None, :]]
        spec = spec * mask * scale
        phi = sfft.ifft2(spec, axes=(-2, -1), workers=_WORKERS)
        env = Envelope(G=np.array(G, float), phi=phi, x0=x0, y0=y0, dx=dxc, dy=dyc)
        kept += env.norm()
        envs.append(env)
    dropped = 1.0 - kept / norm_in if norm_in > 0 else 0.0
    info = dict(q_cut=q_cut, filt=filt, dx_coarse=(dxc, dyc), fine_box=(Lx, Ly), fine_dx=(dx, dy))
    return Orders(envs=envs, t=float(t), dropped=float(dropped), norm_in=norm_in, info=info)


def reconstruct_field(orders, dx, Nx, Ny, origin, component=None) -> np.ndarray | None:
    """Re-assemble psi(r) = sum_G e^{iG.r} chi_G(r) on a fine grid (testing).

    Each envelope is band-limited-interpolated to spacing dx (needs
    dx_c/dx integer and the fine origin on the same point lattice)."""
    dxf, dyf = _pair(dx)
    x0f, y0f = origin
    out = None
    for e in orders.envs:
        p = _to_numpy(e.phi)
        if component is not None:
            p = p[component : component + 1]
        rx, ry = e.dx / dxf, e.dy / dyf
        if abs(rx - round(rx)) > 1e-9 or abs(ry - round(ry)) > 1e-9:
            raise ValueError("coarse/fine spacings not commensurate")
        rx, ry = int(round(rx)), int(round(ry))
        nyc, nxc = p.shape[-2:]
        up, _, _ = fourier_resample(p, (e.dx, e.dy), (nxc * rx, nyc * ry))
        up = up.reshape(p.shape[0], nyc * ry, nxc * rx)
        ix = (e.x0 - x0f) / dxf
        iy = (e.y0 - y0f) / dyf
        if abs(ix - round(ix)) > 1e-6 or abs(iy - round(iy)) > 1e-6:
            raise ValueError("grids not aligned")
        ix, iy = int(round(ix)), int(round(iy))
        if out is None:
            out = np.zeros((p.shape[0], Ny, Nx), complex)
        xs = x0f + dxf * np.arange(Nx)
        ys = y0f + dyf * np.arange(Ny)
        carrier = np.exp(1j * e.G[1] * ys)[:, None] * np.exp(1j * e.G[0] * xs)[None, :]
        sx0, sx1 = max(0, ix), min(Nx, ix + up.shape[-1])
        sy0, sy1 = max(0, iy), min(Ny, iy + up.shape[-2])
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        out[:, sy0:sy1, sx0:sx1] += up[:, sy0 - iy : sy1 - iy, sx0 - ix : sx1 - ix] * carrier[None, sy0:sy1, sx0:sx1]
    return out


# ----------------------------------------------------------------------------
# 2. propagation of the orders
# ----------------------------------------------------------------------------
def _sep_phase(B, like, ax, ay):
    """separable q-space factors exp(-i ax(kx)), exp(-i ay(ky)) as backend arrays"""
    ex = B.asarray(np.exp(-1j * ax), like)
    ey = B.asarray(np.exp(-1j * ay), like)
    return ex[None, None, :], ey[None, :, None]


NLS_HIGHK_FRAC = 0.75  # the tail's high-k weight: the norm at |k| > 0.75 k_Nyquist (as in meanfield_stage)
NLS_HIGHK_GROWTH = 10.0  # the tail fails when that weight grows by more than this (the conservation gate's 10)


def _highk_weight(B, pk, kx, ky, frac=NLS_HIGHK_FRAC) -> float:
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    kny2 = min(np.abs(kx).max(), np.abs(ky).max()) ** 2
    mask = B.as_real((k2 > frac**2 * kny2).astype(float), pk)
    a2 = B.abs2(pk)
    tot = float(B.sum(a2))
    return float(B.sum(a2 * mask[None])) / tot if tot > 0 else 0.0


def _nls_comoving(phi, dx, dy, t_a, t_b, g2D, omega_z, dphi=0.02, dt_max=0.5, s0=1.0):
    """i d_t phi = [-1/2 lap + g(t) sum_c |phi_c|^2] phi on a periodic box,
    Strang splitting with merged kinetic half-steps; the time integral of g
    over each potential sub-step is exact (g of `g_of_t` with release width s0). Step size:
    int g/g2D dt <= dphi (at s0 = 1), dt <= dt_max.

    The step is not limited by the kinetic phase per step (k_max^2 dt / 2 reaches ~80 rad on the dx = 0.5
    coarse grid, where meanfield_stage keeps it below pi): the split step's resonant instability needs a
    nonlinear phase per step that the tail's dilute orders do not have (~1e-2 rad; the result agrees with a
    30x finer schedule to 4e-5 of the image peak). Its symptom, growth of the weight at |k| > 0.75 k_Nyq, is
    measured instead (`propagate_orders` fails on growth above NLS_HIGHK_GROWTH).
    Returns (phi, n_steps, diag) with diag = dict(kin_phase_max, highk=(start, end))."""
    B = _bk(phi)
    nc, Ny, Nx = phi.shape
    kx, ky = _kvec(Nx, dx), _kvec(Ny, dy)
    # step boundaries
    ts = [t_a]
    while ts[-1] < t_b - 1e-12:
        t = ts[-1]
        h = dphi * math.sqrt(1.0 + (omega_z * t) ** 2)
        h = min(h, dt_max, t_b - t)
        ts.append(t + h)
    ts = np.array(ts)
    hs = np.diff(ts)
    Phi = np.diff(g_integral(ts, g2D, omega_z, s0))
    pk = B.fft2(phi)
    hk0 = _highk_weight(B, pk, kx, ky)
    # first half kick
    ex, ey = _sep_phase(B, phi, 0.25 * hs[0] * kx**2, 0.25 * hs[0] * ky**2)
    pk = pk * ex * ey
    for j in range(len(hs)):
        p = B.ifft2(pk)
        n = B.abs2(p).sum(0)
        p = p * B.expi(-Phi[j] * n)[None]
        pk = B.fft2(p)
        hk = 0.5 * hs[j] + (0.5 * hs[j + 1] if j + 1 < len(hs) else 0.0)
        ex, ey = _sep_phase(B, phi, 0.5 * hk * kx**2, 0.5 * hk * ky**2)
        pk = pk * ex * ey
    diag = dict(
        kin_phase_max=float(0.5 * hs.max() * (np.abs(kx).max() ** 2 + np.abs(ky).max() ** 2)),
        highk=(hk0, _highk_weight(B, pk, kx, ky)),
    )
    return B.ifft2(pk), len(hs), diag


def spin_rotate_orders(orders, J_perp: float, tau: float) -> Orders:
    """Every two-component envelope rotated by exp(-i J_perp sigma_x tau):
        phi_1 -> cos(J tau) phi_1 - i sin(J tau) phi_2,   phi_2 -> cos(J tau) phi_2 - i sin(J tau) phi_1,
    the coupling term of `meanfield_stage` integrated exactly. In the tail it commutes with everything else:
    the kinetic energy is the same for both components and the interaction g(t) (|phi_1|^2 + |phi_2|^2) is
    SU(2) invariant, so the rotation can be applied after the free or mean-field propagation of each order.
    One-component orders, J_perp = 0 or tau = 0 are returned unchanged. Returns a new Orders."""
    th = float(J_perp) * float(tau)
    if th == 0.0 or not orders.envs or orders.envs[0].phi.shape[0] != 2:
        return orders
    c, s = math.cos(th), math.sin(th)
    out = []
    for e in orders.envs:
        B = _bk(e.phi)
        p = B.copy(e.phi)
        p[0], p[1] = c * e.phi[0] - 1j * s * e.phi[1], c * e.phi[1] - 1j * s * e.phi[0]
        out.append(Envelope(G=e.G, phi=p, x0=e.x0, y0=e.y0, dx=e.dx, dy=e.dy))
    info = {**orders.info, "spin_rotation": dict(J_perp=float(J_perp), tau=float(tau), angle=th)}
    return Orders(envs=out, t=orders.t, dropped=orders.dropped, norm_in=orders.norm_in, info=info)


def propagate_orders(
    orders,
    t_end,
    *,
    periodic=False,
    eps=1e-6,
    eps_crop=1e-14,
    margin=8.0,
    displace=True,
    g2D=None,
    omega_z=OMEGA_Z,
    dphi=0.05,
    dt_max=2.0,
    phase_tol=0.02,
    s0=1.0,
    highk_check="raise",
) -> Orders:
    """Propagate every envelope from orders.t to t_end (time since release); s0: the release width of
    the mean-field tail's g(t) (`g_of_t`).

    Linear (default, g2D=None): exact free propagation of each order,
        chi_G(r, t_end) = e^{-i|G|^2 tau/2} phi_G^free(r - G tau, tau),
        phi_G^free = IFFT[ phi~_G(q) e^{-i q^2 tau/2} ],  tau = t_end - orders.t,
    evaluated in a co-moving zero-padded box sized from the field's support:
        half-width = (real-space extent at eps_crop)/2 + 1.05 q_sup*tau + margin,
    with q_sup the largest |q_x| (|q_y|) where |phi~|^2 > eps * max (plus the sound speed sqrt(2 g n_max)
    for an order on the mean-field tail); the 5 % is headroom for the spreading tails.
    The displacement G tau is split into an integer number of coarse cells
    (origin shift) and a sub-cell remainder applied as e^{-i q.f}.

    Mean-field tail (g2D given): each order evolves in its co-moving frame
    under i d_t phi = [-1/2 lap + g(t) sum_c|phi_c|^2] phi (Galilean
    invariance makes this exact once the orders no longer overlap); orders
    whose maximal nonlinear phase  max|phi|^2 int g dt  is below phase_tol
    (rad) are propagated linearly. Steps: int g/g2D dt <= dphi, dt <= dt_max. Every NLS order records its
    largest kinetic phase per step and its high-k weight (start, end) in info['propagate']; a growth of that
    weight above NLS_HIGHK_GROWTH x max(start, floor) (floor 1e-14, 1e-8 in single precision) raises
    (highk_check='raise', default), warns ('warn') or is only recorded ('off').

    periodic=True: keep each box (periodic field, e.g. Talbot tests); the
    whole displacement is applied as a phase.
    displace=False: skip the G tau drift and the carrier phase (co-moving
    envelope, for tests).
    Returns a new Orders at time t_end.
    """
    tau = float(t_end - orders.t)
    out = []
    if highk_check not in ("raise", "warn", "off"):
        raise ValueError(f"highk_check must be 'raise', 'warn' or 'off', got {highk_check!r}")
    info = dict(boxes=[], lost=[], nls_steps=[], nls_kin_phase_max=[], nls_highk=[])
    for e in orders.envs:
        phi = e.phi
        B = _bk(phi)
        if phi.ndim == 2:
            phi = phi[None]
        G = np.asarray(e.G, float)
        x0, y0 = e.x0, e.y0
        dx, dy = e.dx, e.dy
        nmax_dens = None
        use_nls = False
        if g2D is not None and g2D > 0 and tau > 0:
            nmax_dens = float(B.to_numpy(B.abs2(phi).sum(0)).max())
            phase = (g_integral(orders.t + tau, g2D, omega_z, s0) - g_integral(orders.t, g2D, omega_z, s0)) * nmax_dens
            use_nls = phase > phase_tol
        lost = 0.0
        if not periodic:
            pn = B.to_numpy(phi)
            ixl, ixh, iyl, iyh, kxs, kys = _support(pn, dx, dy, eps_crop, eps)
            if use_nls:  # interaction-driven velocities (sound speed)
                vmf = math.sqrt(2 * g2D * nmax_dens)
                kxs, kys = kxs + vmf, kys + vmf
            hx = 0.5 * (ixh - ixl + 1) * dx + kxs * abs(tau) + margin + 0.05 * kxs * abs(tau)
            hy = 0.5 * (iyh - iyl + 1) * dy + kys * abs(tau) + margin + 0.05 * kys * abs(tau)
            Nxn, Nyn = _fast_even(math.ceil(2 * hx / dx)), _fast_even(math.ceil(2 * hy / dy))
            cx = x0 + 0.5 * (ixl + ixh) * dx
            cy = y0 + 0.5 * (iyl + iyh) * dy
            phi, x0, y0, lost = _rebox(phi, x0, y0, dx, dy, (Nyn, Nxn), (cx, cy))
        Ny, Nx = phi.shape[-2:]
        kx, ky = _kvec(Nx, dx), _kvec(Ny, dy)
        d = G * tau if displace else np.zeros(2)
        if periodic:
            fx, fy, nx, ny = d[0], d[1], 0, 0
        else:
            nx, ny = int(round(d[0] / dx)), int(round(d[1] / dy))
            fx, fy = d[0] - nx * dx, d[1] - ny * dy
        nsteps = 0
        if use_nls:
            phi, nsteps, nd = _nls_comoving(
                phi, dx, dy, orders.t, orders.t + tau, g2D, omega_z, dphi=dphi, dt_max=dt_max, s0=s0
            )
            info["nls_kin_phase_max"].append(nd["kin_phase_max"])
            info["nls_highk"].append(nd["highk"])
            single = str(phi.dtype).endswith("complex64")
            h0, h1 = nd["highk"]
            if h1 > NLS_HIGHK_GROWTH * max(h0, 1e-8 if single else 1e-14) and highk_check != "off":
                msg = (
                    f"propagate_orders: the high-k weight of order G = {G} grew from {h0:.2e} to {h1:.2e} in the "
                    f"mean-field tail ({nsteps} steps, kinetic phase up to {nd['kin_phase_max']:.1f} rad per step): "
                    "split-step instability; lower dt_max or dphi"
                )
                if highk_check == "raise":
                    raise RuntimeError(msg)
                import warnings

                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            ax, ay = kx * fx, ky * fy
        else:
            ax, ay = 0.5 * tau * kx**2 + kx * fx, 0.5 * tau * ky**2 + ky * fy
        pk = B.fft2(phi)
        ex, ey = _sep_phase(B, phi, ax, ay)
        phi = B.ifft2(pk * ex * ey)
        if displace:
            phi = phi * complex(np.exp(-0.5j * (G @ G) * tau))
        out.append(Envelope(G=G, phi=phi, x0=x0 + nx * dx, y0=y0 + ny * dy, dx=dx, dy=dy))
        info["boxes"].append((Nx * dx, Ny * dy))
        info["lost"].append(lost)
        info["nls_steps"].append(nsteps)
    return Orders(
        envs=out,
        t=float(t_end),
        dropped=orders.dropped,
        norm_in=orders.norm_in,
        info={**orders.info, "propagate": info},
    )


# ----------------------------------------------------------------------------
# 3. mean-field stage on the full fine field
# ----------------------------------------------------------------------------
def meanfield_stage(
    psi,
    dx,
    t1,
    g2D,
    omega_z=OMEGA_Z,
    *,
    origin=None,
    dt="auto",
    box="auto",
    eps=1e-6,
    eps_crop=1e-14,
    margin=8.0,
    dx_mf=None,
    J_perp=0.0,
    record=True,
    record_every=1,
    phase_target=MF_PHASE_TARGET,
    phase_max=MF_PHASE_MAX,
    dt_check="raise",
    highk_frac=0.75,
    s0=1.0,
) -> MFResult:
    """Nonlinear 2D expansion of the full field for 0 <= t <= t1 (lattice off):

        i d_t psi_c = [-1/2 lap + g(t) n] psi_c + J_perp psi_{c'},
        n = sum_c |psi_c|^2,  g(t) = g2D / (s0 sqrt(1 + (omega_z t / s0^2)^2))   (`g_of_t`).

    s0 : the vertical width at release in units of l_z: 1 (default) for a state with the oscillator's
          width (fixed g2D), the cloud's NPSE width for an NPSE state (`npse_release_width`; the
          dense pockets have s0 = 1.04 at n = 3, 1.15 at n = 11.6, and the s0 = 1 coupling is then
          off by a factor s0 at t = 0 and by 1/s0 late).

    psi : (Ny,Nx) or (ncomp,Ny,Nx) (numpy or torch). With two components
          (psi1, psi2) both expand and interact (SU(2) g); J_perp couples
          them (microwave left on; default 0 = off at release).
    The field must vanish at the box edges (finite cloud). box='auto' pads
    (or crops) to   extent(eps_crop) + 2 k_sup t1 + 2 margin,  k_sup from
    the |psi~|^2 > eps*max support (orders and their envelopes);
    box=(Lx, Ly) forces a size; box=None keeps the input box (periodic).
    dx_mf : optional coarser spacing (FFT truncation, dropped weight reported).
    Strang splitting, fixed dt, merged kinetic half-steps; the potential
    sub-step uses the exact int g dt, so n is conserved within it.
    Time step. The split-step NLS is parametrically unstable once the largest kinetic phase
    per step, 0.5 dt k2_max (k2_max = kx_Nyq^2 + ky_Nyq^2 of the grid actually used, after
    reboxing / dx_mf), approaches pi. dt='auto' (default) takes the largest dt <= 0.01 with
    phase <= phase_target (2.0 rad): dt = 3.17e-3 at 8 pts/a. An
    explicit dt whose phase exceeds phase_max (3.0 rad) raises ValueError
    (dt_check='raise', default), warns ('warn') or is accepted ('off').
    diag['kin_phase_per_step'] records the value; diag['highk_weight'] the fraction of
    the norm at |k| > highk_frac * k_Nyq (start, end) -- a grown parametric
    instability shows up there first.
    Returns MFResult(psi(t1), dx, dy, origin, t1, diag) with diag holding
    t_K (all step boundaries), t_T, T (kinetic energy at the boundaries, every
    `record_every` steps and at the end), t_V, I = int n^2 (at every potential
    sub-step), norm, box, steps. Precision: a torch complex64 input runs in single precision; a numpy
    complex64 input comes back complex128 (the float64 phase arrays promote it), which costs speed and
    memory, not accuracy.
    """
    is_torch = isinstance(psi, _torch.Tensor)
    p, was2d = _as_stack(psi)
    B = _bk(p)
    dx, dy = _pair(dx)
    nc, Ny, Nx = p.shape
    x0, y0 = origin if origin is not None else (-0.5 * Nx * dx, -0.5 * Ny * dy)
    diag = {}
    if dx_mf is not None:
        dxm, dym = _pair(dx_mf)
        if dxm > dx * (1 + 1e-12) or dym > dy * (1 + 1e-12):
            Nxn = _fast_even(math.ceil(Nx * dx / dxm))
            Nyn = _fast_even(math.ceil(Ny * dy / dym))
            pn, (dx, dy), drop = fourier_resample(B.to_numpy(p), (dx, dy), (Nxn, Nyn))
            p = B.asarray(pn, p) if is_torch else pn.astype(p.dtype)
            nc, Ny, Nx = p.shape
            diag["dx_mf_dropped"] = drop
    if box is not None:
        pn = B.to_numpy(p)
        ixl, ixh, iyl, iyh, kxs, kys = _support(pn, dx, dy, eps_crop, eps)
        edge = (np.abs(pn[:, :, [0, -1]]) ** 2).max() + (np.abs(pn[:, [0, -1], :]) ** 2).max()
        diag["edge_density_rel"] = float(edge / (np.abs(pn) ** 2).max())
        if box == "auto":
            hx = 0.5 * (ixh - ixl + 1) * dx + kxs * t1 + margin
            hy = 0.5 * (iyh - iyl + 1) * dy + kys * t1 + margin
        else:
            hx, hy = 0.5 * box[0], 0.5 * box[1]
        Nxn, Nyn = _fast_even(math.ceil(2 * hx / dx)), _fast_even(math.ceil(2 * hy / dy))
        cx = x0 + 0.5 * (ixl + ixh) * dx
        cy = y0 + 0.5 * (iyl + iyh) * dy
        p, x0, y0, lost = _rebox(p, x0, y0, dx, dy, (Nyn, Nxn), (cx, cy))
        diag["rebox_lost"] = lost
        diag["k_sup"] = (kxs, kys)
        nc, Ny, Nx = p.shape
    diag["box"] = (Nx * dx, Ny * dy)
    diag["grid"] = (Nx, Ny)
    if isinstance(dt, str):
        if dt != "auto":
            raise ValueError("dt must be a number or 'auto'")
        dt = stable_dt(dx, dy, Nx, Ny, phase=phase_target)
    else:
        ph = kinetic_phase_per_step(dt, dx, dy, Nx, Ny)
        if ph > phase_max and dt_check != "off" and g2D != 0:  # g = 0: exact for any dt
            msg = (
                f"meanfield_stage: kinetic phase per step 0.5*dt*k2_max = {ph:.2f} rad > "
                f"{phase_max} (dt = {dt:g}, dx = {dx:.4g}); the split-step NLS is unstable near "
                f"pi -- use dt='auto' (dt <= {stable_dt(dx, dy, Nx, Ny, phase=phase_target):.3g})"
            )
            if dt_check == "raise":
                raise ValueError(msg)
            import warnings

            warnings.warn(msg, RuntimeWarning, stacklevel=2)
    nsteps = max(1, int(math.ceil(t1 / dt - 1e-12))) if t1 > 0 else 0
    if nsteps == 0:
        res = p[0] if was2d else p
        return MFResult(res, dx, dy, (x0, y0), 0.0, diag)
    h = t1 / nsteps
    diag["kin_phase_per_step"] = kinetic_phase_per_step(h, dx, dy, Nx, Ny)
    kx, ky = _kvec(Nx, dx), _kvec(Ny, dy)
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    k2b = B.as_real(k2, p)
    kny2 = min(np.abs(kx).max(), np.abs(ky).max()) ** 2
    hk_mask = B.as_real((k2 > (highk_frac**2) * kny2).astype(float), p)

    def highk(pk):
        a2 = B.abs2(pk)
        return B.sum(a2 * hk_mask[None]) / B.sum(a2)

    Kfull = B.asarray(np.exp(-0.5j * h * k2), p)[None]
    Khalf = B.asarray(np.exp(-0.25j * h * k2), p)[None]
    ts = h * np.arange(nsteps + 1)
    Phi = np.diff(g_integral(ts, g2D, omega_z, s0))
    diag["s0"] = float(s0)
    cJ, sJ = math.cos(J_perp * h), math.sin(J_perp * h)
    dA = dx * dy
    NN = Nx * Ny

    def kin(pk):
        return 0.5 * B.sum(B.abs2(pk) * k2b[None]) * dA / NN

    Ts, tTs, Is, norms = [], [], [], []
    pk = B.fft2(p)
    hk0 = highk(pk)
    if record:
        Ts.append(kin(pk))
        tTs.append(0.0)
    pk = pk * Khalf
    for j in range(nsteps):
        p = B.ifft2(pk)
        n = B.abs2(p).sum(0)
        if record:
            Is.append(B.sum(n * n) * dA)
            norms.append(B.sum(n) * dA)
        ph = B.expi(-Phi[j] * n)[None]
        if J_perp != 0.0 and nc == 2:
            p0 = cJ * p[0] - 1j * sJ * p[1]
            p1 = cJ * p[1] - 1j * sJ * p[0]
            p = B.copy(p)
            p[0], p[1] = p0, p1
        p = p * ph
        pk = B.fft2(p)
        if record and ((j + 1) % record_every == 0 or j + 1 == nsteps):
            Ts.append(kin(pk))
            tTs.append(ts[j + 1])
        pk = pk * (Kfull if j + 1 < nsteps else Khalf)
    diag["highk_weight"] = (hk0, highk(pk))
    p = B.ifft2(pk)
    if record:
        diag.update(
            t_K=ts, t_T=np.array(tTs), T=np.array(Ts), t_V=ts[:-1] + 0.5 * h, I=np.array(Is), norm=np.array(norms)
        )
    diag["steps"] = nsteps
    diag["dt"] = h
    res = p[0] if was2d else p
    return MFResult(res, dx, dy, (x0, y0), float(t1), diag)


def released_kinetic_energy(diag, g2D, omega_z) -> float:
    """-1/2 int_0^t1 g(t) dI/dt dt from a meanfield_stage record (the in-plane
    kinetic energy that the 2D interaction must have released; exact identity
    dT/dt = -1/2 g(t) dI/dt for the 2D GPE with g = g(t))."""
    I, tK = diag["I"], diag["t_K"]
    if len(I) < 2:
        return 0.0
    dI = np.diff(I)  # centred on tK[1:-1]
    return float(-0.5 * np.sum(g_of_t(tK[1:-1], g2D, omega_z, diag.get("s0", 1.0)) * dI))


# ----------------------------------------------------------------------------
# 4. imaging
# ----------------------------------------------------------------------------
# Absorption imaging of a dilute cloud is coherent: the probe interferes with the forward-scattered
# field, so the optical depth is LINEAR in the column density with the AMPLITUDE pupil as transfer
# function (Hung et al., NJP 13, 075019 (2011), Sec. 3.2). Weak absorption, resonant probe,
# aberration-free circular pupil; f is the spatial frequency in cycles/um:
#   in focus      H(f) = 1 for f < NA/lambda (0.885/um at NA 0.69), 0 beyond;
#   defocus z     H -> H cos(pi lambda z f^2);
#   pulse push    the 50 us pulse pushes the atoms along the beam by z(t) = R v_r t^2/2 and diffuses them
#                 in-plane by <x^2>(t) = R v_r^2 t^3/9, R = (Gamma/2) s/(1+s); the image is the pulse
#                 average of the PRODUCT <exp(-(2 pi f)^2 <x^2>(t)/2) cos(pi lambda z(t) f^2)>_t;
#   depth of field a cloud of rms thickness sigma_z along the line of sight multiplies by
#                 exp(-(pi lambda sigma_z f^2)^2/2); for a TOF image along z, sigma_z = l_z sqrt(1 +
#                 w_z^2 t^2)/sqrt(2) (27 um at 18 ms: 4.35 um fringes x 0.002, 19 um structure x 0.98).


def push_kinematics(s, pulse=T_PULSE, n_t=512) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resonant-pulse push at saturation s: midpoint times t (s), push z(t) (um),
    in-plane diffusion <x^2>(t) (um^2) and the photons scattered per atom."""
    t = (np.arange(n_t) + 0.5) / n_t * pulse
    R = 0.5 * GAMMA_D2 * s / (1.0 + s)
    z = 0.5 * R * V_RECOIL * t**2 * 1e6
    x2 = R * V_RECOIL**2 * t**3 / 9.0 * 1e12
    return t, z, x2, R * pulse


def _radial_eval(fun, f, n_tab=4097):
    """Evaluate a smooth radial function fun(f_1d) on an array of |f| through a dense
    1D table (exact on the table nodes, linear interpolation in between)."""
    f = np.asarray(f, float)
    fmax = float(f.max()) if f.size else 0.0
    if f.size <= n_tab:
        return fun(f.ravel()).reshape(f.shape)
    tab = np.linspace(0.0, fmax, n_tab)
    return np.interp(f, tab, fun(tab))


def coherent_transfer_f(
    f, s=0.0, NA=NA_IMG, wavelength_um=LAMBDA_IMG_UM, pulse=T_PULSE, sigma_z_um=0.0, focus_um=0.0, pupil=True, n_t=512
) -> np.ndarray:
    """Coherent absorption-imaging transfer H(f) of the density (see the block comment):
    top-hat pupil (f < NA/lambda) x pulse average of [recoil diffusion x cos(pi lambda
    (focus + z_push(t)) f^2)] x depth-of-field exp(-(pi lambda sigma_z f^2)^2/2).
    f: |spatial frequency| (cycles/um), any shape. s: saturation parameter (0: no push)."""
    f0 = np.abs(np.asarray(f, float))
    f = np.atleast_1d(f0)
    lam = float(wavelength_um)
    H = np.ones_like(f)
    fc = NA / lam
    if pupil:
        H = H * (f < fc)
    if s > 0 or focus_um != 0.0:
        if s > 0:
            _, z, x2, _ = push_kinematics(s, pulse, n_t)
        else:
            z, x2 = np.zeros(1), np.zeros(1)

        def pulse_avg(ff):
            acc = np.zeros_like(ff)
            k2 = (2 * np.pi * ff) ** 2
            pf = np.pi * lam * ff**2
            for zi, xi in zip(z, x2):
                acc += np.exp(-0.5 * k2 * xi) * np.cos(pf * (focus_um + zi))
            return acc / len(z)

        live = H != 0
        if np.any(live):
            H[live] = H[live] * _radial_eval(pulse_avg, f[live])
    if sigma_z_um > 0:
        H = H * np.exp(-0.5 * (np.pi * lam * sigma_z_um * f**2) ** 2)
    return H.reshape(f0.shape)


def coherent_transfer_k(k, s=0.0, **kw) -> np.ndarray:
    """coherent_transfer_f with the wavenumber k in code units (rad per a)."""
    return coherent_transfer_f(np.asarray(k, float) / (2 * np.pi * A_UM), s=s, **kw)


def sigma_z_tof_um(t_ms, l_z_um=L_Z_UM, omega_z=OMEGA_Z_SI) -> float:
    """rms thickness of the released pancake's column density along z after t_ms:
    l_z sqrt(1 + w_z^2 t^2)/sqrt(2) (non-interacting Gaussian expansion)."""
    t = float(t_ms) * 1e-3
    return l_z_um * math.sqrt(1.0 + (omega_z * t) ** 2) / math.sqrt(2.0)


def insitu_transfer(period_um, s=0.0, **kw) -> float | np.ndarray:
    """Coherent in-situ transfer at a spatial period (um) for saturation s of the 50 us pulse
    (~0.97 / 0.91 / 0.63-0.73 at s = 0.01 / 0.03 / 0.1 for 4.35 um). A float, or an array for an
    array of periods."""
    H = coherent_transfer_f(1.0 / np.asarray(period_um, float), s=s, **kw)
    return float(H) if np.ndim(H) == 0 else H


def tof_transfer(period_um, t_ms=18.0, s=0.0, **kw) -> float | np.ndarray:
    """Coherent transfer of a TOF image taken along z after t_ms of expansion at a
    spatial period (um): depth of field sigma_z = sigma_z_tof_um(t_ms) x in-focus pupil
    x pulse push (s). 18 ms: 0.002 at 4.35 um, 0.98 at 19 um."""
    f = 1.0 / np.asarray(period_um, float)
    H = coherent_transfer_f(f, s=s, sigma_z_um=sigma_z_tof_um(t_ms), **kw)
    return float(H) if np.ndim(H) == 0 else H


def _transfer_2d(kx, ky, s, NA, sigma_z_a, focus_a):
    """(Ny, Nx) coherent transfer on the FFT grid kx (Nx,), ky (Ny,) in rad/a."""
    k = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    return coherent_transfer_k(k, s=s, NA=NA, sigma_z_um=sigma_z_a * A_UM, focus_um=focus_a * A_UM)


def _bin_density(n, x0, y0, dx, dy, xe, ye, transfer):
    """Exact pixel integrals of the transfer-filtered, band-limited (trigonometric)
    interpolant of the density sampled on the box grid:
        I_p = Re sum_k n^(k) H(k) e^{ik.(r_p - r0)} P^2 sinc(kx P/2) sinc(ky P/2) / N,
    H = transfer(kx, ky) -> (Ny, Nx) array, evaluated by two dense matrix products; only pixels overlapping the box
    are evaluated (the box is assumed to carry ~zero density at its edges).
    Requires the density to be resolved by the grid (dx <= pi/(2 q_max))."""
    Ny, Nx = n.shape
    xs_lo, xs_hi = x0 - 0.5 * dx, x0 + (Nx - 0.5) * dx
    ys_lo, ys_hi = y0 - 0.5 * dy, y0 + (Ny - 0.5) * dy
    img = np.zeros((len(ye) - 1, len(xe) - 1))
    jx = np.nonzero((xe[1:] > xs_lo) & (xe[:-1] < xs_hi))[0]
    jy = np.nonzero((ye[1:] > ys_lo) & (ye[:-1] < ys_hi))[0]
    if len(jx) == 0 or len(jy) == 0:
        return img
    Px = xe[1:] - xe[:-1]
    Py = ye[1:] - ye[:-1]
    xc = 0.5 * (xe[1:] + xe[:-1])[jx]
    yc = 0.5 * (ye[1:] + ye[:-1])[jy]
    kx, ky = _kvec(Nx, dx), _kvec(Ny, dy)
    # Nyquist modes: use the symmetric (cosine) interpolant
    nh = sfft.fft2(n, workers=_WORKERS) * transfer(kx, ky)

    def mat(c, k, P, N, s0):
        M = np.exp(1j * np.outer(c - s0, k)) * (P[:, None] * np.sinc(np.outer(P, k) / (2 * np.pi)))
        if N % 2 == 0:
            M[:, N // 2] = np.cos((c - s0) * k[N // 2]) * P * np.sinc(P * k[N // 2] / (2 * np.pi))
        return M

    Mx = mat(xc, kx, Px[jx], Nx, x0)
    My = mat(yc, ky, Py[jy], Ny, y0)
    blk = (My @ nh @ Mx.T).real / (Nx * Ny)
    img[jy[0] : jy[-1] + 1, jx[0] : jx[-1] + 1] = blk
    return img


def _imaging_transfer(s_sat, NA, dof, t, focus):
    """The coherent transfer callable and its description."""
    if dof is None or dof is False:
        sz = 0.0
    elif dof is True:
        if t is None:
            raise ValueError("dof=True needs the expansion time t")
        sz = sigma_z_tof_um(t * T_UNIT_MS) / A_UM
    else:
        sz = float(dof)  # explicit rms thickness (a)
    desc = dict(
        model="coherent",
        s_sat=float(s_sat),
        NA=float(NA),
        sigma_z_a=sz,
        focus_a=float(focus),
        f_cut_um=NA / LAMBDA_IMG_UM,
    )
    return (lambda kx, ky: _transfer_2d(kx, ky, s_sat, NA, sz, focus)), desc


def _pixel_edges(center, half, pixel, offset):
    K = int(math.ceil(half / pixel)) + 1
    k = np.arange(-K, K + 2)
    return center + (k - 0.5 + offset) * pixel


def image(
    orders,
    pixel=PIXEL_TOF,
    center=(0.0, 0.0),
    fov=None,
    offset=(0.0, 0.0),
    component=None,
    per_order=True,
    s_sat=0.0,
    NA=NA_IMG,
    dof=None,
    focus=0.0,
) -> TOFImage:
    """TOF absorption image n_TOF = sum_G |chi_G|^2 (cross terms neglected), seen through the
    coherent imaging transfer (the block comment above coherent_transfer_f) and binned into square
    pixels of side `pixel` (a), centred at center + (k + offset) * pixel. The transfer: sharp low-pass
    of the density at f = NA/lambda, times the pulse-averaged push/diffusion at saturation s_sat
    (0 = no push), times the depth-of-field factor if dof is set (dof=True: sigma_z of the pancake
    expanded for orders.t, i.e. a TOF image along z; a number: sigma_z in a; None/False: off).
    focus: offset of the focal plane from the cloud centre (a).
    fov: half-widths (hx, hy) in a; default: everything.
    component: which field component to image (None = sum).
    Returns TOFImage (atoms per pixel if psi was normalised to atom number).
    The transfer is 1 at k = 0, so the imaged atom number is unchanged."""
    if fov is None:
        hx = hy = 0.0
        for e in orders.envs:
            hx = max(hx, abs(e.x0 - center[0]), abs(e.x[-1] - center[0]))
            hy = max(hy, abs(e.y0 - center[1]), abs(e.y[-1] - center[1]))
    else:
        hx, hy = _pair(fov)
    xe = _pixel_edges(center[0], hx, pixel, offset[0])
    ye = _pixel_edges(center[1], hy, pixel, offset[1])
    H, tdesc = _imaging_transfer(s_sat, NA, dof, orders.t, focus)
    imgs = [_bin_density(e.density(component), e.x0, e.y0, e.dx, e.dy, xe, ye, H) for e in orders.envs]
    tot = np.sum(imgs, axis=0)
    cb = 0.0
    if tot.sum() > 0:
        pos = [np.sqrt(np.clip(I, 0, None)) for I in imgs]
        for k in range(len(pos)):
            for l in range(k + 1, len(pos)):
                cb += 2 * np.sum(pos[k] * pos[l])
        cb /= tot.sum()
    return TOFImage(
        img=tot,
        xc=0.5 * (xe[1:] + xe[:-1]),
        yc=0.5 * (ye[1:] + ye[:-1]),
        pixel=pixel,
        per_order=imgs if per_order else None,
        cross_bound=float(cb),
        info=dict(t=orders.t, **tdesc),
    )


def image_density(
    n,
    dx,
    origin,
    pixel=PIXEL_TOF,
    center=(0.0, 0.0),
    fov=None,
    offset=(0.0, 0.0),
    s_sat=0.0,
    NA=NA_IMG,
    dof=None,
    focus=0.0,
    t=None,
) -> TOFImage:
    """Same imaging chain for a density given on a grid (e.g. brute force);
    dof=True needs the expansion time t (code units)."""
    dx, dy = _pair(dx)
    n = np.asarray(n, float)
    x0, y0 = origin
    if fov is None:
        hx = max(abs(x0 - center[0]), abs(x0 + dx * n.shape[1] - center[0]))
        hy = max(abs(y0 - center[1]), abs(y0 + dy * n.shape[0] - center[1]))
    else:
        hx, hy = _pair(fov)
    xe = _pixel_edges(center[0], hx, pixel, offset[0])
    ye = _pixel_edges(center[1], hy, pixel, offset[1])
    H, tdesc = _imaging_transfer(s_sat, NA, dof, t, focus)
    img = _bin_density(n, x0, y0, dx, dy, xe, ye, H)
    return TOFImage(
        img=img, xc=0.5 * (xe[1:] + xe[:-1]), yc=0.5 * (ye[1:] + ye[:-1]), pixel=pixel, info=dict(t=t, **tdesc)
    )


def visibility(im, d_sat, center=(0.0, 0.0), win0=0.25, win1=0.4, directions=("+x", "-x", "+y", "-y")) -> dict:
    """Satellite visibility of the paper,
        V = (P_av - P_min)/(P_av + P_min),  P_av = (P0 + P1)/2,
    along line cuts through the zeroth-order peak towards each first moire
    satellite (expected at distance d_sat = 2 pi t / L_M from the order).
    P0 = max of the cut within win0*d_sat of the centre; P1 = the LOCAL
    maximum of the cut (a pixel higher than both neighbours) closest to d_sat
    within win1*d_sat of it; P_min = min strictly between the maxima.
    In the near field (R0 >~ q_M t/2) the cut also carries Talbot fringes of
    period L_M, so P1/P_min can be fringe extrema -- check s1, smin.
    If no local maximum exists the satellite is unresolved: resolved=False,
    P1 = value at the pixel nearest d_sat, and V is still reported from it.
    Also returned: V_sat = (P1 - P_min)/(P1 + P_min) (satellite-local
    contrast) and P1/P0. In the near field this metric picks Talbot fringes: prefer C_sat.
    Returns dict with V (mean over directions) and per-direction details."""
    img, xc, yc = im.img, im.xc, im.yc
    i0 = int(np.argmin(np.abs(xc - center[0])))
    j0 = int(np.argmin(np.abs(yc - center[1])))
    res = {}
    Vs, Vsat, nres = [], [], 0
    for d in directions:
        if d[1] == "x":
            prof = img[j0, :]
            s = (xc - center[0]) * (1 if d[0] == "+" else -1)
        else:
            prof = img[:, i0]
            s = (yc - center[1]) * (1 if d[0] == "+" else -1)
        sel = s >= -0.5 * im.pixel
        order = np.argsort(s[sel])
        ss, pp = s[sel][order], prof[sel][order]
        m0 = ss <= win0 * d_sat
        k0 = np.nonzero(m0)[0][np.argmax(pp[m0])]
        cand = [
            k
            for k in range(max(k0 + 2, 1), len(pp) - 1)
            if abs(ss[k] - d_sat) <= win1 * d_sat and pp[k] > pp[k - 1] and pp[k] >= pp[k + 1]
        ]
        resolved = len(cand) > 0
        if resolved:
            k1 = min(cand, key=lambda k: abs(ss[k] - d_sat))
        else:
            k1 = int(np.argmin(np.abs(ss - d_sat)))
        P0, P1 = pp[k0], pp[k1]
        if k1 - k0 >= 2:
            kmin = k0 + 1 + int(np.argmin(pp[k0 + 1 : k1]))
        else:
            kmin = k1
        Pmin = pp[kmin]
        Pav = 0.5 * (P0 + P1)
        V = (Pav - Pmin) / (Pav + Pmin) if Pav + Pmin > 0 else 0.0
        vs = (P1 - Pmin) / (P1 + Pmin) if P1 + Pmin > 0 else 0.0
        Vs.append(V)
        Vsat.append(vs)
        nres += resolved
        res[d] = dict(
            V=float(V),
            V_sat=float(vs),
            resolved=bool(resolved),
            P0=float(P0),
            P1=float(P1),
            Pmin=float(Pmin),
            s0=float(ss[k0]),
            s1=float(ss[k1]),
            smin=float(ss[kmin]),
            ratio_P1_P0=float(P1 / P0) if P0 > 0 else np.nan,
        )
    res["V"] = float(np.mean(Vs))
    res["V_std"] = float(np.std(Vs))
    res["V_sat"] = float(np.mean(Vsat))
    res["n_resolved"] = int(nres)
    return res


def _disk_mean(im, cx, cy, r, sub=5):
    """Mean image value over the disk |x - c| < r (pixel weights = covered fraction,
    sub x sub supersampling); NaN if the disk lies outside the image."""
    P = im.pixel
    i = np.nonzero(np.abs(im.xc - cx) <= r + P)[0]
    j = np.nonzero(np.abs(im.yc - cy) <= r + P)[0]
    if len(i) == 0 or len(j) == 0:
        return float("nan")
    o = (np.arange(sub) + 0.5) / sub - 0.5
    xs = (im.xc[i][:, None] + P * o[None, :]).ravel()
    ys = (im.yc[j][:, None] + P * o[None, :]).ravel()
    inside = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2 < r * r
    w = inside.reshape(len(j), sub, len(i), sub).mean(axis=(1, 3))
    return float((w * im.img[np.ix_(j, i)]).sum() / w.sum()) if w.sum() > 0 else float("nan")


def satellite_contrast(
    im, d_sat, radius=None, center=(0.0, 0.0), step=None, directions=((1, 0), (-1, 0), (0, 1), (0, -1))
) -> dict:
    """Satellite-local contrast with NON-overlapping disks:

        C_sat = (P1 - P_min)/(P1 + P_min),

    P1 = mean image value over a disk of radius r at the expected satellite position
    centre + d_sat e (d_sat = q_M t), P_min = the smallest disk mean for disk centres
    s e with s in [2r, d_sat - 2r], i.e. disks that overlap neither the central disk
    (radius r at the zeroth order) nor the satellite disk. r <= d_sat/4 (at d_sat/4 the
    segment shrinks to s = d_sat/2). Default r = d_sat/8: the disk diameter is then about one
    moire period at 18 ms (12.0 a vs L_M = 11.0 a), so the disks also average the near-field
    Talbot fringes, and P_min can sit up to 0.75 d_sat out, where the dip of a separated
    satellite lies (N = 600 cloud: dip at 36 a for d_sat = 48 a).
    C_sat > 0 means the image dips between the zeroth order and the satellite position
    and rises again: a separated satellite. A smooth blob gives C_sat < 0 ("no
    satellite"; resolved = False). Returns dict(C_sat = mean over directions,
    C_sat_clip = max(C_sat, 0), resolved (all directions C_sat > 0), n_resolved,
    P0, per-direction P1, P_min, s_min, C_sat)."""
    r = d_sat / 8.0 if radius is None else float(radius)
    if r > 0.25 * d_sat + 1e-12:
        raise ValueError("satellite_contrast: radius must be <= d_sat/4")
    step = 0.25 * im.pixel if step is None else step
    P0 = _disk_mean(im, center[0], center[1], r)
    Cs, out, nres = [], {}, 0
    for e in directions:
        e = np.asarray(e, float) / np.hypot(*e)
        P1 = _disk_mean(im, center[0] + d_sat * e[0], center[1] + d_sat * e[1], r)
        ss = np.arange(2 * r, d_sat - 2 * r + 1e-9, step)
        if len(ss) == 0:
            ss = np.array([0.5 * d_sat])
        mins = np.array([_disk_mean(im, center[0] + s_ * e[0], center[1] + s_ * e[1], r) for s_ in ss])
        if not np.isfinite(mins).any():
            raise ValueError(
                f"satellite_contrast: no finite disk mean between the centre and the satellite along {tuple(e)} "
                "(all-NaN image, or the disks lie outside it)"
            )
        k = int(np.nanargmin(mins))
        Pmin = float(mins[k])
        C = (P1 - Pmin) / (P1 + Pmin) if (P1 + Pmin) > 0 else float("nan")
        Cs.append(C)
        nres += bool(C > 0)
        out[tuple(e)] = dict(
            C_sat=float(C), P1=P1, Pmin=Pmin, s_min=float(ss[k]), ratio_P1_P0=P1 / P0 if P0 > 0 else float("nan")
        )
    out["C_sat"] = float(np.mean(Cs))
    out["C_sat_clip"] = max(out["C_sat"], 0.0)
    out["C_sat_std"] = float(np.std(Cs))
    out["n_resolved"] = int(nres)
    out["resolved"] = bool(nres == len(Cs))
    out["P0"] = P0
    out["radius"] = r
    return out


def satellite_contrast_profile(
    im,
    d_sat,
    center=(0.0, 0.0),
    half_angle=None,
    bin_width=None,
    s_max=None,
    sub=3,
    window=0.3,
    directions=((1, 0), (-1, 0), (0, 1), (0, -1)),
) -> dict:
    """Fringe-averaged radial-profile version of the satellite-local contrast.

    For each satellite direction e the image is averaged over the wedge of polar
    half-angle `half_angle` (default asin(1/4) = 14.5 deg, the angle a disk of radius
    d_sat/4 subtends at the satellite) and binned in the radius s with bins of width
    `bin_width` (default one moire period L_M = 11.01 a, which averages the near-field
    Talbot fringes); sub x sub supersampling of every pixel.
    P1 = the largest bin value with |s - d_sat| <= window d_sat; P_min = the smallest
    bin strictly between the central bin (s < bin_width) and the P1 bin (bins are disjoint,
    so these never overlap either). C_prof = (P1 - P_min)/(P1 + P_min); resolved
    if C_prof > 0 and the P1 bin is a local maximum of the profile. Returns
    dict(C_prof (mean), resolved, n_resolved, s (bin centres), profile (mean over
    directions), per-direction details)."""
    hw = math.asin(0.25) if half_angle is None else float(half_angle)
    bw = L_MOIRE if bin_width is None else float(bin_width)
    smax = 1.6 * d_sat if s_max is None else float(s_max)
    P = im.pixel
    o = (np.arange(sub) + 0.5) / sub - 0.5
    xs = (im.xc[:, None] + P * o[None, :]).ravel() - center[0]
    ys = (im.yc[:, None] + P * o[None, :]).ravel() - center[1]
    vals = np.repeat(np.repeat(im.img, sub, axis=0), sub, axis=1)
    X, Y = np.meshgrid(xs, ys)
    Rr = np.hypot(X, Y)
    edges = np.arange(0.0, smax + bw, bw)
    sc = 0.5 * (edges[1:] + edges[:-1])
    Cs, profs, out, nres = [], [], {}, 0
    for e in directions:
        e = np.asarray(e, float) / np.hypot(*e)
        ang = np.arctan2(X * e[1] - Y * e[0], X * e[0] + Y * e[1])
        sel = (np.abs(ang) <= hw) | (Rr < 0.5 * bw)
        idx = np.digitize(Rr[sel], edges) - 1
        ok = (idx >= 0) & (idx < len(sc))
        num = np.bincount(idx[ok], weights=vals[sel][ok], minlength=len(sc))
        cnt = np.bincount(idx[ok], minlength=len(sc))
        prof = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        win = np.nonzero(np.abs(sc - d_sat) <= window * d_sat)[0]
        if len(win) == 0 or not np.isfinite(prof[win]).any():
            raise ValueError(
                f"satellite_contrast_profile: no finite profile bin within {window:g} d_sat of d_sat = {d_sat:g} "
                f"along {tuple(e)} (all-NaN image, an image smaller than the satellite distance, or bins of "
                f"{bw:g} wider than the window)"
            )
        k1 = int(win[np.nanargmax(prof[win])])
        # P_min bins: strictly between the central bin (index 0) and the P1 bin
        mid = np.arange(1, k1)
        if len(mid) == 0 or not np.isfinite(prof[mid]).any():
            kmin = k1
        else:
            kmin = int(mid[np.nanargmin(prof[mid])])
        P1, Pmin = float(prof[k1]), float(prof[kmin])
        C = (P1 - Pmin) / (P1 + Pmin) if (P1 + Pmin) > 0 else float("nan")
        locmax = k1 > 0 and k1 + 1 < len(prof) and prof[k1] >= prof[k1 - 1] and prof[k1] >= prof[k1 + 1]
        res_d = bool(C > 0 and locmax)
        nres += res_d
        Cs.append(C)
        profs.append(prof)
        out[tuple(e)] = dict(
            C_prof=float(C),
            P1=P1,
            Pmin=Pmin,
            s1=float(sc[k1]),
            s_min=float(sc[kmin]),
            local_max=bool(locmax),
            resolved=res_d,
        )
    out["C_prof"] = float(np.mean(Cs))
    out["C_prof_clip"] = max(out["C_prof"], 0.0)
    out["n_resolved"] = int(nres)
    out["resolved"] = bool(nres == len(Cs))
    out["s"] = sc
    out["profile"] = np.nanmean(np.array(profs), axis=0)
    out["bin_width"] = bw
    out["half_angle"] = hw
    return out


# ----------------------------------------------------------------------------
# convenience pipeline
# ----------------------------------------------------------------------------
T_MF_STAGE = ms_to_code(_P.T_MF_STAGE_MS)  # the mean-field stage before the per-order tail (twisted_bilayer.config)


def tof_pipeline(
    psi,
    dx,
    G_list,
    t_tof=T_TOF,
    t1=T_MF_STAGE,
    g2D=G2D,
    omega_z=OMEGA_Z,
    *,
    origin=None,
    image_component=None,
    dt="auto",
    dx_mf=None,
    q_cut=np.pi,
    dx_coarse=0.5,
    filt="disc",
    tail="mf",
    pixel=PIXEL_TOF,
    eps=1e-6,
    J_perp=0.0,
    fov=None,
    offset=(0.0, 0.0),
    mf_box="auto",
    s_sat=0.0,
    NA=NA_IMG,
    dof=True,
    focus=0.0,
    dt_check="raise",
    s0=1.0,
) -> dict:
    """meanfield_stage (t <= t1) -> demodulate -> propagate_orders -> image.
    tail='mf' (default: per-order mean-field tail) or 'linear' (not converged at t1 = 1 ms: 5.0 %
    t1-truncation error at R0 = 25 a against 0.15 % with the MF tail, n_bar = 2).
    Defaults: dt='auto' (kinetic phase <= 2 rad/step), coherent imaging along z with the
    depth-of-field factor of the expanded pancake (dof=True) and no pulse push (s_sat = 0).
    s0: the vertical release width of `g_of_t` for the stage and the tail: 1 (default, the oscillator
    width: right for a fixed-g2D state), a number, or "npse" (`npse_release_width` of the input density:
    use it for an NPSE state).
    J_perp: the interlayer coupling, on for the whole expansion (default 0 = off at release): in the stage as
    its split-step term, after it as the exact spin rotation exp(-i J_perp sigma_x (t_tof - t1)) of every
    order (`spin_rotate_orders`; without a stage over the whole t_tof). Returns dict(image, orders, orders_t1,
    mf, s0); orders_t1 are the orders at the end of the stage, before the tail's rotation."""
    if isinstance(s0, str):
        if s0 != "npse":
            raise ValueError(f"s0 must be a number or 'npse', got {s0!r}")
        p_in, _ = _as_stack(psi)
        s0 = npse_release_width(_bk(p_in).to_numpy(_bk(p_in).abs2(p_in).sum(0)), g2D, omega_z)
    s0 = float(s0)
    if t1 > 0 and g2D > 0:
        mf = meanfield_stage(
            psi,
            dx,
            t1,
            g2D,
            omega_z,
            origin=origin,
            dt=dt,
            dx_mf=dx_mf,
            J_perp=J_perp,
            eps=eps,
            box=mf_box,
            dt_check=dt_check,
            s0=s0,
        )
        psi_t1, dxx, org, t_dem = mf.psi, (mf.dx, mf.dy), mf.origin, t1
    else:  # no stage, or non-interacting: the stage would be a free propagation
        mf, psi_t1, dxx, org, t_dem = None, psi, dx, origin, 0.0
    orders = demodulate(psi_t1, dxx, G_list, q_cut=q_cut, dx_coarse=dx_coarse, origin=org, filt=filt, t=t_dem)
    if tail == "mf" and g2D > 0:
        prop = propagate_orders(orders, t_tof, eps=eps, g2D=g2D, omega_z=omega_z, s0=s0)
    else:
        prop = propagate_orders(orders, t_tof, eps=eps)
    if J_perp != 0.0:  # the microwave stays on after the stage (or for the whole expansion without one)
        prop = spin_rotate_orders(prop, J_perp, float(t_tof) - float(t_dem))
    im = image(
        prop, pixel=pixel, component=image_component, fov=fov, offset=offset, s_sat=s_sat, NA=NA, dof=dof, focus=focus
    )
    return dict(image=im, orders=prop, orders_t1=orders, mf=mf, s0=s0)
