"""Observables of a (converged) field: superfluid fraction by phase twist
(energy and current), Leggett brackets, participation ratio, moire contrast,
coarse-graining. All functions accept batched fields (B, C, *N) and return
per-member tensors (B,) unless stated.

Conventions (lattice units hbar = 1, m = 1/(2 kappa)):

    E(theta) - E(0) = N kappa f_s theta^2 / L^2          (energy route)
    f_s = L^2 I / (2 kappa N theta),  I = number current through a
          cross-section perpendicular to the twist axis   (current route)

Both must agree for a converged state; the current route is first order in
the convergence error, the energy route second order but needs E(0) and
E(theta) converged to well below N kappa theta^2 / L^2.
"""

from __future__ import annotations

import numpy as np
import torch

from .grid import Grid
from .physics import BilayerGPE


def superfluid_fraction_energy(
    E_twist: torch.Tensor, E_zero: torch.Tensor, N: torch.Tensor, L: float, theta: float, kappa: float
) -> torch.Tensor:
    return (E_twist - E_zero) * L * L / (N * kappa * theta * theta)


def cross_section_current(engine: BilayerGPE, psi: torch.Tensor, axis: int) -> torch.Tensor:
    """I_axis = integral of j_axis over the (d-1)-dimensional cross-section,
    averaged along the axis (exactly constant for a stationary state), (B,)."""
    j = engine.current(psi)[axis]  # (B, *N)
    g = engine.grid
    perp_volume = g.volume / g.L[axis]
    return j.mean(dim=tuple(range(1, j.ndim))) * perp_volume


def superfluid_fraction_current(engine: BilayerGPE, psi: torch.Tensor, axis: int, theta: float) -> torch.Tensor:
    """f_s along `axis` from the current of a state relaxed with twist theta
    on that axis (the engine must carry that twist)."""
    I = cross_section_current(engine, psi, axis)
    N = engine.norm(psi)
    L = engine.grid.L[axis]
    return L * L * I / (2.0 * engine.kappa * N * theta)


def leggett_bounds(n: torch.Tensor, grid: Grid, axis: int = 0, eps: float = 1e-300):
    """Leggett brackets for superflow along `axis` from the density alone
    (Perez-Cruz, Astrakharchik & Massignan 2025; the 1D identity when d = 1):

        lower = (1/nbar) < [ <1/n>_axis ]^-1 >_perp     (independent 1D channels in parallel)
        upper = (1/nbar) [ < 1/<n>_perp >_axis ]^-1     (phase depends on the axis coordinate only)

    n: (B, *N) or (B, 1, *N). Returns (lower, upper), each (B,)."""
    if n.ndim == grid.d + 2:
        n = n[:, 0]
    sd = tuple(range(1, n.ndim))
    ax = 1 + axis
    perp = tuple(d for d in sd if d != ax)
    nbar = n.mean(dim=sd)
    inv = 1.0 / n.clamp_min(eps)
    harmonic_along = 1.0 / inv.mean(dim=ax)  # (B, *perp)
    lower = harmonic_along.mean(dim=tuple(range(1, harmonic_along.ndim))) / nbar if perp else harmonic_along / nbar
    if perp:
        m = n.mean(dim=perp)  # (B, N_axis)
        upper = 1.0 / (1.0 / m.clamp_min(eps)).mean(dim=1) / nbar
    else:
        upper = lower
    return lower, upper


def _line_samples(c: np.ndarray, mx: np.ndarray, my: np.ndarray, f0: tuple, step: tuple, S: int, D: int) -> np.ndarray:
    """Exact (band-limited) values of the periodic field with Fourier coefficients c[my, mx]
    (fft2 / N^2) at the P = S D points f0 + (j / S) step / D, j = 0..P-1, in box-fraction
    coordinates: one closed line of length D a along the integer direction `step` = (a, b)
    (gcd 1, a^2 + b^2 = D), S samples per a. One 1D FFT: the modes are binned by M = a mx + b my,
    exact while P exceeds the span of M."""
    P = S * D
    M = (step[0] * mx + step[1] * my) % P
    w = c * np.exp(2j * np.pi * (mx * f0[0] + my * f0[1]))
    B = np.bincount(M.ravel(), weights=w.real.ravel(), minlength=P) + 1j * np.bincount(
        M.ravel(), weights=w.imag.ravel(), minlength=P
    )
    return (np.fft.ifft(B) * P).real


def leggett_bounds_aligned(
    n,
    dx: float,
    axis: str = "x",
    lattice: int = 1,
    L: float | None = None,
    pq: tuple = (22, 1),
    samples_per_a: int | None = None,
) -> tuple[float, float]:
    """Leggett bounds for flow along an axis of lattice 1 or 2 (paper labels) on the commensurate
    supercell, a square box of side L = sqrt(p^2 + q^2) a with tan(theta/2) = q/p. n: a 2D numpy
    density in grid order [ix, iy].

    Lattice 1 has axes e_u = (cos a, -sin a), e_v = (sin a, cos a) (rotated by -theta/2), lattice 2
    the mirror image; axis 'x' is flow along e_u, 'y' along e_v. The formulas of `leggett_bounds` with
    lines along the lattice axes instead of the box axes:

      lower = (1/nbar) < [ <1/n>_line(u) ]^-1 >_v,    upper = (1/nbar) [ < 1/<n>_line(v) >_u ]^-1

    A line along a lattice axis closes after D = p^2 + q^2 lattice constants (485 a for m = 22) and the
    lines through v and v + a coincide, so the outer averages run over one lattice constant. The density
    is sampled on the lines by exact Fourier interpolation, samples_per_a points per a (default: the
    smallest number free of aliasing, at least 16).

    For one lattice at g = 0 both bounds give m/m* (0.65194 at 4 E_R) where the box frame gives
    [0.43, 1.00]. With Rabi-coupled components a bound from the total density is a diagnostic."""
    n = np.asarray(n, dtype=float).T  # [iy, ix]
    Ny, Nx = n.shape
    p, q = int(pq[0]), int(pq[1])
    D = p * p + q * q
    L = float(np.sqrt(D)) if L is None else L
    if Nx != Ny or abs(Nx * dx - L) > 1e-9 * L or abs(L - np.sqrt(D)) > 1e-9 * L:
        raise ValueError(
            f"leggett_bounds_aligned needs the square supercell of side sqrt({D}) a "
            f"(got {Nx} x {Ny}, dx = {dx}, L = {L})"
        )
    if lattice not in (1, 2) or axis not in ("x", "y"):
        raise ValueError("lattice must be 1 or 2, axis 'x' or 'y'")
    s = -1 if lattice == 1 else 1  # e_u = (p, s q)/sqrt(D), e_v = (-s q, p)/sqrt(D)
    e_u, e_v = (p, s * q), (-s * q, p)
    flow, perp = (e_u, e_v) if axis == "x" else (e_v, e_u)
    N = Nx
    S = samples_per_a or max(16, int(np.ceil(((p + q) * (N - 1) + 1) / D)))
    c = np.fft.fft2(n) / n.size
    k = np.rint(np.fft.fftfreq(N) * N).astype(np.int64)
    mx, my = k[None, :], k[:, None]
    nbar = float(n.mean())
    floor = 1e-12 * nbar
    lows, means = [], []
    for j in range(S):
        t = j / S  # offset across the lines, in a
        vals = np.maximum(_line_samples(c, mx, my, (t * perp[0] / D, t * perp[1] / D), flow, S, D), floor)
        lows.append(1.0 / np.mean(1.0 / vals))
        means.append(np.mean(_line_samples(c, mx, my, (t * flow[0] / D, t * flow[1] / D), perp, S, D)))
    lower = float(np.mean(lows)) / nbar
    upper = float(1.0 / np.mean(1.0 / np.maximum(np.asarray(means), floor))) / nbar
    return lower, upper


def participation_ratio(n: torch.Tensor, grid: Grid) -> torch.Tensor:
    """(int n)^2 / (V int n^2) in (0, 1]: the fraction of the box the density
    occupies (1 for uniform)."""
    if n.ndim == grid.d + 2:
        n = n[:, 0]
    sd = tuple(range(1, n.ndim))
    return (n.sum(dim=sd) * grid.dV) ** 2 / (grid.volume * (n * n).sum(dim=sd) * grid.dV)


def gaussian_blur(f: torch.Tensor, grid: Grid, sigma: float) -> torch.Tensor:
    """Periodic Gaussian filter of width sigma (in a) over the spatial axes
    of a real field (…, *N) -- the imaging resolution, or a coarse-graining
    over the primary lattice."""
    if sigma <= 0.0:
        return f
    sd = tuple(range(f.ndim - grid.d, f.ndim))
    fk = torch.fft.fftn(f, dim=sd)
    ksq = None
    for i, ax in enumerate(sd):
        shape = [1] * f.ndim
        shape[ax] = grid.N[i]
        t = (grid.k[i] ** 2).reshape(shape)
        ksq = t if ksq is None else ksq + t
    return torch.fft.ifftn(fk * torch.exp(-0.5 * sigma * sigma * ksq), dim=sd).real


def contrast(n: torch.Tensor, grid: Grid, blur_sigma: float = 0.0) -> torch.Tensor:
    """(max - min) / (max + min) of the (optionally blurred) density, (B,).
    With blur_sigma ~ a the primary lattice is washed out and this is the
    real-space moire visibility of the paper's Fig. 4i (S_max - S_min)/(S_max + S_min)."""
    if n.ndim == grid.d + 2:
        n = n[:, 0]
    nb = gaussian_blur(n, grid, blur_sigma)
    sd = tuple(range(1, nb.ndim))
    hi = nb.amax(dim=sd)
    lo = nb.amin(dim=sd)
    return (hi - lo) / (hi + lo)


def component_fractions(engine: BilayerGPE, psi: torch.Tensor) -> torch.Tensor:
    """N_sigma / N, (B, C)."""
    cn = engine.component_norms(psi)
    return cn / cn.sum(dim=1, keepdim=True)


# -- moire-cell (pocket) bookkeeping ------------------------------------------------


def pocket_cells(grid: Grid, L_box: float | None = None):
    """The four moire cells of the sqrt(m^2+n^2) supercell: squares of side
    L/2 centred at (0,0), (L/2,0), (0,L/2), (L/2,L/2) (periodic). They are NOT
    equivalent: the lattices are in
    register on a common WELL at (0,0), on a common BOND at (L/2,0) and
    (0,L/2), and on a common PLAQUETTE centre at (L/2,L/2); V1+V2 = 0, 2V0,
    2V0, 4V0 there. Returns a list of (name, class, centre, boolean mask (N0, N1))."""
    L = grid.L[0] if L_box is None else L_box
    x = grid.x[0].reshape(-1, 1)
    y = grid.x[1].reshape(1, -1)
    half = 0.25 * L
    out = []
    for name, cls, (cx, cy) in (
        ("AA", "well", (0.0, 0.0)),
        ("B1", "bond", (0.5 * L, 0.0)),
        ("B2", "bond", (0.0, 0.5 * L)),
        ("P", "plaquette", (0.5 * L, 0.5 * L)),
    ):
        dx = (x - cx + 0.5 * L) % L - 0.5 * L
        dy = (y - cy + 0.5 * L) % L - 0.5 * L
        mask = (dx.abs() < half) & (dy.abs() < half)
        out.append((name, cls, (cx, cy), mask))
    return out


def pocket_populations(n: torch.Tensor, grid: Grid, blur_sigma: float = 0.5) -> dict:
    """Per moire cell: atom number N_p; the primary-cell-averaged density at
    the cell centre (Gaussian blur of width blur_sigma); and the choke
    density n_choke = minimum of the blurred density on the straight line
    from the AA (well) centre to the bond centre (L/2, 0). n: (B, *N) or
    (B, 1, *N), 2D only. Returns dict of (B,) tensors."""
    if n.ndim == grid.d + 2:
        n = n[:, 0]
    if grid.d != 2:
        raise ValueError("pocket_populations is for 2D fields")
    nb = gaussian_blur(n, grid, blur_sigma)
    out = {}
    N0, N1 = grid.N
    for name, _cls, (cx, cy), mask in pocket_cells(grid):
        out[f"N_{name}"] = (n * mask).sum(dim=(1, 2)) * grid.dV
        ix = int(round(cx / grid.dx[0])) % N0 + (N0 // 2)
        iy = int(round(cy / grid.dx[1])) % N1 + (N1 // 2)
        out[f"n_{name}"] = nb[:, ix % N0, iy % N1]
    # choke: along y = 0 from x = 0 to x = L/2
    i0 = N0 // 2
    i1 = (i0 + int(round(0.5 * grid.L[0] / grid.dx[0]))) % N0
    seg = nb[:, i0 : i1 + 1, N1 // 2] if i1 > i0 else nb[:, i0:, N1 // 2]
    out["n_choke"] = seg.amin(dim=1)
    out["AA_over_choke"] = out["n_AA"] / out["n_choke"].clamp_min(1e-30)
    return out


def export_state(engine: BilayerGPE, psi: torch.Tensor, path: str, **meta) -> None:
    """Write a field and its diagnostics as npz: psi1, psi2 (complex64), n1, n2, n (float32), mu,
    E_kin, E_pot, E_int, E_rabi, E_total, norm, residual, grid N/L/dx, twist, `meta`, and a `units`
    record. Lattice units (a = 1, E_R = 1, hbar = 1); code units (hbar = m = a = 1) have
    E_code = E * pi^2/2 and t_code = t / (pi^2/2), stored as `E_R_in_code_units`. Fields are in
    grid order [ix, iy]."""
    from .backend import to_numpy
    from .units import E_R_CODE

    t = engine.energy_terms(psi)
    res = engine.residual(psi)
    d = to_numpy(psi[0].real ** 2 + psi[0].imag ** 2).astype(np.float32)
    arrays = dict(
        psi1=to_numpy(psi[0, 0]).astype(np.complex64),
        n1=d[0],
        n=d.sum(axis=0),
        mu=float(t["mu"][0]),
        E_kin=float(t["E_kin"][0]),
        E_pot=float(t["E_pot"][0]),
        E_int=float(t["E_int"][0]),
        E_rabi=float(t["E_rabi"][0]),
        E_total=float(t["E_total"][0]),
        norm=float(t["norm"][0]),
        residual=float(res[0]),
        N=np.array(engine.grid.N),
        L=np.array(engine.grid.L),
        dx=np.array(engine.grid.dx),
        twist=np.zeros(engine.grid.d) if engine.twist is None else to_numpy(engine.twist[0]),
        E_R_in_code_units=E_R_CODE,
        units="lattice: a=1, E_R=1, hbar=1, m=pi^2/2",
    )
    if psi.shape[1] == 2:
        arrays["psi2"] = to_numpy(psi[0, 1]).astype(np.complex64)
        arrays["n2"] = d[1]
    for k, v in meta.items():
        arrays[k] = v
    np.savez_compressed(path, **arrays)
