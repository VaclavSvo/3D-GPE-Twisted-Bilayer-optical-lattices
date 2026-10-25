"""Bogoliubov-de Gennes excitations of a real (zero-current) ground state,
matrix-free on the engine's FFT grid, at any Bloch quasi-momentum q of the box.

Linearisation. With
psi_sigma = e^{-i mu t} [psi0_sigma + u_sigma e^{-i w t} + v_sigma^* e^{+i w t}]
in the engine's GPE (local mean field mu_int(n), = g n for contact), the
fluctuation obeys

    w u = L u + M v,        -w v = L v + M u,

    L = -kappa (nabla + i q)^2 + V_sigma + delta_sigma + mu_int(n) - mu + Omega_R sigma_x + X
    M = X,      (X f)_sigma = g'(r) psi0_sigma sum_sigma' psi0_sigma' f_sigma',   g' = d mu_int/d n,

delta_1 = +Delta/2, delta_2 = -Delta/2. For contact interactions g' = g and
mu_int(n) = g n; for a density functional (engine.npse.NPSE2D) the
diagonal keeps mu_int(n) and only the exchange kernel becomes mu_int'(n).
psi0 is periodic on the box, so the BdG operator commutes with the box
translations acting on the pair (u, v); its eigenvectors are Bloch states and
u and v carry the SAME Bloch factor: u = e^{i q.r} u~, v = e^{i q.r} v~ with
u~, v~ periodic. (The realness of psi0, the gauge fixed below, gives M* = M,
which the reduction uses; the common Bloch factor does not need it.) This module works
with the periodic parts; the kinetic energy is kappa |k + q|^2 on both, which
is the engine's twist trick (twist theta = q L).

Reduction. Write D = L - X = H_GP(q) - mu (the GP Hamiltonian at the frozen
density, Hermitian) and

    A = L + M = D + 2X,     B = L - M = D,     f = u + v,     h = u - v.

Adding and subtracting the two BdG equations gives

    A f = w h,       B h = w f       =>   B A f = w^2 f,   A B h = w^2 h.

A > 0; B >= 0 at a stable minimum, with B psi0 = H psi0 - mu psi0 = 0 at
q = 0 (the GP residual vector). The normalisation int sum_sigma (|u|^2 - |v|^2)
= Re int f^* h = int f^* A f / w is positive for w > 0: the physical branch.
u = (f + h)/2, v = (f - h)/2.

Zero mode (q = 0 only). (u, v) = (psi0, -psi0), i.e. f = 0, h = psi0, is an
exact zero mode (the U(1) Goldstone mode, Hugenholtz-Pines); its Jordan
partner is f = A^{-1} psi0, h = 0 (the number mode). Every mode with w != 0
satisfies psi0^H f = 0 (from B h = w f) and (A^{-1} psi0)^H h = 0 (from
A f = w h), so the zero mode is projected out exactly by these two
constraints: f orthogonal to psi0, h oblique-orthogonal to P0 = A^{-1} psi0
(one preconditioned CG solve). On the constrained spaces A and B are both
positive definite. With C = 2 and Omega_R = 0 the relative phase
(psi1, -psi2) is a second Goldstone mode and is projected the same way.
If in addition both components see the same D (V1 = V2, Delta = 0: the
SU(2)-symmetric gas, since the exchange kernel couples the total density only),
the spin rotation chi = (-psi2, psi1) is in the kernel of A as well as of B,
so A^{-1} does not exist on it: then Z = (psi0, chi) spans ker B, the
h-constraints are chi itself and A^+ psi0 (solved in the complement of chi),
and the dense solver lifts chi in A before its Cholesky factorisation.

Eigensolver. The lowest few w of a spectrum spanning [~1e-2, kappa k_max^2 ~ 1e2]
E_R, matrix-free. The equivalent definite pencil (A B A, A) has a condition
number cubic in the kinetic energy, and shift-invert needs inner solves with
the ill-conditioned B, so `_lr_lobpcg` minimises the linear-response functional
rho(f, h) = (f^H A f + h^H B h) / (2 |Re f^H h|) instead (Thouless 1961; Bai & Li,
SIAM J. Matrix Anal. Appl. 33, 1075 (2012) and 34, 392 (2013), LOBP4dCG):
separate search spaces for f and h, Rayleigh-Ritz on the small pencil
[[0, W], [W^H, 0]] z = (1/w) diag(A~, B~) z (well posed even for singular
W = U^H V), and the kinetic preconditioner (kappa |k+q|^2 + s)^-1. Its
conditioning is linear in kappa k^2 (~10^2 iterations).
`method="dense"` (n <~ 6000 unknowns) diagonalises R B R^H with A = R^H R:
the exact reference of the tests.

Units and conventions are the engine's (a = 1, E_R = 1, hbar = 1,
kappa = 1/pi^2); w is in E_R. The BdG always runs in double precision.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from .physics import BilayerGPE

# -- block linear algebra on (m, C, *N) blocks -----------------------------------------


def _gram(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """X^H Y for blocks of m_x and m_y vectors: (m_x, m_y)."""
    return X.reshape(X.shape[0], -1).conj() @ Y.reshape(Y.shape[0], -1).T


def _comb(X: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """sum_i X_i a_ij: block (m, ...) times coefficients (m, k) -> (k, ...)."""
    out = a.to(X.dtype).T @ X.reshape(X.shape[0], -1)
    return out.reshape((a.shape[1],) + X.shape[1:])


def _norms(X: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((X.reshape(X.shape[0], -1).abs() ** 2).sum(dim=1))


def _independent(Z: torch.Tensor, rtol: float = 1e-8) -> torch.Tensor:
    """The vectors of the block Z (z, ...) that are not (to rtol of their norm) combinations of the ones
    kept before them, in order and unchanged: a dependent zero-mode candidate would make the projection's
    Gram matrices singular."""
    keep = []
    for i in range(Z.shape[0]):
        v = Z[i : i + 1]
        r = v
        if keep:
            Q = torch.cat(keep)
            r = v - _comb(Q, torch.linalg.solve(_gram(Q, Q), _gram(Q, v)))
        if float(_norms(r)[0]) > rtol * float(_norms(v)[0]):
            keep.append(v)
    return torch.cat(keep)


def _orth_against(Q, AQ, X, AX):
    """X - Q (Q^H X) (Q orthonormal), twice for stability; the operator image
    AX is carried along."""
    for _ in range(2):
        c = _gram(Q, X)
        X = X - _comb(Q, c)
        AX = AX - _comb(AQ, c) if AX is not None else None
    return X, AX


def _svqb(X, AX=None, drop: float = 1e-12):
    """Orthonormalise the block (columns normalised first, then the Gram
    matrix's eigenbasis; nearly dependent directions are dropped), twice.
    Returns (Q, AQ, cond) with the operator image carried along and cond the
    largest amplification sqrt(e_max / e_min) of the first pass (how much a
    mismatch between X and AX can grow)."""
    cond = 1.0
    for p in range(2):
        nr = _norms(X).clamp_min(1e-300)
        X = X / nr.reshape((-1,) + (1,) * (X.ndim - 1)).to(X.dtype)
        if AX is not None:
            AX = AX / nr.reshape((-1,) + (1,) * (AX.ndim - 1)).to(AX.dtype)
        G = _gram(X, X)
        G = 0.5 * (G + G.conj().T)
        e, Z = torch.linalg.eigh(G)
        keep = e > drop * float(e.max())
        if p == 0:
            cond = math.sqrt(float(e.max()) / float(e[keep].min()))
        T = Z[:, keep] / torch.sqrt(e[keep]).to(Z.dtype)
        X = _comb(X, T)
        AX = _comb(AX, T) if AX is not None else None
    return X, AX, cond


def _reduced_lr(At: torch.Tensor, Bt: torch.Tensor, W: torch.Tensor, k: int):
    """The k lowest positive w of  At a = w W b,  Bt b = w W^H a  (At, Bt > 0),
    from the Hermitian-definite pencil [[0, W], [W^H, 0]] z = (1/w) diag(At, Bt) z.
    Returns w (k,) ascending, a (m1, k), b (m2, k)."""
    m1, m2 = At.shape[0], Bt.shape[0]
    Z = torch.zeros((m1 + m2, m1 + m2), dtype=W.dtype, device=W.device)
    Z[:m1, m1:] = W
    Z[m1:, :m1] = W.conj().T
    Dm = torch.zeros_like(Z)
    Dm[:m1, :m1] = 0.5 * (At + At.conj().T)
    Dm[m1:, m1:] = 0.5 * (Bt + Bt.conj().T)
    Lc = torch.linalg.cholesky(Dm)
    Y = torch.linalg.solve_triangular(Lc, Z, upper=False)
    Y = torch.linalg.solve_triangular(Lc, Y.conj().T, upper=False).conj().T  # L^-1 Z L^-H
    Y = 0.5 * (Y + Y.conj().T)
    ev, Ev = torch.linalg.eigh(Y)
    zv = torch.linalg.solve_triangular(Lc.conj().T, Ev, upper=True)
    order = torch.argsort(ev, descending=True)[:k]
    inv_w = ev[order]
    if bool((inv_w <= 0).any()):
        raise RuntimeError(
            "reduced BdG problem has fewer positive modes than requested "
            "(the search space is too small or the state is unstable)"
        )
    zv = zv[:, order]
    return 1.0 / inv_w, zv[:m1], zv[m1:]


def _lr_lobpcg(apply_A, apply_B, prec_A, prec_B, X0, n_want, proj_f, proj_h, tol=1e-8, maxiter=1000, verbose=False):
    """Lowest n_want w of A f = w h, B h = w f (A, B Hermitian positive
    definite on the spaces selected by proj_f, proj_h), by a block LOBPCG-type
    iteration on the linear-response minimisation principle; block size
    X0.shape[0] >= n_want. Returns (w, F, H, residuals, iterations)."""
    k = X0.shape[0]
    shp = lambda t: t.reshape((-1,) + (1,) * (X0.ndim - 1))
    U, _, _ = _svqb(proj_f(X0))
    Vb, _, _ = _svqb(proj_h(X0))
    AU, BV = apply_A(U), apply_B(Vb)
    PF = PH = None
    nF_old = nH_old = 0
    it = 0

    def residuals(F, AF, H, BH, wr):
        RH = AF - H * shp(wr).to(F.dtype)  # A f - w h
        RF = BH - F * shp(wr).to(F.dtype)  # B h - w f
        scale = wr.abs() * torch.sqrt(_norms(F) ** 2 + _norms(H) ** 2)
        return RH, RF, torch.sqrt(_norms(RH) ** 2 + _norms(RF) ** 2) / scale.clamp_min(1e-300)

    while True:
        At, Bt, W = _gram(U, AU), _gram(Vb, BV), _gram(U, Vb)
        w, a, b = _reduced_lr(At, Bt, W, k)
        F, AF = _comb(U, a), _comb(AU, a)
        H, BH = _comb(Vb, b), _comb(BV, b)
        if it > 0:
            # previous-direction blocks: the part of the update outside the old block
            PF = _comb(U[nF_old:], a[nF_old:])
            PH = _comb(Vb[nH_old:], b[nH_old:])
        wr = w.real.to(F.real.dtype)
        RH, RF, res = residuals(F, AF, H, BH, wr)
        if verbose and (it % 25 == 0):
            print(
                f"    lr-lobpcg it {it}: max residual (wanted) {float(res[:n_want].max()):.2e}, w0 = {float(wr[0]):.6g}"
            )
        if float(res[:n_want].max()) < tol or it >= maxiter:
            # confirm with freshly applied operators (the carried images drift slowly)
            AF, BH = apply_A(F), apply_B(H)
            RH, RF, res = residuals(F, AF, H, BH, wr)
            if float(res[:n_want].max()) < tol or it >= maxiter:
                break
        active = res > tol
        dF = proj_f(prec_A(RH[active]))
        dH = proj_h(prec_B(RF[active]))
        # new bases: [current block | new and previous directions, orthonormalised
        # against it]. The operator images of the current block are carried
        # (its Gram matrix is well conditioned); those of the new directions are
        # recomputed: carrying them through a near-dependent orthonormalisation
        # amplifies the mismatch between X and its image, and it compounds.
        QF, AQF, cF = _svqb(F, AF)
        QH, BQH, cH = _svqb(H, BH)
        if cF > 1e2:
            AQF = apply_A(QF)
        if cH > 1e2:
            BQH = apply_B(QH)
        newF = torch.cat([dF] + ([PF] if PF is not None else []))
        newH = torch.cat([dH] + ([PH] if PH is not None else []))
        newF, _ = _orth_against(QF, None, newF, None)
        newH, _ = _orth_against(QH, None, newH, None)
        newF, _, _ = _svqb(newF)
        newH, _, _ = _svqb(newH)
        newF, newH = proj_f(newF), proj_h(newH)
        U, AU = torch.cat([QF, newF]), torch.cat([AQF, apply_A(newF)])
        Vb, BV = torch.cat([QH, newH]), torch.cat([BQH, apply_B(newH)])
        nF_old, nH_old = QF.shape[0], QH.shape[0]
        it += 1
    return wr, F, H, res, it


def _cg(apply_op, b, prec, tol=1e-12, maxiter=5000):
    """Preconditioned conjugate gradients for a Hermitian positive definite
    operator on a block of right-hand sides (columns solved independently).
    A column that has converged exactly (r = 0, so p^H A p = 0 and r^H z = 0)
    while another one is still iterating keeps its x: its step and its
    direction update are set to zero instead of 0/0."""
    shp = lambda t: t.reshape((-1,) + (1,) * (b.ndim - 1)).to(b.dtype)
    x = torch.zeros_like(b)
    r = b.clone()
    z = prec(r)
    p = z.clone()
    rz = _gram(r, z).diagonal().real
    nb = _norms(b).clamp_min(1e-300)
    zero = torch.zeros_like(rz)
    for it in range(maxiter):
        Ap = apply_op(p)
        pAp = _gram(p, Ap).diagonal().real
        alpha = torch.where(pAp != 0, rz / pAp, zero)
        x = x + shp(alpha) * p
        r = r - shp(alpha) * Ap
        if float((_norms(r) / nb).max()) < tol:
            return x, it + 1
        z = prec(r)
        rz_new = _gram(r, z).diagonal().real
        p = z + shp(torch.where(rz != 0, rz_new / rz, zero)) * p
        rz = rz_new
    warnings.warn(f"CG did not reach tol {tol:g} in {maxiter} iterations", stacklevel=2)
    return x, maxiter


# -- results ---------------------------------------------------------------------------


@dataclass
class BdGResult:
    """Modes at one Bloch q. omega (n,) ascending in E_R; u, v (n, C, *N): the
    periodic parts (u = e^{i q.r} u~), normalised to int sum_sigma
    (|u|^2 - |v|^2) dV = 1; f = u + v, h = u - v; residual (n,) relative
    residual of A f = w h, B h = w f."""

    q: tuple
    omega: torch.Tensor
    u: torch.Tensor
    v: torch.Tensor
    residual: torch.Tensor
    iterations: int
    method: str
    participation_ratio: torch.Tensor = None
    spin_weight: torch.Tensor = None
    depletion: torch.Tensor = None
    local_spin_weight: torch.Tensor = None
    norm_check: torch.Tensor = None
    zero_mode_omega: float | None = None  # dense method at q = 0: the projected-out eigenvalue
    info: dict = field(default_factory=dict)

    def table(self, n: int | None = None) -> str:
        n = len(self.omega) if n is None else min(n, len(self.omega))
        rows = [f"{'#':>3s} {'omega':>12s} {'PR':>7s} {'spin':>7s} {'spin_loc':>8s} {'depl':>10s} {'resid':>9s}"]
        for i in range(n):
            rows.append(
                f"{i:3d} {float(self.omega[i]):12.6f} {float(self.participation_ratio[i]):7.3f} "
                f"{float(self.spin_weight[i]):7.3f} {float(self.local_spin_weight[i]):8.3f} "
                f"{float(self.depletion[i]):10.3e} {float(self.residual[i]):9.1e}"
            )
        return "\n".join(rows)


# -- the operator --------------------------------------------------------------------------


class BdG:
    """The BdG operators about a real ground state psi0 (1, C, *N) of `engine`
    at Bloch quasi-momentum q (d,) (units 1/a; reduced to the box's first
    mini-zone, |q_i| <= pi/L_i). apply_L, apply_M, apply_A, apply_B act on
    blocks (m, C, *N) of periodic parts."""

    def __init__(self, engine: BilayerGPE, psi0: torch.Tensor, q=None, mu: float | None = None, imag_tol: float = 1e-6):
        g = engine.grid
        self.engine, self.grid, self.C, self.kappa = engine, g, engine.C, engine.kappa
        if engine.twist is not None and float(engine.twist.abs().max()) != 0.0:
            raise NotImplementedError(
                "BdG about a current-carrying (twisted) state needs the full "
                "non-Hermitian problem; relax psi0 with twist 0"
            )
        # engine.batch_hint looks at g, Omega_R, delta and the twist only; a batched potential (B, C, *N) would
        # broadcast the blocks wrongly, so it is checked here as well
        v_batch = 1 if engine.V is None else int(engine.V.shape[0])
        if engine.batch_hint != 1 or psi0.shape[0] != 1 or v_batch != 1:
            raise ValueError(
                "BdG works on one state: batch size 1, scalar parameters and one potential "
                f"(batch_hint {engine.batch_hint}, psi0 batch {psi0.shape[0]}, V batch {v_batch})"
            )
        self.device = g.device
        self.rdtype = torch.float64
        self.cdtype = torch.complex128

        # gauge fix: psi0 real up to a global phase
        p = psi0.detach().to(self.device, self.cdtype)
        s = (p * p).sum()
        phase = torch.angle(s) / 2.0 if abs(complex(s)) > 0 else torch.tensor(0.0)
        p = p * torch.exp(-1j * phase)
        if float(p.real.sum()) < 0:
            p = -p
        nrm = float(torch.sqrt((p.abs() ** 2).sum()))
        self.imag_fraction = float(torch.sqrt((p.imag**2).sum())) / max(nrm, 1e-300)
        if self.imag_fraction > imag_tol:
            warnings.warn(
                f"psi0 is not real up to a global phase: |Im psi0| / |psi0| = "
                f"{self.imag_fraction:.2e} (> {imag_tol:g}); the real-psi0 reduction "
                f"discards the imaginary part",
                stacklevel=2,
            )
        self.psi0 = p.real.contiguous()  # (1, C, *N) real
        self.n = (self.psi0**2).sum(dim=1, keepdim=True)
        self.N_atoms = float(g.integrate(self.n).sum())

        # local terms
        self.mean_field, self.gprime = _mean_field_and_kernel(engine, self.n)
        V = engine.V
        loc = torch.zeros((1, self.C) + g.N, dtype=self.rdtype, device=self.device)
        if V is not None:
            loc = loc + V.to(self.rdtype)
        if self.mean_field is not None:
            loc = loc + self.mean_field
        if self.C == 2:
            dl = float(engine.delta) if not torch.is_tensor(engine.delta) else float(engine.delta.reshape(-1)[0])
            loc = loc + torch.tensor([0.5 * dl, -0.5 * dl], dtype=self.rdtype, device=self.device).reshape(
                (1, 2) + (1,) * g.d
            )
            om = engine.Omega_R
            self.Omega_R = float(om) if not torch.is_tensor(om) else float(om.reshape(-1)[0])
        else:
            self.Omega_R = 0.0
        if mu is None and engine.dtype == torch.complex128:
            mu = float(engine.energy_terms(self.psi0.to(engine.dtype))["mu"][0])
        elif mu is None:
            # mu = <psi0| H_GP |psi0> / <psi0|psi0> from this operator's float64 blocks, so that it does not
            # inherit a single-precision engine's rounding of mu (its V and parameters are what they are)
            self.loc = loc
            mu = float((self.psi0 * self._D_gamma(self.psi0)).sum() / (self.psi0 * self.psi0).sum())
        self.mu = float(mu)
        self.loc = loc - self.mu
        self.gpsi0 = None if self.gprime is None else self.gprime * self.psi0

        # Bloch q, reduced to the first mini-zone
        q = [0.0] * g.d if q is None else [float(x) for x in q]
        if len(q) != g.d:
            raise ValueError(f"q must have {g.d} components")
        self.q = tuple(qi - (2 * math.pi / L) * round(qi * L / (2 * math.pi)) for qi, L in zip(q, g.L))
        self.at_gamma = all(abs(qi) < 1e-12 for qi in self.q)
        self.real = self.at_gamma  # q = 0: everything is real
        self.kin = self.kappa * self._ksq(self.q)  # (1, 1, *N)
        self.kin_r = self.kin[..., : g.N[-1] // 2 + 1].contiguous() if self.real else None
        self.block_dtype = self.rdtype if self.real else self.cdtype

        # GP residual of psi0 in this operator (B psi0 at q = 0)
        r = self._D_gamma(self.psi0)
        self.gp_residual = float(_norms(r)[0] / _norms(self.psi0)[0])
        self._zero = None

    # -- operators -----------------------------------------------------------------------
    def _kinetic(self, F):
        g = self.grid
        if torch.is_complex(F):
            return g.ifft(g.fft(F) * self.kin)
        if self.kin_r is None:  # a real block at q != 0: no real symmetry, go complex
            return g.ifft(g.fft(F) * self.kin)
        Fk = torch.fft.rfftn(F, dim=g.sdims)
        return torch.fft.irfftn(Fk * self.kin_r, s=g.N, dim=g.sdims)

    def _ksq(self, q):
        """|k + q|^2 as a (1, 1, *N) float64 tensor, from the grid's float64
        k vectors whatever the engine's precision."""
        g = self.grid
        ksq = None
        for ax in range(g.d):
            t = (g.axis_view(g.k64[ax].to(self.device), ax) + float(q[ax])) ** 2
            ksq = t if ksq is None else ksq + t
        return ksq

    def _D_gamma(self, F):
        """H_GP(q = 0) - mu on a real field: the zero-mode residual check."""
        g = self.grid
        kin0 = self.kappa * self._ksq([0.0] * g.d)
        out = g.ifft(g.fft(F.to(self.cdtype)) * kin0).real + self.loc * F
        if self.C == 2 and self.Omega_R != 0.0:
            out = out + self.Omega_R * F.flip(1)
        return out

    def apply_D(self, F: torch.Tensor) -> torch.Tensor:
        """D = H_GP(q) - mu at the frozen density (= B = L - M)."""
        out = self._kinetic(F) + self.loc.to(F.dtype) * F
        if self.C == 2 and self.Omega_R != 0.0:
            out = out + self.Omega_R * F.flip(1)
        return out

    def apply_X(self, F: torch.Tensor) -> torch.Tensor:
        """(X f)_sigma = g' psi0_sigma sum_sigma' psi0_sigma' f_sigma' (= M)."""
        if self.gpsi0 is None:
            return torch.zeros_like(F)
        return self.gpsi0.to(F.dtype) * (self.psi0.to(F.dtype) * F).sum(dim=1, keepdim=True)

    def apply_L(self, F):
        return self.apply_D(F) + self.apply_X(F)

    def apply_M(self, F):
        return self.apply_X(F)

    def apply_A(self, F):
        return self.apply_D(F) + 2.0 * self.apply_X(F)

    def apply_B(self, F):
        return self.apply_D(F)

    def preconditioner(self, shift: float):
        """F -> F(k) / (kappa |k + q|^2 + shift): the kinetic preconditioner."""
        g = self.grid
        inv = 1.0 / (self.kin + shift)
        inv_r = inv[..., : g.N[-1] // 2 + 1].contiguous() if self.real else None

        def apply(F):
            if torch.is_complex(F) or inv_r is None:
                return g.ifft(g.fft(F) * inv)
            return torch.fft.irfftn(torch.fft.rfftn(F, dim=g.sdims) * inv_r, s=g.N, dim=g.sdims)

        return apply

    def default_shift(self) -> float:
        """Preconditioner shift: half the spread of the local terms,
        (max |V + mu_int - mu| + Omega_R) / 2, at least 1 E_R (on the m = 4
        supercell shifts 3-6 converge in ~100 iterations, 1 or 11 in ~120,
        25 in ~180)."""
        return max(1.0, 0.5 * (float(self.loc.abs().max()) + abs(self.Omega_R)))

    # -- zero modes ------------------------------------------------------------------------
    def zero_modes(self):
        """At q = 0: (Z, P) with Z (z, C, *N) the Goldstone h-vectors (psi0, and
        (psi1, -psi2) when C = 2 and Omega_R = 0) and P the h-constraint vectors,
        P^H h = 0 for every mode with w != 0: P = A^{-1} Z. None at q != 0.
        The set is reduced to its independent directions (`_independent`): with one
        component empty, (psi1, -psi2) = -psi0 and there is one Goldstone mode, not two.
        When the spin rotation chi (`spin_kernel`) is in ker A: Z = independent
        (psi0, chi, (psi1, -psi2)) (the last one is in the span of the first two for
        a ground state) and P = (A^+ Z_i for Z_i != chi, chi), A^+ by CG on the
        complement of chi (operator and preconditioner projected)."""
        if not self.at_gamma:
            return None
        if self._zero is None:
            Z = [self.psi0]
            if self.C == 2 and self.Omega_R == 0.0:
                Z.append(
                    self.psi0
                    * torch.tensor([1.0, -1.0], dtype=self.rdtype, device=self.device).reshape(
                        (1, 2) + (1,) * self.grid.d
                    )
                )
            chi = self.spin_kernel()
            if chi is None:
                Z = _independent(torch.cat(Z)).to(self.block_dtype)
                P, _ = _cg(self.apply_A, Z, self.preconditioner(self.default_shift()))
            else:
                Z = _independent(torch.cat([Z[0], chi, Z[1]])).to(self.block_dtype)
                chi = chi.to(self.block_dtype)
                cn2 = float(_norms(chi)[0]) ** 2

                def perp(Y):  # remove the chi component
                    return Y - _comb(chi, _gram(chi, Y) / cn2)

                prec = self.preconditioner(self.default_shift())
                rest = [0] + list(range(2, Z.shape[0]))  # Z[1] is chi (kept: it is orthogonal to psi0)
                Pr, _ = _cg(lambda Y: perp(self.apply_A(perp(Y))), perp(Z[rest]), lambda Y: perp(prec(perp(Y))))
                Pr = perp(Pr)
                P = torch.cat([Pr[:1], chi, Pr[1:]])
            self._zero = (Z, P)
        return self._zero

    def spin_kernel(self):
        """chi = (-psi2, psi1), the generator of the spin rotation, if it is in the kernel
        of A (C = 2, Omega_R = 0, q = 0), else None. A chi = (-D_1 psi2, D_2 psi1) (X chi = 0
        identically): zero when both components see the same D (V1 = V2, Delta = 0) and psi0
        is stationary, and then |A chi| / |chi| is the GP residual of psi0. Detected as
        |A chi| / |chi| <= max(10 gp_residual, 1e-12 s) and <= 1e-6 s, s = `default_shift`
        (an asymmetry between the components gives |A chi| / |chi| ~ |V1 - V2|, Delta)."""
        if not (self.at_gamma and self.C == 2 and self.Omega_R == 0.0):
            return None
        chi = torch.cat([-self.psi0[:, 1:2], self.psi0[:, 0:1]], dim=1)
        rel = float(_norms(self.apply_A(chi))[0]) / float(_norms(chi)[0])
        s = self.default_shift()
        return chi if rel <= max(10.0 * self.gp_residual, 1e-12 * s) and rel <= 1e-6 * s else None

    def _projectors(self):
        zm = self.zero_modes()
        if zm is None:
            ident = lambda Y: Y
            return ident, ident
        Z, P = zm
        Gzz = _gram(Z, Z)
        Gpz = _gram(P, Z)

        def proj_f(Y):  # f orthogonal to every zero mode h-vector
            c = torch.linalg.solve(Gzz, _gram(Z, Y))
            return Y - _comb(Z, c)

        def proj_h(Y):  # h with P^H h = 0, removing the Z components
            c = torch.linalg.solve(Gpz, _gram(P, Y))
            return Y - _comb(Z, c)

        return proj_f, proj_h

    # -- solvers -------------------------------------------------------------------------
    def solve(
        self,
        n_modes: int = 8,
        method: str = "lobpcg",
        tol: float = 1e-8,
        maxiter: int = 2000,
        guard: int | None = None,
        shift: float | None = None,
        x0: torch.Tensor | None = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> BdGResult:
        """The n_modes lowest positive-norm modes. method: "lobpcg" (matrix-free,
        default) or "dense" (exact, small problems). guard: extra block vectors
        (default max(6, n_modes // 2)); x0: optional starting block (warm start)."""
        if method == "dense":
            w, F, H, res, its, w0 = self._solve_dense(n_modes)
        elif method == "lobpcg":
            guard = max(6, n_modes // 2) if guard is None else guard
            k = n_modes + guard
            shift = self.default_shift() if shift is None else shift
            prec = self.preconditioner(shift)
            gen = torch.Generator(device="cpu").manual_seed(int(seed))
            X0 = torch.randn((k, self.C) + self.grid.N, generator=gen, dtype=torch.float64)
            if not self.real:
                X0 = torch.complex(X0, torch.randn((k, self.C) + self.grid.N, generator=gen, dtype=torch.float64))
            X0 = prec(prec(X0.to(self.device)))  # smooth random start
            if x0 is not None:  # warm start: previous modes first
                m = min(k, x0.shape[0])
                x0 = x0[:m].to(self.device)
                X0[:m] = x0.real.to(self.rdtype) if self.real else x0.to(self.cdtype)
            proj_f, proj_h = self._projectors()
            w, F, H, res, its = _lr_lobpcg(
                self.apply_A,
                self.apply_B,
                prec,
                prec,
                X0,
                n_modes,
                proj_f,
                proj_h,
                tol=tol,
                maxiter=maxiter,
                verbose=verbose,
            )
            if float(res[:n_modes].max()) >= tol:
                warnings.warn(
                    f"BdG lobpcg at q = {self.q}: residual {float(res[:n_modes].max()):.1e} "
                    f"above tol {tol:g} after {its} iterations",
                    stacklevel=2,
                )
            w0 = None
        else:
            raise ValueError(f"method must be 'lobpcg' or 'dense', got {method!r}")
        return self._finish(w[:n_modes], F[:n_modes], H[:n_modes], res[:n_modes], its, method, w0)

    def _dense_matrix(self, op, nvec, dtype, chunk=512):
        cols = []
        shape = (self.C,) + self.grid.N
        for i in range(0, nvec, chunk):
            m = min(chunk, nvec - i)
            E = torch.zeros((m, nvec), dtype=dtype, device=self.device)
            E[torch.arange(m), torch.arange(i, i + m)] = 1.0
            cols.append(op(E.reshape((m,) + shape)).reshape(m, nvec))
        return torch.cat(cols).T  # column j = op(e_j)

    def _solve_dense(self, n_modes):
        nvec = self.C * self.grid.numel
        if nvec > 12000:
            raise ValueError(f"dense BdG on {nvec} unknowns: use method='lobpcg'")
        dt = self.block_dtype
        Am = self._dense_matrix(self.apply_A, nvec, dt)
        Bm = self._dense_matrix(self.apply_B, nvec, dt)
        Am = 0.5 * (Am + Am.conj().T)
        Bm = 0.5 * (Bm + Bm.conj().T)
        chi = self.spin_kernel() if self.at_gamma else None
        if chi is not None:
            # chi in ker A and ker B: lift it in A (A + s chi chi^H / |chi|^2). Every mode with w != 0 has
            # f orthogonal to chi, so A f is unchanged; chi itself becomes a w = 0 mode, dropped below
            # with the Goldstone pair (Z holds both)
            cv = chi.reshape(-1, 1).to(dt) / float(_norms(chi)[0])
            Am = Am + self.default_shift() * (cv @ cv.conj().T)
        R = torch.linalg.cholesky(Am).conj().T  # A = R^H R
        S = R @ Bm @ R.conj().T
        lam, Y = torch.linalg.eigh(0.5 * (S + S.conj().T))
        Fall = torch.linalg.solve_triangular(R, Y, upper=True)  # f = R^-1 y
        w0 = None
        idx = torch.arange(nvec, device=self.device)
        if self.at_gamma:
            Z, _ = self.zero_modes()
            Zf = Z.reshape(Z.shape[0], -1).T  # (nvec, z)
            ov = (Zf.conj().T @ Fall).abs().sum(dim=0) / Fall.abs().pow(2).sum(dim=0).sqrt()
            drop = torch.topk(ov, Z.shape[0]).indices
            w0 = float(torch.sign(lam[drop[0]]) * torch.sqrt(lam[drop[0]].abs()))
            keep = torch.ones(nvec, dtype=torch.bool, device=self.device)
            keep[drop] = False
            idx = idx[keep]
        idx = idx[:n_modes]
        lam_k = lam[idx]
        if bool((lam_k <= 0).any()):
            raise RuntimeError(
                f"non-positive w^2 in the BdG spectrum: {lam_k[lam_k <= 0].tolist()} "
                f"(dynamically unstable or not a ground state)"
            )
        w = torch.sqrt(lam_k)
        F = Fall[:, idx]
        H = (Am @ F) / w.to(F.dtype)
        res = torch.linalg.norm(Bm @ H - F * w.to(F.dtype), dim=0) / (
            w * torch.sqrt(torch.linalg.norm(F, dim=0) ** 2 + torch.linalg.norm(H, dim=0) ** 2)
        )
        shape = (self.C,) + self.grid.N
        F = F.T.reshape((-1,) + shape)
        H = H.T.reshape((-1,) + shape)
        return w, F, H, res, 0, w0

    def _finish(self, w, F, H, res, its, method, w0):
        g = self.grid
        F = F.to(self.cdtype)
        H = H.to(self.cdtype)
        sd = tuple(range(-g.d, 0))
        # normalise Re int f^* h dV = 1  (= int sum (|u|^2 - |v|^2))
        fh = (F.conj() * H).real.sum(dim=(1,) + sd) * g.dV
        if bool((fh <= 0).any()):
            warnings.warn("a mode has non-positive norm int(|u|^2 - |v|^2)", stacklevel=2)
        s = 1.0 / torch.sqrt(fh.abs())
        F = F * s.reshape((-1,) + (1,) * (F.ndim - 1))
        H = H * s.reshape((-1,) + (1,) * (H.ndim - 1))
        u, v = 0.5 * (F + H), 0.5 * (F - H)
        norm_check = (u.abs() ** 2 - v.abs() ** 2).sum(dim=(1,) + sd) * g.dV
        out = BdGResult(
            q=self.q,
            omega=w.real.to(self.rdtype),
            u=u,
            v=v,
            residual=res.real.to(self.rdtype),
            iterations=its,
            method=method,
            norm_check=norm_check,
            zero_mode_omega=w0,
            info=dict(gp_residual=self.gp_residual, mu=self.mu),
        )
        (out.participation_ratio, out.spin_weight, out.depletion, out.local_spin_weight) = self.mode_observables(u, v)
        return out

    # -- mode observables --------------------------------------------------------------------
    def mode_observables(self, u: torch.Tensor, v: torch.Tensor):
        """Per mode (each (m,)):

        * participation ratio of rho = sum_sigma |u|^2 + |v|^2:
          (int rho)^2 / (V int rho^2); within a
          degenerate multiplet it depends on the solver's basis -- compare
          multiplet averages;
        * spin weight int |u1 - s u2|^2 + |v1 - s v2|^2  /  (2 int sum |u|^2 + |v|^2)
          (the ratio used for the 1D toy model's BdG spectrum), s = sign(int psi1 psi2)
          the condensate's relative sign (-1 for this engine's +Omega_R sigma_x
          lower dressed state; +1 in the toy's -Omega/2 sigma_x convention).
          In [0, 1]; with equal layer densities 0 for a density mode, 1 for a
          pure spin-flip mode, 1/2 for a mode in one layer only; spin-like above
          0.5. With unequal layer densities it mixes the two (a density mode has
          u1 != s u2 and a spin weight above 0, a spin mode one below 1). 0 for
          C = 1;
        * local spin weight: the fraction carried by the component orthogonal
          to the LOCAL condensate spinor, chi_perp = (-psi0_2, psi0_1)/|psi0|
          (the locally density-neutral part): 0 for a density mode and 1 for a
          spin mode whatever the layer densities; equal to the spin weight where
          the densities coincide;
        * depletion int sum |v|^2 dV.
        Returns (pr, spin, depletion, local_spin)."""
        g = self.grid
        sd = tuple(range(-g.d, 0))
        rho = (u.abs() ** 2 + v.abs() ** 2).sum(dim=1)  # (m, *N)
        tot = rho.sum(dim=sd) * g.dV
        pr = tot**2 / (g.volume * (rho**2).sum(dim=sd) * g.dV)
        if self.C == 2:
            s = -1.0 if float((self.psi0[:, 0] * self.psi0[:, 1]).sum()) < 0 else 1.0
            spin = (((u[:, 0] - s * u[:, 1]).abs() ** 2 + (v[:, 0] - s * v[:, 1]).abs() ** 2).sum(dim=sd) * g.dV) / (
                2.0 * tot
            )
            amp = torch.sqrt(self.n).clamp_min(1e-300)
            c1, c2 = self.psi0[:, 0] / amp[:, 0], self.psi0[:, 1] / amp[:, 0]  # (1, *N)
            su = -c2 * u[:, 0] + c1 * u[:, 1]
            sv = -c2 * v[:, 0] + c1 * v[:, 1]
            local = ((su.abs() ** 2 + sv.abs() ** 2).sum(dim=sd) * g.dV) / tot
        else:
            spin = torch.zeros_like(tot)
            local = torch.zeros_like(tot)
        depl = (v.abs() ** 2).sum(dim=(1,) + sd) * g.dV
        return pr, spin, depl, local


def _mean_field_and_kernel(engine: BilayerGPE, n: torch.Tensor):
    """(mu_int(n), d mu_int/d n) as real (1, 1, *N) tensors, or (None, None)
    without interactions (the functional's mu_int and dmu_int, e.g. npse.NPSE2D)."""
    f = engine.interaction
    if f is not None:
        return f.mu_int(n).to(torch.float64), f.dmu_int(n).to(torch.float64)
    gval = engine.g
    if torch.is_tensor(gval):
        if gval.numel() != 1:
            raise ValueError("BdG needs a scalar g")
        gval = float(gval.reshape(-1)[0])
    if gval == 0.0:
        return None, None
    return gval * n, torch.full_like(n, gval)


# -- drivers -----------------------------------------------------------------------------


def bdg_bands(
    engine: BilayerGPE,
    psi0: torch.Tensor,
    q_list,
    n_modes: int = 8,
    method: str = "lobpcg",
    warm_start: bool = True,
    verbose: bool = False,
    **solve_kw,
) -> dict:
    """BdG modes along a list of Bloch momenta. Returns numpy arrays
    energies, participation, spin_weight, local_spin_weight, depletion,
    residual (n_q, n_modes),
    iterations (n_q,), q (n_q, d), the GP residual of psi0, and the BdGResult
    per q under "results". warm_start: start each q from the previous modes
    (both points must be off Gamma, where the zero-mode constraint differs)."""
    out = {
        k: []
        for k in (
            "energies",
            "participation",
            "spin_weight",
            "local_spin_weight",
            "depletion",
            "residual",
            "iterations",
            "q",
        )
    }
    results = []
    prev = None
    for q in q_list:
        bd = BdG(engine, psi0, q=q)
        x0 = None
        if warm_start and prev is not None and not bd.at_gamma and not prev[1]:
            x0 = prev[0]
        r = bd.solve(n_modes, method=method, x0=x0, verbose=verbose, **solve_kw)
        prev = ((r.u + r.v), bd.at_gamma)
        results.append(r)
        out["energies"].append(r.omega.cpu().numpy())
        out["participation"].append(r.participation_ratio.cpu().numpy())
        out["spin_weight"].append(r.spin_weight.cpu().numpy())
        out["local_spin_weight"].append(r.local_spin_weight.cpu().numpy())
        out["depletion"].append(r.depletion.cpu().numpy())
        out["residual"].append(r.residual.cpu().numpy())
        out["iterations"].append(r.iterations)
        out["q"].append(np.array(bd.q))
        if verbose:
            print(
                f"  q = {tuple(round(x, 5) for x in bd.q)}: {r.iterations} its, "
                f"w = {np.round(r.omega.cpu().numpy()[:4], 6)}"
            )
    res = {k: np.array(v) for k, v in out.items()}
    res["gp_residual"] = results[0].info["gp_residual"] if results else None
    res["results"] = results
    return res


def sound_speed(
    engine: BilayerGPE,
    psi0: torch.Tensor,
    L: float | None = None,
    n_points: int = 3,
    axis: int = 0,
    n_modes: int = 4,
    spin_max: float = 0.5,
    min_ratio: float = 10.0,
    method: str = "lobpcg",
    **solve_kw,
) -> dict:
    """Sound speed along `axis` from the lowest density-like BdG branch.

    The branch is sampled at the Bloch momenta q_j = 2 pi j / L, j = 1..n_points:
    the lowest phonons of a system of length L made of L / L_box copies of the
    box (so L must be >= 2 n_points L_box: q inside the box's mini-zone; the
    default is 4 n_points L_box). At each q the mode taken is the lowest one
    with spin weight < spin_max (the density branch). Points with
    w < min_ratio x (GP residual of psi0) are skipped (the smallest-q phonon
    is depressed by the ground-state error). The fit is
    w/q = c + b q^2 (least squares; c = w/q for a single point), which removes
    the leading lattice/Bogoliubov curvature of the branch.

    Returns dict(c, b, q, omega, used, spin_weight, gp_residual)."""
    g = engine.grid
    L_box = g.L[axis]
    if L is None:
        L = 4.0 * n_points * L_box
    qs = [2.0 * math.pi * j / L for j in range(1, n_points + 1)]
    if qs[-1] > math.pi / L_box * (1 + 1e-12):
        raise ValueError(
            f"q = 2 pi n_points / L = {qs[-1]:.4g} lies outside the box mini-zone "
            f"(pi / L_box = {math.pi / L_box:.4g}): take L >= 2 n_points L_box, "
            f"or a smaller box (one primitive cell) for the same q"
        )
    ws, sws = [], []
    gp_res = None
    prev = None
    for qv in qs:
        q = [0.0] * g.d
        q[axis] = qv
        bd = BdG(engine, psi0, q=q)
        gp_res = bd.gp_residual
        r = bd.solve(n_modes, method=method, x0=prev, **solve_kw)
        prev = r.u + r.v
        sel = torch.nonzero(r.spin_weight < spin_max).reshape(-1)
        if len(sel) == 0:
            raise RuntimeError(f"no density-like mode among the lowest {n_modes} at q = {qv:.4g}")
        i = int(sel[0])
        ws.append(float(r.omega[i]))
        sws.append(float(r.spin_weight[i]))
    q_arr, w_arr = np.array(qs), np.array(ws)
    used = w_arr > min_ratio * gp_res
    if used.sum() == 0:
        raise RuntimeError("every sampled phonon is below min_ratio x the ground-state residual")
    qu, wu = q_arr[used], w_arr[used]
    if len(qu) == 1:
        c, b = wu[0] / qu[0], 0.0
    else:
        Mfit = np.stack([np.ones_like(qu), qu**2], axis=1)
        (c, b), *_ = np.linalg.lstsq(Mfit, wu / qu, rcond=None)
    return dict(
        c=float(c), b=float(b), q=q_arr, omega=w_arr, used=used, spin_weight=np.array(sws), gp_residual=gp_res, L=L
    )
