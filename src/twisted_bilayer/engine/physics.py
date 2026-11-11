"""The two-component split-step Fourier engine.

    i d/dt psi_1 = [ -kappa nabla^2 + V_1 + Delta/2 + g n ] psi_1 + Omega_R psi_2
    i d/dt psi_2 = [ -kappa nabla^2 + V_2 - Delta/2 + g n ] psi_2 + Omega_R psi_1

with n = |psi_1|^2 + |psi_2|^2 (SU(2)-symmetric interaction), kappa = 1/pi^2
in lattice units (hbar = 1). This is Eq. (10)-(11) of the Meng et al.
Supplement with their sign and coupling conventions: the off-diagonal matrix
element is Omega_R itself (so a bare two-level system oscillates as
sin^2(Omega_R t)), and the detuning enters as +Delta/2 on component 1.

Field layout: (B, C, *N) -- batch, component (C = 1 or 2), then d spatial
axes. Omega_R, Delta, g and the twist may be per-batch tensors, so one call
propagates a whole parameter sweep.

Split-step composition is Strang (order 2) or Yoshida (order 4). The local step is EXACT for the
two-component problem: at a point, the local Hamiltonian is c I + b_z sigma_z + b_x sigma_x with
c = (V_1 + V_2)/2 + g n, b_z = (V_1 - V_2 + Delta)/2, b_x = Omega_R, and
n is conserved by that 2x2 unitary (it only moves atoms between components),
so exp(-i dt H_loc) with n frozen is the exact solution of the local ODE.
That is why the SU(2)-symmetric g is more than a convenience here.

Twisted boundaries psi(r + L e_i) = e^{i theta_i} psi(r) are implemented as
psi = e^{i q.r} phi with phi periodic, q_i = theta_i / L_i: the engine works
on phi and shifts every k by q (kinetic operator, energy, H phi, current).
"""

from __future__ import annotations

import math

import torch

from .backend import REAL_OF
from .grid import Grid
from .units import KAPPA

# Yoshida (1990): S4(dt) = S2(theta dt) S2((1 - 2 theta) dt) S2(theta dt)
_YOSHIDA_THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_YOSHIDA_MID = 1.0 - 2.0 * _YOSHIDA_THETA


class BilayerGPE:
    """The two-component (or one-component) GPE on a periodic grid, batched over parameters.

    Units are the lattice units of `engine.units`: lengths in the lattice constant a, energies in the recoil
    energy E_R, hbar = 1, so time is in hbar / E_R and the mass enters only through kappa = 1/pi^2. Fields
    psi have shape (B, C, *N): batch member, component, then the grid axes; |psi|^2 is atoms per a^d (d = the
    grid's dimension), so `norm` integrates to the atom number.

    Parameters (each scalar, or a (B,) sequence / tensor batched over the members; the batched ones must all
    have length B or 1):
        grid: the periodic `Grid` (box lengths in a; its device sets the engine's). Read-only afterwards.
        V: the potential in E_R: (*N), (1 or B, 1, *N) shared by both components (a trap, one lattice), or
            (1 or B, C, *N) per component (the two layers' lattices). None: no potential. Finite everywhere.
        g: contact coupling in E_R a^d per atom (the mean field is g n, n = |psi_1|^2 + |psi_2|^2: SU(2)
            symmetric). In 2D the quasi-2D g_2D of `engine.units`. Must be >= 0 unless allow_attractive.
        Omega_R: the interlayer (Rabi) coupling in E_R, the off-diagonal element of the local 2 x 2
            Hamiltonian (+Omega_R sigma_x). Its sign selects which dressed branch is lower: see
            `ground_state.initial_state`.
        delta: detuning in E_R, entering as +delta/2 on component 1 and -delta/2 on component 2.
        kappa: kinetic prefactor in E_R a^2 (H_kin = -kappa nabla^2); 1/pi^2 in lattice units. Finite, > 0;
            read-only afterwards.
        n_components: C, 1 or 2 (Omega_R and delta act only for 2).
        twist: phase twists theta_i in rad across the box, psi(r + L_i e_i) = e^{i theta_i} psi(r): None, a
            (d,) sequence, or a (B, d) tensor; changed later with `set_twist`.
        dtype: torch.complex128 (default) or torch.complex64; real parts use the matching real dtype.
        interaction: optional density functional replacing g n (e.g. `engine.npse.NPSE2D`, the quasi-2D
            NPSE); then g is ignored in H and E (pass g = 0). Not batchable.
        allow_attractive: accept g < 0. Default False: an attractive g is refused as a likely sign error.

    Construction raises ValueError for a non-finite input, kappa <= 0, g < 0 without allow_attractive,
    batched inputs of different lengths, or a non-complex dtype. No physics range is imposed.

    Main methods: `run_block` (real time, order 2 or 4), `imaginary_step`, `energy_terms` (E_kin, E_pot, E_int,
    E_delta, E_rabi, E_total in E_R and mu in E_R per member), `apply_H`, `residual` (||H psi - mu psi|| /
    ||psi|| in E_R, the ground-state criterion), `current`, `norm`, `to_dtype`. Ground states:
    `engine.ground_state.relax`."""

    def __init__(
        self,
        grid: Grid,
        V: torch.Tensor | None = None,
        g=0.0,
        Omega_R=0.0,
        delta=0.0,
        kappa: float = KAPPA,
        n_components: int = 2,
        twist=None,
        dtype: torch.dtype = torch.complex128,
        interaction=None,
        allow_attractive: bool = False,
    ):
        if n_components not in (1, 2):
            raise ValueError("n_components must be 1 or 2")
        if dtype not in REAL_OF:
            raise ValueError(f"dtype must be a complex torch dtype ({', '.join(map(str, REAL_OF))}), got {dtype!r}")
        kappa = float(kappa)
        if not (math.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"kappa must be finite and > 0, got {kappa!r}")
        self._grid = grid
        self.C = n_components
        self._kappa = kappa
        self.dtype = dtype
        self.rdtype = REAL_OF[dtype]
        self.device = grid.device
        self.V = None if V is None else self._as_potential(V)
        self.g = self._param(g)
        # optional density functional replacing g n (engine.npse.NPSE2D):
        # mu_int(n) in H, e_int(n) in E. Not batchable in g.
        self.interaction = interaction
        self.Omega_R = self._param(Omega_R)
        self.delta = self._param(delta)
        self.set_twist(twist)
        self._ops_cache: dict = {}
        self._ksq = None
        self._check_inputs(allow_attractive)

    # -- read-only geometry ---------------------------------------------------
    @property
    def grid(self) -> Grid:
        """The spatial grid (read-only: the cached kinetic operators depend on it; build a new engine, or use
        `to_dtype`, for another grid)."""
        return self._grid

    @property
    def kappa(self) -> float:
        """The kinetic prefactor kappa in H = -kappa nabla^2 (1/pi^2 in lattice units, E_R a^2 / pi^2;
        read-only: the cached kinetic operators depend on it)."""
        return self._kappa

    def _check_inputs(self, allow_attractive: bool) -> None:
        """Construction checks: every input finite, g >= 0 unless allow_attractive, the batched
        parameters of one length (or 1), the potential's leading axis 1 or that length. No physics range
        limits."""

        def finite(name, x):
            ok = bool(torch.isfinite(x).all()) if torch.is_tensor(x) else math.isfinite(float(x))
            if not ok:
                raise ValueError(f"{name} must be finite everywhere")

        if self.V is not None:
            finite("V", self.V)
        for name in ("g", "Omega_R", "delta"):
            finite(name, getattr(self, name))
        if self.twist is not None:
            finite("twist", self.twist)
        g = self.g
        g_min = float(g.min()) if torch.is_tensor(g) else float(g)
        if g_min < 0.0 and not allow_attractive:
            raise ValueError(f"g = {g_min!r} < 0 (attraction): pass allow_attractive=True to run it")
        lengths = {}
        for name in ("g", "Omega_R", "delta"):
            p = getattr(self, name)
            if torch.is_tensor(p) and p.shape[0] > 1:
                lengths[name] = p.shape[0]
        if self.twist is not None and self.twist.shape[0] > 1:
            lengths["twist"] = self.twist.shape[0]
        if self.V is not None and self.V.shape[0] > 1:
            lengths["V"] = self.V.shape[0]
        if len(set(lengths.values())) > 1:
            raise ValueError(f"batched inputs differ in length: {lengths} (each must be 1 or the batch size)")

    # -- parameters -----------------------------------------------------------
    def _as_potential(self, V: torch.Tensor) -> torch.Tensor:
        """A potential as a real (1 or B, C, *N) tensor: a component-less
        (1, 1, *N) potential (a trap, a single lattice) is shared by both
        components; a (…, *N) tensor without leading axes gets them."""
        V = torch.as_tensor(V).to(self.device, self.rdtype)
        while V.ndim < self.grid.d + 2:
            V = V.unsqueeze(0)
        if V.shape[1] == 1 and self.C == 2:
            V = V.expand(V.shape[0], 2, *V.shape[2:])
        if V.shape[1] != self.C:
            raise ValueError(f"potential has {V.shape[1]} components, engine has {self.C}")
        return V

    def _param(self, x):
        """Scalar -> float; (B,) tensor/sequence -> (B, 1, *1) real tensor."""
        if torch.is_tensor(x):
            if x.ndim == 0:
                return float(x)
            return x.to(self.device, self.rdtype).reshape((-1, 1) + (1,) * self.grid.d)
        if isinstance(x, (list, tuple)):
            return self._param(torch.tensor(x, dtype=torch.float64))
        return float(x)

    @staticmethod
    def _is_zero(p) -> bool:
        return (not torch.is_tensor(p)) and p == 0.0

    def _mu_int(self, n: torch.Tensor):
        """The local mean-field term: g n, or mu_int(n) of the interaction
        functional; None when there is no interaction."""
        if self.interaction is not None:
            return self.interaction.mu_int(n)
        if self._is_zero(self.g):
            return None
        return n * self.g

    def _e_int(self, n: torch.Tensor):
        if self.interaction is not None:
            return self.interaction.e_int(n)
        if self._is_zero(self.g):
            return None
        return 0.5 * self.g * n * n

    def set_twist(self, twist) -> None:
        """twist: None, (d,) sequence, or (B, d) tensor of phase twists theta_i."""
        self._ops_cache = {}
        self._ksq = None
        if twist is None:
            self.twist = None
            self.q = [None] * self.grid.d
            return
        t = torch.as_tensor(twist, dtype=torch.float64)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if t.shape[1] != self.grid.d:
            raise ValueError(f"twist must have {self.grid.d} components, got {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("twist must be finite")
        self.twist = t
        self.q = []
        for ax in range(self.grid.d):
            qa = t[:, ax] / self.grid.L[ax]
            if t.shape[0] == 1:
                self.q.append(None if float(qa) == 0.0 else float(qa))
            else:
                self.q.append(qa.to(self.device, self.rdtype).reshape((-1, 1) + (1,) * self.grid.d))

    @property
    def batch_hint(self) -> int:
        """Largest leading dimension among the batched parameters (1 if none)."""
        b = 1
        for p in (self.g, self.Omega_R, self.delta, *self.q):
            if torch.is_tensor(p):
                b = max(b, p.shape[0])
        return b

    def to_dtype(self, dtype: torch.dtype) -> BilayerGPE:
        """A copy of this engine in another complex precision."""
        eng = BilayerGPE.__new__(BilayerGPE)
        eng.__dict__.update(self.__dict__)
        eng.dtype, eng.rdtype = dtype, REAL_OF[dtype]
        eng._grid = self.grid.to(eng.rdtype)
        eng.V = None if self.V is None else self.V.to(eng.rdtype)
        for name in ("g", "Omega_R", "delta"):
            p = getattr(self, name)
            setattr(eng, name, p.to(eng.rdtype) if torch.is_tensor(p) else p)
        eng._ops_cache, eng._ksq = {}, None
        eng.set_twist(self.twist)
        return eng

    # -- k-space pieces -------------------------------------------------------
    def k_axis(self, ax: int) -> torch.Tensor:
        return self.grid.k_shifted(ax, self.q[ax])

    def ksq(self) -> torch.Tensor:
        """sum_i (k_i + q_i)^2, broadcastable (B or 1, 1, *N) -- dense. Cached."""
        if self._ksq is None:
            s = None
            for ax in range(self.grid.d):
                t = self.k_axis(ax) ** 2
                s = t if s is None else s + t
            self._ksq = s
        return self._ksq

    def _kin_ops(self, tau: float, imaginary: bool = False) -> torch.Tensor:
        """exp(-i tau kappa K^2) (or exp(-tau kappa K^2) in imaginary time) as one
        dense broadcastable tensor, cached per tau."""
        key = (float(tau), imaginary)
        ops = self._ops_cache.get(key)
        if ops is None:
            coef = (-self.kappa * tau) if imaginary else (-1j * self.kappa * tau)
            ops = torch.exp(coef * self.ksq().to(self.dtype))
            self._ops_cache[key] = ops
        return ops

    def _kin_step(self, psi: torch.Tensor, ops: torch.Tensor) -> torch.Tensor:
        return self.grid.ifft(self.grid.fft(psi) * ops)

    # -- local step ------------------------------------------------------------
    def density(self, psi: torch.Tensor) -> torch.Tensor:
        """Total density n = sum_sigma |psi_sigma|^2, shape (B, 1, *N)."""
        return (psi.real**2 + psi.imag**2).sum(dim=1, keepdim=True)

    def _kick(self, psi: torch.Tensor, dt: float, imaginary: bool = False) -> torch.Tensor:
        """psi <- exp(-i dt H_loc) psi (or exp(-dt H_loc)) pointwise; exact 2x2."""
        n = self.density(psi)
        V = self.V
        gn = self._mu_int(n)
        if self.C == 1:
            c = gn
            if V is not None:
                c = V if c is None else c + V
            if c is None:
                return psi
            if imaginary:
                return psi * torch.exp(-dt * c)
            return psi * torch.exp((-1j * dt) * c.to(self.dtype))

        # two components
        if V is not None:
            V1, V2 = V[:, 0:1], V[:, 1:2]
            c = 0.5 * (V1 + V2)
            bz = 0.5 * (V1 - V2)
        else:
            c, bz = None, None
        if gn is not None:
            c = gn if c is None else c + gn
        if not self._is_zero(self.delta):
            bz = 0.5 * self.delta if bz is None else bz + 0.5 * self.delta
        bx = self.Omega_R
        p1, p2 = psi[:, 0:1], psi[:, 1:2]

        if bz is None and self._is_zero(bx):
            if c is None:
                return psi
            ph = torch.exp(-dt * c) if imaginary else torch.exp((-1j * dt) * c.to(self.dtype))
            return psi * ph

        # b = sqrt(bz^2 + bx^2) as a real tensor broadcastable to (B,1,*N)
        b2 = None
        if bz is not None:
            b2 = bz * bz
        if not self._is_zero(bx):
            b2 = bx * bx if b2 is None else b2 + bx * bx
        if not torch.is_tensor(b2):
            b2 = torch.full((1, 1) + (1,) * self.grid.d, float(b2), dtype=self.rdtype, device=self.device)
        b = torch.sqrt(b2)
        tiny = torch.finfo(b.dtype).tiny
        if imaginary:
            cb = torch.cosh(dt * b)
            sb = torch.where(b > 0, torch.sinh(dt * b) / b.clamp_min(tiny), torch.full_like(b, dt))
            pref = torch.exp(-dt * c) if c is not None else None
            # exp(-dt (bz sz + bx sx)) = cosh(dt b) I - sinh(dt b)/b (bz sz + bx sx)
            a11 = cb - (sb * bz if bz is not None else 0.0)
            a22 = cb + (sb * bz if bz is not None else 0.0)
            a12 = -(sb * bx) if not self._is_zero(bx) else None
            a11, a22 = a11.to(self.dtype), a22.to(self.dtype)
            q1 = a11 * p1
            q2 = a22 * p2
            if a12 is not None:
                a12 = a12.to(self.dtype)
                q1 = q1 + a12 * p2
                q2 = q2 + a12 * p1
            out = torch.cat([q1, q2], dim=1)
            if pref is not None:
                out = out * pref
            return out
        # real time: exp(-i dt (bz sz + bx sx)) = cos(dt b) I - i sin(dt b)/b (bz sz + bx sx)
        cb = torch.cos(dt * b)
        sb = torch.where(b > 0, torch.sin(dt * b) / b.clamp_min(tiny), torch.full_like(b, dt))
        a11 = cb.to(self.dtype) - (1j * (sb * bz).to(self.dtype) if bz is not None else 0.0)
        a22 = cb.to(self.dtype) + (1j * (sb * bz).to(self.dtype) if bz is not None else 0.0)
        q1 = a11 * p1
        q2 = a22 * p2
        if not self._is_zero(bx):
            a12 = -1j * (sb * bx).to(self.dtype)
            q1 = q1 + a12 * p2
            q2 = q2 + a12 * p1
        out = torch.cat([q1, q2], dim=1)
        if c is not None:
            out = out * torch.exp((-1j * dt) * c.to(self.dtype))
        return out

    # -- real-time propagation --------------------------------------------------
    def run_block(self, psi: torch.Tensor, n_steps: int, dt: float, order: int = 2) -> torch.Tensor:
        """Advance psi by n_steps real-time steps of dt (hbar / E_R) with the Strang (order 2) or Yoshida
        (order 4) split step. Returns the new field (psi is not modified in place, but is consumed: do not
        reuse the input). `choose_dt` gives an accuracy-safe dt."""
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        if n_steps == 0:
            return psi
        if order == 2:
            return self._run_block_2nd(psi, n_steps, dt)
        if order == 4:
            return self._run_block_4th(psi, n_steps, dt)
        raise ValueError(f"order must be 2 or 4, got {order!r}")

    def _run_block_2nd(self, psi, n_steps, dt):
        kin_half = self._kin_ops(0.5 * dt)
        kin_full = self._kin_ops(dt)
        psi = self._kin_step(psi, kin_half)
        for _ in range(n_steps - 1):
            psi = self._kin_step(self._kick(psi, dt), kin_full)
        psi = self._kin_step(self._kick(psi, dt), kin_half)
        return psi

    def _run_block_4th(self, psi, n_steps, dt):
        th = _YOSHIDA_THETA
        dt_edge = th * dt
        dt_mid = _YOSHIDA_MID * dt
        kin_edge = self._kin_ops(0.5 * th * dt)
        kin_mid = self._kin_ops(0.5 * (1.0 - th) * dt)
        kin_edge_full = self._kin_ops(th * dt)
        psi = self._kin_step(psi, kin_edge)
        for step in range(n_steps):
            psi = self._kin_step(self._kick(psi, dt_edge), kin_mid)
            psi = self._kin_step(self._kick(psi, dt_mid), kin_mid)
            psi = self._kick(psi, dt_edge)
            psi = self._kin_step(psi, kin_edge if step == n_steps - 1 else kin_edge_full)
        return psi

    def imaginary_step(self, psi: torch.Tensor, dt: float, kin_half=None) -> torch.Tensor:
        """One Strang step in imaginary time (no renormalisation)."""
        if kin_half is None:
            kin_half = self._kin_ops(0.5 * dt, imaginary=True)
        psi = self._kin_step(psi, kin_half)
        psi = self._kick(psi, dt, imaginary=True)
        return self._kin_step(psi, kin_half)

    # -- functionals ---------------------------------------------------------------
    def norm(self, psi: torch.Tensor) -> torch.Tensor:
        """Total atom number per batch member, shape (B,)."""
        return self.grid.integrate(self.density(psi)).reshape(-1)

    def component_norms(self, psi: torch.Tensor) -> torch.Tensor:
        """(B, C) atom numbers per component."""
        return self.grid.integrate(psi.real**2 + psi.imag**2)

    def kinetic_energy(self, psi: torch.Tensor) -> torch.Tensor:
        """kappa sum_sigma int |(nabla + i q) psi_sigma|^2 by Parseval, (B,)."""
        psi_k = self.grid.fft(psi)
        dens_k = (psi_k.real**2 + psi_k.imag**2) * self.ksq()
        return self.kappa * (self.grid.dV / self.grid.numel) * dens_k.sum(dim=(1,) + self.grid.sdims)

    def energy_terms(self, psi: torch.Tensor) -> dict:
        """Every term of E[psi] per batch member, each shape (B,)."""
        n = self.density(psi)
        g = self.grid
        out = {"norm": g.integrate(n).reshape(-1), "E_kin": self.kinetic_energy(psi)}
        V = self.V
        if V is not None:
            out["E_pot"] = g.integrate((V * (psi.real**2 + psi.imag**2)).sum(dim=1)).reshape(-1)
        else:
            out["E_pot"] = torch.zeros_like(out["norm"])
        e_int = self._e_int(n)
        mu_int = self._mu_int(n)
        if e_int is None:
            out["E_int"] = torch.zeros_like(out["norm"])
            E_mu_int = torch.zeros_like(out["norm"])
        else:
            out["E_int"] = g.integrate(e_int).reshape(-1)
            E_mu_int = g.integrate(n * mu_int).reshape(-1)  # <mean field> = 2 E_int for g n^2/2
        if self.C == 2:
            d1 = psi[:, 0].real ** 2 + psi[:, 0].imag ** 2
            d2 = psi[:, 1].real ** 2 + psi[:, 1].imag ** 2
            if self._is_zero(self.delta):
                out["E_delta"] = torch.zeros_like(out["norm"])
            else:
                dl = self.delta.reshape(-1) if torch.is_tensor(self.delta) else self.delta
                out["E_delta"] = 0.5 * dl * g.integrate(d1 - d2)
            bx = self.Omega_R
            if self._is_zero(bx):
                out["E_rabi"] = torch.zeros_like(out["norm"])
            else:
                cross = (psi[:, 0].conj() * psi[:, 1]).real
                om = bx.reshape(-1) if torch.is_tensor(bx) else bx
                out["E_rabi"] = 2.0 * om * g.integrate(cross)
        else:
            out["E_delta"] = torch.zeros_like(out["norm"])
            out["E_rabi"] = torch.zeros_like(out["norm"])
        out["E_total"] = out["E_kin"] + out["E_pot"] + out["E_int"] + out["E_delta"] + out["E_rabi"]
        # mu = <H> / N with the mean-field term counted once more than in E
        out["mu"] = (out["E_kin"] + out["E_pot"] + E_mu_int + out["E_delta"] + out["E_rabi"]) / out["norm"]
        return out

    def apply_H(self, psi: torch.Tensor) -> torch.Tensor:
        """H psi with the mean field at the current density (B, C, *N)."""
        psi_k = self.grid.fft(psi)
        Hpsi = self.kappa * self.grid.ifft(psi_k * self.ksq().to(self.dtype))
        n = self.density(psi)
        V = self.V
        loc = None
        if V is not None:
            loc = V
        gn = self._mu_int(n)
        if gn is not None:
            loc = gn if loc is None else loc + gn
        if loc is not None:
            Hpsi = Hpsi + loc.to(self.dtype) * psi
        if self.C == 2:
            if not self._is_zero(self.delta):
                Hpsi = Hpsi + torch.cat([0.5 * self.delta * psi[:, 0:1], -0.5 * self.delta * psi[:, 1:2]], dim=1).to(
                    self.dtype
                )
            if not self._is_zero(self.Omega_R):
                Hpsi = Hpsi + self.Omega_R * torch.cat([psi[:, 1:2], psi[:, 0:1]], dim=1)
        return Hpsi

    def residual(self, psi: torch.Tensor) -> torch.Tensor:
        """||H psi - mu psi|| / ||psi|| per batch member (L2 with the volume
        element), the convergence criterion of the ground-state solver."""
        terms = self.energy_terms(psi)
        mu = terms["mu"].reshape((-1, 1) + (1,) * self.grid.d).to(self.dtype)
        r = self.apply_H(psi) - mu * psi
        rr = self.grid.integrate((r.real**2 + r.imag**2).sum(dim=1)).reshape(-1)
        return torch.sqrt(rr / terms["norm"])

    def current(self, psi: torch.Tensor) -> list[torch.Tensor]:
        """Number-current density j_i = 2 kappa sum_sigma Im(psi_sigma^* (d_i + i q_i) psi_sigma),
        one (B, *N) tensor per axis."""
        psi_k = self.grid.fft(psi)
        out = []
        for ax in range(self.grid.d):
            dpsi = self.grid.ifft((1j * self.k_axis(ax).to(self.dtype)) * psi_k)
            j = 2.0 * self.kappa * (psi.conj() * dpsi).imag
            out.append(j.sum(dim=1))
        return out

    def momentum_distribution(self, psi: torch.Tensor) -> torch.Tensor:
        """|psi_sigma(k)|^2 on the fftshifted k grid, normalised so that its
        sum over k times dk^d = N: shape (B, C, *N)."""
        psi_k = self.grid.fft(psi) * self.grid.dV
        nk = psi_k.real**2 + psi_k.imag**2
        return torch.fft.fftshift(nk, dim=self.grid.sdims) / (2.0 * math.pi) ** self.grid.d


def local_energy_scale(engine: BilayerGPE, psi: torch.Tensor) -> float:
    """max |V| + g n_peak + |Omega_R| + |Delta|/2: the largest local energy,
    which bounds the first imaginary-time step (dt * E_loc <~ 1)."""
    n_peak = float(engine.density(psi).max())
    s = 0.0
    if engine.V is not None:
        s += float(engine.V.abs().max())
    if engine.interaction is not None:
        s += abs(float(engine.interaction.mu_int(torch.tensor(n_peak, dtype=torch.float64))))
    else:
        s += n_peak * (float(engine.g.abs().max()) if torch.is_tensor(engine.g) else abs(engine.g))
    for p, w in ((engine.Omega_R, 1.0), (engine.delta, 0.5)):
        s += w * (float(p.abs().max()) if torch.is_tensor(p) else abs(p))
    return s


def choose_dt(
    engine: BilayerGPE,
    psi: torch.Tensor,
    frac: float = 0.2,
    order: int = 2,
    order4_multiplier: float = 1.0,
    max_kinetic_phase: float = 1.5,
) -> float:
    """Accuracy-safe real-time dt: phase per step <= frac for the fastest
    Fourier mode and for the largest local energy (|V| + g n_peak + |Omega_R| +
    |Delta|/2). order=4 multiplies by order4_multiplier -- calibrate it per
    scenario (a short sweep does not certify it: a multiplier that passes a short
    run can still blow up on a long one).

    Split-step is unitary for any dt, but NOT stable for any dt with a
    nonlinearity: when the kinetic phase per step kappa k_max^2 dt exceeds pi,
    pairs of grid modes with (eps_k + eps_k') dt = 2 pi m are parametrically
    amplified by g n (the standard split-step NLS instability; its onset on the
    supercell is at ~3-4 rad). The order-4 multiplier
    must therefore never push kappa k_max^2 dt past max_kinetic_phase (1.5 rad
    by default, with margin below pi): that cap is applied last."""
    E_kin_max = engine.kappa * engine.grid.k_sq_max
    loc = local_energy_scale(engine, psi)
    dt = frac / max(E_kin_max, 1e-300)
    if loc > 0:
        dt = min(dt, frac / loc)
    if order == 4:
        dt *= order4_multiplier
    elif order != 2:
        raise ValueError(f"order must be 2 or 4, got {order!r}")
    if E_kin_max > 0:
        dt = min(dt, max_kinetic_phase / E_kin_max)
    return dt
