"""Ground states: imaginary-time dt ladder with a residual criterion, plus a
preconditioned-gradient polish for the slow modes.

Why a ladder: split-step imaginary time is a
contraction for any dt; what dt sets is the relaxation rate exp(-dt E_exc)
of each excitation, and the modes that hold the last of the excess energy
are the LOWEST ones. On the supercell those are the long-wavelength phonons
(E ~ c 2pi/L ~ 0.01 E_R) and, in the pocket regime, the inter-pocket phase
modes (E ~ J^M, 1e-3 E_R and below). At the accuracy step dt ~ 1e-3 they
relax at 1e-5 per step -- hopeless. Large dt stages kill them in hundreds of
steps at the price of an O(dt^2) splitting error on the fixed point; each
smaller stage then only has to remove that error, which is a high-energy
(fast) correction. A stage ends when the residual ||H psi - mu psi|| stalls,
not when the energy stops moving: the energy is quadratic in the error and
goes flat long before the state is converged.

Why the residual: the quantity every observable here is sensitive to (f_s
from a twist, the current, the inter-pocket phases) is first order in the
error of psi, and the energy change is second order: check convergence on
the residual and on the current, not on the energy.

The polish step (`precondition_polish`) is steepest descent on E with the
kinetic preconditioner P = (kappa k^2 + alpha)^-1 (Antoine, Levitt & Tang,
J. Comput. Phys. 343, 92 (2017), their "PG" method with a Laplace
preconditioner): every k-mode gets a comparable step, so it does not suffer
the dt_min bottleneck. It is used after the ladder when the residual is
still above tol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .physics import BilayerGPE, local_energy_scale


@dataclass
class RelaxResult:
    psi: torch.Tensor
    residual: torch.Tensor  # (B,) final ||H psi - mu psi|| / ||psi||
    mu: torch.Tensor  # (B,)
    energy: torch.Tensor  # (B,) E_total
    converged: torch.Tensor  # (B,) bool
    steps: int = 0
    stages: list = field(default_factory=list)  # (dt, steps, max residual) per stage
    stop_reason: str = ""  # why the last stage ended: "tol", "stall", "slow", "max_steps", "tau_floor" or "ladder"

    def report(self) -> str:
        lines = [f"  stage dt={dt:.3g}: {n} steps, max residual {r:.2e}" for dt, n, r in self.stages]
        ok = int(self.converged.sum())
        lines.append(
            f"  {ok}/{len(self.converged)} converged, {self.steps} steps total, "
            f"residual max {float(self.residual.max()):.2e}, stopped on {self.stop_reason or '-'}"
        )
        return "\n".join(lines)


def _bcast(x, psi):
    """(B,) -> (B, 1, *1) for multiplying a (B, C, *N) field."""
    return x.reshape((-1, 1) + (1,) * (psi.ndim - 2))


def renormalise(engine: BilayerGPE, psi: torch.Tensor, n_particles) -> torch.Tensor:
    """Scale each batch member to n_particles atoms (scalar or (B,))."""
    norm = engine.norm(psi)
    target = torch.as_tensor(n_particles, dtype=norm.dtype, device=norm.device)
    factor = torch.sqrt(target / norm)
    return psi * _bcast(factor, psi).to(psi.dtype)


def _relative_phases(engine: BilayerGPE, batch: int) -> list[float]:
    """The default relative phase of component 2 (rad) of `initial_state`: [p] when every batch member gets the
    same phase, else one phase per member. A member starts on its lower dressed branch: pi for Omega_R > 0, 0 for
    Omega_R < 0; a member with Omega_R = 0 (no branch to choose) takes the batch's phase, pi when any Omega_R > 0."""
    om = engine.Omega_R
    if not torch.is_tensor(om):
        return [math.pi if float(om) > 0.0 else 0.0]
    vals = [float(x) for x in om.reshape(-1)]
    default = math.pi if max(vals) > 0.0 else 0.0
    phases = [math.pi if x > 0.0 else 0.0 if x < 0.0 else default for x in vals]
    if all(p == phases[0] for p in phases):
        return [phases[0]]
    if len(phases) != batch:
        raise ValueError(
            f"Omega_R mixes signs over {len(phases)} batch members but the start has batch={batch}: each member "
            "needs its own relative phase (pass batch=len(Omega_R), or relative_phase explicitly)"
        )
    return phases


def initial_state(
    engine: BilayerGPE,
    n_particles,
    batch: int = 1,
    kind: str = "uniform",
    fractions=(0.5, 0.5),
    seed: int | None = None,
    width: float | None = None,
    phase_noise: float = 0.0,
    relative_phase: float | None = None,
) -> torch.Tensor:
    """A starting field (B, C, *N) with n_particles atoms per member (scalar or (B,)); batch = B.

    kind: "uniform"  -- flat density (the right start on a periodic supercell)
          "gaussian" -- centred Gaussian of the given width (in a; default L_min / 8) (for traps)
          "thomas_fermi" -- sqrt(max(mu - V, 0)) with mu fitted to N (needs V)
    seed: of the phase_noise field (None: torch's unseeded generator).
    fractions: share of atoms in each component (C = 2).
    relative_phase: phase of component 2 relative to component 1 (rad), the
    same for every member. With the paper's +Omega_R sigma_x coupling the
    LOWER dressed state is the antisymmetric combination, so the default
    (None) is chosen per batch member: pi when Omega_R > 0, 0 when
    Omega_R < 0 or absent; a member with Omega_R = 0 takes pi when any member
    has Omega_R > 0. A batch that mixes signs needs batch = len(Omega_R).
    Starting symmetric would converge to the upper dressed state and stay
    there: the layer-swap + y -> -y mirror is an exact symmetry of the
    bilayer that imaginary time cannot break.
    phase_noise: standard deviation (rad) of a random phase field added on
    top, for independent-seed checks of the inter-pocket phases.
    """
    g = engine.grid
    shape = (batch, engine.C) + g.N
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))
    if kind == "uniform":
        amp = torch.ones(shape, dtype=engine.rdtype, device=g.device)
    elif kind == "gaussian":
        w = (min(g.L) / 8.0) if width is None else float(width)
        r2 = sum(g.coord(ax) ** 2 for ax in range(g.d))
        amp = torch.exp(-0.5 * r2 / (w * w)).expand(shape).clone()
    elif kind == "thomas_fermi":
        if engine.V is None:
            raise ValueError("thomas_fermi needs a potential")
        V = engine.V.mean(dim=1, keepdim=True) if engine.V.shape[1] > 1 else engine.V
        gval = float(engine.g.mean()) if torch.is_tensor(engine.g) else float(engine.g)
        gval = max(gval, 1e-12)
        target = float(torch.as_tensor(n_particles, dtype=torch.float64).mean())
        lo, hi = float(V.min()), float(V.max()) + gval * target / g.volume * 10.0 + 1.0
        for _ in range(80):  # bisection on mu for int (mu - V)/g = N
            mu = 0.5 * (lo + hi)
            Ncur = float(g.integrate((mu - V).clamp_min(0.0)).mean()) / gval
            lo, hi = (mu, hi) if Ncur < target else (lo, mu)
        amp = torch.sqrt((mu - V).clamp_min(0.0) / gval).expand(shape).clone()
    else:
        raise ValueError(f"unknown initial state kind {kind!r}")
    if engine.C == 2:
        f = torch.tensor(fractions, dtype=engine.rdtype, device=g.device).sqrt()
        amp = amp * f.reshape(1, 2, *([1] * g.d))
    psi = amp.to(engine.dtype)
    if engine.C == 2:
        phases = _relative_phases(engine, batch) if relative_phase is None else [float(relative_phase)]
        if any(p != 0.0 for p in phases):
            ph = torch.ones((len(phases), 2) + (1,) * g.d, dtype=engine.dtype, device=g.device)
            for i, p in enumerate(phases):
                ph[i, 1] = complex(math.cos(p), math.sin(p))
            psi = psi * ph
    if phase_noise > 0.0:
        ph = torch.randn(shape, generator=gen, dtype=torch.float64) * phase_noise
        psi = psi * torch.exp(1j * ph.to(g.device, engine.rdtype)).to(engine.dtype)
    return renormalise(engine, psi, n_particles)


def dt_ladder(
    engine: BilayerGPE, psi: torch.Tensor, dt_start: float = 0.5, dt_min: float | None = None, factor: float = 4.0
) -> list[float]:
    """dt_first = min(dt_start, 1/E_loc), then /factor down to the accuracy
    step dt_min = 0.5 / (kappa k_max^2)."""
    if dt_min is None:
        dt_min = 0.5 / (engine.kappa * engine.grid.k_sq_max)
    E_loc = local_energy_scale(engine, psi)
    d = min(float(dt_start), 1.0 / max(E_loc, 1e-30))
    schedule = []
    while d > 2.0 * dt_min:
        schedule.append(d)
        d /= factor
    schedule.append(dt_min)
    return schedule


def relax(
    engine: BilayerGPE,
    psi0: torch.Tensor,
    n_particles,
    tol: float = 1e-7,
    dt_schedule: list[float] | None = None,
    dt_start: float = 0.5,
    dt_min: float | None = None,
    max_steps_per_stage: int = 2000,
    check_every: int = 50,
    stall_ratio: float = 0.9,
    polish_steps: int = 20000,
    polish_tau: float = 0.5,
    polish_beta: float = 0.9,
    polish_alpha: float | None = None,
    polish_stall_windows: int | None = None,
    verbose: bool = False,
) -> RelaxResult:
    """Ground state: imaginary-time dt ladder, then the preconditioned
    momentum polish, both on the residual criterion (polish_stall_windows:
    see `precondition_polish`; stop_reason records why it ended).

    Each ladder stage runs until every batch member has residual < tol, or
    the max residual has improved by less than (1 - stall_ratio) over one
    check window (the O(dt^2) plateau of that stage, or the slow-mode floor),
    or max_steps_per_stage. If the ladder ends above tol and polish_steps >
    0, `precondition_polish` takes over -- on the supercell it does most of
    the work: the ladder is worth ~1e-3 in residual in a few hundred steps,
    the polish the remaining six orders of magnitude.

    Parameters (energies in the engine's units, E_R in lattice units; times in hbar / E_R):
        engine: the `BilayerGPE` (its dtype sets the precision of the solve).
        psi0: the start (B, C, *N); renormalised to n_particles first.
        n_particles: atoms per batch member (scalar or (B,)); kept fixed throughout.
        tol: residual ||H psi - mu psi|| / ||psi|| (E_R) at which a member counts as converged.
        dt_schedule: the imaginary-time steps of the ladder; None builds it with `dt_ladder` from dt_start
            and dt_min (default 0.5 / (kappa k_max^2)); [] skips the ladder (polish only).
        max_steps_per_stage, check_every, stall_ratio: the ladder's step cap per stage, the steps between
            residual checks, and the fraction the residual must fall below per check to stay in a stage.
        polish_steps: the polish's step budget (0: no polish). polish_tau, polish_beta, polish_alpha,
            polish_stall_windows: tau, beta, alpha and stall_windows of `precondition_polish`.
        verbose: print one line per stage and per polish window.

    Returns a `RelaxResult`: psi, residual, mu, energy (E_total) and converged per member, the total steps,
    the stages and the stop_reason."""
    psi = renormalise(engine, psi0.to(engine.dtype), n_particles)
    schedule = dt_schedule if dt_schedule is not None else dt_ladder(engine, psi, dt_start, dt_min)
    total = 0
    stages = []
    res = engine.residual(psi)
    for si, dt in enumerate(schedule):
        kin_half = engine._kin_ops(0.5 * dt, imaginary=True)
        prev = float(res.max())
        n_stage = 0
        while n_stage < max_steps_per_stage:
            for _ in range(check_every):
                psi = engine.imaginary_step(psi, dt, kin_half)
                psi = renormalise(engine, psi, n_particles)
            n_stage += check_every
            res = engine.residual(psi)
            cur = float(res.max())
            if cur < tol:
                break
            if cur > stall_ratio * prev:
                break  # this stage's splitting plateau (or stalled): move on
            prev = cur
        total += n_stage
        stages.append((dt, n_stage, float(res.max())))
        if verbose:
            print(
                f"  imaginary time stage {si + 1}/{len(schedule)} dt={dt:.3g}: "
                f"{n_stage} steps, max residual {float(res.max()):.2e}"
            )
        if float(res.max()) < tol:
            break
    stop = "tol" if float(res.max()) < tol else "ladder"
    if float(res.max()) >= tol and polish_steps > 0:
        info: dict = {}
        psi, res, n_pol = precondition_polish(
            engine,
            psi,
            n_particles,
            tol=tol,
            max_steps=polish_steps,
            tau=polish_tau,
            beta=polish_beta,
            alpha=polish_alpha,
            check_every=check_every,
            stall_windows=polish_stall_windows,
            info=info,
            verbose=verbose,
        )
        total += n_pol
        stages.append((0.0, n_pol, float(res.max())))
        stop = info.get("stop_reason", "")
    terms = engine.energy_terms(psi)
    return RelaxResult(
        psi=psi,
        residual=res,
        mu=terms["mu"],
        energy=terms["E_total"],
        converged=res < tol,
        steps=total,
        stages=stages,
        stop_reason=stop,
    )


def _stalled(hist: list, windows: int) -> bool:
    """True when the last residual is above 0.995 x the one `windows` check windows earlier: the residual
    fell by less than 0.5 % over that span (windows = 1: the one-window test)."""
    return len(hist) > windows and hist[-1] > 0.995 * hist[-1 - windows]


def _too_slow(hist: list, windows: int, windows_left: float, tol: float) -> bool:
    """True when the residual has fallen by less than 0.5 % per window on average over the last `windows`
    check windows (the one-window stall threshold) and, at that rate, would still be above tol after
    `windows_left` more windows. Extrapolated at its present rate, such a descent ends unconverged; stopping
    it saves the rest of the step budget. A descent faster than 0.5 % per window always goes on (as with the one-window
    test), and so does a slower one that can still reach tol in the budget."""
    if len(hist) <= windows or hist[-1 - windows] <= 0.0:
        return False
    q = (hist[-1] / hist[-1 - windows]) ** (1.0 / windows)
    if not 0.995 < q < 1.0:  # faster than the one-window threshold, or not falling (the stall test's case)
        return False
    return hist[-1] * q ** max(windows_left, 0.0) > tol


_E_TERMS = ("E_kin", "E_pot", "E_int", "E_delta", "E_rabi")


def _energy(engine: BilayerGPE, psi: torch.Tensor) -> tuple:
    """(E_total, sum of |terms|) per batch member: the energy and the scale of its rounding error."""
    t = engine.energy_terms(psi)
    return t["E_total"], sum(t[k].abs() for k in _E_TERMS)


def _energy_rose(E, E_prev, eps: float) -> bool:
    """True when any batch member's total energy rose by more than its rounding level: 1e3 eps times the sum
    of the magnitudes of its terms (so a total that cancels to ~0 keeps a meaningful scale). E, E_prev:
    (E_total, scale) pairs as returned by `_energy`."""
    (e, s), (e0, s0) = E, E_prev
    e, s, e0, s0 = (torch.as_tensor(x, dtype=torch.float64).reshape(-1) for x in (e, s, e0, s0))
    return bool(((e - e0) > 1e3 * eps * torch.maximum(s, s0)).any())


def _energy_fell(E, E_prev, eps: float, res) -> bool:
    """True when no batch member's total energy rose by more than its rounding level (as in `_energy_rose`) and
    the member with the largest residual `res` (B,) fell by more than it: the state is still descending."""
    (e, s), (e0, s0) = E, E_prev
    e, s, e0, s0 = (torch.as_tensor(x, dtype=torch.float64).reshape(-1) for x in (e, s, e0, s0))
    level = 1e3 * eps * torch.maximum(s, s0)
    if bool(((e - e0) > level).any()):
        return False
    i = int(torch.as_tensor(res).reshape(-1).argmax()) if e.numel() > 1 else 0
    return bool((e - e0)[i] < -level[i])


def precondition_polish(
    engine: BilayerGPE,
    psi: torch.Tensor,
    n_particles,
    tol: float = 1e-8,
    max_steps: int = 2000,
    tau: float = 0.5,
    beta: float = 0.9,
    alpha: float | None = None,
    check_every: int = 50,
    stall_windows: int | None = None,
    info: dict | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Preconditioned gradient descent with momentum (heavy ball) on E[psi]
    at fixed N:

        v   <- beta v - tau P (H psi - mu psi),   P = (kappa k^2 + alpha)^-1
        psi <- renormalise(psi + v)

    alpha defaults to the local energy scale (|V| + g n + Omega_R), which
    keeps the step stable for every k; the momentum is what beats the
    condition number: the lowest supercell modes (E ~ 0.01 E_R) relax at a
    rate ~ tau E/alpha per step without it, ~ sqrt of that with it.

    Overshoot: when the residual rises between two checks the momentum is
    reset, tau is halved (never below 1e-3 tau_0, stop_reason "tau_floor")
    and the run restarts from the best state -- unless, in a complex128
    engine, the energy says the state is still descending. The rounding
    level of E is 1e3 eps times the sum of the magnitudes of its terms:
    - E fell by more than its rounding level (and rose for no member): the
      descent goes on at the same tau, however often this happens, also in
      the first window after a restart, and the stall and rate tests start
      again from this window. The descent lowers E, not the residual: near a
      stationary point (a plateau, or a slow exit along a soft direction)
      the residual stays flat or rises for many windows while E falls
      steadily, and restarting there halves tau down to its floor (a 12 E_R
      trapped cloud at fixed g: E fell by 2e-4 E_R per window, 7e4 times its
      rounding level, through ten restarts).
    - E within its rounding level: one such rise is let through and left
      out of the stall and rate tests; a second rise in a row, or one in the
      first window after a restart, is an overshoot (an unstable mode grows
      the residual by a factor per window while its energy change stays
      below the rounding level; a state that does not lower E right after a
      restart from the best one is at a floor).
    - E rose: an overshoot.
    Near the residual's own rounding level (below 1e4 eps alpha) the
    residual-only test applies in complex128 too: there a rise is rounding
    noise, and halving tau is what lowers the floor. A single-precision
    engine keeps the residual-only test throughout (its energy changes near
    the rounding floor are below its precision), so every complex64 solve
    is unchanged.

    Stop: at tol, at max_steps, or when the residual has stopped falling: by
    less than 0.5 % over the last `stall_windows` check windows (since the
    last restart; "stall"). stall_windows = 1 is the one-window test, which
    also ends a slow but steady descent (soft modes of a large cloud fall by
    ~0.1-0.3 % per window); with 4 a descent faster than ~0.13 % per window
    goes on, but only while it can still reach tol: in a complex128 engine a
    descent slower than 0.5 % per window (over the same span) that, at its
    present rate, would still be above tol at max_steps stops at once
    ("slow"; an extrapolation from the last stall_windows windows; not near
    the rounding level, where the stall test decides). Default
    (None): 4 for a complex128 engine, 1 for a single-precision one, which
    stops at its rounding floor where a longer window only costs steps (so
    every complex64 solve, and the complex64 first stage of a two-precision
    solve, is unchanged). A passed `info` dict receives stop_reason ("tol", "stall",
    "slow", "max_steps", "tau_floor").

    The "slow" stop is an extrapolation, not a verdict on the state: a deep
    lattice's soft phase mode can need 1e5 steps or more (the 12 E_R NPSE
    cloud 95 500, twisted solves at 20-24 E_R 124 250-269 900). Give such a
    run the budget it needs (max_steps; `TrappedSpec.polish_steps`,
    `Vertical.polish_steps`, `SweepSpec.max_polish_steps`).

    Parameters: tol, the residual to reach (E_R); max_steps, the step budget;
    tau (dimensionless; the step is tau P r with P in 1 / E_R) and beta, the
    heavy ball's step and momentum; alpha (E_R), the preconditioner's shift
    (default the local energy scale); check_every, steps per residual check
    (a window). Returns (psi_best, residual_best (B,), steps): the state of
    the lowest residual seen, not necessarily the last."""
    g = engine.grid
    if alpha is None:
        alpha = max(local_energy_scale(engine, psi), 1e-3)
    P = (1.0 / (engine.kappa * engine.ksq() + alpha)).to(engine.dtype)
    res = engine.residual(psi)
    prev = float(res.max())
    hist = [prev]  # window residuals since the last restart
    if stall_windows is None:
        stall_windows = 4 if engine.dtype == torch.complex128 else 1
    stall_windows = max(1, int(stall_windows))
    energy_test = engine.dtype == torch.complex128
    eps = float(torch.finfo(engine.rdtype).eps)
    floor = 1e4 * eps * float(torch.as_tensor(alpha).max())  # the residual's rounding level: below it the
    # residual-only rules apply in complex128 too
    E_prev = _energy(engine, psi) if energy_test else None
    tau0 = tau
    v = torch.zeros_like(psi)
    psi_best, res_best, E_best = psi, res, E_prev
    bumped = False  # the last window's residual rose while E fell (tolerated once)
    n = 0
    reason = "max_steps"
    while n < max_steps:
        for _ in range(check_every):
            terms = engine.energy_terms(psi)
            mu = _bcast(terms["mu"], psi).to(psi.dtype)
            r = engine.apply_H(psi) - mu * psi
            v = beta * v - tau * g.ifft(P * g.fft(r))
            psi = renormalise(engine, psi + v, n_particles)
        n += check_every
        res = engine.residual(psi)
        cur = float(res.max())
        E_cur = _energy(engine, psi) if energy_test else None
        if verbose:
            print(f"  polish {n} steps: max residual {cur:.2e} (tau {tau:.3g})")
        if cur < float(res_best.max()):
            psi_best, res_best, E_best = psi, res, E_cur
        if cur < tol:
            reason = "tol"
            break
        if cur > prev:
            if energy_test and cur > floor and _energy_fell(E_cur, E_prev, eps, res):
                # the residual rose while E fell measurably: still descending (a slow exit from near a stationary
                # point, where the residual first rises). Go on at this tau, as often as it happens; the stall and
                # rate tests start again from this window (they compare residuals, and this is not a stall)
                bumped = True
                hist = [cur]
                prev, E_prev = cur, E_cur
                continue
            if energy_test and not bumped and cur > floor and not _energy_rose(E_cur, E_prev, eps):
                # one rise of the residual while E did not rise: go on, and leave this window out of the
                # stall and rate tests (they compare residuals, and a bump is not a stall)
                bumped = True
                prev, E_prev = cur, E_cur
                continue
            else:  # overshoot (E rose too, or the residual rose twice running): restart from the best state
                psi, res, E_prev = psi_best, res_best, E_best
                v = torch.zeros_like(psi)
                tau *= 0.5
                bumped = True  # after a restart a rise is an overshoot at once (the best state is a floor)
                if tau < 1e-3 * tau0:
                    reason = "tau_floor"
                    break
                prev = float(res.max())
                hist = [prev]
                continue
        else:
            bumped = False
        hist.append(cur)
        if _stalled(hist, stall_windows):
            reason = "stall"
            break
        if (
            energy_test
            and stall_windows > 1
            and cur > floor
            and _too_slow(hist, stall_windows, (max_steps - n) / check_every, tol)
        ):
            reason = "slow"
            break
        prev, E_prev = cur, E_cur
    if info is not None:
        info["stop_reason"] = reason
    return psi_best, res_best, n
