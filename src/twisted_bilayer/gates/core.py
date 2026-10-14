"""Gate primitives (a result record, three generic gates, a suite, a report) and the
engine-independent gates on a recorded history.

Conventions (shared by every engine driver):
- A gate never raises on a physics failure; it returns GateResult(passed=False, ...).
  It may raise on a programming error (missing key).
- `detail` is one line a reader can act on: measured value, reference, tolerance.
- `kind` names the class of check: 'conservation', 'analytic', 'identity', 'anchor',
  'preflight', 'convergence'. The report groups by kind.
- Tolerances are explicit arguments with documented defaults; nothing is hidden.

A `history` is a dict of equal-length sequences, one entry per recorded frame. Keys an
engine may provide (all optional except t and norm):
    t          code time
    norm       total atom number
    N2_over_N  population of component 2
    E_total    energy (gated only when the Hamiltonian is time-independent; pass
               energy_from_t for the end of a ramp)
    highk      spectral weight above k_cut (fraction of the norm); flat for a healthy
               split step, growing for the resonant instability (kinetic phase per step near pi)
    mu         chemical potential or energy per particle (imaginary time: must converge)
    residual   fixed-point residual (imaginary time)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    kind: str = "conservation"
    measured: float | None = None
    reference: float | None = None
    tol: float | None = None


def rel_drift_gate(name: str, values, tol: float, kind: str = "conservation") -> GateResult:
    """Max relative drift from the first value (norm, energy: nonzero reference)."""
    vals = [float(v) for v in values]
    ref = vals[0]
    if abs(ref) < 1e-30:
        return GateResult(name, False, "reference value is zero; use abs_drift_gate", kind)
    drift = max(abs(v - ref) / abs(ref) for v in vals)
    return GateResult(
        name, drift < tol, f"max rel drift {drift:.3e} (tol {tol:.1e}, ref={ref:.9g})", kind, drift, ref, tol
    )


def abs_drift_gate(name: str, values, scale: float, tol: float, kind: str = "conservation") -> GateResult:
    """Max absolute drift against a characteristic scale (quantities whose reference is ~0)."""
    vals = [float(v) for v in values]
    ref = vals[0]
    drift = max(abs(v - ref) for v in vals)
    lim = tol * max(scale, 1e-30)
    return GateResult(
        name,
        drift < lim,
        f"abs drift {drift:.3e} (tol {lim:.3e} = {tol:.1e} x scale {scale:.3e})",
        kind,
        drift,
        ref,
        lim,
    )


def value_gate(
    name: str,
    measured: float,
    reference: float,
    tol: float,
    relative: bool = True,
    kind: str = "analytic",
    unit: str = "",
) -> GateResult:
    """|measured - reference| against tol (relative to |reference| unless relative=False)."""
    m, r = float(measured), float(reference)
    err = abs(m - r)
    if relative:
        if abs(r) < 1e-30:
            return GateResult(name, False, "reference is zero; use relative=False", kind, m, r, tol)
        err = err / abs(r)
    ok = err < tol
    how = "rel" if relative else "abs"
    return GateResult(name, ok, f"{m:.9g}{unit} vs {r:.9g}{unit}: {how} err {err:.3e} (tol {tol:.1e})", kind, m, r, tol)


@dataclass
class GateSuite:
    """A named collection of gates with metadata about the run they judge."""

    engine: str
    scenario: str
    results: list[GateResult] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def add(self, r: GateResult) -> GateResult:
        self.results.append(r)
        return r

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def report(self) -> str:
        return report(self.results, header=f"{self.engine} / {self.scenario}")

    def to_json(self, path=None) -> str:
        return to_json(self, path)


def report(results: list[GateResult], header: str | None = None) -> str:
    lines = []
    if header:
        lines.append(f"== gates: {header}")
    kinds = []
    for r in results:
        if r.kind not in kinds:
            kinds.append(r.kind)
    for k in kinds:
        lines.append(f"-- {k}")
        for r in results:
            if r.kind == k:
                lines.append(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    n_fail = sum(not r.passed for r in results)
    lines.append("ALL GATES PASSED" if n_fail == 0 else f"GATE FAILURE: {n_fail} of {len(results)} failed")
    return "\n".join(lines)


def to_json(suite: GateSuite, path=None) -> str:
    def clean(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x

    d = {
        "engine": suite.engine,
        "scenario": suite.scenario,
        "passed": suite.passed,
        "meta": suite.meta,
        "elapsed_s": time.time() - suite.started,
        "results": [{k: clean(v) for k, v in asdict(r).items()} for r in suite.results],
    }
    s = json.dumps(d, indent=1, default=str)
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


def conservation_gates(
    history: dict,
    tol_norm: float = 1e-6,
    tol_energy: float = 1e-5,
    energy_from_t: float = 0.0,
    J_perp: float = None,
    tol_N2: float = 1e-8,
    highk_growth: float = 10.0,
) -> list[GateResult]:
    """Norm and energy drift; N2/N conservation when J_perp == 0; high-k weight growth."""
    out = [rel_drift_gate("norm", history["norm"], tol_norm)]
    if "E_total" in history:
        E = list(history["E_total"])
        if energy_from_t > 0 and "t" in history:
            idx = [i for i, t in enumerate(history["t"]) if t >= energy_from_t]
            if len(idx) >= 2:
                g = rel_drift_gate("energy", E[idx[0] :], tol_energy)
                g.detail += f" [gated from t = {history['t'][idx[0]]:.4g}]"
                out.append(g)
            else:
                out.append(GateResult("energy", True, "not gated: ramp covers the run"))
        else:
            out.append(rel_drift_gate("energy", E, tol_energy))
    if J_perp is not None and abs(J_perp) < 1e-15 and "N2_over_N" in history:
        out.append(abs_drift_gate("N2_over_N (J_perp = 0)", history["N2_over_N"], 1.0, tol_N2, kind="identity"))
    if "highk" in history:
        h = [float(x) for x in history["highk"]]
        ref = max(h[0], 1e-14)
        growth = max(h) / ref
        out.append(
            GateResult(
                "high-k weight",
                growth < highk_growth,
                f"max/initial = {growth:.3g} (tol {highk_growth:g}); initial {h[0]:.3e}, max {max(h):.3e}",
                "conservation",
                growth,
                1.0,
                highk_growth,
            )
        )
    return out


def imaginary_time_gates(
    history: dict, tol_mu_plateau: float = 1e-7, residual_target: float = 1e-8, window: int = 20
) -> list[GateResult]:
    """Imaginary time must converge: mu plateau over the last `window` frames and a final residual."""
    out = []
    if "mu" in history and len(history["mu"]) >= window:
        mu = [float(x) for x in history["mu"]][-window:]
        spread = (max(mu) - min(mu)) / max(abs(mu[-1]), 1e-30)
        out.append(
            GateResult(
                "mu plateau",
                spread < tol_mu_plateau,
                f"spread over last {window} frames {spread:.3e} (tol {tol_mu_plateau:.1e})",
                "convergence",
                spread,
                0.0,
                tol_mu_plateau,
            )
        )
    if "residual" in history:
        r = float(history["residual"][-1])
        out.append(
            GateResult(
                "final residual",
                r < residual_target,
                f"{r:.3e} (target {residual_target:.1e})",
                "convergence",
                r,
                0.0,
                residual_target,
            )
        )
    return out


def preflight_gates(
    dx_tuple: tuple,
    dt: float,
    phase_max: float = 2.6,
    k_char: float = None,
    safety: float = 2.0,
    kappa: float = None,
    kappa_max: float = 0.8,
) -> list[GateResult]:
    """Checks no post-run gate can make: the split-step phase per step (resonant
    instability above pi), aliasing of a characteristic wavenumber, and the
    quasi-2D parameter kappa for NPSE validity."""
    import math

    from .references import kinetic_phase_per_step

    out = []
    ph = kinetic_phase_per_step(None, dt, dx_tuple)
    out.append(
        GateResult(
            "kinetic phase per step",
            ph < phase_max,
            f"k2max dt/2 = {ph:.3f} rad (limit {phase_max:g}; instability above pi)",
            "preflight",
            ph,
            math.pi,
            phase_max,
        )
    )
    if k_char is not None:
        k_nyq = min(math.pi / d for d in dx_tuple)
        out.append(
            GateResult(
                "resolution",
                k_nyq > safety * abs(k_char),
                f"k_nyquist {k_nyq:.3g} vs {safety:g} x k_char {safety * abs(k_char):.3g}",
                "preflight",
                k_nyq,
                safety * abs(k_char),
                None,
            )
        )
    if kappa is not None:
        out.append(
            GateResult(
                "quasi-2D parameter",
                kappa < kappa_max,
                f"kappa = g2D n_peak / hbar omega_z = {kappa:.3f} (NPSE validated to {kappa_max:g})",
                "preflight",
                kappa,
                kappa_max,
                None,
            )
        )
    return out


# ---------------------------------------------------------------- driver helpers
def new_history() -> dict:
    return dict(t=[], norm=[], E_total=[], N2_over_N=[], highk=[])


def tagged(results, tag: str):
    """Prefix every result's name with `tag` (in place) and return the results."""
    for r in results:
        r.name = f"{tag}: {r.name}"
    return results


def fit_period(t, y, omega_guess: float) -> float:
    """Period of y(t) = A + B cos(W t + phi) by least squares (W0 = omega_guess)."""
    import numpy as np
    from scipy.optimize import curve_fit

    t, y = np.asarray(t, float), np.asarray(y, float)

    def f(t, A, B, W, ph):
        return A + B * np.cos(W * t + ph)

    popt, _ = curve_fit(f, t, y, p0=[y.mean(), 0.5 * (y.max() - y.min()), omega_guess, 0.0])
    return 2.0 * math.pi / abs(popt[2])
