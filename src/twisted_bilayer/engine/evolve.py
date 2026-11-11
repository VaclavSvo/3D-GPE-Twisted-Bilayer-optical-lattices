"""The shared "advance in blocks, record diagnostics at each boundary" loop,
written once so that scenarios and tests record exactly the same quantities.

What is recorded at t0 and after every block (`run_and_record`):

  "t"                       time (lattice units, hbar / E_R)
  every key of engine.energy_terms(psi)
                            norm, E_kin, E_pot, E_int, E_delta, E_rabi, E_total, mu
  "N1", "N2"                component atom numbers (C = 2 only)
  "P0", "P1", ...           total momentum along each spatial axis,
                            P_i = m int j_i d^dr = int j_i / (2 kappa) d^dr
                            (units hbar / a), j from engine.current

Values are python floats for B = 1 and lists of B per-member floats for
B > 1. For B = 1 the history is what twisted_bilayer.gates.core.conservation_gates
consumes (norm, E_total).
"""

from __future__ import annotations

import time

import torch

from .physics import BilayerGPE


def format_duration(seconds: float) -> str:
    """Compact h/m/s, for progress lines."""
    seconds = max(0.0, float(seconds))
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    if seconds < 5400.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


def _to_py(v: torch.Tensor, B: int):
    v = v.reshape(-1)
    if B == 1:
        return float(v[0])
    if v.numel() == 1:
        v = v.expand(B)
    return [float(x) for x in v.tolist()]


def _batch_size(engine: BilayerGPE, psi: torch.Tensor) -> int:
    """Members of the run: psi's batch, or the engine's batched parameters when
    a single field is broadcast over a sweep (run_block broadcasts it too)."""
    return max(psi.shape[0], engine.batch_hint)


def record_frame(engine: BilayerGPE, psi: torch.Tensor) -> dict:
    """One frame of diagnostics (see the module docstring), as python numbers."""
    B = _batch_size(engine, psi)
    out = {}
    for k, v in engine.energy_terms(psi).items():
        out[k] = _to_py(v, B)
    if engine.C == 2:
        cn = engine.component_norms(psi)  # (B, 2)
        out["N1"] = _to_py(cn[:, 0], B)
        out["N2"] = _to_py(cn[:, 1], B)
    m = 1.0 / (2.0 * engine.kappa)
    for ax, j in enumerate(engine.current(psi)):
        out[f"P{ax}"] = _to_py(m * engine.grid.integrate(j), B)
    return out


def run_and_record(
    engine: BilayerGPE,
    psi: torch.Tensor,
    dt: float,
    order: int,
    n_blocks: int,
    steps_per_block: int,
    observer=None,
    progress: bool = False,
    t0: float = 0.0,
) -> tuple[dict, torch.Tensor]:
    """Advance psi through n_blocks x steps_per_block real-time steps of dt,
    recording diagnostics at t0 and after every block.

    Returns (history, psi):
      history   -- dict of equal-length lists (module docstring), n_blocks + 1 frames.
      psi       -- the final field. The input is consumed (engine.run_block).

    observer -- optional object with on_frame(psi, t, block_idx), called at
        t0 and after every block on the LIVE field, before anything is
        reduced (the hook a ramp observer uses).

    Times are t0 + i * steps_per_block * dt, computed by multiplication, not
    accumulated, so frame times carry no round-off drift.
    """
    if n_blocks < 0 or steps_per_block < 0:
        raise ValueError("n_blocks and steps_per_block must be >= 0")
    history: dict = {"t": []}
    B = _batch_size(engine, psi)
    if psi.shape[0] != B:
        if psi.shape[0] != 1:
            raise ValueError(f"psi has batch {psi.shape[0]}, the engine's parameters {B}")
        psi = psi.expand((B,) + tuple(psi.shape[1:])).clone()  # every frame has B members

    def record(block_idx: int, t: float):
        history["t"].append(float(t))
        for k, v in record_frame(engine, psi).items():
            history.setdefault(k, []).append(v)
        if observer is not None:
            observer.on_frame(psi, t, block_idx)

    record(0, t0)
    wall0 = time.time()
    for i in range(n_blocks):
        psi = engine.run_block(psi, steps_per_block, dt, order=order)
        record(i + 1, t0 + (i + 1) * steps_per_block * dt)
        if progress:
            done = i + 1
            per_block = (time.time() - wall0) / done
            print(
                f"\r  block {done}/{n_blocks}  t={history['t'][-1]:.4g}  {per_block:.2f}s/block"
                f"  ETA {format_duration(per_block * (n_blocks - done))}      ",
                end="",
                flush=True,
            )
    if progress:
        print()
    return history, psi
