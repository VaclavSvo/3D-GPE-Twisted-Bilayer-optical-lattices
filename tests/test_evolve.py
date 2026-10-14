"""engine.evolve.run_and_record: what it records, and that the recorded
history is right.

* Bare Rabi oscillation (V = None, g = 0, uniform): the local 2x2 step is
  exact and the kinetic step acts on k = 0 only, so N2/N = sin^2(Omega_R t)
  holds at every frame to round-off (Omega_R is the off-diagonal
  element). Also the batched version, one Omega_R per member.
* A 1D lattice ground state (V0 = 8, g = 0.2) is stationary: norm, energy and
  momentum are conserved at order 2 and 4, and psi(t) = e^{-i mu t} psi(0)
  with the mu of energy_terms.
* A kicked lattice state is not stationary: the energy error of the scheme
  scales as dt^order (2 for Strang, 4 for Yoshida), while the total momentum
  genuinely changes (dP/dt = -int n dV/dx, Bloch dynamics).
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer.engine.evolve import record_frame, run_and_record
from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, relax
from twisted_bilayer.engine.lattice import lattice_1d
from twisted_bilayer.engine.physics import BilayerGPE, choose_dt

CPU = torch.device("cpu")


def _drift(h: dict, key: str, scale=None) -> float:
    """Worst member's max |v(t) - v(0)|, relative to |v(0)| or to `scale` (float or per-member list)."""
    v = np.array(h[key], dtype=np.float64)
    v = v.reshape(len(v), -1)  # (frames, members)
    d = np.abs(v - v[0]).max(axis=0)
    ref = np.abs(v[0]) if scale is None else np.broadcast_to(np.asarray(scale, dtype=np.float64), d.shape)
    return float((d / ref).max())


class _Recorder:
    def __init__(self):
        self.calls = []

    def on_frame(self, psi, t, block_idx):
        self.calls.append((block_idx, t, tuple(psi.shape)))


@pytest.mark.parametrize("order", [2, 4])
def test_rabi_oscillation_every_frame(order):
    # a coarse grid on purpose: the state is uniform, so only k = 0 is occupied,
    # and dx = 1 makes choose_dt's step 0.1 (min of 0.2 / (kappa k_max^2) = 0.1 and
    # 0.2 / Omega_R = 0.4): 64 steps cover one full period pi / Omega_R of N2/N
    g = Grid((8, 8), (8.0, 8.0), device=CPU)
    Omega = 0.5
    eng = BilayerGPE(g, V=None, g=0.0, Omega_R=Omega)
    N = 16.0
    psi = initial_state(eng, N, fractions=(1.0, 0.0))
    dt = choose_dt(eng, psi, order=order)
    obs = _Recorder()
    n_blocks, spb = 16, 4  # t_end = 6.4 > the period pi / Omega
    h, psi_out = run_and_record(eng, psi, dt, order, n_blocks, spb, observer=obs)
    keys = {"t", "norm", "E_kin", "E_pot", "E_int", "E_delta", "E_rabi", "E_total", "mu", "N1", "N2", "P0", "P1"}
    assert set(h) == keys
    assert all(len(v) == n_blocks + 1 for v in h.values())
    assert all(isinstance(x, float) for x in h["N2"])  # B = 1: plain floats
    t = np.array(h["t"])
    np.testing.assert_allclose(t, dt * spb * np.arange(n_blocks + 1), rtol=0, atol=1e-15)
    frac = np.array(h["N2"]) / np.array(h["norm"])
    assert np.abs(frac - np.sin(Omega * t) ** 2).max() < 1e-8
    np.testing.assert_allclose(np.array(h["N1"]) + np.array(h["N2"]), h["norm"], rtol=1e-13)
    # E = 2 Omega Re int psi1* psi2 = 0 identically (psi2 = -i tan(Omega t) psi1): measure
    # the energy against the Rabi scale Omega N, the momentum against N 2pi/L
    assert _drift(h, "norm") < 1e-10
    assert _drift(h, "E_total", Omega * N) < 1e-10
    assert max(_drift(h, f"P{ax}", N * 2 * math.pi / 8.0) for ax in (0, 1)) < 1e-10
    # observer: t0 and every block, on the live field
    assert [c[0] for c in obs.calls] == list(range(n_blocks + 1))
    assert [c[1] for c in obs.calls] == h["t"]
    assert obs.calls[-1][2] == (1, 2, 8, 8)


def test_rabi_batched_members():
    """One call, two Omega_R: history entries are per-member lists."""
    g = Grid((8, 8), (8.0, 8.0), device=CPU)
    Om = torch.tensor([0.5, 0.2], dtype=torch.float64)
    eng = BilayerGPE(g, V=None, g=0.0, Omega_R=Om)
    psi = initial_state(eng, 8.0, batch=2, fractions=(1.0, 0.0))
    dt = choose_dt(eng, psi, order=4)
    h, _ = run_and_record(eng, psi, dt, 4, 8, 4)
    assert all(isinstance(x, list) and len(x) == 2 for x in h["N2"])
    t = np.array(h["t"])
    frac = np.array(h["N2"]) / np.array(h["norm"])  # (frames, 2)
    assert np.abs(frac - np.sin(np.outer(t, Om.numpy())) ** 2).max() < 1e-8
    assert _drift(h, "norm") < 1e-10
    assert _drift(h, "E_total", [0.5 * 8.0, 0.2 * 8.0]) < 1e-10
    assert max(_drift(h, f"P{ax}", 8.0 * 2 * math.pi / 8.0) for ax in (0, 1)) < 1e-10
    # a single field broadcast over the sweep records the same per-member history
    psi1 = initial_state(eng, 8.0, batch=1, fractions=(1.0, 0.0))
    h1, out1 = run_and_record(eng, psi1, dt, 4, 8, 4)
    assert out1.shape[0] == 2 and len(h1["norm"][0]) == 2
    np.testing.assert_allclose(np.array(h1["N2"]), np.array(h["N2"]), rtol=0, atol=1e-13)


def _lattice_ground_state(C):
    g = Grid((64,), (4.0,), device=CPU)  # 4 sites, 16 points per a
    V = lattice_1d(g, 8.0)
    Omega = 0.25 if C == 2 else 0.0
    if C == 2:
        V = torch.cat([V, V], dim=1)  # the engine wants (1, C, *N)
    eng = BilayerGPE(g, V=V, g=0.2, Omega_R=Omega, n_components=C)
    psi0 = initial_state(eng, 4.0)
    # initial_state starts component 2 at relative phase pi for Omega_R > 0:
    # the lower dressed state of +Omega_R sigma_x is the antisymmetric one
    r = relax(eng, psi0, 4.0, tol=1e-9, polish_steps=2000)
    assert bool(r.converged.all()), r.report()
    if C == 2:
        assert float(eng.energy_terms(r.psi)["E_rabi"]) == pytest.approx(-Omega * 4.0, rel=1e-9)
    return g, eng, r


@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize("order", [2, 4])
def test_lattice_ground_state_conserved(C, order):
    g, eng, r = _lattice_ground_state(C)
    dt = choose_dt(eng, r.psi, order=order)
    h, psi = run_and_record(eng, r.psi.clone(), dt, order, n_blocks=10, steps_per_block=20)
    assert _drift(h, "norm") < 1e-12
    assert _drift(h, "E_total") < 1e-6
    # momentum against the largest the kinetic energy allows, sqrt(N E_kin / kappa)
    p_char = math.sqrt(max(h["norm"]) * max(h["E_kin"]) / eng.kappa)
    assert _drift(h, "P0", p_char) < 1e-6
    assert ("N1" in h) == (C == 2)
    # stationary: psi(t) = exp(-i mu t) psi(0); splitting error ~dt^order
    mu, t = float(r.mu[0]), h["t"][-1]
    ov = complex((r.psi.conj() * psi).sum() * g.dV) / 4.0
    assert abs(ov - complex(math.cos(mu * t), -math.sin(mu * t))) < 1e-6
    n0, n1 = eng.density(r.psi), eng.density(psi)
    assert float((n1 - n0).abs().max() / n0.max()) < 1e-5


def test_kicked_lattice_state_energy_order_and_momentum():
    g, eng, r = _lattice_ground_state(1)
    kick = torch.exp(1j * (math.pi * g.coord(0))).to(eng.dtype)  # k = pi = 2 x 2pi/L, periodic
    T = 0.5
    for order in (2, 4):
        drift = []
        for dt in (0.005, 0.0025):
            n = int(round(T / dt))
            h, _ = run_and_record(eng, r.psi * kick, dt, order, 10, n // 10)
            drift.append(_drift(h, "E_total"))
            assert drift[-1] < 1e-4
            p_char = math.sqrt(max(h["norm"]) * max(h["E_kin"]) / eng.kappa)
            assert _drift(h, "P0", p_char) > 1e-4  # Bloch dynamics: P is not conserved
        assert drift[0] / drift[1] == pytest.approx(2.0**order, rel=0.1)


def test_record_frame_3d_momentum():
    g = Grid((8, 8, 6), (2.0, 2.0, 1.5), device=CPU)
    eng = BilayerGPE(g, V=None, g=0.1, Omega_R=0.3)
    h, _ = run_and_record(eng, initial_state(eng, 2.0, batch=3, kind="gaussian", width=0.4), 1e-3, 2, 2, 3)
    assert {"P0", "P1", "P2"} <= set(h) and len(h["P2"][0]) == 3
    # momentum of a boosted state: P = N k exactly (single plane wave)
    k = 2 * math.pi / 2.0
    boosted = (initial_state(eng, 2.0) * torch.exp(1j * k * g.coord(0))).to(eng.dtype)
    fr = record_frame(eng, boosted)
    assert fr["P0"] == pytest.approx(2.0 * k, rel=1e-12)
    assert abs(fr["P1"]) < 1e-12 and abs(fr["P2"]) < 1e-12
