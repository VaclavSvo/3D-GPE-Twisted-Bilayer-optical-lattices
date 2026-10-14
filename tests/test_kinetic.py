"""The split-step engine against exact solutions: free plane waves, the
order of the Yoshida composition, harmonic-oscillator ground states, and
the two-level (Rabi) dynamics of the microwave coupling.

Units: a = 1, E_R = 1, hbar = 1, m = pi^2/2, kappa = 1/pi^2 (see engine.units).
"""

import math

import numpy as np
import pytest
import torch

from twisted_bilayer.engine.grid import Grid
from twisted_bilayer.engine.ground_state import initial_state, relax
from twisted_bilayer.engine.lattice import harmonic_trap
from twisted_bilayer.engine.physics import BilayerGPE
from twisted_bilayer.engine.units import KAPPA, M_LATTICE

CPU = torch.device("cpu")
F64 = torch.float64


def _norm_diff(grid, a, b):
    return float(torch.sqrt(grid.integrate((a - b).abs() ** 2).sum(dim=1)).max())


# -- free particle ----------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize(
    "N,L,modes",
    [
        ((32,), (3.0,), (5,)),
        ((16, 24), (2.0, 3.5), (3, -7)),
        ((8, 12, 10), (1.0, 1.5, 2.0), (2, -3, 4)),
    ],
)
def test_free_plane_wave_exact_phase(order, N, L, modes):
    """exp(i k.r) on the grid is an eigenvector of the FFT kinetic step, so
    psi(t) = exp(-i kappa k^2 t) psi(0) to rounding for any dt and order;
    the second component (another k) checks the C = 2 path with Omega = 0."""
    g = Grid(N, L, device=CPU)
    k1 = [2 * math.pi * mj / l for mj, l in zip(modes, L)]
    k2 = [2 * math.pi * (-mj + 1) / l for mj, l in zip(modes, L)]
    ph1 = sum(k * g.coord(ax) for ax, k in enumerate(k1))
    ph2 = sum(k * g.coord(ax) for ax, k in enumerate(k2))
    psi0 = torch.cat([torch.exp(1j * ph1), torch.exp(1j * ph2)], dim=1).to(torch.complex128)
    psi0 = psi0.expand((1, 2) + g.N).clone()
    eng = BilayerGPE(g, None, g=0.0, Omega_R=0.0, n_components=2)
    t, n = 0.37, 7
    psi = eng.run_block(psi0.clone(), n, t / n, order=order)
    e1 = KAPPA * sum(k * k for k in k1)
    e2 = KAPPA * sum(k * k for k in k2)
    exact = torch.cat([psi0[:, :1] * np.exp(-1j * e1 * t), psi0[:, 1:] * np.exp(-1j * e2 * t)], dim=1)
    assert float((psi - exact).abs().max()) < 1e-12
    # single-component engine: the same
    eng1 = BilayerGPE(g, None, g=0.0, n_components=1)
    psi1 = eng1.run_block(psi0[:, :1].clone(), n, t / n, order=order)
    assert float((psi1 - exact[:, :1]).abs().max()) < 1e-12


# -- Yoshida convergence rate ------------------------------------------------------


def test_yoshida_order4_convergence_rate():
    """Error of the order-4 composition vs dt on a nonlinear 1D problem
    (harmonic trap omega = 1, g = 1, displaced Gaussian, T = 1): the
    log-log slope over dt = T/16 .. T/64 must be 4 (3.7..4.3), and the
    Strang slope 2. Reference: order 4 with dt = T/2048."""
    omega = 1.0
    l = 1.0 / math.sqrt(M_LATTICE * omega)
    g = Grid((64,), (20 * l,), device=CPU)
    eng = BilayerGPE(g, harmonic_trap(g, (omega,)), g=1.0, n_components=1)
    x = g.coord(0)
    psi0 = torch.exp(-((x - 1.5 * l) ** 2) / (2 * l * l)).to(torch.complex128)
    psi0 = psi0 / torch.sqrt(eng.norm(psi0)).reshape(-1, 1, 1)
    T = 1.0
    ref = eng.run_block(psi0.clone(), 2048, T / 2048, order=4)
    ns = [16, 32, 64]
    dts = [T / n for n in ns]
    err4 = [_norm_diff(g, eng.run_block(psi0.clone(), n, T / n, order=4), ref) for n in ns]
    err2 = [_norm_diff(g, eng.run_block(psi0.clone(), n, T / n, order=2), ref) for n in ns]
    slope4 = np.polyfit(np.log(dts), np.log(err4), 1)[0]
    slope2 = np.polyfit(np.log(dts), np.log(err2), 1)[0]
    assert 3.7 < slope4 < 4.3, (err4, slope4)
    assert 1.9 < slope2 < 2.1, (err2, slope2)
    assert err4[-1] < 1e-6
    # norm is conserved by both (unitary steps)
    assert float(eng.norm(eng.run_block(psi0.clone(), 16, T / 16, order=4))[0]) == pytest.approx(1.0, abs=1e-13)


# -- harmonic oscillator ground state ------------------------------------------------


@pytest.mark.parametrize("d,N", [(1, 48), (2, 48), (3, 32)])
def test_qho_ground_state(d, N):
    """V = (1/2) m omega^2 r^2, m = pi^2/2, g = 0: E0 = d omega / 2 and
    <x^2> = 1/(2 m omega) per axis (|psi|^2 ~ exp(-x^2/w^2), w^2 = 1/(m omega)).
    Box 16 oscillator lengths: tail and k-space truncation < 1e-15."""
    omega = 1.0
    l = 1.0 / math.sqrt(M_LATTICE * omega)
    g = Grid((N,) * d, (16 * l,) * d, device=CPU)
    eng = BilayerGPE(g, harmonic_trap(g, (omega,) * d), g=0.0, n_components=1)
    psi0 = initial_state(eng, 1.0, kind="gaussian", width=1.5 * l)
    res = relax(eng, psi0, 1.0, tol=1e-9, polish_steps=2000)
    assert bool(res.converged.all()), res.report()
    E = float(res.energy[0] / eng.norm(res.psi)[0])
    assert E == pytest.approx(d * omega / 2, abs=1e-12)
    n = eng.density(res.psi)
    for ax in range(d):
        x2 = float(g.integrate(g.coord(ax) ** 2 * n)[0, 0])
        assert 2 * x2 == pytest.approx(1.0 / (M_LATTICE * omega), rel=1e-7)


# -- two-level dynamics -------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4])
def test_bare_rabi_oscillation(order):
    """Uniform field, all atoms in |1>, Delta = 0: the population of |2> is
    sin^2(Omega_R t) exactly (Omega_R is the off-diagonal matrix element,
    Supplement Eq. 4 of Meng et al., Nature 615, 231 (2023)); the kinetic step does nothing to a k = 0 state, and the
    local 2x2 step is exact, so any dt works. Batched over Omega_R, with
    the SU(2)-symmetric g > 0 (it only adds a common phase)."""
    g = Grid((8, 8), (1.0, 1.0), device=CPU)
    Om = torch.tensor([0.05, 0.24, 1.0], dtype=F64)
    eng = BilayerGPE(g, None, g=0.3, Omega_R=Om, n_components=2)
    psi = initial_state(eng, 2.0, batch=3, fractions=(1.0, 0.0))
    dt, t = 0.13, 0.0
    for _ in range(6):
        psi = eng.run_block(psi, 5, dt, order=order)
        t += 5 * dt
        frac = eng.component_norms(psi)[:, 1] / eng.norm(psi)
        assert torch.allclose(frac, torch.sin(Om * t) ** 2, rtol=0, atol=1e-14)
    assert torch.allclose(eng.norm(psi), torch.full((3,), 2.0, dtype=F64), rtol=1e-14)


def test_dressed_state_energy():
    """Uniform two-level ground state: E/N = -sqrt(Omega_R^2 + Delta^2/4)
    (lower eigenvalue of [[Delta/2, Omega_R], [Omega_R, -Delta/2]]), batched
    over (Omega_R, Delta). The start has opposite signs in the two
    components: with Delta = 0 the equal-sign start is the exact upper
    dressed state and imaginary time cannot leave it (next test)."""
    g = Grid((8,), (1.0,), device=CPU)
    Om = torch.tensor([0.24, 0.5, 0.1, 1.0], dtype=F64)
    De = torch.tensor([0.0, 0.3, -0.4, 2.0], dtype=F64)
    eng = BilayerGPE(g, None, g=0.0, Omega_R=Om, delta=De, n_components=2)
    psi0 = initial_state(eng, 1.0, batch=4)
    # (initial_state already starts component 2 at relative phase pi)
    res = relax(eng, psi0, 1.0, tol=1e-12, polish_steps=2000)
    E = res.energy / eng.norm(res.psi)
    exact = -torch.sqrt(Om**2 + De**2 / 4)
    assert torch.allclose(E, exact, rtol=0, atol=1e-12)
    # populations of the lower dressed state: N1/N = (1 - Delta/(2b))/2
    frac1 = eng.component_norms(res.psi)[:, 0] / eng.norm(res.psi)
    assert torch.allclose(frac1, 0.5 * (1 - De / (2 * (-exact))), rtol=0, atol=1e-10)


def test_equal_sign_start_is_upper_dressed_state():
    """Pins a trap of the sign convention: H_rabi = +Omega_R sigma_x, so an
    equal-sign start (relative_phase=0) is the UPPER dressed state at Delta =
    0, an exact eigenstate (residual 0) with E/N = +Omega_R, and relax returns
    it. initial_state therefore defaults to relative phase pi when Omega_R > 0,
    which gives the lower state E/N = -Omega_R."""
    g = Grid((8,), (1.0,), device=CPU)
    eng = BilayerGPE(g, None, g=0.0, Omega_R=0.24, n_components=2)
    res = relax(eng, initial_state(eng, 1.0, relative_phase=0.0), 1.0, tol=1e-12)
    assert float(res.energy[0]) == pytest.approx(+0.24, abs=1e-14)
    assert float(res.residual[0]) < 1e-14
    res = relax(eng, initial_state(eng, 1.0), 1.0, tol=1e-12)
    assert float(res.energy[0]) == pytest.approx(-0.24, abs=1e-12)
