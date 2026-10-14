"""bilayer.grid.Grid: the conventions every other module relies on
(dx = L/N with no endpoint, origin on the grid for even N, FFT-ordered k,
(B, C, *N) layout with the spatial axes last)."""

import math

import pytest
import torch

from twisted_bilayer.engine.grid import Grid

CPU = torch.device("cpu")
GRIDS = [((16,), (3.0,)), ((12, 20), (2.5, 4.0)), ((8, 10, 6), (1.0, 2.0, 3.0))]


@pytest.mark.parametrize("N,L", GRIDS)
def test_spacing_is_L_over_N(N, L):
    g = Grid(N, L, device=CPU)
    for ax in range(g.d):
        assert g.dx[ax] == L[ax] / N[ax]
        x = g.x64[ax]
        assert torch.allclose(
            x[1:] - x[:-1], torch.full((N[ax] - 1,), g.dx[ax], dtype=torch.float64), rtol=0, atol=1e-14
        )
        # periodic box is exactly L: the point after the last one is x_0 + L
        assert abs(float(x[-1] + g.dx[ax] - (x[0] + L[ax]))) < 1e-14
    assert g.dV == pytest.approx(math.prod(g.dx), rel=1e-15)
    assert g.volume == pytest.approx(math.prod(L), rel=1e-15)
    assert g.numel == math.prod(N)
    assert g.sdims == tuple(range(-g.d, 0))


@pytest.mark.parametrize("N,L", GRIDS)
def test_origin_on_grid_for_even_N(N, L):
    g = Grid(N, L, device=CPU)
    for ax in range(g.d):
        assert N[ax] % 2 == 0
        assert float(g.x64[ax][N[ax] // 2]) == 0.0  # exactly, AA point on the grid
        assert float(g.x64[ax][0]) == pytest.approx(-L[ax] / 2, abs=1e-14)


def test_odd_N_has_no_point_at_minus_half_L():
    g = Grid((9,), (3.0,), device=CPU)
    assert float(g.x64[0][4]) == 0.0
    assert float(g.x64[0][0]) == pytest.approx(-4 * 3.0 / 9, abs=1e-15)


@pytest.mark.parametrize("N,L", GRIDS)
def test_k_vectors(N, L):
    g = Grid(N, L, device=CPU)
    for ax in range(g.d):
        n, l = N[ax], L[ax]
        k = g.k64[ax]
        expect = torch.tensor([2 * math.pi * (j if j < n // 2 else j - n) / l for j in range(n)], dtype=torch.float64)
        assert torch.allclose(k, expect, rtol=0, atol=1e-13)
        assert float(k[1]) == pytest.approx(2 * math.pi / l, rel=1e-14)
        assert float(k[n // 2]) == pytest.approx(-math.pi / g.dx[ax], rel=1e-14)
        assert g.k_nyquist[ax] == pytest.approx(math.pi / g.dx[ax], rel=1e-15)
    assert g.k_sq_max == pytest.approx(sum((math.pi / dx) ** 2 for dx in g.dx), rel=1e-14)


def test_k_shifted():
    g = Grid((8, 8), (2.0, 2.0), device=CPU)
    k0 = g.k_shifted(1)
    assert k0.shape == (1, 1, 1, 8)
    assert torch.equal(g.k_shifted(1, 0.0), k0)
    assert torch.allclose(g.k_shifted(1, 0.3), k0 + 0.3, atol=1e-15)
    q = torch.tensor([0.1, 0.2], dtype=torch.float64).reshape(2, 1, 1, 1)
    kb = g.k_shifted(0, q)
    assert kb.shape == (2, 1, 8, 1)
    assert torch.allclose(kb[1, 0, :, 0], g.k[0] + 0.2, atol=1e-15)


@pytest.mark.parametrize("N,L", GRIDS)
def test_integrate_constant(N, L):
    g = Grid(N, L, device=CPU)
    f = torch.full((3, 2) + g.N, 2.5, dtype=torch.float64)
    out = g.integrate(f)
    assert out.shape == (3, 2)
    assert torch.allclose(out, torch.full((3, 2), 2.5 * g.volume, dtype=torch.float64), rtol=1e-14)


@pytest.mark.parametrize("N,L", GRIDS)
def test_fft_round_trip_and_plane_wave(N, L):
    g = Grid(N, L, device=CPU)
    gen = torch.Generator().manual_seed(1)
    psi = torch.randn((2, 2) + g.N, dtype=torch.complex128, generator=gen)
    assert torch.allclose(g.ifft(g.fft(psi)), psi, rtol=0, atol=1e-14)
    # the FFT acts on the spatial axes only: a plane wave with the first
    # nonzero k on axis 0 lands in bin 1 of that axis, all batch members alike
    pw = torch.exp(1j * g.k[0][1] * g.coord(0)).expand((2, 2) + g.N).to(torch.complex128)
    pk = g.fft(pw)
    idx = (slice(None), slice(None)) + (1,) + (0,) * (g.d - 1)
    assert torch.allclose(pk[idx].abs(), torch.full((2, 2), float(g.numel), dtype=torch.float64))
    assert float(pk.abs().sum()) == pytest.approx(4 * g.numel, rel=1e-12)


def test_axis_view_shapes():
    g = Grid((4, 6, 8), (1.0, 1.0, 1.0), device=CPU)
    for ax in range(3):
        v = g.x[ax]
        assert g.axis_view(v, ax).shape == tuple(g.N[ax] if i == 2 + ax else 1 for i in range(5))
        assert g.axis_view(v, ax, lead=0).shape == tuple(g.N[ax] if i == ax else 1 for i in range(3))
        assert g.coord(ax, lead=1).shape == tuple(g.N[ax] if i == 1 + ax else 1 for i in range(4))


def test_to_dtype():
    g = Grid((8, 8), (2.0, 3.0), device=CPU)
    assert g.to(torch.float64) is g
    g32 = g.to(torch.float32)
    assert g32.N == g.N and g32.L == g.L and g32.dx == g.dx
    assert g32.real_dtype == torch.float32
    assert all(t.dtype == torch.float32 for t in g32.x + g32.k)
    assert all(t.dtype == torch.float64 for t in g32.x64 + g32.k64)  # host masters stay fp64
    assert torch.allclose(g32.x[1].double(), g.x[1], atol=1e-6)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Grid((8, 8), (1.0,))
    with pytest.raises(ValueError):
        Grid((1,), (1.0,))
    with pytest.raises(ValueError):
        Grid((4, 4, 4, 4), (1.0,) * 4)
