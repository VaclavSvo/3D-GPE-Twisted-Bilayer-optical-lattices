"""The NPSE quasi-2D interaction (bilayer/npse.py): limits, the quartic,
and the sandwich fixed-g2D <= NPSE <= 3D on a small thin-z problem."""

import torch

from twisted_bilayer.engine.npse import NPSE2D
from twisted_bilayer.engine.units import lattice_units

torch.set_num_threads(2)


def test_small_density_limit_is_g2d():
    u = lattice_units()
    f = NPSE2D(u.g2d, u.omega_z)
    n = torch.tensor([1e-6, 1e-4, 1e-2], dtype=torch.float64)
    assert torch.allclose(f.mu_int(n), u.g2d * n, rtol=1e-3)  # contact g2d n
    assert torch.allclose(f.e_int(n), 0.5 * u.g2d * n * n, rtol=1e-3)  # contact g2d n^2 / 2
    assert torch.allclose(f.width(torch.zeros(1, dtype=torch.float64)), torch.ones(1, dtype=torch.float64))


def test_quartic_root_and_envelope():
    u = lattice_units()
    f = NPSE2D(u.g2d, u.omega_z)
    n = torch.tensor([0.3, 3.0, 30.0, 300.0], dtype=torch.float64)
    s = f.width(n)
    x = u.g2d * n / u.omega_z
    assert torch.allclose(s**4 - x * s - 1.0, torch.zeros_like(s), atol=1e-12)
    # mu_int = d e_int / d n (envelope theorem: s is stationary)
    h = 1e-6
    dmu = (f.e_int(n + h) - f.e_int(n - h)) / (2 * h)
    assert torch.allclose(dmu, f.mu_int(n), rtol=1e-7)
    # the width grows with density, the local interaction energy is below g2d n
    assert bool((s[1:] > s[:-1]).all())
    assert bool((f.mu_int(n) < u.g2d * n).all())


def test_npse_is_closer_to_3d_than_fixed_g():
    """Thin-z sandwich at V0 = 8, Omega_R = 1, filling 3 on a (3,1) supercell
    (pocket regime, where the fixed width is 10-20 % off): the NPSE column
    density and mu must be closer to the 3D ones than the fixed-g2d ones."""
    from twisted_bilayer.engine.thin_z import thin_z_check

    r = thin_z_check(
        V0=8.0,
        Omega_R=1.0,
        filling=3.0,
        m=3,
        n=1,
        points_per_a=8,
        Nz=16,
        z_extent_lz=3.5,
        tol=1e-6,
        with_fs=False,
        npse=True,
        verbose=False,
        dtype_relax="complex128",
        dtype_polish="complex128",
    )
    n = r.numbers
    assert n["density_max_rel_dev_npse"] < 0.5 * n["density_max_rel_dev"]
    err_fixed = abs(n["mu_2d"] - n["mu_3d_minus_zpe"])
    err_npse = abs(n["mu_2d_npse"] - n["mu_3d_minus_zpe"])
    assert err_npse < 0.5 * err_fixed
    assert n["n_peak_2d"] < n["n_peak_3d"] * 1.0 + 1e-9  # fixed width underestimates the pocket density


def test_thin_z_stage_tolerances(monkeypatch):
    """thin_z_check's solves: one complex128 stage relaxes straight to tol; a complex64 stage, the only
    one or the first of two, stops at max(tol, 1e-5), its rounding floor; the complex128 polish after a
    complex64 stage goes to tol."""
    import torch

    from twisted_bilayer.engine import thin_z as TZ

    real = TZ.relax

    def run(dr, dp):
        seen = []

        def spy(eng, psi, n, tol=1e-7, **kw):
            seen.append((eng.dtype, tol))
            return real(eng, psi, n, tol=tol, **kw)

        monkeypatch.setattr(TZ, "relax", spy)
        kw = dict(V0=4.0, Omega_R=0.2, m=2, n=1, points_per_a=6, Nz=8, z_extent_lz=3.0, npse=False, verbose=False)
        TZ.thin_z_check(tol=1e-7, dtype_relax=dr, dtype_polish=dp, **kw)
        return seen

    assert {t for _, t in run("complex128", "complex128")} == {1e-7}
    assert {t for _, t in run("complex64", "complex64")} == {1e-5}
    two = run("complex64", "complex128")
    assert {t for d, t in two if d == torch.complex64} == {1e-5} and {t for d, t in two if d == torch.complex128} == {
        1e-7
    }
