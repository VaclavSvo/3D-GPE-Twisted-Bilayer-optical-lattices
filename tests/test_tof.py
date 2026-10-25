"""
Verification of the TOF pipeline (pytest; CPU; ~3 min):  python -m pytest -q tof/tests

(a) brute force vs demodulated method (synthetic lattice x moire x Gaussian)
(b) Talbot revival T_T = L_M^2/pi and half-period contrast inversion
(c) Gaussian free expansion width R0 sqrt(1 + t^2/R0^4) and drift G t
(d) norm conservation per order and in total
(e) mean-field stage: released in-plane kinetic energy (identity with g(t)),
    constant-g energy conservation, Castin-Dum 2D scaling
plus: numpy/torch agreement, t1-consistency with g=0, visibility definition;
(f) mean-field-stage dt gate and high-k monitor, the coherent imaging transfer (in focus,
    pulse push, depth of field), image() of known cosine patterns, satellite-local contrast.
"""

import functools
import math
import time
import warnings

import numpy as np
import pytest
import scipy.fft as sfft
from scipy.integrate import solve_ivp

from twisted_bilayer import tof


def _report(name, **kw):
    print(name, " ".join(f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()))


# ----------------------------------------------------------------------------
# (a) brute force vs demodulated method
# ----------------------------------------------------------------------------
LM = 11.01


def _synthetic_a(dx=1 / 8, L=400.0, R0=8.0):
    """Gaussian(R0) x [1 + 0.4 cos(2pi x/11.01) + 0.4 cos(2pi y/11.01)]^1/2
    x u(x)u(y), u = 1 + 0.458 cos(2 pi x): plane-wave amplitude ratio 0.229 per
    axis, i.e. order weights 0.819 / 0.043 / 0.0022 (the q=0 Bloch state at
    4 E_R)."""
    N = int(round(L / dx))
    x = -L / 2 + dx * np.arange(N)
    ex = np.exp(-(x**2) / (2 * R0**2))
    lat = 1 + 0.458 * np.cos(2 * np.pi * x)
    X, Y = np.meshgrid(x, x, sparse=True)
    psi = (ex * lat)[None, :] * (ex * lat)[:, None]
    psi = psi * np.sqrt(1 + 0.4 * np.cos(2 * np.pi * X / LM) + 0.4 * np.cos(2 * np.pi * Y / LM))
    return psi.astype(complex), dx, (-L / 2, -L / 2), N


@functools.cache
def _case_a():
    t = 10.0
    psi, dx, org, N = _synthetic_a()
    t0 = time.time()
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    ph = np.exp(-0.5j * t * k**2)
    psit = sfft.ifft2(sfft.fft2(psi, workers=-1) * ph[None, :] * ph[:, None], workers=-1)
    t_bf = time.time() - t0
    t0 = time.time()
    orders = tof.demodulate(psi, dx, tof.square_G(), q_cut=np.pi, dx_coarse=0.5, origin=org)
    prop = tof.propagate_orders(orders, t)
    t_dm = time.time() - t0
    return dict(psi=psi, psit=psit, dx=dx, org=org, N=N, orders=orders, prop=prop, t=t, t_bf=t_bf, t_dm=t_dm)


def _canvas(prop, x0, dxc, n):
    """sum_G |chi_G|^2 on the coarse lattice x0 + i dxc (n x n)."""
    can = np.zeros((n, n))
    for e in prop.envs:
        ix, iy = int(round((e.x0 - x0) / dxc)), int(round((e.y0 - x0) / dxc))
        d = e.density()
        sx0, sx1 = max(0, ix), min(n, ix + d.shape[1])
        sy0, sy1 = max(0, iy), min(n, iy + d.shape[0])
        if sx1 > sx0 and sy1 > sy0:
            can[sy0:sy1, sx0:sx1] += d[sy0 - iy : sy1 - iy, sx0 - ix : sx1 - ix]
    return can


def test_a_bruteforce_vs_demodulated():
    c = _case_a()
    nbf = np.abs(c["psit"]) ** 2
    r = 4  # dx_coarse / dx_fine
    nb = nbf[::r, ::r]
    can = _canvas(c["prop"], c["org"][0], c["dx"] * r, nb.shape[0])
    peak = nb.max()
    n_c = nb.shape[0]
    cen = slice(n_c // 2 - 60, n_c // 2 + 60)  # |x|,|y| < 30 a around order 0
    err_c = np.abs(can[cen, cen] - nb[cen, cen]).max() / peak
    err_all = np.abs(can - nb).max() / peak
    # pixel images (the observable)
    im_dm = tof.image(c["prop"], fov=150.0)
    im_bf = tof.image_density(nbf, c["dx"], c["org"], fov=150.0)
    err_img = np.abs(im_dm.img - im_bf.img).max() / im_bf.img.max()
    # full field incl. cross terms and carrier phases
    rec = tof.reconstruct_field(c["prop"], c["dx"], c["N"], c["N"], c["org"])[0]
    err_psi = np.abs(rec - c["psit"]).max() / np.abs(c["psit"]).max()
    _report(
        "a_bruteforce",
        err_central=err_c,
        err_all=err_all,
        err_image=err_img,
        err_field=err_psi,
        dropped=c["orders"].dropped,
        cross_bound=im_dm.cross_bound,
        t_bruteforce_s=c["t_bf"],
        t_demod_s=c["t_dm"],
        weights=np.round(c["orders"].weights(), 4).tolist(),
    )
    assert err_c < 1e-2  # the specification
    assert err_c < 1e-4  # what is actually achieved
    assert err_all < 1e-4
    assert err_img < 1e-4
    assert err_psi < 1e-3  # limited by the |q| > pi content (dropped ~1e-9)


# ----------------------------------------------------------------------------
# (b) Talbot
# ----------------------------------------------------------------------------
def _talbot_field(N=384, eps=0.25, sqrt_form=False):
    L = tof.L_SUPER
    dx = L / N
    x = -L / 2 + dx * np.arange(N)
    X, Y = np.meshgrid(x, x)
    q = 2 * np.pi / tof.L_MOIRE
    b = tof.lattice_G(2)
    b1, b2 = b[5], b[7]  # m=0,n=+1 / m=+1,n=0 ... any two independent vectors
    u = (1 + 0.458 * np.cos(b1[0] * X + b1[1] * Y)) * (1 + 0.458 * np.cos(b2[0] * X + b2[1] * Y))
    if sqrt_form:
        mo = np.sqrt(1 + 0.4 * np.cos(q * X) + 0.4 * np.cos(q * Y))
    else:
        mo = 1 + eps * (np.cos(q * X) + np.cos(q * Y))
    return (u * mo).astype(complex), dx, L, N, (-L / 2, -L / 2)


def test_b_talbot():
    L = tof.L_SUPER
    TT = tof.talbot_time(tof.L_MOIRE)
    assert abs(TT - tof.L_MOIRE**2 / np.pi) < 1e-12
    out = {}
    for sqrt_form in (False, True):
        psi, dx, L, N, org = _talbot_field(sqrt_form=sqrt_form)
        o = tof.demodulate(psi, dx, tof.lattice_G(2), dx_coarse=L / 48 * (1 + 1e-9), origin=org)
        assert o.envs[0].shape == (48, 48)
        full = tof.propagate_orders(o, TT, periodic=True, displace=False)
        half = tof.propagate_orders(o, TT / 2, periodic=True, displace=False)
        quarter = tof.propagate_orders(o, TT / 4, periodic=True, displace=False)
        e_rev = e_half = 0.0
        scale = np.abs(o.envs[0].phi[0]).max()  # errors relative to the order-0 peak
        for e0, eT, eH, _eQ in zip(o.envs, full.envs, half.envs, quarter.envs):
            a0 = e0.phi[0]
            e_rev = max(e_rev, np.abs(eT.phi[0] - a0).max() / scale)
            # half period: phi(r, T/2) = phi(r + (L_M/2, L_M/2)) ; L_M/2 = 12 cells
            sh = np.roll(np.roll(a0, -12, axis=0), -12, axis=1)
            e_half = max(e_half, np.abs(eH.phi[0] - sh).max() / scale)
        # contrast inversion of the zeroth-order density
        n0 = o.envs[0].density()
        nH = half.envs[0].density()
        nQ = quarter.envs[0].density()
        # first moire harmonic of the density (index 2: L = 2 L_M) flips sign
        h0, hH = np.fft.fft2(n0), np.fft.fft2(nH)
        cc = max((hH[0, 2] / h0[0, 2]).real, (hH[2, 0] / h0[2, 0]).real)
        cc_im = max(abs((hH[0, 2] / h0[0, 2]).imag), abs((hH[2, 0] / h0[2, 0]).imag))
        C0 = (n0.max() - n0.min()) / (n0.max() + n0.min())
        CQ = (nQ.max() - nQ.min()) / (nQ.max() + nQ.min())
        # full field incl. drifts and carriers vs brute-force periodic propagation
        k = 2 * np.pi * np.fft.fftfreq(N, dx)
        for tt in (TT, 0.37 * TT):
            ph = np.exp(-0.5j * tt * k**2)
            bf = sfft.ifft2(sfft.fft2(psi) * ph[None, :] * ph[:, None])
            pr = tof.propagate_orders(o, tt, periodic=True)
            rec = tof.reconstruct_field(pr, dx, N, N, org)[0]
            out[(sqrt_form, tt)] = np.abs(rec - bf).max() / np.abs(bf).max()
        _report(
            f"b_talbot_{'sqrt' if sqrt_form else 'cos'}",
            T_T=TT,
            err_revival=e_rev,
            err_halfshift=e_half,
            harmonic1_ratio_half=cc,
            harmonic1_dev=abs(cc + 1),
            contrast_t0=C0,
            contrast_T4=CQ,
            err_field_T=out[(sqrt_form, TT)],
            err_field_037T=out[(sqrt_form, 0.37 * TT)],
            dropped=o.dropped,
        )
        # pure cosine: everything exact. sqrt form: harmonics |q| > pi are dropped
        # (weight ~1e-9) or leak into the neighbouring order's disc, where they
        # sit on the odd q-lattice and do not revive -> ~1e-5 residual.
        tol = 1e-10 if not sqrt_form else 1e-4
        assert e_rev < tol
        assert e_half < tol
        tolh = 1e-10 if not sqrt_form else 1e-6
        assert abs(cc + 1) < tolh and cc_im < tolh  # contrast inversion at T_T/2
        assert max(out[(sqrt_form, TT)], out[(sqrt_form, 0.37 * TT)]) < (1e-10 if not sqrt_form else 1e-3)


# ----------------------------------------------------------------------------
# (c) Gaussian free expansion
# ----------------------------------------------------------------------------
def test_c_gaussian_expansion():
    R0 = 8.0
    dx, L = 1 / 8, 160.0
    N = int(L / dx)
    x = -L / 2 + dx * np.arange(N)
    G = tof.lattice_G(2)[1]  # rotated, off the FFT grid of this box
    g = np.exp(-(x**2) / (2 * R0**2))
    psi = (g[None, :] * g[:, None]) * np.exp(1j * (G[0] * x[None, :] + G[1] * x[:, None]))
    o = tof.demodulate(psi, dx, tof.lattice_G(2), dx_coarse=0.5)
    w = o.weights()
    single = tof.Orders(envs=[o.envs[1]], t=o.t)  # the only populated order
    errs = []
    for t in (10.0, tof.T_TOF, 300.0):
        p = tof.propagate_orders(single, t, eps=1e-16, eps_crop=1e-30)
        e = p.envs[0]
        n = e.density()
        X, Y = np.meshgrid(e.x, e.y)
        Nn = n.sum()
        mx, my = (n * X).sum() / Nn, (n * Y).sum() / Nn
        vx = (n * (X - mx) ** 2).sum() / Nn
        vy = (n * (Y - my) ** 2).sum() / Nn
        Rt2 = R0**2 * (1 + t**2 / R0**4)  # n ~ exp(-r^2/R(t)^2)
        ew = max(abs(vx / (Rt2 / 2) - 1), abs(vy / (Rt2 / 2) - 1))
        ec = max(abs(mx - G[0] * t), abs(my - G[1] * t))
        errs.append((t, ew, ec))
        _report(
            f"c_gauss_t{t:.1f}",
            width_rel_err=ew,
            centroid_err=ec,
            R_t=math.sqrt(Rt2),
            box=p.info["propagate"]["boxes"][0][0],
        )
    assert w[1] > 1 - 1e-12
    for _t, ew, ec in errs:
        assert ew < 1e-6
        assert ec < 1e-6


# ----------------------------------------------------------------------------
# (d) norm conservation
# ----------------------------------------------------------------------------
def test_d_norm_conservation():
    c = _case_a()
    o, p = c["orders"], c["prop"]
    n_in = np.sum(np.abs(c["psi"]) ** 2) * c["dx"] ** 2
    tot_dem = o.norms().sum()
    e_split = abs(tot_dem / n_in - (1 - o.dropped) - 0)
    e_order = np.max(np.abs(p.norms() / o.norms() - 1))
    e_tot = abs(p.norms().sum() / tot_dem - 1)
    # mean-field tail (per-order NLS) also conserves each order's norm
    small = tof.demodulate(*_synthetic_a(L=96.0)[:2], tof.square_G(), dx_coarse=0.5)
    pm = tof.propagate_orders(small, 6.0, g2D=0.1, omega_z=0.0, dphi=0.05)
    e_mf = np.max(np.abs(pm.norms() / small.norms() - 1))
    # Wigner-Seitz filter is an exact partition: nothing dropped except >BZ content
    ows = tof.demodulate(c["psi"], c["dx"], tof.square_G(), dx_coarse=0.5, filt="ws", origin=c["org"])
    _report(
        "d_norm",
        dropped_disc=o.dropped,
        dropped_ws=ows.dropped,
        err_split=e_split,
        err_per_order=e_order,
        err_total=e_tot,
        err_per_order_mftail=e_mf,
        nls_steps=max(pm.info["propagate"]["nls_steps"]),
    )
    assert e_split < 1e-12
    assert e_order < 1e-12
    assert e_tot < 1e-12
    assert e_mf < 1e-10
    assert o.dropped < 1e-8
    assert ows.dropped < o.dropped


# ----------------------------------------------------------------------------
# (e) mean-field stage
# ----------------------------------------------------------------------------
def _tf_ground_state(mu, R, g, dx, L, steps=800, dtau=0.08):
    """2D GPE ground state in V = w^2 r^2/2 with mu = w^2 R^2/2 (imag. time)."""
    N = int(round(L / dx))
    x = -L / 2 + dx * np.arange(N)
    X, Y = np.meshgrid(x, x)
    w = math.sqrt(2 * mu) / R
    V = 0.5 * w**2 * (X**2 + Y**2)
    psi = np.sqrt(np.clip(mu - V, 0, None) / g + 1e-6).astype(complex)
    Natom = np.sum(np.abs(psi) ** 2) * dx * dx
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    Kh = np.exp(-0.25 * dtau * (k[None, :] ** 2 + k[:, None] ** 2))
    for _ in range(steps):
        psi = sfft.ifft2(sfft.fft2(psi, workers=-1) * Kh, workers=-1)
        psi *= np.exp(-dtau * (V + g * np.abs(psi) ** 2))
        psi = sfft.ifft2(sfft.fft2(psi, workers=-1) * Kh, workers=-1)
        psi *= math.sqrt(Natom / (np.sum(np.abs(psi) ** 2) * dx * dx))
    return psi, w, Natom, (-L / 2, -L / 2), x


def _moments(psi, x, dx):
    n = np.abs(psi) ** 2
    X, Y = np.meshgrid(x, x)
    return np.sum(n * (X**2 + Y**2)) * dx * dx


def _kinetic(psi, dx):
    N = psi.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(N, dx)
    pk = sfft.fft2(psi)
    return 0.5 * np.sum(np.abs(pk) ** 2 * (k[None, :] ** 2 + k[:, None] ** 2)) * dx * dx / N**2


@functools.cache
def _gs():
    return _tf_ground_state(mu=4.0, R=20.0, g=tof.G2D, dx=0.2, L=64.0)


def test_e_meanfield_energy():
    psi0, w, Natom, org, x = _gs()
    g, wz = tof.G2D, tof.OMEGA_Z
    dx = 0.2
    I0 = np.sum(np.abs(psi0) ** 4) * dx * dx
    Eint0 = 0.5 * g * I0
    r2_0 = _moments(psi0, x, dx)
    T0 = _kinetic(psi0, dx)

    def cd(gfun, t1):
        sol = solve_ivp(lambda t, y: [y[1], w**2 * gfun(t) / y[0] ** 3], (0, t1), [1.0, 0.0], rtol=1e-11, atol=1e-13)
        return sol.y[0, -1], sol.y[1, -1]

    # (e1) time-dependent g(t), t1 = 1 ms (and 2 ms)
    res = {}
    for t1 in (tof.ms_to_code(1.0), tof.ms_to_code(2.0)):
        mf = tof.meanfield_stage(psi0, dx, t1, g, wz, origin=org, dt="auto", box=None)
        d = mf.diag
        dT = d["T"][-1] - d["T"][0]
        dT_id = tof.released_kinetic_energy(d, g, wz)
        lam, lamd = cd(lambda t: 1 / math.sqrt(1 + (wz * t) ** 2), t1)
        T_cd = 0.5 * lamd**2 * r2_0
        r2_1 = _moments(mf.psi, x, dx)
        res[t1] = (dT, dT_id, T_cd, r2_1 / r2_0, lam**2)
        _report(
            f"e1_mf_g(t)_t1={t1:.2f}",
            dT_kin=dT,
            released_identity=dT_id,
            rel_err_identity=abs(dT / dT_id - 1),
            T_castin_dum=T_cd,
            rel_err_castin_dum=abs(dT / T_cd - 1),
            r2_ratio=r2_1 / r2_0,
            lambda2_cd=lam**2,
            E_int0=Eint0,
            fraction_released=dT / Eint0,
            norm_drift=abs(d["norm"][-1] / d["norm"][0] - 1),
        )
        assert abs(dT / dT_id - 1) < 0.05
        assert abs(dT / T_cd - 1) < 0.05
        assert abs((r2_1 / r2_0 - 1) / (lam**2 - 1) - 1) < 0.05
        assert abs(d["norm"][-1] / d["norm"][0] - 1) < 1e-11
    # (e2) constant g: all interaction energy is eventually released in-plane;
    #      2D TF scaling lambda^2 = 1 + w^2 t^2, T = E_int0 (1 - 1/lambda^2)
    t1 = 1.5 / w
    mf = tof.meanfield_stage(psi0, dx, t1, g, 0.0, origin=org, dt="auto", box=(96.0, 96.0))
    d = mf.diag
    E = d["T"][1:] + 0.5 * g * d["I"]  # T after kick j, I during kick j (O(dt^2) offset)
    E_tot0 = T0 + Eint0
    lam2 = 1 + (w * t1) ** 2
    dT = d["T"][-1] - d["T"][0]
    T_sc = Eint0 * (1 - 1 / lam2)
    _report(
        "e2_mf_const_g",
        t1=t1,
        dT_kin=dT,
        T_scaling=T_sc,
        rel_err_scaling=abs(dT / T_sc - 1),
        energy_drift=abs(E[-1] / E_tot0 - 1),
        released_fraction=dT / Eint0,
    )
    assert abs(E[-1] / E_tot0 - 1) < 1e-3
    assert abs(dT / T_sc - 1) < 0.05


# ----------------------------------------------------------------------------
# extra consistency tests
# ----------------------------------------------------------------------------
def test_meanfield_stage_with_g0_equals_free_propagation():
    psi, dx, org, N = _synthetic_a(L=96.0)
    t1, t = 3.0, 12.0
    mf = tof.meanfield_stage(psi, dx, t1, 0.0, 0.0, origin=org, dt=0.05)
    o1 = tof.demodulate(mf.psi, (mf.dx, mf.dy), tof.square_G(), origin=mf.origin, t=t1)
    p1 = tof.propagate_orders(o1, t)
    o0 = tof.demodulate(psi, dx, tof.square_G(), origin=org)
    p0 = tof.propagate_orders(o0, t)
    im1, im0 = tof.image(p1, fov=120.0), tof.image(p0, fov=120.0)
    err = np.abs(im1.img - im0.img).max() / im0.img.max()
    _report("consistency_t1_g0", err_image=err, mf_box=mf.diag["box"][0])
    assert err < 1e-6


def test_pipeline_end_to_end():
    """tof_pipeline (MF stage -> demod -> MF tail -> image) on a small two-component
    cloud: runs, conserves the imaged atom number, sees the satellites."""
    psi, dx, org, N = _synthetic_a(L=64.0, R0=6.0)
    two = np.stack([psi, -psi]) * 2.0
    out = tof.tof_pipeline(
        two, dx, tof.square_G(), t_tof=12.0, t1=0.5, g2D=tof.G2D, origin=org, image_component=1, tail="mf", dt="auto"
    )
    n_in = np.sum(np.abs(two[1]) ** 2) * dx * dx
    # the only loss is the weight the low-pass filter drops after the MF stage
    # (nonlinear mixing products between the orders; 2e-5 at this high density)
    drop = out["orders_t1"].dropped
    err = abs(out["image"].img.sum() / n_in - (1 - drop))
    sc = tof.satellite_contrast(out["image"], 2 * np.pi * 12.0 / LM)
    _report("pipeline", image_norm_err=err, dropped_after_mf=drop, C_sat=sc["C_sat"], mf_box=out["mf"].diag["box"][0])
    assert err < 1e-8
    assert drop < 1e-4
    assert -1 < sc["C_sat"] < 1


@pytest.mark.parametrize("g2D", [tof.G2D, 0.0])
def test_pipeline_keeps_J_perp_on_through_the_tail(g2D):
    """With J_perp != 0 the coupling acts for the whole t_tof (it used to stop at t1): the interaction is SU(2)
    and the kinetic energy the same in both layers, so a cloud released in layer 1 images as cos^2(J t) (layer
    1) and sin^2(J t) (layer 2) of the J = 0 image, pixel by pixel; also without a stage (g2D = 0)."""
    psi, dx, org, N = _synthetic_a(L=64.0, R0=6.0)
    one = np.stack([psi, 0 * psi]) * 2.0
    J, t = 0.3, 12.0

    def img(Jp, comp):
        kw = dict(t_tof=t, t1=0.5, g2D=g2D, origin=org, image_component=comp, tail="mf", dt="auto", J_perp=Jp)
        return tof.tof_pipeline(one, dx, tof.square_G(), **kw)["image"].img

    ref = img(0.0, 0)
    for comp, w in ((0, math.cos(J * t) ** 2), (1, math.sin(J * t) ** 2)):
        err = np.abs(img(J, comp) - w * ref).max() / ref.max()
        assert err < 1e-10, (comp, err)
    o = tof.demodulate(one, dx, tof.square_G(), origin=org)
    assert tof.spin_rotate_orders(o, J, 0.0) is o and tof.spin_rotate_orders(o, 0.0, t) is o


def test_mf_tail_records_and_guards_its_high_k_weight():
    """The comoving NLS of the tail is not limited by the kinetic phase per step (~30 rad here); its
    split-step instability shows as growth of the weight at |k| > 0.75 k_Nyq, which is recorded per order and
    raises above 10x. At the experiment's density the weight stays flat (~1e-13); a 30x denser cloud with
    10x larger steps blows up (weight 0.66) and is refused."""
    psi, dx, org, N = _synthetic_a(L=64.0, R0=6.0)
    o = tof.demodulate(np.stack([psi, -psi]) * 2.0, dx, tof.square_G(), origin=org)
    p = tof.propagate_orders(o, 12.0, g2D=tof.G2D).info["propagate"]
    assert len(p["nls_highk"]) == len(p["nls_kin_phase_max"]) == sum(n > 0 for n in p["nls_steps"]) > 0
    assert all(h1 < 10 * max(h0, 1e-14) and h1 < 1e-11 for h0, h1 in p["nls_highk"]), p["nls_highk"]
    assert max(p["nls_kin_phase_max"]) > math.pi  # the step is not kinetic-phase limited, by design
    dense = tof.Orders(
        envs=[tof.Envelope(G=e.G, phi=3 * e.phi, x0=e.x0, y0=e.y0, dx=e.dx, dy=e.dy) for e in o.envs], t=o.t
    )
    kw = dict(g2D=10 * tof.G2D, dphi=0.5, dt_max=2.0)
    with pytest.raises(RuntimeError, match="high-k weight"):
        tof.propagate_orders(dense, 12.0, **kw)
    bad = tof.propagate_orders(dense, 12.0, highk_check="off", **kw).info["propagate"]["nls_highk"]
    assert max(h1 for _, h1 in bad) > 0.1


def test_degenerate_inputs_fail_clearly_and_resampling_keeps_real_fields_real():
    """A zero field, zero-norm orders and an all-NaN image are clear ValueErrors (they were numpy errors or
    NaN); zero padding splits the even grid's Nyquist coefficient, so the band-limited interpolant of the
    real cos(pi x / dx) is real (it had an imaginary part of 1)."""
    from twisted_bilayer.tof import tof as tof_module

    x = np.arange(8.0)
    f = (np.cos(np.pi * x)[None, :] * np.cos(np.pi * x)[:, None]).astype(complex)
    g, (dxn, dyn), drop = tof.fourier_resample(f, 1.0, (16, 16))
    xf = 0.5 * np.arange(16)
    assert dxn == dyn == 0.5 and drop == 0.0
    assert np.abs(g - np.cos(np.pi * xf)[None, :] * np.cos(np.pi * xf)[:, None]).max() < 1e-13
    smooth = np.exp(-((x - 3.5) ** 2) / 4)[None, :] * np.ones((8, 1))
    g2, _, _ = tof.fourier_resample(smooth.astype(complex), 1.0, (16, 16))
    assert np.abs(g2[::2, ::2] - smooth).max() < 1e-13  # the original samples are kept
    with pytest.raises(ValueError, match="zero or not finite"):
        tof_module._support(np.zeros((1, 16, 16), complex), 0.5, 0.5, 1e-14, 1e-6)
    env = tof.Envelope(G=np.zeros(2), phi=np.zeros((1, 4, 4), complex), x0=0.0, y0=0.0, dx=1.0, dy=1.0)
    with pytest.raises(ValueError, match="total norm"):
        tof.Orders(envs=[env, env]).weights()
    assert len(tof.Orders(envs=[]).weights()) == 0
    xc = np.arange(64) - 32.0
    im = tof.TOFImage(img=np.full((64, 64), np.nan), xc=xc, yc=xc, pixel=1.0)
    with pytest.raises(ValueError, match="no finite disk mean"):
        tof.satellite_contrast(im, 16.0)
    with pytest.raises(ValueError, match="no finite profile bin"):
        tof.satellite_contrast_profile(im, 16.0, bin_width=4.0)


def test_torch_backend():
    torch = pytest.importorskip("torch")
    nthr = torch.get_num_threads()
    torch.set_num_threads(1)  # avoid OpenMP oversubscription on shared CPUs
    try:
        _torch_checks(torch)
    finally:
        torch.set_num_threads(nthr)


def _torch_checks(torch):
    psi, dx, org, N = _synthetic_a(L=64.0, R0=5.0)
    o = tof.demodulate(psi, dx, tof.square_G(), origin=org)
    ot = tof.Orders(
        envs=[tof.Envelope(G=e.G, phi=torch.from_numpy(e.phi), x0=e.x0, y0=e.y0, dx=e.dx, dy=e.dy) for e in o.envs],
        t=o.t,
    )
    for kw in (dict(), dict(g2D=0.3, omega_z=tof.OMEGA_Z)):
        pn = tof.propagate_orders(o, 6.0, **kw)
        pt = tof.propagate_orders(ot, 6.0, **kw)
        assert isinstance(pt.envs[0].phi, torch.Tensor)
        err = max(np.abs(a.phi - b.phi.numpy()).max() for a, b in zip(pn.envs, pt.envs))
        _report(f"torch_propagate_{'mf' if kw else 'lin'}", max_abs_diff=err)
        assert err < 1e-10
    mfn = tof.meanfield_stage(psi, dx, 0.5, tof.G2D * 5, tof.OMEGA_Z, origin=org, dt="auto")
    mft = tof.meanfield_stage(torch.from_numpy(psi), dx, 0.5, tof.G2D * 5, tof.OMEGA_Z, origin=org, dt="auto")
    err = np.abs(mfn.psi - mft.psi.numpy()).max() / np.abs(mfn.psi).max()
    _report("torch_meanfield", rel_diff=err)
    assert err < 1e-10


def test_visibility_definition():
    P = tof.PIXEL_TOF
    xc = P * np.arange(-40, 41)
    X, Y = np.meshgrid(xc, xc)
    d = 13 * P
    gs = lambda x0, y0: np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * (0.6 * P) ** 2))
    img = 0.1 + gs(0, 0) + 0.5 * (gs(d, 0) + gs(-d, 0) + gs(0, d) + gs(0, -d))
    im = tof.TOFImage(img=img, xc=xc, yc=xc, pixel=P)
    v = tof.visibility(im, d)
    Pav = 0.5 * (1.1 + 0.6)
    Vexp = (Pav - 0.1) / (Pav + 0.1)
    _report("visibility_def", V=v["V"], V_expected=Vexp, V_sat=v["V_sat"])
    assert abs(v["V"] - Vexp) < 1e-6
    assert abs(v["V_sat"] - 0.5 / 0.7) < 1e-6 and v["n_resolved"] == 4
    # a monotonic tail (no satellite) must be flagged as unresolved
    im2 = tof.TOFImage(img=0.1 + gs(0, 0) * 0 + np.exp(-(X**2 + Y**2) / (2 * (8 * P) ** 2)), xc=xc, yc=xc, pixel=P)
    v2 = tof.visibility(im2, d)
    assert v2["n_resolved"] == 0


# ============================================================================
# (f) time step, high-k monitor, imaging transfer, metrics
# ============================================================================
# ----------------------------------------------------------------------------
# mean-field stage time step
# ----------------------------------------------------------------------------
def _cloud(L=48.0, dx=1 / 8, R0=5.0, nbar=2.0):
    N = int(round(L / dx))
    x = -L / 2 + dx * np.arange(N)
    X, Y = np.meshgrid(x, x)
    u = (1 + 0.458 * np.cos(2 * np.pi * X)) * (1 + 0.458 * np.cos(2 * np.pi * Y)) / 1.104882
    psi = np.sqrt(nbar) * np.exp(-(X**2 + Y**2) / (2 * R0**2)) * u
    return np.stack([psi, -psi]).astype(complex) / math.sqrt(2), dx, (-L / 2, -L / 2)


def test_kinetic_phase_and_stable_dt():
    """8 pts/a: k2_max = 2 (8 pi)^2 = 1263.3 -> 6.32 rad/step at dt = 0.01;
    dt for 2 rad = 3.166e-3."""
    ph = tof.kinetic_phase_per_step(0.01, 1 / 8, 1 / 8, 1650, 1650)
    assert abs(ph - 0.5 * 0.01 * 2 * (8 * np.pi) ** 2) < 1e-9
    assert abs(ph - 6.3165) < 1e-3
    dt = tof.stable_dt(1 / 8, 1 / 8, 1650, 1650)
    assert abs(dt - 4.0 / (2 * (8 * np.pi) ** 2)) < 1e-12
    assert tof.stable_dt(1.0, 1.0, 64, 64) == tof.MF_DT_CAP  # coarse grid: capped at 0.01


def test_meanfield_dt_gate():
    psi, dx, org = _cloud()
    t1 = 0.3
    with pytest.raises(ValueError):
        tof.meanfield_stage(psi, dx, t1, tof.G2D, tof.OMEGA_Z, origin=org, dt=0.01)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tof.meanfield_stage(psi, dx, t1, tof.G2D, tof.OMEGA_Z, origin=org, dt=0.01, dt_check="warn")
        assert any(issubclass(x.category, RuntimeWarning) for x in w)
    # g = 0: the split step is exact for any dt -> no gate
    tof.meanfield_stage(psi, dx, t1, 0.0, tof.OMEGA_Z, origin=org, dt=0.05)
    mf = tof.meanfield_stage(psi, dx, t1, tof.G2D, tof.OMEGA_Z, origin=org)  # dt='auto'
    assert mf.diag["kin_phase_per_step"] <= tof.MF_PHASE_TARGET + 1e-12
    assert mf.diag["dt"] <= 0.01
    hk0, hk1 = mf.diag["highk_weight"]
    # smooth Bloch cloud: only the physical nonlinear mixing products (3G at |k| = 6 pi =
    # 0.75 k_Nyq) reach the monitored band, at the 1e-10 level
    assert hk0 < 1e-11 and hk1 < 1e-8, (hk0, hk1)


def test_meanfield_auto_dt_converged():
    """dt='auto' (2 rad/step) vs dt/4: fine-grid densities (lattice scale included) agree
    to < 3e-5 of the peak (short stage, strong coupling 5 g2D; measured 1.3e-5, O(dt^2))."""
    psi, dx, org = _cloud(L=40.0, R0=4.0)
    t1 = 0.6
    a = tof.meanfield_stage(psi, dx, t1, 5 * tof.G2D, tof.OMEGA_Z, origin=org)
    b = tof.meanfield_stage(psi, dx, t1, 5 * tof.G2D, tof.OMEGA_Z, origin=org, dt=0.25 * a.diag["dt"])
    na, nb = np.abs(a.psi[1]) ** 2, np.abs(b.psi[1]) ** 2
    err = np.abs(na - nb).max() / nb.max()
    assert err < 3e-5, err


# ----------------------------------------------------------------------------
# transfer functions
# ----------------------------------------------------------------------------
def test_coherent_insitu_transfer_values():
    """In focus 1 at 4.35 um (f = 0.23/um < NA/lambda = 0.885/um), 0 for the
    395 nm lattice; MTF_eff(4.35 um) ~ 0.97 / 0.91 / 0.63-0.73 at s = 0.01 / 0.03 / 0.1."""
    assert tof.insitu_transfer(4.35, s=0.0) == 1.0
    assert tof.insitu_transfer(0.395, s=0.0) == 0.0
    assert abs(tof.NA_IMG / tof.LAMBDA_IMG_UM - 0.8843) < 1e-3
    m = {s: tof.insitu_transfer(4.35, s=s) for s in (0.01, 0.03, 0.1)}
    assert 0.96 < m[0.01] < 0.985, m
    assert 0.89 < m[0.03] < 0.93, m
    assert 0.63 < m[0.1] < 0.73, m


def test_transfer_is_average_of_product():
    """The pulse average is <diffusion x cos(defocus)>, not <diffusion> <cos>; check against
    a direct trapezoid integration and that the two differ at s = 0.1."""
    s, P = 0.1, 4.35
    f = 1 / P
    t = np.linspace(0, tof.T_PULSE, 20001)
    R = 0.5 * tof.GAMMA_D2 * s / (1 + s)
    z = 0.5 * R * tof.V_RECOIL * t**2 * 1e6
    x2 = R * tof.V_RECOIL**2 * t**3 / 9 * 1e12
    dif = np.exp(-0.5 * (2 * np.pi * f) ** 2 * x2)
    dfc = np.cos(np.pi * tof.LAMBDA_IMG_UM * z * f**2)
    prod = np.trapezoid(dif * dfc, t) / tof.T_PULSE
    sep = (np.trapezoid(dif, t) / tof.T_PULSE) * (np.trapezoid(dfc, t) / tof.T_PULSE)
    H = tof.insitu_transfer(P, s=s)
    assert abs(H - prod) < 2e-5, (H, prod)
    assert abs(prod - sep) > 1e-3


def test_tof_depth_of_field():
    """sigma_z = l_z sqrt(1 + w_z^2 t^2)/sqrt(2) = 27 um at 18 ms; coherent DOF
    transfer 0.002 at 4.35 um, 0.98 at 19 um."""
    sz = tof.sigma_z_tof_um(18.0)
    assert abs(sz - 27.27) < 0.05, sz
    assert 0.001 < tof.tof_transfer(4.35, 18.0) < 0.003
    assert abs(tof.tof_transfer(19.0, 18.0) - 0.98) < 0.005
    lam = tof.LAMBDA_IMG_UM
    ref = math.exp(-0.5 * (math.pi * lam * sz / 4.35**2) ** 2)
    assert abs(tof.tof_transfer(4.35, 18.0) - ref) < 1e-12


def _cos_density(L=64.0, N=256, eps1=0.3, m1=8, eps2=0.2, m2=24):
    dx = L / N
    x = -L / 2 + dx * np.arange(N)
    k1, k2 = 2 * np.pi * m1 / L, 2 * np.pi * m2 / L
    n = 1 + eps1 * np.cos(k1 * x) + eps2 * np.cos(k2 * x)
    return np.broadcast_to(n[None, :], (N, N)).copy(), dx, (-L / 2, -L / 2), k1, k2


def test_image_coherent_cosine():
    """image_density with the coherent model: a cosine below the cutoff is transferred
    with H(k) sinc(k P/2) (pixel integral), one above the cutoff (f = 0.95/um) vanishes;
    push + depth of field multiply by coherent_transfer_k; the atom number is kept."""
    n, dx, org, k1, k2 = _cos_density()
    P = 1.0
    assert k1 / (2 * np.pi * tof.A_UM) < 0.885 < k2 / (2 * np.pi * tof.A_UM)
    for kw, H1 in (
        (dict(), 1.0),
        (dict(s_sat=0.1, dof=3.0), float(tof.coherent_transfer_k(k1, s=0.1, sigma_z_um=3.0 * tof.A_UM))),
    ):
        im = tof.image_density(n, dx, org, pixel=P, fov=(16.0, 16.0), **kw)
        xc = im.xc
        expect = P * P * (1 + 0.3 * H1 * np.sinc(k1 * P / (2 * np.pi)) * np.cos(k1 * xc))
        j = len(im.yc) // 2
        err = np.abs(im.img[j] - expect).max()
        assert err < 1e-9, (kw, err)
        assert 0.1 < H1 <= 1.0


def test_image_orders_dof_from_time():
    """image(orders, dof=True) uses sigma_z of the pancake expanded for orders.t."""
    n, dx, org, k1, _ = _cos_density(eps2=0.0)
    o = tof.Orders(
        envs=[tof.Envelope(G=np.zeros(2), phi=np.sqrt(n)[None].astype(complex), x0=org[0], y0=org[1], dx=dx, dy=dx)],
        t=tof.ms_to_code(2.0),
    )
    im = tof.image(o, pixel=1.0, fov=(16.0, 16.0), dof=True)
    sz_a = tof.sigma_z_tof_um(2.0) / tof.A_UM
    assert abs(im.info["sigma_z_a"] - sz_a) < 1e-12
    H1 = float(tof.coherent_transfer_k(k1, sigma_z_um=sz_a * tof.A_UM))
    expect = 1 + 0.3 * H1 * np.sinc(k1 / (2 * np.pi)) * np.cos(k1 * im.xc)
    j = len(im.yc) // 2
    assert np.abs(im.img[j] - expect).max() < 1e-9
    assert H1 < 0.99


# ----------------------------------------------------------------------------
# satellite-local contrast
# ----------------------------------------------------------------------------
def _spots(sig_c=2.0, sig_s=2.0, A_s=0.2, b=0.01, d_px=20, half=45, blob=None):
    P = tof.PIXEL_TOF
    xc = P * np.arange(-half, half + 1)
    X, Y = np.meshgrid(xc, xc)
    d = d_px * P
    g = lambda x0, y0, s: np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * (s * P) ** 2))  # noqa: E731
    if blob is not None:
        img = b + g(0, 0, blob)
    else:
        img = b + g(0, 0, sig_c) + A_s * (g(d, 0, sig_s) + g(-d, 0, sig_s) + g(0, d, sig_s) + g(0, -d, sig_s))
    return tof.TOFImage(img=img, xc=xc, yc=xc, pixel=P), d


def test_satellite_contrast_separated():
    """Separated satellites: C_sat = (P1 - P_min)/(P1 + P_min) with P1 = b + A 2 s^2 (1 -
    e^{-r^2/2s^2})/r^2 (disk mean of a centred Gaussian, r = d/8) and P_min = b
    (non-overlapping disks); the profile version is positive and resolved."""
    im, d = _spots()
    r = d / 8  # default radius
    sc = tof.satellite_contrast(im, d)
    s = 2.0 * im.pixel
    P1 = 0.01 + 0.2 * 2 * s * s * (1 - math.exp(-r * r / (2 * s * s))) / (r * r)
    Pmin = 0.01
    C = (P1 - Pmin) / (P1 + Pmin)
    assert sc["resolved"] and sc["n_resolved"] == 4
    assert abs(sc["C_sat"] - C) < 0.02, (sc["C_sat"], C)
    for e in ((1.0, 0.0), (0.0, 1.0)):
        assert 2 * r - 1e-9 <= sc[e]["s_min"] <= d - 2 * r + 1e-9
    pr = tof.satellite_contrast_profile(im, d)
    assert pr["resolved"] and pr["C_prof"] > 0.5, pr["C_prof"]
    with pytest.raises(ValueError):
        tof.satellite_contrast(im, d, radius=0.3 * d)


def test_satellite_contrast_blob_flagged():
    """A pure Gaussian blob (no satellite): C_sat < 0 and 'no satellite' in both versions."""
    im, d = _spots(blob=6.0, b=0.0)
    sc = tof.satellite_contrast(im, d)
    pr = tof.satellite_contrast_profile(im, d)
    assert sc["C_sat"] < 0 and not sc["resolved"] and sc["n_resolved"] == 0
    assert sc["C_sat_clip"] == 0.0
    assert not pr["resolved"]


def test_satellite_contrast_weak_nearfield_satellite():
    """A weak satellite (amplitude 0.03 of the central peak) whose copy is as wide as the
    central one (sigma = 3 px, d = 13 px, as for the N = 600 cloud at 18 ms: dip at ~0.75 d):
    both satellite-local contrasts are positive and resolved (a simulated N = 600 TOF image
    whose disk profile dips at 36 a and peaks at 48 a is this case)."""
    im, d = _spots(sig_c=3.0, sig_s=3.0, A_s=0.03, b=0.0, d_px=13)
    sc = tof.satellite_contrast(im, d)
    pr = tof.satellite_contrast_profile(im, d)
    assert sc["resolved"] and 0.05 < sc["C_sat"] < 0.6, sc["C_sat"]
    assert pr["resolved"] and pr["C_prof"] > 0.02, pr["C_prof"]  # measured 0.045 (C_sat 0.091)


# ----------------------------------------------------------------------------
# the release width of the mean-field coupling
# ----------------------------------------------------------------------------
def test_release_width_coupling():
    """g(t) of a pancake released from width s0 l_z: s0 = 1 is the oscillator formula
    g2D / sqrt(1 + w^2 t^2) exactly; the integral is the closed form of the quadrature; g(0) = g2D / s0
    and g -> g2D s0 / (w t) late (the wider start expands more slowly); the NPSE width of a density is
    1 for n -> 0 and solves the quartic of engine.npse; meanfield_stage records s0 and with s0 = 1 is
    unchanged."""
    from scipy.integrate import quad

    from twisted_bilayer.engine.npse import NPSE2D

    g2, wz = 0.7, 1.3
    t = np.linspace(0.0, 30.0, 7)
    assert np.allclose(tof.g_of_t(t, g2, wz), g2 / np.sqrt(1 + (wz * t) ** 2), rtol=0, atol=1e-15)
    assert np.allclose(tof.g_integral(t, g2, wz), g2 * np.arcsinh(wz * t) / wz, rtol=0, atol=1e-15)
    for s0 in (1.0, 1.15):
        num = quad(lambda u, s=s0: tof.g_of_t(u, g2, wz, s), 0.0, 12.0, epsabs=1e-13)[0]
        assert tof.g_integral(12.0, g2, wz, s0) == pytest.approx(num, rel=1e-10)
        assert tof.g_of_t(0.0, g2, wz, s0) == pytest.approx(g2 / s0)
        assert tof.g_of_t(1e6, g2, wz, s0) == pytest.approx(g2 * s0 / (wz * 1e6), rel=1e-9)
    assert tof.g_integral(2.0, g2, 0.0, 1.15) == pytest.approx(g2 * 2.0 / 1.15)
    assert tof.npse_release_width(np.full(5, 1e-9), g2, wz) == pytest.approx(1.0, abs=1e-8)
    n = np.array([3.0, 11.6])
    import torch

    s_engine = NPSE2D(g2, wz).width(torch.as_tensor(n)).numpy()
    w = n * n
    assert tof.npse_release_width(n, g2, wz) == pytest.approx(float((s_engine * w).sum() / w.sum()), rel=1e-12)
    psi, dx, org, _ = _synthetic_a(L=64.0, R0=6.0)
    a = tof.meanfield_stage(psi, dx, 0.5, 5 * tof.G2D, tof.OMEGA_Z, origin=org)
    b = tof.meanfield_stage(psi, dx, 0.5, 5 * tof.G2D, tof.OMEGA_Z, origin=org, s0=1.0)
    c = tof.meanfield_stage(psi, dx, 0.5, 5 * tof.G2D, tof.OMEGA_Z, origin=org, s0=1.2)
    assert a.diag["s0"] == 1.0 and np.array_equal(np.asarray(a.psi), np.asarray(b.psi))
    assert c.diag["s0"] == 1.2 and np.abs(np.asarray(c.psi) - np.asarray(a.psi)).max() > 1e-6
