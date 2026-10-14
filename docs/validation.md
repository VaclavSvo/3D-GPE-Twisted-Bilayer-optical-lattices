# Validation

**89 of 89 physics gates pass** on CPU (2 threads), torch 2.14.1+cu130 (full settings, 383 s).

A gate is a pass/fail check of the engine against something known independently: an analytic result, an exact identity of the Hamiltonian, or a number reproduced by a separately written code. Thresholds and their sources are in `src/twisted_bilayer/gates/references.py`; a threshold is never loosened to make a gate pass. Regenerate this page with

```bash
twisted-bilayer gates --full --trapped --out outputs/gates
python scripts/validation_report.py outputs/gates/gates.json
```

| kind | gates |
| --- | --- |
| conservation | 18 |
| analytic | 17 |
| anchor | 17 |
| convergence | 13 |
| identity | 13 |
| preflight | 11 |

## A. Exact single-particle results

Harmonic-oscillator ground state and breathing, free Gaussian expansion, band bottoms against plane-wave diagonalisation, Bloch weights, and two-level Rabi oscillation with and without detuning.

| gate | result | |
| --- | --- | --- |
| A1 HO ground state: E0 vs hbar omega (2D) | 0.202642367 vs 0.202642367: rel err 0.000e+00 (tol 1.0e-04) | pass |
| A1 HO ground state: rms x width vs 1/sqrt(2 m omega) | 0.707106781 vs 0.707106781: rel err 1.220e-13 (tol 1.0e-04) | pass |
| A1 HO ground state: rms y width vs 1/sqrt(2 m omega) | 0.707106781 vs 0.707106781: rel err 1.220e-13 (tol 1.0e-04) | pass |
| A1 HO ground state: final residual | 4.967e-14 (target 2.0e-11) | pass |
| A2 HO breathing: period of <r^2>(t) vs 2 pi / 2 omega | 15.5031381 vs 15.5031383: rel err 1.432e-08 (tol 1.0e-03) | pass |
| A2 HO breathing: norm | max rel drift 3.240e-13 (tol 1.0e-09, ref=1) | pass |
| A2 HO breathing: energy | max rel drift 2.618e-08 (tol 1.0e-07, ref=0.223159907) | pass |
| A2 HO breathing: N2_over_N (J_perp = 0) | abs drift 0.000e+00 (tol 1.000e-08 = 1.0e-08 x scale 1.000e+00) | pass |
| A2 HO breathing: high-k weight | max/initial = 4e-09 (tol 10); initial 3.088e-25, max 4.001e-23 | pass |
| A3 free expansion: sigma(t) vs free_gaussian_sigma | max rel err 1.835e-13 over 11 frames (sigma 0.5000 -> 1.1180, x2.236; tol 1e-4) | pass |
| A3 free expansion: growth >= 2x | sigma(t_end)/sigma0 = 2.2361 (reference 2.2361) | pass |
| A3 free expansion: norm | max rel drift 3.997e-15 (tol 1.0e-09, ref=1) | pass |
| A3 free expansion: energy | max rel drift 5.479e-15 (tol 1.0e-07, ref=0.202642367) | pass |
| A3 free expansion: N2_over_N (J_perp = 0) | abs drift 0.000e+00 (tol 1.000e-08 = 1.0e-08 x scale 1.000e+00) | pass |
| A3 free expansion: high-k weight | max/initial = 7.28e-06 (tol 10); initial 7.283e-20, max 7.283e-20 | pass |
| A4 band bottom: 1D V0 sin^2, E/N vs plane waves | 1.54486139 E_R vs 1.5448614 E_R: rel err 2.630e-09 (tol 1.0e-06) | pass |
| A4 band bottom: 1D mu = E/N (linear) | 1.54486139 vs 1.54486139: rel err 0.000e+00 (tol 1.0e-10) | pass |
| A4 band bottom 1D: final residual | 8.644e-14 (target 2.0e-11) | pass |
| A4 band bottom: 2D square lattice, E/N vs 2 x 1D | 3.08972278 vs 3.08972279: rel err 2.630e-09 (tol 2.0e-06) | pass |
| A4 band bottom: 2D square lattice, E/N vs 3.089723 anchor | 3.08972278 E_R vs 3.089723 E_R: rel err 7.002e-08 (tol 2.0e-06) | pass |
| A4 band bottom 2D: final residual | 2.220e-14 (target 2.0e-11) | pass |
| A4 band bottom: sqrt(485) supercell lattice 1, 176^2, E/N vs anchor | 3.08972279 E_R vs 3.089723 E_R: rel err 6.740e-08 (tol 2.0e-06) | pass |
| A4 band bottom: sqrt(485) supercell lattice 1, 176^2, E/N vs 2 x 1D plane waves | 3.08972279 vs 3.08972279: rel err 6.229e-12 (tol 2.0e-06) | pass |
| A5 Bloch weights: square lattice k = 0 weight vs \|c0\|^4 | 0.820528344 vs 0.820528348: rel err 4.925e-09 (tol 1.0e-03) | pass |
| A5 Bloch weights: square lattice primary peak (worst of 4) vs \|c0 c1\|^2 | 0.042493348 vs 0.0424933475: rel err 1.293e-08 (tol 1.0e-03) | pass |
| A5 Bloch weights: square lattice diagonal peak (worst of 4) vs \|c1\|^4 | 0.00220063651 vs 0.00220063644: rel err 3.078e-08 (tol 1.0e-03) | pass |
| A5 Bloch weights: Franck-Condon <F> = \|c0\|^4 vs 0.8205 anchor | 0.820528344 vs 0.8205: rel err 3.454e-05 (tol 5.0e-03) | pass |
| A5 Bloch weights: supercell 176^2 k = 0 weight vs \|c0\|^4 | 0.820528348 vs 0.820528348: rel err 1.513e-11 (tol 1.0e-03) | pass |
| A5 Bloch weights: supercell 176^2 lattice-1 primary peak (worst of 4) vs \|c0 c1\|^2 | 0.0424933475 vs 0.0424933475: rel err 2.968e-11 (tol 1.0e-03) | pass |
| A6 Rabi J=0.1 delta=0 (code): N2/N(t) vs rabi_population | max abs err 8.327e-16 over 81 frames (tol 1e-8; max N2/N 0.999568) | pass |
| A6 Rabi J=0.1 delta=0 (code): norm | max rel drift 2.442e-15 (tol 1.0e-12, ref=1) | pass |
| A6 Rabi J=0.1 delta=0 (code): high-k weight | max/initial = 0 (tol 10); initial 0.000e+00, max 0.000e+00 | pass |
| A6 Rabi J=0.1 delta=0.15 (code): N2/N(t) vs rabi_population | max abs err 1.055e-15 over 81 frames (tol 1e-8; max N2/N 0.639956) | pass |
| A6 Rabi J=0.1 delta=0.15 (code): E/N at t = 0 vs +delta/2 (\|1> at +delta/2) | 0.0151981775 vs 0.0151981775: rel err 0.000e+00 (tol 1.0e-12) | pass |
| A6 Rabi J=0.1 delta=0.15 (code): norm | max rel drift 7.339e-14 (tol 1.0e-12, ref=1) | pass |
| A6 Rabi J=0.1 delta=0.15 (code): energy | max rel drift 7.396e-14 (tol 1.0e-10, ref=0.0151981775) | pass |
| A6 Rabi J=0.1 delta=0.15 (code): high-k weight | max/initial = 0 (tol 10); initial 0.000e+00, max 0.000e+00 | pass |

## B. Exact identities on the twisted supercell

Gauge symmetry H(-J) = sz H(J) sz, conservation of N2/N without coupling, C4 isotropy and the energy = current routes of the superfluid fraction, and time reversibility dt -> -dt.

| gate | result | |
| --- | --- | --- |
| B7 gauge copy H(-J) = sz H(J) sz: E_total(-J) vs E_total(+J) | 1403.49864 vs 1403.49864: rel err 1.620e-16 (tol 1.0e-08) | pass |
| B7 gauge copy H(-J) = sz H(J) sz: max \|n(-J) - n(+J)\| / n_max | 1.241e-15 (tol 1e-6) | pass |
| B7 gauge copy H(-J) = sz H(J) sz: E_rabi(-J) = E_rabi(+J) (sign absorbed by psi_2 -> -psi_2) | -103.993578 vs -103.993578: rel err 1.367e-16 (tol 1.0e-08) | pass |
| B7 gauge copy H(-J) = sz H(J) sz (+J): final residual | 2.673e-11 (target 2.0e-09) | pass |
| B7 gauge copy H(-J) = sz H(J) sz (-J): final residual | 2.673e-11 (target 2.0e-09) | pass |
| B8 J = 0 real time (NPSE, 88^2): norm | max rel drift 5.391e-15 (tol 1.0e-09, ref=485) | pass |
| B8 J = 0 real time (NPSE, 88^2): energy | max rel drift 3.094e-06 (tol 1.0e-05, ref=1507.62387) | pass |
| B8 J = 0 real time (NPSE, 88^2): N2_over_N (J_perp = 0) | abs drift 2.109e-15 (tol 1.000e-10 = 1.0e-10 x scale 1.000e+00) | pass |
| B8 J = 0 real time (NPSE, 88^2): high-k weight | max/initial = 1.13 (tol 10); initial 6.995e-04, max 7.874e-04 | pass |
| B8 J = 0 real time (NPSE, 88^2): kinetic phase per step | 0.315 rad (< pi) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): ground state carries no momentum | \|P\|/N = 1.323e-09 hbar/a (tol 1e-6) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): C4 isotropy f_s^xx vs f_s^yy (energy) | 0.612806943 vs 0.612806943: rel err 1.812e-16 (tol 1.0e-04) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): energy route vs current route (x) | 0.612806943 vs 0.61278171: rel err 4.118e-05 (tol 2.0e-04) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): 0 < f_s <= 1 | f_s = 0.612807 | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): f_s vs 0.6128 anchor | 0.612806943 vs 0.6128: rel err 1.133e-05 (tol 3.0e-04) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1): twisted states polished | residuals x 2.0e-10, y 2.0e-10 E_R (target 2.0e-09) | pass |
| B9 twist f_s (176^2, theta = 0.2, fixed g, nbar = 1) untwisted: final residual | 6.886e-11 (target 2.0e-09) | pass |
| B10 reversibility dt -> -dt (NPSE, 88^2): max \|psi_back - psi_0\| / max \|psi_0\| | 7.311e-14 after 400 + 400 steps (tol 1e-9; forward state moved by 1.036) | pass |
| B10 reversibility dt -> -dt (NPSE, 88^2): forward evolution non-trivial | max \|psi(T) - psi_0\| / max \|psi_0\| = 1.036 (> 1e-2) | pass |
| B10 reversibility dt -> -dt (NPSE, 88^2): norm | max rel drift 2.227e-15 (tol 1.0e-09, ref=485) | pass |
| B10 reversibility dt -> -dt (NPSE, 88^2): energy | max rel drift 4.733e-08 (tol 1.0e-05, ref=1403.45976) | pass |
| B10 reversibility dt -> -dt (NPSE, 88^2): high-k weight | max/initial = 1.01 (tol 10); initial 6.995e-04, max 7.050e-04 | pass |

## C. Numerical anchors

Dressed band bottom, chemical potential and in-situ contrast at the experiment's coupling, each reproduced by the author's earlier, separately written solvers (not distributed).

| gate | result | |
| --- | --- | --- |
| C11 dressed band bottom J=1.19802 code, g = 0, 176^2: E0 x E_R_CODE vs anchor | 14.1770557 vs 14.1770556: rel err 2.232e-10 (tol 1.0e-08) | pass |
| C11 dressed band bottom J=1.19802 code, g = 0, 176^2: E0 vs parameters.E0_BAND_BOTTOM | 14.1770557 vs 14.1770556: rel err 2.232e-10 (tol 1.0e-08) | pass |
| C11 dressed band bottom J=1.19802 code, g = 0, 176^2: E0 vs the 2.872872 E_R anchor | 2.87287212 E_R vs 2.872872 E_R: rel err 4.131e-08 (tol 3.0e-07) | pass |
| C11 dressed band bottom J=1.19802 code, g = 0, 176^2: final residual | 1.269e-12 (target 2.0e-11) | pass |
| C11 dressed band bottom J=0.599 code, g = 0, 176^2: E0 x E_R_CODE vs anchor | 14.7293973 vs 14.7293973: rel err 2.325e-10 (tol 1.0e-08) | pass |
| C11 dressed band bottom J=0.599 code, g = 0, 176^2: E0 vs parameters.E0_BAND_BOTTOM | 14.7293973 vs 14.7293973: rel err 2.325e-10 (tol 1.0e-08) | pass |
| C11 dressed band bottom J=0.599 code, g = 0, 176^2: final residual | 1.177e-12 (target 2.0e-11) | pass |
| C12 mu (0.243, nbar = 1) npse lambda = 1, g2D 5.29 nm, 176^2: mu vs anchor | 2.90843849 E_R vs 2.908439 E_R: rel err 1.761e-07 (tol 5.0e-06) | pass |
| C12 mu (0.243, nbar = 1) npse lambda = 1, g2D 5.29 nm, 176^2: ladder converged and polished | converged=True, residual 5.3e-11 E_R (target 2.0e-09) | pass |
| C13 C_psf (0.243, nbar = 1) npse 176^2: C_psf vs anchor | 0.745614832 vs 0.7456: abs err 1.483e-05 (tol 4.0e-03) | pass |
| C12 mu (0.243, nbar = 1) fixed g, g2D 5.29 nm, 176^2: mu vs anchor | 2.90910482 E_R vs 2.909104 E_R: rel err 2.831e-07 (tol 1.0e-05) | pass |
| C12 mu (0.243, nbar = 1) fixed g, g2D 5.29 nm, 176^2: ladder converged and polished | converged=True, residual 6.9e-11 E_R (target 2.0e-09) | pass |
| C13 C_psf (0.243, nbar = 1) fixed 176^2: C_psf vs anchor | 0.730250532 vs 0.731: abs err 7.495e-04 (tol 4.0e-03) | pass |
| C12 mu (0.243, nbar = 1) npse lambda = 1, engine g2D (a_s = 100 a0), 176^2: mu vs anchor | 2.90844652 E_R vs 2.908446 E_R: rel err 1.785e-07 (tol 5.0e-06) | pass |
| C12 mu (0.243, nbar = 1) npse lambda = 1, engine g2D (a_s = 100 a0), 176^2: ladder converged and polished | converged=True, residual 5.3e-11 E_R (target 2.0e-09) | pass |

## D. Pre-flight checks

Kinetic phase per time step of the automatic time step (orders 2 and 4), grid resolution, unit identities, and the validity range of the quasi-2D (NPSE) reduction.

| gate | result | |
| --- | --- | --- |
| D engine choose_dt order 2: kinetic phase per step | k2max dt/2 = 0.200 rad (limit 2.6; instability above pi) | pass |
| D engine choose_dt order 2: resolution | k_nyquist 25.1 vs 2 x k_char 12.6 | pass |
| D engine choose_dt order 2: kappa k2max dt (E_R) vs k2max dt/2 (code) | 0.2 vs 0.2: rel err 4.163e-16 (tol 1.0e-12) | pass |
| D engine choose_dt order 4 (multiplier 20): kinetic phase per step | k2max dt/2 = 1.500 rad (limit 2.6; instability above pi) | pass |
| D engine choose_dt order 4 (multiplier 20): resolution | k_nyquist 25.1 vs 2 x k_char 12.6 | pass |
| D engine choose_dt order 4 (multiplier 20): kappa k2max dt (E_R) vs k2max dt/2 (code) | 1.5 vs 1.5: rel err 4.441e-16 (tol 1.0e-12) | pass |
| D engine choose_dt order 4 (multiplier 20): largest Yoshida sub-step theta_Y dt | 2.027 rad (outer step 1.500 rad; the cap 1.5 rad applies to the outer step; the split step is unstable near pi) | pass |
| D NPSE validity nbar = 1, J = 1.19802 code: quasi-2D parameter | kappa = g2D n_peak / hbar omega_z = 0.518 (NPSE validated to 0.8); n_peak = 8.941 atoms/a^2 (88^2, lambda = 1) | pass |

## E. Trapped cloud

A cloud of N = 2000 atoms in a 20 Hz trap: contrasts and size against the author's earlier quasi-2D solver (not distributed), convergence, conservation, and the cloud staying inside the box.

| gate | result | |
| --- | --- | --- |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: C_raw vs anchor | 0.585539034 vs 0.585539: abs err 3.421e-08 (tol 2.0e-03) | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: C_coh vs anchor | 0.574831816 vs 0.5748301: abs err 1.716e-06 (tol 2.0e-03) | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: R_rms_a vs anchor | 15.3656197 vs 15.36562: rel err 2.268e-08 (tol 2.0e-03) | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: polish converged | residual 1.8e-10 E_R (target 2.0e-10), 1850 steps | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: N | 2000 vs 2000: rel err 2.274e-16 (tol 1.0e-10) | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: N2/N = 1/2 (SU(2), delta = 0) | 0.5 vs 0.5: abs err 0.000e+00 (tol 1.0e-08) | pass |
| E trapped N = 2000, J = 0.243, NPSE lambda = 1, 20 Hz, 528^2: cloud inside the box | edge density / peak = 2.1e-04 (< 1e-3) | pass |
