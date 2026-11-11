"""Regression references of the check panels (`twisted_bilayer.panel`): numbers the panels must reproduce,
each with its tolerance and source. Lattice units (E_R, a); mu and E/N of the 3D cloud without hbar omega_z/2.

QUICK_3D: the small trapped cloud of `trapped_3d_quick` (m = 4, 3 x 3 cells of 32 points, N = 60, 200 Hz,
V0 = 4 E_R, Omega_R = 0.24277 E_R, N_z = 16 over +-5 l_z), complex128, residual < 1e-9 E_R. Source: a CPU run
of `trapped3d.solve_models` with these settings. Tolerances: 1e-8 E_R on energies (the
solve converges the residual to ~5e-10), 1e-6 on densities and contrasts (device and FFT library).

EXPERIMENT_3D: the experiment's cloud of `gpu_precision` (m = 22, 3 x 3 supercells at 176 points, N = 2000,
20 Hz), complex128. Source: `gpu_precision` on two different GPUs (N_z = 48 and N_z = 64),
identical in every printed digit. Tolerances cover the printed rounding (8 decimals on energies,
5 on contrasts) and, for the 3D cloud, a coarser N_z (the panel runs N_z = 32; `gpu_3d_nz` puts N_z >= 16
within the complex64 floor).
"""

QUICK_3D = {
    "npse": dict(mu=2.977094990189019, E_per_N=2.956943328093223, C_raw=0.21231683371143956, R_rms=2.1934433984039545),
    "fixed": dict(
        mu=2.9784741528497305, E_per_N=2.957559113725757, C_raw=0.19779886774293215, R_rms=2.2115201542025713
    ),
    "3d": dict(mu=2.9772194104915553, E_per_N=2.9570065667896, C_raw=0.20912314320979955, R_rms=2.195225245897904),
}
QUICK_3D_TOL = dict(mu=1e-8, E_per_N=1e-8, C_raw=1e-6, R_rms=1e-6)

EXPERIMENT_3D = {
    "npse": dict(mu=2.93012937, E_per_N=2.91395617, C_raw=0.58545, C_coh=0.57474),
    "fixed": dict(mu=2.93084770, E_per_N=2.91435809, C_raw=0.56112, C_coh=0.55056),
    "3d": dict(mu=2.93019829, E_per_N=2.91399919, C_raw=0.58194, C_coh=0.57126),
}
EXPERIMENT_3D_TOL = {
    "npse": dict(mu=2e-8, E_per_N=2e-8, C_raw=1e-5, C_coh=1e-5),
    "fixed": dict(mu=2e-8, E_per_N=2e-8, C_raw=1e-5, C_coh=1e-5),
    "3d": dict(mu=1e-6, E_per_N=1e-6, C_raw=3e-5, C_coh=3e-5),
}
