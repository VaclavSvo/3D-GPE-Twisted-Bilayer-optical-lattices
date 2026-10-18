"""Quasi-2D interaction with a density-dependent vertical width (the 2D
reduction of Salasnich's non-polynomial Schrodinger equation).

The fixed-width reduction psi(r, z) = psi(r) phi_0(z) gives g_2D = g_3D /
(sqrt(2 pi) l_z) and is valid while g_2D n << hbar omega_z. In the AA pockets
n_2D reaches tens of atoms per a^2 and g_2D n / hbar omega_z is O(1): the
pancake broadens locally and the fixed g_2D overestimates the interaction
energy there. The variational fix: a Gaussian phi(z) of local width
sigma(n) chosen to minimise the energy per unit area

    e(n, s) = n (omega_z/4)(1/s^2 + s^2) + g_2D n^2 / (2 s),   s = sigma / l_z,

which gives the quartic  s^4 - x s - 1 = 0,  x = g_2D n / omega_z  (hbar = 1),
and (envelope theorem) the local chemical potential

    mu_int(n) = (omega_z/4)(1/s^2 + s^2) + g_2D n / s - omega_z/2,

with the zero-point omega_z/2 subtracted so that mu_int -> g_2D n as n -> 0.
e_int(n) = e(n, s(n)) - n omega_z/2 replaces g n^2/2 in the energy functional
and mu_int(n) replaces g n in the Hamiltonian. The quartic is solved
pointwise by Newton's method (6 iterations from s_0 = max(1, x^(1/3))).

Use: BilayerGPE(..., interaction=NPSE2D(g2d, omega_z)) instead of g=g2d.
Validated against full 3D (thin_z.thin_z_check(..., npse=True); trapped3d.solve_models for the trapped
cloud): the fixed-g_2D energy lies above the 3D one (psi(x, y) phi_0(z) is a 3D trial state), while the
NPSE lies slightly below it, because the local width ignores the kinetic energy of its own gradient; on
the experiment's trapped cloud the NPSE is within 7e-5 E_R of 3D in mu and 0.6 % in the contrast.

Reference: Salasnich, Parola & Reatto, Phys. Rev. A 65, 043614 (2002);
Salasnich & Malomed, Phys. Rev. A 79, 053620 (2009) (quasi-2D form).
"""

from __future__ import annotations

import torch


class NPSE2D:
    def __init__(self, g2d: float, omega_z: float, newton_iters: int = 6):
        self.g2d, self.omega_z, self.iters = float(g2d), float(omega_z), int(newton_iters)

    def width(self, n: torch.Tensor) -> torch.Tensor:
        """s(n) = sigma / l_z, root of s^4 - x s - 1 = 0, x = g_2D n / omega_z."""
        x = (self.g2d / self.omega_z) * n
        s = torch.clamp(x.abs() ** (1.0 / 3.0), min=1.0)
        for _ in range(self.iters):
            f = s**4 - x * s - 1.0
            fp = 4.0 * s**3 - x
            s = s - f / fp
        return s

    def mu_int(self, n: torch.Tensor) -> torch.Tensor:
        s = self.width(n)
        s2 = s * s
        return 0.25 * self.omega_z * (1.0 / s2 + s2) + self.g2d * n / s - 0.5 * self.omega_z

    def e_int(self, n: torch.Tensor) -> torch.Tensor:
        s = self.width(n)
        s2 = s * s
        return n * 0.25 * self.omega_z * (1.0 / s2 + s2) + 0.5 * self.g2d * n * n / s - 0.5 * self.omega_z * n

    def dmu_int(self, n: torch.Tensor) -> torch.Tensor:
        """d mu_int / d n = (g_2D / s) [1 - x / (2 (4 s^3 - x))], the pointwise
        coupling of the BdG exchange term (engine.bdg). From mu_int =
        d e/d n at fixed s (envelope theorem), d mu/d s = -g_2D n / (2 s^2)
        on the quartic, and d s/d n = s (g_2D/omega_z) / (4 s^3 - x)."""
        s = self.width(n)
        x = (self.g2d / self.omega_z) * n
        return (self.g2d / s) * (1.0 - x / (2.0 * (4.0 * s**3 - x)))

    def describe(self) -> str:
        return f"NPSE quasi-2D: g2d = {self.g2d:.5g}, omega_z = {self.omega_z:.4g}"
