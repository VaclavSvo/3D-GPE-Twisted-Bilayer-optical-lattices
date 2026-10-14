"""The two-component Gross-Pitaevskii engine (PyTorch, CPU or CUDA).

Lattice units throughout: a = 1 (lattice constant), E_R = 1 (recoil energy), hbar = 1, so m = pi^2/2 and
the kinetic operator is -(1/pi^2) nabla^2. The device is chosen when this package is first imported:
CUDA when available, unless TWISTED_BILAYER_FORCE_CPU=1.

The names most scripts need are importable from here:

    from twisted_bilayer.engine import BilayerGPE, Grid, NPSE2D, bilayer_potential, moire_geometry, relax
"""

from .grid import Grid
from .ground_state import initial_state, relax
from .lattice import bilayer_potential, harmonic_trap, moire_geometry, square_lattice
from .npse import NPSE2D
from .physics import BilayerGPE, choose_dt
from .units import KAPPA, lattice_units

__all__ = [
    "KAPPA",
    "NPSE2D",
    "BilayerGPE",
    "Grid",
    "bilayer_potential",
    "choose_dt",
    "harmonic_trap",
    "initial_state",
    "lattice_units",
    "moire_geometry",
    "relax",
    "square_lattice",
]
