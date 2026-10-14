"""The engine's units module takes its SI constants from the parameters module.

(i)   every constant it re-exports is the parameters module's object itself;
(ii)  the derived unit system is bit-identical to the pinned one
      (the pinned literals below);
(iii) the two named exceptions (a_s = 100 a0, the 86.909180 u mass) sit at their recorded offsets;
(iv)  the E_R <-> code-unit conversions are the parameters module's and invert each other.
"""

import math

import pytest

from twisted_bilayer import parameters as P
from twisted_bilayer.engine import units


def test_constants_come_from_parameters():
    assert units.HBAR is P.HBAR_SI or units.HBAR == P.HBAR_SI
    assert units.U_ATOMIC_MASS == P.AMU_SI
    assert units.BOHR_RADIUS == P.A0_SI and units.A_LATTICE_M == P.A_LATTICE_SI
    assert units.NU_Z_HZ == P.NU_Z_HZ and units.E_R_CODE == P.E_R_CODE == math.pi**2 / 2
    assert units.er_to_code is P.er_to_code and units.t_er_to_code is P.t_er_to_code


@pytest.mark.parametrize(
    "key,value",
    [
        ("E_R_Hz", 3678.38430727536),
        ("t0_s", 4.326762235721477e-05),
        ("g2d", 0.015763786407976224),
        ("g3d", 0.034114903852878936),
        ("omega_z", 0.27185848907144683),
        ("l_z", 0.8633634830180625),
    ],
)
def test_derived_units_unchanged(key, value):
    """The pinned values of lattice_units(), to the last bit."""
    assert getattr(units.lattice_units(), key) == value


def test_named_exceptions_and_their_effect():
    a_s, param_a_s, off, _ = units.EXCEPTIONS["RB87_SCATTERING_LENGTH_M"]
    assert a_s == 100.0 * P.A0_SI and param_a_s == P.A_S_SI and abs(off - 3.3499e-4) < 1e-8
    m, param_m, off_m, _ = units.EXCEPTIONS["RB87_MASS_U"]
    assert m == 86.909180 and abs(off_m + 6.0638e-9) < 1e-12
    c = units.compare_with_parameters()
    assert abs(c["g2d"]["rel"] - 3.350e-4) < 2e-7  # a_s (3.3499e-4) and mass (+3.0e-9)
    assert abs(c["E_R_Hz"]["rel"] - 6.064e-9) < 1e-11
    assert abs(c["g2d"]["parameters"] - P.G2D_ER) < 1e-18


def test_conversions_round_trip():
    for E in (0.24277, 4.0, 0.0157585):
        assert abs(units.code_to_er(units.er_to_code(E)) - E) < 1e-15 * max(1.0, E)
    assert abs(units.er_to_code(0.24277) - 1.198020) < 2e-6  # Omega_R = 2 pi x 893 Hz
    assert abs(units.t_er_to_code(units.t_code_to_er(84.302)) - 84.302) < 1e-12  # 18 ms
    # E * t is invariant: a phase is the same number in both unit systems
    assert abs(units.er_to_code(3.0) * units.t_er_to_code(2.0) - 6.0) < 1e-14
