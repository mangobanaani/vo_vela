"""
Physical constants in CGS units for neutron star calculations.
"""

import numpy as np

# ============================================================================
# Fundamental Constants (CGS)
# ============================================================================

# Speed of light (cm/s)
c = 2.99792458e10

# Gravitational constant (cm^3/g/s^2)
G = 6.67430e-8

# Planck constant (erg s)
h = 6.62607015e-27
hbar = h / (2 * np.pi)

# Boltzmann constant (erg/K)
k_B = 1.380649e-16

# Electron volt (erg)
eV = 1.602176634e-12
MeV = 1e6 * eV

# ============================================================================
# Particle Masses (g)
# ============================================================================

# Neutron mass (g)
m_n = 1.67492749804e-24

# Proton mass (g)
m_p = 1.67262192369e-24

# Electron mass (g)
m_e = 9.1093837015e-28

# Atomic mass unit (g)
m_u = 1.66053906660e-24

# ============================================================================
# Astrophysical Constants
# ============================================================================

# Solar mass (g)
M_sun = 1.98841e33

# Solar radius (cm)
R_sun = 6.957e10

# Parsec (cm)
pc = 3.0857e18

# Year (seconds)
year = 3.15576e7

# Day (seconds)
day = 86400.0

# ============================================================================
# Nuclear Physics Constants
# ============================================================================

# Nuclear saturation density (g/cm^3)
rho_0 = 2.8e14

# Nuclear saturation density (fm^-3)
n_0 = 0.16

# Quantum of circulation (cm^2/s)
kappa = hbar / (2 * m_n)

# Symmetry energy at saturation (MeV)
S_0 = 32.0  # Typical value, ~30-34 MeV

# Nuclear incompressibility (MeV)
K_0 = 240.0  # Typical value, ~220-260 MeV

# ============================================================================
# Typical Neutron Star Parameters
# ============================================================================

# Canonical neutron star mass (g)
M_NS_canonical = 1.4 * M_sun

# Typical neutron star radius (cm)
R_NS_typical = 12e5  # 12 km

# Vela pulsar rotation frequency (Hz)
nu_Vela = 11.2  # Hz

# Vela angular velocity (rad/s)
Omega_Vela = 2 * np.pi * nu_Vela

# Typical pulsar magnetic field (G)
B_pulsar_typical = 1e12

# ============================================================================
# Vortex Physics Parameters
# ============================================================================

def vortex_spacing(Omega):
    """
    Vortex spacing for given rotation rate.

    Parameters:
    -----------
    Omega : float
        Angular velocity (rad/s)

    Returns:
    --------
    b : float
        Inter-vortex spacing (cm)
    """
    return np.sqrt(kappa / (2 * Omega))

def coherence_length(Delta, v_F=None):
    """
    Coherence length from pairing gap.

    Parameters:
    -----------
    Delta : float
        Pairing gap (erg)
    v_F : float, optional
        Fermi velocity (cm/s). If None, uses typical value.

    Returns:
    --------
    xi : float
        Coherence length (cm)
    """
    if v_F is None:
        # Typical Fermi velocity for neutrons at 0.5 rho_0
        v_F = 0.3 * c  # ~ 10^10 cm/s
    return hbar * v_F / Delta

# Typical values at 0.5 rho_0
b_typical = vortex_spacing(Omega_Vela)  # ~ 10^-4 cm
Delta_typical = 0.1 * MeV  # ~ 10^-6 erg
xi_typical = coherence_length(Delta_typical)  # ~ 10^-12 cm
log_b_over_xi = np.log(b_typical / xi_typical)  # ~ 20

# ============================================================================
# Unit Conversions
# ============================================================================

def MeV_to_erg(E_MeV):
    """Convert MeV to erg"""
    return E_MeV * MeV

def erg_to_MeV(E_erg):
    """Convert erg to MeV"""
    return E_erg / MeV

def MeV_to_K(E_MeV):
    """Convert MeV to Kelvin"""
    return E_MeV * MeV / k_B

def K_to_MeV(T_K):
    """Convert Kelvin to MeV"""
    return T_K * k_B / MeV

def density_g_cm3_to_fm3(rho):
    """Convert density from g/cm^3 to fm^-3"""
    return rho / (m_u * 1e39)

def density_fm3_to_g_cm3(n):
    """Convert density from fm^-3 to g/cm^3"""
    return n * m_u * 1e39

# ============================================================================
# Derived Quantities
# ============================================================================

# Schwarzschild radius for canonical NS (cm)
R_Schwarzschild_canonical = 2 * G * M_NS_canonical / c**2  # ~ 4.1 km

# Compactness parameter
compactness_canonical = R_Schwarzschild_canonical / R_NS_typical  # ~ 0.34

# Surface gravity (cm/s^2)
g_surface_canonical = G * M_NS_canonical / R_NS_typical**2

# Free-fall time (s)
t_freefall_canonical = np.sqrt(R_NS_typical**3 / (G * M_NS_canonical))

# Keplerian frequency at surface (Hz)
nu_Keplerian = np.sqrt(G * M_NS_canonical / R_NS_typical**3) / (2 * np.pi)

if __name__ == "__main__":
    # Print some useful values for reference
    print("=" * 60)
    print("Physical Constants for Vortex Oscillation Spectroscopy")
    print("=" * 60)
    print(f"\nFundamental Constants:")
    print(f"  c = {c:.3e} cm/s")
    print(f"  G = {G:.3e} cm^3/g/s^2")
    print(f"  ℏ = {hbar:.3e} erg·s")
    print(f"  k_B = {k_B:.3e} erg/K")

    print(f"\nNuclear Constants:")
    print(f"  ρ₀ = {rho_0:.3e} g/cm³")
    print(f"  κ = ℏ/(2m_n) = {kappa:.3e} cm²/s")
    print(f"  S₀ = {S_0:.1f} MeV")

    print(f"\nVela Pulsar:")
    print(f"  ν = {nu_Vela:.2f} Hz")
    print(f"  Ω = {Omega_Vela:.2f} rad/s")
    print(f"  b = {b_typical:.3e} cm")

    print(f"\nVortex Parameters:")
    print(f"  Typical Δ = {Delta_typical/MeV:.2f} MeV")
    print(f"  Typical ξ = {xi_typical:.3e} cm")
    print(f"  ln(b/ξ) ≈ {log_b_over_xi:.1f}")

    print(f"\nCanonical NS:")
    print(f"  M = {M_NS_canonical/M_sun:.1f} M_")
    print(f"  R = {R_NS_typical/1e5:.1f} km")
    print(f"  R_S = {R_Schwarzschild_canonical/1e5:.2f} km")
    print(f"  Compactness = {compactness_canonical:.3f}")

    print(f"\nUnit Conversions:")
    print(f"  1 MeV = {MeV:.3e} erg")
    print(f"  1 MeV/k_B = {MeV_to_K(1):.3e} K")
    print(f"  1 day = {day:.0f} s")
    print("=" * 60)
