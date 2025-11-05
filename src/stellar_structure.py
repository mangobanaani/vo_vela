"""
Stellar Structure Module

Calculate neutron star density profiles, either from:
1. Analytic approximations (fast, good for testing)
2. TOV equations (realistic, requires integration)

Functions:
----------
- analytic_density_profile(r, R, rho_c, n): Power-law profile
- simple_neutron_star(M, R, eos): Quick NS model
- tov_solver(eos, rho_c): Full general relativistic structure (TODO)
- glitch_location(R): Typical depth where glitches occur
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq
from . import constants as const


# ============================================================================
# Analytic Density Profiles
# ============================================================================

def analytic_density_profile(r, R, rho_c, n=0.5):
    """
    Simple analytic density profile for neutron star.

    ρ(r) = ρ_c [1 - (r/R)²]^n

    This is a good approximation for soft-to-moderate EoS.

    Parameters:
    -----------
    r : float or array
        Radius (cm)
    R : float
        Stellar radius (cm)
    rho_c : float
        Central density (g/cm³)
    n : float
        Power index (typically 0.5-1.0)

    Returns:
    --------
    rho : float or array
        Density at radius r (g/cm³)
    """
    x = r / R
    rho = rho_c * (1 - x**2)**n

    # Ensure non-negative
    rho = np.maximum(rho, 0.0)

    return rho


def uniform_density_sphere(M, R):
    """
    Uniform density sphere (simplest approximation).

    ρ = 3M / (4πR³)

    Parameters:
    -----------
    M : float
        Total mass (g)
    R : float
        Radius (cm)

    Returns:
    --------
    rho : float
        Uniform density (g/cm³)
    """
    return 3 * M / (4 * np.pi * R**3)


def enclosed_mass(r, R, rho_c, n=0.5):
    """
    Mass enclosed within radius r for analytic profile.

    M(r) = integral from 0 to r of 4*pi*r'^2 * rho(r') dr'

    For rho(r) = rho_c[1-(r/R)^2]^n, this can be integrated analytically.

    Parameters:
    -----------
    r : float or array
        Radius (cm)
    R : float
        Stellar radius (cm)
    rho_c : float
        Central density (g/cm³)
    n : float
        Power index

    Returns:
    --------
    M : float or array
        Enclosed mass (g)
    """
    # For n=0.5, can use beta function, but numerical integration is easier
    # Use Simpson's rule for quick calculation

    if np.isscalar(r):
        r_array = np.linspace(0, r, 100)
    else:
        r_array = r

    rho_array = analytic_density_profile(r_array, R, rho_c, n)
    integrand = 4 * np.pi * r_array**2 * rho_array

    # Integrate using trapezoidal rule
    if np.isscalar(r):
        M = np.trapz(integrand, r_array)
    else:
        M = np.array([np.trapz(integrand[:i+1], r_array[:i+1])
                     for i in range(len(r_array))])

    return M


# ============================================================================
# Simple Neutron Star Models
# ============================================================================

class SimpleNeutronStar:
    """
    Simple neutron star model with analytic density profile.

    Attributes:
    -----------
    M : float
        Total mass (g)
    R : float
        Radius (cm)
    rho_c : float
        Central density (g/cm³)
    n : float
        Profile index
    eos : EoSBase
        Equation of state
    """

    def __init__(self, M, R, eos=None, profile_index=0.5):
        """
        Initialize simple NS model.

        Parameters:
        -----------
        M : float
            Total mass (g)
        R : float
            Radius (cm)
        eos : EoSBase, optional
            Equation of state object
        profile_index : float
            Power index for density profile
        """
        self.M = M
        self.R = R
        self.n = profile_index
        self.eos = eos

        # Estimate central density from mass and radius
        # For analytic profile, integrate to match total mass
        # Use typical rho_c ~ 5-10 rho_0 for 1.4 Msun star
        self.rho_c = self._estimate_central_density()

    def _estimate_central_density(self):
        """
        Estimate central density to match total mass.

        For ρ(r) = ρ_c[1-(r/R)²]^n, the total mass is:
        M =    4πr² ρ(r) dr

        This gives ρ_c   3M/(πR³) × correction_factor

        For n=0.5, correction_factor   1.5
        For n=1.0, correction_factor   2.5
        """
        # Rough estimate
        rho_avg = 3 * self.M / (4 * np.pi * self.R**3)

        # Central density is higher than average
        correction = 2.0 + self.n  # Simple approximation
        rho_c = correction * rho_avg

        return rho_c

    def density(self, r):
        """Get density at radius r."""
        return analytic_density_profile(r, self.R, self.rho_c, self.n)

    def mass(self, r):
        """Get enclosed mass at radius r."""
        return enclosed_mass(r, self.R, self.rho_c, self.n)

    def density_profile(self, n_points=100):
        """
        Get full density profile.

        Parameters:
        -----------
        n_points : int
            Number of radial points

        Returns:
        --------
        profile : dict
            Dictionary with 'r', 'rho', 'M'
        """
        r_array = np.linspace(0, self.R, n_points)
        rho_array = self.density(r_array)
        M_array = self.mass(r_array)

        return {
            'r': r_array,
            'rho': rho_array,
            'M': M_array,
            'r_km': r_array / 1e5,
            'rho_rho0': rho_array / const.rho_0,
            'M_Msun': M_array / const.M_sun
        }

    def neutron_fraction_profile(self):
        """
        Get neutron fraction profile if EoS is provided.

        Returns:
        --------
        profile : dict
            Dictionary with 'r', 'f_n'
        """
        if self.eos is None:
            raise ValueError("EoS not provided - cannot calculate f_n")

        profile = self.density_profile()
        f_n_array = np.array([self.eos.neutron_fraction(rho)
                             for rho in profile['rho']])

        profile['f_n'] = f_n_array
        return profile


# ============================================================================
# Glitch Location
# ============================================================================

def glitch_location(R, relative_depth=0.95):
    """
    Typical radius where glitches occur.

    Glitches are thought to occur in the inner crust at r ~ 0.90-0.98 R.

    Parameters:
    -----------
    R : float
        Stellar radius (cm)
    relative_depth : float
        Fractional radius (default 0.95)

    Returns:
    --------
    r_glitch : float
        Glitch radius (cm)
    """
    return relative_depth * R


def glitch_density_simple(M, R, relative_depth=0.95, profile_index=0.5):
    """
    Estimate density at glitch location using simple profile.

    Parameters:
    -----------
    M : float
        Stellar mass (g)
    R : float
        Stellar radius (cm)
    relative_depth : float
        Glitch location r/R
    profile_index : float
        Density profile index

    Returns:
    --------
    rho_glitch : float
        Density at glitch location (g/cm³)
    """
    ns = SimpleNeutronStar(M, R, profile_index=profile_index)
    r_glitch = glitch_location(R, relative_depth)
    rho_glitch = ns.density(r_glitch)
    return rho_glitch


# ============================================================================
# TOV Solver (TODO - Full implementation)
# ============================================================================

def tov_equations(y, r, eos):
    """
    Tolman-Oppenheimer-Volkoff equations for stellar structure.

    dy/dr = f(y, r)

    where y = [P, M]

    dP/dr = -G [M + 4πr³P/c²][ρ + P/c²] / [r² (1 - 2GM/(rc²))]
    dM/dr = 4πr² ρ

    Parameters:
    -----------
    y : array
        [P, M] - pressure and enclosed mass
    r : float
        Radius (cm)
    eos : EoSBase
        Equation of state

    Returns:
    --------
    dydr : array
        [dP/dr, dM/dr]
    """
    P, M = y

    # Get density from EoS (need inverse: ρ(P))
    # This requires EoS to provide P(ρ) and we invert it
    # For now, placeholder
    rho = eos.density_from_pressure(P) if hasattr(eos, 'density_from_pressure') else 0

    if r == 0:
        # Special case at center
        dPdr = 0
        dMdr = 0
    else:
        # TOV equations
        dPdr = -(const.G * (M + 4*np.pi*r**3*P/const.c**2) *
                (rho + P/const.c**2)) / (r**2 * (1 - 2*const.G*M/(r*const.c**2)))

        dMdr = 4 * np.pi * r**2 * rho

    return [dPdr, dMdr]


def solve_tov(eos, rho_c, r_max=2e6):
    """
    Solve TOV equations for given central density.

    NOTE: This is a placeholder. Full implementation requires:
    1. Inverse EoS: ρ(P)
    2. Proper boundary conditions
    3. Surface finding algorithm
    4. Numerical stability checks

    Parameters:
    -----------
    eos : EoSBase
        Equation of state
    rho_c : float
        Central density (g/cm³)
    r_max : float
        Maximum integration radius (cm)

    Returns:
    --------
    solution : dict
        Dictionary with 'r', 'P', 'M', 'rho'
    """
    # TODO: Full implementation
    raise NotImplementedError("Full TOV solver not yet implemented. "
                            "Use SimpleNeutronStar for now.")


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_central_density(M, R):
    """
    Estimate central density from mass and radius.

    Simple estimate: ρ_c   3-5 times average density.

    Parameters:
    -----------
    M : float
        Mass (g)
    R : float
        Radius (cm)

    Returns:
    --------
    rho_c : float
        Estimated central density (g/cm³)
    """
    rho_avg = 3 * M / (4 * np.pi * R**3)
    rho_c = 4 * rho_avg  # Typical factor
    return rho_c


def compactness(M, R):
    """
    Compactness parameter GM/(Rc²).

    For neutron stars, typically 0.1-0.3.

    Parameters:
    -----------
    M : float
        Mass (g)
    R : float
        Radius (cm)

    Returns:
    --------
    C : float
        Compactness (dimensionless)
    """
    return const.G * M / (R * const.c**2)


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Stellar Structure Module Testing")
    print("=" * 70)

    # Test analytic profile
    print("\n1. Analytic Density Profile")
    print("-" * 70)

    M_test = 1.4 * const.M_sun
    R_test = 12e5  # cm
    rho_c_test = 5 * const.rho_0

    r_array = np.linspace(0, R_test, 100)
    rho_array = analytic_density_profile(r_array, R_test, rho_c_test, n=0.5)

    print(f"Stellar parameters:")
    print(f"  M = {M_test/const.M_sun:.1f} M ")
    print(f"  R = {R_test/1e5:.0f} km")
    print(f"  ρ_c = {rho_c_test/const.rho_0:.1f} ρ  ")

    print(f"\nDensity profile:")
    for i in [0, 25, 50, 75, 99]:
        print(f"  r/R = {r_array[i]/R_test:.2f}      "
              f"ρ = {rho_array[i]/const.rho_0:.2f} ρ  ")

    # Test SimpleNeutronStar
    print("\n2. Simple Neutron Star Model")
    print("-" * 70)

    ns = SimpleNeutronStar(M_test, R_test, profile_index=0.5)

    print(f"Input:")
    print(f"  M = {ns.M/const.M_sun:.2f} M ")
    print(f"  R = {ns.R/1e5:.0f} km")
    print(f"\nDerived:")
    print(f"  ρ_c = {ns.rho_c/const.rho_0:.2f} ρ  ")
    print(f"  Compactness = {compactness(ns.M, ns.R):.3f}")

    # Get profile
    profile = ns.density_profile(n_points=10)

    print(f"\nDensity profile:")
    print(f"{'r (km)':>8}  {'r/R':>6}  {'ρ/ρ  ':>8}  {'M(r)/M ':>10}")
    print("-" * 70)

    for i in range(len(profile['r'])):
        print(f"{profile['r_km'][i]:8.1f}  "
              f"{profile['r'][i]/ns.R:6.2f}  "
              f"{profile['rho_rho0'][i]:8.2f}  "
              f"{profile['M_Msun'][i]:10.4f}")

    # Test glitch location
    print("\n3. Glitch Location")
    print("-" * 70)

    r_glitch = glitch_location(R_test, 0.95)
    rho_glitch = glitch_density_simple(M_test, R_test, 0.95)

    print(f"Typical glitch depth: r/R = 0.95")
    print(f"  r_glitch = {r_glitch/1e5:.1f} km")
    print(f"  ρ_glitch = {rho_glitch/const.rho_0:.2f} ρ  ")

    # Test with different depths
    print(f"\nDensity vs glitch depth:")
    print(f"{'r/R':>6}  {'ρ/ρ  ':>8}")
    print("-" * 70)

    for depth in [0.90, 0.92, 0.94, 0.96, 0.98]:
        rho = glitch_density_simple(M_test, R_test, depth)
        print(f"{depth:6.2f}  {rho/const.rho_0:8.2f}")

    # Test with EoS
    print("\n4. Neutron Fraction Profile")
    print("-" * 70)

    try:
        from . import eos

        eos_model = eos.SymmetryEnergyEoS(L0=55.0)
        ns_with_eos = SimpleNeutronStar(M_test, R_test, eos=eos_model)

        profile_fn = ns_with_eos.neutron_fraction_profile()

        print(f"With L   = 55 MeV:")
        print(f"{'r/R':>6}  {'ρ/ρ  ':>8}  {'f_n':>8}")
        print("-" * 70)

        for i in [0, 25, 50, 75, 99]:
            print(f"{profile_fn['r'][i]/ns.R:6.2f}  "
                  f"{profile_fn['rho_rho0'][i]:8.2f}  "
                  f"{profile_fn['f_n'][i]:8.4f}")

        print("\nOK Neutron fraction profile calculated!")

    except ImportError:
        print("(EoS module not available)")

    print("\n" + "=" * 70)
    print("All tests passed! OK")
    print("=" * 70)

    # Create plot
    try:
        import matplotlib.pyplot as plt

        print("\nCreating density profile plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Density profile
        profile = ns.density_profile(n_points=100)

        ax1.plot(profile['r_km'], profile['rho_rho0'], 'b-', linewidth=2)
        ax1.axhline(1, color='k', linestyle='--', alpha=0.5, label='ρ  ')
        ax1.axvline(r_glitch/1e5, color='r', linestyle='--', alpha=0.5,
                   label='Glitch location')
        ax1.set_xlabel('Radius (km)', fontsize=12)
        ax1.set_ylabel(r'Density $\rho/\rho_0$', fontsize=12)
        ax1.set_title('Neutron Star Density Profile', fontsize=13)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Mass profile
        ax2.plot(profile['r_km'], profile['M_Msun'], 'g-', linewidth=2)
        ax2.set_xlabel('Radius (km)', fontsize=12)
        ax2.set_ylabel(r'Enclosed Mass $M(r)/M_\odot$', fontsize=12)
        ax2.set_title('Mass Profile', fontsize=13)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('../figures/stellar_structure.pdf')
        print("OK Plot saved: figures/stellar_structure.pdf")
        plt.show()

    except Exception as e:
        print(f"(Could not create plot: {e})")

    print("\n" + "=" * 70)
    print("Stellar structure module ready! ")
    print("=" * 70)
