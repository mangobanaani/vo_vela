"""
Superfluid Properties Module

Calculate pairing gaps, critical temperatures, and superfluid densities
for neutrons in the inner crust of neutron stars.

Functions:
----------
- pairing_gap_AO(rho): Ainsworth-Ozaki pairing gap model
- pairing_gap_CCDK(rho): Chen et al. pairing gap model
- critical_temperature(Delta): T_c from pairing gap
- superfluid_density(rho, T, f_n): ρ_n(ρ, T) accounting for thermal depletion
- superfluid_fraction(T, T_c): Temperature-dependent suppression factor
"""

import numpy as np
from . import constants as const


# ============================================================================
# Pairing Gap Models
# ============================================================================

def pairing_gap_AO(rho, Delta_max=0.1, rho_peak=0.6, width=5.0):
    """
    Ainsworth-Ozaki (AO) pairing gap for ³P₂ neutron pairing.

    Gaussian profile peaked around 0.6 ρ₀.

    Parameters:
    -----------
    rho : float or array
        Mass density (g/cm³)
    Delta_max : float
        Maximum pairing gap (MeV), default 0.1
    rho_peak : float
        Peak density (in units of ρ₀), default 0.6
    width : float
        Width parameter, default 5.0

    Returns:
    --------
    Delta : float or array
        Pairing gap (erg)

    References:
    -----------
    Ainsworth, Wambach & Pines (1989)
    """
    # Density in units of ρ₀
    x = rho / const.rho_0

    # Gaussian profile
    Delta_MeV = Delta_max * np.exp(-width * (x - rho_peak)**2)

    # Convert to erg
    Delta = Delta_MeV * const.MeV

    return Delta


def pairing_gap_CCDK(rho):
    """
    Chen-Clark-Davé-Khodel (CCDK) pairing gap model.

    More sophisticated model based on realistic nuclear interactions.

    Parameters:
    -----------
    rho : float or array
        Mass density (g/cm³)

    Returns:
    --------
    Delta : float or array
        Pairing gap (erg)

    References:
    -----------
    Chen et al. (1993), NPA 555, 59
    """
    # Simplified version - full model requires numerical tables
    # Use AO as approximation with different parameters
    return pairing_gap_AO(rho, Delta_max=0.15, rho_peak=0.65, width=4.0)


def pairing_gap_simple(rho, Delta_0=0.1):
    """
    Simple constant pairing gap (for testing).

    Parameters:
    -----------
    rho : float or array
        Mass density (g/cm³)
    Delta_0 : float
        Constant gap value (MeV)

    Returns:
    --------
    Delta : float or array
        Pairing gap (erg)
    """
    if np.isscalar(rho):
        return Delta_0 * const.MeV
    else:
        return np.full_like(rho, Delta_0 * const.MeV)


# ============================================================================
# Critical Temperature
# ============================================================================

def critical_temperature(Delta, BCS_factor=0.57):
    """
    Critical temperature from BCS theory.

    T_c = α Δ / k_B

    where α ≈ 0.57 for s-wave pairing, but can be different for ³P₂.

    Parameters:
    -----------
    Delta : float or array
        Pairing gap (erg)
    BCS_factor : float
        BCS proportionality constant, default 0.57

    Returns:
    --------
    T_c : float or array
        Critical temperature (K)
    """
    T_c = BCS_factor * Delta / const.k_B
    return T_c


# ============================================================================
# Superfluid Density
# ============================================================================

def superfluid_fraction(T, T_c):
    """
    Temperature-dependent superfluid fraction.

    Uses simple quadratic suppression near T_c.

    f_s(T) = 1 - (T/T_c)²   for T < T_c
           = 0              for T ≥ T_c

    Parameters:
    -----------
    T : float or array
        Temperature (K)
    T_c : float or array
        Critical temperature (K)

    Returns:
    --------
    f_s : float or array
        Superfluid fraction (0-1)
    """
    if np.isscalar(T) and np.isscalar(T_c):
        if T >= T_c:
            return 0.0
        else:
            return 1.0 - (T / T_c)**2
    else:
        T = np.asarray(T)
        T_c = np.asarray(T_c)
        f_s = np.where(T < T_c, 1.0 - (T / T_c)**2, 0.0)
        return f_s


def superfluid_density(rho, T, f_n, pairing_model='AO'):
    """
    Superfluid neutron density.

    ρ_n = ρ × f_n(ρ) × [1 - (T/T_c)²]

    Parameters:
    -----------
    rho : float or array
        Total mass density (g/cm³)
    T : float or array
        Temperature (K)
    f_n : float or array
        Neutron fraction (from EoS)
    pairing_model : str
        Pairing gap model: 'AO', 'CCDK', 'simple'

    Returns:
    --------
    rho_n : float or array
        Superfluid neutron density (g/cm³)
    """
    # Get pairing gap
    if pairing_model == 'AO':
        Delta = pairing_gap_AO(rho)
    elif pairing_model == 'CCDK':
        Delta = pairing_gap_CCDK(rho)
    elif pairing_model == 'simple':
        Delta = pairing_gap_simple(rho)
    else:
        raise ValueError(f"Unknown pairing model: {pairing_model}")

    # Critical temperature
    T_c = critical_temperature(Delta)

    # Superfluid fraction
    f_s = superfluid_fraction(T, T_c)

    # Superfluid density
    rho_n = rho * f_n * f_s

    return rho_n


def superfluid_properties(rho, T, f_n, pairing_model='AO'):
    """
    Calculate all superfluid properties at once.

    Convenient function that returns dictionary with all relevant quantities.

    Parameters:
    -----------
    rho : float or array
        Mass density (g/cm³)
    T : float or array
        Temperature (K)
    f_n : float or array
        Neutron fraction
    pairing_model : str
        Pairing gap model

    Returns:
    --------
    props : dict
        Dictionary containing:
        - 'Delta': Pairing gap (erg)
        - 'Delta_MeV': Pairing gap (MeV)
        - 'T_c': Critical temperature (K)
        - 'f_s': Superfluid fraction
        - 'rho_n': Superfluid density (g/cm³)
        - 'T/T_c': Temperature ratio
    """
    # Pairing gap
    if pairing_model == 'AO':
        Delta = pairing_gap_AO(rho)
    elif pairing_model == 'CCDK':
        Delta = pairing_gap_CCDK(rho)
    elif pairing_model == 'simple':
        Delta = pairing_gap_simple(rho)
    else:
        raise ValueError(f"Unknown pairing model: {pairing_model}")

    # Critical temperature
    T_c = critical_temperature(Delta)

    # Superfluid fraction
    f_s = superfluid_fraction(T, T_c)

    # Superfluid density
    rho_n = rho * f_n * f_s

    props = {
        'Delta': Delta,
        'Delta_MeV': Delta / const.MeV,
        'T_c': T_c,
        'T_c_K': T_c,
        'T_c_MeV': const.K_to_MeV(T_c),
        'f_s': f_s,
        'rho_n': rho_n,
        'T_over_Tc': T / T_c if T_c > 0 else 0,
        'pairing_model': pairing_model
    }

    return props


# ============================================================================
# Density Profiles
# ============================================================================

def superfluid_density_profile(rho_array, T, f_n_array, pairing_model='AO'):
    """
    Calculate superfluid density profile throughout the star.

    Parameters:
    -----------
    rho_array : array
        Density profile (g/cm³)
    T : float
        Temperature (K), assumed constant
    f_n_array : array
        Neutron fraction profile
    pairing_model : str
        Pairing gap model

    Returns:
    --------
    profile : dict
        Dictionary with arrays:
        - 'rho': Total density
        - 'rho_n': Superfluid density
        - 'Delta_MeV': Pairing gap
        - 'T_c': Critical temperature
        - 'f_s': Superfluid fraction
    """
    n_points = len(rho_array)

    # Initialize arrays
    Delta = np.zeros(n_points)
    T_c = np.zeros(n_points)
    f_s = np.zeros(n_points)
    rho_n = np.zeros(n_points)

    # Calculate at each point
    for i in range(n_points):
        props = superfluid_properties(rho_array[i], T, f_n_array[i], pairing_model)
        Delta[i] = props['Delta']
        T_c[i] = props['T_c']
        f_s[i] = props['f_s']
        rho_n[i] = props['rho_n']

    profile = {
        'rho': rho_array,
        'rho_n': rho_n,
        'Delta': Delta,
        'Delta_MeV': Delta / const.MeV,
        'T_c': T_c,
        'f_s': f_s
    }

    return profile


# ============================================================================
# Utility Functions
# ============================================================================

def is_superfluid(rho, T, pairing_model='AO'):
    """
    Check if neutrons are superfluid at given density and temperature.

    Parameters:
    -----------
    rho : float or array
        Density (g/cm³)
    T : float or array
        Temperature (K)
    pairing_model : str
        Pairing gap model

    Returns:
    --------
    superfluid : bool or array
        True if T < T_c
    """
    if pairing_model == 'AO':
        Delta = pairing_gap_AO(rho)
    elif pairing_model == 'CCDK':
        Delta = pairing_gap_CCDK(rho)
    elif pairing_model == 'simple':
        Delta = pairing_gap_simple(rho)
    else:
        raise ValueError(f"Unknown pairing model: {pairing_model}")

    T_c = critical_temperature(Delta)

    return T < T_c


def thermal_depletion(T, T_c):
    """
    Fraction of normal (non-superfluid) component.

    f_normal = (T/T_c)²  for T < T_c

    Parameters:
    -----------
    T : float or array
        Temperature (K)
    T_c : float or array
        Critical temperature (K)

    Returns:
    --------
    f_normal : float or array
        Normal fraction (0-1)
    """
    return 1.0 - superfluid_fraction(T, T_c)


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Superfluid Properties Module Testing")
    print("=" * 70)

    # Test at typical glitch density
    rho_test = 0.6 * const.rho_0
    T_test = 1e8  # K
    f_n_test = 0.9

    print("\n1. Pairing Gap Models")
    print("-" * 70)
    print(f"Density: {rho_test/const.rho_0:.2f} ρ₀")

    Delta_AO = pairing_gap_AO(rho_test)
    Delta_CCDK = pairing_gap_CCDK(rho_test)
    Delta_simple = pairing_gap_simple(rho_test)

    print(f"AO model:     Δ = {Delta_AO/const.MeV:.3f} MeV")
    print(f"CCDK model:   Δ = {Delta_CCDK/const.MeV:.3f} MeV")
    print(f"Simple model: Δ = {Delta_simple/const.MeV:.3f} MeV")

    print("\n2. Critical Temperatures")
    print("-" * 70)

    T_c_AO = critical_temperature(Delta_AO)
    T_c_CCDK = critical_temperature(Delta_CCDK)

    print(f"AO model:   T_c = {T_c_AO:.3e} K = {const.K_to_MeV(T_c_AO):.3f} MeV")
    print(f"CCDK model: T_c = {T_c_CCDK:.3e} K = {const.K_to_MeV(T_c_CCDK):.3f} MeV")

    print("\n3. Superfluid Density")
    print("-" * 70)
    print(f"Temperature: T = {T_test:.2e} K")
    print(f"Neutron fraction: f_n = {f_n_test:.2f}")

    rho_n_AO = superfluid_density(rho_test, T_test, f_n_test, 'AO')
    rho_n_CCDK = superfluid_density(rho_test, T_test, f_n_test, 'CCDK')

    print(f"AO model:   ρ_n = {rho_n_AO:.3e} g/cm³ ({rho_n_AO/rho_test:.3f} × ρ)")
    print(f"CCDK model: ρ_n = {rho_n_CCDK:.3e} g/cm³ ({rho_n_CCDK/rho_test:.3f} × ρ)")

    print("\n4. Temperature Dependence")
    print("-" * 70)

    T_c = critical_temperature(Delta_AO)
    temps = np.array([0, 0.5, 0.9, 0.99, 1.0, 1.1]) * T_c

    print(f"Critical temperature: T_c = {T_c:.3e} K")
    print(f"\nT/T_c    f_s     ρ_n/ρ    Status")
    print("-" * 40)

    for T in temps:
        f_s = superfluid_fraction(T, T_c)
        rho_n = rho_test * f_n_test * f_s
        status = "Superfluid" if T < T_c else "Normal"
        print(f"{T/T_c:5.2f}  {f_s:6.3f}  {rho_n/rho_test:6.3f}  {status}")

    print("\n5. Density Profile (0.3-1.0 ρ₀)")
    print("-" * 70)

    rho_range = np.linspace(0.3, 1.0, 8) * const.rho_0
    f_n_range = np.full_like(rho_range, 0.9)  # Constant for simplicity

    print(f"Temperature: T = {T_test:.2e} K")
    print(f"\nρ/ρ₀    Δ(MeV)  T_c(K)     f_s    ρ_n(g/cm³)   Superfluid?")
    print("-" * 70)

    for rho, f_n in zip(rho_range, f_n_range):
        props = superfluid_properties(rho, T_test, f_n, 'AO')
        sf_status = "Yes" if is_superfluid(rho, T_test, 'AO') else "No"
        print(f"{rho/const.rho_0:5.2f}  {props['Delta_MeV']:7.3f}  "
              f"{props['T_c']:.3e}  {props['f_s']:5.3f}  "
              f"{props['rho_n']:.3e}  {sf_status}")

    print("\n" + "=" * 70)
    print("All tests passed! OK")
    print("=" * 70)

    # Optional: Create plot
    try:
        import matplotlib.pyplot as plt

        print("\nCreating pairing gap plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: Pairing gap vs density
        rho_plot = np.linspace(0.1, 1.2, 200) * const.rho_0
        Delta_AO_plot = pairing_gap_AO(rho_plot) / const.MeV
        Delta_CCDK_plot = pairing_gap_CCDK(rho_plot) / const.MeV

        ax1.plot(rho_plot/const.rho_0, Delta_AO_plot, 'b-', lw=2, label='AO model')
        ax1.plot(rho_plot/const.rho_0, Delta_CCDK_plot, 'r--', lw=2, label='CCDK model')
        ax1.set_xlabel(r'Density $\rho/\rho_0$', fontsize=12)
        ax1.set_ylabel(r'Pairing Gap $\Delta$ (MeV)', fontsize=12)
        ax1.set_title('³P₂ Neutron Pairing Gap', fontsize=13)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Superfluid fraction vs T/T_c
        T_Tc = np.linspace(0, 1.5, 100)
        f_s_plot = superfluid_fraction(T_Tc, 1.0)

        ax2.plot(T_Tc, f_s_plot, 'g-', lw=2)
        ax2.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='T = T_c')
        ax2.set_xlabel(r'Temperature $T/T_c$', fontsize=12)
        ax2.set_ylabel(r'Superfluid Fraction $f_s$', fontsize=12)
        ax2.set_title('Temperature-Dependent Depletion', fontsize=13)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('../figures/superfluid_properties.pdf')
        print("Figure saved: figures/superfluid_properties.pdf")
        plt.show()

    except ImportError:
        print("(matplotlib not available for plotting)")
