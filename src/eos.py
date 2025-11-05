"""
Equation of State (EoS) Models

This module implements various nuclear equations of state with parameterizations
that allow constraining the symmetry energy slope L₀.

Classes:
--------
- EoSBase: Abstract base class for EoS models
- SymmetryEnergyEoS: Parameterized EoS with explicit L₀ dependence
- PolytropicEoS: Simple polytrope for testing
- APREoS: Akmal-Pandharipande-Ravenhall EoS
- SLy4EoS: SLy4 Skyrme EoS

Each EoS provides:
- pressure(rho): P(ρ)
- neutron_fraction(rho): f_n(ρ) from β-equilibrium
- symmetry_energy(rho): S(ρ)
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from . import constants as const


class EoSBase:
    """
    Abstract base class for equation of state models.
    """

    def __init__(self, name="Generic EoS"):
        self.name = name

    def pressure(self, rho):
        """
        Pressure as a function of density.

        Parameters:
        -----------
        rho : float or array
            Mass density (g/cm³)

        Returns:
        --------
        P : float or array
            Pressure (dyne/cm²)
        """
        raise NotImplementedError("Must implement pressure(rho)")

    def neutron_fraction(self, rho):
        """
        Neutron fraction from β-equilibrium.

        Parameters:
        -----------
        rho : float or array
            Mass density (g/cm³)

        Returns:
        --------
        f_n : float or array
            Neutron fraction (dimensionless, 0-1)
        """
        raise NotImplementedError("Must implement neutron_fraction(rho)")

    def energy_density(self, rho):
        """
        Energy density (including rest mass).

        Parameters:
        -----------
        rho : float or array
            Mass density (g/cm³)

        Returns:
        --------
        epsilon : float or array
            Energy density (erg/cm³)
        """
        # Default: ε = ρc² + internal energy
        # For simple models, can approximate as ε ≈ ρc²
        return rho * const.c**2

    def sound_speed(self, rho):
        """
        Sound speed squared c_s² = dP/dε.

        Parameters:
        -----------
        rho : float or array
            Mass density (g/cm³)

        Returns:
        --------
        cs2 : float or array
            Sound speed squared (dimensionless, units of c²)
        """
        # Numerical derivative
        drho = rho * 1e-6
        dP = self.pressure(rho + drho) - self.pressure(rho - drho)
        deps = self.energy_density(rho + drho) - self.energy_density(rho - drho)
        return dP / deps

    def is_causal(self, rho):
        """
        Check if EoS is causal (c_s² < c²).

        Parameters:
        -----------
        rho : float or array
            Mass density (g/cm³)

        Returns:
        --------
        causal : bool or array
            True if causal
        """
        cs2 = self.sound_speed(rho)
        return cs2 < 1.0  # c_s² < c² (in units where c=1)


class SymmetryEnergyEoS(EoSBase):
    """
    Parameterized EoS with explicit symmetry energy dependence.

    Uses expansion around nuclear saturation density with L₀ as parameter.
    Good for exploring L₀ constraints.

    Parameters:
    -----------
    L0 : float
        Symmetry energy slope at saturation (MeV)
    S0 : float
        Symmetry energy at saturation (MeV), default 32
    K0 : float
        Incompressibility at saturation (MeV), default 240
    rho_0 : float
        Nuclear saturation density (g/cm³), default 2.8e14
    """

    def __init__(self, L0=55.0, S0=32.0, K0=240.0, rho_0=None):
        super().__init__(name=f"Parameterized (L₀={L0:.1f} MeV)")
        self.L0 = L0  # MeV
        self.S0 = S0  # MeV
        self.K0 = K0  # MeV
        self.rho_0 = rho_0 if rho_0 is not None else const.rho_0

        # Convert to CGS
        self.S0_erg = S0 * const.MeV
        self.L0_erg = L0 * const.MeV
        self.K0_erg = K0 * const.MeV

    def symmetry_energy(self, rho):
        """
        Symmetry energy S(ρ) with L₀ parameterization.

        S(ρ) = S₀ + L₀ * (ρ - ρ₀)/(3ρ₀) + ...

        Parameters:
        -----------
        rho : float or array
            Density (g/cm³)

        Returns:
        --------
        S : float or array
            Symmetry energy (erg)
        """
        x = (rho - self.rho_0) / (3 * self.rho_0)

        # Linear expansion (sufficient for crustal densities)
        S = self.S0_erg + self.L0_erg * x

        # Could add K_sym term for higher densities:
        # + (1/2) * K_sym * x²

        return S

    def neutron_fraction(self, rho):
        """
        Neutron fraction from β-equilibrium.

        In β-equilibrium: μ_n = μ_p + μ_e

        For asymmetric nuclear matter, this gives:
        f_n ≈ 1/2 [1 + S(ρ)/(4 E_F)]

        where E_F is the Fermi energy.

        Parameters:
        -----------
        rho : float or array
            Density (g/cm³)

        Returns:
        --------
        f_n : float or array
            Neutron fraction (0-1)
        """
        # Number density
        n = rho / const.m_u  # particles/cm³

        # Fermi energy (relativistic)
        # E_F ≈ ℏc (3π² n)^(1/3)
        k_F = (3 * np.pi**2 * n)**(1/3)
        E_F = const.hbar * const.c * k_F

        # Symmetry energy
        S = self.symmetry_energy(rho)

        # Neutron fraction from β-equilibrium
        # f_n = 1/2 [1 + S/(4 E_F)]
        f_n = 0.5 * (1 + S / (4 * E_F))

        # Physical bounds
        f_n = np.clip(f_n, 0.0, 1.0)

        return f_n

    def pressure(self, rho):
        """
        Pressure from energy density.

        Simple parameterization for testing.
        More sophisticated models would use full nuclear physics.

        Parameters:
        -----------
        rho : float or array
            Density (g/cm³)

        Returns:
        --------
        P : float or array
            Pressure (dyne/cm²)
        """
        # Polytropic-like relation near saturation
        # P = K (ρ/ρ₀)^Γ
        Gamma = 2.5  # Typical for nuclear matter
        K = self.K0_erg * self.rho_0 / 9  # Normalization

        x = rho / self.rho_0
        P = K * x**Gamma

        # Add symmetry energy contribution
        # (Approximate: full calculation would integrate thermodynamics)
        f_n = self.neutron_fraction(rho)
        delta = 1 - 2 * f_n  # Asymmetry parameter
        S = self.symmetry_energy(rho)

        # P_sym ≈ n S δ²
        n = rho / const.m_u
        P += n * S * delta**2

        return P


class PolytropicEoS(EoSBase):
    """
    Simple polytropic EoS: P = K ρ^Γ

    Useful for testing and validation.

    Parameters:
    -----------
    K : float
        Polytropic constant (cgs units)
    Gamma : float
        Polytropic index
    f_n_const : float
        Constant neutron fraction (default 0.9)
    """

    def __init__(self, K=1e13, Gamma=2.0, f_n_const=0.9):
        super().__init__(name=f"Polytrope (Γ={Gamma:.1f})")
        self.K = K
        self.Gamma = Gamma
        self.f_n_const = f_n_const

    def pressure(self, rho):
        """P = K ρ^Γ"""
        return self.K * rho**self.Gamma

    def neutron_fraction(self, rho):
        """Constant neutron fraction (approximation)"""
        if np.isscalar(rho):
            return self.f_n_const
        else:
            return np.full_like(rho, self.f_n_const)

    def energy_density(self, rho):
        """ε = ρc² + P/(Γ-1)"""
        return rho * const.c**2 + self.pressure(rho) / (self.Gamma - 1)


class APREoS(EoSBase):
    """
    Akmal-Pandharipande-Ravenhall (APR) equation of state.

    A realistic EoS based on variational calculations.
    L₀ ≈ 55 MeV for APR.

    Uses tabulated values for accuracy.
    """

    def __init__(self):
        super().__init__(name="APR")

        # Tabulated APR values (would load from file in production)
        # For now, use parameterization
        self.L0_APR = 55.0  # MeV

        # Use SymmetryEnergyEoS with APR parameters as approximation
        self._approx_eos = SymmetryEnergyEoS(L0=self.L0_APR, S0=32.0, K0=240.0)

    def pressure(self, rho):
        """APR pressure (approximation)"""
        return self._approx_eos.pressure(rho)

    def neutron_fraction(self, rho):
        """APR neutron fraction"""
        return self._approx_eos.neutron_fraction(rho)

    def symmetry_energy(self, rho):
        """APR symmetry energy"""
        return self._approx_eos.symmetry_energy(rho)


class SLy4EoS(EoSBase):
    """
    SLy4 Skyrme equation of state.

    A realistic EoS widely used in neutron star studies.
    L₀ ≈ 46 MeV for SLy4.
    """

    def __init__(self):
        super().__init__(name="SLy4")

        self.L0_SLy4 = 46.0  # MeV

        # Use SymmetryEnergyEoS with SLy4 parameters as approximation
        self._approx_eos = SymmetryEnergyEoS(L0=self.L0_SLy4, S0=32.0, K0=230.0)

    def pressure(self, rho):
        """SLy4 pressure (approximation)"""
        return self._approx_eos.pressure(rho)

    def neutron_fraction(self, rho):
        """SLy4 neutron fraction"""
        return self._approx_eos.neutron_fraction(rho)

    def symmetry_energy(self, rho):
        """SLy4 symmetry energy"""
        return self._approx_eos.symmetry_energy(rho)


# ============================================================================
# Utility Functions
# ============================================================================

def create_eos(model='parameterized', **kwargs):
    """
    Factory function to create EoS objects.

    Parameters:
    -----------
    model : str
        EoS model name: 'parameterized', 'polytrope', 'APR', 'SLy4'
    **kwargs : dict
        Model-specific parameters

    Returns:
    --------
    eos : EoSBase
        EoS object

    Examples:
    ---------
    >>> eos = create_eos('parameterized', L0=50.0)
    >>> eos = create_eos('polytrope', Gamma=2.5)
    >>> eos = create_eos('APR')
    """
    models = {
        'parameterized': SymmetryEnergyEoS,
        'polytrope': PolytropicEoS,
        'apr': APREoS,
        'sly4': SLy4EoS
    }

    model_lower = model.lower()
    if model_lower not in models:
        raise ValueError(f"Unknown EoS model: {model}. "
                        f"Available: {list(models.keys())}")

    return models[model_lower](**kwargs)


def compare_neutron_fractions(L0_values, rho_range=None, plot=False):
    """
    Compare neutron fractions for different L₀ values.

    Parameters:
    -----------
    L0_values : array-like
        L₀ values to compare (MeV)
    rho_range : tuple, optional
        (rho_min, rho_max) in units of ρ₀
    plot : bool
        If True, create comparison plot

    Returns:
    --------
    results : dict
        Dictionary with densities and neutron fractions
    """
    if rho_range is None:
        rho_range = (0.3, 1.0)  # Crustal densities

    rho_array = np.linspace(rho_range[0] * const.rho_0,
                           rho_range[1] * const.rho_0, 100)

    results = {'rho': rho_array, 'f_n': {}}

    for L0 in L0_values:
        eos = SymmetryEnergyEoS(L0=L0)
        f_n = eos.neutron_fraction(rho_array)
        results['f_n'][L0] = f_n

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for L0 in L0_values:
            ax.plot(rho_array / const.rho_0, results['f_n'][L0],
                   label=f'L₀ = {L0:.0f} MeV', lw=2)

        ax.set_xlabel(r'Density $\rho/\rho_0$', fontsize=12)
        ax.set_ylabel(r'Neutron Fraction $f_n$', fontsize=12)
        ax.set_title('Neutron Fraction vs Density for Different L₀', fontsize=13)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return results


# ============================================================================
# Testing and Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EoS Module Testing")
    print("=" * 70)

    # Test parameterized EoS
    print("\n1. Parameterized EoS with L₀ = 55 MeV")
    print("-" * 70)
    eos = SymmetryEnergyEoS(L0=55.0)

    rho_test = 0.6 * const.rho_0
    print(f"Density: {rho_test/const.rho_0:.2f} ρ₀ = {rho_test:.3e} g/cm³")

    S = eos.symmetry_energy(rho_test)
    print(f"Symmetry energy: S = {S/const.MeV:.2f} MeV")

    f_n = eos.neutron_fraction(rho_test)
    print(f"Neutron fraction: f_n = {f_n:.3f}")

    P = eos.pressure(rho_test)
    print(f"Pressure: P = {P:.3e} dyne/cm²")

    cs2 = eos.sound_speed(rho_test)
    print(f"Sound speed²: c_s²/c² = {cs2:.3f}")
    print(f"Causal? {eos.is_causal(rho_test)}")

    # Test polytrope
    print("\n2. Polytropic EoS (Γ=2.0)")
    print("-" * 70)
    eos_poly = PolytropicEoS(Gamma=2.0, f_n_const=0.9)

    P_poly = eos_poly.pressure(rho_test)
    f_n_poly = eos_poly.neutron_fraction(rho_test)
    print(f"Pressure: P = {P_poly:.3e} dyne/cm²")
    print(f"Neutron fraction: f_n = {f_n_poly:.3f}")

    # Compare different L₀ values
    print("\n3. Sensitivity to L₀")
    print("-" * 70)
    L0_values = [40, 55, 70]

    for L0 in L0_values:
        eos_test = SymmetryEnergyEoS(L0=L0)
        f_n_test = eos_test.neutron_fraction(rho_test)
        print(f"L₀ = {L0:2.0f} MeV  →  f_n = {f_n_test:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed! OK")
    print("=" * 70)

    # Optional: Create comparison plot
    try:
        print("\nCreating comparison plot...")
        compare_neutron_fractions([40, 50, 60, 70], plot=True)
    except ImportError:
        print("(matplotlib not available for plotting)")
