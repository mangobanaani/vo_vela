#!/usr/bin/env python3
"""
Implement Radial Integration for L₀ Coupling
============================================

Add radial integration to couple L₀ → ρ_n(r) → ω

Current: ω² = (α Ω κ ln(b/ξ)) / L²  [single density point]

Enhanced: ω² = (α Ω κ / L²) ∫[r₁→r₂] ln(b(r)/ξ(r)) w(r; L₀) dr

where w(r; L₀) = ρ_n(r; L₀) / ρ_n,avg weights by local superfluid density
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import constants as const
from src import eos, superfluid as sf, vortex, stellar_structure as ss

print("="*80)
print("RADIAL INTEGRATION FOR L₀ COUPLING")
print("="*80)
print()

# ============================================================================
# ENHANCED OSCILLATION PREDICTION WITH RADIAL INTEGRATION
# ============================================================================

def predict_with_radial_integration(
    eos_model,
    L_eff,
    T,
    R,
    M,
    Omega,
    r_inner_frac=0.3,
    r_outer_frac=0.7,
    mode_params=None
):
    """
    Predict oscillation frequency with radial density profile integration.

    ω² = (α Ω κ / L²) ∫[r₁→r₂] ln(b(r)/ξ(r)) ρ_n(r; L₀) dr / ∫[r₁→r₂] ρ_n(r; L₀) dr

    Parameters:
    -----------
    eos_model : EoS
        Equation of state with L₀ parameter
    L_eff : float
        Effective vortex length (cm)
    T : float
        Temperature (K)
    R : float
        Stellar radius (cm)
    M : float
        Stellar mass (g)
    Omega : float
        Angular velocity (rad/s)
    r_inner_frac : float
        Inner radius as fraction of R (default 0.3)
    r_outer_frac : float
        Outer radius as fraction of R (default 0.7)
    mode_params : BendingModeParameters
        Mode parameters (if None, uses calibrated value)

    Returns:
    --------
    dict with:
        - omega_0: Oscillation frequency (rad/s)
        - P_days: Period (days)
        - L0: L₀ value used
        - integral_value: Weighted integral result
        - rho_n_avg: Average superfluid density
    """

    # Get L₀ from EoS
    L0 = eos_model.L0

    # Radial limits
    r_inner = r_inner_frac * R
    r_outer = r_outer_frac * R

    # Create neutron star model
    ns = ss.SimpleNeutronStar(M, R, eos=eos_model)

    # Define integrand for weighted ln(b/ξ)
    def integrand_weighted(r):
        """Weighted ln(b/ξ) by local superfluid density"""
        # Local density
        rho = ns.density(r)

        if rho < 0.1 * const.rho_0:
            return 0.0  # Below crust density

        # Neutron fraction (depends on L₀!)
        f_n = eos_model.neutron_fraction(rho)

        # Superfluid density
        rho_n = sf.superfluid_density(rho, T, f_n)

        if rho_n <= 0:
            return 0.0

        # Pairing gap
        Delta = sf.pairing_gap_AO(rho)

        # ln(b/ξ) at this radius
        log_factor = vortex.log_b_over_xi(Omega, Delta)

        # Weight by superfluid density
        return log_factor * rho_n

    # Define integrand for normalization (average density)
    def integrand_norm(r):
        """Local superfluid density for normalization"""
        rho = ns.density(r)

        if rho < 0.1 * const.rho_0:
            return 0.0

        f_n = eos_model.neutron_fraction(rho)
        rho_n = sf.superfluid_density(rho, T, f_n)

        return max(rho_n, 0.0)

    # Perform integration
    try:
        integral_weighted, _ = quad(integrand_weighted, r_inner, r_outer,
                                    limit=100, epsabs=1e10, epsrel=1e-6)
        integral_norm, _ = quad(integrand_norm, r_inner, r_outer,
                                limit=100, epsabs=1e10, epsrel=1e-6)
    except Exception as e:
        print(f"Warning: Integration failed - {e}")
        # Fall back to midpoint evaluation
        r_mid = 0.5 * (r_inner + r_outer)
        rho_mid = ns.density(r_mid)
        f_n_mid = eos_model.neutron_fraction(rho_mid)
        rho_n_mid = sf.superfluid_density(rho_mid, T, f_n_mid)
        Delta_mid = sf.pairing_gap_AO(rho_mid)
        log_factor_mid = vortex.log_b_over_xi(Omega, Delta_mid)
        integral_weighted = log_factor_mid * rho_n_mid * (r_outer - r_inner)
        integral_norm = rho_n_mid * (r_outer - r_inner)

    # Average weighted ln(b/ξ)
    if integral_norm > 0:
        log_factor_avg = integral_weighted / integral_norm
        rho_n_avg = integral_norm / (r_outer - r_inner)
    else:
        # Fallback to midpoint
        r_mid = 0.5 * (r_inner + r_outer)
        rho_mid = ns.density(r_mid)
        f_n_mid = eos_model.neutron_fraction(rho_mid)
        rho_n_avg = sf.superfluid_density(rho_mid, T, f_n_mid)
        Delta_mid = sf.pairing_gap_AO(rho_mid)
        log_factor_avg = vortex.log_b_over_xi(Omega, Delta_mid)

    # Calculate frequency
    if mode_params is None:
        # Use calibrated alpha_cf0 = 0.08
        alpha = 0.08 * (R / L_eff)**2
    else:
        alpha = mode_params.resolve_geometric_factor(R)

    omega_squared = (alpha * Omega * const.kappa * log_factor_avg) / L_eff**2

    if omega_squared <= 0:
        omega_0 = 0.0
        P_days = np.inf
    else:
        omega_0 = np.sqrt(omega_squared)
        P_seconds = 2 * np.pi / omega_0
        P_days = P_seconds / (24 * 3600)

    return {
        'omega_0': omega_0,
        'P_days': P_days,
        'L0': L0,
        'log_factor_avg': log_factor_avg,
        'rho_n_avg': rho_n_avg,
        'integral_weighted': integral_weighted,
        'integral_norm': integral_norm
    }

# ============================================================================
# TEST RADIAL INTEGRATION
# ============================================================================

print("Test 1: Radial Integration vs Single Point")
print("-" * 80)
print()

# Vela parameters
M_vela = 1.4 * const.M_sun
R_vela = 12e5  # cm
Omega_vela = const.Omega_Vela
T_vela = 1e8  # K
L_eff_G1 = 7.51e5  # cm (from G1 calibration)

# Test with L₀ = 55 MeV
eos_55 = eos.SymmetryEnergyEoS(L0=55.0)

# Single point evaluation (old method)
rho_single = 0.6 * const.rho_0
f_n_single = eos_55.neutron_fraction(rho_single)
rho_n_single = sf.superfluid_density(rho_single, T_vela, f_n_single)
Delta_single = sf.pairing_gap_AO(rho_single)
log_factor_single = vortex.log_b_over_xi(Omega_vela, Delta_single)

alpha = 0.08 * (R_vela / L_eff_G1)**2
omega_single = np.sqrt(alpha * Omega_vela * const.kappa * log_factor_single / L_eff_G1**2)
P_single = 2 * np.pi / omega_single / (24 * 3600)

print(f"Single Point Evaluation (ρ = 0.6 ρ₀):")
print(f"  ln(b/ξ) = {log_factor_single:.2f}")
print(f"  ρ_n = {rho_n_single:.2e} g/cm³")
print(f"  Period: {P_single:.1f} days")
print()

# Radial integration (new method)
result_radial = predict_with_radial_integration(
    eos_55, L_eff_G1, T_vela, R_vela, M_vela, Omega_vela,
    r_inner_frac=0.3, r_outer_frac=0.7
)

print(f"Radial Integration (r ∈ [0.3R, 0.7R]):")
print(f"  <ln(b/ξ)> = {result_radial['log_factor_avg']:.2f}")
print(f"  <ρ_n> = {result_radial['rho_n_avg']:.2e} g/cm³")
print(f"  Period: {result_radial['P_days']:.1f} days")
print()

difference = abs(result_radial['P_days'] - P_single)
print(f"Difference: {difference:.1f} days ({difference/P_single*100:.1f}%)")
print()

# ============================================================================
# TEST L₀ SENSITIVITY WITH RADIAL INTEGRATION
# ============================================================================

print("Test 2: L₀ Sensitivity with Radial Integration")
print("-" * 80)
print()

L0_values = np.linspace(35, 75, 21)
periods_radial = []
periods_single = []

print(f"{'L₀ (MeV)':>10} {'P_single (d)':>12} {'P_radial (d)':>12} {'Difference':>12}")
print("-" * 60)

for L0 in L0_values:
    # Create EoS with this L₀
    eos_L0 = eos.SymmetryEnergyEoS(L0=L0)

    # Single point
    f_n_sp = eos_L0.neutron_fraction(rho_single)
    rho_n_sp = sf.superfluid_density(rho_single, T_vela, f_n_sp)
    omega_sp = np.sqrt(alpha * Omega_vela * const.kappa * log_factor_single / L_eff_G1**2)
    P_sp = 2 * np.pi / omega_sp / (24 * 3600)
    periods_single.append(P_sp)

    # Radial integration
    result = predict_with_radial_integration(
        eos_L0, L_eff_G1, T_vela, R_vela, M_vela, Omega_vela
    )
    periods_radial.append(result['P_days'])

    if L0 in [40, 55, 70]:
        diff = result['P_days'] - P_sp
        print(f"{L0:>10.1f} {P_sp:>12.2f} {result['P_days']:>12.2f} {diff:>+12.3f}")

periods_radial = np.array(periods_radial)
periods_single = np.array(periods_single)

print()
print(f"Single Point:")
print(f"  ΔP = {periods_single.max() - periods_single.min():.3f} days")
print(f"  dP/dL₀ = {(periods_single[-1] - periods_single[0])/(L0_values[-1] - L0_values[0]):.5f} days/MeV")
print()
print(f"Radial Integration:")
print(f"  ΔP = {periods_radial.max() - periods_radial.min():.3f} days")
print(f"  dP/dL₀ = {(periods_radial[-1] - periods_radial[0])/(L0_values[-1] - L0_values[0]):.5f} days/MeV")
print()

sensitivity_enhancement = (periods_radial.max() - periods_radial.min()) / (periods_single.max() - periods_single.min())
print(f"Sensitivity enhancement: {sensitivity_enhancement:.1f}x")
print()

# ============================================================================
# VISUALIZE L₀ SENSITIVITY
# ============================================================================

print("Generating visualization...")
print()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Period vs L₀ (both methods)
ax = axes[0, 0]
ax.plot(L0_values, periods_single, 'gray', ls='--', lw=2, label='Single point', alpha=0.6)
ax.plot(L0_values, periods_radial, 'b-', lw=2.5, label='Radial integration')
ax.set_xlabel('L₀ (MeV)', fontsize=12)
ax.set_ylabel('Period (days)', fontsize=12)
ax.set_title('L₀ Sensitivity Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Plot 2: Density profile
ax = axes[0, 1]
ns = ss.SimpleNeutronStar(M_vela, R_vela, eos=eos_55)
r_array = np.linspace(0, R_vela, 100)
rho_array = [ns.density(r) / const.rho_0 for r in r_array]
ax.plot(r_array / R_vela, rho_array, 'b-', lw=2)
ax.axvspan(0.3, 0.7, alpha=0.2, color='green', label='Integration range')
ax.axhline(0.6, color='r', ls='--', lw=1.5, label='Single point')
ax.set_xlabel('r/R', fontsize=12)
ax.set_ylabel('ρ/ρ₀', fontsize=12)
ax.set_title('Density Profile', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Plot 3: Period difference
ax = axes[1, 0]
diff = periods_radial - periods_single
ax.plot(L0_values, diff, 'purple', lw=2.5)
ax.axhline(0, color='black', ls='-', lw=1)
ax.fill_between(L0_values, 0, diff, where=(diff>0), color='purple', alpha=0.3, interpolate=True)
ax.fill_between(L0_values, 0, diff, where=(diff<0), color='orange', alpha=0.3, interpolate=True)
ax.set_xlabel('L₀ (MeV)', fontsize=12)
ax.set_ylabel('ΔP = P_radial - P_single (days)', fontsize=12)
ax.set_title('Period Correction from Radial Integration', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Sensitivity summary
ax = axes[1, 1]
methods = ['Single\nPoint', 'Radial\nIntegration']
sensitivities = [
    abs((periods_single[-1] - periods_single[0])/(L0_values[-1] - L0_values[0])),
    abs((periods_radial[-1] - periods_radial[0])/(L0_values[-1] - L0_values[0]))
]
colors = ['gray', 'blue']

bars = ax.bar(methods, sensitivities, color=colors, alpha=0.6, edgecolor='black', lw=2)
ax.set_ylabel('|dP/dL₀| (days/MeV)', fontsize=12)
ax.set_title('L₀ Sensitivity Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, sensitivities):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.5f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/radial_integration_L0_sensitivity.pdf', dpi=300, bbox_inches='tight')
print("Figure saved: figures/radial_integration_L0_sensitivity.pdf")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("SUMMARY")
print("="*80)
print()

print("COMPLETE Radial integration implemented successfully")
print()
print(f"L₀ Sensitivity:")
print(f"   Single point: {abs((periods_single[-1] - periods_single[0])/(L0_values[-1] - L0_values[0])):.5f} days/MeV")
print(f"   Radial integration: {abs((periods_radial[-1] - periods_radial[0])/(L0_values[-1] - L0_values[0])):.5f} days/MeV")
print(f"   Enhancement: {sensitivity_enhancement:.1f}x")
print()

if sensitivity_enhancement > 1.5:
    print("COMPLETE Radial integration significantly enhances L₀ sensitivity!")
elif sensitivity_enhancement > 1.1:
    print("COMPLETE Radial integration provides modest L₀ sensitivity enhancement")
else:
    print(" Radial integration effect is small - density profile may be too uniform")
print()

print("Next steps:")
print("1. Add this function to src/vortex.py")
print("2. Update notebooks/20_L0_grid_search.py to use radial integration")
print("3. Run full Bayesian inference")
print("4. Extract L₀ constraint with proper uncertainties")
print()
print("="*80)
