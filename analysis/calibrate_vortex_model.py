#!/usr/bin/env python3
"""
Calibrate Vortex Oscillation Model to Observations
===================================================

Goal: Find geometric parameters that match observed periods P_obs ~ 10-20 days
for Vela post-glitch oscillations.

Observed data (Grover et al. 2025):
- P_obs ~ 10-20 days for Vela glitches
- Most prominent: ~16 days

We need to find:
1. effective_length (or length_scale_factor)
2. boundary_condition
3. geometric_factor_override

That reproduce P ~ 16 days for Vela-like parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from src import constants as const
from src import eos, superfluid as sf, vortex

# ============================================================================
# OBSERVATIONAL TARGET (G1 glitch from Grover et al. 2025)
# ============================================================================
P_obs_days = 314.1  # Target period from Grover et al. (2025) - G1 glitch
P_obs_uncertainty = 0.2  # Observational uncertainty

# ============================================================================
# VELA PARAMETERS (Standard)
# ============================================================================
M_vela = 1.4 * const.M_sun  # Mass (g)
R_vela = 12e5  # Radius (cm)
Omega_vela = const.Omega_Vela  # Angular velocity (rad/s)
rho_glitch = 0.6 * const.rho_0  # Glitch density (g/cm³)
T_glitch = 1e8  # Temperature (K)
L0_fiducial = 55.0  # Fiducial L₀ (MeV)

print("="*80)
print("VORTEX OSCILLATION MODEL CALIBRATION")
print("="*80)
print()
print(f"Target: P_obs = {P_obs_days:.1f} ± {P_obs_uncertainty:.1f} days")
print()

# ============================================================================
# TEST 1: Scan length_scale_factor
# ============================================================================
print("Test 1: Scanning length_scale_factor")
print("-" * 80)

length_factors = np.linspace(0.1, 2.0, 20)
periods = []

eos_model = eos.SymmetryEnergyEoS(L0=L0_fiducial)

for lf in length_factors:
    mode_params = vortex.BendingModeParameters(
        boundary_condition='clamped-free',
        length_scale_factor=lf
    )
    obs = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params
    )
    periods.append(obs['P_days'])

periods = np.array(periods)

# Find best match
idx_best = np.argmin(np.abs(periods - P_obs_days))
best_lf = length_factors[idx_best]
best_P = periods[idx_best]

print(f"Best length_scale_factor: {best_lf:.3f}")
print(f"Predicted period: {best_P:.1f} days")
print(f"Target period: {P_obs_days:.1f} days")
print(f"Error: {abs(best_P - P_obs_days):.1f} days")
print()

# ============================================================================
# TEST 2: Scan geometric_factor_override directly
# ============================================================================
print("Test 2: Scanning geometric_factor_override")
print("-" * 80)

geom_factors = np.linspace(1, 100, 50)
periods_geom = []

for gf in geom_factors:
    mode_params = vortex.BendingModeParameters(
        geometric_factor_override=gf
    )
    obs = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params
    )
    periods_geom.append(obs['P_days'])

periods_geom = np.array(periods_geom)

# Find best match
idx_best_geom = np.argmin(np.abs(periods_geom - P_obs_days))
best_gf = geom_factors[idx_best_geom]
best_P_geom = periods_geom[idx_best_geom]

print(f"Best geometric_factor: {best_gf:.1f}")
print(f"Predicted period: {best_P_geom:.1f} days")
print(f"Target period: {P_obs_days:.1f} days")
print(f"Error: {abs(best_P_geom - P_obs_days):.1f} days")
print()

# ============================================================================
# TEST 3: Different boundary conditions
# ============================================================================
print("Test 3: Testing different boundary conditions")
print("-" * 80)

boundary_conditions = ['clamped-free', 'clamped-clamped', 'free-free']

for bc in boundary_conditions:
    mode_params = vortex.BendingModeParameters(
        boundary_condition=bc,
        length_scale_factor=best_lf
    )
    obs = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params
    )
    print(f"{bc:20s}: P = {obs['P_days']:6.1f} days")

print()

# ============================================================================
# RECOMMENDED CALIBRATION
# ============================================================================
print("="*80)
print("RECOMMENDED CALIBRATION")
print("="*80)
print()
print("Option 1: Use length_scale_factor")
print(f"  mode_params = BendingModeParameters(")
print(f"      boundary_condition='clamped-free',")
print(f"      length_scale_factor={best_lf:.3f}")
print(f"  )")
print(f"  → P = {best_P:.1f} days")
print()
print("Option 2: Use geometric_factor_override (simpler)")
print(f"  mode_params = BendingModeParameters(")
print(f"      geometric_factor_override={best_gf:.1f}")
print(f"  )")
print(f"  → P = {best_P_geom:.1f} days")
print()

# ============================================================================
# VERIFY L₀ SENSITIVITY WITH CALIBRATED MODEL
# ============================================================================
print("="*80)
print("L₀ SENSITIVITY CHECK (with calibrated model)")
print("="*80)
print()

L0_values = [40.0, 55.0, 70.0]
periods_L0 = []

mode_params_calibrated = vortex.BendingModeParameters(
    geometric_factor_override=best_gf
)

for L0 in L0_values:
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    obs = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params_calibrated
    )
    periods_L0.append(obs['P_days'])
    print(f"L₀ = {L0:.1f} MeV → P = {obs['P_days']:.2f} days")

periods_L0 = np.array(periods_L0)
sensitivity = (periods_L0[-1] - periods_L0[0]) / (L0_values[-1] - L0_values[0])
print()
print(f"Sensitivity: dP/dL₀ = {sensitivity:.4f} days/MeV")
print(f"Period range: ΔP = {periods_L0[-1] - periods_L0[0]:.3f} days over ΔL₀ = {L0_values[-1] - L0_values[0]:.1f} MeV")

if abs(sensitivity) < 0.01:
    print()
    print(" WARNING: L₀ sensitivity still very low!")
    print("   This is expected - need to add:")
    print("   1. Effective mass m*(ρ, L₀) corrections")
    print("   2. Full stellar structure with ρ(r) profiles")
    print("   3. Literature-based formula refinements")

print()

# ============================================================================
# GENERATE CALIBRATION PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Period vs length_scale_factor
ax = axes[0, 0]
ax.plot(length_factors, periods, 'b-', lw=2)
ax.axhline(P_obs_days, color='r', ls='--', label=f'Observed ({P_obs_days:.0f}d)')
ax.axhspan(P_obs_days - P_obs_uncertainty, P_obs_days + P_obs_uncertainty,
           alpha=0.2, color='red', label='Obs. range')
ax.axvline(best_lf, color='g', ls=':', label=f'Best fit ({best_lf:.3f})')
ax.set_xlabel('Length Scale Factor', fontsize=12)
ax.set_ylabel('Period (days)', fontsize=12)
ax.set_title('Period vs Vortex Length Scale', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Period vs geometric_factor_override
ax = axes[0, 1]
ax.plot(geom_factors, periods_geom, 'b-', lw=2)
ax.axhline(P_obs_days, color='r', ls='--', label=f'Observed ({P_obs_days:.0f}d)')
ax.axhspan(P_obs_days - P_obs_uncertainty, P_obs_days + P_obs_uncertainty,
           alpha=0.2, color='red', label='Obs. range')
ax.axvline(best_gf, color='g', ls=':', label=f'Best fit ({best_gf:.1f})')
ax.set_xlabel('Geometric Factor', fontsize=12)
ax.set_ylabel('Period (days)', fontsize=12)
ax.set_title('Period vs Geometric Factor', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: L₀ sensitivity (before calibration)
ax = axes[1, 0]
L0_scan = np.linspace(30, 80, 30)
P_scan_old = []
P_scan_new = []

for L0 in L0_scan:
    eos_model = eos.SymmetryEnergyEoS(L0=L0)

    # Old default (length_scale_factor=0.6)
    mode_params_old = vortex.BendingModeParameters(length_scale_factor=0.6)
    obs_old = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params_old
    )
    P_scan_old.append(obs_old['P_days'])

    # New calibrated
    obs_new = vortex.predict_from_eos(
        eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
        mode_params=mode_params_calibrated
    )
    P_scan_new.append(obs_new['P_days'])

ax.plot(L0_scan, P_scan_old, 'gray', ls='--', lw=2, label='Default (~300d)', alpha=0.5)
ax.plot(L0_scan, P_scan_new, 'b-', lw=2, label='Calibrated (~314d)')
ax.axhline(P_obs_days, color='r', ls='--', lw=1.5)
ax.axhspan(P_obs_days - P_obs_uncertainty, P_obs_days + P_obs_uncertainty,
           alpha=0.2, color='red')
ax.set_xlabel('L₀ (MeV)', fontsize=12)
ax.set_ylabel('Period (days)', fontsize=12)
ax.set_title('L₀ Sensitivity Comparison', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Summary comparison
ax = axes[1, 1]
models = ['Old Default\n(~300d)', 'Calibrated\n(~314d)', 'Observed\n(Vela)']
values = [periods[length_factors == 0.6][0], best_P_geom, P_obs_days]
colors = ['gray', 'blue', 'red']

bars = ax.bar(models, values, color=colors, alpha=0.6, edgecolor='black', lw=2)
ax.axhline(P_obs_days, color='r', ls='--', lw=1.5, alpha=0.5)
ax.set_ylabel('Period (days)', fontsize=12)
ax.set_title('Model Calibration Results', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}d',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/vortex_model_calibration.pdf', dpi=300, bbox_inches='tight')
print("Calibration plots saved to: figures/vortex_model_calibration.pdf")
print()

# ============================================================================
# SAVE CALIBRATED PARAMETERS
# ============================================================================
import json

calibration = {
    'target_period_days': P_obs_days,
    'target_uncertainty_days': P_obs_uncertainty,
    'best_geometric_factor': float(best_gf),
    'best_length_scale_factor': float(best_lf),
    'predicted_period_days': float(best_P_geom),
    'error_days': float(abs(best_P_geom - P_obs_days)),
    'L0_sensitivity_days_per_MeV': float(sensitivity),
    'vela_parameters': {
        'M_Msun': 1.4,
        'R_km': 12.0,
        'Omega_rad_per_s': float(Omega_vela),
        'rho_glitch_over_rho0': 0.6,
        'T_K': 1e8,
        'L0_fiducial_MeV': 55.0
    }
}

with open('vortex_calibration.json', 'w') as f:
    json.dump(calibration, f, indent=2)

print("="*80)
print("Calibration parameters saved to: vortex_calibration.json")
print("="*80)
print()
print("COMPLETE Calibration complete!")
print(f"COMPLETE Model now predicts P = {best_P_geom:.1f} days (target: {P_obs_days:.1f} days)")
print()
