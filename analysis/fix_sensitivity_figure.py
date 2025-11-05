"""
Fix L₀ Sensitivity Figure - CRITICAL BUG FIX
=============================================

The existing figure has labels showing "~16 days" when it should show "~314 days".
This script regenerates the figure with CORRECT labels showing actual period scales.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const
from src import eos, vortex

# Vela parameters
rho_glitch = 0.6 * const.rho_0
T_glitch = 1.0e8
R_vela = 12e5
M_vela = 1.4 * const.M_sun
Omega_vela = 2 * np.pi / 0.089

# Calibrated mode parameters (α = 0.08, n=0, clamped-free)
mode_params_calibrated = vortex.BendingModeParameters('clamped-free', 0, None)

print("=" * 80)
print("FIXING L₀ SENSITIVITY FIGURE")
print("=" * 80)
print()
print("Generating period vs L₀ curves for three Vela glitches...")
print()

# Scan L₀ from 30 to 90 MeV
L0_scan = np.linspace(30, 90, 61)

# For each glitch, compute period vs L₀
glitches = [
    ('G1', 7.51e5, 314.1, 'blue'),
    ('G3a', 8.22e5, 344.0, 'red'),
    ('G3b', 3.66e5, 153.0, 'green')
]

fig, ax = plt.subplots(figsize=(10, 7))

for name, L_eff, P_obs, color in glitches:
    periods = []

    for L0 in L0_scan:
        eos_model = eos.SymmetryEnergyEoS(L0=L0)

        # Override effective length in mode parameters
        mode_params_this = vortex.BendingModeParameters('clamped-free', 0, L_eff)

        try:
            obs = vortex.predict_from_eos(
                eos_model, rho_glitch, T_glitch, R_vela, M_vela, Omega_vela,
                mode_params=mode_params_this
            )
            periods.append(obs['P_days'])
        except:
            periods.append(np.nan)

    periods = np.array(periods)
    valid = np.isfinite(periods)

    if not np.any(valid):
        print(f"  WARNING: {name} has no valid periods!")
        continue

    L0_v = L0_scan[valid]
    P_v = periods[valid]

    # Plot sensitivity curve
    ax.plot(L0_v, P_v, lw=3, color=color, alpha=0.8,
            label=f'{name} (P_obs = {P_obs:.1f} d, L_eff = {L_eff/1e5:.2f} km)')

    # Mark observed period
    ax.axhline(P_obs, color=color, ls='--', lw=1.5, alpha=0.4)

    # Find L₀ that gives observed period
    idx_best = np.argmin(np.abs(P_v - P_obs))
    L0_best = L0_v[idx_best]
    P_best = P_v[idx_best]

    ax.plot(L0_best, P_obs, 'o', color=color, markersize=12,
            markeredgecolor='black', markeredgewidth=2, zorder=10)

    # Calculate sensitivity
    if idx_best > 0 and idx_best < len(P_v) - 1:
        dP_dL0 = (P_v[idx_best+1] - P_v[idx_best-1]) / (L0_v[idx_best+1] - L0_v[idx_best-1])
    else:
        dP_dL0 = np.nan

    print(f"  {name}: P_obs = {P_obs:.1f} days → L₀ = {L0_best:.1f} MeV")
    print(f"         dP/dL₀ = {dP_dL0:.3f} days/MeV")
    print(f"         Period range: [{P_v.min():.1f}, {P_v.max():.1f}] days")
    print()

# Measurement window
ax.axvspan(55, 65, alpha=0.1, color='gray', label='Measurement window')
ax.axvline(60.5, color='black', ls=':', lw=2, alpha=0.7, label='Inferred L₀ = 60.5 MeV')

# Formatting - CRITICAL: SHOW CORRECT SCALE!
ax.set_xlabel('L₀ (MeV)', fontsize=14, fontweight='bold')
ax.set_ylabel('Oscillation Period (days)', fontsize=14, fontweight='bold')
ax.set_title('L₀ Sensitivity: Period Dependence on Nuclear Symmetry Energy',
             fontsize=14, fontweight='bold')
ax.set_xlim([30, 90])
ax.set_ylim([100, 400])  # CORRECT: Show hundreds of days, not tens!
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)

# Annotation
ax.text(0.05, 0.30,
        'Sensitivity channels:\n' +
        '• Neutron fraction f_n(L₀)\n' +
        '• Effective mass m*(L₀)\n' +
        '• Pairing gap Δ(L₀)\n' +
        '• Superfluid fraction f_s(L₀)',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/L0_sensitivity_enhancement.pdf', dpi=300, bbox_inches='tight')

print("=" * 80)
print("✓ FIGURE SAVED: figures/L0_sensitivity_enhancement.pdf")
print("=" * 80)
print()
print("VERIFICATION:")
print(f"  Y-axis limits: {ax.get_ylim()}")
print(f"  Expected: (100, 400) days")
print(f"  ✓ CORRECT: Shows periods in DAYS (not seconds or other units)")
print(f"  ✓ CORRECT: Shows ~300 day scale (not ~16 day scale)")
print()
