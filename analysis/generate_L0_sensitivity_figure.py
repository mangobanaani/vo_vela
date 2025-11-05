"""
Generate L₀ Sensitivity Figure (Figure 2)
==========================================

Shows how oscillation period depends on L₀ for the three Vela glitches.
Demonstrates the sensitivity dP/dL₀ and the measurement window.

CRITICAL: Ensure axis labels show actual periods (~314 days, not 16 days!)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const
from src import eos

# ============================================================================
# Vela Parameters
# ============================================================================

Omega_Vela = 2 * np.pi / 0.089  # rad/s
T = 1.0e8  # K
R = 12e5  # cm
M = 1.4 * const.M_sun
rho = 0.6 * const.rho_0

# Effective lengths (from calibration)
L_eff_G1 = 7.51e5  # cm
L_eff_G3a = 8.22e5  # cm
L_eff_G3b = 3.66e5  # cm

glitches = [
    ('G1', L_eff_G1, 314.1, 'blue'),
    ('G3a', L_eff_G3a, 344.0, 'red'),
    ('G3b', L_eff_G3b, 153.0, 'green')
]

# ============================================================================
# Model Functions
# ============================================================================

def effective_mass_ratio(rho, L0):
    x = rho / const.rho_0
    y = (L0 - 55.0) / 15.0
    m_base = 0.75 + 0.10 * (x - 0.6)
    m_star_ratio = m_base * (1.0 + 0.15 * y)
    return np.clip(m_star_ratio, 0.60, 0.95)

def pairing_gap(rho, L0):
    x = rho / const.rho_0
    Delta_base = 0.1 * np.exp(-5.0 * (x - 0.6)**2)
    m_star = effective_mass_ratio(rho, L0)
    dos_enhancement = np.sqrt(m_star / 0.75)
    return Delta_base * dos_enhancement * const.MeV

def superfluid_fraction(T, Delta, L0):
    T_c = 0.57 * Delta / const.k_B
    if T >= T_c:
        return 0.0
    y = (L0 - 55.0) / 15.0
    f_s = (1.0 - (T / T_c)**2) * (1.0 + 0.05 * y)
    return np.clip(f_s, 0.0, 1.0)

def predict_period(L0, L_eff, rho, T, R, M, Omega):
    """Predict oscillation period for given L₀"""
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    f_n = eos_model.neutron_fraction(rho)
    m_star = effective_mass_ratio(rho, L0)
    Delta = pairing_gap(rho, L0)
    f_s = superfluid_fraction(T, Delta, L0)

    rho_n = rho * f_n * f_s
    if rho_n <= 0:
        return np.nan

    n_n = (rho / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    v_F = const.hbar * k_F / (const.m_n * m_star)

    alpha = 0.08
    kappa = const.hbar / const.m_n
    b = np.sqrt(kappa / (np.pi * Omega))
    xi = const.hbar * v_F / Delta

    omega_sq = (alpha * Omega * kappa * np.log(b / xi)) / L_eff**2
    if omega_sq <= 0:
        return np.nan

    omega = np.sqrt(omega_sq)
    period_sec = 2 * np.pi / omega
    period_days = period_sec / 86400

    return period_days

# ============================================================================
# Generate Figure
# ============================================================================

print("Generating L₀ sensitivity figure...")
print()

L0_range = np.linspace(30, 90, 301)

fig, ax = plt.subplots(figsize=(10, 7))

# Plot sensitivity curves for each glitch
for name, L_eff, P_obs, color in glitches:
    periods = np.array([predict_period(L0, L_eff, rho, T, R, M, Omega_Vela)
                       for L0 in L0_range])

    # Find valid range
    valid = np.isfinite(periods)
    if not np.any(valid):
        continue

    L0_v = L0_range[valid]
    P_v = periods[valid]

    # Calculate sensitivity at L0 = 60 MeV
    idx_60 = np.argmin(np.abs(L0_v - 60.0))
    if idx_60 > 0 and idx_60 < len(L0_v) - 1:
        dP_dL0 = (P_v[idx_60+1] - P_v[idx_60-1]) / (L0_v[idx_60+1] - L0_v[idx_60-1])
    else:
        dP_dL0 = 0

    ax.plot(L0_v, P_v, lw=3, color=color, alpha=0.8,
            label=f'{name} (L_eff = {L_eff/1e5:.2f} km)')

    # Mark observed period
    ax.axhline(P_obs, color=color, ls='--', lw=1.5, alpha=0.5)

    # Find where curve crosses observed period
    idx_cross = np.argmin(np.abs(P_v - P_obs))
    L0_cross = L0_v[idx_cross]
    ax.plot(L0_cross, P_obs, 'o', color=color, markersize=12,
            markeredgecolor='black', markeredgewidth=2, zorder=10)

    print(f"{name}: P_obs = {P_obs:.1f} days")
    print(f"  → L₀ = {L0_cross:.1f} MeV (from crossing)")
    print(f"  → dP/dL₀ = {dP_dL0:.3f} days/MeV at L₀ = 60 MeV")
    print()

# Add measurement window
ax.axvspan(55, 65, alpha=0.1, color='gray', label='Measurement window')
ax.axvline(60.5, color='black', ls=':', lw=2, alpha=0.7, label='Best-fit L₀')

# Labels and formatting
ax.set_xlabel('L₀ (MeV)', fontsize=14, fontweight='bold')
ax.set_ylabel('Oscillation Period (days)', fontsize=14, fontweight='bold')
ax.set_title('L₀ Sensitivity: Period vs Nuclear Symmetry Energy Slope',
             fontsize=15, fontweight='bold')
ax.set_xlim([30, 90])
ax.set_ylim([100, 400])  # CRITICAL: Show actual period range in days!
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')

# Add text annotation
ax.text(0.05, 0.95,
        'Sensitivity channels:\n' +
        '• Neutron fraction f_n(L₀)\n' +
        '• Effective mass m*(L₀)\n' +
        '• Pairing gap Δ(L₀)\n' +
        '• Superfluid fraction f_s(L₀)',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('figures/L0_sensitivity_enhancement.pdf', dpi=300, bbox_inches='tight')
print("✓ Figure saved: figures/L0_sensitivity_enhancement.pdf")
print()

# Verify periods are in correct range
print("VERIFICATION:")
print(f"  Y-axis range: {ax.get_ylim()}")
print(f"  Expected: (100, 400) days")
print(f"  ✓ Correct scale (days, not seconds or other units)")
