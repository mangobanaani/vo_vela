"""
Independent Calibration of PSR J1522-5735
==========================================

PROPER TEST: Calibrate α independently for J1522-5735, then compare L₀ with Vela.

If L₀ is universal (nuclear physics parameter), both pulsars should yield
the same L₀ even with different α values.

Method:
- Use J1522's two observed oscillations: 248d, 135d
- For each candidate L₀, find α and (L_eff,1, L_eff,2) that best fit both periods
- Choose L₀ that gives most physically reasonable α and L_eff
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const
from src import eos

# ============================================================================
# PSR J1522-5735 Parameters
# ============================================================================

P_spin = 204e-3  # s
Omega_J1522 = 2 * np.pi / P_spin
M = 1.4 * const.M_sun
R = 12e5  # cm
rho = 0.6 * const.rho_0
T = 5e7  # K (cooler than Vela, older pulsar)

# Two observed oscillations
P_obs_1 = 248.0  # days
P_obs_2 = 135.0  # days
sigma_P = 5.0  # uncertainty

print("=" * 80)
print("INDEPENDENT CALIBRATION: PSR J1522-5735")
print("=" * 80)
print()
print(f"Pulsar: PSR J1522-5735")
print(f"  Spin period: {P_spin*1e3:.1f} ms")
print(f"  Ω = {Omega_J1522:.2f} rad/s (vs Vela: {2*np.pi/0.089:.2f} rad/s)")
print(f"  Observed oscillations: {P_obs_1:.0f} d, {P_obs_2:.0f} d")
print()

# ============================================================================
# Model (same as Vela analysis)
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

def predict_period(alpha, L_eff, L0, Omega):
    """Predict period given α, L_eff, L₀"""
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    f_n = eos_model.neutron_fraction(rho)
    m_star_ratio = effective_mass_ratio(rho, L0)
    Delta = pairing_gap(rho, L0)
    T_c = 0.57 * Delta / const.k_B

    if T >= T_c:
        f_s = 0.0
    else:
        f_s = 1.0 - (T / T_c)**2
        y = (L0 - 55.0) / 15.0
        f_s = np.clip(f_s * (1.0 + 0.05 * y), 0.0, 1.0)

    rho_n = rho * f_n * f_s
    if rho_n <= 0:
        return np.inf

    n_n = (rho / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    m_star = const.m_n * m_star_ratio
    v_F = const.hbar * k_F / m_star

    # Oscillation frequency
    kappa = const.hbar / const.m_n
    b = np.sqrt(kappa / (np.pi * Omega))
    xi = const.hbar * v_F / Delta

    if b <= xi or b <= 0 or xi <= 0:
        return np.inf

    omega_sq = (alpha * Omega * kappa * np.log(b / xi)) / L_eff**2

    if omega_sq <= 0:
        return np.inf

    omega = np.sqrt(omega_sq)
    return 2 * np.pi / omega / 86400  # days

# ============================================================================
# Calibration: Find best (α, L_eff,1, L_eff,2) for each L₀
# ============================================================================

def chi_squared(params, L0):
    """χ² for fitting both oscillations"""
    alpha, L_eff_1, L_eff_2 = params

    # Bounds check
    if alpha <= 0 or alpha > 1.0:
        return 1e10
    if L_eff_1 <= 0 or L_eff_1 > 15e5:
        return 1e10
    if L_eff_2 <= 0 or L_eff_2 > 15e5:
        return 1e10

    P_pred_1 = predict_period(alpha, L_eff_1, L0, Omega_J1522)
    P_pred_2 = predict_period(alpha, L_eff_2, L0, Omega_J1522)

    if not np.isfinite(P_pred_1) or not np.isfinite(P_pred_2):
        return 1e10

    chi2 = ((P_pred_1 - P_obs_1) / sigma_P)**2 + ((P_pred_2 - P_obs_2) / sigma_P)**2

    return chi2

# Scan L₀
L0_scan = np.linspace(40, 80, 41)
results = []

print("Calibrating for each L₀ candidate...")
print()

for L0 in L0_scan:
    # Initial guess: α = 0.1, L_eff similar to Vela
    x0 = [0.1, 6e5, 4e5]

    # Minimize χ²
    res = minimize(chi_squared, x0, args=(L0,),
                   method='Nelder-Mead',
                   options={'maxiter': 1000, 'xatol': 1e-6})

    if res.success:
        alpha_fit, L_eff_1, L_eff_2 = res.x
        chi2_min = res.fun

        # Verify predictions
        P1 = predict_period(alpha_fit, L_eff_1, L0, Omega_J1522)
        P2 = predict_period(alpha_fit, L_eff_2, L0, Omega_J1522)

        results.append({
            'L0': L0,
            'alpha': alpha_fit,
            'L_eff_1': L_eff_1,
            'L_eff_2': L_eff_2,
            'chi2': chi2_min,
            'P1_pred': P1,
            'P2_pred': P2
        })

if not results:
    print("ERROR: No successful fits!")
    sys.exit(1)

# Find best fit (minimum χ²)
results = sorted(results, key=lambda x: x['chi2'])
best = results[0]

print("=" * 80)
print("CALIBRATION RESULTS")
print("=" * 80)
print()
print(f"Best fit:")
print(f"  L₀ = {best['L0']:.1f} MeV")
print(f"  α = {best['alpha']:.4f} (cf. Vela: α = 0.08)")
print(f"  L_eff,1 = {best['L_eff_1']/1e5:.2f} km ({best['L_eff_1']/R*100:.1f}% of R)")
print(f"  L_eff,2 = {best['L_eff_2']/1e5:.2f} km ({best['L_eff_2']/R*100:.1f}% of R)")
print(f"  χ² = {best['chi2']:.3f}")
print()
print(f"Period fits:")
print(f"  Oscillation 1: P_obs = {P_obs_1:.1f} d, P_pred = {best['P1_pred']:.1f} d")
print(f"  Oscillation 2: P_obs = {P_obs_2:.1f} d, P_pred = {best['P2_pred']:.1f} d")
print()

# Show top 5 fits
print("Top 5 L₀ candidates:")
print(f"{'L₀ (MeV)':>10} {'α':>8} {'χ²':>8} {'L_eff,1 (km)':>14} {'L_eff,2 (km)':>14}")
print("-" * 70)
for r in results[:5]:
    print(f"{r['L0']:10.1f} {r['alpha']:8.4f} {r['chi2']:8.3f} "
          f"{r['L_eff_1']/1e5:14.2f} {r['L_eff_2']/1e5:14.2f}")
print()

# ============================================================================
# Comparison with Vela
# ============================================================================

print("=" * 80)
print("COMPARISON: VELA vs J1522-5735")
print("=" * 80)
print()
print(f"{'Pulsar':<15} {'Ω (rad/s)':<12} {'α':<10} {'L₀ (MeV)':<12}")
print("-" * 50)
print(f"{'Vela':<15} {2*np.pi/0.089:<12.2f} {0.08:<10.4f} {60.5:<12.1f}")
print(f"{'J1522-5735':<15} {Omega_J1522:<12.2f} {best['alpha']:<10.4f} {best['L0']:<12.1f}")
print()

L0_diff = abs(best['L0'] - 60.5)
print(f"L₀ difference: {L0_diff:.1f} MeV")
print()

if L0_diff < 5:
    print("✓ CONSISTENT: L₀ values agree within ~5 MeV")
    print("  → Nuclear physics parameter appears universal")
    print("  → Different α values reflect different stellar properties")
    print("  → Method validated across different pulsars!")
else:
    print("✗ INCONSISTENT: L₀ values differ by > 5 MeV")
    print("  → Either:")
    print("    (1) Model assumptions wrong (density, temperature, etc.)")
    print("    (2) Glitch physics fundamentally different")
    print("    (3) Method has uncontrolled systematics")

print()
print("=" * 80)

# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: χ² vs L₀
ax = axes[0]
L0_vals = [r['L0'] for r in results]
chi2_vals = [r['chi2'] for r in results]

ax.plot(L0_vals, chi2_vals, 'b-', lw=2)
ax.axvline(best['L0'], color='red', ls='--', lw=2, label=f"Best: {best['L0']:.1f} MeV")
ax.axvline(60.5, color='green', ls=':', lw=2, label='Vela: 60.5 MeV')
ax.set_xlabel('L₀ (MeV)', fontsize=12)
ax.set_ylabel('χ²', fontsize=12)
ax.set_title('Goodness of Fit vs L₀', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: α vs L₀
ax = axes[1]
alpha_vals = [r['alpha'] for r in results]

ax.plot(L0_vals, alpha_vals, 'b-', lw=2)
ax.axhline(0.08, color='green', ls=':', lw=2, label='Vela: α = 0.08')
ax.axvline(best['L0'], color='red', ls='--', lw=1.5, alpha=0.7)
ax.set_xlabel('L₀ (MeV)', fontsize=12)
ax.set_ylabel('Calibrated α', fontsize=12)
ax.set_title('Geometric Factor vs L₀', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Comparison bar chart
ax = axes[2]
pulsars = ['Vela', 'J1522-5735']
L0s = [60.5, best['L0']]
alphas_plot = [0.08, best['alpha']]

x = np.arange(len(pulsars))
width = 0.35

bars1 = ax.bar(x - width/2, L0s, width, label='L₀ (MeV)', color='blue', alpha=0.7)
ax_twin = ax.twinx()
bars2 = ax_twin.bar(x + width/2, alphas_plot, width, label='α', color='red', alpha=0.7)

ax.set_ylabel('L₀ (MeV)', fontsize=12, color='blue')
ax_twin.set_ylabel('α', fontsize=12, color='red')
ax.set_xticks(x)
ax.set_xticklabels(pulsars)
ax.set_title('Vela vs J1522-5735', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars1, L0s):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,
            f'{val:.1f}', ha='center', va='bottom', fontsize=10, color='blue')

for bar, val in zip(bars2, alphas_plot):
    height = bar.get_height()
    ax_twin.text(bar.get_x() + bar.get_width()/2, height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('figures/j1522_independent_calibration.pdf', dpi=300, bbox_inches='tight')
print("Figure saved: figures/j1522_independent_calibration.pdf")
print()
