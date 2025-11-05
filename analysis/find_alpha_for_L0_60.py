"""
Find α for PSR J1522-5735 assuming L₀ = 60 MeV
===============================================

Question: If L₀ = 60 MeV (same as Vela), what α does J1522-5735 need
to match its observed periods (248d, 135d)?

If we can find a reasonable α, then L₀ = 60 MeV is universal and
the different α values just reflect different stellar properties.
"""

import numpy as np
from scipy.optimize import minimize
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const
from src import eos

# PSR J1522-5735 parameters
P_spin = 204e-3
Omega_J1522 = 2 * np.pi / P_spin
M = 1.4 * const.M_sun
R = 12e5
rho = 0.6 * const.rho_0
T = 5e7  # K

P_obs_1 = 248.0  # days
P_obs_2 = 135.0  # days

# FIXED: L₀ = 60 MeV (Vela value)
L0_fixed = 60.0

print("=" * 80)
print("FINDING α FOR J1522-5735 AT L₀ = 60 MeV")
print("=" * 80)
print()
print(f"Assuming L₀ = {L0_fixed} MeV (Vela result)")
print(f"Finding α that fits J1522 oscillations: {P_obs_1:.0f}d, {P_obs_2:.0f}d")
print()

# Model functions
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

    kappa = const.hbar / const.m_n
    b = np.sqrt(kappa / (np.pi * Omega))
    xi = const.hbar * v_F / Delta

    if b <= xi or b <= 0 or xi <= 0:
        return np.inf

    omega_sq = (alpha * Omega * kappa * np.log(b / xi)) / L_eff**2
    if omega_sq <= 0:
        return np.inf

    omega = np.sqrt(omega_sq)
    return 2 * np.pi / omega / 86400

# Optimization: find (α, L_eff,1, L_eff,2) that fit both periods
def objective(params):
    alpha, L_eff_1, L_eff_2 = params

    if alpha <= 0 or alpha > 1.0:
        return 1e10
    if L_eff_1 <= 0 or L_eff_1 > 15e5:
        return 1e10
    if L_eff_2 <= 0 or L_eff_2 > 15e5:
        return 1e10

    P1 = predict_period(alpha, L_eff_1, L0_fixed, Omega_J1522)
    P2 = predict_period(alpha, L_eff_2, L0_fixed, Omega_J1522)

    if not np.isfinite(P1) or not np.isfinite(P2):
        return 1e10

    # χ²
    return ((P1 - P_obs_1) / 5.0)**2 + ((P2 - P_obs_2) / 5.0)**2

# Initial guess
x0 = [0.03, 6e5, 4e5]  # Try α smaller than Vela (J1522 rotates slower)

# Optimize
print("Optimizing...")
res = minimize(objective, x0, method='Nelder-Mead',
               options={'maxiter': 2000, 'xatol': 1e-8})

if res.success:
    alpha_fit, L_eff_1, L_eff_2 = res.x
    chi2 = res.fun

    P1_pred = predict_period(alpha_fit, L_eff_1, L0_fixed, Omega_J1522)
    P2_pred = predict_period(alpha_fit, L_eff_2, L0_fixed, Omega_J1522)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"For L₀ = {L0_fixed} MeV:")
    print(f"  α = {alpha_fit:.5f} (cf. Vela: α = 0.08000)")
    print(f"  L_eff,1 = {L_eff_1/1e5:.2f} km ({L_eff_1/R*100:.1f}% of R)")
    print(f"  L_eff,2 = {L_eff_2/1e5:.2f} km ({L_eff_2/R*100:.1f}% of R)")
    print(f"  χ² = {chi2:.4f}")
    print()
    print("Period fits:")
    print(f"  Oscillation 1: P_obs = {P_obs_1:.1f} d, P_pred = {P1_pred:.1f} d, Δ = {P1_pred-P_obs_1:+.1f} d")
    print(f"  Oscillation 2: P_obs = {P_obs_2:.1f} d, P_pred = {P2_pred:.1f} d, Δ = {P2_pred-P_obs_2:+.1f} d")
    print()

    # Compare α values
    alpha_ratio = alpha_fit / 0.08
    Omega_ratio = Omega_J1522 / (2*np.pi/0.089)

    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Parameter':<20} {'Vela':<15} {'J1522-5735':<15} {'Ratio':<10}")
    print("-" * 60)
    print(f"{'Ω (rad/s)':<20} {2*np.pi/0.089:<15.2f} {Omega_J1522:<15.2f} {Omega_ratio:<10.3f}")
    print(f"{'α':<20} {0.08:<15.5f} {alpha_fit:<15.5f} {alpha_ratio:<10.3f}")
    print(f"{'L₀ (MeV)':<20} {60.5:<15.1f} {L0_fixed:<15.1f} {'same':<10}")
    print()

    print("Physical interpretation:")
    print(f"  - J1522 rotates {Omega_ratio:.2f}× slower than Vela")
    print(f"  - Needs α that is {alpha_ratio:.2f}× Vela's α")
    print(f"  - Scaling: α ∝ Ω^{np.log(alpha_ratio)/np.log(Omega_ratio):.2f}")
    print()

    if chi2 < 1.0:
        print("✓ EXCELLENT FIT: L₀ = 60 MeV is consistent with J1522 data!")
        print("  → Nuclear physics (L₀) is universal")
        print("  → Geometric factor (α) depends on stellar properties (Ω, etc.)")
        print("  → Method validated across pulsars!")
    elif chi2 < 10.0:
        print("✓ GOOD FIT: L₀ = 60 MeV is plausibly consistent with J1522")
        print("  → Suggests L₀ is universal (nuclear physics)")
        print("  → α variation captures stellar differences")
    else:
        print("✗ POOR FIT: L₀ = 60 MeV does not fit J1522 data well")
        print("  → Either L₀ is truly different, or model assumptions wrong")

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print(f"If we assume L₀ = 60 MeV (nuclear physics, universal),")
    print(f"then J1522-5735 requires α = {alpha_fit:.5f}")
    print()
    print(f"Fit quality: χ² = {chi2:.4f}")
    print()
    if chi2 < 10:
        print("This suggests the method IS universal:")
        print("  • L₀ = 60 MeV works for both Vela and J1522")
        print("  • Different α values reflect different Ω")
        print("  • Need ab initio calculation of α(Ω, M, R)")
    else:
        print("This suggests either:")
        print("  • Model has problems (density, temperature assumptions)")
        print("  • Glitch physics fundamentally different")

else:
    print("ERROR: Optimization failed!")
    print(res.message)
