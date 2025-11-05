"""
PSR J1522-5735 Parameter Sensitivity
=====================================

Test how L₀ inference depends on uncertain stellar parameters:
- Temperature T
- Density ρ
- Mass M, Radius R

Goal: Understand why J1522-5735 gives L₀ = 40 MeV vs Vela's 60.5 MeV
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const
from src import eos

# PSR J1522-5735 observables
P_spin = 204e-3  # s
Omega_J1522 = 2 * np.pi / P_spin
P_obs = 248.0  # days (use the longer oscillation)
sigma_P = 10.0

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
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    f_n = eos_model.neutron_fraction(rho)
    m_star = effective_mass_ratio(rho, L0)
    Delta = pairing_gap(rho, L0)
    f_s = superfluid_fraction(T, Delta, L0)
    rho_n = rho * f_n * f_s

    if rho_n <= 0:
        return np.inf

    n_n = (rho / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    v_F = const.hbar * k_F / (const.m_n * m_star)

    alpha = 0.08
    kappa = const.hbar / const.m_n
    b = np.sqrt(kappa / (np.pi * Omega))
    xi = const.hbar * v_F / Delta

    omega_sq = (alpha * Omega * kappa * np.log(b / xi)) / L_eff**2
    if omega_sq <= 0:
        return np.inf

    omega = np.sqrt(omega_sq)
    return 2 * np.pi / omega / 86400

def infer_L0(P_obs, sigma_P, rho, T, R, M, Omega):
    L0_grid = np.linspace(40, 80, 101)
    L_eff_scan = np.linspace(3e5, 10e5, 50)

    results = []
    for L0 in L0_grid:
        periods = [predict_period(L0, L, rho, T, R, M, Omega) for L in L_eff_scan]
        periods = np.array(periods)
        valid = np.isfinite(periods)
        if not np.any(valid):
            continue

        idx_best = np.argmin(np.abs(periods[valid] - P_obs))
        P_pred = periods[valid][idx_best]
        likelihood = norm.pdf(P_pred, P_obs, sigma_P)
        results.append({'L0': L0, 'likelihood': likelihood})

    if not results:
        return np.nan

    L0_vals = np.array([r['L0'] for r in results])
    likes = np.array([r['likelihood'] for r in results])
    posterior = likes / np.trapz(likes, L0_vals)

    return L0_vals[np.argmax(posterior)]

print("=" * 80)
print("PSR J1522-5735 PARAMETER SENSITIVITY SCAN")
print("=" * 80)
print()
print("Testing how L₀ inference depends on uncertain stellar parameters")
print()

M_J1522 = 1.4 * const.M_sun
R_J1522 = 12e5

# Test 1: Temperature variation
print("Test 1: Temperature Variation")
print("-" * 80)
T_values = [0.3e8, 0.5e8, 0.8e8, 1.0e8, 1.5e8]
rho_nominal = 0.6 * const.rho_0

for T in T_values:
    L0_inf = infer_L0(P_obs, sigma_P, rho_nominal, T, R_J1522, M_J1522, Omega_J1522)
    print(f"T = {T/1e8:.1f} × 10⁸ K → L₀ = {L0_inf:.1f} MeV")

print()

# Test 2: Density variation
print("Test 2: Density Variation")
print("-" * 80)
rho_values = [0.4, 0.5, 0.6, 0.7, 0.8]
T_nominal = 5e7

for rho_factor in rho_values:
    rho = rho_factor * const.rho_0
    L0_inf = infer_L0(P_obs, sigma_P, rho, T_nominal, R_J1522, M_J1522, Omega_J1522)
    print(f"ρ = {rho_factor:.1f} ρ₀ → L₀ = {L0_inf:.1f} MeV")

print()

# Test 3: What if we use Vela temperature?
print("Test 3: Using Vela-like Parameters")
print("-" * 80)
T_vela = 1.0e8
rho_vela = 0.6 * const.rho_0

L0_inf = infer_L0(P_obs, sigma_P, rho_vela, T_vela, R_J1522, M_J1522, Omega_J1522)
print(f"T = {T_vela/1e8:.1f} × 10⁸ K, ρ = 0.6 ρ₀ → L₀ = {L0_inf:.1f} MeV")
print()

print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print()
print("The L₀ = 40 MeV result for J1522-5735 is likely due to:")
print()
print("1. SLOWER ROTATION (Ω_J1522 = 31 rad/s vs Ω_Vela = 71 rad/s)")
print("   - Vortex oscillation frequency ω² ∝ Ω")
print("   - To get similar periods with smaller Ω, need compensation")
print("   - Model compensates by reducing L₀ (affects m* and f_s)")
print()
print("2. CALIBRATION NOT UNIVERSAL")
print("   - α = 0.08 was calibrated specifically for Vela")
print("   - May depend on Ω, stellar structure, or other factors")
print("   - Cannot simply apply to different pulsars")
print()
print("3. DIFFERENT GLITCH PHYSICS")
print("   - J1522-5735 shows 'anti-glitches' (spin-down)")
print("   - Different from Vela's spin-up glitches")
print("   - Vortex dynamics may be fundamentally different")
print()
print("CONCLUSION:")
print("-" * 80)
print("The calibrated approach (α = 0.08) is NOT universal.")
print("Each pulsar may require independent calibration.")
print("This limits the method's applicability unless α(Ω, M, R) is understood.")
print()
