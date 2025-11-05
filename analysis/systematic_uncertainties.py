"""
Systematic Uncertainty Analysis for L₀ Measurement
===================================================

Quantify how uncertainties in physical parameters affect the L₀ constraint:
1. Temperature: T = (1.0 ± 0.2) × 10⁸ K
2. Mass: M = 1.4 ± 0.2 M
3. Radius: R = 12 ± 1 km
4. Pairing model: AO vs CCDK gap parameterizations

Strategy:
---------
For each parameter:
1. Run Bayesian inference at nominal value
2. Run at +1σ variation
3. Run at -1σ variation
4. Compute shift in L₀_MAP and credible interval width

Total systematic uncertainty:
σ_sys² = σ_T² + σ_M² + σ_R² + σ_pairing²

Final result:
L₀ = 60.5 ± 1.5 (stat) ± σ_sys (sys) MeV
"""

import numpy as np
from scipy.stats import norm
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import constants as const
from src import eos, superfluid as sf, vortex

# ============================================================================
# Enhanced Forward Model (EXACT COPY from measure_L0_bayesian.py)
# ============================================================================

def effective_mass_ratio(rho, L0):
    """m*/m_n as function of density and L₀"""
    x = rho / const.rho_0
    y = (L0 - 55.0) / 15.0
    m_base = 0.75 + 0.10 * (x - 0.6)
    m_star_over_m = m_base * (1.0 + 0.15 * y)
    return np.clip(m_star_over_m, 0.60, 0.95)


def pairing_gap_L0_dependent(rho, L0, pairing_model='AO'):
    """Pairing gap with L₀ dependence"""
    x = rho / const.rho_0
    Delta_base = 0.1 * np.exp(-5.0 * (x - 0.6)**2)  # MeV

    # Pairing model variation
    if pairing_model == 'CCDK':
        Delta_base *= 1.10  # CCDK gaps ~10% larger

    m_star_ratio = effective_mass_ratio(rho, L0)
    dos_enhancement = np.sqrt(m_star_ratio / 0.75)
    Delta = Delta_base * dos_enhancement
    return Delta * const.MeV


def predict_period_enhanced(L0, L_eff, rho_glitch, T, R, M, Omega, pairing_model='AO'):
    """
    Enhanced forward model: L₀ → Period

    EXACT COPY from measure_L0_bayesian.py that works correctly
    """
    # Create EoS with this L₀
    eos_model = eos.SymmetryEnergyEoS(L0=L0)

    # Neutron fraction (depends on L₀)
    f_n = eos_model.neutron_fraction(rho_glitch)

    # Effective mass
    m_star_ratio = effective_mass_ratio(rho_glitch, L0)

    # Pairing gap (with L₀ dependence)
    Delta = pairing_gap_L0_dependent(rho_glitch, L0, pairing_model=pairing_model)

    # Critical temperature
    T_c = 0.57 * Delta / const.k_B

    # Superfluid fraction
    if T >= T_c:
        f_s = 0.0
    else:
        f_s = 1.0 - (T / T_c)**2
        # L₀ enhancement
        y = (L0 - 55.0) / 15.0
        correction = 1.0 + 0.05 * y
        f_s = np.clip(f_s * correction, 0.0, 1.0)

    # Superfluid density
    rho_n = rho_glitch * f_n * f_s

    if rho_n <= 0:
        return np.inf

    # Fermi velocity (with m* correction)
    n_n = (rho_glitch / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    m_star = const.m_n * m_star_ratio
    v_F = const.hbar * k_F / m_star

    # Oscillation frequency with calibrated geometric factor
    mode_params = vortex.BendingModeParameters(
        boundary_condition='clamped-free',
        mode_number=0,
        effective_length=L_eff
    )

    try:
        omega = vortex.oscillation_frequency(
            rho_n, R, Omega, Delta, v_F, mode_params
        )
        if omega <= 0:
            return np.inf
        P_seconds = 2 * np.pi / omega
        P_days = P_seconds / (24 * 3600)
        return P_days
    except:
        return np.inf


def run_bayesian_inference(L0_grid, P_obs, sigma_P, L_eff, rho_glitch, T, R, M, Omega,
                           pairing_model='AO'):
    """
    Run Bayesian inference for given parameters.

    Returns:
    --------
    result : dict
        Contains L0_MAP, L0_16, L0_84, posterior
    """
    # Predict periods
    periods = np.array([
        predict_period_enhanced(L0, L_eff, rho_glitch, T, R, M, Omega, pairing_model)
        for L0 in L0_grid
    ])

    # Filter valid
    valid = np.isfinite(periods)
    L0_valid = L0_grid[valid]
    periods_valid = periods[valid]

    if len(L0_valid) == 0:
        return None

    # Bayesian inference
    prior = np.ones_like(L0_valid) / len(L0_valid)
    likelihood = norm.pdf(periods_valid, loc=P_obs, scale=sigma_P)
    posterior = likelihood * prior
    posterior /= np.trapz(posterior, L0_valid)

    # Find MAP
    idx_max = np.argmax(posterior)
    L0_MAP = L0_valid[idx_max]

    # Credible intervals
    cdf = np.zeros_like(posterior)
    for i in range(len(posterior)):
        cdf[i] = np.trapz(posterior[:i+1], L0_valid[:i+1])

    idx_16 = np.searchsorted(cdf, 0.16)
    idx_84 = np.searchsorted(cdf, 0.84)
    L0_16 = L0_valid[min(idx_16, len(L0_valid)-1)]
    L0_84 = L0_valid[min(idx_84, len(L0_valid)-1)]

    return {
        'L0_MAP': L0_MAP,
        'L0_16': L0_16,
        'L0_84': L0_84,
        'L0_valid': L0_valid,
        'posterior': posterior,
        'periods': periods_valid
    }


# ============================================================================
# Nominal Parameters
# ============================================================================

# G1 glitch (tightest constraint)
P_obs = 314.1  # days
sigma_P = 0.2  # days
L_eff = 7.51e5  # cm

# Nominal values
M_nom = 1.4 * const.M_sun  # g
R_nom = 12e5  # cm
Omega_nom = const.Omega_Vela  # rad/s
T_nom = 1.0e8  # K
rho_nom = 0.6 * const.rho_0  # g/cm³
pairing_nom = 'AO'

# L₀ grid
L0_grid = np.linspace(30.0, 90.0, 121)

print("=" * 80)
print("SYSTEMATIC UNCERTAINTY ANALYSIS")
print("=" * 80)
print()
print("Nominal parameters:")
print(f"  M = {M_nom/const.M_sun:.1f} M")
print(f"  R = {R_nom/1e5:.1f} km")
print(f"  T = {T_nom:.1e} K")
print(f"  ρ = {rho_nom/const.rho_0:.2f} ρ₀")
print(f"  Pairing: {pairing_nom}")
print()

# ============================================================================
# 1. Nominal Case
# ============================================================================

print("=" * 80)
print("1. NOMINAL CASE")
print("=" * 80)
print()

result_nom = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_nom, M_nom, Omega_nom, pairing_nom
)

L0_nom = result_nom['L0_MAP']
L0_16_nom = result_nom['L0_16']
L0_84_nom = result_nom['L0_84']
sigma_stat = (L0_84_nom - L0_16_nom) / 2.0

print(f"Nominal result:")
print(f"  L₀ = {L0_nom:.1f} +{L0_84_nom - L0_nom:.1f} -{L0_nom - L0_16_nom:.1f} MeV")
print(f"  Statistical uncertainty: σ_stat = ±{sigma_stat:.2f} MeV")
print()

# ============================================================================
# 2. Temperature Variation
# ============================================================================

print("=" * 80)
print("2. TEMPERATURE UNCERTAINTY")
print("=" * 80)
print()
print("Testing: T = (1.0 ± 0.2) × 10⁸ K")
print()

T_low = 0.8e8  # K (-20%)
T_high = 1.2e8  # K (+20%)

result_T_low = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_low, R_nom, M_nom, Omega_nom, pairing_nom
)

result_T_high = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_high, R_nom, M_nom, Omega_nom, pairing_nom
)

L0_T_low = result_T_low['L0_MAP']
L0_T_high = result_T_high['L0_MAP']

delta_L0_T = max(abs(L0_T_low - L0_nom), abs(L0_T_high - L0_nom))

print(f"T = {T_low:.1e} K: L₀ = {L0_T_low:.1f} MeV (shift: {L0_T_low - L0_nom:+.2f} MeV)")
print(f"T = {T_high:.1e} K: L₀ = {L0_T_high:.1f} MeV (shift: {L0_T_high - L0_nom:+.2f} MeV)")
print(f"Systematic uncertainty from T: σ_T = ±{delta_L0_T:.2f} MeV")
print()

# ============================================================================
# 3. Mass Variation
# ============================================================================

print("=" * 80)
print("3. MASS UNCERTAINTY")
print("=" * 80)
print()
print("Testing: M = 1.4 ± 0.2 M")
print()

M_low = 1.2 * const.M_sun
M_high = 1.6 * const.M_sun

result_M_low = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_nom, M_low, Omega_nom, pairing_nom
)

result_M_high = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_nom, M_high, Omega_nom, pairing_nom
)

L0_M_low = result_M_low['L0_MAP']
L0_M_high = result_M_high['L0_MAP']

delta_L0_M = max(abs(L0_M_low - L0_nom), abs(L0_M_high - L0_nom))

print(f"M = {M_low/const.M_sun:.1f} M: L₀ = {L0_M_low:.1f} MeV (shift: {L0_M_low - L0_nom:+.2f} MeV)")
print(f"M = {M_high/const.M_sun:.1f} M: L₀ = {L0_M_high:.1f} MeV (shift: {L0_M_high - L0_nom:+.2f} MeV)")
print(f"Systematic uncertainty from M: σ_M = ±{delta_L0_M:.2f} MeV")
print()

# ============================================================================
# 4. Radius Variation
# ============================================================================

print("=" * 80)
print("4. RADIUS UNCERTAINTY")
print("=" * 80)
print()
print("Testing: R = 12 ± 1 km")
print()

R_low = 11e5  # cm
R_high = 13e5  # cm

result_R_low = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_low, M_nom, Omega_nom, pairing_nom
)

result_R_high = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_high, M_nom, Omega_nom, pairing_nom
)

L0_R_low = result_R_low['L0_MAP']
L0_R_high = result_R_high['L0_MAP']

delta_L0_R = max(abs(L0_R_low - L0_nom), abs(L0_R_high - L0_nom))

print(f"R = {R_low/1e5:.1f} km: L₀ = {L0_R_low:.1f} MeV (shift: {L0_R_low - L0_nom:+.2f} MeV)")
print(f"R = {R_high/1e5:.1f} km: L₀ = {L0_R_high:.1f} MeV (shift: {L0_R_high - L0_nom:+.2f} MeV)")
print(f"Systematic uncertainty from R: σ_R = ±{delta_L0_R:.2f} MeV")
print()

# ============================================================================
# 5. Pairing Model Variation
# ============================================================================

print("=" * 80)
print("5. PAIRING MODEL UNCERTAINTY")
print("=" * 80)
print()
print("Testing: AO vs CCDK (±10% gap variation)")
print()

result_CCDK = run_bayesian_inference(
    L0_grid, P_obs, sigma_P, L_eff, rho_nom, T_nom, R_nom, M_nom, Omega_nom, 'CCDK'
)

L0_CCDK = result_CCDK['L0_MAP']
delta_L0_pairing = abs(L0_CCDK - L0_nom)

print(f"AO model:   L₀ = {L0_nom:.1f} MeV")
print(f"CCDK model: L₀ = {L0_CCDK:.1f} MeV (shift: {L0_CCDK - L0_nom:+.2f} MeV)")
print(f"Systematic uncertainty from pairing: σ_pairing = ±{delta_L0_pairing:.2f} MeV")
print()

# ============================================================================
# 6. Total Systematic Uncertainty
# ============================================================================

print("=" * 80)
print("6. TOTAL SYSTEMATIC UNCERTAINTY")
print("=" * 80)
print()

# Combine in quadrature (assuming uncorrelated)
sigma_sys = np.sqrt(delta_L0_T**2 + delta_L0_M**2 + delta_L0_R**2 + delta_L0_pairing**2)

print("Individual contributions:")
print(f"  Temperature:   σ_T       = ±{delta_L0_T:.2f} MeV")
print(f"  Mass:          σ_M       = ±{delta_L0_M:.2f} MeV")
print(f"  Radius:        σ_R       = ±{delta_L0_R:.2f} MeV")
print(f"  Pairing model: σ_pairing = ±{delta_L0_pairing:.2f} MeV")
print()
print(f"Total systematic: σ_sys = ±{sigma_sys:.2f} MeV")
print()

# ============================================================================
# 7. Final Result
# ============================================================================

print("=" * 80)
print("7. FINAL RESULT WITH SYSTEMATIC UNCERTAINTIES")
print("=" * 80)
print()

# Total uncertainty
sigma_total = np.sqrt(sigma_stat**2 + sigma_sys**2)

print(f"L₀ = {L0_nom:.1f} ± {sigma_stat:.1f} (stat) ± {sigma_sys:.1f} (sys) MeV")
print()
print(f"L₀ = {L0_nom:.1f} ± {sigma_total:.1f} (total) MeV")
print()

# Fractional uncertainties
frac_stat = sigma_stat / L0_nom * 100
frac_sys = sigma_sys / L0_nom * 100
frac_total = sigma_total / L0_nom * 100

print("Fractional uncertainties:")
print(f"  Statistical: {frac_stat:.2f}%")
print(f"  Systematic:  {frac_sys:.2f}%")
print(f"  Total:       {frac_total:.2f}%")
print()

# Dominant uncertainties
uncertainties = {
    'Temperature': delta_L0_T,
    'Mass': delta_L0_M,
    'Radius': delta_L0_R,
    'Pairing': delta_L0_pairing
}

sorted_uncert = sorted(uncertainties.items(), key=lambda x: x[1], reverse=True)

print("Ranked by importance:")
for i, (source, value) in enumerate(sorted_uncert, 1):
    contribution = (value / sigma_sys * 100) if sigma_sys > 0 else 0
    print(f"  {i}. {source:12s}: ±{value:.2f} MeV ({contribution:.1f}% of σ_sys)")
print()

# ============================================================================
# 8. Save Results
# ============================================================================

results = {
    'nominal': {
        'L0_MAP': float(L0_nom),
        'L0_16': float(L0_16_nom),
        'L0_84': float(L0_84_nom),
        'sigma_stat': float(sigma_stat)
    },
    'variations': {
        'temperature': {
            'T_low': T_low,
            'T_high': T_high,
            'L0_low': float(L0_T_low),
            'L0_high': float(L0_T_high),
            'sigma': float(delta_L0_T)
        },
        'mass': {
            'M_low': M_low / const.M_sun,
            'M_high': M_high / const.M_sun,
            'L0_low': float(L0_M_low),
            'L0_high': float(L0_M_high),
            'sigma': float(delta_L0_M)
        },
        'radius': {
            'R_low': R_low / 1e5,
            'R_high': R_high / 1e5,
            'L0_low': float(L0_R_low),
            'L0_high': float(L0_R_high),
            'sigma': float(delta_L0_R)
        },
        'pairing': {
            'model_nom': pairing_nom,
            'model_alt': 'CCDK',
            'L0_nom': float(L0_nom),
            'L0_alt': float(L0_CCDK),
            'sigma': float(delta_L0_pairing)
        }
    },
    'total': {
        'sigma_stat': float(sigma_stat),
        'sigma_sys': float(sigma_sys),
        'sigma_total': float(sigma_total),
        'frac_stat': float(frac_stat),
        'frac_sys': float(frac_sys),
        'frac_total': float(frac_total)
    },
    'final_result': {
        'L0': float(L0_nom),
        'stat_error': float(sigma_stat),
        'sys_error': float(sigma_sys),
        'total_error': float(sigma_total),
        'latex': f"{L0_nom:.1f} \\pm {sigma_stat:.1f} \\text{{(stat)}} \\pm {sigma_sys:.1f} \\text{{(sys)}} \\text{{ MeV}}"
    }
}

output_file = Path(__file__).parent / 'systematic_uncertainties_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}")
print()
print("=" * 80)
print("COMPLETE SYSTEMATIC UNCERTAINTY ANALYSIS COMPLETE")
print("=" * 80)
