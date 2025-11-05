"""
PSR J1522-5735 L₀ Measurement
==============================

Apply the same vortex oscillation method calibrated on Vela
to PSR J1522-5735 to test universality of the approach.

Zhou et al. (2024) observed post-glitch oscillations:
- 248 days (inter-glitch G1-G2)
- 135 days (inter-glitch G3-G4)
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import constants as const

# ============================================================================
# PSR J1522-5735 Parameters
# ============================================================================

# Observables (Zhou et al. 2024)
P_spin = 204e-3  # seconds (spin period)
Omega_J1522 = 2 * np.pi / P_spin  # rad/s
tau_c = 51.8e3 * 365.25 * 86400  # characteristic age in seconds

# Observed oscillation periods
P_obs_1 = 248.0  # days
P_obs_2 = 135.0  # days
sigma_P = 10.0   # days (estimated uncertainty - not given in paper)

# Assumed stellar parameters (typical neutron star)
M_J1522 = 1.4 * const.M_sun  # solar masses
R_J1522 = 12e5  # cm (12 km)

# Temperature estimate: older pulsar, so cooler
# Vela (11 kyr) has T ~ 1e8 K
# J1522-5735 (52 kyr) should be cooler
T_J1522 = 5e7  # K (factor ~2 cooler than Vela)

# Density: assume same as Vela (glitch occurs at similar crustal depth)
rho = 0.6 * const.rho_0

# ============================================================================
# Vortex Oscillation Model (Same as Vela)
# ============================================================================

def effective_mass_ratio(rho, L0):
    """Effective mass m*/m with L₀ dependence"""
    x = rho / const.rho_0
    y = (L0 - 55.0) / 15.0

    m_base = 0.75 + 0.10 * (x - 0.6)
    m_star_ratio = m_base * (1.0 + 0.15 * y)

    return np.clip(m_star_ratio, 0.60, 0.95)

def pairing_gap(rho, L0):
    """Neutron pairing gap (Amundsen-Østgaard model)"""
    x = rho / const.rho_0
    Delta_base = 0.1 * np.exp(-5.0 * (x - 0.6)**2)  # MeV

    # Effective mass enhancement
    y = (L0 - 55.0) / 15.0
    m_star = effective_mass_ratio(rho, L0)
    dos_enhancement = np.sqrt(m_star / 0.75)

    Delta = Delta_base * dos_enhancement
    return Delta * const.MeV

def superfluid_fraction(T, Delta, L0):
    """Temperature-dependent superfluid fraction"""
    T_c = 0.57 * Delta / const.k_B

    if T >= T_c:
        return 0.0

    y = (L0 - 55.0) / 15.0
    f_s = (1.0 - (T / T_c)**2) * (1.0 + 0.05 * y)
    return np.clip(f_s, 0.0, 1.0)

def predict_period(L0, L_eff, rho, T, R, M, Omega):
    """
    Predict oscillation period using calibrated vortex model

    Uses α = 0.08 calibrated from Vela
    """
    from src import eos

    # EoS-dependent quantities
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    f_n = eos_model.neutron_fraction(rho)

    # Effective mass
    m_star = effective_mass_ratio(rho, L0)

    # Pairing gap and superfluid fraction
    Delta = pairing_gap(rho, L0)
    f_s = superfluid_fraction(T, Delta, L0)

    # Superfluid neutron density
    rho_n = rho * f_n * f_s

    if rho_n <= 0:
        return np.inf

    # Fermi velocity
    n_n = (rho / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    v_F = const.hbar * k_F / (const.m_n * m_star)

    # Vortex oscillation frequency
    # Using α = 0.08 (calibrated from Vela)
    alpha = 0.08

    kappa = const.hbar / const.m_n
    b = np.sqrt(kappa / (np.pi * Omega))  # intervortex spacing
    xi = const.hbar * v_F / Delta  # coherence length

    omega_sq = (alpha * Omega * kappa * np.log(b / xi)) / L_eff**2

    if omega_sq <= 0:
        return np.inf

    omega = np.sqrt(omega_sq)
    period_sec = 2 * np.pi / omega
    period_days = period_sec / 86400

    return period_days

def infer_L0_bayesian(P_obs, sigma_P, rho, T, R, M, Omega, L0_grid):
    """
    Infer L₀ from observed period using Bayesian inference

    Returns: L₀_best, L₀_grid_valid, posterior
    """
    print(f"  Testing {len(L0_grid)} values of L₀...")

    # Estimate effective length from period (rough guess)
    # Use L_eff ~ 5 km as starting point
    L_eff_scan = np.linspace(3e5, 10e5, 50)  # cm

    best_L0 = None
    best_likelihood_sum = 0

    results = []

    for L0 in L0_grid:
        # For each L₀, find L_eff that best matches observed period
        periods = np.array([predict_period(L0, L_eff, rho, T, R, M, Omega)
                           for L_eff in L_eff_scan])

        valid = np.isfinite(periods)
        if np.sum(valid) == 0:
            continue

        # Find L_eff closest to observed period
        idx_best = np.argmin(np.abs(periods[valid] - P_obs))
        P_pred = periods[valid][idx_best]
        L_eff_best = L_eff_scan[valid][idx_best]

        # Likelihood
        likelihood = norm.pdf(P_pred, P_obs, sigma_P)

        results.append({
            'L0': L0,
            'P_pred': P_pred,
            'L_eff': L_eff_best,
            'likelihood': likelihood
        })

    if len(results) == 0:
        return np.nan, None, None

    # Convert to arrays
    L0_vals = np.array([r['L0'] for r in results])
    likelihoods = np.array([r['likelihood'] for r in results])

    # Posterior (assuming flat prior)
    posterior = likelihoods / np.trapz(likelihoods, L0_vals)

    # Best-fit L₀
    idx_max = np.argmax(posterior)
    L0_best = L0_vals[idx_max]

    print(f"  → Best L₀ = {L0_best:.1f} MeV")
    print(f"  → Predicted period = {results[idx_max]['P_pred']:.1f} days")
    print(f"  → Effective length = {results[idx_max]['L_eff']/1e5:.2f} km")

    return L0_best, L0_vals, posterior, results

# ============================================================================
# Analysis
# ============================================================================

print("=" * 80)
print("PSR J1522-5735 L₀ MEASUREMENT")
print("=" * 80)
print()
print("Pulsar parameters:")
print(f"  Spin period: {P_spin*1e3:.1f} ms")
print(f"  Spin frequency: {Omega_J1522/(2*np.pi):.3f} Hz")
print(f"  Characteristic age: {tau_c/(365.25*86400*1e3):.1f} kyr")
print(f"  Mass (assumed): {M_J1522/const.M_sun:.1f} M_☉")
print(f"  Radius (assumed): {R_J1522/1e5:.0f} km")
print(f"  Temperature (estimated): {T_J1522/1e8:.1f} × 10⁸ K")
print()
print("Observed oscillations (Zhou et al. 2024):")
print(f"  Oscillation 1: {P_obs_1:.0f} days")
print(f"  Oscillation 2: {P_obs_2:.0f} days")
print()
print("Method: Same vortex model + α = 0.08 calibrated from Vela")
print("=" * 80)
print()

L0_grid = np.linspace(40, 80, 201)

# Analyze oscillation 1 (248 days)
print("Analyzing Oscillation 1 (248 days)...")
L0_1, L0_vals_1, post_1, results_1 = infer_L0_bayesian(
    P_obs_1, sigma_P, rho, T_J1522, R_J1522, M_J1522, Omega_J1522, L0_grid
)
print()

# Analyze oscillation 2 (135 days)
print("Analyzing Oscillation 2 (135 days)...")
L0_2, L0_vals_2, post_2, results_2 = infer_L0_bayesian(
    P_obs_2, sigma_P, rho, T_J1522, R_J1522, M_J1522, Omega_J1522, L0_grid
)
print()

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Oscillation 1 (248 d): L₀ = {L0_1:.1f} MeV")
print(f"Oscillation 2 (135 d): L₀ = {L0_2:.1f} MeV")
print()

if not np.isnan(L0_1) and not np.isnan(L0_2):
    L0_mean = (L0_1 + L0_2) / 2
    L0_spread = abs(L0_1 - L0_2) / 2
    print(f"Combined: L₀ = {L0_mean:.1f} ± {L0_spread:.1f} MeV")
    print()

    # Compare with Vela
    L0_vela = 60.5
    print(f"Comparison:")
    print(f"  Vela:       L₀ = {L0_vela:.1f} MeV")
    print(f"  J1522-5735: L₀ = {L0_mean:.1f} ± {L0_spread:.1f} MeV")
    print()

    difference = abs(L0_mean - L0_vela)
    print(f"Difference: {difference:.1f} MeV")

    if difference < 5:
        print("✓ CONSISTENT: Results agree within ~5 MeV")
        print("  → Method appears universal across pulsars")
    else:
        print("✗ INCONSISTENT: Results differ by > 5 MeV")
        print("  → Suggests calibration may not be universal")
        print("  → Or stellar parameter assumptions are wrong")

print()
print("=" * 80)

# ============================================================================
# Visualization
# ============================================================================

if L0_vals_1 is not None and L0_vals_2 is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Oscillation 1
    ax1.plot(L0_vals_1, post_1, 'b-', lw=2.5, label=f'248 d oscillation')
    ax1.axvline(L0_1, color='blue', ls='--', lw=2, alpha=0.7)
    ax1.axvline(60.5, color='red', ls='--', lw=2, alpha=0.7, label='Vela result')
    ax1.fill_between([40, 80], [0, 0], [1, 1], alpha=0.05, color='red')
    ax1.set_xlabel('L₀ (MeV)', fontsize=13)
    ax1.set_ylabel('Posterior Probability', fontsize=13)
    ax1.set_title(f'PSR J1522-5735 Oscillation 1\nL₀ = {L0_1:.1f} MeV',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Oscillation 2
    ax2.plot(L0_vals_2, post_2, 'g-', lw=2.5, label=f'135 d oscillation')
    ax2.axvline(L0_2, color='green', ls='--', lw=2, alpha=0.7)
    ax2.axvline(60.5, color='red', ls='--', lw=2, alpha=0.7, label='Vela result')
    ax2.set_xlabel('L₀ (MeV)', fontsize=13)
    ax2.set_ylabel('Posterior Probability', fontsize=13)
    ax2.set_title(f'PSR J1522-5735 Oscillation 2\nL₀ = {L0_2:.1f} MeV',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/psr_j1522_L0_measurement.pdf', dpi=300, bbox_inches='tight')
    print("Figure saved: figures/psr_j1522_L0_measurement.pdf")
    print()
