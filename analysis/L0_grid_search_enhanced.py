#!/usr/bin/env python3
"""
Enhanced L₀ Grid Search with Working Sensitivity Model
======================================================

Uses the enhanced model from measure_L0_bayesian.py that we know works.
Applies it to all three glitches for a joint constraint.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import constants as const, eos, superfluid as sf, vortex

# Enhanced model functions (from measure_L0_bayesian.py)
def effective_mass_ratio(rho, L0):
    x = rho / const.rho_0
    y = (L0 - 55.0) / 15.0
    m_base = 0.75 + 0.10 * (x - 0.6)
    m_star_over_m = m_base * (1.0 + 0.15 * y)
    return np.clip(m_star_over_m, 0.60, 0.95)

def pairing_gap_L0_dependent(rho, L0):
    x = rho / const.rho_0
    Delta_base = 0.1 * np.exp(-5.0 * (x - 0.6)**2)
    m_star_ratio = effective_mass_ratio(rho, L0)
    dos_enhancement = np.sqrt(m_star_ratio / 0.75)
    Delta = Delta_base * dos_enhancement
    return Delta * const.MeV

def predict_period_enhanced(L0, L_eff, rho_glitch, T, R, M, Omega):
    eos_model = eos.SymmetryEnergyEoS(L0=L0)
    f_n = eos_model.neutron_fraction(rho_glitch)
    m_star_ratio = effective_mass_ratio(rho_glitch, L0)
    Delta = pairing_gap_L0_dependent(rho_glitch, L0)
    T_c = 0.57 * Delta / const.k_B
    if T >= T_c:
        f_s = 0.0
    else:
        f_s = 1.0 - (T / T_c)**2
        y = (L0 - 55.0) / 15.0
        correction = 1.0 + 0.05 * y
        f_s = np.clip(f_s * correction, 0.0, 1.0)
    rho_n = rho_glitch * f_n * f_s
    if rho_n <= 0:
        return np.inf
    n_n = (rho_glitch / const.m_n) * f_n
    k_F = (3 * np.pi**2 * n_n)**(1/3)
    m_star = const.m_n * m_star_ratio
    v_F = const.hbar * k_F / m_star
    mode_params = vortex.BendingModeParameters(
        boundary_condition='clamped-free', mode_number=0, effective_length=L_eff)
    try:
        omega = vortex.oscillation_frequency(rho_n, R, Omega, Delta, v_F, mode_params)
        if omega <= 0:
            return np.inf
        P_seconds = 2 * np.pi / omega
        P_days = P_seconds / (24 * 3600)
        return P_days
    except:
        return np.inf

# Glitch data
glitches = {
    'G1': {'P_obs': 314.1, 'sigma': 0.2, 'L_eff': 7.51e5},
    'G3a': {'P_obs': 344.0, 'sigma': 6.0, 'L_eff': 8.22e5},
    'G3b': {'P_obs': 153.0, 'sigma': 3.0, 'L_eff': 3.66e5}
}

M = 1.4 * const.M_sun
R = 12e5
Omega = const.Omega_Vela
rho = 0.6 * const.rho_0
T = 1e8

print("="*80)
print("ENHANCED L₀ GRID SEARCH - ALL GLITCHES")
print("="*80)
print()

L0_grid = np.linspace(30, 90, 121)
posteriors = {}

for name, data in glitches.items():
    print(f"Processing {name}...")
    periods = []
    for L0 in L0_grid:
        P = predict_period_enhanced(L0, data['L_eff'], rho, T, R, M, Omega)
        periods.append(P)
    periods = np.array(periods)
    valid = np.isfinite(periods) & (periods > 0) & (periods < 1000)
    L0_valid = L0_grid[valid]
    periods_valid = periods[valid]
    prior = np.ones_like(L0_valid) / len(L0_valid)
    likelihood = norm.pdf(periods_valid, loc=data['P_obs'], scale=data['sigma'])
    posterior = likelihood * prior
    if np.trapz(posterior, L0_valid) > 0:
        posterior /= np.trapz(posterior, L0_valid)
    posteriors[name] = {'L0': L0_valid, 'posterior': posterior, 'periods': periods_valid}
    idx_map = np.argmax(posterior)
    L0_map = L0_valid[idx_map]
    cumulative = np.cumsum(posterior) * (L0_valid[1] - L0_valid[0])
    idx_16 = np.argmin(np.abs(cumulative - 0.16))
    idx_84 = np.argmin(np.abs(cumulative - 0.84))
    L0_16 = L0_valid[idx_16]
    L0_84 = L0_valid[idx_84]
    print(f"  L₀ = {L0_map:.1f} +{L0_84-L0_map:.1f} -{L0_map-L0_16:.1f} MeV")
    print()

# Joint posterior
print("Computing joint posterior...")
L0_joint = L0_grid
joint_post = np.ones_like(L0_joint)
for name, data in posteriors.items():
    interp_post = np.interp(L0_joint, data['L0'], data['posterior'], left=0, right=0)
    joint_post *= interp_post
joint_post /= np.trapz(joint_post, L0_joint) if np.trapz(joint_post, L0_joint) > 0 else 1
idx_map = np.argmax(joint_post)
L0_map = L0_joint[idx_map]
cumulative = np.cumsum(joint_post) * (L0_joint[1] - L0_joint[0])
idx_16 = np.argmin(np.abs(cumulative - 0.16))
idx_84 = np.argmin(np.abs(cumulative - 0.84))
L0_16 = L0_joint[idx_16]
L0_84 = L0_joint[idx_84]

print()
print("="*80)
print("JOINT CONSTRAINT FROM ALL THREE GLITCHES")
print("="*80)
print()
print(f" L₀ = {L0_map:.1f} +{L0_84-L0_map:.1f} -{L0_map-L0_16:.1f} MeV (68% CI)")
print()

# Save results
results = {
    'individual': {
        name: {
            'L0_MAP': float(data['L0'][np.argmax(data['posterior'])]),
            'L0_16': float(data['L0'][np.argmin(np.abs(np.cumsum(data['posterior']) * (data['L0'][1]-data['L0'][0]) - 0.16))]),
            'L0_84': float(data['L0'][np.argmin(np.abs(np.cumsum(data['posterior']) * (data['L0'][1]-data['L0'][0]) - 0.84))])
        }
        for name, data in posteriors.items()
    },
    'joint': {
        'L0_MAP': float(L0_map),
        'L0_16': float(L0_16),
        'L0_84': float(L0_84)
    }
}

with open('L0_grid_search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved: L0_grid_search_results.json")
print()
print("="*80)
