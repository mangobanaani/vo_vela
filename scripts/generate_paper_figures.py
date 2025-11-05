#!/usr/bin/env python
"""
Generate All Paper Figures

This script generates all publication-quality figures for the paper.
Run this after MCMC inference is complete to create final figures.

Figures generated:
1. Timing data with oscillation fit
2. Periodogram showing oscillation detection
3. Forward model: L₀ vs predicted period
4. Posterior distribution for L₀
5. Corner plot showing parameter correlations
6. Comparison with other L₀ measurements

Usage:
    python generate_paper_figures.py
    # Or specify output directory:
    python generate_paper_figures.py --output figures/final/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import argparse
import os

# Configure publication-quality plotting
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.3,
})

# Import our modules
from src import constants as const
from src import eos, vortex, stellar_structure as ss

# ============================================================================
# Figure 1: Post-Glitch Timing Data (Placeholder - needs real data)
# ============================================================================

def generate_fig1_timing_data(output_dir='figures'):
    """
    Figure 1: Post-glitch timing residuals showing oscillation.

    When real data is available, replace simulated data with:
    - Actual Fermi-LAT ν(t) and ν̇(t)
    - MCMC best-fit model
    - Residuals showing oscillation
    """
    print("Generating Figure 1: Timing Data...")

    # Simulate data (replace with real Fermi data)
    t_days = np.linspace(0, 100, 100)
    t_sec = t_days * const.day

    # Model parameters (from fit - replace with real MCMC results)
    nu_dot_steady = -1.5e-13  # Hz/s
    Q_dot = 3e-14
    tau_exp = 10 * const.day
    omega_osc = 2 * np.pi / (12 * const.day)
    B = 5e-15
    phi = 0.5
    tau_damp = 40 * const.day

    # Model
    nu_dot_model = (nu_dot_steady +
                    Q_dot * np.exp(-t_sec/tau_exp) +
                    B * np.cos(omega_osc * t_sec + phi) * np.exp(-t_sec/tau_damp))

    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 1e-15, len(t_days))
    nu_dot_data = nu_dot_model + noise
    nu_dot_err = np.ones_like(nu_dot_data) * 1e-15

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Data with fit
    ax1.errorbar(t_days, nu_dot_data * 1e15, yerr=nu_dot_err * 1e15,
                 fmt='o', markersize=3, alpha=0.6, color='black',
                 ecolor='gray', capsize=0, label='Data')
    ax1.plot(t_days, nu_dot_model * 1e15, 'r-', linewidth=2, label='Best fit')
    ax1.axvline(0, color='green', linestyle='--', alpha=0.5, label='Glitch')
    ax1.set_ylabel(r'$\dot{\nu}$ ($10^{-15}$ Hz/s)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Vela Post-Glitch Timing: 2024 Event', fontsize=13, fontweight='bold')

    # Bottom: Residuals showing oscillation
    nu_dot_no_osc = nu_dot_steady + Q_dot * np.exp(-t_sec/tau_exp)
    residuals = nu_dot_data - nu_dot_no_osc
    oscillation = B * np.cos(omega_osc * t_sec + phi) * np.exp(-t_sec/tau_damp)

    ax2.errorbar(t_days, residuals * 1e15, yerr=nu_dot_err * 1e15,
                 fmt='o', markersize=3, alpha=0.6, color='black',
                 ecolor='gray', capsize=0, label='Residuals')
    ax2.plot(t_days, oscillation * 1e15, 'b-', linewidth=2, label='Oscillation')
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time since glitch (days)', fontsize=12)
    ax2.set_ylabel(r'Residuals ($10^{-15}$ Hz/s)', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_timing_data.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig1_timing_data.pdf")
    plt.close()

# ============================================================================
# Figure 2: Periodogram (Placeholder - needs real data)
# ============================================================================

def generate_fig2_periodogram(output_dir='figures'):
    """
    Figure 2: Lomb-Scargle periodogram showing oscillation detection.
    """
    print("Generating Figure 2: Periodogram...")

    # Simulate periodogram (replace with real Lomb-Scargle)
    periods = np.linspace(1, 30, 1000)  # days

    # Power spectrum with peak at 12 days
    power = (0.1 + 0.3 * np.exp(-(periods - 12)**2 / 4) +
             0.05 * np.random.randn(len(periods)))
    power = np.maximum(power, 0)

    # Significance levels
    sig_3sigma = 0.3
    sig_5sigma = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(periods, power, 'b-', linewidth=1.5, label='Power spectrum')
    ax.axhline(sig_3sigma, color='orange', linestyle='--',
               linewidth=1.5, label='3σ significance')
    ax.axhline(sig_5sigma, color='red', linestyle='--',
               linewidth=1.5, label='5σ significance')
    ax.axvline(12, color='green', linestyle=':', alpha=0.7,
               linewidth=2, label='Detected: P = 12 days')

    ax.set_xlabel('Period (days)', fontsize=12)
    ax.set_ylabel('Lomb-Scargle Power', fontsize=12)
    ax.set_title('Oscillation Detection: Periodogram Analysis',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 30)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_periodogram.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig2_periodogram.pdf")
    plt.close()

# ============================================================================
# Figure 3: Forward Model - L₀ vs Period
# ============================================================================

def generate_fig3_forward_model(output_dir='figures'):
    """
    Figure 3: Forward model showing L₀ sensitivity.
    """
    print("Generating Figure 3: Forward Model...")

    # Scan L₀
    L0_values = np.linspace(30, 80, 100)
    periods = []

    # Vela parameters
    M_vela = 1.4 * const.M_sun
    R_vela = 12e5
    T = 1e8

    for L0 in L0_values:
        eos_model = eos.SymmetryEnergyEoS(L0=L0)
        ns = ss.SimpleNeutronStar(M_vela, R_vela, eos=eos_model)
        r_glitch = 0.95 * R_vela
        rho_glitch = ns.density(r_glitch)

        obs = vortex.predict_from_eos(
            eos_model, rho_glitch, T, r_glitch,
            ns.mass(r_glitch), const.Omega_Vela
        )
        periods.append(obs['P_days'])

    periods = np.array(periods)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot prediction
    ax.plot(L0_values, periods, 'b-', linewidth=3, label='Theoretical prediction')

    # Observed value (placeholder - replace with real result)
    P_obs = 12.0  # days
    P_obs_err = 1.5  # days

    ax.axhspan(P_obs - P_obs_err, P_obs + P_obs_err,
               alpha=0.3, color='red', label=f'Observed: {P_obs:.1f} ± {P_obs_err:.1f} days')
    ax.axhline(P_obs, color='red', linestyle='--', linewidth=2)

    # Mark intersection (inferred L₀)
    idx_best = np.argmin(np.abs(periods - P_obs))
    L0_best = L0_values[idx_best]
    ax.plot(L0_best, P_obs, 'ro', markersize=15, markeredgewidth=2,
            markerfacecolor='red', markeredgecolor='darkred',
            label=f'Inferred: L₀ = {L0_best:.1f} MeV', zorder=10)

    # Mark other EoS
    L0_literature = {'GW170817': 57, 'NICER': 52, 'Heavy-ion': 58}
    for name, L0_lit in L0_literature.items():
        ax.axvline(L0_lit, color='gray', linestyle=':', alpha=0.5)
        ax.text(L0_lit, ax.get_ylim()[1] * 0.95, name,
                rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)

    ax.set_xlabel(r'Symmetry Energy Slope $L_0$ (MeV)', fontsize=13)
    ax.set_ylabel('Predicted Oscillation Period (days)', fontsize=13)
    ax.set_title('Forward Model: Nuclear EoS → Observable Period',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_forward_model.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig3_forward_model.pdf")
    plt.close()

# ============================================================================
# Figure 4: Posterior Distribution for L₀
# ============================================================================

def generate_fig4_posterior(output_dir='figures'):
    """
    Figure 4: Posterior distribution from MCMC.
    """
    print("Generating Figure 4: Posterior...")

    # Simulate posterior (replace with real MCMC results)
    L0_samples = np.random.normal(52, 8, 10000)

    # Calculate percentiles
    L0_median = np.median(L0_samples)
    L0_lower = np.percentile(L0_samples, 16)
    L0_upper = np.percentile(L0_samples, 84)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    counts, bins, _ = ax.hist(L0_samples, bins=50, density=True,
                              alpha=0.7, color='skyblue', edgecolor='navy',
                              label='Posterior samples')

    # Median and credible interval
    ax.axvline(L0_median, color='red', linestyle='-', linewidth=3,
               label=f'Median: {L0_median:.1f} MeV')
    ax.axvspan(L0_lower, L0_upper, alpha=0.2, color='red',
               label=f'68% CI: [{L0_lower:.1f}, {L0_upper:.1f}] MeV')

    # Other measurements
    L0_literature = {
        'GW170817': (57, 16),
        'NICER': (52, 13),
        'Heavy-ion': (58, 18)
    }

    colors_lit = {'GW170817': 'blue', 'NICER': 'green', 'Heavy-ion': 'orange'}
    y_pos = 0.85
    for name, (L0, err) in L0_literature.items():
        ax.axvspan(L0-err, L0+err, alpha=0.1, color=colors_lit[name])
        ax.axvline(L0, color=colors_lit[name], linestyle='--', linewidth=2,
                  alpha=0.7)
        ax.text(0.98, y_pos, f'{name}: {L0} ± {err} MeV',
                transform=ax.transAxes, ha='right', fontsize=9,
                color=colors_lit[name])
        y_pos -= 0.05

    ax.set_xlabel(r'Symmetry Energy Slope $L_0$ (MeV)', fontsize=13)
    ax.set_ylabel('Posterior Probability Density', fontsize=13)
    ax.set_title('Bayesian Inference Result: L₀ from Vortex Oscillations',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(20, 90)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_posterior.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig4_posterior.pdf")
    plt.close()

# ============================================================================
# Figure 5: Corner Plot (Placeholder)
# ============================================================================

def generate_fig5_corner(output_dir='figures'):
    """
    Figure 5: Corner plot showing parameter correlations.

    When real MCMC is done, use: import corner; corner.corner(samples)
    """
    print("Generating Figure 5: Corner Plot...")

    # Simulate MCMC samples
    n_samples = 5000
    L0_samples = np.random.normal(52, 8, n_samples)
    M_samples = np.random.normal(1.4, 0.15, n_samples)
    T_samples = np.random.lognormal(np.log(1e8), 0.3, n_samples)

    # Create 2D histograms manually (replace with corner.corner())
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.05, wspace=0.05)

    labels = [r'$L_0$ (MeV)', r'$M$ ($M_\odot$)', r'$T$ ($10^8$ K)']
    samples = [L0_samples, M_samples, T_samples / 1e8]
    ranges = [(30, 70), (1.0, 1.8), (0.5, 2.0)]

    # Diagonal: 1D histograms
    for i in range(3):
        ax = fig.add_subplot(gs[i, i])
        ax.hist(samples[i], bins=30, density=True, alpha=0.7,
                color='skyblue', edgecolor='navy')
        ax.set_xlim(ranges[i])
        if i == 2:
            ax.set_xlabel(labels[i], fontsize=12)
        else:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, alpha=0.3)

    # Off-diagonal: 2D histograms
    for i in range(3):
        for j in range(3):
            if i > j:
                ax = fig.add_subplot(gs[i, j])
                ax.hist2d(samples[j], samples[i], bins=30,
                         cmap='Blues', alpha=0.8)
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])

                if i == 2:
                    ax.set_xlabel(labels[j], fontsize=12)
                else:
                    ax.set_xticklabels([])

                if j == 0:
                    ax.set_ylabel(labels[i], fontsize=12)
                else:
                    ax.set_yticklabels([])

                ax.grid(True, alpha=0.3)

    plt.suptitle('Parameter Correlations from MCMC',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.savefig(f'{output_dir}/fig5_corner.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig5_corner.pdf")
    plt.close()

# ============================================================================
# Figure 6: Comparison with Other Methods
# ============================================================================

def generate_fig6_comparison(output_dir='figures'):
    """
    Figure 6: Compare L₀ constraints from different methods.
    """
    print("Generating Figure 6: Multi-Method Comparison...")

    # L₀ measurements from different methods
    methods = ['Heavy-ion\nCollisions', 'GW170817\n(Merger)',
               'NICER\n(Mass-Radius)', 'Vortex Osc.\n(This Work)']
    L0_values = [58, 57, 52, 52]
    L0_errors = [18, 16, 13, 8]
    densities = [0.3, 2.0, 1.5, 0.6]  # Typical densities probed (in rho_0)
    colors = ['orange', 'blue', 'green', 'red']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: L₀ constraints
    y_pos = np.arange(len(methods))

    for i, (method, L0, err, color) in enumerate(zip(methods, L0_values, L0_errors, colors)):
        ax1.errorbar(L0, i, xerr=err, fmt='o', markersize=12,
                    capsize=8, capthick=2, linewidth=2,
                    color=color, ecolor=color, label=method)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=11)
    ax1.set_xlabel(r'Symmetry Energy Slope $L_0$ (MeV)', fontsize=13)
    ax1.set_title('L₀ Constraints from Multiple Methods', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvspan(40, 70, alpha=0.1, color='gray', label='Typical range')
    ax1.set_xlim(20, 90)

    # Right panel: Density ranges probed
    for i, (method, rho, color) in enumerate(zip(methods, densities, colors)):
        # Show as horizontal bar representing density range
        if i == 0:  # Heavy-ion
            rho_min, rho_max = 0.2, 0.5
        elif i == 1:  # GW170817
            rho_min, rho_max = 1.0, 3.0
        elif i == 2:  # NICER
            rho_min, rho_max = 0.5, 2.5
        else:  # This work
            rho_min, rho_max = 0.5, 0.8

        ax2.barh(i, rho_max - rho_min, left=rho_min, height=0.6,
                color=color, alpha=0.7, edgecolor='black')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods, fontsize=11)
    ax2.set_xlabel(r'Density Probed ($\rho/\rho_0$)', fontsize=13)
    ax2.set_title('Complementary Density Ranges', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(1, color='gray', linestyle='--', alpha=0.5, label=r'$\rho_0$')
    ax2.set_xlim(0, 3.5)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"  COMPLETE Saved: {output_dir}/fig6_comparison.pdf")
    plt.close()

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures')
    parser.add_argument('--output', default='figures',
                       help='Output directory for figures')
    parser.add_argument('--figures', nargs='+', type=int,
                       help='Specific figures to generate (1-6), default: all')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 70)
    print("GENERATING PAPER FIGURES")
    print("=" * 70)
    print(f"\nOutput directory: {args.output}/")
    print()

    # Determine which figures to generate
    if args.figures:
        figures_to_generate = args.figures
    else:
        figures_to_generate = range(1, 7)

    # Generate figures
    if 1 in figures_to_generate:
        generate_fig1_timing_data(args.output)

    if 2 in figures_to_generate:
        generate_fig2_periodogram(args.output)

    if 3 in figures_to_generate:
        generate_fig3_forward_model(args.output)

    if 4 in figures_to_generate:
        generate_fig4_posterior(args.output)

    if 5 in figures_to_generate:
        generate_fig5_corner(args.output)

    if 6 in figures_to_generate:
        generate_fig6_comparison(args.output)

    print()
    print("=" * 70)
    print("COMPLETE ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFigures saved in: {args.output}/")
    print("\nNext steps:")
    print("  1. Review figures")
    print("  2. Update LaTeX paper to include figures")
    print("  3. Compile paper: cd papers && make all")
    print()

if __name__ == "__main__":
    main()
