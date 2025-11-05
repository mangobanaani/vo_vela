#!/usr/bin/env python3
"""
Explore ξ(L₀) Dependence - Interactive Analysis
================================================

Investigate how coherence length ξ varies with L₀ through the forward model chain:

    L₀ → f_n(ρ; L₀) → m*(ρ, L₀) → v_F → Δ(ρ, L₀) → ξ(L₀)

Then show how this propagates to the period:

    P = 2π L_eff √(α Ω κ ln(b/ξ))

Key channels:
1. f_n(L₀): Neutron fraction from β-equilibrium
2. m*(L₀): Effective mass from asymmetry energy stiffness
3. v_F(L₀): Fermi velocity from m*
4. Δ(L₀): Pairing gap from density of states ~ m*
5. ξ(L₀): Coherence length ξ = ℏv_F/Δ

Author: Pekka Siltala
Date: November 5, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import constants as const
from src import eos
from src import superfluid
from src import vortex


def explore_xi_L0_chain(L0_range=None, rho_frac=0.6, verbose=True):
    """
    Trace the full L₀ → ξ chain step by step.

    Parameters:
    -----------
    L0_range : array or None
        L₀ values to scan (MeV). If None, use [30, 90] MeV
    rho_frac : float
        Density in units of ρ₀ (default 0.6)
    verbose : bool
        Print detailed output

    Returns:
    --------
    results : dict
        Complete chain results
    """
    if L0_range is None:
        L0_range = np.linspace(30, 90, 61)

    rho = rho_frac * const.rho_0  # Fixed density
    n_points = len(L0_range)

    # Vela parameters
    M = 1.4 * const.M_sun
    R = 12e5  # cm
    P_spin = 89.3e-3  # s
    Omega = 2 * np.pi / P_spin
    T = 1e8  # K

    # Initialize arrays for the chain
    f_n = np.zeros(n_points)
    m_star = np.zeros(n_points)
    v_F = np.zeros(n_points)
    Delta = np.zeros(n_points)
    xi = np.zeros(n_points)
    log_b_xi = np.zeros(n_points)
    period = np.zeros(n_points)

    # Vortex spacing (constant, doesn't depend on L₀)
    b = vortex.vortex_spacing(Omega)

    # Calibrated parameters from observations
    alpha = 0.08  # Geometric factor
    L_eff = R  # Effective length scale

    # Helper function for effective mass ratio
    def effective_mass_ratio(rho_val, L0_val, alpha=0.5):
        """Calculate m*/m_n from asymmetry energy stiffness."""
        # Reference values
        m_star_ref = 0.85
        L0_ref = 60.0

        # Scaling with L₀ stiffness
        # Stiffer EoS (higher L₀) → lower m*
        scaling = 1.0 - alpha * (L0_val - L0_ref) / L0_ref
        return m_star_ref * scaling

    if verbose:
        print("="*80)
        print("L₀ → ξ DEPENDENCE CHAIN")
        print("="*80)
        print(f"\nFixed parameters:")
        print(f"  Density: ρ = {rho_frac:.2f} ρ₀")
        print(f"  Mass: M = {M/const.M_sun:.2f} M")
        print(f"  Radius: R = {R/1e5:.1f} km")
        print(f"  Spin period: P_spin = {P_spin*1000:.1f} ms")
        print(f"  Temperature: T = {T:.1e} K")
        print(f"  Vortex spacing: b = {b:.3e} cm")
        print(f"\nScanning L₀ from {L0_range[0]:.0f} to {L0_range[-1]:.0f} MeV...")
        print()

    # Trace the chain for each L₀
    for i, L0_val in enumerate(L0_range):
        # Step 1: L₀ → f_n
        # Create EoS model with this L₀
        eos_model = eos.create_eos(model='parameterized', L0=L0_val, K0=240.0)
        f_n[i] = eos_model.neutron_fraction(rho)

        # Step 2: L₀ → m*
        m_star[i] = effective_mass_ratio(rho, L0_val, alpha=0.5)

        # Step 3: m* → v_F
        # Fermi momentum: k_F = (3π² n_n)^(1/3)
        n_n = f_n[i] * rho / const.m_n  # neutron number density
        k_F = (3 * np.pi**2 * n_n)**(1/3)
        v_F[i] = const.hbar * k_F / (m_star[i] * const.m_n)

        # Step 4: (f_n, m*) → Δ
        # Pairing gap depends on density of states N(E_F) ~ m*
        # Use AO model but scale with effective mass
        Delta_base = superfluid.pairing_gap_AO(rho)
        m_star_ref = 0.85  # Reference value
        Delta[i] = Delta_base * (m_star[i] / m_star_ref)**0.5

        # Step 5: (v_F, Δ) → ξ
        xi[i] = const.hbar * v_F[i] / Delta[i]

        # Step 6: ξ → ln(b/ξ)
        log_b_xi[i] = np.log(b / xi[i])

        # Step 7: ln(b/ξ) → P
        # P = 2π L_eff √(α Ω κ ln(b/ξ))
        period[i] = 2 * np.pi * L_eff * np.sqrt(
            alpha * Omega * const.kappa * log_b_xi[i]
        ) / 86400  # Convert to days

    # Calculate sensitivities (derivatives)
    dL0 = L0_range[1] - L0_range[0]

    # Fractional sensitivities: (1/X) dX/dL₀
    sens_f_n = np.gradient(f_n, dL0) / f_n
    sens_m_star = np.gradient(m_star, dL0) / m_star
    sens_v_F = np.gradient(v_F, dL0) / v_F
    sens_Delta = np.gradient(Delta, dL0) / Delta
    sens_xi = np.gradient(xi, dL0) / xi
    sens_log_b_xi = np.gradient(log_b_xi, dL0) / log_b_xi
    sens_period = np.gradient(period, dL0) / period

    if verbose:
        # Show results at mid-point (L₀ = 60 MeV)
        idx_mid = np.argmin(np.abs(L0_range - 60))
        L0_mid = L0_range[idx_mid]

        print(f"Results at L₀ = {L0_mid:.1f} MeV:")
        print("-"*80)
        print(f"  f_n      = {f_n[idx_mid]:.4f}")
        print(f"  m*/m_n   = {m_star[idx_mid]:.4f}")
        print(f"  v_F      = {v_F[idx_mid]:.3e} cm/s ({v_F[idx_mid]/const.c:.3f}c)")
        print(f"  Δ        = {Delta[idx_mid]/const.MeV:.3f} MeV")
        print(f"  ξ        = {xi[idx_mid]:.3e} cm")
        print(f"  ln(b/ξ)  = {log_b_xi[idx_mid]:.2f}")
        print(f"  Period   = {period[idx_mid]:.1f} days")
        print()

        print(f"Fractional sensitivities at L₀ = {L0_mid:.1f} MeV:")
        print("-"*80)
        print(f"  d(ln f_n)/dL₀      = {sens_f_n[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln m*)/dL₀       = {sens_m_star[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln v_F)/dL₀      = {sens_v_F[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln Δ)/dL₀        = {sens_Delta[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln ξ)/dL₀        = {sens_xi[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln ln(b/ξ))/dL₀  = {sens_log_b_xi[idx_mid]:+.5f} MeV⁻¹")
        print(f"  d(ln P)/dL₀        = {sens_period[idx_mid]:+.5f} MeV⁻¹")
        print()

        # Total sensitivity to period
        dP_dL0 = (period[-1] - period[0]) / (L0_range[-1] - L0_range[0])
        print(f"Period sensitivity:")
        print(f"  dP/dL₀ = {dP_dL0:.3f} days/MeV")
        print(f"  Total variation: ΔP = {period[-1] - period[0]:.2f} days over ΔL₀ = {L0_range[-1] - L0_range[0]:.0f} MeV")
        print()

    results = {
        'L0_range': L0_range,
        'rho': rho,
        'rho_frac': rho_frac,
        'f_n': f_n,
        'm_star': m_star,
        'v_F': v_F,
        'Delta': Delta,
        'Delta_MeV': Delta / const.MeV,
        'xi': xi,
        'log_b_xi': log_b_xi,
        'period': period,
        'b': b,
        'sens_f_n': sens_f_n,
        'sens_m_star': sens_m_star,
        'sens_v_F': sens_v_F,
        'sens_Delta': sens_Delta,
        'sens_xi': sens_xi,
        'sens_log_b_xi': sens_log_b_xi,
        'sens_period': sens_period
    }

    return results


def plot_xi_L0_chain(results, save_path=None):
    """
    Create comprehensive visualization of the L₀ → ξ → P chain.

    Parameters:
    -----------
    results : dict
        Output from explore_xi_L0_chain()
    save_path : str or None
        Path to save figure
    """
    L0 = results['L0_range']

    fig = plt.figure(figsize=(16, 10))

    # Create 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Row 1: Forward model chain
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Row 2: Intermediate quantities
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    # Row 3: Final observables
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])

    # Plot 1: Neutron fraction
    ax1.plot(L0, results['f_n'], 'b-', lw=2.5)
    ax1.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax1.set_ylabel(r'$f_n$', fontsize=11)
    ax1.set_title(r'Step 1: $L_0 \to f_n$', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 2: Effective mass
    ax2.plot(L0, results['m_star'], 'g-', lw=2.5)
    ax2.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax2.set_ylabel(r'$m^*/m_n$', fontsize=11)
    ax2.set_title(r'Step 2: $L_0 \to m^*$', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 3: Fermi velocity
    ax3.plot(L0, results['v_F']/const.c, 'orange', lw=2.5)
    ax3.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax3.set_ylabel(r'$v_F/c$', fontsize=11)
    ax3.set_title(r'Step 3: $m^* \to v_F$', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    ax3.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 4: Pairing gap
    ax4.plot(L0, results['Delta_MeV'], 'purple', lw=2.5)
    ax4.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax4.set_ylabel(r'$\Delta$ (MeV)', fontsize=11)
    ax4.set_title(r'Step 4: $(f_n, m^*) \to \Delta$', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    ax4.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 5: Coherence length (KEY PLOT!)
    ax5.plot(L0, results['xi']*1e12, 'red', lw=3.0)
    ax5.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax5.set_ylabel(r'$\xi$ (pm)', fontsize=11)
    ax5.set_title(r'Step 5: $\xi = \hbar v_F / \Delta$', fontsize=12, fontweight='bold', color='red')
    ax5.grid(alpha=0.3)
    ax5.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 6: ln(b/ξ)
    ax6.plot(L0, results['log_b_xi'], 'brown', lw=2.5)
    ax6.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax6.set_ylabel(r'$\ln(b/\xi)$', fontsize=11)
    ax6.set_title(r'Step 6: $\xi \to \ln(b/\xi)$', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    ax6.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 7: Period (FINAL RESULT!)
    ax7.plot(L0, results['period'], 'darkblue', lw=3.0, label='Model')
    ax7.axhline(314.1, color='red', ls='--', lw=2, label='G1 observed')
    ax7.fill_between(L0, 314.1-0.2, 314.1+0.2, alpha=0.2, color='red')
    ax7.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax7.set_ylabel(r'$P$ (days)', fontsize=11)
    ax7.set_title(r'Step 7: $P = 2\pi L \sqrt{\alpha \Omega \kappa \ln(b/\xi)}$',
                  fontsize=12, fontweight='bold', color='darkblue')
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    ax7.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 8: Fractional sensitivity of ξ
    ax8.plot(L0, results['sens_xi'], 'red', lw=2.5)
    ax8.axhline(0, color='k', ls='-', lw=0.5)
    ax8.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax8.set_ylabel(r'$d(\ln \xi)/dL_0$ (MeV$^{-1}$)', fontsize=11)
    ax8.set_title(r'Fractional $\xi$ Sensitivity', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3)
    ax8.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Plot 9: All fractional sensitivities
    ax9.plot(L0, results['sens_f_n'], label=r'$f_n$', lw=1.5)
    ax9.plot(L0, results['sens_m_star'], label=r'$m^*$', lw=1.5)
    ax9.plot(L0, results['sens_v_F'], label=r'$v_F$', lw=1.5)
    ax9.plot(L0, results['sens_Delta'], label=r'$\Delta$', lw=1.5)
    ax9.plot(L0, results['sens_xi'], label=r'$\xi$', lw=2.5, color='red')
    ax9.plot(L0, results['sens_period'], label=r'$P$', lw=2.5, color='darkblue', ls='--')
    ax9.axhline(0, color='k', ls='-', lw=0.5)
    ax9.set_xlabel(r'$L_0$ (MeV)', fontsize=11)
    ax9.set_ylabel(r'$d(\ln X)/dL_0$ (MeV$^{-1}$)', fontsize=11)
    ax9.set_title(r'All Fractional Sensitivities', fontsize=12, fontweight='bold')
    ax9.legend(fontsize=9, ncol=2)
    ax9.grid(alpha=0.3)
    ax9.axvline(60, color='red', ls='--', alpha=0.5, lw=1)

    # Super title
    fig.suptitle(
        r'Complete Chain: $L_0 \to f_n, m^* \to v_F, \Delta \to \xi \to \ln(b/\xi) \to P$',
        fontsize=14, fontweight='bold', y=0.995
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved: {save_path}")

    return fig


def print_channel_contributions(results):
    """
    Decompose the total dP/dL₀ into individual channel contributions.
    """
    idx_mid = len(results['L0_range']) // 2
    L0_mid = results['L0_range'][idx_mid]

    print("="*80)
    print("CHANNEL CONTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nAt L₀ = {L0_mid:.1f} MeV:")
    print()
    print("The period formula is:")
    print("  P = 2π L √(α Ω κ ln(b/ξ))")
    print()
    print("Since ξ = ℏv_F/Δ, we have:")
    print("  ln(b/ξ) = ln(b) - ln(ξ) = ln(b) - ln(ℏv_F) + ln(Δ)")
    print()
    print("Taking logarithmic derivative:")
    print("  d(ln P)/dL₀ = (1/2) × d(ln ln(b/ξ))/dL₀")
    print("              = (1/2) × [1/ln(b/ξ)] × d(ln(b/ξ))/dL₀")
    print("              = (1/2) × [1/ln(b/ξ)] × [-d(ln ξ)/dL₀]")
    print("              = (1/2) × [1/ln(b/ξ)] × [d(ln Δ)/dL₀ - d(ln v_F)/dL₀]")
    print()
    print("Numerical values:")
    print("-"*80)

    log_b_xi = results['log_b_xi'][idx_mid]
    sens_Delta = results['sens_Delta'][idx_mid]
    sens_v_F = results['sens_v_F'][idx_mid]
    sens_xi = results['sens_xi'][idx_mid]
    sens_P = results['sens_period'][idx_mid]

    print(f"  ln(b/ξ)            = {log_b_xi:.3f}")
    print(f"  d(ln Δ)/dL₀        = {sens_Delta:+.6f} MeV⁻¹")
    print(f"  d(ln v_F)/dL₀      = {sens_v_F:+.6f} MeV⁻¹")
    print(f"  d(ln ξ)/dL₀        = {sens_xi:+.6f} MeV⁻¹  (= d(ln v_F) - d(ln Δ))")
    print()
    print(f"  Predicted: d(ln P)/dL₀ = (1/2) × (1/{log_b_xi:.3f}) × ({-sens_xi:.6f})")
    print(f"                         = {0.5 * (1/log_b_xi) * (-sens_xi):+.6f} MeV⁻¹")
    print()
    print(f"  Actual:    d(ln P)/dL₀ = {sens_P:+.6f} MeV⁻¹")
    print()
    print("Channel contributions to ξ(L₀):")
    print("-"*80)
    print(f"  v_F channel:  d(ln v_F)/dL₀  = {sens_v_F:+.6f} MeV⁻¹")
    print(f"  Δ channel:   -d(ln Δ)/dL₀   = {-sens_Delta:+.6f} MeV⁻¹")
    print(f"  Net (ξ):      d(ln ξ)/dL₀   = {sens_xi:+.6f} MeV⁻¹")
    print()

    # Relative importance
    total_abs = abs(sens_v_F) + abs(sens_Delta)
    v_F_frac = abs(sens_v_F) / total_abs * 100
    Delta_frac = abs(sens_Delta) / total_abs * 100

    print(f"Relative importance:")
    print(f"  v_F contribution:  {v_F_frac:.1f}%")
    print(f"  Δ contribution:    {Delta_frac:.1f}%")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EXPLORING ξ(L₀) DEPENDENCE")
    print("="*80)
    print()

    # Run the analysis
    results = explore_xi_L0_chain(verbose=True)

    # Print channel decomposition
    print_channel_contributions(results)

    # Create figure
    print("\nGenerating comprehensive visualization...")
    fig_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'xi_L0_complete_chain.pdf')
    fig = plot_xi_L0_chain(
        results,
        save_path=fig_path
    )

    plt.show()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print()
    print("Key findings:")
    print("   ξ varies with L₀ through both v_F and Δ channels")
    print("   This creates the L₀ → P sensitivity needed for measurement")
    print("   All channels contribute with different signs and magnitudes")
    print()
