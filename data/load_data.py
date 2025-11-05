#!/usr/bin/env python3
"""
Load Vela pulsar data for vortex oscillation analysis.

This module provides functions to load observational data used in the
L0 measurement from post-glitch oscillations.

Author: Pekka Siltala
Institution: Aalto University
Date: November 5, 2025
"""

import os
import numpy as np
import pandas as pd


def get_data_dir():
    """Get the data directory path."""
    return os.path.dirname(os.path.abspath(__file__))


def load_vela_glitches():
    """
    Load Vela glitch oscillation data.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns:
        - Glitch: str, glitch identifier (G1, G3a, G3b)
        - Date_MJD: float, Modified Julian Date
        - Date_Cal: str, calendar date
        - Delta_Omega_over_Omega: float, fractional spin-up
        - Period_days: float, oscillation period (days)
        - Period_err_days: float, uncertainty in period (days)
        - Amplitude_Hz: float, oscillation amplitude
        - Significance_sigma: float, detection significance
        - Duration_days: float, oscillation visibility duration

    Source:
    -------
    Grover et al. (2025), arXiv:2506.02100
    """
    data_file = os.path.join(get_data_dir(), 'vela', 'glitches',
                             'vela_glitch_oscillations.csv')

    df = pd.read_csv(data_file, comment='#')
    return df


def load_vela_parameters():
    """
    Load Vela pulsar system parameters.

    Returns:
    --------
    params : dict
        Dictionary of Vela pulsar parameters including:
        - Timing parameters (P, nu, nu_dot)
        - Physical parameters (M, R, T)
        - Rotation parameters (Omega, I)
        - Superfluid parameters (Delta, Tc)

    Source:
    -------
    ATNF Pulsar Catalogue + literature compilation
    """
    params_file = os.path.join(get_data_dir(), 'vela', 'vela_parameters.txt')

    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.split('#')[0].strip()  # Remove inline comments

            # Try to convert to float if possible
            try:
                value = float(value)
            except ValueError:
                # Keep as string if not numeric
                pass

            params[key] = value

    return params


def load_L0_literature():
    """
    Load L0 values from literature for comparison.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns:
        - Method: str, measurement method
        - L0_MeV: float, L0 value (MeV)
        - Error_minus_MeV: float, lower uncertainty
        - Error_plus_MeV: float, upper uncertainty
        - Reference: str, citation
        - Year: int, publication year
        - Notes: str, additional information

    Sources:
    --------
    Compiled from multiple publications (2010-2025)
    """
    lit_file = os.path.join(get_data_dir(), 'comparison_data',
                            'L0_literature_values.csv')

    df = pd.read_csv(lit_file, comment='#')
    return df


def get_glitch_data(glitch_id='G1'):
    """
    Get data for a specific glitch.

    Parameters:
    -----------
    glitch_id : str
        Glitch identifier ('G1', 'G3a', or 'G3b')

    Returns:
    --------
    data : dict
        Dictionary with glitch parameters:
        - period: float, oscillation period (days)
        - period_err: float, uncertainty (days)
        - mjd: float, glitch date
        - delta_omega: float, fractional spin-up
        - amplitude: float, oscillation amplitude (Hz)
        - significance: float, detection significance (sigma)
    """
    df = load_vela_glitches()
    row = df[df['Glitch'] == glitch_id]

    if len(row) == 0:
        raise ValueError(f"Glitch {glitch_id} not found. "
                        f"Available: {df['Glitch'].values}")

    row = row.iloc[0]

    data = {
        'period': row['Period_days'],
        'period_err': row['Period_err_days'],
        'mjd': row['Date_MJD'],
        'delta_omega': row['Delta_Omega_over_Omega'],
        'amplitude': row['Amplitude_Hz'],
        'significance': row['Significance_sigma']
    }

    return data


def print_data_summary():
    """Print summary of available data."""
    print("=" * 70)
    print("VELA PULSAR DATA SUMMARY")
    print("=" * 70)
    print()

    # Glitches
    print("Glitch Oscillations:")
    print("-" * 70)
    df = load_vela_glitches()
    print(df.to_string(index=False))
    print()

    # Parameters
    print("Pulsar Parameters:")
    print("-" * 70)
    params = load_vela_parameters()
    important_keys = ['P_spin_ms', 'nu_Hz', 'Mass_Msun', 'Radius_km',
                      'Temperature_K', 'Distance_kpc']
    for key in important_keys:
        if key in params:
            print(f"  {key:20s} = {params[key]}")
    print()

    # Literature comparison
    print("Literature L0 Values (selected):")
    print("-" * 70)
    lit = load_L0_literature()
    lit_subset = lit[lit['Method'].str.contains('Heavy-ion|GW170817|NICER|Vortex')]
    print(lit_subset[['Method', 'L0_MeV', 'Error_minus_MeV', 'Year']].to_string(index=False))
    print()

    print("=" * 70)


if __name__ == "__main__":
    # Demonstrate data loading
    print_data_summary()

    # Example: Get G1 data
    print("\nExample: Loading G1 data for analysis")
    print("-" * 70)
    g1_data = get_glitch_data('G1')
    print(f"G1 period: {g1_data['period']:.1f} ± {g1_data['period_err']:.1f} days")
    print(f"G1 significance: {g1_data['significance']:.1f} sigma")
    print(f"G1 fractional spin-up: {g1_data['delta_omega']:.2e}")
