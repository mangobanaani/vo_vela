# Data Directory

This directory contains observational data for the vortex oscillation L0 measurement.

## Contents

### 1. Vela Pulsar Glitch Oscillations

**File:** `vela/glitches/vela_glitch_oscillations.csv`

Post-glitch oscillation periods measured from Vela pulsar glitches:
- **G1** (2016-12-12): P = 314.1 ± 0.2 days (5.8 sigma detection)
- **G3a** (2021-09-17): P = 344.0 ± 1.5 days (3.2 sigma detection)
- **G3b** (2021-09-17): P = 153.0 ± 2.0 days (2.9 sigma detection)

**Source:** Grover et al. (2025), arXiv:2506.02100

### 2. Vela Pulsar System Parameters

**File:** `vela/vela_parameters.txt`

Complete parameter set for Vela pulsar including:
- Timing parameters (P, nu, nu_dot)
- Physical parameters (M, R, T, distance)
- Glitch history
- Rotation and moment of inertia
- Superfluid parameters (pairing gap, critical temperature)
- Vortex properties (spacing, coherence length)

**Sources:** ATNF Pulsar Catalogue, NICER constraints, literature compilation

### 3. Literature L0 Values

**File:** `comparison_data/L0_literature_values.csv`

Compilation of L0 measurements from other methods for comparison:
- Heavy-ion collisions (2012-2023)
- GW170817 neutron star merger (2017-2018)
- NICER X-ray observations (2019-2021)
- Chiral effective field theory
- Nuclear experiments (PREX, dipole polarizability, etc.)
- This work (vortex oscillations, 2025)

**Total entries:** 16 independent measurements

## Data Format

### Glitch Oscillations (CSV)

```csv
Glitch,Date_MJD,Date_Cal,Delta_Omega_over_Omega,Period_days,Period_err_days,Amplitude_Hz,Significance_sigma,Duration_days
G1,57736,2016-12-12,1.6e-6,314.1,0.2,2.3e-9,5.8,120
```

**Columns:**
- `Glitch`: Glitch identifier
- `Date_MJD`: Modified Julian Date of glitch
- `Date_Cal`: Calendar date (YYYY-MM-DD)
- `Delta_Omega_over_Omega`: Fractional spin-up (ΔΩ/Ω)
- `Period_days`: Oscillation period (days)
- `Period_err_days`: 1-sigma uncertainty (days)
- `Amplitude_Hz`: Oscillation amplitude in frequency
- `Significance_sigma`: Detection significance (σ)
- `Duration_days`: Oscillation visibility duration (days)

### Literature L0 Values (CSV)

```csv
Method,L0_MeV,Error_minus_MeV,Error_plus_MeV,Reference,Year,Notes
Heavy-ion collisions,58.7,6.0,6.0,Tsang et al. PRC,2012,Au+Au and Sn+Sn collisions
```

**Columns:**
- `Method`: Experimental/observational technique
- `L0_MeV`: Central value (MeV)
- `Error_minus_MeV`: Lower uncertainty (MeV)
- `Error_plus_MeV`: Upper uncertainty (MeV)
- `Reference`: Primary citation
- `Year`: Publication year
- `Notes`: Additional information

## Loading Data

Use the provided Python module to load data:

```python
from data.load_data import load_vela_glitches, load_vela_parameters, load_L0_literature

# Load glitch oscillation data
glitches = load_vela_glitches()
print(glitches)

# Load Vela parameters
params = load_vela_parameters()
print(f"Vela mass: {params['Mass_Msun']} solar masses")
print(f"Vela radius: {params['Radius_km']} km")

# Load literature comparison data
literature = load_L0_literature()
print(literature[['Method', 'L0_MeV', 'Error_minus_MeV']])

# Get specific glitch data
from data.load_data import get_glitch_data
g1 = get_glitch_data('G1')
print(f"G1 period: {g1['period']:.1f} ± {g1['period_err']:.1f} days")
```

Or run the module directly for a summary:

```bash
cd data
python load_data.py
```

## Data Quality

### Glitch G1 (Primary Measurement)
- **Period precision:** 0.2 days (0.06% relative)
- **Detection significance:** 5.8 sigma (high confidence)
- **Duration:** 120 days (well-sampled)
- **Used for:** Main L0 constraint

### Glitches G3a, G3b (Consistency Checks)
- **Period precision:** 1.5-2.0 days (0.4-1.3% relative)
- **Detection significance:** 2.9-3.2 sigma (marginal)
- **Duration:** 60-80 days
- **Used for:** Multi-glitch joint constraint and validation

## References

### Observational Data

**Grover et al. (2025)**
"Post-glitch oscillations in the Vela pulsar"
arXiv:2506.02100
*Source of oscillation periods and glitch parameters*

**ATNF Pulsar Catalogue**
Manchester et al. (2005), AJ 129, 1993
https://www.atnf.csiro.au/research/pulsar/psrcat/
*Source of timing parameters and basic properties*

### Comparison Data

**Heavy-ion collisions:**
- Tsang et al. (2012), PRC 86, 015803
- Zhang & Chen (2023), PRC 108, 024317

**GW170817:**
- Abbott et al. (2017), PRL 119, 161101
- Most et al. (2018), PRL 120, 261103

**NICER:**
- Riley et al. (2019), ApJL 887, L21
- Miller et al. (2021), ApJL 918, L28

## Citation

If you use this data, please cite:

**For oscillation data:**
```bibtex
@article{grover2025vela,
  author = {V. Grover and others},
  title = {Post-glitch oscillations in the Vela pulsar},
  journal = {arXiv preprint},
  year = {2025},
  eprint = {2506.02100},
  archivePrefix = {arXiv}
}
```

**For this analysis:**
```bibtex
@article{siltala2025_L0,
  author = {Pekka Siltala},
  title = {Nuclear Symmetry Energy Measurement from Vortex Oscillations
           in Pulsar Glitches},
  year = {2025},
  institution = {Aalto University},
  note = {Manuscript in preparation}
}
```

## Data Availability Statement

The Vela pulsar timing data used to extract oscillation periods are publicly available from:
- Fermi-LAT: https://fermi.gsfc.nasa.gov/ssc/data/access/lat/ephems/
- Analysis products: https://github.com/mangobanaani/vo_vela

## Contact

**Pekka Siltala**
Email: pekka.siltala@aalto.fi
Institution: Aalto University, Finland

---

**Last updated:** November 5, 2025
