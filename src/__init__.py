"""
Vortex Oscillation Spectroscopy: Constraining the Nuclear EoS from Pulsar Glitches

This package implements the complete framework for extracting equation of state
constraints from post-glitch vortex oscillations in neutron stars.

Modules:
--------
- eos: Nuclear equation of state models
- stellar_structure: TOV solver and density profiles
- superfluid: Pairing gaps, critical temperatures, superfluid densities
- vortex: Vortex line tension and oscillation frequencies
- timing: Pulsar timing data analysis and fitting
- inference: Bayesian inference for EoS parameters
- constants: Physical constants in CGS units
"""

from __future__ import annotations

import os
import sys


if sys.platform == "darwin":
    # Skip the NumPy Accelerate sanity check when running on macOS systems
    # where Apple's Accelerate BLAS may trigger spurious crashes.
    os.environ.setdefault("NUMPY_SKIP_MAC_OS_CHECK", "1")

__version__ = "0.1.0"
__author__ = "Your Name"

from . import constants

# Import modules as they are implemented
try:
    from . import eos
except ImportError:
    pass

try:
    from . import stellar_structure
except ImportError:
    pass

try:
    from . import superfluid
except ImportError:
    pass

try:
    from . import vortex
except ImportError:
    pass

try:
    from . import timing
except ImportError:
    pass

try:
    from . import inference
except ImportError:
    pass

__all__ = ["constants"]
