#!/usr/bin/env python3
"""
ChemRefine: Automated conformer sampling and refinement using ORCA.

An automated and interoperable manager for computational chemistry workflows.
"""

__version__ = "1.3.1"
__author__ = "Sterling Group"
__email__ = "dal950773@utdallas.edu"


# ------------------------
# Public data structures
# ------------------------

from .types import (
    EngineConfig,
    SamplingConfig,
    StepContext,
    StepResult,
    PipelineState,
)


# ------------------------
# Public main interface
# ------------------------

from .core import ChemRefiner


# ------------------------
# Public API
# ------------------------

__all__ = [
    # Main class
    "ChemRefiner",

    # Pipeline types
    "EngineConfig",
    "SamplingConfig",
    "StepContext",
    "StepResult",
    "PipelineState",
]
