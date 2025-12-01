"""
Parley TAP: Tree of Attacks for Prompt Injection

A framework for generating and testing prompt injection attacks against AI agents
using the Tree of Attacks (TAP) methodology.
"""

__version__ = "0.1.0"

from .attacks.injection_parley import InjectionAttackFrameworkImpl
from .core.injection_types import (
    InjectionConfig,
    TrainingExample,
)

__all__ = [
    "InjectionAttackFrameworkImpl",
    "InjectionConfig",
    "TrainingExample",
]

