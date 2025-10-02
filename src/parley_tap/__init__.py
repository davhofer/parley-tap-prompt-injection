"""
Parley TAP: Tree of Attacks for Prompt Injection

A framework for generating and testing prompt injection attacks against AI agents
using the Tree of Attacks (TAP) methodology.
"""

__version__ = "0.1.0"

from .attacks.injection_parley import InjectionAttackFrameworkImpl, load_training_examples
from .core.injection_types import (
    InjectionConfig,
    TrainingExample,
    ToolCallMatch,
    AggregationStrategy,
)

__all__ = [
    "InjectionAttackFrameworkImpl",
    "load_training_examples",
    "InjectionConfig",
    "TrainingExample",
    "ToolCallMatch",
    "AggregationStrategy",
]