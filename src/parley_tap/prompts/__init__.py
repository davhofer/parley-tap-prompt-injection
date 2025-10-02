"""Prompt templates for attacks and evaluations."""

from .prompts import (
    get_prompt_for_evaluator_score,
    get_prompt_for_evaluator_on_topic,
    get_prompt_for_attacker,
    get_prompt_for_target,
)
from .injection_prompts import (
    get_prompt_for_injection_attacker,
    get_prompt_for_injection_evaluator_score,
    get_prompt_for_injection_evaluator_relevance,
    get_prompt_for_injection_target,
    build_injection_context_prompt,
)

__all__ = [
    "get_prompt_for_evaluator_score",
    "get_prompt_for_evaluator_on_topic",
    "get_prompt_for_attacker",
    "get_prompt_for_target",
    "get_prompt_for_injection_attacker",
    "get_prompt_for_injection_evaluator_score",
    "get_prompt_for_injection_evaluator_relevance",
    "get_prompt_for_injection_target",
    "build_injection_context_prompt",
]