"""Prompt templates for prompt injection attacks."""

from .injection_prompts import (
    get_prompt_for_injection_attacker,
    get_prompt_for_injection_evaluator_score,
    get_prompt_for_injection_evaluator_relevance,
    get_prompt_for_injection_target,
    build_injection_context_prompt,
)

__all__ = [
    "get_prompt_for_injection_attacker",
    "get_prompt_for_injection_evaluator_score",
    "get_prompt_for_injection_evaluator_relevance",
    "get_prompt_for_injection_target",
    "build_injection_context_prompt",
]
