"""Prompt templates for prompt injection attacks."""

from .injection_prompts import (
    get_prompt_for_injection_evaluator_score,
    get_prompt_for_injection_evaluator_relevance,
    get_attacker_system_prompt_v2,
    get_initial_context_prompt,
    get_feedback_prompt,
)

__all__ = [
    "get_prompt_for_injection_evaluator_score",
    "get_prompt_for_injection_evaluator_relevance",
    "get_attacker_system_prompt_v2",
    "get_initial_context_prompt",
    "get_feedback_prompt",
]
