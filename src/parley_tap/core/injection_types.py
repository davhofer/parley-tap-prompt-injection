"""
Data structures for prompt injection attacks using Tree of Attacks approach.

This module extends the base TAP framework to support:
- Training on multiple conversation examples
- Targeting specific tool calls rather than general jailbreaks
- Evaluating injection strings across diverse contexts
"""

import typing as t
from enum import Enum
from pydantic import BaseModel

from .types import Message, Role, Tool


class TrialAggregationStrategy(str, Enum):
    """Methods for aggregating scores across reliability trials for a single sample."""

    BEST = "best"  # Take the best (max) score across trials
    MEAN = "mean"  # Take the average score across trials


class SampleAggregationStrategy(str, Enum):
    """Methods for aggregating scores across multiple samples (multi-sample mode)."""

    MEAN = "mean"  # Take the average score across all samples
    MIN = "min"  # Take the minimum score (all samples must succeed)
    SUCCESS_RATE = "success_rate"  # Percentage of samples above individual threshold


class TrainingExample(BaseModel):
    """
    A single training example for prompt injection attacks.

    Represents a conversation history with {optim_str} marking the injection point,
    along with the tools we want to trick the agent into calling.
    """

    example_id: str
    conversation_history: t.List[Message]  # Chat messages with {optim_str} placeholder
    target_tool_calls: t.List[
        str
    ]  # Full tool calls (as JSON strings) the attack should trigger
    attacker_goal: str  # High-level goal the attacker is trying to achieve
    available_tools: t.List[Tool]  # Tools available to the target model in this context
    user_task_context: t.Optional[str] = None  # The legitimate user task (to distinguish from attacker goal)


class InjectionFeedback(BaseModel):
    """
    Feedback from attacker model for injection string generation.

    Similar to original Feedback but focused on injection strings
    rather than full jailbreak prompts.

    For single-sample mode: uses injection_string directly.
    For multi-sample mode: uses prefix + attacker_goal + suffix to construct injection.
    """

    # Single-sample mode field
    injection_string: str = ""  # The actual injection content to insert (single-sample)

    # Multi-sample mode fields (PREFIX + goal + SUFFIX format)
    prefix: str = ""  # Text before the attacker goal
    suffix: str = ""  # Text after the attacker goal

    improvement: str  # Analysis of previous attempt and proposed changes

    @property
    def is_multi_sample(self) -> bool:
        """Check if this feedback is for multi-sample mode (has prefix/suffix)."""
        return bool(self.prefix) and not bool(self.injection_string)

    def get_injection_for_goal(self, attacker_goal: str) -> str:
        """
        Get the full injection string for a given attacker goal.

        For single-sample: returns injection_string directly.
        For multi-sample: returns prefix + attacker_goal + suffix.
        """
        if self.injection_string:
            return self.injection_string
        return f"{self.prefix}{attacker_goal}{self.suffix}"


class InjectionResult(BaseModel):
    """
    Result of testing an injection string against a single training example.

    Captures what tools were actually triggered and success metrics.
    """

    example_id: str
    injection_string: str
    triggered_tools: t.List[str]  # Tools actually called by the model
    raw_response: str  # Full model response for analysis
    success_score: float  # 0-1 score for this specific example (aggregated from trials)
    parsing_successful: bool  # Whether tool calls could be extracted
    execution_error: t.Optional[str] = None  # Any errors during evaluation

    # Reliability tracking
    trial_results: t.Optional[t.List[float]] = None  # Scores from individual trials


class AggregatedResult(BaseModel):
    """
    Aggregated results across all training examples for a single injection string.

    For multi-sample mode, also stores prefix and suffix separately.
    """

    injection_string: str  # For single-sample, or constructed string for display
    individual_results: t.List[InjectionResult]
    aggregated_score: float  # Final score after aggregation (mean across examples)
    success_rate: float  # Percentage of examples that succeeded
    total_examples: int

    # Multi-sample mode fields
    prefix: str = ""  # Prefix used (multi-sample mode)
    suffix: str = ""  # Suffix used (multi-sample mode)

    @property
    def is_multi_sample(self) -> bool:
        """Check if this result is from multi-sample mode."""
        return bool(self.prefix)


class InjectionTreeNode(BaseModel):
    """
    Tree node for injection attack search, extending the original TreeNode concept.

    Each node represents a state in the injection string optimization process.
    """

    children: t.List["InjectionTreeNode"]
    conversation: t.List[Message]  # Attacker conversation history
    feedback: t.Optional[InjectionFeedback]
    aggregated_result: t.Optional[AggregatedResult]  # Results across all examples
    on_topic: t.Optional[bool]  # Whether injection is relevant to examples

    @property
    def score(self) -> t.Optional[float]:
        """Get the aggregated score for this node."""
        return (
            self.aggregated_result.aggregated_score if self.aggregated_result else None
        )


class InjectionConfig(BaseModel):
    """
    Configuration for injection attack generation and evaluation.
    """

    # Training data
    training_examples: t.List[TrainingExample]

    # Evaluation settings
    success_threshold: float = 0.8  # Score threshold for stopping (aggregated score)

    # Multi-sample aggregation settings
    sample_aggregation_strategy: SampleAggregationStrategy = SampleAggregationStrategy.MEAN
    # For SUCCESS_RATE strategy:
    sample_success_threshold: float = 0.7  # Individual sample must score above this to count as "success"
    required_success_rate: float = 0.8  # Fraction of samples that must succeed (0.0-1.0)

    # Reliability testing parameters
    reliability_trials: int = 1  # Number of times to test each injection
    trial_aggregation_strategy: TrialAggregationStrategy = TrialAggregationStrategy.BEST  # How to aggregate trial scores

    # Injection constraints
    max_injection_length: int = 1000  # Maximum characters in injection string
    min_injection_length: int = 10  # Minimum characters in injection string
    preserve_conversation_flow: bool = False  # Should injection fit naturally

    # Tree search parameters (inherited from original TAP)
    root_nodes: int = 3
    branching_factor: int = 3
    width: int = 10
    depth: int = 10

    # Parallelization settings
    max_parallel_workers: int = 1  # Max concurrent workers for sample evaluation (1 = sequential)

    # Retry settings for API calls
    max_retries: int = 5  # Maximum number of retry attempts
    retry_initial_delay: float = 1.0  # Initial delay in seconds before first retry
    retry_max_delay: float = 60.0  # Maximum delay between retries
    retry_backoff_multiplier: float = 2.0  # Multiplier for exponential backoff

    # Debug flag
    debug: bool = False


class InjectionAttackFramework:
    """
    Main framework for running Tree of Attacks against prompt injection targets.

    This class orchestrates the entire attack process:
    1. Generate injection string candidates using attacker model
    2. Test each candidate against all training examples
    3. Aggregate results and prune search tree
    4. Repeat until success threshold is reached
    """

    def __init__(
        self, config: InjectionConfig, target_chat, evaluator_chat, attacker_chat
    ):
        """Initialize the framework with configuration and model interfaces."""
        self.config = config
        self.target_chat = target_chat  # Model being attacked
        self.evaluator_chat = evaluator_chat  # Model for scoring attacks
        self.attacker_chat = attacker_chat  # Model for generating attacks

    def generate_injection_candidates(
        self, node: InjectionTreeNode
    ) -> t.List[InjectionFeedback]:
        """
        Generate new injection string candidates from an existing node.

        Uses the attacker model to propose improved injection strings based on
        previous results and conversation context.
        """
        raise NotImplementedError()

    def evaluate_injection_string(self, injection_string: str) -> AggregatedResult:
        """
        Test an injection string against all training examples.

        For each example:
        1. Insert injection string at specified point in conversation
        2. Get target model response
        3. Extract and parse tool calls from response
        4. Score success based on target tool call matching
        5. Aggregate scores across all examples
        """
        raise NotImplementedError()

    def check_injection_relevance(self, injection_string: str) -> bool:
        """
        Check if an injection string is relevant to the training examples.

        Uses evaluator model to determine if the injection attempts to
        achieve the target tool calls in the given contexts.
        """
        raise NotImplementedError()

    def run_attack(self) -> t.Optional[str]:
        """
        Execute the full Tree of Attacks process for prompt injection.

        Returns the best injection string found, or None if no successful
        attack was discovered within the search parameters.
        """
        raise NotImplementedError()
