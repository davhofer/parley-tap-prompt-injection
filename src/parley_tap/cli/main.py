#!/usr/bin/env python3
"""
Command-line interface for Parley TAP prompt injection attacks.
"""

import argparse
import sys
from pathlib import Path

from ..attacks.injection_parley import InjectionAttackFrameworkImpl, load_training_examples
from ..core.models import load_models
from ..core.injection_types import (
    InjectionConfig,
    ToolCallMatch,
    AggregationStrategy,
)


def main():
    """Main CLI entry point for Parley TAP."""
    parser = argparse.ArgumentParser(
        description="Parley TAP: Tree of Attacks for Prompt Injection"
    )

    parser.add_argument(
        "training_file",
        help="JSON file containing training examples",
    )

    # Model selection
    parser.add_argument(
        "--target-model",
        default="gpt-4o-mini",
        choices=["gpt-3.5", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
                 "llama-13b", "llama-70b", "vicuna-13b", "mistral-small-together"],
        help="Target model to attack (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--evaluator-model",
        default="gpt-4o-mini",
        choices=["gpt-3.5", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
                 "llama-13b", "llama-70b", "vicuna-13b", "mistral-small-together"],
        help="Evaluator model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--attacker-model",
        default="gpt-4o",
        choices=["gpt-3.5", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
                 "llama-13b", "llama-70b", "vicuna-13b", "mistral-small-together"],
        help="Attacker model (default: gpt-4o)",
    )

    # Model parameters
    parser.add_argument("--target-temp", type=float, default=0.0, help="Target temperature")
    parser.add_argument("--target-top-p", type=float, default=0.0, help="Target top-p")
    parser.add_argument("--target-max-tokens", type=int, default=1024, help="Target max tokens")

    parser.add_argument("--evaluator-temp", type=float, default=0.0, help="Evaluator temperature")
    parser.add_argument("--evaluator-top-p", type=float, default=0.0, help="Evaluator top-p")
    parser.add_argument("--evaluator-max-tokens", type=int, default=10, help="Evaluator max tokens")

    parser.add_argument("--attacker-temp", type=float, default=1.0, help="Attacker temperature")
    parser.add_argument("--attacker-top-p", type=float, default=1.0, help="Attacker top-p")
    parser.add_argument("--attacker-max-tokens", type=int, default=1024, help="Attacker max tokens")

    # Attack parameters
    parser.add_argument("--root-nodes", type=int, default=3, help="Number of root nodes")
    parser.add_argument("--branching-factor", type=int, default=3, help="Branching factor")
    parser.add_argument("--width", type=int, default=10, help="Search width")
    parser.add_argument("--depth", type=int, default=10, help="Search depth")
    parser.add_argument("--success-threshold", type=float, default=0.8, help="Success threshold")

    # Tool matching strategy
    parser.add_argument(
        "--tool-match",
        default="substring",
        choices=["exact", "json", "substring", "semantic"],
        help="Tool call matching strategy",
    )

    # Aggregation strategy
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "min"],
        help="Score aggregation strategy",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Check if training file exists
    if not Path(args.training_file).exists():
        print(f"Error: Training file '{args.training_file}' not found")
        sys.exit(1)

    try:
        # Load models
        target_chat, evaluator_chat, attacker_chat = load_models(args)

        # Load training examples
        training_examples = load_training_examples(args.training_file, debug=args.debug)

        # Create configuration
        config = InjectionConfig(
            training_examples=training_examples,
            tool_match_strategy=ToolCallMatch(args.tool_match),
            aggregation_strategy=AggregationStrategy(args.aggregation),
            success_threshold=args.success_threshold,
            root_nodes=args.root_nodes,
            branching_factor=args.branching_factor,
            width=args.width,
            depth=args.depth,
            debug=args.debug,
        )

        # Create and run framework
        framework = InjectionAttackFrameworkImpl(
            config, target_chat, evaluator_chat, attacker_chat
        )

        print("Starting Parley TAP attack...")
        print(f"Training examples: {len(training_examples)}")
        print(f"Target model: {args.target_model}")
        print(f"Attacker model: {args.attacker_model}")
        print(f"Evaluator model: {args.evaluator_model}")
        print()

        # Run the attack
        best_injection = framework.run_attack()

        print("\n" + "="*50)
        print("ATTACK COMPLETED")
        print("="*50)
        print(f"Best injection found: {best_injection}")

    except Exception as e:
        print(f"Error running attack: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()