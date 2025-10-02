# Parley

This is a minimal implementation of the "Tree of Attacks (TAP): Jailbreaking Black-Box LLMs Automatically" Research by Robust Intelligence.

[Using AI to Automatically Jailbreak GPT-4 and Other LLMs in Under a Minute](https://www.robustintelligence.com/blog-posts/using-ai-to-automatically-jailbreak-gpt-4-and-other-llms-in-under-a-minute)

# Design

- [x] Clean, expand, and restructure all the system prompts
- [x] Use API-based model calling via OpenAI, TogetherAI, and Mistral
- [x] Refactor the tree/leaf branching for simplicity
- [ ] Implement max conversation history to stay within attacker context window
- [ ] Add WandB logging for history tracking
- [ ] Add support for local models  

We've leveraged the OpenAI, Mistral, and TogetherAI APIs to implement support for the following models:

- `gpt-3.5`
- `gpt-4`
- `gpt-4-turbo`
- `llama-13b`
- `llama-70b`
- `vicuna-13b`
- `mistral-small-together`
- `mistral-small`
- `mistral-medium`

You can configure these models using the `--target-*`, `--evaluator-*`, and `--atacker-*` arguments.

# Installation

## From GitHub

```bash
pip install git+https://github.com/yourusername/parley-tap-prompt-injection.git
```

## From source

```bash
git clone https://github.com/yourusername/parley-tap-prompt-injection.git
cd parley-tap-prompt-injection
uv pip install -e .
```

# Usage

## Prerequisites

Configure required API keys based on the models you'd like to use:
```bash
export OPENAI_API_KEY='...'
export TOGETHER_API_KEY='...'
export MISTRAL_API_KEY='...'
```

## Command Line Interface

```bash
# Basic usage with training file
parley-tap examples/example_training_data.json

# Advanced usage with custom models and parameters
parley-tap examples/example_training_data.json \
  --target-model gpt-4o-mini \
  --attacker-model gpt-4o \
  --width 5 --depth 3 --debug
```

## Python API

```python
from parley_tap import InjectionAttackFrameworkImpl, InjectionConfig
from parley_tap.attacks import load_training_examples, load_models

# Load training examples and models
training_examples = load_training_examples("examples/example_training_data.json")
target_chat, evaluator_chat, attacker_chat = load_models(args)

# Configure and run attack
config = InjectionConfig(training_examples=training_examples)
framework = InjectionAttackFrameworkImpl(config, target_chat, evaluator_chat, attacker_chat)
best_injection = framework.run_attack()
```

# Docstring

```
Parley: A Tree of Attacks (TAP) LLM Jailbreaking Implementation

positional arguments:
  goal                  Goal of the conversation (use 'extract' for context extraction mode)

options:
  -h, --help            show this help message and exit
  --target-model {gpt-3.5,gpt-4,gpt-4-turbo,llama-13b,llama-70b,vicuna-13b,mistral-small-together,mistral-small,mistral-medium}
                        Target model (default: gpt-4-turbo)
  --target-temp TARGET_TEMP
                        Target temperature (default: 0.3)
  --target-top-p TARGET_TOP_P
                        Target top-p (default: 1.0)
  --target-max-tokens TARGET_MAX_TOKENS
                        Target max tokens (default: 1024)
  --evaluator-model {gpt-3.5,gpt-4,gpt-4-turbo,llama-13b,llama-70b,vicuna-13b,mistral-small-together,mistral-small,mistral-medium}
                        Evaluator model (default: gpt-4-turbo)
  --evaluator-temp EVALUATOR_TEMP
                        Evaluator temperature (default: 0.5)
  --evaluator-top-p EVALUATOR_TOP_P
                        Evaluator top-p (default: 0.1)
  --evaluator-max-tokens EVALUATOR_MAX_TOKENS
                        Evaluator max tokens (default: 10)
  --attacker-model {gpt-3.5,gpt-4,gpt-4-turbo,llama-13b,llama-70b,vicuna-13b,mistral-small-together,mistral-small,mistral-medium}
                        Attacker model (default: mistral-small)
  --attacker-temp ATTACKER_TEMP
                        Attacker temperature (default: 1.0)
  --attacker-top-p ATTACKER_TOP_P
                        Attacker top-p (default: 1.0)
  --attacker-max-tokens ATTACKER_MAX_TOKENS
                        Attacker max tokens (default: 1024)
  --root-nodes ROOT_NODES
                        Tree of thought root node count (default: 3)
  --branching-factor BRANCHING_FACTOR
                        Tree of thought branching factor (default: 3)
  --width WIDTH         Tree of thought width (default: 10)
  --depth DEPTH         Tree of thought depth (default: 10)
  --stop-score STOP_SCORE
                        Stop when the score is above this value (default: 8.0)
```
