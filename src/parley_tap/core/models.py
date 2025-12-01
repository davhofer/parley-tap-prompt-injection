import argparse
import functools
import os
import typing as t

import openai
from .types import ChatFunction, Message, Parameters, Role
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def _chat_openai(
    client: OpenAI, messages: t.List[Message], parameters: Parameters
) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    # Build request parameters
    request_params = {
        "model": parameters.model,
        "messages": t.cast(t.List[ChatCompletionMessageParam], messages),
        "temperature": parameters.temperature,
        "top_p": parameters.top_p,
    }

    model_lower = parameters.model.lower()

    # Check if this is a reasoning model that has parameter restrictions
    is_reasoning_model = any(x in model_lower for x in ["gpt-5", "o1-", "o3-"])

    # Newer OpenAI models require max_completion_tokens instead of max_tokens
    if any(x in model_lower for x in ["gpt-5", "o1", "o3"]):  # "gpt-4o",
        request_params["max_completion_tokens"] = parameters.max_tokens
    else:
        request_params["max_tokens"] = parameters.max_tokens

    # Reasoning models (gpt-5, o1, o3) only support temperature=1.0
    # Remove temperature param for these models to use default
    if is_reasoning_model:
        del request_params["temperature"]
        del request_params["top_p"]

    # Add tools if provided
    if parameters.tools:
        request_params["tools"] = [tool.model_dump() for tool in parameters.tools]
        if parameters.tool_choice:
            request_params["tool_choice"] = parameters.tool_choice

    response = client.chat.completions.create(**request_params)

    response_message = response.choices[0].message

    # Debug: print full response info when content is empty
    if not response_message.content:
        print(
            f"    [DEBUG] Empty response - finish_reason: {response.choices[0].finish_reason}, refusal: {getattr(response_message, 'refusal', None)}"
        )

    message = Message(
        role=Role(response_message.role),
        content=str(response_message.content) if response_message.content else "",
    )

    # If the response contains tool calls, return them alongside the message
    if hasattr(response_message, "tool_calls") and response_message.tool_calls:
        tool_calls = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_message.tool_calls
        ]
        return message, tool_calls

    return message


def chat_openai(
    messages: t.List[Message], parameters: Parameters
) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    return _chat_openai(OpenAI(), messages, parameters)


def chat_together(
    messages: t.List[Message], parameters: Parameters
) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

    return _chat_openai(client, messages, parameters)


def chat_vllm(
    messages: t.List[Message], parameters: Parameters
) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    """Chat with a local vLLM server using OpenAI-compatible API."""
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get(
        "VLLM_API_KEY", "dummy"
    )  # vLLM doesn't require auth by default

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    return _chat_openai(client, messages, parameters)


# Model registry mapping model names to (function, model_id) tuples
Models: t.Dict[str, t.Tuple] = {
    "gpt-3.5": (chat_openai, "gpt-3.5-turbo"),
    "gpt-4": (chat_openai, "gpt-4"),
    "gpt-4-turbo": (chat_openai, "gpt-4-1106-preview"),
    "gpt-4o": (chat_openai, "gpt-4o"),
    "gpt-4o-mini": (chat_openai, "gpt-4o-mini"),
    "gpt-4o-mini-2024-07-18": (
        chat_openai,
        "gpt-4o-mini-2024-07-18",
    ),  # Specific version
    "gpt-5": (chat_openai, "gpt-5-2025-08-07"),
    "gpt-5.1": (chat_openai, "gpt-5.1-2025-11-13"),
    "gpt-5-mini": (chat_openai, "gpt-5-mini-2025-08-07"),
    "llama-13b": (chat_together, "togethercomputer/llama-2-13b-chat"),
    "llama-70b": (chat_together, "togethercomputer/llama-2-70b-chat"),
    "vicuna-13b": (chat_together, "lmsys/vicuna-13b-v1.5"),
    "mistral-small-together": (chat_together, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    # Local vLLM models
    "qwen3-4b": (chat_vllm, "Qwen/Qwen3-4B-Instruct-2507"),
}


def load_models(
    args: argparse.Namespace,
) -> t.Tuple[ChatFunction, ChatFunction, ChatFunction]:
    """
    Load and configure target, evaluator, and attacker models.

    Args:
        args: Namespace with model configuration including:
            - target_model, evaluator_model, attacker_model: Model names from Models dict
            - *_temp, *_top_p, *_max_tokens: Model parameters

    Returns:
        Tuple of (target_chat, evaluator_chat, attacker_chat) functions
    """
    target_func, target_model = Models[args.target_model]
    target_chat = t.cast(
        ChatFunction,
        functools.partial(
            target_func,
            parameters=Parameters(
                model=target_model,
                temperature=args.target_temp,
                top_p=args.target_top_p,
                max_tokens=args.target_max_tokens,
            ),
        ),
    )

    evaluator_func, evaluator_model = Models[args.evaluator_model]
    evaluator_chat = t.cast(
        ChatFunction,
        functools.partial(
            evaluator_func,
            parameters=Parameters(
                model=evaluator_model,
                temperature=args.evaluator_temp,
                top_p=args.evaluator_top_p,
                max_tokens=args.evaluator_max_tokens,
            ),
        ),
    )

    attacker_func, attacker_model = Models[args.attacker_model]
    attacker_chat = t.cast(
        ChatFunction,
        functools.partial(
            attacker_func,
            parameters=Parameters(
                model=attacker_model,
                temperature=args.attacker_temp,
                top_p=args.attacker_top_p,
                max_tokens=args.attacker_max_tokens,
            ),
        ),
    )

    return target_chat, evaluator_chat, attacker_chat
