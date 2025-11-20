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
        "max_tokens": parameters.max_tokens,
        "top_p": parameters.top_p,
    }

    # Add tools if provided
    if parameters.tools:
        request_params["tools"] = [tool.model_dump() for tool in parameters.tools]
        if parameters.tool_choice:
            request_params["tool_choice"] = parameters.tool_choice

    response = client.chat.completions.create(**request_params)

    response_message = response.choices[0].message
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
    "llama-13b": (chat_together, "togethercomputer/llama-2-13b-chat"),
    "llama-70b": (chat_together, "togethercomputer/llama-2-70b-chat"),
    "vicuna-13b": (chat_together, "lmsys/vicuna-13b-v1.5"),
    "mistral-small-together": (chat_together, "mistralai/Mixtral-8x7B-Instruct-v0.1"),
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
