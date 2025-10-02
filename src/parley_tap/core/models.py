import os
import typing as t

import openai
from .types import Message, Parameters, Role
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
        content=str(response_message.content) if response_message.content else ""
    )
    
    # If the response contains tool calls, return them alongside the message
    if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
        tool_calls = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            }
            for tool_call in response_message.tool_calls
        ]
        return message, tool_calls
    
    return message


def chat_openai(messages: t.List[Message], parameters: Parameters) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    return _chat_openai(OpenAI(), messages, parameters)



def chat_together(messages: t.List[Message], parameters: Parameters) -> t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]:
    client = openai.OpenAI(
        api_key=os.environ["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1",
    )

    return _chat_openai(client, messages, parameters)
