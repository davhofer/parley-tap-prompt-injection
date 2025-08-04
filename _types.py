import typing as t
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class Message(BaseModel):
    role: Role
    content: str


class Feedback(BaseModel):
    prompt: str
    improvement: str


class TreeNode(BaseModel):
    children: t.List["TreeNode"]
    conversation: t.List[Message]
    feedback: t.Optional[Feedback]
    response: t.Optional[str]
    on_topic: t.Optional[bool]
    score: t.Optional[int]


class ToolFunction(BaseModel):
    """Function definition for OpenAI-style tool/function calling."""
    name: str
    description: str
    parameters: t.Dict[str, t.Any]  # JSON Schema for parameters


class Tool(BaseModel):
    """Tool definition for OpenAI-style tool calling."""
    type: str = "function"
    function: ToolFunction


class Parameters(BaseModel):
    model: str
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.9
    tools: t.Optional[t.List[Tool]] = None  # For tool/function calling
    tool_choice: t.Optional[t.Union[str, t.Dict[str, t.Any]]] = None  # "auto", "none", or specific tool


ChatFunction = t.Callable[[t.List[Message]], t.Union[Message, t.Tuple[Message, t.List[t.Dict[str, t.Any]]]]]
Conversation = t.List[Message]
