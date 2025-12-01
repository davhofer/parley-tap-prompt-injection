import json
import typing as t

from parley_tap.core.types import Message


def format_conversation(messages: t.List[Message]) -> str:
    """Format conversation in a compact style with full tool call details.

    Args:
        messages: List of Message objects to format

    Returns:
        Formatted string with one line per message, including role, content,
        and tool calls with full arguments/parameters

    Example:
        >>> messages = [Message(role=Role.user, content="Hello")]
        >>> print(format_conversation_compact(messages))
        USER: Hello
    """
    lines = []

    for msg in messages:
        role = msg.role.value.upper()

        # Format tool calls with full details
        if msg.tool_calls:
            tool_details = []
            for tc in msg.tool_calls:
                # Extract function name and arguments from different formats
                func_name = tc.get("function", {}).get(
                    "name", tc.get("name", "unknown")
                )
                func_args = tc.get("function", {}).get(
                    "arguments", tc.get("arguments", {})
                )

                # Format arguments nicely
                if isinstance(func_args, str):
                    # If arguments are JSON string, keep as is
                    args_str = func_args
                elif isinstance(func_args, dict):
                    # If dict, convert to compact JSON
                    args_str = json.dumps(func_args, separators=(",", ":"))
                else:
                    args_str = str(func_args)

                tool_details.append(f"{func_name}({args_str})")

            tool_summary = f" [tool calls: {'; '.join(tool_details)}]"
        else:
            tool_summary = ""

        # Format content
        content = msg.content or "<no content>"
        if msg.tool_call_id:
            content = f"[response to tool call] {content}"

        lines.append(f"{role}: {content}{tool_summary}")

    return "\n".join(lines)
