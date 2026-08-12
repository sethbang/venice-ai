"""
Venice AI Helper Utilities
=========================

Convenience helpers for common patterns when working with the Venice AI SDK.

- :func:`tool_from_model` -- create a :class:`Tool` from a Pydantic model
- :func:`tool_from_function` -- create a :class:`Tool` from a typed Python function
- :class:`Conversation` -- thin wrapper for building multi-turn message lists
- :func:`cosine_similarity` -- score two embedding vectors in [-1, 1]
- :func:`detect_image_format` -- sniff (extension, mime_type) from raw image bytes
- :func:`fit_image_bytes` -- resize an image to fit within a max-dimension box
- :func:`extract_thinking_blocks` -- parse ``<thinking>`` / ``<think>`` tags
- :func:`normalize_duration_seconds` -- parse ``5`` / ``"5"`` / ``"5s"`` / ``"5 seconds"`` to int
"""

from __future__ import annotations

import inspect
import io
import math
import re
import types
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .core.models.common import Tool, ToolFunction
from .types.api.requests.chat import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from .types.api.requests.common import (
    MessageContentPartParam,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from ._client import VeniceClient
    from .types.api.chat import ChatCompletionResponse, ToolCall, ToolLoopResult
    from .types.api.requests.common import Tool

__all__ = [
    "tool_from_model",
    "tool_from_function",
    "Conversation",
    "cosine_similarity",
    "detect_image_format",
    "extract_thinking_blocks",
    "fit_image_bytes",
    "normalize_duration_seconds",
]


# ---------------------------------------------------------------------------
# Thinking-block extraction
# ---------------------------------------------------------------------------


_THINKING_PATTERNS = (r"<thinking>(.*?)</thinking>", r"<think>(.*?)</think>")


def extract_thinking_blocks(content: str | list[Any]) -> tuple[list[str], str]:
    """Extract reasoning/thinking blocks from a response's content.

    Some Venice models surface chain-of-thought reasoning inline in the
    assistant's ``content`` wrapped in ``<thinking>...</thinking>`` or
    ``<think>...</think>`` tags. This helper splits those out so callers can
    display them separately from the final answer.

    For models that put reasoning in the dedicated ``reasoning_content``
    field instead, prefer :attr:`ChatCompletionResponse.thinking_blocks`,
    which checks both server shapes.

    :param content: ``message.content`` value — either a string or a list
        of multimodal parts (the list form is stringified before parsing).
    :return: ``(blocks, cleaned_content)`` — a list of the extracted block
        contents, and the original ``content`` with the matched tags
        removed and trailing whitespace stripped.
    """
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)

    blocks: list[str] = []
    cleaned = content

    for pattern in _THINKING_PATTERNS:
        found = re.findall(pattern, cleaned, re.DOTALL)
        if found:
            blocks.extend(found)
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL)

    return blocks, cleaned.strip()


# ---------------------------------------------------------------------------
# Image format detection
# ---------------------------------------------------------------------------


def detect_image_format(data: bytes) -> tuple[str, str]:
    """Sniff image format from magic bytes.

    Returns ``(extension, mime_type)`` for supported formats, e.g.
    ``("png", "image/png")``. For unrecognized bytes returns
    ``("bin", "application/octet-stream")``.

    Recognizes JPEG, PNG, WebP, and GIF — the formats Venice models
    return. Use this when you need to save an image to disk with the
    correct extension or assemble a ``data:`` URI from raw bytes.

    :param data: Raw image bytes (typically the result of
        ``base64.b64decode(response.images[0])``).
    """
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg", "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"RIFF") and b"WEBP" in data[:20]:
        return "webp", "image/webp"
    if data.startswith(b"GIF8"):
        return "gif", "image/gif"
    return "bin", "application/octet-stream"


def fit_image_bytes(
    data: bytes,
    *,
    max_dim: int = 1024,
    quality: int = 85,
) -> bytes:
    """Resize an image so neither side exceeds ``max_dim`` pixels.

    Returns *data* unchanged when the image already fits. Otherwise scales
    down preserving aspect ratio and re-encodes as JPEG.

    Why this matters: some Venice vision models accept smaller maximum input
    dimensions than others. ``venice-uncensored-1-2`` and
    ``venice-uncensored-role-play`` return an opaque HTTP 500
    (``"Inference processing failed"``) when handed multi-megapixel inputs
    that other vision models accept fine. Pre-resizing client-side sidesteps
    that. Defaults of ``max_dim=1024`` / ``quality=85`` produce ~150 KB
    photographs that every Venice vision model accepts.

    Pillow is a hard SDK dependency, so no extras gating.

    :param data: Raw image bytes (PNG, JPEG, WebP, or GIF).
    :param max_dim: Maximum width or height in pixels.
    :param quality: JPEG quality (1-100).
    :return: Resized JPEG bytes, or *data* unchanged if already within
        ``max_dim``.
    """
    from PIL import Image

    src = Image.open(io.BytesIO(data))
    width, height = src.size
    if max(width, height) <= max_dim:
        return data
    src.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    # Use a distinct name post-convert: Image.open() returns ImageFile, but
    # .convert() returns the Image.Image base — separate names keep mypy
    # happy without an explicit annotation.
    img: Image.Image = src.convert("RGB") if src.mode != "RGB" else src
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Vector similarity
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Returns a value in ``[-1.0, 1.0]``: ``1.0`` for identical direction,
    ``0.0`` for orthogonal, ``-1.0`` for opposite. Pure Python — no numpy
    dependency — so it works on the raw ``embedding`` lists returned by
    :meth:`client.embeddings.create`.

    :raises ValueError: If *a* and *b* have different lengths or are empty,
        or if either vector has zero magnitude (cosine is undefined).
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have the same length: got {len(a)} and {len(b)}")
    if not a:
        raise ValueError("Vectors must be non-empty")

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


_DURATION_PATTERN = re.compile(
    r"""
    ^\s*               # optional leading whitespace
    (?P<value>\d+)     # one or more digits
    \s*                # optional whitespace before unit
    (?:s|sec|secs|second|seconds)?  # optional unit suffix (case-insensitive)
    \s*$               # optional trailing whitespace
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_duration_seconds(value: int | str) -> int:
    """Parse ``5`` / ``"5"`` / ``"5s"`` / ``"5 seconds"`` to an integer.

    Liberal in what we accept, strict in what we return — image / music /
    video resources all coerce duration values through this helper before
    validating against per-model enums and before sending the request.

    Examples:
        >>> normalize_duration_seconds(5)
        5
        >>> normalize_duration_seconds("5")
        5
        >>> normalize_duration_seconds("5s")
        5
        >>> normalize_duration_seconds("5 SECONDS")
        5

    :raises ValueError: If *value* cannot be parsed as a positive integer
        number of seconds.
    """
    if isinstance(value, bool):
        # bool is a subclass of int — refuse it explicitly so True doesn't
        # silently become 1 second.
        raise ValueError("duration_seconds cannot be a bool")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"duration_seconds must be positive, got {value}")
        return value
    if isinstance(value, str):
        match = _DURATION_PATTERN.match(value)
        if match is None:
            raise ValueError(
                f"Could not parse duration_seconds={value!r}; expected an int "
                f"or a string like '5', '5s', or '5 seconds'."
            )
        parsed = int(match.group("value"))
        if parsed <= 0:
            raise ValueError(f"duration_seconds must be positive, got {parsed}")
        return parsed
    raise ValueError(f"duration_seconds must be int or str, got {type(value).__name__}: {value!r}")


# ---------------------------------------------------------------------------
# Type hint → JSON Schema mapping
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[type, dict[str, str]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
}


def _python_type_to_json_schema(hint: type) -> dict[str, Any]:
    """Convert a Python type hint to JSON Schema.

    Supports: ``str``, ``int``, ``float``, ``bool``, ``list[T]``,
    ``dict[str, V]``, ``Optional[T]`` / ``T | None``, ``Literal[...]``.

    Both ``typing.Optional[int]`` and the PEP 604 form ``int | None`` are
    accepted — they're treated identically. Multi-member unions like
    ``Union[A, B]`` / ``A | B`` (non-Optional) are not supported; use
    :func:`tool_from_model` with a Pydantic discriminated union for those.

    Bare ``list`` / ``dict`` (without type arguments) are rejected — pass
    ``list[str]`` / ``dict[str, int]`` so the schema can describe items.

    :raises TypeError: For unsupported type hints.
    """
    origin = get_origin(hint)
    args = get_args(hint)

    # Simple types — but reject bare ``list`` / ``dict`` (no type args, no items info)
    if hint in _TYPE_MAP:
        if hint is list:
            raise TypeError(
                "Bare list is not supported — use list[T] so the schema can describe item types."
            )
        if hint is dict:
            raise TypeError(
                "Bare dict is not supported — use dict[str, V] so the schema can describe value types."
            )
        return dict(_TYPE_MAP[hint])

    # Union[...] / PEP 604 ``T | None`` handling — origin is ``typing.Union`` for the
    # ``typing`` form, ``types.UnionType`` for the PEP 604 form. Treat both alike.
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[T] / T | None — defer required-ness to the caller
            return _python_type_to_json_schema(non_none[0])
        raise TypeError(
            f"Unions with multiple non-None members are not supported "
            f"({hint!r}). Use tool_from_model() with a Pydantic discriminated union."
        )

    # list[T]
    if origin is list and args:
        return {"type": "array", "items": _python_type_to_json_schema(args[0])}

    # dict[str, V]
    if origin is dict and args and len(args) == 2:
        return {"type": "object", "additionalProperties": _python_type_to_json_schema(args[1])}

    # Literal["a", "b"]
    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    raise TypeError(
        f"Unsupported type {hint} for tool_from_function(). "
        f"Supported: str, int, float, bool, list[T], dict[str, V], "
        f"Optional[T] / T | None, Literal[...]. "
        f"For complex types, use tool_from_model() with a Pydantic BaseModel."
    )


# ---------------------------------------------------------------------------
# tool_from_model
# ---------------------------------------------------------------------------


def tool_from_model(
    model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Create a :class:`Tool` definition from a Pydantic ``BaseModel`` subclass.

    Uses Pydantic's :meth:`~pydantic.BaseModel.model_json_schema` to generate
    the JSON Schema for the function parameters.

    :param model: A Pydantic ``BaseModel`` subclass.
    :param name: Override the function name (defaults to the model class name).
    :param description: Override the description (defaults to the model docstring).
    :return: A :class:`Tool` ready to pass to ``tools=[...]``.
    """
    return Tool(
        type="function",
        function=ToolFunction(
            name=name or model.__name__,
            description=description or model.__doc__ or "",
            parameters=model.model_json_schema(),
            strict=False,
        ),
        id=None,
    )


# ---------------------------------------------------------------------------
# tool_from_function
# ---------------------------------------------------------------------------


def tool_from_function(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """Create a :class:`Tool` definition from a Python function's type hints.

    Inspects the function signature and type annotations to build a JSON
    Schema for the ``parameters`` field.  Only a subset of types is supported
    (see :func:`_python_type_to_json_schema`); for richer schemas, use
    :func:`tool_from_model` with a Pydantic model.

    :param fn: The function to introspect.
    :param name: Override the tool name (defaults to ``fn.__name__``).
    :param description: Override the description (defaults to ``fn``'s docstring).
    :return: A :class:`Tool` ready to pass to ``tools=[...]``.
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        hint = hints.get(param_name, str)
        properties[param_name] = _python_type_to_json_schema(hint)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return Tool(
        type="function",
        function=ToolFunction(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn) or "",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            strict=False,
        ),
        id=None,
    )


# ---------------------------------------------------------------------------
# Conversation helper
# ---------------------------------------------------------------------------


class Conversation:
    """Convenience helper for building multi-turn message lists.

    Maintains an ordered list of chat messages and provides chainable
    methods for appending user turns, assistant responses, and tool results.

    For production use cases requiring token management, persistence,
    or conversation branching, manage messages directly.

    Example::

        conv = Conversation(system="You are a helpful assistant.")
        conv.add_user("What's the weather?")
        response = await client.chat.completions.create(
            model=model, messages=conv.messages,
        )
        conv.add_response(response)
        conv.add_user("And tomorrow?")
    """

    def __init__(self, *, system: str | None = None) -> None:
        self._messages: list[UserMessage | AssistantMessage | SystemMessage | ToolMessage] = []
        if system:
            self._messages.append(SystemMessage(content=system))

    @property
    def messages(self) -> list[UserMessage | AssistantMessage | SystemMessage | ToolMessage]:
        """Return a shallow copy of the message list."""
        return list(self._messages)

    def add_user(
        self,
        content: str | list[MessageContentPartParam],
    ) -> Conversation:
        """Append a user message and return ``self`` for chaining.

        Accepts either a plain string or a list of multimodal content parts.
        Each part can be a typed object (:class:`TextContent`,
        :class:`ImageContent`, :class:`AudioContent`, :class:`VideoContent`)
        or a plain ``dict`` matching one of the corresponding TypedDict
        shapes (e.g. ``{"type": "text", "text": "hi"}``).
        """
        self._messages.append(UserMessage(content=content))
        return self

    def add_response(self, response: ChatCompletionResponse, choice_index: int = 0) -> Conversation:
        """Append an assistant message extracted from a completion response."""
        self._messages.append(AssistantMessage.from_response(response, choice_index))
        return self

    def add_assistant_message(
        self,
        content: str | None = None,
        *,
        tool_calls: list[ToolCall] | None = None,
    ) -> Conversation:
        """Append an assistant message directly.

        Useful in agent loops where you want to inject an assistant turn —
        often a tool-call turn — without first wrapping it in a
        :class:`ChatCompletionResponse`.
        """
        self._messages.append(AssistantMessage(content=content, tool_calls=tool_calls))
        return self

    def add_tool_result(self, tool_call_id: str, content: str) -> Conversation:
        """Append a tool result message."""
        self._messages.append(ToolMessage(tool_call_id=tool_call_id, content=content))
        return self

    async def run_with_tools(
        self,
        client: VeniceClient,
        *,
        model: str,
        tools: Sequence[Callable[..., Any] | Tool],
        on_tool_call: Callable[[ToolCall, Any], None] | None = None,
        on_tool_error: Callable[[ToolCall, Exception], str] | None = None,
        parallel: bool = False,
        max_iterations: int = 10,
        **create_kwargs: Any,
    ) -> ToolLoopResult:
        """Run :meth:`ChatCompletions.run_with_tools` against this conversation.

        Thin wrapper that takes the conversation's current messages as input,
        runs the tool-orchestration loop, and **appends** every new message
        the loop produced (assistant tool-call turns, tool results, and the
        final assistant turn) to this conversation. The conversation is
        left ready for the next user turn.

        :param client: The Venice client to drive completions through.
        :param model: Model id to use for every iteration.
        :param tools: See :meth:`ChatCompletions.run_with_tools`.
        :param on_tool_call: See :meth:`ChatCompletions.run_with_tools`.
        :param on_tool_error: See :meth:`ChatCompletions.run_with_tools`.
        :param parallel: See :meth:`ChatCompletions.run_with_tools`.
        :param max_iterations: See :meth:`ChatCompletions.run_with_tools`.
        :param create_kwargs: Forwarded to ``chat.completions.create`` on
            every iteration.
        :return: The :class:`ToolLoopResult` from the underlying call.
            Note that ``result.messages`` is a separate copy — the
            conversation's own messages are mutated in place to reflect
            the same final history.
        """
        starting_len = len(self._messages)
        result = await client.chat.completions.run_with_tools(
            model=model,
            messages=self._messages,
            tools=tools,
            on_tool_call=on_tool_call,
            on_tool_error=on_tool_error,
            parallel=parallel,
            max_iterations=max_iterations,
            **create_kwargs,
        )
        # run_with_tools doesn't mutate the input `messages` — it returns a
        # fresh list. Transcribe the new tail (every turn the loop produced,
        # ending with the terminal assistant message) into this conversation.
        self._messages.extend(result.messages[starting_len:])
        return result
