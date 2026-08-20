# Streaming chat completions

Sourced from `src/venice_ai/streaming.py` and `src/venice_ai/resources/chat/completions.py`. The v2 streaming API is intentionally typed and lifecycle-managed — the bare `async for chunk in stream` pattern from OpenAI/v1 still works but bypasses the v2 idiom.

## The canonical shape

```python
import asyncio, sys
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage, SystemMessage, StreamOptions


async def main() -> None:
    async with VeniceClient() as client:
        stream = await client.chat.completions.stream(
            model=await client.models.resolve_chat(),
            messages=[
                SystemMessage(content="You are concise."),
                UserMessage(content="Tell me a story about a robot."),
            ],
            stream_options=StreamOptions(include_usage=True),    # see "Two prerequisites" below
            max_completion_tokens=300,
            temperature=0.7,
        )

        async with stream:                                       # mandatory — guarantees cleanup
            async for delta in stream.collect_with_deltas():     # see "Three iteration modes" below
                sys.stdout.write(delta)
                sys.stdout.flush()
            sys.stdout.write("\n")

            if stream.final_response and stream.final_response.usage:
                u = stream.final_response.usage
                print(f"[usage] prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")
```

## Three iteration modes — the critical distinction

`ChatStream` (returned from `client.chat.completions.stream(...)`) exposes three iteration helpers. **The choice matters** because they differ in whether `final_response` gets populated.

| Method | Yields live deltas? | Populates `stream.final_response`? | When to use |
|---|---|---|---|
| `stream.text_deltas()` | yes | **NO** | "I just want to print to stdout" — no usage / final response needed. Lightest weight. |
| `stream.collect_with_deltas()` | yes | yes | The general-purpose choice. Live deltas AND access to the assembled response after iteration. |
| `await stream.collect()` | no | yes (returns it directly) | "Just give me the final response" — equivalent to a non-streaming `create()` but bills the same. Rarely the right call. |

If you reach for `text_deltas()` and then try to read `stream.final_response.usage`, you get `None`. This is the most common streaming bug.

## Two prerequisites for `final_response.usage` to be populated

1. **`stream_options=StreamOptions(include_usage=True)`** on the `stream(...)` call. Without this, the server doesn't emit a final usage chunk, and `final_response.usage` stays `None` even with `collect_with_deltas()`.
2. **`collect_with_deltas()` or `collect()`** — `text_deltas()` doesn't store the final response, so even if the usage is in the stream, you can't access it.

Both are required.

## `async with stream:` is mandatory

`ChatStream` is an async context manager. Use `async with stream:` to guarantee the underlying HTTP response is closed even on early exit / exception. Bare `async for chunk in stream` over the raw `ChatStream` object works syntactically but leaks the connection if iteration breaks early.

```python
# WRONG — leaks connection on exception or break
stream = await client.chat.completions.stream(...)
async for delta in stream.text_deltas():
    if delta == "STOP":
        break

# RIGHT
stream = await client.chat.completions.stream(...)
async with stream:
    async for delta in stream.text_deltas():
        if delta == "STOP":
            break
```

## Underlying chunks (advanced)

If you want low-level access to the raw `ChatCompletionChunk` objects (e.g., to inspect tool-call deltas, finish reasons, or per-choice content), iterate `stream` directly inside the `async with`:

```python
async with stream:
    async for chunk in stream:                  # AsyncIterator[ChatCompletionChunk]
        for choice in chunk.choices or []:
            if choice.delta.tool_calls:
                ...                              # streaming tool calls
            if choice.delta.content:
                sys.stdout.write(choice.delta.content)
        if chunk.usage:                          # populated on the final chunk only
            print(f"[chunk usage] {chunk.usage}")
```

This gives you everything `collect_with_deltas` does internally; use it when the higher-level helpers don't fit.

## OpenAI-style `create(stream=True)` — works, but bypasses the idiom

The SDK accepts `client.chat.completions.create(model=..., messages=..., stream=True, ...)` and returns an iterable of chunks. People often reach for this pattern by default (it's the OpenAI muscle memory).

It works. But:
- You lose `text_deltas()` / `collect_with_deltas()` / `collect()` helpers.
- `stream.final_response` doesn't exist on the bare iterator — you have to assemble it yourself.
- The `async with stream:` cleanup is harder to wire (the iterator from `create(stream=True)` isn't itself a context manager).

Prefer `client.chat.completions.stream(...)` + `async with stream:` for new code.

## Concurrent streams

Multiple streams can run concurrently — wrap each in its own `async with`:

```python
async def stream_one(client, prompt: str) -> tuple[str, int]:
    full = []
    stream = await client.chat.completions.stream(
        model=await client.models.resolve_chat(),
        messages=[UserMessage(content=prompt)],
        stream_options=StreamOptions(include_usage=True),
    )
    async with stream:
        async for delta in stream.collect_with_deltas():
            full.append(delta)
        usage = stream.final_response.usage if stream.final_response else None
    return "".join(full), usage.total_tokens if usage else 0


results = await client.gather(
    [stream_one(client, p) for p in prompts],
    max_concurrency=3,
)
```

`client.gather(..., max_concurrency=N)` is the right primitive — bare `asyncio.gather(*coros)` will fan out unbounded.

## Partial-failure recovery

A stream may fail mid-iteration (server-side timeout, connection drop). The SDK raises the typed error (e.g., `APIConnectionError`) at the `async for` line; the `async with` block ensures the response is closed. Retry from the beginning — there's no in-stream resume in this API.

```python
from venice_ai.exceptions import APIConnectionError, APITimeoutError

async def stream_with_retries(client, **kwargs) -> str:
    for attempt in range(3):
        try:
            stream = await client.chat.completions.stream(**kwargs)
            text = []
            async with stream:
                async for delta in stream.text_deltas():
                    text.append(delta)
            return "".join(text)
        except (APIConnectionError, APITimeoutError):
            if attempt == 2:
                raise
            await asyncio.sleep(2 ** attempt)
```

For animated rendering (typewriter effect, etc.), see `examples/chat/streaming_chat.py` in the SDK.

## Streaming with tool calls

`run_with_tools` doesn't currently expose a streaming variant — the agent loop is fundamentally request/response. If you need to stream the *terminal* assistant message after the loop converges, you can hand-roll the loop using `create()` for the tool rounds and `stream()` for the final round:

```python
# pseudocode — see references/tool-loops.md for the manual loop
async def stream_after_tools(...) -> str:
    messages = [...]
    while True:
        resp = await client.chat.completions.create(model=m, messages=messages, tools=tools)
        msg = resp.choices[0].message
        if msg.tool_calls is None:
            # Final round — switch to streaming for output
            stream = await client.chat.completions.stream(model=m, messages=messages)
            async with stream:
                async for delta in stream.text_deltas():
                    sys.stdout.write(delta)
            return
        messages.append(msg)
        for call in msg.tool_calls:
            ...                             # dispatch + append ToolMessage
```

## Streaming structured output

You can stream a structured-output call via `create(stream=True, response_format=Cls)` and assemble the result, but `parse()` (the auto-validating sibling of `create()`) **explicitly does not support streaming** — it raises `ValueError` if you pass `stream=True`. For typed structured output you generally want the full response anyway; streaming structured output is a niche use case.

## Common bugs

- **Using `text_deltas()` and reading `final_response.usage`** → `None`. Use `collect_with_deltas()`.
- **Forgetting `StreamOptions(include_usage=True)`** → `final_response.usage` is `None` even with `collect_with_deltas()`.
- **Bare `async for chunk in stream:`** without `async with stream:` → connection leaks on early exit.
- **OpenAI-style `create(stream=True) + async for chunk:`** → works but bypasses the v2 idioms; harder to surface usage.
- **Trying to iterate the same stream twice** → raises `StreamConsumedError`.
- **Catching `StreamClosedError` for post-`async with` access** → it never fires; the SDK doesn't raise `StreamClosedError`. Re-iterating a consumed stream raises `StreamConsumedError` instead.

## Related references

- `tool-loops.md` — agent loops; streaming the terminal turn.
- `headers-and-metadata.md` — `stream.final_response.headers`, `.response_rate_limits`, `.deprecation_info`, `.balance_info`.
- `migration-v1-to-v2.md` — `ChatStream` and the streaming idioms are new shape in v2.
