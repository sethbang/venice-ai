#!/usr/bin/env python3
"""
Venice AI SDK - Chat File Inputs
================================

Venice supports an OpenAI-compatible ``type: file`` content part: you attach a
document to a user message and the server extracts its text for the model to
read. Supported formats include PDF, EPUB, DOCX, PPTX, XLSX, txt, md, csv,
json, and most source-code files.

Two ways to supply the file via ``UserMessage.builder().file(...)``:

1. **Inline (``data:`` URL)** — base64-encode the bytes into a
   ``data:<mime>;base64,<...>`` URL. Best for local files. This example uses
   this path so it runs without any external hosting.
2. **Public URL** — pass an ``https://...`` URL the API can fetch. Shown in
   ``file_from_public_url()`` (skipped at runtime unless you supply one).

We prove the model actually read the attachment by embedding a distinctive fact
in the document and asking a question only answerable from it.
"""

import asyncio
import base64
import sys

from venice_ai import VeniceClient
from venice_ai.types.api import SystemMessage, UserMessage

# A small Markdown document with a fact the model can't know otherwise.
_DOC_MARKDOWN = """\
# Internal Field Note — Project Halcyon

- Project codename: **Halcyon**
- Lead engineer: Marisol Okonkwo
- Launch window: the third Tuesday of November 2027
- Secret build token: HX-4417-ZULU

This note is confidential and exists only inside this document.
"""


def _markdown_data_url(markdown: str) -> str:
    """Encode markdown text as a ``data:`` URL for the file content part."""
    b64 = base64.b64encode(markdown.encode("utf-8")).decode("ascii")
    return f"data:text/markdown;base64,{b64}"


async def file_from_data_url() -> None:
    """Attach a local document as a base64 ``data:`` URL and query it."""
    print("📎 File input via data: URL")
    print("-" * 40)

    data_url = _markdown_data_url(_DOC_MARKDOWN)

    # builder() lets you mix a file part and a text question in one message.
    user_message = (
        UserMessage.builder()
        .file(data_url, filename="field_note.md")
        .text("From the attached note, what is the secret build token?")
        .build()
    )

    async with VeniceClient() as client:
        response = await client.chat.completions.create(
            model=await client.models.resolve_chat(),
            messages=[
                SystemMessage(content="Answer using only the attached document."),
                user_message,
            ],
            max_completion_tokens=200,
            temperature=0.0,
        )

    answer = response.text or ""
    print("   Q: secret build token?")
    print(f"   A: {answer.strip()}")

    # Verify the model genuinely read the file rather than hallucinating.
    if "HX-4417-ZULU" in answer:
        print("   ✅ Model extracted the token from the attached file.")
    else:
        raise RuntimeError(
            "Model did not return the token from the attached file — "
            f"file extraction may have failed. Got: {answer!r}"
        )


async def file_from_public_url(url: str | None = None) -> None:
    """Attach a document by public URL (skipped unless a URL is provided)."""
    print("\n🌐 File input via public URL")
    print("-" * 40)

    if not url:
        print("   ℹ️ Skipped — pass a public document URL to run this path:")
        print("      UserMessage.builder().file('https://example.com/report.pdf')")
        return

    user_message = (
        UserMessage.builder()
        .file(url)
        .text("Summarize the attached document in two sentences.")
        .build()
    )

    async with VeniceClient() as client:
        response = await client.chat.completions.create(
            model=await client.models.resolve_chat(),
            messages=[user_message],
            max_completion_tokens=300,
        )
    print(f"   Summary: {(response.text or '').strip()}")


async def main() -> None:
    """Run the chat file-input examples."""
    print("🚀 Venice AI Chat File Inputs")
    print("=" * 50)

    await file_from_data_url()
    await file_from_public_url()

    print("\n✨ File input examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - UserMessage.builder().file(data_url, filename=...)")
    print("   - Inline base64 data: URLs vs. public URLs")
    print("   - Mixing a file part with a text question in one message")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
