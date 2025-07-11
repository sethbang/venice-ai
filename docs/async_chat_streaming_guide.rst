.. _async_chat_streaming_guide:

Asynchronous Chat Streaming Guide
=================================

This guide demonstrates how to use the `~venice_ai.AsyncVeniceClient` to stream chat completions asynchronously. Streaming allows you to receive parts of the response as they are generated, which can be useful for interactive applications.

Prerequisites
-------------

- Ensure you have the `venice-ai` package installed.
- Set your `VENICE_API_KEY` environment variable.

Core Concepts
-------------

- **AsyncVeniceClient**: The primary client for asynchronous interactions.
- **client.chat.completions.create()**: The method used to generate chat responses.
- **stream=True**: Parameter to enable streaming.
- **AsyncIterator**: The streaming response is an async iterator that yields `ChatCompletionChunk` objects.

Example: Streaming a Chat Response
-----------------------------------

The following example shows how to send a prompt and stream the response:

.. code-block:: python
   :linenos:

   import os
   import asyncio
   from venice_ai import AsyncVeniceClient, APIError

   async def main():
       """Demonstrates asynchronous chat streaming with Venice AI."""
       try:
           # Ensure VENICE_API_KEY is set in your environment
           if not os.getenv("VENICE_API_KEY"):
               print("Error: VENICE_API_KEY environment variable not set.")
               return

           async with AsyncVeniceClient() as client:
               print("Streaming chat response from Venice AI...")
               print("---")
               
               prompt_messages = [
                   {"role": "user", "content": "Tell me a short story about a brave knight and a friendly dragon."}
               ]
               
               # Using a common model, replace if needed
               model_id = "llama-3.2-3b"

               stream = await client.chat.completions.create(
                   model=model_id,
                   messages=prompt_messages,
                   stream=True,
                   max_completion_tokens=150 # Optional: limit response length
               )
               
               full_response = []
               async for chunk in stream:
                   # Ensure 'choices' and 'delta' are present and valid
                   if chunk.choices and len(chunk.choices) > 0:
                       delta = chunk.choices[0].delta
                       if delta and delta.content:
                           content_delta = delta.content
                           print(content_delta, end="", flush=True)
                           full_response.append(content_delta)
               
               print("\n---\nStream finished.")
               # print(f"Full assembled response: {''.join(full_response)}")

       except APIError as e:
           print(f"\nAn API Error occurred: {e.status_code} - {e.message}")
           if e.body:
               print(f"Error details: {e.body}")
       except Exception as e:
           print(f"\nAn unexpected error occurred: {e}")

   if __name__ == "__main__":
       asyncio.run(main())

Explanation
-----------

1.  **Import necessary modules**: `os` for environment variables, `asyncio` for running async code, and `~venice_ai.AsyncVeniceClient`, `APIError` from `venice_ai`.
2.  **Define an async main function**: All asynchronous operations happen within this function.
3.  **Initialize AsyncVeniceClient**: The client is instantiated within an `async with` block to ensure proper resource management (automatic `aclose()` on exit).
4.  **Define a prompt**: A simple user message is created.
5.  **Call `create` with `stream=True`**:
    -   `model`: Specify a valid chat model ID.
    -   `messages`: Provide the list of messages.
    -   `stream=True`: This is crucial for enabling streaming.
6.  **Iterate over the stream**: The `async for chunk in stream:` loop processes each `ChatCompletionChunk` as it arrives.
7.  **Extract content**: The `content_delta` is extracted from `chunk['choices'][0]['delta'].get('content', '')`.
8.  **Print content**: The partial content is printed immediately. `flush=True` ensures it's displayed without buffering.
9.  **Error Handling**: Includes `try...except` blocks to catch potential `APIError` or other exceptions.

Running the Example
-------------------

Save the code as a Python file (e.g., `run_async_stream.py`) and run it from your terminal:

.. code-block:: bash

   python run_async_stream.py

You should see the story about the brave knight printed to your console, word by word, as it's generated by the AI.

Key Takeaways
-------------

- Asynchronous streaming is ideal for applications requiring real-time feedback.
- Always use `async with AsyncVeniceClient(...)` for proper client lifecycle management.
- Handle chunks carefully, as they represent partial updates to the overall message.
- Implement robust error handling for API interactions.

Further Exploration
-------------------

- Explore other parameters of the `create` method to customize model behavior (e.g., `temperature`, `max_completion_tokens`).
- When using models with larger context windows like "Venice Large" (128k tokens), you can adjust `max_completion_tokens` accordingly to leverage the full context for longer responses or more detailed analysis.
- Integrate this streaming logic into a web application or a command-line interface for a more interactive experience.
- Refer to the :ref:`API Reference <api_reference>` for detailed information on all available methods and parameters.

Synchronous Streaming
~~~~~~~~~~~~~~~~~~~~~

While this guide focuses on asynchronous streaming with `AsyncVeniceClient`,
the library also supports synchronous streaming for chat completions using
the `VeniceClient`.

You can find an example of synchronous streaming in the main docstring for the
:class:`~venice_ai._client.VeniceClient` class within the API Reference.