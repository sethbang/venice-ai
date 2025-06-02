import asyncio
import base64
import io
import json
import tempfile

import streamlit as st
from PIL import Image as PILImage
import tiktoken
from pydantic import BaseModel

from venice_ai import AsyncVeniceClient
from venice_ai.exceptions import APIError, AuthenticationError
from venice_ai.types.audio import Voice, ResponseFormat
from typing import Any, Dict, List, Optional, Type, Union

import inspect

def run_async_in_streamlit(awaitable_or_coro_func):
    """
    Helper to run async functions or awaitables in Streamlit's synchronous environment.
    If the awaitable resolves to an async generator, it collects its items into a list.
    """
    # First, ensure we have an awaitable (coroutine or async generator object)
    # If it's a coroutine function, call it to get the coroutine
    # This part is tricky because client.chat.completions.create() is a method call,
    # so what's passed is already a coroutine object.

    # Let's assume `awaitable_or_coro_func` is typically a coroutine object.
    # Run the initial awaitable (which might be a coroutine that returns an async generator)
    result = asyncio.run(awaitable_or_coro_func)

    # Now, check if this result itself is an async generator
    if inspect.isasyncgen(result):
        # If it is, we need to run another async function to collect its items
        async def collect_async_gen_items(async_gen):
            return [item async for item in async_gen]
        return asyncio.run(collect_async_gen_items(result))
    else:
        # If the result is not an async generator, return it directly
        # (e.g., it's the final data from a non-streaming call)
        return result


def get_venice_client():
    """Retrieve or initialize the Venice AI client using the API key from session state."""
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        st.error("API Key not set. Please enter your Venice AI API Key in the sidebar.")
        return None
    try:
        return AsyncVeniceClient(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Error initializing client: {str(e)}")
        return None

# Set page configuration
st.set_page_config(page_title="Venice AI Streamlit Showcase", layout="wide")
st.title("Venice AI Streamlit Showcase")
st.markdown("Explore and test various functionalities of the Venice AI Python package.")

# Sidebar for API Key Management
with st.sidebar:
    st.header("Connection")
    api_key_input = st.text_input("Venice AI API Key", type="password", key="api_key_input_sidebar")
    if st.button("Connect / Update Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            st.success("API Key updated successfully.")
            # Quick test to validate the key
            try:
                client = AsyncVeniceClient(api_key=api_key_input)
                run_async_in_streamlit(client.models.list()) # Removed limit=1 as it's not supported
                st.success("API Key validated successfully.")
            except AuthenticationError:
                st.error("Invalid API Key. Please check and try again.")
            except Exception as e:
                st.warning(f"Could not validate API Key: {str(e)}")
        else:
            st.error("Please enter an API Key.")

if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.warning("API Key is required to use this application. Enter your Venice AI API Key in the sidebar.")
    st.stop()

# Tabbed Interface
tabs = st.tabs(["Chat", "Models", "Image", "Audio", "Embeddings", "Billing", "API Keys", "Utilities"])

# Chat Tab
with tabs[0]:  # Chat
    st.header("Chat Completions")
    client = get_venice_client()
    
    # Initialize chat history
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input widgets
    @st.cache_data
    def get_chat_models():
        if client:
            model_list = run_async_in_streamlit(client.models.list(type="chat"))
            # Handle dictionary response from API
            return [model['id'] for model in model_list['data']] if 'data' in model_list and model_list['data'] else ["No models available"]
        return ["No models available"]

    @st.cache_data
    def get_image_models():
        client = get_venice_client() # Ensure client is available
        if client:
            model_list = run_async_in_streamlit(client.models.list(type="image"))
            if 'data' in model_list and model_list['data']:
                return [model['id'] for model in model_list['data']]
            return ["No image models available"]
        return ["No image models available"]
    
    selected_model_chat = st.selectbox("Select Model", options=get_chat_models(), key="chat_model")
    stream_response_chat = st.checkbox("Stream Response", value=True, key="chat_stream_cb")
    
    with st.expander("Advanced Options"):
        temperature_chat = st.slider("Temperature", min_value=0.0, max_value=2.0, step=0.1, value=0.7, key="chat_temp")
        max_completion_tokens_chat = st.slider("Max Completion Tokens", min_value=1, max_value=8192, step=1, value=1024, key="chat_max_tokens")
        tools_json_chat = st.text_area("Tools (JSON)", value='', placeholder='[{"type": "function", "function": {"name": "get_weather", ...}}]', key="chat_tools")
        tool_choice_chat = st.radio("Tool Choice", options=["auto", "none", "required"], index=0, key="chat_tool_choice")
    
    if prompt := st.chat_input("Your message..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if client:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                api_params = {
                    "model": selected_model_chat,
                    "messages": st.session_state.chat_messages,
                    "temperature": temperature_chat,
                    "max_completion_tokens": max_completion_tokens_chat,
                    "stream": stream_response_chat
                }
                if tools_json_chat:
                    try:
                        tools = json.loads(tools_json_chat)
                        
                        # Validate tools schema according to API specifications
                        validation_error = None
                        if not isinstance(tools, list):
                            validation_error = "Tools must be a JSON array"
                        else:
                            for i, tool in enumerate(tools):
                                if not isinstance(tool, dict):
                                    validation_error = f"Tool at index {i} must be an object"
                                    break
                                
                                if "type" not in tool:
                                    validation_error = f"Tool at index {i} must have a 'type' field"
                                    break
                                    
                                if "function" not in tool:
                                    validation_error = f"Tool at index {i} must have a 'function' field"
                                    break
                                    
                                function = tool.get("function")
                                if not isinstance(function, dict):
                                    validation_error = f"Tool at index {i}: 'function' must be an object"
                                    break
                                    
                                if "name" not in function:
                                    validation_error = f"Tool at index {i}: 'function' object must have a 'name' field"
                                    break
                        
                        if validation_error:
                            st.error(f"Invalid tools structure: {validation_error}")
                        else:
                            api_params["tools"] = tools
                            api_params["tool_choice"] = tool_choice_chat
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON in Tools: {str(e)}")
                
                try:
                    if stream_response_chat:
                        full_response_content = ""
                        response_stream = run_async_in_streamlit(client.chat.completions.create(**api_params))
                        for chunk in response_stream: # chunk is a dict
                            content_delta = ""
                            choices = chunk.get('choices')
                            if choices and len(choices) > 0:
                                delta = choices[0].get('delta')
                                if delta:
                                    content_delta = delta.get('content', "")
                            
                            if content_delta:
                                full_response_content += content_delta
                                message_placeholder.markdown(full_response_content + "▌")
                        message_placeholder.markdown(full_response_content)
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_response_content})
                    else: # Non-streaming case
                        response = run_async_in_streamlit(client.chat.completions.create(**api_params)) # response is likely a dict
                        full_response_content = ""
                        tool_calls_content = ""
                        
                        response_choices = response.get('choices')
                        if response_choices and len(response_choices) > 0:
                            message = response_choices[0].get('message')
                            if message:
                                full_response_content = message.get('content', "")
                                
                                tool_calls = message.get('tool_calls')
                                if tool_calls:
                                    tool_calls_content = "\nTool Calls:\n"
                                    for tc in tool_calls: # tc is a dict
                                        function_call = tc.get('function')
                                        if function_call:
                                            tool_calls_content += f"- Function: {function_call.get('name')}, Args: {function_call.get('arguments')}\n"
                                            
                        full_assistant_response = (full_response_content + tool_calls_content).strip()
                        message_placeholder.markdown(full_assistant_response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": full_assistant_response})
                except AuthenticationError as e:
                    st.error(f"Authentication Error: {str(e)}. Please check your API Key.")
                except APIError as e:
                    if hasattr(e, 'status_code'):
                        if e.status_code == 429:
                            st.warning("Rate limit exceeded, please try again later.")
                        elif e.status_code == 503:
                            st.warning("Model is at capacity, please try again later.")
                        else:
                            st.error(f"API Error: {str(e)}")
                    else:
                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

# Models Tab
with tabs[1]:  # Models
    st.header("Models Information")
    model_type_filter = st.selectbox("Filter by Type", options=["all", "chat", "image", "embedding", "audio"], index=0, key="model_filter")
    if st.button("List Models", key="list_models_btn"):
        client = get_venice_client()
        if client:
            with st.spinner("Fetching models..."):
                try:
                    api_model_type_filter = model_type_filter if model_type_filter != "all" else None
                    model_list_obj = run_async_in_streamlit(client.models.list(type=api_model_type_filter))
                    processed_data = []
                    if model_list_obj and 'data' in model_list_obj and model_list_obj['data']:
                        for model_dict in model_list_obj['data']: # model_dict is a dictionary
                            # Attempt to get type from model_spec if not at top level
                            model_type = model_dict.get('type')
                            if not model_type and 'model_spec' in model_dict and isinstance(model_dict['model_spec'], dict):
                                # Basic inference for type based on capabilities, can be expanded
                                capabilities = model_dict['model_spec'].get('capabilities', {})
                                if capabilities.get('supportsVision', False): # Example for image
                                    model_type = "image"
                                elif not capabilities.get('optimizedForCode', True): # Assuming chat models are not optimized for code
                                    model_type = "chat"
                                # Add more inferences for 'embedding', 'audio' if possible from model_spec
                                else:
                                    model_type = "unknown" # Default if not easily inferred

                            processed_data.append({
                                "ID": model_dict.get('id', 'N/A'),
                                "Type": model_type if model_type else model_dict.get('type', 'N/A'), # Use inferred if top-level 'type' is missing
                                "Name": model_dict.get('name', model_dict.get('id', 'N/A')),
                                # Context length might be nested, e.g., in model_spec.availableContextTokens
                                "Context Length": model_dict.get('context_length', model_dict.get('model_spec', {}).get('availableContextTokens', 'N/A'))
                            })
                    if processed_data:
                        st.dataframe(processed_data, use_container_width=True)
                    else:
                        st.info("No models found or failed to retrieve data (after processing).")
                except APIError as e:
                    if hasattr(e, 'status_code'):
                        if e.status_code == 429:
                            st.warning("Rate limit exceeded, please try again later.")
                        elif e.status_code == 503:
                            st.warning("Model is at capacity, please try again later.")
                        else:
                            st.error(f"API Error: {str(e)}")
                    else:
                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error fetching models: {str(e)}")

# Image Tab
with tabs[2]:  # Image
    st.header("Image Generation & Upscaling")
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            image_model_options = get_image_models()
            current_image_model_index = 0
            # Handle empty or placeholder list to prevent index error
            if not image_model_options or image_model_options == ["No image models available"]:
                image_model_options = ["No image models available"] # Ensure it's a list with at least one item
                # Consider disabling the selectbox or showing a more prominent message if no models
            
            image_model = st.selectbox("Select Model", options=image_model_options, index=current_image_model_index, key="image_model")
            image_prompt = st.text_area("Prompt", placeholder="Enter your image generation prompt here...", key="image_prompt")
            image_width = st.slider("Width", min_value=256, max_value=2048, step=64, value=1024, key="image_width")
            image_height = st.slider("Height", min_value=256, max_value=2048, step=64, value=1024, key="image_height")
            image_steps = st.slider("Steps", min_value=10, max_value=30, step=1, value=20, key="image_steps")
            
            @st.cache_data
            def get_image_styles():
                client = get_venice_client()
                if client:
                    try:
                        styles_response = run_async_in_streamlit(client.image.list_styles())
                        if styles_response and 'data' in styles_response and styles_response['data']:
                            # Add "none" as first option if not already present
                            styles = ["none"] + [style for style in styles_response['data'] if style.lower() != "none"]
                            return styles
                        else:
                            return ["none", "cinematic", "photographic", "anime", "fantasy", "digital art"]
                    except APIError as e:
                        if hasattr(e, 'status_code'):
                            if e.status_code == 429:
                                st.warning("Rate limit exceeded, please try again later.")
                            elif e.status_code == 503:
                                st.warning("Model is at capacity, please try again later.")
                            else:
                                st.warning(f"Failed to fetch image styles: {str(e)}")
                        else:
                            st.warning(f"Failed to fetch image styles: {str(e)}")
                    except Exception as e:
                        st.warning(f"Failed to fetch image styles: {str(e)}")
                        return ["none", "cinematic", "photographic", "anime", "fantasy", "digital art"]
                return ["none", "cinematic", "photographic", "anime", "fantasy", "digital art"]
                
            image_style = st.selectbox("Style (Optional)", options=get_image_styles(), index=0, key="image_style")
            with st.expander("Advanced Generation Options"):
                image_negative_prompt = st.text_area(
                    "Negative Prompt",
                    placeholder="Describe what NOT to include...",
                    key="image_negative_prompt",
                    help="Elements to avoid in the image. Max 1500 characters (model specific)."
                )
                
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    image_cfg_scale = st.slider(
                        "CFG Scale",
                        min_value=0.1,
                        max_value=20.0,
                        step=0.1,
                        value=7.5,
                        key="image_cfg_scale",
                        help="Controls prompt adherence (0.1-20.0). Higher is stricter. API default: 7.5 (varies by model)."
                    )
                    image_lora_strength = st.slider(
                        "LoRA Strength",
                        min_value=0,
                        max_value=100,
                        step=1,
                        value=50,
                        key="image_lora_strength",
                        help="LoRA strength (0-100) if model uses LoRAs. API default: N/A (only applies if model has LoRAs)."
                    )
                    image_seed = st.number_input(
                        "Seed",
                        min_value=-99999999,
                        max_value=99999999,
                        value=0, # API default is 0 (or random if 0 not sent)
                        step=1,
                        key="image_seed",
                        help="Random seed (-99999999 to 99999999). 0 is API default (can mean random)."
                    )

                with col_adv2:
                    image_format = st.selectbox(
                        "Image Format",
                        options=["webp", "png", "jpeg"],
                        index=0,  # webp is API default
                        key="image_format",
                        help="Output image format. API default: webp"
                    )
                    image_embed_exif = st.checkbox(
                        "Embed EXIF Metadata",
                        value=False,
                        key="image_embed_exif",
                        help="Embed prompt/settings in image EXIF. API default: false"
                    )
                    image_hide_watermark = st.checkbox(
                        "Hide Watermark",
                        value=False,
                        key="image_hide_watermark",
                        help="Attempt to hide the Venice AI watermark. API default: false"
                    )
                    image_safe_mode = st.checkbox(
                        "Safe Mode",
                        value=True, # API default is true
                        key="image_safe_mode",
                        help="Enable content filtering. API default: true"
                    )
            if st.button("Generate Image", key="gen_image_btn"):
                client = get_venice_client()
                if client and image_prompt:
                    with st.spinner("Generating image..."):
                        try:
                            # Prepare parameters for the API call
                            api_params = {
                                "model": image_model,
                                "prompt": image_prompt,
                                "width": image_width,
                                "height": image_height,
                                "steps": image_steps,
                                "style_preset": image_style if image_style != "none" else None,
                                
                                # Newly added parameters
                                "cfg_scale": image_cfg_scale,
                                "embed_exif_metadata": image_embed_exif,
                                "format": image_format,
                                "hide_watermark": image_hide_watermark,
                                "safe_mode": image_safe_mode,
                                "seed": image_seed,
                                "lora_strength": image_lora_strength, # Pass directly
                            }

                            # Conditionally add negative_prompt if it has content
                            if image_negative_prompt:
                                api_params["negative_prompt"] = image_negative_prompt
                            
                            # The 'return_binary' parameter defaults to False in the API if not sent.
                            # The current application logic decodes base64 images, which aligns with
                            # return_binary=False (or not sent). So, no need to explicitly add it.

                            response = run_async_in_streamlit(client.image.generate(**api_params))
                            if response and response.get("images") and isinstance(response.get("images"), list) and len(response.get("images")) > 0 and isinstance(response.get("images")[0], str):
                                b64_data = response["images"][0]
                                img_bytes = base64.b64decode(b64_data)
                                pil_image = PILImage.open(io.BytesIO(img_bytes))
                                st.session_state.generated_image = pil_image
                                st.success("Image generated successfully.")
                            else:
                                st.error("Failed to generate image or no image data in response.")
                        except APIError as e:
                            if hasattr(e, 'status_code'):
                                if e.status_code == 429:
                                    st.warning("Rate limit exceeded, please try again later.")
                                elif e.status_code == 503:
                                    st.warning("Model is at capacity, please try again later.")
                                else:
                                    st.error(f"API Error: {str(e)}")
                            else:
                                st.error(f"API Error: {str(e)}")
                        except Exception as e:
                            st.error(f"Error generating image: {str(e)}")
        with col2:
            if 'generated_image' in st.session_state:
                st.image(st.session_state.generated_image, caption="Generated Image", use_container_width=True)

# Audio Tab
with tabs[3]:  # Audio
    st.header("Text-to-Speech Generation")
    audio_model = st.selectbox("Select Model", options=["tts-kokoro"], index=0, key="audio_model")
    audio_text = st.text_area("Input Text", placeholder="Enter text to synthesize...", key="audio_text")
    audio_voice = st.selectbox("Select Voice", options=[v.value for v in Voice], index=4, key="audio_voice")
    audio_format = st.radio("Response Format", options=[rf.value for rf in ResponseFormat if rf.value in ["mp3", "opus", "aac", "flac"]], index=0, key="audio_format")
    audio_speed = st.slider("Speech Speed", min_value=0.25, max_value=4.0, step=0.05, value=1.0, key="audio_speed")
    audio_stream = st.checkbox("Stream Audio", value=False, key="audio_stream")
    if st.button("Generate Speech", key="gen_audio_btn"):
        client = get_venice_client()
        if client and audio_text:
            with st.spinner("Generating speech..."):
                try:
                    selected_voice = Voice(audio_voice)
                    selected_format = ResponseFormat(audio_format)
                    temp_file_suffix = f".{selected_format.value}"
                    if audio_stream:
                        audio_chunks = []
                        response_stream = run_async_in_streamlit(client.audio.create_speech(
                            model=audio_model,
                            input=audio_text,
                            voice=selected_voice,
                            response_format=selected_format,
                            speed=audio_speed,
                            stream=True
                        ))
                        for chunk in response_stream:
                            audio_chunks.append(chunk)
                        if audio_chunks:
                            audio_bytes = b"".join(audio_chunks)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix, mode='wb') as tmp_audio_file:
                                tmp_audio_file.write(audio_bytes)
                                temp_file_path = tmp_audio_file.name
                            st.audio(temp_file_path, format=f"audio/{selected_format.value}")
                            st.success("Speech generated successfully (streamed and saved).")
                        else:
                            st.error("No audio data received from stream.")
                    else:
                        audio_bytes = run_async_in_streamlit(client.audio.create_speech(
                            model=audio_model,
                            input=audio_text,
                            voice=selected_voice,
                            response_format=selected_format,
                            speed=audio_speed,
                            stream=False
                        ))
                        if audio_bytes:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=temp_file_suffix, mode='wb') as tmp_audio_file:
                                tmp_audio_file.write(audio_bytes)
                                temp_file_path = tmp_audio_file.name
                            st.audio(temp_file_path, format=f"audio/{selected_format.value}")
                            st.success("Speech generated successfully.")
                        else:
                            st.error("No audio data received.")
                except APIError as e:
                    if hasattr(e, 'status_code'):
                        if e.status_code == 429:
                            st.warning("Rate limit exceeded, please try again later.")
                        elif e.status_code == 503:
                            st.warning("Model is at capacity, please try again later.")
                        else:
                            st.error(f"API Error: {str(e)}")
                    else:
                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error generating speech: {str(e)}")

# Embeddings Tab
with tabs[4]:  # Embeddings
    st.header("Text Embeddings Generation")
    emb_model = st.selectbox("Select Model", options=["text-embedding-bge-m3"], index=0, key="emb_model")
    emb_text = st.text_area("Input Text(s)", placeholder="Enter text or JSON array of texts for batch processing", key="emb_text")
    if st.button("Generate Embeddings", key="gen_emb_btn"):
        client = get_venice_client()
        if client and emb_text:
            with st.spinner("Generating embeddings..."):
                try:
                    input_data = emb_text
                    try:
                        parsed_json = json.loads(emb_text)
                        if isinstance(parsed_json, list) and all(isinstance(item, str) for item in parsed_json):
                            input_data = parsed_json
                    except json.JSONDecodeError:
                        pass
                    response = run_async_in_streamlit(client.embeddings.create(model=emb_model, input=input_data))
                    embeddings_list = [item.embedding for item in response.data] if response.data else []
                    if embeddings_list:
                        st.json(embeddings_list)
                        st.success("Embeddings generated successfully.")
                    else:
                        st.error("No embeddings generated or response format unexpected.")
                except APIError as e:
                    if hasattr(e, 'status_code'):
                        if e.status_code == 429:
                            st.warning("Rate limit exceeded, please try again later.")
                        elif e.status_code == 503:
                            st.warning("Model is at capacity, please try again later.")
                        else:
                            st.error(f"API Error: {str(e)}")
                    else:
                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error generating embeddings: {str(e)}")

# Billing Tab
with tabs[5]:  # Billing
    st.header("Billing Information")
    
    # Pagination controls
    col1, col2 = st.columns(2)
    with col1:
        page_number = st.number_input("Page Number", min_value=1, value=1, step=1, help="Page number for pagination")
    with col2:
        items_per_page = st.number_input("Items Per Page", min_value=1, max_value=500, value=20, step=10, help="Number of items to display per page")
    
    if st.button("Get Usage Data", key="get_usage_btn"):
        client = get_venice_client()
        if client:
            with st.spinner("Fetching usage data..."):
                try:
                    # Include pagination parameters in the API call
                    response = run_async_in_streamlit(client.billing.get_usage(page=page_number, limit=items_per_page))
                    
                    # Process usage data
                    if hasattr(response, 'model_dump'):
                        processed_data = response.model_dump(mode='json')
                    elif isinstance(response, dict):
                        processed_data = response
                    else:
                        processed_data = json.loads(json.dumps(response, default=str))
                    
                    # Display usage data
                    st.json(processed_data)
                    
                    # Display pagination information
                    pagination_info = {}
                    if hasattr(response, 'headers'):
                        # Extract pagination headers if available
                        pagination_headers = [
                            'x-pagination-page', 'x-pagination-limit',
                            'x-pagination-total', 'x-pagination-total-pages'
                        ]
                        for header in pagination_headers:
                            if header in response.headers:
                                key = header.replace('x-pagination-', '')
                                pagination_info[key] = response.headers[header]
                    
                    if pagination_info:
                        st.subheader("Pagination Information")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Current Page", pagination_info.get('page', page_number))
                            st.metric("Total Items", pagination_info.get('total', 'Unknown'))
                        with col2:
                            st.metric("Items Per Page", pagination_info.get('limit', items_per_page))
                            st.metric("Total Pages", pagination_info.get('total-pages', 'Unknown'))
                    
                    st.success("Usage data retrieved successfully.")
                except APIError as e:
                    if hasattr(e, 'status_code'):
                        if e.status_code == 429:
                            st.warning("Rate limit exceeded, please try again later.")
                        elif e.status_code == 503:
                            st.warning("Model is at capacity, please try again later.")
                        else:
                            st.error(f"API Error: {str(e)}")
                    else:
                        st.error(f"API Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error fetching usage data: {str(e)}")

# API Keys Tab
with tabs[6]:  # API Keys
    st.header("API Key Management")
    
    # Create tabs for listing and creating API keys
    api_key_tabs = st.tabs(["List API Keys", "Create API Key"])
    
    # List API Keys Tab
    with api_key_tabs[0]:
        # Pagination controls
        col1, col2 = st.columns(2)
        with col1:
            keys_page_number = st.number_input("Page Number", min_value=1, value=1, step=1, help="Page number for pagination", key="keys_page")
        with col2:
            keys_items_per_page = st.number_input("Items Per Page", min_value=1, max_value=500, value=10, step=5, help="Number of items to display per page", key="keys_limit")
        
        if st.button("List API Keys", key="list_api_keys_btn"):
            client = get_venice_client()
            if client:
                with st.spinner("Fetching API keys..."):
                    try:
                        # Include pagination parameters in the API call
                        api_keys_response = run_async_in_streamlit(client.api_keys.list(page=keys_page_number, limit=keys_items_per_page))
                        
                        # Process API keys data
                        processed_data = []
                        if api_keys_response and api_keys_response.get('data'):
                            st.session_state.api_keys_data = api_keys_response.get('data')  # Store for deletion functionality
                            for key_obj in api_keys_response.get('data', []): # Ensure key_obj is a dict
                                processed_data.append({
                                    "ID": key_obj.get('id', 'N/A'),
                                    "Name": key_obj.get('description', 'N/A'), # API returns 'description'
                                    "Created At": str(key_obj.get('createdAt', 'N/A')), # API returns 'createdAt'
                                    "Last Used At": str(key_obj.get('lastUsedAt', 'N/A')), # API returns 'lastUsedAt'
                                    "Prefix": key_obj.get('last6Chars', 'N/A'), # API returns 'last6Chars'
                                    "Actions": "Delete"
                                })
                        
                        # Display API keys
                        if processed_data:
                            st.dataframe(processed_data, use_container_width=True)
                            
                            # API Key Deletion Section
                            st.subheader("Delete API Key")
                            key_to_delete = st.selectbox(
                                "Select API Key to delete",
                                options=[f"{key_data['ID']} ({key_data['Prefix']})" for key_data in processed_data],
                                format_func=lambda x: x,
                                key="delete_key_select"
                            )
                            
                            if key_to_delete:
                                key_id = key_to_delete.split(" ")[0]  # Extract ID from the formatted string
                                with st.expander("Delete API Key Confirmation"):
                                    st.warning(f"⚠️ You are about to delete API Key with ID: {key_id}")
                                    st.warning("This action cannot be undone!")
                                    confirm_delete = st.checkbox("I understand and want to delete this API Key", key="confirm_delete_key")
                                    
                                    if st.button("Delete API Key", key="delete_key_btn", disabled=not confirm_delete):
                                        try:
                                            delete_response = run_async_in_streamlit(client.api_keys.delete(id=key_id))
                                            if delete_response and hasattr(delete_response, 'success') and delete_response.success:
                                                st.success(f"API Key {key_id} deleted successfully!")
                                                st.rerun()  # Refresh the page to update the API keys list
                                            else:
                                                st.error("Failed to delete API Key. Please try again.")
                                        except APIError as e:
                                            if hasattr(e, 'status_code'):
                                                if e.status_code == 429:
                                                    st.warning("Rate limit exceeded, please try again later.")
                                                elif e.status_code == 503:
                                                    st.warning("Model is at capacity, please try again later.")
                                                else:
                                                    st.error(f"API Error: {str(e)}")
                                            else:
                                                st.error(f"API Error: {str(e)}")
                                        except Exception as e:
                                            st.error(f"Error deleting API Key: {str(e)}")
                        else:
                            st.info("No API Keys found.")
                        
                        # Display pagination information if available
                        pagination_info = {}
                        if hasattr(api_keys_response, 'headers'):
                            # Extract pagination headers if available
                            pagination_headers = [
                                'x-pagination-page', 'x-pagination-limit',
                                'x-pagination-total', 'x-pagination-total-pages'
                            ]
                            for header in pagination_headers:
                                if header in api_keys_response.headers:
                                    key = header.replace('x-pagination-', '')
                                    pagination_info[key] = api_keys_response.headers[header]
                        
                        if pagination_info:
                            st.subheader("Pagination Information")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current Page", pagination_info.get('page', keys_page_number))
                                st.metric("Total Items", pagination_info.get('total', 'Unknown'))
                            with col2:
                                st.metric("Items Per Page", pagination_info.get('limit', keys_items_per_page))
                                st.metric("Total Pages", pagination_info.get('total-pages', 'Unknown'))
                        else:
                            # If no pagination headers are found, display a message
                            st.info("Pagination might not be supported for API keys listing.")
                        
                        st.success("API Keys listed successfully.")
                    except APIError as e:
                        if hasattr(e, 'status_code'):
                            if e.status_code == 429:
                                st.warning("Rate limit exceeded, please try again later.")
                            elif e.status_code == 503:
                                st.warning("Model is at capacity, please try again later.")
                            else:
                                st.error(f"API Error: {str(e)}")
                        else:
                            st.error(f"API Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error listing API keys: {str(e)}")
    
    # Create API Key Tab
    with api_key_tabs[1]:
        st.subheader("Create New API Key")
        st.info("⚠️ Note: The API key will be shown only once after creation. Make sure to store it securely!")
        
        with st.form("create_api_key_form"):
            # Required fields
            api_key_type = st.selectbox(
                "API Key Type",
                options=["INFERENCE", "ADMIN"],
                help="Admin keys have full access to the API while inference keys are only able to call inference endpoints."
            )
            
            description = st.text_input(
                "Description",
                placeholder="Enter a description for this API key",
                help="A descriptive name for this API key"
            )
            
            # Optional fields - Expiration
            st.subheader("Expiration (Optional)")
            use_expiration = st.checkbox("Set expiration date", key="use_expiration")
            expires_at = None
            if use_expiration:
                expiration_date = st.date_input(
                    "Expiration Date",
                    value=None,
                    help="The date when this API key will expire. Leave blank for no expiration."
                )
                if expiration_date:
                    expires_at = expiration_date.isoformat()  # Format as YYYY-MM-DD
            
            # Optional fields - Consumption Limits
            st.subheader("Consumption Limits (Optional)")
            use_consumption_limits = st.checkbox("Set consumption limits", key="use_consumption")
            consumption_limit = None
            if use_consumption_limits:
                col1, col2 = st.columns(2)
                with col1:
                    usd_limit = st.number_input(
                        "USD Limit",
                        min_value=0.0,
                        step=10.0,
                        help="Maximum USD to be consumed by this key"
                    )
                with col2:
                    vcu_limit = st.number_input(
                        "VCU Limit",
                        min_value=0.0,
                        step=10.0,
                        help="Maximum VCU to be consumed by this key"
                    )
                consumption_limit = {"usd": usd_limit, "vcu": vcu_limit}
            
            # Submit button
            submit_button = st.form_submit_button("Create API Key")
            
        # Handle form submission
        if submit_button:
            if not description:
                st.error("Description is required!")
            else:
                client = get_venice_client()
                if client:
                    try:
                        # Prepare request parameters
                        create_params = {
                            "apiKeyType": api_key_type,
                            "description": description
                        }
                        
                        # Add optional parameters if provided
                        if expires_at:
                            create_params["expiresAt"] = expires_at
                        
                        if consumption_limit:
                            create_params["consumptionLimit"] = consumption_limit
                        
                        # Create the API key
                        with st.spinner("Creating API key..."):
                            response = run_async_in_streamlit(client.api_keys.create(**create_params))
                            
                            if response and hasattr(response, 'data') and hasattr(response.data, 'apiKey'):
                                # Display the API key (shown only once)
                                st.success("API Key created successfully!")
                                st.code(response.data.apiKey, language="text")
                                
                                # Important warning about storing the key
                                st.warning("⚠️ IMPORTANT: This API key is displayed only once and cannot be retrieved later. Please save it securely now!")
                                
                                # Show other details
                                st.json({
                                    "apiKeyType": getattr(response.data, 'apiKeyType', api_key_type),
                                    "description": description,
                                    "expiresAt": expires_at if expires_at else "No expiration"
                                })
                            else:
                                st.error("Failed to create API key. Please try again.")
                    except APIError as e:
                        if hasattr(e, 'status_code'):
                            if e.status_code == 429:
                                st.warning("Rate limit exceeded, please try again later.")
                            elif e.status_code == 503:
                                st.warning("Model is at capacity, please try again later.")
                            else:
                                st.error(f"API Error: {str(e)}")
                        else:
                            st.error(f"API Error: {str(e)}")
                    except Exception as e:
                        st.error(f"Error creating API key: {str(e)}")

# Utilities Tab
with tabs[7]:  # Utilities
    st.header("Utilities")
    with st.expander("Token Estimator"):
        token_text = st.text_area("Text for Token Estimation", placeholder="Enter text or JSON messages", key="token_text")
        token_model = st.selectbox("Model for Estimation", options=["qwen3-4b-chat", "gpt-4", "gpt-3.5-turbo"], index=0, key="token_model")
        if st.button("Estimate Tokens", key="est_tokens_btn"):
            if token_text:
                try:
                    if token_model == "qwen3-4b-chat":
                        # Assuming qwen3-4b-chat uses a tokenizer compatible with gpt-3.5-turbo/gpt-4,
                        # which typically use "cl100k_base".
                        encoding = tiktoken.get_encoding("cl100k_base")
                    else:
                        # For "gpt-4" and "gpt-3.5-turbo", encoding_for_model should work.
                        encoding = tiktoken.encoding_for_model(token_model)
                    num_tokens = len(encoding.encode(token_text))
                    st.text(f"Estimated Token Count: {num_tokens}")
                    st.success("Tokens estimated successfully.")
                except Exception as e:
                    st.error(f"Error estimating tokens: {str(e)}")
            else:
                st.error("Input text cannot be empty.")