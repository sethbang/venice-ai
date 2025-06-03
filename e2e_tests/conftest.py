import os
import pytest
import pytest_asyncio
from typing import List, Tuple
from venice_ai import VeniceClient, AsyncVeniceClient
from venice_ai.exceptions import VeniceError, APIError # Import APIError

# Fixture to load the API key from environment variables
@pytest.fixture(scope="session")
def api_key():
    """Loads the API key from the VENICE_API_KEY environment variable."""
    key = os.environ.get("VENICE_API_KEY")
    if not key:
        pytest.fail("VENICE_API_KEY environment variable not set.")
    return key

# Fixture for the synchronous VeniceClient
@pytest.fixture(scope="session")
def venice_client(api_key):
    """Provides a synchronous VeniceClient instance."""
    client = VeniceClient(api_key=api_key)
    # Use internal attributes for logging, ensuring they exist
    print(f"Initializing VeniceClient with API key: {client._api_key[:4]}...{client._api_key[-4:]}")
    print(f"VeniceClient using base URL: {client._base_url}")
    yield client
    client.close()

# Fixture for the asynchronous VeniceClient
@pytest_asyncio.fixture(scope="function")
async def async_venice_client(api_key):
    """Provides an asynchronous AsyncVeniceClient instance."""
    client = AsyncVeniceClient(api_key=api_key)
    # Use internal attributes for logging, ensuring they exist
    print(f"Initializing AsyncVeniceClient with API key: {client._api_key[:4]}...{client._api_key[-4:]}")
    print(f"AsyncVeniceClient using base URL: {client._base_url}")
    yield client
    await client.close()

# Fixture for managing resources created during tests for cleanup
@pytest.fixture(scope="function")
def created_resources(venice_client: VeniceClient): # Add venice_client dependency
    """Fixture to track and clean up resources created during tests."""
    resources_to_clean: List[Tuple[str, str]] = []
    yield resources_to_clean
    
    print("\nCleaning up created resources...")
    if not resources_to_clean:
        print("No resources to clean up.")
        return

    for resource_type, resource_id in resources_to_clean:
        print(f"Attempting to clean up {resource_type}: {resource_id}")
        try:
            if resource_type == "api_key":
                # Ensure the API key used for the client has permissions to delete other keys.
                # This typically means the client should be initialized with an ADMIN API key.
                # The `venice_client` fixture uses the VENICE_API_KEY from env.
                # This key MUST have admin privileges for cleanup to work.
                venice_client.api_keys.delete(api_key_id=resource_id)
                print(f"Successfully deleted API key: {resource_id}")
            # Add other resource types here as needed
            # elif resource_type == "character":
            #     venice_client.characters.delete(character_id=resource_id) # Example
            #     print(f"Successfully deleted character: {resource_id}")
            else:
                print(f"Unknown resource type '{resource_type}' for ID '{resource_id}'. Skipping cleanup.")
        except APIError as e: # Catch APIError specifically for status_code and message
            print(f"API Error cleaning up {resource_type} {resource_id}: Status {e.status_code} - {e.message}")
        except VeniceError as e: # Catch other VeniceErrors
            print(f"Venice Client Error cleaning up {resource_type} {resource_id}: {e}")
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred during cleanup of {resource_type} {resource_id}: {e}")
    print("Resource cleanup finished.")

@pytest.fixture(scope="session")
def default_chat_model_id(venice_client: VeniceClient) -> str:
    """
    Fetches a suitable default chat model ID that supports function calling if possible.
    Fails the test session if no suitable model is found.
    """
    print("\nFetching available models to determine a default chat model ID for E2E tests...")
    try:
        print("Attempting to call venice_client.models.list()...", flush=True)
        models_response = venice_client.models.list() # This returns a ModelList TypedDict
        print(f"Raw models_response from API: {models_response}", flush=True)
    
        models_data = models_response.get("data")
        if models_data is None:
             print("Models API response does not contain a 'data' field.", flush=True)
             pytest.fail("Models API response does not contain a 'data' field. Cannot determine default chat model.")
        if not isinstance(models_data, list) or not models_data:
            print(f"Models 'data' field is not a list or is empty. Raw response: {models_response}", flush=True)
            pytest.fail("No models returned in 'data' field or 'data' is empty. Cannot determine a default chat model.")
    
        print(f"Successfully fetched models_data. Count: {len(models_data)}", flush=True)
        suitable_models_with_tool_support: List[str] = []
        suitable_chat_models: List[str] = []
    
        print(f"Starting to process {len(models_data)} models...", flush=True)

        for model_info in models_data: # model_info is a Model TypedDict
            model_id = model_info.get("id")
            model_spec = model_info.get("model_spec", {}) # Keep for capabilities

            raw_model_type_api_val = model_info.get("type") # This is a string like "text", "image", etc. from API
            model_type_str = ""
            if isinstance(raw_model_type_api_val, str):
                model_type_str = raw_model_type_api_val.lower()
            elif raw_model_type_api_val is not None:
                # This case should ideally not happen if API is consistent and returns strings for type
                print(f"Warning: Model ID '{model_id}' has unexpected type for 'type' field: {type(raw_model_type_api_val)}, value: {raw_model_type_api_val}", flush=True)
            # if raw_model_type_api_val is None, model_type_str remains ""

            capabilities = model_spec.get("capabilities", {})
            supports_tool_calls = capabilities.get("supportsFunctionCalling", False)

            print(f"  Processing model: ID='{model_id}', Type='{model_type_str}', Supports Tools='{supports_tool_calls}'")
            print(f"    Full model_info: {model_info}") # ADDED: Print full model info for debugging

            if model_id and model_type_str == "text": # Corrected: compare with "text"
                if supports_tool_calls:
                    print(f"    -> Qualified as tool-supporting chat model: {model_id}")
                    suitable_models_with_tool_support.append(model_id)
                else:
                    print(f"    -> Qualified as chat model (no tool support): {model_id}")
                    suitable_chat_models.append(model_id)
            else:
                print(f"    -> Skipping model: ID='{model_id}', Type='{model_type_str}' (not 'text')") # Corrected: print "text"

        if not suitable_models_with_tool_support and not suitable_chat_models:
            print("No models were found matching type 'text'.") # Corrected: print "text"
            print("Attempting fallback: searching for 'chat' or 'text' in model ID for any model type...")
            for model_info in models_data:
                model_id = model_info.get("id")
                # Fallback: check if 'chat' or 'text' is in the model ID string itself
                if model_id and ("chat" in model_id.lower() or "text" in model_id.lower()):
                    model_spec = model_info.get("model_spec", {})
                    capabilities = model_spec.get("capabilities", {})
                    supports_tool_calls = capabilities.get("supportsFunctionCalling", False)
                    # Avoid re-adding if already processed by primary logic (though unlikely if type wasn't 'text_generation')
                    if model_id not in suitable_models_with_tool_support and model_id not in suitable_chat_models:
                        if supports_tool_calls:
                            print(f"    -> Fallback qualified (ID heuristic, tool support): {model_id}")
                            suitable_models_with_tool_support.append(model_id)
                        else:
                            print(f"    -> Fallback qualified (ID heuristic, no tool support): {model_id}")
                            suitable_chat_models.append(model_id)

        if suitable_models_with_tool_support:
            common_tool_models = [m for m in suitable_models_with_tool_support if any(sub.lower() in m.lower() for sub in ["llama-3.2-3b", "gpt-4", "claude-3", "gemini", "command-r"])]
            selected_model_id = common_tool_models[0] if common_tool_models else suitable_models_with_tool_support[0]
            print(f"Selected default chat model (supports tool calls): {selected_model_id}")
            return selected_model_id
        elif suitable_chat_models:
            common_chat_models = [m for m in suitable_chat_models if any(sub.lower() in m.lower() for sub in ["llama-3.2-3b", "gpt-3.5", "claude-3", "gemini", "command"])]
            selected_model_id = common_chat_models[0] if common_chat_models else suitable_chat_models[0]
            print(f"Selected default chat model (no tool support found, using fallback): {selected_model_id}")
            return selected_model_id
        else:
            pytest.fail("No suitable text_generation models (or fallback chat models by ID) found from API. Cannot proceed.")
            
    except APIError as e:
        print(f"Caught APIError during model fetching: Status {getattr(e, 'status_code', 'N/A')} - {getattr(e, 'message', str(e))}", flush=True)
        pytest.fail(f"API Error while fetching models: Status {getattr(e, 'status_code', 'N/A')} - {getattr(e, 'message', str(e))}. Cannot determine default chat model.")
    except VeniceError as e:
        print(f"Caught VeniceError during model fetching: {e}", flush=True)
        pytest.fail(f"Venice Client Error while fetching models: {e}. Cannot determine default chat model.")
    except Exception as e:
        print(f"Caught unexpected Exception during model fetching: {e}", flush=True)
        pytest.fail(f"Unexpected error while fetching models: {e}. Cannot determine default chat model.")
    return "" # Should not be reached