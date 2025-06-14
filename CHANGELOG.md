# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2025-06-13

### Fixed

- **Documentation Build Issues**: Fixed empty sections in Sphinx API reference documentation that were appearing in Read the Docs builds
  - Updated `.readthedocs.yaml` to properly install the `venice_ai` package during documentation builds
  - Added missing imports in `src/venice_ai/resources/__init__.py` for `ApiKeys`, `AsyncApiKeys`, `Audio`, `AsyncAudio`, `Billing`, `AsyncBilling`, `Embeddings`, `AsyncEmbeddings`, and `AsyncModels`
  - Added comprehensive type imports in `src/venice_ai/types/__init__.py` for image, api_keys, audio, embeddings, and billing modules
  - Added explicit `__all__` list to `src/venice_ai/exceptions.py` for better module discovery
  - Fixed missing `ModelTraitList` and `ModelCompatibilityList` exports in types package
- **Test Runner & Coverage**: Refactored `test_runner.py` to use `pytest-cov` directly, resolving significant code coverage reporting inaccuracies when running tests in parallel with `pytest-xdist`.
- **Embedding Tests**: Updated `e2e_tests/test_05_embeddings.py` with improved and corrected end-to-end tests for embedding functionalities.
- **CI Workflow**: Modified `.github/workflows/python-publish.yaml` to enhance test execution, enabling or optimizing parallel test runs.

## [1.1.0] - 2025-06-09

### Added

- Implemented support for `logprobs` and `top_logprobs` parameters in Chat Completions API, allowing users to retrieve token likelihoods. Includes E2E tests and documentation updates.

#### **🏗️ Core SDK Architecture & Client Enhancements**

- **BaseClient Foundation**: Introduced [`BaseClient`](src/venice_ai/_client.py:38) class providing shared functionality for both sync and async clients, including common initialization logic, retry configuration, and transport setup.
- **Advanced HTTP Configuration**: Added comprehensive HTTP client configuration options to both `VeniceClient` and `AsyncVeniceClient`:
  - Support for custom `httpx.Client`/`httpx.AsyncClient` instances
  - Direct configuration of proxy, transport, limits, cert, verify, trust_env, HTTP/1.1, HTTP/2 settings
  - Custom event hooks and default encoding support
  - Follow redirects and max redirects configuration
- **Global Timeout Management**: Implemented `default_timeout` parameter for setting global timeout defaults across all API calls, with per-request override capability.
- **Automatic Retry System**: Integrated `httpx-retries` library with configurable retry behavior:
  - Configurable `max_retries` (default: 2)
  - Adjustable `retry_backoff_factor` (default: 0.1)
  - Customizable `retry_status_forcelist` (default: [429, 500, 502, 503, 504])
  - Respect for `Retry-After` headers in rate limit responses
- **Sentinel Type System**: Added [`NotGiven`](src/venice_ai/utils.py:10) sentinel type and `NOT_GIVEN` constant for distinguishing between `None` and not-provided parameters.

#### **🎵 Audio API Major Expansion**

- **Streaming Audio Support**: Implemented method overloads for [`create_speech()`](src/venice_ai/resources/audio.py:68) supporting both streaming and non-streaming audio generation:
  - `stream=False`: Returns `bytes` for immediate audio data
  - `stream=True`: Returns `Iterator[bytes]` for streaming audio chunks
- **Voice Management System**: Added comprehensive [`get_voices()`](src/venice_ai/resources/audio.py:231) method with advanced filtering:
  - Filter by model ID, gender (male/female/unknown), and region code
  - Automatic voice metadata parsing from voice IDs
  - Language and accent detection for 15+ supported regions
- **Enhanced Voice Metadata**: Implemented [`REGION_LANGUAGE_MAPPING`](src/venice_ai/resources/audio.py:29) supporting:
  - English variants: American, British, Canadian, Scottish, Welsh, Australian, Indian
  - International languages: German, Spanish, French, Italian, Japanese, Korean, Portuguese, Russian, Mandarin Chinese
- **Improved Parameter Handling**: Set sensible defaults for audio generation (`response_format="mp3"`, `speed=1.0`).
- **Raw Response Support**: Added [`_request_raw_response()`](src/venice_ai/_resource.py:69) and [`_arequest_raw_response()`](src/venice_ai/_resource.py:185) methods for handling binary audio content and streaming responses.

#### **👥 Characters API Implementation**

- **Character Listing**: Implemented [`Characters.list()`](src/venice_ai/resources/characters.py:24) method with support for extra headers, query parameters, and custom timeouts.
- **Enhanced Character Model**: Completely redesigned [`Character`](src/venice_ai/types/characters.py:6) Pydantic model with modern fields:
  - Core identification: `slug`, `name`, `description`
  - AI capabilities: `system_prompt`, `user_prompt`, `vision_enabled`
  - Media support: `image_url`, `voice_id`
  - Organization: `category_tags`
  - Timestamps: `created_at`, `updated_at` with proper datetime handling
- **Simplified Character List**: Streamlined [`CharacterList`](src/venice_ai/types/characters.py:49) model for cleaner API responses.

#### **🔧 Enhanced Error Handling & Resilience**

- **Retry-After Header Parsing**: Implemented [`_parse_retry_after_header()`](src/venice_ai/exceptions.py:400) function supporting:
  - Integer seconds format (e.g., "120")
  - HTTP-date format (e.g., "Wed, 21 Oct 2015 07:28:00 GMT")
  - Timezone-aware datetime calculations
  - Server time synchronization using response `Date` header
- **Enhanced RateLimitError**: Extended [`RateLimitError`](src/venice_ai/exceptions.py:230) with `retry_after_seconds` attribute for intelligent retry logic.
- **Improved Error Context**: Better error message formatting and context preservation across the exception hierarchy.

#### **🧪 Comprehensive Testing Infrastructure**

- **Massive Test Suites**: Added extensive functional test coverage:
  - [`venice_sdk_async_test.py`](venice_sdk_async_test.py:1): 126k lines of async functionality tests
  - [`venice_sdk_sync_test.py`](venice_sdk_sync_test.py:1): 49k lines of sync functionality tests
- **HTTP Configuration Testing**: New [`tests/test_client_http_config.py`](tests/test_client_http_config.py:1) for validating advanced HTTP client options.
- **Enhanced API Coverage**: Expanded test coverage for:
  - Audio streaming and non-streaming modes with various parameters
  - Characters API functionality and error handling
  - Chat completions with tool usage, JSON format, and streaming
  - Image generation with advanced parameters (negative_prompt, seed, format)
  - API key management including Web3 token functionality
  - Retry mechanism behavior and configuration
  - Global timeout functionality across all endpoints

#### **📚 Documentation & Project Infrastructure**

- **Comprehensive Changelog**: Created this detailed changelog following Keep a Changelog format.
- **Contributing Guidelines**: Added [`CONTRIBUTING.md`](CONTRIBUTING.md:1) with clear issue reporting guidelines.
- **Enhanced API Documentation**: Updated [`docs/api.rst`](docs/api.rst:1) with 168 new lines covering:
  - Advanced HTTP client configuration examples
  - Retry mechanism documentation
  - Global timeout usage patterns
- **Utility Documentation**: Added [`docs/client_utilities.rst`](docs/client_utilities.rst:1) documenting `estimate_token_count` and `validate_chat_messages` utilities.
- **README Overhaul**: Major [`README.md`](README.md:1) updates (144 lines changed) including:
  - Advanced HTTP Client Configuration section with three configuration approaches
  - Updated all code examples to include `default_timeout` parameter
  - Enhanced feature list highlighting automatic retry functionality
  - Improved error handling examples and best practices

### Changed

#### **🔄 Client Architecture Improvements**

- **Inheritance Hierarchy**: `AsyncVeniceClient` now inherits from `BaseClient` for shared functionality and consistent behavior.
- **Request Method Simplification**: Removed manual HTTP 503 retry loops from client request methods (`_request`, `_arequest`, and related stream/multipart methods) in favor of `httpx-retries` integration.
- **Enhanced Documentation**: Significantly expanded docstrings for both sync and async clients with detailed parameter descriptions and usage examples.

#### **📦 Project Configuration & Metadata**

- **Version Bump**: Updated from `1.0.3` to `1.1.0` reflecting significant new features and improvements.
- **Dependency Management**: Added `httpx-retries = "^0.4.0"` as a core dependency for retry functionality.
- **Enhanced Discoverability**: Expanded keywords from 5 to 9 terms: `ai`, `api-client`, `generative-ai`, `llm`, `machine-learning`, `ml`, `sdk`, `venice`, `venice-ai`.
- **Refined Classifiers**: Updated PyPI classifiers:
  - Removed Python 3.10 support (now requires Python 3.11+)
  - Added "Development Status :: 4 - Beta"
  - Added comprehensive topic classifiers for chat, image generation, speech, text processing
  - Added "Typing :: Typed" classifier for type hint support
- **Project URLs**: Added "Issue Tracker" and "Changelog" links for better project navigation.
- **Test Configuration**: Enabled parallel test execution with `pytest-xdist` (`addopts = "-n auto"`).

#### **🎯 API Method Enhancements**

- **Characters API**: Enhanced [`Characters.list()`](src/venice_ai/resources/characters.py:24) with additional parameters for headers, query parameters, body, and timeout customization.
- **Audio API**: Improved [`create_speech()`](src/venice_ai/resources/audio.py:68) with better error handling, streaming support, and parameter validation.
- **Consistent Parameter Patterns**: Standardized optional parameter handling across all API methods using the new `NotGiven` sentinel system.

### Fixed

#### **🐛 API Functionality Corrections**

- **API Key Management**: Corrected API key delete method to use query parameters instead of request body, aligning with API specification.
- **Image Upscale Response Handling**: Fixed Image Upscale functional tests to correctly handle `bytes` response type instead of expecting JSON.
- **Audio Error Processing**: Improved error handling in audio generation to properly consume response bodies before raising exceptions, preventing connection leaks.

#### **🧪 Testing Reliability**

- **Embeddings Test Stability**: Made Embeddings functional tests robustly skipped due to persistent API authentication issues in test environments, preventing false test failures.
- **Response Type Validation**: Enhanced test assertions to properly validate response types across different API endpoints.

### Removed

#### **🗑️ Cleanup & Simplification**

- **Legacy Files**: Removed development and example files:
  - [`app.py`](app.py:1): 883-line example/demo application
  - [`dummy_image.png`](dummy_image.png:1) and [`dummy_image_async.png`](dummy_image_async.png:1): Test image files
  - [`tests/resources/test_billing.py`](tests/resources/test_billing.py:1): 79-line billing test file
- **Billing API Simplification**: Removed [`export()`](src/venice_ai/resources/billing.py:161) method (71 lines) that provided CSV billing data export functionality.
- **Obsolete Type Definitions**: Removed incorrect/placeholder `CharacterChatCompletionRequest` Pydantic model and associated `Stats` model.
- **Deprecated Features**: Removed plans for `ResponseTransformer`/`AsyncResponseTransformer` as existing streaming utilities were deemed sufficient.

#### **📋 Documentation Cleanup**

- **Streamlined Character Documentation**: Simplified character model documentation to focus on current functionality rather than legacy fields.

### Security

#### **🔒 Enhanced Error Information**

- **Rate Limit Intelligence**: `RateLimitError` now safely parses and exposes `Retry-After` header information without leaking sensitive data.
- **Timeout Configuration**: Global timeout settings provide better protection against hanging requests and resource exhaustion.

### Performance

#### **⚡ Efficiency Improvements**

- **Automatic Retries**: Intelligent retry mechanism reduces manual retry logic and improves success rates for transient failures.
- **Parallel Testing**: Enabled parallel test execution reducing CI/CD pipeline duration.
- **Streaming Optimization**: Enhanced audio streaming implementation for better memory efficiency with large audio files.
- **Connection Management**: Improved HTTP connection lifecycle management through better integration with `httpx` features.

---

## [1.0.3] - 2025-06-06

### Added

- Initial release of the Venice AI Python SDK with comprehensive API coverage
- Support for Chat Completions, Image Generation, Audio (TTS), Models, API Keys, Billing, and Characters endpoints
- Both synchronous and asynchronous client implementations
- Comprehensive error handling with custom exception hierarchy
- Type-hinted interfaces for better developer experience
- Resource-oriented client design pattern
- Streaming support for chat completions
- Comprehensive test suite with functional and unit tests
- Sphinx-based documentation system
- Poetry-based dependency management and packaging

### Changed

- Established baseline functionality and API coverage

### Fixed

- Initial bug fixes and stabilization for public release

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. For detailed technical information about any changes, please refer to the git commit history or the linked source files.
