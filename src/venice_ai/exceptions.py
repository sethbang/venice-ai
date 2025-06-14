from typing import Optional, Any
import httpx
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "VeniceError",
    "APIError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseProcessingError",
    "MissingStreamClassError",
    "StreamConsumedError",
    "StreamClosedError",
]

class VeniceError(Exception):
    """
    Base exception for all errors raised by the Venice AI client.
    
    This is the parent class for all custom exceptions in the Venice AI library.
    All other exception classes inherit from this base exception.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object associated with the error.
    :type request: Optional[httpx.Request]
    :param response: Optional. The ``httpx.Response`` object if the error originated from an API call.
    :type response: Optional[httpx.Response]
    
    :ivar message: Human-readable description of the error.
    :vartype message: str
    :ivar request_obj: The ``httpx.Request`` object associated with the error, if available.
    :vartype request_obj: Optional[httpx.Request]
    :ivar response_obj: The ``httpx.Response`` object if the error originated from an API call.
    :vartype response_obj: Optional[httpx.Response]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: Optional[httpx.Response] = None) -> None:
        super().__init__(message)
        self.request_obj = request
        self.response_obj = response
        self.message = message
        
    @property
    def request(self) -> Optional[httpx.Request]:
        return self.request_obj

    @property
    def response(self) -> Optional[httpx.Response]:
        return self.response_obj

class APIError(VeniceError):
    """
    Raised when the API returns a non-2xx status code.
    
    This is a general exception for API-related errors, including server errors (5xx status codes)
    and unhandled client errors (4xx status codes). More specific exception subclasses are available
    for common HTTP status codes.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    
    :ivar status_code: HTTP status code of the response.
    :vartype status_code: int
    :ivar body: Parsed response body, if available.
    :vartype body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response)
        self.status_code = response.status_code
        self.body = body

class AuthenticationError(APIError):
    """
    Raised for 401 Unauthorized errors, typically due to an invalid API key.
    
    This exception is raised when the API returns a 401 status code, indicating that
    the request lacks valid authentication credentials. This commonly occurs when:
    
    - The API key is missing or invalid
    - The API key has been revoked or expired
    - The API key lacks the necessary permissions for the requested operation
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class PermissionDeniedError(APIError):
    """
    Raised for 403 Forbidden errors when access is denied.
    
    This exception is raised when the API returns a 403 status code, indicating that
    the client does not have permission to perform the requested action. The request
    was valid and authenticated, but the server is refusing to authorize it.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class InvalidRequestError(APIError):
    """
    Raised for 400 Bad Request errors due to invalid request parameters.
    
    This exception is raised when the API returns a 400 status code, indicating that
    the server cannot process the request due to client error. This typically occurs when:
    
    - Required fields are missing from the request
    - Parameter values are invalid or malformed
    - The request payload format is incorrect
    - File size exceeds limits (413 status code also maps to this exception)
    - Unsupported content type (415 status code also maps to this exception)
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class NotFoundError(APIError):
    """
    Raised for 404 Not Found errors when a requested resource is not found.
    
    This exception is raised when the API returns a 404 status code, indicating that
    the requested resource could not be found. This commonly occurs when:
    
    - An incorrect model name is specified
    - A character slug does not exist
    - An API endpoint path is invalid
    - A resource identifier (ID) is not found
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class ConflictError(APIError):
    """
    Raised for 409 Conflict errors when a resource conflict occurs.
    
    This exception is raised when the API returns a 409 status code, indicating that
    the request could not be completed due to a conflict with the current state of
    the resource. This may occur when:
    
    - Attempting to create a resource that already exists
    - Concurrent modifications to the same resource
    - Business logic constraints prevent the operation
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class UnprocessableEntityError(APIError):
    """
    Raised for 422 Unprocessable Entity errors due to validation failures.
    
    This exception is raised when the API returns a 422 status code, indicating that
    the request was well-formed but contained semantic errors that prevented it from
    being processed. This typically occurs when:
    
    - Request data fails server-side validation rules
    - Business logic constraints are violated
    - Data format is correct but values are semantically invalid
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

class RateLimitError(APIError):
    """
    Raised for 429 Too Many Requests errors when rate limits are exceeded.
    
    This exception is raised when the API returns a 429 status code, indicating that
    the client has sent too many requests in a given time frame and has exceeded the
    rate limit. The client should wait before making additional requests.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    :param retry_after_seconds: Optional. The number of seconds to wait before retrying,
        parsed from the Retry-After header.
    :type retry_after_seconds: Optional[int]
    
    :ivar retry_after_seconds: Number of seconds to wait before retrying, if available.
    :vartype retry_after_seconds: Optional[int]
    """
    retry_after_seconds: Optional[int]

    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None, retry_after_seconds: Optional[int] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.retry_after_seconds = retry_after_seconds

class InternalServerError(APIError):
    """
    Raised for 500 Internal Server Error and other 5xx server-side errors.
    
    This exception is raised when the API returns a 5xx status code, indicating that
    an error occurred on the API server's end. This includes various server-side
    failures such as:
    
    - Internal server errors (500)
    - Service unavailable (503)
    - Gateway timeout (504)
    - Inference failures
    - Upscale failures
    - Unknown server errors
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param response: The ``httpx.Response`` object from the API call.
    :type response: httpx.Response
    :param body: Optional. The parsed response body, if available.
    :type body: Optional[Any]
    """
    pass # INFERENCE_FAILED, UPSCALE_FAILED, UNKNOWN_ERROR maps here

# Add other specific errors as needed (e.g., FileSizeError for 413)

class APIConnectionError(VeniceError):
    """
    Raised when there's an issue connecting to the Venice AI API.
    
    This exception is raised when network-level connectivity issues prevent the client
    from establishing a connection to the API server. This could be due to:
    
    - Network connectivity problems
    - DNS resolution failures
    - Connection timeouts during establishment
    - SSL/TLS handshake failures
    - Proxy configuration issues

    :param message: Human-readable description of the error. Defaults to "Connection error".
    :type message: str
    :param original_error: Optional. The original exception that caused this error.
    :type original_error: Optional[Exception]
    :param request: Optional. The ``httpx.Request`` object associated with the error.
    :type request: Optional[Any]
    :param response: Optional. The ``httpx.Response`` object if available.
    :type response: Optional[httpx.Response]
    
    :ivar original_error: Original exception that caused this error.
    :vartype original_error: Optional[Exception]
    :ivar request: ``httpx.Request`` object associated with the error, if available.
    :vartype request: Optional[Any]
    """
    def __init__(self, message: str = "Connection error", *, original_error: Optional[Exception] = None, request: Optional[Any] = None, response: Optional[httpx.Response] = None) -> None:
        super().__init__(message, request=request, response=response)
        self.original_error = original_error

class APITimeoutError(VeniceError):
    """
    Raised when an API request times out.
    
    This exception is raised when a request to the Venice AI API takes too long to
    complete and exceeds the configured timeout limit. This can occur during:
    
    - Long-running operations (e.g., image generation, large file processing)
    - Network latency issues
    - Server processing delays
    - Read timeout while waiting for response data

    :param message: Human-readable description of the error. Defaults to "Request timed out".
    :type message: str
    :param original_error: Optional. The original exception that caused this error.
    :type original_error: Optional[Exception]
    :param request: Optional. The ``httpx.Request`` object associated with the error.
    :type request: Optional[Any]
    :param response: Optional. The ``httpx.Response`` object if available.
    :type response: Optional[httpx.Response]
    
    :ivar original_error: Original exception that caused this error.
    :vartype original_error: Optional[Exception]
    :ivar request: ``httpx.Request`` object associated with the error, if available.
    :vartype request: Optional[Any]
    """
    def __init__(self, message: str = "Request timed out", *, original_error: Optional[Exception] = None, request: Optional[Any] = None, response: Optional[httpx.Response] = None) -> None:
        super().__init__(message, request=request, response=response)
        self.original_error = original_error
        
class APIResponseProcessingError(VeniceError):
    """
    Raised when there's an error processing the API response.
    
    This exception is raised when the client successfully receives a response from the
    Venice AI API but encounters an error while processing the response data. This could
    be due to:
    
    - Unexpected response format or structure
    - JSON parsing errors
    - Missing expected fields in the response
    - Data type conversion failures
    - Response validation errors

    :param message: Human-readable description of the error.
    :type message: str
    :param original_error: Optional. The original exception that caused this error.
    :type original_error: Optional[Exception]
    :param response: Optional. The ``httpx.Response`` object if available.
    :type response: Optional[httpx.Response]
    
    :ivar original_error: Original exception that caused this error.
    :vartype original_error: Optional[Exception]
    """
    def __init__(self, message: str, *, original_error: Optional[Exception] = None, response: Optional[httpx.Response] = None) -> None:
        super().__init__(message, response=response)
        self.original_error = original_error

class MissingStreamClassError(VeniceError):
    """
    Raised when stream=True but no stream_cls is provided.
    
    This exception is raised when attempting to use streaming functionality but the
    required stream class parameter is not provided. This typically occurs in chat
    completions or other streaming operations where the client needs to know how to
    handle the streamed response data.
    
    :param message: Human-readable description of the error.
    :type message: str
    """
    pass

class StreamConsumedError(APIConnectionError):
    """
    Raised when an attempt is made to operate on a stream that has already been consumed.
    
    This exception is raised when trying to iterate over a stream that has already been
    fully consumed or exhausted. Once a stream has been consumed, it cannot be re-iterated.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object associated with the error.
    :type request: Optional[httpx.Request]
    :param response: Optional. The ``httpx.Response`` object if available.
    :type response: Optional[httpx.Response]
    """
    pass

class StreamClosedError(APIConnectionError):
    """
    Raised when an attempt is made to operate on a stream whose underlying connection has been closed.
    
    This exception is raised when trying to iterate over a stream whose underlying httpx.Response
    object has been closed. Once a stream's underlying connection is closed, it cannot be iterated.
    
    :param message: Human-readable description of the error.
    :type message: str
    :param request: Optional. The ``httpx.Request`` object associated with the error.
    :type request: Optional[httpx.Request]
    :param response: Optional. The ``httpx.Response`` object if available.
    :type response: Optional[httpx.Response]
    """
    pass

def _parse_retry_after_header(header_value: str, response_date_str: Optional[str] = None) -> Optional[int]:
    """
    Parse the Retry-After header value and return the delay in seconds.
    
    The Retry-After header can contain either:
    1. An integer representing seconds to wait
    2. An HTTP-date string representing when to retry
    
    :param header_value: The value of the Retry-After header
    :type header_value: str
    :param response_date_str: Optional Date header from the response to use as server time
    :type response_date_str: Optional[str]
    
    :return: Number of seconds to wait, or None if parsing fails
    :rtype: Optional[int]
    """
    try:
        # Try to parse as integer (seconds)
        return int(header_value)
    except ValueError:
        try:
            # Try to parse as HTTP-date
            retry_after_dt = parsedate_to_datetime(header_value)
            if retry_after_dt.tzinfo is None:
                # Ensure timezone aware for comparison
                retry_after_dt = retry_after_dt.replace(tzinfo=timezone.utc)

            # Determine the "now" time
            now_dt: datetime
            if response_date_str:
                server_now_dt = parsedate_to_datetime(response_date_str)
                if server_now_dt.tzinfo is None:
                    server_now_dt = server_now_dt.replace(tzinfo=timezone.utc)
                now_dt = server_now_dt
            else:
                now_dt = datetime.now(timezone.utc)
            
            # Calculate the difference in seconds
            delta = (retry_after_dt - now_dt).total_seconds()
            return max(0, int(delta))  # Return non-negative integer seconds
        except (TypeError, ValueError):
            # Handles errors from parsedate_to_datetime or if it's not a valid date
            return None

def _make_status_error(
    message: Optional[str], *, request: Optional[httpx.Request] = None, body: Optional[Any], response: httpx.Response
) -> APIError:
    """
    Creates a specific APIError subclass based on the HTTP status code and response body.
    
    This internal helper function is used to map HTTP error responses to the
    appropriate custom exception class. It analyzes the HTTP status code and
    creates the most specific exception type available.
    
    :param message: Optional. A default error message.
    :type message: Optional[str]
    :param request: Optional. The ``httpx.Request`` object that led to the error.
    :type request: Optional[httpx.Request]
    :param body: Optional. The parsed response body, potentially containing error details.
    :type body: Optional[Any]
    :param response: The ``httpx.Response`` object.
    :type response: httpx.Response

    :return: An instance of an :class:`APIError` subclass corresponding to the HTTP status code.
    :rtype: APIError
    """
    status_code = response.status_code
    
    # Initialize with the generic message passed from the client or a default HTTP status message
    # This 'message' is typically "API error {status_code} for {method} {url}"
    base_message = message if message else f"HTTP Status {status_code}"
    err_msg = base_message # Start with the base message

    detail_from_json: Optional[str] = None
    code_from_json: Optional[str] = None
    actual_raw_text_from_response: Optional[str] = None

    # Attempt to get structured error details or raw text
    # The 'body' argument here is what _translate_httpx_error_to_api_error determined:
    # - a dict if JSON was successfully parsed by _translate
    # - a string if _translate got raw text
    # - None if _translate failed to get anything meaningful

    if isinstance(body, dict): # Body is pre-parsed JSON from _translate
        error_data = body.get("error")
        if isinstance(error_data, dict):
            detail_from_json = error_data.get("message") or error_data.get("detail")
            code_from_json = error_data.get("code")
        elif isinstance(error_data, str): # e.g. body = {"error": "some string error"}
            detail_from_json = error_data
    elif isinstance(body, str): # Body is raw text passed in from _translate
        actual_raw_text_from_response = body
    # This block is intentionally left blank.
    # The `body` parameter is now the single source of truth for the response body.
    # The logic to parse the response has been centralized in `_translate_httpx_error_to_api_error`
    # in the client classes, which correctly handles async operations.
    # `_make_status_error` should not re-parse the response.
    elif body is None:
        pass
    
    logger.debug(f"[_make_status_error] Initial base_message: '{base_message}'")
    logger.debug(f"[_make_status_error] Received body type: {type(body)}, value: {body}")
    logger.debug(f"[_make_status_error] detail_from_json: '{detail_from_json}', code_from_json: '{code_from_json}'")
    logger.debug(f"[_make_status_error] actual_raw_text_from_response: '{actual_raw_text_from_response}'")

    # Construct the final error message based on what was found
    # err_msg is already initialized to base_message. We append details to it.

    # err_msg is already initialized to base_message.
    if detail_from_json: # Most specific message from JSON error body
        # Prepend the base_message (generic or custom passed message)
        # Then append the specific detail from JSON.
        err_msg = f"{base_message}: {detail_from_json}"
        if code_from_json:
            err_msg = f"{err_msg} (Code: {code_from_json})"
    elif code_from_json: # JSON error body had a code but no message/detail
        err_msg = f"{base_message} (Code: {code_from_json})"
    elif actual_raw_text_from_response and isinstance(body, str): # Text came from body arg being a string
        err_msg = f"{base_message}: {actual_raw_text_from_response}"
    # Else (if body was None and actual_raw_text_from_response came from response.text,
    # OR if nothing specific was found), err_msg remains the initial base_message.

    logger.debug(f"[_make_status_error] Final err_msg: '{err_msg}' for status_code: {status_code}")

    if status_code == 400:
        return InvalidRequestError(message=err_msg, request=request, response=response, body=body)
    if status_code == 401:
        return AuthenticationError(message=err_msg, request=request, response=response, body=body)
    if status_code == 403:
        return PermissionDeniedError(message=err_msg, request=request, response=response, body=body)
    if status_code == 404:
        # Always return NotFoundError for 404 status codes
        # This is needed to match the test expectation in test_make_status_error_status_codes
        return NotFoundError(message=err_msg, request=request, response=response, body=body)
    if status_code == 409:
         return ConflictError(message=err_msg, request=request, response=response, body=body)
    if status_code == 413: # File size
        return InvalidRequestError(message=err_msg, request=request, response=response, body=body) # Reusing InvalidRequest for now
    if status_code == 415: # Content type
        return InvalidRequestError(message=err_msg, request=request, response=response, body=body) # Reusing InvalidRequest for now
    if status_code == 422:
         return UnprocessableEntityError(message=err_msg, request=request, response=response, body=body)
    if status_code == 429:
        # Parse Retry-After header for rate limit errors
        retry_after_header = response.headers.get("Retry-After")
        date_header = response.headers.get("Date")
        parsed_retry_after_seconds: Optional[int] = None
        if retry_after_header:
            parsed_retry_after_seconds = _parse_retry_after_header(retry_after_header, date_header)
        return RateLimitError(message=err_msg, request=request, response=response, body=body, retry_after_seconds=parsed_retry_after_seconds)
    # Only treat actual HTTP 5xx status codes as InternalServerError
    if 500 <= status_code < 600:
        return InternalServerError(message=err_msg, request=request, response=response, body=body) # Includes 500-599 only

    # Fallback for other 4xx errors
    if 400 <= status_code < 500:
         return APIError(message=f"Unhandled 4xx error: {err_msg}", request=request, response=response, body=body)

    # This is the catch-all fallback for non-standard status codes
    # Must return the base APIError class itself (not a subclass)
    return APIError(message=err_msg, request=request, response=response, body=body)