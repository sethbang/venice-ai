from typing import Optional, Any
import httpx

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
    """
    def __init__(self, message: str, *, request: Optional[httpx.Request] = None, response: httpx.Response, body: Optional[Any] = None) -> None:
        super().__init__(message, request=request, response=response, body=body)

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
    
    # Ensure body is a dictionary for consistent error handling
    # If body is not a dictionary (e.g., it's a string from a non-JSON response),
    # wrap it in a structured dictionary format
    if body is not None and not isinstance(body, dict):
        raw_text_content = str(body)
        body = {
            "error": f"Non-JSON response from API (status {status_code}): {raw_text_content}"
        }
    
    # Try to parse error details from response body if available
    # Assuming error structure like {"error": {"code": "...", "message": "..."}} or similar
    # Adjust parsing based on actual error response structure
    http_status_prefix = f"HTTP Status {status_code}"
    # Ensure the http_status_prefix is always present, then append the more specific message if provided.
    err_msg = f"{http_status_prefix}: {message}" if message else http_status_prefix
    err_code = None
    if body and isinstance(body, dict):
        error_data = body.get("error")
        if isinstance(error_data, dict):
            detail_from_dict = error_data.get("message") or error_data.get("detail")
            code_from_dict = error_data.get("code")
            
            if detail_from_dict:
                err_msg = f"{err_msg}: {detail_from_dict}"
            
            if code_from_dict:
                # Append code, ensuring it's added even if detail_from_dict was None but code exists
                # Or if detail_from_dict was present, it appends after the detail.
                err_msg = f"{err_msg} (Code: {code_from_dict})"
                
        elif isinstance(error_data, str):
            # If error_data is a string, append it directly as the detail
            err_msg = f"{err_msg}: {error_data}"

    if status_code == 400:
        return InvalidRequestError(err_msg, request=request, response=response, body=body)
    if status_code == 401:
        return AuthenticationError(err_msg, request=request, response=response, body=body)
    if status_code == 403:
        return PermissionDeniedError(err_msg, request=request, response=response, body=body)
    if status_code == 404:
        # Always return NotFoundError for 404 status codes
        # This is needed to match the test expectation in test_make_status_error_status_codes
        return NotFoundError(err_msg, request=request, response=response, body=body)
    if status_code == 409:
         return ConflictError(err_msg, request=request, response=response, body=body)
    if status_code == 413: # File size
        return InvalidRequestError(err_msg, request=request, response=response, body=body) # Reusing InvalidRequest for now
    if status_code == 415: # Content type
        return InvalidRequestError(err_msg, request=request, response=response, body=body) # Reusing InvalidRequest for now
    if status_code == 422:
         return UnprocessableEntityError(err_msg, request=request, response=response, body=body)
    if status_code == 429:
        return RateLimitError(err_msg, request=request, response=response, body=body)
    # Only treat actual HTTP 5xx status codes as InternalServerError
    if 500 <= status_code < 600:
        return InternalServerError(err_msg, request=request, response=response, body=body) # Includes 500-599 only

    # Fallback for other 4xx errors
    if 400 <= status_code < 500:
         return APIError(f"Unhandled 4xx error: {err_msg}", request=request, response=response, body=body)

    # This is the catch-all fallback for non-standard status codes
    # Must return the base APIError class itself (not a subclass)
    return APIError(err_msg, request=request, response=response, body=body)