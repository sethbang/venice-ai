import httpx

DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0) # 60s default timeout
DEFAULT_MAX_RETRIES = 2

# HTTP 503 retry configuration
MAX_HTTP_503_RETRIES = 1  # This means 1 retry, leading to a total of 2 attempts.
HTTP_503_RETRY_DELAY_SECONDS = 2  # Delay in seconds between retries.