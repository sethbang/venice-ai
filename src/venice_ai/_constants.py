import httpx

DEFAULT_BASE_URL = "https://api.venice.ai/api/v1"
DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0) # 60s default timeout
DEFAULT_MAX_RETRIES = 2