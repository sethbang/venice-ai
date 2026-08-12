"""VCR Test Utilities and Compatibility Documentation.

This module provides documentation and utilities for VCR (Video Cassette Recorder)
test recording/replay, explaining how production code patterns work with VCR.

VCR is a test utility that records HTTP interactions and replays them during tests.
This enables fast, deterministic testing without requiring actual API calls.

## VCR Compatibility in Production Code

While production code should NOT contain test-specific logic, certain defensive
programming patterns naturally accommodate VCR's behavior:

### HTTP Response Streaming Challenges

VCR-recorded responses don't support true streaming because they're replayed from
static YAML files. Production code that handles various HTTP client behaviors
naturally accommodates this:

1. **Empty Content Fallback** (image.py, audio.py):
   - Production: Tries multiple methods to read response content
   - Why: Different HTTP clients/proxies may consume streams differently
   - VCR Benefit: Falls back to `_content` attribute when stream is consumed

2. **Full Body Fallback** (audio.py):
   - Production: If streaming fails, reads full body and chunks it
   - Why: Handles non-streaming responses gracefully
   - VCR Benefit: VCR responses work because they provide full body access

These patterns are good defensive programming that happen to work well with VCR,
rather than VCR-specific workarounds.

## VCR Configuration

VCR is configured in `tests/conftest.py` with:
- Cassette storage in `tests/integration/cassettes/` and `tests/e2e/cassettes/`
- Automatic API key scrubbing (security)
- Response body sanitization (removes sensitive data)
- Record mode: ONCE (record if missing, otherwise replay)

## Usage in Tests

```python
import pytest

@pytest.mark.integration
async def test_api_call(venice_client, vcr_cassette):
    '''Test with automatic HTTP recording/replay.'''
    with vcr_cassette:
        response = await venice_client.chat.completions.create(
            model="llama-3.2-3b",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.choices[0].message.content
```

## Re-recording Cassettes

When the Venice AI API changes, re-record cassettes:

```bash
# 1. Set API key
export VENICE_API_KEY="your-api-key"

# 2. Delete old cassettes
rm -rf tests/integration/cassettes/*.yaml
rm -rf tests/e2e/cassettes/*.yaml

# 3. Run tests to re-record
poetry run pytest tests/integration/ -v
poetry run pytest tests/e2e/ -v

# 4. Verify cassettes were created
ls tests/integration/cassettes/
ls tests/e2e/cassettes/

# 5. Commit updated cassettes
git add tests/*/cassettes/
git commit -m "chore: update VCR cassettes for API changes"
```

## CI/CD Considerations

- Set `VENICE_CI_MODE=true` to prevent cassette recording in CI
- Cassettes are committed to repository (after sanitization)
- No API key needed to run tests with cassettes

## Best Practices

1. **Never reference VCR in production code** - Use defensive patterns instead
2. **Document why defensive patterns exist** - Explain HTTP client variability
3. **Keep VCR logic in test utilities** - This module and conftest.py
4. **Sanitize sensitive data** - Automatic via before_record_response hook
5. **Test cassette replay** - Verify tests work without API key

## Troubleshooting

**Issue**: Tests fail with VCR but work with real API
- **Cause**: VCR can't replay streaming responses properly
- **Fix**: Production code should handle this via fallback patterns

**Issue**: Cassettes contain sensitive data
- **Cause**: Sanitization hook not configured properly
- **Fix**: Check `before_record_response` in vcr_config (conftest.py)

**Issue**: Tests record in CI when they shouldn't
- **Cause**: VENICE_CI_MODE not set
- **Fix**: Add VENICE_CI_MODE=true to CI environment
"""

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def is_vcr_active() -> bool:
    """Check if VCR is currently active in the test environment.

    This function detects if VCR (pytest-recording) is loaded in the current
    Python process. This is useful for test utilities that need to know if
    HTTP interactions are being recorded/replayed.

    Returns:
        bool: True if VCR is active (vcr or pytest_recording in sys.modules),
              False otherwise.

    Note:
        Production code should NEVER use this function. It exists solely for
        test infrastructure and utilities.

    VCR Record Modes:
        VCR does not expose the current record mode at runtime. If you need to
        check or modify VCR behavior, access the VCR configuration directly in
        your conftest.py or test fixtures.

    Example:
        >>> # In test utilities only!
        >>> if is_vcr_active():
        ...     print("HTTP calls will be recorded/replayed")
    """
    return "vcr" in sys.modules or "pytest_recording" in sys.modules


# Note: No response wrapping or client modification utilities needed.
# Production code's defensive patterns naturally work with VCR.

__all__ = [
    "is_vcr_active",
]
