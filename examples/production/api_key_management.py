#!/usr/bin/env python3
"""
Venice AI SDK - Production API Key Management
==============================================

This example demonstrates production-ready API key management patterns.
Learn how to securely handle and rotate API keys in production environments.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage


class SecureKeyManager:
    """
    Production-ready API key manager with secure storage and rotation.

    Best Practices:
    - Never hardcode API keys
    - Use environment variables or secret managers
    - Support key rotation without downtime
    - Log key usage for audit trails
    - Validate keys before use
    """

    def __init__(self, env_var: str = "VENICE_API_KEY"):
        self.env_var = env_var
        self._key: str | None = None
        self._key_metadata: dict = {}

    def load_key(self) -> str | None:
        """Load API key from environment variable."""
        key = os.getenv(self.env_var)
        if key:
            self._key = key
            self._key_metadata = {
                "loaded_at": datetime.now().isoformat(),
                "source": "environment",
                "env_var": self.env_var,
            }
            print(f"✅ API key loaded from {self.env_var}")
            return key
        return None

    def load_from_file(self, key_file: Path) -> str | None:
        """
        Load API key from secure file.

        Security Notes:
        - File should have restrictive permissions (0600)
        - Should not be in version control
        - Consider using encrypted storage
        """
        try:
            if not key_file.exists():
                print(f"❌ Key file not found: {key_file}")
                return None

            # Check file permissions (Unix-like systems)
            if hasattr(os, "stat"):
                stat_info = key_file.stat()
                mode = stat_info.st_mode & 0o777
                if mode != 0o600:
                    print(f"⚠️  Warning: Key file permissions are {oct(mode)}, should be 0600")

            # Load key
            key = key_file.read_text().strip()
            self._key = key
            self._key_metadata = {
                "loaded_at": datetime.now().isoformat(),
                "source": "file",
                "file_path": str(key_file),
            }
            print(f"✅ API key loaded from file: {key_file}")
            return key

        except Exception as e:
            print(f"❌ Failed to load key from file: {e}")
            return None

    def validate_key(self, key: str) -> bool:
        """
        Validate API key format.

        Add your own validation logic based on Venice AI key format.
        """
        if not key or len(key) < 10:  # noqa: SIM103 — keep extension point below
            return False

        # Add specific validation for Venice AI key format
        # Example: check prefix, length, character set, etc.

        return True

    def get_key(self) -> str | None:
        """Get the current API key."""
        return self._key

    def rotate_key(self, new_key: str) -> bool:
        """
        Rotate to a new API key.

        In production:
        1. Validate new key
        2. Test new key works
        3. Update configuration
        4. Invalidate old key
        """
        if not self.validate_key(new_key):
            print("❌ New key validation failed")
            return False

        old_key = self._key
        self._key = new_key
        self._key_metadata.update(
            {
                "rotated_at": datetime.now().isoformat(),
                "previous_key_prefix": old_key[:8] if old_key else None,
            }
        )

        print("✅ API key rotated successfully")
        return True


async def environment_variable_pattern() -> bool:
    """Demonstrate loading API key from environment variables.

    Returns ``True`` always — a missing env var is an expected local skip, not
    the code failure this example demonstrates.
    """
    print("🔐 Environment Variable Pattern")
    print("-" * 40)

    print("✅ Best Practice: Use environment variables")
    print()
    print("1. Set environment variable:")
    print("   export VENICE_API_KEY='your-api-key-here'")
    print()
    print("2. Load in application:")
    print("   ```python")
    print("   api_key = os.getenv('VENICE_API_KEY')")
    print("   if not api_key:")
    print("       raise ValueError('VENICE_API_KEY not set')")
    print("   client = VeniceClient(api_key=api_key)")
    print("   ```")
    print()

    # Demonstration
    manager = SecureKeyManager()
    key = manager.load_key()

    if key:
        print("✅ Key loaded successfully")
        print(f"   Key prefix: {key[:8]}...")
        print(f"   Metadata: {json.dumps(manager._key_metadata, indent=2)}")
    else:
        print("ℹ️  No key in environment (set VENICE_API_KEY to demonstrate)")

    return True


async def key_rotation_pattern() -> bool:
    """Demonstrate zero-downtime key rotation.

    Pure informational output — always returns ``True``.
    """
    print("\n🔄 Key Rotation Pattern")
    print("-" * 40)

    print("✅ Production Key Rotation Strategy:")
    print()
    print("1. Generate new API key in Venice AI dashboard")
    print()
    print("2. Test new key:")
    print("   ```python")
    print("   async def test_key(api_key: str) -> bool:")
    print("       try:")
    print("           client = VeniceClient(api_key=api_key)")
    print("           await client.models.list()  # Test call")
    print("           return True")
    print("       except AuthenticationError:")
    print("           return False")
    print("   ```")
    print()
    print("3. Deploy new key with rollback capability:")
    print("   - Update environment variable")
    print("   - Restart services gradually")
    print("   - Monitor error rates")
    print("   - Keep old key active during transition")
    print()
    print("4. Invalidate old key after confirmation:")
    print("   - Verify all services using new key")
    print("   - Disable old key in dashboard")
    print("   - Document rotation in audit log")

    return True


async def multi_environment_pattern() -> bool:
    """Demonstrate managing keys across environments.

    Pure informational output — always returns ``True``.
    """
    print("\n🌍 Multi-Environment Pattern")
    print("-" * 40)

    print("✅ Managing Keys Across Environments:")
    print()
    print("Development:")
    print("   - Use .env files (not in git)")
    print("   - Separate dev API key")
    print("   - Lower rate limits acceptable")
    print()
    print("Staging:")
    print("   - Use secret manager (AWS Secrets, Vault)")
    print("   - Production-like key with limits")
    print("   - Test rotation procedures")
    print()
    print("Production:")
    print("   - Use secret manager (required)")
    print("   - Dedicated high-limit key")
    print("   - Automated rotation")
    print("   - Comprehensive monitoring")
    print()

    print("💡 Example Multi-Environment Setup:")
    print("   ```python")
    print("   def get_api_key(environment: str) -> str:")
    print("       if environment == 'development':")
    print("           return os.getenv('VENICE_API_KEY_DEV')")
    print("       elif environment == 'staging':")
    print("           return get_from_secret_manager('staging/venice-api-key')")
    print("       elif environment == 'production':")
    print("           return get_from_secret_manager('prod/venice-api-key')")
    print("       else:")
    print("           raise ValueError(f'Unknown environment: {environment}')")
    print("   ```")

    return True


async def security_best_practices() -> bool:
    """Demonstrate security best practices for API key management.

    Pure informational output — always returns ``True``.
    """
    print("\n🛡️ Security Best Practices")
    print("-" * 40)

    print("✅ Critical Security Rules:")
    print()
    print("1. ❌ NEVER commit keys to version control")
    print("   - Add .env to .gitignore")
    print("   - Scan for accidentally committed keys")
    print("   - Use git-secrets or similar tools")
    print()
    print("2. ✅ Use environment-specific keys")
    print("   - Separate keys per environment")
    print("   - Limit permissions by environment")
    print("   - Easy to rotate if compromised")
    print()
    print("3. ✅ Implement key rotation")
    print("   - Regular rotation schedule (90 days)")
    print("   - Automated rotation process")
    print("   - Zero-downtime rotation")
    print()
    print("4. ✅ Monitor and audit key usage")
    print("   - Log all API key operations")
    print("   - Alert on unusual usage patterns")
    print("   - Track key lifecycle events")
    print()
    print("5. ✅ Secure key storage")
    print("   - Use secret managers (AWS Secrets, Vault)")
    print("   - Encrypt at rest")
    print("   - Restrict access (IAM policies)")
    print()
    print("6. ✅ Handle key compromise")
    print("   - Immediate revocation procedure")
    print("   - Generate new key")
    print("   - Audit for damage")
    print("   - Update affected systems")

    return True


async def testing_with_keys() -> bool:
    """Demonstrate testing patterns that don't expose real keys.

    Pure informational output — always returns ``True``.
    """
    print("\n🧪 Testing Patterns")
    print("-" * 40)

    print("✅ Safe Testing Practices:")
    print()
    print("1. Use Test Keys:")
    print("   - Dedicated test API keys")
    print("   - Lower rate limits")
    print("   - Separate billing")
    print()
    print("2. Mock in Unit Tests:")
    print("   ```python")
    print("   @patch('os.getenv')")
    print("   def test_client_creation(mock_getenv):")
    print("       mock_getenv.return_value = 'test-key'")
    print("       client = VeniceClient(api_key=os.getenv('VENICE_API_KEY'))")
    print("       assert client is not None")
    print("   ```")
    print()
    print("3. Use VCR for Integration Tests:")
    print("   - Record real API responses")
    print("   - Replay without real key")
    print("   - Sanitize recordings")
    print()
    print("4. CI/CD Secrets:")
    print("   - Store in CI secret manager")
    print("   - Inject at runtime")
    print("   - Never log key values")

    return True


async def production_client_pattern() -> bool:
    """Demonstrate production-ready client initialization.

    Returns ``True`` on success (or a clean no-key skip), ``False`` if the live
    API call failed — so a real failure surfaces instead of being swallowed.
    """
    print("\n🏭 Production Client Pattern")
    print("-" * 40)

    # Load API key securely
    manager = SecureKeyManager()
    api_key = manager.load_key()

    if not api_key:
        print("ℹ️  No API key available — skipping live client demo")
        return True

    # Production client with proper error handling
    try:
        async with VeniceClient(api_key=api_key) as client:
            print("✅ Production client initialized")

            # Validate key works
            chat_model = await client.models.resolve_chat()

            # Test request
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content="Health check")],
                max_completion_tokens=5,
            )

            print("✅ Client validated successfully")
            print(f"   Model: {chat_model}")
            print(f"   Response: {response.text}")

    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        print("   Check API key validity")
        print("   Verify network connectivity")
        print("   Review error logs")
        return False

    return True


async def main() -> int:
    """Run all API key management examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("=" * 60)
    print("Venice AI SDK - Production API Key Management")
    print("=" * 60)

    results: list[tuple[str, bool]] = [
        ("environment_variable_pattern", await environment_variable_pattern()),
        ("key_rotation_pattern", await key_rotation_pattern()),
        ("multi_environment_pattern", await multi_environment_pattern()),
        ("security_best_practices", await security_best_practices()),
        ("testing_with_keys", await testing_with_keys()),
        ("production_client_pattern", await production_client_pattern()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 60)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("🔑 Key Takeaways:")
    print("   1. Never hardcode API keys")
    print("   2. Use environment variables or secret managers")
    print("   3. Implement regular key rotation")
    print("   4. Monitor and audit key usage")
    print("   5. Have a key compromise response plan")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
