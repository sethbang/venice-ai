# SIWE auth for Venice (mode-2 deep dive)

Sourced from `src/venice_ai/auth/x402.py` — and validated end-to-end against `api.venice.ai`.

## What `X402Auth` actually does

`X402Auth(private_key=...)` produces a Sign-In-With-X (SIWE / EIP-4361) header that proves wallet ownership to Venice. Venice validates the signature server-side and treats the wallet as the authenticated principal. The header goes in `X-Sign-In-With-X` (NOT `Authorization: Bearer`).

**`X402Auth` builds two distinct signatures.** It signs the SIWE `X-Sign-In-With-X` auth header via `build_header()` for read-only ledger queries (`balance`, `transactions`) and for SIWE-authenticated chat/image/etc. requests when you have prepaid balance. It also signs the EIP-3009 `X-402-Payment` payment envelope via `build_payment_header(requirement, ...)` for top-ups. The two are different signatures (EIP-191/SIWE over a sign-in message vs EIP-712 `transferWithAuthorization` over a USDC transfer) but both live on `X402Auth`; the high-level `client.x402.top_up_with(...)` wraps the payment side (see `balance-and-topup.md` and the SKILL.md mode-3 example).

## Constructor

```python
from venice_ai.auth.x402 import X402Auth

auth = X402Auth(
    private_key=os.environ["WALLET_PRIVATE_KEY"],   # required — 0x-prefixed or bare 64 hex chars
    chain_id=8453,                                  # default — Base mainnet
    ttl_seconds=600,                                # default — 10 min SIWE TTL
)
```

**There is no `wallet_address=` kwarg.** The address is derived from the private key. Read it via the `auth.wallet_address` property (returns the EIP-55 checksummed address):

```python
print(auth.wallet_address)        # "0x1234567890AbCdeF1234567890abCDeF12345678"
```

If you have the private key, you have the address — that's why passing both is redundant.

## Header lifecycle

`auth.build_header(nonce=None, now=None)` produces the base64-encoded `X-Sign-In-With-X` header value. The structure (decoded):

```json
{
  "address":   "0x1234567890AbCdeF1234567890abCDeF12345678",
  "message":   "outerface.venice.ai wants you to sign in...\n...nonce: <hex>...",
  "signature": "0x<hex>",
  "timestamp": 1714939200000,
  "chainId":   8453
}
```

The SIWE message conforms to EIP-4361. The signature is over the EIP-191 personal_sign hash of the message. Server-side, Venice (a) validates the signature recovers to `address`, (b) checks the message hasn't expired (per `expiration_time`), (c) checks `domain` matches `outerface.venice.ai`.

### Token TTL and caching

The SIWE token is good for `ttl_seconds` (default 600). **Cache and reuse it within that window** — re-signing on every call wastes ECDSA work and adds latency.

```python
import time

class CachedSIWE:
    """Cache a SIWE header within its expiry window."""
    def __init__(self, auth: X402Auth):
        self.auth = auth
        self._header: str | None = None
        self._expires_at: float = 0.0

    def header(self) -> str:
        if self._header is None or time.time() >= self._expires_at - 30:    # 30s safety margin
            self._header = self.auth.build_header()
            # build_header generates a fresh nonce + issuedAt; use ttl_seconds for expiry estimate
            self._expires_at = time.time() + self.auth._ttl_seconds          # private but stable
        return self._header
```

Or, if your usage patterns are bursty, just construct a new `X402Auth` per session and let it sign fresh tokens lazily — the overhead is real but small (~10ms per call on commodity hardware).

### When NOT to cache

If your wallet is multi-signing (rotating signers, hardware module rotation), don't cache — sign per call so each signature is from the current key. For single-key bots, cache freely.

## Calling Venice via SIWE (mode 2 in practice)

Validated flow:

```python
import asyncio, os
import aiohttp
from venice_ai.auth.x402 import X402Auth


async def chat_via_siwe(question: str) -> str:
    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
    sign_in_header = auth.build_header()

    async with aiohttp.ClientSession() as http:
        # Models catalog accepts SIWE auth (free read)
        async with http.get(
            "https://api.venice.ai/api/v1/models",
            headers={"X-Sign-In-With-X": sign_in_header},
            params={"type": "text"},
        ) as r:
            r.raise_for_status()
            data = await r.json()
            model_id = data["data"][0]["id"]

        # Chat completion — SIWE auth, debited from prepaid balance
        async with http.post(
            "https://api.venice.ai/api/v1/chat/completions",
            headers={
                "X-Sign-In-With-X": sign_in_header,           # auth
                "Content-Type": "application/json",
                # NO Authorization: Bearer header
            },
            json={"model": model_id, "messages": [{"role": "user", "content": question}]},
        ) as r:
            r.raise_for_status()
            data = await r.json()
            return data["choices"][0]["message"]["content"]
```

Validated 2026-05-05 against `api.venice.ai`: the call succeeded with `X-Sign-In-With-X` alone, debited the prepaid ledger by ~$0.0015 for a small completion, and returned a normal `chat.completions` response.

## Use `VeniceClient(auth=X402Auth(...))` for SIWE-only mode

On SDK ≥ 2.0.0, the cleanest Mode-2 path is to pass `auth=` directly to `VeniceClient`:

```python
async with VeniceClient(auth=X402Auth(private_key=...)) as client:
    response = await client.chat.completions.create(...)
```

The SDK skips the `Authorization: Bearer` header (since no `api_key` is provided) and attaches a cached `X-Sign-In-With-X` header on every request. The token is cached for `auth.ttl_seconds - 30s` (safety margin) so repeated calls don't waste signing roundtrips.

When both `api_key=` and `auth=` are passed, the API key wins for default request auth; the auth instance is retained so callers can still pass it explicitly to per-call `auth=` kwargs (e.g., `client.x402.balance(auth=auth)`).

## Pre-2.0 fallback: drop to aiohttp / httpx

If you can't upgrade past SDK 2.0.0, the constructor used to require `api_key` and the only mode-2 path was to drop to a raw HTTP client. The pattern looked like:

```python
auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
sign_in_header = auth.build_header()    # base64 SIWE token, valid for ttl_seconds (default 600)

async with aiohttp.ClientSession() as http:
    # 1. Discover a chat model (the public catalog accepts SIWE auth)
    async with http.get(
        "https://api.venice.ai/api/v1/models",
        headers={"X-Sign-In-With-X": sign_in_header},
        params={"type": "text"},
    ) as r:
        r.raise_for_status()
        data = await r.json()
        model_id = data["data"][0]["id"]

    # 2. Chat completion via SIWE — no Authorization: Bearer header
    async with http.post(
        "https://api.venice.ai/api/v1/chat/completions",
        headers={"X-Sign-In-With-X": sign_in_header, "Content-Type": "application/json"},
        json={"model": model_id, "messages": [{"role": "user", "content": question}]},
    ) as r:
        r.raise_for_status()
        data = await r.json()
        return data["choices"][0]["message"]["content"]
```

You'd cache the header yourself within its TTL window and refresh on expiry. On modern SDKs this is unnecessary — the `auth=` constructor param handles all of it.

## What goes wrong if SIWE fails

Common error responses from `outerface.venice.ai`:

| Server response | Meaning |
|---|---|
| 401 with `code: "INVALID_SIGNATURE"` | Signature didn't recover to the claimed address. Check the private key. |
| 401 with `code: "EXPIRED_SIGNATURE"` | The SIWE message's `expirationTime` is in the past. Refresh the header. |
| 401 with `code: "INVALID_CHAIN_ID"` | `chainId` in the header doesn't match Venice's expected chain. Default 8453 (Base). |
| 401 with `code: "INVALID_DOMAIN"` | The SIWE message's `domain` field is not `outerface.venice.ai`. Don't override the domain. |
| 402 with structured `topUpInstructions` | Auth succeeded; prepaid balance is exhausted. Top up — see `balance-and-topup.md`. |

## Common bugs

- **Passing `wallet_address=` to `X402Auth`** — there's no such kwarg. The address is derived.
- **Caching a `X402Auth` instance for >`ttl_seconds` and reusing the same `build_header()` output** — the underlying SIWE message expires; you'll see 401s. Wrap in a `CachedSIWE`-style helper that refreshes.
- **Sending `Authorization: Bearer <api_key>` AND `X-Sign-In-With-X`** — confuses the server's auth pipeline. Pick one. (For `client.x402.balance`, the SDK sends the API key for the HTTP request and SIWE in `X-Sign-In-With-X` to identify the wallet — that's correct because the read endpoint requires API-key access to the route, plus SIWE to scope to a wallet. Don't try to reason about it from first principles; trust the SDK there.)
- **Signing the EIP-191 message manually** when you have `X402Auth` already — that class does the right thing.
- **Treating `auth.build_header()` output as plaintext JSON** — it's base64-encoded. Decode with `base64.b64decode(...)` then `json.loads(...)` if you want to inspect.

## Related references

- `balance-and-topup.md` — the EIP-3009 payment payload (different from SIWE auth).
- `agent-frameworks.md` — Coinbase Agentkit / Eliza / x402-axios alternatives that handle SIWE for you.
- `wallet-security.md` — managing the private key safely.
