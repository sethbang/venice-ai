# Wallet security for Venice x402 agents

The wallet IS the agent's identity AND its payment instrument. Compromise = unauthorized AI calls + drained USDC. Treat the private key like a production credential — most "convenience" patterns are wrong.

## What's at stake

A leaked `WALLET_PRIVATE_KEY` lets the attacker:
- Sign EIP-3009 USDC transfers up to the wallet's balance (drainable in a single transaction)
- Sign SIWE tokens authenticating as your wallet to Venice (consume your prepaid ledger)
- Execute any other on-chain action the wallet is authorized for

There is no "Venice-only" key — the same private key works for any signing operation on the same address.

## Operational baseline

| Practice | Why |
|---|---|
| **Never commit the key to git.** Use `.env` files (gitignored) or a secret manager. | A `.env` in a public repo is the most common leak vector. |
| **Separate wallets per environment.** Dev wallet, staging wallet, prod wallet — all different. | Limits blast radius. The `VENICE_X402_TEST_*` env-var convention exists for exactly this. |
| **Limit wallet balance to operational need.** Top up small amounts often, not large amounts rarely. | Reduces loss on compromise. |
| **Rotate keys on suspected exposure.** A new wallet, transfer balance, update env vars. | Treat any "did I commit that?" doubt as a yes. |
| **Audit on-chain history.** Periodically check the wallet's transaction log. | Detects unauthorized activity. |

## Storage options

### Bad

- Plaintext in source code or comments — `# private key: 0x...` is a leak.
- `~/.bashrc` / `~/.zshrc` exports — leaks via shell history, screen-shares, system backups.
- `.env` committed to git (even if "the repo is private now") — git history is forever.
- Slack DMs / Notion docs / shared drives — none are designed for credentials.

### OK

- `.env` files **gitignored** and never committed. Acceptable for dev / single-user. Watch out for IDE indexing, cloud sync, and "show me your shell" screen-shares.
- `direnv` with `.envrc` — same caveats as `.env`; gitignore the file.

### Better

- OS keychain (macOS Keychain, Linux Secret Service, Windows Credential Manager) — accessed via `keyring` Python package or platform CLI.
- Cloud secret managers (AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, HashiCorp Vault, 1Password Connect, Doppler) — the right tool for production / CI.
- Hardware wallets (Ledger, Trezor) — for high-value wallets, the key never leaves the device. Signing requires explicit confirmation. Trade-off: you can't fully automate.

### Best (production)

- A signing service running in a TEE / confidential VM, with the private key wrapped to the enclave. The agent calls the signing service; the key never appears in the agent's process memory.
- KMS-managed keys with strict per-operation IAM (e.g., AWS KMS with a usage policy that restricts which addresses you can transfer to).

The right level of paranoia scales with the wallet balance. A test wallet with $11 USDC: `.env` is fine. A prod wallet running an autonomous agent business: KMS or TEE.

## Handling the key in code

```python
import os

private_key = os.environ.get("WALLET_PRIVATE_KEY")
if not private_key:
    raise RuntimeError("WALLET_PRIVATE_KEY is not set; refusing to run without explicit credential")
```

Things to avoid:

- **Logging the key.** Even at `DEBUG` level, even "just for testing". Log filters fail; logs end up in shared sinks.
- **Embedding the key in error messages.** A traceback that includes `WALLET_PRIVATE_KEY=0x...` will end up in Sentry / Datadog / a customer support ticket.
- **Passing the key as a command-line arg.** `ps aux` shows command-line args.
- **Caching the key in a long-lived module-global.** Limits the window where it's in memory; better to load on demand.
- **Sending the key in HTTP requests.** Even to "just a debug endpoint."

The signing operation needs the key only at the moment of signing — load, sign, drop the reference:

```python
def sign_topup(payment_header_request: dict) -> str:
    private_key = os.environ["WALLET_PRIVATE_KEY"]
    try:
        return _build_and_sign(payment_header_request, private_key)
    finally:
        # In Python, you can't truly zero a string in memory, but you can
        # at least drop the reference so it's eligible for GC.
        private_key = None
```

(Python's string immutability means the bytes remain in memory until GC. For real key-zeroing semantics, use `bytearray` or a TEE-backed signing service.)

## Splitting auth and payment keys

You can use DIFFERENT wallets for SIWE auth (read-only ledger queries) and for paying topups. The auth wallet only needs SIWE-signing capability; the payment wallet holds USDC.

**Why bother?** Limits blast radius. A compromised SIWE-only key can read your ledger and consume your prepaid balance, but can't transfer USDC out of the payment wallet.

Practical setup:

```python
# SIWE-auth-only wallet — no USDC, no on-chain risk
auth_pk = os.environ["VENICE_AUTH_PRIVATE_KEY"]
auth = X402Auth(private_key=auth_pk)

# Payment wallet — holds USDC, only signs topup payments
payment_pk = os.environ["VENICE_PAYMENT_PRIVATE_KEY"]

# Reads use auth wallet
balance = await client.x402.balance(auth=auth)

# Topups use payment wallet
header = sign_topup_with(private_key=payment_pk, ...)
result = await client.x402.top_up(payment_header=header)
```

**Caveat**: each wallet has its own SIWE identity and prepaid ledger on Venice. The payment wallet's USDC tops up the payment wallet's ledger; the auth wallet has a separate ledger (which would need its own topups for SIWE-auth chat calls). For most use cases, one wallet is simpler. Two-wallet split is a defense-in-depth pattern — only worthwhile when wallet balances are large.

## Topup pre-flight validation

The `balance-and-topup.md` validation checklist is also a security control:

```python
EXPECTED_NETWORK = "eip155:8453"
EXPECTED_USDC    = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
MAX_AMOUNT_UNITS = 10_000_000

assert req["network"] == EXPECTED_NETWORK
assert req["asset"].lower() == EXPECTED_USDC.lower()
assert int(req["amount"]) <= MAX_AMOUNT_UNITS
```

A compromised TLS connection or DNS hijack could redirect your topup request to an attacker-controlled `payTo` address. The validation against expected network/asset/amount caps catches that — within reason. (It can't catch a `payTo` swap if you don't pin the address, but most attacks don't have that level of control.)

Pinning `payTo` IS reasonable for fully-deterministic deployments — hardcode Venice's known settlement address and abort if the 402 returns anything else:

```python
EXPECTED_PAYTO = "0x2670B922ef37C7Df47158725C0CC407b5382293F"  # validated against Venice docs
assert req["payTo"].lower() == EXPECTED_PAYTO.lower(), f"unexpected payTo {req['payTo']}"
```

The risk is that Venice rotates its settlement address — your script breaks until you update. Acceptable trade-off for high-value automation.

## Funding hygiene

- **Don't fund an agent wallet from an exchange directly.** Exchange withdrawals are KYC'd; don't link your identity to your bot's wallet unnecessarily. Use an intermediate hot wallet.
- **Don't fund with more than the agent will spend in a week.** Top-up cycles cost a Base transaction fee each time (~$0.001), so weekly is a reasonable cadence.
- **Set up monitoring.** Alert on unexpected balance drops or transactions you didn't initiate.

## Compromised-key playbook

If you suspect the key is compromised:

1. **Immediately move funds.** From a different machine, sign a USDC transfer to a fresh wallet. Don't use the compromised key for anything else.
2. **Rotate Venice config.** Generate a new wallet, update `WALLET_PRIVATE_KEY` env var, redeploy.
3. **Audit transaction history.** Block explorer (e.g., basescan.org/address/0x...) — review all transactions since the compromise window.
4. **Audit Venice ledger.** `client.x402.transactions(auth=auth)` will show consumption from the prepaid balance — useful for detecting unauthorized AI usage.
5. **Review log retention.** If logs / Sentry / Slack contain the key (or messages that hint at it), purge them.

## Common bugs

- **Committing `.env` to git** with `git add -A` after just adding `WALLET_PRIVATE_KEY=...`. `.gitignore` first, key second.
- **Logging the SIWE header** — it contains the wallet address but not the private key. Address leakage is OK; signature reuse risk is mitigated by per-message nonces. Logging the SIWE token within its TTL window is fine but be aware it temporarily authenticates the wallet.
- **Hardcoding `WALLET_PRIVATE_KEY` as a Python string literal** for "quick testing". Five minutes later, you commit the file.
- **Sharing a wallet across multiple agents** — limits per-agent attribution, complicates rotation, expands compromise blast radius.
- **Leaving funded test wallets running indefinitely** — sweep unused funds back to a safe wallet rather than leaving USDC idle on-chain in a test wallet.

## Related references

- `siwe-auth.md` — what `X402Auth` does and doesn't do (auth, not payment).
- `balance-and-topup.md` — the topup flow with validation checks.
- `agent-frameworks.md` — framework-specific key handling (Coinbase Agentkit, Eliza, etc.).
- General security guidance: see `senior-security` skill if you need broader threat modeling.
