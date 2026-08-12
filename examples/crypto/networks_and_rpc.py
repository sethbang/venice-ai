#!/usr/bin/env python3
"""
Venice AI SDK - Crypto JSON-RPC Proxy Examples
==============================================

Venice exposes a JSON-RPC 2.0 proxy over a set of supported blockchains via the
``client.crypto`` resource. This example demonstrates the full surface:

- ``client.crypto.networks()``   — discover the supported network slugs
- ``client.crypto.rpc(...)``      — forward a single read-only JSON-RPC call
- ``client.crypto.batch_rpc(...)``— forward 2-3 read-only calls in one batch

No models are involved — this is a thin pass-through to on-chain RPC nodes, so
there is no resolver call here (a chat/image model is never needed).

Key idea: ``networks()`` is *authoritative*. A slug taken from its return value
cannot 400 as "Unsupported RPC network", whereas a hardcoded guess can. So we
discover first, then feed the chosen slug into both ``rpc()`` and ``batch_rpc()``
— the crypto analog of resolver-based model selection.

Prerequisites:
- Install: pip install venice-ai
- Set API key: export VENICE_API_KEY="your-api-key"
  (``networks()`` is public, but ``rpc()``/``batch_rpc()`` bill credits and need a key.)
"""

import asyncio
import sys

from venice_ai import (
    PaymentRequiredError,
    PermissionDeniedError,
    VeniceClient,
)

# Read-only EVM methods that take no params and are cheap to call. These are the
# canonical "is the chain reachable" probes — safe to run repeatedly.
EVM_READ_METHODS = ("eth_chainId", "eth_blockNumber", "eth_gasPrice")


def _select_evm_network(networks: list[str]) -> str | None:
    """Pick an EVM-compatible network slug from the authoritative list.

    The read-only methods used below (``eth_*``) are EVM-specific, so we must not
    land on a non-EVM chain (e.g. a Solana slug). We prefer Ethereum mainnet, then
    any Ethereum-flavored slug, then any slug whose name hints at EVM compatibility.
    Returns ``None`` if nothing suitable is found.
    """
    if not networks:
        return None
    # Exact mainnet match first.
    if "ethereum-mainnet" in networks:
        return "ethereum-mainnet"
    # Any Ethereum-flavored slug next.
    for slug in networks:
        if "ethereum" in slug.lower():
            return slug
    # Fall back to other common EVM chains by name hint.
    evm_hints = ("base", "arbitrum", "optimism", "polygon", "bsc", "avalanche")
    for slug in networks:
        if any(hint in slug.lower() for hint in evm_hints):
            return slug
    return None


async def list_networks(client: VeniceClient) -> tuple[bool, list[str]]:
    """Discover the supported crypto RPC networks.

    Returns ``(ok, networks)``. ``networks()`` is a public endpoint, so this is
    the most likely demo to succeed regardless of account entitlements.
    """
    print("🌐 Supported Crypto RPC Networks")
    print("-" * 40)

    try:
        networks = await client.crypto.networks()
    except Exception as e:  # noqa: BLE001 - surface any discovery failure honestly
        print(f"❌ Failed to list networks: {e}")
        return False, []

    print(f"✅ Proxy supports {len(networks)} network(s):")
    for slug in networks:
        print(f"   • {slug}")
    return True, networks


async def single_rpc_call(client: VeniceClient, network: str) -> bool:
    """Forward a single read-only JSON-RPC call (``eth_chainId``).

    On success returns ``True``. If the crypto proxy is not enabled on this
    account (402/403), degrades to a clear skip and returns ``True`` so the demo
    does not hard-fail on an entitlement gap. Any other error returns ``False``.
    """
    print("\n🔗 Single JSON-RPC Call")
    print("-" * 40)
    print(f"📍 Network: {network}")
    print("🛠️  Method: eth_chainId (read-only, no params)")

    try:
        # rpc() takes the network slug, a JSON-RPC method, optional params and id.
        # eth_chainId returns the chain ID as a hex string (e.g. "0x1" for mainnet).
        resp = await client.crypto.rpc(
            network=network,
            method="eth_chainId",
            params=[],
            id=1,
        )
    except (PaymentRequiredError, PermissionDeniedError) as e:
        # Crypto proxy not enabled / out of credits on this account — not a bug.
        print(f"⏭️  Skipping: crypto RPC not available on this account ({type(e).__name__}).")
        return True
    except Exception as e:  # noqa: BLE001 - any other failure is a real failure
        print(f"❌ rpc() failed: {e}")
        return False

    # JSON-RPC returns HTTP 200 even for per-request errors, so check .error first
    # (.result can legitimately be None for some methods).
    if resp.error is not None:
        print(f"❌ JSON-RPC error {resp.error.code}: {resp.error.message}")
        return False

    chain_id_hex = resp.result
    print("✅ Call succeeded:")
    print(f"   id:       {resp.id}")
    print(f"   result:   {chain_id_hex}")
    if isinstance(chain_id_hex, str) and chain_id_hex.startswith("0x"):
        print(f"   decimal:  {int(chain_id_hex, 16)}")

    # The typed billing accessors are surfaced from the HTTP response headers.
    if resp.rpc_credits is not None:
        print(f"   credits:  {resp.rpc_credits}")
    if resp.rpc_cost_usd is not None:
        print(f"   cost USD: {resp.rpc_cost_usd:.8f}")
    return True


async def batch_rpc_call(client: VeniceClient, network: str) -> bool:
    """Forward a batch of 3 read-only JSON-RPC calls in a single request.

    Returns ``True`` on success, skips cleanly (``True``) on an entitlement gap,
    and returns ``False`` on any other failure.
    """
    print("\n📦 Batch JSON-RPC Call")
    print("-" * 40)
    print(f"📍 Network: {network}")
    print(f"🛠️  Methods: {', '.join(EVM_READ_METHODS)}")

    # Each request carries an id so we can correlate responses back to methods —
    # JSON-RPC does not guarantee that responses come back in request order.
    id_to_method = {i + 1: method for i, method in enumerate(EVM_READ_METHODS)}
    requests = [
        {"method": method, "params": [], "id": rpc_id} for rpc_id, method in id_to_method.items()
    ]

    try:
        batch = await client.crypto.batch_rpc(network=network, requests=requests)
    except (PaymentRequiredError, PermissionDeniedError) as e:
        print(f"⏭️  Skipping: crypto RPC not available on this account ({type(e).__name__}).")
        return True
    except Exception as e:  # noqa: BLE001 - any other failure is a real failure
        print(f"❌ batch_rpc() failed: {e}")
        return False

    print(f"✅ Batch returned {len(batch)} response(s):")
    ok = True
    # Correlate by id rather than position (ordering is not guaranteed).
    for item in batch:
        method = id_to_method.get(item.id, "<unknown>") if item.id is not None else "<unknown>"
        if item.error is not None:
            # A per-item RPC error does not fail the whole batch — report and continue.
            print(f"   • {method} (id={item.id}): ❌ error {item.error.code}: {item.error.message}")
            ok = False
        else:
            print(f"   • {method} (id={item.id}): {item.result}")

    # Billing headers cover the whole batch and live on the wrapper.
    if batch.rpc_credits is not None:
        print(f"   batch credits:  {batch.rpc_credits}")
    if batch.rpc_cost_usd is not None:
        print(f"   batch cost USD: {batch.rpc_cost_usd:.8f}")
    return ok


async def main() -> int:
    """Run the crypto RPC proxy demos.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner. Entitlement gaps degrade to skips (still counted as success).
    """
    print("🚀 Venice AI Crypto JSON-RPC Proxy Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = []

    async with VeniceClient() as client:
        # 1) Discover networks (authoritative — feeds the slug into the RPC calls).
        networks_ok, networks = await list_networks(client)
        results.append(("list_networks", networks_ok))

        network = _select_evm_network(networks)
        if network is None:
            # No EVM-compatible slug to exercise eth_* against — skip the RPC demos
            # cleanly rather than hard-fail. networks() itself still proves coverage.
            print("\n⏭️  No EVM-compatible network slug available; skipping rpc()/batch_rpc().")
        else:
            print(f"\n🎯 Selected network for RPC demos: {network}")
            # 2) Single JSON-RPC call.
            results.append(("single_rpc_call", await single_rpc_call(client, network)))
            # 3) Batch JSON-RPC call.
            results.append(("batch_rpc_call", await batch_rpc_call(client, network)))

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️  {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Crypto RPC proxy examples completed!")

    print("\n💡 Key concepts demonstrated:")
    print("   - Discovering supported chains with client.crypto.networks()")
    print("   - Single read-only JSON-RPC via client.crypto.rpc()")
    print("   - Batched read-only JSON-RPC via client.crypto.batch_rpc()")
    print("   - Checking .error before .result; correlating batch items by id")
    print("   - Reading typed billing headers (.rpc_credits / .rpc_cost_usd)")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have internet connection.", file=sys.stderr)
        sys.exit(1)
