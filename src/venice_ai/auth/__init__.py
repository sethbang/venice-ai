"""Authentication helpers beyond simple Bearer tokens.

Exposes wallet-based auth for the ``/x402/*`` endpoints:

* :class:`venice_ai.auth.x402.X402Auth` — EVM (EIP-4361 SIWE + EIP-3009)
  flow. Requires the ``x402`` extra::

      pip install 'venice-py[x402]'

* :class:`venice_ai.auth.x402_solana.SolanaX402Auth` — Solana ("exact"
  SVM settlement) flow. Requires the ``x402-solana`` extra::

      pip install 'venice-py[x402-solana]'

Both names are imported lazily (PEP 562 ``__getattr__``) so that installing
only one extra does not force the other's optional dependencies — importing
``venice_ai.auth`` never pulls in eth-account/siwe or solders until the
corresponding class is actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .x402 import X402Auth
    from .x402_solana import SolanaX402Auth

__all__ = ["X402Auth", "SolanaX402Auth"]


def __getattr__(name: str) -> object:
    if name == "X402Auth":
        from .x402 import X402Auth

        return X402Auth
    if name == "SolanaX402Auth":
        from .x402_solana import SolanaX402Auth

        return SolanaX402Auth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
