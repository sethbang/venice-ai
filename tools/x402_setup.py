"""
Generate a fresh wallet for x402 live testing. Writes the private key to
.env (gitignored) and echoes ONLY the public address to stdout.

NEVER prints, logs, or commits the private key. Idempotent: if a wallet
already exists in .env, prints the address and exits 0 without overwriting.

Usage:
  poetry run python tools/x402_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from eth_account import Account

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
KEY_LINE = "VENICE_X402_TEST_PRIVATE_KEY"
ADDR_LINE = "VENICE_X402_TEST_ADDRESS"


def read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def write_env(updates: dict[str, str]) -> None:
    existing: list[str] = []
    if ENV_PATH.exists():
        existing = ENV_PATH.read_text().splitlines()
    keys_in_file: set[str] = set()
    new_lines: list[str] = []
    for line in existing:
        k, _, _ = line.partition("=")
        k = k.strip()
        if k in updates:
            keys_in_file.add(k)
            new_lines.append(f"{k}={updates[k]}")
        else:
            new_lines.append(line)
    for k, v in updates.items():
        if k not in keys_in_file:
            new_lines.append(f"{k}={v}")
    ENV_PATH.write_text("\n".join(new_lines) + "\n")


def main() -> int:
    env = read_env()
    if env.get(KEY_LINE) and env.get(ADDR_LINE):
        print(f"Existing wallet detected: {env[ADDR_LINE]}")
        print("Refusing to overwrite. Delete the lines from .env if you want a fresh wallet.")
        return 0

    Account.enable_unaudited_hdwallet_features()
    acct = Account.create()
    addr = acct.address
    pkhex = acct.key.hex()
    write_env({KEY_LINE: pkhex, ADDR_LINE: addr})
    print("Wallet generated.")
    print(f"  Address (share OK):  {addr}")
    print("  Private key:         (written to .env, NEVER displayed)")
    print()
    print("Next: fund this address per Venice docs. Then run the live tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
