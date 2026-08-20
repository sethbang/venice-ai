# Billing — `client.billing.*`

Sourced from `src/venice_ai/resources/billing.py` and `src/venice_ai/types/api/billing.py`. The billing surface covers prepaid USD/DIEM balances, per-call usage records, and (beta) aggregated analytics. Everything below is for traditional API-key accounts; x402 wallet billing is a separate surface — see `venice-py-x402/references/balance-and-topup.md`.

## `get_balance` — current USD/DIEM headroom

```python
balance = await client.billing.get_balance()
# BillingBalanceResponse:
#   balance.can_consume          : bool | None
#   balance.consumption_currency : Literal["USD", "VCU", "DIEM", "BUNDLED_CREDITS"] | None
#   balance.balances             : BillingBalances | None    ← NESTED, can be None
#   balance.diem_epoch_allocation: float | None
# BillingBalances:
#   balance.balances.usd  : float | None
#   balance.balances.diem : float | None
```

**The `.balances.` nesting is the trap.** `balance.usd` does not exist — agents who flatten the access path get an `AttributeError` (or worse, a silent `None` from a defensive `getattr`). The wire fields are camelCase (`canConsume`, `consumptionCurrency`, `diemEpochAllocation`), but `populate_by_name=True` lets you read snake_case attributes.

```python
balance = await client.billing.get_balance()
print(f"Can consume: {balance.can_consume}")
if balance.balances:
    print(f"USD:  {balance.balances.usd}")
    print(f"DIEM: {balance.balances.diem}")
```

`balance_info` on response objects (`response.balance_info.usd`) is a *different* shape — that one is flat and represents the post-call remaining balance, sourced from response headers. Don't confuse them.

## `get_usage_history` — cursor-paginated per-call usage records

```python
page = await client.billing.get_usage_history(
    startTimestamp="2026-04-01T00:00:00Z",           # first page only
    endTimestamp="2026-05-01T00:00:00Z",             # first page only
    currency="USD",                                  # "USD" | "DIEM" | "BUNDLED_CREDITS"; None for all
    pageSize=1000,                                    # 10..1000, default 1000
)
# BillingUsageHistoryResponse:
#   page.data       : list[BillingUsageEntry]         ← ascending timestamp order
#   page.nextCursor : str | None                      ← None on the last page
```

The endpoint is a **keyset walk**: the first request takes the filters, and every response carries a `nextCursor`. A continuation request sends **only** the cursor — the filters travel inside it, and the server rejects filters supplied alongside a cursor (the SDK raises `ValueError` before the request if you try):

```python
if page.nextCursor:
    page = await client.billing.get_usage_history(cursor=page.nextCursor)
```

Set `format=BillingFormatEnum.CSV` to receive raw bytes instead of the typed object — handy for dumping straight to a file (the next-page token then rides in the `x-next-cursor` response header rather than the body).

Billing endpoints can be slow; the SDK wraps each request in a 10-second timeout that surfaces as `BillingTimeoutError` — widen the range and retry.

For unbounded enumeration use the paginator helper, which threads the cursor for you:

```python
async for entry in client.billing.iter_usage_history(currency="USD", page_size=1000):
    print(entry.timestamp, entry.amount, entry.sku)
```

## `get_usage_analytics` — beta aggregate dashboard

```python
analytics = await client.billing.get_usage_analytics(lookback="30d")
# OR specify both endpoints:
analytics = await client.billing.get_usage_analytics(
    startDate="2026-04-01", endDate="2026-05-01",      # YYYY-MM-DD here, NOT ISO 8601
)
```

This wraps a **beta** endpoint — schema and behavior may change. Returns aggregates by date, model, and API key. Source: `Billing.get_usage_analytics` in `src/venice_ai/resources/billing.py`.

## When to read which method

- **`get_balance`** — pre-flight check before a long batch ("do we have headroom?"). Cheap (one call). Don't poll it tightly; the value moves with every paid response.
- **`response.balance_info`** — post-call balance from response headers. Free (no extra request) but only present when the server emits the header (typically prepaid accounts). Use this for a running tally during a session.
- **`get_usage_history` / `iter_usage_history`** — historical reconciliation, per-call audit, generating invoices. Heavier — walk the cursor.
- **`get_usage_analytics`** — dashboards. Beta.

## Pitfalls

- **Reading `balance.usd`** instead of `balance.balances.usd` — the nesting is real.
- **Polling `get_balance` after every call** — read `response.balance_info` instead; it's emitted on the same request.
- **Resending filters with a cursor** on `get_usage_history` — a continuation carries the cursor alone; filters alongside it are rejected. The `iter_usage_history` helper handles this for you.
- **Confusing this with x402** — different surface, different shape. `client.x402.balance(...)` returns a `data`-envelope shape (see `venice-py-x402/references/balance-and-topup.md`).

## See also

- `venice-py/references/response-shapes.md` — quick cross-reference for all balance / usage shapes
- `venice-py-production/references/cost-tracking.md` — the `CostTracker` / `BudgetManager` layer that consumes balance info automatically
- `venice-py-x402/references/balance-and-topup.md` — wallet-funded billing surface, distinct from this
