#!/usr/bin/env python3
"""
Venice AI SDK - Billing and Usage Analytics
===========================================

This example demonstrates how to retrieve and analyze billing data using the Venice AI SDK:
- Checking the current account balance (client.billing.get_balance)
- Getting usage data in different formats (JSON/CSV)
- Aggregated usage analytics by date/model/key (client.billing.get_usage_analytics, Beta)
- Analyzing costs and consumption patterns
- Understanding billing entries and inference details
- Working with cursor pagination for large datasets
- Exporting data for external analysis
"""

import asyncio
import sys
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import NotFoundError, PermissionDeniedError
from venice_ai.types import BillingFormatEnum
from venice_ai.types.api.billing import (
    BillingBalanceResponse,
    BillingUsageHistoryResponse,
    UsageAnalyticsResponse,
)

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/billing/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def get_recent_usage() -> bool:
    """Retrieve and analyze recent usage data."""
    print("📊 Recent Usage Analysis")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=30)

            start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

            print(f"📅 Analyzing usage from {start_date_str} to {end_date_str}")

            # Get usage history (first page of the cursor-paginated walk)
            usage_response = await client.billing.get_usage_history(
                format=BillingFormatEnum.JSON,
                startTimestamp=start_date_str,
                endTimestamp=end_date_str,
                currency="USD",
                pageSize=100,
            )
            assert isinstance(usage_response, BillingUsageHistoryResponse)

            usage_entries = usage_response.data
            has_more = usage_response.nextCursor is not None

            print("\n📋 Usage Summary:")
            print(f"   📦 Retrieved {len(usage_entries)} entries on this page")
            if has_more:
                print("   📄 More entries available — follow nextCursor for the rest")

            if usage_entries:
                # Handle as Pydantic objects (proper typed responses)
                # Venice API returns debits as negative amounts (balance deductions).
                # Filter to USD-only for the totals so both per-SKU and grand totals agree.
                usd_entries = [e for e in usage_entries if e.currency == "USD"]
                total_usd = abs(sum(e.amount for e in usd_entries))
                total_diem = sum(abs(e.amount) for e in usage_entries if e.currency == "DIEM")
                total_units = sum(e.units for e in usd_entries)

                print("\n💰 Cost Analysis (USD entries only):")
                print(f"   💵 Total USD: ${total_usd:.4f}")
                print(f"   📦 USD entries: {len(usd_entries)} of {len(usage_entries)}")
                if total_diem > 0:
                    print(f"   💎 Total DIEM (excluded from USD totals): {total_diem:.4f}")
                print(f"   📊 Total units consumed (USD entries): {total_units:.2f}")

                # Per-SKU breakdown — same currency filter as the totals above,
                # so the per-SKU sums add up to total_usd.
                product_analysis = {}
                for entry in usd_entries:
                    sku = entry.sku
                    amount = entry.amount
                    units = entry.units
                    currency = entry.currency

                    if sku not in product_analysis:
                        product_analysis[sku] = {
                            "count": 0,
                            "total_cost": 0.0,
                            "total_units": 0.0,
                            "currency": currency,
                        }
                    product_analysis[sku]["count"] += 1
                    product_analysis[sku]["total_cost"] += abs(
                        amount
                    )  # Use absolute value for display
                    product_analysis[sku]["total_units"] += units

                print("\n🏷️ Usage by Product SKU:")
                for sku, data in sorted(
                    product_analysis.items(), key=lambda x: x[1]["total_cost"], reverse=True
                ):
                    print(f"   📦 {sku}:")
                    print(f"      💰 Cost: ${data['total_cost']:.4f} {data['currency']}")
                    print(f"      📊 Units: {data['total_units']:.2f}")
                    print(f"      📈 Requests: {data['count']}")
                    print()

                # Show recent entries
                print("🕐 Most Recent Entries (last 5):")
                for i, entry in enumerate(usage_entries[:5], 1):
                    print(f"\n   {i}. {entry.timestamp}")
                    print(f"      🏷️ SKU: {entry.sku}")
                    print(f"      💰 Cost: ${entry.amount:.4f} {entry.currency}")
                    print(f"      📊 Units: {entry.units:.2f} @ ${entry.pricePerUnitUsd:.6f}/unit")
                    if entry.notes:
                        print(f"      📝 Notes: {entry.notes}")

            else:
                print("ℹ️ No usage data found for the specified period")
                print("💡 Try extending the date range or check if you've made any API calls")

            return True

        except Exception as e:
            print(f"❌ Error retrieving usage data: {e}")
            print("💡 Note: Billing data access requires appropriate API permissions")
            return False


async def analyze_inference_details() -> bool:
    """Analyze detailed inference information from billing data."""
    print("\n🔍 Inference Details Analysis")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get recent usage data with inference details
            end_date = datetime.now(UTC)
            start_date = end_date - timedelta(days=7)  # Last week

            usage_response = await client.billing.get_usage_history(
                format=BillingFormatEnum.JSON,
                startTimestamp=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                endTimestamp=end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                pageSize=50,
            )
            assert isinstance(usage_response, BillingUsageHistoryResponse)

            usage_entries = usage_response.data

            # Find entries with inference details
            inference_entries = [
                entry for entry in usage_entries if entry.inferenceDetails is not None
            ]

            if inference_entries:
                print(f"🧠 Found {len(inference_entries)} entries with inference details")

                # Analyze token usage
                total_input_tokens = 0
                total_output_tokens = 0
                total_requests = len(inference_entries)

                for entry in inference_entries:
                    details = entry.inferenceDetails
                    if details:
                        input_tokens = details.promptTokens or 0
                        output_tokens = details.completionTokens or 0

                        total_input_tokens += input_tokens
                        total_output_tokens += output_tokens

                print("\n📊 Token Usage Analysis:")
                print(f"   📥 Total input tokens: {total_input_tokens:,}")
                print(f"   📤 Total output tokens: {total_output_tokens:,}")
                print(f"   📋 Total inference requests: {total_requests}")

                if total_requests > 0:
                    avg_input = total_input_tokens / total_requests
                    avg_output = total_output_tokens / total_requests
                    print(f"   📊 Average input tokens per request: {avg_input:.1f}")
                    print(f"   📊 Average output tokens per request: {avg_output:.1f}")

                # Show detailed breakdown for recent entries
                print("\n🔬 Detailed Request Analysis (last 3):")
                for i, entry in enumerate(inference_entries[:3], 1):
                    details = entry.inferenceDetails
                    if details:
                        print(f"\n   {i}. Request from {entry.timestamp}")
                        print(f"      💰 Cost: ${entry.amount:.6f} {entry.currency}")

                        # Show token details
                        input_tokens = details.promptTokens or 0
                        output_tokens = details.completionTokens or 0
                        total_tokens = input_tokens + output_tokens

                        print(f"      📥 Input tokens: {input_tokens:,}")
                        print(f"      📤 Output tokens: {output_tokens:,}")
                        print(f"      📊 Total tokens: {total_tokens:,}")

                        # Calculate cost per token if possible
                        if total_tokens > 0:
                            cost_per_token = entry.amount / total_tokens
                            print(f"      💱 Cost per token: ${cost_per_token:.8f}")

                        # Show request ID if available
                        if details.requestId:
                            print(f"      🆔 Request ID: {details.requestId}")

                        # Show model info via the entry's SKU
                        print(f"      🤖 SKU: {entry.sku}")

            else:
                print("ℹ️ No inference details found in recent billing data")
                print("💡 Inference details are only available for LLM requests")

            return True

        except Exception as e:
            print(f"❌ Error analyzing inference details: {e}")
            return False


async def demonstrate_pagination() -> bool:
    """Demonstrate cursor pagination for large billing datasets."""
    print("\n📄 Cursor Pagination Demonstration")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # The first request takes the filters; each response carries a
            # nextCursor token. A continuation request sends ONLY that token —
            # the filters travel inside the cursor.
            print("📖 Walking usage-history pages by cursor...")

            page_num = 0
            total_processed = 0
            cursor: str | None = None

            while page_num < 3:  # Limit to first 3 pages for demo
                if cursor is None:
                    usage_response = await client.billing.get_usage_history(
                        format=BillingFormatEnum.JSON, pageSize=10
                    )
                else:
                    usage_response = await client.billing.get_usage_history(cursor=cursor)
                assert isinstance(usage_response, BillingUsageHistoryResponse)

                entries = usage_response.data
                page_num += 1

                print(f"\n📄 Page {page_num}:")
                print(f"   📦 Entries on this page: {len(entries)}")

                if entries:
                    page_cost = sum(entry.amount for entry in entries if entry.currency == "USD")
                    print(f"   💰 Page total: ${page_cost:.6f} USD")

                    # Entries arrive in ascending timestamp order.
                    oldest = entries[0].timestamp
                    latest = entries[-1].timestamp
                    print(f"   📅 Timestamp range: {oldest} to {latest}")

                total_processed += len(entries)

                cursor = usage_response.nextCursor
                if cursor is None:
                    print(f"\n✅ Reached the end of the walk ({total_processed} entries)")
                    break

            if cursor is not None:
                print(
                    "\n💡 Demonstration limited to 3 pages. "
                    "Use iter_usage_history() to walk everything automatically."
                )

            return True

        except Exception as e:
            print(f"❌ Error demonstrating pagination: {e}")
            return False


async def export_to_csv() -> bool:
    """Demonstrate CSV export functionality."""
    print("\n📁 CSV Export Demonstration")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            print("📥 Requesting CSV export...")

            # Get usage data in CSV format
            csv_data = await client.billing.get_usage_history(
                format=BillingFormatEnum.CSV, pageSize=100
            )

            # Debug what type we actually got
            print(f"🔍 CSV response type: {type(csv_data)}")
            print(f"🔍 CSV response content preview: {str(csv_data)[:200]}...")

            # Handle both bytes and string responses
            if isinstance(csv_data, bytes):
                csv_content = csv_data
                csv_text_str = csv_data.decode("utf-8")
            elif isinstance(csv_data, str):
                csv_content = csv_data.encode("utf-8")
                csv_text_str = csv_data
            else:
                print(f"❌ Unexpected CSV response format: {type(csv_data)}")
                print(f"   Content: {csv_data}")
                return False

            # Save to file
            filename = RESULTS_DIR / f"venice_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            with open(filename, "wb") as f:
                f.write(csv_content)

            print(f"✅ CSV data exported to: {filename}")
            print(f"📊 File size: {len(csv_content)} bytes")

            # Preview the CSV content
            lines = csv_text_str.splitlines()

            print("\n📄 CSV Preview (first 5 lines):")
            for i, line in enumerate(lines[:5]):
                if line.strip():  # Skip empty lines
                    print(f"   {i + 1}: {line}")

            if len(lines) > 5:
                print(f"   ... and {len(lines) - 5} more lines")

            print("\n💡 Use this CSV file for:")
            print("   📊 Spreadsheet analysis (Excel, Google Sheets)")
            print("   📈 Business intelligence tools")
            print("   🔍 Custom data analysis scripts")
            print("   📋 Financial reporting and auditing")

            return True

        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return False


async def cost_analysis_by_timeframe() -> bool:
    """Analyze costs across different timeframes."""
    print("\n📈 Cost Analysis by Timeframe")
    print("-" * 40)

    # Each timeframe is fetched in its own inner try (with continue), so the
    # outer try would not catch a failure when every timeframe fails. Track
    # success explicitly so any failed/timed-out fetch yields a non-zero exit.
    ok = True
    async with VeniceClient() as client:
        try:
            # Define different timeframes
            timeframes = [("Last 24 hours", 1), ("Last 7 days", 7), ("Last 30 days", 30)]

            for period_name, days in timeframes:
                print(f"\n📅 {period_name}:")

                try:
                    end_date = datetime.now(UTC)
                    start_date = end_date - timedelta(days=days)

                    # Use very small limit for timeframe analysis to avoid timeouts
                    limit = 5 if days == 1 else 10 if days == 7 else 15

                    print(f"   📥 Requesting {limit} entries for analysis...")

                    # For recent data (24h), omit endTimestamp to avoid slow range queries.
                    # usage-history requires pageSize >= 10, so fetch a full page and
                    # slice down to the small sample size this section wants.
                    page_size = max(limit, 10)
                    if days == 1:
                        usage_response = await client.billing.get_usage_history(
                            format=BillingFormatEnum.JSON,
                            startTimestamp=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            # No endTimestamp - much faster for recent data!
                            pageSize=page_size,
                        )
                    else:
                        usage_response = await client.billing.get_usage_history(
                            format=BillingFormatEnum.JSON,
                            startTimestamp=start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            endTimestamp=end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            pageSize=page_size,
                        )
                    assert isinstance(usage_response, BillingUsageHistoryResponse)

                    entries = usage_response.data[:limit]

                    if entries:
                        # Filter to USD-only before summing so the "USD" label is
                        # honest — mixing DIEM/BUNDLED_CREDITS amounts into a USD total would be
                        # misleading. This mirrors get_recent_usage above so every
                        # section reports costs the same way.
                        usd_entries = [e for e in entries if e.currency == "USD"]
                        # Venice API returns debits as negative amounts (deductions).
                        total_cost = abs(sum(e.amount for e in usd_entries))
                        total_units = sum(e.units for e in usd_entries)
                        avg_cost_per_request = total_cost / len(usd_entries) if usd_entries else 0

                        print(f"   💰 Total cost: ${total_cost:.6f} USD")
                        print(f"   📊 Total units (USD entries): {total_units:.2f}")
                        print(
                            f"   📈 Number of requests: {len(usd_entries)} USD "
                            f"of {len(entries)} total"
                        )
                        print(f"   📊 Average cost per request: ${avg_cost_per_request:.6f}")

                        # Calculate daily average
                        daily_avg = total_cost / days if days > 0 else total_cost
                        print(f"   📊 Daily average: ${daily_avg:.6f}")
                    else:
                        print("   ℹ️ No usage data found")

                except TimeoutError:
                    print(f"   ⏰ Request timed out for {period_name}")
                    print("   💡 Try reducing the time range or limit for this period")
                    ok = False
                    continue
                except Exception as e:
                    print(f"   ❌ Error analyzing {period_name}: {e}")
                    ok = False
                    continue

        except Exception as e:
            print(f"❌ Error analyzing costs by timeframe: {e}")
            ok = False

    return ok


async def show_balance() -> bool:
    """Display the account's current balance via ``billing.get_balance()``.

    This hits the stable ``GET /billing/balance`` endpoint and reports the
    remaining DIEM/USD balances plus the DIEM epoch allocation. Every field on
    :class:`BillingBalanceResponse` is optional, so guard before dereferencing.
    """
    print("\n💳 Account Balance")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            balance = await client.billing.get_balance()
            assert isinstance(balance, BillingBalanceResponse)

            # can_consume tells you whether the account can currently run
            # inference (i.e. it has spendable balance / is in good standing).
            if balance.can_consume is not None:
                status = "✅ yes" if balance.can_consume else "🚫 no"
                print(f"   🟢 Can consume inference: {status}")
            if balance.consumption_currency:
                print(f"   🪙 Consumption currency: {balance.consumption_currency}")

            # balances is itself optional; only dereference when present.
            if balance.balances is not None:
                if balance.balances.usd is not None:
                    print(f"   💵 USD balance:  ${balance.balances.usd:.4f}")
                if balance.balances.diem is not None:
                    print(f"   💎 DIEM balance: {balance.balances.diem:.4f}")
            else:
                print("   ℹ️ No per-currency balance breakdown returned")

            if balance.diem_epoch_allocation is not None:
                print(f"   📐 DIEM epoch allocation: {balance.diem_epoch_allocation:.4f}")

            return True

        except Exception as e:
            print(f"❌ Error retrieving balance: {e}")
            print("💡 Note: Billing data access requires appropriate API permissions")
            return False


async def show_usage_analytics() -> bool:
    """Summarize aggregated usage via ``billing.get_usage_analytics()`` (Beta).

    This wraps the beta ``GET /billing/usage-analytics`` endpoint, which returns
    pre-aggregated breakdowns by date, model, and API key — ideal for dashboards.
    We use ``lookback="30d"`` (a relative period) rather than start/end dates so
    we don't depend on the exact date format the endpoint expects; lookback and
    explicit dates are mutually exclusive.

    The endpoint is beta and emits a ``FutureWarning`` by design. If the account
    isn't entitled to it (or the endpoint isn't deployed), we degrade to a clear
    skip rather than failing the whole run.
    """
    print("\n📊 Usage Analytics (Beta)")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            print("📥 Fetching aggregated analytics for the last 30 days...")

            # The beta endpoint intentionally raises a FutureWarning; surface it
            # honestly rather than letting it spam every line of output.
            with warnings.catch_warnings():
                warnings.simplefilter("once", category=FutureWarning)
                analytics = await client.billing.get_usage_analytics(lookback="30d")
            assert isinstance(analytics, UsageAnalyticsResponse)

            print(f"   🗓️ Lookback window: {analytics.lookback}")
            print(f"   📅 Days with usage: {len(analytics.byDate)}")

            # Per-day totals (show the most recent few days).
            if analytics.byDate:
                print("\n   📈 Recent daily totals:")
                for day in analytics.byDate[-5:]:
                    print(f"      {day.date}: ${day.USD:.4f} USD / {day.DIEM:.4f} DIEM")

            # Top models by spend (the API already sorts highest-first).
            if analytics.byModel:
                print("\n   🤖 Top models by USD spend:")
                for model in analytics.byModel[:5]:
                    mtype = f" [{model.modelType}]" if model.modelType else ""
                    print(
                        f"      {model.modelName}{mtype}: "
                        f"${model.totalUsd:.4f} USD / {model.totalDiem:.4f} DIEM "
                        f"({model.totalUnits:.0f} {model.unitType})"
                    )

            # Breakdown by API key (or 'Web App' for dashboard usage).
            if analytics.byKey:
                print("\n   🔑 Spend by API key:")
                for key in analytics.byKey[:5]:
                    print(
                        f"      {key.description}: "
                        f"${key.totalUsd:.4f} USD / {key.totalDiem:.4f} DIEM"
                    )

            if not (analytics.byDate or analytics.byModel or analytics.byKey):
                print("   ℹ️ No usage recorded in the analytics window")

            return True

        except (NotFoundError, PermissionDeniedError) as e:
            # Genuine unavailability/unentitlement on this account: this beta
            # endpoint may not be deployed or accessible. Degrade to a skip so
            # one missing beta feature doesn't fail the whole run.
            print(f"⏭️ Usage analytics unavailable on this account: {e}")
            print("💡 The usage-analytics endpoint is beta and may be gated.")
            return True

        except Exception as e:
            print(f"❌ Error retrieving usage analytics: {e}")
            return False


async def main() -> int:
    """Run all billing analytics examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Billing & Usage Analytics Examples")
    print("=" * 70)

    results: list[tuple[str, bool]] = [
        ("show_balance", await show_balance()),
        ("get_recent_usage", await get_recent_usage()),
        ("analyze_inference_details", await analyze_inference_details()),
        ("demonstrate_pagination", await demonstrate_pagination()),
        ("export_to_csv", await export_to_csv()),
        ("cost_analysis_by_timeframe", await cost_analysis_by_timeframe()),
        ("show_usage_analytics", await show_usage_analytics()),
    ]

    failed = [name for name, ok in results if not ok]

    print("\n✨ Billing analytics examples completed!")
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
    print("\n💡 Key concepts demonstrated:")
    print("   - Checking the current account balance (get_balance)")
    print("   - Retrieving usage data with flexible filtering")
    print("   - Aggregated usage analytics by date/model/key (get_usage_analytics, Beta)")
    print("   - Analyzing costs by product SKU")
    print("   - Understanding inference details and token usage")
    print("   - Working with cursor pagination for large datasets")
    print("   - Exporting data to CSV for external analysis")
    print("   - Cost analysis across different timeframes")
    print("   - Currency handling (USD and DIEM)")
    print("   - Billing entry structure and metadata")
    print("\n📊 Use cases for billing data:")
    print("   - Cost monitoring and budget management")
    print("   - Usage optimization and efficiency analysis")
    print("   - Billing reconciliation and auditing")
    print("   - Capacity planning and forecasting")
    print("   - Integration with accounting systems")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have billing data access.", file=sys.stderr)
        sys.exit(1)
