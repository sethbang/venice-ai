"""
Billing and usage models for Venice AI API.

This module contains Pydantic models for billing usage tracking,
pagination, usage statistics, and aggregated usage analytics.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InferenceDetails(BaseModel):
    """Details about inference request"""

    model_config = ConfigDict(extra="allow")

    requestId: str | None = Field(None, description="Unique identifier for inference request")
    promptTokens: float | None = Field(None, description="Tokens in prompt (LLM usage only)")
    completionTokens: float | None = Field(
        None, description="Tokens in completion (LLM usage only)"
    )
    inferenceExecutionTime: float | None = Field(None, description="Execution time in milliseconds")


class BillingUsageEntry(BaseModel):
    """A single billing usage entry from GET /billing/usage-history."""

    model_config = ConfigDict(extra="allow")

    sku: str = Field(..., description="Product SKU associated with the billing usage entry")
    amount: float = Field(..., description="Total amount charged for the entry")
    # usage-history returns historical rows: the spec enum (USD/DIEM/BUNDLED_CREDITS)
    # constrains new rows and the filter, but legacy rows may carry retired currencies
    # (e.g. "VCU"). Keep the documented values visible to type-checkers while letting
    # any unmodeled value round-trip rather than raising and killing the whole page.
    currency: Literal["USD", "DIEM", "BUNDLED_CREDITS"] | str = Field(
        ..., description="Currency charged for the entry"
    )
    units: float = Field(..., description="Number of units consumed")
    pricePerUnitUsd: float = Field(..., description="Price per unit in USD")
    notes: str = Field(..., description="Notes about the billing usage entry")
    timestamp: str = Field(..., description="When the billing usage entry was created")
    inferenceDetails: InferenceDetails | None = Field(
        None, description="Details about related inference request"
    )


class BillingUsageHistoryResponse(BaseModel):
    """Response for GET /billing/usage-history.

    A cursor-paginated page of usage entries in ascending timestamp order.
    ``nextCursor`` carries the walk forward; ``None`` marks the final page.
    """

    model_config = ConfigDict(extra="allow")

    data: list[BillingUsageEntry] = Field(
        ..., description="Usage entries in ascending timestamp order"
    )
    nextCursor: str | None = Field(
        None,
        description=(
            "Continuation token for the next page, sent as the ``cursor`` query "
            "parameter. ``None`` means this is the last page."
        ),
    )


class BillingBalances(BaseModel):
    """Nested balance amounts by currency."""

    model_config = ConfigDict(extra="allow")

    diem: float | None = Field(default=None, description="Remaining DIEM balance")
    usd: float | None = Field(default=None, description="Remaining USD balance")


class BillingBalanceResponse(BaseModel):
    """Response model for GET /billing/balance."""

    can_consume: bool | None = Field(
        default=None,
        alias="canConsume",
        description="Whether the account can currently consume inference",
    )
    consumption_currency: Literal["USD", "VCU", "DIEM", "BUNDLED_CREDITS"] | None = Field(
        default=None,
        alias="consumptionCurrency",
        description="Currency currently used for consumption",
    )
    balances: BillingBalances | None = Field(
        default=None, description="Per-currency balance amounts"
    )
    diem_epoch_allocation: float | None = Field(
        default=None,
        alias="diemEpochAllocation",
        description="DIEM epoch allocation used for usage-percentage calculations",
    )

    model_config = ConfigDict(populate_by_name=True)


# ============================================================================
# Usage Analytics Models (Beta)
# ============================================================================


class UsageAnalyticsByDate(BaseModel):
    """Daily usage totals for a specific date."""

    model_config = ConfigDict(extra="allow")

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    USD: float = Field(..., description="Total usage in USD for that day")
    DIEM: float = Field(..., description="Total usage in DIEM for that day")


class UsageAnalyticsModelBreakdown(BaseModel):
    """Usage breakdown by token/unit type within a model."""

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        ..., description="Token type (e.g., 'Input', 'Output', 'Cache Read', 'Cache Write')"
    )
    usd: float = Field(..., description="USD amount for this breakdown")
    diem: float = Field(..., description="DIEM amount for this breakdown")
    units: float = Field(..., description="Number of units for this breakdown")


class UsageAnalyticsByModel(BaseModel):
    """Usage breakdown for a specific model."""

    model_config = ConfigDict(extra="allow")

    modelName: str = Field(..., description="Display name of the model (e.g., 'Llama 3.3 70B')")
    unitType: str = Field(
        ..., description="Type of units consumed (tokens, images, chars, minutes, seconds)"
    )
    modelType: str | None = Field(
        None, description="Type of model (LLM, IMAGE, TTS, ASR, VIDEO), or null"
    )
    totalUsd: float = Field(..., description="Total USD spent on this model")
    totalDiem: float = Field(..., description="Total DIEM spent on this model")
    totalUnits: float = Field(..., description="Total units consumed for this model")
    breakdown: list[UsageAnalyticsModelBreakdown] | None = Field(
        None,
        description="Array of usage breakdowns by type (only present if multiple types)",
    )


class UsageAnalyticsByKey(BaseModel):
    """Usage breakdown for a specific API key."""

    model_config = ConfigDict(extra="allow")

    apiKeyId: str | None = Field(
        None, description="The API key ID, or null if usage was from the web app"
    )
    description: str = Field(..., description="API key description or 'Web App'")
    totalUsd: float = Field(..., description="Total USD spent via this key")
    totalDiem: float = Field(..., description="Total DIEM spent via this key")
    totalUnits: float = Field(..., description="Total units consumed via this key")


class UsageAnalyticsResponse(BaseModel):
    """Response model for the billing usage analytics endpoint (Beta).

    .. beta::
        This model corresponds to a beta API endpoint. The schema may change
        without notice in future API versions.

    Provides aggregated usage analytics with breakdowns by date, model, and
    API key. Data is cached server-side for 10 minutes.
    """

    model_config = ConfigDict(extra="allow")

    lookback: str = Field(
        ...,
        description=(
            "The lookback period used for the query. Either in 'Nd' format "
            "(e.g., '7d') or 'startDate:endDate' format."
        ),
    )
    byDate: list[UsageAnalyticsByDate] = Field(
        ..., description="Daily usage totals for the requested period"
    )
    byModel: list[UsageAnalyticsByModel] = Field(
        ..., description="Usage breakdown by model, sorted by total spend (highest first)"
    )
    byModelDaily: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Daily chart data for top 8 models. Each entry contains a 'date' "
            "(timestamp) plus model names as keys with DIEM usage values."
        ),
    )
    byModelDailyUsd: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "USD-denominated sibling of ``byModelDaily``: daily chart data for "
            "top 8 models with USD usage values."
        ),
    )
    topModels: list[str] = Field(
        ..., description="Array of the top 8 model names by usage, for chart legends"
    )
    byKey: list[UsageAnalyticsByKey] = Field(
        ..., description="Usage breakdown by API key, sorted by total spend (highest first)"
    )
    byKeyDaily: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Daily chart data for top 8 API keys. Each entry contains a 'date' "
            "(timestamp) plus key descriptions as keys with DIEM usage values."
        ),
    )
    byKeyDailyUsd: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "USD-denominated sibling of ``byKeyDaily``: daily chart data for "
            "top 8 API keys with USD usage values."
        ),
    )
    topKeyNames: list[str] = Field(
        ..., description="Array of the top 8 API key descriptions by usage, for chart legends"
    )


class UsageAnalyticsQueryParams(BaseModel):
    """Query parameters for the billing usage analytics endpoint (Beta).

    You can specify the time period using either:
    - ``lookback``: A relative period like "7d", "30d", up to "90d"
    - ``startDate`` and ``endDate``: A custom date range in YYYY-MM-DD format.
      Both are required if either is provided.

    If no parameters are specified, the default lookback period is 7 days.
    """

    lookback: str | None = Field(
        None,
        description=(
            "Relative lookback period (e.g., '7d', '30d', up to '90d'). "
            "Cannot be used with startDate/endDate."
        ),
    )
    startDate: str | None = Field(
        None,
        description="Start date in YYYY-MM-DD format. Required if endDate is provided.",
    )
    endDate: str | None = Field(
        None,
        description="End date in YYYY-MM-DD format. Required if startDate is provided.",
    )


__all__ = [
    "InferenceDetails",
    "BillingUsageEntry",
    "BillingUsageHistoryResponse",
    "BillingBalanceResponse",
    # Usage Analytics (Beta)
    "UsageAnalyticsByDate",
    "UsageAnalyticsModelBreakdown",
    "UsageAnalyticsByModel",
    "UsageAnalyticsByKey",
    "UsageAnalyticsResponse",
    "UsageAnalyticsQueryParams",
]
