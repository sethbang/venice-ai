"""Forward-compatibility hardening for low-level response models.

These response models carry ``model_config = ConfigDict(extra="allow")`` so a
new server-side field lands on ``model_extra`` instead of being silently
dropped (plain ``BaseModel`` default ``extra="ignore"``) or rejected
(``VeniceBaseModel`` inherits ``extra="forbid"``). This test pins that
contract so a future refactor can't quietly regress it.
"""

import pytest

from venice_ai.core.models.common import VeniceParameters
from venice_ai.types.api.audio import (
    AudioResponse,
    AudioTranscriptionResponse,
    ClonedVoice,
    TranscriptionChar,
    TranscriptionSegment,
    TranscriptionTimestamps,
    TranscriptionWord,
    VoiceDetail,
    VoiceList,
)
from venice_ai.types.api.augment import (
    AugmentSearchResult,
    AugmentTextParserResponse,
)
from venice_ai.types.api.billing import (
    BillingBalances,
    BillingUsageEntry,
    BillingUsageHistoryResponse,
    InferenceDetails,
    UsageAnalyticsByDate,
    UsageAnalyticsByKey,
    UsageAnalyticsByModel,
    UsageAnalyticsModelBreakdown,
    UsageAnalyticsResponse,
)
from venice_ai.types.api.characters import (
    Character,
    CharacterReview,
    CharacterReviewsPagination,
    CharacterReviewsSummary,
    CharacterStats,
)
from venice_ai.types.api.common import (
    CompletionTokensDetails,
    PromptTokensDetails,
)
from venice_ai.types.api.crypto import (
    BatchJsonRpcResponse,
    CryptoNetworksResponse,
    JsonRpcError,
    JsonRpcResponse,
)
from venice_ai.types.api.embeddings import EmbeddingObject, EmbeddingUsage
from venice_ai.types.api.images import ImageStylesResponse, SimpleImageData
from venice_ai.types.api.models import PricingTier, UpscalePricing
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoFailedStatus,
    VideoProcessingStatus,
)
from venice_ai.types.api.x402 import (
    X402BalanceData,
    X402BalanceResponse,
    X402TopUpData,
    X402TopUpResponse,
    X402Transaction,
    X402TransactionsData,
    X402TransactionsPagination,
    X402TransactionsResponse,
)

FORWARD_COMPAT_MODELS = [
    # core/models/common.py
    VeniceParameters,
    # types/api/common.py (real home of the token-details models)
    CompletionTokensDetails,
    PromptTokensDetails,
    # types/api/audio.py
    VoiceDetail,
    ClonedVoice,
    VoiceList,
    AudioResponse,
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionChar,
    TranscriptionTimestamps,
    AudioTranscriptionResponse,
    # types/api/video.py
    VideoProcessingStatus,
    VideoFailedStatus,
    VideoCompletedStatus,
    # types/api/embeddings.py
    EmbeddingObject,
    EmbeddingUsage,
    # types/api/characters.py
    CharacterStats,
    Character,
    CharacterReview,
    CharacterReviewsPagination,
    CharacterReviewsSummary,
    # types/api/billing.py
    InferenceDetails,
    BillingUsageEntry,
    BillingUsageHistoryResponse,
    BillingBalances,
    UsageAnalyticsByDate,
    UsageAnalyticsModelBreakdown,
    UsageAnalyticsByModel,
    UsageAnalyticsByKey,
    UsageAnalyticsResponse,
    # types/api/augment.py
    AugmentSearchResult,
    AugmentTextParserResponse,
    # types/api/crypto.py
    JsonRpcResponse,
    BatchJsonRpcResponse,
    JsonRpcError,
    CryptoNetworksResponse,
    # types/api/x402.py
    X402BalanceData,
    X402BalanceResponse,
    X402TopUpData,
    X402TopUpResponse,
    X402Transaction,
    X402TransactionsPagination,
    X402TransactionsData,
    X402TransactionsResponse,
    # types/api/images.py
    SimpleImageData,
    ImageStylesResponse,
    # types/api/models.py
    PricingTier,
    UpscalePricing,
]


@pytest.mark.parametrize("model_cls", FORWARD_COMPAT_MODELS, ids=lambda c: c.__name__)
def test_forward_compat_models_allow_extra(model_cls):
    """Each targeted response model must preserve unknown server fields."""
    assert model_cls.model_config.get("extra") == "allow"
