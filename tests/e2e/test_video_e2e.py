"""
End-to-end tests for Venice AI Video resource using VCR recording/replay.

These tests exercise the complete video generation lifecycle through the real
Venice AI API (or pre-recorded VCR cassettes):
  - Quoting (price estimation)
  - Queuing (text-to-video and image-to-video)
  - Retrieving (polling for status)
  - Completing (cleanup)

## Recording New Cassettes

Cassettes live under ``tests/e2e/cassettes/`` and are gitignored — they are a
local-dev convenience, never a committed fixture. To refresh them after an API
change, re-run the tests you want with ``VENICE_VCR_RECORD=all``; the root
conftest overwrites exactly those cassettes and restores the previous recording
if a test fails, so a transient error never destroys a good one::

    VENICE_API_KEY=... VENICE_VCR_RECORD=all \\
        poetry run pytest tests/e2e/test_video_e2e.py

Queueing a video costs real credit on every un-cassetted run, so the optional
parameters below are derived at their cheapest supported setting.

## Security Note

Cassettes are automatically scrubbed of sensitive data
(Authorization headers etc.) by the root conftest's ``vcr_config`` fixture.
"""

import asyncio
import base64
import os
from io import BytesIO

import pytest
import pytest_asyncio
from PIL import Image

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import (
    APIError,
    APIStatusError,
    InvalidRequestError,
    PaymentRequiredError,
    VeniceError,
)
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
)

# ---------------------------------------------------------------------------
# Video models are resolved DYNAMICALLY from the live API — never hardcoded.
# Venice retires models and ships new generations under new ID conventions
# (wan-2.6-* and wan-2-7-* are distinct generations that coexist, not a
# rename), and a stale hardcoded ID silently 400s.
#
# Resolution deliberately happens at the top of each test, *before* the body
# enters its ``vcr_cassette`` context. Inside a cassette, ``GET /models`` is
# served from whatever catalog snapshot that cassette happened to record, so a
# model picked there can be months out of date — and because the choice is
# cached for the whole session, it leaks into tests whose own cassette is
# missing and which therefore talk to the live API. Resolving outside the
# cassette keeps the model id and its advertised constraints consistent with the
# API actually being called. An env var can still pin a specific model.
#
# (A fixture would read better but cannot work here: async fixtures run on the
# session event loop while test bodies run on a per-function loop, so an HTTP
# call made during fixture setup binds the client's aiohttp session to the wrong
# loop.)
# ---------------------------------------------------------------------------
_VIDEO_MODEL_CACHE: dict[str, str] = {}


async def _resolve_video_model(client, video_type: str, env_var: str) -> str:
    """Resolve (and session-cache) a video model of ``video_type``; skip if none."""
    if video_type not in _VIDEO_MODEL_CACHE:
        override = os.environ.get(env_var)
        if override:
            _VIDEO_MODEL_CACHE[video_type] = override
        else:
            try:
                _VIDEO_MODEL_CACHE[video_type] = await client.models.resolve_video(
                    video_type=video_type
                )
            except (VeniceError, APIError) as exc:
                pytest.skip(f"No {video_type} model available on this account: {exc}")
    return _VIDEO_MODEL_CACHE[video_type]


async def _resolve_t2v(client) -> str:
    return await _resolve_video_model(client, "text-to-video", "VENICE_E2E_VIDEO_T2V_MODEL")


async def _resolve_i2v(client) -> str:
    return await _resolve_video_model(client, "image-to-video", "VENICE_E2E_VIDEO_I2V_MODEL")


def _resolution_rank(resolution: str) -> int:
    """Approximate pixel height, so the cheapest advertised option can be picked.

    Video pricing scales with resolution and these tests bill real credit on
    every run, so "all optional params" populates ``resolution`` with the
    smallest one the model offers rather than the first one listed. Venice
    spells resolutions several ways: ``480p``/``768P`` (height), ``2K``/``4k``
    (thousands), and ``2x``/``4x`` (upscale factors, which belong to upscale
    models and are ranked last).
    """
    value = resolution.strip().lower()
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        return 1 << 30
    number = int(digits)
    if value.endswith("x"):
        return 1 << 20  # upscale factor, not a height — never the cheap pick
    if value.endswith("k"):
        return number * 1000
    return number


async def _optional_video_params(client, model: str, *, audio: bool) -> dict:
    """Build the widest *valid* optional-parameter set for ``model``.

    Which optional parameters a video model accepts varies per model, and
    sending an unsupported one is a hard 400 rather than a silent ignore —
    ``audio`` on a model with ``audio_configurable=False`` returns
    ``"This model does not support audio configuration"``. So the values come
    from the model's own advertised constraints instead of being hardcoded,
    which keeps these tests meaningful as the catalog turns over.

    Note that ``supports_audio`` and ``audio_configurable`` are different
    claims: minimax-h3 generates audio but rejects the ``audio`` parameter.
    Only the latter gates sending it.
    """
    caps = await client.models.get_capabilities(model)
    durations = list(caps.durations or [])
    params: dict = {
        "duration_seconds": "5s" if not durations or "5s" in durations else durations[0],
    }
    if caps.resolutions:
        params["resolution"] = min(caps.resolutions, key=_resolution_rank)
    if caps.aspect_ratios:
        params["aspect_ratio"] = caps.aspect_ratios[0]
    if caps.audio_configurable:
        params["audio"] = audio
    return params


def _generate_test_image_data_url(width: int = 256, height: int = 256) -> str:
    """Generate a small solid-color JPEG as a data URL for I2V tests.

    Using a data URL avoids external HTTP dependencies (e.g. Wikimedia
    rate-limiting) that previously caused 'corrupted or unreadable' errors.
    The Venice API requires a minimum of 240×240 pixels for I2V.
    """
    img = Image.new("RGB", (width, height), color=(135, 206, 235))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=50)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# A small inline image for I2V tests (data URL avoids external fetch failures)
VIDEO_TEST_IMAGE_URL = os.environ.get(
    "VENICE_E2E_VIDEO_IMAGE_URL",
    _generate_test_image_data_url(),
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@pytest.mark.asyncio
class TestVideoE2E:
    """End-to-end tests for the Video resource using VCR cassettes."""

    @pytest_asyncio.fixture
    async def venice_client(self):
        """Create VeniceClient for E2E testing with intelligent rate limiting."""
        api_key = os.getenv("VENICE_API_KEY")
        if not api_key:
            pytest.skip("VENICE_API_KEY environment variable required for E2E tests")

        client = create_test_venice_client(
            api_key=api_key,
            scheduler_mode=SchedulerMode.INTELLIGENT,
            enable_redis=False,
        )
        try:
            yield client
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # quote() tests
    # ------------------------------------------------------------------

    async def test_quote_t2v_basic(self, venice_client, vcr_cassette):
        """Quote a text-to-video generation and verify response shape."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=t2v_model,
                    duration_seconds="5s",
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote is not None
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_quote_t2v_all_params(self, venice_client, vcr_cassette):
        """Quote with every optional parameter the model supports populated."""
        t2v_model = await _resolve_t2v(venice_client)
        t2v_optional_params = await _optional_video_params(venice_client, t2v_model, audio=True)
        # Guard against the derived set quietly collapsing: every video model
        # advertises resolutions, so an absent one means the capability lookup
        # degraded and this test would be exercising almost no optional params.
        assert "resolution" in t2v_optional_params, (
            f"no optional params derived for {t2v_model}: {t2v_optional_params}"
        )
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=t2v_model,
                    **t2v_optional_params,
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_quote_upscale_with_video_url(self, venice_client, vcr_cassette):
        """Quote an upscale (v2v) request using video_url + upscale_factor."""
        i2v_model = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                quote = await venice_client.video.quote(
                    model=i2v_model,
                    duration_seconds="5s",
                    resolution="1080p",
                    upscale_factor=2,
                    video_url=VIDEO_TEST_IMAGE_URL,
                )

                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc) or "not supported" in str(exc).lower():
                    pytest.skip(f"Upscale combination unsupported for this model: {exc}")
                raise

    # ------------------------------------------------------------------
    # queue() tests
    # ------------------------------------------------------------------

    async def test_queue_t2v_basic(self, venice_client, vcr_cassette):
        """Queue a minimal text-to-video request and get a queue_id back."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A serene mountain landscape with flowing clouds",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None
                assert len(result.queue_id) > 0
                assert result.model is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_queue_t2v_all_optional_params(self, venice_client, vcr_cassette):
        """Queue with every optional T2V parameter the model supports populated."""
        t2v_model = await _resolve_t2v(venice_client)
        t2v_optional_params = await _optional_video_params(venice_client, t2v_model, audio=True)
        # Guard against the derived set quietly collapsing: every video model
        # advertises resolutions, so an absent one means the capability lookup
        # degraded and this test would be exercising almost no optional params.
        assert "resolution" in t2v_optional_params, (
            f"no optional params derived for {t2v_model}: {t2v_optional_params}"
        )
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A kitten chasing a laser pointer",
                    negative_prompt="blurry, ugly, low quality",
                    **t2v_optional_params,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_queue_i2v(self, venice_client, vcr_cassette):
        """Queue an image-to-video request."""
        i2v_model = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=i2v_model,
                    prompt="Bring this image to life with subtle motion",
                    duration_seconds="5s",
                    image_url=VIDEO_TEST_IMAGE_URL,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise

    async def test_queue_i2v_all_optional_params(self, venice_client, vcr_cassette):
        """Queue I2V with every optional param the model supports populated."""
        i2v_model = await _resolve_i2v(venice_client)
        i2v_optional_params = await _optional_video_params(venice_client, i2v_model, audio=False)
        assert "resolution" in i2v_optional_params, (
            f"no optional params derived for {i2v_model}: {i2v_optional_params}"
        )
        with vcr_cassette:
            try:
                result = await venice_client.video.submit(
                    model=i2v_model,
                    prompt="Pan across the scene slowly",
                    negative_prompt="low quality, distorted",
                    image_url=VIDEO_TEST_IMAGE_URL,
                    **i2v_optional_params,
                )

                assert isinstance(result, VideoQueueResponse)
                assert result.queue_id is not None

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # retrieve() tests
    # ------------------------------------------------------------------

    async def test_retrieve_after_queue(self, venice_client, vcr_cassette):
        """Queue → retrieve immediately; expect PROCESSING or COMPLETED."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A river flowing through a forest",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )

                status = await venice_client.video.retrieve(
                    model=t2v_model,
                    queue_id=queued.queue_id,
                )

                # Immediately after queuing we typically get PROCESSING
                assert isinstance(
                    status,
                    (VideoProcessingStatus, VideoCompletedStatus, VideoFailedStatus),
                )

                if isinstance(status, VideoProcessingStatus):
                    assert status.status == "PROCESSING"
                    assert status.average_execution_time >= 0
                elif isinstance(status, VideoCompletedStatus):
                    assert status.status == "COMPLETED"
                elif isinstance(status, VideoFailedStatus):
                    assert status.status == "FAILED"

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_retrieve_with_delete_media_on_completion(self, venice_client, vcr_cassette):
        """Verify delete_media_on_completion flag is accepted."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="Waves crashing on a rocky shore",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )

                status = await venice_client.video.retrieve(
                    model=t2v_model,
                    queue_id=queued.queue_id,
                    delete_media_on_completion=True,
                )

                assert isinstance(
                    status,
                    (VideoProcessingStatus, VideoCompletedStatus, VideoFailedStatus),
                )

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_retrieve_poll_until_done(self, venice_client, vcr_cassette):
        """Queue and poll until COMPLETED or FAILED (with timeout)."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A slow-motion droplet splashing into water",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )

                max_polls = 60  # ~5 min with 5s sleep
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v_model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        assert status.url is not None or status.status == "COMPLETED"
                        return  # success
                    elif isinstance(status, VideoFailedStatus):
                        # Generation failed – not a test failure per se
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    assert isinstance(status, VideoProcessingStatus)
                    await asyncio.sleep(5)

                pytest.skip("Video generation did not complete within timeout")

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # complete() tests
    # ------------------------------------------------------------------

    async def test_complete_after_retrieval(self, venice_client, vcr_cassette):
        """Full lifecycle: queue → poll → complete."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                queued = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A butterfly landing on a flower",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )

                # Poll until done (or timeout)
                max_polls = 60
                final_status: VideoCompletedStatus | None = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v_model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("Video generation did not complete within timeout")

                # Now call complete.
                # When the video was delivered as binary (url is None),
                # the API may return success=False because there is no
                # server-side media to clean up.  Both outcomes are valid.
                try:
                    result = await venice_client.video.cancel(
                        model=t2v_model,
                        queue_id=queued.queue_id,
                    )

                    assert isinstance(result, VideoCompleteResponse)
                    if final_status.url is not None:
                        # URL-based delivery → cleanup should succeed
                        assert result.success is True
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    # ------------------------------------------------------------------
    # Error handling tests
    # ------------------------------------------------------------------

    async def test_queue_invalid_model(self, venice_client, vcr_cassette):
        """Queue with a non-existent model should raise an API error."""
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError)):
            await venice_client.video.submit(
                model="definitely-invalid-video-model-xyz",
                prompt="This should fail",
                duration_seconds="5s",
            )

    async def test_retrieve_invalid_queue_id(self, venice_client, vcr_cassette):
        """Retrieve with a bogus queue_id should raise an API error."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError, ValueError)):
            await venice_client.video.retrieve(
                model=t2v_model,
                queue_id="nonexistent-queue-id-00000000",
            )

    async def test_complete_invalid_queue_id(self, venice_client, vcr_cassette):
        """Complete with a bogus queue_id should raise an API error."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette, pytest.raises((VeniceError, APIError, APIStatusError)):
            await venice_client.video.cancel(
                model=t2v_model,
                queue_id="nonexistent-queue-id-00000000",
            )

    # ------------------------------------------------------------------
    # Full workflow: quote → queue → retrieve → complete
    # ------------------------------------------------------------------

    async def test_full_video_workflow(self, venice_client, vcr_cassette):
        """End-to-end: quote → queue → poll → complete."""
        t2v_model = await _resolve_t2v(venice_client)
        with vcr_cassette:
            try:
                # Step 1 – Quote
                quote = await venice_client.video.quote(
                    model=t2v_model,
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )
                assert isinstance(quote, VideoQuoteResponse)
                assert quote.quote >= 0

                # Step 2 – Queue
                queued = await venice_client.video.submit(
                    model=t2v_model,
                    prompt="A time-lapse of clouds rolling over a city skyline",
                    duration_seconds="5s",
                    aspect_ratio="16:9",
                )
                assert isinstance(queued, VideoQueueResponse)
                assert queued.queue_id

                # Step 3 – Poll until done
                max_polls = 60
                final_status = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=t2v_model,
                        queue_id=queued.queue_id,
                    )

                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"Video generation failed: {status.error}")
                        return

                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("Video generation did not complete within timeout")

                assert final_status.status == "COMPLETED"

                # Step 4 – Complete (cleanup)
                # When the video was delivered as binary (url is None),
                # the API may return success=False because there is no
                # server-side media to clean up.  Both outcomes are valid.
                try:
                    cleanup = await venice_client.video.cancel(
                        model=t2v_model,
                        queue_id=queued.queue_id,
                    )
                    assert isinstance(cleanup, VideoCompleteResponse)
                    if final_status.url is not None:
                        assert cleanup.success is True
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"Video model unavailable: {exc}")
                raise

    async def test_full_i2v_workflow(self, venice_client, vcr_cassette):
        """End-to-end image-to-video: quote → queue → poll → complete."""
        i2v_model = await _resolve_i2v(venice_client)
        with vcr_cassette:
            try:
                # Quote — /video/quote ignores prompt/image refs;
                # price is driven by model + duration + resolution.
                quote = await venice_client.video.quote(
                    model=i2v_model,
                    duration_seconds="5s",
                )
                assert isinstance(quote, VideoQuoteResponse)

                # Queue
                queued = await venice_client.video.submit(
                    model=i2v_model,
                    prompt="Animate the scene with gentle motion",
                    duration_seconds="5s",
                    image_url=VIDEO_TEST_IMAGE_URL,
                )
                assert queued.queue_id

                # Poll
                max_polls = 60
                final_status = None
                for _ in range(max_polls):
                    status = await venice_client.video.retrieve(
                        model=i2v_model,
                        queue_id=queued.queue_id,
                    )
                    if isinstance(status, VideoCompletedStatus):
                        final_status = status
                        break
                    elif isinstance(status, VideoFailedStatus):
                        pytest.skip(f"I2V generation failed: {status.error}")
                        return
                    await asyncio.sleep(5)

                if final_status is None:
                    pytest.skip("I2V generation did not complete within timeout")

                # Complete
                try:
                    cleanup = await venice_client.video.cancel(
                        model=i2v_model,
                        queue_id=queued.queue_id,
                    )
                    assert isinstance(cleanup, VideoCompleteResponse)
                except InvalidRequestError as exc:
                    # Binary-delivered videos (url is None) consume the server-side
                    # job on retrieval, so /video/complete reports "Request ID is
                    # invalid" — an expected terminal state, not a failure. A
                    # URL-delivered video must still complete cleanly.
                    assert final_status.url is None, (
                        f"video.complete() failed for a URL-delivered video: {exc}"
                    )
                    assert "request id is invalid" in str(exc).lower()

            except PaymentRequiredError as exc:
                pytest.skip(f"Insufficient balance for video generation: {exc}")
            except (VeniceError, APIError, APIStatusError) as exc:
                if _is_model_unavailable(exc):
                    pytest.skip(f"I2V model unavailable: {exc}")
                raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_model_unavailable(exc: Exception) -> bool:
    """Return True when the error suggests the model simply isn't available."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "not found",
            "not supported",
            "model",
            "invalid_model",
            "unavailable",
            "does not exist",
        )
    )
