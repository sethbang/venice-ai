"""Unit tests for the .save() / .save_all() methods on response types."""

import base64

import pytest

from venice_ai.types.api.audio import AudioResponse
from venice_ai.types.api.base import TimingInfo
from venice_ai.types.api.images import ImageGenerationResponse


def _png_b64(byte: int = 0xAA) -> str:
    """Return a tiny base64-encoded blob for round-trip verification."""
    return base64.b64encode(bytes([byte] * 4)).decode("ascii")


def _real_png_b64() -> str:
    """Return base64 of bytes that start with the PNG magic header."""
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16).decode("ascii")


def _real_webp_b64() -> str:
    """Return base64 of bytes that start with the WebP magic header."""
    return base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 \x00\x00").decode("ascii")


def _real_jpeg_b64() -> str:
    """Return base64 of bytes that start with the JPEG magic header."""
    return base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x00").decode("ascii")


def _make_image_response(n: int = 1) -> ImageGenerationResponse:
    return ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [_png_b64(i) for i in range(n)],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )


# ---------------------------------------------------------------------------
# ImageGenerationResponse.save
# ---------------------------------------------------------------------------


def test_image_save_round_trip(tmp_path):
    response = _make_image_response()
    out = response.save(tmp_path / "subdir" / "image.png")
    assert out.read_bytes() == base64.b64decode(response.images[0])


def test_image_save_creates_parent_dirs(tmp_path):
    response = _make_image_response()
    target = tmp_path / "deep" / "nested" / "x.png"
    response.save(target)
    assert target.exists()


def test_image_save_raises_on_existing_without_overwrite(tmp_path):
    response = _make_image_response()
    target = tmp_path / "image.png"
    response.save(target)
    with pytest.raises(FileExistsError):
        response.save(target)


def test_image_save_overwrite_true_replaces(tmp_path):
    response = _make_image_response()
    target = tmp_path / "image.png"
    target.write_bytes(b"old")
    response.save(target, overwrite=True)
    assert target.read_bytes() == base64.b64decode(response.images[0])


# ---------------------------------------------------------------------------
# ImageGenerationResponse.save_all
# ---------------------------------------------------------------------------


def test_image_save_all_writes_each_index(tmp_path):
    response = _make_image_response(n=3)
    paths = response.save_all(tmp_path / "out", prefix="frame", ext="png")
    assert len(paths) == 3
    assert [p.name for p in paths] == ["frame_0.png", "frame_1.png", "frame_2.png"]
    for i, p in enumerate(paths):
        assert p.read_bytes() == base64.b64decode(response.images[i])


def test_image_save_all_strips_leading_dot_in_ext(tmp_path):
    response = _make_image_response(n=2)
    paths = response.save_all(tmp_path, ext=".jpeg")
    assert all(p.suffix == ".jpeg" for p in paths)


def test_image_save_auto_extension_when_path_has_no_suffix(tmp_path):
    """save() with extensionless path should sniff format and append it."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [_real_webp_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    out = response.save(tmp_path / "result")
    assert out.suffix == ".webp"
    assert out.name == "result.webp"
    assert out.read_bytes().startswith(b"RIFF")


def test_image_save_respects_explicit_suffix(tmp_path):
    """save() with explicit suffix should NOT auto-detect — user knows best."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [_real_webp_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    # Bytes are WebP but caller said .png — respect their choice
    out = response.save(tmp_path / "result.png")
    assert out.suffix == ".png"
    assert out.read_bytes().startswith(b"RIFF")


def test_image_save_unknown_format_falls_back_to_png(tmp_path):
    response = _make_image_response()  # bytes([0xAA]*4) — unknown format
    out = response.save(tmp_path / "result")
    assert out.suffix == ".png"


@pytest.mark.parametrize(
    "b64_factory,expected_ext",
    [
        (_real_png_b64, "png"),
        (_real_webp_b64, "webp"),
        (_real_jpeg_b64, "jpg"),
    ],
)
def test_image_save_all_auto_detects_default_ext(tmp_path, b64_factory, expected_ext):
    """save_all() with ext=None should sniff format from first image's bytes."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [b64_factory(), b64_factory()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    paths = response.save_all(tmp_path / "out", prefix="frame")
    assert all(p.suffix == f".{expected_ext}" for p in paths)
    assert len(paths) == 2


def test_image_save_all_explicit_ext_overrides_detection(tmp_path):
    """save_all() with explicit ext should NOT auto-detect."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [_real_webp_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    paths = response.save_all(tmp_path, ext="png")
    assert paths[0].suffix == ".png"


def test_image_save_all_empty_images_returns_empty_list(tmp_path):
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-empty",
            "images": [],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    assert response.save_all(tmp_path) == []


def test_image_save_all_raises_before_writing_when_collision(tmp_path):
    response = _make_image_response(n=3)
    # Pre-create only the second target → save_all must abort before any write.
    (tmp_path / "image_1.png").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        response.save_all(tmp_path)
    # The other targets must NOT have been written.
    assert not (tmp_path / "image_0.png").exists()
    assert not (tmp_path / "image_2.png").exists()
    assert (tmp_path / "image_1.png").read_bytes() == b"old"


# ---------------------------------------------------------------------------
# AudioResponse.save
# ---------------------------------------------------------------------------


def test_audio_save_round_trip(tmp_path):
    response = AudioResponse(content=b"audio-bytes")
    out = response.save(tmp_path / "subdir" / "speech.mp3")
    assert out.read_bytes() == b"audio-bytes"


def test_audio_save_raises_on_existing_without_overwrite(tmp_path):
    response = AudioResponse(content=b"audio-bytes")
    target = tmp_path / "speech.mp3"
    response.save(target)
    with pytest.raises(FileExistsError):
        response.save(target)


def test_audio_save_overwrite_replaces(tmp_path):
    response = AudioResponse(content=b"new")
    target = tmp_path / "speech.mp3"
    target.write_bytes(b"old")
    response.save(target, overwrite=True)
    assert target.read_bytes() == b"new"


# ---------------------------------------------------------------------------
# ImageGenerationResponse.bytes
# ---------------------------------------------------------------------------


def test_image_bytes_round_trip():
    """response.bytes(i) returns exactly the decoded base64 of images[i]."""
    response = _make_image_response(n=3)
    for i in range(3):
        assert response.bytes(i) == base64.b64decode(response.images[i])


def test_image_bytes_default_index_is_zero():
    response = _make_image_response(n=2)
    assert response.bytes() == base64.b64decode(response.images[0])


def test_image_bytes_raises_index_error_out_of_range():
    response = _make_image_response(n=1)
    with pytest.raises(IndexError):
        response.bytes(5)


def test_image_bytes_returns_actual_image_payload_for_real_format():
    """bytes() should return the decoded magic-byte-bearing payload for real PNG/WebP/JPEG."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-1",
            "images": [_real_png_b64(), _real_webp_b64(), _real_jpeg_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    assert response.bytes(0).startswith(b"\x89PNG")
    assert response.bytes(1).startswith(b"RIFF")
    assert response.bytes(2).startswith(b"\xff\xd8\xff")


# ---------------------------------------------------------------------------
# save_all per-image format detection
# ---------------------------------------------------------------------------


def test_image_save_all_detects_format_per_image_when_ext_none(tmp_path):
    """Mixed PNG + WebP + JPEG batch should produce correctly-suffixed files each."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-mixed",
            "images": [_real_png_b64(), _real_webp_b64(), _real_jpeg_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    paths = response.save_all(tmp_path / "out", prefix="mixed")
    assert [p.suffix for p in paths] == [".png", ".webp", ".jpg"]
    # Each saved file should still match its source bytes.
    for i, p in enumerate(paths):
        assert p.read_bytes() == base64.b64decode(response.images[i])


def test_image_save_all_with_explicit_ext_uses_uniform_suffix_for_mixed(tmp_path):
    """Explicit ext should override per-image detection (caller knows best)."""
    response = ImageGenerationResponse.model_validate(
        {
            "id": "img-mixed",
            "images": [_real_png_b64(), _real_webp_b64(), _real_jpeg_b64()],
            "request": None,
            "timing": TimingInfo(
                inferenceDuration=0.0,
                inferencePreprocessingTime=0.0,
                inferenceQueueTime=0.0,
                total=0.0,
            ).model_dump(),
        }
    )
    paths = response.save_all(tmp_path / "out", ext="png")
    assert all(p.suffix == ".png" for p in paths)
