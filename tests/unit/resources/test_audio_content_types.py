"""TDD: OGG/OGA/WebM audio content-types (audit MED #6).

`.ogg` was detected by magic bytes but absent from content_type_map (so it fell
back to application/octet-stream), and WebM/EBML wasn't detected at all.
"""

from unittest.mock import Mock

import pytest

from venice_ai.resources.audio import Audio


@pytest.fixture
def audio() -> Audio:
    return Audio(Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "magic,exp_name,exp_ct",
    [
        (b"OggS\x00\x00\x00\x00", "audio.ogg", "audio/ogg"),
        (b"\x1a\x45\xdf\xa3\x00\x00\x00\x00", "audio.webm", "audio/webm"),
    ],
)
async def test_prepare_detects_ogg_and_webm_from_magic(audio, magic, exp_name, exp_ct):
    _content, name, ct = await audio._prepare_audio_file(magic)
    assert name == exp_name
    assert ct == exp_ct


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fname,exp_ct",
    [("clip.ogg", "audio/ogg"), ("clip.oga", "audio/ogg"), ("clip.webm", "audio/webm")],
)
async def test_prepare_content_type_for_extension(audio, fname, exp_ct, tmp_path):
    p = tmp_path / fname
    p.write_bytes(b"\x00\x01\x02\x03")
    _content, _name, ct = await audio._prepare_audio_file(str(p))
    assert ct == exp_ct
