"""Consolidated Enumerations for Venice AI Types.

This module provides all enumeration types used across the Venice AI platform,
consolidating previously scattered enum definitions into a single, well-organized
location. These enums provide type-safe constants for various API operations and
configurations.

**Available Enumerations:**

* **Voice**: 80+ voice options for text-to-speech generation across multiple languages
* **ResponseFormat**: Audio output format options (MP3, AAC, OPUS, FLAC, WAV, PCM)
* **BillingFormatEnum**: Billing data export formats (JSON, CSV)
* **ModelType**: AI model type categories (TEXT, IMAGE, EMBEDDING, TTS, UPSCALE, INPAINT)

**Design Philosophy:**

This module serves as the single source of truth for all enumeration types,
making it easy to:

* Discover all available enum options in one place
* Maintain consistency across the codebase
* Avoid duplication and synchronization issues
* Provide comprehensive documentation for all enum values

**Usage Pattern:**

Enums should be imported from this module rather than from legacy locations::

    from venice_ai.types.enums import Voice, ResponseFormat, BillingFormatEnum, ModelType

    # Use in API calls
    voice = Voice.AF_ALLOY
    format = ResponseFormat.MP3
    billing_format = BillingFormatEnum.JSON
    model_type = ModelType.TEXT

**Backward Compatibility:**

Legacy import paths are maintained for backward compatibility through re-exports
in the original modules (audio.py, billing.py, models.py), but new code should
import directly from this module.
"""

from enum import StrEnum

from venice_ai.core.models.common import ModelType as ModelType  # noqa: F401 — re-exported

__all__ = [
    "Voice",
    "ResponseFormat",
    "BillingFormatEnum",
    "ModelType",
    # Video-related enums
    "VideoModelType",
    "VideoStatus",
    "VideoDuration",
    "VideoResolution",
    "VideoPrivacy",
    "VideoAspectRatio",
]


# ============================================================================
# Audio Enumerations
# ============================================================================


class Voice(StrEnum):
    """Kokoro TTS voices for speech generation in the Venice AI Audio API.

    This enumeration defines the **Kokoro** TTS voice options. For other TTS
    models (Qwen3, xAI, Orpheus, Inworld, Chatterbox, ElevenLabs, MiniMax,
    Gemini, Gradium), pass voice names as plain string literals — the ``voice``
    request field is an open ``str``, so any documented voice is accepted; see
    the swagger voice enum for the full list. Each Kokoro voice represents
    different speaker characteristics including gender, accent, and vocal
    qualities. Voice names follow a pattern indicating language/region and
    gender (e.g., ``af`` for American Female, ``am`` for American Male).

    **Voice Naming Convention:**

    * **Region Prefix**: Two-letter code indicating language/locale and gender
    * **Voice Name**: Unique identifier for the specific voice character

    **Region Code Patterns:**

    * ``af_*``: American Female voices
    * ``am_*``: American Male voices
    * ``bf_*``: British Female voices
    * ``bm_*``: British Male voices
    * ``zf_*``: Chinese (Mandarin) Female voices
    * ``zm_*``: Chinese (Mandarin) Male voices
    * ``ff_*``: French Female voices
    * ``hf_*``: Hindi Female voices
    * ``hm_*``: Hindi Male voices
    * ``if_*``: Italian Female voices
    * ``im_*``: Italian Male voices
    * ``jf_*``: Japanese Female voices
    * ``jm_*``: Japanese Male voices
    * ``pf_*``: Portuguese Female voices
    * ``pm_*``: Portuguese Male voices
    * ``ef_*``: Spanish Female voices
    * ``em_*``: Spanish Male voices

    **Use Cases:**

    * Selecting appropriate voices for different languages and content types
    * Creating region-specific or gender-specific audio content
    * Building voice selection interfaces with language and gender filters
    * Implementing voice recommendation systems based on content characteristics

    **Example:**

    ::

        from venice_ai.types.enums import Voice

        # Select American female voice
        voice = Voice.AF_NOVA

        # Select British male voice
        voice = Voice.BM_GEORGE

        # Select Chinese female voice
        voice = Voice.ZF_XIAOXIAO
    """

    # American Female Voices
    AF_ALLOY = "af_alloy"
    AF_AOEDE = "af_aoede"
    AF_BELLA = "af_bella"
    AF_HEART = "af_heart"
    AF_JADZIA = "af_jadzia"
    AF_JESSICA = "af_jessica"
    AF_KORE = "af_kore"
    AF_NICOLE = "af_nicole"
    AF_NOVA = "af_nova"
    AF_RIVER = "af_river"
    AF_SARAH = "af_sarah"
    AF_SKY = "af_sky"

    # American Male Voices
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"
    AM_SANTA = "am_santa"

    # British Female Voices
    BF_ALICE = "bf_alice"
    BF_EMMA = "bf_emma"
    BF_LILY = "bf_lily"

    # British Male Voices
    BM_DANIEL = "bm_daniel"
    BM_FABLE = "bm_fable"
    BM_GEORGE = "bm_george"
    BM_LEWIS = "bm_lewis"

    # Chinese (Mandarin) Female Voices
    ZF_XIAOBEI = "zf_xiaobei"
    ZF_XIAONI = "zf_xiaoni"
    ZF_XIAOXIAO = "zf_xiaoxiao"
    ZF_XIAOYI = "zf_xiaoyi"

    # Chinese (Mandarin) Male Voices
    ZM_YUNJIAN = "zm_yunjian"
    ZM_YUNXI = "zm_yunxi"
    ZM_YUNXIA = "zm_yunxia"
    ZM_YUNYANG = "zm_yunyang"

    # French Female Voices
    FF_SIWIS = "ff_siwis"

    # Hindi Female Voices
    HF_ALPHA = "hf_alpha"
    HF_BETA = "hf_beta"

    # Hindi Male Voices
    HM_OMEGA = "hm_omega"
    HM_PSI = "hm_psi"

    # Italian Female Voices
    IF_SARA = "if_sara"

    # Italian Male Voices
    IM_NICOLA = "im_nicola"

    # Japanese Female Voices
    JF_ALPHA = "jf_alpha"
    JF_GONGITSUNE = "jf_gongitsune"
    JF_NEZUMI = "jf_nezumi"
    JF_TEBUKURO = "jf_tebukuro"

    # Japanese Male Voices
    JM_KUMO = "jm_kumo"

    # Portuguese Female Voices
    PF_DORA = "pf_dora"

    # Portuguese Male Voices
    PM_ALEX = "pm_alex"
    PM_SANTA = "pm_santa"

    # Spanish Female Voices
    EF_DORA = "ef_dora"

    # Spanish Male Voices
    EM_ALEX = "em_alex"
    EM_SANTA = "em_santa"


class ResponseFormat(StrEnum):
    """Available audio response formats for speech generation output.

    This enumeration defines the supported audio file formats that can be
    requested when generating speech from text. The format determines the
    encoding, compression, and quality characteristics of the returned audio
    data from the text-to-speech endpoint. Different formats offer trade-offs
    between file size, quality, and compatibility.

    **Format Characteristics:**

    * **MP3**: Lossy compression, widely compatible, good balance of quality and size
    * **AAC**: Advanced lossy compression, better quality than MP3 at same bitrate
    * **OPUS**: Modern lossy codec, excellent for speech, low latency
    * **FLAC**: Lossless compression, highest quality, larger file size
    * **WAV**: Uncompressed PCM audio, maximum quality, very large files
    * **PCM**: Raw uncompressed audio data, no container format

    **Use Cases:**

    * **Web applications**: MP3 or AAC for broad browser compatibility
    * **High-quality archival**: FLAC or WAV for maximum fidelity
    * **Real-time streaming**: OPUS for low latency and efficient bandwidth usage
    * **Audio processing**: WAV or PCM for raw audio manipulation
    * **Mobile applications**: AAC or OPUS for efficient bandwidth usage

    **Quality vs. Size Trade-offs:**

    * Smallest: OPUS, AAC, MP3
    * Medium: FLAC
    * Largest: WAV, PCM

    **Example:**

    ::

        from venice_ai.types.enums import ResponseFormat

        # For web playback
        format = ResponseFormat.MP3

        # For high-quality archival
        format = ResponseFormat.FLAC

        # For real-time streaming
        format = ResponseFormat.OPUS
    """

    MP3 = "mp3"
    """MP3 format - widely compatible lossy compression.

    MPEG-1 Audio Layer 3 format provides good quality with moderate
    file sizes and universal compatibility across devices and platforms.
    Ideal for general-purpose web applications and content distribution.
    """

    AAC = "aac"
    """AAC format - advanced lossy compression with better quality than MP3.

    Advanced Audio Coding provides superior quality compared to MP3 at
    equivalent bitrates, with excellent compatibility on modern devices.
    Recommended for mobile applications and quality-focused web use.
    """

    OPUS = "opus"
    """OPUS format - modern codec optimized for speech and low latency.

    Highly efficient lossy codec specifically optimized for speech and
    music, offering excellent quality at low bitrates with minimal latency.
    Perfect for real-time streaming and bandwidth-constrained applications.
    """

    FLAC = "flac"
    """FLAC format - lossless compression for maximum quality.

    Free Lossless Audio Codec provides bit-perfect audio reproduction
    with compression, maintaining original quality while reducing file size.
    Ideal for archival purposes and applications requiring pristine audio.
    """

    WAV = "wav"
    """WAV format - uncompressed PCM audio with container.

    Waveform Audio File Format contains uncompressed PCM audio data,
    providing maximum compatibility and ease of processing. Suitable
    for audio editing and systems requiring standard uncompressed audio.
    """

    PCM = "pcm"
    """PCM format - raw uncompressed audio data without container.

    Pulse-Code Modulation provides raw audio samples without any
    container format overhead. Useful for low-level audio processing
    and systems requiring direct access to sample data.
    """


# ============================================================================
# Billing Enumerations
# ============================================================================


class BillingFormatEnum(StrEnum):
    """Defines available output formats for billing usage data responses.

    This enumeration specifies the complete set of supported data formats that can be
    requested when retrieving billing usage information from the Venice AI API. Each
    format is optimized for different use cases, from programmatic processing to data
    analysis and reporting, providing flexibility in how billing data is consumed
    and integrated into various systems and workflows.

    **Format Selection Strategy:**

    * **JSON**: Ideal for programmatic access, API integrations, and web applications
    * **CSV**: Perfect for data analysis, spreadsheet import, and business intelligence tools

    **Format Details:**

    * **JSON format**: Returns a structured `BillingUsageHistoryResponse` with full type
      information, a ``nextCursor`` continuation token, and nested objects. Suitable for
      applications requiring precise data types, validation, and programmatic manipulation.

    * **CSV format**: Returns raw comma-separated values as bytes, optimized for import
      into Excel, Google Sheets, or data analysis tools. Includes headers and flattened
      data structure for easy consumption by spreadsheet applications.

    **Use Cases:**

    * **Application Integration**: JSON format for real-time billing data access in applications
    * **Business Analytics**: CSV format for importing into Excel, Tableau, or other BI tools
    * **Reporting Systems**: Format selection based on downstream processing requirements
    * **Data Export**: CSV for sharing billing data with external systems and stakeholders
    * **Automated Processing**: JSON for automated billing workflows and integrations
    * **Financial Reconciliation**: CSV for importing into accounting systems

    **API Integration:**

    Used to specify the desired response format in billing usage API requests,
    allowing clients to receive data in the most appropriate format for their
    specific use case and processing requirements.

    **Example:**

    ::

        from venice_ai.types.enums import BillingFormatEnum

        # For programmatic access
        format = BillingFormatEnum.JSON

        # For spreadsheet analysis
        format = BillingFormatEnum.CSV
    """

    JSON = "json"
    """JSON format - returns structured data as BillingUsageHistoryResponse.

    Provides fully structured billing data with complete type information, a
    ``nextCursor`` continuation token, and nested objects. Ideal for programmatic
    access, web applications, and systems requiring precise data types and
    validation. The response follows the standard BillingUsageHistoryResponse
    model structure with full Pydantic validation and type safety.

    **Response Structure:**

    * Type: `BillingUsageHistoryResponse`
    * Includes: nextCursor continuation token, detailed usage entries, inference details
    * Validation: Full Pydantic model validation
    * Format: Structured JSON with nested objects
    """

    CSV = "csv"
    """CSV format - returns raw CSV data as bytes for export and analysis purposes.

    Provides billing data in comma-separated values format, optimized for
    import into spreadsheets, business intelligence tools, and data analysis
    platforms. Returns raw bytes data that can be written directly to files
    or processed through CSV parsing libraries. Headers include all relevant
    billing fields for comprehensive data analysis.

    **Response Structure:**

    * Type: `bytes`
    * Includes: CSV headers and flattened data rows
    * Validation: None (raw bytes)
    * Format: RFC 4180 compliant CSV with headers

    **CSV Columns:**

    Typically includes: timestamp, amount, currency, SKU, units, price per unit,
    notes, and inference details (when applicable).
    """


# ============================================================================
# Model Enumerations
# ============================================================================

# ModelType is re-exported from venice_ai.core.models.common (canonical location)
# via the import at the top of this file. This avoids duplication while
# maintaining backward compatibility for ``from venice_ai.types.enums import ModelType``.


# ============================================================================
# Video Enumerations
# ============================================================================


class VideoModelType(StrEnum):
    """
    Video generation model subtypes.

    Distinguishes between text-to-video and image-to-video models.
    API-validated values from 31 video models.
    """

    TEXT_TO_VIDEO = "text-to-video"
    IMAGE_TO_VIDEO = "image-to-video"


class VideoStatus(StrEnum):
    """
    Video generation queue status values.

    Returned by the retrieve endpoint to indicate generation progress.
    """

    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class VideoDuration(StrEnum):
    """
    Supported video duration options.

    API-validated: These are ALL durations found across 31 video models.
    Individual models support subsets - check model.model_spec.constraints.durations.
    """

    SECONDS_4 = "4s"
    SECONDS_5 = "5s"
    SECONDS_6 = "6s"
    SECONDS_8 = "8s"
    SECONDS_10 = "10s"
    SECONDS_12 = "12s"
    SECONDS_14 = "14s"
    SECONDS_15 = "15s"
    SECONDS_16 = "16s"
    SECONDS_18 = "18s"
    SECONDS_20 = "20s"
    SECONDS_30 = "30s"


class VideoResolution(StrEnum):
    """
    Supported video resolution options.

    API-validated: These are ALL resolutions found across 31 video models.
    Individual models support subsets - check model.model_spec.constraints.resolutions.
    Some models have empty resolutions (resolution determined by model/input).
    """

    P480 = "480p"
    P580 = "580p"
    P720 = "720p"
    P1080 = "1080p"
    P1440 = "1440p"
    P2160 = "2160p"


class VideoPrivacy(StrEnum):
    """
    Video model privacy levels.

    API-validated values:
    - PRIVATE: No prompt data is stored in any capacity
    - ANONYMIZED: Provider maintains anonymized prompt data
    """

    PRIVATE = "private"
    ANONYMIZED = "anonymized"


class VideoAspectRatio(StrEnum):
    """
    Supported video aspect ratio options.

    API-validated: These are ALL aspect ratios found across 31 video models.
    Individual models support subsets - check model.model_spec.constraints.aspect_ratios.
    Image-to-video models often have empty aspect_ratios (uses input image ratio).
    """

    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"
    RATIO_4_3 = "4:3"
    RATIO_3_4 = "3:4"
