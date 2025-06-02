"""
Type definitions for Venice AI Audio API.

This module contains TypedDict definitions and Enums for request objects
in the Venice AI Audio API, covering the speech creation endpoint.
"""

from enum import Enum
from typing import Optional, TypedDict


class Voice(str, Enum):
    """
    Available voices for speech generation in the Venice AI Audio API.
    
    This enumeration defines the complete set of voice options that can be used
    when generating speech from text via the text-to-speech endpoint. Each voice
    represents different speaker characteristics including gender, accent, and
    vocal qualities. Voice names follow a pattern indicating language/region
    and gender (e.g., ``af`` for American Female, ``am`` for American Male).
    """
    
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
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"
    AM_SANTA = "am_santa"
    BF_ALICE = "bf_alice"
    BF_EMMA = "bf_emma"
    BF_LILY = "bf_lily"
    BM_DANIEL = "bm_daniel"
    BM_FABLE = "bm_fable"
    BM_GEORGE = "bm_george"
    BM_LEWIS = "bm_lewis"
    ZF_XIAOBEI = "zf_xiaobei"
    ZF_XIAONI = "zf_xiaoni"
    ZF_XIAOXIAO = "zf_xiaoxiao"
    ZF_XIAOYI = "zf_xiaoyi"
    ZM_YUNJIAN = "zm_yunjian"
    ZM_YUNXI = "zm_yunxi"
    ZM_YUNXIA = "zm_yunxia"
    ZM_YUNYANG = "zm_yunyang"
    FF_SIWIS = "ff_siwis"
    HF_ALPHA = "hf_alpha"
    HF_BETA = "hf_beta"
    HM_OMEGA = "hm_omega"
    HM_PSI = "hm_psi"
    IF_SARA = "if_sara"
    IM_NICOLA = "im_nicola"
    JF_ALPHA = "jf_alpha"
    JF_GONGITSUNE = "jf_gongitsune"
    JF_NEZUMI = "jf_nezumi"
    JF_TEBUKURO = "jf_tebukuro"
    JM_KUMO = "jm_kumo"
    PF_DORA = "pf_dora"
    PM_ALEX = "pm_alex"
    PM_SANTA = "pm_santa"
    EF_DORA = "ef_dora"
    EM_ALEX = "em_alex"
    EM_SANTA = "em_santa"
    
    # Aliases for backward compatibility with existing code and tests
    NOVA = AF_NOVA  # Alias for AF_NOVA (American Female Nova voice)
    ALLOY = AF_ALLOY  # Alias for AF_ALLOY (American Female Alloy voice)
    ONYX = AM_ONYX  # Alias for AM_ONYX (American Male Onyx voice)
    SHIMMER = AF_RIVER  # Alias for AF_RIVER (approximation for legacy SHIMMER voice)


class ResponseFormat(str, Enum):
    """
    Available audio response formats for speech generation output.
    
    This enumeration defines the supported audio file formats that can be
    requested when generating speech from text. The format determines the
    encoding, compression, and quality characteristics of the returned audio
    data from the text-to-speech endpoint. Different formats offer trade-offs
    between file size, quality, and compatibility.
    """
    
    MP3 = "mp3"
    AAC = "aac"
    OPUS = "opus"
    FLAC = "flac"
    WAV = "wav"


class CreateSpeechRequest(TypedDict, total=False):
    """
    Request parameters for creating speech audio from text input.
    
    This TypedDict defines the structure for requests to the POST /audio/speech
    endpoint, which converts text into spoken audio using specified voice
    characteristics and output format. The request allows customization of
    voice selection, audio format, playback speed, and user identification
    for tracking purposes.
    
    Attributes:
        model: ID of the model to use for speech generation (e.g., "tts-kokoro").
        input: The text to convert to speech. Maximum length varies by model.
        voice: The voice to use for the generated audio. See :class:`~Voice` for available options.
        response_format: Optional. The format to return the audio in. Defaults to "mp3".
            See :class:`~ResponseFormat` for available formats.
        speed: Optional. The speed of the generated audio. Select a value from 0.25 to 4.0.
            Defaults to 1.0.
        user: Optional. A unique identifier representing the end-user, which can help Venice AI
            to monitor and detect abuse.
    """
    
    model: str  # Required: ID of the speech model to use for generation
    input: str  # Required: The text to convert to speech
    voice: Voice  # Required: Voice to use for the generated audio
    response_format: ResponseFormat  # Optional: Format of returned audio (defaults to "mp3")
    speed: Optional[float]  # Optional: Speed of the generated audio (0.25-4.0, defaults to 1.0)
    user: Optional[str]  # Optional: Unique identifier representing the end-user for monitoring