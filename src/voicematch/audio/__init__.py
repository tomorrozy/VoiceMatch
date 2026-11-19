"""Audio decoding and the symmetric preprocessing pipeline.

Both the singer library and user queries must pass through exactly the same
code, so the preprocess configuration and ``pipeline_signature`` live here and
are shared by later stages.
"""

from voicematch.audio.io import load_audio
from voicematch.audio.preprocess import (
    PREPROCESS_LIBRARY,
    PREPROCESS_QUERY,
    PreprocessConfig,
    pipeline_signature,
    preprocess,
    separate_vocals,
)

__all__ = [
    "PREPROCESS_LIBRARY",
    "PREPROCESS_QUERY",
    "PreprocessConfig",
    "load_audio",
    "pipeline_signature",
    "preprocess",
    "separate_vocals",
]
