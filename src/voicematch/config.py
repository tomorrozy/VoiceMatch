"""Pipeline-wide constants.

These values are deliberately hard-coded rather than read from environment
variables: results must be reproducible from the code revision alone.
Changing any value here changes the meaning of every stored embedding and
therefore requires a new ``PIPELINE_VERSION``.
"""

SAMPLE_RATE: int = 24000
LUFS_TARGET: float = -23.0
WINDOW_SEC: float = 10.0
MIN_SPEECH_SEC: float = 8.0
MAX_INPUT_SEC: float = 60.0
FRAME_LENGTH: int = 1024
HOP_LENGTH: int = 256
EMBED_DIM: int = 256
# v2: ``preprocess`` applies its peak limiter, which alters the signal every
# stored embedding was computed from; the library version hash includes this
# constant (G4), so the limiter had to move it.
PIPELINE_VERSION: str = "pipeline-v2"
