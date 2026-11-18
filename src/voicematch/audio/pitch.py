"""The pipeline's single pYIN pitch track, shared by ranking, gating and features.

Four stages ask "which frames are pitched?": the window selector ranks candidate
windows by their voiced share, the quality gate refuses audio with too few
pitched frames (design 5.6's fourth rejection criterion), the feature layer
derives its f0 descriptors, and the style layer measures vibrato from the same
track.  Each of them must ask the same question of the same frame grid, so the
tracker is defined once here instead of being copied into each module (finding
I2: the gate's copy was energy-based, so unpitched hiss passed it while every
pitch descriptor was ``nan``).

The frame grid and the pitch range come from :mod:`voicematch.config` and are
part of the pipeline identity: changing either changes which frames count as
voiced, and therefore every stored feature vector.
"""

import librosa
import numpy as np

from voicematch.config import FRAME_LENGTH, HOP_LENGTH

#: Pitch-tracking range.  It spans low male voices and high female voices, and
#: is shared by ranking, gating and the f0 descriptors so "voiced" means the
#: same thing everywhere.
F0_MIN_HZ: float = 65.0
F0_MAX_HZ: float = 1000.0


def pitch_track(y: np.ndarray, sr: int) -> np.ndarray:
    """Return f0 per frame (``nan`` where unvoiced) on the pipeline frame grid.

    This is the only pYIN call in the package.  It is not validated here: every
    caller validates its own input (mono, positive rate) before measuring
    anything, exactly as it does for the other stage helpers.
    """
    f0, _voiced_flag, _probability = librosa.pyin(
        y,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    return f0
