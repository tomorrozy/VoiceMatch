"""Explainable pitch descriptors: how high, how wide, how steady.

The track is the pipeline's single pYIN call
(:func:`voicematch.audio.pitch.pitch_track`), run with the same pitch range and
frame grid the window ranking and the quality gate use, so a frame is "voiced"
here exactly when it is voiced there (finding I2 collapsed the three copies of
the tracker into one).

A window with no voiced frames has an undefined pitch: every descriptor becomes
``nan`` and no exception is raised.  Deciding that such audio must be refused is
the quality gate's job (and the user sees the reason there); the feature layer
must not throw on content it cannot analyse.
"""

import numpy as np

from voicematch.audio.pitch import pitch_track

#: The descriptor names.  Part of the feature-vector interface: later stages and
#: the explanation report look them up by name.
_PITCH_KEYS = ("median_f0_hz", "range_semitones", "f0_jitter")


def pitch_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Return the pitch descriptors of one audio window.

    - ``median_f0_hz`` — median f0 over voiced frames; the "how high" of a
      voice, robust to the occasional octave error of a frame-level tracker.
    - ``range_semitones`` — ``12 * log2(p95 / p5)`` of the voiced f0; an octave
      of singing reports slightly under 12 because the percentiles trim the
      tracker's edge frames.
    - ``f0_jitter`` — ``std(f0) / mean(f0)`` over the voiced frames; a steady
      tone is near zero, unstable pitch pressure is larger.

    All three are ``nan`` when no frame is voiced (silence, noise, breath).  A
    non-mono array or a non-positive sample rate raises ``ValueError`` instead:
    that is a caller bug, not content.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")

    f0 = pitch_track(y, sr)
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        # No f0 at all: return the NaN dictionary explicitly.  Letting mean,
        # std or percentile run on the empty slice would warn instead.
        return {key: float("nan") for key in _PITCH_KEYS}

    low, high = np.percentile(voiced, (5.0, 95.0))
    return {
        "median_f0_hz": float(np.median(voiced)),
        "range_semitones": float(12.0 * np.log2(high / low)),
        "f0_jitter": float(np.std(voiced) / np.mean(voiced)),
    }
