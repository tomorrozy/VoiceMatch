"""The aggregated feature vector: four dimensions, one value per descriptor.

The four descriptor producers stay independent; this module only assembles
their dictionaries into the single ``FeatureVector`` that the library builder,
the calibration layer and the explanation report pass around.  Assembly is
deliberately trivial: a descriptor that a producer could not measure is carried
as ``nan``, never replaced by a default.  The percentile mapping reads ``nan``
as "not comparable", whereas 0 or 100 would invent a position in the pool.

``DIMENSION_OF`` is the explicit descriptor-to-dimension table.  It is written
out by hand rather than derived from name prefixes so that a renamed or newly
added descriptor must be registered here: a descriptor missing from the table
would silently drop out of the dimension aggregation and the report.
"""

from dataclasses import dataclass

import numpy as np

from voicematch.features.pitch import pitch_features
from voicematch.features.resonance import resonance_features
from voicematch.features.style import style_features
from voicematch.features.timbre import timbre_features

#: Every descriptor name, mapped to the scoring dimension it belongs to.  This
#: is an interface, not a convenience: the calibration layer aggregates
#: per-descriptor deltas through it, so each name must appear exactly once.
DIMENSION_OF: dict[str, str] = {
    # pitch
    "median_f0_hz": "pitch",
    "range_semitones": "pitch",
    "f0_jitter": "pitch",
    # timbre
    "mfcc1_mean": "timbre",
    "mfcc2_mean": "timbre",
    "mfcc3_mean": "timbre",
    "mfcc4_mean": "timbre",
    "mfcc5_mean": "timbre",
    "mfcc6_mean": "timbre",
    "mfcc7_mean": "timbre",
    "mfcc8_mean": "timbre",
    "mfcc9_mean": "timbre",
    "mfcc10_mean": "timbre",
    "mfcc11_mean": "timbre",
    "mfcc12_mean": "timbre",
    "mfcc13_mean": "timbre",
    "spectral_centroid_hz": "timbre",
    "spectral_contrast_mean": "timbre",
    "ltas_slope": "timbre",
    # resonance
    "f1_hz": "resonance",
    "f2_hz": "resonance",
    "f3_hz": "resonance",
    "low_band_ratio": "resonance",
    "hnr_db": "resonance",
    # style
    "vibrato_rate_hz": "style",
    "vibrato_extent_cents": "style",
    "slur_ratio": "style",
    "onset_hardness": "style",
    "breathiness": "style",
}


@dataclass(frozen=True)
class FeatureVector:
    """One audio window's descriptors, grouped by scoring dimension.

    Each group is the dictionary returned by the corresponding producer, keyed
    by the descriptor names registered in ``DIMENSION_OF``.  A missing
    measurement is ``nan``; nothing is ever filled in.
    """

    pitch: dict[str, float]
    timbre: dict[str, float]
    resonance: dict[str, float]
    style: dict[str, float]


def extract_features(x: np.ndarray, sr: int) -> FeatureVector:
    """Return the four descriptors groups of one audio window.

    This only joins the four producer outputs.  Content they cannot analyse
    comes back as ``nan`` and is preserved as such; a non-mono array or a
    non-positive sample rate raises ``ValueError`` (the producers' caller-bug
    contract), not ``nan``.
    """
    return FeatureVector(
        pitch=pitch_features(x, sr),
        timbre=timbre_features(x, sr),
        resonance=resonance_features(x, sr),
        style=style_features(x, sr),
    )
