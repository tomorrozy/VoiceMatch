"""Explainable acoustic descriptors for the four scoring dimensions.

Pitch, timbre, resonance and singing style each get their own module returning a
flat ``dict[str, float]``.  The names are an interface: the feature vector,
percentile mapping and calibration layer all read the descriptors by name, and
the explanation report shows them to the user unchanged.

``extract_features`` assembles the four groups into one ``FeatureVector``;
``fit_percentiles``/``to_percentiles`` turn raw values into 0-100 ranks within a
same-gender reference pool.
"""

from voicematch.features.extract import DIMENSION_OF, FeatureVector, extract_features
from voicematch.features.percentile import (
    PercentileModel,
    fit_percentiles,
    to_percentiles,
)
from voicematch.features.pitch import pitch_features
from voicematch.features.resonance import resonance_features
from voicematch.features.style import style_features
from voicematch.features.timbre import timbre_features

__all__ = [
    "DIMENSION_OF",
    "FeatureVector",
    "PercentileModel",
    "extract_features",
    "fit_percentiles",
    "pitch_features",
    "resonance_features",
    "style_features",
    "timbre_features",
    "to_percentiles",
]
