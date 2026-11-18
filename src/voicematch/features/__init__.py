"""Explainable acoustic descriptors for the four scoring dimensions.

Pitch, timbre, resonance and singing style each get their own module returning a
flat ``dict[str, float]``.  The names are an interface: the feature vector,
percentile mapping and calibration layer all read the descriptors by name, and
the explanation report shows them to the user unchanged.
"""

from voicematch.features.pitch import pitch_features
from voicematch.features.timbre import timbre_features

__all__ = ["pitch_features", "timbre_features"]
