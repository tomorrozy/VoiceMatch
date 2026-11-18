"""Frozen vocal encoders that turn one audio window into one similarity vector.

``VocalEncoder`` is the interface later stages type against; ``HubertEncoder``
is the frozen HuBERT-base backbone used in production, and ``HashEncoder`` is
the deterministic stand-in that keeps everything runnable without the optional
``ml`` extra.  ``pool_frames`` is the fixed mean+std reduction between the
backbone's frame features and the stored embedding (ruling R3).
"""

from voicematch.embed.encoder import (
    MODEL_ID,
    HashEncoder,
    HubertEncoder,
    VocalEncoder,
)
from voicematch.embed.pooling import pool_frames

__all__ = [
    "MODEL_ID",
    "HashEncoder",
    "HubertEncoder",
    "VocalEncoder",
    "pool_frames",
]
