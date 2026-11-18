"""The versioned singer library: build it from material, save it, load it.

``build_library`` turns ``(Singer, material path)`` pairs into a
``LibraryIndex``: segments cut with the shared automatic window rule, one
``FeatureVector`` and one embedding per segment, one prototype per singer, and a
version that is a structural consequence of the material and the pipeline/model
identity (G4).  ``save_library``/``load_library`` persist that index as one
directory of ``manifest.json`` + ``prototypes.npz`` + ``embeddings.npy``.

Later stages read the index rather than recomputing anything: ranking compares
against ``prototypes``, calibration and explanation read the per-segment
``features`` and ``embeddings``, and every result reports ``version``.
"""

from voicematch.library.build import build_library
from voicematch.library.schema import LibraryIndex, Segment, Singer
from voicematch.library.store import load_library, save_library

__all__ = [
    "LibraryIndex",
    "Segment",
    "Singer",
    "build_library",
    "load_library",
    "save_library",
]
