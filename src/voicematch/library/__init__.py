"""The versioned singer library: build it from material, save it, load it.

``build_library`` turns ``(Singer, material path)`` pairs into a
``LibraryIndex``: segments cut with the shared automatic window rule, one
``FeatureVector`` and one embedding per segment, one prototype per singer, the
identity of the encoder that produced the embeddings (``model_id``, R24), and a
version that is a structural consequence of the material, the pipeline and that
encoder identity (G4).  ``save_library``/``load_library`` persist that index as
one directory of ``manifest.json`` + ``prototypes.npz`` + ``embeddings.npy``.

Later stages read the index rather than recomputing anything: ranking compares
against ``prototypes``, calibration and explanation read the per-segment
``features`` and ``embeddings``, and every result reports ``version``.  The
query path checks ``model_id`` against its own encoder, so a library is never
ranked with vectors from a different model (R24).

``pairing_status`` reads the artefact itself to say whether the library can
supply cross-recording calibration pairs (R29), so a query can report "this
library cannot be calibrated" instead of only returning ``calibrated: false``
with no reason.
"""

from voicematch.library.build import (
    CROSS_RECORDING,
    EMPTY_LIBRARY,
    SINGLE_RECORDING,
    UNVERIFIABLE_LEGACY,
    PairingStatus,
    SingleMaterialSingerWarning,
    build_library,
    pairing_status,
)
from voicematch.library.catalog import CatalogError, load_catalog
from voicematch.library.intake import (
    IntakeError,
    ingest,
    load_manifest,
    read_catalog_rights,
)
from voicematch.library.schema import LibraryIndex, Segment, Singer
from voicematch.library.store import load_library, save_library

__all__ = [
    "CROSS_RECORDING",
    "CatalogError",
    "EMPTY_LIBRARY",
    "IntakeError",
    "LibraryIndex",
    "PairingStatus",
    "SINGLE_RECORDING",
    "Segment",
    "Singer",
    "SingleMaterialSingerWarning",
    "UNVERIFIABLE_LEGACY",
    "build_library",
    "ingest",
    "load_catalog",
    "load_library",
    "load_manifest",
    "pairing_status",
    "read_catalog_rights",
    "save_library",
]
