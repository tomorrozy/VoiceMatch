"""Deterministic reduction of frame-level features to one window embedding.

A frozen self-supervised backbone emits one vector per frame; a scored segment
needs a single vector, so the frames are pooled.  The plan says "mean+std 拼接后
降维到 ``EMBED_DIM``" without saying how the two statistics are squeezed into
256 values.  Ruling R3 fixes that here: take the first ``EMBED_DIM // 2`` = 128
dimensions of the frame mean, followed by the first 128 dimensions of the frame
std.

Those slices are fixed on purpose.  A random projection — seeded or not —
would mix all 768 hidden dimensions into every output dimension through a
matrix, so a library rebuilt with a different ``numpy``/BLAS/projection matrix
could produce different vectors for the same audio and G4 ("same audio + same
versions → byte-identical output") would be lost.  Changing the slice layout is
therefore a library-version event — bump ``PIPELINE_VERSION`` and rebuild the
library — not a tuning knob.
"""

import numpy as np

from voicematch.config import EMBED_DIM

#: Dimensions taken from each statistic.  ``EMBED_DIM`` is even by construction:
#: the reduction is 128 mean + 128 std.
_HALF: int = EMBED_DIM // 2


def pool_frames(frames: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Pool ``(n_frames, n_dims)`` frame features into one ``(EMBED_DIM,)`` vector.

    The output is ``mean[:128] ++ std[:128]`` over the frames, computed
    per feature dimension.  ``weights`` optionally reweights the frames (for
    example by frame energy or attention); ``None`` means uniform weights.  The
    weighted statistics are ``sum(w·x)/sum(w)`` and
    ``sqrt(sum(w·(x-mean)²)/sum(w))`` — population form, so a single frame has
    a zero std rather than a division by zero.

    Degenerate cases are defined rather than left to warnings: one frame or a
    run of identical frames pools to ``[frame[:128], zeros]`` (std is exactly
    zero, no ``nan``).  Zero frames has no mean at all and raises
    ``ValueError``; so does a frame tensor narrower than 128 dimensions, since
    padding it would invent representation dimensions.  The result is
    ``float32`` to feed the stored embedding directly.
    """
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"expected a 2-D (frames, features) array, got shape {values.shape}"
        )
    n_frames, n_dims = values.shape
    if n_frames == 0:
        raise ValueError("cannot pool zero frames: the window has no audio frames")
    if n_dims < _HALF:
        raise ValueError(
            f"frames need at least {_HALF} feature dimensions to fill "
            f"EMBED_DIM, got {n_dims}"
        )

    if weights is None:
        frame_weights = np.ones(n_frames, dtype=np.float64)
    else:
        frame_weights = np.asarray(weights, dtype=np.float64)
        if frame_weights.ndim != 1 or frame_weights.shape[0] != n_frames:
            raise ValueError(
                "weights must be 1-D with one value per frame, got shape "
                f"{frame_weights.shape}"
            )
        if not np.all(np.isfinite(frame_weights)) or np.any(frame_weights < 0.0):
            raise ValueError("weights must be finite and non-negative")

    total = float(frame_weights.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")

    mean = (frame_weights[:, None] * values).sum(axis=0) / total
    std = np.sqrt((frame_weights[:, None] * (values - mean) ** 2).sum(axis=0) / total)
    return np.concatenate([mean[:_HALF], std[:_HALF]]).astype(np.float32)
