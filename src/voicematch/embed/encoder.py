"""Frozen vocal encoders: the similarity backbone behind "who does this sound like".

Two encoders implement the same :class:`VocalEncoder` interface:

- :class:`HubertEncoder` is the production encoder.  It runs frozen HuBERT-base
  (LibriSpeech 960h, not fine-tuned), takes the hidden state of the ninth
  transformer layer, pools the frames with :func:`~voicematch.embed.pooling.pool_frames`
  and L2-normalises the result.  Per the design, the backbone is never trained
  in v1 — only the calibration layer on top of it is.
- :class:`HashEncoder` is a deterministic stand-in.  It has no perceptual meaning
  but produces a stable, unit-norm, near-orthogonal-per-input vector, which lets
  the library builder, ranking and evaluation tests run without torch.

Reproducibility (G4): the backbone is frozen and the pooling reduction is fixed
(ruling R3).  The values that define what an embedding means — :data:`MODEL_ID`,
:data:`MODEL_SAMPLE_RATE` and :data:`MODEL_LAYER` — are module constants, with
:data:`MODEL_ID` recorded alongside every stored result.

torch/torchaudio are an optional extra (``voicematch[ml]``).  They are imported
when :class:`HubertEncoder` is constructed, never when this module is imported,
so the rest of the package — and its tests — work without them.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import numpy as np

from voicematch.config import EMBED_DIM, FRAME_LENGTH
from voicematch.embed.pooling import pool_frames

#: The frozen backbone's identifier.  Stored next to every embedding; a value
#: change invalidates every stored vector and needs a new ``PIPELINE_VERSION``.
MODEL_ID: str = "HUBERT_BASE"

#: The rate HuBERT expects.  :meth:`HubertEncoder.encode` resamples pipeline
#: audio (24 kHz) to this, for library and query audio through the same code.
MODEL_SAMPLE_RATE: int = 16000

#: Transformer layer whose hidden state is pooled.  This is ``hidden_states[9]``
#: in HuggingFace notation (where ``hidden_states[0]`` is the feature-projection
#: output): the output of the ninth encoder layer.  torchaudio's
#: ``extract_features`` returns only the layer outputs, so it is index 8 there.
MODEL_LAYER: int = 9

#: ``HashEncoder`` hashes one pipeline frame of float32 samples per block.
_HASH_BLOCK_BYTES: int = FRAME_LENGTH * 4

#: Maps a 16-bit unsigned integer exactly onto ``[-1, 1]``.
_UINT16_SCALE: float = 2.0 / 65535.0


@runtime_checkable
class VocalEncoder(Protocol):
    """The encoder interface the library builder and the query path share.

    ``encode`` takes mono audio sampled at ``sr`` and returns a unit-norm
    ``(EMBED_DIM,)`` ``float32`` vector; the cosine similarity of two such
    vectors is the embedding part of the similarity score.
    """

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Return the L2-normalised ``(EMBED_DIM,)`` representation of ``x``."""
        ...


class HashEncoder(VocalEncoder):
    """Deterministic hash-based stand-in for :class:`HubertEncoder`.

    This is a test double, not an embedding model: it exists so that the library
    builder, ranking, calibration and evaluation can be exercised without the
    ``ml`` extra (a 360 MB download and torch).  It must never be used to
    produce a real score.

    The scheme is pure hashing with no random seed, so it is reproducible on any
    machine and any run (G4): the audio bytes are hashed one pipeline frame at a
    time, the sample rate is hashed into every block, the per-block digests are
    concatenated into one whole-recording seed, and SHA-256 counter mode
    deterministically expands that seed into ``EMBED_DIM`` 16-bit values that
    are mapped onto ``[-1, 1]`` and L2-normalised.  Because every component is
    at least ~1.5e-5 away from zero the mapped vector is never the zero vector,
    so normalisation cannot divide by zero.  Distinct recordings give
    near-orthogonal vectors (the dot product of two independent ones concentrates
    around 0), while the same ``(samples, sample rate)`` always gives the same
    bytes.
    """

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
        if sr <= 0:
            raise ValueError(f"sample rate must be positive, got {sr}")

        raw = _expand(_digest_stream(y, sr), EMBED_DIM * 2)
        units = np.frombuffer(raw, dtype=">u2").astype(np.float64)
        vector = units * _UINT16_SCALE - 1.0
        vector /= np.linalg.norm(vector)
        return vector.astype(np.float32)


class HubertEncoder(VocalEncoder):
    """Frozen HuBERT-base encoder — the production similarity backbone.

    The model is loaded once per process (see :func:`_load_model`) and is never
    trained or fine-tuned.  Audio arrives at the pipeline rate
    (:data:`~voicematch.config.SAMPLE_RATE`); resampling to HuBERT's 16 kHz
    happens here, inside ``encode``, so the library side and the query side run
    exactly the same code and cannot disagree about the representation.
    """

    def __init__(self) -> None:
        self._model = _load_model()

    def encode(self, x: np.ndarray, sr: int) -> np.ndarray:
        import torch
        import torchaudio.functional

        y = np.asarray(x, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
        if sr <= 0:
            raise ValueError(f"sample rate must be positive, got {sr}")

        # One batch of one utterance: ``(1, samples)``.
        waveform = torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0)
        if sr != MODEL_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, MODEL_SAMPLE_RATE)
        with torch.inference_mode():
            # ``extract_features`` returns the outputs of the transformer layers
            # only (no feature-projection output), hence ``MODEL_LAYER - 1``.
            states, _ = self._model.extract_features(waveform, num_layers=MODEL_LAYER)
            frames = states[MODEL_LAYER - 1].squeeze(0).cpu().numpy()
            pooled = pool_frames(frames)
            vector = torch.nn.functional.normalize(torch.from_numpy(pooled), dim=0)
        return vector.cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the frozen HuBERT-base model, once per process.

    The optional dependency is imported here, not at module level, so importing
    this module works without ``voicematch[ml]``; :class:`HubertEncoder` then
    fails at construction with an actionable ``ImportError``.
    """
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "HubertEncoder needs the 'ml' extra: pip install voicematch[ml]"
        ) from exc
    return torchaudio.pipelines.HUBERT_BASE.get_model().eval()


def _digest_stream(x: np.ndarray, sr: int) -> bytes:
    """Return the concatenated SHA-256 digests of one audio array's blocks.

    A block is one pipeline frame of samples (``FRAME_LENGTH`` × 4 bytes of
    float32).  The sample rate is hashed into every block so that the same
    samples stamped with a different rate are a different input.  Every later
    hash consumes the whole stream, so a change anywhere in the recording
    changes every output value.
    """
    header = b"voicematch.hash-encoder.v1|" + str(int(sr)).encode("ascii")
    raw = np.ascontiguousarray(x, dtype=np.float32).tobytes()
    return b"".join(
        hashlib.sha256(header + b"|" + raw[offset : offset + _HASH_BLOCK_BYTES]).digest()
        for offset in range(0, len(raw), _HASH_BLOCK_BYTES)
    )


def _expand(seed: bytes, n_bytes: int) -> bytes:
    """Stretch ``seed`` to ``n_bytes`` deterministically with SHA-256 counter mode.

    SHA-256 is not a pseudorandom generator that needs a stored seed: the same
    ``seed`` yields the same bytes on every machine and every run, which is what
    ``HashEncoder``'s determinism rests on.
    """
    out = bytearray()
    counter = 0
    while len(out) < n_bytes:
        out += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:n_bytes])
