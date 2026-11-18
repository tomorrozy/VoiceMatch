"""Tests for the frozen vocal encoders.

``HashEncoder`` carries the behavioural assertions.  It is the deterministic
stand-in that lets every later task's tests run without the optional ``ml``
extra: torch/torchaudio are imported when ``HubertEncoder`` is constructed,
never when this module is imported.

``HubertEncoder`` is the production encoder, so it is no longer left
unverified: torchaudio is installed and ``test_hubert_encoder_embeds_real_audio``
runs the real frozen HuBERT-base forward pass.  The optional-dependency guard
keys on ``torchaudio`` (I4: keying on ``torch`` skipped the missing-extra test
while the frontend the encoder actually needs was absent), and the actionable
``ImportError`` is additionally pinned by a test that blocks the import, so the
branch is covered whether or not the extra is installed here.

The brief's two ``HashEncoder`` tests are kept verbatim.  The remaining tests
pin the contract the rest of the pipeline types and persists against: a
unit-norm float32 ``(EMBED_DIM,)`` vector, byte-level determinism (G4),
near-orthogonality for distinct inputs, ``MODEL_ID`` as the recorded backbone
identifier, and the optional-dependency isolation above.
"""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest

from voicematch.config import EMBED_DIM, SAMPLE_RATE
from voicematch.embed import encoder as encoder_module
from voicematch.embed.encoder import (
    MODEL_ID,
    MODEL_LAYER,
    MODEL_SAMPLE_RATE,
    HashEncoder,
    HubertEncoder,
    VocalEncoder,
)
from voicematch.embed.pooling import pool_frames

TORCHAUDIO_INSTALLED = importlib.util.find_spec("torchaudio") is not None


def _audio(seed, seconds=3.0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, int(SAMPLE_RATE * seconds)).astype(np.float32)


def test_embedding_is_unit_norm_and_deterministic():
    enc = HashEncoder()
    x = np.random.default_rng(5).normal(0, .1, 24000 * 3).astype(np.float32)
    a, b = enc.encode(x, 24000), enc.encode(x, 24000)
    assert a.shape == (EMBED_DIM,) and abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert np.allclose(a, b)


def test_different_audio_gives_different_vectors():
    enc = HashEncoder()
    x = np.random.default_rng(6).normal(0, .1, 24000 * 3).astype(np.float32)
    y = np.random.default_rng(7).normal(0, .1, 24000 * 3).astype(np.float32)
    assert np.dot(enc.encode(x, 24000), enc.encode(y, 24000)) < 0.95


def test_determinism_does_not_depend_on_encoder_instance_state():
    x = _audio(8)

    first = HashEncoder().encode(x, SAMPLE_RATE)
    second = HashEncoder().encode(x, SAMPLE_RATE)

    assert np.array_equal(first, second)


@pytest.mark.parametrize("length", [1, 100, SAMPLE_RATE, SAMPLE_RATE * 3])
def test_every_length_yields_a_unit_norm_float32_vector(length):
    rng = np.random.default_rng(length)
    x = rng.normal(0.0, 0.1, length).astype(np.float32)

    vector = HashEncoder().encode(x, SAMPLE_RATE)

    assert vector.shape == (EMBED_DIM,)
    assert vector.dtype == np.float32
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    assert np.all(np.isfinite(vector))


def test_silence_and_constant_audio_still_give_unit_norm_vectors():
    for x in (np.zeros(SAMPLE_RATE, np.float32), np.ones(SAMPLE_RATE, np.float32)):
        vector = HashEncoder().encode(x, SAMPLE_RATE)

        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_many_distinct_inputs_are_near_orthogonal():
    enc = HashEncoder()
    vectors = [enc.encode(_audio(seed), SAMPLE_RATE) for seed in range(20, 28)]
    dots = [
        abs(float(np.dot(a, b)))
        for index, a in enumerate(vectors)
        for b in vectors[index + 1:]
    ]

    assert max(dots) < 0.5


def test_the_sample_rate_is_part_of_the_identity():
    # The same samples stamped with a different rate are a different input:
    # ``encode`` is a function of (samples, sample rate), not of bytes alone.
    x = _audio(9)

    a = HashEncoder().encode(x, SAMPLE_RATE)
    b = HashEncoder().encode(x, MODEL_SAMPLE_RATE)

    assert abs(float(np.dot(a, b))) < 0.5


def test_encode_does_not_mutate_its_input():
    x = _audio(10)
    before = x.copy()

    HashEncoder().encode(x, SAMPLE_RATE)

    assert np.array_equal(x, before)


def test_caller_bugs_raise_value_error():
    enc = HashEncoder()
    with pytest.raises(ValueError):
        enc.encode(np.zeros((2, SAMPLE_RATE), np.float32), SAMPLE_RATE)
    with pytest.raises(ValueError):
        enc.encode(np.zeros(SAMPLE_RATE, np.float32), 0)


def test_hash_encoder_implements_the_vocal_encoder_protocol():
    assert isinstance(HashEncoder(), VocalEncoder)


def test_package_exports_the_public_interface():
    import voicematch.embed as embed

    assert embed.HashEncoder is HashEncoder
    assert embed.HubertEncoder is HubertEncoder
    assert embed.VocalEncoder is VocalEncoder
    assert embed.MODEL_ID == MODEL_ID
    assert embed.pool_frames is pool_frames


def test_frozen_backbone_identity_is_recorded():
    # These three values define what an embedding means (G4): the backbone, the
    # rate it expects and the layer whose output is pooled.  They are part of
    # the stored record and cannot drift without a new pipeline version.
    assert MODEL_ID == "HUBERT_BASE"
    assert MODEL_SAMPLE_RATE == 16000
    assert MODEL_LAYER == 9


def test_importing_the_encoders_does_not_import_torch():
    # Importing this module must work without the ``ml`` extra.  Run in a fresh
    # interpreter so a torch import from any other test cannot mask a violation.
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from voicematch.embed import HashEncoder\n"
        "HashEncoder().encode(np.zeros(24000, dtype=np.float32), 24000)\n"
        "print(int('torch' in sys.modules), int('torchaudio' in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 0"


@pytest.mark.skipif(
    TORCHAUDIO_INSTALLED,
    reason="torchaudio is installed, so the missing-extra path is unavailable",
)
def test_hubert_encoder_requires_the_ml_extra():
    with pytest.raises(ImportError, match=r"voicematch\[ml\]"):
        HubertEncoder()


def test_missing_torchaudio_raises_an_actionable_import_error(monkeypatch):
    # The encoder needs the *audio frontend*, so the actionable failure is
    # about torchaudio (I4), and blocking the import exercises the branch that
    # the environment-dependent skip above cannot reach once it is installed.
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    encoder_module._load_model.cache_clear()
    try:
        with pytest.raises(ImportError, match=r"voicematch\[ml\]"):
            HubertEncoder()
    finally:
        encoder_module._load_model.cache_clear()


def test_hubert_encoder_embeds_real_audio():
    pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    enc = HubertEncoder()

    a = enc.encode(x, SAMPLE_RATE)
    b = enc.encode(x, SAMPLE_RATE)

    assert a.shape == (EMBED_DIM,)
    assert a.dtype == np.float32
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-5
    assert np.allclose(a, b)
    assert isinstance(enc, VocalEncoder)
