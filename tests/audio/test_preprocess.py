import numpy as np
import pyloudnorm as pyln
import pytest

from voicematch.audio.preprocess import (PREPROCESS_LIBRARY, PREPROCESS_QUERY,
                                         PreprocessConfig, pipeline_signature, preprocess)
from voicematch.audio.quality import assess_quality
from voicematch.config import LUFS_TARGET, SAMPLE_RATE


def test_query_and_library_pipelines_are_identical():
    assert PREPROCESS_QUERY == PREPROCESS_LIBRARY
    assert pipeline_signature(PREPROCESS_QUERY) == pipeline_signature(PREPROCESS_LIBRARY)


def test_loudness_is_normalized_to_target():
    import pyloudnorm as pyln

    from voicematch.config import LUFS_TARGET

    x = np.random.default_rng(0).normal(0, 0.01, 24000 * 5).astype(np.float32)
    out = preprocess(x, PREPROCESS_QUERY)
    assert abs(pyln.Meter(24000).integrated_loudness(out) - LUFS_TARGET) < 0.5


def test_dereverb_not_implemented():
    with pytest.raises(NotImplementedError):
        preprocess(np.zeros(24000, np.float32),
                   PreprocessConfig(separate_vocals=False, dereverb=True, lufs_target=-23.0))


# --- C2: the loudness gain is limited, so no sample exceeds full scale -------


def test_quiet_peaky_audio_is_limited_instead_of_exceeding_full_scale():
    # A low-frequency rumble: the loudness weighting de-emphasises its energy,
    # so its integrated loudness is low while its waveform still peaks at 0.9.
    # Normalising to the LUFS target alone would multiply the signal by more
    # than 1.8 -- design 5.1 step 6 ("限幅", dropped from the pipeline) is what
    # keeps every sample inside [-1, 1].
    times = np.arange(SAMPLE_RATE * 10) / SAMPLE_RATE
    x = (0.9 * np.sin(2 * np.pi * 10.0 * times)).astype(np.float32)

    loudness = pyln.Meter(SAMPLE_RATE).integrated_loudness(x.astype(np.float64))
    assert 0.9 * 10.0 ** ((LUFS_TARGET - loudness) / 20.0) > 1.0

    out = preprocess(x, PREPROCESS_QUERY)

    assert float(np.max(np.abs(out))) <= 1.0
    # The gate measures the decoded signal, so the limiter's work is not
    # reported as clipping of the source (C2).
    assert "clipped" not in assess_quality(x, SAMPLE_RATE).reasons
