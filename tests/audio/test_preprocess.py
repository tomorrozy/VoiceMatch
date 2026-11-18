import numpy as np
import pytest

from voicematch.audio.preprocess import (PREPROCESS_LIBRARY, PREPROCESS_QUERY,
                                         PreprocessConfig, pipeline_signature, preprocess)


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
