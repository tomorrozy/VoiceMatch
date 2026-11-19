from voicematch import config as c


def test_pipeline_constants_are_pinned():
    assert c.SAMPLE_RATE == 24000
    assert c.LUFS_TARGET == -23.0
    assert c.WINDOW_SEC == 10.0
    assert c.MIN_SPEECH_SEC == 8.0
    assert c.MAX_INPUT_SEC == 60.0
    assert c.EMBED_DIM == 256
    assert c.PIPELINE_VERSION.startswith("pipeline-v")


def test_pitch_and_embedding_constants_are_pinned():
    assert c.FRAME_LENGTH == 1024
    assert c.HOP_LENGTH == 256
    assert c.EMBED_DIM == 256
    assert c.PIPELINE_VERSION == "pipeline-v2"


def test_a_signal_altering_change_bumps_the_pipeline_version():
    # PIPELINE_VERSION is part of the library version hash (G4): a change that
    # alters the post-preprocess signal without moving it produces two different
    # libraries under one version.  ``preprocess`` gained its peak limiter and
    # the constant stayed "pipeline-v1"; this pins the bump so the next such
    # change has to notice it too.
    assert c.PIPELINE_VERSION == "pipeline-v2"
