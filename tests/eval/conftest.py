"""Shared fixtures for the evaluation tests: a small, genuinely calibrated stack.

The evaluation runner is the first place where every stage meets, so its
fixture is a *real* stack rather than hand-authored vectors: four singers built
from synthesized material through :class:`HashEncoder`, and a
``CalibrationModel`` fitted on that library's own segment pairs.  Nothing is
written to disk for the model.

Why four singers (R21 carry-over)
---------------------------------
``build_pairs`` only forms a negative pair between two singers of the *same*
fold, with ``n_folds = min(5, max(2, n_singers // 2))``.  Two singers give one
fold of two singers, so no fold holds two singers that could form a negative
pair, no pairs are emitted, and ``fit_calibration`` returns the canonical
cosine-only fallback (``auc = nan``, ``is_calibrated`` false).  The evaluation
test would then never exercise the calibrated ranking path.  Four singers --
two per gender -- give two folds of two, a measured out-of-fold AUC and
``is_calibrated == True``.  The female pool is ``s1``/``s2`` and the male pool
``s3``/``s4``, so a test can also check that ranking is scoped to the query's
own gender.

Why 30 s of material per singer
-------------------------------
Two non-overlapping 10 s windows are enough for ``build_pairs`` to have
same-singer positives and same-fold negatives for every singer, and this
fixture is session-scoped, so 30 s keeps the one-time build small.  (The
library-test fixture's 60 s exists for a different requirement: five windows
per material.)

The emitted tones are deliberately distinct in f0 so the calibration fit sees
real feature deltas; nothing here asserts what that fit *means*, only that it
is a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicematch.calibrate.model import CalibrationModel, fit_calibration
from voicematch.calibrate.pairs import build_pairs
from voicematch.config import SAMPLE_RATE
from voicematch.embed.encoder import HashEncoder
from voicematch.library.build import build_library
from voicematch.library.schema import LibraryIndex, Singer

#: Material written for each fixture singer; enough for two 10 s windows.
MATERIAL_SEC: float = 30.0

#: One distinct f0 per singer, so the fixture's segment features actually vary.
TONE_F0: dict[str, float] = {
    "s1": 220.0,
    "s2": 330.0,
    "s3": 140.0,
    "s4": 180.0,
}

#: Two singers per gender, so every query has a real pool to rank.
GENDER_OF: dict[str, str] = {
    "s1": "female",
    "s2": "female",
    "s3": "male",
    "s4": "male",
}

#: The f0 of a query written with the default helper arguments.
DEFAULT_QUERY_F0: float = 220.0

#: One voiced burst, then this much true digital silence (see ``syllables``).
VOICE_SEC: float = 0.5
GAP_SEC: float = 0.1


@dataclass(frozen=True)
class TinyStack:
    """The three objects ``run_eval`` consumes, built once per test session."""

    index: LibraryIndex
    calibration: CalibrationModel
    encoder: HashEncoder


def tone(
    f0: float = DEFAULT_QUERY_F0,
    secs: float = 10.0,
    *,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return ``secs`` of a pure sine at ``f0``; ``amplitude=0`` is silence."""
    times = np.arange(int(sr * secs)) / sr
    return (amplitude * np.sin(2 * np.pi * f0 * times)).astype(np.float32)


def write_tone(
    path: Path,
    f0: float = DEFAULT_QUERY_F0,
    secs: float = 10.0,
    *,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> None:
    """Write a pure tone to ``path``.

    A pure tone is useful for *rejection* tests: its frame energies are nearly
    constant, so the quality gate's noise floor is the signal itself and the
    recording is refused for low SNR (ruling R12).
    """
    sf.write(path, tone(f0, secs, amplitude=amplitude, sr=sr), sr)


def syllables(
    f0: float = DEFAULT_QUERY_F0,
    secs: float = 10.0,
    *,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Return a voiced-burst/silence pattern that the quality gate accepts.

    Each ``VOICE_SEC`` burst of a sine is followed by ``GAP_SEC`` of true
    digital silence.  The bursts are long enough that the energy-based
    ``speech_intervals`` cover more than ``MIN_SPEECH_SEC`` of the clip, while
    the silent gaps push the noise-floor percentile to zero, so the measured
    SNR is comfortably above the gate.  A pure tone (see ``tone``) is refused;
    this is the "clean recording" the accepted-path tests need.
    """
    period = VOICE_SEC + GAP_SEC
    period_samples = int(sr * period)
    burst_times = np.arange(period_samples) / sr
    burst = (amplitude * np.sin(2 * np.pi * f0 * burst_times)).astype(np.float32)
    burst[int(sr * VOICE_SEC) :] = 0.0

    out = np.zeros(int(sr * secs), dtype=np.float32)
    position = 0
    while position + period_samples <= out.size:
        out[position : position + period_samples] = burst
        position += period_samples
    return out


def write_syllables(
    path: Path,
    f0: float = DEFAULT_QUERY_F0,
    secs: float = 10.0,
    *,
    amplitude: float = 0.3,
    sr: int = SAMPLE_RATE,
) -> None:
    """Write an accepted (non-silent) query recording to ``path``."""
    sf.write(path, syllables(f0, secs, amplitude=amplitude, sr=sr), sr)


def _stack_entries(root: Path) -> list[tuple[Singer, Path]]:
    """Write two takes per fixture singer and return the build entries.

    Two takes, not one: a positive calibration pair must span two *different
    recordings* of a singer (two windows of one take teach "same session =
    same person").  With a single take per singer the fixture would emit no
    positives and fall back to the cosine-only model, so the calibrated path
    this fixture exists to exercise would never run.  Two 15 s takes keep the
    per-singer total at 30 s while giving each singer two recordings.
    """
    entries: list[tuple[Singer, Path]] = []
    for singer_id in sorted(TONE_F0):
        singer = Singer(singer_id, f"Singer {singer_id}", GENDER_OF[singer_id], "pop")
        for take, amplitude in (("a", 0.30), ("b", 0.32)):
            material = root / f"{singer_id}-{take}.wav"
            # Same f0 across takes so the singer is still one identity, but a
            # slightly different amplitude makes the two files distinct bytes,
            # hence distinct material ids -- two recordings of one voice, not
            # one recording written twice (identical bytes would hash equal and
            # collapse into one material, yielding no cross-take positive).
            write_tone(
                material,
                f0=TONE_F0[singer_id],
                secs=MATERIAL_SEC / 2,
                amplitude=amplitude,
            )
            entries.append((singer, material))
    return entries


@pytest.fixture(scope="session")
def tiny_stack(tmp_path_factory) -> TinyStack:
    """A four-singer library plus the calibration model fitted on its pairs."""
    root = tmp_path_factory.mktemp("voicematch-eval-material")
    encoder = HashEncoder()
    index = build_library(_stack_entries(root), encoder)
    X, y, groups = build_pairs(index, index.embeddings)
    calibration = fit_calibration(X, y, groups)
    return TinyStack(index=index, calibration=calibration, encoder=encoder)


@pytest.fixture
def write_wav():
    """Factory writing a pure tone; ``amplitude=0`` writes digital silence."""
    return write_tone


@pytest.fixture
def write_syllable_query():
    """Factory writing a recording the quality gate accepts."""
    return write_syllables
