"""Explainable singing-style descriptors: vibrato, slurs, attacks, breath.

These are the four "唱腔" descriptors.  They answer *how* a phrase is sung rather
than what the voice sounds like: whether the pitch oscillates periodically, how
the pitch moves between notes, how abruptly notes start, and how much
high-frequency noise rides on the tone.

The vibrato estimator follows the plan's algorithm literally (band-pass the
cent track at 3-8 Hz, peak-to-peak extent from the standard deviation, FFT peak
for the rate) because the extent's unit is an interface: a sinusoid of amplitude
``A`` cents has ``std = A / sqrt(2)``, so ``2 * sqrt(2) * std`` is the
peak-to-peak extent ``2A``, which is how vibrato extent is conventionally
reported in the singing-voice literature.

Like the pitch and timbre modules, this layer never throws for content it cannot
analyse.  A window shorter than one analysis frame has no frame grid and gets
``nan`` for every descriptor; a descriptor that cannot be measured on an
otherwise usable window (no voiced frame for vibrato or slurs, no envelope rise
for onset hardness, no low-band energy for breathiness) is ``nan`` on its own.
Nothing warns.  The quality gate owns the decision to refuse such audio.
"""

import librosa
import numpy as np
from scipy import ndimage, signal

from voicematch.audio.windows import F0_MAX_HZ, F0_MIN_HZ
from voicematch.config import FRAME_LENGTH, HOP_LENGTH

#: The descriptor names.  Part of the feature-vector interface: later stages and
#: the explanation report look them up by name.
_STYLE_KEYS = (
    "vibrato_rate_hz",
    "vibrato_extent_cents",
    "slur_ratio",
    "onset_hardness",
    "breathiness",
)

# Vocal vibrato sits around 4-7 Hz; the 3-8 Hz band keeps slow pitch drift
# (under 3 Hz) and note-to-note movement (well under 3 Hz) out of the estimate.
_VIBRATO_BAND_HZ = (3.0, 8.0)
_VIBRATO_FILTER_ORDER = 4

#: Below this peak-to-peak extent the "vibrato" is just tracker jitter, and the
#: FFT peak of the noise would be meaningless, so the rate is reported as 0.0.
_VIBRATO_EXTENT_FLOOR_CENTS = 10.0

#: A frame-to-frame f0 move counts as a slur/portamento when the implied rate is
#: in this window, in semitones per second.  The unit matters: a per-frame window
#: silently depends on ``HOP_LENGTH`` (0.5-4 st/frame at 93.75 frames/s is
#: 47-375 st/s, i.e. a glissando rather than legato), whereas singers' connected
#: phrasing sits around 2-12 st/s.
_SLUR_MIN_RATE_ST_PER_SEC = 2.0
_SLUR_MAX_RATE_ST_PER_SEC = 12.0

#: Denoising window for the cent track, in frames (~53 ms).  pYIN quantizes f0
#: into 0.1-semitone (10-cent) bins, so a perfectly steady tone can flip between
#: neighbouring bins and produce fake adjacent-frame steps of up to ~9 st/s --
#: right inside the slur window.  The window is far shorter than a sung
#: portamento (100-300 ms), so real connected movement survives.
_SLUR_SMOOTHING_FRAMES = 5

#: Bands of the breathiness ratio: 2-8 kHz energy over 0-2 kHz energy.
_LOW_BAND_HZ = 2000.0
_HIGH_BAND_HZ = 8000.0

#: A frame-to-frame envelope rise must be at least this fraction of the window's
#: peak energy to count as an onset edge instead of frame-grid ripple.
_ONSET_RISE_FRACTION = 0.1


def style_features(x: np.ndarray, sr: int) -> dict[str, float]:
    """Return the singing-style descriptors of one audio window.

    - ``vibrato_rate_hz`` — frequency of the periodic f0 oscillation, from the
      FFT peak of the 3-8 Hz band-passed cent track; ``0.0`` when the extent is
      below 10 cents.
    - ``vibrato_extent_cents`` — peak-to-peak vibrato depth, measured as
      ``2 * sqrt(2) * std`` of the band-passed cent track (for a sinusoidal
      vibrato of ±A cents this is 2A).
    - ``slur_ratio`` — share of adjacent voiced-frame pairs whose f0 moves at
      2-12 semitones **per second**, i.e. how often the voice slides in
      connected movement rather than holding or jumping between notes.
    - ``onset_hardness`` — mean steepness of the energy envelope's rising edges,
      in peak energy per frame.  The envelope is the per-frame peak of ``x**2``;
      a windowed RMS would ripple with the phase of a steady tone, and the
      descriptor must measure the attack, not that ripple.
    - ``breathiness`` — 2-8 kHz energy divided by 0-2 kHz energy; air noise adds
      high-frequency energy without a low-frequency fundamental to match.

    A non-mono array or a non-positive sample rate raises ``ValueError`` instead:
    that is a caller bug, not content.  Audio shorter than one analysis frame,
    or with no usable frame content, returns ``nan`` for every key.
    """
    y = np.asarray(x, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"expected mono 1-D audio, got shape {y.shape}")
    if sr <= 0:
        raise ValueError(f"sample rate must be positive, got {sr}")
    if y.size < FRAME_LENGTH:
        # Shorter than one frame: no frame grid, and librosa would warn that
        # n_fft exceeds the signal length.
        return {key: float("nan") for key in _STYLE_KEYS}

    voiced = _pitch_track(y, sr)
    voiced = voiced[np.isfinite(voiced)]
    vibrato_rate, vibrato_extent = _vibrato_descriptors(voiced, sr)
    return {
        "vibrato_rate_hz": vibrato_rate,
        "vibrato_extent_cents": vibrato_extent,
        "slur_ratio": _slur_ratio(voiced, sr),
        "onset_hardness": _onset_hardness(y),
        "breathiness": _breathiness(y, sr),
    }


def _pitch_track(y: np.ndarray, sr: int) -> np.ndarray:
    """Return f0 per frame (NaN where unvoiced) on the pipeline frame grid.

    The range and frame grid are the ones shared through ``audio.windows``, so a
    frame is voiced here exactly when it is voiced for window ranking.
    """
    f0, _voiced_flag, _probability = librosa.pyin(
        y,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    return f0


def _vibrato_descriptors(voiced: np.ndarray, sr: int) -> tuple[float, float]:
    """Return ``(rate_hz, extent_cents)`` from the sequence of voiced f0 values.

    ``voiced`` has the unvoiced frames removed, exactly as the plan prescribes.
    A voicing gap therefore compresses the time base, so the rate is an
    approximation on audio with many unvoiced stretches; the quality gate and
    the voice-dominant windows keep that from mattering in practice.
    """
    if voiced.size < 2:
        return float("nan"), float("nan")

    # Cent track relative to the window's own median: the descriptor describes
    # the shape of the oscillation, not the absolute pitch, and using the median
    # keeps one octave error of a frame-level tracker from offsetting the track.
    cents = 1200.0 * np.log2(voiced / float(np.median(voiced)))
    filtered = _vibrato_band(cents - float(np.mean(cents)), sr)
    if filtered is None:
        return float("nan"), float("nan")

    extent = float(2.0 * np.sqrt(2.0) * np.std(filtered))
    if extent < _VIBRATO_EXTENT_FLOOR_CENTS:
        return 0.0, extent

    spectrum = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(filtered.size, d=HOP_LENGTH / sr)
    return float(frequencies[int(np.argmax(spectrum))]), extent


def _vibrato_band(cents: np.ndarray, sr: int) -> np.ndarray | None:
    """Band-pass a cent track at 3-8 Hz; ``None`` when it cannot be filtered.

    The track is sampled at one value per analysis hop, so the filter's sample
    rate is ``sr / HOP_LENGTH``.  ``sosfiltfilt`` is zero-phase, so the FFT peak
    sits at the true vibrato rate instead of a filter-delayed one.  The pad
    length is clamped so a very short track still filters instead of raising.
    """
    if cents.size < 2:
        return None
    hop_seconds = HOP_LENGTH / sr
    frame_rate = 1.0 / hop_seconds
    if frame_rate / 2.0 <= _VIBRATO_BAND_HZ[1]:
        # The frame grid cannot even represent the 3-8 Hz band.
        return None
    sos = signal.butter(
        _VIBRATO_FILTER_ORDER,
        _VIBRATO_BAND_HZ,
        btype="band",
        fs=frame_rate,
        output="sos",
    )
    # scipy's default pad length, clamped to what this track can supply.
    default_padlen = 3 * (2 * sos.shape[0] + 1)
    return signal.sosfiltfilt(
        sos, cents, padlen=min(default_padlen, cents.size - 1)
    )


def _slur_ratio(voiced: np.ndarray, sr: int) -> float:
    """Return the share of adjacent voiced frames moving 2-12 semitones/s.

    The rate is semitones per second, not per frame: dividing by the frame
    interval makes the descriptor independent of ``HOP_LENGTH`` and puts the
    window where connected singing actually lives.  Note changes faster than
    12 st/s are jumps, not slurs, and are excluded.

    The cent track is denoised first (see ``_SLUR_SMOOTHING_FRAMES``): pYIN's
    10-cent bin quantization otherwise turns a steady tone's bin flips into
    apparent 2-9 st/s movement, which would make the descriptor measure the
    tracker instead of the singer.
    """
    if voiced.size < 2:
        return float("nan")
    cents = 1200.0 * np.log2(voiced)
    smoothed = _smoothed_cents(cents)
    rates = np.abs(np.diff(smoothed)) / 100.0 * (sr / HOP_LENGTH)
    in_window = (rates >= _SLUR_MIN_RATE_ST_PER_SEC) & (
        rates <= _SLUR_MAX_RATE_ST_PER_SEC
    )
    return float(np.mean(in_window))


def _smoothed_cents(cents: np.ndarray) -> np.ndarray:
    """Denoise a cent track: median-filter bin flips, then average the staircase.

    The median filter drops isolated jumps between neighbouring pYIN bins while
    preserving a monotone ramp; the moving average then turns the remaining
    quantization staircase into a smooth local slope.  Tracks shorter than the
    window are left as they are.
    """
    if cents.size < _SLUR_SMOOTHING_FRAMES:
        return cents
    denoised = ndimage.median_filter(
        cents, size=_SLUR_SMOOTHING_FRAMES, mode="nearest"
    )
    return ndimage.uniform_filter1d(
        denoised, size=_SLUR_SMOOTHING_FRAMES, mode="nearest"
    )


def _onset_hardness(y: np.ndarray) -> float:
    """Return the mean steepness of the energy envelope's rising edges.

    The envelope is the per-frame peak of ``x**2``.  A windowed RMS of a steady
    tone ripples with the phase of the tone inside the frame, and treating that
    ripple as a rising edge would report a hard onset for a sustained note; the
    per-frame peak does not ripple because every frame of a sustained tone
    contains its crest.  Rising steps smaller than ``_ONSET_RISE_FRACTION`` of
    the window's peak energy are ripple, not attacks, and are ignored.
    """
    frames = librosa.util.frame(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    envelope = np.max(frames**2, axis=0)
    if envelope.size < 2 or not np.any(envelope):
        return float("nan")

    rises = np.diff(envelope)
    material = rises[rises > _ONSET_RISE_FRACTION * float(np.max(envelope))]
    if material.size == 0:
        return 0.0
    return float(np.mean(material))


def _breathiness(y: np.ndarray, sr: int) -> float:
    """Return 2-8 kHz energy divided by 0-2 kHz energy on the frame grid."""
    spectrum = (
        np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)) ** 2
    )
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    low_band = float(spectrum[frequencies <= _LOW_BAND_HZ].sum())
    high_band = float(
        spectrum[
            (frequencies > _LOW_BAND_HZ)
            & (frequencies <= min(_HIGH_BAND_HZ, sr / 2.0))
        ].sum()
    )
    if low_band <= 0.0:
        return float("nan")
    return high_band / low_band
