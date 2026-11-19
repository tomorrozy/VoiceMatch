"""Tests for the reproducibility identifiers (I5).

Design G4 promises that the same audio, library and model versions give the
same output.  That promise is only checkable on another machine if the
environment is pinned and the installed versions are recorded:

- the runtime dependencies in ``pyproject.toml`` are exact ``==`` pins, so a
  fresh install resolves the same versions this branch was verified with;
- the built library's ``manifest.json`` records the resolved versions of the
  numerical stack (``deps``), so a rebuild that silently upgrades numpy or
  librosa is detectable instead of producing different vectors under the same
  library version.

The digest-of-vectors half of I5 was ruled out of scope (ledger R26): this is
the cheap, detectable half.
"""

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

#: The project root (``pyproject.toml`` lives next to ``src`` and ``tests``).
_ROOT = Path(__file__).resolve().parents[1]

#: The distributions the pipeline's numbers depend on; each must be pinned.
_RUNTIME_DEPENDENCIES = (
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "pyloudnorm",
    "praat-parselmouth",
    "scikit-learn",
)


def _pyproject() -> dict:
    """Parse this repository's ``pyproject.toml``."""
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_runtime_dependencies_are_pinned_to_exact_versions():
    requirements = _pyproject()["project"]["dependencies"]

    assert len(requirements) == len(_RUNTIME_DEPENDENCIES)
    for requirement in requirements:
        name, separator, version = requirement.partition("==")
        assert separator == "==", f"{requirement!r} is not an exact == pin"
        assert name.strip() in _RUNTIME_DEPENDENCIES
        assert version.strip(), f"{requirement!r} has no pinned version"


def test_the_pins_match_the_installed_versions():
    # A pin that disagrees with the environment this branch was verified in
    # would record a different numerical stack than the one the tests ran on.
    requirements = _pyproject()["project"]["dependencies"]

    for requirement in requirements:
        name, _separator, version = requirement.partition("==")
        assert metadata.version(name.strip()) == version.strip(), (
            f"{name.strip()} is installed at {metadata.version(name.strip())} "
            f"but pyproject.toml pins {version.strip()}"
        )


#: The optional ``ml`` extra's distributions.  The suite must run in a plain
#: install (``tests/embed/test_encoder.py`` guards the real HuBERT test for the
#: same reason), so the pin check below is skipped unless both are installed.
_ML_EXTRA_DISTRIBUTIONS = ("torch", "torchaudio")


def _distribution_installed(name: str) -> bool:
    """Return whether ``name``'s installed metadata can be read."""
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


requires_ml_extra = pytest.mark.skipif(
    not all(_distribution_installed(name) for name in _ML_EXTRA_DISTRIBUTIONS),
    reason="the optional 'ml' extra (torch and torchaudio) is not installed",
)


@requires_ml_extra
def test_the_ml_extra_pins_the_verified_torch_audio_pair():
    # The production encoder is only reproducible if the backbone and its audio
    # frontend are pinned together; this pair is the one the HuBERT test ran on.
    ml_extra = _pyproject()["project"]["optional-dependencies"]["ml"]

    pinned = {
        requirement.partition("==")[0].strip()
        for requirement in ml_extra
        if "==" in requirement
    }
    assert {"torch", "torchaudio"} <= pinned
    for requirement in ml_extra:
        name, separator, version = requirement.partition("==")
        if separator and name.strip() in {"torch", "torchaudio"}:
            assert metadata.version(name.strip()) == version.strip()


def test_an_unresolvable_version_is_recorded_as_unknown():
    # The manifest must always be writable: a missing distribution is recorded
    # as "unknown" rather than failing the save.
    from voicematch.library import store

    assert store._dependency_versions()  # the real environment resolves them
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            store,
            "_DEPENDENCY_PACKAGES",
            ("numpy", "surely-not-installed-distribution"),
        )
        versions = store._dependency_versions()
    assert versions["numpy"] == metadata.version("numpy")
    assert versions["surely-not-installed-distribution"] == "unknown"
