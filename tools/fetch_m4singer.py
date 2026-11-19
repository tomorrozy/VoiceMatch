"""Fetch a small, fixed subset of M4Singer for pipeline validation.

M4Singer is a CC BY-NC-SA 4.0 corpus of clean Mandarin solo vocals from 20
professional singers.  This script is for RESEARCH / pipeline validation only:
it selects ten singers (five female voice folders: Soprano/Alto, five male:
Tenor/Bass) and assembles one ~60-second material WAV per singer from their
short clips, plus the manifest consumed by ``voicematch.cli ingest``.

Why concatenate?  M4Singer clips are 3-10 s each, but the pipeline admits a
window only when >=8 s of speech falls inside one 10 s window.  A single clip
therefore usually yields no window.  Joining a singer's clips (with a short
silence between) produces material shaped like the design assumes: one longer
take from which the shared window rule cuts five segments.  This is still real
vocals from one singer; it is NOT a benchmark and supports no accuracy claim.

The selection is deterministic (sorted file names, first N per folder) so two
runs produce the same catalog version.  Requires network access to the Hugging
Face mirror; set HF_ENDPOINT if the default is unreachable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "AKRTR/m4singer"
# Voice folders encode the singer identity; Soprano/Alto are female and
# Tenor/Bass are male in M4Singer's catalogue.
FEMALE = ["Soprano-1", "Soprano-2", "Soprano-3", "Alto-1", "Alto-2"]
MALE = ["Tenor-1", "Tenor-2", "Tenor-3", "Bass-1", "Bass-2"]
# Target material length in seconds; the pipeline needs 60 s to cut five
# non-overlapping 10 s windows.  Clips are 3-10 s, so join enough of them.
TARGET_SEC = 62.0
GAP_SEC = 0.15  # short silence between clips so joins are not clicks
MAX_CLIPS = 24
ROOT = Path(__file__).resolve().parents[1] / "material" / "m4singer"


def main() -> None:
    api = HfApi()
    files = [s.rfilename for s in api.dataset_info(REPO).siblings]
    root = ROOT
    manifest_singers = []
    total = 0

    for folder in FEMALE + MALE:
        clips = sorted(n for n in files if n.startswith(folder + "/") and n.endswith(".wav"))
        if not clips:
            raise SystemExit(f"{folder} has no clips")
        gender = "female" if folder in FEMALE else "male"
        singer_id = folder.lower().replace("-", "_")
        singer_dir = root / "audio" / singer_id
        singer_dir.mkdir(parents=True, exist_ok=True)

        pieces: list[np.ndarray] = []
        used = 0
        for clip in clips[:MAX_CLIPS]:
            cached = Path(
                hf_hub_download(REPO, clip, repo_type="dataset", local_dir=root / "hf_cache")
            )
            total += cached.stat().st_size
            audio, sr = sf.read(cached, dtype="float32", always_2d=True)
            pieces.append(audio.mean(axis=1))
            used += 1
            if sum(len(p) for p in pieces) / sr >= TARGET_SEC:
                break
        gap = np.zeros(int(GAP_SEC * sr), dtype=np.float32)
        joined = np.concatenate([piece for p in pieces for piece in (p, gap)])
        out_wav = singer_dir / "material.wav"
        sf.write(out_wav, joined, sr)
        manifest_singers.append(
            {
                "singer_id": singer_id,
                "name": f"M4Singer {folder}",
                "gender": gender,
                "style_tag": folder.split("-")[0].lower(),
                "dir": f"audio/{singer_id}",
            }
        )
        print(f"{folder}: {used} clips -> {len(joined)/sr:.1f}s", flush=True)

    manifest = {
        "source": f"M4Singer ({REPO}), CC BY-NC-SA 4.0",
        "license": "CC-BY-NC-SA-4.0",
        "intended_use": "research",
        "rights_verified": False,
        "consent_ref": None,
        "singers": manifest_singers,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest: {manifest_path}")
    print(f"Downloaded {total / 1e6:.1f} MB across {len(manifest_singers)} singers")
    print("\nNext:")
    print(f"  python -m voicematch.cli ingest --manifest {manifest_path} --out {root / 'catalog'}")
    print(f"  python -m voicematch.cli build --catalog {root / 'catalog' / 'catalog.json'} --out {root / 'lib'} --encoder hubert")


if __name__ == "__main__":
    sys.exit(main())
