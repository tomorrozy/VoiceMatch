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

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

REPO = "AKRTR/m4singer"
# Voice folders encode the singer identity; Soprano/Alto are female and
# Tenor/Bass are male in M4Singer's catalogue.
FEMALE_10 = ["Soprano-1", "Soprano-2", "Soprano-3", "Alto-1", "Alto-2"]
MALE_10 = ["Tenor-1", "Tenor-2", "Tenor-3", "Bass-1", "Bass-2"]
# All 20 voice folders in M4Singer's catalogue (10 female, 10 male).  Used by
# --all to scale the library while keeping genders balanced.
FEMALE_ALL = ["Soprano-1", "Soprano-2", "Soprano-3"] + [f"Alto-{i}" for i in range(1, 8)]
MALE_ALL = [f"Tenor-{i}" for i in range(1, 8)] + ["Bass-1", "Bass-2", "Bass-3"]
# Target material length in seconds; the pipeline needs 60 s to cut five
# non-overlapping 10 s windows.  Clips are 3-10 s, so join enough of them.
DEFAULT_TARGET_SEC = 62.0
GAP_SEC = 0.15  # short silence between clips so joins are not clicks
MAX_CLIPS = 24
ROOT = Path(__file__).resolve().parents[1] / "material" / "m4singer"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="use all 20 singers instead of the default 10")
    parser.add_argument("--out", type=Path, default=ROOT, help="material output directory")
    parser.add_argument(
        "--target-sec",
        type=float,
        default=DEFAULT_TARGET_SEC,
        help="target material seconds per singer (62 s yields ~5 windows)",
    )
    parser.add_argument(
        "--takes-per-singer",
        type=int,
        default=1,
        help=(
            "write N separate takes (one per song) per singer instead of one "
            "concatenated material; >1 gives cross-take calibration positives"
        ),
    )
    parser.add_argument(
        "--take-sec",
        type=float,
        default=42.0,
        help="target seconds per take when --takes-per-singer > 1",
    )
    args = parser.parse_args(argv)
    if args.takes_per_singer < 1:
        parser.error("--takes-per-singer must be >= 1")

    female = FEMALE_ALL if args.all else FEMALE_10
    male = MALE_ALL if args.all else MALE_10
    target_sec = args.target_sec

    api = HfApi()
    files = [s.rfilename for s in api.dataset_info(REPO).siblings]
    root = args.out
    manifest_singers = []
    used_by_singer: dict[str, list[str]] = {}
    total = 0

    for folder in female + male:
        clips = sorted(n for n in files if n.startswith(folder + "/") and n.endswith(".wav"))
        if not clips:
            raise SystemExit(f"{folder} has no clips")
        gender = "female" if folder in female else "male"
        singer_id = folder.lower().replace("-", "_")
        singer_dir = root / "audio" / singer_id
        singer_dir.mkdir(parents=True, exist_ok=True)

        consumed: list[str] = []
        if args.takes_per_singer == 1:
            # One concatenated take (historical behaviour): join clips of the
            # first song(s) until target_sec is reached.
            per_take = [(None, clips, target_sec)]
        else:
            # One take per DISTINCT SONG, so cross-take pairs are genuine
            # (same singer, different song).  Group by song name, keep the
            # first N songs in sorted order, one concatenated take each.
            songs: dict[str, list[str]] = {}
            for clip in clips:
                songs.setdefault(re.sub(r"\d+\.wav$", "", Path(clip).name), []).append(clip)
            chosen = sorted(songs)[: args.takes_per_singer]
            if len(chosen) < args.takes_per_singer:
                raise SystemExit(
                    f"{folder} has only {len(chosen)} distinct songs, "
                    f"need {args.takes_per_singer}"
                )
            per_take = [(song, songs[song], args.take_sec) for song in chosen]

        for take_index, (song, take_clips, take_sec) in enumerate(per_take):
            pieces: list[np.ndarray] = []
            sr = 48000  # M4Singer clips are 48 kHz; refreshed on every read.
            for clip in take_clips[:MAX_CLIPS]:
                cached = Path(
                    hf_hub_download(REPO, clip, repo_type="dataset", local_dir=root / "hf_cache")
                )
                total += cached.stat().st_size
                audio, sr = sf.read(cached, dtype="float32", always_2d=True)
                pieces.append(audio.mean(axis=1))
                consumed.append(Path(clip).name)
                if sum(len(p) for p in pieces) / sr >= take_sec:
                    break
            gap = np.zeros(int(GAP_SEC * sr), dtype=np.float32)
            joined = np.concatenate([piece for p in pieces for piece in (p, gap)])
            suffix = "material" if args.takes_per_singer == 1 else f"take{take_index}"
            sf.write(singer_dir / f"{suffix}.wav", joined, sr)
            if args.takes_per_singer > 1:
                print(f"  {folder} take{take_index} ({song}): {len(joined)/sr:.1f}s", flush=True)
        used_by_singer[singer_id] = consumed
        manifest_singers.append(
            {
                "singer_id": singer_id,
                "name": f"M4Singer {folder}",
                "gender": gender,
                "style_tag": folder.split("-")[0].lower(),
                "dir": f"audio/{singer_id}",
            }
        )
        print(
            f"{folder}: {len(consumed)} clips -> "
            f"{args.takes_per_singer} take(s)",
            flush=True,
        )

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
    # Record exactly which clips built the material, so the holdout evaluator
    # can pick genuinely unseen recordings of the same singers.
    (root / "used_clips.json").write_text(
        json.dumps(used_by_singer, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {manifest_path}")
    print(f"Downloaded {total / 1e6:.1f} MB across {len(manifest_singers)} singers")
    print("\nNext:")
    print(f"  python -m voicematch.cli ingest --manifest {manifest_path} --out {root / 'catalog'}")
    print(f"  python -m voicematch.cli build --catalog {root / 'catalog' / 'catalog.json'} --out {root / 'lib'} --encoder hubert")


if __name__ == "__main__":
    sys.exit(main())
