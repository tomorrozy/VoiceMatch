"""Batch hold-out evaluation: top1/top3 on unseen M4Singer material.

This script builds a hold-out query set from M4Singer clips that were NOT used
to build the library (a different song per singer), then runs the project's
own :func:`voicematch.eval.run.run_eval` to report top1/top3 with all its
denominators.  It is a **sanity check that the pipeline works on real, unseen
vocal material** -- NOT a product accuracy result:

- the "label" is the folder the clip came from (dataset ground truth of who
  sang it), not a blind human "these voices sound alike" judgement;
- one clip per singer, so the numbers are a smoke test of generalisation, not
  a benchmark;
- the eval set is written with ``annotator_count=1`` and flagged in a sidecar
  as a synthetic self-label, so nobody mistakes it for human annotation.

The library must already exist (see ``tools/fetch_m4singer.py``).  The encoder
and calibration are re-derived from the library exactly as the CLI does, so a
result is reproducible from the library version + code revision.

Usage:
    python tools/holdout_eval.py
    python tools/holdout_eval.py --library material/m4singer/lib \\
        --material material/m4singer --encoder hubert
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from voicematch.calibrate.model import fit_calibration  # noqa: E402
from voicematch.calibrate.pairs import build_pairs  # noqa: E402
from voicematch.config import SAMPLE_RATE  # noqa: E402
from voicematch.embed.encoder import HashEncoder, HubertEncoder  # noqa: E402
from voicematch.eval.dataset import load_eval_set  # noqa: E402
from voicematch.eval.run import calibration_version, run_eval  # noqa: E402
from voicematch.library.store import load_library  # noqa: E402

REPO = "AKRTR/m4singer"
QUERY_SEC = 18.0  # long enough to pass the gate and yield a 10 s window
MAX_CLIPS_PER_QUERY = 12  # pack enough real clips so sparse songs still qualify
ROOT = Path(__file__).resolve().parents[1] / "material" / "m4singer"


def _jsonable(value):
    """Map non-finite floats to null at the JSON boundary (G4/invariant 4)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _song_of(name: str) -> str:
    """Strip the trailing clip number to get the song identity."""
    return re.sub(r"\d+\.wav$", "", name)


def build_holdout_set(material_root: Path) -> tuple[Path, dict]:
    """Download one unseen song per singer, write query WAVs + eval-set JSON.

    The song is chosen from clips not listed in ``used_clips.json`` (the clips
    that built the material), so a query is genuinely unseen.  Deterministic:
    sorted song names, first one per singer.
    """
    used = json.loads((material_root / "used_clips.json").read_text(encoding="utf-8"))
    catalog = json.loads((material_root / "catalog" / "catalog.json").read_text(encoding="utf-8"))
    singers = {s["singer_id"]: s for s in catalog["singers"]}

    all_files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".wav")]
    # Recover each singer's original dataset folder (Soprano-1, Tenor-2, ...)
    # by matching a clip name that built its material against the file list.
    # used_clips.json is the provenance the fetch tool wrote for exactly this.
    folder_of = {}
    for sid, consumed in used.items():
        first = consumed[0]
        match = [n for n in all_files if Path(n).name == first]
        if match:
            folder_of[sid] = match[0].split("/")[0]

    eval_dir = material_root / "holdout"
    clips_dir = eval_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    records = []
    provenance = {}
    for singer_id in sorted(used):
        folder = folder_of[singer_id]
        used_names = set(used[singer_id])
        candidates = [
            n for n in all_files
            if n.startswith(folder + "/") and Path(n).name not in used_names
        ]
        # A genuine hold-out must be a DIFFERENT song: later slices of the
        # material's own song are technically unseen but share the same
        # recording session, which would make the number look better than a
        # real generalisation check.  Exclude every song the material used.
        material_songs = {_song_of(name) for name in used[singer_id]}
        songs: dict[str, list[str]] = {}
        for n in sorted(candidates):
            song_name = _song_of(Path(n).name)
            if song_name in material_songs:
                continue
            songs.setdefault(song_name, []).append(n)
        if not songs:
            raise SystemExit(f"no unseen song for {singer_id}")
        song = sorted(songs)[0]  # deterministic
        # Join as many real clips of this song as needed to exceed QUERY_SEC,
        # so a query has enough voiced material for the admission rule.  Only
        # if the whole song is still short do we tile it as a last resort.
        song_clips = songs[song][:MAX_CLIPS_PER_QUERY]

        pieces = []
        for clip in song_clips:
            cached = Path(hf_hub_download(REPO, clip, repo_type="dataset", local_dir=material_root / "hf_cache"))
            audio, sr = sf.read(cached, dtype="float32", always_2d=True)
            pieces.append(audio.mean(axis=1))
        joined = np.concatenate(pieces)
        if len(joined) / sr < QUERY_SEC:
            reps = int(math.ceil(QUERY_SEC * sr / len(joined)))
            joined = np.tile(joined, reps)
        joined = joined[: int(QUERY_SEC * sr)]
        query_name = f"{singer_id}.wav"
        sf.write(clips_dir / query_name, joined, sr)

        gender = singers[singer_id]["gender"]
        pool = [sid for sid, s in singers.items() if s["gender"] == gender]
        records.append(
            {
                "query_id": f"holdout-{singer_id}",
                "path": f"clips/{query_name}",
                "gender": gender,
                "candidates": sorted(pool),
                "picked": [singer_id],
                "annotator_count": 1,
            }
        )
        provenance[singer_id] = {"folder": folder, "song": song, "clips": [Path(c).name for c in song_clips]}

    eval_path = eval_dir / "eval_set.json"
    eval_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (eval_dir / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "label_source": "dataset folder = singer identity (ground truth), NOT blind human annotation",
                "annotator_count_note": "1 = one catalogue label, not one human annotator",
                "caveat": "smoke test of generalisation on unseen material; NOT a product accuracy benchmark",
                "source": f"M4Singer {REPO}, CC BY-NC-SA 4.0, research use only",
                "singers": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path, provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=ROOT / "lib")
    parser.add_argument("--material", type=Path, default=ROOT)
    parser.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    parser.add_argument(
        "--cosine-only",
        action="store_true",
        help="diagnostic: skip calibration (canonical pure-cosine ranking)",
    )
    args = parser.parse_args(argv)

    index = load_library(args.library)
    if index.model_id != ("HUBERT_BASE" if args.encoder == "hubert" else "HashEncoder"):
        print(
            f"encoder {args.encoder!r} does not match library model {index.model_id!r}",
            file=sys.stderr,
        )
        return 2

    eval_path, provenance = build_holdout_set(args.material)
    queries = load_eval_set(eval_path)

    encoder = HubertEncoder() if args.encoder == "hubert" else HashEncoder()
    # Empty input yields the canonical cosine-only model (R22): a diagnostic
    # that isolates the frozen representation from the learned calibration.
    pairs = (
        (np.empty((0, 5)), np.empty(0), np.empty(0))
        if args.cosine_only
        else build_pairs(index, index.embeddings)
    )
    model = fit_calibration(*pairs)
    result = run_eval(queries, index, model, encoder)

    # A raw top1 is meaningless without the pool size: 0.33 over a 10-singer
    # pool is far better than 0.33 over a 5-singer pool.  Report the per-query
    # random-guess baseline so two libraries of different sizes are comparable.
    pool_sizes = [
        sum(1 for s in index.singers.values() if s.gender == q.gender) for q in queries
    ]
    chance_top1 = [1.0 / n for n in pool_sizes if n > 0]
    chance_top3 = [min(3, n) / n for n in pool_sizes if n > 0]
    baseline = {
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else None,
        "chance_top1": float(np.mean(chance_top1)) if chance_top1 else None,
        "chance_top3": float(np.mean(chance_top3)) if chance_top3 else None,
    }

    report = {
        "status": "ok",
        "command": "holdout_eval",
        "encoder": args.encoder,
        "library_version": index.version,
        "calibration_version": calibration_version(model),
        "calibrated": model.is_calibrated,
        "baseline": baseline,
        "rights": (args.material / "catalog" / "RIGHTS.json").exists()
        and json.loads((args.material / "catalog" / "RIGHTS.json").read_text(encoding="utf-8")),
        "caveat": "unseen-material smoke test with catalogue ground-truth labels; NOT a blind-listening benchmark",
        "metrics": result,
    }
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
