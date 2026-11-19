"""Batch hold-out evaluation: top1/top3 on unseen M4Singer material.

This script builds a hold-out query set from M4Singer clips that were NOT used
to build the library (a different song per singer), then runs the project's
own :func:`voicematch.eval.run.run_eval` to report top1/top3 with all its
denominators.  It is a **sanity check that the pipeline works on real, unseen
vocal material** -- NOT a product accuracy result:

- the "label" is the folder the clip came from (dataset ground truth of who
  sang it), not a blind human "these voices sound alike" judgement;
- one query per held-out *song* per singer (one song by default,
  ``--queries-per-singer`` for more), so the numbers are a smoke test of
  generalisation, not a benchmark;
- the eval set is written with ``annotator_count=1`` and flagged in a sidecar
  as a synthetic self-label, so nobody mistakes it for human annotation.

The library must already exist (see ``tools/fetch_m4singer.py``).  The encoder
and calibration are re-derived from the library exactly as the CLI does, so a
result is reproducible from the library version + code revision.

Why an uncalibratable library is refused
----------------------------------------
``fit_calibration`` never fails on a library it cannot learn from: it returns
the canonical cosine-only fallback (``coef = [1, 0, 0, 0, 0]``, ``auc = nan``).
Scoring with that model is *identical* to pure cosine, so a top1 computed from
it is a cosine top1 that happens to be printed next to ``"calibrated": false``
-- and in a slide or a commit message the qualifier is the first thing to
disappear.  Without the gate, a ten-singer, one-recording-each library would
report the fallback's top1 as if it measured the calibrated product.

The same holds for a library built before R29 stamped ``material_id``: its
positives may be two windows of one recording, the leak that scored an
in-library AUC of 1.000 and then collapsed on unseen songs (R29 section 7 says
such a library MUST NOT be cited as accuracy evidence).  The artefact cannot
distinguish a positive that spans two recordings from two windows of one take,
which is itself the reason to refuse it.

So the default path now refuses to produce any metric unless the calibration
is measurable *and* the library's pairing structure is verifiable:

- ``auc`` measured out-of-fold (``model.is_calibrated``), and
- ``pairing_status(index).status`` is not ``unverifiable_legacy``.

A refusal is a one-line JSON document on **stdout** with ``status: "error"``,
``reason`` in {``uncalibrated_library``, ``unverifiable_legacy_library``}, the
pair census and R29 state that caused it, and exit status ``2``.  No hold-out
audio is downloaded first: the gate runs before the query set is assembled.

``--cosine-only`` is the explicit way through for the uncalibratable case,
because it *is* the honest description of what such a run measures: the flag
already means "no calibration, canonical cosine ranking", and every report and
per-query row it produces is tagged ``metric_kind: "cosine_only_diagnostic"``.
A number that a reader might quote as calibrated is never produced by accident.

A pre-R29 library has one more route, because its numbers are reproducible in
principle and sometimes needed (re-running an old experiment).  Setting
``--allow-unverifiable-legacy`` keeps the calibrated path but labels the run
``metric_kind: "calibrated_unverified_legacy"`` and ``r29_verified: false``,
prints one warning line on stderr, and never applies to the other refusal --
an uncalibratable library stays refused, since there the "calibrated" number
would be the cosine fallback with no provenance distinction at all.

Exit status: ``0`` for a completed run, ``2`` for a refused run or a
library/encoder mismatch, ``3`` when the per-query sidecar disagrees with
``run_eval``.

Usage:
    python tools/holdout_eval.py
    python tools/holdout_eval.py --library material/m4singer/lib \\
        --material material/m4singer --encoder hubert
    python tools/holdout_eval.py --material material/m4singer20_3take \\
        --library material/m4singer20_3take/lib --queries-per-singer 3
    # a library that cannot be calibrated: explicit diagnostic mode only
    python tools/holdout_eval.py --library material/m4singer/lib \\
        --material material/m4singer --cosine-only
    # pre-R29 library, calibrated numbers for reproducing an old experiment
    python tools/holdout_eval.py --library material/m4singer/lib \\
        --material material/m4singer --allow-unverifiable-legacy

``--per-query-out`` additionally writes one row per query (rank of the true
singer, Top-1/Top-3 hit, the Top-3 ids and scores).  It re-runs the query path,
so it doubles the runtime; the rows are checked against ``run_eval``'s
aggregates and the run fails loudly if they disagree.  Two such files (one
calibrated, one ``--cosine-only``) are what ``tools/paired_compare.py`` turns
into a paired significance test.
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

from voicematch.audio.io import load_audio  # noqa: E402
from voicematch.audio.preprocess import PREPROCESS_QUERY, preprocess  # noqa: E402
from voicematch.audio.quality import assess_quality  # noqa: E402
from voicematch.audio.windows import select_best_window  # noqa: E402
from voicematch.calibrate.model import fit_calibration  # noqa: E402
from voicematch.calibrate.pairs import build_pairs  # noqa: E402
from voicematch.calibrate.score import (  # noqa: E402
    NonFinitePrototypeError,
    rank_singers,
)
from voicematch.config import SAMPLE_RATE  # noqa: E402
from voicematch.embed.encoder import HashEncoder, HubertEncoder  # noqa: E402
from voicematch.eval.dataset import load_eval_set  # noqa: E402
from voicematch.eval.run import (  # noqa: E402
    calibration_version,
    run_eval,
    truncate_to_input_contract,
)
from voicematch.features.extract import extract_features  # noqa: E402
from voicematch.library import (  # noqa: E402
    UNVERIFIABLE_LEGACY,
    PairingStatus,
    pairing_status,
)
from voicematch.library.store import load_library  # noqa: E402

REPO = "AKRTR/m4singer"
QUERY_SEC = 18.0  # long enough to pass the gate and yield a 10 s window
MAX_CLIPS_PER_QUERY = 12  # pack enough real clips so sparse songs still qualify
ROOT = Path(__file__).resolve().parents[1] / "material" / "m4singer"

#: ``metric_kind`` of a run whose numbers come from a fitted calibration.
CALIBRATED_METRIC: str = "calibrated"

#: ``metric_kind`` of a run that measures the frozen embedding only.  It labels
#: both the report and every per-query row, so a sidecar cannot be read as a
#: calibrated result after the report is separated from it.
COSINE_ONLY_DIAGNOSTIC: str = "cosine_only_diagnostic"

#: ``metric_kind`` of a calibrated run on a library whose pairing structure
#: cannot be verified (pre-R29, no ``material_id``).  Only produced under
#: ``--allow-unverifiable-legacy``; the label is the price of the escape hatch.
CALIBRATED_UNVERIFIED_LEGACY: str = "calibrated_unverified_legacy"

#: Stable refusal token: the library yields no measurable calibration.
UNCALIBRATED_LIBRARY: str = "uncalibrated_library"

#: Stable refusal token: the library predates ``material_id``, so R29 cannot be
#: verified from the artefact and its positives may be same-recording pairs.
UNVERIFIABLE_LIBRARY: str = "unverifiable_legacy_library"


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


def _cached_clip(material_root: Path, clip: str) -> Path:
    """Return a clip from the material cache, downloading only when absent.

    ``hf_hub_download`` is the fallback, not the fast path: a warm cache means
    an evaluation can be re-run offline, and the download path writes and
    removes temporary files that a sandboxed runner may refuse to delete in
    bulk.  Both paths return the same file, so the audio is identical.
    """
    local = material_root / "hf_cache" / clip
    if local.exists():
        return local
    return Path(
        hf_hub_download(REPO, clip, repo_type="dataset", local_dir=material_root / "hf_cache")
    )


def _folders_from_clips(
    used: dict[str, list[str]],
    all_files: list[str],
    manifest_path: Path | None = None,
) -> dict[str, str]:
    """Recover each singer's dataset folder, manifest first, clips as fallback.

    Filenames are **not** unique across folders: M4Singer's singers perform the
    same songs, so ``baiyang0000.wav`` exists under many voice folders.  The
    previous version of this tool matched one filename against the global list
    and took the first hit; for some singers that attributed the hold-out query
    to another singer's audio, which is a guaranteed miss that reads as a model
    error.  Two sources are used instead, in order of authority:

    1. the fetch tool's manifest, whose singer name is ``M4Singer <folder>`` --
       the folder the material was actually assembled from;
    2. intersection of the folders that contain *every* clip in
       ``used_clips.json``.  This is usually one folder, but two voice folders
       can share a whole song list, so an ambiguous intersection is refused
       rather than resolved by sort order.
    """
    by_name: dict[str, set[str]] = {}
    for name in all_files:
        by_name.setdefault(Path(name).name, set()).add(name.split("/")[0])

    folders: dict[str, str] = {}
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for singer in manifest.get("singers", []):
            name = str(singer.get("name", ""))
            if name.startswith("M4Singer "):
                folders[str(singer["singer_id"])] = name[len("M4Singer ") :].strip()

    for singer_id, consumed in used.items():
        if singer_id in folders:
            continue
        candidates: set[str] | None = None
        for clip in consumed:
            here = by_name.get(Path(clip).name, set())
            candidates = here if candidates is None else candidates & here
            if not candidates:
                raise SystemExit(
                    f"cannot recover the dataset folder of {singer_id!r}: "
                    f"its clips do not share one folder"
                )
        if candidates is None or len(candidates) != 1:
            raise SystemExit(
                f"cannot recover the dataset folder of {singer_id!r}: clip "
                f"names are consistent with {sorted(candidates or [])}; add its "
                f"singer name to {manifest_path} or a used_clips entry"
            )
        folders[singer_id] = next(iter(candidates))

    known = {name.split("/")[0] for name in all_files}
    for singer_id, folder in folders.items():
        if folder not in known:
            raise SystemExit(
                f"singer {singer_id!r}: folder {folder!r} is not in the dataset"
            )
    return folders


def build_holdout_set(
    material_root: Path, queries_per_singer: int = 1
) -> tuple[Path, dict]:
    """Write query WAVs and the eval-set JSON for unseen songs.

    ``queries_per_singer`` (default 1) asks for N *different* unseen songs per
    singer.  More queries per singer is the only way to buy statistical power
    here: adding singers widens the candidate pool and makes the question
    harder at the same time, while adding held-out songs of the same singers
    keeps the pool fixed and only reduces query noise.

    Songs are chosen deterministically: from clips not listed in
    ``used_clips.json`` (the material's own recordings), sorted by song name,
    first N that are not any song the material used.  A song that happens to
    yield no admissible window is *not* replaced after the fact -- the runner
    reports it as skipped, and swapping it for a friendlier song would be
    post-hoc query selection.
    """
    if queries_per_singer < 1:
        raise SystemExit("queries_per_singer must be at least 1")
    used = json.loads((material_root / "used_clips.json").read_text(encoding="utf-8"))
    catalog = json.loads((material_root / "catalog" / "catalog.json").read_text(encoding="utf-8"))
    singers = {s["singer_id"]: s for s in catalog["singers"]}

    all_files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings if s.rfilename.endswith(".wav")]
    # Recover each singer's original dataset folder (Soprano-1, Tenor-2, ...)
    # from the clips that built its material.  Filenames collide across folders
    # (same songs, different singers), so one filename is not enough; see
    # :func:`_folders_from_clips`.
    folder_of = _folders_from_clips(used, all_files, material_root / "manifest.json")

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
        chosen = sorted(songs)[:queries_per_singer]
        if len(chosen) < queries_per_singer:
            raise SystemExit(
                f"{singer_id} has only {len(chosen)} unseen songs, "
                f"need {queries_per_singer}"
            )

        gender = singers[singer_id]["gender"]
        pool = [sid for sid, s in singers.items() if s["gender"] == gender]
        for song in chosen:
            # Join as many real clips of this song as needed to exceed
            # QUERY_SEC, so a query has enough voiced material for the
            # admission rule.  Only if the whole song is still short do we
            # tile it as a last resort.
            song_clips = songs[song][:MAX_CLIPS_PER_QUERY]
            pieces = []
            for clip in song_clips:
                cached = _cached_clip(material_root, clip)
                audio, sr = sf.read(cached, dtype="float32", always_2d=True)
                pieces.append(audio.mean(axis=1))
            joined = np.concatenate(pieces)
            if len(joined) / sr < QUERY_SEC:
                reps = int(math.ceil(QUERY_SEC * sr / len(joined)))
                joined = np.tile(joined, reps)
            joined = joined[: int(QUERY_SEC * sr)]
            query_id = f"holdout-{singer_id}--{song}"
            query_name = f"{query_id}.wav"
            sf.write(clips_dir / query_name, joined, sr)

            records.append(
                {
                    "query_id": query_id,
                    "path": f"clips/{query_name}",
                    "gender": gender,
                    "candidates": sorted(pool),
                    "picked": [singer_id],
                    "annotator_count": 1,
                }
            )
            provenance[query_id] = {
                "singer_id": singer_id,
                "folder": folder,
                "song": song,
                "clips": [Path(c).name for c in song_clips],
            }

    eval_path = eval_dir / "eval_set.json"
    eval_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (eval_dir / "PROVENANCE.json").write_text(
        json.dumps(
            {
                "label_source": "dataset folder = singer identity (ground truth), NOT blind human annotation",
                "annotator_count_note": "1 = one catalogue label, not one human annotator",
                "caveat": "smoke test of generalisation on unseen material; NOT a product accuracy benchmark",
                "source": f"M4Singer {REPO}, CC BY-NC-SA 4.0, research use only",
                "queries_per_singer": queries_per_singer,
                "queries": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_path, provenance


def per_query_rows(
    queries, index, model, encoder, metric_kind: str
) -> list[dict]:
    """Rank every query on the product path and record where the truth landed.

    ``run_eval`` returns aggregates only, which cannot say *which* queries the
    calibration helped -- and a paired test between two runs needs exactly that.
    This replays the same per-query path (quality gate, shared window rule,
    features, frozen encoder, ``rank_singers`` over the whole gender pool) and
    records the rank of the true singer.  ``main`` compares the resulting hit
    rates with ``run_eval``'s and refuses to write the file if they disagree, so
    a drift between this copy and the runner fails the run instead of producing
    a plausible sidecar about a different pipeline.

    Every row carries ``metric_kind``: a sidecar is often copied away from the
    report that explains it, and "cosine_only_diagnostic" must survive that
    trip.  The field is additive, so ``tools/paired_compare.py`` and older
    readers keep working.
    """
    rows: list[dict] = []
    for query in queries:
        audio, truncated = truncate_to_input_contract(load_audio(query.path))
        report = assess_quality(audio, SAMPLE_RATE)
        row = {
            "query_id": query.query_id,
            "singer_id": min(query.picked) if query.picked else None,
            "gender": query.gender,
            "metric_kind": metric_kind,
            "accepted": not report.rejected,
            "skipped": False,
            "truncated": truncated,
            "rank_of_truth": None,
            "top1": False,
            "top3": False,
            "ranking": [],
            "scores": [],
        }
        if not report.rejected:
            processed = preprocess(audio, PREPROCESS_QUERY)
            window = select_best_window(processed, SAMPLE_RATE)
            if window is None:
                row["skipped"] = True
            else:
                start, end = window
                segment = processed[
                    int(round(start * SAMPLE_RATE)) : int(round(end * SAMPLE_RATE))
                ]
                features = extract_features(segment, SAMPLE_RATE)
                embedding = encoder.encode(segment, SAMPLE_RATE)
                pool_size = sum(
                    1
                    for singer in index.singers.values()
                    if singer.gender == query.gender
                )
                try:
                    scored = rank_singers(
                        features,
                        embedding,
                        index,
                        model,
                        query.gender,
                        top_k=max(3, pool_size),
                    )
                except NonFinitePrototypeError:
                    row["skipped"] = True
                else:
                    ids = [entry.singer_id for entry in scored]
                    row["ranking"] = ids[:3]
                    row["scores"] = [
                        round(float(entry.score), 6) for entry in scored[:3]
                    ]
                    hits = [
                        position
                        for position, singer_id in enumerate(ids, start=1)
                        if singer_id in query.picked
                    ]
                    if hits:
                        row["rank_of_truth"] = hits[0]
                        row["top1"] = hits[0] == 1
                        row["top3"] = hits[0] <= 3
        rows.append(row)
    return rows


def _same_rate(observed: float, reported: float) -> bool:
    """Compare a per-query rate with the runner's, treating nan == nan."""
    if math.isnan(observed) and math.isnan(reported):
        return True
    return math.isclose(observed, reported, rel_tol=0.0, abs_tol=1e-12)


def pair_census(pairs) -> dict[str, int]:
    """Count the rows the calibration would learn from.

    ``positives`` is the number that decides whether a calibrated headline can
    exist at all: with none, ``fit_calibration`` returns the canonical
    cosine-only fallback and every score is a cosine wrapped in a
    probability-shaped number.  ``folds`` is reported because a library can have
    positives and still be unmeasurable -- one singer fold means no
    singer-disjoint out-of-fold AUC (R21).
    """
    _values, labels, groups = pairs
    labels = np.asarray(labels)
    total = int(labels.size)
    positives = int(np.sum(labels == 1)) if total else 0
    folds = int(np.unique(np.asarray(groups)).size) if total else 0
    return {
        "total": total,
        "positives": positives,
        "negatives": total - positives,
        "folds": folds,
    }


def calibration_refusal(
    *,
    calibrated: bool,
    pairing: PairingStatus,
    census: dict[str, int],
    library_version: str,
    calibration_version: str,
    cosine_only: bool,
    allow_unverifiable_legacy: bool = False,
) -> dict | None:
    """Return the error document that blocks an uncalibrated top1, or ``None``.

    Two states are refused, for the same reason: the number would be a cosine
    number wearing a calibrated label.

    1. ``calibrated`` is false -- no measurable out-of-fold AUC, so
       ``fit_calibration`` returned the canonical fallback.
    2. ``pairing.status`` is ``unverifiable_legacy`` -- segments without
       ``material_id``; the positives may be two windows of one recording, the
       leak R29 section 7 forbids citing as accuracy evidence.

    ``cosine_only`` bypasses both because the caller has already declared the
    run to be an encoder-only diagnostic; the report is labelled as one instead
    of the request being served silently.

    ``allow_unverifiable_legacy`` bypasses only the second state: an old library
    can still be re-measured, but the run keeps the machine-readable mark
    (``metric_kind: calibrated_unverified_legacy``, ``r29_verified: false``)
    that says its positives are unverified.  It deliberately does **not** lift
    the first refusal: there the calibrated score is the cosine fallback, so
    there is nothing to reproduce and no label that would make it true.
    """
    if cosine_only:
        return None

    # Check measurability independently of the legacy hatch.  A pre-R29 library
    # can also have no measurable AUC; in that combined state the hatch must
    # not return early, because there is no fitted calibration to reproduce.
    if not calibrated:
        if census["positives"] == 0:
            why = "there is no same-singer positive pair at all"
        elif census["folds"] < 2:
            why = (
                f"its {census['positives']} positives live in "
                f"{census['folds']} singer fold(s), so no singer-disjoint "
                "out-of-fold AUC is measurable"
            )
        else:
            why = "its out-of-fold AUC is not measurable"
        reason = UNCALIBRATED_LIBRARY
        detail = (
            f"library {library_version} cannot be calibrated: {why} "
            f"({census['positives']} positives in {census['total']} pair rows). "
            "Scoring falls back to the canonical cosine model, so a top1 here "
            "would be a cosine top1 reported as a calibrated one."
        )
        hint = (
            "rebuild with at least two different recordings per singer (R29), "
            "or pass --cosine-only for an explicitly labelled cosine diagnostic"
        )
    elif pairing.status == UNVERIFIABLE_LEGACY:
        if allow_unverifiable_legacy:
            return None
        reason = UNVERIFIABLE_LIBRARY
        detail = (
            f"library {library_version} has segments without material_id, so R29 "
            "cannot be verified from the artefact: its positive pairs may be two "
            "windows of one recording -- the leak that scores a perfect "
            "in-library AUC and then collapses on unseen songs. A top1 from this "
            "library would be evidence of that leak, not of singer similarity."
        )
        hint = (
            "rebuild the library with the current builder (it stamps "
            "material_id), pass --cosine-only for an explicitly labelled cosine "
            "diagnostic, or pass --allow-unverifiable-legacy only to reproduce "
            "an old experiment (the output remains marked unverified)"
        )
    else:
        return None

    return {
        "status": "error",
        "command": "holdout_eval",
        "reason": reason,
        "detail": detail,
        "hint": hint,
        "library_version": library_version,
        "calibration_version": calibration_version,
        "pairing": {
            "status": pairing.status,
            "single_material_singers": list(pairing.single_material_singers),
        },
        "pairs": census,
    }


def metric_kind_for(*, cosine_only: bool, r29_verified: bool) -> str:
    """Name what the numbers in a completed run measure.

    ``cosine_only`` wins over the library state: the run did not fit a
    calibration at all, so the pairing question does not arise.  Otherwise a
    verifiable library yields ``calibrated`` and a pre-R29 one (only reachable
    through ``--allow-unverifiable-legacy``) yields the marked third value.
    """
    if cosine_only:
        return COSINE_ONLY_DIAGNOSTIC
    return CALIBRATED_METRIC if r29_verified else CALIBRATED_UNVERIFIED_LEGACY


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
    parser.add_argument(
        "--allow-unverifiable-legacy",
        action="store_true",
        help=(
            "run the calibrated path on a pre-R29 library (no material_id) "
            "instead of refusing it. The report is marked "
            "metric_kind=calibrated_unverified_legacy and r29_verified=false; "
            "it does NOT unblock a library that cannot be calibrated"
        ),
    )
    parser.add_argument(
        "--queries-per-singer",
        type=int,
        default=1,
        help=(
            "number of different unseen songs per singer (default 1). More "
            "queries reduce query noise without changing the candidate pool; "
            "adding singers changes the pool and the difficulty"
        ),
    )
    parser.add_argument(
        "--per-query-out",
        type=Path,
        default=None,
        help=(
            "write one row per query (rank of truth, top1/top3, top-3 ids); "
            "re-runs the query path, so runtime roughly doubles"
        ),
    )
    args = parser.parse_args(argv)

    index = load_library(args.library)
    if index.model_id != ("HUBERT_BASE" if args.encoder == "hubert" else "HashEncoder"):
        print(
            f"encoder {args.encoder!r} does not match library model {index.model_id!r}",
            file=sys.stderr,
        )
        return 2

    encoder = HubertEncoder() if args.encoder == "hubert" else HashEncoder()
    # The library's own pair matrix is always built: the report's `pairs` block
    # describes what the library offers (and what the gate inspected), even on a
    # diagnostic run that deliberately refuses to learn from it.  Empty input
    # yields the canonical cosine-only model (R22), which isolates the frozen
    # representation from the learned calibration.
    pairs = build_pairs(index, index.embeddings)
    census = pair_census(pairs)
    model = fit_calibration(
        *((np.empty((0, 5)), np.empty(0), np.empty(0)) if args.cosine_only else pairs)
    )

    # The gate runs before the hold-out set is assembled: a refused run must not
    # download clips or write query WAVs first, and must produce no metric at
    # all -- see the module docstring for why the fallback's top1 is worse than
    # no number.
    pairing = pairing_status(index)
    r29_verified = pairing.status != UNVERIFIABLE_LEGACY
    refusal = calibration_refusal(
        calibrated=model.is_calibrated,
        pairing=pairing,
        census=census,
        library_version=index.version,
        calibration_version=calibration_version(model),
        cosine_only=args.cosine_only,
        allow_unverifiable_legacy=args.allow_unverifiable_legacy,
    )
    if refusal is not None:
        print(json.dumps(refusal, ensure_ascii=False, sort_keys=True))
        return 2
    if not r29_verified and not args.cosine_only:
        # The escape hatch is granted, not hidden: one line a human will see in
        # a terminal, plus a field a machine will see in the report.
        print(
            "holdout_eval: warning: this library predates R29 (no material_id); "
            "its positives may be same-recording pairs, so the calibrated "
            "numbers below are unverified_legacy -- do not cite them as "
            "accuracy evidence",
            file=sys.stderr,
        )

    metric_kind = metric_kind_for(cosine_only=args.cosine_only, r29_verified=r29_verified)
    eval_path, provenance = build_holdout_set(args.material, args.queries_per_singer)
    queries = load_eval_set(eval_path)
    result = run_eval(queries, index, model, encoder)

    if args.per_query_out is not None:
        rows = per_query_rows(queries, index, model, encoder, metric_kind)
        ranked = [row for row in rows if row["ranking"] and row["singer_id"]]
        for key in ("top1", "top3"):
            observed = (
                sum(1 for row in ranked if row[key]) / len(ranked)
                if ranked
                else float("nan")
            )
            if not _same_rate(observed, float(result[key])):
                print(
                    f"per-query {key}={observed!r} disagrees with run_eval "
                    f"{result[key]!r}; refusing to write {args.per_query_out}",
                    file=sys.stderr,
                )
                return 3
        args.per_query_out.parent.mkdir(parents=True, exist_ok=True)
        args.per_query_out.write_text(
            json.dumps(_jsonable(rows), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

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
        # "calibrated" / "calibrated_unverified_legacy" /
        # "cosine_only_diagnostic": the report and the per-query rows carry the
        # same label, so a number cannot be quoted as a calibrated result after
        # the qualifier is dropped.
        "metric_kind": metric_kind,
        # Machine-readable version of the same fact: the library's pairing
        # structure was verified, so its calibration pairs span recordings.
        "r29_verified": r29_verified,
        # R29 state of the library and what calibration had to learn from.
        # A `calibrated: true` next to a non-empty single_material_singers lists
        # the singers that contributed no cross-recording positive.
        "pairing": {
            "status": pairing.status,
            "single_material_singers": list(pairing.single_material_singers),
        },
        "pairs": census,
        "queries_per_singer": args.queries_per_singer,
        "baseline": baseline,
        "rights": (args.material / "catalog" / "RIGHTS.json").exists()
        and json.loads((args.material / "catalog" / "RIGHTS.json").read_text(encoding="utf-8")),
        "caveat": (
            "unseen-material smoke test with catalogue ground-truth labels; "
            "NOT a blind-listening benchmark"
            if r29_verified or args.cosine_only
            else "unseen-material smoke test with catalogue ground-truth labels; "
            "NOT a blind-listening benchmark; R29 UNVERIFIED (pre-R29 library, "
            "positives may be same-recording pairs) -- reproduce-only, not evidence"
        ),
        "metrics": result,
        "per_query_out": str(args.per_query_out) if args.per_query_out else None,
    }
    print(json.dumps(_jsonable(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
