"""零成本演练：把真人盲听基准集的整条链跑一遍（教学用，不产出数据）。

⚠️ 本脚本生成的录音是**合成音**、标注是**脚本自己编的**，不是真人采集。
   跑出来的 G1 数字毫无意义，只用来演示"每一步在做什么、产物长什么样"。
   产物全部落在独立目录，目录内有 DEMO-ONLY.txt；绝不写入正式
   material/benchmark/，也绝不能当作基准集或引用其指标。

它按 00-START-HERE.md 里的**同一套 CLI 命令**依次执行：

    preflight  检查库/性别池/catalog 是否满足 8 位候选契约
    run        samples -> sheet -> adjudicate -> freeze -> evaluate

Usage:
    C:/Python/python.exe tools/benchmark_walkthrough.py preflight \
        --catalog material/m4singer20_3take/catalog/catalog.json \
        --library material/m4singer20_3take/lib
    C:/Python/python.exe tools/benchmark_walkthrough.py run \
        --catalog material/m4singer20_3take/catalog/catalog.json \
        --library material/m4singer20_3take/lib \
        --out material/benchmark_walkthrough
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLI = PROJECT_ROOT / "tools" / "benchmark.py"

# 设计 6.2：每条 query 展示 Top-3 + 5 位同性别随机歌手 = 8 位。
REQUIRED_POOL = 8

DEMO_NOTICE = """DEMO ONLY — 演练产物，不是数据
========================================
本目录由 tools/benchmark_walkthrough.py 生成，用于教学演示工具链如何串起来。

  * 录音是合成正弦音，不是真人清唱；
  * 标注是脚本按 provenance 的系统 Top-3 自己编的，不是 3 位真人独立听标；
  * 因此 kappa、Top-1/Top-3、受理率等所有数字都毫无意义；
  * MANIFEST.json 里的 label_source 是工具链固定声明，在演练场景下
    不代表真人盲听多数标注。

严禁把本目录任何产物写进报告、当作基准集，或与 M4Singer 留出评估/真人
基准集的数字混用。真人采集请按 docs/benchmark/00-START-HERE.md 执行。
"""


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _singers_from_manifest(manifest: dict) -> list[tuple[str, str]]:
    singers = manifest.get("singers", {})
    if isinstance(singers, dict):
        return [(sid, str(info.get("gender", ""))) for sid, info in singers.items()]
    return [
        (str(item.get("singer_id")), str(item.get("gender", "")))
        for item in singers
    ]


def preflight(catalog: Path, library: Path) -> dict:
    """检查开工前置条件；返回可打印的结果字典，失败抛 ValueError。"""
    problems: list[str] = []
    if not catalog.is_file():
        problems.append(f"catalog 不存在: {catalog}")
    manifest_path = library / "manifest.json"
    if not manifest_path.is_file():
        problems.append(f"库 manifest 不存在: {manifest_path}")

    pools: dict[str, int] = {}
    lib_ids: set[str] = set()
    cat_ids: set[str] = set()
    if not problems:
        singers = _singers_from_manifest(_json(manifest_path))
        pools = dict(Counter(g for _, g in singers))
        lib_ids = {sid for sid, _ in singers}
        cat = _json(catalog)
        cat_ids = {str(s["singer_id"]) for s in cat.get("singers", [])}
        for gender in ("female", "male"):
            n = pools.get(gender, 0)
            if n < REQUIRED_POOL:
                problems.append(
                    f"同性别池 {gender}={n} < {REQUIRED_POOL}：候选会静默降级成 "
                    f"{n} 选 1，违反设计 6.2 的 8 位契约。请改用 20 人库。"
                )
        missing = sorted(lib_ids - cat_ids)
        if missing:
            problems.append(
                f"库里有 {len(missing)} 位歌手在 catalog 中找不到（样本会对不上）: "
                f"{missing[:5]}"
            )

    result = {
        "status": "ok" if not problems else "fail",
        "library": str(library),
        "catalog": str(catalog),
        "gender_pools": pools,
        "required_pool": REQUIRED_POOL,
        "n_library_singers": len(lib_ids),
        "n_catalog_singers": len(cat_ids),
        "rights_declared": (library / "catalog" / "RIGHTS.json").is_file(),
        "problems": problems,
    }
    if problems:
        raise ValueError(result)
    return result


def _voiced(secs: float, f0: float, amp: float = 0.30) -> np.ndarray:
    """合成"清唱"：带短静音间隔的周期正弦，能通过查询侧质量门与选窗。"""
    rate = 16000
    period = int(rate * 0.6)
    t = np.arange(period) / rate
    burst = (amp * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    burst[int(rate * 0.5):] = 0.0
    out = np.zeros(int(rate * secs), dtype=np.float32)
    pos = 0
    while pos + period <= out.size:
        out[pos:pos + period] = burst
        pos += period
    return out


def _run_cli(step: str, description: str, *args: str) -> dict:
    """按手册里的同一条命令调用 tools/benchmark.py，并打印教学说明。"""
    cmd = [sys.executable, str(CLI), *args]
    print(f"\n[{step}] {description}")
    print("  $ " + " ".join(str(a) for a in args))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("  !! 命令失败：")
        print("  stdout:", proc.stdout.strip()[:800])
        print("  stderr:", proc.stderr.strip()[-1500:])
        raise RuntimeError(f"step {step} failed with exit {proc.returncode}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        payload = {"raw_stdout": proc.stdout.strip()[:400]}
    print("  -> " + json.dumps(payload, ensure_ascii=False)[:600])
    return payload


def _fake_annotations(sheet: list[dict], provenance: dict) -> list[dict]:
    """编 3 位标注者的标注。⭐ 这不是真人数据，仅用于让链路跑通。

    规则：首选取 provenance 的系统 Top-3 首位（一定在该 query 的候选里，
    因为设计保证"系统答案在场"）；第 3 位标注者对最后一条选"都不像"，
    制造一点真实感与标签多样性，避免 kappa 因标签恒定变成 nan。
    """
    top_by_qid = {
        q["query_id"]: q["system_top"]
        for q in provenance["queries"]
    }
    annotations: list[dict] = []
    last_qid = sheet[-1]["query_id"]
    for entry in sheet:
        qid = entry["query_id"]
        top = top_by_qid.get(qid) or []
        if not top:
            # 不可打分的 query 没有系统 Top-3；用候选首位作占位首选。
            top = list(entry["candidates"][:1])
        first = top[0]
        for rater in ("rater1", "rater2", "rater3"):
            if rater == "rater3" and qid == last_qid:
                picked: list[str] = []
            else:
                picked = [first]
            annotations.append({
                "query_id": qid,
                "annotator_id": rater,
                "picked": picked,
            })
    return annotations


def run(catalog: Path, library: Path, out: Path, encoder: str, seed: int) -> None:
    print("=" * 68)
    print("真人盲听基准集 · 零成本演练（DEMO，不产出任何可用数据）")
    print("=" * 68)

    info = preflight(catalog, library)
    print("\n[preflight] 前置检查通过")
    print("  " + json.dumps(info, ensure_ascii=False))

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "DEMO-ONLY.txt").write_text(DEMO_NOTICE, encoding="utf-8")
    audio_dir = out / "audio"
    audio_dir.mkdir(exist_ok=True)

    # --- 第 0 步：合成录音 + specs.json（正式流程里这两样来自真人）---
    print("\n[0/6] 造 3 条合成清唱 + specs.json（正式流程由真人录音与登记替代）")
    voices = [
        ("demo-f1", "audio/demo-f1.wav", "female", "subj-f", "phone-a", 220.0),
        ("demo-f2", "audio/demo-f2.wav", "female", "subj-f", "phone-b", 262.0),
        ("demo-m1", "audio/demo-m1.wav", "male", "subj-m", "phone-a", 146.0),
    ]
    specs = []
    for qid, rel, gender, subj, device, f0 in voices:
        sf.write(out / rel, _voiced(20.0, f0), 16000)
        specs.append({
            "query_id": qid, "path": rel, "gender": gender,
            "subject_id": subj, "device_id": device,
        })
    (out / "specs.json").write_text(
        json.dumps(specs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("  -> 3 条合成录音 + specs.json")

    samples_dir = out / "samples"
    sheet_path = out / "sheet.json"
    prov_path = out / "provenance.json"
    ann_path = out / "annotations.json"
    adj_path = out / "adjudicated.json"
    frozen_dir = out / "frozen"

    _run_cli(
        "1/6", "导出候选试听样本（标注者要听到每位歌手）",
        "samples", "--catalog", str(catalog),
        "--out-dir", str(samples_dir), "--secs", "10",
    )
    _run_cli(
        "2/6", "生成盲听清单；系统 Top-3 分离到内部 provenance",
        "sheet", "--specs", str(out / "specs.json"),
        "--specs-dir", str(out), "--library", str(library),
        "--encoder", encoder, "--out", str(sheet_path),
        "--provenance", str(prov_path), "--seed", str(seed),
    )

    sheet = _json(sheet_path)
    provenance = _json(prov_path)
    n8 = sum(1 for e in sheet if len(e["candidates"]) == REQUIRED_POOL)
    print(f"  -> 校验：{n8}/{len(sheet)} 条清单恰好 8 位候选")
    if n8 != len(sheet):
        raise RuntimeError("候选数不是 8；检查是否选错库（见 00-START-HERE.md §2.3）")

    # --- 第 3 步：编 3 份标注（正式流程由 3 位真人独立听标替代）---
    print("\n[3/6] 编 3 位标注者的标注（⭐ 脚本编的，不是真人；正式流程由 3 人独立听标替代）")
    annotations = _fake_annotations(sheet, provenance)
    ann_path.write_text(
        json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"  -> {len(annotations)} 条标注 = {len(sheet)} 条 query × 3 人")

    _run_cli(
        "4/6", "kappa 一致性检验 + 严格多数裁决",
        "adjudicate", "--annotations", str(ann_path),
        "--sheet", str(sheet_path), "--out", str(adj_path),
    )
    _run_cli(
        "5/6", "冻结版本化基准集（写盘即读回自校验）",
        "freeze", "--sheet", str(sheet_path),
        "--adjudicated", str(adj_path), "--out", str(frozen_dir),
        "--library", str(library), "--encoder", encoder,
    )

    # 即使有人只把 frozen/ 拷走，警告也必须跟着走：MANIFEST 里打 DEMO 标记。
    manifest_path = frozen_dir / "MANIFEST.json"
    manifest = _json(manifest_path)
    manifest["demo"] = True
    manifest["demo_warning"] = (
        "SYNTHETIC WALKTHROUGH - recordings are generated tones and annotations "
        "were fabricated by a script. label_source does NOT represent real human "
        "blind annotation here. No metric from this set is evidence."
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (frozen_dir / "DEMO-ONLY.txt").write_text(DEMO_NOTICE, encoding="utf-8")
    print("  -> 已往 MANIFEST.json 写入 demo=true 标记（防止误用）")

    report = _run_cli(
        "6/6", "跑 G1（★ 演练数字无意义，只看流程）",
        "evaluate", "--eval-set", str(frozen_dir / "eval_set.json"),
        "--library", str(library), "--encoder", encoder,
    )
    metrics = report.get("metrics", {})
    if metrics.get("top1") == 1.0:
        print("  ※ 演练里 top1=1.0 是必然：模拟标注直接抄了 provenance 的系统答案。")
        print("    真人标注不会这样；这里的数字只用来演示输出长什么样。")

    print("\n" + "=" * 68)
    print("演练完成。看这些产物，对照 00-START-HERE.md 第 5 节：")
    print(f"  {out / 'specs.json'}            录音清单（正式：你填）")
    print(f"  {out / 'samples'}/            候选试听样本")
    print(f"  {sheet_path}           发给标注者的清单（不含系统答案）")
    print(f"  {prov_path}           ★ 内部，绝不外发")
    print(f"  {ann_path}       3 人标注（正式：真人各导一份）")
    print(f"  {adj_path}      kappa + 多数裁决")
    print(f"  {frozen_dir}/            eval_set.json + MANIFEST.json")
    print("\n⚠️  本目录是 DEMO：录音是合成音、标注是脚本编的，所有数字都不是结论。")
    print("    真人采集请按 docs/benchmark/00-START-HERE.md 从『第 1 阶段』开始。")
    print("=" * 68)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--catalog", type=Path, required=True)
    common.add_argument("--library", type=Path, required=True)

    sub.add_parser("preflight", parents=[common], help="check prerequisites")

    p_run = sub.add_parser("run", parents=[common], help="run the full walkthrough")
    p_run.add_argument("--out", type=Path, default=Path("material/benchmark_walkthrough"))
    p_run.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    p_run.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            print(json.dumps(preflight(args.catalog, args.library),
                             ensure_ascii=False, indent=2))
        else:
            run(args.catalog, args.library, args.out, args.encoder, args.seed)
    except ValueError as exc:
        # preflight 失败把整个 result 打印出来（含 problems），exit 2。
        detail = exc.args[0]
        print(json.dumps(detail if isinstance(detail, dict) else {"detail": str(exc)},
                         ensure_ascii=False, indent=2))
        return 2
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"status": "error", "detail": str(exc)},
                         ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
