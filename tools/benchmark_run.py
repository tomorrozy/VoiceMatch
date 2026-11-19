"""真人盲听基准集的 runbook 执行器（设计 6.1/6.2）。

`tools/benchmark.py` 提供的是**单步**命令；这条链真正难的地方不是单步，而是
"现在轮到谁、下一步该敲什么、哪些产物必须等人"。本工具只做三件事：

    init       建工作目录骨架 + 配置 + specs 模板（不覆盖已有文件）
    preflight  在录音开录前就查出错：文件名重名、时长不够、性别拼错、路径不存在
    status     按顺序报告七步的状态，并直接给出下一步该敲的命令
    run        自动执行可自动化的步骤（幂等：已完成则跳过，除非 --force）
    merge      把 3 位标注者各自导出的 annotations-*.json 合并、去重、校验

为什么 preflight 值钱
--------------------
标注界面按**文件名**匹配音频（`basename`）。两条录音重名就会串音，而这种
错误在标注完成后才暴露，返工成本是重录 + 重标。preflight 把这条约束变成
机器检查。同理：单条录音短于 8 s 永远过不了"10 s 窗内 ≥8 s 发声"的受理门槛，
留着它只会拉低受理率并制造一条不可评 query。

边界
----
本工具不判断证据是否成立，也不改 `benchmark.py` 的任何口径：真值仍由
`adjudicate` 的多数裁决产生，G1 仍由 `run_eval` 计算。它只负责顺序、幂等和
"缺什么"的报错。

Usage:
    python tools/benchmark_run.py init --dir material/benchmark \\
        --library material/m4singer_3take/lib \\
        --catalog material/m4singer20_3take/catalog/catalog.json
    python tools/benchmark_run.py preflight --dir material/benchmark
    python tools/benchmark_run.py status --dir material/benchmark
    python tools/benchmark_run.py run --dir material/benchmark --steps sheet
    python tools/benchmark_run.py merge --dir material/benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TOOL = REPO_ROOT / "tools" / "benchmark.py"

GENDERS = ("female", "male")
#: 输入契约上限：超过会被 `truncate_to_input_contract` 截断，不算错误。
INPUT_CONTRACT_SEC = 60.0
#: 受理门槛：10 s 窗内需 ≥8 s 有效发声，短于 8 s 的录音不可能通过。
MIN_USABLE_SEC = 8.0
DEFAULT_RATERS = 3
DEFAULT_KAPPA = 0.6


# --------------------------------------------------------------------------
# 产物与步骤
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """链路中的一步。``human`` 描述该步之前必须由人完成的事（None 表示全自动）。"""

    name: str
    needs: tuple[str, ...]
    makes: tuple[str, ...]
    human: str | None = None


ARTIFACTS = {
    "diagnosis": "diagnosis.json",
    "samples": "samples/samples.json",
    "sheet": "sheet.json",
    "annotations": "annotations.json",
    "adjudicated": "adjudicated.json",
    "eval_set": "frozen/eval_set.json",
    "manifest": "frozen/MANIFEST.json",
}

STEPS: tuple[Step, ...] = (
    Step("diagnose", ("catalog",), ("diagnosis",)),
    Step("samples", ("catalog",), ("samples",)),
    Step(
        "sheet",
        ("specs", "library"),
        ("sheet",),
        human="把真人录音放进 audio/ 并登记到 specs.json",
    ),
    Step(
        "merge",
        ("sheet",),
        ("annotations",),
        human=f"{DEFAULT_RATERS} 位标注者各导出 annotations-<id>.json 到 annotations/",
    ),
    Step("adjudicate", ("sheet", "annotations"), ("adjudicated",)),
    Step("freeze", ("sheet", "adjudicated", "library"), ("eval_set", "manifest")),
    Step("evaluate", ("eval_set", "library"), ()),
)


def _emit(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
def load_config(root: Path) -> dict:
    path = root / "config.json"
    config = _read_json(path) if path.is_file() else {}
    config.setdefault("encoder", "hubert")
    config.setdefault("seed", 0)
    config.setdefault("k", 8)
    config.setdefault("kappa_threshold", DEFAULT_KAPPA)
    config.setdefault("raters", DEFAULT_RATERS)
    config.setdefault("specs", "specs.json")
    config.setdefault("samples_secs", 10.0)
    config.setdefault("adjudicated", "adjudicated.json")
    for key in ("library", "catalog"):
        if config.get(key):
            config[key] = str((root / config[key]).resolve()) if not Path(
                config[key]
            ).is_absolute() else config[key]
    return config


def _artifact(root: Path, name: str, config: dict | None = None) -> Path:
    if name == "adjudicated" and config and config.get("adjudicated"):
        raw = Path(config["adjudicated"])
        return raw if raw.is_absolute() else root / raw
    if name in ARTIFACTS:
        return root / ARTIFACTS[name]
    if name == "specs":
        return root / "specs.json"
    if name == "library":
        raise KeyError("library is configured, not an artifact")
    raise KeyError(name)


def _annotation_exports(root: Path) -> list[Path]:
    folder = root / "annotations"
    return sorted(folder.glob("*.json")) if folder.is_dir() else []


def _step_state(root: Path, step: Step, config: dict) -> dict:
    made = [_artifact(root, name, config) for name in step.makes]
    needs: list[str] = []
    missing: list[str] = []
    if step.name == "merge" and not _annotation_exports(root):
        missing.append(f"{root / 'annotations'}/*.json（0 份标注导出）")
    for name in step.needs:
        if name == "library":
            raw = config.get("library")
            if not raw:
                missing.append("library (config.json)")
            elif not Path(raw).is_dir():
                missing.append(f"library {raw}")
            continue
        if name == "catalog":
            raw = config.get("catalog")
            if not raw:
                missing.append("catalog (config.json)")
            elif not Path(raw).is_file():
                missing.append(f"catalog {raw}")
            continue
        path = _artifact(root, name, config)
        needs.append(str(path))
        if not path.exists():
            missing.append(str(path))
        elif name == "specs":
            # 文件存在但没有任何录音，等于还没开始采集；不能算 ready，
            # 否则 sheet 会"成功"产出一份 0 条的清单。
            try:
                if not _read_json(path):
                    missing.append(f"{path}（0 条录音）")
            except json.JSONDecodeError:
                missing.append(f"{path}（JSON 解析失败）")
    done = bool(made) and all(path.exists() for path in made)
    # 只有"缺的正是人该给的东西"才算 waiting_for_human；缺库/缺目录是配置问题。
    human_inputs = {"sheet": str(root / "specs.json"), "merge": str(root / "annotations")}
    wanted = human_inputs.get(step.name)
    human_blocked = bool(wanted) and any(item.startswith(wanted) for item in missing)
    if done:
        state = "done"
    elif human_blocked:
        state = "waiting_for_human"
    elif missing:
        state = "blocked"
    else:
        state = "ready"
    return {
        "step": step.name,
        "state": state,
        "human": step.human,
        "missing": missing,
        "outputs": [str(path) for path in made],
    }


def _command_for(step: Step, root: Path, config: dict) -> list[str] | None:
    """返回 ``benchmark.py`` 的命令行；``merge`` 由本工具内部完成，返回 None。"""
    library = config.get("library")
    catalog = config.get("catalog")
    encoder = config["encoder"]
    if step.name == "diagnose":
        return [
            sys.executable, str(BENCHMARK_TOOL), "diagnose",
            "--catalog", catalog, "--out", str(root / "diagnosis.json"),
        ]
    if step.name == "samples":
        return [
            sys.executable, str(BENCHMARK_TOOL), "samples",
            "--catalog", catalog, "--out-dir", str(root / "samples"),
            "--secs", str(config["samples_secs"]),
        ]
    if step.name == "sheet":
        return [
            sys.executable, str(BENCHMARK_TOOL), "sheet",
            "--specs", str(root / config["specs"]), "--specs-dir", str(root),
            "--library", library, "--encoder", encoder,
            "--out", str(root / "sheet.json"),
            "--provenance", str(root / "provenance.json"),
            "--seed", str(config["seed"]), "--k", str(config["k"]),
        ]
    if step.name == "adjudicate":
        return [
            sys.executable, str(BENCHMARK_TOOL), "adjudicate",
            "--annotations", str(root / "annotations.json"),
            "--sheet", str(root / "sheet.json"),
            "--out", str(root / "adjudicated.json"),
            "--kappa-threshold", str(config["kappa_threshold"]),
        ]
    if step.name == "freeze":
        return [
            sys.executable, str(BENCHMARK_TOOL), "freeze",
            "--sheet", str(root / "sheet.json"),
            "--adjudicated", str(_artifact(root, "adjudicated", config)),
            "--out", str(root / "frozen"), "--library", library,
            "--encoder", encoder,
            "--kappa-threshold", str(config["kappa_threshold"]),
        ]
    if step.name == "evaluate":
        return [
            sys.executable, str(BENCHMARK_TOOL), "evaluate",
            "--eval-set", str(root / "frozen/eval_set.json"),
            "--library", library, "--encoder", encoder,
        ]
    return None


# --------------------------------------------------------------------------
# specs 校验（纯函数，便于测试）
# --------------------------------------------------------------------------
def _duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return info.frames / info.samplerate if info.samplerate else None
    except Exception:
        return None


def validate_specs(
    specs, specs_dir: Path, k: int = 8
) -> tuple[list[dict], list[dict], dict]:
    """返回 ``(errors, warnings, stats)``。errors 非空即拒绝进入录音/清单阶段。"""
    errors: list[dict] = []
    warnings: list[dict] = []
    if not isinstance(specs, list):
        errors.append({"code": "not_a_list", "message": "specs.json 必须是 JSON 数组"})
        return errors, warnings, {}

    seen_ids: dict[str, int] = {}
    seen_names: dict[str, str] = {}
    subjects: set[str] = set()
    devices: set[str] = set()
    genders: dict[str, int] = {}
    durations: dict[str, float] = {}

    for index, item in enumerate(specs):
        where = f"specs[{index}]"
        if not isinstance(item, dict):
            errors.append({"code": "not_an_object", "message": f"{where} 不是对象"})
            continue
        query_id = str(item.get("query_id") or "").strip()
        raw_path = str(item.get("path") or "").strip()
        gender = str(item.get("gender") or "").strip()

        if not query_id:
            errors.append({"code": "missing_query_id", "message": f"{where} 缺 query_id"})
        elif query_id in seen_ids:
            errors.append(
                {
                    "code": "duplicate_query_id",
                    "message": f"query_id {query_id!r} 重复（{seen_ids[query_id]} 与 {where}）",
                }
            )
        else:
            seen_ids[query_id] = index

        if gender not in GENDERS:
            errors.append(
                {
                    "code": "bad_gender",
                    "message": f"{query_id or where} 的 gender={gender!r}，只能是 {GENDERS}",
                }
            )
        else:
            genders[gender] = genders.get(gender, 0) + 1

        if not raw_path:
            errors.append({"code": "missing_path", "message": f"{query_id or where} 缺 path"})
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = (specs_dir / path).resolve()

        # 界面按文件名匹配音频：重名即串音。
        basename = path.name.lower()
        if basename in seen_names:
            errors.append(
                {
                    "code": "duplicate_filename",
                    "message": (
                        f"{query_id} 与 {seen_names[basename]} 的音频文件名同为 "
                        f"{path.name!r}；标注界面按文件名匹配，必须唯一"
                    ),
                }
            )
        else:
            seen_names[basename] = query_id or where

        if not path.is_file():
            errors.append(
                {"code": "missing_audio", "message": f"{query_id} 的文件不存在：{path}"}
            )
            continue
        seconds = _duration(path)
        if seconds is None:
            errors.append(
                {"code": "unreadable_audio", "message": f"{query_id} 无法解码：{path}"}
            )
        else:
            durations[query_id] = round(seconds, 2)
            if seconds < MIN_USABLE_SEC:
                errors.append(
                    {
                        "code": "too_short",
                        "message": (
                            f"{query_id} 只有 {seconds:.1f}s；10 s 窗内需 ≥{MIN_USABLE_SEC:.0f}s "
                            f"有效发声，这条永远过不了受理门槛"
                        ),
                    }
                )
            elif seconds > INPUT_CONTRACT_SEC:
                warnings.append(
                    {
                        "code": "over_contract",
                        "message": (
                            f"{query_id} 有 {seconds:.1f}s，超过 {INPUT_CONTRACT_SEC:.0f}s "
                            f"输入契约，评估时会被截断（不影响合法性，但别指望用全长）"
                        ),
                    }
                )

        subject = str(item.get("subject_id") or "").strip()
        device = str(item.get("device_id") or "").strip()
        if subject:
            subjects.add(subject)
        else:
            warnings.append(
                {
                    "code": "missing_subject_id",
                    "message": f"{query_id} 缺 subject_id，G3 跨设备稳定性无法按人聚合",
                }
            )
        if device:
            devices.add(device)

    if specs:
        if len(devices) < 2:
            warnings.append(
                {
                    "code": "single_device",
                    "message": "只有 1 个 device_id；G3 要求同一人至少 2 种设备/房间",
                }
            )
        if len(subjects) < 10:
            warnings.append(
                {
                    "code": "pilot_scale",
                    "message": f"只有 {len(subjects)} 位真人；10–15 人属 pilot，不能对外称准确率",
                }
            )
        elif len(subjects) < 60:
            warnings.append(
                {
                    "code": "below_formal_scale",
                    "message": f"{len(subjects)} 位；正式基准建议 60–100 人，否则 CI 半宽约 ±0.10",
                }
            )
        for gender in GENDERS:
            if gender not in genders:
                warnings.append(
                    {
                        "code": "empty_gender",
                        "message": f"{gender} 没有任何录音，该性别池无法评估",
                    }
                )

    stats = {
        "n_specs": len(specs),
        "n_subjects": len(subjects),
        "n_devices": len(devices),
        "by_gender": genders,
        "durations_sec": durations,
    }
    return errors, warnings, stats


def preflight(root: Path, config: dict) -> dict:
    specs_path = root / config["specs"]
    if not specs_path.is_file():
        return {
            "status": "error",
            "errors": [{"code": "missing_specs", "message": f"找不到 {specs_path}"}],
            "warnings": [],
            "next": f"按 docs/benchmark/02-capture-checklist.md 录音，并登记到 {specs_path}",
        }
    specs = _read_json(specs_path)
    errors, warnings, stats = validate_specs(specs, root, k=int(config["k"]))
    if not specs:
        errors.append(
            {
                "code": "empty_specs",
                "message": f"{specs_path} 里没有任何录音；先按采集规范录入再跑",
            }
        )

    library = config.get("library")
    if library and Path(library).is_dir():
        manifest_path = Path(library) / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_json(manifest_path)
            expected = "HUBERT_BASE" if config["encoder"] == "hubert" else "HashEncoder"
            if manifest.get("model_id") != expected:
                errors.append(
                    {
                        "code": "encoder_mismatch",
                        "message": (
                            f"库的 model_id 是 {manifest.get('model_id')!r}，与 "
                            f"--encoder {config['encoder']} 不符，sheet 会拒绝运行"
                        ),
                    }
                )
            pools = {"female": 0, "male": 0}
            for singer in manifest.get("singers", []):
                gender = singer.get("gender")
                if gender in pools:
                    pools[gender] += 1
            for gender, size in sorted(pools.items()):
                if size < int(config["k"]):
                    warnings.append(
                        {
                            "code": "library_pool_too_small",
                            "message": (
                                f"库里 {gender} 只有 {size} 位，少于 k={config['k']}；"
                                f"候选清单会不足 {config['k']} 位"
                            ),
                        }
                    )
        else:
            warnings.append(
                {"code": "library_manifest_missing", "message": f"{manifest_path} 不存在"}
            )

    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "next": (
            "修正上述错误后重跑 preflight"
            if errors
            else "python tools/benchmark_run.py run --steps diagnose,samples,sheet"
        ),
    }


# --------------------------------------------------------------------------
# 标注合并
# --------------------------------------------------------------------------
def merge_annotations(
    root: Path, raters: int
) -> tuple[dict, list[dict], list[dict]]:
    """合并 ``annotations/*.json``；返回 ``(payload, errors, warnings)``。"""
    folder = root / "annotations"
    files = sorted(folder.glob("*.json")) if folder.is_dir() else []
    errors: list[dict] = []
    warnings: list[dict] = []
    if not files:
        return {}, [{"code": "no_annotation_files", "message": f"{folder} 里没有 *.json"}], []

    sheet_path = root / "sheet.json"
    allowed: dict[str, set[str]] = {}
    if sheet_path.is_file():
        allowed = {
            item["query_id"]: set(item["candidates"]) for item in _read_json(sheet_path)
        }

    merged: dict[tuple[str, str], dict] = {}
    per_file: dict[str, int] = {}
    for path in files:
        try:
            raw = _read_json(path)
        except json.JSONDecodeError as exc:
            errors.append({"code": "bad_json", "message": f"{path.name}: {exc}"})
            continue
        if not isinstance(raw, list):
            errors.append({"code": "not_a_list", "message": f"{path.name} 不是 JSON 数组"})
            continue
        per_file[path.name] = len(raw)
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                errors.append(
                    {"code": "bad_entry", "message": f"{path.name}[{index}] 不是对象"}
                )
                continue
            query_id = str(item.get("query_id") or "")
            annotator = str(item.get("annotator_id") or "")
            picked = item.get("picked", [])
            if not query_id or not annotator:
                errors.append(
                    {
                        "code": "missing_field",
                        "message": f"{path.name}[{index}] 缺 query_id 或 annotator_id",
                    }
                )
                continue
            if not isinstance(picked, list):
                errors.append(
                    {"code": "picked_not_list", "message": f"{path.name}[{index}] picked 不是数组"}
                )
                continue
            key = (query_id, annotator)
            if key in merged:
                errors.append(
                    {
                        "code": "duplicate_annotation",
                        "message": (
                            f"{annotator} 对 {query_id} 有两条标注（多半是同一人导出了两次）；"
                            f"同一标注者重复计入会变成双倍投票"
                        ),
                    }
                )
                continue
            stray = sorted(set(picked) - allowed[query_id]) if query_id in allowed else []
            if stray:
                errors.append(
                    {
                        "code": "pick_outside_candidates",
                        "message": f"{annotator} 在 {query_id} 选了名单外的 {stray}",
                    }
                )
                continue
            if query_id not in allowed and allowed:
                warnings.append(
                    {"code": "unknown_query", "message": f"{query_id} 不在 sheet.json 里"}
                )
            merged[key] = {
                "query_id": query_id,
                "annotator_id": annotator,
                "picked": list(picked),
            }

    counts: dict[str, set[str]] = {}
    for query_id, annotator in merged:
        counts.setdefault(query_id, set()).add(annotator)
    for query_id, annotators in sorted(counts.items()):
        if len(annotators) < raters:
            warnings.append(
                {
                    "code": "thin_annotation",
                    "message": f"{query_id} 只有 {len(annotators)} 位标注者（目标 {raters}）",
                }
            )
    if allowed:
        for query_id in sorted(set(allowed) - set(counts)):
            warnings.append(
                {"code": "unannotated_query", "message": f"{query_id} 没有任何标注"}
            )

    payload = {
        "queries": len(counts),
        "raters": len({annotator for _, annotator in merged}),
        "files": per_file,
        "annotations": [merged[key] for key in sorted(merged)],
    }
    return payload, errors, warnings


# --------------------------------------------------------------------------
# 子命令
# --------------------------------------------------------------------------
def cmd_init(args) -> int:
    root: Path = args.dir
    for name in ("", "audio", "samples", "annotations", "reviews"):
        (root / name).mkdir(parents=True, exist_ok=True)

    specs_path = root / "specs.json"
    created = []
    if not specs_path.exists():
        _write_json(specs_path, [])
        created.append(str(specs_path))
    template = root / "specs.template.json"
    if not template.exists():
        _write_json(
            template,
            [
                {
                    "query_id": "p001-a",
                    "path": "audio/p001-a.wav",
                    "gender": "female",
                    "subject_id": "p001",
                    "device_id": "phone-a",
                }
            ],
        )
        created.append(str(template))

    config_path = root / "config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    if args.library:
        config["library"] = str(Path(args.library).resolve())
    if args.catalog:
        config["catalog"] = str(Path(args.catalog).resolve())
    config.setdefault("encoder", args.encoder)
    config.setdefault("seed", 0)
    config.setdefault("k", 8)
    config.setdefault("kappa_threshold", DEFAULT_KAPPA)
    config.setdefault("raters", DEFAULT_RATERS)
    config.setdefault("specs", "specs.json")
    config.setdefault("samples_secs", 10.0)
    config.setdefault("adjudicated", "adjudicated.json")
    _write_json(config_path, config)

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(README_TEMPLATE, encoding="utf-8")
        created.append(str(readme))

    _emit(
        {
            "status": "ok",
            "command": "init",
            "dir": str(root),
            "created": created,
            "config": str(config_path),
            "next": (
                "按 docs/benchmark/02-capture-checklist.md 录音 -> 填 specs.json -> "
                "python tools/benchmark_run.py preflight"
            ),
        }
    )
    return 0


def cmd_preflight(args) -> int:
    root: Path = args.dir
    config = load_config(root)
    payload = preflight(root, config)
    payload["command"] = "preflight"
    _emit(payload)
    return 0 if payload["status"] == "ok" else 2


def cmd_status(args) -> int:
    root: Path = args.dir
    config = load_config(root)
    states = [_step_state(root, step, config) for step in STEPS]
    pending = next((s for s in states if s["state"] != "done"), None)
    payload = {
        "status": "ok",
        "command": "status",
        "dir": str(root),
        "library": config.get("library"),
        "catalog": config.get("catalog"),
        "steps": states,
        "next_step": pending["step"] if pending else None,
        "next_command": (
            _suggest(pending, root)
            if pending
            else "全部完成；把 frozen/MANIFEST.json 与 evaluate 输出一起归档"
        ),
    }
    _emit(payload)
    return 0


def _suggest(state: dict, root: Path) -> str:
    """下一条命令；若卡在人手上，就先说人该做什么，别给一条跑不通的命令。"""
    if state["state"] == "waiting_for_human":
        return f"先完成：{state['human']}；然后 python tools/benchmark_run.py preflight --dir {root}"
    if state["step"] == "merge":
        return f"python tools/benchmark_run.py merge --dir {root}"
    return f"python tools/benchmark_run.py run --dir {root} --steps {state['step']}"


def _pairwise_commands(root: Path) -> list[str]:
    """kappa 未达标时的退路：对争议 query 做两两比较重标。"""
    return [
        (
            f"python tools/benchmark.py pairwise-sheet --adjudicated {root / 'adjudicated.json'} "
            f"--sheet {root / 'sheet.json'} --out {root / 'pairs.json'} --top 4"
        ),
        (
            f"python tools/benchmark.py pairwise-adjudicate --pairs {root / 'pairs.json'} "
            f"--annotations {root / 'pair-annotations.json'} "
            f"--out {root / 'adjudicated-pairwise.json'}"
        ),
        (
            f"python tools/benchmark_run.py run --dir {root} --steps freeze "
            f"--adjudicated {root / 'adjudicated-pairwise.json'}"
        ),
    ]


def cmd_run(args) -> int:
    root: Path = args.dir
    config = load_config(root)
    if args.adjudicated:
        config["adjudicated"] = str(args.adjudicated)
    wanted = [name.strip() for name in args.steps.split(",") if name.strip()]
    if wanted == ["all"]:
        wanted = [step.name for step in STEPS]
    known = [step.name for step in STEPS]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        _emit(
            {
                "status": "error",
                "command": "run",
                "message": f"未知步骤 {unknown}",
                "known": known,
            }
        )
        return 2

    executed: list[dict] = []
    low_kappa = False
    for name in known:  # 始终按链路顺序执行，避免乱序产出半成品
        if name not in wanted:
            continue
        step = next(s for s in STEPS if s.name == name)
        state = _step_state(root, step, config)
        if state["state"] == "done" and not args.force:
            executed.append({"step": name, "result": "skipped_done"})
            continue
        if state["missing"]:
            executed.append(
                {
                    "step": name,
                    "result": "blocked",
                    "missing": state["missing"],
                    "need": state["human"],
                }
            )
            break

        if name == "merge":
            payload, errors, warnings = merge_annotations(root, int(config["raters"]))
            if errors:
                executed.append(
                    {"step": name, "result": "error", "errors": errors, "warnings": warnings}
                )
                break
            _write_json(root / "annotations.json", payload["annotations"])
            executed.append(
                {
                    "step": name,
                    "result": "ok",
                    "queries": payload["queries"],
                    "raters": payload["raters"],
                    "warnings": warnings,
                }
            )
            continue

        command = _command_for(step, root, config)
        if command is None:
            executed.append({"step": name, "result": "no_command"})
            continue
        completed = subprocess.run(
            command, cwd=str(REPO_ROOT), capture_output=True, text=True
        )
        if completed.returncode != 0:
            executed.append(
                {
                    "step": name,
                    "result": "error",
                    "exit": completed.returncode,
                    "stderr": (completed.stderr or "").strip()[-2000:],
                    "stdout": (completed.stdout or "").strip()[-2000:],
                }
            )
            break
        try:
            detail = json.loads((completed.stdout or "").strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            detail = {"raw": (completed.stdout or "").strip()[-1000:]}
        executed.append({"step": name, "result": "ok", "detail": detail})
        # kappa 不达标是"问法不稳"，不是"可以继续"。默认在这里停住并给出
        # pairwise 重标路径；要继续必须显式 --allow-low-kappa，留下审计痕迹。
        if name == "adjudicate" and not args.allow_low_kappa:
            kappa = detail.get("kappa_mean")
            threshold = float(config["kappa_threshold"])
            if isinstance(kappa, (int, float)) and kappa < threshold:
                executed[-1]["result"] = "needs_pairwise"
                executed[-1]["kappa"] = kappa
                executed[-1]["threshold"] = threshold
                low_kappa = True
                break

    failed = any(item["result"] in {"error", "blocked"} for item in executed)
    states = [_step_state(root, step, config) for step in STEPS]
    pending = next((s for s in states if s["state"] != "done"), None)
    if low_kappa:
        payload = {
            "status": "needs_pairwise",
            "command": "run",
            "executed": executed,
            "reason": (
                f"kappa {executed[-1].get('kappa')} 低于门槛 {config['kappa_threshold']}："
                "8 选 1 的问法不稳，按设计 6.2 改用两两比较重标，不要调低门槛"
            ),
            "pairwise_commands": _pairwise_commands(root),
        }
        _emit(payload)
        return 2
    _emit(
        {
            "status": "error" if failed else "ok",
            "command": "run",
            "executed": executed,
            "next_step": pending["step"] if pending else None,
            "next_command": _suggest(pending, root) if pending else None,
        }
    )
    return 0 if not failed else 2


def cmd_merge(args) -> int:
    root: Path = args.dir
    config = load_config(root)
    payload, errors, warnings = merge_annotations(root, int(config["raters"]))
    if errors:
        _emit(
            {
                "status": "error",
                "command": "merge",
                "errors": errors,
                "warnings": warnings,
            }
        )
        return 2
    _write_json(root / "annotations.json", payload["annotations"])
    _emit(
        {
            "status": "ok",
            "command": "merge",
            "out": str(root / "annotations.json"),
            "queries": payload["queries"],
            "raters": payload["raters"],
            "files": payload["files"],
            "warnings": warnings,
            "next": f"python tools/benchmark_run.py run --dir {root} --steps adjudicate",
        }
    )
    return 0


README_TEMPLATE = """\
# 真人盲听基准集（工作目录）

这里的音频与标注都不入库（`material/` 已在 `.gitignore`）。
方法、采集规范与判据见 `docs/benchmark/`：

- `01-consent-form.md` 同意书（评估用途 / 训练用途分开勾选）
- `02-capture-checklist.md` 录音规范（3 段、不同设备、≥8 s 有效发声、≤60 s）
- `03-annotation-guide.md` 标注说明
- `04-recruitment.md` 招募话术与筛选
- `annotation-ui.html` 离线标注界面（按文件名匹配音频，名字必须唯一）

目录：

    audio/        真人录音（query_id.wav）
    specs.json    录音登记表（preflight 会校验）
    samples/      候选歌手试听样本（工具生成）
    annotations/  各标注者导出的 annotations-<id>.json，原始文件，不改
    reviews/      争议 query 的二次复核记录
    frozen/       冻结后的 eval_set.json + MANIFEST.json（唯一可信数字来源）

跑法（任何一步都可以单独重跑，已完成会跳过）：

    python tools/benchmark_run.py preflight --dir <本目录>
    python tools/benchmark_run.py run --dir <本目录> --steps diagnose,samples,sheet
    # 真人：录音、填 specs、3 人盲听、导出到 annotations/
    python tools/benchmark_run.py status --dir <本目录>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the working-directory skeleton")
    p_init.add_argument("--dir", type=Path, default=Path("material/benchmark"))
    p_init.add_argument("--library", type=Path)
    p_init.add_argument("--catalog", type=Path)
    p_init.add_argument("--encoder", choices=("hubert", "hash"), default="hubert")
    p_init.set_defaults(func=cmd_init)

    p_pre = sub.add_parser("preflight", help="validate specs.json before recording")
    p_pre.add_argument("--dir", type=Path, default=Path("material/benchmark"))
    p_pre.set_defaults(func=cmd_preflight)

    p_status = sub.add_parser("status", help="report pipeline state and the next command")
    p_status.add_argument("--dir", type=Path, default=Path("material/benchmark"))
    p_status.set_defaults(func=cmd_status)

    p_run = sub.add_parser("run", help="execute automated steps in pipeline order")
    p_run.add_argument("--dir", type=Path, default=Path("material/benchmark"))
    p_run.add_argument("--steps", default="all", help="comma separated, or 'all'")
    p_run.add_argument("--force", action="store_true", help="redo steps already done")
    p_run.add_argument(
        "--adjudicated",
        type=Path,
        help="freeze this adjudication file instead of adjudicated.json (pairwise path)",
    )
    p_run.add_argument(
        "--allow-low-kappa",
        action="store_true",
        help="continue past a sub-threshold kappa (recorded in MANIFEST; needs a reason)",
    )
    p_run.set_defaults(func=cmd_run)

    p_merge = sub.add_parser("merge", help="merge per-rater annotation exports")
    p_merge.add_argument("--dir", type=Path, default=Path("material/benchmark"))
    p_merge.set_defaults(func=cmd_merge)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
