# VoiceMatch 交接说明

> 面向接手这套代码的另一个 AI。读完这一份应当能直接继续工作，不必重读历史对话。
> 最后更新：2026-11-19

---

## 1. 一句话

一个「录一段清唱 → 告诉你最像哪位歌手」的产品，走**准确性优先**路线（不是传播优先）。代码是一个可离线运行的 Python 包 `voicematch`，已实现从音频到排序的完整链路，**尚未接入真实数据**。

## 2. 状态快照

| 项 | 值 |
|---|---|
| 仓库 | `C:\Users\Lenovo\Desktop\测试`（git，本机） |
| 分支 | `feat/voicematch-core-engine`，HEAD `af30a41`，共 35 个提交 |
| 基线 | `main` = `2ab734f`（只有 docs） |
| 测试 | `python -m pytest -q` → **293 passed, 1 skipped**（~130s） |
| 规模 | `src/` 32 个模块 + `tests/` ~25 个文件，约 400 KB |
| 环境 | Windows，Python 3.13.14（`C:\Python\python.exe`），`pip install -e .` 已装 |
| 依赖 | 全部精确 pin（见 `pyproject.toml`）；`ml` extra = `torch==2.15.0` + `torchaudio==2.11.1` |

分支**尚未合并**，等用户选择合并方式（本地合并 / PR / 保留）。

## 3. 为什么是「准」优先

上一轮讨论里用户明确：第一版要验证的是「准不准」，不是「能不能刷屏」。所以整个设计的判据是**可度量、可复现、可证伪**：

- 「像不像」的权威是人耳，因此核心不是余弦相似度，而是**校准层**。
- 没有盲听基准集，「准」只是宣称。基准集（60–100 人真人清唱 + 3 人多数标注）是整个项目最长的一条链，也是唯一还没启动的部分。

## 4. 架构

```
音频 → 解码/重采样(24k mono) → 预处理(LUFS −23, 限幅) ──┐
                                                        ├→ 质量门限(拒答) → 自动选窗(10s)
                                                        └→ 有效发声区间(能量基，唯一定义)
选窗 → 四维声学特征(pitch/timbre/resonance/style) + 冻结 HuBERT 嵌入(256d) → 歌手原型
                                                                              ↓
                    Bradley-Terry 校准(cos + 四维差 → 感知相似度) → 同性别候选池排序 + 归因
                                                                              ↓
                                                              CLI 一行 JSON / 评估 harness
```

模块分区：

- `audio/` — `io`（解码）、`preprocess`（对称管线）、`quality`（拒答）、`windows`（选窗）、`pitch`（唯一的 pYIN 入口）
- `features/` — 四个维度 + `extract`（汇总）+ `percentile`（同性别分位 0–100）
- `embed/` — `encoder`（HuBERT 冻结 / `HashEncoder` 测试替身）、`pooling`、`prototype`
- `library/` — `schema`、`build`、`store`（版本化 manifest + npy）
- `calibrate/` — `pairs`、`model`（逻辑回归）、`score`（排序）
- `eval/` — `metrics`、`dataset`、`run`、`candidates`
- `cli.py` — `python -m voicematch.cli compare <wav> --library <dir> --gender {male,female} [--encoder hash|hubert] [--separate]`

## 5. 必须守住的不变量（违反即产品失真）

1. **两侧同构**：歌手库与用户查询走同一个 `preprocess`、同一组参数，`pipeline_signature` 有 CI 契约测试。一侧分离人声一侧不分离 = 比的是制作而不是嗓音。
2. **「有效发声」只有一个定义**：`audio/quality.speech_intervals`（能量基）。门限和选窗准入都用它；pYIN 只用于打分与拒答的 pitch 覆盖度。
3. **口径成对出现**：`top1/top3` 只在**被受理且有答案**的查询上算，且必须与 `acceptance_rate` 同时报告（受理率下限 85%）。「都不像」不算 miss，单独计 `n_none_like`。
4. **`nan` 不是 0**：缺失的分位/特征保持 `nan`，绝不填 0 或 100（会凭空造出「最像」）；`-inf`/`nan` 不得进入 JSON（序列化边界转 `null`）。
5. **可复现**：结果里带 `model_id` / `library_version` / `calibration_version`；库版本由素材哈希 + 管线版本 + 模型身份结构性地决定，不靠人记得改。`PIPELINE_VERSION` 现为 `pipeline-v2`。
6. **校准不可越权**：`CalibrationModel.is_calibrated` 为 False 时，分数只能用于排序，**不得**当概率百分比展示。

## 6. 历史上做过的重要裁定（推翻了计划原文的地方）

这些是上一轮与用户/复审博弈的结论，**不要凭直觉改回去**：

| 编号 | 裁定 | 原因 |
|---|---|---|
| R2 | `pair_features` 不接收原型向量 | 256 维原型给不出四维差 |
| R3 | 池化用固定 128+128 切片，不用随机投影 | 随机投影会让每次重建的嵌入不同 |
| R4 | 分位 = `idx/(n-1)*100` | 原公式把最小值映射到 1/n 而不是 0 |
| R9 | 「都不像」是答案不是 miss | 否则把库覆盖度问题算成排序错误 |
| R15 | 窗长 = `min(10s, 总时长)` | 否则 9 秒的合规录音产出 0 个窗、永远进不了打分 |
| R17 | 有效发声只保留能量基一种定义 | 门限用能量、选窗用音高，才会出现「门限接受但选窗为空」 |
| R18 | `slur_ratio` 按 semitone/秒 | 原 0.5st/帧 ≈ 47st/s，测的是滑音不是连音，该维度恒为 0 |
| R19 | `LibraryIndex` 值相等 + 不可哈希 | frozen dataclass 生成的 `__eq__` 在 ndarray 上直接抛异常 |
| R20 | 校准纯 z-score | 两级缩放会互相抵消、无法从模型复现 |
| R21 | 配对与 AUC 严格按歌手分折 | 「anchor 歌手留出」会让配对另一侧漏进训练集，抬高 AUC |
| R22 | 退化模型统一规范元组 + `is_calibrated` | 否则「没跑校准」与「跑了但测不出」无法区分 |
| R24 | 库记录 `model_id`，CLI 不匹配即拒跑 | 用 Hash 查询比 HuBERT 原型是无意义向量空间拼接 |
| R25 | v1 只接受清唱，不做伴奏检测 | 检测不可靠；改为输出 `assumed: "a_cappella"` 显式声明 |
| R26 | 依赖精确 pin + manifest 记录版本 | 换 numpy/librosa 会静默改变所有已存向量 |
| R27 | G3 指标加 `g3_measurable` 标记 | 缺字段时返回 `nan` 而非假装测过 |

完整台账（R1–R28 + 每个任务的复审记录）在 `.superpowers/sdd/2026-11-18-voicematch-core-engine/progress.md`。

## 7. 已知缺口（按优先级）

1. **没有真实数据**。所有指标只在合成夹具上跑过。没有真人清唱、没有盲听标注集，`top1/top3/auc` 目前不代表任何产品结论。
2. **HuBERT 只在一组版本错配上验证过**：`torchaudio==2.11.1` 配 `torch==2.15.0`（上游通常锁步发布）。HuBERT 测试真跑过、通过，但正式建库前建议换成官方锁步版本再验一次。
3. **阶段 2 报告页未做**（设计里明确另立计划）：四维分位图、Top-3 展示、置信度与拒答文案、可截图报告页。`features/percentile.py` 已实现但**没有生产调用方**，等阶段 2 接。
4. **G3（同人三设备稳定性）只有骨架**：schema 支持 `subject_id/device_id`，`run_eval` 在字段齐全时算 `top3_overlap`，但没有采集。
5. **伴奏输入被拒而非分离**：`--separate` 会结构化拒绝。v2 若要支持跟唱，需要两侧同时启用分离并重建库。
6. **若干已记录的次要问题**（约 40 条）在各任务的 `task-*-report.md` 与台账里，均已分诊，无阻塞项。

## 8. 怎么跑

```bash
cd "C:\Users\Lenovo\Desktop\测试"
python -m pip install -e ".[dev]"        # 若要跑 HuBERT，再加 ".[ml]"
python -m pytest -q                      # 293 passed, 1 skipped

# 需要先有库（目前只有测试夹具构造过）：
#   build_library(entries, encoder) → save_library(index, dir)
python -m voicematch.cli compare query.wav --library lib/ --gender female
```

测试用的 `HashEncoder` 是确定性的替身，**只用于测试**；任何真实结果都必须用 `--encoder hubert`。

## 9. 建议的下一步顺序

1. 用户选合并方式，把分支落进 `main`。
2. 用官方锁步的 torch/torchaudio 重跑一次 HuBERT 测试，锁死这个配对。
3. 建第一批歌手库（10 人，男女各半，studio 为主），跑通 `build_library → save_library → compare`。
4. 启动真人采集与盲听标注（60–100 人 × 3 段，3 人多数，kappa 门槛）——这是最长的一条链，应当与 3 并行。
5. 有了基准集，才谈得上「准不准」；届时 `auc` 与 `top3` 才有意义，再决定是否微调嵌入。
6. 阶段 2 报告页另立计划，不要现在做。

## 10. 待用户决定

- 分支如何合并（本地合并 `main` / 推 PR / 保留）。
- 是否删掉 `.superpowers/sdd/2026-11-18-voicematch-core-engine/`（台账、14 份任务报告、终审与复审 diff 包）。目前保留着。
