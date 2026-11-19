# 约束 R29：校准正对必须跨录音

**状态**：已生效，代码强制（`build.py` / `pairs.py` / `cli.py` + 测试）
**范围**：歌手库构建、校准训练、任何以"校准后分数"对外报告的产物
**关系**：延续 R1–R28 台账；是 R21（singer-disjoint AUC）之外的另一条独立性要求——R21 保证**歌手不泄漏**，R29 保证**录音不泄漏**。二者缺一，AUC 都不能当作泛化证据。

---

## 1. 约束陈述

对任何参与校准训练的歌手库，以下四条必须同时成立：

1. **每歌手至少两份独立录音。** 每位歌手在库内必须有 ≥ 2 份**内容不同**的素材（不同歌曲，或同一首歌的不同场次）。
2. **正对只能跨素材。** 校准的正对（`label=1`）MUST 由同一歌手、**不同 `material_id`** 的两段构成；同一份录音切出的两段 MUST NOT 构成正对。
3. **不满足必须显式暴露。** 构建时若某歌手只有一份可用录音，MUST 发出 `SingleMaterialSingerWarning`，并在 CLI 结果的 `single_material_singers` 中列出；交付/生产构建 MUST 使用 `--strict-pairs`，让它直接失败（`pairing_constraint`），不落盘。
4. **报告必须带状态。** 任何"校准后"指标 MUST 说明该库是否满足第 1 条。构建结果用 `single_material_singers`，查询结果用 `pairing.status`（`cross_recording` / `single_recording` / `unverifiable_legacy` / `empty`）加 `pairing.single_material_singers`；仅凭 `calibrated=false` 不足以说明原因（"歌手太少"和"没有跨录音正对"的修法完全不同）。单素材库通常落到 `calibrated=false` / `auc=nan`，分数只可用于排序；但注意 `calibrated=true` 与 `unverifiable_legacy` 可以同时出现——旧库会被拟合出系数，那条路径同样是泄漏，不得称为概率或"相似度百分比"。**评估类产物 MUST 在 `calibrated=false` 或 `pairing.status=unverifiable_legacy` 时拒绝给出 top1/top3**——回退余弦的 top1 与校准 top1 在数值上无法区分，一旦离开上下文就会被当成后者；只有显式声明 `--cosine-only` 的诊断跑数可出数，并必须带 `metric_kind: cosine_only_diagnostic`。

---

## 2. 实证依据

**结论先行：R29 的正当性来自"训练信号里不能有录音身份"，而不是"校准更准"。** 校准相对余弦是否真有净增益，现有证据不支持下结论（2.3）。

### 2.1 泄漏是实测的（与留出标签无关）

10 位歌手、每人一份 120 s 单素材、旧配对规则（`material_id` 为空，同录音两段算正对）：77 个正对全部来自同一次录音，库内 AUC **1.000**。这不是成绩，是泄漏——模型完美地区分了"这段来自哪次录音"。

该库现在会被评估器直接拒绝（`unverifiable_legacy_library`）：它的 top1 只能反映泄漏，不能作为歌手相似度的证据。这就是"必须能判断"比"必须更好"更重要的原因。

### 2.2 新规则下泄漏消失

同 10 人、改为每人 3 首不同歌（30 份素材、74 段、196 个跨录音正对）：库内 AUC **0.951**，不再饱和。作为对照，同一份录音从 62 s 拉到 120 s 时，纯余弦的 top1/top3 完全不变（0.500 / 0.900）——多出来的窗口被自动选窗吸收，**"加时长"不等于"加录音"**。

### 2.3 修正留出标签后的表现（同池、同留出集）

| 10 人 · 池 5 | 校准 | 纯余弦 |
|---|---|---|
| top1 | **0.875**（7/8） | 0.750（6/8） |
| top3 | **1.000**（8/8） | 0.750（6/8） |
| 真值平均名次 | 1.13 | 1.88 |
| 随机基线 | 0.20 / 0.60 | 同左 |

差 1 条 top1（McNemar 精确检验 p = 1.0）、2 条 top3。**不足以下"校准更准"的结论。**

20 人 · 池 10 的复验更不支撑：校准 top1 0.562 vs 余弦 0.688，top3 0.812 vs 0.688，方向不一致、CI 全跨 0（见 `docs/scale-experiment.md`）。

因此本文只主张 R29 保证的事：**正对不含录音身份**；不主张校准带来精度增益。

> **历史注记（已作废，勿引用）**：本文初稿引用的"校准 0.300 / 余弦 0.500"来自一个受
> 留出标签 bug 影响的评估器（10 条查询中 2 条由另一位歌手的音频拼成），且该库本身是
> `unverifiable_legacy`。该对照无法在现行代码下重跑，也不再作为证据。

---

## 3. 失败机制

同一份录音内的两段共享的东西，远多于"同一位歌手"：

- 同一支话筒、同一间房、同一次增益设置带来的信道特征；
- 同一段旋律、同一处换气、同一段伴奏残留；
- 相邻窗口在时间上连续，嵌入几乎相同。

校准层只有 5 个自由度（嵌入余弦 + 四维特征差），在"歌手身份"与"录音身份"高度共线时，它会优先拟合后者——后者在训练集里是**完全可分**的。于是库内 AUC 饱和到 1.000，而任何新歌都落在这个判据之外。跨录音配对把"录音身份"从正对里剔除，模型才被迫学习歌手本身。

---

## 4. 代码强制点

| 层 | 位置 | 机制 |
|---|---|---|
| 数据模型 | `library/schema.py` `Segment.material_id` | 每段记录素材内容哈希；`""` 表示"每段自成素材"（旧库/合成夹具的兼容语义） |
| 构建 | `library/build.py` | 写入 `material_id`；按**去重后的有效素材数**判定，=1 时发 `SingleMaterialSingerWarning(singer_id=...)` |
| 配对 | `calibrate/pairs.py` | 正对仅跨不同 `material_id`；同素材两段不构成正对，负对规则不变 |
| CLI | `cli.py` `build` | `single_material_singers` 列表（与 `skipped` 分离）；`--strict-pairs` → `pairing_constraint` 错误且不写库 |
| 查询 | `cli.py` `compare` | `pairing.status` + `pairing.single_material_singers`，从已加载索引读取；`unverifiable_legacy` 专门防旧库冒充合规 |
| 评估 | `tools/holdout_eval.py` | 默认路径在 `calibrated=false` 或 `pairing.status="unverifiable_legacy"` 时**拒绝产出任何指标**：stdout 一行 JSON 错误文档（`uncalibrated_library` / `unverifiable_legacy_library`）、exit 2，且在拉取留出音频之前就退出；`--cosine-only` 是唯一出口，报告与逐条查询行都带 `metric_kind: cosine_only_diagnostic` |
| 测试 | `tests/library/test_build.py`、`tests/library/test_build_cli.py`、`tests/eval/test_cli.py`、`tests/calibrate/test_pairs.py`、`tests/tools/test_holdout_gate.py` | 单素材告警、两份不同素材不告警、**两份相同字节仍告警**、第二份无有效窗仍告警、strict 模式拒绝且不落盘、`pairing_status` 四态、compare 三态透出、评估闸门两态与 cosine-only 出口 |

两个容易漏判的边界，代码与测试都按"否"处理：

- **字节相同的两份文件是同一份素材**（内容哈希相同）。复制一个 wav 冒充第二次录音，不会消除告警——这是刻意的。
- **一份没有有效窗口的素材不算数。** 若第二份素材因静音/发声不足被跳过，该歌手仍按单素材处理。

---

## 5. 操作要求（素材侧）

1. 每歌手准备 **≥ 2 首不同歌**，或同一首歌的**不同场次**（不同时间/房间/设备均可，但不能是同一次录音的两个导出）。
2. 每份素材满足管线准入：10 s 窗内 ≥ 8 s 有效发声（设计 5.6）。素材太短会被跳过，等于没准备。
3. 推荐按设计 5.5 做到 **3–5 首/歌手**，且至少 2 首能产出窗口；冗余用于抵抗单份素材被跳过。
4. 不要用"每歌手 60 s"理解要求——**60 s 是一份录音的下限，不是每歌手的总量**。

---

## 6. 自检与验收

```bash
# 1) 构建（研究/冒烟）
python -m voicematch.cli build --catalog catalog.json --out lib/ --encoder hubert
#   判据：single_material_singers == []

# 2) 交付/CI：不满足即失败，且不产生库目录
python -m voicematch.cli build --catalog catalog.json --out lib/ --encoder hubert --strict-pairs

# 3) 直接看配对结构（auc=nan 表示校准不可用，不是"待测"）
python -c "
from voicematch.library.store import load_library
from voicematch.calibrate.pairs import build_pairs
from voicematch.calibrate.model import fit_calibration
idx = load_library('lib')
X, y, g = build_pairs(idx, idx.embeddings)
m = fit_calibration(X, y, g)
print('materials', len({s.material_id for s in idx.segments}), 'pos', int((y == 1).sum()))
print('calibrated', m.is_calibrated, 'auc', m.auc)
"

# 4) 查询侧同样能看到库的状态
python -m voicematch.cli compare query.wav --library lib/ --gender female
#   判据：pairing.status == "cross_recording" 且 single_material_singers == []
```

**一个必须会读的组合**：`calibrated: true` 配上 `pairing.status: "unverifiable_legacy"` —— 说明这是旧库，正对里混着同录音切片，AUC 与"校准已启用"都不可信，**必须重建**。这也是为什么配对状态不能从 `calibrated` 反推。

**判据只有一个：`pos > 0` 且 `calibrated=true` 且 `pairing.status="cross_recording"`。不要用库内 AUC 判断是否满足约束**——AUC 越接近 1.000，越应该怀疑正对里有同录音泄漏。

---

## 7. 无法满足时的合规退路

真实素材暂时凑不齐两份录音时，允许以下退路，但都必须显式标注：

- **以未校准状态交付**：`calibrated=false`，对外只称"余弦相似度排序"，不称概率、不称百分比相似度；报告写明"库未满足 R29，校准未启用"。评估侧不接受这种状态下的"默认路径"跑数——`holdout_eval` 直接拒绝，只允许 `--cosine-only` 的显式诊断，且该诊断的数字同样不得称为准确率。
- **旧库必须重建**：在 R29 之前构建、`material_id` 为空的库，其 AUC 1.000 属于同录音泄漏产物，MUST NOT 作为任何准确性证据；重建后它会正确落到 `calibrated=false`。
- **研究/冒烟可用，产品不放行**：内部验证管线能跑通是合法用途；`--strict-pairs` 是产品线的闸门。

---

## 8. 与其他约束的关系

- **R21（singer-disjoint AUC）**：解决"歌手泄漏"——正负对不能跨折。R29 解决"录音泄漏"——正对不能同素材。两者都满足，AUC 才是泛化证据。
- **R24（库记录编码器身份）**：`material_id` 与 `model_id` 都是随库走的结构性事实，不由操作者记忆。
- **G4（同音频同版本同输出）**：`material_id` 是素材内容哈希，路径无关，不影响版本的可复现性；同一批素材仍产出同一 `library_version`。
- **设计 5.4 / 5.5**：5.4 的"同歌手不同歌曲/段落为正"由本文收紧为"不同录音"，5.5 的"每人 5 首"天然满足。

---

## 9. 变更记录

- 2026-11-19 建立（R29）。触发原因：10 人 / 120 s 单素材库 AUC 1.000、留出 top1 0.300；改为 3 take 后 AUC 0.951、top1 0.700（同池、同留出集、同数据量）。同日补 `--strict-pairs` 与构建期告警。
- 2026-11-19 复核：发现留出集文件夹归属 bug（2/10 查询取自错误歌手）。§2 已重写——作废"校准 0.300 / 余弦 0.500"对照（标签 bug + `unverifiable_legacy` 库），改为"库内 AUC 1.000 的泄漏证据 + 修正后 0.875 vs 0.750（n=8，p=1.0）+ 20 人 0.562 vs 0.688"；明确 R29 只主张**正对不含录音身份**，不主张校准带来精度增益。
- 2026-11-19 补评估闸门：`holdout_eval` 在不可校准（`auc=nan`）或旧库（`unverifiable_legacy`）时拒绝输出 top1/top3，报告新增 `metric_kind` / `pairing` / `pairs` 三个字段，逐条查询行同标 `metric_kind`。此前的工具会把回退余弦的 top1 正常打印，只在旁边写一句 `calibrated:false`。
- 2026-11-19 查询侧透出：判定收敛为单一入口 `library.build.pairing_status()`（构建告警、`compare` 负载、`holdout_eval` 闸门共用）。`compare` 新增 `pairing.status`（`cross_recording` / `single_recording` / `unverifiable_legacy` / `empty`）与 `pairing.single_material_singers`，使 `calibrated=false` 可解释，并让"`calibrated=true` 但库不可验证"这一危险组合显式可见。
