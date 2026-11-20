# VoiceMatch 训练方法

> 本文说明「训练」在这套系统里到底指什么、用什么数据、怎么拟合、如何验证。
> 面向想复现或改进排序质量的人。所有结论以代码与 [设计文档](docs/2026-11-18-voice-singer-match-design.md) 第 5 节为准。

## 一句话

**v1 不训练嵌入底座，只训练一个极小的校准层**：用「库内配对」数据拟合一个 Bradley-Terry / RankNet 风格的逻辑回归，把「冻结 HuBERT 余弦相似度 + 四维声学特征差」映射成感知相似度。理由是"像不像"的权威是人耳，而底座冻结、校准层参数少、可解释、可回滚、样本需求低。

## 1. 训练什么、不训练什么

| 组件 | v1 是否训练 | 说明 |
|---|---|---|
| 嵌入底座（HuBERT / wav2vec2 中间层） | ❌ 冻结 | 直接使用冻结表征的中层帧特征，不做 ArcFace / triplet 微调（25 位歌手按歌手切分后每折类别太少，训不动也验不了） |
| **校准层** | ✅ 训练 | 唯一被训练的部分：5 个系数 + 均值/尺度/截距，详见下文 |
| 四维声学特征（pitch/timbre/resonance/style） | ❌ 手工定义 | 规则固定的可解释量，非学习得到 |

微调底座属于 v2，只有在库与标注规模上来、且确实看到瓶颈后才考虑。

## 2. 训练数据：库内配对

校准层不需要真实用户标注即可起步，数据来自**歌手库自身的片段配对**：

- **正对（y=1）**：同一歌手的**两段不同录音**（不同歌曲 / 不同场次）。
- **负对（y=0）**：不同歌手的两段片段。
- **硬约束（R29）**：同一份录音切出的两段**不构成正对**——它们共享场次、房间与伴奏底噪，会教出「同场次=同人」的假规律（实测 hold-out top1 下降）。构建期会告警，`--strict-pairs` 会直接拒绝只有单份录音的歌手。
- **配对特征（5 维）**，顺序固定为 `FEATURE_NAMES = (cos, pitch, timbre, resonance, style)`：
  - `cos` — 冻结嵌入的余弦相似度
  - 其余 4 维 — 两位歌手在音高/音色/共鸣/唱腔上的特征差

配对按歌手 id 排序后分折：`n_folds = min(5, max(2, n_singers // 2))`，负对只在**同折内**两位歌手之间构造，因此每对的两位歌手都在同一折——这是下面「歌手不相交 AUC」成立的前提。每歌手的正负对做固定种子平衡，避免"片段多的歌手"主导类别比例。

## 3. 模型与拟合

实现：`src/voicematch/calibrate/model.py::fit_calibration`。

- **模型**：`sklearn.linear_model.LogisticRegression`，`C=1.0`、`max_iter=1000`。
- **归一化**：对配对特征做 z-score（列均值 `mean`、列标准差 `scale`，常量列 `scale` 固定为 1）。**只有这一层归一化**——打分时严格是：
  ```
  score = sigmoid(coef · ((x - mean) / scale) + intercept)
  ```
- **输出**：一个 `CalibrationModel`，由**五个字段**即可完整复现（G4）：`coef`(5) / `mean`(5) / `scale`(5) / `intercept` / `auc`。打分与归因只依赖这五个数，不含训练数据，可存可重载。
- **归因**：每维对 logit 的绝对贡献 `|coef_i · z_i| / Σ|coef_j · z_j|`，合计为 1（无贡献时为 0），对应报告页「主要因为音色接近」等文案。

### 退化情形不是错误

库太小或太偏斜是正常状态，代码返回**规范化的余弦兜底模型**而非报错：

| 情形 | 结果 |
|---|---|
| 无训练行（如 2 歌手库，每折 1 人无负对） | `coef=[1,0,0,0,0]`、`mean=0`、`scale=1`、`intercept=0`、`auc=nan` |
| 只有一个类别（无正负对比） | 同上，**与上一行完全相同的五元组**（可靠哨兵） |
| 折数 < 2 | 系数在全部行上拟合，但无留出歌手可测，`auc=nan` |
| 常量列 | `scale` 固定 1，该维贡献 `coef·0=0`，其余照常可用 |

`auc=nan` 表示"不可测量"，**不是** 1.0 也不是 0.0。`is_calibrated` 由 `auc` 是否为实测值派生：为 False 时分数**只能排序，不得当成校准百分比**展示（兜底模型下 `score` 恰为 `sigmoid(cos)`，对余弦单调）。

## 4. 训练何时发生（重要）

**校准层不在建库时训练存储，而在每次 `compare` 时从库内配对即时拟合**：

```
compare → load_library → _calibration_for(index):
            build_pairs(index, index.embeddings) → fit_calibration(X, y, groups)
```

这样结果只由「库版本 + 代码版本」决定（G4 逐字节可复现），不依赖一份可能过期的落盘模型。输出里同时写入 `model_id` / `library_version` / `calibration_version` 三个版本号。库内的配对状态通过 `pairing` 字段暴露给前端，决定是否按百分比展示。

## 5. 如何验证「训得好不好」

### 歌手不相交的折外 AUC

`auc` 用 `GroupKFold` 的折外预测计算：把每位歌手的成对数据整组放进同一折，scaler 在每折内重新拟合，因此这个 AUC 衡量的是**对未见歌手**的完整流程表现，而非在全量数据上过拟合的逻辑回归。

### 真人盲听基准集（G1，尚未启动）

这是整个项目最长、也是唯一还没跑起来的链条——**没有它，"准"只是宣称**：

- query 必须是**真实业余清唱**，不能用库内留出录音（库内留出测的是闭集歌手识别，且校准映射不迁移到手机前的普通人）。
- 60–100 人采集，每人 3 段；3 位不同的人独立听标（一致性用 Cohen's kappa，过低则改用纯 pairwise 提问）。
- 库内留出配对**只作阶段 0 的管线自检**，在报告中标为"代理指标"，不进 G1 口径。
- 口径必须成对：`top1/top3` 只在被受理且有答案的查询上算，并与 `acceptance_rate`（下限 85%）同时报告；"都不像"不算 miss，单独计 `n_none_like`。

## 6. 实操：从零构建并校准

```bash
# 1. 安装（含训练/推理需要的精确 pin 数值栈与 ml extra）
pip install -e ".[ml]"

# 2. 按 docs/catalog.example.json 准备目录：每位歌手 ≥2 段不同录音的干净清唱
#    （只有一段的歌手会让校准无正对，--strict-pairs 会拒绝）

# 3. 建库（会校验 catalog、写版本化 manifest + npy）
python -m voicematch.cli build --catalog catalog.json --out lib/ --strict-pairs

# 4. 比对：校准层在该步从库内配对即时拟合
python -m voicematch.cli compare query.wav --library lib/ --gender female

# 5. 跑测试 / 评估
python -m pytest -q
python tools/scorer_bakeoff.py     # 打分器对比
python tools/holdout_eval.py       # 留出自检（代理指标，非 G1）
```

## 7. 训练用途的数据合规（硬边界）

- **默认**：用户上传与标注数据**只用于评估，不用于训练**；校准层优先用库内配对起步，不依赖真实用户数据。
- **二次授权**：只有取得明确授权，真实标注才可转为训练集。
- **素材许可**：默认库 M4Singer 为 CC BY-NC-SA 4.0（`commercial_use: false`），本项目定位非商业研究；商用须换有授权的库并取得演唱者书面授权。
- 详见 [配对约束](docs/2026-11-19-calibration-pairing-constraint.md)、[校准挑战](docs/2026-11-19-scorer-bakeoff.md)、[规模实验](docs/scale-experiment.md)。
