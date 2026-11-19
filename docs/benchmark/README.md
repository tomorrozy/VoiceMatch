# 真人盲听基准集采集工具链

> 把"真人采集 → 盲听标注 → 多数裁决 → 冻结基准集"变成可执行的填表式流程。
> 这是全项目最长、也是唯一必须由真人参与的一条链；本目录的工具把其中可自动化的部分做掉。

---

## 责任边界

| 环节 | 谁做 | 工具 |
|---|---|---|
| 招募、签署同意书 | 真人（组织者 + 参与者） | `01-consent-form.md` |
| 清唱录音（60–100 人 × 3 段） | 真人（参与者） | `02-capture-checklist.md` |
| 盲听标注（3 人独立） | 真人（标注者） | `annotation-ui.html` + `03-annotation-guide.md` |
| 导出候选试听样本 | 工具 | `tools/benchmark.py samples` |
| 生成盲听清单 + 系统答案分离 | 工具 | `tools/benchmark.py sheet` |
| kappa 检验 + 严格多数裁决 | 工具 | `tools/benchmark.py adjudicate` |
| kappa 不达标时的 pairwise 重标 | 工具 + 真人 | `tools/benchmark.py pairwise-sheet` / `pairwise-adjudicate` |
| 冻结 eval set + 版本/来源锁定 | 工具 | `tools/benchmark.py freeze` |

---

## 前置条件

- 已建好歌手库（`python -m voicematch.cli build ...`），研究/内部用途即可。
- 已把真人录音整理成 `specs.json`（见下），相对路径以 `--specs-dir` 为基准。

### specs.json 格式

```json
[
  {
    "query_id": "p001-a",
    "path": "audio/p001-a.wav",
    "gender": "female",
    "subject_id": "p001",
    "device_id": "phone-a"
  }
]
```

`subject_id`/`device_id` 可省略，但 G3 跨设备稳定性要求它们存在。

---

## 第 0 步：导出候选试听样本

标注者要能听到每位候选歌手的一段清唱。用库素材导出（每位歌手取其干净发声窗，
与建库同一条选窗规则；样本仅供内部标注，不对外分发原唱片段）：

```bash
python tools/benchmark.py samples \
  --catalog material/m4singer/catalog/catalog.json \
  --out-dir material/benchmark/samples \
  --secs 10
```

产出 `samples/sample_<singer_id>.wav` 与 `samples/samples.json`（singer_id → 展示名/性别/文件）。
素材无可发声窗的歌手会被跳过并在 `warnings` 里列出——标注界面会把该歌手标为"无试听样本"。

## 第 1 步：生成盲听清单

```bash
python tools/benchmark.py sheet \
  --specs material/benchmark/specs.json \
  --specs-dir material/benchmark \
  --library material/m4singer/lib \
  --encoder hubert \
  --out material/benchmark/sheet.json \
  --provenance material/benchmark/provenance.json \
  --seed 0
```

- `sheet.json`：发给标注者，只含 query_id / path / gender / candidates / 元数据，
  **不含系统 Top-3、不含分数**。
- `provenance.json`：**内部文件，绝不发给标注者**，记录系统当时对每条 query 的预测，供审计。
- 质量门拒绝或无可打分窗的录音也会生成候选清单（随机同性别歌手），留在基准集里，
  以保证 `run_eval` 的受理率分母诚实。

## 第 2 步：收集标注

把 `sheet.json`、`samples/`（第 0 步产物）和查询录音放进同一个目录，

用浏览器打开 `annotation-ui.html`，点"选择基准集目录"选中该目录即可逐条标注：
界面自动播放查询、加载 8 位候选的试听样本，标注者勾选 1–2 位或"都不像"，
填标注者 ID 后导出 `annotations-<id>.json`。该界面完全离线（File API 读本地文件），
支持"保存进度到本机/载入进度"，3 位标注者各导出一份后合并。

也可按 `03-annotation-guide.md` 手工标注。合并后的标注文件是一个 JSON list：

```json
[
  {"query_id": "p001-a", "annotator_id": "rater1", "picked": ["f3", "f7"]},
  {"query_id": "p001-a", "annotator_id": "rater2", "picked": ["f3"]},
  {"query_id": "p001-a", "annotator_id": "rater3", "picked": []}
]
```

`picked: []` 表示"都不像"。

## 第 3 步：kappa 检验 + 多数裁决

```bash
python tools/benchmark.py adjudicate \
  --annotations material/benchmark/annotations.json \
  --sheet material/benchmark/sheet.json \
  --out material/benchmark/adjudicated.json \
  --kappa-threshold 0.6
```

输出一行 JSON：`kappa_mean`、每个 pair 的 kappa、低于门槛告警、争议 query 列表。
门槛默认 **0.6**（`KAPPA_THRESHOLD_DEFAULT`，Landis & Koch 的 moderate 上界），
可用 `--kappa-threshold` 覆盖但需有理由。kappa 低于门槛 → 问法不稳，**不要调低门槛**，
走下面的 pairwise 重标路径。

### 若 kappa 未达标：pairwise 重标（设计 6.2）

对争议 query 把候选拆成两两比较（"A 和 B 哪个更像？"），降低 8 选 1 的认知负担：

```bash
# 1) 从争议 query 生成比较任务（--top N 限制每位歌手的比较规模）
python tools/benchmark.py pairwise-sheet \
  --adjudicated material/benchmark/adjudicated.json \
  --sheet material/benchmark/sheet.json \
  --out material/benchmark/pairs.json --top 4

# 2) 标注者回答每对：winner = "a" / "b" / null（分不出），合并成 pair-annotations.json
#    格式：{"query_id","pair_id","annotator_id","winner"}

# 3) 裁决成与 adjudicate 同格式的 adjudicated-pairwise.json
python tools/benchmark.py pairwise-adjudicate \
  --pairs material/benchmark/pairs.json \
  --annotations material/benchmark/pair-annotations.json \
  --out material/benchmark/adjudicated-pairwise.json
```

该文件的 `label_method=pairwise`，冻结后写入 MANIFEST，审计时能看出这批标注是重标而来。
第 4 步的 `--adjudicated` 改指向 `adjudicated-pairwise.json` 即可，其余不变。

## 第 4 步：冻结基准集

```bash
python tools/benchmark.py freeze \
  --sheet material/benchmark/sheet.json \
  --adjudicated material/benchmark/adjudicated.json \
  --out material/benchmark/frozen \
  --library material/m4singer/lib \
  --encoder hubert
```

产出 `frozen/eval_set.json`（可被 `run_eval` / CLI 直接消费）与 `frozen/MANIFEST.json`
（benchmark_version、kappa、争议 query、真值来源声明、权利声明）。

## 第 5 步：跑出真实 G1

```bash
python tools/benchmark.py evaluate \
  --eval-set material/benchmark/frozen/eval_set.json \
  --library material/m4singer/lib \
  --encoder hubert
```

> 注意：这一步的指标才是"产品准确率"（盲听多数真值），
> 与 M4Singer 留出评估的代理指标（数据集 ground truth）严格区分。
> MANIFEST 的 `label_source` 就是这条区分的机器可读凭据。

---

## 不变量（违反即失真）

1. sheet 文件里永不出现 `system_top` 或分数；系统答案只在独立的 provenance 文件。
2. 受理率分母包含被质量门拒绝的录音——所以清单阶段剔除它们是错的。
3. kappa 测不到（单人/标签恒定）是 `nan`，不是 0.0；也不能据 nan 判"通过门槛"。
4. 冻结即校验：写盘后用 `load_eval_set` 读回，坏集直接失败。
5. 争议 query 不删，只在 MANIFEST 列出，交给二次校验。
