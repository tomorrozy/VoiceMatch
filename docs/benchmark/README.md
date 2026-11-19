# 真人盲听基准集采集工具链

> 把"真人采集 → 盲听标注 → 多数裁决 → 冻结基准集"变成可执行的填表式流程。
> 这是全项目最长、也是唯一必须由真人参与的一条链；本目录的工具把其中可自动化的部分做掉。

> **你是组织者、想问"我到底该怎么做" → 先读 [`00-START-HERE.md`](00-START-HERE.md)。**
> 那份是现场操作手册（谁做什么、每步命令、验收判据、常见坑）；本文讲工具链本身与不变量。

---

## 责任边界

| 环节 | 谁做 | 工具 |
|---|---|---|
| 开工前置检查（库/性别池） | 工具 | `tools/benchmark_walkthrough.py preflight` |
| 零成本演练（先自己走一遍） | 工具 | `tools/benchmark_walkthrough.py run` |
| 真实采集的顺序 / 幂等 / 标注合并 | 工具 | `tools/benchmark_run.py`（见下） |
| 招募、筛选、排期 | 真人（组织者） | `04-recruitment.md` |
| 签署同意书 | 真人（组织者 + 参与者） | `01-consent-form.md` |
| 清唱录音（60–100 人 × 3 段） | 真人（参与者） | `02-capture-checklist.md` |
| 盲听标注（3 人独立） | 真人（标注者） | `annotation-ui.html` + `03-annotation-guide.md` |
| 诊断素材可用性 + 替换建议 | 工具 | `tools/benchmark.py diagnose` |
| 导出候选试听样本 | 工具 | `tools/benchmark.py samples` |
| 生成盲听清单 + 系统答案分离 | 工具 | `tools/benchmark.py sheet` |
| kappa 检验 + 严格多数裁决 | 工具 | `tools/benchmark.py adjudicate` |
| kappa 不达标时的 pairwise 重标 | 工具 + 真人 | `tools/benchmark.py pairwise-sheet` / `pairwise-adjudicate` |
| 冻结 eval set + 版本/来源锁定 | 工具 | `tools/benchmark.py freeze` |

### 两个入口，别混

| 想干什么 | 用哪个 | 产物可信吗 |
|---|---|---|
| 还没招到人，先看整条链长什么样 | `tools/benchmark_walkthrough.py` | ❌ 合成音 + 脚本编的标注，只有 `DEMO-ONLY` 教学价值 |
| 真实录音要跑数 | `tools/benchmark_run.py` | ✅ 真值仍由 3 人多数裁决产生 |

`benchmark_run.py` 不重实现任何口径，只加四件现场最缺的事：

```bash
# 下文 python 一律指项目解释器 C:/Python/python.exe
python tools/benchmark_run.py init   --dir material/benchmark \
    --library material/m4singer20_3take/lib \
    --catalog material/m4singer20_3take/catalog/catalog.json   # 骨架 + config + specs 模板
python tools/benchmark_run.py preflight --dir material/benchmark  # 查重名/时长/性别/路径/编码器
python tools/benchmark_run.py status --dir material/benchmark      # 七步到哪了 + 下一条命令
python tools/benchmark_run.py run    --dir material/benchmark --steps diagnose,samples,sheet
python tools/benchmark_run.py merge  --dir material/benchmark      # 合并 3 份导出，拒重复投票
```

- **preflight 拦两类最贵的错**：音频**重名**（界面按文件名匹配会串音）、**短于 8 秒**
  （永远过不了受理门槛）。这两种错在标注完成后才暴露，返工 = 重录 + 重标。
- **run 是幂等的**：已完成的步骤跳过；`adjudicate` 遇到 kappa 低于门槛会**停住**并给出
  pairwise 重标命令，要继续必须显式 `--allow-low-kappa`（门槛值会写进 MANIFEST）。

---

## 前置条件

- 已建好歌手库（`python -m voicematch.cli build ...`），研究/内部用途即可。
- 已把真人录音整理成 `specs.json`（见下），相对路径以 `--specs-dir` 为基准。
- **所有命令用项目解释器 `C:/Python/python.exe`**（PATH 里的 python 缺依赖）。

### 库的选择：必须用 20 人库

设计 6.2 要求每条 query 展示 **Top-3 + 5 位同性别随机歌手 = 8 位候选**。
`candidate_set` 在同性别池不足 8 人时**不报错，只静默返回整个池**：

| 库 | 同性别池 | 结果 |
|---|---|---|
| `material/m4singer/lib`（10 人） | 5 / 5 | ❌ 静默降级成 5 选 1，问法失真 |
| `material/m4singer20_3take/lib`（20 人） | 10 / 10 | ✅ 凑齐 8 位，且 3-take 满足 R29 配对约束 |

本文档命令统一使用（前者给 samples/diagnose，后者给 sheet/freeze/evaluate）：

```
CATALOG=material/m4singer20_3take/catalog/catalog.json
LIB=material/m4singer20_3take/lib
```

开工前先跑一次前置检查，全绿再继续：

```bash
C:/Python/python.exe tools/benchmark_walkthrough.py preflight --catalog $CATALOG --library $LIB
```

> 换库后必须重新导出 samples，否则候选歌手 id 与试听样本对不上。
> 冻结时权利声明按 `--rights` > `lib/catalog/RIGHTS.json` > 同级 `catalog/RIGHTS.json`
> 的顺序查找，会自动带进 MANIFEST 的 `rights` 字段。

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

## 演练：真人到位前先零成本走一遍

`tools/benchmark_walkthrough.py run` 用**合成录音 + 脚本编的标注**把下面 6 条命令
完整跑一遍，产物落在独立目录并打 `demo=true` 标记。它只证明链路能跑通，
**所有数字都不是结论**（详见 `00-START-HERE.md` 第 3 节）：

```bash
C:/Python/python.exe tools/benchmark_walkthrough.py run \
  --catalog $CATALOG --library $LIB --out material/benchmark_walkthrough
```

建议组织者先跑这个，看清每步产物，再开始招人。

---

## 第 0 步：诊断素材 + 导出候选试听样本

**先诊断**（可选但推荐）：每位歌手逐素材报告时长、质量门结论、可用窗数，
对切不出窗的素材给出替换建议（换更长/更干净的清唱，或换一首歌/场次）：

```bash
C:/Python/python.exe tools/benchmark.py diagnose \
  --catalog $CATALOG \
  --out material/benchmark/diagnosis.json
```

输出 `n_needs_replacement` 与歌手 id 列表；`diagnosis.json` 逐条给出 `status`(ok/needs_replacement)、
`n_windows`、`reject_reasons` 与 `advice`。

**再导出试听样本**：标注者要能听到每位候选歌手的一段清唱。每位歌手取其库内实际
使用的发声窗（**与 `build_library` 同一条路径**：`PREPROCESS_LIBRARY` + `select_windows`，
不套查询侧的 60 s 上限），样本仅供内部标注，不对外分发原唱片段：

```bash
C:/Python/python.exe tools/benchmark.py samples \
  --catalog $CATALOG \
  --out-dir material/benchmark/samples \
  --secs 10
```

产出 `samples/sample_<singer_id>.wav` 与 `samples/samples.json`（singer_id → 展示名/性别/文件）。

> **已修复的真实 bug**：早期版本误用查询侧管线（先截到 60 s 再选窗），把 M4Singer
> `soprano_2` 位于 54–64 s 的唯一可用窗截掉，误报"无可发声窗"。现改用库路径后，
> 10 位歌手全部产出样本，`diagnose` 也确认全部可用。样本必须与库同源同口径。

## 第 1 步：生成盲听清单

```bash
C:/Python/python.exe tools/benchmark.py sheet \
  --specs material/benchmark/specs.json \
  --specs-dir material/benchmark \
  --library $LIB \
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

用浏览器打开 `annotation-ui.html`，点"选择基准集目录"选中该目录即可逐条标注。
界面有两种模式：

- **多选标注**（默认）：加载 `sheet.json`，自动播放查询、加载 8 位候选样本，
  标注者勾选 1–2 位或"都不像"，导出 `annotations-<id>.json`；
- **pairwise 比较**：kappa 未达标重标时，把 `pairs.json` 和 `sheet.json` 放进同一目录，
  切到该模式，界面逐对播放查询 + A/B，标注者选 A/B/分不出，导出
  `pair-annotations-<id>.json`。

界面完全离线（File API 读本地文件），支持"保存进度到本机/载入进度"，
3 位标注者各导出一份后合并。

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
C:/Python/python.exe tools/benchmark.py adjudicate \
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
C:/Python/python.exe tools/benchmark.py pairwise-sheet \
  --adjudicated material/benchmark/adjudicated.json \
  --sheet material/benchmark/sheet.json \
  --out material/benchmark/pairs.json --top 4

# 2) 标注者回答每对：winner = "a" / "b" / null（分不出），合并成 pair-annotations.json
#    格式：{"query_id","pair_id","annotator_id","winner"}

# 3) 裁决成与 adjudicate 同格式的 adjudicated-pairwise.json
C:/Python/python.exe tools/benchmark.py pairwise-adjudicate \
  --pairs material/benchmark/pairs.json \
  --annotations material/benchmark/pair-annotations.json \
  --out material/benchmark/adjudicated-pairwise.json
```

该文件的 `label_method=pairwise`，冻结后写入 MANIFEST，审计时能看出这批标注是重标而来。
第 4 步的 `--adjudicated` 改指向 `adjudicated-pairwise.json` 即可，其余不变。

## 第 4 步：冻结基准集

```bash
C:/Python/python.exe tools/benchmark.py freeze \
  --sheet material/benchmark/sheet.json \
  --adjudicated material/benchmark/adjudicated.json \
  --out material/benchmark/frozen \
  --library $LIB \
  --encoder hubert
```

产出 `frozen/eval_set.json`（可被 `run_eval` / CLI 直接消费）与 `frozen/MANIFEST.json`
（benchmark_version、kappa、争议 query、真值来源声明、权利声明）。

## 第 5 步：跑出真实 G1

```bash
C:/Python/python.exe tools/benchmark.py evaluate \
  --eval-set material/benchmark/frozen/eval_set.json \
  --library $LIB \
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
