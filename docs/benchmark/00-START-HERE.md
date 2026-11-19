# 从这里开始：真人盲听基准集现场手册

> **给组织者。** 这份文档回答"我到底该怎么做"。
> 一句话：**你只需要做三件我无法代做的事——招人、录音、找 3 个人独立听标。**
> 其余全部是复制命令、看一行 JSON 输出、按验收判据打勾。

工具链已经建好并测过（435 个测试通过）。本文档不重复工具原理，只讲**现场动作**；
原理与不变量见同目录 `README.md`。

---

## 0. 先搞清楚边界（30 秒）

| 你（真人）必须做 | 工具自动做 | 绝对不能做 |
|---|---|---|
| 招募 60–100 位普通业余歌手，收同意书 | 诊断素材、导出每位候选的试听样本 | 让同一个人标 3 次冒充 3 位标注者 |
| 每人录 3 段清唱（不同设备/场景） | 生成 8 位候选的盲听清单、藏起系统答案 | 把 `provenance.json` 发给标注者 |
| 找 3 位**不同**的普通听众独立听标 | kappa 一致性检验 + 严格多数裁决 | kappa 不达标时调低门槛 |
| — | 冻结版本化基准集、算真实 G1 | 把不合格录音从清单里删掉 |

> **为什么不能代做**：真人清唱、独立听感判断、多数裁决的真值，是"产品准确率"的
> 定义本身。任何合成/自动标签都只是代理指标，不能作为 G1 结论。

---

## 1. 全景：7 个阶段

| # | 阶段 | 谁 | 输入 → 输出 | 做完的标志 |
|---|---|---|---|---|
| A | 一次性准备 | 你（10 分钟） | 建目录 + 确认库 | `preflight` 全绿 |
| 0 | 演练（推荐） | 工具 | 合成样例 | 跑通 6 条命令，看到每步产物 |
| 1 | 招募 + 同意书 | 真人 | → 签署的同意书 + 参与者代号 | 男女分层、每人一份 |
| 2 | 清唱录音 | 真人 | → `audio/*.wav` + `specs.json` | 每条满足 8s/60s 契约 |
| 3 | 生成盲听清单 | 工具 | `specs.json` → `sheet.json` + `provenance.json` | 每条 8 位候选 |
| 4 | 3 人独立听标 | 真人 | → 3 份 `annotations-<id>.json` | 3 份齐、互不通气 |
| 5 | 裁决 + 冻结 | 工具 | → `frozen/eval_set.json` + `MANIFEST.json` | kappa ≥ 0.6 |
| 6 | 跑真实 G1 | 工具 | → 一行指标 JSON | `label_source` 正确 |

只有 **1、2、4** 需要真人。第 0 步是零成本的自我教学，强烈建议先做。

---

## 2. 开工前：一次性准备（约 10 分钟）

### 2.1 解释器（必看，否则命令全报错）

PATH 里的 `python` 是另一个缺依赖的解释器。**所有命令必须用项目解释器**：

```bash
C:/Python/python.exe --version
```

本文档与 `README.md` 的命令都写全了 `C:/Python/python.exe`，可直接复制粘贴，
不需要再替换。先确认上面这行能打印出版本号再往下走。

### 2.2 目录骨架

在工作区根目录建一个**专用目录**，真人材料与内部文件分开放：

```
material/benchmark/
├── audio/                  # 真人清唱录音（只放这个）
├── specs.json              # 录音清单（你填，模板见 2.4）
├── samples/                # 工具产出：候选歌手试听样本
├── samples.json            # 工具产出
├── diagnosis.json          # 工具产出（可选）
├── sheet.json              # 工具产出：发给标注者
├── provenance.json         # 工具产出：★内部，绝不外发
├── annotations-*.json      # 3 位标注者各一份
├── adjudicated.json        # 工具产出
└── frozen/                 # 工具产出：eval_set.json + MANIFEST.json
```

```bash
mkdir -p material/benchmark/audio
```

也可以一条命令建好骨架（含 `specs.json` 空表、模板、`config.json`，已有文件不覆盖）：

```bash
C:/Python/python.exe tools/benchmark_run.py init --dir material/benchmark \
  --library material/m4singer20_3take/lib \
  --catalog material/m4singer20_3take/catalog/catalog.json
```

> **发给标注者的"标注包"** = `sheet.json` + `samples/` + `audio/` + `annotation-ui.html`
> + `03-annotation-guide.md` 打包成一个文件夹。
> **`provenance.json` 永远不进这个包**——它含系统 Top-3，泄露即盲听失效。

### 2.3 选库：必须用 20 人库（这是个真坑）

设计 6.2 要求每条 query 展示 **Top-3 + 5 位同性别随机歌手 = 8 位候选**。
`candidate_set` 在同性别池不足 8 人时**不会报错，只会静默返回整个池**：

| 库 | 同性别池 | 结果 |
|---|---|---|
| `material/m4singer/lib`（10 人） | 5 / 5 | ❌ 静默降级成 5 选 1，问法失真 |
| `material/m4singer20_3take/lib`（20 人） | 10 / 10 | ✅ 凑齐 8 位，且 3-take 满足 R29 配对约束 |

**统一使用这一组路径**（前者给 samples/diagnose，后者给 sheet/freeze/evaluate）：

```bash
CATALOG=material/m4singer20_3take/catalog/catalog.json
LIB=material/m4singer20_3take/lib
```

> 二者歌手 id 一致（soprano_1…bass_3），样本与清单能对上。**若你换了库，必须重新导出 samples**。

### 2.4 前置检查与工作目录初始化

**① 检查库能不能出 8 位候选**（演练脚本自带，也可单独跑）：

```bash
C:/Python/python.exe tools/benchmark_walkthrough.py preflight \
  --catalog material/m4singer20_3take/catalog/catalog.json \
  --library material/m4singer20_3take/lib
```

全绿才继续。若报"同性别池 < 8"，说明选错库，回到 2.3。

**② 正式采集：用编排器建骨架、录音前自检、看进度**（`tools/benchmark_run.py`）：

```bash
C:/Python/python.exe tools/benchmark_run.py init --dir material/benchmark \
  --library material/m4singer20_3take/lib \
  --catalog material/m4singer20_3take/catalog/catalog.json
C:/Python/python.exe tools/benchmark_run.py preflight --dir material/benchmark
C:/Python/python.exe tools/benchmark_run.py status --dir material/benchmark
```

- `init` 建目录骨架 + config + `specs.json` 模板（**不覆盖**已有文件）；
- `preflight` 查**录音重名、短于 8 秒、性别拼错、路径不存在**——这些错到标注完成后才暴露，返工 = 重录 + 重标；
- `status` 报七步走到哪、下一条该敲什么命令；`run` 幂等执行可自动化步骤，`merge` 合并 3 份标注并拒绝重复投票。

> **两个 preflight 别混**：①查**库**（能否凑齐 8 位候选），②查**录音文件**（合不合格）。
> 第 5 节的手敲命令用来理解每一步；正式跑数推荐用 `benchmark_run.py status/run` 编排，
> 它不重实现任何口径，只是把现场最容易漏的顺序与幂等补上。

---

## 3. 第 0 天：先零成本走一遍（强烈建议）

真人还没招到时，先用**合成录音 + 脚本编的模拟标注**把 6 条命令完整跑一遍，
看清每步产物长什么样、命令怎么串。**这是教学，不是数据**：

```bash
C:/Python/python.exe tools/benchmark_walkthrough.py run \
  --catalog material/m4singer20_3take/catalog/catalog.json \
  --library material/m4singer20_3take/lib \
  --out material/benchmark_walkthrough
```

它会依次执行 `samples → sheet → adjudicate → freeze → evaluate` 并逐步打印中文说明。
产物全部落在 `material/benchmark_walkthrough/`，目录内有 `DEMO-ONLY.txt` 警告。

> ⚠️ **演练里的标注是脚本自己编的，G1 数字毫无意义。**
> 它只证明"链路能跑通"，绝不能写进报告或当作基准集。`MANIFEST.json` 里的
> `label_source` 是工具链的固定声明，在演练场景下不代表真人标注。

看完演练产物后再开始招人，你会清楚每一步在等什么。

---

## 4. 正式采集：三件真人任务怎么做

### 4.1 招募（第 1 阶段）

- **人数**：60–100 位**普通业余歌手**，男女分层（尽量各半）。目标用户画像 = 普通听众/唱歌爱好者，**不要**声乐专业者。另备 5–10 人专业小组做二次校验（看技术维度是否被捕捉）。
- **话术要点**："录 3 段清唱，用于评估一个'像哪位歌手'的功能准不准；录音只给匿名听众听，不对外发布；可随时要求删除。"
- **同意书**：每位一份 `01-consent-form.md`，**必须本人签**。"用于评估"是前提；"用于训练"默认不同意、需单独勾选。
- **代号**：只用脱敏代号（`p001`…），同意书上不写真实姓名与录音文件的直接关联表由你另行加密保管（用于响应删除请求）。
- **排班建议**：不要一次招满。先按下面 §7 的 MVP 招 5 人跑通，再扩到全量。
- **细化到可照读**：配额（性别/年龄/设备/声线）、三条触达话术、5 问筛选问卷、
  3 天排期模板、常见拒绝处理，见 [`04-recruitment.md`](04-recruitment.md)。

### 4.2 清唱录音（第 2 阶段）

每位参与者 **3 段**，每段：

- 纯清唱、**无伴奏**；有效发声 ≥ 8 秒、总时长 ≤ 60 秒；
- 三段尽量覆盖**不同设备或场景**（`device_id` 至少 2 个不同值，否则 G3 跨设备稳定性测不了）；
- 由系统自动选窗，**不要人工精挑最好听的乐句**——精挑会抬高 G1 且不可迁移；
- 质量门（时长/削波/SNR/发声占比）由管线判定，采集人员只记录、不替它判断。

用 `02-capture-checklist.md` 逐人登记。常见不合格：伴奏残留、有效发声不足 8s、
削波、iOS Safari 的 AGC/降噪（若用了要单独标注，不与安卓/Web 混算 G3）。

**填 `specs.json`**（相对 `material/benchmark/`）：

```json
[
  {"query_id":"p001-a","path":"audio/p001-a.wav","gender":"female","subject_id":"p001","device_id":"phone-a"},
  {"query_id":"p001-b","path":"audio/p001-b.wav","gender":"female","subject_id":"p001","device_id":"phone-b"},
  {"query_id":"p001-c","path":"audio/p001-c.wav","gender":"female","subject_id":"p001","device_id":"web"}
]
```

`query_id` 全局唯一；`subject_id` 同一人 3 段相同；`device_id` 记录设备/通道。
**质量门拒绝的录音也要留在 specs 里**——清单会为它生成随机候选，让 `run_eval`
正确拒答，受理率分母才诚实。

### 4.3 3 人独立听标（第 4 阶段）

- **必须是 3 位不同的人**，各标一份，互不通气、不讨论。自己标 3 次会让 kappa 恒为 1，是无效数据。
- 凑人方式：朋友/同事/众包均可；三人不必同时在线，界面支持"保存进度到本机/载入进度"。
- **发放标注包**：`sheet.json` + `samples/` + `audio/` + `annotation-ui.html` + `03-annotation-guide.md`，每人一份拷贝。**核对包内没有 `provenance.json`。**
- **操作**：浏览器打开 `annotation-ui.html` → 选目录 → 填标注者 ID → 逐条听查询、点候选试听 → 勾最像的 1–2 位或"都不像" → 导出 `annotations-<id>.json`。
- **回收**：收齐 3 份，合并成一个 JSON list（把三份数组拼起来即可，字段不变）：

```json
[
  {"query_id":"p001-a","annotator_id":"rater1","picked":["soprano_1","alto_4"]},
  {"query_id":"p001-a","annotator_id":"rater2","picked":["soprano_1"]},
  {"query_id":"p001-a","annotator_id":"rater3","picked":[]}
]
```

`picked: []` = "都不像"，是正当答案。

---

## 5. 每一步的命令与验收

> 以下命令都在**工作区根目录**执行，解释器已写全，可直接复制。
> 先设好两个变量（Git Bash 里执行；Windows CMD 请自行替换成完整路径）：
>
> ```bash
> CATALOG=material/m4singer20_3take/catalog/catalog.json
> LIB=material/m4singer20_3take/lib
> ```

> **正式跑数更省事**：用 `tools/benchmark_run.py status --dir material/benchmark` 看下一步、
> `run --steps ...` 幂等执行、`merge` 合并 3 份标注（见 §2.4）。
> 下面逐条手敲的命令是让你理解每一步的输入 / 输出 / 验收；编排器内部调的就是它们，
> 并没有另起一套口径。

### 5.1 诊断 + 导出试听样本

```bash
C:/Python/python.exe tools/benchmark.py diagnose --catalog $CATALOG --out material/benchmark/diagnosis.json
C:/Python/python.exe tools/benchmark.py samples  --catalog $CATALOG --out-dir material/benchmark/samples --secs 10
```

**验收**：`n_needs_replacement: 0`；`samples/` 里每位歌手一个 wav。若有歌手被标
`needs_replacement`，按 `diagnosis.json` 的 `advice` 换更长/更干净的清唱素材后重建库。

### 5.2 生成盲听清单

```bash
C:/Python/python.exe tools/benchmark.py sheet \
  --specs material/benchmark/specs.json --specs-dir material/benchmark \
  --library $LIB --encoder hubert \
  --out material/benchmark/sheet.json \
  --provenance material/benchmark/provenance.json --seed 0
```

**验收**：`n_queries` = 录音条数；`sheet.json` 每条 `candidates` **恰好 8 位**；
`grep` 确认 sheet 里没有 `system_top`。`n_scorable` 告诉你有多少条能被系统打分
（少的那部分会走随机候选，仍留在集内）。

### 5.3 kappa 检验 + 多数裁决

```bash
C:/Python/python.exe tools/benchmark.py adjudicate \
  --annotations material/benchmark/annotations.json \
  --sheet material/benchmark/sheet.json \
  --out material/benchmark/adjudicated.json
```

**验收**：`below_threshold: false`（kappa 均值 ≥ 0.6）→ 进 5.5。
若为 `true` → **不要调低门槛**，走 5.4 的 pairwise 重标。

### 5.4 （条件分支）kappa 不达标：pairwise 重标

```bash
C:/Python/python.exe tools/benchmark.py pairwise-sheet \
  --adjudicated material/benchmark/adjudicated.json \
  --sheet material/benchmark/sheet.json \
  --out material/benchmark/pairs.json --top 4
# 把 pairs.json + sheet.json + samples/ 放进标注包，界面切到 "pairwise 比较" 模式，
# 3 人各导出 pair-annotations-<id>.json，合并成 pair-annotations.json
C:/Python/python.exe tools/benchmark.py pairwise-adjudicate \
  --pairs material/benchmark/pairs.json \
  --annotations material/benchmark/pair-annotations.json \
  --out material/benchmark/adjudicated-pairwise.json
```

之后 5.5 的 `--adjudicated` 改指向 `adjudicated-pairwise.json`。
MANIFEST 会记 `label_method=pairwise`，审计时能看出这批是重标而来。

### 5.5 冻结基准集

```bash
C:/Python/python.exe tools/benchmark.py freeze \
  --sheet material/benchmark/sheet.json \
  --adjudicated material/benchmark/adjudicated.json \
  --out material/benchmark/frozen --library $LIB --encoder hubert
```

**验收**：产出 `frozen/eval_set.json` + `frozen/MANIFEST.json`；冻结时已用
`load_eval_set` 读回自校验，坏集会直接报错。此后**锁版本**，不再改标注。

### 5.6 跑真实 G1

```bash
C:/Python/python.exe tools/benchmark.py evaluate \
  --eval-set material/benchmark/frozen/eval_set.json \
  --library $LIB --encoder hubert
```

**验收**：`label_source = "blind_human_majority_annotation"`；
`metrics` 里同时看 **Top-1 / Top-3 命中率** 与 **受理率（≥ 0.85）**——
只在被受理且有答案的 query 上算命中率，必须同时报受理率，否则可靠拒答会刷高命中率。

---

## 6. 判据速查表

| 判据 | 门槛 | 不达标怎么办 |
|---|---|---|
| Cohen's kappa 均值 | ≥ 0.6 | 改 pairwise 重标，**不调门槛** |
| 候选数 | 每条恰好 8 位 | 换 20 人库（见 2.3） |
| 受理率 | ≥ 0.85 | 查录音质量，不要删不合格 query |
| G3 Top-3 重合 | 同一人 3 设备 ≥ 2/3 | 查设备/场景是否真的不同 |
| "都不像"占比 | 无硬门槛 | **占比高 = 扩库信号**，不是调阈值理由 |
| kappa = `nan` | — | 单人/标签恒定测不到，**不是 0，也不能算通过** |

---

## 7. 常见坑（照抄前扫一眼）

1. **用错 python** → PATH 里的 python 没装依赖。统一 `C:/Python/python.exe`。
2. **选了 10 人库** → 静默 5 选 1。用 `material/m4singer20_3take/`。
3. **把 provenance 发给标注者** → 盲听失效。发标注包前核对。
4. **自己标 3 次** → kappa 假性 1.0。必须 3 个不同的人。
5. **删掉质量不合格的录音** → 受理率虚高成 1.0。留在 specs 里。
6. **人工精挑乐句** → G1 虚高不可迁移。让管线自动选窗。
7. **换库不重导 samples** → 候选 id 对不上。换库必重导。
8. **规模一次拉满** → 建议 MVP：先 5 人 × 3 段 = 15 条跑通全流程、
   确认问法与 kappa 稳，再扩到 60–100 人 × 3 段。
9. **把演练数字当结论** → `benchmark_walkthrough` 的标注是脚本编的。

---

## 8. 什么时候算"做完了"

同时满足三条：

1. `frozen/eval_set.json` 与 `MANIFEST.json` 已生成并锁版本；
2. `evaluate` 输出的 `label_source` 是 `blind_human_majority_annotation`；
3. 报告里 Top-1/Top-3 与受理率一起给出，且注明这是**盲听多数真值**下的产品准确率。

> **与现有数字的区别**：仓库里 M4Singer 留出评估的 top1 0.60 / top3 0.70 是
> **数据集 ground-truth 代理指标**，不是产品准确率。只有上面这条链给出的才是 G1。
> MANIFEST 的 `label_source` 就是这条区分的机器可读凭据。

---

## 9. 真人完成后仍需你判断的

- 60–100 人的招募渠道与预算；
- 3 位标注者的补偿与排班；
- 声音作为敏感个人信息的加密存储、最短保留期、删除请求的响应流程；
- M4Singer 是 CC BY-NC-SA 且歌手匿名：**研究/内部评估可用，不可商用**，
  也不满足"最像哪位歌手"的产品语义。若要进产品，需换成已授权、可署名的歌手库。
