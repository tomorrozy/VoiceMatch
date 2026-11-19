# VoiceMatch 本机分析服务

把 `src/voicematch` 的引擎包装成本机 HTTP 服务，供小程序调用。默认只监听
`127.0.0.1`，不对外网开放。

## 快速开始

```bash
# 直接启动：默认自动选用 material/m4singer_3take/lib（真实 HuBERT 库）
C:\Python\python.exe server/app.py --port 8000
```

看到下面这行就是就绪（首次启动要加载 360 MB 的 HuBERT 主干，约 5–15 秒）：

```
[voicematch] 就绪：encoder=hubert model=HUBERT_BASE library=fc99f0eac38ba34e singers={'female': 5, 'male': 5}
[voicematch] 监听 http://127.0.0.1:8000
```

库的选择顺序：`--library` 显式指定 → `material/m4singer_3take/lib` → `server/demo_library`。
真实库不存在时会自动回退演示库，并在 `/health` 的 `encoder` 字段里显示 `hash`。

模型缓存会**自动接管**：如果库目录的上一级有 `hf_cache/`，服务会把它设为
`TORCH_HOME`，不会重复下载 360 MB 主干。

```bash
# 自检
curl http://127.0.0.1:8000/health

# 一次完整分析（两步：受理 → 轮询）
curl -X POST "http://127.0.0.1:8000/compare" \
     -F "file=@material/m4singer_3take/audio/alto_1/take1.wav" \
     -F "gender=female"
# → 202 {"status":"accepted","task_id":"ojlpnwT3u79Z","state":"queued",…}

curl "http://127.0.0.1:8000/task/ojlpnwT3u79Z?wait=20"
# → 200 {"state":"done","progress":100,"result":{…}}   或 running + 进度

# 只想一把拿到结果（调试用，连接会被占住约 5 秒）
curl -X POST "http://127.0.0.1:8000/compare?mode=sync" \
     -F "file=@material/m4singer_3take/audio/alto_1/take1.wav" \
     -F "gender=female"
```

小程序端：把 `miniprogram/config.js` 的 `USE_MOCK` 改成 `false`（当前默认已是
`false`），并在开发者工具「详情 → 本地设置」勾选「不校验合法域名…」。

### 没有真实素材时：演示库

```bash
C:\Python\python.exe tools/make_demo_library.py --out server/demo_library
C:\Python\python.exe server/app.py --library server/demo_library --port 8000
```

## 接口

### `GET /health`

```json
{
  "status": "ok",
  "library": "…/material/m4singer_3take/lib",
  "model_id": "HUBERT_BASE",
  "library_version": "fc99f0eac38ba34e",
  "calibration_version": "4b0a72587c9c3f34",
  "calibrated": true,
  "pairing": {
    "status": "cross_recording",
    "single_material_singers": [],
    "notice": ""
  },
  "encoder": "hubert",
  "singers": { "female": 5, "male": 5 },
  "segments": 74,
  "queue": { "workers": 1, "queued": 0, "running": 0, "tracked": 0, "max_pending": 20 }
}
```

`queue` 是排查"为什么这次慢"的第一现场：`queued>0` 说明在排队，`running>0`
说明正在打分。

`pairing` 是库的 R29 配对状态（与 `voicematch.cli compare` 同名字段）：

| status | 含义 | 前端该怎么办 |
|---|---|---|
| `cross_recording` | 每位歌手都有 ≥2 份录音，校准可训练 | 正常展示百分比 |
| `single_recording` | 有歌手只有一份录音（名单在 `single_material_singers`） | 展示 `notice`，分数只当排序 |
| `unverifiable_legacy` | 旧版本构建的库，无法验证配对是否跨录音 | 展示 `notice`，**不得当百分比** |
| `empty` | 库为空 | 不应到达此分支 |

`notice` 是一句可直接展示的中文说明；空串表示无需提示。状态 token 不要
原样渲染到界面上。

### `GET /singers?gender=female`

返回库内歌手。`gender` 可省略。

### `POST /compare`

**默认异步**：只做受理，立刻返回 `202` + `task_id`，打分在后台工作线程里进行。
这样移动网络下不会因为长连接被掐断而白算一次，用户也能看到真实进度。

```json
{
  "status": "accepted",
  "task_id": "ojlpnwT3u79Z",
  "state": "queued",
  "queue_position": 0,
  "poll_after_ms": 500,
  "progress": 0,
  "message": "排队中",
  "elapsed_ms": 0,
  "result": null,
  "error": null
}
```

加 `?mode=sync` 回到同步语义：直接返回结果文档（`200`），连接一直占到打分结束。
用 curl 手动试的时候更省事，小程序里不要用。

三种请求体任选其一：

| 方式 | 请求 | 说明 |
|---|---|---|
| multipart | `file=@x.wav` + 表单 `gender=female` | `wx.uploadFile` 用的就是这种 |
| raw | `Content-Type: audio/wav`，性别走 `?gender=` | `curl --data-binary` 最方便 |
| JSON | `{"gender":"female","audio_base64":"…"}` | 便于脚本调用 |

响应与 `voicematch.cli compare` 同构，额外补三个字段：

- `results[].dimensions` — 四维分位（pitch/timbre/resonance/style，0–100）。
  某维度没有可用测量时为 `null`，**不填 0**（0 会凭空表示"最不像"）。
- `results[].attribution` — 四键归因（去掉引擎内部的 `cos` 后，四维重新归一化
  为合计 1.0）。原始五键保留在 `attribution_raw`。
- `pairing` — 与 `/health` 同构的 R29 配对状态块。页面据此决定分数按
  "感知相似度百分比"还是"仅排序"展示（当前小程序在 `status` 非
  `cross_recording` 时降级为排序序号，并展示 `notice` 提示条）。

`status` 三态：

| status | 含义 | results |
|---|---|---|
| `ok` | 通过质量门限并完成排序 | Top-3 |
| `rejected` | 质量门限拒答，`quality.reasons` 列出全部原因 | 空 |
| `no_usable_window` | 过了门限，但没有足够稳定的发声片段 | 空 |

失败统一返回 `{"status":"error","reason":"<稳定 token>","detail":"<可读说明>"}`，
常见 token：`missing_audio`、`library_not_found`、`no_singers_in_pool`、
`non_finite_prototype`、`missing_ml_extra`。

### `GET /task/<task_id>`

异步模式下用这个查进度和取结果。`?wait=<秒>` 是长轮询：最多阻塞这么久
（服务端上限 25 秒），任务一结束立刻返回，比高频短轮询省请求。

```json
{
  "task_id": "ojlpnwT3u79Z",
  "state": "running",
  "stage": "embedding",
  "message": "生成声线嵌入",
  "progress": 76,
  "elapsed_ms": 5419,
  "result": null,
  "error": null
}
```

| state | 含义 | 客户端该做什么 |
|---|---|---|
| `queued` | 排队中，`queue_position` 是前面还有几个 | 继续轮询（间隔 0.5–1 秒） |
| `running` | 正在打分，`progress` 0–100、`message` 可直接展示 | 继续轮询 |
| `done` | 完成，`result` 是上面那种结果文档 | 读 `result`，按 `result.status` 分支 |
| `failed` | 失败，`error` 是 `{reason, detail}` | 按 `reason` 提示用户 |

`stage` 会依次经过 `decode → quality → preprocess → window → features →
embedding → scoring → done`，进度是按各阶段实际耗时给的权重，不是匀速动画。

任务 id 查不到返回 `404 task_not_found`（结果在内存保留 15 分钟，服务重启会丢）。
排队超过 20 个时 `POST /compare` 返回 `503 queue_full` 并带 `Retry-After: 5`。

### `DELETE /task/<task_id>`

取消一次分析。分两种情况：

| 任务状态 | 处理 | 返回 `state` |
|---|---|---|
| `queued` | 直接从队列摘掉，**不会被执行** | `cancelled` |
| `running` | 推理线程不中断（CPU 推理没有安全的抢占点），但结果会被丢弃 | `cancelled` |

```json
{ "task_id": "ojlpnwT3u79Z", "state": "cancelled", "message": "任务已取消" }
```

取消是幂等的：对已 `done` / `failed` 的任务再调用不会报错，原状态不变。
查不到返回 `404 task_not_found`。

已经跑起来的推理没法真正中止（HuBERT 前向没有检查点），所以取消的收益是
**用户立刻脱身 + 结果不被采用 + 临时音频立即删除**，而不是省掉那几秒 CPU。

## 歌手库：默认库与自建库

### 默认库（`material/m4singer_3take/lib`）

- 10 位歌手（5 女 5 男）、74 个片段、28 段不同录音，`model_id=HUBERT_BASE`
- 素材来自 **M4Singer（CC BY-NC-SA）**，仓库里的 `RIGHTS.json` 记录
  `commercial_use: false`。**只能用于内部研究/链路验证，不可上线商用**——
  商用需要换成有商业授权的素材，或自行采集并取得演唱者书面同意。

### 换成自己的库

```bash
# 1. 安装 ML 依赖（torch/torchaudio，约 360 MB）
C:\Python\python.exe -m pip install -e ".[ml]"

# 2. 按 docs/catalog.example.json 准备 catalog，每个人至少两段不同录音
#    （校准层的正对要求"同歌手、不同录音"，只有一段会退化成纯余弦排序，R29）
C:\Python\python.exe -m voicematch.cli build \
    --catalog my_catalog.json --out server/library --strict-pairs

# 3. 用自建库启动
C:\Python\python.exe server/app.py --library server/library --port 8000
```

服务会按库里的 `model_id` 自动选择编码器，不用手动指定 `--encoder`；库是
`HUBERT_BASE` 但环境缺 torch 时，会返回 `missing_ml_extra` 而不是静默降级。

## 真机调试

1. 服务改为 `--host 0.0.0.0`，并放行防火墙对应端口：

   ```bash
   C:\Python\python.exe server/app.py --port 8000 --host 0.0.0.0
   ```

   启动日志会直接打出可用地址和 config 改法，不用自己去 `ipconfig` 里翻：

   ```
   [voicematch] 手机可访问（需同一 Wi-Fi）：
   [voicematch]   http://172.22.158.60:8000
   [voicematch] 小程序改 miniprogram/config.js：API_BASE: 'http://172.22.158.60:8000'
   ```

   列出的地址可能有多个（含虚拟机网卡），选手机所在 Wi-Fi 网段的那个。

2. `miniprogram/config.js` 的 `API_BASE` 改成上面打出的地址。
3. 手机与电脑同一 Wi-Fi；真机预览需在开发者工具打开「不校验合法域名」。
   若连不上，先确认 Windows 防火墙对 `python.exe` 放行了 8000 端口。
4. 正式上线必须换成已在小程序后台配置的 HTTPS 域名——当前 HTTP 仅限本地联调。

## 安全边界

- 默认只监听 `127.0.0.1`；`--host 0.0.0.0` 会把服务暴露到局域网，仅调试时使用。
- 请求体上限 12 MB；上传内容写临时文件，**任务处理结束即删除**（成功、失败都删），
  不落库、不打日志。
- 任务 id 是 `secrets.token_urlsafe(9)` 生成的随机串，不可枚举；结果在内存保留
  15 分钟，进程重启即清空。
- 不返回 traceback，只给异常类型名与消息；完整栈只打到服务端 stderr。
- 没有鉴权。**不要直接暴露到公网**；上线前需要加鉴权、限流与 HTTPS。

## 已知限制

- 小程序录音经系统 AGC/降噪后处理，与引擎「两侧严格对称预处理」的目标存在
  偏差，会削弱 HNR、气声类特征的可靠性。设计文档 5.7 已把采集通道列为 G3 的
  验收前提。
- 打分是**队列串行**的：`--workers 1`（默认）时一次只算一个，第 2 个请求排队。
  这是刻意的——HuBERT 在 CPU 上本来就不并行，多开线程只会互抢核心。要提吞吐
  得先换推理后端（GPU / onnx），不是调线程数。
- 任务队列在内存里，服务重启后所有未取走的结果失效（客户端会收到
  `404 task_not_found`，重新提交即可）。要跨重启保留就得落库，本机联调不需要。
- 默认库只有 10 位歌手，池子很小：`calibrated` 的统计意义有限，Top-3 只说明
  「在这 10 人里最接近」，不代表全歌手空间的准确率。真实的准确率结论仍取决于
  设计文档 6.1 要求的真人盲听基准集。
