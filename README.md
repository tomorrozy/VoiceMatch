# VoiceMatch（声纹研究所）

录一段清唱，告诉你**你的声线最接近哪一类嗓音**。一个「趣味性声线测试」小程序，走**准确性优先**路线：核心不是余弦相似度，而是可度量、可复现的校准排序。

> 当前展示层为**去人名版**：结果只呈现嗓声特征（如「醇厚中低音 · 音色质感接近」），不显示任何真实或虚构歌手姓名。项目**仅供非商业用途**。

---

## 它能做什么

1. 小程序录音 → 上传本机分析服务 → 异步打分 → Top-3 声线结果 + 四维解析报告
2. 质量门限自动拒答（静音、过短、噪声过大）
3. 结果页可生成分享海报；报告页给出音高 / 音色 / 共鸣 / 唱腔四维分位
4. 历史记录仅存本机，可随时清空

处理链路：

```
清唱音频 → 解码/重采样(24k mono) → 对称预处理(LUFS −23, 限幅) → 质量门限(拒答)
   → 自动选窗(10s) → 四维声学特征 + 冻结 HuBERT 嵌入(256d) → 歌手原型
   → Bradley-Terry 校准(cos + 四维差 → 感知相似度) → 同性别池排序 + 归因
   → 嗓声画像（去人名文案）→ CLI 一行 JSON / 小程序展示
```

## 目录结构

| 路径 | 作用 |
|---|---|
| `src/voicematch/` | 可离线运行的 Python 引擎：`audio`（解码/预处理/质量/选窗）、`features`（四维特征+分位）、`embed`（HuBERT）、`library`（版本化库）、`calibrate`（校准排序）、`eval`（评估）、`cli.py`（命令行入口） |
| `server/` | 本机 HTTP 服务：把引擎包装成异步任务接口，供小程序调用。默认只监听 `127.0.0.1` |
| `miniprogram/` | 原生微信小程序，6 个页面（首页/录音/结果/报告/历史/隐私） |
| `tools/` | 素材抓取、建库、演示库生成、盲听基准与评估脚本 |
| `docs/` | 设计文档、校准/配对约束、评估结论与交接说明 |
| `tests/` | 单元测试（`python -m pytest -q`） |
| `material/` | 第三方声线素材与库（**已 gitignore，不在仓库内**） |

详细文档：[训练方法](TRAINING.md) · [引擎与整体设计](docs/HANDOFF.md) · [分析服务接口](server/README.md) · [小程序端](miniprogram/README.md)

## 快速开始

需要 Python ≥ 3.11（本机验证版本 3.13）。

```bash
# 1. 安装依赖（基础数值栈）
pip install -e .

# 2. 启动本机分析服务（默认自动选用 material/ 下的真实库，缺失时回退演示库）
python server/app.py --port 8000

# 3. 自检
curl http://127.0.0.1:8000/health

# 4. 跑一次分析（异步：受理 → 长轮询）
curl -X POST "http://127.0.0.1:8000/compare?gender=female" \
     -H "Content-Type: audio/wav" --data-binary "@query.wav"
# → 202 {"task_id": "...", "state": "queued", ...}
curl "http://127.0.0.1:8000/task/<task_id>?wait=20"
```

小程序端：用微信开发者工具导入 `miniprogram/`，保持 `config.js` 中 `USE_MOCK: false`、`API_BASE: 'http://127.0.0.1:8000'`，并勾选「不校验合法域名」。只想看界面可设 `USE_MOCK: true`，无需后端。

命令行直接比对：

```bash
python -m voicematch.cli compare query.wav --library <库目录> --gender female
```

没有真实素材时可用演示库：`python tools/make_demo_library.py --out server/demo_library`。

## 展示层设计：为什么没有人名

结果文案由 `miniprogram/services/voiceProfile.js` 生成，**只读取声学派生字段**（`style` 声部/风格、`attribution` 四维归因、`dimensions` 本人分位），不使用 `name` / `singer_id` 等身份字段：

- 候选之间的差异来自各自不同的 `style` + `attribution`
- `dimensions` 是查询者本人的四维分位（所有候选共用），只用于描述「你本人的声线」
- 主标签示例：清亮高音 / 醇厚中低音 / 颗粒感声线；报告画像示例：偏高音区 · 偏明亮 · 共鸣适中

这样既保住了「测出结果」的可解释性与传播感，又避开姓名权 / 声音权风险（人格权侵权不以营利为前提，非商用同样需要回避真实姓名）。

## 合规与许可（重要）

- **素材**：默认库来自 **M4Singer**，许可 `CC BY-NC-SA 4.0`（`material/m4singer_3take/catalog/RIGHTS.json` 标记 `commercial_use: false`、`intended_use: research`）。项目定位为**非商业的个人学习与研究**，禁止任何广告变现、付费解锁、带货或引流营利。
- **发布义务**：若日后公开发布供他人使用，需按 CC BY-NC-SA 署名，例如「声线素材来自 M4Singer 数据集，采用 CC BY-NC-SA 4.0 许可，仅用于非商业用途」。
- **商用前提**：若要商业化，必须换成有商业授权的声线库，并取得真实演唱者的书面授权（姓名 + 声音 + 肖像）。
- **服务边界**：本机服务默认只监听回环地址、无鉴权，**不要直接暴露到公网**。

## 状态与已知限制

- 已实现：音频到排序的完整离线链路、版本化歌手库、校准排序、本机异步服务、小程序全流程（含去人名改造）。
- 未启动 / 待完善：真人盲听基准集（「准」的权威判据）、真实 AppID 与主体认证、隐私保护指引过审、HTTPS 备案域名。
- 默认库仅 10 位歌手（男女各 5），Top-3 只表示「在这 10 人里最接近」，不代表全歌手空间的准确率。

具体限制见 [server/README.md](server/README.md) 与 [miniprogram/README.md](miniprogram/README.md) 的「已知限制」小节。
