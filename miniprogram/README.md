# 小程序端（VoiceMatch / 声纹研究所）

原生微信小程序，主包 6 个页面。录音 → 上传分析 → Top-3 歌手 + 可解释报告。

## 目录结构

```
miniprogram/
├── app.js / app.json / app.wxss     全局：隐私授权、结果暂存、版本更新、异常兜底
├── config.js                        唯一的运行配置（USE_MOCK / API_BASE / 三类超时）
├── sitemap.json                     只允许首页被搜索收录
├── services/
│   ├── api.js                       上传、两段式进度、轮询、错误分类、取消、health
│   ├── history.js                   历史记录（本机 storage，上限 20 条）
│   └── mock.js                      离线预览用的模拟结果（USE_MOCK=true 时生效）
├── utils/util.js                    时长格式化、分数展示
└── pages/
    ├── index/     首页：性别选择、隐私弹窗、历史/反馈入口
    ├── record/    录音页：权限 → 录音 → 试听 → 上传/分析进度 → 取消
    ├── result/    结果页：Top-3、质量拒答、错误重试、分享海报、朋友圈
    ├── report/    报告页：四维分位、归因说明
    ├── history/   历史页：回看 / 单删 / 清空
    └── privacy/   隐私说明页：用途、权利、清除本机数据、打开官方指引
```

## 本地跑起来

1. 启动分析服务（见 `../server/README.md`）：

   ```bash
   C:\Python\python.exe server/app.py --port 8000
   ```

   `netstat -ano | grep :8000` 确认只有一个 PID——Windows 允许两个进程同时监听
   同一端口，请求会随机落到旧进程，表现为「改了代码没生效」。

2. 开发者工具导入本目录，「详情 → 本地设置」勾选
   **不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书**。
3. `config.js` 保持 `USE_MOCK: false`、`API_BASE: 'http://127.0.0.1:8000'`。

只想看界面、不起后端：把 `USE_MOCK` 改成 `true`，结果来自 `services/mock.js`。

### 真机预览

- 服务改 `--host 0.0.0.0`（启动日志会直接打印手机可访问的地址和 `config.js` 改法）；
- `API_BASE` 换成电脑的局域网 IP；
- 手机与电脑同一 Wi-Fi，Windows 防火墙放行 `python.exe` 的 8000 端口。

## 接口契约

`services/api.js` 对齐 `server/app.py` 的异步任务模型：

| 步骤 | 接口 | 说明 |
|---|---|---|
| 上传 | `POST /compare`（multipart: `file` + `gender`） | 返回 `202 {task_id, state:"queued"}` |
| 轮询 | `GET /task/<id>?wait=<秒>` | `state` 走 queued → running → done/failed/cancelled，带 `progress` |
| 取消 | `DELETE /task/<id>` | 排队中直接摘队；运行中结果丢弃 |
| 自检 | `GET /health` | 歌手数、模型、`calibrated`、队列水位 |

进度条上传占 30%、分析占 70%，阶段文案用服务端返回的 `message`（排队时显示
「前面还有 N 个」），不是匀速假动画。

## 上线前必须完成（代码替代不了）

1. **真实 AppID + 主体认证 + 服务类目**：`project.config.json` 当前是 `touristappid`，游客号无法真机预览、上传、配置域名。
2. **后台配置《用户隐私保护指引》并过审**，声明麦克风（录音）用途。未配置时
   `wx.getPrivacySetting` 恒返回 `needAuthorization=false`，官方授权弹窗不会出现
   —— 这是平台机制，不是代码问题。
3. **HTTPS + 备案域名**：`request` / `uploadFile` 合法域名必须配置，`API_BASE` 换成正式域名。
4. **商用歌手库**：当前 `material/m4singer_3take` 是 CC BY-NC-SA、歌手匿名，只能内部验证；上线需要商业授权素材或自采集 + 演唱者书面同意。

## 隐私处理要点

- 声音属敏感个人信息：首页弹窗单独同意（`open-type="agreePrivacyAuthorization"`，微信侧留痕），录音页再兜底一次官方授权。
- 从历史页直达录音页会绕过首页弹窗，因此 `record.js` 在申请麦克风前调用 `ensurePrivacy()`。
- 低版本基础库无官方隐私能力时降级为本地标记，不阻断使用。
- 录音临时文件不落库；历史记录只存结构化结果，隐私页可一键清空。

## 已知限制

- 历史记录仅本机 storage，换设备或清缓存即丢；云端同步需要用户体系。
- 海报保存到相册后由用户手动转发；一键分享图片需 `wx.showShareImageMenu`（≥2.14.4）。
- 服务端单工作线程串行推理，并发靠排队；结果在内存，服务重启后旧 `task_id` 返回 404。
