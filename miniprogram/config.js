/**
 * 小程序运行配置
 *
 * USE_MOCK
 *   true  → 使用 services/mock.js 的模拟结果，不依赖后端，适合看界面
 *   false → 调用 API_BASE 上的真实引擎服务（server/app.py）
 *
 * API_BASE
 *   本机联调：开发者工具里用 http://127.0.0.1:8000，并在「详情 → 本地设置」
 *   勾选「不校验合法域名、web-view（业务域名）、TLS 版本以及 HTTPS 证书」。
 *   真机预览：改成电脑的局域网 IP（如 http://192.168.1.10:8000），手机与电脑
 *   需在同一网络，且服务启动时 --host 0.0.0.0。
 *   正式上线：必须是已在小程序后台配置的 HTTPS 域名。
 *
 * 三个超时是分开的，因为三段的失败语义不同：
 *   UPLOAD_TIMEOUT   读本地文件 + 传输，60 秒录音约几百 KB
 *   POLL_TIMEOUT     单次查询任务状态，只该是几十毫秒
 *   ANALYZE_TIMEOUT  从上传开始到出结果的总预算，超时让用户重试而不是无限转圈
 */
module.exports = {
  // false = 调用真实引擎（需先启动 server/app.py）；true = 用模拟结果只看界面
  USE_MOCK: false,
  API_BASE: 'http://127.0.0.1:8000',
  UPLOAD_TIMEOUT: 30000,
  POLL_TIMEOUT: 8000,
  // 引擎排队 + 推理的总上限。真机弱网下可以再放宽，但别超过 2 分钟——
  // 用户等不了那么久，不如让他重录一段更干净的
  ANALYZE_TIMEOUT: 90000,
  // 健康检查只做一次 HTTP 往返，不能等太久，否则首屏被拖住
  HEALTH_TIMEOUT: 3000
};
