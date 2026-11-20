/**
 * 真实引擎服务调用层（异步任务版）
 *
 * 对应 server/app.py：
 *   POST /compare      上传录音 → 202 + task_id（受理，不阻塞等待打分）
 *   GET  /task/<id>    轮询进度 / 取结果
 *   GET  /health       服务自检
 *
 * 为什么要两段式：HuBERT 推理在服务端要几秒，同步接口会把 HTTP 连接一直挂着，
 * 移动网络下容易中途断开（服务端其实已经算完，结果白丢）。改成受理 + 轮询后，
 * 上传成功就能给用户看真实进度，断一次也能继续轮询。
 *
 * 对外仍然返回与 services/mock.js 完全一致的文档（都对齐 voicematch.cli），
 * 页面层不区分数据来源，也不关心背后是同步还是异步。
 *
 * 进度回调签名（第二个参数从旧版的裸数字升级为对象）
 *   onProgress({ phase: 'upload' | 'analyze', pct: 0-100, text: '分析中…' })
 *
 * 失败时统一 reject 一个 Error，并附加：
 *   reason      稳定 token（bad_gender / timeout / network / task_not_found ...）
 *   detail      服务端原文或 wx 的 errMsg，便于排查
 *   statusCode  HTTP 状态码；网络层失败为 0
 */
const config = require('../config.js');

const REASON_TEXT = {
  bad_gender: '性别参数不正确，请返回首页重新选择',
  missing_audio: '没有收到录音数据，请重新录制',
  empty_body: '上传内容为空，请重新录制',
  bad_content_length: '上传内容不完整，请重试',
  body_too_large: '录音文件过大，请缩短到 60 秒以内',
  bad_json: '上传数据格式错误，请重试',
  bad_base64: '音频编码失败，请重新录制',
  bad_multipart: '上传格式错误，请重试',
  no_singers_in_pool: '该性别声线库为空，请联系管理员',
  non_finite_prototype: '声线库数据异常，暂时无法打分',
  library_not_found: '分析服务未加载声线库',
  unknown_encoder: '分析服务编码器配置异常',
  queue_full: '分析服务正忙，请稍后重试',
  task_not_found: '这次分析已过期，请重新提交',
  task_expired: '这次分析已过期，请重新提交',
  cancelled: '已取消',
  not_found: '分析服务接口不存在，请检查 API_BASE',
  timeout: '分析服务响应超时，请重试',
  network: '无法连接分析服务，请确认服务已启动且网络可达',
  bad_response: '分析服务返回了无法解析的数据',
  server_error: '分析服务内部错误，请稍后重试'
};

function makeError(reason, detail, statusCode) {
  const err = new Error(REASON_TEXT[reason] || '分析失败，请稍后重试');
  err.reason = reason || 'unknown';
  err.detail = detail || '';
  err.statusCode = statusCode || 0;
  return err;
}

function classifyNetworkError(errMsg) {
  return /timeout/i.test(String(errMsg || '')) ? 'timeout' : 'network';
}

function parseBody(raw) {
  if (raw && typeof raw === 'object') return raw;
  try {
    return JSON.parse(raw);
  } catch (e) {
    return null;
  }
}

function emit(onProgress, phase, pct, text) {
  if (typeof onProgress !== 'function') return;
  try {
    onProgress({ phase, pct: Math.max(0, Math.min(100, Math.round(pct))), text });
  } catch (e) {
    // 进度回调是 UI 侧代码，出错不能影响主流程
  }
}

/** 排队中把「前面还有几个」说清楚，比单纯"排队中"少很多焦虑。 */
function stageText(info) {
  if (info.state === 'queued') {
    const position = Number(info.queue_position || 0);
    return position > 0 ? `排队中，前面还有 ${position} 个` : '排队中…';
  }
  return info.message || '分析中…';
}

/**
 * 上传录音，拿到 task_id。
 * @param {object} opts 同 compare 的 handlers
 * @returns {Promise<string|object>} 任务 id；同步模式下直接是结果文档
 */
function upload(filePath, gender, opts) {
  const handlers = opts || {};
  return new Promise((resolve, reject) => {
    const task = wx.uploadFile({
      url: `${config.API_BASE}/compare`,
      filePath,
      name: 'file',
      formData: { gender: gender || 'female' },
      timeout: config.UPLOAD_TIMEOUT,
      success(res) {
        const data = parseBody(res.data);
        if (!data) {
          reject(makeError('bad_response', String(res.data || '').slice(0, 200), res.statusCode));
          return;
        }
        if (res.statusCode === 202 && data.task_id) {
          // 把 task_id 交给页面：取消分析时要用它调 DELETE
          if (typeof handlers.onTask === 'function') {
            try {
              handlers.onTask(data.task_id);
            } catch (e) {}
          }
          resolve(data.task_id);
          return;
        }
        // 兼容同步模式（服务端 ?mode=sync）与旧版本服务：直接就给了结果
        if (res.statusCode >= 200 && res.statusCode < 300 && data.status && !data.task_id) {
          resolve(data);
          return;
        }
        reject(makeError(data.reason || 'server_error', data.detail || '', res.statusCode));
      },
      fail(err) {
        const msg = (err && err.errMsg) || '';
        reject(makeError(classifyNetworkError(msg), msg, 0));
      }
    });

    if (task && typeof task.onProgressUpdate === 'function') {
      task.onProgressUpdate((res) => {
        const pct = res.progress || 0;
        emit(handlers.onProgress, 'upload', pct, pct >= 100 ? '上传完成，开始分析…' : `上传中 ${pct}%`);
      });
    }
  });
}

/**
 * 轮询任务直到出结果。
 *
 * 轮询间隔前 6 秒用 500ms（引擎很快时体感更好），之后放慢到 900ms 省请求。
 * 单次请求失败不立刻报错，连续 3 次才判定断网 —— 地铁里信号抖一下不该让整次
 * 分析作废，服务端任务还在跑。
 */
function poll(taskId, handlers) {
  const opts = handlers || {};
  const started = Date.now();
  let failures = 0;

  return new Promise((resolve, reject) => {
    const tick = () => {
      if (typeof opts.isAlive === 'function' && !opts.isAlive()) {
        reject(makeError('cancelled', '页面已关闭', 0));
        return;
      }
      if (Date.now() - started > config.ANALYZE_TIMEOUT) {
        reject(makeError('timeout', `等待超过 ${Math.round(config.ANALYZE_TIMEOUT / 1000)} 秒`, 0));
        return;
      }

      wx.request({
        url: `${config.API_BASE}/task/${taskId}`,
        method: 'GET',
        timeout: config.POLL_TIMEOUT,
        success(res) {
          failures = 0;
          const data = parseBody(res.data);
          if (!data) {
            reject(makeError('bad_response', String(res.data || '').slice(0, 200), res.statusCode));
            return;
          }
          if (res.statusCode === 404) {
            reject(makeError(data.reason || 'task_not_found', data.detail || '', 404));
            return;
          }
          if (res.statusCode >= 500) {
            reject(makeError(data.reason || 'server_error', data.detail || '', res.statusCode));
            return;
          }
          if (data.state === 'done') {
            emit(opts.onProgress, 'analyze', 100, '分析完成');
            resolve(data.result);
            return;
          }
          if (data.state === 'cancelled') {
            reject(makeError('cancelled', '任务已取消', res.statusCode));
            return;
          }
          if (data.state === 'failed') {
            const error = data.error || {};
            reject(makeError(error.reason || 'server_error', error.detail || '', res.statusCode));
            return;
          }

          emit(opts.onProgress, 'analyze', data.progress || 0, stageText(data));
          const elapsed = Date.now() - started;
          setTimeout(tick, elapsed < 6000 ? 500 : 900);
        },
        fail(err) {
          failures += 1;
          if (failures >= 3) {
            const msg = (err && err.errMsg) || '';
            reject(makeError(classifyNetworkError(msg), msg, 0));
            return;
          }
          setTimeout(tick, 800);
        }
      });
    };
    tick();
  });
}

/**
 * 上传录音并打分。
 * @param {string} filePath 录音临时文件路径
 * @param {string} gender 'female' | 'male'
 * @param {object} [handlers]
 * @param {function} [handlers.onProgress] 进度回调，参数 { phase, pct, text }
 * @param {function} [handlers.isAlive]    返回 false 时停止轮询（页面已卸载）
 * @returns {Promise<object>} 与 CLI 同构的结果文档
 */
async function compare(filePath, gender, handlers) {
  const opts = handlers || {};
  if (!filePath) {
    throw makeError('missing_audio', 'filePath 为空', 0);
  }

  const task = await upload(filePath, gender, opts);
  // 同步模式的兼容返回：已经是结果文档
  if (typeof task !== 'string') {
    emit(opts.onProgress, 'analyze', 100, '分析完成');
    return task;
  }
  return poll(task, opts);
}

/**
 * 取消服务端任务。页面已经停止轮询，这里只是让后台别白算。
 * 无论成功失败都 resolve —— 取消失败不该把用户堵在页面上。
 * @returns {Promise<object|null>} 服务端任务快照
 */
function cancelTask(taskId) {
  return new Promise((resolve) => {
    if (!taskId) {
      resolve(null);
      return;
    }
    wx.request({
      url: `${config.API_BASE}/task/${taskId}`,
      method: 'DELETE',
      timeout: config.POLL_TIMEOUT,
      success(res) {
        resolve(parseBody(res.data) || null);
      },
      fail() {
        resolve(null);
      }
    });
  });
}

/** 服务自检，返回库版本、可用声线数与队列现状。 */
function health() {
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${config.API_BASE}/health`,
      method: 'GET',
      timeout: config.HEALTH_TIMEOUT,
      success(res) {
        const data = parseBody(res.data);
        if (res.statusCode === 200 && data && data.status === 'ok') {
          resolve(data);
          return;
        }
        reject(makeError((data && data.reason) || 'server_error',
          (data && data.detail) || '', res.statusCode));
      },
      fail(err) {
        reject(makeError(classifyNetworkError(err && err.errMsg),
          (err && err.errMsg) || '', 0));
      }
    });
  });
}

module.exports = { compare, cancelTask, health, REASON_TEXT };
