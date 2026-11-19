/**
 * 历史记录：只存本机 storage，最多 20 条。
 *
 * 每条记录保留完整 result 文档，这样在模型/歌手库升级后仍能回看当时的结果。
 * 单条约 2 KB，20 条远低于小程序 10 MB 的 storage 上限。
 *
 * 记什么：ok / rejected / no_usable_window。拒答也保留，用户能回看“上次为什么没过”。
 * 不记：status=error（网络或服务异常，不是一次有效的测试结果）。
 */

const KEY = 'voicematch_history';
const LIMIT = 20;

function read() {
  try {
    const list = wx.getStorageSync(KEY);
    return Array.isArray(list) ? list : [];
  } catch (e) {
    return [];
  }
}

function write(list) {
  try {
    wx.setStorageSync(KEY, list);
  } catch (e) {
    console.error('history write failed:', e);
  }
}

/**
 * @param {object} result 与 CLI 同构的结果文档
 * @param {object} [meta] { gender, seconds } 录音上下文
 * @returns {object|null} 保存的记录；不应保存时返回 null
 */
function add(result, meta) {
  if (!result || result.status === 'error') return null;

  const info = meta || {};
  const top = result.results && result.results[0] ? result.results[0] : null;
  const pairing = result.pairing || null;
  const pairingOk = !pairing || pairing.status === 'cross_recording';

  const item = {
    id: `${Date.now()}-${Math.floor(Math.random() * 1000)}`,
    createdAt: Date.now(),
    status: result.status,
    gender: info.gender || (result.query && result.query.gender) || 'female',
    seconds: info.seconds || 0,
    name: top ? top.name : '',
    style: top ? (top.style || '') : '',
    score: top ? top.score : 0,
    calibrated: !!result.calibrated,
    // 校准合规且跨录音配对时，分数才可当百分比展示
    scoreAsPercent: !!result.calibrated && pairingOk,
    model_id: result.model_id || '',
    library_version: result.library_version || '',
    calibration_version: result.calibration_version || '',
    result
  };

  write([item].concat(read()).slice(0, LIMIT));
  return item;
}

function list() {
  return read();
}

function get(id) {
  return read().find((item) => item.id === id);
}

function remove(id) {
  const next = read().filter((item) => item.id !== id);
  write(next);
  return next;
}

function clear() {
  try {
    wx.removeStorageSync(KEY);
  } catch (e) {
    console.error('history clear failed:', e);
  }
}

module.exports = { add, list, get, remove, clear };
