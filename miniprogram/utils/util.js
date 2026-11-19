// utils/util.js — 通用工具

/** 秒数格式化为 m:ss */
function formatSec(sec) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}

/** 相似度分数转百分比展示（仅 calibrated=true 时才当作百分比，否则只用于排序） */
function scorePercent(score, calibrated) {
  const pct = Math.round((score || 0) * 100);
  return calibrated ? `${pct}%` : `第 ${pct} 位`;
}

/** 时间戳格式化为 MM-DD HH:mm */
function formatTime(ts) {
  const d = new Date(ts || Date.now());
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

module.exports = { formatSec, scorePercent, formatTime };
