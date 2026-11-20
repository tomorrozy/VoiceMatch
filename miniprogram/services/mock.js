/**
 * Mock 数据服务层
 *
 * 数据结构严格对齐后端 voicematch.cli 的“一行 JSON”输出（见 cli.py::_compare），
 * 这样把 mockCompare() 换成真实的 wx.request / 云函数调用时，页面层无需改动。
 *
 * 对齐字段：status / results[{singer_id,name,score,calibrated,attribution}]
 *          quality{speech_sec,snr_db,clipped_ratio,voiced_ratio,rejected,reasons}
 *          model_id / library_version / calibration_version / calibrated / encoder
 *          query{path,gender} / assumed / truncated / reason
 *          pairing{status,single_material_singers,notice} —— R29 库配对状态
 *
 * 扩展字段（仅前端展示用，真实后端暂不返回，接入时做兼容默认值）：
 *          results[].dimensions —— 四维分位（音高/音色/共鸣/唱腔，0-100）
 *          results[].attributionText —— 归因的中文可读文案
 */

// 示例声线档案（纯嗓声特征，无任何人名；仅用于离线预览）
// name 字段保留为空字符串仅为结构对齐，展示层一律走 voiceProfile 生成的嗓声标签。
const SINGERS = {
  female: [
    { singer_id: 'F001', name: '', style: '流行 · 中低音' },
    { singer_id: 'F002', name: '', style: '民谣 · 气声' },
    { singer_id: 'F003', name: '', style: 'R&B · 高音' },
    { singer_id: 'F004', name: '', style: '流行 · 明亮' },
    { singer_id: 'F005', name: '', style: '抒情 · 醇厚' }
  ],
  male: [
    { singer_id: 'M001', name: '', style: '流行 · 中音' },
    { singer_id: 'M002', name: '', style: '摇滚 · 颗粒感' },
    { singer_id: 'M003', name: '', style: '民谣 · 低音' },
    { singer_id: 'M004', name: '', style: 'R&B · 假声' },
    { singer_id: 'M005', name: '', style: '抒情 · 高音' }
  ]
};

// 四维分位模板（0-100）。真实值由 features/percentile.py 在同性别池内计算。
const DIM_KEYS = ['pitch', 'timbre', 'resonance', 'style'];
const DIM_LABELS = {
  pitch: '音高',
  timbre: '音色',
  resonance: '共鸣',
  style: '唱腔'
};

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function round(n, digits = 2) {
  const p = Math.pow(10, digits);
  return Math.round(n * p) / p;
}

/**
 * 生成一条 Mock 结果，字段结构对齐 CLI。
 * @param {string} gender 'female' | 'male'
 * @param {number} seed 简单伪随机种子，让同一性别多次测试结果稳定可复现
 */
function mockCompare(gender, seed = 7) {
  const pool = SINGERS[gender] || SINGERS.female;
  const rand = (i) => {
    const x = Math.sin((seed + i) * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  };

  // Top-3：分数递减，模拟校准后的感知相似度（非概率百分比）
  const scoreBase = [0.86, 0.79, 0.73];
  const results = pool.slice(0, 3).map((singer, i) => {
    const jitter = (rand(i) - 0.5) * 0.04;
    const score = round(clamp(scoreBase[i] + jitter, 0, 1), 3);

    // 四维分位：每位歌手一种“像”的侧重，用于报告页解释
    const dims = {
      pitch: Math.round(clamp(52 + rand(i * 4 + 1) * 40, 0, 100)),
      timbre: Math.round(clamp(58 + rand(i * 4 + 2) * 40, 0, 100)),
      resonance: Math.round(clamp(45 + rand(i * 4 + 3) * 45, 0, 100)),
      style: Math.round(clamp(50 + rand(i * 4 + 4) * 45, 0, 100))
    };

    // 归因：四维贡献占比（合计 1.0），对齐 CLI 的 attribution 语义
    const raw = DIM_KEYS.map((k, j) => 0.1 + rand(i * 9 + j) * 0.4);
    const sum = raw.reduce((a, b) => a + b, 0);
    const attribution = {};
    DIM_KEYS.forEach((k, j) => { attribution[k] = round(raw[j] / sum, 3); });

    // 找贡献最大的维度，生成可读归因文案
    const topDim = DIM_KEYS.reduce((a, b) => attribution[a] >= attribution[b] ? a : b);
    const attributionText = `主要因为${DIM_LABELS[topDim]}接近`;

    return {
      singer_id: singer.singer_id,
      name: singer.name,
      style: singer.style,
      score: score,
      calibrated: true, // 校准未就绪时后端会置 false，此时分数只能排序、不可当百分比
      attribution: attribution,
      attributionText: attributionText,
      dimensions: dims
    };
  });

  return {
    status: 'ok',
    reason: null,
    calibrated: true,
    encoder: 'hubert',
    model_id: 'hubert-base-960h-frozen',
    library_version: 'pipeline-v2+lib-mock-001',
    calibration_version: 'calib-mock-001',
    query: { gender: gender, path: 'mock://recorder.tmp' },
    assumed: 'a_cappella',
    truncated: false,
    // R29：库的配对状态。Mock 代表"合规库"，notice 为空即不显示提示
    pairing: { status: 'cross_recording', single_material_singers: [], notice: '' },
    quality: {
      speech_sec: round(11.4 + rand(99) * 3, 1),
      snr_db: round(18 + rand(98) * 6, 1),
      clipped_ratio: 0,
      voiced_ratio: round(0.72 + rand(97) * 0.15, 2),
      rejected: false,
      reasons: []
    },
    results: results
  };
}

/**
 * 生成一条“质量拒答”结果，用于演示 CLI 的 status=rejected 分支（G5：会拒答）。
 */
function mockRejected(reason) {
  const reasonMap = {
    short: '有效发声时长不足 8 秒，请用正常音量清唱一段再试',
    noisy: '环境噪声过大，请到安静的地方重新录制',
    silent: '没有检测到人声，请检查麦克风后重试'
  };
  const key = reason || 'short';
  return {
    status: 'rejected',
    reason: key,
    calibrated: false,
    encoder: 'hubert',
    model_id: 'hubert-base-960h-frozen',
    library_version: 'pipeline-v2+lib-mock-001',
    calibration_version: 'calib-mock-001',
    query: { gender: 'female', path: 'mock://recorder.tmp' },
    assumed: 'a_cappella',
    truncated: false,
    pairing: { status: 'cross_recording', single_material_singers: [], notice: '' },
    quality: {
      speech_sec: 4.2,
      snr_db: 9.5,
      clipped_ratio: 0.01,
      voiced_ratio: 0.31,
      rejected: true,
      reasons: [key, reasonMap[key]]
    },
    results: []
  };
}

module.exports = {
  mockCompare,
  mockRejected,
  DIM_LABELS,
  DIM_KEYS
};
