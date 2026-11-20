/**
 * 嗓声画像：把后端结果翻译成"描述声音本身"的文案，不含任何歌手姓名。
 *
 * 设计依据（去人名 / 合规）：
 * - 结果只依赖声学派生字段 —— style（声部或风格标签）、attribution（四维贡献）、
 *   dimensions（查询者本人的四维分位）。绝不使用 name / singer_id 等身份字段。
 * - 每个候选的差异化来自 style + attribution；这两个字段在真实后端里逐候选不同。
 * - dimensions 是查询者本人的分位（所有候选共用），只用来描述"你本人的声线"，
 *   不参与候选之间的区分。
 *
 * 真实库 style：soprano / alto / tenor / bass（声部代号）
 * Mock 库 style：形如 "流行 · 中低音"、"民谣 · 气声"（风格 · 音区）
 * 两种格式都做兼容解析。
 */

// 声部代号 → 嗓声基调（真实库）
const VOICE_ROLE = {
  soprano: { tag: '清亮高音', tone: '明亮通透、位置偏高', register: 'high' },
  alto: { tag: '醇厚中低音', tone: '温润饱满、叙事感强', register: 'lowmid' },
  tenor: { tag: '通透男高音', tone: '明亮有张力、穿透力好', register: 'high' },
  bass: { tag: '沉稳低音', tone: '厚实低沉、胸腔共鸣明显', register: 'low' }
};

// 中文风格/音区关键词 → 嗓声基调（Mock 库 + 中文 style 兼容）
// 顺序即优先级，先匹配到的生效。
const TONE_RULES = [
  { re: /颗粒|沙哑|烟嗓|摇滚/, tag: '颗粒感声线', tone: '带颗粒与沙哑质感、边缘清晰', register: 'mid' },
  { re: /气声/, tag: '气声质感', tone: '气声比例高、轻盈偏软', register: 'mid' },
  { re: /假声|海豚|哨音/, tag: '轻盈假声', tone: '假声区轻盈、头腔集中', register: 'high' },
  { re: /醇厚|浑厚|温暖|抒情|中低音/, tag: '醇厚中低音', tone: '温润厚实、低频饱满', register: 'lowmid' },
  { re: /低音|低沉/, tag: '沉稳低音', tone: '低沉稳重、胸腔共鸣明显', register: 'low' },
  { re: /清亮|明亮|通透/, tag: '清亮高音', tone: '明亮通透、位置偏高', register: 'high' },
  { re: /高音/, tag: '高音区', tone: '音域偏高、集中度高', register: 'high' },
  { re: /R&B|r&b|转音|灵动/, tag: '灵动转音', tone: '转音灵活、装饰音丰富', register: 'mid' },
  { re: /民谣|质朴/, tag: '质朴民谣嗓', tone: '自然质朴、不加修饰', register: 'mid' },
  { re: /中音|流行/, tag: '中音区', tone: '音区居中、自然平衡', register: 'mid' }
];

// 主导维度 → 归因短语（attribution 中占比最大的维度）
const DIM_REASON = {
  pitch: '音高走向接近',
  timbre: '音色质感接近',
  resonance: '共鸣方式接近',
  style: '唱腔习惯接近'
};
const DIM_LABEL = {
  pitch: '音高', timbre: '音色', resonance: '共鸣', style: '唱腔'
};
const DIM_ORDER = ['pitch', 'timbre', 'resonance', 'style'];

function normStyle(style) {
  return String(style || '').trim().toLowerCase();
}

/** 解析 style → { tag, tone, register, matched } */
function parseStyle(style, gender) {
  const s = String(style || '').trim();
  const key = normStyle(style);

  // 1) 真实库声部代号（精确匹配）
  if (VOICE_ROLE[key]) {
    return { ...VOICE_ROLE[key], matched: true };
  }

  // 2) 中文风格 / 音区关键词
  for (const rule of TONE_RULES) {
    if (rule.re.test(s)) {
      return { tag: rule.tag, tone: rule.tone, register: rule.register, matched: true };
    }
  }

  // 3) 兜底：仅按性别给一个中性描述（不含身份信息）
  return gender === 'male'
    ? { tag: '男声声线', tone: '自然男声、质感居中', register: 'mid', matched: false }
    : { tag: '女声声线', tone: '自然女声、质感居中', register: 'mid', matched: false };
}

/** 找 attribution 中占比最大的维度 */
function topDimension(attribution) {
  if (!attribution || typeof attribution !== 'object') return null;
  let best = null;
  let bestVal = -1;
  for (const k of DIM_ORDER) {
    const v = attribution[k];
    if (typeof v === 'number' && isFinite(v) && v > bestVal) {
      bestVal = v;
      best = k;
    }
  }
  return best;
}

/**
 * 为一个候选结果生成嗓声描述。
 * @param {object} item results[] 中的一项（含 style / attribution）
 * @param {string} gender 'female' | 'male'
 * @returns {{tag:string, tone:string, reason:string, reasonDim:string|null, detail:string, title:string}}
 *   tag    —— 短标签，用于榜单行首
 *   tone   —— 嗓音质感描述
 *   reason —— "音高走向接近"这类归因短语
 *   detail —— "音色质感接近 · 明亮通透、位置偏高"完整一句话
 *   title  —— 用于大标题/分享/海报的主标语
 */
function describeVoice(item, gender) {
  const base = parseStyle(item && item.style, gender);
  const dim = topDimension(item && item.attribution);
  const reason = dim ? DIM_REASON[dim] : '整体声线接近';
  const detail = `${reason} · ${base.tone}`;
  return {
    tag: base.tag,
    tone: base.tone,
    register: base.register,
    reason,
    reasonDim: dim,
    detail,
    // 主标语：声部/风格标签 + 归因，信息量足够且不含姓名
    title: `${base.tag} · ${reason}`
  };
}

/**
 * 用查询者本人的四维分位生成定性声线画像（报告页顶部/副标题）。
 * @param {object} dims {pitch,timbre,resonance,style} 各 0-100，缺失为 null
 * @returns {{line:string, parts:Array<{label,text}>}}
 */
function profileFromDims(dims) {
  const d = dims || {};
  const val = (k) => (typeof d[k] === 'number' && isFinite(d[k])) ? d[k] : null;

  const band = (v, low, mid, high) => (v == null ? null : (v < 33 ? low : (v > 66 ? high : mid)));
  const map = {
    pitch: { low: '偏低音区', mid: '中音区', high: '偏高音区' },
    timbre: { low: '偏醇厚', mid: '中性', high: '偏明亮' },
    resonance: { low: '共鸣偏内敛', mid: '共鸣适中', high: '共鸣饱满' },
    style: { low: '唱腔偏简洁', mid: '唱腔适中', high: '装饰音丰富' }
  };

  const parts = DIM_ORDER.map((k) => {
    const v = val(k);
    const b = band(v, map[k].low, map[k].mid, map[k].high);
    return {
      key: k,
      label: DIM_LABEL[k],
      value: v,
      text: b || '—'
    };
  });

  const usable = parts.filter((p) => p.text !== '—').map((p) => p.text);
  const line = usable.length ? usable.join(' · ') : '声线特征数据不足';
  return { line, parts };
}

module.exports = {
  describeVoice,
  profileFromDims,
  DIM_LABEL,
  DIM_REASON
};
