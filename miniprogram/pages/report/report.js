// pages/report/report.js
const app = getApp();

// 四维文案配置（对应 features/percentile.py 的四个维度语义）
const DIM_META = {
  pitch: { label: '音高', icon: '🎵', color: '#7c6cff', desc: '你在同性别池中的音高位置，越高说明音域偏向高音区' },
  timbre: { label: '音色', icon: '🎨', color: '#4de0d0', desc: '明亮度与厚度，反映声音的"质感"，是最像一个人的维度' },
  resonance: { label: '共鸣', icon: '💫', color: '#ff7eb6', desc: '胸腔/口腔/头腔的共鸣倾向，影响声音的饱满度' },
  style: { label: '唱腔', icon: '🎤', color: '#ffb45a', desc: '颤音、滑音与断句习惯，是"怎么唱"的个性特征' }
};

Page({
  data: {
    result: null,
    top: null,
    dims: [],
    attrs: [],
    calibrated: true,
    // R29：库配对状态不满足时，报告顶部说明分数性质（文案由服务端给）
    pairingNotice: ''
  },

  onLoad() {
    const result = app.getResult();
    if (!result || !result.results || !result.results.length) {
      wx.showToast({ title: '暂无报告，请先测试', icon: 'none' });
      setTimeout(() => wx.navigateBack(), 1200);
      return;
    }
    const top = result.results[0];

    // 四维分位（真实后端返回 null 表示该维度没有可用测量，显示为 -- 而不是 0）
    const dims = Object.keys(DIM_META).map((key) => {
      const meta = DIM_META[key];
      const raw = top.dimensions ? top.dimensions[key] : null;
      const usable = typeof raw === 'number' && isFinite(raw);
      return {
        key,
        ...meta,
        value: usable ? Math.round(raw) : null,
        width: usable ? Math.max(0, Math.min(100, raw)) : 0,
        display: usable ? String(Math.round(raw)) : '--'
      };
    });

    // 归因占比（CLI 的 attribution 字段）
    const attrs = Object.keys(DIM_META).map((key) => {
      const meta = DIM_META[key];
      const ratio = (top.attribution && top.attribution[key] != null) ? top.attribution[key] : 0.25;
      return { key, label: meta.label, color: meta.color, pct: Math.round(ratio * 100) };
    });

    this.setData({
      result,
      top,
      dims,
      attrs,
      calibrated: !!result.calibrated,
      // 报告页可截图外传，库的局限必须在图里可见，不能只靠结果页提示。
      pairingNotice: (result.pairing && result.pairing.notice) || ''
    });
  },

  onSaveTip() {
    wx.showToast({ title: '可长按页面截图保存', icon: 'none' });
  },

  onShareAppMessage() {
    return {
      title: `我的声音解析报告：最像${this.data.top ? this.data.top.name : '某位歌手'}`,
      path: '/pages/index/index'
    };
  },

  onShareTimeline() {
    return {
      title: `我的声音解析报告：最像${this.data.top ? this.data.top.name : '某位歌手'}`
    };
  }
});
