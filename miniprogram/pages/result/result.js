// pages/result/result.js
const app = getApp();
const { scorePercent } = require('../../utils/util.js');
const { REASON_TEXT } = require('../../services/api.js');
const { describeVoice, profileFromDims } = require('../../services/voiceProfile.js');

// 质量拒答的稳定 token（见 src/voicematch/audio/quality.py）→ 用户可读文案
const QUALITY_REASON_TEXT = {
  too_short_speech: '有效发声不足 8 秒，请用正常音量完整唱一句',
  clipped: '音量过大导致破音，请离麦克风稍远一些',
  low_snr: '环境噪声过大，请到安静的地方重新录制',
  low_voiced_ratio: '检测到的有效人声太少，请靠近麦克风、连贯地唱'
};

function roundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + w, y, x + w, y + h, radius);
  ctx.arcTo(x + w, y + h, x, y + h, radius);
  ctx.arcTo(x, y + h, x, y, radius);
  ctx.arcTo(x, y, x + w, y, radius);
  ctx.closePath();
}

Page({
  data: {
    result: null,
    rejectReason: '',
    errorInfo: null,
    // R29：库能否支撑跨录音校准。为 false 时分数只当排序，不显示百分比。
    scoreAsPercent: false,
    // 后端给的一句中文说明（合规时为空串，不显示提示条）
    pairingNotice: '',
    posterSaving: false
  },

  onLoad() {
    const result = app.getResult();
    if (!result) {
      wx.showToast({ title: '暂无结果，请重新测试', icon: 'none' });
      setTimeout(() => wx.navigateBack(), 1200);
      return;
    }

    // 服务端错误信封：没有 quality / results，不能走拒答分支
    if (result.status === 'error') {
      const reason = result.reason || 'server_error';
      this.setData({
        result,
        errorInfo: {
          message: REASON_TEXT[reason] || '分析服务返回异常，请稍后重试',
          detail: result.detail || ''
        }
      });
      return;
    }

    // R29：库若只有单份录音（或由旧版本构建、无法验证），校准分数不能当
    // 百分比展示——这正是"calibrated 为 true 也可能是泄漏"的那类库。旧版服务
    // 与 mock 没有 pairing 字段时按合规处理，保持向后兼容。
    const pairing = result.pairing || null;
    const pairingOk = !pairing || pairing.status === 'cross_recording';
    const scoreAsPercent = !!result.calibrated && pairingOk;
    const pairingNotice = (result.status === 'ok' && pairing && pairing.notice) || '';

    // 分数展示：calibrated=true 且配对合规，才当百分比（对齐 CLI 不变量）
    // 同时给每个候选附嗓声画像：只依赖 style/attribution，不含任何歌手姓名。
    if (result.results && result.results.length) {
      const gender = (result.query && result.query.gender) || app.globalData.gender || 'female';
      const dims = result.results[0] && result.results[0].dimensions;
      result.voiceLine = profileFromDims(dims).line;
      result.results = result.results.map((item) => {
        const voice = describeVoice(item, gender);
        return {
          ...item,
          voice,
          scoreText: scorePercent(item.score, scoreAsPercent)
        };
      });
    }

    let rejectReason = '';
    if (result.status === 'rejected') {
      const reasons = (result.quality && result.quality.reasons) || [];
      const texts = [];
      reasons.forEach((token) => {
        const text = QUALITY_REASON_TEXT[token];
        if (text && texts.indexOf(text) < 0) texts.push(text);
      });
      rejectReason = texts.length ? texts.join('；') : '录音质量未达标，请重新录制';
    } else if (result.status === 'no_usable_window') {
      rejectReason = '录音通过了质量检查，但没有找到足够稳定的发声片段用于打分，请唱得更连贯一些。';
    }

    this.setData({ result, rejectReason, scoreAsPercent, pairingNotice });
  },

  onReport() {
    wx.navigateTo({ url: '/pages/report/report' });
  },

  onRetry() {
    // 录音页还在栈里，返回即可保留刚才的录音，用户可直接重试
    wx.navigateBack({ delta: 1 });
  },

  onBackHome() {
    wx.reLaunch({ url: '/pages/index/index' });
  },

  onAgain() {
    // 回到录音页（关闭结果页再回退，避免页面栈堆积）
    const pages = getCurrentPages();
    const recordPage = pages.find((p) => p.route === 'pages/record/record');
    if (recordPage) {
      wx.navigateBack({ delta: pages.length - 1 - pages.indexOf(recordPage) });
    } else {
      wx.redirectTo({ url: '/pages/record/record' });
    }
  },

  /**
   * 生成分享海报：canvas 2d 离屏绘制 → 导出临时文件 → 保存相册。
   * 只对 status=ok 开放；拒答/错误态没有可分享的结果。
   */
  async onSavePoster() {
    if (this.data.posterSaving) return;
    const result = this.data.result;
    if (!result || result.status !== 'ok' || !result.results || !result.results.length) {
      wx.showToast({ title: '这次没有可分享的结果', icon: 'none' });
      return;
    }

    this.setData({ posterSaving: true });
    wx.showLoading({ title: '生成海报…', mask: true });
    try {
      const canvas = await this.getPosterCanvas();
      if (!canvas) throw new Error('canvas not ready');

      const info = wx.getWindowInfo ? wx.getWindowInfo() : wx.getSystemInfoSync();
      const dpr = Math.min(info.pixelRatio || 2, 3);
      const W = 750;
      const H = 1000;

      canvas.width = W * dpr;
      canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);

      this.drawPoster(ctx, W, H);

      const filePath = await new Promise((resolve, reject) => {
        wx.canvasToTempFilePath({
          canvas,
          x: 0,
          y: 0,
          width: W,
          height: H,
          destWidth: W * 2,
          destHeight: H * 2,
          success: (res) => resolve(res.tempFilePath),
          fail: reject
        });
      });

      await this.saveToAlbum(filePath);
      wx.showToast({ title: '已保存到相册', icon: 'success' });
    } catch (err) {
      console.error('poster failed:', err);
      if (!err || !err._handled) {
        wx.showToast({ title: '生成失败，请重试', icon: 'none' });
      }
    } finally {
      wx.hideLoading();
      this.setData({ posterSaving: false });
    }
  },

  getPosterCanvas() {
    return new Promise((resolve) => {
      wx.createSelectorQuery()
        .in(this)
        .select('#posterCanvas')
        .fields({ node: true, size: true })
        .exec((res) => resolve(res && res[0] ? res[0].node : null));
    });
  },

  saveToAlbum(filePath) {
    return new Promise((resolve, reject) => {
      if (!wx.saveImageToPhotosAlbum) {
        wx.showToast({ title: '当前版本不支持保存相册', icon: 'none' });
        reject({ _handled: true });
        return;
      }
      wx.saveImageToPhotosAlbum({
        filePath,
        success: resolve,
        fail: (err) => {
          const msg = (err && err.errMsg) || '';
          if (/auth deny|auth denied|authorize no response/i.test(msg)) {
            err._handled = true;
            wx.showModal({
              title: '需要相册权限',
              content: '保存海报需要你授权“保存到相册”，可在设置中开启。',
              confirmText: '去设置',
              success: (r) => {
                if (r.confirm && wx.openSetting) wx.openSetting();
              }
            });
          }
          reject(err);
        }
      });
    });
  },

  drawPoster(ctx, W, H) {
    const result = this.data.result;
    const top = result.results[0];

    const bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, '#151632');
    bg.addColorStop(0.55, '#1c1d45');
    bg.addColorStop(1, '#0f1029');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    const glowA = ctx.createRadialGradient(120, 80, 0, 120, 80, 360);
    glowA.addColorStop(0, 'rgba(124,108,255,0.35)');
    glowA.addColorStop(1, 'rgba(124,108,255,0)');
    ctx.fillStyle = glowA;
    ctx.fillRect(0, 0, W, 520);

    const glowB = ctx.createRadialGradient(650, 140, 0, 650, 140, 300);
    glowB.addColorStop(0, 'rgba(77,224,208,0.22)');
    glowB.addColorStop(1, 'rgba(77,224,208,0)');
    ctx.fillStyle = glowB;
    ctx.fillRect(320, 0, W - 320, 520);

    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '24px sans-serif';
    ctx.fillText('VOICEMATCH · 声线报告', 80, 86);

    ctx.fillStyle = '#ffffff';
    ctx.font = '34px sans-serif';
    ctx.fillText('我的声线类型', 80, 160);

    const voice = top.voice || describeVoice(top, app.globalData.gender);
    const label = voice.tag || '声线';
    let nameSize = 82;
    ctx.font = `bold ${nameSize}px sans-serif`;
    while (ctx.measureText(label).width > 590 && nameSize > 40) {
      nameSize -= 4;
      ctx.font = `bold ${nameSize}px sans-serif`;
    }
    const nameGrad = ctx.createLinearGradient(80, 220, 680, 280);
    nameGrad.addColorStop(0, '#4de0d0');
    nameGrad.addColorStop(1, '#a99dff');
    ctx.fillStyle = nameGrad;
    ctx.fillText(label, 80, 270);

    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '26px sans-serif';
    ctx.fillText(voice.reason || '', 80, 320);

    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.font = '28px sans-serif';
    ctx.fillText(this.data.scoreAsPercent ? '感知相似度' : '排序', 80, 390);

    ctx.fillStyle = '#4de0d0';
    ctx.font = 'bold 64px sans-serif';
    ctx.fillText(top.scoreText || '', 80, 462);

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(80, 510);
    ctx.lineTo(670, 510);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '24px sans-serif';
    ctx.fillText('最接近的 3 种声线', 80, 560);

    (result.results || []).slice(0, 3).forEach((item, i) => {
      const y = 610 + i * 62;
      ctx.fillStyle = i === 0 ? '#4de0d0' : 'rgba(255,255,255,0.35)';
      ctx.font = '26px sans-serif';
      ctx.fillText(String(i + 1), 80, y);

      ctx.fillStyle = i === 0 ? '#ffffff' : 'rgba(255,255,255,0.8)';
      ctx.font = i === 0 ? 'bold 32px sans-serif' : '30px sans-serif';
      const rowVoice = item.voice || describeVoice(item, app.globalData.gender);
      ctx.fillText(rowVoice.tag, 120, y);

      ctx.textAlign = 'right';
      ctx.fillStyle = i === 0 ? '#4de0d0' : 'rgba(255,255,255,0.55)';
      ctx.font = '26px sans-serif';
      ctx.fillText(item.scoreText || '', 670, y);
      ctx.textAlign = 'left';
    });

    const dims = top.dimensions || {};
    const meta = [
      { key: 'pitch', label: '音高', color: '#7c6cff' },
      { key: 'timbre', label: '音色', color: '#4de0d0' },
      { key: 'resonance', label: '共鸣', color: '#ff7eb6' },
      { key: 'style', label: '唱腔', color: '#ffb45a' }
    ];

    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '24px sans-serif';
    ctx.fillText('四维声线画像', 80, 830);

    meta.forEach((m, i) => {
      const y = 870 + i * 32;
      const value = typeof dims[m.key] === 'number' ? dims[m.key] : 0;

      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      ctx.font = '22px sans-serif';
      ctx.fillText(m.label, 80, y);

      roundRect(ctx, 150, y - 14, 380, 12, 6);
      ctx.fillStyle = 'rgba(255,255,255,0.12)';
      ctx.fill();

      roundRect(ctx, 150, y - 14, Math.max(6, 380 * value / 100), 12, 6);
      ctx.fillStyle = m.color;
      ctx.fill();

      ctx.textAlign = 'right';
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.fillText(String(Math.round(value)), 620, y);
      ctx.textAlign = 'left';
    });

    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = '22px sans-serif';
    ctx.fillText('声纹研究所 · 娱乐参考，非专业声乐评估', 80, 985);
  },

  onShareTimeline() {
    const top = this.data.result && this.data.result.results && this.data.result.results[0];
    const tag = top && top.voice ? top.voice.tag : '';
    return {
      title: tag ? `我的声线是「${tag}」！测测你的声线类型` : '测测你的声线类型'
    };
  },

  onShareAppMessage() {
    const top = this.data.result && this.data.result.results && this.data.result.results[0];
    const tag = top && top.voice ? top.voice.tag : '';
    return {
      title: tag ? `我的声线是「${tag}」！测测你的声线类型` : '测测你的声线类型',
      path: '/pages/index/index'
    };
  }
});
