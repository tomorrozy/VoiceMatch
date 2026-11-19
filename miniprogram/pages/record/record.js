// pages/record/record.js
const app = getApp();
const config = require('../../config.js');
const api = require('../../services/api.js');
const { mockCompare } = require('../../services/mock.js');
const history = require('../../services/history.js');
const { formatSec } = require('../../utils/util.js');

const MAX_SEC = 60;   // CLI 输入契约：总时长上限 60 秒
const MIN_SEC = 8;    // 设计文档 5.6：有效发声 ≥ 8 秒

/**
 * RecorderManager 是全局单例，且没有 offStart/offStop 这类解绑方法。
 * 如果在 onLoad 里注册监听，页面反复进出就会不断叠加回调（旧回调拿着已销毁
 * 的页面实例）。所以监听只在模块级注册一次，回调通过 activePage 找到当前
 * 仍然存活的那个页面实例。
 */
let recorderBound = false;
let activePage = null;

function bindRecorderOnce(page) {
  if (recorderBound || !page.recorder) return;
  recorderBound = true;
  const recorder = page.recorder;

  recorder.onStart(() => {
    const current = activePage;
    if (!current) return;
    current.setData({
      recording: true,
      statusText: '正在录音…',
      hintText: '唱完一句后点击停止'
    });
  });

  recorder.onStop((res) => {
    const current = activePage;
    if (!current) return;
    current.clearTimer();
    const dur = Math.min(MAX_SEC, (Date.now() - current.startTime) / 1000);
    current.setData({
      recording: false,
      hasRecord: !!res.tempFilePath,
      tempFilePath: res.tempFilePath || '',
      seconds: dur,
      durationText: formatSec(dur),
      statusText: '录制完成',
      hintText: res.tempFilePath ? '先试听，满意再提交' : '录音失败，请重试'
    });
    if (res.tempFilePath) current.prepareAudio(res.tempFilePath);
  });

  recorder.onError((err) => {
    const current = activePage;
    if (!current) return;
    current.clearTimer();
    current.setData({
      recording: false,
      statusText: '录音出错',
      hintText: '请检查麦克风后重试'
    });
    console.error('recorder error:', err);
    wx.showToast({ title: '录音失败，请重试', icon: 'none' });
  });

  // 来电等系统打断：停止录音，交给 onStop 统一收尾
  if (recorder.onInterruptionBegin) {
    recorder.onInterruptionBegin(() => {
      const current = activePage;
      if (!current || !current.data.recording) return;
      try { recorder.stop(); } catch (e) {}
    });
  }
}

Page({
  data: {
    recording: false,
    requesting: false,
    hasRecord: false,
    submitting: false,
    cancelling: false,
    playing: false,
    seconds: 0,
    durationText: '0:00',
    statusText: '准备就绪',
    hintText: '点击下方按钮开始录音',
    tempFilePath: '',
    uploadPct: 0,
    submitText: '',
    serviceState: 'checking',
    serviceText: '正在检查分析服务…',
    serviceDetail: ''
  },

  recorder: null,
  audio: null,
  timer: null,
  startTime: 0,

  onLoad() {
    this.recorder = wx.getRecorderManager ? wx.getRecorderManager() : null;
    if (!this.recorder) return;

    activePage = this;
    this._cancelled = false;
    this._taskId = '';

    // 录音参数以微信官方文档为准；这里请求 16k 单声道以贴近引擎 24k mono 的输入假设
    this.recordOptions = {
      duration: MAX_SEC * 1000,
      sampleRate: 16000,
      numberOfChannels: 1,
      format: 'mp3',
      encodeBitRate: 48000
    };

    bindRecorderOnce(this);
    this.checkService();
  },

  onHide() {
    // 切后台时停止录音：继续录会把环境噪声和无关对话录进来，白白拉低质量分
    if (this.data.recording && this.recorder) {
      try { this.recorder.stop(); } catch (e) {}
    }
    this.stopAudio();
  },

  onUnload() {
    // E-09：页面销毁时清理定时器、播放器与活动页引用
    this.clearTimer();
    this.stopAudio();
    if (this.audio) {
      this.audio.destroy();
      this.audio = null;
    }
    if (activePage === this) activePage = null;
    if (this.recorder && this.data.recording) {
      try { this.recorder.stop(); } catch (e) {}
    }
  },

  clearTimer() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  },

  // ---- 试听 --------------------------------------------------------

  /** 录音落盘后把路径交给播放器，首次调用时创建上下文。 */
  prepareAudio(path) {
    if (!wx.createInnerAudioContext) return;
    if (!this.audio) {
      this.audio = wx.createInnerAudioContext();
      this.audio.onEnded(() => {
        if (activePage === this) this.setData({ playing: false });
      });
      this.audio.onStop(() => {
        if (activePage === this) this.setData({ playing: false });
      });
      this.audio.onError((err) => {
        if (activePage !== this) return;
        this.setData({ playing: false });
        console.error('innerAudio error:', err);
        wx.showToast({ title: '播放失败，请重试', icon: 'none' });
      });
    }
    this.audio.stop();
    this.audio.src = path;
    this.setData({ playing: false });
  },

  stopAudio() {
    if (this.audio && this.data.playing) {
      try { this.audio.stop(); } catch (e) {}
    }
    if (this.data.playing) this.setData({ playing: false });
  },

  onTogglePlay() {
    if (!this.data.tempFilePath) return;
    if (!this.audio) this.prepareAudio(this.data.tempFilePath);
    if (this.data.playing) {
      this.audio.stop();
      this.setData({ playing: false });
      return;
    }
    this.audio.play();
    this.setData({ playing: true });
  },

  // ---- 服务状态 ----------------------------------------------------

  /** 查询服务与库版本，只用于页面提示，不阻塞录音。 */
  async checkService() {
    if (config.USE_MOCK) {
      this.setData({ serviceState: 'mock', serviceText: '演示模式 · 使用本地模拟结果' });
      return;
    }
    if (this._checking) return;
    this._checking = true;
    this.setData({ serviceState: 'checking', serviceText: '正在检查分析服务…' });

    try {
      const info = await api.health();
      if (activePage !== this) return;
      const singers = info.singers || {};
      const total = (singers.female || 0) + (singers.male || 0);
      const version = String(info.library_version || '').slice(0, 8);
      this.setData({
        serviceState: 'ok',
        serviceText: `分析服务已连接 · ${total} 位歌手 · 库 ${version}`,
        serviceDetail: `${info.encoder || ''} / ${info.model_id || ''}${info.calibrated ? ' / 已校准' : ' / 未校准'}`
      });
    } catch (err) {
      if (activePage !== this) return;
      this.setData({
        serviceState: 'down',
        serviceText: '分析服务未连接，点此重试',
        serviceDetail: err.message || '请先启动 server/app.py'
      });
    } finally {
      this._checking = false;
    }
  },

  onCheckService() {
    if (this.data.serviceState === 'checking') return;
    this.checkService();
  },

  // ---- 录音 --------------------------------------------------------

  /**
   * 隐私授权兜底。
   *
   * 正常路径是首页弹窗同意后才进本页；但从历史页回看后再「重新测一次」、
   * 或分享卡片直达本页，都会绕过首页弹窗。此时若微信侧仍 needAuthorization，
   * 直接调 wx.authorize 会被基础库以「隐私未授权」拒绝，用户只看到一句
   * 莫名其妙的失败提示。所以录音前补一次官方授权。
   *
   * 低版本基础库没有官方能力，直接放行，行为与改造前一致。
   */
  async ensurePrivacy() {
    const privacy = (app.globalData && app.globalData.privacy) || {};
    if (!privacy.supported) return true;

    const info = await app.refreshPrivacySetting();
    if (!info.needAuthorization) return true;

    const granted = await app.ensurePrivacyAuthorize();
    if (!granted) {
      wx.showModal({
        title: '需要同意隐私保护指引',
        content: '录音会采集你的声音并上传分析，需要你先同意《用户隐私保护指引》。',
        confirmText: '查看指引',
        success: (r) => {
          if (r.confirm && wx.openPrivacyContract) wx.openPrivacyContract();
        }
      });
      return false;
    }
    return true;
  },

  async ensureRecordPermission() {
    try {
      const setting = await wx.getSetting();
      if (setting.authSetting['scope.record']) return true;
      await wx.authorize({ scope: 'scope.record' });
      return true;
    } catch (e) {
      // 用户拒绝过 → 引导去设置页打开
      wx.showModal({
        title: '需要麦克风权限',
        content: '录音需要你授权麦克风权限，请在设置中开启。',
        confirmText: '去设置',
        success: (r) => {
          if (r.confirm && wx.openSetting) wx.openSetting();
        }
      });
      return false;
    }
  },

  async onToggleRecord() {
    if (this.data.recording) {
      this.recorder.stop();
      return;
    }
    if (!this.recorder) {
      wx.showToast({ title: '当前微信版本不支持录音', icon: 'none' });
      return;
    }
    this.setData({ requesting: true });
    const privacyOk = await this.ensurePrivacy();
    if (activePage !== this) return;
    if (!privacyOk) {
      this.setData({ requesting: false });
      return;
    }
    const ok = await this.ensureRecordPermission();
    if (activePage !== this) return;
    this.setData({ requesting: false });
    if (!ok) return;

    this.stopAudio();
    try {
      this.startTime = Date.now();
      this.recorder.start(this.recordOptions);
      this.setData({ seconds: 0, durationText: '0:00' });
      this.clearTimer();
      this.timer = setInterval(() => {
        if (!activePage) return;
        const s = (Date.now() - this.startTime) / 1000;
        if (s >= MAX_SEC) {
          this.recorder.stop();
          return;
        }
        this.setData({ seconds: s, durationText: formatSec(s) });
      }, 200);
    } catch (e) {
      wx.showToast({ title: '启动录音失败', icon: 'none' });
    }
  },

  onReRecord() {
    this.stopAudio();
    if (this.audio) {
      // 清掉路径，避免下次试听播到上一段
      this.audio.src = '';
    }
    this.setData({
      hasRecord: false, tempFilePath: '', seconds: 0, durationText: '0:00',
      statusText: '准备就绪', hintText: '点击下方按钮开始录音',
      uploadPct: 0, submitText: '', playing: false
    });
  },

  // ---- 提交与取消 --------------------------------------------------

  async onSubmit() {
    if (activePage !== this || this.data.submitting) return;

    if (!this.data.tempFilePath) {
      wx.showToast({ title: '还没有录音', icon: 'none' });
      return;
    }

    // 引擎按「有效发声 ≥ 8 秒」拒答；总时长都不足 8 秒时必然不达标，先在前端拦住
    if (this.data.seconds < MIN_SEC) {
      wx.showModal({
        title: '录音太短',
        content: `至少需要 ${MIN_SEC} 秒有效发声，当前 ${formatSec(this.data.seconds)}。请重新录一段完整句子。`,
        showCancel: false
      });
      return;
    }

    this.stopAudio();

    const gender = app.globalData.gender || 'female';
    this._cancelled = false;
    this._taskId = '';
    this.setData({
      submitting: true, cancelling: false, uploadPct: 0, submitText: '准备上传…'
    });

    try {
      let result;

      if (config.USE_MOCK) {
        result = mockCompare(gender);
        // 留一点"分析中"的观感，避免结果瞬间闪现
        await new Promise((resolve) => setTimeout(resolve, 600));
      } else {
        // 真实分支：上传 → 服务端受理成异步任务 → 轮询进度 → 拿结果
        result = await api.compare(this.data.tempFilePath, gender, {
          onProgress: (info) => {
            if (activePage !== this) return;
            // 上传占进度条前 30%，分析占后 70%：分段映射让进度条一路向前，
            // 不会在"上传 100% → 分析 0%"时突然回跳
            const pct = info.phase === 'upload'
              ? Math.round(info.pct * 0.3)
              : 30 + Math.round(info.pct * 0.7);
            this.setData({ uploadPct: pct, submitText: info.text });
          },
          onTask: (taskId) => {
            this._taskId = taskId;
          },
          isAlive: () => activePage === this && !this._cancelled
        });
      }

      // 分析期间用户可能已经返回上一页：此时不能再跳转，否则会把结果页
      // 压到别的页面上，用户回来找不着上下文
      if (activePage !== this) return;

      if (result.status === 'error') {
        throw new Error(result.detail || '分析服务返回错误');
      }

      app.setResult(result);
      // 成功与拒答都记历史：用户能看到"上次为什么没过"
      history.add(result, { gender, seconds: Math.round(this.data.seconds) });
      wx.navigateTo({ url: '/pages/result/result' });
    } catch (err) {
      if (activePage !== this) return;
      // 用户主动取消或离开导致的停止不算失败，不弹窗打扰
      if (err && err.reason === 'cancelled') return;

      console.error('compare failed:', err);
      wx.showModal({
        title: '分析失败',
        content: err.message || '请稍后重试',
        confirmText: '重试',
        cancelText: '知道了',
        success: (res) => {
          if (res.confirm) this.onSubmit();
        }
      });
      // 网络类失败顺手刷新顶部连接状态，用户能看到服务是否恢复
      if (err.reason === 'network' || err.reason === 'timeout') {
        this.checkService();
      }
    } finally {
      if (activePage === this) {
        this.setData({ submitting: false, cancelling: false, uploadPct: 0, submitText: '' });
      }
    }
  },

  /** 用户放弃这次分析：先停轮询，再让服务端别白算。 */
  async onCancelAnalyse() {
    if (!this.data.submitting || this.data.cancelling) return;
    this._cancelled = true;
    this.setData({ cancelling: true, submitText: '正在取消…' });

    const taskId = this._taskId;
    await api.cancelTask(taskId);

    if (activePage !== this) return;
    this.setData({
      submitting: false, cancelling: false, uploadPct: 0, submitText: ''
    });
    wx.showToast({ title: '已取消', icon: 'none' });
  }
});
