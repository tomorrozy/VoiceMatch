// app.js
App({
  globalData: {
    // 最近一次测试结果（存内存，页面间传递用）
    lastResult: null,
    // 用户选择的性别候选池
    gender: 'female',
    // 微信隐私授权状态（官方能力，基础库 >= 2.32.3）
    privacy: {
      supported: false,
      needAuthorization: false,
      contractName: '《用户隐私保护指引》'
    }
  },

  onLaunch() {
    this.globalData.privacy.supported = !!(wx.getPrivacySetting && wx.requirePrivacyAuthorize);

    // 提前查一次，页面首屏不用等异步结果
    if (this.globalData.privacy.supported) {
      this.refreshPrivacySetting();
    }

    // 新版本静默下载完成后提示重启
    this.checkUpdate();

    // 探测录音管理器能力
    if (!wx.getRecorderManager) {
      wx.showModal({
        title: '版本过低',
        content: '当前微信版本不支持录音，请升级微信后再试。',
        showCancel: false
      });
    }
  },

  onPageNotFound() {
    wx.reLaunch({ url: '/pages/index/index' });
  },

  onError(err) {
    // 全局兜底：只留日志，避免白屏；具体错误由各页面自行提示
    console.error('app error:', err);
  },

  checkUpdate() {
    if (!wx.getUpdateManager) return;
    const manager = wx.getUpdateManager();
    manager.onUpdateReady(() => {
      wx.showModal({
        title: '更新提示',
        content: '新版本已经准备好，是否重启应用？',
        success: (res) => {
          if (res.confirm) manager.applyUpdate();
        }
      });
    });
    manager.onUpdateFailed(() => {});
  },

  /**
   * 查询微信侧记录的隐私同意状态。
   * 低版本没有该接口时返回 supported=false，由页面走本地降级。
   */
  refreshPrivacySetting() {
    return new Promise((resolve) => {
      if (!wx.getPrivacySetting) {
        resolve(this.globalData.privacy);
        return;
      }
      wx.getPrivacySetting({
        success: (res) => {
          this.globalData.privacy = {
            supported: true,
            needAuthorization: !!res.needAuthorization,
            contractName: res.privacyContractName || '《用户隐私保护指引》'
          };
          resolve(this.globalData.privacy);
        },
        fail: () => resolve(this.globalData.privacy)
      });
    });
  },

  /**
   * 触发官方隐私授权弹窗（由微信渲染，用户同意后微信侧记录状态）。
   * 仅在 refreshPrivacySetting 返回 needAuthorization=true 时调用。
   */
  ensurePrivacyAuthorize() {
    return new Promise((resolve) => {
      if (!wx.requirePrivacyAuthorize) {
        resolve(true);
        return;
      }
      wx.requirePrivacyAuthorize({
        success: () => {
          this.globalData.privacy.needAuthorization = false;
          resolve(true);
        },
        fail: () => resolve(false)
      });
    });
  },

  setResult(result) {
    this.globalData.lastResult = result;
  },

  getResult() {
    return this.globalData.lastResult;
  }
});
