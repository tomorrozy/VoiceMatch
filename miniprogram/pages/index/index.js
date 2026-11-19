// pages/index/index.js
const app = getApp();

Page({
  data: {
    gender: 'female',
    bars: [40, 70, 50, 90, 60, 100, 55, 80, 45, 75, 95, 50, 65, 85, 42, 72, 58, 88, 48, 68],
    showPrivacy: false,
    privacySupported: false,
    privacyContractName: '《用户隐私保护指引》',
    privacyIntent: ''
  },

  onLoad() {
    this.initPrivacy();
  },

  /** 官方隐私状态优先；低版本回退到本地标记。 */
  async initPrivacy() {
    try {
      const info = await app.refreshPrivacySetting();
      if (info.supported) {
        this.setData({
          privacySupported: true,
          privacyContractName: info.contractName,
          showPrivacy: !!info.needAuthorization,
          privacyIntent: info.needAuthorization ? 'browse' : ''
        });
        return;
      }
    } catch (e) {
      // 查询失败按低版本降级处理
    }
    this.setData({
      privacySupported: false,
      showPrivacy: !this.hasLocalAgreed()
    });
  },

  hasLocalAgreed() {
    try {
      return !!wx.getStorageSync('privacy_agreed');
    } catch (e) {
      return false;
    }
  },

  onGender(e) {
    this.setData({ gender: e.currentTarget.dataset.g });
  },

  onHistory() {
    wx.navigateTo({ url: '/pages/history/history' });
  },

  async onStart() {
    let need = false;
    if (this.data.privacySupported) {
      const info = await app.refreshPrivacySetting();
      need = !!info.needAuthorization;
      this.setData({ privacyContractName: info.contractName });
    } else {
      need = !this.hasLocalAgreed();
    }

    if (need) {
      this.setData({ showPrivacy: true, privacyIntent: 'start' });
      return;
    }
    this.goRecord();
  },

  goRecord() {
    app.globalData.gender = this.data.gender;
    wx.navigateTo({ url: '/pages/record/record' });
  },

  /**
   * 官方按钮 open-type="agreePrivacyAuthorization" 的回调。
   * 微信在用户点击后记录同意状态，这里只负责关闭弹窗与继续流程。
   */
  onAgreePrivacy() {
    try { wx.setStorageSync('privacy_agreed', true); } catch (e) {}
    app.globalData.privacy.needAuthorization = false;
    const shouldStart = this.data.privacyIntent === 'start';
    this.setData({ showPrivacy: false, privacyIntent: '' });
    if (shouldStart) this.goRecord();
  },

  onRejectPrivacy() {
    this.setData({ showPrivacy: false, privacyIntent: '' });
    wx.showToast({ title: '同意隐私说明后即可开始', icon: 'none' });
  },

  onOpenPrivacyContract() {
    if (wx.openPrivacyContract) {
      wx.openPrivacyContract({
        fail: () => wx.navigateTo({ url: '/pages/privacy/privacy' })
      });
      return;
    }
    wx.navigateTo({ url: '/pages/privacy/privacy' });
  },

  onClosePrivacy() {
    this.setData({ showPrivacy: false });
  }
});
