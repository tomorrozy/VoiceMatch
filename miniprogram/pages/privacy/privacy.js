// pages/privacy/privacy.js
Page({
  data: {
    privacySupported: false,
    contractName: '《用户隐私保护指引》'
  },

  onLoad() {
    const app = getApp();
    if (app && app.refreshPrivacySetting) {
      app.refreshPrivacySetting().then((info) => {
        this.setData({
          privacySupported: !!info.supported,
          contractName: info.contractName
        });
      });
    }
  },

  /** 打开微信后台配置的《用户隐私保护指引》。 */
  onOpenContract() {
    if (!wx.openPrivacyContract) {
      wx.showToast({ title: '当前版本不支持，请在小程序后台查看', icon: 'none' });
      return;
    }
    wx.openPrivacyContract({
      fail: () => wx.showToast({ title: '隐私保护指引尚未配置', icon: 'none' })
    });
  },

  onClear() {
    wx.showModal({
      title: '确认清除',
      content: '将删除本机的录音与结果缓存，此操作不可撤销。',
      confirmColor: '#ee0a24',
      success: (res) => {
        if (!res.confirm) return;
        try {
          wx.removeStorageSync('privacy_agreed');
          // 历史记录也属于本机缓存，一起清掉，「可删除」才算完整兑现
          require('../../services/history.js').clear();
          getApp().setResult(null);
          wx.showToast({ title: '已清除本机缓存', icon: 'success' });
        } catch (e) {
          wx.showToast({ title: '清除失败', icon: 'none' });
        }
      }
    });
  }
});
