// pages/history/history.js
const app = getApp();
const history = require('../../services/history.js');
const { scorePercent, formatTime } = require('../../utils/util.js');

const STATUS_TEXT = {
  ok: '分析完成',
  rejected: '质量未通过',
  no_usable_window: '无可用片段'
};

Page({
  data: {
    list: [],
    summary: '还没有测试记录'
  },

  onShow() {
    this.refresh();
  },

  refresh() {
    const list = history.list().map((item) => ({
      ...item,
      timeText: formatTime(item.createdAt),
      scoreText: scorePercent(item.score, item.scoreAsPercent),
      statusText: STATUS_TEXT[item.status] || '分析完成'
    }));
    this.setData({
      list,
      summary: list.length
        ? `共 ${list.length} 条记录，最多保留 20 条`
        : '还没有测试记录'
    });
  },

  onOpen(e) {
    const item = this.data.list[e.currentTarget.dataset.index];
    if (!item) return;
    app.setResult(item.result);
    wx.navigateTo({ url: '/pages/result/result' });
  },

  onDelete(e) {
    const item = this.data.list[e.currentTarget.dataset.index];
    if (!item) return;
    wx.showModal({
      title: '删除这条记录？',
      content: '删除后不可恢复。',
      confirmColor: '#ee0a24',
      success: (res) => {
        if (!res.confirm) return;
        history.remove(item.id);
        this.refresh();
        wx.showToast({ title: '已删除', icon: 'success' });
      }
    });
  },

  onClear() {
    if (!this.data.list.length) return;
    wx.showModal({
      title: '清空历史记录？',
      content: '所有本机结果都会被删除，删除后不可恢复。',
      confirmColor: '#ee0a24',
      success: (res) => {
        if (!res.confirm) return;
        history.clear();
        this.refresh();
        wx.showToast({ title: '已清空', icon: 'success' });
      }
    });
  },

  onGoRecord() {
    wx.navigateTo({ url: '/pages/record/record' });
  }
});
