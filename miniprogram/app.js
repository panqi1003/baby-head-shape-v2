App({
  globalData: {
    // 后端 API 地址 — 部署后改成云托管域名
    apiBase: 'http://127.0.0.1:8000',
    // 分析结果 (跨页面传递)
    analysisResult: null
  },

  onLaunch() {
    // 检查网络状态
    wx.getNetworkType({
      success(res) {
        if (res.networkType === 'none') {
          wx.showToast({ title: '当前无网络连接', icon: 'none' })
        }
      }
    })
  }
})
