const app = getApp()

Page({
  data: {
    photoType: 'top'  // 'top' or 'side'
  },

  onLoad(options) {
    // 接收拍照类型参数
    if (options.type) {
      this.setData({ photoType: options.type })
    }
  },

  takePhoto() {
    const ctx = wx.createCameraContext()
    const that = this
    ctx.takePhoto({
      quality: 'high',
      success(res) {
        // 把照片路径存到全局数据
        if (that.data.photoType === 'top') {
          app.globalData._cameraTopPhoto = res.tempImagePath
        } else {
          app.globalData._cameraSidePhoto = res.tempImagePath
        }
        wx.navigateBack()
      },
      fail() {
        wx.showToast({ title: '拍照失败', icon: 'none' })
      }
    })
  },

  onCameraError() {
    wx.showToast({ title: '相机启动失败', icon: 'none' })
  },

  goBack() {
    wx.navigateBack()
  }
})
