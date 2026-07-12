const app = getApp()

Page({
  data: {
    photoType: 'top'  // 'top' or 'side'
  },

  data: {
    photoType: 'top',
    side: ''  // left/right
  },

  onLoad(options) {
    if (options.type) this.setData({ photoType: options.type })
    if (options.side) this.setData({ side: options.side })
  },

  takePhoto() {
    const ctx = wx.createCameraContext()
    const that = this
    ctx.takePhoto({
      quality: 'high',
      success(res) {
        if (that.data.photoType === 'top') {
          app.globalData._cameraTopPhoto = res.tempImagePath
        } else {
          app.globalData._cameraSidePhoto = res.tempImagePath
          app.globalData._cameraSideWhich = that.data.side
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
