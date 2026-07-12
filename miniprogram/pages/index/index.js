const app = getApp()

Page({
  data: {
    topPhoto: '',
    leftSidePhoto: '',
    rightSidePhoto: '',
    ageMonths: '',
    useReference: true,
    useAI: true,
    analyzing: false,
    photoType: '',
    hasReference: false,
    refChecked: false,
  },

  takeTopPhoto() { this.setData({ photoType: 'top' }); wx.navigateTo({ url: '/pages/camera/camera?type=top' }) },
  takeLeftSidePhoto() { this.setData({ photoType: 'side' }); wx.navigateTo({ url: '/pages/camera/camera?type=side&side=left' }) },
  takeRightSidePhoto() { this.setData({ photoType: 'side' }); wx.navigateTo({ url: '/pages/camera/camera?type=side&side=right' }) },

  onShow() {
    const app = getApp()
    const top = app.globalData._cameraTopPhoto
    if (top) { this.setData({ topPhoto: top }); this.checkReference(top); app.globalData._cameraTopPhoto = null }

    const sidePath = app.globalData._cameraSidePhoto
    const sideWhich = app.globalData._cameraSideWhich
    if (sidePath && sideWhich) {
      if (sideWhich === 'left') this.setData({ leftSidePhoto: sidePath })
      else this.setData({ rightSidePhoto: sidePath })
      app.globalData._cameraSidePhoto = null
      app.globalData._cameraSideWhich = null
    }
  },

  onAgeInput(e) { this.setData({ ageMonths: e.detail.value }) },
  toggleReference(e) {
    if (!this.data.hasReference) {
      wx.showToast({ title: '照片中未检测到参照物，无法开启校准', icon: 'none', duration: 2500 })
      this.setData({ useReference: false })
      return
    }
    this.setData({ useReference: e.detail })
  },

  // 预检照片中是否有参照物
  checkReference(filePath) {
    const that = this
    wx.uploadFile({
      url: app.globalData.apiBase + '/check_reference',
      filePath: filePath,
      name: 'image',
      success(res) {
        try {
          const data = JSON.parse(res.data)
          if (!data.has_reference) {
            that.setData({ hasReference: false, useReference: false, refChecked: true })
          } else {
            that.setData({ hasReference: true, useReference: true, refChecked: true })
          }
        } catch (e) {
          that.setData({ hasReference: false, refChecked: true })
        }
      },
      fail() {
        that.setData({ hasReference: false, refChecked: true })
      }
    })
  },
  toggleAI(e) { this.setData({ useAI: e.detail }) },

  // ====== 分析流程: 俯视 → 侧面 → 综合 ======
  startAnalyze() {
    const that = this
    if (!that.data.topPhoto) {
      wx.showToast({ title: '请先拍摄俯视图', icon: 'none' })
      return
    }
    that.setData({ analyzing: true })
    that.step1TopAnalysis()
  },

  // 阶段1: 俯视图分析
  step1TopAnalysis() {
    const that = this
    const formData = {
      use_reference: String(that.data.useReference),
      auto_detect: String(that.data.useReference),
      guide_frame: 'true',  // 通过相机页拍摄，已用引导框对齐
    }
    const age = String(that.data.ageMonths || '').trim()
    if (age) formData.age_months = age

    let done = false
    const timer = setTimeout(() => {
      if (!done) { done = true; that.setData({ analyzing: false }); wx.showToast({ title: '俯视图分析超时', icon: 'none' }) }
    }, 20000)

    wx.uploadFile({
      url: app.globalData.apiBase + '/analyze',
      filePath: that.data.topPhoto,
      name: 'top_image',
      formData: formData,
      success(res) {
        if (done) return; done = true; clearTimeout(timer)
        try {
          const data = JSON.parse(res.data)
          if (data.success) {
            that._topResult = data
            // 有侧面图 → 阶段2, 否则直接综合
            if (that.data.leftSidePhoto || that.data.rightSidePhoto) {
              that.step2SideAnalysis()
            } else {
              that.step3CombinedAnalysis()
            }
          } else {
            that.setData({ analyzing: false })
            wx.showToast({ title: data.error || '分析失败', icon: 'none', duration: 3000 })
          }
        } catch (e) {
          that.setData({ analyzing: false })
          wx.showToast({ title: '响应异常', icon: 'none' })
        }
      },
      fail() {
        if (done) return; done = true; clearTimeout(timer)
        that.setData({ analyzing: false })
        wx.showToast({ title: '网络请求失败', icon: 'none', duration: 4000 })
      }
    })
  },

  // 阶段2: 侧面图分析 (左右分别上传)
  step2SideAnalysis() {
    const that = this
    that._leftSideResult = null
    that._rightSideResult = null
    let pending = 0

    const uploadSide = (filePath, sideWhich) => {
      pending++
      let done = false
      const timer = setTimeout(() => {
        if (!done) { done = true; pending--; if (pending === 0) that.step3CombinedAnalysis() }
      }, 15000)

      wx.uploadFile({
        url: app.globalData.apiBase + '/analyze_side',
        filePath: filePath,
        name: 'image',
        formData: { guide_frame: 'true', side: sideWhich },
        success(res) {
          if (done) return; done = true; clearTimeout(timer); pending--
          try {
            const data = JSON.parse(res.data)
            if (data.success) {
              data._side = sideWhich
              if (sideWhich === 'left') that._leftSideResult = data
              else that._rightSideResult = data
            }
          } catch (e) {}
          if (pending === 0) that.step3CombinedAnalysis()
        },
        fail() {
          if (done) return; done = true; clearTimeout(timer); pending--
          if (pending === 0) that.step3CombinedAnalysis()
        }
      })
    }

    if (that.data.leftSidePhoto) uploadSide(that.data.leftSidePhoto, 'left')
    if (that.data.rightSidePhoto) uploadSide(that.data.rightSidePhoto, 'right')
  },

  // 阶段3: 综合分析 (有AI则调)
  step3CombinedAnalysis() {
    const that = this
    const allData = {
      top: that._topResult,
      sideLeft: that._leftSideResult || null,
      sideRight: that._rightSideResult || null,
    }

    if (that.data.useAI) {
      wx.request({
        url: app.globalData.apiBase + '/ai_analysis',
        method: 'POST',
        header: { 'Content-Type': 'application/json' },
        data: {
          top_measurements: that._topResult.measurements,
          top_analysis: that._topResult.analysis,
          side_left_measurements: that._leftSideResult ? that._leftSideResult.measurements : null,
          side_left_analysis: that._leftSideResult ? that._leftSideResult.analysis : null,
          side_right_measurements: that._rightSideResult ? that._rightSideResult.measurements : null,
          side_right_analysis: that._rightSideResult ? that._rightSideResult.analysis : null,
          age_months: (that.data.ageMonths || '') ? parseInt(that.data.ageMonths) : null
        },
        success(r) {
          try {
            if (r.data && r.data.success) {
              allData.ai = r.data.ai_analysis
              allData.fb = r.data.fallback_advice
            }
          } catch (e) {}
          that.finish(allData)
        },
        fail() { that.finish(allData) }
      })
    } else {
      that.finish(allData)
    }
  },

  finish(allData) {
    this.setData({ analyzing: false })
    app.globalData.analysisResult = allData
    wx.navigateTo({ url: '/pages/result/result' })
  },

  onShareAppMessage() {
    return { title: '宝宝头型分析 - 拍照测一测', path: '/pages/index/index' }
  }
})
