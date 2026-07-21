const app = getApp()

Page({
  data: {
    topPhoto: '',
    rightSidePhoto: '',
    ageMonths: '',
    useReference: true,
    useAI: true,
    analyzing: false,
    stepText: '',
    photoType: '',
    hasReference: false,
    refChecked: false,
    guideFrameActive: true,
  },

  // 隐私授权: 首次拍照/选图前触发官方隐私弹窗
  ensurePrivacy(cb) {
    if (wx.requirePrivacyAuthorize) {
      wx.requirePrivacyAuthorize({
        success: () => cb(),   // 用户同意 (或此前已同意)
        fail: (err) => {
          const msg = (err && err.errMsg) || ''
          // 后台隐私指引未配置(开发期常见) → 放行, 上线前必须在mp后台补填
          if (msg.indexOf('not declared') >= 0 || msg.indexOf('api scope') >= 0) {
            console.warn('隐私指引未配置, 开发期放行:', msg)
            cb()
          } else {
            wx.showToast({ title: '需同意隐私协议才能使用拍照功能', icon: 'none', duration: 2500 })
          }
        }
      })
    } else {
      cb()  // 低版本基础库无此API, 直接放行
    }
  },

  takeTopPhoto() {
    this.ensurePrivacy(() => {
      this.setData({ photoType: 'top', guideFrameActive: true })
      wx.navigateTo({ url: '/pages/camera/camera?type=top' })
    })
  },
  takeRightSidePhoto() {
    this.ensurePrivacy(() => {
      this.setData({ photoType: 'side', guideFrameActive: true })
      wx.navigateTo({ url: '/pages/camera/camera?type=side&side=right' })
    })
  },

  // 从相册选择照片 (无引导框对齐, 比例估算降级)
  chooseFromAlbum(e) {
    const type = e.currentTarget.dataset.type
    const that = this
    that.ensurePrivacy(() => {
      wx.chooseMedia({
        count: 1,
        mediaType: ['image'],
        sourceType: ['album'],
        success(res) {
          const path = res.tempFiles[0].tempFilePath
          that.setData({ guideFrameActive: false })
          if (type === 'top') {
            that.setData({ topPhoto: path })
            that.checkReference(path)
          } else if (type === 'right') {
            that.setData({ rightSidePhoto: path })
          }
        }
      })
    })
  },

  onShow() {
    const app = getApp()
    const top = app.globalData._cameraTopPhoto
    if (top) { this.setData({ topPhoto: top }); this.checkReference(top); app.globalData._cameraTopPhoto = null }

    const sidePath = app.globalData._cameraSidePhoto
    const sideWhich = app.globalData._cameraSideWhich
    if (sidePath && sideWhich === 'right') {
      this.setData({ rightSidePhoto: sidePath })
      app.globalData._cameraSidePhoto = null
      app.globalData._cameraSideWhich = null
    }
  },

  onAgeInput(e) {
    // 只保留数字
    const v = String(e.detail.value || '').replace(/[^\d]/g, '')
    this.setData({ ageMonths: v })
    return v
  },
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
    const age = String(that.data.ageMonths || '').trim()
    if (!age) {
      wx.showToast({ title: '请填写宝宝月龄', icon: 'none' })
      return
    }
    const ageNum = parseInt(age)
    if (isNaN(ageNum) || ageNum < 0 || ageNum > 36) {
      wx.showToast({ title: '月龄请填 0-36 的数字', icon: 'none' })
      return
    }
    that.setData({ analyzing: true, stepText: '正在分析俯视图...' })
    that.step1TopAnalysis()
  },

  // 阶段1: 俯视图分析
  step1TopAnalysis() {
    const that = this
    const formData = {
      use_reference: String(that.data.useReference),
      auto_detect: String(that.data.useReference),
      guide_frame: String(that.data.guideFrameActive),  // 相机拍照=true, 相册=false
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
            wx.setStorageSync('_topResult', data)
            wx.setStorageSync('_ageMonths', that.data.ageMonths)
            // 有侧面图 → 阶段2, 否则直接综合
            if (that.data.rightSidePhoto) {
              that.setData({ stepText: '正在分析侧面图...' })
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

  // 阶段2: 侧面图分析 (单次上传)
  step2SideAnalysis() {
    const that = this
    let done = false
    const timer = setTimeout(() => {
      if (!done) {
        done = true
        that.setData({ rightSidePhoto: '' })
        wx.showToast({ title: '侧面分析超时，已跳过', icon: 'none', duration: 2000 })
        that.step3CombinedAnalysis()
      }
    }, 15000)

    wx.uploadFile({
      url: app.globalData.apiBase + '/analyze_side',
      filePath: that.data.rightSidePhoto,
      name: 'image',
      formData: { guide_frame: String(that.data.guideFrameActive), side: 'right' },
      success(res) {
        if (done) return; done = true; clearTimeout(timer)
        try {
          const data = JSON.parse(res.data)
          if (data.success) {
            that._sideResult = data
            wx.setStorageSync('_sideResult', data)
            that.step3CombinedAnalysis()
          } else {
            // 侧面失败 → 降级继续综合
            wx.showToast({ title: '侧面未识别，已跳过侧面分析', icon: 'none', duration: 2000 })
            that.step3CombinedAnalysis()
          }
        } catch (e) {
          wx.showToast({ title: '侧面响应异常，已跳过', icon: 'none', duration: 2000 })
          that.step3CombinedAnalysis()
        }
      },
      fail() {
        if (done) return; done = true; clearTimeout(timer)
        that.setData({ rightSidePhoto: '' })
        wx.showToast({ title: '侧面网络失败，已跳过', icon: 'none', duration: 2000 })
        that.step3CombinedAnalysis()
      }
    })
  },

  // 阶段3: 综合分析
  step3CombinedAnalysis() {
    const that = this
    const allData = {
      top: that._topResult,
      sideRight: that._sideResult || null,
    }

    that.setData({ stepText: 'AI 综合分析中...' })
    if (that.data.useAI) {
      wx.request({
        url: app.globalData.apiBase + '/ai_analysis',
        method: 'POST',
        header: { 'Content-Type': 'application/json' },
        data: {
          top_measurements: that._topResult.measurements,
          top_analysis: that._topResult.analysis,
          side_measurements: that._sideResult ? that._sideResult.measurements : null,
          side_analysis: that._sideResult ? that._sideResult.analysis : null,
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
      that.setData({ stepText: '分析完成' })
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
