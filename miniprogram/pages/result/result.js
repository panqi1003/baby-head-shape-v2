const app = getApp()

Page({
  data: {
    // 俯视图
    topAnnotatedImg: '', scaleNote: '',
    topCI: '', topCVAI: '', topCVA: '',
    topLength: '', topWidth: '', topCirc: '',
    topConfidence: 0, topDesc: '',
    topCompareImg: '', topSimilarity: 0,
    hasReference: false,

    // 侧面图（面朝右）
    hasSide: false,
    sideCompareImg: '', sideSimilarity: 0, sideDesc: '',

    // 综合分析
    hasAI: false, aiExplanation: '',
    tips: [], tummyTime: '', nextStep: '', fallbackText: ''
  },

  onLoad() {
    let r = app.globalData.analysisResult
    // 内存中无数据时从Storage恢复
    if (!r) {
      const top = wx.getStorageSync('_topResult')
      const side = wx.getStorageSync('_sideResult')
      if (top) {
        r = { top, sideRight: side || null }
      }
    }
    if (!r) {
      wx.showToast({ title: '数据加载失败', icon: 'none' })
      setTimeout(() => wx.navigateBack(), 1500)
      return
    }
    this.parse(r)
  },

  parse(data) {
    const top = data.top || {}
    const m = top.measurements || {}
    const ta = top.analysis || {}

    // 侧面
    const side = data.sideRight
    const hasSide = !!(side && side.analysis)

    // 综合分析
    const ai = data.ai || {}
    const fb = data.fb || {}
    const useAI = ai && !ai.error
    const combined = useAI ? ai : fb

    // 俯视描述文字 (去掉严重度标签, 纯描述)
    const topDesc = ta.summary || ''

    // 侧面描述: 基于相似度给出参考说明
    const sideSim = (side && side.standard_compare && side.standard_compare.similarity_score) || 0
    let sideDesc = ''
    if (hasSide) {
      if (sideSim >= 70) {
        sideDesc = '后枕弧度与参考曲线接近，轮廓圆润。'
      } else if (sideSim >= 40) {
        sideDesc = '后枕弧度与参考曲线存在一定差异，建议关注睡姿。'
      } else {
        sideDesc = '后枕弧度与参考曲线差异较明显，建议定期拍照对比。'
      }
    }

    this.setData({
      // 俯视图
      topAnnotatedImg: top.annotated_image || '',
      scaleNote: top.scale_note || '',
      hasReference: !!top.has_reference,
      topCI: (m.ci || 0).toFixed(1),
      topCVAI: (m.cvai || 0).toFixed(1),
      topCVA: (m.cva_mm || 0).toFixed(1),
      topLength: (m.head_length_mm || 0).toFixed(1),
      topWidth: (m.head_width_mm || 0).toFixed(1),
      topCirc: (m.head_circumference_mm || 0).toFixed(1),
      topDesc,
      topConfidence: Math.round((m.confidence || 0) * 100),
      topCompareImg: (top.standard_compare && top.standard_compare.image) || '',
      topSimilarity: (top.standard_compare && top.standard_compare.similarity_score) || 0,

      // 侧面图
      hasSide,
      sideCompareImg: (hasSide && side.compare_image) || '',
      sideSimilarity: sideSim,
      sideDesc,

      // 综合分析
      hasAI: useAI && !!ai.explanation,
      aiExplanation: (ai.explanation || '').slice(0, 300),
      tips: combined.daily_tips || [],
      tummyTime: combined.tummy_time_advice || (combined.intervention_plan && combined.intervention_plan.tummy_time) || '',
      nextStep: combined.next_step || '',
      fallbackText: (!useAI && combined.explanation) ? combined.explanation : ''
    })
  },

  goBack() { wx.navigateBack() }
})
