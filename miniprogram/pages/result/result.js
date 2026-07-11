const app = getApp()

Page({
  data: {
    // 俯视图
    topAnnotatedImg: '',
    scaleNote: '',
    topCI: '', topCVAI: '', topCVA: '',
    topLength: '', topWidth: '', topCirc: '',
    topAnalysis: {},
    topSeverityTag: 'primary',

    // 侧面图
    hasSide: false,
    sideAnnotatedImg: '',
    sideScaleNote: '',
    sideAnalysis: {},
    sideTagType: 'primary',

    // 综合分析
    hasAI: false,
    aiExplanation: '',
    tips: [],
    tummyTime: '',
    nextStep: '',
    fallbackText: ''
  },

  onLoad() {
    const result = app.globalData.analysisResult
    if (!result) {
      wx.showToast({ title: '数据加载失败', icon: 'none' })
      setTimeout(() => wx.navigateBack(), 1500)
      return
    }
    this.parseResult(result)
  },

  parseResult(data) {
    // ====== 俯视图 ======
    const top = data.top || {}
    const m = top.measurements || {}
    const ta = top.analysis || {}

    const sevToTag = { '正常': 'success', '轻度': 'warning', '中度': 'danger', '重度': 'danger' }

    // ====== 侧面图 ======
    const side = data.side || {}
    const hasSide = !!(side && side.measurements)
    const si = side.measurements || {}
    const sa = side.analysis || {}
    const flatToTag = { '正常圆润': 'success', '轻度扁平': 'warning', '中度扁平': 'danger', '明显扁平': 'danger' }

    // ====== 综合分析 ======
    const ai = data.ai || {}
    const fb = data.fallback || {}
    const useAI = ai && !ai.error
    const combined = useAI ? ai : fb

    this.setData({
      // 俯视图
      topAnnotatedImg: top.annotated_image || '',
      scaleNote: top.scale_note || '',
      hasReference: top.has_reference !== false,

      // 侧面图比例尺
      sideScaleNote: (sa && sa.scale_method) ? ('侧面: ' + sa.scale_method) : '',
      topCI: (m.ci || 0).toFixed(1),
      topCVAI: (m.cvai || 0).toFixed(1),
      topCVA: (m.cva_mm || 0).toFixed(1),
      topLength: (m.head_length_mm || 0).toFixed(1),
      topWidth: (m.head_width_mm || 0).toFixed(1),
      topCirc: (m.head_circumference_mm || 0).toFixed(1),
      topAnalysis: ta,
      topSeverityTag: sevToTag[ta.severity] || 'primary',

      // 侧面图
      hasSide: hasSide,
      sideAnnotatedImg: side.annotated_image || '',
      sideAnalysis: sa,
      sideTagType: flatToTag[sa.flatness_category] || 'primary',

      // 综合分析
      hasAI: useAI && !!ai.explanation,
      aiExplanation: (ai.explanation || '').slice(0, 300),
      tips: combined.daily_tips
        || ((combined.intervention_plan && combined.intervention_plan.repositioning) || []),
      tummyTime: combined.tummy_time_advice
        || (combined.intervention_plan && combined.intervention_plan.tummy_time)
        || '',
      nextStep: combined.next_step || '',
      fallbackText: (!useAI && combined.explanation) ? combined.explanation : ''
    })
  },

  goBack() {
    wx.navigateBack()
  }
})
