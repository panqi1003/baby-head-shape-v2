const app = getApp()

Page({
  data: {
    topAnnotatedImg: '', scaleNote: '',
    topCI: '', topCVAI: '', topCVA: '',
    topLength: '', topWidth: '', topCirc: '',
    topAnalysis: {}, topSeverityTag: 'primary',

    hasLeftSide: false, leftAnnotatedImg: '', leftAnalysis: {}, leftTagType: 'primary',
    hasRightSide: false, rightAnnotatedImg: '', rightAnalysis: {}, rightTagType: 'primary',

    hasAI: false, aiExplanation: '',
    tips: [], tummyTime: '', nextStep: '', fallbackText: ''
  },

  onLoad() {
    const r = app.globalData.analysisResult
    if (!r) { wx.showToast({ title: '数据加载失败', icon: 'none' }); setTimeout(() => wx.navigateBack(), 1500); return }
    this.parse(r)
  },

  parse(data) {
    const top = data.top || {}
    const m = top.measurements || {}
    const ta = top.analysis || {}
    const sevToTag = { '正常': 'success', '轻度': 'warning', '中度': 'danger', '重度': 'danger' }
    const flatToTag = { '正常圆润': 'success', '轻度扁平': 'warning', '中度扁平': 'danger', '明显扁平': 'danger' }

    // 综合分析
    const ai = data.ai || {}
    const fb = data.fallback || {}
    const useAI = ai && !ai.error
    const combined = useAI ? ai : fb

    this.setData({
      topAnnotatedImg: top.annotated_image || '',
      scaleNote: top.scale_note || '',
      topCI: (m.ci || 0).toFixed(1), topCVAI: (m.cvai || 0).toFixed(1), topCVA: (m.cva_mm || 0).toFixed(1),
      topLength: (m.head_length_mm || 0).toFixed(1), topWidth: (m.head_width_mm || 0).toFixed(1),
      topCirc: (m.head_circumference_mm || 0).toFixed(1),
      topAnalysis: ta, topSeverityTag: sevToTag[ta.severity] || 'primary',

      // 左侧面
      hasLeftSide: !!(data.sideLeft && data.sideLeft.analysis),
      leftAnnotatedImg: (data.sideLeft && data.sideLeft.annotated_image) || '',
      leftAnalysis: (data.sideLeft && data.sideLeft.analysis) || {},
      leftTagType: flatToTag[(data.sideLeft && data.sideLeft.analysis && data.sideLeft.analysis.flatness_category)] || 'primary',

      // 右侧面
      hasRightSide: !!(data.sideRight && data.sideRight.analysis),
      rightAnnotatedImg: (data.sideRight && data.sideRight.annotated_image) || '',
      rightAnalysis: (data.sideRight && data.sideRight.analysis) || {},
      rightTagType: flatToTag[(data.sideRight && data.sideRight.analysis && data.sideRight.analysis.flatness_category)] || 'primary',

      // 综合
      hasAI: useAI && !!ai.explanation,
      aiExplanation: (ai.explanation || '').slice(0, 300),
      tips: combined.daily_tips || ((combined.intervention_plan && combined.intervention_plan.repositioning) || []),
      tummyTime: combined.tummy_time_advice || (combined.intervention_plan && combined.intervention_plan.tummy_time) || '',
      nextStep: combined.next_step || '',
      fallbackText: (!useAI && combined.explanation) ? combined.explanation : ''
    })
  },

  goBack() { wx.navigateBack() }
})
