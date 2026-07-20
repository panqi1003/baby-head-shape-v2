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
    tips: [], tummyTime: '', nextStep: '', fallbackText: '',

    // 导出
    ageMonths: ''
  },

  onLoad() {
    let r = app.globalData.analysisResult
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
    const age = wx.getStorageSync('_ageMonths')
    if (age) this.setData({ ageMonths: age })
  },

  parse(data) {
    const top = data.top || {}
    const m = top.measurements || {}
    const ta = top.analysis || {}

    const side = data.sideRight
    const hasSide = !!(side && side.analysis)

    // CI/CVAI 状态
    const ci = parseFloat(m.ci) || 0
    let ciStatus = '', ciColor = ''
    if (ci >= 75 && ci <= 85) { ciStatus = '在正常范围'; ciColor = '#16a34a' }
    else if (ci > 85) { ciStatus = '偏扁头倾向'; ciColor = '#d97706' }
    else { ciStatus = '偏长头倾向'; ciColor = '#d97706' }

    const cvai = parseFloat(m.cvai) || 0
    let cvaiStatus = '', cvaiColor = ''
    if (cvai < 3.5) { cvaiStatus = '对称性良好'; cvaiColor = '#16a34a' }
    else if (cvai < 6.25) { cvaiStatus = '轻微不对称'; cvaiColor = '#d97706' }
    else { cvaiStatus = '不对称较明显'; cvaiColor = '#ef4444' }

    const ai = data.ai || {}
    const fb = data.fb || {}
    const useAI = ai && !ai.error
    const combined = useAI ? ai : fb

    const topDesc = ta.summary || ''

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
      topCIStatus: ciStatus, topCIStatusColor: ciColor,
      topCVAIStatus: cvaiStatus, topCVAIStatusColor: cvaiColor,

      hasSide,
      sideCompareImg: (hasSide && side.compare_image) || '',
      sideSimilarity: sideSim,
      sideDesc,

      hasAI: useAI && !!ai.explanation,
      aiExplanation: (ai.explanation || '').slice(0, 300),
      tips: combined.daily_tips || [],
      tummyTime: combined.tummy_time_advice || (combined.intervention_plan && combined.intervention_plan.tummy_time) || '',
      nextStep: combined.next_step || '',
      fallbackText: (!useAI && combined.explanation) ? combined.explanation : ''
    })
  },

  // ====== 报告导出 ======

  _wx(method, opts) {
    return new Promise((resolve, reject) => {
      method(Object.assign({ success: resolve, fail: reject }, opts))
    })
  },

  async exportReport() {
    const setting = await this._wx(wx.getSetting)
    if (setting.authSetting['scope.writePhotosAlbum'] === false) {
      wx.showModal({
        title: '需要相册权限',
        content: '保存报告需授权相册写入，是否前往设置？',
        confirmText: '去设置',
        success: m => { if (m.confirm) wx.openSetting() }
      })
      return
    }
    if (!setting.authSetting['scope.writePhotosAlbum']) {
      try { await this._wx(wx.authorize, { scope: 'scope.writePhotosAlbum' }) }
      catch (_) { return }
    }

    wx.showLoading({ title: '生成报告中...' })
    try {
      await this._drawReport()
      wx.hideLoading()
    } catch (e) {
      wx.hideLoading()
      console.error('导出失败:', e)
      wx.showToast({ title: '保存失败，请重试', icon: 'none' })
    }
  },

  async _drawReport() {
    const query = wx.createSelectorQuery()
    const nodeRes = await new Promise(resolve =>
      query.select('#reportCanvas').fields({ node: true, size: true }).exec(resolve)
    )
    const canvas = nodeRes[0].node
    const ctx = canvas.getContext('2d')

    const W = 750, dpr = 2, M = 30
    const d = this.data

    // 加载图片
    const topB64 = d.topCompareImg
    const sideB64 = d.sideCompareImg
    let topImg = null, sideImg = null
    if (topB64 && topB64.startsWith('data:image')) {
      const path = await this._b64toFile(topB64, 'top')
      topImg = await this._loadImg(canvas, path)
    }
    if (d.hasSide && sideB64 && sideB64.startsWith('data:image')) {
      const path = await this._b64toFile(sideB64, 'side')
      sideImg = await this._loadImg(canvas, path)
    }
    const hasSideImg = !!(d.hasSide && sideImg)

    // ---- 预计算高度 ----
    const dualW = 335, dualH = 268
    const singleW = 520, singleH = 390
    const imgH = hasSideImg ? dualH : singleH

    const aiText = d.aiExplanation || ''
    const tips = d.tips || []
    const nextStep = d.nextStep || ''
    const hasAI = !!aiText
    const hasTips = tips.length > 0 || !!nextStep
    let aiCardH = 0
    if (hasAI || hasTips) {
      aiCardH = 24
      if (hasAI) {
        const text = aiText.slice(0, 200)
        const lines = Math.ceil(text.length / 28)
        aiCardH += 28 + lines * 30 + 4
      }
      if (hasTips) {
        const recs = [...tips.slice(0, 2)]
        if (nextStep) recs.push(nextStep)
        aiCardH += 8 + 24 + recs.length * 24
      }
      aiCardH += 24
    }

    let totalH = 150 + 18 + 16
    totalH += topImg ? (imgH + 30) : 0
    totalH += topImg ? (24 + 32 + 104 + 20 + 80) : 0
    totalH += (hasAI || hasTips) ? (24 + aiCardH + 24) : 0
    totalH += 1 + 16 + 24 + 28 + 20 + 24 + 30

    canvas.width = W * dpr
    canvas.height = totalH * dpr
    ctx.scale(dpr, dpr)

    let y = 0

    // ===== 1. Header (150px) =====
    const grad = ctx.createLinearGradient(0, 0, W, 0)
    grad.addColorStop(0, '#3b82f6'); grad.addColorStop(1, '#6366f1')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, 150)

    ctx.fillStyle = '#ffffff'; ctx.font = 'bold 34px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText('宝宝头型分析报告', W / 2, 58)

    const now = new Date()
    const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    const ageStr = d.ageMonths ? `${d.ageMonths}个月` : ''
    ctx.font = '26px sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.75)'
    ctx.fillText([dateStr, ageStr].filter(Boolean).join(' · '), W / 2, 106)

    // 圆弧过渡
    ctx.fillStyle = '#ffffff'
    ctx.beginPath(); ctx.moveTo(0, 150); ctx.lineTo(W, 150)
    ctx.lineTo(W, 168); ctx.quadraticCurveTo(W / 2, 176, 0, 168); ctx.fill()
    y = 184

    // ===== 2. 对比图 =====
    if (topImg) {
      if (hasSideImg && sideImg) {
        const ix1 = M, ix2 = M + dualW + 20
        ctx.shadowColor = 'rgba(0,0,0,0.06)'; ctx.shadowBlur = 12; ctx.shadowOffsetY = 2

        ctx.fillStyle = '#f1f5f9'
        this._roundRect(ctx, ix1, y, dualW, dualH, 14); ctx.fill()
        ctx.save(); ctx.beginPath(); this._roundRect(ctx, ix1, y, dualW, dualH, 14); ctx.clip()
        ctx.drawImage(topImg, ix1, y, dualW, dualH); ctx.restore()

        ctx.fillStyle = '#f1f5f9'
        this._roundRect(ctx, ix2, y, dualW, dualH, 14); ctx.fill()
        ctx.save(); ctx.beginPath(); this._roundRect(ctx, ix2, y, dualW, dualH, 14); ctx.clip()
        ctx.drawImage(sideImg, ix2, y, dualW, dualH); ctx.restore()

        ctx.shadowColor = 'transparent'
        this._drawBadge(ctx, ix1 + dualW - 8, y + dualH - 8, `重合度 ${d.topSimilarity}%`)
        this._drawBadge(ctx, ix2 + dualW - 8, y + dualH - 8, `重合度 ${d.sideSimilarity}%`)
      } else {
        const ix = (W - singleW) / 2
        ctx.shadowColor = 'rgba(0,0,0,0.06)'; ctx.shadowBlur = 12; ctx.shadowOffsetY = 2

        ctx.fillStyle = '#f1f5f9'
        this._roundRect(ctx, ix, y, singleW, singleH, 14); ctx.fill()
        ctx.save(); ctx.beginPath(); this._roundRect(ctx, ix, y, singleW, singleH, 14); ctx.clip()
        ctx.drawImage(topImg, ix, y, singleW, singleH); ctx.restore()

        ctx.shadowColor = 'transparent'
        this._drawBadge(ctx, ix + singleW - 8, y + singleH - 8, `重合度 ${d.topSimilarity}%`)
      }

      y += imgH + 8
      ctx.fillStyle = '#94a3b8'; ctx.font = '18px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText('绿色=宝宝实测轮廓  |  白色=标准头型参考', W / 2, y + 18)
      y += 30

      // ===== 3. 测量数据 =====
      y += 24
      ctx.fillStyle = '#475569'; ctx.font = 'bold 24px sans-serif'; ctx.textAlign = 'left'
      ctx.fillText('测量数据', M, y + 18)
      y += 32

      // 核心指标
      const coreW = 330, coreH = 104, coreGap = 30
      this._drawCoreCard(ctx, M, y, coreW, coreH,
        { value: d.topCI, label: '头颅指数 CI', status: d.topCIStatus, color: d.topCIStatusColor })
      this._drawCoreCard(ctx, M + coreW + coreGap, y, coreW, coreH,
        { value: d.topCVAI, unit: '%', label: '不对称指数', status: d.topCVAIStatus, color: d.topCVAIStatusColor })
      y += coreH + 20

      // 辅助指标
      const auxItems = [
        { value: d.topLength, unit: 'mm', label: '头长' },
        { value: d.topWidth, unit: 'mm', label: '头宽' },
        { value: d.topCirc, unit: 'mm', label: '头围(估)' },
        { value: d.topCVA, unit: 'mm', label: '不对称值' },
      ]
      const auxW = 157, auxH = 72, auxGap = 20
      const auxTotalW = auxItems.length * auxW + (auxItems.length - 1) * auxGap
      const auxStartX = (W - auxTotalW) / 2

      for (let i = 0; i < auxItems.length; i++) {
        const ax = auxStartX + i * (auxW + auxGap), it = auxItems[i]
        ctx.fillStyle = '#f8fafc'
        this._roundRect(ctx, ax, y, auxW, auxH, 10); ctx.fill()
        ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1
        this._roundRect(ctx, ax, y, auxW, auxH, 10); ctx.stroke()

        ctx.fillStyle = '#1e293b'; ctx.font = 'bold 28px sans-serif'; ctx.textAlign = 'center'
        ctx.fillText(it.value + it.unit, ax + auxW / 2, y + 30)
        ctx.fillStyle = '#94a3b8'; ctx.font = '20px sans-serif'
        ctx.fillText(it.label, ax + auxW / 2, y + 56)
      }
      y += auxH + 24
    }

    // ===== 4. AI 智能分析 =====
    if (hasAI || hasTips) {
      const aiW = W - 2 * M, aiX = M, aiPad = 24

      ctx.shadowColor = 'rgba(0,0,0,0.05)'; ctx.shadowBlur = 10; ctx.shadowOffsetY = 1
      ctx.fillStyle = '#f5f3ff'
      this._roundRect(ctx, aiX, y, aiW, aiCardH, 14); ctx.fill()
      ctx.shadowColor = 'transparent'

      ctx.fillStyle = '#8b5cf6'
      ctx.fillRect(aiX, y + 8, 4, aiCardH - 16)

      let textY = y + aiPad + 22

      if (hasAI) {
        ctx.fillStyle = '#7c3aed'; ctx.font = 'bold 22px sans-serif'; ctx.textAlign = 'left'
        ctx.fillText('智能分析', aiX + aiPad + 8, textY)
        textY += 28

        const text = aiText.slice(0, 200)
        ctx.fillStyle = '#334155'; ctx.font = '22px sans-serif'
        for (let i = 0; i < text.length; i += 28) {
          ctx.fillText(text.slice(i, i + 28), aiX + aiPad + 8, textY)
          textY += 30
        }
        textY += 4
      }

      if (hasTips) {
        textY += 8
        const recs = [...tips.slice(0, 2)]
        if (nextStep) recs.push(nextStep)
        ctx.fillStyle = '#6366f1'; ctx.font = 'bold 20px sans-serif'
        ctx.fillText('💡 建议', aiX + aiPad + 8, textY)
        textY += 24
        ctx.fillStyle = '#475569'; ctx.font = '20px sans-serif'
        for (const r of recs) {
          ctx.fillText('· ' + r, aiX + aiPad + 8, textY)
          textY += 24
        }
      }
      y += aiCardH + 24
    }

    // ===== 5. Footer =====
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1
    ctx.setLineDash([4, 6])
    ctx.beginPath(); ctx.moveTo(M, y); ctx.lineTo(W - M, y); ctx.stroke()
    ctx.setLineDash([])
    y += 16

    ctx.fillStyle = '#94a3b8'; ctx.font = '20px sans-serif'; ctx.textAlign = 'center'
    const meta = [d.scaleNote, `可信度 ${d.topConfidence}%`].filter(Boolean).join('  |  ')
    ctx.fillText(meta, W / 2, y + 16)
    y += 28

    ctx.fillStyle = '#cbd5e1'; ctx.font = '18px sans-serif'
    ctx.fillText('本报告由AI生成，仅供家庭参考，不构成医疗诊断', W / 2, y + 16)
    y += 20
    ctx.fillText('宝宝头型分析 · 微信小程序', W / 2, y + 16)
    y += 30

    // ===== 6. 导出保存 =====
    const realH = Math.ceil(y * dpr)
    const tempRes = await this._wx(wx.canvasToTempFilePath, {
      canvas, x: 0, y: 0, width: W * dpr, height: realH, destWidth: W * dpr, destHeight: realH
    })
    await this._wx(wx.saveImageToPhotosAlbum, { filePath: tempRes.tempFilePath })
    wx.showToast({ title: '报告已保存到相册', icon: 'success' })
  },

  /** 重合度角标 (图片右下角 overlay) */
  _drawBadge(ctx, rightX, bottomY, text) {
    const padX = 14, padY = 8
    ctx.font = 'bold 20px sans-serif'
    const tw = ctx.measureText(text).width
    const bw = tw + padX * 2, bh = 34
    const bx = rightX - bw, by = bottomY - bh

    ctx.fillStyle = 'rgba(0,0,0,0.55)'
    this._roundRect(ctx, bx, by, bw, bh, 17); ctx.fill()
    ctx.fillStyle = '#ffffff'; ctx.textAlign = 'center'
    ctx.fillText(text, bx + bw / 2, by + 24)
  },

  /** 核心指标卡片 (带顶部状态色条) */
  _drawCoreCard(ctx, x, y, w, h, { value, unit, label, status, color }) {
    ctx.fillStyle = '#f0f9ff'
    this._roundRect(ctx, x, y, w, h, 14); ctx.fill()

    ctx.fillStyle = color || '#16a34a'
    ctx.fillRect(x + 8, y + 1, w - 16, 3)

    const valStr = value + (unit || '')
    ctx.fillStyle = '#1e293b'; ctx.font = 'bold 36px sans-serif'; ctx.textAlign = 'center'
    ctx.fillText(valStr, x + w / 2, y + 46)
    ctx.fillStyle = '#64748b'; ctx.font = '22px sans-serif'
    ctx.fillText(label, x + w / 2, y + 70)
    ctx.fillStyle = color || '#16a34a'; ctx.font = '20px sans-serif'
    ctx.fillText(status, x + w / 2, y + 92)
  },

  /** base64 → 临时文件 */
  _b64toFile(b64, tag) {
    return new Promise((resolve, reject) => {
      const fs = wx.getFileSystemManager()
      const path = `${wx.env.USER_DATA_PATH}/report_${tag}_${Date.now()}.jpg`
      const data = b64.replace(/^data:image\/\w+;base64,/, '')
      fs.writeFile({ filePath: path, data, encoding: 'base64', success: () => resolve(path), fail: reject })
    })
  },

  /** Canvas 加载图片 */
  _loadImg(canvas, src) {
    return new Promise((resolve, reject) => {
      const img = canvas.createImage()
      img.onload = () => resolve(img)
      img.onerror = reject
      img.src = src
    })
  },

  /** 圆角矩形路径 */
  _roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath()
    ctx.moveTo(x + r, y)
    ctx.lineTo(x + w - r, y)
    ctx.arcTo(x + w, y, x + w, y + r, r)
    ctx.lineTo(x + w, y + h - r)
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
    ctx.lineTo(x + r, y + h)
    ctx.arcTo(x, y + h, x, y + h - r, r)
    ctx.lineTo(x, y + r)
    ctx.arcTo(x, y, x + r, y, r)
    ctx.closePath()
  },

  goBack() { wx.navigateBack() }
})
