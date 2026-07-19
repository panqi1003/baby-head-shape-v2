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
    // 读取月龄供导出使用
    const age = wx.getStorageSync('_ageMonths')
    if (age) this.setData({ ageMonths: age })
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

  // ====== 报告导出 ======

  _wx(method, opts) {
    return new Promise((resolve, reject) => {
      method(Object.assign({ success: resolve, fail: reject }, opts))
    })
  },

  async exportReport() {
    // 1. 权限
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
    // 2. 获取 Canvas 节点
    const query = wx.createSelectorQuery()
    const nodeRes = await new Promise(resolve =>
      query.select('#reportCanvas').fields({ node: true, size: true }).exec(resolve)
    )
    const canvas = nodeRes[0].node
    const ctx = canvas.getContext('2d')

    // Canvas 尺寸 (高DPI 2x)
    const W = 750, dpr = 2
    canvas.width = W * dpr
    canvas.height = 1200 * dpr
    ctx.scale(dpr, dpr)

    const d = this.data

    // 3. 加载图片 (base64 → temp files)
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

    // 4. 绘制卡片
    let y = 0

    // --- Header 渐变 ---
    const grad = ctx.createLinearGradient(0, 0, W, 0)
    grad.addColorStop(0, '#3b82f6')
    grad.addColorStop(1, '#6366f1')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, 140)
    y = 140

    // 标题
    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 38px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('宝宝头型分析报告', W / 2, 55)

    // 日期 + 月龄
    const now = new Date()
    const dateStr = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
    const ageStr = d.ageMonths ? `${d.ageMonths}个月` : ''
    const subStr = [dateStr, ageStr].filter(Boolean).join(' · ')
    ctx.font = '22px sans-serif'
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.fillText(subStr, W / 2, 100)

    // 圆角过渡
    ctx.fillStyle = '#ffffff'
    ctx.beginPath(); ctx.moveTo(0, 140); ctx.lineTo(W, 140); ctx.lineTo(W, 156); ctx.quadraticCurveTo(W / 2, 170, 0, 156); ctx.fill()

    y = 180

    // --- 对比图 ---
    if (topImg) {
      const imgW = hasSideImg ? 340 : 500
      const imgH = hasSideImg ? 270 : 400
      const imgX = hasSideImg ? 30 : (W - imgW) / 2

      // 俯视图
      ctx.fillStyle = '#f1f5f9'
      this._roundRect(ctx, imgX, y, imgW, imgH, 12)
      ctx.fill()
      ctx.save()
      ctx.beginPath(); this._roundRect(ctx, imgX, y, imgW, imgH, 12); ctx.clip()
      ctx.drawImage(topImg, imgX, y, imgW, imgH)
      ctx.restore()
      ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1
      ctx.beginPath(); this._roundRect(ctx, imgX, y, imgW, imgH, 12); ctx.stroke()

      ctx.fillStyle = '#64748b'; ctx.font = '20px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(`俯视图 · 重合度 ${d.topSimilarity}%`, imgX + imgW / 2, y + imgH + 26)

      // 侧面图
      if (hasSideImg && sideImg) {
        const sx = imgX + imgW + 20
        ctx.fillStyle = '#f1f5f9'
        this._roundRect(ctx, sx, y, imgW, imgH, 12)
        ctx.fill()
        ctx.save()
        ctx.beginPath(); this._roundRect(ctx, sx, y, imgW, imgH, 12); ctx.clip()
        ctx.drawImage(sideImg, sx, y, imgW, imgH)
        ctx.restore()
        ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1
        ctx.beginPath(); this._roundRect(ctx, sx, y, imgW, imgH, 12); ctx.stroke()

        ctx.fillStyle = '#64748b'; ctx.font = '20px sans-serif'
        ctx.fillText(`侧面图 · 重合度 ${d.sideSimilarity}%`, sx + imgW / 2, y + imgH + 26)
      }

      y += imgH + 48
    }

    // --- 测量数据 ---
    ctx.fillStyle = '#1e293b'; ctx.font = 'bold 26px sans-serif'; ctx.textAlign = 'left'
    ctx.fillText('测量数据', 30, y)
    y += 40

    const gridX = 30, cellW = (W - 60) / 2
    const items = [
      { label: '头颅指数 CI', value: d.topCI, unit: '', note: '参考 75-85' },
      { label: '不对称指数', value: d.topCVAI, unit: '%', note: '正常 小于3.5%' },
      { label: '头长', value: d.topLength, unit: 'mm', note: '眉心到后脑勺' },
      { label: '头宽', value: d.topWidth, unit: 'mm', note: '两耳之间宽度' },
      { label: '头围（估算）', value: d.topCirc, unit: 'mm', note: '基于椭圆近似' },
      { label: '不对称值', value: d.topCVA, unit: 'mm', note: '正常 小于5mm' },
    ]
    const cellH = 80
    for (let i = 0; i < items.length; i++) {
      const cx = gridX + (i % 2) * cellW
      const cy = y + Math.floor(i / 2) * cellH
      const it = items[i]

      ctx.fillStyle = '#f8fafc'
      this._roundRect(ctx, cx + 2, cy + 2, cellW - 4, cellH - 4, 8); ctx.fill()

      ctx.fillStyle = '#1e293b'; ctx.font = 'bold 30px sans-serif'; ctx.textAlign = 'center'
      ctx.fillText(it.value + it.unit, cx + cellW / 2, cy + 28)
      ctx.fillStyle = '#64748b'; ctx.font = '20px sans-serif'
      ctx.fillText(it.label, cx + cellW / 2, cy + 52)
      ctx.fillStyle = '#94a3b8'; ctx.font = '18px sans-serif'
      ctx.fillText(it.note, cx + cellW / 2, cy + 72)
    }
    y += Math.ceil(items.length / 2) * cellH + 32

    // --- AI 分析 ---
    if (d.aiExplanation) {
      ctx.fillStyle = '#1e293b'; ctx.font = 'bold 26px sans-serif'; ctx.textAlign = 'left'
      ctx.fillText('AI 分析', 30, y)
      y += 36

      const text = d.aiExplanation.slice(0, 200)
      ctx.fillStyle = '#334155'; ctx.font = '22px sans-serif'
      const charsPerLine = 26, lineH = 34
      for (let i = 0; i < text.length; i += charsPerLine) {
        const line = text.slice(i, i + charsPerLine)
        ctx.fillText(line, 30, y)
        y += lineH
      }
      y += 16
    }

    // --- 分隔线 ---
    ctx.strokeStyle = '#f1f5f9'; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(W - 30, y); ctx.stroke()
    y += 24

    // --- Meta ---
    ctx.fillStyle = '#64748b'; ctx.font = '20px sans-serif'; ctx.textAlign = 'center'
    const meta = [d.scaleNote, `可信度 ${d.topConfidence}%`].filter(Boolean).join(' · ')
    ctx.fillText(meta, W / 2, y)
    y += 28
    ctx.fillStyle = '#cbd5e1'; ctx.font = '18px sans-serif'
    ctx.fillText('本报告仅供家庭参考，不构成医疗建议', W / 2, y)

    // 计算实际需要的高度，裁剪 canvas
    const realH = Math.ceil((y + 50) * dpr)
    // 全量导出即可，canvas 已绑完

    // 5. 导出并保存
    const tempRes = await this._wx(wx.canvasToTempFilePath, {
      canvas, x: 0, y: 0, width: W * dpr, height: realH, destWidth: W * dpr, destHeight: realH
    })
    await this._wx(wx.saveImageToPhotosAlbum, { filePath: tempRes.tempFilePath })
    wx.showToast({ title: '报告已保存到相册', icon: 'success' })
  },

  /** base64 → 临时文件, 返回路径 */
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
