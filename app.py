"""
Baby Head Shape Analyzer - FastAPI v0.2
Multi-angle + DeepSeek AI
"""

import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from head_analyzer import analyze_head_shape, COIN_DIAMETER_MM
from ai_advisor import analyze_with_deepseek, generate_fallback_advice


def _top_analysis_text(m):
    """俯视图独立分析 (规则引擎)"""
    ci, cvai = m['ci'], m['cvai']
    parts = []
    if 75 <= ci <= 85:
        parts.append(f"头颅指数 CI={ci:.1f}，在常见范围内")
    elif ci > 85:
        parts.append(f"头颅指数 CI={ci:.1f}，偏扁头倾向")
    else:
        parts.append(f"头颅指数 CI={ci:.1f}，偏长头倾向")

    if cvai < 3.5:
        parts.append(f"不对称指数 CVAI={cvai:.1f}%，左右对称性良好")
    elif cvai < 6.25:
        parts.append(f"不对称指数 CVAI={cvai:.1f}%，有轻微不对称")
    else:
        parts.append(f"不对称指数 CVAI={cvai:.1f}%，不对称较明显")

    return {"summary": " · ".join(parts), "ci": ci, "cvai": cvai, "severity": m['severity']}


def _side_analysis_text(flatness, score):
    """侧面图独立分析 (规则引擎)"""
    if score < 0.15:
        return "后枕部轮廓圆润，曲率正常。"
    elif score < 0.30:
        return "后枕部有轻微扁平迹象，建议多变换睡姿、增加俯趴时间。"
    elif score < 0.50:
        return "后枕部扁平较明显，建议关注睡姿并定期拍照对比。"
    else:
        return "后枕部扁平明显，建议咨询儿科医生进行专业评估。"

app = FastAPI(title="Baby Head Shape Analyzer", version="0.2.0")

BASE_DIR = Path(__file__).parent
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


def _read_img(upload: UploadFile, max_side: int = 1200) -> Optional[np.ndarray]:
    contents = upload.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=INDEX_HTML,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.post("/analyze")
async def analyze(
    top_image: UploadFile = File(...),
    reference_mm: float = Form(COIN_DIAMETER_MM),
    auto_detect: bool = Form(True),
    use_reference: bool = Form(True),
    age_months: int = Form(None),
):
    """俯视图分析 — 返回测量数据 + 独立分析结论"""
    img = _read_img(top_image)
    if img is None:
        return JSONResponse({"success": False, "error": "can not parse top image"}, status_code=400)

    result = analyze_head_shape(img, reference_mm=reference_mm,
        auto_detect_reference=auto_detect, use_reference=use_reference,
        age_months=age_months)
    if not result.success:
        return JSONResponse({"success": False, "error": result.error_message,
            "steps": result.processing_steps})

    _, buf = cv2.imencode('.jpg', result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode('utf-8')

    m = result.measurements
    top_data = {
        "head_length_mm": m.head_length_mm, "head_width_mm": m.head_width_mm,
        "head_circumference_mm": m.head_circumference_mm,
        "ci": m.ci, "cvai": m.cvai, "cva_mm": m.cva_mm,
        "severity": m.severity.value, "confidence": m.confidence,
    }

    # 俯视图独立分析 (规则引擎, 快)
    top_analysis = _top_analysis_text(top_data)

    # 判断是否实际检测到了参照物
    has_ref = any("参照物" in s for s in result.processing_steps)

    return {
        "success": True,
        "steps": result.processing_steps,
        "measurements": top_data,
        "analysis": top_analysis,
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "has_reference": has_ref,
        "scale_note": "已检测到参照物，测量精度较高" if has_ref else "未检测到参照物，使用平均头宽估算，精度较低",
    }


@app.post("/analyze_side")
async def analyze_side(image: UploadFile = File(...)):
    """侧面图分析 — 返回测量数据 + 独立分析结论"""
    img = _read_img(image)
    if img is None:
        return JSONResponse({"success": False, "error": "无法解析侧面图"}, status_code=400)

    try:
        from side_analyzer import analyze_side_profile
        result = analyze_side_profile(img)
        if not result:
            return {"success": False, "error": "侧面分析失败"}

        from sam_detector import detect_head
        sam_out = detect_head(img)
        side_b64 = None
        if sam_out:
            sa = img.copy()
            cv2.drawContours(sa, [sam_out[1]], -1, (0, 200, 80), 2)
            _, sb = cv2.imencode('.jpg', sa, [cv2.IMWRITE_JPEG_QUALITY, 85])
            side_b64 = base64.b64encode(sb).decode('utf-8')

        # 侧面独立分析 (规则引擎)
        flatness = result.get('flatness_category', '')
        flatness_score = result.get('posterior_flatness', 0)
        side_analysis = {
            "flatness_category": flatness,
            "summary": f"后枕部{flatness}",
            "detail": _side_analysis_text(flatness, flatness_score),
        }

        return {
            "success": True,
            "measurements": result,
            "analysis": side_analysis,
            "annotated_image": f"data:image/jpeg;base64,{side_b64}" if side_b64 else None,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/check_reference")
async def check_reference(image: UploadFile = File(...)):
    """快速检测照片中是否有参照物 (硬币等圆形物体)，排除肤色区域误检"""
    img = _read_img(image)
    if img is None:
        return {"has_reference": False}

    from head_analyzer import detect_coin, detect_skin_adaptive, clean_mask
    import cv2, numpy as np

    try:
        skin = detect_skin_adaptive(img)
        skin = clean_mask(skin)
    except Exception:
        skin = np.zeros(img.shape[:2], dtype=np.uint8)

    # 只用肤色排除法检测, 不用全图回退 (回退会把头部当硬币)
    coin = detect_coin(img, skin)
    return {"has_reference": coin is not None}


@app.post("/ai_analysis")
async def ai_analysis(request: Request):
    """
    综合分析: 接收俯视图+侧面图测量数据，调用 DeepSeek。
    请求体: JSON {top_measurements, top_analysis, side_measurements, side_analysis, age_months}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "JSON 格式错误"}, status_code=400)

    top_data = body.get("top_measurements", {})
    side_data = body.get("side_measurements", None)
    age_months = body.get("age_months", None)

    try:
        ai_result = analyze_with_deepseek(top_data, side_data, age_months)
        fallback = generate_fallback_advice(top_data, side_data)
        return {"success": True, "ai_analysis": ai_result, "fallback_advice": fallback}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
