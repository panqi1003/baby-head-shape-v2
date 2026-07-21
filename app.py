"""
Baby Head Shape Analyzer - FastAPI v0.3.1
Multi-angle + DeepSeek AI
"""

import base64
import cv2
import logging
import os
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# ── 加载 .env (必须在读取 os.environ 之前!) ─────────────
BASE_DIR = Path(__file__).parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# ── 配置 (load_dotenv 之后读取，确保 .env 值生效) ───────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format='%(asctime)s %(levelname)s %(message)s'
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
API_SECRET_KEY = os.environ.get("API_SECRET_KEY", None)
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "true").lower() not in ("false", "0", "no")
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX", "30"))

from head_analyzer import analyze_head_shape, COIN_DIAMETER_MM
from ai_advisor import analyze_with_deepseek, generate_fallback_advice

# ── 简单速率限制 ────────────────────────────────────────
_rate_limits: Dict[str, list] = {}

def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    if client_ip not in _rate_limits:
        _rate_limits[client_ip] = []
    _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if t > window_start]
    if len(_rate_limits[client_ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limits[client_ip].append(now)
    return True


def _top_analysis_text(m: dict) -> dict:
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
    return {"summary": " | ".join(parts), "ci": ci, "cvai": cvai, "severity": m['severity']}


def _side_analysis_text(flatness_score: float) -> str:
    if flatness_score < 0.15:
        return "后枕部轮廓圆润，弧度接近参考曲线。"
    elif flatness_score < 0.30:
        return "后枕部有轻微扁平迹象，建议多变换睡姿、增加俯趴时间。"
    elif flatness_score < 0.50:
        return "后枕部扁平较明显，建议关注睡姿并定期拍照对比。"
    else:
        return "后枕部扁平较明显，建议带照片给儿科医生参考。"


# ── 生命周期 ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Baby Head Shape Analyzer v0.3.1 starting...")
    yield
    logging.info("Service stopped")


app = FastAPI(title="Baby Head Shape Analyzer", version="0.3.1", lifespan=lifespan)

# ── CORS (allow_credentials=False when origins=*) ────────
_cors_origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins.split(","),
    allow_credentials=False if _cors_origins == "*" else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 鉴权 ────────────────────────────────────────────────
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not AUTH_REQUIRED:
        return await call_next(request)
    if request.url.path.startswith("/analyze") or request.url.path.startswith("/ai_analysis"):
        if not API_SECRET_KEY:
            return JSONResponse(
                {"detail": "Server not configured: API_SECRET_KEY missing. Set it in .env or environment."},
                status_code=500,
            )
        if request.headers.get("X-API-Key") != API_SECRET_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)


# ── 限流 ────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/analyze"):
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            return JSONResponse({"detail": "请求过于频繁，请稍后再试"}, status_code=429)
    return await call_next(request)


# ── 静态文件 ────────────────────────────────────────────
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _read_index_html() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


def _read_img(upload: UploadFile, max_side: int = 1200) -> Optional[np.ndarray]:
    contents = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        logging.warning("Upload too large: %d bytes", len(contents))
        while True:
            chunk = upload.file.read(64 * 1024)
            if not chunk:
                break
        return None
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
    return HTMLResponse(content=_read_index_html(),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.1"}


@app.post("/analyze")
async def analyze(
    top_image: UploadFile = File(...),
    reference_mm: float = Form(COIN_DIAMETER_MM),
    auto_detect: bool = Form(True),
    use_reference: bool = Form(True),
    age_months: int = Form(None),
    guide_frame: bool = Form(False),
):
    img = _read_img(top_image)
    if img is None:
        return JSONResponse({"success": False, "error": "无法解析俯视图，请确认上传了有效的图片"}, status_code=400)
    result = analyze_head_shape(img, reference_mm=reference_mm,
        auto_detect_reference=auto_detect, use_reference=use_reference,
        age_months=age_months, guide_frame=guide_frame)
    if not result.success:
        return JSONResponse({"success": False, "error": result.error_message, "steps": result.processing_steps})
    _, buf = cv2.imencode('.jpg', result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64 = base64.b64encode(buf).decode('utf-8')
    m = result.measurements
    top_data = {"head_length_mm": m.head_length_mm, "head_width_mm": m.head_width_mm,
        "head_circumference_mm": m.head_circumference_mm, "ci": m.ci, "cvai": m.cvai,
        "cva_mm": m.cva_mm, "severity": m.severity.value, "confidence": m.confidence}
    top_analysis = _top_analysis_text(top_data)
    has_ref = any("检测到参照物" in s for s in result.processing_steps)
    has_guide = any("引导框" in s for s in result.processing_steps)
    if has_ref:
        scale_note = "参照物校准 · 精度较高"
    elif has_guide:
        scale_note = "引导框 + 年龄估算 · 精度良好"
    else:
        scale_note = "年龄估算 · 精度一般"
    return {
        "success": True, "steps": result.processing_steps,
        "measurements": top_data, "analysis": top_analysis,
        "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        "has_reference": has_ref, "scale_note": scale_note,
        "standard_compare": getattr(result, 'standard_compare', None),
    }


@app.post("/analyze_side")
async def analyze_side(image: UploadFile = File(...), guide_frame: bool = Form(False), side: str = Form("")):
    img = _read_img(image)
    if img is None:
        return JSONResponse({"success": False, "error": "无法解析侧面图，请确认上传了有效的图片"}, status_code=400)
    try:
        from side_analyzer import analyze_side_profile
        result = analyze_side_profile(img)
        if not result:
            return {"success": False, "error": "侧面分析失败，请重试"}
        contour_list = result.pop("_head_contour", None)
        comp_data, side_b64, compare_b64 = None, None, None
        if contour_list and len(contour_list) >= 20:
            contour = np.array(contour_list, dtype=np.int32).reshape(-1, 1, 2)
            sa = img.copy()
            cv2.drawContours(sa, [contour], -1, (0, 200, 80), 2)
            _, sb = cv2.imencode('.jpg', sa, [cv2.IMWRITE_JPEG_QUALITY, 85])
            side_b64 = base64.b64encode(sb).decode('utf-8')
            try:
                from standard_compare import draw_comparison
                comp_img, comp_data = draw_comparison(img, contour, view='side', side_result=result, side=side or 'left')
                _, cb = cv2.imencode('.jpg', comp_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                compare_b64 = base64.b64encode(cb).decode('utf-8')
                if comp_data:
                    for k in ('gap_max', 'gap_avg', 'gap_top', 'gap_mid', 'gap_bot'):
                        if k in comp_data:
                            result[k] = comp_data[k]
            except Exception:
                logging.warning("Side comparison failed", exc_info=True)
        fs = result.get('posterior_flatness', 0)
        return {
            "success": True, "side": side, "measurements": result,
            "analysis": {"side": side, "flatness_category": result.get('flatness_category', ''),
                "summary": f"后枕部{result.get('flatness_category', '')}",
                "detail": _side_analysis_text(fs)},
            "annotated_image": f"data:image/jpeg;base64,{side_b64}" if side_b64 else None,
            "standard_compare": comp_data, "compare_image": f"data:image/jpeg;base64,{compare_b64}" if compare_b64 else None,
        }
    except Exception:
        logging.error("Side analysis error", exc_info=True)
        return JSONResponse({"success": False, "error": "分析出错，请重试"}, status_code=500)


@app.post("/check_reference")
async def check_reference(image: UploadFile = File(...)):
    img = _read_img(image)
    if img is None:
        return {"has_reference": False}
    from head_analyzer import detect_coin, detect_skin_adaptive, clean_mask
    try:
        skin = detect_skin_adaptive(img)
        skin = clean_mask(skin)
    except Exception:
        skin = np.zeros(img.shape[:2], dtype=np.uint8)
    coin = detect_coin(img, skin)
    return {"has_reference": coin is not None}


@app.post("/ai_analysis")
async def ai_analysis(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"success": False, "error": "JSON 格式错误"}, status_code=400)
    top_data = body.get("top_measurements", {})
    side_raw = body.get("side_measurements", None)
    side_data = {k: v for k, v in side_raw.items() if not k.startswith("_")} if side_raw else None
    age_months = body.get("age_months", None)
    try:
        ai_result = analyze_with_deepseek(top_data, side_data, age_months)
        if ai_result is None:
            ai_result = generate_fallback_advice(top_data, side_data)
            return {"success": True, "ai_analysis": None, "fallback_advice": ai_result}
        return {"success": True, "ai_analysis": ai_result, "fallback_advice": None}
    except Exception:
        logging.error("AI analysis error", exc_info=True)
        try:
            fb = generate_fallback_advice(top_data, side_data)
            return {"success": True, "ai_analysis": None, "fallback_advice": fb}
        except Exception:
            return JSONResponse({"success": False, "error": "分析出错，请重试"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
