"""
婴幼儿头型分析引擎 — MVP 版本
纯传统计算机视觉，零训练数据，零 ML 依赖。

技术路线:
  1. 肤色检测 (YCrCb + HSV 双通道融合) → 提取头部区域
  2. 形态学处理 → 去噪 + 闭合
  3. 最大连通域 → 头部轮廓
  4. 椭圆拟合 → 头长 / 头宽
  5. 硬币检测 (霍夫圆) → 尺度校准 像素→毫米
  6. CI / CVAI 计算 → 风险分级 → 指导建议
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import Enum

# SAM 模型 (延迟加载)
_sam_model = None

def _get_sam_model():
    """延迟加载 MobileSAM，首次调用时自动下载模型(约40MB)"""
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM('mobile_sam.pt')
    return _sam_model


# ============================================================
# 数据结构
# ============================================================

class Severity(Enum):
    NORMAL = "正常"
    MILD = "轻度"
    MODERATE = "中度"
    SEVERE = "重度"


@dataclass
class HeadMeasurements:
    """头型测量结果"""
    head_length_mm: float          # 头长 (前后径)
    head_width_mm: float           # 头宽 (左右径)
    head_circumference_mm: float   # 头围 (估算)
    ci: float                      # 头颅指数 CI = 宽/长 × 100
    cvai: float                    # 颅顶不对称指数 (%)
    cva_mm: float                  # 颅顶不对称值 (mm)
    severity: Severity             # 严重程度
    confidence: float              # 分析置信度 0-1


@dataclass
class AnalysisResult:
    """完整分析结果"""
    success: bool
    measurements: Optional[HeadMeasurements] = None
    annotated_image: Optional[np.ndarray] = None
    error_message: str = ""
    processing_steps: List[str] = field(default_factory=list)


# ============================================================
# 可调参数
# ============================================================

# 肤色检测 — YCrCb 空间阈值 (婴儿肤色)
SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

# HSV 肤色阈值 (辅助)
SKIN_HSV_LOWER = np.array([0, 20, 40], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([25, 150, 255], dtype=np.uint8)

# 形态学参数
MORPH_KERNEL_SIZE = 5
MORPH_CLOSE_ITERATIONS = 3

# 椭圆拟合最小轮廓点数
MIN_CONTOUR_POINTS = 30

# 硬币检测 (1元人民币直径 = 25mm)
COIN_DIAMETER_MM = 25.0
HOUGH_CIRCLE_DP = 1.2
HOUGH_CIRCLE_MIN_DIST = 100
HOUGH_CIRCLE_PARAM1 = 100
HOUGH_CIRCLE_PARAM2 = 30
HOUGH_CIRCLE_MIN_RADIUS = 20
HOUGH_CIRCLE_MAX_RADIUS = 150

# CI 分级阈值
CI_BRACHY_MILD = 85
CI_BRACHY_MODERATE = 90
CI_BRACHY_SEVERE = 100
CI_DOLICHO_MILD = 75
CI_DOLICHO_MODERATE = 70

# CVAI 分级阈值 (Argentea 标准)
CVAI_MILD = 3.5
CVAI_MODERATE = 6.25
CVAI_SEVERE = 8.75


# ============================================================
# 1. 图像预处理
# ============================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """CLAHE 自适应直方图均衡化，改善不均匀光照"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ============================================================
# 2. 肤色分割 (自适应)
# ============================================================

def _sample_center_skin(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从画面中心区域采样，估计当前照片的肤色均值和范围。
    中心 1/3 区域的像素最可能是婴儿头部。
    返回 (mean_ycrcb, std_ycrcb)
    """
    h, w = image.shape[:2]
    # 取中心 1/3 区域
    y1, y2 = h // 3, 2 * h // 3
    x1, x2 = w // 3, 2 * w // 3
    center_region = image[y1:y2, x1:x2]

    ycrcb = cv2.cvtColor(center_region, cv2.COLOR_BGR2YCrCb)
    pixels = ycrcb.reshape(-1, 3).astype(np.float32)

    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)

    return mean, std


def detect_skin_adaptive(image: np.ndarray) -> np.ndarray:
    """
    自适应肤色检测:
      1. 采样中心区域 → 学习当前照片的肤色分布
      2. 以 mean ± 2.5*std 为阈值 (比固定阈值宽得多)
      3. 同时保留宽泛的 YCrCb 回退范围
      4. 取并集 → 确保不遗漏
    """
    h, w = image.shape[:2]
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 方案A: 固定宽阈值 (做保底)
    # Cr: 125-180 (比之前的 133-173 更宽), Cb: 70-140 (比之前的 77-127 更宽)
    fixed_lower = np.array([0, 125, 70], dtype=np.uint8)
    fixed_upper = np.array([255, 180, 140], dtype=np.uint8)
    skin_fixed = cv2.inRange(ycrcb, fixed_lower, fixed_upper)

    # 方案B: 自适应阈值 (从当前照片学习)
    try:
        mean, std = _sample_center_skin(image)
        # mean ± 2.5 std, 至少留 ±20 的边距
        margin = np.maximum(std * 2.5, 20)
        adaptive_lower = np.clip(mean - margin, 0, 255).astype(np.uint8)
        adaptive_upper = np.clip(mean + margin, 0, 255).astype(np.uint8)
        skin_adaptive = cv2.inRange(ycrcb, adaptive_lower, adaptive_upper)
    except Exception:
        skin_adaptive = skin_fixed  # fallback

    # 方案C: HSV 宽阈值 (对高光/阴影区域有补充作用)
    hsv_lower = np.array([0, 15, 30], dtype=np.uint8)
    hsv_upper = np.array([30, 180, 255], dtype=np.uint8)
    skin_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # 三路融合
    combined = cv2.bitwise_or(skin_fixed, skin_adaptive)
    combined = cv2.bitwise_or(combined, skin_hsv)

    return combined


# ============================================================
# GrabCut 精修 (可选, 提升轮廓精度)
# ============================================================

def refine_with_grabcut(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    用 GrabCut 从初始肤色 mask 精修头部边界。
    如果 GrabCut 失败或结果更差，自动回退到原始 mask。
    """
    h, w = image.shape[:2]
    original_fg_pixels = np.sum(mask > 0)

    # 准备 GrabCut mask
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    gc_mask[mask > 0] = cv2.GC_PR_FGD  # 肤色区域 → 可能前景

    # 边缘区域 → 可能背景
    border = 10
    gc_mask[:border, :] = cv2.GC_PR_BGD
    gc_mask[-border:, :] = cv2.GC_PR_BGD
    gc_mask[:, :border] = cv2.GC_PR_BGD
    gc_mask[:, -border:] = cv2.GC_PR_BGD

    # 腐蚀肤色区域中心 → 确定前景 (GrabCut 需要)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    sure_fg = cv2.erode(mask, kernel, iterations=3)

    if np.sum(sure_fg > 0) < 100:
        # 确定前景太小，GrabCut 不可靠，回退
        return mask

    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        gc_mask, _, _ = cv2.grabCut(image, gc_mask, None, bgd_model, fgd_model,
                                      3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return mask

    # 提取前景
    result = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 安全检查: 如果 GrabCut 结果太差，回退原始 mask
    result_fg_pixels = np.sum(result > 0)
    if result_fg_pixels < original_fg_pixels * 0.3 or result_fg_pixels < 100:
        return mask

    return result


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """形态学清理：闭合小孔洞 + 去除孤立噪点"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


# ============================================================
# 3. 头部轮廓提取 (智能筛选, 抵抗背景干扰)
# ============================================================

# 婴儿头部面积占比范围 (相对于整张照片)
MIN_HEAD_AREA_RATIO = 0.03   # 头部至少占画面 3%
MAX_HEAD_AREA_RATIO = 0.60   # 头部最多占画面 60%

# 婴儿平均头部尺寸参考 (用于无参照物模式)
# 新生儿头宽约 85-95mm, 3-12月约 90-105mm
AVG_INFANT_HEAD_WIDTH_MM = 95.0  # 默认值 (3-6月龄平均)


def estimate_head_width_by_age(age_months: Optional[int]) -> float:
    """
    根据月龄估算婴儿平均头宽 (mm)。
    基于 WHO 0-24 月龄头围生长标准反推头宽。
    """
    if age_months is None:
        return AVG_INFANT_HEAD_WIDTH_MM

    # WHO 头围 P50 (cm): 0m=35, 3m=40, 6m=43, 9m=45, 12m=46, 18m=48, 24m=49
    # 头宽 ≈ 头围 / π * 0.72 (经验系数)
    age_to_circ = {0: 35, 1: 37, 2: 39, 3: 40, 4: 41.5, 5: 42.5, 6: 43.5,
                   7: 44.3, 8: 45, 9: 45.5, 10: 46, 11: 46.3, 12: 46.8,
                   15: 47.5, 18: 48, 21: 48.5, 24: 49}

    m = min(age_months, 24)
    # 找最接近的月龄
    keys = sorted(age_to_circ.keys())
    for i, k in enumerate(keys):
        if k >= m:
            circ = age_to_circ[k]
            break
    else:
        circ = age_to_circ[keys[-1]]

    return round(circ / math.pi * 7.2, 1)  # 头围(cm) → 头宽(mm)


def _score_contour(contour: np.ndarray, img_center: Tuple[int, int],
                    img_area: int) -> float:
    """
    对肤色轮廓打分 (越高越像头部)。
    综合: 中心位置 + 椭圆度 + 面积合理性
    """
    area = cv2.contourArea(contour)
    if area < 50:
        return 0.0

    area_ratio = area / img_area

    # 面积太大或太小 → 扣分
    if area_ratio < MIN_HEAD_AREA_RATIO:
        return 0.0
    if area_ratio > MAX_HEAD_AREA_RATIO:
        area_score = max(0, 1.0 - (area_ratio - MAX_HEAD_AREA_RATIO) * 3)
    else:
        area_score = 1.0

    # 距画面中心越近越好 (头部通常居中)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return 0.0
    cx_c = M['m10'] / M['m00']
    cy_c = M['m01'] / M['m00']
    max_dist = math.sqrt(img_area) / 2
    dist = math.sqrt((cx_c - img_center[0])**2 + (cy_c - img_center[1])**2)
    center_score = max(0, 1.0 - dist / max_dist)

    # 椭圆度 (拟合椭圆的重叠率越高越好)
    if len(contour) >= 10:
        try:
            ellipse = cv2.fitEllipse(contour)
            # 生成椭圆 mask 并计算与轮廓的重叠度
            e_mask = np.zeros((int(img_center[1] * 2), int(img_center[0] * 2)),
                               dtype=np.uint8)
            cv2.ellipse(e_mask, ellipse, 255, -1)
            cnt_mask = np.zeros_like(e_mask)
            cv2.drawContours(cnt_mask, [contour], -1, 255, -1)
            intersection = np.sum((e_mask > 0) & (cnt_mask > 0))
            union = np.sum((e_mask > 0) | (cnt_mask > 0))
            ellipse_score = intersection / union if union > 0 else 0
        except Exception:
            ellipse_score = 0.3
    else:
        ellipse_score = 0.1

    # 综合评分 (椭圆度权重最高, 能有效排除手臂/不规则背景物体)
    return area_score * 0.2 + center_score * 0.3 + ellipse_score * 0.5


def extract_head_contour(mask: np.ndarray,
                          image_shape: Tuple[int, int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    从肤色 mask 中智能提取头部轮廓。
    不再简单取最大面积, 而是综合评分 (位置+形状+面积),
    有效排除背景中的肤色干扰 (手臂、家具等)。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = image_shape[:2]
    img_center = (w / 2, h / 2)
    img_area = h * w

    # 按综合评分排序
    scored = [(c, _score_contour(c, img_center, img_area)) for c in contours]
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored[0][1] < 0.1:
        return None

    best_contour = scored[0][0]

    # 凸包用于 mask
    hull = cv2.convexHull(best_contour)
    head_mask = np.zeros_like(mask)
    cv2.drawContours(head_mask, [hull], -1, 255, -1)

    return head_mask, best_contour


# ============================================================
# 4. 椭圆拟合
# ============================================================

# ============================================================
# 头发干扰补偿
# ============================================================

def detect_hair_interference(contour: np.ndarray) -> float:
    """
    检测头发干扰程度 (0=无干扰, 1=严重干扰)。

    原理:
      1. 先用 Douglas-Peucker 平滑去除 SAM 像素锯齿
      2. 再计算轮廓不规则度 = 周长/等面积椭圆周长
      3. 综合纹理特征 (在轮廓外侧区域检测高频纹理 → 头发)
    """
    area = cv2.contourArea(contour)
    if area < 1:
        return 0.0

    # 平滑: 去除 SAM 输出的像素级锯齿 (epsilon = 0.5% 轮廓弧长)
    arc_len = cv2.arcLength(contour, True)
    epsilon = arc_len * 0.005
    smooth = cv2.approxPolyDP(contour, epsilon, True)

    smooth_perimeter = cv2.arcLength(smooth, True)
    # 等面积正圆周长
    ideal_perimeter = 2 * math.pi * math.sqrt(area / math.pi)
    if ideal_perimeter < 1:
        return 0.0

    # 平滑后的不规则度: 1.0=完美圆, >1.03=有头发
    irregularity = smooth_perimeter / ideal_perimeter

    # 映射: 1.00=0%, 1.03=30%, 1.08=60%, 1.15+=100%
    hair_score = max(0, min(1.0, (irregularity - 1.0) / 0.12))
    return round(hair_score, 2)


def build_hair_mask(image: np.ndarray, head_mask: np.ndarray) -> np.ndarray:
    """
    在 SAM 分割出的头部区域内，识别哪些像素是头发。

    特征1 (深色): HSV 中 V < 80 的像素很可能是深色头发
    特征2 (纹理): 局部标准差 > 15 的区域 (头发纹理 vs 光滑头皮)

    返回: hair_mask (255=头发, 0=头皮/皮肤)
    """
    h, w = image.shape[:2]

    # 仅分析 SAM mask 内的区域
    roi = cv2.bitwise_and(image, image, mask=head_mask)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 特征1: 深色 = 头发候选
    _, _, v_channel = cv2.split(hsv)
    dark_mask = (v_channel < 80).astype(np.uint8) * 255
    dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=head_mask)

    # 特征2: 纹理密度 (局部标准差)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    sqr = (gray.astype(np.float32) - blur.astype(np.float32)) ** 2
    local_std = np.sqrt(cv2.GaussianBlur(sqr, (7, 7), 0))
    texture_mask = (local_std > 12).astype(np.uint8) * 255
    texture_mask = cv2.bitwise_and(texture_mask, texture_mask, mask=head_mask)

    # 综合: 深色 AND 有纹理 = 头发
    hair_raw = cv2.bitwise_and(dark_mask, texture_mask)

    # 形态学闭合: 连接碎发区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.morphologyEx(hair_raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return hair_mask


def refine_contour_under_hair(image: np.ndarray, head_mask: np.ndarray,
                                contour: np.ndarray, hair_score: float) -> np.ndarray:
    """
    头发补偿: 用颜色+纹理检测头发，从 SAM 轮廓中刨除头发区域。

    1. 构建头发 mask (深色+纹理)
    2. 头发 mask 膨胀到轮廓边界
    3. 原始 mask 减去头发 → 得到头皮真实边界
    4. 提取新轮廓
    """
    if hair_score < 0.05:
        return contour

    h, w = image.shape[:2]

    # 构建头发 mask
    hair_mask = build_hair_mask(image, head_mask)

    hair_ratio = np.sum(hair_mask > 0) / (np.sum(head_mask > 0) + 1)
    if hair_ratio < 0.02:
        return contour  # 几乎没有头发

    # 将头发区域从 mask 中移除
    clean_mask = cv2.bitwise_and(head_mask, head_mask, mask=cv2.bitwise_not(hair_mask))

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 提取新轮廓
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return contour

    # 取最大的连通域
    best = max(contours, key=cv2.contourArea)

    # 安全检查: 新轮廓面积不能太小
    orig_area = cv2.contourArea(contour)
    new_area = cv2.contourArea(best)
    if new_area < orig_area * 0.6:
        return contour  # 头发抠太多了, 回退

    return best


def fit_head_ellipse(contour: np.ndarray) -> Optional[Tuple]:
    """拟合椭圆 → ((cx, cy), (major_axis, minor_axis), angle_deg)"""
    if len(contour) < MIN_CONTOUR_POINTS:
        return None
    try:
        return cv2.fitEllipse(contour)
    except cv2.error:
        return cv2.minAreaRect(contour)


def measure_head_pca(contour: np.ndarray) -> Optional[dict]:
    """
    PCA 主成分分析 — 比椭圆拟合更精确的头型测量。

    原理:
      1. 对轮廓点做 PCA → 找到头长方向(第一主成分)和头宽方向(第二主成分)
      2. 沿主成分方向投影所有轮廓点 → 实际头长
      3. 在中点做垂直线 → 交轮廓于两点 → 实际头宽
      4. 沿 30° 对角线方向跨度 → CVAI

    返回: {length_px, width_px, cvai_px, cva_px, center, angle_deg}
    """
    if len(contour) < MIN_CONTOUR_POINTS:
        return None

    # 1. 用椭圆拟合确定主轴方向 (椭圆天然对称, 不受轮廓不对称影响)
    pts = contour.reshape(-1, 2).astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean

    # 先拟合椭圆拿方向
    try:
        hull = cv2.convexHull(contour)
        ellipse = cv2.fitEllipse(hull)
        ellipse_angle_rad = math.radians(ellipse[2])
        v1_from_ellipse = np.array([math.cos(ellipse_angle_rad),
                                     math.sin(ellipse_angle_rad)])
    except Exception:
        v1_from_ellipse = None

    # PCA 找主方向 (用于跨度测量)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    v1_pca = eigenvectors[:, idx[0]]
    v2 = eigenvectors[:, idx[1]]

    # 主轴方向 = 椭圆方向 (更稳定), 回退到 PCA
    if v1_from_ellipse is not None:
        v1 = v1_from_ellipse
    else:
        v1 = v1_pca

    # 确保 v2 垂直于 v1
    v2 = np.array([-v1[1], v1[0]])

    angle_deg = math.degrees(math.atan2(v1[1], v1[0]))

    # 2. 确定前后方向 (两种方法综合)
    #    方法A: 鼻尖朝上 → y 最小 = 前
    #    方法B: 形状分析 → 窄尖的一端 = 前(面部), 宽圆的一端 = 后(枕部)
    #    综合二者, 不一致时以形状为准

    # 沿 v1 把轮廓分成前后两半
    proj_on_v1 = np.dot(pts - mean, v1)
    v1_min, v1_max = proj_on_v1.min(), proj_on_v1.max()
    mid = (v1_min + v1_max) / 2
    half1_mask = proj_on_v1 < mid  # v1 负方向的一半
    half2_mask = proj_on_v1 >= mid  # v1 正方向的一半

    if np.sum(half1_mask) < 5 or np.sum(half2_mask) < 5:
        # 分半失败, 回退
        ap_dir = v1
        lr_dir = v2
    else:
        # 形状法: 窄的那端 = 前面(鼻尖), 宽的那端 = 后面(后枕)
        def half_width_at_end(mask):
            pts_half = pts[mask]
            if len(pts_half) < 3:
                return 0
            # 取最远端 30% 的点的宽度
            proj = np.dot(pts_half - mean, v1)
            cutoff = np.percentile(abs(proj), 70)
            end_pts = pts_half[abs(proj) >= cutoff]
            proj_v2_half = np.dot(end_pts - mean, v2)
            return proj_v2_half.max() - proj_v2_half.min() if len(end_pts) >= 3 else 0

        w1 = half_width_at_end(half1_mask)
        w2 = half_width_at_end(half2_mask)

        if w1 > 0 and w2 > 0:
            # 窄的 = 前面
            if w1 < w2:
                front_end_is_v1_min = True
            else:
                front_end_is_v1_min = False
        else:
            # 回退到鼻尖朝上法
            y1 = np.mean(pts[half1_mask, 1])
            y2 = np.mean(pts[half2_mask, 1])
            front_end_is_v1_min = (y1 < y2)  # 更靠上的 = 前面

        # 构建前后方向
        if front_end_is_v1_min:
            ap_dir = -v1  # v1 负方向 = 前
        else:
            ap_dir = v1   # v1 正方向 = 前

        lr_dir = np.array([-ap_dir[1], ap_dir[0]])

    # 头长 = 沿前后方向的跨度
    proj_ap = np.dot(centered, ap_dir)
    length_px = proj_ap.max() - proj_ap.min()

    # 头宽 = 沿左右方向的跨度 (取中间 60% 避免前后端收窄)
    proj_lr = np.dot(centered, lr_dir)
    mid_mask = (proj_ap > -length_px * 0.3) & (proj_ap < length_px * 0.3)
    if np.sum(mid_mask) > 10:
        width_px = proj_lr[mid_mask].max() - proj_lr[mid_mask].min()
    else:
        width_px = proj_lr.max() - proj_lr.min()

    # 更新 v1/v2 为正确的方向 (用于后续可视化)
    v1 = ap_dir
    v2 = lr_dir

    # 4. CVAI: 30° 对角线跨度差异
    diag_angle = math.radians(30)
    v1_rad = math.atan2(v1[1], v1[0])
    d1_dir = np.array([math.cos(v1_rad + diag_angle), math.sin(v1_rad + diag_angle)])
    d2_dir = np.array([math.cos(v1_rad - diag_angle), math.sin(v1_rad - diag_angle)])
    d1_span = np.dot(centered, d1_dir).max() - np.dot(centered, d1_dir).min()
    d2_span = np.dot(centered, d2_dir).max() - np.dot(centered, d2_dir).min()

    if min(d1_span, d2_span) < 1.0:
        cvai_px, cva_px = 0.0, 0.0
    else:
        cva_px = abs(d1_span - d2_span)
        cvai_px = cva_px / min(d1_span, d2_span) * 100

    return {
        "length_px": length_px,
        "width_px": width_px,
        "cvai_px": cvai_px,
        "cva_px": cva_px,
        "center": mean,
        "angle_deg": angle_deg,
        "v1": v1, "v2": v2,
    }


# ============================================================
# 5. 硬币检测 (尺度校准)
# ============================================================

def detect_coin(image: np.ndarray, skin_mask: np.ndarray) -> Optional[Tuple]:
    """霍夫圆检测硬币 → ((cx, cy), radius_px)。排除肤色区域避免误检头部。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # 在非肤色区域搜索 (避免头部被当作大圆)
    non_skin = cv2.bitwise_not(skin_mask)
    gray_masked = cv2.bitwise_and(gray, gray, mask=non_skin)

    circles = cv2.HoughCircles(
        gray_masked, cv2.HOUGH_GRADIENT,
        dp=HOUGH_CIRCLE_DP, minDist=HOUGH_CIRCLE_MIN_DIST,
        param1=HOUGH_CIRCLE_PARAM1, param2=HOUGH_CIRCLE_PARAM2,
        minRadius=HOUGH_CIRCLE_MIN_RADIUS, maxRadius=HOUGH_CIRCLE_MAX_RADIUS,
    )
    if circles is None or len(circles) == 0:
        # 降级: 不加 mask 重试
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT,
            dp=HOUGH_CIRCLE_DP, minDist=HOUGH_CIRCLE_MIN_DIST,
            param1=HOUGH_CIRCLE_PARAM1, param2=HOUGH_CIRCLE_PARAM2,
            minRadius=HOUGH_CIRCLE_MIN_RADIUS, maxRadius=HOUGH_CIRCLE_MAX_RADIUS,
        )
        if circles is None or len(circles) == 0:
            return None

    circles = np.uint16(np.around(circles))
    # 排除与肤色区域重叠 >30% 的圆 (避免头部误检)
    valid = []
    for c in circles[0]:
        cx, cy, r = int(c[0]), int(c[1]), int(c[2])
        # 检查圆心是否在肤色区域内
        if 0 <= cy < skin_mask.shape[0] and 0 <= cx < skin_mask.shape[1]:
            if skin_mask[cy, cx] > 0:
                continue  # 圆心在皮肤上 → 跳过
        valid.append(c)
    if not valid:
        return None
    best = max(valid, key=lambda c: c[2])
    return (int(best[0]), int(best[1])), int(best[2])


def detect_coin_by_circularity(image: np.ndarray) -> Optional[Tuple]:
    """降级方案：基于轮廓圆形度筛选圆形物体"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if 0.7 < circularity < 1.3:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            score = circularity * area
            if score > best_score:
                best_score = score
                best = ((int(cx), int(cy)), int(radius))
    return best


# ============================================================
# 6. CVAI 计算
# ============================================================

# ============================================================
# 7. 分级 & 建议
# ============================================================

def classify_severity(ci: float, cvai: float) -> Severity:
    """CI + CVAI 综合判定严重程度"""
    def _ci_sev(c):
        if c > CI_BRACHY_SEVERE:     return Severity.SEVERE
        if c > CI_BRACHY_MODERATE:   return Severity.MODERATE
        if c > CI_BRACHY_MILD:       return Severity.MILD
        if c < CI_DOLICHO_MODERATE:  return Severity.MODERATE
        if c < CI_DOLICHO_MILD:      return Severity.MILD
        return Severity.NORMAL

    def _cvai_sev(c):
        if c > CVAI_SEVERE:   return Severity.SEVERE
        if c > CVAI_MODERATE: return Severity.MODERATE
        if c > CVAI_MILD:     return Severity.MILD
        return Severity.NORMAL

    order = [Severity.NORMAL, Severity.MILD, Severity.MODERATE, Severity.SEVERE]
    ci_s = _ci_sev(ci)
    cv_s = _cvai_sev(cvai)
    return ci_s if order.index(ci_s) > order.index(cv_s) else cv_s


def generate_advice(meas: HeadMeasurements) -> dict:
    """根据测量结果生成分级指导建议"""
    sev = meas.severity

    templates = {
        Severity.NORMAL: {
            "summary": "宝宝头型在正常范围内，继续保持良好习惯即可。",
            "actions": [
                "多变换宝宝睡姿，避免长时间固定一个方向",
                "每天俯趴训练 (Tummy Time) 累计30-60分钟",
                "喂奶和抱宝宝时交替左右手",
                "减少婴儿座椅/摇椅的连续使用时间",
            ],
            "monitoring": "建议每 1-2 个月拍照对比一次，观察趋势。",
        },
        Severity.MILD: {
            "summary": "轻微不对称/扁头倾向，通过家庭护理大多可改善。",
            "actions": [
                "增加俯趴训练至每天累计60分钟以上",
                "宝宝清醒时多趴着玩，减少头部受压",
                "调整婴儿床方向或玩具位置，引导宝宝主动转头",
                "如果4周后无明显改善，建议咨询儿科医生",
            ],
            "monitoring": "建议每 2-4 周拍照复测，追踪 CI/CVAI 变化。",
        },
        Severity.MODERATE: {
            "summary": "中度不对称/扁头，建议尽快就医评估。",
            "actions": [
                "预约儿科或儿童康复科医生进行专业评估",
                "在医生指导下进行俯趴训练和体位矫正",
                "医生可能建议矫形头盔治疗 (3-18月龄适用)",
            ],
            "monitoring": "每 1-2 周复测。4周内无改善应尽快就医。",
        },
        Severity.SEVERE: {
            "summary": "较严重的不对称/畸形，强烈建议立即就医。",
            "actions": [
                "尽快预约儿科/颅面外科/儿童康复科医生",
                "矫形头盔可能是必要干预手段 (黄金期: 4-8月龄)",
                "需排除颅缝早闭等需要手术干预的疾病",
            ],
            "monitoring": "遵医嘱定期复诊。自测仅作辅助记录。",
        },
    }

    return templates[sev]


# ============================================================
# 8. 可视化标注
# ============================================================

def _draw_annotations(image: np.ndarray, contour: np.ndarray,
                       ellipse: Tuple, coin_info: Optional[Tuple],
                       measurements: HeadMeasurements,
                       center: Tuple[float, float],
                       angle_deg: float) -> np.ndarray:
    """在图像上叠加分析标注"""
    annotated = image.copy()
    h, w = annotated.shape[:2]
    cx, cy = int(center[0]), int(center[1])

    GREEN = (0, 200, 80)
    BLUE = (220, 120, 0)
    RED = (50, 50, 220)
    YELLOW = (0, 220, 230)
    WHITE = (255, 255, 255)
    ORANGE = (0, 140, 240)

    # 头部轮廓
    cv2.drawContours(annotated, [contour], -1, GREEN, 2)

    # 拟合椭圆
    cv2.ellipse(annotated, (int(ellipse[0][0]), int(ellipse[0][1])),
                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                ellipse[2], 0, 360, BLUE, 2)

    # 长轴 (头长)
    a_rad = math.radians(angle_deg)
    major_half = int(ellipse[1][0] / 2)
    dx = int(major_half * math.cos(a_rad))
    dy = int(major_half * math.sin(a_rad))
    cv2.line(annotated, (cx - dx, cy - dy), (cx + dx, cy + dy), RED, 2)

    # 短轴 (头宽)
    minor_half = int(ellipse[1][1] / 2)
    dx2 = int(minor_half * math.cos(a_rad + math.pi / 2))
    dy2 = int(minor_half * math.sin(a_rad + math.pi / 2))
    cv2.line(annotated, (cx - dx2, cy - dy2), (cx + dx2, cy + dy2), YELLOW, 2)

    # 硬币标记 (仅圆圈)
    if coin_info is not None:
        (ccx, ccy), cr = coin_info
        cv2.circle(annotated, (ccx, ccy), cr, ORANGE, 2)

    return annotated


# ============================================================
# 10. 主分析流程
# ============================================================

def analyze_head_shape(
    image: np.ndarray,
    reference_mm: float = COIN_DIAMETER_MM,
    auto_detect_reference: bool = True,
    use_reference: bool = True,
    age_months: Optional[int] = None,
    guide_frame: bool = False,
) -> AnalysisResult:
    """
    婴儿头型分析主入口。

    参数:
      image: BGR 格式头顶俯拍照片
      reference_mm: 参照物实际尺寸(mm), 默认硬币 25mm
      auto_detect_reference: 是否自动检测硬币
      use_reference: 是否使用参照物校准。False 时用婴儿平均头宽估算
      age_months: 宝宝月龄
      guide_frame: 是否用了引导框拍照。True 时用画面比例估算距离

    返回:
      AnalysisResult (success, measurements, annotated_image, error_message)
    """
    result = AnalysisResult(success=False)
    steps = result.processing_steps
    h, w = image.shape[:2]

    if h < 100 or w < 100:
        result.error_message = "图片尺寸过小，请上传 >= 100x100 的照片"
        return result

    # Step 1: SAM 零样本头部检测 (替代肤色检测 — 不受肤色/光照/背景影响)
    from sam_detector import detect_head as detect_head_sam, create_guide_mask
    guide_mask = create_guide_mask(h, w, 'top') if guide_frame else None
    sam_result = detect_head_sam(image, guide_mask=guide_mask)
    if sam_result is None:
        # SAM 失败 → 回退到肤色检测
        enhanced = preprocess_image(image)
        skin_mask = detect_skin_adaptive(enhanced)
        skin_ratio = np.sum(skin_mask > 0) / (h * w)
        if skin_ratio < 0.02:
            result.error_message = "未检测到头部。请确保: 1)头顶正俯视角 2)光线充足 3)背景简洁"
            return result
        skin_mask = clean_mask(skin_mask)
        skin_mask = refine_with_grabcut(enhanced, skin_mask)
        skin_mask = clean_mask(skin_mask)
        head_out = extract_head_contour(skin_mask, image.shape)
        if head_out is None:
            result.error_message = "无法识别头部轮廓"
            return result
        head_mask, head_contour = head_out
        steps.append(f"肤色检测+轮廓提取 ({len(head_contour)} 点)")
    else:
        head_mask, head_contour = sam_result
        skin_mask = head_mask  # 用于硬币检测排除
        steps.append(f"SAM 头部检测完成 ({len(head_contour)} 点)")

    # Step 3.5: 头发干扰检测与补偿
    hair_score = detect_hair_interference(head_contour)
    if hair_score > 0.05:
        refined = refine_contour_under_hair(image, head_mask, head_contour, hair_score)
        if refined is not None and len(refined) > 20:
            head_contour = refined
            # 更新 head_mask
            head_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(head_mask, [head_contour], -1, 255, -1)
            steps.append(f"头发补偿 (干扰度:{hair_score:.0%}, 轮廓内收)")
    else:
        steps.append(f"头发干扰度: {hair_score:.0%} (无显著影响)")

    # Step 4: PCA 精确测量 (主) + 椭圆拟合 (可视化)
    ellipse = fit_head_ellipse(head_contour)  # 用于标注图
    pca_data = measure_head_pca(head_contour)  # 用于实际测量

    if pca_data is None:
        result.error_message = "无法分析头部轮廓，形状不理想"
        return result

    length_px = pca_data["length_px"]
    width_px = pca_data["width_px"]
    angle_deg = pca_data["angle_deg"]
    cx = pca_data["center"][0]
    cy = pca_data["center"][1]
    cvai_px_val = pca_data["cvai_px"]
    cva_px_val = pca_data["cva_px"]
    steps.append(f"PCA 测量完成 (头长:{length_px:.0f}px 头宽:{width_px:.0f}px)")

    # 如果没有椭圆(罕见), 用 PCA 中心+轴长构造一个用于标注
    if ellipse is None:
        ellipse = ((cx, cy), (length_px, width_px), angle_deg)

    # Step 5: 尺度校准
    coin_info = None
    scale_mm_per_px = None

    if use_reference and auto_detect_reference:
        coin_info = detect_coin(image, skin_mask)
        if coin_info is None:
            coin_info = detect_coin_by_circularity(image)

    if coin_info is not None:
        _, coin_r = coin_info
        scale_mm_per_px = reference_mm / (coin_r * 2)
        steps.append(f"检测到参照物 (半径:{coin_r}px, 比例:{scale_mm_per_px:.4f}mm/px)")
    else:
        # 无参照物时估算比例
        if guide_frame:
            # 引导框: 头填满 56% 画面 → 距离固定 → 从照片分辨率反推
            # 420rpx 引导框 / 750rpx 屏宽 = 56%
            head_px_est = w * 0.56  # 头在照片中约占 56% 宽度
            est_width = estimate_head_width_by_age(age_months)
            scale_mm_per_px = est_width / head_px_est
            steps.append(f"引导框模式 (头占{head_px_est:.0f}px, 估算头宽{est_width}mm)")
        else:
            # 年龄自适应头宽估算 (用 PCA 测的头宽 px)
            est_width = estimate_head_width_by_age(age_months)
            scale_mm_per_px = est_width / width_px
            if age_months:
                steps.append(f"无参照物，使用{age_months}月龄估算头宽{est_width}mm")
            else:
                steps.append(f"无参照物，使用默认头宽{est_width}mm (建议填写月龄提高精度)")

    # Step 6: CI 计算 (基于 PCA 直接测量)
    hl_mm = length_px * scale_mm_per_px
    hw_mm = width_px * scale_mm_per_px
    ci = hw_mm / hl_mm * 100

    # 头围 (Ramanujan)
    a, b = hl_mm / 2, hw_mm / 2
    hc_mm = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    steps.append(f"CI={ci:.1f} (头长:{hl_mm:.1f}mm 头宽:{hw_mm:.1f}mm)")

    # Step 7: CVAI (基于 PCA 对角线)
    cvai = cvai_px_val
    cva_mm = cva_px_val * scale_mm_per_px
    steps.append(f"CVAI={cvai:.1f}%")

    # Step 8: 分级
    severity = classify_severity(ci, cvai)

    # 置信度估算
    confidence = 0.4
    skin_ratio = np.sum(head_mask > 0) / (h * w)
    if 0.05 < skin_ratio < 0.5:
        confidence += 0.15
    if coin_info is not None:
        confidence += 0.35  # 有参照物 → 高置信度
    else:
        confidence += 0.10  # 无参照物 → 基础置信度
    confidence = min(confidence, 1.0)

    measurements = HeadMeasurements(
        head_length_mm=round(hl_mm, 1),
        head_width_mm=round(hw_mm, 1),
        head_circumference_mm=round(hc_mm, 1),
        ci=round(ci, 1),
        cvai=round(cvai, 1),
        cva_mm=round(cva_mm, 1),
        severity=severity,
        confidence=round(confidence, 2),
    )

    # Step 9: 标注图
    annotated = _draw_annotations(image, head_contour, ellipse, coin_info,
                                   measurements, (cx, cy), angle_deg)

    result.success = True
    result.measurements = measurements
    result.annotated_image = annotated
    return result


# ============================================================
# 命令行测试入口
# ============================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python head_analyzer.py <照片路径>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"无法读取: {sys.argv[1]}")
        sys.exit(1)

    print(f"分析: {sys.argv[1]} ({img.shape[1]}×{img.shape[0]})")
    print("-" * 40)

    result = analyze_head_shape(img)
    for s in result.processing_steps:
        print(s)
    print("-" * 40)

    if result.success:
        m = result.measurements
        print(f"\n[测量结果]")
        print(f"   头长 (前后径):  {m.head_length_mm} mm")
        print(f"   头宽 (左右径):  {m.head_width_mm} mm")
        print(f"   头围 (估算):    {m.head_circumference_mm} mm")
        print(f"   头颅指数 (CI):  {m.ci}  (正常 75-85)")
        print(f"   不对称指数:     {m.cvai}%")
        print(f"   不对称值 (CVA): {m.cva_mm} mm")
        print(f"   严重程度:        {m.severity.value}")
        print(f"   置信度:          {m.confidence:.0%}")

        advice = generate_advice(m)
        print(f"\n[建议] {advice['summary']}")
        print("[行动]:")
        for a in advice["actions"]:
            print(f"   - {a}")
        print(f"[监测] {advice['monitoring']}")

        out = sys.argv[1].rsplit('.', 1)[0] + '_analyzed.jpg'
        cv2.imwrite(out, result.annotated_image)
        print(f"\n[标注图] {out}")
    else:
        print(f"[失败] {result.error_message}")
