"""
侧面头型分析 — PCA 曲率法测量后枕部扁平度
"""

import cv2
import numpy as np
import math
from typing import Optional, Dict


def analyze_side_profile(image: np.ndarray, guide_frame: bool = False) -> Optional[Dict]:
    """
    分析侧面婴儿头部照片。

    改进算法:
      1. SAM 分割头部
      2. PCA 找前后方向 (第一主成分)
      3. 将后枕部轮廓曲率与理想圆弧对比
      4. 扁平度 = 曲率偏差 / 期望半径

    返回 dict 或 None
    """
    h, w = image.shape[:2]

    from sam_detector import detect_head
    from head_analyzer import detect_hair_interference, refine_contour_under_hair

    sam_result = detect_head(image)
    if sam_result is None:
        return None

    head_mask, head_contour = sam_result

    # 头发干扰补偿 — 与俯视图相同的处理
    hair_score = detect_hair_interference(head_contour)
    if hair_score > 0.05:
        refined = refine_contour_under_hair(image, head_mask, head_contour, hair_score)
        if refined is not None and len(refined) > 20:
            head_contour = refined

    if len(head_contour) < 30:
        return None

    # PCA 找前后轴
    pts = head_contour.reshape(-1, 2).astype(np.float64)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    v1 = eigenvectors[:, idx[0]]   # 前后方向 (头长)
    v2 = eigenvectors[:, idx[1]]   # 上下方向

    cx, cy = mean[0], mean[1]

    # 沿 v1 分两半, 用曲率差异自动区分面部和枕部
    #   面部: 距离方差大 (鼻子/额头/下巴轮廓不规则)
    #   枕部: 距离方差小 (后脑勺相对光滑)
    proj = np.dot(centered, v1)
    half_a_mask = proj < 0
    half_b_mask = proj >= 0

    pts_a = centered[half_a_mask]
    pts_b = centered[half_b_mask]
    if len(pts_a) < 15 or len(pts_b) < 15:
        return None

    dists_a = np.linalg.norm(pts_a, axis=1)
    dists_b = np.linalg.norm(pts_b, axis=1)

    # 距离方差大的 = 面部 (不规则), 方差小的 = 枕部 (平滑)
    cv_a = np.std(dists_a) / (np.mean(dists_a) + 1e-6)
    cv_b = np.std(dists_b) / (np.mean(dists_b) + 1e-6)

    if cv_a > cv_b:
        front_pts, front_dists = pts_a, dists_a
        back_pts, back_dists = pts_b, dists_b
    else:
        front_pts, front_dists = pts_b, dists_b
        back_pts, back_dists = pts_a, dists_a

    expected_radius = np.median(front_dists)
    if expected_radius < 10:
        return None

    # 后枕扁平度: 后枕部实际距离 vs 面部期望半径
    back_median = np.median(back_dists)
    flatness_raw = max(0, 1.0 - back_median / (expected_radius + 1e-6))

    # 曲率变异: 后枕部距离的标准差/均值 → 越大越不规则
    curvature_cv = np.std(back_dists) / (np.mean(back_dists) + 1e-6)

    # 综合扁平度 (0-1)
    flatness_score = flatness_raw * 0.7 + min(curvature_cv * 1.5, 0.3)
    flatness_score = round(min(1.0, flatness_score), 3)

    # 分类
    if flatness_score < 0.15:
        category = "正常圆润"
    elif flatness_score < 0.30:
        category = "轻度扁平"
    elif flatness_score < 0.50:
        category = "中度扁平"
    else:
        category = "明显扁平"

    head_length_px = float(proj.max() - proj.min())
    head_height_px = float(np.dot(centered, v2).max() - np.dot(centered, v2).min())

    return {
        "posterior_flatness": flatness_score,
        "head_length_px": round(head_length_px, 1),
        "head_height_px": round(head_height_px, 1),
        "ear_position_ratio": 0.5,
        "flatness_category": category,
        "curvature_cv": round(curvature_cv, 3),
        "expected_radius": round(expected_radius, 1),
        "scale_method": "引导框" if guide_frame else "默认估算",
    }
