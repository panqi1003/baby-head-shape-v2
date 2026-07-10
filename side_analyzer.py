"""
侧面头型分析 — PCA 曲率法测量后枕部扁平度
"""

import cv2
import numpy as np
import math
from typing import Optional, Dict


def analyze_side_profile(image: np.ndarray) -> Optional[Dict]:
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

    # 将点分为前(面部)半和后(枕)半
    proj = np.dot(centered, v1)
    back_mask = proj < 0   # 后枕部
    front_mask = proj >= 0

    back_pts = centered[back_mask]
    if len(back_pts) < 15:
        return None

    # 计算后枕部点到中心的距离
    back_dists = np.linalg.norm(back_pts, axis=1)

    # 用前部点估算头部"理想半径" (前部通常比较圆)
    front_pts = centered[front_mask]
    if len(front_pts) > 15:
        front_dists = np.linalg.norm(front_pts, axis=1)
        expected_radius = np.median(front_dists)
    else:
        expected_radius = np.median(back_dists)

    if expected_radius < 10:
        return None

    # 后枕扁平度: 后枕部实际距离 vs 理想半径
    # 取后枕部最远 P5 距离 (排除异常点)
    back_p95 = np.percentile(back_dists, 95)
    back_median = np.median(back_dists)

    # 扁平评分: 0=与理想圆弧一致(圆润), 1=极度扁平
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
    }
