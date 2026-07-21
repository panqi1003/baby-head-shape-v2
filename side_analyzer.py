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

      1. SAM 分割头部
      2. PCA 找前后方向 (第一主成分)
      3. 将后枕部轮廓曲率与理想圆弧对比
      4. 扁平度 = 曲率偏差 / 期望半径
    """
    h, w = image.shape[:2]

    from sam_detector import detect_head

    # 侧面图不需要 guide_mask — SAM点提示本身足够
    sam_result = detect_head(image, view='side')
    if sam_result is None:
        return None

    head_mask, head_contour = sam_result

    # 侧面图不做头发补偿 (MiMo Pro评审):
    # 1. 侧面轮廓含鼻/下巴/耳/颈, 远离圆形, hair_score恒饱和, 门控失效
    # 2. 后枕正是头发最厚区域, 刨除后轮廓沿发际线走, 扁平度测量对象被偷换

    if len(head_contour) < 30:
        return None

    # v5 极值点法找后脑勺
    pts = head_contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    y_min, y_max = ys.min(), ys.max()

    # 中间带: y 20%-75%, 排除头顶头发和脖子
    y_lo = y_min + (y_max - y_min) * 0.20
    y_hi = y_min + (y_max - y_min) * 0.75
    mid_mask = (ys >= y_lo) & (ys <= y_hi)
    if np.sum(mid_mask) < 30:
        mid_mask = np.ones(n, dtype=bool)

    mid_pts = pts[mid_mask]
    mid_idx = np.where(mid_mask)[0]
    leftmost_local = np.argmin(mid_pts[:, 0])
    leftmost_global = mid_idx[leftmost_local]

    # Local smoothing - dynamic window based on total contour size
    w_smooth = max(5, n // 8)
    lo = max(0, leftmost_global - w_smooth)
    hi = min(n, leftmost_global + w_smooth + 1)
    back_pt = np.mean(pts[lo:hi], axis=0)

    # 质心 + 后脑方向
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    back_v = back_pt - np.array([cx, cy])
    back_dir_angle = math.atan2(back_v[1], back_v[0])

    # ±70°取后脑勺弧段 (用原始轮廓角度)
    all_angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ad = np.abs(all_angles - back_dir_angle)
    ad = np.minimum(ad, 2 * np.pi - ad)
    back_mask = ad < np.radians(70)
    back_idx_all = np.where(back_mask)[0]

    if len(back_idx_all) < 15:
        return None

    # 取最大连续段
    gaps_arr = np.diff(back_idx_all)
    big = np.where(gaps_arr > 10)[0]
    if len(big) > 0:
        # 利用 big 数组直接计算段边界，避免遗漏首元素
        segs = []
        cs = 0
        for gap_idx in big:
            segs.append((cs, gap_idx))
            cs = gap_idx + 1
        segs.append((cs, len(back_idx_all) - 1))
        segs.sort(key=lambda x: -(x[1] - x[0]))
        s, e = segs[0]
        back_idx_all = back_idx_all[s:e + 1]

    back_pts = pts[back_idx_all]
    back_centered = back_pts - np.array([cx, cy])
    back_dists = np.linalg.norm(back_centered, axis=1)

    # 期望半径 = 后脑勺弧段到质心中的中位距离
    expected_radius = np.median(back_dists)

    if expected_radius < 10:
        return None

    back_median = np.median(back_dists)

    # 扁平度 (公式不变)
    flatness_raw = max(0, 1.0 - back_median / (expected_radius + 1e-6))
    curvature_cv = np.std(back_dists) / (np.mean(back_dists) + 1e-6)
    flatness_score = flatness_raw * 0.7 + min(curvature_cv * 1.5, 0.3)
    flatness_score = round(min(1.0, flatness_score), 3)

    if flatness_score < 0.15:
        category = "正常圆润"
    elif flatness_score < 0.30:
        category = "轻度扁平"
    elif flatness_score < 0.50:
        category = "中度扁平"
    else:
        category = "明显扁平"

    head_length_px = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    head_height_px = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))

    contour_list = head_contour.reshape(-1, 2).tolist() if head_contour is not None else None

    # 后枕部角度范围
    back_angles_arr = np.arctan2(back_centered[:, 1], back_centered[:, 0])
    back_angle_start = float(np.degrees(back_angles_arr.min()))
    back_angle_end = float(np.degrees(back_angles_arr.max()))

    # v1 改为后脑方向 (保持 _pca_v1 字段名兼容)
    result_dict = {
        "posterior_flatness": flatness_score,
        "head_length_px": round(head_length_px, 1),
        "head_height_px": round(head_height_px, 1),
        "ear_position_ratio": 0.5,
        "flatness_category": category,
        "curvature_cv": round(curvature_cv, 3),
        "expected_radius": round(expected_radius, 1),
        "scale_method": "默认估算",
        "_head_contour": contour_list,
        "_pca_center": [float(cx), float(cy)],
        "_pca_v1": [float(math.cos(back_dir_angle)), float(math.sin(back_dir_angle))],
        "_back_angles": [back_angle_start, back_angle_end],
    }

    return result_dict
