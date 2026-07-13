"""
标准头型对比模块
- 生成理想头型轮廓
- Hu矩形状相似度计算
- 叠加对比图生成
"""

import cv2
import numpy as np


# ============================================================
# PART 1: 生成理想头型轮廓
# ============================================================

def _make_ideal_top_contour(h, w):
    """生成理想俯视图轮廓 (椭圆, CI=78 标准头型)"""
    cx, cy = w // 2, h // 2
    base = min(w, h) * 0.28
    rx, ry = int(base * 0.78), int(base)  # CI=78 → width=0.78*length
    angles = np.linspace(0, 2 * np.pi, 200)
    x = (cx + rx * np.cos(angles)).astype(np.int32)
    y = (cy + ry * np.sin(angles)).astype(np.int32)
    contour = np.column_stack([x, y]).reshape(-1, 1, 2)
    return contour, (cx, cy, rx, ry)


def _make_ideal_side_contour(h, w):
    """生成理想侧面头型轮廓"""
    cx, cy = w // 2, int(h * 0.45)
    rx = int(w * 0.22)
    ry = int(h * 0.22)
    angles = np.linspace(0, 2 * np.pi, 200)
    x = (cx + rx * np.cos(angles)).astype(np.int32)
    y = (cy + ry * np.sin(angles)).astype(np.int32)
    contour = np.column_stack([x, y]).reshape(-1, 1, 2)
    return contour, (cx, cy, rx, ry)


# ============================================================
# PART 2: 形状对比 (基于实际临床指标)
# ============================================================

def compute_top_similarity(user_contour):
    """
    俯视图相似度: 基于 CI 偏离度
    理想 CI=78, 每偏差 1% 扣 2 分
    CVAI < 3.5% 满分, 每超 1% 扣 10 分
    """
    if user_contour is None or len(user_contour) < 20:
        return 0, 999

    # 用椭圆拟合估算 CI
    pts = user_contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0, 999

    try:
        ellipse = cv2.fitEllipse(pts)
        (cx, cy), (major, minor), angle = ellipse
        ci_est = min(major, minor) / max(major, minor) * 100
        ci_dev = abs(ci_est - 78)
        ci_score = max(0, 100 - ci_dev * 2)
    except Exception:
        ci_score, ci_dev = 50, 99

    return round(ci_score), round(ci_dev, 1)


def compute_side_similarity(result_dict):
    """
    侧面相似度: 基于扁平度
    flatness=0(圆润) → 100分, flatness=0.5(明显扁平) → 0分
    """
    flatness = result_dict.get('posterior_flatness', 0.5) if result_dict else 0.5
    score = max(0, int(100 - flatness * 200))
    return score


# ============================================================
# PART 3: 叠加对比图
# ============================================================

def draw_comparison(image, user_contour, view='top', side_result=None):
    """
    在图像上叠加用户轮廓(绿色)和理想轮廓(白色虚线), 返回标注图和对比数据。
    俯视图: 基于CI偏差评分
    侧面图: 基于扁平度评分
    """
    h, w = image.shape[:2]
    result = image.copy()

    if view == 'top':
        ideal_contour, _ = _make_ideal_top_contour(h, w)
        score, ci_dev = compute_top_similarity(user_contour)
        comp_data = {"similarity_score": score, "ci_deviation": ci_dev}
    else:
        ideal_contour, _ = _make_ideal_side_contour(h, w)
        score = compute_side_similarity(side_result)
        comp_data = {"similarity_score": score}

    # 用户轮廓 - 绿色实线
    if user_contour is not None and len(user_contour) >= 10:
        cv2.drawContours(result, [user_contour], -1, (0, 200, 80), 2)

    # 理想轮廓 - 白色虚线
    for i in range(0, len(ideal_contour), 6):
        cv2.circle(result, tuple(ideal_contour[i][0]), 1, (255, 255, 255), -1)

    return result, comp_data


# ============================================================
# PART 4: 图例说明文字
# ============================================================

LEGEND_TEXT = {
    'top': "绿线=实测轮廓 | 白虚线=标准椭圆(CI=78)",
    'side': "绿线=实测轮廓 | 白虚线=标准头型",
}
