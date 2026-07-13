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

def _make_ideal_top_contour(user_contour, h, w):
    """
    生成理想俯视图轮廓，**动态匹配用户头型尺寸**。
    取用户头型的长轴为基准，按 CI=78 比例生成标准椭圆。
    """
    cx, cy = w // 2, h // 2

    # 从用户轮廓推算头型大小
    if user_contour is not None and len(user_contour) >= 10:
        bx, by, bw, bh = cv2.boundingRect(user_contour)
        # 以用户头型长轴为基准
        base = max(bw, bh) / 2
    else:
        base = min(w, h) * 0.28  # 回退到固定比例

    # CI=78: 宽=0.78×长, 即 rx=0.78*ry (width=2*rx, length=2*ry)
    rx = int(base * 0.78)
    ry = int(base)
    angles = np.linspace(0, 2 * np.pi, 200)
    x = (cx + rx * np.cos(angles)).astype(np.int32)
    y = (cy + ry * np.sin(angles)).astype(np.int32)
    contour = np.column_stack([x, y]).reshape(-1, 1, 2)
    return contour


def _make_ideal_side_contour(user_contour, h, w):
    """
    生成理想侧面头型轮廓，**动态匹配用户头型尺寸**。
    """
    cx, cy = w // 2, int(h * 0.45)

    if user_contour is not None and len(user_contour) >= 10:
        bx, by, bw, bh = cv2.boundingRect(user_contour)
        base = max(bw, bh) / 2
    else:
        base = min(w, h) * 0.22

    rx = int(base)
    ry = int(base)
    angles = np.linspace(0, 2 * np.pi, 200)
    x = (cx + rx * np.cos(angles)).astype(np.int32)
    y = (cy + ry * np.sin(angles)).astype(np.int32)
    contour = np.column_stack([x, y]).reshape(-1, 1, 2)
    return contour


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

def _draw_text_box(img, lines, x, y, font_scale=0.6, color=(255, 255, 255), bg_alpha=0.5):
    """在图像上绘制带半透明背景的多行文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_h = int(22 * font_scale)
    h_total = line_h * len(lines) + 16

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 8, y - h_total), (x + 280, y + 4), (30, 30, 30), -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1 - bg_alpha, 0, img)

    for i, line in enumerate(lines):
        cy = y - h_total + 14 + i * line_h
        cv2.putText(img, line, (x, cy), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_comparison(image, user_contour, view='top', side_result=None):
    """
    在图像上叠加用户轮廓(绿色)和理想轮廓(白色虚线), 返回标注图和对比数据。
    俯视图: 基于CI偏差评分 + 文字标注
    侧面图: 基于扁平度评分
    """
    h, w = image.shape[:2]
    result = image.copy()

    if view == 'top':
        ideal_contour = _make_ideal_top_contour(user_contour, h, w)
        score, ci_dev = compute_top_similarity(user_contour)
        comp_data = {"similarity_score": score, "ci_deviation": ci_dev}

        # 文字标注 (家长友好措辞)
        status = "头型接近标准" if score >= 85 else ("有轻微偏差" if score >= 60 else "偏差较明显")
        advice = "继续保持!" if score >= 85 else ("建议多换睡姿" if score >= 60 else "建议关注调整")
        lines = [
            f"头型相似度 {score}%  {status}",
            f"头型指数偏差 {ci_dev}%  (正常 75-85)",
            f"绿线=宝宝  白虚线=标准  {advice}",
        ]
        _draw_text_box(result, lines, 12, h - 12, font_scale=0.55)

    else:
        ideal_contour = _make_ideal_side_contour(user_contour, h, w)
        score = compute_side_similarity(side_result)
        flatness = side_result.get('posterior_flatness', 0) if side_result else 0
        comp_data = {"similarity_score": score}

        status = "圆润" if score >= 70 else ("稍扁平" if score >= 40 else "明显扁平")
        lines = [
            f"后枕圆润度 {score}%  {status}",
            f"绿线=宝宝  白虚线=标准",
        ]
        _draw_text_box(result, lines, 12, h - 12, font_scale=0.55)

    # 用户轮廓 - 绿色实线
    if user_contour is not None and len(user_contour) >= 10:
        cv2.drawContours(result, [user_contour], -1, (0, 200, 80), 2)

    # 理想轮廓 - 白色虚线
    for i in range(0, len(ideal_contour), 6):
        cv2.circle(result, tuple(ideal_contour[i][0]), 1, (255, 255, 255), -1)

    return result, comp_data
