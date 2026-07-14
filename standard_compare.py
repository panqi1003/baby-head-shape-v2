"""
标准头型对比模块
- 生成理想头型轮廓
- Hu矩形状相似度计算
- 叠加对比图生成
"""

import cv2
import numpy as np


# ============================================================
# PART 1: 加载标准头型轮廓
# ============================================================

_STD_ELLIPSE = {}  # 俯视拟合椭圆缓存

def _load_std_contour(view, side='left'):
    """加载标准头型轮廓。俯视返回拟合椭圆参数, 侧面返回原始轮廓(左/右)。"""
    import os
    cache_key = f"{view}_{side}"
    if cache_key not in _STD_ELLIPSE:
        std_dir = os.path.join(os.path.dirname(__file__), '标准头型')
        if view == 'top':
            path = os.path.join(std_dir, 'std_top.npy')
            if os.path.exists(path):
                raw = np.load(path, allow_pickle=True)
                pts = raw.reshape(-1, 2).astype(np.float32)
                if len(pts) >= 5:
                    ellipse = cv2.fitEllipse(pts)
                    _STD_ELLIPSE[cache_key] = ellipse
                else:
                    _STD_ELLIPSE[cache_key] = None
        else:
            # 侧面: 加载合成标准轮廓 (std_cache优先)
            fname = f'std_side_{side}.npy'
            path = os.path.join(os.path.dirname(__file__), 'std_cache', fname)
            if not os.path.exists(path):
                path = os.path.join(std_dir, fname)
            _STD_ELLIPSE[cache_key] = np.load(path, allow_pickle=True) if os.path.exists(path) else None
    return _STD_ELLIPSE.get(cache_key)


def _align_and_scale_contour(std_data, user_contour, h, w):
    """
    MiMo Pro design: median radius ratio scaling.
    Top: 360-point polar sampling, median(r_user/r_std), robust to hair spikes.
    Side: bbox scaling.
    """
    if std_data is None or user_contour is None or len(user_contour) < 10:
        return None

    user_pts = user_contour.reshape(-1, 2).astype(np.float32)

    # user center: fit ellipse center
    if len(user_pts) >= 5:
        u_ellipse = cv2.fitEllipse(user_pts)
        (user_cx, user_cy), _, _ = u_ellipse
    else:
        bx, by, bw, bh = cv2.boundingRect(user_contour)
        user_cx, user_cy = bx + bw // 2, by + bh // 2

    # 极坐标采样角度 (top和side共用)
    phis = np.linspace(0, 2*np.pi, 360, endpoint=False)

    if isinstance(std_data, tuple) and len(std_data) == 3:
        # === Top view: median radius ratio (MiMo Pro algorithm) ===
        (_, _), (major, minor), angle = std_data
        a, b = major / 2, minor / 2  # semi-axes
        rad = np.radians(angle)

        # 1. Standard ellipse 360-degree radius
        dphi = phis - rad
        cos_d = np.cos(dphi)
        sin_d = np.sin(dphi)
        r_std = (a * b) / np.sqrt((b * cos_d)**2 + (a * sin_d)**2 + 1e-10)

        # 2. User contour -> polar -> interpolate to 360 points
        dx = user_pts[:, 0] - user_cx
        dy = user_pts[:, 1] - user_cy
        r_user_raw = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        order = np.argsort(theta)
        r_user = np.interp(phis, theta[order], r_user_raw[order], period=2*np.pi)

        # 3. Median scale ratio (robust to hair spikes)
        ratios = r_user / (r_std + 1e-10)
        scale = np.percentile(ratios, 60)  # 60分位: 比median稍大, 比75紧凑

        # 4. Generate scaled ellipse, 200 points
        new_a, new_b = a * scale, b * scale
        t = np.linspace(0, 2*np.pi, 200)
        ex = user_cx + new_a * np.cos(t) * np.cos(rad) - new_b * np.sin(t) * np.sin(rad)
        ey = user_cy + new_a * np.cos(t) * np.sin(rad) + new_b * np.sin(t) * np.cos(rad)
        return np.column_stack([ex, ey]).astype(np.int32).reshape(-1, 1, 2)
    else:
        # === Side view: bbox scaling (开放曲线, 不能用极坐标) ===
        std_pts = std_data.reshape(-1, 2).astype(np.float32)
        sw = std_pts[:, 0].max() - std_pts[:, 0].min()
        sh = std_pts[:, 1].max() - std_pts[:, 1].min()
        if sw < 5 or sh < 5:
            return None
        scx = (std_pts[:, 0].min() + std_pts[:, 0].max()) / 2
        scy = (std_pts[:, 1].min() + std_pts[:, 1].max()) / 2
        bx, by, bw, bh = cv2.boundingRect(user_contour)
        ucx, ucy = bx + bw // 2, by + bh // 2
        scale = max(bw / sw, bh / sh)
        aligned = std_pts.copy()
        aligned[:, 0] = (aligned[:, 0] - scx) * scale + ucx
        aligned[:, 1] = (aligned[:, 1] - scy) * scale + ucy
        return aligned.astype(np.int32).reshape(-1, 1, 2)
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


def draw_comparison(image, user_contour, view='top', side_result=None, side='left'):
    """
    俯视图: 标准椭圆+动态对齐
    侧面图: PCA理想圆弧(不依赖标准图片)
    """
    h, w = image.shape[:2]
    result = image.copy()
    ideal_contour = None

    if view == 'top':
        # === 俯视图: 标准椭圆 ===
        std_contour = _load_std_contour(view)
        ideal_contour = _align_and_scale_contour(std_contour, user_contour, h, w)
        if ideal_contour is None:
            return result, {"similarity_score": 0}

        score, ci_dev = compute_top_similarity(user_contour)
        comp_data = {"similarity_score": score, "ci_deviation": ci_dev}
        status = "头型接近标准" if score >= 85 else ("有轻微偏差" if score >= 60 else "偏差较明显")
        advice = "继续保持!" if score >= 85 else ("建议多换睡姿" if score >= 60 else "建议关注调整")
        _draw_text_box(result, [
            f"头型相似度 {score}%  {status}",
            f"头型指数偏差 {ci_dev}%  (正常 75-85)",
            f"绿线=宝宝  白线=标准  {advice}",
        ], 12, h - 12, font_scale=0.60)

        # 白线轮廓
        overlay = result.copy()
        cv2.drawContours(overlay, [ideal_contour], -1, (40, 40, 40), 4)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        overlay2 = result.copy()
        cv2.drawContours(overlay2, [ideal_contour], -1, (255, 255, 255), 2)
        cv2.addWeighted(overlay2, 0.75, result, 0.25, 0, result)

    else:
        # === 侧面: RANSAC圆拟合后脑点 → 后枕理想弧 ===
        score = compute_side_similarity(side_result)
        comp_data = {"similarity_score": score}

        if user_contour is not None and len(user_contour) >= 10:
            pts = user_contour.reshape(-1, 2).astype(np.float32)
            mid_x = (pts[:,0].min() + pts[:,0].max()) / 2

            # 后脑侧 = 点数少的一侧
            left_n = int(np.sum(pts[:,0] < mid_x))
            right_n = int(np.sum(pts[:,0] > mid_x))
            if left_n < right_n:
                back_pts = pts[pts[:,0] < mid_x]
            else:
                back_pts = pts[pts[:,0] > mid_x]

            if len(back_pts) >= 10:
                # RANSAC 圆拟合
                best_c, best_n = None, 0
                for _ in range(200):
                    idx = np.random.choice(len(back_pts), 3, replace=False)
                    p1, p2, p3 = back_pts[idx]
                    A = np.array([[p1[0],p1[1],1],[p2[0],p2[1],1],[p3[0],p3[1],1]], dtype=np.float64)
                    if abs(np.linalg.det(A)) < 1e-6: continue
                    D = np.array([-p1[0]**2-p1[1]**2, -p2[0]**2-p2[1]**2, -p3[0]**2-p3[1]**2], dtype=np.float64)
                    try:
                        sol = np.linalg.solve(A, D)
                        cx = float(-sol[0]/2); cy = float(-sol[1]/2)
                        r = float(np.sqrt(max(0, cx**2 + cy**2 - sol[2])))
                        if r < 50 or r > 5000: continue
                        dists = np.abs(np.sqrt((back_pts[:,0]-cx)**2 + (back_pts[:,1]-cy)**2) - r)
                        n = int(np.sum(dists < 80))
                        if n > best_n: best_n = n; best_c = (cx, cy, r * 1.05)
                    except: pass

                if best_c is not None:
                    cx, cy, r = best_c
                    # 后脑最突点角度 ± 45°
                    backmost_i = np.argmin(back_pts[:,0]) if left_n < right_n else np.argmax(back_pts[:,0])
                    bm_angle = np.degrees(np.arctan2(back_pts[backmost_i,1]-cy, back_pts[backmost_i,0]-cx))
                    a1, a2 = bm_angle - 55, bm_angle + 55
                    t = np.linspace(np.radians(a1), np.radians(a2), 50)
                    frac = np.linspace(0, 1, 50)  # 0=顶部, 1=底部
                    r_adj = r * (0.97 + 0.08 * (1 - frac))  # 顶部1.05x→底部0.97x
                    ax = cx + r_adj * np.cos(t); ay = cy + r_adj * np.sin(t)
                    ideal = np.column_stack([ax, ay]).astype(np.int32).reshape(-1, 1, 2)

                    overlay = result.copy()
                    cv2.polylines(overlay, [ideal], False, (60, 60, 60), 5)
                    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
                    overlay2 = result.copy()
                    cv2.polylines(overlay2, [ideal], False, (255, 255, 255), 2)
                    cv2.addWeighted(overlay2, 0.75, result, 0.25, 0, result)

        status = "圆润" if score >= 70 else ("稍扁平" if score >= 40 else "明显扁平")
        _draw_text_box(result, [
            f"后枕圆润度 {score}%  {status}",
            f"白弧=理想弧度  绿线=实测",
        ], 12, h - 12, font_scale=0.60)

    # 用户轮廓 - 绿色实线
    if user_contour is not None and len(user_contour) >= 10:
        cv2.drawContours(result, [user_contour], -1, (0, 230, 50), 3)

    # 图例
    legend_y = 36
    font_legend = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(result, (w - 130, legend_y), 7, (0, 230, 50), -1)
    cv2.putText(result, "宝宝实测", (w - 116, legend_y + 6), font_legend, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(result, (w - 130, legend_y + 28), 7, (255, 255, 255), -1)
    cv2.putText(result, "标准", (w - 116, legend_y + 34), font_legend, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return result, comp_data
