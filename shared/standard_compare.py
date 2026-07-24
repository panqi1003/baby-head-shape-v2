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

import threading
_STD_ELLIPSE = {}  # contour/ellipse cache
_STD_CACHE_LOCK = threading.Lock()

def _load_std_contour(view, side='left'):
    """Load standard contour. Thread-safe with DCL caching."""
    import os
    cache_key = f"{view}_{side}"
    if cache_key not in _STD_ELLIPSE:
        with _STD_CACHE_LOCK:
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
                    # side: 加载合成标准轮廓 (std_cache优先)
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

# ideal CI values by age (WHO/流行病学文献综合)
_IDEAL_CI_BY_AGE = {
    0: 80, 1: 81, 2: 82, 3: 82, 4: 83, 5: 83, 6: 82,
    7: 82, 8: 81, 9: 81, 10: 80, 11: 80, 12: 79,
    15: 79, 18: 78, 21: 78, 24: 78,
}


def _ideal_ci_for_age(age_months):
    """按月龄查理想CI值, 未提供月龄默认78"""
    if age_months is None:
        return 78
    # 找最接近的月龄
    ages = sorted(_IDEAL_CI_BY_AGE.keys())
    for a in ages:
        if age_months <= a:
            return _IDEAL_CI_BY_AGE[a]
    return _IDEAL_CI_BY_AGE[ages[-1]]


def compute_top_similarity(user_contour, age_months=None):
    """
    俯视图相似度: 基于 CI 偏离度
    理想 CI 按月龄查表(默认78), 每偏差 1% 扣 2 分
    """
    if user_contour is None or len(user_contour) < 20:
        return 0, 999

    pts = user_contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 0, 999

    try:
        ellipse = cv2.fitEllipse(pts)
        (cx, cy), (major, minor), angle = ellipse
        ci_est = min(major, minor) / max(major, minor) * 100
        ideal = _ideal_ci_for_age(age_months)
        ci_dev = abs(ci_est - ideal)
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


def draw_comparison(image, user_contour, view='top', side_result=None, side='left', age_months=None):
    """
    俯视图: 标准椭圆+动态对齐
    侧面图: PCA理想圆弧(不依赖标准图片)
    """
    h, w = image.shape[:2]
    result = image.copy()
    ideal_contour = None

    # 用户轮廓绿线先画: 白线(半透明)后叠加, 重合处两条都可见
    if user_contour is not None and len(user_contour) >= 10:
        cv2.drawContours(result, [user_contour], -1, (0, 230, 50), 3)

    if view == 'top':
        # === 俯视图: 标准椭圆 ===
        std_contour = _load_std_contour(view)
        ideal_contour = _align_and_scale_contour(std_contour, user_contour, h, w)
        if ideal_contour is None:
            return result, {"similarity_score": 0}

        score, ci_dev = compute_top_similarity(user_contour, age_months)
        comp_data = {"similarity_score": score, "ci_deviation": ci_dev}

        # 白线轮廓
        overlay = result.copy()
        cv2.drawContours(overlay, [ideal_contour], -1, (40, 40, 40), 4)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
        overlay2 = result.copy()
        cv2.drawContours(overlay2, [ideal_contour], -1, (255, 255, 255), 3)
        cv2.addWeighted(overlay2, 0.75, result, 0.25, 0, result)

    else:
        # === 侧面: Snake v2 PCA自适应弧线 ===
        score = compute_side_similarity(side_result)
        comp_data = {"similarity_score": score}

        if user_contour is not None and len(user_contour) >= 10 and side_result is not None:
            pts = user_contour.reshape(-1, 2).astype(np.float32)
            pca_center = side_result.get('_pca_center')
            pca_v1 = side_result.get('_pca_v1')
            expected_radius = side_result.get('expected_radius', 0)

            if pca_center and pca_v1 and expected_radius > 10:
                cx, cy = pca_center
                v1 = np.array(pca_v1)

                # v5: 后脑方向直接用 _pca_v1 (现在是 back_direction)
                ba = np.degrees(np.arctan2(v1[1], v1[0]))

                # 起始半径 = expected_radius × 1.02 (起点在绿线外侧)
                r = expected_radius * 1.02

                # 不对称角度: L50 R80 (后脑勺下半+上半, 不爬到头顶)
                a1, a2 = ba - 50, ba + 80
                t = np.linspace(np.radians(a1), np.radians(a2), 60)
                snake = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])

                # Snake迭代: 40轮, 只向外, 步长0.3, 最多外移15%, 3次平滑
                cap = r * 0.15
                for _ in range(40):
                    ns = snake.copy()
                    for i in range(60):
                        dc = np.sqrt((snake[i, 0] - cx) ** 2 + (snake[i, 1] - cy) ** 2)
                        di = np.argmin(np.sqrt(
                            (pts[:, 0] - snake[i, 0]) ** 2 + (pts[:, 1] - snake[i, 1]) ** 2))
                        dt = np.sqrt((pts[di, 0] - cx) ** 2 + (pts[di, 1] - cy) ** 2)
                        if dt > dc:
                            new_pos = snake[i] + 0.3 * (pts[di] - snake[i])
                            nd = np.sqrt((new_pos[0] - cx) ** 2 + (new_pos[1] - cy) ** 2)
                            if nd > r + cap:
                                d = new_pos - np.array([cx, cy])
                                d /= np.linalg.norm(d)
                                new_pos = np.array([cx, cy]) + d * (r + cap)
                            ns[i] = new_pos
                    for _ in range(3):
                        s = ns.copy()
                        for i in range(1, 59):
                            s[i] = 0.5 * ns[i] + 0.25 * (ns[i - 1] + ns[i + 1])
                        ns = s
                    snake = ns

                ideal = snake.astype(np.int32).reshape(-1, 1, 2)

                # 间隙指标 (给AI分析用)
                dc_final = np.sqrt((snake[:, 0] - cx) ** 2 + (snake[:, 1] - cy) ** 2)
                di_arr = np.array([np.argmin(np.sqrt(
                    (pts[:, 0] - snake[i, 0]) ** 2 + (pts[:, 1] - snake[i, 1]) ** 2))
                    for i in range(60)])
                dt_final = np.sqrt((pts[di_arr, 0] - cx) ** 2 + (pts[di_arr, 1] - cy) ** 2)
                gaps = dt_final - dc_final  # 正值=绿线比白弧更外凸, 负值=绿线内收
                gap_max = float(np.max(gaps))
                gap_avg = float(np.mean(gaps))
                # 分三段: 顶(0-19) 中(20-39) 底(40-59)
                gap_top = float(np.mean(gaps[0:20]))
                gap_mid = float(np.mean(gaps[20:40]))
                gap_bot = float(np.mean(gaps[40:60]))
                comp_data["gap_max"] = round(gap_max, 1)
                comp_data["gap_avg"] = round(gap_avg, 1)
                comp_data["gap_top"] = round(gap_top, 1)
                comp_data["gap_mid"] = round(gap_mid, 1)
                comp_data["gap_bot"] = round(gap_bot, 1)

                # 相似度改为基于白弧间隙 (与用户看到的对比图一致)
                # inward = 绿线比白弧内收的平均量, 归一化到半径
                inward = float(np.mean(np.maximum(0, -gaps)))
                ratio = inward / (r + 1e-6)
                score = max(0, min(100, int(round(100 - ratio * 2500))))
                comp_data["similarity_score"] = score

                overlay = result.copy()
                cv2.polylines(overlay, [ideal], False, (60, 60, 60), 5)
                cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
                overlay2 = result.copy()
                cv2.polylines(overlay2, [ideal], False, (255, 255, 255), 3)
                cv2.addWeighted(overlay2, 0.75, result, 0.25, 0, result)



    return result, comp_data
