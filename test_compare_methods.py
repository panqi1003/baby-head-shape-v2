"""
PCA vs 极值点法 对比 v2 — PCA侧用真正的线上代码
"""
import cv2, numpy as np, math, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 线上版本
from sam_detector import detect_head
from side_analyzer import analyze_side_profile
from standard_compare import draw_comparison

# 去重后的唯一真实侧面图 (3个宝宝 + 2张独立原图)
REAL_FILES = [
    'real_side.jpg',                     # 宝宝1: 4284x5712
    'test_neck_real_side_jpg.jpg',       # 宝宝2: 900x1200
    'test_54945_side.jpg',               # 宝宝3: 4284x5712
    '7682d493e0e21620c2a35d9e37f681dd.png',  # 独立原图4
    '943e524bb386bfbfe1e34fe403e5e826.png',  # 独立原图5
]

def v5_method(pts):
    """极值点法 找后脑勺"""
    n = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    y_min, y_max = ys.min(), ys.max()
    y_lo = y_min + (y_max - y_min) * 0.20
    y_hi = y_min + (y_max - y_min) * 0.75
    mid_mask = (ys >= y_lo) & (ys <= y_hi)
    if np.sum(mid_mask) < 30:
        mid_mask = np.ones(n, dtype=bool)

    mid_pts = pts[mid_mask]
    mid_idx = np.where(mid_mask)[0]
    leftmost_local = np.argmin(mid_pts[:, 0])
    leftmost_global = mid_idx[leftmost_local]

    w = 30
    lo = max(0, leftmost_global - w)
    hi = min(n, leftmost_global + w + 1)
    back_pt = np.mean(pts[lo:hi], axis=0)

    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    back_v = back_pt - np.array([cx, cy])
    ba = np.degrees(math.atan2(back_v[1], back_v[0]))

    all_angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    ad = np.abs(all_angles - ba)
    ad = np.minimum(ad, 360 - ad)
    back_mask = ad < 70
    back_idx = np.where(back_mask)[0]

    gaps = np.diff(back_idx)
    big = np.where(gaps > 10)[0]
    if len(big) > 0:
        segs, cs = [], 0
        for i in range(1, len(back_idx)):
            if back_idx[i] - back_idx[i-1] > 10:
                segs.append((cs, i-1)); cs = i
        segs.append((cs, len(back_idx)-1))
        segs.sort(key=lambda x: -(x[1]-x[0]))
        s, e = segs[0]
        back_idx = back_idx[s:e+1]

    back_pts = pts[back_idx]
    expected_radius = np.median(np.linalg.norm(back_pts - np.array([cx, cy]), axis=1))
    return ba, cx, cy, expected_radius, back_pts

def draw_snake_arc(img, cx, cy, ba, expected_radius, contour_pts):
    """Snake风格白弧"""
    R = expected_radius * 0.94
    n_pts = 60
    angles = np.linspace(np.radians(ba - 70), np.radians(ba + 70), n_pts)
    snake = np.column_stack([cx + R * np.cos(angles), cy + R * np.sin(angles)])

    for _ in range(40):
        ns = snake.copy()
        for i in range(n_pts):
            sx, sy = snake[i]
            sd = np.sqrt((sx - cx)**2 + (sy - cy)**2)
            di = np.argmin(np.sqrt((contour_pts[:, 0] - sx)**2 + (contour_pts[:, 1] - sy)**2))
            dt = np.sqrt((contour_pts[di, 0] - cx)**2 + (contour_pts[di, 1] - cy)**2)
            if dt > sd:
                ns[i] = snake[i] + 0.3 * (contour_pts[di] - snake[i])
        for _ in range(3):
            s = ns.copy()
            for i in range(1, n_pts-1):
                s[i] = 0.5*ns[i] + 0.25*(ns[i-1]+ns[i+1])
            ns = s
        snake = ns

    pts_i = snake.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], False, (80, 80, 80), 5)
    cv2.polylines(img, [pts_i], False, (255, 255, 255), 3)

def main():
    for fname in REAL_FILES:
        if not os.path.exists(fname):
            print(f"  SKIP {fname}")
            continue

        print(f"\n处理: {fname}")
        img = cv2.imread(fname)
        if img is None: continue
        h_orig, w_orig = img.shape[:2]
        SCALE = 1000 / max(h_orig, w_orig)
        nw, nh = int(w_orig*SCALE), int(h_orig*SCALE)

        # ===== 线上版本: 真正的 side_analyzer + standard_compare ====
        side_result = analyze_side_profile(img)
        if side_result is None:
            print(f"  线上: 侧面分析失败")
            continue

        contour_list = side_result.pop("_head_contour", None)
        contour_raw = np.array(contour_list, dtype=np.int32).reshape(-1, 1, 2)
        pts = contour_raw.reshape(-1, 2).astype(np.float64)

        # 用真正线上 draw_comparison 生成
        online_img, comp_data = draw_comparison(img, contour_raw, view='side', side_result=side_result, side='')
        online_img = cv2.resize(online_img, (nw, nh))
        cv2.putText(online_img, f"ONLINE: snake gap", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if comp_data:
            cv2.putText(online_img, f"sim={comp_data.get('similarity_score','?')}% gap_avg={comp_data.get('gap_avg','?'):.1f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # PCA 参数(供对比)
        pca_ba = np.degrees(math.atan2(side_result['_pca_v1'][1], side_result['_pca_v1'][0]))
        pca_r = side_result['expected_radius']
        print(f"  线上: ba={pca_ba:.0f} R={pca_r:.0f} flatness={side_result['posterior_flatness']} {side_result['flatness_category']}")

        # ===== v5 版本 =====
        v5_ba, v5_cx, v5_cy, v5_r, back_pts = v5_method(pts)
        v5_img = cv2.resize(img.copy(), (nw, nh))
        # 轮廓缩放到与图片一致
        contour_scaled = (contour_raw.astype(np.float64) * SCALE).astype(np.int32)
        cv2.drawContours(v5_img, [contour_scaled], -1, (0, 220, 60), 3)
        draw_snake_arc(v5_img, v5_cx*SCALE, v5_cy*SCALE, v5_ba, v5_r*SCALE, pts*SCALE)
        cv2.putText(v5_img, f"v5: ba={v5_ba:.0f} R={v5_r:.0f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(v5_img, "v5 (extreme-point)", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        print(f"  v5:   ba={v5_ba:.0f} R={v5_r:.0f}")

        # ===== 并排 =====
        cmp = np.hstack([online_img, v5_img])
        cv2.line(cmp, (nw, 0), (nw, nh), (100,100,100), 2)
        cv2.putText(cmp, "ONLINE", (nw//2-50, nh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        cv2.putText(cmp, "v5", (nw+nw//2-20, nh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        tag = os.path.splitext(fname)[0][:20]
        out = f'test_compare_{tag}.jpg'
        cv2.imwrite(out, cmp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  → {out}")

    print("\n全部完成")

if __name__ == '__main__':
    main()
