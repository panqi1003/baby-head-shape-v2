"""
极值点法找后脑勺 v5:
- 中间带(排除头顶头发+脖子)找最左点 → 后脑勺
- 计算正确的 back_angle / expected_radius
- 画 Snake 风格白弧 (简化版: 直接画圆弧, 不迭代)
- 对每张真实侧面图生成对比图
"""
import cv2, numpy as np, math, sys, os, glob
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sam_detector import detect_head

REAL_SIDE_FILES = [
    'real_side.jpg',
    'clean_test_side.jpg',
    'debug_verify_real_side..jpg',
    'test_neck_real_side_jpg.jpg',
    'test_neck2_real_side_jpg.jpg',
    'test_54945_side.jpg',
    'test_final_side.jpg',
    '真实面朝右侧视图.jpg',
    '5494546fe0a961aae9703e89ce4614e4.jpg',
    '7682d493e0e21620c2a35d9e37f681dd.png',
    '943e524bb386bfbfe1e34fe403e5e826.png',
]

def find_back_of_head(pts):
    """在中间带找后脑勺: 返回 (back_angle, back_center, expected_radius, back_pts)"""
    n = len(pts)
    xs, ys = pts[:, 0], pts[:, 1]
    y_min, y_max = ys.min(), ys.max()

    # 中间带: y 在 20%-75% 范围 → 排除头顶(头发)和脖子
    y_lo = y_min + (y_max - y_min) * 0.20
    y_hi = y_min + (y_max - y_min) * 0.75
    mid_mask = (ys >= y_lo) & (ys <= y_hi)
    mid_idx = np.where(mid_mask)[0]
    if len(mid_idx) < 30:
        # fallback: 全范围
        mid_mask = np.ones(n, dtype=bool)
        mid_idx = np.arange(n)

    # 在中间带找最左点
    mid_pts = pts[mid_mask]
    leftmost_local_idx = np.argmin(mid_pts[:, 0])
    leftmost_global_idx = mid_idx[leftmost_local_idx]

    # 轻度平滑找最左
    w = 30
    lo = max(0, leftmost_global_idx - w)
    hi = min(n, leftmost_global_idx + w + 1)
    local_avg = np.mean(pts[lo:hi], axis=0)
    back_pt = local_avg

    # 质心
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])

    # 后脑方向: 质心 → 后脑最左点
    back_v = back_pt - np.array([cx, cy])
    back_angle = np.degrees(math.atan2(back_v[1], back_v[0]))

    # ±70° 取后脑勺弧段点
    all_angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    ad = np.abs(all_angles - back_angle)
    ad = np.minimum(ad, 360 - ad)
    back_mask = ad < 70
    back_idx = np.where(back_mask)[0]

    # 取最大连续段 (排除角度环绕断裂)
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
    back_center = np.mean(back_pts, axis=0)
    expected_radius = np.median(np.linalg.norm(back_pts - np.array([cx, cy]), axis=1))

    return {
        'back_angle': back_angle,
        'back_center': back_center,
        'expected_radius': expected_radius,
        'back_pts': back_pts,
        'back_idx': back_idx,
        'cx': cx, 'cy': cy,
        'n_back': len(back_pts),
        'n_total': n,
    }

def draw_snake_arc(img, result, color=(255,255,255), thickness=3):
    """在图上画 Snake 风格白弧 (简化: 圆弧 40轮只向外)"""
    cx, cy = result['cx'], result['cy']
    ba = result['back_angle']
    R = result['expected_radius'] * 0.94

    # 60个点均匀分布在 ba ± 70°
    n_pts = 60
    angles = np.linspace(np.radians(ba - 70), np.radians(ba + 70), n_pts)
    snake = np.column_stack([cx + R * np.cos(angles), cy + R * np.sin(angles)])

    # 40轮迭代, 只向外移
    for _ in range(40):
        ns = snake.copy()
        for i in range(n_pts):
            sx, sy = snake[i]
            sd = np.sqrt((sx - cx)**2 + (sy - cy)**2)
            # 找最近轮廓点
            pts = result['back_pts']
            di = np.argmin(np.sqrt((pts[:, 0] - sx)**2 + (pts[:, 1] - sy)**2))
            dt = np.sqrt((pts[di, 0] - cx)**2 + (pts[di, 1] - cy)**2)
            if dt > sd:
                ns[i] = snake[i] + 0.3 * (pts[di] - snake[i])
        # 平滑 3轮
        for _ in range(3):
            s = ns.copy()
            for i in range(1, n_pts-1):
                s[i] = 0.5*ns[i] + 0.25*(ns[i-1]+ns[i+1])
            ns = s
        snake = ns

    # 画线: 深色阴影 + 白线
    pts_i = snake.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts_i], False, (100, 100, 100), thickness+2)
    cv2.polylines(img, [pts_i], False, color, thickness)
    return img

def main():
    for fname in REAL_SIDE_FILES:
        if not os.path.exists(fname):
            print(f"  [SKIP] {fname} 不存在")
            continue

        print(f"\n{'='*50}")
        print(f"处理: {fname}")
        img = cv2.imread(fname)
        if img is None:
            print(f"  读取失败")
            continue

        h_orig, w_orig = img.shape[:2]
        print(f"  尺寸: {w_orig}x{h_orig}")

        sam_result = detect_head(img, view='side')
        if sam_result is None:
            # 尝试缩小
            img2 = cv2.resize(img, (w_orig//2, h_orig//2))
            sam_result = detect_head(img2, view='side')
            if sam_result is None:
                print(f"  SAM 失败, 跳过")
                continue

        mask, contour = sam_result
        pts = contour.reshape(-1, 2).astype(np.float64)
        print(f"  SAM: {len(pts)} 点")

        # 找后脑勺
        result = find_back_of_head(pts)
        print(f"  后脑勺: {result['n_back']}/{result['n_total']} 点 "
              f"({result['n_back']/result['n_total']*100:.0f}%)")
        print(f"  back_angle: {result['back_angle']:.0f}°")
        print(f"  expected_radius: {result['expected_radius']:.0f}px")

        # 画对比图
        comp = img.copy()
        # 画绿线轮廓
        cv2.drawContours(comp, [contour], -1, (0, 220, 60), 3)
        # 画 Snake 白弧
        comp = draw_snake_arc(comp, result, (255, 255, 255), 4)

        # 画辅助标记
        cx, cy = int(result['cx']), int(result['cy'])
        cv2.circle(comp, (cx, cy), 6, (0, 0, 255), -1)
        ba = result['back_angle']
        R = int(result['expected_radius'])
        end_x = int(cx + R * 0.8 * math.cos(np.radians(ba)))
        end_y = int(cy + R * 0.8 * math.sin(np.radians(ba)))
        cv2.line(comp, (cx, cy), (end_x, end_y), (0, 255, 255), 2)

        # 缩小保存
        SCALE = 1200 / max(h_orig, w_orig)
        comp_s = cv2.resize(comp, (int(w_orig*SCALE), int(h_orig*SCALE)))
        out_name = f'test_v5_{os.path.splitext(fname)[0]}.jpg'
        cv2.imwrite(out_name, comp_s, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  → {out_name}")

    print(f"\n{'='*50}")
    print("全部完成")

if __name__ == '__main__':
    main()
