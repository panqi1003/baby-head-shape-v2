"""
受限Snake — 每次只看一个参数
"""
import cv2, numpy as np, math, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sam_detector import detect_head

def v5_params(pts):
    n = len(pts); xs, ys = pts[:,0], pts[:,1]
    y_min, y_max = ys.min(), ys.max()
    mid_mask = (ys >= y_min+(y_max-y_min)*0.20) & (ys <= y_min+(y_max-y_min)*0.75)
    if np.sum(mid_mask) < 30: mid_mask = np.ones(n, dtype=bool)
    mid_pts = pts[mid_mask]; mid_idx = np.where(mid_mask)[0]
    lm = np.argmin(mid_pts[:,0])
    lo, hi = max(0, mid_idx[lm]-30), min(n, mid_idx[lm]+31)
    back_pt = np.mean(pts[lo:hi], axis=0)
    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
    ba = np.degrees(math.atan2(back_pt[1]-cy, back_pt[0]-cx))
    all_angles = np.degrees(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))
    ad = np.abs(all_angles - ba); ad = np.minimum(ad, 360-ad)
    back_idx = np.where(ad < 70)[0]
    gaps = np.diff(back_idx); big = np.where(gaps > 10)[0]
    if len(big) > 0:
        segs, cs = [], 0
        for i in range(1, len(back_idx)):
            if back_idx[i]-back_idx[i-1] > 10: segs.append((cs,i-1)); cs=i
        segs.append((cs, len(back_idx)-1)); segs.sort(key=lambda x: -(x[1]-x[0]))
        back_idx = back_idx[segs[0][0]:segs[0][1]+1]
    er = np.median(np.linalg.norm(pts[back_idx]-np.array([cx,cy]), axis=1))
    return ba, cx, cy, er

def draw_snake(img, cx, cy, ba, ideal_R, contour_pts, iters, max_pct, color):
    R = ideal_R * 0.94; n_pts = 60
    angles = np.linspace(np.radians(ba-70), np.radians(ba+70), n_pts)
    snake = np.column_stack([cx+R*np.cos(angles), cy+R*np.sin(angles)])
    cap = R * max_pct if max_pct > 0 else 9999
    for _ in range(iters):
        ns = snake.copy()
        for i in range(n_pts):
            sx, sy = snake[i]; sd = np.sqrt((sx-cx)**2+(sy-cy)**2)
            di = np.argmin(np.sqrt((contour_pts[:,0]-sx)**2+(contour_pts[:,1]-sy)**2))
            dt = np.sqrt((contour_pts[di,0]-cx)**2+(contour_pts[di,1]-cy)**2)
            if dt > sd:
                new_pos = snake[i] + 0.3*(contour_pts[di]-snake[i])
                nd = np.sqrt((new_pos[0]-cx)**2+(new_pos[1]-cy)**2)
                if nd > R + cap:
                    d = new_pos - np.array([cx,cy]); d /= np.linalg.norm(d)
                    new_pos = np.array([cx,cy]) + d*(R+cap)
                ns[i] = new_pos
        for _ in range(3):
            s = ns.copy()
            for i in range(1, n_pts-1): s[i] = 0.5*ns[i]+0.25*(ns[i-1]+ns[i+1])
            ns = s
        snake = ns
    pts_i = snake.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts_i], False, (60,60,60), 3)
    cv2.polylines(img, [pts_i], False, color, 2)

def make_one(fname, iters, max_pct, label, out_name):
    img = cv2.imread(fname)
    r = detect_head(img, view='side')
    if r is None: return None
    mask, contour = r
    pts = contour.reshape(-1, 2).astype(np.float64)
    ba, cx, cy, er = v5_params(pts)

    h, w = img.shape[:2]; SCALE = 900/max(h,w)
    out = cv2.resize(img.copy(), (int(w*SCALE), int(h*SCALE)))
    cxs, cys = cx*SCALE, cy*SCALE; er_s = er*SCALE; ptss = pts*SCALE

    cv2.drawContours(out, [(contour.astype(np.float64)*SCALE).astype(np.int32)], -1, (0,220,60), 2)

    # 纯圆 (虚线参考)
    R_pure = er_s * 0.94
    angles = np.linspace(np.radians(ba-70), np.radians(ba+70), 60)
    pure = np.column_stack([cxs+R_pure*np.cos(angles), cys+R_pure*np.sin(angles)]).astype(np.int32)
    cv2.polylines(out, [pure.reshape(-1,1,2)], False, (120,120,120), 1)

    # Snake 白弧
    draw_snake(out, cxs, cys, ba, er_s, ptss, iters, max_pct, (255,255,255))

    desc = f"{label}" if label else f"iters={iters} cap={int(max_pct*100)}%"
    cv2.putText(out, desc, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, "white=snake  gray=ideal circle  green=SAM", (10, out.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

    cv2.imwrite(out_name, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_name

# 3个宝宝
FILES = [
    ('real_side.jpg', '宝宝1'),
    ('test_54945_side.jpg', '宝宝2'),
    ('943e524bb386bfbfe1e34fe403e5e826.png', '宝宝3'),
]

# 逐个参数: 纯圆, +5%, +10%, +15%, 线上(无限制)
for fname, baby in FILES:
    tag = os.path.splitext(fname)[0][:12]
    print(f"\n=== {baby} ({fname}) ===")

    # A: 纯圆
    p = make_one(fname, 0, 0, f'{baby} 纯圆 (ideal circle)', f'test_step_{tag}_A_pure.jpg')
    print(f"  A: {p}")

    # B: 受限 +5%
    p = make_one(fname, 40, 0.05, f'{baby} Snake受限 最多外移5%', f'test_step_{tag}_B_cap5.jpg')
    print(f"  B: {p}")

    # C: 受限 +10%
    p = make_one(fname, 40, 0.10, f'{baby} Snake受限 最多外移10%', f'test_step_{tag}_C_cap10.jpg')
    print(f"  C: {p}")

    # D: 受限 +15%
    p = make_one(fname, 40, 0.15, f'{baby} Snake受限 最多外移15%', f'test_step_{tag}_D_cap15.jpg')
    print(f"  D: {p}")

    # E: 无限制 (=线上Snake行为)
    p = make_one(fname, 40, 0, f'{baby} Snake无限制 (40轮自由外移)', f'test_step_{tag}_E_free.jpg')
    print(f"  E: {p}")

print("\n全部完成: test_step_*_A/B/C/D/E 每宝宝5张")
