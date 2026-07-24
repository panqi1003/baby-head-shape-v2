"""v5最终参数 vs 线上 对比"""
import cv2, numpy as np, math, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sam_detector import detect_head
from side_analyzer import analyze_side_profile
from standard_compare import draw_comparison

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

def draw_v5_arc(img, cx, cy, ba, ideal_R, contour_pts, start_mult=1.02, cap_pct=0.15, deg_l=50, deg_r=80):
    R = ideal_R * start_mult; n_pts = 60
    angles = np.linspace(np.radians(ba-deg_l), np.radians(ba+deg_r), n_pts)
    snake = np.column_stack([cx+R*np.cos(angles), cy+R*np.sin(angles)])
    cap = R * cap_pct
    for _ in range(40):
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
    cv2.polylines(img, [pts_i], False, (60,60,60), 4)
    cv2.polylines(img, [pts_i], False, (255,255,255), 2)

FILES = [
    'real_side.jpg',
    'test_neck_real_side_jpg.jpg',
    'test_54945_side.jpg',
    '943e524bb386bfbfe1e34fe403e5e826.png',
    '7682d493e0e21620c2a35d9e37f681dd.png',
]

for fname in FILES:
    img = cv2.imread(fname)
    if img is None: continue
    h_orig, w_orig = img.shape[:2]
    SCALE = 1000 / max(h_orig, w_orig)
    nw, nh = int(w_orig*SCALE), int(h_orig*SCALE)

    # ==== 线上 ====
    sr = analyze_side_profile(img)
    if sr is None: continue
    contour_list = sr.pop("_head_contour", None)
    contour_raw = np.array(contour_list, dtype=np.int32).reshape(-1, 1, 2)
    pts = contour_raw.reshape(-1, 2).astype(np.float64)

    online_img, comp = draw_comparison(img, contour_raw, view='side', side_result=sr, side='')
    online_img = cv2.resize(online_img, (nw, nh))
    cv2.putText(online_img, "ONLINE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if comp:
        cv2.putText(online_img, f"sim={comp.get('similarity_score','?')}%", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # ==== v5 ====
    ba, cx, cy, er = v5_params(pts)
    v5_img = cv2.resize(img.copy(), (nw, nh))
    contour_scaled = (contour_raw.astype(np.float64) * SCALE).astype(np.int32)
    cv2.drawContours(v5_img, [contour_scaled], -1, (0, 220, 60), 3)
    draw_v5_arc(v5_img, cx*SCALE, cy*SCALE, ba, er*SCALE, pts*SCALE)
    cv2.putText(v5_img, f"v5 R=1.02 L50 R80 cap15%", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ==== 并排 ====
    cmp = np.hstack([online_img, v5_img])
    cv2.line(cmp, (nw, 0), (nw, nh), (100,100,100), 2)
    cv2.putText(cmp, "ONLINE", (nw//2-50, nh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
    cv2.putText(cmp, "v5", (nw+nw//2-20, nh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    tag = os.path.splitext(fname)[0][:15]
    out = f'test_final_compare_{tag}.jpg'
    cv2.imwrite(out, cmp, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{fname} → {out}")
    print(f"  v5: ba={ba:.0f} R={er:.0f} 1.02R+15% L50 R80")

print("\n完成")
