"""单张: Snake受限 40%"""
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

def draw_snake(img, cx, cy, ba, ideal_R, contour_pts, max_pct, color):
    R = ideal_R * 0.94; n_pts = 60
    angles = np.linspace(np.radians(ba-70), np.radians(ba+70), n_pts)
    snake = np.column_stack([cx+R*np.cos(angles), cy+R*np.sin(angles)])
    cap = R * max_pct
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
    cv2.polylines(img, [pts_i], False, (60,60,60), 3)
    cv2.polylines(img, [pts_i], False, color, 2)

FILES = [
    ('real_side.jpg','宝宝1'),
    ('test_54945_side.jpg','宝宝2'),
    ('943e524bb386bfbfe1e34fe403e5e826.png','宝宝3'),
]
for fname, baby in FILES:
    img = cv2.imread(fname)
    r = detect_head(img, view='side')
    if r is None: continue
    mask, contour = r
    pts = contour.reshape(-1, 2).astype(np.float64)
    ba, cx, cy, er = v5_params(pts)

    h, w = img.shape[:2]; SCALE = 900/max(h,w)
    out = cv2.resize(img.copy(), (int(w*SCALE), int(h*SCALE)))
    cxs, cys = cx*SCALE, cy*SCALE; er_s = er*SCALE; ptss = pts*SCALE

    cv2.drawContours(out, [(contour.astype(np.float64)*SCALE).astype(np.int32)], -1, (0,220,60), 2)

    # 纯圆 (灰虚线)
    R = er_s * 0.94
    angles = np.linspace(np.radians(ba-70), np.radians(ba+70), 60)
    pure = np.column_stack([cxs+R*np.cos(angles), cys+R*np.sin(angles)]).astype(np.int32)
    cv2.polylines(out, [pure.reshape(-1,1,2)], False, (120,120,120), 1)

    draw_snake(out, cxs, cys, ba, er_s, ptss, 0.60, (255,255,255))
    cv2.putText(out, f'{baby} Snake受限 最多外移60%', (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, "white=snake(cap60%) gray=ideal circle green=SAM", (10, out.shape[0]-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    tag = os.path.splitext(fname)[0][:12]
    cv2.imwrite(f'test_step_{tag}_G_cap60.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{baby} → test_step_{tag}_G_cap60.jpg")
print("完成: _G_cap60.jpg ×3")
