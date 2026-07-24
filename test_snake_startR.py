"""白弧起始半径+不对称角度测试"""
import cv2, numpy as np, math, os
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

def draw_arc(img, cx, cy, ba, ideal_R, contour_pts, start_mult, cap_pct, deg_left, deg_right, color, label, y_off):
    R = ideal_R * start_mult; n_pts = 60
    angles = np.linspace(np.radians(ba-deg_left), np.radians(ba+deg_right), n_pts)
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
    cv2.polylines(img, [pts_i], False, (60,60,60), 3)
    cv2.polylines(img, [pts_i], False, color, 2)
    cv2.putText(img, label, (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

FNAME = '943e524bb386bfbfe1e34fe403e5e826.png'
img = cv2.imread(FNAME)
r = detect_head(img, view='side')
mask, contour = r
pts = contour.reshape(-1, 2).astype(np.float64)
ba, cx, cy, er = v5_params(pts)

h, w = img.shape[:2]; SCALE = 900/max(h,w)
out = cv2.resize(img.copy(), (int(w*SCALE), int(h*SCALE)))
cxs, cys = cx*SCALE, cy*SCALE; er_s = er*SCALE; ptss = pts*SCALE
cv2.drawContours(out, [(contour.astype(np.float64)*SCALE).astype(np.int32)], -1, (0,220,60), 2)

# start_mult, cap, deg_left, deg_right, color, label
TESTS = [
    (1.05, 0.15, 70, 80,  (0,255,200),   '1.05R +15% R=80'),
    (1.05, 0.15, 70, 90,  (0,200,255),   '1.05R +15% R=90'),
    (1.05, 0.15, 70, 100, (150,140,255), '1.05R +15% R=100'),
    (1.05, 0.15, 70, 110, (255,255,255), '1.05R +15% R=110'),
]
for i, t in enumerate(TESTS):
    draw_arc(out, cxs, cys, ba, er_s, ptss, *t, 25+i*28)

cv2.putText(out, "green=SAM  L=左侧角度 R=右侧角度", (10, out.shape[0]-10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
cv2.imwrite('test_startR_943e.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("→ test_startR_943e.jpg")
