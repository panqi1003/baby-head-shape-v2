"""测试不同角度范围对白弧覆盖的影响"""
import cv2, numpy as np, math, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sam_detector import detect_head

def v5_back_angle(pts):
    n = len(pts); xs, ys = pts[:,0], pts[:,1]
    y_min, y_max = ys.min(), ys.max()
    mid_mask = (ys >= y_min+(y_max-y_min)*0.20) & (ys <= y_min+(y_max-y_min)*0.75)
    if np.sum(mid_mask) < 30: mid_mask = np.ones(n, dtype=bool)
    mid_pts = pts[mid_mask]; mid_idx = np.where(mid_mask)[0]
    lm = np.argmin(mid_pts[:,0])
    lo, hi = max(0, mid_idx[lm]-30), min(n, mid_idx[lm]+31)
    back_pt = np.mean(pts[lo:hi], axis=0)
    cx, cy = np.mean(pts[:,0]), np.mean(pts[:,1])
    return np.degrees(math.atan2(back_pt[1]-cy, back_pt[0]-cx)), cx, cy

def draw_arc(img, cx, cy, ba, R, half_angle, contour_pts, color, label, y_offset):
    n_pts = 60
    angles = np.linspace(np.radians(ba-half_angle), np.radians(ba+half_angle), n_pts)
    snake = np.column_stack([cx+R*np.cos(angles), cy+R*np.sin(angles)])
    for _ in range(40):
        ns = snake.copy()
        for i in range(n_pts):
            sx, sy = snake[i]; sd = np.sqrt((sx-cx)**2+(sy-cy)**2)
            di = np.argmin(np.sqrt((contour_pts[:,0]-sx)**2+(contour_pts[:,1]-sy)**2))
            dt = np.sqrt((contour_pts[di,0]-cx)**2+(contour_pts[di,1]-cy)**2)
            if dt > sd: ns[i] = snake[i] + 0.3*(contour_pts[di]-snake[i])
        for _ in range(3):
            s = ns.copy()
            for i in range(1, n_pts-1): s[i] = 0.5*ns[i]+0.25*(ns[i-1]+ns[i+1])
            ns = s
        snake = ns
    pts_i = snake.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(img, [pts_i], False, (60,60,60), 4)
    cv2.polylines(img, [pts_i], False, color, 2)
    cv2.putText(img, f"{label} +-{half_angle}", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

FNAME = '943e524bb386bfbfe1e34fe403e5e826.png'
img = cv2.imread(FNAME)
r = detect_head(img, view='side')
if r is None: sys.exit("SAM failed")
mask, contour = r
pts = contour.reshape(-1, 2).astype(np.float64)
ba, cx, cy = v5_back_angle(pts)

# 后脑勺点到质心距离的中位数
all_angles = np.degrees(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))
ad = np.abs(all_angles - ba); ad = np.minimum(ad, 360-ad)
er = np.median(np.linalg.norm(pts[ad < 70] - np.array([cx,cy]), axis=1))

h, w = img.shape[:2]; SCALE = 900/max(h,w)
out = cv2.resize(img.copy(), (int(w*SCALE), int(h*SCALE)))
cv2.drawContours(out, [(contour.astype(np.float64)*SCALE).astype(np.int32)], -1, (0,220,60), 2)
cv2.putText(out, f"GREEN = SAM contour", (10, h-10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,60), 1)

R = er * SCALE * 0.94
cxs, cys = cx*SCALE, cy*SCALE
ptss = pts*SCALE

# 5 种角度范围: ±40° ~ ±80°
HALF_ANGLES = [40, 50, 60, 70, 80]
COLORS = [(0,255,200), (0,220,255), (100,180,255), (200,140,255), (255,200,100)]

for i, ha in enumerate(HALF_ANGLES):
    draw_arc(out, cxs, cys, ba, R, ha, ptss, COLORS[i], f"A{ha*2}", 25+i*30)

# 线上 PCA 版本做对比
centered = pts - np.array([cx, cy])
cov = np.cov(centered.T); eigvals, eigvecs = np.linalg.eigh(cov)
v1 = eigvecs[:, np.argsort(eigvals)[::-1][0]]
pca_ba = np.degrees(math.atan2(-v1[1], -v1[0]))
angles = np.linspace(np.radians(pca_ba-70), np.radians(pca_ba+70), 60)
R_pca = np.median(np.linalg.norm(centered[np.dot(centered,v1)>=0], axis=1)) * SCALE * 0.94
snake_pca = np.column_stack([cxs+R_pca*np.cos(angles), cys+R_pca*np.sin(angles)])
cv2.polylines(out, [snake_pca.astype(np.int32).reshape(-1,1,2)], False, (0,0,255), 2)
cv2.putText(out, "ONLINE (PCA)", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

OUT = 'test_angle_range_943e.jpg'
cv2.imwrite(OUT, out, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(f"→ {OUT}")
print(f"  back_angle={ba:.0f}  R={er:.0f}")
print(f"  角度范围: ±40°~±80°  + PCA线上版(红色)")
print(f"  彩色=不同范围, 绿色=SAM轮廓")
