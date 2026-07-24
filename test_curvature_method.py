"""
找后脑勺 v4 — 极值点+平滑法 (修复版)
"""
import cv2, numpy as np, math, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from sam_detector import detect_head

img = cv2.imread('real_side.jpg')
if img is None: sys.exit("图片加载失败")
h_orig, w_orig = img.shape[:2]

sam_result = detect_head(img, view='side')
if sam_result is None: sys.exit("SAM failed")
mask, contour = sam_result
pts = contour.reshape(-1, 2).astype(np.float64)
n = len(pts)
print(f"Contour: {n} 点")

# ===== 1. 平滑轮廓 =====
smooth_win = 40
pts_smooth = np.zeros_like(pts)
for i in range(n):
    lo = max(0, i - smooth_win)
    hi = min(n, i + smooth_win + 1)
    pts_smooth[i] = np.mean(pts[lo:hi], axis=0)

# ===== 2. 极值点 =====
x_min_idx = np.argmin(pts_smooth[:, 0])
x_max_idx = np.argmax(pts_smooth[:, 0])
back_pt = pts_smooth[x_min_idx]
front_pt = pts_smooth[x_max_idx]
face_dir = front_pt - back_pt
face_angle = np.degrees(math.atan2(face_dir[1], face_dir[0]))

# ===== 3. 质心 + 后脑方向 =====
cx, cy = np.mean(pts_smooth[:, 0]), np.mean(pts_smooth[:, 1])
back_dir_v = back_pt - np.array([cx, cy])
back_angle = np.degrees(math.atan2(back_dir_v[1], back_dir_v[0]))
print(f"面朝方向: {face_angle:.0f}°  后脑方向: {back_angle:.0f}°")

# ===== 4. 选后脑勺点: back_angle ± 70° =====
all_angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
half_range = 70
ad = np.abs(all_angles - back_angle)
ad = np.minimum(ad, 360 - ad)
back_mask = ad < half_range
back_idx = np.where(back_mask)[0]

if len(back_idx) == 0:
    back_mask = ad < 90
    back_idx = np.where(back_mask)[0]

print(f"back_idx: {len(back_idx)} 点, [{back_idx[0]}, {back_idx[-1]}]")

# 处理可能的跳跃 (环形不连续)
gaps = np.diff(back_idx)
big = np.where(gaps > 10)[0]
if len(big) > 0:
    print(f"  跳跃: {len(big)} 处, 取最大连续段")
    segs, cs = [], 0
    for i in range(1, len(back_idx)):
        if back_idx[i] - back_idx[i-1] > 10:
            segs.append((cs, i-1)); cs = i
    segs.append((cs, len(back_idx)-1))
    segs.sort(key=lambda x: -(x[1]-x[0]))
    s, e = segs[0]
    back_idx = back_idx[s:e+1]
    print(f"  最大段: [{back_idx[0]}, {back_idx[-1]}] ({len(back_idx)}点)")

back_pts = pts[back_idx]
best_len = len(back_pts)

# ===== 5. 拟合圆 =====
a, b, c_pt = back_pts[0], back_pts[len(back_pts)//2], back_pts[-1]
ma = np.array([(a[0]+b[0])/2, (a[1]+b[1])/2])
mb = np.array([(b[0]+c_pt[0])/2, (b[1]+c_pt[1])/2])
da, db = np.array([-(b[1]-a[1]), b[0]-a[0]]), np.array([-(c_pt[1]-b[1]), c_pt[0]-b[0]])
try:
    ts = np.linalg.lstsq(np.column_stack([da, -db]), mb - ma, rcond=None)[0]
    fit_center = ma + ts[0] * da
except np.linalg.LinAlgError:
    fit_center = np.mean(back_pts, axis=0)
fit_radius = np.median(np.linalg.norm(back_pts - fit_center, axis=1))
expected_radius = np.median(np.linalg.norm(back_pts - np.array([cx, cy]), axis=1))

# ===== 6. 对比 PCA =====
centered = pts - np.array([cx, cy])
cov = np.cov(centered.T)
eigvals, eigvecs = np.linalg.eigh(cov)
v1 = eigvecs[:, np.argsort(eigvals)[::-1][0]]
pca_back = np.dot(centered, v1) < 0
front_pts_c = centered[~pca_back]
pca_r = np.median(np.linalg.norm(front_pts_c, axis=1)) if len(front_pts_c) > 15 else 0

print(f"\n--- 结果 ---")
print(f"后脑勺: {best_len}点 ({best_len/n*100:.1f}%)")
print(f"拟合半径: {fit_radius:.0f}px")
print(f"expected_radius: {expected_radius:.0f}px")
print(f"\nPCA expected_radius: {pca_r:.0f}px  back: {np.sum(pca_back)}点")
print(f"比值: {expected_radius/pca_r:.2f}" if pca_r > 0 else "")

# ===== 可视化 =====
vis = img.copy()
SCALE = 1500 / max(h_orig, w_orig)
vis = cv2.resize(vis, (int(w_orig*SCALE), int(h_orig*SCALE)))
pts_s = pts * SCALE; pts_sm = pts_smooth * SCALE
cx_s, cy_s = cx * SCALE, cy * SCALE; fc_s = fit_center * SCALE

for i in range(n-1):
    cv2.line(vis, tuple(pts_sm[i].astype(int)), tuple(pts_sm[i+1].astype(int)), (128,128,128), 1)
for i in range(n-1):
    cv2.line(vis, tuple(pts_s[i].astype(int)), tuple(pts_s[i+1].astype(int)), (0,180,80), 1)

# 后脑勺黄色加粗
for i in range(len(back_pts)-1):
    cv2.line(vis, tuple((back_pts[i]*SCALE).astype(int)),
             tuple((back_pts[i+1]*SCALE).astype(int)), (0,255,255), 4)

cv2.circle(vis, tuple((back_pt*SCALE).astype(int)), 8, (255,255,0), -1)
cv2.circle(vis, tuple((front_pt*SCALE).astype(int)), 8, (0,0,255), -1)
cv2.arrowedLine(vis, tuple((back_pt*SCALE).astype(int)), tuple((front_pt*SCALE).astype(int)), (255,0,255), 2)
cv2.circle(vis, (int(fc_s[0]), int(fc_s[1])), int(fit_radius*SCALE), (255,255,255), 2)
cv2.circle(vis, (int(fc_s[0]), int(fc_s[1])), 4, (255,255,255), -1)
cv2.circle(vis, (int(cx_s), int(cy_s)), 4, (0,0,255), -1)

er_s = expected_radius * SCALE
cv2.ellipse(vis, (int(cx_s), int(cy_s)), (int(er_s), int(er_s)), 0,
            back_angle-70, back_angle+70, (0,255,255), 1)

cv2.putText(vis, f"face={face_angle:.0f}deg  back={back_angle:.0f}deg",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
cv2.putText(vis, f"Back: {best_len}pts  R_fit={fit_radius:.0f}  R_exp={expected_radius:.0f}",
            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
cv2.putText(vis, f"PCA_R={pca_r:.0f}  ratio={expected_radius/pca_r:.2f}" if pca_r>0 else "",
            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
cv2.putText(vis, "Yellow=back of head  Gray=smoothed  Green=raw contour",
            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

cv2.imwrite('test_curvature_back_segment.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("\n可视化: test_curvature_back_segment.jpg")
