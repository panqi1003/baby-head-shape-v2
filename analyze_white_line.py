"""
分析 side_mimo_final.jpg 中白线轮廓的连续性、断裂和贴合度
"""
import cv2
import numpy as np

img = cv2.imread("D:\\baby-head-shape\\side_mimo_final.jpg")
if img is None:
    print("无法读取图片")
    exit()

h, w = img.shape[:2]
print(f"图片尺寸: {w}x{h}")

# ========== 1. 提取白色轮廓 ==========
# 白色: 高亮度、低饱和度
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 白色掩码: 饱和度低(<50), 亮度高(>200)
white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

# 去除图例区域 (右上角, y<100)
white_mask[:100, :] = 0
# 去除底部文字区域 (y > h-60)
white_mask[h-60:, :] = 0

white_pixels = np.sum(white_mask > 0)
print(f"白色像素数量: {white_pixels}")

# 找到白色像素的坐标
ys, xs = np.where(white_mask > 0)
if len(xs) == 0:
    print("未检测到白色轮廓线")
    exit()

# ========== 2. 提取轮廓点 ==========
# 找轮廓
contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\n找到 {len(contours)} 个白色轮廓区域")

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    print(f"  轮廓{i}: 面积={area:.0f}, 周长={perimeter:.0f}, 边界框=({x},{y},{bw},{bh}), 点数={len(cnt)}")

# 合并所有白色点
all_white_pts = np.column_stack([xs, ys])

# ========== 3. 构建白线轮廓 (按极坐标排序) ==========
# 白线的中心
cx = np.median(xs)
cy = np.median(ys)
print(f"\n白色轮廓中心: ({cx:.1f}, {cy:.1f})")

# 极坐标排序
angles = np.arctan2(ys - cy, xs - cx)
order = np.argsort(angles)
sorted_xs = xs[order]
sorted_ys = ys[order]
sorted_angles = angles[order]

# 按角度分段: 找角度跳变
angle_diffs = np.diff(sorted_angles)
# 调整 wrap-around
angle_diffs = (angle_diffs + np.pi) % (2*np.pi) - np.pi
gap_threshold = 0.15  # 角度跳变 > 0.15 rad 视为潜在断裂
gap_indices = np.where(np.abs(angle_diffs) > gap_threshold)[0]

print(f"\n角度跳变分析:")
print(f"  总点数: {len(sorted_xs)}")
print(f"  跳变点数 (> {gap_threshold:.2f} rad): {len(gap_indices)}")

if len(gap_indices) > 0:
    print("  跳变位置:")
    for idx in gap_indices:
        a1 = np.degrees(sorted_angles[idx])
        a2 = np.degrees(sorted_angles[idx+1])
        gap = abs(angle_diffs[idx])
        print(f"    点{idx}: {a1:.1f}° → {a2:.1f}°, 跳变={np.degrees(gap):.1f}°")
else:
    print("  无明显跳变")

# ========== 4. 连续性分析 ==========
# 将排序后的点按极坐标扇区划分
n_sectors = 360
sector_counts = np.zeros(n_sectors)
for a in sorted_angles:
    sector = int((a + np.pi) / (2 * np.pi) * n_sectors) % n_sectors
    sector_counts[sector] += 1

empty_sectors = np.where(sector_counts == 0)[0]
partial_sectors = np.where(sector_counts < 1)[0]  # < 1 pixel per sector

print(f"\n连续性分析 (360扇区):")
print(f"  空扇区数: {len(empty_sectors)}/{n_sectors}")
print(f"  连续性: {(n_sectors - len(empty_sectors))/n_sectors*100:.1f}%")

if len(empty_sectors) > 0:
    # 找出空扇区的连续段
    gaps = []
    gap_start = empty_sectors[0]
    for i in range(1, len(empty_sectors)):
        if empty_sectors[i] != empty_sectors[i-1] + 1:
            gaps.append((gap_start, empty_sectors[i-1]))
            gap_start = empty_sectors[i]
    gaps.append((gap_start, empty_sectors[-1]))
    
    print(f"  断裂段 (角度范围):")
    for s, e in gaps:
        a_start = s * 360 / n_sectors - 180
        a_end = e * 360 / n_sectors - 180
        span = (e - s + 1) * 360 / n_sectors
        print(f"    {a_start:.0f}° ~ {a_end:.0f}° (跨度 {span:.1f}°)")

# ========== 5. 与绿色轮廓的贴合度分析 ==========
# 提取绿色轮廓
green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
green_ys, green_xs = np.where(green_mask > 0)
print(f"\n绿色轮廓像素数: {len(green_xs)}")

if len(green_xs) > 10:
    gcx = np.median(green_xs)
    gcy = np.median(green_ys)
    
    # 绿线极坐标采样 (360点)
    g_angles = np.arctan2(green_ys - gcy, green_xs - gcx)
    g_dists = np.sqrt((green_xs - gcx)**2 + (green_ys - gcy)**2)
    
    # 白线极坐标采样 (360点)
    w_dists = np.sqrt((sorted_xs - cx)**2 + (sorted_ys - cy)**2)
    
    # 按角度插值白线到360点
    phis = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    
    # 白线在每个角度的半径 (取该扇区内的平均)
    w_r_interp = np.zeros(360)
    for i, phi in enumerate(phis):
        sector_mask = np.abs(sorted_angles - phi) < (2*np.pi/360)
        if np.any(sector_mask):
            w_r_interp[i] = np.mean(w_dists[sector_mask])
        else:
            w_r_interp[i] = 0
    
    # 绿线在每个角度的半径
    g_r_interp = np.zeros(360)
    for i, phi in enumerate(phis):
        sector_mask = np.abs(g_angles - phi) < (2*np.pi/360)
        if np.any(sector_mask):
            g_r_interp[i] = np.mean(g_dists[sector_mask])
        else:
            g_r_interp[i] = 0
    
    # 计算偏差 (仅比较两者都有数据的角度)
    both_valid = (w_r_interp > 0) & (g_r_interp > 0)
    if np.any(both_valid):
        deviations = np.abs(w_r_interp - g_r_interp)
        deviations_valid = deviations[both_valid]
        
        # 按区域分析偏差
        front_mask = both_valid & (phis > -np.pi/4) & (phis < np.pi/4)  # 前部(面部)
        back_mask = both_valid & ((phis > 3*np.pi/4) | (phis < -3*np.pi/4))  # 后部(枕部)
        top_mask = both_valid & (phis > np.pi/4) & (phis < 3*np.pi/4)  # 顶部
        bottom_mask = both_valid & (phis > -3*np.pi/4) & (phis < -np.pi/4)  # 底部
        
        print(f"\n白线 vs 绿线贴合度分析:")
        print(f"  有效对比角度数: {np.sum(both_valid)}/360")
        print(f"  平均偏差: {np.mean(deviations_valid):.1f} px")
        print(f"  最大偏差: {np.max(deviations_valid):.1f} px")
        print(f"  偏差标准差: {np.std(deviations_valid):.1f} px")
        
        if np.any(front_mask):
            print(f"  前部(面部) 平均偏差: {np.mean(deviations[front_mask]):.1f} px")
        if np.any(back_mask):
            print(f"  后部(枕部) 平均偏差: {np.mean(deviations[back_mask]):.1f} px")
        if np.any(top_mask):
            print(f"  顶部 平均偏差: {np.mean(deviations[top_mask]):.1f} px")
        if np.any(bottom_mask):
            print(f"  底部 平均偏差: {np.mean(deviations[bottom_mask]):.1f} px")
        
        # 找出偏差最大的区域
        max_dev_idx = np.argmax(deviations_valid)
        valid_indices = np.where(both_valid)[0]
        worst_phi = phis[valid_indices[max_dev_idx]]
        worst_angle = np.degrees(worst_phi)
        print(f"  最大偏差位置: {worst_angle:.0f}° (偏差 {deviations_valid[max_dev_idx]:.1f} px)")

# ========== 6. 白线多边形特征分析 ==========
# 用多边形近似白线
# 先获取主要白线轮廓
if len(contours) > 0:
    # 选择最大的轮廓
    main_cnt = max(contours, key=cv2.contourArea)
    
    # 多边形近似
    perim = cv2.arcLength(main_cnt, True)
    for eps_factor in [0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(main_cnt, eps_factor * perim, True)
        print(f"  多边形近似 (eps={eps_factor:.2f}): {len(approx)} 个顶点")

print("\n" + "="*60)
print("总结:")
print("="*60)
