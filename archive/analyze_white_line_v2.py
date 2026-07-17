"""
分析 side_mimo_final.jpg 中白线轮廓 - 聚焦头部区域
"""
import cv2
import numpy as np

img = cv2.imread("D:\\baby-head-shape\\side_mimo_final.jpg")
h, w = img.shape[:2]
print(f"图片尺寸: {w}x{h}")

# ========== 1. 从代码逻辑推导白线是如何生成的 ==========
print("="*60)
print("【白线生成逻辑分析】(从 standard_compare.py 代码)")
print("="*60)
print("""
draw_comparison() 中白线的绘制步骤:
1. 加载标准头型轮廓 (_load_std_contour) → 从 std_cache/std_side_left.npy 或 std_cache/std_side_right.npy
2. 对齐缩放 (_align_and_scale_contour):
   - 极坐标采样 360 点
   - 计算 user/std 的 60th percentile 缩放比例
   - 将标准轮廓缩放并对齐到 user 轮廓中心
   - 闭合: 如果首尾距离>3px 则补上第一点
3. 绘制: cv2.drawContours(overlay2, [ideal_contour], -1, (255,255,255), 2)
   - 线宽 2px, 白色实线
   - 加深色阴影增加对比
""")

# ========== 2. 精确提取白线 ==========
print("="*60)
print("【白线轮廓提取与分析】")
print("="*60)

# 白线是纯白色 (255,255,255), 线宽约2px
# 策略: 在 HSV 空间中, 找高亮度(>240)、极低饱和度(<20) 的像素
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 更精确的白色检测
white_mask = np.zeros((h, w), dtype=np.uint8)

# 条件: H任意, S<30, V>220
for y in range(h):
    for x in range(w):
        r, g, b = rgb[y, x]
        s = hsv[y, x, 1]
        v = hsv[y, x, 2]
        # 纯白色: 亮度高 + 饱和度低 + 三通道接近
        if v > 220 and s < 30 and abs(int(r)-int(g)) < 20 and abs(int(g)-int(b)) < 20:
            white_mask[y, x] = 255

# 只保留头部区域 (从绿色轮廓的位置推断)
# 绿色轮廓的边界框
green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
green_ys, green_xs = np.where(green_mask > 0)
if len(green_xs) > 0:
    gy_min, gy_max = green_ys.min(), green_ys.max()
    gx_min, gx_max = green_xs.min(), green_xs.max()
    # 扩展边框
    margin = 50
    head_region_y = (max(0, gy_min - margin), min(h, gy_max + margin))
    head_region_x = (max(0, gx_min - margin), min(w, gx_max + margin))
    print(f"头部区域: x=[{head_region_x[0]}, {head_region_x[1]}], y=[{head_region_y[0]}, {head_region_y[1]}]")
else:
    head_region_y = (0, h)
    head_region_x = (0, w)

# 裁剪到头部区域
head_mask = np.zeros_like(white_mask)
head_mask[head_region_y[0]:head_region_y[1], head_region_x[0]:head_region_x[1]] = \
    white_mask[head_region_y[0]:head_region_y[1], head_region_x[0]:head_region_x[1]]

white_count = np.sum(head_mask > 0)
print(f"头部区域白色像素数: {white_count}")

# ========== 3. 检测白线轮廓 (用更严格的方式) ==========
# 使用形态学操作增强线性结构
kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# 先膨胀连接邻近像素
dilated = cv2.dilate(head_mask, kernel_line, iterations=1)
# 再腐蚀回去
cleaned = cv2.erode(dilated, kernel_line, iterations=1)

# 找轮廓
contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\n找到 {len(contours)} 个白色轮廓区域")

# 按面积排序
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

print("\n主要轮廓 (面积从大到小):")
for i, cnt in enumerate(contours_sorted[:10]):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    n_pts = len(cnt)
    
    # 计算多边形近似
    eps = 0.03 * perimeter
    approx = cv2.approxPolyDP(cnt, eps, True)
    n_approx = len(approx)
    
    print(f"  #{i}: 面积={area:.0f}, 周长={perimeter:.0f}, "
          f"边界框=({x},{y},{bw},{bh}), 原始点数={n_pts}, 近似顶点={n_approx}")

# ========== 4. 分析最大的白线轮廓 (应该是标准头型轮廓) ==========
if len(contours_sorted) > 0:
    main_cnt = contours_sorted[0]
    area = cv2.contourArea(main_cnt)
    perimeter = cv2.arcLength(main_cnt, True)
    
    print(f"\n{'='*60}")
    print(f"【最大白线轮廓详细分析】")
    print(f"{'='*60}")
    print(f"面积: {area:.0f} px²")
    print(f"周长: {perimeter:.0f} px")
    
    # 多边形近似
    eps_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    print(f"\n多边形近似 (不同 eps):")
    for eps in eps_vals:
        approx = cv2.approxPolyDP(main_cnt, eps * perimeter, True)
        print(f"  eps={eps:.2f} → {len(approx)} 个顶点")
    
    # ========== 5. 连续性分析 ==========
    # 将轮廓点按距离排序 (追踪路径)
    pts = main_cnt.reshape(-1, 2)
    n = len(pts)
    
    # 计算相邻点距离
    dists = np.zeros(n)
    for i in range(n):
        next_i = (i + 1) % n
        dists[i] = np.linalg.norm(pts[next_i] - pts[i])
    
    print(f"\n连续性分析:")
    print(f"  轮廓点数: {n}")
    print(f"  相邻点平均距离: {np.mean(dists):.2f} px")
    print(f"  相邻点最大距离: {np.max(dists):.2f} px")
    print(f"  相邻点最小距离: {np.min(dists):.2f} px")
    print(f"  相邻点距离标准差: {np.std(dists):.2f} px")
    
    # 找出大的跳变 (可能是断裂)
    gap_threshold = np.mean(dists) + 2 * np.std(dists)
    large_gaps = np.where(dists > gap_threshold)[0]
    print(f"  大跳变点 (> {gap_threshold:.1f} px): {len(large_gaps)} 个")
    
    if len(large_gaps) > 0:
        for idx in large_gaps:
            next_idx = (idx + 1) % n
            angle = np.degrees(np.arctan2(pts[next_idx,1]-pts[idx,1], 
                                           pts[next_idx,0]-pts[idx,0]))
            print(f"    点{idx}→{next_idx}: 距离={dists[idx]:.1f}px, "
                  f"位置=({pts[idx,0]},{pts[idx,1]}), 角度={angle:.1f}°")
    
    # ========== 6. 形状特征 ==========
    # 凸包分析
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 轮廓的紧凑度
    compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 长宽比
    x, y, bw, bh = cv2.boundingRect(main_cnt)
    aspect_ratio = bw / bh if bh > 0 else 0
    
    print(f"\n形状特征:")
    print(f"  凸度 (solidity): {solidity:.3f} (1.0=完全凸, <1.0=有凹陷)")
    print(f"  紧凑度: {compactness:.3f} (1.0=完美圆)")
    print(f"  长宽比: {aspect_ratio:.3f} (宽/高)")
    print(f"  边界框: ({x},{y}, {bw}x{bh})")

    # ========== 7. 与绿色轮廓的对比 ==========
    print(f"\n{'='*60}")
    print(f"【白线 vs 绿线贴合度分析】")
    print(f"{'='*60}")
    
    # 白线极坐标分析
    white_pts = main_cnt.reshape(-1, 2).astype(np.float64)
    wcx = np.mean(white_pts[:, 0])
    wcy = np.mean(white_pts[:, 1])
    w_angles = np.arctan2(white_pts[:, 1] - wcy, white_pts[:, 0] - wcx)
    w_dists = np.sqrt((white_pts[:, 0] - wcx)**2 + (white_pts[:, 1] - wcy)**2)
    
    # 绿线
    green_pts_xy = np.column_stack([green_xs, green_ys]).astype(np.float64)
    gcx = np.mean(green_pts_xy[:, 0])
    gcy = np.mean(green_pts_xy[:, 1])
    g_angles = np.arctan2(green_pts_xy[:, 1] - gcy, green_pts_xy[:, 0] - gcx)
    g_dists = np.sqrt((green_pts_xy[:, 0] - gcx)**2 + (green_pts_xy[:, 1] - gcy)**2)
    
    # 360 度扇区平均半径
    n_sectors = 72  # 每5度一个扇区
    sector_size = 2 * np.pi / n_sectors
    
    w_radii = np.zeros(n_sectors)
    g_radii = np.zeros(n_sectors)
    
    for i in range(n_sectors):
        phi = -np.pi + (i + 0.5) * sector_size
        # 白线在该扇区的平均半径
        mask = np.abs(((w_angles - phi + np.pi) % (2*np.pi)) - np.pi) < sector_size / 2
        if np.any(mask):
            w_radii[i] = np.mean(w_dists[mask])
        # 绿线在该扇区的平均半径
        mask_g = np.abs(((g_angles - phi + np.pi) % (2*np.pi)) - np.pi) < sector_size / 2
        if np.any(mask_g):
            g_radii[i] = np.mean(g_dists[mask_g])
    
    # 计算偏差
    both_valid = (w_radii > 0) & (g_radii > 0)
    deviations = np.abs(w_radii - g_radii)
    
    print(f"  有效对比扇区: {np.sum(both_valid)}/{n_sectors}")
    
    if np.any(both_valid):
        # 按区域分组
        sector_angles = np.array([(-np.pi + (i + 0.5) * sector_size) * 180 / np.pi for i in range(n_sectors)])
        
        # 定义区域 (婴儿侧面: 右侧=前/面部, 左侧=后/枕部)
        # 对于面朝右的侧面: 0°=右(前), 180°/-180°=左(后)
        front = (sector_angles > -45) & (sector_angles < 45)
        back = (sector_angles > 135) | (sector_angles < -135)
        top = (sector_angles > 45) & (sector_angles < 135)
        bottom = (sector_angles < -45) & (sector_angles > -135)
        
        print(f"\n  分区域偏差:")
        for name, mask_r in [("前部(面部)", front), ("后部(枕部)", back), 
                              ("顶部", top), ("底部", bottom)]:
            combined = both_valid & mask_r
            if np.any(combined):
                dev = deviations[combined]
                print(f"    {name}: 平均={np.mean(dev):.1f}px, 最大={np.max(dev):.1f}px")
        
        print(f"\n  全局: 平均偏差={np.mean(deviations[both_valid]):.1f}px, "
              f"最大={np.max(deviations[both_valid]):.1f}px")
        
        # 找最大偏差位置
        max_idx = np.argmax(deviations * both_valid)
        angle_deg = sector_angles[max_idx]
        print(f"  最大偏差位置: {angle_deg:.0f}° (偏差 {deviations[max_idx]:.1f}px)")

# ========== 8. 绘制诊断图 ==========
debug_img = img.copy()

# 画白色掩码轮廓
if len(contours_sorted) > 0:
    cv2.drawContours(debug_img, [contours_sorted[0]], -1, (0, 0, 255), 2)

# 画绿色轮廓
cv2.drawContours(debug_img, [green_mask], -1, (0, 255, 0), 1)

cv2.imwrite("D:\\baby-head-shape\\debug_white_line_analysis.jpg", debug_img)
print(f"\n诊断图已保存: debug_white_line_analysis.jpg")

print(f"\n{'='*60}")
print(f"【结论】")
print(f"{'='*60}")
