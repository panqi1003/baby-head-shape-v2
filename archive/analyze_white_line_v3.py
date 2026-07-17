"""
分析 side_mimo_final.jpg 中白线轮廓 - 高效版
从代码逻辑 + 图像分析两个角度分析白线
"""
import cv2
import numpy as np

img = cv2.imread("D:\\baby-head-shape\\side_mimo_final.jpg")
h, w = img.shape[:2]
print(f"图片尺寸: {w}x{h}")

# ========== 1. 白线生成逻辑 ==========
print("="*60)
print("【一、白线生成逻辑】(从 standard_compare.py)")
print("="*60)
print("""
白线 = 标准头型轮廓, 经过以下处理:
1. 从 .npy 文件加载原始标准轮廓
2. 极坐标采样 360 点, 计算 60th percentile 缩放比例
3. 将标准轮廓缩放+平移到宝宝轮廓中心
4. 首尾距离>3px 时闭合
5. 用 cv2.drawContours 画白色实线 (线宽=2px)

关键: 白线是标准轮廓的 **缩放副本**, 不是自由绘制的曲线。
它是闭合的多边形, 顶点数量取决于 .npy 文件中的原始点数。
""")

# ========== 2. 从代码回溯标准轮廓 ==========
print("="*60)
print("【二、标准轮廓来源分析】")
print("="*60)

import os
std_dir = "D:\\baby-head-shape\\标准头型"
cache_dir = "D:\\baby-head-shape\\std_cache"

# 检查可用的标准轮廓文件
for side in ['left', 'right']:
    for d, label in [(cache_dir, 'std_cache'), (std_dir, '标准头型')]:
        path = os.path.join(d, f'std_side_{side}.npy')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            pts = data.reshape(-1, 2)
            print(f"  {label}/std_side_{side}.npy: {len(pts)} 个点")
            # 分析点间距
            dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
            print(f"    点间距: min={dists.min():.1f}, max={dists.max():.1f}, mean={dists.mean():.1f}, std={dists.std():.1f}")
            # 检查是否闭合
            closure_gap = np.linalg.norm(pts[0] - pts[-1])
            print(f"    首尾间距: {closure_gap:.1f}px")

# ========== 3. 从图像提取白线轮廓 ==========
print(f"\n{'='*60}")
print("【三、从图像提取白线轮廓】")
print("="*60)

# 方法: 使用高亮白色像素 + 形态学处理
# 白线: 线宽约2px, 颜色接近(255,255,255)
bgr = img
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# 快速白色检测 (向量化)
r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
v = np.max(rgb, axis=2).astype(np.float32)
s = (v - np.min(rgb, axis=2).astype(np.float32)) / (v + 1e-6)

# 白色条件: V>220, S<0.15, 三通道接近
white_mask = ((v > 220) & (s < 0.15) & 
              (np.abs(r.astype(int) - g.astype(int)) < 25) & 
              (np.abs(g.astype(int) - b.astype(int)) < 25)).astype(np.uint8) * 255

# 排除底部文字区域 (y > h-80)
white_mask[h-80:, :] = 0
# 排除右上角图例区域
white_mask[:100, w-200:] = 0
# 排除图片边框
white_mask[:3, :] = 0
white_mask[-3:, :] = 0
white_mask[:, :3] = 0
white_mask[:, -3:] = 0

total_white = np.sum(white_mask > 0)
print(f"白色像素总数: {total_white}")

# 找轮廓
contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"白色轮廓区域数: {len(contours)}")

# 按面积排序
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

# 分析前10个最大区域
print(f"\n前10个白色区域:")
for i, cnt in enumerate(contours_sorted[:10]):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    n_pts = len(cnt)
    eps3 = 0.03 * perimeter
    approx = cv2.approxPolyDP(cnt, eps3, True)
    print(f"  #{i}: 面积={area:.0f}, 周长={perimeter:.0f}, "
          f"边界=({x},{y},{bw}x{bh}), 点数={n_pts}, eps3近似={len(approx)}顶点")

# ========== 4. 识别白线轮廓 ==========
print(f"\n{'='*60}")
print("【四、识别白线轮廓】")
print("="*60)

# 白线轮廓的特征:
# - 围绕头部的闭合轮廓
# - 周长较长
# - 不是填充区域 (面积相对周长较小)

for i, cnt in enumerate(contours_sorted[:5]):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    
    # 判断是否是线状轮廓 (面积小但周长长)
    if perimeter > 0:
        fill_ratio = area / (bw * bh) if bw * bh > 0 else 0
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    else:
        fill_ratio = compactness = 0
    
    # 判断是否可能是白线轮廓
    is_line_like = perimeter > 200 and fill_ratio < 0.3 and compactness < 0.3
    
    pts = cnt.reshape(-1, 2)
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    
    print(f"  #{i}: 面积={area:.0f}, 周长={perimeter:.0f}, "
          f"填充率={fill_ratio:.3f}, 紧凑度={compactness:.3f}, "
          f"中心=({cx:.0f},{cy:.0f}), {'← 可能是白线' if is_line_like else ''}")

# ========== 5. 深入分析最可能的白线轮廓 ==========
print(f"\n{'='*60}")
print("【五、白线轮廓详细分析】")
print("="*60)

# 找最可能是白线的轮廓 (周长最长的非填充区域)
main_cnt = None
for cnt in contours_sorted:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    fill_ratio = area / (bw * bh) if bw * bh > 0 else 0
    
    # 白线特征: 围绕头部, 不是填充色块
    # 头部区域大约在图片中部
    cx, cy = x + bw//2, y + bh//2
    in_head_region = (100 < cx < w-100) and (100 < cy < h-100)
    reasonable_size = perimeter > 500 and bh > 200
    
    if in_head_region and reasonable_size and fill_ratio < 0.2:
        main_cnt = cnt
        break

if main_cnt is not None:
    area = cv2.contourArea(main_cnt)
    perimeter = cv2.arcLength(main_cnt, True)
    x, y, bw, bh = cv2.boundingRect(main_cnt)
    pts = main_cnt.reshape(-1, 2)
    n = len(pts)
    
    print(f"选定白线轮廓:")
    print(f"  面积: {area:.0f} px²")
    print(f"  周长: {perimeter:.0f} px")
    print(f"  边界框: ({x},{y}, {bw}x{bh})")
    print(f"  点数: {n}")
    print(f"  中心: ({np.mean(pts[:,0]):.0f}, {np.mean(pts[:,1]):.0f})")
    
    # 多边形近似
    print(f"\n  多边形近似 (顶点数 vs 精度):")
    for eps in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        approx = cv2.approxPolyDP(main_cnt, eps * perimeter, True)
        error = cv2.matchShapes(main_cnt, approx, cv2.CONTOURS_MATCH_I2, 0)
        print(f"    eps={eps:.2f} → {len(approx)} 顶点, 形状误差={error:.4f}")
    
    # 连续性: 相邻点间距分析
    dists = np.sqrt(np.sum(np.diff(pts, axis=0, append=pts[0:1])**2, axis=1))
    print(f"\n  连续性分析 (相邻点间距):")
    print(f"    均值: {np.mean(dists):.2f} px")
    print(f"    标准差: {np.std(dists):.2f} px")
    print(f"    最小: {np.min(dists):.2f} px")
    print(f"    最大: {np.max(dists):.2f} px")
    print(f"    中位数: {np.median(dists):.2f} px")
    
    # 找大间距 (断裂)
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    gap_thresh = mean_d + 2 * std_d
    large_gaps = np.where(dists > gap_thresh)[0]
    print(f"\n  大间距 (>{gap_thresh:.1f}px): {len(large_gaps)} 个")
    if len(large_gaps) > 0:
        for idx in large_gaps[:5]:
            next_idx = (idx + 1) % n
            print(f"    点{idx}→{next_idx}: {dists[idx]:.1f}px, "
                  f"({pts[idx,0]},{pts[idx,1]})→({pts[next_idx,0]},{pts[next_idx,1]})")
    
    # 凸包分析
    hull = cv2.convexHull(main_cnt)
    hull_area = cv2.contourArea(hull)
    hull_perim = cv2.arcLength(hull, True)
    solidity = area / hull_area if hull_area > 0 else 0
    
    # 凹陷点
    hull_set = set(map(tuple, hull.reshape(-1, 2)))
    indentations = [i for i in range(n) if tuple(pts[i]) not in hull_set]
    
    print(f"\n  形状分析:")
    print(f"    凸度: {solidity:.3f} (1.0=完全凸)")
    print(f"    凹陷点数: {len(indentations)} / {n} ({len(indentations)/n*100:.1f}%)")
    
    # 曲率变化 (用角度变化衡量)
    angles = []
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        v1 = pts[prev_i] - pts[i]
        v2 = pts[next_i] - pts[i]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_a = np.clip(cos_a, -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        angles.append(angle)
    angles = np.array(angles)
    
    sharp_angles = np.where(angles < 150)[0]  # 角度<150° 为锐角
    print(f"    锐角顶点 (<150°): {len(sharp_angles)} / {n} ({len(sharp_angles)/n*100:.1f}%)")
    
    # ========== 6. 与绿色轮廓对比 ==========
    print(f"\n{'='*60}")
    print("【六、白线 vs 绿线贴合度】")
    print("="*60)
    
    green_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (35, 50, 50), (85, 255, 255))
    g_ys, g_xs = np.where(green_mask > 0)
    
    if len(g_xs) > 10:
        # 白线: 极坐标采样
        wcx, wcy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        w_r = np.sqrt((pts[:, 0] - wcx)**2 + (pts[:, 1] - wcy)**2)
        w_theta = np.arctan2(pts[:, 1] - wcy, pts[:, 0] - wcx)
        
        # 绿线: 极坐标采样
        gcx, gcy = np.mean(g_xs), np.mean(g_ys)
        g_r = np.sqrt((g_xs - gcx)**2 + (g_ys - gcy)**2)
        g_theta = np.arctan2(g_ys - gcy, g_xs - gcx)
        
        # 360度扇区平均半径
        phis = np.linspace(-np.pi, np.pi, 72, endpoint=False)  # 每5度
        sector_w = 2 * np.pi / 72
        
        w_radii = np.zeros(72)
        g_radii = np.zeros(72)
        
        for i, phi in enumerate(phis):
            # 白线
            mask = np.abs(((w_theta - phi + np.pi) % (2*np.pi)) - np.pi) < sector_w/2
            if np.any(mask):
                w_radii[i] = np.mean(w_r[mask])
            # 绿线
            mask_g = np.abs(((g_theta - phi + np.pi) % (2*np.pi)) - np.pi) < sector_w/2
            if np.any(mask_g):
                g_radii[i] = np.mean(g_r[mask_g])
        
        both = (w_radii > 0) & (g_radii > 0)
        devs = np.abs(w_radii - g_radii)
        sector_angles = np.degrees(phis)
        
        if np.any(both):
            print(f"  有效对比扇区: {np.sum(both)}/72")
            print(f"  平均偏差: {np.mean(devs[both]):.1f} px")
            print(f"  最大偏差: {np.max(devs[both]):.1f} px")
            
            # 分区域
            front = (sector_angles > -45) & (sector_angles < 45)
            back = (sector_angles > 135) | (sector_angles < -135)
            top = (sector_angles > 45) & (sector_angles < 135)
            bottom = (sector_angles < -45) & (sector_angles > -135)
            
            for name, m in [("前部(面部)", front), ("后部(枕部)", back), 
                            ("顶部", top), ("底部", bottom)]:
                c = both & m
                if np.any(c):
                    print(f"    {name}: avg={np.mean(devs[c]):.1f}px, max={np.max(devs[c]):.1f}px")
            
            max_idx = np.argmax(devs * both)
            print(f"  最大偏差角度: {sector_angles[max_idx]:.0f}° ({devs[max_idx]:.1f}px)")

else:
    print("未找到明确的白线轮廓")
    # 回退: 分析图像中线宽约2px的白色结构
    print("\n使用替代方法: 分析细线结构...")
    
    # 用边缘检测找白线
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 255)
    # 只保留头部区域
    edges[:100, :] = 0
    edges[h-80:, :] = 0
    
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_sorted = sorted(edge_contours, key=cv2.arcLength, reverse=True)
    
    for i, cnt in enumerate(edge_sorted[:5]):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        x, y, bw, bh = cv2.boundingRect(cnt)
        print(f"  边缘#{i}: 面积={area:.0f}, 周长={perimeter:.0f}, 边界=({x},{y},{bw}x{bh})")

print(f"\n{'='*60}")
print("【分析完成】")
print("="*60)
