"""
婴儿头型标准轮廓提取工具

从医学标准侧面头型图中提取红色虚线轮廓，
用于叠加到用户照片上作为参考线。

原理:
1. HSV色彩检测 → 找到所有红色像素
2. 连通域分析 → 每条虚线段成为一个独立区域
3. 质心提取 → 每段取中心点
4. 最近邻排序 → 从上到下追踪轮廓路径
5. 高斯平滑 → 消除锯齿
6. 线性插值 → 生成密集轮廓点
"""

import cv2
import numpy as np
import os


def extract_dashed_contour(image_path, side='right', x_threshold=540):
    """
    从红色虚线图中提取轮廓
    
    Args:
        image_path: 标准头型图片路径
        side: 'right' 或 'left', 指定提取哪一侧的轮廓
        x_threshold: 左右分界的X坐标阈值
    
    Returns:
        contour: (N, 2) 数组, 轮廓点坐标
        dashes: 原始虚线段中心点列表
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'无法读取图片: {image_path}')
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 红色检测 (HSV色彩空间)
    # 红色在HSV中跨越0°/180°边界, 需要两段检测
    mask1 = cv2.inRange(hsv, (0, 40, 40), (15, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 40, 40), (180, 255, 255))
    red_mask = mask1 | mask2
    
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask, connectivity=8
    )
    
    # 提取指定侧的虚线段
    dashes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 100 < area < 600:  # 虚线段大小范围
            cx, cy = centroids[i]
            if side == 'right' and cx >= x_threshold:
                # 排除底部水平带状区域 (可能是图例)
                if not (1145 < cy < 1170 and cx > 690):
                    dashes.append((cx, cy))
            elif side == 'left' and cx < x_threshold:
                if not (1145 < cy < 1170 and cx < 350):
                    dashes.append((cx, cy))
    
    if len(dashes) < 5:
        raise ValueError(f'检测到的虚线段太少 ({len(dashes)}), 请检查图片')
    
    # 最近邻排序 (从上到下追踪)
    ordered = _order_points_nearest(dashes)
    
    # 高斯平滑 + 线性插值
    contour = _smooth_and_interpolate(ordered, window=5, sigma=2.0, step=3)
    
    return contour, dashes


def _order_points_nearest(points):
    """最近邻排序: 从最上方点开始, 每次找最近的未访问点"""
    pts = list(points)
    pts.sort(key=lambda p: p[1])  # 按Y排序找起点
    ordered = [pts[0]]
    remaining = pts[1:]
    
    while remaining:
        last = ordered[-1]
        dists = [((p[0]-last[0])**2 + (p[1]-last[1])**2)**0.5 for p in remaining]
        min_idx = dists.index(min(dists))
        ordered.append(remaining.pop(min_idx))
    
    return ordered


def _smooth_and_interpolate(ordered_points, window=5, sigma=2.0, step=3):
    """
    高斯平滑 + 线性插值
    
    Args:
        ordered_points: 排序后的点列表
        window: 高斯窗口大小
        sigma: 高斯标准差
        step: 插值步长 (像素)
    """
    pts = np.array(ordered_points, dtype=np.float64)
    
    # 高斯权重
    weights = np.exp(-0.5 * (np.arange(window) - window//2)**2 / sigma**2)
    weights /= weights.sum()
    
    # 对x和y分别平滑
    sx = np.convolve(pts[:, 0], weights, mode='same')
    sy = np.convolve(pts[:, 1], weights, mode='same')
    smooth = np.column_stack([sx, sy])
    
    # 线性插值, 生成密集点
    dense = []
    for i in range(len(smooth) - 1):
        p1, p2 = smooth[i], smooth[i+1]
        dist = np.linalg.norm(p2 - p1)
        n = max(2, int(dist / step))
        for t in np.linspace(0, 1, n, endpoint=False):
            dense.append(p1 + t * (p2 - p1))
    dense.append(smooth[-1])
    
    return np.array(dense)


def overlay_contour_on_image(user_img, contour, color=(255, 255, 255), thickness=2):
    """
    将轮廓叠加到用户照片上
    
    Args:
        user_img: 用户照片 (BGR格式)
        contour: 轮廓点数组 (N, 2)
        color: 线条颜色, 默认白色
        thickness: 线条粗细
    
    Returns:
        result: 叠加后的图片
    """
    result = user_img.copy()
    for i in range(len(contour) - 1):
        p1 = (int(contour[i][0]), int(contour[i][1]))
        p2 = (int(contour[i+1][0]), int(contour[i+1][1]))
        cv2.line(result, p1, p2, color, thickness)
    return result


def scale_contour_to_fit(contour, target_width, target_height):
    """
    将轮廓缩放到指定尺寸
    
    Args:
        contour: 原始轮廓点数组
        target_width: 目标宽度
        target_height: 目标高度
    
    Returns:
        scaled: 缩放后的轮廓点数组
    """
    x_min, y_min = contour.min(axis=0)
    x_max, y_max = contour.max(axis=0)
    
    # 计算缩放比例
    scale_x = target_width / (x_max - x_min)
    scale_y = target_height / (y_max - y_min)
    scale = min(scale_x, scale_y)
    
    # 缩放并居中
    scaled = contour.copy().astype(np.float64)
    scaled[:, 0] = (scaled[:, 0] - x_min) * scale
    scaled[:, 1] = (scaled[:, 1] - y_min) * scale
    
    # 居中
    offset_x = (target_width - (scaled[:, 0].max() - scaled[:, 0].min())) / 2
    offset_y = (target_height - (scaled[:, 1].max() - scaled[:, 1].min())) / 2
    scaled[:, 0] += offset_x
    scaled[:, 1] += offset_y
    
    return scaled.astype(np.int32)


def create_verification_image(image_path, contour, dashes, output_path):
    """生成验证图片: 显示原始虚线段和提取的轮廓"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    debug = img.copy()
    
    # 画原始虚线段中心 (红色圆点)
    for cx, cy in dashes:
        cv2.circle(debug, (int(cx), int(cy)), 6, (0, 0, 255), -1)
    
    # 画平滑轮廓 (绿色线)
    for i in range(len(contour) - 1):
        p1 = (int(contour[i][0]), int(contour[i][1]))
        p2 = (int(contour[i+1][0]), int(contour[i+1][1]))
        cv2.line(debug, p1, p2, (0, 255, 0), 2)
    
    # 起点终点标记
    cv2.circle(debug, (int(contour[0][0]), int(contour[0][1])), 10, (255, 255, 0), 3)
    cv2.circle(debug, (int(contour[-1][0]), int(contour[-1][1])), 10, (0, 255, 255), 3)
    
    cv2.imwrite(output_path, debug)
    print(f'验证图已保存: {output_path}')


if __name__ == '__main__':
    base_dir = os.path.join('D:', os.sep, 'baby-head-shape', '标准头型')
    
    # 提取面朝右轮廓 (面部侧)
    right_img = os.path.join(base_dir, '面朝右标准头型.jpg')
    right_contour, right_dashes = extract_dashed_contour(right_img, 'right', 540)
    
    # 提取面朝左轮廓 (面部侧)
    left_img = os.path.join(base_dir, '面朝左标准头型.jpg')
    left_contour, left_dashes = extract_dashed_contour(left_img, 'left', 540)
    
    print(f'面朝右轮廓: {len(right_dashes)}段 → {len(right_contour)}点')
    print(f'  X=[{right_contour[:,0].min():.0f},{right_contour[:,0].max():.0f}], '
          f'Y=[{right_contour[:,1].min():.0f},{right_contour[:,1].max():.0f}]')
    print(f'面朝左轮廓: {len(left_dashes)}段 → {len(left_contour)}点')
    print(f'  X=[{left_contour[:,0].min():.0f},{left_contour[:,0].max():.0f}], '
          f'Y=[{left_contour[:,1].min():.0f},{left_contour[:,1].max():.0f}]')
    
    # 保存轮廓数据
    np.save(os.path.join(base_dir, 'right_contour_final.npy'), right_contour)
    np.save(os.path.join(base_dir, 'left_contour_final.npy'), left_contour)
    
    # 生成验证图
    create_verification_image(
        right_img, right_contour, right_dashes,
        os.path.join(base_dir, 'verify_right_contour.jpg')
    )
    create_verification_image(
        left_img, left_contour, left_dashes,
        os.path.join(base_dir, 'verify_left_contour.jpg')
    )
    
    print('\n提取完成!')
