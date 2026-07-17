"""
Snake v1 备份 — 2026-07-15
硬编码参数版本的 Snake 主动轮廓算法，用于侧面标准头型对比。

参数来源：用户逐张调参确认
- face-right: a1=165, a2=260, r=(major+minor)/4 * 0.94
- face-left:  a1=345, a2=440, r=(major+minor)/4 * 0.94

核心逻辑：
1. fitEllipse → 中心(cx,cy)、长短轴(major,minor)
2. 初始半径 r = (major+minor)/4 * 0.94
3. 面朝方向检测: mid_x 左右点数比较
4. 硬编码角度范围初始化圆弧
5. Snake 迭代: 40轮, 只向外移动(0.3步长), 3次平滑
"""

import cv2
import numpy as np


def snake_side_comparison(target_pts):
    """
    对侧面SAM轮廓运行Snake主动轮廓，生成标准后枕弧线。

    Args:
        target_pts: SAM提取的头部轮廓点 (N, 2) 或 (N, 1, 2)

    Returns:
        snake: Snake变形后的弧线点 (60, 2)
    """
    if target_pts.ndim == 3:
        target_pts = target_pts.reshape(-1, 2)
    target_pts = target_pts.astype(np.float32)

    # 1. 椭圆拟合 → 中心+半径
    e = cv2.fitEllipse(target_pts)
    (cx, cy), (major, minor), _ = e
    r = (major + minor) / 4 * 0.94

    # 2. 面朝方向检测
    mid_x = (target_pts[:, 0].min() + target_pts[:, 0].max()) / 2
    fr = np.sum(target_pts[:, 0] > mid_x) > np.sum(target_pts[:, 0] < mid_x)
    a1, a2 = (165, 260) if fr else (345, 440)

    # 3. 初始化Snake圆弧
    t = np.linspace(np.radians(a1), np.radians(a2), 60)
    snake = np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])

    # 4. Snake迭代
    for _ in range(40):
        ns = snake.copy()
        for i in range(60):
            dc = np.sqrt((snake[i, 0] - cx) ** 2 + (snake[i, 1] - cy) ** 2)
            di = np.argmin(np.sqrt((target_pts[:, 0] - snake[i, 0]) ** 2 + (target_pts[:, 1] - snake[i, 1]) ** 2))
            dt = np.sqrt((target_pts[di, 0] - cx) ** 2 + (target_pts[di, 1] - cy) ** 2)
            if dt > dc:  # 只允许向外移动
                ns[i] = snake[i] + 0.3 * (target_pts[di] - snake[i])
        # 平滑 (3次)
        for _ in range(3):
            s = ns.copy()
            for i in range(1, 59):
                s[i] = 0.5 * ns[i] + 0.25 * (ns[i - 1] + ns[i + 1])
            ns = s
        snake = ns

    return snake


# ============================================================
# 测试入口 (直接运行: python snake_v1_backup.py)
# ============================================================
if __name__ == "__main__":
    import sys
    from side_analyzer import analyze_side_profile

    # 用当前图片测试
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_side_realistic.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"找不到图片: {img_path}")
        sys.exit(1)

    result = analyze_side_profile(img)
    if result is None:
        print("侧面分析失败")
        sys.exit(1)

    contour_list = result.get("_head_contour")
    if contour_list is None:
        print("未找到轮廓")
        sys.exit(1)

    contour = np.array(contour_list, dtype=np.int32)
    snake = snake_side_comparison(contour)

    # 画图保存
    out = img.copy()
    cv2.drawContours(out, [contour], -1, (0, 230, 50), 3)  # 绿色实测
    for pt in snake.astype(np.int32):
        cv2.circle(out, tuple(pt), 2, (255, 255, 255), -1)
    cv2.polylines(out, [snake.astype(np.int32)], False, (255, 255, 255), 2)

    out_path = img_path.replace(".jpg", "_snake_v1.jpg")
    cv2.imwrite(out_path, out)
    print(f"已保存: {out_path}")
