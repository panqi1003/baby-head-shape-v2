import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_green_pixels(image_path):
    # 使用PIL读取图像
    try:
        pil_img = Image.open(image_path)
        img = np.array(pil_img)
    except Exception as e:
        print(f"无法读取图像: {image_path}, 错误: {e}")
        return
    
    # 如果图像是RGBA，转换为RGB
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 转换为HSV颜色空间，以便更好地检测绿色
    # PIL使用RGB，需要先转换为HSV
    from colorsys import rgb_to_hsv
    
    # 创建HSV掩码
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            r, g, b = img[y, x] / 255.0
            hue, sat, val = rgb_to_hsv(r, g, b)
            # 绿色的HSV范围
            if 0.25 <= hue <= 0.45 and sat > 0.3 and val > 0.3:
                mask[y, x] = 255
    
    # 找到绿色像素的坐标
    green_pixels = np.where(mask > 0)
    
    if len(green_pixels[0]) == 0:
        print("未检测到绿色像素")
        return
    
    # 计算绿色像素的边界框
    y_min, y_max = np.min(green_pixels[0]), np.max(green_pixels[0])
    x_min, x_max = np.min(green_pixels[1]), np.max(green_pixels[1])
    
    print(f"图像尺寸: {img.shape}")
    print(f"绿色像素数量: {len(green_pixels[0])}")
    print(f"绿色像素边界框: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # 分析垂直和水平分布
    # 垂直分布（y轴）
    y_hist = np.sum(mask > 0, axis=1)
    # 水平分布（x轴）
    x_hist = np.sum(mask > 0, axis=0)
    
    # 找到峰值（可能对应十字线）
    y_peaks = np.where(y_hist > np.max(y_hist) * 0.5)[0]
    x_peaks = np.where(x_hist > np.max(x_hist) * 0.5)[0]
    
    print(f"垂直方向主要位置 (y): {y_peaks[:10]}... (共{len(y_peaks)}个)")
    print(f"水平方向主要位置 (x): {x_peaks[:10]}... (共{len(x_peaks)}个)")
    
    # 检查是否有连续的垂直线或水平线
    # 垂直线：在某个x位置有连续的绿色像素
    vertical_lines = []
    for x in range(x_min, x_max+1):
        column = mask[:, x]
        green_count = np.sum(column > 0)
        if green_count > img.shape[0] * 0.3:  # 如果超过30%的高度是绿色
            vertical_lines.append(x)
    
    # 水平线：在某个y位置有连续的绿色像素
    horizontal_lines = []
    for y in range(y_min, y_max+1):
        row = mask[y, :]
        green_count = np.sum(row > 0)
        if green_count > img.shape[1] * 0.3:  # 如果超过30%的宽度是绿色
            horizontal_lines.append(y)
    
    print(f"检测到的垂直线位置 (x): {vertical_lines[:10]}... (共{len(vertical_lines)}个)")
    print(f"检测到的水平线位置 (y): {horizontal_lines[:10]}... (共{len(horizontal_lines)}个)")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('原始提取图像')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('绿色掩码')
    
    plt.subplot(1, 3, 3)
    plt.plot(y_hist, range(len(y_hist)), label='垂直分布')
    plt.plot(x_hist, range(len(x_hist)), label='水平分布')
    plt.legend()
    plt.title('像素分布')
    
    plt.tight_layout()
    plt.savefig('green_analysis.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'green_pixels': len(green_pixels[0]),
        'bbox': (x_min, y_min, x_max, y_max),
        'vertical_lines': vertical_lines,
        'horizontal_lines': horizontal_lines
    }

if __name__ == "__main__":
    result = analyze_green_pixels(r"D:\baby-head-shape\标准头型\top_extracted.jpg")
    print("\n分析完成，结果已保存到 green_analysis.jpg")