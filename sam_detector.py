"""
SAM 零样本头部检测
- 俯视图: 点提示模式，3点+提前退出，~1.5秒
- 侧面图: bbox 提示 + 多点合并回退，~2秒
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple

_sam_model = None


def _get_model():
    global _sam_model
    if _sam_model is None:
        from ultralytics import SAM
        _sam_model = SAM('mobile_sam.pt')
    return _sam_model


def _score_mask(mask, h, w, img_center, img_area):
    """对单个 mask 评分，返回 (score, contour)"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    area_ratio = area / img_area
    if area_ratio < 0.02 or area_ratio > 0.65:
        return 0, None

    M = cv2.moments(contour)
    if M['m00'] == 0:
        return 0, None
    cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
    max_dist = math.sqrt(img_area) / 2
    dist = math.sqrt((cx - img_center[0])**2 + (cy - img_center[1])**2)
    center_score = max(0, 1.0 - dist / max_dist)

    # 椭圆度
    if len(contour) >= 10:
        try:
            hull = cv2.convexHull(contour)
            el = cv2.fitEllipse(hull)
            em = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(em, el, 255, -1)
            cm = np.zeros_like(em)
            cv2.drawContours(cm, [hull], -1, 255, -1)
            inter = np.sum((em > 0) & (cm > 0))
            union = np.sum((em > 0) | (cm > 0))
            ellipse_score = inter / union if union > 0 else 0
        except Exception:
            ellipse_score = 0.2
    else:
        ellipse_score = 0.1

    area_norm = min(area_ratio / 0.15, 1.0)
    score = area_norm * 0.25 + center_score * 0.25 + ellipse_score * 0.50
    return score, contour


def _mask_span_ratio(mask, w):
    """返回 mask 在图像左右半的分布: 左半占比 / 总占比。0.5=均匀分布"""
    left = mask[:, :w//2].mean()
    right = mask[:, w//2:].mean()
    total = left + right
    if total < 1e-9:
        return 0
    return left / total


def _detect_with_bbox(model, image_small, h, w, img_center, img_area, margin=0.12):
    """用 bbox 提示检测，返回 (mask, contour, score) 或 (None, None, 0)"""
    try:
        bbox = [[int(w*margin), int(h*margin), int(w*(1-margin)), int(h*(1-margin))]]
        results = model(image_small, bboxes=bbox)
        if results and results[0].masks and len(results[0].masks.data) > 0:
            mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
            score, contour = _score_mask(mask, h, w, img_center, img_area)
            return mask, contour, score
    except Exception:
        pass
    return None, None, 0


def _detect_with_points(model, image_small, points, h, w, img_center, img_area):
    """多点独立调用，返回最佳 (mask, contour, score)"""
    best_score, best_mask, best_contour = 0, None, None
    for pt in points:
        try:
            results = model(image_small, points=[pt], labels=[1])
        except Exception:
            continue
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            continue
        mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        score, contour = _score_mask(mask, h, w, img_center, img_area)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_contour = contour
    return best_mask, best_contour, best_score


def _merge_point_masks(model, image_small, points, h, w, img_center, img_area):
    """多点独立调用 → 合并所有 mask → convex hull → 填洞"""
    masks = []
    for pt in points:
        try:
            results = model(image_small, points=[pt], labels=[1])
            if results and results[0].masks and len(results[0].masks.data) > 0:
                mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
                score, _ = _score_mask(mask, h, w, img_center, img_area)
                if score >= 0.3:
                    masks.append(mask)
        except Exception:
            continue

    if len(masks) < 2:
        return None, None, 0

    # 并集
    merged = np.maximum.reduce(masks)
    merged_u8 = (merged).astype(np.uint8)

    # 形态学闭运算填小洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(merged_u8, cv2.MORPH_CLOSE, kernel)

    # convex hull 填大凹槽
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, 0

    all_pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    hull = cv2.convexHull(all_pts.astype(np.float32))
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)

    # 验证: 面积不能超过 65%
    hull_area = hull_mask.sum() / 255 / (h * w)
    if hull_area < 0.05 or hull_area > 0.65:
        return None, None, 0

    score, contour = _score_mask(hull_mask, h, w, img_center, img_area)
    return hull_mask, contour, score


def create_guide_mask(h, w, view='top', center_y=0.55):
    """
    创建引导框椭圆 mask，排除背景干扰。
    center_y: 椭圆中心在图像高度的比例 (俯视=0.5, 侧面=0.55 头部偏上)
    """
    cx = w // 2
    cy = int(h * (0.5 if view == 'top' else center_y))
    if view == 'top':
        rx = int(w * 0.28)
        ry = int(h * 0.33)
    else:
        rx = int(w * 0.26)
        ry = int(h * 0.31)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    # 边缘羽化，避免硬边界干扰 SAM
    mask = cv2.GaussianBlur(mask, (11, 11), 3)
    return mask


def detect_head(image: np.ndarray, view: str = 'top',
                guide_mask: np.ndarray = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    零样本头部检测。

    view='top':  俯视图 — 点提示，3点+提前退出，~1.5秒
    view='side': 侧面图 — 点提示(同俯视)，bbox回退，~1.5秒
    guide_mask:  可选，仅俯视图使用(侧面图忽略)
    """
    h_orig, w_orig = image.shape[:2]

    # 缩小到 540px
    max_side = 540
    if max(h_orig, w_orig) > max_side:
        scale = max_side / max(h_orig, w_orig)
        image_small = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)))
    else:
        scale = 1.0
        image_small = image

    h, w = image_small.shape[:2]
    img_center = (w / 2, h / 2)
    img_area = h * w

    # 俯视图: 引导框 mask 涂黑椭圆外区域
    if view == 'top' and guide_mask is not None:
        if guide_mask.shape[:2] != (h, w):
            guide_mask = cv2.resize(guide_mask, (w, h))
        gm = (guide_mask / 255.0).astype(np.float32)
        image_small = (image_small.astype(np.float32) * gm[:, :, None]).astype(np.uint8)

    try:
        model = _get_model()
    except Exception:
        return None

    cx, cy = w // 2, h // 2
    offset = min(w, h) // 5
    best_score, best_mask, best_contour = 0, None, None

    # ========== 统一策略: 点提示 + 提前退出 ==========
    # 俯视和侧面使用相同的点提示策略
    # MiMo Pro 审查结论: SAM点提示对"点击物体→分割整个物体"非常擅长
    # 侧面不需要crop/bbox/guide_mask — 这些反而干扰SAM
    prompt_points = [
        [cx, cy],                  # 正中 (大概率命中)
        [cx, cy - offset],         # 偏上 (头顶)
        [cx + offset, cy],         # 偏右
    ]

    for i, pt in enumerate(prompt_points):
        try:
            results = model(image_small, points=[pt], labels=[1])
        except Exception:
            continue
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            continue
        mask = (results[0].masks.data[0].cpu().numpy() * 255).astype(np.uint8)
        score, contour = _score_mask(mask, h, w, img_center, img_area)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_contour = contour
        # 提前退出: 正中点得分 >0.6 就认为找到了
        if i == 0 and best_score > 0.6:
            break

    # 侧面图点提示失败 → bbox 回退
    if view == 'side' and best_mask is None:
        mask, contour, score = _detect_with_bbox(
            model, image_small, h, w, img_center, img_area, margin=0.10)
        if mask is not None and score >= 0.3:
            best_score, best_mask, best_contour = score, mask, contour

    # ========== 回退: 自动模式 ==========
    if best_mask is None:
        try:
            results = model(image_small, retina_masks=True)
            if results and results[0].masks and len(results[0].masks.data) > 0:
                for i in range(min(len(results[0].masks.data), 20)):
                    mask = (results[0].masks.data[i].cpu().numpy() * 255).astype(np.uint8)
                    score, contour = _score_mask(mask, h, w, img_center, img_area)
                    if score > best_score:
                        best_score = score
                        best_mask = mask
                        best_contour = contour
        except Exception:
            pass

    if best_mask is None or best_score < 0.3:
        return None

    if scale != 1.0:
        best_mask = cv2.resize(best_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_contour = max(cnts, key=cv2.contourArea)

    return best_mask, best_contour
