"""
SAM 零样本头部检测 — 点提示模式，3点+提前退出，~1.5秒
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


def detect_head(image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    MobileSAM 点提示模式 — 3个点 + 提前退出，~1.5秒。
    返回: (head_mask, head_contour) 或 None
    """
    h_orig, w_orig = image.shape[:2]

    # 缩小到 540px (比 640 更快)
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

    try:
        model = _get_model()
    except Exception:
        return None

    # 3 个提示点 (减少 2 个, 省 ~1 秒)
    cx, cy = w // 2, h // 2
    offset = min(w, h) // 5
    prompt_points = [
        [cx, cy],                  # 正中 (大概率命中)
        [cx, cy - offset],         # 偏上 (头顶)
        [cx + offset, cy],         # 偏右
    ]

    best_score, best_mask, best_contour = 0, None, None

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

        # 提前退出: 正中点得分 >0.6 就认为找到了, 不再试其余点
        if i == 0 and best_score > 0.6:
            break

    # 回退: 自动模式
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
