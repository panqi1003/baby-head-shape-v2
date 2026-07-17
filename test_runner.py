"""
自动化测试体系: 合成图生成 → 全流程分析 → 数值验证 → MiMo视觉验证 → 报告
用法: python test_runner.py [--quick] [--no-mimo]
"""
import cv2, numpy as np, os, sys, time, json, argparse, math

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# PART 1: 合成测试图生成器
# ============================================================

def _add_hair_texture(img, mask, color=(55, 40, 30), coverage=0.4):
    """在头部mask内添加头发纹理"""
    h, w = img.shape[:2]
    # 简单 Perlin-like 噪声
    noise = np.random.randn(h, w) * 30
    noise = cv2.GaussianBlur(noise, (21, 21), 5)
    hair_region = (mask > 0) & (np.random.random((h, w)) < coverage)
    # 头发在顶部和侧面更多
    y_gradient = np.tile(np.linspace(0, 1, h).reshape(-1, 1), (1, w))
    hair_prob = coverage * (1.0 - y_gradient * 0.5)  # 顶部概率更高
    hair_mask = (mask > 0) & (np.random.random((h, w)) < hair_prob)
    for c in range(3):
        img[:, :, c] = np.where(hair_mask, np.clip(img[:, :, c] + noise, 0, 255), img[:, :, c])
    # 发际线加深
    img = img.astype(np.float32)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1.0 - hair_mask.astype(np.float32) * 0.4)
    return np.clip(img, 0, 255).astype(np.uint8)


def _add_body(img, head_mask, head_bottom_y):
    """在头部下方添加脖子和身体区域"""
    h, w = img.shape[:2]
    body_top = head_bottom_y
    body_h = h - body_top
    if body_h < 20:
        return img

    # 脖子: 窄区域
    neck_w = w // 6
    neck_cx = w // 2
    neck_rect = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(neck_rect, (neck_cx - neck_w//2, body_top),
                  (neck_cx + neck_w//2, body_top + body_h//3), 255, -1)
    cv2.GaussianBlur(neck_rect, (15, 15), 5, dst=neck_rect)

    # 肩膀: 渐变展开
    shoulder = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[neck_cx - neck_w//2, body_top + body_h//4],
                    [w//2 - w//3, h-1], [w//2 + w//3, h-1],
                    [neck_cx + neck_w//2, body_top + body_h//4]], np.int32)
    cv2.fillConvexPoly(shoulder, pts, 200)
    cv2.GaussianBlur(shoulder, (31, 31), 10, dst=shoulder)

    body_mask = np.clip(neck_rect.astype(float) + shoulder.astype(float), 0, 255).astype(np.uint8)
    # 肤色填充
    skin = np.array([185, 150, 140], dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.where(body_mask > 10,
                                (img[:, :, c].astype(float) * 0.3 + skin[c] * 0.7).astype(np.uint8),
                                img[:, :, c])
    # 衣服颜色填充肩膀以下
    cloth = np.array([220, 180, 200], dtype=np.uint8)
    cloth_region = shoulder > 50
    for c in range(3):
        img[:, :, c] = np.where(cloth_region, cloth[c], img[:, :, c])

    return img


def _draw_coin(img, cx, cy, r=25):
    """在指定位置画一元硬币"""
    cv2.circle(img, (cx, cy), r, (200, 200, 200), -1)
    cv2.circle(img, (cx, cy), r, (150, 150, 150), 2)
    cv2.circle(img, (cx, cy), r-4, (150, 150, 150), 1)
    # 硬币纹理
    cv2.circle(img, (cx, cy), r//3, (120, 120, 120), 1)


def make_test_case(case_id, variant):
    """
    生成一张测试图, 返回 (image, ground_truth_dict).

    variant 可选:
      'normal'  - 正常头型
      'brachy'  - 扁头(宽>长)
      'dolicho' - 长头(长>宽)
      'asym'    - 不对称(偏一侧)
      'flat'    - 后枕扁平(侧面)
      'with_body' - 含身体(测试crop)
      'off_center' - 头部偏移
      'dark_bg'   - 深色背景
      'with_coin' - 含硬币
    """
    h, w = 600, 600
    img = np.ones((h, w, 3), dtype=np.uint8)
    gt = {'variant': variant, 'img_size': (w, h)}

    # 背景
    if 'dark' in variant:
        bg_color = (40, 40, 50)
    else:
        bg_color = (200 + np.random.randint(-20, 20),
                     195 + np.random.randint(-20, 20),
                     190 + np.random.randint(-20, 20))
    img[:] = bg_color

    # 头部参数
    cx_base = w // 2
    cy_base = h // 2
    if 'off_center' in variant:
        cx_base += np.random.randint(-60, 60)
        cy_base += np.random.randint(-50, 50)

    # 椭圆轴
    if variant == 'brachy':
        rx, ry = 160, 140  # 宽>长 (扁头)
    elif variant == 'dolicho':
        rx, ry = 120, 170  # 长>宽 (长头)
    else:
        rx, ry = 145, 155  # 正常

    gt['ellipse'] = (cx_base, cy_base, rx, ry)
    gt['expected_ci'] = round(min(rx, ry) / max(rx, ry) * 100, 1)

    if variant == 'asym':
        rx += np.random.randint(10, 25)  # 一侧更宽

    # 画头部
    head_color = np.random.randint(170, 190), np.random.randint(135, 155), np.random.randint(125, 145)
    cv2.ellipse(img, (cx_base, cy_base), (rx, ry), 0, 0, 360, head_color, -1)

    # 头部mask (用于后续纹理)
    head_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(head_mask, (cx_base, cy_base), (rx, ry), 0, 0, 360, 255, -1)

    # 鼻尖标记 (俯视图: 顶部小突起)
    if variant not in ('flat',):
        nose_y = cy_base - ry + 5
        cv2.circle(img, (cx_base, nose_y), 10, tuple(c + 10 for c in head_color), -1)

    # 侧面图特有: 后脑勺扁平
    if variant == 'flat':
        # 把后脑勺一侧压扁 (左侧)
        flat_region = img[cy_base-ry:cy_base+ry, :cx_base]
        flat_mask = np.zeros_like(flat_region)
        flat_mask[:, :] = bg_color
        alpha = np.tile(np.linspace(0, 0.6, flat_region.shape[1]).reshape(1, -1, 1), (flat_region.shape[0], 1, 1))
        img[cy_base-ry:cy_base+ry, :cx_base] = (flat_region.astype(float) * (1-alpha) + bg_color * alpha).astype(np.uint8)

    # 头发纹理
    if np.random.random() > 0.3:  # 70% 有头发
        img = _add_hair_texture(img, head_mask)

    # 身体
    if 'with_body' in variant:
        body_y = cy_base + ry - 15
        img = _add_body(img, head_mask, body_y)
        gt['has_body'] = True
    else:
        gt['has_body'] = False

    # 硬币
    if 'coin' in variant:
        coin_cx = w - 70 + np.random.randint(-10, 10)
        coin_cy = h - 70 + np.random.randint(-10, 10)
        _draw_coin(img, coin_cx, coin_cy)
        gt['has_coin'] = True
        gt['coin_pos'] = (coin_cx, coin_cy)
    else:
        gt['has_coin'] = False

    return img, gt


# ============================================================
# PART 2: 分析执行器
# ============================================================

def run_top_analysis(img):
    """跑俯视图分析，返回 (result, annotated_image_base64)"""
    from head_analyzer import analyze_head_shape
    result = analyze_head_shape(img, reference_mm=25.0,
                                auto_detect_reference=True, use_reference=True,
                                guide_frame=True, age_months=3)
    if result.success:
        m = result.measurements
        _, buf = cv2.imencode('.jpg', result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return {
            'ci': m.ci, 'cvai': m.cvai, 'cva_mm': m.cva_mm,
            'head_length_mm': m.head_length_mm, 'head_width_mm': m.head_width_mm,
            'severity': m.severity.value, 'has_reference': any('参照物' in s for s in result.processing_steps),
        }, buf
    return None, None


def run_side_analysis(img):
    """跑侧面分析，返回 (result, annotated_image_bytes)"""
    from side_analyzer import analyze_side_profile
    result = analyze_side_profile(img)
    if result:
        contour_list = result.get('_head_contour')
        side_buf = None
        similarity = None
        if contour_list and len(contour_list) >= 20:
            contour = np.array(contour_list, dtype=np.int32).reshape(-1, 1, 2)
            sa = img.copy()
            cv2.drawContours(sa, [contour], -1, (0, 200, 80), 3)
            _, side_buf = cv2.imencode('.jpg', sa, [cv2.IMWRITE_JPEG_QUALITY, 85])
            # Snake对比 (与 app.py 相同路径: draw_comparison 只调一次)
            try:
                from standard_compare import draw_comparison
                _, comp_data = draw_comparison(img, contour, view='side', side_result=result)
                similarity = comp_data.get('similarity_score')
            except Exception:
                pass
        return {
            'flatness': result.get('posterior_flatness'),
            'category': result.get('flatness_category'),
            'head_length_px': result.get('head_length_px'),
            'similarity': similarity,
        }, side_buf
    return None, None


# ============================================================
# PART 3: 数值验证器
# ============================================================

def validate_top(gt, result):
    """验证俯视图分析结果"""
    issues = []
    actual_ci = result['ci']
    variant = gt.get('variant', 'normal')
    expected_ci = gt.get('expected_ci')

    # CI修复后恒有 头长≥头宽 → CI不可能超过100
    if actual_ci > 100.5:
        issues.append(f"CI超过100(长宽互换修复未生效): {actual_ci:.1f}")

    # 与合成椭圆的理论CI对比 (±12容差: SAM轮廓/头发噪声)
    if expected_ci and abs(actual_ci - expected_ci) > 12:
        issues.append(f"CI偏离理论值: 实际{actual_ci:.1f} 理论{expected_ci}")

    # 绝对尺寸仅在有硬币参照物时验证
    if gt.get('has_coin'):
        if result['head_length_mm'] < 50 or result['head_length_mm'] > 200:
            issues.append(f"头长异常: {result['head_length_mm']:.0f}mm")
        if result['head_width_mm'] < 40 or result['head_width_mm'] > 180:
            issues.append(f"头宽异常: {result['head_width_mm']:.0f}mm")

    # CVAI: 除'asym'外应基本对称
    if variant != 'asym' and result['cvai'] > 8:
        issues.append(f"对称头型CVAI过高: {result['cvai']:.1f}%")

    # 不管什么头型CVAI不应超过50
    if result['cvai'] > 50:
        issues.append(f"CVAI异常(>50%): {result['cvai']:.1f}%")

    return issues


def validate_side(gt, result):
    """验证侧面分析结果"""
    issues = []
    variant = gt.get('variant', 'normal')

    if variant == 'flat':
        # 扁平头型应该有较高 flatness
        if result['flatness'] < 0.1:
            issues.append(f"预期扁平但flatness过低: {result['flatness']:.3f}")
    else:
        # 正常头型 flatness 不应过高
        if result['flatness'] > 0.5:
            issues.append(f"正常头型flatness过高: {result['flatness']:.3f}")

    if result['head_length_px'] < 100:
        issues.append(f"头长像素过小: {result['head_length_px']:.0f}px")

    # Snake gap 相似度: 应在 0-100 范围内
    sim = result.get('similarity')
    if sim is not None and (sim < 0 or sim > 100):
        issues.append(f"相似度超出范围: {sim}")

    return issues


# ============================================================
# PART 4: MiMo 视觉验证器
# ============================================================

def mimo_verify(image_path, view_type, case_id):
    """用 MiMo 验证标注图质量"""
    prompt = f"这是测试用例{case_id}的{view_type}分析标注图。绿色曲线是检测到的头部轮廓。判断: (1)轮廓是否准确贴合头部(不偏大/偏小/偏移) (2)是否误包含背景或身体 (3)标注是否合理。用中文一句话回答,以OK或NG开头。"
    try:
        r = os.popen(f'mimo run "{prompt}" -f "{image_path}" -m xiaomi/mimo-v2.5 2>&1').read()
        return r.strip()
    except Exception as e:
        return f"ERROR: {e}"


def mimo_quick_check(image_path, view_type, case_id):
    """快速 MiMo 检查 (缩短prompt)"""
    try:
        r = os.popen(f'mimo run "绿线贴合头部吗？是否含身体？OK或NG中文回答" -f "{image_path}" -m xiaomi/mimo-v2.5 2>&1').read()
        return r.strip()
    except Exception:
        return "SKIP"


# ============================================================
# PART 5: 主测试编排
# ============================================================

VARIANTS_TOP = ['normal', 'brachy', 'dolicho', 'asym', 'off_center', 'dark_bg', 'with_coin']
VARIANTS_SIDE = ['normal', 'flat', 'with_body', 'off_center', 'dark_bg']


def run_all_tests(quick=False, with_mimo=True):
    """运行全部测试"""
    print("=" * 60)
    print("自动化视觉测试体系 v1.0")
    print("=" * 60)

    report = {'total': 0, 'passed': 0, 'failed': [], 'mimo_checks': []}
    test_images = []

    # ---- 俯视图测试 ----
    print("\n[俯视图测试]")
    from sam_detector import detect_head, create_guide_mask

    for i, variant in enumerate(VARIANTS_TOP):
        case_id = f"top_{variant}"
        img, gt = make_test_case(i, variant)
        test_images.append((case_id, img, gt, 'top'))

        # 直接SAM检测
        gm = create_guide_mask(*img.shape[:2])
        t0 = time.time()
        sam_out = detect_head(img, view='top', guide_mask=gm)
        dt = (time.time() - t0) * 1000

        if sam_out:
            mask, contour = sam_out
            area = mask.sum() / 255 / (img.shape[0] * img.shape[1]) * 100
        else:
            area = 0

        # 俯视图全流程分析
        r, ann_buf = run_top_analysis(img)
        if r:
            issues = validate_top(gt, r)
            status = 'PASS' if not issues else 'FAIL'
            report['total'] += 1
            if status == 'PASS':
                report['passed'] += 1
            else:
                report['failed'].append({'case': case_id, 'issues': issues, 'measurements': r})
            print(f"  [{status}] {case_id}: CI={r['ci']:.1f} CVAI={r['cvai']:.1f}% SAM_area={area:.1f}% ⏱{dt:.0f}ms")
            if issues:
                for iss in issues:
                    print(f"         ⚠ {iss}")
        else:
            print(f"  [FAIL] {case_id}: 分析失败")

    # ---- 侧面图测试 ----
    print("\n[侧面图测试]")
    for i, variant in enumerate(VARIANTS_SIDE):
        case_id = f"side_{variant}"
        img, gt = make_test_case(i + 10, variant)
        test_images.append((case_id, img, gt, 'side'))

        # 直接SAM检测
        gm = None  # 侧面不需要guide_mask
        t0 = time.time()
        sam_out = detect_head(img, view='side')
        dt = (time.time() - t0) * 1000

        if sam_out:
            mask, contour = sam_out
            area = mask.sum() / 255 / (img.shape[0] * img.shape[1]) * 100
        else:
            area = 0

        # 侧面全流程分析
        r, ann_buf = run_side_analysis(img)
        if r:
            issues = validate_side(gt, r)
            status = 'PASS' if not issues else 'FAIL'
            report['total'] += 1
            if status == 'PASS':
                report['passed'] += 1
            else:
                report['failed'].append({'case': case_id, 'issues': issues, 'measurements': r})
            print(f"  [{status}] {case_id}: flatness={r['flatness']:.3f} {r['category']} SAM_area={area:.1f}% ⏱{dt:.0f}ms")
            if issues:
                for iss in issues:
                    print(f"         ⚠ {iss}")

            # 保存标注图用于MiMo验证 (全部保存)
            if ann_buf is not None:
                ann_path = f"test_{case_id}_annotated.jpg"
                with open(ann_path, 'wb') as f:
                    f.write(ann_buf)
        else:
            print(f"  [FAIL] {case_id}: 分析失败")

    # ---- MiMo 视觉抽查 (只在非quick模式) ----
    if with_mimo and not quick and report['failed']:
        print("\n[MiMo 视觉抽查 - 失败用例]")
        for fail in report['failed'][:3]:  # 最多查3个
            case_id = fail['case']
            ann_path = f"test_{case_id}_annotated.jpg"
            if os.path.exists(ann_path):
                view = 'top' if 'top_' in case_id else 'side'
                verdict = mimo_quick_check(ann_path, view, case_id)
                print(f"  MiMo on {case_id}: {verdict[:200]}")
                report['mimo_checks'].append({'case': case_id, 'verdict': verdict})

    # ---- 报告 ----
    print(f"\n{'=' * 60}")
    passed = report['passed']
    total = report['total']
    pct = passed / total * 100 if total > 0 else 0
    print(f"结果: {passed}/{total} 通过 ({pct:.0f}%)")
    if report['failed']:
        print(f"失败用例:")
        for f in report['failed']:
            print(f"  {f['case']}: {', '.join(f['issues'])}")

    # 保存报告
    with open('test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: test_report.json")

    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='跳过MiMo验证')
    parser.add_argument('--no-mimo', action='store_true', help='不使用MiMo')
    args = parser.parse_args()

    report = run_all_tests(quick=args.quick, with_mimo=not args.no_mimo)
    sys.exit(0 if report['failed'] == [] else 1)
