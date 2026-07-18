"""
自测脚本 — 覆盖 SAM 检测 → 分析全流程
用法: python self_test.py
"""
import cv2, numpy as np, os, sys, time, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def make_top_synthetic():
    """合成俯视图: 椭圆头部 + 鼻尖标记 + 硬币"""
    h, w = 500, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 210
    # 头部椭圆 (略长, 模拟前后径>左右径)
    cv2.ellipse(img, (w//2, h//2), (145, 155), 0, 0, 360, (175, 145, 130), -1)
    # 鼻尖 (上方小突起)
    cv2.circle(img, (w//2, h//2-148), 12, (180, 150, 135), -1)
    # 硬币 (右下角, 直径25mm = ~30px in this scale)
    cv2.circle(img, (w-60, h-60), 28, (180, 180, 180), -1)
    cv2.circle(img, (w-60, h-60), 28, (140, 140, 140), 2)
    return img


def make_side_synthetic():
    """合成侧面图: 水平椭圆头部 + 鼻子突起 + 深色头发"""
    h, w = 480, 640
    img = np.ones((h, w, 3), dtype=np.uint8) * 210
    # 头部水平椭圆
    cv2.ellipse(img, (w//2, h//2), (170, 125), 0, 0, 360, (175, 145, 130), -1)
    # 鼻子突起 (右侧)
    nose = np.array([[w//2+155, h//2-20], [w//2+190, h//2+10], [w//2+155, h//2+35]], np.int32)
    cv2.fillConvexPoly(img, nose, (180, 150, 135))
    # 后脑勺稍平 (左侧)
    cv2.ellipse(img, (w//2-150, h//2-5), (30, 85), 0, -50, 50, (175, 145, 130), -1)
    return img


def test_sam_top():
    """测试俯视图 SAM 检测"""
    from sam_detector import detect_head
    img = make_top_synthetic()
    result = detect_head(img, view='top')
    if result is None:
        raise RuntimeError("俯视图 SAM 检测失败!")
    mask, contour = result
    h, w = mask.shape
    area_pct = mask.sum() / 255 / (h*w) * 100
    if not (10 < area_pct < 60):
        raise RuntimeError(f"俯视图 mask 面积异常: {area_pct:.1f}%")
    print(f"  [PASS] 俯视图 SAM: area={area_pct:.1f}%, contour_pts={len(contour)}")
    return result


def test_sam_side():
    """测试侧面图 SAM 检测"""
    from sam_detector import detect_head
    img = make_side_synthetic()
    result = detect_head(img, view='side')
    if result is None:
        raise RuntimeError("侧面图 SAM 检测失败!")
    mask, contour = result
    h, w = mask.shape
    area_pct = mask.sum() / 255 / (h*w) * 100
    if not 10 < area_pct < 60:
        raise RuntimeError(f"侧面图 mask 面积异常: {area_pct:.1f}%")
    # 检查是否覆盖左右两侧 (span check)
    h, w = img.shape[:2]
    left_frac = mask[:, :w//2].sum() / max(mask.sum(), 1)
    if not (0.15 < left_frac < 0.85):
        raise RuntimeError(f"侧面图 mask 偏一侧: left_frac={left_frac:.2f}")
    print(f"  [PASS] 侧面图 SAM: area={area_pct:.1f}%, left_frac={left_frac:.2f}, contour_pts={len(contour)}")
    return result


def test_head_analyzer():
    """测试俯视图分析全流程"""
    from head_analyzer import analyze_head_shape
    img = make_top_synthetic()
    result = analyze_head_shape(img, reference_mm=25.0,
                                auto_detect_reference=True, use_reference=True,
                                guide_frame=True)
    if not result.success: raise RuntimeError(f"俯视图分析失败: {result.error_message}")
    m = result.measurements
    # 验证基本合理性
    if not (60 < m.head_length_mm < 200): raise RuntimeError(f"头长异常: {m.head_length_mm}")
    if not (50 < m.head_width_mm < 180): raise RuntimeError(f"头宽异常: {m.head_width_mm}")
    if not (70 < m.ci < 125): raise RuntimeError(f"CI 异常: {m.ci}")
    print(f"  [PASS] 俯视图分析: 头长={m.head_length_mm:.0f}mm 头宽={m.head_width_mm:.0f}mm "
          f"CI={m.ci:.1f} CVAI={m.cvai:.1f}% severity={m.severity.value}")
    return result


def test_side_analyzer():
    """测试侧面图分析全流程"""
    from side_analyzer import analyze_side_profile
    img = make_side_synthetic()
    result = analyze_side_profile(img)
    if result is None:
        raise RuntimeError("侧面图分析失败!")
    if 'posterior_flatness' not in result: raise RuntimeError('result缺少posterior_flatness字段')
    if 'flatness_category' not in result: raise RuntimeError('result缺少flatness_category字段')
    flat = result['posterior_flatness']
    cat = result['flatness_category']
    if not (0 <= flat <= 1): raise RuntimeError(f"扁平度异常: {flat}")
    print(f"  [PASS] 侧面图分析: flatness={flat:.3f} category={cat} "
          f"head_length={result.get('head_length_px', '?')}px")
    return result


def test_ai_advisor():
    """测试 AI 分析 (规则引擎回退)"""
    from ai_advisor import generate_fallback_advice, analyze_with_deepseek
    top_data = {"ci": 78.5, "cvai": 2.1, "cva_mm": 2.3, "severity": "normal",
                "head_length_mm": 135, "head_width_mm": 106}
    side_data = {"posterior_flatness": 0.12, "flatness_category": "正常圆润"}

    # 测试回退逻辑 (不调 API, 防超时)
    fb = generate_fallback_advice(top_data, side_data)
    if not fb: raise RuntimeError("回退建议生成失败!")
    if not ('explanation' in fb or 'summary' in fb): raise RuntimeError(f"回退缺少 explanation/summary: {fb.keys()}")
    if 'daily_tips' not in fb: raise RuntimeError("回退缺少 daily_tips")
    print(f"  [PASS] AI 回退: summary={fb.get('summary', fb.get('explanation', ''))[:60]}...")
    return fb


def test_full_pipeline():
    """端到端测试: 模拟小程序三个阶段的调用"""
    print("\n[端到端流程模拟]")
    from head_analyzer import analyze_head_shape
    from side_analyzer import analyze_side_profile
    from ai_advisor import generate_fallback_advice

    # 阶段1: 俯视图
    top_img = make_top_synthetic()
    top_result = analyze_head_shape(top_img, reference_mm=25.0,
                                    auto_detect_reference=True, use_reference=True,
                                    guide_frame=True)
    if not top_result.success: raise RuntimeError(f"阶段1失败: {top_result.error_message}")
    m = top_result.measurements
    top_data = {"ci": m.ci, "cvai": m.cvai, "cva_mm": m.cva_mm, "severity": m.severity.value,
                "head_length_mm": m.head_length_mm, "head_width_mm": m.head_width_mm}
    print(f"  阶段1 [PASS]: CI={m.ci:.1f} CVAI={m.cvai:.1f}%  severity={m.severity.value}")

    # 阶段2: 左右侧面
    side_img = make_side_synthetic()
    side_result = analyze_side_profile(side_img)
    if side_result is None: raise RuntimeError("阶段2失败")
    print(f"  阶段2 [PASS]: flatness={side_result['posterior_flatness']:.3f} {side_result['flatness_category']}")

    # 阶段3: 综合分析
    fb = generate_fallback_advice(top_data, side_result)
    if not (fb and 'daily_tips' in fb): raise RuntimeError("阶段3失败")
    print(f"  阶段3 [PASS]: tips数={len(fb.get('daily_tips',[]))} next_step={fb.get('next_step','')}")

    print("  [PASS] 端到端流程通过!")


def test_edge_cases():
    """边界测试: 极小图/纯黑图应优雅失败不崩溃"""
    from head_analyzer import analyze_head_shape
    from side_analyzer import analyze_side_profile

    results = []

    # 极小图
    tiny = np.ones((50, 50, 3), dtype=np.uint8) * 200
    r = analyze_head_shape(tiny, use_reference=False, auto_detect_reference=False)
    results.append(("极小俯视图不崩溃", not r.success))

    # 纯黑图
    black = np.zeros((500, 500, 3), dtype=np.uint8)
    r2 = analyze_head_shape(black, use_reference=False, auto_detect_reference=False)
    results.append(("纯黑俯视图不崩溃", not r2.success))

    # 侧面极小
    r3 = analyze_side_profile(tiny)
    results.append(("极小侧面图不崩溃", r3 is None))

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] 边界测试: {name}")

    return results


if __name__ == '__main__':
    print("=" * 55)
    print("婴儿头型分析 - 自动化自测")
    print("=" * 55)

    total, passed = 0, 0

    tests = [
        ("俯视图 SAM 检测", test_sam_top),
        ("侧面图 SAM 检测", test_sam_side),
        ("俯视图分析器", test_head_analyzer),
        ("侧面图分析器", test_side_analyzer),
        ("AI 回退建议", test_ai_advisor),
        ("端到端流程", test_full_pipeline),
        ("边界测试", test_edge_cases),
    ]

    for name, fn in tests:
        total += 1
        try:
            print(f"\n[{name}]")
            result = fn()
            if name == "边界测试":
                ok = all(r[1] for r in result)
                if not ok:
                    raise RuntimeError("边界测试未全部通过")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] 失败: {e}")

    print(f"\n{'='*55}")
    print(f"结果: {passed}/{total} 通过" + (" [PASS]" if passed == total else " [FAIL]"))
    sys.exit(0 if passed == total else 1)
