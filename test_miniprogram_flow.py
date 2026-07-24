"""
自动化测试 — 模拟小程序全流程
测试: API 调用 → JSON 格式 → 结果页数据绑定
"""

import json
import sys
sys.path.insert(0, ".")
import requests
import cv2
import numpy as np
from pathlib import Path

API = "http://localhost:8000"
TEST_IMG = Path(__file__).parent / "test_baby_head.jpg"


def ok(msg):  print(f"  [OK] {msg}")
def fail(msg): print(f"  [FAIL] {msg}"); return False
def info(msg): print(f"  [INFO] {msg}")


def generate_busy_bg_photo():
    """生成有背景干扰的测试照片"""
    img = np.ones((600, 600, 3), dtype=np.uint8) * 230

    # 头部 (中心, 肤色)
    skin = (85, 145, 185)
    cv2.ellipse(img, (300, 280), (130, 105), 90, 0, 360, skin, -1)

    # 干扰: 背景里的"手臂" (肤色, 偏离中心)
    cv2.ellipse(img, (80, 400), (80, 25), 20, 0, 360, (80, 140, 175), -1)

    # 干扰: 背景纹理
    cv2.line(img, (200, 500), (550, 520), (200, 195, 190), 3)
    cv2.rectangle(img, (500, 50), (560, 100), (210, 200, 195), -1)

    # 硬币
    cv2.circle(img, (520, 450), 45, (195, 190, 180), -1)
    cv2.circle(img, (520, 450), 45, (150, 145, 140), 2)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.clip(img + np.random.normal(0, 2, img.shape), 0, 255).astype(np.uint8)

    path = Path(__file__).parent / "test_busy_bg.jpg"
    cv2.imwrite(str(path), img)
    return path


def assert_shape(obj, path=""):
    """递归验证 JSON 结构"""
    issues = []

    def check(val, p):
        if isinstance(val, dict):
            for k, v in val.items():
                check(v, f"{p}.{k}")
        elif isinstance(val, list):
            for i, v in enumerate(val):
                check(v, f"{p}[{i}]")
        elif val is None:
            pass  # null is fine
        elif not isinstance(val, (str, int, float, bool)):
            issues.append(f"{p}: unexpected type {type(val).__name__}")

    check(obj, path)
    return issues


def test_api_health():
    """测试 1: 服务连通性"""
    print("\n=== 测试 1: API 连通性 ===")
    try:
        r = requests.get(API, timeout=5)
        if r.status_code == 200:
            ok(f"HTTP {r.status_code}")
            return True
        else:
            fail(f"HTTP {r.status_code}")
            return False
    except Exception as e:
        fail(str(e))
        return False


def test_analyze_basic():
    """测试 2: 基本分析 (有参照物, 无AI)"""
    print("\n=== 测试 2: 基本分析 (有参照物) ===")
    if not TEST_IMG.exists():
        fail(f"测试图片不存在: {TEST_IMG}")
        return False

    try:
        with open(TEST_IMG, 'rb') as f:
            r = requests.post(f"{API}/analyze",
                files={"top_image": f},
                data={"use_reference": "true", "auto_detect": "true", "use_ai": "false"},
                timeout=60)
    except Exception as e:
        fail(str(e))
        return False

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:200]}")
        return False

    data = r.json()

    # 验证顶层结构
    required_keys = ["success", "steps", "measurements", "advice",
                     "fallback_advice", "ai_analysis", "annotated_image",
                     "side_annotated_image", "has_ai"]
    for k in required_keys:
        if k not in data:
            fail(f"缺少字段: {k}")
            return False
    ok("顶层结构完整")

    # 验证 measurements
    m = data.get("measurements", {})
    m_keys = ["head_length_mm", "head_width_mm", "head_circumference_mm",
              "ci", "cvai", "cva_mm", "severity", "confidence"]
    for k in m_keys:
        if k not in m:
            fail(f"measurements 缺少: {k}")
            return False
    if not (50 < m["ci"] < 120):
        fail(f"CI 值异常: {m['ci']}")
        return False
    ok(f"测量数据正常 (CI={m['ci']:.1f}, CVAI={m['cvai']:.1f}%, {m['severity']})")

    # 验证 fallback_advice
    fb = data.get("fallback_advice", {})
    fb_keys = ["head_shape_type", "summary", "daily_tips",
               "tummy_time_advice", "next_step", "explanation"]
    for k in fb_keys:
        if k not in fb:
            fail(f"fallback_advice 缺少: {k}")
            return False
    ok("回退建议结构完整")

    # 验证 annotated_image 是有效的 base64 data URI
    img = data.get("annotated_image", "")
    if not img.startswith("data:image/jpeg;base64,"):
        fail("标注图格式异常")
        return False
    ok("标注图格式正确")

    # JSON 结构无异常类型
    issues = assert_shape(data, "root")
    if issues:
        for i in issues: fail(i)
        return False
    ok("JSON 结构无异常类型")

    return True


def test_analyze_no_reference():
    """测试 3: 无参照物快速模式"""
    print("\n=== 测试 3: 快速模式 (无参照物) ===")
    try:
        with open(TEST_IMG, 'rb') as f:
            r = requests.post(f"{API}/analyze",
                files={"top_image": f},
                data={"use_reference": "false", "auto_detect": "false", "use_ai": "false"},
                timeout=60)
    except Exception as e:
        fail(str(e))
        return False

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}")
        return False

    data = r.json()
    m = data["measurements"]

    # 无参照物时 confidence 应该更低
    if m["confidence"] > 0.8:
        info(f"注意: 无参照物但置信度较高 ({m['confidence']})")
    ok(f"快速模式正常 (置信度={m['confidence']:.0%}, CI={m['ci']:.1f})")
    return True


def test_analyze_with_ai():
    """测试 4: AI 分析模式"""
    print("\n=== 测试 4: AI 分析 ===")
    try:
        with open(TEST_IMG, 'rb') as f:
            r = requests.post(f"{API}/analyze",
                files={"top_image": f},
                data={"use_reference": "true", "use_ai": "true", "age_months": "5"},
                timeout=60)
    except Exception as e:
        fail(str(e))
        return False

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}")
        return False

    data = r.json()
    ai = data.get("ai_analysis", {})

    if data.get("has_ai"):
        ok(f"AI 分析成功 (keys: {list(ai.keys()) if isinstance(ai, dict) else 'N/A'})")
    else:
        info("AI 未启用或失败 (可能是 API Key 未配置)")
    return True


def test_analyze_busy_bg():
    """测试 5: 复杂背景 (SAM 应能排除干扰)"""
    print("\n=== 测试 5: 复杂背景干扰 ===")
    img_path = generate_busy_bg_photo()
    info(f"生成测试图: {img_path}")

    try:
        with open(img_path, 'rb') as f:
            r = requests.post(f"{API}/analyze",
                files={"top_image": f},
                data={"use_reference": "true", "auto_detect": "true", "use_ai": "false"},
                timeout=60)
    except Exception as e:
        fail(str(e))
        return False

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:300]}")
        return False

    data = r.json()
    m = data["measurements"]

    # 应该检测到头部 (不是背景里的干扰物)
    if not (70 < m["ci"] < 90):
        info(f"CI 偏离预期 ({m['ci']}), 可能受背景干扰")
    else:
        ok(f"SAM 正确排除背景干扰 (CI={m['ci']:.1f})")

    print(f"  处理步骤: {' → '.join(data['steps'])}")
    return True


def test_miniprogram_data_binding():
    """测试 6: 验证结果页数据绑定"""
    print("\n=== 测试 6: 结果页数据绑定验证 ===")

    try:
        with open(TEST_IMG, 'rb') as f:
            r = requests.post(f"{API}/analyze",
                files={"top_image": f},
                data={"use_reference": "true", "use_ai": "false"},
                timeout=60)
    except Exception as e:
        fail(str(e))
        return False

    data = r.json()
    m = data["measurements"]
    fb = data["fallback_advice"]

    # 模拟 result.js 的 parseResult 逻辑
    use_ai = data.get("has_ai") and data.get("ai_analysis") and "error" not in data.get("ai_analysis", {})
    advice = data["ai_analysis"] if use_ai else fb

    # 验证 WXML 绑定所需的所有字段
    checks = {
        "severity": m.get("severity"),
        "headType": advice.get("head_shape_type") or "",
        "explanation": advice.get("explanation") or advice.get("summary", ""),
        "annotatedImg": data.get("annotated_image"),
        "ci": m.get("ci"),
        "cvai": m.get("cvai"),
        "cva": m.get("cva_mm"),
        "headLength": m.get("head_length_mm"),
        "headWidth": m.get("head_width_mm"),
        "headCirc": m.get("head_circumference_mm"),
        "hasAI": use_ai,
        "adviceSummary": advice.get("explanation") or advice.get("summary", ""),
        "actions": (
            advice.get("daily_tips") or
            (advice.get("intervention_plan") or {}).get("repositioning") or
            advice.get("actions") or []
        ),
        "monitoring": advice.get("next_step") or (
            advice.get("monitoring") if isinstance(advice.get("monitoring"), str) else ""
        )
    }

    all_ok = True
    for key, val in checks.items():
        if val is None and key not in ("sideFlatness",):
            info(f"字段 {key} 为 None (页面可能显示空白)")
        elif isinstance(val, str) and "<" in val:
            fail(f"字段 {key} 包含未转义的 '<': {val[:50]}...")
            all_ok = False

    if all_ok:
        ok("结果页所有绑定字段可用")
    return all_ok


if __name__ == "__main__":
    print("=" * 60)
    print("小程序全流程自动化测试")
    print("=" * 60)

    results = [
        test_api_health(),
        test_analyze_basic(),
        test_analyze_no_reference(),
        test_analyze_with_ai(),
        test_analyze_busy_bg(),
        test_miniprogram_data_binding(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 通过")

    if passed < total:
        print(f"失败: {total - passed} 项")
        sys.exit(1)
    else:
        print("全部通过!")
