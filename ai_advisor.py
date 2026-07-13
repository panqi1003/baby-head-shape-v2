"""
DeepSeek V4 Flash 智能分析 — 将测量数据转为个性化指导
"""

import json
import os
import requests
from pathlib import Path
from typing import Optional, Dict

# 尝试从 .env 文件加载
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"  # DeepSeek V4 Flash


def _build_prompt(
    top_measurements: Dict,
    side_measurements: Optional[Dict] = None,
    age_months: Optional[int] = None,
) -> str:
    """构建发送给 DeepSeek 的结构化 prompt"""

    m = top_measurements

    prompt = f"""你是一位有经验的育儿顾问。请根据以下婴儿头型测量数据，给出温暖、实用、不制造焦虑的分析和建议。

重要原则:
- 语气像朋友聊天，不要用医学诊断口吻
- 强调大多数情况通过日常调整都能改善
- 不确定的事情建议咨询医生，但不要吓唬家长
- 不要使用"危险信号""红旗征象""需警惕"等制造焦虑的表述

## 测量数据

### 俯视图 (头顶正上方)
- 头长 (前后径): {m.get('head_length_mm', 'N/A')} mm
- 头宽 (左右径): {m.get('head_width_mm', 'N/A')} mm
- 头围 (估算): {m.get('head_circumference_mm', 'N/A')} mm
- 头颅指数 (CI): {m.get('ci', 'N/A')} (常见范围: 75-85)
- 不对称指数 (CVAI): {m.get('cvai', 'N/A')}% (多数宝宝 <3.5%)
- 不对称值 (CVA): {m.get('cva_mm', 'N/A')} mm
- 算法分级: {m.get('severity', 'N/A')}
- 算法置信度: {m.get('confidence', 'N/A')}
"""

    if side_measurements:
        prompt += f"""
### 侧面图(左侧)
- 后枕部扁平度: {side_measurements.get('posterior_flatness', 'N/A')} (0=圆润, 1=偏平)
- 分类: {side_measurements.get('flatness_category', 'N/A')}
"""
        if side_measurements.get('right_flatness') is not None:
            prompt += f"""### 侧面图(右侧)
- 后枕部扁平度: {side_measurements.get('right_flatness', 'N/A')}
- 分类: {side_measurements.get('right_category', 'N/A')}
"""
    else:
        prompt += """
### 侧面图
未上传侧面照片。
"""

    if age_months:
        prompt += f"""
### 宝宝信息
- 月龄: {age_months} 个月
"""

    prompt += """
## 请按以下格式回复 (JSON):

```json
{
  "head_shape_type": "头型大致对称 / 略微偏头 / 略微扁头 / 偏头较明显 / 扁头较明显",
  "summary": "用一两句话通俗描述宝宝头型现在的情况，语气温暖",
  "daily_tips": ["日常可以这样做1", "日常可以这样做2", ...],
  "tummy_time_advice": "俯趴练习的具体建议 (频率、时长)",
  "next_step": "建议 4 周后再次拍照对比 / 建议找儿科医生看看 / 建议继续保持",
  "explanation": "用150-200字通俗解释宝宝头型现状、可能原因、接下来怎么做。语气像朋友聊天，温暖专业。强调婴幼儿头骨可塑性强，多数通过日常调整能改善。不确定的事情建议咨询医生但不要说得很严重。"
}
```

请确保回复是合法的 JSON，不要有其他文字。"""

    return prompt


def analyze_with_deepseek(
    top_measurements: Dict,
    side_measurements: Optional[Dict] = None,
    age_months: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Dict:
    """
    调用 DeepSeek V4 Flash 进行智能分析。

    参数:
      top_measurements: 俯视图测量数据
      side_measurements: 侧面图测量数据 (可选)
      age_months: 宝宝月龄 (可选)
      api_key: DeepSeek API Key (可选, 默认用环境变量)

    返回:
      DeepSeek 分析结果 dict, 或 {"error": ...}
    """
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        return {
            "error": "未配置 DEEPSEEK_API_KEY。请设置环境变量或在调用时传入。",
            "fallback": True,
        }

    prompt = _build_prompt(top_measurements, side_measurements, age_months)

    try:
        resp = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "你是一位有经验的育儿顾问，请用 JSON 格式回复。语气像朋友聊天，温暖实用，不制造焦虑，不确定的事建议咨询医生但不要说得很严重。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
            },
            timeout=30,
        )

        if resp.status_code != 200:
            return {
                "error": f"DeepSeek API 返回错误: {resp.status_code}",
                "detail": resp.text[:500],
                "fallback": True,
            }

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # 提取 JSON (可能被 markdown code block 包裹)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        return result

    except requests.exceptions.Timeout:
        return {"error": "DeepSeek API 超时", "fallback": True}
    except json.JSONDecodeError:
        return {
            "error": "DeepSeek 返回格式异常",
            "raw": content[:500] if 'content' in dir() else "",
            "fallback": True,
        }
    except Exception as e:
        return {"error": str(e), "fallback": True}


def generate_fallback_advice(m: Dict, side: Optional[Dict] = None) -> Dict:
    """
    当 DeepSeek 不可用时的本地回退建议 (升级版)。
    基于 CI/CVAI + 月龄的规则引擎。
    """
    ci = m.get("ci", 80)
    cvai = m.get("cvai", 2)
    severity = m.get("severity", "正常")

    # 分型 (温和表述)
    if cvai > 6.25:
        head_type = "偏头较明显"
    elif ci > 90:
        head_type = "扁头较明显"
    elif ci < 72:
        head_type = "头型偏长"
    elif cvai > 3.5:
        head_type = "略微偏头"
    elif ci > 85:
        head_type = "略微扁头"
    else:
        head_type = "头型大致对称"

    # 后枕扁平度
    flatness = "未知"
    if side:
        flatness = side.get("flatness_category", "未知")

    return {
        "head_shape_type": head_type,
        "summary": f"宝宝头型{head_type}，头颅指数 CI={ci:.0f}（常见范围75-85）。婴幼儿头骨可塑性强，通过日常调整多数都能改善。",
        "daily_tips": [
            "每次睡觉时交替宝宝头朝向床头/床尾",
            "喂奶时交替左右手臂",
            "在宝宝不常转头的一侧放有趣的玩具或光源，引导主动转头",
            "减少安全座椅、摇椅等约束装置的连续使用时间",
            "调整婴儿床位置，让宝宝需要转头才能看到房间里的活动",
        ],
        "tummy_time_advice": "每天累计30-60分钟，分5-6次进行。在宝宝清醒、情绪好时进行，家长在旁边看护。",
        "next_step": (
            "建议找儿科医生看看" if severity in ("中度", "重度")
            else "建议 4 周后再次拍照对比，观察变化趋势"
        ),
        "explanation": f"亲爱的家长，宝宝头型看起来{head_type}。这在婴儿中非常常见，通常是因为宝宝长时间保持同一睡姿造成的，不是什么大问题。好消息是，婴幼儿头骨很软、可塑性很强，通过增加俯趴时间、多变换睡姿和抱姿等简单的日常调整，大多数宝宝的头型都能逐步改善。建议坚持 2-4 周后再次拍照对比。日常多观察宝宝的转头是否灵活对称。如果有不确定的地方，带宝宝体检时可以顺便问问儿科医生。",
    }
