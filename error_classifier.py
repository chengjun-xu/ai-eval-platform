"""
Bad Case 归因分类模块
=====================
自动将评测中的错误回答分类到具体失败模式，并生成可行动的分析与改进建议。

分类体系（对齐 JD "系统化归因"）：
1. **指令遵循失败** — 模型未按要求格式输出（如选择题没输出字母）
2. **知识缺失/遗忘** — 回答有明显事实错误或缺失关键知识
3. **逻辑断裂** — 推理步骤前后矛盾或跳跃
4. **证据不一致** — 回答与提供的证据上下文矛盾
5. **幻觉模式** — 生成了不存在的引用、数据、或虚构事实
6. **拒答策略不当** — 模型过度拒绝回答应当能回答的问题
7. **工具调用失败** — 工具/函数调用参数错误（预留）
8. **其他/未分类**

面试亮点：展示"将诊断结论转化为训练策略"的能力。
"""
import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 自定义分类支持 — 用户可编辑 custom_categories.json 添加自有错误分类
# 文件路径: data/custom_categories.json
# ---------------------------------------------------------------------------

CUSTOM_CATEGORIES_FILE = Path(__file__).parent / "data" / "custom_categories.json"

# 加载失败时返回空
_loaded_custom = None


def _load_custom_categories() -> dict:
    """加载用户自定义分类，支持 patterns、severity、training_suggestion"""
    global _loaded_custom
    if _loaded_custom is not None:
        return _loaded_custom
    path = CUSTOM_CATEGORIES_FILE
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                _loaded_custom = data
                return data
        except (json.JSONDecodeError, IOError):
            pass
    _loaded_custom = {}
    return {}


def _create_default_custom_categories():
    """如果文件不存在，创建默认模板"""
    if not CUSTOM_CATEGORIES_FILE.exists():
        CUSTOM_CATEGORIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "custom_categories": {
                "sample_custom": {
                    "label": "自定义示例",
                    "emoji": "🧪",
                    "description": "这是一个自定义分类示例",
                    "severity": "中",
                    "patterns": ["示例关键词", "sample_keyword"],
                    "benchmark_types": ["mmlu", "open_ended"],
                    "training_suggestion": "根据自定义类别的归因结果进行针对性微调",
                }
            },
            "_help": {
                "说明": "在此文件中添加自定义错误分类。每个 key 作为分类 ID，必须包含：label(显示名), emoji(图标), description(描述), severity(严重度: 严重/高/中/低), patterns(正则或关键词列表), benchmark_types(适用评测类型列表), training_suggestion(训练建议)。patterns 中可以使用正则表达式。"
            }
        }
        with open(CUSTOM_CATEGORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)


_create_default_custom_categories()


def _match_custom_categories(predicted: str, benchmark_type: str) -> list[dict]:
    """对用户自定义分类进行匹配检测

    返回匹配上的自定义分类列表
    """
    custom = _load_custom_categories()
    matches = []
    predicted_lower = (predicted or "").lower()
    for cat_id, cat_info in custom.get("custom_categories", {}).items():
        patterns = cat_info.get("patterns", [])
        btypes = cat_info.get("benchmark_types", [])
        if btypes and benchmark_type not in btypes:
            continue
        for pat in patterns:
            try:
                if re.search(pat, predicted_lower, re.IGNORECASE):
                    matches.append({
                        "category": f"custom:{cat_id}",
                        "label": cat_info.get("label", cat_id),
                        "emoji": cat_info.get("emoji", "⚙️"),
                        "severity": cat_info.get("severity", "中"),
                        "detail": cat_info.get("description", ""),
                        "training_suggestion": cat_info.get(
                            "training_suggestion",
                            "请根据自定义分类的归因结果进行针对性微调",
                        ),
                    })
                    break
            except re.error:
                pass
    return matches


def get_all_categories() -> dict:
    """获取所有分类（内置 + 自定义），前端展示用"""
    all_cats = dict(ERROR_CATEGORIES)
    custom = _load_custom_categories()
    for cat_id, cat_info in custom.get("custom_categories", {}).items():
        all_cats[f"custom:{cat_id}"] = {
            "label": cat_info.get("label", cat_id),
            "emoji": cat_info.get("emoji", "⚙️"),
            "description": cat_info.get("description", ""),
            "severity": cat_info.get("severity", "中"),
            "training_suggestion": cat_info.get("training_suggestion", ""),
            "is_custom": True,
        }
    return all_cats


# ---------------------------------------------------------------------------
# 错误分类常量
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = {
    "instruction_following": {
        "label": "指令遵循失败",
        "emoji": "📋",
        "description": "模型未按要求格式输出（如选择题没输出字母、代码没输出函数）",
        "severity": "中",
        "training_suggestion": "SFT 阶段补充格式约束数据，在 prompt 中加入 few-shot 示例",
    },
    "knowledge_gap": {
        "label": "知识缺失/遗忘",
        "emoji": "📚",
        "description": "回答有明显事实错误或关键知识缺失",
        "severity": "高",
        "training_suggestion": "补充领域知识 SFT 数据，使用 RAG 增强；考虑课程学习增加难度",
    },
    "logic_break": {
        "label": "逻辑断裂",
        "emoji": "🔗",
        "description": "推理步骤前后矛盾、遗漏关键环节或跳跃结论",
        "severity": "高",
        "training_suggestion": "构造 CoT 训练数据，强化中间推理步骤监督",
    },
    "evidence_inconsistency": {
        "label": "证据不一致",
        "emoji": "🔍",
        "description": "回答与提供的上下文/证据矛盾",
        "severity": "高",
        "training_suggestion": "RAG 场景下补充证据约束数据，训练模型严格基于上下文回答",
    },
    "hallucination": {
        "label": "幻觉模式",
        "emoji": "👻",
        "description": "编造不存在的引用、数据、统计或虚构事实",
        "severity": "严重",
        "training_suggestion": "构造偏好数据（DPO），对幻觉回答给予负反馈；引入事实性奖励模型",
    },
    "refusal": {
        "label": "拒答策略不当",
        "emoji": "🚫",
        "description": "模型不当拒绝回答本应回答的问题",
        "severity": "中",
        "training_suggestion": "校准安全对齐边界，区分真实不安全与过度拒绝场景",
    },
    "tool_call_failure": {
        "label": "工具调用失败",
        "emoji": "🔧",
        "description": "工具/函数调用参数格式错误或调用策略不当",
        "severity": "高",
        "training_suggestion": "Agent 场景下增加工具调用训练数据，强化格式约束",
    },
    "uncategorized": {
        "label": "未分类",
        "emoji": "❓",
        "description": "无法自动归类的错误",
        "severity": "低",
        "training_suggestion": "建议人工审查",
    },
}

# ---------------------------------------------------------------------------
# 关键词信号检测
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    r"作为.*AI|作为.*助手", r"我(是|只)(一个)?人工智能",
    r"我不能|我无法|我做不到|抱歉.*不能|对不起.*无法",
    r"请咨询|请寻求|建议您(咨询|联系|前往)",
    r"出于安全|出于伦理|出于道德|出于法律",
    r"这不(是|属于)医学|我不是医生|这不是医疗建议",
    r"我没有足够的|我缺乏|我不确定|我没有把握",
    r"sorry.*can'?t|I cannot|I'm not able|I'm an AI",
    r"consult.*doctor|seek.*professional|for safety",
]

_INSTRUCTION_FAILURE_PATTERNS = [
    # 选择题期望输出字母，但模型输出长篇文字
    # 代码期望输出实现，但模型输出非代码
    r"^[^A-Da-d\s]{10,}",  # 非字母开头且内容>10字符（选择题）
]

_HALLUCINATION_PATTERNS = [
    r"根据[我研究|我查询|我查阅]",  # 未联网却声称查了资料
    r"引用数据\d{4}|统计(显示|表明)\d{4}",
    r"据[某][研究|调查|报告]",
    r"最新研究|最新数据|最新统计",
]

_LOGIC_BREAK_PATTERNS = [
    r"但.*所以|然而.*因此",  # 逻辑跳跃
    r"因为.*所以.*因为",  # 循环论证
]

_KNOWLEDGE_GAP_MARKERS = [
    "不知道", "不清楚", "不太了解",
    "我没有相关信息", "我没有学过",
    "这是错误的", "不对", "你错了",
]

# ---------------------------------------------------------------------------
# 核心分类器
# ---------------------------------------------------------------------------

def classify_error(question: str, expected: str, predicted: str,
                   benchmark_type: str = "mmlu",
                   judge_reason: str = "") -> dict:
    """对单个 Bad Case 进行归因分类

    Args:
        question: 题目原文
        expected: 参考答案/期望输出
        predicted: 模型实际输出
        benchmark_type: mmlu / gsm8k / humaneval / open_ended / medical
        judge_reason: LLM-as-Judge 的评语（仅 open_ended 有）

    Returns:
        {
            "category": str (ERROR_CATEGORIES 的 key),
            "signals": [str],  # 检测到的信号
            "confidence": float (0~1),
            "detail": str,     # 简短归因说明
        }
    """
    predicted_lower = (predicted or "").lower()
    expected_lower = (expected or "").lower()
    question_lower = (question or "").lower()
    combined = f"{predicted_lower} {judge_reason.lower()}"

    signals = []

    # 1. 拒答检测
    for pattern in _REFUSAL_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            signals.append(f"拒答信号: {pattern}")

    # 2. 指令遵循检测
    if benchmark_type in ("mmlu", "medical", "medical_custom"):
        # 选择题：期望是 A/B/C/D，但模型没输出字母
        if expected_lower.strip() in "abcd":
            if predicted_lower.strip() not in "abcd" and not re.search(r"\b[A-D]\b", predicted):
                if len(predicted) > 3:
                    signals.append("指令遵循失败: 未按要求输出选项字母")

    if benchmark_type == "humaneval":
        # 代码题：期望输出代码，但模型输出了解释
        if not re.search(r"```|def |class |function|import", predicted):
            signals.append("指令遵循失败: 未输出可执行代码")

    if benchmark_type == "gsm8k":
        # 数学题：期望输出数值
        if not re.search(r"\d+", predicted):
            signals.append("指令遵循失败: 未输出数值答案")

    # 3. 幻觉检测
    for pattern in _HALLUCINATION_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            signals.append(f"幻觉信号: 编造引用或数据")

    # 4. 知识缺失 — 模型承认不知道
    for marker in _KNOWLEDGE_GAP_MARKERS:
        if marker in predicted_lower:
            signals.append(f"知识缺失信号: 模型承认不知道")
            break

    # 5. 逻辑断裂（GSM8K 特有）
    if benchmark_type == "gsm8k":
        for pattern in _LOGIC_BREAK_PATTERNS:
            if re.search(pattern, predicted_lower):
                signals.append(f"逻辑断裂信号: 推理步骤跳跃")
                break

    # 6. 如果 LLM-as-Judge 给出了评语，从中提取信号
    if judge_reason:
        if any(w in judge_reason for w in ["幻觉", "虚构", "编造", "hallucin", "make up"]):
            signals.append("Judge 判定: 幻觉")
        if any(w in judge_reason for w in ["拒绝", "refus", "拒绝回答", "不能回答"]):
            signals.append("Judge 判定: 拒答")
        if any(w in judge_reason for w in ["不完整", "缺失", "遗漏", "incomplete", "miss"]):
            signals.append("Judge 判定: 回答不完整")
        if any(w in judge_reason for w in ["错误", "不准", "factual", "错误事实"]):
            signals.append("Judge 判定: 事实错误")

    # 7. 自定义分类检测
    custom_matches = _match_custom_categories(predicted, benchmark_type)

    # 8. 最终分类决策
    category = _decide_category(signals, benchmark_type, len(predicted or ""), custom_matches)

    result = {
        "category": category,
        "label": ERROR_CATEGORIES[category]["label"],
        "emoji": ERROR_CATEGORIES[category]["emoji"],
        "severity": ERROR_CATEGORIES[category]["severity"],
        "signals": signals[:3],
        "confidence": min(0.5 + 0.15 * len(signals), 0.95) if signals else 0.3,
        "detail": _generate_detail(category, signals, benchmark_type),
        "training_suggestion": ERROR_CATEGORIES[category]["training_suggestion"],
        "custom_matches": custom_matches,
    }
    return result


def _decide_category(signals: list[str], benchmark_type: str,
                     predicted_len: int,
                     custom_matches: list[dict] | None = None) -> str:
    """综合信号做最终分类决策"""
    signal_text = " ".join(signals).lower()

    # 优先级从高到低
    if any("拒答" in s for s in signals):
        return "refusal"
    if any("指令遵循" in s for s in signals):
        return "instruction_following"
    if any("幻觉" in s for s in signals) or any("编造" in s for s in signals):
        return "hallucination"
    if any("知识缺失" in s for s in signals) or any("不知道" in s for s in signals):
        return "knowledge_gap"
    if any("逻辑断裂" in s for s in signals):
        return "logic_break"
    if any("证据不一致" in s for s in signals):
        return "evidence_inconsistency"
    if any("事实错误" in s for s in signals):
        return "knowledge_gap"
    if any("不完整" in s for s in signals):
        return "knowledge_gap"

    return "uncategorized"


def _generate_detail(category: str, signals: list[str],
                     benchmark_type: str) -> str:
    """生成简短归因说明"""
    cat_info = ERROR_CATEGORIES.get(category, {})
    detail = cat_info.get("description", "未知错误")
    if signals:
        detail += f" — 检测到：{'; '.join(signals[:2])}"
    return detail


# ---------------------------------------------------------------------------
# 批量分析
# ---------------------------------------------------------------------------

def analyze_bad_cases(bad_cases: list[dict]) -> dict:
    """对评测报告中的所有 Bad Case 进行批量归因分析

    Args:
        bad_cases: 从评测结果中收集的错误题目列表
            [{"benchmark": str, "question": str, "expected": str,
              "predicted": str, "judge_reason": str, ...}]

    Returns:
        {
            "category_counts": {分类名: 数量},
            "category_details": {分类名: [各题分类结果]},
            "top_problems": [str],       # 主要问题
            "actionable_insights": [str], # 改进建议
            "severity_summary": str,
        }
    """
    if not bad_cases:
        return {
            "category_counts": {},
            "total": 0,
            "top_problems": [],
            "actionable_insights": ["✅ 暂无 Bad Case"],
            "severity_summary": "无",
        }

    results = []
    for bc in bad_cases:
        # 推断 benchmark 类型
        bench = bc.get("benchmark", "").lower()
        if "mmlu" in bench or "med" in bench:
            btype = "medical" if "med" in bench else "mmlu"
        elif "gsm" in bench:
            btype = "gsm8k"
        elif "human" in bench or "code" in bench:
            btype = "humaneval"
        elif "open" in bench or "eval" in bench:
            btype = "open_ended"
        else:
            btype = "general"

        result = classify_error(
            question=bc.get("question", ""),
            expected=bc.get("expected", ""),
            predicted=bc.get("predicted", ""),
            benchmark_type=btype,
            judge_reason=bc.get("judge_reason", ""),
        )
        result["benchmark"] = bench
        results.append(result)

    # 统计（含自定义分类）
    counts = {}
    for r in results:
        cat = r["label"]
        counts[cat] = counts.get(cat, 0) + 1
        # 自定义分类单独统计
        for cm in r.get("custom_matches", []):
            cm_label = cm["label"]
            counts[cm_label] = counts.get(cm_label, 0) + 1

    # 按数量排序
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    # 主要问题
    top_problems = [f"{emoji} {label} ({cnt} 题)"
                    for label, cnt in sorted_counts
                    for cat_name, cat_info in ERROR_CATEGORIES.items()
                    if cat_info["label"] == label
                    for emoji in [cat_info["emoji"]]]

    # 改进建议（按严重度排序）
    severe_order = {"严重": 0, "高": 1, "中": 2, "低": 3}
    unique_suggestions = []
    seen = set()
    for r in sorted(results, key=lambda x: severe_order.get(x["severity"], 99)):
        if r["training_suggestion"] not in seen:
            unique_suggestions.append(r["training_suggestion"])
            seen.add(r["training_suggestion"])

    # 严重度总结
    severe_count = sum(1 for r in results if r["severity"] in ("严重", "高"))
    severity = "严重" if severe_count >= len(results) * 0.5 else \
               "高" if severe_count >= len(results) * 0.3 else \
               "中" if severe_count > 0 else "低"

    return {
        "category_counts": dict(sorted_counts),
        "total": len(results),
        "details": results,
        "top_problems": top_problems[:5],
        "actionable_insights": unique_suggestions[:5],
        "severity_summary": severity,
    }


if __name__ == "__main__":
    # 自测
    test_cases = [
        {"benchmark": "MMLU", "question": "1+1=?", "expected": "A", "predicted": "我觉得这个问题很有趣..."},
        {"benchmark": "MMLU", "question": "感冒怎么治？", "expected": "B", "predicted": "我不能提供医疗建议，请咨询医生"},
        {"benchmark": "GSM8K", "question": "小明有3个苹果，吃了1个，还剩几个？",
         "expected": "2", "predicted": "小明有3个苹果，他吃了1个，因为他不喜欢苹果核，所以...我认为答案是5"},
    ]
    result = analyze_bad_cases(test_cases)
    print(f"Total bad cases: {result['total']}")
    print(f"Category counts: {result['category_counts']}")
    print(f"Top problems: {result['top_problems']}")
    print(f"Actionable insights: {result['actionable_insights']}")
