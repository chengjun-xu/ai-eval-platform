"""
评测 Agent — 自然语言驱动的自动评测引擎
支持解析"帮我评测 DeepSeek V4 Flash"等自然语言指令，
自动选择模型、Benchmark、Judge，编排并执行评测流程。
"""
import json
import re
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent / "data"


# ── 意图解析 ────────────────────────────────────────────────────────────────

def load_models() -> list:
    path = DATA_DIR / "models.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def load_judge_models() -> list:
    path = DATA_DIR / "judge_models.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return []


def load_benchmarks() -> list:
    """返回 (id, name, category, question_count) 列表"""
    datasets_dir = DATA_DIR / "datasets"
    benchmarks = []
    mapping = {
        "mmlu_sample": "MMLU",
        "mmlu_extended": "MMLU",
        "gsm8k_sample": "GSM8K",
        "gsm8k_extended": "GSM8K",
        "humaneval_sample": "HumanEval",
        "humaneval_extended": "HumanEval",
        "open_ended_sample": "OpenEval",
    }
    for fname in sorted(datasets_dir.glob("*_extended.json")) + sorted(datasets_dir.glob("*_sample.json")):
        key = fname.stem
        name = mapping.get(key, key)
        version = "扩展版" if "_extended" in key else "基础版"
        with open(fname, encoding="utf-8") as f:
            items = json.load(f)
        bid = key.replace("_sample", "").replace("_extended", "") + ("_ext" if "_extended" in key else "")
        benchmarks.append({
            "id": bid,
            "name": name,
            "version": version,
            "category": "知识" if "mmlu" in name.lower() else
                       "数学" if "gsm" in name.lower() else
                       "代码" if "code" in name.lower() or "humaneval" in name.lower() else
                       "开放题",
            "count": len(items),
            "extended": "_extended" in key,
        })
    return benchmarks


def parse_intent(text: str) -> dict:
    """
    解析自然语言评测指令。
    
    支持的句式:
    - "评测 xxx" → 识别模型名
    - "跑 MMLU 和 GSM8K" / "跑全部" → 选择 Benchmark
    - "用 Judge" / "用 xxx 作为 Judge" → 选择 Judge 模型
    - "只跑基础版" / "用扩展版" → 选择数据集版本

    返回: {
        "model_id": str | None,
        "model_name": str | None,
        "benchmark_ids": list[str] | None (None=全部),
        "judge_id": str | None (None=不使用),
        "use_extended": bool | None (None=自动),
        "confidence": float (0~1),
    }
    """
    result = {
        "model_id": None,
        "model_name": None,
        "benchmark_ids": None,
        "judge_id": None,
        "use_extended": None,
        "confidence": 0.5,
    }
    text_lower = text.lower()

    # 1. 识别模型
    models = load_models()
    if not models:
        return {**result, "error": "没有注册任何模型，请先去「模型管理」添加模型"}

    for m in models:
        if m["name"].lower() in text_lower or m["id"].lower() in text_lower:
            result["model_id"] = m["id"]
            result["model_name"] = m["name"]
            result["confidence"] += 0.3
            break

    if not result["model_id"]:
        # 尝试用第一个注册的模型
        result["model_id"] = models[0]["id"]
        result["model_name"] = models[0]["name"]
        result["confidence"] -= 0.1

    # 2. 识别 Benchmark
    benchmarks = load_benchmarks()
    if not benchmarks:
        return {**result, "error": "没有可用数据集"}

    # 检查是否指定了具体 benchmark
    mentioned = []
    for keyword, names in [
        ("mmlu", ["mmlu", "知识", "多学科"]),
        ("gsm8k", ["gsm8k", "数学", "算术"]),
        ("humaneval", ["humaneval", "代码", "编程", "human"]),
        ("open_ended", ["open", "开放题", "自由回答"]),
        ("ceval_custom", ["ceval", "中文评测", "中文综合"]),
    ]:
        if any(n in text_lower for n in names):
            mentioned.append(keyword)

    if mentioned:
        # 只跑提到的
        result["benchmark_ids"] = []
        for bid, b in [(bm["id"], bm) for bm in benchmarks]:
            for m_keyword in mentioned:
                if m_keyword in bid or m_keyword in b["name"].lower():
                    if bid not in result["benchmark_ids"]:
                        result["benchmark_ids"].append(bid)
        result["confidence"] += 0.2
    elif any(w in text_lower for w in ["全部", "所有", "全跑", "都跑"]):
        # 跑所有扩展版
        result["benchmark_ids"] = [b["id"] for b in benchmarks if b["extended"]]
        if not result["benchmark_ids"]:
            result["benchmark_ids"] = [b["id"] for b in benchmarks]
        result["confidence"] += 0.2

    # 3. 版本选择
    if any(w in text_lower for w in ["基础", "sample"]):
        result["use_extended"] = False
    elif any(w in text_lower for w in ["扩展", "extended", "全部"]):
        result["use_extended"] = True

    # 4. Judge 模型
    judge_models = load_judge_models()
    if judge_models and any(w in text_lower for w in ["judge", "评分", "开放", "open", "评价"]):
        # 尝试匹配指定 Judge
        for jm in judge_models:
            if jm["name"].lower() in text_lower or jm["id"].lower() in text_lower:
                result["judge_id"] = jm["id"]
                break
        if not result["judge_id"] and judge_models:
            # 默认用第一个 Judge
            result["judge_id"] = judge_models[0]["id"]
        result["confidence"] += 0.2

    # 如果没有明确选择 Benchmark 且没指定 Judge，默认跑 MMLU+GSM8K
    if result["benchmark_ids"] is None:
        result["benchmark_ids"] = [
            b["id"] for b in benchmarks
            if b["extended"] and b["name"] in ["MMLU", "GSM8K"]
        ][:2]
        result["confidence"] = max(0.3, result["confidence"])

    return result


def describe_plan(intent: dict) -> str:
    """将解析结果转为可读的执行计划"""
    if "error" in intent:
        return f"❌ {intent['error']}"

    parts = [f"🤖 **评测模型**: {intent.get('model_name', '?')}"]
    
    if intent.get("benchmark_ids"):
        benches = []
        for bid in intent["benchmark_ids"]:
            friendly = {
                "mmlu": "MMLU（知识理解）", "mmlu_ext": "MMLU 扩展版（知识理解）",
                "gsm8k": "GSM8K（数学推理）", "gsm8k_ext": "GSM8K 扩展版（数学推理）",
                "humaneval": "HumanEval（代码生成）", "humaneval_ext": "HumanEval 扩展版（代码）",
                "open_ended": "OpenEval（开放题）",
            }
            benches.append(friendly.get(bid, bid))
        parts.append(f"📋 **评测内容**: {' + '.join(benches)}")
    
    if intent.get("judge_id"):
        parts.append(f"⚖️  **Judge 评分**: 启用")
    else:
        parts.append(f"⚖️  **Judge 评分**: 未启用")
    
    parts.append(f"\n📊 **预计调用次数**: ~{len(intent.get('benchmark_ids', [])) * 5} 次 API 调用")
    
    return "\n".join(parts)


# ── 执行编排 ────────────────────────────────────────────────────────────────

def prepare_eval(intent: dict) -> dict:
    """
    将意图转换为 start_eval() 的入参。
    返回: {"model": dict, "benchmark_ids": list, "judge_model": dict | None, ok: bool}
    """
    models = load_models()
    model = next((m for m in models if m["id"] == intent.get("model_id")), None)
    if not model:
        return {"ok": False, "error": f"模型 '{intent.get('model_name')}' 未注册"}

    judge_model = None
    if intent.get("judge_id"):
        judge_models = load_judge_models()
        judge_model = next((jm for jm in judge_models if jm["id"] == intent["judge_id"]), None)

    return {
        "ok": True,
        "model": model,
        "benchmark_ids": intent.get("benchmark_ids", []),
        "judge_model": judge_model,
    }
