"""
自动化数据生产引擎
====================
支持三种数据生产方式，产出可直接注册为 benchmark 的评测数据。

模块组成：
  1. Self-Instruct  — 从种子指令通过 LLM 扩写生成新指令
  2. Evol-Instruct  — 对现有指令进行进化增强（加约束、深化、多步等）
  3. Synthetic RL  — 生成偏好对用于 RLHF/DPO 训练数据

使用方式（所有函数）：
  generate_* -> list[dict]  → 自行处理
  generate_*_and_register   → 直接注册为平台 benchmark
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── LLM 调用封装 ────────────────────────────────────────────────────────

def _call_llm(model: dict, prompt: str, timeout: int = 60) -> dict:
    """调用 LLM 返回 parsed 结果（复用 eval_runner 风格）"""
    import requests as _req
    url = model.get("api_base", "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {model.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model.get("model_name") or model.get("name", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    start = time.time()
    try:
        resp = _req.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = round(time.time() - start, 2)
        if resp.status_code != 200:
            return {"error": f"API ERROR {resp.status_code}", "content": ""}
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        return {
            "content": content,
            "latency": elapsed,
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception as e:
        return {"error": str(e), "content": ""}


def _parse_json_from_llm(text: str) -> list[dict] | None:
    """从 LLM 回复中提取 JSON 数组（兼容各种格式包裹）"""
    # 尝试 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        pass
    # 尝试按行解析（每行一个 JSON object）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    results = []
    for line in lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
        except json.JSONDecodeError:
            continue
    return results if results else None


# ════════════════════════════════════════════════════════════════════════
# 1. Self-Instruct  — 种子指令扩写
# ════════════════════════════════════════════════════════════════════════

SELF_INSTRUCT_PROMPT_TEMPLATE = """你是一个评测数据生成器。请参考以下种子指令，生成新的评测题目。

要求：
- 生成的题目必须与种子指令风格一致但主题不同
- 每个题目包含：question, answer, category, difficulty
- category 从以下选择：{categories}
- difficulty 为 easy/medium/hard
- 保持题目合理、可回答、答案明确

种子指令示例：
{seed_examples}

请严格按照 JSON 数组格式输出，每个元素格式：
{{"question": "...", "answer": "...", "category": "...", "difficulty": "medium"}}

生成 {num_items} 个新题目："""


def generate_self_instruct(
    seed_items: list[dict],
    model: dict,
    num_items: int = 10,
    categories: list[str] | None = None,
) -> list[dict]:
    """Self-Instruct：从种子指令扩写生成新评测题目

    Args:
        seed_items: [{"question":..., "answer":..., "category":..., ...}, ...]
        model: LLM 模型配置
        num_items: 要生成的数量
        categories: 允许的类别列表

    Returns:
        [{"id", "question", "answer", "category", "difficulty", "source"}, ...]
    """
    if not seed_items:
        raise ValueError("种子指令不能为空")

    if categories is None:
        categories = ["知识理解", "数学推理", "代码能力", "推理能力", "综合能力"]

    # 取少量种子示例用于 prompt
    samples = random.sample(seed_items, min(5, len(seed_items)))
    seed_examples = "\n".join(
        f"- Q: {s['question']}\n  A: {s['answer']}\n  C: {s.get('category', '通用')}"
        for s in samples
    )

    prompt = SELF_INSTRUCT_PROMPT_TEMPLATE.format(
        categories=", ".join(categories),
        seed_examples=seed_examples,
        num_items=num_items,
    )

    result = _call_llm(model, prompt, timeout=120)
    if result.get("error"):
        raise RuntimeError(f"LLM 调用失败: {result['error']}")

    items = _parse_json_from_llm(result["content"])
    if not items:
        # 兜底：尝试按行解析
        items = []
        for line in result["content"].split("\n"):
            line = line.strip().strip(",")
            if line.startswith("{"):
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # 格式化输出
    output = []
    for i, it in enumerate(items[:num_items]):
        q = it.get("question", "").strip()
        a = it.get("answer", "").strip()
        if not q or not a:
            continue
        output.append({
            "id": f"self_instruct_{i}",
            "question": q,
            "answer": a,
            "category": it.get("category", "通用"),
            "difficulty": it.get("difficulty", "medium"),
            "source": "self-instruct",
        })

    # 去重（基于 question 前 50 字符）
    seen = set()
    deduped = []
    for item in output:
        key = item["question"][:50]
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


# ════════════════════════════════════════════════════════════════════════
# 2. Evol-Instruct  — 指令进化增强
# ════════════════════════════════════════════════════════════════════════

EVOL_STRATEGIES = {
    "constraints": "增加约束条件",      # 添加边界条件/限制
    "deepen": "深化推理",              # 要求多步推理/深度分析
    "reasoning_steps": "增加推理步骤",   # 需要更多中间推理
    "complicate_input": "复杂化输入",   # 增加干扰信息/多条件
    "cross_domain": "跨域融合",         # 融合两个不同领域知识
}


EVOL_PROMPT_TEMPLATE = """你是一个数据进化引擎。请对以下题目进行"{strategy}"进化。

进化说明：{strategy_description}

原始题目：
题目: {question}
答案: {answer}
类别: {category}

进化要求：
- 保留原题核心知识/技能点
- 在不改变答案方向的前提下增加难度
- 答案需要明确且可验证
- 保持题目合理、无歧义

请严格按照以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{"question": "...", "answer": "...", "category": "...", "difficulty": "hard", "evolution_note": "...做了何种进化..."}}
"""


def evolve_instruction(
    item: dict,
    model: dict,
    strategy: str = "constraints",
) -> dict | None:
    """对单条指令进行进化增强

    Args:
        item: {"question", "answer", "category", ...}
        model: LLM 模型配置
        strategy: 进化策略（EVOL_STRATEGIES 的 key）

    Returns:
        进化后的指令 dict 或 None
    """
    desc = EVOL_STRATEGIES.get(strategy, strategy)
    prompt = EVOL_PROMPT_TEMPLATE.format(
        strategy=strategy,
        strategy_description=desc,
        question=item.get("question", ""),
        answer=item.get("answer", ""),
        category=item.get("category", "通用"),
    )

    result = _call_llm(model, prompt, timeout=60)
    if result.get("error"):
        return None

    parsed = _parse_json_from_llm(result["content"])
    if not parsed:
        return None

    ev = parsed[0]
    q = ev.get("question", "").strip()
    a = ev.get("answer", "").strip()
    if not q or not a:
        return None

    return {
        "id": f"evolved_{item.get('id', 'unknown')}_{strategy}",
        "question": q,
        "answer": a,
        "category": ev.get("category", item.get("category", "通用")),
        "difficulty": "hard",
        "source": f"evol-instruct:{strategy}",
        "original_id": item.get("id", ""),
        "evolution_note": ev.get("evolution_note", ""),
    }


def batch_evolve(
    items: list[dict],
    model: dict,
    strategies: list[str] | None = None,
    evolve_ratio: float = 1.0,
    max_items: int = 50,
) -> list[dict]:
    """批量进化指令

    Args:
        items: 原始指令列表
        model: LLM 模型配置
        strategies: 要使用的进化策略列表
        evolve_ratio: 每条指令平均进化比例（1.0 = 每条进化一次）
        max_items: 最多进化多少条

    Returns:
        进化后的指令列表
    """
    if strategies is None:
        strategies = list(EVOL_STRATEGIES.keys())

    # 采样要进化的子集
    sample = items[:max_items]
    results = []
    for idx, item in enumerate(sample):
        # 每条指令随机选一种策略
        strategy = random.choice(strategies)
        evolved = evolve_instruction(item, model, strategy)
        if evolved:
            results.append(evolved)
        # 控制频率
        if idx >= max_items:
            break

    return results


# ════════════════════════════════════════════════════════════════════════
# 3. Synthetic Data for RL — 偏好对生成
# ════════════════════════════════════════════════════════════════════════

RL_GENERATE_PROMPT_TEMPLATE = """请回答以下指令，给出两个不同风格的回答。
第一个回答：{style_a}
第二个回答：{style_b}

指令: {instruction}

请严格按照以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{"response_a": "...[第一个回答的完整内容]...", "response_b": "...[第二个回答的完整内容]..."}}
"""

RL_JUDGE_PROMPT_TEMPLATE = """请比较以下两个回答，判断哪个更好，说明理由。

指令: {instruction}

回答 A: {response_a}

回答 B: {response_b}

请根据以下标准判断：
1. 准确性：是否正确回答了指令要求
2. 完整性：是否覆盖了关键点
3. 清晰度：表达是否清晰有条理
4. 实用性：回答是否实际有用

请严格按照以下 JSON 格式输出（只输出 JSON，不要其他内容）：
{{"preferred": "A" 或 "B", "reason": "...判断理由...", "scores": {{"A": 0-10, "B": 0-10}}}}
"""


def generate_rl_preference_pairs(
    instructions: list[str],
    model: dict,
    judge_model: dict | None = None,
    style_a: str = "详细且严谨",
    style_b: str = "简洁且直接",
    num_pairs: int = 10,
) -> list[dict]:
    """生成 RLHF 偏好对数据

    Args:
        instructions: 指令列表（纯文本）
        model: 用于生成回答的 LLM
        judge_model: 用于判断偏好的 LLM（为 None 则用 model 自身）
        style_a: 回答 A 的风格描述
        style_b: 回答 B 的风格描述
        num_pairs: 最多生成多少对

    Returns:
        [{"instruction", "chosen", "rejected", "chosen_score", "rejected_score", "reason"}, ...]
    """
    if judge_model is None:
        judge_model = model

    results = []
    for idx, instr in enumerate(instructions[:num_pairs]):
        if not instr.strip():
            continue

        # Step 1: 生成两个不同风格的回复
        gen_prompt = RL_GENERATE_PROMPT_TEMPLATE.format(
            instruction=instr,
            style_a=style_a,
            style_b=style_b,
        )
        gen_result = _call_llm(model, gen_prompt, timeout=60)
        if gen_result.get("error"):
            continue

        parsed = _parse_json_from_llm(gen_result["content"])
        if not parsed:
            continue

        resp_a = parsed[0].get("response_a", "").strip()
        resp_b = parsed[0].get("response_b", "").strip()
        if not resp_a or not resp_b:
            continue

        # Step 2: Judge 判断偏好
        judge_prompt = RL_JUDGE_PROMPT_TEMPLATE.format(
            instruction=instr,
            response_a=resp_a,
            response_b=resp_b,
        )
        judge_result = _call_llm(judge_model, judge_prompt, timeout=60)
        if judge_result.get("error"):
            # Judge 失败时默认 A 为 chosen
            chosen, rejected = resp_a, resp_b
            chosen_score, rejected_score = 5, 5
            reason = "Judge 未响应，默认 A"
        else:
            jp = _parse_json_from_llm(judge_result["content"])
            if jp:
                j = jp[0]
                preferred = j.get("preferred", "A")
                scores = j.get("scores", {})
                try:
                    score_a = float(scores.get("A", 5))
                    score_b = float(scores.get("B", 5))
                except (TypeError, ValueError):
                    score_a, score_b = 5, 5

                if preferred.upper() == "A":
                    chosen, rejected = resp_a, resp_b
                    chosen_score, rejected_score = score_a, score_b
                else:
                    chosen, rejected = resp_b, resp_a
                    chosen_score, rejected_score = score_b, score_a
                reason = j.get("reason", "")
            else:
                chosen, rejected = resp_a, resp_b
                chosen_score, rejected_score = 5, 5
                reason = ""

        results.append({
            "id": f"rl_pair_{idx}",
            "instruction": instr,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "reason": reason,
            "source": "synthetic-rl",
        })

    return results


# ════════════════════════════════════════════════════════════════════════
# 注册到平台
# ════════════════════════════════════════════════════════════════════════


def register_as_benchmark(
    items: list[dict],
    benchmark_name: str,
    datasets_dir: str | Path,
) -> str:
    """将生产的数据注册为平台 benchmark

    Self-Instruct / Evol-Instruct 产出可直接注册为开放题 benchmark。
    Synthetic RL 产出注册为偏好对 benchmark。

    Args:
        items: 数据项列表
        benchmark_name: benchmark 标识符
        datasets_dir: data/datasets/ 目录

    Returns:
        保存的文件名
    """
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{benchmark_name}_custom.json"
    filepath = datasets_dir / filename

    # 确保所有条目都有 id
    for i, item in enumerate(items):
        item.setdefault("id", str(i))
        item.setdefault("category", "通用")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    return filename


def generate_self_instruct_and_register(
    seed_items: list[dict],
    model: dict,
    benchmark_name: str,
    datasets_dir: str | Path,
    num_items: int = 10,
    categories: list[str] | None = None,
) -> dict:
    """Self-Instruct → 注册为 benchmark"""
    items = generate_self_instruct(
        seed_items, model, num_items=num_items, categories=categories
    )
    filename = register_as_benchmark(items, benchmark_name, datasets_dir)
    return {"filename": filename, "count": len(items), "items": items}


def batch_evolve_and_register(
    items: list[dict],
    model: dict,
    benchmark_name: str,
    datasets_dir: str | Path,
    strategies: list[str] | None = None,
    max_items: int = 50,
) -> dict:
    """Evol-Instruct → 注册为 benchmark"""
    evolved = batch_evolve(items, model, strategies=strategies, max_items=max_items)
    filename = register_as_benchmark(evolved, benchmark_name, datasets_dir)
    return {"filename": filename, "count": len(evolved), "items": evolved}


def generate_rl_pairs_and_register(
    instructions: list[str],
    model: dict,
    benchmark_name: str,
    datasets_dir: str | Path,
    judge_model: dict | None = None,
    num_pairs: int = 10,
) -> dict:
    """Synthetic RL → 注册为 benchmark"""
    pairs = generate_rl_preference_pairs(
        instructions, model, judge_model=judge_model, num_pairs=num_pairs
    )
    filename = register_as_benchmark(pairs, benchmark_name, datasets_dir)
    return {"filename": filename, "count": len(pairs), "items": pairs}
