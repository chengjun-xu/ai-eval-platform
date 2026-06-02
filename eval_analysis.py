"""评测分析工具 — 回归检测 & 评分一致性
========================================
共享统计基础设施。

回归检测：
  - 双样本 bootstrapped 检验
  - 按 benchmark/subject 检测显著性下降
  - 生成回归报告

评分一致性：
  - Cohen's Kappa
  - Judge 偏差分析
  - 分布对比
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

random.seed(42)


# ════════════════════════════════════════════════════════════════════════
# 1. 回归检测
# ════════════════════════════════════════════════════════════════════════


def detect_regression(run_a: dict, run_b: dict,
                      ci: float = 0.95,
                      min_decline: float = 2.0) -> dict:
    """检测 run_b 相对 run_a 是否有显著回归

    Args:
        run_a: 基线评测结果（较早/较好的版本）
        run_b: 对比评测结果
        ci: 置信水平
        min_decline: 最小下降百分点（过滤噪音）

    Returns:
        {
            "regressions": [{benchmark, score_a, score_b, decline, p_value}, ...],
            "improvements": [...],
            "summary": {total_regressed, total_improved, unchanged, avg_decline}
        }
    """
    benchmarks_a = run_a.get("benchmarks", {})
    benchmarks_b = run_b.get("benchmarks", {})

    regressions = []
    improvements = []

    for bid, b_res_b in benchmarks_b.items():
        b_res_a = benchmarks_a.get(bid)
        if not b_res_a:
            continue

        score_a = b_res_a.get("score", 0)
        score_b = b_res_b.get("score", 0)
        decline = score_a - score_b

        # Bootstrap 显著性检验
        details_a = b_res_a.get("details", [])
        details_b = b_res_b.get("details", [])
        p_value = _bootstrap_p_value(details_a, details_b, n_iter=1000)

        entry = {
            "benchmark": bid,
            "score_a": round(score_a, 1),
            "score_b": round(score_b, 1),
            "decline": round(decline, 1),
            "p_value": round(p_value, 4),
            "significant": p_value < (1 - ci),
        }

        if decline >= min_decline and p_value < (1 - ci):
            entry["severity"] = "high" if decline > 5 else "medium"
            regressions.append(entry)
        elif decline <= -min_decline and p_value < (1 - ci):
            entry["severity"] = "high" if decline < -5 else "medium"
            improvements.append(entry)

    # 排序
    regressions.sort(key=lambda x: x["decline"], reverse=True)
    improvements.sort(key=lambda x: abs(x["decline"]), reverse=True)

    total_regressed = len(regressions)
    total_improved = len(improvements)
    unchanged = len(benchmarks_b) - total_regressed - total_improved
    avg_decline = round(
        sum(r["decline"] for r in regressions) / total_regressed, 1
    ) if regressions else 0

    return {
        "model_a": run_a.get("model_name", "?"),
        "model_b": run_b.get("model_name", "?"),
        "run_a_id": run_a.get("run_id", ""),
        "run_b_id": run_b.get("run_id", ""),
        "regressions": regressions,
        "improvements": improvements,
        "summary": {
            "total_regressed": total_regressed,
            "total_improved": total_improved,
            "unchanged": unchanged,
            "avg_decline": avg_decline,
            "overall_delta": round(
                (run_b.get("overall_score", 0) or 0)
                - (run_a.get("overall_score", 0) or 0), 1
            ),
        },
    }


def _bootstrap_p_value(details_a: list, details_b: list,
                       n_iter: int = 1000) -> float:
    """Bootstrap 法计算两个评测结果差异的 p 值"""
    scores_a = [1.0 if d.get("correct", False) else 0.0 for d in details_a]
    scores_b = [1.0 if d.get("correct", False) else 0.0 for d in details_b]

    n_a, n_b = len(scores_a), len(scores_b)
    if n_a == 0 or n_b == 0:
        return 1.0

    mean_a = sum(scores_a) / n_a
    mean_b = sum(scores_b) / n_b
    obs_diff = mean_a - mean_b

    # 混合后重采样
    pooled = scores_a + scores_b
    count_extreme = 0
    for _ in range(n_iter):
        sample_a = [random.choice(pooled) for _ in range(n_a)]
        sample_b = [random.choice(pooled) for _ in range(n_b)]
        diff = (sum(sample_a) / n_a) - (sum(sample_b) / n_b)
        if abs(diff) >= abs(obs_diff):
            count_extreme += 1

    return count_extreme / n_iter


# ════════════════════════════════════════════════════════════════════════
# 2. 评分一致性分析
# ════════════════════════════════════════════════════════════════════════


def cohens_kappa(ratings_a: list[float], ratings_b: list[float],
                 n_categories: int = 5) -> dict:
    """计算两个 Judge 的 Cohen's Kappa 一致性

    Args:
        ratings_a: Judge A 的评分列表（1-5）
        ratings_b: Judge B 的评分列表（1-5）
        n_categories: 评分等级数

    Returns:
        {"kappa": float, "agreement": float, "interpretation": str}
    """
    if len(ratings_a) != len(ratings_b) or len(ratings_a) == 0:
        return {"kappa": 0, "agreement": 0, "interpretation": "数据不足"}

    # 离散化（取整）
    a_disc = [max(1, min(n_categories, round(r))) for r in ratings_a]
    b_disc = [max(1, min(n_categories, round(r))) for r in ratings_b]

    n = len(a_disc)
    # 混淆矩阵
    matrix = [[0] * n_categories for _ in range(n_categories)]
    for a, b in zip(a_disc, b_disc):
        matrix[a - 1][b - 1] += 1

    # 观测一致率
    observed = sum(matrix[i][i] for i in range(n_categories)) / n

    # 期望一致率
    row_sums = [sum(row) / n for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(n_categories)) / n
                for j in range(n_categories)]
    expected = sum(row_sums[i] * col_sums[i] for i in range(n_categories))

    # Kappa
    if expected == 1:
        kappa = 1.0
    else:
        kappa = (observed - expected) / (1 - expected)

    # 解释
    if kappa >= 0.81:
        interp = "几乎完全一致"
    elif kappa >= 0.61:
        interp = "显著一致"
    elif kappa >= 0.41:
        interp = "中等一致"
    elif kappa >= 0.21:
        interp = "一般一致"
    else:
        interp = "一致性差"

    return {
        "kappa": round(kappa, 4),
        "agreement": round(observed * 100, 1),
        "interpretation": interp,
        "n": n,
    }


def analyze_score_distribution(details: list[dict],
                               score_key: str = "judge_score") -> dict:
    """分析评分分布偏差

    Returns:
        {"mean": float, "std": float, "histogram": [counts_by_score],
         "bias": "偏高"/"偏低"/"正常", "score_gap": float}
    """
    scores = [d.get(score_key, 0) or 0 for d in details if d.get(score_key)]
    if not scores:
        return {"mean": 0, "std": 0, "histogram": [], "bias": "无数据", "score_gap": 0}

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = math.sqrt(variance)

    # 直方图（1-5 分）
    histogram = [0] * 5
    for s in scores:
        idx = max(0, min(4, round(s) - 1))
        histogram[idx] += 1

    # 偏差判断
    if mean > 4.0:
        bias = "偏高（宽松评分）"
    elif mean < 2.5:
        bias = "偏低（严格评分）"
    else:
        bias = "正常"

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "histogram": histogram,
        "bias": bias,
        "score_gap": round(max(scores) - min(scores), 1),
        "n": len(scores),
    }


def compare_judges(judge_results: dict[str, list[dict]]) -> dict:
    """对比多个 Judge 模型在相同题目上的评分一致性

    Args:
        judge_results: {judge_name: [detail_dict, ...]}

    Returns:
        每个 Judge pair 的 Kappa 和分布分析
    """
    judges = list(judge_results.keys())
    pairs = {}

    # 提取评分
    ratings = {}
    for j, details in judge_results.items():
        ratings[j] = [d.get("judge_score", 3) or 3 for d in details]

    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            j1, j2 = judges[i], judges[j]
            kappa = cohens_kappa(ratings[j1], ratings[j2])
            pairs[f"{j1} ↔ {j2}"] = kappa

    # 每个 judge 的分布
    distributions = {
        j: analyze_score_distribution(
            judge_results[j], score_key="judge_score"
        )
        for j in judges
    }

    return {
        "pairs": pairs,
        "distributions": distributions,
        "judges": judges,
    }
