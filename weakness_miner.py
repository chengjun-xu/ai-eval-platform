"""
弱项挖掘引擎 (Weakness Miner)
===============================
自动分析模型评测结果，定位能力短板、错误模式和改进方向。
"""
import json
from pathlib import Path
from typing import Optional
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"

# 能力维度排序（用于展示）
DIM_ORDER = [
    "知识理解", "数学推理", "代码能力", "推理能力",
    "中文专项", "医疗专业", "安全合规",
    "多模态", "多语言", "综合能力",
    "红队/对抗", "长上下文", "自定义", "其他",
]

# Benchmark ID → 类别映射（运行时的 fallback）
BENCHMARK_CATEGORIES: dict[str, str] = {}


def _load_runs(user: str = "") -> list[dict]:
    """加载所有评测记录"""
    runs_file = DATA_DIR / "eval_runs.json"
    if not runs_file.exists():
        return []
    try:
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    if user:
        runs = [r for r in runs if r.get("user", "") == user]
    return runs


def _get_category(benchmark_id: str) -> str:
    """获取 benchmark 的类别（优先用运行时的缓存）"""
    if benchmark_id in BENCHMARK_CATEGORIES:
        return BENCHMARK_CATEGORIES[benchmark_id]
    # 从 app 的 load_benchmarks 获取
    try:
        from app import load_benchmarks
        for b in load_benchmarks():
            if b["id"] == benchmark_id:
                BENCHMARK_CATEGORIES[benchmark_id] = b.get("category", "其他")
                return BENCHMARK_CATEGORIES[benchmark_id]
    except Exception:
        pass
    # 从 id 推断
    if benchmark_id.startswith("hf_"):
        return "知识理解"
    if benchmark_id.startswith("oc_"):
        return "知识理解"
    return "其他"


def analyze_model(model_name: str, user: str = "") -> dict:
    """分析单个模型的弱项

    Returns:
        {
            "model_name": str,
            "overall_score": float,
            "total_runs": int,
            "dimensions": [  # 按能力维度聚合
                {"name": "数学推理", "score": 62.5, "benchmarks": [...], "count": 3, "weakness": "major"},
                ...
            ],
            "weakest_benchmarks": [  # 最低分 benchmark TOP N
                {"name": "MATH-Lvl5", "score": 35.0, "category": "数学推理", "correct": 7, "total": 20},
                ...
            ],
            "error_distribution": {  # 错误类型分布（如果有 error_classifier）
                "知识缺失": 12,
                "逻辑断裂": 8,
                ...
            },
            "suggestions": [  # 针对性改进建议
                "在**数学推理**维度表现较弱（平均 62.5%），建议重点提升高难度数学题的推理能力",
                ...
            ],
            "detailed_error_types": {  # 每个 benchmark 的错误类型详情
                "gsm8k": {"知识缺失": 5, "逻辑断裂": 3},
                ...
            },
            "category_ranking": [  # 按分数从低到高排列的类别
                {"name": "代码能力", "score": 55.0, "level": "weak"},
                ...
            ],
        }
    """
    runs = _load_runs(user)
    model_runs = [r for r in runs if r.get("model_name", "") == model_name]

    if not model_runs:
        return {"model_name": model_name, "error": "没有找到该模型的评测记录"}

    # 取最新一次完整运行（优先 completed）
    completed = [r for r in model_runs if r.get("status") == "completed"]
    target_runs = completed if completed else model_runs[-1:]

    # ── 1. 按 Benchmark 聚合 ──
    bench_scores: dict[str, dict] = {}
    for r in target_runs:
        for bid, bd in r.get("benchmarks", {}).items():
            score = bd.get("score", 0)
            if bid not in bench_scores or score > (bench_scores[bid].get("max_score", 0) or 0):
                # 取最高分
                cat = _get_category(bid)
                bench_scores[bid] = {
                    "id": bid,
                    "name": bid.upper(),
                    "category": cat,
                    "score": score,
                    "correct": bd.get("correct", 0),
                    "total": bd.get("total", 0),
                    "max_score": score,
                }

    # ── 2. 按能力维度聚合 ──
    dim_groups: dict[str, dict] = {}
    for bid, info in bench_scores.items():
        cat = info["category"]
        if cat not in dim_groups:
            dim_groups[cat] = {"name": cat, "total_score": 0, "count": 0, "benchmarks": []}
        dim_groups[cat]["total_score"] += info["score"]
        dim_groups[cat]["count"] += 1
        dim_groups[cat]["benchmarks"].append({
            "id": info["id"],
            "name": info["name"],
            "score": info["score"],
            "correct": info["correct"],
            "total": info["total"],
        })

    dimensions = []
    for cat, g in dim_groups.items():
        avg = round(g["total_score"] / g["count"], 1) if g["count"] else 0
        weakness = "major" if avg < 50 else ("minor" if avg < 70 else "none")
        dimensions.append({
            "name": cat,
            "score": avg,
            "count": g["count"],
            "weakness": weakness,
            "benchmarks": sorted(g["benchmarks"], key=lambda x: x["score"]),
        })

    # 按 DIM_ORDER 排序
    dim_order_map = {n: i for i, n in enumerate(DIM_ORDER)}
    dimensions.sort(key=lambda d: dim_order_map.get(d["name"], 99))

    # ── 3. 弱项排名 ──
    all_bench_list = sorted(bench_scores.values(), key=lambda x: x["score"])

    # ── 4. 错误分布 ──
    all_details = []
    for r in target_runs:
        for bd in r.get("benchmarks", {}).values():
            all_details.extend(bd.get("details", []) or [])

    error_dist = {}
    detailed_error_types = {}
    for dt in all_details:
        err_cat = dt.get("category") or dt.get("error", "未分类")
        if err_cat not in error_dist:
            error_dist[err_cat] = 0
        error_dist[err_cat] += 1

    # ── 5. 改进建议 ──
    suggestions = []
    # 按维度弱项
    for dim in sorted(dimensions, key=lambda d: d["score"]):
        if dim["weakness"] == "major":
            suggestions.append(
                f"**{dim['name']}** 维度表现严重不足（平均 {dim['score']}%），"
                f"涵盖 {dim['count']} 个 benchmark ⚠️"
            )
        elif dim["weakness"] == "minor":
            suggestions.append(
                f"**{dim['name']}** 维度有提升空间（平均 {dim['score']}%），"
                f"建议优先关注分数最低的 benchmark"
            )

    # 按单个 benchmark 弱项
    for b in all_bench_list[:5]:
        if b["score"] < 60:
            suggestions.append(
                f"**{b['name']}**（{b['category']}）得分仅 {b['score']}%，"
                f"正确 {b['correct']}/{b['total']} 题"
            )

    # 总体建议
    overall_scores = [d["score"] for d in dimensions if d["count"] > 0]
    if overall_scores:
        avg_all = round(sum(overall_scores) / len(overall_scores), 1)
        if avg_all < 65:
            suggestions.append(
                f"模型整体平均分 {avg_all}%，建议先检查模型配置（API Base、温度参数）是否正常"
            )

    # ── 6. 错误类型详情（按 benchmark） ──
    for dt in all_details:
        bid = dt.get("benchmark_id", "")
        err_cat = dt.get("category") or dt.get("error", "未分类")
        if bid not in detailed_error_types:
            detailed_error_types[bid] = {}
        if err_cat not in detailed_error_types[bid]:
            detailed_error_types[bid][err_cat] = 0
        detailed_error_types[bid][err_cat] += 1

    total_questions = sum(b["total"] for b in all_bench_list)
    total_correct = sum(b["correct"] for b in all_bench_list)
    overall = round(total_correct / total_questions * 100, 1) if total_questions else 0

    # ── 分数排名（从弱到强） ──
    category_ranking = []
    for dim in sorted(dimensions, key=lambda d: d["score"]):
        level = "weak" if dim["score"] < 50 else ("below_avg" if dim["score"] < 70 else "average" if dim["score"] < 85 else "strong")
        category_ranking.append({
            "name": dim["name"],
            "score": dim["score"],
            "level": level,
            "count": dim["count"],
        })

    return {
        "model_name": model_name,
        "overall_score": overall,
        "total_runs": len(target_runs),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "dimensions": dimensions,
        "weakest_benchmarks": all_bench_list[:15],
        "strongest_benchmarks": all_bench_list[-5:][::-1],
        "error_distribution": error_dist,
        "suggestions": suggestions,
        "category_ranking": category_ranking,
        "detailed_error_types": detailed_error_types,
        "all_benchmarks": all_bench_list,
    }


def compare_weaknesses(model_names: list[str], user: str = "") -> dict:
    """多模型对比弱项"""
    results = {}
    for name in model_names:
        results[name] = analyze_model(name, user)

    # 找每个维度的「最弱模型」
    dim_comparison = {}
    for name, analysis in results.items():
        for dim in analysis.get("dimensions", []):
            dn = dim["name"]
            if dn not in dim_comparison:
                dim_comparison[dn] = []
            dim_comparison[dn].append({"model": name, "score": dim["score"]})

    # 每个维度标注谁最弱
    highlights = {}
    for dim, scores in dim_comparison.items():
        scores.sort(key=lambda x: x["score"])
        for s in scores:
            highlights.setdefault(s["model"], []).append(f"{dim}（{s['score']}%）")
        # 标注最弱
        if len(scores) > 1 and scores[0]["score"] < scores[-1]["score"] - 10:
            weak_model = scores[0]["model"]
            highlights.setdefault(weak_model, [])
            # 在第一个位置插入高亮
            if f"{dim} 最弱" not in highlights[weak_model]:
                pass  # 已包含

    return {
        "models": results,
        "dim_comparison": dim_comparison,
        "highlights": highlights,
    }


def get_all_models_with_data(user: str = "") -> list[str]:
    """获取有评测数据的模型列表"""
    runs = _load_runs(user)
    seen = set()
    result = []
    for r in runs:
        mn = r.get("model_name", "")
        if mn and mn not in seen:
            seen.add(mn)
            result.append(mn)
    return result
