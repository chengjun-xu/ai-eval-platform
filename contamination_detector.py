"""
数据泄露/污染检测模块
====================
检测评测数据集中是否存在数据泄露/污染风险，包括：
1. n-gram overlap 分析 — 题目之间、题目与常见训练集的重复度
2. 自污染检测 — 数据集内部相似题判断
3. 记忆化检测 — 超长精确匹配检查（预示模型可能"背过"答案）
4. 污染缓解分数 — 汇总风险评估

面试亮点：展示对"可持续评测"中数据泄露检测机制的理解。
"""
import re
import hashlib
from typing import Any

# ---------------------------------------------------------------------------
# n-gram 工具
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """简单中英文分词"""
    text = text.lower().strip()
    # 中文按字切 + 英文按词切
    tokens = []
    buffer = ""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf':
            if buffer:
                tokens.extend(buffer.split())
                buffer = ""
            tokens.append(ch)  # 单字
        elif ch.isalnum() or ch in "-_.":
            buffer += ch
        else:
            if buffer:
                tokens.extend(buffer.split())
                buffer = ""
            if ch.strip():
                tokens.append(ch)
    if buffer:
        tokens.extend(buffer.split())
    return tokens


def _ngrams(tokens: list[str], n: int) -> set[str]:
    """提取 n-gram 集合"""
    return set(" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard 相似度"""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── TF-IDF 语义相似度 ─────────────────────────────────────────────────────
# 使用 sklearn 的 TfidfVectorizer（如果可用），否则用简单的词频 TF 作为降级

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的 TF-IDF 余弦相似度

    使用 sklearn 的 TfidfVectorizer（精确），降级模式使用词袋 Jaccard。
    针对中文使用字符级 n-gram（1-4字）以捕获语义。
    """
    if _HAS_SKLEARN:
        import numpy as np
        try:
            # 中文用字符 n-gram，英文用单词 n-gram
            # max_df 根据文档数动态调整，避免在少文本场景下误过滤
            vec = _Tfidf(stop_words=None, max_features=2000,
                        analyzer='char', ngram_range=(2, 4))
            tfidf = vec.fit_transform([text_a, text_b])
            tfidf_dense = tfidf.toarray()
            dot = np.dot(tfidf_dense[0], tfidf_dense[1])
            norm = np.linalg.norm(tfidf_dense[0]) * np.linalg.norm(tfidf_dense[1])
            return float(dot / norm) if norm > 0 else 0.0
        except Exception:
            pass
    # 降级：词袋 Jaccard
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    return _jaccard_similarity(tokens_a, tokens_b)


def _semantic_self_contamination(texts: list[str]) -> dict:
    """使用 TF-IDF 向量检测数据集的语义自相似度

    返回：
        {
            "avg_semantic_similarity": float,  # 平均两两语义相似度
            "max_semantic_similarity": float,   # 最高相似度
            "similar_pair_count": int,          # 高度相似对的数量 (>0.7)
            "note": str,                        # 解释
        }
    """
    if len(texts) < 2:
        return {"avg_semantic_similarity": 0.0, "max_semantic_similarity": 0.0,
                "similar_pair_count": 0, "note": "数据量不足"}

    n = len(texts)
    sims = []
    # 抽样计算，避免 O(n²)
    max_pairs = min(n * (n - 1) // 2, 100)
    count = 0
    high_sim_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if count >= max_pairs:
                break
            sim = _tfidf_cosine_similarity(texts[i], texts[j])
            sims.append(sim)
            if sim > 0.7:
                high_sim_pairs += 1
            count += 1
        if count >= max_pairs:
            break

    avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
    max_sim = round(max(sims), 4) if sims else 0.0

    note = ""
    if avg_sim > 0.5:
        note = "⚠️ 数据集内部语义相似度偏高，可能存在冗余题目"
    elif avg_sim > 0.3:
        note = "📋 数据集中度正常，部分题目语义相近"
    else:
        note = "✅ 数据集题目间语义区分度较好"

    return {
        "avg_semantic_similarity": avg_sim,
        "max_semantic_similarity": max_sim,
        "similar_pair_count": high_sim_pairs,
        "note": note,
    }


# ---------------------------------------------------------------------------
# 单题污染检测
# ---------------------------------------------------------------------------

# 常见的高泄露风险 n-gram 长度
# 短 n-gram 重叠说明问题主题相似，长 n-gram 重叠说明可能是"背过"的题目
CONTAMINATION_LEVELS = [
    (3, 0.1, "低"),   # 3-gram 重叠 >10% → 弱信号
    (5, 0.15, "中"),  # 5-gram 重叠 >15% → 中风险
    (9, 0.1, "高"),   # 9-gram 重叠 >10% → 高风险
    (13, 0.05, "严重"), # 13-gram 重叠 >5% → 严重泄露
]


def _get_question_text(question: dict) -> str:
    """从统一格式的题目中提取文本"""
    text = question.get("question", "") or question.get("description", "") or ""
    # 加上选项
    choices = question.get("choices", {})
    if choices:
        for k, v in choices.items():
            text += f" {k}. {v}"
    return text


def check_contamination(question_text: str, reference_texts: list[str]) -> dict:
    """检测单题对参考语料库的泄露风险

    Returns:
        {
            "max_risk": "低"/"中"/"高"/"严重"/"安全",
            "n_gram_results": {n: {"overlap": float, "n_shared": int, "max_source": str}},
            "overall_score": float (0~1, 越高越危险),
            "exact_match_found": bool,
        }
    """
    q_tokens = _tokenize(question_text)
    if len(q_tokens) < 5:
        return {"max_risk": "安全", "n_gram_results": {}, "overall_score": 0.0, "exact_match_found": False}

    results = {}
    max_risk_level = "安全"
    max_risk_score = 0.0
    exact_match = False

    for n, threshold, risk_label in CONTAMINATION_LEVELS:
        if len(q_tokens) < n:
            continue
        q_ngrams = _ngrams(q_tokens, n)
        best_overlap = 0.0
        best_source = ""
        for ref_text in reference_texts:
            ref_tokens = _tokenize(ref_text)
            if len(ref_tokens) < n:
                continue
            ref_ngrams = _ngrams(ref_tokens, n)
            sim = _jaccard_similarity(q_ngrams, ref_ngrams)
            if sim > best_overlap:
                best_overlap = sim
                best_source = ref_text[:60]

        # 检查精确匹配
        if n >= 9 and best_overlap >= 0.8:
            exact_match = True

        level_score = 0.0
        if best_overlap > threshold:
            level_score = min(best_overlap / threshold, 1.0)
            # 更新风险等级
            risk_order = {"低": 1, "中": 2, "高": 3, "严重": 4}
            if risk_order.get(risk_label, 0) > risk_order.get(max_risk_level, 0):
                max_risk_level = risk_label

        results[str(n)] = {
            "overlap": round(best_overlap, 4),
            "n_shared": 0,  # 不精确计算共享数，太慢
            "max_source": best_source[:40],
            "risk_level": risk_label if best_overlap > threshold else "安全",
        }
        max_risk_score = max(max_risk_score, level_score)

    return {
        "max_risk": max_risk_level if exact_match else (max_risk_level if max_risk_score >= 0.3 else "安全"),
        "n_gram_results": results,
        "overall_score": round(max_risk_score, 3),
        "exact_match_found": exact_match,
    }


# ---------------------------------------------------------------------------
# 数据集整体检测
# ---------------------------------------------------------------------------

def analyze_dataset(dataset_items: list[dict],
                    reference_corpus: list[str] | None = None) -> dict:
    """对整个数据集进行泄露分析

    Args:
        dataset_items: 数据集条目列表
        reference_corpus: 可选的参考语料（如训练集样本）

    Returns:
        {
            "dataset_level": {各项汇总指标},
            "item_level": {id -> 各题污染详情},
            "risk_distribution": {风险等级: 数量},
            "overall_risk": 低/中/高,
            "self_contamination": {内部相似度},
        }
    """
    question_texts = []
    question_map = {}  # id -> text

    for item in dataset_items:
        qid = item.get("id", str(hash(str(item))))
        text = _get_question_text(item)
        question_texts.append(text)
        question_map[qid] = text

    # 1. 自污染检测（题目间重复度）
    self_contamination = _check_self_contamination(question_texts)

    # 2. 语义自相似度检测（TF-IDF）
    semantic_sim = _semantic_self_contamination(question_texts)

    # 3. 对参考语料的污染检测
    corpus = reference_corpus or []
    item_results = {}
    risk_dist = {"安全": 0, "低": 0, "中": 0, "高": 0, "严重": 0}

    for qid, text in question_map.items():
        if corpus:
            result = check_contamination(text, corpus)
        else:
            # 无外部语料时，用内部其他题目作为参考
            other_texts = [t for i, t in enumerate(question_texts)
                          if t != text]
            result = check_contamination(text, other_texts)
        item_results[qid] = result
        risk_dist[result["max_risk"]] = risk_dist.get(result["max_risk"], 0) + 1

    # 3. 汇总
    scores = [r["overall_score"] for r in item_results.values()]
    avg_score = sum(scores) / len(scores) if scores else 0
    max_score = max(scores) if scores else 0

    # 总体风险
    risk_counts = {k: v for k, v in risk_dist.items() if v > 0}
    high_risk_count = risk_dist.get("高", 0) + risk_dist.get("严重", 0)
    total = len(question_texts)

    if high_risk_count / total >= 0.1:
        overall_risk = "高"
    elif high_risk_count / total >= 0.05:
        overall_risk = "中"
    elif high_risk_count > 0:
        overall_risk = "低"
    else:
        overall_risk = "安全"

    return {
        "data_size": total,
        "overall_risk": overall_risk,
        "overall_contamination_score": round(avg_score, 3),
        "max_contamination_score": round(max_score, 3),
        "high_risk_count": high_risk_count,
        "risk_distribution": risk_dist,
        "item_level": item_results,
        "self_contamination": self_contamination,
        "semantic_similarity": semantic_sim,
        "suggestion": _generate_suggestion(overall_risk, high_risk_count, total),
    }


def _check_self_contamination(texts: list[str]) -> dict:
    """题目间自污染检测"""
    if len(texts) < 2:
        return {"similar_pairs": 0, "max_similarity": 0.0, "note": "数据量不足"}

    # 用 5-gram Jaccard 判断两两相似度
    tokenized = [_tokenize(t) for t in texts]
    ngrams_list = [_ngrams(t, 5) for t in tokenized]

    similar_pairs = 0
    max_sim = 0.0
    total_pairs = 0
    N = len(ngrams_list)

    for i in range(N):
        for j in range(i+1, N):
            sim = _jaccard_similarity(ngrams_list[i], ngrams_list[j])
            total_pairs += 1
            if sim > 0.3:
                similar_pairs += 1
                max_sim = max(max_sim, sim)

    return {
        "similar_pairs": similar_pairs,
        "max_similarity": round(max_sim, 3),
        "total_pairs": total_pairs,
        "similarity_rate": round(similar_pairs / total_pairs, 4) if total_pairs else 0,
    }


def _generate_suggestion(risk: str, high_count: int, total: int) -> str:
    """生成可操作的建议"""
    suggestions = []
    if risk in ("高", "严重"):
        suggestions.append(f"⚠️ 发现 {high_count}/{total} 题存在高泄露风险，建议重建或替换这些题目。")
        suggestions.append("💡 小样本集应每季度更新一次，避免模型通过数据记忆提升分数。")
    elif risk == "中":
        suggestions.append(f"⚠️ 有 {high_count}/{total} 题存在中等风险，建议重点检查这些题目是否有公开来源。")
    elif risk == "低":
        suggestions.append(f"📋 仅有 {high_count} 题有微弱泄露信号，整体安全，建议持续监控。")
    else:
        suggestions.append("✅ 数据集未检测到明显泄露，可持续使用。")
    suggestions.append("📊 推荐做法：小样本高敏感集 + 大样本回归集的组合策略。")
    return "\n".join(suggestions)


# ---------------------------------------------------------------------------
# 检测常用已知污染源（示例库）
# ---------------------------------------------------------------------------

_KNOWN_CONTAMINATED_BENCHMARKS = {
    "mmlu": {"risk": "中", "note": "MMLU 部分题目已在 GPT-4/GPT-4o 等训练集中出现"},
    "gsm8k": {"risk": "低", "note": "GSM8K 题目逻辑独特，泄露风险较低"},
    "humaneval": {"risk": "高", "note": "HumanEval 题目已在大量代码训练集中出现，建议配合变体使用"},
}


def get_known_contamination_risk(benchmark_name: str) -> dict:
    """获取已知 benchmark 的业界公认泄露风险"""
    key = benchmark_name.lower().replace(" ", "_")
    return _KNOWN_CONTAMINATED_BENCHMARKS.get(key, {"risk": "未知", "note": ""})


# ---------------------------------------------------------------------------
# 快速工具：对评测报告中的 Bad Case 做归因辅助
# ---------------------------------------------------------------------------

def describe_contamination_in_report(contamination_result: dict,
                                     benchmark_name: str = "") -> str:
    """生成用于评测报告的一段总结"""
    if not contamination_result:
        return ""

    overall = contamination_result.get("overall_risk", "安全")
    score = contamination_result.get("overall_contamination_score", 0)

    known = get_known_contamination_risk(benchmark_name)
    known_note = f"\n业界共识：{known['note']}" if known.get("note") else ""

    return (
        f"**数据泄露风险：{overall}**（污染指数 {score}）"
        f"{known_note}"
        f"\n高/严重风险题数：{contamination_result.get('high_risk_count', 0)} / {contamination_result.get('data_size', 0)}"
    )


if __name__ == "__main__":
    # 简单自测
    test_items = [
        {"id": "q1", "question": "患者男性65岁，有20年吸烟史，刺激性干咳痰中带血丝",
         "choices": {"A": "肺脓肿", "B": "肺结核", "C": "肺癌", "D": "错构瘤"}},
        {"id": "q2", "question": "某患者心悸手抖体重下降3个月，甲状腺II度肿大",
         "choices": {"A": "亚甲炎", "B": "Graves病", "C": "桥本", "D": "腺瘤"}},
    ]
    result = analyze_dataset(test_items)
    print(f"Overall risk: {result['overall_risk']}")
    print(f"Score: {result['overall_contamination_score']}")
    print(f"Self-contamination: {result['self_contamination']}")
