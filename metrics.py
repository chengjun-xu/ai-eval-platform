"""
客观评测指标：纯 Python 实现，无外部依赖。

提供：ROUGE-L, BLEU, Exact Match, F1 Score
适用于开放生成任务的文本质量评估。
"""

import re
import math
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """中文/英文通用的简单分词：中文字单独成词，英文按空格分词"""
    text = text.lower().strip()
    # 在中文和英文之间插入空格
    result = []
    for c in text:
        if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf':
            # 中文汉字前后加空格隔离
            result.append(f' {c} ')
        elif c.isalnum():
            result.append(c)
        else:
            result.append(' ')
    text = ''.join(result)
    tokens = text.split()
    return [t for t in tokens if t]


def _lcs_length(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _normalize(text: str) -> str:
    """标准化文本：去标点、去空格、小写"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\u4e00-\u9fff\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ═══════════════════════════════════════════════════════════════
# ROUGE-L
# ═══════════════════════════════════════════════════════════════

def rouge_l(reference: str, hypothesis: str) -> dict:
    """计算 ROUGE-L（基于最长公共子序列）

    Returns: {"precision": float, "recall": float, "f1": float}
    """
    ref_tokens = _tokenize(_normalize(reference))
    hyp_tokens = _tokenize(_normalize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ═══════════════════════════════════════════════════════════════
# BLEU (简化版，1-gram + 2-gram)
# ═══════════════════════════════════════════════════════════════

def bleu(reference: str, hypothesis: str, max_n: int = 2) -> dict:
    """计算 BLEU 分数

    Args:
        max_n: 最大 n-gram 阶数（默认 2，够用）

    Returns: {"bleu": float, "precisions": [float], "brevity_penalty": float}
    """
    ref_tokens = _tokenize(_normalize(reference))
    hyp_tokens = _tokenize(_normalize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return {"bleu": 0.0, "precisions": [0.0], "brevity_penalty": 0.0}

    precisions = []
    for n in range(1, max_n + 1):
        # 提取 n-gram
        ref_ngrams = Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)
        )

        # 计算精确率
        matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0.0)

    # Brevity Penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if hyp_tokens else 0.0

    # BLEU = BP * exp(sum(1/n * log(p_n)))
    if all(p == 0 for p in precisions):
        bleu_score = 0.0
    else:
        log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n
        bleu_score = bp * math.exp(log_avg)

    return {
        "bleu": round(bleu_score, 4),
        "precisions": [round(p, 4) for p in precisions],
        "brevity_penalty": round(bp, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Exact Match
# ═══════════════════════════════════════════════════════════════

def exact_match(reference: str, hypothesis: str) -> bool:
    """精确匹配（标准化后比较）"""
    return _normalize(reference) == _normalize(hypothesis)


# ═══════════════════════════════════════════════════════════════
# F1 Score (Token-level)
# ═══════════════════════════════════════════════════════════════

def f1_score(reference: str, hypothesis: str) -> dict:
    """计算 Token 级 F1 分数

    Returns: {"precision": float, "recall": float, "f1": float}
    """
    ref_tokens = _tokenize(_normalize(reference))
    hyp_tokens = _tokenize(_normalize(hypothesis))

    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)

    common = sum((ref_counter & hyp_counter).values())
    precision = common / len(hyp_tokens) if hyp_tokens else 0.0
    recall = common / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ═══════════════════════════════════════════════════════════════
# 综合指标
# ═══════════════════════════════════════════════════════════════

def compute_all_metrics(reference: str, hypothesis: str) -> dict:
    """一次性计算所有客观指标"""
    return {
        "rouge_l": rouge_l(reference, hypothesis),
        "bleu": bleu(reference, hypothesis),
        "exact_match": exact_match(reference, hypothesis),
        "f1": f1_score(reference, hypothesis),
    }
