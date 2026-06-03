"""
多模态评测模块
==============
支持图像评测（VQA）和语音评测（ASR）。

图像评测：
  - 调用支持 Vision 的模型 API（多模态 messages）
  - 对比模型回答与参考答案，计算准确率和评分

语音评测：
  - 调用 Whisper 类 API 进行语音转写
  - 计算 WER（词错误率）和 CER（字错误率）
"""

import json
import re
import time
import os
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # type: ignore


# ============================================================================
# 图像评测
# ============================================================================

def _call_llm_vision(model: dict, text: str, image_url: str | list[str],
                     timeout: int = 60) -> str:
    """调用支持 Vision 的模型 API，发送图文消息

    Args:
        model: 模型配置 dict（含 api_base, api_key, model_name）
        text: 文本提示词
        image_url: 图片 URL 或 URL 列表
        timeout: 超时秒数

    Returns:
        与 _call_llm 相同的格式：含元数据前缀或 [API ERROR ...]
    """
    if requests is None:
        return "[ERROR: requests 库未安装]"

    url = model.get("api_base", "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {model.get('api_key', '')}",
        "Content-Type": "application/json",
    }

    # 构造多模态 content
    urls = [image_url] if isinstance(image_url, str) else image_url
    content_parts = [{"type": "text", "text": text}]
    for url_item in urls[:5]:  # 最多 5 张图
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": url_item},
        })

    payload = {
        "model": model.get("model_name") or model.get("name", "gpt-4o"),
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = round(time.time() - start, 2)
        if resp.status_code != 200:
            return f"[API ERROR {resp.status_code}]: {resp.text[:200]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        meta = f"__LATENCY__:{elapsed}|__PROMPT_TOKENS__:{prompt_tokens}|__COMPLETION_TOKENS__:{completion_tokens}|__TOTAL_TOKENS__:{total_tokens}"
        return f"{meta}\n{content}"
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR timeout]: 请求超时 ({elapsed}s)"
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR {type(e).__name__}]: {str(e)[:200]}"


def _parse_llm_response(raw: str) -> dict:
    """解析 LLM 返回（与 eval_runner 兼容）

    Returns:
        {"content": str, "latency": float, "prompt_tokens": int,
         "completion_tokens": int, "total_tokens": int}
    """
    result = {
        "content": raw,
        "latency": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if raw.startswith("[API ERROR") or raw.startswith("[ERROR:"):
        return result

    m = re.match(
        r"__LATENCY__:([\d.]+)\|__PROMPT_TOKENS__:(\d+)\|__COMPLETION_TOKENS__:(\d+)\|__TOTAL_TOKENS__:(\d+)\n(.*)",
        raw, re.DOTALL
    )
    if m:
        result["latency"] = float(m.group(1))
        result["prompt_tokens"] = int(m.group(2))
        result["completion_tokens"] = int(m.group(3))
        result["total_tokens"] = int(m.group(4))
        result["content"] = m.group(5).strip()
    return result


def _eval_vqa(model: dict, questions: list, progress_callback) -> dict:
    """执行 VQA 视觉问答评测

    数据集格式：
        {"id": "...", "image": "<url>", "question": "...", "answer": "..."}

    Returns:
        与 eval_runner 兼容的结果 dict
    """
    total = len(questions)
    correct = 0
    details = []
    total_latency = 0.0
    total_tokens = 0

    for i, q in enumerate(questions):
        img_url = q.get("image", "")
        question_text = q.get("question", "")
        expected = q.get("answer", "").strip()

        prompt = (
            f"请根据图片回答以下问题。只输出答案，不要输出其他内容。\n\n"
            f"问题: {question_text}"
        )

        raw = _call_llm_vision(model, prompt, img_url)
        parsed = _parse_llm_response(raw)
        resp = parsed["content"]
        total_latency += parsed["latency"]
        total_tokens += parsed["total_tokens"]

        # 判断是否正确：精确匹配或包含匹配
        is_correct = False
        if resp:
            resp_clean = resp.strip().lower().rstrip(".,!?")
            exp_clean = expected.lower().rstrip(".,!?")
            if resp_clean == exp_clean:
                is_correct = True
            elif exp_clean in resp_clean:
                is_correct = True

        if is_correct:
            correct += 1

        details.append({
            "id": q["id"],
            "category": q.get("category", "图像理解"),
            "question": question_text,
            "image": img_url[:100] if len(img_url) > 100 else img_url,
            "expected": expected,
            "predicted": resp[:200],
            "correct": is_correct,
            "latency": parsed["latency"],
            "total_tokens": parsed["total_tokens"],
        })

        progress_callback(i + 1, total, f"VQA {i+1}/{total}")

    return {
        "score": round(correct / total * 100, 1) if total else 0,
        "correct": correct,
        "total": total,
        "details": details,
        "avg_latency": round(total_latency / total, 2) if total else 0,
        "total_tokens": total_tokens,
        "is_vision": True,
    }


def _eval_vqa_judge(model: dict, questions: list, progress_callback,
                    judge_model: dict | None = None) -> dict:
    """VQA 评测 + LLM-as-Judge 评分

    当有 Judge 模型时，用 Judge 对模型回答质量进行综合评分。
    适合开放式图像问答（无标准答案）。
    """
    total = len(questions)
    details = []
    total_score = 0.0
    use_judge = judge_model is not None
    total_latency = 0.0
    total_tokens = 0

    for i, q in enumerate(questions):
        img_url = q.get("image", "")
        question_text = q.get("question", "")

        prompt = f"请根据图片回答以下问题。\n\n问题: {question_text}"
        raw = _call_llm_vision(model, prompt, img_url)
        parsed = _parse_llm_response(raw)
        resp = parsed["content"]
        total_latency += parsed["latency"]
        total_tokens += parsed["total_tokens"]

        judge_score = None
        judge_reason = ""
        if use_judge:
            judge_prompt = _build_vqa_judge_prompt(q, resp)
            judge_raw = _call_llm_text(judge_model, judge_prompt, timeout=60)
            judge_parsed = _parse_llm_response(judge_raw)
            total_latency += judge_parsed["latency"]
            total_tokens += judge_parsed["total_tokens"]
            judge_score, judge_reason = _parse_judge_score(judge_parsed["content"])

        is_correct = (judge_score or 0) >= 4 if use_judge else True
        total_score += judge_score or 3.0

        details.append({
            "id": q["id"],
            "category": q.get("category", "图像理解"),
            "question": question_text,
            "image": img_url[:100] if len(img_url) > 100 else img_url,
            "expected": q.get("answer", ""),
            "predicted": resp[:200],
            "correct": is_correct,
            "judge_score": judge_score,
            "judge_reason": judge_reason[:100] if judge_reason else "",
            "latency": parsed["latency"],
            "total_tokens": parsed["total_tokens"],
        })

        progress_callback(i + 1, total, f"VQA(judge) {i+1}/{total}")

    avg_score = round(total_score / total, 1) if total else 0
    return {
        "score": round(avg_score / 5 * 100, 1) if use_judge else 0,
        "correct": sum(1 for d in details if d["correct"]),
        "total": total,
        "avg_judge_score": avg_score,
        "details": details,
        "is_vision": True,
        "avg_latency": round(total_latency / total, 2) if total else 0,
        "total_tokens": total_tokens,
    }


def _build_vqa_judge_prompt(question: dict, model_response: str) -> str:
    """构造 VQA Judge 评分提示词"""
    ref = question.get("answer", "")
    return f"""你是一个专业的图像问答评测员。请对以下模型回答进行评分。

## 评分标准 (1-5分)
- 5分: 完美。回答准确、完整、贴合图片内容。
- 4分: 良好。回答基本正确，略有不足。
- 3分: 一般。部分正确但有明显遗漏或错误。
- 2分: 较差。大部分不正确或偏离图片内容。
- 1分: 很差。完全错误或无法回答。

## 题目
{question['question']}

## 参考答案
{ref if ref else "无"}

## 模型回答
{model_response}

请先给出 1-5 分的分数，然后在下一行给出简短评语。
格式：
分数: <数字>
评语: <你的评语>"""


def _parse_judge_score(text: str) -> tuple:
    """从 Judge 回复解析分数（兼容 eval_runner 格式）"""
    if not text or text.startswith("[API ERROR"):
        return 3.0, "评分失败: " + (text[:50] if text else "无响应")
    m = re.search(r"分数[：:]\s*(\d+(?:\.\d+)?)", text)
    if m:
        score = float(m.group(1))
    else:
        m = re.search(r"\b([1-5])(?:/5|\s*分)", text)
        if m:
            score = float(m.group(1))
        else:
            score = 3.0
    score = max(1.0, min(5.0, score))
    m2 = re.search(r"评语[：:]\s*(.*)", text)
    reason = m2.group(1).strip() if m2 else text.strip()[:200]
    return score, reason


def _call_llm_text(model: dict, prompt: str, timeout: int = 60) -> str:
    """调用纯文本 LLM API（用于 Judge 评分）

    与 eval_runner 中的 _call_llm 相同。
    """
    if requests is None:
        return "[ERROR: requests 库未安装]"

    url = model.get("api_base", "").rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {model.get('api_key', '')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model.get("model_name") or model.get("name", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        elapsed = round(time.time() - start, 2)
        if resp.status_code != 200:
            return f"[API ERROR {resp.status_code}]: {resp.text[:200]}"
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        meta = f"__LATENCY__:{elapsed}|__PROMPT_TOKENS__:{prompt_tokens}|__COMPLETION_TOKENS__:{completion_tokens}|__TOTAL_TOKENS__:{total_tokens}"
        return f"{meta}\n{content}"
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR timeout]: 请求超时 ({elapsed}s)"
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR {type(e).__name__}]: {str(e)[:200]}"


# ============================================================================
# 语音评测
# ============================================================================

def _call_whisper_api(model: dict, audio_path: str, timeout: int = 120) -> str:
    """调用 Whisper 类 API 进行语音转写

    Args:
        model: 模型配置 dict
        audio_path: 音频文件的本地路径或 URL
        timeout: 超时秒数

    Returns:
        转写文本，或 "[ERROR ...]" 前缀的错误消息
    """
    if requests is None:
        return "[ERROR: requests 库未安装]"

    api_base = model.get("api_base", "").rstrip("/")
    api_key = model.get("api_key", "")
    model_name = model.get("model_name") or "whisper-1"

    # 判断是本地文件还是 URL
    is_url = audio_path.startswith(("http://", "https://"))

    url = f"{api_base}/audio/transcriptions"

    start = time.time()
    try:
        if is_url:
            # URL 模式：先下载到临时文件
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_path).suffix or ".tmp")
            try:
                dl_resp = requests.get(audio_path, timeout=timeout)
                dl_resp.raise_for_status()
                tmp.write(dl_resp.content)
                tmp.close()
                local_path = tmp.name
            except Exception as e:
                return f"[API ERROR download]: 音频下载失败: {str(e)[:200]}"

            with open(local_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/" + Path(audio_path).suffix.lstrip("."))}
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"model": model_name, "response_format": "json"}
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            try:
                os.unlink(local_path)
            except Exception:
                pass
        else:
            # 本地文件模式
            with open(audio_path, "rb") as f:
                files = {"file": (Path(audio_path).name, f, "audio/" + Path(audio_path).suffix.lstrip("."))}
                headers = {"Authorization": f"Bearer {api_key}"}
                data = {"model": model_name, "response_format": "json"}
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

        elapsed = round(time.time() - start, 2)
        if resp.status_code != 200:
            return f"[API ERROR {resp.status_code}]: {resp.text[:200]}"
        data = resp.json()
        transcript = data.get("text", "").strip()
        if not transcript:
            return "[API ERROR]: 转写结果为空"

        meta = f"__LATENCY__:{elapsed}|__PROMPT_TOKENS__:0|__COMPLETION_TOKENS__:0|__TOTAL_TOKENS__:0"
        return f"{meta}\n{transcript}"
    except requests.exceptions.Timeout:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR timeout]: 转写超时 ({elapsed}s)"
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return f"[API ERROR {type(e).__name__}]: {str(e)[:200]}"


def _compute_wer(reference: str, hypothesis: str) -> float:
    """计算词错误率 (Word Error Rate)"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Levenshtein 距离（词级别）
    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )
    return round(dp[n][m] / n, 4)


def _compute_cer(reference: str, hypothesis: str) -> float:
    """计算字错误率 (Character Error Rate)"""
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    n = len(ref_chars)
    m = len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return round(dp[n][m] / n, 4)


def _eval_asr(model: dict, questions: list, progress_callback) -> dict:
    """执行 ASR 语音转写评测

    数据集格式：
        {"id": "...", "audio": "<path_or_url>", "transcript": "..."}

    评测指标：
        - WER: 词错误率（越低越好）
        - CER: 字错误率（越低越好）
        - W Acc: 词准确率 (1 - WER)
        - C Acc: 字准确率 (1 - CER)

    Returns:
        与 eval_runner 兼容的结果 dict
    """
    total = len(questions)
    details = []
    total_wer = 0.0
    total_cer = 0.0
    total_latency = 0.0

    for i, q in enumerate(questions):
        audio = q.get("audio", "")
        expected = q.get("transcript", "").strip()

        raw = _call_whisper_api(model, audio)
        parsed = _parse_llm_response(raw)
        resp = parsed["content"]
        total_latency += parsed["latency"]

        # 计算 WER / CER
        wer = _compute_wer(expected, resp)
        cer = _compute_cer(expected, resp)
        total_wer += wer
        total_cer += cer

        # 判断正确：WER < 0.5 算正确
        is_correct = wer < 0.5

        details.append({
            "id": q["id"],
            "category": q.get("category", "语音识别"),
            "audio": audio[:100] if len(audio) > 100 else audio,
            "expected": expected[:200],
            "predicted": resp[:200],
            "correct": is_correct,
            "wer": wer,
            "cer": cer,
            "word_accuracy": round(1 - wer, 4),
            "char_accuracy": round(1 - cer, 4),
            "latency": parsed["latency"],
        })

        progress_callback(i + 1, total, f"ASR {i+1}/{total}")

    avg_wer = round(total_wer / total, 4) if total else 0
    avg_cer = round(total_cer / total, 4) if total else 0
    correct = sum(1 for d in details if d["correct"])

    # WER → 百分制分数：WER 0% → 100分, WER 50% → 50分, WER 100% → 0分
    score = round((1 - avg_wer) * 100, 1)

    return {
        "score": score,
        "correct": correct,
        "total": total,
        "avg_wer": avg_wer,
        "avg_cer": avg_cer,
        "is_asr": True,
        "details": details,
        "avg_latency": round(total_latency / total, 2) if total else 0,
        "total_tokens": 0,
    }


# ============================================================================
# Benchmark 注册表（供 eval_runner 导入使用）
# ============================================================================

BENCHMARK_EVALS = {
    "vqa": _eval_vqa,
    "vqa_judge": _eval_vqa_judge,
    "asr": _eval_asr,
}
