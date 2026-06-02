"""数据挖掘 Pipeline
=====================
从多种来源自动挖掘评测题目，一键注册到平台。

支持来源：
  1. 纯文本 / 知识文档
  2. PDF / DOCX
  3. HuggingFace 数据集
  4. CSV / JSON

架构：
  Source → Chunker → Extractor → Normalizer → 平台 JSON 格式
                                ↑
                          可选 LLM 辅助提取
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# 1. 工具函数
# ════════════════════════════════════════════════════════════════════════

QUESTION_PATTERNS = [
    r"([^。！？\n]+[？?])",                          # 以问号结尾的句子
    r"(?:请|请回答|请解释|请说明|什么是|为什么|如何|怎样|列举|描述|分析|比较|说说)[^。？]*[？?]?",  # 疑问句式
    r"(?:题目[：:]?\s*)(.*?)(?:[。？?]|$)",          # "题目：..."
    r"(?:问题[：:]?\s*)(.*?)(?:[。？?]|$)",          # "问题：..."
]

ANSWER_PATTERNS = [
    r"(?:答案[：:]?\s*)(.*?)(?:[\n\r]|$)",           # "答案：..."
    r"(?:答[：:]?\s*)(.*?)(?:[\n\r]|$)",             # "答：..."
    r"(?:参考答案[：:]?\s*)(.*?)(?:[\n\r]|$)",       # "参考答案：..."
]

MCQ_CHOICE_PATTERN = re.compile(
    r"([A-Da-d])[.、．)\s]\s*([^\nA-Da-d]+)", re.MULTILINE
)


def _split_into_paragraphs(text: str) -> list[str]:
    """将文本分割成段落"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _extract_mcq_choices(text: str) -> dict[str, str] | None:
    """从文本中提取选项（A. xxx / B. xxx）"""
    matches = MCQ_CHOICE_PATTERN.findall(text)
    if len(matches) >= 2:
        return {m[0].upper(): m[1].strip() for m in matches}
    return None


def _is_likely_question(text: str) -> bool:
    """判断一段文本是否可能是问句"""
    return any(re.search(p, text) for p in QUESTION_PATTERNS)


def _find_answer(text: str, paragraph: str = "") -> str:
    """从文本中寻找答案"""
    # 先在本段落找
    for p in ANSWER_PATTERNS:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()
    # 在下一段落找
    if paragraph:
        for p in ANSWER_PATTERNS:
            m = re.search(p, paragraph)
            if m:
                return m.group(1).strip()
    return ""


# ════════════════════════════════════════════════════════════════════════
# 2. 纯文本挖掘器
# ════════════════════════════════════════════════════════════════════════


def mine_from_text(text: str,
                   max_questions: int = 100,
                   # 可选注入 LLM 调用函数
                   llm_extract_fn: Callable | None = None) -> list[dict]:
    """从纯文本中挖掘评测题目

    Args:
        text: 输入文本
        max_questions: 最多挖掘多少个
        llm_extract_fn: fn(text) -> [{"question":"", "answer":"", ...}]
                        如果提供，优先使用 LLM 提取

    Returns:
        [{"id": int, "question": str, "answer": str, "category": str,
          "choices": dict|None, "source": str}]
    """
    # 如果用 LLM 提取
    if llm_extract_fn:
        try:
            results = llm_extract_fn(text)
            if isinstance(results, list) and len(results) > 0:
                for i, r in enumerate(results):
                    r.setdefault("id", str(i))
                    r.setdefault("category", "通用")
                    r.setdefault("choices", None)
                    r.setdefault("source", "text_llm")
                return results[:max_questions]
        except Exception as e:
            logger.warning(f"LLM 提取失败，回退到规则提取: {e}")

    # 规则提取
    results = []
    paragraphs = _split_into_paragraphs(text)

    # 清理函数：去掉题目文本中的"选项："和"答案："部分
    def _clean_question(q: str) -> str:
        q = re.sub(r"\n?答案[：:].*", "", q)  # 去掉"答案：B"
        q = re.sub(r"\n?选项[：:].*", "", q)   # 去掉"选项："
        q = re.sub(r"^(?:问题|题目|Question)\s*\d*[.、．:：）)]?\s*", "", q)  # "问题1："
        return q.strip()

    seen = set()
    for i, para in enumerate(paragraphs):
        if len(results) >= max_questions:
            break
        if len(para) < 10:
            continue

        # 检查是否为选择题（含 A/B/C/D 选项）
        choices = _extract_mcq_choices(para)
        question_text = para
        answer = ""

        if choices:
            # 去掉选项部分得到题目文本
            question_clean = re.sub(r"[A-Da-d][.、．)\s]\s*[^\nA-Da-d]+", "", para)
            question_clean = question_clean.strip()
            if question_clean:
                question_text = question_clean

            # 寻找标注的答案
            answer = _find_answer(para)
            # 如果没找到，把第一个选项作为默认答案
            if not answer:
                answer = list(choices.keys())[0]

            dedup_key = question_text[:60]
            if dedup_key not in seen:
                seen.add(dedup_key)
                # 清理题目文本
                cleaned = _clean_question(question_text)
                results.append({
                    "id": str(len(results)),
                    "question": cleaned,
                    "answer": answer,
                    "choices": choices,
                    "category": "通用",
                    "source": "text_rule",
                })
        elif _is_likely_question(para):
            dedup_key = para[:60]
            if dedup_key not in seen:
                seen.add(dedup_key)
                # 尝试在下一个段落找答案
                next_para = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
                answer = _find_answer(para, next_para)
                # 检查下段是否含选项
                next_choices = _extract_mcq_choices(next_para)
                if next_choices:
                    # 这对是选择题
                    results.append({
                        "id": str(len(results)),
                        "question": para,
                        "answer": answer or list(next_choices.keys())[0],
                        "choices": next_choices,
                        "category": "通用",
                        "source": "text_rule",
                    })
                else:
                    results.append({
                        "id": str(len(results)),
                        "question": para,
                        "answer": answer,
                        "choices": None,
                        "category": "通用",
                        "source": "text_rule",
                    })

    return results


# ════════════════════════════════════════════════════════════════════════
# 3. 文件挖掘器
# ════════════════════════════════════════════════════════════════════════


def _extract_text_from_pdf(filepath: str) -> str:
    """从 PDF 提取文本（可选依赖 pymupdf / pdfminer）"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(filepath)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except ImportError:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except ImportError:
        pass
    try:
        import pdfminer
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(filepath)
    except ImportError:
        pass
    raise ImportError(
        "需要安装 PDF 提取库: pip install pymupdf 或 pip install pdfplumber"
    )


def _extract_text_from_docx(filepath: str) -> str:
    """从 DOCX 提取文本（可选依赖 python-docx）"""
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        raise ImportError("需要安装: pip install python-docx")


def _extract_text_from_csv(filepath: str,
                           question_col: str = "question",
                           answer_col: str = "answer") -> list[dict]:
    """从 CSV 提取评测数据"""
    results = []
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            q = row.get(question_col, "").strip()
            a = row.get(answer_col, "").strip()
            if q:
                choices = None
                # 检查是否有选项列 (choice_a, choice_b, ...)
                choice_cols = [k for k in row.keys()
                               if k.lower().startswith("choice_") or k.startswith("选项")]
                if choice_cols:
                    choices = {}
                    for ck in sorted(choice_cols):
                        label = ck.replace("choice_", "").replace("选项", "").upper()
                        choices[label] = row[ck]
                results.append({
                    "id": str(i),
                    "question": q,
                    "answer": a,
                    "choices": choices,
                    "category": row.get("category", row.get("分类", "通用")),
                    "source": "csv",
                })
    return results


def _extract_text_from_json(filepath: str,
                            question_key: str = "question",
                            answer_key: str = "answer") -> list[dict]:
    """从 JSON 文件提取评测数据"""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    results = []
    for i, item in enumerate(data):
        q = item.get(question_key) or item.get("问题") or item.get("input") or ""
        if not q:
            continue
        a = item.get(answer_key) or item.get("答案") or item.get("output") or ""
        choices = item.get("choices", item.get("选项", None))
        if isinstance(choices, list):
            labels = ["A", "B", "C", "D"]
            choices = {labels[j]: c for j, c in enumerate(choices) if c}
        results.append({
            "id": str(i),
            "question": q,
            "answer": a,
            "choices": choices,
            "category": item.get("category", item.get("分类", "通用")),
            "source": "json",
        })
    return results


def mine_from_file(filepath: str, **kwargs) -> list[dict]:
    """从文件挖掘评测题目

    Args:
        filepath: 文件路径
        **kwargs: 传递给具体挖掘器的参数

    Returns:
        [{"id", "question", "answer", "choices", "category", "source"}]
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        text = _extract_text_from_pdf(filepath)
        return mine_from_text(text, **kwargs)
    elif ext == ".docx":
        text = _extract_text_from_docx(filepath)
        return mine_from_text(text, **kwargs)
    elif ext == ".csv":
        return _extract_text_from_csv(filepath, **kwargs)
    elif ext == ".json":
        return _extract_text_from_json(filepath, **kwargs)
    elif ext == ".txt":
        with open(filepath, encoding="utf-8") as f:
            text = f.read()
        return mine_from_text(text, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")


# ════════════════════════════════════════════════════════════════════════
# 4. HF 数据集挖掘器
# ════════════════════════════════════════════════════════════════════════


def mine_from_hf(hf_path: str,
                 config: str | None = None,
                 split: str = "train",
                 sample_size: int = 50,
                 question_field: str | None = None,
                 answer_field: str | None = None) -> list[dict]:
    """从 HuggingFace 数据集挖掘评测题目

    用 LLM 从 HF 数据行中提取 Q&A 对。

    Args:
        hf_path: HF 数据集路径
        config: 子集配置
        split: 数据分割
        sample_size: 采样数量
        question_field: 指定问题字段（自动检测）
        answer_field: 指定答案字段（自动检测）

    Returns:
        [{"id", "question", "answer", "choices", "category", "source"}]
    """
    from datasets import load_dataset

    try:
        if config:
            ds = load_dataset(hf_path, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(hf_path, split=split, trust_remote_code=True)
    except Exception:
        # 尝试其他 split
        for alt in ["test", "validation", "train"]:
            try:
                if config:
                    ds = load_dataset(hf_path, config, split=alt, trust_remote_code=True)
                else:
                    ds = load_dataset(hf_path, split=alt, trust_remote_code=True)
                break
            except Exception:
                continue
        else:
            raise ValueError(f"无法加载 HF 数据集: {hf_path}")

    # 自动检测字段
    sample = ds[0] if len(ds) > 0 else {}
    text_fields = [k for k, v in sample.items() if isinstance(v, str) and len(v) > 10]

    if question_field is None:
        for key in ["question", "text", "input", "sentence", "ctx", "prompt"]:
            if key in sample:
                question_field = key
                break
        if not question_field and text_fields:
            question_field = text_fields[0]

    if answer_field is None:
        for key in ["answer", "label", "output", "target", "response", "completion"]:
            if key in sample:
                answer_field = key
                break
        if not answer_field and len(text_fields) > 1:
            answer_field = text_fields[1]

    import random
    indices = list(range(len(ds)))
    if len(indices) > sample_size:
        indices = random.sample(indices, sample_size)

    results = []
    for idx in indices:
        row = ds[idx]
        q = str(row.get(question_field, "")) if question_field else ""
        a = str(row.get(answer_field, "")) if answer_field else ""
        choices = None

        # 检测选项字段
        choice_fields = [k for k in sample.keys()
                         if k.startswith("choice") or k in ("choices", "options", "endings")]
        for cf in choice_fields:
            val = row[cf]
            if isinstance(val, dict):
                choices = val
                break
            elif isinstance(val, list):
                labels = ["A", "B", "C", "D"]
                choices = {labels[j]: str(v) for j, v in enumerate(val)}
                break

        if q:
            results.append({
                "id": str(idx),
                "question": q,
                "answer": a,
                "choices": choices,
                "category": str(row.get("category", "通用")),
                "source": f"hf:{hf_path}",
            })

    return results


# ════════════════════════════════════════════════════════════════════════
# 5. 注册到平台
# ════════════════════════════════════════════════════════════════════════


def register_as_benchmark(items: list[dict],
                          benchmark_name: str,
                          datasets_dir: str | Path,
                          suffix: str = "_custom") -> str:
    """将挖掘结果注册为平台 benchmark JSON 文件

    Args:
        items: [{"id", "question", "answer", "choices", "category"}, ...]
        benchmark_name: benchmark 标识符（如 "code_qa"）
        datasets_dir: data/datasets/ 目录
        suffix: _custom / _official / _extended

    Returns:
        保存的文件名
    """
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{benchmark_name}{suffix}.json"
    filepath = datasets_dir / filename

    # 确保所有条目都有 id
    for i, item in enumerate(items):
        item.setdefault("id", str(i))
        item.setdefault("category", "通用")

    # 检查是否已存在
    existing = []
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            existing = json.load(f)

    combined = existing + items
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    logger.info(f"已注册 {len(combined)} 题到 {filename} (新增 {len(items)})")
    return filename
