#!/usr/bin/env python3
"""
下载官方 Benchmark 数据集并转换为平台 JSON 格式。
支持：MMLU（57学科）、GSM8K、HumanEval
从国内镜像 hf-mirror.com 下载，速度更快。
"""

import json
import os
import sys
import urllib.request
import urllib.error
import io
import gzip
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
MIRROR = "https://hf-mirror.com"


def download_json(url: str, timeout: int = 30) -> dict | list:
    """下载 JSON 数据"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        # 如果是 gzip 压缩
        if resp.headers.get("Content-Encoding") == "gzip":
            data = gzip.decompress(data)
        return json.loads(data)


def download_binary(url: str, timeout: int = 60) -> bytes:
    """下载二进制文件"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ═══════════════════════════════════════════════════════════════
# MMLU — 57学科，每科 test 分卷
# 来源：https://github.com/hendrycks/test（原始数据）
# HF 上 lukaemon/mmlu 把每科拆成了单独的 config
# ═══════════════════════════════════════════════════════════════

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


def _mmlu_parquet_url(subject: str, split: str = "test") -> str:
    """MMLU 数据集以 parquet 格式存储在 HuggingFace"""
    # 用 HF API 获取 parquet 文件列表
    return f"{MIRROR}/datasets/lukaemon/mmlu/raw/main/data/{subject}/{split}-00000-of-00001.parquet"


def _fetch_hf_dataset_via_api(dataset_id: str, config: str, split: str) -> list[dict]:
    """通过 HuggingFace Datasets Server API 获取数据（不需要安装 datasets 库）

    API 文档：https://huggingface.co/docs/datasets-server/en/quick_start
    """
    import json
    url = f"{MIRROR}/api/datasets/{dataset_id}/parquet/{config}/{split}"
    print(f"    获取数据集文件列表...")
    try:
        meta = download_json(url, timeout=15)
    except Exception as e:
        print(f"    ⚠️ API 请求失败: {e}")
        return []

    rows = []
    # 尝试从 HF Datasets Server 的 /rows 端点逐批获取
    rows_url = f"{MIRROR}/datasets/{dataset_id}/raw/data/{split}/{config}.jsonl"
    # 实际上 MMLU 数据每个 subject 的 test split 以 parquet 存储
    # 我们用另一种方式：直接从 GitHub 原始数据仓库获取

    # 回退方案：从 hendrycks/test GitHub 仓库获取 CSV
    return _fetch_mmlu_from_github(subject=config)


def _fetch_mmlu_from_github(subject: str) -> list[dict]:
    """从 MMLU 原始 GitHub 仓库下载 CSV 格式数据"""
    import csv
    url = f"https://raw.githubusercontent.com/hendrycks/test/master/{subject}_test.csv"
    print(f"    {subject}_test.csv...", end=" ")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
            reader = csv.DictReader(content.splitlines(),
                                    fieldnames=["input", "A", "B", "C", "D", "target"])
            questions = []
            for row in reader:
                choices = {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]}
                questions.append({
                    "id": f"mmlu_{subject}_{len(questions)}",
                    "category": subject,
                    "question": row["input"],
                    "choices": choices,
                    "answer": row["target"].strip(),
                })
            print(f"✅ {len(questions)} 题")
            return questions
    except Exception as e:
        print(f"❌ {e}")
        return []


def download_mmlu() -> dict[str, list]:
    """下载全部 57 学科的 MMLU test 数据"""
    print(f"\n{'='*50}")
    print(f"📥 下载 MMLU（57学科）")
    print(f"{'='*50}")

    all_questions = []
    total = 0
    for subject in MMLU_SUBJECTS:
        questions = _fetch_mmlu_from_github(subject)
        all_questions.extend(questions)
        total += len(questions)
        if total > 0:
            print(f"  已累计: {total} 题")

    if not all_questions:
        print("❌ MMLU 下载失败，请检查网络")
        return {}

    # 保存为平台 JSON 格式
    output = DATA_DIR / "mmlu_official.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n✅ MMLU 已保存: {output} ({len(all_questions)} 题, 57学科)")

    return {"mmlu_official": all_questions}


# ═══════════════════════════════════════════════════════════════
# GSM8K — 小学数学题
# 来源：https://github.com/openai/grade-school-math
# ═══════════════════════════════════════════════════════════════

def download_gsm8k() -> dict[str, list]:
    print(f"\n{'='*50}")
    print(f"📥 下载 GSM8K（小学数学推理）")
    print(f"{'='*50}")

    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    questions = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
            for i, line in enumerate(data.strip().splitlines()):
                if not line.strip():
                    continue
                item = json.loads(line)
                # 从 answer 提取最终数字
                answer = item.get("answer", "")
                # 格式："There are 15 apples. #### 15"
                if "####" in answer:
                    expected = answer.split("####")[-1].strip()
                else:
                    expected = answer.strip()
                questions.append({
                    "id": f"gsm8k_{i}",
                    "category": "math",
                    "question": item.get("question", ""),
                    "answer": expected,
                    "reference_answer": answer,
                })
        print(f"✅ {len(questions)} 题")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return {}

    output = DATA_DIR / "gsm8k_official.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"✅ GSM8K 已保存: {output} ({len(questions)} 题)")
    return {"gsm8k_official": questions}


# ═══════════════════════════════════════════════════════════════
# HumanEval — 代码补全
# 来源：https://github.com/openai/human-eval
# ═══════════════════════════════════════════════════════════════

def download_humaneval() -> dict[str, list]:
    print(f"\n{'='*50}")
    print(f"📥 下载 HumanEval（代码生成）")
    print(f"{'='*50}")

    # 从 GitHub 直接下 JSONL
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    tasks = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            # gzip 解压
            import gzip
            data = gzip.decompress(raw).decode("utf-8")
            for line in data.strip().splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                tasks.append({
                    "id": f"humaneval_{item['task_id'].split('/')[-1]}",
                    "category": "code_generation",
                    "description": item.get("prompt", "")[:80],
                    "prompt": item.get("prompt", ""),
                    "entry_point": item.get("entry_point", ""),
                    "test": item.get("test", ""),
                    "canonical_solution": item.get("canonical_solution", ""),
                })
        print(f"✅ {len(tasks)} 题")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return {}

    output = DATA_DIR / "humaneval_official.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"✅ HumanEval 已保存: {output} ({len(tasks)} 题)")
    return {"humaneval_official": tasks}


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("📦 官方 Benchmark 数据集下载器")
    print("=" * 50)

    datasets = {}

    # MMLU（57学科，~14000题）
    mmlu_data = download_mmlu()
    datasets.update(mmlu_data)

    # GSM8K（~1300题）
    gsm8k_data = download_gsm8k()
    datasets.update(gsm8k_data)

    # HumanEval（164题）
    humaneval_data = download_humaneval()
    datasets.update(humaneval_data)

    print(f"\n{'='*50}")
    print(f"📊 汇总")
    print(f"{'='*50}")
    for name, data in datasets.items():
        print(f"  ✅ {name}: {len(data)} 题")
    print(f"\n📁 保存位置: {DATA_DIR}")
    print("✅ 完成！现在可以在平台上选择官方版本进行评测了。")
