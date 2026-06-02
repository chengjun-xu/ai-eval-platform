"""OpenCompass 评测适配器
=========================
将 OpenCompass 作为可选的评测后端。平台优先使用本地/API 评测，
当选择 OC-native benchmark 时，自动转给 OpenCompass 执行。

架构：
  平台选 benchmark → 判断 source 类型
    ├─ local/huggingface → 本地 eval_runner
    └─ opencompass → opencompass_adapter.run_eval()
                      ├─ 生成 OC config
                      ├─ 运行 CLI
                      └─ 解析结果 → 返回平台统一格式

零依赖：平台不依赖 opencompass，适配器自动检测可用性。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# 1. 可用性检测
# ════════════════════════════════════════════════════════════════════════

_HAS_OPENCOMPASS: bool | None = None
_OC_PACKAGE_DIR: str | None = None


def _lazy_import_opencompass() -> bool:
    """懒加载 OpenCompass（只在需要运行评测时导入）"""
    global _HAS_OPENCOMPASS, _OC_PACKAGE_DIR
    if _HAS_OPENCOMPASS is not None:
        return _HAS_OPENCOMPASS
    try:
        import opencompass  # noqa: F401
        _OC_PACKAGE_DIR = os.path.dirname(opencompass.__file__)
        _HAS_OPENCOMPASS = True
    except ImportError:
        _HAS_OPENCOMPASS = False
    return _HAS_OPENCOMPASS


def _check_pip_installed() -> bool:
    """快速检测 opencompass 是否安装（不实际导入包）

    使用 pip show 或检查安装路径，几毫秒完成。
    """
    import importlib.metadata
    try:
        importlib.metadata.version("opencompass")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


OC_AVAILABLE = _check_pip_installed()


def check_opencompass() -> bool:
    """检测 OpenCompass 是否安装可用"""
    return _lazy_import_opencompass()


# ════════════════════════════════════════════════════════════════════════
# 2. OC-native benchmark 注册表
# ════════════════════════════════════════════════════════════════════════
# OpenCompass 有规范化评测协议的 benchmark（特定 few-shot、评分方式）
# 通过这些 benchmark 时，推荐走 OpenCompass

OC_DATASET_MAP: dict[str, dict[str, Any]] = {
    # --- MMLU (57学科, 5-shot) ---
    "oc_mmlu": {
        "name": "MMLU (OC)", "full_name": "MMLU via OpenCompass (5-shot)",
        "category": "知识理解", "icon": "book",
        "description": "OpenCompass 标准化 MMLU 评测（5-shot, gen mode），57学科。",
        "eval_type": "mmlu", "question_count": 14042,
        "oc_config_key": "mmmlu",
        "source": "opencompass",
    },
    "oc_gsm8k": {
        "name": "GSM8K (OC)", "full_name": "GSM8K via OpenCompass (8-shot CoT)",
        "category": "数学推理", "icon": "calculator",
        "description": "OpenCompass 标准化 GSM8K 评测（8-shot Chain-of-Thought）。",
        "eval_type": "gsm8k", "question_count": 1319,
        "oc_config_key": "gsm8k",
        "source": "opencompass",
    },
    "oc_humaneval": {
        "name": "HumanEval (OC)",
        "full_name": "HumanEval via OpenCompass",
        "category": "代码能力", "icon": "code",
        "description": "OpenCompass 标准化 HumanEval 评测。pass@1 指标。",
        "eval_type": "humaneval", "question_count": 164,
        "oc_config_key": "humaneval",
        "source": "opencompass",
    },
    "oc_bbh": {
        "name": "BBH (OC)",
        "full_name": "BIG-Bench Hard via OpenCompass (3-shot CoT)",
        "category": "推理能力", "icon": "trending-up",
        "description": "OpenCompass 标准化 BBH 评测（3-shot CoT），27子集。",
        "eval_type": "gsm8k", "question_count": 6511,
        "oc_config_key": "bbh",
        "source": "opencompass",
    },
    "oc_ceval": {
        "name": "C-Eval (OC)",
        "full_name": "C-Eval via OpenCompass (5-shot)",
        "category": "中文专项", "icon": "bookmark",
        "description": "OpenCompass 标准化 C-Eval 评测（5-shot），52学科。",
        "eval_type": "mmlu", "question_count": 13948,
        "oc_config_key": "ceval",
        "source": "opencompass",
    },
    "oc_arc_challenge": {
        "name": "ARC-C (OC)",
        "full_name": "ARC Challenge via OpenCompass (25-shot)",
        "category": "知识理解", "icon": "brain",
        "description": "OpenCompass 标准化 ARC Challenge 评测（25-shot）。",
        "eval_type": "mmlu", "question_count": 1172,
        "oc_config_key": "arc_c",
        "source": "opencompass",
    },
    "oc_hellaswag": {
        "name": "HellaSwag (OC)",
        "full_name": "HellaSwag via OpenCompass (10-shot)",
        "category": "知识理解", "icon": "zap",
        "description": "OpenCompass 标准化 HellaSwag 评测（10-shot）。",
        "eval_type": "mmlu", "question_count": 10042,
        "oc_config_key": "hellaswag",
        "source": "opencompass",
    },
    "oc_truthfulqa": {
        "name": "TruthfulQA (OC)",
        "full_name": "TruthfulQA via OpenCompass (0-shot)",
        "category": "知识理解", "icon": "check-circle",
        "description": "OpenCompass 标准化 TruthfulQA 评测（0-shot）。",
        "eval_type": "mmlu", "question_count": 817,
        "oc_config_key": "truthfulqa",
        "source": "opencompass",
    },
    "oc_math": {
        "name": "MATH (OC)",
        "full_name": "MATH via OpenCompass (4-shot CoT)",
        "category": "数学推理", "icon": "sigma",
        "description": "OpenCompass 标准化 MATH 评测（4-shot CoT）。",
        "eval_type": "gsm8k", "question_count": 5000,
        "oc_config_key": "math",
        "source": "opencompass",
    },
}


def load_oc_benchmarks() -> list[dict]:
    """返回 OpenCompass benchmark 列表（仅需检测 pip 包，不实际导入）"""
    if not OC_AVAILABLE:
        return []
    return [
        {
            "id": bid,
            "name": info["name"],
            "full_name": info["full_name"],
            "category": info["category"],
            "icon": info["icon"],
            "description": info["description"],
            "question_count": info["question_count"],
            "source": "opencompass",
        }
        for bid, info in OC_DATASET_MAP.items()
    ]


def is_oc_benchmark(benchmark_id: str) -> bool:
    return benchmark_id in OC_DATASET_MAP


# ════════════════════════════════════════════════════════════════════════
# 3. Config 生成
# ════════════════════════════════════════════════════════════════════════

# API meta template — OpenCompass 用这个定义 API 对话模板
API_META_TEMPLATE = {
    "round": [
        {"role": "HUMAN", "api_cell": [], "role_range": ("HUMAN", "BOT")},
        {"role": "BOT", "api_cell": [], "generate": True},
    ],
    "reserved_roles": [
        {"role": "SYSTEM", "api_cell": []},
    ],
}


def _generate_oc_config(model: dict, benchmark_ids: list[str], work_dir: str) -> str:
    """生成 OpenCompass config.py 文件内容

    Args:
        model: 平台模型配置 {api_base, api_key, model_name, name}
        benchmark_ids: OC benchmark ID 列表
        work_dir: 输出目录

    Returns:
        config.py 文件内容
    """
    # 1. 模型配置
    api_base = model.get("api_base", "").rstrip("/")
    api_key = model.get("api_key", "")
    model_name = model.get("model_name") or model.get("name", "default")
    model_abbr = model.get("name", model_name).replace(" ", "-")

    # 2. 用 import 方式引入数据集
    model_section = f"""
# ── 模型 ──
api_meta_template = {json.dumps(API_META_TEMPLATE, indent=2)}

from opencompass.models import OpenAISDK

models = [
    dict(
        type=OpenAISDK,
        abbr='{model_abbr}',
        path='{model_name}',
        key='{api_key}',
        openai_api_base='{api_base}/chat/completions' if '{api_base}' else None,
        meta_template=api_meta_template,
        max_out_len=4096,
        batch_size=1,
    )
]
"""

    # 3. 数据集配置 — 从 OC 内置 configs 中 import
    dataset_imports = []
    dataset_refs = []
    for bid in benchmark_ids:
        info = OC_DATASET_MAP.get(bid)
        if not info:
            continue
        config_key = info["oc_config_key"]
        # 根据 benchmark 类型查找对应的 OC config 模块
        oc_pkg = _OC_PACKAGE_DIR or "/dev/null"
        config_path = Path(oc_pkg) / "configs" / "datasets"
        # 找到匹配的 .py 文件
        matched = list(config_path.rglob(f"*{config_key}*.py"))
        if not matched:
            logger.warning(f"未找到 OC config: {config_key}")
            continue
        # 用相对路径作 import
        rel_path = matched[0].relative_to(Path(oc_pkg).parent)
        module_path = str(rel_path).replace("/", ".").replace(".py", "")
        dataset_imports.append(f"from {module_path} import {config_key}_datasets")
        dataset_refs.append(f"    *{config_key}_datasets,")

    datasets_section = f"""
# ── 数据集 ──
{'    '.join(dataset_imports)}

datasets = [
{'    '.join(dataset_refs)}
]
"""

    # 4. 执行配置
    exec_section = f"""
# ── 执行 ──
work_dir = r'{work_dir}'
"""

    return model_section + datasets_section + exec_section


# ════════════════════════════════════════════════════════════════════════
# 4. 结果解析
# ════════════════════════════════════════════════════════════════════════


def _parse_oc_results(work_dir: str) -> dict[str, dict]:
    """解析 OpenCompass 输出目录，提取各 benchmark 结果

    OpenCompass 输出结构：
      {work_dir}/
        {timestamp}/
          summaries/       <- summarizer JSON
          predictions/     <- 模型回答
          ...

    Returns:
        {benchmark_id: {"score": float, "details": [...]}}
    """
    work_path = Path(work_dir)
    if not work_path.exists():
        return {}

    # 找最新的时间戳目录
    timestamp_dirs = sorted(
        [d for d in work_path.iterdir() if d.is_dir() and d.name[:8].isdigit()],
        reverse=True,
    )
    if not timestamp_dirs:
        return {}

    results_dir = timestamp_dirs[0]

    # 读取 summarizer 结果
    summaries_dir = results_dir / "summaries"
    if not summaries_dir.exists():
        return {}

    oc_results = {}
    for f in sorted(summaries_dir.glob("*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "dataset" in item:
                        ds_name = item["dataset"]
                        score = item.get("score", item.get("accuracy", 0))
                        oc_results[ds_name] = {
                            "score": round(score * 100, 1) if score < 1 else score,
                            "oc_raw": item,
                        }
            elif isinstance(data, dict):
                for ds_name, metrics in data.items():
                    score = metrics.get("score", metrics.get("accuracy", 0))
                    oc_results[ds_name] = {
                        "score": round(score * 100, 1) if score < 1 else score,
                        "oc_raw": metrics,
                    }
        except (json.JSONDecodeError, IOError):
            continue

    return oc_results


# ════════════════════════════════════════════════════════════════════════
# 5. 执行入口
# ════════════════════════════════════════════════════════════════════════


def run_oc_eval(model: dict, benchmark_ids: list[str],
                progress_callback=None,
                timeout: int = 1800) -> dict:
    """通过 OpenCompass 执行评测

    Args:
        model: 平台模型配置
        benchmark_ids: OC benchmark ID 列表
        progress_callback: 进度回调 fn(done, total, msg)
        timeout: 超时秒数（默认 30 分钟）

    Returns:
        平台统一格式: {benchmark_id: {score, correct, total, details}}
    """
    if not _lazy_import_opencompass():
        return {
            bid: {"score": 0, "correct": 0, "total": 0, "error": "OpenCompass 未安装"}
            for bid in benchmark_ids
        }

    if progress_callback:
        progress_callback(0, len(benchmark_ids), "生成 OpenCompass 配置...")

    # 创建临时工作目录
    work_dir = tempfile.mkdtemp(prefix="oc_eval_")

    # 生成并写入 config
    config_content = _generate_oc_config(model, benchmark_ids, work_dir)
    config_path = os.path.join(work_dir, "config.py")
    with open(config_path, "w") as f:
        f.write(config_content)

    if progress_callback:
        progress_callback(1, len(benchmark_ids), "运行 OpenCompass 评测...")

    # 运行 OpenCompass
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "opencompass.cli.main", config_path],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        elapsed = round(time.time() - start)
        logger.info(f"OpenCompass completed in {elapsed}s (rc={result.returncode})")
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start)
        logger.error(f"OpenCompass 超时 ({elapsed}s)")
        return {
            bid: {"score": 0, "correct": 0, "total": 0,
                  "error": f"OpenCompass 执行超时 ({timeout}s)"}
            for bid in benchmark_ids
        }
    except FileNotFoundError:
        return {
            bid: {"score": 0, "correct": 0, "total": 0,
                  "error": "OpenCompass CLI 未找到"}
            for bid in benchmark_ids
        }

    # 解析结果
    oc_results = _parse_oc_results(work_dir)

    if progress_callback:
        progress_callback(len(benchmark_ids), len(benchmark_ids), "解析 OpenCompass 结果...")

    # 映射回平台格式
    platform_results = {}
    for bid in benchmark_ids:
        info = OC_DATASET_MAP.get(bid, {})
        config_key = info.get("oc_config_key", bid)
        oc_data = oc_results.get(config_key, {})

        score = oc_data.get("score", 0)
        platform_results[bid] = {
            "score": score,
            "correct": oc_data.get("oc_raw", {}).get("correct", 0),
            "total": oc_data.get("oc_raw", {}).get("total", oc_data.get("oc_raw", {}).get("num", 0)),
            "details": [],
            "avg_latency": 0,
            "total_tokens": 0,
            "error": result.stderr[:500] if result.returncode != 0 else "",
            "oc_output": result.stdout[-2000:] if result.returncode == 0 else "",
        }
        if not platform_results[bid].get("total"):
            platform_results[bid]["total"] = info.get("question_count", 0)

    # 清理临时文件
    import shutil
    try:
        shutil.rmtree(work_dir)
    except (OSError, PermissionError):
        pass

    return platform_results


# ════════════════════════════════════════════════════════════════════════
# 6. 扫描 OC 已支持的所有 benchmark 列表
# ════════════════════════════════════════════════════════════════════════


def scan_oc_datasets() -> list[str]:
    """扫描 OpenCompass 安装包中所有可用数据集配置"""
    if not _OC_PACKAGE_DIR:
        return []
    config_path = Path(_OC_PACKAGE_DIR) / "configs" / "datasets"
    if not config_path.exists():
        return []
    datasets = []
    for d in sorted(config_path.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            datasets.append(d.name)
    return datasets
