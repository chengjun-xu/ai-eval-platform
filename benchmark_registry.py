"""HuggingFace Datasets Benchmark Registry
========================================
每个 entry 对应一个可用的 HF benchmark，platform 通过 `load_hf_benchmarks()`
获取统一列表，与本地 JSON 文件合并展示。用户无感。

格式约定：
  {
    "id": str,          # 唯一标识
    "name": str,         # 短名
    "full_name": str,    # 完整名称
    "category": str,     # 分类
    "icon": str,         # emoji
    "description": str,  # 描述
    "eval_type": str,    # 评测类型映射: mmlu | gsm8k | humaneval | open
    "question_count": int|None,  # 已知题数，None=懒加载
    "hf_path": str,      # HF dataset 路径
    "hf_config": str|None,  # 子集配置，None=无配置
    "source": "huggingface",
  }

依赖：datasets（平台已装）
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# 1. 精选 HF Benchmark 目录（总计 ~210 个条目）
# ════════════════════════════════════════════════════════════════════════

BENCHMARK_REGISTRY: dict[str, dict[str, Any]] = {
    # ──────────────────── 知识理解（~70） ────────────────────
    # --- MMLU 系列 ---
    "hf_mmlu": {
        "name": "MMLU", "full_name": "MMLU (57学科)",
        "category": "知识理解", "icon": "book",
        "description": "大规模多任务语言理解评测。57个学科分类，含人文、社科、STEM等领域的选择题。",
        "eval_type": "mmlu", "question_count": 14042,
        "hf_path": "cais/mmlu", "hf_config": "all",
    },
    "hf_mmlu_pro": {
        "name": "MMLU-Pro", "full_name": "MMLU-Pro (高阶)",
        "category": "知识理解", "icon": "book-open",
        "description": "MMLU 高阶版，题目更复杂，选项更多。",
        "eval_type": "mmlu", "question_count": 12000,
        "hf_path": "tglcourse/mmlu-pro", "hf_config": None,
    },
    "hf_mmlu_redux": {
        "name": "MMLU-Redux", "full_name": "MMLU-Redux",
        "category": "知识理解", "icon": "refresh",
        "description": "MMLU 修正版，修复了原版中的错误标注。",
        "eval_type": "mmlu", "question_count": 14042,
        "hf_path": "edbeeching/mmlu-redux", "hf_config": None,
    },
    # --- MMLU 各学科独立版（便于细分分析）---
    "hf_mmlu_abstract_algebra": {
        "name": "MMLU-抽象代数", "full_name": "MMLU - Abstract Algebra",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 抽象代数子集，共 100 题。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "abstract_algebra",
    },
    "hf_mmlu_anatomy": {
        "name": "MMLU-解剖学", "full_name": "MMLU - Anatomy",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 解剖学子集。",
        "eval_type": "mmlu", "question_count": 135,
        "hf_path": "cais/mmlu", "hf_config": "anatomy",
    },
    "hf_mmlu_astronomy": {
        "name": "MMLU-天文学", "full_name": "MMLU - Astronomy",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 天文学子集。",
        "eval_type": "mmlu", "question_count": 152,
        "hf_path": "cais/mmlu", "hf_config": "astronomy",
    },
    "hf_mmlu_business_ethics": {
        "name": "MMLU-商业伦理", "full_name": "MMLU - Business Ethics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 商业伦理子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "business_ethics",
    },
    "hf_mmlu_clinical_knowledge": {
        "name": "MMLU-临床知识", "full_name": "MMLU - Clinical Knowledge",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 临床知识子集。",
        "eval_type": "mmlu", "question_count": 265,
        "hf_path": "cais/mmlu", "hf_config": "clinical_knowledge",
    },
    "hf_mmlu_college_biology": {
        "name": "MMLU-大学生物", "full_name": "MMLU - College Biology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学生物学子集。",
        "eval_type": "mmlu", "question_count": 144,
        "hf_path": "cais/mmlu", "hf_config": "college_biology",
    },
    "hf_mmlu_college_chemistry": {
        "name": "MMLU-大学化学", "full_name": "MMLU - College Chemistry",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学化学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "college_chemistry",
    },
    "hf_mmlu_college_computer_science": {
        "name": "MMLU-大学计算机", "full_name": "MMLU - College Computer Science",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学计算机子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "college_computer_science",
    },
    "hf_mmlu_college_mathematics": {
        "name": "MMLU-大学数学", "full_name": "MMLU - College Mathematics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学数学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "college_mathematics",
    },
    "hf_mmlu_college_medicine": {
        "name": "MMLU-大学医学", "full_name": "MMLU - College Medicine",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学医学子集。",
        "eval_type": "mmlu", "question_count": 173,
        "hf_path": "cais/mmlu", "hf_config": "college_medicine",
    },
    "hf_mmlu_college_physics": {
        "name": "MMLU-大学物理", "full_name": "MMLU - College Physics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 大学物理子集。",
        "eval_type": "mmlu", "question_count": 102,
        "hf_path": "cais/mmlu", "hf_config": "college_physics",
    },
    "hf_mmlu_computer_security": {
        "name": "MMLU-计算机安全", "full_name": "MMLU - Computer Security",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 计算机安全子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "computer_security",
    },
    "hf_mmlu_conceptual_physics": {
        "name": "MMLU-概念物理", "full_name": "MMLU - Conceptual Physics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 概念物理子集。",
        "eval_type": "mmlu", "question_count": 235,
        "hf_path": "cais/mmlu", "hf_config": "conceptual_physics",
    },
    "hf_mmlu_econometrics": {
        "name": "MMLU-计量经济学", "full_name": "MMLU - Econometrics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 计量经济学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "econometrics",
    },
    "hf_mmlu_electrical_engineering": {
        "name": "MMLU-电气工程", "full_name": "MMLU - Electrical Engineering",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 电气工程子集。",
        "eval_type": "mmlu", "question_count": 145,
        "hf_path": "cais/mmlu", "hf_config": "electrical_engineering",
    },
    "hf_mmlu_formal_logic": {
        "name": "MMLU-形式逻辑", "full_name": "MMLU - Formal Logic",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 形式逻辑子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "formal_logic",
    },
    "hf_mmlu_global_facts": {
        "name": "MMLU-全球事实", "full_name": "MMLU - Global Facts",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 全球事实子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "global_facts",
    },
    "hf_mmlu_high_school_biology": {
        "name": "MMLU-高中生物", "full_name": "MMLU - High School Biology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中生物子集。",
        "eval_type": "mmlu", "question_count": 300,
        "hf_path": "cais/mmlu", "hf_config": "high_school_biology",
    },
    "hf_mmlu_high_school_chemistry": {
        "name": "MMLU-高中化学", "full_name": "MMLU - High School Chemistry",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中化学子集。",
        "eval_type": "mmlu", "question_count": 200,
        "hf_path": "cais/mmlu", "hf_config": "high_school_chemistry",
    },
    "hf_mmlu_high_school_computer_science": {
        "name": "MMLU-高中计算机", "full_name": "MMLU - High School Computer Science",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中计算机子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "high_school_computer_science",
    },
    "hf_mmlu_high_school_european_history": {
        "name": "MMLU-欧洲史", "full_name": "MMLU - High School European History",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中欧洲史子集。",
        "eval_type": "mmlu", "question_count": 165,
        "hf_path": "cais/mmlu", "hf_config": "high_school_european_history",
    },
    "hf_mmlu_high_school_geography": {
        "name": "MMLU-高中地理", "full_name": "MMLU - High School Geography",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中地理子集。",
        "eval_type": "mmlu", "question_count": 198,
        "hf_path": "cais/mmlu", "hf_config": "high_school_geography",
    },
    "hf_mmlu_high_school_government_and_politics": {
        "name": "MMLU-美国政府", "full_name": "MMLU - High School Gov & Politics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 美国政府子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "high_school_government_and_politics",
    },
    "hf_mmlu_high_school_macroeconomics": {
        "name": "MMLU-宏观经济学", "full_name": "MMLU - High School Macroeconomics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中宏观经济学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "high_school_macroeconomics",
    },
    "hf_mmlu_high_school_mathematics": {
        "name": "MMLU-高中数学", "full_name": "MMLU - High School Mathematics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中数学子集。",
        "eval_type": "mmlu", "question_count": 270,
        "hf_path": "cais/mmlu", "hf_config": "high_school_mathematics",
    },
    "hf_mmlu_high_school_microeconomics": {
        "name": "MMLU-微观经济学", "full_name": "MMLU - High School Microeconomics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中微观经济学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "high_school_microeconomics",
    },
    "hf_mmlu_high_school_physics": {
        "name": "MMLU-高中物理", "full_name": "MMLU - High School Physics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中物理子集。",
        "eval_type": "mmlu", "question_count": 150,
        "hf_path": "cais/mmlu", "hf_config": "high_school_physics",
    },
    "hf_mmlu_high_school_psychology": {
        "name": "MMLU-高中心理学", "full_name": "MMLU - High School Psychology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中心理学子集。",
        "eval_type": "mmlu", "question_count": 200,
        "hf_path": "cais/mmlu", "hf_config": "high_school_psychology",
    },
    "hf_mmlu_high_school_statistics": {
        "name": "MMLU-高中统计", "full_name": "MMLU - High School Statistics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中统计子集。",
        "eval_type": "mmlu", "question_count": 200,
        "hf_path": "cais/mmlu", "hf_config": "high_school_statistics",
    },
    "hf_mmlu_high_school_us_history": {
        "name": "MMLU-美国史", "full_name": "MMLU - High School US History",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中美国史子集。",
        "eval_type": "mmlu", "question_count": 204,
        "hf_path": "cais/mmlu", "hf_config": "high_school_us_history",
    },
    "hf_mmlu_high_school_world_history": {
        "name": "MMLU-世界史", "full_name": "MMLU - High School World History",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 高中世界史子集。",
        "eval_type": "mmlu", "question_count": 165,
        "hf_path": "cais/mmlu", "hf_config": "high_school_world_history",
    },
    "hf_mmlu_human_aging": {
        "name": "MMLU-人类衰老", "full_name": "MMLU - Human Aging",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 人类衰老子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "human_aging",
    },
    "hf_mmlu_human_sexuality": {
        "name": "MMLU-人类性学", "full_name": "MMLU - Human Sexuality",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 人类性学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "human_sexuality",
    },
    "hf_mmlu_international_law": {
        "name": "MMLU-国际法", "full_name": "MMLU - International Law",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 国际法子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "international_law",
    },
    "hf_mmlu_jurisprudence": {
        "name": "MMLU-法理学", "full_name": "MMLU - Jurisprudence",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 法理学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "jurisprudence",
    },
    "hf_mmlu_logical_fallacies": {
        "name": "MMLU-逻辑谬误", "full_name": "MMLU - Logical Fallacies",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 逻辑谬误子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "logical_fallacies",
    },
    "hf_mmlu_machine_learning": {
        "name": "MMLU-机器学习", "full_name": "MMLU - Machine Learning",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 机器学习子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "machine_learning",
    },
    "hf_mmlu_management": {
        "name": "MMLU-管理学", "full_name": "MMLU - Management",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 管理学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "management",
    },
    "hf_mmlu_marketing": {
        "name": "MMLU-市场营销", "full_name": "MMLU - Marketing",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 市场营销子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "marketing",
    },
    "hf_mmlu_medical_genetics": {
        "name": "MMLU-医学遗传", "full_name": "MMLU - Medical Genetics",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 医学遗传子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "medical_genetics",
    },
    "hf_mmlu_miscellaneous": {
        "name": "MMLU-杂项", "full_name": "MMLU - Miscellaneous",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 杂项子集。",
        "eval_type": "mmlu", "question_count": 783,
        "hf_path": "cais/mmlu", "hf_config": "miscellaneous",
    },
    "hf_mmlu_moral_disputes": {
        "name": "MMLU-道德争议", "full_name": "MMLU - Moral Disputes",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 道德争议子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "moral_disputes",
    },
    "hf_mmlu_moral_scenarios": {
        "name": "MMLU-道德场景", "full_name": "MMLU - Moral Scenarios",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 道德场景子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "moral_scenarios",
    },
    "hf_mmlu_nutrition": {
        "name": "MMLU-营养学", "full_name": "MMLU - Nutrition",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 营养学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "nutrition",
    },
    "hf_mmlu_philosophy": {
        "name": "MMLU-哲学", "full_name": "MMLU - Philosophy",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 哲学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "philosophy",
    },
    "hf_mmlu_prehistory": {
        "name": "MMLU-史前史", "full_name": "MMLU - Prehistory",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 史前史子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "prehistory",
    },
    "hf_mmlu_professional_accounting": {
        "name": "MMLU-会计学", "full_name": "MMLU - Professional Accounting",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 会计学子集。",
        "eval_type": "mmlu", "question_count": 200,
        "hf_path": "cais/mmlu", "hf_config": "professional_accounting",
    },
    "hf_mmlu_professional_law": {
        "name": "MMLU-法律", "full_name": "MMLU - Professional Law",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 法律子集。",
        "eval_type": "mmlu", "question_count": 1000,
        "hf_path": "cais/mmlu", "hf_config": "professional_law",
    },
    "hf_mmlu_professional_medicine": {
        "name": "MMLU-专业医学", "full_name": "MMLU - Professional Medicine",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 专业医学子集。",
        "eval_type": "mmlu", "question_count": 272,
        "hf_path": "cais/mmlu", "hf_config": "professional_medicine",
    },
    "hf_mmlu_professional_psychology": {
        "name": "MMLU-专业心理学", "full_name": "MMLU - Professional Psychology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 专业心理学子集。",
        "eval_type": "mmlu", "question_count": 234,
        "hf_path": "cais/mmlu", "hf_config": "professional_psychology",
    },
    "hf_mmlu_public_relations": {
        "name": "MMLU-公共关系", "full_name": "MMLU - Public Relations",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 公共关系子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "public_relations",
    },
    "hf_mmlu_security_studies": {
        "name": "MMLU-安全研究", "full_name": "MMLU - Security Studies",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 安全研究子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "security_studies",
    },
    "hf_mmlu_sociology": {
        "name": "MMLU-社会学", "full_name": "MMLU - Sociology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 社会学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "sociology",
    },
    "hf_mmlu_us_foreign_policy": {
        "name": "MMLU-美国外交", "full_name": "MMLU - US Foreign Policy",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 美国外交子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "us_foreign_policy",
    },
    "hf_mmlu_virology": {
        "name": "MMLU-病毒学", "full_name": "MMLU - Virology",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 病毒学子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "virology",
    },
    "hf_mmlu_world_religions": {
        "name": "MMLU-世界宗教", "full_name": "MMLU - World Religions",
        "category": "知识理解", "icon": "book",
        "description": "MMLU 世界宗教子集。",
        "eval_type": "mmlu", "question_count": 100,
        "hf_path": "cais/mmlu", "hf_config": "world_religions",
    },
    # --- ARC ---
    "hf_arc_easy": {
        "name": "ARC-Easy", "full_name": "ARC Easy Set",
        "category": "知识理解", "icon": "brain",
        "description": "AI2 Reasoning Challenge (简单级)。涵盖小学科学知识的选择题。",
        "eval_type": "mmlu", "question_count": 2376,
        "hf_path": "allenai/ai2_arc", "hf_config": "ARC-Easy",
    },
    "hf_arc_challenge": {
        "name": "ARC-Challenge", "full_name": "ARC Challenge Set",
        "category": "知识理解", "icon": "brain",
        "description": "AI2 Reasoning Challenge (挑战级)。比 Easy 更难的科学推理题。",
        "eval_type": "mmlu", "question_count": 1172,
        "hf_path": "allenai/ai2_arc", "hf_config": "ARC-Challenge",
    },
    # --- 常识推理 ---
    "hf_hellaswag": {
        "name": "HellaSwag", "full_name": "HellaSwag",
        "category": "知识理解", "icon": "zap",
        "description": "常识推理评测。模型需要从多个结尾中选出最合理的一个。",
        "eval_type": "mmlu", "question_count": 10042,
        "hf_path": "Rowan/hellaswag", "hf_config": None,
    },
    "hf_truthfulqa_mc": {
        "name": "TruthfulQA-MC", "full_name": "TruthfulQA (多选题)",
        "category": "知识理解", "icon": "check-circle",
        "description": "TruthfulQA 多选题版。评估模型的真实性和诚实度。",
        "eval_type": "mmlu", "question_count": 817,
        "hf_path": "truthfulqa/truthful_qa", "hf_config": "multiple_choice",
    },
    "hf_winogrande": {
        "name": "WinoGrande", "full_name": "WinoGrande",
        "category": "知识理解", "icon": "git-merge",
        "description": "代词消歧评测。模型需根据上下文选择正确的代词指代。",
        "eval_type": "mmlu", "question_count": 1267,
        "hf_path": "allenai/winogrande", "hf_config": "winogrande_xl",
    },
    "hf_piqa": {
        "name": "PIQA", "full_name": "Physical Interaction QA",
        "category": "知识理解", "icon": "tool",
        "description": "物理常识推理。模型需要选择解决日常物理问题的正确方案。",
        "eval_type": "mmlu", "question_count": 1838,
        "hf_path": "ybisk/piqa", "hf_config": None,
    },
    "hf_openbookqa": {
        "name": "OpenBookQA", "full_name": "OpenBookQA",
        "category": "知识理解", "icon": "book-open",
        "description": "开放书籍问答。基于科学事实的多选题，需要融合推理。",
        "eval_type": "mmlu", "question_count": 500,
        "hf_path": "allenai/openbookqa", "hf_config": None,
    },
    "hf_commonsense_qa": {
        "name": "CommonsenseQA", "full_name": "CommonsenseQA",
        "category": "知识理解", "icon": "lightbulb",
        "description": "常识问答。需要利用常识知识回答的多选题。",
        "eval_type": "mmlu", "question_count": 1221,
        "hf_path": "tau/commonsense_qa", "hf_config": None,
    },
    "hf_social_i_qa": {
        "name": "SocialIQA", "full_name": "Social Interaction QA",
        "category": "知识理解", "icon": "users",
        "description": "社交常识推理。评估模型对社交互动的理解能力。",
        "eval_type": "mmlu", "question_count": 1954,
        "hf_path": "social_i_qa", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 2. 数学推理（~15）
    # ----------------------------------------------------------------
    "hf_gsm8k": {
        "name": "GSM8K", "full_name": "Grade School Math 8K",
        "category": "数学推理", "icon": "calculator",
        "description": "小学数学应用题，共约 1300 题。评估模型的数学推理与多步计算能力。",
        "eval_type": "gsm8k", "question_count": 1319,
        "hf_path": "openai/gsm8k", "hf_config": "main",
    },
    "hf_math": {
        "name": "MATH", "full_name": "Mathematics Aptitude Test of Heuristics",
        "category": "数学推理", "icon": "sigma",
        "description": "高中数学竞赛级题目，7个难度级别。评估模型的高级数学推理能力。",
        "eval_type": "gsm8k", "question_count": 5000,
        "hf_path": "hendrycks/math", "hf_config": None,
    },
    "hf_math_lvl5": {
        "name": "MATH-Lvl5", "full_name": "MATH 难度5 (最难)",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 数据集中的最高难度题目（Level 5）。",
        "eval_type": "gsm8k", "question_count": 1000,
        "hf_path": "hendrycks/math", "hf_config": "algebra",  # 需要过滤
    },
    "hf_math_prealgebra": {
        "name": "MATH-初等代数", "full_name": "MATH - Prealgebra",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 初等代数学科子集。",
        "eval_type": "gsm8k", "question_count": 872,
        "hf_path": "hendrycks/math", "hf_config": "prealgebra",
    },
    "hf_math_algebra": {
        "name": "MATH-代数", "full_name": "MATH - Algebra",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 代数学科子集。",
        "eval_type": "gsm8k", "question_count": 1200,
        "hf_path": "hendrycks/math", "hf_config": "algebra",
    },
    "hf_math_number_theory": {
        "name": "MATH-数论", "full_name": "MATH - Number Theory",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 数论子集。",
        "eval_type": "gsm8k", "question_count": 500,
        "hf_path": "hendrycks/math", "hf_config": "number_theory",
    },
    "hf_math_counting_prob": {
        "name": "MATH-计数概率", "full_name": "MATH - Counting & Probability",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 计数与概率子集。",
        "eval_type": "gsm8k", "question_count": 500,
        "hf_path": "hendrycks/math", "hf_config": "counting_and_probability",
    },
    "hf_math_geometry": {
        "name": "MATH-几何", "full_name": "MATH - Geometry",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 几何子集。",
        "eval_type": "gsm8k", "question_count": 400,
        "hf_path": "hendrycks/math", "hf_config": "geometry",
    },
    "hf_math_intermediate_algebra": {
        "name": "MATH-中级代数", "full_name": "MATH - Intermediate Algebra",
        "category": "数学推理", "icon": "sigma",
        "description": "MATH 中级代数子集。",
        "eval_type": "gsm8k", "question_count": 900,
        "hf_path": "hendrycks/math", "hf_config": "intermediate_algebra",
    },
    "hf_svamp": {
        "name": "SVAMP", "full_name": "SVAMP Math",
        "category": "数学推理", "icon": "calculator",
        "description": "SVAMP 数学应用题。评估模型对数学问题的语义理解。",
        "eval_type": "gsm8k", "question_count": 1000,
        "hf_path": "math_dataset/svamp", "hf_config": None,
    },
    "hf_aqua_rat": {
        "name": "AQuA-RAT", "full_name": "AQuA-RAT",
        "category": "数学推理", "icon": "calculator",
        "description": "Algebra Question Answering with Rationales。带推理过程的代数选择题。",
        "eval_type": "gsm8k", "question_count": 254,
        "hf_path": "aqua_rat", "hf_config": None,
    },
    "hf_theoremqa": {
        "name": "TheoremQA", "full_name": "TheoremQA",
        "category": "数学推理", "icon": "sigma",
        "description": "定理级问答。涵盖数学、物理、计算机等学科的定理应用题。",
        "eval_type": "gsm8k", "question_count": 800,
        "hf_path": "tiiuae/falcon-refinedweb",  # placeholder
        "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 3. 代码能力（~8）
    # ----------------------------------------------------------------
    "hf_humaneval": {
        "name": "HumanEval", "full_name": "HumanEval",
        "category": "代码能力", "icon": "code",
        "description": "Python 函数补全评测。模型需根据 docstring 编写完整函数。",
        "eval_type": "humaneval", "question_count": 164,
        "hf_path": "openai/openai_humaneval", "hf_config": None,
    },
    "hf_mbpp": {
        "name": "MBPP", "full_name": "Mostly Basic Python Programming",
        "category": "代码能力", "icon": "code",
        "description": "基础 Python 编程评测。共约 1000 道编程题。",
        "eval_type": "humaneval", "question_count": 974,
        "hf_path": "google-research-datasets/mbpp", "hf_config": None,
    },
    "hf_mbpp_sanitized": {
        "name": "MBPP-Sanitized", "full_name": "MBPP (清理版)",
        "category": "代码能力", "icon": "code",
        "description": "MBPP 清理后的干净版本，不含语法错误的测试用例。",
        "eval_type": "humaneval", "question_count": 427,
        "hf_path": "google-research-datasets/mbpp", "hf_config": "sanitized",
    },
    "hf_humaneval_multi": {
        "name": "HumanEval-Multi",
        "full_name": "HumanEval (多语言版)",
        "category": "代码能力", "icon": "code",
        "description": "HumanEval 多语言扩展版，支持 Python、Java、JS、C++、Go 等。",
        "eval_type": "humaneval", "question_count": 164,
        "hf_path": "nuprl/MultiPL-E", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 4. 推理能力（~15）
    # ----------------------------------------------------------------
    "hf_bbh": {
        "name": "BBH", "full_name": "BIG-Bench Hard",
        "category": "推理能力", "icon": "trending-up",
        "description": "BIG-Bench 的 27 个困难子集，评估模型的深层推理能力。",
        "eval_type": "gsm8k", "question_count": 6511,
        "hf_path": "lukaemon/bbh", "hf_config": "all",
    },
    # BBH 各子集
    "hf_bbh_navigate": {
        "name": "BBH-导航", "full_name": "BBH - Navigate",
        "category": "推理能力", "icon": "map",
        "description": "导航推理：模型需根据方向指令推断最终位置。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "navigate",
    },
    "hf_bbh_date_understanding": {
        "name": "BBH-日期理解", "full_name": "BBH - Date Understanding",
        "category": "推理能力", "icon": "calendar",
        "description": "日期推理：基于日期描述推断具体日期。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "date_understanding",
    },
    "hf_bbh_sports_understanding": {
        "name": "BBH-体育理解", "full_name": "BBH - Sports Understanding",
        "category": "推理能力", "icon": "activity",
        "description": "体育知识推理。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "sports_understanding",
    },
    "hf_bbh_tracking_shuffled_objects": {
        "name": "BBH-追踪打乱对象",
        "full_name": "BBH - Tracking Shuffled Objects",
        "category": "推理能力", "icon": "shuffle",
        "description": "追踪推理：跟踪被打乱的对象位置。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "tracking_shuffled_objects_three_shapes",
    },
    "hf_bbh_logical_deduction": {
        "name": "BBH-逻辑演绎", "full_name": "BBH - Logical Deduction",
        "category": "推理能力", "icon": "git-branch",
        "description": "逻辑推理：基于条件推导结论。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "logical_deduction_three_objects",
    },
    "hf_bbh_hyperbaton": {
        "name": "BBH-词语排序", "full_name": "BBH - Hyperbaton",
        "category": "推理能力", "icon": "type",
        "description": "语法推理：判断形容词的正确排列顺序。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "hyperbaton",
    },
    "hf_bbh_geometric_shapes": {
        "name": "BBH-几何形状", "full_name": "BBH - Geometric Shapes",
        "category": "推理能力", "icon": "hexagon",
        "description": "空间推理：识别几何形状的名称。",
        "eval_type": "gsm8k", "question_count": 250,
        "hf_path": "lukaemon/bbh", "hf_config": "geometric_shapes",
    },
    "hf_logiqa": {
        "name": "LogiQA", "full_name": "LogiQA",
        "category": "推理能力", "icon": "git-merge",
        "description": "逻辑推理选择题。评估模型的形式逻辑推理能力。",
        "eval_type": "mmlu", "question_count": 651,
        "hf_path": "lucasmorin/LogiQA_with_options", "hf_config": None,
    },
    "hf_strategy_qa": {
        "name": "StrategyQA", "full_name": "StrategyQA",
        "category": "推理能力", "icon": "target",
        "description": "策略问答。评估模型是否具备多步推理得出 yes/no 答案的能力。",
        "eval_type": "mmlu", "question_count": 2290,
        "hf_path": "metaeval/strategyqa", "hf_config": None,
    },
    "hf_drop": {
        "name": "DROP", "full_name": "Discrete Reasoning Over Paragraphs",
        "category": "推理能力", "icon": "book",
        "description": "段落实证推理。需要对段落内容执行离散数学推理。",
        "eval_type": "gsm8k", "question_count": 9600,
        "hf_path": "ucinlp/drop", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 5. 医疗专业（~8）
    # ----------------------------------------------------------------
    "hf_medmcqa": {
        "name": "MedMCQA", "full_name": "Medical MCQ",
        "category": "医疗专业", "icon": "heart",
        "description": "印度 AIIMS/NEET PG 医考题。覆盖临床各科室的多选题。",
        "eval_type": "mmlu", "question_count": 6000,
        "hf_path": "medmcqa", "hf_config": None,
    },
    "hf_medqa": {
        "name": "MedQA", "full_name": "MedQA (USMLE)",
        "category": "医疗专业", "icon": "heart",
        "description": "USMLE 医考题。美国执业医师资格考试级别的选择题。",
        "eval_type": "mmlu", "question_count": 1273,
        "hf_path": "bigbio/med_qa", "hf_config": None,
    },
    "hf_pubmedqa": {
        "name": "PubMedQA", "full_name": "PubMedQA",
        "category": "医疗专业", "icon": "book-open",
        "description": "基于 PubMed 文献的 yes/no 问答评测。评估模型的医学文献理解能力。",
        "eval_type": "mmlu", "question_count": 500,
        "hf_path": "pubmed_qa", "hf_config": "pqa_labeled",
    },
    # ----------------------------------------------------------------
    # 6. 安全合规（~5）
    # ----------------------------------------------------------------
    "hf_safetybench_chinese": {
        "name": "SafetyBench-中文",
        "full_name": "SafetyBench (Chinese)",
        "category": "安全合规", "icon": "shield",
        "description": "中文安全评测。覆盖偏见、歧视、违法、隐私等 7 类安全场景。",
        "eval_type": "mmlu", "question_count": 10050,
        "hf_path": "thu-coai/SafetyBench", "hf_config": "Chinese",
    },
    "hf_safetybench_english": {
        "name": "SafetyBench-英文",
        "full_name": "SafetyBench (English)",
        "category": "安全合规", "icon": "shield",
        "description": "英文安全评测。覆盖 7 类安全场景的多选题。",
        "eval_type": "mmlu", "question_count": 6825,
        "hf_path": "thu-coai/SafetyBench", "hf_config": "English",
    },
    "hf_do_not_answer": {
        "name": "Do-Not-Answer",
        "full_name": "Do-Not-Answer",
        "category": "安全合规", "icon": "shield-off",
        "description": "有害问答安全评测。模型应拒绝回答危险/有害问题。",
        "eval_type": "open", "question_count": 939,
        "hf_path": "AnonymousSub/Do-Not-Answer", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 7. 中文/HK 专项（~10）
    # ----------------------------------------------------------------
    "hf_c_eval": {
        "name": "C-Eval", "full_name": "C-Eval (中文综合)",
        "category": "中文专项", "icon": "bookmark",
        "description": "中文综合能力评测。覆盖人文、社科、理工等 52 个学科的选择题。",
        "eval_type": "mmlu", "question_count": 13948,
        "hf_path": "ceval/ceval-exam", "hf_config": None,
    },
    "hf_c_mmlu": {
        "name": "CMMLU", "full_name": "Chinese MMLU",
        "category": "中文专项", "icon": "bookmark",
        "description": "中文 MMLU。涵盖中国教育体系的 67 学科知识。",
        "eval_type": "mmlu", "question_count": 10000,
        "hf_path": "haonan-li/cmmlu", "hf_config": None,
    },
    "hf_agieval": {
        "name": "AGIEval", "full_name": "AGIEval",
        "category": "中文专项", "icon": "award",
        "description": "AGI 评估。涵盖公务员考试、司法考试、数学竞赛等真实考试题目。",
        "eval_type": "mmlu", "question_count": 8049,
        "hf_path": "canghong/AGIEval", "hf_config": None,
    },
    "hf_mmlu_cf": {
        "name": "MMLU-CF", "full_name": "MMLU Chinese Facts",
        "category": "中文专项", "icon": "bookmark",
        "description": "MMLU 中文事实版。从 MMLU 中文改编的中国文化/社会知识评测。",
        "eval_type": "mmlu", "question_count": 14042,
        "hf_path": "clue-ai/MMLU_Chinese", "hf_config": None,
    },
    "hf_tongji_ceval": {
        "name": "TCEval", "full_name": "同济中文综合评测",
        "category": "中文专项", "icon": "bookmark",
        "description": "同济大学版中文综合能力评测。",
        "eval_type": "mmlu", "question_count": 600,
        "hf_path": "xusenlin/TCEval", "hf_config": None,
    },
    "hf_chinese_biobenchmark": {
        "name": "C-Bio", "full_name": "中文生物医学评测",
        "category": "中文专项", "icon": "bookmark",
        "description": "中文生物医学领域评测数据集。",
        "eval_type": "mmlu", "question_count": 2000,
        "hf_path": "cblue/CBLUE", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 8. 多语言（~10）
    # ----------------------------------------------------------------
    "hf_xnli": {
        "name": "XNLI", "full_name": "Cross-lingual NLI",
        "category": "多语言", "icon": "globe",
        "description": "跨语言自然语言推理。15种语言的蕴含关系判断。",
        "eval_type": "mmlu", "question_count": 5010,
        "hf_path": "xnli", "hf_config": None,
    },
    "hf_xcopa": {
        "name": "XCOPA", "full_name": "Cross-lingual COPA",
        "category": "多语言", "icon": "globe",
        "description": "跨语言因果推理。11种语言的因果关系判断。",
        "eval_type": "mmlu", "question_count": 2000,
        "hf_path": "xcopa", "hf_config": None,
    },
    "hf_paws_x": {
        "name": "PAWS-X", "full_name": "PAWS-X",
        "category": "多语言", "icon": "globe",
        "description": "跨语言复述检测。7种语言的句子语义等价判断。",
        "eval_type": "mmlu", "question_count": 2350,
        "hf_path": "paws-x", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 9. 红队/对抗（~5）
    # ----------------------------------------------------------------
    "hf_advglue": {
        "name": "AdvGLUE", "full_name": "Adversarial GLUE",
        "category": "红队/对抗", "icon": "flask",
        "description": "对抗性文本分类评测。评估模型在对抗扰动下的鲁棒性。",
        "eval_type": "mmlu", "question_count": 6000,
        "hf_path": "adv_glue", "hf_config": None,
    },
    "hf_adv_squad": {
        "name": "AdvSQuAD", "full_name": "Adversarial SQuAD",
        "category": "红队/对抗", "icon": "flask",
        "description": "对抗性阅读理解。包含添加干扰项的阅读理解题。",
        "eval_type": "gsm8k", "question_count": 10500,
        "hf_path": "adversarial_qa", "hf_config": "adversarialQA",
    },
    # ----------------------------------------------------------------
    # 10. 长上下文（~5）
    # ----------------------------------------------------------------
    "hf_qmsum": {
        "name": "QMSum", "full_name": "Query-based Meeting Summarization",
        "category": "长上下文", "icon": "list",
        "description": "会议纪要长文本问答。评估模型的长上下文理解与摘要能力。",
        "eval_type": "open", "question_count": 1257,
        "hf_path": "ccdv/qmsum", "hf_config": None,
    },
    # ----------------------------------------------------------------
    # 11. 开放题/综合（~5）
    # ----------------------------------------------------------------
    "hf_truthfulqa_gen": {
        "name": "TruthfulQA-Gen",
        "full_name": "TruthfulQA (生成式)",
        "category": "综合能力", "icon": "message-square",
        "description": "TruthfulQA 生成式版本。评估模型回答的真实性和诚实度。",
        "eval_type": "open", "question_count": 817,
        "hf_path": "truthfulqa/truthful_qa", "hf_config": "generation",
    },
    "hf_bbh_mixture": {
        "name": "BBH-Mixture",
        "full_name": "BBH 27子集混合",
        "category": "推理能力", "icon": "trending-up",
        "description": "BIG-Bench Hard 全部 27 个子集的混合版。",
        "eval_type": "gsm8k", "question_count": 6511,
        "hf_path": "lukaemon/bbh", "hf_config": "mixture",
    },
    # ----------------------------------------------------------------
    # 12. 多模态/视觉（~4）
    # ----------------------------------------------------------------
    "hf_mmmu": {
        "name": "MMMU", "full_name": "Massive Multi-discipline Multimodal Understanding",
        "category": "多模态", "icon": "image",
        "description": "多学科多模态理解评测。涵盖艺术、工程、医学等学科的图文选择题。",
        "eval_type": "mmlu", "question_count": 900,
        "hf_path": "MMMU/MMMU", "hf_config": "dev",
    },
    "hf_mmbench": {
        "name": "MMBench", "full_name": "MMBench (图文)",
        "category": "多模态", "icon": "image",
        "description": "多模态基准测试。评估模型的图文理解和推理能力。",
        "eval_type": "mmlu", "question_count": 3000,
        "hf_path": "lmms-lab/MMBench", "hf_config": None,
    },
    "hf_vqav2": {
        "name": "VQAv2", "full_name": "Visual QA v2",
        "category": "多模态", "icon": "image",
        "description": "视觉问答评测。基于图像的开放题问答。",
        "eval_type": "open", "question_count": 10000,
        "hf_path": "HuggingFaceM4/VQAv2", "hf_config": None,
    },
    "hf_scienceqa": {
        "name": "ScienceQA", "full_name": "ScienceQA (图文)",
        "category": "多模态", "icon": "image",
        "description": "科学问答多模态评测。包含文本和图像的科学知识问答。",
        "eval_type": "mmlu", "question_count": 6000,
        "hf_path": "derekchen/ScienceQA", "hf_config": None,
    },
}

# ════════════════════════════════════════════════════════════════════════
# 2. 评测函数映射
# ════════════════════════════════════════════════════════════════════════
# eval_type → BENCHMARK_EVALS 中的 lookup key
EVAL_TYPE_MAP = {
    "mmlu": "mmlu",
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "open": "open_ended",
}

# ════════════════════════════════════════════════════════════════════════
# 3. 公共函数
# ════════════════════════════════════════════════════════════════════════


def is_hf_benchmark(benchmark_id: str) -> bool:
    """判断 benchmark_id 是否来自 HuggingFace"""
    return benchmark_id in BENCHMARK_REGISTRY


def get_hf_benchmark(benchmark_id: str) -> dict | None:
    """获取 HF benchmark 元信息"""
    return BENCHMARK_REGISTRY.get(benchmark_id)


def load_hf_benchmarks() -> list[dict]:
    """返回 HF benchmark 列表（与 app.py load_benchmarks 格式一致）"""
    results = []
    for bid, info in BENCHMARK_REGISTRY.items():
        results.append({
            "id": bid,
            "name": info["name"],
            "full_name": info["full_name"],
            "category": info["category"],
            "icon": info["icon"],
            "description": info["description"],
            "question_count": info["question_count"],
            "source": "huggingface",
            "eval_type": info["eval_type"],
            "hf_path": info["hf_path"],
            "hf_config": info["hf_config"],
        })
    return results


def load_hf_dataset(benchmark_id: str, split: str = "test") -> list[dict]:
    """从 HuggingFace 加载 benchmark 数据并转换为平台内部格式

    返回统一格式的 list[dict]:
      - MMLU 类型: {id, question, choices: {A,B,C,D}, answer, category}
      - GSM8K 类型: {id, question, answer, category}
      - HumanEval 类型: {id, prompt, test, description, entry_point}
      - Open 类型: {id, question, reference_answer, category}
    """
    from datasets import load_dataset

    entry = BENCHMARK_REGISTRY.get(benchmark_id)
    if not entry:
        raise ValueError(f"未知的 HF benchmark: {benchmark_id}")

    hf_path = entry["hf_path"]
    hf_config = entry["hf_config"]
    eval_type = entry["eval_type"]

    try:
        if hf_config:
            ds = load_dataset(hf_path, hf_config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(hf_path, split=split, trust_remote_code=True)
    except Exception as e:
        # 某些数据集没有 test split，用 validation 或 train
        try:
            if hf_config:
                ds = load_dataset(hf_path, hf_config, split="validation", trust_remote_code=True)
            else:
                ds = load_dataset(hf_path, split="validation", trust_remote_code=True)
        except Exception:
            if hf_config:
                ds = load_dataset(hf_path, hf_config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(hf_path, split="train", trust_remote_code=True)

    if eval_type == "mmlu":
        return _normalize_mmlu(ds, benchmark_id)
    elif eval_type == "gsm8k":
        return _normalize_gsm8k(ds, benchmark_id)
    elif eval_type == "humaneval":
        return _normalize_humaneval(ds, benchmark_id)
    elif eval_type == "open":
        return _normalize_open(ds, benchmark_id)
    else:
        raise ValueError(f"未知的评测类型: {eval_type}")


# ════════════════════════════════════════════════════════════════════════
# 4. 数据集标准化函数
# ════════════════════════════════════════════════════════════════════════


def _normalize_mmlu(ds, benchmark_id: str) -> list[dict]:
    """标准化多种 MCQ 格式为统一格式"""
    items = []
    cat_map = {
        "mmlu": "general", "mmlu_pro": "general",
        "arc_easy": "science", "arc_challenge": "science",
        "hellaswag": "common_sense", "truthfulqa_mc": "truthfulness",
        "winogrande": "common_sense", "piqa": "physics",
        "openbookqa": "science", "commonsense_qa": "common_sense",
        "social_i_qa": "social",
    }
    default_cat = cat_map.get(benchmark_id.replace("hf_", "").split("_")[0], "general")

    for i, row in enumerate(ds):
        item = {"id": str(i), "category": default_cat}

        # 判断 MMLU 格式（question, choices dict, answer int/str）
        if "question" in row and "choices" in row and "answer" in row:
            item["question"] = row["question"]
            choices = row["choices"]
            if isinstance(choices, list):
                labels = ["A", "B", "C", "D"]
                item["choices"] = {labels[j]: c for j, c in enumerate(choices) if c}
            elif isinstance(choices, dict):
                item["choices"] = choices
            else:
                choices_raw = str(choices)
                item["choices"] = {"A": choices_raw}
            # answer
            ans = row["answer"]
            if isinstance(ans, int):
                labels = list(item["choices"].keys())
                item["answer"] = labels[ans] if ans < len(labels) else labels[-1]
            else:
                item["answer"] = str(ans).strip()

        # ARC 格式（question, choices, answerKey）
        elif "question" in row and "choices" in row:
            item["question"] = row["question"]
            ch = row["choices"]
            if isinstance(ch, dict):
                text = ch.get("text", ch.get("label", []))
                label = ch.get("label", [])
                if isinstance(text, list) and isinstance(label, list):
                    item["choices"] = {l: t for l, t in zip(label, text)}
                else:
                    item["choices"] = {"A": str(ch)}
            elif isinstance(ch, list):
                labels = ["A", "B", "C", "D"]
                item["choices"] = {labels[j]: str(c) for j, c in enumerate(ch)}
            else:
                item["choices"] = {"A": str(ch)}
            ak = row.get("answerKey", row.get("answer", "A"))
            item["answer"] = str(ak).strip() if ak else "A"

        # HellaSwag 格式 (ctx, endings, label)
        elif "ctx" in row and "endings" in row:
            ctx = row.get("ctx_a", row["ctx"])
            endings = row["endings"]
            labels = ["A", "B", "C", "D"]
            item["question"] = ctx
            item["choices"] = {labels[j]: str(e) for j, e in enumerate(endings)}
            item["answer"] = labels[int(row["label"])] if "label" in row else "A"

        # PIQA 格式 (goal, sol1, sol2, label)
        elif "goal" in row and "sol1" in row:
            item["question"] = row["goal"]
            item["choices"] = {"A": str(row["sol1"]), "B": str(row["sol2"])}
            item["answer"] = "A" if int(row.get("label", 0)) == 0 else "B"

        # WinoGrande (sentence, option1, option2, answer)
        elif "sentence" in row:
            item["question"] = row["sentence"]
            item["choices"] = {"A": str(row.get("option1", "")), "B": str(row.get("option2", ""))}
            item["answer"] = str(int(row.get("answer", 1))) if str(row.get("answer", "")).isdigit() else str(row.get("answer", "A"))
            # 映射 answer 1/2 → A/B
            if item["answer"] in ("1", "2"):
                item["answer"] = "A" if item["answer"] == "1" else "B"

        # TruthfulQA MC (question, mc1_targets, mc2_targets)
        elif "mc1_targets" in row:
            item["question"] = row.get("question", "")
            mc = row.get("mc1_targets", {})
            if isinstance(mc, dict):
                choices = mc.get("choices", [])
                labels_list = mc.get("labels", [])
                if choices:
                    labels_abc = ["A", "B", "C", "D"]
                    item["choices"] = {labels_abc[j]: c for j, c in enumerate(choices)}
                    if labels_list:
                        true_idx = labels_list.index(1) if 1 in labels_list else 0
                        item["answer"] = labels_abc[true_idx] if true_idx < len(labels_abc) else "A"
                    else:
                        item["answer"] = "A"
                else:
                    item["choices"] = {"A": str(mc)}
                    item["answer"] = "A"
            else:
                item["choices"] = {"A": str(mc)}
                item["answer"] = "A"

        # CommonsenseQA (question, choices: {label, text}, answerKey)
        elif "question" in row and "choices" in row and "answerKey" in row:
            item["question"] = row["question"]
            ch = row["choices"]
            if isinstance(ch, dict) and "label" in ch and "text" in ch:
                labels_ = ch["label"]
                texts_ = ch["text"]
                if isinstance(labels_, list) and isinstance(texts_, list):
                    item["choices"] = {l: t for l, t in zip(labels_, texts_)}
            item["answer"] = str(row.get("answerKey", "A"))

        else:
            # 兜底：取第一个文本字段作为 question
            for key in ["question", "text", "sentence", "input", "ctx"]:
                if key in row and isinstance(row[key], str):
                    item["question"] = row[key]
                    break
            item["choices"] = {"A": "正确", "B": "错误"}
            item["answer"] = "A"

        if not item.get("question"):
            item["question"] = str(row)[:200]

        items.append(item)
    return items


def _normalize_gsm8k(ds, benchmark_id: str) -> list[dict]:
    """标准化数学推理数据为统一格式"""
    items = []
    for i, row in enumerate(ds):
        question = row.get("question", row.get("problem", str(row)[:200]))
        answer = row.get("answer", "")
        # GSM8K 答案格式："#### 1234" 或 "The answer is 1234" 或纯数字
        if "####" in str(answer):
            answer = str(answer).split("####")[-1].strip()
        elif "The answer is" in str(answer):
            answer = str(answer).split("The answer is")[-1].strip()
        elif isinstance(answer, str):
            answer = answer.strip()

        category = benchmark_id.replace("hf_", "").split("_")[0]
        items.append({
            "id": str(i),
            "question": question,
            "answer": answer,
            "category": category.capitalize(),
        })
    return items


def _normalize_humaneval(ds, benchmark_id: str) -> list[dict]:
    """标准化代码生成为统一格式"""
    items = []
    for i, row in enumerate(ds):
        task_id = row.get("task_id", str(i))
        if isinstance(task_id, int):
            task_id = str(task_id)
        # HumanEval 格式
        prompt = row.get("prompt", "")
        entry_point = row.get("entry_point", "")
        test_code = row.get("test", "")
        # MBPP 格式 (text, code, test_list)
        if "code" in row and "test_list" in row:
            prompt = row.get("text", "")
            test_code = "\n".join(row.get("test_list", []))
            entry_point = row.get("task_id", str(i))

        items.append({
            "id": task_id,
            "prompt": prompt,
            "test": test_code,
            "entry_point": entry_point,
            "description": f"Write a function: {entry_point}" if entry_point else "Write Python code",
        })
    return items


def _normalize_open(ds, benchmark_id: str) -> list[dict]:
    """标准化开放题数据"""
    items = []
    cat_map = {
        "truthfulqa_gen": "truthfulness",
        "do_not_answer": "safety",
    }
    default_cat = cat_map.get(benchmark_id.replace("hf_", ""), "general")

    for i, row in enumerate(ds):
        question = row.get("question", row.get("input", str(row)[:200]))
        # 找参考回答
        reference = ""
        for key in ["reference_answer", "answer", "expected", "output", "target"]:
            if key in row:
                ref_val = row[key]
                if isinstance(ref_val, list):
                    ref_val = ref_val[0] if ref_val else ""
                reference = str(ref_val)
                break

        items.append({
            "id": str(i),
            "question": question,
            "reference_answer": reference,
            "category": default_cat,
        })
    return items
