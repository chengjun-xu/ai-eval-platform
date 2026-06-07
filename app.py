"""
AI 大模型评测平台 - Flask Application
=====================================
支持模型注册、评测执行、结果展示的一体化 LLM 评测平台。
"""
import json
import os
import uuid
import time as _time
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import eval_agent

from eval_runner import start_eval, get_run_status, list_completed_runs, list_user_runs, list_running_jobs

# ── HuggingFace Benchmark 注册表 ──────────────────────────────────────
try:
    from benchmark_registry import load_hf_benchmarks
    HAS_HF_BENCHMARKS = True
except ImportError:
    HAS_HF_BENCHMARKS = False
    def load_hf_benchmarks(): return []

# ── OpenCompass 适配器 ────────────────────────────────────────────────
try:
    from opencompass_adapter import load_oc_benchmarks, OC_AVAILABLE, scan_oc_datasets
    HAS_OC = OC_AVAILABLE
except ImportError:
    HAS_OC = False
    def load_oc_benchmarks(): return []
    def scan_oc_datasets(): return []

# ── 数据挖掘 Pipeline ─────────────────────────────────────────────────
try:
    from data_mining_pipeline import mine_from_text, mine_from_file, mine_from_hf, register_as_benchmark
    HAS_DATA_MINING = True
except ImportError:
    HAS_DATA_MINING = False
    def mine_from_text(*a, **kw): return []
    def mine_from_file(*a, **kw): return []
    def mine_from_hf(*a, **kw): return []
    def register_as_benchmark(*a, **kw): return ""

# ── 自动化数据生产引擎 ─────────────────────────────────────────────────
try:
    from data_production import (
        generate_self_instruct,
        batch_evolve,
        generate_rl_preference_pairs,
        register_as_benchmark as dp_register_as_benchmark,
    )
    HAS_DATA_PRODUCTION = True
except ImportError:
    HAS_DATA_PRODUCTION = False
    def generate_self_instruct(*a, **kw): raise NotImplementedError("data_production 模块未安装")
    def batch_evolve(*a, **kw): raise NotImplementedError("data_production 模块未安装")
    def generate_rl_preference_pairs(*a, **kw): raise NotImplementedError("data_production 模块未安装")
    def dp_register_as_benchmark(*a, **kw): return ""

# ── 长尾场景生成器 ───────────────────────────────────────────────────
try:
    from longtail_generator import LongtailGenerator, register_longtail_as_benchmark
    HAS_LONGTAIL = True
except ImportError:
    HAS_LONGTAIL = False
    class LongtailGenerator:
        def generate(self, *a, **kw): return []
    def register_longtail_as_benchmark(*a, **kw): return ""

# ── 评测分析（回归检测 + 评分一致性）────────────────────────────────────
try:
    from eval_analysis import detect_regression, compare_judges, analyze_score_distribution
    HAS_ANALYSIS = True
except ImportError:
    HAS_ANALYSIS = False
    def detect_regression(*a, **kw): return {"regressions": [], "summary": {"total_regressed": 0}}
    def compare_judges(*a, **kw): return {"pairs": {}}
    def analyze_score_distribution(*a, **kw): return {}

# ── 弱项挖掘 ─────────────────────────────────────────────────────────
try:
    from weakness_miner import analyze_model, compare_weaknesses, get_all_models_with_data
    HAS_WEAKNESS_MINER = True
except ImportError:
    HAS_WEAKNESS_MINER = False
    def analyze_model(*a, **kw): return {"error": "弱项挖掘模块未加载"}
    def compare_weaknesses(*a, **kw): return {}
    def get_all_models_with_data(*a): return []

# ── 大数据集流式计数（避免加载 12MB JSON 到内存） ──────────────────────────

_BENCHMARK_CACHE: dict[str, tuple[float, list]] = {}  # path -> (mtime, items_count_or_list)

def _count_json_items(path: Path) -> int | list:
    """流式统计 JSON 数组元素个数，大数据集只计数、小数据集返回列表"""
    stat = path.stat()
    cache_key = str(path.absolute())
    cached = _BENCHMARK_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime:
        return cached[1]

    fsize = stat.st_size
    if fsize > 500_000:  # >500KB 快速统计行首 `{` 个数
        count = 0
        with open(path, "rb") as f:
            for line in f:
                # 统计以 { 开头的非空行（每个 JSON 对象一行）
                stripped = line.lstrip()
                if stripped and stripped[0:1] == b'{':
                    count += 1
        result = count
    else:
        with open(path, encoding="utf-8") as f:
            result = json.load(f)
            result = len(result) if isinstance(result, list) else 0

    _BENCHMARK_CACHE[cache_key] = (stat.st_mtime, result)
    return result


# ---- 污染检测 + 错误归因 -----------------------------------------------------
try:
    from contamination_detector import analyze_dataset, describe_contamination_in_report
    HAS_CONTAMINATION = True
except ImportError:
    HAS_CONTAMINATION = False
    def analyze_dataset(*a, **kw): return {}
    def describe_contamination_in_report(*a, **kw): return ""

try:
    from error_classifier import analyze_bad_cases
    HAS_ERROR_CLASSIFIER = True
except ImportError:
    HAS_ERROR_CLASSIFIER = False
    def analyze_bad_cases(*a): return {}

try:
    from rubric_manager import list_rubric_templates, RUBRIC_TEMPLATES
    HAS_RUBRIC = True
except ImportError:
    HAS_RUBRIC = False
    RUBRIC_TEMPLATES = {}
    def list_rubric_templates(): return {}

# ---------------------------------------------------------------------------
# 应用初始化
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("EVAL_PLATFORM_SECRET")
if not app.secret_key:
    import warnings
    warnings.warn(
        "⚠ EVAL_PLATFORM_SECRET 环境变量未设置，使用生成的随机密钥（重启后会话会失效）"
    )
    app.secret_key = os.urandom(32).hex()
app.config['SESSION_COOKIE_NAME'] = 'ai_eval_session'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 小时
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 上传限制

# ── API Key 加密 ─────────────────────────────────────────────────────────
# 使用 EVAL_PLATFORM_SECRET 作为加密密钥，对存储的 API Key 进行简单加密
_ENCRYPTION_KEY = (app.secret_key or os.urandom(16).hex())[:32].encode()

def _encrypt_api_key(plaintext: str) -> str:
    """简单加密 API Key（非生产级安全，防明文存储）"""
    if not plaintext:
        return ""
    try:
        from cryptography.fernet import Fernet
        import base64
        key = base64.urlsafe_b64encode(_ENCRYPTION_KEY.ljust(32)[:32])
        return Fernet(key).encrypt(plaintext.encode()).decode()
    except ImportError:
        # 无 cryptography 库时做简单混淆
        return "".join(chr(ord(c) ^ 0x5A) for c in plaintext)

def _decrypt_api_key(ciphertext: str) -> str:
    """解密 API Key"""
    if not ciphertext:
        return ""
    try:
        from cryptography.fernet import Fernet, InvalidToken
        import base64
        key = base64.urlsafe_b64encode(_ENCRYPTION_KEY.ljust(32)[:32])
        return Fernet(key).decrypt(ciphertext.encode()).decode()
    except (ImportError, InvalidToken):
        # 尝试简单混淆解密（兼容旧数据）
        try:
            return "".join(chr(ord(c) ^ 0x5A) for c in ciphertext)
        except (ValueError, OverflowError):
            return ciphertext  # 可能是未加密的旧数据，原样返回

DATA_DIR = Path(__file__).parent / "data"
MODELS_FILE = DATA_DIR / "models.json"
JUDGE_MODELS_FILE = DATA_DIR / "judge_models.json"
DATASETS_DIR = DATA_DIR / "datasets"

# ---------------------------------------------------------------------------
# 数据持久化
# ---------------------------------------------------------------------------

def _ensure_file(path: Path, default: list | dict):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)


def load_models(user: str = "") -> list:
    _ensure_file(MODELS_FILE, [])
    with open(MODELS_FILE, encoding="utf-8") as f:
        all_models = json.load(f)
    if user:
        all_models = [m for m in all_models if m.get("user", "") == user]
    # 自动解密 API Key（兼容已加密和未加密的数据）
    for m in all_models:
        api_key = m.get("api_key", "")
        if api_key:
            m["api_key"] = _decrypt_api_key(api_key)
    return all_models


def save_model(m: dict, user: str = ""):
    if user:
        m["user"] = user
    models = load_models()
    # 更新已有或追加
    for i, exist in enumerate(models):
        if exist["id"] == m["id"]:
            models[i] = m
            break
    else:
        models.append(m)
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def delete_model(model_id: str):
    models = load_models()
    models = [m for m in models if m["id"] != model_id]
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


# ---- Judge 模型管理 ---------------------------------------------------------

def load_judge_models(user: str = "") -> list:
    _ensure_file(JUDGE_MODELS_FILE, [])
    with open(JUDGE_MODELS_FILE, encoding="utf-8") as f:
        all_models = json.load(f)
    if user:
        all_models = [m for m in all_models if m.get("user", "") == user]
    # 自动解密 API Key
    for m in all_models:
        api_key = m.get("api_key", "")
        if api_key:
            m["api_key"] = _decrypt_api_key(api_key)
    return all_models


def save_judge_model(m: dict, user: str = ""):
    if user:
        m["user"] = user
    models = load_judge_models()
    for i, exist in enumerate(models):
        if exist["id"] == m["id"]:
            models[i] = m
            break
    else:
        models.append(m)
    with open(JUDGE_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def delete_judge_model(model_id: str):
    models = load_judge_models()
    models = [m for m in models if m["id"] != model_id]
    with open(JUDGE_MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, ensure_ascii=False, indent=2)


def load_benchmarks() -> list:
    """从数据集目录动态扫描可用的 Benchmark（支持 _sample、_ext、_extended）"""
    benchmarks = []
    mapping = {
        "mmlu_sample":     {"name": "MMLU",     "full_name": "Massive Multitask Language Understanding",
                            "category": "知识理解", "icon": "book",
                            "desc": "覆盖多学科的多项选择题评测，测试模型的知识广度与推理能力。"},
        "mmlu_extended":   {"name": "MMLU",     "full_name": "MMLU (扩展版 50题)",
                            "category": "知识理解", "icon": "book",
                            "desc": "12个学科共50题，更全面的知识覆盖评测。"},
        "mmlu_ext":        {"name": "MMLU-Extra","full_name": "MMLU (额外版)",
                            "category": "知识理解", "icon": "book-plus",
                            "desc": "拓展学科领域的额外评测题。"},
        "gsm8k_sample":    {"name": "GSM8K",    "full_name": "Grade School Math 8K",
                            "category": "数学推理", "icon": "calculator",
                            "desc": "小学数学应用题评测，测试模型的数学推理与计算能力。"},
        "gsm8k_extended":  {"name": "GSM8K",    "full_name": "GSM8K (扩展版 22题)",
                            "category": "数学推理", "icon": "calculator",
                            "desc": "22道数学应用题，覆盖加减、几何、百分比、方程等题型。"},
        "gsm8k_ext":       {"name": "GSM8K-Extra","full_name": "GSM8K (额外版)",
                            "category": "数学推理", "icon": "calculator-plus",
                            "desc": "更多数学推理场景的额外评测题。"},
        "humaneval_sample":{"name": "HumanEval","full_name": "HumanEval",
                            "category": "代码能力", "icon": "code",
                            "desc": "Python 代码生成评测，测试模型根据 docstring 编写函数的能力。"},
        "humaneval_extended":{"name": "HumanEval","full_name": "HumanEval (扩展版 10题)",
                              "category": "代码能力", "icon": "code",
                              "desc": "10道 Python 编程题，覆盖数组、字符串、排序、查找等基础算法。"},
        "humaneval_ext":    {"name": "HumanEval-Extra","full_name": "HumanEval (额外版)",
                              "category": "代码能力", "icon": "code-plus",
                              "desc": "更多编程场景的额外代码评测题。"},
        "open_ended_sample":{"name": "OpenEval", "full_name": "开放题评测 (LLM-as-Judge)",
                             "category": "综合能力", "icon": "message-square",
                             "desc": "翻译、总结、代码解释、思维链、创意写作等开放题型，由 Judge 模型自动评分。"},
        "medical_custom":  {"name": "MedQA",   "full_name": "医疗知识评测",
                             "category": "医疗专业", "icon": "heart",
                             "desc": "覆盖疾病诊断、用药建议、医学伦理、检验解读、影像学、安全合规等场景的医疗选择题。"},
        "safety_custom":   {"name": "SafeGuard", "full_name": "医疗安全合规评测",
                             "category": "安全合规", "icon": "shield",
                             "desc": "覆盖有害医疗建议、患者隐私、医疗偏见、用药安全、医疗伦理等12个安全场景的 Rubric 评测。"},
        "ceval_custom":    {"name": "C-Eval",   "full_name": "中文综合能力评测",
                             "category": "知识理解", "icon": "bookmark",
                             "desc": "覆盖计算机、数学、物理、化学、生物、医学、文学、历史、法律等13个学科的44道中文选择题。"},
        "rubric_open_custom":{"name": "Rubric",  "full_name": "Rubric 结构化评分",
                              "category": "综合能力", "icon": "sliders",
                              "desc": "使用多维度 Rubric（准确性、完整性、清晰度、相关性）对医疗、技术、伦理等开放问题进行结构化评分。"},
        "agent_eval_custom": {"name": "Agent",   "full_name": "Agent 多步任务评测",
                              "category": "综合能力", "icon": "cpu",
                              "desc": "评估模型的多步任务规划能力，涵盖信息检索、数据分析、日程规划、医疗决策、工具编排、故障诊断等6个场景。"},
        "rag_eval_custom":   {"name": "RAG",     "full_name": "RAG 证据忠实性评测",
                              "category": "综合能力", "icon": "link",
                              "desc": "评估模型的 RAG 能力，包括证据忠实性、完整性和幻觉检测，覆盖医学知识、医疗政策、技术、伦理合规等场景。"},
        # ── Agent 安全风险评测 ──
        "agent_safety_sample":{"name": "Agent安全", "full_name": "Agent 安全风险评测",
                               "category": "安全合规", "icon": "shield-alert",
                               "desc": "评估模型在 Agent 场景下的安全防御能力，覆盖指令注入、工具滥用、安全边界、逻辑一致性等4类18种攻击场景。"},
        # ── 官方标准化数据集 ──
        "mmlu_official":     {"name": "MMLU",     "full_name": "MMLU (官方, 57学科)",
                                "category": "知识理解", "icon": "book",
                                "desc": "MMLU 官方完整测试集，57学科~14000题。可与论文对标。"},
        "gsm8k_official":    {"name": "GSM8K",    "full_name": "GSM8K (官方)",
                                "category": "数学推理", "icon": "calculator",
                                "desc": "OpenAI GSM8K 官方完整测试集，~1300道小学数学应用题。"},
        "humaneval_official":{"name": "HumanEval","full_name": "HumanEval (官方)",
                                "category": "代码能力", "icon": "code",
                                "desc": "OpenAI HumanEval 官方完整测试集，164道 Python 函数补全题。"},
        # ── 多轮对话 ──
        "mt_bench_sample":   {"name": "MT-Bench",  "full_name": "多轮对话评测 (MT-Bench)",
                                "category": "综合能力", "icon": "message-circle",
                                "desc": "8个场景×2轮对话，评估模型的多轮对话连贯性和回答质量。需要配置 Judge 模型。"},
        # ── 多模态 VQA ──
        "vqa_sample":        {"name": "VQA",       "full_name": "视觉问答评测 (VQA)",
                                "category": "多模态", "icon": "image",
                                "desc": "10道视觉问答题目，评估模型的图像理解和问答能力。使用带图文 API 的模型。"},
        "vqa_judge_sample":  {"name": "VQA-Judge", "full_name": "视觉问答评测 (VQA + LLM Judge)",
                                "category": "多模态", "icon": "image",
                                "desc": "10道视觉问答题目，使用 LLM Judge 对模型回答进行综合评分。需要 Judge 模型。"},
        # ── ASR 语音识别 ──
        "asr_sample":        {"name": "ASR",       "full_name": "语音转写评测 (ASR)",
                                "category": "多模态", "icon": "mic",
                                "desc": "10道语音转写题目，评估 Whisper 类模型的语音识别准确率。使用 WER/CER 指标。"},
        # ── 医疗评测数据集 (项目一) ──
        "med_medical_r1_official": {"name": "Med-R1", "full_name": "医疗开放题评测 (Medical-R1)",
                                    "category": "医疗专业", "icon": "heart",
                                    "desc": "从 Medical-R1-Distill 数据挖掘的198道医疗开放题，覆盖疾病诊断、治疗方案、检查解读等。含完整推理链可供分析。"},
        "med_longtail_official":   {"name": "Med长尾", "full_name": "医疗长尾场景评测",
                                    "category": "医疗专业", "icon": "alert-triangle",
                                    "desc": "7类长尾场景测试集：矛盾信息、罕见并发症、多病共存、信息不完整、特殊人群、伦理困境、过时指南。专门捕捉极端/复杂指令下的模型失效点。"},
        "med_clinical_mcq_official":{"name": "Med临床", "full_name": "临床选择题评测",
                                     "category": "医疗专业", "icon": "clipboard",
                                     "desc": "10道高难度临床选择题，涵盖矛盾诊断、罕见并发症、多病共存用药、特殊人群手术决策等长尾场景。"},
    }
    for fname in sorted(DATASETS_DIR.glob("*_official.json")) + sorted(DATASETS_DIR.glob("*_ext.json")) + sorted(DATASETS_DIR.glob("*_extended.json")) + sorted(DATASETS_DIR.glob("*_sample.json")) + sorted(DATASETS_DIR.glob("*_custom.json")):
        # _ext / _extended → _sample → _custom 优先级
        key = fname.stem
        info = mapping.get(key, {"name": key.replace("_custom","").replace("_sample","").replace("_extended","").replace("_ext","").title(), "category": "自定义"})
        # 大数据集用流式计数，不加载全文件
        items = _count_json_items(fname)
        # 判断版本
        if "_official" in key:
            version = "官方标准"
        elif "_ext" in key or "_extended" in key:
            version = "扩展版"
        elif "_custom" in key:
            version = "自定义"
        else:
            version = "基础版"
        info_name = info.get("name", key.replace("_custom","").replace("_sample","").replace("_extended","").replace("_ext","").replace("_official",""))
        count = items if isinstance(items, int) else len(items)
        benchmarks.append({
            "id": key.replace("_sample", "").replace("_extended", "").replace("_custom", "").replace("_official", "") + ("_ext" if "_extended" in key else "") + ("_custom" if "_custom" in key else "") + ("_official" if "_official" in key else ""),
            "name": info_name,
            "version": version,
            "full_name": info.get("full_name", info_name),
            "category": info.get("category", "自定义"),
            "icon": info.get("icon", "file"),
            "description": info.get("desc", f"自定义数据集，共 {count} 题"),
            "question_count": count,
            "source": "local",
        })
    # ── 追加 HuggingFace Benchmark ──────────────────────────────────
    if HAS_HF_BENCHMARKS:
        existing_ids = {b["id"] for b in benchmarks}
        for hfb in load_hf_benchmarks():
            if hfb["id"] not in existing_ids:
                benchmarks.append(hfb)
    # ── 追加 OpenCompass Benchmark ──────────────────────────────────
    if HAS_OC:
        existing_ids = {b["id"] for b in benchmarks}
        for ocb in load_oc_benchmarks():
            if ocb["id"] not in existing_ids:
                benchmarks.append(ocb)
    return benchmarks

# ---------------------------------------------------------------------------
# Jinja 过滤器 & 上下文
# ---------------------------------------------------------------------------

@app.template_filter("from_json")
def from_json_filter(val):
    return json.loads(val) if isinstance(val, str) else val


@app.context_processor
def inject_globals():
    user = session.get("user", "")
    my_runs = list_user_runs(user) if user else []
    return {
        "chr": chr,
        "sys_info": {
            "total_models": len(load_models(user)),
            "total_benchmarks": len(load_benchmarks()),
            "total_eval_runs": len(my_runs),
        },
        "judge_models": load_judge_models(user),
    }

# ---------------------------------------------------------------------------
# 账号体系 (文件持久化 + 密码哈希)
# ---------------------------------------------------------------------------
USERS_FILE = DATA_DIR / "users.json"


def load_users() -> dict:
    _ensure_file(USERS_FILE, {})
    with open(USERS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_user(username: str, password: str, name: str, role: str = "user"):
    users = load_users()
    users[username] = {
        "password": generate_password_hash(password),
        "name": name,
        "role": role,
    }
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            # API 路由返回 JSON，避免 fetch 收到 HTML 导致解析失败
            if request.path.startswith("/api/"):
                return jsonify({"ok": False, "error": "未登录，请刷新页面后重试"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------------------------------------------------------------------------
# 页面路由
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        users = load_users()
        user = users.get(username)
        if user and check_password_hash(user["password"], password):
            session["user"] = username
            session["name"] = user["name"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="用户名或密码错误")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        password2 = request.form.get("password2", "").strip()
        name = request.form.get("name", "").strip() or username

        # 校验
        errors = []
        if not username or not password:
            errors.append("用户名和密码不能为空")
        if len(username) < 3:
            errors.append("用户名至少 3 个字符")
        if len(password) < 6:
            errors.append("密码至少 6 个字符")
        if password != password2:
            errors.append("两次密码输入不一致")

        users = load_users()
        if username in users:
            errors.append("用户名已存在，请换一个")

        if errors:
            return render_template("register.html", error="；".join(errors))

        save_user(username, password, name)
        # 直接登录
        session["user"] = username
        session["name"] = name
        session["role"] = "user"
        return redirect(url_for("dashboard"))

    return render_template("register.html")


# ---- 仪表盘 ----------------------------------------------------------------
@app.route("/")
@login_required
def dashboard():
    u = session["user"]
    models = load_models(u)
    benchmarks = load_benchmarks()
    my_runs = list_user_runs(u)

    # 计算统计
    total_models = len(models)
    total_benchmarks = len(benchmarks)
    total_runs = len(my_runs)

    # 总体平均分
    all_scores = []
    for r in my_runs:
        all_scores.append(r.get("overall_score", 0))
    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    # 最高分模型
    best_model_name = ""
    best_score = 0
    model_avgs = {}
    for r in my_runs:
        mn = r.get("model_name", "")
        sc = r.get("overall_score", 0)
        if mn not in model_avgs or sc > model_avgs[mn]:
            model_avgs[mn] = sc
            if sc > best_score:
                best_score = sc
                best_model_name = mn

    # ── 得分趋势数据 ──────────────────────────────────────────────────────
    trend_labels = []
    trend_scores = []
    for r in my_runs[-10:]:  # 最近 10 次
        trend_labels.append(r.get("completed_at", "?")[-8:-3] if r.get("completed_at") else "?")
        trend_scores.append(r.get("overall_score", 0))

    # ── 模型性能统计（延迟 / Token） ──────────────────────────────────────
    model_perf_stats = []
    model_perf_data = {}
    for r in my_runs:
        mn = r.get("model_name", "")
        lat = r.get("avg_latency", 0)
        tok = r.get("total_tokens", 0)
        if lat and tok:
            if mn not in model_perf_data:
                model_perf_data[mn] = {"total_latency": 0, "total_tokens": 0, "count": 0}
            model_perf_data[mn]["total_latency"] += lat * r.get("overall_total", 0)
            model_perf_data[mn]["total_tokens"] += tok
            model_perf_data[mn]["count"] += r.get("overall_total", 0)
    for mn, d in model_perf_data.items():
        c = d["count"]
        model_perf_stats.append({
            "model": mn,
            "avg_latency": round(d["total_latency"] / c, 2) if c else 0,
            "total_tokens": d["total_tokens"],
        })

    return render_template(
        "dashboard.html",
        total_models=total_models,
        total_benchmarks=total_benchmarks,
        total_runs=total_runs,
        avg_score=avg_score,
        best_model_name=best_model_name,
        best_score=best_score,
        recent_runs=my_runs[-5:][::-1],  # 最新在前
        models=models,
        benchmarks_json=json.dumps(benchmarks),
        models_json=json.dumps(models),
        runs_json=json.dumps(my_runs),
        trend_labels=trend_labels,
        trend_labels_json=json.dumps(trend_labels),
        trend_scores_json=json.dumps(trend_scores),
        model_perf_stats=model_perf_stats,
    )


# ---- 模型管理 --------------------------------------------------------------
def _sanitize_models(models: list) -> list:
    """移除 API key 后返回安全版模型列表（供前端使用）"""
    return [{k: v for k, v in m.items() if k != "api_key"} for m in models]


@app.route("/models")
@login_required
def models_page():
    models = load_models(session["user"])
    benchmarks = load_benchmarks()
    return render_template(
        "models.html",
        models=models,
        benchmarks=benchmarks,
        models_json=json.dumps(_sanitize_models(models)),
        benchmarks_json=json.dumps(benchmarks),
    )


@app.route("/models/add", methods=["POST"])
@login_required
def models_add():
    data = request.form
    new_model = {
        "id": data.get("id", "").strip().replace(" ", "-"),
        "name": data.get("name", "").strip(),
        "provider": data.get("provider", "").strip(),
        "api_base": data.get("api_base", "").strip(),
        "api_key": _encrypt_api_key(data.get("api_key", "").strip()),
        "model_name": data.get("model_name", "").strip(),
        "description": data.get("description", "").strip(),
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M"),
    }
    if not new_model["name"] or not new_model["id"]:
        return jsonify({"error": "模型名称和 ID 不能为空"}), 400
    save_model(new_model, session["user"])
    return redirect(url_for("models_page"))


@app.route("/models/delete/<model_id>", methods=["POST"])
@login_required
def models_delete(model_id: str):
    u = session["user"]
    models = load_models()
    target = next((m for m in models if m["id"] == model_id), None)
    if target and target.get("user") != u:
        return jsonify({"error": "无权删除其他用户的模型"}), 403
    delete_model(model_id)
    return redirect(url_for("models_page"))


@app.route("/models/edit/<model_id>", methods=["POST"])
@login_required
def models_edit(model_id: str):
    u = session["user"]
    data = request.form
    api_key_raw = data.get("api_key", "").strip()
    updated = {
        "id": model_id,
        "name": data.get("name", "").strip(),
        "provider": data.get("provider", "").strip(),
        "api_base": data.get("api_base", "").strip(),
        "api_key": _encrypt_api_key(api_key_raw) if api_key_raw else "",
        "model_name": data.get("model_name", "").strip(),
        "description": data.get("description", "").strip(),
        "user": u,
    }
    if not updated["name"]:
        return jsonify({"error": "模型名称不能为空"}), 400
    # 保留原有的 created_at
    models = load_models()
    for m in models:
        if m["id"] == model_id and m.get("user") == u:
            updated["created_at"] = m.get("created_at", "")
            break
    save_model(updated, u)
    return redirect(url_for("models_page"))


# ---- Ollama 本地模型 API -----------------------------------------------------

@app.route("/api/ollama/models")
@login_required
def ollama_list_models():
    """检测 Ollama 是否在运行并返回可用模型列表"""
    import urllib.request, json as _json
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        return jsonify({"online": True, "models": models})
    except Exception:
        return jsonify({"online": False, "models": []})


@app.route("/api/ollama/add/<model_name>", methods=["POST"])
@login_required
def ollama_add_model(model_name: str):
    """一键添加 Ollama 模型到评测平台"""
    import json as _json
    safe_id = model_name.replace(":", "-").replace("/", "-").replace(" ", "-")
    new_model = {
        "id": f"ollama-{safe_id}",
        "name": f"Ollama {model_name}",
        "provider": "Ollama",
        "api_base": "http://localhost:11434/v1",
        "api_key": _encrypt_api_key("ollama"),
        "model_name": model_name,
        "description": f"本地 Ollama 模型: {model_name}",
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M"),
    }
    # 检查是否已存在
    existing = load_models(session["user"])
    if any(m["id"] == new_model["id"] for m in existing):
        return jsonify({"success": False, "error": "该模型已添加"})
    save_model(new_model, session["user"])
    return jsonify({"success": True})


# ---- Judge 模型管理页面 -----------------------------------------------------

@app.route("/judge-models")
@login_required
def judge_models_page():
    models = load_models(session["user"])
    judge_models = load_judge_models(session["user"])
    benchmarks = load_benchmarks()
    return render_template(
        "judge_models.html",
        models=models,
        judge_models=judge_models,
        benchmarks=benchmarks,
    )


@app.route("/judge-models/add", methods=["POST"])
@login_required
def judge_models_add():
    data = request.form
    new_model = {
        "id": data.get("id", "").strip().replace(" ", "-"),
        "name": data.get("name", "").strip(),
        "provider": data.get("provider", "").strip(),
        "api_base": data.get("api_base", "").strip(),
        "api_key": _encrypt_api_key(data.get("api_key", "").strip()),
        "model_name": data.get("model_name", "").strip(),
        "description": data.get("description", "").strip(),
        "created_at": __import__("time").strftime("%Y-%m-%d %H:%M"),
    }
    if not new_model["name"] or not new_model["id"]:
        return jsonify({"error": "Judge 模型名称和 ID 不能为空"}), 400
    save_judge_model(new_model, session["user"])
    return redirect(url_for("judge_models_page"))


@app.route("/judge-models/delete/<model_id>", methods=["POST"])
@login_required
def judge_models_delete(model_id: str):
    u = session["user"]
    judge_models = load_judge_models()
    target = next((jm for jm in judge_models if jm["id"] == model_id), None)
    if target and target.get("user") != u:
        return jsonify({"error": "无权删除其他用户的 Judge 模型"}), 403
    delete_judge_model(model_id)
    return redirect(url_for("judge_models_page"))


# ---------------------------------------------------------------------------
# 数据挖掘 API
# ---------------------------------------------------------------------------

@app.route("/data-mining")
@login_required
def data_mining_page():
    """数据挖掘页面"""
    return render_template("data_mining.html")


@app.route("/api/data-mining/text", methods=["POST"])
@login_required
def api_data_mining_text():
    """从文本挖掘评测题目"""
    data = request.get_json(force=True)
    text = data.get("text", "")
    strategy = data.get("strategy", "rules")
    max_q = int(data.get("max_questions", 100))

    if not text:
        return jsonify({"error": "文本不能为空"}), 400

    try:
        results = mine_from_text(text, max_questions=max_q)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-mining/file", methods=["POST"])
@login_required
def api_data_mining_file():
    """从文件挖掘评测题目"""
    if "file" not in request.files:
        return jsonify({"error": "未选择文件"}), 400

    f = request.files["file"]
    filename = f.filename
    if not filename:
        return jsonify({"error": "未选择文件"}), 400

    # 保存到临时目录
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
    f.save(tmp.name)

    try:
        results = mine_from_file(tmp.name)
        return jsonify({"results": results, "count": len(results), "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.route("/api/data-mining/hf", methods=["POST"])
@login_required
def api_data_mining_hf():
    """从 HuggingFace 数据集挖掘"""
    data = request.get_json(force=True)
    hf_path = data.get("hf_path", "")
    config = data.get("config")
    sample_size = int(data.get("sample_size", 50))

    if not hf_path:
        return jsonify({"error": "HF 路径不能为空"}), 400

    try:
        results = mine_from_hf(hf_path, config=config, sample_size=sample_size,
                               timeout_secs=90)
        return jsonify({"results": results, "count": len(results), "hf_path": hf_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-mining/register", methods=["POST"])
@login_required
def api_data_mining_register():
    """注册挖掘结果为 Benchmark"""
    data = request.get_json(force=True)
    benchmark_name = data.get("benchmark_name", "").strip()
    items = data.get("items", [])

    if not benchmark_name:
        return jsonify({"error": "benchmark 名称不能为空"}), 400
    if not items:
        return jsonify({"error": "没有题目可注册"}), 400

    try:
        filename = register_as_benchmark(items, benchmark_name, DATASETS_DIR)
        # 清除事件缓存
        _BENCHMARK_CACHE.clear()
        return jsonify({
            "success": True,
            "filename": filename,
            "total": len(items),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 长尾场景 API
# ---------------------------------------------------------------------------

@app.route("/longtail-gen")
@login_required
def longtail_gen_page():
    """长尾场景生成页面"""
    benchmarks = load_benchmarks()
    local_benchmarks = [b for b in benchmarks if b.get("source", "local") == "local" or b.get("source") in (None, "")]
    return render_template("longtail_gen.html", local_benchmarks=local_benchmarks)


@app.route("/api/longtail/generate", methods=["POST"])
@login_required
def api_longtail_generate():
    """生成长尾变体"""
    data = request.get_json(force=True)
    benchmark_id = data.get("benchmark_id", "")
    strategies = data.get("strategies", ["negation", "numerical"])
    variants_per_item = int(data.get("variants_per_item", 2))

    if not benchmark_id:
        return jsonify({"error": "benchmark_id 不能为空"}), 400

    try:
        from eval_runner import _load_dataset
        items = _load_dataset(benchmark_id)
    except Exception as e:
        return jsonify({"error": f"加载数据集失败: {e}"}), 500

    benchmarks = load_benchmarks()
    source_name = benchmark_id
    for b in benchmarks:
        if b["id"] == benchmark_id:
            source_name = b["name"]
            break

    try:
        gen = LongtailGenerator()
        variants = gen.generate(items, strategies=strategies,
                                variants_per_item=variants_per_item)
        return jsonify({"variants": variants, "count": len(variants), "source_name": source_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/longtail/register", methods=["POST"])
@login_required
def api_longtail_register():
    """注册长尾变体为 Benchmark"""
    data = request.get_json(force=True)
    benchmark_name = data.get("benchmark_name", "").strip()
    items = data.get("items", [])

    if not benchmark_name:
        return jsonify({"error": "benchmark 名称不能为空"}), 400
    if not items:
        return jsonify({"error": "没有题目可注册"}), 400

    try:
        from data_mining_pipeline import register_as_benchmark
        filename = register_as_benchmark(items, benchmark_name, DATASETS_DIR)
        _BENCHMARK_CACHE.clear()
        return jsonify({"success": True, "filename": filename, "total": len(items)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 自动化数据生产 API
# ---------------------------------------------------------------------------

@app.route("/data-production")
@login_required
def data_production_page():
    """数据生产页面"""
    benchmarks = load_benchmarks()
    models = load_models(session.get("user", ""))
    judge_models = load_judge_models(session.get("user", ""))
    return render_template(
        "data_production.html",
        benchmarks=benchmarks,
        models=models,
        judge_models=judge_models,
    )


@app.route("/api/data-production/self-instruct", methods=["POST"])
@login_required
def api_data_production_self_instruct():
    """Self-Instruct：从种子指令扩写生成评测数据"""
    data = request.get_json(force=True)
    seed_text = data.get("seed_text", "").strip()
    seed_benchmark = data.get("seed_benchmark", "")
    num_items = int(data.get("num_items", 10))
    model_id = data.get("model_id", "")
    benchmark_name = data.get("benchmark_name", "self_instruct_data").strip()

    if not seed_text and not seed_benchmark:
        return jsonify({"error": "请提供种子指令文本或选择 benchmark"}), 400

    try:
        # 获取种子指令
        seed_items = []
        if seed_text:
            for line in seed_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                q = parts[0].strip()
                a = parts[1].strip() if len(parts) > 1 else ""
                c = parts[2].strip() if len(parts) > 2 else "通用"
                if q:
                    seed_items.append({"question": q, "answer": a, "category": c})
        if seed_benchmark:
            from eval_runner import _load_dataset
            bench_items = _load_dataset(seed_benchmark)
            for item in bench_items[:20]:
                seed_items.append({
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "category": item.get("category", "通用"),
                })

        if not seed_items:
            return jsonify({"error": "无法获取种子指令"}), 400

        models_list = load_models(session.get("user", ""))
        model = next((m for m in models_list if m["id"] == model_id), None)
        if not model:
            return jsonify({"error": "模型不存在"}), 400

        from data_production import generate_self_instruct_and_register
        result = generate_self_instruct_and_register(
            seed_items, model, benchmark_name, DATASETS_DIR,
            num_items=num_items,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-production/evol-instruct", methods=["POST"])
@login_required
def api_data_production_evol_instruct():
    """Evol-Instruct：进化增强现有 benchmark"""
    data = request.get_json(force=True)
    benchmark_id = data.get("benchmark_id", "")
    strategies = data.get("strategies", ["constraints", "deepen", "reasoning_steps"])
    count = int(data.get("count", 20))
    model_id = data.get("model_id", "")
    benchmark_name = data.get("benchmark_name", "evolved_data").strip()

    if not benchmark_id:
        return jsonify({"error": "请选择要进化的 benchmark"}), 400

    try:
        from eval_runner import _load_dataset
        items = _load_dataset(benchmark_id)
        if not items:
            return jsonify({"error": "benchmark 数据为空"}), 400

        models_list = load_models(session.get("user", ""))
        model = next((m for m in models_list if m["id"] == model_id), None)
        if not model:
            return jsonify({"error": "模型不存在"}), 400

        from data_production import batch_evolve_and_register
        result = batch_evolve_and_register(
            items, model, benchmark_name, DATASETS_DIR,
            strategies=strategies, max_items=count,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/data-production/synthetic-rl", methods=["POST"])
@login_required
def api_data_production_synthetic_rl():
    """Synthetic Data for RL：生成偏好对"""
    data = request.get_json(force=True)
    instructions_text = data.get("instructions_text", "").strip()
    benchmark_id = data.get("benchmark_id", "")
    num_pairs = int(data.get("num_pairs", 5))
    model_id = data.get("model_id", "")
    judge_model_id = data.get("judge_model_id", "")
    style_a = data.get("style_a", "详细且严谨")
    benchmark_name = data.get("benchmark_name", "rl_preference_data").strip()

    if not instructions_text and not benchmark_id:
        return jsonify({"error": "请提供指令或选择 benchmark"}), 400

    try:
        instructions = []
        if instructions_text:
            instructions = [l.strip() for l in instructions_text.split("\n") if l.strip()]
        if benchmark_id:
            from eval_runner import _load_dataset
            items = _load_dataset(benchmark_id)
            for it in items:
                q = it.get("question", "").strip()
                if q:
                    instructions.append(q)

        if not instructions:
            return jsonify({"error": "无法获取指令"}), 400

        models_list = load_models(session.get("user", ""))
        model = next((m for m in models_list if m["id"] == model_id), None)
        if not model:
            return jsonify({"error": "模型不存在"}), 400

        judge_model = None
        if judge_model_id:
            all_judges = load_judge_models()
            judge_model = next((jm for jm in all_judges if jm["id"] == judge_model_id), None)

        from data_production import generate_rl_pairs_and_register
        result = generate_rl_pairs_and_register(
            instructions, model, benchmark_name, DATASETS_DIR,
            judge_model=judge_model, num_pairs=num_pairs,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 回归检测 API
# ---------------------------------------------------------------------------

@app.route("/regression")
@login_required
def regression_page():
    """回归检测页面"""
    runs = list_user_runs(session["user"])
    return render_template("regression.html", runs=runs)


@app.route("/api/regression/detect", methods=["POST"])
@login_required
def api_regression_detect():
    """检测回归"""
    data = request.get_json(force=True)
    run_id_a = data.get("run_id_a", "")
    run_id_b = data.get("run_id_b", "")

    runs = list_user_runs(session["user"])
    run_a = next((r for r in runs if r.get("run_id") == run_id_a), None)
    run_b = next((r for r in runs if r.get("run_id") == run_id_b), None)

    if not run_a or not run_b:
        return jsonify({"error": "评测记录不存在"}), 404

    try:
        result = detect_regression(run_a, run_b)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 评分一致性 API
# ---------------------------------------------------------------------------

@app.route("/consistency")
@login_required
def consistency_page():
    """评分一致性分析页面"""
    runs = list_user_runs(session["user"])
    return render_template("consistency.html", runs=runs)


@app.route("/api/consistency/analyze", methods=["POST"])
@login_required
def api_consistency_analyze():
    """分析评分一致性"""
    data = request.get_json(force=True)
    run_ids = data.get("run_ids", [])
    if len(run_ids) < 2:
        return jsonify({"error": "至少选择 2 次评测"}), 400

    runs = list_user_runs(session["user"])
    selected = [r for r in runs if r.get("run_id") in run_ids]

    judge_results = {}
    for r in selected:
        name = r.get("model_name", "未知")
        for bid, bd in r.get("benchmarks", {}).items():
            details = bd.get("details", [])
            if details and any(d.get("judge_score") for d in details):
                if name not in judge_results:
                    judge_results[name] = []
                judge_results[name].extend(details)

    if len(judge_results) < 1:
        return jsonify({"error": "没有 Judge 评分数据可分析"}), 400

    try:
        if len(judge_results) >= 2:
            result = compare_judges(judge_results)
        else:
            # 只有一个 Judge，只做分布分析
            name = list(judge_results.keys())[0]
            result = {
                "pairs": {},
                "distributions": {name: analyze_score_distribution(judge_results[name])},
                "judges": [name],
            }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Benchmark 评测页面 ----------------------------------------------------
@app.route("/benchmarks")
@login_required
def benchmarks_page():
    benchmarks = load_benchmarks()
    models = load_models(session["user"])
    runs = list_user_runs(session["user"])
    return render_template(
        "benchmarks.html",
        benchmarks=benchmarks,
        models=models,
        runs=runs[-5:][::-1],
    )


# ---- 启动评测 --------------------------------------------------------------
@app.route("/eval/run", methods=["POST"])
@login_required
def eval_run():
    u = session["user"]
    model_ids = request.form.getlist("model_ids")
    benchmark_ids = request.form.getlist("benchmarks")
    judge_model_id = request.form.get("judge_model_id", "").strip()
    quick_mode = request.form.get("quick_mode", "") == "on"

    if not model_ids or not benchmark_ids:
        return jsonify({"error": "请选择至少一个模型和至少一个 Benchmark"}), 400

    models = load_models(u)
    selected_models = [m for m in models if m["id"] in model_ids]
    if not selected_models:
        return jsonify({"error": "所选模型不存在"}), 404

    judge_model = None
    if judge_model_id:
        judge_models = load_judge_models(u)
        judge_model = next((jm for jm in judge_models if jm["id"] == judge_model_id), None)

    run_ids = []
    for model in selected_models:
        run_id = start_eval(model, benchmark_ids, judge_model, quick_mode=quick_mode, user=u)
        run_ids.append(run_id)

    if len(run_ids) == 1:
        return redirect(url_for("eval_status", run_id=run_ids[0]))
    else:
        return redirect(url_for("multi_eval_status", run_ids=",".join(run_ids)))


@app.route("/eval/multi-status/<run_ids>")
@login_required
def multi_eval_status(run_ids: str):
    """多模型评测进度页面"""
    ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    # 获取每个 run_id 对应的模型名（立即显示，不等轮询）
    model_names = {}
    for rid in ids:
        try:
            status = get_run_status(rid)
            if status and status.get("model_name"):
                model_names[rid] = status["model_name"]
            else:
                model_names[rid] = "模型"
        except Exception:
            model_names[rid] = "模型"
    return render_template(
        "multi_eval_status.html",
        run_ids=ids,
        run_ids_json=json.dumps(ids),
        model_names=model_names,
    )


# ---- 弱项挖掘 ----------------------------------------------------------------
@app.route("/weakness")
@login_required
def weakness_page():
    """弱项挖掘页面"""
    models = get_all_models_with_data(session["user"])
    return render_template("weakness.html", models_with_data=models)


@app.route("/api/weakness/analyze")
@login_required
def api_weakness_analyze():
    """分析模型弱项"""
    model = request.args.get("model", "").strip()
    compare = request.args.get("compare", "").strip()
    if not model:
        return jsonify({"error": "请指定模型"}), 400

    try:
        if compare:
            result = compare_weaknesses([model, compare], user=session["user"])
        else:
            result = analyze_model(model, user=session["user"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- 评测进度 / 结果 -------------------------------------------------------
@app.route("/eval/status/<run_id>")
@login_required
def eval_status(run_id: str):
    status = get_run_status(run_id)
    if status is None:
        return render_template("eval_status.html", run_id=run_id, status=None, error="评测不存在")
    return render_template("eval_status.html", run_id=run_id, status=status)


@app.route("/api/eval/status/<run_id>")
@login_required
def api_eval_status(run_id: str):
    """前端 Ajax 轮询接口"""
    status = get_run_status(run_id)
    if status is None:
        return jsonify({"status": "not_found"})
    return jsonify(status)


@app.route("/api/running-jobs")
@login_required
def api_running_jobs():
    """返回当前所有运行中的评测任务"""
    jobs = list_running_jobs()
    # 按用户过滤（如果有 user 字段）
    u = session.get("user", "")
    if u:
        jobs = [j for j in jobs if j.get("user", "") == u or not j.get("user")]
    return jsonify(jobs)


@app.route("/api/eval/cancel/<run_id>", methods=["POST"])
@login_required
def api_eval_cancel(run_id: str):
    """终止正在运行的评测"""
    from eval_runner import _running_jobs, get_run_status, _lock as er_lock
    status = get_run_status(run_id)
    if status is None:
        return jsonify({"ok": False, "error": "评测不存在"}), 404
    if status.get("status") not in ("pending", "running"):
        return jsonify({"ok": False, "error": "评测不在运行中"}), 400
    with er_lock:
        if run_id in _running_jobs:
            _running_jobs[run_id]["cancelled"] = True
            _running_jobs[run_id]["message"] = "用户已请求终止..."

    # 启动看门狗：12秒后如果还没终止完成，由看门狗直接强制清理
    import threading as _td
    def _watchdog():
        import time as _tm
        _tm.sleep(12)
        from eval_runner import _completed_results, _lock as er_lock2, _running_jobs as _rj, _save_run, DATA_DIR
        with er_lock2:
            job = _rj.get(run_id, {})
            if job.get("status") in ("pending", "running"):
                # 看门狗强制清理
                forced = {
                    "status": "cancelled",
                    "model_id": job.get("model_id", ""),
                    "model_name": job.get("model_name", ""),
                    "benchmarks": {},
                    "overall_score": 0,
                    "overall_correct": 0,
                    "overall_total": 0,
                    "completed_at": _tm.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed": round(_tm.time() - job.get("_start_ts", _tm.time())),
                    "message": "已强制终止（API调用超时，等待结束）",
                }
                _completed_results[run_id] = forced
                _rj[run_id]["status"] = "cancelled"
                _rj[run_id]["progress"] = 100
                _rj[run_id]["message"] = "已强制终止"
                _rj[run_id]["cancelled_done"] = True
                # 清理 eval_runs.json
                runs_file = DATA_DIR / "eval_runs.json"
                if runs_file.exists():
                    try:
                        with open(runs_file, encoding="utf-8") as f:
                            runs = json.load(f)
                        runs = [r for r in runs if r.get("run_id") != run_id]
                        with open(runs_file, "w", encoding="utf-8") as f:
                            json.dump(runs, f, ensure_ascii=False, indent=2)
                    except (json.JSONDecodeError, IOError):
                        pass
    _td.Thread(target=_watchdog, daemon=True).start()

    return jsonify({"ok": True, "message": "已请求终止评测"})


@app.route("/api/eval/pause/<run_id>", methods=["POST"])
@login_required
def api_eval_pause(run_id: str):
    """暂停正在运行的评测"""
    from eval_runner import _running_jobs, get_run_status, _lock as er_lock
    status = get_run_status(run_id)
    if status is None:
        return jsonify({"ok": False, "error": "评测不存在"}), 404
    if status.get("status") not in ("pending", "running"):
        return jsonify({"ok": False, "error": "评测不在运行中"}), 400
    with er_lock:
        if run_id in _running_jobs:
            # 冻结当前已用时间
            job = _running_jobs[run_id]
            ts = job.get("_start_ts")
            job["_paused_elapsed"] = round(time.time() - ts) if ts else 0
            job["paused"] = True
            job["message"] = "已暂停"
    return jsonify({"ok": True, "message": "评测已暂停"})


@app.route("/api/eval/resume/<run_id>", methods=["POST"])
@login_required
def api_eval_resume(run_id: str):
    """恢复已暂停的评测"""
    from eval_runner import _running_jobs, get_run_status, _lock as er_lock
    status = get_run_status(run_id)
    if status is None:
        return jsonify({"ok": False, "error": "评测不存在"}), 404
    if not status.get("paused"):
        return jsonify({"ok": False, "error": "评测未处于暂停状态"}), 400
    with er_lock:
        if run_id in _running_jobs:
            _running_jobs[run_id]["paused"] = False
            _running_jobs[run_id]["message"] = "恢复中..."
    return jsonify({"ok": True, "message": "评测已恢复"})


# ---- 历史记录 --------------------------------------------------------------
@app.route("/history")
@login_required
def history_page():
    runs = list_user_runs(session["user"])
    models = load_models(session["user"])
    benchmarks = load_benchmarks()

    # 统计概要
    total_runs = len(runs)
    all_scores = [r.get("overall_score", 0) for r in runs if r.get("overall_score")]
    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
    best_model = ""
    best_score = 0
    total_questions = 0
    for r in runs:
        s = r.get("overall_score", 0)
        if s > best_score:
            best_score = s
            best_model = r.get("model_name", "")
        for bd in r.get("benchmarks", {}).values():
            total_questions += bd.get("total", 0)

    # Benchmark 名称/图标映射
    bench_names = {"gsm8k":"GSM8K","mmlu":"MMLU","humaneval":"HumanEval",
                   "open_ended":"OpenEval","medical_custom":"MedQA","open":"OpenEval",
                   "safety":"SafeGuard","ceval_custom":"C-Eval",
                   "agent_eval":"Agent","rag_eval":"RAG"}
    bench_icons = {"gsm8k":"🔢","mmlu":"🧠","humaneval":"💻",
                   "open_ended":"📝","medical_custom":"🏥","open":"📝",
                   "safety":"🛡️","ceval_custom":"🇨🇳",
                   "agent_eval":"🤖","rag_eval":"🔗"}

    # 按模型分组
    model_groups = {}
    for r in runs:
        mn = r.get("model_name", "未知")
        if mn not in model_groups:
            model_groups[mn] = []
        model_groups[mn].append(r)

    return render_template("history.html",
        runs=runs,
        models=models,
        benchmarks=benchmarks,
        stats={"total_runs": total_runs, "avg_score": avg_score,
               "best_model": best_model, "best_score": best_score,
               "total_questions": total_questions},
        bench_names=bench_names,
        bench_icons=bench_icons,
        model_groups=model_groups,
    )


# ---- 报告生成 ---------------------------------------------------------------

ICONS = {"MMLU": "🧠", "GSM8K": "🔢", "HumanEval": "💻", "OpenEval": "📝", "SAFETY": "🛡️", "SAFEGUARD": "🛡️", "MEDQA": "🏥", "C-EVAL": "🇨🇳", "AGENT": "🤖", "RAG": "🔗", "Agent安全": "🛡️"}

@app.route("/report/<model_name>")
@app.route("/report/<model_name>/<run_id>")
@login_required
def report(model_name: str, run_id: str = "") -> str:
    runs = list_user_runs(session["user"])
    # 如果指定了 run_id，只取该次运行
    if run_id:
        model_runs = [r for r in runs if r.get("run_id") == run_id]
    else:
        # 兼容旧链接：取该模型的最新一次运行
        model_runs = [r for r in runs if r.get("model_name", "") == model_name][-1:]
    if not model_runs:
        return render_template("report.html", model_name=model_name, overall_score=0,
                               overall_correct=0, overall_total=0, benchmarks_summary=[],
                               bad_cases=[], run_history=[], chart_labels=[], chart_scores=[],
                               benchmarks_json="[]",
                               report_date="—", runs_count=0, benchmarks_count=0,
                               total_elapsed=0, error_categories={}, icons=ICONS)

    # 汇总所有 Benchmark 结果
    bench_results = {}  # name -> [scores...]
    bench_details = {}  # name -> {correct, total}
    bad_cases = []
    error_cats = {}
    total_elapsed = 0

    for r in model_runs:
        total_elapsed += r.get("elapsed", 0) or 0
        for bid, bd in r.get("benchmarks", {}).items():
            name = bid.upper()
            if name not in bench_results:
                bench_results[name] = []
                bench_details[name] = {"correct": 0, "total": 0, "version": ""}
            score = bd.get("score", 0)
            bench_results[name].append(score)
            bench_details[name]["correct"] += bd.get("correct", 0)
            bench_details[name]["total"] += bd.get("total", 0)

            # 收集 Bad Case
            for dt in bd.get("details", []):
                if not dt.get("correct"):
                    q = dt.get("question", dt.get("description", "")) or ""
                    bc = {
                        "benchmark": name,
                        "question": q,
                        "expected": str(dt.get("expected", ""))[:50],
                        "predicted": str(dt.get("predicted", ""))[:50],
                        "error": dt.get("error", ""),
                    }
                    if dt.get("judge_score") is not None:
                        bc["judge_score"] = dt["judge_score"]
                    bad_cases.append(bc)
                    cls = dt.get("category", "其他") or "其他"
                    error_cats[cls] = error_cats.get(cls, 0) + 1

    # 计算最终得分
    benchmarks_summary = []
    chart_labels = []
    chart_scores = []
    for name in sorted(bench_results.keys()):
        scores = bench_results[name]
        avg_score = round(sum(scores) / len(scores), 1)
        # 当前次评测的 CI
        ci_data = None
        for r in model_runs[-1:]:
            for bid, bd in r.get("benchmarks", {}).items():
                if bid.upper() == name:
                    details = bd.get("details", [])
                    if details:
                        from eval_runner import bootstrap_ci
                        tuples = []
                        for dt in details:
                            q = dt.get("question", dt.get("description", ""))
                            exp = str(dt.get("expected", ""))
                            cor = dt.get("correct", False)
                            tuples.append((q, exp, cor))
                        ci_data = bootstrap_ci(tuples)
        benchmarks_summary.append({
            "name": name,
            "score": avg_score,
            "correct": bench_details[name]["correct"],
            "total": bench_details[name]["total"],
            "version": bench_details[name]["version"],
            "ci": ci_data,
        })
        chart_labels.append(name)
        chart_scores.append(avg_score)

    overall_correct = sum(b["correct"] for b in benchmarks_summary)
    overall_total = sum(b["total"] for b in benchmarks_summary)
    overall_score = round(overall_correct / overall_total * 100, 1) if overall_total else 0
    # 总体 Bootstrap 置信区间（取最近一次运行的综合 CI）
    overall_ci = None
    if model_runs:
        last_run = model_runs[-1]
        if last_run.get("confidence_interval"):
            overall_ci = last_run["confidence_interval"]

    report_date = model_runs[-1].get("completed_at", "")[:10] if model_runs[-1].get("completed_at") else "—"
    quick_mode_info = any(r.get("quick_mode") for r in model_runs)

    # ── 污染检测分析 ──────────────────────────────────────────────────────
    contamination_reports = {}
    if HAS_CONTAMINATION and benchmarks_summary:
        from pathlib import Path
        datasets_dir = Path(__file__).parent / "data" / "datasets"
        bench_id_map = {"MMLU": "mmlu", "GSM8K": "gsm8k", "HUMANEVAL": "humaneval",
                        "OPENTEVAL": "open_ended", "MEDQA": "medical_custom",
                        "SAFETY": "safety", "SAFEGUARD": "safety",
                        "OPENEVAL": "open_ended", "C-EVAL": "ceval_custom",
                        "AGENT": "agent_eval", "RAG": "rag_eval"}
        for bs in benchmarks_summary:
            name = bs["name"]
            bid = bench_id_map.get(name.upper(), name.lower())
            candidates = [
                datasets_dir / f"{bid}_ext.json",
                datasets_dir / f"{bid}_extended.json",
                datasets_dir / f"{bid}_sample.json",
                datasets_dir / f"{bid}_custom.json",
                datasets_dir / f"{bid}.json",
            ]
            for cpath in candidates:
                if cpath.exists():
                    try:
                        import json
                        with open(cpath) as f:
                            items = json.load(f)
                        cr = analyze_dataset(items)
                        # 预生成简短建议摘要，避免模板中用 chr(10)
                        if cr.get("suggestion"):
                            lines = cr["suggestion"].split("\n")
                            cr["short_suggestion"] = lines[0][:60] + "…" if len(lines[0]) > 60 else lines[0]
                        else:
                            cr["short_suggestion"] = ""
                        contamination_reports[name] = cr
                    except Exception:
                        pass
                    break

    # ── Bad Case 归因分析 ─────────────────────────────────────────────────
    attribution = {}
    if HAS_ERROR_CLASSIFIER and bad_cases:
        attribution = analyze_bad_cases(bad_cases)

    return render_template(
        "report.html",
        model_name=model_name,
        overall_score=overall_score,
        overall_correct=overall_correct,
        overall_total=overall_total,
        benchmarks_summary=benchmarks_summary,
        bad_cases=bad_cases,
        error_categories=error_cats,
        run_history=model_runs[::-1],
        chart_labels=chart_labels,
        chart_scores=chart_scores,
        overall_ci=overall_ci,
        quick_mode=quick_mode_info,
        report_date=report_date,
        runs_count=len(model_runs),
        benchmarks_count=len(benchmarks_summary),
        total_elapsed=total_elapsed,
        icons=ICONS,
        contamination_reports=contamination_reports,
        attribution=attribution,
        benchmarks_json=json.dumps(load_benchmarks()),
    )


# ---- Agent 安全风险报告 -------------------------------------------------


@app.route("/agent-safety-report/<model_name>")
@app.route("/agent-safety-report/<model_name>/<run_id>")
@login_required
def agent_safety_report(model_name: str, run_id: str = ""):
    """Agent 安全风险评测结果报告页"""
    u = session["user"]
    runs = list_user_runs(u)
    if run_id:
        model_runs = [r for r in runs if r.get("run_id") == run_id]
    else:
        model_runs = [r for r in runs if r.get("model_name") == model_name
                      and r.get("benchmarks", {}).get("agent_safety")
                      or r.get("benchmarks", {}).get("agent_safety_sample")]

    if not model_runs:
        return render_template("agent_safety_report.html",
                               model_name=model_name, overall_score=0,
                               total=0, safe_count=0, details=[],
                               category_summary={}, total_tokens=0,
                               avg_latency=0, report_date="—",
                               safe_rate=0)

    last_run = model_runs[-1]
    benchmarks = last_run.get("benchmarks", {})

    # 查找 agent_safety 结果
    safety_result = benchmarks.get("agent_safety") or benchmarks.get("agent_safety_sample") or {}
    details = safety_result.get("details", [])
    cat_summary = safety_result.get("category_summary", {})
    overall_score = safety_result.get("score", 0)
    total = safety_result.get("total", 0)
    correct = safety_result.get("correct", 0)
    total_tokens = safety_result.get("total_tokens", 0)
    avg_latency = safety_result.get("avg_latency", 0)
    report_date = last_run.get("completed_at", "")[:10] or "—"
    safe_rate = round(correct / total * 100, 1) if total else 0

    return render_template(
        "agent_safety_report.html",
        model_name=model_name,
        overall_score=overall_score,
        total=total,
        safe_count=correct,
        safe_rate=safe_rate,
        details=details,
        category_summary=cat_summary,
        total_tokens=total_tokens,
        avg_latency=avg_latency,
        report_date=report_date,
    )


# ---- 对比（支持统计显著性）---------------------------------------------------
@app.route("/compare")
@login_required
def compare_page():
    models = load_models(session["user"])
    benchmarks = load_benchmarks()
    runs = list_user_runs(session["user"])

    # 为有评测结果的模型补充分数数据和置信区间
    model_eval_scores = {}
    for r in runs:
        mn = r.get("model_name", "")
        if mn not in model_eval_scores or r.get("overall_score", 0) > model_eval_scores[mn].get("overall_score", 0):
            model_eval_scores[mn] = {
                "overall_score": r.get("overall_score", 0),
                "benchmarks": {k: v["score"] for k, v in r.get("benchmarks", {}).items()},
                "ci": r.get("confidence_interval"),
                "run_id": r.get("run_id"),
            }

    # 给模型添加兼容字段
    # 为模型分配不同颜色
    _COLOR_PALETTE = ['#6366f1', '#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#a855f7', '#ec4899']
    rich_models = []
    for idx, m in enumerate(models):
        scores_data = model_eval_scores.get(m["name"], {})
        ci = scores_data.get("ci")
        rich_models.append({
            # 排除敏感字段，显式构建
            "id": m.get("id", ""),
            "name": m.get("name", ""),
            "provider": m.get("provider", ""),
            "api_base": m.get("api_base", ""),
            "model_name": m.get("model_name", ""),
            "description": m.get("description", ""),
            "created_at": m.get("created_at", ""),
            "user": m.get("user", ""),
            "score": scores_data.get("overall_score", 0),
            "scores": scores_data.get("benchmarks", {}),
            "color": _COLOR_PALETTE[idx % len(_COLOR_PALETTE)],
            "cost_per_1k": "—",
            "latency_ms": "—",
            "context_window": 0,
            "version": m.get("created_at", ""),
            "ci_lower": ci["lower"] * 100 if ci else None,
            "ci_upper": ci["upper"] * 100 if ci else None,
            "run_id": scores_data.get("run_id"),
        })

    # 计算显著性差异 — 如果只有两个模型有评测数据
    significance = None
    rich_with_scores = [m for m in rich_models if m.get("ci_lower") is not None]
    if len(rich_with_scores) >= 2:
        m1, m2 = rich_with_scores[0], rich_with_scores[1]
        # 简单 CI 重叠判断
        ci1_low, ci1_high = m1["ci_lower"], m1["ci_upper"]
        ci2_low, ci2_high = m2["ci_lower"], m2["ci_upper"]
        overlap = not (ci1_high < ci2_low or ci2_high < ci1_low)
        significance = {
            "model_a": m1["name"],
            "model_b": m2["name"],
            "score_a": m1["score"],
            "score_b": m2["score"],
            "ci_overlap": overlap,
            "significant": not overlap,
        }

    return render_template(
        "compare.html",
        models=rich_models,
        benchmarks=benchmarks,
        models_json=json.dumps(rich_models),
        benchmarks_json=json.dumps(benchmarks),
        significance=significance,
    )


# ---- 数据集管理 -------------------------------------------------------------

ALLOWED_DATASET_FORMATS = {"json"}


def _safe_dataset_path(filename: str) -> Path | None:
    """验证并返回安全的数据集路径，防止路径穿越"""
    resolved = (DATASETS_DIR / filename).resolve()
    base = DATASETS_DIR.resolve()
    # 确保文件在 DATASETS_DIR 内
    if not str(resolved).startswith(str(base)):
        return None
    return resolved


def _infer_dataset_category_fast(path: Path) -> str:
    """快速推断数据集类别（只读取 JSON 的第一个元素，不加载全文件）"""
    try:
        with open(path, "rb") as f:
            header = f.read(4096)
        if not header.strip().startswith(b"["):
            return "未知"
        decoder = json.JSONDecoder()
        raw = header.decode("utf-8", errors="replace").lstrip().lstrip("[").lstrip()
        obj, _ = decoder.raw_decode(raw)
        if "choices" in obj and "answer" in obj:
            return "选择题 (MMLU 格式)"
        if "prompt" in obj and "test" in obj:
            return "编程题 (HumanEval 格式)"
        if "reference_answer" in obj:
            return "开放题 (OpenEval 格式)"
        if "answer" in obj:
            return "数学题 (GSM8K 格式)"
        return "通用"
    except Exception:
        return "无法解析"


def _infer_dataset_category(items: list) -> str:
    """根据数据内容推断类别"""
    if not items:
        return "未知"
    sample = items[0]
    if "choices" in sample and "answer" in sample:
        return "选择题 (MMLU 格式)"
    if "prompt" in sample and "test" in sample:
        return "编程题 (HumanEval 格式)"
    if "reference_answer" in sample:
        return "开放题 (OpenEval 格式)"
    if "answer" in sample:
        return "数学题 (GSM8K 格式)"
    return "通用"


@app.route("/datasets")
@login_required
def datasets_page():
    """数据集管理页（使用元数据缓存，避免每次全量加载大 JSON）"""
    datasets_dir = DATASETS_DIR
    cache_file = datasets_dir / ".datasets_cache.json"
    
    # 检查缓存有效性
    cache_valid = False
    cache = {}  # 预绑定，避免 Pyright 报错
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            cache_valid = all(
                datasets_dir.joinpath(fn).stat().st_mtime == meta["mtime"]
                for fn, meta in cache.get("files", {}).items()
                if datasets_dir.joinpath(fn).exists()
            )
        except Exception:
            cache_valid = False
    
    if cache_valid:
        files_info = cache["files_info"]
    else:
        # 重建缓存（只解析大 JSON 的头几个元素 + 计数）
        files_info = []
        for f in sorted(datasets_dir.glob("*.json")):
            if f.name.startswith("."):
                continue
            fsize = f.stat().st_size
            count_result = _count_json_items(f)
            if isinstance(count_result, int):
                questions = count_result
            else:
                questions = len(count_result)
            cat = _infer_dataset_category_fast(f)
            version = "扩展版" if ("_ext" in f.stem or "_extended" in f.stem) else "自定义" if "_custom" in f.stem else "基础版" if "_sample" in f.stem else "标准"
            files_info.append({
                "name": f.name,
                "stem": f.stem,
                "size": f"{fsize/1024:.1f} KB",
                "questions": questions,
                "category": cat,
                "version": version,
                "path": str(f),
            })
        # 持久化缓存到磁盘（服务器重启后依然有效）
        cache_data = {
            "files": {fi["name"]: {"mtime": datasets_dir.joinpath(fi["name"]).stat().st_mtime} for fi in files_info if fi["name"] != ".datasets_cache.json"},
            "files_info": files_info,
        }
        try:
            cache_file.write_text(json.dumps(cache_data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # 污染检测概览移至 API（不在页面加载时执行）
    contamination_data = {}

    return render_template("datasets.html", files=files_info, contamination=contamination_data)


@app.route("/api/datasets/contamination/<filename>")
@login_required
def api_dataset_contamination(filename: str):
    """分析指定数据集的数据泄露风险"""
    fpath = _safe_dataset_path(filename)
    if fpath is None:
        return jsonify({"error": "无效的文件名"}), 400
    if not HAS_CONTAMINATION:
        return jsonify({"error": "污染检测模块未加载"}), 400
    if not fpath.exists():
        return jsonify({"error": "文件不存在"}), 404
    try:
        with open(fpath, encoding="utf-8") as f:
            items = json.load(f)
        result = analyze_dataset(items) if isinstance(items, list) else {}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Rubric 管理 -----------------------------------------------------------

@app.route("/rubrics")
@login_required
def rubrics_page():
    """Rubric 评分模板管理页面"""
    templates = list_rubric_templates()
    return render_template("rubrics.html",
                           templates=templates,
                           has_rubric=HAS_RUBRIC)


@app.route("/api/rubrics")
@login_required
def api_rubrics():
    """返回 Rubric 模板列表"""
    return jsonify(list_rubric_templates())


@app.route("/datasets/upload", methods=["POST"])
@login_required
def datasets_upload():
    """上传数据集文件"""
    if "file" not in request.files:
        return jsonify({"error": "请选择文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400
    if not file.filename.endswith(".json"):
        return jsonify({"error": "仅支持 JSON 格式"}), 400

    name = request.form.get("name", "").strip()
    # 防止上传文件名中的路径穿越
    safe_name = name.replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_") if name else file.filename.replace(".json", "").replace("/", "_").replace("\\", "_")
    save_name = f"{safe_name}_custom.json"
    save_path = DATASETS_DIR / save_name

    # 检查文件内容
    try:
        content = file.read().decode("utf-8")
        items = json.loads(content)
    except Exception as e:
        return jsonify({"error": f"JSON 解析失败: {e}"}), 400

    if not isinstance(items, list) or len(items) == 0:
        return jsonify({"error": "数据集必须是非空的 JSON 数组"}), 400

    # 校验第一个条目
    sample = items[0]
    if not any(k in sample for k in ["id", "question", "prompt"]):
        return jsonify({"error": "数据缺少必要字段: id/question/prompt"}), 400

    # 保存
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    cat = _infer_dataset_category(items)
    return jsonify({
        "ok": True,
        "filename": save_name,
        "questions": len(items),
        "category": cat,
    })


@app.route("/datasets/delete/<filename>", methods=["POST"])
@login_required
def datasets_delete(filename: str):
    """删除数据集"""
    # 只允许删除 _custom 自定义文件
    if "_custom" not in filename:
        return jsonify({"error": "只能删除自定义数据集"}), 400
    fpath = _safe_dataset_path(filename)
    if fpath is None:
        return jsonify({"error": "无效的文件名"}), 400
    if fpath.exists():
        fpath.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "文件不存在"}), 404


@app.route("/datasets/preview/<filename>")
@login_required
def datasets_preview(filename: str):
    """预览数据集内容（返回前5条，避免传输 13MB 全量 JSON）"""
    fpath = _safe_dataset_path(filename)
    if fpath is None:
        return jsonify({"error": "无效的文件名"}), 400
    if not fpath.exists():
        return jsonify({"error": "文件不存在"}), 404
    try:
        fsize = fpath.stat().st_size
        with open(fpath, encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            return jsonify({"error": "数据集格式错误（需要 JSON 数组）"}), 400
        total = len(items)
        sample = items[:5]
        return jsonify({
            "total": total,
            "sample": sample,
            "size": f"{fsize/1024:.1f} KB",
        })
    except Exception as e:
        return jsonify({"error": f"解析失败: {str(e)}"}), 500


@app.route("/datasets/edit/<filename>", methods=["POST"])
@login_required
def datasets_edit(filename: str):
    """重命名自定义数据集"""
    if "_custom" not in filename:
        return jsonify({"error": "只能编辑自定义数据集"}), 400
    new_name = request.form.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "名称不能为空"}), 400
    safe_name = new_name.replace(" ", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
    save_name = f"{safe_name}_custom.json"
    if save_name == filename:
        return jsonify({"ok": True, "filename": save_name})
    src = _safe_dataset_path(filename)
    if src is None:
        return jsonify({"error": "无效的文件名"}), 400
    dst = DATASETS_DIR / save_name
    if dst.exists():
        return jsonify({"error": f"目标文件名 {save_name} 已存在"}), 400
    import shutil
    shutil.move(str(src), str(dst))
    return jsonify({"ok": True, "filename": save_name})


@app.route("/api/history/search")
@login_required
def history_search():
    """搜索评测历史"""
    u = session["user"]
    q = request.args.get("q", "").strip().lower()
    runs = list_user_runs(u)

    if not q:
        return jsonify({"runs": runs})

    results = []
    for r in runs:
        model = (r.get("model_name", "") or "").lower()
        note = (r.get("note", "") or "").lower()
        completed = (r.get("completed_at", "") or "").lower()
        run_id = (r.get("run_id", "") or "").lower()
        # 搜索 benchmark 名称和分数
        bench_text = " ".join(
            f"{bid} {bd.get('score', 0)}"
            for bid, bd in r.get("benchmarks", {}).items()
        ).lower()

        if q in model or q in note or q in completed or q in run_id or q in bench_text:
            results.append(r)

    return jsonify({"runs": results, "total": len(results)})


# ---------------------------------------------------------------------------
# 评测 Agent — 自然语言驱动
# ---------------------------------------------------------------------------

@app.route("/agents")
@login_required
def agents_page():
    """Agent 对话界面"""
    models = load_models(session["user"])
    judge_models = load_judge_models(session["user"])
    benchmarks = eval_agent.load_benchmarks()
    return render_template("agent.html", models=models, judge_models=judge_models,
                           benchmarks=benchmarks)


@app.route("/agents/parse", methods=["POST"])
@login_required
def agents_parse():
    """解析用户指令，返回执行计划（不执行）"""
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"error": "请输入评测指令"}), 400
    
    intent = eval_agent.parse_intent(text)
    plan_text = eval_agent.describe_plan(intent)
    
    if "error" in intent:
        return jsonify({"ok": False, "error": intent["error"], "plan": plan_text})
    
    return jsonify({
        "ok": True,
        "plan": plan_text,
        "intent": intent,
    })


@app.route("/agents/run", methods=["POST"])
@login_required
def agents_run():
    """确认并执行评测"""
    intent = request.json.get("intent", {})
    
    prep = eval_agent.prepare_eval(intent)
    if not prep["ok"]:
        return jsonify({"ok": False, "error": prep.get("error", "准备失败")}), 400

    run_id = start_eval(prep["model"], prep["benchmark_ids"], prep.get("judge_model"), user=session["user"])
    return jsonify({
        "ok": True,
        "run_id": run_id,
        "status_url": url_for("eval_status", run_id=run_id),
    })


# ---- 历史记录管理：删除 / 备注 / 清空 ---------------------------------------

def _rewrite_runs(predicate_fn):
    """通用的评测记录修改函数，predicate_fn(run) 返回 True 保留，False 删除"""
    runs_file = DATA_DIR / "eval_runs.json"
    if not runs_file.exists():
        return
    try:
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)
        new_runs = [r for r in runs if predicate_fn(r)]
        with open(runs_file, "w", encoding="utf-8") as f:
            json.dump(new_runs, f, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, IOError):
        pass


@app.route("/history/delete/<run_id>", methods=["POST"])
@login_required
def history_delete(run_id):
    u = session["user"]

    def keep(r):
        return not (r.get("run_id") == run_id and r.get("user") == u)

    _rewrite_runs(keep)
    from eval_runner import _completed_results as _cr
    _cr.pop(run_id, None)
    return jsonify({"ok": True})


@app.route("/history/clear", methods=["POST"])
@login_required
def history_clear():
    u = session["user"]

    def keep(r):
        return r.get("user") != u

    _rewrite_runs(keep)
    from eval_runner import _completed_results as _cr
    for k in list(_cr.keys()):
        _cr.pop(k, None)
    return jsonify({"ok": True})


@app.route("/history/note/<run_id>", methods=["POST"])
@login_required
def history_note(run_id):
    u = session["user"]
    note = request.json.get("note", "").strip()

    runs_file = DATA_DIR / "eval_runs.json"
    try:
        with open(runs_file, encoding="utf-8") as f:
            runs = json.load(f)
        for r in runs:
            if r.get("run_id") == run_id and r.get("user") == u:
                if note:
                    r["note"] = note
                else:
                    r.pop("note", None)
                break
        with open(runs_file, "w", encoding="utf-8") as f:
            json.dump(runs, f, ensure_ascii=False, indent=2)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 56)
    print("  🚀  ByteBrain AI 评测平台")
    print("  ────────────────────────────────")
    print(f"  地址:  http://127.0.0.1:5001")
    print(f"  退出:  按 Ctrl+C 停止服务器")
    print("=" * 56)
    print()
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
