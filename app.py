"""
AI 大模型评测平台 - Flask Application
=====================================
支持模型注册、评测执行、结果展示的一体化 LLM 评测平台。
"""
import json
import os
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from eval_runner import start_eval, get_run_status, list_completed_runs, list_user_runs
import eval_agent
import uuid
import shutil

# ── 新增模块：污染检测 + 错误归因 ────────────────────────────────────────────
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
app.secret_key = os.urandom(32).hex()

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
        return [m for m in all_models if m.get("user", "") == user]
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
        return [m for m in all_models if m.get("user", "") == user]
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
    """从数据集目录动态扫描可用的 Benchmark（支持 _sample 和 _extended）"""
    benchmarks = []
    mapping = {
        "mmlu_sample":     {"name": "MMLU",     "full_name": "Massive Multitask Language Understanding",
                            "category": "知识理解", "icon": "book",
                            "desc": "覆盖多学科的多项选择题评测，测试模型的知识广度与推理能力。"},
        "mmlu_extended":   {"name": "MMLU",     "full_name": "MMLU (扩展版)",
                            "category": "知识理解", "icon": "book",
                            "desc": "12个学科共50题，更全面的知识覆盖评测。"},
        "gsm8k_sample":    {"name": "GSM8K",    "full_name": "Grade School Math 8K",
                            "category": "数学推理", "icon": "calculator",
                            "desc": "小学数学应用题评测，测试模型的数学推理与计算能力。"},
        "gsm8k_extended":  {"name": "GSM8K",    "full_name": "GSM8K (扩展版)",
                            "category": "数学推理", "icon": "calculator",
                            "desc": "22道数学应用题，覆盖加减、几何、百分比、方程等题型。"},
        "humaneval_sample":{"name": "HumanEval","full_name": "HumanEval",
                            "category": "代码能力", "icon": "code",
                            "desc": "Python 代码生成评测，测试模型根据 docstring 编写函数的能力。"},
        "humaneval_extended":{"name": "HumanEval","full_name": "HumanEval (扩展版)",
                              "category": "代码能力", "icon": "code",
                              "desc": "10道 Python 编程题，覆盖数组、字符串、排序、查找等基础算法。"},
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
    }
    for fname in sorted(DATASETS_DIR.glob("*_extended.json")) + sorted(DATASETS_DIR.glob("*_sample.json")) + sorted(DATASETS_DIR.glob("*_custom.json")):
        # _extended → _sample → _custom 优先级
        key = fname.stem
        info = mapping.get(key, {"name": key.replace("_custom","").replace("_sample","").replace("_extended","").title(), "category": "自定义"})
        with open(fname, encoding="utf-8") as f:
            items = json.load(f)
        # 判断版本
        if "_extended" in key:
            version = "扩展版"
        elif "_custom" in key:
            version = "自定义"
        else:
            version = "基础版"
        info_name = info.get("name", key.replace("_custom","").replace("_sample","").replace("_extended",""))
        benchmarks.append({
            "id": key.replace("_sample", "").replace("_extended", "").replace("_custom", "") + ("_ext" if "_extended" in key else "") + ("_custom" if "_custom" in key else ""),
            "name": info_name,
            "version": version,
            "full_name": info.get("full_name", info_name),
            "category": info.get("category", "自定义"),
            "icon": info.get("icon", "file"),
            "description": info.get("desc", f"自定义数据集，共 {len(items)} 题"),
            "question_count": len(items),
        })
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
    )


# ---- 模型管理 --------------------------------------------------------------
@app.route("/models")
@login_required
def models_page():
    models = load_models(session["user"])
    benchmarks = load_benchmarks()
    return render_template(
        "models.html",
        models=models,
        benchmarks=benchmarks,
        models_json=json.dumps(models),
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
        "api_key": data.get("api_key", "").strip(),
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
    delete_model(model_id)
    return redirect(url_for("models_page"))


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
        "api_key": data.get("api_key", "").strip(),
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
    delete_judge_model(model_id)
    return redirect(url_for("judge_models_page"))


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
    model_id = request.form.get("model_id", "").strip()
    benchmark_ids = request.form.getlist("benchmarks")
    judge_model_id = request.form.get("judge_model_id", "").strip()
    quick_mode = request.form.get("quick_mode", "") == "on"

    if not model_id or not benchmark_ids:
        return jsonify({"error": "请选择模型和至少一个 Benchmark"}), 400

    models = load_models(u)
    model = next((m for m in models if m["id"] == model_id), None)
    if not model:
        return jsonify({"error": "模型不存在"}), 404

    judge_model = None
    if judge_model_id:
        judge_models = load_judge_models(u)
        judge_model = next((jm for jm in judge_models if jm["id"] == judge_model_id), None)

    run_id = start_eval(model, benchmark_ids, judge_model, quick_mode=quick_mode, user=session["user"])
    return redirect(url_for("eval_status", run_id=run_id))


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

ICONS = {"MMLU": "🧠", "GSM8K": "🔢", "HumanEval": "💻", "OpenEval": "📝", "SAFETY": "🛡️", "SAFEGUARD": "🛡️", "MEDQA": "🏥", "C-EVAL": "🇨🇳", "AGENT": "🤖", "RAG": "🔗"}

@app.route("/report/<model_name>")
@login_required
def report(model_name: str):
    runs = list_user_runs(session["user"])
    # 过滤出该模型的所有运行
    model_runs = [r for r in runs if r.get("model_name", "") == model_name]
    if not model_runs:
        return render_template("report.html", model_name=model_name, overall_score=0,
                               overall_correct=0, overall_total=0, benchmarks_summary=[],
                               bad_cases=[], run_history=[], chart_labels=[], chart_scores=[],
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
    rich_models = []
    for m in models:
        scores_data = model_eval_scores.get(m["name"], {})
        ci = scores_data.get("ci")
        rich_models.append({
            **m,
            "score": scores_data.get("overall_score", 0),
            "scores": scores_data.get("benchmarks", {}),
            "color": "#6366f1",
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
    """数据集管理页"""
    benchmarks = load_benchmarks()
    datasets_dir = DATASETS_DIR
    files_info = []
    for f in sorted(datasets_dir.glob("*.json")):
        size = f.stat().st_size
        with open(f, encoding="utf-8") as fh:
            try:
                items = json.load(fh)
                count = len(items) if isinstance(items, list) else 1
            except Exception:
                count = 0
        cat = _infer_dataset_category(items) if count > 0 else "无法解析"
        version = "扩展版" if "_extended" in f.stem else "自定义" if "_custom" in f.stem else "基础版" if "_sample" in f.stem else "标准"
        files_info.append({
            "name": f.name,
            "stem": f.stem,
            "size": f"{size/1024:.1f} KB",
            "questions": count,
            "category": cat,
            "version": version,
            "path": str(f),
        })

    # 污染检测概览
    contamination_data = {}
    if HAS_CONTAMINATION:
        for fi in files_info:
            if fi["questions"] > 0:
                fpath = datasets_dir / fi["name"]
                try:
                    with open(fpath) as f:
                        items = json.load(f)
                    if isinstance(items, list) and len(items) > 0:
                        cr = analyze_dataset(items)
                        contamination_data[fi["name"]] = {
                            "risk": cr.get("overall_risk", "未知"),
                            "score": cr.get("overall_contamination_score", 0),
                            "high_risk": cr.get("high_risk_count", 0),
                        }
                except Exception:
                    pass

    return render_template("datasets.html", files=files_info, contamination=contamination_data)


@app.route("/api/datasets/contamination/<filename>")
@login_required
def api_dataset_contamination(filename: str):
    """分析指定数据集的数据泄露风险"""
    if not HAS_CONTAMINATION:
        return jsonify({"error": "污染检测模块未加载"}), 400
    fpath = DATASETS_DIR / filename
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
    safe_name = name.replace(" ", "_").replace(".", "_") if name else file.filename.replace(".json", "")
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
    fpath = DATASETS_DIR / filename
    if fpath.exists():
        fpath.unlink()
        return jsonify({"ok": True})
    return jsonify({"error": "文件不存在"}), 404


@app.route("/datasets/preview/<filename>")
@login_required
def datasets_preview(filename: str):
    """预览数据集内容"""
    fpath = DATASETS_DIR / filename
    if not fpath.exists():
        return jsonify({"error": "文件不存在"}), 404
    with open(fpath, encoding="utf-8") as f:
        items = json.load(f)
    return jsonify({
        "total": len(items),
        "sample": items[:5] if isinstance(items, list) else items,
    })


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
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
