# ByteBrain — AI LLM Evaluation Platform

> 一个功能完备的大模型评测平台，支持多维度、多场景的模型评估、自动化数据生产、弱项挖掘与回归分析。

---

## ✨ 核心功能一览

### 📊 评测能力

| **能力** | **说明** |
|----------|----------|
| **15+ 类 Benchmark** | 知识理解、数学推理、代码生成、医疗专业、安全合规、Agent 多步任务、RAG 忠实性、多轮对话等 |
| **3 种评测后端** | 本地 API 评测 → HuggingFace 数据集 → OpenCompass（可选） |
| **LLM-as-Judge** | 开放题自动评分，支持自定义 Judge 模型 |
| **Bootstrap 置信区间** | 1000 次重采样，评估结果可靠性 |
| **快速回归模式** | 分层抽样 20 题，秒级出分 |
| **统计显著性检验** | 模型对比 + CI 重叠判断 |

### 🔬 高级分析

| **模块** | **功能** |
|----------|----------|
| **数据泄露检测** | n-gram 重叠分析（3/5/9/13-gram）、4 级风险预警、自污染检测 |
| **Bad Case 归因分类** | 7 类失败模式自动归类 + 训练改进建议 |
| **弱项挖掘** | 按 Benchmark/维度/类别分析模型短板 |
| **回归检测** | 历史分数对比，自动标记显著退化 |
| **Judge 一致性分析** | 交叉验证多个 Judge 模型的评分稳定性 |

### 🛠️ 数据生产管线

| **功能** | **说明** |
|----------|----------|
| **Self-Instruct** | 从种子指令通过 LLM 扩写生成新评测数据 |
| **Evol-Instruct** | 对现有指令进化增强（加约束、深化、多步推理） |
| **合成 RL 数据** | 生成偏好对用于 RLHF/DPO 训练数据 |
| **长尾场景生成** | negation/numerical/ambiguous 等 8 种策略 |
| **数据挖掘** | 从文本/文件/HuggingFace 数据集挖掘评测题目 |
| **Rubric 结构化评分** | 5 个预定义模板 × 4 维度加权评分 |

---

## 🗺️ 平台架构

```
ai-eval-platform/
├── app.py                        # Flask 主应用（路由、鉴权、页面）
├── eval_runner.py                # 评测执行引擎（后台线程）
├── eval_agent.py                 # 自然语言驱动的自动评测 Agent
│
├── 📦 数据生产
│   ├── data_production.py        # Self-Instruct / Evol-Instruct / RL 数据生成
│   ├── data_mining_pipeline.py   # 从文本/文件/HF 数据挖掘评测题目
│   ├── longtail_generator.py     # 长尾场景生成器（8 种策略）
│   └── scripts/
│       ├── download_official_datasets.py   # 下载 MMLU/GSM8K/HumanEval 官方集
│       ├── mine_medical_dataset.py         # 医疗数据挖掘
│       └── update_medical_dataset.sh       # 医疗数据集更新脚本
│
├── 📊 评测分析
│   ├── benchmark_registry.py     # HuggingFace Benchmark 注册表
│   ├── opencompass_adapter.py    # OpenCompass 可选后端适配器
│   ├── eval_analysis.py          # 回归检测 + Judge 评分一致性
│   ├── weakness_miner.py         # 模型弱项挖掘
│   ├── contamination_detector.py # 数据泄露检测
│   ├── error_classifier.py       # Bad Case 归因分类
│   ├── rubric_manager.py         # Rubric 结构化评分管理器
│   └── metrics.py                # BLEU / ROUGE-L / F1 / 余弦相似度
│
├── 📁 data/
│   ├── models.json               # 模型注册信息（含 API key，已 gitignore）
│   ├── judge_models.json         # Judge 模型信息
│   ├── eval_runs.json            # 评测历史记录
│   └── datasets/                 # 评测数据集
│       ├── mmlu_sample.json / mmlu_official.json       # MMLU 知识理解
│       ├── gsm8k_sample.json / gsm8k_official.json     # GSM8K 数学推理
│       ├── humaneval_sample.json / humaneval_official.json  # HumanEval 代码
│       ├── open_ended_sample.json / mt_bench_sample.json   # 开放题/多轮对话
│       ├── medical_custom.json / med_*_official.json       # 医疗评测
│       ├── safety_custom.json           # SafeGuard 安全合规
│       ├── ceval_custom.json            # C-Eval 中文综合
│       ├── rubric_open_custom.json      # Rubric 结构化评分
│       ├── agent_eval_custom.json       # Agent 多步任务
│       └── rag_eval_custom.json         # RAG 证据忠实性
│
├── templates/                    # Jinja2 模板（18 个页面）
│   ├── base.html                 # 布局骨架
│   ├── dashboard.html            # 仪表盘
│   ├── models.html / judge_models.html  # 模型管理
│   ├── benchmarks.html           # 评测执行
│   ├── eval_status.html / multi_eval_status.html  # 评测状态
│   ├── report.html               # 评测报告
│   ├── compare.html              # 模型对比
│   ├── history.html              # 历史记录
│   ├── data_mining.html          # 数据挖掘
│   ├── data_production.html      # 自动数据生产
│   ├── longtail_gen.html         # 长尾场景生成
│   ├── weakness.html             # 弱项分析
│   ├── consistency.html          # Judge 一致性
│   ├── regression.html           # 回归检测
│   └── rubrics.html              # Rubric 管理
│
└── static/                       # CSS/JS 静态资源
    ├── css/style.css
    └── js/main.js
```

### 评测流程

```
注册模型 → 选择 Benchmark → 执行评测
                                    ↓
后台线程: 调用 LLM API → 规则/Judge 评分 → Bootstrap CI
                                    ↓
            Bad Case 归因 → 污染检测 → 弱项分析 → 报告生成
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- `pip install flask requests`

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/chengjun-xu/ai-eval-platform.git
cd ai-eval-platform

# 安装依赖
pip install flask requests

# （可选）OpenCompass 后端
# pip install opencompass

# 设置 session 密钥（推荐）
export EVAL_PLATFORM_SECRET="your-random-secret-here"

# 启动服务
python3 app.py

# 浏览器访问
# http://localhost:5001
```

### 启动后

1. 点击 **「立即注册」** 创建账号
2. 登录后即可使用全部功能
3. 在「模型管理」页面注册你的 LLM（支持 OpenAI 兼容 API）
4. 在「Benchmark 评测」页面选择模型和数据集执行评测

---

## 🛡️ 安全说明

| **措施** | **说明** |
|----------|----------|
| 密码哈希 | `werkzeug.security.generate_password_hash`（bcrypt） |
| Session 密钥 | 通过 `EVAL_PLATFORM_SECRET` 环境变量配置，未设置则自动生成随机密钥 |
| API Key 保护 | 模型 API key 仅后端使用，**不会发送到前端**（`data/models.json` 已 gitignore） |
| 用户隔离 | 模型/评测数据按用户分区，删除操作校验所有权 |
| 数据集隔离 | 运行时数据文件 (`data/*.json`) 已 .gitignore，不会误提交 |

### 启动时如看到警告

```
UserWarning: ⚠ EVAL_PLATFORM_SECRET 环境变量未设置
```

说明 session 密钥未配置，重启后所有用户会话会失效。建议设置：

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
export EVAL_PLATFORM_SECRET="生成的64位十六进制字符串"
```

---

## 🧪 支持的数据集

### 官方标准化数据集

| **数据集** | **格式** | **题量** | **能力维度** |
|-----------|----------|---------|-------------|
| MMLU (官方) | 选择题 | ~14,000 | 57 学科知识理解 |
| GSM8K (官方) | 数学推理 | ~1,300 | 小学数学应用题 |
| HumanEval (官方) | 代码生成 | 164 | Python 函数补全 |

### 扩展与自定义数据集

| **类别** | **数据集** | **题量** | **说明** |
|----------|-----------|---------|----------|
| **知识理解** | MMLU sample / ext | 15 / 50 | 轻量版 + 扩展版 |
| **数学推理** | GSM8K sample / ext | 8 / 22 | 分层抽样 + 扩展 |
| **代码能力** | HumanEval sample / ext | 5 / 10 | 基础版 + 扩展 |
| **综合能力** | OpenEval | 10 | LLM-as-Judge 开放题 |
| | MT-Bench sample | 8×2 轮 | 多轮对话评测 |
| **医疗专业** | MedQA | 50 | 医疗选择题 |
| | Med-R1 (官方) | 198 | 医疗开放题（含推理链） |
| | Med 长尾 (官方) | 7 类 | 罕见病/多病共存/伦理困境 |
| | Med 临床 (官方) | 10 | 高难度临床选择题 |
| **安全合规** | SafeGuard | 12 | 医疗安全 Rubric 评分 |
| **中文** | C-Eval | 44 | 13 学科中文题 |
| **Agent** | Agent | 6 | 多步任务规划 |
| **RAG** | RAG | 8 | 证据忠实性评测 |

### 自定义数据集格式

```json
// 选择题 (MMLU 格式)
{"id": "q1", "category": "医学", "question": "...",
 "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer": "A"}

// 数学题 (GSM8K 格式)
{"id": "q1", "question": "...", "answer": "42"}

// 编程题 (HumanEval 格式)
{"id": "q1", "prompt": "def foo():", "test": "assert foo() == 42",
 "description": "..."}

// 开放题 (OpenEval 格式)
{"id": "q1", "question": "...", "reference_answer": "..."}

// 多轮对话 (MT-Bench 格式)
{"id": "01", "category": "写作",
 "conversations": [
   {"role": "user", "content": "..."},
   {"role": "assistant", "content": "..."},
   {"role": "user", "content": "..."}
 ]}
```

上传到「数据集管理」页面即可自动识别注册。

---

## 📚 模块详解

### 自动化数据生产 (`data_production.py`)

三种生产方式，可用于构建你自己的评测数据集：

| **方式** | **输入** | **产出** | **应用场景** |
|----------|---------|----------|-------------|
| Self-Instruct | 种子指令（文本或 benchmark） | 新评测题 | 从零创建数据集 |
| Evol-Instruct | 现有 benchmark | 增强版题目 | 提升题目难度/多样性 |
| RL 偏好对 | 种子指令 | 正反例对 | DPO/RLHF 训练数据 |

### 长尾场景生成 (`longtail_generator.py`)

8 种场景策略：

```
negation         → 否定式改写（"不需要 vs 需要"）
numerical        → 数值置换
ambiguous        → 模糊条件
multi_step       → 多步推理
boundary         → 边界条件
temporal         → 时间偏移
adversarial      → 对抗性改写
domain_shift     → 领域迁移
```

### 弱项挖掘 (`weakness_miner.py`)

按 Benchmark、类别、难度层级、错误类型分析模型短板，支持多模型横向对比。

### 评测分析 (`eval_analysis.py`)

| **功能** | **说明** |
|----------|----------|
| 回归检测 | 同模型 vs 历史，Bonferroni 校正 |
| Judge 一致性 | Kappa 系数 + 分数分布对比 |
| 分数分布 | 柱状图 + 描述统计 |

---

## 📄 License

MIT
