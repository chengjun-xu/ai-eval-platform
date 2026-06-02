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

## 🚀 快速开始

> **只需 3 步，5 分钟内开始评测你的第一个模型。**

### 环境要求

- **Python 3.10+**（推荐 3.11）
- **pip**（Python 包管理器）
- 一台能联网的电脑（macOS / Linux / Windows 均可）

### 第 1 步：克隆项目

```bash
git clone https://github.com/chengjun-xu/ai-eval-platform.git
cd ai-eval-platform
```

> 💡 **没有 git？** 也可以直接去 GitHub 页面点「Code → Download ZIP」，解压后进入目录。

### 第 2 步：安装依赖

```bash
# 方法一：使用 requirements.txt（推荐）
pip install -r requirements.txt

# 方法二：直接指定
pip install flask requests
```

> 📦 全部依赖仅 **2 个包**（flask + requests），其他全是 Python 标准库，无需复杂环境配置。

#### （可选）启用 OpenCompass 后端

```bash
pip install opencompass
```

> 不安装也完全不影响使用，平台会自动检测并优雅降级。

### 第 3 步：启动服务

```bash
# 推荐：设置 session 密钥（不设置也能用，但重启后需要重新登录）
export EVAL_PLATFORM_SECRET="your-random-secret-here"

# 启动
python3 app.py
```

看到以下输出说明启动成功：

```
✅ 模型管理: models.json 已加载 (N 个模型)
✅ 数据集注册: MMLU, GSM8K, HumanEval, C-Eval ...
💡 EVAL_PLATFORM_SECRET 环境变量未设置（使用自动生成的 session 密钥）
 * Running on http://127.0.0.1:5001
```

> 🔗 浏览器访问 **http://localhost:5001**

### 第 4 步：开始使用

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 点击 **「立即注册」** | 创建你的账户 |
| 2 | 登录后进入 **「模型管理」** | 添加你的 LLM（支持 OpenAI 兼容 API） |
| 3 | 进入 **「Benchmark 评测」** | 选择一个模型 + 数据集 → 点击评测 |
| 4 | 查看 **「评测报告」** | 自动生成得分、Bad Case、弱项分析 |

> 🎬 **第一次评测建议：** 先选 MMLU sample（15 题，几秒出结果），体验完整流程后再跑大数据集。

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
│   ├── models.json               # 模型注册信息（已 gitignore）
│   ├── judge_models.json         # Judge 模型信息
│   ├── eval_runs.json            # 评测历史记录
│   └── datasets/                 # 评测数据集（随仓库一起下载）
│       ├── mmlu_sample.json / mmlu_official.json       # MMLU
│       ├── gsm8k_sample.json / gsm8k_official.json     # GSM8K
│       ├── humaneval_sample.json / humaneval_official.json  # HumanEval
│       ├── ceval_custom.json            # C-Eval 中文综合
│       ├── safety_custom.json           # SafeGuard 安全合规
│       ├── agent_safety_sample.json     # Agent 安全风险评测
│       ├── agent_eval_custom.json       # Agent 多步任务
│       ├── rag_eval_custom.json         # RAG 证据忠实性
│       └── ...                     # 更多数据集详见下方支持列表
│
├── templates/                    # Jinja2 模板（18+ 个页面）
│   ├── base.html                 # 布局骨架
│   ├── dashboard.html            # 仪表盘
│   ├── models.html / judge_models.html  # 模型管理
│   ├── benchmarks.html           # 评测执行
│   ├── eval_status.html / multi_eval_status.html  # 评测状态
│   ├── report.html               # 评测报告
│   ├── compare.html              # 模型对比
│   ├── history.html              # 历史记录
│   ├── data_production.html      # 自动数据生产
│   ├── report_agent_safety.html  # Agent 安全评测报告
│   └── ...                       # 更多分析页面
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

## 🧪 支持的数据集

### 官方标准化数据集

| **数据集** | **格式** | **题量** | **能力维度** |
|-----------|----------|---------|-------------|
| MMLU (官方) | 选择题 | ~14,000 | 57 学科知识理解 |
| GSM8K (官方) | 数学推理 | ~1,300 | 小学数学应用题 |
| HumanEval (官方) | 代码生成 | 164 | Python 函数补全 |

> ⚠️ 官方数据集（`*_official.json`）体积较大，需要通过 `scripts/download_official_datasets.py` 下载。仓库默认只包含 sample 和 ext（扩展版）数据集，clone 后即可直接使用。

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
| | Agent 安全风险评测 | 18 | 指令注入/工具滥用/安全边界/逻辑一致性 |
| **中文** | C-Eval | 44 | 13 学科中文题 |
| **Agent** | Agent 多步任务 | 6 | 多步任务规划 |
| **RAG** | RAG 证据忠实性 | 8 | 证据忠实性评测 |

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

// Agent 安全评测
{"id": "s1", "category": "指令注入", "scenario": "...",
 "prompt": "...", "attack_vector": "...",
 "risk_level": "高", "expected_behavior": "..."}

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

## 🛡️ 安全说明

| **措施** | **说明** |
|----------|----------|
| 密码哈希 | `werkzeug.security.generate_password_hash`（bcrypt） |
| Session 密钥 | 通过 `EVAL_PLATFORM_SECRET` 环境变量配置，未设置则自动生成随机密钥 |
| API Key 保护 | 模型 API key 仅后端使用，**不会发送到前端**（`data/models.json` 已 gitignore） |
| 用户隔离 | 模型/评测数据按用户分区，删除操作校验所有权 |
| 数据集隔离 | 运行时数据文件 (`data/*.json`) 已 .gitignore，不会误提交 |

### Session 密钥配置

启动时如看到以下警告：

```
UserWarning: ⚠ EVAL_PLATFORM_SECRET 环境变量未设置
```

说明 session 密钥未配置，重启后所有用户会话会失效。建议设置：

```bash
# 生成随机密钥
python3 -c "import secrets; print(secrets.token_hex(32))"

# 设置环境变量（将输出的密钥替换进去）
export EVAL_PLATFORM_SECRET="生成的64位十六进制字符串"

# 启动服务
python3 app.py
```

> 💡 也可以将 `export` 命令写入 `~/.bashrc` 或 `~/.zshrc`，免去每次手动设置。

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

## ❓ 常见问题

### Q: 启动报错 `ModuleNotFoundError: No module named 'flask'`

```bash
# 安装依赖
pip install -r requirements.txt
```

> 建议先确认 Python 版本：`python3 --version`（需要 3.10+）

### Q: 页面能打开，但评测一直显示"等待中"？

1. 检查模型 API key 是否正确配置（模型管理页面）
2. 检查网络是否能访问 API 服务
3. 查看终端输出，看是否有错误日志

### Q: 跑官方数据集（MMLU/GSM8K 完整版）找不到题目？

```bash
# 需要先下载官方数据集
python3 scripts/download_official_datasets.py
```

> 仓库只包含 sample 和 ext 版本，可直接使用。官方数据集体积较大，需要单独下载。

### Q: 重启后需要重新登录？

设置 `EVAL_PLATFORM_SECRET` 环境变量即可解决：

```bash
export EVAL_PLATFORM_SECRET="your-secret-key"
python3 app.py
```

### Q: 支持哪些模型 API？

所有兼容 **OpenAI API 格式** 的服务均可：
- OpenAI（GPT-4o / o1 / o3-mini）
- DeepSeek（deepseek-chat / deepseek-reasoner）
- Anthropic（通过 OpenAI 兼容代理）
- 本地部署的 vLLM / Ollama（需要暴露 OpenAI 兼容接口）

### Q: 端口 5001 被占用了？

```bash
# 修改 app.py 最后一行的端口号
# 或直接启动时指定
python3 app.py --port=5002
```

### Q: 如何下载官方数据集？

```bash
python3 scripts/download_official_datasets.py
```

该脚本会从 HuggingFace 下载 MMLU / GSM8K / HumanEval 官方完整数据集并转换为平台所需的 JSON 格式。

### Q: 如何贡献自己的数据集？

1. 按照上方「自定义数据集格式」准备 JSON 文件
2. 放入 `data/datasets/` 目录
3. 重启平台即可在 Benchmark 列表中看到
4. 欢迎提 PR 贡献到仓库！

---

## 📄 License

MIT License — 随意使用、修改、分发。
