# ByteBrain — AI LLM Evaluation Platform

> 一个功能完备的大模型评测平台，支持多维度、多场景的模型评估。

## ✨ 核心功能

### 📊 10 类 Benchmark
| Benchmark | 类型 | 能力维度 |
|-----------|------|---------|
| **MMLU** | 选择题 | 知识理解（57学科） |
| **GSM8K** | 数学推理 | 数学能力 |
| **HumanEval** | 代码生成 | 编程能力 |
| **OpenEval** | 开放题 | 综合能力 + LLM-as-Judge |
| **MedQA** 🏥 | 医疗选择题 | 医疗知识 |
| **SafeGuard** 🛡️ | 安全合规 | 12 个医疗安全场景 + Rubric 评分 |
| **C-Eval** 🇨🇳 | 中文多学科 | 13 个学科 44 道中文题 |
| **Rubric** 📋 | 结构化评分 | 5 个 Rubric 模板 × 4 维度加权评分 |
| **Agent** 🤖 | 多步任务规划 | 步骤完整性/工具合理性/逻辑可行性 |
| **RAG** 🔗 | 证据忠实性 | 忠实性/完整性/幻觉检测 |

### 🚩 数据泄露检测
- n-gram 重叠分析（3/5/9/13-gram）
- 4 级风险预警（安全/低/中/高/严重）
- 自污染检测（数据集内部相似度评估）
- 已知污染源数据库

### 🔍 Bad Case 归因分类
自动将错误答案归因到 7 类失败模式，并给出训练建议：

| 分类 | 改进建议 |
|------|---------|
| 📋 指令遵循失败 | SFT 数据增强 |
| 📚 知识缺失/遗忘 | 领域 SFT + RAG |
| 🔗 逻辑断裂 | CoT 训练数据 |
| 👻 幻觉模式 | DPO + 事实性奖励模型 |
| 🚫 拒答策略不当 | 校准安全对齐边界 |
| 🔍 证据不一致 | RAG 约束数据 |
| 🔧 工具调用失败 | Agent 训练数据 |

### 📋 Rubric 结构化评分
- 5 个预定义模板：通用、医疗、代码、翻译、安全
- 每个模板 4 个评分维度 × 1-5 分标准
- 加权总分 + 维度雷达图
- 可自定义模板

### 📈 统计方法
- Bootstrap 置信区间（1000 次重采样）
- 快速回归模式（分层抽样 20 题）
- 模型对比 + 统计显著性检验

---

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Flask

### 安装与运行

```bash
# 克隆仓库
git clone https://github.com/chengjun-xu/ai-eval-platform.git
cd ai-eval-platform

# 安装依赖
pip install flask requests

# 启动服务
python3 app.py

# 打开浏览器访问
# http://localhost:5001
```

### 启动后

打开浏览器访问 **http://localhost:5001**

1. 点击 **「立即注册」** 创建你的账号
2. 登录后即可使用全部功能

---

## 🏗️ 平台架构

```
ai-eval-platform/
├── app.py                  # Flask 主应用（路由、鉴权、页面）
├── eval_runner.py           # 评测执行引擎（后台线程）
├── eval_agent.py            # 自然语言驱动的自动评测 Agent
├── contamination_detector.py # 数据泄露检测模块
├── error_classifier.py      # Bad Case 归因分类模块
├── rubric_manager.py        # Rubric 结构化评分管理器
├── data/
│   ├── models.json          # 模型注册信息
│   ├── judge_models.json    # Judge 模型信息
│   ├── eval_runs.json       # 评测历史记录
│   └── datasets/            # 评测数据集
│       ├── mmlu_sample.json
│       ├── gsm8k_sample.json
│       ├── humaneval_sample.json
│       ├── open_ended_sample.json
│       ├── medical_custom.json     # MedQA
│       ├── safety_custom.json      # SafeGuard
│       ├── ceval_custom.json       # C-Eval
│       ├── rubric_open_custom.json # Rubric
│       ├── agent_eval_custom.json  # Agent
│       └── rag_eval_custom.json    # RAG
├── templates/               # Jinja2 模板
└── static/                  # CSS/JS 静态资源
```

### 评测流程
```
用户选择模型 + Benchmark → 后台线程执行 → 调用 LLM API → 评分（规则/Judge）
→ Bootstrap CI → Bad Case归因 → 污染检测 → 报告生成
```

---

## 🔧 添加自定义数据集

1. 进入「数据集管理」页面
2. 上传 JSON 文件（支持 MMLU/GSM8K/HumanEval/OpenEval 格式）
3. 系统自动识别数据格式并注册
4. 在「运行评测」页面即可选择新数据集

### 数据格式
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
```

---

## 📄 License

MIT
