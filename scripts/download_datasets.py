"""
从 HuggingFace 下载真实 Benchmark 数据集并转换为平台格式。
使用 requests 直接下载原始 JSON 文件（无需 datasets 库）。
"""
import json
import os
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── HuggingFace 数据集原始文件 URL ──────────────────────────────────────────

def download_file(url: str, local_path: Path) -> bool:
    """下载文件，跳过已存在的"""
    if local_path.exists() and local_path.stat().st_size > 100:
        print(f"  ⏭️  已存在: {local_path.name}")
        return True
    print(f"  ⬇️  下载: {local_path.name}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


# ── MMLU ────────────────────────────────────────────────────────────────────

def download_mmlu():
    """MMLU: 57个学科，每个学科 ~100道测试题"""
    out = DATA_DIR / "mmlu_extended.json"
    all_questions = []
    
    # MMLU 有 57 个学科，先下载部分学科作为扩展
    subjects = [
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
    
    for subject in subjects:
        url = f"https://raw.githubusercontent.com/hendrycks/test/master/{subject}_test.json"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            items = resp.json()
            for item in items:
                all_questions.append({
                    "id": f"mmlu_{subject}_{len(all_questions)}",
                    "category": subject.replace("_", " ").title(),
                    "question": item.get("question", ""),
                    "choices": item.get("choices", {}),
                    "answer": item.get("answer", ""),
                })
            print(f"  ✅ {subject}: {len(items)} 题")
        except Exception as e:
            print(f"  ⚠️  {subject}: 跳过 ({e})")
    
    # 如果线上没拉到，用本地增强版
    if len(all_questions) < 50:
        print("  ⚠️  线上下载不足，生成本地扩展数据...")
        all_questions = _generate_mmlu_extended()
    
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)
    print(f"\n  📊 MMLU 合计: {len(all_questions)} 题 → {out.name}")
    return len(all_questions)


def _generate_mmlu_extended() -> list:
    """如果线上拉不到，生成扩展样本"""
    questions = []
    subjects_extended = {
        "法律": [
            ("在刑法中，正当防卫超过必要限度造成重大损害的，应当负刑事责任，但是应当（  ）处罚。",
             {"A": "从轻", "B": "减轻或者免除", "C": "从重", "D": "加重"}, "B"),
            ("我国宪法规定，中华人民共和国的一切权力属于（  ）。",
             {"A": "人民", "B": "中国共产党", "C": "全国人大", "D": "国务院"}, "A"),
            ("民法调整的是（  ）之间的财产关系和人身关系。",
             {"A": "平等主体", "B": "上下级", "C": "管理与被管理", "D": "刑事犯罪"}, "A"),
        ],
        "物理": [
            ("光在真空中的传播速度约为（  ）。", {"A": "3×10⁶ m/s", "B": "3×10⁷ m/s", "C": "3×10⁸ m/s", "D": "3×10⁹ m/s"}, "C"),
            ("牛顿第一定律也称为（  ）。", {"A": "惯性定律", "B": "加速度定律", "C": "作用与反作用定律", "D": "万有引力定律"}, "A"),
            ("下列哪个是矢量？", {"A": "温度", "B": "质量", "C": "速度", "D": "能量"}, "C"),
            ("电功率的国际单位是（  ）。", {"A": "伏特", "B": "安培", "C": "瓦特", "D": "欧姆"}, "C"),
            ("声音在下列哪种介质中传播最快？", {"A": "空气", "B": "水", "C": "铁", "D": "真空"}, "C"),
            ("光的折射定律由谁提出？", {"A": "牛顿", "B": "斯涅尔", "C": "斐索", "D": "麦克斯韦"}, "B"),
            ("功的计算公式是（  ）。", {"A": "W = F·s", "B": "W = F/s", "C": "W = m·v", "D": "W = m·g·h"}, "A"),
            ("下列哪种现象属于光的干涉？", {"A": "彩虹", "B": "肥皂泡上的彩色条纹", "C": "海市蜃楼", "D": "影子"}, "B"),
        ],
        "化学": [
            ("以下哪种元素的化学符号是 'Fe'？", {"A": "铁", "B": "氟", "C": "氦", "D": "汞"}, "A"),
            ("水的化学式是（  ）。", {"A": "H₂O", "B": "CO₂", "C": "NaCl", "D": "CH₄"}, "A"),
            ("下列哪个是强酸？", {"A": "碳酸", "B": "盐酸", "C": "醋酸", "D": "硼酸"}, "B"),
            ("元素周期表中，同周期元素从左到右，金属性如何变化？",
             {"A": "增强", "B": "减弱", "C": "不变", "D": "先增后减"}, "B"),
            ("下列哪个不是有机化合物？", {"A": "甲烷", "B": "乙醇", "C": "氯化钠", "D": "乙酸"}, "C"),
            ("pH=7 表示溶液呈（  ）。", {"A": "酸性", "B": "碱性", "C": "中性", "D": "不确定"}, "C"),
        ],
        "生物": [
            ("人体内氧气运输主要依靠（  ）。", {"A": "白细胞", "B": "血小板", "C": "红细胞", "D": "血浆"}, "C"),
            ("光合作用的主要场所是（  ）。", {"A": "线粒体", "B": "叶绿体", "C": "细胞核", "D": "核糖体"}, "B"),
            ("人体最大的器官是（  ）。", {"A": "心脏", "B": "肝脏", "C": "皮肤", "D": "大脑"}, "C"),
            ("DNA 的双螺旋结构由谁发现？", {"A": "达尔文", "B": "孟德尔", "C": "沃森和克里克", "D": "巴斯德"}, "C"),
            ("人体有多少对染色体？", {"A": "22", "B": "23", "C": "24", "D": "46"}, "B"),
            ("下列哪个是分解者？", {"A": "小草", "B": "兔子", "C": "蘑菇", "D": "老鹰"}, "C"),
        ],
        "计算机科学": [
            ("下列哪种数据结构遵循 '后进先出' (LIFO) 原则？", {"A": "队列", "B": "栈", "C": "链表", "D": "数组"}, "B"),
            ("时间复杂度 O(n log n) 的排序算法是（  ）。", {"A": "冒泡排序", "B": "选择排序", "C": "归并排序", "D": "插入排序"}, "C"),
            ("HTTP 状态码 404 表示（  ）。", {"A": "成功", "B": "重定向", "C": "未找到", "D": "服务器错误"}, "C"),
            ("SQL 中用于查询数据的关键字是（  ）。", {"A": "INSERT", "B": "UPDATE", "C": "SELECT", "D": "DELETE"}, "C"),
            ("以下哪个是面向对象编程的特性？", {"A": "封装", "B": "编译", "C": "链接", "D": "调试"}, "A"),
        ],
        "数学": [
            ("sin²(θ) + cos²(θ) = （  ）。", {"A": "0", "B": "1", "C": "2", "D": "不确定"}, "B"),
            ("微分方程 dy/dx = 2x 的通解是（  ）。", {"A": "y = x² + C", "B": "y = 2x + C", "C": "y = x²/2 + C", "D": "y = ln x + C"}, "A"),
            ("矩阵 A 的逆矩阵存在的充要条件是（  ）。",
             {"A": "det(A) = 0", "B": "det(A) ≠ 0", "C": "A 是方阵", "D": "A 是对称阵"}, "B"),
            ("自然常数 e 的数值约等于（  ）。", {"A": "2.14", "B": "2.72", "C": "3.14", "D": "1.62"}, "B"),
        ],
        "历史": [
            ("第一次世界大战爆发的导火索是什么事件？", {"A": "萨拉热窝事件", "B": "珍珠港事件", "C": "慕尼黑协定", "D": "十月革命"}, "A"),
            ("中华人民共和国成立于哪一年？", {"A": "1945", "B": "1949", "C": "1950", "D": "1954"}, "B"),
            ("文艺复兴起源于哪个国家？", {"A": "英国", "B": "法国", "C": "意大利", "D": "西班牙"}, "C"),
        ],
        "地理": [
            ("世界上面积最大的国家是（  ）。", {"A": "中国", "B": "美国", "C": "俄罗斯", "D": "加拿大"}, "C"),
            ("长江是中国最长的河流，它的长度约为（  ）。", {"A": "3000 km", "B": "5000 km", "C": "6300 km", "D": "8000 km"}, "C"),
            ("地球上最长的纬线是（  ）。", {"A": "赤道", "B": "北回归线", "C": "南回归线", "D": "北极圈"}, "A"),
            ("时区是根据什么划分的？", {"A": "纬线", "B": "经线", "C": "海陆分布", "D": "人口密度"}, "B"),
        ],
        "经济": [
            ("通货膨胀最直接的衡量指标是（  ）。", {"A": "GDP", "B": "CPI", "C": "PPI", "D": "PMI"}, "B"),
            ("供给和需求决定（  ）。", {"A": "成本", "B": "价格", "C": "利润", "D": "税收"}, "B"),
            ("GDP 的全称是（  ）。", {"A": "国民生产总值", "B": "国内生产总值", "C": "国民收入", "D": "人均收入"}, "B"),
        ],
        "逻辑": [
            ("如果所有 A 都是 B，且所有 B 都是 C，那么以下哪个结论必然成立？",
             {"A": "所有 C 都是 A", "B": "所有 A 都是 C", "C": "有些 C 不是 A", "D": "没有 A 是 C"}, "B"),
            ("命题'如果下雨，地会湿'的逆否命题是（  ）。",
             {"A": "如果地湿，则下雨", "B": "如果没下雨，地不会湿",
              "C": "如果地没湿，则没下雨", "D": "下雨且地没湿"}, "C"),
        ],
        "天文": [
            ("太阳系中距离太阳最近的行星是（  ）。", {"A": "金星", "B": "水星", "C": "火星", "D": "地球"}, "B"),
            ("地球的卫星是（  ）。", {"A": "火星", "B": "月球", "C": "太阳", "D": "金星"}, "B"),
            ("光年是什么单位？", {"A": "时间", "B": "速度", "C": "距离", "D": "亮度"}, "C"),
        ],
        "医学": [
            ("人体最大的器官是（  ）。", {"A": "心脏", "B": "肝脏", "C": "皮肤", "D": "大脑"}, "C"),
            ("血压测量中，收缩压指（  ）。", {"A": "心脏舒张时", "B": "心脏收缩时", "C": "平均动脉压", "D": "静脉压"}, "B"),
            ("胰岛素用于治疗（  ）。", {"A": "高血压", "B": "糖尿病", "C": "癌症", "D": "感冒"}, "B"),
        ],
    }
    
    idx = 0
    for category, items in subjects_extended.items():
        for q, choices, answer in items:
            questions.append({
                "id": f"mmlu_ext_{idx:04d}",
                "category": category,
                "question": q,
                "choices": choices,
                "answer": answer,
            })
            idx += 1
    return questions


# ── GSM8K ────────────────────────────────────────────────────────────────────

def download_gsm8k():
    """GSM8K: 小学数学应用题的测试集"""
    out = DATA_DIR / "gsm8k_extended.json"
    
    # 尝试从 HuggingFace raw 下载
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    questions = []
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n"):
                if line:
                    item = json.loads(line)
                    questions.append({
                        "id": f"gsm8k_ext_{len(questions)}",
                        "category": _classify_gsm8k(item["question"]),
                        "question": item["question"],
                        "answer": _extract_gsm8k_answer(item["answer"]),
                    })
            print(f"  ✅ 下载成功: {len(questions)} 题")
    except Exception as e:
        print(f"  ⚠️  线上下载失败: {e}")
    
    if len(questions) < 30:
        print("  ⚠️  生成本地扩展...")
        questions = _generate_gsm8k_extended()
    
    with open(out, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"  📊 GSM8K 合计: {len(questions)} 题 → {out.name}")
    return len(questions)


def _classify_gsm8k(question: str) -> str:
    """粗略分类"""
    if any(k in question for k in ["速度", "公里", "小时", "距离", "相遇"]):
        return "速度/行程"
    if any(k in question for k in ["百分", "%", "打折", "利率", "比例"]):
        return "百分比/比例"
    if any(k in question for k in ["面", "周长", "三角", "圆", "几何"]):
        return "几何"
    if any(k in question for k in ["概率", "可能"]):
        return "概率"
    if any(k in question for k in ["分", "除", "每"]):
        return "分数/除法"
    return "综合"


def _extract_gsm8k_answer(text: str) -> str:
    """从 '#### 数字' 格式中提取答案"""
    import re
    m = re.search(r"####\s*(-?\d+(?:[,.]\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(-?\d+(?:[,.]\d+)?)\s*$", text.strip())
    if m:
        return m.group(1).replace(",", "")
    return text.strip()[:50]


def _generate_gsm8k_extended() -> list:
    """扩展的 GSM8K 样本"""
    problems = [
        ("小明有 15 个苹果，他给了小红 5 个，然后又买了 8 个。请问小明现在有多少个苹果？", "18", "加减法"),
        ("一个长方形的长是 12 厘米，宽是 8 厘米。它的周长是多少厘米？", "40", "几何"),
        ("一台电脑原价 8000 元，打折后降价 15%。打折后的价格是多少元？", "6800", "百分比"),
        ("一辆车以每小时 60 公里的速度行驶了 2.5 小时。它走了多少公里？", "150", "速度"),
        ("一个班级有 40 名学生，其中 60% 是女生。女生有多少人？", "24", "百分比"),
        ("甲、乙两人分别从 A、B 两地同时出发，相向而行。甲的速度是 5 km/h，乙的速度是 7 km/h，两地相距 36 km。几小时后两人相遇？", "3", "速度"),
        ("一本书 300 页，小明第一天读了全书的 1/3，第二天读了剩下的 1/4。第三天从第几页开始读？", "151", "分数"),
        ("商店进了 200 件 T 恤，以每件 80 元的价格出售。如果全部卖出，总收入是多少？", "16000", "乘法"),
        ("一个水池装有进水管和出水管，单独开进水管 4 小时注满，单独开出水管 6 小时排空。同时打开，几小时注满？", "12", "工作效率"),
        ("小张买了 3 支笔和 2 个本子，共花了 22 元。如果每支笔 4 元，每个本子多少元？", "5", "方程"),
        ("圆的半径为 7 cm，它的面积是多少？（π≈3.14）", "153.86", "几何"),
        ("一件商品先降价 20%，再涨价 20%，最终价格是原价的百分之几？", "96", "百分比"),
        ("甲乙两人共有 120 元，甲的钱是乙的 3 倍。乙有多少元？", "30", "方程"),
        ("一个等差数列的首项为 3，公差为 5，第 10 项是多少？", "48", "数列"),
        ("一项工程，甲队单独做 10 天完成，乙队单独做 15 天完成。两队合作几天完成？", "6", "工作效率"),
        ("从 1 到 100 中，既是 2 的倍数又是 3 的倍数的数有多少个？", "16", "集合"),
        ("一个正方体的表面积是 150 平方厘米，它的一个面的面积是多少？", "25", "几何"),
        ("某商品进价 50 元，售价 75 元，利润率是多少？", "50", "百分比"),
        ("小李考试，语文 85 分，数学 90 分，英语 95 分，三科平均分是多少？", "90", "综合"),
        ("一个圆形花坛的直径为 10 米，围花坛走一圈需要走多少米？（π≈3.14）", "31.4", "几何"),
        ("100 元分给 5 个人，每人分到的钱数各不相同且都是整数。分得最少的人最多拿多少元？", "18", "综合"),
        ("一位工人每小时生产 12 个零件，每天工作 8 小时，5 天可以生产多少个零件？", "480", "乘法"),
    ]
    return [
        {"id": f"gsm8k_ext_{i:04d}", "category": cat, "question": q, "answer": a}
        for i, (q, a, cat) in enumerate(problems)
    ]


# ── HumanEval ───────────────────────────────────────────────────────────────

def download_humaneval():
    """HumanEval: Python 代码生成"""
    out = DATA_DIR / "humaneval_extended.json"
    questions = []
    
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            import gzip
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as f:
                for line in f:
                    item = json.loads(line)
                    questions.append({
                        "id": f"humaneval_{item['task_id']}",
                        "prompt": item["prompt"],
                        "description": item["entry_point"],
                        "test": item["test"],
                        "category": _classify_humaneval(item["entry_point"], item["prompt"]),
                    })
            print(f"  ✅ 下载成功: {len(questions)} 题")
    except Exception as e:
        print(f"  ⚠️  线上下载失败: {e}")
    
    if len(questions) < 5:
        print("  ⚠️  生成本地扩展...")
        questions = _generate_humaneval_extended()
    
    with open(out, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"  📊 HumanEval 合计: {len(questions)} 题 → {out.name}")
    return len(questions)


def _classify_humaneval(entry_point: str, prompt: str) -> str:
    if "sort" in entry_point or "max" in entry_point or "min" in entry_point:
        return "基础操作"
    if "sum" in entry_point or "add" in entry_point or "calc" in entry_point:
        return "数学计算"
    if "search" in entry_point or "find" in entry_point or "two_sum" in entry_point:
        return "数组/查找"
    if "palindrome" in entry_point or "string" in entry_point or "str" in entry_point:
        return "字符串"
    if "tree" in entry_point or "node" in entry_point or "list" in entry_point:
        return "数据结构"
    return "其他"


def _generate_humaneval_extended() -> list:
    """HumanEval 扩展样本"""
    problems = [
        {
            "id": "humaneval_ext_00",
            "prompt": "def two_sum(nums, target):\n    \"\"\"返回数组中两个数和为 target 的下标。\"\"\"\n    ",
            "description": "两数之和",
            "test": "assert two_sum([2,7,11,15], 9) == [0,1]\nassert two_sum([3,2,4], 6) == [1,2]",
            "category": "数组/哈希",
        },
        {
            "id": "humaneval_ext_01",
            "prompt": "def is_palindrome(s):\n    \"\"\"判断字符串是否为回文（忽略大小写和非字母数字字符）。\"\"\"\n    ",
            "description": "回文判断",
            "test": "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False",
            "category": "字符串",
        },
        {
            "id": "humaneval_ext_02",
            "prompt": "def fibonacci(n):\n    \"\"\"返回斐波那契数列的第 n 项（n 从 0 开始）。\"\"\"\n    ",
            "description": "斐波那契数列",
            "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55",
            "category": "递归/DP",
        },
        {
            "id": "humaneval_ext_03",
            "prompt": "def find_max(nums):\n    \"\"\"返回列表中的最大值。\"\"\"\n    ",
            "description": "找最大值",
            "test": "assert find_max([3, 7, 2, 9, 5]) == 9\nassert find_max([-1, -5, -3]) == -1",
            "category": "基础",
        },
        {
            "id": "humaneval_ext_04",
            "prompt": "def is_prime(n):\n    \"\"\"判断 n 是否为质数。\"\"\"\n    ",
            "description": "质数判断",
            "test": "assert is_prime(7) == True\nassert is_prime(10) == False\nassert is_prime(2) == True",
            "category": "数学",
        },
        {
            "id": "humaneval_ext_05",
            "prompt": "def factorial(n):\n    \"\"\"计算 n 的阶乘。\"\"\"\n    ",
            "description": "阶乘",
            "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(3) == 6",
            "category": "数学",
        },
        {
            "id": "humaneval_ext_06",
            "prompt": "def bubble_sort(arr):\n    \"\"\"对数组进行冒泡排序（原地排序）。\"\"\"\n    ",
            "description": "冒泡排序",
            "test": "assert bubble_sort([3,1,4,1,5]) == [1,1,3,4,5]\nassert bubble_sort([]) == []\nassert bubble_sort([1]) == [1]",
            "category": "排序",
        },
        {
            "id": "humaneval_ext_07",
            "prompt": "def count_words(text):\n    \"\"\"统计字符串中的单词数（按空格分割）。\"\"\"\n    ",
            "description": "单词计数",
            "test": "assert count_words('hello world') == 2\nassert count_words('') == 0\nassert count_words('a b c d') == 4",
            "category": "字符串",
        },
        {
            "id": "humaneval_ext_08",
            "prompt": "def binary_search(arr, target):\n    \"\"\"在有序数组中二分查找目标值，返回下标，未找到返回 -1。\"\"\"\n    ",
            "description": "二分查找",
            "test": "assert binary_search([1,2,3,4,5], 3) == 2\nassert binary_search([1,2,3,4,5], 6) == -1\nassert binary_search([], 1) == -1",
            "category": "查找",
        },
        {
            "id": "humaneval_ext_09",
            "prompt": "def remove_duplicates(nums):\n    \"\"\"去除列表中的重复元素，返回新列表保持原顺序。\"\"\"\n    ",
            "description": "去重",
            "test": "assert remove_duplicates([1,2,2,3,3,3]) == [1,2,3]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1,1,1]) == [1]",
            "category": "基础",
        },
    ]
    return problems


# ── 主流程 ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  📥 下载并扩展 Benchmark 数据集")
    print("=" * 50)
    print()
    
    total = 0
    
    print("\n1️⃣  MMLU")
    total += download_mmlu()
    
    print("\n2️⃣  GSM8K")
    total += download_gsm8k()
    
    print("\n3️⃣  HumanEval")
    total += download_humaneval()
    
    print("\n" + "=" * 50)
    print(f"  ✅ 完成！共 {total} 道题目")
    print("=" * 50)


if __name__ == "__main__":
    main()
