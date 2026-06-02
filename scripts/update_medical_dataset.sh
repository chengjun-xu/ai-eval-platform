#!/usr/bin/env bash
# 医疗评测数据集更新脚本
# 用法: bash update_medical_dataset.sh [--force]
# 从HF镜像更新 Medical-R1 数据集，合并到现有评测集

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATASETS_DIR="$PROJECT_DIR/data/datasets"
LOG_FILE="$PROJECT_DIR/data/dataset_update.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始更新医疗评测数据集..." | tee -a "$LOG_FILE"

# 1. 运行数据挖掘脚本
echo "→ 运行数据挖掘..." | tee -a "$LOG_FILE"
cd "$PROJECT_DIR"
python3 scripts/mine_medical_dataset.py 2>&1 | tee -a "$LOG_FILE"

# 2. 检查更新后的文件
echo "" | tee -a "$LOG_FILE"
echo "→ 当前数据集文件:" | tee -a "$LOG_FILE"
ls -lh "$DATASETS_DIR"/med_*.json "$DATASETS_DIR"/longtail_*.json 2>/dev/null | tee -a "$LOG_FILE"

# 3. 统计信息
echo "" | tee -a "$LOG_FILE"
for f in "$DATASETS_DIR"/med_*.json "$DATASETS_DIR"/longtail_*.json; do
    if [ -f "$f" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))")
        echo "   $(basename $f): $count 题" | tee -a "$LOG_FILE"
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 更新完成" | tee -a "$LOG_FILE"
