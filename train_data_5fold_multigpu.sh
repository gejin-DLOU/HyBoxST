#!/bin/bash
set -euo pipefail

# =========================
# Basic settings
# =========================
DATA_ROOT="./hest1k_datasets"
DATASET="kidney"      # kidney / colorectum / skin / lung
MODEL_NAME="hyboxst" # hyboxst / hyboxst_base
ABLATION_TAG="0"

# HVG: selected_hvg_gene_list.txt
# HMHVG: selected_gene_list.txt
GENE_LIST_FILENAME="selected_gene_list.txt"
GENE_TAG="HMHVG"

SPLIT_ROOT="./split/${DATASET}/flod_5"
EXP_PREFIX="${DATASET}_${GENE_TAG}_${ABLATION_TAG}" 

# =========================
# GPU settings
# =========================
# 三卡版本：使用物理 GPU 1,2,3
# 如果只想用两张卡，改成：GPUS=(1 2)
# 如果只想用一张卡，改成：GPUS=(1)
GPUS=(2 3)
NUM_GPUS=${#GPUS[@]}

# =========================
# Training settings
# =========================
BATCH_SIZE=128
EPOCHS=100
NUM_WORKERS=4
PATIENCE=10
LR=1e-4
LAST_LAYER=11


# =========================
# Summary settings
# =========================
SUMMARY_SCRIPT="./summarize_5fold_results.py"
SUMMARY_DIR="./experiments/summary/${DATASET}_${GENE_TAG}_${ABLATION_TAG}_5fold"
RESULT_SEARCH_DIRS=("./experiments")

# =========================
# Check HF token
# =========================
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN is not set."
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    exit 1
fi

export HF_HUB_DISABLE_XET=1

# =========================
# Check split files
# =========================
for FOLD in 0 1 2 3 4
do
    SPLIT_FILE="${SPLIT_ROOT}/sample_split_flod_${FOLD}.json"
    if [ ! -f "${SPLIT_FILE}" ]; then
        echo "Error: split file not found: ${SPLIT_FILE}"
        exit 1
    fi
done

# =========================
# Function: train one fold
# =========================
run_fold() {
    local FOLD=$1
    local PHYSICAL_GPU=$2

    local SPLIT_FILE="${SPLIT_ROOT}/sample_split_flod_${FOLD}.json"
    local EXP_NAME="${EXP_PREFIX}_fold${FOLD}"

    echo "========================================"
    echo "Start training ${DATASET} ${GENE_TAG} ${ABLATION_TAG} fold ${FOLD} on GPU ${PHYSICAL_GPU}"
    echo "Split file: ${SPLIT_FILE}"
    echo "Experiment: ${EXP_NAME}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU} python ./main.py \
        --data_root ${DATA_ROOT} \
        --dataset ${DATASET} \
        --gpu 0 \
        --model_name ${MODEL_NAME} \
        --gene_list_filename ${GENE_LIST_FILENAME} \
        --split_dir ${SPLIT_FILE} \
        --experiment_name ${EXP_NAME} \
        --huggingface_token "${HF_TOKEN}" \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --num_workers ${NUM_WORKERS} \
        --patience ${PATIENCE} \
        --lr ${LR} \
        --fold ${FOLD} \
        --last_layer ${LAST_LAYER} \
        --pathway_resource_dir "./pathway_resources/${DATASET}"

    echo "Fold ${FOLD} finished on GPU ${PHYSICAL_GPU}."
}
#--pathway_resource_dir "./pathway_resources/shuffle/${DATASET}_gene_label_shuffle_seed42" \
# --disable_calibration_auto_disable 
# =========================
# Train 5 folds with GPU queue
# =========================
PIDS=()
RUNNING_FOLDS=()

for FOLD in 0 1 2 3 4
do
    GPU_INDEX=$((FOLD % NUM_GPUS))
    PHYSICAL_GPU=${GPUS[$GPU_INDEX]}

    run_fold ${FOLD} ${PHYSICAL_GPU} &
    PIDS+=($!)
    RUNNING_FOLDS+=(${FOLD})

    # 每启动 NUM_GPUS 个任务，等待这一批完成
    if [ ${#PIDS[@]} -eq ${NUM_GPUS} ]; then
        echo "Waiting for current batch: folds ${RUNNING_FOLDS[*]}"
        for PID in "${PIDS[@]}"
        do
            wait ${PID}
        done
        PIDS=()
        RUNNING_FOLDS=()
    fi
done

# 等待最后不足 NUM_GPUS 的剩余任务
if [ ${#PIDS[@]} -gt 0 ]; then
    echo "Waiting for final batch: folds ${RUNNING_FOLDS[*]}"
    for PID in "${PIDS[@]}"
    do
        wait ${PID}
    done
fi

echo "All 5 folds finished."

# =========================
# Summarize 5-fold metrics
# =========================
if [ ! -f "${SUMMARY_SCRIPT}" ]; then
    echo "Error: summary script not found: ${SUMMARY_SCRIPT}"
    echo "Please put summarize_5fold_results.py in the HyBoxST root directory."
    exit 1
fi

python "${SUMMARY_SCRIPT}" \
    --exp_prefix "${EXP_PREFIX}" \
    --folds 0 1 2 3 4 \
    --search_dirs "${RESULT_SEARCH_DIRS[@]}" \
    --result_glob "final_test_result*.json" \
    --output_dir "${SUMMARY_DIR}"
