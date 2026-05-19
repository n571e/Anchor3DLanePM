#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/ztc2025/anaconda3/etc/profile.d/conda.sh
conda activate anchor3dlane-cu121

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/gen-efficientnet-pytorch"

SEED="${SEED:-3407}"
PARALLEL="${PARALLEL:-1}"
GPU_LIST="${GPU_LIST:-0 1 2}"

run_train_eval() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local work_dir="$4"
  local checkpoint="$5"
  local eval_dir="$6"

  mkdir -p "$work_dir" "$eval_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[$(date '+%F %T')] START train ${name} on gpu ${gpu}"
    python -u tools/train.py "$config" \
      --work-dir "$work_dir" \
      --no-validate \
      --gpu-id 0 \
      --seed "$SEED" 2>&1 | tee "$work_dir/launcher.log"

    echo "[$(date '+%F %T')] START eval ${name} on gpu ${gpu}"
    python -u tools/test.py "$config" "$work_dir/$checkpoint" \
      --show-dir "$eval_dir" 2>&1 | tee "$eval_dir/launcher.log"
    echo "[$(date '+%F %T')] DONE ${name}"
  )
}

declare -a NAMES=(
  "bundle_warm_10k_feature_bias_prioronly_lowloss_codefix"
  "bundle_warm_10k_feature_bias_iter0_lowloss_codefix"
  "bundle_warm_10k_feature_bias_prior_iter0_lowloss_codefix"
)

declare -a CONFIGS=(
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_feature_bias_prioronly_lowloss_codefix_bs8.py"
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_feature_bias_iter0_lowloss_codefix_bs8.py"
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_feature_bias_prior_iter0_lowloss_codefix_bs8.py"
)

declare -a WORK_DIRS=(
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_prioronly_lowloss_codefix_bs8"
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_iter0_lowloss_codefix_bs8"
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_prior_iter0_lowloss_codefix_bs8"
)

declare -a EVAL_DIRS=(
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_prioronly_lowloss_codefix_bs8_iter10000"
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_iter0_lowloss_codefix_bs8_iter10000"
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_feature_bias_prior_iter0_lowloss_codefix_bs8_iter10000"
)

read -r -a GPUS <<< "$GPU_LIST"

if [[ "$PARALLEL" == "1" ]]; then
  declare -a PIDS=()
  for idx in "${!NAMES[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    run_train_eval "$gpu" "${NAMES[$idx]}" "${CONFIGS[$idx]}" "${WORK_DIRS[$idx]}" "iter_10000.pth" "${EVAL_DIRS[$idx]}" &
    PIDS+=("$!")
  done

  failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  exit "$failed"
fi

for idx in "${!NAMES[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_train_eval "$gpu" "${NAMES[$idx]}" "${CONFIGS[$idx]}" "${WORK_DIRS[$idx]}" "iter_10000.pth" "${EVAL_DIRS[$idx]}"
done
