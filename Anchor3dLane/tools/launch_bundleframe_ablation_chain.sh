#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/ztc2025/anaconda3/etc/profile.d/conda.sh
conda activate anchor3dlane-cu121

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/gen-efficientnet-pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"

run_train_eval() {
  local name="$1"
  local config="$2"
  local work_dir="$3"
  local checkpoint="$4"
  local eval_dir="$5"

  mkdir -p "$work_dir" "$eval_dir"
  echo "[$(date '+%F %T')] START train ${name}"
  python -u tools/train.py "$config" \
    --work-dir "$work_dir" \
    --no-validate \
    --gpu-id "$GPU_ID" \
    --seed "$SEED" 2>&1 | tee "$work_dir/launcher.log"

  echo "[$(date '+%F %T')] START eval ${name}"
  python -u tools/test.py "$config" "$work_dir/$checkpoint" \
    --show-dir "$eval_dir" 2>&1 | tee "$eval_dir/launcher.log"
  echo "[$(date '+%F %T')] DONE ${name}"
}

run_train_eval \
  "baseline_warm_30k_resume" \
  "../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_30k_resume_bs8.py" \
  "output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_30k_resume_bs8" \
  "iter_30000.pth" \
  "output/eval_ablation/openlane/anchor3dlanepp_baseline_warm_r18_30k_resume_bs8_iter30000"

run_train_eval \
  "bundle_warm_10k_noinject" \
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_noinject_bs8.py" \
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_noinject_bs8" \
  "iter_10000.pth" \
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_noinject_bs8_iter10000"

run_train_eval \
  "bundle_warm_10k_lowloss" \
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_bs8.py" \
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_lowloss_bs8" \
  "iter_10000.pth" \
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_lowloss_bs8_iter10000"

run_train_eval \
  "bundle_warm_10k_inject1" \
  "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_inject1_bs8.py" \
  "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_inject1_bs8" \
  "iter_10000.pth" \
  "output/eval_ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_inject1_bs8_iter10000"
