#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/ztc2025/anaconda3/etc/profile.d/conda.sh
conda activate anchor3dlane-cu121

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/gen-efficientnet-pytorch"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

eval_ckpt() {
  local name="$1"
  local config="$2"
  local work_dir="$3"
  local iter="$4"
  local eval_dir="output/eval_ablation/openlane/${name}_iter${iter}"

  if [[ -f "${eval_dir}/evaluation_result.json" ]]; then
    echo "[$(date '+%F %T')] SKIP existing ${name} iter_${iter}"
    return
  fi
  mkdir -p "$eval_dir"
  echo "[$(date '+%F %T')] START eval ${name} iter_${iter}"
  python -u tools/test.py "$config" "${work_dir}/iter_${iter}.pth" \
    --show-dir "$eval_dir" 2>&1 | tee "$eval_dir/launcher.log"
  echo "[$(date '+%F %T')] DONE eval ${name} iter_${iter}"
}

for iter in 2500 5000 7500; do
  eval_ckpt \
    "anchor3dlanepp_baseline_warm_r18_10k_bs8" \
    "../configs_v2/openlane/anchor3dlanepp_ablation_baseline_warm_r18_10k_bs8.py" \
    "output/ablation/openlane/anchor3dlanepp_baseline_warm_r18_10k_bs8" \
    "$iter"

  eval_ckpt \
    "bundlelane_frame_anchor_warm_r18_10k_bs8" \
    "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_bs8.py" \
    "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_bs8" \
    "$iter"

  eval_ckpt \
    "bundlelane_frame_anchor_warm_r18_10k_lowloss_bs8" \
    "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_lowloss_bs8.py" \
    "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_lowloss_bs8" \
    "$iter"

  eval_ckpt \
    "bundlelane_frame_anchor_warm_r18_10k_detach_no_xinject_bs8" \
    "../configs_v2/openlane/bundlelane_ablation_frame_anchor_warm_r18_10k_detach_no_xinject_bs8.py" \
    "output/ablation/openlane/bundlelane_frame_anchor_warm_r18_10k_detach_no_xinject_bs8" \
    "$iter"
done
