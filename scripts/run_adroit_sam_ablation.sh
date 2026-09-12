#!/usr/bin/env bash
# =============================================================================
# run_adroit_sam_ablation.sh  — SAM Ablation Study on CalQL (pen-binary-v0)
# Chạy từ đầu (không load checkpoint), 5 runs tuần tự
#
# Run mapping:
#   Run 0  [Baseline]       CalQL gốc, không SAM
#   Run 1  [NaiveSAM]       SAM toàn bộ offline, rho cố định 0.05
#   Run 2  [LateStage]      SAM chỉ 20% cuối offline, rho cố định 0.05
#   Run 3  [AdaptiveRho]    SAM toàn bộ offline, cosine_decay rho
#   Run 4  [Combined]       SAM 20% cuối + cosine_decay rho (best proposal)
# =============================================================================

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tri/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

ENV=pen-binary-v0
OUTPUT_DIR=./saved_models
PROJECT=Cal-QL-SAM-Ablation

# Hyperparams khớp với run_adroit.sh gốc
COMMON_ARGS="
  --env=$ENV
  --save_model=False
  --logging.output_dir=$OUTPUT_DIR
  --logging.online
  --logging.project=$PROJECT
  --seed=0
  --cql_min_q_weight=1.0
  --policy_arch=512-512
  --qf_arch=512-512-512
  --offline_eval_every_n_epoch=2
  --online_eval_every_n_env_steps=1000
  --eval_n_trajs=20
  --n_train_step_per_epoch_offline=1000
  --n_pretrain_epochs=20
  --max_online_env_steps=3e4
  --mixing_ratio=0.5
  --reward_scale=10.0
  --reward_bias=5.0
  --enable_calql=True
"

run_experiment() {
    local RUN_ID=$1
    local RUN_LABEL=$2
    shift 2
    local EXTRA_ARGS="$@"

    echo ""
    echo "============================================================"
    echo "  [$RUN_LABEL] run_id=$RUN_ID"
    echo "============================================================"
    XLA_PYTHON_CLIENT_PREALLOCATE=false python -m JaxCQL.conservative_sac_main_sam \
        $COMMON_ARGS \
        --run_id=$RUN_ID \
        --run_label="$RUN_LABEL" \
        $EXTRA_ARGS
    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  ERROR: $RUN_LABEL failed (exit $EXIT)"
    else
        echo "  DONE: $RUN_LABEL"
    fi
}

# ── Run 0: Baseline CalQL (no SAM) ──────────────────────────────────────────
run_experiment 0 "R0-Baseline-CalQL" \
    --use_sam=False

# ── Run 1: Naive SAM (100% offline, fixed rho) ──────────────────────────────
run_experiment 1 "R1-NaiveSAM-fixed0.05" \
    --use_sam=True \
    --sam_start_ratio=0.0 \
    --sam_rho_schedule=fixed \
    --sam_rho_max=0.05 \
    --sam_rho_min=0.005

# ── Run 2: Late-Stage SAM (80%→100%, fixed rho) ─────────────────────────────
run_experiment 2 "R2-LateStage-fixed0.05" \
    --use_sam=True \
    --sam_start_ratio=0.8 \
    --sam_rho_schedule=fixed \
    --sam_rho_max=0.05 \
    --sam_rho_min=0.005

# ── Run 3: Adaptive Radius SAM (100%, cosine decay) ─────────────────────────
run_experiment 3 "R3-AdaptiveRho-cosine" \
    --use_sam=True \
    --sam_start_ratio=0.0 \
    --sam_rho_schedule=cosine_decay \
    --sam_rho_max=0.05 \
    --sam_rho_min=0.005

# ── Run 4: Combined Best (80%→100% + cosine decay) ──────────────────────────
run_experiment 4 "R4-Combined-Late+Cosine" \
    --use_sam=True \
    --sam_start_ratio=0.8 \
    --sam_rho_schedule=cosine_decay \
    --sam_rho_max=0.05 \
    --sam_rho_min=0.005

echo ""
echo "============================================================"
echo "  ALL 5 ABLATION RUNS COMPLETE"
echo "  WandB project: $PROJECT  |  env: $ENV"
echo "============================================================"
