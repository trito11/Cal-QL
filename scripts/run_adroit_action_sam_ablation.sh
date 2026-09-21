#!/usr/bin/env bash
# =============================================================================
# run_adroit_action_sam_ablation.sh
# Standardized Ablation Study Suite: Cal-QL + Action SAM + Actor SAM
# Target Environment: pen-binary-v0
#
# Ma trận đánh giá khoa học chuẩn mực (Orthogonal & Độc lập):
# Tất cả các Run 1 -> 4 đều dùng chung nền tảng V_ref Dual Anchor cố định
# để loại bỏ hoàn toàn nhiễu từ V động học theo Q, giúp đánh giá chính xác SAM:
#
#   Run 0 [Baseline-CalQL]           : Cal-QL chuẩn đối sánh (tác giả, mc_returns tĩnh, mixing=0.5)
#   Run 1 [Foundation-VRefAnchor]    : Cal-QL + Monotonic V_ref Dual Anchor (Snapshot epoch 20, No SAM)
#   Run 2 [Ablation-CriticSAM]       : Run 1 (V_ref) + Critic Action-SAM (rho: 0.05 -> 0.02, No Actor-SAM)
#   Run 3 [Ablation-ActorSAM]        : Run 1 (V_ref) + Actor-SAM (actor_rho: 0.02 -> 0.005, No Critic-SAM)
#   Run 4 [Proposed-ActorCriticSAM]  : Run 1 (V_ref) + Critic Action-SAM + Actor-SAM (Cốt lõi đề xuất)
#   Run 5 [FullBoost-AllIncluded]    : Run 4 + Dynamic Buffer Mixing (-1.0) + Soft CQL Online (0.2)
#   Run 6 [VRef-CriticSAM]           : Bóc tách Critic Action-SAM trên nền V_ref Dual Anchor
#   Run 7 [QSAM-ActorSAM]            : Actor áp dụng SAM từ pha Offline như trong Q-SAM (No Critic-SAM)
# =============================================================================

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tri/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tri/miniconda3/envs/Cal-QL/bin/python

ENV=${1:-"pen-binary-v0"}
TARGET_RUN=${2:-"1-5"}
MAX_STEPS_ARG=${3:-""}

if [ "$ENV" = "pen-binary-v0" ]; then
    DEFAULT_STEPS="1e5"
    EVAL_EVERY="1000"
elif [ "$ENV" = "door-binary-v0" ]; then
    DEFAULT_STEPS="2e5"
    EVAL_EVERY="2000"
else
    DEFAULT_STEPS="5e4"
    EVAL_EVERY="1000"
fi
MAX_ONLINE_STEPS=${MAX_STEPS_ARG:-$DEFAULT_STEPS}
PROJECT="Cal-QL-SAM-${ENV}"
OUTPUT_DIR="./saved_models"

echo "========================================================================"
echo "  Target Environment: $ENV"
echo "  Max Online Steps:   $MAX_ONLINE_STEPS (Eval every $EVAL_EVERY steps)"
echo "  Target Runs:        $TARGET_RUN"
echo "  Project:            $PROJECT"
echo "========================================================================"

# Các siêu tham số chuẩn mực của Adroit Hand
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
  --online_eval_every_n_env_steps=$EVAL_EVERY
  --eval_n_trajs=20
  --n_train_step_per_epoch_offline=1000
  --n_pretrain_epochs=20
  --max_online_env_steps=$MAX_ONLINE_STEPS
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
    echo "========================================================================"
    echo "  STARTING: [$RUN_LABEL] (run_id=$RUN_ID on $ENV)"
    echo "  EXTRA ARGS: $EXTRA_ARGS"
    echo "========================================================================"
    
    if [ "$RUN_ID" = "0" ]; then
        # Run 0 chuẩn đối sánh Cal-QL gốc: chạy trực tiếp module chuẩn conservative_sac_main
        PYTHONPATH=. $PYTHON -m JaxCQL.conservative_sac_main \
            $COMMON_ARGS \
            $EXTRA_ARGS
    else
        PYTHONPATH=. $PYTHON -m JaxCQL.conservative_sac_main_sam \
            $COMMON_ARGS \
            --run_id=$RUN_ID \
            --run_label="${ENV}-${RUN_LABEL}" \
            $EXTRA_ARGS
    fi

    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  [ERROR] $RUN_LABEL failed with exit code $EXIT"
    else
        echo "  [SUCCESS] $RUN_LABEL completed successfully!"
    fi
}

# Chọn chạy đơn lẻ từng Run hoặc theo dải:
#   ./run_adroit_action_sam_ablation.sh pen-binary-v0 2      (chỉ test R_mc + Critic-SAM)
#   ./run_adroit_action_sam_ablation.sh pen-binary-v0 7      (chạy Run 7 Q-SAM Actor-SAM)
#   ./run_adroit_action_sam_ablation.sh pen-binary-v0 1-7
#   ./run_adroit_action_sam_ablation.sh door-binary-v0 0-7
should_run() {
    local id=$1
    if [ "$TARGET_RUN" = "all" ]; then
        return 0
    elif [ "$TARGET_RUN" = "1-5" ] && [ "$id" -ge 1 ] && [ "$id" -le 5 ]; then
        return 0
    elif [ "$TARGET_RUN" = "0-5" ] && [ "$id" -ge 0 ] && [ "$id" -le 5 ]; then
        return 0
    elif [ "$TARGET_RUN" = "1-6" ] && [ "$id" -ge 1 ] && [ "$id" -le 6 ]; then
        return 0
    elif [ "$TARGET_RUN" = "0-6" ] && [ "$id" -ge 0 ] && [ "$id" -le 6 ]; then
        return 0
    elif [ "$TARGET_RUN" = "1-7" ] && [ "$id" -ge 1 ] && [ "$id" -le 7 ]; then
        return 0
    elif [ "$TARGET_RUN" = "0-7" ] && [ "$id" -ge 0 ] && [ "$id" -le 7 ]; then
        return 0
    elif [ "$TARGET_RUN" = "$id" ]; then
        return 0
    fi
    return 1
}

# ── Run 0: Baseline Cal-QL (Chuẩn đối sánh gốc tác giả) ──────────────────────
if should_run 0; then
    run_experiment 0 "R0-Baseline-CalQL" \
        --mixing_ratio=0.5
fi

# ── Run 1: Cal-QL + Monotonic V_ref Dual Anchor (Nền tảng chuẩn hóa) ────────
if should_run 1; then
    run_experiment 1 "R1-Foundation-VRefAnchor" \
        --use_v_ref=True \
        --use_adv_action=False \
        --use_actor_sam=False \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 2: Bóc tách Critic Action-SAM trên Cal-QL gốc (Thuần R_mc, KHÔNG V_ref) ──
if should_run 2; then
    run_experiment 2 "R2-CalQL-Rmc-CriticSAM" \
        --use_v_ref=False \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.02 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=5.0 \
        --use_actor_sam=False \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 3: Bóc tách Actor-SAM (Nền V_ref + Actor-SAM) ────────────────────────
if should_run 3; then
    run_experiment 3 "R3-Ablation-ActorSAM" \
        --use_v_ref=False \
        --use_adv_action=False \
        --use_actor_sam=True \
        --actor_sam_start_epoch=20 \
        --actor_rho=0.02 \
        --actor_rho_online=0.005 \
        --actor_rho_decay=True \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 4: Phương pháp Đề xuất Cốt lõi (Nền V_ref + Critic-SAM + Actor-SAM) ─
if should_run 4; then
    run_experiment 4 "R4-Proposed-ActorCriticSAM" \
        --use_v_ref=False \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.02 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=5.0 \
        --use_actor_sam=True \
        --actor_sam_start_epoch=20 \
        --actor_rho=0.02 \
        --actor_rho_online=0.005 \
        --actor_rho_decay=True \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 5: Full Boost (Run 4 + Dynamic Buffer Mixing + Soft CQL Online) ──────
if should_run 5; then
    run_experiment 5 "R5-FullBoost-AllIncluded" \
        --use_v_ref=False \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.02 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=5.0 \
        --use_actor_sam=True \
        --actor_sam_start_epoch=20 \
        --actor_rho=0.02 \
        --actor_rho_online=0.005 \
        --actor_rho_decay=True \
        --tau=0.7 \
        --mixing_ratio=-1.0 \
        --cql_min_q_weight_online=0.2 \
        --v_offline_only_online=False
fi

# ── Run 6: Bóc tách Critic Action-SAM trên nền V_ref Dual Anchor ───────────────
if should_run 6; then
    run_experiment 6 "R6-VRef-CriticSAM" \
        --use_v_ref=True \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.02 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=5.0 \
        --use_actor_sam=False \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 7: Bóc tách Actor-SAM Offline theo Q-SAM (Actor áp dụng SAM như trong Q-SAM) ──
# Đặc tính Q-SAM:
# - Áp dụng SAM trực tiếp lên Actor ngay từ pha Offline (warmup 10% offline = epoch 2, tương tự 100k/1M steps của Q-SAM; có thể chỉnh thành 10 nếu muốn warmup 50%)
# - Không dùng Critic Action-SAM (--use_adv_action=False)
# - Không dùng V_ref (--use_v_ref=False, đối sánh trực tiếp với Cal-QL chuẩn)
if should_run 7; then
    run_experiment 7 "R7-QSAM-ActorSAM" \
        --use_v_ref=False \
        --use_adv_action=False \
        --use_actor_sam=True \
        --actor_sam_start_epoch=2 \
        --actor_rho=0.02 \
        --actor_rho_online=0.005 \
        --actor_rho_decay=True \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

echo ""
echo "========================================================================"
echo "  ABLATION STUDY SUITE COMPLETED!"
echo "========================================================================"
