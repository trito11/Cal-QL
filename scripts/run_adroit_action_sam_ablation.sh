#!/usr/bin/env bash
# =============================================================================
# run_adroit_action_sam_ablation.sh
# Comprehensive Ablation Study Suite: Cal-QL + Expectile V(s) + Action SAM
# Target Environment: pen-binary-v0
#
# Nghiên cứu bóc tách ảnh hưởng của các thành phần chống sụt giảm (Anti-Dip):
#   Run 0 [Baseline-CalQL]        : Cal-QL chuẩn (không SAM, tau=0.7, mixing=0.5)
#   Run 1 [Expectile-SAM-Fixed]   : Bản gốc trước cải tiến (rho=0.05 cố định, mixing=0.5)
#   Run 2 [Ablation-DynamicMixing]: + Dynamic Mixing Ratio (mixing=-1.0)
#   Run 3 [Ablation-OnlineRhoDecay]: + Suy giảm bán kính SAM online (rho: 0.05 -> 0.01)
#   Run 4 [Ablation-ProtectedV]   : + Bảo vệ V_psi chỉ cập nhật trên offline data
#   Run 5 [Ablation-UncertaintySAM]: + Uncertainty-Aware SAM (|Q1 - Q2|)
#   Run 6 [Full-Combined-AntiDip] : TỔNG HỢP TOÀN BỘ (Dynamic Mixing + Protected V +
#                                   Uncertainty SAM + Online Decay + Warmup + CQL Online 0.2)
#   Run 7 [ActorCritic-SAM-Boost] : BỨT PHÁ VƯỢT CAL-QL (Actor-SAM robust policy +
#                                   Sweet-Spot Critic SAM rho_online=0.03 + Anti-Dip)
# =============================================================================

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tri/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

PYTHON=/home/tri/miniconda3/envs/Cal-QL/bin/python

ENV=pen-binary-v0
OUTPUT_DIR=./saved_models
PROJECT=Cal-QL-ActionSAM-Ablation

# Các siêu tham số chuẩn mực của Adroit Pen
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
    echo "  STARTING: [$RUN_LABEL] (run_id=$RUN_ID)"
    echo "  EXTRA ARGS: $EXTRA_ARGS"
    echo "========================================================================"
    
    PYTHONPATH=. $PYTHON -m JaxCQL.conservative_sac_main_sam \
        $COMMON_ARGS \
        --run_id=$RUN_ID \
        --run_label="$RUN_LABEL" \
        $EXTRA_ARGS

    local EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "  [ERROR] $RUN_LABEL failed with exit code $EXIT"
    else
        echo "  [SUCCESS] $RUN_LABEL completed successfully!"
    fi
}

# Chọn chạy đơn lẻ từng Run bằng tham số: ./run_adroit_action_sam_ablation.sh <run_id>
# Hoặc chạy toàn bộ tuần tự nếu không truyền tham số: ./run_adroit_action_sam_ablation.sh all
TARGET_RUN=${1:-"all"}

# ── Run 0: Baseline Cal-QL (Không SAM, tau=0.7, mixing=0.5) ───────────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "0" ]; then
    run_experiment 0 "R0-Baseline-CalQL" \
        --use_adv_action=False \
        --tau=0.7 \
        --mixing_ratio=0.5 \
        --v_offline_only_online=False
fi

# ── Run 1: Expectile SAM Fixed (rho=0.05 cố định, tau=0.8, mixing=0.5) ───────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "1" ]; then
    run_experiment 1 "R1-Expectile-SAM-Fixed" \
        --use_adv_action=True \
        --rho=0.05 \
        --tau=0.8 \
        --mixing_ratio=0.5 \
        --rho_online=-1.0 \
        --sam_rho_decay=False \
        --use_uncertainty_sam=False \
        --v_offline_only_online=False
fi

# ── Run 2: Ablation Dynamic Mixing Ratio (mixing=-1.0) ────────────────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "2" ]; then
    run_experiment 2 "R2-Ablation-DynamicMixing" \
        --use_adv_action=True \
        --rho=0.05 \
        --tau=0.8 \
        --mixing_ratio=-1.0 \
        --rho_online=-1.0 \
        --sam_rho_decay=False \
        --use_uncertainty_sam=False \
        --v_offline_only_online=False
fi

# ── Run 3: Ablation Online Rho Decay (rho: 0.05 -> 0.01) ──────────────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "3" ]; then
    run_experiment 3 "R3-Ablation-OnlineRhoDecay" \
        --use_adv_action=True \
        --rho=0.05 \
        --tau=0.8 \
        --mixing_ratio=0.5 \
        --rho_online=0.01 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=False \
        --v_offline_only_online=False
fi

# ── Run 4: Ablation Protected V_psi (v_offline_only_online=True) ──────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "4" ]; then
    run_experiment 4 "R4-Ablation-ProtectedV" \
        --use_adv_action=True \
        --rho=0.05 \
        --tau=0.8 \
        --mixing_ratio=0.5 \
        --rho_online=-1.0 \
        --sam_rho_decay=False \
        --use_uncertainty_sam=False \
        --v_offline_only_online=True
fi

# ── Run 5: Ablation Uncertainty-Aware SAM (|Q1 - Q2|) ─────────────────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "5" ]; then
    run_experiment 5 "R5-Ablation-UncertaintySAM" \
        --use_adv_action=True \
        --rho=0.05 \
        --tau=0.8 \
        --mixing_ratio=0.5 \
        --rho_online=-1.0 \
        --sam_rho_decay=False \
        --use_uncertainty_sam=True \
        --uncertainty_scale=10.0 \
        --v_offline_only_online=False
fi

# ── Run 6: Full Combined Anti-Dip (TỔNG HỢP TOÀN BỘ CẢI TIẾN) ─────────────────
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "6" ]; then
    run_experiment 6 "R6-Full-Combined-AntiDip" \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.01 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=10.0 \
        --v_offline_only_online=True \
        --tau=0.8 \
        --mixing_ratio=-1.0 \
        --cql_min_q_weight_online=0.2
fi

# ── Run 7: Actor-Critic Action-SAM Boost (BỨT PHÁ VƯỢT CAL-QL) ────────────────
# Kết hợp:
# - Critic SAM ở Sweet-Spot: rho_online=0.03, uncertainty_scale=2.5 (rho_eff ~ 0.02)
# - Actor-SAM: use_actor_sam=True, actor_rho=0.02 (tối ưu Policy trên flat plateaus)
# - Toàn bộ nền tảng Anti-Dip: Protected V_psi, Dynamic Mixing, CQL Online 0.2
if [ "$TARGET_RUN" = "all" ] || [ "$TARGET_RUN" = "7" ]; then
    run_experiment 7 "R7-ActorCritic-ActionSAM-Boost" \
        --use_adv_action=True \
        --sam_start_epoch=10 \
        --rho=0.05 \
        --rho_online=0.03 \
        --sam_rho_decay=True \
        --use_uncertainty_sam=True \
        --uncertainty_scale=2.5 \
        --v_offline_only_online=True \
        --tau=0.8 \
        --mixing_ratio=-1.0 \
        --cql_min_q_weight_online=0.2 \
        --use_actor_sam=True \
        --actor_rho=0.02
fi

echo ""
echo "========================================================================"
echo "  ABLATION STUDY SUITE COMPLETED!"
echo "========================================================================"

