#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/tri/.mujoco/mujoco210/bin:/usr/lib/nvidia
export D4RL_SUPPRESS_IMPORT_ERROR=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export CUDA_VISIBLE_DEVICES=0
# export WANDB_DISABLED=True

# Supported environments:
# kitchen-complete-v0, kitchen-partial-v0, kitchen-mixed-v0
env=${1:-kitchen-complete-v0}
seed=${2:-0}

PYTHON=/home/tri/miniconda3/envs/Cal-QL/bin/python

echo "=========================================================="
echo "Running Cal-QL on Franka Kitchen"
echo "Env: $env | Seed: $seed"
echo "=========================================================="

PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false $PYTHON -m JaxCQL.conservative_sac_main \
    --env=$env \
    --logging.output_dir="./saved_models" \
    --logging.online \
    --seed=$seed \
    --logging.project=Cal-QL-Kitchen \
    --save_model=True \
    --cql_min_q_weight=0.5 \
    --policy_arch=256-256 \
    --qf_arch=256-256-256 \
    --offline_eval_every_n_epoch=10 \
    --online_eval_every_n_env_steps=2000 \
    --eval_n_trajs=20 \
    --n_train_step_per_epoch_offline=1000 \
    --n_pretrain_epochs=250 \
    --max_online_env_steps=1e6 \
    --mixing_ratio=0.5 \
    --reward_scale=1.0 \
    --reward_bias=0.0 \
    --enable_calql=True
