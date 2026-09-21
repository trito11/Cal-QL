#!/bin/bash
# Script to run CQL + Q-SAM on D4RL environments
# Usage:
#   bash run_cql_qsam.sh [env_name] [seed] [device]
# Example:
#   bash run_cql_qsam.sh halfcheetah-medium-v2 0 cuda

ENV_NAME=${1:-"halfcheetah-medium-v2"}
SEED=${2:-0}
DEVICE=${3:-"cuda"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/tri/miniconda3/envs/Cal-QL/bin/python"

cd "$SCRIPT_DIR/algorithms/offline"

echo "=========================================================="
echo "Starting CQL + Q-SAM"
echo "Env:    $ENV_NAME"
echo "Seed:   $SEED"
echo "Device: $DEVICE"
echo "=========================================================="

$PYTHON_BIN cql_Q-SAM.py \
    --env "$ENV_NAME" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --rho 0.0001 \
    --gamma 0.9 \
    --sam_start_step 100000 \
    --discor True \
    --max_timesteps 1000000 \
    --eval_freq 5000
