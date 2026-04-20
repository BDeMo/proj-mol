#!/bin/bash
# Stage 2: Molecular Encoder and Instance-Centric Baselines
# Trains ProtoNet and MAML under specified episodic configurations.
#
# Env vars (all optional):
#   DATASETS  N_WAY  K_SHOT  EPISODES  PATIENCE  DEVICE

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Stage 2: Encoder & Baselines Training"
echo "=============================================="

DATASETS="${DATASETS:-tox21}"
N_WAY="${N_WAY:-5}"
K_SHOT="${K_SHOT:-1}"
EPISODES="${EPISODES:-30000}"
PATIENCE="${PATIENCE:-20}"
DEVICE="${DEVICE:-cuda}"

COMMON=(--datasets $DATASETS --n_way $N_WAY --k_shot $K_SHOT
        --episodes_train $EPISODES --patience $PATIENCE --device $DEVICE)

echo "Config: ${N_WAY}-way ${K_SHOT}-shot, datasets=${DATASETS}, episodes=${EPISODES}"

echo ""
echo "--- Training Prototypical Network ---"
python train.py --method proto "${COMMON[@]}"

echo ""
echo "--- Training MAML ---"
python train.py --method maml "${COMMON[@]}"

echo ""
echo "=== Stage 2 Complete ==="
echo "Checkpoints saved in ./checkpoints/"
echo "Training logs saved in ./logs/"
