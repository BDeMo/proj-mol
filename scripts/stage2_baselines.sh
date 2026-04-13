#!/bin/bash
# Stage 2: Molecular Encoder and Instance-Centric Baselines
# Trains ProtoNet and MAML under specified episodic configurations.

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Stage 2: Encoder & Baselines Training"
echo "=============================================="

DATASETS="${DATASETS:-tox21}"
N_WAY="${N_WAY:-5}"
K_SHOT="${K_SHOT:-1}"
EPISODES="${EPISODES:-10000}"
DEVICE="${DEVICE:-cuda}"

echo "Config: ${N_WAY}-way ${K_SHOT}-shot, datasets=${DATASETS}"

# Train Prototypical Network
echo ""
echo "--- Training Prototypical Network ---"
python train.py \
    --method proto \
    --datasets ${DATASETS} \
    --n_way ${N_WAY} \
    --k_shot ${K_SHOT} \
    --episodes_train ${EPISODES} \
    --device ${DEVICE}

# Train MAML
echo ""
echo "--- Training MAML ---"
python train.py \
    --method maml \
    --datasets ${DATASETS} \
    --n_way ${N_WAY} \
    --k_shot ${K_SHOT} \
    --episodes_train ${EPISODES} \
    --device ${DEVICE}

echo ""
echo "=== Stage 2 Complete ==="
echo "Checkpoints saved in ./checkpoints/"
echo "Training logs saved in ./logs/"
