#!/bin/bash
# Stage 3: Graph-of-Shots — Core Method Training
# Trains the Graph-of-Shots model with specified affinity function and meta-k.
#
# Env vars (all optional):
#   DATASETS  N_WAY  K_SHOT  AFFINITY  META_K  REFINE_STEPS
#   EPISODES  PATIENCE  DEVICE

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Stage 3: Graph-of-Shots Training"
echo "=============================================="

DATASETS="${DATASETS:-tox21}"
N_WAY="${N_WAY:-5}"
K_SHOT="${K_SHOT:-1}"
AFFINITY="${AFFINITY:-cosine}"
META_K="${META_K:-5}"
REFINE_STEPS="${REFINE_STEPS:-2}"
EPISODES="${EPISODES:-30000}"
PATIENCE="${PATIENCE:-20}"
DEVICE="${DEVICE:-cuda}"

echo "Config: ${N_WAY}-way ${K_SHOT}-shot, affinity=${AFFINITY}, meta_k=${META_K}, refine=${REFINE_STEPS}, episodes=${EPISODES}"

python train.py \
    --method gos \
    --datasets $DATASETS \
    --n_way $N_WAY \
    --k_shot $K_SHOT \
    --affinity $AFFINITY \
    --meta_k $META_K \
    --refine_steps $REFINE_STEPS \
    --episodes_train $EPISODES \
    --patience $PATIENCE \
    --device $DEVICE

echo ""
echo "=== Stage 3 Complete ==="
echo "Checkpoint saved in ./checkpoints/"
