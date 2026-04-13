#!/bin/bash
# Run the complete Graph-of-Shots pipeline end-to-end.
# Usage: bash scripts/run_all.sh [DATASETS]
# Example: bash scripts/run_all.sh "tox21 sider"

set -e
cd "$(dirname "$0")/.."

export DATASETS="${1:-tox21}"
export DEVICE="${DEVICE:-cuda}"

echo "=============================================="
echo "Graph-of-Shots: Full Pipeline"
echo "Datasets: ${DATASETS}"
echo "Device: ${DEVICE}"
echo "=============================================="

# Stage 1: Data Pipeline
bash scripts/stage1_data_pipeline.sh

# Stage 2: Baselines (5-way 1-shot and 5-way 5-shot)
for K in 1 5; do
    export K_SHOT=$K
    export N_WAY=5
    bash scripts/stage2_baselines.sh
done

# Stage 3: Graph-of-Shots (all affinity functions, 5-way 1-shot and 5-way 5-shot)
for K in 1 5; do
    for AFF in cosine bilinear attention; do
        export K_SHOT=$K
        export N_WAY=5
        export AFFINITY=$AFF
        bash scripts/stage3_graph_of_shots.sh
    done
done

# Stage 4: Evaluation
bash scripts/stage4_evaluate.sh

echo ""
echo "=============================================="
echo "ALL STAGES COMPLETE"
echo "=============================================="
