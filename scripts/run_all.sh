#!/bin/bash
# Run the complete Graph-of-Shots pipeline end-to-end.
#
# Usage:
#   bash scripts/run_all.sh                             # defaults: all four datasets, cuda
#   DATASETS="tox21 sider" bash scripts/run_all.sh      # override datasets
#   DEVICE=cpu bash scripts/run_all.sh                  # force CPU
#   EPISODES=5000 bash scripts/run_all.sh               # shorter smoke run
#
# Outputs are written to:
#   ./checkpoints/     -- best model weights per experiment
#   ./logs/            -- training history + main evaluation JSON
#   ./ablation_results/ -- ablation tables + convergence/sample-efficiency plots
#   ./run.log          -- full stdout+stderr of this run

set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export DEVICE="${DEVICE:-cuda}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"

LOG_FILE="${LOG_FILE:-./run.log}"

# Redirect everything to both screen and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Graph-of-Shots: Full Pipeline"
echo "Datasets: ${DATASETS}"
echo "Device:   ${DEVICE}"
echo "Episodes: ${EPISODES} (patience ${PATIENCE})"
echo "Log:      ${LOG_FILE}"
echo "Started:  $(date)"
echo "=============================================="

# ─── Stage 1: Data pipeline sanity check ───────────────────────────────────
bash scripts/stage1_data_pipeline.sh

# ─── Stage 2: Baselines (ProtoNet + MAML) across all three configs ─────────
for CONFIG in "5 1" "5 5" "10 5"; do
    read -r N_WAY K_SHOT <<< "$CONFIG"
    export N_WAY K_SHOT
    bash scripts/stage2_baselines.sh
done

# ─── Stage 3: Graph-of-Shots core + all ablations ──────────────────────────
# Main GoS runs: all three configs with default cosine affinity
for CONFIG in "5 1" "5 5" "10 5"; do
    read -r N_WAY K_SHOT <<< "$CONFIG"
    export N_WAY K_SHOT
    export AFFINITY=cosine
    export META_K=5
    bash scripts/stage3_graph_of_shots.sh
done

# Affinity ablation (5-way 5-shot)
for AFF in bilinear attention; do
    export N_WAY=5 K_SHOT=5 AFFINITY=$AFF META_K=5
    bash scripts/stage3_graph_of_shots.sh
done

# k-NN sparsification ablation (5-way 5-shot, cosine)
for K in 3 10 100; do
    export N_WAY=5 K_SHOT=5 AFFINITY=cosine META_K=$K
    bash scripts/stage3_graph_of_shots.sh
done

# Sample efficiency (5-way with K=2, K=10 — K=1 and K=5 already done above)
for K in 2 10; do
    export N_WAY=5 K_SHOT=$K AFFINITY=cosine META_K=5
    bash scripts/stage3_graph_of_shots.sh
done

# ─── Stage 4: Meta-test evaluation + ablation plots ────────────────────────
bash scripts/stage4_evaluate.sh

echo ""
echo "=============================================="
echo "ALL STAGES COMPLETE"
echo "Finished: $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  - Main results:        ./logs/*_test_results.json"
echo "  - Ablation summary:    ./ablation_results/ablation_summary.json"
echo "  - Convergence plots:   ./ablation_results/convergence_*.png"
echo "  - Sample efficiency:   ./ablation_results/sample_efficiency.png"
echo "  - Full log:            ${LOG_FILE}"
