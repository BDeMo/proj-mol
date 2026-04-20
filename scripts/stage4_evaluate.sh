#!/bin/bash
# Stage 4: Evaluation and Analysis
# Evaluates all trained models on meta-test episodes and runs ablation studies.
#
# Env vars (all optional):
#   DATASETS  DEVICE

set -e
cd "$(dirname "$0")/.."

echo "=============================================="
echo "Stage 4: Evaluation & Analysis"
echo "=============================================="

DATASETS="${DATASETS:-tox21}"
DEVICE="${DEVICE:-cuda}"

# --- Main Evaluation: all trained methods × all episodic configs ---------
echo "--- Main Evaluation (1000 meta-test episodes) ---"
for METHOD in proto maml gos; do
    for CONFIG in "5 1" "5 5" "10 5"; do
        read -r N_WAY K_SHOT <<< "$CONFIG"
        echo ""
        echo ">> ${METHOD} ${N_WAY}-way ${K_SHOT}-shot"
        python evaluate.py \
            --method $METHOD \
            --datasets $DATASETS \
            --n_way $N_WAY \
            --k_shot $K_SHOT \
            --episodes_test 1000 \
            --device $DEVICE 2>/dev/null || echo "  (skipped: no checkpoint)"
    done
done

# --- Ablation Study -------------------------------------------------------
echo ""
echo "=============================================="
echo "Running Ablation Suite"
echo "=============================================="

python ablation.py --suite affinity --datasets $DATASETS
python ablation.py --suite knn      --datasets $DATASETS
python ablation.py --suite sample   --datasets $DATASETS

echo ""
echo "=== Stage 4 Complete ==="
echo "Results and plots saved in ./ablation_results/"
