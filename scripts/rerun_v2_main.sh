#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Re-run the 6 v2 main Phase-1 configs (5w1s / 5w5s / 10w5s × {no-ssl,
# ssl}) with --exp_name, then evaluate on 1000 meta-test episodes.
#
# Why: those runs used the OLD code (no --exp_name), so both no-ssl and
# ssl wrote to the same checkpoint file.  The later eval pass loaded
# checkpoints that didn't exist → random-model evaluation → bogus
# numbers in logs_v2_rerun/v2_*_test_results.json.
#
# Usage: bash scripts/rerun_v2_main.sh
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"
LOG_DIR="${LOG_DIR:-./logs_v2_rerun}"
mkdir -p "$LOG_DIR" checkpoints

if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    if command -v nvidia-smi &>/dev/null; then
        mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    else
        GPU_LIST=(0)
    fi
fi
N_GPU=${#GPU_LIST[@]}

SSL_CKPT=./checkpoints/ssl_encoder.pt
if [[ ! -f "$SSL_CKPT" ]]; then
    echo "ERROR: missing $SSL_CKPT — run SSL pretrain first"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Rerun v2 main Phase-1 (6 jobs, with --exp_name)        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  GPUs     : ${GPU_LIST[*]}  (${N_GPU} workers)"
echo "║  Episodes : ${EPISODES}  (patience ${PATIENCE})"
echo "║  Log dir  : ${LOG_DIR}"
echo "║  Started  : $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Remove the stale Phase-1 checkpoints that went under the OLD name
echo ""
echo ">>> Removing stale Phase-1 checkpoints"
for f in checkpoints/gos_v2_5w1s_cosine_k5_*.pt \
         checkpoints/gos_v2_5w5s_cosine_k5_*.pt \
         checkpoints/gos_v2_10w5s_cosine_k5_*.pt \
         checkpoints/v2_5w1s_no-ssl_*.pt \
         checkpoints/v2_5w1s_ssl_*.pt \
         checkpoints/v2_5w5s_no-ssl_*.pt \
         checkpoints/v2_5w5s_ssl_*.pt \
         checkpoints/v2_10w5s_no-ssl_*.pt \
         checkpoints/v2_10w5s_ssl_*.pt; do
    [[ -f "$f" ]] && rm -v "$f" || true
done

# ── Build job list ────────────────────────────────────────────────
COMMON="--method gos_v2 --datasets $DATASETS --episodes_train $EPISODES \
--patience $PATIENCE --device cuda --log_dir $LOG_DIR --save_dir ./checkpoints"

JOBS=()
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    CORE="--n_way $N --k_shot $K --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite --v2_contrastive_lambda 0.5"
    JOBS+=("v2_${N}w${K}s_no-ssl|$COMMON --exp_name v2_${N}w${K}s_no-ssl $CORE")
    JOBS+=("v2_${N}w${K}s_ssl|$COMMON --exp_name v2_${N}w${K}s_ssl $CORE --ssl_ckpt $SSL_CKPT")
done
TOTAL=${#JOBS[@]}
echo ""
echo ">>> ${TOTAL} training jobs queued"

# ── Round-robin dispatcher ────────────────────────────────────────
declare -A BUSY
for G in "${GPU_LIST[@]}"; do BUSY[$G]=""; done

wait_any() {
    while true; do
        for g in "${!BUSY[@]}"; do
            local pid="${BUSY[$g]}"
            if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                BUSY[$g]=""; echo "$g"; return
            fi
        done
        sleep 2
    done
}

for i in "${!JOBS[@]}"; do
    entry="${JOBS[$i]}"
    tag="${entry%%|*}"; args="${entry#*|}"
    free=""
    for g in "${!BUSY[@]}"; do
        [[ -z "${BUSY[$g]}" ]] && { free=$g; break; }
    done
    [[ -z "$free" ]] && free=$(wait_any)
    echo "[$((i+1))/$TOTAL]  ${tag} → GPU ${free}"
    (
        CUDA_VISIBLE_DEVICES=$free python train.py $args > "$LOG_DIR/${tag}.log" 2>&1
        echo "[$(date +%H:%M:%S)]  done   ${tag}"
    ) &
    BUSY[$free]=$!
done
echo ""
echo ">>> All training dispatched, waiting..."
wait

# ── Evaluate ──────────────────────────────────────────────────────
echo ""
echo ">>> Meta-test evaluation (1000 episodes each)"
: > "$LOG_DIR/eval.log"
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    for VAR in "no-ssl" "ssl"; do
        TAG="v2_${N}w${K}s_${VAR}"
        SSL_ARG=""
        [[ "$VAR" == "ssl" ]] && SSL_ARG="--ssl_ckpt $SSL_CKPT"
        CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python evaluate.py \
            --method gos_v2 --exp_name "$TAG" \
            --datasets $DATASETS --n_way $N --k_shot $K \
            --episodes_test 1000 --device cuda \
            --log_dir $LOG_DIR --save_dir ./checkpoints $SSL_ARG \
            2>&1 | tee -a "$LOG_DIR/eval.log" \
            || echo "  (missing ${TAG})"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Phase-1 rerun complete                                 ║"
echo "║  Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:  grep -E 'Best val|ROC-AUC' ${LOG_DIR}/v2_*.log"
