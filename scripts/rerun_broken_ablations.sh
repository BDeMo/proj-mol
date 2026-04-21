#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Re-run only the ablation jobs that got corrupted by the checkpoint-name
# collision bug (A2 failed to load state_dict, A4/A5 skipped because
# resume saw episode 30000).
#
# The 6 main-comparison jobs (Phase 1) already produced correct val AUCs:
#   v2_5w1s_no-ssl   0.5830
#   v2_5w1s_ssl      0.6031
#   v2_5w5s_no-ssl   0.7089
#   v2_5w5s_ssl      0.7280
#   v2_10w5s_no-ssl  0.6947
#   v2_10w5s_ssl     0.7115
#
# We re-run:
#   * A1 (α sweep)              — previously ran but on the corrupted
#                                 checkpoint shared with Phase 1; redo
#                                 to be safe
#   * A2 (meta-GNN depth/type)  — all 6 jobs failed with state_dict error
#   * A3 (edge masking)         — both jobs touched corrupted ckpt
#   * A4 (λ sweep)              — all 4 jobs exited with "nothing to do"
#   * A5 (SSL regimes)          — all 3 jobs exited with "nothing to do"
#
# Plus a meta-test evaluation pass on the 6 Phase-1 checkpoints
# (the previous eval also loaded the wrong file).
#
# Usage: bash scripts/rerun_broken_ablations.sh
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"
LOG_DIR="${LOG_DIR:-./logs_v2}"
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
    echo "ERROR: missing $SSL_CKPT — run Phase 0 first"
    echo "  bash scripts/run_all_v2_parallel.sh  (it will skip completed Phase 1 via resume)"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Rerun: Ablations A1–A5 (Phase 1 main runs kept)        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  GPUs     : ${GPU_LIST[*]}  (${N_GPU} workers)"
echo "║  Episodes : ${EPISODES}  (patience ${PATIENCE})"
echo "║  Started  : $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Clean up corrupted ablation checkpoints (NOT Phase-1 ones) ────
echo ""
echo ">>> Removing corrupted ablation checkpoints"
for f in checkpoints/gos_v2_5w5s_cosine_k5_*.pt \
         checkpoints/A1_*.pt checkpoints/A2_*.pt checkpoints/A3_*.pt \
         checkpoints/A4_*.pt checkpoints/A5_*.pt; do
    [[ -f "$f" ]] && rm -v "$f" || true
done

# ── Job list (ablations only) ─────────────────────────────────────
COMMON="--datasets $DATASETS --episodes_train $EPISODES --patience $PATIENCE \
--device cuda --log_dir $LOG_DIR --save_dir ./checkpoints"

JOBS=()
add_job() {
    JOBS+=("$1|--method gos_v2 --exp_name $1 $COMMON $2")
}

# A1 α sweep (SSL-init so the residual effect is measured on the strong encoder)
for A in 0.0 0.3 0.5 0.7 1.0; do
    add_job "A1_alpha${A}" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init $A --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
done

# A2 meta-GNN depth × type
for GNN in gcn gat; do
    for L in 1 2 3; do
        add_job "A2_${GNN}_L${L}" "--n_way 5 --k_shot 5 --affinity cosine \
--meta_k 5 --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type $GNN \
--meta_gnn_layers $L --v2_bipartite --v2_contrastive_lambda 0.5 \
--ssl_ckpt $SSL_CKPT"
    done
done

# A3 edge masking
add_job "A3_full" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_no_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
add_job "A3_bipartite" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"

# A4 contrastive λ sweep
for L in 0.0 0.1 0.5 1.0; do
    add_job "A4_lambda${L}" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda $L --ssl_ckpt $SSL_CKPT"
done

# A5 SSL regimes
add_job "A5_scratch" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5"
add_job "A5_ssl_finetune" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
add_job "A5_ssl_frozen" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT --freeze_encoder"

TOTAL=${#JOBS[@]}
echo ""
echo ">>> ${TOTAL} ablation jobs queued"

# ── Dispatcher ────────────────────────────────────────────────────
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
echo ">>> All ablations dispatched, waiting..."
wait

# ── Re-evaluate Phase-1 checkpoints (on 1000 meta-test episodes) ─
echo ""
echo ">>> Re-evaluating Phase-1 main checkpoints on meta-test"
> "$LOG_DIR/eval.log"
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    for TAG in "v2_${N}w${K}s_no-ssl" "v2_${N}w${K}s_ssl"; do
        SSL_ARG=""
        [[ "$TAG" == *_ssl ]] && SSL_ARG="--ssl_ckpt $SSL_CKPT"
        CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python evaluate.py \
            --method gos_v2 --exp_name "$TAG" \
            --datasets $DATASETS --n_way $N --k_shot $K \
            --episodes_test 1000 --device cuda \
            --log_dir $LOG_DIR --save_dir ./checkpoints $SSL_ARG \
            2>&1 | tee -a "$LOG_DIR/eval.log" \
            || echo "  (missing ckpt ${TAG})"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Rerun complete                                         ║"
echo "║  Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:  grep -E 'Best val|ROC-AUC' ${LOG_DIR}/*.log"
