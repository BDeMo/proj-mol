#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Graph-of-Shots v2 — multi-GPU experiment runner.
#
# Auto-detects all GPUs (nvidia-smi) and distributes 26 training runs
# across them via a job queue.  Each GPU processes jobs serially; jobs
# of different GPUs run in parallel.  Per-job resume is enabled by
# default so interrupted jobs pick up where they stopped.
#
# Usage:
#   bash scripts/run_all_v2_parallel.sh
#
# Env overrides:
#   GPUS         comma-list, e.g. "0,1,2,3"  (default: all visible)
#   DATASETS     default: "tox21 toxcast muv sider"
#   EPISODES     default: 30000
#   PATIENCE     default: 20
#   SSL_EPOCHS   default: 30
#   SKIP_SSL=1   reuse existing ./checkpoints/ssl_encoder.pt
#   LOG_DIR      per-job log dir (default: ./logs_v2)
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"
export SSL_EPOCHS="${SSL_EPOCHS:-30}"
LOG_DIR="${LOG_DIR:-./logs_v2}"
mkdir -p "$LOG_DIR" checkpoints ablation_v2

# ── detect GPUs ───────────────────────────────────────────────────
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

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Graph-of-Shots v2 — Parallel Runner                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  GPUs     : ${GPU_LIST[*]}  (${N_GPU} workers)"
echo "║  Datasets : ${DATASETS}"
echo "║  Episodes : ${EPISODES}  (patience ${PATIENCE})"
echo "║  Log dir  : ${LOG_DIR}"
echo "║  Started  : $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

SSL_CKPT=./checkpoints/ssl_encoder.pt

# ── Phase 0: SSL pretraining (runs on GPU 0, blocks the rest) ─────
if [[ "${SKIP_SSL:-0}" == "1" && -f "$SSL_CKPT" ]]; then
    echo ""
    echo ">>> [Phase 0] SKIP — reusing $SSL_CKPT"
else
    echo ""
    echo ">>> [Phase 0] SSL pretraining on GPU ${GPU_LIST[0]} (~${SSL_EPOCHS} epochs)"
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python pretrain_ssl.py \
        --datasets $DATASETS --device cuda \
        --epochs $SSL_EPOCHS --batch_size 256 \
        --out $SSL_CKPT \
        2>&1 | tee "$LOG_DIR/phase0_ssl.log"
fi

# ── Build the job list ────────────────────────────────────────────
# Format per line: "TAG|ARGS..."
JOBS=()

# Phase 1: main v2 × 3 configs × {no-ssl, ssl}
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    BASE="--method gos_v2 --datasets $DATASETS --n_way $N --k_shot $K \
--affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints"
    JOBS+=("v2_${N}w${K}s_no-ssl|$BASE")
    JOBS+=("v2_${N}w${K}s_ssl|$BASE --ssl_ckpt $SSL_CKPT")
done

# Phase 2: A1 α sweep
for A in 0.0 0.3 0.5 0.7 1.0; do
    JOBS+=("A1_alpha${A}|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init $A --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
done

# Phase 3: A2 meta-GNN depth × type
for GNN in gcn gat; do
    for L in 1 2 3; do
        JOBS+=("A2_${GNN}_L${L}|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type $GNN --meta_gnn_layers $L --v2_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
    done
done

# Phase 4: A3 edge masking
JOBS+=("A3_full|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_no_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
JOBS+=("A3_bipartite|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")

# Phase 5: A4 contrastive λ sweep
for L in 0.0 0.1 0.5 1.0; do
    JOBS+=("A4_lambda${L}|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda $L \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
done

# Phase 6: A5 SSL pretraining regimes
JOBS+=("A5_scratch|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
JOBS+=("A5_ssl_finetune|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")
JOBS+=("A5_ssl_frozen|--method gos_v2 --datasets $DATASETS \
--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT --freeze_encoder \
--episodes_train $EPISODES --patience $PATIENCE --device cuda \
--log_dir $LOG_DIR --save_dir ./checkpoints")

TOTAL_JOBS=${#JOBS[@]}
echo ""
echo ">>> Total training jobs: $TOTAL_JOBS  (will distribute over $N_GPU GPUs)"

# ── Dispatcher: round-robin + bounded parallelism ─────────────────
# Use a FIFO queue: dispatch $N_GPU jobs at once, wait for any to finish,
# then dispatch the next.  Each dispatched job pins to one GPU via
# CUDA_VISIBLE_DEVICES.
declare -A GPU_BUSY
for G in "${GPU_LIST[@]}"; do GPU_BUSY[$G]=""; done

run_job() {
    local gpu=$1 tag=$2 args=$3
    local logf="$LOG_DIR/${tag}.log"
    echo "[$(date +%H:%M:%S)]  start  tag=${tag}  gpu=${gpu}  log=${logf}"
    (
        CUDA_VISIBLE_DEVICES=$gpu python train.py $args \
            > "$logf" 2>&1
        echo "[$(date +%H:%M:%S)]  done   tag=${tag}  gpu=${gpu}  exit=$?"
    ) &
    GPU_BUSY[$gpu]=$!
}

wait_any() {
    # Wait for any of the running PIDs to finish; return the freed GPU.
    while true; do
        for g in "${!GPU_BUSY[@]}"; do
            local pid="${GPU_BUSY[$g]}"
            if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                GPU_BUSY[$g]=""
                echo "$g"
                return
            fi
        done
        sleep 2
    done
}

echo ""
for i in "${!JOBS[@]}"; do
    entry="${JOBS[$i]}"
    tag="${entry%%|*}"
    args="${entry#*|}"

    # find a free GPU, or wait for one
    free_gpu=""
    for g in "${!GPU_BUSY[@]}"; do
        if [[ -z "${GPU_BUSY[$g]}" ]]; then
            free_gpu=$g; break
        fi
    done
    if [[ -z "$free_gpu" ]]; then
        free_gpu=$(wait_any)
    fi

    echo "[$((i+1))/$TOTAL_JOBS]  dispatching ${tag} to GPU ${free_gpu}"
    run_job "$free_gpu" "$tag" "$args"
done

# wait for all remaining jobs
echo ""
echo ">>> All jobs dispatched, waiting for the last wave..."
wait

# ── Evaluation (cheap, can reuse GPU 0 sequentially) ──────────────
echo ""
echo ">>> Running 1000-episode meta-test evaluation"
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    for SSL_TAG in "no-ssl" "ssl"; do
        SSL_ARG=""
        [[ $SSL_TAG == "ssl" ]] && SSL_ARG="--ssl_ckpt $SSL_CKPT"
        CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python evaluate.py \
            --method gos_v2 --datasets $DATASETS --n_way $N --k_shot $K \
            --episodes_test 1000 --device cuda \
            --log_dir $LOG_DIR --save_dir ./checkpoints $SSL_ARG \
            2>&1 | tee -a "$LOG_DIR/eval.log" \
            || echo "  (skipped — no checkpoint for ${N}w${K}s ${SSL_TAG})"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  v2 Parallel Pipeline Complete                          ║"
echo "║  Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Per-job logs:   $LOG_DIR/*.log"
echo "Checkpoints:    ./checkpoints/*.pt"
