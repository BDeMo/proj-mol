#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Graph-of-Shots v2 — multi-GPU experiment runner.
#
# Auto-detects all GPUs (nvidia-smi) and distributes training jobs across
# them via a round-robin dispatcher + bounded parallelism.  Every job
# passes --exp_name <tag> so each run writes to its own checkpoint —
# no cross-run overwrites, resume works per-tag.
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"
export SSL_EPOCHS="${SSL_EPOCHS:-30}"
LOG_DIR="${LOG_DIR:-./logs/v2}"
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

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Graph-of-Shots v2 — Parallel Runner                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  GPUs     : ${GPU_LIST[*]}  (${N_GPU} workers)"
echo "║  Datasets : ${DATASETS}"
echo "║  Episodes : ${EPISODES}  (patience ${PATIENCE})"
echo "║  Started  : $(date)"
echo "╚══════════════════════════════════════════════════════════╝"

SSL_CKPT=./checkpoints/ssl_encoder.pt

# ── Phase 0: SSL pretraining ──────────────────────────────────────
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

# ── Job list ──────────────────────────────────────────────────────
# Every job carries --exp_name <TAG> so checkpoints do not collide.
JOBS=()

COMMON="--datasets $DATASETS --episodes_train $EPISODES --patience $PATIENCE \
--device cuda --log_dir $LOG_DIR --save_dir ./checkpoints"

add_job() {
    # $1: tag  $2: extra args (method / hyperparams)
    JOBS+=("$1|--method gos_v2 --exp_name $1 $COMMON $2")
}

# Phase 1: main v2 × 3 configs × {no-ssl, ssl}
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    CORE="--n_way $N --k_shot $K --affinity cosine --meta_k 5 --refine_steps 2 \
--v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite --v2_contrastive_lambda 0.5"
    add_job "v2_${N}w${K}s_no-ssl" "$CORE"
    add_job "v2_${N}w${K}s_ssl"    "$CORE --ssl_ckpt $SSL_CKPT"
done

# Phase 2: A1 α sweep  (5w5s, SSL-init by default so we measure residual effect)
for A in 0.0 0.3 0.5 0.7 1.0; do
    add_job "A1_alpha${A}" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init $A --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
done

# Phase 3: A2 meta-GNN depth × type
for GNN in gcn gat; do
    for L in 1 2 3; do
        add_job "A2_${GNN}_L${L}" "--n_way 5 --k_shot 5 --affinity cosine \
--meta_k 5 --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type $GNN \
--meta_gnn_layers $L --v2_bipartite --v2_contrastive_lambda 0.5 \
--ssl_ckpt $SSL_CKPT"
    done
done

# Phase 4: A3 edge masking
add_job "A3_full" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_no_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
add_job "A3_bipartite" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"

# Phase 5: A4 contrastive λ sweep
for L in 0.0 0.1 0.5 1.0; do
    add_job "A4_lambda${L}" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda $L --ssl_ckpt $SSL_CKPT"
done

# Phase 6: A5 SSL pretraining regimes
add_job "A5_scratch" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5"
add_job "A5_ssl_finetune" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT"
add_job "A5_ssl_frozen" "--n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
--refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
--v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT --freeze_encoder"

TOTAL_JOBS=${#JOBS[@]}
echo ""
echo ">>> Total training jobs: $TOTAL_JOBS  (will distribute over $N_GPU GPUs)"

# ── Round-robin dispatcher ────────────────────────────────────────
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

    free_gpu=""
    for g in "${!GPU_BUSY[@]}"; do
        if [[ -z "${GPU_BUSY[$g]}" ]]; then free_gpu=$g; break; fi
    done
    if [[ -z "$free_gpu" ]]; then
        free_gpu=$(wait_any)
    fi

    echo "[$((i+1))/$TOTAL_JOBS]  dispatching ${tag} to GPU ${free_gpu}"
    run_job "$free_gpu" "$tag" "$args"
done

echo ""
echo ">>> All jobs dispatched, waiting for the last wave..."
wait

# ── Evaluation ────────────────────────────────────────────────────
# Evaluate each --exp_name checkpoint on 1000 meta-test episodes.
echo ""
echo ">>> Running 1000-episode meta-test evaluation"

eval_one() {
    local tag=$1 n=$2 k=$3 extra=$4
    CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} python evaluate.py \
        --method gos_v2 --exp_name "$tag" \
        --datasets $DATASETS --n_way $n --k_shot $k \
        --episodes_test 1000 --device cuda \
        --log_dir $LOG_DIR --save_dir ./checkpoints $extra \
        2>&1 | tee -a "$LOG_DIR/eval.log" \
        || echo "  (skipped — no checkpoint ${tag})"
}

for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    eval_one "v2_${N}w${K}s_no-ssl" $N $K ""
    eval_one "v2_${N}w${K}s_ssl"    $N $K "--ssl_ckpt $SSL_CKPT"
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  v2 Parallel Pipeline Complete                          ║"
echo "║  Finished: $(date)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Per-job logs:   $LOG_DIR/*.log"
echo "Checkpoints:    ./checkpoints/*.pt"
