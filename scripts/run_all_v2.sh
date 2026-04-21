#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Graph-of-Shots v2 — one-click experiment runner.
#
# Covers:
#   Phase 0 : SSL encoder pretraining (M5)              — once, cached on disk
#   Phase 1 : Main v2 comparison, 3 configs × 2 init    — v2 vs ProtoNet
#   Phase 2 : Ablation A1  α sweep                       (M1 residual)
#   Phase 3 : Ablation A2  meta-GNN depth/type           (M2 over-smoothing)
#   Phase 4 : Ablation A3  edge masking                  (M3 bipartite)
#   Phase 5 : Ablation A4  contrastive λ sweep           (M4 InfoNCE)
#   Phase 6 : Ablation A5  SSL pretraining regimes       (M5 impact)
#   Phase 7 : Meta-test evaluation for all checkpoints
#
# Usage:
#   bash scripts/run_all_v2.sh                            # full run
#   SKIP_SSL=1 bash scripts/run_all_v2.sh                 # skip Phase 0
#   DATASETS="tox21 sider" bash scripts/run_all_v2.sh     # subset
#   DEVICE=cuda:1 bash scripts/run_all_v2.sh              # pick GPU
#
# Env overrides:
#   DATASETS    default: tox21 toxcast muv sider
#   DEVICE      default: cuda
#   EPISODES    default: 30000
#   PATIENCE    default: 20
#   SSL_EPOCHS  default: 30
#   LOG_FILE    default: ./run_v2.log
#   SKIP_SSL=1  reuse existing ./checkpoints/ssl_encoder.pt
# ═══════════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")/.."

export DATASETS="${DATASETS:-tox21 toxcast muv sider}"
export DEVICE="${DEVICE:-cuda}"
export EPISODES="${EPISODES:-30000}"
export PATIENCE="${PATIENCE:-20}"
export SSL_EPOCHS="${SSL_EPOCHS:-30}"
LOG_FILE="${LOG_FILE:-./run_v2.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

mkdir -p checkpoints logs_v2 ablation_v2

echo "╔══════════════════════════════════════════════════╗"
echo "║  Graph-of-Shots v2 — Full Experiment Pipeline   ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Datasets : ${DATASETS}"
echo "║  Device   : ${DEVICE}"
echo "║  Episodes : ${EPISODES}  (patience ${PATIENCE})"
echo "║  SSL ep.  : ${SSL_EPOCHS}"
echo "║  Started  : $(date)"
echo "╚══════════════════════════════════════════════════╝"

COMMON_EPIS=(--datasets $DATASETS --episodes_train $EPISODES \
             --patience $PATIENCE --device $DEVICE \
             --log_dir ./logs_v2 --save_dir ./checkpoints)

SSL_CKPT=./checkpoints/ssl_encoder.pt

# ─── Phase 0 : SSL pretraining ────────────────────────────────────────
if [[ "${SKIP_SSL:-0}" == "1" && -f "$SSL_CKPT" ]]; then
    echo ">>> [Phase 0] SKIP — reusing $SSL_CKPT"
else
    echo ">>> [Phase 0] SSL pretraining (~${SSL_EPOCHS} epochs)"
    python pretrain_ssl.py \
        --datasets $DATASETS --device $DEVICE \
        --epochs $SSL_EPOCHS --batch_size 256 \
        --out $SSL_CKPT
fi

# helper
run_gos_v2() {
    # $1=tag  $2…=extra args
    local tag="$1"; shift
    echo ""
    echo "────────────────────────────────────────"
    echo " >>> ${tag}"
    echo "────────────────────────────────────────"
    python train.py --method gos_v2 "${COMMON_EPIS[@]}" "$@"
}

# ─── Phase 1 : Main v2 vs ProtoNet comparison ─────────────────────────
# 5w1s / 5w5s / 10w5s × {no-SSL, SSL-init} — 6 runs total.
echo ""
echo "======================================================="
echo " Phase 1 : Main v2 comparison across episodic configs"
echo "======================================================="
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    # (a) v2 without SSL
    run_gos_v2 "v2_${N}w${K}s_no-ssl" \
        --n_way $N --k_shot $K --affinity cosine --meta_k 5 \
        --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
        --v2_bipartite --v2_contrastive_lambda 0.5
    # (b) v2 with SSL-init encoder
    run_gos_v2 "v2_${N}w${K}s_ssl" \
        --n_way $N --k_shot $K --affinity cosine --meta_k 5 \
        --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
        --v2_bipartite --v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT
done

# ─── Phase 2 : A1  Residual α sweep ───────────────────────────────────
echo ""
echo "======================================================="
echo " Phase 2 : Ablation A1 — residual α sweep (5w5s)"
echo "======================================================="
for A in 0.0 0.3 0.5 0.7 1.0; do
    run_gos_v2 "A1_alpha${A}" \
        --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
        --refine_steps 2 --v2_alpha_init $A --v2_gnn_type gat \
        --v2_bipartite --v2_contrastive_lambda 0.5
done

# ─── Phase 3 : A2  Meta-GNN depth/type ────────────────────────────────
echo ""
echo "======================================================="
echo " Phase 3 : Ablation A2 — meta-GNN depth × type (5w5s)"
echo "======================================================="
for GNN in gcn gat; do
    for L in 1 2 3; do
        run_gos_v2 "A2_${GNN}_L${L}" \
            --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
            --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type $GNN \
            --meta_gnn_layers $L --v2_bipartite \
            --v2_contrastive_lambda 0.5
    done
done

# ─── Phase 4 : A3  Edge masking ───────────────────────────────────────
echo ""
echo "======================================================="
echo " Phase 4 : Ablation A3 — edge masking (5w5s)"
echo "======================================================="
run_gos_v2 "A3_full" \
    --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
    --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
    --v2_no_bipartite --v2_contrastive_lambda 0.5
run_gos_v2 "A3_bipartite" \
    --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
    --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
    --v2_bipartite --v2_contrastive_lambda 0.5

# ─── Phase 5 : A4  Contrastive λ sweep ────────────────────────────────
echo ""
echo "======================================================="
echo " Phase 5 : Ablation A4 — contrastive λ sweep (5w5s)"
echo "======================================================="
for L in 0.0 0.1 0.5 1.0; do
    run_gos_v2 "A4_lambda${L}" \
        --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
        --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
        --v2_bipartite --v2_contrastive_lambda $L
done

# ─── Phase 6 : A5  SSL pretraining regimes ────────────────────────────
echo ""
echo "======================================================="
echo " Phase 6 : Ablation A5 — SSL pretraining (5w5s)"
echo "======================================================="
# a) from scratch (no --ssl_ckpt)
run_gos_v2 "A5_scratch" \
    --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
    --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
    --v2_bipartite --v2_contrastive_lambda 0.5
# b) SSL-init + fine-tune
run_gos_v2 "A5_ssl_finetune" \
    --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
    --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
    --v2_bipartite --v2_contrastive_lambda 0.5 --ssl_ckpt $SSL_CKPT
# c) SSL-init + frozen encoder
run_gos_v2 "A5_ssl_frozen" \
    --n_way 5 --k_shot 5 --affinity cosine --meta_k 5 \
    --refine_steps 2 --v2_alpha_init 0.7 --v2_gnn_type gat \
    --v2_bipartite --v2_contrastive_lambda 0.5 \
    --ssl_ckpt $SSL_CKPT --freeze_encoder

# ─── Phase 7 : Meta-test evaluation ───────────────────────────────────
echo ""
echo "======================================================="
echo " Phase 7 : Meta-test evaluation (1000 episodes each)"
echo "======================================================="
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    # base v2
    for SSL_TAG in "no-ssl" "ssl"; do
        SSL_ARG=""
        [[ $SSL_TAG == "ssl" ]] && SSL_ARG="--ssl_ckpt $SSL_CKPT"
        python evaluate.py --method gos_v2 \
            --datasets $DATASETS --n_way $N --k_shot $K \
            --episodes_test 1000 --device $DEVICE \
            --log_dir ./logs_v2 --save_dir ./checkpoints $SSL_ARG \
            2>/dev/null || echo "  (skipped — no checkpoint for ${N}w${K}s ${SSL_TAG})"
    done
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  v2 Experiment Pipeline Complete                ║"
echo "║  Finished: $(date)"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Results:"
echo "  logs_v2/            history.json + *_test_results.json"
echo "  checkpoints/        best_*.pt + ssl_encoder.pt"
echo "  $LOG_FILE           full stdout/stderr"
