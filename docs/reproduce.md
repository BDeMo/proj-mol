# Reproducing the Results

All commands assume you are at the repo root.

## 0. Environment setup

```bash
git clone https://github.com/BDeMo/proj-mol.git
cd proj-mol
pip install -r requirements.txt
```

Hardware assumed: one or more NVIDIA GPUs with CUDA 11.8+. CPU works
but is 50× slower. PyG `torch-scatter` / `torch-sparse` wheels must
match your torch version.

## 1. First-run data download (once)

The pipeline pulls MoleculeNet via PyG on first load and caches it in
`data_cache/`.

```bash
# verify data pipeline works (prints per-task statistics)
bash scripts/stage1_data_pipeline.sh
```

After this you should see:
- `data_cache/tox21/`, `data_cache/toxcast/`, `data_cache/muv/`,
  `data_cache/sider/` — raw MoleculeNet processed by PyG
- `data_cache/_merged_muv-sider-tox21-toxcast_mp16.pt` — ~22 MB
  consolidated cache used by our loader

## 2. v1 baseline + GoS (single GPU, ~3 h)

```bash
# Stage 2: ProtoNet + FOMAML across 3 configs (~30 min each on A100)
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    N_WAY=$N K_SHOT=$K bash scripts/stage2_baselines.sh
done

# Stage 3: GoS v1 across 3 configs + affinity/kNN ablations
for CFG in "5 1" "5 5" "10 5"; do
    read -r N K <<< "$CFG"
    N_WAY=$N K_SHOT=$K bash scripts/stage3_graph_of_shots.sh
done

# Stage 4: 1000-episode meta-test + affinity/kNN/sample-eff ablation
bash scripts/stage4_evaluate.sh
```

Outputs land in `logs/` and `checkpoints/`. Corresponding archived
logs for comparison: `logs/v1_stages/` and `logs/v1_final_eval/`.

## 3. v2 SSL pretrain + ablation + eval (multi-GPU, ~1–1.5 h on 8×A100)

Easiest path — one command:

```bash
bash scripts/run_all_v2_parallel.sh
```

This runs:

1. **Phase 0 — SSL pretrain** (~15 min, GPU 0 only, sequential).
   Output: `checkpoints/ssl_encoder.pt` and
   `logs/v2/phase0_ssl.log`.
2. **Phase 1 — main v2 × 3 configs × {no-SSL, SSL}** = 6 training
   jobs.
3. **Phases 2–6 — ablations A1 through A5** = 20 training jobs.
4. **Phase 7 — 1000-ep meta-test evaluation** for Phase-1 checkpoints.

Per-job logs: `logs/v2/<tag>.log`. Checkpoints:
`checkpoints/<tag>_best.pt` and `checkpoints/<tag>_last.pt`.

Env-var overrides:

```bash
GPUS=0,1,2,3 bash scripts/run_all_v2_parallel.sh      # subset of GPUs
DATASETS="tox21 sider" bash scripts/run_all_v2_parallel.sh  # fewer datasets
EPISODES=15000 bash scripts/run_all_v2_parallel.sh    # shorter runs
SKIP_SSL=1 bash scripts/run_all_v2_parallel.sh        # reuse existing SSL ckpt
```

## 4. Individual runs (manual)

### SSL pretraining only
```bash
python pretrain_ssl.py --epochs 30 --batch_size 256 \
    --out checkpoints/ssl_encoder.pt
```

### A single v2 config
```bash
python train.py --method gos_v2 \
    --exp_name v2_5w5s_ssl \
    --datasets tox21 toxcast muv sider \
    --n_way 5 --k_shot 5 \
    --affinity cosine --meta_k 5 --refine_steps 2 \
    --v2_alpha_init 0.7 --v2_gnn_type gat --v2_bipartite \
    --v2_contrastive_lambda 0.5 \
    --ssl_ckpt checkpoints/ssl_encoder.pt \
    --episodes_train 30000 --patience 20 \
    --save_dir checkpoints --log_dir logs/v2
```

### Meta-test on a trained checkpoint
```bash
python evaluate.py --method gos_v2 \
    --exp_name v2_5w5s_ssl \
    --datasets tox21 toxcast muv sider \
    --n_way 5 --k_shot 5 \
    --episodes_test 1000 \
    --ssl_ckpt checkpoints/ssl_encoder.pt \
    --save_dir checkpoints --log_dir logs/v2
```

`evaluate.py` writes
`logs/v2/<exp_name>_test_results.json` with
`{roc_auc_mean, roc_auc_std, roc_auc_ci95, accuracy_mean, …}`.

## 5. Resuming interrupted runs

`train.py` saves `<save_dir>/<exp_name>_last.pt` at every evaluation
round (model + optimizer + episode counter + history). Re-invoke the
same command and it picks up where it stopped. SIGINT / SIGTERM
handlers flush a `_last.pt` before exit (useful for SLURM
pre-emption).

```bash
# Auto-resume (default):
python train.py --method gos_v2 --exp_name v2_5w5s_ssl ...

# Force from scratch:
python train.py ... --no_resume

# Resume from explicit path:
python train.py ... --resume_ckpt checkpoints/some_other.pt
```

## 6. Building figures / PPT

Figures are generated from the `logs/v2_rerun/*.log` numbers by the
plotting script (currently inline in
`presentation/_plot_v2_results.py`; regeneration is a manual step —
numbers are hard-coded from the logs).

The 12-slide PPT is
`presentation/Graph-of-Shots.pptx`. To rebuild from the source
script, see commit messages for the `_build_pptx.py` that was
temporarily added to `presentation/`.

## 7. Checkpoint naming

Every run is uniquely keyed by `--exp_name`. If omitted,
`utils.build_exp_name(args)` auto-generates a config-aware name like
`gos_v2_5w5s_cosine_k5_r2_a0.7_gatL1_bp_lam0.5_ssl` so distinct
configurations never overwrite each other.

File convention per run (all under `checkpoints/`):
- `<exp_name>_best.pt` — model state_dict with highest val AUC
- `<exp_name>_last.pt` — full training state for resume (model +
  optimizer + episode + history + best_val_auc + patience_counter)

History per run (under `<log_dir>/`):
- `<exp_name>_history.json` — list of per-eval (val_auc, val_loss,
  train_loss, val_acc)
- `<exp_name>_test_results.json` — 1000-episode meta-test output from
  `evaluate.py`
