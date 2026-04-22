# Changelog

Reverse-chronological list of commits that shipped changes users (or
future us) need to know about. Commit hashes on
[github.com/BDeMo/proj-mol](https://github.com/BDeMo/proj-mol).

## 2026-04-21  — v2 + SSL beats ProtoNet

| Commit | Summary |
|---|---|
| `8082732` | Consolidate all run logs under `logs/`; track them in git |
| `bf6a348` | PPT: embed new v2 results figures (main + 5-panel ablation grid) |
| `dd33170` | Add `scripts/rerun_v2_main.sh` — re-run 6 Phase-1 configs with `--exp_name` |
| `579fdc5` | **Fix checkpoint-name collision**: `--exp_name` + config-aware auto-naming in `utils.build_exp_name` |
| `5837709` | Training: line-based progress prints + data cache + 8-GPU parallel runner |
| `24a0a66` | Add resume mechanism: auto-checkpoint `<exp>_last.pt` + auto-resume + SIGINT handler |
| `7aeaadc` | PPT: condense to 14 slides, native MS shapes, unified Calibri, per-equation formula PNGs |
| `f12b7c9` | **Hotfix**: strip per-graph `y`/`smiles` in `molnet_loader` to allow cross-dataset `Batch.from_data_list` collate |
| `842fdd2` | Rebuild PPT: native text + math-rendered formula images + clean diagrams |
| `aaf7708` | **Hotfix**: SSL augmentations strip `y`/`smiles` to avoid Batch collate failure |
| `20d25b1` | Add GoS v2: residual prototype + GAT + bipartite + contrastive + SSL pretrain |
| `896d260` | Fix pptx: strip float-EMU coords that MS Office rejects as corrupt |
| `376d70b` | Add edge features (GINEConv with bond features) + iterative refinement + one-shot `run_all.sh` |
| `d0c5062` | **Fix GoS near-random performance**: use prototype head instead of shared `Linear(d, N-way)` classifier |
| `e3087c0` | Fix MAML RuntimeError: detach inner loop params and use original weights for meta-gradient |
| `e1b23ae` | Fix crash when `n_way > available tasks` for small datasets |
| `2603873` | Fix Linux compat: `+x` on scripts, remove `.claude/`, update `.gitignore` |
| `b88b637` | Add `.gitattributes` to enforce LF line endings |
| `48edddb` | Initial commit: Graph-of-Shots few-shot molecular classification |

## Bug fixes worth remembering

### Checkpoint-name collision (`579fdc5`)
Before this commit every run of shape `(method, n_way, k_shot, affinity, meta_k)`
shared an exp_name. The 21 v2 5w5s ablations all wrote to the same
`gos_v2_5w5s_cosine_k5_*.pt`. Consequences:
- A2 ablations (different GNN depth/type) crashed with `state_dict`
  size mismatch when resume loaded a stale file.
- A4/A5 ablations finished instantly with "resume found ep 30000,
  nothing to do".
- Meta-test evaluation loaded whichever run wrote last.

**Fix**: `utils.build_exp_name(args)` now composes a name from every
hyper-parameter that influences the model, and `--exp_name` lets scripts
pin a human-readable tag per job. See `logs/v2_initial/` for the broken
run and `logs/v2_rerun/` for the clean one.

### Prototype head replaces shared Linear (`d0c5062`)
Our first GoS used `self.classifier = nn.Linear(d, N_way)` on top of
the meta-GNN output. It collapsed to ≈0.50 AUC because episodic
training assigns "class 0" to a different assay every episode — the
shared head receives contradictory gradients. The fix is a prototype
head: scatter-mean support embeddings to get per-class prototypes, then
classify queries by negative Euclidean distance. This mirrors
ProtoNet's success: **zero persistent per-class parameters**.

### Cross-dataset `Batch.from_data_list` collate (`f12b7c9`, `aaf7708`)
MoleculeNet datasets have different `y` widths (Tox21 = 12, ToxCast =
617, MUV = 17, SIDER = 27). When an episode mixed datasets, PyG's
`Batch` tried `torch.cat` on the `y` tensors and crashed. We now strip
`y` and `smiles` from every `Data` before returning it from
`load_molnet`. Labels live only in `task_indices`, which is what we
actually use.

### SSL augmentation crash (`aaf7708`)
`drop_nodes` and `drop_edges` created a fresh `Data(...)` that did not
carry over the original's `y`/`smiles`, breaking cross-dataset batches
in the contrastive objective. Switched to `data.clone()` then
in-place edits.

### FOMAML meta-gradient (`e3087c0`)
The FOMAML inner loop called `torch.autograd.grad(..., create_graph=False)`
which produced leaf tensors with `requires_grad=False`; a subsequent
inner step then failed with "element 0 of tensors does not require
grad". Fixed by explicitly `param.clone().detach().requires_grad_(True)`
at each inner step and reverting to the original weights for the outer
meta-gradient.

## Design-level changes

### v2 introduces five toggles (`20d25b1`)
- `--v2_alpha_init` — M1 residual prototype coefficient
- `--v2_gnn_type {gcn, gat}` — M2 meta-GNN backbone
- `--v2_bipartite / --v2_no_bipartite` — M3 edge masking
- `--v2_contrastive_lambda` — M4 InfoNCE weight
- `--ssl_ckpt` + `--freeze_encoder` — M5 SSL pretraining

### Log directory layout (`8082732`)
Previously logs sat in four parallel top-level directories
(`logs/`, `logs_v2/`, `logs_v2_rerun/`, `logs_final/`, `runs/`).
Everything is now under `logs/<subpath>/`:
- `logs/v1_stages/` — v1 full-pipeline stage logs
- `logs/v1_final_eval/` — v1 final 1000-episode evaluation
- `logs/v2_initial/` — v2 first run with the collision bug
- `logs/v2_rerun/` — v2 current best; v2 + SSL beat ProtoNet

Default `LOG_DIR` in runner scripts points under `./logs/`.
