# Repository Structure

Everything referenced here is a path relative to the repo root
(`<repo>/` = `/workspace/uva-proj-mol/` on Windows, wherever you cloned
it on Linux).

```
<repo>/
├── config.py                      # All CLI arguments (argparse)
├── utils.py                       # Seeding, metrics, build_exp_name
├── train.py                       # Episodic training entry point
├── evaluate.py                    # 1000-episode meta-test evaluation
├── ablation.py                    # v1 ablation driver (v1 era)
├── pretrain_ssl.py                # M5 SSL pretraining entry point
│
├── data/                          # Dataset pipeline
│   ├── __init__.py
│   ├── molnet_loader.py           # load_all_datasets, task_indices,
│   │                                cache to data_cache/*.pt
│   ├── task_splitter.py           # meta-train / val / test split
│   └── episode_sampler.py         # N-way K-shot episodic sampler
│
├── models/                        # All networks
│   ├── __init__.py
│   ├── mpnn_encoder.py            # 3×GINEConv + global mean pool
│   ├── proto_net.py               # Prototypical Network baseline
│   ├── maml.py                    # FOMAML baseline
│   ├── graph_of_shots.py          # GoS v1 (meta-graph + meta-GNN + prototype head)
│   ├── graph_of_shots_v2.py       # GoS v2 (M1–M4 toggles)
│   └── ssl_pretrain.py            # M5 MolCLR-style augmentations + InfoNCE
│
├── scripts/                       # Experiment runners
│   ├── stage1_data_pipeline.sh    # v1: verify data pipeline
│   ├── stage2_baselines.sh        # v1: train ProtoNet + MAML
│   ├── stage3_graph_of_shots.sh   # v1: train GoS v1
│   ├── stage4_evaluate.sh         # v1: 1000-ep eval + ablations
│   ├── run_all.sh                 # v1: full serial pipeline
│   ├── run_all_v2.sh              # v2 serial (single-GPU) runner
│   ├── run_all_v2_parallel.sh     # v2 multi-GPU (round-robin, --exp_name)
│   ├── rerun_broken_ablations.sh  # v2: re-run A1–A5 after the
│   │                                 checkpoint-collision fix
│   └── rerun_v2_main.sh           # v2: re-run 6 Phase-1 main configs
│                                     with --exp_name
│
├── logs/                          # All experiment logs (tracked in git)
│   ├── v1_stages/                 # v1 stage-by-stage output
│   │   ├── stage1.log             #   data pipeline + dataset stats
│   │   ├── stage2.log             #   ProtoNet & MAML training
│   │   ├── stage3.log             #   GoS v1 training + ablations
│   │   └── stage4_all.log         #   full v1 evaluation
│   ├── v1_final_eval/             # v1: final 1000-ep evaluation re-run
│   │   └── stage{1,2,3,4_all}.log
│   ├── v2_initial/                # v2: first run (checkpoint-collision bug)
│   │   ├── phase0_ssl.log         #   SSL pretrain (30 epochs)
│   │   ├── v2_{5w1s,5w5s,10w5s}_{no-ssl,ssl}.log    (6 main runs)
│   │   ├── A{1,2,3,4,5}_*.log     #   ablations (most failed)
│   │   └── eval.log               #   evaluation (wrong checkpoints)
│   └── v2_rerun/                  # v2: current best; v2+SSL beat ProtoNet
│       ├── A{1,2,3,4,5}_*.log     #   clean ablations
│       ├── A{1-5}_*_history.json  #   per-ablation val curves
│       ├── v2_*_test_results.json #   meta-test JSON (random-init warn)
│       └── eval.log
│
├── presentation/                  # Slide deck + assets
│   ├── Graph-of-Shots.pptx        # 12-slide deck (final)
│   ├── Graph-of-Shots_Figures.pdf # combined figure PDF
│   ├── Graph-of-Shots_Presentation_Draft.docx  # speaker-note draft
│   ├── figures/                   # Result plots embedded in slides
│   │   ├── fig_v2_main.png        #   main bar chart (v2+SSL wins)
│   │   ├── fig_v2_ablations.png   #   2×3 ablation grid
│   │   ├── fig1_training_dynamics.png  # (legacy, v1)
│   │   ├── fig2_main.png                # (legacy, v1)
│   │   ├── fig3_ablations.png           # (legacy, v1)
│   │   └── fig4_sample_eff.png          # (legacy, v1)
│   └── formulas/                  # Individual formula PNGs (mathtext)
│       └── f{05,07,08,12,13,14,15,25,26,28}_*.png
│
├── docs/                          # This directory
│   ├── README.md                  # docs index
│   ├── repo_structure.md          # THIS FILE
│   ├── results.md                 # numerical results + paths
│   ├── reproduce.md               # reproduction commands
│   └── changelog.md               # key commits, bug fixes
│
├── README.md                      # Top-level onboarding
├── requirements.txt
├── .gitignore                     # checkpoints/ + data_cache/ only
└── .gitattributes                 # LF line endings enforced
```

## Paths ignored by git

Two directories are generated at runtime and excluded from commits:

- `checkpoints/` — `.pt` files, tens of MB each
- `data_cache/` — PyG `InMemoryDataset` cache + `_merged_*.pt` (~22 MB)

Both are rebuilt automatically on the first `python train.py` and cached
thereafter.
