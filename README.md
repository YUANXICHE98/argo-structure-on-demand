# ARGO: Structure on Demand

Reference implementation for the ARGO position-paper experiments on structure-level routing in spatiotemporal prediction.

## Core claim

Routing is only as meaningful as the structural alternatives it is asked to choose among. This code focuses on diagnosing whether multi-branch graph/hypergraph models expose differentiated candidate structures before learned routing is evaluated.

## What is included

- `src/models/projection_space_routing_hypergraph.py`: projection-space structural routing model for weekly CAMELS experiments.
- `scripts/run_weekly_projection_space_routing.py`: training entrypoint for the CAMELS projection-space model.
- `scripts/evaluate_projection_space_routing_ablation.py`: post-hoc router interventions, including learned, uniform, shuffled, fixed-space, routed-only, and static-only variants.
- `scripts/diagnose_projection_space_structure.py`: structural differentiation diagnostics for candidate hypergraphs.
- `scripts/run_argo_traffic_fast.py`: compact METR-LA strong-prior boundary experiment.
- `scripts/run_position_paper_safety_checks.sh`: convenience script for the main reproducibility checks.

## What is not included

- Raw CAMELS-US or METR-LA data.
- Large checkpoints or generated result folders.
- Paper drafts, internal notes, exploratory experiments, or archived failed branches.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch installation may vary by platform. If you use CUDA or Apple MPS, install the backend-specific PyTorch build recommended by the official PyTorch website.

## Data

Place datasets under the layout described in `data/README.md`.

The code expects preprocessed CAMELS-US files compatible with `src/data/camels_loader.py` and METR-LA arrays at:

```text
data/traffic_datasets/real/metr_la_data.npy
data/traffic_datasets/real/metr_la_adj.npy
```

## Training CAMELS ARGO

Example single run:

```bash
python scripts/run_weekly_projection_space_routing.py \
  --device cpu \
  --seed 42 \
  --epochs 30 \
  --batch-size 8 \
  --result-dir results/weekly_projection_space_routing_seed42 \
  --use-orthogonal-init \
  --nce-loss-weight 0.1
```

Useful ablation variants:

```bash
# No orthogonal initialization / no NCE auxiliary term
python scripts/run_weekly_projection_space_routing.py \
  --device cpu \
  --seed 42 \
  --epochs 30 \
  --batch-size 8 \
  --result-dir results/weekly_projection_space_routing_noortho_seed42

# Disable sparse top-k routing by using all three spaces
python scripts/run_weekly_projection_space_routing.py \
  --device cpu \
  --seed 42 \
  --epochs 30 \
  --batch-size 8 \
  --top-k 3 \
  --result-dir results/weekly_projection_space_routing_topk3_seed42 \
  --use-orthogonal-init
```

## Post-hoc router interventions

```bash
python scripts/evaluate_projection_space_routing_ablation.py \
  --device cpu \
  --checkpoint results/weekly_projection_space_routing_seed42/projection_space_routing_best.pt \
  --output-json results/weekly_projection_space_routing_seed42/router_interventions.json \
  --output-md results/weekly_projection_space_routing_seed42/router_interventions.md
```

This evaluates learned routing against uniform, shuffled, fixed-space, routed-only, uniform-routed-only, and static-only interventions.

## Structure differentiation diagnostics

```bash
python scripts/diagnose_projection_space_structure.py \
  --device cpu \
  --checkpoint results/weekly_projection_space_routing_seed42/projection_space_routing_best.pt \
  --output-json results/weekly_projection_space_routing_seed42/structure_diagnosis.json \
  --output-md results/weekly_projection_space_routing_seed42/structure_diagnosis.md
```

For a quick smoke test, add `--max-batches 1`.

## Traffic strong-prior boundary experiment

```bash
python scripts/run_argo_traffic_fast.py
```

This experiment is included as a strong-prior boundary check: when road topology is available, the learned structure component may be supplementary rather than necessary.

## Reproducibility notes

Exact numeric reproduction is not guaranteed across hardware, PyTorch versions, BLAS backends, and dataset preprocessing variants. The intended reproducibility target is pattern-level agreement:

- router interventions should expose whether learned routing is materially different from uniform or fixed alternatives;
- structure diagnostics should report whether candidate hypergraphs are differentiated;
- traffic strong-prior runs should report the learned prior gate and show how strongly the model relies on the fixed road prior.

For paper-style results, run at least three seeds and report mean/std. If your numbers differ from a table in the paper, inspect the generated `results.json`, command-line config, hardware backend, and data split before assuming a code error.

## Safety checks before reporting results

If checkpoints exist, run:

```bash
bash scripts/run_position_paper_safety_checks.sh
```

By default this script runs lightweight diagnostics. Set environment variables as needed:

```bash
DEVICE=cpu MAX_BATCHES=10 \
FULL_CKPT=results/weekly_projection_space_routing_seed42/projection_space_routing_best.pt \
NOORTHO_CKPT=results/weekly_projection_space_routing_noortho_seed42/projection_space_routing_best.pt \
bash scripts/run_position_paper_safety_checks.sh
```

## Citation

If you use this code, please cite the accompanying ARGO / Structure on Demand paper when available.
