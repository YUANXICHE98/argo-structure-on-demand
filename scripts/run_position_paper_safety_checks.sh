#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEVICE="${DEVICE:-cpu}"
MAX_BATCHES="${MAX_BATCHES:-5}"
FULL_CKPT="${FULL_CKPT:-results/weekly_projection_space_routing_seed42/projection_space_routing_best.pt}"
NOORTHO_CKPT="${NOORTHO_CKPT:-results/weekly_projection_space_routing_noortho_seed42/projection_space_routing_best.pt}"
mkdir -p results/position_paper_safety
if [[ -f "$FULL_CKPT" ]]; then
  python scripts/diagnose_projection_space_structure.py --checkpoint "$FULL_CKPT" --device "$DEVICE" --max-batches "$MAX_BATCHES" --output-json results/position_paper_safety/full_structure_diagnosis.json --output-md results/position_paper_safety/full_structure_diagnosis.md
  python scripts/evaluate_projection_space_routing_ablation.py --checkpoint "$FULL_CKPT" --device "$DEVICE" --output-json results/position_paper_safety/full_router_interventions.json --output-md results/position_paper_safety/full_router_interventions.md
else
  echo "[skip] missing FULL_CKPT=$FULL_CKPT" >&2
fi
if [[ -f "$NOORTHO_CKPT" ]]; then
  python scripts/diagnose_projection_space_structure.py --checkpoint "$NOORTHO_CKPT" --device "$DEVICE" --max-batches "$MAX_BATCHES" --output-json results/position_paper_safety/noortho_structure_diagnosis.json --output-md results/position_paper_safety/noortho_structure_diagnosis.md
else
  echo "[skip] missing NOORTHO_CKPT=$NOORTHO_CKPT" >&2
fi
if [[ -f results/argo_traffic_fast/results.json ]]; then
  cp results/argo_traffic_fast/results.json results/position_paper_safety/traffic_strong_prior_boundary.json
fi
echo "Done. Outputs are in results/position_paper_safety/."
