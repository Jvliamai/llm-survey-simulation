#!/usr/bin/env bash
# Evaluate all research questions and write the tables to outputs/tables/.
#
#   bash scripts/05_evaluate.sh [n_trials]

set -euo pipefail
N_TRIALS="${1:-100}"
OUT=outputs/tables
mkdir -p "${OUT}/rq1" "${OUT}/rq2" "${OUT}/rq3" "${OUT}/discussion"

echo "=== RQ1: replication ==="
python experiments/extension_a_replication.py --n-trials "${N_TRIALS}" --out-dir "${OUT}/rq1"

echo ""
echo "=== RQ2: subgroup-aware rectification ==="
python experiments/extension_b_stratified.py --n-trials "${N_TRIALS}" --out-dir "${OUT}/rq2"
python experiments/extension_b_rmse.py --n-trials "${N_TRIALS}" --out-dir "${OUT}/rq2"

echo ""
echo "=== RQ3: budget allocation (Domain-FT vs. ICL) ==="
python experiments/extension_c_allocation.py evaluate --focal q1 --n-trials "${N_TRIALS}" --out-dir "${OUT}/rq3"
python experiments/extension_c_figure.py --focal q1 --out-dir "${OUT}/rq3"
python experiments/extension_c_correlation.py

echo ""
echo "=== Discussion: predictability and correlations ==="
python experiments/predictability.py
python experiments/llm_correlation.py