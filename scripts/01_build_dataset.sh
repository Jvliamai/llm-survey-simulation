#!/usr/bin/env bash
# Build the analysis datasets for both focal questions.
#
# This is the only step that touches raw SOEP files.
# bash scripts/01_build_dataset.sh /path/to/soep/v41 [mirror|full]

set -euo pipefail

SOEP_DIR="${1:?Usage: $0 <soep-dir> [design]}"
DESIGN="${2:-mirror}"

for q in q1 q2; do
    echo "=== Building soep_${q} (design=${DESIGN}) ==="
    python -m soepsim.preprocessing.build_dataset \
        --soep-dir "${SOEP_DIR}" \
        --focal "${q}" \
        --out-dir "datasets/soep_${q}" \
        --design "${DESIGN}" \
        --language de \
        --history-variant full
done

echo ""
echo "Check theta* and the sample flow before spending GPU time:"
echo "  cat datasets/soep_q1/outputs/results/preprocessing_report.json"
