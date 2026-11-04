#!/usr/bin/env bash
# End-to-end Extension A, with the cluster dependencies wired up.
#
#   bash scripts/run_all_extension_a.sh /path/to/soep/v41

set -euo pipefail
SOEP_DIR="${1:?Usage: $0 <soep-dir>}"

echo "=== 1/5  Building datasets (local) ==="
bash scripts/01_build_dataset.sh "${SOEP_DIR}" mirror

echo ""
echo "theta* for both questions:"
python - <<'PY'
import json, pathlib
for q in ("q1", "q2"):
    p = pathlib.Path(f"datasets/soep_{q}/outputs/results/preprocessing_report.json")
    if p.exists():
        r = json.loads(p.read_text())
        print(f"  {q}: theta*={r['theta_star']:.4f}  skew={r['skewness']:+.3f}  "
              f"n={r['n_final']}  (ATP: {r['atp_theta_star']})")
PY

echo ""
read -r -p "Distributions look right? Continue to GPU stages? [y/N] " ok
[ "${ok}" = "y" ] || { echo "Stopped."; exit 0; }

for FOCAL in q1 q2; do
    echo "=== ${FOCAL}: submitting persona, fine-tune and simulation jobs ==="
    P=$(FOCAL=$FOCAL sbatch --parsable scripts/02_personas.slurm)
    F=$(FOCAL=$FOCAL sbatch --parsable scripts/04_finetune.slurm)
    FOCAL=$FOCAL sbatch --array=0,1,2,4,5,6,8,9,10 --dependency=afterok:"${P}":"${F}" scripts/03_simulate.slurm
    echo "  personas=${P}  finetune=${F}"
done

echo ""
echo "Submitted. Watch with:  squeue -u $USER -l"
echo "When everything is COMPLETED, run:  bash scripts/05_evaluate.sh"
