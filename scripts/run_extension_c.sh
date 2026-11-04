#!/usr/bin/env bash
# Extension C end to end (budget allocation, Domain-FT vs. ICL).
#
#   FOCAL=q1 bash scripts/run_extension_c.sh
#
#   python experiments/extension_c_allocation.py evaluate --focal ${FOCAL}
 
set -euo pipefail
FOCAL="${FOCAL:-q1}"
SEED="${SEED:-0}"
DATASET_DIR="${DATASET_DIR:-datasets/soep_${FOCAL}_full}"
WORK="${DATASET_DIR}/extc/seed${SEED}"
 
if [ ! -f "${DATASET_DIR}/data/human_test.csv" ]; then
    echo "ERROR: ${DATASET_DIR}/data/human_test.csv missing -- build the design=full dataset first."
    exit 1
fi
 
python experiments/extension_c_allocation.py prepare --focal "${FOCAL}" --seeds "${SEED}" \
    --dataset-dir "${DATASET_DIR}" "$@"
 
n_lines() { grep -c . "$1"; }
N_FT=$(n_lines "${WORK}/jobs_finetune.txt")
N_SIM_FT=$(n_lines "${WORK}/jobs_simulate_ft.txt")
N_SIM_OTHER=$(n_lines "${WORK}/jobs_simulate_other.txt")
 
export FOCAL SEED DATASET_DIR
FT_JOB=$(sbatch --parsable --array=0-$((N_FT - 1)) scripts/extension_c_finetune.slurm)
echo "fine-tuning:           job ${FT_JOB} (${N_FT} tasks)"
 
OTHER_JOB=$(JOBLIST=other sbatch --parsable --export=ALL,JOBLIST=other \
    --array=0-$((N_SIM_OTHER - 1)) scripts/extension_c_simulate.slurm)
echo "ICL + alpha=0 sims:    job ${OTHER_JOB} (${N_SIM_OTHER} tasks)"
 
SIM_FT_JOB=$(sbatch --parsable --export=ALL,JOBLIST=ft --dependency=afterok:${FT_JOB} \
    --array=0-$((N_SIM_FT - 1)) scripts/extension_c_simulate.slurm)
echo "FT sims (after FT):    job ${SIM_FT_JOB} (${N_SIM_FT} tasks)"
 
echo ""
echo "Watch with:  squeue -u \$USER"
echo "Then:        python experiments/extension_c_allocation.py evaluate --focal ${FOCAL} --dataset-dir ${DATASET_DIR}"