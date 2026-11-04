# Replicating and Extending LLM-Based Survey Simulation in a German-Language Context

Replication of Krsteski et al. (2026) on SOEP. Synthetic responses from three open-weight LLMs are combined with a small human sample via rectification to estimate population means, extended by stratified rectification for subgroups and a fixed-budget comparison of fine-tuning, in-context learning and rectification.

## Data protection

The SOEP data are **not** included in this repository. They are subject to a data use agreement
with DIW Berlin and can be requested from the Research Data Center of the SOEP
(SOEP-Core v41, EU Edition, [doi:10.5684/soep.core.v41eu](https://doi.org/10.5684/soep.core.v41eu)).

All person-level files derived from them (datasets, personas, predictions, fine-tuning data,
LoRA adapters) are excluded via `.gitignore`; only aggregated results are part of the
repository. All models run locally, and no SOEP data are sent to external APIs.

## Requirements

- Access to the SOEP-Core v41 (EU Edition) data
- A SLURM cluster with GPUs for steps 2 and 3 (the experiments were run on single NVIDIA H100
  GPUs of the LRZ AI Systems)
- A Hugging Face account with access to `meta-llama/Llama-3.1-8B-Instruct` (gated model);
  Ministral 8B and Qwen3 8B are openly available
- Python 3.11 and conda

## Overview of the pipeline

| Step | What it does | Where | Needed for |
|---|---|---|---|
| 0 | Set up the environment | local and cluster | all |
| 1 | Build the analysis datasets from the raw SOEP files | local or cluster (CPU) | all |
| 2 | Generate personas, fine-tune and synthesise responses | cluster (GPU) | RQ1, RQ2 |
| 3 | Fine-tune and synthesise for each adaptation share | cluster (GPU) | RQ3 |
| 4 | Evaluate and write all tables and figures | cluster or local (CPU) | all |

With existing predictions, steps 1 to 3 can be skipped (see [Step 4](#step-4-evaluation)).

All commands are run from the repository root.

## Step 0: Setup

**On the cluster** (LRZ AI Systems, SLURM):

```bash
git clone https://github.com/Jvliamai/llm-survey-simulation.git
cd llm-survey-simulation
bash scripts/lrz_setup.sh          # creates the conda environment "soepsim"
conda activate soepsim
huggingface-cli login              # access token for the gated Llama 3.1 weights
```

Model weights are downloaded to `$HOME/hf_cache` by default. To use a different location,
set it before submitting jobs:

```bash
export HF_HOME=/path/to/hf_cache
```

**Locally**, only for step 1 (no GPU needed):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install pandas numpy typer pyreadstat pyarrow
pip install -e . --no-deps
```

## Step 1: Build the datasets

```bash
bash scripts/01_build_dataset.sh /path/to/soep/v41 mirror   # RQ1/RQ2: datasets/soep_q1, datasets/soep_q2
bash scripts/01_build_dataset.sh /path/to/soep/v41 full     # RQ3:     datasets/soep_q1_full
```

On Windows, or without bash, call the module directly:

```bash
python -m soepsim.preprocessing.build_dataset --soep-dir /path/to/soep/v41 --focal q1 --design mirror --out-dir datasets/soep_q1
python -m soepsim.preprocessing.build_dataset --soep-dir /path/to/soep/v41 --focal q2 --design mirror --out-dir datasets/soep_q2
python -m soepsim.preprocessing.build_dataset --soep-dir /path/to/soep/v41 --focal q1 --design full   --out-dir datasets/soep_q1_full
```

`mirror` draws random samples at the size of the ATP samples in the original study
(691 respondents for Q1, 643 for Q2). `full` applies the same filters without subsampling
(4,897 respondents for Q1).

If step 1 runs locally, copy the `data/` folders to the cluster afterwards, e.g.

```bash
scp -r datasets/soep_q1/data <user>@login.ai.lrz.de:~/llm-survey-simulation/datasets/soep_q1/
```
and likewise for `soep_q2` and `soep_q1_full`.

## Step 2: Synthetic responses for RQ1 and RQ2

Submit the jobs on the cluster:

```bash
for FOCAL in q1 q2; do
    P=$(FOCAL=$FOCAL sbatch --parsable scripts/02_personas.slurm)
    F=$(FOCAL=$FOCAL sbatch --parsable scripts/04_finetune.slurm)
    FOCAL=$FOCAL sbatch --array=0,1,2,4,5,6,8,9,10 --dependency=afterok:$P:$F scripts/03_simulate.slurm
done
```

This generates the personas, trains one LoRA adapter per model for Domain-FT,
and synthesises responses for all combinations of model and method. The model and generation
settings are in `datasets/soep_q*/configs/`.

## Step 3: Synthetic responses for RQ3

```bash
FOCAL=q1 bash scripts/run_extension_c.sh
```

This prepares the adaptation sets and submits the fine-tuning and simulation jobs for
Domain-FT and ICL. 

## Step 4: Evaluation

Once all jobs have finished:

```bash
bash scripts/05_evaluate.sh
```

If the simulated responses are already available, steps 1 to 3 can be skipped. The evaluation
only needs the following folders to be in place:

```
datasets/soep_q1/data/                        datasets/soep_q1/outputs/simulated_data/
datasets/soep_q2/data/                        datasets/soep_q2/outputs/simulated_data/
datasets/soep_q1_full/extc/seed0/data/        datasets/soep_q1_full/extc/seed0/predictions/
datasets/soep_q1_full/extc/seed0/split.json
```

## Repository structure

```
datasets/
  soep_q1/, soep_q2/     configs/ (model and method settings), data/ and outputs/ (not tracked)
  soep_q1_full/          dataset for RQ3; extc/ holds the split, configs and predictions (not tracked)
experiments/             evaluation scripts for RQ1-RQ3 and the predictability analysis
outputs/tables/          aggregated results used in the thesis
scripts/                 setup, dataset construction and SLURM jobs
src/soepsim/
  preprocessing/         loading, recoding and dataset construction
  methods/               synthesis (Hugging Face inference), personas, random baseline
  prompts.py, icl.py     prompt construction and example retrieval for ICL
  ppi.py, stratified.py  global and stratified rectification
  metrics.py, evaluate.py
```

## Reference

Krsteski, S., Russo, G., Chang, S., West, R. and Gligorić, K. (2026). Valid survey simulations
with limited human data: The roles of prompting, fine-tuning, and rectification.
*Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*,
pp. 10887–10906.
