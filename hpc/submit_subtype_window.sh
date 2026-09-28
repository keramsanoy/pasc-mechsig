#!/bin/bash
# Launch one LSF job per (feature window, symptom group) for the secondary
# analysis S1 (scripts/run_symptom_groups.py). Each group writes to its own
# results/symptom_groups/perc97/<window>/<config>/iterations_<group>.csv, so
# jobs never collide, and PASC_SKIP_EXISTING lets a re-launch resume.
#
# Prerequisite: one extraction pass with database access to populate the
# per-window feature caches and subtype_labels.parquet (no modelling). Pin BLAS
# threads to 1 on a login node, or OpenBLAS spawns one thread per core:
#
#   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ANTONY_N_JOBS=4
#   PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 PASC_FEATURE_WINDOWS=0_90 \
#     python scripts/run_symptom_groups.py
#
# Then submit one modelling job per group (reuses the caches, no database):
#   hpc/submit_subtype_window.sh 0_90 cognitive              # one group
#   for S in cardiorespiratory fatigue anxiety_depression gastrointestinal cognitive; do \
#       hpc/submit_subtype_window.sh 0_90 $S; done           # all five, in parallel
#
# Omit the group argument to run all groups in a single job:
#   hpc/submit_subtype_window.sh 0_90

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
[[ -f "$HERE/env.local.sh" ]] && source "$HERE/env.local.sh"

W="${1:-0_90}"
S="${2:-}"
mkdir -p "$REPO_DIR/logs"

if [[ -n "$S" ]]; then
    JOB="subtype_${W}_${S}"
    SUBTYPE_ENV="PASC_SUBTYPES=${S}"
else
    JOB="subtype_${W}"
    SUBTYPE_ENV=""
fi

bsub -P "$LSF_ALLOCATION" \
     -J "$JOB" \
     -o "$REPO_DIR/logs/${JOB}_%J.out" \
     -e "$REPO_DIR/logs/${JOB}_%J.err" \
     -W 24:00 -M 64000 -n 16 -q "$LSF_QUEUE" \
     -R "span[hosts=1]" -R "rusage[mem=64000]" \
     "source ~/.bashrc; conda activate $CONDA_ENV; \
      export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
             ANTONY_N_JOBS=15 PASC_REEXTRACT=0 PASC_SKIP_EXISTING=1 \
             PASC_FEATURE_WINDOWS=${W} ${SUBTYPE_ENV}; \
      cd $REPO_DIR; \
      python scripts/run_symptom_groups.py"
