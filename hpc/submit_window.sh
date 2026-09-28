#!/bin/bash
# Launch one LSF job per feature window so the configuration runs proceed in
# parallel instead of serially under a single wall-time. Each window writes to
# its own results/main/perc97/<window>/ directory, so the jobs never collide,
# and PASC_SKIP_EXISTING lets a re-launch resume from the configurations
# already saved.
#
# Prerequisite: the per-window feature matrices exist in cohort_parquets/
# (one extraction pass with database access):
#   PASC_REEXTRACT=1 PASC_EXTRACT_ONLY=1 python run_enhanced_mechsig.py
#
# Usage:
#   hpc/submit_window.sh 0_21                      # one window
#   for W in 0_21 0_30 0_60 0_90 30_60 60_90; do hpc/submit_window.sh $W; done   # all six
#
# Runs whatever ACTIVE_CONFIGS is set to in run_enhanced_mechsig.py (or
# PASC_ACTIVE_CONFIGS from the environment).

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
[[ -f "$HERE/env.local.sh" ]] && source "$HERE/env.local.sh"

W="${1:?usage: hpc/submit_window.sh <window>  e.g. 0_21}"
mkdir -p "$REPO_DIR/logs"

bsub -P "$LSF_ALLOCATION" \
     -J "mechsig_${W}" \
     -o "$REPO_DIR/logs/mechsig_${W}_%J.out" \
     -e "$REPO_DIR/logs/mechsig_${W}_%J.err" \
     -W 48:00 -M 64000 -n 16 -q "$LSF_QUEUE" \
     -R "span[hosts=1]" -R "rusage[mem=64000]" \
     "source ~/.bashrc; conda activate $CONDA_ENV; \
      export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
             ANTONY_N_JOBS=15 PASC_REEXTRACT=0 PASC_SKIP_EXISTING=1 PASC_FAST_MODE=0 \
             PASC_FEATURE_WINDOWS=${W} ${PASC_ACTIVE_CONFIGS:+PASC_ACTIVE_CONFIGS=$PASC_ACTIVE_CONFIGS}; \
      cd $REPO_DIR; \
      python run_enhanced_mechsig.py"
