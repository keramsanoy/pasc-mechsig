#!/bin/bash
# Single LSF job that runs every window and configuration serially.
# For the per-window parallel launch used for the thesis see submit_window.sh.
#
#   bsub < hpc/submit_enhanced_mechsig.sh
#
# The #BSUB header cannot read variables, so fill in the allocation and queue
# before submitting (or use submit_window.sh, which reads hpc/env.local.sh).
#BSUB -P <your-lsf-project>
#BSUB -q <your-queue>
#BSUB -J enhanced_mechsig
#BSUB -o logs/enhanced_mechsig_%J.out
#BSUB -e logs/enhanced_mechsig_%J.err
#BSUB -W 24:00
#BSUB -M 64000
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64000]"

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
[[ -f "$HERE/env.local.sh" ]] && source "$HERE/env.local.sh"

mkdir -p "$REPO_DIR/logs"
source ~/.bashrc
conda activate "$CONDA_ENV"

echo "Job started at: $(date)  (job $LSB_JOBID on $HOSTNAME)"
cd "$REPO_DIR"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ANTONY_N_JOBS=15
python scripts/run_main_analysis.py
echo "Job completed at: $(date)"
