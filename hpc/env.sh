#!/bin/bash
# Site-specific settings for the LSF submit scripts. Copy to hpc/env.local.sh
# (git-ignored) and edit; the submit scripts source it when present.
#
# The thesis ran on an LSF cluster with a 16-core, 64 GB compute-node
# allocation per job; the values below are those settings with the
# account-specific names removed.
export LSF_ALLOCATION="${LSF_ALLOCATION:-<your-lsf-project>}"   # bsub -P
export LSF_QUEUE="${LSF_QUEUE:-<your-queue>}"                    # bsub -q
export CONDA_ENV="${CONDA_ENV:-pasc}"                            # conda environment with requirements.txt installed
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
