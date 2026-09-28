import os

# Keep the forests small and deterministic in tests.
os.environ.setdefault("ANTONY_N_JOBS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
