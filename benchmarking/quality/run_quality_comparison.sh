#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv310/bin/python}"
THREADS="${THREADS:-1}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"

"$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
  --output-dir benchmarking/quality/results \
  --benchmark-groups synthetic \
  --methods hdbscan_generic,optimized_hdbscan,score_sg,score_sg_random \
  --runtime-sample-sizes 5000,10000 \
  --runtime-dimensions 32 \
  --runtime-centers 10 \
  --sample-sizes 5000,10000 \
  --dimensions 2,32 \
  --distributions gaussian_sparse \
  --k-max 50 \
  --k-min 2 \
  --seeds 42 \
  --random-state 42 \
  --resume \
  "$@"
