#!/usr/bin/env bash

################################################################################
#
# CoreSG Quality Benchmark - Quick Runner (ARI and HAI to 50k)
#
# Simplified version of run_article_quality_to_50k.sh with predefined settings
# for the standard article quality experiments.
#
# Usage:
#   ./run_quality_comparison_to_50k.sh
#   THREADS=4 ./run_quality_comparison_to_50k.sh
#   SKIP_EXISTING=1 ./run_quality_comparison_to_50k.sh --limit 10
#
# Environment Variables (all optional):
#   PYTHON_BIN       - Path to Python interpreter (default: python3)
#   THREADS          - Number of threads (default: 1)
#   OUTPUT_DIR       - Results output directory (default: benchmarking/quality/results)
#   SKIP_EXISTING    - Skip configurations with existing results (default: 0)
#   LOG_LEVEL        - Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)
#
# Additional Arguments:
#   Pass any arguments after the script name to quality_comparison.py
#   Example: ./run_quality_comparison_to_50k.sh --limit 10 --stop-on-error
#
################################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Thread control
PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${THREADS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarking/quality/results}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Export thread settings to all numerical libraries
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export NUMBA_NUM_THREADS="$THREADS"

echo "================================== CoreSG Quality Benchmark =================================="
echo "Python: $PYTHON_BIN"
echo "Threads: $THREADS"
echo "Output: $OUTPUT_DIR"
echo "Skip existing: $SKIP_EXISTING"
echo "Log level: $LOG_LEVEL"
echo "Extra args: $*"
echo "=============================================================================================="

# Run the quality comparison for the article's configuration:
# - Groups: synthetic distributions matching runtime experiments
# - Methods: all four (HDBSCAN, Optimized HDBSCAN, ScoreSG, ScoreSG Random)
# - Sample sizes: 5k, 10k, 20k, 30k, 40k, 50k (capped at 50k for exact HDBSCAN)
# - Dimensions: 2, 10, 20, 32, 64, 128 (diagnostic)
# - Distributions: all nine (gaussian, poisson, chi_square, gamma, beta, von_mises, gumbel, logistic, gaussian_sparse)
# - k parameters: k_min=2, k_max=50

"$PYTHON_BIN" benchmarking/quality/quality_comparison.py \
  --output-dir "$OUTPUT_DIR" \
  --benchmark-groups synthetic \
  --methods hdbscan_generic,optimized_hdbscan,score_sg,score_sg_random \
  --sample-sizes 5000,10000,20000,30000,40000,50000 \
  --dimensions 2,10,20,32,64,128 \
  --distributions gaussian,poisson,chi_square,gamma,beta,von_mises,gumbel,logistic,gaussian_sparse \
  --k-min 2 \
  --k-max 50 \
  --seeds 42 \
  --random-state 42 \
  --resume \
  --log-level "$LOG_LEVEL" \
  "$@"

echo "=============================================================================================="
echo "✓ Quality benchmark completed"
echo "Results saved to: $OUTPUT_DIR"
echo "Key outputs:"
echo "  - quality_comparison_by_k.csv"
echo "  - quality_comparison_summary.csv"
echo "  - quality_comparison_manifest.json"
echo "=============================================================================================="
