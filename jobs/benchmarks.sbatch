#!/usr/bin/env bash
# Comparable comparator benchmark: scNODE, scDiffEq and ARTEMIS, three fits each.
# Each task trains with gene-Sinkhorn checkpoint selection on the internal holdout
# donors (the same donors and rule as scCody), then reports on the 71
# external donors. Profiles: configs/benchmarks/<method>_fit<k>_{train,report}.args
#
#   mkdir -p artifacts/comparators_final/logs
#   sbatch jobs/benchmarks.sbatch             # all 9 tasks
#   sbatch --array=0,3,6 jobs/benchmarks.sbatch   # fit 0 of each method
#
# Task order: 0-2 scNODE fits 0-2, 3-5 scDiffEq, 6-8 ARTEMIS. Finished steps are skipped.

#SBATCH --job-name=benchmarks
#SBATCH --partition=long
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:long:1
#SBATCH --array=0-8
#SBATCH --output=artifacts/comparators_final/logs/slurm-%A_%a.out
#SBATCH --error=artifacts/comparators_final/logs/slurm-%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON="${PYTHON:-python}"
ARTEMIS_PYTHON="${ARTEMIS_PYTHON:-python}"
ARTEMIS_SRC="${ARTEMIS_SRC:-$REPO_DIR/sccody_analyses/benchmarks/vendor/artemis}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Submit this script as a Slurm array}"
METHODS=(scnode scnode scnode scdiffeq scdiffeq scdiffeq artemis artemis artemis)
method="${METHODS[$TASK_ID]:?Array task must be 0-8}"
fit=$((TASK_ID % 3))
if [[ "$method" == "artemis" ]]; then
  PYTHON="$ARTEMIS_PYTHON"
fi
root="artifacts/comparators_final/$method/fold_$fit"
checkpoint="$root/checkpoints/${method}_fit${fit}_${method}.pt"
report_marker="$root/report/${method}_fit${fit}_${method}_calibration_projection.csv"

cd "$REPO_DIR"
mkdir -p "$root" artifacts/comparators_final/logs
for required in data/aida_age_holdout_train.h5ad data/aida_counts_hvg.h5ad \
                configs/benchmarks/${method}_fit${fit}_train.args configs/benchmarks/${method}_fit${fit}_report.args \
                configs/donors/internal_${fit}.json configs/donors/external_${fit}.json \
                configs/benchmarks/metadata/fit${fit}_internal_fixed_sources_bidirectional.json \
                configs/benchmarks/metadata/fit${fit}_internal_fixed_source_age19_forward.json \
                configs/benchmarks/metadata/fit${fit}_fixed_sources_bidirectional.json \
                configs/benchmarks/metadata/fit${fit}_fixed_source_age19_forward.json configs/datasets/normal_pbmc.yaml; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 2; }
done

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
threads=$(( ${SLURM_CPUS_PER_TASK:-16} / 2 ))
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads NUMEXPR_NUM_THREADS=$threads

run() {
  local step="$1"; shift
  local cmd=("$PYTHON" -u -m sccody_analyses.benchmarks.run "$method" "@configs/benchmarks/${method}_fit${fit}_${step}.args")
  if [[ "$method" == "artemis" ]]; then
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    cmd+=(--artemis-src "$ARTEMIS_SRC")
  fi
  echo "[launch] method=$method fit=$fit step=$step"
  # Keep the final selection in stdout; exceptions and warnings remain on stderr.
  "${cmd[@]}" | awk '/^\[selected\]/ { print; fflush() }'
}

if [[ -s "$checkpoint" ]]; then
  echo "[skip] train: $checkpoint exists"
else
  run train
fi

if [[ -s "$report_marker" ]]; then
  echo "[skip] report: $report_marker exists"
else
  run report
fi
dc_marker="$root/report_dc/${method}_fit${fit}_${method}_calibration_projection.csv"
if [[ ! -s "$dc_marker" ]]; then
  run dc_report
fi
echo "[done] method=$method fit=$fit"
