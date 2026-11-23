#!/bin/bash
# Submit a v5 chain with a fixed number of 2-day jobs per stage of the first
# phase (each job resumes the one before it, afterany):
#   scripts/rft/submit_juno_stages.sh <schedule json> <warmup jobs> <rft jobs> <gdpo jobs>
# For the h200 nodes (2 GPUs each):
#   SBATCH_ARGS="-p h200 --gres=gpu:2" scripts/rft/submit_juno_stages.sh \
#       scripts/rft/configs/schedule_v5_juno_h200.json 1 2 3
# Prints "<job id> <stage>" per job. Cancel the chain with scancel on every id.
set -euo pipefail
SCHEDULE="${1:?schedule json}"
cd "$(dirname "$0")/../.."
previous=""
for stage_count in "warmup ${2:?warmup jobs}" "rft ${3:?rft jobs}" "gdpo ${4:?gdpo jobs}"; do
    set -- $stage_count
    for _ in $(seq 1 "$2"); do
        dependency=${previous:+--dependency=afterany:$previous}
        previous=$(sbatch --parsable ${SBATCH_ARGS:-} $dependency scripts/rft/juno_schedule.sbatch "$SCHEDULE" "$1")
        echo "$previous $1"
    done
done
