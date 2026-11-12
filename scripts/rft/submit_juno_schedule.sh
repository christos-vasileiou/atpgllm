#!/bin/bash
# Submit a chain of v5 schedule jobs on juno (each resumes the one before).
#   scripts/rft/submit_juno_schedule.sh <jobs in the chain> [schedule json]
# Prints the job ids. Cancel the chain with scancel on every id.
set -euo pipefail
N="${1:?number of chained 2-day jobs}"
SCHEDULE="${2:-scripts/rft/configs/schedule_v5_juno_h100.json}"
cd "$(dirname "$0")/../.."
previous=""
for _ in $(seq 1 "$N"); do
    if [ -z "$previous" ]; then
        previous=$(sbatch --parsable scripts/rft/juno_schedule.sbatch "$SCHEDULE")
    else
        previous=$(sbatch --parsable --dependency=afterany:"$previous" scripts/rft/juno_schedule.sbatch "$SCHEDULE")
    fi
    echo "$previous"
done
