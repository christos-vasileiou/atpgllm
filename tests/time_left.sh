#!/usr/bin/env bash

# Usage: ./time_left.sh <EndTime> [PrevJobRuntime]
# Example: ./time_left.sh 2026-08-11T08:00:00 00-12:30:00

EndTime="${1:-}"
PrevJobRuntime="${2:-02-00:00:00}"

# 1. Prompt for EndTime if missing
if [ -z "$EndTime" ]; then
    read -rp "Enter End Time (e.g. 2026-08-11T08:00:00): " EndTime
fi

# 2. Parse EndTime into Epoch Seconds (Cross-platform Linux/macOS)
if date --version >/dev/null 2>&1; then
    end_sec=$(date -d "$EndTime" +%s 2>/dev/null)
else
    end_sec=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$EndTime" +%s 2>/dev/null)
fi

if [ -z "$end_sec" ]; then
    echo "Error: Invalid EndTime format. Use YYYY-MM-DDTHH:MM:SS" >&2
    exit 1
fi

# 3. Parse PrevJobRuntime (dd-hh:mm:ss) into Seconds
if [[ "$PrevJobRuntime" =~ ^([0-9]+)-([0-9]{2}):([0-9]{2}):([0-9]{2})$ ]]; then
    pj_d="${BASH_REMATCH[1]}"
    pj_h="${BASH_REMATCH[2]}"
    pj_m="${BASH_REMATCH[3]}"
    pj_s="${BASH_REMATCH[4]}"
    
    prev_job_sec=$(( (pj_d * 86400) + (10#$pj_h * 3600) + (10#$pj_m * 60) + 10#$pj_s ))
else
    echo "Error: Invalid PrevJobRuntime format. Use dd-hh:mm:ss (e.g. 01-04:30:00)" >&2
    exit 1
fi

# 4. Calculate Constants & Math
# Target job duration: 2-00:00:00 = 2 days = 172800 seconds
MAX_JOB_SEC=$((2 * 86400))

# Time remaining for the active job to hit 2 days
job_remaining_sec=$((MAX_JOB_SEC - prev_job_sec))

# Clamp job_remaining_sec to 0 if the job has already reached/passed 2 days
if [ "$job_remaining_sec" -lt 0 ]; then
    job_remaining_sec=0
fi

now_sec=$(date +%s)

# Formula: remaining_time = EndTime - now - (2-00:00:00 - PrevJobRuntime)
diff_sec=$((end_sec - now_sec - job_remaining_sec))

if [ "$diff_sec" -le 0 ]; then
    echo "00-00:00:00 (Time has already passed)"
    exit 0
fi

# 5. Format Output back to dd-hh:mm:ss
days=$((diff_sec / 86400))
hours=$(((diff_sec % 86400) / 3600))
mins=$(((diff_sec % 3600) / 60))
secs=$((diff_sec % 60))

printf "%02d-%02d:%02d:%02d\n" "$days" "$hours" "$mins" "$secs"