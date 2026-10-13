#!/bin/bash
# Copy what a GRPO run needs to a cluster with its own filesystem (e.g. juno),
# in the sibling layout the launcher expects:
#
#   <root>/libatpgllm            git clone of this checkout's HEAD (via a bundle)
#   <root>/data_preprocessing    git clone of the paired dp HEAD, plus its
#                                git-ignored sim_config.json and ASAP7 cell Verilog
#   <root>/data/freeset/<v4>     the dataset
#   <root>/.runtime/tetramax     the TetraMAX seat/lock directory (empty)
#   <root>/libatpgllm/runs/<sft run>/best   the SFT checkpoint GRPO starts from
#
# Usage (from any directory):
#   scripts/train/stage_to_cluster.sh <user>@<host>:<root> [<sft checkpoint dir>]
#   e.g. scripts/train/stage_to_cluster.sh me@juno-vscode.hpcre.utdallas.edu:/scratch/me/atpg \
#            runs/sft_granite_4.2_8b_v4/checkpoint-120
#
# Commit first: only committed code is staged (the bundles carry HEAD), so the
# run's code_provenance.jsonl names real commits. Re-running updates the clones
# in place (fetch + checkout) and rsyncs only what changed.
# The Python environment on the cluster is not created here; install
# libatpgllm/requirements.txt (see uv.lock) and `pip install -e libatpgllm` there.
set -euo pipefail

DEST="${1:?usage: $0 <user@host:root> [sft checkpoint dir]}"
CHECKPOINT="${2:-}"
HOST="${DEST%%:*}"
ROOT="${DEST#*:}"
[ "$HOST" != "$DEST" ] && [ -n "$ROOT" ] || { echo "ERROR: destination must be <user@host>:<root>"; exit 1; }

LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$(cd "$LIB/.." && pwd)"
DP="${DATA_PREPROCESSING_ROOT:-$PROJECT/data_preprocessing}"
DATASET="${DATASET:-$PROJECT/data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v4}"
SFT_RUN="${SFT_RUN:-sft_granite_4.2_8b_v4}"

for repo in "$LIB" "$DP"; do
    if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
        echo "ERROR: $repo has uncommitted changes; commit them so the cluster runs a known commit."
        exit 1
    fi
done
[ -d "$DATASET/train" ] || { echo "ERROR: dataset not found: $DATASET"; exit 1; }
if [ -n "$CHECKPOINT" ]; then
    case "$CHECKPOINT" in /*) ;; *) CHECKPOINT="$LIB/$CHECKPOINT" ;; esac
    [ -f "$CHECKPOINT/adapter_config.json" ] || { echo "ERROR: no adapter_config.json in $CHECKPOINT"; exit 1; }
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
stage_repo() {  # <local repo> <remote name>
    local repo="$1" name="$2" branch
    branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD)"
    git -C "$repo" bundle create "$WORK/$name.bundle" HEAD >/dev/null 2>&1
    rsync -a "$WORK/$name.bundle" "$HOST:$ROOT/.stage/"
    ssh "$HOST" bash -s -- "$ROOT" "$name" "$branch" <<'REMOTE'
set -euo pipefail
root="$1"; name="$2"; branch="$3"
if [ -d "$root/$name/.git" ]; then
    git -C "$root/$name" fetch -q "$root/.stage/$name.bundle" HEAD
    git -C "$root/$name" checkout -q -B "$branch" FETCH_HEAD
else
    git -c advice.detachedHead=false clone -q "$root/.stage/$name.bundle" "$root/$name"
    git -C "$root/$name" checkout -q -B "$branch" FETCH_HEAD 2>/dev/null || git -C "$root/$name" checkout -q -B "$branch"
fi
echo "  $name: $(git -C "$root/$name" log --oneline -1) on $branch"
REMOTE
}

echo "Staging to $HOST:$ROOT"
ssh "$HOST" "mkdir -p '$ROOT/.stage' '$ROOT/data/freeset' '$ROOT/.runtime/tetramax'"
stage_repo "$LIB" libatpgllm
stage_repo "$DP" data_preprocessing

echo "Copying data_preprocessing's git-ignored runtime files (sim_config.json, ASAP7 cell Verilog)"
rsync -a "$DP/sim_config.json" "$HOST:$ROOT/data_preprocessing/"
rsync -aL --relative "$DP/./lib/asap7sc7p5t_28/verilog/" "$HOST:$ROOT/data_preprocessing/"

echo "Copying the dataset $(basename "$DATASET") ($(du -sh "$DATASET" | cut -f1))"
rsync -a --delete "$DATASET/" "$HOST:$ROOT/data/freeset/$(basename "$DATASET")/"

if [ -n "$CHECKPOINT" ]; then
    echo "Copying SFT checkpoint $CHECKPOINT to runs/$SFT_RUN/best (no optimizer state)"
    ssh "$HOST" "mkdir -p '$ROOT/libatpgllm/runs/$SFT_RUN'"
    rsync -a --delete --exclude 'optimizer.pt' --exclude 'global_step*' \
        "$CHECKPOINT/" "$HOST:$ROOT/libatpgllm/runs/$SFT_RUN/best/"
    manifest="$(dirname "$CHECKPOINT")/sft_data_manifest.json"
    [ -f "$manifest" ] && rsync -a "$manifest" "$HOST:$ROOT/libatpgllm/runs/$SFT_RUN/"
fi

cat <<EOF

Staged. On the cluster:
  cd $ROOT/libatpgllm
  # once: create the Python environment (requirements.txt / uv.lock) and pip install -e .
  ./scripts/train/submit_training_code.sh scripts/train/configs/grpo_granite_4.2_8b_v4_h100_juno_tetramax.conf
The launcher runs a TetraMAX preflight simulation before any GPU work and stops
with a message if TetraMAX or its license is unavailable on the compute node.
EOF
