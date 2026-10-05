#!/bin/sh
# Sequential experiment queue. Runs ONE job at a time on purpose: two
# concurrent runs exhausted this 7 GB box and the server killed both mid-run
# (AGENTS.md does not cover this; it cost three restarts).
#
# Each line is: <log-tag> <args...>. A line beginning with # is a comment.
# Usage: setsid scripts/run_queue.sh <queuefile> <donefile>
#
# Resumable: a job whose run directory already holds final_validation.json is
# skipped. The server restarted repeatedly during a sweep and each time the
# queue restarted from the first job, so completed runs were repeated. Skip on
# the training run's own completion marker, which train.py writes only at the
# end.
set -u
cd /home/prash/college/minor/marl-fleet-rebalancing || exit 1
QUEUE="$1"
DONE="$2"

while IFS= read -r line; do
    case "$line" in
        ''|\#*) continue ;;
    esac
    tag=$(printf '%s\n' "$line" | awk '{print $1}')
    rest=$(printf '%s\n' "$line" | cut -d' ' -f2-)
    rundir=$(printf '%s\n' "$line" | awk '{for(i=1;i<=NF;i++) if($i=="--name") print $(i+1)}')
    if [ -n "$rundir" ] && [ -f "runs/$rundir/final_validation.json" ]; then
        echo "[queue] SKIP  $tag (runs/$rundir already complete)" >> "$DONE"
        continue
    fi
    echo "[queue] START $tag  $(date -Is)" >> "$DONE"
    # shellcheck disable=SC2086
    uv run python -u -m marl_fleet_rebalancing.experiments.train $rest \
        > "runs/_logs/$tag.log" 2>&1
    echo "[queue] END   $tag rc=$?  $(date -Is)" >> "$DONE"
done < "$QUEUE"
echo "[queue] ALL DONE $(date -Is)" >> "$DONE"
