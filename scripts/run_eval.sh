#!/usr/bin/env bash
# ============================================================
#  AgentDDx — crash-proof evaluation runner
#
#  Runs all 3 conditions on the full MedQA test set (n=1273).
#  Each condition is supervised: if it dies (API error, OOM, network
#  drop, killed process), it is automatically restarted and RESUMES
#  from its last checkpoint. It keeps restarting until the condition
#  has actually completed all N questions.
#
#  Usage:
#     chmod +x run_eval.sh
#     ./run_eval.sh                 # full run, n=1273
#     N=600 ./run_eval.sh           # smaller run
#
#  Safe to re-run: it resumes, never restarts from zero.
#  Safe to Ctrl-C: just run it again, it picks up where it left off.
# ============================================================

set -uo pipefail   # NOT -e: we WANT to survive child failures

# Always operate from the repository root, whichever directory we were called from.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

N="${N:-1273}"
CONDITIONS=(llm_only llm_pubmed full_system)
MAX_RESTARTS="${MAX_RESTARTS:-200}"   # generous; each restart resumes, no work lost
COOLDOWN="${COOLDOWN:-60}"            # seconds after a crash (lets rate limits reset)

mkdir -p logs results

# ── how many questions has this condition already logged? ──
progress() {
  local cond=$1
  python3 - "$cond" <<'PY'
import json, sys, os
cond = sys.argv[1]
f = os.path.join("results", f"eval_{cond}.json")
if not os.path.exists(f):
    print(0); sys.exit()
try:
    print(len(json.load(open(f)).get("log", [])))
except Exception:
    print(0)   # corrupt/partial write -> treat as 0, it will re-checkpoint
PY
}

# ── supervise ONE condition until it reaches N ──
supervise() {
  local cond=$1
  local log="logs/${cond}.log"
  local tries=0

  while true; do
    local done_n
    done_n=$(progress "$cond")

    if [ "$done_n" -ge "$N" ]; then
      echo "[$(date +%H:%M:%S)] ✅ $cond COMPLETE ($done_n/$N)"
      return 0
    fi

    if [ "$tries" -ge "$MAX_RESTARTS" ]; then
      echo "[$(date +%H:%M:%S)] ❌ $cond gave up after $tries restarts (at $done_n/$N)"
      return 1
    fi

    tries=$((tries + 1))
    echo "[$(date +%H:%M:%S)] ▶ $cond  attempt #$tries  (resuming at $done_n/$N)"

    # evaluate.py auto-resumes from eval_<cond>.json if it exists
    python3 evaluate.py --n "$N" --condition "$cond" >> "$log" 2>&1

    local rc=$?
    local now
    now=$(progress "$cond")

    if [ "$now" -ge "$N" ]; then
      echo "[$(date +%H:%M:%S)] ✅ $cond COMPLETE ($now/$N)"
      return 0
    fi

    # Made no progress at all? Something is systematically broken
    # (bad API key, no network, missing dep) — back off harder.
    if [ "$now" -le "$done_n" ]; then
      echo "[$(date +%H:%M:%S)] ⚠ $cond made NO progress (rc=$rc, still $now). Check $log"
      sleep $((COOLDOWN * 3))
    else
      echo "[$(date +%H:%M:%S)] ↻ $cond crashed (rc=$rc) at $now/$N — restarting in ${COOLDOWN}s"
      sleep "$COOLDOWN"
    fi
  done
}

echo "============================================================"
echo "  AgentDDx evaluation — n=$N"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Started: $(date)"
echo "  Logs: logs/<condition>.log"
echo "============================================================"

# Run the three conditions in PARALLEL, each with its own supervisor.
pids=()
for c in "${CONDITIONS[@]}"; do
  supervise "$c" &
  pids+=($!)
done

# Wait for all supervisors
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done

echo
echo "============================================================"
echo "  All conditions finished at $(date)"
for c in "${CONDITIONS[@]}"; do
  echo "    $c: $(progress "$c")/$N"
done
echo "============================================================"

if [ "$fail" -ne 0 ]; then
  echo "⚠ Some conditions did not complete. Re-run ./run_eval.sh to continue."
  exit 1
fi

echo
echo "Merging into final report..."
python3 evaluate.py --merge results/eval_llm_only.json results/eval_llm_pubmed.json results/eval_full_system.json

echo
echo "DONE. See results/eval_report_merged.json"
