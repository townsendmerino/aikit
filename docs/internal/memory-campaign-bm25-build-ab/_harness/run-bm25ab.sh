#!/usr/bin/env bash
# Job 5 (aikit docs/internal/task-memory-campaign.md, "Job 5"): BM25 Build A/B, v1.70.2 vs 8ba2b3c, paired and
# interleaved, on aikit's tree and a 100k-chunk synthetic corpus; Builder.Add reported. Self-contained: builds both
# arms from git worktrees of $REPO, writes everything under $WORK, removes its worktrees on exit.
set -euo pipefail
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH GOTOOLCHAIN=${GOTOOLCHAIN:-go1.27.2} GOWORK=off
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$HOME/tmp/aikit-mc-8ba2b3c}
WORK=${WORK:-$HOME/tmp/aikit-mc-night/bm25ab-work}
OLD=${OLD:-v1.70.2} NEW=${NEW:-8ba2b3c}
ROUNDS=${ROUNDS:-15} REPS=${REPS:-5} SYNTH_DOCS=${SYNTH_DOCS:-100000}
LOG=${LOG:-$HOME/tmp/aikit-mc-night/logs/bm25ab.log}
mkdir -p "$WORK" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "== $(date '+%F %T %Z') $(hostname) $(go version): $OLD vs $NEW, $ROUNDS rounds × $REPS reps, synthetic $SYNTH_DOCS docs"
cleanup() { git -C "$REPO" worktree remove --force "$WORK/old" 2>/dev/null || true; git -C "$REPO" worktree remove --force "$WORK/new" 2>/dev/null || true; }
trap cleanup EXIT
cleanup
git -C "$REPO" worktree add -q --detach "$WORK/old" "$OLD"
git -C "$REPO" worktree add -q --detach "$WORK/new" "$NEW"
echo "old $(git -C "$WORK/old" log --oneline -1)"; echo "new $(git -C "$WORK/new" log --oneline -1)"
mkmod() { # mkmod DIR SRCDIR TREE
  rm -rf "$1"; mkdir -p "$1"; cp "$2"/*.go "$1"/
  printf 'module bm25ab\n\ngo 1.27.0\n\nrequire github.com/townsendmerino/aikit v0.0.0-00010101000000-000000000000\n\nreplace github.com/townsendmerino/aikit => %s\n' "$3" > "$1/go.mod"
  (cd "$1" && go mod tidy >/dev/null)
}
mkmod "$WORK/arm-old" "$HERE/bench" "$WORK/old"; (cd "$WORK/arm-old" && go build -o "$WORK/bench-old" .)
mkmod "$WORK/arm-new" "$HERE/bench" "$WORK/new"; (cd "$WORK/arm-new" && go build -tags aikitnew -o "$WORK/bench-new" .)
mkmod "$WORK/gen" "$HERE/gen" "$WORK/new"; (cd "$WORK/gen" && go build -o "$WORK/gen-bin" .)
"$WORK/gen-bin" -repo "$WORK/new" -out-tree "$WORK/tree.tokens" -out-synth "$WORK/synth.tokens" -synth-docs "$SYNTH_DOCS"
RES="$WORK/results.jsonl"; : > "$RES"
start=$(date +%s)
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then order="old new"; else order="new old"; fi
  for corpus in tree synth; do
    for arm in $order; do
      line=$("$WORK/bench-$arm" -tokens "$WORK/$corpus.tokens" -reps "$REPS")
      echo "{\"round\":$r,\"corpus\":\"$corpus\",\"arm\":\"$arm\",${line#\{}" >> "$RES"
    done
  done
  el=$(( $(date +%s) - start ))
  echo "round $r/$ROUNDS done, ${el}s elapsed, ~$(( el * (ROUNDS - r) / r ))s left"
done
python3 "$HERE/analyze.py" "$RES"
