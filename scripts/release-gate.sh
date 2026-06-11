#!/usr/bin/env bash
# release-gate.sh <version> — validate that the repo is ready to release vX.Y.Z.
#
# Three tags shipped in two days and each cycle leaked a small process gap (a missing
# CHANGELOG entry; stale docs nearly tagged; a forgotten GitHub Release). This is the
# gate (roadmap §1.2). It checks:
#
#   1. CHANGELOG.md has a `## [X.Y.Z]` section AND a `[X.Y.Z]:` compare link.
#   2. apidiff shows NO incompatible change in any Hard-tier package vs the previous
#      tag — the stated release bar.
#   3. The core module pulls in no external dependency beyond golang.org/x/text
#      (the cgo-free, dependency-light invariant the README promises).
#
# Used by .github/workflows/release.yml (tag-triggered) and runnable locally:
#   scripts/release-gate.sh 1.4.0
set -uo pipefail

VER="${1:?usage: release-gate.sh <version, e.g. 1.4.0 (no leading v)>}"
MOD="github.com/townsendmerino/aikit"
# Hard-tier packages — the 1.0 compatibility guarantee (README "Stability tiers").
HARD="topk ann bm25 fuse embed encoder chunk"

fail=0
err() { echo "::error::release-gate: $*"; fail=1; }

# (1) CHANGELOG section + compare link.
if ! grep -qE "^## \[${VER}\]" CHANGELOG.md; then
	err "CHANGELOG.md has no '## [${VER}]' section"
fi
if ! grep -qE "^\[${VER}\]: " CHANGELOG.md; then
	err "CHANGELOG.md has no '[${VER}]:' compare link"
fi

# (2) apidiff — no Hard-tier breakage vs the previous tag.
PREV="$(git tag --list 'v*' --sort=-v:refname | grep -vx "v${VER}" | head -1)"
if [ -z "$PREV" ]; then
	echo "release-gate: no previous tag — skipping apidiff"
else
	echo "release-gate: apidiff Hard tier ${PREV} → current tree"
	if ! command -v apidiff >/dev/null 2>&1 && [ ! -x "$(go env GOPATH)/bin/apidiff" ]; then
		go install golang.org/x/exp/cmd/apidiff@latest || err "could not install apidiff"
	fi
	APIDIFF="$(command -v apidiff || echo "$(go env GOPATH)/bin/apidiff")"
	WT="$(mktemp -d)"
	git worktree add -q -f "$WT" "$PREV" || err "worktree add $PREV failed"
	for p in $HARD; do
		( cd "$WT" && "$APIDIFF" -w "/tmp/gate-${p}.api" "${MOD}/${p}" ) 2>/dev/null
		inc="$("$APIDIFF" -incompatible "/tmp/gate-${p}.api" "${MOD}/${p}" 2>&1)"
		if [ -n "$inc" ]; then
			err "apidiff: incompatible change in Hard-tier '${p}' vs ${PREV}:"
			echo "$inc"
		fi
	done
	git worktree remove --force "$WT" 2>/dev/null
	git worktree prune
fi

# (3) Core dependency invariant: nothing external beyond golang.org/x/text.
ext="$(go list -deps -f '{{if not .Standard}}{{.ImportPath}}{{end}}' ./... 2>/dev/null \
	| grep -v "^${MOD}" | grep -v '^golang.org/x/text' | grep -v '^$' || true)"
if [ -n "$ext" ]; then
	err "core module pulls unexpected external dependency(ies): $(echo "$ext" | tr '\n' ' ')"
fi

if [ "$fail" -eq 0 ]; then
	echo "release-gate: v${VER} OK ✓"
else
	echo "release-gate: v${VER} FAILED ✗"
fi
exit "$fail"
