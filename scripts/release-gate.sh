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

# Experimental symbols that live INSIDE a Hard-tier package. The HARD list is
# package-granular, but some Hard packages carry both Hard and Experimental
# surface, and the README "Stability tiers" explicitly places these outside the
# 1.0 guarantee ("may change in any release, minor or patch"). apidiff is also
# package-granular, so without this an incompatible change to a documented-
# Experimental symbol (e.g. removing a field from the int8 encoder.LoadQ8 path)
# would wrongly fail the gate. We drop apidiff lines whose LEADING symbol is one
# of these. Keep in sync with README's Experimental tier.
#
# Scoped to `encoder` for now — it is the package with both tiers that has had an
# Experimental symbol actually change. `ann`/`bm25`/`fuse`/`embed` also carry
# Experimental surface (HNSW/FlatI8/QueryFilter, TokenizePlain, RSF, Truncate);
# add them here when a change there surfaces (note: a method on a Hard-tier type,
# e.g. Flat.QueryFilter, leads with the Hard type name, so it needs a `.Method`
# pattern rather than a leading-symbol one — encoder's Experimental surface is all
# its own types/funcs, so the simple leading-symbol match below is exact for it).
experimental_syms() {
	case "$1" in
	encoder) echo "Backend RegisterBackend NewBackend LoadQ8 ModelQ8 WeightsQ8 LayerWeightsQ8 LoadBERT BERT LoadSPLADE SPLADE LoadCrossEncoder CrossEncoder" ;;
	*) echo "" ;;
	esac
}

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
		# Drop incompatible changes to documented-Experimental symbols in this
		# package (outside the 1.0 guarantee — see experimental_syms above). Match
		# the apidiff line whose leading symbol is Experimental: "- Sym: ...",
		# "- Sym.Field: ...", or "- (*Sym).Method: ...".
		exsyms="$(experimental_syms "$p")"
		if [ -n "$exsyms" ] && [ -n "$inc" ]; then
			pat="$(echo "$exsyms" | tr ' ' '|')"
			excluded="$(echo "$inc" | grep -E "^- \(?\*?(${pat})[).: ]" || true)"
			inc="$(echo "$inc" | grep -vE "^- \(?\*?(${pat})[).: ]" || true)"
			if [ -n "$excluded" ]; then
				echo "release-gate: '${p}' — allowed Experimental-tier changes (excluded from Hard check):"
				echo "$excluded" | sed 's/^/    /'
			fi
		fi
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
