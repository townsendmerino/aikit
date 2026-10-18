#!/usr/bin/env python3
"""analyze.py RESULTS.jsonl — applies job 5's registered rule (docs/internal/task-memory-campaign.md, "Job 5"):
per corpus, the median over paired rounds of new/old Build time; ACCEPT at <= 1.10 on both corpora, otherwise
INVESTIGATE (no automatic revert). Builder.Add is reported against both Builds, with no bar."""
import json, statistics, sys

BAR = 1.10
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
corpora = sorted({r["corpus"] for r in rows})
verdicts = []
for c in corpora:
    by = {}
    for r in rows:
        if r["corpus"] == c:
            by.setdefault(r["round"], {})[r["arm"]] = r
    rounds = sorted(k for k, v in by.items() if "old" in v and "new" in v)
    ratios = [by[k]["new"]["build_ns"] / by[k]["old"]["build_ns"] for k in rounds]
    b_new = [by[k]["new"]["builder_ns"] / by[k]["new"]["build_ns"] for k in rounds]
    b_old = [by[k]["new"]["builder_ns"] / by[k]["old"]["build_ns"] for k in rounds]
    ms = lambda arm, key: [by[k][arm][key] / 1e6 for k in rounds]
    med = statistics.median(ratios)
    ok = med <= BAR
    verdicts.append(ok)
    print(f"[{c}] {len(rounds)} paired rounds")
    for arm, key, label in (("old", "build_ns", "v1.70.2 Build"), ("new", "build_ns", "8ba2b3c Build"), ("new", "builder_ns", "8ba2b3c Builder.Add+Build")):
        v = ms(arm, key)
        print(f"  {label:28s} median {statistics.median(v):9.2f} ms  range {min(v):9.2f}–{max(v):9.2f} ms")
    print(f"  Build new/old: median {med:.3f}×  (rounds {min(ratios):.3f}–{max(ratios):.3f})  bar ≤ {BAR:.2f}× → {'ACCEPT' if ok else 'INVESTIGATE'}")
    print(f"  Builder.Add vs new Build: median {statistics.median(b_new):.3f}×; vs v1.70.2 Build: median {statistics.median(b_old):.3f}×  (reported, no bar)")
print("VERDICT: " + ("ACCEPT — Build within 1.10× of v1.70.2 on both corpora" if all(verdicts) and len(verdicts) == 2
                     else "INVESTIGATE — Build above 1.10× of v1.70.2 on at least one corpus (no automatic revert)" if len(verdicts) == 2
                     else f"INCONCLUSIVE — {len(verdicts)} corpora measured, want 2"))
