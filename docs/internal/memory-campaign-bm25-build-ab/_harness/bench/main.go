// bench times bm25.Build (and, built with -tags aikitnew, Builder.Add + Builder.Build) over a tokens blob: one warm-up,
// then -reps timed runs, and prints one JSON line. The same source is linked against each arm's bm25.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/townsendmerino/aikit/bm25"
)

var sink any

func timeIt(reps int, fn func()) []int64 {
	fn() // warm-up
	out := make([]int64, reps)
	for i := range out {
		runtime.GC()
		t := time.Now()
		fn()
		out[i] = time.Since(t).Nanoseconds()
	}
	return out
}

func median(v []int64) int64 {
	s := append([]int64(nil), v...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[len(s)/2]
}

func main() {
	path := flag.String("tokens", "", "tokens blob")
	reps := flag.Int("reps", 5, "timed runs")
	flag.Parse()
	b, err := os.ReadFile(*path)
	if err == nil {
		var docs [][]string
		if docs, err = bm25.UnmarshalTokens(b); err == nil {
			out := map[string]any{}
			bs := timeIt(*reps, func() { sink = bm25.Build(docs) })
			out["build_ns"], out["build_samples"] = median(bs), bs
			if ok := builderAvailable(); ok {
				bb := timeIt(*reps, func() { sink = builderBuild(docs) })
				out["builder_ns"], out["builder_samples"] = median(bb), bb
			}
			j, _ := json.Marshal(out)
			fmt.Println(string(j))
			return
		}
	}
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
