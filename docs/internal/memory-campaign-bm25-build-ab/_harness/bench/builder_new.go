//go:build aikitnew

package main

import "github.com/townsendmerino/aikit/bm25"

func builderAvailable() bool { return true }

func builderBuild(docs [][]string) any {
	b := bm25.NewBuilder()
	for _, d := range docs {
		b.Add(d)
	}
	return b.Build()
}
