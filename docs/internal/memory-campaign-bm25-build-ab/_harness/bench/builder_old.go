//go:build !aikitnew

package main

func builderAvailable() bool           { return false }
func builderBuild(docs [][]string) any { return nil }
