//go:build race

package linalg

// raceEnabled is set in a -race build: tests that count allocations through a sync.Pool skip the count there.
const raceEnabled = true
