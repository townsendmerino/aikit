//go:build race

package ann

// raceEnabled is set in a -race build: the slow, measurement-only tests skip there.
const raceEnabled = true
