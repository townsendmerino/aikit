package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// peakRows are the throughput probes: register-only loops that take an iteration count
// and no memory.
var peakRows = []kernreg.Row{
	{Symbol: "fmaPeakAMD64", Arch: "amd64", NoSlices: "a register-only FMA peak probe (takes an iteration count)"},
	{Symbol: "fmaPeakARM64", Arch: "arm64", NoSlices: "a register-only FMA peak probe (takes an iteration count)"},
	{Symbol: "sdotIssuePeak", Arch: "arm64", NoSlices: "a register-only SDOT issue-rate probe (takes an iteration count)"},
}
