package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var kquantRows = []kernreg.Row{
	{Symbol: "dotQ4KAVX2", Arch: "amd64", Dispatch: "dotQ4K", Oracle: "TestQ4K_kernelMatchesGo"},
	{Symbol: "dotQ5KAVX2", Arch: "amd64", Dispatch: "dotQ5K", Oracle: "TestQ56K_kernelMatchesGo"},
	{Symbol: "dotQ6KAVX2", Arch: "amd64", Dispatch: "dotQ6K", Oracle: "TestQ56K_kernelMatchesGo"},
	{Symbol: "dotPartials16SDOT", Arch: "arm64", Dispatch: "dotPartials16", Oracle: "TestKQuantDotPartials_matchesScalar"},
}
