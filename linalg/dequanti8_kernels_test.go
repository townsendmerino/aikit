package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

var dequanti8Rows = []kernreg.Row{
	{Symbol: "dequantI8AVX2", Arch: "amd64", Dispatch: "dequantRowInt8", Oracle: "TestDequantizeRowsInt8_bitIdentical"},
	{Symbol: "dequantI8NEON", Arch: "arm64", Dispatch: "dequantRowInt8", Oracle: "TestDequantizeRowsInt8_bitIdentical"},
}
