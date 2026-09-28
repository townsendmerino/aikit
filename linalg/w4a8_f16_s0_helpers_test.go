package linalg

import "math"

func mathFloat32bits(f float32) uint32     { return math.Float32bits(f) }
func mathFloat32frombits(b uint32) float32 { return math.Float32frombits(b) }
