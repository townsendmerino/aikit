package linalg

import "github.com/townsendmerino/aikit/internal/kernreg"

// f16scaleAdapters is empty on amd64: f16scale_arm64.s is arm64 only.
func f16scaleAdapters() []kernreg.Adapter { return nil }
