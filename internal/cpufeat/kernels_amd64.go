//go:build amd64

package cpufeat

// Kernels reports the arch, the tiers the hardware has (detected) and the tiers the flags leave on (active), for
// linalg.ActiveKernels.
func Kernels() (arch string, detected, active []string) {
	add := func(s *[]string, name string, on bool) {
		if on {
			*s = append(*s, name)
		}
	}
	vnni, vl := DetectAVX512VNNI(), DetectAVX512VL()
	add(&detected, "avx2", DetectAVX2())
	add(&detected, "f16c", DetectF16C())
	add(&detected, "avx512vnni", vnni)
	add(&detected, "avx512vnni+vl", vnni && vl)
	add(&detected, "popcnt", DetectPOPCNT())
	add(&active, "avx2", HasAVX2)
	add(&active, "f16c", HasF16C)
	add(&active, "avx512vnni", HasAVX512VNNI)
	add(&active, "avx512vnni+vl", HasAVX512VNNIVL)
	add(&active, "popcnt", HasPOPCNT)
	return "amd64", detected, active
}

// DisableTopTier turns the highest active ISA tier off (and the flags derived from it at init) and returns its name, or "" when none is left: AVX-512 VNNI first,
// then AVX2. F16C and POPCNT are separate features no dispatcher's wrong answer is attributed to here.
func DisableTopTier() string {
	if HasAVX512VNNI || HasAVX512VNNIVL {
		HasAVX512VNNI, HasAVX512VNNIVL = false, false
		return "avx512vnni"
	}
	if HasAVX2 {
		HasAVX2, HasQ4KAVX2 = false, false
		return "avx2"
	}
	return ""
}
