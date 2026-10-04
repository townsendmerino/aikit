//go:build amd64

package linalg

func archKernels() (arch string, detected, active []string) {
	add := func(s *[]string, name string, on bool) {
		if on {
			*s = append(*s, name)
		}
	}
	vnni, vl := detectAVX512VNNI(), detectAVX512VL()
	add(&detected, "avx2", detectAVX2())
	add(&detected, "f16c", detectF16C())
	add(&detected, "avx512vnni", vnni)
	add(&detected, "avx512vnni+vl", vnni && vl)
	add(&detected, "popcnt", detectPOPCNT())
	add(&active, "avx2", hasAVX2)
	add(&active, "f16c", hasF16C)
	add(&active, "avx512vnni", hasAVX512VNNI)
	add(&active, "avx512vnni+vl", hasAVX512VNNIVL)
	add(&active, "popcnt", hasPOPCNT)
	return "amd64", detected, active
}

// disableTopTier turns the highest active ISA tier off (and the flags derived from it at init) and returns its name, or "" when none is left: AVX-512 VNNI first,
// then AVX2. F16C and POPCNT are separate features no dispatcher's wrong answer is attributed to here.
func disableTopTier() string {
	if hasAVX512VNNI || hasAVX512VNNIVL {
		hasAVX512VNNI, hasAVX512VNNIVL = false, false
		return "avx512vnni"
	}
	if hasAVX2 {
		hasAVX2, hasQ4KAVX2 = false, false
		return "avx2"
	}
	return ""
}
