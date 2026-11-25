//go:build amd64

package cpufeat

// CPUID executes CPUID with the given leaf/subleaf. Implemented in
// cpuid_amd64.s.
//
//go:noescape
func CPUID(eaxIn, ecxIn uint32) (eax, ebx, ecx, edx uint32)

// XGETBV reads XCR0 (the OS-enabled extended-state mask). Implemented
// in cpuid_amd64.s.
//
//go:noescape
func XGETBV() (eax, edx uint32)

// HasAVX2 is true when the CPU and OS both support AVX2 + FMA + YMM
// state save. Computed once at init; the kernels branch on it.
var HasAVX2 = DetectAVX2()

// osSavesYMM reports whether the OS preserves XMM and YMM state across context switches (CPUID.1:ECX OSXSAVE and AVX set, XCR0 bits 1 and 2): the
// precondition for executing any VEX-encoded instruction, whatever feature bit the instruction itself needs.
func osSavesYMM() bool {
	const bitOSXSAVE, bitAVX = 1 << 27, 1 << 28
	_, _, ecx1, _ := CPUID(1, 0)
	if ecx1&(bitOSXSAVE|bitAVX) != bitOSXSAVE|bitAVX {
		return false
	}
	xcr0, _ := XGETBV()
	return xcr0&(1<<1|1<<2) == 1<<1|1<<2
}

// DetectAVX2 asks the hardware and the OS, every call; HasAVX2 is its answer at init unless a tag or a repair turned it off.
func DetectAVX2() bool {
	maxLeaf, _, _, _ := CPUID(0, 0)
	if maxLeaf < 7 {
		return false
	}
	// Leaf 1: ECX feature bits. Need OSXSAVE (27), AVX (28), FMA (12).
	const (
		bitFMA     = 1 << 12
		bitOSXSAVE = 1 << 27
		bitAVX     = 1 << 28
	)
	_, _, ecx1, _ := CPUID(1, 0)
	if ecx1&(bitFMA|bitOSXSAVE|bitAVX) != (bitFMA | bitOSXSAVE | bitAVX) {
		return false
	}
	// OSXSAVE set ⇒ XGETBV is usable. XCR0 bits 1 (SSE/XMM) and 2 (AVX/
	// YMM) must both be set for the OS to preserve YMM across context
	// switches; without that, executing AVX would corrupt state.
	xcr0, _ := XGETBV()
	const xcr0YMM = 1<<1 | 1<<2
	if xcr0&xcr0YMM != xcr0YMM {
		return false
	}
	// Leaf 7, subleaf 0: EBX bit 5 = AVX2.
	_, ebx7, _, _ := CPUID(7, 0)
	const bitAVX2 = 1 << 5
	return ebx7&bitAVX2 != 0
}

// HasAVX512VNNI is true when the CPU reports AVX512F + AVX512BW + AVX512_VNNI
// (CPUID leaf 7) and the OS has enabled the full ZMM + opmask extended state
// (XCR0 bits 1,2,5,6,7 — SSE, AVX, opmask, ZMM_Hi256, Hi16_ZMM). Computed once
// at init, same pattern as HasAVX2.
var HasAVX512VNNI = DetectAVX512VNNI()

// DetectAVX512VNNI asks the hardware and the OS, every call (see HasAVX512VNNI).
func DetectAVX512VNNI() bool {
	maxLeaf, _, _, _ := CPUID(0, 0)
	if maxLeaf < 7 {
		return false
	}
	// OSXSAVE (leaf 1, ECX bit 27) must be set before XGETBV is safe to run —
	// same prerequisite DetectAVX2 checks.
	const bitOSXSAVE = 1 << 27
	_, _, ecx1, _ := CPUID(1, 0)
	if ecx1&bitOSXSAVE == 0 {
		return false
	}
	// Leaf 7, subleaf 0: EBX bit 16 = AVX512F, bit 30 = AVX512BW (byte/word
	// ops — VPDPBUSD's operand granularity); ECX bit 11 = AVX512_VNNI.
	_, ebx7, ecx7, _ := CPUID(7, 0)
	const (
		bitAVX512F    = 1 << 16
		bitAVX512BW   = 1 << 30
		bitAVX512VNNI = 1 << 11
	)
	if ebx7&(bitAVX512F|bitAVX512BW) != (bitAVX512F | bitAVX512BW) {
		return false
	}
	if ecx7&bitAVX512VNNI == 0 {
		return false
	}
	// XCR0 bits 1 (SSE/XMM), 2 (AVX/YMM), 5 (opmask, k0-k7), 6 (ZMM_Hi256,
	// upper half of ZMM0-15), 7 (Hi16_ZMM, all of ZMM16-31) must ALL be set
	// for the OS to preserve full AVX-512 state across a context switch;
	// without that, a ZMM instruction would silently corrupt state.
	xcr0, _ := XGETBV()
	const xcr0AVX512 = 1<<1 | 1<<2 | 1<<5 | 1<<6 | 1<<7
	return xcr0&xcr0AVX512 == xcr0AVX512
}

// HasAVX512VNNIVL is HasAVX512VNNI (dot_i8_avx512vnni_amd64.go) plus
// AVX512VL (CPUID leaf 7, EBX bit 31). dotI8AVX512VNNI only ever touches
// full ZMM registers, which needs just AVX512F+BW+VNNI; this file's kernel
// uses the 256-bit (YMM) VPDPBUSD encoding to match one 32-byte quant group
// per instruction, and that encoding is only legal with VL present. Every
// shipping VNNI CPU (Cascade Lake onward on Intel, Zen 4 onward on AMD)
// bundles VL in as part of the same AVX-512 baseline, so this is not
// expected to ever differ from HasAVX512VNNI in practice — but dispatch
// shouldn't rely on an unchecked assumption for an instruction that SIGILLs
// if it's wrong, so it's checked explicitly rather than reusing
// HasAVX512VNNI directly. Computed once at init, same pattern as
// HasAVX512VNNI/HasAVX2.
var HasAVX512VNNIVL = HasAVX512VNNI && DetectAVX512VL()

// DetectAVX512VL reports CPUID leaf 7 EBX bit 31, AVX512VL; HasAVX512VNNIVL needs it as well as VNNI.
func DetectAVX512VL() bool {
	_, ebx7, _, _ := CPUID(7, 0)
	const bitAVX512VL = 1 << 31
	return ebx7&bitAVX512VL != 0
}

// HasF16C is true when the CPU has F16C and the OS saves YMM state (DetectF16C); the binary16 scale kernels branch on it.
var HasF16C = DetectF16C()

// HasQ4KAVX2 is HasAVX2 with F16C: the AVX2 Q4_K kernel also needs F16C (VCVTPH2PS) for the super-block's d and dmin.
var HasQ4KAVX2 = HasAVX2 && DetectF16C()

// DetectF16C reports CPUID.1:ECX bit 29 (every AVX2 CPU ships it; checked rather than assumed) and that the OS saves YMM state: VCVTPH2PS is a VEX
// instruction, so the CPUID bit alone is not enough to run it.
func DetectF16C() bool {
	_, _, ecx1, _ := CPUID(1, 0)
	return ecx1&(1<<29) != 0 && osSavesYMM()
}

// HasPOPCNT reports whether the CPU implements POPCNT (SSE4.2 era, 2008 and
// later). Detected at init for the same reason HasAVX2 is: so one binary runs
// the fast kernel where it can and the portable one where it cannot, instead of
// the build's GOAMD64 level deciding for every user.
//
// No OS-support check is needed, unlike AVX2 — POPCNT adds no architectural
// state, so there is nothing for the kernel to save or restore.
var HasPOPCNT = DetectPOPCNT()

// DetectPOPCNT asks the hardware, every call (see HasPOPCNT).
func DetectPOPCNT() bool {
	maxLeaf, _, _, _ := CPUID(0, 0)
	if maxLeaf < 1 {
		return false
	}
	// Leaf 1: ECX bit 23 = POPCNT.
	_, _, ecx1, _ := CPUID(1, 0)
	const bitPOPCNT = 1 << 23
	return ecx1&bitPOPCNT != 0
}
