//go:build arm64 && linux

package linalg

import (
	"encoding/binary"
	"os"
)

// hwcapASIMDDP is the AArch64 HWCAP bit for the DotProd extension (SDOT/UDOT).
const hwcapASIMDDP = 1 << 20

// detectDotProd reports whether the CPU implements DotProd, by reading AT_HWCAP
// from the kernel-supplied auxiliary vector. /proc/self/auxv is a flat sequence
// of (type, value) uint64 pairs (little-endian on arm64) terminated by AT_NULL
// (type 0); AT_HWCAP has type 16. Reading auxv works under qemu-user too — it
// constructs the guest auxv from the emulated CPU's capabilities — so the SDOT
// path stays exercisable in CI without DotProd silicon. Any read/parse failure
// returns false, falling back to the base SMULL/SADALP kernel.
func detectDotProd() bool {
	const atNull, atHWCAP = 0, 16
	buf, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	for i := 0; i+16 <= len(buf); i += 16 {
		typ := binary.LittleEndian.Uint64(buf[i:])
		val := binary.LittleEndian.Uint64(buf[i+8:])
		if typ == atNull {
			break
		}
		if typ == atHWCAP {
			return val&hwcapASIMDDP != 0
		}
	}
	return false
}
