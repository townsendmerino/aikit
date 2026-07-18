package mmap

import (
	"os"
	"strconv"
	"strings"
	"unsafe"
)

// PageAlignedInterior returns the page-aligned interior of raw: the sub-slice that
// starts at the first page boundary at or after raw's start and ends at the last
// page boundary at or before raw's end. It returns nil when raw spans less than a
// full page interior.
//
// This is the span Advise (MADV_DONTNEED) can release without touching a neighbor's
// page — rounding the start up and the end down guarantees the released pages lie
// entirely within raw, so evicting one member never disturbs the bytes of another
// that shares a boundary page. The few boundary bytes it omits are negligible
// against a multi-MB tensor or index block.
func PageAlignedInterior(raw []byte) []byte {
	if len(raw) == 0 {
		return nil
	}
	pg := uintptr(os.Getpagesize())
	start := uintptr(unsafe.Pointer(&raw[0]))
	as := (start + pg - 1) &^ (pg - 1)            // round start up to a page
	ae := (start + uintptr(len(raw))) &^ (pg - 1) // round end down to a page
	if ae <= as {
		return nil
	}
	off := int(as - start)
	return raw[off : off+int(ae-as)]
}

// AutoBudget picks a default span-cache budget: about half of available RAM,
// falling back to 8 GiB when available RAM can't be read (non-Linux, or no
// MemAvailable line). Callers that want a specific cap should pass their own.
func AutoBudget() int64 {
	const fallback = 8 << 30
	if avail := AvailableRAM(); avail > 0 {
		return avail / 2
	}
	return fallback
}

// AvailableRAM reads MemAvailable from /proc/meminfo (Linux) and returns it in
// bytes; it returns 0 where that can't be read (non-Linux, or the field is absent).
// A missing /proc/meminfo is not an error — callers treat 0 as "unknown" and fall
// back to a fixed budget (see AutoBudget).
func AvailableRAM() int64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		f := strings.Fields(line) // "MemAvailable:  12345678 kB"
		if len(f) >= 2 {
			if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
				return kb * 1024
			}
		}
	}
	return 0
}
