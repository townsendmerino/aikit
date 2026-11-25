package shape

import (
	"strings"
	"testing"
)

func panicMsg(f func()) (msg string) {
	defer func() {
		if r := recover(); r != nil {
			msg, _ = r.(string)
		}
	}()
	f()
	return ""
}

func TestMul(t *testing.T) {
	for _, c := range []struct{ a, b, want int }{{1 << 40, 1 << 40, -1}, {-1, 4, -1}, {4, -1, -1}, {0, 1 << 62, 0}, {1000, 1000, 1_000_000}} {
		if got := Mul(c.a, c.b); got != c.want {
			t.Errorf("Mul(%d, %d) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}

func TestRequire(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func()
		want string // "" = no panic
	}{
		{"len ok", func() { RequireLen("K", "a", 5, 4) }, ""},
		{"len short", func() { RequireLen("K", "a", 3, 4) }, "linalg: K: a length 3 < required 4"},
		{"len overflow", func() { RequireLen("K", "a", 3, -1) }, "linalg: K dimension product overflows int (a)"},
		{"exact ok", func() { RequireExactLen("K", "b", 4, 4) }, ""},
		{"exact long", func() { RequireExactLen("K", "b", 5, 4) }, "linalg: K: b length 5 != required 4"},
		{"exact overflow", func() { RequireExactLen("K", "b", 5, -1) }, "linalg: K dimension product overflows int (b)"},
	} {
		if got := panicMsg(c.f); got != c.want || (c.want != "" && !strings.HasPrefix(got, "linalg: ")) {
			t.Errorf("%s: panic %q, want %q", c.name, got, c.want)
		}
	}
}
