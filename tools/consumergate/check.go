package main

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/townsendmerino/aikit/tools/gate"
)

// Pair is one module@version to verify.
type Pair struct {
	Module, Version string
}

func (p Pair) String() string { return p.Module + "@" + p.Version }

// config carries the per-run knobs (proxy, retry, target GOOS) so the check functions stay
// pure w.r.t. process globals — the shape a migrated gpu_gate/gpu_device would reuse. goos
// is the GOOS the compile subprocess actually targets (`go env GOOS`, which honors an
// overriding GOOS env), so the n/a reason matches the header rather than the tool's own
// build OS — they differ only when the compile tier is cross-targeted.
type config struct {
	proxy    string
	attempts int
	backoff  time.Duration
	goos     string
}

// A version that does not exist is a module DEFECT (FAIL). A proxy that cannot be reached
// is the gate being UNABLE TO JUDGE (INCONCLUSIVE). They must not be confused, so the
// resolve failure is classified by its message.
var reUnreachable = regexp.MustCompile(`(?i)disabled by GOPROXY|dial tcp|no such host|connection refused|i/o timeout|TLS handshake|server misbehaving|proxyconnect|network is unreachable|temporary failure in name resolution|reset by peer|[^0-9]5[0-9][0-9] `)
var reAbsent = regexp.MustCompile(`(?i)not found|unknown revision|invalid version|does not contain|no matching versions|404`)
var reNoPackages = regexp.MustCompile(`matched no packages`)

// checkPair is the unit of work: a fresh scratch module (a consumer importing ONE module
// sees only that module's graph; sharing one scratch would let MVS raise a shared
// dependency and mask exactly the v0.0.0 this gate exists to catch), then two tiers.
func checkPair(cfg config, p Pair) gate.Cell {
	cell := gate.Cell{Name: p.String()}

	sb, err := os.MkdirTemp("", "consumergate")
	if err != nil {
		cell.Outcome = gate.Inconclusive
		cell.Fields = []gate.Field{{Key: "setup", State: "INCONCLUSIVE", Detail: err.Error()}}
		return cell
	}
	defer os.RemoveAll(sb) //nolint:errcheck // a scratch module under the OS temp dir: nothing to report
	if out, err := goRun(cfg, sb, "go", "mod", "init", "consumergate.invalid/scratch"); err != nil {
		cell.Outcome = gate.Inconclusive
		cell.Fields = []gate.Field{{Key: "setup", State: "INCONCLUSIVE", Detail: firstLine(out)}}
		return cell
	}

	// resolve tier — platform-independent. Retry a not-found with bounded backoff (a
	// freshly published path can lag on the proxy), then go red — never green-by-timeout.
	rf := gate.Field{Key: "resolve"}
	wait := cfg.backoff
	for attempt := 1; ; attempt++ {
		out, err := goRun(cfg, sb, "go", "get", p.String())
		if err == nil {
			_, err = goRun(cfg, sb, "go", "mod", "download")
		}
		if err == nil {
			rf.State = "ok"
			break
		}
		if reUnreachable.MatchString(out) {
			rf.State, rf.Detail = "unreachable", matchLine(out, reUnreachable)
			break
		}
		if attempt >= cfg.attempts {
			rf.State = "absent"
			if reAbsent.MatchString(out) {
				rf.Detail = fmt.Sprintf("not found on proxy after %d attempts", attempt)
			} else {
				rf.Detail = firstLine(out)
			}
			break
		}
		time.Sleep(wait)
		wait *= 2
	}

	// compile tier — only meaningful if resolve worked; classify ok / n/a / FAIL exactly
	// as gpu_device.sh does: no packages for this GOOS is n/a, not a failure.
	cf := gate.Field{Key: "compile", State: "—"}
	if rf.State == "ok" {
		out, err := goRunCGOoff(cfg, sb, "go", "build", p.Module+"/...")
		switch {
		case reNoPackages.MatchString(out):
			cf.State, cf.Detail = "n/a", "no packages on "+cfg.goos
		case err != nil:
			cf.State, cf.Detail = "FAIL", buildError(out)
		default:
			cf.State = "ok"
		}
	}

	cell.Fields = []gate.Field{rf, cf}
	switch {
	case rf.State == "unreachable":
		cell.Outcome = gate.Inconclusive
	case rf.State != "ok" || cf.State == "FAIL":
		cell.Outcome = gate.Fail
	default:
		cell.Outcome = gate.OK
	}
	return cell
}

// run executes a go subprocess in the scratch dir under the consumer environment: GOWORK
// is OFF (a workspace overrides module resolution with local dirs and would hide the very
// defect a consumer hits), GOPROXY is the public proxy with NO ,direct fallback (a
// direct-git fallback would pass here while consumers fail), and -mod=mod so `go get` may
// edit the scratch go.mod. Returns combined output and the exit error.
func goRun(cfg config, dir, name string, args ...string) (string, error) {
	return goRunEnv(cfg, dir, nil, name, args...)
}

func goRunCGOoff(cfg config, dir, name string, args ...string) (string, error) {
	return goRunEnv(cfg, dir, []string{"CGO_ENABLED=0"}, name, args...)
}

func goRunEnv(cfg config, dir string, extra []string, name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GOWORK=off",
		"GOFLAGS=-mod=mod",
		"GOPROXY="+cfg.proxy,
	)
	cmd.Env = append(cmd.Env, extra...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func firstLine(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			return trim(ln)
		}
	}
	return ""
}

func matchLine(s string, re *regexp.Regexp) string {
	for ln := range strings.SplitSeq(s, "\n") {
		if re.MatchString(ln) {
			return trim(ln)
		}
	}
	return firstLine(s)
}

// buildError returns the first line that is neither a `go: downloading/added` progress
// note nor a bare `# pkg` header — the actual compile error.
func buildError(s string) string {
	for ln := range strings.SplitSeq(s, "\n") {
		t := strings.TrimSpace(ln)
		if t == "" || strings.HasPrefix(t, "#") ||
			strings.HasPrefix(t, "go: downloading") || strings.HasPrefix(t, "go: added") ||
			strings.HasPrefix(t, "go: found") {
			continue
		}
		return trim(ln)
	}
	return firstLine(s)
}

func trim(ln string) string {
	ln = strings.TrimSpace(ln)
	if len(ln) > 90 {
		ln = ln[:90]
	}
	return ln
}

// shortLabel is a compact name for the scope summary: "root" for the root module, the
// subpath (e.g. "gpu/annmetal") for everything under it.
func shortLabel(mod string) string {
	if mod == rootPath {
		return "root"
	}
	return strings.TrimPrefix(mod, rootPath+"/")
}
