module github.com/townsendmerino/aikit

go 1.26.5

require (
	golang.org/x/sys v0.47.0
	golang.org/x/text v0.40.0
)

// v1.8.0 shipped before the release gate passed (missing CHANGELOG compare link)
// and without the GGUF nested-array parser hardening — superseded by v1.8.1.
retract v1.8.0
