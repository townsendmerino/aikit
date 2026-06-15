module github.com/townsendmerino/aikit

go 1.26.3

require golang.org/x/text v0.37.0

// v1.8.0 shipped before the release gate passed (missing CHANGELOG compare link)
// and without the GGUF nested-array parser hardening — superseded by v1.8.1.
retract v1.8.0
