module github.com/townsendmerino/aikit/gpu/qwencuda

go 1.27.0

require (
	github.com/townsendmerino/aikit v1.33.0
	github.com/townsendmerino/aikit/gpu v0.32.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	github.com/eitamring/gocudrv v0.3.2 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.40.0 // indirect
)

replace github.com/townsendmerino/aikit => ../../

replace github.com/townsendmerino/aikit/gpu => ../
