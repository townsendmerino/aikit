module github.com/townsendmerino/aikit/examples/gpu-ann

go 1.27.0

require (
	github.com/townsendmerino/aikit v1.34.0
	github.com/townsendmerino/aikit/gpu/anncuda v0.0.0
	github.com/townsendmerino/aikit/gpu/annmetal v0.0.0
)

require (
	github.com/ebitengine/purego v0.10.1 // indirect
	github.com/eitamring/gocudrv v0.3.2 // indirect
	github.com/townsendmerino/aikit/gpu v0.32.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/townsendmerino/aikit => ../..

replace github.com/townsendmerino/aikit/gpu => ../../gpu

replace github.com/townsendmerino/aikit/gpu/annmetal => ../../gpu/annmetal

replace github.com/townsendmerino/aikit/gpu/anncuda => ../../gpu/anncuda
