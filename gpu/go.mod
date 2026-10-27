// Frozen on the v0.35.x series: fixes only. It serves gpu/anncuda, gpu/annmetal, gpu/enccuda and gpu/encmetal; the
// device code goinfer uses moved into goinfer (cuda/internal/gpudev, metal/internal/gpudev).
module github.com/townsendmerino/aikit/gpu

go 1.27.0

require github.com/ebitengine/purego v0.10.1

require github.com/eitamring/gocudrv v0.3.2
