// Package gpu is aikit's cgo-free native-GPU device substrate: Metal on darwin
// (metal.go) and CUDA on linux (cuda.go), presenting the same Device / Buffer /
// Queue / Pipeline / Encoder vocabulary on both. On every other platform this
// package is empty and registers no ann.Backend, so ann.FlatI8 scores on the CPU.
//
// STATUS: FROZEN FOR FIXES, AND ONLY PART OF IT IS MAINTAINED. This module stays on
// the gpu/v0.35.x tag series and takes fixes only; nothing new is added here. The device
// code goinfer uses lives in goinfer now, which keeps its own copy.
//
// What is maintained is the surface aikit's four GPU backends use (gpu/anncuda,
// gpu/annmetal, gpu/enccuda, gpu/encmetal): opening a Device; Buffer and HostBuffer and
// the upload and download calls (Upload, UploadBatch, Download, ReadToHost, HostCopy);
// compiling a Library into a Pipeline; a Queue with Run1D, Run2D, Launch (LaunchConfig,
// Arg) and Sync; and, for the encoder backends, Device.NewViT with ViT.GEMMF32Plan. That
// is 63 of the package's 233 exported names.
//
// The other 170 are UNMAINTAINED. They are kept exported, because removing an exported
// name breaks a build pinned to a tag, but nothing in aikit, goinfer or ken calls them
// and a fix is not promised for them. They are of two kinds: the ViT tower surface the
// retired vision backends used (the Kernel* constants, most of ViT's fields, the
// attention and grid helpers), and device features goinfer used before it took its own
// copy (on Metal: residency sets, shared events, autorelease pools; on CUDA: graphs,
// events, mapped host buffers, the quantized GEMV). The name-by-name lists, one per
// platform, with the method that produced them, are in the aikit repository at
// docs/gpu-exported-names-2026-11-10/ (darwin-arm64.tsv and linux-amd64.tsv):
// https://github.com/townsendmerino/aikit/tree/main/docs/gpu-exported-names-2026-11-10
//
// THIS FILE CARRIES THE PACKAGE COMMENT AND HAS NO BUILD TAG, deliberately. It used
// to live in doc_other.go, which is tagged `!darwin && !linux` — so on the two
// platforms anyone actually builds for, the package doc was whichever of cuda.go or
// metal.go happened to be compiled, and both open with a file-level explainer rather
// than "Package gpu ...". staticcheck's ST1000 says so on both, and nobody saw it
// because golangci-lint had never been run against this module (2026-08-12).
package gpu
