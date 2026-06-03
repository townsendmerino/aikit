// Command gemma is a demo CLI that runs a local Gemma 3 (270M / 1B)
// checkpoint through aikit's pure-Go decoder and streams the completion to
// stdout.
//
// Status: SCAFFOLD. The wiring (flags → tokenizer.Load → decoder.Load →
// Generate → stream) is complete and compiles; the underlying forward pass,
// tokenizer and BF16 loader are stubbed per docs/gemma-decoder-plan.md, so
// the program runs and reports exactly which milestone is outstanding rather
// than producing tokens. This is the harness the milestones fill in.
//
// Usage:
//
//	go run ./demo/gemma --model ~/models/gemma-3-270m --prompt "Hello, world"
//	go run ./demo/gemma --model ~/models/gemma-3-1b --prompt "..." --max 128 --temp 0.7 --backend cpu
//
// Get a checkpoint (HF layout: config.json + model.safetensors +
// tokenizer.model/tokenizer.json):
//
//	huggingface-cli download google/gemma-3-270m --local-dir ~/models/gemma-3-270m
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/townsendmerino/aikit/decoder"
	"github.com/townsendmerino/aikit/tokenizer"
)

func main() {
	var (
		modelDir = flag.String("model", "", "path to a Gemma 3 checkpoint dir (config.json + model.safetensors + tokenizer)")
		prompt   = flag.String("prompt", "Hello, world", "prompt text")
		maxTok   = flag.Int("max", 128, "max tokens to generate")
		temp     = flag.Float64("temp", 0.0, "sampling temperature (0 = greedy)")
		topK     = flag.Int("top-k", 0, "top-k filter (0 = off)")
		topP     = flag.Float64("top-p", 0.0, "top-p / nucleus (0 = off)")
		seed     = flag.Int64("seed", 0, "sampling RNG seed")
		backend  = flag.String("backend", "cpu", "compute backend: cpu | webgpu")
	)
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, "error: --model is required (path to a Gemma 3 checkpoint dir)")
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*modelDir, *prompt, *maxTok, *backend, decoder.SamplingParams{
		Temperature: *temp,
		TopK:        *topK,
		TopP:        *topP,
		Seed:        *seed,
	}); err != nil {
		fmt.Fprintf(os.Stderr, "\n%v\n", err)
		// A scaffold/NYI error is expected today — exit 1, not a crash.
		os.Exit(1)
	}
}

func run(modelDir, prompt string, maxTok int, backend string, sp decoder.SamplingParams) error {
	// 1) Tokenizer.
	tk, err := tokenizer.Load(filepath.Join(modelDir))
	if err != nil {
		return fmt.Errorf("load tokenizer: %w", err)
	}

	// 2) Model + backend.
	t0 := time.Now()
	model, err := decoder.Load(modelDir, decoder.Options{Backend: backend})
	if err != nil {
		return fmt.Errorf("load model: %w", err)
	}
	cfg := model.Config()
	fmt.Fprintf(os.Stderr, "loaded %d-layer model (hidden %d, vocab %d) in %s [backend=%s]\n",
		cfg.NumLayers, cfg.HiddenDim, cfg.VocabSize, time.Since(t0).Round(time.Millisecond), backend)

	// 3) Encode prompt.
	ids, err := tk.Encode(prompt, true /* addBOS */)
	if err != nil {
		return fmt.Errorf("encode prompt: %w", err)
	}

	// 4) Generate + stream.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	fmt.Printf("%s", prompt)
	stream, gen := model.Generate(ctx, ids, maxTok, sp)
	for id := range stream {
		piece, derr := tk.DecodePiece(id)
		if derr != nil {
			return fmt.Errorf("decode token %d: %w", id, derr)
		}
		fmt.Print(piece)
	}
	fmt.Println()
	return gen.Err()
}
