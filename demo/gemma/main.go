// Command gemma is a demo CLI that runs a local Gemma 3 (270M / 1B)
// checkpoint through aikit's pure-Go decoder and streams the completion to
// stdout.
//
// Status: working (M1–M6). Loads a real checkpoint, tokenizes the prompt,
// and streams a completion — greedy by default, or temperature / top-k /
// top-p sampling via the flags. Perf is the naive CPU backend until M7 wires
// the SIMD/parallel linalg.
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
	"unicode/utf8"

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

	// 4) Generate + stream. Decode the whole running id slice each step and
	// print only the newly-completed bytes, holding back any trailing
	// incomplete UTF-8 — a byte-fallback token is a single (possibly partial)
	// byte, so per-token DecodePiece would emit broken multibyte characters.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	fmt.Printf("%s", prompt)
	stream, gen := model.Generate(ctx, ids, maxTok, sp)
	var out []int
	printed := 0
	flush := func(final bool) error {
		text, derr := tk.Decode(out)
		if derr != nil {
			return derr
		}
		b := []byte(text)
		end := len(b)
		if !final {
			end = completeUTF8Len(b)
		}
		if end > printed {
			os.Stdout.Write(b[printed:end])
			printed = end
		}
		return nil
	}
	for id := range stream {
		out = append(out, id)
		if err := flush(false); err != nil {
			return fmt.Errorf("decode: %w", err)
		}
	}
	if err := flush(true); err != nil {
		return fmt.Errorf("decode: %w", err)
	}
	fmt.Println()
	return gen.Err()
}

// completeUTF8Len returns the length of the longest prefix of b that ends on a
// complete UTF-8 rune boundary, so a partial trailing byte-fallback sequence is
// held back until the bytes that finish the character arrive.
func completeUTF8Len(b []byte) int {
	i := 0
	for i < len(b) {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			break // incomplete trailing sequence — hold back from here
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	return i
}
