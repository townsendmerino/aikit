package decoder

import (
	"encoding/json"
	"fmt"
	"math"
)

// RoPE frequency scaling (multi-model-plan G4). HF's rope_scaling object
// transforms the base inverse-frequency table so a model trained at one context
// length serves a longer one. Only the variants whose math this file implements
// load; the rest (yarn, longrope/su, dynamic) are rejected loudly by the
// adapter rather than loaded with the wrong positional frequencies.

// ropeScalingKind enumerates the rope_scaling.rope_type values this build
// supports. "none" is the implicit default (no rope_scaling object).
type ropeScalingKind int

const (
	ropeScaleNone   ropeScalingKind = iota
	ropeScaleLinear                 // positions divided by factor (inv_freq /= factor)
	ropeScaleLlama3                 // Llama-3.1+/3.2 piecewise wavelength interpolation
)

// ropeScaling holds the resolved scaling parameters. Only the fields the active
// kind needs are populated.
type ropeScaling struct {
	kind   ropeScalingKind
	factor float64 // linear + llama3

	// llama3 only:
	lowFreqFactor   float64
	highFreqFactor  float64
	origMaxPosition float64
}

// parseRopeScaling reads config.json's rope_scaling object. A null/empty object
// means no scaling (returns nil, nil). An object whose rope_type this build
// doesn't implement is an error — the caller surfaces it so the load fails
// loudly instead of producing wrong frequencies. HF has used both "rope_type"
// and the legacy "type" key; both are accepted.
func parseRopeScaling(raw json.RawMessage) (*ropeScaling, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var obj struct {
		RopeType        string  `json:"rope_type"`
		Type            string  `json:"type"`
		Factor          float64 `json:"factor"`
		LowFreqFactor   float64 `json:"low_freq_factor"`
		HighFreqFactor  float64 `json:"high_freq_factor"`
		OrigMaxPosition float64 `json:"original_max_position_embeddings"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("rope_scaling: %w", err)
	}
	kind := obj.RopeType
	if kind == "" {
		kind = obj.Type
	}
	switch kind {
	case "linear":
		if obj.Factor <= 0 {
			return nil, fmt.Errorf("rope_scaling(linear): factor must be >0, got %v", obj.Factor)
		}
		return &ropeScaling{kind: ropeScaleLinear, factor: obj.Factor}, nil
	case "llama3":
		if obj.Factor <= 0 || obj.HighFreqFactor == obj.LowFreqFactor || obj.OrigMaxPosition <= 0 {
			return nil, fmt.Errorf("rope_scaling(llama3): bad params factor=%v low=%v high=%v origMax=%v",
				obj.Factor, obj.LowFreqFactor, obj.HighFreqFactor, obj.OrigMaxPosition)
		}
		return &ropeScaling{
			kind: ropeScaleLlama3, factor: obj.Factor,
			lowFreqFactor: obj.LowFreqFactor, highFreqFactor: obj.HighFreqFactor,
			origMaxPosition: obj.OrigMaxPosition,
		}, nil
	default:
		return nil, fmt.Errorf("rope_scaling rope_type=%q unsupported (have: linear, llama3; yarn/longrope/dynamic are a follow-up)", kind)
	}
}

// computeInvFreq builds the per-dimension inverse-frequency table RoPE rotates
// with: invFreq[d] = base^(-2d/rotaryDim) for d in [0, rotaryDim/2), with any
// scaling transform applied. rotaryDim is the number of rotated dims (== head
// dim for full rotary, < head dim for Phi's partial_rotary_factor).
func computeInvFreq(base float64, rotaryDim int, sc *ropeScaling) []float64 {
	half := rotaryDim / 2
	inv := make([]float64, half)
	for d := 0; d < half; d++ {
		inv[d] = math.Pow(base, -float64(2*d)/float64(rotaryDim))
	}
	if sc == nil {
		return inv
	}
	switch sc.kind {
	case ropeScaleLinear:
		for d := range inv {
			inv[d] /= sc.factor
		}
	case ropeScaleLlama3:
		applyLlama3Scaling(inv, sc)
	}
	return inv
}

// applyLlama3Scaling reproduces HF's _compute_llama3_parameters: split the
// frequency band by wavelength into a high-frequency part (kept), a
// low-frequency part (divided by factor), and a middle part (smoothly
// interpolated between the two). Wavelength = 2π / inv_freq.
func applyLlama3Scaling(inv []float64, sc *ropeScaling) {
	lowWavelen := sc.origMaxPosition / sc.lowFreqFactor   // long wavelength threshold
	highWavelen := sc.origMaxPosition / sc.highFreqFactor // short wavelength threshold
	for d, f := range inv {
		wavelen := 2 * math.Pi / f
		switch {
		case wavelen < highWavelen:
			// high frequency: unchanged
		case wavelen > lowWavelen:
			inv[d] = f / sc.factor
		default:
			smooth := (sc.origMaxPosition/wavelen - sc.lowFreqFactor) / (sc.highFreqFactor - sc.lowFreqFactor)
			inv[d] = (1-smooth)*(f/sc.factor) + smooth*f
		}
	}
}
