package webgpu_test

import (
	"strings"
	"testing"

	"github.com/townsendmerino/aikit/gpu/webgpu"
)

func TestWebGPU_WGSLShadersNonEmpty(t *testing.T) {
	shaders := map[string]string{
		"matmul_f32":  webgpu.MatmulF32WGSL,
		"matmul_w8a8": webgpu.MatmulW8A8WGSL,
		"layernorm":   webgpu.LayerNormWGSL,
		"rmsnorm":     webgpu.RMSNormWGSL,
		"gelu":        webgpu.GELUWGSL,
		"silu":        webgpu.SiLUWGSL,
		"attention":   webgpu.AttentionWGSL,
	}

	for name, src := range shaders {
		if len(strings.TrimSpace(src)) == 0 {
			t.Errorf("shader %s is empty", name)
		}
		if !strings.Contains(src, "@compute") {
			t.Errorf("shader %s missing @compute entry point", name)
		}
	}
}

func TestWebGPU_AttentionOnlineSoftmaxStructure(t *testing.T) {
	src := webgpu.AttentionWGSL
	// Attention shader must use online softmax and NOT require an external Scores buffer.
	if strings.Contains(src, "Scores:") {
		t.Errorf("attention.wgsl should not have an external Scores storage buffer")
	}
	if !strings.Contains(src, "var<workgroup> Ks:") || !strings.Contains(src, "var<workgroup> Vs:") {
		t.Errorf("attention.wgsl missing workgroup Ks/Vs tile buffers")
	}
	if !strings.Contains(src, "corr = exp(m - newMax)") {
		t.Errorf("attention.wgsl missing online softmax correction factor")
	}
}

func TestWebGPU_MicroTilingStructure(t *testing.T) {
	f32Src := webgpu.MatmulF32WGSL
	if !strings.Contains(f32Src, "acc00") || !strings.Contains(f32Src, "acc11") {
		t.Errorf("matmul_f32.wgsl missing 2x2 micro-tile accumulators")
	}

	w8a8Src := webgpu.MatmulW8A8WGSL
	if !strings.Contains(w8a8Src, "unpack4") {
		t.Errorf("matmul_w8a8.wgsl missing unpack4 function")
	}
	if !strings.Contains(w8a8Src, "acc00") || !strings.Contains(w8a8Src, "acc11") {
		t.Errorf("matmul_w8a8.wgsl missing 2x2 micro-tile accumulators")
	}
}

func TestWebGPU_LayerNormFusedReductionStructure(t *testing.T) {
	src := webgpu.LayerNormWGSL
	if !strings.Contains(src, "smemSum") || !strings.Contains(src, "smemSq") {
		t.Errorf("layernorm.wgsl missing fused smemSum or smemSq workgroup arrays")
	}
	if !strings.Contains(src, "meanSq - mean * mean") {
		t.Errorf("layernorm.wgsl missing single-pass variance calculation")
	}
}

func TestWebGPU_RMSNormVectorizedStructure(t *testing.T) {
	src := webgpu.RMSNormWGSL
	if !strings.Contains(src, "array<vec4<f32>>") {
		t.Errorf("rmsnorm.wgsl missing vec4<f32> array binding")
	}
	if !strings.Contains(src, "dot(v, v)") {
		t.Errorf("rmsnorm.wgsl missing dot(v, v) sum of squares vectorization")
	}
}
