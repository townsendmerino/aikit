//go:build gpu

package gpu

import (
	"fmt"

	"github.com/cogentcore/webgpu/wgpu"
)

// matmulShaderWGSL computes dst[m,n] = Σ_k a[m,k]·b[n,k], i.e. dst =
// a·bᵀ — the encoder's matmulBT contract: a is [M,K] row-major, b is
// [N,K] row-major (the PyTorch [out,in] weight layout, so no transpose
// is needed), dst is [M,N] row-major.
//
// One invocation per output element, 16×16 workgroup. This is the
// NAIVE kernel — no shared-memory tiling, every invocation streams a
// full a-row and b-row from global memory. It is correct and proves
// the whole upload/dispatch/readback pipeline; a tiled kernel that
// stages K-strips into workgroup memory is the throughput follow-up
// (see docs/cpu-acceleration.md).
const matmulShaderWGSL = `
struct Dims { m: u32, k: u32, n: u32, _pad: u32 };

@group(0) @binding(0) var<storage, read>       a:    array<f32>;
@group(0) @binding(1) var<storage, read>       b:    array<f32>;
@group(0) @binding(2) var<storage, read_write> dst:  array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= dims.m || col >= dims.n) {
        return;
    }
    let K = dims.k;
    let aBase = row * K;
    let bBase = col * K;
    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < K; i = i + 1u) {
        acc = acc + a[aBase + i] * b[bBase + i];
    }
    dst[row * dims.n + col] = acc;
}
`

// Context holds the persistent WebGPU objects (device, queue, compiled
// pipeline). Create one with New and reuse it across MatmulBT calls —
// device/pipeline creation is expensive, per-call buffer allocation is
// not. Not safe for concurrent use by multiple goroutines; wrap in your
// own mutex or use one Context per worker if you need that.
type Context struct {
	instance *wgpu.Instance
	adapter  *wgpu.Adapter
	device   *wgpu.Device
	queue    *wgpu.Queue
	shader   *wgpu.ShaderModule
	pipeline *wgpu.ComputePipeline
	layout   *wgpu.BindGroupLayout
}

// New initializes a GPU context: instance → adapter (high-performance
// preference) → device → compiled matmul pipeline. Returns an error if
// no adapter/device is available (e.g. a headless box with no GPU), so
// callers can fall back to the CPU path or skip GPU tests cleanly.
func New() (*Context, error) {
	inst := wgpu.CreateInstance(nil)
	adapter, err := inst.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		inst.Release()
		return nil, fmt.Errorf("gpu: request adapter: %w", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: request device: %w", err)
	}
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
		Label:          "matmulBT",
		WGSLDescriptor: &wgpu.ShaderModuleWGSLDescriptor{Code: matmulShaderWGSL},
	})
	if err != nil {
		device.Release()
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: compile shader: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label:   "matmulBT",
		Compute: wgpu.ProgrammableStageDescriptor{Module: shader, EntryPoint: "main"},
		// Layout nil ⇒ auto layout inferred from the shader bindings.
	})
	if err != nil {
		shader.Release()
		device.Release()
		adapter.Release()
		inst.Release()
		return nil, fmt.Errorf("gpu: create pipeline: %w", err)
	}
	return &Context{
		instance: inst,
		adapter:  adapter,
		device:   device,
		queue:    device.GetQueue(),
		shader:   shader,
		pipeline: pipeline,
		layout:   pipeline.GetBindGroupLayout(0),
	}, nil
}

// Backend reports the underlying graphics backend ("Metal", "Vulkan",
// "D3D12", …) — useful for test logging to confirm which GPU API is in
// play.
func (c *Context) Backend() string {
	return c.adapter.GetInfo().BackendType.String()
}

// Close releases all GPU resources. Safe to call once; the Context must
// not be used afterward.
func (c *Context) Close() {
	c.pipeline.Release()
	c.shader.Release()
	c.queue.Release()
	c.device.Release()
	c.adapter.Release()
	c.instance.Release()
}

// MatmulBT computes dst = a·bᵀ on the GPU and returns the [M,N] result.
// a must hold ≥ M*K f32s, b ≥ N*K. This allocates fresh GPU buffers,
// uploads a and b, dispatches the kernel, and reads the result back —
// so it pays full host↔device transfer on every call. That is the
// foundation's known cost; resident weights/activations are the
// follow-up that makes the GPU actually win.
func (c *Context) MatmulBT(a, b []float32, M, K, N int) ([]float32, error) {
	if M <= 0 || K <= 0 || N <= 0 {
		return nil, fmt.Errorf("gpu: matmulBT non-positive dim M=%d K=%d N=%d", M, K, N)
	}
	if len(a) < M*K || len(b) < N*K {
		return nil, fmt.Errorf("gpu: matmulBT input too small: len(a)=%d need %d, len(b)=%d need %d",
			len(a), M*K, len(b), N*K)
	}
	dstSize := uint64(M * N * 4)

	aBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "a", Contents: wgpu.ToBytes(a[:M*K]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create a buffer: %w", err)
	}
	defer aBuf.Release()

	bBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "b", Contents: wgpu.ToBytes(b[:N*K]), Usage: wgpu.BufferUsageStorage,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create b buffer: %w", err)
	}
	defer bBuf.Release()

	dstBuf, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "dst", Size: dstSize,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create dst buffer: %w", err)
	}
	defer dstBuf.Release()

	// Dims uniform: 3 u32 + 1 pad word (uniform buffers need 16-byte size).
	dims := []uint32{uint32(M), uint32(K), uint32(N), 0}
	dimsBuf, err := c.device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label: "dims", Contents: wgpu.ToBytes(dims), Usage: wgpu.BufferUsageUniform,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create dims buffer: %w", err)
	}
	defer dimsBuf.Release()

	stage, err := c.device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "stage", Size: dstSize,
		Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create staging buffer: %w", err)
	}
	defer stage.Release()

	bindGroup, err := c.device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Layout: c.layout,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: aBuf, Size: aBuf.GetSize()},
			{Binding: 1, Buffer: bBuf, Size: bBuf.GetSize()},
			{Binding: 2, Buffer: dstBuf, Size: dstBuf.GetSize()},
			{Binding: 3, Buffer: dimsBuf, Size: dimsBuf.GetSize()},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("gpu: create bind group: %w", err)
	}
	defer bindGroup.Release()

	enc, err := c.device.CreateCommandEncoder(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: create command encoder: %w", err)
	}
	defer enc.Release()

	pass := enc.BeginComputePass(nil)
	pass.SetPipeline(c.pipeline)
	pass.SetBindGroup(0, bindGroup, nil)
	// global_invocation_id.x ranges over rows (M), .y over cols (N);
	// 16×16 threads per workgroup, so ceil-divide the counts.
	pass.DispatchWorkgroups((uint32(M)+15)/16, (uint32(N)+15)/16, 1)
	if err := pass.End(); err != nil {
		pass.Release()
		return nil, fmt.Errorf("gpu: end compute pass: %w", err)
	}
	pass.Release()

	if err := enc.CopyBufferToBuffer(dstBuf, 0, stage, 0, dstSize); err != nil {
		return nil, fmt.Errorf("gpu: copy dst→stage: %w", err)
	}
	cmd, err := enc.Finish(nil)
	if err != nil {
		return nil, fmt.Errorf("gpu: finish encoder: %w", err)
	}
	defer cmd.Release()
	c.queue.Submit(cmd)

	// Map the staging buffer and block until the GPU work + map complete.
	mapStatus := wgpu.BufferMapAsyncStatusUnknown
	if err := stage.MapAsync(wgpu.MapModeRead, 0, dstSize, func(s wgpu.BufferMapAsyncStatus) {
		mapStatus = s
	}); err != nil {
		return nil, fmt.Errorf("gpu: map async: %w", err)
	}
	c.device.Poll(true, nil) // wait=true: flush queue + fire map callback
	if mapStatus != wgpu.BufferMapAsyncStatusSuccess {
		return nil, fmt.Errorf("gpu: staging map failed: %v", mapStatus)
	}

	raw := stage.GetMappedRange(0, uint(dstSize))
	out := make([]float32, M*N)
	copy(out, wgpu.FromBytes[float32](raw))
	if err := stage.Unmap(); err != nil {
		return nil, fmt.Errorf("gpu: unmap staging: %w", err)
	}
	return out, nil
}
