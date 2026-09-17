// layernorm.wgsl — workgroup-parallel mean/variance LayerNorm in WGSL.
// Computes out[row, d] = ((x[row, d] - mean) / sqrt(variance + eps)) * w[d] + b[d].
//
// Dispatched with 1 workgroup per row, 256 threads per workgroup.

struct Params {
    rows: u32,
    dim:  u32,
    eps:  f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       w:      array<f32>;
@group(0) @binding(2) var<storage, read>       b:      array<f32>;
@group(0) @binding(3) var<storage, read_write> out:    array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> smemSum: array<f32, 256>;
var<workgroup> smemSq:  array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    let r = wgid.x;
    if (r >= params.rows) {
        return;
    }
    let tid = lid.x;
    let dim = params.dim;
    let rOffset = r * dim;

    // Pass 1: Fused Mean & Variance reduction (single global read pass)
    var sumVal: f32 = 0.0;
    var sumSqVal: f32 = 0.0;
    for (var i: u32 = tid; i < dim; i += WG_SIZE) {
        let v = x[rOffset + i];
        sumVal += v;
        sumSqVal += v * v;
    }
    smemSum[tid] = sumVal;
    smemSq[tid]  = sumSqVal;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            smemSum[tid] += smemSum[tid + s];
            smemSq[tid]  += smemSq[tid + s];
        }
        workgroupBarrier();
    }
    let mean: f32 = smemSum[0] / f32(dim);
    let meanSq: f32 = smemSq[0] / f32(dim);
    let variance: f32 = max(0.0, meanSq - mean * mean);
    let invStd: f32 = inverseSqrt(variance + params.eps);
    workgroupBarrier();

    // Pass 2: Normalize and scale/bias writeback
    for (var i: u32 = tid; i < dim; i += WG_SIZE) {
        out[rOffset + i] = ((x[rOffset + i] - mean) * invStd) * w[i] + b[i];
    }
}

