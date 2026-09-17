// rmsnorm.wgsl — workgroup-parallel vectorized RMS normalization in WGSL.
// Computes out[r, d] = (x[r, d] * inverseSqrt(mean(x^2) + eps)) * w[d].
//
// Operates on vec4<f32> for 128-bit coalesced memory transactions.
// Dispatched with 1 workgroup per row, 256 threads per workgroup.

struct Params {
    rows: u32,
    dim:  u32, // total float dimension (must be divisible by 4)
    eps:  f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       x:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       w:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out:    array<vec4<f32>>;
@group(0) @binding(3) var<uniform>             params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> smem: array<f32, 256>;

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
    let dim4 = dim / 4u;
    let rOffset4 = r * dim4;

    // Pass 1: Vectorized sum of squares reduction (dot(v,v) computes v.x^2+v.y^2+v.z^2+v.w^2)
    var sumSq: f32 = 0.0;
    for (var i: u32 = tid; i < dim4; i += WG_SIZE) {
        let v = x[rOffset4 + i];
        sumSq += dot(v, v);
    }
    smem[tid] = sumSq;
    workgroupBarrier();

    for (var s: u32 = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        workgroupBarrier();
    }
    let invRms: f32 = inverseSqrt(smem[0] / f32(dim) + params.eps);
    workgroupBarrier();

    // Pass 2: Vectorized normalize and weight scale writeback
    for (var i: u32 = tid; i < dim4; i += WG_SIZE) {
        out[rOffset4 + i] = (x[rOffset4 + i] * invRms) * w[i];
    }
}


