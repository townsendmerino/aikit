// matmul_f32.wgsl — 2D workgroup-tiled matrix multiplication with 2x2 register micro-tiling for WebGPU.
// Computes C[M,N] = A[M,K] · B[N,K]ᵀ (MatmulBT contract: B row-major [N,K]).
//
// 16×16 threads per workgroup computing a 32×32 output tile (2×2 outputs per thread).
// Per K-strip of 16, the 256 threads cooperatively stage A[32×16] and B[32×16] into workgroup
// memory, barrier, and accumulate 4 outer-product outputs in registers.

struct Dims {
    m:    u32,
    k:    u32,
    n:    u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       A:    array<f32>;
@group(0) @binding(1) var<storage, read>       B:    array<f32>;
@group(0) @binding(2) var<storage, read_write> C:    array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u;

var<workgroup> As: array<array<f32, 16>, 32>;
var<workgroup> Bs: array<array<f32, 16>, 32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx; // 0..255

    let M = dims.m;
    let K = dims.k;
    let N = dims.n;

    let m0 = wgid.y * TILE_M;
    let n0 = wgid.x * TILE_N;

    // 2x2 micro-tile outputs per thread
    var acc00: f32 = 0.0;
    var acc01: f32 = 0.0;
    var acc10: f32 = 0.0;
    var acc11: f32 = 0.0;

    // Each thread loads 2 elements into As [32, 16] and 2 elements into Bs [32, 16]
    let loadRow0 = tid / 16u;        // 0..15
    let loadRow1 = loadRow0 + 16u;   // 16..31
    let loadCol  = tid % 16u;        // 0..15

    for (var k0: u32 = 0u; k0 < K; k0 += TILE_K) {
        let kA = k0 + loadCol;

        // Stage As
        let globA0 = m0 + loadRow0;
        if (globA0 < M && kA < K) {
            As[loadRow0][loadCol] = A[globA0 * K + kA];
        } else {
            As[loadRow0][loadCol] = 0.0;
        }

        let globA1 = m0 + loadRow1;
        if (globA1 < M && kA < K) {
            As[loadRow1][loadCol] = A[globA1 * K + kA];
        } else {
            As[loadRow1][loadCol] = 0.0;
        }

        // Stage Bs
        let globB0 = n0 + loadRow0;
        if (globB0 < N && kA < K) {
            Bs[loadRow0][loadCol] = B[globB0 * K + kA];
        } else {
            Bs[loadRow0][loadCol] = 0.0;
        }

        let globB1 = n0 + loadRow1;
        if (globB1 < N && kA < K) {
            Bs[loadRow1][loadCol] = B[globB1 * K + kA];
        } else {
            Bs[loadRow1][loadCol] = 0.0;
        }

        workgroupBarrier();

        // 2x2 outer-product accumulation
        let r0 = ty * 2u;
        let r1 = r0 + 1u;
        let c0 = tx * 2u;
        let c1 = c0 + 1u;

        for (var kk: u32 = 0u; kk < TILE_K; kk += 1u) {
            let a0 = As[r0][kk];
            let a1 = As[r1][kk];
            let b0 = Bs[c0][kk];
            let b1 = Bs[c1][kk];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        workgroupBarrier();
    }

    let outR0 = m0 + ty * 2u;
    let outR1 = outR0 + 1u;
    let outC0 = n0 + tx * 2u;
    let outC1 = outC0 + 1u;

    if (outR0 < M && outC0 < N) {
        C[outR0 * N + outC0] = acc00;
    }
    if (outR0 < M && outC1 < N) {
        C[outR0 * N + outC1] = acc01;
    }
    if (outR1 < M && outC0 < N) {
        C[outR1 * N + outC0] = acc10;
    }
    if (outR1 < M && outC1 < N) {
        C[outR1 * N + outC1] = acc11;
    }
}
