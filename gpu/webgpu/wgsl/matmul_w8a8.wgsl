// matmul_w8a8.wgsl — 2D workgroup-tiled int8 quantized GEMM with 2x2 micro-tiling for WebGPU.
// Computes C[M,N] = (A_i8[M,K] · B_i8[N,K]ᵀ) * aScale[M] * bScale[N].
//
// Stored as packed 4-byte words (u32), unpacked into 4 signed int8 lanes per word.
// 16×16 threads per workgroup computing a 32×32 output tile (2×2 outputs per thread).
// Accumulates exactly in 32-bit signed integer math (i32), matching CPU linalg.MatmulBTW8A8.

struct Dims {
    m:    u32,
    k:    u32,
    n:    u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       A_packed: array<u32>;
@group(0) @binding(1) var<storage, read>       B_packed: array<u32>;
@group(0) @binding(2) var<storage, read>       aScale:   array<f32>;
@group(0) @binding(3) var<storage, read>       bScale:   array<f32>;
@group(0) @binding(4) var<storage, read_write> C:        array<f32>;
@group(0) @binding(5) var<uniform>             dims:     Dims;

const TILE_M: u32 = 32u;
const TILE_N: u32 = 32u;
const TILE_K: u32 = 16u; // 16 words = 64 elements of K

// Unpack a 32-bit word containing 4 little-endian signed int8 values.
fn unpack4(w: u32) -> vec4<i32> {
    return vec4<i32>(
        (i32(w) << 24) >> 24,
        (i32(w) << 16) >> 24,
        (i32(w) << 8) >> 24,
        i32(w) >> 24
    );
}

var<workgroup> As: array<array<u32, 16>, 32>;
var<workgroup> Bs: array<array<u32, 16>, 32>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx; // 0..255

    let M = dims.m;
    let N = dims.n;
    let KWords = (dims.k + 3u) / 4u;

    let m0 = wgid.y * TILE_M;
    let n0 = wgid.x * TILE_N;

    var acc00: i32 = 0;
    var acc01: i32 = 0;
    var acc10: i32 = 0;
    var acc11: i32 = 0;

    let loadRow0 = tid / 16u;
    let loadRow1 = loadRow0 + 16u;
    let loadCol  = tid % 16u;

    for (var k0: u32 = 0u; k0 < KWords; k0 += TILE_K) {
        let kA = k0 + loadCol;

        let globA0 = m0 + loadRow0;
        if (globA0 < M && kA < KWords) {
            As[loadRow0][loadCol] = A_packed[globA0 * KWords + kA];
        } else {
            As[loadRow0][loadCol] = 0u;
        }

        let globA1 = m0 + loadRow1;
        if (globA1 < M && kA < KWords) {
            As[loadRow1][loadCol] = A_packed[globA1 * KWords + kA];
        } else {
            As[loadRow1][loadCol] = 0u;
        }

        let globB0 = n0 + loadRow0;
        if (globB0 < N && kA < KWords) {
            Bs[loadRow0][loadCol] = B_packed[globB0 * KWords + kA];
        } else {
            Bs[loadRow0][loadCol] = 0u;
        }

        let globB1 = n0 + loadRow1;
        if (globB1 < N && kA < KWords) {
            Bs[loadRow1][loadCol] = B_packed[globB1 * KWords + kA];
        } else {
            Bs[loadRow1][loadCol] = 0u;
        }

        workgroupBarrier();

        let r0 = ty * 2u;
        let r1 = r0 + 1u;
        let c0 = tx * 2u;
        let c1 = c0 + 1u;

        for (var kk: u32 = 0u; kk < TILE_K; kk += 1u) {
            let aVec0 = unpack4(As[r0][kk]);
            let aVec1 = unpack4(As[r1][kk]);
            let bVec0 = unpack4(Bs[c0][kk]);
            let bVec1 = unpack4(Bs[c1][kk]);

            acc00 += dot(aVec0, bVec0);
            acc01 += dot(aVec0, bVec1);
            acc10 += dot(aVec1, bVec0);
            acc11 += dot(aVec1, bVec1);
        }

        workgroupBarrier();
    }

    let outR0 = m0 + ty * 2u;
    let outR1 = outR0 + 1u;
    let outC0 = n0 + tx * 2u;
    let outC1 = outC0 + 1u;

    if (outR0 < M && outC0 < N) {
        C[outR0 * N + outC0] = f32(acc00) * aScale[outR0] * bScale[outC0];
    }
    if (outR0 < M && outC1 < N) {
        C[outR0 * N + outC1] = f32(acc01) * aScale[outR0] * bScale[outC1];
    }
    if (outR1 < M && outC0 < N) {
        C[outR1 * N + outC0] = f32(acc10) * aScale[outR1] * bScale[outC0];
    }
    if (outR1 < M && outC1 < N) {
        C[outR1 * N + outC1] = f32(acc11) * aScale[outR1] * bScale[outC1];
    }
}
