// attention.wgsl — query-tiled online-softmax (FlashAttention-style) multi-head attention in WGSL.
// Computes out[i, h, :] = softmax( (q[i, h, :] · k[:, h, :]ᵀ) * scale ) · v[:, h, :]
//
// Key Advantages:
// 1. Zero dynamic/global scratch memory for scores: eliminates the large O(nH * np^2) Scores buffer.
// 2. K and V tiles (16 rows each) are staged cooperatively in workgroup memory.
// 3. Staging K and V once amortizes global VRAM reads across all queries in the workgroup tile.
// 4. Memory traffic drops from O(np^2) global reads/writes to O(np * hd).
//
// Workgroup geometry:
// - Workgroup size: 32 threads (1 warp / 1 wavefront), one query per thread.
// - Grid size: nH * ((np + 31) / 32) workgroups.

struct AttnParams {
    np:     u32, // sequence length (patch count or token count)
    nH:     u32, // number of heads
    hd:     u32, // head dimension
    scale:  f32, // 1.0 / sqrt(hd)
};

@group(0) @binding(0) var<storage, read>       Q:      array<f32>;
@group(0) @binding(1) var<storage, read>       K:      array<f32>;
@group(0) @binding(2) var<storage, read>       V:      array<f32>;
@group(0) @binding(3) var<storage, read_write> Out:    array<f32>;
@group(0) @binding(4) var<uniform>             params: AttnParams;

const QTILE: u32 = 32u;
const KTILE: u32 = 16u;
const MAX_HD: u32 = 128u;

var<workgroup> Ks: array<array<f32, 128>, 16>;
var<workgroup> Vs: array<array<f32, 128>, 16>;

@compute @workgroup_size(32, 1, 1)
fn main(
    @builtin(workgroup_id)        wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>
) {
    let np = params.np;
    let nH = params.nH;
    let hd = params.hd;
    let scale = params.scale;

    let tilesPerHead = (np + QTILE - 1u) / QTILE;
    let blk = wgid.x;
    if (blk >= nH * tilesPerHead) {
        return;
    }

    let h = blk / tilesPerHead;
    let tileIdx = blk % tilesPerHead;
    let qStart = tileIdx * QTILE;
    let tid = lid.x;
    let i = qStart + tid;

    let hidden = nH * hd;
    let off = h * hd;
    let active = (i < np);

    var m: f32 = -3.402823466e+38;
    var l: f32 = 0.0;
    var acc: array<f32, 128>;
    for (var d: u32 = 0u; d < MAX_HD; d += 1u) {
        acc[d] = 0.0;
    }

    for (var k0: u32 = 0u; k0 < np; k0 += KTILE) {
        let chunk = min(KTILE, np - k0);

        // Cooperative staging of K and V into workgroup memory
        let totalElements = chunk * hd;
        for (var idx = tid; idx < totalElements; idx += 32u) {
            let kk = idx / hd;
            let d = idx % hd;
            let kj = k0 + kk;
            Ks[kk][d] = K[kj * hidden + off + d];
            Vs[kk][d] = V[kj * hidden + off + d];
        }
        workgroupBarrier();

        if (active) {
            let qOffset = i * hidden + off;
            var s: array<f32, 16>;
            var blockMax: f32 = -3.402823466e+38;

            for (var kk: u32 = 0u; kk < chunk; kk += 1u) {
                var dotVal: f32 = 0.0;
                for (var d: u32 = 0u; d < hd; d += 1u) {
                    dotVal += Q[qOffset + d] * Ks[kk][d];
                }
                let sv = dotVal * scale;
                s[kk] = sv;
                if (sv > blockMax) {
                    blockMax = sv;
                }
            }

            let newMax = max(m, blockMax);
            let corr = exp(m - newMax);
            l *= corr;
            for (var d: u32 = 0u; d < hd; d += 1u) {
                acc[d] *= corr;
            }

            var blockSum: f32 = 0.0;
            for (var kk: u32 = 0u; kk < chunk; kk += 1u) {
                let expVal = exp(s[kk] - newMax);
                blockSum += expVal;
                for (var d: u32 = 0u; d < hd; d += 1u) {
                    acc[d] += expVal * Vs[kk][d];
                }
            }

            l += blockSum;
            m = newMax;
        }
        workgroupBarrier();
    }

    if (active) {
        let outOffset = i * hidden + off;
        let inv = 1.0 / l;
        for (var d: u32 = 0u; d < hd; d += 1u) {
            Out[outOffset + d] = acc[d] * inv;
        }
    }
}
