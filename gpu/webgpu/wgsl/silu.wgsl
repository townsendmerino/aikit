// silu.wgsl — vectorized gated SiLU activation in WGSL.
// Computes gate[i] = (gate[i] / (1.0 + exp(-gate[i]))) * up[i].
//
// Operates on vec4<f32> for 128-bit memory transactions.

struct Params {
    n_vec4: u32,
    n_tail: u32,
    _pad0:  u32,
    _pad1:  u32,
};

@group(0) @binding(0) var<storage, read_write> gate_vec4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       up_vec4:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> gate_tail: array<f32>;
@group(0) @binding(3) var<storage, read>       up_tail:   array<f32>;
@group(0) @binding(4) var<uniform>             params:    Params;

fn silu_scalar(g: f32, u: f32) -> f32 {
    return (g / (1.0 + exp(-g))) * u;
}

fn silu_vec4(g: vec4<f32>, u: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        silu_scalar(g.x, u.x),
        silu_scalar(g.y, u.y),
        silu_scalar(g.z, u.z),
        silu_scalar(g.w, u.w)
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx < params.n_vec4) {
        gate_vec4[idx] = silu_vec4(gate_vec4[idx], up_vec4[idx]);
    } else {
        let tailIdx = idx - params.n_vec4;
        if (tailIdx < params.n_tail) {
            gate_tail[tailIdx] = silu_scalar(gate_tail[tailIdx], up_tail[tailIdx]);
        }
    }
}

