// gelu.wgsl — vectorized tanh-GELU activation in WGSL.
// Computes x[i] = 0.5 * x[i] * (1.0 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3))).
//
// Operates on vec4<f32> for 128-bit memory transactions.

struct Params {
    n_vec4: u32,
    n_tail: u32,
    _pad0:  u32,
    _pad1:  u32,
};

@group(0) @binding(0) var<storage, read_write> data:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tail:   array<f32>;
@group(0) @binding(2) var<uniform>             params: Params;

const C: f32 = 0.7978845608028654; // sqrt(2/pi)

fn gelu_scalar(v: f32) -> f32 {
    return 0.5 * v * (1.0 + tanh(C * (v + 0.044715 * v * v * v)));
}

fn gelu_vec4(v: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        gelu_scalar(v.x),
        gelu_scalar(v.y),
        gelu_scalar(v.z),
        gelu_scalar(v.w)
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    if (idx < params.n_vec4) {
        data[idx] = gelu_vec4(data[idx]);
    } else {
        let tailIdx = idx - params.n_vec4;
        if (tailIdx < params.n_tail) {
            tail[tailIdx] = gelu_scalar(tail[tailIdx]);
        }
    }
}

