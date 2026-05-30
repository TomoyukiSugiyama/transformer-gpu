const SIZE: u32 = 256u;

struct Dims {
    size: u32,
}

@group(0) @binding(0) var<storage, read>       dy:    array<f32>;  // (n,)
@group(0) @binding(1) var<storage, read>       gate:  array<f32>;  // (n,)
@group(0) @binding(2) var<storage, read>       up:    array<f32>;  // (n,)
@group(0) @binding(3) var<storage, read_write> d_out: array<f32>;  // (2n,): [d_gate | d_up]
@group(0) @binding(4) var<uniform>             dims:  Dims;

@compute @workgroup_size(SIZE, 1, 1)
fn swiglu_elementwise_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= dims.size { return; }

    let g   = gate[idx];
    let sig = 1.0 / (1.0 + exp(-g));       // σ(gate)
    let swish_g     = g * sig;              // swish(gate)
    let swish_prime = sig * (1.0 + g * (1.0 - sig));  // swish'(gate)

    // d_gate[idx] = dy * up * swish'(gate)
    d_out[idx]              = dy[idx] * up[idx] * swish_prime;
    // d_up[idx]   = dy * swish(gate)
    d_out[dims.size + idx]  = dy[idx] * swish_g;
}