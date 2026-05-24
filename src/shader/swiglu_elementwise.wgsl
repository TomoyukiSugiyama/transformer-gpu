const SIZE: u32 = 256u;

struct Dims {
    size: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(SIZE, 1, 1)
fn swiglu_elementwise(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= dims.size { return; }
    let g= gate[idx];
    out[idx] = g/(1 + exp(-g)) * up[idx];
}