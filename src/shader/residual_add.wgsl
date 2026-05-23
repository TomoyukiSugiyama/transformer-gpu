const SIZE: u32 = 256u;

struct Dims {
    size: u32,
}

@group(0) @binding(0) var<storage, read> x1: array<f32>;
@group(0) @binding(1) var<storage, read> x2: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(SIZE)
fn residual_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = gid.x;
    if idx > dims.size { return;}
    out[idx] = x1[idx] + x2[idx];
}