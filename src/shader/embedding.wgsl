const SIZE: u32 = 256u;

struct Dims {
    d_model: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(SIZE, 1, 1)
fn residual_add(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    let token_id = token_ids[i];
    let d = dims.d_model;

    for(var j: u32 = 0u; j < d; j++) {
        out[i * d + j] = weight[token_id * d + j];
    }
}