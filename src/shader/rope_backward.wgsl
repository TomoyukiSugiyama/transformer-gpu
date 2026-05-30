const SIZE: u32 = 256u;

struct Dims {
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(SIZE, 1, 1)
fn rope_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pos = wid.x;
    let i = lid.x;
    let half = dims.d_head / 2u;
    if i >= half { return; }

    let base_x = pos * dims.d_head;
    let base_tbl = pos * half;

    let x0 = x[base_x + 2u * i];
    let x1 = x[base_x + 2u * i + 1];
    let c = cos_table[base_tbl + i];
    let s = sin_table[base_tbl + i];

    out[base_x + 2u * i]     = c * x0 + s * x1;
    out[base_x + 2u * i + 1] = -s * x0 + c * x1;
}