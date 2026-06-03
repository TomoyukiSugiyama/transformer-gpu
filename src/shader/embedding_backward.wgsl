const SIZE: u32 = 256u;

struct Dims {
    d_model: u32,
    seq_len: u32,
}

@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> dweight: array<atomic<i32>>;
@group(0) @binding(3) var<uniform> dims: Dims;

fn atomic_add_f32(dst: u32, val: f32) {
    loop {
        let old_i = atomicLoad(&dweight[dst]);
        let new_i = bitcast<i32>(bitcast<f32>(old_i) + val);
        let r = atomicCompareExchangeWeak(&dweight[dst], old_i, new_i);
        if r.exchanged { break; }
    }
}

@compute @workgroup_size(SIZE, 1, 1)
fn embedding_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let d = dims.d_model;
    let s = dims.seq_len;
    let total    = s * d;
    if idx >= total { return; }
    
    let pos = idx / d;
    let j = idx % d;
    let id = token_ids[pos];
    let dst = id * d + j;
    
    let val = dy[idx];
    // atomicAdd(&dweight[dst], bitcast<i32>(val));

    atomic_add_f32(
        dst,
        dy[idx]
    );
}