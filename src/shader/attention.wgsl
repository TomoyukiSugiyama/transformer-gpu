const TILE: u32 = 16u;

struct Dims {
    seq: u32,
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
// @group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileQ: array<array<f32, 16u>, 16u>;
var<workgroup> tileK: array<array<f32, 16u>, 16u>;

@compute @workgroup_size(TILE, TILE, 1)
fn attention(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.d_head + TILE - 1u) / TILE;
    // 1 / √d_k
    let scale = 1.0 / sqrt(f32(dims.d_head));
    // QK^T / √d_k, casual mask
    for(var t: u32 = 0u; t < num_tiles; t++) {
        let q_col = t * TILE + lid.x;
        let k_row = t * TILE + lid.y;
        
        tileQ[lid.y][lid.x] = select(0.0, Q[row * dims.d_head + q_col], row < dims.seq && q_col < dims.d_head);
        tileK[lid.y][lid.x] = select(0.0, K[col * dims.d_head + k_row], k_row < dims.d_head && col < dims.seq);

        workgroupBarrier();

        for(var i: u32 = 0u; i < TILE; i++) {
            acc += tileQ[lid.y][i] * tileK[lid.x][i];
        }

        workgroupBarrier();
   
    }

    if row < dims.seq && col < dims.d_head {
        if col > row {
            O[row * dims.seq + col] = -1e9;            
        }else{
            O[row * dims.seq + col] = acc * scale;
        }
    }    
}