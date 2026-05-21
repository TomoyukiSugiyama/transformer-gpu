const TILE: u32 = 16u;

struct Dims {
    seq: u32,
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> qkt_q: array<f32>;
@group(0) @binding(1) var<storage, read> qkt_k: array<f32>;
@group(0) @binding(2) var<storage, read_write> qkt_score: array<f32>;
@group(0) @binding(3) var<uniform> qkt_dims: Dims;

var<workgroup> tile_q: array<array<f32, 16u>, 16u>;
var<workgroup> tile_k: array<array<f32, 16u>, 16u>;

@compute @workgroup_size(TILE, TILE, 1)
fn qkt(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;

    var acc: f32 = 0.0;
    let num_tiles = (qkt_dims.d_head + TILE - 1u) / TILE;
    // 1 / √d_k
    let scale = 1.0 / sqrt(f32(qkt_dims.d_head));
    // QK^T / √d_k
    for(var t: u32 = 0u; t < num_tiles; t++) {
        let q_col = t * TILE + lid.x;
        let k_row = t * TILE + lid.y;
        
        tile_q[lid.y][lid.x] = select(0.0, qkt_q[row * qkt_dims.d_head + q_col], row < qkt_dims.seq && q_col < qkt_dims.d_head);
        tile_k[lid.y][lid.x] = select(0.0, qkt_k[col * qkt_dims.d_head + k_row], k_row < qkt_dims.d_head && col < qkt_dims.seq);

        workgroupBarrier();

        for(var i: u32 = 0u; i < TILE; i++) {
            acc += tile_q[lid.y][i] * tile_k[lid.x][i];
        }

        workgroupBarrier();
   
    }

    if row < qkt_dims.seq && col < qkt_dims.seq {
        qkt_score[row * qkt_dims.seq + col] = acc * scale;
    }

}

// Softmax causal カーネル
struct SoftmaxDims {
    seq: u32,
}

@group(0) @binding(0) var<storage, read_write> sm_scores: array<f32>;  // seq × seq
@group(0) @binding(1) var<uniform>             sm_dims:   SoftmaxDims;


@compute @workgroup_size(64, 1, 1)
fn softmax_causal(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;
    if row >= sm_dims.seq { return; }

    // 1. max 探索 (mask 範囲)
    var max_val: f32 = -1e9;
    for(var j: u32 = 0u; j <= row; j++){
        let s = sm_scores[row * sm_dims.seq + j];
        if s > max_val { max_val = s; }
    }

    // 2. exp + sum + casual mask
    var sum_val: f32 = 0.0;
    for(var j: u32 = 0u; j < sm_dims.seq; j++){
        if j <= row {
            let e = exp(sm_scores[row * sm_dims.seq + j] - max_val);
            sm_scores[row * sm_dims.seq + j] = e;
            sum_val += e;
        } else {
            // casual mask
            sm_scores[row * sm_dims.seq + j] = 0.0;
        }
    }

    // 3. 正規化
    for(var j: u32 = 0u; j <= row; j++){
        sm_scores[row * sm_dims.seq + j] /= sum_val; 
    }
}