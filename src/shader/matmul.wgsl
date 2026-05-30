const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    trans_a: u32,
    trans_b: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE, 1)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;
    let trans_a = dims.trans_a == 1u;
    let trans_b = dims.trans_b == 1u;

    let dim_c_row = dims.M;
    let dim_c_col = dims.N;
    var dim_ab_stride = dims.K;

    let a_rows = select(dim_c_row,dim_ab_stride, trans_a);
    let a_cols = select(dim_ab_stride, dim_c_row, trans_a);

    let b_rows = select(dim_ab_stride, dim_c_col, trans_b);
    let b_cols = select(dim_c_col, dim_ab_stride, trans_b);

    let a_col_idx = lid.x;
    let b_row_idx = lid.y;
    for(var t: u32 = 0u; t < num_tiles; t++) {
        let a_r = select(row, t * TILE + a_col_idx, trans_a);
        let a_c = select(t * TILE + a_col_idx, row, trans_a);

        tileA[lid.y][lid.x] = select(0.0, A[a_r * a_cols + a_c],
            a_r < a_rows &&
            a_c < a_cols);

        let b_r = select(t * TILE + b_row_idx, col, trans_b);
        let b_c = select(col, t * TILE + b_row_idx, trans_b);
        tileB[lid.y][lid.x] = select(0.0, B[b_r * b_cols + b_c],
            b_r < b_rows &&
            b_c < b_cols);

        workgroupBarrier();

        for(var i: u32 = 0u; i < TILE; i++) {
            acc += tileA[lid.y][i] * tileB[i][lid.x];
        }

        workgroupBarrier();
    }

    if (row < dim_c_row && col < dim_c_col) {
        C[row * dim_c_col + col] = acc;
    }
}