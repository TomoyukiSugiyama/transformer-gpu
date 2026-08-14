const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32,
    trans_a: u32,
    trans_b: u32,
}

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> B: array<f32>;

@group(0) @binding(2)
var<storage, read_write> C: array<f32>;

@group(0) @binding(3)
var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, TILE>, TILE>;
var<workgroup> tileB: array<array<f32, TILE>, TILE>;


@compute
@workgroup_size(TILE, TILE, 1)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;

    let trans_a = dims.trans_a != 0u;
    let trans_b = dims.trans_b != 0u;

    let m = dims.M;
    let k = dims.K;
    let n = dims.N;

    // 元配列のrow-major shape
    let a_rows = select(m, k, trans_a);
    let a_cols = select(k, m, trans_a);

    let b_rows = select(k, n, trans_b);
    let b_cols = select(n, k, trans_b);

    var acc = 0.0;
    let num_tiles = (k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t++) {
        let a_r = select(
            row,
            t * TILE + lid.x,
            trans_a
        );

        let a_c = select(
            t * TILE + lid.x,
            row,
            trans_a
        );

        if (a_r < a_rows && a_c < a_cols) {
            tileA[lid.y][lid.x] =
                A[a_r * a_cols + a_c];
        } else {
            tileA[lid.y][lid.x] = 0.0;
        }

        let b_r = select(
            t * TILE + lid.y,
            col,
            trans_b
        );

        let b_c = select(
            col,
            t * TILE + lid.y,
            trans_b
        );

        if (b_r < b_rows && b_c < b_cols) {
            tileB[lid.y][lid.x] =
                B[b_r * b_cols + b_c];
        } else {
            tileB[lid.y][lid.x] = 0.0;
        }

        workgroupBarrier();

        for (var p: u32 = 0u; p < TILE; p++) {
            acc +=
                tileA[lid.y][p] *
                tileB[p][lid.x];
        }

        workgroupBarrier();
    }

    if (row < m && col < n) {
        C[row * n + col] = acc;
    }
}