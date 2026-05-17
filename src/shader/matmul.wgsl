const TILE: u32 = 16u;

struct Dims {
    M: u32,
    K: u32,
    N: u32
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<array<f32, 16u>, 16u>;
var<workgroup> tileB: array<array<f32, 16u>, 16u>;

@compute @workgroup_size(TILE, TILE, 1)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;

    var acc: f32 = 0.0;
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for(var t: u32 = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + lid.y;
        let b_row = t * TILE + lid.x;

        tileA[lid.x][lid.y] = select(0.0, A[row * dims.K + a_col], row < dims.M && a_col < dims.K);
        tileB[lid.x][lid.y] = select(0.0, B[b_row * dims.N + col], b_row < dims.K && col < dims.N);

        workgroupBarrier();

        for(var i: u32 = 0u; i < TILE; i++) {
            acc += tileA[lid.x][i] * tileB[i][lid.y];
        }

        workgroupBarrier();
    }

    if row < dims.M && col < dims.N {
        C[row * dims.N + col] = acc;
    }
}