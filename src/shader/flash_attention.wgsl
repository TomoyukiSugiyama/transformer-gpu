const BR: u32 = 64u;
const BC: u32 = 16u;
const MAX_D_HEAD: u32 = 128u;
const NEG_INF: f32 = -3.402823466e+38;

struct FaDims {
    seq: u32,
    d_head: u32,
}

@group(0) @binding(0)
var<storage, read> fa_q: array<f32>;

@group(0) @binding(1)
var<storage, read> fa_k: array<f32>;

@group(0) @binding(2)
var<storage, read> fa_v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> fa_out_lse: array<f32>;
// layout:
//   fa_out_lse[0 .. seq*d_head]       = output O
//   fa_out_lse[seq*d_head .. seq*d_head+seq] = LSE

@group(0) @binding(4)
var<uniform> fa_dims: FaDims;

var<workgroup> tile_q: array<f32, BR * MAX_D_HEAD>;
var<workgroup> tile_k: array<f32, BC * MAX_D_HEAD>;
var<workgroup> tile_v: array<f32, BC * MAX_D_HEAD>;


@compute
@workgroup_size(BR, 1, 1)
fn flash_attention(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq = fa_dims.seq;
    let d_head = fa_dims.d_head;

    let row = wid.x * BR + lid.x;
    let in_range = row < seq;

    let scale = 1.0 / sqrt(f32(d_head));
    let num_col_tiles = (seq + BC - 1u) / BC;

    var m_old = NEG_INF;
    var l_old = 0.0;
    var o: array<f32, MAX_D_HEAD>;

    // 未初期化読み出しを防ぐ
    for (var d: u32 = 0u; d < MAX_D_HEAD; d++) {
        o[d] = 0.0;
    }

    // Query tileのロード
    if in_range {
        for (var d: u32 = 0u; d < d_head; d++) {
            tile_q[lid.x * d_head + d] =
                fa_q[row * d_head + d];
        }
    } else {
        // barrier前に共有領域を初期化しておく
        for (var d: u32 = 0u; d < d_head; d++) {
            tile_q[lid.x * d_head + d] = 0.0;
        }
    }

    workgroupBarrier();

    for (var t: u32 = 0u; t < num_col_tiles; t++) {
        let base_k = t * BC;

        // K/V tileのロード
        var load_idx = lid.x;

        loop {
            if load_idx >= BC * d_head {
                break;
            }

            let k_row = base_k + load_idx / d_head;
            let d = load_idx % d_head;

            if k_row < seq {
                tile_k[load_idx] =
                    fa_k[k_row * d_head + d];

                tile_v[load_idx] =
                    fa_v[k_row * d_head + d];
            } else {
                tile_k[load_idx] = 0.0;
                tile_v[load_idx] = 0.0;
            }

            load_idx += BR;
        }

        workgroupBarrier();

        if in_range {
            var score: array<f32, BC>;
            var valid: array<bool, BC>;

            // Q[row] · K[j] / sqrt(d_head)
            for (var j: u32 = 0u; j < BC; j++) {
                let k_row = base_k + j;

                valid[j] =
                    k_row < seq &&
                    k_row <= row;

                if valid[j] {
                    var dot = 0.0;

                    for (var d: u32 = 0u; d < d_head; d++) {
                        dot +=
                            tile_q[lid.x * d_head + d] *
                            tile_k[j * d_head + d];
                    }

                    score[j] = dot * scale;
                } else {
                    score[j] = 0.0;
                }
            }

            // 有効要素だけでtile内最大値を計算
            var m_new = m_old;

            for (var j: u32 = 0u; j < BC; j++) {
                if valid[j] {
                    if score[j] > m_new {
                        m_new = score[j];
                    }
                }
            }

            // causal attentionではrow >= 0なので、
            // 少なくとも1つは有効要素が存在する
            let correction = exp(m_old - m_new);

            var l_new =
                l_old * correction;

            for (var j: u32 = 0u; j < BC; j++) {
                if valid[j] {
                    l_new += exp(score[j] - m_new);
                }
            }

            // 出力の更新
            for (var d: u32 = 0u; d < d_head; d++) {
                var acc =
                    o[d] * l_old * correction;

                for (var j: u32 = 0u; j < BC; j++) {
                    if valid[j] {
                        let p =
                            exp(score[j] - m_new);

                        acc +=
                            p * tile_v[j * d_head + d];
                    }
                }

                o[d] = acc / l_new;
            }

            m_old = m_new;
            l_old = l_new;
        }

        // 次のtileがtile_k/tile_vを上書きする前に同期
        workgroupBarrier();
    }

    if in_range {
        for (var d: u32 = 0u; d < d_head; d++) {
            fa_out_lse[row * d_head + d] = o[d];
        }

        // LSEは一度だけ書く
        fa_out_lse[seq * d_head + row] =
            m_old + log(l_old);
    }
}