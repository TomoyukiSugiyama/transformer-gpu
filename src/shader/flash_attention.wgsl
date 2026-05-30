const Br: u32 = 64u;
const Bc: u32 = 16u;
const MAX_D_HEAD: u32 = 128u;

struct faDims {
    seq: u32,
    d_head: u32,
}

@group(0) @binding(0) var<storage, read> fa_q: array<f32>;
@group(0) @binding(1) var<storage, read> fa_k: array<f32>;
@group(0) @binding(2) var<storage, read> fa_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> fa_scores: array<f32>; // O: [seq*d_head] + L: [seq]
@group(0) @binding(4) var<uniform> fa_dims: faDims;

var<workgroup> tile_q: array<f32, Br * MAX_D_HEAD>; // [Br][d_head]
var<workgroup> tile_k: array<f32, Bc * MAX_D_HEAD>; // [Bc][d_head]
var<workgroup> tile_v: array<f32, Bc * MAX_D_HEAD>; // [Bc][d_head]

@compute @workgroup_size(Br, 1, 1)
fn flash_attention(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let d_head = fa_dims.d_head;
    let seq = fa_dims.seq;
    let row = wid.x * Br + lid.x;
    let in_range = row < seq;

    var m_old = -3.4e38; // max
    var l_old = 0.0; // sum
    var o: array<f32, MAX_D_HEAD>; // output

    let num_col_tiles = (seq + Bc - 1u) / Bc;
    // 1 / √d_k
    let scale = 1.0 / sqrt(f32(d_head));

    if in_range {
        // tile_q にロード
        for (var d: u32 = 0u; d < d_head; d++) {
            tile_q[lid.x * d_head + d] = fa_q[row * d_head + d];
        }
    }

    workgroupBarrier();

    for(var t: u32 = 0u; t < num_col_tiles; t++) {
        let base_k = t * Bc;

        // tile_k, tile_v にロード
        var i = lid.x;
        loop {
            if i >= Bc * d_head { break; }
            let k_row = base_k + i / d_head;
            let d = i % d_head;
            if k_row < seq {
                tile_k[i] = fa_k[k_row * d_head + d];
                tile_v[i] = fa_v[k_row * d_head + d];
            }else{
                tile_k[i] = 0.0;
                tile_v[i] = 0.0;
            }
            i += Br;
        }

        workgroupBarrier();

        if in_range {
            // 1. Q[row] · K[t*Bc..(t+1)*Bc]^T / √d_k を計算, causal mask → s (スコア)
            var s: array<f32, Bc>;
            for(var j: u32 = 0u; j < Bc; j++) {
                let k_row = base_k + j;
                s[j] = -3.4e38;
                // causal mask
                if k_row < seq && k_row <= row {
                    s[j] = 0.0;
                    for(var d: u32 = 0u; d < d_head; d++) {
                        s[j] += 
                        tile_q[lid.x * d_head + d] * 
                        tile_k[j * d_head + d];
                    }
                    s[j] *= scale;
                }
            }

            // 2. online softmax で m, l, o を更新
            // https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            // m_new = max(m_old, max(タイルt))
            var m_new = m_old;
            for(var j: u32 = 0u; j < Bc; j++) {
                if s[j] > m_new { m_new = s[j]; }
            }

            // l_new = l_old * exp(m_old - m_new) + sum(exp(タイルt - m_new))
            let correction = exp(m_old - m_new);
            var l_new = l_old * correction;
            for(var j: u32 = 0u; j < Bc; j++) {
                l_new += exp(s[j] - m_new);
            }

            // o = (o * l_old * exp(m_old - m_new) + sum(exp(タイルt - m_new) * V_t)) / l_new
            for(var d: u32 = 0u; d < d_head; d++) {
                o[d] = o[d] * l_old * correction;
                for(var j: u32 = 0u; j < Bc; j++) {
                    let k_row = base_k + j;
                    if k_row < seq {
                        o[d] += exp(s[j] - m_new)
                                * tile_v[j * d_head + d];
                    }
                }
                o[d] /= l_new;
            }

            m_old = m_new;
            l_old = l_new;

        }

        workgroupBarrier();
    }

    if in_range {
        for(var d: u32 = 0u; d < d_head; d++){
            fa_scores[row * d_head + d] = o[d];
            // L[i] = m_i + log(l_i)
            fa_scores[seq * d_head + row] = m_old + log(l_old);
        }
    }
}