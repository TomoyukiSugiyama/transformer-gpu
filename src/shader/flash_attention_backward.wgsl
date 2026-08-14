// binding layout:
//   0: inputs   [do_ | q | k | v | l | d_vec]  (各 seq*d_head, seq, seq)
//   1: dq       [seq * d_head]  atomic<i32>
//   2: dk       [seq * d_head]  f32
//   3: dv       [seq * d_head]  f32
//   4: dims     uniform [seq, d_head, 0, 0]

const BR: u32 = 64u;
const MAX_D_HEAD: u32 = 128u;

struct AttentionDims {
    seq: u32,
    d_head: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> inputs: array<f32>;

// dq is stored as float bits inside atomic<i32>
@group(0) @binding(1)
var<storage, read_write> dq: array<atomic<i32>>;

@group(0) @binding(2)
var<storage, read_write> dk: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dv: array<f32>;

@group(0) @binding(4)
var<uniform> dims: AttentionDims;

var<workgroup> wg_s: array<f32, BR>;
var<workgroup> wg_dov: array<f32, BR>;


fn atomic_add_f32(
    idx: u32,
    value: f32,
) {
    loop {
        let old_bits = atomicLoad(&dq[idx]);
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<i32>(old_value + value);

        let result = atomicCompareExchangeWeak(
            &dq[idx],
            old_bits,
            new_bits,
        );

        if result.exchanged {
            break;
        }
    }
}


@compute
@workgroup_size(BR, 1, 1)
fn attention_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq = dims.seq;
    let d_head = dims.d_head;

    // 1 workgroup = 1 key position
    let j = wid.x;
    let d = lid.x;

    let scale = 1.0 / sqrt(f32(d_head));

    // Rust側がdispatch_workgroups(seq, 1, 1)なので常にtrue
    let in_range = j < seq;

    let off_do = 0u;
    let off_q = seq * d_head;
    let off_k = seq * d_head * 2u;
    let off_v = seq * d_head * 3u;
    let off_l = seq * d_head * 4u;
    let off_dvec = seq * d_head * 4u + seq;

    var acc_dk = 0.0;
    var acc_dv = 0.0;

    // causal mask:
    // key jはquery i >= jからのみ参照される
    for (var i: u32 = j; i < seq; i++) {
        if d < d_head {
            wg_s[d] =
                inputs[off_q + i * d_head + d] *
                inputs[off_k + j * d_head + d];

            wg_dov[d] =
                inputs[off_do + i * d_head + d] *
                inputs[off_v + j * d_head + d];
        } else {
            wg_s[d] = 0.0;
            wg_dov[d] = 0.0;
        }

        workgroupBarrier();

        // d_head <= BRを前提にしたreduction
        var stride = BR / 2u;

        loop {
            if stride == 0u {
                break;
            }

            if d < stride {
                wg_s[d] += wg_s[d + stride];
                wg_dov[d] += wg_dov[d + stride];
            }

            workgroupBarrier();
            stride /= 2u;
        }

        if d < d_head {
            let score =
                wg_s[0] * scale;

            let do_dot_v =
                wg_dov[0];

            // L[i] = logsumexp(score[i, :])
            let p =
                exp(score - inputs[off_l + i]);

            // D[i] = dot(dO[i], O[i])
            let d_i =
                inputs[off_dvec + i];

            // dS[i,j] = P[i,j] * (dP[i,j] - D[i])
            let ds =
                p * (do_dot_v - d_i);

            // dV[j,d] += P[i,j] * dO[i,d]
            acc_dv +=
                p * inputs[off_do + i * d_head + d];

            // dK[j,d] += scale * dS[i,j] * Q[i,d]
            acc_dk +=
                scale *
                ds *
                inputs[off_q + i * d_head + d];

            // dQ[i,d] += scale * dS[i,j] * K[j,d]
            atomic_add_f32(
                i * d_head + d,
                scale *
                ds *
                inputs[off_k + j * d_head + d],
            );
        }

        // 全スレッドが次のiの共有メモリ書き込みへ進む前に同期
        workgroupBarrier();
    }

    if in_range && d < d_head {
        dk[j * d_head + d] = acc_dk;
        dv[j * d_head + d] = acc_dv;
    }
}