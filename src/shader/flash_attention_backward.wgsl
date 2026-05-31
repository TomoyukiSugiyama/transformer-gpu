// binding layout:
//   0: inputs   [do_ | q | k | v | l | d_vec]  (各 seq*d_head, seq, seq)
//   1: dq       [seq * d_head]  atomic<i32>
//   2: dk       [seq * d_head]  f32
//   3: dv       [seq * d_head]  f32
//   4: dims     uniform [seq, d_head, 0, 0]

const BR: u32 = 64u;
const MAX_D_HEAD: u32 = 128u;

@group(0) @binding(0) var<storage, read>       inputs : array<f32>;
@group(0) @binding(1) var<storage, read_write> dq     : array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> dk     : array<f32>;
@group(0) @binding(3) var<storage, read_write> dv     : array<f32>;
@group(0) @binding(4) var<uniform>             dims   : vec4<u32>;

var<workgroup> wg_s    : array<f32, MAX_D_HEAD>; // q[i]・k[j]
var<workgroup> wg_dov  : array<f32, MAX_D_HEAD>; // do[i]・v[j]

fn atomic_add_f32(idx: u32, val: f32) {
    loop {
        let old_i = atomicLoad(&dq[idx]);
        let new_i = bitcast<i32>(bitcast<f32>(old_i) + val);
        let r = atomicCompareExchangeWeak(&dq[idx], old_i, new_i);
        if r.exchanged { break; }
    }
}

@compute @workgroup_size(BR)
fn attention_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let seq    = dims.x;
    let d_head = dims.y;
    let scale  = 1.0 / sqrt(f32(d_head));

    let j = gid.x / BR;
    let d = lid.x;

    if j >= seq { return; }

    let off_do_  = 0u;
    let off_q    = seq * d_head;
    let off_k    = seq * d_head * 2u;
    let off_v    = seq * d_head * 3u;
    let off_l    = seq * d_head * 4u;
    let off_dvec = seq * d_head * 4u + seq;

    var acc_dk = 0.0f;
    var acc_dv = 0.0f;

    for (var i = j; i < seq; i++) {

        if d < d_head {
            wg_s[d]   = inputs[off_q + i * d_head + d]
                      * inputs[off_k + j * d_head + d];
            wg_dov[d] = inputs[off_do_ + i * d_head + d]
                      * inputs[off_v   + j * d_head + d];
        } else {
            wg_s[d]   = 0.0f;
            wg_dov[d] = 0.0f;
        }
        workgroupBarrier();

        // tree reduce (lid.x=0 が担当)
        // d_head <= BR=64 を仮定、log2(BR)=6 ステップ
        var stride = BR / 2u;
        loop {
            if stride == 0u { break; }
            if d < stride {
                wg_s[d]   += wg_s[d + stride];
                wg_dov[d] += wg_dov[d + stride];
            }
            workgroupBarrier();
            stride /= 2u;
        }

        if d >= d_head { continue; }

        let s_ij   = wg_s[0] * scale;
        let do_v_j = wg_dov[0];

        let p_ij   = exp(s_ij - inputs[off_l + i]);
        let d_i    = inputs[off_dvec + i];
        let ds_ij  = p_ij * (do_v_j - d_i);

        acc_dv += p_ij  * inputs[off_do_ + i * d_head + d];
        acc_dk += scale * ds_ij * inputs[off_q + i * d_head + d];
        atomic_add_f32(
            i * d_head + d,
            scale * ds_ij * inputs[off_k + j * d_head + d]
        );
    }

    if d < d_head {
        dk[j * d_head + d] = acc_dk;
        dv[j * d_head + d] = acc_dv;
    }
}