struct Dims {
    seq: u32,
    vocab_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0)
var<storage, read> logits: array<f32>;

@group(0) @binding(1)
var<storage, read> targets: array<u32>;

@group(0) @binding(2)
var<storage, read_write> losses: array<f32>;

@group(0) @binding(3)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(4)
var<uniform> dims: Dims;


@compute
@workgroup_size(1, 1, 1)
fn cross_entropy_loss(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;

    if row >= dims.seq {
        return;
    }

    let vocab_size = dims.vocab_size;
    let base = row * vocab_size;
    let target_i = targets[row];

    // Rust側でtarget範囲を検証済みであることを前提とする
    var max_val = logits[base];

    for (var j: u32 = 1u; j < vocab_size; j++) {
        let v = logits[base + j];

        if v > max_val {
            max_val = v;
        }
    }

    var sum_exp = 0.0;

    for (var j: u32 = 0u; j < vocab_size; j++) {
        sum_exp += exp(logits[base + j] - max_val);
    }

    let lse = max_val + log(sum_exp);

    losses[row] =
        lse - logits[base + target_i];

    for (var j: u32 = 0u; j < vocab_size; j++) {
        let p = exp(logits[base + j] - lse);

        grad[base + j] =
            select(p, p - 1.0, j == target_i);
    }
}