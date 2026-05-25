struct Dims {
    seq: u32,
    vocab_size: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<u32>;
@group(0) @binding(2) var<storage, read_write> losses: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad: array<f32>;
@group(0) @binding(4) var<uniform> dims: Dims;

@compute @workgroup_size(1)
fn cross_entropy_loss(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let row = gid.x;
    if row >= dims.seq { return; }

    let vocab_size = dims.vocab_size;

    // max_val = max_j logits[i,j]
    var max_val:f32= -1e9;
    for(var j: u32 = 0u; j < vocab_size; j++){
        let v = logits[row * vocab_size + j];
        if v > max_val { max_val = v; }
    }

    // sum_exp = Σ_j exp(logits[i,j] - max_val)
    var sum_exp:f32 = 0.0;
    for(var j: u32 = 0u; j < vocab_size; j++){
        sum_exp += exp(logits[row * vocab_size + j] - max_val);
    }

    // lse = max_val + log(sum_exp)
    let lse = max_val + log(sum_exp);

    // loss_i = lse - logits[i, target_i]
    let target_i = targets[row];
    losses[row] = lse - logits[row * vocab_size + target_i];

    // dlogits[i,j] = exp(logits[i,j] - lse) - 1{j=target_i}
    for(var j: u32 = 0u; j < vocab_size; j++){
        let s = exp(logits[row * vocab_size + j] - lse);
        grad[row * vocab_size + j] = select(s, s - 1.0, j == target_i);
    }
}