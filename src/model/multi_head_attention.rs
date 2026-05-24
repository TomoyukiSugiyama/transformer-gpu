use crate::{
    gpu_context::GpuContext,
    kernel::{attention::attention_gpu, matmul::matmul_gpu, rope::rope_gpu},
    util::{concat_columns_into, split_columns},
};

pub fn multi_head_attention_gpu(
    ctx: &GpuContext,
    seq: u32,
    d_model: u32,
    n_heads: u32,
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    assert!(
        d_model % n_heads == 0,
        "d_model must to divisible by n_heads"
    );
    assert!(n_heads > 0, "n_heads must be > 0");
    let d_head = d_model / n_heads;

    let q = matmul_gpu(ctx, x, w_q, seq, d_model, d_model);
    let k = matmul_gpu(ctx, x, w_k, seq, d_model, d_model);
    let v = matmul_gpu(ctx, x, w_v, seq, d_model, d_model);
    let q_heads = split_columns(&q, d_model as usize, d_head as usize, n_heads as usize);
    let k_heads = split_columns(&k, d_model as usize, d_head as usize, n_heads as usize);
    let v_heads = split_columns(&v, d_model as usize, d_head as usize, n_heads as usize);

    let q_heads: Vec<Vec<f32>> = q_heads
        .iter()
        .map(|q| rope_gpu(ctx, q, d_head as usize, &cos_table, &sin_table))
        .collect();
    let k_heads: Vec<Vec<f32>> = k_heads
        .iter()
        .map(|k| rope_gpu(ctx, k, d_head as usize, &cos_table, &sin_table))
        .collect();

    let mut attn = Vec::with_capacity(n_heads as usize);
    for i in 0..n_heads as usize {
        attn.push(attention_gpu(
            ctx,
            &q_heads[i],
            &k_heads[i],
            &v_heads[i],
            seq,
            d_head,
        ));
    }

    let attn = concat_columns_into(
        &attn,
        seq as usize,
        d_model as usize,
        d_head as usize,
        n_heads as usize,
    );

    matmul_gpu(ctx, &attn, w_o, seq, d_model, d_model)
}

// CPU リファレンス
#[cfg(test)]
pub fn multi_head_attention_cpu(
    seq: usize,
    d_model: usize,
    n_heads: usize,
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    use crate::kernel::{matmul::matmul_cpu, rope::rope_cpu};

    assert!(
        d_model % n_heads == 0,
        "d_model must to divisible by n_heads"
    );
    assert!(n_heads > 0, "n_heads must be > 0");
    let d_head = d_model / n_heads;

    let q = matmul_cpu(x, w_q, seq, d_model, d_model);
    let k = matmul_cpu(x, w_k, seq, d_model, d_model);
    let v = matmul_cpu(x, w_v, seq, d_model, d_model);
    let q_heads = split_columns(&q, d_model as usize, d_head as usize, n_heads as usize);
    let k_heads = split_columns(&k, d_model as usize, d_head as usize, n_heads as usize);
    let v_heads = split_columns(&v, d_model as usize, d_head as usize, n_heads as usize);

    let q_heads: Vec<Vec<f32>> = q_heads
        .iter()
        .map(|q| rope_cpu(q, d_head, &cos_table, &sin_table))
        .collect();
    let k_heads: Vec<Vec<f32>> = k_heads
        .iter()
        .map(|k| rope_cpu(k, d_head, &cos_table, &sin_table))
        .collect();
    let mut attn = Vec::with_capacity(n_heads as usize);
    for i in 0..n_heads as usize {
        use crate::kernel::attention::attention_cpu;

        attn.push(attention_cpu(
            &q_heads[i],
            &k_heads[i],
            &v_heads[i],
            seq,
            d_head,
        ));
    }

    let attn = concat_columns_into(
        &attn,
        seq as usize,
        d_model as usize,
        d_head as usize,
        n_heads as usize,
    );

    matmul_cpu(&attn, w_o, seq, d_model, d_model)
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::rope::create_table,
        model::multi_head_attention::{multi_head_attention_cpu, multi_head_attention_gpu},
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_multi_head_attention() {
        let seq: usize = 1024;
        let d_model: usize = 64;
        let n_heads: usize = 4;
        let x: Vec<f32> = random_f32(seq * d_model, 31);
        let w_q: Vec<f32> = random_f32(d_model * d_model, 32);
        let w_k: Vec<f32> = random_f32(d_model * d_model, 33);
        let w_v: Vec<f32> = random_f32(d_model * d_model, 34);
        let w_o: Vec<f32> = random_f32(d_model * d_model, 35);
        let d_head = d_model / n_heads;
        let max_len = 1024;
        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(d_head, max_len, base);
        let cpu = multi_head_attention_cpu(
            seq, d_model, n_heads, &x, &w_q, &w_k, &w_v, &w_o, &cos_table, &sin_table,
        );
        let ctx = GpuContext::new();
        let gpu = multi_head_attention_gpu(
            &ctx,
            seq as u32,
            d_model as u32,
            n_heads as u32,
            &x,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &cos_table,
            &sin_table,
        );

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
