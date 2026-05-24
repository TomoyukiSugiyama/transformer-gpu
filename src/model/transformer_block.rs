use crate::{
    kernel::{residual_add::residual_add_gpu, rms_norm::rms_norm_gpu, swiglu::swiglu_gpu},
    model::multi_head_attention::multi_head_attention_gpu,
};

fn transformer_block_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    seq: u32,
    d_model: u32,
    n_heads: u32,
    d_ff: u32,
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    let gamma: Vec<f32> = vec![1.0; d_model as usize];
    let eps = 1e-6;
    let norm1 = rms_norm_gpu(&device, &queue, &x, &gamma, eps, d_model as u32);
    let mha = multi_head_attention_gpu(
        &device, &queue, seq, d_model, n_heads, &norm1, w_q, w_k, w_v, w_o, cos_table, sin_table,
    );
    let add = residual_add_gpu(&device, &queue, x, &mha);
    let norm2 = rms_norm_gpu(&device, &queue, &add, &gamma, eps, d_model as u32);
    let ffn = swiglu_gpu(
        &device, &queue, &norm2, w_gate, w_up, w_down, seq, d_model, d_ff,
    );
    let out = residual_add_gpu(&device, &queue, &add, &ffn);

    out
}

// CPU リファレンス
#[cfg(test)]
fn transformer_block_cpu(
    seq: usize,
    d_model: usize,
    n_heads: usize,
    d_ff: usize,
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    use crate::{
        kernel::{residual_add::residual_add_cpu, rms_norm::rms_norm_cpu, swiglu::swiglu_cpu},
        model::multi_head_attention::multi_head_attention_cpu,
    };
    let gamma: Vec<f32> = vec![1.0; d_model as usize];
    let eps = 1e-6;
    let norm1 = rms_norm_cpu(&x, &gamma, eps, d_model);
    let mha = multi_head_attention_cpu(
        seq, d_model, n_heads, &norm1, w_q, w_k, w_v, w_o, cos_table, sin_table,
    );
    let add = residual_add_cpu(x, &mha);
    let norm2 = rms_norm_cpu(&add, &gamma, eps, d_model);
    let ffn = swiglu_cpu(&norm2, w_gate, w_up, w_down, seq, d_model, d_ff);
    let out = residual_add_cpu(&add, &ffn);

    out
}

#[cfg(test)]
mod test {
    use crate::{
        kernel::rope::create_table,
        model::transformer_block::{transformer_block_cpu, transformer_block_gpu},
        test_utils::{assert_close, gpu_context, random_f32},
    };

    #[test]
    fn test_transformer_block() {
        let seq: usize = 1024;
        let d_model: usize = 64;
        let d_ff = 128;
        let n_heads: usize = 4;
        let x: Vec<f32> = random_f32(seq * d_model, 31);
        let w_q: Vec<f32> = random_f32(d_model * d_model, 32);
        let w_k: Vec<f32> = random_f32(d_model * d_model, 33);
        let w_v: Vec<f32> = random_f32(d_model * d_model, 34);
        let w_o: Vec<f32> = random_f32(d_model * d_model, 35);
        let w_gate: Vec<f32> = random_f32(d_model * d_ff, 36);
        let w_up: Vec<f32> = random_f32(d_model * d_ff, 37);
        let w_down: Vec<f32> = random_f32(d_ff * d_model, 38);
        let d_head = d_model / n_heads;
        let max_len = 1024;
        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(d_head, max_len, base);

        let (device, queue) = gpu_context();
        let cpu = transformer_block_cpu(
            seq, d_model, n_heads, d_ff, &x, &w_q, &w_k, &w_v, &w_o, &w_gate, &w_up, &w_down,
            &cos_table, &sin_table,
        );
        let gpu = transformer_block_gpu(
            &device,
            &queue,
            seq as u32,
            d_model as u32,
            n_heads as u32,
            d_ff as u32,
            &x,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &w_gate,
            &w_up,
            &w_down,
            &cos_table,
            &sin_table,
        );

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
    }
}
