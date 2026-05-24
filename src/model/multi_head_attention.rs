use crate::kernel::{attention::attention_gpu, matmul::matmul_gpu};

pub fn multi_head_attention_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    seq: u32,
    d_model: u32,
    n_heads: u32,
    x: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
) -> Vec<f32> {
    assert!(
        d_model % n_heads == 0,
        "d_model must to divisible by n_heads"
    );
    assert!(n_heads > 0, "n_heads must be > 0");
    let d_head = d_model / n_heads;

    let q = matmul_gpu(device, queue, x, w_q, seq, d_model, d_model);
    let k = matmul_gpu(device, queue, x, w_k, seq, d_model, d_model);
    let v = matmul_gpu(device, queue, x, w_v, seq, d_model, d_model);
    let q_heads = split_columns(&q, d_model as usize, d_head as usize, n_heads as usize);
    let k_heads = split_columns(&k, d_model as usize, d_head as usize, n_heads as usize);
    let v_heads = split_columns(&v, d_model as usize, d_head as usize, n_heads as usize);

    let mut attn = Vec::with_capacity(n_heads as usize);
    for i in 0..n_heads as usize {
        attn.push(attention_gpu(
            device,
            queue,
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

    matmul_gpu(device, queue, &attn, w_o, seq, d_model, d_model)
}

pub fn split_columns(x: &[f32], d_model: usize, d_head: usize, n_heads: usize) -> Vec<Vec<f32>> {
    assert!(
        x.len() % d_model == 0,
        "split_columns: x.len()={} not divisible by d_model={}",
        x.len(),
        d_model
    );
    assert!(
        d_model % n_heads == 0,
        "split_columns: d_model={} not divisible by n_heads={}",
        d_model,
        n_heads
    );
    let seq = x.len() / d_model;

    (0..n_heads)
        .map(|h| {
            let mut data = Vec::with_capacity(seq * d_head);
            for i in 0..seq {
                let start = i * d_model + h * d_head;
                data.extend_from_slice(&x[start..start + d_head]);
            }
            data
        })
        .collect()
}

pub fn concat_columns_into(
    parts: &[Vec<f32>],
    seq: usize,
    d_model: usize,
    d_head: usize,
    n_heads: usize,
) -> Vec<f32> {
    assert!(!parts.is_empty(), "concat_columns_into: empty parts");
    assert_eq!(
        parts.len(),
        n_heads,
        "parts.len()={} and n_heads={} must have the same length",
        parts.len(),
        n_heads
    );
    assert!(parts.len() > 0, "parts.len() is empty",);
    assert_eq!(
        parts[0].len(),
        seq * d_head,
        "parts[0].len()={} and seq*d_head={} must have the same length",
        parts[0].len(),
        seq * d_head
    );

    let mut out = vec![0.0f32; seq * d_model];
    for (h, part) in parts.iter().enumerate() {
        for i in 0..seq {
            let dst = i * d_model + h * d_head;
            let src = i * d_head;
            out[dst..dst + d_head].copy_from_slice(&part[src..src + d_head]);
        }
    }

    out
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
) -> Vec<f32> {
    use crate::kernel::matmul::matmul_cpu;

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
        model::multi_head_attention::{multi_head_attention_cpu, multi_head_attention_gpu},
        test_utils::{assert_close, gpu_context, random_f32},
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
        let cpu = multi_head_attention_cpu(seq, d_model, n_heads, &x, &w_q, &w_k, &w_v, &w_o);
        let (device, queue) = gpu_context();
        let gpu = multi_head_attention_gpu(
            &device,
            &queue,
            seq as u32,
            d_model as u32,
            n_heads as u32,
            &x,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
        );

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
