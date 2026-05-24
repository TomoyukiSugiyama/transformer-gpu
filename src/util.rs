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