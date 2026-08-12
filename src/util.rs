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

pub fn finite_slice(name: &str, xs: &[f32]) -> bool {
    let mut ok = true;
    for &x in xs {
        if !x.is_finite() {
            println!("{name} became NaN/Inf");
            ok = false;
            break;
        }
        if x.abs() > 1e1 {
            println!("{name} overflow-ish: {x:e}");
            ok = false;
            break;
        }
    }
    ok
}

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub fn random_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-scale..scale)).collect()
}

#[cfg(test)]
mod test {
    use crate::util::{concat_columns_into, split_columns};

    #[test]
    fn test_split_columns() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d_model = 4;
        let n_heads = 2;
        let d_head = d_model / n_heads;
        let out = split_columns(&x, d_model, d_head, n_heads);
        let exp = vec![vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0, 7.0, 8.0]];

        assert_eq!(out.len(), exp.len());
        assert_eq!(out[0].len(), exp[0].len());
        assert_eq!(out, exp);
    }

    #[test]
    fn test_concat_columns_into() {
        let x = vec![vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0, 7.0, 8.0]];
        let seq = 2;
        let d_model = 4;
        let n_heads = 2;
        let d_head = d_model / n_heads;
        let out = concat_columns_into(&x, seq, d_model, d_head, n_heads);
        let exp: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        assert_eq!(out.len(), exp.len());
        assert_eq!(out, exp);
    }

    #[test]
    fn test_split_concat_roundtrip() {
        let seq = 4;
        let d_model = 8;
        let n_heads = 4;
        let d_head = d_model / n_heads;
        let x: Vec<f32> = (0..seq * d_model).map(|i| i as f32).collect();

        let heads = split_columns(&x, d_model, d_head, n_heads);
        let reconstructed = concat_columns_into(&heads, seq, d_model, d_head, n_heads);

        assert_eq!(x, reconstructed);
    }
}
