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
    if xs.is_empty() {
        println!("{name}: empty tensor");
        return true;
    }

    let s = tensor_stats(xs);

    if s.non_finite > 0 {
        println!(
            "{name}: NaN/Inf count={} len={} rms={:.4e} max_abs={:.4e}",
            s.non_finite,
            s.len,
            s.rms,
            s.max_abs,
        );
        return false;
    }

    true
}

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub fn random_f32(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-scale..scale)).collect()
}

#[derive(Clone, Copy, Debug)]
pub struct TensorStats {
    pub rms: f32,
    pub max_abs: f32,
    pub non_finite: usize,
    pub len: usize,
}

pub fn tensor_stats(xs: &[f32]) -> TensorStats {
    if xs.is_empty() {
        return TensorStats {
            rms: 0.0,
            max_abs: 0.0,
            non_finite: 0,
            len: 0,
        };
    }

    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mut non_finite = 0usize;

    for &x in xs {
        if !x.is_finite() {
            non_finite += 1;
            continue;
        }

        sum_sq += (x as f64) * (x as f64);
        max_abs = max_abs.max(x.abs());
    }

    TensorStats {
        rms: (sum_sq / xs.len() as f64).sqrt() as f32,
        max_abs,
        non_finite,
        len: xs.len(),
    }
}

pub fn require_finite(name: &str, xs: &[f32]) {
    let s = tensor_stats(xs);
    assert_eq!(
        s.non_finite, 0,
        "{name}: contains {} NaN/Inf values; rms={}, max_abs={}",
        s.non_finite, s.rms, s.max_abs
    );
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
