pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

pub fn matmul_gpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;

    #[test]
    fn test_matmul_gpu_vs_cpu() {
        let m = 64;
        let k = 64;
        let n = 64;
        // ランダム入力（再現性のため固定シード）
        let a = random_f32(m * k, 42);
        let b = random_f32(k * n, 43);

        let cpu_out = matmul_cpu(&a, &b, m, k, n);
        let gpu_out = matmul_gpu(&a, &b, m, k, n); // wgpu 経由

        assert_close(&gpu_out, &cpu_out, 1e-4, 1e-5);
    }
}
