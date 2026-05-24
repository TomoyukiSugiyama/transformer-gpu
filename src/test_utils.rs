// 出力結果に誤差が生じるため、rel_eps (相対誤差)、abs_eps (絶対誤差)を設定
pub fn assert_close(gpu: &[f32], cpu: &[f32], rel_eps: f32, abs_eps: f32) {
    assert_eq!(gpu.len(), cpu.len(), "shape mismatch");
    let mut max_err = 0.0f32;
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let abs_diff = (g - c).abs();
        let rel_diff = abs_diff / (c.abs().max(1e-6));
        if abs_diff > abs_eps && rel_diff > rel_eps {
            panic!("index {i}: gpu={g:.6}, cpu={c:.6}, rel_err={rel_diff:.2e} > {rel_eps:.2e}");
        }
        max_err = max_err.max(rel_diff);
    }
    println!("max_rel_err = {max_err:.2e}  [PASS]");
}
