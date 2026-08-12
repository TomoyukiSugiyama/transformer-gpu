use crate::{
    gpu_context::GpuContext,
    kernel::{
        matmul::{matmul_backward, matmul_forward},
        swiglu_elementwise::{swiglu_elementwise, swiglu_elementwise_backward},
    },
    model_config::ModelConfig,
    util::{finite_slice, random_f32},
};

#[derive(Default)]
pub struct FfnForwardCache {
    pub pre_gate: Vec<f32>,
    pub up: Vec<f32>,
    pub swigle_out: Vec<f32>,
    pub x_in: Vec<f32>,
}

pub struct Ffn {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

#[derive(Default)]
pub struct FfnBackward {
    pub dx: Vec<f32>,
    pub dw_gate: Vec<f32>,
    pub dw_up: Vec<f32>,
    pub dw_down: Vec<f32>,
}

impl Ffn {
    pub fn new(cfg: &ModelConfig) -> Self {
        let scale_in = (1.0 / cfg.d_model as f32).sqrt();
        let scale_down = (1.0 / cfg.d_ff as f32).sqrt();
        Self {
            w_gate: random_f32(cfg.d_model * cfg.d_ff, 36, scale_in),
            w_up: random_f32(cfg.d_model * cfg.d_ff, 37, scale_in),
            w_down: random_f32(cfg.d_ff * cfg.d_model, 38, scale_down),
        }
    }

    pub fn forward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        x: &[f32],
        cache: &mut FfnForwardCache,
    ) -> Vec<f32> {
        let seq = x.len() / cfg.d_model;
        cache.x_in = x.to_vec();
        finite_slice("fwd:ffn:x_in", &cache.x_in);
        cache.pre_gate = matmul_forward(
            ctx,
            x,
            &self.w_gate,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );
        finite_slice("fwd:ffn:matmul_forward:pre_gate", &cache.pre_gate);

        cache.up = matmul_forward(
            ctx,
            x,
            &self.w_up,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );
        finite_slice("fwd:ffn:matmul_forward:up", &cache.up);

        cache.swigle_out = swiglu_elementwise(ctx, &cache.pre_gate, &cache.up);
        finite_slice("fwd:ffn:swiglu_elementwise:swigle_out", &cache.swigle_out);

        let y = matmul_forward(
            ctx,
            &cache.swigle_out,
            &self.w_down,
            seq as u32,
            cfg.d_ff as u32,
            cfg.d_model as u32,
        );

        finite_slice("fwd:ffn:matmul_forward:y", &y);

        y
    }

    pub fn backward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        dy: &[f32],
        cache: &mut FfnForwardCache,
    ) -> FfnBackward {
        let seq = dy.len() / cfg.d_model;

        // da = dy @ w_down^T # (seq, d_ff)
        // dW_down = a^T @ dy # (d_ff, d_model)
        let (da, dw_down) = matmul_backward(
            ctx,
            &dy,
            &cache.swigle_out,
            &self.w_down,
            seq as u32,
            cfg.d_ff as u32,
            cfg.d_model as u32,
        );

        // d_pre_gate, d_up = swiglu_elementwise()
        let (d_pre_gate, d_up) = swiglu_elementwise_backward(ctx, &da, &cache.pre_gate, &cache.up);

        // dx_from_up = d_up @ W_up^T
        // dW_up = x^T @ d_up
        let (dx_from_up, dw_up) = matmul_backward(
            ctx,
            &d_up,
            &cache.x_in,
            &self.w_up,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );

        // dx_from_gate = d_pre_gate @ W_gate^T
        // dW_gate = x^T @ d_pre_gate
        let (dx_from_gate, dw_gate) = matmul_backward(
            ctx,
            &d_pre_gate,
            &cache.x_in,
            &self.w_gate,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );

        // dx = dx_from_gate + dx_from_up
        let dx: Vec<f32> = dx_from_up
            .iter()
            .zip(dx_from_gate.iter())
            .map(|(u, g)| u + g)
            .collect();

        FfnBackward {
            dx,
            dw_gate,
            dw_up,
            dw_down,
        }
    }

    // CPU リファレンス
    #[cfg(test)]
    pub fn forward_cpu(
        &self,
        cfg: &ModelConfig,
        x: &[f32],
        cache: &mut FfnForwardCache,
    ) -> Vec<f32> {
        use crate::kernel::{
            matmul::matmul_forward_cpu, swiglu_elementwise::swiglu_elementwise_cpu,
        };

        let seq = x.len() / cfg.d_model;
        cache.x_in = x.to_vec();
        cache.pre_gate = matmul_forward_cpu(x, &self.w_gate, seq, cfg.d_model, cfg.d_ff);

        cache.up = matmul_forward_cpu(x, &self.w_up, seq, cfg.d_model, cfg.d_ff);

        cache.swigle_out = swiglu_elementwise_cpu(&cache.pre_gate, &cache.up);
        let y = matmul_forward_cpu(&cache.swigle_out, &self.w_down, seq, cfg.d_ff, cfg.d_model);

        y
    }

    #[cfg(test)]
    pub fn backward_cpu(
        &self,
        cfg: &ModelConfig,
        dy: &[f32],
        cache: &mut FfnForwardCache,
    ) -> FfnBackward {
        use crate::kernel::{
            matmul::matmul_backward_cpu, swiglu_elementwise::swiglu_elementwise_backward_cpu,
        };

        let seq = dy.len() / cfg.d_model;

        // da = dy @ w_down^T # (seq, d_ff)
        // dW_down = a^T @ dy # (d_ff, d_model)
        let (da, dw_down) = matmul_backward_cpu(
            &dy,
            &cache.swigle_out,
            &self.w_down,
            seq,
            cfg.d_ff,
            cfg.d_model,
        );

        // d_pre_gate, d_up = swiglu_elementwise()
        let (d_pre_gate, d_up) = swiglu_elementwise_backward_cpu(&da, &cache.pre_gate, &cache.up);

        // dx_from_up = d_up @ W_up^T
        // dW_up = x^T @ d_up
        let (dx_from_up, dw_up) =
            matmul_backward_cpu(&d_up, &cache.x_in, &self.w_up, seq, cfg.d_model, cfg.d_ff);

        // dx_from_gate = d_pre_gate @ W_gate^T
        // dW_gate = x^T @ d_pre_gate
        let (dx_from_gate, dw_gate) = matmul_backward_cpu(
            &d_pre_gate,
            &cache.x_in,
            &self.w_gate,
            seq,
            cfg.d_model,
            cfg.d_ff,
        );

        // dx = dx_from_gate + dx_from_up
        let dx = dx_from_up
            .iter()
            .zip(dx_from_gate.iter())
            .map(|(u, g)| u + g)
            .collect();

        FfnBackward {
            dx,
            dw_gate,
            dw_up,
            dw_down,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        model::ffn::{Ffn, FfnForwardCache},
        model_config::ModelConfig,
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_swiglu() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            d_model: 2,
            d_ff: 4,
            ..Default::default()
        };

        let x: Vec<f32> = vec![1.0, 2.0];
        let w_gate: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w_up: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let w_down: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ffn = Ffn {
            w_gate,
            w_up,
            w_down,
        };
        let mut cache_cpu = FfnForwardCache::default();
        let mut cache = FfnForwardCache::default();
        let cpu = ffn.forward_cpu(&cfg, &x, &mut cache_cpu);
        let gpu = ffn.forward(&ctx, &cfg, &x, &mut cache);

        // gate = x・w_gate
        // | 1.0 2.0 || 1.0 1.0 1.0 1.0 |
        //            | 1.0 1.0 1.0 1.0 |
        // =
        // | 3.0 3.0 3.0 3.0 |
        // up = x・w_up
        // | 3.0 3.0 3.0 3.0 |
        // a = swish(g) * u
        // swish = x / (1.0 + (-x).exp())
        // | 8.57316 8.57316 8.57316 8.57316|
        // y = a・w_down
        // | 8.57316 8.57316 8.57316 8.57316 || 1.0 1.0 |
        //                                    | 1.0 1.0 |
        //                                    | 1.0 1.0 |
        //                                    | 1.0 1.0 |
        // =
        // | 34.29264 34.29264 |
        let exp: Vec<f32> = vec![34.29264, 34.29264];
        cpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!(
                    (*c - *e).abs() < 1e-4,
                    "CPU index={} got={:.6} exp={:.6}",
                    i,
                    c,
                    e
                );
            });
        gpu.iter()
            .zip(exp.iter())
            .enumerate()
            .for_each(|(i, (c, e))| {
                assert!(
                    (*c - *e).abs() < 1e-4,
                    "GPU index={} got={:.6} exp={:.6}",
                    i,
                    c,
                    e
                );
            });
    }

    #[test]
    fn test_swiglu_random() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            d_model: 64,
            d_ff: 128,
            ..Default::default()
        };
        let seq: usize = 4;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 42, scale);
        let w_gate: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 43, scale);
        let w_up: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 44, scale);
        let w_down: Vec<f32> = random_f32(cfg.d_ff * cfg.d_model, 45, scale);
        let ffn = Ffn {
            w_gate,
            w_up,
            w_down,
        };
        let mut cache_cpu = FfnForwardCache::default();
        let mut cache = FfnForwardCache::default();
        let cpu = ffn.forward_cpu(&cfg, &x, &mut cache_cpu);
        let gpu = ffn.forward(&ctx, &cfg, &x, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_swiglu_random_backward() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            d_model: 64,
            d_ff: 128,
            ..Default::default()
        };
        let seq: usize = 4;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 42, scale);
        let w_gate: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 43, scale);
        let w_up: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 44, scale);
        let w_down: Vec<f32> = random_f32(cfg.d_ff * cfg.d_model, 45, scale);
        let ffn = Ffn {
            w_gate,
            w_up,
            w_down,
        };
        let mut cache_cpu = FfnForwardCache::default();
        let mut cache_gpu = FfnForwardCache::default();
        let dx_cpu = ffn.forward_cpu(&cfg, &x, &mut cache_cpu);
        let dx_gpu = ffn.forward(&ctx, &cfg, &x, &mut cache_gpu);
        let cpu = ffn.backward_cpu(&cfg, &dx_cpu, &mut cache_cpu);
        let gpu = ffn.backward(&ctx, &cfg, &dx_gpu, &mut cache_gpu);

        assert_close(&gpu.dx, &cpu.dx, 1e-4, 1e-5);
        assert_close(&gpu.dw_gate, &cpu.dw_gate, 1e-4, 1e-5);
        assert_close(&gpu.dw_up, &cpu.dw_up, 1e-4, 1e-5);
        assert_close(&gpu.dw_down, &cpu.dw_down, 1e-4, 1e-5);
    }

    #[test]
    fn test_ffn_w_down_directional_derivative() {
        let cfg = ModelConfig {
            d_model: 8,
            d_ff: 16,
            n_layers: 1,
            // テストで必要な残りを既存設定から埋める
            ..Default::default()
        };

        let mut ffn = Ffn::new(&cfg);
        let seq = 3usize;

        let x = random_f32(seq * cfg.d_model, 1001, 0.1);
        let dy = random_f32(seq * cfg.d_model, 1002, 0.1);

        let mut cache = FfnForwardCache::default();
        let _y = ffn.forward_cpu(&cfg, &x, &mut cache);
        let bwd = ffn.backward_cpu(&cfg, &dy, &mut cache);

        let mut v = random_f32(ffn.w_down.len(), 1003, 1.0);

        let v_norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt() as f32;

        for x in &mut v {
            *x /= v_norm;
        }

        let analytic: f32 = bwd.dw_down.iter().zip(v.iter()).map(|(&g, &d)| g * d).sum();

        let eps = 1e-3f32;
        let original = ffn.w_down.clone();

        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w += eps * d;
        }
        let mut plus_cache = FfnForwardCache::default();
        let y_plus = ffn.forward_cpu(&cfg, &x, &mut plus_cache);
        let loss_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);
        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w -= eps * d;
        }
        let mut minus_cache = FfnForwardCache::default();
        let y_minus = ffn.forward_cpu(&cfg, &x, &mut minus_cache);
        let loss_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);

        let abs_err = (analytic - numerical).abs();
        let rel_err = abs_err / (analytic.abs() + numerical.abs() + 1e-6);

        eprintln!(
            "w_down directional derivative: \
         analytic={analytic:.7e} \
         numerical={numerical:.7e} \
         abs_err={abs_err:.7e} \
         rel_err={rel_err:.7e}",
        );

        assert!(
            rel_err < 1e-3,
            "w_down gradient mismatch: analytic={analytic}, numerical={numerical}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_ffn_w_down_directional_derivative_gpu() {
        let cfg = ModelConfig {
            d_model: 8,
            d_ff: 16,
            n_layers: 1,
            // テストで必要な残りを既存設定から埋める
            ..Default::default()
        };

        let mut ffn = Ffn::new(&cfg);
        let seq = 3usize;

        let x = random_f32(seq * cfg.d_model, 1001, 0.1);
        let dy = random_f32(seq * cfg.d_model, 1002, 0.1);

        // let mut cache = FfnForwardCache::default();
        // let _y = ffn.forward_cpu(&cfg, &x, &mut cache);
        // let bwd = ffn.backward_cpu(&cfg, &dy, &mut cache);

        let mut v = random_f32(ffn.w_down.len(), 1003, 1.0);

        let v_norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt() as f32;

        for x in &mut v {
            *x /= v_norm;
        }

        let ctx = GpuContext::new();

        let mut cache_gpu = FfnForwardCache::default();
        let _ = ffn.forward(&ctx, &cfg, &x, &mut cache_gpu);
        let gpu_bwd = ffn.backward(&ctx, &cfg, &dy, &mut cache_gpu);

        let analytic: f32 = gpu_bwd
            .dw_down
            .iter()
            .zip(v.iter())
            .map(|(&g, &d)| g * d)
            .sum();

        // let analytic: f32 = bwd.dw_down.iter().zip(v.iter()).map(|(&g, &d)| g * d).sum();

        let eps = 1e-3f32;
        let original = ffn.w_down.clone();

        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w += eps * d;
        }
        let mut plus_cache = FfnForwardCache::default();
        let y_plus = ffn.forward_cpu(&cfg, &x, &mut plus_cache);
        let loss_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);
        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w -= eps * d;
        }
        let mut minus_cache = FfnForwardCache::default();
        let y_minus = ffn.forward_cpu(&cfg, &x, &mut minus_cache);
        let loss_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);

        let abs_err = (analytic - numerical).abs();
        let rel_err = abs_err / (analytic.abs() + numerical.abs() + 1e-6);

        eprintln!(
            "w_down directional derivative: \
         analytic={analytic:.7e} \
         numerical={numerical:.7e} \
         abs_err={abs_err:.7e} \
         rel_err={rel_err:.7e}",
        );

        assert!(
            rel_err < 1e-3,
            "w_down gradient mismatch: analytic={analytic}, numerical={numerical}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_ffn_w_down_directional_derivative_gpu_d256() {
        let cfg = ModelConfig {
            d_model: 256,
            d_ff: 1024,
            n_layers: 1,
            // テストで必要な残りを既存設定から埋める
            ..Default::default()
        };

        let mut ffn = Ffn::new(&cfg);
        let seq = 4usize;

        let x = random_f32(seq * cfg.d_model, 1001, 0.1);
        let dy = random_f32(seq * cfg.d_model, 1002, 0.1);

        // let mut cache = FfnForwardCache::default();
        // let _y = ffn.forward_cpu(&cfg, &x, &mut cache);
        // let bwd = ffn.backward_cpu(&cfg, &dy, &mut cache);

        let mut v = random_f32(ffn.w_down.len(), 1003, 1.0);

        let v_norm = v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt() as f32;

        for x in &mut v {
            *x /= v_norm;
        }

        let ctx = GpuContext::new();

        let mut cache_gpu = FfnForwardCache::default();
        let _ = ffn.forward(&ctx, &cfg, &x, &mut cache_gpu);
        let gpu_bwd = ffn.backward(&ctx, &cfg, &dy, &mut cache_gpu);

        let analytic: f32 = gpu_bwd
            .dw_down
            .iter()
            .zip(v.iter())
            .map(|(&g, &d)| g * d)
            .sum();

        // let analytic: f32 = bwd.dw_down.iter().zip(v.iter()).map(|(&g, &d)| g * d).sum();

        let eps = 1e-1_f32;
        let original = ffn.w_down.clone();

        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w += eps * d;
        }
        let mut plus_cache = FfnForwardCache::default();
        let y_plus = ffn.forward_cpu(&cfg, &x, &mut plus_cache);
        let loss_plus: f32 = y_plus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);
        for (w, &d) in ffn.w_down.iter_mut().zip(v.iter()) {
            *w -= eps * d;
        }
        let mut minus_cache = FfnForwardCache::default();
        let y_minus = ffn.forward_cpu(&cfg, &x, &mut minus_cache);
        let loss_minus: f32 = y_minus.iter().zip(dy.iter()).map(|(&y, &g)| y * g).sum();

        ffn.w_down.clone_from(&original);

        let numerical = (loss_plus - loss_minus) / (2.0 * eps);

        let abs_err = (analytic - numerical).abs();
        let rel_err = abs_err / (analytic.abs() + numerical.abs() + 1e-6);

        eprintln!(
            "w_down directional derivative: \
         analytic={analytic:.7e} \
         numerical={numerical:.7e} \
         abs_err={abs_err:.7e} \
         rel_err={rel_err:.7e}",
        );

        assert!(
            rel_err < 1e-3,
            "w_down gradient mismatch: analytic={analytic}, numerical={numerical}, rel_err={rel_err}"
        );
    }

}
