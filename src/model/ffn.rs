use crate::{
    gpu_context::GpuContext,
    kernel::{matmul::matmul_forward, swiglu_elementwise::swiglu_elementwise},
    model_config::ModelConfig,
    util::random_f32,
};

#[derive(Default)]
pub struct FfnForwardCache {
    pub pre_gate: Vec<f32>,
    pub up: Vec<f32>,
}

pub struct Ffn {
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
}

impl Ffn {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            w_gate: random_f32(cfg.d_model * cfg.d_ff, 36),
            w_up: random_f32(cfg.d_model * cfg.d_ff, 37),
            w_down: random_f32(cfg.d_ff * cfg.d_model, 38),
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
        cache.pre_gate = matmul_forward(
            ctx,
            x,
            &self.w_gate,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );

        cache.up = matmul_forward(
            ctx,
            x,
            &self.w_up,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );

        let a = swiglu_elementwise(ctx, &cache.pre_gate, &cache.up);
        let y = matmul_forward(
            ctx,
            &a,
            &self.w_down,
            seq as u32,
            cfg.d_ff as u32,
            cfg.d_model as u32,
        );

        y
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
        cache.pre_gate = matmul_forward_cpu(x, &self.w_gate, seq, cfg.d_model, cfg.d_ff);

        cache.up = matmul_forward_cpu(x, &self.w_up, seq, cfg.d_model, cfg.d_ff);

        let a = swiglu_elementwise_cpu(&cache.pre_gate, &cache.up);
        let y = matmul_forward_cpu(&a, &self.w_down, seq, cfg.d_ff, cfg.d_model);

        y
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
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 42);
        let w_gate: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 43);
        let w_up: Vec<f32> = random_f32(cfg.d_model * cfg.d_ff, 44);
        let w_down: Vec<f32> = random_f32(cfg.d_ff * cfg.d_model, 45);
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
}
