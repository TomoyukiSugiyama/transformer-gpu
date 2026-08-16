use crate::{
    gpu_context::GpuContext,
    kernel::{
        residual_add::residual_add,
        rms_norm::{rms_norm, rms_norm_backward},
    },
    model::{
        attention::{Attention, AttentionBackward, AttentionForwardCache},
        ffn::{Ffn, FfnBackward, FfnForwardCache},
    },
    model_config::ModelConfig,
};

#[derive(Default)]
pub struct TransformerBlockForwardCache {
    pub x_in: Vec<f32>,
    pub norm1_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub add1_out: Vec<f32>,
    pub norm2_out: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub attn: AttentionForwardCache,
    pub ffn: FfnForwardCache,
}

pub struct TransformerBlock {
    pub attn: Attention,
    pub ffn: Ffn,
    pub gamma_1: Vec<f32>, // RMSNorm の γ
    pub gamma_2: Vec<f32>,
}

pub struct TransformerBlockBackward {
    pub dx: Vec<f32>,
    pub attn_backward: AttentionBackward,
    pub ffn_backward: FfnBackward,
    pub d_gamma_1: Vec<f32>, // RMSNorm の γ
    pub d_gamma_2: Vec<f32>,
}

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            attn: Attention::new(&cfg),
            ffn: Ffn::new(cfg),
            gamma_1: vec![1.0; cfg.d_model],
            gamma_2: vec![1.0; cfg.d_model],
        }
    }

    pub fn forward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        x: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut TransformerBlockForwardCache,
    ) -> Vec<f32> {
        cache.x_in = x.to_vec();
        cache.norm1_out = rms_norm(&ctx, &x, &self.gamma_1, cfg.eps, cfg.d_model as u32);
        cache.attn_out = self.attn.forward(
            ctx,
            cfg,
            &cache.norm1_out,
            cos_table,
            sin_table,
            &mut cache.attn,
        );
        cache.add1_out = residual_add(&ctx, x, &cache.attn_out);
        cache.norm2_out = rms_norm(
            &ctx,
            &cache.add1_out,
            &self.gamma_2,
            cfg.eps,
            cfg.d_model as u32,
        );
        cache.ffn_out = self.ffn.forward(ctx, cfg, &cache.norm2_out, &mut cache.ffn);
        residual_add(&ctx, &cache.add1_out, &cache.ffn_out)
    }

    pub fn backward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        dy: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut TransformerBlockForwardCache,
    ) -> TransformerBlockBackward {
        // residual_add backward: out = add1_out + ffn_out
        // d_add1_out_a = dy, d_ffn_out = dy
        let d_ffn_out = dy.to_vec();
        let d_add1_out_a = dy.to_vec();

        // FFN backward
        let ffn_backward = self.ffn.backward(&ctx, cfg, &d_ffn_out, &mut cache.ffn);

        // rms_norm2 backward: norm2_out = rms_norm(add1_out, γ2)
        let (d_add1_out_b, d_gamma_2) = rms_norm_backward(
            &ctx,
            &ffn_backward.dx,
            &cache.add1_out,
            &self.gamma_2,
            cfg.eps,
            cfg.d_model as u32,
        );

        // 2つの d_add1_out を合算
        let d_add1_out: Vec<f32> = d_add1_out_a
            .iter()
            .zip(d_add1_out_b.iter())
            .map(|(a, b)| a + b)
            .collect();

        // 1段目 residual_add backward: add1_out = x + attn_out
        let d_attn_out = d_add1_out.clone();
        let dx_a = d_add1_out;

        // Attention backward
        let attn_backward =
            self.attn
                .backward(ctx, cfg, &d_attn_out, cos_table, sin_table, &mut cache.attn);

        // rms_norm1 backward: norm1_out = rms_norm(x_in, γ1)
        let (dx_b, d_gamma_1) = rms_norm_backward(
            ctx,
            &attn_backward.dx, // d_norm1_out
            &cache.x_in,       // forward の入力
            &self.gamma_1,
            cfg.eps,
            cfg.d_model as u32,
        );

        // dx を合算
        let dx: Vec<f32> = dx_a.iter().zip(&dx_b).map(|(a, b)| a + b).collect();

        TransformerBlockBackward {
            dx,
            attn_backward,
            ffn_backward,
            d_gamma_1,
            d_gamma_2,
        }
    }

    // CPU リファレンス
    #[cfg(test)]
    pub fn forward_cpu(
        &self,
        cfg: &ModelConfig,
        x: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut TransformerBlockForwardCache,
    ) -> Vec<f32> {
        use crate::kernel::{residual_add::residual_add_cpu, rms_norm::rms_norm_cpu};
        cache.x_in = x.to_vec();
        cache.norm1_out = rms_norm_cpu(&x, &self.gamma_1, cfg.eps, cfg.d_model);
        cache.attn_out =
            self.attn
                .forward_cpu(cfg, &cache.norm1_out, cos_table, sin_table, &mut cache.attn);
        cache.add1_out = residual_add_cpu(x, &cache.attn_out);
        cache.norm2_out = rms_norm_cpu(&cache.add1_out, &self.gamma_2, cfg.eps, cfg.d_model);
        cache.ffn_out = self.ffn.forward_cpu(cfg, &cache.norm2_out, &mut cache.ffn);
        residual_add_cpu(&cache.add1_out, &cache.ffn_out)
    }

    #[cfg(test)]
    pub fn backward_cpu(
        &self,
        cfg: &ModelConfig,
        dy: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut TransformerBlockForwardCache,
    ) -> TransformerBlockBackward {
        use crate::kernel::rms_norm::rms_norm_backward_cpu;

        let d_ffn_out = dy.to_vec();

        let ffn_backward = self.ffn.backward_cpu(cfg, &d_ffn_out, &mut cache.ffn);
        let (d_norm2_dx, d_gamma_2) = rms_norm_backward_cpu(
            &ffn_backward.dx,
            &cache.add1_out,
            &self.gamma_2,
            cfg.eps,
            cfg.d_model,
        );

        let d_add1_out: Vec<f32> = dy
            .iter()
            .zip(d_norm2_dx.iter())
            .map(|(a, b)| a + b)
            .collect();

        let attn_backward =
            self.attn
                .backward_cpu(cfg, &d_add1_out, cos_table, sin_table, &mut cache.attn);
        let (d_norm1_dx, d_gamma_1) = rms_norm_backward_cpu(
            &attn_backward.dx, // norm1_out への上流勾配
            &cache.x_in,
            &self.gamma_1,
            cfg.eps,
            cfg.d_model,
        );

        let dx: Vec<f32> = d_add1_out
            .iter()
            .zip(d_norm1_dx.iter())
            .map(|(a, b)| a + b)
            .collect();

        TransformerBlockBackward {
            dx,
            attn_backward,
            ffn_backward,
            d_gamma_1,
            d_gamma_2,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::rope::create_table,
        model::transformer_block::{TransformerBlock, TransformerBlockForwardCache},
        model_config::ModelConfig,
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_transformer_block() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let mut cache = TransformerBlockForwardCache::default();
        let seq = 64usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_transformer_block_backward() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let mut cache = TransformerBlockForwardCache::default();
        let seq = 64usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let dy_cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let dy_gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        let cpu = tf.backward_cpu(&cfg, &dy_cpu, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.backward(&ctx, &cfg, &dy_gpu, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu.dx, &cpu.dx, 1e-4, 1e-5);
        assert_close(&gpu.attn_backward.dx, &cpu.attn_backward.dx, 1e-4, 1e-5);
        assert_close(&gpu.attn_backward.dw_q, &cpu.attn_backward.dw_q, 1e-4, 1e-5);
        assert_close(&gpu.attn_backward.dw_k, &cpu.attn_backward.dw_k, 1e-4, 1e-5);
        assert_close(&gpu.attn_backward.dw_v, &cpu.attn_backward.dw_v, 1e-4, 1e-5);
        assert_close(&gpu.attn_backward.dw_o, &cpu.attn_backward.dw_o, 1e-4, 1e-5);
        assert_close(&gpu.ffn_backward.dx, &cpu.ffn_backward.dx, 1e-4, 1e-5);
        assert_close(
            &gpu.ffn_backward.dw_gate,
            &cpu.ffn_backward.dw_gate,
            1e-4,
            1e-5,
        );
        assert_close(&gpu.ffn_backward.dw_up, &cpu.ffn_backward.dw_up, 1e-4, 1e-5);
        assert_close(
            &gpu.ffn_backward.dw_down,
            &cpu.ffn_backward.dw_down,
            1e-4,
            1e-5,
        );
        assert_close(&gpu.d_gamma_1, &cpu.d_gamma_1, 1e-4, 1e-5);
        assert_close(&gpu.d_gamma_2, &cpu.d_gamma_2, 1e-4, 1e-5);
    }

    #[test]
    fn test_one_seq() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let mut cache = TransformerBlockForwardCache::default();

        let seq = 1usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_one_n_heads() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            n_heads: 1,
            ..Default::default()
        };
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let mut cache = TransformerBlockForwardCache::default();

        let seq = 64usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_d_model_equals_d_ff() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            d_model: 64,
            d_ff: 64,
            ..Default::default()
        };
        let mut cache_cpu = TransformerBlockForwardCache::default();
        let mut cache = TransformerBlockForwardCache::default();

        let seq = 64usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
