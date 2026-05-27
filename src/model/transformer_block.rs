use crate::{
    gpu_context::GpuContext,
    kernel::{residual_add::residual_add, rms_norm::rms_norm},
    model::{
        ffn::{Ffn, FfnForwardCache},
        attention::{Attention, AttentionForwardCache},
    },
    model_config::ModelConfig,
    util::random_f32,
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

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            attn: Attention::new(&cfg),
            ffn: Ffn::new(cfg),
            gamma_1: random_f32(cfg.d_model, 38),
            gamma_2: random_f32(cfg.d_model, 38),
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
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
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
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
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
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 2e-1, 1e-2);
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
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
    }
}
