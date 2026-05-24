use crate::{
    gpu_context::GpuContext,
    kernel::{residual_add::residual_add_gpu, rms_norm::rms_norm_gpu, swiglu::swiglu_gpu},
    model::multi_head_attention::MultiHeadAttention,
    model_config::ModelConfig,
    util::random_f32,
};

pub struct TransformerBlock {
    pub mha: MultiHeadAttention,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub gamma_1: Vec<f32>, // RMSNorm の γ
    pub gamma_2: Vec<f32>,
}

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            mha: MultiHeadAttention::new(&cfg),
            w_gate: random_f32(cfg.d_model * cfg.d_ff, 36),
            w_up: random_f32(cfg.d_model * cfg.d_ff, 37),
            w_down: random_f32(cfg.d_ff * cfg.d_model, 38),
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
    ) -> Vec<f32> {
        let seq = x.len() / cfg.d_model;
        let norm1 = rms_norm_gpu(&ctx, &x, &self.gamma_1, cfg.eps, cfg.d_model as u32);
        let mha = self.mha.forward(ctx, cfg, &norm1, cos_table, sin_table);
        let add = residual_add_gpu(&ctx, x, &mha);
        let norm2 = rms_norm_gpu(&ctx, &add, &self.gamma_2, cfg.eps, cfg.d_model as u32);
        let ffn = swiglu_gpu(
            &ctx,
            &norm2,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
        );
        let out = residual_add_gpu(&ctx, &add, &ffn);

        out
    }

    // CPU リファレンス
    #[cfg(test)]
    pub fn forward_cpu(
        &self,
        cfg: &ModelConfig,
        x: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
    ) -> Vec<f32> {
        use crate::kernel::{
            residual_add::residual_add_cpu, rms_norm::rms_norm_cpu, swiglu::swiglu_cpu,
        };
        let seq = x.len() / cfg.d_model;
        let norm1 = rms_norm_cpu(&x, &self.gamma_1, cfg.eps, cfg.d_model);
        let mha = self.mha.forward_cpu(cfg, &norm1, cos_table, sin_table);
        let add = residual_add_cpu(x, &mha);
        let norm2 = rms_norm_cpu(&add, &self.gamma_2, cfg.eps, cfg.d_model);
        let ffn = swiglu_cpu(
            &norm2,
            &self.w_gate,
            &self.w_up,
            &self.w_down,
            seq,
            cfg.d_model,
            cfg.d_ff,
        );
        let out = residual_add_cpu(&add, &ffn);

        out
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext, kernel::rope::create_table,
        model::transformer_block::TransformerBlock, model_config::ModelConfig,
        test_utils::assert_close, util::random_f32,
    };

    #[test]
    fn test_transformer_block() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let seq = 64usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
    }

    #[test]
    fn test_one_seq() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let seq = 1usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
    }

    #[test]
    fn test_one_n_heads() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            n_heads: 1,
            ..Default::default()
        };
        let seq = 64usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

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
        let seq = 64usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);

        let base: f32 = 10000.0;
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

        let tf = TransformerBlock::new(&cfg);
        let cpu = tf.forward_cpu(&cfg, &x, &cos_table, &sin_table);
        let gpu = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

        assert_close(&gpu, &cpu, 1e-2, 1e-3);
    }
}
