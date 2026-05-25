use crate::{
    gpu_context::GpuContext,
    kernel::{
        embedding::embedding_gpu, matmul::matmul_gpu, rms_norm::rms_norm_gpu, rope::create_table,
    },
    model::transformer_block::TransformerBlock,
    model_config::ModelConfig,
    util::random_f32,
};

pub struct LanguageModel {
    pub embedding: Vec<f32>,
    pub blocks: Vec<TransformerBlock>,
    pub final_gamma: Vec<f32>,
    pub lm_head: Vec<f32>,
}

impl LanguageModel {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            embedding: random_f32(cfg.vocab_size * cfg.d_model, 10),
            blocks: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(cfg))
                .collect(),
            final_gamma: random_f32(cfg.d_model, 11),
            lm_head: random_f32(cfg.d_model * cfg.vocab_size, 12),
        }
    }

    pub fn forward(&self, ctx: &GpuContext, cfg: &ModelConfig, token_ids: &[u32]) -> Vec<f32> {
        let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);
        let mut x = embedding_gpu(ctx, token_ids, &self.embedding, cfg.d_model);

        for block in &self.blocks {
            x = block.forward(ctx, cfg, &x, &cos_table, &sin_table);
        }
        x = rms_norm_gpu(ctx, &x, &self.final_gamma, cfg.eps, cfg.d_model as u32);
        let logits = matmul_gpu(
            ctx,
            &x,
            &self.lm_head,
            seq as u32,
            cfg.d_model as u32,
            cfg.vocab_size as u32,
        );

        logits
    }

    #[cfg(test)]
    pub fn forward_cpu(&self, cfg: &ModelConfig, token_ids: &[u32]) -> Vec<f32> {
        use crate::kernel::{embedding::embedding_cpu, matmul::matmul_cpu, rms_norm::rms_norm_cpu};

        let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);
        let mut x = embedding_cpu(token_ids, &self.embedding, cfg.d_model);

        for block in &self.blocks {
            x = block.forward_cpu(cfg, &x, &cos_table, &sin_table);
        }
        x = rms_norm_cpu(&x, &self.final_gamma, cfg.eps, cfg.d_model);
        let logits = matmul_cpu(&x, &self.lm_head, seq, cfg.d_model, cfg.vocab_size);

        logits
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        model::language_model::LanguageModel,
        model_config::ModelConfig,
        test_utils::{assert_close, random_token_ids},
    };

    #[test]
    fn test_language_model() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };

        let token_ids = random_token_ids(64, cfg.vocab_size, 10);

        let lm = LanguageModel::new(&cfg);
        let cpu = lm.forward_cpu(&cfg, &token_ids);
        let gpu = lm.forward(&ctx, &cfg, &token_ids);

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
