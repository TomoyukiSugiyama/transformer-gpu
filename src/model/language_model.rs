use crate::{
    gpu_context::GpuContext,
    kernel::{
        embedding::embedding, matmul::matmul_forward, rms_norm::rms_norm, rope::create_table,
    },
    model::transformer_block::{TransformerBlock, TransformerBlockForwardCache},
    model_config::ModelConfig,
    util::random_f32,
};

#[derive(Default)]
pub struct LanguageModelForwardCache {
    pub token_ids: Vec<u32>,
    pub x0: Vec<f32>,
    pub blocks: Vec<TransformerBlockForwardCache>,
    pub final_norm_in: Vec<f32>,
    pub final_norm_out: Vec<f32>,
    pub logits: Vec<f32>,
}

impl LanguageModelForwardCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            blocks: (0..n_layers)
                .map(|_| TransformerBlockForwardCache::default())
                .collect(),
            ..Default::default()
        }
    }
}

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

    pub fn forward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        token_ids: &[u32],
        cache: &mut LanguageModelForwardCache,
    ) -> Vec<f32> {
        let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        cache.token_ids = token_ids.to_vec();

        let mut x = embedding(ctx, token_ids, &self.embedding, cfg.d_model);
        cache.x0 = x.clone();

        self.blocks
            .iter()
            .zip(cache.blocks.iter_mut())
            .for_each(|(block, cache)| {
                x = block.forward(ctx, cfg, &x, &cos_table, &sin_table, cache);
            });

        cache.final_norm_in = x.clone();
        cache.final_norm_out = rms_norm(ctx, &x, &self.final_gamma, cfg.eps, cfg.d_model as u32);

        cache.logits = matmul_forward(
            ctx,
            &cache.final_norm_out,
            &self.lm_head,
            seq as u32,
            cfg.d_model as u32,
            cfg.vocab_size as u32,
        );

        cache.logits.clone()
    }

    #[cfg(test)]
    pub fn forward_cpu(
        &self,
        cfg: &ModelConfig,
        token_ids: &[u32],
        cache: &mut LanguageModelForwardCache,
    ) -> Vec<f32> {
        use crate::kernel::{
            embedding::embedding_cpu, matmul::matmul_forward_cpu, rms_norm::rms_norm_cpu,
        };

        let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        cache.token_ids = token_ids.to_vec();

        let mut x = embedding_cpu(token_ids, &self.embedding, cfg.d_model);
        cache.x0 = x.clone();

        self.blocks
            .iter()
            .zip(cache.blocks.iter_mut())
            .for_each(|(block, cache)| {
                x = block.forward_cpu(cfg, &x, &cos_table, &sin_table, cache);
            });

        cache.final_norm_in = x.clone();
        cache.final_norm_out = rms_norm_cpu(&x, &self.final_gamma, cfg.eps, cfg.d_model);

        cache.logits = matmul_forward_cpu(
            &cache.final_norm_out,
            &self.lm_head,
            seq,
            cfg.d_model,
            cfg.vocab_size,
        );

        cache.logits.clone()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        model::language_model::{LanguageModel, LanguageModelForwardCache},
        model_config::ModelConfig,
        test_utils::{assert_close, random_token_ids},
    };

    #[test]
    fn test_language_model() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = LanguageModelForwardCache::new(cfg.n_layers);
        let mut cache = LanguageModelForwardCache::new(cfg.n_layers);

        let token_ids = random_token_ids(64, cfg.vocab_size, 10);

        let lm = LanguageModel::new(&cfg);
        let cpu = lm.forward_cpu(&cfg, &token_ids, &mut cache_cpu);
        let gpu = lm.forward(&ctx, &cfg, &token_ids, &mut cache);

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
