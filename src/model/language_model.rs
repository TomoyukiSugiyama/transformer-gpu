use crate::{
    checkpoint::{Checkpointable, WeightMap},
    gpu_context::GpuContext,
    kernel::{
        embedding::{embedding, embedding_backward},
        matmul::{matmul_backward, matmul_forward},
        rms_norm::{rms_norm, rms_norm_backward},
        rope::create_table,
    },
    model::transformer_block::{
        TransformerBlock, TransformerBlockBackward, TransformerBlockForwardCache,
    },
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

pub struct LanguageModelBackward {
    pub dx: Vec<f32>,
    pub d_embedding: Vec<f32>,
    pub d_blocks: Vec<TransformerBlockBackward>,
    pub d_final_gamma: Vec<f32>,
    pub d_lm_head: Vec<f32>,
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

    pub fn backward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        d_logits: &[f32],
        cache: &mut LanguageModelForwardCache,
    ) -> LanguageModelBackward {
        let seq = cache.token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let (dx, d_lm_head) = matmul_backward(
            &ctx,
            &d_logits,
            &cache.final_norm_out,
            &self.lm_head,
            seq as u32,
            cfg.d_model as u32,
            cfg.vocab_size as u32,
        );

        let (mut dx, d_final_gamma) = rms_norm_backward(
            &ctx,
            &dx,
            &cache.final_norm_in,
            &self.final_gamma,
            cfg.eps,
            cfg.d_model as u32,
        );

        let mut d_blocks = Vec::with_capacity(cfg.n_layers as usize);
        for i in (0..cfg.n_layers).rev() {
            let backward = self.blocks[i].backward(
                &ctx,
                cfg,
                &dx,
                &cos_table,
                &sin_table,
                &mut cache.blocks[i],
            );
            dx = backward.dx.clone();
            d_blocks.push(backward);
        }
        d_blocks.reverse();

        let d_embedding =
            embedding_backward(&ctx, &dx, &cache.token_ids, cfg.vocab_size, cfg.d_model);

        LanguageModelBackward {
            dx,
            d_embedding,
            d_blocks,
            d_final_gamma,
            d_lm_head,
        }
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

    #[cfg(test)]
    pub fn backward_cpu(
        &self,
        cfg: &ModelConfig,
        d_logits: &[f32],
        cache: &mut LanguageModelForwardCache,
    ) -> LanguageModelBackward {
        use crate::kernel::{
            embedding::embedding_backward_cpu, matmul::matmul_backward_cpu,
            rms_norm::rms_norm_backward_cpu,
        };

        let seq = cache.token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let (dx, d_lm_head) = matmul_backward_cpu(
            &d_logits,
            &cache.final_norm_out,
            &self.lm_head,
            seq,
            cfg.d_model,
            cfg.vocab_size,
        );

        let (mut dx, d_final_gamma) = rms_norm_backward_cpu(
            &dx,
            &cache.final_norm_in,
            &self.final_gamma,
            cfg.eps,
            cfg.d_model,
        );

        let mut d_blocks = Vec::with_capacity(cfg.n_layers as usize);
        for i in (0..cfg.n_layers).rev() {
            let backward =
                self.blocks[i].backward_cpu(cfg, &dx, &cos_table, &sin_table, &mut cache.blocks[i]);
            dx = backward.dx.clone();
            d_blocks.push(backward);
        }
        d_blocks.reverse();

        let d_embedding =
            embedding_backward_cpu(&dx, &cache.token_ids, cfg.vocab_size, cfg.d_model);

        LanguageModelBackward {
            dx,
            d_embedding,
            d_blocks,
            d_final_gamma,
            d_lm_head,
        }
    }
}

impl Checkpointable for LanguageModel {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_vector("embedding", self.embedding.clone());
        map.insert_vector("final_gamma", self.final_gamma.clone());
        map.insert_vector("lm_head", self.lm_head.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            map.insert_vector(&format!("block.{i}.gamma_1"), block.gamma_1.clone());
            map.insert_vector(&format!("block.{i}.gamma_2"), block.gamma_2.clone());
            map.insert_vector(&format!("block.{i}.wq"), block.attn.w_q.clone());
            map.insert_vector(&format!("block.{i}.wk"), block.attn.w_k.clone());
            map.insert_vector(&format!("block.{i}.wv"), block.attn.w_v.clone());
            map.insert_vector(&format!("block.{i}.wo"), block.attn.w_o.clone());
            map.insert_vector(&format!("block.{i}.w_gate"), block.ffn.w_gate.clone());
            map.insert_vector(&format!("block.{i}.w_up"), block.ffn.w_up.clone());
            map.insert_vector(&format!("block.{i}.w_down"), block.ffn.w_down.clone());
        }
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> std::io::Result<()> {
        self.embedding = map.get_vector("embedding")?.clone();
        self.final_gamma = map.get_vector("final_gamma")?.clone();
        self.lm_head = map.get_vector("lm_head")?.clone();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.gamma_1 = map.get_vector(&format!("block.{i}.gamma_1"))?.clone();
            block.gamma_2 = map.get_vector(&format!("block.{i}.gamma_2"))?.clone();
            block.attn.w_q = map.get_vector(&format!("block.{i}.wq"))?.clone();
            block.attn.w_k = map.get_vector(&format!("block.{i}.wk"))?.clone();
            block.attn.w_v = map.get_vector(&format!("block.{i}.wv"))?.clone();
            block.attn.w_o = map.get_vector(&format!("block.{i}.wo"))?.clone();
            block.ffn.w_gate = map.get_vector(&format!("block.{i}.w_gate"))?.clone();
            block.ffn.w_up = map.get_vector(&format!("block.{i}.w_up"))?.clone();
            block.ffn.w_down = map.get_vector(&format!("block.{i}.w_down"))?.clone();
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::cross_entropy_loss::{cross_entropy_loss, cross_entropy_loss_cpu},
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

    #[test]
    fn test_language_model_backward() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = LanguageModelForwardCache::new(cfg.n_layers);
        let mut cache = LanguageModelForwardCache::new(cfg.n_layers);

        let token_ids = random_token_ids(64, cfg.vocab_size, 10);
        let seq = token_ids.len();
        let input_ids = &token_ids[..seq - 1];
        let input_seq = input_ids.len();
        let targets: Vec<usize> = token_ids[1..].iter().map(|&t| t as usize).collect();

        let lm = LanguageModel::new(&cfg);
        let logits_cpu = lm.forward_cpu(&cfg, &input_ids, &mut cache_cpu);
        let logits_gpu = lm.forward(&ctx, &cfg, &input_ids, &mut cache);

        let (loss_cpu, d_logits_cpu) =
            cross_entropy_loss_cpu(&logits_cpu, &targets, input_seq, cfg.vocab_size);
        let (loss_gpu, d_logits_gpu) =
            cross_entropy_loss(&ctx, &logits_gpu, &targets, input_seq, cfg.vocab_size);

        let cpu = lm.backward_cpu(&cfg, &d_logits_cpu, &mut cache_cpu);
        let gpu = lm.backward(&ctx, &cfg, &d_logits_gpu, &mut cache);

        assert!((loss_gpu - loss_cpu).abs() < 1e-4);
        assert_close(&gpu.dx, &cpu.dx, 5e-1, 1e-3);

        gpu.d_blocks
            .iter()
            .zip(cpu.d_blocks.iter())
            .for_each(|(g_block, c_block)| {
                assert_close(&g_block.dx, &c_block.dx, 5e-1, 1e-3);
                assert_close(&g_block.d_gamma_1, &c_block.d_gamma_1, 5e-1, 1e-3);
                assert_close(&g_block.d_gamma_2, &c_block.d_gamma_2, 5e-1, 1e-3);
                assert_close(
                    &g_block.ffn_backward.dx,
                    &c_block.ffn_backward.dx,
                    5e-1,
                    1e-3,
                );
                assert_close(
                    &g_block.ffn_backward.dw_gate,
                    &c_block.ffn_backward.dw_gate,
                    5e-1,
                    1e-3,
                );
                assert_close(
                    &g_block.ffn_backward.dw_up,
                    &c_block.ffn_backward.dw_up,
                    5e-1,
                    1e-3,
                );
                assert_close(
                    &g_block.ffn_backward.dw_down,
                    &c_block.ffn_backward.dw_down,
                    5e-1,
                    1e-3,
                );
            });
        assert_close(&gpu.d_embedding, &cpu.d_embedding, 5e-1, 1e-3);
        assert_close(&gpu.d_final_gamma, &cpu.d_final_gamma, 5e-1, 1e-3);
        assert_close(&gpu.d_lm_head, &cpu.d_lm_head, 5e-1, 1e-3);
    }
}
