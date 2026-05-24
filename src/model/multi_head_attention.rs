use crate::{
    gpu_context::GpuContext,
    kernel::{attention::attention_gpu, matmul::matmul_gpu, rope::rope_gpu},
    model_config::ModelConfig,
    util::{concat_columns_into, random_f32, split_columns},
};
pub struct MultiHeadAttention {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl MultiHeadAttention {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            w_q: random_f32(cfg.d_model * cfg.d_model, 32),
            w_k: random_f32(cfg.d_model * cfg.d_model, 33),
            w_v: random_f32(cfg.d_model * cfg.d_model, 34),
            w_o: random_f32(cfg.d_model * cfg.d_model, 35),
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
        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(cfg.n_heads > 0, "n_heads must be > 0");
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        let q = matmul_gpu(
            ctx,
            x,
            &self.w_q,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let k = matmul_gpu(
            ctx,
            x,
            &self.w_k,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let v = matmul_gpu(
            ctx,
            x,
            &self.w_v,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let q_heads = split_columns(
            &q,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let k_heads = split_columns(
            &k,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let v_heads = split_columns(
            &v,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        let q_heads: Vec<Vec<f32>> = q_heads
            .iter()
            .map(|q| rope_gpu(ctx, q, d_head as usize, &cos_table, &sin_table))
            .collect();
        let k_heads: Vec<Vec<f32>> = k_heads
            .iter()
            .map(|k| rope_gpu(ctx, k, d_head as usize, &cos_table, &sin_table))
            .collect();

        let mut attn = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            attn.push(attention_gpu(
                ctx,
                &q_heads[i],
                &k_heads[i],
                &v_heads[i],
                seq as u32,
                d_head as u32,
            ));
        }

        let attn = concat_columns_into(&attn, seq, cfg.d_model, d_head, cfg.n_heads);

        matmul_gpu(
            ctx,
            &attn,
            &self.w_o,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        )
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
        use crate::kernel::{matmul::matmul_cpu, rope::rope_cpu};

        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(cfg.n_heads > 0, "n_heads must be > 0");
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        let q = matmul_cpu(x, &self.w_q, seq, cfg.d_model, cfg.d_model);
        let k = matmul_cpu(x, &self.w_k, seq, cfg.d_model, cfg.d_model);
        let v = matmul_cpu(x, &self.w_v, seq, cfg.d_model, cfg.d_model);
        let q_heads = split_columns(
            &q,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let k_heads = split_columns(
            &k,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let v_heads = split_columns(
            &v,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        let q_heads: Vec<Vec<f32>> = q_heads
            .iter()
            .map(|q| rope_cpu(q, d_head, &cos_table, &sin_table))
            .collect();
        let k_heads: Vec<Vec<f32>> = k_heads
            .iter()
            .map(|k| rope_cpu(k, d_head, &cos_table, &sin_table))
            .collect();
        let mut attn = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            use crate::kernel::attention::attention_cpu;

            attn.push(attention_cpu(
                &q_heads[i],
                &k_heads[i],
                &v_heads[i],
                seq,
                d_head,
            ));
        }

        let attn = concat_columns_into(
            &attn,
            seq as usize,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        matmul_cpu(&attn, &self.w_o, seq, cfg.d_model, cfg.d_model)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext, kernel::rope::create_table,
        model::multi_head_attention::MultiHeadAttention, model_config::ModelConfig,
        test_utils::assert_close, util::random_f32,
    };

    #[test]
    fn test_multi_head_attention() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mha = MultiHeadAttention::new(&cfg);
        let seq = 64usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let cpu = mha.forward_cpu(&cfg, &x, &cos_table, &sin_table);
        let gpu = mha.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
