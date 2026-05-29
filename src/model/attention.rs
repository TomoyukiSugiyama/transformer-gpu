use crate::{
    gpu_context::GpuContext,
    kernel::{attention::attention, matmul::matmul, rope::rope},
    model_config::ModelConfig,
    util::{concat_columns_into, random_f32, split_columns},
};

#[derive(Default)]
pub struct AttentionForwardCache {
    pub q: Vec<f32>,             // (seq × d_model)
    pub k: Vec<f32>,             // (seq × d_model)
    pub v: Vec<f32>,             // (seq × d_model)
    pub q_rope: Vec<Vec<f32>>,   // [n_heads] (seq × d_head)
    pub k_rope: Vec<Vec<f32>>,   // [n_heads] (seq × d_head)
    pub v_heads: Vec<Vec<f32>>,  // [n_heads] (seq × d_head)
    pub attn_out: Vec<Vec<f32>>, // [n_heads] (seq × d_head)
    pub wo_in: Vec<f32>,         // (seq × d_model) w_o への入力
}

pub struct Attention {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

impl Attention {
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
        cache: &mut AttentionForwardCache,
    ) -> Vec<f32> {
        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(cfg.n_heads > 0, "n_heads must be > 0");
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        cache.q = matmul(
            ctx,
            x,
            &self.w_q,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        cache.k = matmul(
            ctx,
            x,
            &self.w_k,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        cache.v = matmul(
            ctx,
            x,
            &self.w_v,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let q_heads = split_columns(
            &cache.q,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let k_heads = split_columns(
            &cache.k,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        cache.v_heads = split_columns(
            &cache.v,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        cache.q_rope = q_heads
            .iter()
            .map(|q| rope(ctx, q, d_head as usize, &cos_table, &sin_table))
            .collect();
        cache.k_rope = k_heads
            .iter()
            .map(|k| rope(ctx, k, d_head as usize, &cos_table, &sin_table))
            .collect();

        cache.attn_out = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            cache.attn_out.push(attention(
                ctx,
                &cache.q_rope[i],
                &cache.k_rope[i],
                &cache.v_heads[i],
                seq as u32,
                d_head as u32,
            ));
        }

        cache.wo_in = concat_columns_into(&cache.attn_out, seq, cfg.d_model, d_head, cfg.n_heads);

        matmul(
            ctx,
            &cache.wo_in,
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
        cache: &mut AttentionForwardCache,
    ) -> Vec<f32> {
        use crate::kernel::{matmul::matmul_cpu, rope::rope_cpu};

        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(cfg.n_heads > 0, "n_heads must be > 0");
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        cache.q = matmul_cpu(x, &self.w_q, seq, cfg.d_model, cfg.d_model);
        cache.k = matmul_cpu(x, &self.w_k, seq, cfg.d_model, cfg.d_model);
        cache.v = matmul_cpu(x, &self.w_v, seq, cfg.d_model, cfg.d_model);
        let q_heads = split_columns(
            &cache.q,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let k_heads = split_columns(
            &cache.k,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );
        let v_heads = split_columns(
            &cache.v,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        cache.q_rope = q_heads
            .iter()
            .map(|q| rope_cpu(q, d_head, &cos_table, &sin_table))
            .collect();
        cache.k_rope = k_heads
            .iter()
            .map(|k| rope_cpu(k, d_head, &cos_table, &sin_table))
            .collect();
        cache.attn_out = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            use crate::kernel::attention::attention_cpu;
            cache.attn_out.push(attention_cpu(
                &cache.q_rope[i],
                &cache.k_rope[i],
                &v_heads[i],
                seq,
                d_head,
            ));
        }

        cache.wo_in = concat_columns_into(
            &cache.attn_out,
            seq as usize,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        matmul_cpu(&cache.wo_in, &self.w_o, seq, cfg.d_model, cfg.d_model)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::rope::create_table,
        model::attention::{Attention, AttentionForwardCache},
        model_config::ModelConfig,
        test_utils::assert_close,
        util::random_f32,
    };

    #[test]
    fn test_multi_head_attention() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = AttentionForwardCache::default();
        let mut cache = AttentionForwardCache::default();
        let attn = Attention::new(&cfg);
        let seq = 64usize;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31);
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let cpu = attn.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = attn.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-3, 1e-4);
    }
}
