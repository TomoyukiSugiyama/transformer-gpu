use crate::{
    gpu_context::GpuContext,
    kernel::{
        attention::{attention, attention_backward},
        matmul::{matmul_backward, matmul_forward},
        rope::{rope, rope_backward},
    },
    model_config::ModelConfig,
    util::{concat_columns_into, random_f32, split_columns},
};

#[derive(Default)]
pub struct AttentionForwardCache {
    pub x: Vec<f32>,             // (seq × d_model)
    pub q_rope: Vec<Vec<f32>>,   // [n_heads] (seq × d_head)
    pub k_rope: Vec<Vec<f32>>,   // [n_heads] (seq × d_head)
    pub v_heads: Vec<Vec<f32>>,  // [n_heads] (seq × d_head)
    pub attn_out: Vec<Vec<f32>>, // [n_heads] (seq × d_head)
    pub attn_l: Vec<Vec<f32>>,   // [n_heads] (seq)
    pub wo_in: Vec<f32>,         // (seq × d_model) w_o への入力
}

pub struct Attention {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
}

pub struct AttentionBackward {
    pub dx: Vec<f32>,
    pub dw_q: Vec<f32>,
    pub dw_k: Vec<f32>,
    pub dw_v: Vec<f32>,
    pub dw_o: Vec<f32>,
}

impl Attention {
    pub fn new(cfg: &ModelConfig) -> Self {
        let scale = (1.0 / cfg.d_model as f32).sqrt();
        Self {
            w_q: random_f32(cfg.d_model * cfg.d_model, 32, scale),
            w_k: random_f32(cfg.d_model * cfg.d_model, 33, scale),
            w_v: random_f32(cfg.d_model * cfg.d_model, 34, scale),
            w_o: random_f32(cfg.d_model * cfg.d_model, 35, scale),
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
        assert!(cfg.d_model > 0);
        assert!(cfg.n_heads > 0);
        assert_eq!(
            cfg.d_model % cfg.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        assert_eq!(
            x.len() % cfg.d_model,
            0,
            "x length must be divisible by d_model"
        );
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        cache.x = x.to_vec();
        let q = matmul_forward(
            ctx,
            x,
            &self.w_q,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let k = matmul_forward(
            ctx,
            x,
            &self.w_k,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );
        let v = matmul_forward(
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
        cache.v_heads = split_columns(
            &v,
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
        cache.attn_l = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            let (o, l) = attention(
                ctx,
                &cache.q_rope[i],
                &cache.k_rope[i],
                &cache.v_heads[i],
                seq as u32,
                d_head as u32,
            );
            cache.attn_out.push(o);
            cache.attn_l.push(l);
        }

        cache.wo_in = concat_columns_into(&cache.attn_out, seq, cfg.d_model, d_head, cfg.n_heads);

        matmul_forward(
            ctx,
            &cache.wo_in,
            &self.w_o,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        )
    }

    pub fn backward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        dy: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut AttentionForwardCache,
    ) -> AttentionBackward {
        assert!(cfg.d_model > 0);
        assert!(cfg.n_heads > 0);

        assert_eq!(
            dy.len() % cfg.d_model,
            0,
            "dy length must be divisible by d_model"
        );

        assert_eq!(
            dy.len(),
            cache.x.len(),
            "dy and cached x must have the same length"
        );
        let seq = dy.len() / cfg.d_model;
        let d_head = cfg.d_head();

        // dW_o = dy @ W_o^T
        // dW = dW_o_in^T @ dy
        let (dwo_in, dw_o) = matmul_backward(
            ctx,
            &dy,
            &cache.wo_in,
            &self.w_o,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );

        let d_atten_out = split_columns(
            &dwo_in,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        let mut dq_rope = Vec::with_capacity(cfg.n_heads as usize);
        let mut dk_rope = Vec::with_capacity(cfg.n_heads as usize);
        let mut dv_heads = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads {
            let (dq, dk, dv) = attention_backward(
                ctx,
                &d_atten_out[i],
                &cache.q_rope[i],
                &cache.k_rope[i],
                &cache.v_heads[i],
                &cache.attn_out[i],
                &cache.attn_l[i],
                seq as u32,
                d_head as u32,
            );
            dq_rope.push(dq);
            dk_rope.push(dk);
            dv_heads.push(dv);
        }

        let dq_heads: Vec<Vec<f32>> = dq_rope
            .iter()
            .map(|dq| rope_backward(ctx, dq, d_head as usize, &cos_table, &sin_table))
            .collect();
        let dk_heads: Vec<Vec<f32>> = dk_rope
            .iter()
            .map(|dk| rope_backward(ctx, dk, d_head as usize, &cos_table, &sin_table))
            .collect();

        let dq = concat_columns_into(&dq_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_q, dw_q) = matmul_backward(
            ctx,
            &dq,
            &cache.x,
            &self.w_q,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );

        let dk = concat_columns_into(&dk_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_k, dw_k) = matmul_backward(
            ctx,
            &dk,
            &cache.x,
            &self.w_k,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );

        let dv = concat_columns_into(&dv_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_v, dw_v) = matmul_backward(
            ctx,
            &dv,
            &cache.x,
            &self.w_v,
            seq as u32,
            cfg.d_model as u32,
            cfg.d_model as u32,
        );

        let dx = dx_q
            .iter()
            .zip(dx_k.iter())
            .zip(dx_v.iter())
            .map(|((q_i, k_i), v_i)| q_i + k_i + v_i)
            .collect();

        AttentionBackward {
            dx,
            dw_q,
            dw_k,
            dw_v,
            dw_o,
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
        cache: &mut AttentionForwardCache,
    ) -> Vec<f32> {
        use crate::kernel::{matmul::matmul_forward_cpu, rope::rope_cpu};

        assert!(
            cfg.d_model % cfg.n_heads == 0,
            "d_model must to divisible by n_heads"
        );
        assert!(cfg.n_heads > 0, "n_heads must be > 0");
        let seq = x.len() / cfg.d_model;
        let d_head = cfg.d_head();

        cache.x = x.to_vec();
        let q = matmul_forward_cpu(x, &self.w_q, seq, cfg.d_model, cfg.d_model);
        let k = matmul_forward_cpu(x, &self.w_k, seq, cfg.d_model, cfg.d_model);
        let v = matmul_forward_cpu(x, &self.w_v, seq, cfg.d_model, cfg.d_model);
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
        cache.v_heads = split_columns(
            &v,
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
        cache.attn_l = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads as usize {
            use crate::kernel::attention::attention_cpu;
            let (o, l) = attention_cpu(
                &cache.q_rope[i],
                &cache.k_rope[i],
                &cache.v_heads[i],
                seq,
                d_head,
            );
            cache.attn_out.push(o);
            cache.attn_l.push(l);
        }

        cache.wo_in = concat_columns_into(
            &cache.attn_out,
            seq as usize,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        matmul_forward_cpu(&cache.wo_in, &self.w_o, seq, cfg.d_model, cfg.d_model)
    }

    #[cfg(test)]
    pub fn backward_cpu(
        &self,
        cfg: &ModelConfig,
        dy: &[f32],
        cos_table: &[f32],
        sin_table: &[f32],
        cache: &mut AttentionForwardCache,
    ) -> AttentionBackward {
        use crate::kernel::{matmul::matmul_backward_cpu, rope::rope_backward_cpu};

        let seq = dy.len() / cfg.d_model;
        let d_head = cfg.d_head();

        // dW_o = dy @ W_o^T
        // dW = dW_o_in^T @ dy
        let (dwo_in, dw_o) =
            matmul_backward_cpu(&dy, &cache.wo_in, &self.w_o, seq, cfg.d_model, cfg.d_model);

        let d_atten_out = split_columns(
            &dwo_in,
            cfg.d_model as usize,
            d_head as usize,
            cfg.n_heads as usize,
        );

        let mut dq_rope = Vec::with_capacity(cfg.n_heads as usize);
        let mut dk_rope = Vec::with_capacity(cfg.n_heads as usize);
        let mut dv_heads = Vec::with_capacity(cfg.n_heads as usize);
        for i in 0..cfg.n_heads {
            use crate::kernel::attention::attention_backward_cpu;

            let (dq, dk, dv) = attention_backward_cpu(
                &d_atten_out[i],
                &cache.q_rope[i],
                &cache.k_rope[i],
                &cache.v_heads[i],
                &cache.attn_out[i],
                &cache.attn_l[i],
                seq,
                d_head,
            );
            dq_rope.push(dq);
            dk_rope.push(dk);
            dv_heads.push(dv);
        }

        let dq_heads: Vec<Vec<f32>> = dq_rope
            .iter()
            .map(|dq| rope_backward_cpu(dq, d_head as usize, &cos_table, &sin_table))
            .collect();
        let dk_heads: Vec<Vec<f32>> = dk_rope
            .iter()
            .map(|dk| rope_backward_cpu(dk, d_head as usize, &cos_table, &sin_table))
            .collect();

        let dq = concat_columns_into(&dq_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_q, dw_q) =
            matmul_backward_cpu(&dq, &cache.x, &self.w_q, seq, cfg.d_model, cfg.d_model);

        let dk = concat_columns_into(&dk_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_k, dw_k) =
            matmul_backward_cpu(&dk, &cache.x, &self.w_k, seq, cfg.d_model, cfg.d_model);

        let dv = concat_columns_into(&dv_heads, seq, cfg.d_model, d_head, cfg.n_heads);
        let (dx_v, dw_v) =
            matmul_backward_cpu(&dv, &cache.x, &self.w_v, seq, cfg.d_model, cfg.d_model);

        let dx = dx_q
            .iter()
            .zip(dx_k.iter())
            .zip(dx_v.iter())
            .map(|((q_i, k_i), v_i)| q_i + k_i + v_i)
            .collect();

        AttentionBackward {
            dx,
            dw_q,
            dw_k,
            dw_v,
            dw_o,
        }
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
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let cpu = attn.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = attn.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_multi_head_attention_backward() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            ..Default::default()
        };
        let mut cache_cpu = AttentionForwardCache::default();
        let mut cache = AttentionForwardCache::default();
        let attn = Attention::new(&cfg);
        let seq = 64usize;
        let scale = 0.1f32;
        let x: Vec<f32> = random_f32(seq * cfg.d_model, 31, scale);
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        let dy_cpu = attn.forward_cpu(&cfg, &x, &cos_table, &sin_table, &mut cache_cpu);
        let dy_gpu = attn.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache);

        let cpu = attn.backward_cpu(&cfg, &dy_cpu, &cos_table, &sin_table, &mut cache_cpu);
        let gpu = attn.backward(&ctx, &cfg, &dy_gpu, &cos_table, &sin_table, &mut cache);

        assert_close(&gpu.dx, &cpu.dx, 1e-4, 1e-5);
        assert_close(&gpu.dw_q, &cpu.dw_q, 1e-4, 1e-5);
        assert_close(&gpu.dw_k, &cpu.dw_k, 1e-4, 1e-5);
        assert_close(&gpu.dw_v, &cpu.dw_v, 1e-4, 1e-5);
        assert_close(&gpu.dw_o, &cpu.dw_o, 1e-4, 1e-5);
    }
}
