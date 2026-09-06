use wgpu::{BufferUsages, util::DeviceExt};

use crate::{
    checkpoint::{Checkpointable, WeightMap},
    gpu_context::GpuContext,
    gpu_tensor::{GpuTensor, read_f32_tensor},
    kernel::{
        cross_entropy_loss::create_cross_entropy_bind_group,
        embedding::{embedding, embedding_backward},
        matmul::{create_matmul_bind_group, encode_matmul_into, matmul_backward, matmul_forward},
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

impl LanguageModelBackward {
    pub fn scale(&mut self, s: f32) {
        for v in self.d_embedding.iter_mut() {
            *v *= s;
        }
        for v in self.d_final_gamma.iter_mut() {
            *v *= s;
        }
        for v in self.d_lm_head.iter_mut() {
            *v *= s;
        }
        for b in self.d_blocks.iter_mut() {
            for v in b.d_gamma_1.iter_mut() {
                *v *= s;
            }
            for v in b.d_gamma_2.iter_mut() {
                *v *= s;
            }
            for v in b.attn_backward.dw_q.iter_mut() {
                *v *= s;
            }
            for v in b.attn_backward.dw_k.iter_mut() {
                *v *= s;
            }
            for v in b.attn_backward.dw_v.iter_mut() {
                *v *= s;
            }
            for v in b.attn_backward.dw_o.iter_mut() {
                *v *= s;
            }
            for v in b.ffn_backward.dw_gate.iter_mut() {
                *v *= s;
            }
            for v in b.ffn_backward.dw_up.iter_mut() {
                *v *= s;
            }
            for v in b.ffn_backward.dw_down.iter_mut() {
                *v *= s;
            }
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        for (a, b) in self.d_embedding.iter_mut().zip(&other.d_embedding) {
            *a += b;
        }
        for (a, b) in self.d_final_gamma.iter_mut().zip(&other.d_final_gamma) {
            *a += b;
        }
        for (a, b) in self.d_lm_head.iter_mut().zip(&other.d_lm_head) {
            *a += b;
        }
        for (bg, bo) in self.d_blocks.iter_mut().zip(&other.d_blocks) {
            for (a, b) in bg.d_gamma_1.iter_mut().zip(&bo.d_gamma_1) {
                *a += b;
            }
            for (a, b) in bg.d_gamma_2.iter_mut().zip(&bo.d_gamma_2) {
                *a += b;
            }
            let (ag, ab) = (&mut bg.attn_backward, &bo.attn_backward);
            for (a, b) in ag.dw_q.iter_mut().zip(&ab.dw_q) {
                *a += b;
            }
            for (a, b) in ag.dw_k.iter_mut().zip(&ab.dw_k) {
                *a += b;
            }
            for (a, b) in ag.dw_v.iter_mut().zip(&ab.dw_v) {
                *a += b;
            }
            for (a, b) in ag.dw_o.iter_mut().zip(&ab.dw_o) {
                *a += b;
            }
            let (fg, fb) = (&mut bg.ffn_backward, &bo.ffn_backward);
            for (a, b) in fg.dw_gate.iter_mut().zip(&fb.dw_gate) {
                *a += b;
            }
            for (a, b) in fg.dw_up.iter_mut().zip(&fb.dw_up) {
                *a += b;
            }
            for (a, b) in fg.dw_down.iter_mut().zip(&fb.dw_down) {
                *a += b;
            }
        }
    }
}

impl LanguageModel {
    pub fn new(cfg: &ModelConfig) -> Self {
        let scale = (1.0 / cfg.d_model as f32).sqrt();
        Self {
            embedding: random_f32(cfg.vocab_size * cfg.d_model, 10, scale),
            blocks: (0..cfg.n_layers)
                .map(|_| TransformerBlock::new(cfg))
                .collect(),
            final_gamma: vec![1.0; cfg.d_model],
            lm_head: random_f32(cfg.d_model * cfg.vocab_size, 12, scale),
        }
    }

    pub fn encode_forward(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        cfg: &ModelConfig,
        token_ids: &[u32],
        cache: &mut LanguageModelForwardCache,
        lm_head_gpu: &mut LmHeadGpuCache,
    ) {
        assert!(cfg.d_model > 0);
        assert!(cfg.d_ff > 0);
        assert!(cfg.vocab_size > 0);
        assert!(cfg.n_layers > 0);
        assert!(cfg.n_heads > 0);

        assert_eq!(
            cfg.d_model % cfg.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        assert_eq!(
            self.blocks.len(),
            cfg.n_layers,
            "model layer count mismatch"
        );

        assert_eq!(
            cache.blocks.len(),
            cfg.n_layers,
            "cache layer count mismatch"
        );

        assert!(!token_ids.is_empty(), "token_ids must not be empty");

        for (i, &token_id) in token_ids.iter().enumerate() {
            assert!(
                (token_id as usize) < cfg.vocab_size,
                "token_ids[{i}]={token_id} out of range"
            );
        }

        assert_eq!(self.embedding.len(), cfg.vocab_size * cfg.d_model);

        assert_eq!(self.lm_head.len(), cfg.d_model * cfg.vocab_size);
        // let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        cache.token_ids = token_ids.to_vec();

        let mut x = embedding(ctx, token_ids, &self.embedding, cfg.vocab_size, cfg.d_model);
        cache.x0 = x.clone();
        // stats("embedding", &x, cfg.d_model);

        self.blocks
            .iter()
            .zip(cache.blocks.iter_mut())
            .for_each(|(block, cache)| {
                x = block.forward(ctx, cfg, &x, &cos_table, &sin_table, cache);
                // stats("block", &x, cfg.d_model);
            });

        cache.final_norm_in = x.clone();
        cache.final_norm_out = rms_norm(ctx, &x, &self.final_gamma, cfg.eps, cfg.d_model as u32);
        // stats("final_norm", &cache.final_norm_out, cfg.d_model);

        self.encode_lm_head_forward_gpu(ctx, encoder, lm_head_gpu, &cache.final_norm_out);
    }

    pub fn forward(
        &self,
        ctx: &GpuContext,
        cfg: &ModelConfig,
        token_ids: &[u32],
        cache: &mut LanguageModelForwardCache,
        lm_head_gpu: Option<&mut LmHeadGpuCache>,
    ) -> Vec<f32> {
        assert!(cfg.d_model > 0);
        assert!(cfg.d_ff > 0);
        assert!(cfg.vocab_size > 0);
        assert!(cfg.n_layers > 0);
        assert!(cfg.n_heads > 0);

        assert_eq!(
            cfg.d_model % cfg.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        assert_eq!(
            self.blocks.len(),
            cfg.n_layers,
            "model layer count mismatch"
        );

        assert_eq!(
            cache.blocks.len(),
            cfg.n_layers,
            "cache layer count mismatch"
        );

        assert!(!token_ids.is_empty(), "token_ids must not be empty");

        for (i, &token_id) in token_ids.iter().enumerate() {
            assert!(
                (token_id as usize) < cfg.vocab_size,
                "token_ids[{i}]={token_id} out of range"
            );
        }

        assert_eq!(self.embedding.len(), cfg.vocab_size * cfg.d_model);

        assert_eq!(self.lm_head.len(), cfg.d_model * cfg.vocab_size);
        let seq = token_ids.len();
        let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);

        cache.token_ids = token_ids.to_vec();

        let mut x = embedding(ctx, token_ids, &self.embedding, cfg.vocab_size, cfg.d_model);
        cache.x0 = x.clone();
        // stats("embedding", &x, cfg.d_model);

        self.blocks
            .iter()
            .zip(cache.blocks.iter_mut())
            .for_each(|(block, cache)| {
                x = block.forward(ctx, cfg, &x, &cos_table, &sin_table, cache);
                // stats("block", &x, cfg.d_model);
            });

        cache.final_norm_in = x.clone();
        cache.final_norm_out = rms_norm(ctx, &x, &self.final_gamma, cfg.eps, cfg.d_model as u32);
        // stats("final_norm", &cache.final_norm_out, cfg.d_model);

        cache.logits = match lm_head_gpu {
            Some(gpu) => self.lm_head_forward_gpu(ctx, gpu, &cache.final_norm_out),
            None => matmul_forward(
                ctx,
                &cache.final_norm_out,
                &self.lm_head,
                seq as u32,
                cfg.d_model as u32,
                cfg.vocab_size as u32,
            ),
        };
        // stats("logits", &cache.logits, cfg.vocab_size);

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

        assert!(seq > 0);

        assert_eq!(
            d_logits.len(),
            seq * cfg.vocab_size,
            "d_logits must be seq * vocab_size"
        );

        assert_eq!(cache.x0.len(), seq * cfg.d_model, "cache.x0 shape mismatch");

        assert_eq!(
            cache.final_norm_in.len(),
            seq * cfg.d_model,
            "cache.final_norm_in shape mismatch"
        );

        assert_eq!(
            cache.final_norm_out.len(),
            seq * cfg.d_model,
            "cache.final_norm_out shape mismatch"
        );

        assert_eq!(
            cache.blocks.len(),
            cfg.n_layers,
            "cache layer count mismatch"
        );
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

    pub fn encode_lm_head_forward_gpu(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &mut LmHeadGpuCache,
        final_norm_out: &[f32],
    ) {
        assert_eq!(final_norm_out.len(), gpu.seq_len * gpu.hidden.shape[1],);

        // TODO: ここだけは前段がまだCPU Vecなのでuploadが残る
        gpu.hidden.write_f32(&ctx.queue, final_norm_out);

        // GPU buffer -> GPU buffer
        encode_matmul_into(
            ctx,
            encoder,
            &gpu.matmul_bind_group,
            gpu.seq_len as u32,
            gpu.logits.shape[1] as u32,
        );
    }

    pub fn lm_head_forward_gpu(
        &self,
        ctx: &GpuContext,
        gpu: &mut LmHeadGpuCache,
        final_norm_out: &[f32],
    ) -> Vec<f32> {
        assert_eq!(final_norm_out.len(), gpu.seq_len * gpu.hidden.shape[1],);

        gpu.hidden.write_f32(&ctx.queue, final_norm_out);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lm_head_forward_encoder"),
            });

        self.encode_lm_head_forward_gpu(ctx, &mut encoder, gpu, final_norm_out);

        ctx.queue.submit([encoder.finish()]);

        // TODO: 次の段階でここを消してcross entropyへ直接つなぐ。
        read_f32_tensor(ctx, &gpu.logits)
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

        let mut x = embedding_cpu(token_ids, &self.embedding, cfg.vocab_size, cfg.d_model);
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

pub struct LmHeadGpuCache {
    // lm_head matmul
    pub weight: GpuTensor, // [d_model, vocab_size]
    pub hidden: GpuTensor, // [seq_len, d_model]
    pub logits: GpuTensor, // [seq_len, vocab_size]
    pub matmul_bind_group: wgpu::BindGroup,
    pub matmul_dims: wgpu::Buffer,

    // cross entropy
    pub targets: GpuTensor,        // [seq_len], U32
    pub loss_per_token: GpuTensor, // [seq_len], F32
    pub d_logits: GpuTensor,       // [seq_len, vocab_size], F32
    pub ce_dims: wgpu::Buffer,
    pub ce_bind_group: wgpu::BindGroup,
    pub seq_len: usize,
}

impl LmHeadGpuCache {
    pub fn new(
        ctx: &GpuContext,
        lm_head: &[f32],
        seq_len: usize,
        d_model: usize,
        vocab_size: usize,
    ) -> Self {
        assert!(seq_len > 0);
        assert!(d_model > 0);
        assert!(vocab_size > 0);

        assert_eq!(
            lm_head.len(),
            d_model * vocab_size,
            "lm_head shape must be [d_model, vocab_size]"
        );

        let storage_rw = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        let storage_read_from_cpu = BufferUsages::STORAGE | BufferUsages::COPY_DST;

        // ------------------------------------------------------------
        // lm_head matmul: hidden [seq, d] @ weight [d, vocab]
        //                 -> logits [seq, vocab]
        // ------------------------------------------------------------

        let hidden = GpuTensor::new_f32(
            &ctx.device,
            vec![seq_len, d_model],
            storage_rw,
            Some("lm_head_hidden".to_owned()),
        );

        let weight = GpuTensor::from_f32(
            &ctx.device,
            lm_head,
            vec![d_model, vocab_size],
            storage_rw,
            Some("lm_head_weight".to_owned()),
        );

        let logits = GpuTensor::new_f32(
            &ctx.device,
            vec![seq_len, vocab_size],
            storage_rw,
            Some("lm_head_logits".to_owned()),
        );

        let m = seq_len;
        let k = d_model;
        let n = vocab_size;

        assert_eq!(hidden.shape, vec![m as usize, k as usize]);
        assert_eq!(weight.shape, vec![k as usize, n as usize]);
        assert_eq!(logits.shape, vec![m as usize, n as usize]);

        let matmul_dims_values = [
            seq_len as u32,
            d_model as u32,
            vocab_size as u32,
            0,
            0,
            0,
            0,
            0,
        ];
        let matmul_dims = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dims"),
                contents: bytemuck::cast_slice(&matmul_dims_values),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let matmul_bind_group = create_matmul_bind_group(
            ctx,
            &hidden,
            &weight,
            &logits,
            &matmul_dims,
            m as u32,
            k as u32,
            n as u32,
            Some("lm_head_matmul_bind_group"),
        );

        // ------------------------------------------------------------
        // Cross entropy:
        // logits [seq, vocab] + targets [seq]
        // -> d_logits [seq, vocab] + loss_per_token [seq]
        // ------------------------------------------------------------

        let targets = GpuTensor::new_u32(
            &ctx.device,
            vec![seq_len],
            storage_read_from_cpu,
            Some("cross_entropy_targets".to_owned()),
        );

        let loss_per_token = GpuTensor::new_f32(
            &ctx.device,
            vec![seq_len],
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            Some("cross_entropy_loss_per_token".to_owned()),
        );

        let d_logits = GpuTensor::new_f32(
            &ctx.device,
            vec![seq_len, vocab_size],
            storage_rw,
            Some("cross_entropy_d_logits".to_owned()),
        );

        let ce_dims_values = [seq_len as u32, vocab_size as u32, 0, 0];

        let ce_dims = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cross_entropy_dims"),
                contents: bytemuck::cast_slice(&ce_dims_values),
                usage: BufferUsages::UNIFORM,
            });

        let ce_bind_group = create_cross_entropy_bind_group(
            ctx,
            &logits,
            &targets,
            &loss_per_token,
            &d_logits,
            &ce_dims,
            seq_len,
            vocab_size,
            Some("cross_entropy_bind_group"),
        );

        Self {
            weight,
            hidden,
            logits,
            matmul_bind_group,
            matmul_dims,
            targets,
            d_logits,
            loss_per_token,
            ce_dims,
            ce_bind_group,
            seq_len,
        }
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
        let gpu = lm.forward(&ctx, &cfg, &token_ids, &mut cache, None);

        assert_close(&gpu, &cpu, 1e-4, 1e-5);
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
        let logits_gpu = lm.forward(&ctx, &cfg, &input_ids, &mut cache, None);

        let (loss_cpu, d_logits_cpu) =
            cross_entropy_loss_cpu(&logits_cpu, &targets, input_seq, cfg.vocab_size);
        let (loss_gpu, d_logits_gpu) =
            cross_entropy_loss(&ctx, &logits_gpu, &targets, input_seq, cfg.vocab_size);

        let cpu = lm.backward_cpu(&cfg, &d_logits_cpu, &mut cache_cpu);
        let gpu = lm.backward(&ctx, &cfg, &d_logits_gpu, &mut cache);

        assert!((loss_gpu - loss_cpu).abs() < 1e-4);
        assert_close(&gpu.dx, &cpu.dx, 1e-4, 1e-5);

        gpu.d_blocks
            .iter()
            .zip(cpu.d_blocks.iter())
            .for_each(|(g_block, c_block)| {
                assert_close(&g_block.dx, &c_block.dx, 1e-4, 1e-5);
                assert_close(&g_block.d_gamma_1, &c_block.d_gamma_1, 1e-4, 1e-5);
                assert_close(&g_block.d_gamma_2, &c_block.d_gamma_2, 1e-4, 1e-5);
                assert_close(
                    &g_block.ffn_backward.dx,
                    &c_block.ffn_backward.dx,
                    1e-4,
                    1e-5,
                );
                assert_close(
                    &g_block.ffn_backward.dw_gate,
                    &c_block.ffn_backward.dw_gate,
                    1e-4,
                    1e-5,
                );
                assert_close(
                    &g_block.ffn_backward.dw_up,
                    &c_block.ffn_backward.dw_up,
                    1e-4,
                    1e-5,
                );
                assert_close(
                    &g_block.ffn_backward.dw_down,
                    &c_block.ffn_backward.dw_down,
                    1e-4,
                    1e-5,
                );
            });
        assert_close(&gpu.d_embedding, &cpu.d_embedding, 1e-4, 1e-5);
        assert_close(&gpu.d_final_gamma, &cpu.d_final_gamma, 1e-4, 1e-5);
        assert_close(&gpu.d_lm_head, &cpu.d_lm_head, 1e-4, 1e-5);
    }
}
