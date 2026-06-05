use std::{io::Write, time::Instant};

use crate::{
    checkpoint::Checkpointable,
    dataset::{Dataset, Split},
    gpu_context::GpuContext,
    kernel::{adam_w::AdamW, cross_entropy_loss::cross_entropy_loss},
    model::language_model::{LanguageModel, LanguageModelForwardCache},
    model_config::ModelConfig,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub struct TrainConfig {
    pub max_steps: usize,
    pub eval_interval: usize,
    pub eval_batches: usize,
    pub seq_len: usize,
    pub log_interval: usize,
    pub lr: f32,
    pub wd: f32,
    pub val_split: f32,
    pub seed: u64,
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            max_steps: 5000,
            eval_interval: 500,
            eval_batches: 20,
            seq_len: 128,
            log_interval: 100,
            lr: 1e-3,
            wd: 0.01,
            val_split: 0.1,
            seed: 42,
            grad_clip: 1.0,
        }
    }
}

pub struct Trainer {
    pub opt: AdamW,
    pub tcfg: TrainConfig,
}

impl Trainer {
    pub fn new(tcfg: TrainConfig) -> Self {
        let opt = AdamW::new_with_wd(tcfg.lr, tcfg.wd);
        Self { opt, tcfg }
    }

    pub fn train_step(
        &mut self,
        ctx: &GpuContext,
        model: &mut LanguageModel,
        cfg: &ModelConfig,
        cache: &mut LanguageModelForwardCache,
        input_ids: &[u32],
    ) -> f32 {
        let seq = input_ids.len() - 1;
        let input = &input_ids[..seq];
        let target: Vec<usize> = input_ids[1..].iter().map(|&t| t as usize).collect();

        let logits = model.forward(ctx, cfg, input, cache);
        let (loss, d_logits) = cross_entropy_loss(ctx, &logits, &target, seq, cfg.vocab_size);

        if !loss.is_finite() {
            return loss;
        }

        let grads = model.backward(ctx, cfg, &d_logits, cache);

        let max_norm = self.tcfg.grad_clip;
        let mut sum_sq = 0.0f32;
        let add_sq = |s: &mut f32, g: &[f32]| *s += g.iter().map(|x| x * x).sum::<f32>();
        add_sq(&mut sum_sq, &grads.d_embedding);
        add_sq(&mut sum_sq, &grads.d_final_gamma);
        add_sq(&mut sum_sq, &grads.d_lm_head);
        for bwd in &grads.d_blocks {
            add_sq(&mut sum_sq, &bwd.d_gamma_1);
            add_sq(&mut sum_sq, &bwd.d_gamma_2);
            add_sq(&mut sum_sq, &bwd.attn_backward.dw_q);
            add_sq(&mut sum_sq, &bwd.attn_backward.dw_k);
            add_sq(&mut sum_sq, &bwd.attn_backward.dw_v);
            add_sq(&mut sum_sq, &bwd.attn_backward.dw_o);
            add_sq(&mut sum_sq, &bwd.ffn_backward.dw_gate);
            add_sq(&mut sum_sq, &bwd.ffn_backward.dw_up);
            add_sq(&mut sum_sq, &bwd.ffn_backward.dw_down);
        }
        let global_norm = sum_sq.sqrt();
        let clip = if global_norm > max_norm {
            max_norm / (global_norm + 1e-6)
        } else {
            1.0
        };

        // --- AdamW step ---
        self.opt.increment_step();

        let g = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| x * clip).collect() };

        self.opt
            .step("embedding", &mut model.embedding, &g(&grads.d_embedding));
        self.opt.step(
            "final_gamma",
            &mut model.final_gamma,
            &g(&grads.d_final_gamma),
        );
        self.opt
            .step("lm_head", &mut model.lm_head, &g(&grads.d_lm_head));

        for (i, (block, bwd)) in model
            .blocks
            .iter_mut()
            .zip(grads.d_blocks.iter())
            .enumerate()
        {
            self.opt.step(
                &format!("b{i}.gamma_1"),
                &mut block.gamma_1,
                &g(&bwd.d_gamma_1),
            );
            self.opt.step(
                &format!("b{i}.gamma_2"),
                &mut block.gamma_2,
                &g(&bwd.d_gamma_2),
            );
            let ab = &bwd.attn_backward;
            self.opt
                .step(&format!("b{i}.wq"), &mut block.attn.w_q, &g(&ab.dw_q));
            self.opt
                .step(&format!("b{i}.wk"), &mut block.attn.w_k, &g(&ab.dw_k));
            self.opt
                .step(&format!("b{i}.wv"), &mut block.attn.w_v, &g(&ab.dw_v));
            self.opt
                .step(&format!("b{i}.wo"), &mut block.attn.w_o, &g(&ab.dw_o));
            let fb = &bwd.ffn_backward;
            self.opt.step(
                &format!("b{i}.w_gate"),
                &mut block.ffn.w_gate,
                &g(&fb.dw_gate),
            );
            self.opt
                .step(&format!("b{i}.w_up"), &mut block.ffn.w_up, &g(&fb.dw_up));
            self.opt.step(
                &format!("b{i}.w_down"),
                &mut block.ffn.w_down,
                &g(&fb.dw_down),
            );
        }

        loss
    }

    pub fn compute_val_loss(
        &self,
        ctx: &GpuContext,
        model: &mut LanguageModel,
        cfg: &ModelConfig,
        cache: &mut LanguageModelForwardCache,
        val_ids: &[u32],
        seed: u64,
    ) -> f32 {
        let seq = self.tcfg.seq_len;
        if val_ids.len() <= seq + 1 {
            return f32::NAN;
        }
        let max_off = val_ids.len() - seq - 1;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total = 0.0f32;

        for _ in 0..self.tcfg.eval_batches {
            let offset = rng.random_range(0..=max_off);
            let window = &val_ids[offset..offset + seq + 1];
            let input = &window[..seq];
            let target: Vec<usize> = window[1..].iter().map(|&t| t as usize).collect();
            let logits = model.forward(ctx, cfg, input, cache);
            let (loss, _) = cross_entropy_loss(ctx, &logits, &target, seq, cfg.vocab_size);
            total += loss;
        }
        total / self.tcfg.eval_batches as f32
    }

    pub fn run(
        &mut self,
        ctx: &GpuContext,
        model: &mut LanguageModel,
        cfg: &ModelConfig,
        // token_ids: &[u32],
        dataset: &Dataset,
    ) {
        let mut rng = StdRng::seed_from_u64(self.tcfg.seed);
        let mut cache = LanguageModelForwardCache::new(cfg.n_layers);
        println!(
            "# d_model={}, n_heads={}, n_kv_heads={}, d_ff={}, n_layers={}, max_seq_len={}, vocab_size={}, dropout={}",
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.d_ff,
            cfg.n_layers,
            cfg.max_seq_len,
            cfg.vocab_size,
            cfg.dropout_p,
        );
        println!(
            "# train_tokens={} val_tokens={}",
            dataset.train.len(),
            dataset.val.len()
        );

        let mut best_val = f32::INFINITY;
        println!("step,loss,ms_per_step,elapsed_s");
        let train_start = Instant::now();
        let mut window_start = Instant::now();

        for step in 1..=self.tcfg.max_steps {
            let window = dataset.sample_window(Split::Train, self.tcfg.seq_len, &mut rng);
            let loss = self.train_step(ctx, model, cfg, &mut cache, &window);

            if step % self.tcfg.log_interval == 0 {
                let window_elapsed = window_start.elapsed();
                let ms_per_step =
                    window_elapsed.as_secs_f64() * 1000.0 / self.tcfg.log_interval as f64;
                let elapsed_s = train_start.elapsed().as_secs_f64();
                println!("{step},{loss:.4},{ms_per_step:.1},{elapsed_s:.1}");
                let _ = std::io::stdout().flush();
                window_start = Instant::now();
            }
            if step % self.tcfg.eval_interval == 0 {
                let vl = self.compute_val_loss(
                    ctx,
                    model,
                    cfg,
                    &mut cache,
                    &dataset.val,
                    self.tcfg.seed + step as u64,
                );
                println!("# step {step} val_loss={vl:.4} ppl={:.2}", vl.exp());
                if vl < best_val {
                    best_val = vl;
                    let mut map = model.to_weight_map();
                    map.insert_scalar("meta.step", step as u64);
                    map.merge("optimizer", self.opt.to_weight_map());
                    map.save("checkpoints/best.ckpt").unwrap();
                    println!("#  → checkpoint saved (val_loss={vl:.4})");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_context::GpuContext;

    #[test]
    fn test_train_step_loss_decreases() {
        let ctx = GpuContext::new();
        let cfg = ModelConfig {
            vocab_size: 10,
            d_model: 16,
            n_heads: 2,
            n_kv_heads: 2,
            d_ff: 32,
            n_layers: 1,
            max_seq_len: 8,
            ..Default::default()
        };
        let mut model = LanguageModel::new(&cfg);
        let mut cache = LanguageModelForwardCache::new(cfg.n_layers);
        let mut trainer = Trainer::new(TrainConfig {
            max_steps: 20,
            seq_len: 7,
            lr: 1e-2,
            log_interval: 5,
            eval_interval: 999,
            ..Default::default()
        });

        let input: Vec<u32> = (0u32..8).map(|i| i % 10).collect();

        let loss_before = trainer.train_step(&ctx, &mut model, &cfg, &mut cache, &input);
        for _ in 0..19 {
            trainer.train_step(&ctx, &mut model, &cfg, &mut cache, &input);
        }
        let loss_after = trainer.train_step(&ctx, &mut model, &cfg, &mut cache, &input);

        assert!(
            loss_after < loss_before,
            "loss should decrease: before={loss_before:.4} after={loss_after:.4}"
        );
    }
}
