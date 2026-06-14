use std::{io::Write, time::Instant};

use crate::{
    checkpoint::{Checkpointable, WeightMap},
    dataset::{Dataset, Split},
    gpu_context::GpuContext,
    kernel::{adam_w::AdamW, cross_entropy_loss::cross_entropy_loss},
    lr_scheduler::{LrScheduleKind, LrScheduler},
    model::language_model::{LanguageModel, LanguageModelBackward, LanguageModelForwardCache},
    model_config::ModelConfig,
    tokenizer::TokenizerKind,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub struct TrainConfig {
    pub batch_size: usize,
    pub eval_interval: usize,
    pub eval_batches: usize,
    pub seq_len: usize,
    pub log_interval: usize,
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: usize,
    pub end_step: usize,
    pub lr_schedule_kind: LrScheduleKind,
    pub wd: f32,
    pub val_split: f32,
    pub seed: u64,
    pub grad_clip: f32,
    pub tokenizer_kind: TokenizerKind,
    pub corpus: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            eval_interval: 500,
            eval_batches: 20,
            seq_len: 128,
            log_interval: 100,
            lr_max: 1e-3,
            lr_min: 1e-4,
            warmup_steps: 200,
            end_step: 5000,
            wd: 0.01,
            val_split: 0.2,
            seed: 42,
            grad_clip: 1.0,
            lr_schedule_kind: LrScheduleKind::WarmupStableDecay { stable_steps: 3800 },
            tokenizer_kind: TokenizerKind::Bpe,
            corpus: String::new(),
        }
    }
}

impl TrainConfig {
    pub fn tiny_shakespeare() -> Self {
        Self {
            batch_size: 16,
            eval_interval: 500,
            eval_batches: 20,
            seq_len: 128,
            log_interval: 20,
            lr_max: 5e-4,
            lr_min: 5e-5,
            warmup_steps: 500,
            end_step: 10000,
            wd: 0.01,
            val_split: 0.2,
            seed: 42,
            grad_clip: 1.0,
            lr_schedule_kind: LrScheduleKind::WarmupStableDecay { stable_steps: 8000 },
            tokenizer_kind: TokenizerKind::Bpe,
            corpus: "corpus/tiny_shakespeare.txt".to_string(),
        }
    }
}

pub struct Trainer {
    pub opt: AdamW,
    pub tcfg: TrainConfig,
}

impl Trainer {
    pub fn new(tcfg: TrainConfig) -> Self {
        let opt = AdamW::new_with_wd(tcfg.lr_min, tcfg.wd);
        Self { opt, tcfg }
    }

    pub fn compute_grads(
        &self,
        ctx: &GpuContext,
        model: &LanguageModel,
        cfg: &ModelConfig,
        cache: &mut LanguageModelForwardCache,
        input_ids: &[u32],
    ) -> Option<(f32, LanguageModelBackward)> {
        let seq = input_ids.len() - 1;
        let input = &input_ids[..seq];
        let target: Vec<usize> = input_ids[1..].iter().map(|&t| t as usize).collect();

        let logits = model.forward(ctx, cfg, input, cache);
        let (loss, d_logits) = cross_entropy_loss(ctx, &logits, &target, seq, cfg.vocab_size);

        if !loss.is_finite() {
            return None;
        }

        let grads = model.backward(ctx, cfg, &d_logits, cache);
        Some((loss, grads))
    }

    pub fn apply_grads(
        &mut self,
        model: &mut LanguageModel,
        grads: &LanguageModelBackward,
        lr: f32,
    ) {
        self.opt.set_lr(lr);
        self.opt.increment_step();

        self.opt
            .step("embedding", &mut model.embedding, &grads.d_embedding);
        self.opt
            .step("final_gamma", &mut model.final_gamma, &grads.d_final_gamma);
        self.opt
            .step("lm_head", &mut model.lm_head, &grads.d_lm_head);

        for (i, (block, bwd)) in model
            .blocks
            .iter_mut()
            .zip(grads.d_blocks.iter())
            .enumerate()
        {
            self.opt
                .step(&format!("b{i}.gamma_1"), &mut block.gamma_1, &bwd.d_gamma_1);
            self.opt
                .step(&format!("b{i}.gamma_2"), &mut block.gamma_2, &bwd.d_gamma_2);
            let ab = &bwd.attn_backward;
            self.opt
                .step(&format!("b{i}.wq"), &mut block.attn.w_q, &ab.dw_q);
            self.opt
                .step(&format!("b{i}.wk"), &mut block.attn.w_k, &ab.dw_k);
            self.opt
                .step(&format!("b{i}.wv"), &mut block.attn.w_v, &ab.dw_v);
            self.opt
                .step(&format!("b{i}.wo"), &mut block.attn.w_o, &ab.dw_o);
            let fb = &bwd.ffn_backward;
            self.opt
                .step(&format!("b{i}.w_gate"), &mut block.ffn.w_gate, &fb.dw_gate);
            self.opt
                .step(&format!("b{i}.w_up"), &mut block.ffn.w_up, &fb.dw_up);
            self.opt
                .step(&format!("b{i}.w_down"), &mut block.ffn.w_down, &fb.dw_down);
        }
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
        cfg: &mut ModelConfig,
        dataset: &Dataset,
        resume_ckpt: Option<&str>,
    ) {
        let mut rng = StdRng::seed_from_u64(self.tcfg.seed);
        let mut ema_loss: Option<f32> = None;
        let mut start_step = 1usize;
        let mut best_val = f32::INFINITY;
        if let Some(path) = resume_ckpt {
            let map = WeightMap::load(path).unwrap();
            cfg.from_weight_map(&map.scoped("meta.model")).unwrap();
            model.from_weight_map(&map).unwrap();
            self.opt.from_weight_map(&map.scoped("optimizer")).unwrap();
            start_step = map.get_scalar("meta.step").unwrap_or(1) as usize;
            best_val = map
                .get_scalar("meta.best_val")
                .map(|v| f32::from_bits(v as u32))
                .unwrap_or(f32::INFINITY);

            // Consume RNG to synchronize
            let max_offset_train = dataset.train.len() - self.tcfg.seq_len - 1;
            let max_offset_val = dataset.val.len() - self.tcfg.seq_len - 1;
            for step in 1..start_step {
                let _ = rng.random_range(0..=max_offset_train);
                if step % self.tcfg.eval_interval == 0 {
                    let _ = rng.random_range(0..=max_offset_val);
                }
            }
            println!("# resume from checkpoint: {}", path);
        }

        let lr_scheduler = LrScheduler::with_kind(
            self.tcfg.lr_max,
            self.tcfg.lr_min,
            self.tcfg.warmup_steps,
            self.tcfg.end_step,
            self.tcfg.lr_schedule_kind,
        );

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
            "# lr_schedule={:?}, lr_max={}, lr_min={}, warmup_steps={}, end_step={}, batch_size={}",
            self.tcfg.lr_schedule_kind,
            self.tcfg.lr_max,
            self.tcfg.lr_min,
            self.tcfg.warmup_steps,
            self.tcfg.end_step,
            self.tcfg.batch_size
        );
        println!(
            "# tokenizer={:?} train_tokens={}, val_tokens={}, val_chars={}, val_chars_per_token={:.4}",
            self.tcfg.tokenizer_kind,
            dataset.train.len(),
            dataset.val.len(),
            dataset.val_text_len_chars,
            if dataset.val.is_empty() {
                0.0
            } else {
                dataset.val_text_len_chars as f32 / dataset.val.len() as f32
            }
        );
        println!("# start_step={}, best_val={}", start_step, best_val);

        println!("step,loss,ema,ppl,ema_ppl,lr,ms_per_step,elapsed_s");
        let train_start = Instant::now();
        let mut window_start = Instant::now();
        for step in start_step..=self.tcfg.end_step {
            let lr = lr_scheduler.get_lr(step);

            let mut total_loss = 0.0f32;
            let mut accum_grads: Option<LanguageModelBackward> = None;

            let mut valid_count = 0usize;
            for _ in 0..self.tcfg.batch_size {
                let window = dataset.sample_window(Split::Train, self.tcfg.seq_len, &mut rng);
                let Some((loss, grads)) = self.compute_grads(ctx, model, cfg, &mut cache, &window)
                else {
                    accum_grads = None;
                    break;
                };

                total_loss += loss;
                valid_count += 1;
                match accum_grads.as_mut() {
                    Some(ag) => ag.add_assign(&grads),
                    None => accum_grads = Some(grads),
                }
            }

            if let Some(grads) = accum_grads {
                // grad_scale = 1/valid_count、apply_grads 内の AdamW に反映
                self.opt.set_grad_scale(valid_count);
                self.apply_grads(model, &grads, lr);
            }
            self.opt.reset_grad_scale();

            let avg_loss = total_loss / self.tcfg.batch_size as f32;
            // ema_loss = alpha * ema_loss + (1 - alpha) * loss
            let alpha = 0.05;
            ema_loss = Some(match ema_loss {
                Some(e) => e * (1.0 - alpha) + avg_loss * alpha,
                None => avg_loss,
            });

            if step % self.tcfg.log_interval == 0 {
                let window_elapsed = window_start.elapsed();
                let ms_per_step =
                    window_elapsed.as_secs_f64() * 1000.0 / self.tcfg.log_interval as f64;
                let elapsed_s = train_start.elapsed().as_secs_f64();
                let ema = ema_loss.unwrap();
                println!(
                    "{step},{avg_loss:.4},{ema:.4},{:.4},{:.4},{lr:.3e},{ms_per_step:.1},{elapsed_s:.1}",
                    avg_loss.exp(),
                    ema.exp()
                );
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
                let bpc = if dataset.val_text_len_chars > 0 && !dataset.val.is_empty() {
                    vl / std::f32::consts::LN_2
                        * (dataset.val.len() as f32 / dataset.val_text_len_chars as f32)
                } else {
                    f32::NAN
                };
                println!(
                    "# val step={step} val_loss={vl:.4} val_ppl={:.4} bpc={:.4}",
                    vl.exp(),
                    bpc
                );
                if vl < best_val {
                    best_val = vl;
                    let mut map = model.to_weight_map();
                    map.insert_scalar("meta.step", step as u64);
                    map.insert_scalar("meta.best_val", best_val.to_bits() as u64);
                    map.merge("meta.model", cfg.to_weight_map());
                    map.merge("optimizer", self.opt.to_weight_map());
                    map.save("checkpoints/best.ckpt").unwrap();
                    println!("# → checkpoint saved (val_loss={vl:.4})");
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
            warmup_steps: 5,
            end_step: 20,
            seq_len: 7,
            lr_min: 1e-2,
            lr_max: 5e-2,
            lr_schedule_kind: LrScheduleKind::WarmupStableDecay { stable_steps: 10 },
            log_interval: 5,
            eval_interval: 999,
            ..Default::default()
        });
        let lr_scheduler = LrScheduler::with_kind(
            trainer.tcfg.lr_max,
            trainer.tcfg.lr_min,
            trainer.tcfg.warmup_steps,
            trainer.tcfg.end_step,
            trainer.tcfg.lr_schedule_kind,
        );

        let input: Vec<u32> = (0u32..8).map(|i| i % 10).collect();

        let mut loss_before = 0.0f32;
        let mut accum_grads_before: Option<LanguageModelBackward> = None;
        if let Some((loss, grads)) = trainer.compute_grads(&ctx, &model, &cfg, &mut cache, &input) {
            loss_before = loss;
            match accum_grads_before.as_mut() {
                Some(ag) => ag.add_assign(&grads),
                None => accum_grads_before = Some(grads),
            }
        }
        trainer.opt.set_grad_scale(trainer.tcfg.batch_size);
        if let Some(grads) = accum_grads_before {
            trainer.apply_grads(&mut model, &grads, lr_scheduler.get_lr(0));
        }
        trainer.opt.reset_grad_scale();

        for step in 2..20 {
            let lr = lr_scheduler.get_lr(step);

            let mut accum_grads: Option<LanguageModelBackward> = None;

            if let Some((_, grads)) = trainer.compute_grads(&ctx, &model, &cfg, &mut cache, &input)
            {
                match accum_grads.as_mut() {
                    Some(ag) => ag.add_assign(&grads),
                    None => accum_grads = Some(grads),
                }
            }

            // grad_scale = 1/batch_size、apply_grads 内の AdamW に反映
            trainer.opt.set_grad_scale(trainer.tcfg.batch_size);
            if let Some(grads) = accum_grads {
                trainer.apply_grads(&mut model, &grads, lr);
            }
            trainer.opt.reset_grad_scale();
        }

        let mut loss_after = 0.0f32;
        let mut accum_grads_after: Option<LanguageModelBackward> = None;
        if let Some((loss, grads)) = trainer.compute_grads(&ctx, &model, &cfg, &mut cache, &input) {
            loss_after = loss;
            match accum_grads_after.as_mut() {
                Some(ag) => ag.add_assign(&grads),
                None => accum_grads_after = Some(grads),
            }
        }
        trainer.opt.set_grad_scale(trainer.tcfg.batch_size);
        if let Some(grads) = accum_grads_after {
            trainer.apply_grads(&mut model, &grads, lr_scheduler.get_lr(0));
        }
        trainer.opt.reset_grad_scale();

        assert!(
            loss_after < loss_before,
            "loss should decrease: before={loss_before:.4} after={loss_after:.4}"
        );
    }
}
