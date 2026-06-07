use rand::RngExt;
use rand::rng;

use crate::char_bpe_tokenizer::CharBpeTokenizer;
use crate::checkpoint::Checkpointable;
use crate::checkpoint::WeightMap;
use crate::gpu_context::GpuContext;
use crate::model::language_model::LanguageModel;
use crate::model::language_model::LanguageModelForwardCache;
use crate::model_config::ModelConfig;

pub fn infer(
    ctx: &GpuContext,
    model: &mut LanguageModel,
    cfg: &mut ModelConfig,
    tokenizer: CharBpeTokenizer,
    prompts: &[&str],
    restore_ckpt: Option<&str>,
) {
    let max_new_token = 100;
    let top_k = 5;
    let top_p = 0.9;
    let temperature = 1.0;
    // let repetition_penalty = 1.2;

    if let Some(path) = restore_ckpt {
        let map = WeightMap::load(path).unwrap();
        cfg.from_weight_map(&map.scoped("meta.model")).unwrap();
        model.from_weight_map(&map).unwrap();

        println!("# restore from checkpoint: {}", path);
    }

    for prompt in prompts {
        println!("\n=== prompt: {:?} ===", prompt);

        // top-k: KV cache 版を使用 (no-cache 版より高速、 出力サンプリング分布は同じ)
        let t_topk = std::time::Instant::now();
        let topk_text = generate_top_k(
            ctx,
            model,
            cfg,
            &tokenizer,
            prompt,
            max_new_token,
            top_k,
            temperature,
        );
        let dt_topk = t_topk.elapsed();
        println!(
            "\n--- top-k (k={top_k}, t={temperature}) [{:.2}s] ---",
            dt_topk.as_secs_f32()
        );
        println!("\n{}", topk_text);

        let t_topp = std::time::Instant::now();
        let topp_text = generate_top_p(
            ctx,
            model,
            cfg,
            &tokenizer,
            prompt,
            max_new_token,
            top_p,
            temperature,
        );
        let dt_topp = t_topp.elapsed();
        println!(
            "\n--- top-p (p={top_p}, t={temperature}) [{:.2}s] ---",
            dt_topp.as_secs_f32()
        );
        println!("\n{}", topp_text);
    }
}

pub fn generate_top_k(
    ctx: &GpuContext,
    model: &mut LanguageModel,
    cfg: &mut ModelConfig,
    tokenizer: &CharBpeTokenizer,
    prompt: &str,
    max_new_token: usize,
    top_k: usize,
    temperature: f32,
) -> String {
    let mut token_ids_u32: Vec<u32> = tokenizer
        .encode_prompt(prompt)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let eos_id = tokenizer.eos_id();
    let mut cache = LanguageModelForwardCache::new(cfg.n_layers);

    for _ in 0..max_new_token {
        let seq = token_ids_u32.len();
        let logits = model.forward(ctx, cfg, &token_ids_u32, &mut cache);

        let next_logits = &logits[(seq - 1) * cfg.vocab_size..seq * cfg.vocab_size];
        let next_id = top_k_sample(next_logits, top_k, temperature);

        if next_id == eos_id {
            break;
        }
        token_ids_u32.push(next_id as u32);
    }

    let token_ids: Vec<usize> = token_ids_u32.iter().map(|&t| t as usize).collect();
    tokenizer.decode(&token_ids)
}

pub fn generate_top_p(
    ctx: &GpuContext,
    model: &mut LanguageModel,
    cfg: &mut ModelConfig,
    tokenizer: &CharBpeTokenizer,
    prompt: &str,
    max_new_token: usize,
    top_p: f32,
    temperature: f32,
) -> String {
    let mut token_ids_u32: Vec<u32> = tokenizer
        .encode_prompt(prompt)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let eos_id = tokenizer.eos_id();
    let mut cache = LanguageModelForwardCache::new(cfg.n_layers);

    for _ in 0..max_new_token {
        let seq = token_ids_u32.len();
        let logits = model.forward(ctx, cfg, &token_ids_u32, &mut cache);

        let next_logits = &logits[(seq - 1) * cfg.vocab_size..seq * cfg.vocab_size];
        let next_id = top_p_sample(next_logits, top_p, temperature);

        if next_id == eos_id {
            break;
        }
        token_ids_u32.push(next_id as u32);
    }

    let token_ids: Vec<usize> = token_ids_u32.iter().map(|&t| t as usize).collect();
    tokenizer.decode(&token_ids)
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

pub fn top_k_sample(logits: &[f32], k: usize, temperature: f32) -> usize {
    assert!(
        temperature > 0.0,
        "temperature must be > 0, got {temperature}"
    );
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);

    let top_logits: Vec<f32> = indexed.iter().map(|(_, l)| l / temperature).collect();
    let probes = softmax(&top_logits);
    let mut rng = rng();
    let r: f32 = rng.random_range(0.0..1.0);
    let mut custom = 0.0;

    for (idx, &p) in probes.iter().enumerate() {
        custom += p;
        if r < custom {
            return indexed[idx].0;
        }
    }
    indexed[0].0
}

pub fn top_p_candidates(logits: &[f32], p: f32, temperature: f32) -> Vec<(usize, f32)> {
    assert!(
        temperature > 0.0,
        "temperature must be > 0, got {temperature}"
    );
    assert!(
        (0.0..=1.0).contains(&p),
        "top-p must be in [0.0, 1.0], got {p}"
    );

    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();
    let probs = softmax(&scaled);

    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 累積確率が初めて p を超える index で打ち切り (inclusive)。
    // p=1.0 のときは浮動小数誤差で 1.0 に達しない可能性があるので、 デフォルトで全件を残す。
    let mut cutoff = indexed.len();
    let mut cum = 0.0;
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cum += prob;
        if cum >= p {
            cutoff = i + 1;
            break;
        }
    }

    let mut kept: Vec<(usize, f32)> = indexed.into_iter().take(cutoff).collect();
    let sum: f32 = kept.iter().map(|&(_, q)| q).sum();
    for entry in kept.iter_mut() {
        entry.1 /= sum;
    }
    kept
}

pub fn top_p_sample(logits: &[f32], p: f32, temperature: f32) -> usize {
    let kept = top_p_candidates(logits, p, temperature);
    let mut rng = rng();
    let r: f32 = rng.random_range(0.0..1.0);
    let mut acc = 0.0;
    for &(idx, prob) in &kept {
        acc += prob;
        if r < acc {
            return idx;
        }
    }
    kept[0].0
}

#[cfg(test)]
mod tests {
    use crate::infer::{top_p_candidates, top_p_sample};

    /// p=1.0 は全候補を残す (確率の合計が 1.0 に正規化される)。
    #[test]
    fn top_p_one_keeps_all_candidates() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 0.5];
        let kept = top_p_candidates(&logits, 1.0, 1.0);
        assert_eq!(kept.len(), logits.len());
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0, got {sum}");

        // 元の vocab id が全て揃っているかチェック
        let mut ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    }

    /// p が極小だと top-1 のみが残り、 確率は 1.0 になる。
    #[test]
    fn top_p_zero_keeps_top_one() {
        let logits = vec![1.0, 5.0, 2.0, 0.5]; // top-1 は index=1
        let kept = top_p_candidates(&logits, 0.0, 1.0);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0, 1);
        assert!((kept[0].1 - 1.0).abs() < 1e-6);
    }

    /// 累積確率が p を「初めて超えた」 index で打ち切る (inclusive)。
    /// 確率が [0.5, 0.3, 0.15, 0.05] となるような logits を構成し、 p=0.7 で 2 候補だけ残るか確認。
    #[test]
    fn top_p_cutoff_is_inclusive_at_first_overshoot() {
        // softmax(logits) ≈ [0.5, 0.3, 0.15, 0.05] となる logits を逆算:
        // log(p_i / p_0) を logits 差として与える
        let probs = [0.5_f32, 0.3, 0.15, 0.05];
        let logits: Vec<f32> = probs.iter().map(|p| p.ln()).collect();

        // p=0.7: 累積 0.5 (1 件) → 0.8 (2 件超過) で打ち切り、 残るのは 2 件
        let kept = top_p_candidates(&logits, 0.7, 1.0);
        assert_eq!(kept.len(), 2);
        // 元 idx 0 と 1 のはず (top-2)
        let mut kept_ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        kept_ids.sort();
        assert_eq!(kept_ids, vec![0, 1]);
        // 再正規化後の合計
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // 比率は元の 0.5:0.3 を保つ (再正規化されただけ)
        // 上位が 0.5/0.8 = 0.625, 次が 0.3/0.8 = 0.375
        let top = kept.iter().find(|&&(i, _)| i == 0).unwrap().1;
        let next = kept.iter().find(|&&(i, _)| i == 1).unwrap().1;
        assert!((top - 0.625).abs() < 1e-5);
        assert!((next - 0.375).abs() < 1e-5);
    }

    /// temperature が確率分布の鋭さを正しく変える:
    /// 同じ logits でも t<1 で分布が尖り、 t>1 で平坦化する。
    /// p を固定したとき、 t<1 では候補が減り、 t>1 では増える方向のはず。
    #[test]
    fn top_p_temperature_affects_distribution_sharpness() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sharp = top_p_candidates(&logits, 0.9, 0.5);
        let flat = top_p_candidates(&logits, 0.9, 2.0);
        // 鋭いほど少数候補で 90% に到達、 平坦ほど多数必要
        assert!(
            sharp.len() <= flat.len(),
            "sharp(t=0.5)={} flat(t=2.0)={}",
            sharp.len(),
            flat.len()
        );
        // 同じ logits で順位は保たれる: top-1 は idx=4 (logit=5.0)
        assert_eq!(sharp[0].0, 4);
        assert_eq!(flat[0].0, 4);
    }

    /// 浮動小数誤差で p=1.0 でも cumsum が 0.999... になっても全候補が残る。
    #[test]
    fn top_p_one_robust_to_float_error() {
        let logits = vec![0.0_f32; 100]; // 一様分布
        let kept = top_p_candidates(&logits, 1.0, 1.0);
        assert_eq!(kept.len(), 100);
        let sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    /// 同じ vocab id は重複しない。
    #[test]
    fn top_p_returns_unique_ids() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kept = top_p_candidates(&logits, 0.8, 1.0);
        let mut ids: Vec<usize> = kept.iter().map(|&(i, _)| i).collect();
        let before = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(before, ids.len(), "ids must be unique");
    }

    /// top_p_sample は kept 集合から id を返す。
    #[test]
    fn top_p_sample_returns_id_from_kept_set() {
        let logits = vec![1.0, 10.0, 2.0]; // top-1 は idx=1 (圧倒的)
        // p=0.5 ならほぼ確実に top-1 のみが kept になる
        let id = top_p_sample(&logits, 0.5, 1.0);
        assert_eq!(id, 1);
    }
}
