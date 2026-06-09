use transformer_gpu::dataset::Dataset;
use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::infer::infer;
use transformer_gpu::model::language_model::LanguageModel;
use transformer_gpu::model_config::ModelConfig;
use transformer_gpu::tokenizer::train_tokenizer;
use transformer_gpu::train::{TrainConfig, Trainer};

fn main() {
    let mut trainer = Trainer::new(TrainConfig::tiny_shakespeare());
    let corpus_text = Dataset::from_file(&trainer.tcfg.corpus).unwrap();

    let ctx = GpuContext::new();
    let mut cfg = ModelConfig {
        ..ModelConfig::tiny_shakespeare()
    };

    let tokenizer = train_tokenizer(trainer.tcfg.tokenizer_kind, &corpus_text, cfg.vocab_size);
    let token_ids: Vec<u32> = tokenizer
        .encode_long(&corpus_text)
        .into_iter()
        .map(|x| x as u32)
        .collect();

    let mut model = LanguageModel::new(&cfg);

    let dataset = Dataset::from_tokens(token_ids, trainer.tcfg.val_split, corpus_text.len());
    trainer.run(&ctx, &mut model, &mut cfg, &dataset, None);

    // 学習後に以下のプロンプトで推論
    let prompts = vec!["I have seen", "O Romeo", "To be or not to be", "What news"];
    infer(
        &ctx,
        &mut model,
        &mut cfg,
        &tokenizer,
        &prompts,
        Some("checkpoints/best.ckpt"),
    );

    // best.ckpt から再開
    // trainer.run(
    //     &ctx,
    //     &mut model,
    //     &mut cfg,
    //     &dataset,
    //     Some("checkpoints/best.ckpt"),
    // );
}
