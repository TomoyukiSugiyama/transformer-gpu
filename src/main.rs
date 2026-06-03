use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::model::language_model::LanguageModel;
use transformer_gpu::model_config::ModelConfig;
use transformer_gpu::train::{TrainConfig, Trainer};

fn main() {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        ..Default::default()
    };
    let mut model = LanguageModel::new(&cfg);
    let mut trainer = Trainer::new(TrainConfig {
        ..Default::default()
    });

    let token_ids: Vec<u32> = (0u32..2000).map(|i| i % cfg.vocab_size as u32).collect();

    trainer.run(&ctx, &mut model, &cfg, &token_ids);
}
