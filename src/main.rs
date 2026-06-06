use transformer_gpu::dataset::Dataset;
use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::model::language_model::LanguageModel;
use transformer_gpu::model_config::ModelConfig;
use transformer_gpu::train::{TrainConfig, Trainer};

fn main() {
    let ctx = GpuContext::new();
    let mut cfg = ModelConfig {
        ..Default::default()
    };
    let mut model = LanguageModel::new(&cfg);
    let mut trainer = Trainer::new(TrainConfig {
        ..Default::default()
    });

    let dataset = Dataset::from_file("corpus/tiny_shakespeare.txt", 0.2).unwrap();

    // trainer.run(&ctx, &mut model, &mut cfg, &dataset, None);

    // resume from best.ckpt
    trainer.run(
        &ctx,
        &mut model,
        &mut cfg,
        &dataset,
        Some("checkpoints/best.ckpt"),
    );
}
