use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::model::language_model::LanguageModel;
use transformer_gpu::model_config::ModelConfig;

fn main() {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        ..Default::default()
    };

    let token_ids = vec![1, 2, 3, 4];

    let lm = LanguageModel::new(&cfg);
    let out = lm.forward(&ctx, &cfg, &token_ids);

    println!("out= {:?}", out);
}
