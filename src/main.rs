use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::model::language_model::{LanguageModel, LanguageModelForwardCache};
use transformer_gpu::model_config::ModelConfig;

fn main() {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        ..Default::default()
    };
    let mut cache = LanguageModelForwardCache::new(cfg.n_layers);

    let token_ids = vec![1, 2, 3, 4];

    let lm = LanguageModel::new(&cfg);
    let out = lm.forward(&ctx, &cfg, &token_ids, &mut cache);

    println!("out= {:?}", out);
}
