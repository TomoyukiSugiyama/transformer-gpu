use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use transformer_gpu::{
    gpu_context::GpuContext,
    kernel::rope::create_table,
    model::transformer_block::{TransformerBlock, TransformerBlockForwardCache},
    model_config::ModelConfig,
    util::random_f32,
};

fn bench_transformer_block(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        max_seq_len: 512,
        ..Default::default()
    };
    let seq = 512usize;
    let scale = (1.0 / cfg.d_model as f32).sqrt();
    let x = random_f32(seq * cfg.d_model, 0, scale);
    let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);
    let block = TransformerBlock::new(&cfg);
    let mut cache = TransformerBlockForwardCache::default();
    c.bench_function("transformer_block seq=512", |bencher| {
        bencher.iter(|| block.forward(&ctx, &cfg, &x, &cos_table, &sin_table, &mut cache));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3));
    targets = bench_transformer_block
}
criterion_main!(benches);
