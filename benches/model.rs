use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use transformer_gpu::{
    gpu_context::GpuContext, kernel::rope::create_table,
    model::transformer_block::TransformerBlock, model_config::ModelConfig, util::random_f32,
};

fn bench_transformer_block(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        vocab_size: 4096,
        d_model: 64,
        n_heads: 4,
        n_kv_heads: 4,
        d_ff: 128,
        n_layers: 4,
        max_seq_len: 512,
        dropout_p: 0.1,
        eps: 1e-6,
        rope_base: 10000.0,
    };
    let seq = 512usize;
    let x = random_f32(seq * cfg.d_model, 0);
    let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, cfg.rope_base);
    let block = TransformerBlock::new(&cfg);

    c.bench_function("transformer_block_gpu seq=512", |bencher| {
        bencher.iter(|| block.forward(&ctx, &cfg, &x, &cos_table, &sin_table));
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_secs(3));
    targets = bench_transformer_block
}
criterion_main!(benches);
