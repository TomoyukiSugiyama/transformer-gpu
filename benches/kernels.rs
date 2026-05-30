use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use transformer_gpu::{
    gpu_context::GpuContext,
    kernel::{
        attention::{attention, before_flash_attention},
        matmul::matmul_forward,
    },
    util::random_f32,
};

fn bench_matmul(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let seq = 512u32;
    let d_model = 64u32;
    let a = random_f32((seq * d_model) as usize, 0);
    let b = random_f32((d_model * d_model) as usize, 1);

    c.bench_function("matmul 512x64x64", |bencher| {
        bencher.iter(|| matmul_forward(&ctx, &a, &b, seq, d_model, d_model));
    });
}

fn bench_attention(c: &mut Criterion) {
    let ctx = GpuContext::new();
    let seq = 512u32;
    let d_head = 16u32;
    let q = random_f32((seq * d_head) as usize, 0);
    let k = random_f32((seq * d_head) as usize, 1);
    let v = random_f32((seq * d_head) as usize, 2);

    let mut group = c.benchmark_group("attention seq=512 d_head=64");

    group.bench_function("flash_attention", |bencher| {
        bencher.iter(|| attention(&ctx, &q, &k, &v, seq, d_head));
    });

    group.bench_function("before_flash_attention", |bencher| {
        bencher.iter(|| before_flash_attention(&ctx, &q, &k, &v, seq, d_head));
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
    .warm_up_time(Duration::from_secs(3));
    targets = bench_matmul, bench_attention
}
criterion_main!(benches);
