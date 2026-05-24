use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::kernel::rope::create_table;
use transformer_gpu::model::transformer_block::transformer_block_gpu;

fn main() {
    let seq: usize = 1024;
    let d_model: usize = 64;
    let d_ff = 128;
    let n_heads: usize = 4;
    let x: Vec<f32> = random_f32(seq * d_model, 31);
    let w_q: Vec<f32> = random_f32(d_model * d_model, 32);
    let w_k: Vec<f32> = random_f32(d_model * d_model, 33);
    let w_v: Vec<f32> = random_f32(d_model * d_model, 34);
    let w_o: Vec<f32> = random_f32(d_model * d_model, 35);
    let w_gate: Vec<f32> = random_f32(d_model * d_ff, 36);
    let w_up: Vec<f32> = random_f32(d_model * d_ff, 37);
    let w_down: Vec<f32> = random_f32(d_ff * d_model, 38);
    let d_head = d_model / n_heads;
    let max_len = 1024;
    let base: f32 = 10000.0;
    let (cos_table, sin_table) = create_table(d_head, max_len, base);

    let ctx = GpuContext::new();

    let out = transformer_block_gpu(
        &ctx,
        seq as u32,
        d_model as u32,
        n_heads as u32,
        d_ff as u32,
        &x,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_gate,
        &w_up,
        &w_down,
        &cos_table,
        &sin_table,
    );

    println!("out= {:?}", out);
}

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

pub fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
}
