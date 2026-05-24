use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::kernel::rope::create_table;
use transformer_gpu::model::transformer_block::{TransformerBlock, TransformerBlockConfig};
use transformer_gpu::util::random_f32;

fn main() {
    let ctx = GpuContext::new();
    let cfg = TransformerBlockConfig {
        seq: 1024,
        d_model: 64,
        n_heads: 4,
        d_ff: 128,
        eps: 1e-6,
    };

    let x: Vec<f32> = random_f32(cfg.seq * cfg.d_model, 31);
    let d_head = cfg.d_model / cfg.n_heads;
    let max_len = 1024;
    let base: f32 = 10000.0;
    let (cos_table, sin_table) = create_table(d_head, max_len, base);

    let tf = TransformerBlock::new(&cfg);
    let out = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

    println!("out= {:?}", out);
}
