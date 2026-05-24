use transformer_gpu::gpu_context::GpuContext;
use transformer_gpu::kernel::rope::create_table;
use transformer_gpu::model::transformer_block::TransformerBlock;
use transformer_gpu::model_config::ModelConfig;
use transformer_gpu::util::random_f32;

fn main() {
    let ctx = GpuContext::new();
    let cfg = ModelConfig {
        ..Default::default()
    };

    let x: Vec<f32> = random_f32(cfg.max_seq_len * cfg.d_model, 31);

    let base: f32 = 10000.0;
    let (cos_table, sin_table) = create_table(cfg.d_head(), cfg.max_seq_len, base);

    let tf = TransformerBlock::new(&cfg);
    let out = tf.forward(&ctx, &cfg, &x, &cos_table, &sin_table);

    println!("out= {:?}", out);
}
