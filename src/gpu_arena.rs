use crate::gpu_tensor::GpuTensor;

pub struct GpuArena {
    pub x: Vec<GpuTensor>,
    pub norm_1: Vec<GpuTensor>,
    pub attn_out: Vec<GpuTensor>,
    pub norm_2: Vec<GpuTensor>,
    pub gate: Vec<GpuTensor>,
    pub up: Vec<GpuTensor>,
    pub ffn_out: Vec<GpuTensor>,

    pub logits: GpuTensor,
    pub d_logits: GpuTensor,
    pub loss: GpuTensor,
}

impl GpuArena {
    pub fn new(
        device: &wgpu::Device,
        n_layers: usize,
        seq_len: usize,
        d_model: usize,
        d_ff: usize,
        vocab_size: usize,
    ) -> Self {
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let model_shape = vec![seq_len, d_model];
        let ffn_shape = vec![seq_len, d_ff];
        let logits_shape = vec![seq_len, vocab_size];

        let model_tensor = |name: &str| {
            GpuTensor::new_f32(device, model_shape.clone(), usage, Some(name.to_owned()))
        };

        let ffn_tensor = |name: &str| {
            GpuTensor::new_f32(device, ffn_shape.clone(), usage, Some(name.to_owned()))
        };

        Self {
            x: (0..=n_layers)
                .map(|i| model_tensor(&format!("x_{i}")))
                .collect(),

            norm_1: (0..n_layers)
                .map(|i| model_tensor(&format!("norm_1_{i}")))
                .collect(),

            attn_out: (0..n_layers)
                .map(|i| model_tensor(&format!("attn_out_{i}")))
                .collect(),

            norm_2: (0..n_layers)
                .map(|i| model_tensor(&format!("norm_2_{i}")))
                .collect(),

            gate: (0..n_layers)
                .map(|i| ffn_tensor(&format!("gate_{i}")))
                .collect(),

            up: (0..n_layers)
                .map(|i| ffn_tensor(&format!("up_{i}")))
                .collect(),

            ffn_out: (0..n_layers)
                .map(|i| model_tensor(&format!("ffn_out_{i}")))
                .collect(),

            logits: GpuTensor::new_f32(device, logits_shape, usage, Some("logits".to_owned())),

            d_logits: GpuTensor::new_f32(
                device,
                vec![seq_len, vocab_size],
                usage,
                Some("d_logits".to_owned()),
            ),

            loss: GpuTensor::new_f32(device, vec![1], usage, Some("loss".to_owned())),
        }
    }
}
