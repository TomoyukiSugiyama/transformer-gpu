pub struct ModelConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub eps: f32,
    pub rope_base: f32,
}

impl ModelConfig {
    /// Llama-3 8B 相当の設定例
    pub fn llama3_small() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            d_ff: 1024,
            n_layers: 6,
            max_seq_len: 2048,
            eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
}
