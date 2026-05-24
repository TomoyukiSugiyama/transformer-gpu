pub struct ModelConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub max_seq_len: usize,
    pub dropout_p: f32,
    pub eps: f32,
    pub rope_base: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256, // テスト用デフォルト
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 4,
            d_ff: 128,
            n_layers: 2,
            max_seq_len: 128,
            dropout_p: 0.1,
            eps: 1e-6,
            rope_base: 10000.0,
        }
    }
}

impl ModelConfig {
    /// Llama-3 8B 相当の設定例
    pub fn llama3_small() -> Self {
        Self {
            vocab_size: 4096,
            d_model: 512,
            n_heads: 8,
            n_kv_heads: 8,
            d_ff: 1024,
            n_layers: 6,
            max_seq_len: 2048,
            dropout_p: 0.1,
            eps: 1e-6,
            rope_base: 10000.0,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
}
