use std::io;

use crate::checkpoint::{Checkpointable, WeightMap};

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
            vocab_size: 256,
            d_model: 64,
            n_heads: 4,
            n_kv_heads: 4,
            d_ff: 128,
            n_layers: 2,
            max_seq_len: 128,
            dropout_p: 0.1,
            eps: 1e-6,
            rope_base: 10_000.0,
        }
    }
}

impl ModelConfig {
    /// Llama 3.2 (1B) 相当の設定例
    pub fn llama3_2() -> Self {
        Self {
            vocab_size: 128_256,
            d_model: 2048,
            n_heads: 32,
            n_kv_heads: 8, // GQA
            d_ff: 8192,
            n_layers: 16,
            max_seq_len: 131_072,
            dropout_p: 0.0,
            eps: 1e-6,
            rope_base: 500_000.0,
        }
    }

    /// GPT-2 small (124M) 相当の設定例
    pub fn gpt2_small() -> Self {
        Self {
            vocab_size: 50_257,
            d_model: 768,
            n_heads: 12,
            n_kv_heads: 12, // MHA
            d_ff: 3072,
            n_layers: 12,
            max_seq_len: 1024,
            dropout_p: 0.1,
            eps: 1e-5,
            rope_base: 10_000.0, // GPT-2 は RoPE ではない
        }
    }

    /// tiny_shakespeare の設定
    pub fn tiny_shakespeare() -> Self {
        Self {
            vocab_size: 4000,
            d_model: 256,
            n_heads: 8,
            n_kv_heads: 8,
            d_ff: 1024,
            n_layers: 4,
            max_seq_len: 128,
            dropout_p: 0.0,
            eps: 1e-6,
            rope_base: 10_000.0,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
}

impl Checkpointable for ModelConfig {
    fn to_weight_map(&self) -> WeightMap {
        let mut map = WeightMap::new();
        map.insert_scalar("vocab_size", self.vocab_size as u64);
        map.insert_scalar("d_model", self.d_model as u64);
        map.insert_scalar("n_heads", self.n_heads as u64);
        map.insert_scalar("n_kv_heads", self.n_kv_heads as u64);
        map.insert_scalar("d_ff", self.d_ff as u64);
        map.insert_scalar("n_layers", self.n_layers as u64);
        map.insert_scalar("max_seq_len", self.max_seq_len as u64);
        map
    }

    fn from_weight_map(&mut self, map: &WeightMap) -> io::Result<()> {
        macro_rules! check {
            ($key:expr, $actual:expr) => {
                if map.get_scalar($key)? != $actual as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "model config mismatch: {} expected={}, got={}",
                            $key,
                            $actual,
                            map.get_scalar($key)?
                        ),
                    ));
                }
            };
        }

        check!("vocab_size", self.vocab_size);
        check!("d_model", self.d_model);
        check!("n_heads", self.n_heads);
        check!("n_kv_heads", self.n_kv_heads);
        check!("d_ff", self.d_ff);
        check!("n_layers", self.n_layers);
        check!("max_seq_len", self.max_seq_len);

        Ok(())
    }
}
