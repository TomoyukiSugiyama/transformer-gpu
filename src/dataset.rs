// src/dataset.rs

use rand::{RngExt, rngs::StdRng};

pub struct Dataset {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
    pub val_chars: usize,
}

impl Dataset {
    /// テキストファイルをバイト列としてトークナイズし、train/val に分割
    pub fn from_file(path: &str, val_split: f32) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_str(&text, val_split))
    }

    /// &str から直接構築（テスト・main.rs 用）
    pub fn from_str(text: &str, val_split: f32) -> Self {
        assert!(val_split > 0.0 && val_split < 1.0);
        let val_chars = (text.len() as f32 * val_split) as usize;
        let all: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let seq_len = all.len();
        let train_seq = (seq_len as f32 * (1.0 - val_split)) as usize;
        assert!(seq_len - train_seq > 0);
        let train = all[..train_seq].to_vec();
        let val = all[train_seq..].to_vec();

        Self {
            train,
            val,
            val_chars,
        }
    }

    /// ランダムウィンドウを1つ取り出す（train_step に渡す用）
    pub fn sample_window(&self, split: Split, seq_len: usize, rng: &mut StdRng) -> Vec<u32> {
        let data = match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
        };

        // offset の最大値 = data.len() - (seq_len + 1)
        let max_off = data.len() - seq_len - 1;
        let offset = rng.random_range(0..=max_off);

        data[offset..offset + seq_len + 1].to_vec()
    }
}

pub enum Split {
    Train,
    Val,
}
