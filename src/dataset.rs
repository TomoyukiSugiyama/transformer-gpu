// src/dataset.rs

use rand::{RngExt, rngs::StdRng};

pub struct Dataset {
    pub train: Vec<u32>,
    pub val: Vec<u32>,
    pub val_text_len_chars: usize,
}

impl Dataset {
    /// テキストファイルを corpus_text に読み込む
    pub fn from_file(path: &str) -> std::io::Result<String> {
        let corpus_text = std::fs::read_to_string(path)?;
        Ok(corpus_text)
    }

    /// &str から直接構築（テスト・main.rs 用）
    pub fn from_str(text: &str, val_split: f32) -> Self {
        assert!(val_split > 0.0 && val_split < 1.0);
        let all: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let seq_len = all.len();
        let train_seq = (seq_len as f32 * (1.0 - val_split)) as usize;
        assert!(seq_len - train_seq > 0);
        let train = all[..train_seq].to_vec();
        let val = all[train_seq..].to_vec();
        let val_text_len_chars = val.len();

        Self {
            train,
            val,
            val_text_len_chars,
        }
    }

    pub fn from_tokens(tokens: Vec<u32>, val_split: f32, corpus_text_len_chars: usize) -> Self {
        assert!(val_split > 0.0 && val_split < 1.0);
        let val_len = (tokens.len() as f32 * val_split) as usize;
        let train_len = tokens.len() - val_len;
        assert!(train_len > 0 && val_len > 0);

        let train = tokens[..train_len].to_vec();
        let val = tokens[train_len..].to_vec();
        let val_text_len_chars = (corpus_text_len_chars as f32 * val_split) as usize;

        Self {
            train,
            val,
            val_text_len_chars,
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
