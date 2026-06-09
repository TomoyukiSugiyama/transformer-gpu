use std::collections::HashMap;

use crate::tokenizer::{Tokenizer, TokenizerKind};

/// char-level tokenizer。
///
/// vocab レイアウト:
///   - id 0: UNK (学習コーパスに存在しない char が prompt 等で来た場合の fallback)
///   - id 1..=N: コーパス出現順 (= ソート順) の各 char
///
/// BOS/EOS/PAD は持たない。 生成停止は `max_new_token` でのみ行う想定なので
/// `eos_id()` は `usize::MAX` を返し、 通常の生成中に偶然一致しないようにする。
/// `pad_id()` は UNK と同じ id 0 を返す (実際の学習ループでは pad は出現しないため
/// マスキング上の影響なし)。
pub struct CharTokenizer {
    id_to_char: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// テキストに含まれる char をユニーク化・ソートして vocab を構築する。
    pub fn train(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();

        let mut id_to_char = Vec::with_capacity(chars.len() + 1);
        // id 0 は UNK (NUL を sentinel として持たせる)
        id_to_char.push('\0');
        id_to_char.extend(chars);

        let char_to_id: HashMap<char, usize> = id_to_char
            .iter()
            .enumerate()
            .skip(1) // id 0 (UNK) は char_to_id に登録しない
            .map(|(i, &c)| (c, i))
            .collect();

        Self {
            id_to_char,
            char_to_id,
        }
    }

    pub fn empty() -> Self {
        Self {
            id_to_char: Vec::new(),
            char_to_id: HashMap::new(),
        }
    }

    fn unk_id(&self) -> usize {
        0
    }

    fn encode_chars(&self, text: &str) -> Vec<usize> {
        let unk = self.unk_id();
        text.chars()
            .map(|c| self.char_to_id.get(&c).copied().unwrap_or(unk))
            .collect()
    }
}

impl Tokenizer for CharTokenizer {
    fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    fn pad_id(&self) -> usize {
        // UNK と同じ id。 学習コーパス内には UNK が出現しないため、
        // pad マスキングで本物のトークンが除外される心配はない。
        0
    }

    fn eos_id(&self) -> usize {
        // EOS を持たないので、 vocab に存在しえない値を返して
        // generate ループの停止条件に絶対一致させない。
        usize::MAX
    }

    fn encode_long(&self, text: &str) -> Vec<usize> {
        // BPE のような BOS/EOS は付けない。
        self.encode_chars(text)
    }

    fn encode_prompt(&self, text: &str) -> Vec<usize> {
        self.encode_chars(text)
    }

    fn decode(&self, ids: &[usize]) -> String {
        let unk = self.unk_id();
        let mut out = String::with_capacity(ids.len());
        for &id in ids {
            if id == unk {
                continue;
            }
            if let Some(&c) = self.id_to_char.get(id) {
                out.push(c);
            }
        }
        out
    }

    fn kind(&self) -> TokenizerKind {
        TokenizerKind::Char
    }
}
