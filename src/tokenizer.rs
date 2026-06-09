use std::io::{Error, ErrorKind, Result};

use crate::{char_bpe_tokenizer::CharBpeTokenizer, char_tokenizer::CharTokenizer};

/// 学習・推論で使用するトークナイザの抽象。
/// 実装は次の 2 種類:
///   - `CharBpeTokenizer`: Unicode char-level BPE。 日本語含む任意の言語で lossless。
///   - `CharTokenizer`: char-level (merge なし)。 vocab はコーパス文字種から自動算出。
pub trait Tokenizer: Send {
    fn vocab_size(&self) -> usize;
    fn pad_id(&self) -> usize;
    /// 生成停止に用いる ID。 char-level など EOS を持たない実装は
    /// `usize::MAX` を返し、 通常の生成中に一致しないようにする。
    fn eos_id(&self) -> usize;
    /// コーパス全体を 1 度だけエンコード（BPE は BOS/EOS を付ける、 char は付けない）。
    fn encode_long(&self, text: &str) -> Vec<usize>;
    /// 推論プロンプト用のエンコード。
    fn encode_prompt(&self, text: &str) -> Vec<usize>;
    fn decode(&self, ids: &[usize]) -> String;
    fn kind(&self) -> TokenizerKind;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    Char,
    CharBpe,
}

impl TokenizerKind {
    pub fn as_u64(&self) -> u64 {
        match self {
            TokenizerKind::Char => 2,
            TokenizerKind::CharBpe => 3,
        }
    }

    pub fn from_u64(v: u64) -> Result<Self> {
        match v {
            2 => Ok(TokenizerKind::Char),
            3 => Ok(TokenizerKind::CharBpe),
            other => Err(Error::new(
                ErrorKind::InvalidData,
                format!("unknown tokenizer kind: {other}"),
            )),
        }
    }
}

/// テキストから新規にトークナイザを学習する。
pub fn train_tokenizer(kind: TokenizerKind, text: &str, vocab_size: usize) -> Box<dyn Tokenizer> {
    match kind {
        TokenizerKind::Char => Box::new(CharTokenizer::train(text)),
        TokenizerKind::CharBpe => Box::new(CharBpeTokenizer::train(text, vocab_size)),
    }
}
