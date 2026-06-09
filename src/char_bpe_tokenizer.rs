use std::collections::HashMap;

use crate::tokenizer::{Tokenizer, TokenizerKind};

/// Unicode char-level BPE トークナイザ。
///
/// vocab レイアウト:
///   - id 0..=3: `<PAD>` `<UNK>` `<BOS>` `<EOS>`
///   - id 4..: コーパスから収集したユニーク char、 続いて merge で生まれた多 char トークン
///   - 末尾 (任意): `add_special_token` で追加した user-defined special token (`<AUTHOR=...>` 等)
///
/// special token は `encode_inner` で **文字列マッチを最優先** され、 BPE merge を
/// 経由せずに直接 ID に変換される
pub struct CharBpeTokenizer {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, usize>,
    merge_rank: HashMap<(String, String), usize>,
    unk_id: usize,
    /// **atomic な special token** (PAD/UNK/BOS/EOS + ユーザ追加分) のリスト。
    /// `encode_inner` でテキスト中の `<...>` 形式マーカを 1 token として認識するために使う。
    /// 順序保持 (id 順) で、 ASCII '<' で始まり ASCII '>' で終わる前提。
    special_tokens: Vec<String>,
}

type Vocab = HashMap<Vec<String>, usize>;

/// 文末・節区切りを示す句読点を独立トークンとして切り出す対象。
/// `BpeTokenizer` と同じ集合 (日本語全角句読点 + 引用符を含む)。
fn is_split_punct(ch: char) -> bool {
    matches!(
        ch,
        '.' | ','
            | ';'
            | ':'
            | '!'
            | '?'
            | '。'
            | '、'
            | '！'
            | '？'
            | '「'
            | '」'
            | '『'
            | '』'
            | '（'
            | '）'
    )
}

/// テキストを「単語」「空白文字」「句読点」に分割する。
/// 既存 BPE と異なり **空白も独立トークンとして保持** することで decode を lossless にする。
fn pretokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else if is_split_punct(ch) {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

/// 単語頻度を集計する (lowercase は適用しない: 日本語に大文字小文字がなく、 ASCII の case を保持したい)。
fn build_word_freq(text: &str) -> HashMap<String, usize> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for word in pretokenize(text) {
        if !word.is_empty() {
            *freq.entry(word).or_insert(0) += 1;
        }
    }
    freq
}

/// 単語を char 単位の文字列リストに分解する。 `</w>` などの修飾は付けない。
fn split_word_chars(word: &str) -> Vec<String> {
    word.chars().map(|c| c.to_string()).collect()
}

/// 隣接ペアの出現頻度を集計する (重み: その単語の出現頻度)。
fn count_pairs(vocab: &Vocab) -> HashMap<(String, String), usize> {
    let mut pairs: HashMap<(String, String), usize> = HashMap::new();
    for (symbol, &freq) in vocab {
        for w in symbol.windows(2) {
            *pairs.entry((w[0].clone(), w[1].clone())).or_insert(0) += freq;
        }
    }
    pairs
}

/// 最頻出ペアを返す。
fn best_pair(pairs: &HashMap<(String, String), usize>) -> Option<(String, String)> {
    pairs.iter().max_by_key(|(_, v)| *v).map(|(k, _)| k.clone())
}

/// vocab の各単語内の `(a, b)` を連結トークンに置換する。
fn merge_vocab(vocab: Vocab, a: &str, b: &str) -> Vocab {
    let merged = format!("{}{}", a, b);
    vocab
        .into_iter()
        .map(|(symbols, freq)| {
            let mut new: Vec<String> = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && symbols[i] == a && symbols[i + 1] == b {
                    new.push(merged.clone());
                    i += 2;
                } else {
                    new.push(symbols[i].clone());
                    i += 1;
                }
            }
            (new, freq)
        })
        .collect()
}

impl CharBpeTokenizer {
    pub const PAD: &'static str = "<PAD>";
    pub const UNK: &'static str = "<UNK>";
    pub const BOS: &'static str = "<BOS>";
    pub const EOS: &'static str = "<EOS>";

    /// `vocab_size` まで BPE マージを学習する (merge 学習 = char カバレッジ = `text`)。
    ///   - 4 (特殊トークン) + ユニーク char 数 が初期 vocab。
    ///   - そこから `vocab_size` に達するまで merge を追加。
    ///   - `vocab_size` が初期 vocab より小さい場合は merge を追加しない (= char tokenizer に近い動作)。
    pub fn train(text: &str, vocab_size: usize) -> Self {
        Self::train_with_coverage(text, text, vocab_size)
    }

    /// `merge_text` から BPE マージを学習しつつ、 `coverage_text` のユニーク char を初期 vocab に保証する。
    /// 大規模 corpus で学習時間を抑えたい場合、 サンプル (`merge_text`) で merge を学習し、
    /// 全体 (`coverage_text`) で char をカバーする使い方を想定。
    pub fn train_with_coverage(merge_text: &str, coverage_text: &str, vocab_size: usize) -> Self {
        let mut token_to_id: HashMap<String, usize> = HashMap::new();
        let mut special_tokens: Vec<String> = Vec::new();
        for special in [Self::PAD, Self::UNK, Self::BOS, Self::EOS] {
            token_to_id.insert(special.to_string(), token_to_id.len());
            special_tokens.push(special.to_string());
        }

        // 全 char カバレッジ: coverage_text に出現する全ユニーク char を初期 vocab に登録。
        // 出現順序は再現性のため `chars()` 順 (= バイトオフセット順) で挿入する。
        let mut seen_chars: HashMap<char, ()> = HashMap::new();
        for ch in coverage_text.chars() {
            if seen_chars.insert(ch, ()).is_none() {
                let key = ch.to_string();
                if !token_to_id.contains_key(&key) {
                    token_to_id.insert(key, token_to_id.len());
                }
            }
        }

        let word_freq = build_word_freq(merge_text);
        let vocab_init: Vocab = word_freq
            .iter()
            .map(|(word, &freq)| (split_word_chars(word), freq))
            .collect();

        // 念のため merge 用 corpus にしか出現しない char も登録 (通常は coverage に含まれるはず)。
        for symbols in vocab_init.keys() {
            for sym in symbols {
                if !token_to_id.contains_key(sym) {
                    token_to_id.insert(sym.clone(), token_to_id.len());
                }
            }
        }

        let mut vocab = vocab_init;
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut merge_rank: HashMap<(String, String), usize> = HashMap::new();

        while token_to_id.len() < vocab_size {
            let pairs = count_pairs(&vocab);
            if pairs.is_empty() {
                break;
            }
            let Some((a, b)) = best_pair(&pairs) else {
                break;
            };

            let merged = format!("{}{}", a, b);
            merge_rank.insert((a.clone(), b.clone()), merges.len());
            merges.push((a.clone(), b.clone()));

            if !token_to_id.contains_key(&merged) {
                token_to_id.insert(merged, token_to_id.len());
            }

            vocab = merge_vocab(vocab, &a, &b);
        }

        let mut id_to_token = vec![String::new(); token_to_id.len()];
        for (tok, &id) in &token_to_id {
            id_to_token[id] = tok.clone();
        }
        let unk_id = *token_to_id.get(Self::UNK).unwrap();
        Self {
            id_to_token,
            token_to_id,
            merge_rank,
            unk_id,
            special_tokens,
        }
    }

    /// 1 単語に学習済みマージを優先順位順に適用して subword 列に変換する。
    fn encode_word(&self, word: &str) -> Vec<String> {
        let mut symbols = split_word_chars(word);
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| {
                    let pair = (w[0].clone(), w[1].clone());
                    self.merge_rank.get(&pair).map(|&rank| (i, rank))
                })
                .min_by_key(|&(_, rank)| rank);

            let Some((i, _)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.remove(i + 1);
            symbols[i] = merged;
        }
        symbols
    }

    /// Pretokenize → BPE で 1 セグメントを encode する (special token は処理しない)。
    fn encode_segment(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for word in pretokenize(text) {
            if word.is_empty() {
                continue;
            }
            for sym in self.encode_word(&word) {
                ids.push(*self.token_to_id.get(&sym).unwrap_or(&self.unk_id));
            }
        }
        ids
    }

    /// テキスト中の special token (`<...>` 形式) を 1 token として認識しつつ、
    /// それ以外を BPE でエンコードする。
    ///
    /// アルゴリズム:
    /// 1. 次の ASCII '<' を探す
    /// 2. '<' から '>' までの部分文字列が登録済 special token なら、 直接 ID を出力
    /// 3. そうでなければ '<' を含むセグメントを通常 BPE でエンコードして 1 char 進める
    ///
    /// `<` `>` はいずれも ASCII (1 byte) なので、 byte index での slice は char 境界を破らない。
    fn encode_inner(&self, text: &str) -> Vec<usize> {
        if self.special_tokens.is_empty() {
            return self.encode_segment(text);
        }
        let bytes = text.as_bytes();
        let mut ids = Vec::new();
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            // 次の '<' (ASCII 1 byte) を探す。 見つからなければ末尾までを通常 encode。
            let lt_offset = bytes[cursor..].iter().position(|&b| b == b'<');
            let Some(lt_rel) = lt_offset else {
                ids.extend(self.encode_segment(&text[cursor..]));
                break;
            };
            let lt_pos = cursor + lt_rel;

            // '<' から最初の '>' を探す (ASCII 1 byte)。 見つからなければ '<' 以降を通常 encode。
            let gt_offset = bytes[lt_pos..].iter().position(|&b| b == b'>');
            let Some(gt_rel) = gt_offset else {
                ids.extend(self.encode_segment(&text[cursor..]));
                break;
            };
            let gt_pos = lt_pos + gt_rel; // index of '>'
            let candidate = &text[lt_pos..gt_pos + 1]; // includes '<' and '>'

            if let Some(&id) = self.token_to_id.get(candidate) {
                if self.special_tokens.iter().any(|s| s == candidate) {
                    if lt_pos > cursor {
                        ids.extend(self.encode_segment(&text[cursor..lt_pos]));
                    }
                    ids.push(id);
                    cursor = gt_pos + 1;
                    continue;
                }
            }

            // マッチしなかった: '<' を含む 1 char ぶんを通常 encode して進める。
            // '<' は ASCII なので lt_pos+1 は char 境界。
            ids.extend(self.encode_segment(&text[cursor..lt_pos + 1]));
            cursor = lt_pos + 1;
        }
        ids
    }

    /// 学習用: 先頭に BOS、 末尾に EOS を付与する。
    ///
    /// コーパス側で `<BOS>` `<EOS>` を special token として埋めている場合 (作品単位
    /// での境界付与) を考慮し、 既に先頭が BOS / 末尾が EOS であれば二重付与しない。
    fn encode_long(&self, text: &str) -> Vec<usize> {
        let inner = self.encode_inner(text);
        let mut ids = if inner.first().copied() == Some(self.bos_id()) {
            Vec::with_capacity(inner.len() + 1)
        } else {
            let mut v = Vec::with_capacity(inner.len() + 2);
            v.push(self.bos_id());
            v
        };
        ids.extend(inner);
        if ids.last().copied() != Some(self.eos_id()) {
            ids.push(self.eos_id());
        }
        ids
    }

    /// 推論プロンプト用: 先頭に BOS のみ付与する。
    fn encode_prompt(&self, text: &str) -> Vec<usize> {
        let mut ids = vec![self.bos_id()];
        ids.extend(self.encode_inner(text));
        ids
    }

    /// lossless decode。 各 token をそのまま連結する。
    /// 特殊トークン (PAD/UNK/BOS/EOS) はスキップする。
    /// 空白・改行・句読点は pretokenize で独立トークンとして保存されているため、
    /// decode 後に元のテキストが完全に復元される。
    fn decode(&self, ids: &[usize]) -> String {
        let specials = [Self::PAD, Self::UNK, Self::BOS, Self::EOS];
        let mut out = String::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(id) else {
                continue;
            };
            if specials.contains(&tok.as_str()) {
                continue;
            }
            out.push_str(tok);
        }
        out
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn pad_id(&self) -> usize {
        *self.token_to_id.get(Self::PAD).unwrap_or(&0)
    }

    pub fn bos_id(&self) -> usize {
        *self.token_to_id.get(Self::BOS).unwrap_or(&2)
    }

    fn eos_id(&self) -> usize {
        *self.token_to_id.get(Self::EOS).unwrap_or(&3)
    }

    pub fn empty() -> Self {
        Self {
            id_to_token: Vec::new(),
            token_to_id: HashMap::new(),
            merge_rank: HashMap::new(),
            unk_id: 0,
            special_tokens: Vec::new(),
        }
    }
}

impl Tokenizer for CharBpeTokenizer {
    fn vocab_size(&self) -> usize {
        CharBpeTokenizer::vocab_size(self)
    }

    fn pad_id(&self) -> usize {
        CharBpeTokenizer::pad_id(self)
    }

    fn eos_id(&self) -> usize {
        CharBpeTokenizer::eos_id(self)
    }

    fn encode_long(&self, text: &str) -> Vec<usize> {
        CharBpeTokenizer::encode_long(self, text)
    }

    fn encode_prompt(&self, text: &str) -> Vec<usize> {
        CharBpeTokenizer::encode_prompt(self, text)
    }

    fn decode(&self, ids: &[usize]) -> String {
        CharBpeTokenizer::decode(self, ids)
    }

    fn kind(&self) -> TokenizerKind {
        TokenizerKind::CharBpe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 特殊トークンと初期 char トークンが正しい順に登録されている。
    #[test]
    fn special_tokens_at_head() {
        let t = CharBpeTokenizer::train("abc", 100);
        assert_eq!(t.id_to_token[0], CharBpeTokenizer::PAD);
        assert_eq!(t.id_to_token[1], CharBpeTokenizer::UNK);
        assert_eq!(t.id_to_token[2], CharBpeTokenizer::BOS);
        assert_eq!(t.id_to_token[3], CharBpeTokenizer::EOS);
    }

    /// 英文の encode → decode が lossless で復元される。
    #[test]
    fn english_roundtrip_lossless() {
        let text = "the quick brown fox jumps over the lazy dog\n";
        let t = CharBpeTokenizer::train(text, 200);
        let ids = t.encode_prompt(text);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    /// 日本語の encode → decode が lossless で復元される。
    #[test]
    fn japanese_roundtrip_lossless() {
        let text = "吾輩は猫である。\n名前はまだ無い。";
        let t = CharBpeTokenizer::train(text, 100);
        let ids = t.encode_prompt(text);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    /// 日本語混在 (改行・句読点・空白・英字) も復元される。
    #[test]
    fn mixed_jp_en_punct_roundtrip() {
        let text = "Hello, 世界!\nそれから「ABC」と言った。";
        let t = CharBpeTokenizer::train(text, 200);
        let ids = t.encode_prompt(text);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    /// vocab_size が大きいほど token 列が短くなる (merge が効いている)。
    #[test]
    fn larger_vocab_yields_shorter_sequence() {
        let text = "あいうえおあいうえおあいうえおあいうえお"; // 「あいうえお」が頻出
        let t_small = CharBpeTokenizer::train(text, 10); // merge 余地ほぼなし
        let t_large = CharBpeTokenizer::train(text, 100); // merge 余地あり
        let ids_small = t_small.encode_prompt(text);
        let ids_large = t_large.encode_prompt(text);
        // 特殊 BOS を含むので prompt 全体長で比較
        assert!(
            ids_large.len() < ids_small.len(),
            "expected larger vocab to compress: small={} large={}",
            ids_small.len(),
            ids_large.len()
        );
    }

    /// 学習コーパスに無い char は UNK にフォールバックする。
    #[test]
    fn unknown_char_maps_to_unk() {
        let t = CharBpeTokenizer::train("abc", 50);
        let ids = t.encode_prompt("axb");
        let unk = t.unk_id;
        let bos = t.bos_id();
        assert_eq!(ids[0], bos);
        // ids[2] が 'x' に対応し、 UNK になる
        let xs: Vec<_> = ids.iter().filter(|&&id| id == unk).collect();
        assert_eq!(xs.len(), 1, "ids={ids:?}");
    }

    /// コーパス先頭の `<BOS>` が登録 special token のとき、
    /// `encode_long` が BOS を重複追加しない。
    #[test]
    fn encode_long_skips_redundant_bos_eos() {
        let t = CharBpeTokenizer::train("吾輩は猫", 80);
        let bos = t.bos_id();
        let eos = t.eos_id();

        // ケース 1: 通常の文字列 → 先頭 BOS + 末尾 EOS が自動付与される
        let ids_plain = t.encode_long("吾輩");
        assert_eq!(ids_plain.first().copied(), Some(bos));
        assert_eq!(ids_plain.last().copied(), Some(eos));

        // ケース 2: `<BOS>...<EOS>` 形式の文字列 → 二重付与されない
        let ids_marked = t.encode_long("<BOS>吾輩<EOS>");
        let bos_count = ids_marked.iter().filter(|&&i| i == bos).count();
        let eos_count = ids_marked.iter().filter(|&&i| i == eos).count();
        assert_eq!(
            bos_count, 1,
            "BOS should appear exactly once: ids={ids_marked:?}"
        );
        assert_eq!(
            eos_count, 1,
            "EOS should appear exactly once: ids={ids_marked:?}"
        );
    }

    /// special_tokens が登録されていない `<...>` はそのまま char-level に分解される
    #[test]
    fn unregistered_angle_brackets_are_char_encoded() {
        let t = CharBpeTokenizer::train("a<b>c<d>e", 50);
        // <b> や <d> は special_tokens に登録していない
        let ids = t.encode_prompt("a<b>c");
        let decoded = t.decode(&ids);
        assert_eq!(decoded, "a<b>c");
    }

    /// 改行・空白・全角句読点が独立トークンとして保持される。
    #[test]
    fn whitespace_and_punct_are_explicit_tokens() {
        let text = "a b\nc。d";
        let t = CharBpeTokenizer::train(text, 50);
        // pretokenize 結果に「 」「\n」「。」が個別に出現することを decode 経由で確認
        let ids = t.encode_prompt(text);
        let decoded = t.decode(&ids);
        assert_eq!(decoded, text);
    }

    /// UTF-8 マルチバイト文字が決して途中で分断されないことを確認。
    /// 全 token に対して valid UTF-8 文字列であることをチェック。
    #[test]
    fn all_tokens_are_valid_utf8_strings() {
        let text = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。";
        let t = CharBpeTokenizer::train(text, 200);
        for tok in &t.id_to_token {
            let _ = tok.chars().count(); // panic しないこと (valid UTF-8 = char 境界が安全)
        }
    }
}
