//! 字符级 Tokenizer：建立 字符 <-> 整数 ID 的确定性双向映射

use std::collections::{BTreeSet, HashMap};

pub struct Tokenizer {
    stoi: HashMap<char, usize>,
    itos: Vec<char>,
}

impl Tokenizer {
    pub fn from_text(text: &str) -> Self {
        // BTreeSet 自动去重且按 Unicode 排序，确保每次运行的编号完全确定
        let unique_chars: BTreeSet<char> = text.chars().collect();
        let itos: Vec<char> = unique_chars.into_iter().collect();
        let stoi: HashMap<char, usize> = itos
            .iter()
            .enumerate()
            .map(|(id, &ch)| (ch, id))
            .collect();
        Self { stoi, itos }
    }

    pub fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|ch| *self.stoi.get(&ch).expect("遇到了词表外的未知字符"))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.itos[id]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_roundtrip() {
        let sample = "床前明月光，疑是地上霜。\n";
        let tok = Tokenizer::from_text(sample);
        let encoded = tok.encode(sample);
        assert_eq!(tok.decode(&encoded), sample);
    }
}
