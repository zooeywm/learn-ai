//! 从 UTF-8 原始字节开始训练可重放的合并规则。

use std::cmp::Reverse;
use std::collections::HashMap;
use std::io::{self, Read, Write};

use step06_modern_llm::{ChatEncoding, EOS_ID};

const FIRST_BYTE_ID: u16 = 4;
const BASE_VOCAB_SIZE: usize = 4 + 256;
const VOCAB_LIMIT: usize = 1152;
const MAX_TOKEN_BYTES: usize = 9;

type Pair = (u16, u16);

pub struct ByteBpe {
    merges: Vec<Pair>,
    tokens: Vec<Vec<u8>>,
}

impl ByteBpe {
    fn new() -> Self {
        let tokens = std::iter::repeat_with(Vec::new)
            .take(usize::from(FIRST_BYTE_ID))
            .chain((0..=u8::MAX).map(|byte| vec![byte]))
            .collect();
        Self {
            merges: Vec::new(),
            tokens,
        }
    }

    fn merged_len(&self, (left, right): Pair) -> usize {
        self.token_bytes(usize::from(left)).len() + self.token_bytes(usize::from(right)).len()
    }

    fn add_merge(&mut self, pair: Pair) -> u16 {
        let id = u16::try_from(self.tokens.len()).expect("词表上限小于 u16::MAX");
        let mut bytes = self.token_bytes(usize::from(pair.0)).to_vec();
        bytes.extend_from_slice(self.token_bytes(usize::from(pair.1)));
        self.merges.push(pair);
        self.tokens.push(bytes);
        id
    }
}

impl ChatEncoding for ByteBpe {
    const PACKAGE: &'static str = "step07-tokenizer";
    const TITLE: &'static str = "Step 07 · 古诗 Mini-LLM（字节级 BPE）";
    const CHECKPOINT_MAGIC: [u8; 4] = *b"BMLM";
    const CONSTRAIN_UTF8: bool = true;

    fn train(texts: &[&str]) -> Self {
        let mut tokenizer = Self::new();
        let mut chunks: Vec<_> = texts
            .iter()
            .flat_map(|text| text.split_inclusive('\n').map(byte_ids))
            .collect();
        while tokenizer.vocab_size() < VOCAB_LIMIT {
            let mut counts = HashMap::new();
            for chunk in &chunks {
                for window in chunk.windows(2) {
                    let pair = (window[0], window[1]);
                    if tokenizer.merged_len(pair) <= MAX_TOKEN_BYTES {
                        *counts.entry(pair).or_insert(0_usize) += 1;
                    }
                }
            }
            let Some((pair, count)) = counts
                .into_iter()
                .max_by_key(|&(pair, count)| (count, Reverse(pair)))
            else {
                break;
            };
            if count < 2 {
                break;
            }
            let id = tokenizer.add_merge(pair);
            for chunk in &mut chunks {
                merge_pair(chunk, pair, id);
            }
        }
        tokenizer
    }

    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn encode(&self, text: &str) -> (Vec<usize>, Vec<char>) {
        let mut ids = Vec::new();
        for chunk in text.split_inclusive('\n') {
            let mut bytes = byte_ids(chunk);
            for (offset, &pair) in self.merges.iter().enumerate() {
                let id = u16::try_from(BASE_VOCAB_SIZE + offset).expect("词表上限小于 u16::MAX");
                merge_pair(&mut bytes, pair, id);
            }
            ids.extend(bytes.into_iter().map(usize::from));
        }
        (ids, Vec::new())
    }

    fn token_bytes(&self, id: usize) -> &[u8] {
        &self.tokens[id]
    }

    fn save_vocab(&self, writer: &mut impl Write) -> io::Result<()> {
        for &(left, right) in &self.merges {
            writer.write_all(&left.to_le_bytes())?;
            writer.write_all(&right.to_le_bytes())?;
        }
        Ok(())
    }

    fn load_vocab(reader: &mut impl Read, vocab_size: usize) -> io::Result<Self> {
        if !(BASE_VOCAB_SIZE..=VOCAB_LIMIT).contains(&vocab_size) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "BPE 词表大小无效",
            ));
        }
        let mut tokenizer = Self::new();
        for id in BASE_VOCAB_SIZE..vocab_size {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            let left = u16::from_le_bytes([bytes[0], bytes[1]]);
            let right = u16::from_le_bytes([bytes[2], bytes[3]]);
            let valid_ids =
                usize::from(left.min(right)) > EOS_ID && usize::from(left.max(right)) < id;
            let pair = (left, right);
            if !valid_ids
                || tokenizer.merged_len(pair) > MAX_TOKEN_BYTES
                || tokenizer.merges.contains(&pair)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "无效的 BPE 合并规则",
                ));
            }
            tokenizer.add_merge(pair);
        }
        Ok(tokenizer)
    }
}

fn byte_ids(text: &str) -> Vec<u16> {
    text.bytes()
        .map(|byte| FIRST_BYTE_ID + u16::from(byte))
        .collect()
}

fn merge_pair(ids: &mut Vec<u16>, pair: Pair, merged_id: u16) {
    let mut read = 0;
    let mut write = 0;
    while read < ids.len() {
        if ids[read] == pair.0 && ids.get(read + 1) == Some(&pair.1) {
            ids[write] = merged_id;
            read += 2;
        } else {
            ids[write] = ids[read];
            read += 1;
        }
        write += 1;
    }
    ids.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_merges_roundtrip_and_reload_unseen_text() {
        let texts = ["明月明月\n明月明月", "举头望明月", "愿君多采撷"];
        let tokenizer = ByteBpe::train(&texts);
        assert!(tokenizer.encode("明月").0.len() < "明月".len());
        let mut saved = Vec::new();
        tokenizer.save_vocab(&mut saved).unwrap();
        let loaded = ByteBpe::load_vocab(&mut saved.as_slice(), tokenizer.vocab_size()).unwrap();

        let input = "  昵称\r\n换一个 🦀𠮷 <|eos|>  ";
        let (ids, oov) = tokenizer.encode(input);
        assert!(oov.is_empty());
        assert!(ids.iter().all(|&id| id > EOS_ID));
        assert_eq!(loaded.encode(input).0, ids);
        let decoded: Vec<_> = ids
            .iter()
            .flat_map(|&id| loaded.token_bytes(id).iter().copied())
            .collect();
        assert_eq!(decoded, input.as_bytes());
        assert_eq!(ByteBpe::train(&texts).merges, tokenizer.merges);
    }

    #[test]
    fn overlapping_pairs_merge_left_to_right() {
        let mut ids = vec![10, 10, 10, 10, 10];
        merge_pair(&mut ids, (10, 10), 260);
        assert_eq!(ids, [260, 260, 10]);
    }
}
