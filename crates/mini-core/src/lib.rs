//! mini-core：各闯关阶段复用的零依赖基础组件
//!
//! - `Rng`               : 确定性伪随机数与高斯初始化采样
//! - `Tokenizer`         : 字符级分词器
//! - `softmax`           : 数值稳定 Softmax
//! - `sample_from_logits`: 温度缩放与轮盘赌采样
//! - `load_poems`        : 读取共享唐诗预训练语料 `data/poems.txt`
//! - `load_sft_dataset`  : 读取标准 JSONL 格式指令微调数据集 `data/sft_poems.jsonl`

pub mod math;
pub mod rng;
pub mod tokenizer;

pub use math::{sample_from_logits, softmax};
pub use rng::Rng;
pub use tokenizer::Tokenizer;

use std::fs;
use std::path::Path;

/// 读取仓库根目录下的共享语料 `data/poems.txt`
pub fn load_poems() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/poems.txt");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取语料失败 {:?}: {}", path, e))
}

/// 读取仓库根目录下的 SFT 指令微调数据集 `data/sft_poems.jsonl`
///
/// 每行为标准 JSON 对象：`{"user": "...", "assistant": "..."}`
pub fn load_sft_dataset() -> Vec<(String, String)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/sft_poems.jsonl");
    let content =
        fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取 SFT 数据集失败 {:?}: {}", path, e));
    parse_sft_jsonl(&content)
}

/// 零依赖解析简单 JSONL 行：`{"user": "...", "assistant": "..."}`
pub fn parse_sft_jsonl(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for (line_idx, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let user = extract_json_string_field(line, "user")
            .unwrap_or_else(|| panic!("SFT JSONL 第 {} 行缺少 \"user\" 字段: {}", line_idx + 1, line));
        let assistant = extract_json_string_field(line, "assistant").unwrap_or_else(|| {
            panic!(
                "SFT JSONL 第 {} 行缺少 \"assistant\" 字段: {}",
                line_idx + 1,
                line
            )
        });
        pairs.push((user, assistant));
    }
    pairs
}

fn extract_json_string_field(line: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let key_pos = line.find(&pattern)?;
    let rest = &line[key_pos + pattern.len()..];
    let colon_pos = rest.find(':')?;
    let after_colon = rest[colon_pos + 1..].trim_start();
    let mut chars = after_colon.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut out = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                other => out.push(other),
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_and_parse_sft_jsonl() {
        let sample = "{\"user\": \"问\\\"题\\\"\", \"assistant\": \"第一行\\n第二行\"}\n\n{\"user\": \"你好\", \"assistant\": \"世界\"}";
        let parsed = parse_sft_jsonl(sample);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], ("问\"题\"".to_string(), "第一行\n第二行".to_string()));
        assert_eq!(parsed[1], ("你好".to_string(), "世界".to_string()));

        let real_data = load_sft_dataset();
        assert!(real_data.len() >= 18);
        assert_eq!(real_data[0].0, "你好");
    }
}
