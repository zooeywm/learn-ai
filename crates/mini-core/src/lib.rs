//! mini-core：各闯关阶段复用的零依赖基础组件
//!
//! - `Rng`               : 确定性伪随机数与高斯初始化采样
//! - `Tokenizer`         : 字符级分词器
//! - `softmax`           : 数值稳定 Softmax
//! - `sample_from_logits`: 温度缩放与轮盘赌采样
//! - `load_poems`        : 读取共享唐诗语料 `data/poems.txt`

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
