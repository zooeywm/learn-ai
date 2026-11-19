//! Step 01: 最简语言模型 (Bigram) —— 纯 Rust 零依赖跑通 LLM 训练完整闭环
//!
//! 这一关故意不引入 Attention 和多层神经网络，而是用最简单的 V×V 查表权重，
//! 亲手跑通所有大语言模型共有的 5 个底层步骤：
//!   1. Tokenizer（字 <-> 整数 ID）
//!   2. 前向传播（查表得到 logits，经 Softmax 变成概率分布）
//!   3. 交叉熵 Loss（衡量预测与真实下一个字的差距）
//!   4. 反向传播 + SGD（求梯度并更新参数）
//!   5. 温度采样生成（自回归逐字接龙）

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;
use std::time::Instant;

// ============================================================================
// 0. 极简伪随机数生成器 (xorshift64)
//    为了做到零依赖、结果可复现，自己写 10 行确定性 RNG，用于温度采样。
// ============================================================================
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        assert!(seed != 0, "xorshift64 种子不能为 0");
        // 用 SplitMix64 混合一次种子，避免小种子（如 42）高位全 0 导致首个随机数接近 0
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Self {
            state: if z == 0 { 1 } else { z },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// 返回 [0.0, 1.0) 均匀分布的浮点数
    fn next_f32(&mut self) -> f32 {
        // 取高 24 位除以 2^24，保证精度均匀落在 [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

// ============================================================================
// 1. 字符级 Tokenizer（分词器）
// ============================================================================
struct Tokenizer {
    stoi: HashMap<char, usize>, // char -> token_id
    itos: Vec<char>,            // token_id -> char
}

impl Tokenizer {
    fn from_text(text: &str) -> Self {
        // 用 BTreeSet 自动去重且按 Unicode 排序，保证每次运行的词表编号完全一致
        let unique_chars: BTreeSet<char> = text.chars().collect();
        let itos: Vec<char> = unique_chars.into_iter().collect();
        let stoi: HashMap<char, usize> =
            itos.iter().enumerate().map(|(id, &ch)| (ch, id)).collect();
        Self { stoi, itos }
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|ch| *self.stoi.get(&ch).expect("遇到了词表外的未知字符"))
            .collect()
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&id| self.itos[id]).collect()
    }
}

// ============================================================================
// 2. Softmax：把任意实数打分 (logits) 转换成和为 1 的概率分布
// ============================================================================
fn softmax(logits: &[f32]) -> Vec<f32> {
    // 数值稳定关键：先减去最大值，防止 exp() 溢出成 inf / NaN
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&z| (z - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

// ============================================================================
// 3. Bigram 语言模型
//    唯一的可训练参数是一张 V × V 的权重表：
//    第 x 行第 j 列 table[x * V + j] 表示：看到字 x 时，预测下一个字是 j 的打分。
// ============================================================================
struct BigramModel {
    vocab_size: usize,
    table: Vec<f32>, // 长度 V * V，按行优先 (row-major) 连续存储
}

impl BigramModel {
    fn new(vocab_size: usize) -> Self {
        // 全 0 初始化：初始时对所有候选字打分相同 -> 概率全为 1/V -> 初始 Loss = ln(V)
        Self {
            vocab_size,
            table: vec![0.0; vocab_size * vocab_size],
        }
    }

    /// 前向传播：查出当前 token 对应的 logits 切片（长度 V）
    fn forward_logits(&self, token_id: usize) -> &[f32] {
        let start = token_id * self.vocab_size;
        &self.table[start..start + self.vocab_size]
    }

    /// 给定一批 (当前字 x, 真实下一个字 y) 样本：
    /// 返回 (平均交叉熵 Loss, 与 table 同形状的梯度数组 grads)
    fn loss_and_grads(&self, pairs: &[(usize, usize)]) -> (f32, Vec<f32>) {
        let v = self.vocab_size;
        let n = pairs.len() as f32;
        let mut total_loss = 0.0f32;
        let mut grads = vec![0.0f32; v * v];

        for &(x, y) in pairs {
            // 1) 前向：取出第 x 行 logits，做 Softmax 得到概率分布 probs
            let logits = self.forward_logits(x);
            let probs = softmax(logits);

            // 2) 交叉熵损失：L = -ln(p_y)
            //    加上 1e-12 防止极端情况下 ln(0) 得到 -inf
            total_loss += -(probs[y] + 1e-12).ln();

            // 3) 反向传播：dL / d(logits_j) = probs[j] - (1 if j == y else 0)
            //    因为总 Loss 除以了样本数 N，梯度也相应除以 N
            let row_start = x * v;
            for j in 0..v {
                let target = if j == y { 1.0 } else { 0.0 };
                grads[row_start + j] += (probs[j] - target) / n;
            }
        }

        (total_loss / n, grads)
    }

    /// 执行一步 SGD 参数更新：w = w - lr * grad
    fn train_step(&mut self, pairs: &[(usize, usize)], lr: f32) -> f32 {
        let (loss, grads) = self.loss_and_grads(pairs);
        for (w, g) in self.table.iter_mut().zip(grads.iter()) {
            *w -= lr * g;
        }
        loss
    }

    /// 自回归温度采样生成
    fn generate(
        &self,
        start_id: usize,
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> Vec<usize> {
        let temp = temperature.max(1e-5);
        let mut ids = vec![start_id];
        let mut curr = start_id;

        for _ in 0..max_new_tokens {
            let scaled_logits: Vec<f32> = self
                .forward_logits(curr)
                .iter()
                .map(|&z| z / temp)
                .collect();
            let probs = softmax(&scaled_logits);

            // 轮盘赌采样：在 [0, 1) 掷骰子，累加概率直到超过骰子值
            let r = rng.next_f32();
            let mut cdf = 0.0f32;
            let mut next_id = self.vocab_size - 1;
            for (id, &p) in probs.iter().enumerate() {
                cdf += p;
                if r < cdf {
                    next_id = id;
                    break;
                }
            }

            ids.push(next_id);
            curr = next_id;
        }

        ids
    }
}

fn main() {
    // 1. 读取唐诗语料（相对本 crate 目录向上两级找到 data/poems.txt）
    let data_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/poems.txt");
    let text = fs::read_to_string(&data_path)
        .unwrap_or_else(|e| panic!("读取语料失败 {:?}: {}", data_path, e));

    // 2. 构建字符级 Tokenizer，并把整篇文本编码为 token IDs
    let tokenizer = Tokenizer::from_text(&text);
    let data = tokenizer.encode(&text);
    let v = tokenizer.vocab_size();

    // 3. 构造所有相邻字符对 (x, y)：看到 data[i]，目标是预测 data[i + 1]
    let pairs: Vec<(usize, usize)> = data.windows(2).map(|w| (w[0], w[1])).collect();

    let mut model = BigramModel::new(v);
    let expected_init_loss = (v as f32).ln();
    println!("================================================================");
    println!("Step 01: 纯 Rust Bigram 语言模型");
    println!(
        "语料长度: {} 字符 | 词表大小 V: {} | 总参数量 V×V: {}",
        data.len(),
        v,
        v * v
    );
    println!(
        "理论初始随机猜测 Loss = ln({}) = {:.4}",
        v, expected_init_loss
    );
    println!("================================================================\n");

    let start_char = '床';
    let start_id = tokenizer.encode(&start_char.to_string())[0];
    let preview = |m: &BigramModel, seed: u64| -> String {
        let mut rng = Rng::new(seed);
        let ids = m.generate(start_id, 30, 0.8, &mut rng);
        tokenizer.decode(&ids).replace('\n', " / ")
    };

    println!(
        "[训练前 Step    0] 「{}」-> {}",
        start_char,
        preview(&model, 42)
    );

    // 4. 训练循环（纯 SGD，全量样本每步更新一次）
    //    注意：对 Bigram 这种极简线性查表，因为全量 869 个样本分摊到 373 行，
    //    单行每步实际只出现 2~3 次，梯度除以 N=869 后很小，所以学习率可以设大一些（如 80.0）。
    let steps = 1500;
    let lr = 80.0;
    let t0 = Instant::now();

    for step in 1..=steps {
        let loss = model.train_step(&pairs, lr);
        if matches!(step, 1 | 50 | 200 | 500 | 1000 | 1500) {
            println!(
                "[Step {:4}/{}] Loss: {:.4} ({:.2}s) | 「{}」-> {}",
                step,
                steps,
                loss,
                t0.elapsed().as_secs_f32(),
                start_char,
                preview(&model, 42)
            );
        }
    }

    // 5. 观察温度 (temperature) 对生成结果的影响
    println!("\n================================================================");
    println!("训练完成！对比不同 temperature 的生成效果（起始字「明」）：");
    println!("================================================================");
    let ming_id = tokenizer.encode("明")[0];
    for temp in [0.2, 0.8, 1.5] {
        let mut rng = Rng::new(2026);
        let out = tokenizer
            .decode(&model.generate(ming_id, 36, temp, &mut rng))
            .replace('\n', " / ");
        println!("temperature = {:.1} -> {}", temp, out);
    }
}

// ============================================================================
// 4. 单元测试（极其重要！从第一关就养成用数值梯度验证数学公式的习惯）
// ============================================================================
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

    #[test]
    fn test_softmax_sums_to_one_and_numerically_stable() {
        // 即使 logits 高达 1000.0，也不应出现 NaN 或 inf
        let probs = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|p| p.is_finite() && *p > 0.0));
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_gradient_check() {
        // 用有限差分法 (L(w + eps) - L(w - eps)) / (2 * eps)
        // 验证我们手推的梯度公式 (probs[j] - target) / N 是否完全正确！
        let v = 4;
        let mut model = BigramModel::new(v);
        // 随便给点非零权重，避免特殊对称点掩盖 Bug
        for (i, w) in model.table.iter_mut().enumerate() {
            *w = ((i as f32) * 0.37).sin();
        }
        let pairs = vec![(0, 1), (1, 2), (0, 3), (2, 0)];

        let (_, analytical_grads) = model.loss_and_grads(&pairs);

        let eps = 1e-3f32;
        for idx in 0..model.table.len() {
            let orig = model.table[idx];

            model.table[idx] = orig + eps;
            let (loss_plus, _) = model.loss_and_grads(&pairs);

            model.table[idx] = orig - eps;
            let (loss_minus, _) = model.loss_and_grads(&pairs);

            model.table[idx] = orig;

            let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let diff = (analytical_grads[idx] - numerical_grad).abs();
            assert!(
                diff < 1e-3,
                "权重索引 {} 梯度不匹配: 解析梯度={} vs 数值梯度={}",
                idx,
                analytical_grads[idx],
                numerical_grad
            );
        }
    }
}
