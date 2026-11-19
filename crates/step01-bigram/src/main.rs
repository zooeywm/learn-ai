//! Step 01: 最简语言模型 (Bigram) —— 纯 Rust 零依赖跑通 LLM 训练完整闭环
//!
//! 通用基础组件（Rng / Tokenizer / softmax / sample_from_logits）已抽离到 `mini-core`，
//! 本文件聚焦于 Bigram 模型本身的前向查表、交叉熵 Loss、手推梯度与 SGD 训练循环。

use std::time::Instant;

use mini_core::{load_poems, sample_from_logits, softmax, Rng, Tokenizer};

// ============================================================================
// 1. Bigram 语言模型
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
            total_loss += -(probs[y] + 1e-12).ln();

            // 3) 反向传播：dL / d(logits_j) = probs[j] - (1 if j == y else 0)
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
        let mut ids = vec![start_id];
        let mut curr = start_id;
        for _ in 0..max_new_tokens {
            let next_id = sample_from_logits(self.forward_logits(curr), temperature, rng);
            ids.push(next_id);
            curr = next_id;
        }
        ids
    }
}

fn main() {
    let text = load_poems();
    let tokenizer = Tokenizer::from_text(&text);
    let data = tokenizer.encode(&text);
    let v = tokenizer.vocab_size();

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
    println!("理论初始随机猜测 Loss = ln({}) = {:.4}", v, expected_init_loss);
    println!("================================================================\n");

    let start_char = '床';
    let start_id = tokenizer.encode(&start_char.to_string())[0];
    let preview = |m: &BigramModel, seed: u64| -> String {
        let mut rng = Rng::new(seed);
        let ids = m.generate(start_id, 30, 0.8, &mut rng);
        tokenizer.decode(&ids).replace('\n', " / ")
    };

    println!("[训练前 Step    0] 「{}」-> {}", start_char, preview(&model, 42));

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
// 2. 单元测试：数值梯度验算
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_check() {
        let v = 4;
        let mut model = BigramModel::new(v);
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
