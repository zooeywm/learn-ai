//! 共享数值计算工具：数值稳定 Softmax 与温度轮盘采样

use crate::Rng;

/// 数值稳定的 Softmax：先减去最大值再做 exp，防止 f32 上溢成 inf / NaN
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&z| (z - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

/// 温度缩放 + 轮盘赌采样：根据 logits / temperature 的 Softmax 概率随机抽取下一个 token ID
pub fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut Rng) -> usize {
    let temp = temperature.max(1e-5);
    let scaled: Vec<f32> = logits.iter().map(|&z| z / temp).collect();
    let probs = softmax(&scaled);

    let r = rng.next_f32();
    let mut cdf = 0.0f32;
    for (id, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return id;
        }
    }
    probs.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one_and_numerically_stable() {
        let probs = softmax(&[1000.0, 1001.0, 1002.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs.iter().all(|p| p.is_finite() && *p > 0.0));
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }
}
