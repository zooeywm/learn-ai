//! 极简确定性伪随机数生成器（SplitMix64 混种 + xorshift64 + Box-Muller 高斯采样）

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
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

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// 返回 [0.0, 1.0) 均匀分布的浮点数
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// 返回 [0, upper) 均匀分布的整数
    pub fn next_usize(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "upper 必须大于 0");
        (self.next_u64() as usize) % upper
    }

    /// 标准正态分布采样 N(0, std^2)，用于神经网络权重随机初始化
    pub fn normal(&mut self, std: f32) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z0 * std
    }
}
