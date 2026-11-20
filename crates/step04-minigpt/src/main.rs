//! Step 04（终极关）: 完整纯 Rust Pre-Norm Transformer Mini-GPT
//!
//! 补齐现代 GPT / LLaMA 架构的最后两块标准拼图：
//!   1. LayerNorm（层归一化）：在进入 Attention / FFN 之前把每个 token 的特征稳压到均值 0、方差 1，
//!      再经可学习的 gamma / beta 缩放平移，让多层深网也能平稳训练；
//!   2. FFN（前馈非线性思考层：Linear(D -> 4D) -> GELU -> Linear(4D -> D)）：
//!      Attention 负责「跨位置搜集前文线索」，FFN 负责「在每个位置做 4 倍扩张的深层非线性推理」；
//!   3. Pre-Norm Transformer Block 多层堆叠（2 层）：
//!        x = x + CausalMultiHeadAttn(LayerNorm1(x))
//!        x = x + FFN(LayerNorm2(x))
//!   4. Final LayerNorm + LM Head + AdamW + 梯度范数裁剪 + Warmup 余弦学习率调度！
//!
//! 本关实战目标：
//!   在 30 首唐诗语料上从零训练完整的 2 层 Mini-GPT，不仅继续稳过 Step 02 的「床前明月光，疑是」，
//!   还彻底治愈 Step 03 单层 Attention 在「春眠不觉晓，」处的「处处处处...」复读症！

use std::f32::consts::PI;
use std::time::Instant;

use mini_core::{load_poems, sample_from_logits, softmax, Rng, Tokenizer};

// ============================================================================
// 1. 完整版 Arena-Tape 自动求导引擎（新增 LayerNorm 与 GELU 精确解析反传）
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorId(usize);

enum Op {
    Leaf,
    Add(TensorId, TensorId),
    MatMul(TensorId, TensorId),
    Gelu(TensorId),
    LayerNorm {
        x: TensorId,
        gamma: TensorId,
        beta: TensorId,
        x_hat: Vec<f32>,
        inv_std: Vec<f32>,
    },
    EmbedPos {
        tok_emb: TensorId,
        pos_emb: TensorId,
        seqs: Vec<Vec<usize>>,
    },
    CausalMultiHeadAttn {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
        attn_weights: Vec<f32>,
    },
    CrossEntropy {
        logits: TensorId,
        targets: Vec<usize>,
    },
}

struct Node {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    grad: Vec<f32>,
    op: Op,
}

struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, rows: usize, cols: usize, data: Vec<f32>, op: Op) -> TensorId {
        assert_eq!(data.len(), rows * cols);
        let id = TensorId(self.nodes.len());
        self.nodes.push(Node {
            rows,
            cols,
            grad: vec![0.0; rows * cols],
            data,
            op,
        });
        id
    }

    fn param(&mut self, p: &Param) -> TensorId {
        self.push(p.rows, p.cols, p.data.clone(), Op::Leaf)
    }

    fn data(&self, id: TensorId) -> &[f32] {
        &self.nodes[id.0].data
    }

    fn grad(&self, id: TensorId) -> &[f32] {
        &self.nodes[id.0].grad
    }

    fn shape(&self, id: TensorId) -> (usize, usize) {
        let n = &self.nodes[id.0];
        (n.rows, n.cols)
    }

    fn embed_pos(&mut self, tok_emb: TensorId, pos_emb: TensorId, seqs: &[Vec<usize>]) -> TensorId {
        let (_v, d) = self.shape(tok_emb);
        let (max_t, pd) = self.shape(pos_emb);
        assert_eq!(d, pd);
        let b_cnt = seqs.len();
        let t_len = seqs[0].len();
        assert!(t_len <= max_t);

        let tok_data = &self.nodes[tok_emb.0].data;
        let pos_data = &self.nodes[pos_emb.0].data;
        let mut out = vec![0.0f32; b_cnt * t_len * d];
        for (b, seq) in seqs.iter().enumerate() {
            for (t, &tok_id) in seq.iter().enumerate() {
                let out_off = (b * t_len + t) * d;
                let tok_off = tok_id * d;
                let pos_off = t * d;
                for c in 0..d {
                    out[out_off + c] = tok_data[tok_off + c] + pos_data[pos_off + c];
                }
            }
        }
        self.push(
            b_cnt * t_len,
            d,
            out,
            Op::EmbedPos {
                tok_emb,
                pos_emb,
                seqs: seqs.to_vec(),
            },
        )
    }

    /// 逐行层归一化：对每个 token 向量做 (x - mean) / sqrt(var + eps) * gamma + beta
    fn layer_norm(&mut self, x: TensorId, gamma: TensorId, beta: TensorId) -> TensorId {
        let (rows, cols) = self.shape(x);
        assert_eq!(self.shape(gamma), (1, cols));
        assert_eq!(self.shape(beta), (1, cols));

        let x_data = &self.nodes[x.0].data;
        let g_data = &self.nodes[gamma.0].data;
        let b_data = &self.nodes[beta.0].data;

        let eps = 1e-5f32;
        let inv_d = 1.0 / cols as f32;
        let mut x_hat = vec![0.0f32; rows * cols];
        let mut inv_std = vec![0.0f32; rows];
        let mut out = vec![0.0f32; rows * cols];

        for r in 0..rows {
            let off = r * cols;
            let mut mean = 0.0f32;
            for c in 0..cols {
                mean += x_data[off + c];
            }
            mean *= inv_d;

            let mut var = 0.0f32;
            for c in 0..cols {
                let diff = x_data[off + c] - mean;
                var += diff * diff;
            }
            var *= inv_d;
            let istd = 1.0 / (var + eps).sqrt();
            inv_std[r] = istd;

            for c in 0..cols {
                let xh = (x_data[off + c] - mean) * istd;
                x_hat[off + c] = xh;
                out[off + c] = xh * g_data[c] + b_data[c];
            }
        }

        self.push(
            rows,
            cols,
            out,
            Op::LayerNorm {
                x,
                gamma,
                beta,
                x_hat,
                inv_std,
            },
        )
    }

    /// GPT-2 标准平滑 GELU 激活函数（解决 ReLU 负区彻底死亡问题）
    fn gelu(&mut self, x: TensorId) -> TensorId {
        let (rows, cols) = self.shape(x);
        let x_data = &self.nodes[x.0].data;
        let k = (2.0f32 / PI).sqrt();
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..out.len() {
            let v = x_data[i];
            let u = k * (v + 0.044715 * v * v * v);
            out[i] = 0.5 * v * (1.0 + u.tanh());
        }
        self.push(rows, cols, out, Op::Gelu(x))
    }

    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ra, ca) = self.shape(a);
        let (rb, cb) = self.shape(b);
        assert_eq!(ca, cb);
        assert!(ra == rb || rb == 1);
        let a_data = &self.nodes[a.0].data;
        let b_data = &self.nodes[b.0].data;
        let mut out = vec![0.0f32; ra * ca];
        for r in 0..ra {
            let br = if rb == 1 { 0 } else { r };
            for c in 0..ca {
                out[r * ca + c] = a_data[r * ca + c] + b_data[br * cb + c];
            }
        }
        self.push(ra, ca, out, Op::Add(a, b))
    }

    fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (m, k) = self.shape(a);
        let (kb, n) = self.shape(b);
        assert_eq!(k, kb);
        let a_data = &self.nodes[a.0].data;
        let b_data = &self.nodes[b.0].data;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_ip = a_data[i * k + p];
                let b_row = p * n;
                let out_row = i * n;
                for j in 0..n {
                    out[out_row + j] += a_ip * b_data[b_row + j];
                }
            }
        }
        self.push(m, n, out, Op::MatMul(a, b))
    }

    fn causal_multi_head_attn(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
    ) -> TensorId {
        let (rows, d_model) = self.shape(q);
        assert_eq!(rows, batch_size * seq_len);
        assert_eq!(d_model % n_head, 0);
        let d_head = d_model / n_head;
        let scale = 1.0 / (d_head as f32).sqrt();

        let q_data = &self.nodes[q.0].data;
        let k_data = &self.nodes[k.0].data;
        let v_data = &self.nodes[v.0].data;
        let mut attn_weights = vec![0.0f32; batch_size * n_head * seq_len * seq_len];
        let mut out = vec![0.0f32; rows * d_model];
        let mut row_scores = vec![0.0f32; seq_len];

        for b in 0..batch_size {
            for h in 0..n_head {
                let h_col = h * d_head;
                let w_base = (b * n_head + h) * seq_len * seq_len;
                for i in 0..seq_len {
                    let qi_off = (b * seq_len + i) * d_model + h_col;
                    let mut max_s = f32::NEG_INFINITY;
                    for j in 0..=i {
                        let kj_off = (b * seq_len + j) * d_model + h_col;
                        let mut dot = 0.0f32;
                        for d in 0..d_head {
                            dot += q_data[qi_off + d] * k_data[kj_off + d];
                        }
                        let s = dot * scale;
                        row_scores[j] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut sum_exp = 0.0f32;
                    for j in 0..=i {
                        let e = (row_scores[j] - max_s).exp();
                        attn_weights[w_base + i * seq_len + j] = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0 / sum_exp;
                    for j in 0..=i {
                        attn_weights[w_base + i * seq_len + j] *= inv_sum;
                    }
                    let out_off = (b * seq_len + i) * d_model + h_col;
                    for j in 0..=i {
                        let w = attn_weights[w_base + i * seq_len + j];
                        let vj_off = (b * seq_len + j) * d_model + h_col;
                        for d in 0..d_head {
                            out[out_off + d] += w * v_data[vj_off + d];
                        }
                    }
                }
            }
        }

        self.push(
            rows,
            d_model,
            out,
            Op::CausalMultiHeadAttn {
                q,
                k,
                v,
                batch_size,
                seq_len,
                n_head,
                attn_weights,
            },
        )
    }

    fn cross_entropy(&mut self, logits: TensorId, targets: &[usize]) -> TensorId {
        let (n, v) = self.shape(logits);
        assert_eq!(n, targets.len());
        let z = &self.nodes[logits.0].data;
        let mut total = 0.0f32;
        for (row, &y) in targets.iter().enumerate() {
            let probs = softmax(&z[row * v..(row + 1) * v]);
            total += -(probs[y].max(1e-12)).ln();
        }
        self.push(
            1,
            1,
            vec![total / n as f32],
            Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
            },
        )
    }

    fn backward(&mut self, loss_id: TensorId) {
        assert_eq!(self.shape(loss_id), (1, 1));
        self.nodes[loss_id.0].grad[0] = 1.0;

        for id in (0..=loss_id.0).rev() {
            let d_out = self.nodes[id].grad.clone();
            if d_out.iter().all(|&g| g == 0.0) {
                continue;
            }

            match &self.nodes[id].op {
                Op::Leaf => {}

                Op::Add(a, b) => {
                    let (a, b) = (*a, *b);
                    let (ra, ca) = self.shape(a);
                    let (rb, cb) = self.shape(b);
                    for r in 0..ra {
                        let br = if rb == 1 { 0 } else { r };
                        for c in 0..ca {
                            let g = d_out[r * ca + c];
                            self.nodes[a.0].grad[r * ca + c] += g;
                            self.nodes[b.0].grad[br * cb + c] += g;
                        }
                    }
                }

                Op::MatMul(a, b) => {
                    let (a, b) = (*a, *b);
                    let (m, k) = self.shape(a);
                    let (_, n) = self.shape(b);
                    let a_data = self.nodes[a.0].data.clone();
                    let b_data = self.nodes[b.0].data.clone();
                    for i in 0..m {
                        for p in 0..k {
                            let mut acc = 0.0f32;
                            let a_ip = a_data[i * k + p];
                            for j in 0..n {
                                let g = d_out[i * n + j];
                                acc += g * b_data[p * n + j];
                                self.nodes[b.0].grad[p * n + j] += a_ip * g;
                            }
                            self.nodes[a.0].grad[i * k + p] += acc;
                        }
                    }
                }

                Op::Gelu(x) => {
                    let x_id = *x;
                    let x_data = self.nodes[x_id.0].data.clone();
                    let k = (2.0f32 / PI).sqrt();
                    for i in 0..d_out.len() {
                        let v = x_data[i];
                        let u = k * (v + 0.044715 * v * v * v);
                        let t = u.tanh();
                        let du = k * (1.0 + 3.0 * 0.044715 * v * v);
                        let d_gelu = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * du;
                        self.nodes[x_id.0].grad[i] += d_out[i] * d_gelu;
                    }
                }

                Op::LayerNorm {
                    x,
                    gamma,
                    beta,
                    x_hat,
                    inv_std,
                } => {
                    let (x, gamma, beta) = (*x, *gamma, *beta);
                    let x_hat = x_hat.clone();
                    let inv_std = inv_std.clone();
                    let (rows, cols) = self.shape(x);
                    let g_data = self.nodes[gamma.0].data.clone();
                    let inv_d = 1.0 / cols as f32;

                    for r in 0..rows {
                        let off = r * cols;
                        let mut mean_g = 0.0f32;
                        let mut mean_g_xh = 0.0f32;
                        for c in 0..cols {
                            let dy = d_out[off + c];
                            let xh = x_hat[off + c];
                            self.nodes[gamma.0].grad[c] += dy * xh;
                            self.nodes[beta.0].grad[c] += dy;
                            let g = dy * g_data[c];
                            mean_g += g;
                            mean_g_xh += g * xh;
                        }
                        mean_g *= inv_d;
                        mean_g_xh *= inv_d;
                        let istd = inv_std[r];
                        for c in 0..cols {
                            let g = d_out[off + c] * g_data[c];
                            let xh = x_hat[off + c];
                            self.nodes[x.0].grad[off + c] += istd * (g - mean_g - xh * mean_g_xh);
                        }
                    }
                }

                Op::EmbedPos {
                    tok_emb,
                    pos_emb,
                    seqs,
                } => {
                    let (tok_emb, pos_emb) = (*tok_emb, *pos_emb);
                    let seqs = seqs.clone();
                    let (_, d) = self.shape(tok_emb);
                    let t_len = seqs[0].len();
                    for (b, seq) in seqs.iter().enumerate() {
                        for (t, &tok_id) in seq.iter().enumerate() {
                            let g_off = (b * t_len + t) * d;
                            for c in 0..d {
                                let g = d_out[g_off + c];
                                self.nodes[tok_emb.0].grad[tok_id * d + c] += g;
                                self.nodes[pos_emb.0].grad[t * d + c] += g;
                            }
                        }
                    }
                }

                Op::CausalMultiHeadAttn {
                    q,
                    k,
                    v,
                    batch_size,
                    seq_len,
                    n_head,
                    attn_weights,
                } => {
                    let (q, k, v) = (*q, *k, *v);
                    let (batch_size, seq_len, n_head) = (*batch_size, *seq_len, *n_head);
                    let weights = attn_weights.clone();
                    let (_, d_model) = self.shape(q);
                    let d_head = d_model / n_head;
                    let scale = 1.0 / (d_head as f32).sqrt();

                    let q_data = self.nodes[q.0].data.clone();
                    let k_data = self.nodes[k.0].data.clone();
                    let v_data = self.nodes[v.0].data.clone();
                    let mut dq = vec![0.0f32; q_data.len()];
                    let mut dk = vec![0.0f32; k_data.len()];
                    let mut dv = vec![0.0f32; v_data.len()];
                    let mut dw_row = vec![0.0f32; seq_len];

                    for b in 0..batch_size {
                        for h in 0..n_head {
                            let h_col = h * d_head;
                            let w_base = (b * n_head + h) * seq_len * seq_len;
                            for i in 0..seq_len {
                                let off = (b * seq_len + i) * d_model + h_col;
                                let mut w_dot_dw = 0.0f32;
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let vj_off = (b * seq_len + j) * d_model + h_col;
                                    let mut dw_ij = 0.0f32;
                                    for d in 0..d_head {
                                        let g = d_out[off + d];
                                        dv[vj_off + d] += w_ij * g;
                                        dw_ij += g * v_data[vj_off + d];
                                    }
                                    dw_row[j] = dw_ij;
                                    w_dot_dw += w_ij * dw_ij;
                                }
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let ds = w_ij * (dw_row[j] - w_dot_dw) * scale;
                                    let kj_off = (b * seq_len + j) * d_model + h_col;
                                    for d in 0..d_head {
                                        dq[off + d] += ds * k_data[kj_off + d];
                                        dk[kj_off + d] += ds * q_data[off + d];
                                    }
                                }
                            }
                        }
                    }

                    for idx in 0..dq.len() {
                        self.nodes[q.0].grad[idx] += dq[idx];
                        self.nodes[k.0].grad[idx] += dk[idx];
                        self.nodes[v.0].grad[idx] += dv[idx];
                    }
                }

                Op::CrossEntropy { logits, targets } => {
                    let logits_id = *logits;
                    let targets = targets.clone();
                    let (n, v) = self.shape(logits_id);
                    let scale = d_out[0] / n as f32;
                    let z = self.nodes[logits_id.0].data.clone();
                    for (row, &y) in targets.iter().enumerate() {
                        let probs = softmax(&z[row * v..(row + 1) * v]);
                        let row_off = row * v;
                        for j in 0..v {
                            let one_hot = if j == y { 1.0 } else { 0.0 };
                            self.nodes[logits_id.0].grad[row_off + j] += (probs[j] - one_hot) * scale;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2. 参数与 AdamW 优化器（支持全局梯度范数裁剪）
// ============================================================================

struct Param {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    decay: bool,
}

impl Param {
    fn randn(rows: usize, cols: usize, std: f32, rng: &mut Rng, decay: bool) -> Self {
        let len = rows * cols;
        Self {
            rows,
            cols,
            data: (0..len).map(|_| rng.normal(std)).collect(),
            m: vec![0.0; len],
            v: vec![0.0; len],
            decay,
        }
    }

    fn ones(rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            rows,
            cols,
            data: vec![1.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
            decay: false,
        }
    }

    fn zeros(rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            rows,
            cols,
            data: vec![0.0; len],
            m: vec![0.0; len],
            v: vec![0.0; len],
            decay: false,
        }
    }

    fn adamw_step(&mut self, grad: &[f32], grad_scale: f32, lr: f32, wd: f32, step: usize) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - b1.powi(step as i32);
        let bc2 = 1.0 - b2.powi(step as i32);
        for i in 0..self.data.len() {
            let g = grad[i] * grad_scale;
            if self.decay {
                self.data[i] -= lr * wd * self.data[i];
            }
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// ============================================================================
// 3. 完整 Pre-Norm Transformer Block 与多层 Mini-GPT
// ============================================================================

struct TransformerBlock {
    ln1_gamma: Param, // (1, D)
    ln1_beta: Param,  // (1, D)
    wq: Param,        // (D, D)
    wk: Param,        // (D, D)
    wv: Param,        // (D, D)
    wo: Param,        // (D, D)
    ln2_gamma: Param, // (1, D)
    ln2_beta: Param,  // (1, D)
    ffn_w1: Param,    // (D, 4D)
    ffn_b1: Param,    // (1, 4D)
    ffn_w2: Param,    // (4D, D)
    ffn_b2: Param,    // (1, D)
}

struct BlockIds {
    ln1_gamma: TensorId,
    ln1_beta: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: TensorId,
    wo: TensorId,
    ln2_gamma: TensorId,
    ln2_beta: TensorId,
    ffn_w1: TensorId,
    ffn_b1: TensorId,
    ffn_w2: TensorId,
    ffn_b2: TensorId,
}

impl TransformerBlock {
    fn new(d_model: usize, d_ff: usize, n_layer: usize, rng: &mut Rng) -> Self {
        let std = (1.0 / d_model as f32).sqrt();
        // GPT-2 经典残差投影缩放：1 / sqrt(2 * n_layer)，防止多层叠加后方差爆炸
        let res_std = std / (2.0 * n_layer as f32).sqrt();
        Self {
            ln1_gamma: Param::ones(1, d_model),
            ln1_beta: Param::zeros(1, d_model),
            wq: Param::randn(d_model, d_model, std, rng, true),
            wk: Param::randn(d_model, d_model, std, rng, true),
            wv: Param::randn(d_model, d_model, std, rng, true),
            wo: Param::randn(d_model, d_model, res_std, rng, true),
            ln2_gamma: Param::ones(1, d_model),
            ln2_beta: Param::zeros(1, d_model),
            ffn_w1: Param::randn(d_model, d_ff, std, rng, true),
            ffn_b1: Param::zeros(1, d_ff),
            ffn_w2: Param::randn(d_ff, d_model, res_std, rng, true),
            ffn_b2: Param::zeros(1, d_model),
        }
    }

    fn num_params(&self) -> usize {
        self.ln1_gamma.data.len()
            + self.ln1_beta.data.len()
            + self.wq.data.len()
            + self.wk.data.len()
            + self.wv.data.len()
            + self.wo.data.len()
            + self.ln2_gamma.data.len()
            + self.ln2_beta.data.len()
            + self.ffn_w1.data.len()
            + self.ffn_b1.data.len()
            + self.ffn_w2.data.len()
            + self.ffn_b2.data.len()
    }

    /// Pre-Norm Transformer Block 标准前向：
    ///   1) x = x + CausalMultiHeadAttn(LayerNorm1(x))
    ///   2) x = x + FFN_GELU(LayerNorm2(x))
    fn forward(
        &self,
        tape: &mut Tape,
        x: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
    ) -> (TensorId, BlockIds) {
        let ln1_gamma = tape.param(&self.ln1_gamma);
        let ln1_beta = tape.param(&self.ln1_beta);
        let wq = tape.param(&self.wq);
        let wk = tape.param(&self.wk);
        let wv = tape.param(&self.wv);
        let wo = tape.param(&self.wo);

        let ln1 = tape.layer_norm(x, ln1_gamma, ln1_beta);
        let q = tape.matmul(ln1, wq);
        let k = tape.matmul(ln1, wk);
        let v = tape.matmul(ln1, wv);
        let attn = tape.causal_multi_head_attn(q, k, v, batch_size, seq_len, n_head);
        let attn_out = tape.matmul(attn, wo);
        let x1 = tape.add(x, attn_out);

        let ln2_gamma = tape.param(&self.ln2_gamma);
        let ln2_beta = tape.param(&self.ln2_beta);
        let ffn_w1 = tape.param(&self.ffn_w1);
        let ffn_b1 = tape.param(&self.ffn_b1);
        let ffn_w2 = tape.param(&self.ffn_w2);
        let ffn_b2 = tape.param(&self.ffn_b2);

        let ln2 = tape.layer_norm(x1, ln2_gamma, ln2_beta);
        let up_raw = tape.matmul(ln2, ffn_w1);
        let up = tape.add(up_raw, ffn_b1);
        let act = tape.gelu(up);
        let down_raw = tape.matmul(act, ffn_w2);
        let down = tape.add(down_raw, ffn_b2);
        let x2 = tape.add(x1, down);

        (
            x2,
            BlockIds {
                ln1_gamma,
                ln1_beta,
                wq,
                wk,
                wv,
                wo,
                ln2_gamma,
                ln2_beta,
                ffn_w1,
                ffn_b1,
                ffn_w2,
                ffn_b2,
            },
        )
    }

    fn param_grad_pairs<'a>(&'a mut self, ids: &BlockIds) -> [(&'a mut Param, TensorId); 12] {
        [
            (&mut self.ln1_gamma, ids.ln1_gamma),
            (&mut self.ln1_beta, ids.ln1_beta),
            (&mut self.wq, ids.wq),
            (&mut self.wk, ids.wk),
            (&mut self.wv, ids.wv),
            (&mut self.wo, ids.wo),
            (&mut self.ln2_gamma, ids.ln2_gamma),
            (&mut self.ln2_beta, ids.ln2_beta),
            (&mut self.ffn_w1, ids.ffn_w1),
            (&mut self.ffn_b1, ids.ffn_b1),
            (&mut self.ffn_w2, ids.ffn_w2),
            (&mut self.ffn_b2, ids.ffn_b2),
        ]
    }
}

struct MiniGpt {
    vocab_size: usize,
    block_size: usize,
    n_head: usize,
    tok_emb: Param,
    pos_emb: Param,
    blocks: Vec<TransformerBlock>,
    ln_f_gamma: Param,
    ln_f_beta: Param,
    w_lm: Param,
    b_lm: Param,
}

struct GptForwardIds {
    tok_emb: TensorId,
    pos_emb: TensorId,
    blocks: Vec<BlockIds>,
    ln_f_gamma: TensorId,
    ln_f_beta: TensorId,
    w_lm: TensorId,
    b_lm: TensorId,
    logits: TensorId,
}

impl MiniGpt {
    fn new(
        vocab_size: usize,
        block_size: usize,
        d_model: usize,
        n_head: usize,
        n_layer: usize,
        rng: &mut Rng,
    ) -> Self {
        let d_ff = d_model * 4;
        let std = (1.0 / d_model as f32).sqrt();
        let blocks = (0..n_layer)
            .map(|_| TransformerBlock::new(d_model, d_ff, n_layer, rng))
            .collect();
        Self {
            vocab_size,
            block_size,
            n_head,
            tok_emb: Param::randn(vocab_size, d_model, 0.2, rng, true),
            pos_emb: Param::randn(block_size, d_model, 0.1, rng, true),
            blocks,
            ln_f_gamma: Param::ones(1, d_model),
            ln_f_beta: Param::zeros(1, d_model),
            w_lm: Param::randn(d_model, vocab_size, std, rng, true),
            b_lm: Param::zeros(1, vocab_size),
        }
    }

    fn num_params(&self) -> usize {
        let block_sum: usize = self.blocks.iter().map(|b| b.num_params()).sum();
        self.tok_emb.data.len()
            + self.pos_emb.data.len()
            + block_sum
            + self.ln_f_gamma.data.len()
            + self.ln_f_beta.data.len()
            + self.w_lm.data.len()
            + self.b_lm.data.len()
    }

    fn forward(&self, tape: &mut Tape, seqs: &[Vec<usize>]) -> GptForwardIds {
        let batch_size = seqs.len();
        let seq_len = seqs[0].len();
        let tok_emb = tape.param(&self.tok_emb);
        let pos_emb = tape.param(&self.pos_emb);
        let mut x = tape.embed_pos(tok_emb, pos_emb, seqs);

        let mut block_ids = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (nx, bids) = blk.forward(tape, x, batch_size, seq_len, self.n_head);
            x = nx;
            block_ids.push(bids);
        }

        let ln_f_gamma = tape.param(&self.ln_f_gamma);
        let ln_f_beta = tape.param(&self.ln_f_beta);
        let x_norm = tape.layer_norm(x, ln_f_gamma, ln_f_beta);

        let w_lm = tape.param(&self.w_lm);
        let b_lm = tape.param(&self.b_lm);
        let logits_raw = tape.matmul(x_norm, w_lm);
        let logits = tape.add(logits_raw, b_lm);

        GptForwardIds {
            tok_emb,
            pos_emb,
            blocks: block_ids,
            ln_f_gamma,
            ln_f_beta,
            w_lm,
            b_lm,
            logits,
        }
    }

    fn train_step(
        &mut self,
        seqs: &[Vec<usize>],
        targets: &[usize],
        lr: f32,
        step: usize,
    ) -> (f32, f32) {
        let mut tape = Tape::new();
        let ids = self.forward(&mut tape, seqs);
        let loss_id = tape.cross_entropy(ids.logits, targets);
        let loss = tape.data(loss_id)[0];

        tape.backward(loss_id);

        // 全局梯度 L2 范数裁剪（max_norm = 1.0），防止深层 Transformer 梯度尖峰
        let mut all_ids = vec![ids.tok_emb, ids.pos_emb, ids.ln_f_gamma, ids.ln_f_beta, ids.w_lm, ids.b_lm];
        for b in &ids.blocks {
            all_ids.extend_from_slice(&[
                b.ln1_gamma, b.ln1_beta, b.wq, b.wk, b.wv, b.wo,
                b.ln2_gamma, b.ln2_beta, b.ffn_w1, b.ffn_b1, b.ffn_w2, b.ffn_b2,
            ]);
        }
        let mut norm_sq = 0.0f32;
        for &tid in &all_ids {
            for &g in tape.grad(tid) {
                norm_sq += g * g;
            }
        }
        let grad_norm = norm_sq.sqrt();
        let max_norm = 1.0f32;
        let clip_scale = if grad_norm > max_norm { max_norm / grad_norm } else { 1.0 };

        let wd = 0.01;
        self.tok_emb.adamw_step(tape.grad(ids.tok_emb), clip_scale, lr, wd, step);
        self.pos_emb.adamw_step(tape.grad(ids.pos_emb), clip_scale, lr, wd, step);
        for (blk, bids) in self.blocks.iter_mut().zip(ids.blocks.iter()) {
            for (param, tid) in blk.param_grad_pairs(bids) {
                param.adamw_step(tape.grad(tid), clip_scale, lr, wd, step);
            }
        }
        self.ln_f_gamma.adamw_step(tape.grad(ids.ln_f_gamma), clip_scale, lr, wd, step);
        self.ln_f_beta.adamw_step(tape.grad(ids.ln_f_beta), clip_scale, lr, wd, step);
        self.w_lm.adamw_step(tape.grad(ids.w_lm), clip_scale, lr, wd, step);
        self.b_lm.adamw_step(tape.grad(ids.b_lm), clip_scale, lr, wd, step);

        (loss, grad_norm)
    }

    fn generate(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> Vec<usize> {
        let mut out = prompt_ids.to_vec();
        for _ in 0..max_new_tokens {
            let start = out.len().saturating_sub(self.block_size);
            let window = vec![out[start..].to_vec()];
            let mut tape = Tape::new();
            let ids = self.forward(&mut tape, &window);
            let t_len = window[0].len();
            let off = (t_len - 1) * self.vocab_size;
            let last_logits = &tape.data(ids.logits)[off..off + self.vocab_size];
            out.push(sample_from_logits(last_logits, temperature, rng));
        }
        out
    }
}

/// 现代大模型标配：线性 Warmup + 余弦退火学习率调度
fn cosine_lr(step: usize, total_steps: usize, warmup_steps: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step <= warmup_steps {
        max_lr * (step as f32 / warmup_steps as f32)
    } else {
        let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps).max(1) as f32;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (PI * progress).cos())
    }
}

// ============================================================================
// 4. 主程序：从零训练 2 层 Pre-Norm Mini-GPT，通关全部唐诗续写挑战！
// ============================================================================

fn main() {
    let text = load_poems();
    let tok = Tokenizer::from_text(&text);
    let data = tok.encode(&text);

    let block_size = 16;
    let d_model = 32;
    let n_head = 4;
    let n_layer = 2;
    let batch_size = 16;
    let steps = 1200;
    let warmup_steps = 80;
    let max_lr = 0.012;
    let min_lr = 0.0015;

    println!("=== Step 04（终极关）: 纯 Rust 完整 Pre-Norm Transformer Mini-GPT ===");
    println!(
        "语料字符数: {}, 词表大小 V: {}, 初始理论 Loss = ln({}) = {:.4}",
        data.len(),
        tok.vocab_size(),
        tok.vocab_size(),
        (tok.vocab_size() as f32).ln()
    );

    let mut rng = Rng::new(42);
    let mut model = MiniGpt::new(tok.vocab_size(), block_size, d_model, n_head, n_layer, &mut rng);
    println!(
        "模型结构: {}层 Pre-Norm Transformer (LN + {}头因果自注意力 + FFN_GELU 32→128→32 + 残差) | 总参数量: {}",
        n_layer,
        n_head,
        model.num_params()
    );

    let max_start = data.len() - block_size - 1;
    let t0 = Instant::now();
    for step in 1..=steps {
        let mut seqs = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size * block_size);
        for _ in 0..batch_size {
            let s = (rng.next_u64() as usize) % max_start;
            seqs.push(data[s..s + block_size].to_vec());
            targets.extend_from_slice(&data[s + 1..s + 1 + block_size]);
        }
        let lr = cosine_lr(step, steps, warmup_steps, max_lr, min_lr);
        let (loss, gnorm) = model.train_step(&seqs, &targets, lr, step);
        if step == 1 || step % 200 == 0 {
            println!(
                "step {:4}/{} | lr = {:.4} | grad_norm = {:.2} | loss = {:.4}",
                step, steps, lr, gnorm, loss
            );
        }
    }
    println!("训练完成！耗时 {:.2}s", t0.elapsed().as_secs_f32());

    println!("\n=== 终极通关续写测试（同时攻克 Step 02 串台 & Step 03「处处处」复读！）===");
    for prompt in [
        "床前明月光，疑是",
        "春眠不觉晓，",
        "白日依山尽，黄河",
        "飞流直下三千尺，疑是",
        "红豆生南国，",
    ] {
        let prompt_ids = tok.encode(prompt);
        let generated = model.generate(&prompt_ids, 20, 0.25, &mut rng);
        println!("输入「{}」 -> {}", prompt, tok.decode(&generated).replace('\n', " / "));
    }
}

// ============================================================================
// 5. 单元测试：LayerNorm 校验 + 全网络数值梯度检验 + 深层因果无泄露检验
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_loss(model: &MiniGpt, seqs: &[Vec<usize>], targets: &[usize]) -> f32 {
        let mut tape = Tape::new();
        let ids = model.forward(&mut tape, seqs);
        let loss_id = tape.cross_entropy(ids.logits, targets);
        tape.data(loss_id)[0]
    }

    #[test]
    fn test_layernorm_zero_mean_unit_var() {
        let mut tape = Tape::new();
        let x = tape.push(1, 4, vec![1.0, 2.0, 3.0, 10.0], Op::Leaf);
        let g = tape.param(&Param::ones(1, 4));
        let b = tape.param(&Param::zeros(1, 4));
        let y = tape.layer_norm(x, g, b);
        let yd = tape.data(y);
        let mean: f32 = yd.iter().sum::<f32>() / 4.0;
        let var: f32 = yd.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "mean = {}", mean);
        assert!((var - 1.0).abs() < 1e-4, "var = {}", var);
    }

    #[test]
    fn test_minigpt_gradient_check() {
        let mut rng = Rng::new(2025);
        let mut model = MiniGpt::new(5, 4, 8, 2, 2, &mut rng);
        let seqs = vec![vec![0, 1, 2, 3], vec![4, 2, 1, 0]];
        let targets = vec![1, 2, 3, 4, 2, 1, 0, 3];

        let mut tape = Tape::new();
        let ids = model.forward(&mut tape, &seqs);
        let loss_id = tape.cross_entropy(ids.logits, &targets);
        tape.backward(loss_id);

        // 校验全部 9 类代表参数（含两层 Block 的 LayerNorm / Attention / GELU FFN）
        let g_tok = tape.grad(ids.tok_emb).to_vec();
        let g_pos = tape.grad(ids.pos_emb).to_vec();
        let g_ln1g = tape.grad(ids.blocks[0].ln1_gamma).to_vec();
        let g_ln1b = tape.grad(ids.blocks[0].ln1_beta).to_vec();
        let g_wq0 = tape.grad(ids.blocks[0].wq).to_vec();
        let g_ffn1 = tape.grad(ids.blocks[0].ffn_w1).to_vec();
        let g_ffn2 = tape.grad(ids.blocks[1].ffn_w2).to_vec();
        let g_lnfg = tape.grad(ids.ln_f_gamma).to_vec();
        let g_wlm = tape.grad(ids.w_lm).to_vec();

        let eps = 2e-3f32;
        let check = |name: &str, analytic: &[f32], getter: fn(&mut MiniGpt) -> &mut [f32], m: &mut MiniGpt| {
            let mut max_rel = 0.0f32;
            for i in 0..analytic.len() {
                let orig = getter(m)[i];
                getter(m)[i] = orig + eps;
                let lp = compute_loss(m, &seqs, &targets);
                getter(m)[i] = orig - eps;
                let lm = compute_loss(m, &seqs, &targets);
                getter(m)[i] = orig;
                let num = (lp - lm) / (2.0 * eps);
                let rel = (num - analytic[i]).abs() / (num.abs() + analytic[i].abs()).max(1.0);
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            assert!(max_rel < 3e-3, "MiniGPT 参数 {} 相对梯度误差过大: {}", name, max_rel);
        };

        check("tok_emb", &g_tok, |m| &mut m.tok_emb.data, &mut model);
        check("pos_emb", &g_pos, |m| &mut m.pos_emb.data, &mut model);
        check("blk0.ln1_gamma", &g_ln1g, |m| &mut m.blocks[0].ln1_gamma.data, &mut model);
        check("blk0.ln1_beta", &g_ln1b, |m| &mut m.blocks[0].ln1_beta.data, &mut model);
        check("blk0.wq", &g_wq0, |m| &mut m.blocks[0].wq.data, &mut model);
        check("blk0.ffn_w1(GELU)", &g_ffn1, |m| &mut m.blocks[0].ffn_w1.data, &mut model);
        check("blk1.ffn_w2", &g_ffn2, |m| &mut m.blocks[1].ffn_w2.data, &mut model);
        check("ln_f_gamma", &g_lnfg, |m| &mut m.ln_f_gamma.data, &mut model);
        check("w_lm", &g_wlm, |m| &mut m.w_lm.data, &mut model);
    }

    #[test]
    fn test_minigpt_causal_no_future_leak() {
        let mut rng = Rng::new(123);
        let model = MiniGpt::new(8, 6, 8, 2, 2, &mut rng);
        let seq_a = vec![vec![1, 2, 3, 4, 5, 6]];
        let seq_b = vec![vec![1, 2, 3, 7, 0, 1]];

        let mut ta = Tape::new();
        let ia = model.forward(&mut ta, &seq_a);
        let mut tb = Tape::new();
        let ib = model.forward(&mut tb, &seq_b);

        for pos in 0..3 {
            for v in 0..model.vocab_size {
                let d = (ta.data(ia.logits)[pos * model.vocab_size + v]
                    - tb.data(ib.logits)[pos * model.vocab_size + v])
                .abs();
                assert!(d < 1e-5, "2层 Transformer 位置 {} 出现未来泄露: {}", pos, d);
            }
        }
    }
}
