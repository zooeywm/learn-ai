//! Step 03: Embedding + 位置编码 + 因果多头自注意力 (Causal Multi-Head Self-Attention)
//!
//! 为什么要发明 Self-Attention（自注意力）？
//!   在 Step 02 里，我们把最近 3 个字硬拼接成固定向量送进 MLP。
//!   这导致模型有一个致命短板——视野被锁死在 3 个字：
//!     输入「床前明月光，疑是」时，它只能看到「光，疑是」里的最后 3 字「，疑是」，
//!     完全看不到前面的「明月」，于是直接串台成《望庐山瀑布》的「银河落九天」！
//!
//! 本关核心突破（Transformer / GPT 的心脏）：
//!   1. Token + Position Embedding：不仅记住「是什么字」，还记住「在第几个位置」；
//!   2. Q / K / V 线性投影：每个位置都生成 Query（我在找什么）、Key（我是谁）、Value（我能提供什么信息）；
//!   3. 缩放点积注意力：scores = (Q @ K^T) / sqrt(d_head)，动态计算当前字与历史每个字的相关度；
//!   4. Causal Mask（因果下三角掩码）：强行把未来位置 (j > i) 设为 -∞，Softmax 后权重归 0，
//!      保证「绝不偷看答案」，同时让一整段长度为 T 的序列可以在一次前向中并行训练全部 T 个预测！
//!   5. Multi-Head（多头）+ Residual（残差连接）：多个头分别关注不同关系，残差把原始字信息直通下游。

use std::time::Instant;

use mini_core::{load_poems, sample_from_logits, softmax, Rng, Tokenizer};

// ============================================================================
// 1. 升级版 Arena-Tape 自动求导引擎（新增 EmbedPos 与 CausalMultiHeadAttn 算子）
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorId(usize);

enum Op {
    Leaf,
    /// 同形状相加 (R, C) + (R, C) 或偏置行广播 (R, C) + (1, C)
    Add(TensorId, TensorId),
    /// 矩阵乘法 (M, K) @ (K, N) -> (M, N)
    MatMul(TensorId, TensorId),
    /// 词向量 + 位置编码：tok_emb(V, D) + pos_emb(T_max, D) + seqs(B × T) -> (B*T, D)
    EmbedPos {
        tok_emb: TensorId,
        pos_emb: TensorId,
        seqs: Vec<Vec<usize>>,
    },
    /// 因果多头自注意力：Q(B*T, D), K(B*T, D), V(B*T, D) -> Context(B*T, D)
    CausalMultiHeadAttn {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_head: usize,
        /// 前向缓存的注意力权重，形状 (B * n_head * T * T)，供反向传播和热力图观察复用
        attn_weights: Vec<f32>,
    },
    /// 融合 Softmax + 交叉熵：logits(N, V) + targets(N) -> (1, 1)
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

    fn attn_weights(&self, id: TensorId) -> &[f32] {
        match &self.nodes[id.0].op {
            Op::CausalMultiHeadAttn { attn_weights, .. } => attn_weights,
            _ => panic!("节点 {:?} 不是 CausalMultiHeadAttn", id),
        }
    }

    /// 词向量查表并叠加绝对位置编码：h0[b, t] = tok_emb[seq[b][t]] + pos_emb[t]
    fn embed_pos(&mut self, tok_emb: TensorId, pos_emb: TensorId, seqs: &[Vec<usize>]) -> TensorId {
        let (_v, d) = self.shape(tok_emb);
        let (max_t, pd) = self.shape(pos_emb);
        assert_eq!(d, pd, "tok_emb 与 pos_emb 维度必须一致");
        let b_cnt = seqs.len();
        let t_len = seqs[0].len();
        assert!(t_len <= max_t, "序列长度 {} 超出位置编码上限 {}", t_len, max_t);

        let tok_data = &self.nodes[tok_emb.0].data;
        let pos_data = &self.nodes[pos_emb.0].data;
        let mut out = vec![0.0f32; b_cnt * t_len * d];

        for (b, seq) in seqs.iter().enumerate() {
            assert_eq!(seq.len(), t_len);
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

    /// 矩阵加法：(R, C) + (R, C) 或 (R, C) + (1, C)
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

    /// 矩阵乘法：(M, K) @ (K, N) -> (M, N)
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

    /// 因果多头自注意力（本关核心算子！）
    /// 输入 Q, K, V 形状均为 (B * T, D)，按列切分为 n_head 个头，每头维度 d_head = D / n_head。
    /// 对每个 batch b、每个头 h、每个查询位置 i：
    ///   scores[j] = (Q[b,i,h] · K[b,j,h]) / sqrt(d_head)   (仅 j <= i；未来 j > i 掩码归 0)
    ///   weights   = softmax(scores[0..=i])
    ///   ctx[b,i,h]= sum_{j=0..=i} weights[j] * V[b,j,h]
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
        assert_eq!(self.shape(k), (rows, d_model));
        assert_eq!(self.shape(v), (rows, d_model));
        assert_eq!(d_model % n_head, 0, "D_MODEL 必须能被 N_HEAD 整除");

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

                    // 1) 缩放点积：只看 0..=i（j > i 自动被因果掩码屏蔽）
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

                    // 2) 数值稳定 Softmax（仅在因果下三角 0..=i 上归一化）
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

                    // 3) 按注意力权重加权汇聚历史 Value
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

    /// 融合 Softmax + 交叉熵 Loss
    fn cross_entropy(&mut self, logits: TensorId, targets: &[usize]) -> TensorId {
        let (n, v) = self.shape(logits);
        assert_eq!(n, targets.len());
        let z = &self.nodes[logits.0].data;
        let mut total_loss = 0.0f32;
        for (row, &y) in targets.iter().enumerate() {
            let probs = softmax(&z[row * v..(row + 1) * v]);
            total_loss += -(probs[y].max(1e-12)).ln();
        }
        self.push(
            1,
            1,
            vec![total_loss / n as f32],
            Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
            },
        )
    }

    /// 倒序遍历 Tape，自动执行精确链式法则反传
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

                    // dA += dOut @ B^T
                    for i in 0..m {
                        for p in 0..k {
                            let mut acc = 0.0f32;
                            for j in 0..n {
                                acc += d_out[i * n + j] * b_data[p * n + j];
                            }
                            self.nodes[a.0].grad[i * k + p] += acc;
                        }
                    }
                    // dB += A^T @ dOut
                    for i in 0..m {
                        for p in 0..k {
                            let a_ip = a_data[i * k + p];
                            for j in 0..n {
                                self.nodes[b.0].grad[p * n + j] += a_ip * d_out[i * n + j];
                            }
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
                            let tok_off = tok_id * d;
                            let pos_off = t * d;
                            for c in 0..d {
                                let g = d_out[g_off + c];
                                self.nodes[tok_emb.0].grad[tok_off + c] += g;
                                self.nodes[pos_emb.0].grad[pos_off + c] += g;
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
                                let dctx_off = (b * seq_len + i) * d_model + h_col;
                                let qi_off = dctx_off;

                                // 1) dV += W^T @ dCtx；同时计算 dW[i, j] = dCtx[i] · V[j]
                                let mut w_dot_dw = 0.0f32;
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let vj_off = (b * seq_len + j) * d_model + h_col;
                                    let mut dw_ij = 0.0f32;
                                    for d in 0..d_head {
                                        let g = d_out[dctx_off + d];
                                        dv[vj_off + d] += w_ij * g;
                                        dw_ij += g * v_data[vj_off + d];
                                    }
                                    dw_row[j] = dw_ij;
                                    w_dot_dw += w_ij * dw_ij;
                                }

                                // 2) Softmax Jacobian 反传：dS[i,j] = W[i,j] * (dW[i,j] - sum_k W[i,k]*dW[i,k]) * scale
                                // 3) Q / K 梯度累加：dQ[i] += dS[i,j] * K[j]，dK[j] += dS[i,j] * Q[i]
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let ds_ij = w_ij * (dw_row[j] - w_dot_dw) * scale;
                                    let kj_off = (b * seq_len + j) * d_model + h_col;
                                    for d in 0..d_head {
                                        dq[qi_off + d] += ds_ij * k_data[kj_off + d];
                                        dk[kj_off + d] += ds_ij * q_data[qi_off + d];
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
                            let target_one_hot = if j == y { 1.0 } else { 0.0 };
                            self.nodes[logits_id.0].grad[row_off + j] +=
                                (probs[j] - target_one_hot) * scale;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 2. 可训练参数与 AdamW 优化器
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
        let data = (0..len).map(|_| rng.normal(std)).collect();
        Self {
            rows,
            cols,
            data,
            m: vec![0.0; len],
            v: vec![0.0; len],
            decay,
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

    fn adamw_step(&mut self, grad: &[f32], lr: f32, weight_decay: f32, step: usize) {
        let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);
        for i in 0..self.data.len() {
            let g = grad[i];
            if self.decay {
                self.data[i] -= lr * weight_decay * self.data[i];
            }
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g;
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            self.data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

// ============================================================================
// 3. 因果多头自注意力语言模型 (Single-Layer Attention + Residual + LM Head)
// ============================================================================

struct AttentionLm {
    vocab_size: usize,
    block_size: usize,
    n_head: usize,
    tok_emb: Param, // (V, D)
    pos_emb: Param, // (BLOCK_SIZE, D)
    wq: Param,      // (D, D)
    wk: Param,      // (D, D)
    wv: Param,      // (D, D)
    wo: Param,      // (D, D)
    w_lm: Param,    // (D, V)
    b_lm: Param,    // (1, V)
}

struct ForwardIds {
    tok_emb: TensorId,
    pos_emb: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: TensorId,
    wo: TensorId,
    w_lm: TensorId,
    b_lm: TensorId,
    attn: TensorId,
    logits: TensorId,
}

impl AttentionLm {
    fn new(vocab_size: usize, block_size: usize, d_model: usize, n_head: usize, rng: &mut Rng) -> Self {
        let std_attn = (1.0 / d_model as f32).sqrt();
        Self {
            vocab_size,
            block_size,
            n_head,
            tok_emb: Param::randn(vocab_size, d_model, 0.25, rng, true),
            pos_emb: Param::randn(block_size, d_model, 0.15, rng, true),
            wq: Param::randn(d_model, d_model, std_attn, rng, true),
            wk: Param::randn(d_model, d_model, std_attn, rng, true),
            wv: Param::randn(d_model, d_model, std_attn, rng, true),
            wo: Param::randn(d_model, d_model, std_attn * 0.5, rng, true),
            w_lm: Param::randn(d_model, vocab_size, std_attn, rng, true),
            b_lm: Param::zeros(1, vocab_size),
        }
    }

    fn num_params(&self) -> usize {
        self.tok_emb.data.len()
            + self.pos_emb.data.len()
            + self.wq.data.len()
            + self.wk.data.len()
            + self.wv.data.len()
            + self.wo.data.len()
            + self.w_lm.data.len()
            + self.b_lm.data.len()
    }

    /// 前向传播：给定一批长度相同的序列 (B × T)，并行算出每个位置的预测 logits(B*T, V)
    fn forward(&self, tape: &mut Tape, seqs: &[Vec<usize>]) -> ForwardIds {
        let batch_size = seqs.len();
        let seq_len = seqs[0].len();

        let tok_emb = tape.param(&self.tok_emb);
        let pos_emb = tape.param(&self.pos_emb);
        let wq = tape.param(&self.wq);
        let wk = tape.param(&self.wk);
        let wv = tape.param(&self.wv);
        let wo = tape.param(&self.wo);
        let w_lm = tape.param(&self.w_lm);
        let b_lm = tape.param(&self.b_lm);

        // 1. Token + Position Embedding
        let h0 = tape.embed_pos(tok_emb, pos_emb, seqs);
        // 2. Q / K / V 投影
        let q = tape.matmul(h0, wq);
        let k = tape.matmul(h0, wk);
        let v = tape.matmul(h0, wv);
        // 3. 因果多头自注意力
        let attn = tape.causal_multi_head_attn(q, k, v, batch_size, seq_len, self.n_head);
        let attn_out = tape.matmul(attn, wo);
        // 4. 残差连接：保留当前字自身信息 + 叠加从历史汇聚来的上下文
        let h1 = tape.add(h0, attn_out);
        // 5. 语言模型输出头
        let logits_raw = tape.matmul(h1, w_lm);
        let logits = tape.add(logits_raw, b_lm);

        ForwardIds {
            tok_emb,
            pos_emb,
            wq,
            wk,
            wv,
            wo,
            w_lm,
            b_lm,
            attn,
            logits,
        }
    }

    fn train_step(
        &mut self,
        seqs: &[Vec<usize>],
        targets: &[usize],
        lr: f32,
        step: usize,
    ) -> f32 {
        let mut tape = Tape::new();
        let ids = self.forward(&mut tape, seqs);
        let loss_id = tape.cross_entropy(ids.logits, targets);
        let loss = tape.data(loss_id)[0];

        tape.backward(loss_id);

        let wd = 0.01;
        self.tok_emb.adamw_step(tape.grad(ids.tok_emb), lr, wd, step);
        self.pos_emb.adamw_step(tape.grad(ids.pos_emb), lr, wd, step);
        self.wq.adamw_step(tape.grad(ids.wq), lr, wd, step);
        self.wk.adamw_step(tape.grad(ids.wk), lr, wd, step);
        self.wv.adamw_step(tape.grad(ids.wv), lr, wd, step);
        self.wo.adamw_step(tape.grad(ids.wo), lr, wd, step);
        self.w_lm.adamw_step(tape.grad(ids.w_lm), lr, wd, step);
        self.b_lm.adamw_step(tape.grad(ids.b_lm), lr, wd, step);

        loss
    }

    /// 自回归生成：注意！哪怕只给 1 个字，自注意力也能立刻开始续写，并在 block_size 内回看全部历史！
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
            let last_off = (t_len - 1) * self.vocab_size;
            let last_logits = &tape.data(ids.logits)[last_off..last_off + self.vocab_size];
            let next_id = sample_from_logits(last_logits, temperature, rng);
            out.push(next_id);
        }
        out
    }

    /// 计算一段诗句上所有头的平均因果注意力矩阵 (T × T)，方便在终端直观观察「每个字回头看了谁」
    fn mean_attention_matrix(&self, seq: &[usize]) -> Vec<f32> {
        let t = seq.len();
        let mut tape = Tape::new();
        let ids = self.forward(&mut tape, &[seq.to_vec()]);
        let weights = tape.attn_weights(ids.attn);
        let mut mean = vec![0.0f32; t * t];
        for h in 0..self.n_head {
            let base = h * t * t;
            for i in 0..t * t {
                mean[i] += weights[base + i] / self.n_head as f32;
            }
        }
        mean
    }
}

// ============================================================================
// 4. 主程序：训练 16 字视野因果自注意力模型 + 破除 Step 02 三字盲区 + 打印注意力矩阵
// ============================================================================

fn main() {
    let text = load_poems();
    let tok = Tokenizer::from_text(&text);
    let data = tok.encode(&text);

    let block_size = 16;
    let d_model = 32;
    let n_head = 4;
    let batch_size = 16;
    let steps = 1200;
    let lr = 0.012;

    println!("=== Step 03: 纯 Rust 因果多头自注意力 (Causal Multi-Head Self-Attention) ===");
    println!(
        "语料总字符数: {}, 词表大小 V: {}, 理论初始 Loss = ln({}) = {:.4}",
        data.len(),
        tok.vocab_size(),
        tok.vocab_size(),
        (tok.vocab_size() as f32).ln()
    );

    let mut rng = Rng::new(42);
    let mut model = AttentionLm::new(tok.vocab_size(), block_size, d_model, n_head, &mut rng);
    println!(
        "模型结构: Token+PosEmb(T={}, D={}) -> {}头因果自注意力(d_head={}) + Residual -> LM Head | 总参数量: {}（比 Step 02 的 52,141 更轻，视野却是 5.3 倍！）",
        block_size,
        d_model,
        n_head,
        d_model / n_head,
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

        let loss = model.train_step(&seqs, &targets, lr, step);
        if step == 1 || step % 200 == 0 {
            println!("step {:4}/{} | loss = {:.4}", step, steps, loss);
        }
    }
    println!("训练完成！耗时 {:.2}s", t0.elapsed().as_secs_f32());

    // 1. 验证：彻底攻克 Step 02 的「床前明月光，疑是」长程依赖难题！
    println!("\n=== 续写对比测试（Step 02 曾在「床前明月光，疑是」处串台成「银河落九天」）===");
    for prompt in ["床前明月光，疑是", "举头望明月，", "白日依山尽，黄河", "春眠不觉晓，"] {
        let prompt_ids = tok.encode(prompt);
        let generated = model.generate(&prompt_ids, 18, 0.3, &mut rng);
        println!("输入「{}」 -> {}", prompt, tok.decode(&generated).replace('\n', " / "));
    }

    // 2. 打印因果自注意力热力矩阵（4 头平均），直观看到下三角掩码与长程回望！
    let inspect_str = "床前明月光，疑是地上霜。";
    let inspect_ids = tok.encode(inspect_str);
    let inspect_chars: Vec<char> = inspect_str.chars().collect();
    let attn = model.mean_attention_matrix(&inspect_ids);
    let t = inspect_ids.len();

    println!("\n=== 因果自注意力权重矩阵（4头平均，行=Query当前字，列=Key历史字，上三角已被掩码屏蔽）===");
    print!("Q\\K  ");
    for ch in &inspect_chars {
        print!(" {:^3} ", ch);
    }
    println!();
    for i in 0..t {
        print!(" {}   ", inspect_chars[i]);
        for j in 0..t {
            if j > i {
                print!("  ·  ");
            } else {
                let w = attn[i * t + j];
                print!("{:4.0}%", w * 100.0);
            }
        }
        println!();
    }
}

// ============================================================================
// 5. 单元测试：数值梯度检验 + 因果掩码零未来泄露测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_loss(model: &AttentionLm, seqs: &[Vec<usize>], targets: &[usize]) -> f32 {
        let mut tape = Tape::new();
        let ids = model.forward(&mut tape, seqs);
        let loss_id = tape.cross_entropy(ids.logits, targets);
        tape.data(loss_id)[0]
    }

    #[test]
    fn test_attention_gradient_check() {
        let mut rng = Rng::new(777);
        let mut model = AttentionLm::new(6, 5, 8, 2, &mut rng);
        let seqs = vec![vec![0, 1, 2, 3, 4], vec![2, 0, 4, 1, 5]];
        let targets = vec![1, 2, 3, 4, 5, 0, 4, 1, 5, 3];

        let mut tape = Tape::new();
        let ids = model.forward(&mut tape, &seqs);
        let loss_id = tape.cross_entropy(ids.logits, &targets);
        tape.backward(loss_id);

        let g_tok = tape.grad(ids.tok_emb).to_vec();
        let g_pos = tape.grad(ids.pos_emb).to_vec();
        let g_wq = tape.grad(ids.wq).to_vec();
        let g_wk = tape.grad(ids.wk).to_vec();
        let g_wv = tape.grad(ids.wv).to_vec();
        let g_wo = tape.grad(ids.wo).to_vec();
        let g_wlm = tape.grad(ids.w_lm).to_vec();
        let g_blm = tape.grad(ids.b_lm).to_vec();

        let eps = 1e-3f32;
        let check = |name: &str,
                     analytic: &[f32],
                     getter: fn(&mut AttentionLm) -> &mut [f32],
                     model: &mut AttentionLm| {
            let mut max_err = 0.0f32;
            let len = analytic.len();
            for i in 0..len {
                let orig = getter(model)[i];
                getter(model)[i] = orig + eps;
                let lp = compute_loss(model, &seqs, &targets);
                getter(model)[i] = orig - eps;
                let lm = compute_loss(model, &seqs, &targets);
                getter(model)[i] = orig;

                let num = (lp - lm) / (2.0 * eps);
                let err = (num - analytic[i]).abs();
                if err > max_err {
                    max_err = err;
                }
            }
            assert!(
                max_err < 2e-3,
                "参数 {} 数值梯度检验失败: max_err = {}",
                name,
                max_err
            );
        };

        check("tok_emb", &g_tok, |m| &mut m.tok_emb.data, &mut model);
        check("pos_emb", &g_pos, |m| &mut m.pos_emb.data, &mut model);
        check("wq", &g_wq, |m| &mut m.wq.data, &mut model);
        check("wk", &g_wk, |m| &mut m.wk.data, &mut model);
        check("wv", &g_wv, |m| &mut m.wv.data, &mut model);
        check("wo", &g_wo, |m| &mut m.wo.data, &mut model);
        check("w_lm", &g_wlm, |m| &mut m.w_lm.data, &mut model);
        check("b_lm", &g_blm, |m| &mut m.b_lm.data, &mut model);
    }

    #[test]
    fn test_causal_mask_no_future_leak() {
        let mut rng = Rng::new(99);
        let model = AttentionLm::new(8, 6, 8, 2, &mut rng);

        // 两个序列前 3 个字完全相同，但后 3 个未来字完全不同
        let seq_a = vec![vec![1, 2, 3, 4, 5, 6]];
        let seq_b = vec![vec![1, 2, 3, 7, 0, 0]];

        let mut tape_a = Tape::new();
        let ids_a = model.forward(&mut tape_a, &seq_a);
        let mut tape_b = Tape::new();
        let ids_b = model.forward(&mut tape_b, &seq_b);

        let la = tape_a.data(ids_a.logits);
        let lb = tape_b.data(ids_b.logits);

        // 因果掩码保证：位置 0, 1, 2 的输出 logits 必须逐位完全相等（零未来泄露！）
        for pos in 0..3 {
            for v in 0..model.vocab_size {
                let diff = (la[pos * model.vocab_size + v] - lb[pos * model.vocab_size + v]).abs();
                assert!(diff < 1e-6, "位置 {} 发生了未来信息泄露！diff = {}", pos, diff);
            }
        }
    }
}
