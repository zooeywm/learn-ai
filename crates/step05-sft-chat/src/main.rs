//! Step 05: SFT 指令微调（把续写机变成迷你 ChatGPT）+ KV Cache（Prefill / Decode 两阶段加速）
//!
//! 关键震撼事实：
//!   **网络结构相对 Step 04 一行不改！** 依然是同一个 2 层 Pre-Norm Decoder-Only Mini-GPT。
//!   从「只会无脑接龙的 Base 模型」进化为「听懂指令、按时闭嘴的 Chat 助手」，只做了 3 件本质的事：
//!     1. 词表新增特殊控制符（Chat Template）：`<|pad|>`, `<|user|>`, `<|assistant|>`, `<|eos|>`
//!     2. 两阶段训练：
//!        - Stage 1 预训练（Pre-training）：在唐诗语料上全位置计算 Loss，学会语言和唐诗知识；
//!        - Stage 2 指令微调（SFT）：用对话模板拼接问答，并用 **Assistant-Only Loss Mask**
//!          把 `<|user|> 问题 <|assistant|>` 部分的 Loss 屏蔽为 0（只读题不扣分），
//!          只在回答正文 + `<|eos|>` 上计算梯度！
//!     3. 两阶段加速推理（Prefill + Decode + KV Cache）：
//!        - Prefill 阶段：一次性并行吃掉整句提问，把每一层的历史 K / V 矩阵存入 `KvCache`；
//!        - Decode 阶段：每生成一个新字只算这 1 个新 token 的 Q/K/V，直接复用缓存库中的历史 K/V，
//!          并在模型预测出 `<|eos|>` 时立刻自动闭嘴！

use std::collections::{BTreeSet, HashMap};
use std::env;
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use mini_core::{load_poems, load_sft_dataset, sample_from_logits, softmax, Rng};

// ============================================================================
// 1. 支持特殊控制符与对话模板的 ChatTokenizer
// ============================================================================

const PAD_ID: usize = 0;
const USER_ID: usize = 1;
const ASST_ID: usize = 2;
const EOS_ID: usize = 3;

struct ChatTokenizer {
    id_to_token: Vec<String>,
    char_to_id: HashMap<char, usize>,
}

impl ChatTokenizer {
    fn from_texts(texts: &[&str]) -> Self {
        let mut chars = BTreeSet::new();
        for &t in texts {
            for c in t.chars() {
                chars.insert(c);
            }
        }
        let mut id_to_token = vec![
            "<|pad|>".to_string(),
            "<|user|>".to_string(),
            "<|assistant|>".to_string(),
            "<|eos|>".to_string(),
        ];
        let mut char_to_id = HashMap::new();
        for c in chars {
            let id = id_to_token.len();
            char_to_id.insert(c, id);
            id_to_token.push(c.to_string());
        }
        Self {
            id_to_token,
            char_to_id,
        }
    }

    fn from_vocab(id_to_token: Vec<String>) -> Self {
        let mut char_to_id = HashMap::new();
        for (id, tok) in id_to_token.iter().enumerate().skip(4) {
            if let Some(c) = tok.chars().next() {
                char_to_id.insert(c, id);
            }
        }
        Self {
            id_to_token,
            char_to_id,
        }
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn encode_plain(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }

    /// 把用户问题打包成推理前缀：`<|user|> 问题 <|assistant|>`
    fn encode_prompt(&self, question: &str) -> Vec<usize> {
        self.encode_prompt_checked(question).0
    }

    /// 返回 `(prompt_ids, oov_chars)`，便于控制台交互时明确提示哪些生僻字不在当前词表中
    fn encode_prompt_checked(&self, question: &str) -> (Vec<usize>, Vec<char>) {
        let mut ids = Vec::with_capacity(question.chars().count() + 2);
        let mut oov = Vec::new();
        ids.push(USER_ID);
        for c in question.chars() {
            if let Some(&id) = self.char_to_id.get(&c) {
                ids.push(id);
            } else {
                oov.push(c);
            }
        }
        ids.push(ASST_ID);
        (ids, oov)
    }

    /// 把一条 (问题, 回答) 打包成 SFT 训练样本：
    ///   tokens = [<|user|>, ...问题..., <|assistant|>, ...回答..., <|eos|>]
    ///   返回 (input_ids, target_ids, loss_mask)
    ///   其中 prompt 部分（target 为问题字符或 <|assistant|>）的 loss_mask = 0.0，
    ///   只有回答正文与 <|eos|> 的 loss_mask = 1.0！
    fn encode_chat_sample(&self, question: &str, answer: &str) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let mut tokens = self.encode_prompt(question);
        let prompt_len = tokens.len(); // 最后一个是 <|assistant|>
        tokens.extend(self.encode_plain(answer));
        tokens.push(EOS_ID);

        let seq_len = tokens.len() - 1;
        let input_ids = tokens[..seq_len].to_vec();
        let target_ids = tokens[1..].to_vec();
        let mut loss_mask = vec![0.0f32; seq_len];
        // input[prompt_len - 1] 正好是 <|assistant|>，它预测的 target 就是回答的第 1 个字！
        for m in loss_mask.iter_mut().skip(prompt_len - 1) {
            *m = 1.0;
        }
        (input_ids, target_ids, loss_mask)
    }

    fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    /// 仅解码回答正文（遇到 `<|eos|>` 停止，并忽略控制符）
    fn decode_reply(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == EOS_ID {
                break;
            }
            if id >= 4 {
                if let Some(s) = self.id_to_token.get(id) {
                    out.push_str(s);
                }
            }
        }
        out
    }
}

// ============================================================================
// 2. Arena-Tape 自动求导引擎（在 Step 04 基础上新增 MaskedCrossEntropy）
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
    /// SFT 核心算子：按位置屏蔽 Loss，mask=0 的提问与 padding 位置零 Loss、零梯度！
    MaskedCrossEntropy {
        logits: TensorId,
        targets: Vec<usize>,
        mask: Vec<f32>,
        denom: f32,
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

    /// 掩码交叉熵：当 `mask` 全为 1.0 时等价于预训练 Loss；
    /// 在 SFT 阶段将提问部分与 padding 置 0.0，只对助手回答 + `<|eos|>` 计算 Loss 与梯度！
    fn masked_cross_entropy(
        &mut self,
        logits: TensorId,
        targets: &[usize],
        mask: &[f32],
    ) -> TensorId {
        let (n, v) = self.shape(logits);
        assert_eq!(n, targets.len());
        assert_eq!(n, mask.len());
        let z = &self.nodes[logits.0].data;
        let denom = mask.iter().sum::<f32>().max(1.0);
        let mut total = 0.0f32;
        for (row, (&y, &m)) in targets.iter().zip(mask.iter()).enumerate() {
            if m <= 0.0 {
                continue;
            }
            let probs = softmax(&z[row * v..(row + 1) * v]);
            total += m * -(probs[y].max(1e-12)).ln();
        }
        self.push(
            1,
            1,
            vec![total / denom],
            Op::MaskedCrossEntropy {
                logits,
                targets: targets.to_vec(),
                mask: mask.to_vec(),
                denom,
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

                Op::MaskedCrossEntropy {
                    logits,
                    targets,
                    mask,
                    denom,
                } => {
                    let logits_id = *logits;
                    let targets = targets.clone();
                    let mask = mask.clone();
                    let denom = *denom;
                    let (_n, v) = self.shape(logits_id);
                    let base_scale = d_out[0] / denom;
                    let z = self.nodes[logits_id.0].data.clone();
                    for (row, (&y, &m)) in targets.iter().zip(mask.iter()).enumerate() {
                        if m <= 0.0 {
                            continue;
                        }
                        let scale = base_scale * m;
                        let probs = softmax(&z[row * v..(row + 1) * v]);
                        let row_off = row * v;
                        for j in 0..v {
                            let one_hot = if j == y { 1.0 } else { 0.0 };
                            self.nodes[logits_id.0].grad[row_off + j] +=
                                (probs[j] - one_hot) * scale;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 3. 参数、AdamW 与与 Step 04 完全一致的 2 层 Pre-Norm Mini-GPT + KV Cache
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

    /// 切换阶段（如预训练 -> SFT）时清空动量缓冲，避免旧阶段梯度惯性干扰微调
    fn reset_optimizer(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
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

/// 纯无梯度的单向量基础算子，供 Prefill / Decode KV Cache 高速推理使用
fn layer_norm_vec(x: &[f32], gamma: &[f32], beta: &[f32]) -> Vec<f32> {
    let d = x.len();
    let inv_d = 1.0 / d as f32;
    let mean = x.iter().sum::<f32>() * inv_d;
    let var = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() * inv_d;
    let inv_std = 1.0 / (var + 1e-5f32).sqrt();
    (0..d)
        .map(|c| (x[c] - mean) * inv_std * gamma[c] + beta[c])
        .collect()
}

fn matvec_row(x: &[f32], w: &Param) -> Vec<f32> {
    let k = w.rows;
    let n = w.cols;
    let mut out = vec![0.0f32; n];
    for p in 0..k {
        let xp = x[p];
        let row = p * n;
        for j in 0..n {
            out[j] += xp * w.data[row + j];
        }
    }
    out
}

fn gelu_scalar(v: f32) -> f32 {
    let k = (2.0f32 / PI).sqrt();
    let u = k * (v + 0.044715 * v * v * v);
    0.5 * v * (1.0 + u.tanh())
}

/// 每一层的 KV Cache：按时间步追加存储 `(seq_len, d_model)` 的历史 Key 与 Value 向量
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

struct KvCache {
    layers: Vec<LayerKv>,
    seq_len: usize,
}

impl KvCache {
    fn new(n_layer: usize) -> Self {
        Self {
            layers: (0..n_layer)
                .map(|_| LayerKv {
                    k: Vec::new(),
                    v: Vec::new(),
                })
                .collect(),
            seq_len: 0,
        }
    }
}

struct TransformerBlock {
    ln1_gamma: Param,
    ln1_beta: Param,
    wq: Param,
    wk: Param,
    wv: Param,
    wo: Param,
    ln2_gamma: Param,
    ln2_beta: Param,
    ffn_w1: Param,
    ffn_b1: Param,
    ffn_w2: Param,
    ffn_b2: Param,
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

    fn reset_optimizer(&mut self) {
        self.ln1_gamma.reset_optimizer();
        self.ln1_beta.reset_optimizer();
        self.wq.reset_optimizer();
        self.wk.reset_optimizer();
        self.wv.reset_optimizer();
        self.wo.reset_optimizer();
        self.ln2_gamma.reset_optimizer();
        self.ln2_beta.reset_optimizer();
        self.ffn_w1.reset_optimizer();
        self.ffn_b1.reset_optimizer();
        self.ffn_w2.reset_optimizer();
        self.ffn_b2.reset_optimizer();
    }

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

    /// 单步 KV Cache 前向：只接收 1 个新 token 的隐藏向量 `x`（长度 D），
    /// 计算它的 `q_new / k_new / v_new`，将 `k_new / v_new` 追加到 `layer_kv`，
    /// 再用 `q_new` 去查询缓存库里的全部历史 K/V！
    fn forward_cached_step(&self, x: &[f32], n_head: usize, layer_kv: &mut LayerKv) -> Vec<f32> {
        let d_model = x.len();
        let d_head = d_model / n_head;
        let scale = 1.0 / (d_head as f32).sqrt();

        let ln1 = layer_norm_vec(x, &self.ln1_gamma.data, &self.ln1_beta.data);
        let q = matvec_row(&ln1, &self.wq);
        let k_new = matvec_row(&ln1, &self.wk);
        let v_new = matvec_row(&ln1, &self.wv);

        layer_kv.k.extend_from_slice(&k_new);
        layer_kv.v.extend_from_slice(&v_new);
        let total_t = layer_kv.k.len() / d_model;

        let mut attn = vec![0.0f32; d_model];
        let mut scores = vec![0.0f32; total_t];
        for h in 0..n_head {
            let h_col = h * d_head;
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..total_t {
                let k_off = j * d_model + h_col;
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q[h_col + d] * layer_kv.k[k_off + d];
                }
                let s = dot * scale;
                scores[j] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for j in 0..total_t {
                let w = scores[j] * inv_sum;
                let v_off = j * d_model + h_col;
                for d in 0..d_head {
                    attn[h_col + d] += w * layer_kv.v[v_off + d];
                }
            }
        }

        let attn_out = matvec_row(&attn, &self.wo);
        let mut x1 = vec![0.0f32; d_model];
        for c in 0..d_model {
            x1[c] = x[c] + attn_out[c];
        }

        let ln2 = layer_norm_vec(&x1, &self.ln2_gamma.data, &self.ln2_beta.data);
        let mut up = matvec_row(&ln2, &self.ffn_w1);
        for (u, &b) in up.iter_mut().zip(self.ffn_b1.data.iter()) {
            *u = gelu_scalar(*u + b);
        }
        let down = matvec_row(&up, &self.ffn_w2);
        let mut x2 = vec![0.0f32; d_model];
        for c in 0..d_model {
            x2[c] = x1[c] + down[c] + self.ffn_b2.data[c];
        }
        x2
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
    d_model: usize,
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
            d_model,
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

    fn reset_optimizer(&mut self) {
        self.tok_emb.reset_optimizer();
        self.pos_emb.reset_optimizer();
        for b in &mut self.blocks {
            b.reset_optimizer();
        }
        self.ln_f_gamma.reset_optimizer();
        self.ln_f_beta.reset_optimizer();
        self.w_lm.reset_optimizer();
        self.b_lm.reset_optimizer();
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

    /// 通用训练一步：预训练传全 1.0 掩码，SFT 传「提问=0.0 / 回答+<|eos|>=1.0」掩码！
    fn train_step(
        &mut self,
        seqs: &[Vec<usize>],
        targets: &[usize],
        loss_mask: &[f32],
        lr: f32,
        step: usize,
    ) -> (f32, f32) {
        let mut tape = Tape::new();
        let ids = self.forward(&mut tape, seqs);
        let loss_id = tape.masked_cross_entropy(ids.logits, targets, loss_mask);
        let loss = tape.data(loss_id)[0];

        tape.backward(loss_id);

        let mut all_ids = vec![
            ids.tok_emb,
            ids.pos_emb,
            ids.ln_f_gamma,
            ids.ln_f_beta,
            ids.w_lm,
            ids.b_lm,
        ];
        for b in &ids.blocks {
            all_ids.extend_from_slice(&[
                b.ln1_gamma,
                b.ln1_beta,
                b.wq,
                b.wk,
                b.wv,
                b.wo,
                b.ln2_gamma,
                b.ln2_beta,
                b.ffn_w1,
                b.ffn_b1,
                b.ffn_w2,
                b.ffn_b2,
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
        let clip_scale = if grad_norm > max_norm {
            max_norm / grad_norm
        } else {
            1.0
        };

        let wd = 0.01;
        self.tok_emb
            .adamw_step(tape.grad(ids.tok_emb), clip_scale, lr, wd, step);
        self.pos_emb
            .adamw_step(tape.grad(ids.pos_emb), clip_scale, lr, wd, step);
        for (blk, bids) in self.blocks.iter_mut().zip(ids.blocks.iter()) {
            for (param, tid) in blk.param_grad_pairs(bids) {
                param.adamw_step(tape.grad(tid), clip_scale, lr, wd, step);
            }
        }
        self.ln_f_gamma
            .adamw_step(tape.grad(ids.ln_f_gamma), clip_scale, lr, wd, step);
        self.ln_f_beta
            .adamw_step(tape.grad(ids.ln_f_beta), clip_scale, lr, wd, step);
        self.w_lm
            .adamw_step(tape.grad(ids.w_lm), clip_scale, lr, wd, step);
        self.b_lm
            .adamw_step(tape.grad(ids.b_lm), clip_scale, lr, wd, step);

        (loss, grad_norm)
    }

    // ------------------------------------------------------------------------
    // 推理阶段 1：Prefill（一次性吃掉整句 Prompt，把每一层的 K/V 填满进缓存）
    // ------------------------------------------------------------------------
    fn prefill(&self, prompt_ids: &[usize]) -> (Vec<f32>, KvCache) {
        assert!(!prompt_ids.is_empty());
        assert!(prompt_ids.len() <= self.block_size);
        let mut cache = KvCache::new(self.blocks.len());
        let mut last_logits = Vec::new();
        for &tok_id in prompt_ids {
            last_logits = self.decode_step(tok_id, &mut cache);
        }
        (last_logits, cache)
    }

    // ------------------------------------------------------------------------
    // 推理阶段 2：Decode（每步只算 1 个新 token，直接复用 KvCache 中的全部历史 K/V）
    // ------------------------------------------------------------------------
    fn decode_step(&self, tok_id: usize, cache: &mut KvCache) -> Vec<f32> {
        let pos = cache.seq_len;
        assert!(pos < self.block_size, "超出上下文窗口 block_size");
        let d = self.d_model;
        let mut x = vec![0.0f32; d];
        let tok_off = tok_id * d;
        let pos_off = pos * d;
        for c in 0..d {
            x[c] = self.tok_emb.data[tok_off + c] + self.pos_emb.data[pos_off + c];
        }

        for (l, blk) in self.blocks.iter().enumerate() {
            x = blk.forward_cached_step(&x, self.n_head, &mut cache.layers[l]);
        }
        cache.seq_len += 1;

        let x_norm = layer_norm_vec(&x, &self.ln_f_gamma.data, &self.ln_f_beta.data);
        let mut logits = matvec_row(&x_norm, &self.w_lm);
        for (z, &b) in logits.iter_mut().zip(self.b_lm.data.iter()) {
            *z += b;
        }
        logits
    }

    /// 使用 Prefill + Decode KV Cache 生成对话回复，遇到 `<|eos|>` 立即自动闭嘴！
    fn generate_chat_kv(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> (Vec<usize>, bool, usize) {
        let (mut logits, mut cache) = self.prefill(prompt_ids);
        let mut generated = Vec::new();
        let mut hit_eos = false;
        let max_gen = max_new_tokens.min(self.block_size.saturating_sub(prompt_ids.len()));

        for _ in 0..max_gen {
            let next_id = if temperature <= 1e-4 {
                argmax(&logits)
            } else {
                sample_from_logits(&logits, temperature, rng)
            };
            generated.push(next_id);
            if next_id == EOS_ID {
                hit_eos = true;
                break;
            }
            if cache.seq_len >= self.block_size {
                break;
            }
            logits = self.decode_step(next_id, &mut cache);
        }
        (generated, hit_eos, cache.seq_len)
    }

    /// 无 KV Cache 的朴素全量重算推理（用于与 KV Cache 对比速度与一致性）
    fn generate_chat_naive(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
    ) -> Vec<usize> {
        let mut seq = prompt_ids.to_vec();
        let mut generated = Vec::new();
        let max_gen = max_new_tokens.min(self.block_size.saturating_sub(prompt_ids.len()));
        for _ in 0..max_gen {
            let window = vec![seq.clone()];
            let mut tape = Tape::new();
            let ids = self.forward(&mut tape, &window);
            let off = (seq.len() - 1) * self.vocab_size;
            let last_logits = &tape.data(ids.logits)[off..off + self.vocab_size];
            let next_id = argmax(last_logits);
            generated.push(next_id);
            if next_id == EOS_ID {
                break;
            }
            seq.push(next_id);
        }
        generated
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn cosine_lr(step: usize, total_steps: usize, warmup_steps: usize, max_lr: f32, min_lr: f32) -> f32 {
    if step <= warmup_steps {
        max_lr * (step as f32 / warmup_steps.max(1) as f32)
    } else {
        let progress = (step - warmup_steps) as f32 / (total_steps - warmup_steps).max(1) as f32;
        min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (PI * progress).cos())
    }
}

// ============================================================================
// 4. 零依赖二进制模型存档（MGPT v1：一次保存词表 + 超参 + 全部权重，毫秒级重载）
// ============================================================================

const CHECKPOINT_MAGIC: &[u8; 4] = b"MGPT";
const CHECKPOINT_VERSION: u32 = 1;

fn default_checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/step05-chat.mgpt")
}

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_param(w: &mut impl Write, p: &Param) -> io::Result<()> {
    write_u32(w, p.data.len() as u32)?;
    for &x in &p.data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn read_param_into(r: &mut impl Read, p: &mut Param) -> io::Result<()> {
    let len = read_u32(r)? as usize;
    if len != p.data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("权重长度不匹配: 期望 {}，文件为 {}", p.data.len(), len),
        ));
    }
    let mut buf = [0u8; 4];
    for slot in &mut p.data {
        r.read_exact(&mut buf)?;
        *slot = f32::from_le_bytes(buf);
    }
    Ok(())
}

fn save_checkpoint(path: &Path, tok: &ChatTokenizer, model: &MiniGpt) -> io::Result<u64> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    w.write_all(CHECKPOINT_MAGIC)?;
    write_u32(&mut w, CHECKPOINT_VERSION)?;
    write_u32(&mut w, model.vocab_size as u32)?;
    write_u32(&mut w, model.block_size as u32)?;
    write_u32(&mut w, model.d_model as u32)?;
    write_u32(&mut w, model.n_head as u32)?;
    write_u32(&mut w, model.blocks.len() as u32)?;

    // 写入词表（加载推理时不再依赖原语料文件！）
    for s in &tok.id_to_token {
        let bytes = s.as_bytes();
        write_u32(&mut w, bytes.len() as u32)?;
        w.write_all(bytes)?;
    }

    // 按固定顺序写入全部模型权重
    write_param(&mut w, &model.tok_emb)?;
    write_param(&mut w, &model.pos_emb)?;
    for blk in &model.blocks {
        write_param(&mut w, &blk.ln1_gamma)?;
        write_param(&mut w, &blk.ln1_beta)?;
        write_param(&mut w, &blk.wq)?;
        write_param(&mut w, &blk.wk)?;
        write_param(&mut w, &blk.wv)?;
        write_param(&mut w, &blk.wo)?;
        write_param(&mut w, &blk.ln2_gamma)?;
        write_param(&mut w, &blk.ln2_beta)?;
        write_param(&mut w, &blk.ffn_w1)?;
        write_param(&mut w, &blk.ffn_b1)?;
        write_param(&mut w, &blk.ffn_w2)?;
        write_param(&mut w, &blk.ffn_b2)?;
    }
    write_param(&mut w, &model.ln_f_gamma)?;
    write_param(&mut w, &model.ln_f_beta)?;
    write_param(&mut w, &model.w_lm)?;
    write_param(&mut w, &model.b_lm)?;
    w.flush()?;

    Ok(fs::metadata(path)?.len())
}

fn load_checkpoint(path: &Path) -> io::Result<(ChatTokenizer, MiniGpt)> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "无效的模型文件头（期望 MGPT）",
        ));
    }
    let ver = read_u32(&mut r)?;
    if ver != CHECKPOINT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("不支持的模型版本: {}", ver),
        ));
    }

    let vocab_size = read_u32(&mut r)? as usize;
    let block_size = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let n_head = read_u32(&mut r)? as usize;
    let n_layer = read_u32(&mut r)? as usize;

    let mut id_to_token = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let len = read_u32(&mut r)? as usize;
        let mut bytes = vec![0u8; len];
        r.read_exact(&mut bytes)?;
        let s = String::from_utf8(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        id_to_token.push(s);
    }
    let tok = ChatTokenizer::from_vocab(id_to_token);

    let mut dummy_rng = Rng::new(1);
    let mut model = MiniGpt::new(vocab_size, block_size, d_model, n_head, n_layer, &mut dummy_rng);
    read_param_into(&mut r, &mut model.tok_emb)?;
    read_param_into(&mut r, &mut model.pos_emb)?;
    for blk in &mut model.blocks {
        read_param_into(&mut r, &mut blk.ln1_gamma)?;
        read_param_into(&mut r, &mut blk.ln1_beta)?;
        read_param_into(&mut r, &mut blk.wq)?;
        read_param_into(&mut r, &mut blk.wk)?;
        read_param_into(&mut r, &mut blk.wv)?;
        read_param_into(&mut r, &mut blk.wo)?;
        read_param_into(&mut r, &mut blk.ln2_gamma)?;
        read_param_into(&mut r, &mut blk.ln2_beta)?;
        read_param_into(&mut r, &mut blk.ffn_w1)?;
        read_param_into(&mut r, &mut blk.ffn_b1)?;
        read_param_into(&mut r, &mut blk.ffn_w2)?;
        read_param_into(&mut r, &mut blk.ffn_b2)?;
    }
    read_param_into(&mut r, &mut model.ln_f_gamma)?;
    read_param_into(&mut r, &mut model.ln_f_beta)?;
    read_param_into(&mut r, &mut model.w_lm)?;
    read_param_into(&mut r, &mut model.b_lm)?;

    Ok((tok, model))
}

// ============================================================================
// 5. 控制台流式对话交互（REPL）：毫秒级加载权重 + Prefill/Decode KV Cache 实时吐字
// ============================================================================

fn run_chat_repl(tok: &ChatTokenizer, model: &MiniGpt) {
    println!("\n======================================================================");
    println!("💬 迷你唐诗 ChatGPT 控制台交互模式（Prefill + Decode KV Cache 流式输出）");
    println!("提示：直接输入问题回车对话，如「你好」「背一首静夜思」「白日依山尽的下一句」");
    println!("命令：/temp <温度> (默认 0.0 贪心最稳，可调如 /temp 0.4) | /help | /quit");
    println!("======================================================================");

    let mut temp = 0.0f32;
    let mut rng = Rng::new(2026);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n你 > ");
        let _ = stdout.flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                println!("\n👋 已退出对话。");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("读取输入失败: {}", e);
                break;
            }
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input, "/quit" | "/exit" | "quit" | "exit") {
            println!("👋 再见！");
            break;
        }
        if input == "/help" {
            println!("试试问这些（来自 data/sft_poems.jsonl）：");
            println!("  · 你好 / 你是谁");
            println!("  · 背一首静夜思 / 背一首春晓 / 背一首登鹳雀楼 / 背一首相思");
            println!("  · 白日依山尽的下一句 / 飞流直下三千尺的下一句 / 锄禾日当午的下一句");
            println!("  · /temp 0.0（确定性回答）或 /temp 0.5（带随机性） | /quit 退出");
            continue;
        }
        if let Some(arg) = input.strip_prefix("/temp") {
            match arg.trim().parse::<f32>() {
                Ok(t) if (0.0..=2.5).contains(&t) => {
                    temp = t;
                    println!("✅ 采样温度已调整为 T = {:.2}", temp);
                }
                _ => println!("⚠️ 用法: /temp <0.0 ~ 2.5>，例如 /temp 0.0 或 /temp 0.4"),
            }
            continue;
        }

        let (mut prompt_ids, oov) = tok.encode_prompt_checked(input);
        if !oov.is_empty() {
            let uniq: String = oov.iter().collect();
            println!(
                "  [提示: 字符「{}」不在当前字符级词表(V={})中，已跳过]",
                uniq,
                tok.vocab_size()
            );
        }
        if prompt_ids.len() >= model.block_size - 1 {
            // 截断过长问题并保留末尾 <|assistant|>
            prompt_ids.truncate(model.block_size - 2);
            prompt_ids.push(ASST_ID);
        }

        // Phase 1: Prefill（一次性吃掉整句提问并填满 KV Cache）
        let t_pre = Instant::now();
        let (mut logits, mut cache) = model.prefill(&prompt_ids);
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        // Phase 2: Decode（逐字解码流式打印，遇到 <|eos|> 自动闭嘴）
        print!("助手 > ");
        let _ = stdout.flush();
        let t_dec = Instant::now();
        let max_gen = model.block_size.saturating_sub(prompt_ids.len());
        let mut gen_tokens = 0usize;
        let mut hit_eos = false;

        for _ in 0..max_gen {
            let next_id = if temp <= 1e-4 {
                argmax(&logits)
            } else {
                sample_from_logits(&logits, temp, &mut rng)
            };
            gen_tokens += 1;
            if next_id == EOS_ID {
                hit_eos = true;
                break;
            }
            if next_id >= 4 {
                if let Some(ch) = tok.id_to_token.get(next_id) {
                    print!("{}", ch);
                    let _ = stdout.flush();
                }
            }
            if cache.seq_len >= model.block_size {
                break;
            }
            logits = model.decode_step(next_id, &mut cache);
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        println!(
            "\n  [Prefill {:.2}ms | Decode {:.2}ms ({} tokens) | KV Cache {} 槽位 | hit_eos={}]",
            prefill_ms, decode_ms, gen_tokens, cache.seq_len, hit_eos
        );
    }
}

// ============================================================================
// 6. 训练流程与命令行入口（支持 train / chat / train --chat / --model <path>）
// ============================================================================

fn train_and_save(model_path: &Path) -> (ChatTokenizer, MiniGpt) {
    let poems = load_poems();
    let qa_pairs = load_sft_dataset();

    let mut all_texts: Vec<&str> = vec![poems.as_str()];
    for (q, a) in &qa_pairs {
        all_texts.push(q.as_str());
        all_texts.push(a.as_str());
    }
    let tok = ChatTokenizer::from_texts(&all_texts);
    let pretrain_data = tok.encode_plain(&poems);

    let block_size = 48;
    let pretrain_seq_len = 24;
    let d_model = 32;
    let n_head = 4;
    let n_layer = 2;

    println!("=== Step 05: SFT 指令微调（Mini-GPT -> 迷你 ChatGPT）+ KV Cache 推理加速 ===");
    println!(
        "词表大小 V = {}（含 4 个控制符: <|pad|>=0, <|user|>=1, <|assistant|>=2, <|eos|>=3）",
        tok.vocab_size()
    );

    let mut rng = Rng::new(42);
    let mut model = MiniGpt::new(
        tok.vocab_size(),
        block_size,
        d_model,
        n_head,
        n_layer,
        &mut rng,
    );
    println!(
        "网络结构: 与 Step 04 一行不改！{}层 Pre-Norm Transformer (D={}, H={}, block_size={}) | 总参数量: {}",
        n_layer,
        d_model,
        n_head,
        block_size,
        model.num_params()
    );

    // ------------------------------------------------------------------------
    // Stage 1: Pre-training（预训练，全位置 Loss Mask = 1.0）
    // ------------------------------------------------------------------------
    println!("\n--- Stage 1: 预训练 Base 模型（唐诗语料，全位置 Loss Mask = 1.0）---");
    let pt_batch = 16;
    let pt_steps = 900;
    let max_start = pretrain_data.len() - pretrain_seq_len - 1;
    let all_ones = vec![1.0f32; pt_batch * pretrain_seq_len];

    let t_pt = Instant::now();
    for step in 1..=pt_steps {
        let mut seqs = Vec::with_capacity(pt_batch);
        let mut targets = Vec::with_capacity(pt_batch * pretrain_seq_len);
        for _ in 0..pt_batch {
            let s = (rng.next_u64() as usize) % max_start;
            seqs.push(pretrain_data[s..s + pretrain_seq_len].to_vec());
            targets.extend_from_slice(&pretrain_data[s + 1..s + 1 + pretrain_seq_len]);
        }
        let lr = cosine_lr(step, pt_steps, 60, 0.012, 0.002);
        let (loss, gnorm) = model.train_step(&seqs, &targets, &all_ones, lr, step);
        if step == 1 || step % 300 == 0 {
            println!(
                "[Pretrain] step {:3}/{} | lr = {:.4} | grad_norm = {:.2} | loss = {:.4}",
                step, pt_steps, lr, gnorm, loss
            );
        }
    }
    println!("Stage 1 预训练完成！耗时 {:.2}s", t_pt.elapsed().as_secs_f32());

    // 先测一下「未经 SFT 的 Base 模型」回答对话问题会怎样！
    let test_questions = [
        "你好",
        "背一首静夜思",
        "白日依山尽的下一句",
        "飞流直下三千尺的下一句",
        "背一首春晓",
    ];
    println!("\n=== 对比 1：刚预训练完的 Base 模型（只会接龙，不懂 <|user|>/<|assistant|>/<|eos|>）===");
    for q in test_questions {
        let prompt = tok.encode_prompt(q);
        let (reply_ids, hit_eos, _) = model.generate_chat_kv(&prompt, 24, 0.0, &mut rng);
        println!(
            "问「{}」\n  Base 回复: {}  [hit_eos={}]",
            q,
            tok.decode(&reply_ids).replace('\n', " / "),
            hit_eos
        );
    }

    // ------------------------------------------------------------------------
    // Stage 2: SFT 指令微调（Assistant-Only Masked Cross-Entropy Loss）
    // ------------------------------------------------------------------------
    println!(
        "\n--- Stage 2: SFT 指令微调（从 data/sft_poems.jsonl 加载 {} 条问答，提问区 Loss Mask = 0）---",
        qa_pairs.len()
    );
    let encoded_samples: Vec<(Vec<usize>, Vec<usize>, Vec<f32>)> = qa_pairs
        .iter()
        .map(|(q, a)| tok.encode_chat_sample(q, a))
        .collect();

    // 打印第 1 条样本的 Loss Mask 直观示意图
    {
        let (ref inp, ref tgt, ref msk) = encoded_samples[0];
        let active: usize = msk.iter().filter(|&&m| m > 0.0).count();
        println!(
            "样本[0] 演示（总步长 {}，其中 mask=0 屏蔽 {} 步，mask=1 计分 {} 步）：",
            inp.len(),
            inp.len() - active,
            active
        );
        let preview: Vec<String> = inp
            .iter()
            .zip(tgt.iter())
            .zip(msk.iter())
            .map(|((&x, &y), &m)| {
                format!(
                    "{}->{}{}",
                    tok.id_to_token[x],
                    tok.id_to_token[y],
                    if m > 0.0 { "[1]" } else { "[0]" }
                )
            })
            .collect();
        println!("  {}", preview.join(" "));
    }

    model.reset_optimizer();
    let sft_steps = 360;
    let sft_batch = 6;
    let t_sft = Instant::now();

    for step in 1..=sft_steps {
        let mut chosen = Vec::with_capacity(sft_batch);
        let mut max_len = 0usize;
        for _ in 0..sft_batch {
            let idx = (rng.next_u64() as usize) % encoded_samples.len();
            max_len = max_len.max(encoded_samples[idx].0.len());
            chosen.push(idx);
        }

        let mut seqs = Vec::with_capacity(sft_batch);
        let mut targets = Vec::with_capacity(sft_batch * max_len);
        let mut mask = Vec::with_capacity(sft_batch * max_len);
        for idx in chosen {
            let (ref inp, ref tgt, ref msk) = encoded_samples[idx];
            let mut s = inp.clone();
            s.resize(max_len, PAD_ID);
            seqs.push(s);

            targets.extend_from_slice(tgt);
            targets.resize(targets.len() + (max_len - tgt.len()), PAD_ID);

            mask.extend_from_slice(msk);
            mask.resize(mask.len() + (max_len - msk.len()), 0.0);
        }

        let lr = cosine_lr(step, sft_steps, 30, 0.008, 0.0008);
        let (loss, gnorm) = model.train_step(&seqs, &targets, &mask, lr, step);
        if step == 1 || step % 90 == 0 {
            println!(
                "[SFT     ] step {:3}/{} | lr = {:.4} | grad_norm = {:.2} | masked_loss = {:.4}",
                step, sft_steps, lr, gnorm, loss
            );
        }
    }
    println!("Stage 2 SFT 微调完成！耗时 {:.2}s", t_sft.elapsed().as_secs_f32());

    // 测试 SFT 后的迷你 ChatGPT！
    println!("\n=== 对比 2：经过 SFT 微调后的迷你 ChatGPT（听懂问题 + 精准答复 + <|eos|> 自动闭嘴！）===");
    for q in test_questions {
        let prompt = tok.encode_prompt(q);
        let (reply_ids, hit_eos, final_kv_len) = model.generate_chat_kv(&prompt, 32, 0.0, &mut rng);
        println!(
            "问「{}」\n  Chat 回复: {}  [hit_eos={}, KV Cache 缓存 {} 槽位]",
            q,
            tok.decode_reply(&reply_ids),
            hit_eos,
            final_kv_len
        );
    }

    // ------------------------------------------------------------------------
    // Prefill + Decode KV Cache vs 无缓存全量重算 速度与一致性实测
    // ------------------------------------------------------------------------
    println!("\n=== Prefill + Decode KV Cache 推理加速实测（重复 30 次「背一首静夜思」）===");
    let bench_prompt = tok.encode_prompt("背一首静夜思");
    let runs = 30;

    let t_naive = Instant::now();
    let mut naive_out = Vec::new();
    for _ in 0..runs {
        naive_out = model.generate_chat_naive(&bench_prompt, 32);
    }
    let naive_ms = t_naive.elapsed().as_secs_f64() * 1000.0;

    let t_kv = Instant::now();
    let mut kv_out = Vec::new();
    for _ in 0..runs {
        let (g, _, _) = model.generate_chat_kv(&bench_prompt, 32, 0.0, &mut rng);
        kv_out = g;
    }
    let kv_ms = t_kv.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(naive_out, kv_out, "KV Cache 输出必须与无缓存全量重算完全一致！");
    println!(
        "输出一致性: 100% 逐 token 精确一致（共生成 {} tokens，含 <|eos|>）",
        kv_out.len()
    );
    println!(
        "无缓存全量重算 ({} 次): {:7.2} ms | Prefill+Decode KV Cache ({} 次): {:6.2} ms | 实测加速比: {:.2}x",
        runs,
        naive_ms,
        runs,
        kv_ms,
        naive_ms / kv_ms.max(1e-6)
    );

    // ------------------------------------------------------------------------
    // 保存模型权重与词表到二进制 Checkpoint，供日后毫秒级加载对话！
    // ------------------------------------------------------------------------
    match save_checkpoint(model_path, &tok, &model) {
        Ok(bytes) => {
            println!(
                "\n💾 模型已保存至: {} ({:.1} KB，含词表 V={} + 全部 {} 个 f32 权重)",
                model_path.display(),
                bytes as f64 / 1024.0,
                tok.vocab_size(),
                model.num_params()
            );
            println!("👉 下次无需重新训练，直接运行以下命令即可秒开控制台对话：");
            println!("   cargo run --release -p step05-sft-chat -- chat");
        }
        Err(e) => eprintln!("⚠️ 保存模型失败 {:?}: {}", model_path, e),
    }

    (tok, model)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut mode = "train";
    let mut chat_after_train = false;
    let mut model_path = default_checkpoint_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "chat" => mode = "chat",
            "train" => mode = "train",
            "--chat" => chat_after_train = true,
            "--model" => {
                if i + 1 < args.len() {
                    model_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                } else {
                    eprintln!("⚠️ --model 需要指定文件路径");
                    return;
                }
            }
            "-h" | "--help" => {
                println!("用法:");
                println!("  cargo run --release -p step05-sft-chat                  # 训练并保存模型到 checkpoints/step05-chat.mgpt");
                println!("  cargo run --release -p step05-sft-chat -- chat          # 直接加载已保存模型进入控制台交互对话");
                println!("  cargo run --release -p step05-sft-chat -- train --chat  # 训练保存后直接进入控制台交互对话");
                println!("  可选参数: --model <path> 指定自定义模型保存/加载路径");
                return;
            }
            other => {
                eprintln!("⚠️ 未知参数: {}（用 --help 查看用法）", other);
                return;
            }
        }
        i += 1;
    }

    if mode == "chat" {
        let t0 = Instant::now();
        match load_checkpoint(&model_path) {
            Ok((tok, model)) => {
                println!(
                    "✅ 已加载模型: {}（耗时 {:.2} ms | V={} | {} 参数）",
                    model_path.display(),
                    t0.elapsed().as_secs_f64() * 1000.0,
                    tok.vocab_size(),
                    model.num_params()
                );
                run_chat_repl(&tok, &model);
            }
            Err(e) => {
                eprintln!("❌ 无法加载模型 {}: {}", model_path.display(), e);
                eprintln!("请先运行一次训练生成模型文件：");
                eprintln!("  cargo run --release -p step05-sft-chat -- train --chat");
            }
        }
    } else {
        let (tok, model) = train_and_save(&model_path);
        if chat_after_train {
            run_chat_repl(&tok, &model);
        }
    }
}

// ============================================================================
// 6. 单元测试：Loss Mask 零泄露 + KV Cache 数值等价 + SFT 掩码梯度校验
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masked_cross_entropy_ignores_prompt() {
        // 验证：只要 mask=0 的位置被任意篡改 targets，Loss 和 logits 梯度都必须完全不受影响！
        let mut tape_a = Tape::new();
        let logits_data = vec![
            0.5, -1.2, 0.8, 0.1, // pos 0 (mask=0, prompt)
            -0.3, 0.9, 0.2, -0.5, // pos 1 (mask=0, prompt)
            1.1, 0.4, -0.7, 0.6, // pos 2 (mask=1, assistant)
            -0.2, 0.3, 1.5, -0.8, // pos 3 (mask=1, eos)
        ];
        let z_a = tape_a.push(4, 4, logits_data.clone(), Op::Leaf);
        let loss_a = tape_a.masked_cross_entropy(z_a, &[0, 1, 2, 3], &[0.0, 0.0, 1.0, 1.0]);
        tape_a.backward(loss_a);

        let mut tape_b = Tape::new();
        let z_b = tape_b.push(4, 4, logits_data, Op::Leaf);
        // 故意把前两个 mask=0 位置的 target 改掉！
        let loss_b = tape_b.masked_cross_entropy(z_b, &[3, 2, 2, 3], &[0.0, 0.0, 1.0, 1.0]);
        tape_b.backward(loss_b);

        assert!((tape_a.data(loss_a)[0] - tape_b.data(loss_b)[0]).abs() < 1e-7);
        // 前两行（mask=0）的梯度必须严格全 0！
        for i in 0..8 {
            assert_eq!(tape_a.grad(z_a)[i], 0.0, "mask=0 位置出现了非零梯度！");
        }
        // 后两行（mask=1）的梯度必须完全一致！
        for i in 8..16 {
            assert!((tape_a.grad(z_a)[i] - tape_b.grad(z_b)[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn test_kv_cache_matches_full_forward() {
        // 验证：Prefill 吃掉前 4 个 token + Decode 逐个追加 3 个 token，
        // 每一步的 logits 与完整 forward(seq[0..=t]) 必须数学上 100% 等价（误差 < 1e-5）！
        let mut rng = Rng::new(777);
        let model = MiniGpt::new(12, 16, 16, 4, 2, &mut rng);
        let seq = vec![1, 5, 7, 2, 9, 4, 3];

        let (prefill_logits, mut cache) = model.prefill(&seq[..4]);
        let mut tape_full = Tape::new();
        let full_ids = model.forward(&mut tape_full, &[seq.clone()]);
        let full_logits = tape_full.data(full_ids.logits);

        // 1) 校验 Prefill 末尾（位置 3）与 full_forward 位置 3 一致
        for v in 0..model.vocab_size {
            let diff = (prefill_logits[v] - full_logits[3 * model.vocab_size + v]).abs();
            assert!(diff < 1e-5, "Prefill logits 不一致: diff = {}", diff);
        }

        // 2) 校验随后 3 次 Decode（位置 4, 5, 6）每一步都与 full_forward 精确一致
        for pos in 4..seq.len() {
            let dec_logits = model.decode_step(seq[pos], &mut cache);
            for v in 0..model.vocab_size {
                let diff = (dec_logits[v] - full_logits[pos * model.vocab_size + v]).abs();
                assert!(
                    diff < 1e-5,
                    "Decode pos {} logits 不一致: diff = {}",
                    pos,
                    diff
                );
            }
        }
        assert_eq!(cache.seq_len, seq.len());
    }

    #[test]
    fn test_sft_masked_gradient_check() {
        let mut rng = Rng::new(2026);
        let mut model = MiniGpt::new(6, 6, 8, 2, 2, &mut rng);
        let seqs = vec![vec![1, 4, 2, 5, 4, 0]];
        let targets = vec![4, 2, 5, 4, 3, 0];
        // 前 2 个是提问（mask=0），中间 3 个是回答+<|eos|>（mask=1），最后 1 个是 padding（mask=0）
        let mask = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

        let compute_loss = |m: &MiniGpt| -> f32 {
            let mut tape = Tape::new();
            let ids = m.forward(&mut tape, &seqs);
            let loss_id = tape.masked_cross_entropy(ids.logits, &targets, &mask);
            tape.data(loss_id)[0]
        };

        let mut tape = Tape::new();
        let ids = model.forward(&mut tape, &seqs);
        let loss_id = tape.masked_cross_entropy(ids.logits, &targets, &mask);
        tape.backward(loss_id);

        let g_tok = tape.grad(ids.tok_emb).to_vec();
        let g_wq0 = tape.grad(ids.blocks[0].wq).to_vec();
        let g_ffn = tape.grad(ids.blocks[1].ffn_w1).to_vec();
        let g_wlm = tape.grad(ids.w_lm).to_vec();

        let eps = 1e-3f32;
        let check =
            |name: &str, analytic: &[f32], getter: fn(&mut MiniGpt) -> &mut [f32], m: &mut MiniGpt| {
                let mut max_rel = 0.0f32;
                for i in 0..analytic.len() {
                    let orig = getter(m)[i];
                    getter(m)[i] = orig + eps;
                    let lp = compute_loss(m);
                    getter(m)[i] = orig - eps;
                    let lm = compute_loss(m);
                    getter(m)[i] = orig;
                    let num = (lp - lm) / (2.0 * eps);
                    let rel = (num - analytic[i]).abs() / (num.abs() + analytic[i].abs()).max(1.0);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
                assert!(
                    max_rel < 3e-3,
                    "SFT Masked 梯度校验失败 {}: max_rel = {}",
                    name,
                    max_rel
                );
            };

        check("tok_emb", &g_tok, |m| &mut m.tok_emb.data, &mut model);
        check("blk0.wq", &g_wq0, |m| &mut m.blocks[0].wq.data, &mut model);
        check("blk1.ffn_w1", &g_ffn, |m| &mut m.blocks[1].ffn_w1.data, &mut model);
        check("w_lm", &g_wlm, |m| &mut m.w_lm.data, &mut model);
    }

    #[test]
    fn test_checkpoint_save_and_load_roundtrip() {
        let mut rng = Rng::new(999);
        let tok = ChatTokenizer::from_texts(&["床前明月光", "你好！"]);
        let model = MiniGpt::new(tok.vocab_size(), 16, 8, 2, 2, &mut rng);

        let tmp_path = env::temp_dir().join("learn_ai_step05_ckpt_test.mgpt");
        let written = save_checkpoint(&tmp_path, &tok, &model).expect("保存 checkpoint 失败");
        assert!(written > 100);

        let (tok2, model2) = load_checkpoint(&tmp_path).expect("加载 checkpoint 失败");
        let _ = fs::remove_file(&tmp_path);

        assert_eq!(tok.id_to_token, tok2.id_to_token);
        assert_eq!(model.num_params(), model2.num_params());

        let prompt = tok.encode_prompt("你好");
        let (z1, _) = model.prefill(&prompt);
        let (z2, _) = model2.prefill(&prompt);
        assert_eq!(z1, z2, "保存并重载后的模型 logits 必须逐位完全一致！");
    }
}
