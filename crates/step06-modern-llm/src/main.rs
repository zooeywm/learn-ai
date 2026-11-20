//! Step 06: 现代工业界 LLM 架构四大升级（LLaMA 3 / DeepSeek / Qwen 同款底座）
//!
//! 把 Step 04/05 的 2019 GPT-2 骨架，一口气升级为当今最主流的现代 Decoder-Only Transformer：
//!   1. RMSNorm（取代 LayerNorm）：去掉均值中心化与 beta 偏置，只按均方根缩放，更快更稳；
//!   2. RoPE 旋转位置编码（彻底消灭绝对 pos_emb 参数表！）：
//!      直接在每一层把 Q / K 向量的相邻两维旋转 m·θ_k 角度，使 Q·K 点积只依赖相对距离 (m - n)；
//!   3. GQA 分组查询注意力（Grouped-Query Attention，取代 MHA）：
//!      4 个 Query 头共享 2 个 KV 头（group_size = 2），KV Cache 显存直接减半（省 50%！）；
//!   4. SwiGLU 门控前馈网络（取代 GELU FFN）：
//!      SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down，无偏置三矩阵门控，表达力更强！
//!
//! 同时保留 Step 05 的完整工程链路：
//!   - 从 `data/poems.txt` 预训练 -> 从 `data/sft_poems.jsonl` 做 Assistant-Only Loss Mask SFT
//!   - GQA + RoPE 真 Prefill / Decode KV Cache（缓存体量比 Step 05 直接减半！）
//!   - 零依赖二进制模型存档 `checkpoints/step06-modern.mllm` + 控制台流式对话 REPL（新增 `/debug` 透视每步 Top-3 概率！）

use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, f32::consts::PI};

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

    fn encode_prompt(&self, question: &str) -> Vec<usize> {
        self.encode_prompt_checked(question).0
    }

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

    fn encode_chat_sample(&self, question: &str, answer: &str) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let mut tokens = self.encode_prompt(question);
        let prompt_len = tokens.len();
        tokens.extend(self.encode_plain(answer));
        tokens.push(EOS_ID);

        let seq_len = tokens.len() - 1;
        let input_ids = tokens[..seq_len].to_vec();
        let target_ids = tokens[1..].to_vec();
        let mut loss_mask = vec![0.0f32; seq_len];
        for m in loss_mask.iter_mut().skip(prompt_len - 1) {
            *m = 1.0;
        }
        (input_ids, target_ids, loss_mask)
    }

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
// 2. RoPE 旋转位置编码预计算表（无任何可学习参数！）
// ============================================================================

struct RopeTable {
    d_head: usize,
    cos: Vec<f32>, // (max_seq_len, d_head / 2)
    sin: Vec<f32>, // (max_seq_len, d_head / 2)
}

impl RopeTable {
    fn new(max_seq_len: usize, d_head: usize, base: f32) -> Self {
        assert_eq!(d_head % 2, 0, "RoPE 要求 d_head 为偶数");
        let half = d_head / 2;
        let mut cos = vec![0.0f32; max_seq_len * half];
        let mut sin = vec![0.0f32; max_seq_len * half];
        for pos in 0..max_seq_len {
            for k in 0..half {
                let freq = 1.0 / base.powf((2 * k) as f32 / d_head as f32);
                let theta = pos as f32 * freq;
                cos[pos * half + k] = theta.cos();
                sin[pos * half + k] = theta.sin();
            }
        }
        Self { d_head, cos, sin }
    }

    /// 对单个向量（长度 = n_heads * d_head）在位置 `pos` 就地施加 RoPE 旋转
    fn apply_vec_inplace(&self, v: &mut [f32], n_heads: usize, pos: usize) {
        let half = self.d_head / 2;
        let base_off = pos * half;
        for h in 0..n_heads {
            let off = h * self.d_head;
            for k in 0..half {
                let c = self.cos[base_off + k];
                let s = self.sin[base_off + k];
                let x0 = v[off + 2 * k];
                let x1 = v[off + 2 * k + 1];
                v[off + 2 * k] = x0 * c - x1 * s;
                v[off + 2 * k + 1] = x0 * s + x1 * c;
            }
        }
    }
}

// ============================================================================
// 3. 现代版 Arena-Tape 自动求导引擎（新增 RmsNorm / Rope / CausalGqa / SwiGlu）
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorId(usize);

enum Op {
    Leaf,
    Add(TensorId, TensorId),
    MatMul(TensorId, TensorId),
    /// 纯词嵌入查找（注意：因为有了 RoPE，不再需要 pos_emb 绝对位置表！）
    EmbedTok {
        tok_emb: TensorId,
        seqs: Vec<Vec<usize>>,
    },
    /// 现代升级 1：RMSNorm(x) = x / rms(x) * gamma（无均值减法、无 beta 偏置！）
    RmsNorm {
        x: TensorId,
        gamma: TensorId,
        inv_rms: Vec<f32>,
    },
    /// 现代升级 2：RoPE 旋转位置编码（正交旋转矩阵，反向即逆旋转 -θ！）
    Rope {
        x: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_heads: usize,
        d_head: usize,
        cos: Vec<f32>,
        sin: Vec<f32>,
    },
    /// 现代升级 3：GQA 分组查询因果注意力（n_q_head 个 Q 头分组共享 n_kv_head 个 KV 头）
    CausalGqa {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_q_head: usize,
        n_kv_head: usize,
        attn_weights: Vec<f32>,
    },
    /// 现代升级 4：SwiGLU 门控激活 = SiLU(gate) ⊙ up
    SwiGlu {
        gate: TensorId,
        up: TensorId,
    },
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

    fn embed_tok(&mut self, tok_emb: TensorId, seqs: &[Vec<usize>]) -> TensorId {
        let (_v, d) = self.shape(tok_emb);
        let b_cnt = seqs.len();
        let t_len = seqs[0].len();
        let tok_data = &self.nodes[tok_emb.0].data;
        let mut out = vec![0.0f32; b_cnt * t_len * d];
        for (b, seq) in seqs.iter().enumerate() {
            for (t, &tok_id) in seq.iter().enumerate() {
                let out_off = (b * t_len + t) * d;
                let tok_off = tok_id * d;
                out[out_off..out_off + d].copy_from_slice(&tok_data[tok_off..tok_off + d]);
            }
        }
        self.push(
            b_cnt * t_len,
            d,
            out,
            Op::EmbedTok {
                tok_emb,
                seqs: seqs.to_vec(),
            },
        )
    }

    /// 现代升级 1：RMSNorm 前向
    fn rms_norm(&mut self, x: TensorId, gamma: TensorId) -> TensorId {
        let (rows, cols) = self.shape(x);
        assert_eq!(self.shape(gamma), (1, cols));
        let x_data = &self.nodes[x.0].data;
        let g_data = &self.nodes[gamma.0].data;
        let eps = 1e-5f32;
        let inv_d = 1.0 / cols as f32;

        let mut inv_rms = vec![0.0f32; rows];
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let off = r * cols;
            let mut sq_sum = 0.0f32;
            for c in 0..cols {
                let v = x_data[off + c];
                sq_sum += v * v;
            }
            let irms = 1.0 / (sq_sum * inv_d + eps).sqrt();
            inv_rms[r] = irms;
            for c in 0..cols {
                out[off + c] = x_data[off + c] * irms * g_data[c];
            }
        }
        self.push(rows, cols, out, Op::RmsNorm { x, gamma, inv_rms })
    }

    /// 现代升级 2：RoPE 旋转位置编码前向（支持任意头数：Q 用 n_q_head，K 用 n_kv_head）
    fn rope(
        &mut self,
        x: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_heads: usize,
        rope: &RopeTable,
    ) -> TensorId {
        let (rows, cols) = self.shape(x);
        let d_head = rope.d_head;
        assert_eq!(rows, batch_size * seq_len);
        assert_eq!(cols, n_heads * d_head);
        let half = d_head / 2;

        let x_data = &self.nodes[x.0].data;
        let mut out = vec![0.0f32; rows * cols];
        for b in 0..batch_size {
            for t in 0..seq_len {
                let row_off = (b * seq_len + t) * cols;
                let cs_off = t * half;
                for h in 0..n_heads {
                    let h_off = row_off + h * d_head;
                    for k in 0..half {
                        let c = rope.cos[cs_off + k];
                        let s = rope.sin[cs_off + k];
                        let x0 = x_data[h_off + 2 * k];
                        let x1 = x_data[h_off + 2 * k + 1];
                        out[h_off + 2 * k] = x0 * c - x1 * s;
                        out[h_off + 2 * k + 1] = x0 * s + x1 * c;
                    }
                }
            }
        }
        self.push(
            rows,
            cols,
            out,
            Op::Rope {
                x,
                batch_size,
                seq_len,
                n_heads,
                d_head,
                cos: rope.cos[..seq_len * half].to_vec(),
                sin: rope.sin[..seq_len * half].to_vec(),
            },
        )
    }

    /// 现代升级 4：SwiGLU 门控算子 = SiLU(gate) ⊙ up
    fn swiglu(&mut self, gate: TensorId, up: TensorId) -> TensorId {
        let (r, c) = self.shape(gate);
        assert_eq!(self.shape(up), (r, c));
        let g_data = &self.nodes[gate.0].data;
        let u_data = &self.nodes[up.0].data;
        let mut out = vec![0.0f32; r * c];
        for i in 0..out.len() {
            let g = g_data[i];
            let sig = 1.0 / (1.0 + (-g).exp());
            let silu = g * sig;
            out[i] = silu * u_data[i];
        }
        self.push(r, c, out, Op::SwiGlu { gate, up })
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

    /// 现代升级 3：GQA 分组查询因果注意力
    ///   q: (B*T, n_q_head * d_head)
    ///   k, v: (B*T, n_kv_head * d_head)
    ///   每个 KV 头被 `group_size = n_q_head / n_kv_head` 个 Query 头共享！
    fn causal_gqa(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_q_head: usize,
        n_kv_head: usize,
    ) -> TensorId {
        let (rows, d_model) = self.shape(q);
        let (k_rows, d_kv) = self.shape(k);
        assert_eq!(rows, batch_size * seq_len);
        assert_eq!(k_rows, rows);
        assert_eq!(self.shape(v), (rows, d_kv));
        assert_eq!(d_model % n_q_head, 0);
        assert_eq!(n_q_head % n_kv_head, 0);
        let d_head = d_model / n_q_head;
        assert_eq!(d_kv, n_kv_head * d_head);
        let group_size = n_q_head / n_kv_head;
        let scale = 1.0 / (d_head as f32).sqrt();

        let q_data = &self.nodes[q.0].data;
        let k_data = &self.nodes[k.0].data;
        let v_data = &self.nodes[v.0].data;
        let mut attn_weights = vec![0.0f32; batch_size * n_q_head * seq_len * seq_len];
        let mut out = vec![0.0f32; rows * d_model];
        let mut row_scores = vec![0.0f32; seq_len];

        for b in 0..batch_size {
            for h in 0..n_q_head {
                let kv_h = h / group_size;
                let q_col = h * d_head;
                let kv_col = kv_h * d_head;
                let w_base = (b * n_q_head + h) * seq_len * seq_len;
                for i in 0..seq_len {
                    let qi_off = (b * seq_len + i) * d_model + q_col;
                    let mut max_s = f32::NEG_INFINITY;
                    for j in 0..=i {
                        let kj_off = (b * seq_len + j) * d_kv + kv_col;
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
                    let out_off = (b * seq_len + i) * d_model + q_col;
                    for j in 0..=i {
                        let w = attn_weights[w_base + i * seq_len + j];
                        let vj_off = (b * seq_len + j) * d_kv + kv_col;
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
            Op::CausalGqa {
                q,
                k,
                v,
                batch_size,
                seq_len,
                n_q_head,
                n_kv_head,
                attn_weights,
            },
        )
    }

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

                Op::EmbedTok { tok_emb, seqs } => {
                    let tok_emb = *tok_emb;
                    let seqs = seqs.clone();
                    let (_, d) = self.shape(tok_emb);
                    let t_len = seqs[0].len();
                    for (b, seq) in seqs.iter().enumerate() {
                        for (t, &tok_id) in seq.iter().enumerate() {
                            let g_off = (b * t_len + t) * d;
                            for c in 0..d {
                                self.nodes[tok_emb.0].grad[tok_id * d + c] += d_out[g_off + c];
                            }
                        }
                    }
                }

                Op::RmsNorm { x, gamma, inv_rms } => {
                    let (x, gamma) = (*x, *gamma);
                    let inv_rms = inv_rms.clone();
                    let (rows, cols) = self.shape(x);
                    let x_data = self.nodes[x.0].data.clone();
                    let g_data = self.nodes[gamma.0].data.clone();
                    let inv_d = 1.0 / cols as f32;

                    for r in 0..rows {
                        let off = r * cols;
                        let irms = inv_rms[r];
                        let mut dot = 0.0f32;
                        for c in 0..cols {
                            let xh = x_data[off + c] * irms;
                            let dy = d_out[off + c];
                            self.nodes[gamma.0].grad[c] += dy * xh;
                            dot += (dy * g_data[c]) * xh;
                        }
                        let mean_dot = dot * inv_d;
                        for c in 0..cols {
                            let xh = x_data[off + c] * irms;
                            let g = d_out[off + c] * g_data[c];
                            self.nodes[x.0].grad[off + c] += irms * (g - xh * mean_dot);
                        }
                    }
                }

                Op::Rope {
                    x,
                    batch_size,
                    seq_len,
                    n_heads,
                    d_head,
                    cos,
                    sin,
                } => {
                    // 正交旋转矩阵 R(θ) 的反向传播正是 R(-θ) = R(θ)^T！
                    let (x, batch_size, seq_len, n_heads, d_head) =
                        (*x, *batch_size, *seq_len, *n_heads, *d_head);
                    let cos = cos.clone();
                    let sin = sin.clone();
                    let cols = n_heads * d_head;
                    let half = d_head / 2;

                    for b in 0..batch_size {
                        for t in 0..seq_len {
                            let row_off = (b * seq_len + t) * cols;
                            let cs_off = t * half;
                            for h in 0..n_heads {
                                let h_off = row_off + h * d_head;
                                for k in 0..half {
                                    let c = cos[cs_off + k];
                                    let s = sin[cs_off + k];
                                    let dy0 = d_out[h_off + 2 * k];
                                    let dy1 = d_out[h_off + 2 * k + 1];
                                    self.nodes[x.0].grad[h_off + 2 * k] += dy0 * c + dy1 * s;
                                    self.nodes[x.0].grad[h_off + 2 * k + 1] += -dy0 * s + dy1 * c;
                                }
                            }
                        }
                    }
                }

                Op::SwiGlu { gate, up } => {
                    let (gate, up) = (*gate, *up);
                    let g_data = self.nodes[gate.0].data.clone();
                    let u_data = self.nodes[up.0].data.clone();
                    for i in 0..d_out.len() {
                        let g = g_data[i];
                        let u = u_data[i];
                        let sig = 1.0 / (1.0 + (-g).exp());
                        let silu = g * sig;
                        let d_silu = sig * (1.0 + g * (1.0 - sig));
                        let dy = d_out[i];
                        self.nodes[gate.0].grad[i] += dy * u * d_silu;
                        self.nodes[up.0].grad[i] += dy * silu;
                    }
                }

                Op::CausalGqa {
                    q,
                    k,
                    v,
                    batch_size,
                    seq_len,
                    n_q_head,
                    n_kv_head,
                    attn_weights,
                } => {
                    let (q, k, v) = (*q, *k, *v);
                    let (batch_size, seq_len, n_q_head, n_kv_head) =
                        (*batch_size, *seq_len, *n_q_head, *n_kv_head);
                    let weights = attn_weights.clone();
                    let (_, d_model) = self.shape(q);
                    let (_, d_kv) = self.shape(k);
                    let d_head = d_model / n_q_head;
                    let group_size = n_q_head / n_kv_head;
                    let scale = 1.0 / (d_head as f32).sqrt();

                    let q_data = self.nodes[q.0].data.clone();
                    let k_data = self.nodes[k.0].data.clone();
                    let v_data = self.nodes[v.0].data.clone();
                    let mut dq = vec![0.0f32; q_data.len()];
                    let mut dk = vec![0.0f32; k_data.len()];
                    let mut dv = vec![0.0f32; v_data.len()];
                    let mut dw_row = vec![0.0f32; seq_len];

                    for b in 0..batch_size {
                        for h in 0..n_q_head {
                            let kv_h = h / group_size;
                            let q_col = h * d_head;
                            let kv_col = kv_h * d_head;
                            let w_base = (b * n_q_head + h) * seq_len * seq_len;
                            for i in 0..seq_len {
                                let q_off = (b * seq_len + i) * d_model + q_col;
                                let mut w_dot_dw = 0.0f32;
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let vj_off = (b * seq_len + j) * d_kv + kv_col;
                                    let mut dw_ij = 0.0f32;
                                    for d in 0..d_head {
                                        let g = d_out[q_off + d];
                                        dv[vj_off + d] += w_ij * g;
                                        dw_ij += g * v_data[vj_off + d];
                                    }
                                    dw_row[j] = dw_ij;
                                    w_dot_dw += w_ij * dw_ij;
                                }
                                for j in 0..=i {
                                    let w_ij = weights[w_base + i * seq_len + j];
                                    let ds = w_ij * (dw_row[j] - w_dot_dw) * scale;
                                    let kj_off = (b * seq_len + j) * d_kv + kv_col;
                                    for d in 0..d_head {
                                        dq[q_off + d] += ds * k_data[kj_off + d];
                                        dk[kj_off + d] += ds * q_data[q_off + d];
                                    }
                                }
                            }
                        }
                    }

                    for idx in 0..dq.len() {
                        self.nodes[q.0].grad[idx] += dq[idx];
                    }
                    for idx in 0..dk.len() {
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
// 4. 参数、AdamW 与现代 LLaMA/DeepSeek 风格 Transformer Block + 50% 省显存 GQA KV Cache
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

fn rms_norm_vec(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let d = x.len();
    let sq_mean = x.iter().map(|&v| v * v).sum::<f32>() / d as f32;
    let inv_rms = 1.0 / (sq_mean + 1e-5f32).sqrt();
    (0..d).map(|c| x[c] * inv_rms * gamma[c]).collect()
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

/// GQA KV Cache：每层每个 token 只存 `d_kv = n_kv_head * d_head` 维，比 MHA 直接省 50% 显存！
struct LayerKv {
    k: Vec<f32>, // (seq_len, d_kv)
    v: Vec<f32>, // (seq_len, d_kv)
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

    /// 返回当前 KV Cache 占用的总浮点数个数
    fn total_floats(&self) -> usize {
        self.layers.iter().map(|l| l.k.len() + l.v.len()).sum()
    }
}

/// 现代 LLaMA / DeepSeek / Qwen 风格 Transformer Block：
///   - 全程无偏置（No Bias）！
///   - Pre-RMSNorm + RoPE + GQA + SwiGLU
struct ModernBlock {
    rms1_gamma: Param, // (1, D)
    wq: Param,         // (D, n_q_head * d_head = D)
    wk: Param,         // (D, n_kv_head * d_head = D_KV)  <-- GQA 减半！
    wv: Param,         // (D, n_kv_head * d_head = D_KV)  <-- GQA 减半！
    wo: Param,         // (D, D)
    rms2_gamma: Param, // (1, D)
    w_gate: Param,     // (D, D_FF)                       <-- SwiGLU 门控路
    w_up: Param,       // (D, D_FF)                       <-- SwiGLU 内容路
    w_down: Param,     // (D_FF, D)                       <-- SwiGLU 下投影
}

struct ModernBlockIds {
    rms1_gamma: TensorId,
    wq: TensorId,
    wk: TensorId,
    wv: TensorId,
    wo: TensorId,
    rms2_gamma: TensorId,
    w_gate: TensorId,
    w_up: TensorId,
    w_down: TensorId,
}

impl ModernBlock {
    fn new(d_model: usize, d_kv: usize, d_ff: usize, n_layer: usize, rng: &mut Rng) -> Self {
        let std = (1.0 / d_model as f32).sqrt();
        let res_std = std / (2.0 * n_layer as f32).sqrt();
        Self {
            rms1_gamma: Param::ones(1, d_model),
            wq: Param::randn(d_model, d_model, std, rng, true),
            wk: Param::randn(d_model, d_kv, std, rng, true),
            wv: Param::randn(d_model, d_kv, std, rng, true),
            wo: Param::randn(d_model, d_model, res_std, rng, true),
            rms2_gamma: Param::ones(1, d_model),
            w_gate: Param::randn(d_model, d_ff, std, rng, true),
            w_up: Param::randn(d_model, d_ff, std, rng, true),
            w_down: Param::randn(d_ff, d_model, res_std, rng, true),
        }
    }

    fn num_params(&self) -> usize {
        self.rms1_gamma.data.len()
            + self.wq.data.len()
            + self.wk.data.len()
            + self.wv.data.len()
            + self.wo.data.len()
            + self.rms2_gamma.data.len()
            + self.w_gate.data.len()
            + self.w_up.data.len()
            + self.w_down.data.len()
    }

    fn reset_optimizer(&mut self) {
        self.rms1_gamma.reset_optimizer();
        self.wq.reset_optimizer();
        self.wk.reset_optimizer();
        self.wv.reset_optimizer();
        self.wo.reset_optimizer();
        self.rms2_gamma.reset_optimizer();
        self.w_gate.reset_optimizer();
        self.w_up.reset_optimizer();
        self.w_down.reset_optimizer();
    }

    fn forward(
        &self,
        tape: &mut Tape,
        x: TensorId,
        batch_size: usize,
        seq_len: usize,
        n_q_head: usize,
        n_kv_head: usize,
        rope: &RopeTable,
    ) -> (TensorId, ModernBlockIds) {
        let rms1_gamma = tape.param(&self.rms1_gamma);
        let wq = tape.param(&self.wq);
        let wk = tape.param(&self.wk);
        let wv = tape.param(&self.wv);
        let wo = tape.param(&self.wo);

        let h1 = tape.rms_norm(x, rms1_gamma);
        let q_raw = tape.matmul(h1, wq);
        let k_raw = tape.matmul(h1, wk);
        let v = tape.matmul(h1, wv);

        // 对 Q 和 K 分别施加 RoPE 旋转位置编码！
        let q = tape.rope(q_raw, batch_size, seq_len, n_q_head, rope);
        let k = tape.rope(k_raw, batch_size, seq_len, n_kv_head, rope);

        let attn = tape.causal_gqa(q, k, v, batch_size, seq_len, n_q_head, n_kv_head);
        let attn_out = tape.matmul(attn, wo);
        let x1 = tape.add(x, attn_out);

        let rms2_gamma = tape.param(&self.rms2_gamma);
        let w_gate = tape.param(&self.w_gate);
        let w_up = tape.param(&self.w_up);
        let w_down = tape.param(&self.w_down);

        let h2 = tape.rms_norm(x1, rms2_gamma);
        let gate = tape.matmul(h2, w_gate);
        let up = tape.matmul(h2, w_up);
        let act = tape.swiglu(gate, up);
        let down = tape.matmul(act, w_down);
        let x2 = tape.add(x1, down);

        (
            x2,
            ModernBlockIds {
                rms1_gamma,
                wq,
                wk,
                wv,
                wo,
                rms2_gamma,
                w_gate,
                w_up,
                w_down,
            },
        )
    }

    /// GQA + RoPE 单步 KV Cache 推理：
    ///   只计算当前 token 的 q_new / k_new / v_new，对 q_new / k_new 施加 pos 位置旋转，
    ///   然后把减半尺寸的 k_new / v_new（长度仅 D_KV）追加到缓存库！
    fn forward_cached_step(
        &self,
        x: &[f32],
        pos: usize,
        n_q_head: usize,
        n_kv_head: usize,
        rope: &RopeTable,
        layer_kv: &mut LayerKv,
    ) -> Vec<f32> {
        let d_model = x.len();
        let d_head = d_model / n_q_head;
        let d_kv = n_kv_head * d_head;
        let group_size = n_q_head / n_kv_head;
        let scale = 1.0 / (d_head as f32).sqrt();

        let h1 = rms_norm_vec(x, &self.rms1_gamma.data);
        let mut q = matvec_row(&h1, &self.wq);
        let mut k_new = matvec_row(&h1, &self.wk);
        let v_new = matvec_row(&h1, &self.wv);

        rope.apply_vec_inplace(&mut q, n_q_head, pos);
        rope.apply_vec_inplace(&mut k_new, n_kv_head, pos);

        layer_kv.k.extend_from_slice(&k_new);
        layer_kv.v.extend_from_slice(&v_new);
        let total_t = layer_kv.k.len() / d_kv;

        let mut attn = vec![0.0f32; d_model];
        let mut scores = vec![0.0f32; total_t];
        for h in 0..n_q_head {
            let kv_h = h / group_size;
            let q_col = h * d_head;
            let kv_col = kv_h * d_head;
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..total_t {
                let k_off = j * d_kv + kv_col;
                let mut dot = 0.0f32;
                for d in 0..d_head {
                    dot += q[q_col + d] * layer_kv.k[k_off + d];
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
                let v_off = j * d_kv + kv_col;
                for d in 0..d_head {
                    attn[q_col + d] += w * layer_kv.v[v_off + d];
                }
            }
        }

        let attn_out = matvec_row(&attn, &self.wo);
        let mut x1 = vec![0.0f32; d_model];
        for c in 0..d_model {
            x1[c] = x[c] + attn_out[c];
        }

        let h2 = rms_norm_vec(&x1, &self.rms2_gamma.data);
        let gate = matvec_row(&h2, &self.w_gate);
        let up = matvec_row(&h2, &self.w_up);
        let mut act = vec![0.0f32; gate.len()];
        for i in 0..act.len() {
            let g = gate[i];
            let silu = g / (1.0 + (-g).exp());
            act[i] = silu * up[i];
        }
        let down = matvec_row(&act, &self.w_down);
        let mut x2 = vec![0.0f32; d_model];
        for c in 0..d_model {
            x2[c] = x1[c] + down[c];
        }
        x2
    }

    fn param_grad_pairs<'a>(&'a mut self, ids: &ModernBlockIds) -> [(&'a mut Param, TensorId); 9] {
        [
            (&mut self.rms1_gamma, ids.rms1_gamma),
            (&mut self.wq, ids.wq),
            (&mut self.wk, ids.wk),
            (&mut self.wv, ids.wv),
            (&mut self.wo, ids.wo),
            (&mut self.rms2_gamma, ids.rms2_gamma),
            (&mut self.w_gate, ids.w_gate),
            (&mut self.w_up, ids.w_up),
            (&mut self.w_down, ids.w_down),
        ]
    }
}

struct ModernLlm {
    vocab_size: usize,
    max_seq_len: usize,
    d_model: usize,
    n_q_head: usize,
    n_kv_head: usize,
    d_ff: usize,
    rope: RopeTable,
    // 注意：彻底没有 pos_emb 参数表了！
    tok_emb: Param,
    blocks: Vec<ModernBlock>,
    rms_f_gamma: Param,
    w_lm: Param,
}

struct ModernForwardIds {
    tok_emb: TensorId,
    blocks: Vec<ModernBlockIds>,
    rms_f_gamma: TensorId,
    w_lm: TensorId,
    logits: TensorId,
}

impl ModernLlm {
    fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_q_head: usize,
        n_kv_head: usize,
        d_ff: usize,
        n_layer: usize,
        rng: &mut Rng,
    ) -> Self {
        assert_eq!(d_model % n_q_head, 0);
        assert_eq!(n_q_head % n_kv_head, 0);
        let d_head = d_model / n_q_head;
        let d_kv = n_kv_head * d_head;
        let std = (1.0 / d_model as f32).sqrt();
        let blocks = (0..n_layer)
            .map(|_| ModernBlock::new(d_model, d_kv, d_ff, n_layer, rng))
            .collect();
        Self {
            vocab_size,
            max_seq_len,
            d_model,
            n_q_head,
            n_kv_head,
            d_ff,
            rope: RopeTable::new(max_seq_len, d_head, 10000.0),
            tok_emb: Param::randn(vocab_size, d_model, 0.2, rng, true),
            blocks,
            rms_f_gamma: Param::ones(1, d_model),
            w_lm: Param::randn(d_model, vocab_size, std, rng, true),
        }
    }

    fn num_params(&self) -> usize {
        let block_sum: usize = self.blocks.iter().map(|b| b.num_params()).sum();
        self.tok_emb.data.len() + block_sum + self.rms_f_gamma.data.len() + self.w_lm.data.len()
    }

    fn reset_optimizer(&mut self) {
        self.tok_emb.reset_optimizer();
        for b in &mut self.blocks {
            b.reset_optimizer();
        }
        self.rms_f_gamma.reset_optimizer();
        self.w_lm.reset_optimizer();
    }

    fn forward(&self, tape: &mut Tape, seqs: &[Vec<usize>]) -> ModernForwardIds {
        let batch_size = seqs.len();
        let seq_len = seqs[0].len();
        assert!(seq_len <= self.max_seq_len);

        let tok_emb = tape.param(&self.tok_emb);
        let mut x = tape.embed_tok(tok_emb, seqs);

        let mut block_ids = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let (nx, bids) = blk.forward(
                tape,
                x,
                batch_size,
                seq_len,
                self.n_q_head,
                self.n_kv_head,
                &self.rope,
            );
            x = nx;
            block_ids.push(bids);
        }

        let rms_f_gamma = tape.param(&self.rms_f_gamma);
        let x_norm = tape.rms_norm(x, rms_f_gamma);
        let w_lm = tape.param(&self.w_lm);
        let logits = tape.matmul(x_norm, w_lm);

        ModernForwardIds {
            tok_emb,
            blocks: block_ids,
            rms_f_gamma,
            w_lm,
            logits,
        }
    }

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

        let mut all_ids = vec![ids.tok_emb, ids.rms_f_gamma, ids.w_lm];
        for b in &ids.blocks {
            all_ids.extend_from_slice(&[
                b.rms1_gamma,
                b.wq,
                b.wk,
                b.wv,
                b.wo,
                b.rms2_gamma,
                b.w_gate,
                b.w_up,
                b.w_down,
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
        for (blk, bids) in self.blocks.iter_mut().zip(ids.blocks.iter()) {
            for (param, tid) in blk.param_grad_pairs(bids) {
                param.adamw_step(tape.grad(tid), clip_scale, lr, wd, step);
            }
        }
        self.rms_f_gamma
            .adamw_step(tape.grad(ids.rms_f_gamma), clip_scale, lr, wd, step);
        self.w_lm
            .adamw_step(tape.grad(ids.w_lm), clip_scale, lr, wd, step);

        (loss, grad_norm)
    }

    fn prefill(&self, prompt_ids: &[usize]) -> (Vec<f32>, KvCache) {
        assert!(!prompt_ids.is_empty());
        assert!(prompt_ids.len() <= self.max_seq_len);
        let mut cache = KvCache::new(self.blocks.len());
        let mut last_logits = Vec::new();
        for &tok_id in prompt_ids {
            last_logits = self.decode_step(tok_id, &mut cache);
        }
        (last_logits, cache)
    }

    fn decode_step(&self, tok_id: usize, cache: &mut KvCache) -> Vec<f32> {
        let pos = cache.seq_len;
        assert!(pos < self.max_seq_len, "超出最大上下文长度 max_seq_len");
        let d = self.d_model;
        let tok_off = tok_id * d;
        // 注意：不再加 pos_emb[pos]！位置信息由每层 Attention 里的 RoPE 注入！
        let mut x = self.tok_emb.data[tok_off..tok_off + d].to_vec();

        for (l, blk) in self.blocks.iter().enumerate() {
            x = blk.forward_cached_step(
                &x,
                pos,
                self.n_q_head,
                self.n_kv_head,
                &self.rope,
                &mut cache.layers[l],
            );
        }
        cache.seq_len += 1;

        let x_norm = rms_norm_vec(&x, &self.rms_f_gamma.data);
        matvec_row(&x_norm, &self.w_lm)
    }

    fn generate_chat_kv(
        &self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> (Vec<usize>, bool, usize, usize) {
        let (mut logits, mut cache) = self.prefill(prompt_ids);
        let mut generated = Vec::new();
        let mut hit_eos = false;
        let max_gen = max_new_tokens.min(self.max_seq_len.saturating_sub(prompt_ids.len()));

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
            if cache.seq_len >= self.max_seq_len {
                break;
            }
            logits = self.decode_step(next_id, &mut cache);
        }
        (generated, hit_eos, cache.seq_len, cache.total_floats())
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
// 5. 零依赖二进制模型存档（MLLM v1：词表 + 现代超参 + 全部无偏置权重）
// ============================================================================

const CKPT_MAGIC: &[u8; 4] = b"MLLM";
const CKPT_VER: u32 = 1;

fn default_checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/step06-modern.mllm")
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

fn save_checkpoint(path: &Path, tok: &ChatTokenizer, model: &ModernLlm) -> io::Result<u64> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(CKPT_MAGIC)?;
    write_u32(&mut w, CKPT_VER)?;
    write_u32(&mut w, model.vocab_size as u32)?;
    write_u32(&mut w, model.max_seq_len as u32)?;
    write_u32(&mut w, model.d_model as u32)?;
    write_u32(&mut w, model.n_q_head as u32)?;
    write_u32(&mut w, model.n_kv_head as u32)?;
    write_u32(&mut w, model.d_ff as u32)?;
    write_u32(&mut w, model.blocks.len() as u32)?;

    for s in &tok.id_to_token {
        let bytes = s.as_bytes();
        write_u32(&mut w, bytes.len() as u32)?;
        w.write_all(bytes)?;
    }

    write_param(&mut w, &model.tok_emb)?;
    for blk in &model.blocks {
        write_param(&mut w, &blk.rms1_gamma)?;
        write_param(&mut w, &blk.wq)?;
        write_param(&mut w, &blk.wk)?;
        write_param(&mut w, &blk.wv)?;
        write_param(&mut w, &blk.wo)?;
        write_param(&mut w, &blk.rms2_gamma)?;
        write_param(&mut w, &blk.w_gate)?;
        write_param(&mut w, &blk.w_up)?;
        write_param(&mut w, &blk.w_down)?;
    }
    write_param(&mut w, &model.rms_f_gamma)?;
    write_param(&mut w, &model.w_lm)?;
    w.flush()?;

    Ok(fs::metadata(path)?.len())
}

fn load_checkpoint(path: &Path) -> io::Result<(ChatTokenizer, ModernLlm)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != CKPT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "无效的模型文件头（期望 MLLM）",
        ));
    }
    let ver = read_u32(&mut r)?;
    if ver != CKPT_VER {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("不支持的模型版本: {}", ver),
        ));
    }

    let vocab_size = read_u32(&mut r)? as usize;
    let max_seq_len = read_u32(&mut r)? as usize;
    let d_model = read_u32(&mut r)? as usize;
    let n_q_head = read_u32(&mut r)? as usize;
    let n_kv_head = read_u32(&mut r)? as usize;
    let d_ff = read_u32(&mut r)? as usize;
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
    let mut model = ModernLlm::new(
        vocab_size,
        max_seq_len,
        d_model,
        n_q_head,
        n_kv_head,
        d_ff,
        n_layer,
        &mut dummy_rng,
    );
    read_param_into(&mut r, &mut model.tok_emb)?;
    for blk in &mut model.blocks {
        read_param_into(&mut r, &mut blk.rms1_gamma)?;
        read_param_into(&mut r, &mut blk.wq)?;
        read_param_into(&mut r, &mut blk.wk)?;
        read_param_into(&mut r, &mut blk.wv)?;
        read_param_into(&mut r, &mut blk.wo)?;
        read_param_into(&mut r, &mut blk.rms2_gamma)?;
        read_param_into(&mut r, &mut blk.w_gate)?;
        read_param_into(&mut r, &mut blk.w_up)?;
        read_param_into(&mut r, &mut blk.w_down)?;
    }
    read_param_into(&mut r, &mut model.rms_f_gamma)?;
    read_param_into(&mut r, &mut model.w_lm)?;

    Ok((tok, model))
}

// ============================================================================
// 6. 控制台流式对话交互（REPL，新增 `/debug` 透视每步 Top-3 概率分布！）
// ============================================================================

fn run_chat_repl(tok: &ChatTokenizer, model: &ModernLlm) {
    println!("\n======================================================================");
    println!("🚀 Step 06 现代 Mini-LLM（RMSNorm + RoPE + GQA + SwiGLU）控制台对话");
    println!("命令：/debug（开关每步 Top-3 概率透视）| /temp <T> | /help | /quit");
    println!("======================================================================");

    let mut temp = 0.0f32;
    let mut debug_probs = false;
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
        if input == "/debug" {
            debug_probs = !debug_probs;
            println!(
                "🔍 每步 Top-3 概率透视已{}",
                if debug_probs { "开启" } else { "关闭" }
            );
            continue;
        }
        if input == "/help" {
            println!("试试输入：你好 / 背一首静夜思 / 背一首春晓 / 红豆生南国的下一句 / 红豆生南国");
            println!("指令：/debug 开关概率透视 | /temp 0.0~2.5 调温度 | /quit 退出");
            continue;
        }
        if let Some(arg) = input.strip_prefix("/temp") {
            match arg.trim().parse::<f32>() {
                Ok(t) if (0.0..=2.5).contains(&t) => {
                    temp = t;
                    println!("✅ 采样温度已调整为 T = {:.2}", temp);
                }
                _ => println!("⚠️ 用法: /temp <0.0 ~ 2.5>"),
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
        if prompt_ids.len() >= model.max_seq_len - 1 {
            prompt_ids.truncate(model.max_seq_len - 2);
            prompt_ids.push(ASST_ID);
        }

        let t_pre = Instant::now();
        let (mut logits, mut cache) = model.prefill(&prompt_ids);
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        print!("助手 > ");
        let _ = stdout.flush();
        let t_dec = Instant::now();
        let max_gen = model.max_seq_len.saturating_sub(prompt_ids.len());
        let mut gen_tokens = 0usize;
        let mut hit_eos = false;
        let mut debug_lines = Vec::new();

        for step_k in 0..max_gen {
            if debug_probs {
                let probs = softmax(&logits);
                let mut idxs: Vec<usize> = (0..probs.len()).collect();
                idxs.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
                let top3: Vec<String> = idxs
                    .iter()
                    .take(3)
                    .map(|&id| format!("{}:{:.1}%", tok.id_to_token[id], probs[id] * 100.0))
                    .collect();
                debug_lines.push(format!("    step {:2} -> [{}]", step_k + 1, top3.join(", ")));
            }
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
            if cache.seq_len >= model.max_seq_len {
                break;
            }
            logits = model.decode_step(next_id, &mut cache);
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let mha_equiv = cache.seq_len * model.blocks.len() * 2 * model.d_model;
        println!(
            "\n  [Prefill {:.2}ms | Decode {:.2}ms ({} tokens) | GQA KV Cache {} floats (vs MHA {} 省 50%!) | hit_eos={}]",
            prefill_ms,
            decode_ms,
            gen_tokens,
            cache.total_floats(),
            mha_equiv,
            hit_eos
        );
        for dl in debug_lines {
            println!("{}", dl);
        }
    }
}

// ============================================================================
// 7. 训练 + 存档 + 对比评测 + CLI 入口
// ============================================================================

fn train_and_save(model_path: &Path) -> (ChatTokenizer, ModernLlm) {
    let poems = load_poems();
    let qa_pairs = load_sft_dataset();

    let mut all_texts: Vec<&str> = vec![poems.as_str()];
    for (q, a) in &qa_pairs {
        all_texts.push(q.as_str());
        all_texts.push(a.as_str());
    }
    let tok = ChatTokenizer::from_texts(&all_texts);
    let pretrain_data = tok.encode_plain(&poems);

    let max_seq_len = 48;
    let pretrain_seq_len = 24;
    let d_model = 32;
    let n_q_head = 4;
    let n_kv_head = 2; // GQA: 4Q 共享 2KV -> KV Cache 显存直接减半！
    let d_ff = 88; // SwiGLU 三矩阵 3 * 32 * 88 ≈ 2 * 32 * 128
    let n_layer = 2;

    println!("=== Step 06: 现代工业界 LLM 架构大升级（LLaMA / DeepSeek / Qwen 同款底座）===");
    println!("四大升级: RMSNorm（无偏置）+ RoPE（零 pos_emb 参数）+ GQA（4Q/2KV 省 50% 缓存）+ SwiGLU");

    let mut rng = Rng::new(42);
    let mut model = ModernLlm::new(
        tok.vocab_size(),
        max_seq_len,
        d_model,
        n_q_head,
        n_kv_head,
        d_ff,
        n_layer,
        &mut rng,
    );
    println!(
        "模型配置: V={}, {}层 ModernBlock (D={}, Q={}头/KV={}头, SwiGLU_FF={}, pos_emb=0) | 总参数量: {}（比 Step 05 的 52,947 更精炼！）",
        tok.vocab_size(),
        n_layer,
        d_model,
        n_q_head,
        n_kv_head,
        d_ff,
        model.num_params()
    );

    // Stage 1: Pre-training
    println!("\n--- Stage 1: 预训练现代底座（唐诗语料，全位置 Loss Mask = 1.0）---");
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
    println!(
        "Stage 1 预训练完成！耗时 {:.2}s",
        t_pt.elapsed().as_secs_f32()
    );

    // Stage 2: SFT 指令微调
    println!(
        "\n--- Stage 2: SFT 指令微调（加载 data/sft_poems.jsonl 共 {} 条问答，Assistant-Only Loss Mask）---",
        qa_pairs.len()
    );
    let encoded_samples: Vec<(Vec<usize>, Vec<usize>, Vec<f32>)> = qa_pairs
        .iter()
        .map(|(q, a)| tok.encode_chat_sample(q, a))
        .collect();

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
    println!(
        "Stage 2 SFT 微调完成！耗时 {:.2}s",
        t_sft.elapsed().as_secs_f32()
    );

    println!("\n=== 现代 Mini-LLM（RMSNorm + RoPE + GQA + SwiGLU）对话实测 ===");
    for q in [
        "你好",
        "背一首静夜思",
        "白日依山尽的下一句",
        "飞流直下三千尺的下一句",
        "背一首春晓",
        "红豆生南国",
    ] {
        let prompt = tok.encode_prompt(q);
        let (reply_ids, hit_eos, kv_len, kv_floats) =
            model.generate_chat_kv(&prompt, 32, 0.0, &mut rng);
        let mha_floats = kv_len * n_layer * 2 * d_model;
        println!(
            "问「{}」\n  回复: {}  [hit_eos={}, GQA KV={} floats vs MHA {} 省 50%!]",
            q,
            tok.decode_reply(&reply_ids),
            hit_eos,
            kv_floats,
            mha_floats
        );
    }

    match save_checkpoint(model_path, &tok, &model) {
        Ok(bytes) => {
            println!(
                "\n💾 现代模型已保存至: {} ({:.1} KB)",
                model_path.display(),
                bytes as f64 / 1024.0
            );
            println!("👉 秒开控制台对话（支持 /debug 透视概率）：");
            println!("   cargo run --release -p step06-modern-llm -- chat");
        }
        Err(e) => eprintln!("⚠️ 保存模型失败 {:?}: {}", model_path, e),
    }

    (tok, model)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut mode = "train";
    let mut chat_after = false;
    let mut model_path = default_checkpoint_path();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "chat" => mode = "chat",
            "train" => mode = "train",
            "--chat" => chat_after = true,
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
                println!("  cargo run --release -p step06-modern-llm                  # 训练并保存现代模型到 checkpoints/step06-modern.mllm");
                println!("  cargo run --release -p step06-modern-llm -- chat          # 毫秒级加载模型进入控制台交互（支持 /debug 概率透视）");
                println!("  cargo run --release -p step06-modern-llm -- train --chat  # 训练保存后直接进入对话");
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
                    "✅ 已加载现代模型: {}（耗时 {:.2} ms | V={} | {} 参数 | GQA {}Q/{}KV）",
                    model_path.display(),
                    t0.elapsed().as_secs_f64() * 1000.0,
                    tok.vocab_size(),
                    model.num_params(),
                    model.n_q_head,
                    model.n_kv_head
                );
                run_chat_repl(&tok, &model);
            }
            Err(e) => {
                eprintln!("❌ 无法加载模型 {}: {}", model_path.display(), e);
                eprintln!("请先运行一次训练：");
                eprintln!("  cargo run --release -p step06-modern-llm -- train --chat");
            }
        }
    } else {
        let (tok, model) = train_and_save(&model_path);
        if chat_after {
            run_chat_repl(&tok, &model);
        }
    }
}

// ============================================================================
// 8. 单元测试：RoPE 相对距离不变性 + GQA KV Cache 等价与省 50% + 全网梯度检验 + 存档校验
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_relative_distance_invariance() {
        // 核心数学定理：RoPE(q, m) · RoPE(k, n) 只与相对距离 (m - n) 有关，
        // 把 m, n 同时平移 +7 步，点积必须精确相等！
        let rope = RopeTable::new(32, 8, 10000.0);
        let q = vec![0.4, -1.1, 0.7, 0.3, -0.5, 1.2, 0.9, -0.2];
        let k = vec![-0.8, 0.6, 1.0, -0.4, 0.2, -0.9, 0.5, 1.1];

        let mut q_a = q.clone();
        let mut k_a = k.clone();
        rope.apply_vec_inplace(&mut q_a, 1, 5);
        rope.apply_vec_inplace(&mut k_a, 1, 2);
        let dot_a: f32 = q_a.iter().zip(k_a.iter()).map(|(a, b)| a * b).sum();

        let mut q_b = q;
        let mut k_b = k;
        rope.apply_vec_inplace(&mut q_b, 1, 12); // 5 + 7
        rope.apply_vec_inplace(&mut k_b, 1, 9);  // 2 + 7
        let dot_b: f32 = q_b.iter().zip(k_b.iter()).map(|(a, b)| a * b).sum();

        assert!(
            (dot_a - dot_b).abs() < 1e-5,
            "RoPE 相对距离不变性失败: dot_a={} vs dot_b={}",
            dot_a,
            dot_b
        );
    }

    #[test]
    fn test_gqa_kv_cache_matches_full_forward_and_halves_memory() {
        let mut rng = Rng::new(777);
        let model = ModernLlm::new(12, 16, 16, 4, 2, 44, 2, &mut rng);
        let seq = vec![1, 5, 7, 2, 9, 4, 3];

        let (prefill_logits, mut cache) = model.prefill(&seq[..4]);
        let mut tape_full = Tape::new();
        let full_ids = model.forward(&mut tape_full, &[seq.clone()]);
        let full_logits = tape_full.data(full_ids.logits);

        for v in 0..model.vocab_size {
            let diff = (prefill_logits[v] - full_logits[3 * model.vocab_size + v]).abs();
            assert!(diff < 1e-5, "GQA Prefill logits 不一致: diff = {}", diff);
        }
        for pos in 4..seq.len() {
            let dec_logits = model.decode_step(seq[pos], &mut cache);
            for v in 0..model.vocab_size {
                let diff = (dec_logits[v] - full_logits[pos * model.vocab_size + v]).abs();
                assert!(diff < 1e-5, "GQA Decode pos {} 不一致: diff = {}", pos, diff);
            }
        }
        // 验证 GQA(4Q/2KV) KV Cache 严格比 MHA(4Q/4KV) 省 50%！
        let mha_floats = seq.len() * model.blocks.len() * 2 * model.d_model;
        assert_eq!(cache.total_floats() * 2, mha_floats);
    }

    #[test]
    fn test_modern_llm_gradient_check() {
        let mut rng = Rng::new(2026);
        let mut model = ModernLlm::new(6, 8, 8, 4, 2, 12, 2, &mut rng);
        let seqs = vec![vec![1, 4, 2, 5, 4, 0]];
        let targets = vec![4, 2, 5, 4, 3, 0];
        let mask = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];

        let compute_loss = |m: &ModernLlm| -> f32 {
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
        let g_rms1 = tape.grad(ids.blocks[0].rms1_gamma).to_vec();
        let g_wq = tape.grad(ids.blocks[0].wq).to_vec();
        let g_wk = tape.grad(ids.blocks[0].wk).to_vec();
        let g_wv = tape.grad(ids.blocks[0].wv).to_vec();
        let g_gate = tape.grad(ids.blocks[1].w_gate).to_vec();
        let g_up = tape.grad(ids.blocks[1].w_up).to_vec();
        let g_down = tape.grad(ids.blocks[1].w_down).to_vec();
        let g_rmsf = tape.grad(ids.rms_f_gamma).to_vec();
        let g_wlm = tape.grad(ids.w_lm).to_vec();

        let eps = 1e-3f32;
        let check = |name: &str,
                     analytic: &[f32],
                     getter: fn(&mut ModernLlm) -> &mut [f32],
                     m: &mut ModernLlm| {
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
                "ModernLlm 参数 {} 相对梯度误差过大: {}",
                name,
                max_rel
            );
        };

        check("tok_emb", &g_tok, |m| &mut m.tok_emb.data, &mut model);
        check("blk0.rms1_gamma", &g_rms1, |m| &mut m.blocks[0].rms1_gamma.data, &mut model);
        check("blk0.wq(RoPE+GQA)", &g_wq, |m| &mut m.blocks[0].wq.data, &mut model);
        check("blk0.wk(RoPE+GQA)", &g_wk, |m| &mut m.blocks[0].wk.data, &mut model);
        check("blk0.wv(GQA)", &g_wv, |m| &mut m.blocks[0].wv.data, &mut model);
        check("blk1.w_gate(SwiGLU)", &g_gate, |m| &mut m.blocks[1].w_gate.data, &mut model);
        check("blk1.w_up(SwiGLU)", &g_up, |m| &mut m.blocks[1].w_up.data, &mut model);
        check("blk1.w_down(SwiGLU)", &g_down, |m| &mut m.blocks[1].w_down.data, &mut model);
        check("rms_f_gamma", &g_rmsf, |m| &mut m.rms_f_gamma.data, &mut model);
        check("w_lm", &g_wlm, |m| &mut m.w_lm.data, &mut model);
    }

    #[test]
    fn test_modern_checkpoint_roundtrip() {
        let mut rng = Rng::new(314);
        let tok = ChatTokenizer::from_texts(&["床前明月光", "你好！"]);
        let model = ModernLlm::new(tok.vocab_size(), 16, 8, 4, 2, 12, 2, &mut rng);

        let tmp = env::temp_dir().join("learn_ai_step06_modern_test.mllm");
        save_checkpoint(&tmp, &tok, &model).expect("保存现代模型失败");
        let (tok2, model2) = load_checkpoint(&tmp).expect("加载现代模型失败");
        let _ = fs::remove_file(&tmp);

        assert_eq!(tok.id_to_token, tok2.id_to_token);
        let prompt = tok.encode_prompt("你好");
        let (z1, _) = model.prefill(&prompt);
        let (z2, _) = model2.prefill(&prompt);
        assert_eq!(z1, z2);
    }
}
