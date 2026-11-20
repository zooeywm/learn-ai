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
use std::f32::consts::PI;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use mini_core::{load_poems, load_sft_dataset, sample_from_logits, softmax, Rng};

const PAD_ID: usize = 0;
const USER_ID: usize = 1;
const ASST_ID: usize = 2;
pub const EOS_ID: usize = 3;

const CONTROL_TOKENS: [&str; 4] = ["<|pad|>", "<|user|>", "<|assistant|>", "<|eos|>"];

/// 分词器负责建立 ID 与原始字节的映射，现代底座负责训练、存档和推理。
pub trait ChatEncoding: Sized {
    const PACKAGE: &'static str;
    const TITLE: &'static str;
    const CHECKPOINT_MAGIC: [u8; 4];
    const TRUNCATE_LONG_PROMPTS: bool = false;
    const CONSTRAIN_UTF8: bool = false;

    fn train(texts: &[&str]) -> Self;
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> (Vec<usize>, Vec<char>);
    fn token_bytes(&self, id: usize) -> &[u8];
    fn save_vocab(&self, writer: &mut impl Write) -> io::Result<()>;
    fn load_vocab(reader: &mut impl Read, vocab_size: usize) -> io::Result<Self>;
}

fn encode_prompt(tokenizer: &impl ChatEncoding, question: &str) -> (Vec<usize>, Vec<char>) {
    let (ids, oov) = tokenizer.encode(question);
    let mut prompt = Vec::with_capacity(ids.len() + 2);
    prompt.push(USER_ID);
    prompt.extend(ids);
    prompt.push(ASST_ID);
    (prompt, oov)
}

fn encode_chat_sample(
    tokenizer: &impl ChatEncoding,
    question: &str,
    answer: &str,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let (mut ids, _) = encode_prompt(tokenizer, question);
    let prompt_len = ids.len();
    ids.extend(tokenizer.encode(answer).0);
    ids.push(EOS_ID);
    let mut mask = vec![1.0; ids.len() - 1];
    mask[..prompt_len - 1].fill(0.0);
    (ids[..ids.len() - 1].to_vec(), ids[1..].to_vec(), mask)
}

pub struct CharTokenizer {
    id_to_token: Vec<String>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    fn from_vocab(id_to_token: Vec<String>) -> Self {
        let char_to_id = id_to_token
            .iter()
            .enumerate()
            .skip(CONTROL_TOKENS.len())
            .filter_map(|(id, token)| token.chars().next().map(|ch| (ch, id)))
            .collect();
        Self {
            id_to_token,
            char_to_id,
        }
    }
}

impl ChatEncoding for CharTokenizer {
    const PACKAGE: &'static str = "step06-modern-llm";
    const TITLE: &'static str = "Step 06 · 现代 Mini-LLM（RMSNorm + RoPE + GQA + SwiGLU）";
    const CHECKPOINT_MAGIC: [u8; 4] = *b"MLLM";
    const TRUNCATE_LONG_PROMPTS: bool = true;

    fn train(texts: &[&str]) -> Self {
        let chars: BTreeSet<char> = texts.iter().flat_map(|text| text.chars()).collect();
        let vocab = CONTROL_TOKENS
            .into_iter()
            .map(str::to_owned)
            .chain(chars.into_iter().map(|ch| ch.to_string()))
            .collect();
        Self::from_vocab(vocab)
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn encode(&self, text: &str) -> (Vec<usize>, Vec<char>) {
        let mut ids = Vec::new();
        let mut oov = Vec::new();
        for ch in text.chars() {
            if let Some(&id) = self.char_to_id.get(&ch) {
                ids.push(id);
            } else {
                oov.push(ch);
            }
        }
        (ids, oov)
    }

    fn token_bytes(&self, id: usize) -> &[u8] {
        self.id_to_token[id].as_bytes()
    }

    fn save_vocab(&self, writer: &mut impl Write) -> io::Result<()> {
        for token in &self.id_to_token {
            write_u32(writer, token.len() as u32)?;
            writer.write_all(token.as_bytes())?;
        }
        Ok(())
    }

    fn load_vocab(reader: &mut impl Read, vocab_size: usize) -> io::Result<Self> {
        let mut vocab = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            let len = read_u32(reader)? as usize;
            let mut bytes = vec![0; len];
            reader.read_exact(&mut bytes)?;
            let token = String::from_utf8(bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            vocab.push(token);
        }
        Ok(Self::from_vocab(vocab))
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
        for (a_row, output) in a_data.chunks_exact(k).zip(out.chunks_exact_mut(n)) {
            for (&a_ip, b_row) in a_row.iter().zip(b_data.chunks_exact(n)) {
                for (value, &b_pj) in output.iter_mut().zip(b_row) {
                    *value += a_ip * b_pj;
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
                    let (_, k) = self.shape(a);
                    let (_, n) = self.shape(b);
                    let a_data = self.nodes[a.0].data.clone();
                    let b_data = self.nodes[b.0].data.clone();
                    for (i, (a_row, output_grad)) in a_data
                        .chunks_exact(k)
                        .zip(d_out.chunks_exact(n))
                        .enumerate()
                    {
                        for (p, (&a_ip, b_row)) in a_row.iter().zip(b_data.chunks_exact(n)).enumerate() {
                            let weight_grad = &mut self.nodes[b.0].grad[p * n..(p + 1) * n];
                            for (grad, &g) in weight_grad.iter_mut().zip(output_grad) {
                                *grad += a_ip * g;
                            }
                            let input_grad: f32 =
                                output_grad.iter().zip(b_row).map(|(&g, &b)| g * b).sum();
                            self.nodes[a.0].grad[i * k + p] += input_grad;
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
    let mut out = vec![0.0f32; w.cols];
    for (&xp, weight_row) in x.iter().zip(w.data.chunks_exact(w.cols)) {
        for (value, &weight) in out.iter_mut().zip(weight_row) {
            *value += xp * weight;
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

struct ChatGeneration {
    ids: Vec<usize>,
    hit_eos: bool,
    cache_len: usize,
    cache_floats: usize,
    decode_time: Duration,
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
        tokenizer: &impl ChatEncoding,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> io::Result<ChatGeneration> {
        let (mut logits, mut cache) = self.prefill(prompt_ids);
        let start = Instant::now();
        let mut generated = Vec::new();
        let mut pending_utf8 = Vec::new();
        let mut hit_eos = false;
        let max_gen = max_new_tokens.min(self.max_seq_len.saturating_sub(prompt_ids.len()));

        for remaining in (0..max_gen).rev() {
            mask_utf8_logits(tokenizer, &mut logits, &pending_utf8, remaining);
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
            if next_id >= CONTROL_TOKENS.len() {
                pending_utf8.extend_from_slice(tokenizer.token_bytes(next_id));
                flush_utf8(&mut pending_utf8, &mut io::sink())?;
            }
            if cache.seq_len >= self.max_seq_len {
                break;
            }
            logits = self.decode_step(next_id, &mut cache);
        }
        Ok(ChatGeneration {
            ids: generated,
            hit_eos,
            cache_len: cache.seq_len,
            cache_floats: cache.total_floats(),
            decode_time: start.elapsed(),
        })
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

const CKPT_VER: u32 = 1;

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

fn save_checkpoint<C: ChatEncoding>(path: &Path, tok: &C, model: &ModernLlm) -> io::Result<u64> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(format!(".{}.tmp", std::process::id()));
    let temporary = PathBuf::from(temporary);
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| {
        write_checkpoint(&mut file, tok, model)?;
        let size = file.metadata()?.len();
        drop(file);
        fs::rename(&temporary, path)?;
        Ok(size)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn write_checkpoint<C: ChatEncoding>(
    file: &mut File,
    tok: &C,
    model: &ModernLlm,
) -> io::Result<()> {
    let mut w = BufWriter::new(file);
    w.write_all(&C::CHECKPOINT_MAGIC)?;
    write_u32(&mut w, CKPT_VER)?;
    write_u32(&mut w, model.vocab_size as u32)?;
    write_u32(&mut w, model.max_seq_len as u32)?;
    write_u32(&mut w, model.d_model as u32)?;
    write_u32(&mut w, model.n_q_head as u32)?;
    write_u32(&mut w, model.n_kv_head as u32)?;
    write_u32(&mut w, model.d_ff as u32)?;
    write_u32(&mut w, model.blocks.len() as u32)?;

    tok.save_vocab(&mut w)?;

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
    w.flush()
}

fn load_checkpoint<C: ChatEncoding>(path: &Path) -> io::Result<(C, ModernLlm)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if magic != C::CHECKPOINT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "模型文件类型与当前分词器不匹配",
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

    let tok = C::load_vocab(&mut r, vocab_size)?;

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

fn token_label(tokenizer: &impl ChatEncoding, id: usize) -> String {
    if let Some(control) = CONTROL_TOKENS.get(id) {
        return (*control).to_owned();
    }
    let bytes = tokenizer.token_bytes(id);
    match str::from_utf8(bytes) {
        Ok(text) if text.chars().any(char::is_whitespace) => format!("{text:?}"),
        Ok(text) => text.to_owned(),
        Err(_) => format!("{bytes:02X?}"),
    }
}

fn decode_reply(tokenizer: &impl ChatEncoding, ids: &[usize]) -> io::Result<String> {
    let bytes: Vec<_> = ids
        .iter()
        .copied()
        .take_while(|&id| id != EOS_ID)
        .filter(|&id| id >= CONTROL_TOKENS.len())
        .flat_map(|id| tokenizer.token_bytes(id).iter().copied())
        .collect();
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn can_finish_utf8(bytes: &[u8], remaining_tokens: usize) -> bool {
    match str::from_utf8(bytes) {
        Ok(_) => true,
        Err(error) if error.error_len().is_none() => {
            let incomplete = &bytes[error.valid_up_to()..];
            let width = incomplete[0].leading_ones() as usize;
            // 单字节 token 足以补齐剩余字节。
            width - incomplete.len() <= remaining_tokens
        }
        Err(_) => false,
    }
}

fn mask_utf8_logits<C: ChatEncoding>(
    tokenizer: &C,
    logits: &mut [f32],
    pending: &[u8],
    remaining_tokens: usize,
) {
    if !C::CONSTRAIN_UTF8 {
        return;
    }
    let mut bytes = pending.to_vec();
    for (id, logit) in logits.iter_mut().enumerate() {
        let allowed = match id {
            EOS_ID => pending.is_empty(),
            PAD_ID | USER_ID | ASST_ID => false,
            _ => {
                bytes.extend_from_slice(tokenizer.token_bytes(id));
                let allowed = can_finish_utf8(&bytes, remaining_tokens);
                bytes.truncate(pending.len());
                allowed
            }
        };
        if !allowed {
            *logit = f32::NEG_INFINITY;
        }
    }
}

fn flush_utf8(pending: &mut Vec<u8>, writer: &mut impl Write) -> io::Result<()> {
    let end = match str::from_utf8(pending) {
        Ok(_) => pending.len(),
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        Err(error) => return Err(io::Error::new(io::ErrorKind::InvalidData, error)),
    };
    writer.write_all(&pending[..end])?;
    pending.drain(..end);
    writer.flush()
}

pub type ChatLookup = fn(&str) -> io::Result<Option<String>>;

fn run_chat_repl<C: ChatEncoding>(
    tok: &C,
    model: &ModernLlm,
    lookup: Option<ChatLookup>,
) -> io::Result<()> {
    println!("\n======================================================================");
    println!("{} · 控制台对话", C::TITLE);
    println!("命令：/debug（开关每步 Top-3 概率透视）| /temp <T> | /help | /quit");
    if lookup.is_some() {
        println!("「诗库 >」为原文查询结果。可用 chat --raw 查看模型生成。");
    }
    println!("======================================================================");

    let mut temp = 0.0f32;
    let mut debug_probs = false;
    let mut rng = Rng::new(2026);
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut lines = stdin.lock().lines();

    loop {
        print!("\n你 > ");
        stdout.flush()?;

        let Some(line) = lines.next().transpose()? else {
            println!("\n已退出对话。");
            break;
        };
        let input = line.trim();
        if line.is_empty() || (C::TRUNCATE_LONG_PROMPTS && input.is_empty()) {
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

        if let Some(answer) = lookup.map_or(Ok(None), |lookup| lookup(input))? {
            println!("诗库 > {answer}");
            continue;
        }

        let question = if C::TRUNCATE_LONG_PROMPTS {
            input
        } else {
            &line
        };
        let (mut prompt_ids, oov) = encode_prompt(tok, question);
        if !oov.is_empty() {
            let uniq: String = oov.iter().collect();
            println!(
                "  [提示: 字符「{}」不在当前字符级词表(V={})中，已跳过]",
                uniq,
                tok.vocab_size()
            );
        }
        if prompt_ids.len() >= model.max_seq_len - 1 {
            if !C::TRUNCATE_LONG_PROMPTS {
                println!(
                    "输入有 {} 个 token，最多允许 {} 个，余下位置用于控制符和回复。",
                    prompt_ids.len() - 2,
                    model.max_seq_len - 4
                );
                continue;
            }
            prompt_ids.truncate(model.max_seq_len - 2);
            prompt_ids.push(ASST_ID);
        }
        if debug_probs {
            let input_tokens: Vec<_> = prompt_ids
                .iter()
                .map(|&id| format!("{id}={}", token_label(tok, id)))
                .collect();
            println!("输入 token  [{}]", input_tokens.join("  "));
        }

        let t_pre = Instant::now();
        let (mut logits, mut cache) = model.prefill(&prompt_ids);
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;

        print!("助手 > ");
        stdout.flush()?;
        let t_dec = Instant::now();
        let max_gen = model.max_seq_len.saturating_sub(prompt_ids.len());
        let mut gen_tokens = 0usize;
        let mut hit_eos = false;
        let mut debug_lines = Vec::new();
        let mut pending_utf8 = Vec::new();

        for step_k in 0..max_gen {
            mask_utf8_logits(tok, &mut logits, &pending_utf8, max_gen - step_k - 1);
            if debug_probs {
                let probs = softmax(&logits);
                let mut idxs: Vec<usize> = (0..probs.len())
                    .filter(|&id| logits[id].is_finite())
                    .collect();
                idxs.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
                let top3: Vec<String> = idxs
                    .iter()
                    .take(3)
                    .map(|&id| format!("{}:{:.1}%", token_label(tok, id), probs[id] * 100.0))
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
            if next_id >= CONTROL_TOKENS.len() {
                pending_utf8.extend_from_slice(tok.token_bytes(next_id));
                flush_utf8(&mut pending_utf8, &mut stdout)?;
            }
            if cache.seq_len >= model.max_seq_len {
                break;
            }
            logits = model.decode_step(next_id, &mut cache);
        }
        debug_assert!(pending_utf8.is_empty());
        let decode_secs = t_dec.elapsed().as_secs_f64();
        let text_tokens = gen_tokens - usize::from(hit_eos);
        let tokens_per_sec = text_tokens as f64 / decode_secs;
        let mha_equiv = cache.seq_len * model.blocks.len() * 2 * model.d_model;
        println!(
            "\n  [Prefill {:.2}ms | Decode {:.2}ms ({} tokens, {:.0} tok/s) | GQA KV Cache {} floats (vs MHA {} 省 50%!) | hit_eos={}]",
            prefill_ms,
            decode_secs * 1000.0,
            text_tokens,
            tokens_per_sec,
            cache.total_floats(),
            mha_equiv,
            hit_eos
        );
        for dl in debug_lines {
            println!("{}", dl);
        }
    }
    Ok(())
}

// ============================================================================
// 7. 训练 + 存档 + 对比评测 + CLI 入口
// ============================================================================

#[derive(Clone, Copy)]
pub struct TrainingConfig {
    pub context: usize,
    pub d_model: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub d_ff: usize,
    pub layers: usize,
    pub pretrain_tokens: usize,
    pub pretrain_steps: usize,
    pub pretrain_batch: usize,
    pub pretrain_lr: f32,
    pub sft_steps: usize,
    pub sft_batch: usize,
    pub sft_lr: f32,
    pub refine_steps: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            context: 48,
            d_model: 32,
            query_heads: 4,
            kv_heads: 2,
            d_ff: 88,
            layers: 2,
            pretrain_tokens: 24,
            pretrain_steps: 900,
            pretrain_batch: 16,
            pretrain_lr: 0.012,
            sft_steps: 360,
            sft_batch: 6,
            sft_lr: 0.008,
            refine_steps: 0,
        }
    }
}

pub struct TrainingData {
    pub poems: String,
    pub qa_pairs: Vec<(String, String)>,
    pub paraphrases: Vec<(String, String)>,
}

impl TrainingData {
    pub fn lesson_data() -> io::Result<Self> {
        Ok(Self {
            poems: load_poems(),
            qa_pairs: load_sft_dataset(),
            paraphrases: Vec::new(),
        })
    }
}

struct SftSchedule {
    steps: usize,
    batch: usize,
    learning_rate: f32,
    bucket_by_length: bool,
}

fn train_sft(
    model: &mut ModernLlm,
    tokenizer: &impl ChatEncoding,
    qa_pairs: &[(String, String)],
    schedule: SftSchedule,
    rng: &mut Rng,
) -> io::Result<()> {
    if qa_pairs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "缺少 SFT 样本"));
    }
    let context = model.max_seq_len;
    let mut encoded_samples: Vec<_> = qa_pairs
        .iter()
        .map(|(question, answer)| encode_chat_sample(tokenizer, question, answer))
        .collect();
    if let Some((input, _, _)) = encoded_samples
        .iter()
        .find(|(input, _, _)| input.len() > context)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("SFT 样本需要 {} 个位置，上下文为 {context}", input.len()),
        ));
    }

    println!(
        "\n--- SFT（{} 条问答 · {} 步 · batch {} · Assistant-Only Loss Mask）---",
        encoded_samples.len(),
        schedule.steps,
        schedule.batch
    );
    if schedule.bucket_by_length {
        encoded_samples.sort_by_key(|(input, _, _)| input.len());
    }
    model.reset_optimizer();
    let t_sft = Instant::now();
    for step in 1..=schedule.steps {
        // 同长度的样本一起训练，减少为最长样本补齐的占位位置。
        let (start, length) = if schedule.bucket_by_length {
            const BUCKET_SIZE: usize = 32;
            let buckets = encoded_samples.len().div_ceil(BUCKET_SIZE);
            let start = rng.next_usize(buckets) * BUCKET_SIZE;
            (start, (encoded_samples.len() - start).min(BUCKET_SIZE))
        } else {
            (0, encoded_samples.len())
        };
        let mut chosen = Vec::with_capacity(schedule.batch);
        let mut max_len = 0usize;
        for _ in 0..schedule.batch {
            let idx = start + (rng.next_u64() as usize) % length;
            max_len = max_len.max(encoded_samples[idx].0.len());
            chosen.push(idx);
        }
        let mut seqs = Vec::with_capacity(schedule.batch);
        let mut targets = Vec::with_capacity(schedule.batch * max_len);
        let mut mask = Vec::with_capacity(schedule.batch * max_len);
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
        let lr = cosine_lr(
            step,
            schedule.steps,
            30,
            schedule.learning_rate,
            schedule.learning_rate / 10.0,
        );
        let (loss, gnorm) = model.train_step(&seqs, &targets, &mask, lr, step);
        if step == 1 || step % 90 == 0 {
            println!(
                "[SFT     ] step {:3}/{} | lr = {:.4} | grad_norm = {:.2} | masked_loss = {:.4}",
                step, schedule.steps, lr, gnorm, loss
            );
        }
    }
    println!("SFT 微调完成，耗时 {:.2}s", t_sft.elapsed().as_secs_f32());

    Ok(())
}

fn refine_chat(
    model: &mut ModernLlm,
    tokenizer: &impl ChatEncoding,
    TrainingData {
        mut qa_pairs,
        paraphrases,
        ..
    }: TrainingData,
    config: TrainingConfig,
) -> io::Result<()> {
    if paraphrases.is_empty() || config.refine_steps == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "此配置缺少改写样本或微调步数",
        ));
    }
    println!(
        "\n加入 {} 条微调样本（含重复加权），并重放原始问答",
        paraphrases.len()
    );
    qa_pairs.extend(paraphrases);
    train_sft(
        model,
        tokenizer,
        &qa_pairs,
        SftSchedule {
            steps: config.refine_steps,
            batch: config.sft_batch,
            learning_rate: config.sft_lr / 3.0,
            bucket_by_length: true,
        },
        &mut Rng::new(2027),
    )
}

fn train_and_save<C: ChatEncoding>(
    model_path: &Path,
    config: TrainingConfig,
    data: TrainingData,
) -> io::Result<(C, ModernLlm)> {
    let TrainingData {
        poems, qa_pairs, ..
    } = &data;
    println!("=== {} ===", C::TITLE);
    println!("语料 {} 字节 · SFT {} 条问答", poems.len(), qa_pairs.len());

    let mut all_texts: Vec<&str> = vec![poems.as_str()];
    for (q, a) in qa_pairs {
        all_texts.push(q.as_str());
        all_texts.push(a.as_str());
    }
    let tok = C::train(&all_texts);
    let pretrain_data = tok.encode(poems).0;

    let TrainingConfig {
        context: max_seq_len,
        d_model,
        query_heads: n_q_head,
        kv_heads: n_kv_head,
        d_ff,
        layers: n_layer,
        pretrain_tokens: pretrain_seq_len,
        pretrain_steps: pt_steps,
        pretrain_batch: pt_batch,
        pretrain_lr,
        sft_steps,
        sft_batch,
        sft_lr,
        ..
    } = config;
    let windows = pretrain_data.len().saturating_sub(pretrain_seq_len);
    if windows == 0 || qa_pairs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "训练语料太短或缺少问答",
        ));
    }

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
        "模型配置: V={}, {}层 ModernBlock (D={}, Q={}头/KV={}头, SwiGLU_FF={}, pos_emb=0) | 总参数量: {}",
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
    println!(
        "预训练 {} tokens · {} 步 · batch {} × {}",
        pretrain_data.len(),
        pt_steps,
        pt_batch,
        pretrain_seq_len
    );
    let all_ones = vec![1.0f32; pt_batch * pretrain_seq_len];

    let t_pt = Instant::now();
    for step in 1..=pt_steps {
        let mut seqs = Vec::with_capacity(pt_batch);
        let mut targets = Vec::with_capacity(pt_batch * pretrain_seq_len);
        for _ in 0..pt_batch {
            let s = (rng.next_u64() as usize) % windows;
            seqs.push(pretrain_data[s..s + pretrain_seq_len].to_vec());
            targets.extend_from_slice(&pretrain_data[s + 1..s + 1 + pretrain_seq_len]);
        }
        let lr = cosine_lr(step, pt_steps, 60, pretrain_lr, pretrain_lr / 6.0);
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

    train_sft(
        &mut model,
        &tok,
        qa_pairs,
        SftSchedule {
            steps: sft_steps,
            batch: sft_batch,
            learning_rate: sft_lr,
            bucket_by_length: false,
        },
        &mut rng,
    )?;
    if config.refine_steps != 0 {
        refine_chat(&mut model, &tok, data, config)?;
    }

    println!("\n=== 现代 Mini-LLM（RMSNorm + RoPE + GQA + SwiGLU）对话实测 ===");
    for q in [
        "你好",
        "背一首静夜思",
        "白日依山尽的下一句",
        "飞流直下三千尺的下一句",
        "背一首春晓",
        "红豆生南国",
    ] {
        let (prompt, _) = encode_prompt(&tok, q);
        let output = model.generate_chat_kv(&tok, &prompt, 48, 0.0, &mut rng)?;
        let mha_floats = output.cache_len * n_layer * 2 * d_model;
        println!(
            "问「{}」\n  回复: {}  [hit_eos={}, GQA KV={} floats vs MHA {} 省 50%!]",
            q,
            decode_reply(&tok, &output.ids)?,
            output.hit_eos,
            output.cache_floats,
            mha_floats
        );
    }

    let bytes = save_checkpoint(model_path, &tok, &model)?;
    println!(
        "\n模型已保存至 {} ({:.1} KB)",
        model_path.display(),
        bytes as f64 / 1024.0
    );
    println!("cargo run --release -p {} -- chat", C::PACKAGE);
    Ok((tok, model))
}

fn bench_decode(tokenizer: &impl ChatEncoding, model: &ModernLlm) -> io::Result<()> {
    let questions = [
        "背一首静夜思",
        "红豆",
        "白日依山尽的下一句",
        "背诵张九龄的《望月怀远》",
        "背诵陈子昂的《登幽州台歌》",
    ];
    let mut rng = Rng::new(2026);
    let mut total_tokens = 0;
    let mut total_time = Duration::ZERO;
    println!("Decode 基准 · 贪心生成 · 每个提问重复 32 次");
    println!("计入 token 选择和 UTF-8 检查，排除 Prefill、终端 I/O 和 EOS 计数。");
    for question in questions {
        let (prompt, _) = encode_prompt(tokenizer, question);
        model.generate_chat_kv(tokenizer, &prompt, 64, 0.0, &mut rng)?;
        let mut tokens = 0;
        let mut time = Duration::ZERO;
        for _ in 0..32 {
            let output = model.generate_chat_kv(tokenizer, &prompt, 64, 0.0, &mut rng)?;
            tokens += output.ids.len() - usize::from(output.hit_eos);
            time += output.decode_time;
        }
        println!(
            "{question} · {tokens} tokens · {:.0} tok/s",
            tokens as f64 / time.as_secs_f64()
        );
        total_tokens += tokens;
        total_time += time;
    }
    println!(
        "合计 {total_tokens} tokens · {:.0} tok/s",
        total_tokens as f64 / total_time.as_secs_f64()
    );
    Ok(())
}

pub fn run_cli<C: ChatEncoding>(
    args: impl IntoIterator<Item = String>,
    mut model_path: PathBuf,
    mut config: TrainingConfig,
    load_data: fn() -> io::Result<TrainingData>,
    mut lookup: Option<ChatLookup>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut mode = "train";
    let mut chat_after = false;
    let has_lookup = lookup.is_some();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "chat" => mode = "chat",
            "bench" => mode = "bench",
            "refine" => mode = "refine",
            "train" => mode = "train",
            "--pretrain-steps" | "--sft-steps" | "--refine-steps" => {
                let steps = args
                    .next()
                    .ok_or("缺少训练步数")?
                    .parse::<std::num::NonZeroUsize>()?
                    .get();
                if arg == "--pretrain-steps" {
                    config.pretrain_steps = steps;
                } else if arg == "--sft-steps" {
                    config.sft_steps = steps;
                } else {
                    config.refine_steps = steps;
                }
            }
            "--chat" => chat_after = true,
            "--raw" if has_lookup => lookup = None,
            "--model" => model_path = PathBuf::from(args.next().ok_or("--model 需要指定文件路径")?),
            "-h" | "--help" => {
                println!(
                    "cargo run --release -p {} -- train [--chat] [--model 路径] [--pretrain-steps N] [--sft-steps N]",
                    C::PACKAGE
                );
                println!(
                    "cargo run --release -p {} -- chat{} [--model 路径]",
                    C::PACKAGE,
                    if has_lookup { " [--raw]" } else { "" }
                );
                println!(
                    "cargo run --release -p {} -- bench [--model 路径]",
                    C::PACKAGE
                );
                if config.refine_steps > 0 {
                    println!(
                        "cargo run --release -p {} -- refine [--refine-steps N] [--model 路径]",
                        C::PACKAGE
                    );
                }
                return Ok(());
            }
            other => return Err(format!("未知参数 {other:?}，用 --help 查看用法").into()),
        }
    }

    if mode != "train" {
        let start = Instant::now();
        let (tok, mut model) = load_checkpoint::<C>(&model_path).map_err(|error| {
            format!(
                "无法加载 {}（{error}）。训练命令为 cargo run --release -p {} -- train",
                model_path.display(),
                C::PACKAGE
            )
        })?;
        println!(
            "已加载 {}（{:.2} ms · V={} · {} 参数）",
            model_path.display(),
            start.elapsed().as_secs_f64() * 1000.0,
            tok.vocab_size(),
            model.num_params()
        );
        match mode {
            "bench" => bench_decode(&tok, &model)?,
            "refine" => {
                refine_chat(&mut model, &tok, load_data()?, config)?;
                save_checkpoint(&model_path, &tok, &model)?;
                println!("微调权重已保存至 {}", model_path.display());
            }
            _ => run_chat_repl(&tok, &model, lookup)?,
        }
    } else {
        let (tok, model) = train_and_save::<C>(&model_path, config, load_data()?)?;
        if chat_after {
            run_chat_repl(&tok, &model, lookup)?;
        }
    }
    Ok(())
}

// ============================================================================
// 8. 单元测试：RoPE 相对距离不变性 + GQA KV Cache 等价与省 50% + 全网梯度检验 + 存档校验
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_candidates_need_room_to_finish_a_character() {
        assert!(can_finish_utf8(&[0xF0, 0x9F], 2));
        assert!(!can_finish_utf8(&[0xF0, 0x9F], 1));
        assert!(can_finish_utf8(&[0xF0, 0x9F, 0xA6, 0x80], 0));
        assert!(!can_finish_utf8(&[0xF0, 0x80], 3));
        assert!(!can_finish_utf8(&[0x9F], 3));
        assert!(!can_finish_utf8(&[0xE6, b'A'], 3));
    }

    #[test]
    fn streaming_utf8_waits_for_partial_characters() {
        let mut output = Vec::new();
        let mut pending = vec![b'A', 0xF0, 0x9F];
        flush_utf8(&mut pending, &mut output).unwrap();
        assert_eq!(output, b"A");
        assert_eq!(pending, [0xF0, 0x9F]);
        pending.extend([0xA6, 0x80, b'B']);
        flush_utf8(&mut pending, &mut output).unwrap();
        assert!(pending.is_empty());
        assert_eq!(output, "A🦀B".as_bytes());
    }

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
        let tok = CharTokenizer::train(&["床前明月光", "你好！"]);
        let model = ModernLlm::new(tok.vocab_size(), 16, 8, 4, 2, 12, 2, &mut rng);

        let tmp = std::env::temp_dir().join(format!(
            "learn_ai_step06_modern_test_{}.mllm",
            std::process::id()
        ));
        save_checkpoint(&tmp, &tok, &model).expect("保存现代模型失败");
        let (tok2, model2) = load_checkpoint::<CharTokenizer>(&tmp).expect("加载现代模型失败");
        let _ = fs::remove_file(&tmp);

        assert_eq!(tok.id_to_token, tok2.id_to_token);
        let (prompt, _) = encode_prompt(&tok, "你好");
        let (z1, _) = model.prefill(&prompt);
        let (z2, _) = model2.prefill(&prompt);
        assert_eq!(z1, z2);
    }
}
