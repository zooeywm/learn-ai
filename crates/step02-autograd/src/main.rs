//! Step 02: 纯 Rust 矩阵张量与自动求导引擎 (Arena-Tape Autograd)
//!
//! 为什么要造自动求导引擎？
//!   Step 01 的 Bigram 只有一层查表，我们可以手推 `probs - target`。
//!   一旦进入多层神经网络（Embedding -> Linear -> ReLU -> Linear -> CrossEntropy）
//!   以及后面的 Transformer，几十个矩阵运算嵌套在一起，手动推导全网梯度既繁琐又极易出错。
//!
//! 本关核心设计（最契合 Rust 所有权的 Arena Tape 模式）：
//!   - 不使用笨重的 Rc<RefCell<Node>>，而是用 `Tape` 里的 `Vec<Node>` 统一管理计算图；
//!   - 每个张量句柄 `TensorId(usize)` 只是轻量 Copy 索引；
//!   - 因为子节点一定在父节点之后加入 Vec，所以 `Vec` 的逆序 `(0..=loss_id).rev()`
//!     天然就是合法的反向拓扑顺序，连递归 DFS 都省了！
//!
//! 本关实战验证：
//!   用这个纯 Rust 自动求导引擎训练一个「看前 3 个字猜第 4 个字」的 2 层神经语言模型（MLP LM），
//!   彻底打破 Step 01 Bigram 停在 Loss=1.25 的单字记忆天花板！

use std::time::Instant;

use mini_core::{load_poems, sample_from_logits, softmax, Rng, Tokenizer};

// ============================================================================
// 1. Arena-Tape 自动求导引擎（本关核心！）
// ============================================================================

/// 轻量张量句柄：只是一个指向 Tape.nodes 数组的下标，可以随意 Copy，零借用冲突。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TensorId(usize);

/// 记录每个节点是如何由父节点算出来的，反向传播时据此按链式法则把梯度回传给父节点。
enum Op {
    Leaf,
    Add(TensorId, TensorId),
    Mul(TensorId, TensorId),
    MatMul(TensorId, TensorId),
    Relu(TensorId),
    /// 查词向量表并拼接上下文：table(V, D) + ctx_indices(B × K) -> (B, K * D)
    EmbedConcat {
        table: TensorId,
        ctx_indices: Vec<Vec<usize>>,
    },
    /// 融合 Softmax + 交叉熵损失：logits(B, V) + targets(B) -> 标量 (1, 1)
    CrossEntropy {
        logits: TensorId,
        targets: Vec<usize>,
    },
}

struct Node {
    rows: usize,
    cols: usize,
    data: Vec<f32>, // 前向计算的结果（row-major，长度 rows * cols）
    grad: Vec<f32>, // 反向传播累加的梯度 d(Loss) / d(data)
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

    /// 注册叶子张量（如模型权重或常数输入）
    fn leaf(&mut self, rows: usize, cols: usize, data: Vec<f32>) -> TensorId {
        self.push(rows, cols, data, Op::Leaf)
    }

    fn param(&mut self, p: &Param) -> TensorId {
        self.leaf(p.rows, p.cols, p.data.clone())
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

    /// 矩阵加法：支持同形状 (R, C) + (R, C)，也支持偏置行广播 (R, C) + (1, C)
    fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ra, ca) = self.shape(a);
        let (rb, cb) = self.shape(b);
        assert_eq!(ca, cb, "Add 列数必须相等");
        assert!(ra == rb || rb == 1, "Add 仅支持同形状或 (R,C)+(1,C) 行广播");

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

    /// 逐元素乘法：(R, C) * (R, C)
    #[allow(dead_code)]
    fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (ra, ca) = self.shape(a);
        assert_eq!((ra, ca), self.shape(b), "Mul 形状必须一致");
        let out: Vec<f32> = self.nodes[a.0]
            .data
            .iter()
            .zip(self.nodes[b.0].data.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        self.push(ra, ca, out, Op::Mul(a, b))
    }

    /// 矩阵乘法：A(M, K) @ B(K, N) -> Out(M, N) —— 深度学习的核心算子
    fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (m, k) = self.shape(a);
        let (kb, n) = self.shape(b);
        assert_eq!(k, kb, "MatMul 内维不匹配: ({m},{k}) @ ({kb},{n})");

        let a_data = &self.nodes[a.0].data;
        let b_data = &self.nodes[b.0].data;
        let mut out = vec![0.0f32; m * n];
        // i-k-j 循环顺序对 CPU 缓存更友好（连续访问 B 的行）
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

    /// ReLU 非线性激活：max(0, x)
    fn relu(&mut self, x: TensorId) -> TensorId {
        let (r, c) = self.shape(x);
        let out: Vec<f32> = self.nodes[x.0].data.iter().map(|&v| v.max(0.0)).collect();
        self.push(r, c, out, Op::Relu(x))
    }

    /// 查词向量并拼接：把每条样本的 K 个上下文 token 向量拼成长度为 K*D 的一行
    fn embed_concat(&mut self, table: TensorId, ctx_indices: Vec<Vec<usize>>) -> TensorId {
        let (_v, d) = self.shape(table);
        let batch = ctx_indices.len();
        let ctx_len = ctx_indices[0].len();
        let out_cols = ctx_len * d;

        let tbl = &self.nodes[table.0].data;
        let mut out = vec![0.0f32; batch * out_cols];
        for (b, ctx) in ctx_indices.iter().enumerate() {
            for (slot, &tok_id) in ctx.iter().enumerate() {
                let src = &tbl[tok_id * d..(tok_id + 1) * d];
                let dst = &mut out[b * out_cols + slot * d..b * out_cols + (slot + 1) * d];
                dst.copy_from_slice(src);
            }
        }
        self.push(
            batch,
            out_cols,
            out,
            Op::EmbedConcat { table, ctx_indices },
        )
    }

    /// 交叉熵损失：把 logits(B, V) 与真实目标 targets(B) 算出标量 Loss (1, 1)
    fn cross_entropy(&mut self, logits: TensorId, targets: Vec<usize>) -> TensorId {
        let (batch, v) = self.shape(logits);
        assert_eq!(batch, targets.len());

        let z = &self.nodes[logits.0].data;
        let mut total_loss = 0.0f32;
        for (b, &y) in targets.iter().enumerate() {
            let probs = softmax(&z[b * v..(b + 1) * v]);
            total_loss += -(probs[y] + 1e-12).ln();
        }
        self.push(
            1,
            1,
            vec![total_loss / (batch as f32)],
            Op::CrossEntropy { logits, targets },
        )
    }

    /// 自动求导魔法核心：从 loss 节点出发，沿 Tape 逆序执行链式法则！
    fn backward(&mut self, loss_id: TensorId) {
        assert_eq!(self.nodes[loss_id.0].data.len(), 1, "只能对标量 Loss 调用 backward");
        self.nodes[loss_id.0].grad[0] = 1.0;

        // 因为每个节点创建时父节点必已存在，逆序遍历天然满足拓扑顺序！
        for id in (0..=loss_id.0).rev() {
            // 先取出当前节点的梯度快照（避免同时可变借用 self.nodes）
            let d_out = self.nodes[id].grad.clone();
            if d_out.iter().all(|&g| g == 0.0) {
                continue;
            }

            match &self.nodes[id].op {
                Op::Leaf => {}

                Op::Add(a, b) => {
                    let (a, b) = (*a, *b);
                    let (ra, ca) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
                    let rb = self.nodes[b.0].rows;
                    for r in 0..ra {
                        let br = if rb == 1 { 0 } else { r };
                        for c in 0..ca {
                            let g = d_out[r * ca + c];
                            self.nodes[a.0].grad[r * ca + c] += g;
                            self.nodes[b.0].grad[br * ca + c] += g;
                        }
                    }
                }

                Op::Mul(a, b) => {
                    let (a, b) = (*a, *b);
                    let a_data = self.nodes[a.0].data.clone();
                    let b_data = self.nodes[b.0].data.clone();
                    for i in 0..d_out.len() {
                        self.nodes[a.0].grad[i] += d_out[i] * b_data[i];
                        self.nodes[b.0].grad[i] += d_out[i] * a_data[i];
                    }
                }

                Op::MatMul(a, b) => {
                    // 若 C = A(M,K) @ B(K,N)，则：
                    //   dA(M,K) += dC(M,N) @ B^T(N,K)
                    //   dB(K,N) += A^T(K,M) @ dC(M,N)
                    let (a, b) = (*a, *b);
                    let (m, k) = (self.nodes[a.0].rows, self.nodes[a.0].cols);
                    let n = self.nodes[b.0].cols;
                    let a_data = self.nodes[a.0].data.clone();
                    let b_data = self.nodes[b.0].data.clone();

                    // dA += dC @ B^T
                    for i in 0..m {
                        for p in 0..k {
                            let mut acc = 0.0f32;
                            for j in 0..n {
                                acc += d_out[i * n + j] * b_data[p * n + j];
                            }
                            self.nodes[a.0].grad[i * k + p] += acc;
                        }
                    }
                    // dB += A^T @ dC
                    for i in 0..m {
                        for p in 0..k {
                            let a_ip = a_data[i * k + p];
                            for j in 0..n {
                                self.nodes[b.0].grad[p * n + j] += a_ip * d_out[i * n + j];
                            }
                        }
                    }
                }

                Op::Relu(x) => {
                    let x = *x;
                    for i in 0..d_out.len() {
                        // ReLU 门控：正区间导数为 1（梯度原样通过），非正区间导数为 0（梯度被挡住）
                        if self.nodes[x.0].data[i] > 0.0 {
                            self.nodes[x.0].grad[i] += d_out[i];
                        }
                    }
                }

                Op::EmbedConcat { table, ctx_indices } => {
                    let table = *table;
                    let d = self.nodes[table.0].cols;
                    let ctx_indices = ctx_indices.clone();
                    let out_cols = ctx_indices[0].len() * d;
                    for (b, ctx) in ctx_indices.iter().enumerate() {
                        for (slot, &tok_id) in ctx.iter().enumerate() {
                            for j in 0..d {
                                self.nodes[table.0].grad[tok_id * d + j] +=
                                    d_out[b * out_cols + slot * d + j];
                            }
                        }
                    }
                }

                Op::CrossEntropy { logits, targets } => {
                    let logits = *logits;
                    let targets = targets.clone();
                    let (batch, v) = (self.nodes[logits.0].rows, self.nodes[logits.0].cols);
                    let scale = d_out[0] / (batch as f32);
                    let z = self.nodes[logits.0].data.clone();
                    for (b, &y) in targets.iter().enumerate() {
                        let probs = softmax(&z[b * v..(b + 1) * v]);
                        for j in 0..v {
                            let target = if j == y { 1.0 } else { 0.0 };
                            self.nodes[logits.0].grad[b * v + j] += (probs[j] - target) * scale;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 3. 可训练参数 + AdamW 优化器（每份参数自带一阶动量 m 和二阶动量 v）
// ============================================================================
struct Param {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
    decay: bool, // 仅对 2D 权重做权重衰减，不对 bias 做
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
        let t = step as i32;
        let bc1 = 1.0 - beta1.powi(t);
        let bc2 = 1.0 - beta2.powi(t);

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
// 4. 实战模型：3 字上下文神经语言模型 (Embedding + 2 层 MLP)
//    对比 Step 01（只能看前 1 个字，Loss 卡在 1.25），现在能看前 3 个字！
// ============================================================================
struct MlpLanguageModel {
    ctx_len: usize,
    emb: Param, // (V, D)
    w1: Param,  // (ctx_len * D, H)
    b1: Param,  // (1, H)
    w2: Param,  // (H, V)
    b2: Param,  // (1, V)
}

impl MlpLanguageModel {
    fn new(vocab_size: usize, ctx_len: usize, emb_dim: usize, hidden_dim: usize, rng: &mut Rng) -> Self {
        let in_dim = ctx_len * emb_dim;
        // He 初始化，避免深层网络梯度消失或爆炸
        let std1 = (2.0 / in_dim as f32).sqrt();
        let std2 = (1.0 / hidden_dim as f32).sqrt();
        Self {
            ctx_len,
            emb: Param::randn(vocab_size, emb_dim, 0.2, rng, true),
            w1: Param::randn(in_dim, hidden_dim, std1, rng, true),
            b1: Param::zeros(1, hidden_dim),
            w2: Param::randn(hidden_dim, vocab_size, std2, rng, true),
            b2: Param::zeros(1, vocab_size),
        }
    }

    fn num_params(&self) -> usize {
        self.emb.data.len()
            + self.w1.data.len()
            + self.b1.data.len()
            + self.w2.data.len()
            + self.b2.data.len()
    }

    /// 完整一步训练：搭计算图 -> 前向算 Loss -> 一行 tape.backward(loss) -> AdamW 更新
    fn train_step(
        &mut self,
        batch_ctx: Vec<Vec<usize>>,
        batch_y: Vec<usize>,
        lr: f32,
        step: usize,
    ) -> f32 {
        let mut tape = Tape::new();
        let emb = tape.param(&self.emb);
        let w1 = tape.param(&self.w1);
        let b1 = tape.param(&self.b1);
        let w2 = tape.param(&self.w2);
        let b2 = tape.param(&self.b2);

        // 前向传播（看起来是不是已经很像 PyTorch 了？）
        let x = tape.embed_concat(emb, batch_ctx); // (B, ctx_len * D)
        let z1 = tape.matmul(x, w1);               // (B, H)
        let z1 = tape.add(z1, b1);                 // (B, H)
        let h = tape.relu(z1);                     // (B, H)
        let logits = tape.matmul(h, w2);           // (B, V)
        let logits = tape.add(logits, b2);         // (B, V)
        let loss_id = tape.cross_entropy(logits, batch_y);

        // 一行自动求导：自动算出 emb, w1, b1, w2, b2 五组参数的全部梯度！
        tape.backward(loss_id);

        let wd = 0.01;
        self.emb.adamw_step(tape.grad(emb), lr, wd, step);
        self.w1.adamw_step(tape.grad(w1), lr, wd, step);
        self.b1.adamw_step(tape.grad(b1), lr, wd, step);
        self.w2.adamw_step(tape.grad(w2), lr, wd, step);
        self.b2.adamw_step(tape.grad(b2), lr, wd, step);

        tape.data(loss_id)[0]
    }

    /// 自回归推理：每次取末尾 ctx_len 个字送入网络，温度采样下一个字
    fn generate(
        &self,
        prompt_ids: &[usize],
        max_new: usize,
        temperature: f32,
        rng: &mut Rng,
    ) -> Vec<usize> {
        assert!(
            prompt_ids.len() >= self.ctx_len,
            "提示词长度至少需要 {} 个字",
            self.ctx_len
        );
        let mut ids = prompt_ids.to_vec();

        for _ in 0..max_new {
            let ctx = ids[ids.len() - self.ctx_len..].to_vec();
            let mut tape = Tape::new();
            let emb = tape.param(&self.emb);
            let w1 = tape.param(&self.w1);
            let b1 = tape.param(&self.b1);
            let w2 = tape.param(&self.w2);
            let b2 = tape.param(&self.b2);

            let x = tape.embed_concat(emb, vec![ctx]);
            let z1 = tape.matmul(x, w1);
            let z1 = tape.add(z1, b1);
            let h = tape.relu(z1);
            let z2 = tape.matmul(h, w2);
            let logits_id = tape.add(z2, b2);

            let next_id = sample_from_logits(tape.data(logits_id), temperature, rng);
            ids.push(next_id);
        }
        ids
    }
}

fn main() {
    let text = load_poems();

    let tok = Tokenizer::from_text(&text);
    let data = tok.encode(&text);
    let v = tok.vocab_size();

    let ctx_len = 3;   // 看前 3 个字，猜第 4 个字！
    let emb_dim = 24;  // 每个字的词向量维度
    let hidden = 96;   // 隐藏层神经元数
    let batch_size = 64;
    let steps = 800;
    let lr = 0.01;

    let mut rng = Rng::new(42);
    let mut model = MlpLanguageModel::new(v, ctx_len, emb_dim, hidden, &mut rng);

    println!("================================================================");
    println!("Step 02: 纯 Rust Arena-Tape 自动求导 + 3 字上下文 MLP 语言模型");
    println!(
        "词表 V: {} | 上下文窗口: {} 字 | 总参数量: {}",
        v,
        ctx_len,
        model.num_params()
    );
    println!("对比 Step 01 Bigram 天花板 Loss ≈ 1.25，看多层网络能降到多少！");
    println!("================================================================\n");

    let prompt = "床前明";
    let prompt_ids = tok.encode(prompt);
    let preview = |m: &MlpLanguageModel| -> String {
        let mut grng = Rng::new(7);
        tok.decode(&m.generate(&prompt_ids, 28, 0.5, &mut grng))
            .replace('\n', " / ")
    };

    println!("[训练前 Step   0] 「{}」-> {}", prompt, preview(&model));

    let t0 = Instant::now();
    for step in 1..=steps {
        let mut batch_ctx = Vec::with_capacity(batch_size);
        let mut batch_y = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let start = rng.next_usize(data.len() - ctx_len);
            batch_ctx.push(data[start..start + ctx_len].to_vec());
            batch_y.push(data[start + ctx_len]);
        }

        let loss = model.train_step(batch_ctx, batch_y, lr, step);
        if matches!(step, 1 | 50 | 150 | 300 | 500 | 800) {
            println!(
                "[Step {:3}/{}] Loss: {:.4} ({:.2}s) | 「{}」-> {}",
                step,
                steps,
                loss,
                t0.elapsed().as_secs_f32(),
                prompt,
                preview(&model)
            );
        }
    }

    println!("\n================================================================");
    println!("训练完成！测试不同 3 字开头（看看「床前明」和「秦时明」还会不会串台）：");
    println!("================================================================");
    for p in ["床前明", "秦时明", "春眠不", "白日依"] {
        let mut grng = Rng::new(99);
        let out = tok
            .decode(&model.generate(&tok.encode(p), 28, 0.5, &mut grng))
            .replace('\n', " / ");
        println!("「{}」-> {}", p, out);
    }
}

// ============================================================================
// 5. 全算子数值梯度单测（每写一个自动求导算子，必须用有限差分验算！）
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// 通用数值梯度检查：对任意构造标量 Loss 的闭包 f(params) 核对所有参数的解析梯度
    fn assert_grad_close(
        params: &mut [Vec<f32>],
        build_and_eval: impl Fn(&[Vec<f32>]) -> (f32, Vec<Vec<f32>>),
        tol: f32,
    ) {
        let (_loss, analytical) = build_and_eval(params);
        let eps = 1e-3f32;

        for p_idx in 0..params.len() {
            for i in 0..params[p_idx].len() {
                let orig = params[p_idx][i];

                params[p_idx][i] = orig + eps;
                let (l_plus, _) = build_and_eval(params);

                params[p_idx][i] = orig - eps;
                let (l_minus, _) = build_and_eval(params);

                params[p_idx][i] = orig;

                let num = (l_plus - l_minus) / (2.0 * eps);
                let ana = analytical[p_idx][i];
                let err = (ana - num).abs();
                assert!(
                    err < tol,
                    "参数组 {} 第 {} 维梯度不匹配: 解析={} vs 数值={} (误差={})",
                    p_idx,
                    i,
                    ana,
                    num,
                    err
                );
            }
        }
    }

    #[test]
    fn test_mul_add_chain_rule() {
        // 测试最经典的标量链式法则：L = (x * w + b) * c
        let mut params = vec![vec![2.0], vec![-1.5], vec![0.7], vec![3.0]];
        assert_grad_close(
            &mut params,
            |ps| {
                let mut t = Tape::new();
                let x = t.leaf(1, 1, ps[0].clone());
                let w = t.leaf(1, 1, ps[1].clone());
                let b = t.leaf(1, 1, ps[2].clone());
                let c = t.leaf(1, 1, ps[3].clone());
                let xw = t.mul(x, w);
                let z = t.add(xw, b);
                let out = t.mul(z, c);
                t.backward(out);
                (
                    t.data(out)[0],
                    vec![
                        t.grad(x).to_vec(),
                        t.grad(w).to_vec(),
                        t.grad(b).to_vec(),
                        t.grad(c).to_vec(),
                    ],
                )
            },
            1e-3,
        );
    }

    #[test]
    fn test_matmul_broadcast_add_relu_cross_entropy() {
        // 端到端同时测试 EmbedConcat + MatMul + 行广播 Add + ReLU + CrossEntropy 全部算子！
        let mut rng = Rng::new(123);
        let (v, d, ctx_len, h) = (5, 3, 2, 4);
        let mut params = vec![
            (0..v * d).map(|_| rng.normal(0.5)).collect::<Vec<_>>(),
            (0..ctx_len * d * h).map(|_| rng.normal(0.5)).collect::<Vec<_>>(),
            (0..h).map(|_| rng.normal(0.2)).collect::<Vec<_>>(),
            (0..h * v).map(|_| rng.normal(0.5)).collect::<Vec<_>>(),
            (0..v).map(|_| rng.normal(0.2)).collect::<Vec<_>>(),
        ];

        let batch_ctx = vec![vec![0, 1], vec![2, 3], vec![1, 4]];
        let batch_y = vec![2, 4, 0];

        assert_grad_close(
            &mut params,
            |ps| {
                let mut t = Tape::new();
                let emb = t.leaf(v, d, ps[0].clone());
                let w1 = t.leaf(ctx_len * d, h, ps[1].clone());
                let b1 = t.leaf(1, h, ps[2].clone());
                let w2 = t.leaf(h, v, ps[3].clone());
                let b2 = t.leaf(1, v, ps[4].clone());

                let x = t.embed_concat(emb, batch_ctx.clone());
                let z1 = t.matmul(x, w1);
                let z1 = t.add(z1, b1);
                let h1 = t.relu(z1);
                let z2 = t.matmul(h1, w2);
                let logits = t.add(z2, b2);
                let loss = t.cross_entropy(logits, batch_y.clone());
                t.backward(loss);

                (
                    t.data(loss)[0],
                    vec![
                        t.grad(emb).to_vec(),
                        t.grad(w1).to_vec(),
                        t.grad(b1).to_vec(),
                        t.grad(w2).to_vec(),
                        t.grad(b2).to_vec(),
                    ],
                )
            },
            2e-3,
        );
    }
}
