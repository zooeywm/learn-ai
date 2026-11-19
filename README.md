# learn-ai：用纯 Rust 从零手搓迷你 Transformer LLM

从零基础出发，不依赖外部深度学习框架，用纯 Rust 一步一步实现一个工作原理 100% 正宗的 Decoder-Only Transformer 大语言模型，并同步沉淀每阶段的学习笔记。

## 闯关路线图

| 阶段 | 代码目录 | 学习笔记 | 核心内容 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 01** | `crates/step01-bigram` | `notes/01-bigram.md` + `01-bigram-visualizer.html` | 字符级 Tokenizer、Bigram 查表模型、Softmax、交叉熵 Loss、手推梯度、SGD、温度采样 | 已完成 |
| **Step 02** | `crates/step02-autograd` | `notes/02-autograd.md` + `02-autograd-visualizer.html` | 矩阵张量、Arena Tape 计算图与自动求导（Autograd）、数值梯度单测、3 字上下文多层神经语言模型 | 进行中 |
| **Step 03** | `crates/step03-attention` | `notes/03-attention.md` | Embedding + 位置编码、因果多头自注意力（Q/K/V + Causal Mask） | 待解锁 |
| **Step 04** | `crates/step04-minigpt` | `notes/04-minigpt.md` | 完整 Transformer Block（LayerNorm / FFN / 残差连接）、AdamW、训练迷你 GPT | 待解锁 |

## 常用命令

```bash
# 运行全部单元测试
cargo test

# 运行第 1 关（Bigram）
cargo run --release -p step01-bigram

# 运行第 2 关（纯 Rust 自动求导引擎 + 3 字上下文神经语言模型）
cargo run --release -p step02-autograd
```
