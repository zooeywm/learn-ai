# learn-ai：用纯 Rust 从零手搓迷你 Transformer LLM

从零基础出发，不依赖外部深度学习框架，用纯 Rust 一步一步实现一个工作原理 100% 正宗的 Decoder-Only Transformer 大语言模型，并同步沉淀每阶段的学习笔记。

## 闯关路线图

| 阶段 | 代码目录 | 学习笔记 | 核心内容 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 01** | `crates/step01-bigram` | `notes/01-bigram.md` | 字符级 Tokenizer、Bigram 查表模型、Softmax、交叉熵 Loss、手推梯度、SGD、温度采样 | 进行中 |
| **Step 02** | `crates/step02-autograd` | `notes/02-autograd.md` | 矩阵张量、计算图与反向传播（Autograd）、数值梯度单测 | 待解锁 |
| **Step 03** | `crates/step03-attention` | `notes/03-attention.md` | Embedding + 位置编码、因果多头自注意力（Q/K/V + Causal Mask） | 待解锁 |
| **Step 04** | `crates/step04-minigpt` | `notes/04-minigpt.md` | 完整 Transformer Block（LayerNorm / FFN / 残差连接）、AdamW、训练迷你 GPT | 待解锁 |

## 常用命令

```bash
# 运行第 1 关演示程序（推荐使用 --release，速度飞快）
cargo run --release -p step01-bigram

# 运行单元测试（包含分词器测试、Softmax 数值稳定测试、数值梯度验算）
cargo test -p step01-bigram
```
