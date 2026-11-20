# learn-ai：用纯 Rust 从零手搓迷你 Transformer LLM

从零基础出发，不依赖外部深度学习框架，用纯 Rust 一步一步实现一个工作原理 100% 正宗的 Decoder-Only Transformer 大语言模型，并同步沉淀每阶段的学习笔记。

## 闯关路线图

| 阶段 | 代码目录 | 学习笔记 | 核心内容 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| **全景单文件 3D 演示** | `notes/learn-ai-visualizer.html` | 单文件离线版 | **Step 00 Tokenizer → Step 01 Bigram → Step 02 Autograd/MLP → Step 03 自注意力 → Step 04 Mini-GPT → Step 05 SFT + KV Cache** 全景串联，支持「上一 Step / 下一 Step」双向 3D 架构迭代过渡动画（直观展示 `+多了 / -少了 / ↻替换 / ✓保留`） | 已完成 |
| **共享基础库** | `crates/mini-core` | — | 复用的 `Rng` / `Tokenizer` / `softmax` / `sample_from_logits` / `load_poems` | 已就绪 |
| **Step 01** | `crates/step01-bigram` | `notes/01-bigram.md` | Bigram 查表模型、交叉熵 Loss、手推梯度、SGD、温度采样 | 已完成 |
| **Step 02** | `crates/step02-autograd` | `notes/02-autograd.md` | 矩阵张量、Arena Tape 计算图与自动求导（Autograd）、数值梯度单测、3 字上下文多层神经语言模型 | 已完成 |
| **Step 03** | `crates/step03-attention` | `notes/03-attention.md` | Embedding + 位置编码、因果多头自注意力（Q/K/V + Causal Mask + Residual）、长程依赖实战 | 已完成 |
| **Step 04** | `crates/step04-minigpt` | `notes/04-minigpt.md` | 完整 2 层 Pre-Norm Transformer（LayerNorm / GELU FFN / 残差高速路 / 梯度裁剪 / Warmup 余弦学习率 / AdamW） | 已完成 |
| **Step 05** | `crates/step05-sft-chat` | `notes/05-sft-chat.md` | SFT 指令微调（特殊控制符 + Assistant-Only Loss Mask）把续写机变成迷你 ChatGPT，以及 Prefill / Decode 两阶段 KV Cache 推理加速（实测 39x+） | 已完成 |

## 常用命令

```bash
# 运行全部 13 个单元测试
cargo test

# 运行第 1 关（Bigram）
cargo run --release -p step01-bigram

# 运行第 2 关（纯 Rust 自动求导引擎 + 3 字上下文神经语言模型）
cargo run --release -p step02-autograd

# 运行第 3 关（纯 Rust 因果多头自注意力模型）
cargo run --release -p step03-attention

# 运行第 4 关（完整 2 层 Pre-Norm Mini-GPT 基座）
cargo run --release -p step04-minigpt

# 运行第 5 关（SFT 指令微调把 Mini-GPT 变成迷你 ChatGPT + Prefill/Decode KV Cache 实测）
cargo run --release -p step05-sft-chat
```
