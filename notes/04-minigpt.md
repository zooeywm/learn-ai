# Step 04（终极关）：完整纯 Rust Pre-Norm Transformer（Mini-GPT）

> **配套可运行代码**：[crates/step04-minigpt/src/main.rs](../crates/step04-minigpt/src/main.rs)  
> **配套 3D 离线动画**：[notes/learn-ai-visualizer.html](./learn-ai-visualizer.html)（单文件全景演化版，点 **`下一 Step (04) →`** 观看 `Step 03 → Step 04` 的 3D 架构进化动画）

---

## 1. 从 Step 03 到真正的 GPT，还差哪两块拼图？

在 Step 03 的终端实战中，单层因果多头自注意力已经用更轻的参数攻克了 `床前明月光，疑是 → 地上霜`，却在另一个地方露出了马脚：

```text
Step 03 输出：输入「春眠不觉晓，」 -> 春眠不觉晓，处处处处处处处处处处...
```

为什么会陷入「处处处处...」复读？因为 Step 03 只有**纯线性 Attention 混合**，还缺现代 Transformer Block 的最后两块标准拼图：

1. **LayerNorm（层归一化）**：把每个 token 向量稳压在均值 `0`、方差 `1`，防止多层叠加时数值爆炸或梯度消失；
2. **FFN（前馈非线性思考层 `D → 4D → D` + GELU）**：
   - **Self-Attention** 负责**跨位置沟通**（“去前文把相关线索借过来”）；
   - **FFN** 负责**在每个位置内部做 4 倍高维非线性思考与计数**（“我已经看到两个「处」了，下一步该接「闻」，不能再接「处」！”）。

把这两块拼图和残差连接组合起来，再**堆叠多层**，就得到了和 GPT-2 / LLaMA / DeepSeek 原理 100% 同源的 **Pre-Norm Decoder-Only Transformer（Mini-GPT）**！

---

## 2. Pre-Norm Transformer Block 完整数据流

对于每一层 Block（本关堆叠 `N_LAYER = 2` 层），输入张量 $x \in \mathbb{R}^{T \times D}$ 按顺序走两条带**残差高速公路**的子层：

$$x_1 = x + \text{CausalMultiHeadAttn}\big(\text{LayerNorm}_1(x)\big)$$

$$x_2 = x_1 + \text{FFN}_{\text{GELU}}\big(\text{LayerNorm}_2(x_1)\big)$$

最后经过一层 **Final LayerNorm** 稳压，再通过 **LM Head** 投影到词表大小 $V$ 得到预测 logits：

$$\text{logits} = \text{LayerNorm}_f(x_{\text{final}}) W_{\text{lm}} + b_{\text{lm}}$$

### 为什么现代大模型全用 Pre-Norm（而不是 2017 原论文的 Post-Norm）？
- **Post-Norm**：`LN(x + SubLayer(x))` —— 梯度回传主干每次都必须穿过 LayerNorm，深层时早期极易训练不稳定；
- **Pre-Norm**：`x + SubLayer(LN(x))` —— 残差主干是一条**完全无阻挡的加法高速公路**，反向传播的梯度可以直接从顶层无损直达底层 Embedding！

---

## 3. 新增核心算子数学与精确反传

### 3.1 LayerNorm（逐 token 归一化）

对第 $r$ 个 token 的 $D$ 维向量 $x_r$：

$$\mu_r = \frac{1}{D}\sum_{c=1}^{D} x_{r,c}, \qquad \sigma_r^2 = \frac{1}{D}\sum_{c=1}^{D} (x_{r,c}-\mu_r)^2, \qquad \hat{x}_{r,c} = \frac{x_{r,c}-\mu_r}{\sqrt{\sigma_r^2 + \epsilon}}$$

$$y_{r,c} = \gamma_c \hat{x}_{r,c} + \beta_c$$

已知上游梯度 $dY$，令 $g_{r,c} = dY_{r,c}\,\gamma_c$，精确解析反传为：

$$d\gamma_c \mathrel{+}= \sum_r dY_{r,c}\,\hat{x}_{r,c}, \qquad d\beta_c \mathrel{+}= \sum_r dY_{r,c}$$

$$dX_{r,c} \mathrel{+}= \frac{1}{\sqrt{\sigma_r^2+\epsilon}}\left(g_{r,c} - \frac{1}{D}\sum_j g_{r,j} - \hat{x}_{r,c}\cdot\frac{1}{D}\sum_j g_{r,j}\hat{x}_{r,j}\right)$$

### 3.2 FFN + 平滑 GELU 激活

$$\text{FFN}(u) = \text{GELU}(u W_1 + b_1)\, W_2 + b_2 \qquad (D \to 4D \to D)$$

采用 GPT-2 经典的平滑 GELU 近似：

$$\text{GELU}(v) = \frac{1}{2} v \Big(1 + \tanh\big(\sqrt{2/\pi}(v + 0.044715\,v^3)\big)\Big)$$

相比 Step 02 的 ReLU（负半轴直接一刀切为 `0`，容易出现 Dead ReLU），GELU 在 `0` 附近平滑过渡，允许微小的负梯度通过，深层 Transformer 训练更丝滑。

---

## 4. 工业界标配训练三件套（本关全部手写实现）

1. **GPT-2 残差投影缩放**：每个 Block 的输出矩阵 $W_O$ 和 $W_2$ 按 $\frac{1}{\sqrt{2 N_{\text{layer}}}}$ 缩小初始化方差，避免多层残差叠加后特征幅度越加越大；
2. **全局梯度 L2 范数裁剪（`max_norm = 1.0`）**：每步反传后汇总全网梯度范数 $\|\mathbf{g}\|_2$，超过 `1.0` 就整体等比缩小，彻底杜绝深网训练偶发梯度爆炸；
3. **线性 Warmup + 余弦退火学习率（Cosine LR）**：前 80 步温和升温让 AdamW 二阶矩稳定，随后按余弦曲线优雅降温精调收敛。

---

## 5. 终极关纯 Rust 实战成绩单

运行：
```bash
cargo test
cargo run --release -p step04-minigpt
```

- **全仓库单元测试**：**10/10 全部通过（0 warning）**，包含：
  - `test_layernorm_zero_mean_unit_var`（均值 `0`、方差 `1` 校验）
  - `test_minigpt_gradient_check`（2 层 Block 全部 9 类参数含 LayerNorm/Attention/GELU FFN 数值梯度校验，相对误差 `< 3e-3`）
  - `test_minigpt_causal_no_future_leak`（2 层堆叠依然严格零未来泄露）
- **实训收敛**（`2层 Pre-Norm Transformer, D=32, 4头, FFN 32→128→32, 49,973 参数`，CPU 仅需 **20.06s**）：
  - Loss 从 `6.4265` 平稳降至 **`0.1076`**，梯度范数全程稳定在 `0.26 ~ 1.67`；
- **终极续写测试（5/5 全满分，彻底根治 Step 02 串台 & Step 03「处处处」复读！）**：
  ```text
  输入「床前明月光，疑是」     -> 床前明月光，疑是地上霜。举头望明月，低头思故乡。
  输入「春眠不觉晓，」         -> 春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
  输入「白日依山尽，黄河」     -> 白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
  输入「飞流直下三千尺，疑是」 -> 飞流直下三千尺，疑是银河落九天。
  输入「红豆生南国，」         -> 红豆生南国，春来发几枝。愿君多采撷，此物最相思。
  ```

---

## 6. 全景通关复盘：你手搓的 Mini-GPT 与千亿级大模型差在哪里？

| 演化阶段 | 核心机制 | 视野 | 解决的痛点 | 工业界千亿模型（GPT-4 / LLaMA / DeepSeek）的对应演进 |
| :--- | :--- | :--- | :--- | :--- |
| **Step 00** | 字符级 Tokenizer | — | 文字 → 整数 ID | 升级为字节级 **BPE**（词表 `50k ~ 150k`） |
| **Step 01** | Bigram + Softmax + CrossEntropy + SGD | 1 字 | 最简概率语言模型闭环 | 保留了同样的 **Next-Token Prediction** 交叉熵目标 |
| **Step 02** | Arena-Tape Autograd + MLP + ReLU + AdamW | 3 字 | 打破单字查表瓶颈，实现自动求导 | 升级为 GPU 并行框架（PyTorch/Triton）+ AdamW |
| **Step 03** | PosEmb + Q/K/V + 因果多头自注意力 | 全窗口 | 打破固定拼接牢笼，动态回望长程历史 | 升级为 **RoPE 旋转位置编码 + GQA + FlashAttention + KV Cache** |
| **Step 04** | **Pre-Norm LayerNorm + GELU FFN + 多层堆叠** | **全窗口深度思考** | **稳住深网、消灭复读，形成完整 GPT** | 升级为 **RMSNorm + SwiGLU FFN / MoE + 32~128 层 + 万亿 Token 语料** |

**恭喜通关！** 你已经用不到两千行零依赖纯 Rust 代码，从第一个字节到完整 Pre-Norm Transformer，真正亲手造出并理解了现代大语言模型的全部核心原理。
