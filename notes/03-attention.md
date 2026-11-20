# Step 03：Embedding + 位置编码 + 因果多头自注意力（Causal Multi-Head Self-Attention）

> **配套可运行代码**：[crates/step03-attention/src/main.rs](../crates/step03-attention/src/main.rs)  
> **配套 3D 离线动画**：[notes/learn-ai-visualizer.html](./learn-ai-visualizer.html)（单文件离线版，切到 **Step 03** 体验因果自注意力 3D 建模与训练/推理一键切换）

---

## 1. 为什么必须发明 Self-Attention？

回顾前两关的演化路线：

| 阶段 | 模型结构 | 能看到多远？ | 暴露出的局限 |
| :--- | :--- | :--- | :--- |
| **Step 01** | Bigram 查表 | 仅 **1 个字** | 「前明**月**→光」与「望明**月**→。」完全分不清，Loss 卡死在 `1.25` |
| **Step 02** | 3 字拼接 MLP | 固定 **3 个字** | 遇到「床前明月光，疑是」，只看末尾 3 字「，疑是」，直接串台成《望庐山瀑布》的「银河落九天」 |
| **Step 03** | **因果多头自注意力** | 整个窗口（本关 **16 字**，可动态按需回望任意历史位置） | 用 **更少的参数（28,853 vs 52,141）**，彻底解决「床前明月光，疑是 → 地上霜」长程呼应！ |

**核心痛点**：如果继续沿用 Step 02 的“拼接”思路，把窗口扩大到 16 字，权重矩阵就会暴涨 5 倍，而且每个位置学到的特征不能共享（第 2 位看到「月」和第 10 位看到「月」完全走不同通道）。

**Self-Attention 的天才之处**：不再把历史字硬塞进固定槽位，而是让**当前字动态向所有历史字“打探照灯”**——谁跟我现在最相关，我就多借一点它的信息！

---

## 2. Token + Position Embedding：为什么一定要加位置编码？

自注意力本质上是对一堆向量的**加权求和**。如果不给位置信息：

- 「床前明月」和「明月床前」在点积计算里只是顺序换了，加权求和结果完全一样（置换等变性）！
- 模型根本分不清「床」在句首还是句尾。

因此 GPT 的入口层非常简洁：

$$h_0[t] = E_{\text{tok}}[\text{token}_t] + E_{\text{pos}}[t]$$

- `tok_emb` 形状 $(V, D)$：记住“**这是什么字**”；
- `pos_emb` 形状 $(T_{\max}, D)$：记住“**它排在第几个位置**”；
- 两者直接逐元素相加，就能同时编码“字义 + 位置”（反传时 `d_h0` 原封不动同时累加给 `tok_emb` 对应行和 `pos_emb` 对应行）。

---

## 3. Q / K / V 与缩放点积注意力

每个位置的向量 $h_0 \in \mathbb{R}^{T \times D}$ 会通过 3 组可学习线性矩阵，投影出三种角色：

$$Q = h_0 W_Q,\qquad K = h_0 W_K,\qquad V = h_0 W_V$$

用「图书馆查资料」来理解最直观：
- **Query ($Q_i$)**：第 $i$ 个字手里拿的**检索问题**（例如「是」字在问：“前文提的是明月光还是三千尺？”）；
- **Key ($K_j$)**：历史上第 $j$ 个字贴在书脊上的**目录标签**（例如「光」字写着“我是静夜思的月光线索”）；
- **Value ($V_j$)**：历史上第 $j$ 个字真正携带的**内容干货**。

### 三步算出注意力输出

1. **相关度打分（缩放点积）**：
   $$S_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d_{\text{head}}}}$$
   - **为什么要除以 $\sqrt{d_{\text{head}}}$？** 当维度 $d$ 变大时，两个随机向量的点积方差会按 $d$ 放大，导致 Softmax 迅速饱和（几乎变成 `1` 和 `0`，梯度趋近 `0`）。除以 $\sqrt{d_{\text{head}}}$ 把方差拉回 `1`，让梯度健康流动。

2. **Causal Mask（因果下三角掩码）+ Softmax**：
   $$\tilde{S}_{i,j} = \begin{cases} S_{i,j}, & j \le i \\ -\infty, & j > i \end{cases}, \qquad A_{i,j} = \text{softmax}_j(\tilde{S}_{i,j})$$
   - **为什么要遮住 $j > i$？** 训练时我们把一整句 `床前明月光，疑是地上霜。` 一次性并行送进去，同时监督每个位置预测下一个字。如果不遮住右上角，$t=3$ 的「月」就能直接偷看到 $t=4$ 的答案「光」，考试作弊导致推理时彻底抓瞎！
   - 设为 $-\infty$ 后，经过 $e^{-\infty} = 0$，未来位置的注意力权重严格为 `0`。我们在单测 `test_causal_mask_no_future_leak` 中专门验证了：**把未来字符全部改掉，前面所有位置的 logits 误差严格 $< 10^{-6}$**！

3. **加权汇聚 Value + 残差连接**：
   $$\text{Context}_i = \sum_{j=0}^{i} A_{i,j} V_j, \qquad h_1 = h_0 + \text{Context} \cdot W_O$$
   - 加上残差连接 `h0 + attn_out` 极其关键：它既保留了“当前字本身是什么”，又叠加了“从前文借来的上下文线索”，同时为梯度提供了一条直达底层 Embedding 的高速公路。

### Multi-Head（多头）的意义

本关把 `D_MODEL = 32` 均分成 `N_HEAD = 4` 个头（每头 `d_head = 8`）：
- 头 1 可以专心盯紧相邻前一个字（类似 Bigram 语法衔接）；
- 头 2 可以跨过标点回头找整句主题词（如从「是」跳回 3 步前的「光」）；
- 多个头并行算完后直接拼接回 `D = 32` 维，再经 $W_O$ 融合。

---

## 4. 自注意力的精确反向传播（Tape 实现）

设上游传给单个头的梯度为 $dC = \frac{\partial L}{\partial \text{Context}} \in \mathbb{R}^{T \times d_h}$，前向保存的下三角权重矩阵为 $A \in \mathbb{R}^{T \times T}$：

1. **对 $V$ 与权重 $A$ 求导**（因为 $C = A V$）：
   $$dV_j \mathrel{+}= \sum_{i \ge j} A_{i,j}\, dC_i, \qquad dA_{i,j} = dC_i \cdot V_j \quad (j \le i)$$
2. **穿过 Softmax 的 Jacobian 反传**（逐行 $i$ 计算）：
   $$dS_{i,j} = A_{i,j} \left(dA_{i,j} - \sum_{k=0}^{i} A_{i,k}\, dA_{i,k}\right) \cdot \frac{1}{\sqrt{d_h}}$$
3. **对 $Q$ 与 $K$ 求导**（因为 $S_{i,j} = \frac{1}{\sqrt{d_h}} Q_i \cdot K_j$）：
   $$dQ_i \mathrel{+}= \sum_{j=0}^{i} dS_{i,j}\, K_j, \qquad dK_j \mathrel{+}= \sum_{i=j}^{T-1} dS_{i,j}\, Q_i$$

在 [crates/step03-attention/src/main.rs](../crates/step03-attention/src/main.rs) 中，单测 `test_attention_gradient_check` 对全部 8 组参数（`tok_emb, pos_emb, wq, wk, wv, wo, w_lm, b_lm`）逐元素做中心有限差分校验，解析梯度与数值梯度误差全部 $< 2 \times 10^{-3}$！

---

## 5. 实战运行结果（纯 Rust 无框架）

运行：
```bash
cargo test
cargo run --release -p step03-attention
```

实测结果：
- **模型规模**：`28,853` 参数（`T=16, D=32, 4 头`），比 Step 02（`52,141` 参数）更轻，视野宽了 **5.3 倍**；
- **训练收敛**：1200 step 耗时约 `8.95s`，Loss 从 `5.98` 迅速降至 `~0.12 - 0.18`；
- **成功攻克 Step 02 的三字盲区**：
  ```text
  输入「床前明月光，疑是」 -> 床前明月光，疑是地上霜。举头望明月，低头思故乡。
  输入「白日依山尽，黄河」 -> 白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
  ```
- **实测因果注意力权重（4 头平均，观察「是」如何跨越标点回望「光」）**：
  - 位置 `月`（预测「光」）：**45% 盯「明」、35% 盯「前」、19% 盯「月」**；
  - 位置 `是`（预测「地」）：**54% 盯「疑」、23% 直接跨 3 格回望「光」**——正是这束 23% 的长程注意力，让模型认出了这是《静夜思》而不是《望庐山瀑布》！

---

## 6. 留给 Step 04（完整 Mini-GPT）的伏笔

细看上面的输出，你会发现一个极有价值的真实现象：
```text
输入「春眠不觉晓，」 -> 春眠不觉晓，处处处处处处处处处处...
```
**为什么会出现「处处处...」复读？**
因为 Step 03 只有**单层纯线性 Attention + Residual**，还缺少 Transformer Block 的另外两块关键拼图：
1. **LayerNorm（层归一化）**：防止多层叠加和长序列下特征数值漂移；
2. **FFN（前馈非线性 MLP：Linear → GELU/ReLU → Linear）+ 多层堆叠**：Attention 负责**跨位置收集信息**，FFN 负责**在每个位置做深层非线性推理与计数（比如记住「处」已经出现两次，下次该接「闻」）**！

下一关 **Step 04**，我们把 **LayerNorm + Causal Multi-Head Attention + FFN + Residual** 拼成真正的 **Pre-Norm Transformer Block** 并堆叠多层，迎来完整的纯 Rust **Mini-GPT**！
