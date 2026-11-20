# Step 06：现代工业界 LLM 四大核心升级（RMSNorm + RoPE + GQA + SwiGLU）

> 对应源码：[`crates/step06-modern-llm/src/main.rs`](../crates/step06-modern-llm/src/main.rs)  
> 3D 交互演进演示（单文件离线版，支持 Step 00 → Step 06 全景切换）：[`notes/learn-ai-visualizer.html`](learn-ai-visualizer.html)

## 一、从 GPT-2（2019）到 LLaMA 3 / DeepSeek / Qwen（2024~2026）

在 Step 04 和 Step 05 里，我们手搓出了完整的 GPT-2 架构并做了 SFT 对话微调与 KV Cache 推理。  
而当今所有主流开源与闭源大模型（LLaMA 3、DeepSeek-V3、Qwen 2.5、Gemma、Mistral），在 Decoder-Only 骨架之上，几乎无一例外完成了**同一套「现代四件套」升级**：

| 模块 | Step 04 / 05（GPT-2 经典版） | Step 06（现代工业界标准版） | 核心收益 |
| :--- | :--- | :--- | :--- |
| **归一化** | `LayerNorm`（减均值 + 除标准差 + $\gamma,\beta$） | **`RMSNorm`**（不减均值、无 $\beta$，只除均方根） | 省去均值计算与偏置参数，GPU 融合算子快 10%～20% |
| **位置编码** | 绝对位置表 `pos_emb[t]`（固定查表） | **`RoPE` 旋转位置编码**（零参数，旋转 Q/K） | 注意力点积只依赖**相对距离 $(m-n)$**，不再怕问题长短错位，支持超长上下文外推 |
| **注意力机制** | `MHA`（4 个 Q 头配 4 个 K/V 头） | **`GQA` 分组查询注意力**（4 个 Q 头共享 2 个 K/V 头） | 效果几乎无损，**KV Cache 显存直接减半（省 50%！）** |
| **前馈网络 FFN** | `Linear(D→4D) → GELU → Linear(4D→D)`（带 Bias） | **`SwiGLU`** `((SiLU(xW_gate) ⊙ xW_up) W_down)`（无 Bias） | 动态门控逐维开关特征，全程零偏置，收敛与表达力全面更强 |

---

## 二、四大升级的数学原理与手搓细节

### 1. RMSNorm（均方根归一化）
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D}\sum_{c=1}^{D} x_c^2 + \epsilon}} \odot \gamma$$
研究表明，LayerNorm 真正稳定深层训练的关键是「**重缩放不变性（除长度）**」，而不是「减去均值平移」。砍掉均值减法和 $\beta$ 偏置后，反向传播也更简洁：
$$\frac{\partial \mathcal{L}}{\partial x} = \text{inv\_rms} \cdot \left(g - \hat{x} \cdot \text{mean}(g \odot \hat{x})\right)$$

### 2. RoPE（旋转位置编码，Rotary Position Embedding）
**彻底删除 `pos_emb` 矩阵（0 个位置参数！）**。在每一层 Attention 算出 `Q` 和 `K` 之后，把每个 head 的相邻两维 $(x_{2k}, x_{2k+1})$ 看成二维平面向量，按当前位置 $m$ 旋转角度 $m\theta_k$（其中 $\theta_k = 10000^{-2k/d_{\text{head}}}$）：

$$\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}$$

由三角函数和差公式，位置 $m$ 的 Query 与位置 $n$ 的 Key 做内积时：
$$\langle R_m q,\; R_n k \rangle = q^\top R_m^\top R_n k = q^\top R_{n - m} k$$
结果**只与相对距离 $(m - n)$ 有关**！且因为旋转矩阵是正交矩阵，反向传播极其优雅——只需把梯度**逆旋转 $-m\theta_k$**！

### 3. GQA（Grouped-Query Attention，分组查询注意力）
大模型长上下文推理的真正瓶颈不是算力，而是 **KV Cache 吃爆显存与带宽**：
- **MHA**（Step 05）：4 个 Q 头各存一套 K/V，每层每字存 `2 × 32 = 64` 个浮点数；
- **GQA**（Step 06）：4 个 Q 头分成 2 组，**每 2 个 Q 头共享 1 个 K/V 头**（`Wk, Wv` 从 `32×32` 瘦身为 `32×16`），每层每字只存 `2 × 16 = 32` 个浮点数——**KV Cache 体积严格砍半（省 50%！）**。
- 反向传播时，同组的多个 Query 头把梯度累加回它们共享的那一个 K/V 头即可。

### 4. SwiGLU（门控前馈网络，无偏置）
$$\text{SwiGLU}(x) = \left(\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})\right) W_{\text{down}}, \quad \text{SiLU}(z) = \frac{z}{1 + e^{-z}}$$
为保持与 GPT-2 `4D` 相当的参数量，工业界取中间维度 $d_{\text{ff}} \approx \frac{8}{3}D$（本程序 `D=32, d_ff=88`）。

---

## 三、纯 Rust 实测战绩（更少参数 + 省 50% KV Cache + 更快收敛！）

| 指标 | Step 05（GPT-2 经典版） | Step 06（现代四件套版） | 变化 |
| :--- | :---: | :---: | :---: |
| **总参数量** | `52,947` | **`48,992`** | **减少 3,955 参数（-7.5%，`pos_emb=0`、无 Bias）** |
| **模型存档体积** | `209.7 KB` | **`194.3 KB`** | 更精炼 |
| **SFT 初始 Loss → 终态 Loss** | `7.8420 → 0.0002` | **`3.5047 → 0.0002`** | 初始适配显著更平滑 |
| **「背一首静夜思」KV Cache 占用** | `4,096 floats` | **`2,048 floats`** | **严格省 50% 显存！** |

---

## 四、运行与控制台交互（支持 `/debug` 概率透视！）

```bash
# 运行全仓库 19 个单元测试（含 RoPE 相对距离不变性、GQA 省 50% 校验、全网数值梯度校验）
cargo test

# 训练并保存现代底座到 checkpoints/step06-modern.mllm
cargo run --release -p step06-modern-llm

# 毫秒级加载现代模型进入控制台交互（输入 /debug 可实时查看每步 Top-3 概率！）
cargo run --release -p step06-modern-llm -- chat
```
