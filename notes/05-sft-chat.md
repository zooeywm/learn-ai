# Step 05：SFT 指令微调（把续写机变成迷你 ChatGPT）+ KV Cache（Prefill / Decode 两阶段加速）

> 对应源码：[`crates/step05-sft-chat/src/main.rs`](../crates/step05-sft-chat/src/main.rs)  
> 指令微调数据集（标准 JSONL）：[`data/sft_poems.jsonl`](../data/sft_poems.jsonl)  
> 3D 交互演进演示（单文件离线版，支持 Step 00 → Step 05 全景切换）：[`notes/learn-ai-visualizer.html`](learn-ai-visualizer.html)

## 一、这一关最震撼的真相：网络结构一行不改！

在 Step 04 结束时，我们手搓出了一个完整的 2 层 Pre-Norm Mini-GPT，它能把 5 首唐诗全部满分续写。但如果你问它：「**你好**」或者「**背一首静夜思**」，它根本不会「回答你」，而是会继续像复读机一样乱接唐诗，而且永远不知道什么时候该闭嘴！

在工业界，刚预训练完的模型叫做 **Base Model（基座模型 / 续写机）**；而能听懂人话、按指令回答、说完自动停下来的 ChatGPT / DeepSeek Chat，叫做 **Chat / Instruct Model（指令微调模型）**。

**最令人意外的是：从 Base 变成 Chat，Transformer 内部代码一行都不用改！**  
依然是同一个 2 层 Pre-Norm Decoder-Only Mini-GPT（`LN1 → CausalAttn → +Res → LN2 → GELU FFN → +Res` × 2 → `Final LN` → `LM Head`）。我们只做了 **3 件事**：

---

## 二、核心机制 1：Chat Template 与 4 个特殊控制符

既然模型本质上永远只是在做「预测下一个 Token」，那怎么让它分辨「哪段是用户在提问、哪段轮到我回答、什么时候该闭嘴」？

答案极其朴素：在词表里预留 **4 个特殊控制符（Special Tokens）**：

| Token | ID | 作用 |
| :--- | :---: | :--- |
| `<\|pad\|>` | `0` | 把一个 Batch 里长短不一的对话右对齐补齐（不参与 Loss，且因因果掩码不会影响前面真实字） |
| `<\|user\|>` | `1` | 标记「用户开始说话」 |
| `<\|assistant\|>` | `2` | 标记「用户说完了，现在轮到助手开始回答！」 |
| `<\|eos\|>` | `3` | **End Of Sequence（停止位）**——模型一旦输出它，推理循环立刻停机闭嘴！ |

每一条问答样本都会被拼成一条标准对话序列：

```text
<|user|> 你 好 <|assistant|> 你 好 ！ 我 是 迷 你 唐 诗 助 手 。 <|eos|>
```

推理时，我们只需把用户的问题包装成前缀 `<|user|> 问题 <|assistant|>` 喂进去，模型就会顺着接龙预测回答正文，直到它自己吐出 `<|eos|>`！

---

## 三、核心机制 2：Assistant-Only Loss Mask（只读题不扣分！）

做 SFT（Supervised Fine-Tuning，有监督指令微调）时，有一个极容易踩坑的细节：

> 如果把整条对话的所有位置都算交叉熵 Loss，模型就会被迫去「背诵用户怎么提问」，甚至学会自己替用户接着提问！

工业界 SFT 的标准做法是 **Assistant-Only Loss Mask（仅助手区掩码交叉熵）**：

$$
\mathcal{L}_{\text{SFT}} = \frac{\sum_{t=0}^{T-1} m_t \cdot \left(-\ln p_t[y_t]\right)}{\max\left(1, \sum_{t=0}^{T-1} m_t\right)}
$$

其中掩码 $m_t \in \{0, 1\}$：
- **提问区 & Padding（$m_t = 0$）**：模型在前向传播时照常通过 Self-Attention 阅读整句提问（理解题意），但在算 Loss 和反传梯度时直接乘以 `0`（**只读题、不扣分、零梯度**）！
- **助手回答区 + `<|eos|>`（$m_t = 1$）**：从 `<|assistant|>` 预测回答第 1 个字开始，一直到最后一个字预测 `<|eos|>`，全部计算 Loss 与梯度！

以本程序实测的第 0 条样本为例（总步长 16，屏蔽 3 步，计分 13 步）：

```text
<|user|>->你[0] 你->好[0] 好-><|assistant|>[0] <|assistant|>->你[1] 你->好[1] ... 手->。[1] 。-><|eos|>[1]
```

注意一个精妙的细节：
- `mask = 0` 并不是「不让模型看问题」！提问区的 Token 在前向依然进入每一层的 Self-Attention，给后面的回答提供 Key / Value 线索；
- 反向传播时，虽然提问位置自身的 `dL/dlogits = 0`，但回答位置的 Attention 梯度依然会顺着 `K / V` 传回到提问 Token 的特征权重里——也就是说，模型不是学「怎么复述问题」，而是学「**看到这样的问题，应该怎样组织回答**」！

---

## 四、核心机制 3：Prefill + Decode 与 KV Cache 推理加速

我们之前聊过一个常见误区：现代大模型推理分成 **Prefill（预填充）** 和 **Decode（解码）** 两阶段，并不是因为模型里有「独立的 Encoder + Decoder 两个架构」，而是**同一个 Decoder-Only 模型在推理时的两种计算形态**，核心枢纽就是 **KV Cache**：

| 阶段 | 输入长度 | 怎么算？ | KV Cache 干什么？ |
| :--- | :---: | :--- | :--- |
| **Prefill（预填充）** | 整句 Prompt（一次吃掉 $T_p$ 个字） | 全部 Prompt Token 并行通过 2 层 Transformer | 把每一层算出来的历史 $K_0 \dots K_{T_p-1}$ 和 $V_0 \dots V_{T_p-1}$ **全部存进缓存库** |
| **Decode（逐字解码）** | 每次仅 **1 个新字** | 只算这 1 个新字的 $q_{\text{new}}, k_{\text{new}}, v_{\text{new}}$，去查缓存库里的全部历史 $K/V$ | 每次把新的 $k_{\text{new}}, v_{\text{new}}$ **追加到缓存库末尾**，直到吐出 `<\|eos\|>` 停机 |

因为因果掩码保证「历史 Token 永远看不到未来 Token」，所以历史 Token 在每一层算出来的 `K` 和 `V` **永远不会再变**！
- **无 KV Cache 朴素推理**：每多生成 1 个字，都要把前面所有字从头再算一遍，总计算量 $O(T^2)$；
- **开启 KV Cache 后**：每个历史字只算一次，总计算量降为 $O(T)$，且数学上与全量重算 **100% 逐位等价**（单测 `test_kv_cache_matches_full_forward` 验证误差 `< 1e-5`）！

---

## 五、纯 Rust 实测结果（Base 翻车 vs SFT 蜕变 + KV Cache 测速）

运行命令：

```bash
cargo test
cargo run --release -p step05-sft-chat
```

### 1. 两阶段训练收敛曲线

- **模型参数**：`2层 Pre-Norm Mini-GPT (D=32, H=4, block_size=48, V=403)`，总参数量 **52,947**
- **Stage 1 预训练（Base）**：900 steps，耗时 **23.46s**，Loss `6.6902 → 0.0490`
- **Stage 2 SFT 微调（Chat）**：18 条唐诗问答，360 steps，耗时 **4.07s**，Masked Loss `7.8420 → 0.0002`

### 2. Base vs SFT 真实对话对照表

| 用户提问 | 刚预训练完的 Base 模型（只会接龙，不懂停机） | SFT 微调后的迷你 ChatGPT（Prefill + Decode KV Cache） |
| :--- | :--- | :--- |
| `你好` | `照我还。京口瓜洲一水间，钟山只隔数重山。 / 竹外桃` (`hit_eos=false`) | `你好！我是迷你唐诗助手。` (`hit_eos=true`, 16 槽位) |
| `背一首静夜思` | `丝绦。不知细叶谁裁出，二月春风似剪刀。 / 野旷天下` (`hit_eos=false`) | `床前明月光，疑是地上霜。举头望明月，低头思故乡。` (`hit_eos=true`, 32 槽位) |
| `白日依山尽的下一句` | `水暖鸭先知。蒌蒿满地芦芽短，正是河豚欲上青苔上时` (`hit_eos=false`) | `黄河入海流。` (`hit_eos=true`, 17 槽位) |
| `飞流直下三千尺的下一句` | `禾日还。 / 只是银河落九天。但使龙城飞将在，二月近` (`hit_eos=false`) | `疑是银河落九天。` (`hit_eos=true`, 21 槽位) |
| `背一首春晓` | `水间，处闻啼鸟。夜半钟声到客船。 / 停车坐爱枫林晚` (`hit_eos=false`) | `春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。` (`hit_eos=true`, 31 槽位) |

### 3. Prefill + Decode KV Cache 加速实测

连续 30 次回答「背一首静夜思」（生成 25 个 Token 含 `<|eos|>`）：
- **逐 Token 一致性**：与无缓存全量重算 **100% 精确一致**
- **无缓存全量重算（30 次）**：`359.33 ms`
- **Prefill + Decode KV Cache（30 次）**：`9.13 ms`
- **实测加速比**：**39.36x**（同时消除了重复计算历史 Token 与构建训练计算图的开销）

---

## 六、模型保存（二进制 Checkpoint）与控制台流式交互对话

训练好的模型默认以零依赖二进制格式 `MGPT v1` 保存到 `checkpoints/step05-chat.mgpt`（约 **210 KB**，内含超参数 + 完整词表 `V=403` + 全部 `52,947` 个 `f32` 权重），日后调用无需重新训练，也不再依赖原始语料文件，**毫秒级加载即可对话**：

```bash
# 1) 训练并保存权重到 checkpoints/step05-chat.mgpt
cargo run --release -p step05-sft-chat

# 2) 直接加载已保存的模型进入控制台流式交互（支持 /temp <T>、/help、/quit）
cargo run --release -p step05-sft-chat -- chat

# 3) 或者训练保存后立刻进入交互对话
cargo run --release -p step05-sft-chat -- train --chat
```

## 七、单元测试验证（全仓库 15/15 通过）

Step 05 与 `mini-core` 新增 5 个核心单测：
1. `test_load_and_parse_sft_jsonl`（`mini-core`）：校验零依赖 JSONL 解析器（含转义引号/换行）与真实 `data/sft_poems.jsonl` 文件加载；
2. `test_masked_cross_entropy_ignores_prompt`：篡改 `mask=0` 位置的 target，验证 Loss 与有效位置梯度不变，且被屏蔽位置梯度**严格为 0.0**；
3. `test_kv_cache_matches_full_forward`：验证 `prefill(seq[..4]) + decode_step(seq[4..])` 每一步的 logits 与完整 `forward` 误差 `< 1e-5`；
4. `test_sft_masked_gradient_check`：在带 0/1 掩码与 padding 的 2 层 Mini-GPT 上完成有限差分数值梯度检验（相对误差 `< 3e-3`）；
5. `test_checkpoint_save_and_load_roundtrip`：验证二进制模型存档保存并重新加载后，词表与 `prefill` 输出 logits **逐位 100% 相等**。
