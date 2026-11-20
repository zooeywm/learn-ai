# Step 08 · DeepSeek-V4.1-Flash 文本主干

这个阶段保留官方主干的 40 层拓扑（层编号 0–39），用较窄的 CPU 配置训练，便于检查因果关系、缓存交接和反向传播。配套数据是八条普通问答。运行时通过学生自己的 BPE 编码文本，模型权重从固定种子随机初始化。

## 层与缓存

| 部位 | Step 08 的实现 |
| --- | --- |
| Embedding / Engram | 复用 Step 07 的字节级 BPE。在 L1 和 L14 用 2、3、4-gram 查各自的哈希桶，经 key/value 投影和归一化门控写入四条残差流。小配置对本地 token ID 哈希。 |
| Encoder L0–L19 | L0 和 L1 用滑窗。L2、L8、L14 各写一个全局 KV 缓存，将每两个位置用逐维 softmax 门控合成一个 KV。后续层复用最近一个来源层的缓存和索引。 |
| Decoder L20–L39 | L20 的全局 KV 从 Encoder 输出经 L20 的注意力归一化和投影构造，压缩比为 1。L20 选候选块，L24、L28、L32、L36 在这些候选块内重评分，其余层复用索引。每层同时维护自己的滑窗。 |
| 单次 mHC | 四条残差流。每个子层生成供下一个子层使用的 pre 权重、当前 post 权重，以及经 Sinkhorn 归一化的残差混合矩阵。 |
| MoE | 每层有一个共享 SwiGLU 专家和 12 个路由专家，每个 token 走得分最高的六个路由专家。将 `sqrt(softplus(score))` 归一化并缩放为混合权重。 |
| 词表输出 | 末层按 pre 权重聚合四条残差流，再经 RMSNorm 和输出投影计算 logits。 |

CPU 配置的隐藏维度为 8、注意力头数为 2、滑窗为 8、上下文上限为 128 token。模型和缓存采用 FP32。离散 Top‑K 索引通过硬选择使用，主语言模型损失不会直接训练索引打分器。当前实现覆盖文本主干，视觉分支、DSpark、官方权重格式、FP4/FP8 和 CUDA 并行属于额外的实现范围。

## 训练和生成

```bash
cargo run --release -p step08-deepseek-flash -- demo
cargo run --release -p step08-deepseek-flash -- generate checkpoints/step08.ds08 ping
```

训练序列为「用户标记 + 问题 + 助手标记 + 答案 + EOS」。问题参与前向，只有答案和 EOS 进入交叉熵。先测初始平均 loss，再训练、存档和生成。这个数据集用于冒烟验证，loss 和同题输出反映拟合情况。

首次冷启动 Prefill 先跑完输入的 Encoder 并写全局 KV，再依次跑 Decoder 暖满各层滑窗。Decode 每步只追加一个新 token 的 Encoder / Decoder 状态。当前冷启动使用整段 Decoder 暖缓存，可用作以后加入 SWA bounded replay 的对照。测试核对分段缓存路径与整段前向的 logits、未来 KV 的因果遮罩，以及经共享 KV 回到 Encoder 的梯度。

## 接入老师答案

把已验证的老师答案整理成 UTF-8 JSONL。每行形如：

```json
{"user":"Name a fruit.","assistant":"Apple."}
```

```bash
cargo run --release -p step08-deepseek-flash -- train teacher.jsonl 400 checkpoints/teacher.ds08
cargo run --release -p step08-deepseek-flash -- generate checkpoints/teacher.ds08 'Name a fruit.'
```

`train` 会用传入的样本重新建学生词表并从头初始化小模型。单条样本的输入和答案合计至多 128 个训练输入 token，超长样本会报出行号。

参考：[DeepSeek‑V4.1‑Flash 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) · [1017 推理参考代码和配置](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/main/inference)。
