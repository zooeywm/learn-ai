# Step 07 · UTF-8 字节级 BPE

Step 06 的字符词表来自训练语料，遇到词表外的字符会跳过。Step 07 从 256 种原始字节出发，连训练时没见过的 `昵称`、`换一个`、`Rust 🚀` 也能无损编码。分词器把完整的输入交给模型，回答是否正确仍取决于语言模型的训练数据与学习结果。

[Rust 分词器](../crates/step07-tokenizer/src/bpe.rs) · [单文件 3D 演进页](learn-ai-visualizer.html)

## 训练并进入对话

```bash
cargo run --release -p step07-tokenizer -- train
cargo run --release -p step07-tokenizer -- chat
cargo run --release -p step07-tokenizer -- bench
```

训练先用 326 首唐诗和 1,223 条问答学习合并规则，再用新 token ID 重新预训练和 SFT。当前微调集有 3,846 个采样条目，包括原始问答、改写、诗句片段和重复加权的人工样本。模型采用 RMSNorm、RoPE、GQA 和 SwiGLU。存档 `checkpoints/step07-bpe.mllm` 同时保存合并规则和模型权重。用 `--pretrain-steps N`、`--sft-steps N` 和 `--refine-steps N` 调整各阶段步数，用 `--model 路径` 指定实验存档。

| 项目 | 默认值 |
| --- | --- |
| 模型 | 3 层，96 维，4 个 Q 头和 2 个 KV 头，SwiGLU 256 维 |
| 词表 / 上下文 | 1,152 / 144 tokens |
| 参数量 | 525,984 |
| 预训练 | 1,500 步，batch 8，每条 32 tokens，峰值学习率 0.0045 |
| SFT | 2,100 步，batch 8，峰值学习率 0.003 |
| 问法与片段微调 | 1,100 步，batch 8，峰值学习率 0.001，按 token 长度组批 |

已有存档时，可以单独运行 `cargo run --release -p step07-tokenizer -- refine` 继续做问法微调。

片段练习从短诗中截取整句和三字片段，并在全部 326 首唐诗中检查出处。若同一片段出现在不同作品里，训练答案会请求补充诗名、作者或更长的片段。人工问答也涵盖已知错字、助手身份和用户身份。 [人工片段样本](../data/step07/fragment-qa.tsv)

在本机 Ryzen 7 8845H 上，发布构建的 `bench` 使用五个问题，每题重复 32 次，每轮共产生 4,352 个 token。当前存档三次测量的 decode 吞吐中位数为 **16,702 tok/s**，计入选词和 UTF‑8 检查，排除预填充和终端输出。

当前存档以片段微调 1,100 步后的权重占 70%，再做 420 步加权重放后的权重占 30% 生成。选取这个比例时，使用了截图中的 8 条输入和原有的 12 条问法作为回归检查，分别通过 8/8 和 11/12。另外测试的 6 条未训练片段与错字均未通过，包括「地上霜」和「疑是地下霜」。

语料以《唐诗三百首》为主，用 OpenCC 转成简体、按题名与作者去重，并补充《望庐山瀑布》等常见篇目。`静夜思` 和 `相思` 采用本课使用的版本。全篇背诵的问答使用 64 字以内的作品，训练还包括首句续写、作者问答和 64 条人工样本。 [上游语料快照](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/%E8%92%99%E5%AD%A6/tangshisanbaishou.json) · [上游 MIT 许可证](../data/step07/LICENSE-chinese-poetry)

在默认对话中输入 `思故乡`、`低头思故乡` 或 `低头思故乡是哪个诗里的`，都会从本地唐诗库找到李白《静夜思》。`诗库 >` 显示从原文取得的答案，`助手 >` 显示模型生成的答案。片段有多个出处时，诗库列出候选。明确问出处但找不到原句时，会显示未找到。若短片段删掉开头一字后恰好匹配一首诗，诗库会提示可能的出处，例如 `巨举头`。

用 `cargo run --release -p step07-tokenizer -- chat --raw` 可以直接观察模型生成。输入 `/debug` 查看输入的 token ID、字节分片，以及屏蔽非法候选之后的 Top‑3 概率（按温度 1 计算）。每行独立处理，输入 `/quit` 退出。

例如，`昵称` 的 UTF‑8 字节始终是 `E6 98 B5 E7 A7 B0`，具体的 token 边界由合并规则决定。生成每个 token 前，Step 07 会屏蔽无法接成合法 UTF‑8 的候选。若最后一个字符还缺字节，解码会保留足够的位置补齐后再结束，控制台也会等字符完整后再输出。

## 词表怎么学出来

1. 保留 4 个控制符，将 256 个单字节分配到 ID 4 到 259。
2. 在每行内统计相邻 token 对。将最常出现的一对合并，分配新 ID，更新语料后重新计数。频次相同按 ID 顺序选择。
3. 到 1,152 个词条、或再无出现至少两次的候选对时停止。单个词条最长 9 字节。编码新文本时按训练顺序重放合并，解码时把各 token 的原始字节按顺序拼回去。

在演进页打开 Step 00，切到「字节级 BPE」。依次点「统计相邻对」和「合并最多的一对」，观察两块怎样合成一个新 token。下方可以用当前规则测试新文本。演进页使用三行短语料来显示逐轮变化。

## 对照 OpenAI 的公开编码

`tiktoken-rs` 可加载 GPT‑4o 的 `o200k_base`，用于观察其分词。用这个约 20 万 ID 的词表训练本课模型，单是 96 维 embedding 和输出层就需要约 3,840 万参数。 [OpenAI 模型编码映射](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py) · [OpenAI 的 BPE 教学实现](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py)

```bash
cargo run --release -p step07-tokenizer --features openai -- inspect 'hello world 🦀'
```

省略 `inspect` 后面的文本可连续输入多行，输入 `/quit` 退出。
