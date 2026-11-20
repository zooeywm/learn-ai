//! Step 08: CPU reference implementation of the DeepSeek-V4.1-Flash text backbone.
mod model;
mod tape;

use std::{
    env, fs,
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use model::{History, MAX_CONTEXT, Model};
use serde::Deserialize;
use step06_modern_llm::{ChatEncoding, EOS_ID};
use step07_tokenizer::ByteBpe;
use tape::Tape;

const USER: usize = 1;
const ASSISTANT: usize = 2;
const DEFAULT_STEPS: usize = 400;
const DATA: &str = include_str!("../../../data/step08/chat.jsonl");
const MAGIC: &[u8; 4] = b"DS08";

#[derive(Deserialize)]
struct Example {
    user: String,
    assistant: String,
}

fn parse_dataset(data: &str) -> Result<Vec<Example>, Box<dyn std::error::Error>> {
    let mut examples = Vec::new();
    for (i, line) in data.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let item: Example =
            serde_json::from_str(line).map_err(|e| format!("JSONL 第 {} 行：{e}", i + 1))?;
        if item.user.trim().is_empty() || item.assistant.trim().is_empty() {
            return Err(format!("JSONL 第 {} 行的 user 或 assistant 为空", i + 1).into());
        }
        examples.push(item);
    }
    if examples.is_empty() {
        return Err("训练集没有样本".into());
    }
    Ok(examples)
}

fn prompt(tokenizer: &ByteBpe, user: &str) -> Vec<usize> {
    let mut ids = tokenizer.encode(user).0;
    ids.insert(0, USER);
    ids.push(ASSISTANT);
    ids
}

struct Adam {
    first: Vec<f32>,
    second: Vec<f32>,
    decay_1: f32,
    decay_2: f32,
}

impl Adam {
    fn new(size: usize) -> Self {
        Self {
            first: vec![0.0; size],
            second: vec![0.0; size],
            decay_1: 1.0,
            decay_2: 1.0,
        }
    }

    fn step(&mut self, weights: &mut [f32], grad: Vec<f32>, rate: f32) {
        let norm = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip = norm.max(1.0).recip();
        self.decay_1 *= 0.9;
        self.decay_2 *= 0.999;
        for (((w, m), v), g) in weights
            .iter_mut()
            .zip(&mut self.first)
            .zip(&mut self.second)
            .zip(grad)
        {
            let g = g * clip;
            *m = 0.9 * *m + 0.1 * g;
            *v = 0.999 * *v + 0.001 * g * g;
            let mean = *m / (1.0 - self.decay_1);
            let variance = *v / (1.0 - self.decay_2);
            *w -= rate * mean / (variance.sqrt() + 1e-8);
        }
    }
}

fn train(data: &str, steps: usize) -> Result<(ByteBpe, Model), Box<dyn std::error::Error>> {
    if steps == 0 {
        return Err("训练步数必须大于 0".into());
    }
    let examples = parse_dataset(data)?;
    let texts: Vec<_> = examples
        .iter()
        .flat_map(|e| [e.user.as_str(), e.assistant.as_str()])
        .collect();
    let tokenizer = ByteBpe::train(&texts);
    let mut model = Model::new(tokenizer.vocab_size());
    let mut optimizer = Adam::new(model.weights.len());
    let mut samples = Vec::with_capacity(examples.len());
    for (i, example) in examples.into_iter().enumerate() {
        let mut ids = prompt(&tokenizer, &example.user);
        let answer_start = ids.len();
        ids.extend(tokenizer.encode(&example.assistant).0);
        ids.push(EOS_ID);
        if ids.len() - 1 > MAX_CONTEXT {
            return Err(format!("第 {} 条样本超过 {MAX_CONTEXT} token 的上下文上限", i + 1).into());
        }
        samples.push((ids, answer_start));
    }
    let evaluate = |model: &Model| -> f32 {
        samples
            .iter()
            .map(|(ids, start)| {
                let mut tape = Tape::new(&model.weights, false);
                let loss = model.loss(&mut tape, ids, *start);
                tape.value(loss)
            })
            .sum::<f32>()
            / samples.len() as f32
    };

    println!(
        "文本主干 20 层 Encoder + 20 层 Decoder · {} 参数 · {} 条训练样本",
        model.weights.len(),
        samples.len()
    );
    println!("初始平均 loss: {:.4}", evaluate(&model));
    for step in 0..steps {
        let (ids, answer_start) = &samples[step % samples.len()];
        let mut tape = Tape::new(&model.weights, true);
        let loss = model.loss(&mut tape, ids, *answer_start);
        let value = tape.value(loss);
        if !value.is_finite() {
            return Err(format!("第 {} 步 loss 非有限值", step + 1).into());
        }
        let grad = tape.backward(loss, model.weights.len());
        if grad.iter().any(|g| !g.is_finite()) {
            return Err(format!("第 {} 步梯度非有限值", step + 1).into());
        }
        optimizer.step(&mut model.weights, grad, 0.002);
        if step == 0 || (step + 1) % 20 == 0 {
            println!("step {:>4}/{steps}, loss: {value:.4}", step + 1);
        }
    }
    println!("训练后平均 loss: {:.4}", evaluate(&model));
    Ok((tokenizer, model))
}

fn can_finish_utf8(bytes: &[u8], remaining: usize) -> bool {
    match std::str::from_utf8(bytes) {
        Ok(_) => true,
        Err(err) if err.error_len().is_none() => {
            let incomplete = &bytes[err.valid_up_to()..];
            let width = match incomplete[0] {
                0xc2..=0xdf => 2,
                0xe0..=0xef => 3,
                _ => 4,
            };
            width - incomplete.len() <= remaining
        }
        Err(_) => false,
    }
}

fn generate(
    tokenizer: &ByteBpe,
    model: &Model,
    question: &str,
    limit: usize,
) -> io::Result<String> {
    let mut ids = prompt(tokenizer, question);
    if ids.len() >= MAX_CONTEXT || limit == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "输入超出上下文上限或生成预算为 0",
        ));
    }
    let mut tape = Tape::new(&model.weights, false);
    let mut history = History::new();
    let mut logits = model.prefill(&mut tape, &mut history, &ids);
    let (local, shared) = history.cache_positions();
    eprintln!(
        "Prefill {} tokens · 滑窗 KV {local} 条 · 共享 KV {shared} 条",
        ids.len()
    );

    let mut text = Vec::new();
    let budget = limit.min(MAX_CONTEXT - ids.len());
    for index in 0..budget {
        let last = index + 1 == budget;
        let token = (EOS_ID..logits.len())
            .filter(|&candidate| {
                if candidate == EOS_ID {
                    return std::str::from_utf8(&text).is_ok();
                }
                let mut bytes = text.clone();
                bytes.extend_from_slice(tokenizer.token_bytes(candidate));
                can_finish_utf8(&bytes, budget - index - 1)
            })
            .max_by(|&a, &b| tape.value(logits[a]).total_cmp(&tape.value(logits[b])));
        let Some(token) = token.filter(|&id| id != EOS_ID) else {
            break;
        };
        text.extend_from_slice(tokenizer.token_bytes(token));
        if !last {
            ids.push(token);
            logits = model.decode_step(&mut tape, &mut history, &ids);
        }
    }
    Ok(String::from_utf8(text).expect("sampler guarantees valid UTF-8"))
}

fn save(path: &Path, tokenizer: &ByteBpe, model: &Model) -> io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut out = BufWriter::new(fs::File::create(path)?);
    out.write_all(MAGIC)?;
    out.write_all(&(tokenizer.vocab_size() as u32).to_le_bytes())?;
    tokenizer.save_vocab(&mut out)?;
    for value in &model.weights {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()
}

fn load(path: &Path) -> io::Result<(ByteBpe, Model)> {
    let mut input = BufReader::new(fs::File::open(path)?);
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes)?;
    if &bytes != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "不是 Step 08 checkpoint",
        ));
    }
    input.read_exact(&mut bytes)?;
    let tokenizer = ByteBpe::load_vocab(&mut input, u32::from_le_bytes(bytes) as usize)?;
    let mut model = Model::new(tokenizer.vocab_size());
    for value in &mut model.weights {
        input.read_exact(&mut bytes)?;
        *value = f32::from_le_bytes(bytes);
        if !value.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "checkpoint 含非有限权重",
            ));
        }
    }
    if input.read(&mut bytes[..1])? != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint 尾部含多余数据",
        ));
    }
    Ok((tokenizer, model))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let default_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/step08.ds08");
    match args.first().map(String::as_str).unwrap_or("demo") {
        "demo" if args.len() <= 1 => {
            let (tokenizer, model) = train(DATA, DEFAULT_STEPS)?;
            save(&default_path, &tokenizer, &model)?;
            println!("ping → {}", generate(&tokenizer, &model, "ping", 16)?);
        }
        "train" if (2..=4).contains(&args.len()) => {
            let data = fs::read_to_string(&args[1])?;
            let steps = args.get(2).map_or(Ok(DEFAULT_STEPS), |s| s.parse())?;
            let path = args.get(3).map_or(default_path.as_path(), Path::new);
            let (tokenizer, model) = train(&data, steps)?;
            save(path, &tokenizer, &model)?;
            println!("checkpoint: {}", path.display());
        }
        "generate" if (3..=4).contains(&args.len()) => {
            let (tokenizer, model) = load(Path::new(&args[1]))?;
            let limit = args.get(3).map_or(Ok(32), |s| s.parse())?;
            println!("{}", generate(&tokenizer, &model, &args[2], limit)?);
        }
        _ => return Err("用法: demo | train <data.jsonl> [steps] [checkpoint] | generate <checkpoint> <question> [max_tokens]".into()),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn partial_utf8_must_fit_in_the_remaining_generation_budget() {
        assert!(super::can_finish_utf8(&[0xf0, 0x9f], 2));
        assert!(!super::can_finish_utf8(&[0xf0, 0x9f], 1));
        assert!(!super::can_finish_utf8(&[0xc2, 0x00], 2));
        assert!(super::can_finish_utf8("中".as_bytes(), 0));
    }
}
