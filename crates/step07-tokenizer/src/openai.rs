//! Step 07 · 用 `o200k_base` 观察 UTF-8 字节级 BPE。

use std::io::{self, BufRead, IsTerminal, Write};
use std::{error::Error, slice, str};

use tiktoken_rs::{CoreBPE, o200k_base};

pub fn run(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let input = args.next();
    if args.next().is_some() {
        return Err("请用引号把待分词的文本作为一个参数传入".into());
    }

    let tokenizer = o200k_base()?;
    match input {
        Some(input) => inspect_text(&tokenizer, &input),
        None => run_repl(&tokenizer),
    }
}

fn run_repl(tokenizer: &CoreBPE) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let interactive = stdin.is_terminal();
    let mut lines = stdin.lock().lines();
    if interactive {
        println!("o200k_base · 输入文本后回车分词，输入 /quit 退出。");
    }

    loop {
        if interactive {
            print!("文本> ");
            io::stdout().flush()?;
        }
        let Some(input) = lines.next().transpose()? else {
            break;
        };
        if input == "/quit" {
            break;
        }
        inspect_text(tokenizer, &input)?;
    }

    Ok(())
}

fn inspect_text(tokenizer: &CoreBPE, input: &str) -> Result<(), Box<dyn Error>> {
    let ids = tokenizer.encode_ordinary(input);
    println!("o200k_base · 输入 {input:?}");
    println!(
        "{} 个 Unicode 字符 · {} 个 UTF-8 字节 · {} 个 BPE token",
        input.chars().count(),
        input.len(),
        ids.len()
    );
    println!("token IDs {ids:?}");
    println!("     ID  UTF-8 字节（十六进制）  文本");

    for id in &ids {
        let bytes = tokenizer.decode_bytes(slice::from_ref(id))?;
        let hex = bytes
            .iter()
            .map(|byte| format!("{byte:02X}"))
            .collect::<Vec<_>>()
            .join(" ");
        match str::from_utf8(&bytes) {
            Ok(piece) => println!("{id:>7}  {hex:<24}  {piece:?}"),
            Err(_) => println!("{id:>7}  {hex:<24}  （UTF-8 片段）"),
        }
    }

    let decoded = String::from_utf8(tokenizer.decode_bytes(&ids)?)?;
    if decoded != input {
        return Err("解码结果与原始输入不一致".into());
    }
    println!("无损还原 {decoded:?}");

    Ok(())
}
