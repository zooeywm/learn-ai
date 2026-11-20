mod lookup;
#[cfg(feature = "openai")]
mod openai;
mod poetry;

use std::{error::Error, ffi::OsString, path::PathBuf};

use step06_modern_llm::run_cli;
use step07_tokenizer::ByteBpe;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args_os()
        .skip(1)
        .map(OsString::into_string)
        .collect::<Result<_, _>>()
        .map_err(|_| "输入必须是有效的 UTF-8")?;
    #[cfg(feature = "openai")]
    if args.first().is_some_and(|arg| arg == "inspect") {
        return openai::run(args.into_iter().skip(1));
    }

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/step07-bpe.mllm");
    run_cli::<ByteBpe>(
        args,
        path,
        poetry::TRAINING,
        poetry::load,
        Some(lookup::answer),
    )
}
