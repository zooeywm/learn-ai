use std::path::PathBuf;

use step06_modern_llm::{CharTokenizer, TrainingConfig, TrainingData, run_cli};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints/step06-modern.mllm");
    run_cli::<CharTokenizer>(
        std::env::args().skip(1),
        path,
        TrainingConfig::default(),
        TrainingData::lesson_data,
        None,
    )
}
