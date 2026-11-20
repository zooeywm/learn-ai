use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;
use std::fs;
use std::io;
use std::path::Path;

use step06_modern_llm::{TrainingConfig, TrainingData};

pub const TRAINING: TrainingConfig = TrainingConfig {
    context: 144,
    d_model: 96,
    query_heads: 4,
    kv_heads: 2,
    d_ff: 256,
    layers: 3,
    pretrain_tokens: 32,
    pretrain_steps: 1500,
    pretrain_batch: 8,
    pretrain_lr: 0.0045,
    sft_steps: 2100,
    sft_batch: 8,
    sft_lr: 0.003,
    refine_steps: 1100,
};

pub(crate) struct Poem<'a> {
    pub title: &'a str,
    pub author: &'a str,
    pub body: &'a str,
    first_two: Option<(&'a str, &'a str)>,
}

pub(crate) fn parse_poem(line: &str) -> io::Result<Poem<'_>> {
    let mut columns = line.split('\t');
    let (Some(title), Some(author), Some(body), None) = (
        columns.next(),
        columns.next(),
        columns.next(),
        columns.next(),
    ) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "唐诗语料应有三列",
        ));
    };
    if title.is_empty() || author.is_empty() || body.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "唐诗语料列不能为空",
        ));
    }
    Ok(Poem {
        title,
        author,
        body,
        first_two: first_two_lines(body),
    })
}

pub fn load() -> io::Result<TrainingData> {
    let folder = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/step07");
    let source = fs::read_to_string(folder.join("tang-poems.tsv"))?;
    let mut poems = String::new();
    let mut titles = HashMap::new();
    let mut first_lines = HashMap::new();
    let mut entries = Vec::new();
    for line in source.lines() {
        let poem = parse_poem(line)?;
        let Poem {
            title,
            author,
            body,
            first_two,
        } = poem;
        writeln!(poems, "《{title}》{author}\n{body}").expect("字符串写入不会失败");
        *titles.entry(title).or_insert(0) += 1;
        if let Some((first, _)) = first_two {
            *first_lines.entry(first).or_insert(0) += 1;
        }
        entries.push(poem);
    }

    let mut questions = HashSet::new();
    let mut qa_pairs = load_curated(&folder.join("sft.tsv"), &mut questions)?;

    for &Poem {
        title,
        author,
        body,
        first_two,
    } in &entries
    {
        let unique_title = titles[title] == 1;
        let mut add = |question: String, answer: &str| {
            if questions.insert(question.clone()) {
                qa_pairs.push((question, answer.to_owned()));
            }
        };
        if body.chars().count() <= 64 {
            add(format!("背诵{author}的《{title}》"), body);
            if unique_title {
                add(format!("背一首{title}"), body);
            }
        }
        if unique_title {
            add(format!("《{title}》的作者是谁"), author);
        }
        if let Some((first, next)) = first_two {
            if first_lines[first] == 1 {
                add(format!("{first}的下一句"), next);
            }
        }
    }
    let fragment_examples = load_curated(&folder.join("fragment-qa.tsv"), &mut questions)?;
    let mut paraphrases = fragment_examples
        .iter()
        .cycle()
        .take(4 * fragment_examples.len())
        .cloned()
        .collect::<Vec<_>>();
    paraphrases.extend(build_paraphrases(
        &entries,
        &titles,
        &qa_pairs,
        &mut questions,
    ));
    add_fragment_questions(&entries, &mut paraphrases, &mut questions);
    // 短片段的训练量上升时，重放基础问候、身份说明和完整背诵。
    for (question, answer) in &qa_pairs {
        if matches!(
            question.as_str(),
            "你好" | "你是谁" | "你会什么" | "红豆" | "相思" | "背一首相思"
        ) {
            paraphrases.extend(std::iter::repeat_n((question.clone(), answer.clone()), 12));
        }
    }
    Ok(TrainingData {
        poems,
        qa_pairs,
        paraphrases,
    })
}

fn load_curated(path: &Path, questions: &mut HashSet<String>) -> io::Result<Vec<(String, String)>> {
    let curated = fs::read_to_string(path)?;
    let mut pairs = Vec::new();
    for (index, line) in curated.lines().enumerate() {
        let Some((question, answer)) = line.split_once('\t') else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} 第 {} 行缺少问答分隔符", path.display(), index + 1),
            ));
        };
        if !questions.insert(question.to_owned()) || question.is_empty() || answer.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} 第 {} 行问答为空或重复", path.display(), index + 1),
            ));
        }
        pairs.push((question.to_owned(), answer.to_owned()));
    }
    Ok(pairs)
}

fn add_fragment_questions(
    poems: &[Poem<'_>],
    pairs: &mut Vec<(String, String)>,
    questions: &mut HashSet<String>,
) {
    // 同一条摘句若属于不同作品，不拿其中一首当确定答案。
    let mut excerpts = BTreeMap::new();
    for (index, poem) in poems.iter().enumerate() {
        if poem.body.chars().count() > 42 {
            continue;
        }
        for sentence in poem.body.split_inclusive(['。', '？', '！']) {
            for clause in sentence.split(['，', '。', '？', '！']) {
                let length = clause.chars().count();
                if !matches!(length, 5 | 7) {
                    continue;
                }
                let snippet: String = if index % 2 == 0 {
                    clause.chars().take(3).collect()
                } else {
                    clause.chars().skip(length - 3).collect()
                };
                for fragment in [clause.to_owned(), snippet] {
                    excerpts.entry(fragment).or_insert((index, sentence));
                }
            }
        }
    }
    for (fragment, (index, sentence)) in excerpts {
        if !questions.insert(fragment.clone()) {
            continue;
        }
        let ambiguous = poems
            .iter()
            .enumerate()
            .any(|(other, poem)| other != index && poem.body.contains(fragment.as_str()));
        let poem = &poems[index];
        let answer = if ambiguous {
            "这段诗句有多个出处。请再给一句诗，或补充诗名和作者。".to_owned()
        } else {
            format!("{}《{}》：{sentence}", poem.author, poem.title)
        };
        pairs.push((fragment, answer));
    }
}

fn build_paraphrases(
    poems: &[Poem<'_>],
    titles: &HashMap<&str, usize>,
    qa_pairs: &[(String, String)],
    questions: &mut HashSet<String>,
) -> Vec<(String, String)> {
    let mut paraphrases = Vec::new();
    let mut add = |question: String, answer: &str| {
        if questions.insert(question.clone()) {
            paraphrases.push((question, answer.to_owned()));
        }
    };
    for (
        index,
        &Poem {
            title,
            author,
            body,
            ..
        },
    ) in poems.iter().enumerate()
    {
        let unique = titles[title] == 1;
        if body.chars().count() <= 64 {
            let prompt = match index % 4 {
                0 => format!("请念一遍{author}的《{title}》"),
                1 => format!("来背一下{author}的{title}"),
                2 => format!("能把{author}的《{title}》背全吗"),
                _ => format!("给我读读{author}写的{title}"),
            };
            add(prompt, body);
            if unique {
                add(format!("请背出《{title}》全文"), body);
            }
        }
        if unique {
            let prompt = match index % 3 {
                0 => format!("谁写的《{title}》"),
                1 => format!("《{title}》出自哪位诗人"),
                _ => format!("《{title}》是哪位诗人写的"),
            };
            add(prompt, author);
        }
    }
    for (index, (question, answer)) in qa_pairs.iter().enumerate() {
        if let Some(line) = question.strip_suffix("的下一句") {
            let prompt = match index % 3 {
                0 => format!("{line}后面怎么接"),
                1 => format!("{line}的后一句是什么"),
                _ => format!("帮我续上{line}的下一句"),
            };
            add(prompt, answer);
            add(format!("「{line}」后接哪句"), answer);
        }
    }

    add(
        "把李白的静夜思从头到尾背出来".into(),
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    );
    add(
        "读一读王维写红豆的那首诗".into(),
        "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    );

    paraphrases
}

fn first_two_lines(body: &str) -> Option<(&str, &str)> {
    let pos = body.find('，')?;
    let first = &body[..pos];
    if !matches!(first.chars().count(), 5 | 7) {
        return None;
    }
    let rest = &body[pos + '，'.len_utf8()..];
    let next = rest.split_inclusive(['，', '。', '？', '！']).next()?;
    Some((first, next))
}
