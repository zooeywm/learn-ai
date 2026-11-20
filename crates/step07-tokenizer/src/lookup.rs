use std::io;

use crate::poetry::{Poem, parse_poem};

const CORPUS: &str = include_str!("../../../data/step07/tang-poems.tsv");
const SOURCE_SUFFIXES: &[&str] = &[
    "是哪个诗里的",
    "是哪首诗里的",
    "是哪一首诗里的",
    "是哪首诗的",
    "是哪首诗",
    "是哪一首诗",
    "出自哪首诗",
    "出自哪里",
];
const SOURCE_PREFIXES: &[&str] = &["哪首诗里有", "哪一首诗里有"];

pub(crate) fn answer(input: &str) -> io::Result<Option<String>> {
    let input = trim_edges(input);
    let source_phrase = SOURCE_SUFFIXES
        .iter()
        .find_map(|suffix| input.strip_suffix(suffix))
        .or_else(|| {
            SOURCE_PREFIXES
                .iter()
                .find_map(|prefix| input.strip_prefix(prefix))
        });
    let phrase = trim_edges(source_phrase.unwrap_or(input));
    let is_source_question = source_phrase.is_some();
    let phrase_len = phrase.chars().take(33).count();
    if !is_source_question && !(2..=32).contains(&phrase_len) {
        return Ok(None);
    }
    if phrase.is_empty() {
        return Ok(Some("请提供要查出处的诗句。".to_owned()));
    }

    let mut titles = Vec::new();
    let mut passages = Vec::new();
    let mut suggestions = Vec::new();
    let shorter = if !is_source_question
        && (3..=7).contains(&phrase_len)
        && phrase
            .chars()
            .all(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
    {
        phrase.char_indices().nth(1).map(|(at, _)| &phrase[at..])
    } else {
        None
    };
    for line in CORPUS.lines() {
        let poem = parse_poem(line)?;
        if poem.title == phrase {
            titles.push(poem);
        } else if poem.body.contains(phrase) {
            passages.push(poem);
        } else if shorter.is_some_and(|short| poem.body.contains(short)) {
            suggestions.push(poem);
        }
    }

    let matched_title = !titles.is_empty();
    let matches = if matched_title { &titles } else { &passages };
    let response = match matches.as_slice() {
        [] if is_source_question => Some(format!(
            "本地唐诗库未找到「{phrase}」。请补充诗名或核对原句。"
        )),
        [] => match (shorter, suggestions.as_slice()) {
            (Some(short), [poem]) => Some(format!(
                "你是想找「{short}」吗？\n{}\n{}",
                attribution(poem),
                excerpt(poem, short)
            )),
            _ => None,
        },
        [poem] => {
            let show_full_poem =
                matched_title || (!is_source_question && poem.body.chars().count() <= 64);
            let excerpt = if show_full_poem {
                poem.body
            } else {
                excerpt(poem, phrase)
            };
            Some(format!("{}\n{excerpt}", attribution(poem)))
        }
        [..] => {
            let candidates = matches
                .iter()
                .take(3)
                .map(attribution)
                .collect::<Vec<_>>()
                .join("、");
            Some(format!(
                "「{phrase}」在 {} 首诗中有匹配，例如 {candidates}。可以用诗名或更长的原句缩小范围。",
                matches.len()
            ))
        }
    };
    Ok(response)
}

fn trim_edges(text: &str) -> &str {
    text.trim_matches(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '「' | '」'
                    | '『'
                    | '』'
                    | '《'
                    | '》'
                    | '“'
                    | '”'
                    | '"'
                    | '？'
                    | '?'
                    | '。'
                    | '！'
                    | '!'
            )
    })
}

fn attribution(poem: &Poem<'_>) -> String {
    format!("{}《{}》", poem.author, poem.title)
}

fn excerpt<'a>(poem: &Poem<'a>, phrase: &str) -> &'a str {
    poem.body
        .split_inclusive(['。', '？', '！'])
        .find(|line| line.contains(phrase))
        .unwrap_or(poem.body)
}

#[cfg(test)]
mod tests {
    use super::answer;

    #[test]
    fn finds_quoted_fragments_and_their_source() {
        for fragment in ["思故乡", "低头思故乡"] {
            let found = answer(fragment).unwrap().unwrap();
            assert!(found.starts_with("李白《静夜思》\n"));
            assert!(found.contains("举头望明月，低头思故乡。"));
        }
        for question in [
            "低头思故乡是哪个诗里的",
            "「低头思故乡」出自哪首诗？",
            "哪首诗里有「低头思故乡」",
        ] {
            assert_eq!(
                answer(question).unwrap().as_deref(),
                Some("李白《静夜思》\n举头望明月，低头思故乡。")
            );
        }
        assert!(answer("红豆").unwrap().unwrap().contains("此物最相思。"));
    }

    #[test]
    fn reports_ambiguous_and_missing_sources() {
        let ambiguous = answer("春风").unwrap().unwrap();
        assert!(ambiguous.contains("首诗中有匹配"));
        assert!(
            answer("低头望火星出自哪首诗")
                .unwrap()
                .unwrap()
                .contains("本地唐诗库未找到")
        );
        assert_eq!(answer("你是谁").unwrap(), None);
    }

    #[test]
    fn suggests_a_unique_match_when_a_character_was_added_at_the_start() {
        assert_eq!(
            answer("巨举头").unwrap().as_deref(),
            Some("你是想找「举头」吗？\n李白《静夜思》\n举头望明月，低头思故乡。")
        );
        assert_eq!(answer("巨春风").unwrap(), None);
    }
}
