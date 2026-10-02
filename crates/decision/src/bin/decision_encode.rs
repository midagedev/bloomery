//! `decision_encode` — a SystemOne request to Clef's token ids and spans, the encoder's debugging handle.
//!
//!     decision_encode --tokenizer <gguf> --request <json> [--ids-out <file>] [--max-length N]
//!
//! Prints one JSON line: `input_ids` and, per question, `question_id`, `question_type`,
//! `question_span`, `option_spans` and `option_ids` (the reference rows' names). `--ids-out` also
//! writes the ids one decimal a line, the file `clef_hidden --ids` reads. `--max-length` defaults to
//! the release's 16384. A flag given twice takes its last value; an unknown flag or a missing value is
//! refused by name.

use std::path::PathBuf;

use decision::encode::{Encoded, MAX_LENGTH, encode};
use decision::json::Json;
use decision::render::dumps;
use decision::request::Request;

type BinError = Box<dyn std::error::Error>;

fn int(n: usize) -> Json {
    Json::Int(n.to_string())
}

fn span((a, b): (usize, usize)) -> Json {
    Json::Array(vec![int(a), int(b)])
}

/// The encoding as the reference rows write it.
fn rows(enc: &Encoded) -> Json {
    let questions = enc
        .questions
        .iter()
        .map(|q| {
            Json::Object(vec![
                ("question_id".into(), Json::Str(q.id.clone())),
                ("question_type".into(), int(q.kind.index())),
                ("question_span".into(), span(q.question_span)),
                (
                    "option_spans".into(),
                    Json::Array(q.option_spans.iter().map(|&s| span(s)).collect()),
                ),
                (
                    "option_ids".into(),
                    Json::Array(q.option_ids.iter().map(|o| Json::Str(o.clone())).collect()),
                ),
            ])
        })
        .collect();
    Json::Object(vec![
        (
            "input_ids".into(),
            Json::Array(enc.ids.iter().map(|&i| Json::Int(i.to_string())).collect()),
        ),
        ("questions".into(), Json::Array(questions)),
    ])
}

fn run() -> Result<(), BinError> {
    let mut a = std::env::args().skip(1);
    let (mut tok, mut request, mut ids_out, mut max) = (None, None, None, MAX_LENGTH);
    while let Some(flag) = a.next() {
        let value = a
            .next()
            .ok_or_else(|| format!("{flag}: a value is due after it"))?;
        match flag.as_str() {
            "--tokenizer" => tok = Some(PathBuf::from(value)),
            "--request" => request = Some(PathBuf::from(value)),
            "--ids-out" => ids_out = Some(PathBuf::from(value)),
            "--max-length" => {
                max = value
                    .parse()
                    .map_err(|e| format!("--max-length {value:?}: {e}"))?
            }
            _ => return Err(format!("unknown flag {flag:?}").into()),
        }
    }
    let tok = tok.ok_or("--tokenizer is required")?;
    let request = request.ok_or("--request is required")?;
    let text =
        std::fs::read_to_string(&request).map_err(|e| format!("{}: {e}", request.display()))?;
    let req = Request::from_json(&decision::json::parse(&text)?)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&tok)?;
    let enc = encode(&tokenizer, &req, max)?;
    if let Some(path) = ids_out {
        let lines: String = enc.ids.iter().map(|i| format!("{i}\n")).collect();
        std::fs::write(&path, lines).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    println!("{}", dumps(&rows(&enc), false));
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("decision_encode: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
