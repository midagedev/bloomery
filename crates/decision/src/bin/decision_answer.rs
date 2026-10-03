//! `decision_answer` — Clef's head over hidden states from a file: the SystemOne response body.
//!
//!     decision_answer --model <gguf> --head <safetensors> --request <json> --hidden <f32 file>
//!                     [--head-config <json>] [--max-length N]
//!
//! `--model` gives the tokenizer and the output head's rows (`output`, else `token_embd`), and its
//! file name is the body's `model`, as `bloomery-serve -m` names the model it serves. The head's
//! config is `--head-config`, else `joint_head_config.json` beside `--head`. `--hidden` is raw f32
//! little-endian, row-major `[n, hidden]`, no header (`clef_hidden --out` writes it): a length that is
//! not whole rows, or a row count other than the encoded request's, is refused by name. The body goes
//! to stdout as one line; stderr gets one line with the head's wall (functional, not a timed
//! measurement). A flag given twice takes its last value; an unknown flag or a missing value is
//! refused by name.

use std::path::PathBuf;
use std::time::Instant;

use decision::answer::answer;
use decision::encode::{MAX_LENGTH, encode};
use decision::head::ClefHead;
use decision::render::dumps;
use decision::request::Request;
use decision::rows::{output_head, output_rows};

type BinError = Box<dyn std::error::Error>;

struct Args {
    model: PathBuf,
    head: PathBuf,
    head_config: Option<PathBuf>,
    request: PathBuf,
    hidden: PathBuf,
    max: usize,
}

fn parse() -> Result<Args, BinError> {
    let mut a = std::env::args().skip(1);
    let (mut model, mut head, mut head_config, mut request, mut hidden) =
        (None, None, None, None, None);
    let mut max = MAX_LENGTH;
    while let Some(flag) = a.next() {
        let value = a
            .next()
            .ok_or_else(|| format!("{flag}: a value is due after it"))?;
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--head" => head = Some(PathBuf::from(value)),
            "--head-config" => head_config = Some(PathBuf::from(value)),
            "--request" => request = Some(PathBuf::from(value)),
            "--hidden" => hidden = Some(PathBuf::from(value)),
            "--max-length" => {
                max = value
                    .parse()
                    .map_err(|e| format!("--max-length {value:?}: {e}"))?
            }
            _ => return Err(format!("unknown flag {flag:?}").into()),
        }
    }
    let need = |v: Option<PathBuf>, flag: &str| {
        v.ok_or_else(|| BinError::from(format!("{flag} is required")))
    };
    Ok(Args {
        model: need(model, "--model")?,
        head: need(head, "--head")?,
        head_config,
        request: need(request, "--request")?,
        hidden: need(hidden, "--hidden")?,
        max,
    })
}

fn run() -> Result<(), BinError> {
    let a = parse()?;
    let text =
        std::fs::read_to_string(&a.request).map_err(|e| format!("{}: {e}", a.request.display()))?;
    let req = Request::from_json(&decision::json::parse(&text)?)?;
    let tokenizer = tokenizer::Tokenizer::from_gguf(&a.model)?;
    let enc = encode(&tokenizer, &req, a.max)?;
    let head = ClefHead::open(&a.head, a.head_config.as_deref())?;
    let width = head.config().hidden_size;
    let bytes = std::fs::read(&a.hidden).map_err(|e| format!("{}: {e}", a.hidden.display()))?;
    if bytes.len() % (4 * width) != 0 {
        return Err(format!(
            "{}: {} bytes are not whole rows of {width} f32",
            a.hidden.display(),
            bytes.len()
        )
        .into());
    }
    let rows = bytes.len() / (4 * width);
    if rows != enc.ids.len() {
        return Err(format!(
            "{}: {rows} rows of hidden states for a request of {} ids",
            a.hidden.display(),
            enc.ids.len()
        )
        .into());
    }
    let hidden: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let split = gguf::Split::open(&a.model)?;
    let t0 = Instant::now();
    let mut stages = Vec::new();
    let logits = head.forward_staged(
        &hidden,
        &enc,
        &mut |ids| output_rows(&split, ids).map_err(|e| e.to_string()),
        &mut stages,
    )?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let name = a.model.file_name().map_or_else(
        || a.model.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    println!("{}", dumps(&answer(&req, &enc, &logits, &name)?, false));
    let (rows_name, _, rows_t) = output_head(&split)?;
    eprintln!(
        "decision_answer: head n={} ms={ms:.1} (functional), lexical rows from {rows_name} ({})",
        enc.ids.len(),
        rows_t.ty
    );
    eprintln!(
        "decision_answer: stages {}",
        decision::head::stage_line(&stages)
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("decision_answer: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}
