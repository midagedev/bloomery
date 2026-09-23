//! `bloomery-tokenize` — the reference's `llama-tokenize`, same flags, same
//! output, so `diff` between the two is the manual check.
//!
//!     bloomery-tokenize -m <gguf> (-p <text> | -f <file> | --stdin)
//!                       [--ids] [--no-bos] [--no-parse-special] [--show-count]
//!     bloomery-tokenize -m <gguf> --info
//!
//! `--info` prints every vocabulary metadata key the loader read, and the
//! derived vocabulary facts, instead of tokenizing. `--log-disable` is
//! accepted and does nothing (this binary logs nothing).

use std::io::{Read, Write};
use std::process::ExitCode;

use tokenizer::{Tokenizer, attr};

struct Args {
    model: String,
    prompt: Option<Vec<u8>>,
    ids: bool,
    no_bos: bool,
    no_parse_special: bool,
    show_count: bool,
    info: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: String::new(),
        prompt: None,
        ids: false,
        no_bos: false,
        no_parse_special: false,
        show_count: false,
        info: false,
    };
    let mut sources = 0;
    let mut it = std::env::args_os().skip(1);
    while let Some(arg) = it.next() {
        let arg = arg.to_string_lossy().into_owned();
        let mut value = || {
            it.next()
                .ok_or_else(|| format!("{arg} requires an argument"))
        };
        match arg.as_str() {
            "-m" | "--model" => a.model = value()?.to_string_lossy().into_owned(),
            "-p" | "--prompt" => {
                sources += 1;
                a.prompt = Some(os_bytes(value()?));
            }
            "-f" | "--file" => {
                sources += 1;
                let path = value()?;
                let bytes =
                    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.to_string_lossy()))?;
                a.prompt = Some(bytes);
            }
            "--stdin" => {
                sources += 1;
                let mut bytes = Vec::new();
                std::io::stdin()
                    .read_to_end(&mut bytes)
                    .map_err(|e| format!("stdin: {e}"))?;
                a.prompt = Some(bytes);
            }
            "--ids" => a.ids = true,
            "--no-bos" => a.no_bos = true,
            "--no-parse-special" => a.no_parse_special = true,
            "--show-count" => a.show_count = true,
            "--log-disable" => {}
            "--info" => a.info = true,
            _ => return Err(format!("unknown option '{arg}'")),
        }
    }
    if a.model.is_empty() {
        return Err("must specify --model".into());
    }
    if sources > 1 {
        return Err("--stdin, --file and --prompt are mutually exclusive".into());
    }
    if sources == 0 && !a.info {
        return Err("must specify one of: --stdin, --file or --prompt".into());
    }
    Ok(a)
}

#[cfg(unix)]
fn os_bytes(s: std::ffi::OsString) -> Vec<u8> {
    use std::os::unix::ffi::OsStringExt;
    s.into_vec()
}

#[cfg(not(unix))]
fn os_bytes(s: std::ffi::OsString) -> Vec<u8> {
    s.to_string_lossy().into_owned().into_bytes()
}

fn info(tok: &Tokenizer, out: &mut impl Write) -> std::io::Result<()> {
    for (k, v) in tok.metadata_read() {
        writeln!(out, "key {k} = {v}")?;
    }
    let mut hist = std::collections::BTreeMap::<u32, usize>::new();
    for id in 0..tok.n_vocab() as u32 {
        *hist.entry(tok.attr(id).unwrap_or(0)).or_default() += 1;
    }
    for (a, n) in hist {
        writeln!(out, "attr {a:#06x} tokens {n}")?;
    }
    let user: Vec<(u32, &str)> = (0..tok.n_vocab() as u32)
        .filter(|&id| tok.attr(id).is_some_and(|a| a & attr::USER_DEFINED != 0))
        .map(|id| (id, tok.text(id).unwrap_or_default()))
        .collect();
    writeln!(out, "user_defined {user:?}")?;
    let s = tok.specials();
    writeln!(out, "specials {s:?}")?;
    writeln!(out, "add_bos {} add_eos {}", tok.add_bos(), tok.add_eos())?;
    writeln!(out, "eog {:?}", tok.eog())?;
    writeln!(out, "special_tokens {}", tok.special_tokens().len())?;
    writeln!(out, "special_overlap {:?}", tok.special_overlap())?;
    writeln!(out, "ambiguous_roles {:?}", tok.ambiguous_roles())?;
    writeln!(out, "merges_off_vocab {}", tok.merges_off_vocab())?;
    let unk_byte = (0..tok.n_vocab() as u32)
        .filter(|&id| tok.attr(id).is_some_and(|a| a & attr::NORMAL != 0))
        .filter(|&id| {
            tok.piece(id, true)
                .windows(12)
                .any(|w| w == b"[UNK_BYTE_0x")
        })
        .count();
    writeln!(out, "normal_tokens_outside_byte_alphabet {unk_byte}")
}

fn main() -> ExitCode {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let tok = match Tokenizer::from_gguf(&args.model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "Error: could not load the vocabulary from '{}': {e}",
                args.model
            );
            return ExitCode::FAILURE;
        }
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let written = if args.info {
        info(&tok, &mut out)
    } else {
        let prompt = args.prompt.as_deref().unwrap_or_default();
        // As `llama-tokenize`: one flag decides BOS and EOS together.
        let add_special = tok.add_bos() && !args.no_bos;
        let ids = tok.encode(prompt, add_special, !args.no_parse_special);
        print_ids(&tok, &ids, &args, &mut out)
    };
    match written.and_then(|()| out.flush()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn print_ids(
    tok: &Tokenizer,
    ids: &[u32],
    args: &Args,
    out: &mut impl Write,
) -> std::io::Result<()> {
    if args.ids {
        let list: Vec<String> = ids.iter().map(u32::to_string).collect();
        writeln!(out, "[{}]", list.join(", "))?;
    } else {
        for &id in ids {
            // `printf("%s")` of the piece: bytes up to the first NUL.
            let piece = tok.piece(id, true);
            let piece = piece.split(|&b| b == 0).next().unwrap_or_default();
            write!(out, "{id:6} -> '")?;
            out.write_all(piece)?;
            writeln!(out, "'")?;
        }
    }
    if args.show_count {
        writeln!(out, "Total number of tokens: {}", ids.len())?;
    }
    Ok(())
}
