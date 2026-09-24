//! `bloomery-serve`: the HTTP server on the mock engine.
//!
//!   bloomery-serve [--host 127.0.0.1] [--port 8080] [--model <first.gguf>]
//!                  [--chat-template-file <path>] [--alias <name>] [--ctx-size <n>]
//!                  [--print-template] [--mock-fail-at <k>] [--slot-save-path <dir>]
//!
//! `--model` reads `tokenizer.chat_template` and `general.name` from the GGUF
//! header only; the engine is always the mock here. The binding to the GPU
//! model is a separate, feature-gated binary (`bloomery-serve-ds41`).
//!
//! `--mock-fail-at k` makes the mock's `k`-th `next` an engine error: the crash
//! path's gate. The process then prints the crash block and exits 70.
//!
//! `--slot-save-path dir` enables `POST /slots/0?action=save|restore` on files
//! in `dir`, which must exist (the process exits 64 otherwise).

use std::path::PathBuf;
use std::process::ExitCode;

use serve::{FATAL_LINGER, MockEngine, ServeError, Server, ServerConfig};

/// Used when neither `--model` nor `--chat-template-file` gives one.
const FALLBACK_TEMPLATE: &str = "{{- bos_token -}}{%- for m in messages -%}<|{{ m['role'] }}|>{{ m['content'] or '' }}\n{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}";

struct Args {
    host: String,
    port: u16,
    model: Option<String>,
    template_file: Option<String>,
    alias: Option<String>,
    ctx: usize,
    print_template: bool,
    fail_at: Option<usize>,
    slot_save_path: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        model: None,
        template_file: None,
        alias: None,
        ctx: 4096,
        print_template: false,
        fail_at: None,
        slot_save_path: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--host" => a.host = val()?,
            "--port" => a.port = val()?.parse().map_err(|e| format!("--port: {e}"))?,
            "--model" | "-m" => a.model = Some(val()?),
            "--chat-template-file" => a.template_file = Some(val()?),
            "--alias" => a.alias = Some(val()?),
            "--ctx-size" | "-c" => {
                a.ctx = val()?.parse().map_err(|e| format!("--ctx-size: {e}"))?
            }
            "--print-template" => a.print_template = true,
            "--mock-fail-at" => {
                a.fail_at = Some(val()?.parse().map_err(|e| format!("--mock-fail-at: {e}"))?)
            }
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(val()?)),
            "--help" | "-h" => {
                return Err("usage: bloomery-serve [--host H] [--port P] [--model GGUF] \
                            [--chat-template-file PATH] [--alias NAME] [--ctx-size N] [--print-template] \
                            [--mock-fail-at K] [--slot-save-path DIR]"
                    .to_owned());
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(64);
        }
    };
    let mut template = FALLBACK_TEMPLATE.to_owned();
    let mut alias = "mock".to_owned();
    if let Some(path) = &args.model {
        let inv = match gguf::inventory_of(path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{path}: {e}");
                return ExitCode::from(66);
            }
        };
        match inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
        {
            Some(t) => template = t.to_owned(),
            None => eprintln!("{path}: no tokenizer.chat_template; using the fallback template"),
        }
        if let Some(n) = inv.value("general.name").and_then(|v| v.as_str()) {
            alias = n.to_owned();
        }
    }
    if let Some(path) = &args.template_file {
        match std::fs::read_to_string(path) {
            Ok(t) => template = t,
            Err(e) => {
                eprintln!("{path}: {e}");
                return ExitCode::from(66);
            }
        }
    }
    if args.print_template {
        print!("{template}");
        return ExitCode::SUCCESS;
    }
    let config = ServerConfig {
        model_alias: args.alias.unwrap_or(alias),
        model_path: args.model.clone().unwrap_or_else(|| "mock".to_owned()),
        chat_template: template,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: args.slot_save_path,
    };
    let engine = match args.fail_at {
        Some(k) => MockEngine::failing_at(args.ctx, k),
        None => MockEngine::new(args.ctx),
    };
    let server = match Server::bind((args.host.as_str(), args.port), Box::new(engine), config) {
        Ok(s) => s,
        Err(e @ ServeError::SlotSavePath(_)) => {
            eprintln!("{e}");
            return ExitCode::from(64);
        }
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(1);
        }
    };
    match server.local_addr() {
        Ok(addr) => eprintln!("bloomery-serve: mock engine, listening on http://{addr}"),
        Err(e) => eprintln!("bloomery-serve: {e}"),
    }
    let e = server.run();
    eprintln!("bloomery-serve: {e}");
    match e {
        // EX_SOFTWARE: the engine, not the listener, ended the run.
        ServeError::Engine(_) => ExitCode::from(70),
        ServeError::Io(_) | ServeError::Template(_) | ServeError::SlotSavePath(_) => {
            ExitCode::from(1)
        }
    }
}
