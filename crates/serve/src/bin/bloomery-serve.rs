//! `bloomery-serve`: the HTTP server on the mock engine.
//!
//!   bloomery-serve [--host 127.0.0.1] [--port 8080] [--model <first.gguf>]
//!                  [--chat-template-file <path>] [--alias <name>] [--ctx-size <n>]
//!                  [--print-template] [--mock-fail-at <k>] [--mock-load-ms <ms>]
//!                  [--slot-save-path <dir>] [--parallel <n>] [--queue-depth <q>]
//!
//! `--model` reads `tokenizer.chat_template` and `general.name` from the GGUF
//! header only; the engine is always the mock here. The binding to the GPU
//! model is a separate, feature-gated binary (`bloomery-serve-ds41`).
//!
//! `--mock-fail-at k` makes the mock's `k`-th `next` an engine error: the crash
//! path's gate. The process then prints the crash block and exits 70.
//!
//! `--mock-load-ms ms` makes the mock's load take `ms` milliseconds: the listener is bound
//! first ([`serve::Listener::early`]) and answers 503 `loading` until the load ends, as the
//! real seats' does while they fetch and load.
//!
//! The start's events are JSON lines on stdout ([`serve::event`]): `bound` when the listener
//! is up, `ready` when requests are served, `error` before a non-zero exit, `stopped` when
//! the server ends. The exit statuses are [`serve::exit::Code`]'s; `--help` prints the usage
//! on stdout and exits 0.
//!
//! `--slot-save-path dir` enables `POST /slots/{id}?action=save|restore` on
//! files in `dir`, which must exist (the process exits 64 otherwise).
//!
//! `--parallel n` (`-np n`) serves `n` slots, each with the whole `--ctx-size`,
//! on a mock of `n` slots; `--queue-depth q` lets at most `q` requests wait past
//! the free slots (`serve::SlotConfig`). A value the server refuses exits 64.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use serve::event::{self, Event};
use serve::exit::{self, Code};
use serve::flag::ApiKeys;
use serve::listener::Phase;
use serve::{
    FATAL_LINGER, Listener, MAX_CONNECTIONS, MockEngine, ServeError, Server, ServerConfig,
    SlotConfig,
};

const USAGE: &str = "usage: bloomery-serve [--host H] [--port P] [--model GGUF] \
                     [--chat-template-file PATH] [--alias NAME] [--ctx-size N] [--print-template] \
                     [--mock-fail-at K] [--mock-load-ms MS] [--slot-save-path DIR] [--parallel N] \
                     [--queue-depth Q] [--api-key KEY] [--api-key-file FNAME]";

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
    load_ms: u64,
    slot_save_path: Option<PathBuf>,
    parallel: usize,
    queue_depth: Option<usize>,
    api_keys: ApiKeys,
}

/// The arguments, or `None` when the usage was asked for.
fn parse_args() -> Result<Option<Args>, String> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        model: None,
        template_file: None,
        alias: None,
        ctx: 4096,
        print_template: false,
        fail_at: None,
        load_ms: 0,
        slot_save_path: None,
        parallel: 1,
        queue_depth: None,
        api_keys: ApiKeys::default(),
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
            "--mock-load-ms" => {
                a.load_ms = val()?.parse().map_err(|e| format!("--mock-load-ms: {e}"))?
            }
            "--slot-save-path" => a.slot_save_path = Some(PathBuf::from(val()?)),
            "--parallel" | "-np" => {
                a.parallel = val()?.parse().map_err(|e| format!("--parallel: {e}"))?
            }
            "--queue-depth" => {
                a.queue_depth = Some(val()?.parse().map_err(|e| format!("--queue-depth: {e}"))?)
            }
            f if serve::flag::KEYS.contains(&f) => a.api_keys.add(f, &val()?)?,
            "--help" | "-h" => return Ok(None),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(Some(a))
}

/// Ends the start that `e` refused: its stderr line, its `error` event, its status.
fn refused(e: &ServeError) -> ExitCode {
    eprintln!("{e}");
    let code = Code::of(e);
    let next: &[&[&str]] = if code == Code::PortInUse {
        &[&["--port", "0"]]
    } else if code == Code::Usage {
        &[&["--help"]]
    } else {
        &[]
    };
    exit::refused(code, &e.to_string(), next)
}

fn main() -> ExitCode {
    event::enable();
    let args = match parse_args() {
        Ok(Some(a)) => a,
        Ok(None) => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(e) => {
            eprintln!("{e}");
            return exit::refused(Code::Usage, &e, &[&["--help"]]);
        }
    };
    let mut template = FALLBACK_TEMPLATE.to_owned();
    let mut alias = "mock".to_owned();
    if let Some(path) = &args.model {
        let inv = match gguf::inventory_of(path) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{path}: {e}");
                return exit::refused(Code::NoModel, &format!("{path}: {e}"), &[]);
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
                return exit::refused(Code::NoModel, &format!("{path}: {e}"), &[]);
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
        chat_preparation: None,
        sampler: None,
        fatal_linger: FATAL_LINGER,
        slot_save_path: args.slot_save_path,
        api_keys: args.api_keys,
    };
    // The listener comes first, so a port that is taken is found before the load and a client sees
    // the start at once.
    let listener = match Listener::early((args.host.as_str(), args.port)) {
        Ok(l) => l,
        Err(e) => return refused(&ServeError::Io(e)),
    };
    listener.progress().phase(Phase::Load);
    event::emit(&Event::Bound {
        url: &event::url(listener.local_addr()),
    });
    std::thread::sleep(Duration::from_millis(args.load_ms));
    let engine = match args.fail_at {
        Some(k) => MockEngine::failing_at(args.ctx, k),
        None => MockEngine::new(args.ctx),
    };
    // --parallel 0, or past the connection limit, is the server's refusal,
    // which it makes before it asks the engine's slots.
    let mock_slots = if (1..=MAX_CONNECTIONS).contains(&args.parallel) {
        args.parallel
    } else {
        1
    };
    let engine = engine.with_slots(mock_slots);
    let slots = SlotConfig {
        parallel: args.parallel,
        queue_depth: args.queue_depth,
        ..SlotConfig::default()
    };
    let server = match Server::bind_listener(listener, Box::new(engine), config, slots) {
        Ok(s) => s,
        Err(e) => return refused(&e),
    };
    match server.local_addr() {
        Ok(addr) => eprintln!("bloomery-serve: mock engine, listening on http://{addr}"),
        Err(e) => eprintln!("bloomery-serve: {e}"),
    }
    let e = server.run();
    eprintln!("bloomery-serve: {e}");
    let code = Code::of(&e);
    exit::refused(code, &e.to_string(), &[])
}
