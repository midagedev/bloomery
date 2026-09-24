//! `bloomery-serve-ds41` — the llama-server-compatible HTTP API on the V4.1 engine.
//!
//!     bloomery-serve-ds41 [--host 127.0.0.1] [--port 8080] [--place a|gate]
//!                         [--ctx C] [--alias NAME]
//!
//! The model is `$BLOOMERY_REF_MODEL`; its first shard gives the vocabulary,
//! the chat template (`tokenizer.chat_template`) and the default alias
//! (`general.name`). The plan line, then the `load`, host and `capture`
//! lines of [`Generator::open`] go to stderr as `generate_ds41` prints them,
//! then `listening on http://<addr>` once the model is loaded and the port is
//! bound (`--port 0` binds a free one). Sampling is the sampler crate's chain
//! with no repetition penalty; `temperature <= 0` is the engine's argmax, the
//! ids `generate_ds41 --tokens <the prompt's ids>` prints.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "bloomery-serve-ds41: built without the `deepseek41` feature; see `just gate-gpu-ds41-serve`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    match drive::run() {
        // EX_SOFTWARE: the engine, not the listener or the load, ended the run.
        Ok(serve::ServeError::Engine(f)) => {
            eprintln!("bloomery-serve-ds41: {f}");
            std::process::ExitCode::from(70)
        }
        Ok(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e.into())),
        Err(e) => bloomery_gpu_gates::exit_with("bloomery-serve-ds41", Err(e)),
    }
}

#[cfg(feature = "deepseek41")]
mod drive {
    use std::sync::Arc;

    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{self, Deepseek41Model};
    use bloomery_gpu_gates::bind::{Ds41Engine, Vocab, sampler_factory};
    use bloomery_gpu_gates::generate::{Generator, OpenArgs, Place};
    use bloomery_gpu_gates::{GateError, ref_model_path};
    use gguf::Split;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::workstation;
    use serve::{FATAL_LINGER, ServeError, Server, ServerConfig};
    use tokenizer::Tokenizer;

    const USAGE: &str = "usage: bloomery-serve-ds41 [--host H] [--port P] [--place a|gate] \
                         [--ctx C] [--alias NAME]";

    struct Args {
        host: String,
        port: u16,
        place: Place,
        ctx: usize,
        alias: Option<String>,
    }

    fn parse_args() -> Result<Args, GateError> {
        let mut a = Args {
            host: "127.0.0.1".to_owned(),
            port: 8080,
            place: Place::A,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            alias: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--help" || flag == "-h" {
                return Err(USAGE.into());
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
            match flag.as_str() {
                "--host" => a.host = v,
                "--port" => a.port = v.parse()?,
                "--place" => a.place = Place::parse(&v)?,
                "--ctx" => a.ctx = v.parse()?,
                "--alias" => a.alias = Some(v),
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        Ok(a)
    }

    /// Loads the model and serves until the listener or the engine fails;
    /// `Ok` carries why the server ended.
    pub fn run() -> Result<ServeError, GateError> {
        let a = parse_args()?;
        let path = ref_model_path()?;
        let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
        let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let template = inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned();
        let name = inv
            .value("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("deepseek-v4.1")
            .to_owned();
        drop(inv);

        let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::read(&split)?;
        drop(split);
        let card = print_plan(&inputs, a.place, a.ctx)?;
        let want_top_k = inputs.hp.indexer.top_k;
        let n_layer = inputs.hp.n_layer;
        let pin_main = !std::env::var("BLOOMERY_PIN_MAIN").is_ok_and(|v| v == "0");
        let open = OpenArgs {
            place: a.place,
            ctx: a.ctx,
            mode: StepMode::Graph,
            pin_main,
        };
        let engine = Ds41Engine::spawn(
            move || {
                Generator::open(
                    open,
                    body::open,
                    |m: &Deepseek41Model| {
                        let top_k = m.body("bloomery-serve-ds41")?.indexer_top_k();
                        if top_k != want_top_k {
                            return Err(format!(
                                "the body selects {top_k} rows per stream, the file's top_k is \
                                 {want_top_k}: a step past that many visible rows would not be \
                                 the model's"
                            )
                            .into());
                        }
                        Ok(format!("layers={n_layer} top_k={top_k}"))
                    },
                    &mut std::io::stderr(),
                )
            },
            vocab,
            card,
        )?;

        let config = ServerConfig {
            model_alias: a.alias.unwrap_or(name),
            model_path: path.display().to_string(),
            chat_template: template,
            sampler: Some(sampler_factory()),
            fatal_linger: FATAL_LINGER,
        };
        let server = Server::bind((a.host.as_str(), a.port), Box::new(engine), config)?;
        eprintln!(
            "bloomery-serve-ds41: place={} ctx={} listening on http://{}",
            a.place.name(),
            a.ctx,
            server.local_addr()?
        );
        Ok(server.run())
    }

    /// The plan the engine is about to load, on stderr; returns the card's name.
    fn print_plan(inputs: &PlanInputs, place: Place, ctx: usize) -> Result<String, GateError> {
        let machine = place.machine()(inputs.model.layers);
        let plan = inputs.plan(&machine, u64::try_from(ctx)?)?;
        let held: Vec<u64> = plan.n_l.iter().copied().filter(|&n| n > 0).collect();
        let card = &plan.cards[0];
        eprintln!(
            "plan place={} card={} ctx_max={} card_experts={} ({} B) host_experts={} ({} B) \
             n_l={}..{} on {} layers card_budget={}",
            place.name(),
            machine.cards[0].name,
            plan.ctx_max,
            card.experts,
            card.expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes,
            held.iter().min().copied().unwrap_or(0),
            held.iter().max().copied().unwrap_or(0),
            held.len(),
            plan.card_budget
                .map_or_else(|| "none".to_string(), |b| b.to_string())
        );
        Ok(machine.cards[0].name.to_string())
    }
}
