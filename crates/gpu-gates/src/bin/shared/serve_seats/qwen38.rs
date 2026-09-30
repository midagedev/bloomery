//! `bloomery-serve-qwen38` — the llama-server-compatible HTTP API on the
//! Qwen3.8-Flash-Next (qwen4exp) engine, for a client to attach to.
//!
//! This module is the seat: `bloomery-serve-qwen38` and `bloomery-serve --model
//! qwen38` are each one call of [`run`], which takes the process's arguments
//! (`--model` already taken out by the one-binary server).
//!
//!     bloomery-serve-qwen38 [--host 127.0.0.1] [--port 8080] [--place a|gate]
//!                           [--ctx-size C] [--alias NAME]
//!                           [--chat-template-file PATH]
//!
//! `--ctx` is `--ctx-size` under its family's other spelling. The model is
//! `$BLOOMERY_REF_MODEL` (a `--model` flag is not taken: the family's
//! binaries read the one the profile pins); its first shard gives the
//! vocabulary, `tokenizer.chat_template` the chat template
//! (`--chat-template-file` replaces it) and `general.name` the default alias.
//! The `plan` record, then the `load` and `capture` lines of the qwen4exp
//! open, go to stderr as `generate_qwen3moe` prints them, then `listening on
//! http://<addr>` once the model is loaded and the port is bound (`--port 0`
//! binds a free one). Sampling is the sampler crate's chain with no
//! repetition penalty; `temperature <= 0` is the engine's argmax, the ids
//! `generate_qwen3moe --tokens <the prompt's ids>` prints for a prompt of at
//! most eight ids (past that the ubatch walk the server and the CLI both take
//! leaves a position's bits a function of its own inputs, so a longer
//! prompt's greedy ids are the walk's, not a step-fed run's).
//!
//! The positions the server serves are the stores the load sized
//! (`--ctx-size`, default 4096): `/props`' `n_ctx` is that number, a prompt
//! that long is a 400 before it reaches the engine, and generation stops
//! there with `truncated`.
//!
//! A request keeps no prefix of another's: the recurrent layers (the GDN
//! state lanes, the PLE hash history) keep no state for an earlier position,
//! so the seat's keep rule grants nothing and every request prefills from a
//! reset — the same ids as a fresh one, never a silently wrong state. The
//! server's host prompt cache is therefore off for this engine (its budget is
//! 0) and slot save/restore answers the server's own 501; every prefix a
//! request shares with what the slot holds is a `cache reuse` record with the
//! rule. Where the body does take positions back — a verify's commit — no
//! request path reaches.
//!
//! The seat an MTP draft would extend is `Q38` below: its `pass` and
//! `pass_rows` are the engine's defaults (no draft), which is where a draft
//! plugs in.
//!
//! Under `BLOOMERY_DRAFT=mtp` the seat drives the session through the
//! runtime's speculative loop with the shared window `app::mtp::MtpDraft` (the
//! shared draft file beside the target, its head reduced under
//! `BLOOMERY_MTP_HEAD_ROWS`): windows of four rows, the greedy ids the
//! plain server's, `pass_rows` 4 and a sampling or id-banning request
//! refused while a draft runs (the engine's own rule, a 400 naming the
//! field). A `load draft=mtp` line follows the `load` line, and
//! `/props`' `engine.draft` names the draft with its resident bytes as the
//! card's `draft` class. Every other word of the lever is refused by name.
//!
//! An engine error ends the process: the request gets a 500, `/health` a 503
//! for a moment, then the crash block (card, position, error) goes to stderr
//! and the exit code is 70.
//!
//! The binary builds under the `deepseek41` feature, whose name is the
//! server surface's scoping (gpu-gates' `bind` and `serve_client`, the
//! `serve` and `sampler` crates), not a model: it runs no V4.1 code.
//!
//! The levers it acts on ([`ACTS_ON`]) are parsed once, at `main`
//! (`bloomery_levers::at_main`), which refuses by name a lever set outside
//! them and a `BLOOMERY_*` name no registry row names; `--levers` prints them
//! with this process's values and exits. The Qwen3.8 levers are
//! `BLOOMERY_QWEN38_EXPERTS` (the plan's expert rule) with
//! `BLOOMERY_CARD_BUDGET` bounding its card plan, the host
//! tier's load settings, and `BLOOMERY_PIN_MAIN`; the ubatch size
//! (`BLOOMERY_QWEN3_UBATCH`) is read where the load sizes its arena. The
//! stderr lines named above are records of the kinds
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds and exits.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use app::mtp::MtpBody;
use bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for;
use bloomery_gpu::arch::qwen3moe::{Body38, Prompt38};
use bloomery_gpu::model::StepMode;
use bloomery_gpu_gates::bind::{
    Seat, SeatEngine, Vocab, model_props, nvidia_smi_index, placement_props, sampler_factory,
};
use bloomery_gpu_gates::nodes::count_kinds;
use bloomery_gpu_gates::record::{self, Record};
use bloomery_gpu_gates::{GateError, ref_model_path};
use cuda_core::sys;
use gguf::Split;
use model::arch::models::HeadRows;
use model::arch::qwen35moe::place::{
    Experts, MtpInputs, PlanInputs, machine_for_experts, read_head_rows,
};
use model::placement::PlanLevers;
use model::placement::workstation::{A6000, CardSpec, RTX_3090};
use refset::arch::qwen4exp::mtp::DRAFT;
use runtime::Target as _;
use serve::{
    CacheNote, DraftProps, Drafted, EngineProps, FATAL_LINGER, Saved, ServeError, Server,
    ServerConfig,
};
use tokenizer::Tokenizer;

use super::drafted::DraftedSeat;

/// The levers `bloomery-serve-qwen38` acts on, for its own `main` and for
/// `gate_qwen38_serve`'s (the gate starts the server with its own
/// environment, so a lever the server would refuse is refused by the gate
/// first; the two lists must stay one).
pub const ACTS_ON: &[&str] = &[
    bloomery_levers::QWEN38_EXPERTS,
    bloomery_levers::CARD_BUDGET,
    bloomery_levers::PIN_MAIN,
    bloomery_levers::HOST_POPULATE,
    bloomery_levers::HOST_LOCK,
    bloomery_levers::CARD_DONTNEED,
    bloomery_levers::R8,
    bloomery_levers::DRAFT,
    bloomery_levers::MTP_HEAD_ROWS,
];

const USAGE: &str = "usage: bloomery-serve-qwen38 [--host H] [--port P] [--place a|gate] \
                     [--ctx-size C] [--alias NAME] [--chat-template-file PATH]";

/// The positions the stores are sized for when `--ctx-size` names none:
/// `generate_qwen3moe`'s default.
const CTX: usize = 4096;

/// Why no prefix is kept: the recurrent layers hold no state for an
/// earlier position.
const KEEP_WHY: &str = "the recurrent state (the GDN lanes, the PLE hash history) keeps no \
                        prefix: every request prefills from a reset";

/// Where `--place` puts the plan's one card, as `generate_qwen3moe`
/// takes it.
#[derive(Clone, Copy)]
enum Place38 {
    /// The A6000, the timing card (the default).
    A,
    /// The 3090, the gate card.
    Gate,
}

impl Place38 {
    fn parse(v: &str) -> Result<Place38, GateError> {
        match v {
            "a" => Ok(Place38::A),
            "gate" => Ok(Place38::Gate),
            other => Err(format!("--place is a or gate, not {other}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Place38::A => "a",
            Place38::Gate => "gate",
        }
    }

    fn spec(self) -> CardSpec {
        match self {
            Place38::A => A6000,
            Place38::Gate => RTX_3090,
        }
    }
}

/// `BLOOMERY_QWEN38_EXPERTS` as the plan's expert rule.
fn experts38(levers: &bloomery_levers::Levers) -> Result<Experts, GateError> {
    match levers.qwen38_experts() {
        "host" => Ok(Experts::Host),
        "card" => Ok(Experts::Card),
        other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
    }
}

struct Args {
    host: String,
    port: u16,
    place: Place38,
    ctx: usize,
    alias: Option<String>,
    /// `--chat-template-file`, replacing the file's own template.
    template_file: Option<PathBuf>,
}

fn parse_args(args: &[String]) -> Result<Args, GateError> {
    let mut a = Args {
        host: "127.0.0.1".to_owned(),
        port: 8080,
        place: Place38::A,
        ctx: CTX,
        alias: None,
        template_file: None,
    };
    let mut it = args.iter().map(|s| s.as_str());
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(USAGE.into());
        }
        let v = it
            .next()
            .ok_or_else(|| format!("{flag} needs a value, or is unknown: {USAGE}"))?;
        match flag {
            "--host" => a.host = v.to_owned(),
            "--port" => a.port = v.parse()?,
            "--place" => a.place = Place38::parse(v)?,
            "--ctx-size" | "--ctx" => a.ctx = v.parse()?,
            "--alias" => a.alias = Some(v.to_owned()),
            "--chat-template-file" => a.template_file = Some(PathBuf::from(v)),
            other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
        }
    }
    if a.ctx == 0 {
        return Err("--ctx-size 0: the stores hold no position".into());
    }
    Ok(a)
}

/// Loads the model and serves until the listener or the engine fails;
/// `Ok` carries why the server ended.
pub fn run(args: &[String]) -> Result<ServeError, GateError> {
    let levers = bloomery_levers::at_main(ACTS_ON)?;
    record::at_main("bloomery-serve-qwen38", record::BLOOMERY_SERVE_QWEN38);
    let a = parse_args(args)?;
    let mtp = match levers.draft() {
        None => false,
        Some("mtp") => true,
        Some(other) => {
            return Err(format!(
                "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window; lookup and \
                 dspark are the V4.1 binaries'"
            )
            .into());
        }
    };
    let head_rows = levers.mtp_head_rows().map(PathBuf::from);
    if !mtp && head_rows.is_some() {
        return Err(
            "BLOOMERY_MTP_HEAD_ROWS reduces the MTP draft's head; it needs BLOOMERY_DRAFT=mtp"
                .into(),
        );
    }
    let experts = experts38(&levers)?;
    let plan_levers = PlanLevers::from_levers(&levers)?;
    let path = ref_model_path()?;
    let vocab = Arc::new(Vocab::new(Tokenizer::from_gguf(&path)?)?);
    let inv = gguf::inventory_of(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let template = match &a.template_file {
        Some(file) => std::fs::read_to_string(file)
            .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))?,
        None => inv
            .value("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
            .to_owned(),
    };
    let name = inv
        .value("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("qwen3.8")
        .to_owned();
    drop(inv);

    // The plan record, and `/props` from the same plan the load runs by:
    // under the draft `plan_mtp_with`'s, its draft's card bytes (granules,
    // store, row map) and its program's arena the card's `draft` class.
    let split = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let inputs = PlanInputs::describe(&split)?;
    let model = model_props(&split, &inputs.model);
    let ub = ubatch_for(a.ctx)?;
    let machine = machine_for_experts(
        a.place.spec(),
        inputs.spec.layers.len(),
        u64::try_from(ub)?,
        experts,
    );
    let mtp_inputs = match mtp {
        false => None,
        true => {
            let rows = match head_rows.as_deref() {
                Some(p) => read_head_rows(p, &split, inputs.spec.vocab)?,
                None => HeadRows::Full,
            };
            let draft =
                Split::open(DRAFT).map_err(|e| format!("open the MTP draft {DRAFT}: {e}"))?;
            Some(MtpInputs::read(&draft, &split, &inputs, rows)?)
        }
    };
    drop(split);
    let (plan, draft_bytes) = match &mtp_inputs {
        None => (
            inputs.plan_with(&machine, u64::try_from(a.ctx)?, &plan_levers, experts)?,
            0,
        ),
        Some(mi) => {
            let with =
                inputs.plan_mtp_with(&machine, u64::try_from(a.ctx)?, &plan_levers, mi, experts)?;
            let bytes = with.draft_card_bytes() + with.arena_bytes;
            (with.plan, bytes)
        }
    };
    Record::new(&record::PLAN38)
        .w("place", a.place.name())
        .w("card", machine.cards[0].name.as_str())
        .w(
            "experts",
            if experts == Experts::Card {
                "card"
            } else {
                "host"
            },
        )
        .u("ctx_max", plan.ctx_max)
        .u("host_experts", plan.host.experts)
        .u("card_experts", plan.cards[0].experts)
        .eprint();
    let gpu = nvidia_smi_index(&machine.cards[0].name)
        .map(|i| format!("GPU{i}"))
        .and_then(|g| placement_props(&plan, &[g]));
    if let Err(e) = &gpu {
        eprintln!("bloomery-serve-qwen38: /props leaves the placement out: {e}");
    }
    let props = EngineProps {
        model: Some(model),
        placement: gpu.ok(),
        ..EngineProps::default()
    };

    let open = SeatArgs {
        place: a.place,
        ctx: a.ctx,
        experts,
        plan_levers,
        host: levers.host(),
        pin_main: levers.pin_main(),
        path: path.clone(),
        mtp,
        head_rows,
        draft_bytes,
    };
    // The prompt cache is off (its budget 0): the seat keeps no prefix
    // worth saving, and slot save/restore is the server's own 501.
    let engine = SeatEngine::spawn(
        move || Q38::open(open),
        a.ctx,
        vocab,
        machine.cards[0].name.clone(),
        props,
        0,
    )?;

    let config = ServerConfig {
        model_alias: a.alias.unwrap_or(name),
        model_path: path.display().to_string(),
        chat_template: template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: None,
    };
    let server = Server::bind((a.host.as_str(), a.port), Box::new(engine), config)?;
    Record::new(&record::LISTENING38)
        .w("place", a.place.name())
        .u("ctx", a.ctx)
        .w("addr", server.local_addr()?)
        .eprint();
    Ok(server.run())
}

/// What the engine thread opens the seat with.
struct SeatArgs {
    place: Place38,
    ctx: usize,
    experts: Experts,
    plan_levers: PlanLevers,
    host: bloomery_levers::HostCfg,
    pin_main: bool,
    path: PathBuf,
    /// `BLOOMERY_DRAFT=mtp`, the MTP draft; unset the plain path.
    mtp: bool,
    /// `BLOOMERY_MTP_HEAD_ROWS`, the draft head's row list.
    head_rows: Option<std::path::PathBuf>,
    /// The plan's `draft` class bytes, which `/props` files.
    draft_bytes: u64,
}

/// The Qwen3.8 session on the engine thread: the session over the model,
/// the draft it drives when one runs (its windows of four rows through
/// the runtime's speculative loop), and the positions its stores were
/// sized for.
struct Q38 {
    s: app::Session<Body38>,
    drafted: DraftedSeat<Body38, { <Body38 as MtpBody>::VERIFY_ROWS }>,
    ctx: usize,
    /// The plan's draft card bytes and arena, for `/props`' `draft` class.
    draft_bytes: u64,
}

impl Q38 {
    /// The session by `a.place` (the `load` and `capture` lines, as
    /// `generate_qwen3moe` prints them; under `mtp` the draft loaded
    /// beside the target, its own `load draft=mtp` line and the verify
    /// passes' capture lines after them), on the calling thread, pinned
    /// to the dispatcher's cpu slot when asked.
    fn open(a: SeatArgs) -> Result<Q38, GateError> {
        const WHAT: &str = "bloomery-serve-qwen38";
        if a.pin_main {
            // The engine thread runs every step; give it the pool's
            // dispatcher slot.
            let _ = threads::pool().pin_caller();
        }
        let t = Instant::now();
        let file = Split::open(&a.path).map_err(|e| format!("open {}: {e}", a.path.display()))?;
        let inputs = PlanInputs::describe(&file)?;
        let ub = ubatch_for(a.ctx)?;
        let machine = machine_for_experts(
            a.place.spec(),
            inputs.spec.layers.len(),
            u64::try_from(ub)?,
            a.experts,
        );
        let mut m = match a.mtp {
            false => {
                let plan =
                    inputs.plan_with(&machine, u64::try_from(a.ctx)?, &a.plan_levers, a.experts)?;
                Body38::open_placed(file, &plan, &inputs, 0, a.host, ub)?
            }
            true => {
                let rows = match a.head_rows.as_deref() {
                    Some(p) => read_head_rows(p, &file, inputs.spec.vocab)?,
                    None => HeadRows::Full,
                };
                let draft =
                    Split::open(DRAFT).map_err(|e| format!("open the MTP draft {DRAFT}: {e}"))?;
                let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
                let plan = inputs.plan_mtp_with(
                    &machine,
                    u64::try_from(a.ctx)?,
                    &a.plan_levers,
                    &mtp,
                    a.experts,
                )?;
                Body38::open_placed_mtp(file, &plan, &inputs, 0, a.host, ub, &draft, &mtp)?
            }
        };
        m.set_mode(StepMode::Graph);
        eprintln!(
            "load arch=qwen4exp resident_bytes={} ctx={} layers={} mode=graph store_bytes={} \
             prefill=auto ubatch={} place={} card_layers={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            a.ctx,
            m.layers().len(),
            m.body(WHAT)?.store_bytes(),
            m.body(WHAT)?.ubatch_rows(),
            a.place.name(),
            m.body(WHAT)?.card_layers(),
            t.elapsed().as_secs_f64()
        );
        if a.mtp {
            let d = m.body(WHAT)?.mtp().ok_or("the load opened no MTP draft")?;
            let bytes = (d.resident_bytes() + d.arena_bytes()) as u64;
            let head = match d.head_map() {
                Some((_, n)) => format!("rows={n}"),
                None => "full".to_string(),
            };
            eprintln!(
                "load draft=mtp resident={} arena={} head={head} card_bytes={bytes} (the \
                 plan's {}) in {:.1} s (runtime value)",
                d.resident_bytes(),
                d.arena_bytes(),
                a.draft_bytes,
                t.elapsed().as_secs_f64(),
            );
        }
        let (launches, memops) = m.body(WHAT)?.step_launches();
        let nodes = m.capture_step()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        eprintln!(
            "capture graph_nodes={nodes} kernel={k} batch_mem_op={b} other={other} (the \
             program counts {launches}, {memops} of them batch_mem_op)"
        );
        if nodes != launches || b != memops || k + b != nodes || other != 0 {
            return Err(format!(
                "the captured step is not the program's: {nodes} nodes ({k} kernel, {b} \
                 batch_mem_op, {other} other) against {launches} launches, {memops} of them \
                 batch_mem_op"
            )
            .into());
        }
        let mut s = app::Session::from_model(m, u32::try_from(a.ctx)?);
        let drafted = DraftedSeat::new(match a.mtp {
            false => None,
            true => {
                let draft = app::mtp::MtpDraft::open(s.model(), Prompt38::Auto, StepMode::Graph)?;
                struct Captures;
                impl app::RowsLog for Captures {
                    fn capture_rows(
                        &mut self,
                        rows: usize,
                        nodes: usize,
                    ) -> Result<(), app::SessionError> {
                        eprintln!("capture verify rows={rows} nodes={nodes}");
                        Ok(())
                    }
                }
                Some(s.with_draft(draft, &mut Captures)?)
            }
        });
        Ok(Q38 {
            s,
            drafted,
            ctx: a.ctx,
            draft_bytes: a.draft_bytes,
        })
    }
}

impl Seat for Q38 {
    fn pos(&self) -> usize {
        self.s.pos() as usize
    }

    fn ctx_max(&self) -> usize {
        self.ctx
    }

    /// The prompt through the ubatch walk `--prefill auto` takes: `gemm`
    /// from nine positions on, `pass` below — never one step per id;
    /// under the draft the draft's own prompt call, its store walked over
    /// the prompt's units.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
        self.drafted.prefill(&mut self.s, ids)
    }

    /// One step; under the draft the rows it left waiting walked first
    /// (`MtpDraft::before_step`: a request that continues the held
    /// sequence joins it here when its prompt call is empty).
    fn step(&mut self, last: u32) -> Result<u32, GateError> {
        self.drafted.step(&mut self.s, last)
    }

    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError> {
        Ok(self.s.model().logits_into(row)?)
    }

    fn reset(&mut self) -> Result<(), GateError> {
        self.drafted.reset(&mut self.s)
    }

    /// One pass from `last`: under the draft the window of four rows,
    /// its kept tokens and counts; without it one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        self.drafted.pass(&mut self.s, last, out)
    }

    /// The most positions one pass runs: the draft's four rows, or one
    /// step without it.
    fn pass_rows(&self) -> usize {
        self.drafted.pass_rows()
    }

    /// The commit's rule, `Body38::kept`: with no verify waiting only
    /// the position the stores stand at is kept — nothing to take back —
    /// and a request's shorter prefix grants nothing. [`KEEP_WHY`] is
    /// the rule string the server's reuse records print.
    fn keep(&self, n: usize) -> (usize, Option<String>) {
        let pos = self.s.pos() as usize;
        if n >= pos {
            return (pos, None);
        }
        (0, Some(KEEP_WHY.to_owned()))
    }

    /// The body's commit, which only a waiting verify serves; a request
    /// path never reaches it (the server resets instead of cutting), so
    /// any other position is refused by name, the body's own message.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError> {
        Ok(self.s.model_mut().rollback(pos)?)
    }

    /// Nowhere: the engine cuts a prompt call only where a later request
    /// can keep the cut, and this engine keeps no cut.
    fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
        Vec::new()
    }

    /// `/props`' `engine.draft` under the MTP draft: its kind, the draft
    /// file, its width and its resident bytes as the card's `draft`
    /// class.
    fn props(&self, mut p: EngineProps) -> EngineProps {
        if !self.drafted.drafts() {
            return p;
        }
        if let Some(place) = p.placement.as_mut() {
            for d in place.devices.iter_mut() {
                if d.device.starts_with("GPU") {
                    d.class_bytes.insert("draft".to_owned(), self.draft_bytes);
                }
            }
        }
        p.draft = Some(DraftProps {
            model: "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf".to_owned(),
            n_max: Some(<Body38 as MtpBody>::WIDTH as u64),
            kind: Some("mtp".to_owned()),
            path: Some(DRAFT.to_owned()),
            device: None,
        });
        p
    }

    /// A Qwen3.8 sequence state is not a value a cache could hold: the
    /// recurrent stores and the PLE history have no copy to put back.
    /// The server's prompt cache is off, so this is never asked; a caller
    /// that asks is refused by name.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
        Err(format!("a snapshot at position {}: {KEEP_WHY}", self.s.pos()).into())
    }

    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError> {
        Err(format!(
            "a resume of a state of {} positions: {KEEP_WHY}",
            state.n_tokens()
        )
        .into())
    }

    /// Only `Reuse` arrives (the cache is off and no call is cut); the
    /// other notes belong to a caching engine and print as they come.
    fn note(note: &CacheNote) {
        if let CacheNote::Reuse {
            common,
            ask,
            kept,
            held,
            reason,
        } = note
        {
            Record::new(&record::CACHE_REUSE)
                .u("common", common)
                .u("ask", ask)
                .u("kept", kept)
                .u("held", held)
                .w("reason", reason.as_deref().unwrap_or("unstated"))
                .eprint();
            return;
        }
        eprintln!("bloomery-serve-qwen38: {note}");
    }
}
