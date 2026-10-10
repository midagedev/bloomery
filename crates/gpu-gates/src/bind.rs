//! The server's engine over a [`Seat`]: [`SeatEngine`] implements
//! `serve::Engine`, [`Vocab`] implements `serve::Tokenizer` over the file's
//! own vocabulary, and [`sampler_factory`] builds each request's sampler from
//! the sampler crate.
//!
//! The seat — the model, its draft if one runs, and the body's rules — lives
//! on one thread of its own, for the whole run. The server calls its engine
//! from whichever connection thread holds the slot; the model's device state
//! and the pinned dispatcher slot belong to the thread that opened them, so
//! every step runs on that thread and the engine is a handle that sends it
//! commands. The opener comes in as a closure, and the seat as a trait this
//! library leaves open (the serve binaries' seat tree, `shared/serve_seats/`,
//! holds its implementers, the device crates' seats), because this module
//! names no device crate (see [`crate::generate`]) or the session crate.
//!
//! How long a prefix of the cache can be kept, and why no longer, is the
//! seat's rule, which the engine thread answers ([`SeatEngine`]'s `keepable`
//! and `keep_limit`); so are the prompt feed, a pass of the draft
//! ([`Seat::pass`]), where a prompt call is cut, and the state the server's
//! prompt cache saves and puts back ([`Seat::snapshot`], [`Seat::resume`]). How
//! far the positions go is the lesser of the cache and the positions the body
//! computes the model at (`spawn`'s `defined`): the server refuses a request
//! past it, so a client's long prompt or `max_tokens` never reaches the body's
//! refusal, which is fatal.
//!
//! A sampled token's logits cross to the calling thread in one buffer the
//! engine owns: lent with each `Next` that wants them, filled on the engine
//! thread, and handed back with the reply, so no token allocates a row.
//!
//! A slot file's state ([`Seat::save_state`], [`Seat::restore_state`])
//! crosses between the server's thread and the engine thread through a pipe
//! of chunks of `PIPE_CHUNK` bytes, at most `PIPE_DEPTH` of them waiting,
//! never staged whole: past the seat's own copy the host holds a few chunks.
//! The seat's error kind crosses with the answer, so only an engine failure
//! ends the server.
//!
//! What `/props` says about the engine ([`model_props`], [`placement_props`])
//! is read from the file's header and the placement plan the engine loads by,
//! once, before the load; the engine hands it back unchanged.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::JoinHandle;

use gguf::Split;
use model::placement::workstation::{self, DeviceId, HostRead};
use model::placement::{Device, Machine, ModelTensors, Plan, Role};
use sampler::{Sampler, SamplerParams};
use serve::media::{MediaFeed, SharedMediaModel};
use serve::{
    CacheNote, Decoder, DeviceProps, Drafted, Engine, EngineError, EngineProps, FATAL_LINGER,
    ModelProps, PlacementProps, ResidencyReset, SamplerFactory, SamplerRefused, SamplingParams,
    Saved, SavedState, ServeError, Server, ServerConfig, SlotConfig, StateError, Tokenizer,
};

use crate::GateError;
use crate::record::Record;

/// The file's vocabulary as the server reads it.
pub struct Vocab {
    tok: Arc<tokenizer::Tokenizer>,
    bos: u32,
    eos: u32,
    user_start: Vec<u32>,
}

impl Vocab {
    /// Wraps a loaded vocabulary; refused when it names no BOS or EOS.
    pub fn new(tok: tokenizer::Tokenizer) -> Result<Vocab, GateError> {
        let sp = tok.specials();
        let bos = sp.bos.ok_or("the vocabulary names no BOS token")?;
        let eos = sp.eos.ok_or("the vocabulary names no EOS token")?;
        Ok(Vocab {
            tok: Arc::new(tok),
            bos,
            eos,
            user_start: Vec::new(),
        })
    }

    /// The same vocabulary with `marker`, one of its special tokens, as the
    /// token that opens a user message ([`Tokenizer::user_start`]); refused
    /// when no special token has that text.
    pub fn with_user_start(mut self, marker: &str) -> Result<Vocab, GateError> {
        let id = self
            .tok
            .special_tokens()
            .iter()
            .copied()
            .find(|&id| self.tok.text(id) == Some(marker))
            .ok_or_else(|| format!("the vocabulary has no special token {marker:?}"))?;
        self.user_start = vec![id];
        Ok(self)
    }

    /// The vocabulary itself.
    pub fn tokenizer(&self) -> &tokenizer::Tokenizer {
        &self.tok
    }
}

impl Tokenizer for Vocab {
    fn encode(&self, text: &str) -> Vec<u32> {
        // The rendered prompt carries the BOS string and the role markers as text.
        self.tok.encode(text, false, true)
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(VocabDecoder {
            tok: Arc::clone(&self.tok),
            pending: Vec::new(),
        })
    }

    fn bos(&self) -> u32 {
        self.bos
    }

    fn eos(&self) -> u32 {
        self.eos
    }

    fn stops(&self) -> Vec<u32> {
        // The reference's end-of-generation set: the header's eos, eot and eom.
        self.tok.eog().to_vec()
    }

    fn add_bos(&self) -> bool {
        self.tok.add_bos()
    }

    fn n_vocab(&self) -> usize {
        self.tok.n_vocab()
    }

    fn user_start(&self) -> Vec<u32> {
        self.user_start.clone()
    }
}

/// `tokenizer::Decoder`'s rule over an owned vocabulary handle: pieces with
/// special tokens rendered, an incomplete UTF-8 tail held, an invalid
/// sequence replaced by U+FFFD. `tokenizer::Decoder` borrows its vocabulary,
/// and the server's decoder must outlive the call that made it.
struct VocabDecoder {
    tok: Arc<tokenizer::Tokenizer>,
    pending: Vec<u8>,
}

impl Decoder for VocabDecoder {
    fn push(&mut self, id: u32) -> Option<String> {
        self.pending.extend_from_slice(self.tok.piece(id, true));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    // The first `valid` bytes were just checked.
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn flush(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}

/// The server's `SamplingParams` as the sampler crate's chain: the penalties
/// over the request's window, `top_k <= 0` keeps every candidate. The server
/// never builds a sampler for `temperature <= 0`: those requests take the
/// engine's argmax.
pub fn sampler_params(p: &SamplingParams) -> SamplerParams {
    SamplerParams {
        temperature: p.temperature,
        top_k: u32::try_from(p.top_k).unwrap_or(0),
        top_p: p.top_p,
        // Past 1 keeps only the best candidate, as 1 does; the chain refuses it.
        min_p: p.min_p.min(1.0),
        repeat_penalty: p.repeat_penalty,
        frequency_penalty: p.frequency_penalty,
        presence_penalty: p.presence_penalty,
        repeat_last_n: p.repeat_last_n,
        seed: p.seed,
    }
}

/// A factory of sampler-crate samplers. A parameter the chain still refuses
/// (a non-finite value) refuses the request, in the chain's own words.
pub fn sampler_factory() -> SamplerFactory {
    Arc::new(
        |p: &SamplingParams| -> Result<serve::Sampler, SamplerRefused> {
            let mut s =
                Sampler::new(sampler_params(p)).map_err(|e| SamplerRefused(e.to_string()))?;
            Ok(Box::new(move |logits: &[f32], recent: &[u32]| {
                s.sample(logits, recent)
            }))
        },
    )
}

/// The chat surface a seat's file gives the server: the Jinja template and
/// the model's name.
pub struct ChatSurface {
    /// `--chat-template-file`'s text where the seat takes the flag, else the
    /// file's `tokenizer.chat_template`.
    pub template: String,
    /// The file's `general.name`, else the seat's default alias.
    pub name: String,
}

impl ChatSurface {
    /// The surface of the file at `path` from its header's `tokenizer.chat_template`
    /// and `general.name` (`template` and `name`): `template_file`'s text
    /// replaces the header's template, and a file with neither is refused by
    /// name; `name` falls to `default_alias` when the file states none.
    pub fn read(
        path: &Path,
        template: Option<&gguf::Value>,
        name: Option<&gguf::Value>,
        template_file: Option<&Path>,
        default_alias: &str,
    ) -> Result<ChatSurface, GateError> {
        let template = match template_file {
            Some(file) => std::fs::read_to_string(file)
                .map_err(|e| format!("--chat-template-file {}: {e}", file.display()))?,
            None => template
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{}: no tokenizer.chat_template", path.display()))?
                .to_owned(),
        };
        let name = name
            .and_then(|v| v.as_str())
            .unwrap_or(default_alias)
            .to_owned();
        Ok(ChatSurface { template, name })
    }
}

/// What a seat's server binds with, past its engine.
pub struct Listen<'a> {
    pub host: &'a str,
    pub port: u16,
    /// `model` in responses and the `/v1/models` id.
    pub alias: String,
    /// The model file, `/props`' `model_path`.
    pub path: &'a Path,
    pub template: String,
    /// `--slot-save-path`; `None` refuses every slot action.
    pub slot_save_path: Option<PathBuf>,
    pub api_keys: serve::flag::ApiKeys,
    /// The slots the engine made at its open, which the server steps together.
    pub parallel: usize,
    pub queue_depth: Option<usize>,
}

/// The server over a seat's `engine`: the sampler crate's chain, the fatal
/// linger every seat keeps, the seat's own slots as the server's. The seat
/// prints its own `listening` record from the bound server and runs it.
pub fn bind_server(engine: SeatEngine, l: Listen<'_>) -> Result<Server, ServeError> {
    let config = ServerConfig {
        model_alias: l.alias,
        model_path: l.path.display().to_string(),
        chat_template: l.template,
        sampler: Some(sampler_factory()),
        fatal_linger: FATAL_LINGER,
        slot_save_path: l.slot_save_path,
        api_keys: l.api_keys,
    };
    let slots = SlotConfig {
        parallel: l.parallel,
        queue_depth: l.queue_depth,
        ..SlotConfig::default()
    };
    Server::bind_with((l.host, l.port), Box::new(engine), config, slots)
}

/// The slots a seat serves and what set the count (`from`, as
/// `model::placement::ctx::slots_of` words it); a set `--ctx` with no
/// `--parallel` prints the one line that says the flag is one request's
/// context.
pub fn slots_given(
    parallel: Option<usize>,
    ctx: Option<usize>,
    default: usize,
) -> Result<(usize, &'static str), GateError> {
    let (slots, from) = model::placement::ctx::slots_of(parallel, ctx.is_some(), default)?;
    if from == "ctx" {
        eprintln!(
            "--ctx-size {} is one request's context; add --parallel N to serve N requests at \
             once (they split it)",
            ctx.unwrap_or_default()
        );
    }
    Ok((slots, from))
}

/// The `parallel` line a seat prints before its load: the slots, a slot's
/// context, the total, what set the count (`from`), and the shape of a round
/// (`pass`) where the seat has one.
pub fn parallel_line(slots: usize, slot_ctx: usize, from: &str, pass: Option<&str>) {
    let pass = pass.map_or_else(String::new, |p| format!(" pass={p}"));
    eprintln!(
        "parallel rule=slots slots={slots} slot_ctx={slot_ctx} total={} from={from}{pass}",
        slots * slot_ctx
    );
}

/// `/props`' placement of `plan` on `machine`: its cards by nvidia-smi
/// index. A placement that cannot be named prints one line under `name` and
/// is left out.
pub fn placement_of(name: &str, machine: &Machine, plan: &Plan<'_>) -> Option<PlacementProps> {
    let placement = machine
        .all_cards()
        .map(|c| nvidia_smi_index(&c.name, c.device).map(|i| format!("GPU{i}")))
        .collect::<Result<Vec<String>, String>>()
        .and_then(|g| placement_props(plan, &g));
    if let Err(e) = &placement {
        eprintln!("{name}: /props leaves the placement out: {e}");
    }
    placement.ok()
}

/// A seat's [`Seat::note`]: a prefix kept less of than shared is a `cache
/// reuse` record; every other note prints as `name`'s line.
pub fn note_line(name: &str, note: &CacheNote) {
    if let CacheNote::Reuse {
        common,
        ask,
        kept,
        held,
        reason,
    } = note
    {
        Record::new(&crate::record::CACHE_REUSE)
            .u("common", common)
            .u("ask", ask)
            .u("kept", kept)
            .u("held", held)
            .w("reason", reason.as_deref().unwrap_or("unstated"))
            .eprint();
        return;
    }
    eprintln!("{name}: {note}");
}

/// The class toktape files a placement role's bytes under: the buckets of its
/// GGUF classifier, which puts the router and the shared expert with the
/// routed experts and every engram tensor with the n-gram tables.
fn class_of(role: Role) -> &'static str {
    match role {
        Role::Attention => "attention",
        Role::Router | Role::HashTable | Role::SharedExpert | Role::RoutedExperts => "experts",
        Role::FfnNorm | Role::DenseFfn => "ffn",
        Role::EngramDense | Role::EngramGain | Role::EngramTable => "ngram",
        Role::TokenEmbedding => "embeddings",
        Role::Head => "output",
        Role::HyperConnection | Role::Unread | Role::Unused => "other",
    }
}

/// A stage's layers as `first-last`.
fn layer_range(layers: &Range<usize>) -> String {
    match layers.end.checked_sub(1) {
        Some(last) if last >= layers.start => format!("{}-{last}", layers.start),
        _ => "none".to_owned(),
    }
}

/// `/props`' `engine.model` of the file `split`, whose tensors its
/// architecture classified as `model`: the architecture, the type that holds
/// the most bytes of the tensors a step reads (the row-gathered tables — the
/// token embedding and the engram tables — and the never-read tensors left
/// out), the bytes of every shard on disk, the layer and expert counts, and the
/// training context. A fact the header does not state is left out.
pub fn model_props(split: &Split, model: &ModelTensors) -> ModelProps {
    let mut by_type: BTreeMap<&'static str, u64> = BTreeMap::new();
    for t in &model.tensors {
        let gathered_or_unread = matches!(
            t.role,
            Role::TokenEmbedding | Role::EngramTable | Role::Unread | Role::Unused
        );
        if let (false, Some(name)) = (gathered_or_unread, t.ty.name()) {
            *by_type.entry(name).or_default() += t.file_bytes;
        }
    }
    let quant = by_type
        .iter()
        .max_by_key(|&(_, &bytes)| bytes)
        .map(|(&name, _)| name.to_owned());
    let bytes = (0..split.shard_count())
        .map(|i| {
            split
                .shard_path(i)
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
        })
        .sum();
    ModelProps {
        format: Some("gguf".to_owned()),
        arch: split.architecture().map(str::to_owned),
        quant,
        bytes,
        files: u64::try_from(split.shard_count()).ok(),
        n_layers: u64::try_from(model.layers).ok(),
        n_experts: Some(model.experts),
        n_experts_used: Some(model.experts_used),
        ctx_train: split.arch_get_u64("context_length"),
    }
}

/// `/props`' `engine.placement` of `plan`, the plan the engine loads by: per
/// card of the machine — the stage cards, then the expert tier cards — named
/// `gpus[c]` (`GPU<n>`, see [`nvidia_smi_index`]), the resident bytes of its
/// segments by class and a stage card's layers (a tier card runs none, and
/// its row has no `layers`); then the host's (`CPU`) the same way; and the
/// cards' KV bytes, cache and shadow. Only the plan's own rows are summed, so
/// a card's classes add up to its `dense_bytes + expert_bytes` and the host's
/// to its `expert_bytes + table_bytes`. The NVMe tier holds no resident bytes
/// and is no device here.
pub fn placement_props(plan: &Plan<'_>, gpus: &[String]) -> Result<PlacementProps, String> {
    let machine: Vec<_> = plan.machine.all_cards().collect();
    if gpus.len() != plan.cards.len() || machine.len() != plan.cards.len() {
        return Err(format!(
            "{} card names for a plan of {} cards on a machine of {}",
            gpus.len(),
            plan.cards.len(),
            machine.len()
        ));
    }
    let mut cards = vec![BTreeMap::<String, u64>::new(); plan.cards.len()];
    let mut host = BTreeMap::<String, u64>::new();
    for r in &plan.rows {
        let t = plan
            .model
            .tensors
            .get(r.tensor)
            .ok_or_else(|| format!("a plan row names tensor {}, past the model", r.tensor))?;
        for seg in &r.segments {
            let device = match seg.device {
                Device::Card(c) => cards.get_mut(c),
                Device::Host => Some(&mut host),
                Device::Nvme | Device::Unused => None,
            };
            if let Some(classes) = device {
                *classes.entry(class_of(t.role).to_owned()).or_default() += seg.resident_bytes;
            }
        }
    }
    let stages = plan.machine.cards.len();
    let mut devices: Vec<DeviceProps> = cards
        .into_iter()
        .zip(gpus)
        .zip(machine)
        .enumerate()
        .map(|(i, ((class_bytes, gpu), card))| DeviceProps {
            device: gpu.clone(),
            class_bytes,
            layers: (i < stages).then(|| layer_range(&card.layers)),
        })
        .collect();
    devices.push(DeviceProps {
        device: "CPU".to_owned(),
        class_bytes: host,
        layers: None,
    });
    Ok(PlacementProps {
        devices,
        vram_kv_bytes: Some(plan.cards.iter().map(|c| c.kv_bytes).sum()),
    })
}

/// The nvidia-smi index of the card a plan's card `name` (resolved to
/// `device` when the plan was, else found by `name`) opens on: the one
/// visible device of this process's census, matched to nvidia-smi's list by
/// its PCI bus id ([`workstation::listed_index`]).
/// Two cards of one name are two buses, so each plan card names its own
/// index; a card not in view, or a bus nvidia-smi does not list once, is an
/// error that says what was listed.
pub fn nvidia_smi_index(name: &str, device: Option<DeviceId>) -> Result<u32, String> {
    let census = bloomery_gpu::census().map_err(|e| format!("the device census: {e}"))?;
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi: {}", out.status));
    }
    workstation::listed_index(name, device, &census, &String::from_utf8_lossy(&out.stdout))
}

/// The most the prompt cache takes by default: llama-server's `--cache-ram`
/// default.
pub const CACHE_RAM_CAP: u64 = 8192 << 20;

/// The prompt cache's budget of a seat whose saved states live in host RAM
/// beside a load's host set (llama-server's `--cache-ram`, in MiB; 0 turns
/// it off), and its terms: set, as given; else the lesser of
/// [`CACHE_RAM_CAP`] and half of what the host's available bytes —
/// `MemAvailable`, or the smaller room under a cgroup v2 limit
/// (`workstation::host_available_read`) — leave past the plan's host need
/// and the residency's churn pool, the need holding the NVMe expert tier's
/// arena ([`CacheRam::of_tier`]) and the checkpoints the load's sequences
/// pin (`residency38::reserve_checkpoints`, the machine's host reserve), 0
/// when nothing is left — a zero the line's `why` names ([`CacheRam::why`]),
/// since a load with no prompt cache still serves correctly, every state
/// recomputed. A plan the NVMe expert tier split defaults to 0 too: the
/// split fills the room past the tier's floor with its arena and the host
/// experts it keeps, so what the reading leaves past the need is the split's
/// rounding or the reading's drift, never room the plan left. A default too
/// small for one saved state is 0 ([`CacheRam::holding`]).
pub struct CacheRam {
    pub ram: u64,
    /// `--cache-ram` given, or the default's terms.
    pub set: bool,
    /// The host's available bytes before the load.
    pub available: u64,
    /// Which reading gave `available` ([`HostRead`]).
    pub reading: HostRead,
    /// The plan's host need.
    pub need: u64,
    /// The residency's churn pool, 0 without one.
    pub pool: u64,
    /// The plan's NVMe expert tier arena, 0 without one: the anonymous
    /// bytes the tier's slots fill beside the plan's own host terms. Inside
    /// `need` already ([`HostNeed::bytes`] counts the arena); kept here for
    /// the line.
    pub tier: u64,
    /// The routed-expert bytes the plan's host leg reads from the NVMe
    /// tier (`HostTotals::nvme_expert_bytes`), 0 on a plan the tier did not
    /// split; above 0 the default budget is 0 (the type's doc).
    pub paged: u64,
    /// The checkpoint bytes the plan's machine reserves
    /// (`residency38::checkpoints_reserved`), 0 for a load that pins none.
    /// Inside `need` already; kept here for the line.
    pub checkpoints: u64,
    /// Why the default budget is 0, named for the line's `why=` term: the
    /// reading left nothing past the need and the pool; the NVMe tier's
    /// split took the room; or the budget was under one saved state. `None`
    /// for a budget above 0 and for a set one (`--cache-ram 0` is the
    /// caller's own ask).
    pub why: Option<String>,
}

impl CacheRam {
    /// The `flag`'s value (`--cache-ram`, `--park-ram`), `mib` MiB, in
    /// bytes; refused by name past u64.
    pub fn parse_mib(flag: &str, mib: &str) -> Result<u64, GateError> {
        let n: u64 = mib.parse()?;
        Ok(n.checked_mul(1 << 20)
            .ok_or_else(|| format!("{flag} {n} MiB passes u64 bytes"))?)
    }

    /// The budget (the type's doc): `set` in bytes as given, else the
    /// default over `need` and `pool`, `need` holding the `checkpoints` the
    /// plan's machine reserves (`residency38::checkpoints_reserved`) —
    /// [`CacheRam::of_tier`] with no NVMe expert tier.
    pub fn of(
        set: Option<u64>,
        need: u64,
        pool: u64,
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        Self::of_tier(set, need, pool, (0, 0), checkpoints)
    }

    /// [`CacheRam::of`] on a plan the NVMe expert tier split: `(tier,
    /// paged)` its arena's bytes — inside `need` already ([`HostNeed::bytes`]
    /// counts the arena) — and the routed-expert bytes its host leg reads
    /// from the tier. A split plan's default is 0 (the type's doc): the
    /// arena's pages are anonymous, so a cache beside the room the split
    /// filled is the kernel's OOM or swap, not a page it reclaims.
    pub fn of_tier(
        set: Option<u64>,
        need: u64,
        pool: u64,
        (tier, paged): (u64, u64),
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        let (available, reading) = workstation::host_available_read()?;
        Self::at(
            set,
            (available, reading),
            need,
            pool,
            (tier, paged),
            checkpoints,
        )
    }

    /// [`CacheRam::of_tier`] at a host reading already taken. A `need`
    /// under the `tier` and `checkpoints` it holds is refused by name: it is
    /// a need read without the arena or of a machine with no checkpoint
    /// reserve, and the budget would take their bytes.
    fn at(
        set: Option<u64>,
        (available, reading): (u64, HostRead),
        need: u64,
        pool: u64,
        (tier, paged): (u64, u64),
        checkpoints: u64,
    ) -> Result<CacheRam, GateError> {
        if tier > need {
            return Err(format!(
                "the prompt cache's need {need} B is under the NVMe tier arena {tier} B it holds: \
                 a need read without the arena (HostNeed::bytes counts it)"
            )
            .into());
        }
        if tier > 0 && paged == 0 {
            return Err(format!(
                "the prompt cache's plan holds an NVMe tier arena of {tier} B and pages no routed \
                 expert: the arena exists only on a plan the tier split"
            )
            .into());
        }
        if checkpoints > need - tier {
            return Err(format!(
                "the prompt cache's need {need} B is under the NVMe tier arena {tier} B and the \
                 checkpoints {checkpoints} B it holds: a need read of a machine with no \
                 checkpoint reserve (residency38::reserve_checkpoints)"
            )
            .into());
        }
        let (ram, why) = match set {
            Some(ram) => (ram, None),
            None if paged > 0 => (
                0,
                Some(format!(
                    "the NVMe tier's split fills the room past its floor: arena {tier} B and the \
                     host experts it keeps inside need {need} B, {paged} B read from the drive; \
                     no prompt cache, every state recomputed"
                )),
            ),
            None => Self::default_of(available, need, pool),
        };
        Ok(CacheRam {
            ram,
            set: set.is_some(),
            available,
            reading,
            need,
            pool,
            tier,
            paged,
            checkpoints,
            why,
        })
    }

    /// This budget, a default under `entry` bytes — the least one state the
    /// seat's cache saves holds (one position, its fixed terms whole) — made
    /// 0 with a `why` naming both: the cache would refuse every state it is
    /// handed (`serve`'s `PromptCache::insert`), so it is off. A set budget
    /// keeps its value: it is the caller's ask, and the seat refuses one too
    /// small by name where its rule reads it.
    #[must_use]
    pub fn holding(self, entry: u64) -> CacheRam {
        if self.set || self.ram == 0 || self.ram >= entry {
            return self;
        }
        CacheRam {
            ram: 0,
            why: Some(format!(
                "the default budget {} B is under one saved state's {entry} B: no prompt cache, \
                 every state recomputed",
                self.ram
            )),
            ..self
        }
    }

    /// The default budget ([`CacheRam`]'s doc): half of what `available`
    /// leaves past `need` and `pool`, capped at [`CACHE_RAM_CAP`]. A
    /// leftover of nothing is a named record, not a silent zero: the seat
    /// serves correctly without a prompt cache (every state recomputed), so
    /// the line states the arithmetic that left nothing instead of the load
    /// refusing.
    fn default_of(available: u64, need: u64, pool: u64) -> (u64, Option<String>) {
        let left = i128::from(available) - i128::from(need) - i128::from(pool);
        if left / 2 <= 0 {
            return (
                0,
                Some(format!(
                    "the reading {available} B leaves {left} B past need {need} B + pool {pool} B: \
                     no prompt cache, every state recomputed"
                )),
            );
        }
        (
            u64::try_from(left / 2).map_or(CACHE_RAM_CAP, |h| h.min(CACHE_RAM_CAP)),
            None,
        )
    }

    /// The `cache` line's terms, the seat's own fields to follow: `cache
    /// ram=… rule=set|default available=… read=… need=… pool=… tier=…
    /// paged=… checkpoints=…`, `read` the reading that gave `available`
    /// ([`HostRead::word`]); a default budget of 0 carries `why=…`
    /// ([`CacheRam::why`]).
    #[must_use]
    pub fn line(&self) -> String {
        let line = format!(
            "cache ram={} rule={} available={} read={} need={} pool={} tier={} paged={} \
             checkpoints={}",
            self.ram,
            if self.set { "set" } else { "default" },
            self.available,
            self.reading.word(),
            self.need,
            self.pool,
            self.tier,
            self.paged,
            self.checkpoints
        );
        match &self.why {
            Some(why) => format!("{line} why={why}"),
            None => line,
        }
    }
}

/// The bytes of one chunk of a slot file's pipe (the module doc).
const PIPE_CHUNK: usize = 4 << 20;

/// The chunks a slot file's pipe holds waiting: with the one each thread
/// holds in hand, the host peak past the seat's own copy is
/// `(PIPE_DEPTH + 2) · PIPE_CHUNK`.
const PIPE_DEPTH: usize = 2;

/// What the engine thread is asked to do.
enum Cmd {
    Prefill(Vec<u32>),
    /// [`Seat::prefill_media`]: `ids` with the images they carry, one feed a
    /// span, every span whole inside the call ([`Engine::prefill_media`]'s
    /// contract).
    PrefillMedia(Vec<u32>, Vec<MediaFeed>),
    /// With the engine's logits buffer when the caller wants the row.
    Next {
        last: u32,
        logits: Option<Vec<f32>>,
    },
    /// A pass from `last`, its kept tokens into the caller's buffer.
    Pass {
        last: u32,
        out: Vec<u32>,
    },
    /// A sampled pass from `last` ([`Seat::pass_sampled`]): the request's
    /// ids so far and its sampler cross to the thread and back with the
    /// reply, the sampler's state with it.
    PassSampled {
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        out: Vec<u32>,
    },
    Reset,
    /// Take back the positions from this one on.
    Rollback(u32),
    /// The longest prefix of at most this many positions a rollback keeps
    /// for a reply of at most this many tokens through passes
    /// ([`Seat::keep_for`]), and the rule that kept less.
    Keep(usize, Option<usize>),
    /// Where to cut a prompt call of `first .. end` at `marks`.
    Splits {
        first: usize,
        end: usize,
        marks: Vec<usize>,
    },
    /// The sequence state, as a value.
    Save,
    /// Replace the sequence state with a saved one.
    Resume(Arc<dyn Saved>),
    /// The selected slot's state written into the pipe ([`save_into`]).
    SaveState(SyncSender<Vec<u8>>),
    /// The selected slot's state read from the pipe ([`restore_from`]).
    RestoreState(Receiver<Vec<u8>>),
    /// Nothing: the reply carries the position the model stands at.
    Pos,
    /// The residency back to its seed.
    ResidencyReset,
    /// Make `slot` the one every later command acts on ([`Seat::select`]).
    Select(usize),
    /// A step of several slots in one command ([`SeatEngine`]'s
    /// `step_slots`): each row's `last` on its own slot, in row order, the
    /// answers and lent rows back with the one reply.
    StepSlots(Vec<SlotStep>),
    /// A drafted pass of several slots in one command ([`SeatEngine`]'s
    /// `advance_slots`): each row's `last` passed on its own slot, in row
    /// order, the kept ids and what each pass drafted back with the one
    /// reply.
    PassSlots(Vec<SlotPassRow>),
}

/// One row of a [`Cmd::StepSlots`] and of a [`Seat::step_slots`] round: the
/// slot, the id it evaluates, the engine's logits row lent for it
/// (`n_vocab` f32, filled only when the row asked), and the engine's
/// answer, which the seat sets.
pub struct SlotStep {
    pub slot: usize,
    pub last: u32,
    pub logits: Option<Vec<f32>>,
    pub next: u32,
}

/// One row of a [`Cmd::PassSlots`] and of a [`Seat::pass_slots`] round: the
/// slot, the id its pass runs, the caller's buffer of the pass's kept ids,
/// and what the pass drafted, which the seat sets.
pub struct SlotPassRow {
    pub slot: usize,
    pub last: u32,
    pub out: Vec<u32>,
    pub drafted: Drafted,
}

/// What a reply carries besides its result.
enum Extra {
    None,
    /// A `Keep`'s rule.
    Why(Option<String>),
    /// A `Splits`' cuts.
    Splits(Vec<usize>),
    /// A `Save`'s state, or its failure with its kind ([`snapshot_of`]).
    Saved(Result<Arc<dyn Saved>, StateError>),
    /// A `Resume`'s outcome with its kind ([`resume_of`]).
    Resumed(Result<(), StateError>),
    /// A `SaveState`'s or `RestoreState`'s outcome, its kind as the seat
    /// gave it.
    State(Result<SavedState, StateError>),
    /// A `Pass`'s buffer, holding its kept tokens on success, and what its
    /// draft proposed and kept.
    Pass(Vec<u32>, Drafted),
    /// A `PassSampled`'s buffer, holding its taken ids on success, what its
    /// draft proposed and kept, and the history and sampler it was sent.
    PassSampled {
        out: Vec<u32>,
        drafted: Drafted,
        history: Vec<u32>,
        sampler: serve::Sampler,
    },
    /// A `ResidencyReset`'s report; `None` for a seat with no residency.
    Residency(Option<ResidencyReset>),
    /// A `StepSlots`' rows, their answers set and their lent rows filled.
    StepSlots(Vec<SlotStep>),
    /// A `PassSlots`' rows, their kept ids and drafted counts set.
    PassSlots(Vec<SlotPassRow>),
}

/// Its answer: the argmax of a `Next` (the kept length of a `Keep`), the
/// logits buffer a `Next` was lent (filled unless the result is an error),
/// the position it stands at afterwards (the position it failed at, on an
/// error), and what a `Keep` or `Splits` returns besides. A `Save`,
/// `Resume`, `SaveState` or `RestoreState` answers in its extra alone, with
/// its kind.
struct Reply {
    result: Result<u32, String>,
    logits: Option<Vec<f32>>,
    pos: usize,
    extra: Extra,
}

/// What the engine thread runs every command on: the model standing at a
/// position, the draft that follows it if one runs, and the body's rules
/// ([`SeatEngine::spawn`] opens it on the thread). A failed call says what
/// failed; the thread adds the command and the position.
pub trait Seat: 'static {
    /// The position the next fed id lands in.
    fn pos(&self) -> usize;
    /// The positions the caches were sized for: `pos` never passes it.
    fn ctx_max(&self) -> usize;
    /// The body's prompt feed of `ids` from `pos` (a batched body's batch, or
    /// one step per id), the draft fed as the call hands its rows over; the
    /// argmax after the last.
    fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError>;
    /// [`Seat::prefill`] where the slice carries images
    /// ([`Engine::prefill_media`]'s contract): each feed's `at` is its span's
    /// first position in `ids`, every span lies whole inside the slice, and
    /// every position of a span carries the model's image token. The default
    /// is [`Seat::prefill`] when the slice carries no image, and a named
    /// refusal when it does.
    fn prefill_media(&mut self, ids: &[u32], feeds: &[MediaFeed]) -> Result<u32, GateError> {
        if feeds.is_empty() {
            return self.prefill(ids);
        }
        Err("this seat takes no image input".into())
    }
    /// The seat's image input, `None` for a seat that takes none
    /// ([`Engine::media_model`]): read once, at the load. The default is
    /// `None`.
    fn media_model(&self) -> Option<SharedMediaModel> {
        None
    }
    /// One step on `last` at `pos`, the draft fed its row; the argmax after it.
    fn step(&mut self, last: u32) -> Result<u32, GateError>;
    /// The head's logits after the last step into `row` (`n_vocab` f32).
    fn logits_into(&self, row: &mut [f32]) -> Result<(), GateError>;
    /// One step on `last` at `pos` ([`Seat::step`]) and, into `row`, the
    /// target's logits of that step ([`Seat::logits_into`]). A seat whose step
    /// runs a draft after the target's step overrides it to read the row
    /// between the two, so the row is the target's whatever the draft writes.
    fn step_row(&mut self, last: u32, row: &mut [f32]) -> Result<u32, GateError> {
        let arg = self.step(last)?;
        self.logits_into(row)
            .map_err(|e| format!("logits of the step: {e}"))?;
        Ok(arg)
    }
    /// One pass from `last` (`serve::Engine::advance`): the kept tokens into
    /// `out`, and what the draft proposed and kept. Without a draft, one step.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, GateError> {
        out.push(self.step(last)?);
        Ok(Drafted::default())
    }
    /// The most positions [`Seat::pass`] runs.
    fn pass_rows(&self) -> usize {
        1
    }
    /// One sampled pass from `last` (`serve::Engine::advance_sampled`): the
    /// draft's proposal verified, each row's id drawn by `sampler` from that
    /// row's logits given `history` and the ids this pass took before it,
    /// up to the first that is not the proposal's next; the taken ids into
    /// `out`, and what the draft proposed and kept. `history` comes back as
    /// it was given, the sampler with its state moved one draw a taken id.
    /// Asked only of a seat that [`Seat::drafts_sampled`]; the default
    /// refuses by name.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: &mut Vec<u32>,
        sampler: &mut serve::Sampler,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, GateError> {
        let _ = (last, history, sampler, out);
        Err("this seat does not draft a sampled request".into())
    }
    /// Whether the seat runs [`Seat::pass_sampled`]: it drafts and reads its
    /// verify rows' logits (`serve::Engine::drafts_sampled`). The default is
    /// false.
    fn drafts_sampled(&self) -> bool {
        false
    }
    /// The sequences this seat serves at once, each a sequence of its own
    /// the seat keeps resident (the server steps them together, one
    /// [`Seat::step`] each in one round). The default is 1: the seat serves
    /// one sequence, and several slots take it in turns (`serve::SwapEngine`).
    fn slots(&self) -> usize {
        1
    }
    /// Make `slot` (below [`Seat::slots`]) the one every later call acts on
    /// until the next select, exchanging the seat's resident sequences. The
    /// default serves slot 0 alone: selecting it is a no-op, any other slot
    /// refused by name.
    fn select(&mut self, slot: usize) -> Result<(), GateError> {
        if slot == 0 {
            Ok(())
        } else {
            Err(format!("slot {slot}: this seat serves slot 0 alone").into())
        }
    }
    /// Whether the seat keeps its draft's state per slot — parked on a
    /// [`Seat::select`], put back with the slot — so a slot's drafted passes
    /// are the passes it would run alone whatever the other slots run
    /// (`serve::Engine::slot_drafts`); the server serves several slots of a
    /// drafting seat only when it declares this. A seat that drafts nothing
    /// needs no declaration. The default is false.
    fn slot_drafts(&self) -> bool {
        false
    }
    /// Whether the seat counts its rounds of several slots — a `slots round`
    /// record a round (`record::slots_round`), and while it serves several
    /// a `slot call` record a call of one (`record::slot_call`, the engine
    /// thread's) — under `BLOOMERY_STEP_STATS`: the lever's value the
    /// binary's `main` parsed, carried by a seat whose binary names the
    /// lever among those it acts on. The default counts
    /// nothing: a binary that does not name the lever refuses it set, so its
    /// seats never count.
    fn step_stats(&self) -> bool {
        false
    }
    /// One round of several slots' steps (`serve::Engine::step_slots` through
    /// [`SeatEngine`]): each row's `last` stepped on its own slot, in row
    /// order, the answers written to `next` and each lent logits row filled.
    /// A row that fails ends the round there; the error is the server's to
    /// die on, and the rows cross back with the reply whole whatever the
    /// result, their lent rows included.
    ///
    /// The default is the fallback for a seat with no one-pass override
    /// ([`step_rows_in_turn`]: a select and a step a row). The seats that
    /// override it, or will: the qwen3moe whole-card seat (`serve_seats::qwen3`,
    /// one pass of the busy rows over the session); Qwen3.6 (round O3b, once
    /// its body serves resident slots); V4.1 (round T3 `stgseat`); GLM
    /// (line3's `glmseatpass`).
    fn step_slots(&mut self, rows: &mut [SlotStep]) -> Result<(), String> {
        step_rows_in_turn(self, rows)
    }
    /// One round of several slots' drafted passes (`serve::Engine::advance_slots`
    /// through [`SeatEngine`]): each row's `last` passed on its own slot, in
    /// row order, the kept ids into `out` and what the draft proposed and
    /// kept into `drafted`. A row that fails ends the round there; the error
    /// is the server's to die on.
    ///
    /// The default is the fallback for a seat with no one-pass override
    /// ([`pass_rows_in_turn`]: a select and a pass a row). The seats that
    /// override it, or will: the Qwen3.8 drafted seat (round O3c, through
    /// `app::mtp::pass_slots`); V4.1 (round T3 `stgseat`); GLM (line3's
    /// `glmseatpass`).
    fn pass_slots(&mut self, rows: &mut [SlotPassRow]) -> Result<(), String> {
        pass_rows_in_turn(self, rows)
    }
    /// Empty caches at position 0.
    fn reset(&mut self) -> Result<(), GateError>;
    /// Take back the positions from `pos` on; `pos` is one [`Seat::keep`]
    /// granted.
    fn rollback(&mut self, pos: u32) -> Result<(), GateError>;
    /// The longest prefix of at most `n` positions (no more than the model
    /// holds) a rollback keeps and the next request can run from, and the
    /// rule that kept less.
    fn keep(&self, n: usize) -> (usize, Option<String>);
    /// [`Seat::keep`] for a request whose reply makes at most `reply` tokens
    /// through passes (`serve::Engine::will_reply`): a seat whose kept prefix
    /// costs its draft weighs one against the other. The default is `keep`'s.
    fn keep_for(&self, n: usize, reply: Option<usize>) -> (usize, Option<String>) {
        let _ = reply;
        self.keep(n)
    }
    /// Where to cut a prompt call of `first .. end` (from where the model
    /// stands) so the `marks` stay keepable (`serve::Engine::prefill_splits`).
    fn splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize>;
    /// The sequence state as a value the server's prompt cache holds. A
    /// refusal by name — a `bloomery_gpu::GpuError` of kind `State` (the
    /// model does not hold what the call needs) or `Shape` (a call or a
    /// state it does not take), anywhere in the error's chain — is the
    /// request's: the server keeps no state and serves on. Any other error is
    /// the engine's, which ends the server.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError>;
    /// Replace the sequence state with `state`, which [`Seat::snapshot`] took
    /// of this seat. A refusal by name ([`Seat::snapshot`]'s kinds) is the
    /// request's: the server drops the state and resets the slot.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError>;
    /// The selected slot's sequence state written to `out` in a form
    /// [`Seat::restore_state`] reads back (`serve::Engine::save_state`), the
    /// state unchanged; the bytes written. A refusal is `Format`, an error
    /// of `out` is the stream's (`Io`), never the model's; only a failure
    /// that leaves the model in no defined state is `Engine`, which ends the
    /// server. The default writes nothing and refuses with `Unsupported`: the
    /// server answers 501. The seats that override it: V4.1
    /// (`serve_seats::ds41`); GLM and Qwen3.8 once their sequence state
    /// (`SeqState`) has a byte form.
    fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
        let _ = out;
        Err(StateError::Unsupported("slot save/restore"))
    }
    /// The selected slot's sequence state replaced with the one `input`
    /// carries, which runs to its end (`serve::Engine::restore_state`); the
    /// positions restored, where the seat then stands. The kinds are
    /// [`Seat::save_state`]'s: a refusal after the seat has started to read
    /// leaves the slot to the server, which resets it. The default reads
    /// nothing and refuses with `Unsupported`; the seats that override it are
    /// [`Seat::save_state`]'s.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
        let _ = input;
        Err(StateError::Unsupported("slot save/restore"))
    }
    /// `/props`' `engine` object: `p`, the opener's, with what only the load
    /// learned (a draft's device and resident bytes). The default adds
    /// nothing.
    fn props(&self, p: EngineProps) -> EngineProps {
        p
    }
    /// The residency back to its seed, the sequence where it stands, and
    /// its report (the seat prints its record); `None`, the default, for a
    /// seat with no residency.
    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, GateError> {
        Ok(None)
    }
    /// Prints what the server's prompt cache did, as the binary's records.
    fn note(note: &CacheNote)
    where
        Self: Sized;
}

/// `serve::Engine` over a [`Seat`] that lives on its own thread. The
/// position is the model's alone: every question about it goes to the thread.
pub struct SeatEngine {
    link: Link,
    worker: Option<JoinHandle<()>>,
    vocab: Arc<Vocab>,
    ctx_max: usize,
    /// [`Seat::pass_rows`].
    rows: usize,
    /// [`Seat::slots`], the opened seat's.
    slots: usize,
    /// [`Seat::slot_drafts`], the opened seat's.
    slot_drafts: bool,
    /// [`Seat::drafts_sampled`], the opened seat's.
    drafts_sampled: bool,
    /// [`Seat::media_model`], the opened seat's.
    media: Option<SharedMediaModel>,
    card: String,
    props: EngineProps,
    cache_ram: u64,
    note: fn(&CacheNote),
    /// The request's reply (`serve::Engine::will_reply`), which each keep
    /// query carries.
    reply: Option<usize>,
}

/// The handle's side of the engine thread: its channels and the logits buffer
/// a sampled step borrows. It keeps no position; the thread's replies carry it.
struct Link {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Reply>,
    /// The logits row a sampled `next` reads (`n_vocab` f32), lent to the
    /// engine thread for the call; empty while lent or when it never came back.
    logits: Vec<f32>,
    /// The logits rows a step of several slots reads, one a row of the round
    /// (`n_vocab` f32, grown once and reused every round): lent with the
    /// command, filled on the engine thread, handed back with the reply.
    scratch: Vec<Vec<f32>>,
}

impl SeatEngine {
    /// Start the engine thread, open the seat on it with `open` (which
    /// writes the load lines), and wait until it is loaded. `defined` is how
    /// many positions the body computes the model at; the engine serves the
    /// lesser of it and the cache. `card` names the device in a crash report;
    /// `props` is what `/props` reports about the engine ([`model_props`],
    /// [`placement_props`]), which the seat completes ([`Seat::props`]);
    /// `cache_ram` is the server's prompt cache budget
    /// (`serve::Engine::cache_ram`). The seat's whole-load capabilities —
    /// [`Seat::pass_rows`], [`Seat::slots`], [`Seat::slot_drafts`],
    /// [`Seat::drafts_sampled`] — cross with the load.
    pub fn spawn<S, F>(
        open: F,
        defined: usize,
        vocab: Arc<Vocab>,
        card: String,
        props: EngineProps,
        cache_ram: u64,
    ) -> Result<SeatEngine, GateError>
    where
        S: Seat,
        F: FnOnce() -> Result<S, GateError> + Send + 'static,
    {
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let (opened, loaded) = mpsc::channel::<
            Result<
                (
                    usize,
                    usize,
                    usize,
                    bool,
                    bool,
                    Option<SharedMediaModel>,
                    EngineProps,
                ),
                String,
            >,
        >();
        let n_vocab = vocab.n_vocab();
        let worker = std::thread::Builder::new()
            .name("engine".to_owned())
            .spawn(move || {
                let mut g = match open() {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = opened.send(Err(e.to_string()));
                        return;
                    }
                };
                if opened
                    .send(Ok((
                        g.ctx_max(),
                        g.pass_rows(),
                        g.slots(),
                        g.slot_drafts(),
                        g.drafts_sampled(),
                        g.media_model(),
                        g.props(props),
                    )))
                    .is_err()
                {
                    return;
                }
                // The slot the server's last select named: slot 0 from the
                // start, none after a round of several, which leaves the
                // seat's selection unspecified until the server selects again
                // (`serve::Engine::step_slots`).
                let mut selected = Some(0);
                for cmd in cmds {
                    let call = one_slot_call(&cmd);
                    let select = match &cmd {
                        Cmd::Select(slot) => Some(*slot),
                        _ => None,
                    };
                    let round = matches!(cmd, Cmd::StepSlots(_) | Cmd::PassSlots(_));
                    let (mut result, logits, extra) = match cmd {
                        Cmd::Keep(n, reply) => {
                            let (k, why) = g.keep_for(n.min(g.pos()), reply);
                            (
                                u32::try_from(k).map_err(|_| {
                                    format!("a kept prefix of at most {n} passes u32")
                                }),
                                None,
                                Extra::Why(why),
                            )
                        }
                        Cmd::Splits { first, end, marks } => {
                            (Ok(0), None, Extra::Splits(g.splits(first, end, &marks)))
                        }
                        Cmd::Pass { last, mut out } => {
                            let at = g.pos();
                            out.clear();
                            match pass(&mut g, last, &mut out) {
                                Ok(d) => (Ok(0), None, Extra::Pass(out, d)),
                                Err(e) => (
                                    Err(format!("pass at position {at}: {e}")),
                                    None,
                                    Extra::Pass(out, Drafted::default()),
                                ),
                            }
                        }
                        Cmd::PassSampled {
                            last,
                            mut history,
                            mut sampler,
                            mut out,
                        } => {
                            let at = g.pos();
                            out.clear();
                            let r =
                                pass_sampled(&mut g, last, &mut history, &mut sampler, &mut out);
                            let (result, drafted) = match r {
                                Ok(d) => (Ok(0), d),
                                Err(e) => (
                                    Err(format!("sampled pass at position {at}: {e}")),
                                    Drafted::default(),
                                ),
                            };
                            let extra = Extra::PassSampled {
                                out,
                                drafted,
                                history,
                                sampler,
                            };
                            (result, None, extra)
                        }
                        Cmd::Save => (Ok(0), None, Extra::Saved(snapshot_of(&mut g))),
                        Cmd::ResidencyReset => match g.residency_reset() {
                            Ok(r) => (Ok(0), None, Extra::Residency(r)),
                            Err(e) => (
                                Err(format!("residency reset at position {}: {e}", g.pos())),
                                None,
                                Extra::None,
                            ),
                        },
                        Cmd::Resume(state) => {
                            (Ok(0), None, Extra::Resumed(resume_of(&mut g, &*state)))
                        }
                        Cmd::SaveState(chunks) => {
                            (Ok(0), None, Extra::State(save_into(&mut g, chunks)))
                        }
                        Cmd::RestoreState(chunks) => {
                            (Ok(0), None, Extra::State(restore_from(&mut g, chunks)))
                        }
                        Cmd::StepSlots(mut rows) => {
                            // The seat's round ([`Seat::step_slots`]): one
                            // command on the thread, its rows back with the
                            // one reply — not two round trips a row.
                            let r = g.step_slots(&mut rows).map(|()| 0);
                            (r, None, Extra::StepSlots(rows))
                        }
                        Cmd::PassSlots(mut rows) => {
                            // The seat's round ([`Seat::pass_slots`]): one
                            // command on the thread, its rows back with the
                            // one reply — not two round trips a row.
                            let r = g.pass_slots(&mut rows).map(|()| 0);
                            (r, None, Extra::PassSlots(rows))
                        }
                        cmd => {
                            let (result, logits) = serve_cmd(&mut g, cmd, n_vocab);
                            (result, logits, Extra::None)
                        }
                    };
                    if let Some(slot) = select {
                        selected = result.is_ok().then_some(slot);
                    }
                    if round {
                        selected = None;
                    }
                    if let Some(call) = call
                        && result.is_ok()
                        && let Err(e) = count_call(&g, call, selected)
                    {
                        result = Err(e);
                    }
                    if replies
                        .send(Reply {
                            result,
                            logits,
                            pos: g.pos(),
                            extra,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })?;
        let (ctx_max, rows, slots, slot_drafts, drafts_sampled, media, props) = match loaded.recv()
        {
            Ok(Ok((c, rows, n, d, ds, m, props))) => (c.min(defined), rows, n, d, ds, m, props),
            Ok(Err(e)) => return Err(e.into()),
            Err(mpsc::RecvError) => return Err("the engine thread ended during the load".into()),
        };
        Ok(SeatEngine {
            link: Link {
                tx: Some(tx),
                rx,
                logits: vec![0.0; n_vocab],
                scratch: Vec::new(),
            },
            worker: Some(worker),
            vocab,
            ctx_max,
            rows,
            slots,
            slot_drafts,
            drafts_sampled,
            media,
            card,
            props,
            cache_ram,
            note: S::note,
            reply: None,
        })
    }

    /// The sequences the engine serves at once, the opened seat's answer
    /// ([`Seat::slots`]): one until the seat made resident slots.
    #[must_use]
    pub fn slots(&self) -> usize {
        self.slots
    }
}

impl Link {
    /// One command; a lent logits buffer comes back to the link.
    fn call(&mut self, cmd: Cmd) -> Result<u32, EngineError> {
        let reply = self.ask(cmd)?;
        if let Some(row) = reply.logits {
            self.logits = row;
        }
        reply.result.map_err(EngineError)
    }

    /// One command and its reply, the position left to the caller.
    fn ask(&self, cmd: Cmd) -> Result<Reply, EngineError> {
        self.send(cmd)?;
        self.reply()
    }

    /// One command sent; [`Link::reply`] waits for its answer.
    fn send(&self, cmd: Cmd) -> Result<(), EngineError> {
        match self.tx.as_ref().map(|tx| tx.send(cmd)) {
            Some(Ok(())) => Ok(()),
            _ => Err(EngineError("the engine thread is gone".to_owned())),
        }
    }

    /// The answer to the command sent last.
    fn reply(&self) -> Result<Reply, EngineError> {
        self.rx
            .recv()
            .map_err(|_| EngineError("the engine thread ended mid-call".to_owned()))
    }

    /// The selected slot's state as a value ([`Seat::snapshot`]), its
    /// failure's kind as [`snapshot_of`] read it.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        let reply = self.ask(Cmd::Save)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Saved(s)) => s,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a save with no state".to_owned(),
            ))),
        }
    }

    /// `state` back into the selected slot ([`Seat::resume`]), its failure's
    /// kind as [`resume_of`] read it.
    fn resume(&self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        let reply = self.ask(Cmd::Resume(Arc::clone(state)))?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Resumed(r)) => r,
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a resume with no outcome".to_owned(),
            ))),
        }
    }

    /// The selected slot's state ([`Seat::save_state`]) into `out` through
    /// the pipe: the engine thread writes the chunks, this thread writes
    /// each to `out` as it comes. An error of `out` closes the pipe — the
    /// seat's next write fails — and is the answer ([`settle`]).
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        let (tx, chunks) = mpsc::sync_channel(PIPE_DEPTH);
        self.send(Cmd::SaveState(tx))?;
        let written = drain(chunks, out);
        let saved = state_of(self.reply()?, "save");
        settle(saved, written)
    }

    /// The selected slot's state replaced with the one `input` carries
    /// ([`Seat::restore_state`]), through the pipe: this thread reads `input`
    /// to its end a chunk at a time, the engine thread reads the chunks. The
    /// bytes the answer counts are the ones the seat read; the server holds
    /// them to the file's.
    fn restore_state(&self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        let (chunks, rx) = mpsc::sync_channel(PIPE_DEPTH);
        self.send(Cmd::RestoreState(rx))?;
        let read = pump(input, chunks);
        let restored = state_of(self.reply()?, "restore");
        settle(restored, read)
    }

    /// A step; with `logits_out`, the engine's buffer is lent for it and
    /// copied out once it is back.
    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let lent = logits_out
            .is_some()
            .then(|| std::mem::take(&mut self.logits));
        let arg = self.call(Cmd::Next { last, logits: lent })?;
        if let Some(out) = logits_out {
            if out.len() != self.logits.len() {
                return Err(EngineError(format!(
                    "the caller's logits buffer holds {}, the engine's {}",
                    out.len(),
                    self.logits.len()
                )));
            }
            out.copy_from_slice(&self.logits);
        }
        Ok(arg)
    }

    /// A pass into the caller's `out`, which crosses to the engine thread and
    /// back; it holds the kept tokens on success.
    fn pass(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        let reply = self.ask(Cmd::Pass {
            last,
            out: std::mem::take(out),
        })?;
        match reply.extra {
            Extra::Pass(kept, d) => {
                *out = kept;
                reply.result.map(|_| d).map_err(EngineError)
            }
            _ => Err(EngineError(
                "the engine thread answered a pass with no tokens".to_owned(),
            )),
        }
    }

    /// A sampled pass into the caller's `out` ([`Cmd::PassSampled`]): the
    /// history and the sampler cross to the engine thread and come back with
    /// the reply, or with the command when the thread is gone before it took
    /// it. Only a thread that ended mid-call keeps the sampler: a stand-in
    /// that refuses by name comes back beside the error.
    fn pass_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, serve::Sampler) {
        let gone = |why: &str| EngineError(why.to_owned());
        let cmd = Cmd::PassSampled {
            last,
            history,
            sampler,
            out: std::mem::take(out),
        };
        let sent = match self.tx.as_ref() {
            Some(tx) => tx.send(cmd).map_err(|mpsc::SendError(cmd)| cmd),
            None => Err(cmd),
        };
        if let Err(cmd) = sent {
            let Cmd::PassSampled {
                history,
                sampler,
                out: back,
                ..
            } = cmd
            else {
                unreachable!("the command that did not cross is the sampled pass built here")
            };
            *out = back;
            return (Err(gone("the engine thread is gone")), history, sampler);
        }
        let reply = match self.reply() {
            Ok(r) => r,
            Err(e) => return (Err(e), Vec::new(), lost_sampler()),
        };
        match reply.extra {
            Extra::PassSampled {
                out: taken,
                drafted,
                history,
                sampler,
            } => {
                *out = taken;
                (
                    reply.result.map(|_| drafted).map_err(EngineError),
                    history,
                    sampler,
                )
            }
            _ => (
                Err(gone(
                    "the engine thread answered a sampled pass with no ids",
                )),
                Vec::new(),
                lost_sampler(),
            ),
        }
    }

    /// A step of several slots in one command
    /// ([`SeatEngine`]'s `step_slots`): the engine thread selects and steps
    /// each row in order on the seat, and the answers and lent rows come
    /// back with the one reply — not two round trips a row. The rows' logits
    /// travel as the link's own scratch, one row a round, grown once and
    /// reused: no token allocates a row.
    fn step_slots(&mut self, rows: &mut [serve::SlotRow<'_>]) -> Result<(), EngineError> {
        let n_vocab = self.logits.len();
        // Two rows of one slot would step its sequence twice in the round,
        // the second on the first's answer: named before anything crosses.
        if let Some((i, r)) = rows
            .iter()
            .enumerate()
            .find(|(i, r)| rows[..*i].iter().any(|p| p.slot == r.slot))
        {
            return Err(EngineError(format!(
                "a round of {} rows names slot {} twice (row {i} again)",
                rows.len(),
                r.slot
            )));
        }
        // Grown, never shrunk: a shorter round keeps the rows it grew.
        if self.scratch.len() < rows.len() {
            self.scratch.resize_with(rows.len(), Vec::new);
        }
        let mut steps = Vec::with_capacity(rows.len());
        let mut refused = None;
        for (i, row) in rows.iter().enumerate() {
            // A row that wants the logits row borrows the link's scratch for
            // it; a caller buffer of another length is a named error before
            // anything crosses.
            let lent = row.logits.as_ref().map_or(Ok(None), |b| {
                if b.len() != n_vocab {
                    Err(EngineError(format!(
                        "a row's logits buffer holds {}, the vocabulary has {n_vocab}",
                        b.len()
                    )))
                } else {
                    let mut v = std::mem::take(&mut self.scratch[i]);
                    v.resize(n_vocab, 0.0);
                    Ok(Some(v))
                }
            });
            match lent {
                Ok(logits) => steps.push(SlotStep {
                    slot: row.slot,
                    last: row.last,
                    logits,
                    next: 0,
                }),
                Err(e) => {
                    refused = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = refused {
            self.take_back(&mut steps);
            return Err(e);
        }
        let reply = self.ask(Cmd::StepSlots(steps))?;
        let Extra::StepSlots(mut back) = reply.extra else {
            return Err(EngineError(
                "the engine thread answered a step of several slots with no rows".to_owned(),
            ));
        };
        for (i, (row, step)) in rows.iter_mut().zip(back.iter_mut()).enumerate() {
            row.next = step.next;
            if let (Some(v), Some(out)) = (step.logits.as_ref(), row.logits.as_deref_mut()) {
                out.copy_from_slice(v);
            }
            self.scratch[i] = step.logits.take().unwrap_or_default();
        }
        reply.result.map(|_| ()).map_err(EngineError)
    }

    /// The scratch rows `steps` hold back into the link, after a call that
    /// did not cross (a refused buffer length): what was taken out goes home.
    fn take_back(&mut self, steps: &mut [SlotStep]) {
        for (i, s) in steps.iter_mut().enumerate() {
            if s.logits.is_some() {
                self.scratch[i] = s.logits.take().unwrap_or_default();
            }
        }
    }

    /// One command for a round of several drafted passes
    /// ([`SeatEngine`]'s `advance_slots`): the engine thread selects and
    /// passes each row in order, on the seat, and each row's kept ids and
    /// drafted counts come back with the one reply — not a select and a pass
    /// round trip a row. The rows' kept-id buffers cross to the thread and
    /// back as the command's own rows.
    fn pass_slots(&mut self, rows: &mut [serve::SlotPass<'_>]) -> Result<(), EngineError> {
        // Two rows of one slot would pass its sequence twice in the round,
        // the second on the first's answer: named before anything crosses.
        if let Some((i, r)) = rows
            .iter()
            .enumerate()
            .find(|(i, r)| rows[..*i].iter().any(|p| p.slot == r.slot))
        {
            return Err(EngineError(format!(
                "a round of {} rows names slot {} twice (row {i} again)",
                rows.len(),
                r.slot
            )));
        }
        let mut sent = Vec::with_capacity(rows.len());
        for r in rows.iter_mut() {
            sent.push(SlotPassRow {
                slot: r.slot,
                last: r.last,
                out: std::mem::take(r.out),
                drafted: Drafted::default(),
            });
        }
        let reply = self.ask(Cmd::PassSlots(sent))?;
        let Extra::PassSlots(mut back) = reply.extra else {
            return Err(EngineError(
                "the engine thread answered a pass of several slots with no rows".to_owned(),
            ));
        };
        for (row, back) in rows.iter_mut().zip(back.iter_mut()) {
            row.drafted = back.drafted;
            *row.out = std::mem::take(&mut back.out);
        }
        reply.result.map(|_| ()).map_err(EngineError)
    }
}

/// Whether a feed of `n` ids may start at `g`'s position: at least one, and
/// all inside the context.
fn check_feed<S: Seat>(g: &S, n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("prefill: no ids to feed".into());
    }
    if g.pos() + n > g.ctx_max() {
        return Err(format!(
            "prefill: position {} + {n} ids exceed the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    Ok(())
}

/// A pass on `g`, refused before it runs when its rows do not fit the
/// context; one that keeps no token or more than its rows is the seat's
/// defect, named.
fn pass<S: Seat + ?Sized>(g: &mut S, last: u32, out: &mut Vec<u32>) -> Result<Drafted, String> {
    let rows = g.pass_rows();
    if g.pos() + rows > g.ctx_max() {
        return Err(format!(
            "a pass of {rows} rows from position {} passes the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    let d = g.pass(last, out).map_err(|e| e.to_string())?;
    if out.is_empty() || out.len() > rows {
        return Err(format!("a pass of {rows} rows kept {} tokens", out.len()));
    }
    Ok(d)
}

/// A sampled pass on `g` ([`Seat::pass_sampled`]), refused before it runs as
/// [`pass`] refuses one; a seat that keeps no id or more than its rows, or
/// hands `history` back changed, is the seat's defect, named.
fn pass_sampled<S: Seat + ?Sized>(
    g: &mut S,
    last: u32,
    history: &mut Vec<u32>,
    sampler: &mut serve::Sampler,
    out: &mut Vec<u32>,
) -> Result<Drafted, String> {
    let rows = g.pass_rows();
    if g.pos() + rows > g.ctx_max() {
        return Err(format!(
            "a pass of {rows} rows from position {} passes the context {}",
            g.pos(),
            g.ctx_max()
        ));
    }
    let given = history.len();
    let d = g
        .pass_sampled(last, history, sampler, out)
        .map_err(|e| e.to_string())?;
    if out.is_empty() || out.len() > rows {
        return Err(format!("a pass of {rows} rows kept {} tokens", out.len()));
    }
    if history.len() != given {
        return Err(format!(
            "a sampled pass handed back a history of {} ids, given {given}",
            history.len()
        ));
    }
    Ok(d)
}

/// The sampler a [`Link::pass_sampled`] hands back when the engine thread
/// ended holding the request's own: it refuses by name if anything draws
/// from it.
fn lost_sampler() -> serve::Sampler {
    Box::new(|_: &[f32], _: &[u32]| -> u32 {
        panic!("this request's sampler ended with the engine thread")
    })
}

/// One command on the engine thread, and the logits buffer a `Next` was lent,
/// handed back whatever the result.
fn serve_cmd<S: Seat>(
    g: &mut S,
    cmd: Cmd,
    n_vocab: usize,
) -> (Result<u32, String>, Option<Vec<f32>>) {
    let at = g.pos();
    let result = match cmd {
        Cmd::Prefill(ids) => {
            let refuse =
                |e: String| format!("prefill of {} ids from position {at}: {e}", ids.len());
            check_feed(g, ids.len())
                .map_err(refuse)
                .and_then(|()| g.prefill(&ids).map_err(|e| refuse(e.to_string())))
        }
        Cmd::PrefillMedia(ids, feeds) => {
            let refuse = |e: String| {
                format!(
                    "prefill of {} ids with {} image(s) from position {at}: {e}",
                    ids.len(),
                    feeds.len()
                )
            };
            check_feed(g, ids.len()).map_err(refuse).and_then(|()| {
                g.prefill_media(&ids, &feeds)
                    .map_err(|e| refuse(e.to_string()))
            })
        }
        Cmd::Next { last, mut logits } => {
            let arg = next_row(g, last, logits.as_deref_mut(), n_vocab);
            return (arg, logits);
        }
        Cmd::Reset => g
            .reset()
            .map(|()| 0)
            .map_err(|e| format!("reset at position {at}: {e}")),
        Cmd::Select(slot) => g
            .select(slot)
            .map(|()| 0)
            .map_err(|e| format!("select of slot {slot}: {e}")),
        Cmd::Rollback(pos) if pos as usize == at => Ok(0),
        Cmd::Rollback(pos) => g
            .rollback(pos)
            .map(|()| 0)
            .map_err(|e| format!("rollback to position {pos} from {at}: {e}")),
        Cmd::Keep(..)
        | Cmd::Splits { .. }
        | Cmd::Save
        | Cmd::Resume(_)
        | Cmd::SaveState(_)
        | Cmd::RestoreState(_)
        | Cmd::Pass { .. }
        | Cmd::PassSampled { .. }
        | Cmd::StepSlots(..)
        | Cmd::PassSlots(..)
        | Cmd::ResidencyReset => Err(
            "a keep, split, pass, step or pass of several slots, save, resume, slot file's save \
             or restore, or residency reset reached the step loop; the engine thread answers it"
                .to_owned(),
        ),
        Cmd::Pos => Ok(0),
    };
    (result, None)
}

/// One step and, into `row`, the target's logits of it ([`Seat::step_row`]).
/// The row is checked here: one of the wrong length is refused before the
/// step runs, and one holding a NaN is the step's error, not the sampler's to
/// absorb.
fn next_row<S: Seat>(
    g: &mut S,
    last: u32,
    row: Option<&mut [f32]>,
    n_vocab: usize,
) -> Result<u32, String> {
    if let Some(row) = row.as_ref()
        && row.len() != n_vocab
    {
        let at = g.pos();
        return Err(format!(
            "a logits buffer of {} at position {at}, the vocabulary has {n_vocab}",
            row.len()
        ));
    }
    next_lent_row(g, last, row)
}

/// [`next_row`] over a row the link already checked against the vocabulary
/// and lent as its own scratch, so the length guard there cannot fire: the
/// ctx-full and NaN guards stay, as a round's rows need them too
/// ([`step_rows_in_turn`]).
fn next_lent_row<S: Seat + ?Sized>(
    g: &mut S,
    last: u32,
    row: Option<&mut [f32]>,
) -> Result<u32, String> {
    let at = g.pos();
    if at >= g.ctx_max() {
        return Err(format!(
            "step at position {at}: step: the context {} is full",
            g.ctx_max()
        ));
    }
    let Some(row) = row else {
        return g
            .step(last)
            .map_err(|e| format!("step at position {at}: {e}"));
    };
    let arg = g
        .step_row(last, row)
        .map_err(|e| format!("step at position {at}: {e}"))?;
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return Err(format!("logit {i} is NaN after position {at}"));
    }
    Ok(arg)
}

/// The fallback round of several slots' steps ([`Seat::step_slots`]'s
/// default): a select and a step a row, in row order — one command, not two
/// round trips a row. A row that fails ends the round there; the error is
/// the server's to die on. A seat that counts its rounds
/// ([`Seat::step_stats`]) prints the `slots round` record: the fallback's
/// passes are its rows.
pub fn step_rows_in_turn<S: Seat + ?Sized>(g: &mut S, rows: &mut [SlotStep]) -> Result<(), String> {
    let mut failed = None;
    for r in rows.iter_mut() {
        let run = g
            .select(r.slot)
            .map_err(|e| e.to_string())
            .and_then(|()| next_lent_row(g, r.last, r.logits.as_deref_mut()));
        match run {
            Ok(next) => r.next = next,
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    if failed.is_none() && g.step_stats() {
        crate::record::slots_round("step", rows.len(), rows.len(), g.slots()).eprint();
    }
    failed.map_or(Ok(()), Err)
}

/// The fallback round of several slots' drafted passes ([`Seat::pass_slots`]'s
/// default): a select and a pass a row, in row order. A row that fails ends
/// the round there; the error is the server's to die on. A seat that counts
/// its rounds ([`Seat::step_stats`]) prints the `slots round` record: the
/// fallback's passes are its rows.
pub fn pass_rows_in_turn<S: Seat + ?Sized>(
    g: &mut S,
    rows: &mut [SlotPassRow],
) -> Result<(), String> {
    let mut failed = None;
    for r in rows.iter_mut() {
        let at = g.pos();
        let run = g.select(r.slot).map_err(|e| e.to_string()).and_then(|()| {
            r.out.clear();
            pass(g, r.last, &mut r.out).map_err(|e| format!("pass at position {at}: {e}"))
        });
        match run {
            Ok(d) => r.drafted = d,
            Err(e) => {
                failed = Some(e);
                break;
            }
        }
    }
    if failed.is_none() && g.step_stats() {
        crate::record::slots_round("pass", rows.len(), rows.len(), g.slots()).eprint();
    }
    failed.map_or(Ok(()), Err)
}

/// The server's call of one slot `cmd` is, as its `slot call` record names
/// it ([`crate::record::slot_call`]): `step` its `next`, `pass` its
/// `advance`, `sampled` its `advance_sampled`; `None` for every other
/// command.
fn one_slot_call(cmd: &Cmd) -> Option<&'static str> {
    match cmd {
        Cmd::Next { .. } => Some("step"),
        Cmd::Pass { .. } => Some("pass"),
        Cmd::PassSampled { .. } => Some("sampled"),
        _ => None,
    }
}

/// A seat that counts its rounds ([`Seat::step_stats`]) and serves several
/// slots prints the `slot call` record of each call of one slot it ran,
/// `call` ([`one_slot_call`]), on `selected`, the slot the server's last
/// select named: beside the `slots round` record of a round of several, the
/// log shows how each of the server's rows ran. A call with no slot named
/// since a round of several breaks the server's select contract
/// (`serve::Engine::step_slots`) and is refused by name: the record would
/// guess its slot.
fn count_call<S: Seat + ?Sized>(g: &S, call: &str, selected: Option<usize>) -> Result<(), String> {
    if !g.step_stats() || g.slots() < 2 {
        return Ok(());
    }
    let slot = selected.ok_or_else(|| {
        format!(
            "a {call} of one slot after a round of several slots with no select since: the \
             server selects before its next call of one slot"
        )
    })?;
    crate::record::slot_call(call, slot, g.slots()).eprint();
    Ok(())
}

/// The selected slot's state as a value ([`Seat::snapshot`]), on the engine
/// thread; a failure with its kind ([`seat_failure`]).
fn snapshot_of<S: Seat + ?Sized>(g: &mut S) -> Result<Arc<dyn Saved>, StateError> {
    let at = g.pos();
    g.snapshot()
        .map_err(|e| seat_failure(&*e, format!("snapshot at position {at}")))
}

/// `state` back into the selected slot ([`Seat::resume`]), on the engine
/// thread; a failure with its kind ([`seat_failure`]).
fn resume_of<S: Seat + ?Sized>(g: &mut S, state: &dyn Saved) -> Result<(), StateError> {
    g.resume(state).map_err(|e| {
        seat_failure(
            &*e,
            format!("resume of a state of {} positions", state.n_tokens()),
        )
    })
}

/// A seat's failure of a prompt-cache state with `what` before it, as the
/// server's kind: a refusal by name (a `GpuError` of kind `State` or
/// `Shape` anywhere in the chain, [`Seat::snapshot`]) is `Format`, the
/// request's, which the server answers by keeping no state or by resetting
/// the slot; any other failure — the card's, the driver's, a fault, an
/// error of no known kind — is `Engine`, which ends it.
fn seat_failure(e: &(dyn std::error::Error + 'static), what: String) -> StateError {
    let refused = std::iter::successors(Some(e), |e| e.source())
        .find_map(|e| e.downcast_ref::<bloomery_gpu::GpuError>())
        .is_some_and(|g| {
            matches!(
                g,
                bloomery_gpu::GpuError::State { .. } | bloomery_gpu::GpuError::Shape { .. }
            )
        });
    if refused {
        StateError::Format(format!("{what}: {e}"))
    } else {
        StateError::Engine(EngineError(format!("{what}: {e}")))
    }
}

/// The selected slot's state ([`Seat::save_state`]) written into the pipe
/// `chunks`, on the engine thread: its last chunk sent and its end dropped
/// before the reply, so the server's thread holds every byte when the
/// answer reaches it. The positions the seat stands at and the bytes it
/// wrote.
fn save_into<S: Seat + ?Sized>(
    g: &mut S,
    chunks: SyncSender<Vec<u8>>,
) -> Result<SavedState, StateError> {
    let at = g.pos();
    let mut w = PipeWriter {
        chunks,
        chunk: Vec::new(),
    };
    let n_bytes = g
        .save_state(&mut w)
        .map_err(|e| at_position(e, "save", at))?;
    w.flush()?;
    Ok(SavedState {
        n_tokens: at,
        n_bytes,
    })
}

/// The selected slot's state replaced from the pipe `chunks`
/// ([`Seat::restore_state`]), on the engine thread, the pipe's end dropped
/// before the reply: the positions restored and the bytes the seat read.
fn restore_from<S: Seat + ?Sized>(
    g: &mut S,
    chunks: Receiver<Vec<u8>>,
) -> Result<SavedState, StateError> {
    let at = g.pos();
    let mut r = PipeReader {
        chunks,
        chunk: Vec::new(),
        at: 0,
        bytes: 0,
    };
    let n_tokens = g
        .restore_state(&mut r)
        .map_err(|e| at_position(e, "restore", at))?;
    Ok(SavedState {
        n_tokens,
        n_bytes: r.bytes,
    })
}

/// `e` with the command and the position the seat stood at when it is the
/// engine's; a refusal or a stream's error is the request's answer as the
/// seat gave it.
fn at_position(e: StateError, what: &str, at: usize) -> StateError {
    match e {
        StateError::Engine(EngineError(m)) => {
            StateError::Engine(EngineError(format!("{what} at position {at}: {m}")))
        }
        other => other,
    }
}

/// A `SaveState` or `RestoreState` reply's outcome, its kind as the seat
/// gave it; a reply without one is the engine thread's defect, named.
fn state_of(reply: Reply, what: &str) -> Result<SavedState, StateError> {
    match (reply.result, reply.extra) {
        (Ok(_), Extra::State(r)) => r,
        (Err(e), _) => Err(StateError::Engine(EngineError(e))),
        (Ok(_), _) => Err(StateError::Engine(EngineError(format!(
            "the engine thread answered a slot file's {what} with no outcome"
        )))),
    }
}

/// The answer of a slot file's state that crossed the pipe: the seat's
/// engine failure first (a failed model is never hidden behind the file),
/// then this thread's error of the stream (which closed the pipe, and so
/// caused whatever the seat answered after it), then the seat's answer.
fn settle(
    seat: Result<SavedState, StateError>,
    stream: io::Result<()>,
) -> Result<SavedState, StateError> {
    match (seat, stream) {
        (Err(e @ StateError::Engine(_)), _) => Err(e),
        (_, Err(e)) => Err(StateError::Io(e)),
        (r, Ok(())) => r,
    }
}

/// Every chunk of a save's pipe into `out`, until the engine thread drops
/// its end; the first error of `out` drops this end (the seat's next write
/// fails) and is returned.
fn drain(chunks: Receiver<Vec<u8>>, out: &mut dyn Write) -> io::Result<()> {
    for chunk in &chunks {
        out.write_all(&chunk)?;
    }
    Ok(())
}

/// `input` to its end into a restore's pipe, [`PIPE_CHUNK`] bytes a chunk,
/// this end dropped after it: the end of `input` reaches the seat as the
/// pipe's. A seat that stops reading drops its end, and the pump stops
/// there. An error of `input` is returned.
fn pump(input: &mut dyn Read, chunks: SyncSender<Vec<u8>>) -> io::Result<()> {
    loop {
        let mut chunk = Vec::with_capacity(PIPE_CHUNK);
        if (&mut *input)
            .take(PIPE_CHUNK as u64)
            .read_to_end(&mut chunk)?
            == 0
        {
            return Ok(());
        }
        if chunks.send(chunk).is_err() {
            return Ok(());
        }
    }
}

/// The engine thread's side of a save's pipe: bytes gathered into a chunk
/// of [`PIPE_CHUNK`], each full chunk sent, the last one by `flush`. A pipe
/// the server's thread closed is `BrokenPipe`.
struct PipeWriter {
    chunks: SyncSender<Vec<u8>>,
    chunk: Vec<u8>,
}

impl PipeWriter {
    fn send(&mut self) -> io::Result<()> {
        let chunk = std::mem::take(&mut self.chunk);
        self.chunks.send(chunk).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "the server stopped taking the slot's state",
            )
        })
    }
}

impl Write for PipeWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.chunk.capacity() == 0 {
            self.chunk.reserve_exact(PIPE_CHUNK);
        }
        let n = buf.len().min(PIPE_CHUNK - self.chunk.len());
        self.chunk.extend_from_slice(&buf[..n]);
        if self.chunk.len() == PIPE_CHUNK {
            self.send()?;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.chunk.is_empty() {
            Ok(())
        } else {
            self.send()
        }
    }
}

/// The engine thread's side of a restore's pipe: the chunks the server's
/// thread read, in order, and the pipe's end as the stream's. `bytes`
/// counts what the seat read.
struct PipeReader {
    chunks: Receiver<Vec<u8>>,
    chunk: Vec<u8>,
    at: usize,
    bytes: u64,
}

impl Read for PipeReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        while self.at == self.chunk.len() {
            match self.chunks.recv() {
                Ok(c) => {
                    self.chunk = c;
                    self.at = 0;
                }
                Err(mpsc::RecvError) => return Ok(0),
            }
        }
        let n = buf.len().min(self.chunk.len() - self.at);
        buf[..n].copy_from_slice(&self.chunk[self.at..self.at + n]);
        self.at += n;
        self.bytes += n as u64;
        Ok(n)
    }
}

impl Engine for SeatEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.vocab.clone()
    }

    /// The seat's ([`Seat::media_model`]), read once at the load: a shared
    /// handle, the same one every request thread prepares images on.
    fn media_model(&self) -> Option<SharedMediaModel> {
        self.media.clone()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.link.call(Cmd::Prefill(ids.to_vec())).map(|_| ())
    }

    /// The seat's ([`Seat::prefill_media`]) as one command on the engine
    /// thread; an empty feed takes [`Engine::prefill`]'s plain command.
    fn prefill_media(&mut self, ids: &[u32], media: &[MediaFeed]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        if media.is_empty() {
            return self.prefill(ids);
        }
        self.link
            .call(Cmd::PrefillMedia(ids.to_vec(), media.to_vec()))
            .map(|_| ())
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        self.link.next(last, logits_out)
    }

    fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
        self.link.pass(last, out)
    }

    fn advance_rows(&self) -> usize {
        self.rows
    }

    /// One command ([`Cmd::PassSampled`]): the seat's sampled pass on the
    /// engine thread, the history and sampler back with the reply.
    fn advance_sampled(
        &mut self,
        last: u32,
        history: Vec<u32>,
        sampler: serve::Sampler,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Vec<u32>, serve::Sampler) {
        self.link.pass_sampled(last, history, sampler, out)
    }

    /// The seat's declaration ([`Seat::drafts_sampled`]), read once at the
    /// spawn.
    fn drafts_sampled(&self) -> bool {
        self.drafts_sampled
    }

    /// The seat's slots ([`Seat::slots`]), read once at the spawn: the
    /// server serves this many sequences at once and steps them together.
    fn slots(&self) -> usize {
        self.slots
    }

    /// The seat's slot ([`Seat::select`]), forwarded to the thread: the
    /// exchange happens there, with every other call the seat runs.
    fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
        self.link.call(Cmd::Select(slot)).map(|_| ())
    }

    /// One command for a round of several rows ([`serve::Engine`'s
    /// `step_slots`]): the thread selects and steps each row in order, on
    /// the seat, and the answers and requested logits rows come back with
    /// the one reply — not a select and a step round trip a row.
    fn step_slots(&mut self, rows: &mut [serve::SlotRow<'_>]) -> Result<(), EngineError> {
        self.link.step_slots(rows)
    }

    /// One command for a round of several drafted passes
    /// ([`serve::Engine`]'s `advance_slots`): the thread selects and passes
    /// each row in order, on the seat, and each row's kept ids and drafted
    /// counts come back with the one reply — not a select and a pass round
    /// trip a row.
    fn advance_slots(&mut self, rows: &mut [serve::SlotPass<'_>]) -> Result<(), EngineError> {
        self.link.pass_slots(rows)
    }

    /// The seat's declaration ([`Seat::slot_drafts`]), read once at the
    /// spawn: the server serves several slots of a drafting engine only when
    /// it holds.
    fn slot_drafts(&self) -> bool {
        self.slot_drafts
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.link.call(Cmd::Reset).map(|_| ())
    }

    fn residency_reset(&mut self) -> Result<Option<ResidencyReset>, EngineError> {
        let reply = self.link.ask(Cmd::ResidencyReset)?;
        reply.result.map_err(EngineError)?;
        match reply.extra {
            Extra::Residency(r) => Ok(r),
            _ => Err(EngineError(
                "the engine thread answered a residency reset with no report".to_owned(),
            )),
        }
    }

    /// The body's rule for the request's reply, asked on the engine thread
    /// (`spawn`'s `keep_for`). A
    /// thread that does not answer grants nothing: the caller resets, and the
    /// next call reports the thread.
    fn will_reply(&mut self, tokens: Option<usize>) {
        self.reply = tokens;
    }

    fn keepable(&self, n: usize) -> usize {
        match self.link.ask(Cmd::Keep(n, self.reply)).map(|r| r.result) {
            Ok(Ok(k)) => (k as usize).min(n),
            _ => 0,
        }
    }

    /// The body's rule, as `keepable` asks it; a thread that does not answer
    /// is named as the reason.
    fn keep_limit(&self, n: usize) -> Option<String> {
        match self.link.ask(Cmd::Keep(n, self.reply)) {
            Ok(Reply {
                extra: Extra::Why(why),
                ..
            }) => why,
            Ok(_) => Some("the engine thread answered a keep query with no rule".to_owned()),
            Err(e) => Some(e.0),
        }
    }

    fn cache_ram(&self) -> u64 {
        self.cache_ram
    }

    /// The seat's state ([`Link::snapshot`]): a refusal by name is the
    /// request's (the server keeps no state), any other failure fatal.
    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        self.link.snapshot()
    }

    /// The state back into the seat ([`Link::resume`]): a refusal by name
    /// is the request's (the server drops the state and resets the slot),
    /// any other failure fatal.
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        self.link.resume(state)
    }

    /// The selected slot's state through the pipe ([`Seat::save_state`]),
    /// its kind carried: `Unsupported` is the server's 501, a refusal or
    /// the file's error the request's, an engine failure fatal.
    fn save_state(&self, out: &mut dyn Write) -> Result<SavedState, StateError> {
        self.link.save_state(out)
    }

    /// [`Engine::save_state`]'s way back ([`Seat::restore_state`]); a
    /// refusal leaves the slot to the server, which resets it.
    fn restore_state(&mut self, input: &mut dyn Read) -> Result<SavedState, StateError> {
        self.link.restore_state(input)
    }

    /// The body's cuts; a thread that does not answer cuts nowhere, and the
    /// next call reports the thread.
    fn prefill_splits(&self, first: usize, end: usize, marks: &[usize]) -> Vec<usize> {
        let cmd = Cmd::Splits {
            first,
            end,
            marks: marks.to_vec(),
        };
        match self.link.ask(cmd) {
            Ok(Reply {
                extra: Extra::Splits(at),
                ..
            }) => at,
            _ => Vec::new(),
        }
    }

    fn note(&self, note: &CacheNote) {
        (self.note)(note);
    }

    /// A cut to where the model stands is nothing to do: the thread answers
    /// it without a rollback.
    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        let pos =
            u32::try_from(n).map_err(|_| EngineError(format!("cut to position {n}: past u32")))?;
        self.link.call(Cmd::Rollback(pos)).map(|_| ())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        match self.link.ask(Cmd::Pos) {
            Ok(r) => format!("card={} position={}", self.card, r.pos),
            Err(e) => format!("card={} position unknown ({})", self.card, e.0),
        }
    }

    fn props_engine(&self) -> EngineProps {
        self.props.clone()
    }
}

impl Drop for SeatEngine {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop; the model is dropped there.
        self.link.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Cmd, Link, Reply};
    use std::sync::mpsc;

    const N_VOCAB: usize = 8;

    /// The factory's sampler is the sampler crate's chain over the request's
    /// parameters, and a parameter the chain refuses (a non-finite
    /// temperature) is the request's refusal in the chain's own words: no
    /// sampler that draws something else stands in for it.
    #[test]
    fn the_sampler_factory_refuses_what_the_chain_refuses() {
        use serve::SamplingParams;
        let factory = super::sampler_factory();
        let greedy = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..SamplingParams::default()
        };
        let mut draw = factory(&greedy).unwrap_or_else(|e| panic!("refused: {e}"));
        assert_eq!(draw(&[0.0, 3.0, 1.0], &[]), 1, "the chain's argmax");
        let Err(refused) = factory(&SamplingParams {
            temperature: f32::NAN,
            ..greedy
        }) else {
            panic!("a NaN temperature built a sampler");
        };
        assert!(
            refused.0.contains("temperature") && refused.0.contains("a finite number"),
            "{refused}"
        );
    }

    /// A link to a thread that answers every `Next` as the engine thread does:
    /// it fills the lent row and hands it back with the argmax. A
    /// `StepSlots` it answers the same way, a row at a time in row order, and
    /// a `PassSlots` keeps `last` and one drafted id a row.
    fn echo_link() -> (Link, std::thread::JoinHandle<()>) {
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let worker = std::thread::spawn(move || {
            for (at, cmd) in cmds.into_iter().enumerate() {
                let reply = match cmd {
                    Cmd::Next { last, mut logits } => {
                        if let Some(row) = logits.as_deref_mut() {
                            row.fill(last as f32);
                        }
                        Reply {
                            result: Ok(last + 1),
                            logits,
                            pos: at + 1,
                            extra: super::Extra::None,
                        }
                    }
                    Cmd::StepSlots(mut rows) => {
                        for r in rows.iter_mut() {
                            if let Some(row) = r.logits.as_deref_mut() {
                                row.fill(r.last as f32);
                            }
                            r.next = r.last + 1;
                        }
                        Reply {
                            result: Ok(0),
                            logits: None,
                            pos: at + 1,
                            extra: super::Extra::StepSlots(rows),
                        }
                    }
                    Cmd::PassSlots(mut rows) => {
                        for r in rows.iter_mut() {
                            r.out = vec![r.last, r.last + 1];
                            r.drafted = serve::Drafted {
                                proposed: 1,
                                accepted: 1,
                            };
                        }
                        Reply {
                            result: Ok(0),
                            logits: None,
                            pos: at + 1,
                            extra: super::Extra::PassSlots(rows),
                        }
                    }
                    _ => panic!("the echo thread answers Next, StepSlots and PassSlots only"),
                };
                if replies.send(reply).is_err() {
                    return;
                }
            }
        });
        let link = Link {
            tx: Some(tx),
            rx,
            logits: vec![0.0; N_VOCAB],
            scratch: Vec::new(),
        };
        (link, worker)
    }

    /// A round of several slots is one command whose reply sets each row's
    /// answer in row order ([`Link::step_slots`]), the rows' logits reading
    /// into the link's own scratch — grown once on the first round, reused
    /// every round, so no token allocates a row; a caller buffer of another
    /// length and a slot named twice are named errors.
    #[test]
    fn a_round_of_slots_answers_in_row_order() {
        use serve::SlotRow;
        let (mut link, worker) = echo_link();
        let mut a = [0.0f32; N_VOCAB];
        let mut b = [0.0f32; N_VOCAB];
        let mut rows = vec![
            SlotRow {
                slot: 0,
                last: 3,
                logits: Some(&mut a),
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 5,
                logits: None,
                next: 0,
            },
            SlotRow {
                slot: 2,
                last: 9,
                logits: Some(&mut b),
                next: 0,
            },
        ];
        link.step_slots(&mut rows).expect("a round of three slots");
        assert_eq!(
            rows.iter().map(|r| r.next).collect::<Vec<_>>(),
            vec![4, 6, 10],
            "each row its own answer, in row order"
        );
        assert_eq!(a, [3.0f32; N_VOCAB]);
        assert_eq!(b, [9.0f32; N_VOCAB]);
        assert_eq!(link.scratch.len(), 3, "the scratch came home");
        let first = link.scratch[0].as_ptr();
        let third = link.scratch[2].as_ptr();
        let mut c = [0.0f32; N_VOCAB];
        let mut rows = vec![
            SlotRow {
                slot: 1,
                last: 7,
                logits: Some(&mut c),
                next: 0,
            },
            SlotRow {
                slot: 0,
                last: 1,
                logits: None,
                next: 0,
            },
        ];
        link.step_slots(&mut rows).expect("a second, shorter round");
        assert_eq!(rows[0].next, 8);
        assert_eq!(rows[1].next, 2);
        assert_eq!(c, [7.0f32; N_VOCAB]);
        assert_eq!(link.scratch[0].as_ptr(), first, "the scratch reused");
        assert_eq!(
            link.scratch[2].as_ptr(),
            third,
            "an untouched row kept its row"
        );
        // A caller buffer of another length is a named error, and the
        // scratch it had lent comes home for the next round.
        let mut short = [0.0f32; N_VOCAB - 1];
        let mut rows = vec![
            SlotRow {
                slot: 0,
                last: 2,
                logits: Some(&mut short),
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 4,
                logits: None,
                next: 0,
            },
        ];
        let e = link
            .step_slots(&mut rows)
            .expect_err("a short buffer among the rows");
        assert!(e.0.contains("holds 7"), "{e}");
        // Two rows of one slot are a named error before anything crosses.
        let mut rows = vec![
            SlotRow {
                slot: 1,
                last: 2,
                logits: None,
                next: 0,
            },
            SlotRow {
                slot: 1,
                last: 4,
                logits: None,
                next: 0,
            },
        ];
        let e = link
            .step_slots(&mut rows)
            .expect_err("one slot twice in a round");
        assert!(e.0.contains("names slot 1 twice"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A round of several drafted passes is one command whose reply gives
    /// each row its own kept ids and drafted counts in row order
    /// ([`Link::pass_slots`]), the caller's buffers crossing with the
    /// command; a slot named twice is a named error.
    #[test]
    fn a_round_of_passes_answers_in_row_order() {
        use serve::SlotPass;
        let (mut link, worker) = echo_link();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut rows = vec![
            SlotPass {
                slot: 1,
                last: 3,
                out: &mut a,
                drafted: serve::Drafted::default(),
            },
            SlotPass {
                slot: 0,
                last: 5,
                out: &mut b,
                drafted: serve::Drafted::default(),
            },
        ];
        link.pass_slots(&mut rows).expect("a round of two passes");
        let counts = rows.iter().map(|r| r.drafted).collect::<Vec<_>>();
        drop(rows);
        assert_eq!(a, vec![3, 4], "the first row's kept ids");
        assert_eq!(b, vec![5, 6], "the second row's kept ids");
        assert_eq!(
            counts,
            vec![
                serve::Drafted {
                    proposed: 1,
                    accepted: 1
                };
                2
            ],
            "each row its own counts, in row order"
        );
        // Two rows of one slot are a named error before anything crosses.
        let mut c = Vec::new();
        let mut d = Vec::new();
        let mut rows = vec![
            SlotPass {
                slot: 2,
                last: 2,
                out: &mut c,
                drafted: serve::Drafted::default(),
            },
            SlotPass {
                slot: 2,
                last: 4,
                out: &mut d,
                drafted: serve::Drafted::default(),
            },
        ];
        let e = link
            .pass_slots(&mut rows)
            .expect_err("one slot twice in a round of passes");
        assert!(e.0.contains("names slot 2 twice"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// Every sampled step reads into the buffer the link was built with: the
    /// row crosses to the engine thread and back, and no token allocates one.
    #[test]
    fn sampled_steps_reuse_one_logits_buffer() {
        let (mut link, worker) = echo_link();
        let built = link.logits.as_ptr();
        let mut out = [0.0f32; N_VOCAB];
        for last in [3u32, 5, 9] {
            let arg = link.next(last, Some(&mut out)).expect("a step");
            assert_eq!(arg, last + 1);
            assert_eq!(out, [last as f32; N_VOCAB], "the row of step {last}");
            assert_eq!(
                link.logits.as_ptr(),
                built,
                "step {last} read into another allocation than the link's"
            );
            assert_eq!(link.logits.len(), N_VOCAB);
        }
        let greedy = link.next(11, None).expect("a greedy step");
        assert_eq!(greedy, 12);
        assert_eq!(link.logits.as_ptr(), built, "a greedy step kept the buffer");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A caller's buffer of another length is a named error, not a partial copy.
    #[test]
    fn a_short_caller_buffer_is_refused() {
        let (mut link, worker) = echo_link();
        let mut short = [0.0f32; N_VOCAB - 1];
        let e = link.next(3, Some(&mut short)).expect_err("a short buffer");
        assert!(e.0.contains("holds 7, the engine's 8"), "{e}");
        link.tx = None;
        worker.join().expect("the echo thread");
    }

    /// A MiB value past u64 bytes names the flag it parsed
    /// ([`CacheRam::parse_mib`]): the park flag's, not the cache's.
    #[test]
    fn a_mib_value_past_u64_names_the_flag_it_parsed() {
        let e = super::CacheRam::parse_mib("--park-ram", &(1u64 << 44).to_string())
            .expect_err("2^44 MiB passes u64 bytes");
        let e = e.to_string();
        assert!(e.contains("--park-ram 17592186044416 MiB"), "{e}");
        assert!(!e.contains("--cache-ram"), "{e}");
    }

    /// The checkpoints a load's sequences pin grow one [`HOST_BUDGET`] a
    /// sequence ([`checkpoint_bytes`]), the bytes the machine's checkpoint
    /// reserve carries.
    #[test]
    fn checkpoint_bytes_grow_a_budget_a_sequence() {
        use crate::residency38::checkpoint_bytes;
        let budget = runtime::seqstate::HOST_BUDGET;
        assert_eq!(checkpoint_bytes(0), 0);
        assert_eq!(checkpoint_bytes(1), budget);
        assert_eq!(checkpoint_bytes(3), 3 * budget);
        assert_eq!(checkpoint_bytes(usize::MAX), u64::MAX);
    }

    /// The unset residency rule's room is what the reading leaves past the
    /// plan's host need ([`mem_left_38`]), the need holding the checkpoints
    /// the load's machine reserves ([`reserve_checkpoints`]) once: a churn
    /// pool that fits past the need of a load that declares none does not
    /// fit past the same plan's need on a machine that reserves two slots',
    /// and the pick turns `MemShort`, naming the room the reserve leaves.
    #[test]
    fn the_residency38_room_reads_the_reserved_need() {
        use crate::residency38::{Seqs, checkpoints_reserved, mem_left_38, reserve_checkpoints};
        use model::placement::{Host, Machine};
        use runtime::seqstate::HOST_BUDGET;
        // A plan of one card layer of 100 experts, its host the rest: the
        // unset rule pins none (P 0), the pool a fixed 10 GiB whatever P is.
        let pick = |mem_left: i128| {
            bloomery_levers::residency38_at_plan(
                [100u64],
                384,
                bloomery_levers::PlanTier::NONE,
                |_pinned: usize| Ok::<u64, std::convert::Infallible>(10u64 << 30),
                i128::MAX,
                mem_left,
            )
            .expect("the pick")
        };
        let declared = |checkpoints: bool| {
            let mut machine = Machine {
                cards: Vec::new(),
                tiers: Vec::new(),
                host: Host {
                    usable_bytes: 0,
                    reserves: Vec::new(),
                },
                unified: None,
            };
            reserve_checkpoints(
                &mut machine,
                Seqs {
                    slots: 2,
                    checkpoints,
                },
            );
            checkpoints_reserved(&machine)
        };
        // The plan's own terms, 12 GiB, and the reserve the need adds.
        let (available, terms) = (24u64 << 30, 12u64 << 30);
        let none = terms + declared(false);
        assert_eq!(pick(mem_left_38(available, none)).pinned, Some(0));
        let two = terms + declared(true);
        assert_eq!(two - none, 2 * HOST_BUDGET);
        let short = pick(mem_left_38(available, two));
        assert_eq!(
            short.why,
            bloomery_levers::Residency38Why::MemShort {
                needs: 10u64 << 30,
                leaves: i128::from(available) - i128::from(terms) - i128::from(2 * HOST_BUDGET),
            }
        );
        assert_eq!(
            short.why.to_string(),
            format!(
                "unset: the churn pool needs {} B, MemAvailable leaves {} B past the plan's host \
                 need",
                10u64 << 30,
                i128::from(available) - i128::from(two)
            )
        );
    }

    mod media {
        use std::sync::Arc;

        use serve::media::{ImageKey, MediaFeed};

        use super::super::{Cmd, Seat, serve_cmd};
        use crate::GateError;

        /// One recorded call: the ids it took, each feed's position and key.
        type Took = Vec<(Vec<u32>, Vec<(usize, u8)>)>;

        /// A feed of a two-position span whose image is keyed `key`, at `at`.
        fn feed(at: usize, key: u8) -> MediaFeed {
            MediaFeed {
                at,
                key: ImageKey([key; 32]),
                prepared: Arc::new(vision::Prepared {
                    span_len: 2,
                    patches: vision::Patches {
                        plan: vision::GridPlan {
                            n_llm_h: 1,
                            n_llm_w: 1,
                            best_h: 14,
                            best_w: 14,
                        },
                        n_vit_h: 1,
                        n_vit_w: 1,
                        patch_len: 3,
                        bf16: vec![0; 3],
                    },
                }),
            }
        }

        /// A seat that takes images: its `prefill_media` records the call it
        /// took (the ids and each feed's position and key) and answers 7; its
        /// `prefill` answers 5; each call advances `pos` past its ids, as a
        /// seat does. Without the override, [`NoImages`] holds the trait's
        /// defaults over the same two.
        struct Images {
            pos: usize,
            took: Took,
        }

        impl Seat for Images {
            fn pos(&self) -> usize {
                self.pos
            }
            fn ctx_max(&self) -> usize {
                1 << 10
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
                self.took.push((ids.to_vec(), Vec::new()));
                self.pos += ids.len();
                Ok(5)
            }
            fn prefill_media(
                &mut self,
                ids: &[u32],
                feeds: &[MediaFeed],
            ) -> Result<u32, GateError> {
                self.took.push((
                    ids.to_vec(),
                    feeds.iter().map(|f| (f.at, f.key.0[0])).collect::<Vec<_>>(),
                ));
                self.pos += ids.len();
                Ok(7)
            }
            fn step(&mut self, _: u32) -> Result<u32, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn reset(&mut self) -> Result<(), GateError> {
                Ok(())
            }
            fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                Ok(())
            }
            fn keep(&self, _: usize) -> (usize, Option<String>) {
                (0, None)
            }
            fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                Vec::new()
            }
            fn snapshot(&mut self) -> Result<Arc<dyn serve::Saved>, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn resume(&mut self, _: &dyn serve::Saved) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn note(_: &serve::CacheNote) {}
        }

        /// [`Images`] with the trait's `prefill_media` default: the same
        /// record shape tells which path each call took.
        struct NoImages {
            took: Took,
        }

        impl NoImages {
            /// The seat over a fresh record.
            fn new() -> NoImages {
                NoImages { took: Vec::new() }
            }
        }

        impl Seat for NoImages {
            fn pos(&self) -> usize {
                0
            }
            fn ctx_max(&self) -> usize {
                1 << 10
            }
            fn prefill(&mut self, ids: &[u32]) -> Result<u32, GateError> {
                self.took.push((ids.to_vec(), Vec::new()));
                Ok(5)
            }
            fn step(&mut self, _: u32) -> Result<u32, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn reset(&mut self) -> Result<(), GateError> {
                Ok(())
            }
            fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                Ok(())
            }
            fn keep(&self, _: usize) -> (usize, Option<String>) {
                (0, None)
            }
            fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                Vec::new()
            }
            fn snapshot(&mut self) -> Result<Arc<dyn serve::Saved>, GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn resume(&mut self, _: &dyn serve::Saved) -> Result<(), GateError> {
                Err("this seat serves prompt calls alone".into())
            }
            fn note(_: &serve::CacheNote) {}
        }

        /// A `PrefillMedia` command is the seat's own `prefill_media` with
        /// its ids and each feed's position and key, and the seat's answer is
        /// the reply's; a call past the context is refused before it runs,
        /// naming the call and the position.
        #[test]
        fn a_prefill_media_command_reaches_the_seat() {
            let mut g = Images {
                pos: 3,
                took: Vec::new(),
            };
            let ids = vec![9u32, 4, 4, 8];
            let feeds = vec![feed(1, 2), feed(4, 3)];
            let (r, logits) = serve_cmd(&mut g, Cmd::PrefillMedia(ids.clone(), feeds), 8);
            assert_eq!(r.expect("the call"), 7);
            assert!(logits.is_none());
            assert_eq!(
                g.took,
                vec![(ids, vec![(1, 2u8), (4, 3)])],
                "one call, the ids and both feeds as the seat took them"
            );
            let long = vec![0u32; 1 << 10];
            let e = serve_cmd(&mut g, Cmd::PrefillMedia(long, vec![feed(0, 1)]), 8)
                .0
                .expect_err("a call past the context");
            assert!(
                e.contains("prefill of 1024 ids with 1 image(s) from position 7"),
                "{e}"
            );
            assert_eq!(g.took.len(), 1, "the refused call reached no seat");
        }

        /// The trait's default ([`Seat::prefill_media`]): a call that carries
        /// no image is the seat's plain `prefill`, one that carries any is a
        /// named refusal — the engine that took no [`Seat::media_model`]
        /// never answers an image with text.
        #[test]
        fn the_default_seat_refuses_an_image_by_name() {
            let mut g = NoImages::new();
            let (r, _) = serve_cmd(&mut g, Cmd::PrefillMedia(vec![1, 2], Vec::new()), 8);
            assert_eq!(r.expect("the plain call"), 5);
            assert_eq!(g.took, vec![(vec![1, 2], Vec::new())]);
            let e = serve_cmd(
                &mut g,
                Cmd::PrefillMedia(vec![1, 2, 3], vec![feed(0, 1)]),
                8,
            )
            .0
            .expect_err("an image the default refuses");
            assert!(e.contains("this seat takes no image input"), "{e}");
            assert_eq!(g.took.len(), 1, "the refused call reached no prefill");
        }
    }

    mod slot_files {
        use std::io::{self, Read, Write};
        use std::sync::Arc;
        use std::sync::mpsc;
        use std::thread::JoinHandle;

        use serve::{CacheNote, EngineError, Saved, SavedState, StateError};

        use bloomery_gpu::GpuError;

        use super::super::{
            Cmd, Extra, Link, PIPE_CHUNK, Reply, Seat, restore_from, resume_of, save_into, settle,
            snapshot_of,
        };
        use crate::GateError;

        /// The [`Seat`] methods a slot file's command never calls.
        macro_rules! no_steps {
            () => {
                fn pos(&self) -> usize {
                    self.pos
                }
                fn ctx_max(&self) -> usize {
                    1 << 10
                }
                fn prefill(&mut self, _: &[u32]) -> Result<u32, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn step(&mut self, _: u32) -> Result<u32, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn logits_into(&self, _: &mut [f32]) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn reset(&mut self) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn rollback(&mut self, _: u32) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn keep(&self, _: usize) -> (usize, Option<String>) {
                    (0, None)
                }
                fn splits(&self, _: usize, _: usize, _: &[usize]) -> Vec<usize> {
                    Vec::new()
                }
                fn note(_: &CacheNote) {}
            };
        }

        /// The prompt-cache methods of a seat that serves slot files alone.
        macro_rules! no_cache {
            () => {
                fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
                    Err("this seat serves slot files alone".into())
                }
                fn resume(&mut self, _: &dyn Saved) -> Result<(), GateError> {
                    Err("this seat serves slot files alone".into())
                }
            };
        }

        /// A seat that keeps the trait's slot-file defaults.
        struct NoFiles {
            pos: usize,
        }

        impl Seat for NoFiles {
            no_steps!();
            no_cache!();
        }

        /// The positions [`Files`] stands at after a restore.
        const RESTORED: usize = 5;

        /// A seat whose state is `state`, which a restore replaces with what
        /// it read; `fail` makes both refuse with that kind instead.
        struct Files {
            pos: usize,
            state: Vec<u8>,
            fail: Option<Fail>,
        }

        #[derive(Clone, Copy)]
        enum Fail {
            Format,
            Engine,
        }

        impl Fail {
            fn error(self) -> StateError {
                match self {
                    Fail::Format => StateError::Format("a state of another model".to_owned()),
                    Fail::Engine => StateError::Engine(EngineError("the card fell off".to_owned())),
                }
            }
        }

        impl Seat for Files {
            no_steps!();
            no_cache!();

            fn save_state(&mut self, out: &mut dyn Write) -> Result<u64, StateError> {
                if let Some(f) = self.fail {
                    return Err(f.error());
                }
                out.write_all(&self.state)?;
                Ok(self.state.len() as u64)
            }

            fn restore_state(&mut self, input: &mut dyn Read) -> Result<usize, StateError> {
                if let Some(f) = self.fail {
                    return Err(f.error());
                }
                self.state.clear();
                input.read_to_end(&mut self.state)?;
                self.pos = RESTORED;
                Ok(RESTORED)
            }
        }

        /// A seat whose prompt-cache calls fail with `fail`'s error.
        struct Caches {
            pos: usize,
            fail: fn() -> GateError,
        }

        impl Seat for Caches {
            no_steps!();

            fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError> {
                Err((self.fail)())
            }

            fn resume(&mut self, _: &dyn Saved) -> Result<(), GateError> {
                Err((self.fail)())
            }
        }

        /// A prompt-cache state of the positions it names, which no seat here
        /// takes back.
        struct Held(usize);

        impl Saved for Held {
            fn n_tokens(&self) -> usize {
                self.0
            }

            fn n_bytes(&self) -> u64 {
                0
            }

            fn keepable(&self, n: usize) -> usize {
                n.min(self.0)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        /// An error whose source is the one it wraps, as a seat's own error
        /// wraps the body's.
        #[derive(Debug)]
        struct Wrapped(GpuError);

        impl std::fmt::Display for Wrapped {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "the session refused")
            }
        }

        impl std::error::Error for Wrapped {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }

        /// A link to a thread that answers a seat's state commands as the
        /// engine thread does ([`save_into`], [`restore_from`],
        /// [`snapshot_of`], [`resume_of`]); joining it hands the seat back.
        fn file_link<S: Seat + Send>(mut seat: S) -> (Link, JoinHandle<S>) {
            let (tx, cmds) = mpsc::channel::<Cmd>();
            let (replies, rx) = mpsc::channel::<Reply>();
            let worker = std::thread::spawn(move || {
                for cmd in cmds {
                    let extra = match cmd {
                        Cmd::SaveState(chunks) => Extra::State(save_into(&mut seat, chunks)),
                        Cmd::RestoreState(chunks) => Extra::State(restore_from(&mut seat, chunks)),
                        Cmd::Save => Extra::Saved(snapshot_of(&mut seat)),
                        Cmd::Resume(state) => Extra::Resumed(resume_of(&mut seat, &*state)),
                        _ => panic!("the file thread answers a seat's state commands only"),
                    };
                    let reply = Reply {
                        result: Ok(0),
                        logits: None,
                        pos: seat.pos(),
                        extra,
                    };
                    if replies.send(reply).is_err() {
                        break;
                    }
                }
                seat
            });
            let link = Link {
                tx: Some(tx),
                rx,
                logits: Vec::new(),
                scratch: Vec::new(),
            };
            (link, worker)
        }

        /// `n` bytes of a pattern that repeats past no chunk boundary.
        fn pattern(n: usize, salt: u8) -> Vec<u8> {
            (0..n).map(|i| (i % 251) as u8 ^ salt).collect()
        }

        /// A writer whose every write fails, as a full disk does.
        struct Full;

        impl Write for Full {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::other("no space left on the device"))
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        /// A state of several chunks and a partial one crosses the pipe byte
        /// for byte, each way: the seat's bytes reach the server's `out`, the
        /// file's bytes reach the seat, and the answers count them.
        #[test]
        fn a_slot_state_crosses_the_pipe_byte_for_byte() {
            let n = 3 * PIPE_CHUNK + 17;
            let state = pattern(n, 0);
            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: state.clone(),
                fail: None,
            });
            let mut out = Vec::new();
            let saved = link.save_state(&mut out).expect("a save");
            assert_eq!(
                saved,
                SavedState {
                    n_tokens: 9,
                    n_bytes: n as u64
                }
            );
            assert!(out == state, "the saved bytes differ from the seat's");
            let file = pattern(n, 0x5a);
            let restored = link.restore_state(&mut file.as_slice()).expect("a restore");
            assert_eq!(
                restored,
                SavedState {
                    n_tokens: RESTORED,
                    n_bytes: n as u64
                }
            );
            link.tx = None;
            let seat = worker.join().expect("the file thread");
            assert!(
                seat.state == file,
                "the seat read other bytes than the file's"
            );
        }

        /// A seat that keeps the trait's defaults answers `Unsupported` both
        /// ways (the server's 501), and a save writes nothing.
        #[test]
        fn the_default_seat_answers_unsupported() {
            let (mut link, worker) = file_link(NoFiles { pos: 3 });
            let mut out = Vec::new();
            let e = link.save_state(&mut out).expect_err("the default save");
            assert!(
                matches!(e, StateError::Unsupported("slot save/restore")),
                "{e:?}"
            );
            assert!(out.is_empty(), "the default save wrote {} bytes", out.len());
            let file = pattern(PIPE_CHUNK + 1, 0);
            let e = link
                .restore_state(&mut file.as_slice())
                .expect_err("the default restore");
            assert!(
                matches!(e, StateError::Unsupported("slot save/restore")),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");
        }

        /// The kind crosses the thread: a refusal stays `Format` (its words
        /// as the seat gave them), an engine failure stays `Engine` (with the
        /// command and the position), and an error of the server's `out` is
        /// `Io`. An engine failure is never hidden behind the stream's error.
        #[test]
        fn a_state_error_keeps_its_kind() {
            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: Vec::new(),
                fail: Some(Fail::Format),
            });
            let e = link
                .restore_state(&mut [1u8, 2, 3].as_slice())
                .expect_err("a refused restore");
            assert!(
                matches!(&e, StateError::Format(m) if m == "a state of another model"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: Vec::new(),
                fail: Some(Fail::Engine),
            });
            let e = link.save_state(&mut Vec::new()).expect_err("a failed save");
            assert!(
                matches!(&e, StateError::Engine(EngineError(m))
                    if m == "save at position 9: the card fell off"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let (mut link, worker) = file_link(Files {
                pos: 9,
                state: pattern(3 * PIPE_CHUNK, 0),
                fail: None,
            });
            let e = link
                .save_state(&mut Full)
                .expect_err("a save to a full disk");
            assert!(
                matches!(&e, StateError::Io(io) if io.to_string() == "no space left on the device"),
                "{e:?}"
            );
            link.tx = None;
            worker.join().expect("the file thread");

            let stream = || Err(io::Error::other("no space left on the device"));
            let engine = Err(StateError::Engine(EngineError(
                "the card fell off".to_owned(),
            )));
            assert!(matches!(
                settle(engine, stream()),
                Err(StateError::Engine(_))
            ));
            let format = Err(StateError::Format("a broken pipe".to_owned()));
            assert!(matches!(settle(format, stream()), Err(StateError::Io(_))));
        }

        /// A prompt-cache state the seat refuses by name — a `GpuError` of
        /// kind `Shape` or `State`, itself or in the chain — reaches the
        /// server as `Format`, the request's (the server keeps no state or
        /// resets the slot), not `Engine`, which would end it. Any other
        /// failure stays the engine's.
        #[test]
        fn a_refused_cache_state_is_the_requests() {
            fn waits() -> GateError {
                Box::new(GpuError::Shape {
                    what: "deepseek41 sequence state",
                    detail: "a call while the pass of slots [0, 1] waits for its commit".to_owned(),
                })
            }
            fn missing() -> GateError {
                Box::new(Wrapped(GpuError::State {
                    what: "deepseek41 sequence state",
                    missing: "a reset",
                }))
            }
            fn lost() -> GateError {
                Box::new(GpuError::Protocol {
                    what: "the host tier",
                    detail: "a go that did not land".to_owned(),
                })
            }
            fn unnamed() -> GateError {
                "a saved state that is not this body's".into()
            }
            let held: Arc<dyn Saved> = Arc::new(Held(4));
            let cases: [(fn() -> GateError, bool); 4] = [
                (waits, true),
                (missing, true),
                (lost, false),
                (unnamed, false),
            ];
            for (fail, refused) in cases {
                let (mut link, worker) = file_link(Caches { pos: 7, fail });
                let saved = link.snapshot().map(|_| ()).expect_err("a failed snapshot");
                let resumed = link.resume(&held).expect_err("a failed resume");
                let want = fail().to_string();
                for (e, head) in [
                    (&saved, "snapshot at position 7: "),
                    (&resumed, "resume of a state of 4 positions: "),
                ] {
                    let m = match e {
                        StateError::Format(m) if refused => m,
                        StateError::Engine(EngineError(m)) if !refused => m,
                        other => panic!("{want}: {other:?}"),
                    };
                    assert_eq!(*m, format!("{head}{want}"));
                }
                link.tx = None;
                worker.join().expect("the file thread");
            }
        }
    }

    mod cache_ram {
        use super::super::{CacheRam, HostRead};
        use crate::generate::Place;
        use crate::residency38::{
            CHECKPOINTS_RESERVE, Seqs, checkpoint_bytes, checkpoints_reserved, glm_machine,
            reserve_checkpoints,
        };
        use model::placement::{Host, Machine};
        use runtime::seqstate::HOST_BUDGET;

        /// A machine of `os` reserved bytes on the host and nothing else.
        fn machine(os: u64) -> Machine {
            Machine {
                cards: Vec::new(),
                tiers: Vec::new(),
                host: Host {
                    usable_bytes: 0,
                    reserves: vec![("os".to_owned(), os)],
                },
                unified: None,
            }
        }

        /// The default budget's zero is a named record, not a silent one: a
        /// reading that leaves nothing past the need and the pool gives
        /// `ram=0` with a `why=` naming the arithmetic, and the `cache` line
        /// carries it; a leftover halves under the cap with no why, and one
        /// byte left halves to a named zero too.
        #[test]
        fn a_default_budget_of_zero_names_its_terms() {
            let (ram, why) = CacheRam::default_of(10_000, 6_000, 1_000);
            assert_eq!((ram, why.is_none()), (1_500, true));
            for available in [6_000 + 1_000, 6_000 + 1_000 + 1] {
                let (ram, why) = CacheRam::default_of(available, 6_000, 1_000);
                assert_eq!(ram, 0, "nothing left at {available} B");
                let why = why.expect("the zero is named");
                for part in [
                    format!("the reading {available} B leaves"),
                    "past need 6000 B + pool 1000 B".to_owned(),
                    "no prompt cache".to_owned(),
                ] {
                    assert!(why.contains(&part), "{part:?} in {why}");
                }
            }
            let (_, why) = CacheRam::default_of(6_000, 6_000, 1_000);
            let line = CacheRam {
                ram: 0,
                set: false,
                available: 6_000,
                reading: HostRead::Given,
                need: 6_000,
                pool: 1_000,
                tier: 0,
                paged: 0,
                checkpoints: 0,
                why,
            }
            .line();
            assert!(line.contains("ram=0"), "{line}");
            assert!(line.contains(" why=the reading 6000 B leaves"), "{line}");
        }

        /// A plan the NVMe tier split takes no default budget: the split
        /// fills the room past the tier's floor (its arena and the host
        /// experts it keeps, inside the need), so a reading past the need is
        /// rounding or drift — on a split with an arena and on one without
        /// (the mapping path), the budget is 0 and `why=` names the split, and
        /// the line prints the arena as `tier` and the drive's bytes as
        /// `paged`. A set budget keeps its value. A need under the arena it
        /// holds, read without it, and an arena on a plan that pages nothing
        /// are refused by name.
        #[test]
        fn a_split_plan_takes_no_default_budget() {
            let checkpoints = checkpoint_bytes(1);
            let (plan, pool, arena, paged) = (6_000u64, 1_000u64, 2_000u64, 3_000u64);
            let available = 100_000 + checkpoints;
            for (tier, paged) in [(arena, paged), (0, paged)] {
                let c = CacheRam::at(
                    None,
                    (available, HostRead::Given),
                    plan + tier + checkpoints,
                    pool,
                    (tier, paged),
                    checkpoints,
                )
                .expect("the budget");
                assert_eq!((c.ram, c.tier, c.paged), (0, tier, paged));
                let why = c.why.as_deref().expect("the zero is named");
                for part in [
                    "the NVMe tier's split fills the room past its floor".to_owned(),
                    format!("arena {tier} B"),
                    format!("{paged} B read from the drive"),
                    "no prompt cache".to_owned(),
                ] {
                    assert!(why.contains(&part), "{part:?} in {why}");
                }
                let line = c.line();
                assert!(
                    line.contains(&format!(" tier={tier} paged={paged} ")),
                    "{line}"
                );
                let set = CacheRam::at(
                    Some(4_096),
                    (available, HostRead::Given),
                    plan + tier + checkpoints,
                    pool,
                    (tier, paged),
                    checkpoints,
                )
                .expect("the set budget");
                assert_eq!((set.ram, set.why.as_deref()), (4_096, None));
            }
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                plan - 5_000,
                pool,
                (arena, paged),
                checkpoints,
            ) else {
                panic!("a need without its arena is refused");
            };
            let e = e.to_string();
            assert!(e.contains("under the NVMe tier arena 2000 B"), "{e}");
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                plan + arena,
                pool,
                (arena, 0),
                checkpoints,
            ) else {
                panic!("an arena on a plan that pages nothing is refused");
            };
            let e = e.to_string();
            assert!(
                e.contains("arena of 2000 B and pages no routed expert"),
                "{e}"
            );
        }

        /// A default budget too small for one saved state is off: `holding`
        /// at one byte over it gives 0 with a `why=` naming both, at the
        /// state's own bytes it keeps the budget, a zero stays the zero it
        /// was, and a set budget keeps its value at any state size.
        #[test]
        fn a_default_under_one_saved_state_is_off() {
            let (need, pool) = (6_000u64, 1_000u64);
            let available = need + pool + 100_000;
            let at = |set: Option<u64>| {
                CacheRam::at(set, (available, HostRead::Given), need, pool, (0, 0), 0)
                    .expect("the budget")
            };
            assert_eq!(at(None).ram, 50_000);
            let off = at(None).holding(50_001);
            assert_eq!(off.ram, 0);
            let why = off.why.as_deref().expect("the zero is named");
            assert!(
                why.contains("the default budget 50000 B is under one saved state's 50001 B"),
                "{why}"
            );
            assert!(off.line().contains("ram=0 rule=default"), "{}", off.line());
            let kept = at(None).holding(50_000);
            assert_eq!((kept.ram, kept.why.as_deref()), (50_000, None));
            let set = at(Some(10)).holding(11);
            assert_eq!((set.ram, set.set, set.why.as_deref()), (10, true, None));
            let none = CacheRam::at(None, (need, HostRead::Given), need, pool, (0, 0), 0)
                .expect("the budget");
            let why = none.why.clone();
            let still = none.holding(u64::MAX);
            assert_eq!((still.ram, still.why), (0, why));
        }

        /// The checkpoint reserve is each load's declaration
        /// (`reserve_checkpoints`): V4.1's body makes no checkpoints, so its
        /// seat declares none and its machine carries no row and reserves 0;
        /// the Qwen3.8 seat's two slots, taking them, carry one row of
        /// `HOST_BUDGET` a slot; a second declaration on one machine is a
        /// panic by name.
        #[test]
        fn the_checkpoint_reserve_is_each_loads_declaration() {
            let declared = |seqs: Seqs| {
                let mut m = machine(1_000);
                reserve_checkpoints(&mut m, seqs);
                m
            };
            let v41 = declared(Seqs {
                slots: 2,
                checkpoints: false,
            });
            assert_eq!(v41.host.reserves, vec![("os".to_owned(), 1_000)]);
            assert_eq!(checkpoints_reserved(&v41), 0);
            let q38 = declared(Seqs {
                slots: 2,
                checkpoints: true,
            });
            assert_eq!(
                q38.host.reserves,
                vec![
                    ("os".to_owned(), 1_000),
                    (CHECKPOINTS_RESERVE.to_owned(), checkpoint_bytes(2))
                ]
            );
            assert_eq!(checkpoints_reserved(&q38), checkpoint_bytes(2));
            let twice = std::panic::catch_unwind(|| {
                let mut m = declared(Seqs {
                    slots: 2,
                    checkpoints: true,
                });
                reserve_checkpoints(&mut m, Seqs::ONE);
            })
            .expect_err("a second declaration panics");
            let said = twice
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| twice.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            assert!(said.contains("reserves its checkpoints already"), "{said}");
        }

        /// A GLM load's machine (`glm_machine`, the one its offer, plans and
        /// open take) reserves every slot's checkpoints, since the GLM body
        /// takes them on every load: one `HOST_BUDGET` at one slot and three
        /// at three. At one slot the prompt cache's default budget is the one
        /// the load's single sequence left when the term sat beside the need
        /// (half of the reading past need, pool and one `HOST_BUDGET`); at
        /// three the two further slots' come off it too.
        #[test]
        fn a_glm_load_reserves_every_slots_checkpoints() {
            let layers = 4;
            let (need, pool) = (6_000u64, 1_000u64);
            let available = need + pool + 4 * HOST_BUDGET + 100_000;
            for slots in [1usize, 3] {
                let m = glm_machine(Place::A, None, slots).expect("plan (a)'s machine")(layers);
                let reserved = checkpoints_reserved(&m);
                assert_eq!(reserved, checkpoint_bytes(slots), "{slots} slots");
                let c = CacheRam::at(
                    None,
                    (available, HostRead::Given),
                    need + reserved,
                    pool,
                    (0, 0),
                    reserved,
                )
                .expect("the budget");
                let beside_one = (available - need - pool - HOST_BUDGET) / 2;
                let slots_u64 = u64::try_from(slots).expect("a slot count fits u64");
                assert_eq!(
                    c.ram,
                    beside_one - (slots_u64 - 1) * HOST_BUDGET / 2,
                    "{slots} slots"
                );
            }
        }

        /// The default budget reads the checkpoints once, inside the need:
        /// a machine that reserves two slots' has them in the plan's need,
        /// so the budget is half of what the reading leaves past that need
        /// and the pool alone, the line printing the reserve as
        /// `checkpoints`; a need under the checkpoints it says it holds (a
        /// need of a machine with no reserve) is refused by name.
        #[test]
        fn a_default_budget_takes_the_reserved_checkpoints_once() {
            let mut m = machine(1_000);
            reserve_checkpoints(
                &mut m,
                Seqs {
                    slots: 2,
                    checkpoints: true,
                },
            );
            let checkpoints = checkpoints_reserved(&m);
            let (need, pool) = (6_000 + checkpoints, 1_000u64);
            let available = need + pool + 100_000;
            let c = CacheRam::at(
                None,
                (available, HostRead::Given),
                need,
                pool,
                (0, 0),
                checkpoints,
            )
            .expect("the budget");
            assert_eq!((c.ram, c.why.as_deref()), (50_000, None));
            assert!(
                c.line().ends_with(&format!(" checkpoints={checkpoints}")),
                "{}",
                c.line()
            );
            let Err(e) = CacheRam::at(
                None,
                (available, HostRead::Given),
                6_000,
                pool,
                (0, 0),
                checkpoints,
            ) else {
                panic!("a need without its machine's checkpoints is refused");
            };
            let e = e.to_string();
            assert!(
                e.contains(&format!("and the checkpoints {checkpoints} B it holds")),
                "{e}"
            );
        }
    }
}
