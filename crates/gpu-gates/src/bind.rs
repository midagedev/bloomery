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
//! What `/props` says about the engine ([`model_props`], [`placement_props`])
//! is read from the file's header and the placement plan the engine loads by,
//! once, before the load; the engine hands it back unchanged.

use std::collections::BTreeMap;
use std::ops::Range;
use std::process::Command;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use gguf::Split;
use model::placement::workstation::{self, DeviceId};
use model::placement::{Device, ModelTensors, Plan, Role};
use runtime::seqstate::HOST_BUDGET;
use sampler::{Sampler, SamplerParams};
use serve::{
    CacheNote, Decoder, DeviceProps, Drafted, Engine, EngineError, EngineProps, ModelProps,
    PlacementProps, ResidencyReset, SamplerFactory, SamplingParams, Saved, StateError, Tokenizer,
};

use crate::GateError;

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

/// The server's `SamplingParams` as the sampler crate's chain: no repetition
/// penalty (llama-server's default `repeat_penalty = 1.0`), `top_k <= 0`
/// keeps every candidate. The server never builds a sampler for
/// `temperature <= 0`: those requests take the engine's argmax.
pub fn sampler_params(p: &SamplingParams) -> SamplerParams {
    SamplerParams {
        temperature: p.temperature,
        top_k: u32::try_from(p.top_k).unwrap_or(0),
        top_p: p.top_p,
        // Past 1 keeps only the best candidate, as 1 does; the chain refuses it.
        min_p: p.min_p.min(1.0),
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: p.seed,
    }
}

/// A factory of sampler-crate samplers. A parameter the chain still refuses
/// (a non-finite value) falls back to the argmax.
pub fn sampler_factory() -> SamplerFactory {
    Arc::new(|p: &SamplingParams| -> serve::Sampler {
        match Sampler::new(sampler_params(p)) {
            Ok(mut s) => Box::new(move |logits: &[f32], recent: &[u32]| s.sample(logits, recent)),
            Err(_) => Box::new(|logits: &[f32], _: &[u32]| serve::sampling::argmax(logits)),
        }
    })
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
/// [`CACHE_RAM_CAP`] and half of what `MemAvailable` leaves past the plan's
/// host need, the residency's churn pool and the checkpoints' host budget
/// ([`HOST_BUDGET`]), 0 when nothing is left.
pub struct CacheRam {
    pub ram: u64,
    /// `--cache-ram` given, or the default's terms.
    pub set: bool,
    /// `MemAvailable` before the load.
    pub available: u64,
    /// The plan's host need.
    pub need: u64,
    /// The residency's churn pool, 0 without one.
    pub pool: u64,
}

impl CacheRam {
    /// `--cache-ram`'s value, `mib` MiB, in bytes; refused by name past u64.
    pub fn parse_mib(mib: &str) -> Result<u64, GateError> {
        let n: u64 = mib.parse()?;
        Ok(n.checked_mul(1 << 20)
            .ok_or_else(|| format!("--cache-ram {n} MiB passes u64 bytes"))?)
    }

    /// The budget (the type's doc): `set` in bytes as given, else the
    /// default over `need` and `pool`.
    pub fn of(set: Option<u64>, need: u64, pool: u64) -> Result<CacheRam, GateError> {
        let available = workstation::host_available()?;
        let ram = set.unwrap_or_else(|| {
            let left = i128::from(available)
                - i128::from(need)
                - i128::from(pool)
                - i128::from(HOST_BUDGET);
            u64::try_from((left / 2).max(0)).map_or(CACHE_RAM_CAP, |h| h.min(CACHE_RAM_CAP))
        });
        Ok(CacheRam {
            ram,
            set: set.is_some(),
            available,
            need,
            pool,
        })
    }

    /// The parked states' budget of `n` slots that take one model in turns
    /// (`serve::SwapEngine`): `set` (`--park-ram`) as given, else the lesser
    /// of [`CACHE_RAM_CAP`] and what `MemAvailable` leaves past the plan's
    /// host need, the churn pool, the checkpoints' host budget and the prompt
    /// cache's budget; refused by name when that leaves nothing.
    pub fn park(&self, set: Option<u64>, n: usize) -> Result<u64, GateError> {
        if let Some(b) = set {
            return Ok(b);
        }
        let b = self.park_or_none(None);
        if b > 0 {
            return Ok(b);
        }
        Err(format!(
            "--parallel {n}: MemAvailable {} B less the plan's host need {} B, the churn \
             pool {} B, the checkpoints {HOST_BUDGET} B and the prompt cache {} B leaves no \
             room to park a slot's state; give --park-ram MIB or a smaller --cache-ram",
            self.available, self.need, self.pool, self.ram
        )
        .into())
    }

    /// [`CacheRam::park`]'s terms, 0 when nothing is left: the elastic
    /// `--parallel` default's budget, which falls to the plain engine there
    /// instead of refusing. `set` (`--park-ram`) as given, as `park` takes it.
    #[must_use]
    pub fn park_or_none(&self, set: Option<u64>) -> u64 {
        if let Some(b) = set {
            return b;
        }
        let left = i128::from(self.available)
            - i128::from(self.need)
            - i128::from(self.pool)
            - i128::from(HOST_BUDGET)
            - i128::from(self.ram);
        u64::try_from(left).map_or(0, |b| b.min(CACHE_RAM_CAP))
    }

    /// The `cache` line's terms, the seat's own fields to follow: `cache
    /// ram=… rule=set|default available=… need=… pool=… checkpoints=…`.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "cache ram={} rule={} available={} need={} pool={} checkpoints={HOST_BUDGET}",
            self.ram,
            if self.set { "set" } else { "default" },
            self.available,
            self.need,
            self.pool
        )
    }
}

/// The most slots an elastic `--parallel` default names: the turns
/// serialize the one engine's decode and each switch moves a whole state,
/// so past a small group every slot the budget names only lengthens each
/// live request's wall; a caller that wants more names it.
pub const PARALLEL_CAP: usize = 4;

/// `--parallel`'s slot count and what decided it, for the `parallel` line a
/// seat that parks states prints: the parked states' `budget` holding one
/// whole-context `state` a slot (one parked while the others run), the flag
/// an upper bound on what it holds, [`PARALLEL_CAP`] the default's, never
/// below 1 (the plain engine, nothing parked).
pub struct Parallel {
    /// The slots, the `--parallel` the server serves.
    pub slots: usize,
    /// `flag` (the flag's bound, what it holds of it), `budget` (the
    /// default, what the budget holds), `plain` (one slot: the flag's, or a
    /// budget that holds no state).
    pub rule: &'static str,
    /// The parked states' budget, 0 with none.
    pub budget: u64,
    /// One slot's whole-context state, the budget's unit.
    pub state: u64,
}

impl Parallel {
    /// The count and the rule (the type's doc).
    #[must_use]
    pub fn of(flag: Option<usize>, budget: u64, state: u64) -> Parallel {
        let holds = 1 + usize::try_from(budget / state.max(1)).unwrap_or(usize::MAX);
        let (slots, rule) = match flag {
            Some(0 | 1) => (1, "plain"),
            Some(f) => (f.min(holds).max(1), "flag"),
            None if holds < 2 => (1, "plain"),
            None => (PARALLEL_CAP.min(holds), "budget"),
        };
        Parallel {
            slots,
            rule,
            budget,
            state,
        }
    }

    /// The `parallel` line's terms: `parallel rule=… slots=… park=…
    /// state=… cap=…`.
    #[must_use]
    pub fn line(&self) -> String {
        format!(
            "parallel rule={} slots={} park={} state={} cap={PARALLEL_CAP}",
            self.rule, self.slots, self.budget, self.state
        )
    }
}

/// What the engine thread is asked to do.
enum Cmd {
    Prefill(Vec<u32>),
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
    /// Nothing: the reply carries the position the model stands at.
    Pos,
    /// The residency back to its seed.
    ResidencyReset,
}

/// What a reply carries besides its result.
enum Extra {
    None,
    /// A `Keep`'s rule.
    Why(Option<String>),
    /// A `Splits`' cuts.
    Splits(Vec<usize>),
    /// A `Save`'s state.
    Saved(Arc<dyn Saved>),
    /// A `Pass`'s buffer, holding its kept tokens on success, and what its
    /// draft proposed and kept.
    Pass(Vec<u32>, Drafted),
    /// A `ResidencyReset`'s report; `None` for a seat with no residency.
    Residency(Option<ResidencyReset>),
}

/// Its answer: the argmax of a `Next` (the kept length of a `Keep`), the
/// logits buffer a `Next` was lent (filled unless the result is an error),
/// the position it stands at afterwards (the position it failed at, on an
/// error), and what a `Keep`, `Splits` or `Save` returns besides.
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
    /// The sequence state as a value the server's prompt cache holds.
    fn snapshot(&mut self) -> Result<Arc<dyn Saved>, GateError>;
    /// Replace the sequence state with `state`, which [`Seat::snapshot`] took
    /// of this seat.
    fn resume(&mut self, state: &dyn Saved) -> Result<(), GateError>;
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
}

impl SeatEngine {
    /// Start the engine thread, open the seat on it with `open` (which
    /// writes the load lines), and wait until it is loaded. `defined` is how
    /// many positions the body computes the model at; the engine serves the
    /// lesser of it and the cache. `card` names the device in a crash report;
    /// `props` is what `/props` reports about the engine ([`model_props`],
    /// [`placement_props`]), which the seat completes ([`Seat::props`]);
    /// `cache_ram` is the server's prompt cache budget
    /// (`serve::Engine::cache_ram`).
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
        let (opened, loaded) = mpsc::channel::<Result<(usize, usize, EngineProps), String>>();
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
                    .send(Ok((g.ctx_max(), g.pass_rows(), g.props(props))))
                    .is_err()
                {
                    return;
                }
                for cmd in cmds {
                    let (result, logits, extra) = match cmd {
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
                        Cmd::Save => match g.snapshot() {
                            Ok(s) => (Ok(0), None, Extra::Saved(s)),
                            Err(e) => (
                                Err(format!("snapshot at position {}: {e}", g.pos())),
                                None,
                                Extra::None,
                            ),
                        },
                        Cmd::ResidencyReset => match g.residency_reset() {
                            Ok(r) => (Ok(0), None, Extra::Residency(r)),
                            Err(e) => (
                                Err(format!("residency reset at position {}: {e}", g.pos())),
                                None,
                                Extra::None,
                            ),
                        },
                        Cmd::Resume(state) => (
                            g.resume(&*state).map(|()| 0).map_err(|e| {
                                format!("resume of a state of {} positions: {e}", state.n_tokens())
                            }),
                            None,
                            Extra::None,
                        ),
                        cmd => {
                            let (result, logits) = serve_cmd(&mut g, cmd, n_vocab);
                            (result, logits, Extra::None)
                        }
                    };
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
        let (ctx_max, rows, props) = match loaded.recv() {
            Ok(Ok((c, rows, props))) => (c.min(defined), rows, props),
            Ok(Err(e)) => return Err(e.into()),
            Err(mpsc::RecvError) => return Err("the engine thread ended during the load".into()),
        };
        Ok(SeatEngine {
            link: Link {
                tx: Some(tx),
                rx,
                logits: vec![0.0; n_vocab],
            },
            worker: Some(worker),
            vocab,
            ctx_max,
            rows,
            card,
            props,
            cache_ram,
            note: S::note,
            reply: None,
        })
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
        let sent = self.tx.as_ref().map(|tx| tx.send(cmd));
        if !matches!(sent, Some(Ok(()))) {
            return Err(EngineError("the engine thread is gone".to_owned()));
        }
        self.rx
            .recv()
            .map_err(|_| EngineError("the engine thread ended mid-call".to_owned()))
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
fn pass<S: Seat>(g: &mut S, last: u32, out: &mut Vec<u32>) -> Result<Drafted, String> {
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
        Cmd::Next { last, mut logits } => {
            let arg = next_row(g, last, logits.as_deref_mut(), n_vocab);
            return (arg, logits);
        }
        Cmd::Reset => g
            .reset()
            .map(|()| 0)
            .map_err(|e| format!("reset at position {at}: {e}")),
        Cmd::Rollback(pos) if pos as usize == at => Ok(0),
        Cmd::Rollback(pos) => g
            .rollback(pos)
            .map(|()| 0)
            .map_err(|e| format!("rollback to position {pos} from {at}: {e}")),
        Cmd::Keep(..)
        | Cmd::Splits { .. }
        | Cmd::Save
        | Cmd::Resume(_)
        | Cmd::Pass { .. }
        | Cmd::ResidencyReset => Err(
            "a keep, split, pass, save, resume or residency reset reached the step loop; the \
             engine thread \
             answers it"
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
    if row.len() != n_vocab {
        return Err(format!(
            "a logits buffer of {} at position {at}, the vocabulary has {n_vocab}",
            row.len()
        ));
    }
    let arg = g
        .step_row(last, row)
        .map_err(|e| format!("step at position {at}: {e}"))?;
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return Err(format!("logit {i} is NaN after position {at}"));
    }
    Ok(arg)
}

impl Engine for SeatEngine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.vocab.clone()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.link.call(Cmd::Prefill(ids.to_vec())).map(|_| ())
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

    fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
        let reply = self.link.ask(Cmd::Save)?;
        match (reply.result, reply.extra) {
            (Ok(_), Extra::Saved(s)) => Ok(s),
            (Ok(_), _) => Err(StateError::Engine(EngineError(
                "the engine thread answered a save with no state".to_owned(),
            ))),
            (Err(e), _) => Err(StateError::Engine(EngineError(e))),
        }
    }

    /// A state the body refuses leaves the model in no defined state: every
    /// refusal is the engine's error, which ends the server.
    fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
        self.link
            .call(Cmd::Resume(Arc::clone(state)))
            .map(|_| ())
            .map_err(StateError::Engine)
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

    /// A link to a thread that answers every `Next` as the engine thread does:
    /// it fills the lent row and hands it back with the argmax.
    fn echo_link() -> (Link, std::thread::JoinHandle<()>) {
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let worker = std::thread::spawn(move || {
            for (at, cmd) in cmds.into_iter().enumerate() {
                let Cmd::Next { last, mut logits } = cmd else {
                    panic!("the echo thread answers Next only");
                };
                if let Some(row) = logits.as_deref_mut() {
                    row.fill(last as f32);
                }
                let reply = Reply {
                    result: Ok(last + 1),
                    logits,
                    pos: at + 1,
                    extra: super::Extra::None,
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
        };
        (link, worker)
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

    /// The elastic slot count (the `Parallel` doc): the default what the
    /// budget holds capped, the flag an upper bound on the same, and every
    /// road's floor the plain engine.
    #[test]
    fn parallel_slots_hold_what_the_budget_holds() {
        use super::{CacheRam, HOST_BUDGET, PARALLEL_CAP, Parallel};
        let s = 1 << 30;
        // The default: one parked state beside the running slot, capped.
        assert_eq!(Parallel::of(None, 0, s).slots, 1);
        assert_eq!(Parallel::of(None, 1, s).slots, 1);
        for (park, want) in [
            (s, 2),
            (2 * s, 3),
            (9 * s, PARALLEL_CAP),
            (90 * s, PARALLEL_CAP),
        ] {
            let p = Parallel::of(None, park, s);
            assert_eq!((p.slots, p.rule), (want, "budget"), "park {park}");
        }
        // The flag: an upper bound on the same holds, never past it.
        for (flag, park, want) in [
            (1, 5 * s, 1),
            (2, 0, 1),
            (2, s, 2),
            (8, 2 * s, 3),
            (8, 90 * s, 8),
        ] {
            let p = Parallel::of(Some(flag), park, s);
            assert_eq!(p.slots, want, "flag {flag} park {park}");
        }
        // The elastic default's budget takes a set `--park-ram` as `park`
        // does, uncapped by the derived terms; without one it is the derived
        // headroom, capped, 0 when nothing is left.
        let ram = CacheRam {
            ram: 1 << 30,
            set: false,
            available: HOST_BUDGET + (5 << 30),
            need: 0,
            pool: 0,
        };
        assert_eq!(ram.park_or_none(None), 4 << 30);
        assert_eq!(ram.park_or_none(Some(2 << 30)), 2 << 30);
        assert_eq!(ram.park_or_none(Some(99 << 30)), 99 << 30);
        let tight = CacheRam {
            available: 1 << 20,
            ..ram
        };
        assert_eq!(tight.park_or_none(None), 0);
    }
}
