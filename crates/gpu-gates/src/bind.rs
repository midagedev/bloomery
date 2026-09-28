//! The server's engine over a [`Seat`]: [`Ds41Engine`] implements
//! `serve::Engine`, [`Vocab`] implements `serve::Tokenizer` over the file's
//! own vocabulary, and [`sampler_factory`] builds each request's sampler from
//! the sampler crate.
//!
//! The seat — the model, its draft if one runs, and the body's rules — lives
//! on one thread of its own, for the whole run. The server calls its engine
//! from whichever connection thread holds the slot; the model's device state
//! and the pinned dispatcher slot belong to the thread that opened them, so
//! every step runs on that thread and the engine is a handle that sends it
//! commands. The opener comes in as a closure, and the seat as a trait the
//! binary implements, because this library does not name a device crate (see
//! [`crate::generate`]) or the session crate.
//!
//! How long a prefix of the cache can be kept, and why no longer, is the
//! seat's rule, which the engine thread answers ([`Ds41Engine`]'s `keepable`
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
use model::placement::{Device, ModelTensors, Plan, Role};
use sampler::{Sampler, SamplerParams};
use serve::{
    CacheNote, Decoder, DeviceProps, Drafted, Engine, EngineError, EngineProps, ModelProps,
    PlacementProps, SamplerFactory, SamplingParams, Saved, StateError, Tokenizer,
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

/// The nvidia-smi index of the card this process opens by the placement name
/// `name` (a device whose name contains it, as the GPU loader finds its
/// card): among nvidia-smi's devices, those so named, narrowed to
/// `CUDA_VISIBLE_DEVICES` when it lists UUIDs. Exactly one, or an error that
/// says what nvidia-smi listed.
pub fn nvidia_smi_index(name: &str) -> Result<u32, String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,uuid", "--format=csv,noheader"])
        .output()
        .map_err(|e| format!("nvidia-smi: {e}"))?;
    if !out.status.success() {
        return Err(format!("nvidia-smi: {}", out.status));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let visible: Option<Vec<String>> = std::env::var("CUDA_VISIBLE_DEVICES").ok().and_then(|v| {
        let ids: Vec<String> = v.split(',').map(|s| s.trim().to_owned()).collect();
        ids.iter().all(|s| s.starts_with("GPU-")).then_some(ids)
    });
    let named: Vec<u32> = text
        .lines()
        .filter_map(|line| {
            let (index, rest) = line.split_once(',')?;
            let (gpu_name, uuid) = rest.rsplit_once(',')?;
            let uuid = uuid.trim();
            let shown = visible
                .as_ref()
                .is_none_or(|v| v.iter().any(|id| uuid.starts_with(id.as_str())));
            if !(gpu_name.contains(name) && shown) {
                return None;
            }
            index.trim().parse().ok()
        })
        .collect();
    match named.as_slice() {
        &[index] => Ok(index),
        _ => Err(format!(
            "{} devices named like {name:?} are visible, not one; nvidia-smi listed:\n{text}",
            named.len()
        )),
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
    /// The longest prefix of at most this many positions a rollback keeps,
    /// and the rule that kept less.
    Keep(usize),
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
/// ([`Ds41Engine::spawn`] opens it on the thread). A failed call says what
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
    /// Prints what the server's prompt cache did, as the binary's records.
    fn note(note: &CacheNote)
    where
        Self: Sized;
}

/// `serve::Engine` over a [`Seat`] that lives on its own thread. The
/// position is the model's alone: every question about it goes to the thread.
pub struct Ds41Engine {
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

impl Ds41Engine {
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
    ) -> Result<Ds41Engine, GateError>
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
                        Cmd::Keep(n) => {
                            let (k, why) = g.keep(n.min(g.pos()));
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
        Ok(Ds41Engine {
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
        Cmd::Keep(_) | Cmd::Splits { .. } | Cmd::Save | Cmd::Resume(_) | Cmd::Pass { .. } => Err(
            "a keep, split, pass, save or resume reached the step loop; the engine thread \
             answers it"
                .to_owned(),
        ),
        Cmd::Pos => Ok(0),
    };
    (result, None)
}

/// One step and, into `row`, its logits. The row is checked here: one of the
/// wrong length or holding a NaN is the step's error, not the sampler's to
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
    let arg = g
        .step(last)
        .map_err(|e| format!("step at position {at}: {e}"))?;
    let Some(row) = row else {
        return Ok(arg);
    };
    if row.len() != n_vocab {
        return Err(format!(
            "a logits buffer of {} at position {at}, the vocabulary has {n_vocab}",
            row.len()
        ));
    }
    g.logits_into(row)
        .map_err(|e| format!("logits after position {at}: {e}"))?;
    if let Some(i) = row.iter().position(|v| v.is_nan()) {
        return Err(format!("logit {i} is NaN after position {at}"));
    }
    Ok(arg)
}

impl Engine for Ds41Engine {
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

    /// The body's rule, asked on the engine thread (`spawn`'s `keep`). A
    /// thread that does not answer grants nothing: the caller resets, and the
    /// next call reports the thread.
    fn keepable(&self, n: usize) -> usize {
        match self.link.ask(Cmd::Keep(n)).map(|r| r.result) {
            Ok(Ok(k)) => (k as usize).min(n),
            _ => 0,
        }
    }

    /// The body's rule, as `keepable` asks it; a thread that does not answer
    /// is named as the reason.
    fn keep_limit(&self, n: usize) -> Option<String> {
        match self.link.ask(Cmd::Keep(n)) {
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

impl Drop for Ds41Engine {
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
}
