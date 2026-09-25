//! The server's engine over a [`Generator`]: [`Ds41Engine`] implements
//! `serve::Engine`, [`Vocab`] implements `serve::Tokenizer` over the file's
//! own vocabulary, and [`sampler_factory`] builds each request's sampler from
//! the sampler crate.
//!
//! The generator lives on one thread of its own, for the whole run. The server
//! calls its engine from whichever connection thread holds the slot; the
//! model's device state and the pinned dispatcher slot belong to the thread
//! that opened them, so every step runs on that thread and the engine is a
//! handle that sends it commands. The opener comes in as a closure because
//! this library does not name a device crate (see [`crate::generate`]).
//!
//! How long a prefix of the cache can be kept is the body's rule, which the
//! opener hands in beside the model ([`Ds41Engine::spawn`]'s `keep`) and the
//! engine thread answers ([`Ds41Engine`]'s `keepable`).
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

use bloomery_gpu::model::ChainBody;
use bloomery_gpu::{GpuError, GpuModel};
use gguf::Split;
use model::placement::{Device, ModelTensors, Plan, Role};
use sampler::{Sampler, SamplerParams};
use serve::{
    Decoder, DeviceProps, Engine, EngineError, EngineProps, ModelProps, PlacementProps,
    SamplerFactory, SamplingParams, Tokenizer,
};

use crate::GateError;
use crate::generate::Generator;

/// The file's vocabulary as the server reads it.
pub struct Vocab {
    tok: Arc<tokenizer::Tokenizer>,
    bos: u32,
    eos: u32,
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
        })
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

    fn add_bos(&self) -> bool {
        self.tok.add_bos()
    }

    fn n_vocab(&self) -> usize {
        self.tok.n_vocab()
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
/// card, named `gpus[c]` (`GPU<n>`, see [`nvidia_smi_index`]), the resident
/// bytes of its segments by class and its stage's layers; then the host's
/// (`CPU`) the same way; and the cards' KV bytes, cache and shadow. Only the
/// plan's own rows are summed, so a card's classes add up to its
/// `dense_bytes + expert_bytes` and the host's to its `expert_bytes +
/// table_bytes`. The NVMe tier holds no resident bytes and is no device here.
pub fn placement_props(plan: &Plan<'_>, gpus: &[String]) -> Result<PlacementProps, String> {
    if gpus.len() != plan.cards.len() {
        return Err(format!(
            "{} card names for a plan of {} cards",
            gpus.len(),
            plan.cards.len()
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
    let mut devices: Vec<DeviceProps> = cards
        .into_iter()
        .zip(gpus)
        .zip(&plan.machine.cards)
        .map(|((class_bytes, gpu), card)| DeviceProps {
            device: gpu.clone(),
            class_bytes,
            layers: Some(layer_range(&card.layers)),
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
    Next {
        last: u32,
        logits: bool,
    },
    Reset,
    /// Take back the positions from this one on.
    Rollback(u32),
    /// The longest prefix of at most this many positions a rollback keeps.
    Keep(usize),
}

/// Its answer: the argmax and the logits of a `Next` (the kept length of a
/// `Keep`), and the position it stands at afterwards (the position it failed
/// at, on an error).
struct Reply {
    result: Result<(u32, Option<Vec<f32>>), String>,
    pos: usize,
}

/// `serve::Engine` over a [`Generator`] that lives on its own thread.
pub struct Ds41Engine {
    tx: Option<Sender<Cmd>>,
    rx: Receiver<Reply>,
    worker: Option<JoinHandle<()>>,
    vocab: Arc<Vocab>,
    ctx_max: usize,
    card: String,
    props: EngineProps,
    pos: usize,
}

impl Ds41Engine {
    /// Start the engine thread, open the model on it with `open` (which
    /// writes the load lines), and wait until it is loaded. `keep` is the
    /// body's rule for `keepable`: the longest prefix of at most `n` positions
    /// the model's rollback keeps, from where it stands. `prefill` is the
    /// body's prompt feed: it feeds the ids from where the model stands and
    /// returns the argmax after the last (a batched body's batch, or
    /// `GpuModel::step`). `card` names the device in a crash report; `props`
    /// is what `/props` reports about the engine ([`model_props`],
    /// [`placement_props`]).
    pub fn spawn<B, F, K, P>(
        open: F,
        keep: K,
        prefill: P,
        vocab: Arc<Vocab>,
        card: String,
        props: EngineProps,
    ) -> Result<Ds41Engine, GateError>
    where
        B: ChainBody + 'static,
        F: FnOnce() -> Result<Generator<B>, GateError> + Send + 'static,
        K: Fn(&GpuModel<B>, usize) -> usize + Send + 'static,
        P: Fn(&mut GpuModel<B>, &[u32]) -> Result<u32, GpuError> + Send + 'static,
    {
        let (tx, cmds) = mpsc::channel::<Cmd>();
        let (replies, rx) = mpsc::channel::<Reply>();
        let (opened, loaded) = mpsc::channel::<Result<usize, String>>();
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
                if opened.send(Ok(g.ctx_max())).is_err() {
                    return;
                }
                for cmd in cmds {
                    let result = match cmd {
                        Cmd::Keep(n) => u32::try_from(keep(g.model(), n))
                            .map(|k| (k, None))
                            .map_err(|_| format!("a kept prefix of at most {n} passes u32")),
                        cmd => serve_cmd(&mut g, cmd, n_vocab, &prefill),
                    };
                    if replies
                        .send(Reply {
                            result,
                            pos: g.pos(),
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            })?;
        let ctx_max = match loaded.recv() {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return Err(e.into()),
            Err(mpsc::RecvError) => return Err("the engine thread ended during the load".into()),
        };
        Ok(Ds41Engine {
            tx: Some(tx),
            rx,
            worker: Some(worker),
            vocab,
            ctx_max,
            card,
            props,
            pos: 0,
        })
    }

    fn call(&mut self, cmd: Cmd) -> Result<(u32, Option<Vec<f32>>), EngineError> {
        let reply = self.ask(cmd)?;
        self.pos = reply.pos;
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
}

/// A body's prompt feed ([`Ds41Engine::spawn`]'s `prefill`): the ids from
/// where the model stands, the argmax after the last.
type PrefillFn<B> = dyn Fn(&mut GpuModel<B>, &[u32]) -> Result<u32, GpuError>;

/// One command on the engine thread. A logits read is checked here: a row of
/// the wrong length or one holding a NaN is the step's error, not the
/// sampler's to absorb.
fn serve_cmd<B: ChainBody>(
    g: &mut Generator<B>,
    cmd: Cmd,
    n_vocab: usize,
    prefill: &PrefillFn<B>,
) -> Result<(u32, Option<Vec<f32>>), String> {
    let at = g.pos();
    match cmd {
        Cmd::Prefill(ids) => {
            let refuse =
                |e: String| format!("prefill of {} ids from position {at}: {e}", ids.len());
            if ids.is_empty() || at + ids.len() > g.ctx_max() {
                return Err(refuse(format!(
                    "the ids must be at least one and fit the context {}",
                    g.ctx_max()
                )));
            }
            prefill(g.model_mut(), &ids)
                .map(|a| (a, None))
                .map_err(|e| refuse(e.to_string()))
        }
        Cmd::Next { last, logits } => {
            let arg = g
                .step(last)
                .map_err(|e| format!("step at position {at}: {e}"))?;
            if !logits {
                return Ok((arg, None));
            }
            let row = g
                .logits()
                .map_err(|e| format!("logits after position {at}: {e}"))?;
            if row.len() != n_vocab {
                return Err(format!(
                    "the head gave {} logits at position {at}, the vocabulary has {n_vocab}",
                    row.len()
                ));
            }
            if let Some(i) = row.iter().position(|v| v.is_nan()) {
                return Err(format!("logit {i} is NaN after position {at}"));
            }
            Ok((arg, Some(row)))
        }
        Cmd::Reset => g
            .model_mut()
            .reset()
            .map(|()| (0, None))
            .map_err(|e| format!("reset at position {at}: {e}")),
        Cmd::Rollback(pos) => g
            .model_mut()
            .rollback(pos)
            .map(|()| (0, None))
            .map_err(|e| format!("rollback to position {pos} from {at}: {e}")),
        Cmd::Keep(n) => Err(format!(
            "a keep query of {n} reached the step loop; the engine thread answers it"
        )),
    }
}

impl Engine for Ds41Engine {
    fn tokenizer(&self) -> Arc<dyn Tokenizer> {
        self.vocab.clone()
    }

    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.call(Cmd::Prefill(ids.to_vec())).map(|_| ())
    }

    fn next(&mut self, last: u32, logits_out: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let want = logits_out.is_some();
        let (arg, row) = self.call(Cmd::Next { last, logits: want })?;
        if let (Some(out), Some(row)) = (logits_out, row) {
            if out.len() != row.len() {
                return Err(EngineError(format!(
                    "the caller's logits buffer holds {}, the head gave {}",
                    out.len(),
                    row.len()
                )));
            }
            out.copy_from_slice(&row);
        }
        Ok(arg)
    }

    fn reset(&mut self) -> Result<(), EngineError> {
        self.call(Cmd::Reset).map(|_| ())
    }

    /// The body's rule, asked on the engine thread (`spawn`'s `keep`). A
    /// thread that does not answer grants nothing: the caller resets, and the
    /// next call reports the thread.
    fn keepable(&self, n: usize) -> usize {
        let n = n.min(self.pos);
        match self.ask(Cmd::Keep(n)).map(|r| r.result) {
            Ok(Ok((k, _))) => (k as usize).min(n),
            _ => 0,
        }
    }

    fn cut(&mut self, n: usize) -> Result<(), EngineError> {
        if n == self.pos {
            return Ok(());
        }
        let pos =
            u32::try_from(n).map_err(|_| EngineError(format!("cut to position {n}: past u32")))?;
        self.call(Cmd::Rollback(pos)).map(|_| ())
    }

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        format!("card={} position={}", self.card, self.pos)
    }

    fn props_engine(&self) -> EngineProps {
        self.props.clone()
    }
}

impl Drop for Ds41Engine {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop; the model is dropped there.
        self.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}
