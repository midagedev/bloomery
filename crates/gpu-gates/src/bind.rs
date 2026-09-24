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

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use bloomery_gpu::model::ChainBody;
use sampler::{Sampler, SamplerParams};
use serve::{Decoder, Engine, EngineError, SamplerFactory, SamplingParams, Tokenizer};

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

/// What the engine thread is asked to do.
enum Cmd {
    Prefill(Vec<u32>),
    Next { last: u32, logits: bool },
    Reset,
}

/// Its answer: the argmax and the logits of a `Next`, and the position it
/// stands at afterwards (the position it failed at, on an error).
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
    pos: usize,
}

impl Ds41Engine {
    /// Start the engine thread, open the model on it with `open` (which
    /// writes the load lines), and wait until it is loaded. `card` names the
    /// device in a crash report.
    pub fn spawn<B, F>(open: F, vocab: Arc<Vocab>, card: String) -> Result<Ds41Engine, GateError>
    where
        B: ChainBody + 'static,
        F: FnOnce() -> Result<Generator<B>, GateError> + Send + 'static,
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
                    let result = serve_cmd(&mut g, cmd, n_vocab);
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
            pos: 0,
        })
    }

    fn call(&mut self, cmd: Cmd) -> Result<(u32, Option<Vec<f32>>), EngineError> {
        let sent = self.tx.as_ref().map(|tx| tx.send(cmd));
        if !matches!(sent, Some(Ok(()))) {
            return Err(EngineError("the engine thread is gone".to_owned()));
        }
        let reply = self
            .rx
            .recv()
            .map_err(|_| EngineError("the engine thread ended mid-call".to_owned()))?;
        self.pos = reply.pos;
        reply.result.map_err(EngineError)
    }
}

/// One command on the engine thread. A logits read is checked here: a row of
/// the wrong length or one holding a NaN is the step's error, not the
/// sampler's to absorb.
fn serve_cmd<B: ChainBody>(
    g: &mut Generator<B>,
    cmd: Cmd,
    n_vocab: usize,
) -> Result<(u32, Option<Vec<f32>>), String> {
    let at = g.pos();
    match cmd {
        Cmd::Prefill(ids) => g
            .prefill(&ids)
            .map(|a| (a, None))
            .map_err(|e| format!("prefill of {} ids from position {at}: {e}", ids.len())),
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

    fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    fn describe(&self) -> String {
        format!("card={} position={}", self.card, self.pos)
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
