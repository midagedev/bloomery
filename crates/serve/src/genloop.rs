//! The generation loop shared by `/completion` and `/v1/chat/completions`.
//!
//! `timings` are the server's wall clock around the engine calls, the way
//! llama-server reports them to bench clients. They are not admissible
//! measurements for this repository: those come only from the lease runners.
//! Like llama-server, the prompt phase ends when the first token's logits are
//! out, and the predicted phase runs from there to the end, so `predicted_ms`
//! spans `predicted_n - 1` decode steps.

use std::io;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};

use crate::engine::{Engine, EngineError, Sampler, SamplerFactory, SamplingParams, Tokenizer};
use crate::stop::StopScan;

/// Request knobs after parsing, llama-server defaults filled in.
#[derive(Clone, Debug)]
pub(crate) struct GenParams {
    /// `-1` is unbounded (up to the context).
    pub n_predict: i64,
    pub sampling: SamplingParams,
    pub stop: Vec<String>,
    pub ignore_eos: bool,
    pub stream: bool,
    pub timings_per_token: bool,
    pub return_progress: bool,
    pub include_usage: bool,
    /// Keep the longest cached prefix of the prompt (llama-server's default `true`).
    pub cache_prompt: bool,
}

/// llama-server's `timings` object, as the server clocked it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Timings {
    /// Prompt tokens evaluated by this request: the prompt less what the
    /// cache kept (`cache_n`).
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub n_ctx: usize,
    pub n_past: usize,
    /// Prompt positions kept from the previous request instead of evaluated.
    pub cache_n: usize,
    /// The whole prompt, kept and evaluated (llama-server's `n_prompt_tokens`;
    /// not a `timings` field).
    pub n_prompt: usize,
}

impl Timings {
    /// The JSON llama-server emits. A zero count gives NaN per-token fields,
    /// which serialize as `null` exactly as nlohmann writes them.
    pub(crate) fn to_json(&self) -> Value {
        let (pn, dn) = (self.prompt_n as f64, self.predicted_n as f64);
        json!({
            "prompt_n": self.prompt_n,
            "prompt_ms": self.prompt_ms,
            "prompt_per_token_ms": self.prompt_ms / pn,
            "prompt_per_second": 1e3 / self.prompt_ms * pn,
            "predicted_n": self.predicted_n,
            "predicted_ms": self.predicted_ms,
            "predicted_per_token_ms": self.predicted_ms / dn,
            "predicted_per_second": 1e3 / self.predicted_ms * dn,
            "n_ctx": self.n_ctx,
            "n_past": self.n_past,
            "cache_n": self.cache_n,
        })
    }
}

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StopKind {
    Eos,
    Word,
    Limit,
}

impl StopKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            StopKind::Eos => "eos",
            StopKind::Word => "word",
            StopKind::Limit => "limit",
        }
    }

    /// OpenAI `finish_reason`.
    pub(crate) fn finish_reason(self) -> &'static str {
        match self {
            StopKind::Eos | StopKind::Word => "stop",
            StopKind::Limit => "length",
        }
    }
}

/// A finished generation.
pub(crate) struct Outcome {
    pub content: String,
    /// Every generated id, the end-of-generation one included.
    pub tokens: Vec<u32>,
    pub stop: StopKind,
    pub stopping_word: String,
    pub truncated: bool,
    pub timings: Timings,
}

/// What the loop hands its sink.
pub(crate) enum Event<'a> {
    /// The prompt is evaluated (llama-server's `prompt_progress`, sent once, complete).
    Prompt(&'a Timings),
    /// Text safe to stream.
    Text(&'a str, &'a Timings),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GenError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The sink failed: the client went away.
    #[error("client: {0}")]
    Client(#[from] io::Error),
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The logits buffer the engine fills, or `None` on a greedy request (empty buffer).
fn out(logits: &mut [f32]) -> Option<&mut [f32]> {
    (!logits.is_empty()).then_some(logits)
}

fn choose(sampler: &mut Option<Sampler>, greedy: u32, logits: &[f32], history: &[u32]) -> u32 {
    match sampler {
        Some(s) => s(logits, history),
        None => greedy,
    }
}

/// The one slot: the engine and the ids its cache holds, one per position.
pub(crate) struct Slot {
    pub engine: Box<dyn Engine>,
    vocab: Arc<dyn Tokenizer>,
    /// Every id the engine has evaluated since its last reset, in order. The
    /// last generated id of a request is not in it: it is never fed back.
    held: Vec<u32>,
}

impl Slot {
    pub(crate) fn new(engine: Box<dyn Engine>) -> Slot {
        Slot {
            vocab: engine.tokenizer(),
            engine,
            held: Vec::new(),
        }
    }

    /// Brings the cache to the longest prefix of `ids` it can keep, leaving at
    /// least the last id for `next`, and returns that length. `want` false
    /// (`cache_prompt: false`) always resets. While an engine call is in flight
    /// `held` is empty, so a failed call leaves no claim about the cache.
    fn reuse(&mut self, ids: &[u32], want: bool) -> Result<usize, EngineError> {
        let common = if want {
            self.held
                .iter()
                .zip(ids)
                .take_while(|(a, b)| a == b)
                .count()
        } else {
            0
        };
        let ask = common.min(ids.len() - 1);
        let k = self.engine.keepable(ask).min(ask);
        let held = std::mem::take(&mut self.held);
        if k == 0 {
            self.engine.reset()?;
        } else if k < held.len() {
            self.engine.cut(k)?;
        }
        self.held = held;
        self.held.truncate(k);
        Ok(k)
    }

    /// `prefill` that books what it fed.
    fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
        let held = std::mem::take(&mut self.held);
        self.engine.prefill(ids)?;
        self.held = held;
        self.held.extend_from_slice(ids);
        Ok(())
    }

    /// `next` that books what it fed.
    fn next(&mut self, last: u32, logits: Option<&mut [f32]>) -> Result<u32, EngineError> {
        let held = std::mem::take(&mut self.held);
        let g = self.engine.next(last, logits)?;
        self.held = held;
        self.held.push(last);
        Ok(g)
    }
}

/// Runs one request on the slot. `ids` is non-empty and shorter than the context.
/// `timings` in the returned outcome are filled even when the sink failed midway
/// (the caller's counters still see the work done). A greedy request never asks the
/// engine for its logits. `tick` sees the timings after every generated token.
pub(crate) fn generate(
    slot: &mut Slot,
    factory: &SamplerFactory,
    ids: &[u32],
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    tick: &mut dyn FnMut(&Timings),
    timings_out: &mut Timings,
) -> Result<Outcome, GenError> {
    let n = ids.len();
    let ctx_max = slot.engine.ctx_max();
    let vocab = Arc::clone(&slot.vocab);
    let mut sampler = (p.sampling.temperature > 0.0).then(|| factory(&p.sampling));
    let mut logits = vec![
        0.0f32;
        if sampler.is_some() {
            vocab.n_vocab()
        } else {
            0
        }
    ];
    let cache_n = slot.reuse(ids, p.cache_prompt)?;
    let t0 = Instant::now();
    slot.prefill(&ids[cache_n..n - 1])?;
    let greedy = slot.next(ids[n - 1], out(&mut logits))?;
    let tim = timings_out;
    *tim = Timings {
        prompt_n: n - cache_n,
        prompt_ms: ms_since(t0),
        n_ctx: ctx_max,
        n_past: n,
        cache_n,
        n_prompt: n,
        ..Timings::default()
    };
    let t1 = Instant::now();
    if p.return_progress {
        sink(Event::Prompt(tim))?;
    }
    let budget = usize::try_from(p.n_predict).unwrap_or(usize::MAX);
    let eos = vocab.eos();
    let mut scan = StopScan::new(p.stop.clone());
    let mut dec = vocab.decoder();
    let mut generated: Vec<u32> = Vec::new();
    let mut stop = StopKind::Limit;
    let mut stopping_word = String::new();
    let mut truncated = false;
    if budget > 0 {
        let mut tok = choose(&mut sampler, greedy, &logits, &generated);
        loop {
            generated.push(tok);
            tim.predicted_n = generated.len();
            tim.predicted_ms = ms_since(t1);
            tick(tim);
            if tok == eos && !p.ignore_eos {
                stop = StopKind::Eos;
                break;
            }
            if let Some(piece) = dec.push(tok) {
                let pushed = scan.push(&piece);
                if !pushed.send.is_empty() {
                    sink(Event::Text(&pushed.send, tim))?;
                }
                if let Some(w) = pushed.stopped {
                    stop = StopKind::Word;
                    stopping_word = w;
                    break;
                }
            }
            if generated.len() >= budget {
                break;
            }
            // Feeding `tok` takes position n + len - 1; the cache holds ctx_max.
            if n + generated.len() > ctx_max {
                truncated = true;
                break;
            }
            let g = slot.next(tok, out(&mut logits))?;
            tim.n_past = n + generated.len();
            tok = choose(&mut sampler, g, &logits, &generated);
        }
    }
    if stop != StopKind::Word {
        let tail = dec.flush();
        let pushed = scan.push(&tail);
        let mut rest = pushed.send;
        if let Some(w) = pushed.stopped {
            stop = StopKind::Word;
            stopping_word = w;
        } else {
            rest.push_str(&scan.finish());
        }
        if !rest.is_empty() {
            sink(Event::Text(&rest, tim))?;
        }
    }
    tim.predicted_ms = ms_since(t1);
    Ok(Outcome {
        content: scan.text().to_owned(),
        tokens: generated,
        stop,
        stopping_word,
        truncated,
        timings: tim.clone(),
    })
}
