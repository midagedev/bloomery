//! Per-token log-probabilities, as llama-server serves them: `n_probs` on
//! `/completion` and `/completions` (`completion_probabilities`), `logprobs`
//! and `top_logprobs` on `/v1/chat/completions` and `/chat/completions`
//! (`choices[0].logprobs.content`), whole and streamed. This module owns the
//! request fields, the computation, the value it leaves per token and both
//! answers' JSON; the generation loop collects one value per generated token
//! ([`Collector`]) and the routes render them ([`Ask::entries`]).
//!
//! # Where the row comes from
//!
//! A token's values read the logits row its id was taken from. A request that
//! asks for probabilities therefore reads that row every token, and takes
//! every token through a plain step:
//!
//! | path | without probabilities | asking |
//! |---|---|---|
//! | plain greedy | `next(last, None)`, no row | `next(last, Some(row))` |
//! | plain sampled | `next(last, Some(row))`, the sampler's row | the same row, read once more |
//! | drafted greedy | [`Engine::advance`] passes, no row | plain `next(last, Some(row))` steps |
//! | drafted sampled | [`Engine::advance_sampled`] passes | plain `next(last, Some(row))` steps |
//! | a row of [`Engine::step_slots`] | the row's own buffer when it samples | the row's own buffer, the request's alone |
//!
//! A request that asks never passes ([`crate::genloop`]'s `takes_passes`), so
//! the drafted paths step: the row of every generated id is then one `next`
//! wrote. Its ids are the ones it takes without asking, by the engine's
//! contracts: `next` returns the same argmax whether it writes the row or not
//! ([`Engine::next`]); a greedy pass keeps, row by row, the target's argmax,
//! the first the one `next(last, None)` returns ([`Engine::advance`]), so the
//! steps take the ids the passes kept; a sampled pass draws exactly one id a
//! kept row from the sampler a plain run draws from, so a drafted and a plain
//! run of one seed take the same ids ([`Engine::advance_sampled`]). Beside
//! another busy slot a sampled request already steps. The probabilities read
//! the row and change nothing in it. The tests pin all four paths on the
//! mocks, and that an asking run on the drafting mock makes no pass.
//!
//! # What it costs
//!
//! A request that asks for nothing: nothing. Its generation holds no
//! [`Collector`]; each token meets one check of that `None` where the row is
//! read and one where the token's text goes out, and no allocation and no row
//! read are added. A request that asks allocates its row buffer (the
//! vocabulary's length) and its top-N scratch once, when it starts; its log
//! grows by amortised doubling. Per generated token, at a vocabulary of
//! V = 150K entries [derived, not measured]:
//! - the row written to the host: 4·V bytes, 600 KB over PCIe;
//! - the first pass (the NaN and +inf check, the max, the top N kept in a
//!   heap of N against its worst entry): V compares at 3–4 cycles an entry,
//!   ~0.5M cycles; the heap's inserts, about N·ln(V/N) of log₂N steps each,
//!   ~1K steps at N = 20 and ~5K at N = 100;
//! - the second pass, Σ exp(l − max) in f64 over f32 exps: V libm `expf`
//!   calls at 10–20 cycles, 1.5–3M cycles;
//! - in all 2–3.5M host cycles on the engine thread, serial after the step
//!   and under no other resource's shadow, so the token's wall grows by them,
//!   and so does every slot's that steps in the same round; the JSON is built
//!   on the request's own HTTP thread, beside the next step;
//! - on an engine that drafts, the passes lost: the request decodes one token
//!   a step where a pass keeps `1 + accepted`, so a draft whose passes keep k
//!   tokens on average leaves that request 1/k of its drafted rate — the
//!   largest of these terms on a drafting seat.
//!
//! A full sort of the vocabulary (V·log₂V ≈ 2.6M compares) is not needed: the
//! sum needs no order and the top N come from the heap.
//!
//! # The values
//!
//! Pre-sampling probabilities, llama-server's default
//! (`populate_token_probs` with `get_token_probabilities`): the softmax of the
//! raw row over the whole vocabulary, read before the loop sets a banned id's
//! logit to −inf (`ignore_eos`), as llama-server reads the logits before its
//! sampler's bias. A token's `logprob` is `l − max − ln Σ exp(l' − max)`, the
//! sum in f64 (llama-server sums in f32). The top N are ranked by logit,
//! highest first, a tie by the lower id (llama-server's `partial_sort` leaves
//! a tie's order unspecified); N above the vocabulary returns the vocabulary.
//! A −inf logit's probability is 0, whose logarithm is written as the lowest
//! f32, as llama-server writes it (JSON has no −inf). A row holding NaN or
//! +inf, or no finite logit, has no distribution: the request ends with a
//! named error ([`crate::genloop`]'s `GenError::Logprobs`), never a value.
//!
//! An entry's `token` and `bytes`:
//! - The generated id's own entry: the text that token released into the
//!   answer, llama-server's `text_to_send` — after the stop scan, so text held
//!   as a possible stop word is released (and named) by the token that ends
//!   the doubt, and a token whose bytes end inside a UTF-8 character releases
//!   `""` and the token that completes it the whole character. Its `bytes` are
//!   that text's. Joined, the entries' texts are the generated text.
//! - An alternative: its piece as the vocabulary's streaming decoder renders
//!   it alone — a special token as its text, as the generated text carries it
//!   (llama-server without `--special` writes a control token as `""`), an
//!   incomplete UTF-8 tail cut off (llama-server's `validate_utf8`). Its
//!   `bytes` are that text's when it is the piece exactly; a piece that is not
//!   valid UTF-8 on its own has `bytes: null`. The server's [`Tokenizer`] is
//!   text-only (a lossy decode), so those bytes cannot be read back, and a
//!   list that looked exact would be wrong.
//!
//! # Divergences from llama-server, and why
//!
//! - Every generated token has its entry, the end-of-generation one included,
//!   so a reply's entries count its `tokens_predicted`. llama-server gives a
//!   token that leaves an incomplete UTF-8 character no entry, and its
//!   non-stream answer drops one entry a token of the stop word's own
//!   tokenization; a client summing a reply's log-probabilities needs every
//!   token's.
//! - A drafted request's entries hold values: llama-server's speculative path
//!   leaves them unset.
//! - A stream sends one event a token while probabilities are asked, as
//!   llama-server does, its text possibly empty. On the chat path an event
//!   whose text makes no delta (a think tag the parser takes) keeps its entries
//!   for the next chunk that has a delta, where llama-server drops them; the
//!   last chunk carries what is left.
//! - Text a generation still held when it ended (a possible stop word, an
//!   incomplete character) is released by the end of generation's entry when
//!   the reply ends on one; a reply cut by its length releases it with no
//!   entry, as llama-server's held text has none either.
//! - `logprobs: true` with `top_logprobs: 0` gives each token its own entry
//!   and no alternatives (llama-server computes nothing at `n_probs = 0`).
//! - A whole answer that asked carries its field even when no token was
//!   generated (`[]`); llama-server leaves an empty one out.
//! - N is capped at [`MAX_TOP`] (llama-server caps nothing); `post_sampling_probs:
//!   true` is refused: the server's sampler hook ([`crate::Sampler`]) hands
//!   back the drawn id alone, not the candidates its chain kept.
//! - `logprobs` on `/completion` stays refused (llama-server reads it as an
//!   alias of `n_probs`): the route's field is `n_probs`, and OpenAI's text
//!   completion count form belongs to that API's own route.
//!
//! # Which routes render them
//!
//! [`completion_ask`] and [`chat_ask`] take the fields out of a body, so the
//! shared steps (`gen_params`) never see them on the two routes that render
//! the values (`completion_plan_logprobs`, `chat_plan_logprobs` in `api`).
//! Every other route builds its generation with `completion_plan` or
//! `chat_plan`, whose `gen_params` refuses `n_probs > 0`, `logprobs` other
//! than `false` and `top_logprobs > 0` by name until its answer renders them.
//! (`/v1/messages` converts an Anthropic body, which has no such field: its
//! conversion drops it as any unknown field.)
//!
//! # How the values reach the answer
//!
//! The engine thread writes each token's value into the request's [`Log`]
//! before it sends any event that counts the token, and the request's HTTP
//! thread reads the entries of the tokens an event counts (its timings'
//! `predicted_n`) — a stream's chunk carries exactly the tokens taken since the
//! last one — and the whole log for a whole answer. The log is shared through
//! the request's [`Ask`]: the engine thread's message to the HTTP thread
//! carries text and timings only.
//!
//! [`Engine::next`]: crate::Engine::next
//! [`Engine::advance`]: crate::Engine::advance
//! [`Engine::advance_sampled`]: crate::Engine::advance_sampled
//! [`Engine::step_slots`]: crate::Engine::step_slots

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{Arc, Mutex};

use serde_json::{Map, Value, json};

use super::{ApiError, get_b, get_i, invalid, json_type, relock};
use crate::engine::Tokenizer;

/// The most alternatives a request may ask for a token. llama-server clamps
/// `n_probs` to the vocabulary and caps nothing; at ~70 bytes of JSON an
/// alternative, a 150K vocabulary would make one token's answer ~10 MB. 100
/// bounds it near 7 KB, five times the chat path's default of 20.
pub(crate) const MAX_TOP: usize = 100;

/// The alternatives `logprobs: true` gives without `top_logprobs`
/// (llama-server's default).
const DEFAULT_TOP: usize = 20;

/// Why a row has no distribution.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RowError {
    #[error("the logits row holds NaN at id {0}")]
    Nan(u32),
    #[error("the logits row holds +inf at id {0}")]
    PosInf(u32),
    #[error("every logit of the row is -inf")]
    NoFinite,
    #[error("the generated id {id} lies past the row of {len} logits")]
    PastRow { id: u32, len: usize },
}

/// A request's probabilities: the alternatives it asked for a token, and the
/// log its generation writes on the engine thread and its answer reads.
#[derive(Clone, Debug)]
pub(crate) struct Ask {
    /// The alternatives asked for (`n_probs`, or `top_logprobs`).
    asked: usize,
    log: Arc<Mutex<Log>>,
}

/// One value per generated token, in order.
#[derive(Debug, Default)]
pub(crate) struct Log {
    /// The alternatives a token gets: what was asked, at most the vocabulary.
    top_n: usize,
    tokens: Vec<Taken>,
    /// Every token's alternatives, best first, back to back: token `k`'s run
    /// from the previous token's `top_end` to its own.
    top: Vec<(u32, f32)>,
    /// The text the tokens released, back to back: token `k`'s runs from the
    /// previous token's `text_end` to its own.
    text: String,
}

/// A generated token's value.
#[derive(Clone, Copy, Debug)]
struct Taken {
    id: u32,
    logprob: f32,
    text_end: usize,
    top_end: usize,
}

impl Ask {
    /// An ask for `asked` alternatives a token over a vocabulary of `n_vocab`.
    pub(crate) fn new(asked: usize, n_vocab: usize) -> Ask {
        Ask {
            asked,
            log: Arc::new(Mutex::new(Log {
                top_n: asked.min(n_vocab),
                ..Log::default()
            })),
        }
    }

    /// The alternatives asked for, as `generation_settings.n_probs` reports it.
    pub(crate) fn asked(&self) -> usize {
        self.asked
    }

    /// The engine thread's side: what reads each row and writes the log. Its
    /// scratch is allocated here, once.
    pub(crate) fn collector(&self) -> Collector {
        let top_n = relock(&self.log).top_n;
        Collector {
            log: Arc::clone(&self.log),
            top_n,
            heap: BinaryHeap::with_capacity(top_n),
            max: f32::NEG_INFINITY,
            ln_sum: 0.0,
            fault: None,
            taken: 0,
        }
    }

    /// The tokens written so far.
    pub(crate) fn len(&self) -> usize {
        relock(&self.log).tokens.len()
    }

    /// The entries of tokens `from..to`, as llama-server writes a
    /// pre-sampling entry, each alternative's text by `vocab`.
    ///
    /// # Panics
    ///
    /// When the log holds fewer than `to` tokens: an answer reads only the
    /// tokens an event counted, each written before the event was sent.
    pub(crate) fn entries(&self, from: usize, to: usize, vocab: &dyn Tokenizer) -> Vec<Value> {
        // Copied out under the lock and rendered after it: the engine thread
        // writes the next token's value meanwhile.
        let part = relock(&self.log).part(from, to);
        (0..part.tokens.len())
            .map(|k| part.entry(k, vocab))
            .collect()
    }

    /// Every token's entry: a whole answer's.
    pub(crate) fn whole(&self, vocab: &dyn Tokenizer) -> Vec<Value> {
        self.entries(0, self.len(), vocab)
    }
}

impl Log {
    /// Tokens `from..to` as a log of their own.
    fn part(&self, from: usize, to: usize) -> Log {
        assert!(
            from <= to && to <= self.tokens.len(),
            "the entries of tokens {from}..{to} were read from a log of {}",
            self.tokens.len()
        );
        let (text_from, top_from) = self.starts(from);
        let (text_to, top_to) = self.starts(to);
        Log {
            top_n: self.top_n,
            tokens: self.tokens[from..to]
                .iter()
                .map(|t| Taken {
                    text_end: t.text_end - text_from,
                    top_end: t.top_end - top_from,
                    ..*t
                })
                .collect(),
            top: self.top[top_from..top_to].to_vec(),
            text: self.text_of(text_from, text_to).to_owned(),
        }
    }

    /// Where token `k`'s text and alternatives start: where the token before
    /// it ends.
    fn starts(&self, k: usize) -> (usize, usize) {
        k.checked_sub(1).map_or((0, 0), |j| {
            (self.tokens[j].text_end, self.tokens[j].top_end)
        })
    }

    fn text_of(&self, from: usize, to: usize) -> &str {
        self.text
            .get(from..to)
            .expect("each token's text is a whole string appended, so it ends on a char boundary")
    }

    fn entry(&self, k: usize, vocab: &dyn Tokenizer) -> Value {
        let t = self.tokens[k];
        let (text_from, top_from) = self.starts(k);
        let text = self.text_of(text_from, t.text_end);
        let top: Vec<Value> = self.top[top_from..t.top_end]
            .iter()
            .map(|&(id, lp)| {
                let (text, bytes) = piece(vocab, id);
                json!({ "id": id, "token": text, "bytes": bytes, "logprob": logprob_json(lp) })
            })
            .collect();
        json!({
            "id": t.id,
            "token": text,
            "bytes": text.as_bytes(),
            "logprob": logprob_json(t.logprob),
            "top_logprobs": top,
        })
    }
}

/// `id`'s piece as the vocabulary's streaming decoder renders it alone, and
/// its bytes: those of that text when it is the piece exactly, `null` when
/// the decoder held an incomplete tail or replaced an invalid sequence.
fn piece(vocab: &dyn Tokenizer, id: u32) -> (String, Value) {
    let mut d = vocab.decoder();
    let text = d.push(id).unwrap_or_default();
    let exact = d.flush().is_empty() && !text.contains(char::REPLACEMENT_CHARACTER);
    let bytes = if exact {
        json!(text.as_bytes())
    } else {
        Value::Null
    };
    (text, bytes)
}

/// A log-probability as JSON: the logarithm of 0 is the lowest f32, as
/// llama-server writes it.
fn logprob_json(lp: f32) -> Value {
    json!(if lp.is_finite() { lp } else { f32::MIN })
}

/// The chat path's `logprobs` object around `entries`.
pub(super) fn chat_object(entries: Vec<Value>) -> Value {
    json!({ "content": entries })
}

/// An entry of the top-N heap; the greatest is the worst: a lower logit, or
/// at an equal one the higher id.
#[derive(Clone, Copy, Debug)]
struct Rank(f32, u32);

impl Ord for Rank {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(self.1.cmp(&o.1))
    }
}

impl PartialOrd for Rank {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

impl PartialEq for Rank {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}

impl Eq for Rank {}

/// The engine thread's side of an [`Ask`]: [`Collector::read`] reads the row
/// a generated id is taken from, [`Collector::take`] writes the id's value
/// with the text it released.
pub(crate) struct Collector {
    log: Arc<Mutex<Log>>,
    top_n: usize,
    /// The best `top_n` of the row read so far, the worst on top.
    heap: BinaryHeap<Rank>,
    max: f32,
    /// ln Σ exp(l − max) over the row read.
    ln_sum: f64,
    /// Why the row read has no distribution.
    fault: Option<RowError>,
    /// The tokens written.
    taken: usize,
}

impl Collector {
    /// Reads `row`, the logits the next generated id is taken from: its max,
    /// its log-sum and its best `top_n`, or why it has none.
    pub(crate) fn read(&mut self, row: &[f32]) {
        self.heap.clear();
        self.fault = None;
        let mut max = f32::NEG_INFINITY;
        for (i, &l) in row.iter().enumerate() {
            let id = u32::try_from(i).expect("a vocabulary fits u32");
            if l.is_nan() {
                self.fault = Some(RowError::Nan(id));
                return;
            }
            if l == f32::INFINITY {
                self.fault = Some(RowError::PosInf(id));
                return;
            }
            max = max.max(l);
            if self.heap.len() < self.top_n {
                self.heap.push(Rank(l, id));
            } else if let Some(mut worst) = self.heap.peek_mut()
                && Rank(l, id) < *worst
            {
                *worst = Rank(l, id);
            }
        }
        if max == f32::NEG_INFINITY {
            self.fault = Some(RowError::NoFinite);
            return;
        }
        let sum: f64 = row.iter().map(|&l| f64::from((l - max).exp())).sum();
        self.max = max;
        self.ln_sum = sum.ln();
    }

    /// Writes generated `id`'s value from the row last read (`row`, which
    /// still holds `id`'s logit as the engine wrote it) with `text`, what the
    /// token released into the answer; the row's fault, if it had one.
    pub(crate) fn take(&mut self, id: u32, row: &[f32], text: &str) -> Result<(), RowError> {
        if let Some(e) = self.fault.take() {
            return Err(e);
        }
        let logit = usize::try_from(id)
            .ok()
            .and_then(|i| row.get(i))
            .copied()
            .ok_or(RowError::PastRow { id, len: row.len() })?;
        let (max, ln_sum) = (self.max, self.ln_sum);
        let logprob = |l: f32| (f64::from(l - max) - ln_sum) as f32;
        let ranked = std::mem::take(&mut self.heap).into_sorted_vec();
        {
            let mut log = relock(&self.log);
            log.top
                .extend(ranked.iter().map(|&Rank(l, alt)| (alt, logprob(l))));
            log.text.push_str(text);
            let t = Taken {
                id,
                logprob: logprob(logit),
                text_end: log.text.len(),
                top_end: log.top.len(),
            };
            log.tokens.push(t);
        }
        let mut scratch = ranked;
        scratch.clear();
        self.heap = BinaryHeap::from(scratch);
        self.taken += 1;
        Ok(())
    }

    /// The tokens written so far.
    pub(crate) fn taken(&self) -> usize {
        self.taken
    }
}

/// `/completion`'s probabilities: `n_probs`, taken out of `b` (so the shared
/// steps never refuse it). `n_probs` absent or `<= 0` asks for none, as in
/// llama-server; past [`MAX_TOP`] it is a 400 naming it, and so is
/// `post_sampling_probs: true` beside an ask.
pub(super) fn completion_ask(
    b: &mut Map<String, Value>,
    n_vocab: usize,
) -> Result<Option<Ask>, ApiError> {
    let n = get_i(b, "n_probs")?;
    b.remove("n_probs");
    let Some(n) = n.filter(|&n| n > 0) else {
        return Ok(None);
    };
    let n = usize::try_from(n)
        .ok()
        .filter(|&n| n <= MAX_TOP)
        .ok_or_else(|| {
            invalid(format!(
                "n_probs {n} is past the {MAX_TOP} alternatives a token this server returns"
            ))
        })?;
    if get_b(b, "post_sampling_probs")? == Some(true) {
        return Err(post_sampling());
    }
    Ok(Some(Ask::new(n, n_vocab)))
}

/// The chat path's probabilities: `logprobs` and `top_logprobs`, taken out of
/// `b`. `logprobs: true` asks for `top_logprobs` alternatives a token (default
/// 20, at most [`MAX_TOP`]); llama-server's two refusals hold
/// (`top_logprobs` without `logprobs: true`, and `logprobs` beside tools in a
/// stream), and a `logprobs` that is not a boolean, a `top_logprobs` that is
/// not an integer in range, and `post_sampling_probs: true` beside an ask are
/// 400s naming them.
pub(super) fn chat_ask(
    b: &mut Map<String, Value>,
    n_vocab: usize,
) -> Result<Option<Ask>, ApiError> {
    let top_set = b.get("top_logprobs").filter(|v| !v.is_null()).cloned();
    let top = get_i(b, "top_logprobs")?;
    let lp = b.remove("logprobs").filter(|v| !v.is_null());
    b.remove("top_logprobs");
    match lp {
        None | Some(Value::Bool(false)) => match top_set {
            None => Ok(None),
            Some(_) => Err(invalid("top_logprobs requires logprobs to be set to true")),
        },
        Some(Value::Bool(true)) => {
            let tools = b
                .get("tools")
                .and_then(Value::as_array)
                .is_some_and(|a| !a.is_empty());
            if tools && get_b(b, "stream")? == Some(true) {
                return Err(invalid("logprobs is not supported with tools + stream"));
            }
            let n = match (top_set, top) {
                (None, _) => DEFAULT_TOP,
                (Some(_), Some(n)) if (0..=MAX_TOP as i64).contains(&n) => {
                    usize::try_from(n).expect("a count in 0..=MAX_TOP")
                }
                (Some(v), _) => {
                    return Err(invalid(format!(
                        "top_logprobs must be an integer in 0..={MAX_TOP}, not {v}"
                    )));
                }
            };
            if get_b(b, "post_sampling_probs")? == Some(true) {
                return Err(post_sampling());
            }
            Ok(Some(Ask::new(n, n_vocab)))
        }
        Some(v) => Err(invalid(format!(
            "logprobs must be a boolean on the chat path, not {}",
            json_type(&v)
        ))),
    }
}

fn post_sampling() -> ApiError {
    invalid(
        "post_sampling_probs is not supported by this server: its sampler hands back the drawn \
         id alone, not the candidates its chain kept",
    )
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use serde_json::{Map, Value, json};

    use super::super::{Server, chat_plan, completion_plan, relock, testserve};
    use super::{Ask, MAX_TOP, RowError};
    use crate::engine::{Engine, EngineError, SlotRow, Tokenizer};
    use crate::mock::{DraftMock, MockCall, MockEngine, MockTokenizer, ScriptedEngine};
    use crate::sched::SlotConfig;

    const BOUND: Duration = Duration::from_secs(10);

    /// The mock's vocabulary: its six specials, then one id a byte.
    const V: usize = 262;

    /// The mock's id of an ASCII letter.
    fn id_of(c: char) -> u64 {
        6 + u64::from(u32::from(c))
    }

    /// ln Σ exp(l − max) over a mock row: its prediction at 4 (the max), `k`
    /// other followers at 2, the rest of the vocabulary at −8.
    fn mock_ln_sum(k: usize) -> f64 {
        let rest = (V - 1 - k) as f64;
        (1.0 + k as f64 * (-2f64).exp() + rest * (-12f64).exp()).ln()
    }

    fn served(engine: Box<dyn Engine>) -> SocketAddr {
        testserve::spawn(engine, testserve::mock_config()).0
    }

    fn post(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
        let (status, text) = testserve::roundtrip(addr, "POST", path, &[], &body.to_string());
        let v = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text}"));
        (status, v)
    }

    /// A stream's `data:` events, `[DONE]` left out.
    fn events(addr: SocketAddr, path: &str, body: &Value) -> Vec<Value> {
        let (status, text) = testserve::roundtrip(addr, "POST", path, &[], &body.to_string());
        assert_eq!(status, 200, "{text}");
        text.split("\n\n")
            .filter_map(|e| e.strip_prefix("data: "))
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("{e}: {d}")))
            .collect()
    }

    fn arr(v: &Value) -> &Vec<Value> {
        v.as_array().unwrap_or_else(|| panic!("not an array: {v}"))
    }

    /// An entry's alternatives as (id, logprob).
    fn alts(e: &Value) -> Vec<(u64, f64)> {
        arr(&e["top_logprobs"])
            .iter()
            .map(|a| {
                (
                    a["id"].as_u64().expect("an id"),
                    a["logprob"].as_f64().expect("a logprob"),
                )
            })
            .collect()
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() < 1e-5
    }

    /// The value one read row gives generated `id`: its entry, `top` alternatives asked.
    fn entry_of(row: &[f32], top: usize, id: u32) -> Result<Value, RowError> {
        let ask = Ask::new(top, row.len());
        let mut c = ask.collector();
        c.read(row);
        c.take(id, row, "x")?;
        Ok(ask.whole(&MockTokenizer).remove(0))
    }

    /// A token's log-probability is its logit less the row's max and log-sum,
    /// and its alternatives are the best `top` of the row in order, each with
    /// its own.
    #[test]
    fn a_token_s_values_are_its_row_s_softmax() {
        let row = [1.0f32, 3.0, 2.0, 0.0];
        let e = entry_of(&row, 2, 2).expect("a distribution");
        let ln_sum = (1f64 + (-1f64).exp() + (-2f64).exp() + (-3f64).exp()).ln();
        assert_eq!(e["id"], 2);
        assert!(
            close(e["logprob"].as_f64().expect("a logprob"), -1.0 - ln_sum),
            "{e}"
        );
        let got = alts(&e);
        assert_eq!(got.iter().map(|a| a.0).collect::<Vec<_>>(), [1, 2], "{e}");
        assert!(
            close(got[0].1, -ln_sum) && close(got[1].1, -1.0 - ln_sum),
            "{e}"
        );
    }

    /// Equal logits rank by the lower id, wherever the heap met them.
    #[test]
    fn equal_logits_rank_by_the_lower_id() {
        let row = [2.0f32, 5.0, 5.0, 2.0, 5.0, 1.0];
        let e = entry_of(&row, 4, 4).expect("a distribution");
        let ids: Vec<u64> = alts(&e).iter().map(|a| a.0).collect();
        assert_eq!(ids, [1, 2, 4, 0], "{e}");
    }

    /// A −inf logit is a probability of 0, written as the lowest f32 (JSON
    /// has no −inf), ranked last; the finite ids split the rest.
    #[test]
    fn a_minus_inf_logit_is_probability_zero_at_the_lowest_f32() {
        let row = [0.0f32, f32::NEG_INFINITY, 0.0];
        let e = entry_of(&row, 3, 1).expect("a distribution");
        assert_eq!(e["logprob"].as_f64(), Some(f64::from(f32::MIN)), "{e}");
        let got = alts(&e);
        assert_eq!(
            got.iter().map(|a| a.0).collect::<Vec<_>>(),
            [0, 2, 1],
            "{e}"
        );
        assert!(
            close(got[0].1, -(2f64.ln())) && close(got[1].1, -(2f64.ln())),
            "{e}"
        );
        assert_eq!(got[2].1, f64::from(f32::MIN), "{e}");
    }

    /// A row with no distribution — a NaN, a +inf, no finite logit — or a
    /// generated id past the row is a named error, never a value.
    #[test]
    fn a_row_without_a_distribution_is_a_named_error() {
        let cases: [(&[f32], u32, RowError); 4] = [
            (&[0.0, f32::NAN, 1.0], 0, RowError::Nan(1)),
            (&[0.0, 1.0, f32::INFINITY], 0, RowError::PosInf(2)),
            (&[f32::NEG_INFINITY; 3], 0, RowError::NoFinite),
            (&[0.0, 1.0], 2, RowError::PastRow { id: 2, len: 2 }),
        ];
        for (row, id, want) in cases {
            assert_eq!(entry_of(row, 2, id).err(), Some(want.clone()), "{row:?}");
        }
    }

    /// A token gets the alternatives asked for, at most the vocabulary; none
    /// asked leaves its own value.
    #[test]
    fn the_alternatives_are_the_ask_up_to_the_vocabulary() {
        let row = [0.5f32, -1.0, 2.0];
        assert_eq!(
            alts(&entry_of(&row, 10, 0).expect("a distribution")).len(),
            3
        );
        let none = entry_of(&row, 0, 0).expect("a distribution");
        assert!(alts(&none).is_empty(), "{none}");
        assert!(none["logprob"].as_f64().is_some_and(|l| l < 0.0), "{none}");
    }

    /// The greedy `/completion` of "abacaba" on the mock: each generated
    /// token's entry carries its id, its text and bytes, its log-probability
    /// and the best three alternatives — all derived from the mock's levels
    /// (4 for its prediction, 2 for another follower, −8 for the rest) — one
    /// entry a generated token.
    #[test]
    fn completion_n_probs_carries_each_token_s_values() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let body = json!({"prompt": "abacaba", "n_predict": 4, "temperature": 0, "n_probs": 3});
        let (status, v) = post(addr, "/completion", &body);
        assert_eq!(status, 200, "{v}");
        let entries = arr(&v["completion_probabilities"]);
        assert_eq!(
            entries.len(),
            v["tokens_predicted"].as_u64().expect("a count") as usize
        );
        // Each step: the token, the follower beside it at 2, if any.
        let steps = [('b', Some('c')), ('a', None), ('b', Some('c')), ('a', None)];
        for (e, (tok, other)) in entries.iter().zip(steps) {
            let ln_sum = mock_ln_sum(usize::from(other.is_some()));
            assert_eq!(e["id"].as_u64(), Some(id_of(tok)), "{e}");
            assert_eq!(e["token"], tok.to_string(), "{e}");
            assert_eq!(e["bytes"], json!([u32::from(tok)]), "{e}");
            assert!(
                close(e["logprob"].as_f64().expect("a logprob"), -ln_sum),
                "{e}"
            );
            let mut want = vec![(id_of(tok), -ln_sum)];
            want.extend(other.map(|c| (id_of(c), -2.0 - ln_sum)));
            want.extend((0..).map(|id| (id, -12.0 - ln_sum)).take(3 - want.len()));
            let got = alts(e);
            assert_eq!(got.len(), 3, "{e}");
            for (g, w) in got.iter().zip(&want) {
                assert!(g.0 == w.0 && close(g.1, w.1), "{e}: want {want:?}");
            }
        }
        assert_eq!(v["generation_settings"]["n_probs"], 3, "{v}");
    }

    /// Streamed, each token's entry rides its own chunk, one a chunk, the
    /// final event none; the entries are the whole answer's, and their texts
    /// join to the streamed content.
    #[test]
    fn completion_n_probs_streams_each_token_with_its_chunk() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let body = json!({"prompt": "abacaba", "n_predict": 6, "temperature": 0, "n_probs": 2});
        let (_, whole) = post(addr, "/completion", &body);
        let mut b = body.clone();
        b["stream"] = json!(true);
        let evs = events(addr, "/completion", &b);
        let (last, chunks) = evs.split_last().expect("a final event");
        assert_eq!(last["stop"], true, "{last}");
        assert!(last.get("completion_probabilities").is_none(), "{last}");
        let mut entries = Vec::new();
        let mut text = String::new();
        for c in chunks {
            let e = arr(&c["completion_probabilities"]);
            assert_eq!(e.len(), 1, "{c}");
            text.push_str(c["content"].as_str().expect("content"));
            entries.extend(e.iter().cloned());
        }
        assert_eq!(&entries, arr(&whole["completion_probabilities"]));
        let joined: String = entries
            .iter()
            .map(|e| e["token"].as_str().expect("a token"))
            .collect();
        assert_eq!(joined, text);
        assert_eq!(text, whole["content"]);
    }

    fn chat_body(stream: bool) -> Value {
        json!({"messages": [{"role": "user", "content": "abacaba"}], "max_tokens": 6,
               "temperature": 0, "logprobs": true, "top_logprobs": 3, "stream": stream})
    }

    /// The chat path's `choices[0].logprobs.content`: the entries the
    /// completion path gives the same prompt, and their texts join to the
    /// message's content.
    #[test]
    fn chat_logprobs_carry_each_token_s_values() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let (status, v) = post(addr, "/v1/chat/completions", &chat_body(false));
        assert_eq!(status, 200, "{v}");
        let content = arr(&v["choices"][0]["logprobs"]["content"]);
        let cmpl = json!({"prompt": "<user>\nabacaba<assistant>\n", "n_predict": 6,
                          "temperature": 0, "n_probs": 3});
        let (_, c) = post(addr, "/completion", &cmpl);
        assert_eq!(content, arr(&c["completion_probabilities"]), "{v}");
        let joined: String = content
            .iter()
            .map(|e| e["token"].as_str().expect("a token"))
            .collect();
        assert_eq!(joined, v["choices"][0]["message"]["content"], "{v}");
        assert!(content.iter().all(|e| alts(e).len() == 3), "{v}");
    }

    /// Streamed, every chunk that carries content carries its tokens'
    /// entries; together they are the whole answer's, and their texts join to
    /// the streamed content.
    #[test]
    fn chat_logprobs_stream_with_the_content() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let (_, whole) = post(addr, "/v1/chat/completions", &chat_body(false));
        let mut entries = Vec::new();
        let mut text = String::new();
        for c in events(addr, "/v1/chat/completions", &chat_body(true)) {
            let Some(choice) = c["choices"].get(0) else {
                continue;
            };
            if let Some(t) = choice["delta"]["content"].as_str() {
                assert!(choice.get("logprobs").is_some(), "{c}");
                text.push_str(t);
            }
            if let Some(e) = choice.get("logprobs") {
                entries.extend(arr(&e["content"]).iter().cloned());
            }
        }
        assert_eq!(&entries, arr(&whole["choices"][0]["logprobs"]["content"]));
        let joined: String = entries
            .iter()
            .map(|e| e["token"].as_str().expect("a token"))
            .collect();
        assert_eq!(joined, text);
    }

    /// A chat event whose text makes no delta — a think tag the parser takes
    /// — keeps its tokens' entries for the next chunk that has a delta, and
    /// the last chunk carries what is left (the end of generation's): no
    /// generated token's entry is dropped.
    #[test]
    fn chat_entries_without_a_delta_ride_the_next_chunk() {
        let addr = served(Box::new(ScriptedEngine::new(64, "<think>ab</think>cd")));
        let mut body = chat_body(true);
        body["max_tokens"] = json!(16);
        let chunks: Vec<(Value, Vec<u64>)> = events(addr, "/v1/chat/completions", &body)
            .into_iter()
            .filter_map(|c| {
                let choice = c["choices"].get(0)?.clone();
                let ids = choice
                    .get("logprobs")
                    .map(|l| {
                        arr(&l["content"])
                            .iter()
                            .map(|e| e["id"].as_u64().expect("an id"))
                            .collect()
                    })
                    .unwrap_or_default();
                Some((choice, ids))
            })
            .collect();
        let carried: Vec<(&Value, &Vec<u64>)> = chunks
            .iter()
            .filter(|(_, ids)| !ids.is_empty())
            .map(|(choice, ids)| (&choice["delta"], ids))
            .collect();
        let (open, close) = (4, 5);
        assert_eq!(
            carried,
            [
                (&json!({"reasoning_content": "a"}), &vec![open, id_of('a')]),
                (&json!({"reasoning_content": "b"}), &vec![id_of('b')]),
                (&json!({"content": "c"}), &vec![close, id_of('c')]),
                (&json!({"content": "d"}), &vec![id_of('d')]),
                (&json!({}), &vec![1]),
            ],
            "{chunks:?}"
        );
    }

    fn message_of(addr: SocketAddr, path: &str, body: &Value) -> (u16, String) {
        let (status, v) = post(addr, path, body);
        (
            status,
            v["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .to_owned(),
        )
    }

    /// llama-server's two refusals on the chat path, in its words:
    /// `top_logprobs` without `logprobs: true` (absent, `false`, or a 0), and
    /// `logprobs` beside tools in a stream; tools without a stream serve.
    #[test]
    fn the_chat_path_refuses_as_llama_server_does() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let base = json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2});
        let with = |extra: Value| {
            let mut b = base.clone();
            for (k, v) in extra.as_object().expect("an object") {
                b[k] = v.clone();
            }
            b
        };
        for extra in [
            json!({"top_logprobs": 3}),
            json!({"top_logprobs": 0}),
            json!({"logprobs": false, "top_logprobs": 2}),
        ] {
            let (status, m) = message_of(addr, "/v1/chat/completions", &with(extra.clone()));
            assert_eq!(
                (status, m.as_str()),
                (400, "top_logprobs requires logprobs to be set to true"),
                "{extra}"
            );
        }
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {}}}]);
        let streamed =
            with(json!({"logprobs": true, "tools": tools, "tool_choice": "none", "stream": true}));
        let (status, m) = message_of(addr, "/v1/chat/completions", &streamed);
        assert_eq!(
            (status, m.as_str()),
            (400, "logprobs is not supported with tools + stream")
        );
        let whole = with(json!({"logprobs": true, "tools": tools, "tool_choice": "none"}));
        let (status, v) = post(addr, "/v1/chat/completions", &whole);
        assert_eq!(status, 200, "{v}");
        assert!(v["choices"][0]["logprobs"]["content"].is_array(), "{v}");
    }

    /// How many alternatives a token gets: `top_logprobs` absent gives
    /// llama-server's 20, 0 gives each token its own entry and none, up to
    /// [`MAX_TOP`] serves and past it is a 400 naming the field; `n_probs` the
    /// same on `/completion`, where 0 or less asks for nothing.
    #[test]
    fn the_alternatives_asked_are_bounded_by_name() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let chat = |top: Option<i64>| {
            let mut b = json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2, "logprobs": true});
            if let Some(n) = top {
                b["top_logprobs"] = json!(n);
            }
            b
        };
        let counts = |v: &Value| -> Vec<usize> {
            arr(&v["choices"][0]["logprobs"]["content"])
                .iter()
                .map(|e| alts(e).len())
                .collect()
        };
        for (top, want) in [(None, 20), (Some(0), 0), (Some(MAX_TOP as i64), MAX_TOP)] {
            let (status, v) = post(addr, "/v1/chat/completions", &chat(top));
            assert_eq!(status, 200, "{v}");
            let got = counts(&v);
            assert!(
                !got.is_empty() && got.iter().all(|&n| n == want),
                "{top:?}: {got:?}"
            );
        }
        for top in [MAX_TOP as i64 + 1, -1] {
            let (status, m) = message_of(addr, "/v1/chat/completions", &chat(Some(top)));
            assert_eq!(status, 400, "{top}: {m}");
            assert!(m.contains("top_logprobs"), "{top}: {m}");
        }
        let cmpl = |n: i64| json!({"prompt": "ab", "n_predict": 2, "n_probs": n});
        let (status, v) = post(addr, "/completion", &cmpl(MAX_TOP as i64));
        assert_eq!(status, 200, "{v}");
        assert!(
            arr(&v["completion_probabilities"])
                .iter()
                .all(|e| alts(e).len() == MAX_TOP)
        );
        let (status, m) = message_of(addr, "/completion", &cmpl(MAX_TOP as i64 + 1));
        assert_eq!(status, 400, "{m}");
        assert!(m.contains("n_probs"), "{m}");
        for n in [0, -3] {
            let (status, v) = post(addr, "/completion", &cmpl(n));
            assert_eq!(status, 200, "{v}");
            assert!(v.get("completion_probabilities").is_none(), "{n}: {v}");
            assert_eq!(v["generation_settings"]["n_probs"], 0, "{v}");
        }
    }

    /// A field this server cannot honour beside an ask is a 400 naming it:
    /// `post_sampling_probs: true` (the sampler hook hands back an id, no
    /// candidates), a `logprobs` that is not a boolean on the chat path, a
    /// `top_logprobs` that is not an integer. `post_sampling_probs` without an
    /// ask changes nothing and is served.
    #[test]
    fn unservable_probability_fields_are_named_400s() {
        let addr = served(Box::new(MockEngine::new(4096)));
        let msgs = json!([{"role": "user", "content": "ab"}]);
        let refused = [
            (
                "/completion",
                json!({"prompt": "ab", "n_probs": 2, "post_sampling_probs": true}),
                "post_sampling_probs",
            ),
            (
                "/v1/chat/completions",
                json!({"messages": msgs, "logprobs": true, "post_sampling_probs": true}),
                "post_sampling_probs",
            ),
            (
                "/v1/chat/completions",
                json!({"messages": msgs, "logprobs": "yes"}),
                "logprobs",
            ),
            (
                "/v1/chat/completions",
                json!({"messages": msgs, "logprobs": true, "top_logprobs": "3"}),
                "top_logprobs",
            ),
            (
                "/v1/chat/completions",
                json!({"messages": msgs, "logprobs": true, "top_logprobs": 2.5}),
                "top_logprobs",
            ),
        ];
        for (path, mut body, field) in refused {
            body["max_tokens"] = json!(2);
            let (status, m) = message_of(addr, path, &body);
            assert_eq!(status, 400, "{body}: {m}");
            assert!(m.contains(field), "{body}: {m}");
        }
        let unasked = json!({"prompt": "ab", "n_predict": 2, "post_sampling_probs": true});
        assert_eq!(post(addr, "/completion", &unasked).0, 200);
    }

    /// The plans every route that renders no probabilities builds on refuse
    /// each probability field by name; `/v1/messages`, whose Anthropic body
    /// has no such field, drops it in its conversion like any unknown field
    /// and answers without values.
    #[test]
    fn routes_that_render_no_probabilities_refuse_them_at_their_plan() {
        let (addr, state, _ended) =
            testserve::spawn(Box::new(MockEngine::new(4096)), testserve::mock_config());
        let obj = |v: Value| -> Map<String, Value> { v.as_object().expect("an object").clone() };
        for (field, value) in [
            ("n_probs", json!(2)),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(2)),
        ] {
            let mut chat = obj(json!({"messages": [{"role": "user", "content": "ab"}]}));
            chat.insert(field.to_owned(), value.clone());
            let e = chat_plan(&state, &chat, None).err().map(|e| e.message);
            assert_eq!(e, Some(format!("{field} is not supported by this server")));
            let mut cmpl = obj(json!({"prompt": "ab"}));
            cmpl.insert(field.to_owned(), value);
            let e = completion_plan(&state, &cmpl).err().map(|e| e.message);
            assert_eq!(e, Some(format!("{field} is not supported by this server")));
        }
        let messages = json!({"messages": [{"role": "user", "content": "ab"}], "max_tokens": 2,
                              "logprobs": true, "top_logprobs": 2});
        let (status, text) =
            testserve::roundtrip(addr, "POST", "/v1/messages", &[], &messages.to_string());
        assert_eq!(status, 200, "{text}");
        assert!(!text.contains("logprob"), "{text}");
    }

    /// The `/completion` of `body` on a fresh server of `engine`: its ids.
    fn ids(engine: Box<dyn Engine>, body: &Value) -> Vec<u64> {
        let mut b = body.clone();
        b["return_tokens"] = json!(true);
        let (status, v) = post(served(engine), "/completion", &b);
        assert_eq!(status, 200, "{v}");
        arr(&v["tokens"])
            .iter()
            .map(|t| t.as_u64().expect("an id"))
            .collect()
    }

    /// Asking changes no generated id: greedy and sampled (one seed), on the
    /// plain mock and on the drafting one, whose asking run makes no pass —
    /// the passes the same request makes without asking keep the ids the
    /// asking run's steps take.
    #[test]
    fn asking_changes_no_generated_id() {
        let greedy = json!({"prompt": "abacadaeabacada", "n_predict": 24, "temperature": 0});
        let sampled =
            json!({"prompt": "abacadaeabacada", "n_predict": 24, "temperature": 0.9, "seed": 7});
        for body in [&greedy, &sampled] {
            let mut asking = body.clone();
            asking["n_probs"] = json!(4);
            let plain = ids(Box::new(MockEngine::new(4096)), body);
            assert_eq!(
                ids(Box::new(MockEngine::new(4096)), &asking),
                plain,
                "{body}"
            );
            let (quiet, loud) = (
                Arc::new(Mutex::new(Vec::new())),
                Arc::new(Mutex::new(Vec::new())),
            );
            let drafted = ids(
                Box::new(DraftMock::new(4096).logged(Arc::clone(&quiet))),
                body,
            );
            let asked = ids(
                Box::new(DraftMock::new(4096).logged(Arc::clone(&loud))),
                &asking,
            );
            assert_eq!((&drafted, &asked), (&plain, &plain), "{body}");
            let passes = |log: &Mutex<Vec<MockCall>>| {
                relock(log).iter().filter(|c| **c != MockCall::Next).count()
            };
            assert!(
                passes(&quiet) > 0,
                "{body}: the request passes without asking"
            );
            assert_eq!(passes(&loud), 0, "{body}: an asking request steps");
        }
    }

    /// A gate the first prompt call waits behind, and the widths (and the
    /// rows given a logits buffer) of every step of several slots.
    #[derive(Default)]
    struct Gate {
        state: Mutex<(usize, bool)>,
        cv: Condvar,
        widths: Mutex<Vec<(usize, usize)>>,
    }

    impl Gate {
        fn enter(&self) {
            let mut g = relock(&self.state);
            g.0 += 1;
            self.cv.notify_all();
            while !g.1 {
                g = self.cv.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        }

        fn reached(&self) -> bool {
            let g = relock(&self.state);
            let (g, _) = self
                .cv
                .wait_timeout_while(g, BOUND, |s| s.0 == 0)
                .unwrap_or_else(|e| e.into_inner());
            g.0 > 0
        }

        fn open(&self) {
            relock(&self.state).1 = true;
            self.cv.notify_all();
        }
    }

    /// The two-slot mock behind a [`Gate`] in `prefill`.
    struct Gated(MockEngine, Arc<Gate>);

    impl Engine for Gated {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.1.enter();
            self.0.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.0.next(last, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.0.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.0.keepable(n)
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            self.0.cut(n)
        }
        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }
        fn describe(&self) -> String {
            self.0.describe()
        }
        fn slots(&self) -> usize {
            self.0.slots()
        }
        fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
            self.0.select_slot(slot)
        }
        fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
            let with = rows.iter().filter(|r| r.logits.is_some()).count();
            relock(&self.1.widths).push((rows.len(), with));
            self.0.step_slots(rows)
        }
    }

    /// A `/completion` of `body` sent on a connection of its own, whose
    /// answer [`answer`] reads later.
    fn send(addr: SocketAddr, body: &Value) -> TcpStream {
        let body = body.to_string();
        let mut s = TcpStream::connect(addr).expect("connect");
        let req = format!(
            "POST /completion HTTP/1.0\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).expect("write");
        s
    }

    /// The answer [`send`]'s connection carries: its status and body, read to
    /// the connection's end.
    fn answer(mut s: TcpStream) -> (u16, Value) {
        let mut raw = String::new();
        s.read_to_string(&mut raw).expect("read");
        let (head, body) = raw.split_once("\r\n\r\n").expect("a head");
        let status = head
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .expect("a status line");
        (
            status,
            serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}")),
        )
    }

    /// Two asking requests stepped together in one call of two rows: each
    /// gets the values its own row gives, the ones it gets alone on a server
    /// of one slot.
    #[test]
    fn two_slots_each_get_their_own_row_s_values() {
        let gate = Arc::new(Gate::default());
        let engine = Gated(MockEngine::new(4096).with_slots(2), Arc::clone(&gate));
        let slots = SlotConfig {
            parallel: 2,
            ..SlotConfig::default()
        };
        let server = Server::bind_with(
            "127.0.0.1:0",
            Box::new(engine),
            testserve::mock_config(),
            slots,
        )
        .expect("bind");
        let addr = server.local_addr().expect("addr");
        let (state, _ended) = server.start().expect("start");
        let body = |prompt: &str| {
            json!({"prompt": prompt, "n_predict": 12, "temperature": 0, "n_probs": 3,
                   "cache_prompt": false})
        };
        let (a, b) = (body("abcabcab"), body("acbacbac"));
        let first = send(addr, &a);
        assert!(gate.reached(), "the first prompt never reached the engine");
        let second = send(addr, &b);
        let deadline = Instant::now() + BOUND;
        while relock(&state.shared.board).waiting() == 0 {
            assert!(Instant::now() < deadline, "the second request never queued");
            thread::sleep(Duration::from_millis(5));
        }
        gate.open();
        let (ra, rb) = (answer(first), answer(second));
        assert_eq!((ra.0, rb.0), (200, 200), "{} | {}", ra.1, rb.1);
        assert!(
            relock(&gate.widths).contains(&(2, 2)),
            "no step of both rows with their buffers: {:?}",
            relock(&gate.widths)
        );
        for (body, together) in [(&a, &ra.1), (&b, &rb.1)] {
            let alone = post(served(Box::new(MockEngine::new(4096))), "/completion", body).1;
            assert_eq!(
                together["completion_probabilities"], alone["completion_probabilities"],
                "{body}"
            );
        }
    }

    /// An entry's text is what its token released: a token that ends inside a
    /// character releases `""` and the one that completes it the character; a
    /// possible stop word held to the end is released by the end of
    /// generation's own entry. An alternative's text is its piece (a special
    /// token's text), and its `bytes` are `null` when the piece is not valid
    /// UTF-8 on its own.
    #[test]
    fn an_entry_s_text_is_what_its_token_released() {
        let addr = served(Box::new(ScriptedEngine::new(64, "é")));
        let (status, v) = post(addr, "/completion", &json!({"prompt": "x", "n_probs": 2}));
        assert_eq!(status, 200, "{v}");
        let e = arr(&v["completion_probabilities"]);
        let texts: Vec<(&str, &Value)> = e
            .iter()
            .map(|e| (e["token"].as_str().expect("a token"), &e["bytes"]))
            .collect();
        assert_eq!(
            texts,
            [
                ("", &json!([])),
                ("é", &json!([195, 169])),
                ("", &json!([]))
            ],
            "{v}"
        );
        assert_eq!(e[2]["id"], 1, "the end of generation's own entry: {v}");
        let first = &e[0]["top_logprobs"];
        assert_eq!(
            (&first[0]["token"], &first[0]["bytes"]),
            (&json!(""), &Value::Null),
            "{v}"
        );
        assert_eq!(first[1]["token"], "<｜begin▁of▁sentence｜>", "{v}");
        assert_eq!(
            first[1]["bytes"],
            json!("<｜begin▁of▁sentence｜>".as_bytes()),
            "{v}"
        );
        let second = &e[1]["top_logprobs"][0];
        assert_eq!(
            (&second["token"], &second["bytes"]),
            (&json!("\u{FFFD}"), &Value::Null),
            "{v}"
        );
        let addr = served(Box::new(ScriptedEngine::new(64, "abE")));
        let (_, v) = post(
            addr,
            "/completion",
            &json!({"prompt": "x", "n_probs": 1, "stop": ["END"]}),
        );
        let texts: Vec<&str> = arr(&v["completion_probabilities"])
            .iter()
            .map(|e| e["token"].as_str().expect("a token"))
            .collect();
        assert_eq!(texts, ["a", "b", "", "E"], "{v}");
        assert_eq!(v["content"], "abE", "{v}");
    }

    /// The mock, its rows poisoned with a NaN at id 5 from its `from`-th step on.
    struct Poisoned(MockEngine, usize);

    impl Engine for Poisoned {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.0.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            let mut out = out;
            let g = self.0.next(last, out.as_deref_mut())?;
            self.1 = self.1.saturating_sub(1);
            if self.1 == 0
                && let Some(row) = out
            {
                row[5] = f32::NAN;
            }
            Ok(g)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.0.reset()
        }
        fn ctx_max(&self) -> usize {
            self.0.ctx_max()
        }
        fn describe(&self) -> String {
            self.0.describe()
        }
    }

    /// A NaN in a row the probabilities read ends that request with a 500
    /// naming the token and the id — at the prompt's step or a later one —
    /// and the server serves the next request.
    #[test]
    fn a_nan_row_ends_the_request_by_name_and_not_the_server() {
        for (from, token) in [(1, 0), (3, 2)] {
            let addr = served(Box::new(Poisoned(MockEngine::new(4096), from)));
            let body = json!({"prompt": "abcabc", "n_predict": 6, "temperature": 0, "n_probs": 2});
            let (status, m) = message_of(addr, "/completion", &body);
            assert_eq!(status, 500, "{m}");
            assert_eq!(
                m,
                format!(
                    "the probabilities of generated token {token}: the logits row holds NaN at id 5"
                )
            );
            let quiet = json!({"prompt": "abcabc", "n_predict": 2});
            assert_eq!(post(addr, "/completion", &quiet).0, 200);
        }
    }
}
