//! The generation loop shared by `/completion` and `/v1/chat/completions`.
//!
//! `timings` are the server's wall clock around the engine calls, the way
//! llama-server reports them to bench clients. They are not admissible
//! measurements for this repository: those come only from the lease runners.
//! Like llama-server, the prompt phase ends when the first token's logits are
//! out, and the predicted phase runs from there to the end, so `predicted_ms`
//! spans `predicted_n - 1` decode steps. `cache_ms`, which llama-server does
//! not report, is the phase before the prompt's: the prompt cache's work on
//! the slot (its state saved, a cached state put back, the cut), which
//! `prompt_ms` does not count. Nor does it count, on a prompt the engine
//! thread runs a call a round ([`Prompt`]), the other slots' decode rounds
//! between its calls: `prompt_ms`, and so `prompt_per_second`, are the
//! prompt's own engine time alone.
//!
//! An engine that drafts ([`Engine::advance_rows`] past 1) takes every greedy
//! token after the first through [`Engine::advance`]: a pass keeps one token
//! or more, which the loop then takes one at a time as it takes a step's. A
//! pass that would run past the context is not run; the positions left take
//! one step each, so the ids end where a plain run's end. `timings` then carry
//! llama-server's `draft_n` and `draft_n_accepted`: the ids the passes proposed
//! and kept. A request that reads its rows ([`Pick::reads_rows`]: it draws,
//! asks for probabilities or follows a program) takes its passes through
//! [`Engine::advance_sampled`] on an engine that drafts sampled requests
//! ([`Engine::drafts_sampled`]), the pass lent the request's pick ([`Pick`])
//! as its sampler: each kept id is the pick a step would make from that
//! position's verified row, its value read from that row. A request that
//! draws samples, or changes its rows: its `logit_bias` added and, under
//! `ignore_eos`, the stop ids at -inf, then its sampler or the argmax
//! ([`biased`]). Beside another busy slot its pass is a row of the round's
//! [`Engine::advance_slots`] with its draw ([`Gen::pass_row`]), its ids the
//! ones its pass alone draws. On another drafting engine such a request
//! reads the logits row every token, which a pass does not give, so it
//! takes one [`Engine::next`] a token and carries no draft counts.
//!
//! A request that asks for probabilities ([`GenParams::logprobs`]) reads the
//! row of every token, a step's or a pass's: each row is read before the
//! loop changes it, and each taken id's value goes to the request's log with
//! the text it released ([`crate::api::logprobs`]). Its stream sends one
//! event a token, the text possibly empty, and the end of generation's own;
//! a request that asks for nothing meets one check of its fixed `None` where
//! a step asks for a row, one where it is read and one where a token's text
//! goes out.
//!
//! A request that follows a program ([`GenParams::constraint`]) takes each
//! token as the program's pick from its row, a step's or a pass's
//! ([`crate::constrain::Walk`]) — a forced id as it stands, otherwise the
//! largest logit among the ids the program allows there, or for a request that
//! samples or penalizes its sampler's draw with every other id masked out —
//! and once the program is spent the vocabulary's stop ids are the only ones
//! left. The request's `logit_bias` is added before the mask, so it reorders
//! allowed ids and lets in no other; a fixed id takes no bias. The row is
//! read only where the pick needs it: a token the program
//! fixes (a forced id, the only child of a choice's node) asks the engine for
//! none unless the request wants probabilities. A pick the program cannot make ends the request with a
//! named error ([`GenError::Constrained`]). A pass draws one id a kept row, so
//! the program moves through the kept ids alone and nothing is taken back. A
//! forced call's program held while the model reasons steps until the
//! model's own close hands it the answer: a pass could keep ids past the
//! close that the program did not pick.
//!
//! A watch (a `tool_choice: auto` request's program) waits in free text for
//! the model's own call opener. Its pick reads the free text it takes itself
//! ([`FreeText`]): the think span's close while its program is held, then the
//! opener. So it passes where any request that reads its rows passes, its
//! reasoning and its free text included: within one pass the ids up to the
//! opener's are free picks, and every id after it is the program's pick from
//! its verified row. A free pick is the unwatched request's (its draw, or the
//! argmax), and a greedy watch that changes no row reads no row on a step
//! until its call opens. Where the opener's id also wrote text past the
//! opener that the program does not start with, the refusal is returned once
//! that id is taken, after its stop words: a stop word that matches the
//! opener ends the request as a stop word does.
//!
//! That is the step arm. The jump arm (on by default, [`Slot::schema_jump`])
//! feeds the ids the program fixes in the calls that read the decisions'
//! rows. The run that opens the answer, the lead, rides the prompt's call
//! ([`Gen::take_lead`]), so the step that closes the prompt's call reads the
//! first decision's row. Each run after a
//! decision is fed as one call ([`Need::Fill`]): the id just taken and the
//! run, the call's last id stepped with it ([`Engine::next_after`]), so its
//! row is the next decision's; a run the bounds cut (the request's budget,
//! the cache's positions) is fed without its last id, which the loop then
//! takes as the step arm does. The call is the engine thread's, run like a
//! prompt's call (cut on the quantum's grid, a call a round beside other busy
//! slots, the last call the `next_after`), never inside a step. Every id fed is an id of
//! the answer, so nothing is taken back. After the last decision nothing is
//! fed or stepped: the closing text and the stop are the tail, taken with no
//! call, the stop the vocabulary's end of sequence ([`Gen::end`]; a request
//! whose stops lack it feeds its closing text as a run and picks its stop
//! from the row). The answer is the step arm's but for that stop id, and the
//! prompt cache holds the answer up to its last decision. A request that
//! asks for probabilities reads every row, and a penalized one counts the
//! fixed ids in its history before each draw, so each takes the step arm.
//! A run shorter than [`MIN_FILL`] ids is taken as the step arm takes it: a
//! call of one id is no cheaper than the step it replaces. On a drafting
//! engine either arm's picks ride its passes where it would step.
//!
//! A request runs as a [`Gen`], one engine call at a time, so the engine
//! thread can make one call of several slots' steps: each running request's
//! next step a row of one [`Engine::step_slots`], its next drafted pass a row
//! of one [`Engine::advance_slots`]. A drafting engine serves one slot unless
//! it keeps its draft's state per slot ([`Engine::slot_drafts`]). Its prompt
//! is one call, cut where the engine asks at its messages, or on an engine
//! that names a prompt quantum ([`Engine::prompt_quantum`]) calls the engine
//! thread runs one a round ([`Prompt`]).

use std::borrow::Cow;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

use serde_json::{Value, json};

use crate::api::logprobs::{Ask, Collector, RowError};
use crate::api::relock;
use crate::constrain::{Constraint, Draw, StepError, Walk};
use crate::engine::{
    CacheNote, Decoder, Drafted, Engine, EngineError, Landed, Pending, Sampler, SamplerFactory,
    SamplerRefused, SamplingParams, SaveInFlight, Saved, SharedPrefix, SlotDraw, SlotPass, SlotRow,
    StateError, Tokenizer,
};
use crate::media::{Held, MediaFeed, MediaSpan, common_prefix, keep_whole_spans};
use crate::promptcache::{PromptCache, Taken};
use crate::reasoning::{ThinkEntry, ThinkSplit, ThinkTags};
use crate::sampling;
use crate::slotfile::{self, Counting};
use crate::stop::{Pushed, StopScan};

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
    /// The think-span budget, in generated ids taken while the span is open:
    /// spending it force-feeds the span's close id, which the model's context
    /// then really carries (llama-server's `--reasoning-budget` per request).
    /// `Some` only where a span can open — a prompt that already closed it has
    /// no reasoning to cap and the budget is silently ignored, as llama-server
    /// ignores the flag when thinking is off.
    pub reasoning_budget: Option<usize>,
    /// The loop guard, the request's `reasoning_loop`: the generation ends
    /// ([`StopKind::Loop`]) once the open think span repeats a cycle
    /// ([`Repeats`]). `true` only where a span can open, as the budget.
    pub reasoning_loop: bool,
    /// Where the prompt left the think span, for the budget's tracker: the
    /// ids before a model-opened span spends nothing, and its close ids are
    /// not forced until the span opens.
    pub think_entry: ThinkEntry,
    /// The tags of the model's think span, the server's: the budget's tracker
    /// reads the span's close in them and forces `think.close` when it is spent.
    pub think: ThinkTags,
    /// The chat parser's splitter of the span ([`crate::dsml::ChatParser`]),
    /// which the budget and the loop guard read so the span ends for them
    /// where it ends for the answer (a Qwen3-Coder call ends it without
    /// `</think>`); `None` reads `think` from `think_entry`.
    pub think_split: Option<ThinkSplit>,
    /// The probabilities the request asked for and the log its generation
    /// writes them to; `None` asks for none. Only the routes that render them
    /// set it.
    pub logprobs: Option<Ask>,
    /// The program the output follows (structured output), `None` for free
    /// text. Every token is the request's pick among the ids the program
    /// allows: the largest logit when it decodes greedily, else its sampler's
    /// draw from the row with every other id masked out.
    pub constraint: Option<Constraint>,
    /// OpenAI's `logit_bias`: each id's addition to its logit before every
    /// pick, by id, none zero; empty changes nothing.
    pub logit_bias: Vec<(u32, f32)>,
}

/// llama-server's `timings` object, as the server clocked it.
#[derive(Clone, Debug, Default)]
pub(crate) struct Timings {
    /// Prompt tokens evaluated by this request: the prompt less what the
    /// cache kept (`cache_n`).
    pub prompt_n: usize,
    /// The prompt's own engine time: its calls and its last id's step, not
    /// the other slots' decode rounds between the calls of a prompt run a
    /// call a round ([`Prompt`]).
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
    pub n_ctx: usize,
    pub n_past: usize,
    /// Prompt positions kept from the previous request instead of evaluated.
    pub cache_n: usize,
    /// The prompt cache's work before the prompt phase: the slot's state
    /// saved, a cached state put back, the cut (ours; llama-server has no
    /// such field).
    pub cache_ms: f64,
    /// The whole prompt, kept and evaluated (llama-server's `n_prompt_tokens`;
    /// not a `timings` field).
    pub n_prompt: usize,
    /// The ids the draft proposed and the ones the target kept, over every
    /// pass (llama-server's `draft_n`, `draft_n_accepted`).
    pub draft_n: usize,
    pub draft_n_accepted: usize,
    /// The passes that verified a proposal (llama-server's
    /// `n_draft_verif_steps`; not a `timings` field).
    pub draft_passes: usize,
    /// What a request that follows a program asked of the engine after its
    /// prompt (ours; `None` for free text).
    pub jump: Option<Jump>,
}

/// The engine calls and picks of a request that follows a program, after its
/// prompt: the prompt's own calls and the step of its last id are
/// `prompt_ms`'s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Jump {
    /// Whether the request took the jump arm.
    pub arm: bool,
    /// The runs fed as calls, and the ids in them.
    pub spans: usize,
    pub span_ids: usize,
    /// The ids picked among a choice's options, and inside free strings.
    pub decisions: usize,
    pub free_steps: usize,
    /// The engine's [`Engine::next`], [`Engine::prefill`] and
    /// [`Engine::next_after`] calls.
    pub next_calls: usize,
    pub prefill_calls: usize,
    pub next_after_calls: usize,
    /// The fixed ids that open the answer, fed by the prompt's call
    /// ([`Gen::take_lead`]).
    pub lead: usize,
    /// The ids taken after the program's last pick with no engine call, the
    /// stop included ([`Gen::tail_id`]).
    pub tail: usize,
}

impl Timings {
    /// The JSON llama-server emits, and `cache_ms`. A zero count gives NaN
    /// per-token fields, which serialize as `null` exactly as nlohmann writes
    /// them. The draft's two counts are there once a pass proposed, as
    /// llama-server's are; a request that follows a program adds
    /// `constrained`, its [`Jump`] counts.
    pub(crate) fn to_json(&self) -> Value {
        let (pn, dn) = (self.prompt_n as f64, self.predicted_n as f64);
        let mut v = json!({
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
            "cache_ms": self.cache_ms,
        });
        if self.draft_n > 0 {
            v["draft_n"] = json!(self.draft_n);
            v["draft_n_accepted"] = json!(self.draft_n_accepted);
        }
        if let Some(j) = &self.jump {
            v["constrained"] = json!({
                "arm": if j.arm { "jump" } else { "step" },
                "spans": j.spans,
                "span_ids": j.span_ids,
                "decisions": j.decisions,
                "free_steps": j.free_steps,
                "next_calls": j.next_calls,
                "prefill_calls": j.prefill_calls,
                "next_after_calls": j.next_after_calls,
                "lead": j.lead,
                "tail": j.tail,
            });
        }
        v
    }

    /// One pass's draft counts into the request's.
    fn book(&mut self, d: Drafted) {
        self.draft_n += d.proposed;
        self.draft_n_accepted += d.accepted;
        self.draft_passes += usize::from(d.proposed > 0);
    }
}

/// One draft depth's counts: the passes that proposed an id at that depth
/// (the `d`-th id of their draft), and those whose target kept it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Depth {
    pub proposed: u64,
    pub accepted: u64,
}

/// One pass's draft counts into `depths`, depth `d` at index `d - 1`: a pass
/// that proposed `p` ids and kept `a` of them (the first `a`, as a target
/// keeps a prefix) proposed at depths 1 to `p` and was accepted at depths 1
/// to `a`.
fn book_depths(depths: &mut Vec<Depth>, d: Drafted) {
    if depths.len() < d.proposed {
        depths.resize(d.proposed, Depth::default());
    }
    for (k, at) in depths[..d.proposed].iter_mut().enumerate() {
        at.proposed += 1;
        at.accepted += u64::from(k < d.accepted);
    }
}

impl std::ops::AddAssign for Depth {
    fn add_assign(&mut self, d: Depth) {
        self.proposed += d.proposed;
        self.accepted += d.accepted;
    }
}

/// Why a generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StopKind {
    Eos,
    Word,
    Limit,
    /// The loop guard found the open think span repeating a cycle
    /// ([`GenParams::reasoning_loop`]).
    Loop,
}

impl StopKind {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            StopKind::Eos => "eos",
            StopKind::Word => "word",
            StopKind::Limit => "limit",
            StopKind::Loop => "reasoning_loop",
        }
    }

    /// OpenAI `finish_reason`, and `reasoning_loop` for the loop guard's stop.
    pub(crate) fn finish_reason(self) -> &'static str {
        match self {
            StopKind::Eos | StopKind::Word => "stop",
            StopKind::Limit => "length",
            StopKind::Loop => "reasoning_loop",
        }
    }
}

/// The server's stderr finish line's cause word: the generation's own stop
/// (`eos`, `stop` for a stop string, `length`, `reasoning_loop`), `cancel` for
/// a request its client or the server's stop cut short, `error` for one that
/// failed. The one mapping, beside [`StopKind::finish_reason`]; the endpoints
/// never derive their own.
pub(crate) fn cause_word(stop: Option<StopKind>, failed: bool) -> &'static str {
    match stop {
        Some(StopKind::Eos) => "eos",
        Some(StopKind::Word) => "stop",
        Some(StopKind::Limit) => "length",
        Some(StopKind::Loop) => "reasoning_loop",
        None if failed => "error",
        None => "cancel",
    }
}

/// A finished generation.
pub(crate) struct Outcome {
    pub content: String,
    /// The special tokens' texts in `content`: byte offset and byte
    /// length, ascending ([`Event::Text`]).
    pub specials: Vec<(usize, usize)>,
    /// Every generated id, the end-of-generation one included.
    pub tokens: Vec<u32>,
    pub stop: StopKind,
    pub stopping_word: String,
    pub truncated: bool,
    pub timings: Timings,
}

impl Outcome {
    /// Whether the token limit (`finish_reason` `length`) cut the generation
    /// short: a call open there is read as far as it got.
    pub(crate) fn cut(&self) -> bool {
        self.stop == StopKind::Limit
    }
}

/// What the loop hands its sink.
pub(crate) enum Event<'a> {
    /// The prompt is evaluated (llama-server's `prompt_progress`, sent once, complete).
    Prompt(&'a Timings),
    /// Text safe to stream, and the special tokens' texts in it, ascending,
    /// each as its byte offset and byte length ([`Tokenizer::is_special`]):
    /// the output parsers read a marker only where one whole token wrote it.
    /// Empty between the calls of a prompt or of a run of fixed ids
    /// ([`Need::Fill`]) run a call a round ([`Prompt`]): the event a client
    /// gone fails on, as a token's text does, its timings the request's so
    /// far.
    Text(&'a str, &'a [(usize, usize)], &'a Timings),
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum GenError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    /// The sink failed: the client went away.
    #[error("client: {0}")]
    Client(#[from] io::Error),
    /// Under `ignore_eos` the sampler chose an end-of-generation id, whose
    /// logit the loop had set to -inf: a sampler defect, not the engine's.
    #[error("ignore_eos: the sampler chose end-of-generation id {0}, whose logit is -inf")]
    Banned(u32),
    /// The row a generated token was taken from has no distribution (a NaN,
    /// a +inf, no finite logit) for the probabilities the request asked for.
    /// It ends the request, wherever in the generation it is met.
    #[error("the probabilities of generated token {token}: {error}")]
    Logprobs { token: usize, error: RowError },
    /// The program of a constrained request allows no id at a step.
    #[error(transparent)]
    Constrained(#[from] StepError),
    /// The model of a forced tool call ended its generation inside the think
    /// span the call follows ([`Constraint::follows_reasoning`]): no call was
    /// made.
    #[error(
        "constrained output: the generation ended inside the think span, before the forced tool \
         call"
    )]
    Unclosed,
    /// A special token the streaming decoder did not release whole (a space a
    /// cleaning vocabulary holds), so the loop cannot place its span in the
    /// text. The request ends; the server keeps serving the others.
    #[error("special token {0} was not released whole by the streaming decoder")]
    SpecialPiece(u32),
}

pub(crate) fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// The logits buffer the engine fills, or `None` on a greedy request (empty buffer).
fn out(logits: &mut [f32]) -> Option<&mut [f32]> {
    (!logits.is_empty()).then_some(logits)
}

/// The last `n` ids of `held` a draw's penalties see, in order: an image
/// span's positions are left out, as llama-server skips the null ids a media
/// chunk holds.
fn prompt_tail(held: &Held, n: usize) -> Vec<u32> {
    let mut spans = held.media.iter().rev().peekable();
    let mut tail: Vec<u32> = Vec::new();
    for (at, &id) in held.ids.iter().enumerate().rev() {
        if tail.len() == n {
            break;
        }
        while spans.next_if(|s| s.at > at).is_some() {}
        if spans.peek().is_some_and(|s| at < s.end()) {
            continue;
        }
        tail.push(id);
    }
    tail.reverse();
    tail
}

/// What a request does to every row before its pick: `logit_bias`'s
/// additions ([`GenParams::logit_bias`]), then -inf at each banned id (the
/// stop ids under `ignore_eos`, as llama-server's `logit_bias_eog`).
#[derive(Clone, Debug, Default)]
struct Bias {
    add: Vec<(u32, f32)>,
    banned: Vec<u32>,
}

impl Bias {
    fn is_empty(&self) -> bool {
        self.add.is_empty() && self.banned.is_empty()
    }

    /// Whether the changed row's largest logit can be another id than the
    /// row's own argmax `g`: an addition that raises some id, or one that
    /// changes `g`. Lowering other ids keeps `g` first.
    fn moves(&self, g: u32) -> bool {
        self.banned.contains(&g) || self.add.iter().any(|&(id, b)| b > 0.0 || id == g)
    }

    /// `logits` changed into `row`. A greedy draw's copy reads -0 as +0, so
    /// its largest logit ties the two by id as the engine's argmax does; any
    /// other copy keeps the signs.
    fn apply(&self, logits: &[f32], greedy: bool, row: &mut Vec<f32>) {
        row.clear();
        row.extend(
            logits
                .iter()
                .map(|&l| if greedy && l == 0.0 { 0.0 } else { l }),
        );
        let at = |id: u32| usize::try_from(id).ok();
        for &(id, b) in &self.add {
            if let Some(l) = at(id).and_then(|i| row.get_mut(i)) {
                *l += b;
            }
        }
        for &id in &self.banned {
            if let Some(l) = at(id).and_then(|i| row.get_mut(i)) {
                *l = f32::NEG_INFINITY;
            }
        }
    }
}

/// The draw of a request that changes its rows ([`Bias`]): `draw` on the
/// changed copy of each row, or for a greedy request the copy's largest
/// logit (first of equals). A step's row ([`Gen::answer`], where a greedy
/// request whose argmax the change keeps first takes it unread) and a pass's
/// verified row ([`Engine::advance_sampled`]) are drawn from alike, so a
/// drafted run takes the ids a plain run takes and a pass never keeps a
/// banned id: where the draft proposed another id than the draw, the draw's
/// id is kept and the pass stops there.
fn biased(mut draw: Option<Sampler>, bias: Bias) -> Sampler {
    let greedy = draw.is_none();
    let mut row: Vec<f32> = Vec::new();
    Box::new(move |logits: &[f32], history: &[u32]| {
        bias.apply(logits, greedy, &mut row);
        match &mut draw {
            Some(s) => s(&row, history),
            None => sampling::argmax(&row),
        }
    })
}

/// What takes each generated id from its logits row ([`Pick::answer`]): the
/// request's draw (its sampler, or the argmax), its change to the rows, the
/// program it follows and the probabilities it asks for. A step asks it once,
/// for the row the step wrote ([`Gen::answer`]). A drafted pass is lent it as
/// the pass's sampler ([`Gen::lend`]), which asks it once a kept row, in
/// position order, for that position's verified row: the pass stops after the
/// first id that is not its draft's, so every id it draws is kept. The pass
/// takes the ids, and holds the values, the steps would take from those rows,
/// and the program moves through the kept ids alone: nothing is taken back.
#[derive(Default)]
struct Pick {
    sampler: Option<Sampler>,
    /// The request's change to its rows: its `logit_bias`, and the stop ids
    /// under `ignore_eos`.
    bias: Bias,
    /// Whether the request is greedy: one that changes its rows draws
    /// through [`biased`] only where the change can move the engine's argmax
    /// ([`Bias::moves`]).
    greedy: bool,
    /// What reads each row and holds the request's probabilities, `None` on
    /// a request that asks for none.
    probs: Option<Collector>,
    /// The request's position in its program, `None` for free text and
    /// while the program is held.
    walk: Option<Walk>,
    /// The program held while the model reasons in the span the prompt left
    /// open ([`Constraint::follows_reasoning`]), or one the model may open
    /// under a watch: it becomes the walk once the model's own close is
    /// taken, a forced call's by [`Gen::spend`], a watch's by its pick
    /// ([`Pick::read`]). A forced call's request steps until then: a pass
    /// could keep ids past the close that the program did not pick.
    held: Option<Walk>,
    /// What a watch's pick reads of the free text it takes, `None` on a
    /// request whose program does not wait for a call ([`FreeText`]).
    free: Option<FreeText>,
    /// The changed copy of the row a program or a watch's free text picks
    /// from, empty while the request changes no row.
    biased_row: Vec<f32>,
    /// The row a sampled request's walk draws from, the ids its program does
    /// not allow masked ([`Draw::Sampled`]), kept for its capacity.
    masked: Vec<f32>,
    /// The rows answered since the pick was last lent to a pass
    /// ([`Gen::lend_out`]): the place among the pass's rows a refusal names.
    answered: usize,
    /// The program's refusal of a pick, with its row's place among its
    /// pass's (`answered` then). A step's, or a pass's at its first row, the
    /// next [`Gen::pump`] returns before it takes the token; one at a later
    /// row is returned once the pass's ids before it are taken
    /// ([`Gen::advanced`]). The refused pick and every id after it go
    /// untaken. A watch's refusal of the text a free id wrote ([`Pick::read`])
    /// names the row after that id's: the id is taken, its stop words read,
    /// and the refusal follows it ([`Gen::answer`] holds a step's).
    refused: Option<(usize, StepError)>,
}

/// What a watch's pick reads of the ids it takes as free text, each as it
/// takes it, inside a pass as on a step: their text, as the generation loop's
/// own decoder releases it, for the call opener ([`Walk::saw`]), and while
/// the program is held, the think span's split, for the model's own close
/// ([`Walk::fed`]). The loop reads a pass's ids only once the pass is over,
/// so the pick reads them itself: the row after the opener's is the
/// program's. The ids inside a call are the program's and are not read: the
/// decoder holds nothing once the opener is in (else that id is refused), so
/// it stands where the loop's does again when the call closes.
struct FreeText {
    dec: Box<dyn Decoder>,
    vocab: Arc<dyn Tokenizer>,
    /// The think span's split while the program is held, `None` once the
    /// model's own close handed the program over.
    split: Option<ThinkSplit>,
}

impl Pick {
    /// Whether the request reads its rows: it draws (it samples or changes
    /// its rows), asks for probabilities or follows a program.
    fn reads_rows(&self) -> bool {
        self.sampler.is_some() || self.probs.is_some() || self.walk.is_some() || self.held.is_some()
    }

    /// The id taken from `row` for the next generated token: `forced` when
    /// the think budget forces its close, else [`Pick::choose`]'s; the row is
    /// read for the probabilities first and the id's value held for its text.
    /// `g` is the engine's argmax of the row where the engine gave it (a
    /// step's), `None` for a pass's verified row.
    fn answer(&mut self, row: &[f32], history: &[u32], forced: Option<u32>, g: Option<u32>) -> u32 {
        if let Some(c) = self.probs.as_mut() {
            c.read(row);
        }
        let free = self.free.is_some() && self.refused.is_none() && self.in_free_text();
        let id = forced.unwrap_or_else(|| self.choose(row, history, g));
        if let Some(c) = self.probs.as_mut() {
            c.pick(id, row);
        }
        self.answered += 1;
        if free && let Err(e) = self.read(id) {
            self.refused = Some((self.answered, e));
        }
        id
    }

    /// Whether the next id is free text: the program held, or a walk that
    /// waits for its call opener.
    fn in_free_text(&mut self) -> bool {
        self.held.is_some() || self.walk.as_mut().is_some_and(Walk::in_free_text)
    }

    /// A watch's free id `id`, just taken, read ([`FreeText`]): while the
    /// program is held, its text into the think span's split, and where the
    /// split ends the span, the program handed over and the text past the
    /// close fed to it; else its text watched for the opener. Once the
    /// opener is in, bytes the decoder still holds are text after it that
    /// no piece showed the program: the named error.
    fn read(&mut self, id: u32) -> Result<(), StepError> {
        let f = self.free.as_mut().expect("a watch reads its free text");
        let piece = f.dec.push(id);
        let text = piece.as_deref().unwrap_or("");
        let walk = match f.split.as_mut() {
            Some(split) => {
                // A special id its piece cannot place ends the request by
                // name where the loop takes it ([`special_at`]), before any
                // id after it.
                let special = special_at(&*f.vocab, id, piece.as_deref()).ok().flatten();
                let s = split.push(text, special.as_slice());
                if !s.closed && s.content.is_empty() {
                    return Ok(());
                }
                f.split = None;
                self.walk = self.walk.take().or_else(|| self.held.take());
                let walk = self.walk.as_mut().expect("a held program to hand over");
                walk.fed(&s.content, &s.specials)?;
                walk
            }
            None => {
                let walk = self.walk.as_mut().expect("a watch's walk");
                walk.saw(id, text)?;
                walk
            }
        };
        if walk.waiting() || f.dec.flush().is_empty() {
            return Ok(());
        }
        Err(walk.stuck("the call opener's id also wrote part of a character after it"))
    }

    /// The request's pick from `row`: its program's pick from the biased copy
    /// (a sampled one drawing among the ids the program allows), else its
    /// draw (a sampled one, or one that changes its rows: [`biased`] draws
    /// from a copy) or the engine's argmax, which a greedy request whose
    /// change cannot move it takes as `g` where the engine gave it. Past a
    /// refusal the pick is the argmax, never taken.
    fn choose(&mut self, row: &[f32], history: &[u32], g: Option<u32>) -> u32 {
        // A pass's row comes without `g`: its stand-in takes the engine's
        // rule, `-0` tied with `+0` by id and a NaN never, as `g` would.
        let argmax = |row: &[f32]| {
            g.unwrap_or_else(|| sampler::argmax((0..=u32::MAX).zip(row.iter().copied())).0)
        };
        if self.refused.is_some() {
            return argmax(row);
        }
        if let Some(walk) = self.walk.as_mut()
            && !walk.in_free_text()
        {
            let row = if self.bias.add.is_empty() || walk.fixed_next() {
                row
            } else {
                self.bias.apply(row, false, &mut self.biased_row);
                &self.biased_row
            };
            let picked = match self.sampler.as_mut() {
                Some(sampler) => walk.choose_by(
                    row,
                    &mut Draw::Sampled {
                        sampler,
                        history,
                        masked: &mut self.masked,
                    },
                ),
                None => walk.choose(row),
            };
            return picked.unwrap_or_else(|e| {
                self.refused = Some((self.answered, e));
                argmax(row)
            });
        }
        // Free text is one path: an unwatched request's, a held program's
        // reasoning and a waiting watch's. A program's request keeps its bias
        // out of its sampler, as the walk applies it ([`Gen::new`]), so its
        // free text draws from the copy made here by [`biased`]'s rule.
        let moves = g.is_none_or(|g| self.bias.moves(g));
        let program = self.walk.is_some() || self.held.is_some();
        let biased = program && !self.bias.is_empty() && (!self.greedy || moves);
        let row = if biased {
            self.bias.apply(row, self.greedy, &mut self.biased_row);
            &self.biased_row
        } else {
            row
        };
        match self.sampler.as_mut() {
            Some(draw) if !self.greedy || moves => draw(row, history),
            None if biased => sampling::argmax(row),
            _ => argmax(row),
        }
    }
}

/// Where special id `id`'s text is in `piece`, the text the streaming decoder
/// released for it: the piece ends with it, after any bytes the decoder held;
/// `None` for an id that is not special. A piece that does not carry the text
/// whole (a space a cleaning decoder holds) cannot place it, and the request
/// ends by name rather than lose the span.
fn special_at(
    vocab: &dyn Tokenizer,
    id: u32,
    piece: Option<&str>,
) -> Result<Option<(usize, usize)>, GenError> {
    if !vocab.is_special(id) {
        return Ok(None);
    }
    let text = vocab.decode(&[id]);
    let at = piece
        .and_then(|p| p.strip_suffix(text.as_str()))
        .map(str::len)
        .ok_or(GenError::SpecialPiece(id))?;
    Ok(Some((at, text.len())))
}

/// Whether a request takes its tokens after the first through the engine's
/// passes when the engine drafts: one that reads no row greedily
/// ([`Engine::advance`]), and one that reads its rows ([`Pick::reads_rows`])
/// on an engine that drafts sampled requests ([`Engine::advance_sampled`]),
/// each kept id its pick from its verified row.
fn takes_passes(reads_rows: bool, engine: &dyn Engine) -> bool {
    !reads_rows || engine.drafts_sampled()
}

/// The engine and its slots: what each slot's cache holds — the ids, one per
/// position, and the image spans among them ([`Held`]) — and the host prompt
/// cache of states the slots held before ([`PromptCache`]), one for every
/// slot. Every method but [`Slot::select`], [`Slot::held_of`],
/// [`Slot::step_slots`] and [`Slot::advance_slots`] acts on the selected slot.
pub(crate) struct Slot {
    pub engine: Box<dyn Engine>,
    vocab: Arc<dyn Tokenizer>,
    /// Every id the selected slot has evaluated since its last reset, in
    /// order, and the image spans among them. The last generated id of a
    /// request is not in it: it is never fed back.
    held: Held,
    /// The other slots' held sequences; the selected slot's entry is empty.
    parked: Vec<Held>,
    /// The slot whose held sequence `held` is.
    cur: usize,
    /// The slot the engine has selected: `None` after a call of several.
    selected: Option<usize>,
    /// Each slot's last request prompt, in ids, the one it fed; 0 when not
    /// known (nothing fed since the slot emptied). A state the slot took back
    /// whole (a cached state, a slot file) counts as all prompt.
    prompts: Vec<usize>,
    /// The selected slot's state while the engine holds another: the parked
    /// state handed back for the request about to start there
    /// ([`Slot::hand_back`]), which that request's [`Slot::reuse`] consumes.
    away: Option<Arc<dyn Saved>>,
    /// The selected slot's state is not on the engine, which holds another
    /// slot's ([`Slot::select_off`]): [`Slot::erase`] drops its ids alone.
    off: bool,
    cache: PromptCache,
    /// Each slot's state copy while it may still run ([`Slot::seal`]): its
    /// rows are not cut, reset or replaced before it has landed.
    saving: Vec<Option<Pending>>,
    /// A seal's state until the next request consumes this slot. A weak
    /// reference leaves the cache free to return its staging.
    sealed: Vec<Option<Weak<dyn Saved>>>,
    /// Whether a request that follows a program takes the jump arm.
    jump: bool,
}

/// Another slot's cache a request takes its prefix from ([`Slot::share_pick`]):
/// the slot, the ids the request shares with it and what the engine keeps
/// of them.
#[derive(Clone, Copy)]
struct Lend {
    from: usize,
    common: usize,
    kept: usize,
}

/// What a cache hit put back ([`Slot::load`]).
struct Loaded {
    positions: usize,
    bytes: u64,
    ms: f64,
    resumed: usize,
    waited_ms: f64,
}

impl Slot {
    /// The engine with one slot.
    #[cfg(test)]
    pub(crate) fn new(engine: Box<dyn Engine>) -> Slot {
        Slot::with_slots(engine, 1)
    }

    /// The engine with `n` slots (at least 1, at most what it declares).
    pub(crate) fn with_slots(engine: Box<dyn Engine>, n: usize) -> Slot {
        Slot {
            vocab: engine.tokenizer(),
            cache: PromptCache::new(engine.cache_ram()),
            engine,
            held: Held::default(),
            parked: vec![Held::default(); n],
            cur: 0,
            selected: Some(0),
            prompts: vec![0; n],
            away: None,
            off: false,
            saving: vec![None; n],
            sealed: vec![None; n],
            jump: true,
        }
    }

    /// Whether a request that follows a program takes the jump arm (the
    /// default) or the step arm alone (`BLOOMERY_SCHEMAJUMP=off`).
    pub(crate) fn schema_jump(mut self, on: bool) -> Slot {
        self.jump = on;
        self
    }

    /// Makes `slot` the selected one, on the engine too when it changes.
    ///
    /// # Panics
    ///
    /// When another slot is selected while the selected one's handed-back
    /// state waits for its request ([`Slot::hand_back`]).
    pub(crate) fn select(&mut self, slot: usize) -> Result<(), EngineError> {
        assert!(
            self.away.is_none() || self.cur == slot,
            "slot {} was handed its parked state back and slot {slot} was selected before a \
             request took it",
            self.cur
        );
        self.off = false;
        if self.selected != Some(slot) {
            self.engine.select_slot(slot)?;
            self.selected = Some(slot);
        }
        if self.cur != slot {
            let parked = std::mem::take(&mut self.parked[slot]);
            self.parked[self.cur] = std::mem::replace(&mut self.held, parked);
            self.cur = slot;
        }
        Ok(())
    }

    /// [`Slot::select`] of a slot whose state the engine does not hold, for
    /// an action that drops it ([`Slot::erase`]); the next select ends it.
    pub(crate) fn select_off(&mut self, slot: usize) -> Result<(), EngineError> {
        self.select(slot)?;
        self.off = true;
        Ok(())
    }

    /// What `slot`'s cache holds: its ids and their image spans.
    pub(crate) fn held_of(&self, slot: usize) -> &Held {
        if slot == self.cur {
            &self.held
        } else {
            &self.parked[slot]
        }
    }

    /// `state`, the selected slot's parked state of the ids it holds, handed
    /// back for the request about to start on it while the engine holds
    /// another: that request's [`Slot::reuse`] puts it into the prompt cache
    /// as it is where the rule saves it (no copy), and back into the engine
    /// only when the request keeps some of it.
    ///
    /// # Panics
    ///
    /// When `state` holds other than the slot's positions: the park table and
    /// the slot disagree.
    pub(crate) fn hand_back(&mut self, state: Arc<dyn Saved>) {
        assert_eq!(
            state.n_tokens(),
            self.held.len(),
            "slot {}'s parked state holds {} positions, the slot {}",
            self.cur,
            state.n_tokens(),
            self.held.len()
        );
        self.away = Some(state);
    }

    fn held_mut(&mut self, slot: usize) -> &mut Held {
        self.sealed[slot] = None;
        if slot == self.cur {
            &mut self.held
        } else {
            &mut self.parked[slot]
        }
    }

    /// One step of several slots ([`Engine::step_slots`]) that books what
    /// each fed. While the call is in flight the rows' ids are empty, so a
    /// failed call leaves no claim about their caches. A row the engine
    /// answers with no id of the vocabulary (its `next` left as it was) is the
    /// engine's error.
    pub(crate) fn step_slots(&mut self, rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
        let held: Vec<Held> = rows
            .iter()
            .map(|r| std::mem::take(self.held_mut(r.slot)))
            .collect();
        self.selected = None;
        for r in rows.iter_mut() {
            r.next = u32::MAX;
        }
        self.engine.step_slots(rows)?;
        let n_vocab = self.vocab.n_vocab();
        if let Some(r) = rows.iter().find(|r| r.next as usize >= n_vocab) {
            return Err(EngineError(format!(
                "a step of {} slots answered slot {} with id {}, past the vocabulary's {n_vocab}",
                rows.len(),
                r.slot,
                r.next
            )));
        }
        for (r, h) in rows.iter().zip(held) {
            let held = self.held_mut(r.slot);
            *held = h;
            held.ids.push(r.last);
        }
        Ok(())
    }

    /// One drafted pass of several slots ([`Engine::advance_slots`]) that
    /// books what each fed: `last` and every kept token but the last. While
    /// the call is in flight the rows' ids are empty, so a failed call leaves
    /// no claim about their caches. A row whose pass kept no token, more than
    /// the engine's rows, or other than one more than its draft's accepted
    /// ids, or a sampled row whose sampler did not come back, is the engine's
    /// error.
    pub(crate) fn advance_slots(&mut self, rows: &mut [SlotPass<'_>]) -> Result<(), EngineError> {
        let held: Vec<Held> = rows
            .iter()
            .map(|r| std::mem::take(self.held_mut(r.slot)))
            .collect();
        self.selected = None;
        for r in rows.iter_mut() {
            r.out.clear();
        }
        self.engine.advance_slots(rows)?;
        let rows_n = self.engine.advance_rows();
        for r in rows.iter() {
            if r.draw.as_ref().is_some_and(|d| d.sampler.is_none()) {
                return Err(EngineError(format!(
                    "slot {}: a sampled pass kept the request's sampler",
                    r.slot
                )));
            }
            if r.out.is_empty()
                || r.out.len() > rows_n
                || r.out.len() != r.drafted.accepted + 1
                || r.drafted.accepted > r.drafted.proposed
            {
                return Err(EngineError(format!(
                    "slot {}: a pass of at most {rows_n} rows kept {} tokens, its draft {} of {} \
                     proposed ids",
                    r.slot,
                    r.out.len(),
                    r.drafted.accepted,
                    r.drafted.proposed
                )));
            }
        }
        for (r, h) in rows.iter().zip(held) {
            let held = self.held_mut(r.slot);
            *held = h;
            held.ids.push(r.last);
            held.ids.extend_from_slice(&r.out[..r.out.len() - 1]);
        }
        Ok(())
    }

    /// Brings the cache to the longest prefix of `req` it can keep, leaving at
    /// least the last id for `next`, and returns that length. `want` false
    /// (`cache_prompt: false`) always resets. While an engine call is in flight
    /// `held` is empty, so a failed call leaves no claim about the cache.
    ///
    /// The prefix is the longest the slot and the request share as sequences
    /// ([`common_prefix`]): the same ids and, inside the shared part, the same
    /// images, and it never ends inside an image span.
    ///
    /// With the prompt cache on: a cached state that keeps more of `req` than
    /// the slot replaces the slot's state; the slot's state goes into the
    /// cache first whenever it is replaced, or the request cuts it and does
    /// not carry the slot's last prompt whole. On a server of one slot the
    /// cut saves only when it keeps less than half of the slot, llama-server's
    /// rule (`f_keep < 0.5`); on one of several, the request sat on this slot
    /// for the prefix it shares where another slot may have been free, so any
    /// cut saves and the choice costs the slot's conversation nothing. The
    /// last-prompt condition is ours: a request that carries the last prompt
    /// whole drops only that prompt's reply, which a client that renders the
    /// reply again (a template that leaves the reasoning out) never sends
    /// back; the price is that a regenerated reply's old state is not kept. A
    /// prefix the engine keeps less of than the request shares is noted with
    /// the engine's reason.
    ///
    /// Another slot whose cache the engine can put more of `req` from than the
    /// slot keeps ([`Engine::share_keepable`]) replaces the slot's state the
    /// same way, its prefix taken on the device ([`Engine::share_prefix`]);
    /// a cached state wins only by keeping more. The slot's state is saved
    /// first as for a cached state, and the share is noted.
    ///
    /// A slot handed its parked state back ([`Slot::hand_back`]) holds it off
    /// the engine: the cache takes that state as it is where the rule saves
    /// it, and the engine takes it back only when the request keeps some of
    /// it. One the request keeps none of leaves the slot holding nothing, so
    /// no prefix it shared is noted.
    fn reuse(&mut self, req: &Held, want: bool) -> Result<usize, EngineError> {
        let away = self.away.take();
        let sealed = self.sealed[self.cur].take();
        // The slot's own state copy lands before its rows are cut, reset or
        // replaced; the copies of the other slots that have landed leave
        // the cache free to evict them.
        self.settle(self.cur)?;
        self.refresh()?;
        let (mut common, mut ask, mut k) = self.keep_of(req, want, away.as_ref());
        let lent = if want { self.share_pick(req, k) } else { None };
        // The pick is taken out before the slot's state is saved: making room
        // for that state may evict it, and the pick then lives on in `picked`.
        let picked = if want && self.cache.enabled() {
            self.cache
                .best(req, lent.map_or(k, |l| l.kept))
                .map(|p| (p.kept, self.cache.take(p)))
        } else {
            None
        };
        let last = self.prompts[self.cur];
        let continues = last > 0 && common >= last;
        let saves_cut = if self.parked.len() > 1 {
            k < self.held.len()
        } else {
            2 * k < self.held.len()
        };
        if self.cache.enabled()
            && !self.held.is_empty()
            && (picked.is_some() || lent.is_some() || (saves_cut && !continues))
            && !sealed
                .as_ref()
                .is_some_and(|s| self.cache.contains(s, &self.held))
        {
            self.save_held(away.as_ref())?;
        }
        if let Some((want_kept, taken)) = picked {
            let slot_kept = k;
            if let Some(l) = self.load(taken, want_kept)? {
                (common, ask, k) = self.keep_of(req, want, None);
                self.engine.note(&CacheNote::Load {
                    positions: l.positions,
                    common,
                    kept: k,
                    slot_kept,
                    bytes: l.bytes,
                    ms: l.ms,
                    resumed: l.resumed,
                    waited_ms: l.waited_ms,
                });
            } else {
                (common, ask, k) = (0, 0, 0);
            }
        } else if let Some(l) = lent {
            let slot_kept = k;
            if let Some((shared, ms)) = self.share(l.from, l.kept)? {
                (common, ask, k) = self.keep_of(req, want, None);
                self.engine.note(&CacheNote::Share {
                    from: l.from,
                    common: l.common,
                    kept: k,
                    slot_kept,
                    mapped_bytes: shared.mapped_bytes,
                    copied_bytes: shared.copied_bytes,
                    ms,
                });
            } else {
                (common, ask, k) = (0, 0, 0);
            }
        } else if let Some(state) = away
            && (k == 0 || !self.put_back(&state)?)
        {
            self.held = Held::default();
            (common, ask, k) = (0, 0, 0);
        }
        if k < ask {
            self.engine.note(&CacheNote::Reuse {
                common,
                ask,
                kept: k,
                held: self.held.len(),
                reason: self.engine.keep_limit(ask),
            });
        }
        let held = std::mem::take(&mut self.held);
        if k == 0 {
            self.rows()?.reset()?;
        } else if k < held.len() {
            self.rows()?.cut(k)?;
        }
        self.held = held;
        self.held.truncate(k);
        Ok(k)
    }

    /// The other slot whose cache holds the most of `req` the engine can
    /// put on the selected slot ([`Engine::share_keepable`]), never ending
    /// inside an image span, if that keeps more than `floor`; of equals, the
    /// lowest slot.
    fn share_pick(&self, req: &Held, floor: usize) -> Option<Lend> {
        let last = req.len().checked_sub(1)?;
        let mut best: Option<Lend> = None;
        for (from, held) in self.parked.iter().enumerate() {
            let bar = best.map_or(floor, |b| b.kept);
            let common = common_prefix(held, req);
            if from == self.cur || common.min(last) <= bar {
                continue;
            }
            let kept = keep_whole_spans(req, common.min(last), |n| {
                self.engine.share_keepable(from, n)
            });
            if kept > bar {
                best = Some(Lend { from, common, kept });
            }
        }
        best
    }

    /// The first `n` positions slot `from` holds onto the selected slot
    /// ([`Engine::share_prefix`]), the slot then holding them as `from`
    /// does; returns what the engine took and the call's wall time in ms. A
    /// share the engine refuses resets the slot, which then holds nothing
    /// (`None`), noted; an engine failure is the request's error.
    fn share(&mut self, from: usize, n: usize) -> Result<Option<(SharedPrefix, f64)>, EngineError> {
        self.held = Held::default();
        let t = Instant::now();
        match self.rows()?.share_prefix(from, n) {
            Ok(shared) => {
                if shared.positions == 0 || shared.positions > n {
                    return Err(EngineError(format!(
                        "the engine took {} positions of a share of {n} from slot {from}",
                        shared.positions
                    )));
                }
                let mut held = self.parked[from].clone();
                held.truncate(shared.positions);
                self.prompts[self.cur] = held.ids.len();
                self.held = held;
                Ok(Some((shared, ms_since(t))))
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.rows()?.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: n,
                    why: format!("the engine refused to share slot {from}'s prefix: {e}"),
                });
                Ok(None)
            }
        }
    }

    /// The engine, for a call that cuts, resets or replaces the selected
    /// slot's rows. Refused by name ([`SaveInFlight`]) while that slot's state
    /// is still being copied: every caller waits for the copy first
    /// ([`Slot::settle`]), so this stands between a missed wait and a copy
    /// that reads rows another call has already rewritten.
    fn rows(&mut self) -> Result<&mut dyn Engine, EngineError> {
        if !self.off && self.saving[self.cur].is_some() {
            return Err(SaveInFlight { slot: self.cur }.into());
        }
        self.sealed[self.cur] = None;
        Ok(self.engine.as_mut())
    }

    /// The positions `req` shares with the slot, the most of them a request
    /// may keep (all but its last), and what the engine keeps of those: what
    /// it would keep of `away`, the slot's state off the engine, when given,
    /// never ending inside an image span ([`keep_whole_spans`]).
    fn keep_of(
        &self,
        req: &Held,
        want: bool,
        away: Option<&Arc<dyn Saved>>,
    ) -> (usize, usize, usize) {
        let common = if want {
            common_prefix(&self.held, req)
        } else {
            0
        };
        let ask = common.min(req.len() - 1);
        let kept = keep_whole_spans(req, ask, |n| {
            away.map_or_else(|| self.engine.keepable(n), |s| s.keepable(n))
        });
        (common, ask, kept)
    }

    /// The slot's state into the prompt cache, unless a cached state already
    /// keeps all of it: `away`, the slot's state off the engine, as it is,
    /// else the engine's snapshot. A snapshot the engine refuses is noted and
    /// not kept; an engine failure is the request's error.
    fn save_held(&mut self, away: Option<&Arc<dyn Saved>>) -> Result<(), EngineError> {
        if self.cache.covers(&self.held) {
            return Ok(());
        }
        if away.is_none() && !self.snapshot_fits()? {
            return Ok(());
        }
        let t = Instant::now();
        let (state, copied) = if let Some(s) = away {
            (Arc::clone(s), false)
        } else {
            match self.engine.snapshot() {
                Ok(s) => (s, true),
                Err(StateError::Engine(e)) => return Err(e),
                Err(e) => {
                    self.engine.note(&CacheNote::Skip {
                        positions: self.held.len(),
                        why: format!("the engine took no snapshot: {e}"),
                    });
                    return Ok(());
                }
            }
        };
        self.keep(self.cur, state, copied, None, t);
        Ok(())
    }

    /// `state`, `slot`'s state of what it holds, which the slots that take
    /// the engine in turns let go of, into the prompt cache as a save puts
    /// it there ([`Slot::reuse`]), unless a cached state already keeps all of
    /// it; the slot drops it after. `copied`: a snapshot was taken for it,
    /// not a parked state handed over as it is.
    pub(crate) fn offer(&mut self, slot: usize, state: Arc<dyn Saved>, copied: bool) {
        if self.cache.enabled() && !self.cache.covers(self.held_of(slot)) {
            self.keep(slot, state, copied, None, Instant::now());
        }
    }

    /// `state` of what `slot` holds into the prompt cache, timed from `t`;
    /// `fallback` names the pool that had no room for it off the request path
    /// ([`CacheNote::Save`]). A state of other than those positions is noted
    /// and not kept.
    fn keep(
        &mut self,
        slot: usize,
        state: Arc<dyn Saved>,
        copied: bool,
        fallback: Option<&'static str>,
        t: Instant,
    ) {
        let held = self.held_of(slot).clone();
        let positions = held.ids.len();
        if state.n_tokens() != positions {
            self.engine.note(&CacheNote::Skip {
                positions,
                why: format!(
                    "the state holds {} positions, slot {slot} holds {positions}",
                    state.n_tokens()
                ),
            });
            return;
        }
        let notes = match fallback {
            Some(fallback) => self
                .cache
                .insert_fallback(held, state, ms_since(t), fallback),
            None => self.cache.insert(held, state, ms_since(t), copied),
        };
        for note in notes {
            self.engine.note(&note);
        }
    }

    /// The request on `slot` ended and the slot holds its state: an engine
    /// that copies on a stream of its own ([`Engine::snapshots_async`]) is
    /// asked for the state now ([`Engine::snapshot_begin`]), which goes into
    /// the cache as a copy still running, so the next request that cuts the
    /// slot finds it saved instead of taking a snapshot on its own path. The
    /// cost here is the call that begins the copy. A pool with no room for it
    /// ([`StateError::NoRoom`]) gives up its landed states, oldest first and
    /// never one whose copy still runs, and is asked again after each, since
    /// the room one state gives back may lie apart from the rest that is
    /// free; only when no landed state is left is the state snapshotted on
    /// this path, `fallback` in the `Save` naming the pool `full` or
    /// `fragmented`. A state that passes the whole pool gives up nothing.
    /// Nothing happens for an engine that copies on the request path, a cache
    /// that is off, a slot that holds nothing or a state a cached one already
    /// keeps all of.
    pub(crate) fn seal(&mut self, slot: usize) -> Result<(), EngineError> {
        if !self.cache.enabled() || !self.engine.snapshots_async() {
            return Ok(());
        }
        self.select(slot)?;
        self.refresh()?;
        if self.held.is_empty() || self.cache.covers(&self.held) {
            return Ok(());
        }
        self.settle(slot)?;
        if !self.snapshot_fits()? {
            return Ok(());
        }
        let t = Instant::now();
        let mut begun = self.engine.snapshot_begin();
        while let Err(StateError::NoRoom { need, held, .. }) = &begun
            && need <= held
            && self.evict_for_room()
        {
            begun = self.engine.snapshot_begin();
        }
        match begun {
            Ok(p) => self.keep_pending(slot, p, ms_since(t)),
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => self.keep_on_path(slot, &e),
        }
    }

    /// The oldest landed state leaves the cache for the engine's staging
    /// room, noted. False when every cached state's copy still runs, or the
    /// cache holds none.
    fn evict_for_room(&mut self) -> bool {
        match self.cache.evict_oldest_landed() {
            Some(note) => {
                self.engine.note(&note);
                true
            }
            None => false,
        }
    }

    fn snapshot_fits(&self) -> Result<bool, EngineError> {
        let note = match self.engine.snapshot_size() {
            Ok(Some(bytes)) => self.cache.admission(self.held.len(), bytes),
            Ok(None) => None,
            Err(StateError::Engine(e)) => return Err(e),
            Err(e) => Some(CacheNote::Skip {
                positions: self.held.len(),
                why: format!("the engine could not size its snapshot: {e}"),
            }),
        };
        if let Some(note) = note {
            self.engine.note(&note);
            return Ok(false);
        }
        Ok(true)
    }

    /// `slot`'s state taken on this path, the engine having begun no copy of
    /// it for `why`: [`StateError::NoRoom`] keeps it with the pool's state
    /// named, any other refusal is noted and keeps nothing.
    fn keep_on_path(&mut self, slot: usize, why: &StateError) -> Result<(), EngineError> {
        let Some(fallback) = why.fallback() else {
            self.engine.note(&CacheNote::Skip {
                positions: self.held.len(),
                why: format!("the engine took no snapshot: {why}"),
            });
            return Ok(());
        };
        let t = Instant::now();
        match self.engine.snapshot() {
            Ok(state) => {
                let sealed = Arc::downgrade(&state);
                self.keep(slot, state, true, Some(fallback), t);
                if self.cache.contains(&sealed, self.held_of(slot)) {
                    self.sealed[slot] = Some(sealed);
                }
                Ok(())
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.engine.note(&CacheNote::Skip {
                    positions: self.held.len(),
                    why: format!("the engine took no snapshot: {e}"),
                });
                Ok(())
            }
        }
    }

    /// `pending`, `slot`'s state of what it holds with its copy running,
    /// into the cache; the slot's rows are held still until it lands. A state
    /// of other than the slot's positions, or one the cache refuses, waits
    /// for its copy and is dropped.
    fn keep_pending(&mut self, slot: usize, pending: Pending, ms: f64) -> Result<(), EngineError> {
        let held = self.held_of(slot).clone();
        let positions = held.ids.len();
        if pending.state().n_tokens() != positions {
            self.engine.note(&CacheNote::Skip {
                positions,
                why: format!(
                    "the state holds {} positions, slot {slot} holds {positions}",
                    pending.state().n_tokens()
                ),
            });
            return self.discard(&pending);
        }
        let notes = self.cache.insert_pending(held, pending.clone(), ms);
        let kept = matches!(notes.last(), Some(CacheNote::Save { .. }));
        for note in notes {
            self.engine.note(&note);
        }
        if kept {
            self.sealed[slot] = Some(Arc::downgrade(pending.state()));
            self.saving[slot] = Some(pending);
            Ok(())
        } else {
            self.discard(&pending)
        }
    }

    /// A copy nobody will read: waited for, so that the staging it fills is
    /// free when the state drops.
    fn discard(&mut self, pending: &Pending) -> Result<(), EngineError> {
        match self.engine.snapshot_wait(pending) {
            Ok(_)
            | Err(
                StateError::Format(_)
                | StateError::Io(_)
                | StateError::Unsupported(_)
                | StateError::NoRoom { .. },
            ) => Ok(()),
            Err(StateError::Engine(e)) => Err(e),
        }
    }

    /// `slot`'s state copy landed, waited for if it still runs; what the
    /// request path waited is in the note. Nothing when none runs.
    fn settle(&mut self, slot: usize) -> Result<(), EngineError> {
        if let Some(p) = self.saving[slot].take() {
            self.land(&p)?;
        }
        Ok(())
    }

    /// The copies that have landed, settled and noted, none waited for.
    fn refresh(&mut self) -> Result<(), EngineError> {
        for p in self.cache.in_flight() {
            match self.engine.snapshot_poll(&p) {
                Ok(Some(landed)) => self.landed(&p, landed, 0.0),
                Ok(None) => {}
                Err(StateError::Engine(e)) => return Err(e),
                Err(e) => self.lose(&p, &e),
            }
        }
        Ok(())
    }

    /// `p`'s copy landed: waited for when it still runs. Returns the ms the
    /// request path waited, or `None` when the copy failed in a way that is
    /// the state's alone (it leaves the cache; an engine failure is the
    /// request's error).
    fn land(&mut self, p: &Pending) -> Result<Option<f64>, EngineError> {
        let t = Instant::now();
        let (landed, waited_ms) = match self.engine.snapshot_poll(p) {
            Ok(Some(landed)) => (landed, 0.0),
            Ok(None) => match self.engine.snapshot_wait(p) {
                Ok(landed) => (landed, ms_since(t)),
                Err(StateError::Engine(e)) => return Err(e),
                Err(e) => {
                    self.lose(p, &e);
                    return Ok(None);
                }
            },
            Err(StateError::Engine(e)) => return Err(e),
            Err(e) => {
                self.lose(p, &e);
                return Ok(None);
            }
        };
        self.landed(p, landed, waited_ms);
        Ok(Some(waited_ms))
    }

    /// `p`'s copy landed with `landed`, the request path having waited
    /// `waited_ms`: its entry may leave the cache from here on.
    fn landed(&mut self, p: &Pending, landed: Landed, waited_ms: f64) {
        for s in &mut self.saving {
            if s.as_ref().is_some_and(|q| q.is(p)) {
                *s = None;
            }
        }
        if self.cache.settle(p) {
            self.engine.note(&CacheNote::Landed {
                positions: p.state().n_tokens(),
                bytes: p.state().n_bytes(),
                calls: landed.calls,
                copy_ms: landed.copy_ms,
                alloc_ms: landed.alloc_ms,
                waited_ms,
            });
        }
    }

    /// `p`'s copy failed for a reason that is the state's alone: the state
    /// leaves the cache and is noted.
    fn lose(&mut self, p: &Pending, why: &StateError) {
        for s in &mut self.saving {
            if s.as_ref().is_some_and(|q| q.is(p)) {
                *s = None;
            }
        }
        self.cache.settle(p);
        self.cache.remove(p.state());
        self.engine.note(&CacheNote::Skip {
            positions: p.state().n_tokens(),
            why: format!("the copy of the state failed: {why}"),
        });
    }

    /// The first `n` positions of a cached state into the engine, the slot
    /// then holding them; returns what went back: the state's positions and
    /// bytes, the positions the engine put back, the wall time of the resume
    /// and what the request path waited for the state's copy to land first.
    /// A state the engine refuses leaves the cache, the engine is reset and
    /// the slot holds nothing (`None`); an engine failure is the request's
    /// error.
    fn load(&mut self, taken: Taken, n: usize) -> Result<Option<Loaded>, EngineError> {
        let Taken {
            mut held,
            state,
            pending,
        } = taken;
        self.held = Held::default();
        let t = Instant::now();
        let waited_ms = match &pending {
            Some(p) => match self.land(p)? {
                Some(ms) => ms,
                None => return self.refuse_load(&state, held.ids.len(), "its copy failed"),
            },
            None => 0.0,
        };
        match self.rows()?.resume_prefix(&state, n) {
            Ok(resumed) => {
                let positions = held.ids.len();
                if resumed > positions {
                    return Err(EngineError(format!(
                        "the engine put back {resumed} positions of a state of {positions}"
                    )));
                }
                held.truncate(resumed);
                let got = Loaded {
                    positions,
                    bytes: state.n_bytes(),
                    ms: ms_since(t),
                    resumed,
                    waited_ms,
                };
                self.prompts[self.cur] = held.ids.len();
                self.held = held;
                Ok(Some(got))
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.cache.remove(&state);
                self.rows()?.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: held.ids.len(),
                    why: format!("the engine refused to take the state back: {e}"),
                });
                Ok(None)
            }
        }
    }

    /// A cached state that cannot be taken back, for `why`: it leaves the
    /// cache, the engine is reset and the slot holds nothing.
    fn refuse_load(
        &mut self,
        state: &Arc<dyn Saved>,
        positions: usize,
        why: &str,
    ) -> Result<Option<Loaded>, EngineError> {
        self.cache.remove(state);
        self.rows()?.reset()?;
        self.engine.note(&CacheNote::Skip {
            positions,
            why: format!("the engine refused to take the state back: {why}"),
        });
        Ok(None)
    }

    /// The slot's own handed-back `state` into the engine; true once it is
    /// there. A state the engine refuses leaves the cache (where the rule
    /// saved it just now), the engine is reset and the slot holds nothing
    /// (false); an engine failure is the request's error.
    fn put_back(&mut self, state: &Arc<dyn Saved>) -> Result<bool, EngineError> {
        let held = std::mem::take(&mut self.held);
        match self.engine.resume(state) {
            Ok(()) => {
                self.held = held;
                Ok(true)
            }
            Err(StateError::Engine(e)) => Err(e),
            Err(e) => {
                self.cache.remove(state);
                self.engine.reset()?;
                self.engine.note(&CacheNote::Skip {
                    positions: held.len(),
                    why: format!("the engine refused to take the slot's parked state back: {e}"),
                });
                Ok(false)
            }
        }
    }

    /// The marks of the prompt call `ids[from..to]`, the messages the prompt
    /// opens ([`Tokenizer::user_start`]): its first and its last inside the
    /// range; and where the engine cuts the call into calls among them
    /// ([`Engine::prefill_splits`]). An answer not among those marks in
    /// order is the engine's error.
    fn marked_cuts(&self, ids: &[u32], from: usize, to: usize) -> Result<Marks, EngineError> {
        let marks = message_starts(&ids[..to], &self.vocab.user_start(), from);
        let cuts = if marks.is_empty() {
            Vec::new()
        } else {
            self.engine.prefill_splits(from, to, &marks)
        };
        if !cuts.iter().all(|u| marks.contains(u)) || !cuts.windows(2).all(|w| w[0] < w[1]) {
            return Err(EngineError(format!(
                "the engine cut the prompt call {from}..{to} at {cuts:?}, not among its marks \
                 {marks:?} in order"
            )));
        }
        Ok(Marks { marks, cuts })
    }

    /// The note of a prompt call `first..end` the engine cut at its marks
    /// `at`; none when it cut nowhere.
    fn note_split(&self, first: usize, end: usize, at: Vec<usize>) {
        if !at.is_empty() {
            self.engine.note(&CacheNote::Split { first, end, at });
        }
    }

    /// [`Engine::prefill_marked`] of the prompt's positions
    /// `at..at + ids.len()` ([`call_parts`] of them: `media` the prompt's
    /// feeds, `marks` the prompt call's) that books what it fed.
    fn prefill_range(
        &mut self,
        ids: &[u32],
        at: usize,
        media: &[MediaFeed],
        marks: &[usize],
    ) -> Result<(), EngineError> {
        let (feeds, inside) = call_parts(at, at + ids.len(), media, marks)?;
        let held = std::mem::take(&mut self.held);
        self.engine.prefill_marked(ids, &feeds, &inside)?;
        self.held = held;
        self.book(ids, at, &feeds);
        Ok(())
    }

    /// `ids` fed at the prompt's positions `at..` with `feeds` (their spans
    /// at the call's own positions), booked as held.
    fn book(&mut self, ids: &[u32], at: usize, feeds: &[MediaFeed]) {
        self.held.ids.extend_from_slice(ids);
        self.held.media.extend(feeds.iter().map(|f| MediaSpan {
            at: at + f.at,
            ..f.span()
        }));
    }

    /// `next` that books what it fed.
    pub(crate) fn next(
        &mut self,
        last: u32,
        logits: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let held = std::mem::take(&mut self.held);
        let g = self.engine.next(last, logits)?;
        self.held = held;
        self.held.push(last);
        Ok(g)
    }

    /// [`Engine::next_after`] that books what it fed: `ids`, at least one.
    pub(crate) fn next_after(
        &mut self,
        ids: &[u32],
        logits: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let held = std::mem::take(&mut self.held);
        let g = self.engine.next_after(ids, logits)?;
        self.held = held;
        self.held.ids.extend_from_slice(ids);
        Ok(g)
    }

    /// The prompt call's last call `ids` at the prompt's positions `at..`
    /// (`media`, the prompt's feeds; `marks`, the prompt call's) and the step
    /// of its last id as one [`Engine::prefill_next`], every id booked
    /// ([`Slot::book`]). A prompt call takes the checkpoints of its marks, so
    /// it is never an answer run's [`Engine::next_after`].
    fn close_call(
        &mut self,
        (ids, at, media, marks): Call<'_>,
        logits: Option<&mut [f32]>,
    ) -> Result<u32, EngineError> {
        let run = ids.len().checked_sub(1).expect("the prompt's last id");
        let (feeds, inside) = call_parts(at, at + run, media, marks)?;
        let held = std::mem::take(&mut self.held);
        let g = self.engine.prefill_next(ids, &feeds, &inside, logits)?;
        self.held = held;
        self.book(ids, at, &feeds);
        Ok(g)
    }

    /// `advance` that books what it fed: `last` and every kept token but the
    /// last. `out` is cleared first. A pass that kept no token, more than the
    /// engine's rows, or other than one more than its draft's accepted ids is
    /// the engine's error.
    pub(crate) fn advance(
        &mut self,
        last: u32,
        out: &mut Vec<u32>,
    ) -> Result<Drafted, EngineError> {
        let held = std::mem::take(&mut self.held);
        out.clear();
        let d = self.engine.advance(last, out)?;
        self.check_pass(out, d)?;
        self.held = held;
        self.held.push(last);
        self.held.ids.extend_from_slice(&out[..out.len() - 1]);
        Ok(d)
    }

    /// [`Engine::advance_sampled`] that books what it fed and checks what it
    /// kept as [`Slot::advance`] does. The engine is lent a copy of
    /// `history` in `lent` (kept for its capacity), which must come back
    /// equal to it, else the engine's error; `sampler` comes back whatever
    /// the outcome.
    pub(crate) fn advance_sampled(
        &mut self,
        last: u32,
        history: &[u32],
        lent: &mut Vec<u32>,
        sampler: Sampler,
        best: Option<usize>,
        out: &mut Vec<u32>,
    ) -> (Result<Drafted, EngineError>, Sampler) {
        let held = std::mem::take(&mut self.held);
        out.clear();
        lent.clear();
        lent.extend_from_slice(history);
        let (d, back, sampler) =
            self.engine
                .advance_sampled(last, std::mem::take(lent), sampler, best, out);
        let d = d.and_then(|d| {
            self.check_pass(out, d)?;
            if back != history {
                return Err(EngineError(format!(
                    "a sampled pass was lent a history of {} ids and handed back another of {}",
                    history.len(),
                    back.len()
                )));
            }
            Ok(d)
        });
        *lent = back;
        if d.is_ok() {
            self.held = held;
            self.held.push(last);
            self.held.ids.extend_from_slice(&out[..out.len() - 1]);
        }
        (d, sampler)
    }

    /// A pass that kept no token, more than the engine's rows, or other than
    /// one more than its draft's accepted ids is the engine's error.
    fn check_pass(&self, out: &[u32], d: Drafted) -> Result<(), EngineError> {
        let rows = self.engine.advance_rows();
        if out.is_empty()
            || out.len() > rows
            || out.len() != d.accepted + 1
            || d.accepted > d.proposed
        {
            return Err(EngineError(format!(
                "a pass of at most {rows} rows kept {} tokens, its draft {} of {} proposed ids",
                out.len(),
                d.accepted,
                d.proposed
            )));
        }
        Ok(())
    }

    /// Writes the slot to `path` ([`slotfile`]'s layout): the file is written
    /// beside it under a temporary name and renamed over it only once complete,
    /// so a failed save leaves any earlier file at `path` as it was. The cache
    /// is unchanged. Returns the positions saved and the file's bytes.
    pub(crate) fn save(&self, path: &Path) -> Result<(usize, u64), StateError> {
        let partial = partial_path(path);
        let r = self.save_to(&partial).and_then(|bytes| {
            std::fs::rename(&partial, path)
                .map(|()| bytes)
                .map_err(StateError::from)
        });
        if r.is_err() {
            // The save's error is the answer; a partial file left behind is
            // named as one and never read.
            let _ = std::fs::remove_file(&partial);
        }
        r.map(|bytes| (self.held.len(), bytes))
    }

    fn save_to(&self, partial: &Path) -> Result<u64, StateError> {
        let mut w = Counting::new(BufWriter::new(File::create(partial)?));
        slotfile::write_header(&mut w, self.vocab.n_vocab(), &self.held)?;
        let head = w.bytes;
        let saved = self.engine.save_state(&mut w)?;
        if saved.n_tokens != self.held.len() || saved.n_bytes != w.bytes - head {
            return Err(StateError::Format(format!(
                "the engine saved {} positions in {} bytes; the slot holds {} positions and {} \
                 bytes reached the file",
                saved.n_tokens,
                saved.n_bytes,
                self.held.len(),
                w.bytes - head
            )));
        }
        w.flush()?;
        let file = w
            .inner
            .into_inner()
            .map_err(io::IntoInnerError::into_error)?;
        file.sync_all()?;
        Ok(w.bytes)
    }

    /// Replaces the cache with the slot saved in `path`. A file refused before
    /// the engine reads its part (unreadable, another tag, version or
    /// vocabulary, a count past the context, an id past the vocabulary, a
    /// span outside the ids) and an engine that does not restore leave the
    /// cache as it was; once the engine has read, any failure resets it and
    /// the slot holds nothing. Returns the positions restored and the file's
    /// bytes.
    pub(crate) fn restore(&mut self, path: &Path) -> Result<(usize, u64), StateError> {
        let mut r = Counting::new(BufReader::new(File::open(path)?));
        let stored = slotfile::read_header(&mut r, self.vocab.n_vocab(), self.engine.ctx_max())?;
        let head = r.bytes;
        self.settle(self.cur).map_err(StateError::Engine)?;
        let held = std::mem::take(&mut self.held);
        let restored = match self
            .rows()
            .map_err(StateError::Engine)?
            .restore_state(&mut r)
        {
            Ok(s) => s,
            Err(StateError::Unsupported(what)) => {
                self.held = held;
                return Err(StateError::Unsupported(what));
            }
            Err(e @ StateError::Engine(_)) => return Err(e),
            Err(e) => {
                self.prompts[self.cur] = 0;
                self.rows()?.reset()?;
                return Err(e);
            }
        };
        let read = r.bytes - head;
        let trailing = r.read(&mut [0u8; 1]);
        let mismatch = match trailing {
            Err(e) => Some(format!("reading past the engine's state: {e}")),
            Ok(n) if n > 0 => Some(format!(
                "the engine read {read} bytes and the file runs on past them"
            )),
            Ok(_) if restored.n_tokens != stored.len() || restored.n_bytes != read => {
                Some(format!(
                    "the engine restored {} positions from {} bytes; the file holds {} positions and \
                 the engine read {read} bytes",
                    restored.n_tokens,
                    restored.n_bytes,
                    stored.len()
                ))
            }
            Ok(_) => None,
        };
        if let Some(m) = mismatch {
            self.prompts[self.cur] = 0;
            self.rows()?.reset()?;
            return Err(StateError::Format(m));
        }
        let n = stored.len();
        self.held = stored;
        self.prompts[self.cur] = n;
        Ok((n, r.bytes))
    }

    /// Drops the whole cache and the ids it held; returns how many it held.
    pub(crate) fn erase(&mut self) -> Result<usize, EngineError> {
        if !self.off {
            self.settle(self.cur)?;
        }
        let held = std::mem::take(&mut self.held);
        self.prompts[self.cur] = 0;
        if !self.off {
            self.rows()?.reset()?;
        }
        Ok(held.len())
    }
}

/// The marks of a prompt call ([`Slot::marked_cuts`]): its message starts,
/// ascending, and the engine's cuts among them.
struct Marks {
    marks: Vec<usize>,
    cuts: Vec<usize>,
}

/// A call of a prompt: its ids, the position of the first, the prompt's
/// feeds and the prompt call's marks.
type Call<'a> = (&'a [u32], usize, &'a [MediaFeed], &'a [usize]);

/// The positions in `from + 1 .. ids.len()` where `marker` starts: the first
/// and the last of them (one when they are the same). An empty marker marks
/// nothing.
fn message_starts(ids: &[u32], marker: &[u32], from: usize) -> Vec<usize> {
    if marker.is_empty() {
        return Vec::new();
    }
    let mut at = (from + 1..ids.len()).filter(|&u| ids[u..].starts_with(marker));
    let first = at.next();
    let last = at.next_back();
    first.into_iter().chain(last).collect()
}

/// Where a save to `path` is written before it is renamed there: a name no
/// slot file can take (a slot file name has no `:`), and no other save's —
/// the process id and a count of the process's saves, so two servers of one
/// process saving into one directory never write one file.
fn partial_path(path: &Path) -> PathBuf {
    static SAVES: AtomicU64 = AtomicU64::new(0);
    let n = SAVES.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".bloomery-slot-{}-{n}:partial", std::process::id()))
}

/// What a prompt's opening kept ([`Gen::open`]): the positions the cache
/// keeps of it, where its prompt call starts, and the wall time of the cache's
/// work (`cache_ms`).
#[derive(Clone, Copy)]
pub(crate) struct Kept {
    pub cache_n: usize,
    cache_ms: f64,
}

/// A prompt the engine thread runs a call a round, between the other busy
/// slots' decode rounds ([`Engine::prompt_quantum`], [`Gen::plan`]): its
/// prompt call cut where the engine asks at its messages
/// ([`Slot::marked_cuts`]), and each of those calls cut again on the
/// quantum's grid ([`quantum_end`]). The cuts follow from the prompt and what
/// the cache kept, whatever the load, and each is one the quantum makes
/// bit-neutral, so the request's bits are those of the prompt run in one go.
pub(crate) struct Prompt {
    ids: Vec<u32>,
    /// The prompt's images, each riding the call that holds its span: the
    /// quantum does not cut inside a span.
    media: Vec<MediaFeed>,
    kept: Kept,
    /// The prompt call's message starts, each call handed those that fall in
    /// it ([`Engine::prefill_marked`]).
    marks: Vec<usize>,
    /// The engine's cuts at the messages, which the prompt call's note names
    /// as an uninterrupted call's does: a quantum's cut keeps nothing more.
    cuts: Vec<usize>,
    /// Every call's end, ascending; the last is the prompt call's, `n - 1`.
    ends: Vec<usize>,
    /// The calls run.
    ran: usize,
    progress: bool,
}

/// The part `at..end` of a prompt call as the engine takes it: the feeds of
/// the image spans it holds (of `media`, the prompt's feeds), at its own
/// positions, and the prompt call's `marks` that fall in it, past `at` and up
/// to its end. A part that would cut a span is the server's error: the cache
/// and the prompt's last position never lie inside one.
fn call_parts(
    at: usize,
    end: usize,
    media: &[MediaFeed],
    marks: &[usize],
) -> Result<(Vec<MediaFeed>, Vec<usize>), EngineError> {
    let mut feeds = Vec::new();
    for f in media {
        let s = f.span();
        if s.end() <= at || s.at >= end {
            continue;
        }
        if s.at < at || s.end() > end {
            return Err(EngineError(format!(
                "the prompt call {at}..{end} cuts the image span {}..{}",
                s.at,
                s.end()
            )));
        }
        feeds.push(MediaFeed {
            at: s.at - at,
            ..f.clone()
        });
    }
    let inside = (marks.iter().copied())
        .filter(|&u| at < u && u <= end)
        .collect();
    Ok((feeds, inside))
}

/// The ends of the calls the prompt call `from..to` runs as: cut at `marks`
/// (ascending, inside the range), and each of those calls on the quantum's
/// grid ([`quantum_end`], `media` the prompt's feeds).
fn quantum_ends(
    from: usize,
    to: usize,
    marks: &[usize],
    media: &[MediaFeed],
    q: usize,
) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut start = from;
    for &end in marks.iter().chain([&to]) {
        while start < end {
            start = quantum_end(start, end, media, q);
            ends.push(start);
        }
    }
    ends
}

/// The end of the next call of a run of positions `at..end` fed on the grid
/// of a quantum `q` ([`Engine::prompt_quantum`]): the first multiple of `q`
/// past `at` that lies before `end` and inside no image span of `media`, when
/// more than `q` positions are left; `end` otherwise. The grid is the
/// positions', so every cut falls on the same positions whatever the call
/// started at, and a span never crosses calls.
fn quantum_end(at: usize, end: usize, media: &[MediaFeed], q: usize) -> usize {
    if end - at <= q {
        return end;
    }
    let inside = |u: usize| {
        media
            .iter()
            .map(MediaFeed::span)
            .any(|s| s.at < u && u < s.end())
    };
    (at / q + 1..)
        .map(|j| j * q)
        .take_while(|&u| u < end)
        .find(|&u| !inside(u))
        .unwrap_or(end)
}

/// What a generation needs next ([`Gen::pump`]).
pub(crate) enum Need {
    /// One step from this id: [`Engine::next`], or a row of
    /// [`Engine::step_slots`], whose answer goes to [`Gen::stepped`].
    Step(u32),
    /// One drafted pass from this id ([`Engine::advance`] into
    /// [`Gen::kept_mut`]), whose count goes to [`Gen::advanced`].
    Advance(u32),
    /// One drafted pass of a request that samples from this id:
    /// [`Gen::advance_sampled`] when its slot is the round's only busy one,
    /// beside another a row of [`Engine::advance_slots`] ([`Gen::pass_row`]).
    Sampled(u32),
    /// A run of ids the program fixes, fed as a call of [`Engine::prefill`]
    /// ([`Gen::fill`]): the engine thread runs it like a prompt's call, and
    /// the request's [`Need::Step`] follows. It is never part of a step.
    Fill,
    /// Nothing more: [`Gen::finish`] ends it.
    Done,
}

/// The fewest ids a run is fed as a call: below it the steps cost the same,
/// and a seat may route one id to its step path.
pub(crate) const MIN_FILL: usize = 2;

/// A run of fixed ids being fed ([`Need::Fill`]).
struct Span {
    /// The ids the calls feed: the id just taken, then the run but its last
    /// id, which the step that follows feeds.
    feed: Vec<u32>,
    fed: usize,
    /// The run: the ids the loop takes next.
    run: Vec<u32>,
    /// Whether the span's last call steps the run's last id too
    /// ([`Engine::next_after`]): `feed` then holds the whole run after the id
    /// just taken.
    close: bool,
}

/// The longest cycle the loop guard looks for, in ids.
const LOOP_PERIOD_MAX: usize = 1024;

/// The fewest ids in a row a loop repeats: each of the span's last ids equals
/// the id one period before it, over this many ids and one whole period.
const LOOP_RUN_MIN: usize = 256;

/// The loop guard's state over the open think span: for each period `p` from
/// 1 to [`LOOP_PERIOD_MAX`], how many of the span's last ids in a row equal
/// the id `p` before them. A run that reaches [`LOOP_RUN_MIN`] and `p` is a
/// loop: the span ends in two whole copies of a `p`-id cycle at least, and
/// [`LOOP_RUN_MIN`] ids or more repeat. One taken id costs a compare a period.
struct Repeats {
    /// Where the span's ids start among the generated ids.
    from: usize,
    /// The run of period `p` at `p - 1`.
    runs: Vec<usize>,
}

impl Repeats {
    fn new() -> Repeats {
        Repeats {
            from: 0,
            runs: vec![0; LOOP_PERIOD_MAX],
        }
    }

    /// Whether the span's ids `span`, after its last id is taken, end in a
    /// loop.
    fn looped(&mut self, span: &[u32]) -> bool {
        let Some((&last, before)) = span.split_last() else {
            return false;
        };
        for (p, run) in (1..=before.len()).zip(self.runs.iter_mut()) {
            if before[before.len() - p] != last {
                *run = 0;
                continue;
            }
            *run += 1;
            if *run >= LOOP_RUN_MIN.max(p) {
                return true;
            }
        }
        false
    }
}

/// The think span's state in [`Gen`], on a request with a budget or the loop
/// guard: the budget left while the span is open, the span's close detection
/// over the generated text (the output's own split is api-side; this instance
/// only decides when the span closes), the ids of the close, `at` naming the
/// next of them the loop force-feeds — `close.len()` while the budget holds —
/// and the loop guard's [`Repeats`].
struct Think {
    /// Budget left, `usize::MAX` without one; the id whose text opens a
    /// model-opened span and the id whose text closes the span do not spend.
    left: usize,
    split: ThinkSplit,
    close: Vec<u32>,
    at: usize,
    /// The loop guard, `None` on a request that did not ask for it.
    repeats: Option<Repeats>,
}

impl Think {
    /// A budget of `left` generated ids (`None`: no budget) over the span
    /// `split` reads, `close` the ids of its close tag, and the loop guard
    /// when `guard`; at 0 the close is forced from the first taken id inside
    /// the span, which for a model-opened span is the first after its start
    /// tag.
    fn new(split: ThinkSplit, left: Option<usize>, close: Vec<u32>, guard: bool) -> Think {
        let n = close.len();
        let left = left.unwrap_or(usize::MAX);
        Think {
            left,
            split,
            close,
            at: if left == 0 { 0 } else { n },
            repeats: guard.then(Repeats::new),
        }
    }

    /// Whether a close id is queued to force: the budget spent with the span
    /// open — a span the model may still open forces nothing.
    fn forcing(&self) -> bool {
        self.split.in_span() && self.at < self.close.len()
    }

    /// The next close id to force.
    fn next_forced(&mut self) -> Option<u32> {
        self.forcing().then(|| {
            let id = self.close[self.at];
            self.at += 1;
            id
        })
    }
}

/// One request's generation on its slot, run a call at a time so the engine
/// thread can step several slots' generations in one engine call. `ids` is
/// non-empty and shorter than the context. [`Gen::timings`] are filled even
/// when the sink failed midway (the caller's counters still see the work
/// done). A request that reads no row ([`Pick::reads_rows`]) never asks the
/// engine for its logits; one that does takes them from its steps' rows or
/// its passes' verified rows ([`takes_passes`]). `tick` sees the timings
/// after every generated token.
pub(crate) struct Gen {
    n: usize,
    ctx_max: usize,
    /// What takes each generated id from its row, the request's own.
    pick: Pick,
    /// Where the pick is while a pass holds it, and an empty one between
    /// passes ([`Gen::lend`]); and the sampler a pass is lent, whose every
    /// draw asks the pick there.
    lent: Arc<Mutex<Pick>>,
    draw: Option<Sampler>,
    /// Whether a pass of several slots holds the pick, from its row's making
    /// ([`Gen::pass_row`]) to the round's count ([`Gen::advanced`]).
    lending: bool,
    /// The most of a row's best logits the pick reads in a pass
    /// ([`Engine::advance_sampled`]'s `best`, [`best_read`]).
    best: Option<usize>,
    stops: Vec<u32>,
    /// The engine's logits row, or empty on a request that reads none.
    logits: Vec<f32>,
    tim: Timings,
    /// The draft's counts by depth ([`book_depths`]): their sums are the
    /// timings' `draft_n` and `draft_n_accepted`. Kept apart from the
    /// timings, which every event copies.
    depths: Vec<Depth>,
    t1: Instant,
    budget: usize,
    scan: StopScan,
    dec: Box<dyn Decoder>,
    /// The vocabulary, for which generated ids are special.
    vocab: Arc<dyn Tokenizer>,
    /// Where a special id's text is in the scan's text, ascending: byte
    /// offset and byte length.
    specials: Vec<(usize, usize)>,
    generated: Vec<u32>,
    /// The ids a draw sees, in order: the prompt's tail (its last
    /// `repeat_last_n` ids, as much of the prompt as there is), then the
    /// generated ids. llama-server feeds every prompt id into the sampler
    /// before the first draw; only the window the penalties read is kept, so
    /// a long prompt costs one bounded copy, not a copy a step. Only a
    /// request that draws (which has a sampler) keeps it.
    history: Vec<u32>,
    stop: StopKind,
    stopping_word: String,
    truncated: bool,
    /// A pass's rows; 1 steps. A request passes when it reads no row, or
    /// reads its rows ([`Pick::reads_rows`]) on an engine that drafts sampled
    /// requests.
    rows: usize,
    /// The last pass's kept tokens, and how many of them the loop has taken.
    kept: Vec<u32>,
    taken: usize,
    /// The program's refusal of a pass's row past its first, or a watch's
    /// refusal of the text an id it took wrote, returned once the ids before
    /// it, all `kept` holds, are taken; a stop among them drops it.
    refused_after: Option<StepError>,
    /// The copy of the history a sampled pass lends the engine
    /// ([`Slot::advance_sampled`]), kept for its capacity.
    lent_history: Vec<u32>,
    /// The token the loop takes next; `None` once there is none.
    tok: Option<u32>,
    /// The think-span budget and loop guard, `None` on a request with neither
    /// (or whose prompt closed the span) and once the span closed.
    think: Option<Think>,
    /// Whether the request takes the jump arm: it follows a program, asks for
    /// no probabilities, has no think budget and no penalty, and the server
    /// has the arm on.
    jump: bool,
    /// The run being fed, `None` between runs.
    span: Option<Span>,
    /// The answer's lead the prompt call carried ([`Gen::take_lead`]).
    lead: Vec<u32>,
    /// The stop id a jump-arm answer ends with once the program is spent
    /// and nothing it fixes is left ([`Gen::tail_id`]): the vocabulary's end
    /// of sequence, `None` when its stops do not hold it (the stop is then
    /// picked from a row).
    end: Option<u32>,
    /// Whether the answer is in its tail: every id left is the program's or
    /// [`Gen::end`], taken without an engine call.
    tail: bool,
}

/// The most of a row's best logits a request of `p` picks its pass's ids
/// from ([`Engine::advance_sampled`]'s `best`): its sampler's
/// ([`SamplingParams::best_read`]) or a greedy request's argmax, plus each
/// id its `bias` lowers, which can make room among the best for the next.
/// `None`, the whole row, for a request that asks for probabilities or
/// follows a program, and for a bias that raises an id, which can come from
/// anywhere in the row.
fn best_read(p: &GenParams, bias: &Bias, greedy: bool) -> Option<usize> {
    let raises = bias.add.iter().any(|&(_, b)| b > 0.0);
    if p.logprobs.is_some() || p.constraint.is_some() || raises {
        return None;
    }
    let k = if greedy { 1 } else { p.sampling.best_read()? };
    Some(k + bias.add.len() + bias.banned.len())
}

/// The sampler a request of `p` draws with, `None` for a greedy one, or the
/// factory's refusal of its parameters. A penalized request at temperature 0
/// takes the sampler's argmax after the penalties, as llama-server's chain
/// does; only an unpenalized one is greedy.
pub(crate) fn sampler_of(
    factory: &SamplerFactory,
    p: &GenParams,
) -> Result<Option<Sampler>, SamplerRefused> {
    (p.sampling.temperature > 0.0 || p.sampling.penalizes())
        .then(|| factory(&p.sampling))
        .transpose()
}

impl Gen {
    /// A generation of `p` for a prompt of `n` ids on `slot`'s engine, drawing
    /// with `sampler` ([`sampler_of`]); nothing runs yet.
    pub(crate) fn new(slot: &Slot, sampler: Option<Sampler>, n: usize, p: &GenParams) -> Gen {
        let stops = slot.vocab.stops();
        let end = Some(slot.vocab.eos()).filter(|e| stops.contains(e));
        let bias = Bias {
            add: p.logit_bias.clone(),
            banned: if p.ignore_eos {
                stops.clone()
            } else {
                Vec::new()
            },
        };
        let greedy = sampler.is_none();
        let best = best_read(p, &bias, greedy);
        // A program picks from its own biased copy of the row ([`Pick::choose`]).
        let sampler = if bias.is_empty() || p.constraint.is_some() {
            sampler
        } else {
            Some(biased(sampler, bias.clone()))
        };
        let open = p.think_entry == ThinkEntry::Open;
        let (walk, held) = match &p.constraint {
            // A forced call and a watched one both wait for the model's own
            // close of a span the prompt left open; a watched call waits for
            // a span the model may still open too, an opener inside its own
            // span reasoning as the parser reads it.
            Some(c)
                if c.follows_reasoning()
                    && (open || (p.think_entry == ThinkEntry::MayOpen && c.waits_for_call())) =>
            {
                (None, Some(Walk::new(c, false)))
            }
            c => (c.as_ref().map(|c| Walk::new(c, open)), None),
        };
        // The span's split the loop's think tracker reads, and a watch's pick
        // while its program is held: both end the span at one id.
        let split = || {
            p.think_split.clone().unwrap_or_else(|| {
                ThinkSplit::for_entry_with(p.think_entry, p.think).reading_specials(&*slot.vocab)
            })
        };
        let free = p
            .constraint
            .as_ref()
            .filter(|c| c.waits_for_call())
            .map(|_| FreeText {
                dec: slot.vocab.decoder(),
                vocab: Arc::clone(&slot.vocab),
                split: held.is_some().then(split),
            });
        let pick = Pick {
            sampler,
            bias,
            greedy,
            probs: p.logprobs.as_ref().map(Ask::collector),
            walk,
            held,
            free,
            biased_row: Vec::new(),
            masked: Vec::new(),
            answered: 0,
            refused: None,
        };
        // A greedy request that changes no row reads the logits only for the
        // probabilities it asked for, or to pick among the ids its program
        // allows.
        let reads = pick.reads_rows();
        let holds = pick.held.is_some();
        let logits = vec![0.0f32; if reads { slot.vocab.n_vocab() } else { 0 }];
        let rows = if takes_passes(reads, slot.engine.as_ref()) {
            slot.engine.advance_rows()
        } else {
            1
        };
        let lent = Arc::new(Mutex::new(Pick::default()));
        let draw: Sampler = {
            let lent = Arc::clone(&lent);
            Box::new(move |row: &[f32], history: &[u32]| {
                relock(&lent).answer(row, history, None, None)
            })
        };
        Gen {
            n,
            ctx_max: slot.engine.ctx_max(),
            pick,
            lent,
            draw: Some(draw),
            lending: false,
            best,
            stops,
            logits,
            tim: Timings::default(),
            depths: Vec::new(),
            t1: Instant::now(),
            budget: usize::try_from(p.n_predict).unwrap_or(usize::MAX),
            scan: StopScan::new(p.stop.clone()),
            dec: slot.vocab.decoder(),
            vocab: Arc::clone(&slot.vocab),
            specials: Vec::new(),
            generated: Vec::new(),
            history: Vec::new(),
            stop: StopKind::Limit,
            stopping_word: String::new(),
            truncated: false,
            rows,
            kept: Vec::with_capacity(rows),
            taken: 0,
            refused_after: None,
            lent_history: Vec::new(),
            tok: None,
            think: (p.reasoning_budget.is_some() || p.reasoning_loop || holds).then(|| {
                Think::new(
                    split(),
                    p.reasoning_budget,
                    slot.vocab.encode(p.think.close),
                    p.reasoning_loop,
                )
            }),
            // A think budget's forced close would compete with a run for the
            // ids after it, and a penalized draw reads the ids a run feeds
            // after it, so a request with either steps.
            jump: slot.jump
                && p.constraint.is_some()
                && p.logprobs.is_none()
                && p.reasoning_budget.is_none()
                && !p.sampling.penalizes(),
            span: None,
            lead: Vec::new(),
            end,
            tail: false,
        }
    }

    /// The timings so far.
    pub(crate) fn timings(&self) -> &Timings {
        &self.tim
    }

    /// The draft's counts by depth so far, depth `d` at index `d - 1`.
    pub(crate) fn draft_depths(&self) -> &[Depth] {
        &self.depths
    }

    /// The buffer a pass's kept tokens go to.
    pub(crate) fn kept_mut(&mut self) -> &mut Vec<u32> {
        &mut self.kept
    }

    /// The logits buffer a step writes, or `None` on a greedy request and on
    /// a step whose token the program fixes, unless probabilities were asked.
    /// Nor does a greedy request that changes no row while its program is
    /// held, or while its watch waits in free text: its reasoning and its
    /// free text take the engine's argmax.
    pub(crate) fn logits_out(&mut self) -> Option<&mut [f32]> {
        let k = &mut self.pick;
        let unread = k.walk.as_mut().is_some_and(Walk::fixed_next)
            || (k.in_free_text() && k.greedy && k.bias.is_empty());
        if unread && k.probs.is_none() {
            return None;
        }
        out(&mut self.logits)
    }

    /// The id the loop takes for an engine answer `g`: the next forced close id
    /// while one is queued (the engine's answer is discarded — the loop never
    /// commits it, so nothing desyncs), else the request's pick from the row
    /// ([`Pick::choose`]). A request that asks for probabilities reads the
    /// row as the engine wrote it. A watch's refusal of the text the id
    /// wrote is returned once the id is taken ([`Gen::pump`]).
    fn answer(&mut self, g: u32) -> u32 {
        let forced = self.think.as_mut().and_then(Think::next_forced);
        let id = self
            .pick
            .answer(&self.logits, &self.history, forced, Some(g));
        self.count_picks();
        let answered = self.pick.answered;
        if let Some((_, e)) = self.pick.refused.take_if(|(at, _)| *at == answered) {
            self.refused_after = Some(e);
        }
        id
    }

    /// The program's pick counts into the jump arm's timings.
    fn count_picks(&mut self) {
        if let (Some(j), Some(walk)) = (self.tim.jump.as_mut(), self.pick.walk.as_ref()) {
            let picks = walk.picks();
            j.decisions = picks.decisions;
            j.free_steps = picks.free;
        }
    }

    /// Whether a close id is queued to force: the budget spent with the span
    /// still open.
    fn forcing(&self) -> bool {
        self.think.as_ref().is_some_and(Think::forcing)
    }

    /// The think budget left, `usize::MAX` with none: a pass keeps at most
    /// `rows` tokens and each spends, so it starts only while they all fit.
    fn think_left(&self) -> usize {
        self.think.as_ref().map_or(usize::MAX, |t| t.left)
    }

    /// The special ids' texts in `text`, the scan's text it last sent, as
    /// offsets into it with their lengths.
    fn sending(&self, text: &str) -> Vec<(usize, usize)> {
        let to = self.scan.sent();
        let from = to - text.len();
        let range = &self.specials[self.specials.partition_point(|&(at, _)| at < from)..];
        range
            .iter()
            .take_while(|&&(at, _)| at < to)
            .map(|&(at, len)| (at - from, len))
            .collect()
    }

    /// One taken id against the think budget and the loop guard: its text
    /// first — a piece that closes the span retires the tracker, the closing
    /// id spending nothing, and starts a held program; a piece that opens a
    /// model-opened span spends nothing either and starts the guard's span
    /// after it — then, only for a span already open, the count, which arms
    /// the forced close at zero, and the guard. Whether the guard found a
    /// loop, or the refusal of a close whose piece also wrote text past it
    /// that the held program did not pick.
    fn spend(
        &mut self,
        piece: Option<&str>,
        special: Option<(usize, usize)>,
        stop: Option<&str>,
    ) -> Result<bool, StepError> {
        let Some(t) = self.think.as_mut() else {
            return Ok(false);
        };
        if let Some(text) = piece {
            let s = t.split.push(text, special.as_slice());
            // The splitter sees even the stop scan's unsent prefix. At
            // handoff no unchosen text may remain there or in the decoder;
            // a matched stop discards the decoder tail instead of sending it.
            if self.pick.held.is_some()
                && (!s
                    .content
                    .trim_matches(crate::reasoning::is_space)
                    .is_empty()
                    || (s.closed && stop.is_none() && !self.dec.flush().is_empty()))
            {
                return Err(StepError::NoAllowed {
                    taken: self.generated.len(),
                    why: "the think span's close also wrote text outside the forced call's program"
                        .to_owned(),
                });
            }
            if s.closed || !s.content.is_empty() {
                self.think = None;
                let k = &mut self.pick;
                k.walk = k.walk.take().or_else(|| k.held.take());
                return Ok(false);
            }
            if s.opened {
                if let Some(r) = t.repeats.as_mut() {
                    r.from = self.generated.len();
                }
                return Ok(false);
            }
        }
        if !t.split.in_span() {
            return Ok(false);
        }
        if t.left > 0 {
            t.left -= 1;
            if t.left == 0 {
                t.at = 0;
            }
        }
        Ok(t.repeats.as_mut().is_some_and(|r| {
            let from = r.from;
            r.looped(&self.generated[from..])
        }))
    }

    /// The prompt on the selected slot: what the cache keeps of it (`held`),
    /// the rest fed — its images with it ([`Engine::prefill_media`]) — and
    /// the first token's logits read.
    #[cfg(test)]
    pub(crate) fn prompt(
        &mut self,
        slot: &mut Slot,
        held: &Held,
        media: &[MediaFeed],
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let kept = self.open(slot, held, p)?;
        self.prompt_whole(slot, &held.ids, media, kept, p, sink)
    }

    /// The prompt's opening on the selected slot: the engine told the reply
    /// the request may make, the cache brought to the longest prefix of
    /// `held` it keeps, a request that draws seeds its sampler's history
    /// with the prompt's tail, and on the jump arm the answer's lead taken
    /// from the walk ([`Gen::take_lead`]). No id of the prompt is fed yet.
    pub(crate) fn open(
        &mut self,
        slot: &mut Slot,
        held: &Held,
        p: &GenParams,
    ) -> Result<Kept, GenError> {
        // The history the draws read opens on the prompt's tail, the whole
        // prompt's last `repeat_last_n` ids — the cached part included, as
        // llama-server feeds `prompt.tokens` whole into the sampler.
        if self.pick.sampler.is_some() {
            self.history = prompt_tail(held, p.sampling.repeat_last_n);
        }
        // A request that passes has its first token from the prompt's step:
        // the passes make at most the rest.
        let passes = takes_passes(self.pick.reads_rows(), slot.engine.as_ref());
        slot.engine.will_reply(match usize::try_from(p.n_predict) {
            Ok(n) if passes => Some(n.saturating_sub(1)),
            Err(_) if passes => None,
            _ => Some(0),
        });
        self.take_lead();
        let t = Instant::now();
        let cache_n = slot.reuse(held, p.cache_prompt)?;
        Ok(Kept {
            cache_n,
            cache_ms: ms_since(t),
        })
    }

    /// The prompt after its opening in one go: `ids[cache_n..n-1]` fed as one
    /// call, cut where the engine asks at its messages, and the last id
    /// stepped. On the jump arm the call carries the answer's lead
    /// ([`Gen::take_lead`]).
    pub(crate) fn prompt_whole(
        &mut self,
        slot: &mut Slot,
        ids: &[u32],
        media: &[MediaFeed],
        kept: Kept,
        p: &GenParams,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let t0 = Instant::now();
        let ids = if self.lead.is_empty() {
            Cow::Borrowed(ids)
        } else {
            Cow::Owned([ids, &self.lead].concat())
        };
        let to = ids.len() - 1;
        let Marks { marks, cuts } = slot.marked_cuts(&ids, kept.cache_n, to)?;
        let mut first = kept.cache_n;
        for &u in &cuts {
            slot.prefill_range(&ids[first..u], first, media, &marks)?;
            first = u;
        }
        slot.note_split(kept.cache_n, to, cuts);
        self.close(
            slot,
            (&ids[first..], first, media, &marks),
            kept,
            t0,
            p.return_progress,
            sink,
        )
    }

    /// The fixed ids that open the answer, which the jump arm's prompt call
    /// carries: the walk's run from its start, at most the ids the request
    /// may generate but one and the positions the cache holds past the
    /// prompt (the step arm feeds the run's `i`-th id at position `n + i`
    /// while it has generated `i + 1 < n_predict` ids and `n + i + 1 <=
    /// ctx_max`), and only when the program goes on past it: the close's
    /// step of its last id gives the row the next pick reads. Empty
    /// otherwise. The lead is held in [`Gen::lead`]; the walk stands past it.
    fn take_lead(&mut self) {
        self.lead.clear();
        if !self.jump {
            return;
        }
        let room = self
            .budget
            .saturating_sub(1)
            .min(self.ctx_max.saturating_sub(self.n));
        let Some(walk) = self.pick.walk.as_mut() else {
            return;
        };
        let mut ahead = walk.ahead();
        let Ok(run) = ahead.forced_run(room) else {
            return;
        };
        if run.is_empty() || ahead.spent() {
            return;
        }
        *walk = ahead;
        self.lead = run;
    }

    /// How many ids of the answer's lead the prompt call carries
    /// ([`Gen::take_lead`], taken at [`Gen::open`]).
    pub(crate) fn lead_len(&self) -> usize {
        self.lead.len()
    }

    /// The prompt after its opening as calls the engine thread runs one a
    /// round ([`Gen::prompt_call`]), its prompt call cut where the engine asks
    /// at its messages and each of those calls on the grid of `q`
    /// ([`Prompt`]). No call runs here; from here the timings carry the prompt
    /// so far. On the jump arm the calls carry the answer's lead
    /// ([`Gen::take_lead`]), as [`Gen::prompt_whole`]'s do.
    pub(crate) fn plan(
        &mut self,
        slot: &Slot,
        mut ids: Vec<u32>,
        media: Vec<MediaFeed>,
        kept: Kept,
        q: NonZeroUsize,
        progress: bool,
    ) -> Result<Prompt, GenError> {
        let t = Instant::now();
        ids.extend_from_slice(&self.lead);
        let to = ids.len() - 1;
        let Marks { marks, cuts } = slot.marked_cuts(&ids, kept.cache_n, to)?;
        let ends = quantum_ends(kept.cache_n, to, &cuts, &media, q.get());
        self.tim = Timings {
            prompt_ms: ms_since(t),
            n_ctx: self.ctx_max,
            n_past: kept.cache_n,
            cache_n: kept.cache_n,
            cache_ms: kept.cache_ms,
            n_prompt: self.n,
            ..Timings::default()
        };
        Ok(Prompt {
            ids,
            media,
            kept,
            marks,
            cuts,
            ends,
            ran: 0,
            progress,
        })
    }

    /// The next call of `p` on the selected slot; true while calls remain.
    /// Every call but the first follows an empty text event, which fails as a
    /// token's text does once the client is gone, so a request whose client
    /// left ends between its calls. The last call notes the engine's cuts at
    /// the messages, as the prompt run in one go does, and steps the prompt's
    /// last id ([`Gen::close`]).
    pub(crate) fn prompt_call(
        &mut self,
        slot: &mut Slot,
        p: &mut Prompt,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<bool, GenError> {
        if p.ran > 0 {
            sink(Event::Text("", &[], &self.tim))?;
        }
        let from = p.ran.checked_sub(1).map_or(p.kept.cache_n, |i| p.ends[i]);
        let to = p.ends[p.ran];
        let t = Instant::now();
        p.ran += 1;
        if p.ran < p.ends.len() {
            slot.prefill_range(&p.ids[from..to], from, &p.media, &p.marks)?;
            self.tim.prompt_ms += ms_since(t);
            self.tim.prompt_n = to.min(self.n) - p.kept.cache_n;
            self.tim.n_past = to;
            return Ok(true);
        }
        slot.note_split(p.kept.cache_n, to, std::mem::take(&mut p.cuts));
        self.close(
            slot,
            (&p.ids[from..], from, &p.media, &p.marks),
            p.kept,
            t,
            p.progress,
            sink,
        )?;
        Ok(false)
    }

    /// The prompt's end: its last call `call` (the ids, their first position,
    /// the prompt's feeds and the prompt call's marks) fed with the step of
    /// its last id ([`Slot::close_call`]), and the first token's logits read, its
    /// timings, its event and the first token in hand: the lead the call
    /// carried ([`Gen::take_lead`]) then the pick from that row, in hand as a
    /// pass's tokens are. Its engine time is what the timings hold of it
    /// (nothing for a prompt run in one go) and the time since `t`.
    fn close(
        &mut self,
        slot: &mut Slot,
        call: Call<'_>,
        kept: Kept,
        t: Instant,
        progress: bool,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<(), GenError> {
        let n = self.n;
        let greedy = slot.close_call(call, self.logits_out())?;
        slot.prompts[slot.cur] = n;
        self.tim = Timings {
            prompt_n: n - kept.cache_n,
            prompt_ms: self.tim.prompt_ms + ms_since(t),
            n_ctx: self.ctx_max,
            n_past: slot.held.len(),
            cache_n: kept.cache_n,
            cache_ms: kept.cache_ms,
            n_prompt: n,
            jump: (self.pick.walk.is_some() || self.pick.held.is_some()).then(|| Jump {
                arm: self.jump,
                lead: self.lead.len(),
                ..Jump::default()
            }),
            ..Timings::default()
        };
        self.t1 = Instant::now();
        if progress {
            sink(Event::Prompt(&self.tim))?;
        }
        if self.budget > 0 {
            let first = self.answer(greedy);
            if self.lead.is_empty() {
                self.tok = Some(first);
            } else {
                self.kept.clone_from(&self.lead);
                self.kept.push(first);
                self.tok = Some(self.kept[0]);
                self.taken = 1;
            }
        }
        Ok(())
    }

    /// Takes the tokens in hand, one at a time, until the generation stops or
    /// needs the engine.
    pub(crate) fn pump(
        &mut self,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
        tick: &mut dyn FnMut(&Timings),
    ) -> Result<Need, GenError> {
        if let Some((_, e)) = self.pick.refused.take() {
            return Err(e.into());
        }
        if let Some(span) = &self.span {
            // A run fed in calls of a quantum asks the sink between them, as
            // a prompt's calls do: a client that went away ends the
            // generation, not the run.
            if span.fed > 0 {
                sink(Event::Text("", &[], &self.tim))?;
            }
            return Ok(Need::Fill);
        }
        while let Some(tok) = self.tok.take() {
            if self.pick.bias.banned.contains(&tok) {
                return Err(GenError::Banned(tok));
            }
            self.generated.push(tok);
            if self.pick.sampler.is_some() {
                self.history.push(tok);
            }
            self.tim.predicted_n = self.generated.len();
            self.tim.predicted_ms = ms_since(self.t1);
            tick(&self.tim);
            if self.stops.contains(&tok) {
                // A forced call still held ends the span unnamed; a watched
                // one ends its reasoning as an unconstrained reply's span
                // does, the answer its own.
                if self.pick.held.as_ref().is_some_and(|w| !w.waiting()) {
                    return Err(GenError::Unclosed);
                }
                self.stop = StopKind::Eos;
                break;
            }
            let piece = self.dec.push(tok);
            let special = special_at(&*self.vocab, tok, piece.as_deref())?;
            if let Some((at, len)) = special {
                self.specials.push((self.scan.text().len() + at, len));
            }
            let before = self.scan.text().len();
            let pushed = match &piece {
                Some(piece) => self.scan.push(piece),
                None => Pushed {
                    send: String::new(),
                    stopped: None,
                },
            };
            // The think budget and the loop guard read the token's text
            // before it goes out, as far as a stop word keeps it, so a close
            // the held program refuses sends nothing past it, and a stop word
            // ends the generation before a loop does.
            let kept = self.scan.text().len().saturating_sub(before);
            let looped = self.spend(
                piece.as_deref().map(|p| &p[..kept]),
                special.filter(|&(at, len)| at + len <= kept),
                pushed.stopped.as_deref(),
            )? && pushed.stopped.is_none();
            // The token's text out. A request that asks for probabilities
            // sends an event a token, its value written first, the text
            // possibly empty; any other sends text alone, when there is some.
            if let Some(c) = self.pick.probs.as_mut() {
                let token = self.generated.len() - 1;
                c.take(tok, &pushed.send)
                    .map_err(|error| GenError::Logprobs { token, error })?;
            }
            if self.pick.probs.is_some() || !pushed.send.is_empty() {
                let specials = self.sending(&pushed.send);
                sink(Event::Text(&pushed.send, &specials, &self.tim))?;
            }
            if let Some(w) = pushed.stopped {
                self.stop = StopKind::Word;
                self.stopping_word = w;
                break;
            }
            if looped {
                self.stop = StopKind::Loop;
                break;
            }
            if self.generated.len() >= self.budget {
                break;
            }
            // While a close id is queued, every taken id is stepped into the
            // engine, whose context then carries the close text: a pass cannot
            // run (one starts only while the budget holds `rows`, below, and
            // keeps at most `rows`, so the budget spends at the earliest with
            // the pass's last kept token taken — none remains for the fast
            // path below to drop), and the context bound stops the forced ids
            // like any token.
            if self.forcing() {
                let (n, len) = (self.n, self.generated.len());
                if n + len > self.ctx_max {
                    self.truncated = true;
                    break;
                }
                return Ok(Need::Step(tok));
            }
            // The last pass evaluated this token already.
            if let Some(&t) = self.kept.get(self.taken) {
                self.taken += 1;
                self.tok = Some(t);
                continue;
            }
            if let Some(e) = self.refused_after.take() {
                return Err(e.into());
            }
            let (n, len) = (self.n, self.generated.len());
            // Feeding `tok` takes position n + len - 1; the cache holds ctx_max.
            if n + len > self.ctx_max {
                self.truncated = true;
                break;
            }
            // Where the step arm feeds `tok`, the jump arm's tail feeds
            // nothing: no row is read after the program's last pick.
            if let Some(next) = self.tail_id()? {
                self.tok = Some(next);
                continue;
            }
            if self.plan_span(tok) {
                return Ok(Need::Fill);
            }
            // A pass takes `rows` positions from there, and every token it
            // keeps spends the think budget: it starts only while the budget
            // holds them all, so no kept token spends past the cap. A forced
            // call's held program steps until it takes over; a watch's pick
            // reads the span's close inside a pass.
            if self.rows > 1
                && self.think_left() >= self.rows
                && n + len + self.rows - 1 <= self.ctx_max
                && self.pick.held.as_ref().is_none_or(Walk::waiting)
            {
                return Ok(if self.pick.reads_rows() {
                    Need::Sampled(tok)
                } else {
                    Need::Advance(tok)
                });
            }
            return Ok(Need::Step(tok));
        }
        Ok(Need::Done)
    }

    /// The engine's answer `g` to a [`Need::Step`].
    pub(crate) fn stepped(&mut self, g: u32) {
        self.tim.n_past = self.n + self.generated.len();
        if let Some(j) = self.tim.jump.as_mut() {
            j.next_calls += 1;
        }
        self.tok = Some(self.answer(g));
    }

    /// The next id of a jump-arm answer whose every id left is the
    /// program's: the walk's fixed id, or [`Gen::end`] once the program is
    /// spent. The tail starts where no pick is left ahead (the walk fixes
    /// every id to the program's end) and the vocabulary names its stop;
    /// `None` before it, on the step arm and for free text. Every id of the
    /// tail is taken where the step arm would step the one before it, with
    /// the same bounds, so the answer is the step arm's but for the stop id:
    /// the step arm picks it from a row, the tail names it. A fixed id the
    /// walk refuses is the request's error, as a pick's is.
    fn tail_id(&mut self) -> Result<Option<u32>, StepError> {
        let (true, Some(end), Some(walk)) = (self.jump, self.end, self.pick.walk.as_mut()) else {
            return Ok(None);
        };
        if !self.tail {
            // A walk with no id fixed ahead that is not spent stands at a
            // pick or inside a free string: it has no run to look along.
            if !walk.fixed_next() && !walk.spent() {
                return Ok(None);
            }
            let mut ahead = walk.ahead();
            if !ahead.forced_run(usize::MAX).is_ok_and(|_| ahead.spent()) {
                return Ok(None);
            }
            self.tail = true;
        }
        let id = if walk.fixed_next() {
            walk.choose(&[])?
        } else {
            end
        };
        if let Some(j) = self.tim.jump.as_mut() {
            j.tail += 1;
        }
        Ok(Some(id))
    }

    /// Whether the ids the program fixes after `tok`, just taken, are fed as
    /// a call: [`MIN_FILL`] of them at least, and no more than the ids the
    /// request may still generate and the positions the cache has left. The
    /// run's ids are then taken from the walk and held in [`Gen::span`]. When
    /// the step arm would step the run's last id too (the request may
    /// generate past the run and the cache holds its position), the span's
    /// last call steps it: one [`Engine::next_after`] whose row is the next
    /// pick's.
    fn plan_span(&mut self, tok: u32) -> bool {
        if !self.jump {
            return false;
        }
        let Some(walk) = self.pick.walk.as_mut() else {
            return false;
        };
        if !walk.fixed_next() {
            return false;
        }
        let (n, len) = (self.n, self.generated.len());
        // The run's last id is fed by the step that follows, which must find
        // its position: the call's ids take positions n + len - 1 on.
        let room = (self.budget - len).min(self.ctx_max + 1 - n - len);
        let mut ahead = walk.ahead();
        let Ok(run) = ahead.forced_run(room) else {
            return false;
        };
        if run.len() < MIN_FILL {
            return false;
        }
        *walk = ahead;
        let after = len + run.len();
        let close = after < self.budget && n + after <= self.ctx_max;
        let mut feed = Vec::with_capacity(run.len() + 1);
        feed.push(tok);
        feed.extend_from_slice(&run[..run.len() - usize::from(!close)]);
        if let Some(j) = self.tim.jump.as_mut() {
            j.spans += 1;
            j.span_ids += feed.len();
        }
        self.span = Some(Span {
            feed,
            fed: 0,
            run,
            close,
        });
        true
    }

    /// One call of the run being fed ([`Need::Fill`]): its ids, or those up
    /// to the next cut on the grid of `quantum` ([`quantum_end`], from the
    /// slot's position) when other slots are busy, fed as one
    /// [`Engine::prefill`] that books them, or as the [`Engine::next_after`]
    /// that steps the run's last id when it is the span's last call and the
    /// span closes ([`Gen::plan_span`]); the request's `n_past` is the slot's
    /// positions after each call. The call that ends the run puts the run in
    /// hand as a pass's tokens are: the loop takes them one at a time, then
    /// the pick from the closing row, or else the run's last id is the one
    /// the next [`Need::Step`] feeds.
    ///
    /// # Panics
    ///
    /// When no run is being fed: [`Gen::pump`] asks for this call only then.
    pub(crate) fn fill(
        &mut self,
        slot: &mut Slot,
        quantum: Option<NonZeroUsize>,
    ) -> Result<(), EngineError> {
        let span = self.span.as_mut().expect("a run being fed");
        let end = quantum.map_or(span.feed.len(), |q| {
            let at = slot.held.len();
            span.fed + quantum_end(at, at + span.feed.len() - span.fed, &[], q.get()) - at
        });
        let closing = span.close && end == span.feed.len();
        let ids = &span.feed[span.fed..end];
        let greedy = if closing {
            let logits = if self.pick.walk.as_mut().is_some_and(Walk::fixed_next) {
                None
            } else {
                out(&mut self.logits)
            };
            Some(slot.next_after(ids, logits)?)
        } else {
            slot.prefill_range(ids, slot.held.len(), &[], &[])?;
            None
        };
        span.fed = end;
        self.tim.n_past = slot.held.len();
        if let Some(j) = self.tim.jump.as_mut() {
            if closing {
                j.next_after_calls += 1;
            } else {
                j.prefill_calls += 1;
            }
        }
        if end < span.feed.len() {
            return Ok(());
        }
        let Some(span) = self.span.take() else {
            return Ok(());
        };
        self.kept = span.run;
        if let Some(g) = greedy {
            let picked = self.answer(g);
            self.kept.push(picked);
        }
        self.tok = Some(self.kept[0]);
        self.taken = 1;
        Ok(())
    }

    /// What a [`Need::Advance`] drafted; its tokens are in [`Gen::kept_mut`].
    /// The force is never armed here by [`Gen::pump`]'s pass gate: a pass
    /// starts only while the budget holds `rows` and keeps at most `rows`, so
    /// its first kept token is always there for the taking. A pass of several
    /// slots that held the pick ([`Gen::pass_row`]) gives it back here.
    pub(crate) fn advanced(&mut self, d: Drafted) {
        if std::mem::take(&mut self.lending) {
            self.take_back();
        }
        self.tim.book(d);
        book_depths(&mut self.depths, d);
        self.tim.n_past = self.n + self.generated.len() + self.kept.len() - 1;
        if let Some((at, e)) = self.pick.refused.take_if(|(at, _)| *at > 0) {
            self.kept.truncate(at);
            self.refused_after = Some(e);
        }
        self.tok = Some(self.kept[0]);
        self.taken = 1;
    }

    /// The request's pick lent to a pass ([`Gen::lend`]), its count of rows
    /// answered from zero.
    fn lend_out(&mut self) {
        self.pick.answered = 0;
        self.lend();
    }

    /// The request's pick into [`Gen::lent`], where every draw of the pass's
    /// sampler ([`Gen::draw`]) asks it, one draw a kept row.
    fn lend(&mut self) {
        std::mem::swap(&mut self.pick, &mut relock(&self.lent));
    }

    /// The pick back from [`Gen::lent`] once its pass is over, with what the
    /// pass's draws moved: the sampler's state, the program, the values held.
    fn take_back(&mut self) {
        self.lend();
        self.count_picks();
    }

    /// A [`Need::Sampled`] pass from `last` on the selected slot
    /// ([`Slot::advance_sampled`]): the request's pick goes to the engine as
    /// the pass's sampler and comes back ([`Gen::lend`]), so the next step or
    /// pass picks on from where this one stopped, and the engine is lent the
    /// history a step's draw is given ([`Gen::answer`]): the prompt's tail
    /// then the generated ids. Its tokens are then in hand as an
    /// [`Engine::advance`] pass's, the loop adding each to the generated ids
    /// as it takes it.
    pub(crate) fn advance_sampled(
        &mut self,
        slot: &mut Slot,
        last: u32,
    ) -> Result<(), EngineError> {
        let draw = self
            .draw
            .take()
            .expect("a request's pass sampler is back between passes");
        self.lend_out();
        let (d, draw) = slot.advance_sampled(
            last,
            &self.history,
            &mut self.lent_history,
            draw,
            self.best,
            &mut self.kept,
        );
        self.take_back();
        self.draw = Some(draw);
        self.advanced(d?);
        Ok(())
    }

    /// The row of a [`Need::Advance`] or a [`Need::Sampled`] pass from `last`
    /// in a round of several slots ([`Slot::advance_slots`]): the kept ids
    /// its `out`, and on a request that reads its rows its draw, the history
    /// and the sampler [`Gen::advance_sampled`] hands its pass, the pick lent
    /// until [`Gen::advanced`] takes the round's count.
    pub(crate) fn pass_row(&mut self, slot: usize, last: u32) -> SlotPass<'_> {
        let draw = if self.pick.reads_rows() {
            self.lend_out();
            self.lending = true;
            Some(SlotDraw {
                history: &self.history,
                sampler: &mut self.draw,
                best: self.best,
            })
        } else {
            None
        };
        SlotPass {
            slot,
            last,
            out: &mut self.kept,
            drafted: Drafted::default(),
            draw,
        }
    }

    /// The text still held, and the outcome. On a request that asks for
    /// probabilities, a generation that ended on an end-of-generation id
    /// writes that id's value here, with the held text as what it released,
    /// and sends it in an event of its own.
    pub(crate) fn finish(
        &mut self,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    ) -> Result<Outcome, GenError> {
        let mut rest = String::new();
        if self.stop != StopKind::Word {
            let tail = self.dec.flush();
            let pushed = self.scan.push(&tail);
            rest = pushed.send;
            if let Some(w) = pushed.stopped {
                self.stop = StopKind::Word;
                self.stopping_word = w;
            } else {
                rest.push_str(&self.scan.finish());
            }
        }
        // `pump` takes the end-of-generation id without its text out: the one
        // generated id that can still lack its value here.
        let valued = match self.pick.probs.as_mut() {
            Some(c) => match self.generated.get(c.taken()) {
                Some(&eog) => {
                    let token = c.taken();
                    c.take(eog, &rest)
                        .map_err(|error| GenError::Logprobs { token, error })?;
                    true
                }
                None => false,
            },
            None => false,
        };
        if valued || !rest.is_empty() {
            let specials = self.sending(&rest);
            sink(Event::Text(&rest, &specials, &self.tim))?;
        }
        self.tim.predicted_ms = ms_since(self.t1);
        let content = self.scan.text().to_owned();
        self.specials.retain(|&(at, _)| at < content.len());
        Ok(Outcome {
            content,
            specials: std::mem::take(&mut self.specials),
            tokens: std::mem::take(&mut self.generated),
            stop: self.stop,
            stopping_word: std::mem::take(&mut self.stopping_word),
            truncated: self.truncated,
            timings: self.tim.clone(),
        })
    }

    /// The whole generation on `slot` alone, its calls one at a time; a run
    /// of fixed ids is fed in calls cut on the quantum's grid, or whole.
    #[cfg(test)]
    #[allow(
        clippy::too_many_arguments,
        reason = "test driver: the arguments of the prompt and pump calls it chains"
    )]
    fn run(
        &mut self,
        slot: &mut Slot,
        held: &Held,
        media: &[MediaFeed],
        p: &GenParams,
        quantum: Option<NonZeroUsize>,
        sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
        tick: &mut dyn FnMut(&Timings),
    ) -> Result<Outcome, GenError> {
        self.prompt(slot, held, media, p, sink)?;
        loop {
            match self.pump(sink, tick)? {
                Need::Done => return self.finish(sink),
                Need::Step(t) => {
                    let g = slot.next(t, self.logits_out())?;
                    self.stepped(g);
                }
                Need::Advance(t) => {
                    let d = slot.advance(t, &mut self.kept)?;
                    self.advanced(d);
                }
                Need::Sampled(t) => self.advance_sampled(slot, t)?,
                Need::Fill => self.fill(slot, quantum)?,
            }
        }
    }
}

/// Runs one request on the slot alone ([`Gen`]); `timings_out` gets its
/// timings whether it ends or fails.
#[cfg(test)]
pub(crate) fn generate(
    slot: &mut Slot,
    factory: &SamplerFactory,
    ids: &[u32],
    p: &GenParams,
    sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    tick: &mut dyn FnMut(&Timings),
    timings_out: &mut Timings,
) -> Result<Outcome, GenError> {
    let (r, timings) = generate_in_quanta(slot, factory, ids, p, None, sink, tick);
    *timings_out = timings;
    r
}

/// [`generate`] with a run of fixed ids fed in calls cut on the quantum's
/// grid ([`quantum_end`]), as the worker feeds one beside other busy slots;
/// the timings come with the result.
#[cfg(test)]
pub(crate) fn generate_in_quanta(
    slot: &mut Slot,
    factory: &SamplerFactory,
    ids: &[u32],
    p: &GenParams,
    quantum: Option<NonZeroUsize>,
    sink: &mut dyn FnMut(Event<'_>) -> io::Result<()>,
    tick: &mut dyn FnMut(&Timings),
) -> (Result<Outcome, GenError>, Timings) {
    let sampler = sampler_of(factory, p).expect("the test's factory takes the request");
    let mut g = Gen::new(slot, sampler, ids.len(), p);
    let r = g.run(slot, &Held::from(ids.to_vec()), &[], p, quantum, sink, tick);
    (r, g.timings().clone())
}

#[cfg(test)]
mod tests {
    use super::{Depth, book_depths, partial_path, prompt_tail};
    use crate::engine::Drafted;
    use crate::media::{Held, ImageKey, MediaSpan};
    use crate::slotfile;
    use std::path::Path;

    /// A pass's draft counts land at their depths, a kept prefix's: three
    /// proposed and one kept, then two proposed and both kept, then a pass
    /// that proposed none, leave depths 1 to 3 at proposed 2, 2, 1 and kept
    /// 2, 1, 0.
    #[test]
    fn a_pass_books_its_draft_at_each_depth() {
        let mut depths = Vec::new();
        for (proposed, accepted) in [(3, 1), (2, 2), (0, 0)] {
            book_depths(&mut depths, Drafted { proposed, accepted });
        }
        let at = |proposed, accepted| Depth { proposed, accepted };
        assert_eq!(depths, vec![at(2, 2), at(2, 1), at(1, 0)]);
    }

    /// The penalties' prompt tail leaves an image span's positions out and
    /// keeps the last `n` of the rest, in order.
    #[test]
    fn prompt_tail_leaves_image_spans_out() {
        let span = |at, len| MediaSpan {
            at,
            len,
            key: ImageKey([0; 32]),
        };
        let held = Held {
            ids: vec![1, 2, 9, 9, 9, 3, 4, 9, 9, 5],
            media: vec![span(2, 3), span(7, 2)],
        };
        assert_eq!(prompt_tail(&held, 64), vec![1, 2, 3, 4, 5]);
        assert_eq!(prompt_tail(&held, 3), vec![3, 4, 5]);
        assert_eq!(prompt_tail(&held, 0), Vec::<u32>::new());
        let plain = Held::from(vec![7, 8, 9]);
        assert_eq!(prompt_tail(&plain, 2), vec![8, 9]);
    }

    /// Two saves to one path write two partial files, neither of which a
    /// slot file can be named.
    #[test]
    fn partial_paths_are_distinct_per_save() {
        let path = Path::new("/slots/a.bin");
        let (a, b) = (partial_path(path), partial_path(path));
        assert_ne!(a, b, "two saves share the partial file {}", a.display());
        for p in [&a, &b] {
            assert_eq!(p.parent(), path.parent());
            let name = p
                .file_name()
                .and_then(|n| n.to_str())
                .expect("a UTF-8 name");
            assert!(
                !slotfile::valid_filename(name),
                "{name} is a valid slot name"
            );
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use std::any::Any;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use std::sync::atomic::Ordering::SeqCst;

    use super::{GenParams, Outcome, Slot, Timings, generate};
    use crate::api::logprobs::Ask;
    use crate::engine::{
        CacheNote, Decoder, Engine, EngineError, SamplingParams, SaveInFlight, Saved, SharedPrefix,
        StateError, Tokenizer,
    };
    use crate::media::{Held, MediaFeed};
    use crate::mock::{MockEngine, MockTokenizer, SaveStats};
    use crate::reasoning::{ThinkEntry, ThinkTags};
    use crate::sampling;

    /// The mock vocabulary, whose `<｜User｜>` (id 2) opens a message.
    struct MarkTok(MockTokenizer);

    impl Tokenizer for MarkTok {
        fn encode(&self, text: &str) -> Vec<u32> {
            self.0.encode(text)
        }
        fn guard<'a>(&self, text: &'a str, mark: &str) -> std::borrow::Cow<'a, str> {
            self.0.guard(text, mark)
        }
        fn encode_marked(&self, text: &str, mark: &str) -> Vec<u32> {
            self.0.encode_marked(text, mark)
        }
        fn decode(&self, ids: &[u32]) -> String {
            self.0.decode(ids)
        }
        fn decoder(&self) -> Box<dyn Decoder> {
            self.0.decoder()
        }
        fn bos(&self) -> Option<u32> {
            self.0.bos()
        }
        fn eos(&self) -> u32 {
            self.0.eos()
        }
        fn add_bos(&self) -> bool {
            self.0.add_bos()
        }
        fn n_vocab(&self) -> usize {
            self.0.n_vocab()
        }
        fn user_start(&self) -> Vec<u32> {
            vec![2]
        }
    }

    /// What a [`Probe`] saw: its notes, the length of every prefill call,
    /// the marks every prompt call's part was handed, and the states it took
    /// and took back.
    #[derive(Default)]
    struct Log {
        notes: Vec<CacheNote>,
        calls: Vec<usize>,
        marks: Vec<Vec<usize>>,
        /// What each request told [`Engine::will_reply`].
        replies: Vec<Option<usize>>,
        snapshots: usize,
        resumes: usize,
    }

    /// The mock engine under a cut rule like V4.1's CED hole: a position
    /// strictly inside a prompt call is not kept, and a cut there falls to the
    /// call's start. It caches its states (`budget` bytes) and, with `split`,
    /// cuts a prompt call at every mark it is offered; with `marked`, it
    /// keeps the marks a call is handed as positions inside that one call.
    struct Probe {
        inner: MockEngine,
        tok: Arc<MarkTok>,
        ctx: Vec<u32>,
        calls: Vec<Range<usize>>,
        budget: u64,
        split: bool,
        /// Cut one position past each mark instead: a defective engine.
        off_mark: bool,
        /// Keep the marks [`Engine::prefill_marked`] hands it.
        marked: bool,
        /// Refuse every state it is handed back.
        refuse: bool,
        /// How long each snapshot sleeps.
        snapshot_takes: Duration,
        log: Arc<Mutex<Log>>,
    }

    fn rule(n: usize, calls: &[Range<usize>]) -> (usize, Option<String>) {
        let mut k = n;
        let mut why = None;
        while let Some(c) = calls.iter().find(|c| c.start < k && k < c.end) {
            why = Some(format!("position {k} lies inside prompt call {c:?}"));
            k = c.start;
        }
        (k, why)
    }

    struct ProbeSaved {
        ctx: Vec<u32>,
        calls: Vec<Range<usize>>,
    }

    impl Saved for ProbeSaved {
        fn n_tokens(&self) -> usize {
            self.ctx.len()
        }
        fn n_bytes(&self) -> u64 {
            4 * self.ctx.len() as u64
        }
        fn keepable(&self, n: usize) -> usize {
            rule(n.min(self.ctx.len()), &self.calls).0
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl Probe {
        fn new(budget: u64, split: bool, log: &Arc<Mutex<Log>>) -> Probe {
            Probe {
                inner: MockEngine::new(4096),
                tok: Arc::new(MarkTok(MockTokenizer)),
                ctx: Vec::new(),
                calls: Vec::new(),
                budget,
                split,
                off_mark: false,
                marked: false,
                refuse: false,
                snapshot_takes: Duration::ZERO,
                log: Arc::clone(log),
            }
        }

        fn slot(budget: u64, split: bool) -> (Slot, Arc<Mutex<Log>>) {
            let log = Arc::new(Mutex::new(Log::default()));
            (Slot::new(Box::new(Probe::new(budget, split, &log))), log)
        }
    }

    impl Engine for Probe {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.tok.clone()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)?;
            let at = self.ctx.len();
            self.calls.push(at..at + ids.len());
            self.ctx.extend_from_slice(ids);
            self.log.lock().expect("the log").calls.push(ids.len());
            Ok(())
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            let g = self.inner.next(last, out)?;
            self.ctx.push(last);
            Ok(g)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.ctx.clear();
            self.calls.clear();
            self.inner.reset()
        }
        fn will_reply(&mut self, tokens: Option<usize>) {
            self.log.lock().expect("the log").replies.push(tokens);
        }
        fn keepable(&self, n: usize) -> usize {
            rule(n.min(self.ctx.len()), &self.calls).0
        }
        fn keep_limit(&self, n: usize) -> Option<String> {
            rule(n.min(self.ctx.len()), &self.calls).1
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            if self.keepable(n) != n {
                return Err(EngineError(format!("probe: {n} is not a kept point")));
            }
            self.inner.cut(n)?;
            self.ctx.truncate(n);
            self.calls.retain_mut(|c| {
                c.end = c.end.min(n);
                c.start < c.end
            });
            Ok(())
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
        fn cache_ram(&self) -> u64 {
            self.budget
        }
        fn snapshots_async(&self) -> bool {
            true
        }

        fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
            std::thread::sleep(self.snapshot_takes);
            self.log.lock().expect("the log").snapshots += 1;
            Ok(Arc::new(ProbeSaved {
                ctx: self.ctx.clone(),
                calls: self.calls.clone(),
            }))
        }
        fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
            self.log.lock().expect("the log").resumes += 1;
            let s = state
                .as_any()
                .downcast_ref::<ProbeSaved>()
                .ok_or_else(|| StateError::Format("not a probe state".to_owned()))?;
            if self.refuse {
                return Err(StateError::Format("probe: refused".to_owned()));
            }
            self.inner.reset()?;
            self.inner.prefill(&s.ctx)?;
            self.ctx.clone_from(&s.ctx);
            self.calls.clone_from(&s.calls);
            Ok(())
        }
        fn prefill_splits(&self, _first: usize, _end: usize, marks: &[usize]) -> Vec<usize> {
            match (self.split, self.off_mark) {
                (true, false) => marks.to_vec(),
                (true, true) => marks.iter().map(|u| u + 1).collect(),
                (false, _) => Vec::new(),
            }
        }
        fn prefill_marked(
            &mut self,
            ids: &[u32],
            media: &[MediaFeed],
            marks: &[usize],
        ) -> Result<(), EngineError> {
            self.log.lock().expect("the log").marks.push(marks.to_vec());
            if !self.marked {
                return if media.is_empty() {
                    self.prefill(ids)
                } else {
                    self.prefill_media(ids, media)
                };
            }
            self.prefill(ids)?;
            let call = self.calls.pop().expect("the call just fed");
            assert!(
                marks.iter().all(|&u| call.start < u && u <= call.end),
                "probe: marks {marks:?} outside the call {call:?}"
            );
            let mut start = call.start;
            for &u in marks.iter().filter(|&&u| u < call.end) {
                self.calls.push(start..u);
                start = u;
            }
            self.calls.push(start..call.end);
            Ok(())
        }
        fn note(&self, note: &CacheNote) {
            self.log.lock().expect("the log").notes.push(note.clone());
        }
    }

    /// One greedy request of at most 8 tokens: what the cache kept, and
    /// what came out.
    fn run(slot: &mut Slot, ids: &[u32]) -> (usize, Vec<u32>) {
        let o = outcome(slot, ids);
        (o.timings.cache_n, o.tokens)
    }

    /// [`run`]'s request, its whole outcome.
    fn outcome(slot: &mut Slot, ids: &[u32]) -> Outcome {
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        let factory = sampling::reference_factory();
        let mut tim = Timings::default();
        generate(
            slot,
            &factory,
            ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation")
    }

    fn enc(text: &str) -> Vec<u32> {
        MockTokenizer.encode(text)
    }

    /// A second conversation between two turns of the first costs the first
    /// nothing: its state goes to the cache and comes back, and the turn
    /// keeps all it held and generates what a fresh prefill does.
    #[test]
    fn an_interleaved_conversation_keeps_the_first_ones_prefix() {
        let (mut slot, log) = Probe::slot(1 << 20, false);
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (_, t1) = run(&mut slot, &a1);
        run(&mut slot, &enc("<｜User｜>xyz uvw xyz "));
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(
            kept,
            a1.len() + t1.len() - 1,
            "{:?}",
            log.lock().expect("log").notes
        );
        let (fresh, _) = Probe::slot(0, false);
        let mut fresh = fresh;
        assert_eq!(run(&mut fresh, &a2).1, t2, "the restored turn's ids");
        let notes = &log.lock().expect("log").notes;
        assert!(
            notes
                .iter()
                .any(|n| matches!(n, CacheNote::Load { positions, .. }
                if *positions == a1.len() + t1.len() - 1)),
            "{notes:?}"
        );
    }

    /// A prefix the engine keeps less of than the request shares is noted,
    /// with the engine's reason.
    #[test]
    fn a_prefix_the_engine_drops_is_noted_with_its_reason() {
        let (mut slot, log) = Probe::slot(0, false);
        let a = enc("<｜User｜>the cat sat on the mat and the ");
        run(&mut slot, &a);
        let mut b = a[..20].to_vec();
        b.extend(enc("dog ran"));
        let (kept, _) = run(&mut slot, &b);
        assert_eq!(kept, 0);
        let notes = &log.lock().expect("log").notes;
        assert!(
            notes.iter().any(
                |n| matches!(n, CacheNote::Reuse { common: 20, ask: 20, kept: 0,
                reason: Some(r), .. } if r.contains("inside prompt call"))
            ),
            "{notes:?}"
        );
    }

    /// A prompt call is cut at its first and last message starts where the
    /// engine asks, and the cut position is then kept by a request that
    /// shares the prompt up to it — with the same ids a fresh prefill gives.
    #[test]
    fn a_prompt_call_is_cut_at_its_messages() {
        let (mut slot, log) = Probe::slot(0, true);
        let system = enc("be brief and kind. ");
        let mut a = system.clone();
        a.extend(enc(
            "<｜User｜>the cat sat<｜Assistant｜>on the mat<｜User｜>and the ",
        ));
        run(&mut slot, &a);
        let u1 = system.len();
        let u2 = a.len() - enc("<｜User｜>and the ").len();
        assert_eq!(
            log.lock().expect("log").calls[..3],
            [u1, u2 - u1, a.len() - 1 - u2],
            "the prompt call's pieces"
        );
        assert_eq!(
            log.lock().expect("log").marks[..3],
            [vec![u1], vec![u2], vec![]],
            "each piece is handed the marks in it: its end"
        );
        let mut b = system;
        b.extend(enc("<｜User｜>a dog ran and the "));
        let (kept, tb) = run(&mut slot, &b);
        assert_eq!(kept, u1, "the system prompt, where the second call starts");
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &b).1, tb);
    }

    /// An engine that cuts nowhere and keeps the marks it is handed gets the
    /// prompt call as one call with its first and last message starts, and a
    /// request that shares the prompt up to the first keeps it — with the
    /// same ids a fresh prefill gives.
    #[test]
    fn a_marked_engine_takes_one_call_with_its_marks() {
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            marked: true,
            ..Probe::new(0, false, &log)
        };
        let mut slot = Slot::new(Box::new(probe));
        let system = enc("be brief and kind. ");
        let mut a = system.clone();
        a.extend(enc(
            "<｜User｜>the cat sat<｜Assistant｜>on the mat<｜User｜>and the ",
        ));
        run(&mut slot, &a);
        let u1 = system.len();
        let u2 = a.len() - enc("<｜User｜>and the ").len();
        {
            let log = log.lock().expect("log");
            assert_eq!(log.calls[..1], [a.len() - 1], "one prompt call");
            assert_eq!(log.marks[0], [u1, u2], "the call's marks");
        }
        let mut b = system;
        b.extend(enc("<｜User｜>a dog ran and the "));
        let (kept, tb) = run(&mut slot, &b);
        assert_eq!(kept, u1, "the system prompt, a mark inside the first call");
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &b).1, tb);
    }

    /// An engine that cuts a prompt call anywhere but at the marks it was
    /// offered fails the request by name; nothing is fed.
    #[test]
    fn a_cut_off_the_marks_is_a_named_error() {
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            off_mark: true,
            ..Probe::new(0, true, &log)
        };
        let mut slot = Slot::new(Box::new(probe));
        let a = enc("be brief and kind. <｜User｜>the cat sat on the mat and the ");
        let p = GenParams {
            n_predict: 4,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        let mut tim = Timings::default();
        let r = generate(
            &mut slot,
            &sampling::reference_factory(),
            &a,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        );
        let e = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("not among its marks"), "{e}");
        assert!(log.lock().expect("log").calls.is_empty());
    }

    /// A full cache: saving the slot's state before the pick is taken back
    /// evicts the oldest state, and the pick is still the one taken back.
    #[test]
    fn the_pick_survives_the_save_that_makes_room() {
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let b = enc("<｜User｜>xyz uvw xyz ");
        let c = enc("<｜User｜>abc def abc def abc ");
        let (mut probe_slot, _) = Probe::slot(1 << 20, false);
        let (_, t1) = run(&mut probe_slot, &a1);
        let held_a = a1.len() + t1.len() - 1;
        let (_, tb) = run(&mut probe_slot, &b);
        let (_, tc) = run(&mut probe_slot, &c);
        let held_c = c.len() + tc.len() - 1;
        let held_b = b.len() + tb.len() - 1;
        assert!(held_b <= held_a, "the fixture needs b no longer than a");
        // Room for a's state and c's, not for b's beside them.
        let budget = 4 * (held_a + held_c) as u64;
        let (mut slot, log) = Probe::slot(budget, false);
        run(&mut slot, &a1);
        run(&mut slot, &b);
        run(&mut slot, &c);
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(kept, held_a, "{:?}", log.lock().expect("log").notes);
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &a2).1, t2);
    }

    /// A state the engine refuses to take back leaves the cache, is noted,
    /// and the request runs from a reset with a fresh prefill's ids.
    #[test]
    fn a_refused_state_leaves_the_cache() {
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            refuse: true,
            ..Probe::new(1 << 20, false, &log)
        };
        let mut slot = Slot::new(Box::new(probe));
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (_, t1) = run(&mut slot, &a1);
        run(&mut slot, &enc("<｜User｜>xyz uvw xyz "));
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (kept, t2) = run(&mut slot, &a2);
        assert_eq!(kept, 0);
        let (mut fresh, _) = Probe::slot(0, false);
        assert_eq!(run(&mut fresh, &a2).1, t2);
        let notes = log.lock().expect("log").notes.clone();
        let refused = |n: &CacheNote| matches!(n, CacheNote::Skip { why, .. } if why.contains("refused to take the state back"));
        assert_eq!(notes.iter().filter(|n| refused(n)).count(), 1, "{notes:?}");
        assert!(
            !slot.cache.covers(&Held::from(a1.clone())),
            "the refused state is still cached"
        );
    }

    /// Each request tells the engine its reply before the cache is reused:
    /// a greedy one the tokens after its first (the passes make at most
    /// those), one that takes no pass — it bans the end of generation —
    /// none.
    #[test]
    fn a_request_tells_the_engine_its_reply_before_the_reuse() {
        let (mut slot, log) = Probe::slot(0, false);
        let ids = enc("<｜User｜>the cat sat on the mat and the ");
        run(&mut slot, &ids);
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: true,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        let factory = sampling::reference_factory();
        let mut tim = Timings::default();
        generate(
            &mut slot,
            &factory,
            &ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        assert_eq!(log.lock().expect("log").replies, [Some(7), Some(0)]);
    }

    /// A greedy request that asks for probabilities takes no pass (each of
    /// its rows is a step's), so it tells the engine none of its reply comes
    /// from one.
    #[test]
    fn an_asking_request_tells_the_engine_it_takes_no_pass() {
        let (mut slot, log) = Probe::slot(0, false);
        let p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: Some(Ask::new(2, MockTokenizer.n_vocab())),
            constraint: None,
            logit_bias: Vec::new(),
        };
        let mut tim = Timings::default();
        generate(
            &mut slot,
            &sampling::reference_factory(),
            &enc("<｜User｜>the cat sat on the mat and the "),
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        assert_eq!(log.lock().expect("log").replies, [Some(0)]);
    }

    /// One request ([`run`]) and what it cost the engine in states: the
    /// snapshots taken and the states taken back.
    fn costed(slot: &mut Slot, log: &Mutex<Log>, ids: &[u32]) -> ((usize, usize), usize, Vec<u32>) {
        let count = || {
            let l = log.lock().expect("log");
            (l.snapshots, l.resumes)
        };
        let before = count();
        let (kept, tokens) = run(slot, ids);
        let after = count();
        ((after.0 - before.0, after.1 - before.1), kept, tokens)
    }

    /// The slot's state goes into the cache when another conversation takes
    /// the slot or a cached state replaces it, and never for a request that
    /// carries the slot's last prompt whole. Each row's (snapshots, resumes):
    /// (i) a first turn, (ii) its next turn, which keeps all of it, (iii)
    /// another conversation, (iv) the first one back from the cache, (v-a) a
    /// short turn of a third, (v-b) its next turn with the reply rendered
    /// again — the reply's first id not the slot's — which keeps less than
    /// half the slot, where llama-server's half rule saves.
    #[test]
    fn a_switch_pays_one_snapshot_and_a_continuation_none() {
        let (mut slot, log) = Probe::slot(1 << 20, false);
        let a1 = enc("<｜User｜>the cat sat on the mat and the ");
        let (i, _, t1) = costed(&mut slot, &log, &a1);
        let mut a2 = a1.clone();
        a2.extend(&t1);
        a2.extend(enc("<｜User｜>and then the "));
        let (ii, _, t2) = costed(&mut slot, &log, &a2);
        let (iii, _, _) = costed(&mut slot, &log, &enc("<｜User｜>xyz uvw xyz "));
        let mut a3 = a2.clone();
        a3.extend(&t2);
        a3.extend(enc("<｜User｜>and so the "));
        let (iv, _, _) = costed(&mut slot, &log, &a3);
        let c1 = enc("aba");
        let (va, _, tc1) = costed(&mut slot, &log, &c1);
        let mut c2 = c1.clone();
        c2.extend(enc("<think></think>ba<｜User｜>ab"));
        assert_ne!(
            Some(&c2[c1.len()]),
            tc1.first(),
            "the fixture: (v-b) parts from the reply at its first id"
        );
        let (vb, kept, _) = costed(&mut slot, &log, &c2);
        let held = c1.len() + tc1.len() - 1;
        assert!(
            2 * kept < held,
            "the fixture: (v-b) keeps {kept} of the {held} positions the slot held, at least half"
        );
        assert_eq!(
            [i, ii, iii, iv, va, vb],
            [(0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (0, 0)],
            "{:?}",
            log.lock().expect("log").notes
        );
    }

    /// The prompt cache's work is the request's `cache_ms`, before its
    /// `prompt_ms`: a request that saves the slot's state reports at least
    /// the time the probe's snapshot sleeps, in its `timings` too.
    #[test]
    fn a_save_is_timed_in_cache_ms() {
        const TAKES: Duration = Duration::from_millis(20);
        let log = Arc::new(Mutex::new(Log::default()));
        let probe = Probe {
            snapshot_takes: TAKES,
            ..Probe::new(1 << 20, false, &log)
        };
        let mut slot = Slot::new(Box::new(probe));
        outcome(&mut slot, &enc("<｜User｜>the cat sat on the mat and the "));
        let t = outcome(&mut slot, &enc("xyz uvw xyz ")).timings;
        assert_eq!(log.lock().expect("log").snapshots, 1, "the second saves");
        assert!(
            t.cache_ms >= TAKES.as_secs_f64() * 1e3,
            "cache_ms {} under the snapshot's {TAKES:?}",
            t.cache_ms
        );
        assert_eq!(t.to_json()["cache_ms"].as_f64(), Some(t.cache_ms));
    }

    /// The mock under [`MockEngine::with_async_snapshots`] in a slot of
    /// `slots` slots, its states copied in `copy` (their room taken from a
    /// pool of `cells` cells of [`CELL`] bytes when given), and a probe of
    /// the notes the server printed.
    fn async_slot(
        slots: usize,
        copy: Duration,
        cells: Option<usize>,
    ) -> (Slot, Arc<SaveStats>, Arc<Mutex<Vec<CacheNote>>>) {
        let (mock, stats) = MockEngine::new(4096)
            .with_slots(slots)
            .with_cache_ram(1 << 20)
            .with_async_snapshots(copy);
        let mock = match cells {
            Some(cells) => mock.with_staging(CELL, cells),
            None => mock,
        };
        let notes = Arc::new(Mutex::new(Vec::new()));
        let engine = Noted {
            inner: mock,
            notes: Arc::clone(&notes),
        };
        (Slot::with_slots(Box::new(engine), slots), stats, notes)
    }

    /// The mock, its notes kept.
    struct Noted {
        inner: MockEngine,
        notes: Arc<Mutex<Vec<CacheNote>>>,
    }

    impl Engine for Noted {
        fn snapshot_size(&self) -> Result<Option<u64>, StateError> {
            self.inner.snapshot_size()
        }
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.inner.next(last, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.inner.keepable(n)
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            self.inner.cut(n)
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
        fn cache_ram(&self) -> u64 {
            self.inner.cache_ram()
        }
        fn slots(&self) -> usize {
            self.inner.slots()
        }
        fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
            self.inner.select_slot(slot)
        }
        fn snapshot(&self) -> Result<Arc<dyn Saved>, StateError> {
            self.inner.snapshot()
        }
        fn snapshots_async(&self) -> bool {
            self.inner.snapshots_async()
        }
        fn snapshot_begin(&self) -> Result<crate::engine::Pending, StateError> {
            self.inner.snapshot_begin()
        }
        fn resume(&mut self, state: &Arc<dyn Saved>) -> Result<(), StateError> {
            self.inner.resume(state)
        }
        fn resume_prefix(&mut self, state: &Arc<dyn Saved>, n: usize) -> Result<usize, StateError> {
            self.inner.resume_prefix(state, n)
        }
        fn share_keepable(&self, from: usize, n: usize) -> usize {
            self.inner.share_keepable(from, n)
        }
        fn share_prefix(&mut self, from: usize, n: usize) -> Result<SharedPrefix, StateError> {
            self.inner.share_prefix(from, n)
        }
        fn note(&self, note: &CacheNote) {
            self.notes.lock().expect("the notes").push(note.clone());
        }
    }

    /// One request on `slot` of `s`, then its end as the worker books it.
    fn served(s: &mut Slot, slot: usize, ids: &[u32]) -> Outcome {
        s.select(slot).expect("a select");
        let o = outcome(s, ids);
        s.seal(slot).expect("a seal");
        o
    }

    const COPY: Duration = Duration::from_millis(60);

    /// A request's end begins the copy of its slot's state; the next
    /// conversation on that slot, arriving after the copy landed, cuts the
    /// slot with no snapshot on its path: none taken, its `cache_ms` well
    /// under the copy's time and no wait noted.
    #[test]
    fn a_conversation_switch_pays_no_snapshot_on_the_request_path() {
        let (mut slot, stats, notes) = async_slot(1, COPY, None);
        served(
            &mut slot,
            0,
            &enc("<｜User｜>the cat sat on the mat and the "),
        );
        std::thread::sleep(COPY * 2);
        let t = served(&mut slot, 0, &enc("xyz uvw xyz ")).timings;
        assert_eq!(
            stats.sync_snapshots.load(SeqCst),
            0,
            "the request path took a snapshot"
        );
        assert_eq!(stats.begun.load(SeqCst), 2, "a copy begun at each end");
        assert!(
            t.cache_ms < COPY.as_secs_f64() * 1e3 / 4.0,
            "cache_ms {} of a request that waits for nothing",
            t.cache_ms
        );
        let n = notes.lock().expect("notes");
        let waited: Vec<f64> = n
            .iter()
            .filter_map(|n| match n {
                CacheNote::Landed { waited_ms, .. } => Some(*waited_ms),
                _ => None,
            })
            .collect();
        assert_eq!(waited, [0.0], "the first copy landed unwaited: {n:?}");
    }

    /// A request that arrives while its slot's copy still runs waits for it
    /// before the slot's rows are cut: the wait is in `cache_ms` and in the
    /// `Landed` note, and no call ever rewrote rows the copy read.
    #[test]
    fn a_request_waits_for_its_slots_copy_before_it_cuts_the_rows() {
        let (mut slot, stats, notes) = async_slot(1, COPY, None);
        served(
            &mut slot,
            0,
            &enc("<｜User｜>the cat sat on the mat and the "),
        );
        let t = served(&mut slot, 0, &enc("xyz uvw xyz ")).timings;
        assert_eq!(stats.hazards.load(SeqCst), 0, "a cut under a copy");
        assert_eq!(stats.sync_snapshots.load(SeqCst), 0);
        assert!(
            t.cache_ms > COPY.as_secs_f64() * 1e3 / 4.0,
            "cache_ms {} of a request that had a copy to wait for",
            t.cache_ms
        );
        let n = notes.lock().expect("notes");
        assert!(
            n.iter()
                .any(|n| matches!(n, CacheNote::Landed { waited_ms, .. } if *waited_ms > 0.0)),
            "no wait noted: {n:?}"
        );
    }

    /// A cut, a reset or a resume of rows a copy still reads is
    /// [`SaveInFlight`], by name, and nothing is rewritten.
    #[test]
    fn a_rewrite_under_a_running_copy_is_save_in_flight() {
        let (mut slot, stats, _) = async_slot(1, COPY, None);
        served(
            &mut slot,
            0,
            &enc("<｜User｜>the cat sat on the mat and the "),
        );
        let err = slot.rows().err().expect("refused while the copy runs");
        assert_eq!(err.0, SaveInFlight { slot: 0 }.to_string());
        assert_eq!(stats.hazards.load(SeqCst), 0);
        slot.settle(0).expect("the wait");
        assert!(slot.rows().is_ok(), "once the copy has landed");
    }

    /// A request that shares a prefix with what another slot holds takes it
    /// from that slot ([`Engine::share_prefix`]) instead of evaluating it:
    /// its `cache_n` is the prefix, one share is noted, and its tokens are a
    /// fresh slot's; the other slot keeps its conversation, whose next turn
    /// keeps all it held; a cached state that keeps more wins over the share.
    #[test]
    fn a_request_takes_the_prefix_another_slot_holds() {
        let notes = Arc::new(Mutex::new(Vec::new()));
        let engine = Noted {
            inner: MockEngine::new(4096)
                .with_slots(2)
                .with_cache_ram(1 << 20)
                .with_shares(),
            notes: Arc::clone(&notes),
        };
        let mut slot = Slot::with_slots(Box::new(engine), 2);
        let system = "<｜User｜>the cat sat on the mat and the dog sat on the log ";
        let a = enc(&format!("{system}and the "));
        let b = enc(&format!("{system}so a "));
        let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
        slot.select(0).expect("a select");
        let ra = outcome(&mut slot, &a);
        slot.select(1).expect("a select");
        let rb = outcome(&mut slot, &b);
        assert_eq!(rb.timings.cache_n, common, "the shared prefix kept");
        let shares: Vec<CacheNote> = (notes.lock().expect("notes").iter())
            .filter(|n| matches!(n, CacheNote::Share { .. }))
            .cloned()
            .collect();
        assert!(
            matches!(
                shares[..],
                [CacheNote::Share { from: 0, kept, slot_kept: 0, .. }] if kept == common
            ),
            "{shares:?}"
        );
        let mut fresh = Slot::new(Box::new(MockEngine::new(4096)));
        assert_eq!(
            rb.tokens,
            outcome(&mut fresh, &b).tokens,
            "a fresh slot's reply"
        );
        let mut a2 = a.clone();
        a2.extend(&ra.tokens);
        a2.extend(enc("<｜User｜>then "));
        slot.select(0).expect("a select");
        let ra2 = outcome(&mut slot, &a2);
        assert_eq!(
            ra2.timings.cache_n,
            a.len() + ra.tokens.len() - 1,
            "slot 0 keeps its turn"
        );
        slot.select(1).expect("a select");
        outcome(&mut slot, &enc("<｜User｜>xyz uvw xyz "));
        let mut b2 = b.clone();
        b2.extend(&rb.tokens);
        b2.extend(enc("<｜User｜>so "));
        let cached = outcome(&mut slot, &b2).timings.cache_n;
        assert_eq!(
            cached,
            b.len() + rb.tokens.len() - 1,
            "the cached state of slot 1's turn keeps more than slot 0's prefix"
        );
        let n = notes.lock().expect("notes");
        assert_eq!(
            n.iter()
                .filter(|n| matches!(n, CacheNote::Share { .. }))
                .count(),
            1,
            "{n:?}"
        );
        assert!(
            n.iter().any(|n| matches!(n, CacheNote::Load { .. })),
            "{n:?}"
        );
        drop(n);
        // Slot 1 moves to another prompt, its turn cached: that state keeps
        // the system prompt of slot 0's next turn, less than slot 0 holds.
        outcome(&mut slot, &enc("<｜User｜>one two three "));
        let mut a3 = a2.clone();
        a3.extend(&ra2.tokens);
        a3.extend(enc("<｜User｜>and "));
        let took = outcome(&mut slot, &a3).timings.cache_n;
        assert_eq!(
            took,
            a2.len() + ra2.tokens.len() - 1,
            "the share keeps more than the cached state"
        );
        let n = notes.lock().expect("notes");
        let last = n
            .iter()
            .rev()
            .find(|n| matches!(n, CacheNote::Share { .. } | CacheNote::Load { .. }));
        assert!(
            matches!(last, Some(CacheNote::Share { from: 0, kept, .. }) if *kept == took),
            "{n:?}"
        );
    }

    /// A pooled mock's cell: a state of the short prompts below fills one,
    /// and [`LONG`]'s two.
    const CELL: u64 = 512;

    const SHORT: [&str; 3] = [
        "<｜User｜>the cat sat on the mat and the ",
        "<｜User｜>xyz uvw xyz uvw xyz ",
        "<｜User｜>one two three four five six ",
    ];

    const LONG: &str = "<｜User｜>the quick brown fox jumps over the lazy dog and the quick brown \
        fox jumps over the lazy dog again and again until the dog gets up and chases the fox over \
        the hill and ";

    /// The save notes the server printed, in order: `(copied, off_path,
    /// fallback)`.
    fn saves(notes: &[CacheNote]) -> Vec<(bool, bool, Option<&'static str>)> {
        notes
            .iter()
            .filter_map(|n| match n {
                CacheNote::Save {
                    copied,
                    off_path,
                    fallback,
                    ..
                } => Some((*copied, *off_path, *fallback)),
                _ => None,
            })
            .collect()
    }

    /// Why each state the server evicted left, in order.
    fn evicted(notes: &[CacheNote]) -> Vec<&'static str> {
        notes
            .iter()
            .filter_map(|n| match n {
                CacheNote::Evict { why, .. } => Some(*why),
                _ => None,
            })
            .collect()
    }

    const NO_ROOM: &str = "the engine's staging pool had no room";

    /// A pool with no room for a state's copy gives up its oldest landed
    /// state, and no other once the copy fits: the copy is begun, no
    /// snapshot is taken on the request path, and the save names no
    /// fallback.
    #[test]
    fn a_full_pool_gives_up_its_oldest_landed_state_and_the_copy_is_begun() {
        let (mut slot, stats, notes) = async_slot(1, Duration::ZERO, Some(2));
        for p in SHORT {
            served(&mut slot, 0, &enc(p));
        }
        assert_eq!(
            stats.no_room.load(SeqCst),
            1,
            "the third copy found no room"
        );
        assert_eq!(stats.begun.load(SeqCst), 3, "a copy begun at each end");
        assert_eq!(
            stats.sync_snapshots.load(SeqCst),
            0,
            "the request path took a snapshot"
        );
        let n = notes.lock().expect("notes");
        assert_eq!(saves(&n), [(true, true, None); 3], "{n:?}");
        assert_eq!(evicted(&n), [NO_ROOM], "{n:?}");
    }

    /// When the room the oldest state gave back lies apart from the rest
    /// that is free, the next oldest leaves too, until the copy fits: it is
    /// begun, and no snapshot is taken on the request path. (The long state
    /// needs two cells next to each other; one of four is free, then two
    /// apart once the oldest left, then three, two of them together.)
    #[test]
    fn a_fragmented_pool_gives_up_landed_states_until_the_copy_fits() {
        let (mut slot, stats, notes) = async_slot(1, Duration::ZERO, Some(4));
        for p in SHORT {
            served(&mut slot, 0, &enc(p));
        }
        served(&mut slot, 0, &enc(LONG));
        assert_eq!(stats.no_room.load(SeqCst), 2, "full, then in pieces");
        assert_eq!(
            stats.begun.load(SeqCst),
            4,
            "the long state's copy was begun"
        );
        assert_eq!(
            stats.sync_snapshots.load(SeqCst),
            0,
            "the request path took a snapshot"
        );
        let n = notes.lock().expect("notes");
        assert_eq!(saves(&n), [(true, true, None); 4], "{n:?}");
        assert_eq!(
            evicted(&n),
            [NO_ROOM; 2],
            "two states left for the room: {n:?}"
        );
    }

    /// When no landed state is left to give up and the pool's free room is
    /// still in pieces, around a state whose copy still runs, the state is
    /// snapshotted on the request path and the `Save` names the pool
    /// `fragmented`. (Of three cells, slot 0's landed state holds the first
    /// and slot 1's running copy the second; the long state needs two next
    /// to each other.)
    #[test]
    fn a_pool_still_without_room_after_the_eviction_falls_back_named() {
        let (mut slot, stats, notes) = async_slot(2, COPY * 4, Some(3));
        served(&mut slot, 0, &enc(SHORT[0]));
        slot.settle(0).expect("the first copy lands");
        served(&mut slot, 1, &enc(SHORT[1]));
        served(&mut slot, 0, &enc(LONG));
        assert_eq!(stats.no_room.load(SeqCst), 2, "full, then in pieces");
        assert_eq!(
            stats.sync_snapshots.load(SeqCst),
            1,
            "the long state's snapshot, on the request path"
        );
        assert_eq!(stats.hazards.load(SeqCst), 0, "a cut under a copy");
        let n = notes.lock().expect("notes");
        assert_eq!(
            saves(&n),
            [
                (true, true, None),
                (true, true, None),
                (true, false, Some("fragmented")),
            ],
            "{n:?}"
        );
        assert_eq!(
            evicted(&n),
            [NO_ROOM],
            "only the landed state left, never the running copy's: {n:?}"
        );
    }

    /// A state that passes the whole pool is snapshotted on the request path
    /// as `full` at once: no state leaves for room that cannot be made.
    #[test]
    fn a_state_over_the_whole_pool_evicts_nothing() {
        let (mut slot, stats, notes) = async_slot(1, Duration::ZERO, Some(1));
        served(&mut slot, 0, &enc(SHORT[0]));
        served(&mut slot, 0, &enc(LONG));
        assert_eq!(stats.sync_snapshots.load(SeqCst), 1);
        let n = notes.lock().expect("notes");
        assert_eq!(
            saves(&n),
            [(true, true, None), (true, false, Some("full"))],
            "{n:?}"
        );
        assert!(evicted(&n).is_empty(), "a state left for nothing: {n:?}");
    }

    #[test]
    fn a_sealed_sparse_state_is_not_copied_again_on_a_switch() {
        let log = Arc::new(Mutex::new(Log::default()));
        let mut slot = Slot::new(Box::new(Probe::new(1 << 20, false, &log)));
        outcome(&mut slot, &[10; 20]);
        slot.seal(0).expect("a sealed state");
        outcome(&mut slot, &[11; 20]);
        assert_eq!(log.lock().expect("the log").snapshots, 1);
    }

    #[test]
    fn an_over_budget_state_is_never_copied_for_seal_or_switch() {
        let (mock, stats) = MockEngine::new(4096)
            .with_cache_ram(CELL)
            .with_async_snapshots(Duration::ZERO);
        let notes = Arc::new(Mutex::new(Vec::new()));
        let mut slot = Slot::new(Box::new(Noted {
            inner: mock.with_staging(CELL, 1),
            notes: Arc::clone(&notes),
        }));
        served(&mut slot, 0, &enc(LONG));
        served(&mut slot, 0, &enc(SHORT[0]));
        assert_eq!(
            stats.sync_snapshots.load(SeqCst),
            0,
            "a switch or fallback copied an oversized state"
        );
        assert_eq!(
            stats.no_room.load(SeqCst),
            0,
            "an oversized state reached staging"
        );
        assert_eq!(
            stats.begun.load(SeqCst),
            1,
            "only the small state is copied"
        );
        assert!(
            notes
                .lock()
                .expect("notes")
                .iter()
                .any(|n| matches!(n, CacheNote::Skip { why, .. } if why.contains("budget")))
        );
    }

    #[test]
    fn snapshot_admission_uses_array_bytes_at_the_exact_budget() {
        for fits in [false, true] {
            let (mut slot, stats, _) = async_slot(1, Duration::ZERO, Some(2));
            outcome(&mut slot, &enc(SHORT[0]));
            let bytes = slot
                .engine
                .snapshot_size()
                .expect("a size")
                .expect("known bytes");
            slot.cache = crate::promptcache::PromptCache::new(bytes - u64::from(!fits));
            slot.seal(0).expect("a seal");
            assert_eq!(stats.begun.load(SeqCst), usize::from(fits));
            assert_eq!(stats.sync_snapshots.load(SeqCst), 0);
        }
    }

    /// A cache hit puts back the prefix the request keeps and nothing past
    /// it: the positions the engine took back are the kept ones, not the
    /// state's, and no state goes back whole.
    #[test]
    fn a_cache_hit_puts_back_only_the_prefix_it_keeps() {
        let (mut slot, stats, notes) = async_slot(1, COPY, None);
        let a1 = enc("<｜User｜>the cat sat on the mat and the dog sat on the log and the ");
        let o1 = served(&mut slot, 0, &a1);
        served(&mut slot, 0, &enc("xyz uvw xyz uvw xyz "));
        // A request that shares only the first 20 ids of the first.
        let mut a3 = a1[..20].to_vec();
        a3.extend(enc("qq rr ss "));
        let o3 = served(&mut slot, 0, &a3);
        assert_eq!(o3.timings.cache_n, 20, "{:?}", notes.lock().expect("notes"));
        assert_eq!(stats.resumed.load(SeqCst), 20, "positions put back");
        assert_eq!(stats.whole_resumes.load(SeqCst), 0);
        let held = a1.len() + o1.tokens.len() - 1;
        assert!(
            held > 20,
            "the fixture: the state holds more than the prefix"
        );
        let n = notes.lock().expect("notes");
        let load = n.iter().find_map(|n| match n {
            CacheNote::Load {
                positions, resumed, ..
            } => Some((*positions, *resumed)),
            _ => None,
        });
        assert_eq!(load, Some((held, 20)), "{n:?}");
    }
}

/// A prompt that carries an image, through [`Gen`] on the mock that takes
/// images: what the cache keeps of it and what the engine is fed. The mock's
/// context holds a span as its key's letters and the prompt `q\n` image `\n`
/// generates from the span, so a reply names the image the cache holds.
#[cfg(test)]
mod media_tests {
    use std::sync::{Arc, Mutex};

    use vision::{GridPlan, Patches};

    use super::{Gen, GenParams, Slot, quantum_ends, sampler_of};
    use crate::engine::{Engine, EngineError, SamplingParams, Tokenizer};
    use crate::media::{Held, ImageKey, MediaFeed, MediaSpan, Prepared, Prompt};
    use crate::mock::{IMAGE_ID, MediaCall, MediaTokenizer, MockEngine};
    use crate::reasoning::{ThinkEntry, ThinkTags};
    use crate::sampling;

    /// `before`, an image of `len` positions and key `[key; 32]`, `after`.
    fn prompt(before: &str, len: usize, key: u8, after: &str) -> Prompt {
        let mut ids = MediaTokenizer.encode(before);
        let at = ids.len();
        ids.extend(std::iter::repeat_n(IMAGE_ID, len));
        ids.extend(MediaTokenizer.encode(after));
        let plan = GridPlan {
            n_llm_h: 1,
            n_llm_w: len,
            best_h: 1,
            best_w: len,
        };
        let feed = MediaFeed {
            at,
            key: ImageKey([key; 32]),
            prepared: Arc::new(Prepared::of_grid(
                len,
                Patches {
                    plan,
                    n_vit_h: 0,
                    n_vit_w: 0,
                    patch_len: 0,
                    bf16: Vec::new(),
                },
            )),
        };
        Prompt {
            held: Held {
                ids,
                media: vec![feed.span()],
            },
            feeds: vec![feed],
        }
    }

    /// One greedy request of four tokens: what the cache kept, and the tokens.
    fn run(slot: &mut Slot, p: &Prompt) -> Result<(usize, Vec<u32>), String> {
        let params = GenParams {
            n_predict: 4,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        let factory = sampling::reference_factory();
        let sampler = sampler_of(&factory, &params).expect("the reference refuses none");
        let mut g = Gen::new(slot, sampler, p.held.len(), &params);
        let o = g
            .run(
                slot,
                &p.held,
                &p.feeds,
                &params,
                None,
                &mut |_| Ok(()),
                &mut |_| {},
            )
            .map_err(|e| format!("{e:?}"))?;
        Ok((o.timings.cache_n, o.tokens))
    }

    /// The media mock on a slot of its own, and its log of image calls.
    fn media_slot() -> (Slot, Arc<Mutex<Vec<MediaCall>>>) {
        let (mock, log) = MockEngine::new(4096).with_media();
        (Slot::new(Box::new(mock)), log)
    }

    /// What `p` generates on a fresh slot.
    fn fresh(p: &Prompt) -> Vec<u32> {
        run(&mut media_slot().0, p).expect("a fresh generation").1
    }

    fn calls(log: &Mutex<Vec<MediaCall>>) -> Vec<MediaCall> {
        std::mem::take(&mut *log.lock().expect("the media log"))
    }

    /// A second chat whose image is another of the same size keeps nothing of
    /// the first's span: the ids are equal (every span position carries the
    /// image token), the image is not, so the reuse stops at the span's start
    /// and the span is fed again, and the reply is the one a fresh slot gives,
    /// not the first image's.
    #[test]
    fn another_image_of_one_size_stops_the_reuse_at_the_span() {
        let (mut slot, log) = media_slot();
        let a = prompt("q\n", 3, 1, "\n");
        let b = prompt("q\n", 3, 2, "\n");
        assert_eq!(a.held.ids, b.held.ids, "one size: the same ids");
        let (kept, from_a) = run(&mut slot, &a).expect("a");
        assert_eq!(kept, 0);
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: a.held.ids[..5].to_vec(),
                spans: a.held.media.clone(),
            }]
        );
        let (kept, from_b) = run(&mut slot, &b).expect("b");
        assert_eq!(kept, 2, "the span's start");
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: vec![IMAGE_ID; 3],
                spans: vec![MediaSpan {
                    at: 0,
                    len: 3,
                    key: ImageKey([2; 32]),
                }],
            }],
            "the span fed again at the call's own positions"
        );
        assert_eq!(from_b, fresh(&b), "the reply of b alone");
        assert_ne!(from_b, from_a, "the replies name their images");
    }

    /// The quantum's grid does not cut inside an image span
    /// ([`quantum_ends`]): the multiples 4 and 8 inside the span over
    /// positions 2 to 10 are passed over, so the span rides one call, and the
    /// call ends on the next multiple, 12. A multiple at a span's start or
    /// end is not inside it: the span over 4 to 11 rides the call from 4 to
    /// 12.
    #[test]
    fn the_quantum_passes_over_the_multiples_inside_a_span() {
        let p = prompt("ab", 9, 1, "cdefgh");
        let to = p.held.ids.len() - 1;
        assert_eq!((p.feeds[0].at, to), (2, 16), "the prompt's layout");
        assert_eq!(quantum_ends(0, to, &[], &p.feeds, 4), [12, 16]);
        let p = prompt("abcd", 8, 1, "efghij");
        let to = p.held.ids.len() - 1;
        assert_eq!((p.feeds[0].at, to), (4, 17), "the prompt's layout");
        assert_eq!(quantum_ends(0, to, &[], &p.feeds, 4), [4, 12, 16, 17]);
    }

    /// The same image again keeps the span whole: the reuse passes its end,
    /// no image is fed, and the reply is a fresh slot's.
    #[test]
    fn the_same_image_keeps_the_span() {
        let (mut slot, log) = media_slot();
        let a = prompt("q\n", 3, 1, "\n");
        run(&mut slot, &a).expect("a");
        calls(&log);
        let again = prompt("q\n", 3, 1, "\nq\n");
        let (kept, got) = run(&mut slot, &again).expect("again");
        assert_eq!(kept, 6, "all a shares with it: past the span 2..5");
        assert_eq!(calls(&log), [], "a span kept whole is not fed again");
        assert_eq!(got, fresh(&again));
    }

    /// The media mock that keeps only multiples of four positions.
    struct Fours(MockEngine);

    impl Engine for Fours {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn media_model(&self) -> Option<crate::media::SharedMediaModel> {
            self.0.media_model()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.0.prefill(ids)
        }
        fn prefill_media(&mut self, ids: &[u32], m: &[MediaFeed]) -> Result<(), EngineError> {
            self.0.prefill_media(ids, m)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.0.next(last, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.0.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.0.keepable(n) / 4 * 4
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
    }

    /// An engine whose keep would end inside a span keeps none of the span:
    /// the keep goes back to the span's start and the engine is asked again,
    /// so the span is fed whole and the reply is a fresh slot's.
    #[test]
    fn a_keep_inside_a_span_goes_back_to_its_start() {
        let (mock, log) = MockEngine::new(4096).with_media();
        let mut slot = Slot::new(Box::new(Fours(mock)));
        let a = prompt("abcde", 5, 1, "\n");
        run(&mut slot, &a).expect("a");
        calls(&log);
        let again = prompt("abcde", 5, 1, "\nq\n");
        let (kept, got) = run(&mut slot, &again).expect("a keep inside the span");
        assert_eq!(
            kept, 4,
            "11 shared, 8 lies inside 5..10, then 5 rounds to 4"
        );
        assert_eq!(
            calls(&log),
            [MediaCall {
                ids: again.held.ids[4..again.held.len() - 1].to_vec(),
                spans: vec![MediaSpan {
                    at: 1,
                    len: 5,
                    key: ImageKey([1; 32]),
                }],
            }]
        );
        assert_eq!(got, fresh(&again));
    }
}

#[cfg(test)]
mod slot_tests {
    use std::sync::Arc;

    use super::Slot;
    use crate::engine::{Engine, EngineError, SlotRow, Tokenizer};
    use crate::media::Held;
    use crate::mock::MockEngine;

    /// The two-slot mock whose steps of several slots answer nothing.
    struct Mute(MockEngine);

    impl Engine for Mute {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.0.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.0.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.0.next(last, out)
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
        fn slots(&self) -> usize {
            self.0.slots()
        }
        fn select_slot(&mut self, slot: usize) -> Result<(), EngineError> {
            self.0.select_slot(slot)
        }
        fn step_slots(&mut self, _rows: &mut [SlotRow<'_>]) -> Result<(), EngineError> {
            Ok(())
        }
    }

    fn rows() -> Vec<SlotRow<'static>> {
        (0..2)
            .map(|slot| SlotRow {
                slot,
                last: 10,
                logits: None,
                next: 0,
            })
            .collect()
    }

    /// A step of several slots books each row's id on its own slot, and a
    /// row the engine leaves unanswered is a named error, never id 0.
    #[test]
    fn a_step_of_several_slots_books_each_and_names_a_row_left_unanswered() {
        let mut slot = Slot::with_slots(Box::new(MockEngine::new(64).with_slots(2)), 2);
        let mut r = rows();
        slot.step_slots(&mut r).expect("the mock answers");
        let want = Held::from(vec![10]);
        assert_eq!((slot.held_of(0), slot.held_of(1)), (&want, &want));
        let mut mute = Slot::with_slots(Box::new(Mute(MockEngine::new(64).with_slots(2))), 2);
        let e = mute
            .step_slots(&mut rows())
            .expect_err("a row left unanswered");
        assert!(e.0.contains("answered slot 0 with id 4294967295"), "{e}");
    }
}

#[cfg(test)]
mod think_tests {
    use std::sync::{Arc, Mutex};

    use super::{GenParams, LOOP_RUN_MIN, Outcome, Repeats, Slot, StopKind, Timings, generate};
    use crate::engine::{Drafted, Engine, EngineError, SamplingParams, Tokenizer};
    use crate::media::MediaFeed;
    use crate::mock::{DraftMock, MockEngine, MockTokenizer};
    use crate::reasoning::{ThinkEntry, ThinkTags};
    use crate::sampling;

    /// An engine behind a log of every id fed to it: `prefill`'s, every
    /// step's `last`, and a pass's `last` and kept tokens but its last.
    struct Fed<E> {
        inner: E,
        fed: Arc<Mutex<Vec<u32>>>,
    }

    impl<E: Engine> Engine for Fed<E> {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.fed.lock().expect("fed").extend_from_slice(ids);
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.fed.lock().expect("fed").push(last);
            self.inner.next(last, out)
        }
        fn prefill_next(
            &mut self,
            ids: &[u32],
            media: &[MediaFeed],
            marks: &[usize],
            out: Option<&mut [f32]>,
        ) -> Result<u32, EngineError> {
            self.fed.lock().expect("fed").extend_from_slice(ids);
            self.inner.prefill_next(ids, media, marks, out)
        }
        fn advance(&mut self, last: u32, out: &mut Vec<u32>) -> Result<Drafted, EngineError> {
            let d = self.inner.advance(last, out)?;
            let mut fed = self.fed.lock().expect("fed");
            fed.push(last);
            fed.extend(&out[..out.len() - 1]);
            Ok(d)
        }
        fn advance_rows(&self) -> usize {
            self.inner.advance_rows()
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// One greedy generation of at most 8 tokens of `prompt` with `budget`,
    /// and every id the engine was fed. A prompt holding `<think>` twice makes
    /// the mock echo `a`, `<think>`, `a`, `<think>` forever after the span
    /// opens, so only the forced close ever closes the span; one holding
    /// `<think></think>` makes it close the span itself on its first token.
    fn run_on<E: Engine + 'static>(
        inner: E,
        prompt: &str,
        budget: Option<usize>,
    ) -> (Vec<u32>, Outcome) {
        run_with(inner, prompt, |p| p.reasoning_budget = budget)
    }

    /// [`run_on`] with the request's parameters as `set` leaves them.
    fn run_with<E: Engine + 'static>(
        inner: E,
        prompt: &str,
        set: impl FnOnce(&mut GenParams),
    ) -> (Vec<u32>, Outcome) {
        let fed = Arc::new(Mutex::new(Vec::new()));
        let mut slot = Slot::new(Box::new(Fed {
            inner,
            fed: Arc::clone(&fed),
        }));
        let ids = MockTokenizer.encode(prompt);
        let mut p = GenParams {
            n_predict: 8,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::of_prompt(prompt),
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        set(&mut p);
        let mut tim = Timings::default();
        let o = generate(
            &mut slot,
            &sampling::reference_factory(),
            &ids,
            &p,
            &mut |_| Ok(()),
            &mut |_| {},
            &mut tim,
        )
        .expect("a generation");
        let fed = std::mem::take(&mut *fed.lock().expect("fed"));
        (fed, o)
    }

    fn run(prompt: &str, budget: Option<usize>) -> (Vec<u32>, Outcome) {
        run_on(MockEngine::new(4096), prompt, budget)
    }

    /// A mock that books every [`Engine::prefill_next`] it takes, then runs
    /// it as the default does.
    struct Closes {
        inner: MockEngine,
        calls: Arc<Mutex<Vec<Vec<u32>>>>,
    }

    impl Engine for Closes {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.inner.next(last, out)
        }
        fn prefill_next(
            &mut self,
            ids: &[u32],
            media: &[MediaFeed],
            marks: &[usize],
            out: Option<&mut [f32]>,
        ) -> Result<u32, EngineError> {
            self.calls.lock().expect("calls").push(ids.to_vec());
            self.inner.prefill_next(ids, media, marks, out)
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// A prompt's last call and the step of its last id reach the engine as
    /// one `prefill_next` carrying the whole prompt, the step of no other id
    /// among them, and the answer is the two calls'.
    #[test]
    fn a_prompt_closes_in_one_call_with_its_last_id() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let closes = Closes {
            inner: MockEngine::new(4096),
            calls: Arc::clone(&calls),
        };
        let prompt = "qaqa";
        let (fed, o) = run_on(closes, prompt, None);
        let (fed_two, two) = run(prompt, None);
        let calls = std::mem::take(&mut *calls.lock().expect("calls"));
        assert_eq!(calls, vec![MockTokenizer.encode(prompt)]);
        assert_eq!(o.tokens, two.tokens);
        assert_eq!(fed, fed_two);
    }

    /// The mock ids these tests spell: `q`, `a`, the span's open and close.
    const Q: u32 = 6 + 113;
    const A: u32 = 6 + 97;
    const CLOSE: u32 = 5;

    /// A budget of 3 spends on the third generated id and the next id the
    /// engine is fed is the close id, which is taken, counted and fed itself.
    #[test]
    fn the_budget_spends_into_the_forced_close() {
        assert_eq!(MockTokenizer.encode("</think>"), vec![CLOSE]);
        let (fed, o) = run("q<think>aa<think>", Some(3));
        // `a`, `<think>`, `a` spent, the close forced, the eos the model
        // answers after it.
        assert_eq!(o.tokens, vec![A, 4, A, CLOSE, MockTokenizer.eos()]);
        assert_eq!(o.timings.predicted_n, 5, "the close counts like any token");
        assert_eq!(o.stop, StopKind::Eos);
        // The prompt's four prefill ids and its last, the three spent ids,
        // then the close.
        assert_eq!(fed, vec![Q, 4, A, A, 4, A, 4, A, CLOSE]);
    }

    /// A budget of 0 forces the close before any model token: the first taken
    /// id is the close id, and the engine is fed nothing the model answered.
    #[test]
    fn a_budget_of_zero_closes_before_any_model_token() {
        let (fed, o) = run("q<think>aa<think>", Some(0));
        assert_eq!(o.tokens, vec![CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![Q, 4, A, A, 4, CLOSE]);
    }

    /// A close the model makes itself retires the tracker: the fed ids and
    /// the tokens are a no-budget run's exactly.
    #[test]
    fn a_natural_close_retires_the_tracker() {
        let prompt = "z<think></think>abab<think>";
        let (fed, with) = run(prompt, Some(4));
        let (fed_free, without) = run(prompt, None);
        assert_eq!(with.tokens, without.tokens);
        assert_eq!(fed, fed_free);
        assert!(
            with.tokens.len() > 4,
            "the budget is smaller than the run, or it pins nothing"
        );
    }

    /// A drafting engine stops passing near the limit: a pass of 2 rows needs
    /// the budget to hold 2, so with 4 the loop steps once at 1 left and the
    /// force fires at exactly 4 taken ids. The pass the span's close itself
    /// opens afterwards is the normal flow resumed, its eos ending the run.
    #[test]
    fn a_pass_waits_for_the_budget_to_hold_its_rows() {
        let (fed, o) = run_on(DraftMock::new(4096), "q<think>aa<think>", Some(4));
        assert_eq!(o.tokens, vec![A, 4, A, 4, CLOSE, MockTokenizer.eos()]);
        // The prompt's four prefill ids and its last, the pass's `a` and
        // `<think>`, then the gated step's `a` (a second pass here would feed
        // `a` and `<think>` again), the armed step's `<think>`, the close, and
        // the eos the pass the retired span opens kept.
        assert_eq!(
            fed,
            vec![Q, 4, A, A, 4, A, 4, A, 4, CLOSE, MockTokenizer.eos()]
        );
    }

    /// The mock ids the model-opened cases spell: `x` and `y`. A prompt
    /// `x<think>yx` ends in plain text, and the mock's first answer is the
    /// `<think>` its earlier occurrence taught (the prompt's `x` follows with
    /// it): the model opens the span itself.
    const X: u32 = 6 + 120;
    const Y: u32 = 6 + 121;

    /// A span the model opens itself: the opening id spends nothing, no close
    /// id is forced before it, and the budget counts the ids after it until
    /// the forced close retires the tracker.
    #[test]
    fn a_model_opened_span_spends_from_its_open_tag() {
        let (fed, o) = run("x<think>yx", Some(2));
        assert_eq!(o.tokens, vec![4, Y, X, CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![X, 4, Y, X, 4, Y, X, CLOSE]);
    }

    /// A budget of 0 over a model-opened span forces the close from the first
    /// id inside it — the model's opening `<think>` passes, as the start tag
    /// is where the sampler begins, and the second taken id is the close.
    #[test]
    fn a_budget_of_zero_forces_after_the_model_opens() {
        let (fed, o) = run("x<think>yx", Some(0));
        assert_eq!(o.tokens, vec![4, CLOSE, MockTokenizer.eos()]);
        assert_eq!(fed, vec![X, 4, Y, X, 4, CLOSE]);
    }

    /// The loop guard over a span of at most 1024 ids, on `inner`.
    fn guarded<E: Engine + 'static>(inner: E, prompt: &str, on: bool) -> Outcome {
        run_with(inner, prompt, |p| {
            p.n_predict = 1024;
            p.reasoning_loop = on;
        })
        .1
    }

    /// The guard ends a span that repeats `a`, `<think>`: the run of period 2
    /// reaches [`LOOP_RUN_MIN`] at the 258th id, which ends the generation
    /// with its text out, on a drafting engine too, whose pass's ids past it
    /// are dropped. Unguarded, the span runs to the limit. A span the model
    /// opens is read after its start tag: `y`, `x`, `<think>` repeat from the
    /// second id, so its loop ends one id later than a read from the first.
    #[test]
    fn the_loop_guard_ends_a_span_that_repeats_a_cycle() {
        let prompt = "q<think>aa<think>";
        for o in [
            guarded(MockEngine::new(4096), prompt, true),
            guarded(DraftMock::new(4096), prompt, true),
        ] {
            assert_eq!((o.stop, o.tokens.len()), (StopKind::Loop, LOOP_RUN_MIN + 2));
            assert_eq!(o.content, "a<think>".repeat(LOOP_RUN_MIN / 2 + 1));
        }
        let free = guarded(MockEngine::new(4096), prompt, false);
        assert_eq!((free.stop, free.tokens.len()), (StopKind::Limit, 1024));
        let opened = guarded(MockEngine::new(4096), "x<think>yx", true);
        assert_eq!(&opened.tokens[..4], [4, Y, X, 4]);
        assert_eq!(
            (opened.stop, opened.tokens.len()),
            (StopKind::Loop, 1 + LOOP_RUN_MIN + 3)
        );
    }

    /// A cycle after the span closed is the answer's: the model closes the
    /// span on its first id, then repeats `a`, `b`, `<think>`, `</think>` to
    /// the limit.
    #[test]
    fn the_loop_guard_leaves_the_answer_alone() {
        let o = guarded(MockEngine::new(4096), "z<think></think>abab<think>", true);
        assert_eq!(o.tokens[0], CLOSE);
        assert_eq!((o.stop, o.tokens.len()), (StopKind::Limit, 1024));
    }

    /// A cycle longer than [`LOOP_RUN_MIN`] is a loop once two whole copies
    /// stand: 300 distinct ids twice end in a loop at the 600th id, not
    /// before. One id repeated is a loop at [`LOOP_RUN_MIN`] + 1 ids; repeats
    /// that are not in a row are none.
    #[test]
    fn a_loop_holds_two_copies_of_its_cycle() {
        let first = |ids: &[u32]| {
            let mut r = Repeats::new();
            (1..=ids.len()).find(|&n| r.looped(&ids[..n]))
        };
        let long: Vec<u32> = (0..300).chain(0..300).chain(0..300).collect();
        assert_eq!(first(&long), Some(600));
        assert_eq!(first(&[7; 400]), Some(LOOP_RUN_MIN + 1));
        assert_eq!(first(&long[..599]), None);
        let at_limit: Vec<u32> = (0..1024).chain(0..1024).collect();
        assert_eq!(first(&at_limit), Some(2048));
        let past_limit: Vec<u32> = (0..1025).chain(0..1025).collect();
        assert_eq!(first(&past_limit), None);
        let sparse: Vec<u32> = (0..300).flat_map(|k| [3 * k, 3 * k, 3 * k + 1]).collect();
        assert_eq!(
            first(&sparse),
            None,
            "300 repeats of period 1, never 2 in a row"
        );
    }
}

#[cfg(test)]
mod sampled_tests {
    use std::sync::{Arc, Mutex};

    use super::{Bias, GenError, GenParams, Outcome, Slot, Timings, biased, generate};
    use crate::engine::{Drafted, Engine, EngineError, Sampler, SamplingParams, Tokenizer};
    use crate::mock::{DraftMock, MockTokenizer};
    use crate::reasoning::{ThinkEntry, ThinkTags};
    use crate::sampling;

    fn ban(ids: &[u32]) -> Bias {
        Bias {
            add: Vec::new(),
            banned: ids.to_vec(),
        }
    }

    #[test]
    fn a_greedy_banning_draw_ties_signed_zeros_by_id() {
        let mut draw = biased(None, ban(&[2]));
        for row in [[-0.0, 0.0, -1.0], [0.0, -0.0, -1.0]] {
            assert_eq!(draw(&row, &[]), 0);
        }
        let mut draw = biased(None, ban(&[0]));
        assert_eq!(draw(&[1.0, -0.0, 0.0], &[]), 1);
    }

    /// A draw's copy keeps the row's zero signs, adds each bias and puts a
    /// banned id at -inf whatever its bias.
    #[test]
    fn a_sampled_biased_draw_keeps_the_rows_zero_signs() {
        let sampler: Sampler = Box::new(|row, history| {
            assert_eq!(row[0].to_bits(), (-0.0f32).to_bits());
            assert_eq!(row[1].to_bits(), 0.0f32.to_bits());
            assert_eq!(row[2], f32::NEG_INFINITY);
            assert_eq!(row[3], 1.5);
            assert_eq!(history, [4, 5]);
            sampling::argmax(row)
        });
        let bias = Bias {
            add: vec![(2, 5.0), (3, 2.5)],
            banned: vec![2],
        };
        let mut draw = biased(Some(sampler), bias);
        assert_eq!(draw(&[-0.0, 0.0, -1.0, -1.0], &[4, 5]), 3);
    }

    /// [`DraftMock`] that logs what each request tells [`Engine::will_reply`].
    struct Replies {
        inner: DraftMock,
        replies: Arc<Mutex<Vec<Option<usize>>>>,
    }

    impl Engine for Replies {
        fn tokenizer(&self) -> Arc<dyn Tokenizer> {
            self.inner.tokenizer()
        }
        fn prefill(&mut self, ids: &[u32]) -> Result<(), EngineError> {
            self.inner.prefill(ids)
        }
        fn next(&mut self, last: u32, out: Option<&mut [f32]>) -> Result<u32, EngineError> {
            self.inner.next(last, out)
        }
        fn advance_rows(&self) -> usize {
            self.inner.advance_rows()
        }
        fn advance_sampled(
            &mut self,
            last: u32,
            history: Vec<u32>,
            sampler: Sampler,
            best: Option<usize>,
            out: &mut Vec<u32>,
        ) -> (Result<Drafted, EngineError>, Vec<u32>, Sampler) {
            self.inner
                .advance_sampled(last, history, sampler, best, out)
        }
        fn drafts_sampled(&self) -> bool {
            self.inner.drafts_sampled()
        }
        fn will_reply(&mut self, tokens: Option<usize>) {
            self.replies.lock().expect("the replies").push(tokens);
        }
        fn reset(&mut self) -> Result<(), EngineError> {
            self.inner.reset()
        }
        fn keepable(&self, n: usize) -> usize {
            self.inner.keepable(n)
        }
        fn cut(&mut self, n: usize) -> Result<(), EngineError> {
            self.inner.cut(n)
        }
        fn ctx_max(&self) -> usize {
            self.inner.ctx_max()
        }
        fn describe(&self) -> String {
            self.inner.describe()
        }
    }

    /// On an engine that drafts sampled requests, a sampled request tells it
    /// the tokens after its first, as a greedy one does (the passes make at
    /// most those), and so does the same request banning the end of
    /// generation, which draws in those passes too.
    #[test]
    fn a_sampled_request_tells_a_drafting_engine_its_passes() {
        let replies = Arc::new(Mutex::new(Vec::new()));
        let mut slot = Slot::new(Box::new(Replies {
            inner: DraftMock::new(4096),
            replies: Arc::clone(&replies),
        }));
        let ids = MockTokenizer.encode("abacadaeabacada");
        for ignore_eos in [false, true] {
            let p = GenParams {
                n_predict: 8,
                sampling: SamplingParams {
                    temperature: 0.8,
                    seed: 3,
                    ..SamplingParams::default()
                },
                stop: Vec::new(),
                ignore_eos,
                stream: false,
                timings_per_token: false,
                return_progress: false,
                include_usage: false,
                cache_prompt: true,
                reasoning_budget: None,
                reasoning_loop: false,
                think_entry: ThinkEntry::Closed,
                think: ThinkTags::THINK,
                think_split: None,
                logprobs: None,
                constraint: None,
                logit_bias: Vec::new(),
            };
            let mut tim = Timings::default();
            generate(
                &mut slot,
                &sampling::reference_factory(),
                &ids,
                &p,
                &mut |_| Ok(()),
                &mut |_| {},
                &mut tim,
            )
            .expect("a generation");
        }
        assert_eq!(*replies.lock().expect("the replies"), [Some(7), Some(7)]);
    }

    /// A llama-bpe vocabulary as the server loads one (`tokenizer::Tokenizer::from_keys`
    /// through [`crate::vocab::Vocab`], whose decoder and special mapping the loop reads):
    /// the four CONTROL tokens below, then the byte-level alphabet minus the `'` the
    /// apostrophe special already spells, a vocabulary holding a text once. `foo ` and `'`
    /// are the spellings whose pieces the space cleaner does not release whole: `foo `
    /// releases `foo` and holds its space, and `'` after a held space is held whole
    /// (` ' ` starts with ` '`), where `<|start_header_id|>` no pattern starts.
    fn llama_bpe_vocab() -> Arc<dyn Tokenizer> {
        use gguf::Value;

        const SPECIALS: [&str; 4] = ["<|end_of_text|>", "<|start_header_id|>", "foo ", "'"];
        let bytes = tokenizer::byte_to_cpt_table().map(|c| char::from_u32(c).unwrap());
        let tokens: Vec<Value> = SPECIALS
            .iter()
            .map(|s| Value::String((*s).to_owned()))
            .chain(
                bytes
                    .iter()
                    .filter(|c| !SPECIALS.contains(&c.to_string().as_str()))
                    .map(|c| Value::String(c.to_string())),
            )
            .collect();
        let types: Vec<Value> = std::iter::repeat_n(Value::I32(3), SPECIALS.len())
            .chain(std::iter::repeat_n(
                Value::I32(1),
                tokens.len() - SPECIALS.len(),
            ))
            .collect();
        let vocab = tokenizer::Tokenizer::from_keys(&[
            ("tokenizer.ggml.model".into(), Value::String("gpt2".into())),
            (
                "tokenizer.ggml.pre".into(),
                Value::String("llama-bpe".into()),
            ),
            ("tokenizer.ggml.tokens".into(), Value::Array(tokens)),
            ("tokenizer.ggml.token_type".into(), Value::Array(types)),
            ("tokenizer.ggml.merges".into(), Value::Array(vec![])),
            ("tokenizer.ggml.eos_token_id".into(), Value::U32(0)),
            ("tokenizer.ggml.add_bos_token".into(), Value::Bool(false)),
        ])
        .unwrap();
        Arc::new(crate::vocab::Vocab::new(vocab).unwrap())
    }

    /// One generation over [`llama_bpe_vocab`]: the prompt `prompt`, then the scripted ids
    /// `script`, its streamed text and special spans collected beside the outcome.
    fn llama_bpe_run(
        vocab: &Arc<dyn Tokenizer>,
        script: Vec<u32>,
        prompt: &str,
        n_predict: i64,
    ) -> (Result<Outcome, GenError>, String, Vec<(usize, usize)>) {
        let mut slot = Slot::new(Box::new(
            crate::mock::ScriptedEngine::from_ids(4096, script).with_tokenizer(Arc::clone(vocab)),
        ));
        let p = GenParams {
            n_predict,
            sampling: SamplingParams {
                temperature: 0.0,
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            ignore_eos: false,
            stream: false,
            timings_per_token: false,
            return_progress: false,
            include_usage: false,
            cache_prompt: true,
            reasoning_budget: None,
            reasoning_loop: false,
            think_entry: ThinkEntry::Closed,
            think: ThinkTags::THINK,
            think_split: None,
            logprobs: None,
            constraint: None,
            logit_bias: Vec::new(),
        };
        let mut text = String::new();
        let mut specials = Vec::new();
        let mut tim = Timings::default();
        let r = generate(
            &mut slot,
            &sampling::reference_factory(),
            &vocab.encode(prompt),
            &p,
            &mut |ev| {
                if let super::Event::Text(t, s, _) = ev {
                    text.push_str(t);
                    specials.extend_from_slice(s);
                }
                Ok(())
            },
            &mut |_| {},
            &mut tim,
        );
        (r, text, specials)
    }

    /// A llama-bpe special id pushed from an empty decoder keeps its whole text in its
    /// piece, so the loop's special mapping finds the text at the piece's end:
    /// `<|start_header_id|>` is 19 bytes none of the cleaner's patterns starts.
    #[test]
    fn a_llama_bpe_special_pushed_from_an_empty_decoder_does_not_panic() {
        let vocab = llama_bpe_vocab();
        let (r, text, specials) = llama_bpe_run(&vocab, vec![1], "x", 1);
        let outcome = r.expect("a generation");
        assert_eq!(text, "<|start_header_id|>");
        assert_eq!(outcome.specials, [(0, 19)]);
        assert_eq!(specials, [(0, 19)]);
    }

    /// A special whose piece the cleaner cuts short ends the request by name:
    /// `foo ` streams `foo` while its whole decoded text keeps the space, so no
    /// place in the piece holds the text.
    ///
    /// FAIL-first: the span mapping expected the special's whole decoded text
    /// at its piece's end and panicked on `foo `.
    #[test]
    fn a_special_the_cleaner_cuts_short_ends_the_request() {
        let vocab = llama_bpe_vocab();
        let (r, ..) = llama_bpe_run(&vocab, vec![2], "x", 1);
        assert!(
            matches!(&r, Err(GenError::SpecialPiece(2))),
            "foo: {:?}",
            r.err()
        );
    }

    /// A special the cleaner holds whole, so no piece carries it at all, ends the
    /// request by name too: a lone `'` after a space makes ` ' `, whose prefix
    /// ` '` the cleaner holds for the piece that completes it, which never comes.
    ///
    /// FAIL-first: without its piece the special's span was lost silently.
    #[test]
    fn a_special_the_cleaner_holds_whole_ends_the_request() {
        let vocab = llama_bpe_vocab();
        let space = *vocab.encode(" ").first().expect("the space is a token");
        let (r, ..) = llama_bpe_run(&vocab, vec![space, 3], "x", 2);
        assert!(
            matches!(&r, Err(GenError::SpecialPiece(3))),
            "apostrophe: {:?}",
            r.err()
        );
    }
}
