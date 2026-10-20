//! The segment program and the walk over it.
//!
//! A program is text first ([`Part`]): forced text, a choice among option
//! texts, a free string. [`compile`] turns it into ids ([`Program`]) with the
//! serve [`Tokenizer`], each forced text and each option canonically on its own
//! (a string's opening quote belongs to the text before it, its closing quote
//! to the text after). A [`Walk`] then answers, at each token, which ids the
//! program allows and takes the masked argmax over them.
//!
//! - `Forced(ids)`: one allowed id at a time.
//! - A choice is a trie of its options' ids. The options must be prefix-free
//!   as token sequences: an option that is a prefix of another (or a repeat of
//!   one) is a named error when the program is built, never a pick the walk
//!   has to guess at. Only a choice that is the last part of its list carries
//!   a continuation per option (a tool call's arguments differ by the tool);
//!   every other choice returns to the parts after it.
//! - A free string takes any token of its class ([`Class`]) while it fits the
//!   length limit, and ends on a token whose text is a non-empty prefix of the
//!   forced text after it: a token `",` ends a string and takes the `,` that
//!   follows. The rest of that text is tokenised canonically from where the
//!   end token stopped.
//! - Once the program is spent the only allowed ids are the vocabulary's
//!   stops. A stop id inside the program would end the generation with the
//!   program unspent, so [`compile`] refuses any text that tokenises to one
//!   and the walk refuses to take one before the program is spent.
//!
//! The pick at every decision is the largest logit among the allowed ids, ties
//! to the lowest id ([`sampling::argmax_in`]): the engine's argmax rule over
//! the allowed set. The walk reads no row at a forced id, nor at the only
//! child of a choice's node ([`Walk::fixed_next`]).

use std::sync::{Arc, Mutex};

use super::Refusal;
use super::vocab::{Class, Vocab};
use crate::api::relock;
use crate::engine::Tokenizer;
use crate::sampling;

/// How a free string is quoted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Style {
    /// A JSON string in a reply no tool-call scan reads: its quotes are the
    /// surrounding text's, and its body takes no `"`, no `\` and no control
    /// character.
    Json,
    /// A JSON string inside a chat format's markup, or in a reply the
    /// request's tool-call scan reads: the body also takes no `<`, since a
    /// tag begins with one and would end the markup's value, or open a call,
    /// inside the string.
    JsonInMarkup,
    /// Raw text that ends at the text after it: the body takes no token that
    /// holds that text's first byte.
    Raw,
}

impl Style {
    /// Whether a string is written as JSON, quoted and escaped.
    pub(crate) fn is_json(self) -> bool {
        matches!(self, Style::Json | Style::JsonInMarkup)
    }

    /// The style of a JSON string inside an object written in this style: a
    /// raw value is markup's, so the strings in an object it holds are in
    /// markup.
    pub(crate) fn nested(self) -> Style {
        match self {
            Style::Json => Style::Json,
            Style::JsonInMarkup | Style::Raw => Style::JsonInMarkup,
        }
    }
}

/// A program as text.
#[derive(Clone, Debug)]
pub(crate) enum Part {
    /// Forced text.
    Text(String),
    /// A choice among option texts, each with the parts that follow it.
    Pick(Vec<Alt>),
    /// A free string of at most `max` characters; the forced text after it
    /// ends it.
    Free { max: Option<usize>, style: Style },
}

/// One option of a [`Part::Pick`].
#[derive(Clone, Debug)]
pub(crate) struct Alt {
    pub text: String,
    /// What follows this option instead of what follows the choice; empty to
    /// continue with the parts after the choice. A non-empty continuation is
    /// the last of its list's parts.
    pub then: Vec<Part>,
}

/// A program as ids.
#[derive(Debug)]
pub(crate) struct Program {
    steps: Vec<Step>,
}

#[derive(Debug)]
enum Step {
    Forced(Vec<u32>),
    Pick(Trie),
    Free(FreeStep),
}

#[derive(Debug)]
struct Trie {
    nodes: Vec<Node>,
    /// The continuation of each option, by option.
    subs: Vec<Option<Arc<Program>>>,
}

impl Trie {
    /// The id and node of the only child of `node`, when it has one.
    fn only_kid(&self, node: usize) -> Option<(u32, usize)> {
        match self.nodes[node].kids.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// The option every path from `node` completes, when it is one alone.
    fn sole_option(&self, node: usize) -> Option<usize> {
        let (mut found, mut open) = (None, vec![node]);
        while let Some(n) = open.pop() {
            if let Some(k) = self.nodes[n].leaf
                && found.replace(k).is_some()
            {
                return None;
            }
            open.extend(self.nodes[n].kids.iter().map(|&(_, child)| child));
        }
        found
    }
}

#[derive(Debug, Default)]
struct Node {
    /// Child nodes by the id that reaches them, ascending.
    kids: Vec<(u32, usize)>,
    /// The option this node completes.
    leaf: Option<usize>,
}

#[derive(Debug)]
struct FreeStep {
    class: Class,
    max: Option<usize>,
    /// The ids that end the string, ascending, with the ids of the text each
    /// leaves to be forced.
    ends: Vec<(u32, Vec<u32>)>,
    end_ids: Vec<u32>,
}

/// What [`compile`] tokenises with.
pub(crate) struct Env<'a> {
    pub tok: &'a dyn Tokenizer,
    pub vocab: &'a Vocab,
}

fn refuse(message: String) -> Refusal {
    Refusal::invalid(message)
}

/// Refuses `ids`, the tokens of `text`, when one is a stop id: the generation
/// would end there with the program unspent.
fn no_stop(ids: &[u32], text: &str, env: &Env<'_>) -> Result<(), Refusal> {
    match ids.iter().find(|id| env.vocab.stops().contains(id)) {
        Some(id) => Err(refuse(format!(
            "the text {text:?} tokenises to the stop id {id}: a stop id cannot occur before the \
             program is spent"
        ))),
        None => Ok(()),
    }
}

/// Joins neighbouring forced texts, so each is tokenised whole.
fn merged(parts: &[Part]) -> Vec<Part> {
    let mut out: Vec<Part> = Vec::new();
    for part in parts {
        match (part, out.last_mut()) {
            (Part::Text(t), Some(Part::Text(last))) => last.push_str(t),
            (Part::Text(t), _) if t.is_empty() => {}
            _ => out.push(part.clone()),
        }
    }
    out
}

/// `parts` as a program of ids.
pub(crate) fn compile(parts: &[Part], env: &Env<'_>) -> Result<Program, Refusal> {
    let parts = merged(parts);
    let mut steps = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        match &parts[i] {
            Part::Text(t) => {
                let ids = env.tok.encode(t);
                no_stop(&ids, t, env)?;
                steps.push(Step::Forced(ids));
            }
            Part::Pick(alts) => {
                if alts.iter().any(|a| !a.then.is_empty()) && i + 1 != parts.len() {
                    return Err(refuse(
                        "internal: a choice with continuations is not last in its list".to_owned(),
                    ));
                }
                steps.push(Step::Pick(trie(alts, env)?));
            }
            Part::Free { max, style } => {
                let Some(Part::Text(tail)) = parts.get(i + 1) else {
                    return Err(refuse(
                        "internal: a free string is not followed by forced text".to_owned(),
                    ));
                };
                steps.push(Step::Free(free_step(*max, *style, tail, env)?));
                i += 1;
            }
        }
        i += 1;
    }
    Ok(Program { steps })
}

fn trie(alts: &[Alt], env: &Env<'_>) -> Result<Trie, Refusal> {
    if alts.is_empty() {
        return Err(refuse("a choice with no option".to_owned()));
    }
    let mut nodes = vec![Node::default()];
    let mut subs = Vec::with_capacity(alts.len());
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(alts.len());
    for (k, alt) in alts.iter().enumerate() {
        let ids = env.tok.encode(&alt.text);
        if ids.is_empty() {
            return Err(refuse(format!(
                "the option {:?} tokenises to no token",
                alt.text
            )));
        }
        no_stop(&ids, &alt.text, env)?;
        let clash = |other: usize| {
            refuse(format!(
                "the options {:?} and {:?} tokenise so that one is a prefix of the other \
                 (ids {:?} and {:?}): the choice between them would be ambiguous",
                alts[other].text, alt.text, seqs[other], ids
            ))
        };
        let mut at = 0;
        for &id in &ids {
            if let Some(other) = nodes[at].leaf {
                return Err(clash(other));
            }
            at = match nodes[at].kids.binary_search_by_key(&id, |&(i, _)| i) {
                Ok(p) => nodes[at].kids[p].1,
                Err(p) => {
                    nodes.push(Node::default());
                    let child = nodes.len() - 1;
                    nodes[at].kids.insert(p, (id, child));
                    child
                }
            };
        }
        if let Some(other) = nodes[at].leaf {
            return Err(clash(other));
        }
        if !nodes[at].kids.is_empty() {
            // An earlier option runs on past this one.
            let other = seqs.iter().position(|s| s.starts_with(&ids)).unwrap_or(0);
            return Err(clash(other));
        }
        nodes[at].leaf = Some(k);
        seqs.push(ids);
        subs.push(if alt.then.is_empty() {
            None
        } else {
            Some(Arc::new(compile(&alt.then, env)?))
        });
    }
    Ok(Trie { nodes, subs })
}

fn free_step(
    max: Option<usize>,
    style: Style,
    tail: &str,
    env: &Env<'_>,
) -> Result<FreeStep, Refusal> {
    let first = tail.as_bytes()[0];
    let class = match style {
        Style::Json | Style::JsonInMarkup => {
            if first != b'"' {
                return Err(refuse(
                    "internal: a JSON string's closing text does not begin with a quote".to_owned(),
                ));
            }
            if style == Style::Json {
                env.vocab.json_class()
            } else {
                env.vocab.json_class_in_markup()
            }
        }
        Style::Raw => {
            if !first.is_ascii() {
                return Err(refuse(format!(
                    "a raw value ends at {tail:?}, whose first character is not ASCII"
                )));
            }
            env.vocab.raw_class(first)
        }
    };
    let mut ends: Vec<(u32, Vec<u32>)> = Vec::new();
    for (id, len) in env.vocab.prefixes_of(tail) {
        let rest = env.tok.encode(&tail[len..]);
        no_stop(&rest, tail, env)?;
        ends.push((id, rest));
    }
    if ends.is_empty() {
        return Err(refuse(format!(
            "no token of the vocabulary is a prefix of {tail:?}, so a free string before it \
             could never end"
        )));
    }
    let end_ids = ends.iter().map(|&(id, _)| id).collect();
    Ok(FreeStep {
        class,
        max,
        ends,
        end_ids,
    })
}

/// One child of a choice's node at a decision: its id, the logit the decision's
/// row holds for it, and the option it leads to alone (`None` when more than
/// one option lies below it).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Kid {
    pub id: u32,
    pub logit: f32,
    pub option: Option<usize>,
}

/// A pick among the children of a choice's node (more than one), as the row
/// held them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Decision {
    /// In ascending id order.
    pub kids: Vec<Kid>,
    /// The child taken.
    pub picked: u32,
}

/// The decisions of the walk that follows a constraint, in order, for the
/// route that reads them once the generation is done. A walk starts the record
/// afresh.
#[derive(Clone, Debug, Default)]
pub(crate) struct Watch(Arc<Mutex<Vec<Decision>>>);

impl Watch {
    /// The decisions so far, taken out.
    pub(crate) fn take(&self) -> Vec<Decision> {
        std::mem::take(&mut *relock(&self.0))
    }
}

/// A request's constraint, built once and shared by the walks that follow it.
#[derive(Clone)]
pub(crate) struct Constraint {
    prog: Arc<Program>,
    /// The ids that close an open think span, forced first when the prompt
    /// leaves one open.
    close: Arc<[u32]>,
    stops: Arc<[u32]>,
    /// Where its walks write their decisions, `None` for a request that reads
    /// none.
    watch: Option<Watch>,
}

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("steps", &self.prog.steps.len())
            .finish_non_exhaustive()
    }
}

impl Constraint {
    pub(crate) fn new(prog: Program, close: Vec<u32>, stops: Arc<[u32]>) -> Constraint {
        Constraint {
            prog: Arc::new(prog),
            close: close.into(),
            stops,
            watch: None,
        }
    }

    /// This constraint with its walks writing their decisions to `watch`.
    pub(crate) fn watched(mut self, watch: &Watch) -> Constraint {
        self.watch = Some(watch.clone());
        self
    }
}

/// A walk that cannot take a step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StepError {
    #[error("constrained output: no token is allowed after {taken} generated: {why}")]
    NoAllowed { taken: usize, why: String },
}

#[derive(Clone, Debug)]
struct Frame {
    prog: Arc<Program>,
    at: usize,
}

#[derive(Clone, Debug)]
enum Ctx {
    /// Between steps.
    Idle,
    Trie {
        prog: Arc<Program>,
        step: usize,
        node: usize,
    },
    Free {
        prog: Arc<Program>,
        step: usize,
        chars: usize,
    },
    /// The program is spent.
    Done,
}

/// How many ids a walk picked from a row, by kind (the ids the program fixes
/// and the stop pick are in neither).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Picks {
    /// Among the children of a choice's node, which has more than one.
    pub decisions: usize,
    /// Inside a free string, its end token included.
    pub free: usize,
}

#[cfg(test)]
thread_local! {
    static AHEADS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The copies [`Walk::ahead`] has made on this thread.
#[cfg(test)]
pub(crate) fn aheads() -> usize {
    AHEADS.with(std::cell::Cell::get)
}

/// One request's position in its program.
#[derive(Clone, Debug)]
pub(crate) struct Walk {
    frames: Vec<Frame>,
    ctx: Ctx,
    /// Forced ids not yet taken, the next one last.
    forced: Vec<u32>,
    stops: Arc<[u32]>,
    taken: usize,
    picks: Picks,
    watch: Option<Watch>,
}

impl Walk {
    /// A walk at the start of `c`'s program; `close_think` forces the think
    /// span's close first (a prompt that left the span open).
    pub(crate) fn new(c: &Constraint, close_think: bool) -> Walk {
        if let Some(w) = &c.watch {
            relock(&w.0).clear();
        }
        Walk {
            frames: vec![Frame {
                prog: Arc::clone(&c.prog),
                at: 0,
            }],
            ctx: Ctx::Idle,
            forced: if close_think {
                c.close.iter().rev().copied().collect()
            } else {
                Vec::new()
            },
            stops: Arc::clone(&c.stops),
            taken: 0,
            picks: Picks::default(),
            watch: c.watch.clone(),
        }
    }

    /// The ids picked from a row so far, by kind.
    pub(crate) fn picks(&self) -> Picks {
        self.picks
    }

    /// A copy to run ahead on; this walk stays where it stands.
    pub(crate) fn ahead(&self) -> Walk {
        #[cfg(test)]
        AHEADS.with(|n| n.set(n.get() + 1));
        self.clone()
    }

    /// Moves to the next decision, or to the next forced id.
    fn settle(&mut self) {
        loop {
            if !self.forced.is_empty() || !matches!(self.ctx, Ctx::Idle) {
                return;
            }
            let Some(frame) = self.frames.last_mut() else {
                self.ctx = Ctx::Done;
                return;
            };
            if frame.at >= frame.prog.steps.len() {
                self.frames.pop();
                continue;
            }
            let step = frame.at;
            frame.at += 1;
            let prog = Arc::clone(&frame.prog);
            match &prog.steps[step] {
                Step::Forced(ids) => self.forced = ids.iter().rev().copied().collect(),
                Step::Pick(_) => {
                    self.ctx = Ctx::Trie {
                        prog,
                        step,
                        node: 0,
                    }
                }
                Step::Free(_) => {
                    self.ctx = Ctx::Free {
                        prog,
                        step,
                        chars: 0,
                    };
                }
            }
        }
    }

    /// Whether the next id is the program's alone to give: a forced id, or the
    /// only child of a choice's node. [`Walk::choose`] reads no row for it.
    pub(crate) fn fixed_next(&mut self) -> bool {
        self.settle();
        if !self.forced.is_empty() {
            return true;
        }
        match &self.ctx {
            Ctx::Trie { prog, step, node } => match &prog.steps[*step] {
                Step::Pick(t) => t.only_kid(*node).is_some(),
                _ => false,
            },
            _ => false,
        }
    }

    /// Whether the program is spent: nothing is left but the stop.
    pub(crate) fn spent(&mut self) -> bool {
        self.settle();
        self.forced.is_empty() && matches!(self.ctx, Ctx::Done)
    }

    /// The run of ids the program fixes from here: forced ids and the only
    /// children of a choice's nodes, up to the next decision, the next free
    /// string or the program's end, and at most `max` of them. The walk stands
    /// past them, as [`Walk::choose`] would leave it after each; a run is
    /// what `choose` gives for as long as [`Walk::fixed_next`] holds, so it
    /// reads no row.
    pub(crate) fn forced_run(&mut self, max: usize) -> Result<Vec<u32>, StepError> {
        let mut run = Vec::new();
        while run.len() < max && self.fixed_next() {
            run.push(self.choose(&[])?);
        }
        Ok(run)
    }

    /// The next id: the forced one, or the largest logit of `row` among the
    /// ids the program allows here; the walk then stands past it. A stop id
    /// before the program is spent is an error.
    pub(crate) fn choose(&mut self, row: &[f32]) -> Result<u32, StepError> {
        self.settle();
        let id = match self.forced.pop() {
            Some(id) => id,
            None => match std::mem::replace(&mut self.ctx, Ctx::Idle) {
                Ctx::Done => {
                    self.ctx = Ctx::Done;
                    sampling::argmax_in(row, self.stops.iter().copied())
                        .ok_or_else(|| self.stuck("the row holds none of the stop ids"))?
                }
                Ctx::Trie { prog, step, node } => self.pick_option(row, prog, step, node)?,
                Ctx::Free { prog, step, chars } => self.pick_free(row, prog, step, chars)?,
                Ctx::Idle => return Err(self.stuck("the walk stands between steps")),
            },
        };
        if self.stops.contains(&id) && !matches!(self.ctx, Ctx::Done) {
            return Err(self.stuck("a stop id occurs before the program is spent"));
        }
        self.taken += 1;
        Ok(id)
    }

    fn stuck(&self, why: &str) -> StepError {
        StepError::NoAllowed {
            taken: self.taken,
            why: why.to_owned(),
        }
    }

    /// One id of the choice at `node`: its only child, else the largest logit
    /// among its children.
    fn pick_option(
        &mut self,
        row: &[f32],
        prog: Arc<Program>,
        step: usize,
        node: usize,
    ) -> Result<u32, StepError> {
        let Step::Pick(t) = &prog.steps[step] else {
            return Err(self.stuck("a choice context stands on another step"));
        };
        let kids = &t.nodes[node].kids;
        let several = kids.len() > 1;
        let picked = t.only_kid(node).or_else(|| {
            sampling::argmax_in(row, kids.iter().map(|&(id, _)| id))
                .and_then(|id| kids.iter().find(|&&(k, _)| k == id).map(|&(_, c)| (id, c)))
        });
        let Some((id, child)) = picked else {
            self.ctx = Ctx::Trie { prog, step, node };
            return Err(self.stuck("the row holds none of the option ids"));
        };
        self.picks.decisions += usize::from(several);
        if several && let Some(w) = &self.watch {
            relock(&w.0).push(Decision {
                kids: kids
                    .iter()
                    .map(|&(kid, below)| Kid {
                        id: kid,
                        logit: row.get(kid as usize).copied().unwrap_or(f32::NAN),
                        option: t.sole_option(below),
                    })
                    .collect(),
                picked: id,
            });
        }
        match t.nodes[child].leaf {
            Some(alt) => {
                if let Some(sub) = &t.subs[alt] {
                    self.frames.push(Frame {
                        prog: Arc::clone(sub),
                        at: 0,
                    });
                }
            }
            None => {
                self.ctx = Ctx::Trie {
                    prog: Arc::clone(&prog),
                    step,
                    node: child,
                };
            }
        }
        Ok(id)
    }

    /// One id of a free string that has `chars` characters: the largest
    /// logit among the class tokens that fit and the tokens that end it.
    fn pick_free(
        &mut self,
        row: &[f32],
        prog: Arc<Program>,
        step: usize,
        chars: usize,
    ) -> Result<u32, StepError> {
        let Step::Free(f) = &prog.steps[step] else {
            return Err(self.stuck("a string context stands on another step"));
        };
        let room = f.max.map_or(usize::MAX, |m| m.saturating_sub(chars));
        let in_class = if room > 0 {
            let n = row.len().min(f.class.n_vocab());
            sampling::argmax_in(
                row,
                (0..n)
                    .filter_map(|i| u32::try_from(i).ok())
                    .filter(|&id| f.class.fits(id, room)),
            )
        } else {
            None
        };
        let ending = sampling::argmax_in(row, f.end_ids.iter().copied());
        let id = match (in_class, ending) {
            (Some(a), Some(b)) => sampling::argmax_in(row, [a, b]),
            (a, b) => a.or(b),
        };
        let Some(id) = id else {
            self.ctx = Ctx::Free { prog, step, chars };
            return Err(self.stuck(if room == 0 {
                "a free string reached its length limit and no end token is allowed"
            } else {
                "the row holds no token a free string may take"
            }));
        };
        self.picks.free += 1;
        match f.end_ids.binary_search(&id) {
            Ok(p) => self.forced = f.ends[p].1.iter().rev().copied().collect(),
            Err(_) => {
                self.ctx = Ctx::Free {
                    prog: Arc::clone(&prog),
                    step,
                    chars: chars + f.class.len_of(id),
                };
            }
        }
        Ok(id)
    }
}
