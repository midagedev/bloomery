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
//!   stops.
//!
//! The pick at every decision is the largest logit among the allowed ids, ties
//! to the lowest id ([`sampling::argmax_in`]): the engine's argmax rule over
//! the allowed set. The walk reads no row at a forced id.

use std::sync::Arc;

use super::Refusal;
use super::vocab::{Class, Vocab};
use crate::engine::Tokenizer;
use crate::sampling;

/// How a free string is quoted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Style {
    /// A JSON string: its quotes are the surrounding text's, and its body takes
    /// no `"`, no `\` and no control character.
    Json,
    /// Raw text that ends at the text after it: the body takes no token that
    /// holds that text's first byte.
    Raw,
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
            Part::Text(t) => steps.push(Step::Forced(env.tok.encode(t))),
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
        Style::Json => {
            if first != b'"' {
                return Err(refuse(
                    "internal: a JSON string's closing text does not begin with a quote".to_owned(),
                ));
            }
            env.vocab.json_class()
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
    let ends: Vec<(u32, Vec<u32>)> = env
        .vocab
        .prefixes_of(tail)
        .into_iter()
        .map(|(id, len)| (id, env.tok.encode(&tail[len..])))
        .collect();
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

/// A request's constraint, built once and shared by the walks that follow it.
#[derive(Clone)]
pub(crate) struct Constraint {
    prog: Arc<Program>,
    /// The ids that close an open think span, forced first when the prompt
    /// leaves one open.
    close: Arc<[u32]>,
    stops: Arc<[u32]>,
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
        }
    }
}

/// A walk that cannot take a step.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StepError {
    #[error("constrained output: no token is allowed after {taken} generated: {why}")]
    NoAllowed { taken: usize, why: String },
}

#[derive(Debug)]
struct Frame {
    prog: Arc<Program>,
    at: usize,
}

#[derive(Debug)]
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

/// One request's position in its program.
#[derive(Debug)]
pub(crate) struct Walk {
    frames: Vec<Frame>,
    ctx: Ctx,
    /// Forced ids not yet taken, the next one last.
    forced: Vec<u32>,
    stops: Arc<[u32]>,
    taken: usize,
}

impl Walk {
    /// A walk at the start of `c`'s program; `close_think` forces the think
    /// span's close first (a prompt that left the span open).
    pub(crate) fn new(c: &Constraint, close_think: bool) -> Walk {
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
        }
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

    /// The next id: the forced one, or the largest logit of `row` among the
    /// ids the program allows here; the walk then stands past it.
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
        self.taken += 1;
        Ok(id)
    }

    fn stuck(&self, why: &str) -> StepError {
        StepError::NoAllowed {
            taken: self.taken,
            why: why.to_owned(),
        }
    }

    /// One id of the choice at `node`: the largest logit among its children.
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
        let picked = sampling::argmax_in(row, kids.iter().map(|&(id, _)| id))
            .and_then(|id| kids.iter().find(|&&(k, _)| k == id).map(|&(_, c)| (id, c)));
        let Some((id, child)) = picked else {
            self.ctx = Ctx::Trie { prog, step, node };
            return Err(self.stuck("the row holds none of the option ids"));
        };
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
