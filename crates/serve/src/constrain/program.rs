//! The segment program and the walk over it.
//!
//! A program is text first ([`Part`]): forced text, a choice among option
//! texts, a free string, a JSON number, a fork among part lists, a JSON array
//! or object of repeated items, a run of parts held once and repeated, and a
//! rule of the program's grammar.
//! [`compile`] turns it into ids ([`Program`]) with the serve [`Tokenizer`],
//! each forced text and each option canonically on its own (a string's opening
//! quote belongs to the text before it, its closing quote to the text after).
//! A [`Walk`] then answers, at each token, which ids the program allows and
//! picks among them: the largest logit, or the request's sampler over the row
//! with every other id masked out ([`Draw`]).
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
//!   forced text that can follow it: a token `",` ends a string and takes the
//!   `,` after it. The rest of that text is tokenised canonically from where
//!   the end token stopped. A number takes the tokens of a JSON number's
//!   characters while they still spell the front of one, and ends the same
//!   way once they spell a whole one.
//! - A fork ([`Part::Any`], [`Part::List`]) takes the text on each side of it
//!   into each of its options, so every option is tokenised whole. The walk
//!   then follows every option whose ids the tokens taken so far match (its
//!   heads), and a fork's options that begin alike share their ids until they
//!   part. An array or object repeats its item after `, ` until it takes its
//!   closing text.
//! - Once the program is spent the only allowed ids are the vocabulary's
//!   stops. A stop id inside the program would end the generation with the
//!   program unspent, so [`compile`] refuses any text that tokenises to one
//!   and the walk refuses to take one before the program is spent.
//!
//! A greedy pick is the largest logit among the allowed ids, ties to the lowest
//! id ([`sampling::argmax_in`]): the engine's argmax rule over the allowed set.
//! The walk reads no row at an id every head fixes alike: a forced id, or the
//! only child of a choice's node ([`Walk::fixed_next`]).
//!
//! A watched program ([`Opener`], a `tool_choice: auto` request) stands in
//! free text before and between its runs: every id allowed, the pick plain
//! greedy or the request's sampler over the whole row, and the ids taken,
//! by a pick or by the engine's pass, watched for the chat format's call
//! opener ([`Walk::saw`]) — the one special token the vocabulary holds for
//! it, or its characters in the text of ordinary ones, across pieces, the
//! text after it in the same piece the program's first. Where the model
//! writes it, the program a forced call compiles runs to the call's close
//! and the walk returns to free text.

use std::cell::Cell;
use std::sync::{Arc, Mutex};

use super::Refusal;
use super::vocab::{Class, Vocab};
use crate::api::relock;
use crate::engine::{Sampler, Tokenizer};
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

    /// The style of a JSON string inside an object or array written in this
    /// style: a raw value is markup's, so the strings in one it holds are in
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
    /// A JSON number, an integer alone when `integer`; the forced text after
    /// it ends it.
    Num { integer: bool },
    /// One of several part lists, each followed by the parts after this one.
    Any(Vec<Vec<Part>>),
    /// A JSON array: no item or `item` again and again apart by `, `.
    List { item: Vec<Part> },
    /// The rule of the program's grammar at this index ([`json_object`]).
    Rule(usize),
    /// After an item of a grammar's array or object: `close`, or `more` (the
    /// next item) and back to this part.
    Rep { close: Vec<Part>, more: Vec<Part> },
    /// A run of parts the program holds once and stands wherever it is
    /// repeated: the object rule's alternatives repeat a property's parts,
    /// and a schema's build costs its size, not its alternatives' product.
    Run(std::sync::Arc<[Part]>),
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
    Forced(Forced),
    Pick(Trie),
    Free(FreeStep),
    Num {
        integer: bool,
    },
    /// The walk goes on in each program, then after this step.
    Fork(Vec<Arc<Program>>),
    /// After an item: `close` and on past this step, or `more` (the next
    /// item) and back to this step.
    Rep {
        close: Arc<Program>,
        more: Arc<Program>,
    },
    /// The program, then the step after this one.
    Sub(Arc<Program>),
    /// The grammar's rule at this index, then the step after this one.
    Rule(usize),
}

#[derive(Debug)]
struct Forced {
    ids: Vec<u32>,
    text: String,
    /// The first special token's byte offset and id: an end piece cannot
    /// consume its text as ordinary tokens.
    first_special: Option<(usize, u32)>,
    /// For each token-reachable character boundary inside the text, by byte
    /// offset ascending, the ids after it. No end token reaches beyond the
    /// longest vocabulary piece. Empty for a text no end token reaches.
    rests: Vec<(usize, Vec<u32>)>,
}

impl Forced {
    fn rest(&self, at: usize) -> Option<&[u32]> {
        let p = self.rests.binary_search_by_key(&at, |r| r.0).ok()?;
        Some(&self.rests[p].1)
    }
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
}

impl FreeStep {
    /// The characters a string of `chars` has left, `usize::MAX` with no
    /// limit.
    fn room(&self, chars: usize) -> usize {
        self.max.map_or(usize::MAX, |m| m.saturating_sub(chars))
    }
}

/// The bytes of text one request's program may tokenise. An `anyOf` repeats
/// the text on each side of it in every option, and the text after a string
/// or a number is tokenised again from each boundary an end token can reach,
/// so this, not the request's size, bounds the compile's work and its ids.
const COMPILE_BYTES: usize = 1 << 24;

/// What [`compile`] tokenises with, and the bytes it has left to tokenise.
pub(crate) struct Env<'a> {
    pub tok: &'a dyn Tokenizer,
    pub vocab: &'a Vocab,
    bytes_left: Cell<usize>,
}

impl<'a> Env<'a> {
    pub(crate) fn new(tok: &'a dyn Tokenizer, vocab: &'a Vocab) -> Self {
        Self {
            tok,
            vocab,
            bytes_left: Cell::new(COMPILE_BYTES),
        }
    }

    /// The ids of `text`, its bytes charged to the compile's budget.
    fn encode(&self, text: &str) -> Result<Vec<u32>, Refusal> {
        let Some(left) = self.bytes_left.get().checked_sub(text.len()) else {
            return Err(outside(format!(
                "the schema's fixed text is too long: its program tokenises more than \
                 {COMPILE_BYTES} bytes (an anyOf repeats the text on each side of it in every \
                 option)"
            )));
        };
        self.bytes_left.set(left);
        Ok(self.tok.encode(text))
    }
}

fn refuse(message: String) -> Refusal {
    Refusal::invalid(message)
}

/// The refusal of a schema that is valid but that this vocabulary or the
/// compile's budget cannot write ([`Refusal::outside_subset`]).
fn outside(message: String) -> Refusal {
    Refusal::outside_subset(message)
}

/// Refuses `ids`, the tokens of `text`, when one is a stop id: the generation
/// would end there with the program unspent.
fn no_stop(ids: &[u32], text: &str, env: &Env<'_>) -> Result<(), Refusal> {
    match ids.iter().find(|id| env.vocab.stops().contains(id)) {
        Some(id) => Err(outside(format!(
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
    compile_in(parts, env, false)
}

/// `parts` as a program; `reached` says whether an end token can reach its
/// first step (an option of a fork, what follows a string or a number, or a
/// watched call's program, which the text after its opener enters).
pub(super) fn compile_in(parts: &[Part], env: &Env<'_>, reached: bool) -> Result<Program, Refusal> {
    let parts = merged(parts);
    let mut steps = Vec::new();
    let mut reach = reached;
    let mut pre = String::new();
    let mut i = 0;
    while i < parts.len() {
        let part = &parts[i];
        i += 1;
        match part {
            Part::Text(t) if matches!(parts.get(i), Some(Part::Any(_) | Part::List { .. })) => {
                pre.clone_from(t);
            }
            Part::Text(t) => {
                steps.push(Step::Forced(forced(t, reach, env)?));
                reach = false;
            }
            Part::Pick(alts) => {
                if alts.iter().any(|a| !a.then.is_empty()) && i != parts.len() {
                    return Err(refuse(
                        "internal: a choice with continuations is not last in its list".to_owned(),
                    ));
                }
                steps.push(Step::Pick(trie(alts, env)?));
                reach = false;
            }
            Part::Free { max, style } => {
                steps.push(Step::Free(free_step(*max, *style, parts.get(i), env)?));
                reach = true;
            }
            Part::Num { integer } => {
                steps.push(Step::Num { integer: *integer });
                reach = true;
            }
            Part::Rule(r) => {
                steps.push(Step::Rule(*r));
                reach = true;
            }
            Part::Rep { close, more } => {
                steps.push(Step::Rep {
                    close: Arc::new(compile_in(close, env, true)?),
                    more: Arc::new(compile_in(more, env, true)?),
                });
                reach = true;
            }
            Part::Run(parts) => {
                steps.push(Step::Sub(Arc::new(compile_in(parts, env, reach)?)));
                reach = true;
            }
            Part::Any(_) | Part::List { .. } => {
                let post = match parts.get(i) {
                    Some(Part::Text(t)) => {
                        i += 1;
                        t.as_str()
                    }
                    _ => "",
                };
                steps.push(fork(part, &std::mem::take(&mut pre), post, env)?);
                reach = true;
            }
        }
    }
    Ok(Program { steps })
}

/// A forced text's step; `reached` adds the rest after each character
/// boundary an end token can reach inside it, bounded by the longest piece.
fn forced(text: &str, reached: bool, env: &Env<'_>) -> Result<Forced, Refusal> {
    let ids = env.encode(text)?;
    no_stop(&ids, text, env)?;
    let first_special = ids
        .iter()
        .position(|&id| env.tok.is_special(id))
        .map(|i| (env.tok.decode(&ids[..i]).len(), ids[i]));
    let mut rests = Vec::new();
    if reached {
        let max_bytes = env.vocab.longest_piece();
        for (at, _) in text
            .char_indices()
            .skip(1)
            .take_while(|&(at, _)| at <= max_bytes)
        {
            let rest = env.encode(&text[at..])?;
            no_stop(&rest, text, env)?;
            rests.push((at, rest));
        }
    }
    Ok(Forced {
        ids,
        text: text.to_owned(),
        first_special,
        rests,
    })
}

/// A program of one forced text that an end token can reach.
fn text_program(text: &str, env: &Env<'_>) -> Result<Arc<Program>, Refusal> {
    Ok(Arc::new(Program {
        steps: vec![Step::Forced(forced(text, true, env)?)],
    }))
}

/// A fork's step, with `pre` and `post`, the texts on each side of it, inside
/// each option. A list's item is compiled once: its leading and trailing texts
/// go into the texts around it (`[` and the lead, the trail and `, ` and the
/// lead, the trail and `]`), so each of those is tokenised whole.
fn fork(part: &Part, pre: &str, post: &str, env: &Env<'_>) -> Result<Step, Refusal> {
    match part {
        Part::Any(alts) => {
            let progs = alts
                .iter()
                .map(|a| {
                    let mut parts = vec![Part::Text(pre.to_owned())];
                    parts.extend(a.iter().cloned());
                    parts.push(Part::Text(post.to_owned()));
                    compile_in(&parts, env, true).map(Arc::new)
                })
                .collect::<Result<_, Refusal>>()?;
            Ok(Step::Fork(progs))
        }
        Part::List { item } => {
            let mut body = merged(item);
            let lead = match body.first() {
                Some(Part::Text(t)) => {
                    let t = t.clone();
                    body.remove(0);
                    t
                }
                _ => String::new(),
            };
            let trail = match body.last() {
                Some(Part::Text(t)) => {
                    let t = t.clone();
                    body.pop();
                    t
                }
                _ => String::new(),
            };
            let body = if body.is_empty() {
                None
            } else {
                Some(Arc::new(compile_in(&body, env, true)?))
            };
            let item_after = |text: String| -> Result<Vec<Step>, Refusal> {
                let mut steps = vec![Step::Forced(forced(&text, true, env)?)];
                steps.extend(body.clone().map(Step::Sub));
                Ok(steps)
            };
            let rep = Step::Rep {
                close: text_program(&format!("{trail}]{post}"), env)?,
                more: Arc::new(Program {
                    steps: item_after(format!("{trail}, {lead}"))?,
                }),
            };
            let mut first = item_after(format!("{pre}[{lead}"))?;
            first.push(rep);
            Ok(Step::Fork(vec![
                text_program(&format!("{pre}[]{post}"), env)?,
                Arc::new(Program { steps: first }),
            ]))
        }
        _ => Err(refuse("internal: a fork of no options".to_owned())),
    }
}

fn trie(alts: &[Alt], env: &Env<'_>) -> Result<Trie, Refusal> {
    if alts.is_empty() {
        return Err(refuse("a choice with no option".to_owned()));
    }
    let mut nodes = vec![Node::default()];
    let mut subs = Vec::with_capacity(alts.len());
    let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(alts.len());
    for (k, alt) in alts.iter().enumerate() {
        let ids = env.encode(&alt.text)?;
        if ids.is_empty() {
            return Err(outside(format!(
                "the option {:?} tokenises to no token",
                alt.text
            )));
        }
        no_stop(&ids, &alt.text, env)?;
        let clash = |other: usize| {
            outside(format!(
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

/// A free string's step. `next` is the part after it: a raw string's class
/// depends on the first byte of that text, which must be there; a JSON
/// string's text after it, when it is in the same list, opens with its
/// closing quote.
fn free_step(
    max: Option<usize>,
    style: Style,
    next: Option<&Part>,
    env: &Env<'_>,
) -> Result<FreeStep, Refusal> {
    let tail = match next {
        Some(Part::Text(t)) => Some(t.as_str()),
        _ => None,
    };
    let class = match style {
        Style::Json | Style::JsonInMarkup => {
            if tail.is_some_and(|t| !t.starts_with('"')) {
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
            let Some(tail) = tail else {
                return Err(refuse(
                    "internal: a raw string is not followed by forced text".to_owned(),
                ));
            };
            let first = tail.as_bytes()[0];
            if !first.is_ascii() {
                return Err(outside(format!(
                    "a raw value ends at {tail:?}, whose first character is not ASCII"
                )));
            }
            env.vocab.raw_class(first)
        }
    };
    if let Some(tail) = tail
        && !env.vocab.prefixes_of(tail)
    {
        return Err(outside(format!(
            "no token of the vocabulary is a prefix of {tail:?}, so a free string before it \
             could never end"
        )));
    }
    Ok(FreeStep { class, max })
}

/// Any JSON object of at least one member, its strings in `style`: the
/// program and the rules it reaches. An empty reply object answers no request,
/// and a model that would open a pretty-printed one (`{` and a newline) is
/// left `{}` or `{"` to pick from, so the empty one is the nested values'
/// alone. Its text is compact with one space after `:` and `,`, as a
/// schema's is. The rules are any JSON value after `": `, after `, ` and after
/// nothing, so the text before a value is tokenised with the value's first
/// characters (a space joins the quote, brace or word after it).
pub(crate) fn json_object(
    style: Style,
    env: &Env<'_>,
) -> Result<(Program, Vec<Arc<Program>>), Refusal> {
    const PRES: [&str; 3] = ["\": ", ", ", ""];
    let (colon, comma, bare) = (0, 1, 2);
    let text = |t: &str| Part::Text(t.to_owned());
    let free = Part::Free { max: None, style };
    let member = |open: &str| vec![text(open), free.clone(), Part::Rule(colon)];
    let mut first = member("{\"");
    first.push(Part::Rep {
        close: vec![text("}")],
        more: member(", \""),
    });
    let object = Part::Any(vec![vec![text("{}")], first.clone()]);
    let array = Part::Any(vec![
        vec![text("[]")],
        vec![
            text("["),
            Part::Rule(bare),
            Part::Rep {
                close: vec![text("]")],
                more: vec![Part::Rule(comma)],
            },
        ],
    ]);
    let rules = PRES
        .iter()
        .map(|pre| {
            let value = Part::Any(vec![
                vec![text(&format!("{pre}\"")), free.clone(), text("\"")],
                vec![text(pre), Part::Num { integer: false }],
                vec![text(pre), object.clone()],
                vec![text(pre), array.clone()],
                vec![text(&format!("{pre}true"))],
                vec![text(&format!("{pre}false"))],
                vec![text(&format!("{pre}null"))],
            ]);
            compile_in(&[value], env, true).map(Arc::new)
        })
        .collect::<Result<_, Refusal>>()?;
    Ok((compile(&first, env)?, rules))
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

/// The text that opens a call in a chat format's markup, and the program a
/// walk follows from it: what a `tool_choice: auto` request's free text is
/// watched for. `special` is the one id that writes `text` when the
/// vocabulary holds it as a special token: the opener is then markup only
/// where that id wrote it, as the format's own scan reads it, and ordinary
/// ids spelling the same characters stay free text. Without one, the opener
/// is watched for on the text of the ids taken, ending a piece of them.
pub(crate) struct Opener {
    pub text: &'static str,
    pub special: Option<u32>,
    pub prog: Arc<Program>,
}

/// A request's constraint, built once and shared by the walks that follow it.
#[derive(Clone)]
pub(crate) struct Constraint {
    prog: Arc<Program>,
    /// The programs a [`Step::Rule`] names, by index.
    rules: Arc<[Arc<Program>]>,
    /// The ids that close an open think span, forced first when the prompt
    /// leaves one open and the program does not follow the reasoning.
    close: Arc<[u32]>,
    /// The program starts where the model closes an open think span itself
    /// (a forced tool call, a watched one), not at once with the span
    /// force-closed (a JSON document).
    follows_reasoning: bool,
    /// The call opener whose writing starts the program where the model
    /// chooses to open one (`tool_choice: auto`): `None` on a program that
    /// starts at once.
    waiting_for_call: Option<Arc<Opener>>,
    vocab: Arc<Vocab>,
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
    pub(crate) fn new(
        prog: Program,
        rules: Vec<Arc<Program>>,
        close: Vec<u32>,
        vocab: Arc<Vocab>,
    ) -> Constraint {
        Constraint {
            prog: Arc::new(prog),
            rules: rules.into(),
            close: close.into(),
            follows_reasoning: false,
            waiting_for_call: None,
            vocab,
            watch: None,
        }
    }

    /// A constraint that waits for the model to open a call itself
    /// (`tool_choice: auto`): free text until the format's opener, the
    /// program `opener` holds from there to the call's close, free text
    /// again past it, each further opener starting the program again. The
    /// model's own close of an open think span comes first, as a forced
    /// call's does.
    pub(crate) fn waiting_for(opener: Opener, close: Vec<u32>, vocab: Arc<Vocab>) -> Constraint {
        Constraint {
            prog: Arc::clone(&opener.prog),
            rules: Vec::new().into(),
            close: close.into(),
            follows_reasoning: true,
            waiting_for_call: Some(Arc::new(opener)),
            vocab,
            watch: None,
        }
    }

    /// This constraint starting after the model's own close of an open think
    /// span ([`Constraint::follows_reasoning`]).
    pub(crate) fn following_reasoning(mut self) -> Constraint {
        self.follows_reasoning = true;
        self
    }

    /// Whether a prompt that left the think span open lets the model reason
    /// to its own close before the program starts: a forced tool call and a
    /// watched one do, a JSON document closes the span first
    /// ([`Walk::new`]).
    pub(crate) fn follows_reasoning(&self) -> bool {
        self.follows_reasoning
    }

    /// Whether the model's own call opener starts this constraint's program
    /// (a `tool_choice: auto` request).
    pub(crate) fn waits_for_call(&self) -> bool {
        self.waiting_for_call.is_some()
    }

    /// Whether the reply is this program's alone from the think span's own
    /// close, so a call opener inside the span is reasoning
    /// ([`crate::dsml::ChatParser::following_reasoning`]): a forced call
    /// holds the span open until then; a watched one leaves the parser's own
    /// end rule in place, which a call opener may end (Qwen3-Coder's).
    pub(crate) fn forced(&self) -> bool {
        self.follows_reasoning && self.waiting_for_call.is_none()
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

/// How a walk picks among the ids its program allows at a decision.
pub(crate) enum Draw<'a> {
    /// The largest logit, ties to the lowest id.
    Greedy,
    /// The request's sampler, drawing from a copy of the row (`masked`, kept
    /// for its capacity) that holds minus infinity at every id the program
    /// does not allow, with the request's `history` for its penalties.
    Sampled {
        sampler: &'a mut Sampler,
        history: &'a [u32],
        masked: &'a mut Vec<f32>,
    },
}

impl Draw<'_> {
    fn pick(&mut self, row: &[f32], ids: impl Iterator<Item = u32>) -> Option<u32> {
        match self {
            Draw::Greedy => sampling::argmax_in(row, ids),
            Draw::Sampled {
                sampler,
                history,
                masked,
            } => {
                #[cfg(test)]
                MASKS.with(|n| n.set(n.get() + 1));
                masked.clear();
                masked.resize(row.len(), f32::NEG_INFINITY);
                let mut any = false;
                for id in ids {
                    if let Some(&l) = row.get(id as usize) {
                        masked[id as usize] = l;
                        any = true;
                    }
                }
                any.then(|| sampler(masked, history))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Frame {
    prog: Arc<Program>,
    at: usize,
}

impl Frame {
    fn of(prog: &Arc<Program>) -> Frame {
        Frame {
            prog: Arc::clone(prog),
            at: 0,
        }
    }
}

/// Where a JSON number stands, by the characters it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumAt {
    Start,
    Minus,
    /// A leading `0`, which no digit follows.
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl NumAt {
    fn next(self, c: u8, integer: bool) -> Option<NumAt> {
        use NumAt::{Dot, E, ESign, Exp, Frac, Int, Minus, Start, Zero};
        Some(match (self, c) {
            (Start, b'-') => Minus,
            (Start | Minus, b'0') => Zero,
            (Start | Minus, b'1'..=b'9') | (Int, b'0'..=b'9') => Int,
            (Zero | Int, b'.') if !integer => Dot,
            (Dot | Frac, b'0'..=b'9') => Frac,
            (Zero | Int | Frac, b'e' | b'E') if !integer => E,
            (E, b'+' | b'-') => ESign,
            (E | ESign | Exp, b'0'..=b'9') => Exp,
            _ => return None,
        })
    }

    /// Where the number stands after a token's text, `None` when the text
    /// is empty or leaves no number.
    fn after(self, text: &str, integer: bool) -> Option<NumAt> {
        if text.is_empty() {
            return None;
        }
        text.bytes().try_fold(self, |at, c| at.next(c, integer))
    }

    /// Whether the characters so far are a whole number.
    fn whole(self) -> bool {
        matches!(self, NumAt::Zero | NumAt::Int | NumAt::Frac | NumAt::Exp)
    }
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
        /// The ids that end it, ascending, read once it is picked from.
        ends: Option<Arc<[u32]>>,
    },
    Num {
        prog: Arc<Program>,
        step: usize,
        at: NumAt,
        /// The ids that end it, ascending, read once it is whole.
        ends: Option<Arc<[u32]>>,
    },
    /// The program is spent.
    Done,
}

/// One position in the program the ids taken so far lead to.
#[derive(Clone, Debug)]
struct Head {
    frames: Vec<Frame>,
    ctx: Ctx,
    /// Forced ids not yet taken, the next one last.
    forced: Vec<u32>,
}

impl Head {
    fn settled(&self) -> bool {
        !self.forced.is_empty() || !matches!(self.ctx, Ctx::Idle)
    }

    /// The id this head alone allows and needs no row for.
    fn fixed(&self) -> Option<u32> {
        if let Some(&id) = self.forced.last() {
            return Some(id);
        }
        match &self.ctx {
            Ctx::Trie { prog, step, node } => match &prog.steps[*step] {
                Step::Pick(t) => t.only_kid(*node).map(|(id, _)| id),
                _ => None,
            },
            _ => None,
        }
    }

    fn done(&self) -> bool {
        self.forced.is_empty() && matches!(self.ctx, Ctx::Done)
    }

    /// The head past its string or number, nothing forced.
    fn after(&self) -> Head {
        Head {
            frames: self.frames.clone(),
            ctx: Ctx::Idle,
            forced: Vec::new(),
        }
    }

    /// The head's place inside its step, beside its frames: the context's
    /// program and step are always the last frame's, one step back.
    fn mark(&self) -> (u8, usize) {
        match &self.ctx {
            Ctx::Idle => (0, 0),
            Ctx::Done => (1, 0),
            Ctx::Trie { node, .. } => (2, *node),
            Ctx::Free { chars, .. } => (3, *chars),
            Ctx::Num { at, .. } => (4, *at as usize),
        }
    }

    fn same(&self, o: &Head) -> bool {
        self.mark() == o.mark()
            && self.forced == o.forced
            && self.frames.len() == o.frames.len()
            && self
                .frames
                .iter()
                .zip(&o.frames)
                .all(|(a, b)| a.at == b.at && Arc::ptr_eq(&a.prog, &b.prog))
    }
}

/// Adds `h` unless a head in the same place is there.
fn push_new(out: &mut Vec<Head>, h: Head) {
    if !out.iter().any(|o| o.same(&h)) {
        out.push(h);
    }
}

/// How many ids a walk picked from a row, by kind (the ids the program fixes
/// and the stop pick are in neither).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Picks {
    /// Among the children of a choice's node, which has more than one, or
    /// among several heads' ids.
    pub decisions: usize,
    /// Inside a free string or a number, its end token included.
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

#[cfg(test)]
thread_local! {
    static MASKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The masked copies of a row [`Draw::Sampled`] has built on this thread.
#[cfg(test)]
pub(crate) fn masks() -> usize {
    MASKS.with(std::cell::Cell::get)
}

/// One request's position in its program.
#[derive(Clone)]
pub(crate) struct Walk {
    /// Every place the ids taken so far lead to: one, but where a fork's
    /// options have begun alike. Empty while the walk waits in free text for
    /// its call opener.
    heads: Vec<Head>,
    rules: Arc<[Arc<Program>]>,
    vocab: Arc<Vocab>,
    taken: usize,
    picks: Picks,
    watch: Option<Watch>,
    /// The call opener this walk watches for in free text (`tool_choice:
    /// auto`): `None` on a walk whose program starts without one.
    waiting: Option<Arc<Opener>>,
    /// The tail of the free text, as far as an opener could still complete
    /// in it: at most the opener's length.
    tail: Vec<u8>,
}

impl std::fmt::Debug for Walk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Walk")
            .field("heads", &self.heads.len())
            .field("taken", &self.taken)
            .finish_non_exhaustive()
    }
}

impl Walk {
    /// A walk at the start of `c`'s program; `close_think` forces the think
    /// span's close first (a prompt that left the span open). A walk that
    /// waits for a call starts in free text instead: its constraint follows
    /// the reasoning, so no span is ever force-closed before it.
    pub(crate) fn new(c: &Constraint, close_think: bool) -> Walk {
        if let Some(w) = &c.watch {
            relock(&w.0).clear();
        }
        let waiting = c.waiting_for_call.clone();
        assert!(
            waiting.is_none() || !close_think,
            "a watched call waits for the model's own think close, not a forced one"
        );
        Walk {
            heads: match &waiting {
                Some(_) => Vec::new(),
                None => vec![Head {
                    frames: vec![Frame::of(&c.prog)],
                    ctx: Ctx::Idle,
                    forced: if close_think {
                        c.close.iter().rev().copied().collect()
                    } else {
                        Vec::new()
                    },
                }],
            },
            rules: Arc::clone(&c.rules),
            vocab: Arc::clone(&c.vocab),
            taken: 0,
            picks: Picks::default(),
            watch: c.watch.clone(),
            waiting,
            tail: Vec::new(),
        }
    }

    /// Whether the walk stands in free text watched for its call opener
    /// (`tool_choice: auto`, before the first call and between calls): every
    /// id allowed, none forced.
    pub(crate) fn waiting(&self) -> bool {
        self.waiting.is_some() && self.heads.is_empty()
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

    fn stops(&self) -> &[u32] {
        self.vocab.stops()
    }

    /// Moves every head to its next decision or forced id.
    /// A whole number the program ends with may also end the answer: a
    /// spent head joins it, so the stop is allowed beside its next digit.
    /// A watched walk whose call closed returns to free text.
    fn settle(&mut self) {
        if !self.heads.iter().all(Head::settled) {
            let mut out = Vec::with_capacity(self.heads.len());
            for h in std::mem::take(&mut self.heads) {
                self.settle_into(h, &mut out);
            }
            self.heads = out;
        }
        if !self.heads.iter().any(Head::done)
            && self.heads.iter().any(|h| {
                h.forced.is_empty()
                    && matches!(&h.ctx, Ctx::Num { at, .. } if at.whole())
                    && h.frames.iter().all(|f| f.at >= f.prog.steps.len())
            })
        {
            self.heads.push(Head {
                frames: Vec::new(),
                ctx: Ctx::Done,
                forced: Vec::new(),
            });
        }
        if self.waiting.is_some() && !self.heads.is_empty() && self.heads.iter().all(Head::done) {
            // The call closed: free text again, watched for the next opener.
            self.heads.clear();
            self.tail.clear();
        }
    }

    /// The heads the step at `h`'s place leads into when it holds no ids of
    /// its own: a sub-program or a rule, each option of a fork, an array's
    /// close (past the step) and its next item (back to the step after it).
    /// `None` for any other step.
    fn enter(&self, h: &Head) -> Option<Vec<Head>> {
        let frame = h.frames.last()?;
        let into = |prog: &Arc<Program>, back: bool| {
            let mut k = h.clone();
            if let Some(f) = k.frames.last_mut()
                && !back
            {
                f.at += 1;
            }
            k.frames.push(Frame::of(prog));
            k
        };
        Some(match frame.prog.steps.get(frame.at)? {
            Step::Sub(p) => vec![into(p, false)],
            Step::Rule(r) => vec![into(&self.rules[*r], false)],
            Step::Fork(alts) => alts.iter().map(|a| into(a, false)).collect(),
            Step::Rep { close, more } => vec![into(close, false), into(more, true)],
            Step::Forced(_) | Step::Pick(_) | Step::Free(_) | Step::Num { .. } => return None,
        })
    }

    fn settle_into(&self, mut h: Head, out: &mut Vec<Head>) {
        loop {
            if h.settled() {
                push_new(out, h);
                return;
            }
            let Some(frame) = h.frames.last() else {
                h.ctx = Ctx::Done;
                push_new(out, h);
                return;
            };
            if frame.at >= frame.prog.steps.len() {
                h.frames.pop();
                continue;
            }
            if let Some(next) = self.enter(&h) {
                for k in next {
                    self.settle_into(k, out);
                }
                return;
            }
            let Some(frame) = h.frames.last_mut() else {
                return;
            };
            let step = frame.at;
            frame.at += 1;
            let prog = Arc::clone(&frame.prog);
            h.ctx = match &prog.steps[step] {
                Step::Forced(f) => {
                    h.forced = f.ids.iter().rev().copied().collect();
                    Ctx::Idle
                }
                Step::Pick(_) => Ctx::Trie {
                    prog,
                    step,
                    node: 0,
                },
                Step::Free(_) => Ctx::Free {
                    prog,
                    step,
                    chars: 0,
                    ends: None,
                },
                _ => Ctx::Num {
                    prog,
                    step,
                    at: NumAt::Start,
                    ends: None,
                },
            };
        }
    }

    /// The heads `h` (past a string or a number, nothing forced) leads to once
    /// it takes `text`, into `out`: through forced texts, forks and rules,
    /// ending inside a forced text with the rest of it forced. The ending
    /// `id` may reach a special token only when it is that token itself.
    fn eat(&self, mut h: Head, mut text: &str, id: u32, out: &mut Vec<Head>) {
        loop {
            if text.is_empty() {
                push_new(out, h);
                return;
            }
            let Some(frame) = h.frames.last() else {
                return;
            };
            if frame.at >= frame.prog.steps.len() {
                h.frames.pop();
                continue;
            }
            if let Some(next) = self.enter(&h) {
                for k in next {
                    self.eat(k, text, id, out);
                }
                return;
            }
            let Some(frame) = h.frames.last_mut() else {
                return;
            };
            let prog = Arc::clone(&frame.prog);
            let Step::Forced(f) = &prog.steps[frame.at] else {
                return;
            };
            if f.first_special
                .is_some_and(|(at, tag)| at < text.len() && (at != 0 || id != tag))
            {
                return;
            }
            frame.at += 1;
            if let Some(rest) = text.strip_prefix(f.text.as_str()) {
                text = rest;
            } else {
                if f.text.starts_with(text)
                    && let Some(rest) = f.rest(text.len())
                {
                    h.forced = rest.iter().rev().copied().collect();
                    push_new(out, h);
                }
                return;
            }
        }
    }

    /// Marks the first byte of each forced text `h` (past a string or a
    /// number) can go on with.
    fn firsts(&self, mut h: Head, marks: &mut [bool; 256]) {
        loop {
            let Some(frame) = h.frames.last() else {
                return;
            };
            if frame.at >= frame.prog.steps.len() {
                h.frames.pop();
                continue;
            }
            if let Some(next) = self.enter(&h) {
                for k in next {
                    self.firsts(k, marks);
                }
                return;
            }
            if let Step::Forced(f) = &frame.prog.steps[frame.at]
                && let Some(&b) = f.text.as_bytes().first()
            {
                marks[usize::from(b)] = true;
            }
            return;
        }
    }

    /// The ids that end a string or a number whose head is `h`, ascending:
    /// those whose text the text that can follow takes.
    fn ends_of(&self, h: &Head) -> Arc<[u32]> {
        let after = h.after();
        let mut marks = [false; 256];
        self.firsts(after.clone(), &mut marks);
        let mut ids = Vec::new();
        let mut out = Vec::new();
        for b in (0..=u8::MAX).filter(|&b| marks[usize::from(b)]) {
            for &id in self.vocab.starting_with(b) {
                if self.stops().contains(&id) {
                    continue;
                }
                out.clear();
                self.eat(after.clone(), self.vocab.piece(id), id, &mut out);
                if !out.is_empty() {
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
        ids.into()
    }

    /// Reads the end ids of each string, and of each whole number, that a
    /// pick may end.
    fn prime(&mut self) {
        for i in 0..self.heads.len() {
            let h = &self.heads[i];
            let wants = h.forced.is_empty()
                && match &h.ctx {
                    Ctx::Free { ends, .. } => ends.is_none(),
                    Ctx::Num { ends, at, .. } => ends.is_none() && at.whole(),
                    _ => false,
                };
            if !wants {
                continue;
            }
            let read = self.ends_of(h);
            if let Ctx::Free { ends, .. } | Ctx::Num { ends, .. } = &mut self.heads[i].ctx {
                *ends = Some(read);
            }
        }
    }

    /// The ids `h` allows, of a row of `n`.
    fn allowed<'a>(&'a self, h: &'a Head, n: usize) -> Box<dyn Iterator<Item = u32> + 'a> {
        if let Some(&id) = h.forced.last() {
            return Box::new(std::iter::once(id));
        }
        let ends = |e: &'a Option<Arc<[u32]>>| e.iter().flat_map(|e| e.iter().copied());
        match &h.ctx {
            Ctx::Done => Box::new(self.stops().iter().copied()),
            Ctx::Trie { prog, step, node } => match &prog.steps[*step] {
                Step::Pick(t) => Box::new(t.nodes[*node].kids.iter().map(|&(id, _)| id)),
                _ => Box::new(std::iter::empty()),
            },
            Ctx::Free {
                prog,
                step,
                chars,
                ends: e,
            } => {
                let Step::Free(f) = &prog.steps[*step] else {
                    return Box::new(std::iter::empty());
                };
                let room = f.room(*chars);
                let n = if room > 0 {
                    n.min(f.class.n_vocab())
                } else {
                    0
                };
                let class = (0..n)
                    .filter_map(|i| u32::try_from(i).ok())
                    .filter(move |&id| f.class.fits(id, room));
                Box::new(class.chain(ends(e)))
            }
            Ctx::Num {
                prog,
                step,
                at,
                ends: e,
            } => {
                let Step::Num { integer } = prog.steps[*step] else {
                    return Box::new(std::iter::empty());
                };
                let at = *at;
                let more = self
                    .vocab
                    .numeric()
                    .iter()
                    .copied()
                    .filter(move |&id| at.after(self.vocab.piece(id), integer).is_some());
                Box::new(more.chain(ends(e).filter(move |_| at.whole())))
            }
            Ctx::Idle => Box::new(std::iter::empty()),
        }
    }

    /// The heads `h` leads to once it takes `id`, into `out`.
    fn take_into(&self, mut h: Head, id: u32, out: &mut Vec<Head>) {
        if let Some(&f) = h.forced.last() {
            if f == id {
                h.forced.pop();
                push_new(out, h);
            }
            return;
        }
        let ends =
            |e: &Option<Arc<[u32]>>| e.as_ref().is_some_and(|e| e.binary_search(&id).is_ok());
        match std::mem::replace(&mut h.ctx, Ctx::Idle) {
            Ctx::Done => {
                if self.stops().contains(&id) {
                    h.ctx = Ctx::Done;
                    push_new(out, h);
                }
            }
            Ctx::Trie { prog, step, node } => {
                let Step::Pick(t) = &prog.steps[step] else {
                    return;
                };
                let kids = &t.nodes[node].kids;
                let Ok(p) = kids.binary_search_by_key(&id, |&(k, _)| k) else {
                    return;
                };
                let child = kids[p].1;
                match t.nodes[child].leaf {
                    Some(alt) => {
                        if let Some(sub) = &t.subs[alt] {
                            h.frames.push(Frame::of(sub));
                        }
                    }
                    None => {
                        h.ctx = Ctx::Trie {
                            prog: Arc::clone(&prog),
                            step,
                            node: child,
                        };
                    }
                }
                push_new(out, h);
            }
            Ctx::Free {
                prog,
                step,
                chars,
                ends: e,
            } => {
                let Step::Free(f) = &prog.steps[step] else {
                    return;
                };
                if ends(&e) {
                    self.eat(h, self.vocab.piece(id), id, out);
                } else if f.class.fits(id, f.room(chars)) {
                    h.ctx = Ctx::Free {
                        prog: Arc::clone(&prog),
                        step,
                        chars: chars + f.class.len_of(id),
                        ends: e,
                    };
                    push_new(out, h);
                }
            }
            Ctx::Num {
                prog,
                step,
                at,
                ends: e,
            } => {
                let Step::Num { integer } = prog.steps[step] else {
                    return;
                };
                let piece = self.vocab.piece(id);
                if at.whole() && ends(&e) {
                    self.eat(h, piece, id, out);
                } else if let Some(at) = at.after(piece, integer) {
                    h.ctx = Ctx::Num {
                        prog,
                        step,
                        at,
                        ends: e,
                    };
                    push_new(out, h);
                }
            }
            Ctx::Idle => {}
        }
    }

    /// Whether the next id is the program's alone to give: one every head
    /// fixes alike, a forced id or the only child of a choice's node.
    /// [`Walk::choose`] reads no row for it.
    pub(crate) fn fixed_next(&mut self) -> bool {
        self.settle();
        self.fixed_id().is_some()
    }

    fn fixed_id(&self) -> Option<u32> {
        let (first, rest) = self.heads.split_first()?;
        let id = first.fixed()?;
        rest.iter().all(|h| h.fixed() == Some(id)).then_some(id)
    }

    /// Whether the program is spent: nothing is left but the stop.
    pub(crate) fn spent(&mut self) -> bool {
        self.settle();
        !self.heads.is_empty() && self.heads.iter().all(Head::done)
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

    /// The next id: the fixed one, or the largest logit of `row` among the
    /// ids the program allows here; the walk then stands past it. A stop id
    /// before the program is spent is an error.
    pub(crate) fn choose(&mut self, row: &[f32]) -> Result<u32, StepError> {
        self.choose_by(row, &mut Draw::Greedy)
    }

    /// The next id: the fixed one, or `draw`'s pick from `row` among the ids
    /// the program allows here; the walk then stands past it. A stop id
    /// before the program is spent, or a drawn id the program does not allow,
    /// is an error, and so is a pick of a waiting walk: its free text is the
    /// request's own draw, which the walk only watches ([`Walk::saw`]).
    pub(crate) fn choose_by(&mut self, row: &[f32], draw: &mut Draw<'_>) -> Result<u32, StepError> {
        if self.waiting() {
            return Err(self.stuck("a waiting walk picks nothing: free text is the request's own"));
        }
        self.settle();
        let id = match self.fixed_id() {
            Some(id) => id,
            None => {
                self.prime();
                self.pick(row, draw)?
            }
        };
        if self.stops().contains(&id) && !self.heads.iter().any(Head::done) {
            return Err(self.stuck("a stop id occurs before the program is spent"));
        }
        let mut out = Vec::with_capacity(self.heads.len());
        for h in std::mem::take(&mut self.heads) {
            self.take_into(h, id, &mut out);
        }
        if out.is_empty() {
            return Err(self.stuck(&format!(
                "the drawn id {id} is not one the program allows here"
            )));
        }
        self.heads = out;
        self.taken += 1;
        Ok(id)
    }

    /// The walk's refusal to go on, for `why`.
    pub(crate) fn stuck(&self, why: &str) -> StepError {
        StepError::NoAllowed {
            taken: self.taken,
            why: why.to_owned(),
        }
    }

    /// Whether the walk stands in free text once its heads have moved on:
    /// [`Walk::waiting`], a call whose last id was taken counted as closed.
    pub(crate) fn in_free_text(&mut self) -> bool {
        self.settle();
        self.waiting()
    }

    /// The id just taken and the text the generation loop's decoder released
    /// for it, fed before the next pick, whether the walk picked the id or a
    /// pass kept it: a waiting walk counts it and watches it for its opener.
    /// Where the vocabulary holds the opener as one special token, that id
    /// opens the program. Otherwise the opener is read on the stream's text,
    /// across pieces, as the chat format's own scan reads it (one id's own
    /// decode can be half a character), and the text after it in the same
    /// piece is the program's first ([`Walk::read`]).
    pub(crate) fn saw(&mut self, id: u32, text: &str) -> Result<(), StepError> {
        if !self.waiting() {
            return Ok(());
        }
        self.taken += 1;
        let o = Arc::clone(
            self.waiting
                .as_ref()
                .expect("a waiting walk watches an opener"),
        );
        match o.special {
            Some(special) => {
                if special == id {
                    self.open(&o);
                }
                Ok(())
            }
            None => self.read(text, &o),
        }
    }

    /// `text`, just taken in free text, read for the text opener `o`: where
    /// the free text holds it, the program opens and takes the text after it
    /// in the same piece, which the program must start with — any other is
    /// the named error, never text the call's program did not pick. The tail
    /// keeps only what a later piece could still finish an opener with.
    fn read(&mut self, text: &str, o: &Opener) -> Result<(), StepError> {
        self.tail.extend_from_slice(text.as_bytes());
        let Some(at) = self
            .tail
            .windows(o.text.len())
            .position(|w| w == o.text.as_bytes())
        else {
            let drop = self.tail.len().saturating_sub(o.text.len() - 1);
            self.tail.drain(..drop);
            return Ok(());
        };
        let rest = String::from_utf8(self.tail.split_off(at + o.text.len()))
            .expect("the bytes after an opener's whole characters are whole characters");
        self.open(o);
        if rest.is_empty() {
            return Ok(());
        }
        let mut out = Vec::new();
        for h in std::mem::take(&mut self.heads) {
            self.eat(h, &rest, u32::MAX, &mut out);
        }
        if out.is_empty() {
            return Err(self.stuck(&format!(
                "the text {rest:?} after the call opener is not the call program's"
            )));
        }
        self.heads = out;
        Ok(())
    }

    /// The call's program, started at its opener's end.
    fn open(&mut self, o: &Opener) {
        self.heads = vec![Head {
            frames: vec![Frame::of(&o.prog)],
            ctx: Ctx::Idle,
            forced: Vec::new(),
        }];
        self.tail.clear();
    }

    /// The content a held walk's think tracker released in the piece that
    /// ended its wait — the text past the span's end, the first content of a
    /// reply whose span the model did not open, or the call opener that ends
    /// the span where the parser's own rule says so (Qwen3-Coder's) — with the
    /// special tokens' texts in it, as byte offsets and lengths: a waiting
    /// walk reads it for its opener and starts the program at one it
    /// completes. Where the vocabulary holds the opener as one special token,
    /// only that token ending the content is an opener — text spelling it is
    /// content — and a piece that ran past the opener opens nothing, the
    /// bytes after it already written. A text opener is read as
    /// [`Walk::saw`] reads one, the text after it the program's. Any other
    /// walk reads none of it.
    pub(crate) fn fed(&mut self, text: &str, specials: &[(usize, usize)]) -> Result<(), StepError> {
        let Some(o) = self.waiting.clone().filter(|_| self.heads.is_empty()) else {
            return Ok(());
        };
        if o.special.is_none() {
            return self.read(text, &o);
        }
        if specials.last().is_some_and(|&(at, len)| {
            at + len == text.len() && len == o.text.len() && text.get(at..) == Some(o.text)
        }) {
            self.open(&o);
        }
        Ok(())
    }

    /// `draw`'s pick among the ids the heads allow, counted by kind; a lone
    /// choice's decision goes to the watch.
    fn pick(&mut self, row: &[f32], draw: &mut Draw<'_>) -> Result<u32, StepError> {
        let n = row.len();
        let [h] = self.heads.as_slice() else {
            let id = draw.pick(row, self.heads.iter().flat_map(|h| self.allowed(h, n)));
            let id =
                id.ok_or_else(|| self.stuck("the row holds none of the ids the program allows"))?;
            if self
                .heads
                .iter()
                .any(|h| matches!(h.ctx, Ctx::Free { .. } | Ctx::Num { .. }))
            {
                self.picks.free += 1;
            } else {
                self.picks.decisions += 1;
            }
            return Ok(id);
        };
        let picked = draw.pick(row, self.allowed(h, n));
        match &h.ctx {
            Ctx::Done => picked.ok_or_else(|| self.stuck("the row holds none of the stop ids")),
            Ctx::Trie { prog, step, node } => {
                let Step::Pick(t) = &prog.steps[*step] else {
                    return Err(self.stuck("a choice context stands on another step"));
                };
                let id =
                    picked.ok_or_else(|| self.stuck("the row holds none of the option ids"))?;
                let kids = &t.nodes[*node].kids;
                if let Some(w) = &self.watch {
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
                self.picks.decisions += 1;
                Ok(id)
            }
            Ctx::Free {
                prog, step, chars, ..
            } => {
                let room = match &prog.steps[*step] {
                    Step::Free(f) => f.room(*chars),
                    _ => 0,
                };
                let id = picked.ok_or_else(|| {
                    self.stuck(if room == 0 {
                        "a free string reached its length limit and no end token is allowed"
                    } else {
                        "the row holds no token a free string may take"
                    })
                })?;
                self.picks.free += 1;
                Ok(id)
            }
            Ctx::Num { .. } => {
                let id =
                    picked.ok_or_else(|| self.stuck("the row holds no token a number may take"))?;
                self.picks.free += 1;
                Ok(id)
            }
            Ctx::Idle => Err(self.stuck("the walk stands between steps")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockTokenizer, SPECIALS};
    use serde_json::json;

    #[test]
    fn suffix_storage_is_bounded_by_the_longest_token() {
        let tok = MockTokenizer;
        let vocab = Vocab::read(&tok);
        let env = Env::new(&tok, &vocab);
        let max_bytes = SPECIALS.iter().map(|s| s.len()).max().unwrap();
        for letter in ["a", "é"] {
            let key = letter.repeat(4096);
            let schema = json!({"type": "object", "properties": {
                "first": {"type": "string"},
                (key.clone()): {"type": "null"}
            }, "required": ["first", key]});
            let parts = super::super::schema::value(&schema, Style::Json, "schema").unwrap();
            let prog = compile(&parts, &env).unwrap();
            let Step::Forced(tail) = prog.steps.last().unwrap() else {
                panic!("the property after the free string is forced");
            };
            assert!(tail.text.len() > 4096);
            assert!(
                tail.rests.iter().all(|(at, _)| *at <= max_bytes),
                "an end token cannot reach beyond its vocabulary's longest piece"
            );
            assert_eq!(
                tail.rest(max_bytes).is_some(),
                tail.text.is_char_boundary(max_bytes),
                "the exact limit is retained only at a character boundary"
            );
            assert!(tail.rests.len() <= max_bytes);
            assert!(
                tail.rests.iter().map(|(_, ids)| ids.len()).sum::<usize>()
                    <= max_bytes * tail.ids.len(),
                "suffix ids must grow linearly with literal length"
            );
        }
    }

    #[test]
    fn taking_a_forced_token_preserves_the_heads_buffers() {
        let tok = MockTokenizer;
        let vocab = Arc::new(Vocab::read(&tok));
        let prog = compile(&[Part::Text("a".repeat(16384))], &Env::new(&tok, &vocab)).unwrap();
        let c = Constraint::new(prog, Vec::new(), Vec::new(), vocab);
        let mut walk = Walk::new(&c, false);
        walk.settle();
        let forced = walk.heads[0].forced.as_ptr();
        let frames = walk.heads[0].frames.as_ptr();
        walk.choose(&[]).unwrap();
        assert_eq!(walk.heads[0].forced.len(), 16383);
        assert_eq!(
            walk.heads[0].forced.as_ptr(),
            forced,
            "move the remaining ids"
        );
        assert_eq!(walk.heads[0].frames.as_ptr(), frames, "move the frames");
    }
}
