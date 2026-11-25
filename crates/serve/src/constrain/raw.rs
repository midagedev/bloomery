//! What a raw value may hold: the tag that ends it and the tags it may not
//! spell, read as one automaton over bytes.
//!
//! A raw value is the text a chat format writes between a parameter's
//! opening tag and its close, byte for byte. The format's scan ends it at the
//! first [`Ends::close`] after it, and cuts the call at some tags of its own
//! wherever they stand ([`Ends::never`]): so a value takes any text that
//! spells neither, and ends where its text completes the close. The automaton
//! (Aho–Corasick over the tags, as a table over bytes) follows how much of a
//! tag the value's tail spells across tokens, so a `<` or a line feed is text
//! like any other until the close is whole.
//!
//! A token's text is its single-id decode, where the bytes of a split
//! character read as U+FFFD: where a tag could go on with a byte past ASCII,
//! such a token cannot be read and is no value text. A special token is no
//! value text either. Where the close's next id is one of those (a close that
//! holds a special token, or a character this vocabulary splits), that id
//! ends the value and the close's other ids follow it forced: the exit.

use std::sync::{Arc, Mutex};

use super::Refusal;
use super::vocab::{Bits, Vocab};
use crate::api::relock;
use crate::engine::Tokenizer;

/// The tags of a raw value, as its chat format's scan reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Ends {
    /// The tag the scan ends the value at: the first one after the value's
    /// start.
    pub close: &'static str,
    /// The tags the scan cuts the call at wherever they stand, a value's text
    /// too.
    pub never: &'static [&'static str],
}

/// What a tag the automaton reaches is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hit {
    No,
    Close,
    Never,
}

/// Where a token's text leaves a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Run {
    /// Still value text, the automaton at this state.
    Value(usize),
    /// The close is whole after this many bytes of the text: the value ends
    /// where the close began.
    Close(usize),
    /// The text spells a tag the value may not, the close where the value
    /// may not spell it, or a split character where a tag could go on.
    Refused,
}

/// One state's tokens: those that leave the value text, and those that
/// complete the close, each with the bytes of its text through the close.
type Read = (Arc<Bits>, Arc<[(u32, usize)]>);

/// The automaton of one [`Ends`] over one vocabulary.
pub(crate) struct RawTags {
    close: &'static str,
    /// The state after a byte: `next[state * 256 + byte]`.
    next: Vec<u16>,
    hit: Vec<Hit>,
    /// The characters of the longest end of a state's text that begins the
    /// close and that the value can still end from: the value's tail that
    /// the close may yet take.
    pend: Vec<usize>,
    /// The characters a value at a state may still have to write before it
    /// can end; `usize::MAX` where no path of one-character tokens ends it.
    need: Vec<usize>,
    /// Whether a byte past ASCII can move the automaton on from a state.
    wild: Vec<bool>,
    /// The id that ends the value from a state that stands for a prefix of
    /// the close, where ordinary text cannot finish it, and the ids after it.
    exits: Vec<Option<(u32, Arc<[u32]>)>>,
    /// The close's own ids hold a special token: ordinary ids that spell it
    /// are refused, and its exits end the value.
    close_special: bool,
    /// Each state's tokens, read on first use.
    read: Mutex<Vec<Option<Read>>>,
}

impl std::fmt::Debug for RawTags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTags")
            .field("close", &self.close)
            .field("states", &self.hit.len())
            .finish_non_exhaustive()
    }
}

fn internal(message: String) -> Refusal {
    Refusal::invalid(format!("internal: {message}"))
}

impl RawTags {
    /// The automaton of `ends` over `tok`'s vocabulary. The tags must be
    /// apart: none inside another, and no tag's end the start of another, so
    /// a tag's first end is where its scan finds it first.
    pub(crate) fn new(ends: Ends, tok: &dyn Tokenizer, vocab: &Vocab) -> Result<RawTags, Refusal> {
        let tags: Vec<&str> = std::iter::once(ends.close)
            .chain(ends.never.iter().copied())
            .collect();
        for (i, a) in tags.iter().enumerate() {
            if a.is_empty() {
                return Err(internal("a raw value's tag is empty".to_owned()));
            }
            for (j, b) in tags.iter().enumerate() {
                let overlap = (1..a.len()).any(|k| b.as_bytes().starts_with(&a.as_bytes()[k..]));
                if i != j && (b.contains(a) || overlap) {
                    return Err(internal(format!(
                        "the raw value's tags {a:?} and {b:?} overlap"
                    )));
                }
            }
        }
        // The trie, each state's own tag and the text it stands for.
        let mut kids: Vec<Vec<(u8, usize)>> = vec![Vec::new()];
        let mut own = vec![Hit::No];
        let mut text: Vec<Vec<u8>> = vec![Vec::new()];
        for (k, tag) in tags.iter().enumerate() {
            let mut s = 0;
            for &b in tag.as_bytes() {
                s = match kids[s].iter().find(|&&(c, _)| c == b) {
                    Some(&(_, t)) => t,
                    None => {
                        kids.push(Vec::new());
                        own.push(Hit::No);
                        let mut t = text[s].clone();
                        t.push(b);
                        text.push(t);
                        let t = kids.len() - 1;
                        kids[s].push((b, t));
                        t
                    }
                };
            }
            own[s] = if k == 0 { Hit::Close } else { Hit::Never };
        }
        let n = kids.len();
        if n > usize::from(u16::MAX) {
            return Err(internal("a raw value's tags are too long".to_owned()));
        }
        // Breadth first: each state's fallback and its row of the table,
        // the fallback's row complete before it.
        let mut next = vec![0u16; n * 256];
        let mut fail = vec![0usize; n];
        let mut hit = own.clone();
        let mut wild = vec![false; n];
        let mut order = std::collections::VecDeque::from([0usize]);
        while let Some(s) = order.pop_front() {
            let f = fail[s];
            if s != 0 {
                let (lo, hi) = (f * 256, s * 256);
                next.copy_within(lo..lo + 256, hi);
                if hit[s] == Hit::No {
                    hit[s] = hit[f];
                }
            }
            wild[s] = kids[s].iter().any(|&(b, _)| !b.is_ascii()) || (s != 0 && wild[f]);
            for &(b, t) in &kids[s] {
                fail[t] = if s == 0 {
                    0
                } else {
                    usize::from(next[f * 256 + usize::from(b)])
                };
                next[s * 256 + usize::from(b)] =
                    u16::try_from(t).expect("the states fit u16, checked above");
                order.push_back(t);
            }
        }
        let on_close: Vec<bool> = text
            .iter()
            .map(|t| !t.is_empty() && ends.close.as_bytes().starts_with(t))
            .collect();
        let close_ids = tok.encode(ends.close);
        let stop = |ids: &[u32], text: &str| match ids.iter().find(|id| vocab.stops().contains(id))
        {
            Some(id) => Err(Refusal::outside_subset(format!(
                "the text {text:?} tokenises to the stop id {id}: a stop id cannot occur before \
                 the program is spent"
            ))),
            None => Ok(()),
        };
        stop(&close_ids, ends.close)?;
        let close_special = close_ids.iter().any(|&id| tok.is_special(id));
        let mut exits = vec![None; n];
        for (s, t) in text.iter().enumerate() {
            let m = t.len();
            if m < ends.close.len()
                && ends.close.as_bytes().starts_with(t)
                && ends.close.is_char_boundary(m)
            {
                let rest = &ends.close[m..];
                let ids = tok.encode(rest);
                stop(&ids, rest)?;
                if let Some((&e, after)) = ids.split_first()
                    && (tok.is_special(e) || vocab.piece(e).contains(char::REPLACEMENT_CHARACTER))
                {
                    exits[s] = Some((e, Arc::from(after)));
                }
            }
        }
        // The states the value can end from at once: an exit, or a start of
        // the close whose rest ordinary text spells, where the close holds no
        // special token.
        let ends_from: Vec<bool> = (0..n)
            .map(|s| exits[s].is_some() || (on_close[s] && !close_special))
            .collect();
        // A state's fallback stands for the longest end of its text the trie
        // holds, and every start of the close is a state: breadth first, the
        // fallback's own is read before it.
        let mut pend = vec![0usize; n];
        let mut order = std::collections::VecDeque::from([0usize]);
        while let Some(s) = order.pop_front() {
            pend[s] = if ends_from[s] {
                String::from_utf8_lossy(&text[s]).chars().count()
            } else if s == 0 {
                0
            } else {
                pend[fail[s]]
            };
            order.extend(kids[s].iter().map(|&(_, t)| t));
        }
        // The room a value needs to end from each state: none where it can
        // end at once; elsewhere the least, over paths of one-character
        // tokens to such a state, of the most characters the value holds
        // past its count now on the way. A step from `s` to `t` counts
        // `1 + pend[s] - pend[t]` characters, fewer where the close takes
        // back characters already counted.
        let single: Vec<u8> = (0..128u8)
            .filter(|&b| {
                vocab
                    .starting_with(b)
                    .iter()
                    .any(|&id| vocab.is_text(id) && vocab.piece(id).len() == 1)
            })
            .collect();
        let mut need: Vec<usize> = (0..n)
            .map(|s| if ends_from[s] { 0 } else { usize::MAX })
            .collect();
        let mut moved = true;
        while moved {
            moved = false;
            for s in 0..n {
                for &b in &single {
                    let t = usize::from(next[s * 256 + usize::from(b)]);
                    if hit[t] != Hit::No || need[t] == usize::MAX {
                        continue;
                    }
                    let via = (1 + pend[s] + need[t]).saturating_sub(pend[t]);
                    if via < need[s] {
                        need[s] = via;
                        moved = true;
                    }
                }
            }
        }
        let tags = RawTags {
            close: ends.close,
            next,
            hit,
            pend,
            need,
            wild,
            exits,
            close_special,
            read: Mutex::new(vec![None; n]),
        };
        // The value can end from its start: its exit, or an ordinary token
        // that begins the close.
        let begins = vocab
            .starting_with(ends.close.as_bytes()[0])
            .iter()
            .any(|&id| {
                vocab.is_text(id)
                    && match tags.run(0, vocab.piece(id)) {
                        Run::Close(_) => true,
                        Run::Value(s) => ends_from[s],
                        Run::Refused => false,
                    }
            });
        if tags.exits[0].is_none() && !begins {
            return Err(Refusal::outside_subset(format!(
                "no token of the vocabulary begins {:?}, so a raw value before it could never end",
                ends.close
            )));
        }
        Ok(tags)
    }

    /// The close.
    pub(crate) fn close(&self) -> &'static str {
        self.close
    }

    /// Where `piece`, taken at `state`, leaves the value.
    pub(crate) fn run(&self, mut state: usize, piece: &str) -> Run {
        for (at, c) in piece.char_indices() {
            if c == char::REPLACEMENT_CHARACTER && self.wild[state] {
                return Run::Refused;
            }
            let mut buf = [0u8; 4];
            for (k, &b) in c.encode_utf8(&mut buf).as_bytes().iter().enumerate() {
                state = usize::from(self.next[state * 256 + usize::from(b)]);
                match self.hit[state] {
                    Hit::No => {}
                    Hit::Close if !self.close_special => return Run::Close(at + k + 1),
                    Hit::Close | Hit::Never => return Run::Refused,
                }
            }
        }
        Run::Value(state)
    }

    /// The characters of the value's tail at `state` that the close may yet
    /// take: under a limit, the value's characters are counted without them,
    /// so a value at its limit can still write its close, and only its
    /// close; a start of the close the value could not end from is counted
    /// as the text it then is.
    pub(crate) fn pending(&self, state: usize) -> usize {
        self.pend[state]
    }

    /// The characters a value at `state` may still have to write before it
    /// can end: under a limit, a token is taken only where the value's count
    /// after it and this still fit, so a value never stands where it cannot
    /// end. `usize::MAX` where it cannot end by one-character tokens.
    pub(crate) fn need(&self, state: usize) -> usize {
        self.need[state]
    }

    /// The id that ends the value at `state` though ordinary text cannot
    /// spell the close's rest there, and the ids forced after it.
    pub(crate) fn exit(&self, state: usize) -> Option<&(u32, Arc<[u32]>)> {
        self.exits[state].as_ref()
    }

    /// The tokens that leave the value text at `state`, and those that
    /// complete the close there with the bytes of their text through it.
    pub(crate) fn tokens(&self, state: usize, vocab: &Vocab) -> Read {
        if let Some(r) = &relock(&self.read)[state] {
            return r.clone();
        }
        let n = vocab.n_vocab();
        let mut ok = Bits::new(n);
        let mut done = Vec::new();
        for id in 0..n {
            let id32 = u32::try_from(id).expect("a vocabulary fits u32");
            if !vocab.is_text(id32) {
                continue;
            }
            match self.run(state, vocab.piece(id32)) {
                Run::Value(_) => ok.set(id),
                Run::Close(at) => done.push((id32, at)),
                Run::Refused => {}
            }
        }
        let r: Read = (Arc::new(ok), done.into());
        relock(&self.read)[state] = Some(r.clone());
        r
    }
}
