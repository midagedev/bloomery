//! The pre-tokenizer of the `deepseek-v3` / `hunyuan-dense` / `joyai-llm`
//! vocabularies, as the reference runs it.
//!
//! The reference applies three regexes in turn (`llm_tokenizer_bpe`):
//!
//! 1. `\p{N}{1,3}`
//! 2. `[一-龥぀-ゟ゠-ヿ]+`
//! 3. `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+|
//!    ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! None has a hand-written splitter there, so all three go through its
//! `std::regex` fallback (`unicode_regex_split`). Each regex re-splits every
//! piece the previous one left, on that piece alone: a lookahead at a piece's
//! end sees the end of input, and the unmatched runs between matches stay
//! pieces of their own. Regexes 1 and 3 name Unicode categories, so they run
//! on the collapsed text (one byte per codepoint, see [`crate::unicode::collapse`])
//! with each `\p{X}` rewritten as "its collapsed byte, plus the ASCII members
//! the reference lists for X". Those ASCII lists are the reference's own
//! (`k_ucat_map`), not Unicode's: `~` is in neither P nor S. Regex 2 names no
//! category and runs on the codepoints themselves.
//!
//! The matchers below are those three regexes under ECMAScript semantics
//! (leftmost start, alternatives in order, greedy quantifiers that back off),
//! written out by hand; none of them can match the empty string.

use crate::unicode::collapse;

const NUMBER: u8 = 0xD1;
const LETTER: u8 = 0xD2;
const PUNCT: u8 = 0xD3;
const MARK: u8 = 0xD4;
const SYMBOL: u8 = 0xD5;

fn is_space(c: u8) -> bool {
    // std::regex `\s` on a `char` in the C locale; non-ASCII whitespace has
    // already collapsed to 0x0B.
    c == b' ' || (0x09..=0x0D).contains(&c)
}

fn is_newline(c: u8) -> bool {
    c == b'\r' || c == b'\n'
}

fn is_number(c: u8) -> bool {
    c == NUMBER || c.is_ascii_digit()
}

fn is_letter(c: u8) -> bool {
    c == LETTER || c.is_ascii_alphabetic()
}

fn is_punct(c: u8) -> bool {
    // `!-#%-*,-/:-;?-@[-]_{}`
    c == PUNCT
        || matches!(c, 0x21..=0x23 | 0x25..=0x2A | 0x2C..=0x2F | 0x3A..=0x3B | 0x3F..=0x40)
        || matches!(c, 0x5B..=0x5D | 0x5F | 0x7B | 0x7D)
}

fn is_symbol(c: u8) -> bool {
    // `$+<=>^`|`
    c == SYMBOL || matches!(c, 0x24 | 0x2B | 0x3C..=0x3E | 0x5E | 0x60 | 0x7C)
}

fn is_letter_or_mark(c: u8) -> bool {
    c == MARK || is_letter(c)
}

fn is_punct_or_symbol(c: u8) -> bool {
    is_punct(c) || is_symbol(c)
}

fn is_cjk(cpt: u32) -> bool {
    matches!(cpt, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

/// Length of the run of `pred` in `c[i..hi]`.
fn run(c: &[u8], i: usize, hi: usize, pred: impl Fn(u8) -> bool) -> usize {
    c[i..hi].iter().take_while(|&&x| pred(x)).count()
}

/// Regex 1 at `i`: one to three numbers.
fn match_number(c: &[u8], i: usize, hi: usize) -> usize {
    run(c, i, hi, is_number).min(3)
}

/// Regex 3 at `i`: the first of its six alternatives that matches.
fn match_word(c: &[u8], i: usize, hi: usize) -> usize {
    let at = |k: usize| c[k];

    // `[<ascii punctuation>][A-Za-z]+` — the class is written out literally
    // and holds every ASCII punctuation byte, `~` included.
    if at(i).is_ascii_punctuation() && i + 1 < hi {
        let n = run(c, i + 1, hi, |x| x.is_ascii_alphabetic());
        if n > 0 {
            return 1 + n;
        }
    }

    // `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
    let x = at(i);
    if !is_newline(x) && !is_letter(x) && !is_punct_or_symbol(x) && i + 1 < hi {
        let n = run(c, i + 1, hi, is_letter_or_mark);
        if n > 0 {
            return 1 + n;
        }
    }
    let n = run(c, i, hi, is_letter_or_mark);
    if n > 0 {
        return n;
    }

    // ` ?[\p{P}\p{S}]+[\r\n]*` — a space is not P or S, so backing the space
    // off never helps.
    let start = if x == b' ' { i + 1 } else { i };
    if start < hi {
        let n = run(c, start, hi, is_punct_or_symbol);
        if n > 0 {
            let end = start + n;
            return end + run(c, end, hi, is_newline) - i;
        }
    }

    // `\s*[\r\n]+`: through the last newline of the whitespace run.
    let w = run(c, i, hi, is_space);
    if let Some(last) = c[i..i + w].iter().rposition(|&x| is_newline(x)) {
        return last + 1;
    }

    // `\s+(?!\S)`: the whole run at the piece's end, else all but its last.
    if w > 0 && i + w == hi {
        return w;
    }
    if w > 1 {
        return w - 1;
    }

    // `\s+`
    w
}

/// Re-split every piece of `pieces` (lengths, in codepoints) with a matcher
/// `m(i, hi)` that returns the match length at `i` inside `[.., hi)`, 0 for
/// none. Unmatched runs become pieces of their own.
fn resplit(pieces: &[usize], out: &mut Vec<usize>, mut m: impl FnMut(usize, usize) -> usize) {
    out.clear();
    let mut lo = 0;
    for &len in pieces {
        let hi = lo + len;
        let mut gap = lo;
        let mut i = lo;
        while i < hi {
            let n = m(i, hi);
            if n == 0 {
                i += 1;
                continue;
            }
            if i > gap {
                out.push(i - gap);
            }
            out.push(n);
            i += n;
            gap = i;
        }
        if gap < hi {
            out.push(hi - gap);
        }
        lo = hi;
    }
}

/// Reusable buffers for [`split`].
#[derive(Default)]
pub(crate) struct Scratch {
    collapsed: Vec<u8>,
    a: Vec<usize>,
    b: Vec<usize>,
}

/// Split one text fragment, given as codepoints, into pre-tokenizer words;
/// returns each word's length in codepoints, in order.
pub(crate) fn split<'s>(cpts: &[u32], s: &'s mut Scratch) -> &'s [usize] {
    s.collapsed.clear();
    s.collapsed.extend(cpts.iter().map(|&cp| collapse(cp)));
    let c = &s.collapsed;

    s.a.clear();
    if !cpts.is_empty() {
        s.a.push(cpts.len());
    }
    resplit(&s.a, &mut s.b, |i, hi| match_number(c, i, hi));
    resplit(&s.b, &mut s.a, |i, hi| {
        cpts[i..hi].iter().take_while(|&&cp| is_cjk(cp)).count()
    });
    resplit(&s.a, &mut s.b, |i, hi| match_word(c, i, hi));
    &s.b
}
