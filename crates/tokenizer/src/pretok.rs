//! The pre-tokenizers of the vocabularies this crate runs, as the reference
//! runs them: [`Pre::DeepseekV3`] (`deepseek-v3`, `hunyuan-dense`,
//! `joyai-llm`), [`Pre::Qwen2`] (`qwen2`, `solar-open`), [`Pre::Glm4`] (`glm4`),
//! [`Pre::Qwen35`] (`qwen35`), [`Pre::Pixtral`] (`pixtral`),
//! [`Pre::MiniMaxM2`] (`minimax-m2`) and [`Pre::Gemma4`] (`gemma4`).
//!
//! ## deepseek-v3
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
//! the reference lists for X" (`k_ucat_map`). Those lists are Unicode's
//! general categories below 0x80: P is `!-#%-*,-/:-;?-@[-]_{}` and S is
//! `` $+<=>^`|~ ``, `~` included (Sm). Regex 2 names no category and runs on the
//! codepoints themselves.
//!
//! The matchers below are those three regexes under ECMAScript semantics
//! (leftmost start, alternatives in order, greedy quantifiers that back off),
//! written out by hand; none of them can match the empty string.
//!
//! ## qwen2
//!
//! One regex, `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|
//! \p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`, which the reference does not
//! run as a regex: `unicode_regex_split_custom_qwen2` is a hand-written
//! splitter that reads the codepoints' flags from its Unicode tables. It
//! differs from a regex engine in one place that matters: a position no
//! alternative matches becomes a one-codepoint piece, so every position is
//! consumed. [`split_custom`] is that function line for line. The flags it
//! reads — letter, number, whitespace — are exactly what the collapsed byte
//! keeps: below 0x80 the reference's tables agree with the ASCII classes
//! used here, and above it a letter collapses to its own byte, a number to
//! its own, whitespace to 0x0B. The contraction test lowercases the next
//! codepoint with the reference's `unicode_tolower`; the only codepoints that
//! table maps onto `s t m d r v e l` are their ASCII capitals, so ASCII
//! lowercasing is the same test.
//!
//! ## glm4
//!
//! The reference maps `glm4` to its CHATGLM4 type, whose one regex is qwen2's
//! with `\p{N}{1,3}` in place of `\p{N}`; that regex goes to
//! `unicode_regex_split_custom_llama3`, the qwen2 splitter with a different
//! digit branch and nothing else. There a run of numbers is cut into words of
//! three from its start, the last one shorter. [`split_custom`] takes that
//! difference as its one parameter, the longest run of numbers one word holds:
//! 1 for qwen2, 3 for glm4. The reference cuts the whole run in one visit and
//! the function here one word per visit; a visit that starts on a number takes
//! the number branch (the contraction needs `'` and the letter branch refuses a
//! number), so the words are the same.
//!
//! ## qwen35
//!
//! One regex, qwen2's with `\p{M}` joined to the letter classes — the letter
//! run is `[\p{L}\p{M}]+` and the "other" run's class drops marks — and the
//! reference runs it as a regex, not a hand-written splitter: its
//! `unicode_regex_split_custom` does not know it, so it goes to the
//! `std::regex` fallback on the collapsed text like the deepseek-v3 regexes,
//! with the same piece-local semantics (a lookahead at a piece's end sees the
//! end of input, and the unmatched runs between matches stay pieces of their
//! own). A mark therefore reaches the pre-tokenizer as its own collapsed byte,
//! and the letter run's optional prefix `[^\r\n\p{L}\p{N}]?` — which the
//! regex leaves as qwen2's — takes one as readily as a space or an apostrophe.
//! [`match_qwen35`] is that regex under ECMAScript semantics, written out by
//! hand, and [`split_qwen35`] its one pass over the fragment.
//!
//! A Hugging Face `tokenizer.json` runs the same regex with Oniguruma, where
//! `(?i:...)` folds case by Unicode: `'s` also matches `'ſ` (U+017F, whose
//! case fold is `s`), and no other codepoint folds to a contraction's letter.
//! Its classes are Unicode 14.0's, so a letter, mark or number Unicode 15.0 or
//! 15.1 added is of no class there ([`crate::unicode::collapse_hf`]).
//! [`Pre::Qwen35Hf`] is that reading, for a vocabulary that declares itself
//! a Hugging Face one.
//!
//! ## pixtral
//!
//! The reference maps `pixtral` to its LLAMA3 type. That type's one regex is
//! the one [`Pre::Glm4`] runs — qwen2's with `\p{N}{1,3}` in place of `\p{N}` —
//! and it goes to the same hand-written splitter (`unicode_regex_split_custom_llama3`),
//! so [`split`] uses [`split_custom`] with a run of three numbers, the `glm4`
//! arm. The reference also sets `ignore_merges` for this name and not for
//! `glm4`: a pre-tokenizer word whose byte-encoded text is itself a vocabulary
//! token is that token, and merges do not run on it. [`Pre::ignore_merges`]
//! carries the flag for any pre, so the two names do not share a variant.
//!
//! ## minimax-m2
//!
//! The reference maps `minimax-m2` to its MINIMAX_M2 type, which shares GPT4O's
//! one expression:
//!
//! `[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|
//! [^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|
//! \p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! `unicode_regex_split_custom` knows no such expression, so it goes to the
//! `std::regex` fallback on the collapsed text like qwen35's, with the same
//! piece-local semantics; `\p{L}` and `\p{N}` inside a class become their
//! collapsed bytes and the ASCII members the reference lists, while `[^a-z]` and
//! `[^A-Z]` stay as written. A non-ASCII letter is one collapsed byte
//! ([`LETTER`]) that is in neither `a-z` nor `A-Z`, so `((?=[\p{L}])([^a-z]))`
//! takes it as a capital and `((?=[\p{L}])([^A-Z]))` as a lowercase letter:
//! every letter outside ASCII is both. The first alternative is a prefix, then
//! capitals and non-ASCII letters (greedy) and one or more lowercase and
//! non-ASCII letters; the star gives back from its end until a lowercase
//! letter can start, so a run of capitals ending in a non-ASCII letter leaves
//! that letter to the plus. The second alternative runs only when the first did
//! not: capitals alone. A contraction follows either, and a mark is not a letter
//! here ([`match_minimax_m2`]). Past the words the alternatives are qwen2's, but
//! the symbol run's tail is `[\r\n/]*`, a slash as well as the newlines. The
//! reference sets `clean_spaces` false for this name; this crate cleans no spaces.
//!
//! ## gemma4
//!
//! One regex, `[^\n]+|[\n]+`, which the reference runs as the hand-written
//! `unicode_regex_split_custom_newlines`: the fragment is cut into runs of
//! U+000A and runs of everything else. `\r` is not a newline here. The words
//! are not GPT-2 byte-level: the reference writes each space as U+2581 before
//! the split and merges raw UTF-8 characters ([`Spelling::Metaspace`]).

use crate::spelling::Spelling;
use crate::unicode::{collapse, collapse_hf};

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
    // `$+<=>^`|~`
    c == SYMBOL || matches!(c, 0x24 | 0x2B | 0x3C..=0x3E | 0x5E | 0x60 | 0x7C | 0x7E)
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

/// `[^\s\p{L}\p{N}]`: the "other" run's class of qwen2, glm4 and minimax-m2.
fn is_other(c: u8) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// `[^\s\p{L}\p{M}\p{N}]`: the qwen35 "other" run's class, which unlike
/// [`is_other`] drops the marks.
fn is_other_qwen35(c: u8) -> bool {
    is_other(c) && c != MARK
}

/// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])` at `i`: its
/// length, 0 for none. Each letter is a literal two-member class, so only its
/// ASCII cases match; `fold(k)` is the lowercase ASCII letter the codepoint at
/// `k` matches (under Hugging Face's Unicode folding, `ſ` as `s`).
fn contraction(c: &[u8], i: usize, hi: usize, fold: impl Fn(usize) -> u8) -> usize {
    if i + 1 < hi && c[i] == b'\'' {
        let one = |x: u8| fold(i + 1) == x;
        if one(b's') || one(b't') || one(b'm') || one(b'd') {
            return 2;
        }
        let two = |a: u8, b: u8| i + 2 < hi && fold(i + 1) == a && fold(i + 2) == b;
        if two(b'r', b'e') || two(b'v', b'e') || two(b'l', b'l') {
            return 3;
        }
    }
    0
}

/// `\s*[\r\n]+|\s+(?!\S)|\s+` at `i`, the last three alternatives of every
/// regex here: through the last newline of the whitespace run, else the whole
/// run at the piece's end, else all but its last, else its one codepoint.
fn match_space(c: &[u8], i: usize, hi: usize) -> usize {
    let w = run(c, i, hi, is_space);
    if let Some(last) = c[i..i + w].iter().rposition(|&x| is_newline(x)) {
        return last + 1;
    }
    if w > 0 && i + w == hi {
        return w;
    }
    if w > 1 { w - 1 } else { w }
}

/// The `qwen35` regex at `i`: the first of its alternatives that matches,
/// under ECMAScript semantics (alternatives in order, greedy quantifiers that
/// back off). None of them can match the empty string, and a mark is a plain
/// codepoint of its own class here: the regex runs on the collapsed bytes.
/// `fold(k)` is the lowercase ASCII letter the codepoint at `k` matches in a
/// contraction.
fn match_qwen35(c: &[u8], i: usize, hi: usize, fold: impl Fn(usize) -> u8) -> usize {
    let at = |k: usize| c[k];

    // `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`
    let n = contraction(c, i, hi, fold);
    if n > 0 {
        return n;
    }

    // `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` — the prefix first (a greedy `?`
    // backs off only when the run after it fails), then the run alone. The
    // prefix class takes a mark, a space or an apostrophe as readily as
    // qwen2's does.
    let x = at(i);
    if !is_newline(x) && !is_letter(x) && !is_number(x) && i + 1 < hi {
        let n = run(c, i + 1, hi, is_letter_or_mark);
        if n > 0 {
            return 1 + n;
        }
    }
    let n = run(c, i, hi, is_letter_or_mark);
    if n > 0 {
        return n;
    }

    // `\p{N}` — one number, however long the run.
    if is_number(x) {
        return 1;
    }

    // ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` — a space is none of the four, so
    // backing the space off never helps.
    let start = if x == b' ' { i + 1 } else { i };
    if start < hi {
        let n = run(c, start, hi, is_other_qwen35);
        if n > 0 {
            let end = start + n;
            return end + run(c, end, hi, is_newline) - i;
        }
    }

    // `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_space(c, i, hi)
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

    // `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_space(c, i, hi)
}

/// A capital or a non-ASCII letter: what `((?=[\p{L}])([^a-z]))` takes. A
/// non-ASCII letter collapses to [`LETTER`], which is in neither `a-z` nor
/// `A-Z`, so it is this and [`is_lower_m2`] at once.
fn is_upper_m2(c: u8) -> bool {
    c.is_ascii_uppercase() || c == LETTER
}

/// A lowercase or a non-ASCII letter: what `((?=[\p{L}])([^A-Z]))` takes.
fn is_lower_m2(c: u8) -> bool {
    c.is_ascii_lowercase() || c == LETTER
}

/// The `minimax-m2` regex at `i`: the first of its seven alternatives that
/// matches, under ECMAScript semantics (alternatives in order, greedy
/// quantifiers that back off). None of them can match the empty string, and a
/// mark is a plain codepoint of its own class here: the regex runs on the
/// collapsed bytes.
fn match_minimax_m2(c: &[u8], i: usize, hi: usize) -> usize {
    let x = c[i];

    // `[^\r\n\p{L}\p{N}]?` opens both word alternatives: a letter at `i` is
    // never the prefix, anything else but a newline or a number is, and with
    // no letter after it the alternatives fail at `i` altogether.
    let word = if is_letter(x) {
        Some(i)
    } else if !is_newline(x) && !is_number(x) && i + 1 < hi {
        Some(i + 1)
    } else {
        None
    };
    if let Some(s) = word {
        let ups = run(c, s, hi, is_upper_m2);
        // `((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+`: the star takes all
        // `ups`, and gives back until the plus can start: at the lowercase
        // letter that ended the run, else at its last non-ASCII letter (a
        // capital is no lowercase letter). Nothing to give back fails it.
        let plus_at = if s + ups < hi && is_lower_m2(c[s + ups]) {
            Some(ups)
        } else {
            c[s..s + ups].iter().rposition(|&b| b == LETTER)
        };
        // `((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*`: only capitals are left
        // (no lowercase letter follows them, or the first alternative would
        // have matched), so the star takes nothing.
        let end = match plus_at {
            Some(j) => s + j + run(c, s + j, hi, is_lower_m2),
            None => s + ups,
        };
        if end > s {
            // `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?`
            let fold = |k: usize| c[k].to_ascii_lowercase();
            return end + contraction(c, end, hi, fold) - i;
        }
    }

    // `\p{N}{1,3}`
    if is_number(x) {
        return match_number(c, i, hi);
    }

    // ` ?[^\s\p{L}\p{N}]+[\r\n/]*` — a space is none of the three, so
    // backing the space off never helps.
    let start = if x == b' ' { i + 1 } else { i };
    if start < hi {
        let n = run(c, start, hi, is_other);
        if n > 0 {
            let end = start + n;
            return end + run(c, end, hi, |b| is_newline(b) || b == b'/') - i;
        }
    }

    // `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_space(c, i, hi)
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

/// A vocabulary's pre-tokenizer, from `tokenizer.ggml.pre`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pre {
    /// `deepseek-v3`, `hunyuan-dense`, `joyai-llm`: three regexes in turn.
    DeepseekV3,
    /// `qwen2`, `solar-open`: the reference's hand-written splitter, one number
    /// a word. Solar's `tokenizer.json` carries qwen2's regex verbatim.
    Qwen2,
    /// `glm4`: the reference's llama3 splitter, qwen2's with up to three
    /// numbers a word.
    Glm4,
    /// `qwen35`: qwen2's regex with `\p{M}` joined to the letter classes,
    /// run as a regex, not a hand-written splitter.
    Qwen35,
    /// `qwen35` as a Hugging Face `tokenizer.json` runs it: contractions fold
    /// case by Unicode, classes are Unicode 14.0's, and BPE skips a byte its
    /// vocabulary lacks. Never named
    /// in a header; a vocabulary with `tokenizer.bloomery.eog_token_ids` gets it.
    Qwen35Hf,
    /// `pixtral`: the llama3 regex [`Pre::Glm4`] runs, and a word that is
    /// itself a token is taken whole ([`Pre::ignore_merges`]).
    Pixtral,
    /// `minimax-m2`: GPT4O's one regex, run as a regex over the collapsed text.
    MiniMaxM2,
    /// `gemma4`: runs of newlines and runs of the rest, over metaspace text.
    Gemma4,
}

impl Pre {
    /// The names this crate runs, as `tokenizer.ggml.pre` spells them.
    pub(crate) const NAMES: [(&'static str, Pre); 10] = [
        ("deepseek-v3", Pre::DeepseekV3),
        ("hunyuan-dense", Pre::DeepseekV3),
        ("joyai-llm", Pre::DeepseekV3),
        ("qwen2", Pre::Qwen2),
        ("solar-open", Pre::Qwen2),
        ("glm4", Pre::Glm4),
        ("qwen35", Pre::Qwen35),
        ("pixtral", Pre::Pixtral),
        ("minimax-m2", Pre::MiniMaxM2),
        ("gemma4", Pre::Gemma4),
    ];

    /// Whether a word that is itself a vocabulary token is taken whole,
    /// before merges. Any pre can carry it; `pixtral` does.
    pub(crate) fn ignore_merges(self) -> bool {
        matches!(self, Pre::Pixtral)
    }

    /// Whether BPE runs as Hugging Face's does ([`crate::bpe::Bpe::new`]).
    pub(crate) fn hf(self) -> bool {
        matches!(self, Pre::Qwen35Hf)
    }

    /// How this pre-tokenizer's words are spelled in the vocabulary.
    pub(crate) fn spelling(self) -> Spelling {
        match self {
            Pre::Gemma4 => Spelling::Metaspace,
            Pre::DeepseekV3
            | Pre::Qwen2
            | Pre::Glm4
            | Pre::Qwen35
            | Pre::Qwen35Hf
            | Pre::Pixtral
            | Pre::MiniMaxM2 => Spelling::ByteLevel,
        }
    }
}

/// Split one text fragment, given as codepoints, into pre-tokenizer words
/// under `pre`; returns each word's length in codepoints, in order.
pub(crate) fn split<'s>(pre: Pre, cpts: &[u32], s: &'s mut Scratch) -> &'s [usize] {
    s.collapsed.clear();
    let class = if pre.hf() { collapse_hf } else { collapse };
    s.collapsed.extend(cpts.iter().map(|&cp| class(cp)));
    match pre {
        Pre::DeepseekV3 => split_deepseek_v3(cpts, s),
        Pre::Qwen2 => {
            split_custom(cpts, &s.collapsed, 1, &mut s.b);
            &s.b
        }
        Pre::Glm4 | Pre::Pixtral => {
            split_custom(cpts, &s.collapsed, 3, &mut s.b);
            &s.b
        }
        Pre::Qwen35 => split_qwen35(cpts, s, false),
        Pre::Qwen35Hf => split_qwen35(cpts, s, true),
        Pre::MiniMaxM2 => {
            split_one_piece(cpts.len(), s, match_minimax_m2);
            &s.b
        }
        Pre::Gemma4 => {
            split_newlines(cpts, &mut s.b);
            &s.b
        }
    }
}

/// `unicode_regex_split_custom_newlines` over one fragment: alternating runs
/// of U+000A and of every other codepoint.
fn split_newlines(cpts: &[u32], out: &mut Vec<usize>) {
    out.clear();
    let mut pos = 0;
    while pos < cpts.len() {
        let newline = cpts[pos] == u32::from(b'\n');
        let n = cpts[pos..]
            .iter()
            .take_while(|&&c| (c == u32::from(b'\n')) == newline)
            .count();
        out.push(n);
        pos += n;
    }
}

/// One regex `m` over the collapsed bytes, the whole fragment of `len`
/// codepoints one piece, as the reference's fallback runs it: the words'
/// lengths go to `s.b`.
fn split_one_piece(len: usize, s: &mut Scratch, mut m: impl FnMut(&[u8], usize, usize) -> usize) {
    s.a.clear();
    if len > 0 {
        s.a.push(len);
    }
    let c = &s.collapsed;
    resplit(&s.a, &mut s.b, |i, hi| m(c, i, hi));
}

/// [`split`] for [`Pre::Qwen35`], and with `hf` for [`Pre::Qwen35Hf`]: its one
/// regex over the collapsed bytes.
fn split_qwen35<'s>(cpts: &[u32], s: &'s mut Scratch, hf: bool) -> &'s [usize] {
    split_one_piece(cpts.len(), s, |c, i, hi| {
        // The lowercase ASCII letter a contraction compares the codepoint at `k` with.
        let fold = |k: usize| {
            if hf && cpts[k] == 0x17F {
                b's'
            } else {
                c[k].to_ascii_lowercase()
            }
        };
        match_qwen35(c, i, hi, fold)
    });
    &s.b
}

/// [`split`] for [`Pre::DeepseekV3`]: the three regexes in turn.
fn split_deepseek_v3<'s>(cpts: &[u32], s: &'s mut Scratch) -> &'s [usize] {
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

/// `unicode_regex_split_custom_qwen2` (`digits` 1) or `_llama3` (`digits` 3)
/// over one fragment: `cpts` its codepoints, `c` their collapsed bytes; the
/// words' lengths go to `out`. Each branch is the reference's, in its order,
/// and names the alternative it stands for.
fn split_custom(cpts: &[u32], c: &[u8], digits: usize, out: &mut Vec<usize>) {
    out.clear();
    let end = cpts.len();
    let letter = |p: usize| p < end && is_letter(c[p]);
    let number = |p: usize| p < end && is_number(c[p]);
    let space = |p: usize| p < end && is_space(c[p]);
    // `[^\s\p{L}\p{N}]` with the reference's "has a flag": true inside the
    // fragment, false past its end.
    let other = |p: usize| p < end && is_other(c[p]);
    let lower = |p: usize| {
        cpts.get(p).map(|&x| {
            if (0x41..=0x5A).contains(&x) {
                x + 0x20
            } else {
                x
            }
        })
    };
    let (mut pos, mut prev) = (0usize, 0usize);
    let mut word = |to: usize, prev: &mut usize| {
        if to > *prev {
            out.push(to - *prev);
        }
        *prev = to;
    };
    while pos < end {
        let cpt = cpts[pos];
        // `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
        if cpt == u32::from(b'\'') && pos + 1 < end {
            let next = lower(pos + 1);
            if b"stmd".iter().any(|&x| next == Some(u32::from(x))) {
                pos += 2;
                word(pos, &mut prev);
                continue;
            }
            if pos + 2 < end {
                let pair = (next, lower(pos + 2));
                let is = |a: u8, b: u8| pair == (Some(u32::from(a)), Some(u32::from(b)));
                if is(b'r', b'e') || is(b'v', b'e') || is(b'l', b'l') {
                    pos += 3;
                    word(pos, &mut prev);
                    continue;
                }
            }
        }
        // `[^\r\n\p{L}\p{N}]?\p{L}+`
        if !is_newline(c[pos]) && !number(pos) && (letter(pos) || letter(pos + 1)) {
            pos += 1;
            while letter(pos) {
                pos += 1;
            }
            word(pos, &mut prev);
            continue;
        }
        // `\p{N}` (qwen2) or `\p{N}{1,3}` (glm4): up to `digits` numbers.
        if number(pos) {
            let start = pos;
            while pos - start < digits && number(pos) {
                pos += 1;
            }
            word(pos, &mut prev);
            continue;
        }
        // `<space>?[^\s\p{L}\p{N}]+[\r\n]*`: a space counts only when what
        // follows it is such a codepoint or the fragment's end.
        let lead = cpt == u32::from(b' ');
        let at = if lead { pos + 1 } else { pos };
        if at >= end || other(at) {
            pos = at;
            while other(pos) {
                pos += 1;
            }
            while pos < end && is_newline(c[pos]) {
                pos += 1;
            }
            word(pos, &mut prev);
            continue;
        }
        let mut n = 0;
        let mut last_newline = None;
        while space(pos + n) {
            if is_newline(c[pos + n]) {
                last_newline = Some(pos + n + 1);
            }
            n += 1;
        }
        // `\s*[\r\n]+`
        if let Some(to) = last_newline {
            pos = to;
            word(pos, &mut prev);
            continue;
        }
        // `\s+(?!\S)`: all but the last of a run that does not end the fragment.
        if n > 1 && pos + n < end {
            pos += n - 1;
            word(pos, &mut prev);
            continue;
        }
        // `\s+`
        if n > 0 {
            pos += n;
            word(pos, &mut prev);
            continue;
        }
        // No alternative: one codepoint.
        pos += 1;
        word(pos, &mut prev);
    }
}

#[cfg(test)]
mod tests {
    use super::{Pre, Scratch, split};

    /// Every name the table runs is one the crate says it runs and the
    /// refusal of another lists.
    #[test]
    fn names_have_one_table() {
        let refusal = crate::Error::UnsupportedPre("gpt-4o".to_string()).to_string();
        for (name, _) in Pre::NAMES {
            assert!(crate::runs_pre_tokenizer(name), "{name}");
            assert!(refusal.contains(name), "{name} not in: {refusal}");
        }
        assert!(!crate::runs_pre_tokenizer("gpt-4o"));
    }

    /// `solar-open` runs the qwen2 splitter: Solar-Open2's `tokenizer.json`
    /// pre-tokenizer is qwen2's regex, character for character.
    #[test]
    fn solar_open_is_qwen2() {
        assert_eq!(crate::named(&Pre::NAMES, "solar-open"), Some(Pre::Qwen2));
    }

    fn words(pre: Pre, text: &str) -> Vec<String> {
        let cpts: Vec<u32> = text.chars().map(u32::from).collect();
        let mut s = Scratch::default();
        let mut at = 0;
        split(pre, &cpts, &mut s)
            .iter()
            .map(|&n| {
                let w: String = text.chars().skip(at).take(n).collect();
                at += n;
                w
            })
            .collect()
    }

    /// The qwen2 splitter's alternatives, one case each, as its reference
    /// function splits them: contractions case-blind, one digit per word, a
    /// leading space or symbol taken by a word, a space kept before
    /// punctuation, whitespace runs backing off before a word.
    #[test]
    fn qwen2_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("it's", &["it", "'s"]),
            ("IT'S", &["IT", "'S"]),
            ("we'RE", &["we", "'RE"]),
            ("1234", &["1", "2", "3", "4"]),
            (" 12", &[" ", "1", "2"]),
            ("a  b", &["a", " ", " b"]),
            ("hello\n\nworld", &["hello", "\n\n", "world"]),
            ("x = y;", &["x", " =", " y", ";"]),
            ("end ", &["end", " "]),
            ("(foo)", &["(foo", ")"]),
            ("Ünïcödé тест", &["Ünïcödé", " тест"]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::Qwen2, text), want, "{text:?}");
        }
    }

    /// The glm4 splitter is qwen2's but for its number branch: a run of
    /// numbers is cut into words of three from its start, and every other
    /// alternative splits as qwen2's does.
    #[test]
    fn glm4_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("7", &["7"]),
            ("12", &["12"]),
            ("123", &["123"]),
            ("1234", &["123", "4"]),
            ("1234567", &["123", "456", "7"]),
            (" 12345", &[" ", "123", "45"]),
            ("abc1234def", &["abc", "123", "4", "def"]),
            ("x²³ ١٢٣٤", &["x", "²³", " ", "١٢٣", "٤"]),
            ("it's", &["it", "'s"]),
            ("a  b", &["a", " ", " b"]),
            ("x = y;", &["x", " =", " y", ";"]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::Glm4, text), want, "{text:?}");
            // `pixtral` is the same regex and the same splitter. What the
            // two names do not share is `ignore_merges`, which runs after
            // this split.
            assert_eq!(words(Pre::Pixtral, text), want, "pixtral {text:?}");
        }
    }

    /// The qwen35 splitter's alternatives beside qwen2's, one case each: a
    /// combining mark after a letter joins its word, a script whose vowels
    /// and virama are marks stays one word, a mark after punctuation joins
    /// the word the punctuation opens, and a lone mark before punctuation is
    /// a word of its own — the pieces below are the regex's, derived by hand
    /// (`[\p{L}\p{M}]+` takes every mark beside a letter, and a mark serves
    /// as the letter run's optional prefix as any non-letter does). Every
    /// case is checked against qwen2 on the same string too, whose splitter
    /// treats a mark as neither letter nor number — a "other" run member and
    /// a possible letter-run prefix — and cuts there.
    ///
    /// FAIL-first: with `\p{M}` left out of the two classes the regex adds
    /// it to — the letter run's and the "other" run's — every row below
    /// fails: a mark no longer joins the letter word ("e\u{301}" splits into
    /// "e" and the mark) and a mark after punctuation no longer opens a
    /// letter word (".\u{301}a" splits at the "a").
    #[test]
    fn qwen35_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              qwen35                    qwen2
            ("e\u{301}", &["e\u{301}"], &["e", "\u{301}"]),
            (
                "\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}",
                &["\u{928}\u{92e}\u{938}\u{94d}\u{924}\u{947}"],
                &["\u{928}\u{92e}\u{938}", "\u{94d}\u{924}", "\u{947}"],
            ),
            (
                "\u{e17}\u{e35}\u{e48}",
                &["\u{e17}\u{e35}\u{e48}"],
                &["\u{e17}", "\u{e35}\u{e48}"],
            ),
            (".\u{301}a", &[".\u{301}a"], &[".\u{301}", "a"]),
            ("\u{301}.", &["\u{301}", "."], &["\u{301}."]),
            ("it's", &["it", "'s"], &["it", "'s"]),
            ("1234", &["1", "2", "3", "4"], &["1", "2", "3", "4"]),
            ("a  b", &["a", " ", " b"], &["a", " ", " b"]),
            ("x = y;", &["x", " =", " y", ";"], &["x", " =", " y", ";"]),
        ];
        for &(text, want35, want2) in cases {
            assert_eq!(words(Pre::Qwen35, text), want35, "qwen35 {text:?}");
            assert_eq!(words(Pre::Qwen2, text), want2, "qwen2 {text:?}");
        }
    }

    /// Hugging Face's qwen35 folds a contraction's case by Unicode: `ſ`
    /// (U+017F) is `s` there, so `'ſ` is a word of its own where qwen35 keeps
    /// it in the letter run; the Kelvin sign folds to `k`, no contraction's
    /// letter, and splits as qwen35's does. A Kaktovik numeral and a Kawi
    /// letter (Unicode 15.0) are of no class there, so the numeral joins the
    /// space before it and the letter leaves the word.
    ///
    /// FAIL-first: with the ASCII fold for both, the first three rows fail;
    /// with the reference's classes for both, the last two.
    #[test]
    fn qwen35_hf_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              hf                        qwen35
            ("x'\u{17f}y", &["x", "'\u{17f}", "y"], &["x", "'\u{17f}y"]),
            (
                "'\u{17f}\u{17f}",
                &["'\u{17f}", "\u{17f}"],
                &["'\u{17f}\u{17f}"],
            ),
            (
                "we'\u{17f}e",
                &["we", "'\u{17f}", "e"],
                &["we", "'\u{17f}e"],
            ),
            ("'\u{17f}", &["'\u{17f}"], &["'\u{17f}"]),
            ("'\u{212a}x", &["'\u{212a}x"], &["'\u{212a}x"]),
            ("IT'S", &["IT", "'S"], &["IT", "'S"]),
            (" \u{1d2c0}", &[" \u{1d2c0}"], &[" ", "\u{1d2c0}"]),
            ("x\u{11f04}", &["x", "\u{11f04}"], &["x\u{11f04}"]),
        ];
        for &(text, hf, legacy) in cases {
            assert_eq!(words(Pre::Qwen35Hf, text), hf, "hf {text:?}");
            assert_eq!(words(Pre::Qwen35, text), legacy, "qwen35 {text:?}");
        }
    }

    /// The `minimax-m2` regex's alternatives, one case each, derived from the
    /// reference's rule (the expression run on the collapsed bytes, where a
    /// non-ASCII letter is both a capital and a lowercase letter): a word is
    /// capitals then lowercase letters, a contraction ends it whatever follows,
    /// digits come in threes, a mark opens a word or joins a symbol run, and a
    /// symbol run takes the newlines and slashes after it.
    ///
    /// FAIL-first: with the digit run cut at four, `1234567` and
    /// `abc1234def` fail; with the tail `[\r\n]*` for `[\r\n/]*`, `x/\n/y`
    /// fails; with a lowercase letter required after the capitals (no give-back
    /// to a non-ASCII letter) `中文ABC` and `한AB` fail.
    #[test]
    fn minimax_m2_cases() {
        let cases: &[(&str, &[&str])] = &[
            // Capitals then lowercase letters make one word; a run of capitals alone is the second alternative.
            ("HelloWorld", &["Hello", "World"]),
            ("ABCdef", &["ABCdef"]),
            ("HELLOWorld", &["HELLOWorld"]),
            ("iPhone", &["i", "Phone"]),
            ("NULL ptr", &["NULL", " ptr"]),
            // A contraction ends the word it follows, in either case, and does not need a word break after it.
            ("it's", &["it's"]),
            ("I'm", &["I'm"]),
            ("IT'S WE'RE", &["IT'S", " WE'RE"]),
            ("we'RE", &["we'RE"]),
            ("can't've", &["can't", "'ve"]),
            ("it'sa", &["it's", "a"]),
            // No contraction: the apostrophe is the next word's prefix.
            ("rock'n'roll", &["rock", "'n", "'roll"]),
            ("'S", &["'S"]),
            // Digits in threes, and a space or letter beside them is no part of the run.
            ("1234567", &["123", "456", "7"]),
            ("abc1234def", &["abc", "123", "4", "def"]),
            (" 12345", &[" ", "123", "45"]),
            ("x²³ ١٢٣٤", &["x", "²³", " ", "١٢٣", "٤"]),
            // The symbol run's tail takes newlines and slashes, in any order.
            ("a/b\n", &["a", "/b", "\n"]),
            ("x/\n/y", &["x", "/\n/", "y"]),
            (".//\n\n", &[".//\n\n"]),
            ("x = y;", &["x", " =", " y", ";"]),
            // Whitespace: through the last newline, else all but the last before a word.
            ("hello\n\nworld", &["hello", "\n\n", "world"]),
            ("a  b", &["a", " ", " b"]),
            ("end ", &["end", " "]),
            ("a \r\n b", &["a", " \r\n", " b"]),
            // A letter outside ASCII is a capital and a lowercase letter at once: the capitals give back to it.
            ("API를", &["API를"]),
            ("中文ABC", &["中文", "ABC"]),
            ("한AB", &["한", "AB"]),
            ("你好 世界", &["你好", " 世界"]),
            ("한국어 텍스트", &["한국어", " 텍스트"]),
            ("Ünïcödé тест", &["Ünïcödé", " тест"]),
            // A mark is no letter: it opens the word after it or joins a symbol run.
            ("e\u{301}x", &["e", "\u{301}x"]),
            ("\u{301}.", &["\u{301}."]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::MiniMaxM2, text), want, "{text:?}");
        }
    }

    /// The gemma4 split cuts only between U+000A and the rest: spaces, tabs,
    /// `\r` and punctuation stay inside their run.
    #[test]
    fn gemma4_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("a b, c", &["a b, c"]),
            ("\n\n\n", &["\n\n\n"]),
            ("a\n\nb\r\n", &["a", "\n\n", "b\r", "\n"]),
            ("\n\tx \n", &["\n", "\tx ", "\n"]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::Gemma4, text), want, "{text:?}");
        }
    }
}
