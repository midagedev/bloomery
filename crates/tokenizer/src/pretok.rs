//! The pre-tokenizers of the vocabularies this crate runs, as the reference
//! runs them: [`Pre::DeepseekV3`] (`deepseek-v3`, `hunyuan-dense`,
//! `joyai-llm`), [`Pre::Qwen2`] (`qwen2`, `solar-open`), [`Pre::Glm4`] (`glm4`),
//! [`Pre::Qwen35`] (`qwen35`), [`Pre::Llama3`] (`lfm2`), [`Pre::Tekken`] (`tekken`, `pixtral`),
//! [`Pre::TekkenNum3`] (`tekken-num3`), [`Pre::MiniMaxM2`] (`minimax-m2`), [`Pre::Gpt4o`] (`gpt-4o`),
//! [`Pre::Gemma4`] (`gemma4`), [`Pre::Raon`] (`raon`) and [`Pre::Flint`] (`flint`).
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
//! A Hugging Face `tokenizer.json` runs the same regex with Oniguruma, as it does qwen35's:
//! `(?i:...)` folds case by Unicode (`'s` also matches `'ſ`) and the classes are Unicode 14.0's
//! ([`Pre::Qwen2Hf`]). The regex has no `\p{M}`, so a mark is a plain codepoint of the symbol
//! class there, as it is in the reference's splitter, and every position matches an alternative:
//! the one-codepoint piece the splitter makes of a position none matches never arises.
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
//! [`match_qwen`] is that regex under ECMAScript semantics, written out by
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
//! ## llama3
//!
//! The reference maps `lfm2` to its LLAMA3 type. That type's one regex is
//! the one [`Pre::Glm4`] runs — qwen2's with `\p{N}{1,3}` in place of `\p{N}` —
//! and it goes to the same hand-written splitter (`unicode_regex_split_custom_llama3`),
//! so [`split`] uses [`split_custom`] with a run of three numbers, the `glm4`
//! arm. The reference also sets `ignore_merges` for this type and not for
//! `glm4`: a pre-tokenizer word whose byte-encoded text is itself a vocabulary
//! token is that token, and merges do not run on it. [`Pre::ignore_merges`]
//! carries the flag for any pre, so the two types do not share a variant.
//!
//! A Hugging Face `tokenizer.json` of this type (Llama 3's and its descendants': a `Split` on this
//! regex, `Isolated`, then `ByteLevel` without a regex of its own, and `model.ignore_merges` true)
//! runs it with Oniguruma, as qwen2's: Unicode case folds in `(?i:...)` and Unicode 14.0's
//! classes. [`Pre::Llama3Hf`] is that reading, qwen2's with the run of numbers cut in threes, for
//! a vocabulary that declares itself a Hugging Face one.
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
//! here ([`match_cased_words`]). Past the words the alternatives are qwen2's, but
//! the symbol run's tail is `[\r\n/]*`, a slash as well as the newlines. The
//! reference sets `clean_spaces` false for this name; this crate cleans no spaces.
//!
//! ## tekken
//!
//! The reference maps `tekken` and `pixtral` to its TEKKEN type, whose one regex is
//!
//! `[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+|
//! [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*|
//! \p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`
//!
//! and sets `ignore_merges`. It is [`Pre::MiniMaxM2`]'s regex without the contractions, one number
//! a word instead of three, and `\p{M}` beside `\p{L}` in the lookaheads, so it runs through the
//! same matcher ([`match_cased_words`] with [`CasedShape::TEKKEN`]) on the collapsed text with the
//! same piece-local semantics. A mark is a letter of either case there ([`TekkenReference`]): it
//! joins the word beside it, and a lone one is a word. A tree older than the one that gives
//! `pixtral` this type (ik is one) reads the name as LLAMA3, which cuts digits in threes and keeps
//! `HelloWorld` whole; ik's TEKKEN looks ahead at `\p{L}` alone, so a mark is no word there.
//!
//! A Hugging Face `tokenizer.json` spells the classes by category
//! (`[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+` and the capitals-only twin) and
//! runs them with Oniguruma over Unicode 14.0's classes: [`Pre::TekkenHf`] reads them as
//! [`Pre::MiniMaxM2Hf`] does (a non-ASCII capital is no lowercase letter there), with no
//! contraction.
//!
//! `tekken-num3` is the same regex with the number run cut in words of three from its end
//! (`\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))`, raon's rule), so [`Pre::TekkenNum3`] runs the same
//! matcher with that one [`DigitRule`] and no Hugging Face twin: Mistral publishes no
//! `tokenizer.json` that spells it.
//!
//! ## raon
//!
//! One regex, which the reference runs as the hand-written
//! `unicode_regex_split_custom_raon` (and which its `std::regex` form would split alike but for
//! the one-codepoint piece a position no alternative matches becomes): a run of Han, hiragana and
//! katakana (`U+4E00..U+9FA5`, `U+3040..U+30FF`) behind an optional space; a run of letters that
//! are none of those, each with its marks, whose words a single space may join; digits cut into
//! words of three from the run's end, the first one the shorter; a run of punctuation behind an
//! optional space with the newlines and slashes after it; a run of whitespace ending in a
//! newline; a whitespace run but its last space when a word follows. The reference applies NFC
//! before this ([`Pre::nfc`]). [`split_raon`] is that function line for line.
//!
//! ## flint
//!
//! Three ordered passes: `\p{N}{1,3}`, `[一-龥぀-ゟ゠-ヿ]+`, then case/mark
//! words with optional contractions, punctuation/symbol runs and whitespace.
//! The first two passes share [`split_ordered`] with deepseek-v3; they isolate
//! numbers and CJK before the final pass can attach a prefix or contraction.
//! In mainline a37cc46104eb50900d5ab4a9ba02a0abdac039db, `unicode_regex_split`
//! maps every letter subcategory (`Lu`, `Ll`, `Lt`, `Lm`, `Lo`) to `L`, even
//! for ASCII. The two word alternatives therefore reduce to
//! `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` plus an optional ASCII contraction.
//! [`match_flint`] implements that effective program, not GPT4O's case split.
//! Official Hugging Face uses true subcategories and Unicode contraction folding:
//! the pretokenizer boundaries of `HelloWorld` and `x'ſy` differ. Matching final
//! IDs with the pinned vocabulary does not establish identical split programs.
//! No normalization or prefix space is added.
//!
//! ## gemma4
//!
//! One regex, `[^\n]+|[\n]+`, which the reference runs as the hand-written
//! `unicode_regex_split_custom_newlines`: the fragment is cut into runs of
//! U+000A and runs of everything else. `\r` is not a newline here. The words
//! are not GPT-2 byte-level: the reference writes each space as U+2581 before
//! the split and merges raw UTF-8 characters ([`Spelling::Metaspace`]).

use crate::spelling::Spelling;
use crate::unicode::{collapse, collapse_hf, letter_case};

const NUMBER: u8 = 0xD1;
const LETTER: u8 = 0xD2;
const PUNCT: u8 = 0xD3;
const MARK: u8 = 0xD4;
const SYMBOL: u8 = 0xD5;
/// A letter of Unicode 14.0's `Lu` or `Lt`, and one of its `Ll`, where a regex tells them apart
/// ([`collapse_cased`]); `Lm` and `Lo` stay [`LETTER`], a member of both.
const UPPER_LETTER: u8 = 0xD6;
const LOWER_LETTER: u8 = 0xD7;

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

/// `\p{N}{1,max}` at `i`: one to `max` numbers, 0 for none.
fn match_number(c: &[u8], i: usize, hi: usize, max: usize) -> usize {
    run(c, i, hi, is_number).min(max)
}

/// How a regex's number alternative cuts a run of numbers.
#[derive(Clone, Copy)]
enum DigitRule {
    /// No number alternative: an earlier pass took the numbers.
    None,
    /// `\p{N}{1,n}`: at most `n` numbers, from the run's start.
    Front(usize),
    /// `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))`: the run in words of three from
    /// its end, the first one the shorter. The `\p{N}{1,2}` beside it in the
    /// regex never runs: at the start of a run of `n` numbers some prefix of
    /// one to three always leaves a multiple of three.
    ThreesFromEnd,
}

impl DigitRule {
    /// The alternative's match at `i`: its length, 0 for none.
    fn at(self, c: &[u8], i: usize, hi: usize) -> usize {
        match self {
            DigitRule::None => 0,
            DigitRule::Front(n) => match_number(c, i, hi, n),
            DigitRule::ThreesFromEnd => {
                let n = run(c, i, hi, is_number);
                match n % 3 {
                    0 if n == 0 => 0,
                    0 => 3,
                    first => first,
                }
            }
        }
    }
}

/// `[^\s\p{L}\p{N}]`: the "other" run's class of qwen2 and glm4.
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

/// `P?W+` at `i`, the word alternative of qwen35 (`W` letters and marks), qwen2's regex (`W`
/// letters) and deepseek-v3's regex 3: the prefix first (a greedy `?` backs off only when the run
/// after it fails), then the run alone. `prefix(x)` is whether `x` is a codepoint of `P`, and
/// `word(x)` whether it is one of `W`.
fn match_prefixed_word(
    c: &[u8],
    i: usize,
    hi: usize,
    prefix: impl Fn(u8) -> bool,
    word: impl Fn(u8) -> bool + Copy,
) -> usize {
    if prefix(c[i]) && i + 1 < hi {
        let n = run(c, i + 1, hi, word);
        if n > 0 {
            return 1 + n;
        }
    }
    run(c, i, hi, word)
}

/// The alternatives every regex here ends with, at `i`: the number rule `digits`,
/// ` ?[other]+[tail]*` and `\s*[\r\n]+|\s+(?!\S)|\s+`. A space is none of `other`, so backing
/// the optional space off never helps.
fn match_tail(
    c: &[u8],
    i: usize,
    hi: usize,
    digits: DigitRule,
    other: impl Fn(u8) -> bool,
    tail: impl Fn(u8) -> bool,
) -> usize {
    let n = digits.at(c, i, hi);
    if n > 0 {
        return n;
    }
    let start = if c[i] == b' ' { i + 1 } else { i };
    if start < hi {
        let n = run(c, start, hi, other);
        if n > 0 {
            let end = start + n;
            return end + run(c, end, hi, tail) - i;
        }
    }
    match_space(c, i, hi)
}

/// The `qwen35` regex at `i` (`word` letters or marks, `other` [`is_other_qwen35`]) or the
/// `qwen2` regex (`word` letters alone, `other` [`is_other`], the marks among its codepoints):
/// the first of the alternatives that matches, under ECMAScript semantics (alternatives in order,
/// greedy quantifiers that back off). None of them can match the empty string, and a mark is a
/// plain codepoint of its own class here: the regex runs on the collapsed bytes. `fold(k)` is the
/// lowercase ASCII letter the codepoint at `k` matches in a contraction; `digits` is the longest
/// run of numbers one word holds (`\p{N}` is 1, `\p{N}{1,3}` is 3).
fn match_qwen(
    c: &[u8],
    i: usize,
    hi: usize,
    fold: impl Fn(usize) -> u8,
    word: impl Fn(u8) -> bool + Copy,
    other: impl Fn(u8) -> bool,
    digits: usize,
) -> usize {
    // `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`
    let n = contraction(c, i, hi, fold);
    if n > 0 {
        return n;
    }

    // `[^\r\n\p{L}\p{N}]?W+`, `W` the word class: the prefix class takes a mark, a space or an
    // apostrophe as readily under either regex.
    let n = match_prefixed_word(
        c,
        i,
        hi,
        |x| !is_newline(x) && !is_letter(x) && !is_number(x),
        word,
    );
    if n > 0 {
        return n;
    }

    // `\p{N}` or `\p{N}{1,3}` (up to `digits` numbers), ` ?[^\s<word>\p{N}]+[\r\n]*` and
    // `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_tail(c, i, hi, DigitRule::Front(digits), other, is_newline)
}

/// Regex 3 at `i`: the first of its six alternatives that matches.
fn match_word(c: &[u8], i: usize, hi: usize) -> usize {
    // `[<ascii punctuation>][A-Za-z]+` — the class is written out literally
    // and holds every ASCII punctuation byte, `~` included.
    if c[i].is_ascii_punctuation() && i + 1 < hi {
        let n = run(c, i + 1, hi, |x| x.is_ascii_alphabetic());
        if n > 0 {
            return 1 + n;
        }
    }

    // `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
    let n = match_prefixed_word(
        c,
        i,
        hi,
        |x| !is_newline(x) && !is_letter(x) && !is_punct_or_symbol(x),
        is_letter_or_mark,
    );
    if n > 0 {
        return n;
    }

    // ` ?[\p{P}\p{S}]+[\r\n]*` and `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_tail(c, i, hi, DigitRule::None, is_punct_or_symbol, is_newline)
}

/// [`collapse_hf`], with a letter of Unicode 14.0 told by its category: `Lu` and `Lt` collapse to
/// [`UPPER_LETTER`], `Ll` to [`LOWER_LETTER`], `Lm` and `Lo` stay [`LETTER`].
fn collapse_cased(cpt: u32) -> u8 {
    match collapse_hf(cpt) {
        LETTER => match letter_case(cpt) {
            1 => UPPER_LETTER,
            2 => LOWER_LETTER,
            _ => LETTER,
        },
        c => c,
    }
}

/// `\p{L}` over [`collapse_cased`]'s bytes.
fn is_letter_cased(c: u8) -> bool {
    matches!(c, LETTER | UPPER_LETTER | LOWER_LETTER) || c.is_ascii_alphabetic()
}

/// What a reading of the `minimax-m2` or `tekken` regex tells apart in a letter. Their word
/// alternatives are `P? U* L+ C?` and `P? U+ L* C?` for the classes below, `C` a contraction
/// (`tekken` has none, [`CasedShape`]) and `P` the prefix `[^\r\n\p{L}\p{N}]` (a mark is one).
trait CasedLetters {
    /// Whether the reading folds `ſ` (U+017F) to `s` in a contraction, as Hugging Face's
    /// Unicode folding does.
    const HF: bool;

    /// `\p{L}`.
    fn letter(c: u8) -> bool;
    /// `U`, which the word alternatives open with.
    fn upper(c: u8) -> bool;
    /// `L`, which they continue with.
    fn lower(c: u8) -> bool;
}

/// The regex as the reference spells it, `((?=[\p{L}])([^a-z]))` for `U` and
/// `((?=[\p{L}])([^A-Z]))` for `L`, over [`collapse`]'s bytes. A non-ASCII letter is one byte,
/// [`LETTER`], that is in neither `a-z` nor `A-Z`, so it is `U` and `L` at once; a mark is no
/// letter here.
struct M2Reference;

impl CasedLetters for M2Reference {
    const HF: bool = false;

    fn letter(c: u8) -> bool {
        is_letter(c)
    }

    fn upper(c: u8) -> bool {
        c.is_ascii_uppercase() || c == LETTER
    }

    fn lower(c: u8) -> bool {
        c.is_ascii_lowercase() || c == LETTER
    }
}

/// The `tekken` regex as the reference spells it, `((?=[\p{L}\p{M}])([^a-z]))` for `U` and
/// `((?=[\p{L}\p{M}])([^A-Z]))` for `L`, over [`collapse`]'s bytes: a non-ASCII letter and a mark
/// are each one byte that is in neither `a-z` nor `A-Z`, so each is `U` and `L` at once. A mark is
/// no `\p{L}`, so it is a prefix too ([`M2Reference`] differs in the mark alone).
struct TekkenReference;

impl CasedLetters for TekkenReference {
    const HF: bool = false;

    fn letter(c: u8) -> bool {
        is_letter(c)
    }

    fn upper(c: u8) -> bool {
        c.is_ascii_uppercase() || matches!(c, LETTER | MARK)
    }

    fn lower(c: u8) -> bool {
        c.is_ascii_lowercase() || matches!(c, LETTER | MARK)
    }
}

/// The regex as a `tokenizer.json` spells it (MiniMax's, Tekken's), over [`collapse_cased`]'s
/// bytes: `U` is `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]` and `L` is `[\p{Ll}\p{Lm}\p{Lo}\p{M}]`, a mark
/// in both.
struct CasedHf;

impl CasedLetters for CasedHf {
    const HF: bool = true;

    fn letter(c: u8) -> bool {
        is_letter_cased(c)
    }

    fn upper(c: u8) -> bool {
        matches!(c, LETTER | UPPER_LETTER | MARK) || c.is_ascii_uppercase()
    }

    fn lower(c: u8) -> bool {
        matches!(c, LETTER | LOWER_LETTER | MARK) || c.is_ascii_lowercase()
    }
}

/// What the case-split regexes differ in beside their letters.
#[derive(Clone, Copy)]
struct CasedShape {
    /// Whether a word may end in a contraction (`minimax-m2` does).
    contraction: bool,
    /// How the run of numbers one word opens is cut ([`DigitRule`]).
    digits: DigitRule,
}

impl CasedShape {
    const MINIMAX_M2: Self = Self {
        contraction: true,
        digits: DigitRule::Front(3),
    };
    const TEKKEN: Self = Self {
        contraction: false,
        digits: DigitRule::Front(1),
    };
    const TEKKEN_NUM3: Self = Self {
        contraction: false,
        digits: DigitRule::ThreesFromEnd,
    };
}

/// The `minimax-m2` regex, or with `shape` the `tekken` one, at `i`, as the reading `K` spells it:
/// the first of its seven alternatives that matches, under ECMAScript semantics (alternatives in
/// order, greedy quantifiers that back off, the optional prefix taken first). None of them can
/// match the empty string, and a mark is a plain codepoint of its own class: the regex runs on the
/// collapsed bytes. `fold(k)` is the lowercase ASCII letter the codepoint at `k` matches in a
/// contraction.
///
/// A mark can be both a prefix and a word codepoint, so the prefix is tried taken and not taken.
/// `U*` gives back until `L+` can start: at the lowercase letter that ended the run, else at the
/// run's last codepoint that is an `L` too (a capital is not). The second word alternative runs
/// only when the first did not: capitals alone. Past the words the alternatives are qwen2's, but
/// the symbol run's tail is `[\r\n/]*`, a slash as well as the newlines.
fn match_cased_words<K: CasedLetters>(
    c: &[u8],
    i: usize,
    hi: usize,
    shape: CasedShape,
    fold: impl Fn(usize) -> u8,
) -> usize {
    let x = c[i];
    let starts: &[usize] = if !is_newline(x) && !K::letter(x) && !is_number(x) && i + 1 < hi {
        &[i + 1, i]
    } else {
        &[i]
    };
    let contract = |end: usize| {
        let tail = if shape.contraction {
            contraction(c, end, hi, &fold)
        } else {
            0
        };
        end + tail - i
    };
    // `P? U* L+ C?`
    for &s in starts {
        let ups = run(c, s, hi, K::upper);
        let from = if s + ups < hi && K::lower(c[s + ups]) {
            Some(s + ups)
        } else {
            c[s..s + ups]
                .iter()
                .rposition(|&b| K::lower(b))
                .map(|p| s + p)
        };
        if let Some(from) = from {
            return contract(from + run(c, from, hi, K::lower));
        }
    }
    // `P? U+ L* C?`
    for &s in starts {
        let ups = run(c, s, hi, K::upper);
        if ups > 0 {
            return contract(s + ups + run(c, s + ups, hi, K::lower));
        }
    }

    // The number rule (`\p{N}` for `tekken`, `\p{N}{1,3}` for `minimax-m2`, the threes from the
    // end for `tekken-num3`), ` ?[^\s\p{L}\p{N}]+[\r\n/]*` and `\s*[\r\n]+|\s+(?!\S)|\s+`
    match_tail(
        c,
        i,
        hi,
        shape.digits,
        |b| !is_space(b) && !K::letter(b) && !is_number(b),
        |b| is_newline(b) || b == b'/',
    )
}

/// Flint's final regex on collapsed text: both case classes are `L|M` in
/// the reference, so their ordered greedy alternatives consume the same run.
fn match_flint(c: &[u8], i: usize, hi: usize) -> usize {
    let x = c[i];
    let prefix = !is_newline(x) && !is_letter(x) && !is_number(x);
    let start = i + usize::from(prefix);
    let n = run(c, start, hi, is_letter_or_mark);
    let end = if n > 0 {
        start + n
    } else {
        // A mark may be both prefix and word; the optional prefix backs off.
        i + run(c, i, hi, is_letter_or_mark)
    };
    if end > i {
        return end + contraction(c, end, hi, |k| c[k].to_ascii_lowercase()) - i;
    }
    let start = i + usize::from(x == b' ');
    let n = run(c, start, hi, is_punct_or_symbol);
    if n > 0 {
        let end = start + n;
        return end + run(c, end, hi, is_newline) - i;
    }
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
    /// `qwen2` as a Hugging Face `tokenizer.json` runs it: contractions fold case by Unicode,
    /// classes are Unicode 14.0's, and BPE skips a byte its vocabulary lacks. Never named in a
    /// header; a vocabulary with `tokenizer.bloomery.eog_token_ids` gets it.
    Qwen2Hf,
    /// `lfm2`: the llama3 regex [`Pre::Glm4`] runs, and a word that is
    /// itself a token is taken whole ([`Pre::ignore_merges`]).
    Llama3,
    /// `lfm2` as a Hugging Face `tokenizer.json` runs it: contractions fold case by Unicode,
    /// classes are Unicode 14.0's, and BPE skips a byte its vocabulary lacks; a word that is itself
    /// a token is still taken whole. Never named in a header; a vocabulary with
    /// `tokenizer.bloomery.eog_token_ids` gets it.
    Llama3Hf,
    /// `tekken`, `pixtral`: the reference's TEKKEN regex, run as a regex over the collapsed text,
    /// and a word that is itself a token is taken whole ([`Pre::ignore_merges`]).
    Tekken,
    /// `tekken-num3`: [`Pre::Tekken`]'s regex with the number run cut in words of three from its
    /// end, and a word that is itself a token is taken whole ([`Pre::ignore_merges`]).
    TekkenNum3,
    /// `tekken` as a Hugging Face `tokenizer.json` spells and runs it: the letter categories apart,
    /// `\p{M}` in the word classes, Unicode 14.0's classes, and BPE skips a byte its vocabulary
    /// lacks; a word that is itself a token is still taken whole. Never named in a header; a
    /// vocabulary with `tokenizer.bloomery.eog_token_ids` gets it.
    TekkenHf,
    /// `minimax-m2`: GPT4O's one regex, run as a regex over the collapsed text.
    MiniMaxM2,
    /// `gpt-4o`: [`Pre::MiniMaxM2`]'s regex under the name of a tree whose UTF-8 reader is the
    /// strict one ([`Pre::strict_utf8`]).
    Gpt4o,
    /// `minimax-m2` as a Hugging Face `tokenizer.json` spells and runs it: the letter
    /// categories apart, `\p{M}` in the word classes, and Unicode 14.0's classes. Never named
    /// in a header; a vocabulary with `tokenizer.bloomery.eog_token_ids` gets it.
    MiniMaxM2Hf,
    /// `gemma4`: runs of newlines and runs of the rest, over metaspace text.
    Gemma4,
    /// `raon`: the reference's hand-written splitter with the CJK run, the letter run joined by
    /// single spaces and the digits in threes from the end; the text is NFC-normalized first.
    Raon,
    /// `flint`: numbers, CJK, then the reference's collapsed case/mark regex.
    Flint,
}

impl Pre {
    /// The names this crate runs, as `tokenizer.ggml.pre` spells them.
    pub(crate) const NAMES: [(&'static str, Pre); 16] = [
        ("deepseek-v3", Pre::DeepseekV3),
        ("hunyuan-dense", Pre::DeepseekV3),
        ("joyai-llm", Pre::DeepseekV3),
        ("qwen2", Pre::Qwen2),
        ("solar-open", Pre::Qwen2),
        ("glm4", Pre::Glm4),
        ("qwen35", Pre::Qwen35),
        ("tekken", Pre::Tekken),
        ("tekken-num3", Pre::TekkenNum3),
        ("pixtral", Pre::Tekken),
        ("lfm2", Pre::Llama3),
        ("minimax-m2", Pre::MiniMaxM2),
        ("gpt-4o", Pre::Gpt4o),
        ("gemma4", Pre::Gemma4),
        ("raon", Pre::Raon),
        ("flint", Pre::Flint),
    ];

    /// The reading of this pre-tokenizer that runs as a Hugging Face `tokenizer.json` does, its
    /// own case; `None` for a name no folder's `tokenizer.json` spells.
    pub(crate) fn hf_reading(self) -> Option<Pre> {
        match self {
            Pre::Qwen2 => Some(Pre::Qwen2Hf),
            Pre::Qwen35 => Some(Pre::Qwen35Hf),
            Pre::Llama3 => Some(Pre::Llama3Hf),
            Pre::Tekken => Some(Pre::TekkenHf),
            Pre::MiniMaxM2 | Pre::Gpt4o => Some(Pre::MiniMaxM2Hf),
            _ => None,
        }
    }

    /// The names in [`Pre::NAMES`] that have an HF reading, comma-separated.
    pub(crate) fn hf_names() -> String {
        let names: Vec<&str> = Self::NAMES
            .iter()
            .filter(|(_, pre)| pre.hf_reading().is_some())
            .map(|&(name, _)| name)
            .collect();
        names.join(", ")
    }

    /// Whether a word that is itself a vocabulary token is taken whole,
    /// before merges. Any pre can carry it; `lfm2`, `tekken` and `tekken-num3` do, the first two
    /// in either reading.
    pub(crate) fn ignore_merges(self) -> bool {
        matches!(
            self,
            Pre::Llama3 | Pre::Llama3Hf | Pre::Tekken | Pre::TekkenHf | Pre::TekkenNum3
        )
    }

    /// Whether the reference NFC-normalizes the text before this pre-tokenizer whatever the
    /// header's normalizer says: the Hugging Face tokenizer of `raon` applies it.
    pub(crate) fn nfc(self) -> bool {
        matches!(self, Pre::Raon)
    }

    /// Whether the reference decodes UTF-8 strictly, independently of NFC.
    /// An overlong form is not a codepoint there; older readers accept it. The trees that read
    /// `gpt-4o`, `raon`, `flint`, `tekken` (every tree that maps `pixtral` to it has the check)
    /// and `tekken-num3` (a name younger than the check) reject it; the trees the other names'
    /// sets are read by (ik, the Solar-Open2 patch, mainline before the check) accept it.
    pub(crate) fn strict_utf8(self) -> bool {
        matches!(
            self,
            Pre::Raon | Pre::Flint | Pre::Gpt4o | Pre::Tekken | Pre::TekkenNum3
        )
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
            | Pre::Qwen2Hf
            | Pre::MiniMaxM2Hf
            | Pre::Llama3
            | Pre::Llama3Hf
            | Pre::Tekken
            | Pre::TekkenHf
            | Pre::TekkenNum3
            | Pre::MiniMaxM2
            | Pre::Gpt4o
            | Pre::Raon
            | Pre::Flint => Spelling::ByteLevel,
        }
    }
}

/// Split one text fragment, given as codepoints, into pre-tokenizer words
/// under `pre`; returns each word's length in codepoints, in order.
pub(crate) fn split<'s>(pre: Pre, cpts: &[u32], s: &'s mut Scratch) -> &'s [usize] {
    s.collapsed.clear();
    let class: fn(u32) -> u8 = match pre {
        Pre::Qwen35Hf | Pre::Qwen2Hf | Pre::Llama3Hf => collapse_hf,
        Pre::MiniMaxM2Hf | Pre::TekkenHf => collapse_cased,
        _ => collapse,
    };
    s.collapsed.extend(cpts.iter().map(|&cp| class(cp)));
    match pre {
        Pre::DeepseekV3 => split_ordered(cpts, s, match_word),
        Pre::Flint => split_ordered(cpts, s, match_flint),
        Pre::Qwen2 => {
            split_custom(cpts, &s.collapsed, 1, &mut s.b);
            &s.b
        }
        Pre::Glm4 | Pre::Llama3 => {
            split_custom(cpts, &s.collapsed, 3, &mut s.b);
            &s.b
        }
        Pre::Qwen35 => split_qwen35(cpts, s, false),
        Pre::Qwen35Hf => split_qwen35(cpts, s, true),
        Pre::Qwen2Hf => split_qwen2_hf(cpts, s, 1),
        Pre::Llama3Hf => split_qwen2_hf(cpts, s, 3),
        Pre::MiniMaxM2 | Pre::Gpt4o => split_cased::<M2Reference>(cpts, s, CasedShape::MINIMAX_M2),
        Pre::MiniMaxM2Hf => split_cased::<CasedHf>(cpts, s, CasedShape::MINIMAX_M2),
        Pre::Tekken => split_cased::<TekkenReference>(cpts, s, CasedShape::TEKKEN),
        Pre::TekkenHf => split_cased::<CasedHf>(cpts, s, CasedShape::TEKKEN),
        Pre::TekkenNum3 => split_cased::<TekkenReference>(cpts, s, CasedShape::TEKKEN_NUM3),
        Pre::Gemma4 => {
            split_newlines(cpts, &mut s.b);
            &s.b
        }
        Pre::Raon => {
            split_raon(cpts, &s.collapsed, &mut s.b);
            &s.b
        }
    }
}

/// `unicode_regex_split_custom_raon` over one fragment: `cpts` its codepoints, `c` their
/// collapsed bytes; the words' lengths go to `out`. Each branch is the reference's, in its order.
fn split_raon(cpts: &[u32], c: &[u8], out: &mut Vec<usize>) {
    out.clear();
    let end = cpts.len();
    let cjk = |p: usize| p < end && is_cjk(cpts[p]);
    // The reference's flags read as empty past the fragment.
    let flag_letter = |p: usize| p < end && is_letter(c[p]);
    let letter = |p: usize| flag_letter(p) && !cjk(p);
    let mark = |p: usize| p < end && c[p] == MARK;
    let number = |p: usize| p < end && is_number(c[p]);
    let space = |p: usize| p < end && is_space(c[p]);
    let punct = |p: usize| p < end && !space(p) && !flag_letter(p) && !number(p);
    let mut pos = 0;
    while pos < end {
        let start = pos;
        let mut p = pos + usize::from(cpts[pos] == u32::from(b' '));
        if cjk(p) {
            // ` ?[CJK]+`
            while cjk(p) {
                p += 1;
            }
            pos = p;
        } else if letter(p) {
            // ` ?(letter mark*)+( (letter mark*)+)*`
            loop {
                p += 1;
                while mark(p) {
                    p += 1;
                }
                if letter(p) {
                    continue;
                }
                if p < end && cpts[p] == u32::from(b' ') && letter(p + 1) {
                    p += 1;
                    continue;
                }
                break;
            }
            pos = p;
        } else if number(pos) {
            // `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))`.
            pos += DigitRule::ThreesFromEnd.at(c, pos, end);
        } else if punct(p) {
            // ` ?[^\s\p{L}\p{N}]+[\r\n/]*`
            while punct(p) {
                p += 1;
            }
            while p < end && (is_newline(c[p]) || cpts[p] == u32::from(b'/')) {
                p += 1;
            }
            pos = p;
        } else {
            // `\s*[\r\n]+`, `\s+(?!\S)`, `\s+`
            p = pos;
            let mut last_newline = pos;
            while space(p) {
                if is_newline(c[p]) {
                    last_newline = p + 1;
                }
                p += 1;
            }
            pos = if last_newline > pos {
                last_newline
            } else if p > pos {
                if p < end && p - pos > 1 { p - 1 } else { p }
            } else {
                pos + 1
            };
        }
        out.push(pos - start);
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

/// The lowercase ASCII letter a contraction compares the codepoint at `k` with: its ASCII
/// lowercase, and under Hugging Face's Unicode folding (`hf`) `ſ` as `s`.
fn fold_at(cpts: &[u32], c: &[u8], k: usize, hf: bool) -> u8 {
    if hf && cpts[k] == 0x17F {
        b's'
    } else {
        c[k].to_ascii_lowercase()
    }
}

/// [`split`] for [`Pre::Qwen35`], and with `hf` for [`Pre::Qwen35Hf`]: its one
/// regex over the collapsed bytes.
fn split_qwen35<'s>(cpts: &[u32], s: &'s mut Scratch, hf: bool) -> &'s [usize] {
    split_one_piece(cpts.len(), s, |c, i, hi| {
        match_qwen(
            c,
            i,
            hi,
            |k| fold_at(cpts, c, k, hf),
            is_letter_or_mark,
            is_other_qwen35,
            1,
        )
    });
    &s.b
}

/// [`split`] for [`Pre::Qwen2Hf`] (`digits` 1) and [`Pre::Llama3Hf`] (`digits` 3): the `qwen2`
/// regex, or its llama3 form, over the collapsed bytes, contractions folded as Hugging Face's.
fn split_qwen2_hf<'s>(cpts: &[u32], s: &'s mut Scratch, digits: usize) -> &'s [usize] {
    split_one_piece(cpts.len(), s, |c, i, hi| {
        match_qwen(
            c,
            i,
            hi,
            |k| fold_at(cpts, c, k, true),
            is_letter,
            is_other,
            digits,
        )
    });
    &s.b
}

/// [`split`] for [`Pre::MiniMaxM2`] (`shape` [`CasedShape::MINIMAX_M2`]) and [`Pre::Tekken`]
/// (`shape` [`CasedShape::TEKKEN`]), and with the reading `K` for [`Pre::MiniMaxM2Hf`] and
/// [`Pre::TekkenHf`]: their one regex over the collapsed bytes.
fn split_cased<'s, K: CasedLetters>(
    cpts: &[u32],
    s: &'s mut Scratch,
    shape: CasedShape,
) -> &'s [usize] {
    split_one_piece(cpts.len(), s, |c, i, hi| {
        match_cased_words::<K>(c, i, hi, shape, |k| fold_at(cpts, c, k, K::HF))
    });
    &s.b
}

/// Numbers, then CJK, then the vocabulary's final word regex, each pass
/// re-splitting only the pieces the previous pass left.
fn split_ordered<'s>(
    cpts: &[u32],
    s: &'s mut Scratch,
    word: impl Fn(&[u8], usize, usize) -> usize,
) -> &'s [usize] {
    let c = &s.collapsed;

    s.a.clear();
    if !cpts.is_empty() {
        s.a.push(cpts.len());
    }
    resplit(&s.a, &mut s.b, |i, hi| match_number(c, i, hi, 3));
    resplit(&s.b, &mut s.a, |i, hi| {
        cpts[i..hi].iter().take_while(|&&cp| is_cjk(cp)).count()
    });
    resplit(&s.a, &mut s.b, |i, hi| word(c, i, hi));
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
        let refusal = crate::Error::UnsupportedPre("o200k_harmony".to_string()).to_string();
        for (name, _) in Pre::NAMES {
            assert!(crate::runs_pre_tokenizer(name), "{name}");
            assert!(refusal.contains(name), "{name} not in: {refusal}");
        }
        assert!(!crate::runs_pre_tokenizer("o200k_harmony"));
    }

    /// `solar-open` runs the qwen2 splitter: Solar-Open2's `tokenizer.json`
    /// pre-tokenizer is qwen2's regex, character for character.
    #[test]
    fn solar_open_is_qwen2() {
        assert_eq!(crate::named(&Pre::NAMES, "solar-open"), Some(Pre::Qwen2));
    }

    /// `lfm2` runs the llama3 splitter: mainline maps `tokenizer.ggml.pre = lfm2` to its
    /// Llama-3 expression with `ignore_merges`, which `Pre::Llama3` is. LFM3's header spells it
    /// so. FAIL-first: a row that names another splitter, or none, fails it.
    #[test]
    fn lfm2_is_llama3() {
        assert_eq!(crate::named(&Pre::NAMES, "lfm2"), Some(Pre::Llama3));
        assert!(crate::runs_pre_tokenizer("lfm2"));
    }

    /// `gpt-4o` is a row of its own (the UTF-8 reader of its tree is the strict one) that runs the
    /// minimax-m2 splitter: the reference maps both names to GPT4O's one expression, so Muse
    /// Spark's `gpt-4o` header splits as `minimax-m2`'s does. `minimax_m2_cases` runs its whole
    /// table under both rows. FAIL-first: a `gpt-4o` row that runs another splitter fails that
    /// table.
    #[test]
    fn gpt_4o_is_minimax_m2() {
        assert_eq!(crate::named(&Pre::NAMES, "gpt-4o"), Some(Pre::Gpt4o));
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

    /// Hand-derived pieces of the ordered program, including piece-local
    /// whitespace/contractions and the reference's loss of case distinctions.
    #[test]
    fn flint_ordered_splits() {
        let cases: &[(&str, &[&str])] = &[
            ("1234567", &["123", "456", "7"]),
            ("abc1234def", &["abc", "123", "4", "def"]),
            ("x²³ ١٢٣٤", &["x", "²³", " ", "١٢٣", "٤"]),
            (" 12345 ", &[" ", "123", "45", " "]),
            ("a中b", &["a", "中", "b"]),
            (" 日本語テストabc", &[" ", "日本語テスト", "abc"]),
            ("a \t中b", &["a", " \t", "中", "b"]),
            ("中'RE", &["中", "'RE"]),
            ("中\u{301}A", &["中", "\u{301}A"]),
            // These are the literal CJK ranges, not every Han character.
            ("\u{4e00}A\u{9fa5}B", &["\u{4e00}", "A", "\u{9fa5}", "B"]),
            ("\u{9fa6}A", &["\u{9fa6}A"]),
            ("\u{3040}A\u{30ff}B", &["\u{3040}", "A", "\u{30ff}", "B"]),
            ("HelloWorld", &["HelloWorld"]),
            ("iPhone ÜBERüber", &["iPhone", " ÜBERüber"]),
            ("A\u{301}B", &["A\u{301}B"]),
            (".\u{301}a", &[".\u{301}a"]),
            ("\u{301}.", &["\u{301}", "."]),
            ("it'sa WE'RE", &["it's", "a", " WE'RE"]),
            ("can't've", &["can't", "'ve"]),
            ("x'ſy", &["x", "'ſy"]),
            ("rock'n'roll", &["rock", "'n", "'roll"]),
            ("x/\n/y", &["x", "/\n", "/y"]),
            ("x = y;", &["x", " =", " y", ";"]),
            ("a  b", &["a", " ", " b"]),
            ("a \r\n b", &["a", " \r\n", " b"]),
            ("a\u{a0}\u{a0}b", &["a", "\u{a0}", "\u{a0}b"]),
            ("end \t", &["end", " \t"]),
            (" \n\t", &[" \n", "\t"]),
            // Unmatched runs are retained, not treated as punctuation.
            ("\0\0.", &["\0\0", "."]),
            ("", &[]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::Flint, text), want, "{text:?}");
        }
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

    /// The raon splitter's alternatives, one case each, as its reference function splits them: a
    /// space may join two letter runs but not a letter run and a number, a mark stays with its
    /// letter, Han and kana split from the Latin and Hangul letters that touch them, digits are cut
    /// in threes from the run's end, a slash or newline after punctuation joins it, and a whitespace
    /// run gives its last space to the word after it.
    ///
    /// FAIL-first: with the digits cut from the run's start ("1234567" as 123|456|7) or the
    /// letter run not joining across one space ("hello world" as two words) the rows below fail.
    #[test]
    fn raon_cases() {
        let cases: &[(&str, &[&str])] = &[
            ("hello world", &["hello world"]),
            ("hello  world", &["hello", " ", " world"]),
            ("it's", &["it", "'", "s"]),
            ("a\u{301}b c", &["a\u{301}b c"]),
            ("12", &["12"]),
            ("123", &["123"]),
            ("1234", &["1", "234"]),
            ("1234567", &["1", "234", "567"]),
            (" 12", &[" ", "12"]),
            ("abc1234def", &["abc", "1", "234", "def"]),
            ("한국어 텍스트", &["한국어 텍스트"]),
            ("日本語テスト", &["日本語テスト"]),
            (" 日本 語", &[" 日本", " 語"]),
            ("Hello,世界!", &["Hello", ",", "世界", "!"]),
            ("x = y;", &["x", " =", " y", ";"]),
            ("a/\nb", &["a", "/\n", "b"]),
            ("a  b", &["a", " ", " b"]),
            ("hello\n\nworld", &["hello", "\n\n", "world"]),
            ("end ", &["end", " "]),
            ("  ", &["  "]),
            ("(foo)", &["(", "foo", ")"]),
        ];
        for &(text, want) in cases {
            assert_eq!(words(Pre::Raon, text), want, "{text:?}");
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
            // `lfm2` is the same regex and the same splitter. What the
            // two names do not share is `ignore_merges`, which runs after
            // this split.
            assert_eq!(words(Pre::Llama3, text), want, "llama3 {text:?}");
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

    /// Hugging Face's qwen2 folds a contraction's case by Unicode as its qwen35 does (`ſ` is `s`,
    /// the Kelvin sign no contraction's letter) and tells the same codepoints of Unicode 15.0 from
    /// the letters and numbers the reference's tables class; a mark is no letter in either reading,
    /// so it opens a word as the prefix.
    ///
    /// FAIL-first: with the ASCII fold for both, the first three rows fail; with the reference's
    /// classes for both, the last two; with marks in the letter run, the mark rows.
    #[test]
    fn qwen2_hf_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              hf                        qwen2
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
            ("'\u{212a}x", &["'\u{212a}x"], &["'\u{212a}x"]),
            ("IT'S", &["IT", "'S"], &["IT", "'S"]),
            ("x\u{301}y", &["x", "\u{301}y"], &["x", "\u{301}y"]),
            (
                "x\u{301}\u{301}",
                &["x", "\u{301}\u{301}"],
                &["x", "\u{301}\u{301}"],
            ),
            (" \u{1d2c0}", &[" \u{1d2c0}"], &[" ", "\u{1d2c0}"]),
            ("x\u{11f04}", &["x", "\u{11f04}"], &["x\u{11f04}"]),
        ];
        for &(text, hf, legacy) in cases {
            assert_eq!(words(Pre::Qwen2Hf, text), hf, "hf {text:?}");
            assert_eq!(words(Pre::Qwen2, text), legacy, "qwen2 {text:?}");
        }
    }

    /// Hugging Face's llama3 regex is qwen2's with up to three numbers a word: it folds a
    /// contraction's case and classes the codepoints as its qwen2 reading does, and cuts a run of
    /// numbers into words of three from its start.
    ///
    /// FAIL-first: with the ASCII fold for both, the first row fails; with the reference's classes
    /// for both, the next two; with one number a word, the last two.
    #[test]
    fn llama3_hf_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              hf                        llama3
            ("x'\u{17f}y", &["x", "'\u{17f}", "y"], &["x", "'\u{17f}y"]),
            (" \u{1d2c0}", &[" \u{1d2c0}"], &[" ", "\u{1d2c0}"]),
            ("x\u{11f04}", &["x", "\u{11f04}"], &["x\u{11f04}"]),
            ("1234567", &["123", "456", "7"], &["123", "456", "7"]),
            (" 12345", &[" ", "123", "45"], &[" ", "123", "45"]),
        ];
        for &(text, hf, legacy) in cases {
            assert_eq!(words(Pre::Llama3Hf, text), hf, "hf {text:?}");
            assert_eq!(words(Pre::Llama3, text), legacy, "llama3 {text:?}");
        }
        assert!(Pre::Llama3Hf.ignore_merges());
        assert_eq!(Pre::Llama3.hf_reading(), Some(Pre::Llama3Hf));
    }

    /// Qwen3's regex split as Hugging Face's `tokenizers` runs it: every text of the pinned corpus
    /// (`hf_pieces_qwen2.py`: contractions in every case, letters and marks and numbers against
    /// each other, spaces, symbols, code, every Unicode version's codepoints) is cut into pieces of
    /// the byte lengths its pre-tokenizer gives.
    ///
    /// FAIL-first: with the reference's splitter in its place, the texts of codepoints Unicode 15.0
    /// added and the `ſ` contractions fail; with marks in the letter run, the mark texts.
    #[test]
    fn qwen2_hf_pieces_are_hugging_faces() {
        let path = format!(
            "{}/../hf/tests/fixtures/qwen2-hf-pieces.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let corpus: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
        ))
        .unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() >= 11_000, "{} cases in {path}", cases.len());
        let mut s = Scratch::default();
        let mut failures = Vec::new();
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u64> = serde_json::from_value(case["pieces"].clone()).unwrap();
            let cpts: Vec<u32> = text.chars().map(u32::from).collect();
            let mut at = 0;
            let ours: Vec<u64> = split(Pre::Qwen2Hf, &cpts, &mut s)
                .iter()
                .map(|&n| {
                    let bytes: usize = cpts[at..at + n]
                        .iter()
                        .map(|&c| char::from_u32(c).unwrap().len_utf8())
                        .sum();
                    at += n;
                    bytes as u64
                })
                .collect();
            if ours != want {
                failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} texts:\n{}",
            failures.len(),
            cases.len(),
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// MiniMax's regex split as Hugging Face's `tokenizers` runs it: every text of the pinned corpus
    /// (`hf_pieces_minimax.py`: contractions in every case, letters of every category against
    /// each other, marks, digits, spaces, symbols, every Unicode version's codepoints) is cut into
    /// pieces of the byte lengths its pre-tokenizer gives. Token ids show a split only where a merge
    /// crosses it; the pieces show it everywhere.
    ///
    /// FAIL-first: with the reference's `minimax-m2` regex, which has no `\p{M}` in its words
    /// and tells only ASCII letters' case apart, the Devanagari and case-change texts fail; with
    /// the letters not told by category, the case-change texts; with no `ſ` fold, the contraction
    /// texts; with the prefix never left out or the capitals never given back, the mark texts.
    #[test]
    fn minimax_m2_hf_pieces_are_hugging_faces() {
        let path = format!(
            "{}/../hf/tests/fixtures/minimax-hf-pieces.json.gz",
            env!("CARGO_MANIFEST_DIR")
        );
        let corpus: serde_json::Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}")),
        ))
        .unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() >= 11_000, "{} cases in {path}", cases.len());
        let mut s = Scratch::default();
        let mut failures = Vec::new();
        for case in cases {
            let text = case["text"].as_str().unwrap();
            let want: Vec<u64> = serde_json::from_value(case["pieces"].clone()).unwrap();
            let cpts: Vec<u32> = text.chars().map(u32::from).collect();
            let mut at = 0;
            let ours: Vec<u64> = split(Pre::MiniMaxM2Hf, &cpts, &mut s)
                .iter()
                .map(|&n| {
                    let bytes: usize = cpts[at..at + n]
                        .iter()
                        .map(|&c| char::from_u32(c).unwrap().len_utf8())
                        .sum();
                    at += n;
                    bytes as u64
                })
                .collect();
            if ours != want {
                failures.push(format!("{text:?}: ours {ours:?}, HF {want:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} texts:\n{}",
            failures.len(),
            cases.len(),
            failures
                .iter()
                .take(20)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
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
        for pre in [Pre::MiniMaxM2, Pre::Gpt4o] {
            for &(text, want) in cases {
                assert_eq!(words(pre, text), want, "{pre:?} {text:?}");
            }
        }
    }

    /// `tekken` and `pixtral` are one row, the reference's TEKKEN type (a tree that reads `pixtral`
    /// as LLAMA3 predates the mapping), which takes a word that is itself a token whole, and it has
    /// a Hugging Face twin that does too.
    ///
    /// A tree that maps `pixtral` to it has the strict UTF-8 reader, so the reference reading
    /// is strict; the Hugging Face twin reads valid text alone, as the other twins do.
    ///
    /// FAIL-first: `pixtral` on `Pre::Llama3`, a missing `tekken` row, `Pre::ignore_merges`
    /// without the two Tekken readings, `Pre::hf_reading` without `Pre::Tekken`, or
    /// `Pre::strict_utf8` without it fails it.
    #[test]
    fn tekken_names() {
        for name in ["tekken", "pixtral"] {
            assert_eq!(crate::named(&Pre::NAMES, name), Some(Pre::Tekken), "{name}");
            assert!(crate::runs_pre_tokenizer(name), "{name}");
        }
        assert_eq!(Pre::Tekken.hf_reading(), Some(Pre::TekkenHf));
        assert!(Pre::Tekken.ignore_merges());
        assert!(Pre::TekkenHf.ignore_merges());
        assert!(Pre::Tekken.strict_utf8());
        assert!(!Pre::TekkenHf.strict_utf8());
    }

    /// The `tekken` regex's alternatives, one case each, derived by hand from the expression run on
    /// the collapsed bytes (a non-ASCII letter is `U` and `L` at once), beside the llama3 split a
    /// tree that predates the `pixtral` mapping gives: one number a word, a word cut where a
    /// lowercase letter meets a capital, no contraction, a symbol run that takes the slashes after
    /// a newline.
    ///
    /// FAIL-first: with `CasedShape::TEKKEN.digits` 3 the digit rows fail; with its `contraction`
    /// true the `it'sa` row; with the symbol run's tail `[\r\n]*` for `[\r\n/]*` the `a/\n/b` row
    /// (and `x/\n/y` of `minimax_m2_cases`); with llama3's split in its place the `HelloWorld`,
    /// `iPhone` and mixed-script rows.
    #[test]
    fn tekken_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              tekken                    llama3
            // `\p{N}`: a number is a word; llama3's `\p{N}{1,3}` takes three from the start.
            ("2026", &["2", "0", "2", "6"], &["202", "6"]),
            // The space before a number opens no word: the words need a letter, ` ?[^\s\p{L}\p{N}]+`
            // a symbol, `\s+(?!\S)` fails at the number, so `\s+` takes the space alone.
            (" 12", &[" ", "1", "2"], &[" ", "12"]),
            // `HelloWorld`: at 0 `U*` takes `H`, `L+` takes `ello` and stops at the capital `W`;
            // the second word is `World`. llama3's `\p{L}+` takes all ten.
            ("HelloWorld", &["Hello", "World"], &["HelloWorld"]),
            // `iPhone`: `U*` is empty at `i`, `L+` takes `i` alone; `P` then `hone` is a word.
            ("iPhone", &["i", "Phone"], &["iPhone"]),
            // No contraction alternative: `'sa` is a prefix and a lowercase run, where llama3 cuts
            // `'s` first.
            ("it'sa", &["it", "'sa"], &["it", "'s", "a"]),
            // At the first `/` no word fits (the next codepoint is `\n`), so the symbol run takes
            // it and `[\r\n/]*` takes `\n/` after it; the last `/` would open `/b` for llama3,
            // whose tail is `[\r\n]*`.
            ("a/\n/b", &["a", "/\n/", "b"], &["a", "/\n", "/b"]),
            // The slash is `b`'s prefix, and `\n` is a run of its own: both agree.
            ("a/b\n", &["a", "/b", "\n"], &["a", "/b", "\n"]),
            // A non-ASCII letter is `U` and `L` at once: `U*` takes `\u{dc}`, `L+` the rest.
            ("Ünïcode", &["Ünïcode"], &["Ünïcode"]),
            // Han, digits, a space and Cyrillic: `H` then `L+` over `ello` and the two Han letters;
            // each digit a word; the space is the prefix of a word whose `U*` takes all three
            // Cyrillic letters and gives the last one back to `L+`.
            (
                "Hello世界2026 Мир",
                &["Hello世界", "2", "0", "2", "6", " Мир"],
                &["Hello世界", "202", "6", " Мир"],
            ),
        ];
        for &(text, tekken, llama3) in cases {
            assert_eq!(words(Pre::Tekken, text), tekken, "tekken {text:?}");
            assert_eq!(words(Pre::Llama3, text), llama3, "llama3 {text:?}");
        }
    }

    /// A mark is a letter of either case in `tekken`'s lookaheads (`[\p{L}\p{M}]`), where
    /// `minimax-m2`'s read `\p{L}` alone: it stays in the word beside it, and a lone one is a word
    /// of its own; both Tekken readings agree on it.
    ///
    /// FAIL-first: with `TekkenReference` as `M2Reference` (no mark in `U` and `L`) every row
    /// fails the `tekken` assertion; with `CasedHf` without the mark in `U` and `L`, the `tekken hf`
    /// one.
    #[test]
    fn tekken_marks_are_word_letters() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              tekken                    minimax-m2
            // `L+` takes `e`, the mark and `x`; minimax-m2's `L+` stops at the mark, a prefix there.
            ("e\u{301}x", &["e\u{301}x"], &["e", "\u{301}x"]),
            // The mark alone is `L+` (the prefix taken first, nothing after it); the dot a symbol run.
            ("\u{301}.", &["\u{301}", "."], &["\u{301}."]),
            // The dot is a prefix, the mark a capital that gives way to `a` as `L+`.
            (".\u{301}a", &[".\u{301}a"], &[".\u{301}", "a"]),
        ];
        for &(text, tekken, minimax) in cases {
            assert_eq!(words(Pre::Tekken, text), tekken, "tekken {text:?}");
            assert_eq!(words(Pre::TekkenHf, text), tekken, "tekken hf {text:?}");
            assert_eq!(words(Pre::MiniMaxM2, text), minimax, "minimax-m2 {text:?}");
        }
    }

    /// Hugging Face's `tekken` regex tells the letters by category and its classes are Unicode
    /// 14.0's: a capital outside ASCII is no lowercase letter there, where the reference's one
    /// byte for every such letter is both, and a codepoint Unicode 15.0 added is of no class (a
    /// Kaktovik numeral and a Kawi letter are neither number nor letter).
    ///
    /// FAIL-first: with `collapse` in place of `collapse_cased` for the Hugging Face reading every
    /// row fails; with `collapse_hf`, which tells no letter's case, the first.
    #[test]
    fn tekken_hf_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            //                              hf                        tekken
            // `\u{dc}` is `Lu`: `U*` takes it, `L+` takes `ber` and stops at the second capital,
            // which starts the next word.
            ("ÜberÜber", &["Über", "Über"], &["ÜberÜber"]),
            // The numeral is of no class: the space and it are one symbol run; as a number it is
            // a word of its own and the space is another.
            (" \u{1d2c0}", &[" \u{1d2c0}"], &[" ", "\u{1d2c0}"]),
            // The Kawi letter is of no class: `x` is the word and the letter a symbol run.
            ("x\u{11f04}", &["x", "\u{11f04}"], &["x\u{11f04}"]),
        ];
        for &(text, hf, reference) in cases {
            assert_eq!(words(Pre::TekkenHf, text), hf, "tekken hf {text:?}");
            assert_eq!(words(Pre::Tekken, text), reference, "tekken {text:?}");
        }
    }

    /// `tekken-num3` is a row of its own that runs the `tekken` splitter with the number branch the
    /// name spells, and takes a word that is itself a token whole with the strict UTF-8 reader of
    /// the tree that reads it (the one tree that does: it is younger than the check).
    ///
    /// FAIL-first: a missing row, `Pre::ignore_merges` without it, or `Pre::strict_utf8` without
    /// it fails it.
    #[test]
    fn tekken_num3_names() {
        assert_eq!(
            crate::named(&Pre::NAMES, "tekken-num3"),
            Some(Pre::TekkenNum3)
        );
        assert!(crate::runs_pre_tokenizer("tekken-num3"));
        assert!(Pre::TekkenNum3.ignore_merges());
        assert!(Pre::TekkenNum3.strict_utf8());
        assert_eq!(Pre::TekkenNum3.hf_reading(), None);
    }

    /// The `tekken-num3` regex is `tekken`'s with the number alternative
    /// `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}` (llama.cpp's `llama-vocab.cpp`, the
    /// `LLAMA_VOCAB_PRE_TYPE_TEKKEN_NUM3` case): a run of numbers is cut in words of three from
    /// its end, the first one the shorter — `1|234|567` where `tekken` cuts every number and
    /// glm4's `\p{N}{1,3}` cuts `123|456|7` from the start. Every other alternative splits as
    /// `tekken`'s does (the words, the symbol runs, the whitespace).
    ///
    /// FAIL-first: with `CasedShape::TEKKEN`'s digit rule (one number a word) every digit row
    /// fails; with glm4's (three from the start) the runs of 4, 5, 7 and 10 fail.
    #[test]
    fn tekken_num3_cases() {
        /// One case: the text, and its words under num3, tekken and glm4.
        type Case = (
            &'static str,
            &'static [&'static str],
            &'static [&'static str],
            &'static [&'static str],
        );
        let cases: &[Case] = &[
            //                              tekken-num3               tekken                    glm4
            ("7", &["7"], &["7"], &["7"]),
            ("12", &["12"], &["1", "2"], &["12"]),
            ("123", &["123"], &["1", "2", "3"], &["123"]),
            ("1234", &["1", "234"], &["1", "2", "3", "4"], &["123", "4"]),
            (
                "12345",
                &["12", "345"],
                &["1", "2", "3", "4", "5"],
                &["123", "45"],
            ),
            (
                "123456",
                &["123", "456"],
                &["1", "2", "3", "4", "5", "6"],
                &["123", "456"],
            ),
            (
                "1234567",
                &["1", "234", "567"],
                &["1", "2", "3", "4", "5", "6", "7"],
                &["123", "456", "7"],
            ),
            // The number words are the same as raon's rule: `1|234|567`.
            (
                "call 5551234567 now",
                &["call", " ", "5", "551", "234", "567", " now"],
                &[
                    "call", " ", "5", "5", "5", "1", "2", "3", "4", "5", "6", "7", " now",
                ],
                &["call", " ", "555", "123", "456", "7", " now"],
            ),
            // Mixed text: the words, the symbol runs and the whitespace cut as `tekken`'s do.
            (
                "abc1234def",
                &["abc", "1", "234", "def"],
                &["abc", "1", "2", "3", "4", "def"],
                &["abc", "123", "4", "def"],
            ),
            (
                " 12345",
                &[" ", "12", "345"],
                &[" ", "1", "2", "3", "4", "5"],
                &[" ", "123", "45"],
            ),
            (
                "Hello世界2026 Мир",
                &["Hello世界", "2", "026", " Мир"],
                &["Hello世界", "2", "0", "2", "6", " Мир"],
                &["Hello世界", "202", "6", " Мир"],
            ),
            // Unicode numbers: `²` and `³` are `No`, `١` to `٤` `Nd`, all one run of numbers; the
            // run of two is one word under num3 and glm4, two under tekken.
            (
                "x²³ ١٢٣٤",
                &["x", "²³", " ", "١", "٢٣٤"],
                &["x", "²", "³", " ", "١", "٢", "٣", "٤"],
                &["x", "²³", " ", "١٢٣", "٤"],
            ),
            ("it'sa", &["it", "'sa"], &["it", "'sa"], &["it", "'s", "a"]),
        ];
        for &(text, num3, tekken, glm4) in cases {
            assert_eq!(words(Pre::TekkenNum3, text), num3, "num3 {text:?}");
            assert_eq!(words(Pre::Tekken, text), tekken, "tekken {text:?}");
            assert_eq!(words(Pre::Glm4, text), glm4, "glm4 {text:?}");
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
