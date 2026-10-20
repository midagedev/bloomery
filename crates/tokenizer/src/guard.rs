//! Where a mark goes in a client's text so that no special token is made of it ([`Tokenizer::guard`]).
//!
//! A special token's spelling in the text is found whole, and a spelling the text begins or finishes
//! is found by its edge: a template that writes `>` after a name makes `<|im_end|>` of a name that
//! ends in `<|im_end|`. Every mark is strictly inside the text, so it is never where a token the
//! template wrote starts or ends beside the text ([`crate::spans_cut`]).

use memchr::memmem::Finder;

/// The spellings of the multi-character special tokens, indexed for the three searches.
pub(crate) struct Spellings {
    needles: Vec<Box<[u8]>>,
    finders: Vec<Finder<'static>>,
    /// The needles starting with a byte, and ending with one.
    by_first: Vec<Vec<usize>>,
    by_last: Vec<Vec<usize>>,
    /// The longest needle, in bytes.
    longest: usize,
}

impl Spellings {
    /// The index of `texts`; a text of one character (or none) is not a spelling a mark could be
    /// put in, and is left out.
    pub(crate) fn new<'t>(texts: impl Iterator<Item = &'t str>) -> Spellings {
        let mut s = Spellings {
            needles: Vec::new(),
            finders: Vec::new(),
            by_first: vec![Vec::new(); 256],
            by_last: vec![Vec::new(); 256],
            longest: 0,
        };
        for text in texts.filter(|t| t.chars().nth(1).is_some()) {
            let bytes = text.as_bytes();
            let at = s.needles.len();
            s.by_first[usize::from(bytes[0])].push(at);
            s.by_last[usize::from(bytes[bytes.len() - 1])].push(at);
            s.longest = s.longest.max(bytes.len());
            s.needles.push(bytes.into());
            s.finders.push(Finder::new(bytes).into_owned());
        }
        s
    }

    /// The ascending byte offsets in `text` where a mark goes, each in `1..text.len()`:
    ///
    /// - beside every spelling the text carries: before it, after it when it starts the text (a
    ///   template's `startswith` of the spelling still sees it), after its first character when
    ///   it is the whole text;
    /// - before a spelling the text ends in the beginning of, and after the end of one the text
    ///   begins with the end of; a text that is only such a piece gets a mark after its first
    ///   character, and one of a single character none.
    pub(crate) fn cuts(&self, text: &str) -> Vec<usize> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        let after_first = text.chars().next().map(char::len_utf8).filter(|&c| c < n);
        let mut at = Vec::new();
        // A spelling starts with a byte the text must hold: one pass over the text spares the
        // search for each spelling whose first byte it lacks.
        let mut held = [false; 256];
        for &b in bytes {
            held[usize::from(b)] = true;
        }
        for (needle, finder) in self.needles.iter().zip(&self.finders) {
            if !held[usize::from(needle[0])] {
                continue;
            }
            let mut from = 0;
            while let Some(m) = finder.find(&bytes[from..]) {
                let (start, end) = (from + m, from + m + needle.len());
                if start > 0 {
                    at.push(start);
                } else if end < n {
                    at.push(end);
                } else {
                    at.extend(after_first);
                }
                from += m + 1;
            }
        }
        // The text ends in the beginning of a spelling.
        for start in n.saturating_sub(self.longest.saturating_sub(1))..n {
            if !text.is_char_boundary(start) {
                continue;
            }
            let rest = &bytes[start..];
            if self.by_first[usize::from(rest[0])]
                .iter()
                .any(|&i| self.needles[i].len() > rest.len() && self.needles[i].starts_with(rest))
            {
                if start > 0 {
                    at.push(start);
                } else {
                    at.extend(after_first);
                }
            }
        }
        // The text begins with the end of a spelling.
        for len in 1..=n.min(self.longest.saturating_sub(1)) {
            if !text.is_char_boundary(len) {
                continue;
            }
            let head = &bytes[..len];
            if self.by_last[usize::from(head[len - 1])]
                .iter()
                .any(|&i| self.needles[i].len() > len && self.needles[i].ends_with(head))
            {
                if len < n {
                    at.push(len);
                } else {
                    at.extend(after_first);
                }
            }
        }
        at.sort_unstable();
        at.dedup();
        at
    }
}
