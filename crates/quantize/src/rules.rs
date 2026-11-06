//! llama-quantize's `--tensor-type-file`: whitespace-separated `regex=type` tokens, no comments; the first
//! rule whose regex `std::regex_search` finds in the lowercased tensor name decides the type.
//!
//! The regex is ECMAScript's, which llama-quantize hands to `std::regex`. This reader takes the subset the rule
//! files use — literals, `\.` and the other escaped punctuation, `.`, `[...]`, `\d \w \s`, `( | )`, `* + ?`,
//! `^ $` — and refuses any other syntax by name: a pattern it would read differently from `std::regex` must not
//! silently decide which tensors are quantized here.

use crate::{Error, Result};

/// One piece of a pattern: what it matches once.
#[derive(Debug, Clone)]
enum Node {
    Lit(char),
    Any,
    Class {
        negate: bool,
        ranges: Vec<(char, char)>,
    },
    Start,
    End,
    Group(Vec<Vec<Item>>),
}

/// A node with its repeat bounds.
#[derive(Debug, Clone)]
struct Item {
    node: Node,
    min: usize,
    max: usize,
}

/// A parsed pattern.
#[derive(Debug, Clone)]
pub struct Pattern {
    alts: Vec<Vec<Item>>,
}

struct Parser<'a> {
    src: &'a [char],
    at: usize,
    text: &'a str,
}

impl Parser<'_> {
    fn refuse<T>(&self, what: &str) -> Result<T> {
        Err(Error::refused(format!(
            "rule pattern {:?}: {what} at {}: the reader takes literals, escaped punctuation, . [..] \\d \\w \\s ( | ) * + ? ^ $",
            self.text, self.at
        )))
    }

    fn peek(&self) -> Option<char> {
        self.src.get(self.at).copied()
    }

    fn alts(&mut self, depth: usize) -> Result<Vec<Vec<Item>>> {
        let mut alts = vec![Vec::new()];
        while let Some(c) = self.peek() {
            match c {
                ')' if depth > 0 => break,
                ')' => return self.refuse("an unmatched )"),
                '|' => {
                    self.at += 1;
                    alts.push(Vec::new());
                }
                _ => {
                    let node = self.atom(depth)?;
                    let (min, max) = match self.peek() {
                        Some('*') => (0, usize::MAX),
                        Some('+') => (1, usize::MAX),
                        Some('?') => (0, 1),
                        Some('{') => return self.refuse("a counted repeat"),
                        _ => (1, 1),
                    };
                    if (min, max) != (1, 1) {
                        self.at += 1;
                        if matches!(self.peek(), Some('?' | '+' | '*')) {
                            return self.refuse("a lazy or stacked quantifier");
                        }
                    }
                    alts.last_mut()
                        .expect("alts starts with one branch")
                        .push(Item { node, min, max });
                }
            }
        }
        Ok(alts)
    }

    fn atom(&mut self, depth: usize) -> Result<Node> {
        let c = self.peek().expect("the caller peeked");
        self.at += 1;
        match c {
            '.' => Ok(Node::Any),
            '^' => Ok(Node::Start),
            '$' => Ok(Node::End),
            '*' | '+' | '?' => self.refuse("a quantifier with nothing to repeat"),
            '{' | '}' => self.refuse("a brace"),
            '(' => {
                if self.peek() == Some('?') {
                    return self.refuse("a (? group (lookahead or non-capturing)");
                }
                let alts = self.alts(depth + 1)?;
                if self.peek() != Some(')') {
                    return self.refuse("an unclosed (");
                }
                self.at += 1;
                Ok(Node::Group(alts))
            }
            '[' => self.class(),
            '\\' => {
                let Some(e) = self.peek() else {
                    return self.refuse("a trailing backslash");
                };
                self.at += 1;
                match e {
                    'd' => Ok(Node::Class {
                        negate: false,
                        ranges: vec![('0', '9')],
                    }),
                    'w' => Ok(Node::Class {
                        negate: false,
                        ranges: vec![('a', 'z'), ('A', 'Z'), ('0', '9'), ('_', '_')],
                    }),
                    's' => Ok(Node::Class {
                        negate: false,
                        ranges: vec![(' ', ' '), ('\t', '\t'), ('\n', '\n'), ('\r', '\r')],
                    }),
                    c if c.is_ascii_alphanumeric() => self.refuse(&format!("the escape \\{c}")),
                    c => Ok(Node::Lit(c)),
                }
            }
            c => Ok(Node::Lit(c)),
        }
    }

    fn class(&mut self) -> Result<Node> {
        let negate = self.peek() == Some('^');
        if negate {
            self.at += 1;
        }
        let mut ranges = Vec::new();
        loop {
            let Some(c) = self.peek() else {
                return self.refuse("an unclosed [");
            };
            self.at += 1;
            if c == ']' && !ranges.is_empty() {
                return Ok(Node::Class { negate, ranges });
            }
            let lo = match c {
                '[' => return self.refuse("a character-class name or nested ["),
                '\\' => {
                    let Some(e) = self.peek() else {
                        return self.refuse("a trailing backslash");
                    };
                    self.at += 1;
                    if e.is_ascii_alphanumeric() {
                        return self.refuse(&format!("the escape \\{e} inside []"));
                    }
                    e
                }
                c => c,
            };
            if self.peek() == Some('-') && self.src.get(self.at + 1).is_some_and(|&n| n != ']') {
                self.at += 1;
                let hi = self.peek().expect("checked above");
                self.at += 1;
                if hi < lo {
                    return self.refuse("a range out of order");
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
    }
}

impl Pattern {
    /// Parse `text`, or refuse the syntax it cannot read.
    pub fn parse(text: &str) -> Result<Pattern> {
        let chars: Vec<char> = text.chars().collect();
        let mut p = Parser {
            src: &chars,
            at: 0,
            text,
        };
        let alts = p.alts(0)?;
        Ok(Pattern { alts })
    }

    /// Whether the pattern matches anywhere in `name` (`std::regex_search`).
    #[must_use]
    pub fn search(&self, name: &str) -> bool {
        let text: Vec<char> = name.chars().collect();
        (0..=text.len()).any(|start| {
            self.alts
                .iter()
                .any(|alt| match_items(alt, 0, &text, start, &mut |_| true))
        })
    }
}

fn match_items(
    items: &[Item],
    i: usize,
    text: &[char],
    pos: usize,
    k: &mut dyn FnMut(usize) -> bool,
) -> bool {
    let Some(it) = items.get(i) else {
        return k(pos);
    };
    repeat(it, 0, text, pos, &mut |p| {
        match_items(items, i + 1, text, p, k)
    })
}

fn repeat(
    it: &Item,
    count: usize,
    text: &[char],
    pos: usize,
    k: &mut dyn FnMut(usize) -> bool,
) -> bool {
    if count < it.max
        && match_node(&it.node, text, pos, &mut |p| {
            // an empty repeat past the minimum adds nothing and would not end
            (p != pos || count < it.min) && repeat(it, count + 1, text, p, k)
        })
    {
        return true;
    }
    count >= it.min && k(pos)
}

fn match_node(n: &Node, text: &[char], pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
    match n {
        Node::Lit(c) => text.get(pos) == Some(c) && k(pos + 1),
        Node::Any => text.get(pos).is_some_and(|c| *c != '\n') && k(pos + 1),
        Node::Class { negate, ranges } => {
            text.get(pos)
                .is_some_and(|c| ranges.iter().any(|(lo, hi)| (lo..=hi).contains(&c)) != *negate)
                && k(pos + 1)
        }
        Node::Start => pos == 0 && k(pos),
        Node::End => pos == text.len() && k(pos),
        Node::Group(alts) => {
            for alt in alts {
                if match_items(alt, 0, text, pos, k) {
                    return true;
                }
            }
            false
        }
    }
}

/// The rules of one file, in order.
#[derive(Debug, Clone, Default)]
pub struct Rules {
    rules: Vec<(String, Pattern, String)>,
}

impl Rules {
    /// Read the file's text: tokens split on whitespace, each `regex=type` split at its first `=`, as
    /// llama-quantize reads it.
    pub fn parse(text: &str) -> Result<Rules> {
        let mut rules = Vec::new();
        for (i, token) in text.split_whitespace().enumerate() {
            let Some((re, ty)) = token.split_once('=') else {
                return Err(Error::refused(format!(
                    "rules token {}: {token:?} is not regex=type",
                    i + 1
                )));
            };
            if re.is_empty() || ty.is_empty() {
                return Err(Error::refused(format!(
                    "rules token {}: {token:?} has an empty regex or type",
                    i + 1
                )));
            }
            rules.push((re.to_string(), Pattern::parse(re)?, ty.to_ascii_uppercase()));
        }
        Ok(Rules { rules })
    }

    /// The type of the first rule that matches `tensor` (lowercased, as llama-quantize matches it).
    #[must_use]
    pub fn type_of(&self, tensor: &str) -> Option<&str> {
        let name = tensor.to_ascii_lowercase();
        self.rules
            .iter()
            .find(|(_, p, _)| p.search(&name))
            .map(|(_, _, t)| t.as_str())
    }

    /// How many rules the file holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the file holds no rule.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(re: &str, name: &str) -> bool {
        Pattern::parse(re).unwrap().search(name)
    }

    #[test]
    fn the_k31_rule_shapes_match_as_regex_search_does() {
        let bump = r"blk\.(0|1|12)\.ffn_(gate|up)_exps\.weight";
        assert!(hit(bump, "blk.12.ffn_up_exps.weight"));
        assert!(hit(bump, "blk.0.ffn_gate_exps.weight"));
        assert!(!hit(bump, "blk.2.ffn_gate_exps.weight"));
        assert!(!hit(bump, "blk.1.ffn_down_exps.weight"));
        // search, not match: a prefix does not hide the name
        assert!(hit(
            r"ffn_(down|gate|up)_exps\.weight",
            "mtp.blk.3.ffn_down_exps.weight"
        ));
        assert!(hit(r"\.attn_k_b\.weight$", "blk.1.attn_k_b.weight"));
        assert!(!hit(r"\.attn_k_b\.weight$", "blk.1.attn_k_b.weight.scale"));
        assert!(hit(r"^(token_embd|output)\.weight$", "output.weight"));
        assert!(!hit(
            r"^(token_embd|output)\.weight$",
            "blk.0.output.weight"
        ));
    }

    #[test]
    fn repeats_classes_and_backtracking() {
        assert!(hit(r"blk\.\d+\.attn", "blk.123.attn_q"));
        assert!(!hit(r"blk\.\d+\.attn", "blk..attn_q"));
        assert!(hit(r"blk\.[0-9]*\.x", "blk..x"));
        assert!(hit(r"a.*b", "xaZZb"));
        assert!(hit(r"colou?r", "color"));
        assert!(hit(r"^(a|ab)c$", "abc"));
        assert!(hit(r"[^0-9]x", "ax"));
        assert!(!hit(r"[^0-9]x", "1x"));
        assert!(hit(r"(a*)*b", "aab"));
    }

    #[test]
    fn syntax_it_would_read_differently_is_refused_by_name() {
        for re in [
            r"a{2}", r"(?:a)", r"\bx", r"a*?", r"(a", r"a)", r"*a", r"\1", r"[a", r"[z-a]",
        ] {
            let e = Pattern::parse(re).unwrap_err().to_string();
            assert!(e.contains("rule pattern"), "{re}: {e}");
        }
    }

    #[test]
    fn rules_are_first_match_on_the_lowercased_name() {
        let r = Rules::parse(
            "blk\\.(3)\\.ffn_down_exps\\.weight=iq2_xxs\nffn_(down|gate|up)_exps\\.weight=IQ1_S\n\\.attn_q\\.weight$=q6_k\n",
        )
        .unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r.type_of("blk.3.ffn_down_exps.weight"), Some("IQ2_XXS"));
        assert_eq!(r.type_of("blk.4.ffn_down_exps.weight"), Some("IQ1_S"));
        assert_eq!(r.type_of("BLK.4.FFN_UP_EXPS.WEIGHT"), Some("IQ1_S"));
        assert_eq!(r.type_of("blk.0.attn_q.weight"), Some("Q6_K"));
        assert_eq!(r.type_of("token_embd.weight"), None);
    }

    #[test]
    fn a_token_without_a_type_is_refused() {
        for t in ["ffn_exps", "=iq1_s", "x="] {
            assert!(Rules::parse(t).is_err(), "{t}");
        }
    }
}
