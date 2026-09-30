//! A Jinja subset, enough for the chat templates GGUF files carry.
//!
//! Environment semantics are Hugging Face's (`trim_blocks`, `lstrip_blocks`, the
//! `loopcontrols` extension, `tojson` as `json.dumps` with its `ensure_ascii`,
//! `indent`, `separators` and `sort_keys`), with the lexer's newline rules
//! (`\r\n` and `\r` read as `\n`, one trailing newline dropped), and `-`
//! whitespace control, `set` (plain and `ns.attr`), `namespace(...)` (a
//! mapping or pairs, then keywords), `for` with tuple unpacking, an `if`
//! filter and `loop.*`, `break`/`continue`, `if`/`elif`/`else`, `macro`
//! outside any `for` and macro (called by name with positional and keyword
//! arguments; an argument not given is its default, else undefined; the body
//! sees its arguments and the template's top-level names as they are at the
//! call, never the caller's loop names; the call's value is the body's output
//! as a string), the conditional expression, `and`/`or`/`not`/`in`,
//! comparisons (a tuple literal only as the right side of `in`/`not in`,
//! where it reads as a list), `+`/`-`/`*`/`/`/`//`/`%`/`~` (Python's rounding of `//` and
//! `%`), subscripts (a string's by character) and slices `[start:stop:step]`
//! (Python's rules), string literals with Python's escapes,
//! `.items()`/`.keys()`/`.values()`/`.get()`/`.strip()`/`.lstrip()`/
//! `.rstrip()`/`.split()`/`.startswith()`/`.endswith()` with Python's
//! arguments, filters `tojson`/`from_json`/`length`/`count`/`trim`/`string`/
//! `lower`/`upper`/`capitalize`/`default`/`items`/`safe`, tests `defined`/`undefined`/`none`/`true`/
//! `false`/`boolean`/`string`/`number`/`mapping`/`sequence`/`iterable`, and
//! `raise_exception`/`range`. Whitespace is Python's `str.isspace`, and
//! `{{ value }}` is Python's `str()`: a list or dict as its `repr`, a float in
//! `repr`'s shortest form. Anything else is a parse or render error, never
//! silent output: an argument the callee does not take, a macro defined
//! inside a `for` or a macro, a macro body that names `varargs`, `kwargs` or
//! `caller`, a macro used as a value, an undefined value, a namespace or a
//! range inside a list or dict literal, `capitalize` on a first character
//! whose title case is not its upper case, a `\N{...}` escape, a division by
//! zero, an arithmetic result that is not a finite number.
//!
//! Values are `serde_json` values plus `Undefined`, ranges and mutable
//! namespaces. Object keys keep the order they were parsed or built in, as a
//! Python dict's (serde_json's `preserve_order`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::Arc;

use serde_json::{Map, Value};

/// A template that failed to parse or to render.
#[derive(Debug, thiserror::Error)]
#[error("chat template: {0}")]
pub struct TemplateError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, TemplateError> {
    Err(TemplateError(msg.into()))
}

/// A parsed chat template.
pub struct ChatTemplate {
    source: String,
    body: Vec<Node>,
}

impl ChatTemplate {
    /// Parses `source`; every branch is parsed, executed or not.
    pub fn parse(source: &str) -> Result<Self, TemplateError> {
        // jinja2's lexer: every newline is `\n`, and one trailing newline is
        // not output (`keep_trailing_newline` off).
        let text = source.replace("\r\n", "\n").replace('\r', "\n");
        let segs = segment(text.strip_suffix('\n').unwrap_or(&text))?;
        let mut p = NodeParser {
            segs,
            at: 0,
            loops: 0,
            in_macro: false,
        };
        let (body, end) = p.block(&[])?;
        if let Some(tag) = end {
            return err(format!("unexpected {{% {tag} %}}"));
        }
        Ok(ChatTemplate {
            source: source.to_owned(),
            body,
        })
    }

    /// The template text as given.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Renders with `vars` as the global context.
    pub fn render(&self, vars: &Map<String, Value>) -> Result<String, TemplateError> {
        let globals: HashMap<String, V> = vars
            .iter()
            .map(|(k, v)| (k.clone(), V::J(v.clone())))
            .collect();
        let mut r = Renderer {
            frames: vec![globals],
            floor: 0,
            calls: 0,
            out: String::new(),
        };
        r.nodes(&self.body)?;
        Ok(r.out)
    }
}

// ---------------------------------------------------------------- segments

#[derive(Debug)]
enum Seg {
    Text(String),
    Out(String),
    Stmt(String),
}

struct Piece {
    seg: Seg,
    /// A tag: `{%-`/`{{-` trims all whitespace before it.
    trim_left: bool,
    /// A block tag under `lstrip_blocks`: spaces and tabs back to the line start go.
    lstrip: bool,
    /// `-%}`/`-}}` trims all whitespace after it.
    trim_right: bool,
    /// `{% %}` or `{# #}` (`trim_blocks` applies).
    block: bool,
    is_tag: bool,
}

/// Splits the source into text and tags and applies whitespace control.
fn segment(src: &str) -> Result<Vec<Seg>, TemplateError> {
    let mut raw: Vec<Piece> = Vec::new();
    let text = |t: &str| Piece {
        seg: Seg::Text(t.to_owned()),
        trim_left: false,
        lstrip: false,
        trim_right: false,
        block: false,
        is_tag: false,
    };
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let open = [("{{", "}}"), ("{%", "%}"), ("{#", "#}")]
            .iter()
            .filter_map(|&(o, c)| src[i..].find(o).map(|at| (i + at, o, c)))
            .min_by_key(|t| t.0);
        let Some((at, o, c)) = open else {
            raw.push(text(&src[i..]));
            break;
        };
        if at > i {
            raw.push(text(&src[i..at]));
        }
        let mut j = at + 2;
        let trim_left = b.get(j) == Some(&b'-');
        let keep_left = b.get(j) == Some(&b'+');
        if trim_left || keep_left {
            j += 1;
        }
        let close = find_close(src, j, c)
            .ok_or_else(|| TemplateError(format!("unclosed {o} at byte {at}")))?;
        let marked = close > j && matches!(b[close - 1], b'-' | b'+');
        let trim_right = marked && b[close - 1] == b'-';
        let inner = src[j..if marked { close - 1 } else { close }]
            .trim_matches(py_space)
            .to_owned();
        let block = o != "{{";
        let seg = match o {
            "{{" => Seg::Out(inner),
            "{%" => Seg::Stmt(inner),
            _ => Seg::Text(String::new()),
        };
        raw.push(Piece {
            seg,
            trim_left,
            lstrip: block && !keep_left,
            trim_right,
            block,
            is_tag: true,
        });
        i = close + 2;
    }
    for k in 0..raw.len() {
        if !raw[k].is_tag {
            continue;
        }
        let (tl, ls, tr, block) = (
            raw[k].trim_left,
            raw[k].lstrip,
            raw[k].trim_right,
            raw[k].block,
        );
        if k > 0
            && !raw[k - 1].is_tag
            && let Seg::Text(t) = &mut raw[k - 1].seg
        {
            if tl {
                t.truncate(t.trim_end_matches(py_space).len());
            } else if ls {
                let kept = t.trim_end_matches([' ', '\t']).len();
                if (kept == 0 && k == 1) || t[..kept].ends_with('\n') {
                    t.truncate(kept);
                }
            }
        }
        if k + 1 < raw.len()
            && !raw[k + 1].is_tag
            && let Seg::Text(t) = &mut raw[k + 1].seg
        {
            if tr {
                *t = t.trim_start_matches(py_space).to_owned();
            } else if block
                && let Some(rest) = t.strip_prefix("\r\n").or_else(|| t.strip_prefix('\n'))
            {
                *t = rest.to_owned();
            }
        }
    }
    Ok(raw.into_iter().map(|p| p.seg).collect())
}

/// Python's `str.isspace`: Rust's `White_Space` and the four separators
/// U+001C–U+001F, which Python counts and Rust does not.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn find_close(src: &str, from: usize, close: &str) -> Option<usize> {
    let b = src.as_bytes();
    let mut i = from;
    let mut quote: Option<u8> = None;
    while i + 1 < b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 1;
                } else if c == q {
                    quote = None;
                }
            }
            None => {
                if c == b'\'' || c == b'"' {
                    quote = Some(c);
                } else if src[i..].starts_with(close) {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------- AST

#[derive(Debug)]
enum Node {
    Text(String),
    Out(Expr),
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For {
        targets: Vec<String>,
        iter: Expr,
        /// `for … in … if cond`: the items the loop runs over.
        filter: Option<Expr>,
        body: Vec<Node>,
    },
    Set {
        name: String,
        attr: Option<String>,
        value: Expr,
    },
    Macro(Arc<Macro>),
    Break,
    Continue,
}

/// A `{% macro name(params) %}` definition.
#[derive(Debug)]
struct Macro {
    name: String,
    /// Each parameter and its default.
    params: Vec<(String, Option<Expr>)>,
    body: Vec<Node>,
}

#[derive(Debug, Clone)]
enum Expr {
    Lit(Value),
    Name(String),
    List(Vec<Expr>),
    Dict(Vec<(Expr, Expr)>),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    /// `obj[start:stop:step]`, each part optional.
    Slice(
        Box<Expr>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
        Option<Box<Expr>>,
    ),
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
    Filter(Box<Expr>, String, Vec<Expr>, Vec<(String, Expr)>),
    Test(Box<Expr>, String, bool),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Bin(Box<Expr>, BinOp, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cond(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

struct NodeParser {
    segs: Vec<Seg>,
    at: usize,
    /// `for` bodies around the tag being parsed, counted inside the
    /// innermost macro body (a `break` there cannot leave the macro).
    loops: usize,
    /// Inside a macro body.
    in_macro: bool,
}

impl NodeParser {
    /// Parses nodes until one of `ends` (a statement keyword) or the end.
    fn block(&mut self, ends: &[&str]) -> Result<(Vec<Node>, Option<String>), TemplateError> {
        let mut nodes = Vec::new();
        while self.at < self.segs.len() {
            let seg = std::mem::replace(&mut self.segs[self.at], Seg::Text(String::new()));
            self.at += 1;
            match seg {
                Seg::Text(t) => {
                    if !t.is_empty() {
                        nodes.push(Node::Text(t));
                    }
                }
                Seg::Out(src) => nodes.push(Node::Out(parse_expr_all(&src)?)),
                Seg::Stmt(src) => {
                    let kw = src.split(py_space).next().unwrap_or("").to_owned();
                    if ends.contains(&kw.as_str()) {
                        // The caller reads the rest of the tag (elif's condition).
                        self.segs[self.at - 1] = Seg::Stmt(src);
                        return Ok((nodes, Some(kw)));
                    }
                    nodes.push(self.statement(&kw, &src)?);
                }
            }
        }
        Ok((nodes, None))
    }

    fn statement(&mut self, kw: &str, src: &str) -> Result<Node, TemplateError> {
        let rest = src[kw.len()..].trim_matches(py_space);
        match kw {
            "if" => self.if_chain(rest),
            "for" => {
                let mut t = Lexer::new(rest)?;
                let mut targets = vec![t.name()?];
                while t.eat_op(",") {
                    targets.push(t.name()?);
                }
                if !t.eat_name("in") {
                    return err(format!("for without `in`: {src}"));
                }
                // jinja2 reads the iterable without a conditional expression:
                // an `if` after it filters the items.
                let iter = t.or()?;
                let filter = if t.eat_name("if") {
                    Some(t.expr()?)
                } else {
                    None
                };
                t.end(src)?;
                self.loops += 1;
                let (body, end) = self.block(&["endfor"])?;
                self.loops -= 1;
                self.expect_end(end, "endfor", src)?;
                Ok(Node::For {
                    targets,
                    iter,
                    filter,
                    body,
                })
            }
            "set" => {
                let mut t = Lexer::new(rest)?;
                let name = t.name()?;
                let attr = if t.eat_op(".") { Some(t.name()?) } else { None };
                if !t.eat_op("=") {
                    return err(format!("block set is not supported: {src}"));
                }
                let value = t.expr()?;
                t.end(src)?;
                Ok(Node::Set { name, attr, value })
            }
            "break" | "continue" => {
                if !rest.is_empty() {
                    return err(format!("{{% {kw} %}} takes nothing: {{% {src} %}}"));
                }
                if self.loops == 0 {
                    return err(format!("{{% {kw} %}} outside a for loop"));
                }
                Ok(if kw == "break" {
                    Node::Break
                } else {
                    Node::Continue
                })
            }
            "macro" => self.macro_def(rest, src),
            _ => err(format!("unsupported tag {{% {src} %}}")),
        }
    }

    /// `{% macro name(a, b=default) %}…{% endmacro %}`.
    fn macro_def(&mut self, rest: &str, src: &str) -> Result<Node, TemplateError> {
        if self.loops > 0 || self.in_macro {
            return err(format!(
                "{{% {src} %}}: a macro inside a for loop or a macro is not supported"
            ));
        }
        let mut t = Lexer::new(rest)?;
        let name = t.name()?;
        t.expect_op("(")?;
        let mut params: Vec<(String, Option<Expr>)> = Vec::new();
        while !t.eat_op(")") {
            let p = t.name()?;
            if params.iter().any(|(q, _)| *q == p) {
                return err(format!("{{% {src} %}}: parameter {p} given twice"));
            }
            let default = if t.eat_op("=") { Some(t.expr()?) } else { None };
            params.push((p, default));
            if !t.eat_op(",") {
                t.expect_op(")")?;
                break;
            }
        }
        t.end(src)?;
        self.in_macro = true;
        let (body, end) = self.block(&["endmacro"])?;
        self.in_macro = false;
        self.expect_end(end, "endmacro", src)?;
        Ok(Node::Macro(Arc::new(Macro { name, params, body })))
    }

    fn if_chain(&mut self, first_cond: &str) -> Result<Node, TemplateError> {
        let mut arms = Vec::new();
        let mut cond = parse_expr_all(first_cond)?;
        loop {
            let (body, end) = self.block(&["elif", "else", "endif"])?;
            arms.push((cond, body));
            let Some(end) = end else {
                return err("if without endif");
            };
            let Seg::Stmt(src) =
                std::mem::replace(&mut self.segs[self.at - 1], Seg::Text(String::new()))
            else {
                return err("internal: lost the closing tag");
            };
            match end.as_str() {
                "elif" => cond = parse_expr_all(src["elif".len()..].trim_matches(py_space))?,
                "else" => {
                    let (els, end) = self.block(&["endif"])?;
                    self.expect_end(end, "endif", &src)?;
                    return Ok(Node::If(arms, els));
                }
                _ => return Ok(Node::If(arms, Vec::new())),
            }
        }
    }

    fn expect_end(
        &mut self,
        end: Option<String>,
        want: &str,
        src: &str,
    ) -> Result<(), TemplateError> {
        if end.as_deref() != Some(want) {
            return err(format!("{src}: missing {{% {want} %}}"));
        }
        // Consume the end tag the block left in place.
        self.segs[self.at - 1] = Seg::Text(String::new());
        Ok(())
    }
}

fn parse_expr_all(src: &str) -> Result<Expr, TemplateError> {
    let mut t = Lexer::new(src)?;
    let e = t.expr()?;
    t.end(src)?;
    Ok(e)
}

// ---------------------------------------------------------------- expressions

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Str(String),
    Int(i64),
    Float(f64),
    Op(&'static str),
}

const OPS: [&str; 24] = [
    "//", "==", "!=", "<=", ">=", "<", ">", "+", "-", "*", "/", "%", "~", "|", ".", ",", ":", "(",
    ")", "[", "]", "{", "}", "=",
];

struct Lexer {
    toks: Vec<Tok>,
    at: usize,
    /// The next primary is the right side of `in`/`not in`, the one place a
    /// tuple literal is read (as a list).
    tuple_in: bool,
}

/// One escape of a string literal after its backslash (`cs[at]` onwards), as
/// jinja2 reads it: the literal's non-ASCII characters in Python's
/// `backslashreplace` form, then `unicode-escape`. Returns where the literal
/// goes on.
fn string_escape(cs: &[char], at: usize, s: &mut String) -> Result<usize, String> {
    let Some(&e) = cs.get(at) else {
        return Err("a dangling escape".into());
    };
    let hex = |n: usize, name: &str| -> Result<u32, String> {
        let digits: String = cs.get(at + 1..at + 1 + n).unwrap_or(&[]).iter().collect();
        if digits.len() != n || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("truncated \\{name} escape"));
        }
        u32::from_str_radix(&digits, 16).map_err(|e| e.to_string())
    };
    let code = |u: u32| {
        char::from_u32(u).ok_or_else(|| {
            if (0xD800..0xE000).contains(&u) {
                format!("a surrogate escape U+{u:04X} (not a character)")
            } else {
                format!("an escape past U+10FFFF: {u:X}")
            }
        })
    };
    let simple = match e {
        '\n' => Some(None),
        '\\' | '\'' | '"' => Some(Some(e)),
        'a' => Some(Some('\u{7}')),
        'b' => Some(Some('\u{8}')),
        'f' => Some(Some('\u{c}')),
        'n' => Some(Some('\n')),
        'r' => Some(Some('\r')),
        't' => Some(Some('\t')),
        'v' => Some(Some('\u{b}')),
        _ => None,
    };
    if let Some(c) = simple {
        s.extend(c);
        return Ok(at + 1);
    }
    match e {
        '0'..='7' => {
            let n = cs[at..]
                .iter()
                .take(3)
                .take_while(|c| ('0'..='7').contains(*c))
                .count();
            let digits: String = cs[at..at + n].iter().collect();
            let u = u32::from_str_radix(&digits, 8).map_err(|e| e.to_string())?;
            s.push(code(u)?);
            Ok(at + n)
        }
        'x' => {
            s.push(code(hex(2, "x")?)?);
            Ok(at + 3)
        }
        'u' => {
            s.push(code(hex(4, "u")?)?);
            Ok(at + 5)
        }
        'U' => {
            s.push(code(hex(8, "U")?)?);
            Ok(at + 9)
        }
        'N' => Err("\\N{...} escapes are not supported".into()),
        // `backslashreplace` turned the character into an escape whose own
        // backslash this one escapes: the escape's text is the output.
        c if !c.is_ascii() => {
            py_hex_escape(c, s);
            Ok(at + 1)
        }
        other => {
            s.push('\\');
            s.push(other);
            Ok(at + 1)
        }
    }
}

impl Lexer {
    fn new(src: &str) -> Result<Self, TemplateError> {
        let mut toks = Vec::new();
        let cs: Vec<char> = src.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if py_space(c) {
                i += 1;
            } else if c == '\'' || c == '"' {
                let mut s = String::new();
                i += 1;
                loop {
                    let Some(&ch) = cs.get(i) else {
                        return err(format!("unterminated string in `{src}`"));
                    };
                    i += 1;
                    if ch == c {
                        break;
                    }
                    if ch == '\\' {
                        i = string_escape(&cs, i, &mut s)
                            .map_err(|e| TemplateError(format!("{e} in `{src}`")))?;
                    } else {
                        s.push(ch);
                    }
                }
                toks.push(Tok::Str(s));
            } else if c.is_ascii_digit() {
                let start = i;
                while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '_') {
                    i += 1;
                }
                let float = i + 1 < cs.len() && cs[i] == '.' && cs[i + 1].is_ascii_digit();
                if float {
                    i += 1;
                    while i < cs.len() && cs[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let text: String = cs[start..i].iter().filter(|&&c| c != '_').collect();
                toks.push(if float {
                    Tok::Float(
                        text.parse()
                            .map_err(|_| TemplateError(format!("bad number {text}")))?,
                    )
                } else {
                    Tok::Int(
                        text.parse()
                            .map_err(|_| TemplateError(format!("bad number {text}")))?,
                    )
                });
            } else if c.is_alphabetic() || c == '_' {
                let start = i;
                while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                    i += 1;
                }
                toks.push(Tok::Name(cs[start..i].iter().collect()));
            } else {
                let rest: String = cs[i..cs.len().min(i + 2)].iter().collect();
                let Some(op) = OPS.iter().find(|op| rest.starts_with(**op)) else {
                    return err(format!("unexpected `{c}` in `{src}`"));
                };
                toks.push(Tok::Op(op));
                i += op.chars().count();
            }
        }
        Ok(Lexer {
            toks,
            at: 0,
            tuple_in: false,
        })
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.at)
    }

    fn peek_name(&self, n: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(x)) if x == n)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn eat_name(&mut self, n: &str) -> bool {
        if self.peek_name(n) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn name(&mut self) -> Result<String, TemplateError> {
        match self.toks.get(self.at).cloned() {
            Some(Tok::Name(n)) => {
                self.at += 1;
                Ok(n)
            }
            other => err(format!("expected a name, found {other:?}")),
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), TemplateError> {
        if self.eat_op(op) {
            Ok(())
        } else {
            err(format!("expected `{op}`, found {:?}", self.peek()))
        }
    }

    fn end(&self, src: &str) -> Result<(), TemplateError> {
        match self.peek() {
            None => Ok(()),
            Some(t) => err(format!("trailing {t:?} in `{src}`")),
        }
    }

    fn expr(&mut self) -> Result<Expr, TemplateError> {
        let e = self.or()?;
        if self.eat_name("if") {
            let cond = self.or()?;
            let els = if self.eat_name("else") {
                Some(Box::new(self.expr()?))
            } else {
                None
            };
            return Ok(Expr::Cond(Box::new(cond), Box::new(e), els));
        }
        Ok(e)
    }

    fn or(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.and()?;
        while self.eat_name("or") {
            e = Expr::Or(Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    fn and(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.not()?;
        while self.eat_name("and") {
            e = Expr::And(Box::new(e), Box::new(self.not()?));
        }
        Ok(e)
    }

    fn not(&mut self) -> Result<Expr, TemplateError> {
        if self.eat_name("not") {
            return Ok(Expr::Not(Box::new(self.not()?)));
        }
        self.compare()
    }

    fn compare(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.concat()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Op("==")) => BinOp::Eq,
                Some(Tok::Op("!=")) => BinOp::Ne,
                Some(Tok::Op("<")) => BinOp::Lt,
                Some(Tok::Op("<=")) => BinOp::Le,
                Some(Tok::Op(">")) => BinOp::Gt,
                Some(Tok::Op(">=")) => BinOp::Ge,
                Some(Tok::Name(n)) if n == "in" => BinOp::In,
                Some(Tok::Name(n))
                    if n == "not"
                        && matches!(self.toks.get(self.at + 1), Some(Tok::Name(m)) if m == "in") =>
                {
                    self.at += 1;
                    BinOp::NotIn
                }
                _ => return Ok(e),
            };
            self.at += 1;
            self.tuple_in = matches!(op, BinOp::In | BinOp::NotIn);
            let rhs = self.concat();
            self.tuple_in = false;
            e = Expr::Bin(Box::new(e), op, Box::new(rhs?));
        }
    }

    fn concat(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.add()?;
        while self.eat_op("~") {
            e = Expr::Bin(Box::new(e), BinOp::Concat, Box::new(self.add()?));
        }
        Ok(e)
    }

    fn add(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.mul()?;
        loop {
            let op = if self.eat_op("+") {
                BinOp::Add
            } else if self.eat_op("-") {
                BinOp::Sub
            } else {
                return Ok(e);
            };
            e = Expr::Bin(Box::new(e), op, Box::new(self.mul()?));
        }
    }

    fn mul(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.unary()?;
        loop {
            let op = if self.eat_op("*") {
                BinOp::Mul
            } else if self.eat_op("//") {
                BinOp::FloorDiv
            } else if self.eat_op("/") {
                BinOp::Div
            } else if self.eat_op("%") {
                BinOp::Mod
            } else {
                return Ok(e);
            };
            e = Expr::Bin(Box::new(e), op, Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr, TemplateError> {
        if self.eat_op("-") {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        let mut e = self.postfix()?;
        loop {
            if self.eat_op("|") {
                let name = self.name()?;
                let (args, kwargs) = if self.eat_op("(") {
                    self.args()?
                } else {
                    (Vec::new(), Vec::new())
                };
                e = Expr::Filter(Box::new(e), name, args, kwargs);
            } else if self.eat_name("is") {
                let negate = self.eat_name("not");
                let name = self.name()?;
                e = Expr::Test(Box::new(e), name, negate);
            } else {
                return Ok(e);
            }
        }
    }

    fn postfix(&mut self) -> Result<Expr, TemplateError> {
        let mut e = self.primary()?;
        loop {
            if self.eat_op(".") {
                e = Expr::Attr(Box::new(e), self.name()?);
            } else if self.eat_op("[") {
                let lo = self.slice_part()?;
                if self.eat_op(":") {
                    let hi = self.slice_part()?;
                    let step = if self.eat_op(":") {
                        self.slice_part()?
                    } else {
                        None
                    };
                    self.expect_op("]")?;
                    e = Expr::Slice(Box::new(e), lo, hi, step);
                } else {
                    self.expect_op("]")?;
                    let Some(lo) = lo else {
                        return err("empty subscript");
                    };
                    e = Expr::Index(Box::new(e), lo);
                }
            } else if self.eat_op("(") {
                let (args, kwargs) = self.args()?;
                e = Expr::Call(Box::new(e), args, kwargs);
            } else {
                return Ok(e);
            }
        }
    }

    /// One part of a subscript: `None` when it is omitted, the next token
    /// being the `:` or `]` that ends it.
    fn slice_part(&mut self) -> Result<Option<Box<Expr>>, TemplateError> {
        if matches!(self.peek(), Some(Tok::Op(":" | "]"))) {
            return Ok(None);
        }
        Ok(Some(Box::new(self.expr()?)))
    }

    /// Call arguments after `(`, through `)`.
    #[allow(
        clippy::type_complexity,
        reason = "positional and keyword lists, read once by the caller"
    )]
    fn args(&mut self) -> Result<(Vec<Expr>, Vec<(String, Expr)>), TemplateError> {
        let mut args = Vec::new();
        let mut kwargs = Vec::new();
        if self.eat_op(")") {
            return Ok((args, kwargs));
        }
        loop {
            let is_kw = matches!(self.peek(), Some(Tok::Name(_)))
                && matches!(self.toks.get(self.at + 1), Some(Tok::Op("=")));
            if is_kw {
                let k = self.name()?;
                self.expect_op("=")?;
                kwargs.push((k, self.expr()?));
            } else {
                args.push(self.expr()?);
            }
            if self.eat_op(")") {
                return Ok((args, kwargs));
            }
            self.expect_op(",")?;
        }
    }

    fn primary(&mut self) -> Result<Expr, TemplateError> {
        let tuple_ok = std::mem::take(&mut self.tuple_in);
        let Some(t) = self.toks.get(self.at).cloned() else {
            return err("expression ends early");
        };
        self.at += 1;
        Ok(match t {
            Tok::Str(s) => Expr::Lit(Value::String(s)),
            Tok::Int(i) => Expr::Lit(Value::from(i)),
            Tok::Float(f) => Expr::Lit(Value::from(f)),
            Tok::Name(n) => match n.as_str() {
                "true" | "True" => Expr::Lit(Value::Bool(true)),
                "false" | "False" => Expr::Lit(Value::Bool(false)),
                "none" | "None" => Expr::Lit(Value::Null),
                _ => Expr::Name(n),
            },
            Tok::Op("(") => {
                let e = self.expr()?;
                if self.peek() == Some(&Tok::Op(",")) {
                    if !tuple_ok {
                        return err(
                            "a tuple literal outside the right side of `in` is not supported",
                        );
                    }
                    let mut items = vec![e];
                    while self.eat_op(",") && !self.eat_op(")") {
                        items.push(self.expr()?);
                        if self.peek() != Some(&Tok::Op(",")) {
                            self.expect_op(")")?;
                            break;
                        }
                    }
                    return Ok(Expr::List(items));
                }
                self.expect_op(")")?;
                e
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.eat_op("]") {
                    items.push(self.expr()?);
                    if !self.eat_op(",") {
                        self.expect_op("]")?;
                        break;
                    }
                }
                Expr::List(items)
            }
            Tok::Op("{") => {
                let mut items = Vec::new();
                while !self.eat_op("}") {
                    let k = self.expr()?;
                    self.expect_op(":")?;
                    items.push((k, self.expr()?));
                    if !self.eat_op(",") {
                        self.expect_op("}")?;
                        break;
                    }
                }
                Expr::Dict(items)
            }
            other => return err(format!("unexpected {other:?}")),
        })
    }
}

// ---------------------------------------------------------------- values

/// A namespace's attributes, in the order they were first set.
type Ns = Rc<RefCell<Vec<(String, V)>>>;

/// Python's `range(start, stop, step)`; `step` is never 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PyRange {
    start: i64,
    stop: i64,
    step: i64,
}

impl PyRange {
    fn len(self) -> usize {
        let (a, b, s) = (
            i128::from(self.start),
            i128::from(self.stop),
            i128::from(self.step),
        );
        let n = if s > 0 {
            if b > a { (b - a + s - 1) / s } else { 0 }
        } else if a > b {
            (a - b - s - 1) / -s
        } else {
            0
        };
        usize::try_from(n).unwrap_or(usize::MAX)
    }

    /// Item `i`, which must be under [`PyRange::len`].
    fn item(self, i: usize) -> i64 {
        let i = i64::try_from(i).expect("a range item index fits i64");
        self.start + i * self.step
    }

    fn items(self) -> Vec<Value> {
        (0..self.len()).map(|i| Value::from(self.item(i))).collect()
    }

    /// Python's `repr`.
    fn repr(self) -> String {
        if self.step == 1 {
            format!("range({}, {})", self.start, self.stop)
        } else {
            format!("range({}, {}, {})", self.start, self.stop, self.step)
        }
    }
}

#[derive(Clone, Debug)]
enum V {
    Undef,
    J(Value),
    Ns(Ns),
    Range(PyRange),
    Macro(Arc<Macro>),
}

/// Where a value must be plain JSON: an element of a list or dict literal,
/// or `tojson`'s input.
#[derive(Clone, Copy)]
enum JsonUse {
    Element,
    ToJson,
}

impl V {
    fn truthy(&self) -> bool {
        match self {
            V::Undef => false,
            V::Ns(_) | V::Macro(_) => true,
            V::Range(r) => r.len() > 0,
            V::J(v) => match v {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            },
        }
    }

    /// The JSON value, where only a JSON value can go: jinja2 keeps an
    /// undefined value, a namespace or a range in a list or dict (this
    /// engine refuses them by name), and `tojson` refuses them as
    /// `json.dumps` does.
    fn into_json(self, at: JsonUse) -> Result<Value, TemplateError> {
        let kind = match self {
            V::J(v) => return Ok(v),
            V::Macro(m) => return err(format!("macro {} used as a value", m.name)),
            V::Undef => "an undefined value",
            V::Ns(_) => "a namespace",
            V::Range(_) => "a range",
        };
        err(match at {
            JsonUse::Element => format!("{kind} inside a list or dict is not supported"),
            JsonUse::ToJson => format!("tojson of {kind}: not JSON serializable"),
        })
    }

    /// Python's `str()`.
    fn render(&self, out: &mut String) -> Result<(), TemplateError> {
        match self {
            V::Undef => {}
            V::J(Value::String(s)) => out.push_str(s),
            other => other.repr(out)?,
        }
        Ok(())
    }

    /// Python's `repr()` (jinja2's `Undefined` for an undefined value).
    fn repr(&self, out: &mut String) -> Result<(), TemplateError> {
        self.repr_in(out, &mut Vec::new())
    }

    /// `repr()` inside the namespaces in `open`: a namespace that holds
    /// itself prints its attributes as `{...}` there, as Python's recursion
    /// guard on a dict's `repr` does.
    fn repr_in(
        &self,
        out: &mut String,
        open: &mut Vec<*const RefCell<Vec<(String, V)>>>,
    ) -> Result<(), TemplateError> {
        match self {
            V::Undef => out.push_str("Undefined"),
            V::J(v) => py_repr(v, out),
            V::Range(r) => out.push_str(&r.repr()),
            V::Ns(ns) if open.contains(&Rc::as_ptr(ns)) => out.push_str("<Namespace {...}>"),
            V::Ns(ns) => {
                open.push(Rc::as_ptr(ns));
                out.push_str("<Namespace {");
                for (i, (k, v)) in ns.borrow().iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    py_str_repr(k, out);
                    out.push_str(": ");
                    v.repr_in(out, open)?;
                }
                out.push_str("}>");
                open.pop();
            }
            V::Macro(m) => return err(format!("macro {} rendered as a value", m.name)),
        }
        Ok(())
    }
}

/// Python's `repr()` of a JSON value.
fn py_repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => py_number(n, out),
        Value::String(s) => py_str_repr(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_repr(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_str_repr(k, out);
                out.push_str(": ");
                py_repr(x, out);
            }
            out.push('}');
        }
    }
}

/// An integer as Python prints it, a float as `float.__repr__` (which
/// `json.dumps` uses too).
fn py_number(n: &serde_json::Number, out: &mut String) {
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
    } else if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
    } else if let Some(f) = n.as_f64() {
        py_float(f, out);
    }
}

/// `float.__repr__` of a finite float: the shortest digits that read back
/// to it, in positional notation when the decimal exponent is in -4..16,
/// else as `d.ddde±XX`.
fn py_float(f: f64, out: &mut String) {
    // `{:e}` prints the same shortest digits, as `d.ddde<exp>`.
    let sci = format!("{:e}", f.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} prints an exponent");
    let exp: i32 = exp.parse().expect("{:e} prints an integer exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if f.is_sign_negative() {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = usize::try_from(exp).expect("non-negative") + 1;
            let (int, frac) = if digits.len() > int_len {
                digits.split_at(int_len)
            } else {
                (digits.as_str(), "")
            };
            out.push_str(int);
            out.extend(std::iter::repeat_n('0', int_len.saturating_sub(int.len())));
            out.push('.');
            out.push_str(if frac.is_empty() { "0" } else { frac });
        } else {
            out.push_str("0.");
            let zeros = usize::try_from(-exp - 1).expect("positive");
            out.extend(std::iter::repeat_n('0', zeros));
            out.push_str(&digits);
        }
    } else {
        let (first, rest) = digits.split_at(1);
        out.push_str(first);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        let _ = write!(out, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
}

/// Python's `str.isprintable` past ASCII. The standard library's debug
/// escape leaves a character alone exactly when it is printable by that
/// definition (general category not Other or Separator), except these
/// ranges, which it escapes as default-ignorable while Python prints them.
fn py_printable(c: char) -> bool {
    const PRINTED: [(u32, u32); 9] = [
        (0x034F, 0x034F),
        (0x115F, 0x1160),
        (0x17B4, 0x17B5),
        (0x180B, 0x180D),
        (0x180F, 0x180F),
        (0x3164, 0x3164),
        (0xFE00, 0xFE0F),
        (0xFFA0, 0xFFA0),
        (0xE0100, 0xE01EF),
    ];
    if c.is_ascii() {
        return (' '..='~').contains(&c);
    }
    let u = u32::from(c);
    if PRINTED.iter().any(|&(lo, hi)| (lo..=hi).contains(&u)) {
        return true;
    }
    // Not the first character: the escape then looks at printability only.
    let s = format!("a{c}");
    s.escape_debug().eq(s.chars())
}

/// Python's `repr()` of a string: single quotes unless the string holds a
/// `'` and no `"`; `\\`, the quote, `\n`, `\r`, `\t` escaped, and every
/// other character that is not printable as `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh`.
fn py_str_repr(s: &str, out: &mut String) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if py_printable(c) => out.push(c),
            c => py_hex_escape(c, out),
        }
    }
    out.push(quote);
}

/// Python's escape of a character by its code point: `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh`, lower-case hex.
fn py_hex_escape(c: char, out: &mut String) {
    let u = u32::from(c);
    let _ = if u < 0x100 {
        write!(out, "\\x{u:02x}")
    } else if u < 0x1_0000 {
        write!(out, "\\u{u:04x}")
    } else {
        write!(out, "\\U{u:08x}")
    };
}

/// `json.dumps` options, as the HF environment's `tojson` passes them.
struct JsonStyle {
    ensure_ascii: bool,
    /// The text of one indent level; `None` is the one-line form.
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

impl JsonStyle {
    /// `json.dumps`'s defaults as `tojson` sets them (`ensure_ascii` off).
    fn plain() -> Self {
        JsonStyle {
            ensure_ascii: false,
            indent: None,
            item_sep: ", ".into(),
            key_sep: ": ".into(),
            sort_keys: false,
        }
    }

    fn dumps(&self, v: &Value) -> String {
        let mut s = String::new();
        self.value(v, 0, &mut s);
        s
    }

    fn newline(&self, level: usize, s: &mut String) {
        if let Some(ind) = &self.indent {
            s.push('\n');
            for _ in 0..level {
                s.push_str(ind);
            }
        }
    }

    fn value(&self, v: &Value, level: usize, s: &mut String) {
        match v {
            Value::Null => s.push_str("null"),
            Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            Value::Number(n) => py_number(n, s),
            Value::String(x) => self.string(x, s),
            Value::Array(a) if a.is_empty() => s.push_str("[]"),
            Value::Object(o) if o.is_empty() => s.push_str("{}"),
            Value::Array(a) => {
                s.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        s.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, s);
                    self.value(x, level + 1, s);
                }
                self.newline(level, s);
                s.push(']');
            }
            Value::Object(o) => {
                let mut items: Vec<(&String, &Value)> = o.iter().collect();
                if self.sort_keys {
                    items.sort_by(|a, b| a.0.cmp(b.0));
                }
                s.push('{');
                for (i, (k, x)) in items.into_iter().enumerate() {
                    if i > 0 {
                        s.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, s);
                    self.string(k, s);
                    s.push_str(&self.key_sep);
                    self.value(x, level + 1, s);
                }
                self.newline(level, s);
                s.push('}');
            }
        }
    }

    /// A JSON string as `json.dumps` writes it: `"`, `\\` and the control
    /// characters escaped (`\b`, `\f`, `\n`, `\r`, `\t` by name), and under
    /// `ensure_ascii` everything past `~` as `\uhhhh` (UTF-16 pairs).
    fn string(&self, x: &str, s: &mut String) {
        s.push('"');
        let mut units = [0u16; 2];
        for c in x.chars() {
            match c {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                '\u{8}' => s.push_str("\\b"),
                '\u{c}' => s.push_str("\\f"),
                '\n' => s.push_str("\\n"),
                '\r' => s.push_str("\\r"),
                '\t' => s.push_str("\\t"),
                c if c < ' ' || (self.ensure_ascii && c > '~') => {
                    for u in c.encode_utf16(&mut units) {
                        let _ = write!(s, "\\u{u:04x}");
                    }
                }
                c => s.push(c),
            }
        }
        s.push('"');
    }
}

/// `tojson`'s arguments: `ensure_ascii`, `indent`, `separators`, `sort_keys`,
/// in that order or by name.
fn tojson_style(args: &[V], kwargs: &[(&str, V)]) -> Result<JsonStyle, TemplateError> {
    const NAMES: [&str; 4] = ["ensure_ascii", "indent", "separators", "sort_keys"];
    if args.len() > NAMES.len() {
        return err("tojson takes at most 4 positional arguments");
    }
    let mut given: [Option<&V>; 4] = [None; 4];
    for (i, a) in args.iter().enumerate() {
        given[i] = Some(a);
    }
    for (k, a) in kwargs {
        let Some(i) = NAMES.iter().position(|n| n == k) else {
            return err(format!("tojson({k}=...) is not supported"));
        };
        if given[i].is_some() {
            return err(format!("tojson: {k} given twice"));
        }
        given[i] = Some(a);
    }
    let [ascii, indent, separators, sort_keys] = given;
    let mut style = JsonStyle::plain();
    style.ensure_ascii = ascii.is_some_and(V::truthy);
    style.sort_keys = sort_keys.is_some_and(V::truthy);
    style.indent = match indent {
        None | Some(V::J(Value::Null)) => None,
        Some(V::J(Value::String(s))) => Some(s.clone()),
        Some(V::J(Value::Bool(b))) => Some(" ".repeat(usize::from(*b))),
        Some(V::J(Value::Number(n))) if n.is_i64() => Some(
            " ".repeat(
                n.as_i64()
                    .and_then(|i| usize::try_from(i).ok())
                    .unwrap_or(0),
            ),
        ),
        Some(other) => {
            return err(format!(
                "tojson indent must be a count or a string, not {other:?}"
            ));
        }
    };
    if style.indent.is_some() {
        style.item_sep = ",".into();
    }
    match separators {
        None | Some(V::J(Value::Null)) => {}
        Some(V::J(Value::Array(p))) => match p.as_slice() {
            [Value::String(item), Value::String(key)] => {
                style.item_sep.clone_from(item);
                style.key_sep.clone_from(key);
            }
            _ => return err("tojson separators must be a pair of strings"),
        },
        Some(_) => return err("tojson separators must be a pair of strings"),
    }
    Ok(style)
}

// ---------------------------------------------------------------- render

/// What a node asks of the loop around it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Flow {
    Next,
    Break,
    Continue,
}

/// Macro calls nested deeper than this are an error, not a stack overflow.
const MAX_MACRO_DEPTH: usize = 64;

struct Renderer {
    frames: Vec<HashMap<String, V>>,
    /// The lowest frame a lookup reads besides the globals (frame 0): a macro
    /// body sees its own frames and the template's top level, never its
    /// caller's.
    floor: usize,
    /// Macro calls in progress.
    calls: usize,
    out: String,
}

impl Renderer {
    fn lookup(&self, name: &str) -> V {
        self.frames[self.floor..]
            .iter()
            .rev()
            .find_map(|f| f.get(name).cloned())
            .or_else(|| {
                (self.floor > 0)
                    .then(|| self.frames[0].get(name).cloned())
                    .flatten()
            })
            .unwrap_or(V::Undef)
    }

    fn assign(&mut self, name: String, v: V) {
        if let Some(f) = self.frames.last_mut() {
            f.insert(name, v);
        }
    }

    /// Renders `nodes` until one asks its loop to break or continue.
    fn nodes(&mut self, nodes: &[Node]) -> Result<Flow, TemplateError> {
        for n in nodes {
            let flow = self.node(n)?;
            if flow != Flow::Next {
                return Ok(flow);
            }
        }
        Ok(Flow::Next)
    }

    fn node(&mut self, n: &Node) -> Result<Flow, TemplateError> {
        match n {
            Node::Text(t) => self.out.push_str(t),
            Node::Out(e) => {
                let v = self.eval(e)?;
                v.render(&mut self.out)?;
            }
            Node::Macro(m) => self.assign(m.name.clone(), V::Macro(Arc::clone(m))),
            Node::Break => return Ok(Flow::Break),
            Node::Continue => return Ok(Flow::Continue),
            Node::If(arms, els) => {
                for (cond, body) in arms {
                    if self.eval(cond)?.truthy() {
                        return self.nodes(body);
                    }
                }
                return self.nodes(els);
            }
            Node::Set { name, attr, value } => {
                let v = self.eval(value)?;
                match attr {
                    None => self.assign(name.clone(), v),
                    Some(a) => match (self.lookup(name), v) {
                        (_, V::Macro(m)) => {
                            return err(format!("macro {} used as a value", m.name));
                        }
                        (V::Ns(ns), v) => ns_set(&ns, a, v),
                        _ => return err(format!("set {name}.{a}: {name} is not a namespace")),
                    },
                }
            }
            Node::For {
                targets,
                iter,
                filter,
                body,
            } => {
                let mut items: Vec<Value> = match self.eval(iter)? {
                    V::Undef => Vec::new(),
                    V::J(Value::Array(a)) => a,
                    V::J(Value::Object(o)) => o.keys().map(|k| Value::String(k.clone())).collect(),
                    V::J(Value::String(s)) => {
                        s.chars().map(|c| Value::String(c.to_string())).collect()
                    }
                    V::J(Value::Null) => Vec::new(),
                    V::Range(r) => r.items(),
                    other => return err(format!("cannot iterate {other:?}")),
                };
                if let Some(cond) = filter {
                    let mut kept = Vec::with_capacity(items.len());
                    for item in items {
                        self.frames.push(bind_targets(targets, item.clone())?);
                        let keep = self.eval(cond).map(|v| v.truthy());
                        self.frames.pop();
                        if keep? {
                            kept.push(item);
                        }
                    }
                    items = kept;
                }
                let len = items.len();
                for i in 0..len {
                    let mut lp = serde_json::json!({
                        "index0": i, "index": i + 1, "first": i == 0, "last": i + 1 == len,
                        "length": len, "revindex": len - i, "revindex0": len - i - 1,
                        "depth": 1, "depth0": 0,
                    });
                    if i > 0 {
                        lp["previtem"] = items[i - 1].clone();
                    }
                    if i + 1 < len {
                        lp["nextitem"] = items[i + 1].clone();
                    }
                    let mut frame = bind_targets(targets, items[i].clone())?;
                    frame.entry("loop".to_owned()).or_insert(V::J(lp));
                    self.frames.push(frame);
                    let r = self.nodes(body);
                    self.frames.pop();
                    if r? == Flow::Break {
                        break;
                    }
                }
            }
        }
        Ok(Flow::Next)
    }

    fn eval(&mut self, e: &Expr) -> Result<V, TemplateError> {
        Ok(match e {
            Expr::Lit(v) => V::J(v.clone()),
            Expr::Name(n) => {
                let v = self.lookup(n);
                if self.floor > 0
                    && matches!(v, V::Undef)
                    && matches!(n.as_str(), "varargs" | "kwargs" | "caller")
                {
                    return err(format!("a macro body naming {n} is not supported"));
                }
                v
            }
            Expr::List(items) => V::J(Value::Array(
                items
                    .iter()
                    .map(|x| self.eval(x)?.into_json(JsonUse::Element))
                    .collect::<Result<_, _>>()?,
            )),
            Expr::Dict(items) => {
                let mut m = Map::new();
                for (k, v) in items {
                    let key = match self.eval(k)? {
                        V::J(Value::String(s)) => s,
                        other => return err(format!("dict key must be a string, got {other:?}")),
                    };
                    m.insert(key, self.eval(v)?.into_json(JsonUse::Element)?);
                }
                V::J(Value::Object(m))
            }
            Expr::Attr(obj, name) => get_attr(&self.eval(obj)?, name),
            Expr::Index(obj, idx) => {
                let o = self.eval(obj)?;
                let i = self.eval(idx)?;
                index(&o, &i)
            }
            Expr::Slice(obj, lo, hi, step) => {
                let o = self.eval(obj)?;
                let mut part = |x: &Option<Box<Expr>>| match x {
                    Some(x) => slice_bound(&self.eval(x)?),
                    None => Ok(None),
                };
                let (lo, hi, step) = (part(lo)?, part(hi)?, part(step)?);
                slice(&o, lo, hi, step)?
            }
            Expr::Call(callee, args, kwargs) => self.call(callee, args, kwargs)?,
            Expr::Filter(x, name, args, kwargs) => {
                let v = self.eval(x)?;
                let args: Vec<V> = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<_, _>>()?;
                let kwargs = self.eval_kwargs(kwargs)?;
                filter(v, name, &args, &kwargs)?
            }
            Expr::Test(x, name, negate) => {
                let v = self.eval(x)?;
                V::J(Value::Bool(test(&v, name)? != *negate))
            }
            Expr::Not(x) => V::J(Value::Bool(!self.eval(x)?.truthy())),
            Expr::Neg(x) => match self.eval(x)? {
                V::J(Value::Number(n)) => match n.as_i64() {
                    Some(i) => V::J(Value::from(
                        i.checked_neg()
                            .ok_or_else(|| TemplateError("integer overflow".into()))?,
                    )),
                    None => V::J(Value::from(-n.as_f64().unwrap_or(0.0))),
                },
                other => return err(format!("cannot negate {other:?}")),
            },
            Expr::And(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { self.eval(b)? } else { l }
            }
            Expr::Or(a, b) => {
                let l = self.eval(a)?;
                if l.truthy() { l } else { self.eval(b)? }
            }
            Expr::Cond(cond, then, els) => {
                if self.eval(cond)?.truthy() {
                    self.eval(then)?
                } else {
                    match els {
                        Some(x) => self.eval(x)?,
                        None => V::Undef,
                    }
                }
            }
            Expr::Bin(a, op, b) => {
                let l = self.eval(a)?;
                let r = self.eval(b)?;
                binop(&l, *op, &r)?
            }
        })
    }

    fn eval_kwargs<'k>(
        &mut self,
        kwargs: &'k [(String, Expr)],
    ) -> Result<Vec<(&'k str, V)>, TemplateError> {
        kwargs
            .iter()
            .map(|(k, a)| Ok((k.as_str(), self.eval(a)?)))
            .collect()
    }

    fn call(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        kwargs: &[(String, Expr)],
    ) -> Result<V, TemplateError> {
        let args: Vec<V> = args
            .iter()
            .map(|a| self.eval(a))
            .collect::<Result<_, _>>()?;
        if let Expr::Name(n) = callee
            && let V::Macro(m) = self.lookup(n)
        {
            return self.call_macro(&m, args, kwargs);
        }
        let kwargs = self.eval_kwargs(kwargs)?;
        match callee {
            Expr::Name(n) if n == "namespace" => namespace(args, kwargs),
            Expr::Name(n) if n == "raise_exception" => {
                let msg = match (args.as_slice(), kwargs.as_slice()) {
                    ([m], []) | ([], [("message", m)]) => to_str(m)?,
                    _ => return err("raise_exception() takes 1 argument (message)"),
                };
                err(format!("raise_exception: {msg}"))
            }
            Expr::Name(n) if n == "range" => range(&args, &kwargs),
            Expr::Attr(obj, method) => {
                let o = self.eval(obj)?;
                call_method(&o, method, &args, &kwargs)
            }
            other => err(format!("unsupported call {other:?}")),
        }
    }

    /// Runs macro `m` on `args` and `kwargs` (evaluated here, in the caller's
    /// scope); its value is the body's output.
    fn call_macro(
        &mut self,
        m: &Arc<Macro>,
        args: Vec<V>,
        kwargs: &[(String, Expr)],
    ) -> Result<V, TemplateError> {
        let name = &m.name;
        if args.len() > m.params.len() {
            return err(format!(
                "macro {name} takes at most {} argument(s), {} given",
                m.params.len(),
                args.len()
            ));
        }
        let mut given: Vec<Option<V>> = args.into_iter().map(Some).collect();
        given.resize(m.params.len(), None);
        for (k, e) in kwargs {
            let Some(i) = m.params.iter().position(|(p, _)| p == k) else {
                return err(format!("macro {name} takes no argument {k}"));
            };
            if given[i].is_some() {
                return err(format!("macro {name}: argument {k} given twice"));
            }
            given[i] = Some(self.eval(e)?);
        }
        if self.calls >= MAX_MACRO_DEPTH {
            return err(format!(
                "macro {name}: calls nested deeper than {MAX_MACRO_DEPTH}"
            ));
        }
        let floor = self.floor;
        let out = std::mem::take(&mut self.out);
        self.frames.push(HashMap::new());
        self.floor = self.frames.len() - 1;
        self.calls += 1;
        let r = self.macro_body(m, given);
        self.calls -= 1;
        self.frames.pop();
        self.floor = floor;
        let body = std::mem::replace(&mut self.out, out);
        r.map(|_| V::J(Value::String(body)))
    }

    /// Binds the parameters in the frame [`Renderer::call_macro`] pushed (a
    /// default is evaluated there, after the parameters before it) and
    /// renders the body.
    fn macro_body(&mut self, m: &Macro, given: Vec<Option<V>>) -> Result<Flow, TemplateError> {
        for ((p, default), v) in m.params.iter().zip(given) {
            let v = match (v, default) {
                (Some(v), _) => v,
                (None, Some(d)) => self.eval(d)?,
                (None, None) => V::Undef,
            };
            self.assign(p.clone(), v);
        }
        self.nodes(&m.body)
    }
}

/// A loop item bound to the loop's target names (unpacked when there are
/// several).
fn bind_targets(targets: &[String], item: Value) -> Result<HashMap<String, V>, TemplateError> {
    let mut frame = HashMap::new();
    if targets.len() == 1 {
        frame.insert(targets[0].clone(), V::J(item));
        return Ok(frame);
    }
    let Value::Array(parts) = item else {
        return err(format!("cannot unpack a non-sequence into {targets:?}"));
    };
    if parts.len() != targets.len() {
        return err(format!(
            "cannot unpack {} values into {targets:?}",
            parts.len()
        ));
    }
    for (t, p) in targets.iter().zip(parts) {
        frame.insert(t.clone(), V::J(p));
    }
    Ok(frame)
}

/// Sets `ns.name`, in place when it is already set (a dict keeps a key's
/// position).
fn ns_set(ns: &Ns, name: &str, v: V) {
    let mut attrs = ns.borrow_mut();
    match attrs.iter_mut().find(|(k, _)| k == name) {
        Some(slot) => slot.1 = v,
        None => attrs.push((name.to_owned(), v)),
    }
}

/// `namespace(mapping_or_pairs, **kwargs)`: jinja2's `dict(*args, **kwargs)`.
fn namespace(args: Vec<V>, kwargs: Vec<(&str, V)>) -> Result<V, TemplateError> {
    let ns: Ns = Rc::new(RefCell::new(Vec::new()));
    match args.as_slice() {
        [] => {}
        [V::J(Value::Object(m))] => {
            for (k, v) in m {
                ns_set(&ns, k, V::J(v.clone()));
            }
        }
        [V::J(Value::Array(pairs))] => {
            for p in pairs {
                let Some([Value::String(k), v]) = p.as_array().map(Vec::as_slice) else {
                    return err(format!(
                        "namespace() takes a mapping or a list of (string, value) pairs, got {p}"
                    ));
                };
                ns_set(&ns, k, V::J(v.clone()));
            }
        }
        [other] => {
            return err(format!(
                "namespace() takes a mapping or a list of pairs, not {other:?}"
            ));
        }
        _ => return err("namespace() takes at most 1 positional argument"),
    }
    for (k, v) in kwargs {
        if let V::Macro(m) = v {
            return err(format!("macro {} used as a value", m.name));
        }
        ns_set(&ns, k, v);
    }
    Ok(V::Ns(ns))
}

/// `range(stop)`, `range(start, stop[, step])`: integers, step not 0.
fn range(args: &[V], kwargs: &[(&str, V)]) -> Result<V, TemplateError> {
    if !kwargs.is_empty() {
        return err("range() takes no keyword arguments");
    }
    let ints: Vec<i64> = args
        .iter()
        .map(|a| {
            as_i64(a).ok_or_else(|| TemplateError(format!("range() takes integers, not {a:?}")))
        })
        .collect::<Result<_, _>>()?;
    let (start, stop, step) = match ints.as_slice() {
        [stop] => (0, *stop, 1),
        [start, stop] => (*start, *stop, 1),
        [start, stop, step] => (*start, *stop, *step),
        _ => return err("range() takes 1 to 3 arguments"),
    };
    if step == 0 {
        return err("range() step must not be zero");
    }
    Ok(V::Range(PyRange { start, stop, step }))
}

fn as_i64(v: &V) -> Option<i64> {
    match v {
        V::J(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

fn get_attr(o: &V, name: &str) -> V {
    match o {
        V::Ns(ns) => ns
            .borrow()
            .iter()
            .find(|(k, _)| k == name)
            .map_or(V::Undef, |(_, v)| v.clone()),
        V::J(Value::Object(m)) => m.get(name).cloned().map_or(V::Undef, V::J),
        V::Range(r) => match name {
            "start" => V::J(Value::from(r.start)),
            "stop" => V::J(Value::from(r.stop)),
            "step" => V::J(Value::from(r.step)),
            _ => V::Undef,
        },
        _ => V::Undef,
    }
}

/// Position `i` (negative from the end) of `len` items, if there is one.
fn position(i: i64, len: usize) -> Option<usize> {
    let n = i64::try_from(len).unwrap_or(i64::MAX);
    let i = if i < 0 { i + n } else { i };
    usize::try_from(i).ok().filter(|&i| i < len)
}

fn index(o: &V, i: &V) -> V {
    match (o, i) {
        (V::J(Value::Object(m)), V::J(Value::String(k))) => {
            m.get(k).cloned().map_or(V::Undef, V::J)
        }
        (V::Ns(_), V::J(Value::String(k))) => get_attr(o, k),
        (V::J(Value::String(s)), V::J(Value::Number(n))) => {
            let Some(i) = n.as_i64() else { return V::Undef };
            position(i, s.chars().count())
                .and_then(|i| s.chars().nth(i))
                .map_or(V::Undef, |c| V::J(Value::String(c.to_string())))
        }
        (V::J(Value::Array(a)), V::J(Value::Number(n))) => {
            let Some(i) = n.as_i64() else { return V::Undef };
            position(i, a.len())
                .map(|i| a[i].clone())
                .map_or(V::Undef, V::J)
        }
        (V::Range(r), V::J(Value::Number(n))) => {
            let Some(i) = n.as_i64() else { return V::Undef };
            position(i, r.len()).map_or(V::Undef, |i| V::J(Value::from(r.item(i))))
        }
        _ => V::Undef,
    }
}

/// A slice part's value: an integer, or `None` for `none` (as omitted).
fn slice_bound(v: &V) -> Result<Option<i64>, TemplateError> {
    match v {
        V::J(Value::Null) => Ok(None),
        V::J(Value::Number(n)) if n.is_i64() => Ok(n.as_i64()),
        other => err(format!(
            "slice indices must be integers or none, not {other:?}"
        )),
    }
}

/// The positions `[start:stop:step]` visits in a sequence of `len` items, in
/// visiting order: Python's `range(*slice(start, stop, step).indices(len))`.
/// A negative bound counts from the end; a bound past either end clamps to
/// it; an omitted one is the end the step walks from, or towards.
fn slice_positions(len: usize, start: Option<i64>, stop: Option<i64>, step: i64) -> Vec<usize> {
    let n = i64::try_from(len).unwrap_or(i64::MAX);
    // What a clamped bound can name: a backward walk stops before position
    // 0, a forward one after position n - 1.
    let (lo, hi) = if step > 0 { (0, n) } else { (-1, n - 1) };
    let bound = |b: Option<i64>, omitted: i64| match b {
        None => omitted,
        Some(i) if i < 0 => i.saturating_add(n).max(lo),
        Some(i) => i.min(hi),
    };
    let (first, end) = if step > 0 {
        (bound(start, lo), bound(stop, hi))
    } else {
        (bound(start, hi), bound(stop, lo))
    };
    std::iter::successors(Some(first), |i| i.checked_add(step))
        .take_while(|&i| if step > 0 { i < end } else { i > end })
        .filter_map(|i| usize::try_from(i).ok())
        .collect()
}

/// `o[lo:hi:step]` of a list or a string (by character).
fn slice(o: &V, lo: Option<i64>, hi: Option<i64>, step: Option<i64>) -> Result<V, TemplateError> {
    let step = match step {
        None => 1,
        Some(0) => return err("slice step cannot be zero"),
        Some(s) => s,
    };
    match o {
        V::J(Value::Array(a)) => Ok(V::J(Value::Array(
            slice_positions(a.len(), lo, hi, step)
                .into_iter()
                .map(|i| a[i].clone())
                .collect(),
        ))),
        V::J(Value::String(s)) => {
            let cs: Vec<char> = s.chars().collect();
            Ok(V::J(Value::String(
                slice_positions(cs.len(), lo, hi, step)
                    .into_iter()
                    .map(|i| cs[i])
                    .collect(),
            )))
        }
        other => err(format!("cannot slice {other:?}")),
    }
}

/// Codepoints whose title case (Python's `str.capitalize` on a first
/// character) is not their upper case: Python 3.12's tables, `chr(c).title()
/// != chr(c).upper()`, 135 codepoints.
fn title_is_not_upper(c: char) -> bool {
    matches!(
        u32::from(c),
        0x00DF
            | 0x01C4..=0x01CC
            | 0x01F1..=0x01F3
            | 0x0587
            | 0x10D0..=0x10FA
            | 0x10FD..=0x10FF
            | 0x1F80..=0x1FAF
            | 0x1FB2..=0x1FB4
            | 0x1FB7
            | 0x1FBC
            | 0x1FC2..=0x1FC4
            | 0x1FC7
            | 0x1FCC
            | 0x1FF2..=0x1FF4
            | 0x1FF7
            | 0x1FFC
            | 0xFB00..=0xFB06
            | 0xFB13..=0xFB17
    )
}

/// Python's `str.capitalize`: the first character in title case, the rest
/// lowered in the context of the whole string (a final sigma after the first
/// character is still final).
fn capitalize(s: &str) -> Result<String, TemplateError> {
    let Some(first) = s.chars().next() else {
        return Ok(String::new());
    };
    if title_is_not_upper(first) {
        return err(format!(
            "capitalize: the title case of U+{:04X} is not its upper case",
            u32::from(first)
        ));
    }
    // A first character has nothing before it, so its lower case in context
    // is its plain lower case: drop that many bytes of the lowered string.
    let lower = s.to_lowercase();
    let skip: usize = first.to_lowercase().map(char::len_utf8).sum();
    Ok(first.to_uppercase().chain(lower[skip..].chars()).collect())
}

/// Checks a method's arguments: at most `max` positional ones, no keywords.
fn positional(
    method: &str,
    args: &[V],
    kwargs: &[(&str, V)],
    max: usize,
) -> Result<(), TemplateError> {
    if !kwargs.is_empty() {
        return err(format!("{method}() takes no keyword arguments"));
    }
    if args.len() > max {
        return err(match max {
            0 => format!("{method}() takes no arguments"),
            1 => format!("{method}() takes at most 1 argument"),
            2 if method == "get" => "get() takes 1 or 2 arguments".to_owned(),
            _ => format!("{method}() takes at most {max} arguments"),
        });
    }
    Ok(())
}

fn call_method(o: &V, method: &str, args: &[V], kwargs: &[(&str, V)]) -> Result<V, TemplateError> {
    Ok(match (o, method) {
        (V::J(Value::Object(m)), "items" | "keys" | "values") => {
            positional(method, args, kwargs, 0)?;
            V::J(Value::Array(match method {
                "items" => m
                    .iter()
                    .map(|(k, v)| Value::Array(vec![Value::String(k.clone()), v.clone()]))
                    .collect(),
                "keys" => m.keys().map(|k| Value::String(k.clone())).collect(),
                _ => m.values().cloned().collect(),
            }))
        }
        (V::J(Value::Object(m)), "get") => {
            positional(method, args, kwargs, 2)?;
            let k = match args {
                [V::J(Value::String(k))] | [V::J(Value::String(k)), _] => k,
                [other] | [other, _] => {
                    return err(format!("get() needs a string key, not {other:?}"));
                }
                _ => return err("get() takes 1 or 2 arguments"),
            };
            match m.get(k) {
                Some(v) => V::J(v.clone()),
                None => args.get(1).cloned().unwrap_or(V::J(Value::Null)),
            }
        }
        (V::J(Value::String(s)), "strip" | "lstrip" | "rstrip") => {
            positional(method, args, kwargs, 1)?;
            // Python's rule: no argument (or `none`) strips whitespace, a string
            // strips any of its characters.
            let set: Option<Vec<char>> = match args.first() {
                None | Some(V::J(Value::Null)) => None,
                Some(V::J(Value::String(c))) => Some(c.chars().collect()),
                Some(other) => return err(format!("{method}() takes a string, not {other:?}")),
            };
            let strips = |c: char| set.as_ref().map_or(py_space(c), |s| s.contains(&c));
            V::J(Value::String(
                match method {
                    "strip" => s.trim_matches(strips),
                    "lstrip" => s.trim_start_matches(strips),
                    _ => s.trim_end_matches(strips),
                }
                .to_owned(),
            ))
        }
        (V::J(Value::String(s)), "split") => {
            if args.len() > 2 {
                return err("split() takes at most 2 arguments");
            }
            let mut given = [args.first(), args.get(1)];
            for (k, a) in kwargs {
                let i = match *k {
                    "sep" => 0,
                    "maxsplit" => 1,
                    other => return err(format!("split() takes no argument {other}")),
                };
                if given[i].is_some() {
                    return err(format!("split(): {k} given twice"));
                }
                given[i] = Some(a);
            }
            V::J(Value::Array(
                split(s, given[0], given[1])?
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            ))
        }
        (V::J(Value::String(s)), "startswith" | "endswith") => {
            positional(method, args, kwargs, 3)?;
            let Some(V::J(Value::String(affix))) = args.first() else {
                return err(format!("{method}() takes a string, not {:?}", args.first()));
            };
            let bound = |i: usize| {
                args.get(i)
                    .map(slice_bound)
                    .transpose()
                    .map(Option::flatten)
            };
            V::J(Value::Bool(tail_match(
                s,
                affix,
                bound(1)?,
                bound(2)?,
                method == "endswith",
            )))
        }
        (other, m) => return err(format!("no method {m}() on {other:?}")),
    })
}

/// CPython's `tailmatch`: whether `s[start:end]` starts (or ends) with
/// `affix`, by character, where a start past the end matches nothing, not
/// even an empty affix.
fn tail_match(s: &str, affix: &str, start: Option<i64>, end: Option<i64>, at_end: bool) -> bool {
    let cs: Vec<char> = s.chars().collect();
    let a: Vec<char> = affix.chars().collect();
    let len = i64::try_from(cs.len()).unwrap_or(i64::MAX);
    let alen = i64::try_from(a.len()).unwrap_or(i64::MAX);
    let from_end = |i: i64| if i < 0 { (i + len).max(0) } else { i };
    let end = end.map_or(len, from_end).min(len);
    let start = start.map_or(0, from_end);
    if end - alen < start {
        return false;
    }
    let at = if at_end { end - alen } else { start };
    let at = usize::try_from(at).expect("a position inside the string");
    cs[at..at + a.len()] == a[..]
}

/// Python's `str.split(sep, maxsplit)`: on `sep` (a non-empty string), or on
/// runs of whitespace with the ends' whitespace dropped when `sep` is absent
/// or `none`; at most `maxsplit` splits when it is given and not negative.
fn split(s: &str, sep: Option<&V>, maxsplit: Option<&V>) -> Result<Vec<String>, TemplateError> {
    let max = match maxsplit {
        None => None,
        Some(V::J(Value::Number(n))) if n.is_i64() => {
            n.as_i64().and_then(|m| usize::try_from(m).ok())
        }
        Some(other) => {
            return err(format!(
                "split() maxsplit must be an integer, not {other:?}"
            ));
        }
    };
    match sep {
        None | Some(V::J(Value::Null)) => Ok(split_whitespace(s, max)),
        Some(V::J(Value::String(sep))) if sep.is_empty() => err("split(): empty separator"),
        Some(V::J(Value::String(sep))) => Ok(match max {
            Some(m) => s
                .splitn(m.saturating_add(1), sep.as_str())
                .map(str::to_owned)
                .collect(),
            None => s.split(sep.as_str()).map(str::to_owned).collect(),
        }),
        Some(other) => err(format!("split() takes a string separator, not {other:?}")),
    }
}

/// `str.split()` with no separator: the words between runs of whitespace; past
/// `max` splits, the rest of the string (its leading whitespace dropped) is
/// the last word.
fn split_whitespace(s: &str, max: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s.trim_start_matches(py_space);
    while !rest.is_empty() {
        if max.is_some_and(|m| out.len() == m) {
            out.push(rest.to_owned());
            break;
        }
        let end = rest.find(py_space).unwrap_or(rest.len());
        out.push(rest[..end].to_owned());
        rest = rest[end..].trim_start_matches(py_space);
    }
    out
}

fn filter(v: V, name: &str, args: &[V], kwargs: &[(&str, V)]) -> Result<V, TemplateError> {
    if name == "tojson" {
        let style = tojson_style(args, kwargs)?;
        return Ok(V::J(Value::String(
            style.dumps(&v.into_json(JsonUse::ToJson)?),
        )));
    }
    if name == "default" {
        return default_filter(v, args, kwargs);
    }
    if !args.is_empty() || !kwargs.is_empty() {
        return err(format!("|{name} takes no arguments"));
    }
    Ok(match name {
        "from_json" => match v {
            V::J(Value::String(s)) => V::J(
                serde_json::from_str(&s).map_err(|e| TemplateError(format!("from_json: {e}")))?,
            ),
            other => return err(format!("from_json on a non-string {other:?}")),
        },
        "length" | "count" => V::J(Value::from(match &v {
            V::J(Value::Array(a)) => a.len(),
            V::J(Value::Object(o)) => o.len(),
            V::J(Value::String(s)) => s.chars().count(),
            V::Range(r) => r.len(),
            // jinja2's Undefined has length 0.
            V::Undef => 0,
            other => {
                let kind = match other {
                    V::J(Value::Null) => "none",
                    V::J(Value::Bool(_)) => "bool",
                    V::J(Value::Number(n)) if n.is_f64() => "float",
                    V::J(_) => "int",
                    V::Ns(_) => "namespace",
                    _ => "macro",
                };
                return err(format!("|{name}: {kind} has no length"));
            }
        })),
        // jinja2's `do_items`: a mapping's pairs, nothing for undefined.
        "items" => V::J(Value::Array(match v {
            V::J(Value::Object(m)) => m
                .into_iter()
                .map(|(k, v)| Value::Array(vec![Value::String(k), v]))
                .collect(),
            V::Undef => Vec::new(),
            other => return err(format!("|items needs a mapping, not {other:?}")),
        })),
        "trim" => V::J(Value::String(to_str(&v)?.trim_matches(py_space).to_owned())),
        // No autoescape, so jinja2's `Markup(value)`: the value's `str()`.
        "string" | "safe" => V::J(Value::String(to_str(&v)?)),
        "lower" => V::J(Value::String(to_str(&v)?.to_lowercase())),
        "upper" => V::J(Value::String(to_str(&v)?.to_uppercase())),
        "capitalize" => V::J(Value::String(capitalize(&to_str(&v)?)?)),
        other => return err(format!("unsupported filter |{other}")),
    })
}

/// jinja2's `default(default_value='', boolean=False)`: the value, or the
/// default when the value is undefined (or, under `boolean`, false).
fn default_filter(v: V, args: &[V], kwargs: &[(&str, V)]) -> Result<V, TemplateError> {
    let mut slots: [Option<&V>; 2] = [args.first(), args.get(1)];
    if args.len() > 2 {
        return err("|default takes at most two arguments");
    }
    for (k, x) in kwargs {
        let at = match *k {
            "default_value" => 0,
            "boolean" => 1,
            other => return err(format!("|default takes no argument {other}")),
        };
        if slots[at].replace(x).is_some() {
            return err(format!("|default: {k} given twice"));
        }
    }
    let boolean = match slots[1] {
        None => false,
        Some(V::J(Value::Bool(b))) => *b,
        Some(other) => return err(format!("|default: boolean is {other:?}, not a bool")),
    };
    let missing = matches!(v, V::Undef) || (boolean && !v.truthy());
    Ok(match (missing, slots[0]) {
        (false, _) => v,
        (true, Some(d)) => d.clone(),
        (true, None) => V::J(Value::String(String::new())),
    })
}

fn to_str(v: &V) -> Result<String, TemplateError> {
    let mut s = String::new();
    v.render(&mut s)?;
    Ok(s)
}

fn test(v: &V, name: &str) -> Result<bool, TemplateError> {
    Ok(match name {
        "defined" => !matches!(v, V::Undef),
        "undefined" => matches!(v, V::Undef),
        "none" => matches!(v, V::J(Value::Null)),
        // Identity with the boolean, as jinja2's: `0 is false` is false.
        "true" => matches!(v, V::J(Value::Bool(true))),
        "false" => matches!(v, V::J(Value::Bool(false))),
        "string" => matches!(v, V::J(Value::String(_))),
        // Python's `bool` is a `Number`.
        "number" => matches!(v, V::J(Value::Number(_) | Value::Bool(_))),
        "boolean" => matches!(v, V::J(Value::Bool(_))),
        "mapping" => matches!(v, V::J(Value::Object(_))),
        // jinja2's Undefined has a length and items; a namespace has neither.
        "sequence" | "iterable" => matches!(
            v,
            V::J(Value::Array(_) | Value::String(_) | Value::Object(_)) | V::Range(_) | V::Undef
        ),
        other => return err(format!("unsupported test `is {other}`")),
    })
}

/// A number or a boolean as Python's `int`, when it is one.
fn py_int(v: &V) -> Option<i64> {
    match v {
        V::J(Value::Number(n)) => n.as_i64(),
        V::J(Value::Bool(b)) => Some(i64::from(*b)),
        _ => None,
    }
}

fn num(v: &V) -> Option<f64> {
    match v {
        V::J(Value::Number(n)) => n.as_f64(),
        V::J(Value::Bool(b)) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

/// Python's `==` on JSON values: booleans are the integers 0 and 1, an int
/// equals a float of the same value, and lists and dicts compare by item
/// (a dict regardless of key order).
fn json_eq(a: &Value, b: &Value) -> bool {
    let (x, y) = (V::J(a.clone()), V::J(b.clone()));
    match (a, b) {
        (Value::Number(_) | Value::Bool(_), Value::Number(_) | Value::Bool(_)) => {
            match (py_int(&x), py_int(&y)) {
                (Some(p), Some(q)) => p == q,
                _ => num(&x) == num(&y),
            }
        }
        (Value::Array(p), Value::Array(q)) => {
            p.len() == q.len() && p.iter().zip(q).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(p), Value::Object(q)) => {
            p.len() == q.len()
                && p.iter()
                    .all(|(k, v)| q.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

fn eq(a: &V, b: &V) -> bool {
    match (a, b) {
        (V::Undef, V::Undef) => true,
        (V::J(x), V::J(y)) => json_eq(x, y),
        // Two ranges are equal when they hold the same items.
        (V::Range(p), V::Range(q)) => {
            let n = p.len();
            n == q.len() && (n == 0 || p.start == q.start && (n == 1 || p.step == q.step))
        }
        (V::Ns(p), V::Ns(q)) => Rc::ptr_eq(p, q),
        _ => false,
    }
}

/// A float result, or the error Python raises where it has no float: a
/// result that is not finite is refused by name (JSON holds no infinity).
fn finite(f: f64) -> Result<Value, TemplateError> {
    if f.is_finite() {
        Ok(Value::from(f))
    } else {
        err(format!("arithmetic result {f} is not a finite number"))
    }
}

/// Python's float `divmod`: the floor quotient and the remainder with the
/// divisor's sign (CPython's `_float_div_mod`).
fn float_div_mod(a: f64, b: f64) -> (f64, f64) {
    let mut m = a % b;
    let mut d = (a - m) / b;
    if m == 0.0 {
        m = 0.0f64.copysign(b);
    } else if (b < 0.0) != (m < 0.0) {
        m += b;
        d -= 1.0;
    }
    let q = if d == 0.0 {
        0.0f64.copysign(a / b)
    } else {
        let f = d.floor();
        if d - f > 0.5 { f + 1.0 } else { f }
    };
    (q, m)
}

fn binop(l: &V, op: BinOp, r: &V) -> Result<V, TemplateError> {
    use BinOp::*;
    let int_pair = py_int(l).zip(py_int(r));
    let overflow = || TemplateError("integer overflow".into());
    Ok(V::J(match op {
        Eq => Value::Bool(eq(l, r)),
        Ne => Value::Bool(!eq(l, r)),
        Lt | Le | Gt | Ge => {
            let ord = match (l, r) {
                (V::J(Value::String(a)), V::J(Value::String(b))) => a.cmp(b),
                _ => match (num(l), num(r)) {
                    (Some(a), Some(b)) => a.total_cmp(&b),
                    _ => return err(format!("cannot compare {l:?} and {r:?}")),
                },
            };
            Value::Bool(match op {
                Lt => ord.is_lt(),
                Le => ord.is_le(),
                Gt => ord.is_gt(),
                _ => ord.is_ge(),
            })
        }
        In | NotIn => {
            let found = match r {
                V::J(Value::Array(a)) => a.iter().any(|x| eq(l, &V::J(x.clone()))),
                V::J(Value::Object(o)) => matches!(l, V::J(Value::String(k)) if o.contains_key(k)),
                V::J(Value::String(s)) => match l {
                    V::J(Value::String(k)) => s.contains(k.as_str()),
                    other => {
                        return err(format!("`in` a string needs a string, not {other:?}"));
                    }
                },
                V::Range(g) => match py_int(l) {
                    Some(x) => {
                        let off = i128::from(x) - i128::from(g.start);
                        let step = i128::from(g.step);
                        off % step == 0 && usize::try_from(off / step).is_ok_and(|i| i < g.len())
                    }
                    None => g.items().into_iter().any(|x| eq(l, &V::J(x))),
                },
                V::Undef => false,
                other => return err(format!("`in` on {other:?}")),
            };
            Value::Bool(found == (op == In))
        }
        Concat => Value::String(to_str(l)? + &to_str(r)?),
        Add => match (l, r) {
            (V::J(Value::String(a)), V::J(Value::String(b))) => Value::String(format!("{a}{b}")),
            (V::J(Value::Array(a)), V::J(Value::Array(b))) => {
                Value::Array(a.iter().chain(b).cloned().collect())
            }
            _ => match (int_pair, num(l), num(r)) {
                (Some((a, b)), _, _) => Value::from(a.checked_add(b).ok_or_else(overflow)?),
                (None, Some(a), Some(b)) => finite(a + b)?,
                _ => return err(format!("cannot add {l:?} and {r:?}")),
            },
        },
        Sub | Mul | Div | FloorDiv | Mod => {
            let zero = || TemplateError("division by zero".into());
            match (int_pair, op) {
                (Some((a, b)), Sub) => Value::from(a.checked_sub(b).ok_or_else(overflow)?),
                (Some((a, b)), Mul) => Value::from(a.checked_mul(b).ok_or_else(overflow)?),
                (Some((a, b)), FloorDiv | Mod) => {
                    if b == 0 {
                        return Err(zero());
                    }
                    // Truncating division, then Python's floor rounding: the
                    // remainder takes the divisor's sign.
                    let (q, m) = (
                        a.checked_div(b).ok_or_else(overflow)?,
                        a.checked_rem(b).ok_or_else(overflow)?,
                    );
                    let floor = m != 0 && (m < 0) != (b < 0);
                    Value::from(match (op, floor) {
                        (FloorDiv, true) => q - 1,
                        (FloorDiv, false) => q,
                        (_, true) => m + b,
                        _ => m,
                    })
                }
                _ => {
                    let (Some(a), Some(b)) = (num(l), num(r)) else {
                        return err(format!("arithmetic on {l:?} and {r:?}"));
                    };
                    if b == 0.0 && matches!(op, Div | FloorDiv | Mod) {
                        return Err(zero());
                    }
                    finite(match op {
                        Sub => a - b,
                        Mul => a * b,
                        Div => a / b,
                        FloorDiv => float_div_mod(a, b).0,
                        _ => float_div_mod(a, b).1,
                    })?
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn render(src: &str, vars: Value) -> String {
        let t = ChatTemplate::parse(src).expect("parses");
        let Value::Object(m) = vars else {
            panic!("vars")
        };
        t.render(&m).expect("renders")
    }

    #[test]
    fn whitespace_control_and_trim_blocks() {
        assert_eq!(
            render("a  {%- if true -%}  b  {%- endif -%}  c", json!({})),
            "abc"
        );
        assert_eq!(
            render("{% if true %}\nx\n{% endif %}\ny", json!({})),
            "x\ny"
        );
        assert_eq!(render("  {% if true %}x{% endif %}", json!({})), "x");
        assert_eq!(render("{{ 'a' }} {{ 'b' }}", json!({})), "a b");
    }

    #[test]
    fn scoping_namespace_loop_and_unpacking() {
        let src = "{%- set ns = namespace(n=0) -%}{%- set outer = 'o' -%}\
                   {%- for k, v in d.items() -%}{%- set outer = 'inner' -%}\
                   {%- set ns.n = ns.n + v -%}{{ loop.index0 }}{{ k }}{%- endfor -%}|{{ ns.n }}|{{ outer }}";
        assert_eq!(render(src, json!({"d": {"a": 1, "b": 2}})), "0a1b|3|o");
    }

    #[test]
    fn a_tuple_reads_as_a_list_on_the_right_of_in_only() {
        let src = "{{ 'a' in ('a', 'b') }}{{ 'c' not in ('a', 'b',) }}{{ 'a' in ('a') }}";
        assert_eq!(render(src, json!({})), "TrueTrueTrue");
        let Err(e) = ChatTemplate::parse("{{ ('a', 'b') }}") else {
            panic!("a tuple outside `in` parses")
        };
        assert!(e.0.contains("tuple literal"), "{e:?}");
    }

    #[test]
    fn items_and_safe_are_jinja2s() {
        let src = "{%- for k, v in d|items %}{{ k }}={{ v }};{% endfor %}|{{ u|items|length }}";
        assert_eq!(render(src, json!({"d": {"b": 1, "a": 2}})), "b=1;a=2;|0");
        let t = ChatTemplate::parse("{{ 's'|items }}").expect("parses");
        let e = t.render(&Map::new()).expect_err("items of a string");
        assert!(e.0.contains("|items needs a mapping"), "{e:?}");
        assert_eq!(
            render("{{ d|tojson|safe }}|{{ d|safe }}", json!({"d": {"a": 1}})),
            "{\"a\": 1}|{'a': 1}"
        );
    }

    #[test]
    fn default_is_jinja2s() {
        let src = "{{ u|default('d') }}|{{ x|default('d') }}|{{ e|default('d') }}|\
                   {{ e|default('d', true) }}|{{ u|default }}|{{ e|default(boolean=true, default_value='k') }}";
        assert_eq!(render(src, json!({"x": "v", "e": ""})), "d|v||d||k");
        for (src, want) in [
            ("{{ u|default(1, true, 2) }}", "at most two"),
            ("{{ u|default(other=1) }}", "no argument other"),
            (
                "{{ u|default(1, boolean=true, default_value=2) }}",
                "given twice",
            ),
            ("{{ u|default(1, 'yes') }}", "not a bool"),
        ] {
            let t = ChatTemplate::parse(src).expect("parses");
            let e = t.render(&Map::new()).expect_err(src);
            assert!(e.0.contains(want), "{src}: {e:?}");
        }
    }

    const V41: &str = include_str!("../tests/fixtures/v41-chat-template.jinja");
    const BOS: &str = "<｜begin▁of▁sentence｜>";

    fn v41(vars: Value) -> String {
        let mut vars = vars;
        vars["bos_token"] = json!(BOS);
        render(V41, vars)
    }

    /// Hand trace of the V4.1 template (thinking off, drop_thinking on): the system
    /// text follows the BOS bare, a user turn opens with `<｜User｜>`, an assistant
    /// turn is `<｜Assistant｜></think>` + content + `<｜end▁of▁sentence｜>`, and the
    /// generation prompt is `<｜Assistant｜></think>`.
    #[test]
    fn v41_three_messages_by_hand() {
        let msgs = json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello!"},
        ]);
        assert_eq!(
            v41(json!({"messages": msgs, "add_generation_prompt": false})),
            "<｜begin▁of▁sentence｜>You are terse.<｜User｜>Hi there<｜Assistant｜></think>Hello!<｜end▁of▁sentence｜>"
        );
        let msgs = json!([
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Hi there"},
        ]);
        assert_eq!(
            v41(json!({"messages": msgs, "add_generation_prompt": true})),
            "<｜begin▁of▁sentence｜>You are terse.<｜User｜>Hi there<｜Assistant｜></think>"
        );
    }

    /// Thinking on: an assistant turn before the last user turn keeps only `</think>`
    /// (drop_thinking), and the generation prompt opens `<think>`.
    #[test]
    fn v41_thinking_by_hand() {
        let msgs = json!([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "r"},
            {"role": "user", "content": "c"},
        ]);
        assert_eq!(
            v41(json!({"messages": msgs, "add_generation_prompt": true, "enable_thinking": true})),
            "<｜begin▁of▁sentence｜><｜User｜>a<｜Assistant｜></think>b<｜end▁of▁sentence｜><｜User｜>c<｜Assistant｜><think>"
        );
    }

    /// The tool branches: schemas through `tojson`, call arguments through
    /// `from_json` and `.items()` (keys in the arguments' order, as jinja2 with
    /// `from_json` as `json.loads`), results as `<tool_result>`.
    #[test]
    fn v41_tools_by_hand() {
        let tools = json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
        let msgs = json!([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": "{\"x\": 1, \"s\": \"v\"}"}}
            ]},
            {"role": "tool", "content": "42"},
        ]);
        let out = v41(json!({"messages": msgs, "tools": tools, "add_generation_prompt": true}));
        assert!(
            out.starts_with("<｜begin▁of▁sentence｜>## Tools\n\nYou have access to a set of tools"),
            "{out}"
        );
        assert!(out.contains("### Available Tool Schemas\n\n{\"name\": \"f\", \"parameters\": {\"type\": \"object\"}}\n\nYou MUST strictly follow"), "{out}");
        assert!(
            out.ends_with(
                "<｜User｜>q<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n\
                 <｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"s\" string=\"true\">v</｜DSML｜parameter>\n\
                 </｜DSML｜invoke>\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>\
                 <｜User｜><tool_result>42</tool_result><｜Assistant｜></think>"
            ),
            "{out}"
        );
    }

    #[test]
    fn tests_filters_and_precedence() {
        let src = "{{ not x is defined }}{{ (m['c'] or 'dflt') }}{{ [1,2] | tojson }}\
                   {{ ('{\"q\": 1}' | from_json)['q'] + 1 }}{{ 'yes' if s is string else 'no' }}";
        assert_eq!(
            render(src, json!({"m": {"c": null}, "s": "t"})),
            "Truedflt[1, 2]2yes"
        );
    }

    fn render_err(src: &str, vars: Value) -> String {
        let t = ChatTemplate::parse(src).expect("parses");
        let Value::Object(m) = vars else {
            panic!("vars")
        };
        t.render(&m).expect_err("a render error").to_string()
    }

    /// Python's slice rules, each pair as jinja2 renders it: omitted and
    /// negative bounds, a step either way, bounds clamped past either end,
    /// `none` as an omitted part, strings by character.
    #[test]
    fn slices_follow_python() {
        let vars = json!({"a": [1, 2, 3, 4, 5], "s": "héllo", "n": null});
        for (src, want) in [
            ("{{ a[::-1] | tojson }}", "[5, 4, 3, 2, 1]"),
            ("{{ a[1:-1] | tojson }}", "[2, 3, 4]"),
            ("{{ a[::2] | tojson }}", "[1, 3, 5]"),
            ("{{ a[-2:] | tojson }}", "[4, 5]"),
            ("{{ a[:-1] | tojson }}", "[1, 2, 3, 4]"),
            ("{{ a[4:1:-2] | tojson }}", "[5, 3]"),
            ("{{ a[10:] | tojson }}", "[]"),
            ("{{ a[-10:2] | tojson }}", "[1, 2]"),
            ("{{ a[:10:-1] | tojson }}", "[]"),
            ("{{ a[::-2] | tojson }}", "[5, 3, 1]"),
            ("{{ a[1:4:2] | tojson }}", "[2, 4]"),
            ("{{ a[none:2] | tojson }}", "[1, 2]"),
            ("{{ a[n:n:n] | tojson }}", "[1, 2, 3, 4, 5]"),
            ("{{ a[-1:-6:-1] | tojson }}", "[5, 4, 3, 2, 1]"),
            ("{{ a[-1::-3] | tojson }}", "[5, 2]"),
            ("{{ a[3:-1:-1] | tojson }}", "[]"),
            ("{{ a[-10::-1] | tojson }}", "[]"),
            ("{{ a[:-7:-1] | tojson }}", "[5, 4, 3, 2, 1]"),
            ("{{ a[2::-1] | tojson }}", "[3, 2, 1]"),
            ("{{ a[1 + 1:] | tojson }}", "[3, 4, 5]"),
            ("{{ s[::-1] }}", "olléh"),
            ("{{ s[1:3] }}", "él"),
            ("{{ s[-3:] }}", "llo"),
            ("{{ s[:] }}", "héllo"),
        ] {
            assert_eq!(render(src, vars.clone()), want, "{src}");
        }
        for src in ["{{ a[::0] }}", "{{ s[::0] }}"] {
            let e = render_err(src, vars.clone());
            assert!(e.contains("slice step cannot be zero"), "{src}: {e}");
        }
    }

    /// `split` and `strip` with a character set, as Python's `str` methods,
    /// and the `true`/`false` tests (identity with the boolean, not truth),
    /// each as jinja2 renders it.
    #[test]
    fn split_strip_and_boolean_tests_follow_python() {
        let vars = json!({"t": "\n\n  x \n", "f": false, "z": 0, "n": null, "u": true});
        for (src, want) in [
            (
                "{{ 'a,b,,c'.split(',') | tojson }}",
                r#"["a", "b", "", "c"]"#,
            ),
            ("{{ ' a  b '.split() | tojson }}", r#"["a", "b"]"#),
            ("{{ 'a,b,c'.split(',', 1) | tojson }}", r#"["a", "b,c"]"#),
            ("{{ '</think>x'.split('</think>')[-1] }}", "x"),
            ("{{ 'abc'.split('abc') | tojson }}", r#"["", ""]"#),
            ("{{ ''.split(',') | tojson }}", r#"[""]"#),
            ("{{ ''.split() | tojson }}", "[]"),
            ("[{{ t.lstrip('\\n') }}]", "[  x \n]"),
            ("[{{ t.rstrip('\\n') }}]", "[\n\n  x ]"),
            ("[{{ t.strip('\\n') }}]", "[  x ]"),
            ("[{{ 'xxaxx'.strip('x') }}]", "[a]"),
            ("[{{ '  a  '.strip() }}]", "[a]"),
            ("[{{ ' \\ta\\n'.lstrip() }}]", "[a\n]"),
            (
                "{{ f is false }}{{ z is false }}{{ n is false }}{{ u is true }}{{ 1 is true }}{{ f is not false }}",
                "TrueFalseFalseTrueFalseFalse",
            ),
        ] {
            assert_eq!(render(src, vars.clone()), want, "{src}");
        }
        let e = render_err("{{ 'a'.split('') }}", vars);
        assert!(e.contains("empty separator"), "{e}");
    }

    /// Macros as jinja2 runs them: the body sees its arguments and the top
    /// level as it is at the call (a later `set` included), never the
    /// caller's loop names; an argument not given is its default or
    /// undefined; the value is the body's output, usable with `+`, `==` and
    /// as a condition; `break` and `continue` leave only their own loop. Each
    /// expected string is jinja2 3.1.2's render in the HF environment.
    #[test]
    fn macros_and_loop_controls_follow_jinja2() {
        for (src, want) in [
            (
                "{% set y = 1 %}{% macro m() %}{{ y }}{% endmacro %}{% set y = 2 %}{{ m() }}",
                "2",
            ),
            (
                "{% macro m(a, b) %}[{{ a }}|{{ b }}|{{ b is defined }}]{% endmacro %}{{ m(1) }}",
                "[1||False]",
            ),
            (
                "{% macro m(a, b='d') %}{{ a }}{{ b }}{% endmacro %}{{ m(b=2, a=1) }}{{ m(1) }}",
                "121d",
            ),
            (
                "{% macro m() %}[{{ i }}]{% endmacro %}{% for i in [1] %}{{ m() }}{% endfor %}",
                "[]",
            ),
            (
                "{% macro m() %}x{% endmacro %}{{ (m() + '\\n') | tojson }}{{ m() == 'x' }}",
                "\"x\\n\"True",
            ),
            (
                "{% macro m() %}{% set q = 1 %}{% endmacro %}{{ m() }}[{{ q is defined }}]",
                "[False]",
            ),
            (
                "{% macro m() %}{{ m2() }}{% endmacro %}{% macro m2() %}two{% endmacro %}{{ m() }}",
                "two",
            ),
            (
                "{% if true %}{% set z = 3 %}{% endif %}{% macro m() %}{{ z }}{% endmacro %}{{ m() }}",
                "3",
            ),
            (
                "{% macro e(x) %}{% if x %}1{% endif %}{% endmacro %}\
                 {% if e(0) %}a{% endif %}{% if e(1) %}b{% endif %}",
                "b",
            ),
            (
                "{% for i in [1, 2, 3] %}{% if i == 2 %}{% break %}{% endif %}{{ i }}{% endfor %}",
                "1",
            ),
            (
                "{% for i in [1, 2, 3] %}{% if i == 2 %}{% continue %}{% endif %}{{ i }}{% endfor %}",
                "13",
            ),
            (
                "{% for i in [1, 2] %}{% for j in [1, 2] %}{% break %}{% endfor %}{{ i }}{% endfor %}",
                "12",
            ),
            (
                "{% macro m() %}{% for j in [1, 2] %}{% if j == 1 %}{% break %}{% endif %}\
                 {% endfor %}ok{% endmacro %}{{ m() }}",
                "ok",
            ),
            (
                "{% for m in [1, 2] %}{% if m == 1 %}{% set r = 'R' %}{% endif %}\
                 {{ r is defined }}{% endfor %}",
                "TrueFalse",
            ),
        ] {
            assert_eq!(render(src, json!({})), want, "{src}");
        }
    }

    /// What jinja2 runs and this engine cannot match is refused by name.
    #[test]
    fn unsupported_macro_uses_are_named_errors() {
        for (src, want) in [
            (
                "{% for i in [1] %}{% macro m() %}{% endmacro %}{% endfor %}",
                "a macro inside a for loop or a macro",
            ),
            ("{% break %}", "outside a for loop"),
            (
                "{% macro m() %}{% break %}{% endmacro %}",
                "outside a for loop",
            ),
        ] {
            let e = ChatTemplate::parse(src).err().map(|e| e.to_string());
            assert!(
                e.as_deref().is_some_and(|e| e.contains(want)),
                "{src}: {e:?}"
            );
        }
        for (src, want) in [
            (
                "{% macro m(a) %}{% endmacro %}{{ m(1, 2) }}",
                "at most 1 argument",
            ),
            (
                "{% macro m(a) %}{% endmacro %}{{ m(c=1) }}",
                "takes no argument c",
            ),
            (
                "{% macro m(a) %}{% endmacro %}{{ m(1, a=2) }}",
                "given twice",
            ),
            ("{% macro m() %}x{% endmacro %}{{ m }}", "macro m rendered"),
            (
                "{% macro m() %}x{% endmacro %}{{ [m] }}",
                "macro m used as a value",
            ),
            (
                "{% macro m() %}{{ varargs }}{% endmacro %}{{ m() }}",
                "naming varargs",
            ),
            (
                "{% macro m() %}{{ m() }}{% endmacro %}{{ m() }}",
                "nested deeper",
            ),
            (
                "{{ 'x' | capitalize(1) }}",
                "|capitalize takes no arguments",
            ),
            ("{{ 'x' | length(1) }}", "|length takes no arguments"),
            ("{{ 'x' | tojson(foo=2) }}", "tojson(foo=...)"),
            ("{{ 'ǆa' | capitalize }}", "U+01C6"),
        ] {
            let e = render_err(src, json!({}));
            assert!(e.contains(want), "{src}: {e}");
        }
    }

    /// `capitalize`, `tojson(ensure_ascii=...)` and string subscripts as
    /// jinja2 renders them.
    #[test]
    fn capitalize_ascii_json_and_string_subscripts_follow_jinja2() {
        for (src, want) in [
            ("{{ 'hELLO wORLD' | capitalize }}", "Hello world"),
            ("{{ 'max' | capitalize }}", "Max"),
            ("{{ 'ΑΣ' | capitalize }}", "Ας"),
            ("{{ '' | capitalize }}", ""),
            ("{{ 3 | capitalize }}", "3"),
            ("{{ 'é' | tojson }}", "\"é\""),
            (
                "{{ 'é😀' | tojson(ensure_ascii=True) }}",
                "\"\\u00e9\\ud83d\\ude00\"",
            ),
            ("{{ 'é' | tojson(ensure_ascii=False) }}", "\"é\""),
            ("{{ 'é' | tojson(1) }}", "\"\\u00e9\""),
            (
                "{{ 'abc'[0] }}|{{ 'abc'[5] is defined }}|{{ 'abc'[-1] }}",
                "a|False|c",
            ),
        ] {
            assert_eq!(render(src, json!({})), want, "{src}");
        }
    }

    fn v(json: &str) -> Value {
        serde_json::from_str(json).expect("vars")
    }

    /// Every case that does not render as jinja2 does: `Ok` is jinja2's output,
    /// `Err` a text this engine's error must contain (where jinja2 fails too,
    /// or renders what this engine refuses by name, marked at the case).
    fn mismatches(cases: &[(&str, Value, Result<&str, &str>)]) -> Vec<String> {
        let mut bad = Vec::new();
        for (src, vars, want) in cases {
            let Value::Object(m) = vars else {
                panic!("vars")
            };
            let got = ChatTemplate::parse(src).and_then(|t| t.render(m));
            match (want, &got) {
                (Ok(w), Ok(g)) if w == g => {}
                (Err(part), Err(e)) if e.to_string().contains(part) => {}
                _ => bad.push(format!("{src}: want {want:?}, got {got:?}")),
            }
        }
        bad
    }

    /// A call's positional and keyword arguments as Python takes them: `split`'s
    /// `sep` and `maxsplit` by name, `range`'s step, `namespace`'s mapping, the
    /// `start`/`end` of `startswith`/`endswith`; an argument Python refuses is
    /// refused by name, never dropped.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn call_arguments_follow_python() {
        let bad = mismatches(&[
            (
                "{{ 'a,b,c'.split(',', maxsplit=1) | tojson }}",
                v(r#"{}"#),
                Ok("[\"a\", \"b,c\"]"),
            ),
            (
                "{{ 'a b c'.split(maxsplit=1) | tojson }}",
                v(r#"{}"#),
                Ok("[\"a\", \"b c\"]"),
            ),
            (
                "{{ 'a,b'.split(sep=',') | tojson }}",
                v(r#"{}"#),
                Ok("[\"a\", \"b\"]"),
            ),
            (
                "{{ 'a,b'.split(',', sep=',') }}",
                v(r#"{}"#),
                Err("split(): sep given twice"),
            ),
            (
                "{{ 'a'.split(foo=1) }}",
                v(r#"{}"#),
                Err("split() takes no argument foo"),
            ),
            (
                "{% for i in range(0, 10, 3) %}{{ i }}{% endfor %}",
                v(r#"{}"#),
                Ok("0369"),
            ),
            (
                "{% for i in range(5, 0, -2) %}{{ i }}{% endfor %}",
                v(r#"{}"#),
                Ok("531"),
            ),
            (
                "{{ range(1, 2, 0) }}",
                v(r#"{}"#),
                Err("range() step must not be zero"),
            ),
            (
                "{{ range(1, 2, 3, 4) }}",
                v(r#"{}"#),
                Err("range() takes 1 to 3 arguments"),
            ),
            (
                "{{ range(stop=3) }}",
                v(r#"{}"#),
                Err("range() takes no keyword arguments"),
            ),
            (
                "{% set ns = namespace({'a': 1}, b=2) %}{{ ns.a }}{{ ns.b }}",
                v(r#"{}"#),
                Ok("12"),
            ),
            (
                "{{ namespace(1) }}",
                v(r#"{}"#),
                Err("namespace() takes a mapping"),
            ),
            (
                "{{ 'x'.startswith(['x']) }}",
                v(r#"{}"#),
                Err("startswith() takes a string"),
            ),
            (
                "{{ 'xy'.startswith('y', 1) }}{{ 'xy'.endswith('x', 0, 1) }}{{ 'xy'.startswith('x', -1) }}{{ 'xy'.endswith('y', none, -1) }}",
                v(r#"{}"#),
                Ok("TrueTrueFalseFalse"),
            ),
            (
                "{{ d.get('a', 1, 2) }}",
                v(r#"{"d": {}}"#),
                Err("get() takes 1 or 2 arguments"),
            ),
            (
                "{{ d.get('a', default=1) }}",
                v(r#"{"d": {}}"#),
                Err("get() takes no keyword arguments"),
            ),
            (
                "{{ d.items(1) }}",
                v(r#"{"d": {}}"#),
                Err("items() takes no arguments"),
            ),
            (
                "{{ 'a'.strip('a', 'b') }}",
                v(r#"{}"#),
                Err("strip() takes at most 1 argument"),
            ),
            (
                "{{ 'a'.strip(chars='a') }}",
                v(r#"{}"#),
                Err("strip() takes no keyword arguments"),
            ),
            (
                "{{ raise_exception('a', 'b') }}",
                v(r#"{}"#),
                Err("raise_exception() takes 1 argument"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// `{{ value }}` is Python's `str()`: lists and dicts as their `repr` (strings
    /// quoted and escaped as `repr` does, floats in `repr`'s shortest form), a
    /// namespace as `<Namespace {...}>`, a range as `range(start, stop)`, dict keys
    /// in insertion order; an undefined value a namespace keeps stays undefined.
    /// What `tojson` cannot serialize is refused as jinja2 refuses it, and an
    /// undefined value or a namespace inside a list or dict literal by name.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn values_render_as_python_str() {
        let bad = mismatches(&[
            (
                "{{ [1, 'a', \"b'c\", none, true, 1.5, {'k': [2]}] }}",
                v(r#"{}"#),
                Ok("[1, 'a', \"b'c\", None, True, 1.5, {'k': [2]}]"),
            ),
            (
                "{% set ns = namespace(a=1, b='x') %}{{ ns }}",
                v(r#"{}"#),
                Ok("<Namespace {'a': 1, 'b': 'x'}>"),
            ),
            (
                "{{ s }}",
                v(
                    r#"{"s": ["\u0301", "\u007f", "\u200b", "\u00a0", "\u00e9", "\ud55c", "\n\t\\", "'", "'\"", "\ufe0f", "\udb40\udc01", "\u0085", "\ud83d\ude00", "\u00ad"]}"#,
                ),
                Ok(
                    "['\u{301}', '\\x7f', '\\u200b', '\\xa0', 'é', '한', '\\n\\t\\\\', \"'\", '\\'\"', '\u{fe0f}', '\\U000e0001', '\\x85', '😀', '\\xad']",
                ),
            ),
            (
                "{{ f }}|{{ f[0] }}|{{ f[2] }}|{{ f[5] }}|{{ f | tojson }}",
                v(
                    r#"{"f": [1e+16, 0.1, 1e-05, 2.0, 123456789.123, -0.0, 1e+22, 5e-324, 1000000000000000.0, 0.0001]}"#,
                ),
                Ok(
                    "[1e+16, 0.1, 1e-05, 2.0, 123456789.123, -0.0, 1e+22, 5e-324, 1000000000000000.0, 0.0001]|1e+16|1e-05|-0.0|[1e+16, 0.1, 1e-05, 2.0, 123456789.123, -0.0, 1e+22, 5e-324, 1000000000000000.0, 0.0001]",
                ),
            ),
            (
                "{{ [x] }}",
                v(r#"{}"#),
                Err("an undefined value inside a list or dict"),
            ), // jinja2 renders '[Undefined]': refused by name
            (
                "{{ {'a': x} }}",
                v(r#"{}"#),
                Err("an undefined value inside a list or dict"),
            ), // jinja2 renders "{'a': Undefined}": refused by name
            (
                "{{ x | tojson }}",
                v(r#"{}"#),
                Err("tojson of an undefined value"),
            ),
            (
                "{% set ns = namespace(a=1) %}{{ ns | tojson }}",
                v(r#"{}"#),
                Err("tojson of a namespace"),
            ),
            (
                "{% set ns = namespace(a=1) %}{{ [ns] }}",
                v(r#"{}"#),
                Err("a namespace inside a list or dict"),
            ), // jinja2 renders "[<Namespace {'a': 1}>]": refused by name
            (
                "{% set ns = namespace(t=none) %}{% set ns.t = m.missing %}{{ ns.t is defined }}|{{ ns.t }}|",
                v(r#"{"m": {}}"#),
                Ok("False||"),
            ),
            (
                "{{ {'b': 1, 'a': 2} }}|{{ {'b': 1, 'a': 2} | tojson }}|{% for k, v in {'b': 1, 'a': 2}.items() %}{{ k }}{% endfor %}",
                v(r#"{}"#),
                Ok("{'b': 1, 'a': 2}|{\"b\": 1, \"a\": 2}|ba"),
            ),
            (
                "{{ range(3) }}|{{ range(1, 5, 2) }}|{{ range(3) | length }}|{{ range(0) }}|{{ range(-2) | length }}",
                v(r#"{}"#),
                Ok("range(0, 3)|range(1, 5, 2)|3|range(0, 0)|0"),
            ),
            (
                "{{ range(3) | tojson }}",
                v(r#"{}"#),
                Err("tojson of a range"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// `tojson` is the HF environment's `json.dumps` filter: `ensure_ascii`,
    /// `indent` (a count or a string), `separators` and `sort_keys`, by name or
    /// in that order.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn tojson_takes_hf_options() {
        let bad = mismatches(&[
            (
                "{{ d | tojson(indent=2) }}",
                v(r#"{"d": {"b": [1, {"c": []}], "a": {}}}"#),
                Ok("{\n  \"b\": [\n    1,\n    {\n      \"c\": []\n    }\n  ],\n  \"a\": {}\n}"),
            ),
            (
                "{{ d | tojson(indent='\\t', sort_keys=true) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Ok("{\n\t\"a\": 1,\n\t\"b\": [\n\t\t1,\n\t\t2\n\t]\n}"),
            ),
            (
                "{{ d | tojson(separators=[',', ':']) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Ok("{\"b\":[1,2],\"a\":1}"),
            ),
            (
                "{{ d | tojson(False, 0) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Ok("{\n\"b\": [\n1,\n2\n],\n\"a\": 1\n}"),
            ),
            (
                "{{ e | tojson(ensure_ascii=true, indent='é') }}",
                v(r#"{"e": {"\u00e9": ["\u00fc"]}}"#),
                Ok("{\né\"\\u00e9\": [\néé\"\\u00fc\"\né]\n}"),
            ),
            (
                "{{ d | tojson(indent=none) }}|{{ d | tojson(indent=-1) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Ok("{\"b\": [1, 2], \"a\": 1}|{\n\"b\": [\n1,\n2\n],\n\"a\": 1\n}"),
            ),
            (
                "{{ d | tojson(foo=1) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Err("tojson(foo=...) is not supported"),
            ),
            (
                "{{ d | tojson(1, 2, 3, 4, 5) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Err("tojson takes at most 4 positional arguments"),
            ),
            (
                "{{ d | tojson(separators=[',']) }}",
                v(r#"{"d": {"b": [1, 2], "a": 1}}"#),
                Err("tojson separators must be a pair of strings"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// Whitespace is Python's `str.isspace`, U+001C–U+001F included: for `strip`,
    /// `split`, `| trim` and `-` whitespace control.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn whitespace_is_python_isspace() {
        let bad = mismatches(&[
            (
                "[{{ w.strip() }}]{{ w2.split() | tojson }}[{{ w3 | trim }}]",
                v(r#"{"w": "\u001c a \u001f", "w2": "a\u001db", "w3": "\u001e x"}"#),
                Ok("[a][\"a\", \"b\"][x]"),
            ),
            (
                "a\u{1c}{{- 'b' }}|{{ 'c' -}}\u{1f}d",
                v(r#"{}"#),
                Ok("ab|cd"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// `| length` of a value with no length is an error (of an undefined one, 0),
    /// and a string literal's escapes are Python's `unicode-escape` (`\N{...}`
    /// refused by name).
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn length_and_string_escapes_follow_jinja2() {
        let bad = mismatches(&[
            ("{{ 3 | length }}", v(r#"{}"#), Err("int has no length")),
            ("{{ x | length }}", v(r#"{}"#), Ok("0")),
            ("{{ none | length }}", v(r#"{}"#), Err("none has no length")),
            ("{{ true | length }}", v(r#"{}"#), Err("bool has no length")),
            (
                "{% set ns = namespace() %}{{ ns | length }}",
                v(r#"{}"#),
                Err("namespace has no length"),
            ),
            (
                "{{ '\\u00e9\\x41\\q\\a\\101\\b\\f\\v\\0' | tojson }}",
                v(r#"{}"#),
                Ok("\"éA\\\\q\\u0007A\\b\\f\\u000b\\u0000\""),
            ),
            ("{{ '\\U0001F600' }}", v(r#"{}"#), Ok("😀")),
            (
                "{{ '\\N{BULLET}' }}",
                v(r#"{}"#),
                Err("\\N{...} escapes are not supported"),
            ), // jinja2 renders '•': refused by name
            ("{{ 'a\\\nb' }}", v(r#"{}"#), Ok("ab")),
            ("{{ '\\x4' }}", v(r#"{}"#), Err("truncated \\x escape")),
            ("{{ '\\ud800' | tojson }}", v(r#"{}"#), Err("surrogate")),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// `for … in … if cond` filters the items before the loop counts them;
    /// `loop.previtem`/`nextitem`/`depth`/`depth0`; `//` and `%` round toward
    /// negative infinity; division by zero and a non-finite result are errors;
    /// `==` and `in` compare booleans with numbers as Python does.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn loop_filters_and_arithmetic_follow_python() {
        let bad = mismatches(&[
            (
                "{% for x in [1, 0, 2] if x %}{{ x }}{{ loop.length }}{{ loop.index }}{% endfor %}",
                v(r#"{}"#),
                Ok("121222"),
            ),
            (
                "{% for k, v in d.items() if v %}{{ k }}{% endfor %}",
                v(r#"{"d": {"b": 1, "a": 0, "c": 2}}"#),
                Ok("bc"),
            ),
            (
                "{% for x in [1, 2, 3] %}{{ loop.previtem }}/{{ loop.nextitem }}/{{ loop.depth }}{{ loop.depth0 }};{% endfor %}",
                v(r#"{}"#),
                Ok("/2/10;1/3/10;2//10;"),
            ),
            (
                "{% for x in [1, 2] %}{{ loop.previtem is defined }}{{ loop.nextitem is defined }}{% endfor %}",
                v(r#"{}"#),
                Ok("FalseTrueTrueFalse"),
            ),
            (
                "{{ 7 // -3 }} {{ 7 % -3 }} {{ -7 // 2 }} {{ -1.5 % 1 }} {{ 1.5 % -1 }} {{ 7.5 // -2 }} {{ -7 % 3 }} {{ 7 % 3 }} {{ -7.5 // 2 }} {{ 2.5 % 1 }}",
                v(r#"{}"#),
                Ok("-3 -2 -4 0.5 -0.5 -4.0 2 1 -4.0 0.5"),
            ),
            ("{{ 1 / 0 }}", v(r#"{}"#), Err("division by zero")),
            ("{{ 1 // 0 }}", v(r#"{}"#), Err("division by zero")),
            ("{{ 1.5 % 0 }}", v(r#"{}"#), Err("division by zero")),
            (
                "{{ f * 10 }}",
                v(r#"{"f": 1e+308}"#),
                Err("not a finite number"),
            ), // jinja2 renders 'inf': refused by name
            (
                "{{ 1 == true }}{{ [1] == [true] }}{{ 1 in [true] }}{{ 0 == false }}{{ 1.0 == true }}{{ {'a': 1} == {'a': true} }}{{ 'a' == 'a' }}{{ none == false }}{{ 2 == true }}",
                v(r#"{}"#),
                Ok("TrueTrueTrueTrueTrueTrueTrueFalseFalse"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }

    /// jinja2's lexer newlines (`\r\n` and `\r` as `\n`, one trailing
    /// newline dropped) and its escape of a non-ASCII character after a
    /// backslash; the tests on a namespace, an undefined value and a range;
    /// a range's items, attributes, truth, membership and equality; booleans
    /// as integers in arithmetic; `in` a string with a non-string refused.
    /// Each `Ok` is jinja2 3.1.6's render (Python 3.14) in the HF environment.
    #[test]
    fn newlines_tests_ranges_and_bools_follow_jinja2() {
        let bad = mismatches(&[
            ("a\n", v(r#"{}"#), Ok("a")),
            ("a\u{d}\nb\u{d}c\n\n", v(r#"{}"#), Ok("a\nb\nc\n")),
            ("{{ 'x\u{d}\ny' | tojson }}", v(r#"{}"#), Ok("\"x\\ny\"")),
            (
                "{{ '\\é' }}|{{ '\\😀' }}",
                v(r#"{}"#),
                Ok("\\xe9|\\U0001f600"),
            ),
            (
                "{% set ns = namespace() %}{{ ns is mapping }}{{ ns is iterable }}{{ ns is sequence }}",
                v(r#"{}"#),
                Ok("FalseFalseFalse"),
            ),
            (
                "{{ x is iterable }}{{ x is sequence }}{{ x is mapping }}{{ x is string }}",
                v(r#"{}"#),
                Ok("TrueTrueFalseFalse"),
            ),
            (
                "{{ range(3) is iterable }}{{ range(3) is sequence }}{{ range(3) is mapping }}{{ true is number }}{{ 1.5 is number }}",
                v(r#"{}"#),
                Ok("TrueTrueFalseTrueTrue"),
            ),
            (
                "{{ range(5)[1] }}{{ range(5)[-1] }}{{ range(1, 9, 3)[2] }}|{{ range(5)[7] }}|{{ range(1, 9, 3).step }}",
                v(r#"{}"#),
                Ok("147||3"),
            ),
            (
                "{% if range(0) %}a{% else %}b{% endif %}{% if range(1) %}c{% endif %}",
                v(r#"{}"#),
                Ok("bc"),
            ),
            (
                "{{ 2 in range(3) }}{{ 5 in range(3) }}{{ 4 in range(0, 9, 2) }}{{ 3 in range(0, 9, 2) }}{{ 2.0 in range(3) }}{{ -1 in range(5, -3, -3) }}",
                v(r#"{}"#),
                Ok("TrueFalseTrueFalseTrueTrue"),
            ),
            (
                "{{ range(3) == range(3) }}{{ range(3) == [0, 1, 2] }}{{ range(0) == range(2, 1) }}",
                v(r#"{}"#),
                Ok("TrueFalseTrue"),
            ),
            ("{{ range(3)[0:2] }}", v(r#"{}"#), Err("cannot slice")), // jinja2 renders 'range(0, 2)': refused by name
            ("{{ 'a' ~ range(2) }}", v(r#"{}"#), Ok("arange(0, 2)")),
            (
                "{% set ns = namespace(a=1) %}{% set ns.b = ns %}{{ ns.b.a }}|{{ ns }}",
                v(r#"{}"#),
                Ok("1|<Namespace {'a': 1, 'b': <Namespace {...}>}>"),
            ),
            (
                "{% set ns = namespace(a=1) %}{% set m = namespace(n=ns) %}{% set ns.m = m %}{{ ns }}",
                v(r#"{}"#),
                Ok("<Namespace {'a': 1, 'm': <Namespace {'n': <Namespace {...}>}>}>"),
            ),
            (
                "{{ true + 1 }}|{{ true * 2.5 }}|{{ 7 / 2 }}|{{ -(3) }}",
                v(r#"{}"#),
                Ok("2|2.5|3.5|-3"),
            ),
            (
                "{{ 1 in 'abc' }}",
                v(r#"{}"#),
                Err("`in` a string needs a string"),
            ),
        ]);
        assert!(
            bad.is_empty(),
            "{} of the cases:\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}
