//! A Jinja subset, enough for the chat templates GGUF files carry.
//!
//! Environment semantics are Hugging Face's (`trim_blocks`, `lstrip_blocks`), and
//! `-` whitespace control, `set` (plain and `ns.attr`), `namespace(...)`, `for`
//! with tuple unpacking and `loop.*`, `if`/`elif`/`else`, the conditional
//! expression, `and`/`or`/`not`/`in`, comparisons, `+`/`-`/`~`, subscripts and
//! slices `[start:stop:step]` (Python's rules: any part omitted, negative
//! bounds from the end), `.items()`/`.keys()`/`.values()`/`.get()`/
//! `.strip()`/`.lstrip()`/`.rstrip()` (whitespace, or the characters given)/
//! `.split()`/`.startswith()`/`.endswith()`, filters `tojson`/`from_json`/
//! `length`/`trim`/`string`/`lower`/`upper`, tests `defined`/`undefined`/`none`/
//! `true`/`false`/`boolean`/`string`/`number`/`mapping`/`sequence`/`iterable`,
//! and `raise_exception`/`range`. Anything else is a parse or render error,
//! never silent output.
//!
//! Values are `serde_json` values plus `Undefined` and mutable namespaces. Object
//! keys iterate in sorted order (serde_json's map), not insertion order: a
//! template that `tojson`s a tool schema prints its keys sorted.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;

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
        let segs = segment(source)?;
        let mut p = NodeParser { segs, at: 0 };
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
            .trim()
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
                t.truncate(t.trim_end().len());
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
                *t = t.trim_start().to_owned();
            } else if block
                && let Some(rest) = t.strip_prefix("\r\n").or_else(|| t.strip_prefix('\n'))
            {
                *t = rest.to_owned();
            }
        }
    }
    Ok(raw.into_iter().map(|p| p.seg).collect())
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
        body: Vec<Node>,
    },
    Set {
        name: String,
        attr: Option<String>,
        value: Expr,
    },
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
    Filter(Box<Expr>, String, Vec<Expr>),
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
                    let kw = src.split_whitespace().next().unwrap_or("").to_owned();
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
        let rest = src[kw.len()..].trim();
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
                let iter = t.expr()?;
                t.end(src)?;
                let (body, end) = self.block(&["endfor"])?;
                self.expect_end(end, "endfor", src)?;
                Ok(Node::For {
                    targets,
                    iter,
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
            _ => err(format!("unsupported tag {{% {src} %}}")),
        }
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
                "elif" => cond = parse_expr_all(src["elif".len()..].trim())?,
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
}

impl Lexer {
    fn new(src: &str) -> Result<Self, TemplateError> {
        let mut toks = Vec::new();
        let cs: Vec<char> = src.chars().collect();
        let mut i = 0;
        while i < cs.len() {
            let c = cs[i];
            if c.is_whitespace() {
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
                        let Some(&e) = cs.get(i) else {
                            return err(format!("dangling escape in `{src}`"));
                        };
                        i += 1;
                        match e {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            'r' => s.push('\r'),
                            '\\' | '\'' | '"' => s.push(e),
                            other => {
                                s.push('\\');
                                s.push(other);
                            }
                        }
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
        Ok(Lexer { toks, at: 0 })
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
            e = Expr::Bin(Box::new(e), op, Box::new(self.concat()?));
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
                let args = if self.eat_op("(") {
                    self.args()?.0
                } else {
                    Vec::new()
                };
                e = Expr::Filter(Box::new(e), name, args);
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

type Ns = Rc<RefCell<Map<String, Value>>>;

#[derive(Clone, Debug)]
enum V {
    Undef,
    J(Value),
    Ns(Ns),
}

impl V {
    fn truthy(&self) -> bool {
        match self {
            V::Undef => false,
            V::Ns(_) => true,
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

    fn into_json(self) -> Value {
        match self {
            V::Undef => Value::Null,
            V::J(v) => v,
            V::Ns(ns) => Value::Object(ns.borrow().clone()),
        }
    }

    fn render(&self, out: &mut String) {
        match self {
            V::Undef => {}
            V::Ns(_) => out.push_str("<Namespace>"),
            V::J(v) => render_py(v, out),
        }
    }
}

/// Python `str()` of a JSON value.
fn render_py(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => match (n.as_i64(), n.as_f64()) {
            (Some(i), _) => {
                let _ = write!(out, "{i}");
            }
            (None, Some(f)) if f.fract() == 0.0 && f.abs() < 1e16 => {
                let _ = write!(out, "{f:.1}");
            }
            _ => {
                let _ = write!(out, "{n}");
            }
        },
        Value::String(s) => out.push_str(s),
        other => out.push_str(&py_json(other)),
    }
}

/// `json.dumps(v, ensure_ascii=False)`: `", "` and `": "` separators.
fn py_json(v: &Value) -> String {
    let mut s = String::new();
    py_json_into(v, &mut s);
    s
}

fn py_json_into(v: &Value, s: &mut String) {
    match v {
        Value::Array(a) => {
            s.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                py_json_into(x, s);
            }
            s.push(']');
        }
        Value::Object(o) => {
            s.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    s.push_str(", ");
                }
                s.push_str(&Value::String(k.clone()).to_string());
                s.push_str(": ");
                py_json_into(x, s);
            }
            s.push('}');
        }
        scalar => s.push_str(&scalar.to_string()),
    }
}

// ---------------------------------------------------------------- render

struct Renderer {
    frames: Vec<HashMap<String, V>>,
    out: String,
}

impl Renderer {
    fn lookup(&self, name: &str) -> V {
        self.frames
            .iter()
            .rev()
            .find_map(|f| f.get(name).cloned())
            .unwrap_or(V::Undef)
    }

    fn assign(&mut self, name: String, v: V) {
        if let Some(f) = self.frames.last_mut() {
            f.insert(name, v);
        }
    }

    fn nodes(&mut self, nodes: &[Node]) -> Result<(), TemplateError> {
        for n in nodes {
            self.node(n)?;
        }
        Ok(())
    }

    fn node(&mut self, n: &Node) -> Result<(), TemplateError> {
        match n {
            Node::Text(t) => self.out.push_str(t),
            Node::Out(e) => {
                let v = self.eval(e)?;
                v.render(&mut self.out);
            }
            Node::If(arms, els) => {
                for (cond, body) in arms {
                    if self.eval(cond)?.truthy() {
                        return self.nodes(body);
                    }
                }
                self.nodes(els)?;
            }
            Node::Set { name, attr, value } => {
                let v = self.eval(value)?;
                match attr {
                    None => self.assign(name.clone(), v),
                    Some(a) => match self.lookup(name) {
                        V::Ns(ns) => {
                            ns.borrow_mut().insert(a.clone(), v.into_json());
                        }
                        _ => return err(format!("set {name}.{a}: {name} is not a namespace")),
                    },
                }
            }
            Node::For {
                targets,
                iter,
                body,
            } => {
                let items: Vec<Value> = match self.eval(iter)? {
                    V::Undef => Vec::new(),
                    V::J(Value::Array(a)) => a,
                    V::J(Value::Object(o)) => o.keys().map(|k| Value::String(k.clone())).collect(),
                    V::J(Value::String(s)) => {
                        s.chars().map(|c| Value::String(c.to_string())).collect()
                    }
                    V::J(Value::Null) => Vec::new(),
                    other => return err(format!("cannot iterate {other:?}")),
                };
                let len = items.len();
                for (i, item) in items.into_iter().enumerate() {
                    let mut frame = HashMap::new();
                    let lp = serde_json::json!({
                        "index0": i, "index": i + 1, "first": i == 0, "last": i + 1 == len,
                        "length": len, "revindex": len - i, "revindex0": len - i - 1,
                    });
                    frame.insert("loop".to_owned(), V::J(lp));
                    if targets.len() == 1 {
                        frame.insert(targets[0].clone(), V::J(item));
                    } else {
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
                    }
                    self.frames.push(frame);
                    let r = self.nodes(body);
                    self.frames.pop();
                    r?;
                }
            }
        }
        Ok(())
    }

    fn eval(&mut self, e: &Expr) -> Result<V, TemplateError> {
        Ok(match e {
            Expr::Lit(v) => V::J(v.clone()),
            Expr::Name(n) => self.lookup(n),
            Expr::List(items) => V::J(Value::Array(
                items
                    .iter()
                    .map(|x| self.eval(x).map(V::into_json))
                    .collect::<Result<_, _>>()?,
            )),
            Expr::Dict(items) => {
                let mut m = Map::new();
                for (k, v) in items {
                    let key = match self.eval(k)? {
                        V::J(Value::String(s)) => s,
                        other => return err(format!("dict key must be a string, got {other:?}")),
                    };
                    m.insert(key, self.eval(v)?.into_json());
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
            Expr::Filter(x, name, args) => {
                let v = self.eval(x)?;
                let args: Vec<V> = args
                    .iter()
                    .map(|a| self.eval(a))
                    .collect::<Result<_, _>>()?;
                filter(v, name, &args)?
            }
            Expr::Test(x, name, negate) => {
                let v = self.eval(x)?;
                V::J(Value::Bool(test(&v, name)? != *negate))
            }
            Expr::Not(x) => V::J(Value::Bool(!self.eval(x)?.truthy())),
            Expr::Neg(x) => match self.eval(x)? {
                V::J(Value::Number(n)) => match n.as_i64() {
                    Some(i) => V::J(Value::from(-i)),
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
        match callee {
            Expr::Name(n) if n == "namespace" => {
                let mut m = Map::new();
                for (k, v) in kwargs {
                    m.insert(k.clone(), self.eval(v)?.into_json());
                }
                Ok(V::Ns(Rc::new(RefCell::new(m))))
            }
            Expr::Name(n) if n == "raise_exception" => {
                let mut msg = String::new();
                if let Some(a) = args.first() {
                    a.render(&mut msg);
                }
                err(format!("raise_exception: {msg}"))
            }
            Expr::Name(n) if n == "range" => {
                let ints: Vec<i64> = args.iter().filter_map(as_i64).collect();
                let (lo, hi) = match ints.as_slice() {
                    [hi] => (0, *hi),
                    [lo, hi, ..] => (*lo, *hi),
                    _ => return err("range needs integer bounds"),
                };
                Ok(V::J(Value::Array((lo..hi).map(Value::from).collect())))
            }
            Expr::Attr(obj, method) => {
                let o = self.eval(obj)?;
                call_method(&o, method, &args)
            }
            other => err(format!("unsupported call {other:?}")),
        }
    }
}

fn as_i64(v: &V) -> Option<i64> {
    match v {
        V::J(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

fn get_attr(o: &V, name: &str) -> V {
    match o {
        V::Ns(ns) => ns.borrow().get(name).cloned().map_or(V::Undef, V::J),
        V::J(Value::Object(m)) => m.get(name).cloned().map_or(V::Undef, V::J),
        _ => V::Undef,
    }
}

fn index(o: &V, i: &V) -> V {
    match (o, i) {
        (V::J(Value::Object(m)), V::J(Value::String(k))) => {
            m.get(k).cloned().map_or(V::Undef, V::J)
        }
        (V::Ns(_), V::J(Value::String(k))) => get_attr(o, k),
        (V::J(Value::Array(a)), V::J(Value::Number(n))) => {
            let Some(i) = n.as_i64() else { return V::Undef };
            let len = i64::try_from(a.len()).unwrap_or(i64::MAX);
            let i = if i < 0 { i + len } else { i };
            usize::try_from(i)
                .ok()
                .and_then(|i| a.get(i))
                .cloned()
                .map_or(V::Undef, V::J)
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

fn call_method(o: &V, method: &str, args: &[V]) -> Result<V, TemplateError> {
    let str_arg = |i: usize| match args.get(i) {
        Some(V::J(Value::String(s))) => Some(s.clone()),
        _ => None,
    };
    Ok(match (o, method) {
        (V::J(Value::Object(m)), "items") => V::J(Value::Array(
            m.iter()
                .map(|(k, v)| Value::Array(vec![Value::String(k.clone()), v.clone()]))
                .collect(),
        )),
        (V::J(Value::Object(m)), "keys") => V::J(Value::Array(
            m.keys().map(|k| Value::String(k.clone())).collect(),
        )),
        (V::J(Value::Object(m)), "values") => V::J(Value::Array(m.values().cloned().collect())),
        (V::J(Value::Object(m)), "get") => {
            let k = str_arg(0).ok_or_else(|| TemplateError("get() needs a string key".into()))?;
            match m.get(&k) {
                Some(v) => V::J(v.clone()),
                None => args.get(1).cloned().unwrap_or(V::J(Value::Null)),
            }
        }
        (V::J(Value::String(s)), "strip" | "lstrip" | "rstrip") => {
            // Python's rule: no argument (or `none`) strips whitespace, a string
            // strips any of its characters.
            let set: Option<Vec<char>> = match args.first() {
                None | Some(V::J(Value::Null)) => None,
                Some(V::J(Value::String(c))) => Some(c.chars().collect()),
                Some(other) => return err(format!("{method}() takes a string, not {other:?}")),
            };
            let strips = |c: char| set.as_ref().map_or(c.is_whitespace(), |s| s.contains(&c));
            V::J(Value::String(
                match method {
                    "strip" => s.trim_matches(strips),
                    "lstrip" => s.trim_start_matches(strips),
                    _ => s.trim_end_matches(strips),
                }
                .to_owned(),
            ))
        }
        (V::J(Value::String(s)), "split") => V::J(Value::Array(
            split(s, args.first(), args.get(1))?
                .into_iter()
                .map(Value::String)
                .collect(),
        )),
        (V::J(Value::String(s)), "startswith") => {
            V::J(Value::Bool(str_arg(0).is_some_and(|p| s.starts_with(&p))))
        }
        (V::J(Value::String(s)), "endswith") => {
            V::J(Value::Bool(str_arg(0).is_some_and(|p| s.ends_with(&p))))
        }
        (other, m) => return err(format!("no method {m}() on {other:?}")),
    })
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
    let mut rest = s.trim_start();
    while !rest.is_empty() {
        if max.is_some_and(|m| out.len() == m) {
            out.push(rest.to_owned());
            break;
        }
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        out.push(rest[..end].to_owned());
        rest = rest[end..].trim_start();
    }
    out
}

fn filter(v: V, name: &str, _args: &[V]) -> Result<V, TemplateError> {
    Ok(match name {
        "tojson" => V::J(Value::String(py_json(&v.into_json()))),
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
            _ => 0,
        })),
        "trim" => V::J(Value::String(to_str(&v).trim().to_owned())),
        "string" => V::J(Value::String(to_str(&v))),
        "lower" => V::J(Value::String(to_str(&v).to_lowercase())),
        "upper" => V::J(Value::String(to_str(&v).to_uppercase())),
        other => return err(format!("unsupported filter |{other}")),
    })
}

fn to_str(v: &V) -> String {
    let mut s = String::new();
    v.render(&mut s);
    s
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
        "number" => matches!(v, V::J(Value::Number(_))),
        "boolean" => matches!(v, V::J(Value::Bool(_))),
        "mapping" => matches!(v, V::J(Value::Object(_)) | V::Ns(_)),
        "sequence" | "iterable" => matches!(
            v,
            V::J(Value::Array(_) | Value::String(_) | Value::Object(_))
        ),
        other => return err(format!("unsupported test `is {other}`")),
    })
}

fn num(v: &V) -> Option<f64> {
    match v {
        V::J(Value::Number(n)) => n.as_f64(),
        V::J(Value::Bool(b)) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn eq(a: &V, b: &V) -> bool {
    match (a, b) {
        (V::Undef, V::Undef) => true,
        (V::J(x), V::J(y)) => match (num(a), num(b)) {
            (Some(p), Some(q)) if x.is_number() && y.is_number() => p == q,
            _ => x == y,
        },
        _ => false,
    }
}

fn binop(l: &V, op: BinOp, r: &V) -> Result<V, TemplateError> {
    use BinOp::*;
    let int_pair = match (l, r) {
        (V::J(Value::Number(a)), V::J(Value::Number(b))) => a.as_i64().zip(b.as_i64()),
        _ => None,
    };
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
                V::J(Value::String(s)) => {
                    matches!(l, V::J(Value::String(k)) if s.contains(k.as_str()))
                }
                V::Undef => false,
                other => return err(format!("`in` on {other:?}")),
            };
            Value::Bool(found == (op == In))
        }
        Concat => Value::String(to_str(l) + &to_str(r)),
        Add => match (l, r) {
            (V::J(Value::String(a)), V::J(Value::String(b))) => Value::String(format!("{a}{b}")),
            (V::J(Value::Array(a)), V::J(Value::Array(b))) => {
                Value::Array(a.iter().chain(b).cloned().collect())
            }
            _ => match (int_pair, num(l), num(r)) {
                (Some((a, b)), _, _) => Value::from(
                    a.checked_add(b)
                        .ok_or_else(|| TemplateError("overflow".into()))?,
                ),
                (None, Some(a), Some(b)) => Value::from(a + b),
                _ => return err(format!("cannot add {l:?} and {r:?}")),
            },
        },
        Sub | Mul | Div | FloorDiv | Mod => {
            if let (Some((a, b)), true) = (int_pair, op != Div) {
                let v = match op {
                    Sub => a.checked_sub(b),
                    Mul => a.checked_mul(b),
                    FloorDiv => (b != 0).then(|| a.div_euclid(b)),
                    _ => (b != 0).then(|| a.rem_euclid(b)),
                };
                Value::from(
                    v.ok_or_else(|| TemplateError("integer overflow or division by zero".into()))?,
                )
            } else {
                let (Some(a), Some(b)) = (num(l), num(r)) else {
                    return err(format!("arithmetic on {l:?} and {r:?}"));
                };
                Value::from(match op {
                    Sub => a - b,
                    Mul => a * b,
                    Div => a / b,
                    FloorDiv => (a / b).floor(),
                    _ => a.rem_euclid(b),
                })
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
    /// `from_json` and `.items()` (keys sorted), results as `<tool_result>`.
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
                 <｜DSML｜parameter name=\"s\" string=\"true\">v</｜DSML｜parameter>\n\
                 <｜DSML｜parameter name=\"x\" string=\"false\">1</｜DSML｜parameter>\n\
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
}
