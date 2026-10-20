//! A request's fields as a program of text.
//!
//! The JSON-schema subset (draft 2020-12 keywords, annotations ignored):
//! - `type: object` with `properties`: every property is written, in the
//!   schema's order, so the output holds each `required` property whether or
//!   not the schema lists it. `required` may only name properties;
//!   `additionalProperties` is accepted (none are written).
//! - `type: string`, with an optional `maxLength`: a free string.
//! - `enum` of strings, of integers or of booleans (one kind), and `const` of
//!   one of those; `type: boolean`.
//! - Any other keyword (`$ref`, `anyOf`, `items`, `pattern`, `minLength`,
//!   `format`, a bare `type: integer`, ...) is a refusal naming it and its path.
//!
//! The JSON text is compact with one space after `:` and `,`. A `<` in a JSON
//! string's text (a constant, an enum option, a property name) is written as
//! `\u003c`, and a free JSON string inside a tool call's markup, or in a reply
//! the request's tool-call scan reads, takes no token with a `<`
//! ([`Style::JsonInMarkup`]), so the text never spells a tag a chat format's
//! scanner looks for; a free string in a reply no scan reads may hold one. A
//! raw string literal (a tool argument in a format that writes strings
//! unquoted) holding a `<` is a refusal. A tool call's arguments use the same
//! subset, written in the style the chat format reads; its parameters are read
//! as one object by [`value`].

use serde_json::{Map, Value};

use super::Refusal;
use super::program::{self, Alt, Constraint, Env, Part, Style};
use super::vocab::Vocab;
use crate::dsml::{ToolFormat, parses_tools};
use crate::engine::Tokenizer;
use crate::reasoning::ThinkTags;

/// Keywords the subset reads, besides the annotations.
const READ: [&str; 7] = [
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "maxLength",
];

/// Keywords that describe a schema and constrain nothing.
const ANNOTATIONS: [&str; 10] = [
    "title",
    "description",
    "$schema",
    "$id",
    "$comment",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// What the fields are read against.
pub(crate) struct Ask<'a> {
    pub tok: &'a dyn Tokenizer,
    pub vocab: &'a Vocab,
    pub format: ToolFormat,
    /// The tags of the model's think span: a prompt that left the span open
    /// has it closed with `think.close` before the program.
    pub think: ThinkTags,
}

fn refuse(path: &str, message: impl std::fmt::Display) -> Refusal {
    Refusal::invalid(format!("{path}: {message}"))
}

fn type_of(v: &Value) -> &'static str {
    ::embed::request::json_type(v)
}

/// The text a JSON string body spells `s` as: the escapes of its characters,
/// without the quotes, and a `<` as `\u003c`.
fn json_inner(s: &str) -> String {
    let q = Value::String(s.to_owned()).to_string();
    q[1..q.len() - 1].replace('<', "\\u003c")
}

/// A scalar a choice or a constant can take.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Scalar {
    Str(String),
    Int(i128),
    Bool(bool),
}

impl Scalar {
    fn of(v: &Value, path: &str) -> Result<Scalar, Refusal> {
        match v {
            Value::String(s) => Ok(Scalar::Str(s.clone())),
            Value::Bool(b) => Ok(Scalar::Bool(*b)),
            Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => Ok(Scalar::Int(i128::from(i))),
                (_, Some(u)) => Ok(Scalar::Int(i128::from(u))),
                _ => Err(refuse(
                    path,
                    format!(
                        "the number {n} is not an integer: only strings, integers and booleans can be chosen or fixed"
                    ),
                )),
            },
            other => Err(refuse(
                path,
                format!(
                    "a {} cannot be chosen or fixed: only strings, integers and booleans can",
                    type_of(other)
                ),
            )),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Scalar::Str(_) => "string",
            Scalar::Int(_) => "integer",
            Scalar::Bool(_) => "boolean",
        }
    }

    /// The text between a string's quotes, or a number's or boolean's whole
    /// text.
    fn body(&self, style: Style) -> String {
        match self {
            Scalar::Str(s) if style.is_json() => json_inner(s),
            Scalar::Str(s) => s.clone(),
            Scalar::Int(i) => i.to_string(),
            Scalar::Bool(b) => b.to_string(),
        }
    }

    /// Whether the value is written between quotes.
    fn quoted(&self, style: Style) -> bool {
        matches!(self, Scalar::Str(_)) && style.is_json()
    }
}

/// Whether a node's value is text: what the chat formats read as a raw
/// string. A `type: string`, a string `const` or a string `enum`.
pub(crate) fn is_text(node: &Value) -> bool {
    node.get("type").and_then(Value::as_str) == Some("string")
        || node.get("const").is_some_and(Value::is_string)
        || node
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|e| e.iter().any(Value::is_string))
}

/// Refuses a keyword the node's kind does not read.
fn only(o: &Map<String, Value>, kind: &str, allowed: &[&str], path: &str) -> Result<(), Refusal> {
    match o
        .keys()
        .find(|k| !allowed.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()))
    {
        Some(k) => Err(refuse(
            path,
            format!("the keyword \"{k}\" is not supported by this server beside {kind}"),
        )),
        None => Ok(()),
    }
}

/// `node` as the parts that write one value of it. `style` is how a string is
/// written: quoted and escaped (JSON, in a reply of its own or inside a chat
/// format's markup), or as the raw text a chat format's string argument holds.
pub(crate) fn value(node: &Value, style: Style, path: &str) -> Result<Vec<Part>, Refusal> {
    let Some(o) = node.as_object() else {
        return Err(refuse(
            path,
            format!("a schema must be an object, not {}", type_of(node)),
        ));
    };
    if let Some(k) = o
        .keys()
        .find(|k| !READ.contains(&k.as_str()) && !ANNOTATIONS.contains(&k.as_str()))
    {
        return Err(refuse(
            path,
            format!(
                "the keyword \"{k}\" is not supported by this server (read: {}, and annotations)",
                READ.join(", ")
            ),
        ));
    }
    let ty = match o.get("type") {
        None => None,
        Some(Value::String(t)) => Some(t.as_str()),
        Some(v) => {
            return Err(refuse(
                path,
                format!("type must be one string, not {}", type_of(v)),
            ));
        }
    };
    if style == Style::Raw {
        raw_literals_hold_no_tag(o, path)?;
    }
    if let Some(c) = o.get("const") {
        only(o, "const", &["const", "type"], path)?;
        let s = Scalar::of(c, &format!("{path}.const"))?;
        agree(ty, s.kind(), path)?;
        return Ok(spell(&s, style));
    }
    if let Some(e) = o.get("enum") {
        only(o, "enum", &["enum", "type"], path)?;
        return enumeration(e, ty, style, path);
    }
    match ty {
        Some("object") => {
            only(
                o,
                "type object",
                &["type", "properties", "required", "additionalProperties"],
                path,
            )?;
            object(o, style.nested(), path)
        }
        Some("string") => {
            only(o, "type string", &["type", "maxLength"], path)?;
            let max = match o.get("maxLength") {
                None => None,
                Some(v) => Some(
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| {
                            refuse(
                                path,
                                format!("maxLength must be a non-negative integer, not {v}"),
                            )
                        })?,
                ),
            };
            Ok(match style {
                Style::Json | Style::JsonInMarkup => vec![
                    Part::Text("\"".to_owned()),
                    Part::Free { max, style },
                    Part::Text("\"".to_owned()),
                ],
                Style::Raw => vec![Part::Free { max, style }],
            })
        }
        Some("boolean") => {
            only(o, "type boolean", &["type"], path)?;
            Ok(vec![Part::Pick(
                ["true", "false"]
                    .map(|t| Alt {
                        text: t.to_owned(),
                        then: Vec::new(),
                    })
                    .to_vec(),
            )])
        }
        Some(other) => Err(refuse(
            path,
            format!(
                "type \"{other}\" is not supported by this server without an enum or a const \
                 (supported: object, string, boolean)"
            ),
        )),
        None => Err(refuse(
            path,
            "a schema needs a type, an enum or a const to be served",
        )),
    }
}

/// Refuses a string literal (`const` or `enum`) with a `<` where the value is
/// written raw: a raw value ends at the text after it, and a `<` is how that
/// text begins.
fn raw_literals_hold_no_tag(o: &Map<String, Value>, path: &str) -> Result<(), Refusal> {
    let literals = o.get("const").into_iter().chain(
        o.get("enum")
            .and_then(Value::as_array)
            .into_iter()
            .flatten(),
    );
    if literals.filter_map(Value::as_str).any(|s| s.contains('<')) {
        return Err(refuse(
            path,
            "raw string literals cannot contain '<': the chat format reads the value up to its \
             closing tag",
        ));
    }
    Ok(())
}

/// A node's `type` against the kind of its values.
fn agree(ty: Option<&str>, kind: &str, path: &str) -> Result<(), Refusal> {
    match ty {
        Some(t) if t != kind => Err(refuse(
            path,
            format!("type \"{t}\" does not match its {kind} values"),
        )),
        _ => Ok(()),
    }
}

/// One scalar written as parts: quoted when it is a JSON string.
fn spell(s: &Scalar, style: Style) -> Vec<Part> {
    let body = s.body(style);
    vec![Part::Text(if s.quoted(style) {
        format!("\"{body}\"")
    } else {
        body
    })]
}

fn enumeration(
    e: &Value,
    ty: Option<&str>,
    style: Style,
    path: &str,
) -> Result<Vec<Part>, Refusal> {
    let here = format!("{path}.enum");
    let Some(items) = e.as_array().filter(|a| !a.is_empty()) else {
        return Err(refuse(
            &here,
            format!("an enum must be a non-empty array, not {e}"),
        ));
    };
    let values: Vec<Scalar> = items
        .iter()
        .map(|v| Scalar::of(v, &here))
        .collect::<Result<_, _>>()?;
    let kind = values[0].kind();
    if let Some(odd) = values.iter().find(|v| v.kind() != kind) {
        return Err(refuse(
            &here,
            format!(
                "an enum of mixed kinds ({kind} and {}) is not supported by this server",
                odd.kind()
            ),
        ));
    }
    agree(ty, kind, path)?;
    for (i, v) in values.iter().enumerate() {
        if values[..i].contains(v) {
            return Err(refuse(
                &here,
                format!("{:?} is listed twice", v.body(style)),
            ));
        }
        if v.body(style).is_empty() {
            return Err(refuse(
                &here,
                "an empty string cannot be one of the options",
            ));
        }
    }
    if values.len() == 1 {
        return Ok(spell(&values[0], style));
    }
    let quoted = values[0].quoted(style);
    let pick = Part::Pick(
        values
            .iter()
            .map(|v| Alt {
                text: v.body(style),
                then: Vec::new(),
            })
            .collect(),
    );
    Ok(if quoted {
        vec![
            Part::Text("\"".to_owned()),
            pick,
            Part::Text("\"".to_owned()),
        ]
    } else {
        vec![pick]
    })
}

/// An object written as JSON: every property, in the schema's order, its
/// strings in the style `inner`.
fn object(o: &Map<String, Value>, inner: Style, path: &str) -> Result<Vec<Part>, Refusal> {
    let props = match o.get("properties") {
        None => None,
        Some(Value::Object(p)) => Some(p),
        Some(v) => {
            return Err(refuse(
                path,
                format!("properties must be an object, not {}", type_of(v)),
            ));
        }
    };
    if let Some(r) = o.get("required") {
        let Some(names) = r.as_array() else {
            return Err(refuse(
                path,
                format!("required must be an array, not {}", type_of(r)),
            ));
        };
        for n in names {
            let known = n
                .as_str()
                .is_some_and(|n| props.is_some_and(|p| p.contains_key(n)));
            if !known {
                return Err(refuse(
                    path,
                    format!("required names {n}, which properties does not hold"),
                ));
            }
        }
    }
    let args: Vec<Arg<'_>> = props
        .into_iter()
        .flatten()
        .map(|(key, schema)| Arg {
            key,
            schema,
            path: format!("{path}.properties.{key}"),
        })
        .collect();
    object_of(&args, inner)
}

/// An object whose properties are `args`, as JSON with its strings in the
/// style `inner`.
fn object_of(args: &[Arg<'_>], inner: Style) -> Result<Vec<Part>, Refusal> {
    let mut parts = vec![Part::Text("{".to_owned())];
    for (i, a) in args.iter().enumerate() {
        let sep = if i == 0 { "" } else { ", " };
        parts.push(Part::Text(format!("{sep}\"{}\": ", json_inner(a.key))));
        parts.extend(value(a.schema, inner, &a.path)?);
    }
    parts.push(Part::Text("}".to_owned()));
    Ok(parts)
}

/// One argument of a tool: its name and schema.
pub(crate) struct Arg<'a> {
    pub key: &'a str,
    pub schema: &'a Value,
    pub path: String,
}

/// A tool the request names.
pub(crate) struct Fun<'a> {
    pub name: &'a str,
    pub args: Vec<Arg<'a>>,
}

impl Fun<'_> {
    /// The arguments as one JSON object inside the call's markup.
    pub(crate) fn object(&self) -> Result<Vec<Part>, Refusal> {
        object_of(&self.args, Style::JsonInMarkup)
    }
}

/// A call as parts: `open`, then the function's name, then what `body` writes
/// after it. With several functions the name is a choice, each option followed
/// by its own function's body; with one it is forced.
pub(crate) fn call(
    funs: &[Fun<'_>],
    open: &str,
    name: impl Fn(&str) -> String,
    body: impl Fn(&Fun<'_>) -> Result<Vec<Part>, Refusal>,
) -> Result<Vec<Part>, Refusal> {
    if let [f] = funs {
        let mut parts = vec![Part::Text(format!("{open}{}", name(f.name)))];
        parts.extend(body(f)?);
        return Ok(parts);
    }
    let alts = funs
        .iter()
        .map(|f| {
            Ok(Alt {
                text: name(f.name),
                then: body(f)?,
            })
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    Ok(vec![Part::Text(open.to_owned()), Part::Pick(alts)])
}

/// The name a function or an argument may have: text a chat format can write
/// between its tags without escaping.
fn plain_name(what: &str, name: &str, path: &str) -> Result<(), Refusal> {
    if name.is_empty() || name.contains(|c: char| c.is_control() || "\"\\<>".contains(c)) {
        return Err(refuse(
            path,
            format!(
                "{what} {name:?} is not a plain name (non-empty, none of \" \\ < > or a control character)"
            ),
        ));
    }
    Ok(())
}

/// The tools of a request, with their arguments in the schema's order.
fn funs<'a>(tools: &'a Value) -> Result<Vec<Fun<'a>>, Refusal> {
    let Some(list) = tools.as_array().filter(|t| !t.is_empty()) else {
        return Err(refuse("tools", "tool_choice needs a non-empty tools array"));
    };
    let mut out: Vec<Fun<'a>> = Vec::new();
    for (i, tool) in list.iter().enumerate() {
        let path = format!("tools[{i}]");
        let f = tool.get("function").unwrap_or(tool);
        let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
        plain_name("the tool name", name, &path)?;
        if out.iter().any(|g| g.name == name) {
            return Err(refuse(&path, format!("the tool {name} is listed twice")));
        }
        let here = format!("{path}.{name}.parameters");
        let args = match f.get("parameters") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Object(p)) => {
                if let Some(t) = p.get("type").filter(|t| t.as_str() != Some("object")) {
                    return Err(refuse(
                        &here,
                        format!("the parameters' type must be \"object\", not {t}"),
                    ));
                }
                // The schema compiler reads the whole object, so a keyword the
                // subset does not read is a refusal here and never ignored.
                let mut root = p.clone();
                root.entry("type")
                    .or_insert_with(|| Value::String("object".to_owned()));
                value(&Value::Object(root), Style::JsonInMarkup, &here)?;
                p.get("properties")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .map(|(key, schema)| {
                        plain_name("the parameter name", key, &here)?;
                        Ok(Arg {
                            key,
                            schema,
                            path: format!("{here}.properties.{key}"),
                        })
                    })
                    .collect::<Result<_, Refusal>>()?
            }
            Some(v) => {
                return Err(refuse(
                    &here,
                    format!("parameters must be an object, not {}", type_of(v)),
                ));
            }
        };
        out.push(Fun { name, args });
    }
    Ok(out)
}

/// What a request's `tool_choice` forces.
enum Forces<'a> {
    Any,
    Named(&'a str),
}

fn tool_choice(o: &Map<String, Value>) -> Result<Option<Forces<'_>>, Refusal> {
    let Some(c) = o.get("tool_choice").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    match c {
        Value::String(s) => match s.as_str() {
            "none" | "auto" => Ok(None),
            "required" => Ok(Some(Forces::Any)),
            other => Err(refuse(
                "tool_choice",
                format!(
                    "\"{other}\" is not supported by this server (none, auto, required or a named function)"
                ),
            )),
        },
        Value::Object(f) => {
            let named = f
                .get("function")
                .and_then(|g| g.get("name"))
                .or_else(|| f.get("name"))
                .and_then(Value::as_str);
            match (f.get("type").and_then(Value::as_str), named) {
                (Some("function"), Some(n)) => Ok(Some(Forces::Named(n))),
                _ => Err(refuse(
                    "tool_choice",
                    format!(
                        "{c} is not supported by this server (a named function is {{\"type\": \"function\", \"function\": {{\"name\": ...}}}})"
                    ),
                )),
            }
        }
        other => Err(refuse(
            "tool_choice",
            format!("must be a string or an object, not {}", type_of(other)),
        )),
    }
}

/// The parts of a forced tool call, written in the chat format's markup.
fn tool_parts(
    o: &Map<String, Value>,
    forces: &Forces<'_>,
    format: ToolFormat,
) -> Result<Vec<Part>, Refusal> {
    let tools = o
        .get("tools")
        .filter(|v| !v.is_null())
        .ok_or_else(|| refuse("tool_choice", "forcing a call needs the request's tools"))?;
    let mut funs = funs(tools)?;
    if let Forces::Named(n) = forces {
        funs.retain(|f| f.name == *n);
        if funs.is_empty() {
            return Err(refuse(
                "tool_choice",
                format!("the function {n} is not among the request's tools"),
            ));
        }
    }
    format.call_parts(&funs)
}

/// `response_format`'s schema, or `None` for a request that asks for text.
fn response_schema(o: &Map<String, Value>) -> Result<Option<&Value>, Refusal> {
    let Some(rf) = o.get("response_format").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    match rf.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .or_else(|| rf.get("schema"));
            match schema {
                Some(s @ Value::Object(_)) => Ok(Some(s)),
                _ => Err(refuse(
                    "response_format",
                    "type json_schema needs json_schema.schema, an object",
                )),
            }
        }
        Some("json_object") => Err(refuse(
            "response_format",
            "type json_object is not supported by this server: it needs a schema (send type \
             json_schema)",
        )),
        _ => Err(refuse(
            "response_format",
            format!("{rf} is not supported by this server (type text or json_schema)"),
        )),
    }
}

/// The constraint a request's fields ask for, `None` for a request that
/// asks for none. A tool call and a schema together are refused: a request
/// has one program.
pub(crate) fn request(
    o: &Map<String, Value>,
    ask: &Ask<'_>,
) -> Result<Option<Constraint>, Refusal> {
    if o.get("json_schema").is_some_and(|v| !v.is_null()) {
        return Err(Refusal::invalid(
            "json_schema is not supported by this server: send the schema as response_format \
             {\"type\": \"json_schema\", \"json_schema\": {\"schema\": ...}}"
                .to_owned(),
        ));
    }
    let schema = response_schema(o)?;
    let forces = tool_choice(o)?;
    let parts = match (schema, &forces) {
        (Some(_), Some(_)) => {
            return Err(refuse(
                "response_format",
                "a json_schema and a forced tool_choice together are not supported by this \
                 server: a request has one program",
            ));
        }
        (Some(s), None) => {
            // The reply is read for a call's markup when the request has tools
            // to scan for, so its strings keep clear of a `<` then.
            let style = if parses_tools(o) {
                Style::JsonInMarkup
            } else {
                Style::Json
            };
            value(s, style, "response_format.json_schema.schema")?
        }
        (None, Some(f)) => tool_parts(o, f, ask.format)?,
        (None, None) => return Ok(None),
    };
    let env = Env {
        tok: ask.tok,
        vocab: ask.vocab,
    };
    let prog = program::compile(&parts, &env)?;
    Ok(Some(Constraint::new(
        prog,
        ask.tok.encode(&format!("{}\n\n", ask.think.close)),
        ask.vocab.stops().clone(),
    )))
}
