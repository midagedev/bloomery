//! A request's fields as a program of text.
//!
//! The JSON-schema subset (draft 2020-12 keywords, annotations ignored):
//! - `type: object` with `properties`: the properties `required` names are
//!   each written, in the schema's order, then the others in order, each of
//!   which may be left out — llama.cpp's object rule
//!   (`json-schema-to-grammar.cpp` `_build_object_rule`). `required` may
//!   only name properties; `additionalProperties` is accepted (none are
//!   written).
//! - `type: string`, with an optional `maxLength`: a free string.
//! - `type: integer` and `type: number`: a free JSON number; `type: boolean`
//!   and `type: null`.
//! - `type: array` with `items`: any count of items, each of the `items`
//!   schema.
//! - `anyOf`: one of its schemas.
//! - `$ref` into the schema itself (`#` or a JSON pointer after it, such as
//!   `#/$defs/name`): the schema it points to, written out where it stands;
//!   `$defs` and `definitions` hold such schemas and constrain nothing. A
//!   `$ref` met again inside the schema it points to is a refusal.
//! - `enum` of strings, of integers or of booleans (one kind), and `const` of
//!   one of those.
//! - Any other keyword (`oneOf`, `allOf`, `pattern`, `minLength`, `minItems`,
//!   `minimum`, `format`, ...) is a refusal naming it and its path.
//!
//! `response_format` `json_object` asks for any JSON object with at least one
//! member, written in the same style ([`program::json_object`]).
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

use std::borrow::Cow;
use std::sync::Arc;

use serde_json::{Map, Value};

use super::Refusal;
use super::program::{self, Alt, Constraint, Env, Opener, Part, Style};
use super::vocab::Vocab;
use crate::dsml::{ToolFormat, parses_tools};
use crate::engine::Tokenizer;
use crate::http::MAX_BODY;
use crate::reasoning::ThinkTags;

/// Keywords the subset reads, besides the annotations.
const READ: [&str; 9] = [
    "type",
    "enum",
    "const",
    "properties",
    "required",
    "additionalProperties",
    "maxLength",
    "items",
    "anyOf",
];

/// Keywords that hold schemas a `$ref` points to and constrain nothing.
const DEFS: [&str; 2] = ["$defs", "definitions"];

/// The schemas a schema may hold once its `$ref`s are written out.
const MAX_SCHEMAS: usize = 4096;

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
    pub vocab: &'a Arc<Vocab>,
    pub format: ToolFormat,
    /// The tags of the model's think span: a prompt that left the span open
    /// has it closed with `think.close` before the program.
    pub think: ThinkTags,
}

/// The refusal of the request field at `path`, its message led by the path.
pub(crate) fn refuse(path: &str, message: impl std::fmt::Display) -> Refusal {
    Refusal {
        field: Some(path.to_owned()),
        ..Refusal::invalid(format!("{path}: {message}"))
    }
}

fn type_of(v: &Value) -> &'static str {
    ::embed::request::json_type(v)
}

/// The text a JSON string body spells `s` as: the escapes of its characters,
/// without the quotes, and a `<` as `\u003c`. `/v1/systemone` cuts each option of a
/// question with the same text, so one function owns it.
fn json_inner(s: &str) -> String {
    ::decision::generative::json_body(s)
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
    if let Some(a) = o.get("anyOf") {
        only(o, "anyOf", &["anyOf"], path)?;
        return any_of(a, style, path);
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
        Some(t @ ("integer" | "number")) => {
            only(o, &format!("type {t}"), &["type"], path)?;
            Ok(vec![Part::Num {
                integer: t == "integer",
            }])
        }
        Some("null") => {
            only(o, "type null", &["type"], path)?;
            Ok(vec![Part::Text("null".to_owned())])
        }
        Some("array") => {
            only(o, "type array", &["type", "items"], path)?;
            let items = o
                .get("items")
                .ok_or_else(|| refuse(path, "type array needs items, the schema of each item"))?;
            Ok(vec![Part::List {
                item: value(items, style.nested(), &format!("{path}.items"))?,
            }])
        }
        Some(other) => Err(refuse(
            path,
            format!(
                "type \"{other}\" is not supported by this server (supported: object, array, \
                 string, integer, number, boolean, null)"
            ),
        )),
        None => Err(refuse(
            path,
            "a schema needs a type, an enum or a const to be served",
        )),
    }
}

/// One of `alts`, the schemas of an `anyOf`. Written raw, a text option is a
/// refusal: a chat format reads a raw value as text or as JSON by its schema,
/// which would not say which.
fn any_of(alts: &Value, style: Style, path: &str) -> Result<Vec<Part>, Refusal> {
    let here = format!("{path}.anyOf");
    let Some(list) = alts.as_array().filter(|a| !a.is_empty()) else {
        return Err(refuse(
            &here,
            format!("anyOf must be a non-empty array, not {alts}"),
        ));
    };
    if style == Style::Raw && list.iter().any(is_text) {
        return Err(refuse(
            &here,
            "an anyOf with a text option is not supported by this server in a chat format's \
             markup, which reads a value as text or as JSON by its schema",
        ));
    }
    let mut options = list
        .iter()
        .enumerate()
        .map(|(i, s)| value(s, style, &format!("{here}[{i}]")))
        .collect::<Result<Vec<_>, Refusal>>()?;
    Ok(if options.len() == 1 {
        options.remove(0)
    } else {
        vec![Part::Any(options)]
    })
}

/// `node` with every `$ref` in it replaced by the schema it points to in
/// `root`, and the definitions dropped. A `$ref` met again inside the schema
/// it points to is a refusal (a recursive schema is not served), and so is a
/// schema that writes out to more than [`MAX_SCHEMAS`] schemas or past the
/// `bytes` left: each schema's path, key and value is charged before it is
/// copied.
fn inline_refs(
    root: &Value,
    node: &Value,
    path: &str,
    seen: &mut Vec<String>,
    left: &mut usize,
    bytes: &mut usize,
) -> Result<Value, Refusal> {
    *left = left.checked_sub(1).ok_or_else(|| {
        refuse(
            path,
            format!(
                "the schema holds more than {MAX_SCHEMAS} schemas once its $refs are written out"
            ),
        )
    })?;
    let charge = |bytes: &mut usize, n: usize| -> Result<(), Refusal> {
        *bytes = bytes.checked_sub(n).ok_or_else(|| {
            refuse(
                path,
                format!("the schema passes {MAX_BODY} bytes once its $refs are written out"),
            )
        })?;
        Ok(())
    };
    charge(bytes, path.len())?;
    let Some(o) = node.as_object() else {
        charge(bytes, node.to_string().len())?;
        return Ok(node.clone());
    };
    if let Some(r) = o.get("$ref") {
        let Some(r) = r.as_str() else {
            return Err(refuse(path, format!("$ref must be a string, not {r}")));
        };
        if let Some(k) = o.keys().find(|k| {
            let k = k.as_str();
            k != "$ref" && !ANNOTATIONS.contains(&k) && !DEFS.contains(&k)
        }) {
            return Err(refuse(
                path,
                format!("the keyword \"{k}\" beside $ref is not supported by this server"),
            ));
        }
        let target = r
            .strip_prefix('#')
            .and_then(|p| root.pointer(p))
            .ok_or_else(|| {
                refuse(
                    path,
                    format!(
                        "$ref {r:?} does not point into this schema (\"#\" or a JSON pointer \
                         after it)"
                    ),
                )
            })?;
        if seen.iter().any(|s| s == r) {
            return Err(refuse(
                path,
                format!(
                    "$ref {r:?} is met again inside the schema it points to: a recursive \
                     schema is not supported by this server"
                ),
            ));
        }
        seen.push(r.to_owned());
        let out = inline_refs(root, target, path, seen, left, bytes);
        seen.pop();
        return out;
    }
    let mut out = Map::new();
    for (k, v) in o {
        if DEFS.contains(&k.as_str()) {
            continue;
        }
        charge(bytes, k.len())?;
        let v = match (k.as_str(), v) {
            ("properties", Value::Object(p)) => Value::Object(
                p.iter()
                    .map(|(name, s)| {
                        charge(bytes, name.len())?;
                        let here = format!("{path}.properties.{name}");
                        Ok((
                            name.clone(),
                            inline_refs(root, s, &here, seen, left, bytes)?,
                        ))
                    })
                    .collect::<Result<_, Refusal>>()?,
            ),
            ("items", s) => inline_refs(root, s, &format!("{path}.items"), seen, left, bytes)?,
            ("anyOf", Value::Array(a)) => Value::Array(
                a.iter()
                    .enumerate()
                    .map(|(i, s)| {
                        inline_refs(root, s, &format!("{path}.anyOf[{i}]"), seen, left, bytes)
                    })
                    .collect::<Result<_, Refusal>>()?,
            ),
            _ => {
                charge(bytes, v.to_string().len())?;
                v.clone()
            }
        };
        out.insert(k.clone(), v);
    }
    Ok(Value::Object(out))
}

/// `schema` with its `$ref`s written out ([`inline_refs`]) within the
/// request's `bytes` left of [`MAX_BODY`], refused when the JSON parser that
/// read the request could not read it back: a value nests no deeper than its
/// schema, so a call the program writes parses again.
fn written_out(schema: &Value, path: &str, bytes: &mut usize) -> Result<Value, Refusal> {
    let mut left = MAX_SCHEMAS;
    let out = inline_refs(schema, schema, path, &mut Vec::new(), &mut left, bytes)?;
    serde_json::from_str(&out.to_string()).map_err(|e| {
        refuse(
            path,
            format!("the schema is too deeply nested once its $refs are written out ({e})"),
        )
    })
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
    let required: Vec<&str> = match o.get("required") {
        None => Vec::new(),
        Some(r) => {
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
            names.iter().filter_map(Value::as_str).collect()
        }
    };
    let args: Vec<Arg<'_>> = props
        .into_iter()
        .flatten()
        .map(|(key, schema)| Arg {
            key,
            schema: Cow::Borrowed(schema),
            path: format!("{path}.properties.{key}"),
            required: required.contains(&key.as_str()),
        })
        .collect();
    object_of(&args, inner)
}

/// An object whose properties are `args`, as JSON with its strings in the
/// style `inner`: the properties `required` names, in the schema's order,
/// each written, then the others in order, each of which may be left out —
/// a comma-led chain where any optional may be the first member written, as
/// llama.cpp's object rule builds it.
fn object_of(args: &[Arg<'_>], inner: Style) -> Result<Vec<Part>, Refusal> {
    /// One member: `sep`, the key, and the property's value.
    fn member(a: &Arg<'_>, sep: &str, inner: Style) -> Result<Vec<Part>, Refusal> {
        let mut parts = vec![Part::Text(format!("{sep}\"{}\": ", json_inner(a.key)))];
        parts.extend(value(&a.schema, inner, &a.path)?);
        Ok(parts)
    }
    let optional: Vec<&Arg<'_>> = args.iter().filter(|a| !a.required).collect();
    let mut parts = vec![Part::Text("{".to_owned())];
    let mut written = false;
    for a in args.iter().filter(|a| a.required) {
        parts.extend(member(a, if written { ", " } else { "" }, inner)?);
        written = true;
    }
    if !written && !optional.is_empty() {
        // No property is required: any optional may be the object's first
        // member, which carries no comma of its own, and none at all is the
        // empty object.
        parts.push(any_start(
            &optional,
            &|a| member(a, "", inner),
            ", ",
            Vec::new(),
        )?);
    } else {
        for a in &optional {
            parts.push(Part::Any(vec![Vec::new(), member(a, ", ", inner)?]));
        }
    }
    parts.push(Part::Text("}".to_owned()));
    Ok(parts)
}

/// One tool's parameters in a chat format's markup: the `required` ones, in
/// the schema's order, each written, then the optional ones in order, each
/// of which may be left out — the object rule ([`object_of`]) where `block`
/// spells one parameter, a run that carries its own framing so any two are
/// neighbours apart. [`any_start`] takes over where no parameter is
/// required and the format writes something of its own for a call that
/// writes none (a DSML invoke's empty line).
pub(crate) fn object_parameters(
    args: &[Arg<'_>],
    block: &impl Fn(&Arg<'_>) -> Result<Vec<Part>, Refusal>,
) -> Result<Vec<Part>, Refusal> {
    let mut parts = Vec::new();
    for a in args.iter().filter(|a| a.required) {
        parts.extend(block(a)?);
    }
    for a in args.iter().filter(|a| !a.required) {
        parts.push(Part::Any(vec![Vec::new(), block(a)?]));
    }
    Ok(parts)
}

/// The optional `args` where any of them may be the first written: one
/// alternative per property, each written bare and followed by a fork per
/// later one — `sep` before it, a comma for a JSON member, nothing for a
/// markup block — with `none` first, what is written where every optional
/// is left out (the empty object; a DSML invoke's empty line). llama.cpp's
/// object rule builds this shape for an object no property of which is
/// required. Each property's bare parts are built once and shared
/// ([`Part::Run`]): the alternatives repeat them, and a schema costs its
/// size, not their product.
pub(crate) fn any_start(
    args: &[&Arg<'_>],
    bare: &impl Fn(&Arg<'_>) -> Result<Vec<Part>, Refusal>,
    sep: &str,
    none: Vec<Part>,
) -> Result<Part, Refusal> {
    let members: Vec<Arc<[Part]>> = args
        .iter()
        .map(|a| bare(a).map(Arc::from))
        .collect::<Result<_, Refusal>>()?;
    let mut alts = vec![none];
    for i in 0..members.len() {
        let mut alt = vec![Part::Run(Arc::clone(&members[i]))];
        for member in &members[i + 1..] {
            let later = vec![Part::Text(sep.to_owned()), Part::Run(Arc::clone(member))];
            alt.push(Part::Any(vec![Vec::new(), later]));
        }
        alts.push(alt);
    }
    Ok(Part::Any(alts))
}

/// One argument of a tool: its name and schema, its `$ref`s written out,
/// and whether the tool's `required` names it.
pub(crate) struct Arg<'a> {
    pub key: &'a str,
    pub schema: Cow<'a, Value>,
    pub path: String,
    pub required: bool,
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
    // One budget for every tool's written-out parameters.
    let mut bytes = MAX_BODY;
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
                // A call's markup writes the arguments as one object's
                // properties, which a root anyOf or $ref would not say.
                if let Some(k) = ["anyOf", "$ref"].into_iter().find(|k| p.contains_key(*k)) {
                    return Err(refuse(
                        &here,
                        format!(
                            "the keyword \"{k}\" is not supported by this server at the root of \
                             a tool's parameters, which are one object's properties"
                        ),
                    ));
                }
                if let Some(t) = p.get("type").filter(|t| t.as_str() != Some("object")) {
                    return Err(refuse(
                        &here,
                        format!("the parameters' type must be \"object\", not {t}"),
                    ));
                }
                // The schema compiler reads the whole object, so a keyword the
                // subset does not read is a refusal here and never ignored.
                let mut root = written_out(&Value::Object(p.clone()), &here, &mut bytes)?;
                if let Some(r) = root.as_object_mut() {
                    r.entry("type")
                        .or_insert_with(|| Value::String("object".to_owned()));
                }
                value(&root, Style::JsonInMarkup, &here)?;
                let props = match root.get_mut("properties") {
                    Some(Value::Object(w)) => std::mem::take(w),
                    _ => Map::new(),
                };
                // `written_out` keeps every property, in order.
                let required: Vec<&str> = p
                    .get("required")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .collect();
                p.get("properties")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                    .zip(props)
                    .map(|((key, _), (_, schema))| {
                        plain_name("the parameter name", key, &here)?;
                        Ok(Arg {
                            key,
                            schema: Cow::Owned(schema),
                            path: format!("{here}.properties.{key}"),
                            required: required.contains(&key.as_str()),
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

/// What `response_format` asks the reply to be.
enum Reply<'a> {
    Text,
    Schema(&'a Value),
    /// Any JSON object.
    Object,
}

/// `response_format`'s ask.
fn response_schema(o: &Map<String, Value>) -> Result<Reply<'_>, Refusal> {
    let Some(rf) = o.get("response_format").filter(|v| !v.is_null()) else {
        return Ok(Reply::Text);
    };
    match rf.get("type").and_then(Value::as_str) {
        Some("text") => Ok(Reply::Text),
        Some("json_object") => Ok(Reply::Object),
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .or_else(|| rf.get("schema"));
            match schema {
                Some(s @ Value::Object(_)) => Ok(Reply::Schema(s)),
                _ => Err(refuse(
                    "response_format",
                    "type json_schema needs json_schema.schema, an object",
                )),
            }
        }
        _ => Err(refuse(
            "response_format",
            format!("{rf} is not supported by this server (type text, json_object or json_schema)"),
        )),
    }
}

/// The constraint of a `tool_choice: auto` request with tools to scan for:
/// free text until the model opens a call — the format's own opener, then
/// the program a forced call compiles from there — and free text again past
/// the call's close, each further opener starting the program again.
/// `Ok(None)` where no watch serves the request: a format no watch reads
/// (Muse Spark's recipient slot names the answer's channels (`self`,
/// `commentary`, `user`) as well as a tool, so no opener of its own can
/// start a call's program), and tools whose parameters the schema subset
/// cannot compile — `auto` constrains what it can serve and leaves the
/// rest to the model, where a forced choice (`required`, a named function)
/// refuses the request by name.
fn watch(
    o: &Map<String, Value>,
    ask: &Ask<'_>,
    env: &Env<'_>,
) -> Result<Option<Constraint>, Refusal> {
    let tools = o
        .get("tools")
        .expect("parses_tools held a non-empty tools array");
    let Some((text, parts)) = funs(tools)
        .ok()
        .and_then(|funs| ask.format.lazy_parts(&funs).ok().flatten())
    else {
        return Ok(None);
    };
    let Ok(prog) = program::compile(&parts, env) else {
        return Ok(None);
    };
    // The opener is watched for by the one special token the vocabulary
    // holds for it, where it holds one, as the format's own scan reads it.
    let special = match ask.tok.encode(text)[..] {
        [id] if ask.tok.is_special(id) => Some(id),
        _ => None,
    };
    Ok(Some(Constraint::waiting_for(
        Opener {
            text,
            special,
            prog: Arc::new(prog),
        },
        ask.tok.encode(&format!("{}\n\n", ask.think.close)),
        Arc::clone(ask.vocab),
    )))
}

/// The constraint a request's fields ask for, `None` for a request that
/// asks for none. A tool call and a schema together are refused: a request
/// has one program.
pub(crate) fn request(
    o: &Map<String, Value>,
    ask: &Ask<'_>,
) -> Result<Option<Constraint>, Refusal> {
    if o.get("json_schema").is_some_and(|v| !v.is_null()) {
        return Err(Refusal {
            field: Some("json_schema".to_owned()),
            ..Refusal::invalid(
                "json_schema is not supported by this server: send the schema as response_format \
                 {\"type\": \"json_schema\", \"json_schema\": {\"schema\": ...}}"
                    .to_owned(),
            )
        });
    }
    let reply = response_schema(o)?;
    let forces = tool_choice(o)?;
    let env = Env::new(ask.tok, ask.vocab);
    // The reply is read for a call's markup when the request has tools to
    // scan for, so its strings keep clear of a `<` then.
    let style = if parses_tools(o) {
        Style::JsonInMarkup
    } else {
        Style::Json
    };
    // `tool_choice: auto` (or none given) with tools to scan for: the
    // program starts where the model opens a call itself. A request without
    // tools, a format no watch reads, and a JSON `response_format` (which
    // keeps its own one program below) have none.
    if forces.is_none() && matches!(reply, Reply::Text) && parses_tools(o) {
        return watch(o, ask, &env);
    }
    let call = forces.is_some();
    let (prog, rules) = match (reply, &forces) {
        (Reply::Text, None) => return Ok(None),
        (Reply::Schema(_) | Reply::Object, Some(_)) => {
            return Err(refuse(
                "response_format",
                "a JSON response_format and a forced tool_choice together are not supported by \
                 this server: a request has one program",
            ));
        }
        (Reply::Schema(s), None) => {
            let path = "response_format.json_schema.schema";
            let mut bytes = MAX_BODY;
            let parts = value(&written_out(s, path, &mut bytes)?, style, path)?;
            let prog = program::compile(&parts, &env).map_err(|r| Refusal {
                field: Some(path.to_owned()),
                ..r
            })?;
            (prog, Vec::new())
        }
        (Reply::Object, None) => program::json_object(style, &env).map_err(|r| Refusal {
            field: Some("response_format".to_owned()),
            ..r
        })?,
        (Reply::Text, Some(f)) => (
            program::compile(&tool_parts(o, f, ask.format)?, &env).map_err(|r| Refusal {
                field: Some("tools".to_owned()),
                ..r
            })?,
            Vec::new(),
        ),
    };
    let c = Constraint::new(
        prog,
        rules,
        ask.tok.encode(&format!("{}\n\n", ask.think.close)),
        Arc::clone(ask.vocab),
    );
    Ok(Some(if call { c.following_reasoning() } else { c }))
}
