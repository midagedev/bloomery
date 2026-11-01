//! Whether a request's schemas are valid: the one owner of every refusal that
//! says a schema contradicts itself or JSON Schema (a [`Refusal`] that is not
//! [`Refusal::outside_subset`]). The compiler reads a schema only after this
//! walk passed it, so it refuses what is outside the subset and nothing else,
//! and a node outside the subset never hides an invalid one: the walk takes
//! every node a schema holds through `properties`, `items`, `anyOf` and the
//! target of a `$ref`, whatever the node's other keywords say, and each tool
//! of a request before any is compiled.
//!
//! What a node is held to, wherever it stands:
//! - a schema is an object or a boolean;
//! - `type` is a type name or a non-empty list of them;
//! - `maxLength` is a non-negative integer, `anyOf` and `enum` are non-empty
//!   arrays, `properties` is an object and `required` an array (a null
//!   `properties` or `required` reads as absent, as some tool-schema
//!   generators write an empty one);
//! - `required` holds strings, and on a `type: object` node names only
//!   properties the node holds, unless a keyword beside it may supply
//!   properties too (`additionalProperties` absent or other than `false`, or
//!   another of [`SUPPLIES_PROPERTIES`]; the compiler serves no such node);
//! - a `const` or `enum` value the compiler can write (a string, an integer
//!   or a boolean) is of the kind `type` names;
//! - `$ref` is a string, and one that starts `#/` points at something.
//!
//! A tool's `parameters` is an object whose `type` is `object` (or a list of
//! types, which the compiler refuses as outside the subset) and is read as an
//! `object` when it names none.

use std::collections::HashSet;

use serde_json::{Map, Value};

use super::{Refusal, Scalar, refuse, type_of};

/// The names JSON Schema gives a value's type.
const TYPES: [&str; 7] = [
    "object", "array", "string", "integer", "number", "boolean", "null",
];

/// Whether `v` is a list of type names, which JSON Schema takes for `type`
/// and the compiler does not read.
pub(super) fn type_list(v: &Value) -> bool {
    v.as_array().is_some_and(|a| {
        !a.is_empty()
            && a.iter()
                .all(|t| t.as_str().is_some_and(|t| TYPES.contains(&t)))
    })
}

/// A `maxLength` as the length it bounds, `None` for a value that is not a
/// non-negative integer.
pub(super) fn max_length(v: &Value) -> Option<usize> {
    v.as_u64().and_then(|n| usize::try_from(n).ok())
}

/// The schema a `$ref` points to in `root` (`#` or a JSON pointer after it).
pub(super) fn target<'v>(root: &'v Value, r: &str) -> Option<&'v Value> {
    r.strip_prefix('#').and_then(|p| root.pointer(p))
}

/// The message of a `$ref` that points to nothing here: dangling when it is
/// a JSON pointer (invalid), an anchor or another document's URL otherwise
/// (outside the subset).
pub(super) fn no_target(r: &str) -> String {
    format!("$ref {r:?} does not point into this schema (\"#\" or a JSON pointer after it)")
}

/// The message of a tool's parameters whose `type` is not `object`.
pub(super) fn parameters_type(t: &Value) -> String {
    format!("the parameters' type must be \"object\", not {t}")
}

/// A tool the request names, its parameters held valid.
pub(super) struct Tool<'a> {
    pub name: &'a str,
    /// `tools[i]`, the path the tool's own refusals lead with.
    pub path: String,
    /// `tools[i].name.parameters`, the path of its schema's root.
    pub here: String,
    pub parameters: Option<&'a Map<String, Value>>,
}

/// The tools of a request, every one's parameters walked: the first invalid
/// schema of any tool is refused by name before any tool is compiled.
pub(super) fn tools(tools: &Value) -> Result<Vec<Tool<'_>>, Refusal> {
    let Some(list) = tools.as_array().filter(|t| !t.is_empty()) else {
        return Err(refuse("tools", "tool_choice needs a non-empty tools array"));
    };
    list.iter()
        .enumerate()
        .map(|(i, tool)| {
            let path = format!("tools[{i}]");
            let f = tool.get("function").unwrap_or(tool);
            let name = f.get("name").and_then(Value::as_str).unwrap_or_default();
            let here = format!("{path}.{name}.parameters");
            let parameters = match f.get("parameters") {
                None | Some(Value::Null) => None,
                Some(v @ Value::Object(p)) => {
                    if let Some(t) = p
                        .get("type")
                        .filter(|t| t.as_str() != Some("object") && !type_list(t))
                    {
                        return Err(refuse(&here, parameters_type(t)));
                    }
                    schema(v, Some("object"), &here)?;
                    Some(p)
                }
                Some(v) => {
                    return Err(refuse(
                        &here,
                        format!("parameters must be an object, not {}", type_of(v)),
                    ));
                }
            };
            Ok(Tool {
                name,
                path,
                here,
                parameters,
            })
        })
        .collect()
}

/// How a node's child extends the node's path.
enum Step<'a> {
    Property(&'a str),
    Items,
    AnyOf(usize),
    /// A `$ref`'s target stands where the `$ref` does.
    Ref,
}

impl Step<'_> {
    fn extend(&self, path: &mut String) {
        match self {
            Step::Property(key) => {
                path.push_str(".properties.");
                path.push_str(key);
            }
            Step::Items => path.push_str(".items"),
            Step::AnyOf(i) => path.push_str(&format!(".anyOf[{i}]")),
            Step::Ref => {}
        }
    }
}

/// The nodes still to walk below one node.
struct Frame<'a> {
    children: std::vec::IntoIter<(Step<'a>, &'a Value)>,
    /// The length of the node's own path.
    path_len: usize,
}

/// Walks the schema `root` that stands at `path`, node by node, refusing the
/// first one that is invalid. `root_type` is the type the root has when it
/// names none (a tool's parameters are an object). Each node is read once,
/// however many `$ref`s point at it, and the walk keeps its own stack, so a
/// chain of `$ref`s or a recursive one costs no more than the nodes it holds.
pub(super) fn schema(root: &Value, root_type: Option<&str>, path: &str) -> Result<(), Refusal> {
    let mut path = path.to_owned();
    let mut seen: HashSet<*const Value> = HashSet::from([std::ptr::from_ref(root)]);
    let mut stack = vec![Frame {
        children: node(root, root_type, &path, root)?.into_iter(),
        path_len: path.len(),
    }];
    while let Some(top) = stack.last_mut() {
        path.truncate(top.path_len);
        let Some((step, child)) = top.children.next() else {
            stack.pop();
            continue;
        };
        step.extend(&mut path);
        if !seen.insert(std::ptr::from_ref(child)) {
            continue;
        }
        stack.push(Frame {
            children: node(child, None, &path, root)?.into_iter(),
            path_len: path.len(),
        });
    }
    Ok(())
}

/// One node held to what it must be (the list above), and the nodes below it.
/// The keywords beside `properties` through which a JSON Schema object may
/// hold a property its `required` names.
const SUPPLIES_PROPERTIES: [&str; 11] = [
    "allOf",
    "anyOf",
    "oneOf",
    "$ref",
    "additionalProperties",
    "patternProperties",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
];

fn node<'a>(
    node: &'a Value,
    root_type: Option<&str>,
    path: &str,
    root: &'a Value,
) -> Result<Vec<(Step<'a>, &'a Value)>, Refusal> {
    let o = match node {
        Value::Object(o) => o,
        // `true` and `false` are schemas, which the compiler does not read.
        Value::Bool(_) => return Ok(Vec::new()),
        other => {
            return Err(refuse(
                path,
                format!("a schema must be an object, not {}", type_of(other)),
            ));
        }
    };
    let mut children = Vec::new();
    if let Some(r) = o.get("$ref") {
        let Some(r) = r.as_str() else {
            return Err(refuse(path, format!("$ref must be a string, not {r}")));
        };
        match target(root, r) {
            Some(t) => children.push((Step::Ref, t)),
            None if r.starts_with("#/") => return Err(refuse(path, no_target(r))),
            None => {}
        }
    }
    let ty = match o.get("type") {
        None => root_type,
        Some(Value::String(t)) if TYPES.contains(&t.as_str()) => Some(t.as_str()),
        Some(Value::String(t)) => {
            return Err(refuse(
                path,
                format!(
                    "type \"{t}\" is not a JSON Schema type ({})",
                    TYPES.join(", ")
                ),
            ));
        }
        Some(v) if type_list(v) => None,
        Some(v) => {
            return Err(refuse(
                path,
                format!("type must be a type name or a list of them, not {v}"),
            ));
        }
    };
    if let Some(v) = o.get("maxLength")
        && max_length(v).is_none()
    {
        return Err(refuse(
            path,
            format!("maxLength must be a non-negative integer, not {v}"),
        ));
    }
    if let Some(c) = o.get("const") {
        fits(ty, c, path)?;
    }
    if let Some(e) = o.get("enum") {
        let Some(values) = e.as_array().filter(|a| !a.is_empty()) else {
            return Err(refuse(
                &format!("{path}.enum"),
                format!("an enum must be a non-empty array, not {e}"),
            ));
        };
        for v in values {
            fits(ty, v, path)?;
        }
    }
    if let Some(a) = o.get("anyOf") {
        let Some(alts) = a.as_array().filter(|a| !a.is_empty()) else {
            return Err(refuse(
                &format!("{path}.anyOf"),
                format!("anyOf must be a non-empty array, not {a}"),
            ));
        };
        children.extend(alts.iter().enumerate().map(|(i, s)| (Step::AnyOf(i), s)));
    }
    let props = match o.get("properties") {
        None | Some(Value::Null) => None,
        Some(Value::Object(p)) => Some(p),
        Some(v) => {
            return Err(refuse(
                path,
                format!("properties must be an object, not {}", type_of(v)),
            ));
        }
    };
    if let Some(r) = o.get("required").filter(|r| !r.is_null()) {
        let Some(names) = r.as_array() else {
            return Err(refuse(
                path,
                format!("required must be an array, not {}", type_of(r)),
            ));
        };
        // An absent `additionalProperties` allows any property; `false`
        // supplies none, and any other value of a supplying keyword may.
        let supplied = o.get("additionalProperties").is_none()
            || SUPPLIES_PROPERTIES.iter().any(|k| {
                o.get(*k)
                    .is_some_and(|v| *k != "additionalProperties" || *v != Value::Bool(false))
            });
        for n in names {
            let Some(name) = n.as_str() else {
                return Err(refuse(
                    path,
                    format!("required must hold property names, not {n}"),
                ));
            };
            if ty == Some("object") && !supplied && !props.is_some_and(|p| p.contains_key(name)) {
                return Err(refuse(
                    path,
                    format!("required names {n}, which properties does not hold"),
                ));
            }
        }
    }
    children.extend(
        props
            .into_iter()
            .flatten()
            .map(|(key, s)| (Step::Property(key), s)),
    );
    children.extend(o.get("items").map(|s| (Step::Items, s)));
    Ok(children)
}

/// A `const` or `enum` value against the node's `type`: a string, an integer
/// or a boolean is of one kind, and `number` takes an integer. A value of
/// another JSON type is the compiler's to refuse as outside the subset.
fn fits(ty: Option<&str>, value: &Value, path: &str) -> Result<(), Refusal> {
    let (Some(t), Some(kind)) = (ty, Scalar::read(value).map(|s| s.kind())) else {
        return Ok(());
    };
    if t == kind || (t == "number" && kind == "integer") {
        return Ok(());
    }
    Err(refuse(
        path,
        format!("type \"{t}\" does not match its {kind} values"),
    ))
}
