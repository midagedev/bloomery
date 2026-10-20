//! CPU-only properties of the public template and streaming scanner APIs.

use jinja::{ChatTemplate, Floats};
use serde_json::{Value, json};
use serve::dsml::{DsmlScan, Scanned};
use serve::glmxml::{ArgTypes, KvScan, KvSpelling};
use serve::hermes::HermesScan;
use serve::qwenxml::{ParamKinds, QwenScan};

#[derive(Clone, Copy)]
pub enum Format {
    Glm,
    Qwen,
    Dsml,
    Hermes,
}

enum Scan {
    Glm(KvScan),
    Qwen(QwenScan),
    Dsml(DsmlScan),
    Hermes(HermesScan),
}

impl Scan {
    fn new(format: Format, tools: Option<&Value>) -> Self {
        match format {
            Format::Glm => Self::Glm(KvScan::new(KvSpelling::Glm, ArgTypes::of_tools(tools))),
            Format::Qwen => Self::Qwen(QwenScan::new(ParamKinds::of_tools(tools))),
            Format::Dsml => Self::Dsml(DsmlScan::default()),
            Format::Hermes => Self::Hermes(HermesScan::default()),
        }
    }

    /// Error wording can quote the buffer at the failing push. Only acceptance
    /// is compared for malformed streams; successful messages compare exactly.
    fn step(&mut self, piece: Option<&str>) -> Result<Scanned, ()> {
        match (self, piece) {
            (Self::Glm(s), Some(p)) => s.push(p).map_err(|_| ()),
            (Self::Glm(s), None) => s.finish().map_err(|_| ()),
            (Self::Qwen(s), Some(p)) => s.push(p).map_err(|_| ()),
            (Self::Qwen(s), None) => s.finish().map_err(|_| ()),
            (Self::Hermes(s), Some(p)) => s.push(p).map_err(|_| ()),
            (Self::Hermes(s), None) => s.finish().map_err(|_| ()),
            (Self::Dsml(s), Some(p)) => Ok(s.push(p)),
            (Self::Dsml(s), None) => Ok(s.finish()),
        }
    }
}

fn collect(
    mut scanner: Scan,
    pieces: impl Iterator<Item = impl AsRef<str>>,
) -> Result<Scanned, ()> {
    let mut out = Scanned::default();
    let mut append = |delta: Scanned| {
        out.content.push_str(&delta.content);
        out.calls.extend(delta.calls);
    };
    append(scanner.step(Some(""))?);
    for piece in pieces {
        append(scanner.step(Some(piece.as_ref()))?);
        append(scanner.step(Some(""))?);
    }
    append(scanner.step(None)?);
    for (index, call) in out.calls.iter().enumerate() {
        assert_eq!(call.index, index, "call indices must remain dense");
        let args: Value = serde_json::from_str(&call.arguments).expect("arguments must be JSON");
        assert!(args.is_object(), "arguments must be an object");
    }
    Ok(out)
}

/// Whole, character-sized, and varying byte-sized pieces must decide the same
/// message. All boundaries are valid UTF-8, as the serving decoder guarantees.
pub fn scan(format: Format, text: &str) {
    let tools = json!([{"type":"function", "function": {
        "name":"f", "parameters":{"properties": {
            "s":{"type":"string"}, "n":{"type":"number"},
            "a":{"type":"array"}, "u":{"type":["string","null"]}
        }}
    }}]);
    for schema in [None, Some(&tools)] {
        let whole = collect(Scan::new(format, schema), std::iter::once(text));
        let chars = collect(
            Scan::new(format, schema),
            text.char_indices()
                .map(|(at, c)| &text[at..at + c.len_utf8()]),
        );
        assert_eq!(whole, chars, "character chunks changed the parse");
        let mut at = 0;
        let chunks = std::iter::from_fn(|| {
            if at == text.len() {
                return None;
            }
            let mut end = (at + 1 + usize::from(text.as_bytes()[at] % 31)).min(text.len());
            while !text.is_char_boundary(end) {
                end += 1;
            }
            let piece = &text[at..end];
            at = end;
            Some(piece)
        });
        let varied = collect(Scan::new(format, schema), chunks);
        assert_eq!(whole, varied, "varying chunks changed the parse");
    }
}

/// Bounded template programs keep data fuzzing out of unbounded user-written
/// range loops. Context values, missing fields, types and both float styles vary.
const TEMPLATES: [&str; 6] = [
    "{% for m in messages %}{{ m.role }}:{{ m.content | tojson }}{% endfor %}",
    "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{{ loop.index }}{{ m.content | trim }}{% endfor %}{{ ns.n }}",
    "{% macro emit(v) %}{{ v | tojson(ensure_ascii=False) }}{% endmacro %}{{ emit(value) }}",
    "{{ value | tojson(indent=2) }}|{{ value | length }}|{{ value is mapping }}",
    "{% if add_generation_prompt %}{{ bos_token }}{% endif %}{{ messages[-1].content[0:8] }}{{ eos_token }}",
    "{% for k, v in value.items() %}{{ k }}={{ v | tojson }}{% endfor %}",
];

pub fn render(data: &[u8]) {
    let Some((&choice, context)) = data.split_first() else {
        return;
    };
    let Ok(Value::Object(vars)) = serde_json::from_slice(context) else {
        return;
    };
    let source = TEMPLATES[usize::from(choice) % TEMPLATES.len()];
    let template = ChatTemplate::parse(source).expect("the render programs must parse");
    for floats in [Floats::Python, Floats::Cpp] {
        let first = template.render_with(&vars, floats).map_err(|e| e.0);
        let second = template.render_with(&vars, floats).map_err(|e| e.0);
        assert_eq!(first, second, "rendering leaked state between requests");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_streams() {
        for (format, source) in [
            (Format::Glm, include_str!("../corpus/glmxml/call")),
            (Format::Qwen, include_str!("../corpus/qwenxml/call")),
            (Format::Dsml, include_str!("../corpus/dsml/call")),
            (Format::Hermes, include_str!("../corpus/hermes/call")),
        ] {
            scan(format, source);
            for (at, _) in source.char_indices() {
                scan(format, &source[..at]);
            }
        }
    }

    #[test]
    fn render_programs() {
        let context = br#"{"messages":[{"role":"user","content":"hello"}],"value":{"s":"<x>","n":1.25},"add_generation_prompt":true}"#;
        for choice in 0..TEMPLATES.len() as u8 {
            let mut data = vec![choice];
            data.extend(context);
            render(&data);
        }
    }
}
