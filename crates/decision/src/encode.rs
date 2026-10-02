//! A request to token ids and spans, as Clef's `encode_record` builds them (text only).
//!
//! Every piece of text is tokenized on its own and the ids are concatenated, as the release's
//! `_tokens` does (`add_special_tokens=False`; special tokens written in the text, such as
//! `<|im_start|>`, become their ids). The strings below are Clef's prompt, copied from the release.

use tokenizer::Tokenizer;

use crate::Error;
use crate::render::render;
use crate::request::{Kind, Request};

/// Clef's system prompt.
pub const CLEF_SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. Each answer \
must be exactly one of that field's allowed options.";

/// The release's default `max_length`.
pub const MAX_LENGTH: usize = 16384;

/// One question's place in the ids: half-open spans over the whole sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedQuestion {
    pub id: String,
    pub kind: Kind,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    /// The option ids in span order (`choice`: sorted by key).
    pub option_ids: Vec<String>,
}

/// A request's model input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoded {
    pub ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
}

/// `req` through `tok`, at most `max_length` ids: the state is cut to what the schema leaves.
pub fn encode(tok: &Tokenizer, req: &Request, max_length: usize) -> Result<Encoded, Error> {
    encode_with(&mut |text| tok.encode(text, false, true), req, max_length)
}

/// [`encode`] over any tokenizer function.
pub(crate) fn encode_with(
    tokens: &mut dyn FnMut(&str) -> Vec<u32>,
    req: &Request,
    max_length: usize,
) -> Result<Encoded, Error> {
    let mut schema = tokens("\n\nSCHEMA FIELDS:\n");
    let mut questions = Vec::with_capacity(req.questions.len());
    for (qi, q) in req.questions.iter().enumerate() {
        schema.extend(tokens(&format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            qi + 1,
            q.id,
            q.kind.word()
        )));
        let q_start = schema.len();
        let instructions = match &q.instructions {
            Some(v) if v.as_str() != Some("") => render(v),
            _ => q.id.clone(),
        };
        schema.extend(tokens(&instructions));
        let q_end = schema.len();
        if q_end == q_start {
            return Err(Error::EmptySpan {
                question: q.id.clone(),
                what: "instructions",
            });
        }
        schema.extend(tokens("\nALLOWED OPTIONS:\n"));
        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (oi, opt) in q.options().into_iter().enumerate() {
            schema.extend(tokens(&format!("OPTION {}: ", oi + 1)));
            let start = schema.len();
            let mut semantics = vec![(
                "option_id".to_string(),
                crate::json::Json::Str(opt.id.clone()),
            )];
            if let Some(d) = opt.description {
                semantics.push(("description".to_string(), d));
            }
            schema.extend(tokens(&render(&crate::json::Json::Object(semantics))));
            option_spans.push((start, schema.len()));
            option_ids.push(opt.id);
            schema.extend(tokens("\n"));
        }
        schema.extend(tokens("END FIELD\n"));
        questions.push(EncodedQuestion {
            id: q.id.clone(),
            kind: q.kind,
            question_span: (q_start, q_end),
            option_spans,
            option_ids,
        });
    }
    let prefix = tokens(&format!(
        "<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ));
    let suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    );
    let mut state = tokens(&render(&req.state));
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(Error::SchemaTooLong {
            fixed,
            max: max_length,
        });
    }
    state.truncate(max_length - fixed);
    let offset = prefix.len() + state.len();
    for q in &mut questions {
        q.question_span = (q.question_span.0 + offset, q.question_span.1 + offset);
        for s in &mut q.option_spans {
            *s = (s.0 + offset, s.1 + offset);
        }
    }
    let mut ids = prefix;
    ids.extend(state);
    ids.extend(schema);
    ids.extend(suffix);
    Ok(Encoded { ids, questions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    /// One id per byte: spans can be read back as text.
    fn bytes(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn text(ids: &[u32], span: (usize, usize)) -> String {
        ids[span.0..span.1]
            .iter()
            .map(|&b| char::from(u8::try_from(b).unwrap()))
            .collect()
    }

    fn request(body: &str) -> Request {
        Request::from_json(&parse(body).unwrap()).unwrap()
    }

    #[test]
    fn spans_cover_the_rendered_pieces() {
        let r = request(
            r#"{"model":"m","state":{"b":1,"a":2.5},"questions":{
            "dept":{"type":"choice","instructions":"Which?","criteria":{"tech":"Bugs","billing":null}},
            "up":{"type":"noul","instructions":""}}}"#,
        );
        let e = encode_with(&mut bytes, &r, MAX_LENGTH).unwrap();
        let q = &e.questions[0];
        assert_eq!(text(&e.ids, q.question_span), "Which?");
        assert_eq!(q.option_ids, ["billing", "tech"]);
        assert_eq!(
            text(&e.ids, q.option_spans[0]),
            r#"{"option_id":"billing"}"#
        );
        assert_eq!(
            text(&e.ids, q.option_spans[1]),
            r#"{"description":"Bugs","option_id":"tech"}"#
        );
        let q = &e.questions[1];
        assert_eq!(text(&e.ids, q.question_span), "up");
        assert_eq!(q.kind, Kind::Noul);
        let whole: String = e
            .ids
            .iter()
            .map(|&b| char::from(u8::try_from(b).unwrap()))
            .collect();
        assert!(whole.starts_with("<|im_start|>system\nRead the complete state"));
        assert!(whole.contains("STATE:\n{\"a\":2.5,\"b\":1}\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: dept\nTYPE: choice\nINSTRUCTION: Which?\nALLOWED OPTIONS:\nOPTION 1: "));
        assert!(whole.contains("END FIELD\n\nFIELD 2\nID: up\nTYPE: noul\nINSTRUCTION: up\n"));
        assert!(whole.ends_with("END FIELD\n\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"));
    }

    #[test]
    fn the_state_is_cut_to_what_the_schema_leaves() {
        let r = request(r#"{"model":"m","state":"0123456789","questions":{"q":{"type":"noul"}}}"#);
        let full = encode_with(&mut bytes, &r, MAX_LENGTH).unwrap();
        let fixed = full.ids.len() - 10;
        let cut = encode_with(&mut bytes, &r, fixed + 4).unwrap();
        assert_eq!(cut.ids.len(), fixed + 4);
        let shift = |s: (usize, usize)| (s.0 - 6, s.1 - 6);
        assert_eq!(
            cut.questions[0].question_span,
            shift(full.questions[0].question_span)
        );
        assert_eq!(text(&cut.ids, cut.questions[0].question_span), "q");
        assert!(encode_with(&mut bytes, &r, fixed).is_ok());
        match encode_with(&mut bytes, &r, fixed - 1) {
            Err(Error::SchemaTooLong { fixed: f, max }) => assert_eq!((f, max), (fixed, fixed - 1)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn an_empty_question_span_is_refused() {
        let r = request(r#"{"model":"m","state":"s","questions":{"":{"type":"noul"}}}"#);
        assert!(matches!(
            encode_with(&mut bytes, &r, MAX_LENGTH),
            Err(Error::EmptySpan {
                what: "instructions",
                ..
            })
        ));
    }

    #[test]
    fn non_string_instructions_render_as_json() {
        let r = request(
            r#"{"model":"m","state":"s","questions":{"q":{"type":"noul","instructions":{"k":[1,2]}}}}"#,
        );
        let e = encode_with(&mut bytes, &r, MAX_LENGTH).unwrap();
        assert_eq!(text(&e.ids, e.questions[0].question_span), r#"{"k":[1,2]}"#);
    }
}
