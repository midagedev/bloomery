//! A request to token ids and spans, as Clef's `encode_record` builds them.
//!
//! Every piece of text is tokenized on its own and the ids are concatenated, as the release's
//! `_tokens` does (`add_special_tokens=False`; special tokens written in the text, such as
//! `<|im_start|>`, become their ids). The strings below are Clef's prompt, copied from the release.
//!
//! A request's images join the prompt as one media piece between `STATE:\n` and the state: the
//! model's image placeholder ([`MediaModel::image_placeholder`], Clef's
//! `<|vision_start|><|image_pad|><|vision_end|>`) once per image and a newline, tokenized in one
//! call. Each image's pad id is expanded to the image's token count ([`encode_media`]; the
//! expansion is `vision::media::expand_spans`, the server's), and those ids count in what the
//! schema leaves for the state, as the release's `fixed_length` counts them. [`encode`] is the
//! entry of a server with no image input: it refuses a request that carries one, by name.

use tokenizer::Tokenizer;
use vision::MediaModel;
use vision::media::{ExpandError, expand_spans};

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
    /// Each image's run of `<|image_pad|>` ids, half-open, in the order of the request's images.
    pub image_spans: Vec<(usize, usize)>,
}

/// What a server with no image input answers to a request that carries one: llama.cpp's message.
pub const NO_IMAGE_INPUT: &str = "image input is not supported - hint: if this is unexpected, you may need to provide the mmproj";

/// `req` through `tok`, at most `max_length` ids: the state is cut to what the schema leaves. A
/// request that carries images is refused ([`Error::NotSupported`]): [`encode_media`] takes it.
pub fn encode(tok: &Tokenizer, req: &Request, max_length: usize) -> Result<Encoded, Error> {
    encode_with(&mut |text| tok.encode(text, false, true), req, max_length)
}

/// `req` through `tok` with its images as `media` takes them and their token counts
/// (`image_tokens[i]`: the image's `nx·ny`, the positions its pad id is expanded to), at most
/// `max_length` ids.
pub fn encode_media(
    tok: &Tokenizer,
    req: &Request,
    media: &dyn MediaModel,
    image_tokens: &[usize],
    max_length: usize,
) -> Result<Encoded, Error> {
    encode_media_with(
        &mut |text| tok.encode(text, false, true),
        req,
        media,
        image_tokens,
        max_length,
    )
}

/// [`encode`] over any tokenizer function.
pub(crate) fn encode_with(
    tokens: &mut dyn FnMut(&str) -> Vec<u32>,
    req: &Request,
    max_length: usize,
) -> Result<Encoded, Error> {
    if !req.images.is_empty() {
        return Err(Error::NotSupported(NO_IMAGE_INPUT.into()));
    }
    encode_inner(tokens, req, None, max_length)
}

/// [`encode_media`] over any tokenizer function.
pub(crate) fn encode_media_with(
    tokens: &mut dyn FnMut(&str) -> Vec<u32>,
    req: &Request,
    media: &dyn MediaModel,
    image_tokens: &[usize],
    max_length: usize,
) -> Result<Encoded, Error> {
    if image_tokens.len() != req.images.len() {
        return Err(Error::Images(format!(
            "{} token count(s) for {} image(s)",
            image_tokens.len(),
            req.images.len()
        )));
    }
    let images = Images {
        placeholder: media.image_placeholder(),
        token: media.image_token(),
        counts: image_tokens,
    };
    encode_inner(tokens, req, Some(images), max_length)
}

/// The image input of one encode: a model's placeholder text and pad id, and each image's count.
struct Images<'a> {
    placeholder: &'a str,
    token: u32,
    counts: &'a [usize],
}

fn encode_inner(
    tokens: &mut dyn FnMut(&str) -> Vec<u32>,
    req: &Request,
    images: Option<Images<'_>>,
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
    let media = match &images {
        Some(i) => media_piece(tokens, i)?,
        None => Vec::new(),
    };
    let suffix = tokens(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    );
    let state = tokens(&render(&req.state));
    // The media piece holds one placeholder per image: its expansion adds `count - 1` ids each.
    let counts = images.as_ref().map_or(&[][..], |i| i.counts);
    let media_len = media.len() + counts.iter().sum::<usize>() - counts.len();
    if media_len > max_length {
        return Err(Error::ImagesTooLong {
            images: counts.len(),
            media: media_len,
            max: max_length,
        });
    }
    let head = prefix.len() + media_len;
    let fixed = head + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(Error::SchemaTooLong {
            fixed,
            max: max_length,
        });
    }
    let state_len = state.len();
    let kept = state_len.min(max_length - fixed);
    let mut ids = prefix;
    ids.extend(media);
    ids.extend(state);
    ids.extend(schema);
    ids.extend(suffix);
    // The placeholders are counted over the whole text, the state before it is cut, so one typed
    // into the state, an instruction or an option is refused whatever the cut leaves of it.
    let mut image_spans = Vec::with_capacity(counts.len());
    if let Some(i) = &images {
        let expanded = expand_spans(&ids, i.token, i.counts).map_err(|e| match e {
            ExpandError::PlaceholderCount {
                placeholders,
                images,
            } => Error::PlaceholderInText {
                placeholders,
                images,
            },
            ExpandError::EmptySpan(i) => Error::Images(format!("image {i} has no tokens")),
        })?;
        ids = expanded.ids;
        image_spans.extend(expanded.spans.iter().map(|r| (r.start, r.end)));
    }
    ids.drain(head + kept..head + state_len);
    let offset = head + kept;
    for q in &mut questions {
        q.question_span = (q.question_span.0 + offset, q.question_span.1 + offset);
        for s in &mut q.option_spans {
            *s = (s.0 + offset, s.1 + offset);
        }
    }
    Ok(Encoded {
        ids,
        questions,
        image_spans,
    })
}

/// The media piece: the model's placeholder once per image and a newline, tokenized in one call.
/// Its ids hold one pad id per image, or the tokenizer is not the model's.
fn media_piece(
    tokens: &mut dyn FnMut(&str) -> Vec<u32>,
    images: &Images<'_>,
) -> Result<Vec<u32>, Error> {
    let n = images.counts.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let piece = tokens(&format!("{}\n", images.placeholder.repeat(n)));
    let pads = piece.iter().filter(|&&t| t == images.token).count();
    if pads != n {
        return Err(Error::Images(format!(
            "the tokenizer reads {n} image placeholder(s) as {pads} pad id(s) ({})",
            images.token
        )));
    }
    Ok(piece)
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

    const START: u32 = 1000;
    const PAD: u32 = 1001;
    const END: u32 = 1002;
    const PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

    /// A model whose placeholder is [`PLACEHOLDER`] and whose pad id is [`PAD`], as [`with_media`] reads it.
    struct Fake;

    impl MediaModel for Fake {
        fn image_placeholder(&self) -> &str {
            PLACEHOLDER
        }

        fn image_token(&self) -> u32 {
            PAD
        }

        fn prepare(&self, _: &vision::Rgb8) -> Result<vision::Prepared, vision::VisionError> {
            Err(vision::VisionError::Limits {
                detail: "a test model prepares nothing".into(),
            })
        }
    }

    fn encoded(r: &Request, counts: &[usize], max: usize) -> Result<Encoded, Error> {
        encode_media_with(&mut with_media, r, &Fake, counts, max)
    }

    /// One id per byte, and the three image specials as Clef's ids.
    fn with_media(text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some(c) = rest.chars().next() {
            let special = [
                ("<|vision_start|>", START),
                ("<|image_pad|>", PAD),
                ("<|vision_end|>", END),
            ]
            .into_iter()
            .find_map(|(t, id)| rest.strip_prefix(t).map(|r| (id, r)));
            match special {
                Some((id, r)) => {
                    out.push(id);
                    rest = r;
                }
                None => {
                    out.extend(c.to_string().bytes().map(u32::from));
                    rest = &rest[c.len_utf8()..];
                }
            }
        }
        out
    }

    fn with_images(state: &str, images: usize) -> Request {
        let urls = vec![r#""data:image/png;base64,AA==""#; images].join(",");
        request(&format!(
            r#"{{"state":{state:?},"images":[{urls}],"questions":{{"q":{{"type":"noul","instructions":"Q?"}}}}}}"#
        ))
    }

    /// The prompt's start through `STATE:\n`.
    fn start() -> usize {
        with_media(&format!(
            "<|im_start|>system\n{CLEF_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))
        .len()
    }

    /// The media piece sits between `STATE:\n` and the state: one delimited span a image, each
    /// expanded to its count, and a newline after the last. The ids of the same request without
    /// images are the rest, and the question spans move by the piece's length. (A missing newline fails
    /// here and on set D's rows.)
    #[test]
    fn images_join_the_prompt_before_the_state() {
        let counts = [3, 2];
        let r = with_images("S", 2);
        let e = encoded(&r, &counts, MAX_LENGTH).unwrap();
        let plain = encoded(
            &request(r#"{"state":"S","questions":{"q":{"type":"noul","instructions":"Q?"}}}"#),
            &[],
            MAX_LENGTH,
        )
        .unwrap();
        let at = start();
        let media = [START, PAD, PAD, PAD, END, START, PAD, PAD, END, 10];
        assert_eq!(e.ids[..at], plain.ids[..at]);
        assert_eq!(e.ids[at..at + media.len()], media);
        assert_eq!(e.ids[at + media.len()..], plain.ids[at..]);
        assert_eq!(e.image_spans, [(at + 1, at + 4), (at + 6, at + 8)]);
        let (q, p) = (&e.questions[0], &plain.questions[0]);
        let shift = |s: (usize, usize)| (s.0 + media.len(), s.1 + media.len());
        assert_eq!(q.question_span, shift(p.question_span));
        assert_eq!(text(&e.ids, q.question_span), "Q?");
        assert_eq!(
            q.option_spans,
            p.option_spans.iter().map(|&s| shift(s)).collect::<Vec<_>>()
        );
    }

    /// A request with no image gets no media piece, not even its newline, and the text entry and the
    /// media entry agree on it.
    #[test]
    fn no_image_is_no_media_piece() {
        let r = request(r#"{"state":"S","questions":{"q":{"type":"noul"}}}"#);
        let text_only = encode_with(&mut with_media, &r, MAX_LENGTH).unwrap();
        let media = encoded(&r, &[], MAX_LENGTH).unwrap();
        assert_eq!(text_only, media);
        assert!(text_only.image_spans.is_empty());
        assert!(!text_only.ids.contains(&START));
        assert_eq!(text_only.ids[start()..start() + 1], [u32::from(b'S')]);
    }

    /// The text entry is a server with no image input: it refuses a request with images, by name.
    #[test]
    fn the_text_entry_refuses_images() {
        match encode_with(&mut with_media, &with_images("S", 1), MAX_LENGTH) {
            Err(Error::NotSupported(m)) => assert_eq!(m, NO_IMAGE_INPUT),
            other => panic!("{other:?}"),
        }
    }

    /// The media ids count in what the schema leaves for the state (the release's `fixed_length`):
    /// the state is cut after them, the schema that does not fit is `SchemaTooLong` with them in its
    /// `fixed`, and images that alone pass the length are `ImagesTooLong`.
    #[test]
    fn the_media_ids_count_in_what_the_schema_leaves() {
        let r = with_images("0123456789", 1);
        let media = 1 + 5 + 1 + 1;
        let full = encoded(&r, &[5], MAX_LENGTH).unwrap();
        let fixed = full.ids.len() - 10;
        assert_eq!(full.image_spans, [(start() + 1, start() + 6)]);
        let cut = encoded(&r, &[5], fixed + 4).unwrap();
        assert_eq!(cut.ids.len(), fixed + 4);
        assert_eq!(text(&cut.ids, cut.questions[0].question_span), "Q?");
        let shift = |s: (usize, usize)| (s.0 - 6, s.1 - 6);
        assert_eq!(
            cut.questions[0].question_span,
            shift(full.questions[0].question_span)
        );
        assert_eq!(
            cut.image_spans, full.image_spans,
            "the cut is after the images"
        );
        match encoded(&r, &[5], fixed - 1) {
            Err(Error::SchemaTooLong { fixed: f, max }) => assert_eq!((f, max), (fixed, fixed - 1)),
            other => panic!("{other:?}"),
        }
        match encoded(&r, &[5], media - 1) {
            Err(Error::ImagesTooLong {
                images,
                media: m,
                max,
            }) => {
                assert_eq!((images, m, max), (1, media, media - 1));
            }
            other => panic!("{other:?}"),
        }
    }

    /// An image placeholder typed into the state, an instruction or an option is refused by name,
    /// whatever the cut leaves of it; with no image the same text is as before.
    #[test]
    fn a_placeholder_in_the_text_is_refused() {
        let pad = "<|image_pad|>";
        let in_state = with_images(&format!("0123{pad}"), 1);
        let full = encoded(&with_images("0123", 1), &[2], MAX_LENGTH).unwrap();
        let fixed = full.ids.len() - 4;
        for max in [MAX_LENGTH, fixed + 2, fixed] {
            match encoded(&in_state, &[2], max) {
                Err(Error::PlaceholderInText {
                    placeholders,
                    images,
                }) => assert_eq!((placeholders, images), (2, 1), "max {max}"),
                other => panic!("max {max}: {other:?}"),
            }
        }
        let body = |instructions: &str, criteria: &str| {
            let t = format!(
                r#"{{"state":"s","images":["data:image/png;base64,AA=="],"questions":{{"q":{{"type":"{criteria}","instructions":{instructions:?}}}}}}}"#
            );
            request(&t)
        };
        let in_instruction = body(&format!("what {pad}?"), "noul");
        assert!(matches!(
            encoded(&in_instruction, &[2], MAX_LENGTH),
            Err(Error::PlaceholderInText {
                placeholders: 2,
                images: 1
            })
        ));
        let in_option = request(&format!(
            r#"{{"state":"s","images":["data:image/png;base64,AA=="],"questions":{{"q":{{"type":"choice","criteria":{{"a":"{pad}"}}}}}}}}"#
        ));
        assert!(matches!(
            encoded(&in_option, &[2], MAX_LENGTH),
            Err(Error::PlaceholderInText { .. })
        ));
        let no_image = request(&format!(
            r#"{{"state":"{pad}","questions":{{"q":{{"type":"noul"}}}}}}"#
        ));
        let e = encode_with(&mut with_media, &no_image, MAX_LENGTH).unwrap();
        assert_eq!(e.ids.iter().filter(|&&t| t == PAD).count(), 1);
    }

    /// Counts that do not pair with the images, an image with none, and a tokenizer that does not
    /// read the placeholder are the engine's, each named.
    #[test]
    fn counts_and_tokenizer_are_checked() {
        let r = with_images("S", 2);
        for counts in [&[3][..], &[3, 2, 1][..], &[3, 0][..]] {
            assert!(
                matches!(encoded(&r, counts, MAX_LENGTH), Err(Error::Images(_))),
                "{counts:?}"
            );
        }
        assert!(matches!(
            encode_media_with(&mut bytes, &r, &Fake, &[3, 2], MAX_LENGTH),
            Err(Error::Images(m)) if m.contains("as 0 pad id(s)")
        ));
    }
}
