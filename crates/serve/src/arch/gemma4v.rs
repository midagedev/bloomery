//! The embedding input contract with the Gemma4 image model.

use super::*;
use crate::mock::MockEmbed;

#[test]
fn image_parts_preserve_delimiters_and_place_spans_in_one_input() {
    let mut side = Side::new(Box::new(MockEmbed::new(128)));
    let rule =
        vision::size::SizeRule::new(16, 3, vision::arch::TokenLimits::new(1, 4).unwrap()).unwrap();
    side.media = Some(Arc::new(vision::arch::gemma4v::Media::new(rule)));
    let image = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg=="}});
    let body = json!({"content":[{"type":"text","text":"a<|image>"},image.clone(),{"type":"text","text":"b"},image]});
    let parsed = inputs(&side, "input", &json!([body])).unwrap_or_else(|e| panic!("{}", e.message));
    assert_eq!(parsed.len(), 1);
    let got = &parsed[0];
    assert_eq!(got.images.len(), 2);
    let mut want = vec![side.tok.bos()];
    want.extend(side.tok.encode("a<|image>"));
    for (i, after) in [(0, "b"), (1, "")] {
        want.extend(side.tok.encode("<|image>"));
        assert_eq!(got.images[i].at, want.len());
        assert_eq!(got.images[i].prepared.span_len, 1);
        want.extend([side.tok.bos(); 1]);
        want.extend(side.tok.encode("<image|>"));
        want.extend(side.tok.encode(after));
    }
    if side.tok.add_eos() {
        want.push(side.tok.eos());
    }
    assert_eq!(got.ids, want);
    let model = side.media.as_ref().expect("image preparer");
    assert_eq!(model.image_token(), None);
    assert!(matches!(
        media::expand_prompt(&[side.tok.bos()], &[], &**model),
        Err(media::MediaError::NoChatImageToken)
    ));
    side.ctx = got.ids.len();
    assert!(inputs(&side, "input", &json!([body.clone()])).is_ok());
    side.ctx = got.images[1].at;
    let mut over = body.clone();
    let parts = over["content"].as_array_mut().unwrap();
    parts.extend(std::iter::repeat_n(parts[1].clone(), 64));
    parts.push(json!({"type":"image_url","image_url":"https://example.com/not-prepared.png"}));
    let error = inputs(&side, "input", &json!([over]))
        .err()
        .expect("context refusal");
    assert_eq!(error.message, "image input exceeds the embedding context");
    side.media = None;
    assert!(inputs(&side, "input", &json!([body])).is_err());
    assert!(
        inputs(
            &side,
            "input",
            &json!({"content":[{"type":"image_url","image_url":"https://example.com/x.png"}]})
        )
        .is_err()
    );
}
