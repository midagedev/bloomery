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
    let body = json!({"content":[{"type":"text","text":"a<|image>"},image.clone(),{"type":"text","text":"b"},image.clone()]});
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
    side.ctx = 8192;
    let second = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGPgEpHTMLIBAAI3ANNbVlHYAAAAAElFTkSuQmCC"}});
    let video = json!({"content":[{"type":"text","text":"Video:"},image.clone(),second.clone()]});
    for capacity in [4, 70, 71, 72, 73, 140] {
        let rule = vision::size::SizeRule::new(
            16,
            3,
            vision::arch::TokenLimits::new(1, capacity).unwrap(),
        )
        .unwrap();
        side.media = Some(Arc::new(vision::arch::gemma4v::Media::new(rule)));
        let result = inputs(&side, "input", &video);
        if capacity < 72 {
            let error = result.err().expect("video exceeds loaded tower capacity");
            assert_eq!(error.code, 400);
            assert!(error.message.contains("loaded projector token capacity"));
            let broken =
                json!({"type":"image_url","image_url":"data:image/jpeg;base64,/9j/2wAEAAA="});
            let error = inputs(
                &side,
                "input",
                &json!({"content":[{"type":"text","text":"Video:"},image.clone(),broken]}),
            )
            .err()
            .expect("prepare the first frame before decoding the next");
            assert!(error.message.contains("loaded projector token capacity"));
        } else {
            assert!(result.is_ok(), "video fits {capacity} tokens");
        }
    }
    let parsed = inputs(&side, "input", &video).unwrap_or_else(|e| panic!("{}", e.message));
    let files = [&image, &second]
        .map(|part| media::load(part["image_url"]["url"].as_str().unwrap()).unwrap());
    assert_ne!(files[0].key, files[1].key);
    let frames = files
        .iter()
        .map(|file| vision::Rgb8::from_bytes(&file.bytes).unwrap())
        .collect::<Vec<_>>();
    let (_, prepared) = vision::arch::gemma4v::prepare_clip(&frames, (1, 1), 16).unwrap();
    assert_ne!(prepared[0].input, prepared[1].input);
    let got = &parsed[0];
    assert_eq!(got.images.len(), prepared.len());
    let mut want = vec![side.tok.bos()];
    want.extend(side.tok.encode("Video:"));
    for ((feed, frame), file) in got.images.iter().zip(&prepared).zip(&files) {
        want.extend(side.tok.encode("<|image>"));
        assert_eq!(feed.at, want.len());
        assert_eq!(&*feed.prepared, frame);
        assert_eq!(feed.key, file.key);
        want.extend(std::iter::repeat_n(side.tok.bos(), frame.span_len));
        want.extend(side.tok.encode("<image|>"));
    }
    if side.tok.add_eos() {
        want.push(side.tok.eos());
    }
    assert_eq!(got.ids, want);
    side.ctx = got.ids.len();
    assert!(inputs(&side, "input", &video).is_ok());
    side.ctx -= 1;
    assert!(inputs(&side, "input", &video).is_err());
    side.ctx = 8192;
    let square = vision::Rgb8::filled(576, 576, [10, 20, 30]);
    for capacity in [120, 121, 122] {
        let rule = vision::size::SizeRule::new(
            16,
            3,
            vision::arch::TokenLimits::new(1, capacity).unwrap(),
        )
        .unwrap();
        let model = vision::arch::gemma4v::Media::new(rule);
        let result = vision::MediaModel::prepare_video(&model, std::slice::from_ref(&square));
        assert_eq!(result.is_ok(), capacity >= 121);
        if let Ok(prepared) = result {
            assert_eq!(prepared[0].span_len, 121);
        }
    }
    for ty in ["input_video", "video_url"] {
        let body = json!({"content":[{"type":ty, (ty):{"data":"data:video/mp4;base64,AA=="}}]});
        let error = inputs(&side, "input", &body)
            .err()
            .expect("encoded video refusal");
        assert_eq!(error.code, 400);
        assert!(error.message.contains("decoded one-FPS image_url frames"));
    }
    for count in [1, 32] {
        let content = std::iter::once(json!({"type":"text","text":"Video:"}))
            .chain(std::iter::repeat_n(image.clone(), count))
            .collect::<Vec<_>>();
        assert!(inputs(&side, "input", &json!({"content":content})).is_ok());
    }
    for content in [
        json!([{"type":"text","text":"Video:"}]),
        json!([{"type":"text","text":"Video:"},{"type":"text","text":"not a frame"}]),
        Value::Array(
            std::iter::once(json!({"type":"text","text":"Video:"}))
                .chain(std::iter::repeat_n(image.clone(), 33))
                .collect(),
        ),
    ] {
        assert!(inputs(&side, "input", &json!({"content":content})).is_err());
    }
    side.ctx = 1;
    assert!(inputs(&side, "input", &video).is_err());
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
