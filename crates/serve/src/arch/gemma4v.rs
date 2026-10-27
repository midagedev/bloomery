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
    let mut want = vec![side.tok.bos().expect("a BOS")];
    want.extend(side.tok.encode("a<|image>"));
    for (i, after) in [(0, "b"), (1, "")] {
        want.extend(side.tok.encode("<|image>"));
        assert_eq!(got.images[i].at, want.len());
        assert_eq!(got.images[i].prepared.span_len, 1);
        want.extend([side.tok.bos().expect("a BOS"); 1]);
        want.extend(side.tok.encode("<image|>"));
        want.extend(side.tok.encode(after));
    }
    if side.tok.add_eos() {
        want.push(side.tok.eos());
    }
    assert_eq!(got.ids, want);
    side.audio = Some(Arc::new(vision::arch::gemma4a::Preprocessor::default()));
    let audio = json!({"type":"input_audio","input_audio":{
        "format":"f32","sample_rate":16000,"channels":1,"data":media::encode_base64(&[0; 8])
    }});
    for audio_first in [false, true] {
        let parts = if audio_first {
            [audio.clone(), image.clone()]
        } else {
            [image.clone(), audio.clone()]
        };
        let mixed = input(&side, "input", &json!({"content":parts}))
            .unwrap_or_else(|e| panic!("{}", e.message));
        assert_eq!(mixed.images.len(), 1);
        assert_eq!(mixed.audio.len(), 1);
        let mut want = vec![side.tok.bos().expect("a BOS")];
        for is_audio in [audio_first, !audio_first] {
            let (begin, end, at) = if is_audio {
                assert_eq!(mixed.audio[0].chunks.len(), 1);
                assert_eq!(mixed.audio[0].chunks[0].tokens(), 1);
                (
                    vision::arch::gemma4a::BEGIN,
                    vision::arch::gemma4a::END,
                    mixed.audio[0].at,
                )
            } else {
                assert_eq!(mixed.images[0].prepared.span_len, 1);
                ("<|image>", "<image|>", mixed.images[0].at)
            };
            want.extend(side.tok.encode(begin));
            assert_eq!(at, want.len());
            want.push(side.tok.bos().expect("a BOS"));
            want.extend(side.tok.encode(end));
        }
        if side.tok.add_eos() {
            want.push(side.tok.eos());
        }
        assert_eq!(mixed.ids, want);
    }
    let model = side.media.as_ref().expect("image preparer");
    assert_eq!(model.image_token(), None);
    assert!(matches!(
        media::expand_prompt(&[side.tok.bos().expect("a BOS")], &[], &**model),
        Err(media::MediaError::NoChatImageToken)
    ));
    // Only context changes: the same valid input fits exactly, but not one position below.
    for ctx in [got.ids.len() - 1, got.ids.len(), got.ids.len() + 1] {
        side.ctx = ctx;
        let result = inputs(&side, "input", &json!([body.clone()]));
        if ctx < got.ids.len() {
            let error = result.err().expect("context refusal");
            assert_eq!(error.code, 400);
        } else {
            let parsed = result.unwrap_or_else(|e| panic!("{}", e.message));
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].ids, got.ids);
            assert_eq!(parsed[0].images.len(), got.images.len());
            for (feed, want) in parsed[0].images.iter().zip(&got.images) {
                assert_eq!(feed.at, want.at);
                assert_eq!(feed.key, want.key);
                assert_eq!(feed.prepared, want.prepared);
            }
        }
    }
    side.ctx = 8192;
    let second = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAAD0lEQVR4nGPgEpHTMLIBAAI3ANNbVlHYAAAAAElFTkSuQmCC"}});
    let video = json!({"content":[{"type":"text","text":"Video:"},image.clone(),second.clone()]});
    let files = [&image, &second]
        .map(|part| media::load(part["image_url"]["url"].as_str().unwrap()).unwrap());
    assert_ne!(files[0].key, files[1].key);
    let frames = files
        .iter()
        .map(|file| vision::Rgb8::from_bytes(&file.bytes).unwrap())
        .collect::<Vec<_>>();
    let (_, prepared) = vision::arch::gemma4v::prepare_clip(&frames, (1, 1), 16).unwrap();
    assert_ne!(prepared[0].input, prepared[1].input);
    assert_eq!(
        prepared
            .iter()
            .map(|frame| frame.span_len)
            .collect::<Vec<_>>(),
        [72, 72]
    );
    let mut want = vec![side.tok.bos().expect("a BOS")];
    want.extend(side.tok.encode("Video:"));
    let mut positions = Vec::new();
    for frame in &prepared {
        want.extend(side.tok.encode("<|image>"));
        positions.push(want.len());
        want.extend(std::iter::repeat_n(
            side.tok.bos().expect("a BOS"),
            frame.span_len,
        ));
        want.extend(side.tok.encode("<image|>"));
    }
    if side.tok.add_eos() {
        want.push(side.tok.eos());
    }
    let assert_video = |parsed: &[Input]| {
        assert_eq!(parsed.len(), 1);
        let got = &parsed[0];
        assert_eq!(got.ids, want);
        assert_eq!(got.images.len(), prepared.len());
        for (((feed, frame), file), at) in
            got.images.iter().zip(&prepared).zip(&files).zip(&positions)
        {
            assert_eq!(feed.at, *at);
            assert_eq!(&*feed.prepared, frame);
            assert_eq!(feed.key, file.key);
        }
    };
    // Both files decode independently above; only the loaded projector capacity varies.
    for capacity in [71, 72, 73] {
        let rule = vision::size::SizeRule::new(
            16,
            3,
            vision::arch::TokenLimits::new(1, capacity).unwrap(),
        )
        .unwrap();
        let model = vision::arch::gemma4v::Media::new(rule);
        let model_result = vision::MediaModel::prepare_video(&model, &frames);
        side.media = Some(Arc::new(model));
        let result = inputs(&side, "input", &video);
        if capacity < 72 {
            assert!(matches!(
                model_result,
                Err(vision::VisionError::Limits { .. })
            ));
            let error = result
                .err()
                .expect("video exceeds loaded projector capacity");
            assert_eq!(error.code, 400);
        } else {
            assert_eq!(model_result.unwrap(), prepared);
            assert_video(&result.unwrap_or_else(|e| panic!("{}", e.message)));
        }
    }
    for ctx in [want.len() - 1, want.len(), want.len() + 1] {
        side.ctx = ctx;
        let result = inputs(&side, "input", &video);
        if ctx < want.len() {
            let error = result.err().expect("video exceeds context");
            assert_eq!(error.code, 400);
        } else {
            assert_video(&result.unwrap_or_else(|e| panic!("{}", e.message)));
        }
    }
    side.ctx = 8192;
    let square = vision::Rgb8::filled(576, 576, [10, 20, 30]);
    let square_frames = std::slice::from_ref(&square);
    let prepared = vision::arch::gemma4v::prepare_video(square_frames, 16).unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].span_len, 121);
    for capacity in [120, 121, 122] {
        let rule = vision::size::SizeRule::new(
            16,
            3,
            vision::arch::TokenLimits::new(1, capacity).unwrap(),
        )
        .unwrap();
        let model = vision::arch::gemma4v::Media::new(rule);
        let result = vision::MediaModel::prepare_video(&model, square_frames);
        if capacity < 121 {
            assert!(matches!(result, Err(vision::VisionError::Limits { .. })));
        } else {
            assert_eq!(result.unwrap(), prepared);
        }
    }
    // A clip of either spelling reaches the server's decoder, which a bare side has none of: the
    // refusal names ffmpeg and the decoded-frames route.
    for part in [
        json!({"type":"input_video","input_video":{"data":"AAAA","format":"mp4"}}),
        json!({"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}),
    ] {
        let error = inputs(&side, "input", &json!({"content":[part]}))
            .err()
            .expect("a clip with no decoder");
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

/// A clip in an embedding input is decoded by the server and planned by the side model's rule:
/// the model's `Video:` text, then one delimited span a group, each group the rows the
/// decoded-frames route prepares for the same frame, decoded once at the plan's indices on the
/// seat's threads. Without a decoder, or with a model that has no rule, it is a named 400.
#[test]
fn a_clip_in_an_embedding_input_is_planned_decoded_and_delimited() {
    use crate::video::MockClips;
    let mut side = Side::new(Box::new(MockEmbed::new(4096)));
    let rule = vision::size::SizeRule::new(16, 3, vision::arch::TokenLimits::new(1, 280).unwrap())
        .unwrap();
    let model = Arc::new(vision::arch::gemma4v::Media::new(rule));
    side.media = Some(Arc::clone(&model) as media::SharedMediaModel);
    let mock = MockClips::new();
    let clip = |spec: &str| {
        json!({"type": "input_video", "input_video": {
            "data": media::encode_base64(spec.as_bytes()), "format": "mp4"
        }})
    };
    let got = inputs_with(
        &side,
        &mock.support(2),
        "input",
        &json!([{"content": [clip("3 1/1 64x64")]}]),
    )
    .unwrap_or_else(|e| panic!("{}", e.message));
    let input = &got[0];
    assert_eq!(input.images.len(), 3);
    // Three seconds at one frame a second, each at the middle of its second.
    assert_eq!(*mock.calls.lock().unwrap(), [(vec![0, 1, 2], 2)]);
    let tok = &side.tok;
    let text = tok.encode("Video:");
    assert_eq!(Some(input.ids[0]), tok.bos());
    assert_eq!(input.ids[1..=text.len()], text[..]);
    let (begin, end) = (tok.encode("<|image>").len(), tok.encode("<image|>").len());
    assert_eq!(input.images[0].at, 1 + text.len() + begin);
    for pair in input.images.windows(2) {
        assert_eq!(
            pair[1].at,
            pair[0].at + pair[0].prepared.span_len + end + begin
        );
    }
    let frames: Vec<vision::Rgb8> = (0..3u8)
        .map(|i| vision::Rgb8 {
            width: 64,
            height: 64,
            data: vec![i; 64 * 64 * 3],
        })
        .collect();
    let want = vision::MediaModel::prepare_video(&*model, &frames).unwrap();
    for (feed, want) in input.images.iter().zip(&want) {
        assert_eq!(feed.prepared.span_len, want.span_len);
        assert_eq!(
            feed.prepared.grid().unwrap().bf16,
            want.grid().unwrap().bf16
        );
    }
    let refused = |video: &crate::video::VideoSupport, side: &Side| {
        inputs_with(
            side,
            video,
            "input",
            &json!([{"content": [clip("3 1/1 64x64")]}]),
        )
        .err()
        .expect("a refusal")
    };
    let absent = refused(&crate::video::VideoSupport::absent("not found"), &side);
    assert_eq!(absent.code, 400);
    assert!(
        absent
            .message
            .starts_with("video input needs ffmpeg and ffprobe on PATH (not found)")
    );
    side.media = None;
    let no_model = refused(&mock.support(2), &side);
    assert_eq!(no_model.message, "video input needs --embd-mmproj");
    let over = {
        let mut small = Side::new(Box::new(MockEmbed::new(100)));
        small.media = Some(Arc::clone(&model) as media::SharedMediaModel);
        refused(&mock.support(2), &small)
    };
    assert!(over.message.starts_with("video: "), "{}", over.message);
}

/// A clip is planned under what the assembled input charges it: BOS and EOS around the input, its
/// `Video:` text once, and a delimiter pair and the tokens of each group. Six 64×64 groups take 81
/// positions each and are charged 590 [derived: 2 + 6 + 6 × (16 + 81)], so at a context of 550 the
/// clip is five groups, not six refused by the assembly. At each group count's charge the input
/// assembles within the context with that many groups, and one position less it has one group
/// fewer; under one group's charge the clip is refused by name.
#[test]
fn a_clips_markup_is_charged_to_its_budget() {
    use crate::video::MockClips;
    let rule = vision::size::SizeRule::new(16, 3, vision::arch::TokenLimits::new(1, 280).unwrap())
        .unwrap();
    let model = Arc::new(vision::arch::gemma4v::Media::new(rule));
    let mock = MockClips::new();
    let clip = json!({"type": "input_video", "input_video": {
        "data": media::encode_base64(b"6 1/1 64x64"), "format": "mp4"
    }});
    let side_at = |ctx: usize| {
        let mut side = Side::new(Box::new(MockEmbed::new(ctx)));
        side.media = Some(Arc::clone(&model) as media::SharedMediaModel);
        side
    };
    let tok = side_at(0).tok;
    let frame = vision::Rgb8::filled(64, 64, [1, 2, 3]);
    let tokens = vision::MediaModel::prepare_video(&*model, std::slice::from_ref(&frame)).unwrap()
        [0]
    .span_len;
    let around = tok.encode("<|image>").len() + tok.encode("<image|>").len();
    let fixed = 2 + tok.encode("Video:").len();
    assert_eq!((tokens, around, fixed), (81, 16, 8));
    let charge = |groups: usize| fixed + groups * (tokens + around);
    let at = |ctx: usize| {
        inputs_with(
            &side_at(ctx),
            &mock.support(1),
            "input",
            &json!([{"content": [clip.clone()]}]),
        )
    };
    let mut cases = vec![(0, 0), (charge(1) - 1, 0), (550, 5), (charge(6) + 100, 6)];
    for groups in 1..=6 {
        cases.push((charge(groups), groups));
        if groups > 1 {
            cases.push((charge(groups) - 1, groups - 1));
        }
    }
    for (ctx, groups) in cases {
        match at(ctx) {
            Ok(got) => {
                assert_eq!(got[0].images.len(), groups, "context {ctx}");
                assert_eq!(got[0].ids.len(), charge(groups), "context {ctx}");
            }
            Err(e) => {
                assert_eq!(groups, 0, "context {ctx}: {}", e.message);
                assert!(e.message.starts_with("video: "), "{}", e.message);
            }
        }
    }
}

/// The other media of an input are charged to its clips' budget as the assembly charges them: a
/// delimiter pair and the tokens of each. A one-token still takes 17 positions and a one-token
/// audio span 17 [derived: 8 + 8 delimiters + 1], so with the 64×64 clip of
/// [`a_clips_markup_is_charged_to_its_budget`] (a group 97 positions, the fixed text 8) a context
/// of 493, which five groups fill alone, leaves four groups beside the still or the audio, in
/// either order; the fifth group comes at 17 positions more, and one position less it is four.
#[test]
fn a_still_and_an_audio_span_are_charged_before_the_clip_is_planned() {
    use crate::video::MockClips;
    let rule = vision::size::SizeRule::new(16, 3, vision::arch::TokenLimits::new(1, 280).unwrap())
        .unwrap();
    let model = Arc::new(vision::arch::gemma4v::Media::new(rule));
    let mock = MockClips::new();
    let clip = json!({"type": "input_video", "input_video": {
        "data": media::encode_base64(b"6 1/1 64x64"), "format": "mp4"
    }});
    let still = json!({"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADUlEQVR4nGM4kWIERAAKxQK9KV6Y1QAAAABJRU5ErkJggg=="}});
    let audio = json!({"type":"input_audio","input_audio":{
        "format":"f32","sample_rate":16000,"channels":1,"data":media::encode_base64(&[0; 8])
    }});
    let side_at = |ctx: usize| {
        let mut side = Side::new(Box::new(MockEmbed::new(ctx)));
        side.media = Some(Arc::clone(&model) as media::SharedMediaModel);
        side.audio = Some(Arc::new(vision::arch::gemma4a::Preprocessor::default()));
        side
    };
    let tok = side_at(0).tok;
    let around = |begin: &str, end: &str| tok.encode(begin).len() + tok.encode(end).len();
    let group = around("<|image>", "<image|>") + 81;
    let fixed = 2 + tok.encode("Video:").len();
    let charge = |groups: usize| fixed + groups * group;
    let other = |kind: &str| {
        if kind == "audio" {
            around(vision::arch::gemma4a::BEGIN, vision::arch::gemma4a::END) + 1
        } else {
            around("<|image>", "<image|>") + 1
        }
    };
    assert_eq!(
        (group, fixed, other("image"), other("audio")),
        (97, 8, 17, 17)
    );
    for (kind, medium) in [("image", &still), ("audio", &audio)] {
        for clip_first in [false, true] {
            let parts = if clip_first {
                [clip.clone(), medium.clone()]
            } else {
                [medium.clone(), clip.clone()]
            };
            for (ctx, groups) in [
                (charge(5), 4),
                (charge(5) + other(kind) - 1, 4),
                (charge(5) + other(kind), 5),
            ] {
                let got = inputs_with(
                    &side_at(ctx),
                    &mock.support(1),
                    "input",
                    &json!([{"content": parts}]),
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "{kind} clip_first {clip_first}, context {ctx}: {}",
                        e.message
                    )
                });
                let input = &got[0];
                let spans = input.images.len() + input.audio.len();
                assert_eq!(
                    spans,
                    groups + 1,
                    "{kind} clip_first {clip_first}, context {ctx}"
                );
                assert_eq!(
                    input.ids.len(),
                    charge(groups) + other(kind),
                    "{kind} clip_first {clip_first}, context {ctx}"
                );
            }
        }
    }
    // Images and audio past the context refuse the input before the clip is decoded.
    let decoded = mock.calls.lock().unwrap().len();
    let refused = inputs_with(
        &side_at(30),
        &mock.support(1),
        "input",
        &json!([{"content": [still.clone(), audio.clone(), clip.clone()]}]),
    )
    .err()
    .expect("a refusal");
    assert_eq!(refused.code, 400);
    assert!(
        refused
            .message
            .ends_with("input exceeds the embedding context"),
        "{}",
        refused.message
    );
    assert_eq!(
        mock.calls.lock().unwrap().len(),
        decoded,
        "the clip was decoded"
    );
}
