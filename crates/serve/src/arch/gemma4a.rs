//! Decoded audio at the embedding request boundary.

use super::*;
use crate::mock::MockEmbed;

#[test]
fn audio_parts_keep_their_delimiters_and_refuse_invalid_pcm() {
    let mut side = Side::new(Box::new(MockEmbed::new(128)));
    side.audio = Some(Arc::new(vision::arch::gemma4a::Preprocessor::default()));
    let pcm: Vec<u8> = [0.125_f32, -0.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let audio = json!({"type":"input_audio","input_audio":{
        "format":"f32","sample_rate":16000,"channels":1,"data":media::encode_base64(&pcm)
    }});
    let body = json!({"content":[{"type":"text","text":"before"},audio.clone(),{"type":"text","text":"after"},audio.clone()]});
    let parsed = input(&side, "input", &body).unwrap_or_else(|e| panic!("{}", e.message));
    assert!(parsed.images.is_empty());
    assert_eq!(parsed.audio.len(), 2);
    let mut want = vec![side.tok.bos().expect("a BOS")];
    want.extend(side.tok.encode("before"));
    for (feed, after) in parsed.audio.iter().zip(["after", ""]) {
        want.extend(side.tok.encode(vision::arch::gemma4a::BEGIN));
        assert_eq!(feed.at, want.len());
        assert_eq!(feed.chunks.len(), 1);
        assert_eq!(feed.chunks[0].tokens(), 1);
        want.push(side.tok.bos().expect("a BOS"));
        want.extend(side.tok.encode(vision::arch::gemma4a::END));
        want.extend(side.tok.encode(after));
    }
    if side.tok.add_eos() {
        want.push(side.tok.eos());
    }
    assert_eq!(parsed.ids, want);
    side.ctx = parsed.ids.len();
    assert!(inputs(&side, "input", &json!([body.clone()])).is_ok());
    side.ctx -= 1;
    let error = inputs(&side, "input", &json!([body.clone()]))
        .err()
        .expect("context limit");
    assert_eq!(error.kind, "exceed_context_size_error");
    side.ctx = parsed.audio[0].at;
    let error = input(&side, "input", &body).err().expect("context refusal");
    assert_eq!(error.message, "audio input exceeds the embedding context");
    side.ctx = 128;
    for (field, value, message) in [
        ("format", json!("wav"), "encoded audio is not supported"),
        ("sample_rate", json!(48000), "sample_rate=16000"),
        ("channels", json!(2), "channels=1"),
        ("data", json!(""), "at least two samples"),
        (
            "data",
            json!(media::encode_base64(&0.0_f32.to_le_bytes())),
            "at least two samples",
        ),
        ("data", json!("AA=="), "multiple of four"),
        ("data", json!("!!!!"), "base64"),
        (
            "data",
            json!(media::encode_base64(
                &[f32::NAN.to_le_bytes(), 0.0_f32.to_le_bytes()].concat()
            )),
            "not finite",
        ),
    ] {
        let mut part = audio.clone();
        part["input_audio"][field] = value;
        let error = input(&side, "input", &json!({"content":[part]}))
            .err()
            .expect("PCM refusal");
        assert!(error.message.contains(message), "{}", error.message);
    }
    for value in [
        Value::Null,
        json!(true),
        json!(0),
        json!(1.0),
        json!([]),
        json!({}),
    ] {
        for field in ["format", "sample_rate", "channels", "data"] {
            let mut part = audio.clone();
            part["input_audio"][field] = value.clone();
            assert!(
                input(&side, "input", &json!({"content":[part]})).is_err(),
                "{field}: {value}"
            );
        }
        let mut part = audio.clone();
        part["input_audio"] = value.clone();
        assert!(
            input(&side, "input", &json!({"content":[part]})).is_err(),
            "input_audio: {value}"
        );
    }
    for field in ["format", "sample_rate", "channels", "data"] {
        let mut part = audio.clone();
        part["input_audio"]
            .as_object_mut()
            .expect("audio object")
            .remove(field);
        assert!(
            input(&side, "input", &json!({"content":[part]})).is_err(),
            "missing {field}"
        );
    }
    side.audio = None;
    let error = input(&side, "input", &json!({"content":[audio]}))
        .err()
        .expect("missing tower");
    assert!(error.message.contains("gemma4a --embd-mmproj"));
}
