use embed::input::{InputPart, assemble, prompt_ids, token_ids};
use embed::request::{Error, Request, check_tokens};
use embed::{Norm, encode_base64, mean, normalize};
use serde_json::json;

#[test]
fn pooling_and_wire_bytes() {
    let rows = [1.0e10, 3.0, 1.0, -4.0, -1.0e10, 1.0];
    assert_eq!(mean(&rows, 2), [1.0 / 3.0, 0.0]);
    for (rows, width) in [(&[][..], 2), (&[1.0][..], 0), (&[1.0][..], 2)] {
        assert!(std::panic::catch_unwind(|| mean(rows, width)).is_err());
    }
    assert_eq!(encode_base64(b""), "");
    let pooled = vec![vec![3.0, -4.0, 12.0]];
    for p in [0, 1, 3] {
        let b = json!({"input": "text", "embd_normalize": p});
        let request = Request::parse(b.as_object().unwrap(), "encoder", &[3], true).unwrap();
        assert_eq!(
            request.answer(&pooled, 1).unwrap()["data"][0]["embedding"],
            json!(normalize(&pooled[0], Norm::of(p).unwrap()).unwrap())
        );
    }
}

#[test]
fn request_shapes_and_named_refusals() {
    for (input, count) in [
        (json!("text"), 1),
        (json!([1, 2]), 1),
        (json!(["text", 2]), 1),
        (json!(["first", "second"]), 2),
        (json!([[1], [2]]), 2),
        (json!({"content": [{"type": "text", "text": "hi"}]}), 1),
    ] {
        let b = json!({"input": input});
        assert_eq!(
            Request::parse(b.as_object().unwrap(), "encoder", &[2, 3], true)
                .unwrap()
                .inputs
                .len(),
            count
        );
    }
    let pooled = vec![vec![3.0, -4.0, 12.0]];
    for oai in [false, true] {
        for name in [None, Some("query"), Some("")] {
            let mut b = json!({"input": "text"});
            if let Some(name) = name {
                b["prompt_name"] = json!(name);
            }
            let r = Request::parse(b.as_object().unwrap(), "encoder", &[3], oai).unwrap();
            assert_eq!(r.prompt_name, name);
        }
        let b = json!({"content": "text", "encoding_format": "base64", "model": "encoder"});
        let r = Request::parse(b.as_object().unwrap(), "encoder", &[3], oai).unwrap();
        assert_eq!(r.key, "content");
        assert_eq!(
            r.answer(&pooled, 1).unwrap(),
            json!([{"embedding": [normalize(&pooled[0], Norm::default()).unwrap()], "index": 0}])
        );
    }
    for (field, value, message) in [
        ("prompt_name", json!(7), "prompt_name must be a string"),
        ("prompt_name", json!(null), "prompt_name must be a string"),
        ("model", json!(1), "model must be a string"),
        ("model", json!("wrong"), "embeddings are served by"),
        (
            "encoding_format",
            json!(1),
            "encoding_format must be a string",
        ),
        (
            "encoding_format",
            json!("hex"),
            "Can be either float or base64",
        ),
        ("embd_normalize", json!(-2), "embd_normalize -2"),
        (
            "embd_normalize",
            json!(2.5),
            "embd_normalize must be an integer",
        ),
        (
            "embd_normalize",
            json!(2147483648u64),
            "embd_normalize 2147483648",
        ),
        ("dimensions", json!(1), "dimensions 1"),
        ("dimensions", json!("2"), "dimensions must be an integer"),
        ("dimensions", json!(2.5), "dimensions must be an integer"),
        ("input", json!([]), "\"input\" must not be empty"),
    ] {
        let mut b = json!({"input": "text"});
        b[field] = value;
        let error = Request::parse(b.as_object().unwrap(), "encoder", &[2, 3], true).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
    let b = json!({});
    assert!(Request::parse(b.as_object().unwrap(), "encoder", &[3], true).is_err());
    let b = json!({"input": "text", "model": null, "encoding_format": null, "embd_normalize": null, "dimensions": 2.0});
    assert!(Request::parse(b.as_object().unwrap(), "encoder", &[2, 3], true).is_ok());
    assert!(matches!(check_tokens(0, 4), Err(Error::Invalid(_))));
    assert!(matches!(
        check_tokens(5, 4),
        Err(Error::Context { tokens: 5, ctx: 4 })
    ));
    check_tokens(4, 4).unwrap();
    let b = json!({"input": "text", "encoding_format": "base64", "embd_normalize": -1});
    let r = Request::parse(b.as_object().unwrap(), "encoder", &[2], true).unwrap();
    assert!(r.answer(&[vec![1.0]], 1).is_err());
    for encoding in ["float", "base64"] {
        let b = json!({"input": [1], "encoding_format": encoding, "embd_normalize": -1});
        let r = Request::parse(b.as_object().unwrap(), "encoder", &[1], true).unwrap();
        for x in [f32::NAN, f32::INFINITY] {
            assert!(r.answer(&[vec![x]], 1).is_err());
        }
    }
    assert_eq!(
        r.answer(&[vec![0.0, -0.0]], 1).unwrap()["data"][0]["embedding"],
        "AAAAAAAAAAA="
    );
}

#[test]
fn mixed_inputs_follow_the_first_elements_special_token_rule() {
    let encode = |s: &str| s.bytes().map(u32::from).collect();
    for (value, want) in [
        (json!("ab"), vec![1, 97, 98, 2]),
        (json!(["ab", 7, "cd"]), vec![1, 97, 98, 2, 7, 99, 100]),
        (json!([7, "ab"]), vec![7, 97, 98]),
        (json!([]), vec![]),
    ] {
        assert_eq!(
            prompt_ids("input", &value, encode, Some(1), Some(2), 256).unwrap(),
            want
        );
    }
    for value in [
        json!(-1),
        json!(256),
        json!(1.5),
        json!(true),
        json!(null),
        json!(4294967296u64),
    ] {
        let error = token_ids(256, &json!([value])).unwrap_err();
        assert!(error.to_string().contains("invalid token id"));
    }
    for value in [json!({"prompt_string": "x"}), json!(null), json!(1)] {
        assert!(prompt_ids("input", &value, encode, None, None, 256).is_err());
    }
    let value = json!([{"multimodal_data": [1]}]);
    assert!(
        prompt_ids("input", &value, encode, None, None, 256)
            .unwrap_err()
            .to_string()
            .contains("Multimodal data provided")
    );
}

#[test]
fn ordered_parts_keep_one_pair_of_special_tokens() {
    let parts = || {
        [
            Ok(InputPart::Text("a")),
            Ok(InputPart::Media {
                begin: "b",
                end: "e",
                positions: 2,
                kind: "image",
            }),
            Ok(InputPart::Text("z")),
        ]
        .into_iter()
    };
    let encode = |s: &str| s.bytes().map(u32::from).collect();
    let input = assemble(parts(), encode, Some(1), Some(2), Some(1), 8).unwrap();
    assert_eq!(input.ids, [1, 97, 98, 1, 1, 101, 122, 2]);
    assert_eq!(input.spans, vec![3..5]);
    assert!(assemble(parts(), encode, Some(1), Some(2), Some(1), 4).is_err());
    assert!(assemble(std::iter::empty(), encode, None, None, Some(1), 8).is_err());
    let e = assemble(parts(), encode, None, Some(2), None, 8)
        .err()
        .expect("a media span with no filler is refused");
    assert!(e.contains("image span has no filler token id"), "{e}");
    let text_only = assemble(
        [Ok(InputPart::Text("a"))].into_iter(),
        encode,
        None,
        Some(2),
        None,
        8,
    );
    assert_eq!(text_only.unwrap().ids, [97, 2]);
    assert!(
        assemble(
            [Ok(InputPart::Media {
                begin: "",
                end: "",
                positions: 0,
                kind: "audio"
            })]
            .into_iter(),
            encode,
            None,
            None,
            Some(1),
            8
        )
        .is_err()
    );
    let empty = assemble(
        [Ok(InputPart::Text(""))].into_iter(),
        encode,
        None,
        None,
        Some(1),
        8,
    );
    assert!(empty.is_err());
}
