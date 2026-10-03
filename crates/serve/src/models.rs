//! `GET /v1/models` (and `/models`): the one model a server serves, in llama.cpp's server's shape:
//! `data`, OpenAI's list, and `models` beside it (`tools/server/server-context.cpp` at `a4cb4c61`,
//! `get_res_models`: its `name` and `model`, the entry's other fields there being placeholders).
//! Every server of this crate answers through [`listing`].

use serde_json::{Value, json};

/// The listing of the model `id`: `created` (unix seconds), `meta` as the server knows it, and
/// `max_model_len` when the server states one.
pub(crate) fn listing(id: &str, created: u64, meta: Value, max_model_len: Option<usize>) -> Value {
    let mut entry = json!({
        "id": id,
        "object": "model",
        "created": created,
        "owned_by": "bloomery",
        "meta": meta,
    });
    if let (Some(n), Value::Object(o)) = (max_model_len, &mut entry) {
        o.insert("max_model_len".to_owned(), json!(n));
    }
    json!({
        "models": [{ "name": id, "model": id }],
        "object": "list",
        "data": [entry],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_lists_name_the_one_model() {
        let v = listing("m.gguf", 7, json!({ "n_ctx": 16 }), Some(16));
        assert_eq!(v["data"][0]["id"], "m.gguf");
        assert_eq!(v["data"][0]["max_model_len"], 16);
        assert_eq!(
            v["models"],
            json!([{ "name": "m.gguf", "model": "m.gguf" }])
        );
        assert!(
            listing("m", 7, json!({}), None)["data"][0]
                .get("max_model_len")
                .is_none()
        );
    }
}
