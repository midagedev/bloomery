//! A model card's facts a server reads: the `README.md` front matter Hugging Face keeps for a repo
//! (its model tree's `base_model` and `base_model_relation`).

/// The one `base_model` of a model card's front matter (the YAML block
/// between its first two `---` lines) whose `base_model_relation` is
/// `quantized`: Hugging Face's own record that the repo holds a
/// quantization of that model. A card with no front matter, another
/// relation, or other than one base model gives `None`.
pub fn quantizes(text: &str) -> Option<String> {
    let mut lines = text.trim_start_matches('\u{feff}').lines();
    if lines.next()?.trim_end() != "---" {
        return None;
    }
    let (mut base, mut relation, mut in_base) = (Vec::new(), None, false);
    for line in lines {
        let line = line.trim_end();
        if line == "---" {
            break;
        }
        let item = line.trim_start().strip_prefix("- ");
        if in_base && let Some(v) = item {
            base.push(unquote(v));
            continue;
        }
        in_base = false;
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "base_model" if value.is_empty() => in_base = true,
            "base_model" => match value.strip_prefix('[').and_then(|v| v.strip_suffix(']')) {
                Some(list) => base.extend(list.split(',').map(unquote)),
                None => base.push(unquote(value)),
            },
            "base_model_relation" => relation = Some(unquote(value)),
            _ => {}
        }
    }
    match (relation.as_deref(), base.as_slice()) {
        (Some("quantized"), [one]) if !one.is_empty() => Some(one.clone()),
        _ => None,
    }
}

/// A YAML scalar without its surrounding quotes.
fn unquote(v: &str) -> String {
    let v = v.trim();
    v.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .or_else(|| v.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
        .unwrap_or(v)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::quantizes;

    #[test]
    fn a_card_names_the_model_its_repo_quantizes() {
        // bartowski's Clef card, as published (front matter's first lines).
        let card = "---\nquantized_by: bartowski\nlicense: apache-2.0\nbase_model: Cloudflare/clef-flash\ntags:\n- clef\n- qwen3.5\nbase_model_relation: quantized\n---\n\n## Llamacpp\nbase_model: Other/x\n";
        assert_eq!(quantizes(card).as_deref(), Some("Cloudflare/clef-flash"));
        for (text, why) in [
            (
                "---\nbase_model:\n  - 'Org/a'\nbase_model_relation: \"quantized\"\n---\n",
                Some("Org/a"),
            ),
            (
                "---\nbase_model: [Org/a]\nbase_model_relation: quantized\n---\n",
                Some("Org/a"),
            ),
            // A finetune, an adapter or a merge keeps no head of its base.
            (
                "---\nbase_model: Org/a\nbase_model_relation: finetune\n---\n",
                None,
            ),
            (
                "---\nbase_model:\n- Org/a\n- Org/b\nbase_model_relation: quantized\n---\n",
                None,
            ),
            ("---\nbase_model: Org/a\n---\n", None),
            // A base model past the front matter, or no front matter, is no card fact.
            (
                "---\nlicense: mit\n---\nbase_model: Org/a\nbase_model_relation: quantized\n",
                None,
            ),
            ("base_model: Org/a\nbase_model_relation: quantized\n", None),
        ] {
            assert_eq!(quantizes(text).as_deref(), why, "{text}");
        }
    }
}
