//! What a server needs of lev (interfaze-ai's, `ggml-org/lev-GGUF`) to seat it: the route its
//! requests are posted to, the context its prompts fill, the backbone architecture, and the decision
//! type that names its file. Its head is the file's own language-model head and its prompt the file's
//! own `systemone` template, so no head file, config or repo is asked for; the policy is
//! llama.cpp's `server-decision.cpp` for lev.

use crate::label::Policy;
use crate::request::Rules;

/// The row's name, as a server's refusals and `/props` give it.
pub const NAME: &str = "lev";

/// The routes a SystemOne request is posted to: llama.cpp's.
pub const ROUTES: &[&str] = &["/v1/systemone"];

/// The context unless the server is told otherwise. [derived] The Clef row's value: a prompt is a
/// question's variant (the state and the options), and the gate suite's longest, a log state, is 6430
/// ids, 2.5 times under it. The file keeps K and V on one layer in four
/// (`block_count` 32, `full_attention_interval` 4, `head_count_kv` 4, `key_length` 256): 8 layers × 2
/// × 4 KV heads × 256 × 2 B = 32 KB a token, so the whole context costs 512 MB.
pub const CTX: usize = 16384;

/// The file architectures the label head reads: Qwen3.5 dense, lev's backbone.
pub const BACKBONES: &[&str] = &["qwen35"];

/// The `<arch>.decision.type` a lev file carries (llama.cpp's `COMMON_DECISION_TYPE_NAMES`).
pub const DECISION_TYPE: &str = "lev";

/// The `--hf` that seats the row, as a refusal names it.
pub const QUANT_REPO: &str = "ggml-org/lev-GGUF:Q4_K_M";

/// lev's rules (llama.cpp: no `choice_sorted`, no `noul_true_first`; checks as its server's), its
/// noul read on a 9-point rating scale (`DECISION_LEV_N_RATINGS`: 0 certainly no ... 8 certainly
/// yes), a choice of two options or more asked in two orders (`n_variants`), the template's keys
/// sorted (`render`: lev was trained with sorted keys) and the temperature buckets by option count
/// (`get_temperature`).
pub const POLICY: Policy = Policy {
    rules: Rules {
        choice_sorted: false,
        noul_true_first: false,
        noul_defaults: false,
        strict: true,
    },
    noul_ratings: Some(9),
    choice_variants: 2,
    sort_keys: true,
    bucket,
};

/// The bucket of a question of `n` options: up to 8 small, up to 26 mid, else large.
#[must_use]
pub fn bucket(n: usize) -> &'static str {
    match n {
        0..=8 => "small",
        9..=26 => "mid",
        _ => "large",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buckets_are_llama_cpps() {
        for (n, want) in [
            (1, "small"),
            (2, "small"),
            (8, "small"),
            (9, "mid"),
            (26, "mid"),
            (27, "large"),
            (255, "large"),
        ] {
            assert_eq!(bucket(n), want, "{n} options");
        }
    }

    /// A noul is read on at least two ratings, which the expected rating divides by.
    #[test]
    fn the_scale_has_two_ratings_or_more() {
        assert!(POLICY.noul_ratings.is_some_and(|n| n >= 2));
    }
}
