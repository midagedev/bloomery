//! The tensor names every family spells alike. A family's own names stay in
//! its `names` module — the GGUF strings are its architecture's — and a name
//! lives here only where the convention is one for all of them.

/// `blk.{l}.{stem}`.
#[must_use]
pub fn blk(l: usize, stem: &str) -> String {
    format!("blk.{l}.{stem}")
}

/// `token_embd.weight`.
#[must_use]
pub fn token_embd() -> String {
    "token_embd.weight".to_string()
}

/// `output_norm.weight`.
#[must_use]
pub fn output_norm() -> String {
    "output_norm.weight".to_string()
}

/// `output.weight`.
#[must_use]
pub fn output() -> String {
    "output.weight".to_string()
}

/// `rope_freqs.weight`: the rope's per-pair frequency factors, one f32 for each pair of the values
/// the rope turns, read only by a family whose rope divides by them.
#[must_use]
pub fn rope_freqs() -> String {
    crate::arch::classify::ROPE_FREQS.to_string()
}

// ------------------------------------------------ the per-layer names families share
//
// Every family that carries one of these tensors spells it alike (llama.cpp's `tn()` table), so
// a family re-exports the ones it reads and defines only the names its architecture adds.

/// `blk.{l}.attn_norm.weight` — the pre-attention RMS gain.
#[must_use]
pub fn attn_norm(l: usize) -> String {
    blk(l, "attn_norm.weight")
}

/// `blk.{l}.attn_sinks.weight` — one learned sink logit per query head.
#[must_use]
pub fn attn_sinks(l: usize) -> String {
    blk(l, "attn_sinks.weight")
}

/// `blk.{l}.attn_q.weight` — the query projection, every head.
#[must_use]
pub fn attn_q(l: usize) -> String {
    blk(l, "attn_q.weight")
}

/// `blk.{l}.attn_k.weight` — the key projection, every key head.
#[must_use]
pub fn attn_k(l: usize) -> String {
    blk(l, "attn_k.weight")
}

/// `blk.{l}.attn_v.weight` — the value projection, every value head.
#[must_use]
pub fn attn_v(l: usize) -> String {
    blk(l, "attn_v.weight")
}

/// `blk.{l}.attn_q_norm.weight` — the RMS gain every query head is normed with, one head wide.
#[must_use]
pub fn attn_q_norm(l: usize) -> String {
    blk(l, "attn_q_norm.weight")
}

/// `blk.{l}.attn_k_norm.weight` — the RMS gain every key head is normed with, one head wide.
#[must_use]
pub fn attn_k_norm(l: usize) -> String {
    blk(l, "attn_k_norm.weight")
}

/// `blk.{l}.attn_output.weight` — the projection from the heads' outputs back to the residual.
#[must_use]
pub fn attn_output(l: usize) -> String {
    blk(l, "attn_output.weight")
}

/// `blk.{l}.ffn_norm.weight` — the pre-FFN RMS gain.
#[must_use]
pub fn ffn_norm(l: usize) -> String {
    blk(l, "ffn_norm.weight")
}

/// `blk.{l}.ffn_gate.weight` — a dense FFN's gate.
#[must_use]
pub fn ffn_gate(l: usize) -> String {
    blk(l, "ffn_gate.weight")
}

/// `blk.{l}.ffn_up.weight` — a dense FFN's up.
#[must_use]
pub fn ffn_up(l: usize) -> String {
    blk(l, "ffn_up.weight")
}

/// `blk.{l}.ffn_down.weight` — a dense FFN's down.
#[must_use]
pub fn ffn_down(l: usize) -> String {
    blk(l, "ffn_down.weight")
}

/// `blk.{l}.ffn_gate_inp.weight` — the router, one logit per expert.
#[must_use]
pub fn ffn_gate_inp(l: usize) -> String {
    blk(l, "ffn_gate_inp.weight")
}

/// `blk.{l}.exp_probs_b.bias` — the router's selection bias, which steers the choice only.
#[must_use]
pub fn exp_probs_b(l: usize) -> String {
    blk(l, "exp_probs_b.bias")
}

/// `blk.{l}.ffn_gate_exps.weight` — the routed experts' gate stack.
#[must_use]
pub fn ffn_gate_exps(l: usize) -> String {
    blk(l, "ffn_gate_exps.weight")
}

/// `blk.{l}.ffn_up_exps.weight` — the routed experts' up stack.
#[must_use]
pub fn ffn_up_exps(l: usize) -> String {
    blk(l, "ffn_up_exps.weight")
}

/// `blk.{l}.ffn_down_exps.weight` — the routed experts' down stack.
#[must_use]
pub fn ffn_down_exps(l: usize) -> String {
    blk(l, "ffn_down_exps.weight")
}

/// `blk.{l}.ffn_gate_shexp.weight` — the shared expert's gate.
#[must_use]
pub fn ffn_gate_shexp(l: usize) -> String {
    blk(l, "ffn_gate_shexp.weight")
}

/// `blk.{l}.ffn_up_shexp.weight` — the shared expert's up.
#[must_use]
pub fn ffn_up_shexp(l: usize) -> String {
    blk(l, "ffn_up_shexp.weight")
}

/// `blk.{l}.ffn_down_shexp.weight` — the shared expert's down.
#[must_use]
pub fn ffn_down_shexp(l: usize) -> String {
    blk(l, "ffn_down_shexp.weight")
}

/// `blk.{l}.indexer.q_proj.weight` — the index-query projection.
#[must_use]
pub fn indexer_q_proj(l: usize) -> String {
    blk(l, "indexer.q_proj.weight")
}

/// `blk.{l}.indexer.k_proj.weight` — the index-key projection.
#[must_use]
pub fn indexer_k_proj(l: usize) -> String {
    blk(l, "indexer.k_proj.weight")
}

/// `blk.{l}.indexer.q_norm.weight` — the RMS gain on an index query.
#[must_use]
pub fn indexer_q_norm(l: usize) -> String {
    blk(l, "indexer.q_norm.weight")
}

/// `blk.{l}.indexer.k_norm.weight` — the RMS gain on an index key.
#[must_use]
pub fn indexer_k_norm(l: usize) -> String {
    blk(l, "indexer.k_norm.weight")
}

// ----------------------------------------------------- the KDA mixer's names ([`crate::arch::kda`])

/// `blk.{l}.ssm_conv1d_{part}.weight` — the conv taps of the `q`, `k` or `v` channels, one set per
/// channel.
#[must_use]
pub fn ssm_conv1d(l: usize, part: char) -> String {
    blk(l, &format!("ssm_conv1d_{part}.weight"))
}

/// `blk.{l}.ssm_f_a.weight` — the forget gate's down projection.
#[must_use]
pub fn ssm_f_a(l: usize) -> String {
    blk(l, "ssm_f_a.weight")
}

/// `blk.{l}.ssm_f_b.weight` — the forget gate's up projection.
#[must_use]
pub fn ssm_f_b(l: usize) -> String {
    blk(l, "ssm_f_b.weight")
}

/// `blk.{l}.ssm_g_a.weight` — the output gate's down projection.
#[must_use]
pub fn ssm_g_a(l: usize) -> String {
    blk(l, "ssm_g_a.weight")
}

/// `blk.{l}.ssm_g_b.weight` — the output gate's up projection.
#[must_use]
pub fn ssm_g_b(l: usize) -> String {
    blk(l, "ssm_g_b.weight")
}

/// `blk.{l}.ssm_beta.weight` — the delta rule's write strength, one per head.
#[must_use]
pub fn ssm_beta(l: usize) -> String {
    blk(l, "ssm_beta.weight")
}

/// `blk.{l}.ssm_a` — the decay's rate per head.
#[must_use]
pub fn ssm_a(l: usize) -> String {
    blk(l, "ssm_a")
}

/// `blk.{l}.ssm_dt.bias` — the decay's bias per channel.
#[must_use]
pub fn ssm_dt_bias(l: usize) -> String {
    blk(l, "ssm_dt.bias")
}

/// `blk.{l}.ssm_norm.weight` — the RMS gain of the gated output norm, one head wide.
#[must_use]
pub fn ssm_norm(l: usize) -> String {
    blk(l, "ssm_norm.weight")
}

/// Layer `l`'s routed experts' gate, up and down stack names, in stack order:
/// `blk.{l}.ffn_{gate,up,down}_exps.weight`, the GGUF convention of every MoE
/// family here.
#[must_use]
pub fn routed_stacks(l: usize) -> [String; 3] {
    [ffn_gate_exps(l), ffn_up_exps(l), ffn_down_exps(l)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{deepseek2, deepseek41, glm5next, mimo2, qwen3moe, qwen35moe};

    /// A family's own gate, up and down stack names of layer `l`.
    macro_rules! own {
        ($family:ident, $l:expr) => {
            [
                $family::names::ffn_gate_exps($l),
                $family::names::ffn_up_exps($l),
                $family::names::ffn_down_exps($l),
            ]
        };
    }

    /// The common names are each family's own: the six families that carry
    /// routed stacks spell a layer's gate, up and down alike, so a family's
    /// table takes them from here. Mutant: the common function's gate and up
    /// swapped — every family's arm catches it.
    #[test]
    fn the_routed_stack_names_are_every_familys_own() {
        for l in [0, 1, 9, 60] {
            let common = routed_stacks(l);
            assert_eq!(common, own!(glm5next, l), "glm5next, layer {l}");
            assert_eq!(common, own!(qwen35moe, l), "qwen35moe, layer {l}");
            assert_eq!(common, own!(qwen3moe, l), "qwen3moe, layer {l}");
            assert_eq!(common, own!(deepseek41, l), "deepseek41, layer {l}");
            assert_eq!(common, own!(deepseek2, l), "deepseek2, layer {l}");
            assert_eq!(common, own!(mimo2, l), "mimo2, layer {l}");
        }
        assert_eq!(
            routed_stacks(3),
            [
                "blk.3.ffn_gate_exps.weight",
                "blk.3.ffn_up_exps.weight",
                "blk.3.ffn_down_exps.weight"
            ]
        );
    }

    /// Each shared per-layer name is the GGUF spelling of its tensor (llama.cpp's `tn()` table, the
    /// strings every file carries), on the layer it is asked for. Mutant: two stems swapped, say
    /// `attn_k` and `attn_v` or the indexer's query and key projections, or `exp_probs_b` spelled
    /// as a weight.
    #[test]
    fn the_per_layer_names_are_the_gguf_spellings() {
        type Name = fn(usize) -> String;
        let table: [(Name, &str); 30] = [
            (attn_norm, "attn_norm.weight"),
            (attn_q, "attn_q.weight"),
            (attn_k, "attn_k.weight"),
            (attn_v, "attn_v.weight"),
            (attn_q_norm, "attn_q_norm.weight"),
            (attn_k_norm, "attn_k_norm.weight"),
            (attn_output, "attn_output.weight"),
            (ffn_norm, "ffn_norm.weight"),
            (ffn_gate, "ffn_gate.weight"),
            (ffn_up, "ffn_up.weight"),
            (ffn_down, "ffn_down.weight"),
            (ffn_gate_inp, "ffn_gate_inp.weight"),
            (exp_probs_b, "exp_probs_b.bias"),
            (ffn_gate_exps, "ffn_gate_exps.weight"),
            (ffn_up_exps, "ffn_up_exps.weight"),
            (ffn_down_exps, "ffn_down_exps.weight"),
            (ffn_gate_shexp, "ffn_gate_shexp.weight"),
            (ffn_up_shexp, "ffn_up_shexp.weight"),
            (indexer_q_proj, "indexer.q_proj.weight"),
            (indexer_k_proj, "indexer.k_proj.weight"),
            (indexer_q_norm, "indexer.q_norm.weight"),
            (indexer_k_norm, "indexer.k_norm.weight"),
            (ssm_f_a, "ssm_f_a.weight"),
            (ssm_f_b, "ssm_f_b.weight"),
            (ssm_g_a, "ssm_g_a.weight"),
            (ssm_g_b, "ssm_g_b.weight"),
            (ssm_beta, "ssm_beta.weight"),
            (ssm_a, "ssm_a"),
            (ssm_dt_bias, "ssm_dt.bias"),
            (ssm_norm, "ssm_norm.weight"),
        ];
        for (name, stem) in table {
            assert_eq!(name(7), format!("blk.7.{stem}"));
        }
        assert_eq!(ffn_down_shexp(7), "blk.7.ffn_down_shexp.weight");
        assert_eq!(ssm_conv1d(7, 'q'), "blk.7.ssm_conv1d_q.weight");
        assert_eq!(ssm_conv1d(7, 'v'), "blk.7.ssm_conv1d_v.weight");
    }

    /// The three model-level names and the block spelling are one function
    /// each. A family re-exports them; swapping two strings here changes
    /// every family's tensor name.
    #[test]
    fn the_model_level_names_and_the_block_spelling() {
        assert_eq!(token_embd(), "token_embd.weight");
        assert_eq!(output_norm(), "output_norm.weight");
        assert_eq!(output(), "output.weight");
        assert_eq!(blk(3, "attn_norm.weight"), "blk.3.attn_norm.weight");
    }
}

// ------------------------------------------------ latent attention and hyper-connections

/// `blk.{l}.attn_q_a.weight` — the query's down projection to its latent.
#[must_use]
pub fn attn_q_a(l: usize) -> String {
    blk(l, "attn_q_a.weight")
}

/// `blk.{l}.attn_q_a_norm.weight` — the RMS gain of the query latent.
#[must_use]
pub fn attn_q_a_norm(l: usize) -> String {
    blk(l, "attn_q_a_norm.weight")
}

/// `blk.{l}.attn_q_b.weight` — the query latent's up projection to the heads.
#[must_use]
pub fn attn_q_b(l: usize) -> String {
    blk(l, "attn_q_b.weight")
}

/// `blk.{l}.attn_kv_a_mqa.weight` — the key and value latent's down projection, with the key
/// values every head shares.
#[must_use]
pub fn attn_kv_a_mqa(l: usize) -> String {
    blk(l, "attn_kv_a_mqa.weight")
}

/// `blk.{l}.attn_kv_a_norm.weight` — the RMS gain of the key and value latent.
#[must_use]
pub fn attn_kv_a_norm(l: usize) -> String {
    blk(l, "attn_kv_a_norm.weight")
}

/// `blk.{l}.attn_k_b.weight` — the per-head key up projection a latent layer absorbs into its
/// query.
#[must_use]
pub fn attn_k_b(l: usize) -> String {
    blk(l, "attn_k_b.weight")
}

/// `blk.{l}.attn_v_b.weight` — the per-head value up projection a latent layer applies to its
/// attended latent.
#[must_use]
pub fn attn_v_b(l: usize) -> String {
    blk(l, "attn_v_b.weight")
}

/// The sub-layer a hyper-connection mix feeds: the mixer or the feed-forward block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sub {
    Attn,
    Ffn,
}

impl Sub {
    fn stem(self) -> &'static str {
        match self {
            Sub::Attn => "attn",
            Sub::Ffn => "ffn",
        }
    }
}

/// `blk.{l}.hc_{attn,ffn}_fn.weight` — the selector projection of the streams.
#[must_use]
pub fn hc_fn(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_fn.weight", sub.stem()))
}

/// `blk.{l}.hc_{attn,ffn}_scale.weight` — the mix's scalars.
#[must_use]
pub fn hc_scale(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_scale.weight", sub.stem()))
}

/// `blk.{l}.hc_{attn,ffn}_base.weight` — the mix's biases.
#[must_use]
pub fn hc_base(l: usize, sub: Sub) -> String {
    blk(l, &format!("hc_{}_base.weight", sub.stem()))
}
