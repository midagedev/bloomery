//! The tensor names every family spells alike. A family's own names stay in
//! its `names` module — the GGUF strings are its architecture's — and a name
//! lives here only where the convention is one for all of them.

/// Layer `l`'s routed experts' gate, up and down stack names, in stack order:
/// `blk.{l}.ffn_{gate,up,down}_exps.weight`, the GGUF convention of every MoE
/// family here.
#[must_use]
pub fn routed_stacks(l: usize) -> [String; 3] {
    ["gate", "up", "down"].map(|part| format!("blk.{l}.ffn_{part}_exps.weight"))
}

#[cfg(test)]
mod tests {
    use super::routed_stacks;
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
}
