//! `kimi_k3` — Kimi K3 on the card: the step ([`body`]) over the delta layers ([`kda`]), the
//! gated latent layers (the core Raon-1 shares, `crate::arch::gated_mla`) and the routed block
//! ([`moe`]), the cross-layer residual's mix ([`crate::attn_res`]) between them, and the state
//! book of the step ([`state`]), the pass of up to eight rows ([`Pass`], [`Body::rows`]) and the
//! placed body's prompt pass of up to [`SHAPES`]' widest rows ([`PassShape`],
//! [`crate::GpuModel::prompt_placed`]), each row the step's bits. The model description is `model::arch::kimi_k3`.

mod body;
mod kda;
pub mod moe;
mod seat;
mod state;
mod tier;

pub use body::pass::{CHUNK, PassShape, PassTail, SHAPES, pass_bytes};
pub use body::placed::Placed;
pub use body::rows::{Pass, Tail};
pub use body::{Body, TAP_STAGES};
pub use tier::NODE_FAMILY;

#[cfg(test)]
mod tests {
    use gguf::GgmlType;
    use gguf::arch::kimi_k3 as table;

    use super::{body::DENSE_TYPES, kda::TYPES};
    use crate::site::{PROJECTED, SiteTy};

    /// A card site list holds every type the table lists for the stems the site serves, but
    /// BF16 (which the load widens), and each of its types is a launch of the common dense site.
    /// Mutant: a table type dropped from a list, or a type no dense launch runs added to one.
    #[test]
    fn the_card_site_lists_hold_the_tables_trunk_types() {
        let lists: [(&[SiteTy], &[&str]); 2] = [
            (
                DENSE_TYPES,
                &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"],
            ),
            (
                TYPES,
                &[
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "ssm_g.weight",
                    "ssm_beta.weight",
                    "ssm_f_a.weight",
                    "ssm_f_b.weight",
                    "attn_output.weight",
                ],
            ),
        ];
        for (list, stems) in lists {
            for ty in list {
                assert!(PROJECTED.contains(ty), "{ty} has no dense launch");
            }
            for stem in stems {
                let listed =
                    table::allowed_types(stem).unwrap_or_else(|| panic!("{stem} is no stem"));
                for ty in listed {
                    if !list.iter().any(|s| s.ggml() == *ty) {
                        assert_eq!(*ty, GgmlType::BF16, "{stem}: {ty} is listed and not run");
                    }
                }
            }
        }
    }
}
