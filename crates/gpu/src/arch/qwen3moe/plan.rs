//! A layer's plan: the chain crate's (`chain::arch::qwen3moe::plan`), the one description of what a
//! layer of a qwen3moe-family chain mixes with and how its experts run, which the card bodies read
//! under the old paths of its items; and what this crate's kernels make of a plan's types — which
//! sites the gemv arm launches as one fused group of Q4_K launches, and which kernel family a
//! routed layer loads.
//!
//! The gemv arm's launch of each site type (at most `GEMV_COLS` rows; the wide arm runs the
//! grouped GEMM of each type):
//! - Q4_K in a fused group ([`qkv_fused`], [`o_fused`], [`input_fused`], [`out_fused`],
//!   [`gate_up_fused`], [`down_sel`]): `ProjKernels::enqueue_qkv` over q, k and v,
//!   `enqueue_o_resid` for an output projection with the residual add in its store, a Q6_K v its
//!   own gemv; the grouped GEMM `gemm_q4k` wide.
//! - Any other type, and a Q4_K one beside a type the group cannot hold: `crate::site`, each
//!   projection its own launch — Q3_K, Q4_K and Q6_K `Gpu::enqueue_gemv_*` (row-major, a
//!   token-major copy past one row), Q5_K the K-quant down `_sel` over one expert
//!   (`q5k_gemv_sel`), Q8_0 `q8_0_gemv` or `q8_0_gemv_mcol` and F32 `f32_tile_gemm`, both over
//!   the f32 rows; wide the grouped GEMM of a K-quant (`gemm_q{3,4,5,6}k`), `gemm_q8_0p` for
//!   Q8_0, the F32 tile for F32.

use crate::site::Form;
pub(super) use chain::arch::qwen3moe::plan::{
    DeltaPlan, DownEntry, FfnPlan, FfnRoute, Flash, GateUpEntry, GqaKind, GqaPlan, Kind35,
    LayerPlan, MixerPlan, Pairs, ResidualPlan, SharedPlan, SharedSites, SiteTy, down_entry,
    gate_up_entry, kind35, kinds35, moe_fits, q35, shared_entry,
};

// The head the chain plans a `GqaKind::Head128` layer at is the one this crate's head-128 kernels
// are built for.
const _: () = assert!(chain::arch::qwen3moe::plan::HEAD_128 as usize == crate::flash_gqa::HEAD);

/// Whether every one of `tys` is Q4_K: a fused group of Q4_K launches (one launch over several
/// matrices, or a projection with its neighbour folded in) runs only then.
fn all_q4k(tys: &[SiteTy]) -> bool {
    tys.iter().all(|t| *t == SiteTy::Q4K)
}

/// Whether `ty` is Q4_K or Q6_K: the two types the fused groups' lead matrix (a q·k·v's v, a delta
/// q·k·v) and the slots' down `_sel` launch.
fn q4k_or_q6k(ty: SiteTy) -> bool {
    matches!(ty, SiteTy::Q4K | SiteTy::Q6K)
}

/// The gemv arm's fused q·k·v of `g`: Q4_K q and k with a Q4_K or Q6_K v (a Q6_K v in its own
/// gemv); otherwise each projection launches alone.
pub(super) fn qkv_fused(g: &GqaPlan) -> bool {
    all_q4k(&[g.q_ty, g.k_ty]) && q4k_or_q6k(g.v_ty)
}

/// The gemv arm's output projection of `g` with the residual add folded in: a Q4_K `attn_output`.
pub(super) fn o_fused(g: &GqaPlan) -> bool {
    all_q4k(&[g.o_ty])
}

/// The gemv arm's two input launches of `d`: a Q4_K or Q6_K q·k·v projection (a Q6_K one in its
/// own gemv) and Q4_K `z`, β and α; otherwise each projection launches alone.
pub(super) fn input_fused(d: &DeltaPlan) -> bool {
    q4k_or_q6k(d.qkv_ty) && all_q4k(&[d.gate_ty, d.beta_ty, d.alpha_ty])
}

/// The gemv arm's output projection of `d` with the residual add folded in: a Q4_K `ssm_out`.
pub(super) fn out_fused(d: &DeltaPlan) -> bool {
    all_q4k(&[d.out_ty])
}

/// The gemv arm's gate·up·SwiGLU of `f` in one launch: a routed FFN's always (the card kernel
/// table's gate·up entry of its type), a dense FFN's Q4_K gate and up; otherwise each launches
/// alone and the SwiGLU after them.
pub(super) fn gate_up_fused(f: &FfnPlan) -> bool {
    match f.route {
        FfnRoute::Router { .. } => true,
        FfnRoute::Dense => all_q4k(&[f.gate_ty, f.up_ty]),
    }
}

/// The gemv arm's down of `f` as the slots' `_sel` over their ids: a routed FFN's always (the
/// table's down entry of its type), a dense FFN's Q4_K or Q6_K; otherwise a dense FFN's one slot a
/// token runs it as a plain projection (`dispatch::site_gemv`).
pub(super) fn down_sel(f: &FfnPlan) -> bool {
    match f.route {
        FfnRoute::Router { .. } => true,
        FfnRoute::Dense => q4k_or_q6k(f.down_ty),
    }
}

/// Whether a routed layer of `f` launches a kernel of `dispatch::FfnKernels`: the table's Q5_K
/// gate·up or down. The family's own Q4_K gate·up and the Q4_K and Q6_K downs launch from the
/// chain's base kernels.
pub(super) fn kq_sel(f: &FfnPlan) -> bool {
    matches!(f.route, FfnRoute::Router { .. })
        && (matches!(gate_up_entry(f.gate_ty, f.up_ty, ""), Ok(GateUpEntry::Q5k))
            || matches!(down_entry(f.down_ty, ""), Ok(DownEntry::Q5k)))
}

/// Whether the gate or the up of a shared expert kept apart reads the f32 normed rows on the gemv
/// arm (a Q8_0 or F32 site), which the norm folded into the router does not write.
pub(super) fn reads_f32(sh: &SharedSites) -> bool {
    [sh.gate_ty, sh.up_ty]
        .iter()
        .any(|t| t.reads() != Form::Q8x128)
}

/// Plans the unit tests of the dispatch and the ubatch share.
#[cfg(test)]
pub(super) mod fixtures {
    use super::{
        FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, LayerPlan, MixerPlan, ResidualPlan, SiteTy,
    };

    /// One layer of `kind` with the attention types `[q, k, v, o]` over the
    /// plain routed FFN of Q4_K stacks.
    pub(in crate::arch::qwen3moe) fn layer(kind: GqaKind, [q, k, v, o]: [SiteTy; 4]) -> LayerPlan {
        let name = |s: &str| s.to_string();
        LayerPlan {
            mixer: MixerPlan::Gqa(GqaPlan {
                kind,
                flash: Flash::Group,
                attn_norm: name("attn_norm"),
                attn_q: name("attn_q"),
                attn_k: name("attn_k"),
                attn_v: name("attn_v"),
                attn_q_norm: name("attn_q_norm"),
                attn_k_norm: name("attn_k_norm"),
                attn_output: name("attn_output"),
                q_ty: q,
                k_ty: k,
                v_ty: v,
                o_ty: o,
                window: None,
                gate: None,
                rope_freqs: None,
            }),
            ffn: FfnPlan {
                ffn_norm: name("ffn_norm"),
                route: FfnRoute::Router {
                    gate_inp: name("gate_inp"),
                    shared: None,
                },
                gate: name("gate"),
                up: name("up"),
                down: name("down"),
                gate_ty: SiteTy::Q4K,
                up_ty: SiteTy::Q4K,
                down_ty: SiteTy::Q4K,
            },
            residual: ResidualPlan::Plain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{FfnPlan, FfnRoute, SiteTy, kq_sel};

    /// A routed layer loads the K-quant `_sel` family for a Q5_K gate·up or down only: the kernels
    /// a Q4_K or Q6_K layer never loads. (Moved from the chain crate's plan tests, where the table
    /// entries it reads stay pinned.)
    #[test]
    fn a_routed_layer_needs_the_kquant_sel_family_for_a_q5k_stack_only() {
        use SiteTy::{Q4K, Q5K, Q6K};
        let sel = |gate_up: SiteTy, down: SiteTy| {
            kq_sel(&FfnPlan {
                ffn_norm: String::new(),
                route: FfnRoute::Router {
                    gate_inp: String::new(),
                    shared: None,
                },
                gate: String::new(),
                up: String::new(),
                down: String::new(),
                gate_ty: gate_up,
                up_ty: gate_up,
                down_ty: down,
            })
        };
        assert!(!sel(Q4K, Q4K));
        assert!(!sel(Q4K, Q6K));
        assert!(sel(Q5K, Q4K));
        assert!(sel(Q4K, Q5K));
        assert!(sel(Q5K, Q6K));
    }
}
