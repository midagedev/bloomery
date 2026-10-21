//! The role tables' row, shared by the families whose tables name their
//! tensors by stem, and the blocks of rows every such family spells alike
//! ([`DENSE`], [`MOE`], [`HC`]). A family's own table holds the rows its
//! architecture alone carries, in its `roles` — the GGUF names are its
//! architecture's strings — and the walk over them is
//! [`classify`](crate::arch::classify)'s.

use crate::placement::Role;

/// A tensor a layer of some kind carries, by stem: its role, and whether
/// the file must carry it (llama.cpp creates it without
/// `TENSOR_NOT_REQUIRED`).
#[derive(Clone, Copy, Debug)]
pub struct Stem {
    pub(crate) name: &'static str,
    pub(crate) role: Role,
    pub(crate) required: bool,
}

/// A stem the file must carry.
pub(crate) const fn req(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: true,
    }
}

/// A stem the file may carry.
pub(crate) const fn opt(name: &'static str, role: Role) -> Stem {
    Stem {
        name,
        role,
        required: false,
    }
}

/// The names of `stems` the file must carry.
pub fn required(stems: &[Stem]) -> impl Iterator<Item = &'static str> + '_ {
    stems.iter().filter(|s| s.required).map(|s| s.name)
}

/// The stems of `parts` in order as one table of `N`; `N` is their count, held by assertion.
pub(crate) const fn join<const N: usize>(parts: &[&[Stem]]) -> [Stem; N] {
    let mut out = [parts[0][0]; N];
    let (mut n, mut p) = (0, 0);
    while p < parts.len() {
        let mut i = 0;
        while i < parts[p].len() {
            out[n] = parts[p][i];
            n += 1;
            i += 1;
        }
        p += 1;
    }
    assert!(n == N, "join: the parts' stems are not N");
    out
}

/// The hyper-connection tensors a trunk layer carries: the mix of each sublayer's input and
/// output from the streams, which every family of hyper-connections names alike.
pub const HC: &[Stem] = &[
    req("hc_attn_fn.weight", Role::HyperConnection),
    req("hc_attn_base.weight", Role::HyperConnection),
    req("hc_attn_scale.weight", Role::HyperConnection),
    req("hc_ffn_fn.weight", Role::HyperConnection),
    req("hc_ffn_base.weight", Role::HyperConnection),
    req("hc_ffn_scale.weight", Role::HyperConnection),
];

/// A dense layer's gate and down stems, which a placement reads to size the forms of the
/// feed-forward block ([`DENSE`]).
pub const DENSE_GATE: &str = "ffn_gate.weight";
pub const DENSE_DOWN: &str = "ffn_down.weight";

/// A dense layer's feed-forward block: the pre-FFN norm, the gate and up projections of the GLU
/// and the down projection.
pub const DENSE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req(DENSE_GATE, Role::DenseFfn),
    req("ffn_up.weight", Role::DenseFfn),
    req(DENSE_DOWN, Role::DenseFfn),
];

/// The router's selection bias, which the graph adds for the selection only and the loader reads
/// when the file carries it.
pub const SELECTION_BIAS: &str = "exp_probs_b.bias";

/// A routed layer's feed-forward block: the pre-FFN norm, the router with its optional selection
/// bias, and the routed stacks.
pub const MOE: &[Stem] = &[
    req("ffn_norm.weight", Role::FfnNorm),
    req("ffn_gate_inp.weight", Role::Router),
    opt(SELECTION_BIAS, Role::Router),
    req("ffn_gate_exps.weight", Role::RoutedExperts),
    req("ffn_up_exps.weight", Role::RoutedExperts),
    req("ffn_down_exps.weight", Role::RoutedExperts),
];
