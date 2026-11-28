//! The two kernels of Inkling 2's MoE leg beside the quantized mat-vecs: the
//! biased sigmoid router (`kernels/router_situ.metal`) and the situ activation
//! of a pair's two dots (`kernels/situ.metal`).
//!
//! The router is a variant of the registered one (`router.rs`): sigmoid
//! scores over 128 experts, a selection bias that steers the choice only, the
//! kept scores renormalized by their sum plus a fixed `1e-20` and scaled by the
//! route scale. It norms nothing on load: its input is the feed-forward mix's
//! output as it is.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{EXPERTS, HIDDEN, TOP_K, router_holds};

use super::{Kernels, WeightAlign, check_weights_align, check_window_len, pack, range, word};

const WHAT: &str = "router sigmoid";
const SITU: &str = "situ";

/// Rows a router threadgroup holds: eight SIMD groups.
const ROUTER_ROWS_PER_GROUP: usize = 8;
/// Threads in the router's threadgroup.
const ROUTER_THREADS: usize = ROUTER_ROWS_PER_GROUP * 32;
/// Threads in the situ combine's threadgroup.
const SITU_THREADS: usize = 256;

/// One router launch.
pub struct RouterSigmoidParams<'w> {
    /// The hidden width: the kernel reads the family's [`HIDDEN`].
    pub k: usize,
    /// The experts to keep, `1..=`[`TOP_K`].
    pub top_k: usize,
    /// The expert rows, `EXPERTS · k`, 16-byte aligned.
    pub rows: Window<'w, f32, MetalDevice>,
    /// The input, `k`, 16-byte aligned.
    pub x: Window<'w, f32, MetalDevice>,
    /// The selection bias, one a expert.
    pub bias: Window<'w, f32, MetalDevice>,
    /// The scores' partials, one a expert. The kernel owns them for the launch.
    pub partials: WindowMut<'w, f32, MetalDevice>,
    /// The ticket counter, one `u32`: zero before the launch, zero after.
    pub ticket: WindowMut<'w, u32, MetalDevice>,
    /// The sigmoid scores, one a expert.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// The kept ids, `top_k`.
    pub ids: WindowMut<'w, u32, MetalDevice>,
    /// The kept weights, `top_k`.
    pub weights: WindowMut<'w, f32, MetalDevice>,
    /// The route scale applied after the norm.
    pub scale: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

/// One router launch with the feed-forward's norm in front of it: the router's
/// own parameters, whose `x` is the row as it is (not normed), the norm's gain
/// and epsilon, and the row the norm writes (the experts' input).
pub struct RouterSigmoidNormParams<'w> {
    /// The router: `x` is the input before the norm.
    pub router: RouterSigmoidParams<'w>,
    /// The norm's gain, `k`, 16-byte aligned.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The normed row, `k`, 16-byte aligned: written by the launch.
    pub normed: WindowMut<'w, f32, MetalDevice>,
    /// The norm's epsilon, finite and positive.
    pub eps: f32,
}

/// One situ combine launch over `n` values.
pub struct SituParams<'w> {
    /// Values.
    pub n: usize,
    /// The gate rows' dot, `n`.
    pub gate: Window<'w, f32, MetalDevice>,
    /// The up rows' dot, `n`.
    pub up: Window<'w, f32, MetalDevice>,
    /// The activation, `n`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The gate term's beta, finite and positive.
    pub gate_beta: f32,
    /// The up term's beta, finite and positive.
    pub value_beta: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

/// The router's shape and windows, refused by name.
fn check_router(p: &RouterSigmoidParams<'_>) -> Result<(), DeviceError> {
    router_holds(p.k).map_err(|e| refuse(e.to_string()))?;
    if p.k != HIDDEN {
        return Err(refuse(format!("hidden {}, the kernel reads {HIDDEN}", p.k)));
    }
    if !(1..=TOP_K).contains(&p.top_k) {
        return Err(refuse(format!(
            "top-k is {}, the kernel takes 1 to {TOP_K}",
            p.top_k
        )));
    }
    if !p.scale.is_finite() {
        return Err(refuse(format!("route scale {} is not finite", p.scale)));
    }
    check_window_len(WHAT, "rows", p.rows.len(), EXPERTS * p.k)?;
    check_window_len(WHAT, "x", p.x.len(), p.k)?;
    check_window_len(WHAT, "bias", p.bias.len(), EXPERTS)?;
    check_window_len(WHAT, "partials", p.partials.len(), EXPERTS)?;
    check_window_len(WHAT, "ticket", p.ticket.len(), 1)?;
    check_window_len(WHAT, "logits", p.logits.len(), EXPERTS)?;
    check_window_len(WHAT, "ids", p.ids.len(), p.top_k)?;
    check_window_len(WHAT, "weights", p.weights.len(), p.top_k)?;
    for (name, start) in [
        ("rows", p.rows.byte_range()?.start.get()),
        ("x", p.x.byte_range()?.start.get()),
    ] {
        check_weights_align(WHAT, name, start, WeightAlign::Float4)?;
    }
    Ok(())
}

/// The router's argument words: top-k, the layer and the route scale.
fn router_words(p: &RouterSigmoidParams<'_>) -> Result<[[u8; 4]; 3], DeviceError> {
    Ok([word(WHAT, "top-k", p.top_k)?, p.layer, p.scale.to_bits()].map(u32::to_ne_bytes))
}

/// Bind the router's partials, ticket and outputs, in the order both entries
/// take them.
fn bind_selection(
    slot: &mut Bindings<'_, '_>,
    p: &RouterSigmoidParams<'_>,
) -> Result<(), DeviceError> {
    slot.set(Binding::of_mut(&p.partials)?)?;
    slot.set(Binding::of_mut(&p.ticket)?)?;
    slot.set(Binding::of_mut(&p.logits)?)?;
    slot.set(Binding::of_mut(&p.ids)?)?;
    slot.set(Binding::of_mut(&p.weights)?)
}

impl Kernels {
    /// Encode the router: one dispatch, which also selects (the last
    /// threadgroup). The shape and every window are checked first; a refused
    /// launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn router_sigmoid(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouterSigmoidParams<'_>,
    ) -> Result<(), DeviceError> {
        check_router(&p)?;
        let args = router_words(&p)?;
        encoder.set_pipeline(&self.ink2.router_sigmoid);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.rows)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.bias)?)?;
        bind_selection(&mut slot, &p)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(EXPERTS / ROUTER_ROWS_PER_GROUP, 1, ROUTER_THREADS)
    }

    /// Encode the router with the feed-forward's norm in front of it: one
    /// dispatch, which norms the row in every threadgroup, stores it, and also
    /// selects. The shape and every window are checked first; a refused launch
    /// encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn router_sigmoid_norm(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: RouterSigmoidNormParams<'_>,
    ) -> Result<(), DeviceError> {
        let r = &p.router;
        check_router(r)?;
        if !(p.eps.is_finite() && p.eps > 0.0) {
            return Err(refuse(format!(
                "norm epsilon {} is not finite and positive",
                p.eps
            )));
        }
        check_window_len(WHAT, "gain", p.gain.len(), r.k)?;
        check_window_len(WHAT, "normed", p.normed.len(), r.k)?;
        for (name, start) in [
            ("gain", p.gain.byte_range()?.start.get()),
            ("normed", p.normed.byte_range()?.start.get()),
        ] {
            check_weights_align(WHAT, name, start, WeightAlign::Float4)?;
        }
        let words = router_words(r)?;
        let args = [words[0], words[1], words[2], p.eps.to_bits().to_ne_bytes()];
        encoder.set_pipeline(&self.ink2.router_sigmoid_norm);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&r.rows)?)?;
        slot.set(Binding::of(&r.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.normed)?)?;
        slot.set(Binding::of(&r.bias)?)?;
        bind_selection(&mut slot, r)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(r.faults.binding())?;
        slot.dispatch_groups(EXPERTS / ROUTER_ROWS_PER_GROUP, 1, ROUTER_THREADS)
    }

    /// Encode the situ combine: one dispatch, a thread a value. The shape and
    /// every window are checked first; a refused launch encodes nothing. A beta
    /// that is not finite and positive is refused here by name; the kernel
    /// raises the product site for the same defect it is handed some other way.
    #[must_use = "a refused launch encoded nothing"]
    pub fn pair_situ(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: SituParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.n == 0 {
            return Err(range(SITU, "0 values"));
        }
        for (name, beta) in [("gate beta", p.gate_beta), ("value beta", p.value_beta)] {
            if !(beta.is_finite() && beta > 0.0) {
                return Err(range(
                    SITU,
                    format!("{name} {beta} is not finite and positive"),
                ));
            }
        }
        check_window_len(SITU, "gate", p.gate.len(), p.n)?;
        check_window_len(SITU, "up", p.up.len(), p.n)?;
        check_window_len(SITU, "y", p.y.len(), p.n)?;
        let args = [
            word(SITU, "n", p.n)?,
            p.layer,
            p.gate_beta.to_bits(),
            p.value_beta.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.pair_situ);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.gate)?)?;
        slot.set(Binding::of(&p.up)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.n.div_ceil(SITU_THREADS), 1, SITU_THREADS)
    }
}
