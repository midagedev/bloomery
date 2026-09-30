//! Qwen3.8's side of adaptive expert residency ([`crate::host::swap`]): the
//! model's [`FileStacks`] for the common file source
//! ([`crate::host::swap_source::FileSwap`]) — [`Qwen38Stacks`], its gate, up
//! and down — with the machine's live delay and deadline.
//!
//! **Parts.** An expert is three parts in stack order: the gate and up (Q4_K,
//! `geo::FF` rows of `geo::HIDDEN` values each) and the down (Q5_1,
//! `geo::HIDDEN` rows of `geo::FF`). On the card each part of slot `s` is the
//! file's bytes of that expert at byte `s · part` of its layer's stack, which
//! the placed load uploads as file bytes in slot order
//! ([`place::card_routed`] keeps all three stacks in `CardFormat::KQuant`,
//! and `super::card38` reads them exactly so). Nothing converts: a staged
//! part is its slot's bytes as they are, so [`Qwen38Stacks::open`] hands back
//! no convert step — and the r8 sidecar, which exists for Q3_K stacks alone
//! (`qdot::repack_q3k_r8`), is undefined input here, refused by name.

use std::sync::Arc;
use std::time::Duration;

use crate::GpuError;
use crate::host::swap_source::{Convert, FileStacks};
use cuda_core::CudaContext;
use gguf::quant::GgmlType;

/// Passes from the boundary that makes a flip to the one it lands at. The
/// victims are host-resident (the churn pool), so a flip waits on no NVMe
/// read, only on its staging and its copy: a planning pass makes at most the
/// rule's `cap` flips ([`runtime::swaprule::SwapParams::mid`]), each one
/// expert's memcpy into the pinned ring on the staging thread and one H2D
/// copy on the machine's copy stream. One expert is a gate and an up and a
/// down — Q4_K bytes for the two `FF`-row stacks and Q5_1 bytes for the
/// `HIDDEN`-row one — so a planning pass's staging is `cap` experts of those
/// bytes over the ring's rate beside the host leg, far inside two passes'
/// wall of decode steps at the model's width [derived]; the staging runs in
/// the host leg's wait window, and a late copy only makes the engine stream
/// wait at the landing.
pub const LIVE_DELAY: u64 = 2;

/// The bound on every host wait of the machine. A boundary runs after the
/// last launched pass's host service has returned: before a launch, or ahead
/// of the next pass after a step's kept row and before its readback
/// (`GpuModel::run_tokens`), while the engine stream still runs that step's
/// last kernels and its head. So the engine stream need not be empty, but
/// nothing on it waits on this thread: every host word its waits read was
/// written by a service that has returned. The one host wait a boundary
/// makes is for a landing flip's staging. The flip's job `n` was issued
/// `LIVE_DELAY` = 2 boundaries back (a live delay of 1 or more, which the
/// machine holds), and its staging waits only for its ring slot's previous
/// copy, an earlier job issued at or before that boundary. That copy
/// waits on the copy stream for its own staging (an earlier, due job: the
/// same argument) and for the boundary event recorded when it was issued,
/// which precedes the last launched pass in the engine stream's order, so
/// the stream reaches it with no further host action — the waits before it
/// are copies whose staging the host already waited for at their own
/// landing, and the host words of passes already served. No wait can
/// close a cycle through the engine stream, and a wait is for the staging
/// thread and the copy stream alone: at most a planning pass's `cap`
/// experts ([`runtime::swaprule::SwapParams::mid`]), far inside this bound.
/// A late copy landing at a boundary made ahead delays the step's readback,
/// which the engine stream orders after that boundary's wait.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// The K-quant types of Qwen3.8's three stacks, in stack order: the Q4_K
/// gate and up and the Q5_1 down its kernels read ([`place::card_routed`]).
const TYPES: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::Q4_K, GgmlType::Q5_1];

/// Qwen3.8's [`FileStacks`] for the common file source: each layer's gate, up
/// and down by name, and nothing to convert — the card holds the file's
/// bytes, so a staged part is its slot's. A model with no shape contract of
/// its own on the stacks; the load's own checks
/// ([`crate::host::swap_source::FileSwap::new`], `super::card38`) hold the
/// parts and rows.
pub struct Qwen38Stacks;

impl FileStacks for Qwen38Stacks {
    fn names(&self, layer: usize) -> Vec<String> {
        vec![
            model::arch::qwen35moe::names::ffn_gate_exps(layer),
            model::arch::qwen35moe::names::ffn_up_exps(layer),
            model::arch::qwen35moe::names::ffn_down_exps(layer),
        ]
    }

    /// Every layer's stacks hold the same types.
    fn types(&self, _layer: usize) -> &[GgmlType] {
        &TYPES
    }

    fn open(
        &self,
        _dims: &[u64],
        _parts: &[usize],
        sidecar: bool,
        _ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError> {
        if sidecar {
            return Err(GpuError::shape(
                "Qwen38Stacks::open",
                "an r8 sidecar beside a qwen4exp file: the sidecar holds Q3_K \
                 stacks (qdot::repack_q3k_r8), and this model's routed gate and up \
                 are Q4_K and its down Q5_1 — undefined input, not a conversion",
            ));
        }
        Ok(None)
    }
}
