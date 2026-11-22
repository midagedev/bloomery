//! The output head of Inkling 2 over f32 weights: the output norm, the
//! projection of every kept vocabulary row and the soft cap in one dispatch
//! (`kernels/head_f32.metal`), then the pick over the stored logits, the
//! quantized heads' own (`logits_argmax`, `head.rs`).
//!
//! The row count is the kept prefix of the file's vocabulary: the weights
//! window holds exactly those rows, the launch of a load-time view.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::HIDDEN;

use super::head::{HEAD_THREADS, MAX_ROWS};
use super::{
    Kernels, WeightAlign, args_bytes, check_weights_align, check_window_len, pack, range, word,
};

const WHAT: &str = "head f32";

/// Rows a logits threadgroup holds: one a SIMD group, eight of them.
const ROWS_PER_GROUP: usize = 8;
/// Threads in the logits threadgroup.
const THREADS: usize = ROWS_PER_GROUP * 32;

/// One f32 head launch.
pub struct HeadF32Params<'w> {
    /// The kept vocabulary rows: any positive count up to the pick's 32-bit
    /// loop, no multiple of a threadgroup's rows needed.
    pub rows: usize,
    /// The hidden width: the kernel stages the family's [`HIDDEN`].
    pub k: usize,
    /// The weights, `rows · k` f32, 16-byte aligned.
    pub weights: Window<'w, f32, MetalDevice>,
    /// The input row, `k`.
    pub x: Window<'w, f32, MetalDevice>,
    /// The output norm's gain, `k`.
    pub gain: Window<'w, f32, MetalDevice>,
    /// The norm's epsilon.
    pub eps: f32,
    /// The soft cap `cap · tanh(logit / cap)`, or none: a cap that is not
    /// finite and positive is refused by name.
    pub softcap: Option<f32>,
    /// The logits, `rows`, 16-byte aligned: the first dispatch's output and
    /// the pick's input.
    pub logits: WindowMut<'w, f32, MetalDevice>,
    /// The winning row, one `u32`.
    pub step_id: WindowMut<'w, u32, MetalDevice>,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the head: the logits dispatch, a barrier that closes its group,
    /// and the pick. The shape and every window are checked first; a refused
    /// launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn head_f32(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: HeadF32Params<'_>,
    ) -> Result<(), DeviceError> {
        if p.rows == 0 {
            return Err(refuse("the row count is 0"));
        }
        if p.rows > MAX_ROWS {
            return Err(refuse(format!(
                "the row count {} passes the pick's {MAX_ROWS}",
                p.rows
            )));
        }
        if p.k != HIDDEN {
            return Err(refuse(format!("K is {}, the kernel stages {HIDDEN}", p.k)));
        }
        if !(p.eps.is_finite() && p.eps > 0.0) {
            return Err(refuse(format!("eps {} is not finite and positive", p.eps)));
        }
        let (cap, capped) = match p.softcap {
            Some(c) if c.is_finite() && c > 0.0 => (c, 1u32),
            Some(c) => {
                return Err(refuse(format!("soft cap {c} is not finite and positive")));
            }
            None => (1.0, 0),
        };
        let weights = p
            .rows
            .checked_mul(p.k)
            .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.rows, p.k)))?;
        check_window_len(WHAT, "weights", p.weights.len(), weights)?;
        check_window_len(WHAT, "x", p.x.len(), p.k)?;
        check_window_len(WHAT, "gain", p.gain.len(), p.k)?;
        check_window_len(WHAT, "logits", p.logits.len(), p.rows)?;
        check_window_len(WHAT, "step id", p.step_id.len(), 1)?;
        // The rows are read as float4s, the pick reads the logits as float4s.
        check_weights_align(
            WHAT,
            "weights",
            p.weights.byte_range()?.start.get(),
            WeightAlign::Float4,
        )?;
        check_weights_align(
            WHAT,
            "logits",
            p.logits.byte_range()?.start.get(),
            WeightAlign::Float4,
        )?;
        let rows = word(WHAT, "the row count", p.rows)?;
        let args = args_bytes([
            rows,
            word(WHAT, "k", p.k)?,
            p.layer,
            p.eps.to_bits(),
            cap.to_bits(),
            capped,
        ]);
        encoder.set_pipeline(&self.ink2.head_f32);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.weights)?)?;
        slot.set(Binding::of(&p.x)?)?;
        slot.set(Binding::of(&p.gain)?)?;
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.rows.div_ceil(ROWS_PER_GROUP), 1, THREADS)?;
        encoder.memory_barrier();
        let pick = args_bytes([rows, p.layer]);
        encoder.set_pipeline(&self.head_pick);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of_mut(&p.logits)?)?;
        slot.set(Binding::of_mut(&p.step_id)?)?;
        slot.set(p.faults.binding())?;
        slot.set(Binding::bytes(pack(&pick))?)?;
        slot.dispatch_groups(1, 1, HEAD_THREADS)
    }
}
