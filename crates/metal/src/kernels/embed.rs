//! The launch of the Q4_K and A4 embeddings.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::matvec::Quant;
use super::{Kernels, args_bytes, check_window_len, pack, word};

/// The threads in an embedding's threadgroup.
const GROUP_THREADS: usize = 256;

/// One embedding: `ids.len()` rows of a quantized table, dequantized to f32.
pub struct EmbedParams<'w> {
    /// The table's type. Q4_K, A4, A8, A4H and A8H have an embedding entry;
    /// Q6_K has none and is refused by name.
    pub quant: Quant,
    /// The table's bytes: `vocab * cols / values_per_block()` blocks.
    pub table: Window<'w, u8, MetalDevice>,
    /// The table's rows. An id at or past it raises the token-id site.
    pub vocab: usize,
    /// The table's columns, a multiple of the type's block values.
    pub cols: usize,
    /// The ids to embed.
    pub ids: Window<'w, u32, MetalDevice>,
    /// The output: `ids.len() * cols` values, row `t` the embedding of `ids[t]`.
    pub out: WindowMut<'w, f32, MetalDevice>,
    /// The fault word a raise lands in. Metal's own type, as
    /// [`NormOnLoad::fault`](super::NormOnLoad) is.
    pub fault: &'w MetalFaultWord,
    /// The layer a raise is recorded in: the caller's.
    pub layer: u32,
}

const WHAT: &str = "embed";

fn refuse(detail: impl Into<String>) -> DeviceError {
    super::range(WHAT, detail)
}

impl Kernels {
    /// Encode one embedding dispatch on `encoder`. Every window is checked
    /// against the shape first; nothing is encoded for a launch that is
    /// refused. An id past the table is not refused here: the ids live on the
    /// device, and the kernel raises the fault word for it.
    #[must_use = "a refused launch encoded nothing"]
    pub fn embed(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: EmbedParams<'_>,
    ) -> Result<(), DeviceError> {
        let count = p.ids.len().get();
        if p.vocab == 0 || p.cols == 0 || count == 0 {
            return Err(refuse(format!(
                "{} ids over {} rows of {} columns is empty",
                count, p.vocab, p.cols
            )));
        }
        if !p.cols.is_multiple_of(p.quant.values_per_block()) {
            return Err(refuse(format!(
                "{} columns are not whole blocks of {}",
                p.cols,
                p.quant.values_per_block()
            )));
        }
        let nblocks = p.cols / p.quant.values_per_block();
        let table_bytes = super::weight_bytes(p.quant, p.vocab, nblocks)
            .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.vocab, p.cols)))?;
        check_window_len(WHAT, "table", p.table.len(), table_bytes)?;
        let values = count
            .checked_mul(p.cols)
            .ok_or_else(|| refuse(format!("{count} ids of {} columns wrap", p.cols)))?;
        check_window_len(WHAT, "out", p.out.len(), values)?;
        let start = p.table.byte_range()?.start.get();
        super::check_weights_align(WHAT, "table", start, super::WeightAlign::Half)?;
        let args = args_bytes([
            word(WHAT, "the vocabulary", p.vocab)?,
            word(WHAT, "the block count", nblocks)?,
            p.layer,
            word(WHAT, "the id count", count)?,
        ]);
        // One thread per value, and a thread's index is a 32-bit word.
        // The columns are a multiple of the group, so the value count is too.
        word(WHAT, "the value count", values)?;
        let pipeline = match p.quant {
            Quant::Q4K => &self.embed_q4k,
            Quant::A4 => &self.embed_a4,
            Quant::A8 => &self.embed_a8,
            Quant::A4H => &self.embed_a4h,
            Quant::A8H => &self.embed_a8h,
            Quant::Q8_0 => &self.embed_q8_0,
            Quant::Q5K => &self.embed_q5k,
            // A quant with no embedding entry is refused by name: the entry
            // is absent, not defaulted.
            Quant::Q6K => {
                return Err(refuse("Q6_K has no embedding entry".to_string()));
            }
            Quant::F32 => {
                return Err(refuse("F32 has no embedding entry".to_string()));
            }
        };
        encoder.set_pipeline(pipeline);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(&p.table)?)?;
        slot.set(Binding::of(&p.ids)?)?;
        slot.set(Binding::of_mut(&p.out)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.fault.binding())?;
        slot.dispatch_groups(values / GROUP_THREADS, 1, GROUP_THREADS)
    }
}
