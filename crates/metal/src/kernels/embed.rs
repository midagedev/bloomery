//! The launch of the Q4_K embedding.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, ComputeEncoder};
use crate::fault::MetalFaultWord;

use super::geometry::BLOCK_VALUES;
use super::matvec::Quant;
use super::{Kernels, args_bytes, check_window_len, word};

/// The threads in an embedding's threadgroup.
const GROUP_THREADS: usize = 256;

/// One embedding: `ids.len()` rows of a Q4_K table, dequantized to f32.
pub struct EmbedParams<'w> {
    /// The table's bytes: `vocab * cols / 256` Q4_K blocks.
    pub table: Window<'w, u8, MetalDevice>,
    /// The table's rows. An id at or past it raises the token-id site.
    pub vocab: usize,
    /// The table's columns, a multiple of 256.
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

fn refuse(detail: String) -> DeviceError {
    DeviceError::Range {
        what: "embed",
        detail,
    }
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
        if !p.cols.is_multiple_of(BLOCK_VALUES) {
            return Err(refuse(format!(
                "{} columns are not whole blocks of {BLOCK_VALUES}",
                p.cols
            )));
        }
        let nblocks = p.cols / BLOCK_VALUES;
        let table_bytes = p
            .vocab
            .checked_mul(nblocks)
            .and_then(|blocks| blocks.checked_mul(Quant::Q4K.block_bytes()))
            .ok_or_else(|| refuse(format!("{} rows of {} columns wrap", p.vocab, p.cols)))?;
        check_window_len("embed", "table", p.table.len(), table_bytes)?;
        let values = count
            .checked_mul(p.cols)
            .ok_or_else(|| refuse(format!("{count} ids of {} columns wrap", p.cols)))?;
        check_window_len("embed", "out", p.out.len(), values)?;
        let start = p.table.byte_range()?.start.get();
        if !start.is_multiple_of(2) {
            return Err(refuse(format!(
                "the table starts {start} bytes into its buffer: the kernel reads 16-bit words"
            )));
        }
        let args = args_bytes([
            word("embed", "the vocabulary", p.vocab)?,
            word("embed", "the block count", nblocks)?,
            p.layer,
            word("embed", "the id count", count)?,
        ]);
        // One thread per value, and a thread's index is a 32-bit word.
        word("embed", "the value count", values)?;
        encoder.set_pipeline(&self.embed);
        encoder.set_binding(0, Binding::of(&p.table)?)?;
        encoder.set_binding(1, Binding::of(&p.ids)?)?;
        encoder.set_binding(2, Binding::of_mut(&p.out)?)?;
        encoder.set_binding(3, Binding::bytes(&args)?)?;
        encoder.set_binding(4, p.fault.binding())?;
        encoder.dispatch(values, GROUP_THREADS)
    }
}
