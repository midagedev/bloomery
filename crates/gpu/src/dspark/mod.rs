//! The DSpark draft on the card, common to every family that drafts with
//! one: [`DsparkBody`] runs fc over the target's tapped rows, keeps the
//! context ring's slots, stages a block, and runs the head chain (norm,
//! projection, the Markov loop, the draft-to-target id map) with one graph
//! per width. The layers between the embedded block and the head are a
//! family's ([`DraftLayers`]): V4.1's live in `bloomery-gpu-deepseek41`, the
//! dense Qwen3-shaped layer in [`dense`]. The draft's geometry is
//! [`bloomery_runtime::dspark::DsparkShape`].

use std::marker::PhantomData;

use cuda_core::DeviceBuffer;

use crate::{Gpu, GpuError};

pub mod dense;
pub mod head;
pub mod ring;
pub mod stage;

/// A family's draft layers: everything between the embedded block rows and
/// the head's input, and the context rows' append.
pub trait DraftLayers {
    /// The draft's hidden width: the rows [`DraftLayers::enqueue_block`]
    /// reads and writes.
    fn hidden(&self) -> usize;

    /// Append `n` context rows to every layer's KV at positions
    /// `first..first + n`. Each row of `x` is already fc over the taps, then
    /// the hidden norm, [`DraftLayers::hidden`] floats; the layer's own K/V
    /// projections and rotation are its own.
    fn append_kv(
        &mut self,
        gpu: &Gpu,
        x: &DeviceBuffer<f32>,
        first: u32,
        n: usize,
    ) -> Result<(), GpuError>;

    /// The block pass: `m` embedded rows of `x` (row 0 the last token's,
    /// rows 1.. the mask token's) at positions `pos..pos + m`, through every
    /// layer. Each row attends to the context ring inside the draft's window
    /// and to every row of the block. Leaves the last layer's output in `x`,
    /// `m` rows, before the head's norm. Capturable: the owner records one
    /// graph per `m`.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
        pos: u32,
        m: usize,
    ) -> Result<(), GpuError>;

    /// Forget every context row: a new generation.
    fn reset(&mut self, gpu: &Gpu) -> Result<(), GpuError>;

    /// Card bytes these layers hold at `ctx` positions.
    fn card_bytes(&self, ctx: usize) -> u64;
}

/// Where the block's embedded rows come from.
#[derive(Debug)]
pub enum EmbedSource {
    /// The target's embedding rows.
    Target,
    /// The draft file's own table, kept on the host; the owner gathers the
    /// rows a pass needs there and widens them to f32.
    OwnHost(HostEmbed),
}

/// A draft's own embedding table on the host: `rows * cols` bf16 values,
/// row-major.
#[derive(Debug)]
pub struct HostEmbed {
    /// Ids the table holds (the target's vocabulary).
    pub rows: usize,
    /// The hidden width.
    pub cols: usize,
    /// The table's bf16 bits.
    pub bf16: Vec<u16>,
}

/// Where the head's projection comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadSource {
    /// The target's output matrix, borrowed (a draft over the target's full
    /// vocabulary).
    Target,
    /// The draft's own projection over its head's rows.
    Own,
}

/// The common draft over a family's layers `L`.
pub struct DsparkBody<L: DraftLayers> {
    layers: PhantomData<L>,
}
