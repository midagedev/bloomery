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

use gguf::GgmlType;

use crate::{DeviceTensor, Gpu, GpuError};

pub mod dense;
pub mod head;
pub mod ring;
pub mod stage;

/// Q8_0 rows on the card in the form the Q8_0 row kernels take
/// (`q8f32::enqueue_q8_0_gemv_mcol`): the quants and the f16 scales apart.
pub struct Q8Rows {
    /// The quants, four to a word, row-major.
    pub qs: DeviceTensor<u32>,
    /// One f16 scale (its bits) per 32 values.
    pub d: DeviceTensor<u16>,
}

/// fc over the taps and the hidden norm after it: rows of
/// `taps.len() * hidden` floats in, `hidden` out.
pub struct FcWeights {
    /// fc's rows.
    pub fc: Q8Rows,
    /// The hidden norm's gain.
    pub norm: DeviceTensor<f32>,
    /// The hidden norm's epsilon.
    pub eps: f32,
}

/// A family's draft layers: everything between the embedded block rows and
/// the head's input, and the context rows' append. Each call that launches
/// is preceded by its stage call, outside any capture, so the per-width and
/// append graphs replay with this call's positions.
pub trait DraftLayers {
    /// The draft's hidden width: the rows [`DraftLayers::enqueue_block`]
    /// reads and writes.
    fn hidden(&self) -> usize;

    /// Stage the next [`DraftLayers::append_kv`]: `n` context rows at
    /// positions `first..first + n`. Never inside a capture.
    fn stage_kv(&mut self, gpu: &Gpu, first: u32, n: usize) -> Result<(), GpuError>;

    /// Append the staged context rows to every layer's KV. Each row of `x` is
    /// already fc over the taps, then the hidden norm,
    /// [`DraftLayers::hidden`] floats; the layer's own K/V projections and
    /// rotation are its own. Capturable.
    fn append_kv(&mut self, gpu: &Gpu, x: &DeviceBuffer<f32>, n: usize) -> Result<(), GpuError>;

    /// Stage the next [`DraftLayers::enqueue_block`]: `m` rows after `last`,
    /// the token at the target's position `pos`, over `ctx` visible context
    /// rows. Under [`EmbedSource::Target`] the layers embed the rows here
    /// from `last` and the mask id; under [`EmbedSource::OwnHost`] the owner
    /// has written them to the block's input. Never inside a capture.
    fn stage_block(
        &mut self,
        gpu: &Gpu,
        last: u32,
        pos: u32,
        m: usize,
        ctx: usize,
    ) -> Result<(), GpuError>;

    /// The staged block pass: `m` rows through every layer, each attending to
    /// the visible context rows and to the block's rows under the file's
    /// mask. Leaves the last layer's output in `x`, `m` rows, before the
    /// head's norm. Capturable: the owner records one graph per `m`.
    fn enqueue_block(
        &mut self,
        gpu: &Gpu,
        x: &mut DeviceBuffer<f32>,
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
    /// The target's embedding rows: the layers embed them in
    /// [`DraftLayers::stage_block`].
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

/// The target's output matrix as the draft's head borrows it: a copy on the
/// draft's card in the target file's type.
pub struct TargetHead {
    /// The matrix's words.
    pub w: DeviceTensor<u32>,
    /// Its type: the owner runs the types it has a row kernel for and refuses
    /// any other by name.
    pub ty: GgmlType,
}

/// Where the head's projection comes from.
pub enum HeadSource {
    /// The target's output matrix, borrowed (a draft over the target's full
    /// vocabulary).
    Target(TargetHead),
    /// The draft's own projection over its head's rows.
    Own(Q8Rows),
}

/// The common draft over a family's layers `L`.
pub struct DsparkBody<L: DraftLayers> {
    layers: PhantomData<L>,
}
