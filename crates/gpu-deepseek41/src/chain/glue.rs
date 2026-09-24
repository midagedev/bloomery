//! What the V4.1 step runs outside its sub-layers: the embedding broadcast
//! into the hyper-connection streams, the engram, the streams' collapse
//! before the head, and the head with its argmax. See [`super`] for the piece
//! contract.
//!
//! The piece, in step order, one token a step:
//! - [`Glue::enqueue_embed`], one launch at the step's start: the token's
//!   embedding row, bf16 in the step image, widened to f32 into every stream
//!   and into layer 0's attention input. The first sub-layer's `pre` is
//!   one-hot on stream 0, so that fold is the row itself and does not run.
//! - [`Glue::enqueue_engram_kv_at`], three launches per engram site, none of
//!   which reads a stream: the site's gathered rows (Q8_0 bytes in the image)
//!   dequantized, `engram_wkv` over them (`q8_0_gemv`, f32 activations), and
//!   the key norm. They depend on the token alone and are first read by the
//!   site layer's engram step, so the step hands them ([`EngramKv`]) to the
//!   MoE sub-layer of the layer before the site, whose host-leg shadow runs
//!   them ([`ShadowWork`]).
//! - [`Glue::enqueue_engram`], two launches at each engram layer: the gate
//!   over the streams the previous ffn's HC_POST left (that ffn ends with
//!   HC_POST alone), then the fold of the gated streams that the layer's
//!   attention reads, by the previous sub-layer's `pre` (the lag).
//! - [`Glue::enqueue_head`], after the last layer: the fold of the last
//!   ffn's streams by that ffn's `pre` (`hc_out`), written into the head's
//!   input, then the head's own launches.
//!
//! [`StepRows`] is the host half: the rows a step's image carries — each
//! token's embedding row from the file, and each site's rows, hashed from the
//! token's n-gram by the engram crate and read out of the site's table by a
//! helper thread ([`RowsLevers::helper`]), so the calling thread never touches
//! the table's mapping.
//!
//! Numbers: the bf16 widening and the rows' dequantization (an f16 scale
//! times an 8-bit code) are exact; everything else is an op this piece
//! composes and does not change.
//!
//! The resident weights come in at each enqueue, not at [`Glue::new`]: the
//! body owns the pieces and its model owns the weights beside it, so a piece
//! that kept a borrow of them would borrow its owner's sibling. Every name is
//! resolved at `new`; an enqueue looks the names up (at capture, once per
//! graph) and allocates nothing.

use std::sync::Arc;
use std::time::Instant;

use bloomery_gpu::head::Head;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, launch_u32};
use cuda_core::{DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f32_f16x2_hi, cvt_f32_f16x2_lo, cvt_f32x2_bf16x2};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use engram::Engram;
use engram::prefetch::{FillMode, HelperOptions, Prefetcher, caller_sibling};
use gguf::quant::GgmlType;
use gguf::{Split, TensorInfo};
use model::arch::deepseek41::hparams::Hparams;
use model::arch::deepseek41::names;
use model::arch::deepseek41::plan::StepPlan;

use crate::chain::ffn::ShadowWork;
use crate::engram_gate::{EngramGateKernels, GateArgs, KeyNormArgs, ROW};
use crate::hc::{HC_MIX, HC_STREAMS, HcKernels};
use crate::params::ImageLayout;

const WHAT: &str = "deepseek41 Glue";

/// Threads of every glue launch.
const THREADS: u32 = 256;

/// Values of one Q8_0 block, and its bytes: an f16 scale, then the codes.
const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 34;

// ---------------------------------------------------------------- kernels

#[cuda_module]
pub(crate) mod glue_kernels {
    use super::*;

    /// The embedding broadcast: thread `i` widens image word `at + i` — two
    /// bf16, the low half first — into values `2i` and `2i + 1` of each of
    /// the `hc` streams (`[s][2·half]`) and of `input`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            params.len() >= at + half,
            streams.len() >= hc * 2 * half,
            input.len() >= 2 * half
        )
    )]
    pub fn ds41_glue_embed(
        params: &[u32],
        at: u32,
        half: u32,
        hc: u32,
        mut streams: DisjointSlice<f32>,
        mut input: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let half = half as usize;
        if i >= half {
            return;
        }
        // SAFETY: i < half, so at + i < at + half <= params.len() by the
        // launch contract.
        let word = unsafe { *params.get_unchecked(at as usize + i) };
        let (lo, hi) = cvt_f32x2_bf16x2(word);
        let n = 2 * half;
        for s in 0..hc as usize {
            let b = s * n + 2 * i;
            // SAFETY: s < hc and 2i + 1 < n, so b + 1 < hc·n <= streams.len()
            // by the launch contract; the two values are this thread's alone.
            unsafe {
                *streams.get_unchecked_mut(b) = lo;
                *streams.get_unchecked_mut(b + 1) = hi;
            }
        }
        // SAFETY: 2i + 1 < n <= input.len() by the launch contract; the two
        // values are this thread's alone.
        unsafe {
            *input.get_unchecked_mut(2 * i) = lo;
            *input.get_unchecked_mut(2 * i + 1) = hi;
        }
    }

    /// One site's gathered rows, dequantized: thread `i` is value `v = i %
    /// key_len` of row `r = i / key_len`, code `v % 32` of block `v / 32`.
    /// The row's bytes start at byte `4·(at + r·row_words)` of the image; a
    /// block is its f16 scale, then 32 codes. `y[i] = scale · code`, exact.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            params.len() >= at + rows * row_words,
            y.len() >= rows * key_len
        )
    )]
    pub fn ds41_glue_engram_rows(
        params: &[u32],
        at: u32,
        rows: u32,
        row_words: u32,
        key_len: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let key_len = key_len as usize;
        if i >= rows as usize * key_len {
            return;
        }
        let (r, v) = (i / key_len, i % key_len);
        let base = at as usize + r * row_words as usize;
        let block = Q8_0_BYTES * (v / Q8_0_BLOCK);
        let code = block + 2 + v % Q8_0_BLOCK;
        // SAFETY: `Glue::new` sets 4·row_words = key_len/32·34, the bytes of
        // a row, so byte `code` of row r lies in words base .. base +
        // row_words, and r < rows gives base + row_words <= at +
        // rows·row_words <= params.len() by the launch contract.
        let (dw, cw) = unsafe {
            (
                *params.get_unchecked(base + block / 4),
                *params.get_unchecked(base + code / 4),
            )
        };
        // A block starts on an even byte: its scale is one half of a word.
        let d = if block.is_multiple_of(4) {
            cvt_f32_f16x2_lo(dw)
        } else {
            cvt_f32_f16x2_hi(dw)
        };
        let q = (cw >> (8 * (code % 4))) as u8 as i8;
        // SAFETY: i < rows·key_len <= y.len() by the launch contract; one
        // thread per value.
        unsafe {
            *y.get_unchecked_mut(i) = d * f32::from(q);
        }
    }
}

// -------------------------------------------------------------- the piece

/// One engram site: its layer, its weights' names, where its rows lie in the
/// image, and per row the scratch its launches leave for the layer's gate.
struct EngramSite {
    layer: usize,
    wkv: String,
    gain_k: String,
    gain_q: String,
    /// Image word of the site's first row.
    at: u32,
    /// Per row ([`Glue::with_rows`]): a pass whose rows run one layer apart
    /// enqueues the other row's token-only work between a row's and its
    /// gate.
    bufs: Vec<SiteRow>,
}

/// One row's scratch at a site.
struct SiteRow {
    /// The rows dequantized: the projection's input.
    x: DeviceBuffer<f32>,
    /// `engram_wkv`'s output: `hc` keys, then the value.
    kv: DeviceBuffer<f32>,
    /// The normalized keys.
    kn: DeviceBuffer<f32>,
    /// The gates, one per stream.
    gate: DeviceBuffer<f32>,
}

impl SiteRow {
    fn device_bytes(&self) -> usize {
        self.x.num_bytes() + self.kv.num_bytes() + self.kn.num_bytes() + self.gate.num_bytes()
    }
}

/// A site's scratch as the gate reads it back.
pub struct SiteBuffers<'a> {
    /// The rows dequantized from the image.
    pub rows: &'a DeviceBuffer<f32>,
    /// `engram_wkv`'s output.
    pub kv: &'a DeviceBuffer<f32>,
    /// The gates of the last engram step at the site.
    pub gate: &'a DeviceBuffer<f32>,
}

/// [`Glue::enqueue_engram`]'s buffers.
pub struct EngramStep<'a> {
    /// The streams the previous ffn's HC_POST left.
    pub streams: &'a DeviceBuffer<f32>,
    /// The previous sub-layer's HC_PRE result ([`HC_MIX`] values): its `pre`
    /// folds the gated streams.
    pub pre: &'a DeviceBuffer<f32>,
    /// The gated streams.
    pub out: &'a mut DeviceBuffer<f32>,
    /// The layer's attention input: the fold of `out`.
    pub input: &'a mut DeviceBuffer<f32>,
}

/// The token-only work of the engram site at `layer`
/// ([`Glue::enqueue_engram_kv_at`]) as shadow work: what the step hands the
/// MoE sub-layer of the layer before the site.
pub struct EngramKv<'a> {
    pub glue: &'a mut Glue,
    /// The resident weights.
    pub w: &'a Weights,
    /// The step image's device copy.
    pub params: &'a DeviceBuffer<u32>,
    /// The site's layer.
    pub layer: usize,
    /// The row whose scratch it writes.
    pub row: usize,
}

impl ShadowWork for EngramKv<'_> {
    fn enqueue(&mut self, gpu: &Gpu) -> Result<(), GpuError> {
        self.glue
            .enqueue_engram_kv_of(gpu, self.w, self.params, self.layer, self.row)
    }
}

/// The glue piece: see the module comment.
pub struct Glue {
    module: glue_kernels::LoadedModule,
    hc: HcKernels,
    engram: EngramGateKernels,
    n_embd: usize,
    eps: f32,
    /// Image word of the embedding row, and the words it takes.
    embd_at: u32,
    half: u32,
    /// Rows a site gathers, their values and their words.
    rows: u32,
    key_len: u32,
    row_words: u32,
    sites: Vec<EngramSite>,
}

impl Glue {
    /// The piece for the model `hp` describes, reading the step image laid
    /// out by `layout`: its launch geometry, every name its launches read,
    /// and each engram site's scratch. The kernels hold a stream row of
    /// [`ROW`] values and four streams; a step runs one token, and the image
    /// must carry each site's rows as Q8_0. One row of site scratch.
    pub fn new(gpu: &Gpu, hp: &Hparams, layout: &ImageLayout) -> Result<Glue, GpuError> {
        Glue::with_rows(gpu, hp, layout, 1)
    }

    /// [`Glue::new`] with `rows` rows of site scratch, for a pass whose rows
    /// run one layer apart.
    pub fn with_rows(
        gpu: &Gpu,
        hp: &Hparams,
        layout: &ImageLayout,
        n_rows: usize,
    ) -> Result<Glue, GpuError> {
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        if n_rows == 0 {
            return Err(refuse("a piece of no rows".to_string()));
        }
        let dims = layout.dims();
        if hp.n_embd != ROW || dims.n_embd != ROW || hp.hc.streams != HC_STREAMS {
            return Err(refuse(format!(
                "a model of rows of {} values in {} streams and an image of rows of {}; the \
                 kernels take {ROW} and {HC_STREAMS}",
                hp.n_embd, hp.hc.streams, dims.n_embd
            )));
        }
        if dims.tokens != 1 {
            return Err(refuse(format!(
                "an image of {} tokens; the piece runs one token a step",
                dims.tokens
            )));
        }
        let en = &hp.engram;
        let rows = en.rows_per_token();
        let row_bytes = en.key_length / Q8_0_BLOCK * Q8_0_BYTES;
        if en.key_length == 0
            || !en.key_length.is_multiple_of(Q8_0_BLOCK)
            || !row_bytes.is_multiple_of(4)
            || dims.engram_bytes != en.layer_ids.len() * rows * row_bytes
        {
            return Err(refuse(format!(
                "engram rows of {} values, {} bytes a token in the image: not {} sites of {rows} \
                 Q8_0 rows in whole words",
                en.key_length,
                dims.engram_bytes,
                en.layer_ids.len()
            )));
        }
        let (embd_at, half, engram_at) = row_offsets(layout)?;
        let row_words = row_bytes / 4;
        let site_words = rows * row_words;
        let stream = gpu.stream();
        let mut sites = Vec::with_capacity(en.layer_ids.len());
        for (s, &layer) in en.layer_ids.iter().enumerate() {
            if hp.layers.get(layer).and_then(|k| k.engram) != Some(s) {
                return Err(refuse(format!(
                    "engram site {s} names layer {layer}, whose kind does not name the site"
                )));
            }
            let at = launch_u32(WHAT, "site rows", engram_at as usize + s * site_words)?;
            sites.push(EngramSite {
                layer,
                wkv: names::engram_wkv(layer),
                gain_k: names::engram_k(layer),
                gain_q: names::engram_q(layer),
                at,
                bufs: (0..n_rows)
                    .map(|_| {
                        Ok(SiteRow {
                            x: DeviceBuffer::zeroed(stream, rows * en.key_length)?,
                            kv: DeviceBuffer::zeroed(stream, (HC_STREAMS + 1) * ROW)?,
                            kn: DeviceBuffer::zeroed(stream, HC_STREAMS * ROW)?,
                            gate: DeviceBuffer::zeroed(stream, HC_STREAMS)?,
                        })
                    })
                    .collect::<Result<Vec<_>, GpuError>>()?,
            });
        }
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launchers check its launch contracts.
        let module = unsafe { glue_kernels::load(gpu.context())? };
        Ok(Glue {
            module,
            hc: HcKernels::load(gpu.context())?,
            engram: EngramGateKernels::load(gpu.context())?,
            n_embd: hp.n_embd,
            eps: hp.rms_eps,
            embd_at,
            half,
            rows: launch_u32(WHAT, "rows", rows)?,
            key_len: launch_u32(WHAT, "key_length", en.key_length)?,
            row_words: launch_u32(WHAT, "row words", row_words)?,
            sites,
        })
    }

    /// Device bytes of the piece's scratch.
    #[must_use]
    pub fn device_bytes(&self) -> usize {
        self.sites
            .iter()
            .flat_map(|s| &s.bufs)
            .map(SiteRow::device_bytes)
            .sum()
    }

    /// Device bytes of one row's scratch, every site's.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.sites
            .iter()
            .filter_map(|s| s.bufs.first())
            .map(SiteRow::device_bytes)
            .sum()
    }

    /// The layers that carry an engram site, in site order.
    pub fn engram_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.sites.iter().map(|s| s.layer)
    }

    /// Row 0's scratch of the site at `layer`; `None` for a layer without
    /// one.
    #[must_use]
    pub fn site_buffers(&self, layer: usize) -> Option<SiteBuffers<'_>> {
        self.site_buffers_of(layer, 0)
    }

    /// Row `row`'s scratch of the site at `layer`; `None` for a layer
    /// without one or a row the piece does not hold.
    #[must_use]
    pub fn site_buffers_of(&self, layer: usize, row: usize) -> Option<SiteBuffers<'_>> {
        self.sites
            .iter()
            .find(|s| s.layer == layer)
            .and_then(|s| s.bufs.get(row))
            .map(|b| SiteBuffers {
                rows: &b.x,
                kv: &b.kv,
                gate: &b.gate,
            })
    }

    /// Enqueue the embedding broadcast: the row the image `params` carries
    /// into every stream of `streams` and into `input`, layer 0's attention
    /// input. One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed(
        &self,
        gpu: &Gpu,
        params: &DeviceBuffer<u32>,
        streams: &mut DeviceBuffer<f32>,
        input: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_embed";
        let n = self.n_embd;
        need(
            WHAT,
            "params",
            params.len(),
            self.embd_at as usize + self.half as usize,
        )?;
        need(WHAT, "streams", streams.len(), HC_STREAMS * n)?;
        need(WHAT, "input", input.len(), n)?;
        let grid = self.half.div_ceil(THREADS);
        let prep = self
            .module
            .prepare_ds41_glue_embed(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_glue_embed(
            gpu.stream(),
            &prep,
            params,
            self.embd_at,
            self.half,
            HC_STREAMS as u32,
            streams,
            input,
        )?;
        Ok(())
    }

    /// Enqueue the token-only work of the engram site at `layer`: its rows
    /// from the image `params` dequantized, `engram_wkv` over them, and the
    /// key norm. Three launches, reading no stream. Asynchronous,
    /// allocation-free, capturable. Row 0's scratch.
    pub fn enqueue_engram_kv_at(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        params: &DeviceBuffer<u32>,
        layer: usize,
    ) -> Result<(), GpuError> {
        self.enqueue_engram_kv_of(gpu, w, params, layer, 0)
    }

    /// [`Glue::enqueue_engram_kv_at`] into row `row`'s scratch, from that
    /// row's image `params`.
    pub fn enqueue_engram_kv_of(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        params: &DeviceBuffer<u32>,
        layer: usize,
        row: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_engram_kv_at";
        let stream = gpu.stream();
        let values = self.rows * self.key_len;
        let grid = values.div_ceil(THREADS);
        let SiteRef {
            at,
            wkv,
            gain_k,
            row: site,
            ..
        } = site_row(&mut self.sites, layer, row, WHAT)?;
        need(
            WHAT,
            "params",
            params.len(),
            at as usize + self.rows as usize * self.row_words as usize,
        )?;
        let prep = self
            .module
            .prepare_ds41_glue_engram_rows(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module.ds41_glue_engram_rows(
            stream,
            &prep,
            params,
            at,
            self.rows,
            self.row_words,
            self.key_len,
            &mut site.x,
        )?;
        let (qs, d) = q8_0(w, wkv, site.x.len(), site.kv.len())?;
        gpu.q8f32()
            .enqueue_q8_0_gemv(stream, qs, d, &site.x, 1, &mut site.kv)?;
        self.engram.enqueue_key_norm(
            stream,
            KeyNormArgs {
                kv: &site.kv,
                gain: gain(w, gain_k, HC_STREAMS * ROW)?,
                eps: self.eps,
                hc: HC_STREAMS,
                m: 1,
                kn: &mut site.kn,
            },
        )?;
        Ok(())
    }

    /// Enqueue the engram step at `layer`: the gate over `a.streams` into
    /// `a.out`, then the fold of `a.out` by `a.pre` into `a.input`. Two
    /// launches. The site's token-only work ([`Glue::enqueue_engram_kv_at`])
    /// must precede it on the stream.
    /// Asynchronous, allocation-free, capturable. Row 0's scratch.
    pub fn enqueue_engram(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        a: EngramStep<'_>,
    ) -> Result<(), GpuError> {
        self.enqueue_engram_of(gpu, w, layer, 0, a)
    }

    /// [`Glue::enqueue_engram`] on row `row`'s scratch, which that row's
    /// token-only work wrote.
    pub fn enqueue_engram_of(
        &mut self,
        gpu: &Gpu,
        w: &Weights,
        layer: usize,
        row: usize,
        a: EngramStep<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "Glue::enqueue_engram";
        let stream = gpu.stream();
        need(WHAT, "pre", a.pre.len(), HC_MIX)?;
        let SiteRef {
            gain_q, row: site, ..
        } = site_row(&mut self.sites, layer, row, WHAT)?;
        self.engram.enqueue_gate(
            stream,
            GateArgs {
                x: a.streams,
                kn: &site.kn,
                kv: &site.kv,
                gain: gain(w, gain_q, HC_STREAMS * ROW)?,
                eps: self.eps,
                hc: HC_STREAMS,
                m: 1,
                out: &mut *a.out,
                gate: &mut site.gate,
            },
        )?;
        self.hc
            .enqueue_fold(stream, a.out, a.pre, self.n_embd, 1, a.input)
    }

    /// Enqueue the head end: `streams`, the last ffn's HC_POST output, folded
    /// by that ffn's HC_PRE result `pre` into the head's input (`hc_out`),
    /// then the head. One launch and the head's. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_head(
        &self,
        gpu: &Gpu,
        w: &Weights,
        streams: &DeviceBuffer<f32>,
        pre: &DeviceBuffer<f32>,
        head: &mut Head,
    ) -> Result<(), GpuError> {
        if head.hidden() != self.n_embd {
            return Err(GpuError::Shape {
                what: "Glue::enqueue_head",
                detail: format!(
                    "a head of {} values, the streams hold rows of {}",
                    head.hidden(),
                    self.n_embd
                ),
            });
        }
        self.hc
            .enqueue_fold(gpu.stream(), streams, pre, self.n_embd, 1, head.input_mut())?;
        head.enqueue(gpu, w)
    }
}

/// The site at `layer` as one enqueue reads it: its image word, its
/// weights' names and row `row`'s scratch.
struct SiteRef<'s> {
    at: u32,
    wkv: &'s str,
    gain_k: &'s str,
    gain_q: &'s str,
    row: &'s mut SiteRow,
}

/// The site at `layer` of `sites`, with row `row`'s scratch; `what` names
/// the caller in the error for a layer without a site or a row the piece
/// does not hold.
fn site_row<'s>(
    sites: &'s mut [EngramSite],
    layer: usize,
    row: usize,
    what: &'static str,
) -> Result<SiteRef<'s>, GpuError> {
    let EngramSite {
        at,
        wkv,
        gain_k,
        gain_q,
        bufs,
        ..
    } = sites
        .iter_mut()
        .find(|s| s.layer == layer)
        .ok_or_else(|| GpuError::Shape {
            what,
            detail: format!("layer {layer} carries no engram site"),
        })?;
    let n = bufs.len();
    let row = bufs.get_mut(row).ok_or_else(|| GpuError::Shape {
        what,
        detail: format!("row {row} of a piece of {n} rows"),
    })?;
    Ok(SiteRef {
        at: *at,
        wkv,
        gain_k,
        gain_q,
        row,
    })
}

/// Token 0's embedding row and engram rows in `layout`'s image, as image
/// words: the row's first word and its word count, and the first engram
/// word. Read through the layout's own view of an image whose word `i`
/// holds `i`, so the offsets are the layout's and are written nowhere else.
fn row_offsets(layout: &ImageLayout) -> Result<(u32, u32, u32), GpuError> {
    let index = (0..layout.words())
        .map(|i| launch_u32(WHAT, "image word", i))
        .collect::<Result<Vec<u32>, _>>()?;
    let view = layout.view(&index)?;
    let (embd, engram) = (view.embd(0), view.engram(0));
    match (embd.first(), engram.first()) {
        (Some(&at), Some(&engram_at)) => {
            Ok((at, launch_u32(WHAT, "row words", embd.len())?, engram_at))
        }
        _ => Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "an image of {} embedding words and {} engram words a token",
                embd.len(),
                engram.len()
            ),
        }),
    }
}

/// The resident Q8_0 planes of `name`, refused unless it projects `k`
/// values onto `rows` rows.
fn q8_0<'w>(
    w: &'w Weights,
    name: &str,
    k: usize,
    rows: usize,
) -> Result<(&'w DeviceTensor<u32>, &'w DeviceTensor<u16>), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { qs, d, k: wk }) if *wk == k && d.rows() == rows => Ok((qs, d)),
        Some(DevWeight::Q8_0 { d, k: wk, .. }) => Err(GpuError::Shape {
            what: WHAT,
            detail: format!(
                "{name} is {} rows of {wk} values, want {rows} of {k}",
                d.rows()
            ),
        }),
        found => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: if found.is_some() { "Q8_0" } else { "resident" },
        }),
    }
}

/// The resident f32 values of `name`, refused unless there are `len`.
fn gain<'w>(w: &'w Weights, name: &str, len: usize) -> Result<&'w DeviceBuffer<f32>, GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w: t, .. }) if t.buf().len() == len => Ok(t.buf()),
        Some(DevWeight::F32 { w: t, .. }) => Err(GpuError::Shape {
            what: WHAT,
            detail: format!("{name} holds {} values, want {len}", t.buf().len()),
        }),
        found => Err(GpuError::Tensor {
            what: WHAT,
            name: name.to_string(),
            need: if found.is_some() { "F32" } else { "resident" },
        }),
    }
}

/// Refuse a buffer `name` of `len` elements shorter than `want`.
fn need(what: &'static str, name: &str, len: usize, want: usize) -> Result<(), GpuError> {
    if len < want {
        return Err(GpuError::Shape {
            what,
            detail: format!("{name}.len() {len} < {want}"),
        });
    }
    Ok(())
}

// ---------------------------------------------------------- the host half

/// The levers of [`StepRows`], read once when it opens
/// ([`RowsLevers::from_env`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowsLevers {
    /// The engram rows are read by a helper thread: the calling thread hashes
    /// the ids, hands them over, reads the embedding row meanwhile and waits.
    /// Off (`BLOOMERY_ENGRAM_HELPER=0`), the calling thread reads them itself.
    pub helper: bool,
    /// Keep [`EngramStats`] (`BLOOMERY_STEP_STATS=1`). Off, a fill reads no
    /// clock and makes no classifying syscall.
    pub stats: bool,
}

impl RowsLevers {
    /// The levers from the environment.
    #[must_use]
    pub fn from_env() -> RowsLevers {
        RowsLevers {
            helper: !std::env::var("BLOOMERY_ENGRAM_HELPER").is_ok_and(|v| v == "0"),
            stats: std::env::var("BLOOMERY_STEP_STATS").is_ok_and(|v| v == "1"),
        }
    }
}

/// What the fills since [`StepRows`] opened read, when it keeps stats
/// ([`RowsLevers::stats`]); all zero otherwise. A row is warm when every page
/// of it was in the page cache before the read, cold otherwise
/// ([`engram::Site::resident_rows`]); both are counted on either path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngramStats {
    pub warm: u64,
    pub cold: u64,
    /// Rows the calling thread read itself: the direct path's.
    pub direct: u64,
    /// The calling thread's time in the engram read: the wait on the helper,
    /// or the direct path's copy.
    pub wait_ns: u64,
    /// The helper's own time on the rows it served: advise, fill and copy.
    pub helper_ns: u64,
    /// The classification's time ([`engram::Site::resident_rows`]): on the
    /// helper before its advise, so inside `wait_ns`, or on the calling thread
    /// before the direct copy, outside it. The stats' own tax.
    pub classify_ns: u64,
}

/// The rows one step's image carries ([`crate::params::StepImage::build`]'s
/// `embd` and `engram`), for a step of a fixed token count. A token's
/// embedding row is the file's `token_embd` row, bf16. A site's rows are the
/// ones the engram crate's hash names for the token's n-gram — the plan's
/// window mapped as the port maps it: the current token, then each older one
/// up to the first missing, the pad id from there on — read out of the
/// site's table by the helper ([`Prefetcher`], advise every row first, then
/// copy), or on the calling thread with the helper off. Either way the bytes
/// are the table's.
pub struct StepRows {
    engram: Arc<Engram>,
    /// The helper, when [`RowsLevers::helper`] is on.
    helper: Option<Prefetcher>,
    /// Kept when [`RowsLevers::stats`] is on.
    stats: Option<EngramStats>,
    /// `token_embd`'s shard and header.
    embd: (usize, TensorInfo),
    n_embd: usize,
    n_vocab: usize,
    /// Rows a site gathers per token, and bytes a row.
    n_cols: usize,
    row_bytes: usize,
    tokens: usize,
    ctx: Vec<u64>,
    /// `[token][site][n_cols]`.
    ids: Vec<u32>,
    embd_rows: Vec<u16>,
    engram_rows: Vec<u8>,
}

impl StepRows {
    /// The host half for steps of `tokens` tokens of `file`, whose
    /// hyperparameters are `hp`: the embedding table checked, the engram
    /// tables opened (headers only) and checked against `hp`, and every
    /// buffer a fill writes allocated.
    ///
    /// The levers come from the environment ([`RowsLevers::from_env`]); a
    /// helper is pinned to the SMT sibling of the calling thread's core when
    /// that thread is pinned ([`caller_sibling`]) — the caller is the pool's
    /// dispatcher, which waits while the helper reads, so that core is idle
    /// then and no worker's core is shared — and floats otherwise.
    pub fn open(file: &Split, hp: &Hparams, tokens: usize) -> Result<StepRows, GpuError> {
        StepRows::open_with(file, hp, tokens, RowsLevers::from_env())
    }

    /// [`StepRows::open`] with the levers named.
    pub fn open_with(
        file: &Split,
        hp: &Hparams,
        tokens: usize,
        levers: RowsLevers,
    ) -> Result<StepRows, GpuError> {
        const WHAT: &str = "StepRows::open";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let name = names::token_embd();
        let (shard, info) = file.find(&name).ok_or_else(|| GpuError::Tensor {
            what: WHAT,
            name: name.clone(),
            need: "in the file",
        })?;
        if info.ty != GgmlType::BF16 || info.dims != [hp.n_embd as u64, hp.n_vocab as u64] {
            return Err(refuse(format!(
                "{name} is {} {:?}, want bf16 [{}, {}]",
                info.ty, info.dims, hp.n_embd, hp.n_vocab
            )));
        }
        let paths = (0..file.shard_count()).filter_map(|i| file.shard_path(i));
        let engram = Engram::open(paths).map_err(|e| GpuError::plan(WHAT, e))?;
        let (hash, en) = (engram.hash(), &hp.engram);
        let row_bytes = en.key_length / Q8_0_BLOCK * Q8_0_BYTES;
        let layers_agree = hash.layer_ids().len() == en.layer_ids.len()
            && hash
                .layer_ids()
                .iter()
                .zip(&en.layer_ids)
                .all(|(&a, &b)| usize::try_from(a).is_ok_and(|a| a == b));
        if !layers_agree
            || hash.n_gram() != en.max_ngram
            || hash.n_cols() != en.rows_per_token()
            || usize::try_from(hash.key_length()).ok() != Some(en.key_length)
            || engram
                .sites()
                .iter()
                .any(|s| usize::try_from(s.row_bytes()).ok() != Some(row_bytes))
        {
            return Err(refuse(format!(
                "the engram tables hash sites {:?} of {}-grams into {} rows of {} values; the \
                 file's hyperparameters say {:?}, {}, {} and {} in Q8_0 rows of {row_bytes} bytes",
                hash.layer_ids(),
                hash.n_gram(),
                hash.n_cols(),
                hash.key_length(),
                en.layer_ids,
                en.max_ngram,
                en.rows_per_token(),
                en.key_length
            )));
        }
        let n_cols = hash.n_cols();
        let sites = engram.sites().len();
        let engram = Arc::new(engram);
        let helper = levers
            .helper
            .then(|| {
                Prefetcher::with_options(
                    Arc::clone(&engram),
                    &vec![n_cols; sites],
                    HelperOptions {
                        mode: FillMode::Touch,
                        cpu: caller_sibling(),
                        classify: levers.stats,
                    },
                )
            })
            .transpose()
            .map_err(|e| GpuError::plan(WHAT, e))?;
        Ok(StepRows {
            helper,
            stats: levers.stats.then(EngramStats::default),
            embd: (shard, info.clone()),
            n_embd: hp.n_embd,
            n_vocab: hp.n_vocab,
            n_cols,
            row_bytes,
            tokens,
            ctx: vec![0; en.max_ngram],
            ids: vec![0; tokens * sites * n_cols],
            embd_rows: vec![0; tokens * hp.n_embd],
            engram_rows: vec![0; tokens * sites * n_cols * row_bytes],
            engram,
        })
    }

    /// Bytes of one engram row.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Fill the rows of `plan`'s step from `file`, the file `open` read.
    pub fn fill(&mut self, file: &Split, plan: &StepPlan) -> Result<(), GpuError> {
        const WHAT: &str = "StepRows::fill";
        let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
        let n_gram = self.ctx.len();
        if plan.tokens.len() != self.tokens || plan.engram_window.len() != self.tokens * n_gram {
            return Err(refuse(format!(
                "a plan of {} tokens and {} n-gram slots; the rows hold {} tokens of {n_gram}",
                plan.tokens.len(),
                plan.engram_window.len(),
                self.tokens
            )));
        }
        if self.helper.is_none() {
            embd_rows_into(
                &self.embd,
                [self.n_embd, self.n_vocab],
                &mut self.embd_rows,
                file,
                plan,
            )?;
        }
        let site_bytes = self.n_cols * self.row_bytes;
        let token_ids = self.engram.sites().len() * self.n_cols;
        for (k, window) in plan.engram_window.chunks_exact(n_gram).enumerate() {
            let ids = &mut self.ids[k * token_ids..][..token_ids];
            let bytes = &mut self.engram_rows[k * token_ids * self.row_bytes..]
                [..token_ids * self.row_bytes];
            let hash = self.engram.hash();
            let mut blocked = false;
            for (slot, &token) in self.ctx.iter_mut().zip(window) {
                blocked |= token.is_none();
                *slot = match token {
                    Some(t) if !blocked => hash.map_token(t),
                    _ => hash.pad_id(),
                };
            }
            for (s, site_ids) in ids.chunks_exact_mut(self.n_cols).enumerate() {
                hash.rows_into(s, &self.ctx, site_ids)
                    .map_err(|e| GpuError::plan(WHAT, e))?;
            }
            match self.helper.as_mut() {
                Some(helper) => {
                    helper
                        .submit(ids.chunks_exact(self.n_cols))
                        .map_err(|e| GpuError::plan(WHAT, e))?;
                    // Only the first token overlaps the embedding rows; a
                    // step of one token, the engine's, is all of it.
                    if k == 0 {
                        let embd = embd_rows_into(
                            &self.embd,
                            [self.n_embd, self.n_vocab],
                            &mut self.embd_rows,
                            file,
                            plan,
                        );
                        // The helper holds a job: take it back before an
                        // error leaves, or the next fill's submit is refused.
                        if let Err(e) = embd {
                            let _ = helper.wait();
                            return Err(e);
                        }
                    }
                    let t0 = self.stats.map(|_| Instant::now());
                    helper.wait().map_err(|e| GpuError::plan(WHAT, e))?;
                    let wait_ns = t0.map_or(0, |t| t.elapsed().as_nanos() as u64);
                    let filled = helper.filled();
                    if filled.len() != bytes.len() {
                        return Err(refuse(format!(
                            "the helper filled {} bytes of a token's {}",
                            filled.len(),
                            bytes.len()
                        )));
                    }
                    bytes.copy_from_slice(filled);
                    if let Some(st) = self.stats.as_mut() {
                        let t = helper.last();
                        st.warm += t.resident_rows;
                        st.cold += t.rows - t.resident_rows;
                        st.wait_ns += wait_ns;
                        st.helper_ns += t.submit_ns + t.fill_ns + t.copy_ns;
                        st.classify_ns += t.classify_ns;
                    }
                }
                None => {
                    for ((site, site_ids), out) in self
                        .engram
                        .sites()
                        .iter()
                        .zip(ids.chunks_exact(self.n_cols))
                        .zip(bytes.chunks_exact_mut(site_bytes))
                    {
                        let t0 = match self.stats.as_mut() {
                            Some(st) => {
                                let tc = Instant::now();
                                let warm = site
                                    .resident_rows(site_ids)
                                    .map_err(|e| GpuError::plan(WHAT, e))?;
                                st.classify_ns += tc.elapsed().as_nanos() as u64;
                                st.warm += warm;
                                st.cold += site_ids.len() as u64 - warm;
                                st.direct += site_ids.len() as u64;
                                Some(Instant::now())
                            }
                            None => None,
                        };
                        site.copy_rows(site_ids, out)
                            .map_err(|e| GpuError::plan(WHAT, e))?;
                        if let (Some(st), Some(t0)) = (self.stats.as_mut(), t0) {
                            st.wait_ns += t0.elapsed().as_nanos() as u64;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// What the fills since open read; all zero unless the rows keep stats
    /// ([`RowsLevers::stats`]).
    #[must_use]
    pub fn engram_stats(&self) -> EngramStats {
        self.stats.unwrap_or_default()
    }

    /// Whether a helper reads the engram rows, and the cpu it is pinned to:
    /// `None` with the helper off, `Some(None)` for a floating helper.
    #[must_use]
    pub fn helper_cpu(&self) -> Option<Option<usize>> {
        self.helper.as_ref().map(Prefetcher::pinned_cpu)
    }

    /// The step's embedding rows, `tokens · n_embd` bf16 bits.
    #[must_use]
    pub fn embd(&self) -> &[u16] {
        &self.embd_rows
    }

    /// The step's engram rows: per token, every site's rows in site order.
    #[must_use]
    pub fn engram(&self) -> &[u8] {
        &self.engram_rows
    }

    /// The row ids site `site` gathered for token `token` of the last fill.
    #[must_use]
    pub fn ids(&self, token: usize, site: usize) -> &[u32] {
        let sites = self.engram.sites().len();
        &self.ids[(token * sites + site) * self.n_cols..][..self.n_cols]
    }
}

/// Every token's embedding row of `plan` into `out` (`n_embd` bf16 bits a
/// token), from `file`'s `token_embd` of `[n_embd, n_vocab]` at `embd` (its
/// shard and header).
fn embd_rows_into(
    embd: &(usize, TensorInfo),
    [n_embd, n_vocab]: [usize; 2],
    out: &mut [u16],
    file: &Split,
    plan: &StepPlan,
) -> Result<(), GpuError> {
    const WHAT: &str = "StepRows::fill";
    let refuse = |detail: String| GpuError::Shape { what: WHAT, detail };
    let (shard, info) = embd;
    let table = file
        .shard(*shard)
        .ok_or_else(|| refuse(format!("the file has no shard {shard}")))?
        .data(info)?;
    let row = 2 * n_embd;
    for (&token, dst) in plan.tokens.iter().zip(out.chunks_exact_mut(n_embd)) {
        let t = token as usize;
        let bytes = (t < n_vocab)
            .then(|| table.get(t * row..(t + 1) * row))
            .flatten()
            .ok_or_else(|| {
                refuse(format!(
                    "token {token}: no row of {n_embd} in a table of {n_vocab} rows, {} bytes",
                    table.len()
                ))
            })?;
        for (v, b) in dst.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *v = u16::from_le_bytes(*b);
        }
    }
    Ok(())
}
