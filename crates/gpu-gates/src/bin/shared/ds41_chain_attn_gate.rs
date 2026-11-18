//! GPU gate for the V4.1 attention piece (`bloomery_gpu_deepseek41::chain::attn`,
//! B5 phase 1) on the model file's own weights over seeded inputs: G1 of the
//! B5 plan cut to the piece (`docs/research/v41-b5-plan-report.md` §4.1). It
//! pins the composition; the B4 op gates pin each op.
//!
//! Steps: position 4 (where no ratio-2 group completes) and position 301
//! (the file's top-k, where no layer selects), planned by the host planner at
//! a context of 512 positions. For every layer of both, the gate uploads the
//! step image `params.rs` builds from the host plan at the step's position,
//! seeds the piece's inputs — the streams and the folded input at the
//! attention's start, the window ring, the compressed rows the layer attends,
//! and on a compressor's layer that compressor's ring — runs the piece, and
//! pins:
//!
//! - **words**, per step: the words the piece's gather left — the window
//!   length, each stream's visible count and `CompGeom` words
//!   (`CompGeom::pack` of the plan), the rope tables — are the plan's;
//! - **structure**: the layer's launches captured into a graph, replayed at
//!   both steps' inputs (the image and every buffer rewritten between
//!   replays) bit-identical to the eager runs, and the graph's node count the
//!   layer kind's launches plus the step's gather;
//! - **candidates (G2s)**, after the layers, on a piece of its own over the
//!   serving context (`workstation::CTX_MAX`; the steps' context holds 512
//!   positions) with two rows of words, the candidate tap armed, the file's
//!   weights of the mask's source layer and its first consumer: synthetic
//!   compressed rows and index keys of the ratio-1 stream (random keys, each
//!   odd block a copy of the one before it, the newest block's synthetic rows
//!   zero) and synthetic inputs, row 1 one position after row 0 at 16,384 /
//!   16,385, 24,575 / 24,576 and 32,767 / 32,768 visible rows; the source of
//!   each row, then the consumer of each — the pair pass's order, so each
//!   consumer reads its own row's kept blocks through the piece's buffer.
//!   Per row (`shared/ds41_cand.rs`): the kept blocks are the reference's
//!   selection over the card's own source scores, the compaction moved each
//!   candidate row's score, the consumer's list is the exact top-k of its
//!   scores over the kept blocks' rows, and at 16,384 nothing selects and the
//!   list is the unmasked one. Red too when no row of the cases sees a pin,
//!   a tie at the last kept place, a list the mask changed, or two rows of a
//!   pass with different kept blocks, and on any fault.
//!
//! FAIL-first: `BLOOMERY_CHAIN_ATTN_MUTATE=<name>` breaks one thing a pin
//! guards — `zero-vis` (every stream's visible count in the image reads 0),
//! `slot-shift` (the image's position reads one on), `no-group` (every
//! stream's group count reads 0), `stale-image` (a replay runs on the other
//! step's image), `extra-launch` (the capture records the gather twice).
//! `BLOOMERY_CHAIN_ATTN_LAYERS=a,b,…` runs those layers only.

use std::collections::BTreeSet;

use bloomery_gpu::weights::Weights;
use bloomery_gpu::{DeviceTensor, Gpu};
use bloomery_gpu_deepseek41::body::{STEP_TOKENS, engram_row_bytes};
use bloomery_gpu_deepseek41::chain::attn::{
    AttnChain, AttnIo, Compressed, Selection, SourceIo, WordsLayout, join_projections,
};
use bloomery_gpu_deepseek41::compress::StepInts;
use bloomery_gpu_deepseek41::params::{ImageDims, ImageLayout, StepImage, Table, rope_specs};
use bloomery_gpu_gates::{GateError, NAN_F16, activations, checks_failed, ref_model_path, verdict};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, f32_to_f16_bits};
use model::arch::Arch;
use model::arch::deepseek41::hparams::{CandidateRole, Hparams};
use model::arch::deepseek41::names;
use model::arch::deepseek41::plan::{Planner, StepPlan};
use model::placement::workstation;

use crate::ds41_cand as cand;

/// The port's names of the compressed streams, in the planner's order.
const STREAMS: [&str; 2] = ["csa", "hca"];
/// The context of the planner the layer clauses run at: the positions their
/// caches hold.
const STEP_CTX: u64 = 512;
/// The steps the layer clauses run at: their labels and positions.
const STEPS: [(&str, u32); 2] = [("pos4", 4), ("pos301", 301)];

// ------------------------------------------------------------ mutations

/// One thing a pin guards, broken on purpose (FAIL-first).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    ZeroVis,
    SlotShift,
    NoGroup,
    StaleImage,
    ExtraLaunch,
}

impl Mutation {
    const ALL: [(&'static str, Mutation); 5] = [
        ("zero-vis", Mutation::ZeroVis),
        ("slot-shift", Mutation::SlotShift),
        ("no-group", Mutation::NoGroup),
        ("stale-image", Mutation::StaleImage),
        ("extra-launch", Mutation::ExtraLaunch),
    ];

    fn from_env() -> Result<Option<Mutation>, GateError> {
        let Ok(v) = std::env::var("BLOOMERY_CHAIN_ATTN_MUTATE") else {
            return Ok(None);
        };
        Mutation::ALL
            .iter()
            .find(|(n, _)| *n == v)
            .map(|&(_, m)| Some(m))
            .ok_or_else(|| {
                let known: Vec<_> = Mutation::ALL.iter().map(|(n, _)| *n).collect();
                format!("BLOOMERY_CHAIN_ATTN_MUTATE={v}: not one of {known:?}").into()
            })
    }
}

/// The layers `BLOOMERY_CHAIN_ATTN_LAYERS` names, or all of them.
fn layer_filter(n_layer: usize) -> Result<Vec<usize>, GateError> {
    let Ok(v) = std::env::var("BLOOMERY_CHAIN_ATTN_LAYERS") else {
        return Ok((0..n_layer).collect());
    };
    let set = v
        .split(',')
        .map(|s| s.trim().parse::<usize>())
        .collect::<Result<BTreeSet<_>, _>>()
        .map_err(|e| format!("BLOOMERY_CHAIN_ATTN_LAYERS={v}: {e}"))?;
    if set.iter().any(|&l| l >= n_layer) {
        return Err(format!("BLOOMERY_CHAIN_ATTN_LAYERS={v}: the file has {n_layer}").into());
    }
    Ok(set.into_iter().collect())
}

// ---------------------------------------------------------- layer kinds

/// A layer as the gate reads it: the plan's stream, the file's
/// compressor, whether it runs the indexer, and which of its projections
/// read a q8_1 input.
#[derive(Clone, Copy, Debug)]
struct Kind {
    stream: Option<usize>,
    /// It owns its stream's compressor: whether that pools with scores
    /// (above ratio 1), whether it owns index keys.
    source: Option<(bool, bool)>,
    indexer: bool,
    /// It has a part in the candidate mask (`Hparams::candidate_role`).
    cand: bool,
    quant: Quant,
    joins: Joins,
}

/// Which of a layer's projection groups `join_projections` joins into
/// one launch: every member Q3_K in the file.
#[derive(Clone, Copy, Debug, Default)]
struct Joins {
    /// q_a, kv and, on an indexer layer, the indexer's weights projection.
    qkv: bool,
    /// On an indexer layer, q_b and the indexer's query projection.
    query: bool,
    /// A gated compressor's kv and gate.
    kv_gate: bool,
}

impl Joins {
    fn of(split: &Split, l: usize, indexer: bool, gated: bool) -> Result<Joins, GateError> {
        let q3k = |name: String| -> Result<bool, GateError> {
            let (_, t) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            Ok(t.ty == GgmlType::Q3_K)
        };
        let proj = !indexer || q3k(names::indexer_proj(l))?;
        Ok(Joins {
            qkv: q3k(names::attn_q_a(l))? && q3k(names::attn_kv(l))? && proj,
            query: indexer && q3k(names::attn_q_b(l))? && q3k(names::indexer_attn_q_b(l))?,
            kv_gate: gated
                && q3k(names::attn_compressor_kv(l))?
                && q3k(names::attn_compressor_gate(l))?,
        })
    }
}

/// Where a layer's projections read q8_1 (a K-quant weight,
/// `Dense::reads_q8_1`): q_a or kv (the norm leaves the normed input's
/// q8_1 form, which an indexer without a compressor then reuses), wo_a
/// (its groups' input quantized first) and wo_b (wo_a's output quantized
/// first).
#[derive(Clone, Copy, Debug, Default)]
struct Quant {
    normed: bool,
    heads: bool,
    wo_a: bool,
}

impl Quant {
    fn of(split: &Split, l: usize) -> Result<Quant, GateError> {
        let q8_1 = |name: String| -> Result<bool, GateError> {
            let (_, t) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the model file"))?;
            Ok(matches!(t.ty, GgmlType::Q3_K | GgmlType::Q4_K))
        };
        Ok(Quant {
            normed: q8_1(names::attn_q_a(l))? || q8_1(names::attn_kv(l))?,
            heads: q8_1(names::attn_output_a(l))?,
            wo_a: q8_1(names::attn_output_b(l))?,
        })
    }
}

impl Kind {
    fn of(hp: &Hparams, planner: &Planner, split: &Split, l: usize) -> Result<Kind, GateError> {
        let kind = &hp.layers[l];
        let gated = kind.compressor.is_some_and(|c| c.gated);
        Ok(Kind {
            stream: planner.layer_stream(l),
            source: kind.compressor.map(|c| (c.gated, kind.index_keys)),
            indexer: kind.indexer,
            cand: hp.candidate_role(l).is_some(),
            quant: Quant::of(split, l)?,
            joins: Joins::of(split, l, kind.indexer, gated)?,
        })
    }

    fn name(&self) -> String {
        match (self.stream, self.source) {
            (None, _) => "window".to_string(),
            (Some(s), None) => format!("{}-reader", STREAMS[s]),
            (Some(s), Some(_)) => format!("{}-source", STREAMS[s]),
        }
    }

    /// The piece's launches for a layer of this kind (the module doc of
    /// `chain::attn`): fourteen on every layer — HC_PRE, the norm, q_a,
    /// its norm, q_b, the q rope, kv, the K/V append, the attention's two,
    /// the inverse rope, wo_a, wo_b, HC_POST — plus the q8_1 of wo_a's
    /// input and of wo_b's where those read one (a norm that leaves a
    /// q8_1 form stays one launch); on a compressor's layer its
    /// projections, the row launch and the index key's three, and on an
    /// indexer layer its two projections, the score and top-k passes and,
    /// without a compressor and without the norm's q8_1 form, the q8_1 of
    /// the normed input, and on a layer the candidate mask gives a role
    /// its two (the source's block keys and selection, a consumer's
    /// compaction and remap) — less one launch per member a join
    /// folds into another: kv (and the indexer's weights projection) into
    /// q_a's, the indexer's query into q_b's, the gate into the
    /// compressor's kv.
    // PIN(2026-09-24): the joins of `join_projections` (ds41dense) fold launches into others.
    // PIN(2026-10-02): the candidate mask's two a role layer (candwire): 2 + 4 × 2 = 10 a pass.
    fn launches(&self) -> usize {
        let folded = usize::from(self.joins.qkv) * (1 + usize::from(self.indexer))
            + usize::from(self.joins.query)
            + usize::from(self.joins.kv_gate);
        14 - folded
            + usize::from(self.quant.heads)
            + usize::from(self.quant.wo_a)
            + self.source.map_or(0, |(gated, keys)| {
                1 + usize::from(gated) + 1 + 3 * usize::from(keys)
            })
            + if self.indexer {
                4 + usize::from(self.source.is_none() && !self.quant.normed)
            } else {
                0
            }
            + 2 * usize::from(self.cand)
    }
}

/// The names of layer `l`'s attention tensors: what the gate makes
/// resident for it.
fn resident_names(l: usize, kind: &Kind) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = [
        names::hc_attn_fn(l),
        names::hc_attn_scale(l),
        names::hc_attn_base(l),
        names::attn_norm(l),
        names::attn_q_a(l),
        names::attn_q_a_norm(l),
        names::attn_q_b(l),
        names::attn_kv(l),
        names::attn_kv_a_norm(l),
        names::attn_sinks(l),
        names::attn_output_a(l),
        names::attn_output_b(l),
    ]
    .into_iter()
    .collect();
    if let Some((gated, keys)) = kind.source {
        set.insert(names::attn_compressor_kv(l));
        set.insert(names::attn_compressor_norm(l));
        if gated {
            set.insert(names::attn_compressor_gate(l));
        }
        if keys {
            set.insert(names::indexer_attn_k(l));
            set.insert(names::indexer_k_norm(l));
        }
    }
    if kind.indexer {
        set.insert(names::indexer_attn_q_b(l));
        set.insert(names::indexer_proj(l));
    }
    set
}

// ---------------------------------------------------------------- steps

/// One step: its plan, the image built from it, and the words the gate
/// uploads (the image, or a mutation of it).
struct StepRun {
    label: &'static str,
    plan: StepPlan,
    image: StepImage,
    upload: Vec<u32>,
}

impl StepRun {
    fn pos(&self) -> usize {
        self.plan.pos[0] as usize
    }

    /// The window the step's token sees: `min(pos + 1, window)` rows.
    fn window_len(&self) -> usize {
        (self.plan.pos[0] - self.plan.raw_first[0] + 1) as usize
    }
}

/// The image's words, `mutation`'s change applied: the offsets come from
/// an image whose every word holds its own index.
fn mutate_image(
    layout: &ImageLayout,
    words: &[u32],
    mutation: Option<Mutation>,
) -> Result<Vec<u32>, GateError> {
    let mut out = words.to_vec();
    let probe: Vec<u32> = (0..u32::try_from(layout.words())?).collect();
    let at = layout.view(&probe)?;
    match mutation {
        Some(Mutation::SlotShift) => out[at.token(0).pos as usize] += 1,
        Some(Mutation::ZeroVis) => {
            for s in 0..layout.streams().len() {
                out[at.stream(s).n_visible[0] as usize] = 0;
            }
        }
        Some(Mutation::NoGroup) => {
            for s in 0..layout.streams().len() {
                out[at.stream(s).groups as usize] = 0;
            }
        }
        _ => {}
    }
    Ok(out)
}

/// The words the piece's gather must leave for `step`, at `words`'
/// offsets, as u32: the plan's counts, `CompGeom::pack` of its streams,
/// the image's tables. `None` where the copy holds nothing.
fn expected_words(words: &WordsLayout, step: &StepRun) -> Result<Vec<Option<u32>>, GateError> {
    let mut want = vec![None; words.len];
    let view = step.image.layout().view(step.image.words())?;
    let len = u32::try_from(step.window_len())?;
    want[words.pos] = Some(step.plan.pos[0]);
    want[words.window_vis] = Some(len);
    want[words.window_vis + 1] = Some(0);
    for (&at, t) in words.tables.iter().zip(Table::ALL) {
        for (i, &w) in view.table(0, t).iter().enumerate() {
            want[at + i] = Some(w);
        }
    }
    for (s, (sw, st)) in words.streams.iter().zip(&step.plan.streams).enumerate() {
        want[sw.vis] = Some(len);
        want[sw.vis + 1] = Some(st.n_visible[0]);
        let mut packed = vec![0u32; sw.geom.words()];
        sw.geom.pack(
            &StepInts {
                write_row: &st.state_write,
                read: &st.state_read,
                persist_src: &st.persist_src,
                persist_dst: &st.persist_dst,
            },
            &mut packed,
        )?;
        for (i, &w) in packed.iter().enumerate() {
            want[sw.step + i] = Some(w);
        }
        if sw.geom.ratio > 1 {
            let t = view
                .row_table(s, 0)
                .ok_or_else(|| format!("stream {s} has no row table"))?;
            for (i, &w) in t.iter().enumerate() {
                want[sw.cs + i] = Some(w);
            }
        }
    }
    Ok(want)
}

// ---------------------------------------------------------------- inputs

/// One layer's seeded inputs at one step: what the gate injects.
struct Inputs {
    s_in: Vec<f32>,
    x_in: Vec<f32>,
    ring: Vec<u16>,
    rows: Option<Vec<u16>>,
    keys: Option<Vec<u16>>,
    comp_ring: Option<(Vec<f32>, Vec<f32>)>,
    /// The compressed rows the step's token sees: the identity list's
    /// length.
    n_vis: usize,
}

/// `n` seeded f16 rows of `width`, in [-0.5, 0.5].
fn f16_rows(n: usize, width: usize, seed: u32) -> Vec<u16> {
    activations(width, n, seed)
        .iter()
        .map(|&v| f32_to_f16_bits(v.clamp(-1.0, 1.0) * 0.5))
        .collect()
}

/// The inputs of a layer of `kind` at `step`: seeded streams, fold, window
/// ring and compressor ring; the stream's rows seeded below the visible
/// count and NaN above it; a compressor's index keys NaN, which it writes.
fn seeded_inputs(
    hp: &Hparams,
    planner: &Planner,
    kind: &Kind,
    step: &StepRun,
    seed: u32,
) -> Inputs {
    let width = hp.head_dim;
    let ctx = planner.ctx_max() as usize;
    let mut inputs = Inputs {
        s_in: activations(4 * hp.n_embd, 1, seed),
        x_in: activations(hp.n_embd, 1, seed + 1),
        ring: f16_rows(ctx.min(hp.window), width, seed + 2),
        rows: None,
        keys: None,
        comp_ring: None,
        n_vis: 0,
    };
    if let Some(s) = kind.stream {
        let st = &step.plan.streams[s];
        let (ratio, n_vis) = (st.ratio as usize, st.n_visible[0] as usize);
        let n = ctx.div_ceil(ratio);
        let mut rows = vec![NAN_F16; n * width];
        rows[..n_vis * width].copy_from_slice(&f16_rows(n_vis, width, seed + 3));
        inputs.rows = Some(rows);
        inputs.n_vis = n_vis;
        if matches!(kind.source, Some((_, true))) {
            inputs.keys = Some(vec![NAN_F16; n * hp.indexer.head_dim]);
        }
        if kind.source.is_some() && ratio > 1 {
            inputs.comp_ring = Some((
                activations(width, ratio, seed + 4),
                activations(width, ratio, seed + 5),
            ));
        }
    }
    inputs
}

// --------------------------------------------------------------- buffers

/// The buffers the gate lends the piece: one of each for every layer, so
/// a captured layer replays at the same addresses.
struct Bufs {
    streams_in: DeviceBuffer<f32>,
    fold_in: DeviceBuffer<f32>,
    streams_out: DeviceBuffer<f32>,
    fold_out: DeviceBuffer<f32>,
    ring: DeviceTensor<u16>,
    /// The ring's shadow, which the append writes and this gate does not
    /// read.
    shadow: DeviceTensor<u16>,
    /// Per stream: its rows, its index keys, above ratio 1 its ring, and
    /// the list its layers read the rows through.
    rows: Vec<DeviceTensor<u16>>,
    keys: Vec<DeviceTensor<u16>>,
    comp_ring: Vec<Option<(DeviceTensor<f32>, DeviceTensor<f32>)>>,
    lists: Vec<DeviceBuffer<u32>>,
    image: DeviceBuffer<u32>,
}

impl Bufs {
    fn new(
        stream: &CudaStream,
        hp: &Hparams,
        planner: &Planner,
        image_words: usize,
        list_len: usize,
    ) -> Result<Bufs, GateError> {
        let ctx = planner.ctx_max() as usize;
        let mut rows = Vec::new();
        let mut keys = Vec::new();
        let mut comp_ring = Vec::new();
        let mut lists = Vec::new();
        for &r in planner.stream_ratios() {
            lists.push(DeviceBuffer::zeroed(stream, STEP_TOKENS * list_len)?);
            let (r, n) = (r as usize, ctx.div_ceil(r as usize));
            rows.push(DeviceTensor::zeroed(stream, n, hp.head_dim)?);
            keys.push(DeviceTensor::zeroed(stream, n, hp.indexer.head_dim)?);
            comp_ring.push(if r > 1 {
                Some((
                    DeviceTensor::zeroed(stream, r, hp.head_dim)?,
                    DeviceTensor::zeroed(stream, r, hp.head_dim)?,
                ))
            } else {
                None
            });
        }
        Ok(Bufs {
            streams_in: DeviceBuffer::zeroed(stream, 4 * hp.n_embd)?,
            fold_in: DeviceBuffer::zeroed(stream, hp.n_embd)?,
            streams_out: DeviceBuffer::zeroed(stream, 4 * hp.n_embd)?,
            fold_out: DeviceBuffer::zeroed(stream, hp.n_embd)?,
            ring: DeviceTensor::zeroed(stream, ctx.min(hp.window), hp.head_dim)?,
            shadow: DeviceTensor::zeroed(stream, ctx, hp.head_dim)?,
            rows,
            keys,
            comp_ring,
            lists,
            image: DeviceBuffer::zeroed(stream, image_words)?,
        })
    }

    fn io(&mut self, kind: &Kind) -> AttnIo<'_> {
        let Bufs {
            streams_in,
            fold_in,
            streams_out,
            fold_out,
            ring,
            shadow,
            rows,
            keys,
            comp_ring,
            lists,
            ..
        } = self;
        let Some(s) = kind.stream else {
            return AttnIo {
                streams_in,
                fold_in,
                streams_out,
                fold_out,
                ring,
                shadow,
                compressed: Compressed::None,
                selection: Selection::None,
            };
        };
        let list = &mut lists[s];
        let (compressed, selection) = match kind.source {
            None => (
                Compressed::Read(&rows[s]),
                if kind.indexer {
                    Selection::Run {
                        keys: Some(&keys[s]),
                        list,
                    }
                } else {
                    Selection::Read(list)
                },
            ),
            Some((_, has_keys)) => (
                Compressed::Source(SourceIo {
                    rows: &mut rows[s],
                    keys: if has_keys { Some(&mut keys[s]) } else { None },
                    ring: comp_ring[s].as_mut().map(|(v, sc)| (v, sc)),
                }),
                if kind.indexer {
                    Selection::Run { keys: None, list }
                } else {
                    Selection::Read(list)
                },
            ),
        };
        AttnIo {
            streams_in,
            fold_in,
            streams_out,
            fold_out,
            ring,
            shadow,
            compressed,
            selection,
        }
    }

    /// Write `c`'s inputs and `words` into the buffers a layer of `kind`
    /// reads.
    fn inject(
        &mut self,
        stream: &CudaStream,
        kind: &Kind,
        c: &Inputs,
        words: &[u32],
    ) -> Result<(), GateError> {
        self.streams_in.copy_from_host(stream, &c.s_in)?;
        self.fold_in.copy_from_host(stream, &c.x_in)?;
        self.ring.buf_mut().copy_from_host(stream, &c.ring)?;
        self.image.copy_from_host(stream, words)?;
        if let Some(s) = kind.stream {
            // The selection of the phase's steps, where no layer selects:
            // the identity over the visible rows. An indexer layer
            // writes its own over it.
            let mut list = vec![u32::MAX; self.lists[s].len()];
            for (i, e) in list.iter_mut().take(c.n_vis).enumerate() {
                *e = u32::try_from(i)?;
            }
            self.lists[s].copy_from_host(stream, &list)?;
        }
        if let (Some(s), Some(rows)) = (kind.stream, &c.rows) {
            self.rows[s].buf_mut().copy_from_host(stream, rows)?;
            if let Some(keys) = &c.keys {
                self.keys[s].buf_mut().copy_from_host(stream, keys)?;
            }
            if let (Some((v, sc)), Some((cv, csc))) = (&mut self.comp_ring[s], &c.comp_ring) {
                v.buf_mut().copy_from_host(stream, cv)?;
                sc.buf_mut().copy_from_host(stream, csc)?;
            }
        }
        Ok(())
    }

    fn read(&self, stream: &CudaStream, kind: &Kind) -> Result<Outputs, GateError> {
        let s = kind.stream;
        Ok(Outputs {
            streams: self.streams_out.to_host_vec(stream)?,
            fold: self.fold_out.to_host_vec(stream)?,
            ring: self.ring.buf().to_host_vec(stream)?,
            rows: s
                .map(|s| self.rows[s].buf().to_host_vec(stream))
                .transpose()?,
            keys: (kind.source.is_some_and(|(_, k)| k))
                .then(|| s.map(|s| self.keys[s].buf().to_host_vec(stream)))
                .flatten()
                .transpose()?,
            comp_ring: match (s, kind.source) {
                (Some(s), Some(_)) => self.comp_ring[s]
                    .as_ref()
                    .map(|(v, sc)| -> Result<_, GateError> {
                        Ok((v.buf().to_host_vec(stream)?, sc.buf().to_host_vec(stream)?))
                    })
                    .transpose()?,
                _ => None,
            },
        })
    }
}

/// What a run left in the buffers the piece writes (and the rows a
/// reading layer only reads).
struct Outputs {
    streams: Vec<f32>,
    fold: Vec<f32>,
    ring: Vec<u16>,
    rows: Option<Vec<u16>>,
    keys: Option<Vec<u16>>,
    comp_ring: Option<(Vec<f32>, Vec<f32>)>,
}

impl Outputs {
    /// Bit for bit the same.
    fn same(&self, o: &Outputs) -> bool {
        let f = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        f(&self.streams, &o.streams)
            && f(&self.fold, &o.fold)
            && self.ring == o.ring
            && self.rows == o.rows
            && self.keys == o.keys
            && match (&self.comp_ring, &o.comp_ring) {
                (Some((a, b)), Some((c, d))) => f(a, c) && f(b, d),
                (None, None) => true,
                _ => false,
            }
    }
}

// ----------------------------------------------------------------- G2s

/// G2s's cases (module doc): the two rows' visible counts of the ratio-1
/// stream, row 1 one position after row 0 as in the pair pass — at the
/// bound and one past it, then two depths that drop a third and half of
/// the blocks.
const CAND_CASES: [[usize; 2]; 3] = [[16_384, 16_385], [24_575, 24_576], [32_767, 32_768]];
/// The synthetic index keys' scale: their scores stand well above the
/// rows the step itself writes, so the newest block ranks low unless
/// pinned.
const CAND_KEY_SCALE: f32 = 16.0;

/// The synthetic index keys of a case whose rows sit at `p0` and `p0 +
/// 1`: random values, each odd block below `p0` the copy of the block
/// before it (equal keys, which the rule breaks to the lower block), and
/// the synthetic rows of row 1's newest block zero (that block's rows
/// below `p0` the lowest scores, so it ranks low unless pinned).
fn cand_keys(rows: usize, width: usize, p0: usize, block: usize, seed: u32) -> Vec<u16> {
    let vals = activations(width, rows, seed);
    let mut keys: Vec<u16> = vals
        .iter()
        .map(|&v| f32_to_f16_bits(v.clamp(-1.0, 1.0) * CAND_KEY_SCALE))
        .collect();
    let row_len = block * width;
    for b in (1..p0 / block).step_by(2) {
        let (lo, hi) = keys.split_at_mut(b * row_len);
        hi[..row_len].copy_from_slice(&lo[(b - 1) * row_len..]);
    }
    let pin = ((p0 + 2).div_ceil(block) - 1) * block;
    if pin < p0 {
        keys[pin * width..p0 * width].fill(0);
    }
    keys
}

/// G2s (module doc): the candidate mask's wiring on the piece's real
/// select path at depth, on synthetic rows and keys.
fn cand_wire(cx: &Cx<'_>, tally: &mut Tally) -> Result<(), GateError> {
    let (gpu, hp, split) = (cx.gpu, cx.hp, cx.split);
    let stream = gpu.stream();
    let mask = hp
        .candidates
        .ok_or("cand: the file carries no candidate mask")?;
    let src = mask.source_layer;
    let consumer = (src + 1..hp.n_layer)
        .find(|&l| hp.candidate_role(l) == Some(CandidateRole::Consumer))
        .ok_or("cand: no consumer above the source layer")?;
    let planner = Planner::from_file(split, hp, workstation::CTX_MAX)?;
    let layout = ImageLayout::new(ImageDims::of(
        hp,
        &planner,
        STEP_TOKENS,
        engram_row_bytes(split, hp)?,
    ))?;
    let (window, yarn) = rope_specs(hp)?;
    let mut piece = AttnChain::with_rows(gpu, hp, 0..hp.n_layer, &layout, &planner, 2)?;
    piece.arm_cand_tap(gpu, true)?;
    let top_k = piece.top_k();
    let mut bufs = Bufs::new(stream, hp, &planner, layout.words(), piece.list_len())?;
    let kinds = [
        Kind::of(hp, &planner, split, src)?,
        Kind::of(hp, &planner, split, consumer)?,
    ];
    let st = match (kinds[0].stream, kinds[1].stream) {
        (Some(a), Some(b)) if a == b => a,
        other => {
            return Err(format!(
                "cand: layers {src} and {consumer} attend streams {other:?}, not one"
            )
            .into());
        }
    };
    let mut keep = resident_names(src, &kinds[0]);
    keep.extend(resident_names(consumer, &kinds[1]));
    let mut w = Weights::load_where(stream, split, |n| keep.contains(n))?;
    join_projections(stream, hp, src..src + 1, &mut w)?;
    join_projections(stream, hp, consumer..consumer + 1, &mut w)?;
    let (blocks, block) = (mask.topk_blocks, mask.block_size);
    let n_rows = bufs.rows[st].rows();
    let latent: Vec<u16> = activations(hp.head_dim, n_rows, 7301)
        .iter()
        .map(|&v| f32_to_f16_bits(v.clamp(-1.0, 1.0) * 0.5))
        .collect();
    bufs.rows[st].buf_mut().copy_from_host(stream, &latent)?;
    let before = vec![1u32; planner.ngram()];
    gpu.clear_fault()?;
    let (mut pin_seen, mut tie_seen, mut changed, mut differ) = (false, false, 0, false);
    let mut all_ok = true;
    for (case, ns) in CAND_CASES.iter().enumerate() {
        let p0 = ns[0] - 1;
        let keys = cand_keys(n_rows, hp.indexer.head_dim, p0, block, 7400 + case as u32);
        bufs.keys[st].buf_mut().copy_from_host(stream, &keys)?;
        for (r, &n) in ns.iter().enumerate() {
            let mut plan = StepPlan::default();
            planner.plan_into(&[1], u32::try_from(n - 1)?, &before, &mut plan)?;
            let mut image = StepImage::new(layout.clone(), &window, &yarn)?;
            let embd = vec![0u8; STEP_TOKENS * layout.dims().embd_bytes];
            let engram = vec![0u8; STEP_TOKENS * layout.dims().engram_bytes];
            image.build(&plan, &embd, &engram)?;
            bufs.image.copy_from_host(stream, image.words())?;
            piece.enqueue_step_of(gpu, &bufs.image, r)?;
        }
        // The rows in the pair pass's order: the source of each row,
        // then the consumer of each, which reads its own row's kept
        // blocks through the piece.
        let mut source = Vec::new();
        for (r, &n) in ns.iter().enumerate() {
            let seed = 7500 + 10 * case as u32 + r as u32;
            bufs.streams_in
                .copy_from_host(stream, &activations(4 * hp.n_embd, 1, seed))?;
            bufs.fold_in
                .copy_from_host(stream, &activations(hp.n_embd, 1, seed + 5))?;
            piece.enqueue_layer_of(gpu, &w, src, r, bufs.io(&kinds[0]), None)?;
            stream.synchronize()?;
            let taps = piece.taps();
            let sel = taps
                .select
                .ok_or("cand: the source left no selection taps")?;
            let scores = sel.scores.to_host_vec(stream)?;
            let kept = sel
                .kept
                .ok_or("cand: the source layer's taps carry no kept blocks")?
                .to_host_vec(stream)?;
            source.push((scores[..n].to_vec(), kept));
        }
        for (r, &n) in ns.iter().enumerate() {
            let seed = 7600 + 10 * case as u32 + r as u32;
            bufs.streams_in
                .copy_from_host(stream, &activations(4 * hp.n_embd, 1, seed))?;
            bufs.fold_in
                .copy_from_host(stream, &activations(hp.n_embd, 1, seed + 5))?;
            let poison = vec![u32::MAX; bufs.lists[st].len()];
            bufs.lists[st].copy_from_host(stream, &poison)?;
            piece.enqueue_layer_of(gpu, &w, consumer, r, bufs.io(&kinds[1]), None)?;
            stream.synchronize()?;
            let taps = piece.taps();
            let sel = taps
                .select
                .ok_or("cand: the consumer left no selection taps")?;
            let compacted = sel.scores.to_host_vec(stream)?;
            let unmasked = sel
                .unmasked
                .ok_or("cand: the consumer's taps carry no unmasked scores (tap not armed)")?
                .to_host_vec(stream)?;
            let list = bufs.lists[st].to_host_vec(stream)?;
            let (scores, kept) = &source[r];
            let row = cand::check(&cand::Card {
                n,
                top_k,
                blocks,
                block,
                source: scores,
                kept,
                unmasked: &unmasked,
                compacted: &compacted,
                list: &list,
            });
            let at_bound = n <= blocks * block;
            let pass = row.pass() && row.selects != at_bound;
            println!(
                "cand case {case} row {r} layers {src}→{consumer}: {} — {}",
                row.line(),
                verdict(pass)
            );
            tally.check(pass, format!("cand case {case} row {r}"));
            all_ok &= pass;
            pin_seen |= row.pin_below;
            tie_seen |= row.tie;
            changed += row.changed;
            if r == 1 && row.selects {
                differ |= source[0].1[..blocks] != source[1].1[..blocks];
            }
        }
    }
    piece.arm_cand_tap(gpu, false)?;
    let fault = gpu.fault()?;
    let clean = fault.is_none();
    println!(
        "cand: fault word after the cases {fault:?} — {}",
        verdict(clean)
    );
    tally.check(clean, "cand fault word".to_string());
    // What the data lets the checks see (`shared/ds41_cand.rs`).
    for (seen, what) in [
        (
            pin_seen,
            "a row whose newest block falls out without its pin",
        ),
        (
            tie_seen,
            "equal block keys on both sides of the last kept place",
        ),
        (changed > 0, "a consumer list the mask changed"),
        (differ, "two rows of one pass with different kept blocks"),
    ] {
        println!("cand witness: {what}: {seen} — {}", verdict(seen));
        tally.check(seen, format!("cand witness: {what}"));
    }
    println!(
        "cand: {} rows {}, {changed} list entries the mask changed",
        2 * CAND_CASES.len(),
        if all_ok {
            "to the rule"
        } else {
            "OFF the rule"
        }
    );
    Ok(())
}

// -------------------------------------------------------------- the run

/// What the layers read besides their own weights.
struct Cx<'a> {
    gpu: &'a Gpu,
    split: &'a Split,
    hp: &'a Hparams,
    planner: &'a Planner,
    mutation: Option<Mutation>,
}

/// Tallies per pin.
#[derive(Default)]
struct Tally {
    runs: usize,
    failed: Vec<String>,
}

impl Tally {
    fn check(&mut self, pass: bool, what: String) {
        self.runs += 1;
        if !pass {
            self.failed.push(what);
        }
    }
}

pub fn run() -> Result<(), GateError> {
    let mutation = Mutation::from_env()?;
    let model = ref_model_path()?;
    let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
    if split.architecture() != Some(Arch::Deepseek41.name()) {
        return Err(format!(
            "{} is not a {} model",
            model.display(),
            Arch::Deepseek41.name()
        )
        .into());
    }
    let hp = Hparams::read(&split)?;
    let planner = Planner::from_file(&split, &hp, STEP_CTX)?;
    let layout = ImageLayout::new(ImageDims::of(
        &hp,
        &planner,
        STEP_TOKENS,
        engram_row_bytes(&split, &hp)?,
    ))?;
    let (window, yarn) = rope_specs(&hp)?;
    let before = vec![1u32; planner.ngram()];
    let mut steps = Vec::new();
    for (label, pos0) in STEPS {
        let mut plan = StepPlan::default();
        planner.plan_into(&[1], pos0, &before, &mut plan)?;
        let mut image = StepImage::new(layout.clone(), &window, &yarn)?;
        let embd = vec![0u8; STEP_TOKENS * layout.dims().embd_bytes];
        let engram = vec![0u8; STEP_TOKENS * layout.dims().engram_bytes];
        image.build(&plan, &embd, &engram)?;
        let upload = mutate_image(&layout, image.words(), mutation)?;
        println!(
            "step {label}: position {pos0}, context {STEP_CTX}, window {} rows; \
                 streams {:?} visible {:?} completing {:?}",
            plan.pos[0] - plan.raw_first[0] + 1,
            planner.stream_ratios(),
            plan.streams
                .iter()
                .map(|s| s.n_visible[0])
                .collect::<Vec<_>>(),
            plan.streams
                .iter()
                .map(|s| s.state_write.clone())
                .collect::<Vec<_>>(),
        );
        steps.push(StepRun {
            label,
            plan,
            image,
            upload,
        });
    }

    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let mut piece = AttnChain::new(&gpu, &hp, 0..hp.n_layer, &layout, &planner)?;
    let mut bufs = Bufs::new(stream, &hp, &planner, layout.words(), piece.list_len())?;
    println!(
        "gate_deepseek41_chain_attn: device {}; model {}; piece scratch {} bytes, {} step \
             words; mutation {mutation:?}",
        gpu.device_name()?,
        model.display(),
        piece.device_bytes(),
        piece.words_layout().len,
    );

    let mut tally = Tally::default();
    for step in &steps {
        words_check(&gpu, &mut piece, &mut bufs, step, &mut tally)?;
    }
    let cx = Cx {
        gpu: &gpu,
        split: &split,
        hp: &hp,
        planner: &planner,
        mutation,
    };
    for l in layer_filter(hp.n_layer)? {
        gate_layer(&cx, &mut piece, &mut bufs, &steps, l, &mut tally)?;
    }
    drop(bufs);
    drop(piece);
    cand_wire(&cx, &mut tally)?;
    let pass = tally.failed.is_empty();
    println!(
        "gate_deepseek41_chain_attn: {} checks, {} failed{} — {}",
        tally.runs,
        tally.failed.len(),
        if pass {
            String::new()
        } else {
            format!(": {}", tally.failed.join("; "))
        },
        verdict(pass)
    );
    if pass { Ok(()) } else { Err(checks_failed()) }
}

/// The words pin of one step: gather its image and read the piece's copy
/// back against the plan's words.
fn words_check(
    gpu: &Gpu,
    piece: &mut AttnChain,
    bufs: &mut Bufs,
    step: &StepRun,
    tally: &mut Tally,
) -> Result<(), GateError> {
    let stream = gpu.stream();
    bufs.image.copy_from_host(stream, &step.upload)?;
    piece.enqueue_step(gpu, &bufs.image)?;
    stream.synchronize()?;
    let got = piece.words().to_host_vec(stream)?;
    let want = expected_words(piece.words_layout(), step)?;
    let wrong: Vec<usize> = want
        .iter()
        .enumerate()
        .filter(|&(i, w)| w.is_some_and(|w| got[i].to_bits() != w))
        .map(|(i, _)| i)
        .collect();
    let pass = wrong.is_empty();
    println!(
        "words step={} pos={}: {} of {} words the plan's{} — {}",
        step.label,
        step.pos(),
        want.iter().flatten().count() - wrong.len(),
        want.iter().flatten().count(),
        wrong.first().map_or(String::new(), |&i| format!(
            "; first off at word {i}: {} for {}",
            got[i].to_bits(),
            want[i].unwrap_or(0)
        )),
        verdict(pass)
    );
    tally.check(pass, format!("words {}", step.label));
    Ok(())
}

/// The structure pin of layer `l`: both steps eager, then the capture and
/// its replays.
fn gate_layer(
    cx: &Cx<'_>,
    piece: &mut AttnChain,
    bufs: &mut Bufs,
    steps: &[StepRun],
    l: usize,
    tally: &mut Tally,
) -> Result<(), GateError> {
    let (gpu, hp) = (cx.gpu, cx.hp);
    let stream = gpu.stream();
    let kind = Kind::of(hp, cx.planner, cx.split, l)?;
    let keep = resident_names(l, &kind);
    let mut w = Weights::load_where(stream, cx.split, |n| keep.contains(n))?;
    join_projections(stream, hp, l..l + 1, &mut w)?;
    let mut eager = Vec::with_capacity(steps.len());
    let mut cases = Vec::with_capacity(steps.len());
    for (i, step) in steps.iter().enumerate() {
        let seed = 100 * u32::try_from(l)? + 10 * u32::try_from(i)?;
        let c = seeded_inputs(hp, cx.planner, &kind, step, seed);
        bufs.inject(stream, &kind, &c, &step.upload)?;
        piece.enqueue_step(gpu, &bufs.image)?;
        piece.enqueue_layer(gpu, &w, l, bufs.io(&kind))?;
        stream.synchronize()?;
        eager.push(bufs.read(stream, &kind)?);
        cases.push(c);
    }
    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause

    // The capture, then a replay at each step's inputs.
    let extra = cx.mutation == Some(Mutation::ExtraLaunch);
    let graph = gpu.capture(|_| {
        piece.enqueue_step(gpu, &bufs.image)?;
        if extra {
            piece.enqueue_step(gpu, &bufs.image)?;
        }
        piece.enqueue_layer(gpu, &w, l, bufs.io(&kind))
    })?;
    let want = 1 + kind.launches();
    let mut same = Vec::with_capacity(steps.len());
    for (i, (step, c)) in steps.iter().zip(&cases).enumerate() {
        let stale = cx.mutation == Some(Mutation::StaleImage) && i > 0;
        let words = if stale {
            &steps[0].upload
        } else {
            &step.upload
        };
        bufs.inject(stream, &kind, c, words)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        same.push(bufs.read(stream, &kind)?.same(&eager[i]));
    }
    let pass = graph.node_count() == want && same.iter().all(|&s| s);
    println!(
        "replay layer={l} kind={} nodes={} predicted={want} replay_equals_eager={} — {}",
        kind.name(),
        graph.node_count(),
        steps
            .iter()
            .zip(&same)
            .map(|(s, &e)| format!("{}:{e}", s.label))
            .collect::<Vec<_>>()
            .join(","),
        verdict(pass)
    );
    tally.check(pass, format!("layer {l} replay"));
    Ok(())
}
