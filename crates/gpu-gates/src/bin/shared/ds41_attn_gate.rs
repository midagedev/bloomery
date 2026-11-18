//! GPU gate for the V4.1 attention (`bloomery_gpu_deepseek41::attn`): the
//! tensor-core segment pass, over a compressed prefix and over the rows the
//! indexer selects, and the merge with each head's sink — on synthetic rows
//! from fixed seeds, with the model's own head count, window and sinks.
//!
//! The kernel is held to our rule: the f16 query, exact f16 products summed
//! in f64, the logit rounded to f32 and scaled in f32, the softmax and the
//! value sum in f64 with the sink in the denominator — within `KERNEL_BAND`,
//! not bit for bit: the kernel sums the products on the tensor cores and the
//! softmax and the values in f32, each in its own order. The rule is written
//! out here; it calls nothing the kernel runs. Every partial and output
//! buffer is NaN before each launch, so a slot a kernel fails to write is
//! caught too, and every row of a buffer that no key names is the f16 NaN, so
//! a row read past a token's counts reaches the output as NaN, which
//! `max_rel_err` refuses.
//!
//! - a depth case per compressed source: a full ring and compressed keys over
//!   many segments, the last one partial, and for the selected rows list
//!   entries past the count that name NaN rows, against our rule with the
//!   model's sinks, plus a rerun;
//! - the same selected list with one visible entry naming the row just past
//!   the stream, which the kernel must refuse by raising its fault
//!   (`FaultSite::AttnSel`), not read as a plausible row;
//! - one captured graph per compressed source replayed with its inputs
//!   rewritten between replays — per token of a five-token batch with the
//!   query and the counts, and over lists and counts with the selected rows —
//!   bit-identical to eager runs (the grid comes from the buffer heights and
//!   the list's length, never from the counts or the lists);
//! - every entry compiled with no local depot.
//!
//! Ring ⧺ staging, "batch = sequence": synthetic rows from fixed seeds,
//! batches of 1, 127, 128, 129 and 300 tokens whose first position is 0, 5,
//! 127, 128 or 300 — before, at and across the wrap of the model's window —
//! each with no compressed rows, a stream prefix and a list. One staged
//! launch over ring ⧺ staging plus the commit against one ring-only launch
//! per token in order, each writing its row into the ring first: every
//! token's output bit for bit and the ring after the commit byte for byte;
//! the ring's slots past its positions and the staging rows past the batch
//! are NaN. One captured staged launch and commit replayed at other first
//! positions against eager runs; and each shape the staged enqueue and the
//! commit refuse, refused by name before any launch.
//!
//! No clause here reads ik's dump.
//! PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause

use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, LAYER_NONE};
use bloomery_gpu_deepseek41::attn::{
    self, AttnArgs, AttnKernels, CommitArgs, LATENT, SelectedRows, Staged,
};
use bloomery_gpu_gates::{
    GateError, KERNEL_BAND, NAN_F16, activations, bits_equal, checks_failed, max_rel_err,
    no_local_depot, open_split, ref_model_path, verdict,
};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, f32_to_f16_bits, half_to_f32};
use model::arch::Arch;
use model::arch::deepseek41::hparams::Hparams;

/// The depth cases: compressed keys visible, and the compressed source's
/// capacity — nine segments, the eighth partial, the ninth neutral. The
/// prefix case's stream holds `DEPTH_COMP_ROWS` rows; the selected-row
/// case's list holds `DEPTH_SEL_STRIDE` entries over a stream of
/// `DEPTH_COMP_ROWS` rows.
const DEPTH_COMP_VISIBLE: usize = 8 * attn::SEG_KEYS - 7;
const DEPTH_COMP_ROWS: usize = 9 * attn::SEG_KEYS;
const DEPTH_SEL_STRIDE: usize = 8 * attn::SEG_KEYS + 5;
/// The visible entry the past-stream case plants: it names row
/// `DEPTH_COMP_ROWS`, the first row past the stream.
const DEPTH_PAST_ENTRY: usize = 100;

/// The batch the prefix graph replays: its tokens, at positions `0 ..`.
const REPLAY_TOKENS: usize = 5;

/// What every clause shares: the model's attention shape.
struct Shape {
    heads: usize,
    /// `attention.sliding_window`: our ring's height.
    window: usize,
    scale: f32,
}

impl Shape {
    fn of(hp: &Hparams) -> Result<Shape, GateError> {
        if hp.head_dim != LATENT {
            return Err(format!(
                "the model's latent is {} values, the kernels take {LATENT}",
                hp.head_dim
            )
            .into());
        }
        Ok(Shape {
            heads: hp.n_head,
            window: hp.window,
            scale: 1.0 / (LATENT as f32).sqrt(),
        })
    }
}

pub fn run() -> Result<(), GateError> {
    let model = ref_model_path()?;
    let split = open_split(Arch::Deepseek41, "gate-gpu-ds41-kernels")?;
    let hp = Hparams::read(&split)?;
    let set = Shape::of(&hp)?;
    // Every clause takes the sinks of the first layer that reads a
    // compressed stream.
    let layer = hp
        .layers
        .iter()
        .position(|k| k.stream.is_some() || k.dense.is_some())
        .ok_or("the model has no layer that reads a compressed stream")?;
    let sinks = sinks_of(&split, layer, set.heads)?;
    println!(
        "gate_deepseek41_attn: model {}, heads {}, window {}, scale {:e}, sinks of layer {layer}",
        model.display(),
        set.heads,
        set.window,
        set.scale
    );

    let gpu = Gpu::new()?;
    let kernels = AttnKernels::load(gpu.context())?;
    let stream = gpu.stream();
    // Every entry compiles with no local depot: a spilled accumulator
    // array changes no output and no band, only the time.
    let mut ok = no_local_depot(&[
        "ds41_attn_seg",
        "ds41_attn_seg_sel",
        "ds41_attn_seg_stage",
        "ds41_attn_seg_sel_stage",
        "ds41_attn_merge",
        "ds41_ring_commit",
    ])?;
    let mut kernel_max = 0.0f32;
    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    ok &= graph_replay(&gpu, &kernels, &set, &sinks)?;
    ok &= depth_case(
        &kernels,
        stream,
        gpu.unlabelled_sink(),
        &set,
        &sinks,
        &mut kernel_max,
    )?;
    ok &= sel_graph_replay(&gpu, &kernels, &set, &sinks)?;
    ok &= sel_depth_case(
        &kernels,
        stream,
        gpu.unlabelled_sink(),
        &set,
        &sinks,
        &mut kernel_max,
    )?;
    ok &= sel_past_stream_case(&kernels, &gpu, &set, &sinks)?;
    ok &= stage_cases(&kernels, stream, gpu.unlabelled_sink(), &set, &sinks)?;
    ok &= stage_graph_replay(&gpu, &kernels, &set, &sinks)?;
    ok &= stage_refusals(&kernels, stream, gpu.unlabelled_sink(), &set)?;

    println!(
        "gate_deepseek41_attn: kernel_rel max {kernel_max:.2e} (band {KERNEL_BAND:e}) over the \
         depth cases — {}",
        verdict(ok)
    );
    if ok { Ok(()) } else { Err(checks_failed()) }
}

// ------------------------------------------------------------ the model

/// `blk.L.attn_sinks.weight`: one f32 logit per head.
fn sinks_of(split: &Split, l: usize, heads: usize) -> Result<Vec<f32>, GateError> {
    let name = format!("blk.{l}.attn_sinks.weight");
    let (shard, info) = split
        .find(&name)
        .ok_or_else(|| format!("{name} is not in the model"))?;
    if info.ty != GgmlType::F32 || info.dims != [heads as u64] {
        return Err(format!(
            "{name} is {:?} {:?}, want F32 [{heads}]",
            info.ty, info.dims
        )
        .into());
    }
    let bytes = split.shard(shard).ok_or("shard out of range")?.data(info)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

// ------------------------------------------------------------ host rules

/// Row `r` of `LATENT`-value f16 rows.
fn row_of(bits: &[u16], r: usize) -> &[u16] {
    &bits[r * LATENT..(r + 1) * LATENT]
}

/// Keys token `t` sees, widened: the ring rows at `slots`, then its
/// `count` compressed rows — a prefix of the stream, or its list's first
/// entries, each clamped to the stream as the kernel reads it.
fn keys_of<'a>(
    ring: &'a [f32],
    slots: &[usize],
    comp: Option<&'a [f32]>,
    sel: Option<(&[u32], usize)>,
    t: usize,
    count: usize,
) -> Vec<&'a [f32]> {
    let mut keys: Vec<&[f32]> = slots
        .iter()
        .map(|&s| &ring[s * LATENT..(s + 1) * LATENT])
        .collect();
    if let Some(c) = comp {
        let height = c.len() / LATENT;
        for j in 0..count {
            let r = sel.map_or(j, |(s, k)| s[t * k + j] as usize);
            assert!(
                r < height,
                "keys_of: entry {r} names no row of a stream of {height}"
            );
            keys.push(&c[r * LATENT..(r + 1) * LATENT]);
        }
    }
    keys
}

/// Every token's keys in the kernel's walk order: ring slots in order,
/// then the compressed keys.
fn walk_keys<'a>(
    ring: &'a [f32],
    comp: Option<&'a [f32]>,
    sel: Option<(&[u32], usize)>,
    vis: &[u32],
) -> Vec<Vec<&'a [f32]>> {
    vis.chunks(2)
        .enumerate()
        .map(|(t, c)| {
            let slots: Vec<usize> = (0..c[0] as usize).collect();
            keys_of(ring, &slots, comp, sel, t, c[1] as usize)
        })
        .collect()
}

fn widen(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| half_to_f32(b)).collect()
}

/// A query row as the f16 values the tensor cores see.
fn query_f16(q: &[f32]) -> Vec<f32> {
    q.iter().map(|&v| half_to_f32(f32_to_f16_bits(v))).collect()
}

/// The exact softmax for one query row as given: each logit the exact
/// sum of the products rounded to f32 and scaled in f32, then the
/// softmax and the value sum in f64 with the sink one more logit in the
/// denominator.
fn exact_rule(q: &[f32], keys: &[&[f32]], scale: f32, sink: f32) -> Vec<f32> {
    let logits: Vec<f64> = keys
        .iter()
        .map(|k| {
            let dot: f64 = k
                .iter()
                .zip(q)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            f64::from((dot as f32) * scale)
        })
        .collect();
    let sink = f64::from(sink);
    let mx = logits.iter().copied().fold(sink, f64::max);
    let mut denom = (sink - mx).exp();
    let mut num = vec![0.0f64; LATENT];
    for (k, &s) in keys.iter().zip(&logits) {
        let w = (s - mx).exp();
        denom += w;
        for (n, &v) in num.iter_mut().zip(*k) {
            *n += w * f64::from(v);
        }
    }
    num.iter().map(|&n| (n / denom) as f32).collect()
}

/// Our rule for one query row: the query rounded to f16, the tensor
/// cores' input, then [`exact_rule`].
fn our_rule(q: &[f32], keys: &[&[f32]], scale: f32, sink: f32) -> Vec<f32> {
    exact_rule(&query_f16(q), keys, scale, sink)
}

/// Our rule over every query row of a launch: `q` holds `keys.len()`
/// tokens of `heads` rows, and row `t * heads + h` sees token `t`'s keys
/// with head `h`'s sink.
fn rule_all(q: &[f32], keys: &[Vec<&[f32]>], sinks: &[f32], scale: f32) -> Vec<f32> {
    let heads = sinks.len();
    let mut out = Vec::with_capacity(q.len());
    for (t, keys) in keys.iter().enumerate() {
        for (h, &sink) in sinks.iter().enumerate() {
            let row = (t * heads + h) * LATENT;
            out.extend(our_rule(&q[row..row + LATENT], keys, scale, sink));
        }
    }
    out
}

// ------------------------------------------------------------ launches

/// One launch's host inputs.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    q: &'a [f32],
    ring: &'a [u16],
    comp: Option<&'a [u16]>,
    /// The lists and their stride, when the compressed rows are
    /// selected.
    sel: Option<(&'a [u32], usize)>,
    vis: &'a [u32],
    sinks: &'a [f32],
    tokens: usize,
    window: usize,
}

/// A launch's device buffers: inputs uploaded once, partials and output
/// refilled with NaN before every launch.
struct Launch {
    q: DeviceBuffer<f32>,
    window: DeviceTensor<u16>,
    comp: Option<DeviceTensor<u16>>,
    sel: Option<(DeviceBuffer<u32>, usize)>,
    vis: DeviceBuffer<u32>,
    sinks: DeviceBuffer<f32>,
    tokens: usize,
    heads: usize,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    y: DeviceBuffer<f32>,
    /// Where the selected-row entry raises a list entry past the stream.
    fault: FaultSink,
}

impl Launch {
    fn new(stream: &CudaStream, i: &Inputs<'_>, fault: FaultSink) -> Result<Launch, GateError> {
        let heads = i.sinks.len();
        let comp = i
            .comp
            .map(|c| DeviceTensor::upload(stream, c, c.len() / LATENT, LATENT))
            .transpose()?;
        let comp_keys = match i.sel {
            Some((_, k)) => k,
            None => comp.as_ref().map_or(0, DeviceTensor::rows),
        };
        let sel = i
            .sel
            .map(|(s, k)| DeviceBuffer::from_host(stream, s).map(|b| (b, k)))
            .transpose()?;
        let segs = attn::segments(i.window, comp_keys);
        let rows = i.tokens * heads;
        Ok(Launch {
            q: DeviceBuffer::from_host(stream, i.q)?,
            window: DeviceTensor::upload(stream, i.ring, i.window, LATENT)?,
            comp,
            sel,
            vis: DeviceBuffer::from_host(stream, i.vis)?,
            sinks: DeviceBuffer::from_host(stream, i.sinks)?,
            tokens: i.tokens,
            heads,
            part_v: DeviceBuffer::from_host(
                stream,
                &vec![f32::NAN; attn::partials_v_len(rows, segs)],
            )?,
            part_ms: DeviceBuffer::from_host(
                stream,
                &vec![f32::NAN; attn::partials_ms_len(rows, segs)],
            )?,
            y: DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * LATENT])?,
            fault,
        })
    }

    /// NaN into the partials and the output (synchronous copies).
    fn poison(&mut self, stream: &CudaStream) -> Result<(), GateError> {
        self.part_v
            .copy_from_host(stream, &vec![f32::NAN; self.part_v.len()])?;
        self.part_ms
            .copy_from_host(stream, &vec![f32::NAN; self.part_ms.len()])?;
        self.y
            .copy_from_host(stream, &vec![f32::NAN; self.y.len()])?;
        Ok(())
    }

    /// The launch's arguments over its buffers.
    fn args(&mut self, scale: f32) -> AttnArgs<'_> {
        AttnArgs {
            q: &self.q,
            window: &self.window,
            compressed: self.comp.as_ref(),
            selected: self.sel.as_ref().map(|(rows, stride)| SelectedRows {
                rows,
                stride: *stride,
            }),
            vis: &self.vis,
            sinks: &self.sinks,
            scale,
            tokens: self.tokens,
            heads: self.heads,
            part_v: &mut self.part_v,
            part_ms: &mut self.part_ms,
            y: &mut self.y,
            fault: self.fault,
        }
    }

    fn enqueue(
        &mut self,
        kernels: &AttnKernels,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<(), GpuError> {
        kernels.enqueue(stream, self.args(scale))
    }

    /// One eager launch on NaN-filled partials and output, read back.
    fn run(
        &mut self,
        kernels: &AttnKernels,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<Vec<f32>, GateError> {
        self.poison(stream)?;
        self.enqueue(kernels, stream, scale)?;
        stream.synchronize()?;
        Ok(self.y.to_host_vec(stream)?)
    }
}

/// One graph captured with a T = 1 launch over a compressed prefix and
/// replayed for every token of a [`REPLAY_TOKENS`]-token batch, the query
/// and the counts rewritten in place between replays: each replay
/// bit-identical to that token's eager T = 1 run. A grid taken from the
/// counts would freeze the captured token's segments.
fn graph_replay(
    gpu: &Gpu,
    kernels: &AttnKernels,
    set: &Shape,
    sinks: &[f32],
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let c = StageCase::new(set, REPLAY_TOKENS, 0, StageKeys::Prefix)?;
    let ring = c.full_ring();
    let per_token = c.heads * LATENT;
    let eager = (0..c.tokens)
        .map(|t| {
            Launch::new(stream, &c.inputs(sinks, &ring, t, 1), gpu.unlabelled_sink())?
                .run(kernels, stream, set.scale)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut launch = Launch::new(stream, &c.inputs(sinks, &ring, 0, 1), gpu.unlabelled_sink())?;
    let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
    let mut same = 0;
    for (t, want) in eager.iter().enumerate() {
        launch
            .q
            .copy_from_host(stream, &c.q[t * per_token..(t + 1) * per_token])?;
        launch
            .vis
            .copy_from_host(stream, &c.vis[2 * t..2 * t + 2])?;
        launch.poison(stream)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        same += usize::from(bits_equal(&launch.y.to_host_vec(stream)?, want));
    }
    let pass = same == eager.len();
    println!(
        "graph keys=prefix tokens={} nodes={} replays={} bit-identical={same}/{} {}",
        c.tokens,
        graph.node_count(),
        eager.len(),
        eager.len(),
        verdict(pass)
    );
    Ok(pass)
}

/// One graph captured with the depth case's selecting launch and replayed
/// over other lists and counts — the list reversed and rotated, the count
/// cut to part of a segment, to one key and to none — the list and the
/// counts rewritten in place between replays: each replay bit-identical
/// to an eager launch of the same inputs. A grid taken from the counts
/// or the list would freeze the captured selection.
fn sel_graph_replay(
    gpu: &Gpu,
    kernels: &AttnKernels,
    set: &Shape,
    sinks: &[f32],
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let (q, ring) = depth_ring(set);
    let (list, comp) = depth_list(None)?;
    let (stride, count) = (DEPTH_SEL_STRIDE, DEPTH_COMP_VISIBLE);
    let cut = count
        .checked_sub(37)
        .ok_or("the replayed list sees under 38 selected rows")?;
    let mut reversed = list.clone();
    reversed[..count].reverse();
    let mut rotated = list.clone();
    rotated[..count].rotate_left(count / 3);
    let variants = [
        (list.clone(), count),
        (reversed, count),
        (list.clone(), cut),
        (rotated, 1),
        (list.clone(), 0),
    ];
    let w = u32::try_from(set.window)?;
    let base_vis = [w, u32::try_from(count)?];
    let base = Inputs {
        q: &q,
        ring: &ring,
        comp: Some(&comp),
        sel: Some((&list[..], stride)),
        vis: &base_vis,
        sinks,
        tokens: 1,
        window: set.window,
    };
    let mut launch = Launch::new(stream, &base, gpu.unlabelled_sink())?;
    let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
    let mut same = 0;
    for (l, c) in &variants {
        let vis = [w, u32::try_from(*c)?];
        let want = Launch::new(
            stream,
            &Inputs {
                sel: Some((&l[..], stride)),
                vis: &vis,
                ..base
            },
            gpu.unlabelled_sink(),
        )?
        .run(kernels, stream, set.scale)?;
        let (sel_buf, _) = launch
            .sel
            .as_mut()
            .ok_or("the captured launch holds no list")?;
        sel_buf.copy_from_host(stream, l)?;
        launch.vis.copy_from_host(stream, &vis)?;
        launch.poison(stream)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        same += usize::from(bits_equal(&launch.y.to_host_vec(stream)?, &want));
    }
    let pass = same == variants.len();
    println!(
        "graph keys=sel/{stride} nodes={} replays={} counts={} bit-identical={same}/{} {}",
        graph.node_count(),
        variants.len(),
        variants
            .iter()
            .map(|(_, c)| c.to_string())
            .collect::<Vec<_>>()
            .join(","),
        variants.len(),
        verdict(pass)
    );
    Ok(pass)
}

/// One token's query rows and a full ring, both synthetic: the depth
/// cases' shared inputs.
fn depth_ring(set: &Shape) -> (Vec<f32>, Vec<u16>) {
    let q = activations(LATENT, set.heads, 13);
    let ring = activations(LATENT, set.window, 7)
        .into_iter()
        .map(f32_to_f16_bits)
        .collect();
    (q, ring)
}

/// The depth shape with a prefix: one token seeing a full ring and a
/// compressed prefix of many segments, the last partial and one more
/// wholly past the count — synthetic rows and query and the model's
/// sinks, against our rule, plus a rerun.
fn depth_case(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
    sinks: &[f32],
    max: &mut f32,
) -> Result<bool, GateError> {
    let w = set.window;
    let (q, ring) = depth_ring(set);
    let mut comp = vec![NAN_F16; DEPTH_COMP_ROWS * LATENT];
    for (c, v) in comp
        .iter_mut()
        .zip(activations(LATENT, DEPTH_COMP_VISIBLE, 11))
    {
        *c = f32_to_f16_bits(v);
    }
    let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
    let (ring_w, comp_w) = (widen(&ring), widen(&comp));
    let ours = rule_all(
        &q,
        &walk_keys(&ring_w, Some(&comp_w), None, &vis),
        sinks,
        set.scale,
    );
    let inputs = Inputs {
        q: &q,
        ring: &ring,
        comp: Some(&comp),
        sel: None,
        vis: &vis,
        sinks,
        tokens: 1,
        window: w,
    };
    let mut launch = Launch::new(stream, &inputs, fault)?;
    let y = launch.run(kernels, stream, set.scale)?;
    let rerun = bits_equal(&y, &launch.run(kernels, stream, set.scale)?);
    let kernel_rel = max_rel_err(&y, &ours)?;
    *max = max.max(kernel_rel);
    let pass = kernel_rel <= KERNEL_BAND && rerun;
    println!(
        "depth keys=prefix window_rows={w}/{w} comp_rows={DEPTH_COMP_VISIBLE}/{DEPTH_COMP_ROWS} \
             segments={} kernel_rel={kernel_rel:.2e} rerun={} {}",
        attn::segments(w, DEPTH_COMP_ROWS),
        if rerun { "bit-identical" } else { "DIFFERS" },
        verdict(pass)
    );
    Ok(pass)
}

/// The depth shape with selected rows: one token seeing a full ring and
/// a list of many segments, its visible entries a scrambled set of
/// stream rows, the last visible segment partial and one more wholly
/// past the count; the entries past the count name rows no visible entry
/// names, which hold NaN. Synthetic rows and query and the model's
/// sinks, against our rule, plus a rerun.
///
/// PIN(2026-09-24): no silent failure — a list entry past the stream is
/// refused, not clamped; this case kept its numeric pin with every entry
/// in the stream, and the planted entry moved to [`sel_past_stream_case`].
fn sel_depth_case(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
    sinks: &[f32],
    max: &mut f32,
) -> Result<bool, GateError> {
    let w = set.window;
    let (q, ring) = depth_ring(set);
    let (list, comp) = depth_list(None)?;
    let rows = DEPTH_COMP_ROWS;
    let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
    let (ring_w, comp_w) = (widen(&ring), widen(&comp));
    let sel = Some((&list[..], DEPTH_SEL_STRIDE));
    let ours = rule_all(
        &q,
        &walk_keys(&ring_w, Some(&comp_w), sel, &vis),
        sinks,
        set.scale,
    );
    let inputs = Inputs {
        q: &q,
        ring: &ring,
        comp: Some(&comp),
        sel,
        vis: &vis,
        sinks,
        tokens: 1,
        window: w,
    };
    let mut launch = Launch::new(stream, &inputs, fault)?;
    let y = launch.run(kernels, stream, set.scale)?;
    let rerun = bits_equal(&y, &launch.run(kernels, stream, set.scale)?);
    let kernel_rel = max_rel_err(&y, &ours)?;
    *max = max.max(kernel_rel);
    let pass = kernel_rel <= KERNEL_BAND && rerun;
    println!(
        "depth keys=sel/{DEPTH_SEL_STRIDE} window_rows={w}/{w} comp_rows={DEPTH_COMP_VISIBLE}/{rows} \
             unused_entries={} past_stream_entries=0 segments={} kernel_rel={kernel_rel:.2e} rerun={} {}",
        DEPTH_SEL_STRIDE - DEPTH_COMP_VISIBLE,
        attn::segments(w, DEPTH_SEL_STRIDE),
        if rerun { "bit-identical" } else { "DIFFERS" },
        verdict(pass)
    );
    Ok(pass)
}

/// The depth case's list — a bijection of the stream's rows but the
/// last, scrambled, visible entry [`DEPTH_PAST_ENTRY`] replaced by
/// `planted` when given — and its stream: the rows the visible entries
/// name hold synthetic values, every other row NaN.
fn depth_list(planted: Option<u32>) -> Result<(Vec<u32>, Vec<u16>), GateError> {
    let rows = DEPTH_COMP_ROWS;
    let mut list: Vec<u32> = (0..DEPTH_SEL_STRIDE)
        .map(|j| u32::try_from((j * 7919 + 13) % (rows - 1)))
        .collect::<Result<_, _>>()?;
    if let Some(r) = planted {
        list[DEPTH_PAST_ENTRY] = r;
    }
    let data: Vec<u16> = activations(LATENT, rows, 11)
        .into_iter()
        .map(f32_to_f16_bits)
        .collect();
    let mut comp = vec![NAN_F16; rows * LATENT];
    for &r in &list[..DEPTH_COMP_VISIBLE] {
        let r = r as usize;
        if r < rows {
            comp[r * LATENT..(r + 1) * LATENT].copy_from_slice(&data[r * LATENT..(r + 1) * LATENT]);
        }
    }
    Ok((list, comp))
}

/// The depth case's list with visible entry [`DEPTH_PAST_ENTRY`] naming
/// row `DEPTH_COMP_ROWS` — the first row past the stream. The kernel
/// must raise [`FaultSite::AttnSel`] on the launch's sink, and the word,
/// read back and cleared, must name exactly that. The output is printed
/// against the host's rule with the entry read as the stream's last row
/// (what the kernel used to do), over a stream whose every row holds
/// values — evidence, not a contract: the step that reads this fault
/// back is refused.
fn sel_past_stream_case(
    kernels: &AttnKernels,
    gpu: &Gpu,
    set: &Shape,
    sinks: &[f32],
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let w = set.window;
    let (q, ring) = depth_ring(set);
    let rows = DEPTH_COMP_ROWS;
    let (list, _) = depth_list(Some(u32::try_from(rows)?))?;
    let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
    let mut clamped = list.clone();
    clamped[DEPTH_PAST_ENTRY] = u32::try_from(rows - 1)?;
    // Every row holds values here, so whichever row a kernel reads for the
    // planted entry, the output stays finite and comparable.
    let comp_clamped: Vec<u16> = activations(LATENT, rows, 11)
        .into_iter()
        .map(f32_to_f16_bits)
        .collect();
    let (ring_w, comp_w) = (widen(&ring), widen(&comp_clamped));
    let last_row = rule_all(
        &q,
        &walk_keys(
            &ring_w,
            Some(&comp_w),
            Some((&clamped[..], DEPTH_SEL_STRIDE)),
            &vis,
        ),
        sinks,
        set.scale,
    );
    let inputs = Inputs {
        q: &q,
        ring: &ring,
        comp: Some(&comp_clamped),
        sel: Some((&list[..], DEPTH_SEL_STRIDE)),
        vis: &vis,
        sinks,
        tokens: 1,
        window: w,
    };
    gpu.clear_fault()?;
    let y = Launch::new(stream, &inputs, gpu.unlabelled_sink())?.run(kernels, stream, set.scale)?;
    let got = gpu.take_fault()?;
    let want = Fault::at(LAYER_NONE, FaultSite::AttnSel);
    let pass = got == Some(want);
    println!(
        "depth keys=sel/{DEPTH_SEL_STRIDE} past_stream_entry={rows} (comp_rows {rows}) \
             rel_to_last_row_rule={:.2e} want=attn_sel got={} {}",
        max_rel_err(&y, &last_row)?,
        got.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    Ok(pass)
}

// ------------------------------------------------------------ ring ⧺ staging

/// The staged cases' batch lengths and first positions: every pair.
const STAGE_TOKENS: [usize; 5] = [1, 127, 128, 129, 300];
const STAGE_BASES: [usize; 5] = [0, 5, 127, 128, 300];
/// The staged cases' compressed stream: its ratio, its height — every
/// case's the same, so one captured graph replays a case of any first
/// position — and the list's stride.
const STAGE_RATIO: usize = 4;
const STAGE_COMP_ROWS: usize = 4 * attn::SEG_KEYS;
const STAGE_SEL_STRIDE: usize = 64;
/// Staging rows past the batch's, NaN.
const STAGE_SPARE: usize = 2;
/// The batch length the staged graph replays at every first position.
const STAGE_REPLAY_TOKENS: usize = 129;

/// The compressed rows a staged case reads besides its window.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StageKeys {
    Window,
    Prefix,
    Sel,
}

impl StageKeys {
    const ALL: [StageKeys; 3] = [StageKeys::Window, StageKeys::Prefix, StageKeys::Sel];

    fn name(self) -> String {
        match self {
            StageKeys::Window => "window".to_string(),
            StageKeys::Prefix => "prefix".to_string(),
            StageKeys::Sel => format!("sel/{STAGE_SEL_STRIDE}"),
        }
    }
}

/// One staged case's host inputs: `tokens` tokens at positions `base ..`
/// over the model's window, every position's latent row synthetic.
struct StageCase {
    tokens: usize,
    base: usize,
    keys: StageKeys,
    window: usize,
    heads: usize,
    /// `[token][head][LATENT]` f32.
    q: Vec<f32>,
    /// Position `p`'s latent row, f16 bits, for `p < base + tokens`.
    rows: Vec<u16>,
    /// The ring before the batch: the last `min(base, window)` positions
    /// below `base` at their slots, NaN in every other slot.
    ring: Vec<u16>,
    /// The batch's rows, row `j` position `base + j`, then
    /// [`STAGE_SPARE`] NaN rows.
    staging: Vec<u16>,
    /// [`STAGE_COMP_ROWS`] stream rows: those below the batch's last
    /// compressed count hold values, the rest NaN.
    stream: Option<Vec<u16>>,
    /// `tokens` lists of [`STAGE_SEL_STRIDE`] entries: the visible ones
    /// a scramble of the token's visible rows, the rest naming NaN rows.
    sel: Option<Vec<u32>>,
    /// Per token: `min(pos + 1, window)` window keys, then its compressed
    /// rows.
    vis: Vec<u32>,
}

impl StageCase {
    fn new(
        set: &Shape,
        tokens: usize,
        base: usize,
        keys: StageKeys,
    ) -> Result<StageCase, GateError> {
        let (w, heads, end) = (set.window, set.heads, base + tokens);
        let f16 = |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(f32_to_f16_bits).collect() };
        let rows = f16(activations(LATENT, end, 17));
        let mut ring = vec![NAN_F16; w * LATENT];
        for p in base.saturating_sub(w)..base {
            let s = p % w;
            ring[s * LATENT..(s + 1) * LATENT].copy_from_slice(row_of(&rows, p));
        }
        let mut staging = vec![NAN_F16; (tokens + STAGE_SPARE) * LATENT];
        staging[..tokens * LATENT].copy_from_slice(&rows[base * LATENT..end * LATENT]);
        let visible = |t: usize| (base + t + 1) / STAGE_RATIO;
        let written = visible(tokens - 1);
        if written >= STAGE_COMP_ROWS {
            return Err(format!(
                "a staged case of {tokens} tokens from {base}: {written} compressed rows, the \
                     stream holds {STAGE_COMP_ROWS} with at least one NaN row"
            )
            .into());
        }
        let stream = (keys != StageKeys::Window).then(|| {
            let mut s = vec![NAN_F16; STAGE_COMP_ROWS * LATENT];
            s[..written * LATENT].copy_from_slice(&f16(activations(LATENT, written, 29)));
            s
        });
        let sel = (keys == StageKeys::Sel)
            .then(|| {
                (0..tokens)
                    .flat_map(|t| {
                        let n = visible(t);
                        let live = n.min(STAGE_SEL_STRIDE);
                        // 7919 is prime and n below it, so the live
                        // entries are distinct rows.
                        (0..STAGE_SEL_STRIDE).map(move |j| {
                            if j < live {
                                (j * 7919 + 13 + t) % n
                            } else {
                                written + j % (STAGE_COMP_ROWS - written)
                            }
                        })
                    })
                    .map(u32::try_from)
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let vis = (0..tokens)
            .flat_map(|t| {
                let comp = match keys {
                    StageKeys::Window => 0,
                    StageKeys::Prefix => visible(t),
                    StageKeys::Sel => visible(t).min(STAGE_SEL_STRIDE),
                };
                [(base + t + 1).min(w), comp]
            })
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(StageCase {
            tokens,
            base,
            keys,
            window: w,
            heads,
            q: activations(LATENT, tokens * heads, 23),
            rows,
            ring,
            staging,
            stream,
            sel,
            vis,
        })
    }

    /// The ring after the whole batch is in it, the state a ring-only launch
    /// reads: every batch position at its slot. The batch must not wrap the
    /// window.
    fn full_ring(&self) -> Vec<u16> {
        assert!(
            self.tokens <= self.window,
            "full_ring: a batch of {} tokens wraps a window of {}",
            self.tokens,
            self.window
        );
        let mut ring = self.ring.clone();
        for p in self.base..self.base + self.tokens {
            let s = p % self.window;
            ring[s * LATENT..(s + 1) * LATENT].copy_from_slice(row_of(&self.rows, p));
        }
        ring
    }

    /// The launch inputs of tokens `t0 .. t0 + n` over `ring`.
    fn inputs<'a>(&'a self, sinks: &'a [f32], ring: &'a [u16], t0: usize, n: usize) -> Inputs<'a> {
        let per_token = self.heads * LATENT;
        let k = STAGE_SEL_STRIDE;
        Inputs {
            q: &self.q[t0 * per_token..(t0 + n) * per_token],
            ring,
            comp: self.stream.as_deref(),
            sel: self.sel.as_ref().map(|s| (&s[t0 * k..(t0 + n) * k], k)),
            vis: &self.vis[2 * t0..2 * (t0 + n)],
            sinks,
            tokens: n,
            window: self.window,
        }
    }
}

/// A staged launch's device buffers: a [`Launch`] of the whole batch over
/// the ring before it, the staging rows and the batch's first position.
struct StageLaunch {
    launch: Launch,
    staging: DeviceTensor<u16>,
    base: DeviceBuffer<u32>,
}

impl StageLaunch {
    fn new(
        stream: &CudaStream,
        c: &StageCase,
        sinks: &[f32],
        fault: FaultSink,
    ) -> Result<StageLaunch, GateError> {
        Ok(StageLaunch {
            launch: Launch::new(stream, &c.inputs(sinks, &c.ring, 0, c.tokens), fault)?,
            staging: DeviceTensor::upload(stream, &c.staging, c.staging.len() / LATENT, LATENT)?,
            base: DeviceBuffer::from_host(stream, &[u32::try_from(c.base)?])?,
        })
    }

    /// The staged attention, then the commit.
    fn enqueue(
        &mut self,
        kernels: &AttnKernels,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<(), GpuError> {
        let staged = Staged {
            rows: &self.staging,
            base: &self.base,
        };
        kernels.enqueue_staged(stream, self.launch.args(scale), staged)?;
        kernels.enqueue_commit(
            stream,
            CommitArgs {
                staged,
                tokens: self.launch.tokens,
                ring: &mut self.launch.window,
            },
        )
    }

    /// One eager run on NaN-filled partials and output: the outputs and
    /// the ring after the commit.
    fn run(
        &mut self,
        kernels: &AttnKernels,
        stream: &CudaStream,
        scale: f32,
    ) -> Result<(Vec<f32>, Vec<u16>), GateError> {
        self.launch.poison(stream)?;
        self.enqueue(kernels, stream, scale)?;
        stream.synchronize()?;
        Ok((
            self.launch.y.to_host_vec(stream)?,
            self.launch.window.buf().to_host_vec(stream)?,
        ))
    }

    /// `c`'s inputs into these buffers, of the same shapes: every buffer
    /// the launches read.
    fn load(&mut self, stream: &CudaStream, c: &StageCase) -> Result<(), GateError> {
        let l = &mut self.launch;
        l.q.copy_from_host(stream, &c.q)?;
        l.vis.copy_from_host(stream, &c.vis)?;
        l.window.buf_mut().copy_from_host(stream, &c.ring)?;
        if let (Some(buf), Some(s)) = (l.comp.as_mut(), c.stream.as_ref()) {
            buf.buf_mut().copy_from_host(stream, s)?;
        }
        if let (Some((buf, _)), Some(s)) = (l.sel.as_mut(), c.sel.as_ref()) {
            buf.copy_from_host(stream, s)?;
        }
        self.staging.buf_mut().copy_from_host(stream, &c.staging)?;
        self.base
            .copy_from_host(stream, &[u32::try_from(c.base)?])?;
        Ok(())
    }
}

/// `c` token by token, the decode step's rule: each token writes its
/// row into its ring slot, then one ring-only launch. Every token's
/// output in order, and the ring the last token leaves.
fn stage_sequence(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
    sinks: &[f32],
    c: &StageCase,
) -> Result<(Vec<f32>, Vec<u16>), GateError> {
    let (w, per_token, k) = (c.window, c.heads * LATENT, STAGE_SEL_STRIDE);
    let mut ring = c.ring.clone();
    let mut launch = Launch::new(stream, &c.inputs(sinks, &ring, 0, 1), fault)?;
    let mut y = Vec::with_capacity(c.tokens * per_token);
    for t in 0..c.tokens {
        let (pos, s) = (c.base + t, (c.base + t) % w);
        ring[s * LATENT..(s + 1) * LATENT].copy_from_slice(row_of(&c.rows, pos));
        launch.window.buf_mut().copy_from_host(stream, &ring)?;
        launch
            .q
            .copy_from_host(stream, &c.q[t * per_token..(t + 1) * per_token])?;
        launch
            .vis
            .copy_from_host(stream, &c.vis[2 * t..2 * t + 2])?;
        if let (Some((buf, _)), Some(sel)) = (launch.sel.as_mut(), c.sel.as_ref()) {
            buf.copy_from_host(stream, &sel[t * k..(t + 1) * k])?;
        }
        y.extend(launch.run(kernels, stream, set.scale)?);
    }
    Ok((y, launch.window.buf().to_host_vec(stream)?))
}

/// One staged case: the staged launch and the commit against
/// [`stage_sequence`] — every token's output bit for bit and the ring
/// byte for byte, the reference finite (it reads no NaN row).
fn stage_case(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
    sinks: &[f32],
    c: &StageCase,
) -> Result<bool, GateError> {
    let per_token = c.heads * LATENT;
    let (want, want_ring) = stage_sequence(kernels, stream, fault, set, sinks, c)?;
    let (y, ring) = StageLaunch::new(stream, c, sinks, fault)?.run(kernels, stream, set.scale)?;
    let same = (0..c.tokens)
        .filter(|&t| {
            let r = t * per_token..(t + 1) * per_token;
            bits_equal(&y[r.clone()], &want[r])
        })
        .count();
    let finite = want.iter().all(|v| v.is_finite());
    let ring_same = ring == want_ring;
    let pass = same == c.tokens && ring_same && finite;
    println!(
        "stage keys={} tokens={} base={} window={}: outputs bit-identical={same}/{} ring={} \
             reference {} {}",
        c.keys.name(),
        c.tokens,
        c.base,
        c.window,
        c.tokens,
        if ring_same { "identical" } else { "DIFFERS" },
        if finite { "finite" } else { "NOT FINITE" },
        verdict(pass)
    );
    Ok(pass)
}

/// Every staged case: each compressed source, batch length and first
/// position.
fn stage_cases(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
    sinks: &[f32],
) -> Result<bool, GateError> {
    let (mut n, mut passed) = (0usize, 0usize);
    for keys in StageKeys::ALL {
        for &tokens in &STAGE_TOKENS {
            for &base in &STAGE_BASES {
                let c = StageCase::new(set, tokens, base, keys)?;
                n += 1;
                passed += usize::from(stage_case(kernels, stream, fault, set, sinks, &c)?);
            }
        }
    }
    let pass = passed == n;
    println!("stage cases={n} pass={passed} {}", verdict(pass));
    Ok(pass)
}

/// One graph captured with a staged launch and its commit at the first
/// of [`STAGE_BASES`] and replayed at each, every input rewritten in
/// place between replays: outputs and ring bit-identical to an eager
/// run of the same case. A grid or a slot taken from the first position
/// would freeze the captured one.
fn stage_graph_replay(
    gpu: &Gpu,
    kernels: &AttnKernels,
    set: &Shape,
    sinks: &[f32],
) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let t = STAGE_REPLAY_TOKENS;
    let cases = STAGE_BASES
        .iter()
        .map(|&b| StageCase::new(set, t, b, StageKeys::Sel))
        .collect::<Result<Vec<_>, _>>()?;
    let mut launch = StageLaunch::new(stream, &cases[0], sinks, gpu.unlabelled_sink())?;
    let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
    let mut same = 0;
    for c in &cases {
        let (want, want_ring) = StageLaunch::new(stream, c, sinks, gpu.unlabelled_sink())?
            .run(kernels, stream, set.scale)?;
        launch.load(stream, c)?;
        launch.launch.poison(stream)?;
        graph.launch(stream)?;
        stream.synchronize()?;
        let y = launch.launch.y.to_host_vec(stream)?;
        let ring = launch.launch.window.buf().to_host_vec(stream)?;
        same += usize::from(bits_equal(&y, &want) && ring == want_ring);
    }
    let pass = same == cases.len();
    println!(
        "graph keys={} staged tokens={t} bases={} nodes={} bit-identical={same}/{} {}",
        StageKeys::Sel.name(),
        STAGE_BASES
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
        graph.node_count(),
        cases.len(),
        verdict(pass)
    );
    Ok(pass)
}

/// Each shape the staged enqueue and the commit refuse: the call returns
/// the shape error whose text names the cause, and nothing launches —
/// the output stays the NaN it was filled with and the ring as uploaded.
fn stage_refusals(
    kernels: &AttnKernels,
    stream: &CudaStream,
    fault: FaultSink,
    set: &Shape,
) -> Result<bool, GateError> {
    let c = StageCase::new(set, 2, 5, StageKeys::Window)?;
    let sinks = vec![0.0f32; set.heads];
    let mut l = StageLaunch::new(stream, &c, &sinks, fault)?;
    l.launch.poison(stream)?;
    let short = DeviceTensor::<u16>::zeroed(stream, 1, LATENT)?;
    let narrow = DeviceTensor::<u16>::zeroed(stream, 2, LATENT / 2)?;
    let no_word = DeviceBuffer::<u32>::zeroed(stream, 0)?;
    let mut no_ring = DeviceTensor::<u16>::zeroed(stream, 0, LATENT)?;
    let scale = set.scale;
    let StageLaunch {
        launch,
        staging,
        base,
    } = &mut l;
    let (staging, base) = (&*staging, &*base);
    let mut got: Vec<(&str, &str, Result<(), GpuError>)> = vec![
        (
            "staged: staging rows fewer than the tokens",
            "staging holds 1 rows, the batch's 2 tokens",
            kernels.enqueue_staged(stream, launch.args(scale), Staged { rows: &short, base }),
        ),
        (
            "staged: staging rows not LATENT wide",
            "staging rows are 256 wide",
            kernels.enqueue_staged(
                stream,
                launch.args(scale),
                Staged {
                    rows: &narrow,
                    base,
                },
            ),
        ),
        (
            "staged: a base buffer of no word",
            "the base buffer holds no word",
            kernels.enqueue_staged(
                stream,
                launch.args(scale),
                Staged {
                    rows: staging,
                    base: &no_word,
                },
            ),
        ),
    ];
    let mut a = launch.args(scale);
    a.window = &no_ring;
    got.push((
        "staged: a ring of no rows",
        "over a ring of no rows",
        kernels.enqueue_staged(
            stream,
            a,
            Staged {
                rows: staging,
                base,
            },
        ),
    ));
    /// One refused commit: its name, the text its error must hold, the
    /// rows and base it stages, its tokens, and whether it commits into
    /// the case's ring (else into a ring of no rows).
    struct Commit<'a> {
        name: &'static str,
        want: &'static str,
        rows: &'a DeviceTensor<u16>,
        base: &'a DeviceBuffer<u32>,
        tokens: usize,
        own_ring: bool,
    }
    let commits = [
        Commit {
            name: "commit: no token",
            want: "a batch of 0 tokens",
            rows: staging,
            base,
            tokens: 0,
            own_ring: true,
        },
        Commit {
            name: "commit: a ring of no rows",
            want: "into a ring of 0 rows",
            rows: staging,
            base,
            tokens: 2,
            own_ring: false,
        },
        Commit {
            name: "commit: staging rows fewer than the tokens",
            want: "staging holds 1 rows, the batch's 2 tokens",
            rows: &short,
            base,
            tokens: 2,
            own_ring: true,
        },
        Commit {
            name: "commit: staging rows not LATENT wide",
            want: "staging rows are 256 wide",
            rows: &narrow,
            base,
            tokens: 2,
            own_ring: true,
        },
        Commit {
            name: "commit: a base buffer of no word",
            want: "the base buffer holds no word",
            rows: staging,
            base: &no_word,
            tokens: 2,
            own_ring: true,
        },
    ];
    for Commit {
        name,
        want,
        rows,
        base,
        tokens,
        own_ring,
    } in commits
    {
        let ring = if own_ring {
            &mut launch.window
        } else {
            &mut no_ring
        };
        let r = kernels.enqueue_commit(
            stream,
            CommitArgs {
                staged: Staged { rows, base },
                tokens,
                ring,
            },
        );
        got.push((name, want, r));
    }
    let mut pass = true;
    for (name, want, r) in &got {
        let (ok, text) = match r {
            Err(GpuError::Shape { detail, .. }) => (detail.contains(want), detail.clone()),
            Err(e) => (false, format!("not a shape error: {e}")),
            Ok(()) => (false, "accepted".to_string()),
        };
        pass &= ok;
        println!("refuse {name}: \"{text}\" names \"{want}\" {}", verdict(ok));
    }
    stream.synchronize()?;
    let quiet = launch.y.to_host_vec(stream)?.iter().all(|v| v.is_nan())
        && launch.window.buf().to_host_vec(stream)? == c.ring;
    println!(
        "refuse: {} refusals, output and ring untouched {}",
        got.len(),
        verdict(quiet)
    );
    Ok(pass && quiet)
}
