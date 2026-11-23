//! The Inkling 2 session's buffers: the weights as the kernels read them, each
//! visit's stores and the step's scratch, every one made at the open through
//! the residency set's [`Alloc`], in the order [`Checked::bytes`] tallies them.
//!
//! What the load does to a stored tensor, and nothing else:
//! - a BF16 projection (the sink rows, the dynamic taps' rows) is widened to
//!   f32, exactly;
//! - the relative-bias table, stored `extent` rows of `d_rel` values, is
//!   transposed to the kernel's `d_rel` rows of `extent` distances;
//! - the loop's message queries, stored as two tables of two rows a core layer
//!   (q and ffn; k and v), become one table of four rows a core layer in the
//!   order the controllers read them: q, k, v, ffn;
//! - a folder's NVFP4 stack (codes and block scales in two planes an expert)
//!   keeps its two planes, the codes plane then the scales plane an expert; a
//!   folder's MXFP8 stack is interleaved into the rows layout, each group's
//!   scale byte before its codes; a folder's fused gate·up stack, whose rows
//!   alternate gate and up, is permuted to the GGUF's order (the gate rows
//!   `[0, ff)`, then the up rows `[ff, 2·ff)`, the order `pair_situ` reads)
//!   in both planes; a GGUF's stack loads as stored;
//! - the head keeps the rows of the file's vocabulary cut.

use bloomery_device::Elems;
use gguf::quant::{GgmlType, dequant_row};
use gguf::{Place, Split};
use model_hf::Weights;

use super::check::{CONV_CHANNELS, Checked, LayerPlan, Site, Stack};
use super::session::{SessionError, refuse};
use crate::buffer::MetalBuffer;
use crate::fault::MetalFaultWord;
use crate::kernels::NvfpLayout;
use crate::residency::Alloc;

/// One stored layer's weights.
pub struct LayerBufs {
    /// The q, relative and gate rows, one buffer in that order.
    pub qrg: MetalBuffer<u8>,
    /// The sink rows, widened to f32.
    pub sink: MetalBuffer<u8>,
    pub k: MetalBuffer<u8>,
    pub v: MetalBuffer<u8>,
    pub k_taps: Option<MetalBuffer<u8>>,
    pub v_taps: Option<MetalBuffer<u8>>,
    pub wo: MetalBuffer<u8>,
    /// The q, k and v paths' input norms: one gain on a windowed layer.
    pub norms: Vec<MetalBuffer<f32>>,
    pub attn_query: Option<MetalBuffer<f32>>,
    pub ffn_query: MetalBuffer<f32>,
    pub ffn_norm: MetalBuffer<f32>,
    pub q_clip: MetalBuffer<f32>,
    pub rel_clip: MetalBuffer<f32>,
    /// `d_rel` rows of `extent` distances.
    pub rel_bias: MetalBuffer<f32>,
    pub convs: [MetalBuffer<f32>; 4],
    pub router: MetalBuffer<f32>,
    pub router_bias: MetalBuffer<f32>,
    pub gate_up: MetalBuffer<u8>,
    pub down: MetalBuffer<u8>,
    /// Each expert's factor of the gate·up and down results, one an expert:
    /// the stack's global scale, 1 for a form that has none.
    pub gate_up_scale: MetalBuffer<f32>,
    pub down_scale: MetalBuffer<f32>,
    /// The core layer's controllers, in the order q, k, v, ffn: each its down,
    /// gain and forget rows.
    pub controllers: Vec<[MetalBuffer<f32>; 3]>,
}

/// One visit's stores.
pub struct VisitBufs {
    pub cache_k: MetalBuffer<u16>,
    pub cache_v: MetalBuffer<u16>,
    /// The four convs' last `taps − 1` inputs, [`CONV_CHANNELS`] a row.
    pub ring: MetalBuffer<f32>,
}

/// The step's scratch.
pub struct Scratch {
    /// The bank's blocks, then the stream's row: row `i` at `i · hidden`.
    pub rows: MetalBuffer<f32>,
    /// The message banks, one a pass the loop runs again, `message_rows`
    /// rows each: that pass's blocks past its first, then its stream. The
    /// step zeroes them and each pass's end adds its rows into its own.
    pub message: MetalBuffer<f32>,
    /// The attention's mixes, one a query row.
    pub amix: MetalBuffer<f32>,
    /// The loop's messages: q, k, v, ffn.
    pub msgs: MetalBuffer<f32>,
    /// The q, k and v paths' inputs after their controllers.
    pub xin: MetalBuffer<f32>,
    /// The four controllers' down rows, q, k, v, ffn: each the rank's values.
    pub lc_h: MetalBuffer<f32>,
    /// The q, relative and gate rows.
    pub qrg: MetalBuffer<f32>,
    pub sinks: MetalBuffer<f32>,
    pub k_raw: MetalBuffer<f32>,
    pub v_raw: MetalBuffer<f32>,
    pub k_taps: MetalBuffer<f32>,
    pub v_taps: MetalBuffer<f32>,
    /// The convs' outputs, clipped and normed in place: `Kcur`, `Vcur`.
    pub k: MetalBuffer<f32>,
    pub v: MetalBuffer<f32>,
    /// The attention's output, before its gate.
    pub att: MetalBuffer<f32>,
    pub gated: MetalBuffer<f32>,
    pub partials: MetalBuffer<f32>,
    /// `wo`'s output and the post-conv's.
    pub ao: MetalBuffer<f32>,
    pub ac: MetalBuffer<f32>,
    /// The FFN's mix, its controller's output and its norm.
    pub fmix: MetalBuffer<f32>,
    pub fin: MetalBuffer<f32>,
    pub fnorm: MetalBuffer<f32>,
    pub router_partials: MetalBuffer<f32>,
    pub router_logits: MetalBuffer<f32>,
    pub route_weights: MetalBuffer<f32>,
    /// Each slot's gate·up rows, activation and down rows.
    pub gate_up: MetalBuffer<f32>,
    pub act: MetalBuffer<f32>,
    pub down: MetalBuffer<f32>,
    /// The experts' sums, a row a visit: the step zeroes them and each
    /// visit's combine adds its slots into its own row. Then the post-conv's
    /// output.
    pub moe: MetalBuffer<f32>,
    pub moe_conv: MetalBuffer<f32>,
    /// The head's mix.
    pub head_mix: MetalBuffer<f32>,
    /// The cut's logits.
    pub logits: MetalBuffer<f32>,
    pub ticket: MetalBuffer<u32>,
    pub route_ids: MetalBuffer<u32>,
    /// The identity map the combine reads its slots through.
    pub slots: MetalBuffer<u32>,
    pub step_id: MetalBuffer<u32>,
}

/// Every buffer of a session.
pub struct Arena {
    pub layers: Vec<LayerBufs>,
    pub visits: Vec<VisitBufs>,
    pub scratch: Scratch,
    /// The embedding table as stored: the host reads a token's row.
    pub embed: MetalBuffer<u8>,
    pub embed_norm: Vec<f32>,
    pub head: MetalBuffer<u8>,
    pub output_norm: MetalBuffer<f32>,
    pub output_query: MetalBuffer<f32>,
    /// Four query rows a core layer: q, k, v, ffn.
    pub loop_queries: MetalBuffer<f32>,
    pub faults: MetalFaultWord,
}

/// The bytes of tensor `name`, its runs joined.
fn raw(weights: &Weights<'_>, name: &str) -> Result<Vec<u8>, SessionError> {
    let part = weights
        .bytes(name)
        .map_err(|e| refuse(format!("{name}: {e}")))?;
    Ok(part.spans().concat())
}

/// The f32 values of the F32 tensor `name`.
fn floats(weights: &Weights<'_>, name: &str) -> Result<Vec<f32>, SessionError> {
    let bytes = raw(weights, name)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// A buffer of `host`'s values.
fn filled<T: bloomery_device::Elem>(
    alloc: Alloc<'_>,
    host: &[T],
) -> Result<MetalBuffer<T>, SessionError> {
    let mut b = alloc.buffer::<T>(Elems(host.len()))?;
    b.write(host)?;
    Ok(b)
}

/// The f32 tensor `name` as a buffer.
fn f32_buf(
    alloc: Alloc<'_>,
    weights: &Weights<'_>,
    name: &str,
) -> Result<MetalBuffer<f32>, SessionError> {
    filled(alloc, &floats(weights, name)?)
}

/// The projection `site`'s bytes in its launch type: a BF16 or F32 site's
/// values as f32, any other as stored.
fn site_bytes(weights: &Weights<'_>, site: &Site) -> Result<Vec<u8>, SessionError> {
    let bytes = raw(weights, &site.name)?;
    let want = match site.stored {
        GgmlType::BF16 => site.rows * site.cols * 2,
        _ => site.bytes(),
    };
    if bytes.len() != want {
        return Err(refuse(format!(
            "{}: {} bytes, the shape's {want}",
            site.name,
            bytes.len()
        )));
    }
    if site.stored != GgmlType::BF16 {
        return Ok(bytes);
    }
    let mut wide = vec![0.0f32; site.rows * site.cols];
    dequant_row(GgmlType::BF16, &bytes, &mut wide)
        .map_err(|e| refuse(format!("{}: {e}", site.name)))?;
    Ok(wide.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// The projection `site` as a buffer.
fn site_buf(
    alloc: Alloc<'_>,
    weights: &Weights<'_>,
    site: &Site,
) -> Result<MetalBuffer<u8>, SessionError> {
    filled(alloc, &site_bytes(weights, site)?)
}

/// The source row of the launch's row `j` of a stack of `rows` rows: the
/// folder's fused gate·up rows alternate gate and up, and the launch reads
/// the gate rows first.
fn gate_up_row(j: usize, rows: usize) -> usize {
    let ff = rows / 2;
    if j < ff { 2 * j } else { 2 * (j - ff) + 1 }
}

/// Expert `e` of the folder's stack `st` in its launch layout, the rows in
/// the launch's order (`fused`: the gate·up permutation): an NVFP4 stack's
/// two planes, every row's codes then every row's scale bytes; an MXFP8
/// stack's rows, a group's scale byte then its codes.
fn expert_layout(
    st: &Stack,
    codes: &[u8],
    scales: &[u8],
    fused: bool,
    out: &mut Vec<u8>,
) -> Result<(), SessionError> {
    let (group, code_bytes) = match st.layout {
        NvfpLayout::Planes => (16, 8),
        NvfpLayout::Mxfp8Rows => (32, 32),
        other => {
            return Err(refuse(format!(
                "{}: a folder's planes in the {other:?} layout",
                st.name
            )));
        }
    };
    let groups = st.k / group;
    if codes.len() != st.rows * groups * code_bytes || scales.len() != st.rows * groups {
        return Err(refuse(format!(
            "{}: an expert's planes are {} and {} bytes, the shape's {} and {}",
            st.name,
            codes.len(),
            scales.len(),
            st.rows * groups * code_bytes,
            st.rows * groups
        )));
    }
    let row = |j: usize| if fused { gate_up_row(j, st.rows) } else { j };
    if st.layout == NvfpLayout::Planes {
        let (cb, sb) = (groups * code_bytes, groups);
        for j in 0..st.rows {
            let r = row(j);
            out.extend_from_slice(&codes[r * cb..(r + 1) * cb]);
        }
        for j in 0..st.rows {
            let r = row(j);
            out.extend_from_slice(&scales[r * sb..(r + 1) * sb]);
        }
        return Ok(());
    }
    for j in 0..st.rows {
        let r = row(j);
        for g in 0..groups {
            out.push(scales[r * groups + g]);
            let at = (r * groups + g) * code_bytes;
            out.extend_from_slice(&codes[at..at + code_bytes]);
        }
    }
    Ok(())
}

/// The stack `st` as a buffer, every expert's rows in the launch's layout, and
/// each expert's factor.
fn stack_buf(
    alloc: Alloc<'_>,
    split: &Split,
    weights: &Weights<'_>,
    st: &Stack,
    fused: bool,
) -> Result<(MetalBuffer<u8>, MetalBuffer<f32>), SessionError> {
    let per = st.expert_bytes();
    let mut bytes = Vec::with_capacity(st.experts * per);
    let scales = if st.planes {
        let Some(Place::Stack { scales, .. }) = split.place(&st.name) else {
            return Err(refuse(format!("{}: no stack of planes", st.name)));
        };
        for e in 0..st.experts {
            let m = split
                .matrix(&st.name, e)
                .map_err(|e| refuse(format!("{}: {e}", st.name)))?;
            expert_layout(st, m.codes, m.block_scales, fused, &mut bytes)?;
        }
        match st.layout {
            NvfpLayout::Planes if scales.len() == st.experts => scales.clone(),
            NvfpLayout::Planes => {
                return Err(refuse(format!(
                    "{}: {} expert scales for {} experts",
                    st.name,
                    scales.len(),
                    st.experts
                )));
            }
            _ => vec![1.0; st.experts],
        }
    } else {
        bytes = raw(weights, &st.name)?;
        match &st.scale {
            // The scale rows are the stack's, not weights of their own: read
            // where the file keeps them.
            Some(name) => split
                .data(name)
                .map_err(|e| refuse(format!("{name}: {e}")))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            None => vec![1.0; st.experts],
        }
    };
    if bytes.len() != st.experts * per {
        return Err(refuse(format!(
            "{}: {} bytes, {} experts of {per} in the {:?} layout",
            st.name,
            bytes.len(),
            st.experts,
            st.layout
        )));
    }
    if let Some(bad) = scales.iter().find(|s| !(s.is_finite() && **s > 0.0)) {
        return Err(refuse(format!("{}: an expert's scale is {bad}", st.name)));
    }
    let stack = filled(alloc, &bytes)?;
    Ok((stack, filled(alloc, &scales)?))
}

/// The relative-bias table `name`, stored `extent` rows of `d_rel` values, as
/// `d_rel` rows of `extent` distances.
fn rel_bias(
    alloc: Alloc<'_>,
    weights: &Weights<'_>,
    name: &str,
) -> Result<MetalBuffer<f32>, SessionError> {
    const D_REL: usize = 16;
    const EXTENT: usize = 1024;
    let stored = floats(weights, name)?;
    let mut t = vec![0.0f32; D_REL * EXTENT];
    for dist in 0..EXTENT {
        for d in 0..D_REL {
            t[d * EXTENT + dist] = stored[dist * D_REL + d];
        }
    }
    filled(alloc, &t)
}

/// Layer `l`'s weights.
fn layer(
    alloc: Alloc<'_>,
    split: &Split,
    weights: &Weights<'_>,
    p: &LayerPlan,
    l: usize,
) -> Result<LayerBufs, SessionError> {
    let qrg: Vec<u8> = p
        .qrg
        .iter()
        .map(|s| site_bytes(weights, s))
        .collect::<Result<Vec<_>, _>>()?
        .concat();
    let norms = if p.windowed {
        vec![f32_buf(alloc, weights, &p.norms[0])?]
    } else {
        p.norms
            .iter()
            .map(|n| f32_buf(alloc, weights, n))
            .collect::<Result<_, _>>()?
    };
    let opt = |s: &Option<Site>| s.as_ref().map(|s| site_buf(alloc, weights, s)).transpose();
    let (gate_up, gate_up_scale) = stack_buf(alloc, split, weights, &p.gate_up, true)?;
    let (down, down_scale) = stack_buf(alloc, split, weights, &p.down, false)?;
    let mut controllers = Vec::new();
    if p.core {
        for m in ["attn", "k", "v", "mlp"] {
            let n = |s: &str| format!("blk.{l}.loop_{m}_{s}.weight");
            controllers.push([
                f32_buf(alloc, weights, &n("down"))?,
                f32_buf(alloc, weights, &n("gain"))?,
                f32_buf(alloc, weights, &n("forget"))?,
            ]);
        }
    }
    Ok(LayerBufs {
        qrg: filled(alloc, &qrg)?,
        sink: site_buf(alloc, weights, &p.sink)?,
        k: site_buf(alloc, weights, &p.k)?,
        v: site_buf(alloc, weights, &p.v)?,
        k_taps: opt(&p.k_taps)?,
        v_taps: opt(&p.v_taps)?,
        wo: site_buf(alloc, weights, &p.wo)?,
        norms,
        attn_query: p
            .attn_query
            .as_ref()
            .map(|(n, _)| f32_buf(alloc, weights, n))
            .transpose()?,
        ffn_query: f32_buf(alloc, weights, &p.ffn_query)?,
        ffn_norm: f32_buf(alloc, weights, &p.ffn_norm)?,
        q_clip: f32_buf(alloc, weights, &p.q_clip)?,
        rel_clip: f32_buf(alloc, weights, &p.rel_clip)?,
        rel_bias: rel_bias(alloc, weights, &p.rel_bias)?,
        convs: [
            f32_buf(alloc, weights, &p.convs[0])?,
            f32_buf(alloc, weights, &p.convs[1])?,
            f32_buf(alloc, weights, &p.convs[2])?,
            f32_buf(alloc, weights, &p.convs[3])?,
        ],
        router: f32_buf(alloc, weights, &p.router)?,
        router_bias: f32_buf(alloc, weights, &p.router_bias)?,
        gate_up,
        down,
        gate_up_scale,
        down_scale,
        controllers,
    })
}

impl Arena {
    /// Every buffer of a session of the checked file `c`.
    ///
    /// # Errors
    /// A buffer the driver refuses, or a tensor whose bytes are not its
    /// shape's, by name.
    pub fn open(
        alloc: Alloc<'_>,
        split: &Split,
        weights: &Weights<'_>,
        c: &Checked,
    ) -> Result<Arena, SessionError> {
        let layers = c
            .layers
            .iter()
            .enumerate()
            .map(|(l, p)| layer(alloc, split, weights, p, l))
            .collect::<Result<Vec<_>, _>>()?;
        let embed = site_buf(alloc, weights, &c.embed)?;
        let head_bytes = site_bytes(weights, &c.head)?;
        let head = filled(alloc, &head_bytes[..c.head.bytes_of(c.cut)])?;
        let n = c.hidden;
        let mut globals = Vec::new();
        for name in ["output_norm.weight", "output_res_query.weight"] {
            globals.push(f32_buf(alloc, weights, name)?);
        }
        let embed_norm = floats(weights, "token_embd_norm.weight")?;
        // Four rows a core layer: q (the attention table's even row), k and v
        // (the k-and-v table's two rows), ffn (the attention table's odd row).
        let attn = floats(weights, "loop_res_query.weight")?;
        let kv = floats(weights, "loop_kv_res_query.weight")?;
        let core = c.core.1 - c.core.0;
        let mut queries = Vec::with_capacity(core * 4 * n);
        for i in 0..core {
            for (table, row) in [
                (&attn, 2 * i),
                (&kv, 2 * i),
                (&kv, 2 * i + 1),
                (&attn, 2 * i + 1),
            ] {
                queries.extend_from_slice(&table[row * n..(row + 1) * n]);
            }
        }
        let loop_queries = filled(alloc, &queries)?;
        let visits = c
            .visits
            .iter()
            .map(|v| {
                let plane = 8 * c.plane_rows(v.windowed) * 128;
                Ok(VisitBufs {
                    cache_k: alloc.buffer(Elems(plane))?,
                    cache_v: alloc.buffer(Elems(plane))?,
                    ring: alloc.buffer(Elems((c.taps - 1) * CONV_CHANNELS))?,
                })
            })
            .collect::<Result<Vec<_>, SessionError>>()?;
        let sizes = c.scratch_f32s();
        let mut f = sizes.iter().map(|&s| alloc.buffer::<f32>(Elems(s)));
        let mut next = || {
            f.next()
                .expect("the scratch list names every buffer the arena makes")
                .map_err(SessionError::from)
        };
        let scratch = Scratch {
            rows: next()?,
            message: next()?,
            amix: next()?,
            msgs: next()?,
            xin: next()?,
            lc_h: next()?,
            qrg: next()?,
            sinks: next()?,
            k_raw: next()?,
            v_raw: next()?,
            k_taps: next()?,
            v_taps: next()?,
            k: next()?,
            v: next()?,
            att: next()?,
            gated: next()?,
            partials: next()?,
            ao: next()?,
            ac: next()?,
            fmix: next()?,
            fin: next()?,
            fnorm: next()?,
            router_partials: next()?,
            router_logits: next()?,
            route_weights: next()?,
            gate_up: next()?,
            act: next()?,
            down: next()?,
            moe: next()?,
            moe_conv: next()?,
            head_mix: next()?,
            logits: next()?,
            ticket: filled(alloc, &[0u32])?,
            route_ids: alloc.buffer(Elems(8))?,
            slots: filled(alloc, &(0..8u32).collect::<Vec<_>>())?,
            step_id: alloc.buffer(Elems(1))?,
        };
        let output_query = globals.pop().expect("two globals");
        let output_norm = globals.pop().expect("two globals");
        Ok(Arena {
            layers,
            visits,
            scratch,
            embed,
            embed_norm,
            head,
            output_norm,
            output_query,
            loop_queries,
            faults: MetalFaultWord::registered(alloc)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The folder's alternating gate·up rows land gate first: launch row `j`
    /// reads source row `2j` below `ff` and `2(j − ff) + 1` from it.
    ///
    /// Mutant: the up half reading `2(j − ff)` reads gate rows twice.
    #[test]
    fn the_fused_rows_land_gate_first() {
        let rows: Vec<usize> = (0..8).map(|j| gate_up_row(j, 8)).collect();
        assert_eq!(rows, [0, 2, 4, 6, 1, 3, 5, 7]);
    }
}
