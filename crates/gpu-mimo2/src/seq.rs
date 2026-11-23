//! A MiMo sequence on the host ([`MimoSeq`]): every layer's K and V rows up
//! to the saved position, and the written end a later cut measures from. A
//! full layer keeps the rows of those positions. A window layer keeps the
//! ring its store holds: the cut's slack reaches past the attention window,
//! so the window's keys alone would not restore a prefix the cut still
//! allows.
//!
//! A save copies on the model's stream and returns once the copy has landed.
//! A resume is refused by name, before any row is copied: a poisoned model;
//! another file, card or context; another store shape; a prefix of no
//! positions (reset the model instead); a prefix longer than the state; a
//! prefix whose depth a ring no longer holds.

use bloomery_gpu::checkpoint::saved::{Identity, SeqView};
use bloomery_gpu::flash_gqa::{HEAD, HEAD_K192};
use bloomery_gpu::{Gpu, GpuError, GpuModel};
use cuda_core::memory::{memcpy_dtoh_async, memcpy_htod_async};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::mimo2::place::{self, PlanInputs};
use models::Mixer;

use super::{Body, Store};

/// What a sequence's errors name.
const WHAT: &str = "mimo2 sequence";

/// A shape the sequence refuses, by name.
fn shape(detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// One layer's K and V on the host, packed a head at stride [`SavedLayer::take`].
struct SavedLayer {
    kv_heads: usize,
    /// Rows the device store holds a head.
    rows: usize,
    window: usize,
    /// Rows saved a head: the ring's height, or the saved positions on a plane.
    /// Zero when the state holds no position.
    take: usize,
    k: Vec<u16>,
    v: Vec<u16>,
}

/// The rows of one copy: `heads` runs, the device store holding `store_rows`
/// a head and the host buffer `host_rows`, of which `copy_rows` move.
#[derive(Clone, Copy)]
struct Heads {
    heads: usize,
    store_rows: usize,
    host_rows: usize,
    copy_rows: usize,
    width: usize,
}

/// A MiMo sequence state ([`seq_save`]).
pub struct MimoSeq {
    who: Identity,
    /// The model's position when the state was saved: the next row a step
    /// would write, which is the positions the state holds.
    positions: u32,
    /// The written end, [`Body`]'s `at` at the save. A cut measures its depth
    /// from here, and a resume stands the body back at it.
    written_end: u32,
    layers: Vec<SavedLayer>,
}

impl MimoSeq {
    /// The positions it holds.
    #[must_use]
    pub fn positions(&self) -> u32 {
        self.positions
    }

    /// The host bytes of the saved rows.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .fold(0usize, |n, layer| {
                n.saturating_add(layer.k.len())
                    .saturating_add(layer.v.len())
            })
            .saturating_mul(size_of::<u16>())
    }

    /// The prefix of at most `n` positions a resume can stand at. A state
    /// with no positions, or a prefix of none, is 0. There is no nearer
    /// point: a ring that does not keep the depth gives 0, as the live keep
    /// does.
    #[must_use]
    pub fn keep_point(&self, n: u32) -> u32 {
        if self.positions == 0 || n == 0 {
            return 0;
        }
        let n = n.min(self.positions);
        let end = self.written_end;
        if n > end {
            return 0;
        }
        let depth = u64::from(end - n);
        let ctx = self.who.ctx as u64;
        let keeps = self.layers.iter().all(|layer| {
            place::cut_depth_keeps(depth, layer.rows as u64, ctx, layer.window as u64)
        });
        if keeps { n } else { 0 }
    }
}

impl SeqView for MimoSeq {
    fn positions(&self) -> u32 {
        MimoSeq::positions(self)
    }

    fn bytes(&self) -> usize {
        MimoSeq::bytes(self)
    }

    fn keep_point(&self, n: u32) -> u32 {
        MimoSeq::keep_point(self, n)
    }
}

/// The sequence state on the host ([`MimoSeq`]): each layer's rows below
/// `m`'s position, a window layer the whole ring. Refused by name on a
/// poisoned model, while a step has been accepted and has not stood, and
/// when the position or the written end is past the stores.
pub fn seq_save(m: &mut GpuModel<Body>) -> Result<MimoSeq, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    let pos = m.pos();
    let (gpu, _, body) = m.body_parts(WHAT)?;
    body.save_seq(gpu, pos)
}

/// Replace `m`'s sequence with `s`. Refused by name, and the model is left
/// as it stands, for an empty state: reset the model instead.
pub fn seq_resume(m: &mut GpuModel<Body>, s: &MimoSeq) -> Result<(), GpuError> {
    seq_resume_prefix(m, s, s.positions()).map(drop)
}

/// [`seq_resume`] of the first `n` positions of `s` the state can be cut to
/// ([`MimoSeq::keep_point`]). The model stands at that many, which is
/// returned, and the body's written end is the one `s` saved. Refused by
/// name as the module doc says, before the model is reset and before any
/// row is copied.
pub fn seq_resume_prefix(m: &mut GpuModel<Body>, s: &MimoSeq, n: u32) -> Result<u32, GpuError> {
    if let Some(fault) = m.poisoned() {
        return Err(GpuError::Poisoned { what: WHAT, fault });
    }
    if n == 0 || s.positions == 0 {
        return Err(shape(format!(
            "a prefix of 0 positions of a state of {} (asked {n}): reset the model instead",
            s.positions
        )));
    }
    m.body(WHAT)?.fits_seq(s)?;
    if n > s.positions {
        return Err(shape(format!(
            "a prefix of {n} positions of a state of {}",
            s.positions
        )));
    }
    let at = s.keep_point(n);
    if at == 0 {
        return Err(shape(format!(
            "a prefix of {n} positions of a state of {} whose written end is {} drops keys a ring \
             no longer holds",
            s.positions, s.written_end
        )));
    }
    m.reset()?;
    m.run_rows(at as usize, WHAT, |gpu, _, body, _, _| {
        body.put_seq(gpu, s, at)?;
        Ok(false)
    })?;
    let (_, _, body) = m.body_parts(WHAT)?;
    body.stand_for_cut(s.written_end);
    Ok(at)
}

/// The host bytes one saved state of `positions` holds at `ctx`, on the file
/// `inputs` describes. A plane layer counts the positions; a window layer
/// counts every row of its ring, which a state of one position already
/// copies. A context or a position count of 0 is 0.
#[must_use]
pub fn seq_bytes(inputs: &PlanInputs, ctx: usize, positions: usize) -> u64 {
    if ctx == 0 || positions == 0 {
        return 0;
    }
    let ctx_u = ctx as u64;
    let mut bytes = 0u64;
    for (layer, spec) in inputs.spec.layers.iter().enumerate() {
        // A non-GQA layer contributes no K or V this body sizes.
        let Mixer::Gqa(g) = &spec.mixer else {
            continue;
        };
        let device_rows = inputs.kv.rows_at(layer, ctx_u);
        let host_rows = saved_rows(device_rows, ctx_u, positions as u64);
        let width = u64::from(g.head_dim).saturating_add(u64::from(g.value_dim));
        bytes = bytes.saturating_add(
            u64::from(g.kv_heads)
                .saturating_mul(host_rows)
                .saturating_mul(width)
                .saturating_mul(size_of::<u16>() as u64),
        );
    }
    bytes
}

/// Rows one layer's save holds. `device_rows` is the store (`KvLayout::rows_at`):
/// the context on a full layer, the ring otherwise. A plane saves the
/// positions; a ring saves every row it holds.
fn saved_rows(device_rows: u64, ctx: u64, positions: u64) -> u64 {
    if device_rows == 0 || positions == 0 {
        0
    } else if device_rows >= ctx {
        positions.min(ctx)
    } else {
        device_rows
    }
}

impl Body {
    /// `s` belongs to this load: the same file, card and context, the same
    /// layer shapes, and a written end the stores can hold.
    fn fits_seq(&self, s: &MimoSeq) -> Result<(), GpuError> {
        if s.who.arch != self.who.arch || s.who.file != self.who.file || s.who.card != self.who.card
        {
            return Err(shape(format!(
                "a state of {} put back on {}",
                s.who, self.who
            )));
        }
        if s.who.ctx > self.ctx {
            return Err(shape(format!(
                "a state of {} positions put back on a body of {} positions: a longer context",
                s.who.ctx, self.ctx
            )));
        }
        if s.who.ctx != self.ctx {
            return Err(shape(format!(
                "a state of {} positions put back on a body of {} positions: another shape",
                s.who.ctx, self.ctx
            )));
        }
        let end = usize::try_from(s.written_end).unwrap_or(usize::MAX);
        if s.written_end < s.positions || end > self.ctx {
            return Err(shape(format!(
                "a state of {} positions whose written end is {} on stores of {}",
                s.positions, s.written_end, self.ctx
            )));
        }
        if s.layers.len() != self.stores.len() || s.layers.len() != self.cfg.len() {
            return Err(shape(format!(
                "a state of {} layers put back on a body of {}",
                s.layers.len(),
                self.stores.len()
            )));
        }
        for (i, (layer, (store, cfg))) in s
            .layers
            .iter()
            .zip(self.stores.iter().zip(&self.cfg))
            .enumerate()
        {
            if layer.kv_heads != cfg.attn.kv_heads
                || layer.rows != store.rows
                || layer.window != cfg.attn.window
            {
                return Err(shape(format!(
                    "layer {i}: a state of {} heads, {} rows, window {} put back on {} heads, {} \
                     rows, window {}",
                    layer.kv_heads,
                    layer.rows,
                    layer.window,
                    cfg.attn.kv_heads,
                    store.rows,
                    cfg.attn.window
                )));
            }
            let ring = store.rows < self.ctx;
            let expect = if s.positions == 0 {
                0
            } else if ring {
                store.rows
            } else {
                usize::try_from(s.positions).unwrap_or(usize::MAX)
            };
            if layer.take != expect {
                return Err(shape(format!(
                    "layer {i}: a state holds {} rows a head of a store of {} rows at {} positions",
                    layer.take, store.rows, s.positions
                )));
            }
            let k = plane_len(layer.kv_heads, layer.take, HEAD_K192);
            let v = plane_len(layer.kv_heads, layer.take, HEAD);
            if k != Some(layer.k.len()) || v != Some(layer.v.len()) {
                return Err(shape(format!(
                    "layer {i}: the saved K and V rows are another shape"
                )));
            }
        }
        Ok(())
    }

    /// The live sequence's rows, copied into a state. The host rows stay
    /// alive until the stream is idle.
    fn save_seq(&self, gpu: &Gpu, pos: u32) -> Result<MimoSeq, GpuError> {
        if let Some(waiting) = self.pending {
            return Err(shape(format!(
                "a step accepted at {waiting} has not stood: a sequence is saved between calls"
            )));
        }
        let positions = usize::try_from(pos).unwrap_or(usize::MAX);
        if positions > self.ctx {
            return Err(shape(format!(
                "a position {pos} past the {} the stores hold",
                self.ctx
            )));
        }
        let end = self.at.max(pos);
        if usize::try_from(end).unwrap_or(usize::MAX) > self.ctx {
            return Err(shape(format!(
                "a written end {end} past the {} the stores hold",
                self.ctx
            )));
        }
        if self.stores.len() != self.cfg.len() {
            return Err(shape(format!(
                "{} stores and {} layers",
                self.stores.len(),
                self.cfg.len()
            )));
        }
        let mut layers = Vec::with_capacity(self.stores.len());
        for (i, (store, cfg)) in self.stores.iter().zip(&self.cfg).enumerate() {
            let ring = store.rows < self.ctx;
            if !ring && positions > store.rows {
                return Err(shape(format!(
                    "layer {i}: a position {pos} past a plane of {} rows",
                    store.rows
                )));
            }
            // A window layer's ring is saved whole: a cut keeps every key the
            // ring still holds, which reaches past the attention window.
            let take = if pos == 0 {
                0
            } else if ring {
                store.rows
            } else {
                positions
            };
            held_store(i, store, cfg.attn.kv_heads)?;
            layers.push(SavedLayer {
                kv_heads: cfg.attn.kv_heads,
                rows: store.rows,
                window: cfg.attn.window,
                take,
                k: alloc_plane(cfg.attn.kv_heads, take, HEAD_K192)?,
                v: alloc_plane(cfg.attn.kv_heads, take, HEAD)?,
            });
        }
        if pos > 0 {
            let stream = gpu.stream();
            let copied = read_layers(stream, &self.stores, &mut layers);
            let synced = sync_stream(stream);
            copied.and(synced)?;
        }
        Ok(MimoSeq {
            who: self.who.clone(),
            positions: pos,
            written_end: end,
            layers,
        })
    }

    /// `s`'s prefix of `at` positions written back. Reset leaves the previous
    /// sequence's rows, so the stores are zeroed first. The host rows of `s`
    /// stay borrowed until the stream is idle.
    fn put_seq(&mut self, gpu: &Gpu, s: &MimoSeq, at: u32) -> Result<(), GpuError> {
        self.clear_stores(gpu)?;
        let stream = gpu.stream();
        let wrote = write_layers(stream, &mut self.stores, &s.layers, at, self.ctx);
        let synced = sync_stream(stream);
        wrote.and(synced)
    }
}

/// The device store holds the heads and rows the layer says it does.
fn held_store(layer: usize, store: &Store, kv_heads: usize) -> Result<(), GpuError> {
    let k = plane_len(kv_heads, store.rows, HEAD_K192);
    let v = plane_len(kv_heads, store.rows, HEAD);
    if k.is_some_and(|n| store.k.len() >= n) && v.is_some_and(|n| store.v.len() >= n) {
        return Ok(());
    }
    Err(shape(format!(
        "layer {layer}: a store of {} and {} values, short of {} heads of {} rows",
        store.k.len(),
        store.v.len(),
        kv_heads,
        store.rows
    )))
}

fn plane_len(heads: usize, rows: usize, width: usize) -> Option<usize> {
    heads.checked_mul(rows).and_then(|n| n.checked_mul(width))
}

fn alloc_plane(heads: usize, rows: usize, width: usize) -> Result<Vec<u16>, GpuError> {
    let n = plane_len(heads, rows, width)
        .ok_or_else(|| shape("a layer's K or V rows overflow usize"))?;
    Ok(vec![0u16; n])
}

fn sync_stream(stream: &CudaStream) -> Result<(), GpuError> {
    stream.synchronize().map_err(|source| GpuError::Driver {
        op: Some("cuStreamSynchronize"),
        source,
    })
}

fn read_layers(
    stream: &CudaStream,
    stores: &[Store],
    layers: &mut [SavedLayer],
) -> Result<(), GpuError> {
    for (i, (store, layer)) in stores.iter().zip(layers.iter_mut()).enumerate() {
        let heads = Heads {
            heads: layer.kv_heads,
            store_rows: store.rows,
            host_rows: layer.take,
            copy_rows: layer.take,
            width: HEAD_K192,
        };
        move_to_host(stream, &store.k, &mut layer.k, heads).map_err(|e| layer_err(i, e))?;
        move_to_host(
            stream,
            &store.v,
            &mut layer.v,
            Heads {
                width: HEAD,
                ..heads
            },
        )
        .map_err(|e| layer_err(i, e))?;
    }
    Ok(())
}

fn write_layers(
    stream: &CudaStream,
    stores: &mut [Store],
    layers: &[SavedLayer],
    at: u32,
    ctx: usize,
) -> Result<(), GpuError> {
    let at = usize::try_from(at).unwrap_or(usize::MAX);
    for (i, (store, layer)) in stores.iter_mut().zip(layers.iter()).enumerate() {
        let copy_rows = if store.rows < ctx { layer.take } else { at };
        let heads = Heads {
            heads: layer.kv_heads,
            store_rows: store.rows,
            host_rows: layer.take,
            copy_rows,
            width: HEAD_K192,
        };
        move_to_device(stream, &store.k, &layer.k, heads).map_err(|e| layer_err(i, e))?;
        move_to_device(
            stream,
            &store.v,
            &layer.v,
            Heads {
                width: HEAD,
                ..heads
            },
        )
        .map_err(|e| layer_err(i, e))?;
    }
    Ok(())
}

/// Name the layer on a shape error. A driver error already names its call.
fn layer_err(layer: usize, err: GpuError) -> GpuError {
    match err {
        GpuError::Shape { what, detail } => GpuError::Shape {
            what,
            detail: format!("layer {layer}: {detail}"),
        },
        other => other,
    }
}

fn spans_fit(dev_len: usize, host_len: usize, g: Heads) -> Result<(), GpuError> {
    if g.copy_rows == 0 {
        return Ok(());
    }
    let store = plane_len(g.heads, g.store_rows, g.width);
    let host = plane_len(g.heads, g.host_rows, g.width);
    let (Some(store), Some(host_n)) = (store, host) else {
        return Err(shape("a layer's K or V rows overflow usize"));
    };
    if g.copy_rows > g.store_rows
        || g.copy_rows > g.host_rows
        || store > dev_len
        || host_n > host_len
    {
        return Err(shape(format!(
            "a copy of {} heads, {} of {} store rows and {} host rows, width {}, into {dev_len} \
             and {host_len} values",
            g.heads, g.copy_rows, g.store_rows, g.host_rows, g.width
        )));
    }
    Ok(())
}

fn row_bytes(rows: usize, width: usize) -> Result<usize, GpuError> {
    rows.checked_mul(width)
        .and_then(|n| n.checked_mul(size_of::<u16>()))
        .ok_or_else(|| shape("a layer's K or V rows overflow usize"))
}

fn byte_off(elems: usize) -> Result<u64, GpuError> {
    u64::try_from(elems)
        .ok()
        .and_then(|n| n.checked_mul(size_of::<u16>() as u64))
        .ok_or_else(|| shape("a layer's K or V rows overflow"))
}

/// Element `h · rows · width`, refused by name when it overflows, and the
/// byte offset of that element.
fn head_at(h: usize, rows: usize, width: usize) -> Result<(usize, u64), GpuError> {
    let elems =
        plane_len(h, rows, width).ok_or_else(|| shape("a layer's K or V rows overflow usize"))?;
    Ok((elems, byte_off(elems)?))
}

fn move_to_host(
    stream: &CudaStream,
    dev: &DeviceBuffer<u16>,
    host: &mut [u16],
    g: Heads,
) -> Result<(), GpuError> {
    spans_fit(dev.len(), host.len(), g)?;
    let n_bytes = row_bytes(g.copy_rows, g.width)?;
    if g.copy_rows == 0 || n_bytes == 0 {
        return Ok(());
    }
    for h in 0..g.heads {
        let (_, dev_at) = head_at(h, g.store_rows, g.width)?;
        let (host_at, _) = head_at(h, g.host_rows, g.width)?;
        let dst = host
            .get_mut(host_at..)
            .ok_or_else(|| shape("a copy leaves the host rows"))?
            .as_mut_ptr();
        // SAFETY: spans_fit put both spans inside their buffers, `host_at`
        // is inside `host`, and the host span is `n_bytes` long. `host` is
        // not read or freed until `stream` is synchronized. `stream` is the
        // stream the store was allocated on.
        unsafe {
            memcpy_dtoh_async(
                dst,
                dev.cu_deviceptr() + dev_at,
                n_bytes,
                stream.cu_stream(),
            )
            .map_err(|source| GpuError::Driver {
                op: Some("cuMemcpyDtoHAsync_v2"),
                source,
            })?;
        }
    }
    Ok(())
}

fn move_to_device(
    stream: &CudaStream,
    dev: &DeviceBuffer<u16>,
    host: &[u16],
    g: Heads,
) -> Result<(), GpuError> {
    spans_fit(dev.len(), host.len(), g)?;
    let n_bytes = row_bytes(g.copy_rows, g.width)?;
    if g.copy_rows == 0 || n_bytes == 0 {
        return Ok(());
    }
    for h in 0..g.heads {
        let (_, dev_at) = head_at(h, g.store_rows, g.width)?;
        let (host_at, _) = head_at(h, g.host_rows, g.width)?;
        let src = host
            .get(host_at..)
            .ok_or_else(|| shape("a copy leaves the host rows"))?
            .as_ptr();
        // SAFETY: spans_fit put both spans inside their buffers, `host_at`
        // is inside `host`, and the host span is `n_bytes` long. `host` is
        // not written or freed until `stream` is synchronized. `stream` is
        // the stream the store was allocated on.
        unsafe {
            memcpy_htod_async(
                dev.cu_deviceptr() + dev_at,
                src,
                n_bytes,
                stream.cu_stream(),
            )
            .map_err(|source| GpuError::Driver {
                op: Some("cuMemcpyHtoDAsync_v2"),
                source,
            })?;
        }
    }
    Ok(())
}
