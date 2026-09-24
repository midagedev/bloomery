//! Attention over the key set window rows ⧺ compressed rows, with K = V: one
//! latent row per key is both its key and its value. Each head's learned
//! sink joins the softmax denominator.
//!
//! Two launches, `bloomery_gpu::flash`'s split design: a tensor-core segment
//! pass whose walk is [`bloomery_gpu::mma_segment_walk`] — the one V2-Lite's
//! `flash_latent_mma` expands — at the latent width, and a merge that folds
//! a row's segments in order and then its head's sink.
//!
//! Keys come from two sources: the window ring (the layer's last `W` latent
//! rows) and the layer stream's compressed rows (none on a layer without a
//! stream). A launch's segments are the window's first, then the compressed
//! rows', and the merge folds them in that order — fixed, so reruns are
//! bit-identical. Token `t` sees `vis[2t]` window rows, `min(pos + 1, W)`,
//! the ring's first rows (its first rows while `pos < W`, all of it once
//! full, since the ring then holds exactly the window), and `vis[2t + 1]`
//! compressed rows, read one of two ways: a prefix of the stream,
//! `(pos + 1) / ratio` rows (`ds41_attn_seg`), or the rows the indexer
//! selects, read in place through its list — the first `vis[2t + 1]` entries
//! of the token's row of the list, in list order (`ds41_attn_seg_sel`). The
//! counts and the lists live in device memory, so a captured step reads them
//! per replay; the grid comes from the ring's and the stream's (or the
//! list's) heights. A batch whose window wraps the ring is outside this
//! contract.
//!
//! A block serves one token: sixteen of its heads, the `mma.sync` `M` axis,
//! over one segment — a launch of T tokens is T one-token launches' blocks
//! in one grid.
//!
//! The arithmetic: the query row rounds to f16 on the tensor cores, so a
//! logit is an f32 accumulation of exact f16 products, times `scale`; the
//! online softmax and the value accumulation are f32; the sink is one more
//! logit in the denominator and adds nothing to the value.

use bloomery_gpu::flash::{
    MMA_BLOCK, MMA_KEYS, MMA_ROWS, MMA_SEG_KEYS, MMA_TILE, mma_dyn_bytes, mma_qwords, online_fold,
};
use bloomery_gpu::{DeviceTensor, GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, SharedArray, kernel, launch_bounds, launch_contract, thread,
};
use cuda_host::cuda_module;
use std::sync::Arc;

/// A latent row's width: the query head, and a key's K and V at once. The
/// segment block has one thread per latent dim.
pub const LATENT: usize = 512;
const _: () = assert!(LATENT == MMA_BLOCK);
// The walk's staging covers a row in `LATENT / 64` words per lane, and the
// padded row stride keeps each `ldmatrix` phase on all thirty-two banks.
const _: () = assert!(LATENT.is_multiple_of(64));
const _: () = assert!((LATENT + 8) * 2 % 128 == 16);
/// [`MMA_BLOCK`] as the `u32` block width a launch takes.
const BLOCK_U32: u32 = MMA_BLOCK as u32;
const _: () = assert!(BLOCK_U32 as usize == MMA_BLOCK);

/// Keys one segment block walks — the V2-Lite tensor-core pass's segment,
/// a multiple of the walk's tile.
pub const SEG_KEYS: usize = MMA_SEG_KEYS;
/// Segments whose partials the merge loads as one batch before it folds
/// them.
const MERGE_BATCH: usize = 16;
const _: () = assert!(SEG_KEYS.is_multiple_of(MMA_KEYS));

/// The key tile's byte offset in the segment block's dynamic shared memory:
/// the query tile comes first.
const KT_OFFSET: usize = mma_qwords(LATENT) * 4;
/// Bytes of dynamic shared memory one segment block takes.
const DYN_BYTES: usize = mma_dyn_bytes(LATENT);
/// `#[launch_contract(dynamic_shared = ...)]` takes an integer literal, so
/// the kernel spells the byte count out; this is the two sides agreeing.
const _: () = assert!(DYN_BYTES == 49920);
/// [`DYN_BYTES`] as the `u32` a launch takes.
const DYN_BYTES_U32: u32 = DYN_BYTES as u32;
const _: () = assert!(DYN_BYTES_U32 as usize == DYN_BYTES);

/// Segments a source of `keys` keys is cut into: the grid's share of it.
#[must_use]
pub fn source_segments(keys: usize) -> usize {
    keys.div_ceil(SEG_KEYS)
}

/// Segments of a launch over a `window_rows`-row ring and `compressed_keys`
/// compressed keys — the stream's height for a prefix, the list's stride for
/// selected rows: the partial slots each query row has.
#[must_use]
pub fn segments(window_rows: usize, compressed_keys: usize) -> usize {
    source_segments(window_rows) + source_segments(compressed_keys)
}

/// Length the `Σ exp·V` partials buffer needs for `q_rows` query rows over
/// `segs` segments.
#[must_use]
pub fn partials_v_len(q_rows: usize, segs: usize) -> usize {
    q_rows * segs * LATENT
}

/// Length the `(max, Σ exp)` partials buffer needs — two f32 per (query row,
/// segment).
#[must_use]
pub fn partials_ms_len(q_rows: usize, segs: usize) -> usize {
    q_rows * segs * 2
}

/// A segment block's body, the one both segment entries expand: the grid
/// decomposition, the token's counts and limit, the neutral partial past a
/// count, the block's shared memory and the walk. A window key is its own
/// ring row; compressed key `key` of token `t` reads row `comp_row` of
/// `comp`. `comp_keys` is the compressed source's key capacity — the grid's
/// share and the bound on every count. `comp_row` is evaluated only for a
/// key below the token's limit, `min(count, comp_keys)`.
macro_rules! segment_pass {
    (
        q: $q:ident,
        win: $win:ident,
        comp: $comp:ident,
        vis: $vis:ident,
        scale: $scale:ident,
        tokens: $tokens:ident,
        n_heads: $n_heads:ident,
        win_rows: $win_rows:ident,
        comp_keys: $comp_keys:expr,
        segs_w: $segs_w:ident,
        segs: $segs:ident,
        part_v: $part_v:ident,
        part_ms: $part_ms:ident,
        comp_row: |$t:ident, $key:ident| $comp_row:expr $(,)?
    ) => {{
        // Per head, the tile's scaled logits and then its weights.
        static mut KLOG: SharedArray<f32, MMA_TILE> = SharedArray::UNINIT;
        static mut KW: SharedArray<f32, MMA_TILE> = SharedArray::UNINIT;
        // Per head, the tile's max-bump rescale and the token's key limit.
        static mut VMS: SharedArray<f32, MMA_ROWS> = SharedArray::UNINIT;
        static mut LIM: SharedArray<u32, MMA_ROWS> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let heads = $n_heads as usize;
        let rows = $tokens as usize * heads;
        let n_seg = $segs as usize;
        // The host requires `heads % MMA_ROWS == 0`: a group is sixteen
        // heads of one token and every row of it is live.
        let groups = rows / MMA_ROWS;
        if b >= groups * n_seg {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        let seg = b / groups;
        let grp = b - seg * groups;
        let base_row = grp * MMA_ROWS;
        let $t = base_row / heads;
        let window = seg < $segs_w as usize;
        let (src, src_keys, first_seg) = if window {
            ($win, $win_rows as usize, 0)
        } else {
            ($comp, $comp_keys, $segs_w as usize)
        };
        // SAFETY: t < tokens, so both of its counts are inside vis (launch
        // contract).
        let count = unsafe { *$vis.get_unchecked(2 * $t + usize::from(!window)) } as usize;
        let limit = count.min(src_keys);
        let lo = (seg - first_seg) * SEG_KEYS;
        if lo >= limit {
            if tid < MMA_ROWS {
                let idx = (base_row + tid) * n_seg + seg;
                // SAFETY: base_row + tid < rows and seg < segs, so both
                // slots are inside part_ms (launch contract).
                unsafe {
                    *$part_ms.get_unchecked_mut(2 * idx) = f32::NEG_INFINITY;
                    *$part_ms.get_unchecked_mut(2 * idx + 1) = 0.0;
                }
            }
            return; // block-uniform, and no key row of this segment is read
        }
        let hi_max = (lo + SEG_KEYS).min(limit);

        // The group's query rows and the tile's key rows as f16, both in the
        // block's dynamic shared memory: the query tile first, the key tile
        // at its end. The launch contract declares exactly [`DYN_BYTES`].
        // SAFETY: each pointer below is this block's own shared allocation;
        // every access is bounded by the declared word count and ordered by
        // `sync_threads`.
        let (qs, kt, klog, kw, vms_sh, lim_sh) = unsafe {
            (
                DynamicSharedArray::<u32, 16>::get(),
                DynamicSharedArray::<u32, 16>::offset(KT_OFFSET),
                SharedArray::as_raw_mut_ptr(&raw mut KLOG),
                SharedArray::as_raw_mut_ptr(&raw mut KW),
                SharedArray::as_raw_mut_ptr(&raw mut VMS),
                SharedArray::as_raw_mut_ptr(&raw mut LIM),
            )
        };
        // The source rows pair-wise: 512 f16 per row, device-allocated, so
        // every u32 read of the walk is aligned.
        let kvw = src.as_ptr().cast::<u32>();
        // Every head of the group is one token's, so all share its limit.
        if tid < MMA_ROWS {
            // SAFETY: tid < MMA_ROWS bounds the store.
            unsafe {
                *lim_sh.add(tid) = limit as u32;
            }
        }
        // K = V: the value is the whole row.
        let rope = 0usize;
        let lat = LATENT;
        let width = LATENT;
        bloomery_gpu::mma_segment_walk! {
            width: LATENT,
            q: $q,
            kvw: kvw,
            key: $key => if window { $key } else { $comp_row },
            scratch: (qs, kt, klog, kw, vms_sh, lim_sh),
            rows: rows,
            base_row: base_row,
            n_seg: n_seg,
            seg: seg,
            lo: lo,
            hi: hi_max,
            width_rt: width,
            rope: rope,
            lat: lat,
            scale: $scale,
            tid: tid,
            part_v: $part_v,
            part_ms: $part_ms,
        }
    }};
}

#[cuda_module]
mod attn_kernels {
    use super::*;

    /// The segment pass over a compressed prefix: block `(segment, group)`
    /// walks one segment of one key source for query rows
    /// `group * MMA_ROWS ..`, sixteen heads of one token, and writes their
    /// partials — `part_ms` holds `(running max, Σ exp)` per (row, segment),
    /// `part_v` the row's un-normalised `Σ exp·V` relative to that max.
    /// Segments `0..segs_w` are the window ring's, `segs_w..segs` the
    /// stream's. A segment past its token's count writes the neutral
    /// partial `(−inf, 0)` and reads no key row, which is what keeps a row
    /// the token does not see out of every result.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        dynamic_shared = 49920,
        requires = (
            vis.len() >= 2 * tokens,
            q.len() >= tokens * n_heads * 512,
            win.len() >= win_rows * 512,
            comp.len() >= comp_rows * 512,
            part_v.len() >= tokens * n_heads * segs * 512,
            part_ms.len() >= tokens * n_heads * segs * 2
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "a kernel entry takes its launch arguments as the device ABI does"
    )]
    pub fn ds41_attn_seg(
        q: &[f32],
        win: &[u16],
        comp: &[u16],
        vis: &[u32],
        scale: f32,
        tokens: u32,
        n_heads: u32,
        win_rows: u32,
        comp_rows: u32,
        segs_w: u32,
        segs: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        segment_pass! {
            q: q,
            win: win,
            comp: comp,
            vis: vis,
            scale: scale,
            tokens: tokens,
            n_heads: n_heads,
            win_rows: win_rows,
            comp_keys: comp_rows as usize,
            segs_w: segs_w,
            segs: segs,
            part_v: part_v,
            part_ms: part_ms,
            comp_row: |t, key| key,
        }
    }

    /// The segment pass over the compressed rows the indexer selects: as
    /// `ds41_attn_seg`, with compressed key `key` of token `t` reading
    /// stream row `sel[t * sel_stride + key]` in place — the list's order is
    /// the key order. `sel_stride` is each token's list capacity, the
    /// compressed source's share of the grid. A list entry past the stream
    /// reads the stream's last row: a wrong row, never a read outside it.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        dynamic_shared = 49920,
        requires = (
            vis.len() >= 2 * tokens,
            q.len() >= tokens * n_heads * 512,
            win.len() >= win_rows * 512,
            comp.len() >= comp_rows * 512,
            comp_rows >= 1,
            sel.len() >= tokens * sel_stride,
            part_v.len() >= tokens * n_heads * segs * 512,
            part_ms.len() >= tokens * n_heads * segs * 2
        )
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "a kernel entry takes its launch arguments as the device ABI does"
    )]
    pub fn ds41_attn_seg_sel(
        q: &[f32],
        win: &[u16],
        comp: &[u16],
        sel: &[u32],
        vis: &[u32],
        scale: f32,
        tokens: u32,
        n_heads: u32,
        win_rows: u32,
        comp_rows: u32,
        sel_stride: u32,
        segs_w: u32,
        segs: u32,
        mut part_v: DisjointSlice<f32>,
        mut part_ms: DisjointSlice<f32>,
    ) {
        segment_pass! {
            q: q,
            win: win,
            comp: comp,
            vis: vis,
            scale: scale,
            tokens: tokens,
            n_heads: n_heads,
            win_rows: win_rows,
            comp_keys: sel_stride as usize,
            segs_w: segs_w,
            segs: segs,
            part_v: part_v,
            part_ms: part_ms,
            comp_row: |t, key| {
                // SAFETY: t < tokens and key < min(count, sel_stride) put
                // the load inside sel (launch contract).
                let r = unsafe { *sel.get_unchecked(t * sel_stride as usize + key) } as usize;
                r.min(comp_rows as usize - 1)
            },
        }
    }

    /// The merge: one block per query row, one thread per latent dim,
    /// folding the row's segments by [`online_fold`] in ascending order —
    /// the window's, then the compressed rows' — and then the head's sink
    /// as one more partial `(sink, 1, 0)`. A neutral partial (`Σ exp == 0`)
    /// is skipped, so a segment the token does not reach is never folded.
    /// The partials are loaded [`MERGE_BATCH`] segments at a time ahead of
    /// their folds; the folds and their order are the one-at-a-time walk's.
    #[kernel]
    #[launch_bounds(512)]
    #[launch_contract(
        domain = 1,
        block = (512, 1, 1),
        requires = (
            part_v.len() >= q_rows * segs * 512,
            part_ms.len() >= q_rows * segs * 2,
            sinks.len() >= n_heads,
            y.len() >= q_rows * 512
        )
    )]
    pub fn ds41_attn_merge(
        q_rows: u32,
        n_heads: u32,
        segs: u32,
        part_v: &[f32],
        part_ms: &[f32],
        sinks: &[f32],
        mut y: DisjointSlice<f32>,
    ) {
        let row = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        if row >= q_rows as usize {
            return; // block-uniform
        }
        let n_seg = segs as usize;
        let mut mx = f32::NEG_INFINITY;
        let mut s = 0.0f32;
        let mut acc = 0.0f32;
        let mut seg = 0usize;
        while seg < n_seg {
            // The batch's loads are issued ahead of its folds, so the walk
            // does not wait on memory twice per segment.
            let mut ms = [0.0f32; 2 * MERGE_BATCH];
            let mut vs = [0.0f32; MERGE_BATCH];
            let mut i = 0usize;
            #[unroll]
            while i < MERGE_BATCH {
                if seg + i < n_seg {
                    let idx = row * n_seg + seg + i;
                    // SAFETY: idx < q_rows * segs, so both slots are inside
                    // part_ms, and idx * LATENT + tid < q_rows * segs *
                    // LATENT <= part_v.len() (launch contract). A neutral
                    // segment's value slot is loaded and not folded.
                    unsafe {
                        ms[2 * i] = *part_ms.get_unchecked(2 * idx);
                        ms[2 * i + 1] = *part_ms.get_unchecked(2 * idx + 1);
                        vs[i] = *part_v.get_unchecked(idx * LATENT + tid);
                    }
                }
                i += 1;
            }
            let mut i = 0usize;
            #[unroll]
            while i < MERGE_BATCH {
                if seg + i < n_seg && ms[2 * i + 1] != 0.0 {
                    (mx, s, acc) = online_fold(mx, s, acc, ms[2 * i], ms[2 * i + 1], vs[i]);
                }
                i += 1;
            }
            seg += MERGE_BATCH;
        }
        // SAFETY: row % n_heads < n_heads <= sinks.len() (launch contract).
        let sink = unsafe { *sinks.get_unchecked(row % n_heads as usize) };
        (_, s, acc) = online_fold(mx, s, acc, sink, 1.0, 0.0);
        let s_inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        // SAFETY: row < q_rows and tid < LATENT, the block's width, so the
        // store is inside y (launch contract).
        unsafe {
            *y.get_unchecked_mut(row * LATENT + tid) = s_inv * acc;
        }
    }
}

/// The loaded V4.1 attention module: `ds41_attn_seg`, `ds41_attn_seg_sel`
/// and `ds41_attn_merge`. Owns no context and no stream — every enqueue
/// takes the caller's stream, so the launches order with the rest of the
/// step and are capturable.
pub struct AttnKernels {
    module: attn_kernels::LoadedModule,
}

/// The compressed rows the indexer selects: row `t` of `rows` holds
/// `stride` stream row indices, of which token `t` attends the first
/// `vis[2t + 1]`, in list order.
pub struct SelectedRows<'a> {
    pub rows: &'a DeviceBuffer<u32>,
    /// Entries per token: each token's list capacity, and the compressed
    /// source's share of the grid.
    pub stride: usize,
}

/// One attention launch's inputs and outputs. Rows are `tokens * heads`
/// query rows of [`LATENT`] f32, row `t * heads + h`, and `y` comes back in
/// the same order.
pub struct AttnArgs<'a> {
    /// The query rows.
    pub q: &'a DeviceBuffer<f32>,
    /// The window ring, `[rows x LATENT]` f16 bits.
    pub window: &'a DeviceTensor<u16>,
    /// The layer stream's compressed rows, `[rows x LATENT]` f16 bits;
    /// `None` on a layer without a stream.
    pub compressed: Option<&'a DeviceTensor<u16>>,
    /// The indexer's lists, when the compressed rows are read through them;
    /// `None` reads a prefix of the stream. Needs `compressed`.
    pub selected: Option<SelectedRows<'a>>,
    /// Per token `t`: `vis[2t]` window rows (a prefix of the ring) and
    /// `vis[2t + 1]` compressed rows (a prefix of the stream, or of the
    /// token's list).
    pub vis: &'a DeviceBuffer<u32>,
    /// Each head's sink logit.
    pub sinks: &'a DeviceBuffer<f32>,
    /// The softmax scale applied to every logit, the sink's excepted.
    pub scale: f32,
    pub tokens: usize,
    pub heads: usize,
    /// [`partials_v_len`] f32 for this launch's rows and [`segments`].
    pub part_v: &'a mut DeviceBuffer<f32>,
    /// [`partials_ms_len`] f32 for the same.
    pub part_ms: &'a mut DeviceBuffer<f32>,
    pub y: &'a mut DeviceBuffer<f32>,
}

impl AttnKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AttnKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { attn_kernels::load(ctx)? };
        Ok(AttnKernels { module })
    }

    /// Enqueue the attention: the segment pass over [`segments`]`(window
    /// rows, compressed keys)` segments per query row — the prefix entry, or
    /// the selected-row entry when `selected` is given — then the merge into
    /// `y`. Two launches whose grids come from the buffers' heights, not
    /// from `vis` or the lists, so a captured graph replays them at any
    /// visible count and any selection. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(&self, stream: &CudaStream, a: AttnArgs<'_>) -> Result<(), GpuError> {
        let what = "ds41 AttnKernels::enqueue";
        let AttnArgs {
            q,
            window,
            compressed,
            selected,
            vis,
            sinks,
            scale,
            tokens,
            heads,
            part_v,
            part_ms,
            y,
        } = a;
        let shape = |detail: String| GpuError::Shape { what, detail };
        if tokens == 0 || heads == 0 || !heads.is_multiple_of(MMA_ROWS) {
            return Err(shape(format!(
                "tokens {tokens} and heads {heads}: both >= 1 and heads a multiple of {MMA_ROWS} \
                 (a segment block is sixteen heads of one token)"
            )));
        }
        let comp_rows = compressed.map_or(0, DeviceTensor::rows);
        for (name, cols) in [
            ("window", Some(window.cols())),
            ("compressed", compressed.map(DeviceTensor::cols)),
        ] {
            if let Some(c) = cols
                && c != LATENT
            {
                return Err(shape(format!("{name} rows are {c} wide, want {LATENT}")));
            }
        }
        let comp_keys = match &selected {
            None => comp_rows,
            Some(s) => {
                if comp_rows == 0 {
                    return Err(shape(
                        "selected rows need a compressed stream of at least one row".to_string(),
                    ));
                }
                if s.rows.len() < tokens * s.stride {
                    return Err(shape(format!(
                        "selected rows hold {}, want >= tokens {tokens} x stride {}",
                        s.rows.len(),
                        s.stride
                    )));
                }
                s.stride
            }
        };
        let q_rows = tokens * heads;
        let segs_w = source_segments(window.rows());
        let segs = segments(window.rows(), comp_keys);
        for (name, have, want) in [
            ("q", q.len(), q_rows * LATENT),
            ("vis", vis.len(), 2 * tokens),
            ("sinks", sinks.len(), heads),
            ("part_v", part_v.len(), partials_v_len(q_rows, segs)),
            ("part_ms", part_ms.len(), partials_ms_len(q_rows, segs)),
            ("y", y.len(), q_rows * LATENT),
        ] {
            if have < want {
                return Err(shape(format!("{name} holds {have}, want >= {want}")));
            }
        }
        // A layer without a stream has no compressed segments: the kernel
        // never reads that source, so the window stands in for its buffer.
        let comp = compressed.unwrap_or(window);
        let seg_grid = launch_u32(what, "segment grid", q_rows / MMA_ROWS * segs)?;
        let merge_grid = launch_u32(what, "merge grid", q_rows)?;
        let tokens = launch_u32(what, "tokens", tokens)?;
        let heads = launch_u32(what, "heads", heads)?;
        let q_rows = launch_u32(what, "q_rows", q_rows)?;
        let win_rows = launch_u32(what, "window rows", window.rows())?;
        let comp_rows = launch_u32(what, "compressed rows", comp_rows)?;
        let segs_w = launch_u32(what, "window segments", segs_w)?;
        let segs = launch_u32(what, "segments", segs)?;
        let seg_config = LaunchConfig1D::new(seg_grid, BLOCK_U32, DYN_BYTES_U32);
        match selected {
            None => {
                let prep = self.module.prepare_ds41_attn_seg(seg_config)?;
                self.module.ds41_attn_seg(
                    stream,
                    &prep,
                    q,
                    window.buf(),
                    comp.buf(),
                    vis,
                    scale,
                    tokens,
                    heads,
                    win_rows,
                    comp_rows,
                    segs_w,
                    segs,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
            Some(s) => {
                let sel_stride = launch_u32(what, "selected stride", s.stride)?;
                let prep = self.module.prepare_ds41_attn_seg_sel(seg_config)?;
                self.module.ds41_attn_seg_sel(
                    stream,
                    &prep,
                    q,
                    window.buf(),
                    comp.buf(),
                    s.rows,
                    vis,
                    scale,
                    tokens,
                    heads,
                    win_rows,
                    comp_rows,
                    sel_stride,
                    segs_w,
                    segs,
                    &mut *part_v,
                    &mut *part_ms,
                )?;
            }
        }
        let prep = self
            .module
            .prepare_ds41_attn_merge(LaunchConfig1D::new(merge_grid, BLOCK_U32, 0))?;
        self.module.ds41_attn_merge(
            stream, &prep, q_rows, heads, segs, part_v, part_ms, sinks, y,
        )?;
        Ok(())
    }
}
