//! The qwen3moe chain's resident state: every intermediate of one layer for
//! the decode step's one row or the prefill's several, shared by all layers
//! (they run in turn), the step's input record the captured graph reads, the
//! rope table, and the per-layer K/V planes. All of it is allocated once at
//! load; nothing here is allocated per step or per prompt.
//!
//! Inputs. Every path hands its first launch one shape, a unit's input
//! record: word [`IN_POS0`] the position of the unit's first row, then its
//! ids from word [`IN_IDS`] — `m + 1` words for `m` rows, written by one
//! filler ([`put_input`]) into pinned host words and moved by one
//! asynchronous copy ([`Inbox`]). The unit's embedding launch derives each
//! row's position and live key count from it on the card, into the arena's
//! `pos` and `n_keys`: the rope turns a row by the table's row at its
//! position ([`RopeRows`]) and appends it there, and the flash reads its
//! live key count.

use super::body::KvQ8;
use super::image::MediaRows;
use super::plan::Flash;
use super::wide::{GEMV_COLS, Wide};
use crate::GpuError;
use crate::checkpoint::saved::{Plane, Spans};
use crate::fault::FaultSink;
use crate::flash_gqa::{
    CachePlanes, FlashGqaKernels, GROUP, GqaArgs, GqaQ8Args, GqaRowsArgs, HEAD, HEAD_256, KvSeq,
    RowTable, partials_ms_len, partials_v_len, partials_v_len_256,
};
use crate::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs, GqaPrefillQ8Args, QueryNorm};
use crate::gemm::GEMM_MAX_SLOTS;
use crate::linear::{self, LinearShape};
use crate::route_softmax::{MAX_TOKENS, RouterDims, RouterOut};
use crate::site::WideForms;
use chain::arch::qwen3moe::router::ubatch_at;

pub(super) use crate::recur::{RecStore, fresh_stamps, stamps_at};
use crate::room::{Roomed, Run, Views};
use crate::rope_neox::{
    NeoxArgs, NeoxQ8Args, NeoxRowsArgs, PartialNeoxArgs, PartialNeoxQ8Args, RopeNeoxKernels,
    q8_plane_lens,
};
use crate::rope_norm::{NormArgs, NormQ8Args, RopeNormKernels};
use crate::tensor::{Q8Act, WindowMut};
use crate::weights::q8_0_planes;
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use gguf::quant::f32_to_f16_bits;
use model::quant::quantize_q8_0;
use std::ops::Range;
use std::sync::Arc;

pub(super) use crate::rope_table::RopeRows;

/// Word offsets of a unit's input record: the position of its first row,
/// then its ids.
pub(super) const IN_POS0: usize = 0;
pub(super) const IN_IDS: usize = 1;

/// The one filler of a unit's input record: `pos0` at [`IN_POS0`] and `ids`
/// from [`IN_IDS`] of `rec`, `ids.len() + 1` words; the words past them are
/// left as they are. Refused when `rec` is shorter.
pub(super) fn put_input(rec: &mut [u32], ids: &[u32], pos0: u32) -> Result<(), GpuError> {
    let words = IN_IDS + ids.len();
    let len = rec.len();
    let rec = rec.get_mut(..words).ok_or_else(|| {
        GpuError::shape(
            "qwen3moe::put_input",
            format!(
                "{} ids take {words} words of a {len}-word record",
                ids.len()
            ),
        )
    })?;
    rec[IN_POS0] = pos0;
    rec[IN_IDS..].copy_from_slice(ids);
    Ok(())
}

/// Input words on both sides: pinned host words the filler writes, their
/// device twin the launches read, and the event that keeps the host side
/// from being written while a copy still reads it. One asynchronous copy
/// moves a prefix across, on the engine stream ahead of the launches or the
/// replay that read it.
pub(super) struct Inbox {
    host: PinnedHostBuffer<u32>,
    dev: DeviceBuffer<u32>,
    /// Recorded behind each copy.
    copied: CudaEvent,
}

impl Inbox {
    /// `words` zeroed words on both sides. Load-time only.
    pub(super) fn new(stream: &CudaStream, words: usize) -> Result<Inbox, GpuError> {
        let ctx = stream.context();
        Ok(Inbox {
            host: PinnedHostBuffer::zeroed(ctx, words)?,
            dev: DeviceBuffer::zeroed(stream, words)?,
            copied: ctx.new_event(None)?,
        })
    }

    /// The host words, writable once the last copy has read them: blocks
    /// until then, and returns at once before the first copy.
    pub(super) fn host_mut(&mut self) -> Result<&mut [u32], GpuError> {
        self.copied.synchronize()?;
        Ok(self.host.as_mut_slice())
    }

    /// Enqueue the copy of host words `0 .. words` to the same device words
    /// on `stream`. Asynchronous.
    pub(super) fn upload(&mut self, stream: &CudaStream, words: usize) -> Result<(), GpuError> {
        let len = self.host.len();
        let src = self.host.as_slice().get(..words).ok_or_else(|| {
            GpuError::shape(
                "qwen3moe::Inbox::upload",
                format!("{words} words of a {len}-word inbox"),
            )
        })?;
        // SAFETY: `words <= len`, the device side's length too, so the window
        // lies inside it; it lives for this enqueue, and the device words stay
        // in place for the inbox's life.
        let mut dst = unsafe { param_view::<u32>(&self.dev, 0, words) };
        // SAFETY: `src` is pinned memory this inbox owns; it is not written
        // before `copied`, recorded right below, has passed
        // ([`Inbox::host_mut`]), nor freed before it (the inbox's drop), so the
        // copy reads it whole.
        unsafe { dst.copy_from_host_async_unchecked(stream, src)? };
        self.copied.record(stream)?;
        Ok(())
    }

    /// The device words.
    pub(super) fn dev(&self) -> &DeviceBuffer<u32> {
        &self.dev
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.dev.num_bytes()
    }
}

impl Drop for Inbox {
    fn drop(&mut self) {
        // The pinned words outlive any copy still reading them. A failure on
        // the drop path is unreportable and ignored.
        let _ = self.copied.synchronize();
    }
}

/// The shapes the arena is cut for, read from the file at load.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dims {
    pub(super) hidden: usize,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    /// Values of an attention head: [`HEAD`] or [`HEAD_256`].
    pub(super) head: usize,
    /// Rows the query projection writes a token: `n_head · head`, twice that
    /// when it writes a gate beside each head's query.
    pub(super) q_rows: usize,
    /// A routed expert's values, or a dense FFN's.
    pub(super) ff: usize,
    /// The router's instance and the experts a token keeps, from the file's
    /// routed shape; `None` for a chain of dense FFNs, whose one slot a token
    /// is the arena's fixed route ([`Route::Dense`]).
    pub(super) router: Option<RouterDims>,
    /// The delta layers' head counts; `None` when the chain has none.
    pub(super) lin: Option<LinearShape>,
    pub(super) ctx: usize,
}

impl Dims {
    /// qwen3moe's: heads of [`HEAD`], a query projection of `n_head · HEAD`
    /// rows, the router `router`, no delta layer.
    pub(super) fn qwen3(
        hidden: usize,
        n_head: usize,
        n_kv: usize,
        ff: usize,
        router: RouterDims,
        ctx: usize,
    ) -> Dims {
        Dims {
            hidden,
            n_head,
            n_kv,
            head: HEAD,
            q_rows: n_head * HEAD,
            ff,
            router: Some(router),
            lin: None,
            ctx,
        }
    }

    /// Values of a token's attention rows: `n_head · head`.
    pub(super) fn attn_len(&self) -> usize {
        self.n_head * self.head
    }

    /// Values of a token's key (or value) rows: `n_kv · head`.
    pub(super) fn kv_len(&self) -> usize {
        self.n_kv * self.head
    }

    /// Expert slots a token takes: the routed ones, plus a folded shared
    /// expert's; a dense FFN's one.
    pub(super) fn slots(&self) -> usize {
        self.router.map_or(1, |r| r.slots())
    }

    /// The router's dims, or a named refusal on a chain of dense FFNs: what a
    /// routed op takes.
    pub(super) fn routed(&self, what: &'static str) -> Result<RouterDims, GpuError> {
        self.router.ok_or(GpuError::state(
            what,
            "a router (this chain's FFNs are dense)",
        ))
    }

    /// The most tokens a ubatch may hold in one GEMM route table: at the
    /// router's slots a token ([`RouterDims::ubatch`]), or a dense chain's
    /// one.
    pub(super) fn ubatch_most(&self) -> usize {
        ubatch_at(self.slots(), GEMM_MAX_SLOTS)
    }
}

/// One layer's K and V planes, as the load's cache format holds them:
/// `[n_kv][ctx][head]` f16 each ([`KvPlanes::F16`]), or — when the load runs
/// its cache in q8_0 ([`KvQ8`]) — the two-plane layout the weights side owns
/// (`weights::q8_0_planes`): per side a codes plane of `head/4` u32 a row
/// and a scales plane of `head/32` u16 ([`rope_neox::q8_plane_lens`]), the
/// rows the quantizing appends write and the q8 read paths walk.
pub(super) enum KvPlanes {
    F16 {
        k: DeviceBuffer<u16>,
        v: DeviceBuffer<u16>,
    },
    Q8 {
        kq: DeviceBuffer<u32>,
        kd: DeviceBuffer<u16>,
        vq: DeviceBuffer<u32>,
        vd: DeviceBuffer<u16>,
    },
}

/// The fields [`NeoxArgs`](crate::rope_neox::NeoxArgs) and
/// [`NeoxQ8Args`](crate::rope_neox::NeoxQ8Args) share, the head-128 append
/// every cache-format arm of [`KvPlanes::append_128`] launches.
pub(super) struct Append128<'a> {
    pub(super) q: &'a mut DeviceBuffer<f32>,
    pub(super) k: &'a mut DeviceBuffer<f32>,
    pub(super) v: &'a DeviceBuffer<f32>,
    pub(super) gq: &'a DeviceBuffer<f32>,
    pub(super) gk: &'a DeviceBuffer<f32>,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) pos: &'a DeviceBuffer<u32>,
    pub(super) eps: f32,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
    pub(super) m: usize,
    pub(super) fault: FaultSink,
}

/// The fields [`PartialNeoxArgs`](crate::rope_neox::PartialNeoxArgs) and
/// [`PartialNeoxQ8Args`](crate::rope_neox::PartialNeoxQ8Args) share, the
/// head-256 append every cache-format arm of [`KvPlanes::append_256`]
/// launches.
pub(super) struct Append256<'a> {
    pub(super) qg: &'a DeviceBuffer<f32>,
    pub(super) q: &'a mut DeviceBuffer<f32>,
    pub(super) k: &'a mut DeviceBuffer<f32>,
    pub(super) v: &'a DeviceBuffer<f32>,
    pub(super) gq: &'a DeviceBuffer<f32>,
    pub(super) gk: &'a DeviceBuffer<f32>,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) pos: &'a DeviceBuffer<u32>,
    pub(super) eps: f32,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
    pub(super) m: usize,
    pub(super) fault: FaultSink,
}

/// The fields [`NormArgs`] and [`NormQ8Args`] share, the NORM turn and append
/// every cache-format arm of [`KvPlanes::append_norm`] launches.
pub(super) struct AppendNorm<'a> {
    pub(super) q: &'a mut DeviceBuffer<f32>,
    pub(super) k: &'a mut DeviceBuffer<f32>,
    pub(super) v: &'a DeviceBuffer<f32>,
    pub(super) table: &'a DeviceBuffer<f32>,
    pub(super) pos: &'a DeviceBuffer<u32>,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
    pub(super) m: usize,
    pub(super) fault: FaultSink,
}

/// The fields [`GqaArgs`](crate::flash_gqa::GqaArgs) and
/// [`GqaQ8Args`](crate::flash_gqa::GqaQ8Args) share, the decode flash every
/// cache-format arm of [`KvPlanes::flash_128`] and [`KvPlanes::flash_256`]
/// launches.
pub(super) struct FlashPass<'a> {
    pub(super) q: &'a DeviceBuffer<f32>,
    pub(super) n_keys: &'a DeviceBuffer<u32>,
    pub(super) scale: f32,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
    pub(super) m: usize,
    pub(super) part_v: &'a mut DeviceBuffer<f32>,
    pub(super) part_ms: &'a mut DeviceBuffer<f32>,
    pub(super) fault: FaultSink,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

/// The fields [`GqaPrefillArgs`](crate::flash_gqa_prefill::GqaPrefillArgs)
/// and [`GqaPrefillQ8Args`](crate::flash_gqa_prefill::GqaPrefillQ8Args)
/// share, the prefill flash every cache-format arm of
/// [`KvPlanes::prefill_128`] and [`KvPlanes::prefill_256`] launches.
pub(super) struct PrefillFlash<'a> {
    pub(super) q: &'a DeviceBuffer<f32>,
    pub(super) n_keys: &'a DeviceBuffer<u32>,
    pub(super) scale: f32,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ctx: usize,
    pub(super) t: usize,
    pub(super) fault: FaultSink,
    pub(super) y: &'a mut DeviceBuffer<f32>,
}

/// The named refusal of a flash a q8_0 cache has no entry for: its reads
/// serve the eight-head layout's tensor-core passes only (the `_q8`
/// paragraphs of `flash_gqa` and `flash_gqa_prefill`), and `pass` names the
/// one asked for. The one owner of that refusal's text: `set_flash_mma`
/// raises it at the call ([`KvPlanes::serves_scalar`]), and a dispatch
/// raises it for a combination no load builds.
fn q8_unserved(what: &'static str, flash: Flash, pass: &str) -> GpuError {
    GpuError::shape(
        what,
        format!(
            "a q8_0 cache's flash runs the eight-head layout's tensor-core pass only; asked for \
             the {pass} pass at the {flash:?} layout"
        ),
    )
}

impl KvPlanes {
    /// Whether these planes serve the scalar flash pass at the layout
    /// `flash`: an f16 cache does, except at the group of seven, whose flash
    /// is a tensor-core pass only; a q8_0 cache's flash is the eight-head
    /// layout's tensor-core pass only — asking for the scalar pass on it is
    /// [`q8_unserved`]'s named refusal, raised by the call that asks
    /// (`set_flash_mma`) before any step dispatches it.
    pub(super) fn serves_scalar(&self, what: &'static str, flash: Flash) -> Result<(), GpuError> {
        match (self, flash) {
            (KvPlanes::F16 { .. }, Flash::Seven) => Err(GpuError::shape(
                what,
                "the scalar pass at the group-of-seven layout: it has a tensor-core pass only",
            )),
            (KvPlanes::F16 { .. }, _) => Ok(()),
            (KvPlanes::Q8 { .. }, _) => Err(q8_unserved(what, flash, "scalar")),
        }
    }

    /// Planes of `n_kv` key heads, `ctx` rows and `head` values in the `kv`
    /// format, each in a room: every head's `[ctx][·]` rows of a plane a run
    /// (the f16 rows, or the q8_0 codes and scales [`q8_plane_lens`] counts), so
    /// a fit backs the rows of the positions it reaches in every head of
    /// every plane ([`Roomed::fit`]). Load-time only.
    pub(super) fn roomed(
        stream: &Arc<CudaStream>,
        (n_kv, ctx, head): (usize, usize, usize),
        kv: KvQ8,
    ) -> Result<Roomed<KvPlanes>, GpuError> {
        fn planes<T, const N: usize>(
            stream: &Arc<CudaStream>,
            (n_kv, ctx): (usize, usize),
            row: usize,
        ) -> Result<Roomed<[DeviceBuffer<T>; N]>, GpuError> {
            let runs: Vec<Run> = (0..n_kv)
                .map(|h| Run {
                    at: h * ctx * row * size_of::<T>(),
                    row: row * size_of::<T>(),
                    per: 1,
                    rows: ctx,
                })
                .collect();
            Roomed::new(Arc::clone(stream), n_kv * ctx * row, &runs, |b| b)
        }
        Ok(match kv {
            KvQ8::F16 => {
                planes::<u16, 2>(stream, (n_kv, ctx), head)?.map(|[k, v]| KvPlanes::F16 { k, v })
            }
            KvQ8::Q8 => {
                let (codes, scales) = q8_plane_lens(head, 1, 1);
                Roomed::join(
                    planes::<u32, 2>(stream, (n_kv, ctx), codes)?,
                    planes::<u16, 2>(stream, (n_kv, ctx), scales)?,
                    |[kq, vq], [kd, vd]| KvPlanes::Q8 { kq, kd, vq, vd },
                )
            }
        })
    }

    /// The layer's planes of `d`'s shape in the `kv` format. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: &Dims, kv: KvQ8) -> Result<KvPlanes, GpuError> {
        KvPlanes::zeroed(stream, (d.n_kv, d.ctx, d.head), kv)
    }

    /// Zeroed planes of `n_kv` key heads, `ctx` rows and `head` values in
    /// the `kv` format. Load-time only.
    pub(super) fn zeroed(
        stream: &CudaStream,
        (n_kv, ctx, head): (usize, usize, usize),
        kv: KvQ8,
    ) -> Result<KvPlanes, GpuError> {
        Ok(match kv {
            KvQ8::F16 => {
                let n = n_kv * ctx * head;
                KvPlanes::F16 {
                    k: DeviceBuffer::zeroed(stream, n)?,
                    v: DeviceBuffer::zeroed(stream, n)?,
                }
            }
            KvQ8::Q8 => {
                let (words, scales) = q8_plane_lens(head, n_kv, ctx);
                KvPlanes::Q8 {
                    kq: DeviceBuffer::zeroed(stream, words)?,
                    kd: DeviceBuffer::zeroed(stream, scales)?,
                    vq: DeviceBuffer::zeroed(stream, words)?,
                    vd: DeviceBuffer::zeroed(stream, scales)?,
                }
            }
        })
    }

    /// Both sides' rows `0..d.ctx` set to the pattern `vals` (`n_kv · ctx ·
    /// head` values, f32): their f16 bits on an f16 cache, the q8_0 form of
    /// each whole 32-value block — the engine's one quantizer's rule
    /// (`model::arch::deepseek2::attn::quantize_q8_0`), packed
    /// ([`weights::q8_0_planes`]) — on a q8_0 one. Gate use; synchronizes
    /// with the caller.
    pub(super) fn fill(
        &mut self,
        stream: &CudaStream,
        vals: &[f32],
        d: &Dims,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "qwen3moe::KvPlanes::fill";
        let want = d.n_kv * d.ctx * d.head;
        if vals.len() != want {
            return Err(GpuError::shape(
                WHAT,
                format!("{} values of a {}-value plane", vals.len(), want),
            ));
        }
        match self {
            KvPlanes::F16 { k, v } => {
                let bits = vals.iter().map(|&x| f32_to_f16_bits(x)).collect::<Vec<_>>();
                k.copy_from_host(stream, &bits)?;
                v.copy_from_host(stream, &bits)?;
            }
            KvPlanes::Q8 { kq, kd, vq, vd } => {
                let blocks = vals
                    .as_chunks::<32>()
                    .0
                    .iter()
                    .map(|b| quantize_q8_0(b))
                    .collect::<Vec<_>>();
                let (codes, scales) = q8_0_planes(&blocks);
                kq.copy_from_host(stream, &codes)?;
                kd.copy_from_host(stream, &scales)?;
                vq.copy_from_host(stream, &codes)?;
                vd.copy_from_host(stream, &scales)?;
            }
        }
        Ok(())
    }

    /// The format these planes hold.
    pub(super) fn format(&self) -> KvQ8 {
        match self {
            KvPlanes::F16 { .. } => KvQ8::F16,
            KvPlanes::Q8 { .. } => KvQ8::Q8,
        }
    }

    /// The planes as the listed flash reads them, in either format.
    pub(super) fn planes(&self) -> CachePlanes<'_> {
        match self {
            KvPlanes::F16 { k, v } => CachePlanes::F16 { kc: k, vc: v },
            KvPlanes::Q8 { kq, kd, vq, vd } => CachePlanes::Q8 { kq, kd, vq, vd },
        }
    }

    /// The f16 planes, or a named refusal on a q8_0 cache: the paths that
    /// read K/V rows as f16 bits (the qwen3.5 family's host taps) load in f16
    /// only.
    pub(super) fn f16(
        &self,
        what: &'static str,
    ) -> Result<(&DeviceBuffer<u16>, &DeviceBuffer<u16>), GpuError> {
        match self {
            KvPlanes::F16 { k, v } => Ok((k, v)),
            KvPlanes::Q8 { .. } => Err(GpuError::state(
                what,
                "the f16 K/V planes (this cache runs q8_0)",
            )),
        }
    }

    /// [`KvPlanes::f16`]'s mutable form.
    pub(super) fn f16_mut(
        &mut self,
        what: &'static str,
    ) -> Result<(&mut DeviceBuffer<u16>, &mut DeviceBuffer<u16>), GpuError> {
        match self {
            KvPlanes::F16 { k, v } => Ok((k, v)),
            KvPlanes::Q8 { .. } => Err(GpuError::state(
                what,
                "the f16 K/V planes (this cache runs q8_0)",
            )),
        }
    }

    /// The head-128 norm, turn and append of `s`'s rows over these planes:
    /// [`enqueue_head_norm_neox_append`](crate::rope_neox::RopeNeoxKernels::enqueue_head_norm_neox_append)'s
    /// f16 launch, or its q8_0 twin's on a q8_0 cache. Asynchronous,
    /// allocation-free, capturable.
    ///
    /// # Safety
    /// Among the first `s.m` positions in `s.pos`, those below `s.ctx` are pairwise distinct.
    pub(super) unsafe fn append_128(
        &mut self,
        neox: &RopeNeoxKernels,
        stream: &CudaStream,
        s: Append128<'_>,
    ) -> Result<(), GpuError> {
        let Append128 {
            q,
            k,
            v,
            gq,
            gk,
            table,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
        } = s;
        match self {
            // SAFETY: the enclosing helper requires distinct cache-valid positions per sequence.
            KvPlanes::F16 { k: kc, v: vc } => unsafe {
                neox.enqueue_head_norm_neox_append(
                    stream,
                    NeoxArgs {
                        q,
                        k,
                        v,
                        gq,
                        gk,
                        table,
                        pos,
                        eps,
                        n_head,
                        n_kv,
                        ctx,
                        m,
                        fault,
                        cache_k: kc,
                        cache_v: vc,
                    },
                )
            },
            // SAFETY: the enclosing helper requires distinct cache-valid positions per sequence.
            KvPlanes::Q8 { kq, kd, vq, vd } => unsafe {
                neox.enqueue_head_norm_neox_append_q8(
                    stream,
                    NeoxQ8Args {
                        q,
                        k,
                        v,
                        gq,
                        gk,
                        table,
                        pos,
                        eps,
                        n_head,
                        n_kv,
                        ctx,
                        m,
                        fault,
                        kq,
                        kd,
                        vq,
                        vd,
                    },
                )
            },
        }
    }

    /// The head-256 norm, partial turn and append of `s`'s rows over these
    /// planes ([`KvPlanes::append_128`]'s head-256 form).
    ///
    /// # Safety
    ///
    /// Among the first `s.m` positions in `s.pos`, those below `s.ctx`
    /// are pairwise distinct.
    pub(super) unsafe fn append_256(
        &mut self,
        neox: &RopeNeoxKernels,
        stream: &CudaStream,
        s: Append256<'_>,
    ) -> Result<(), GpuError> {
        let Append256 {
            qg,
            q,
            k,
            v,
            gq,
            gk,
            table,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
        } = s;
        match self {
            KvPlanes::F16 { k: kc, v: vc } =>
            // SAFETY: the enclosing helper requires distinct cache-valid positions.
            unsafe {
                neox.enqueue_head_norm_neox_append_256(
                    stream,
                    PartialNeoxArgs {
                        qg,
                        q,
                        k,
                        v,
                        gq,
                        gk,
                        table,
                        pos,
                        eps,
                        n_head,
                        n_kv,
                        ctx,
                        m,
                        fault,
                        cache_k: kc,
                        cache_v: vc,
                    },
                )
            },
            KvPlanes::Q8 { kq, kd, vq, vd } =>
            // SAFETY: append_256 requires distinct cache-valid positions.
            unsafe {
                neox.enqueue_head_norm_neox_append_256_q8(
                    stream,
                    PartialNeoxQ8Args {
                        qg,
                        q,
                        k,
                        v,
                        gq,
                        gk,
                        table,
                        pos,
                        eps,
                        n_head,
                        n_kv,
                        ctx,
                        m,
                        fault,
                        kq,
                        kd,
                        vq,
                        vd,
                    },
                )
            },
        }
    }

    /// The head-128 NORM turn and append of `s`'s rows over these planes, no
    /// head norm: [`RopeNormKernels::enqueue_rope_norm_append`]'s f16 launch,
    /// or its q8_0 twin's on a q8_0 cache. Asynchronous, allocation-free,
    /// capturable.
    pub(super) fn append_norm(
        &mut self,
        norm: &RopeNormKernels,
        stream: &CudaStream,
        s: AppendNorm<'_>,
    ) -> Result<(), GpuError> {
        let AppendNorm {
            q,
            k,
            v,
            table,
            pos,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
        } = s;
        match self {
            KvPlanes::F16 { k: kc, v: vc } => norm.enqueue_rope_norm_append(
                stream,
                NormArgs {
                    q,
                    k,
                    v,
                    table,
                    pos,
                    n_head,
                    n_kv,
                    ctx,
                    m,
                    fault,
                    cache_k: kc,
                    cache_v: vc,
                },
            ),
            KvPlanes::Q8 { kq, kd, vq, vd } => norm.enqueue_rope_norm_append_q8(
                stream,
                NormQ8Args {
                    q,
                    k,
                    v,
                    table,
                    pos,
                    n_head,
                    n_kv,
                    ctx,
                    m,
                    fault,
                    kq,
                    kd,
                    vq,
                    vd,
                },
            ),
        }
    }

    /// The head-128 decode flash of `s`'s rows over these planes at the
    /// layout `flash` names: the eight-head pass ([`KvPlanes::flash_128`]), or
    /// the group-of-seven pass ([`FlashGqaKernels::enqueue_pass_g7`], its q8_0
    /// twin on a q8_0 cache), which is tensor-core only: `mma` false is
    /// refused by name. Any other layout has no head-128 entry.
    pub(super) fn flash_head128(
        &mut self,
        k: &FlashGqaKernels,
        stream: &CudaStream,
        s: FlashPass<'_>,
        flash: Flash,
        mma: bool,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "qwen3moe::KvPlanes::flash_head128";
        match flash {
            Flash::Group => self.flash_128(k, stream, s, mma),
            Flash::Seven => {
                let FlashPass {
                    q,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                } = s;
                match self {
                    KvPlanes::F16 { k: kc, v: vc } => k.enqueue_pass_g7(
                        stream,
                        GqaArgs {
                            q,
                            kc,
                            vc,
                            n_keys,
                            scale,
                            n_kv,
                            ctx,
                            m,
                            part_v,
                            part_ms,
                            fault,
                            y,
                        },
                        mma,
                    ),
                    KvPlanes::Q8 { kq, kd, vq, vd } if mma => k.enqueue_pass_g7_q8(
                        stream,
                        GqaQ8Args {
                            q,
                            kq,
                            kd,
                            vq,
                            vd,
                            n_keys,
                            scale,
                            n_kv,
                            ctx,
                            m,
                            part_v,
                            part_ms,
                            fault,
                            y,
                        },
                    ),
                    KvPlanes::Q8 { .. } => Err(q8_unserved(WHAT, flash, "scalar")),
                }
            }
            Flash::Quads | Flash::Pairs => Err(GpuError::shape(
                WHAT,
                format!("the {flash:?} layout, which only the head-256 flash entries have"),
            )),
        }
    }

    /// The head-128 prefill flash of `s`'s rows over these planes at the
    /// layout `flash` names: the eight-head pass ([`KvPlanes::prefill_128`]) or
    /// the group-of-seven pass ([`FlashGqaPrefill::enqueue_g7`], its q8_0 twin
    /// on a q8_0 cache). Any other layout has no head-128 entry.
    pub(super) fn prefill_head128(
        &mut self,
        prefill: &FlashGqaPrefill,
        stream: &CudaStream,
        s: PrefillFlash<'_>,
        flash: Flash,
    ) -> Result<(), GpuError> {
        match flash {
            Flash::Group => self.prefill_128(prefill, stream, s),
            Flash::Seven => {
                let PrefillFlash {
                    q,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                } = s;
                match self {
                    KvPlanes::F16 { k: kc, v: vc } => prefill.enqueue_g7(
                        stream,
                        GqaPrefillArgs {
                            q,
                            kc,
                            vc,
                            n_keys,
                            scale,
                            n_head,
                            n_kv,
                            ctx,
                            t,
                            fault,
                            y,
                        },
                    ),
                    KvPlanes::Q8 { kq, kd, vq, vd } => prefill.enqueue_g7_q8(
                        stream,
                        GqaPrefillQ8Args {
                            q,
                            kq,
                            kd,
                            vq,
                            vd,
                            n_keys,
                            scale,
                            n_head,
                            n_kv,
                            ctx,
                            t,
                            fault,
                            y,
                        },
                    ),
                }
            }
            Flash::Quads | Flash::Pairs => Err(GpuError::shape(
                "qwen3moe::KvPlanes::prefill_head128",
                format!("the {flash:?} layout, which only the head-256 flash entries have"),
            )),
        }
    }

    /// The head-128 decode flash of `s`'s rows over these planes
    /// ([`FlashGqaKernels::enqueue_pass`]'s f16 launch, the tensor-core pass
    /// when `mma`, or its q8_0 twin's, which is that pass only: the scalar
    /// pass on a q8_0 cache is refused by name).
    pub(super) fn flash_128(
        &mut self,
        flash: &FlashGqaKernels,
        stream: &CudaStream,
        s: FlashPass<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        let FlashPass {
            q,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = s;
        match self {
            KvPlanes::F16 { k: kc, v: vc } => flash.enqueue_pass::<GROUP>(
                stream,
                GqaArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
                mma,
            ),
            KvPlanes::Q8 { kq, kd, vq, vd } if mma => flash.enqueue_pass_q8(
                stream,
                GqaQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
            ),
            KvPlanes::Q8 { .. } => Err(q8_unserved(
                "qwen3moe::KvPlanes::flash_128",
                Flash::Group,
                "scalar",
            )),
        }
    }

    /// The head-256 decode flash of `s`'s rows over these planes, the pass
    /// `flash` names (the pairs' pass scalar only), the f16 launches or, on
    /// the eight-head or quad layout's tensor-core pass, its q8_0 twin's; any other
    /// pass on a q8_0 cache is refused by name.
    pub(super) fn flash_256(
        &mut self,
        k: &FlashGqaKernels,
        stream: &CudaStream,
        s: FlashPass<'_>,
        n_head: usize,
        flash: Flash,
        mma: bool,
    ) -> Result<(), GpuError> {
        let FlashPass {
            q,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = s;
        match (self, flash) {
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Group) => k.enqueue_pass_256(
                stream,
                GqaArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
                mma,
            ),
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Quads) => k.enqueue_pass_256_p4(
                stream,
                GqaArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
                n_head,
                mma,
            ),
            // The pairs' pass is scalar: the load refuses the tensor-core one.
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Pairs) => k.enqueue_pass_256_p2(
                stream,
                GqaArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
                n_head,
            ),
            (KvPlanes::Q8 { kq, kd, vq, vd }, Flash::Group) if mma => k.enqueue_pass_256_q8(
                stream,
                GqaQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
            ),
            (KvPlanes::Q8 { kq, kd, vq, vd }, Flash::Quads) if mma => k.enqueue_pass_256_p4_q8(
                stream,
                GqaQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    part_v,
                    part_ms,
                    fault,
                    y,
                },
                n_head,
            ),
            (KvPlanes::Q8 { .. }, flash) => Err(q8_unserved(
                "qwen3moe::KvPlanes::flash_256",
                flash,
                if mma { "tensor-core" } else { "scalar" },
            )),
            (KvPlanes::F16 { .. }, Flash::Seven) => Err(GpuError::shape(
                "qwen3moe::KvPlanes::flash_256",
                "the group-of-seven layout, which only the head-128 flash entries have",
            )),
        }
    }

    /// The head-128 prefill flash of `s`'s rows over these planes
    /// ([`FlashGqaPrefill::enqueue`]'s f16 launch or its q8_0 twin's).
    pub(super) fn prefill_128(
        &mut self,
        prefill: &FlashGqaPrefill,
        stream: &CudaStream,
        s: PrefillFlash<'_>,
    ) -> Result<(), GpuError> {
        let PrefillFlash {
            q,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = s;
        match self {
            KvPlanes::F16 { k: kc, v: vc } => prefill.enqueue(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
            KvPlanes::Q8 { kq, kd, vq, vd } => prefill.enqueue_q8(
                stream,
                GqaPrefillQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
        }
    }

    /// The head-256 prefill flash of `s`'s rows over these planes, the pass
    /// `flash` names, the f16 launches or, on the eight-head layout, its
    /// q8_0 twin's; a packed layout on a q8_0 cache is refused by name.
    pub(super) fn prefill_256(
        &mut self,
        k: &FlashGqaPrefill,
        stream: &CudaStream,
        s: PrefillFlash<'_>,
        flash: Flash,
    ) -> Result<(), GpuError> {
        let PrefillFlash {
            q,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = s;
        match (self, flash) {
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Group) => k.enqueue_256(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Quads) => k.enqueue_256_p4(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
            (KvPlanes::F16 { k: kc, v: vc }, Flash::Pairs) => k.enqueue_256_p2(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
            (KvPlanes::Q8 { kq, kd, vq, vd }, Flash::Group) => k.enqueue_256_q8(
                stream,
                GqaPrefillQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
            ),
            (KvPlanes::Q8 { .. }, flash @ (Flash::Quads | Flash::Pairs | Flash::Seven)) => Err(
                q8_unserved("qwen3moe::KvPlanes::prefill_256", flash, "prefill"),
            ),
            (KvPlanes::F16 { .. }, Flash::Seven) => Err(GpuError::shape(
                "qwen3moe::KvPlanes::prefill_256",
                "the group-of-seven layout, which only the head-128 flash entries have",
            )),
        }
    }

    /// Qwen3.8's gated pack-of-four prompt flash of `s`'s `[q | gate]` rows
    /// over these planes: `enqueue_256_p4_gated` on an f16 cache, its q8_0
    /// twin on a q8_0 one.
    pub(super) fn prefill_256_gated(
        &self,
        k: &FlashGqaPrefill,
        stream: &CudaStream,
        s: PrefillFlash<'_>,
        query: QueryNorm<'_>,
    ) -> Result<(), GpuError> {
        let PrefillFlash {
            q,
            n_keys,
            scale,
            n_head,
            n_kv,
            ctx,
            t,
            fault,
            y,
        } = s;
        match self {
            KvPlanes::F16 { k: kc, v: vc } => k.enqueue_256_p4_gated(
                stream,
                GqaPrefillArgs {
                    q,
                    kc,
                    vc,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
                query,
            ),
            KvPlanes::Q8 { kq, kd, vq, vd } => k.enqueue_256_p4_gated_q8(
                stream,
                GqaPrefillQ8Args {
                    q,
                    kq,
                    kd,
                    vq,
                    vd,
                    n_keys,
                    scale,
                    n_head,
                    n_kv,
                    ctx,
                    t,
                    fault,
                    y,
                },
                query,
            ),
        }
    }

    /// Whether the per-row launches over several sequences' planes
    /// ([`KvPlanes::append_128_rows`], [`KvPlanes::flash_128_rows`]) serve
    /// these planes under the flash pass `mma`: the f16 cache's tensor-core
    /// pass. The scalar pass and a q8_0 cache have no per-row entry.
    pub(super) fn rows_serve(&self, mma: bool) -> bool {
        mma && matches!(self, KvPlanes::F16 { .. })
    }

    /// Whether the launches over several sequences' caches with a table per
    /// sequence (`crate::flash_gqa::KvSeq`: the head-256 append and the
    /// listed flash) serve these planes: the f16 cache's. A q8_0 cache has no
    /// such entry.
    pub(super) fn seqs_serve(&self) -> bool {
        matches!(self, KvPlanes::F16 { .. })
    }

    /// These planes as the sequence of rows `rows` turned by `rope`
    /// ([`KvSeq`]); a q8_0 cache's are refused by name
    /// ([`KvPlanes::f16_mut`], [`KvPlanes::seqs_serve`]).
    pub(super) fn kv_seq<'a>(
        &'a mut self,
        rows: Range<usize>,
        rope: &'a DeviceBuffer<f32>,
    ) -> Result<KvSeq<'a>, GpuError> {
        let (k, v) = self.f16_mut("qwen3moe::KvPlanes::kv_seq")?;
        Ok(KvSeq { rows, k, v, rope })
    }

    /// These planes as rows `rows` of a per-row table
    /// ([`RowTable::put_f16`]); a q8_0 cache's are refused by name.
    pub(super) fn put_rows<'a>(
        &'a mut self,
        table: &mut RowTable<'a>,
        rows: Range<usize>,
    ) -> Result<(), GpuError> {
        match self {
            KvPlanes::F16 { k, v } => table.put_f16(rows, k, v),
            KvPlanes::Q8 { .. } => Err(GpuError::shape(
                "qwen3moe::KvPlanes::put_rows",
                "a q8_0 cache's planes in a per-row table: no per-row entry reads them \
                 (KvPlanes::rows_serve)",
            )),
        }
    }

    /// The head-128 norm, turn and append of `s`'s rows over several
    /// sequences' planes, row `t` into the planes `table` gives it
    /// ([`RopeNeoxKernels::enqueue_head_norm_neox_append_rows`]): each row
    /// [`KvPlanes::append_128`]'s row over its own planes. Asynchronous,
    /// allocation-free, capturable.
    ///
    /// # Safety
    /// Within each sequence in `table`, cache-valid positions in `s.pos` are pairwise distinct.
    pub(super) unsafe fn append_128_rows(
        neox: &RopeNeoxKernels,
        stream: &CudaStream,
        table: &RowTable<'_>,
        s: Append128<'_>,
    ) -> Result<(), GpuError> {
        let Append128 {
            q,
            k,
            v,
            gq,
            gk,
            table: rope,
            pos,
            eps,
            n_head,
            n_kv,
            ctx,
            m,
            fault,
        } = s;
        // SAFETY: the enclosing helper requires distinct cache-valid positions per sequence.
        unsafe {
            neox.enqueue_head_norm_neox_append_rows(
                stream,
                table,
                NeoxRowsArgs {
                    q,
                    k,
                    v,
                    gq,
                    gk,
                    table: rope,
                    pos,
                    eps,
                    n_head,
                    n_kv,
                    ctx,
                    m,
                    fault,
                },
            )
        }
    }

    /// The head-128 tensor-core decode flash of `s`'s rows over several
    /// sequences' planes, row `t` reading the planes `table` gives it
    /// ([`FlashGqaKernels::enqueue_pass_rows`]): each row
    /// [`KvPlanes::flash_128`]'s row over its own planes. The scalar pass
    /// (`mma` false) is refused by name: no per-row entry serves it.
    /// Asynchronous, allocation-free, capturable.
    pub(super) fn flash_128_rows(
        flash: &FlashGqaKernels,
        stream: &CudaStream,
        table: &RowTable<'_>,
        s: FlashPass<'_>,
        mma: bool,
    ) -> Result<(), GpuError> {
        if !mma {
            return Err(GpuError::shape(
                "qwen3moe::KvPlanes::flash_128_rows",
                "the scalar pass over a per-row table: the per-row flash is the tensor-core \
                 pass (KvPlanes::rows_serve)",
            ));
        }
        let FlashPass {
            q,
            n_keys,
            scale,
            n_kv,
            ctx,
            m,
            part_v,
            part_ms,
            fault,
            y,
        } = s;
        flash.enqueue_pass_rows(
            stream,
            table,
            GqaRowsArgs {
                q,
                n_keys,
                scale,
                n_kv,
                ctx,
                m,
                part_v,
                part_ms,
                fault,
                y,
            },
        )
    }

    /// The planes' rows of the positions below `n` as a sequence state
    /// carries them ([`Spans::by_position`]), each plane `[n_kv][ctx][·]`:
    /// the K and the V plane of `head` f16 a row, or on a q8_0 cache the
    /// codes and scales planes of each side ([`q8_plane_lens`]'s rows), so a
    /// prefix of the state moves only its own rows in either format.
    pub(super) fn spans(
        &mut self,
        (n_kv, ctx, head): (usize, usize, usize),
        n: usize,
    ) -> Result<Vec<Spans<'_>>, GpuError> {
        let planes: Vec<(Plane<'_>, usize)> = match self {
            KvPlanes::F16 { k, v } => vec![(k.into(), head), (v.into(), head)],
            KvPlanes::Q8 { kq, kd, vq, vd } => vec![
                (kq.into(), head / 4),
                (kd.into(), head / 32),
                (vq.into(), head / 4),
                (vd.into(), head / 32),
            ],
        };
        planes
            .into_iter()
            .map(|(plane, row)| Spans::by_position(plane, n_kv, ctx * row, n, row))
            .collect()
    }

    pub(super) fn bytes(&self) -> usize {
        fn bytes<T>(b: &DeviceBuffer<T>) -> usize {
            b.num_bytes()
        }
        match self {
            KvPlanes::F16 { k, v } => bytes(k) + bytes(v),
            KvPlanes::Q8 { kq, kd, vq, vd } => bytes(kq) + bytes(kd) + bytes(vq) + bytes(vd),
        }
    }
}

impl Views for KvPlanes {
    fn dissolve(self) {
        match self {
            KvPlanes::F16 { k, v } => {
                k.dissolve();
                v.dissolve();
            }
            KvPlanes::Q8 { kq, kd, vq, vd } => {
                kq.dissolve();
                kd.dissolve();
                vq.dissolve();
                vd.dissolve();
            }
        }
    }
}

/// What an arena holds beyond the K-quant chain's buffers, from the sites
/// its plans launch (`Forms::of`): the wide arm's activation forms — each
/// input's q8_1 blocks of 128 for a K-quant site, its q8 blocks of 32 for a
/// Q8_0 site, and the SwiGLU rows' f32 rows for a down that reads those
/// (`WideForms`) — and on the gemv arm the gate and up rows of an unfused
/// gate·up, and the row-major output of a K-quant site launched alone at
/// more than one row (`cols` values a column); and on a chain with a layer
/// that keeps its shared expert apart, that expert's buffers (`apart`); and
/// on a chain whose gated head-256 layer has an output projection of a type
/// other than a K-quant, the f32 rows of the gated flash output that
/// projection reads (`out_gate`).
/// [`Forms::KQUANT`] is a chain of fused Q4_K and Q6_K sites: the buffers
/// the arena held before any other type ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Forms {
    pub(super) hid: WideForms,
    pub(super) attn: WideForms,
    pub(super) h: WideForms,
    pub(super) glu: bool,
    pub(super) cols: usize,
    pub(super) apart: Option<ApartForms>,
    pub(super) out_gate: bool,
}

/// What a chain whose layer keeps its shared expert apart from its routed
/// stacks (`plan::SharedPlan::Apart`) reads beyond a joined chain: on the
/// wide arm, the forms of the shared expert's SwiGLU rows its downs read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ApartForms {
    pub(super) h: WideForms,
}

impl Forms {
    pub(super) const KQUANT: Forms = Forms {
        hid: WideForms::q8(true, false),
        attn: WideForms::q8(true, false),
        h: WideForms::q8(true, false),
        glu: false,
        cols: 0,
        apart: None,
        out_gate: false,
    };
}

/// The gemv arm's gate and up rows of an unfused gate·up, `slots · ff` a
/// token each: the SwiGLU reads both into `h`.
pub(super) struct Glu {
    pub(super) g: DeviceBuffer<f32>,
    pub(super) u: DeviceBuffer<f32>,
}

/// The gemv arm's buffers of a layer that keeps its shared expert apart
/// (`Forms::apart`), for at most [`GEMV_COLS`] tokens: the routed slots'
/// card list (`slots` a token: the routed ids, the shared slot
/// [`crate::hybrid::HOST`]), the shared expert's gate, up and SwiGLU rows
/// (`ff` a token) with the SwiGLU's q8_1 form (`act[m − 1]` of `m` columns),
/// its output (`hidden` a token), and the host sum the combine adds: zero,
/// since a whole-card load serves every routed slot on the card.
pub(super) struct ApartRows {
    pub(super) sel: DeviceBuffer<u32>,
    pub(super) g: DeviceBuffer<f32>,
    pub(super) u: DeviceBuffer<f32>,
    pub(super) h: DeviceBuffer<f32>,
    pub(super) act: Vec<Q8Act>,
    pub(super) y: DeviceBuffer<f32>,
    pub(super) zero: DeviceBuffer<f32>,
}

impl ApartRows {
    /// The buffers for `narrow` tokens of `d`, every one zero. Load-time
    /// only.
    fn new(stream: &CudaStream, d: &Dims, narrow: usize) -> Result<ApartRows, GpuError> {
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(ApartRows {
            sel: DeviceBuffer::zeroed(stream, narrow * d.slots())?,
            g: f(narrow * d.ff)?,
            u: f(narrow * d.ff)?,
            h: f(narrow * d.ff)?,
            act: (1..=narrow)
                .map(|m| Q8Act::with_k(stream, m, d.ff))
                .collect::<Result<Vec<_>, _>>()?,
            y: f(narrow * d.hidden)?,
            zero: f(narrow * d.hidden)?,
        })
    }

    /// Device bytes.
    fn bytes(&self) -> usize {
        [&self.g, &self.u, &self.h, &self.y, &self.zero]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.sel.num_bytes()
            + self.act.iter().map(Q8Act::device_bytes).sum::<usize>()
    }
}

/// The arena, in chain order: every intermediate of one layer for up to
/// `rows` tokens, token-major, shared by all layers (they run in turn). The
/// decode step's arena has one row, a pass's [`GEMV_COLS`], a prompt's up to
/// its ubatch size; a unit of `m <= rows` tokens uses the first `m` rows.
/// What only an op's gemv arm reads — the `m`-column activations, the flash
/// pass's partials, a Q6_K projection's row-major copy, the row windows — is
/// cut for at most [`GEMV_COLS`] rows; past that the ops run their wide arm
/// over the arena's [`Wide`] part.
pub(super) struct Arena {
    pub(super) dims: Dims,
    pub(super) rows: usize,
    /// The layer's input residual (the embedding rows for layer 0), and its
    /// output: the combine writes the next layer's input here.
    pub(super) x: DeviceBuffer<f32>,
    /// Row `t` of `x`, for `t` below `min(rows, GEMV_COLS)`.
    pub(super) x_rows: Vec<WindowMut<'static, f32>>,
    /// Row `t`'s position and live key count, which the embedding launch
    /// writes: the rope turns by the table's row `pos[t]` and appends there,
    /// the flash attends over `n_keys[t]` keys.
    pub(super) pos: DeviceBuffer<u32>,
    pub(super) n_keys: DeviceBuffer<u32>,
    pub(super) normed: DeviceBuffer<f32>,
    /// q8_1 of the attention-normed rows, `act_x[m − 1]` for `m` of them: q,
    /// k and v all read it.
    pub(super) act_x: Vec<Q8Act>,
    /// The q, k and v rows in one allocation — the one output of the q·k·v
    /// launch — and a non-owning window onto each block of `rows` tokens.
    /// With an output gate the query block holds each head's `[q | gate]`.
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) q: WindowMut<'static, f32>,
    pub(super) k: WindowMut<'static, f32>,
    pub(super) v: WindowMut<'static, f32>,
    /// The normed and turned queries, `[rows][n_head][head]`, when the query
    /// block holds gates beside them (the rope writes them out of place);
    /// `None` when the rope turns `q` in place.
    pub(super) q_out: Option<DeviceBuffer<f32>>,
    /// The flash output times the sigmoid of its gate, `[rows][n_head][head]`,
    /// the f32 rows an output projection other than a K-quant reads
    /// ([`Forms::out_gate`]); `None` on a chain with no such layer. A buffer of
    /// its own, so `q_out` still holds the queries the flash read.
    pub(super) gated_rows: Option<DeviceBuffer<f32>>,
    /// A Q6_K value projection's output at more than one token: `q6k_gemv`
    /// writes it row-major, and a copy puts it into `v` token-major. `None`
    /// on a one-row arena, where the gemv writes `v` itself.
    pub(super) v_cols: Option<DeviceBuffer<f32>>,
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The attention rows, `n_head · head` per token; a delta layer's gated
    /// norm writes its `n_v · 128` values a token here too, the output
    /// projection's input either way.
    pub(super) attn: DeviceBuffer<f32>,
    pub(super) act_attn: Vec<Q8Act>,
    /// The FFN's input residual: `x` plus the attention output.
    pub(super) ffn_inp: DeviceBuffer<f32>,
    /// q8_1 of the FFN-normed rows, the experts' gate·up input. At more
    /// than one row the router reads their f32 twin `normed`; at one row the
    /// router's launch writes these and keeps the normed row to itself.
    pub(super) act_ffn: Vec<Q8Act>,
    /// The router's ids, token-major `slots` per token, are the down's
    /// selector: slot `t · slots + j` of a pass is token `t`'s slot `j`.
    pub(super) route: Route,
    /// The selected experts' SwiGLU rows, per token slot-major `slots · ff`.
    pub(super) h: DeviceBuffer<f32>,
    /// q8_1 of `h`, one column per slot: `act_h[m − 1]` holds the `m ·
    /// slots` columns of `m` tokens, the down's input.
    pub(super) act_h: Vec<Q8Act>,
    /// The down outputs, per token slot-major `slots · hidden`.
    pub(super) down: DeviceBuffer<f32>,
    /// A delta layer's intermediates; `None` when the chain has none.
    pub(super) gdn: Option<GdnArena>,
    /// What the ops' wide arm reads; `Some` iff `rows > GEMV_COLS`.
    pub(super) wide: Option<Wide>,
    /// The gemv arm's unfused gate·up rows (`Forms::glu`).
    pub(super) glu: Option<Glu>,
    /// The gemv arm's row-major output of a K-quant site launched alone, at
    /// more than one row (`Forms::cols` values a column); `None` on a
    /// one-row arena.
    pub(super) cols: Option<DeviceBuffer<f32>>,
    /// The gemv arm's buffers of a layer that keeps its shared expert apart
    /// (`Forms::apart`); `None` on a chain whose every layer joins it.
    pub(super) apart: Option<ApartRows>,
    /// The media call's rows the unit about to walk takes over its
    /// embeddings ([`Arena::write_media`]); `Some` only on a prompt arena of
    /// a media load.
    pub(super) media: Option<MediaRows>,
}

/// Where the FFN's slots come from: a router launch's results — the plain
/// router's `k` slots a token, or the gated router's `k + 1` (the shared
/// expert's last) — or a dense chain's fixed route.
pub(super) enum Route {
    Plain(RouterOut),
    Gated(RouterOut),
    Dense(DenseRoute),
}

/// A dense FFN's slots: token `t`'s one slot is slot `t`, on expert 0 of the
/// one-expert stacks at weight 1, written once at load and never again. So
/// the routed launches after the router run a dense FFN as they run a routed
/// one, and the combine's `1·down + resid` is `down + resid` exactly.
pub(super) struct DenseRoute {
    ids: DeviceBuffer<u32>,
    weights: DeviceBuffer<f32>,
}

impl DenseRoute {
    /// The route of `rows` tokens. Load-time only.
    fn new(stream: &CudaStream, rows: usize) -> Result<DenseRoute, GpuError> {
        Ok(DenseRoute {
            ids: DeviceBuffer::zeroed(stream, rows)?,
            weights: DeviceBuffer::from_host(stream, &vec![1.0f32; rows])?,
        })
    }
}

impl Route {
    /// The slots' expert ids, token-major.
    pub(super) fn ids(&self) -> &DeviceBuffer<u32> {
        match self {
            Route::Plain(r) => &r.ids,
            Route::Gated(r) => &r.ids,
            Route::Dense(r) => &r.ids,
        }
    }

    /// The slots' weights, token-major.
    pub(super) fn weights(&self) -> &DeviceBuffer<f32> {
        match self {
            Route::Plain(r) => &r.weights,
            Route::Gated(r) => &r.weights,
            Route::Dense(r) => &r.weights,
        }
    }

    /// The plain router's buffers, or a named refusal.
    pub(super) fn plain(&mut self, what: &'static str) -> Result<&mut RouterOut, GpuError> {
        match self {
            Route::Plain(r) => Ok(r),
            Route::Gated(_) | Route::Dense(_) => {
                Err(GpuError::state(what, "the plain router's buffers"))
            }
        }
    }

    /// The gated router's buffers, or a named refusal.
    pub(super) fn gated(&mut self, what: &'static str) -> Result<&mut RouterOut, GpuError> {
        match self {
            Route::Gated(r) => Ok(r),
            Route::Plain(_) | Route::Dense(_) => {
                Err(GpuError::state(what, "the gated router's buffers"))
            }
        }
    }

    fn bytes(&self) -> usize {
        match self {
            Route::Plain(r) => r.bytes(),
            Route::Gated(r) => r.bytes(),
            Route::Dense(r) => r.ids.num_bytes() + r.weights.num_bytes(),
        }
    }
}

/// A delta layer's intermediates for up to `rows` tokens, token-major. The
/// four input projections land in one allocation, `[x | z | b | a]` in
/// blocks of `rows` tokens: the q·k·v channels `x` (`C` a token), the output
/// gate `z` (`n_v · 128`), β's and α's raw projections (`n_v` each) — so
/// one three-matrix launch writes three neighbouring blocks.
pub(super) struct GdnArena {
    pub(super) shape: LinearShape,
    pub(super) proj: DeviceBuffer<f32>,
    /// The four blocks, and the tails from `z` and from `b` a three- or
    /// two-matrix launch writes.
    pub(super) x: WindowMut<'static, f32>,
    pub(super) z: WindowMut<'static, f32>,
    pub(super) b: WindowMut<'static, f32>,
    pub(super) a: WindowMut<'static, f32>,
    pub(super) from_z: WindowMut<'static, f32>,
    pub(super) from_b: WindowMut<'static, f32>,
    /// A Q6_K q·k·v projection's output at more than one token, row-major
    /// as `q6k_gemv` writes it, for at most [`GEMV_COLS`] tokens; a copy
    /// puts it into `x` token-major. `None` on a one-row arena.
    pub(super) x_cols: Option<DeviceBuffer<f32>>,
    /// The conv's output `[rows][C]` (q normed and scaled, k normed, v),
    /// β and decay `[rows][n_v]`, the delta output `[rows][n_v][128]`.
    pub(super) conv: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
    pub(super) decay: DeviceBuffer<f32>,
    pub(super) o: DeviceBuffer<f32>,
    /// The lanes the walks over this arena run their delta launches by
    /// ([`LaneRun`]); `None` on a load whose stores keep one lane in place.
    pub(super) lanes: Option<LaneRun>,
}

/// How the delta launches of the walks over one arena use the stamped lanes
/// of a load that verifies drafted rows: the body's lane word (a
/// non-owning view of [`LaneWord`]'s device word, so a walk reads the
/// committed lane whatever its input record holds) and whether the walk is
/// a verify, whose row `j` writes lane `(c + j) mod lanes` (`each`), or one
/// that writes the state after its last row back into lane `c`.
pub(super) struct LaneRun {
    pub(super) word: WindowMut<'static, u32>,
    pub(super) each: bool,
}

impl GdnArena {
    fn new(stream: &CudaStream, shape: LinearShape, rows: usize) -> Result<GdnArena, GpuError> {
        let (c, zl, nv) = (shape.channels(), shape.n_v * linear::HEAD, shape.n_v);
        let at_z = rows * c;
        let at_b = at_z + rows * zl;
        let at_a = at_b + rows * nv;
        let end = at_a + rows * nv;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let proj = f(end)?;
        // SAFETY: each window lies inside `proj` (`end` values): the blocks
        // tile it and the two tails run from their block to its end; `proj`
        // moves into the struct beside them (a move of the handle, not of the
        // allocation), where it outlives them.
        let (x, z, b, a, from_z, from_b) = unsafe {
            (
                cells_view(&proj, 0, at_z),
                cells_view(&proj, at_z, rows * zl),
                cells_view(&proj, at_b, rows * nv),
                cells_view(&proj, at_a, rows * nv),
                cells_view(&proj, at_z, end - at_z),
                cells_view(&proj, at_b, end - at_b),
            )
        };
        Ok(GdnArena {
            shape,
            proj,
            x,
            z,
            b,
            a,
            from_z,
            from_b,
            x_cols: (rows > 1).then(|| f(rows.min(GEMV_COLS) * c)).transpose()?,
            conv: f(rows * c)?,
            beta: f(rows * nv)?,
            decay: f(rows * nv)?,
            o: f(rows * zl)?,
            lanes: None,
        })
    }

    /// Where the `b` and `a` blocks start in the tail from `z`, and the `a`
    /// block in the tail from `b`: `(b − z, a − z, a − b)`.
    pub(super) fn tail_offsets(&self) -> (usize, usize, usize) {
        let (z, b) = (self.z.len(), self.b.len());
        (z, z + b, b)
    }

    fn bytes(&self) -> usize {
        [&self.proj, &self.conv, &self.beta, &self.decay, &self.o]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.x_cols.as_ref().map_or(0, DeviceBuffer::num_bytes)
    }
}

/// Non-owning views of a [`LaneWord`]'s device word ([`LaneWord::with_views`]).
pub(super) type LaneViews = Vec<WindowMut<'static, u32>>;

/// The lane word every delta launch of a verifying load's walks reads: the
/// lane that holds the committed state, which only a verify's commit moves
/// (`linear::delta::kept_lane`) and a reset sets back to 0. One device word
/// written by an asynchronous copy ahead of the launches that read it, over
/// `lanes` lanes.
pub(super) struct LaneWord {
    word: WindowMut<'static, u32>,
    inbox: Inbox,
    lane: u32,
    lanes: usize,
}

impl LaneWord {
    /// Lane 0 of `lanes` lanes. Load-time only.
    pub(super) fn new(stream: &CudaStream, lanes: usize) -> Result<LaneWord, GpuError> {
        Ok(LaneWord::with_views(stream, lanes, 0)?.0)
    }

    /// Lane 0 of `lanes` lanes, and `views` non-owning views of its device
    /// word for the arenas whose walks read it ([`Arena::run_lanes`]). A view
    /// must not outlive the word, nor be used after the struct is dropped:
    /// the body that holds both keeps them together. Load-time only.
    pub(super) fn with_views(
        stream: &CudaStream,
        lanes: usize,
        views: usize,
    ) -> Result<(LaneWord, LaneViews), GpuError> {
        let inbox = Inbox::new(stream, 1)?;
        // SAFETY: word 0 of the inbox's one device word for each window, and
        // the inbox moves into the struct beside the first of them (a move of
        // the handle, not of the allocation), where it outlives it; the views
        // are the caller's to keep inside that lifetime (above).
        let (word, views) = unsafe {
            (
                param_view::<u32>(inbox.dev(), 0, 1),
                (0..views)
                    .map(|_| param_view::<u32>(inbox.dev(), 0, 1))
                    .collect(),
            )
        };
        Ok((
            LaneWord {
                word,
                inbox,
                lane: 0,
                lanes,
            },
            views,
        ))
    }

    /// Set the word to `lane` (below the word's lanes, else refused by name)
    /// and enqueue its copy. Never inside a capture.
    pub(super) fn set(&mut self, stream: &CudaStream, lane: u32) -> Result<(), GpuError> {
        if lane as usize >= self.lanes {
            return Err(GpuError::shape(
                "qwen3moe::LaneWord::set",
                format!("lane {lane} of {} lanes", self.lanes),
            ));
        }
        self.inbox.host_mut()?[0] = lane;
        self.inbox.upload(stream, 1)?;
        self.lane = lane;
        Ok(())
    }

    /// The word after a verify from the committed lane keeps its first `k`
    /// rows: the lane row `k − 1` wrote ([`linear::delta::kept_lane`]),
    /// enqueued as [`LaneWord::set`] does. A `k` outside `1..=lanes` is
    /// refused by name, the word where it was.
    pub(super) fn keep(&mut self, stream: &CudaStream, k: usize) -> Result<(), GpuError> {
        let lanes = u32::try_from(self.lanes).map_err(|_| {
            GpuError::shape(
                "qwen3moe::LaneWord::keep",
                format!("{} lanes past u32", self.lanes),
            )
        })?;
        let k = u32::try_from(k)
            .ok()
            .filter(|k| (1..=lanes).contains(k))
            .ok_or_else(|| {
                GpuError::shape(
                    "qwen3moe::LaneWord::keep",
                    format!("{k} kept rows of a verify over {lanes} lanes (1..={lanes})"),
                )
            })?;
        self.set(stream, linear::delta::kept_lane(self.lane, k, lanes))
    }

    /// The lane the word holds.
    pub(super) fn lane(&self) -> u32 {
        self.lane
    }

    /// The device word.
    pub(super) fn word(&self) -> &DeviceBuffer<u32> {
        &self.word
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// A layer's state store: K/V planes by position, or a recurrent store.
pub(super) enum LayerStore {
    Kv(KvPlanes),
    Rec(RecStore),
}

/// A layer's store as one layer's enqueue borrows it.
pub(super) enum StoreMut<'a> {
    Kv(&'a mut KvPlanes),
    Rec(&'a mut RecStore),
}

impl LayerStore {
    pub(super) fn as_mut(&mut self) -> StoreMut<'_> {
        match self {
            Self::Kv(k) => StoreMut::Kv(k),
            Self::Rec(r) => StoreMut::Rec(r),
        }
    }

    pub(super) fn bytes(&self) -> usize {
        match self {
            Self::Kv(k) => k.bytes(),
            Self::Rec(r) => r.bytes(),
        }
    }
}

/// The committed lane of a recurrent store of one lane: the lane word every
/// record carries while a load holds one lane.
pub(super) const LANE: u32 = 0;

/// The decode step's input record — its position and its token, and on a
/// chain with recurrent layers the lane word ([`SP_LANE`]) — in one
/// [`Inbox`], so a step's refresh is one fill and one copy, and the windows
/// the captured graph reads.
pub(super) struct StepParams {
    /// Non-owning windows into the inbox's device words: the token (the
    /// embedding's one id), the position word and the lane word.
    token: WindowMut<'static, u32>,
    pos0: WindowMut<'static, u32>,
    lane: Option<WindowMut<'static, u32>>,
    words: usize,
    inbox: Inbox,
}

/// The lane word's offset in the decode step's record, after its token.
pub(super) const SP_LANE: usize = IN_IDS + 1;

/// What a unit's first launch reads: its ids (the unit's row count is their
/// length), the word holding the first position of the input record they
/// are a window of, and where the window starts in that record — row `t` of
/// the unit is position `pos0 + first + t`; and on a chain with recurrent
/// layers the record's lane word, the state lane a delta launch reads and
/// writes.
pub(super) struct Io<'a> {
    pub(super) ids: &'a DeviceBuffer<u32>,
    pub(super) pos0: &'a DeviceBuffer<u32>,
    pub(super) first: usize,
    pub(super) lane: Option<&'a DeviceBuffer<u32>>,
}

impl StepParams {
    /// A zeroed record (token 0 at position 0), with the lane word when
    /// `lane`. Load-time only.
    pub(super) fn new(stream: &CudaStream, lane: bool) -> Result<StepParams, GpuError> {
        let words = IN_IDS + 1 + usize::from(lane);
        let inbox = Inbox::new(stream, words)?;
        // SAFETY: each window is one word of the inbox's `words` device words
        // (`IN_POS0`, `IN_IDS` < `words`, and `SP_LANE` < `words` when the
        // lane word is asked for), and the inbox moves into the struct beside
        // them (a move of the handle, not of the allocation), where it
        // outlives them.
        let (token, pos0, lane) = unsafe {
            (
                param_view::<u32>(inbox.dev(), IN_IDS, 1),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                lane.then(|| param_view::<u32>(inbox.dev(), SP_LANE, 1)),
            )
        };
        Ok(StepParams {
            token,
            pos0,
            lane,
            words,
            inbox,
        })
    }

    /// Write `token` at position `pos` (and the lane word [`LANE`]) and
    /// enqueue its copy — the step's refresh. Asynchronous: the step's
    /// launches behind it read it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: u32,
    ) -> Result<(), GpuError> {
        let has_lane = self.lane.is_some();
        let host = self.inbox.host_mut()?;
        put_input(host, &[token], pos)?;
        if has_lane {
            host[SP_LANE] = LANE;
        }
        self.inbox.upload(stream, self.words)
    }

    /// [`StepParams::write`] of a step at `pos` whose token is word 0 of
    /// `token` on the card: the record's copy, then that word copied over
    /// the record's token, both on `stream` behind the work that writes it.
    pub(super) fn write_chained(
        &mut self,
        stream: &CudaStream,
        token: &DeviceBuffer<u32>,
        pos: u32,
    ) -> Result<(), GpuError> {
        self.write(stream, 0, pos)?;
        // SAFETY: word 0 of `token`, which the caller holds for the call; the
        // window lives for this enqueue.
        let src = unsafe { param_view::<u32>(token, 0, 1) };
        self.token.copy_from_device_async(&src, stream)?;
        Ok(())
    }

    /// The step's input, as its first launch reads it.
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.token,
            pos0: &self.pos0,
            first: 0,
            lane: self.lane.as_deref(),
        }
    }

    /// Device bytes of the record.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// A pass's input record — its first position, up to `ROWS` ids and the
/// lane word after them — and the windows a pass of `m` rows reads, cut once
/// at load so a captured pass of each width reads fixed addresses.
pub(super) struct PassRecord<const ROWS: usize> {
    /// Ids `0..m` at `m − 1`.
    ids: Vec<WindowMut<'static, u32>>,
    pos0: WindowMut<'static, u32>,
    lane: WindowMut<'static, u32>,
    /// The rows the last write planned.
    planned: Option<usize>,
    inbox: Inbox,
}

impl<const ROWS: usize> PassRecord<ROWS> {
    /// The lane word's offset in the record, after its ids.
    const LANE_AT: usize = IN_IDS + ROWS;
    /// Words of the record.
    const WORDS: usize = Self::LANE_AT + 1;

    /// A zeroed record. Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<Self, GpuError> {
        const { assert!(ROWS > 0, "a pass record holds at least one row") };
        let inbox = Inbox::new(stream, Self::WORDS)?;
        // SAFETY: every window lies inside the inbox's `WORDS` device words
        // (`IN_IDS + m <= LANE_AT < WORDS`, `IN_POS0 < WORDS`), and the inbox
        // moves into the struct beside them (a move of the handle, not of the
        // allocation), where it outlives them.
        let (ids, pos0, lane) = unsafe {
            (
                (1..=ROWS)
                    .map(|m| param_view::<u32>(inbox.dev(), IN_IDS, m))
                    .collect(),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
                param_view::<u32>(inbox.dev(), Self::LANE_AT, 1),
            )
        };
        Ok(PassRecord {
            ids,
            pos0,
            lane,
            planned: None,
            inbox,
        })
    }

    /// Write `tokens` from position `pos` and the lane word, and enqueue the
    /// record's copy. Asynchronous: the pass behind it reads it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos: u32,
    ) -> Result<(), GpuError> {
        if !(1..=ROWS).contains(&tokens.len()) {
            return Err(GpuError::shape(
                "qwen3moe::PassRecord::write",
                format!("{} rows; a pass takes 1..={ROWS}", tokens.len()),
            ));
        }
        let host = self.inbox.host_mut()?;
        put_input(host, tokens, pos)?;
        host[Self::LANE_AT] = LANE;
        self.inbox.upload(stream, Self::WORDS)?;
        self.planned = Some(tokens.len());
        Ok(())
    }

    /// The rows the last write planned; `None` before the first.
    pub(super) fn planned(&self) -> Option<usize> {
        self.planned
    }

    /// The input of a pass of `m` rows, as its first launch reads it.
    pub(super) fn io(&self, m: usize) -> Result<Io<'_>, GpuError> {
        let ids = m
            .checked_sub(1)
            .and_then(|i| self.ids.get(i))
            .ok_or_else(|| {
                GpuError::shape(
                    "qwen3moe::PassRecord::io",
                    format!("a pass of {m} rows (1..={ROWS})"),
                )
            })?;
        Ok(Io {
            ids,
            pos0: &self.pos0,
            first: 0,
            lane: Some(&self.lane),
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// A non-owning window of `len` `T` over `parent`, from element `off` of the
/// parent's u32 grid. Dropping it releases its context handle and frees
/// nothing.
///
/// # Safety
///
/// `off + len · size_of::<T>() / 4` must be within `parent`, `T` four-byte
/// aligned, and `parent` must outlive the window unmoved in memory (a
/// captured graph bakes the address in).
pub(super) unsafe fn param_view<T>(
    parent: &DeviceBuffer<u32>,
    off: usize,
    len: usize,
) -> WindowMut<'static, T> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<u32>()) as u64;
    // SAFETY: the range is the caller's contract above, inside `parent`.
    unsafe { WindowMut::from_raw(ptr, len, parent.context()) }
}

/// A non-owning window of `len` cells over `parent` from cell `off`,
/// released as [`param_view`]'s.
///
/// # Safety
///
/// `off + len <= parent.len()`, and `parent` must outlive the window unmoved
/// in memory (a captured graph bakes the address in).
pub(super) unsafe fn cells_view<T>(
    parent: &DeviceBuffer<T>,
    off: usize,
    len: usize,
) -> WindowMut<'static, T> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<T>()) as u64;
    // SAFETY: the range is the caller's contract above, inside `parent`.
    unsafe { WindowMut::from_raw(ptr, len, parent.context()) }
}

impl Arena {
    /// Allocate the arena of a chain of fused K-quant sites for `d` and up
    /// to `rows` tokens ([`Forms::KQUANT`]). Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims, rows: usize) -> Result<Arena, GpuError> {
        Arena::with(stream, d, rows, Forms::KQUANT)
    }

    /// Allocate the arena for `d`, up to `rows` tokens and the sites' forms
    /// `forms`: `wide::arena_bytes` device bytes. Load-time only.
    pub(super) fn with(
        stream: &CudaStream,
        d: Dims,
        rows: usize,
        forms: Forms,
    ) -> Result<Arena, GpuError> {
        let (q_len, kv_len, attn_len) = (d.q_rows, d.kv_len(), d.attn_len());
        let gated = q_len != attn_len;
        let narrow = rows.min(GEMV_COLS);
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let acts = |k: usize| {
            (1..=narrow)
                .map(|m| Q8Act::with_k(stream, m, k))
                .collect::<Result<Vec<_>, _>>()
        };
        let x = f(rows * d.hidden)?;
        let qkv = f(rows * (q_len + 2 * kv_len))?;
        let route = match d.router {
            None => Route::Dense(DenseRoute::new(stream, rows)?),
            Some(r) => {
                let bufs = if rows > MAX_TOKENS {
                    RouterOut::for_ubatch(stream, r, rows)?
                } else {
                    RouterOut::with_tokens(stream, r, rows)?
                };
                if r.gated() {
                    Route::Gated(bufs)
                } else {
                    Route::Plain(bufs)
                }
            }
        };
        let part_v = if d.head == HEAD_256 {
            partials_v_len_256(narrow, d.n_head)
        } else {
            partials_v_len(narrow, d.n_head)
        };
        // SAFETY: every window below lies inside its parent — row `t <
        // narrow <= rows` of `x` (`hidden` values), and the three blocks that tile `qkv`
        // (`rows · (q_len + 2·kv_len)`) — and each parent moves into the
        // arena beside its windows (a move of the handle, not of the
        // allocation), where it outlives them.
        let (x_rows, q, k, v) = unsafe {
            (
                (0..narrow)
                    .map(|t| cells_view(&x, t * d.hidden, d.hidden))
                    .collect(),
                cells_view(&qkv, 0, rows * q_len),
                cells_view(&qkv, rows * q_len, rows * kv_len),
                cells_view(&qkv, rows * (q_len + kv_len), rows * kv_len),
            )
        };
        Ok(Arena {
            x,
            x_rows,
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
            normed: f(rows * d.hidden)?,
            act_x: acts(d.hidden)?,
            qkv,
            q,
            k,
            v,
            q_out: gated.then(|| f(rows * attn_len)).transpose()?,
            gated_rows: forms.out_gate.then(|| f(rows * attn_len)).transpose()?,
            v_cols: (rows > 1).then(|| f(narrow * kv_len)).transpose()?,
            part_v: f(part_v)?,
            part_ms: f(partials_ms_len(narrow, d.n_head))?,
            attn: f(rows * attn_len)?,
            act_attn: acts(attn_len)?,
            ffn_inp: f(rows * d.hidden)?,
            act_ffn: acts(d.hidden)?,
            route,
            h: f(rows * d.slots() * d.ff)?,
            act_h: (1..=narrow)
                .map(|m| Q8Act::with_slots(stream, m * d.slots(), d.ff))
                .collect::<Result<Vec<_>, _>>()?,
            down: f(rows * d.slots() * d.hidden)?,
            gdn: d
                .lin
                .map(|shape| GdnArena::new(stream, shape, rows))
                .transpose()?,
            wide: (rows > GEMV_COLS)
                .then(|| Wide::new(stream, &d, rows, forms))
                .transpose()?,
            glu: forms
                .glu
                .then(|| {
                    Ok::<_, GpuError>(Glu {
                        g: f(narrow * d.slots() * d.ff)?,
                        u: f(narrow * d.slots() * d.ff)?,
                    })
                })
                .transpose()?,
            cols: (rows > 1 && forms.cols > 0)
                .then(|| f(narrow * forms.cols))
                .transpose()?,
            apart: forms
                .apart
                .map(|_| ApartRows::new(stream, &d, narrow))
                .transpose()?,
            media: None,
            dims: d,
            rows,
        })
    }

    /// Copy the media rows the unit about to walk holds over the embeddings
    /// in `x` ([`MediaRows::write`]); nothing when the arena holds no media
    /// call's rows.
    pub(super) fn write_media(&mut self, stream: &CudaStream) -> Result<(), GpuError> {
        let Arena { x, media, .. } = self;
        match media {
            Some(rows) => rows.write(stream, x),
            None => Ok(()),
        }
    }

    /// Run the delta launches of the walks over this arena by the stamped
    /// lanes ([`LaneRun`]): `word` the body's lane word, `each` whether the
    /// walk is a verify. Refused by name on an arena with no delta
    /// intermediates. Load-time only.
    pub(super) fn run_lanes(
        &mut self,
        word: WindowMut<'static, u32>,
        each: bool,
    ) -> Result<(), GpuError> {
        let g = self.gdn.as_mut().ok_or(GpuError::state(
            "qwen3moe::Arena::run_lanes",
            "an arena with delta intermediates",
        ))?;
        g.lanes = Some(LaneRun { word, each });
        Ok(())
    }

    /// `m − 1`, the index of the `m`-column activations a gemv arm of `m`
    /// rows reads, or a named refusal when the arena holds none of `m`
    /// columns (`m` past `min(rows, GEMV_COLS)`, or 0).
    pub(super) fn col(&self, m: usize) -> Result<usize, GpuError> {
        let n = self.act_x.len();
        if m == 0 || m > n {
            return Err(GpuError::shape(
                "qwen3moe::Arena::col",
                format!("a gemv arm of {m} rows; the arena's activations hold 1..={n} columns"),
            ));
        }
        Ok(m - 1)
    }

    /// Where the key and value blocks start in `qkv`.
    pub(super) fn qkv_offsets(&self) -> (usize, usize) {
        let (q_len, kv_len) = (self.dims.q_rows, self.dims.kv_len());
        (self.rows * q_len, self.rows * (q_len + kv_len))
    }

    /// Device bytes of the arena (the planes are counted by their owner; the
    /// windows once, through their parents).
    pub(super) fn bytes(&self) -> usize {
        let bufs = [
            &self.x,
            &self.normed,
            &self.qkv,
            &self.part_v,
            &self.part_ms,
            &self.attn,
            &self.ffn_inp,
            &self.h,
            &self.down,
        ];
        let acts = self
            .act_x
            .iter()
            .chain(&self.act_attn)
            .chain(&self.act_ffn)
            .chain(&self.act_h);
        bufs.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + self.v_cols.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.q_out.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.gated_rows.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.gdn.as_ref().map_or(0, GdnArena::bytes)
            + self.wide.as_ref().map_or(0, Wide::bytes)
            + self
                .glu
                .as_ref()
                .map_or(0, |g| g.g.num_bytes() + g.u.num_bytes())
            + self.cols.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.apart.as_ref().map_or(0, ApartRows::bytes)
            + self.route.bytes()
            + acts.map(Q8Act::device_bytes).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::super::body::TapRows;
    use super::{Inbox, LaneWord, PassRecord, cells_view};
    use cuda_core::DeviceBuffer;
    use std::sync::Arc;

    /// The views over the input words and the arena rows give back the
    /// context handles they take: an upload, a lane word with its views, a
    /// pass record, the layer taps and a row view leave the context's count
    /// where it was.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it on the box"]
    fn hw_the_views_give_back_their_context_handles() {
        let (ctx, stream) = crate::capsync::fresh_stream(0).expect("CUDA device 0 with a stream");
        let mut inbox = Inbox::new(&stream, 4).expect("an inbox");
        let rows = DeviceBuffer::<f32>::zeroed(&stream, 8).expect("the rows");
        let before = Arc::strong_count(&ctx);
        inbox.upload(&stream, 4).expect("an upload");
        {
            let _lane = LaneWord::with_views(&stream, 2, 3).expect("a lane word with three views");
            let _record = PassRecord::<3>::new(&stream).expect("a pass record");
            let mut taps = None;
            TapRows::set(&mut taps, &stream, 3, 4, true).expect("three layer taps");
            assert_eq!(taps.as_ref().map(|t| t.rows.len()), Some(3));
            TapRows::set(&mut taps, &stream, 3, 4, true).expect("the taps again");
            // SAFETY: values 4 .. 8 of `rows`, which outlives the view.
            let _row = unsafe { cells_view(&rows, 4, 4) };
        }
        stream.synchronize().expect("the upload");
        let after = Arc::strong_count(&ctx);
        eprintln!("view context handles: before {before} after {after}");
        assert_eq!(after, before, "the views leave no context handle");
    }
}
