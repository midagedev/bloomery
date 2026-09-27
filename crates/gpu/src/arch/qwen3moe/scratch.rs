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

use super::router::{N_USED, RouterOut};
use crate::GpuError;
use crate::flash_gqa::{HEAD, partials_ms_len, partials_v_len};
use crate::rope_table::{Direction, RopeTable};
use crate::tensor::{Q8Act, window};
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer, PinnedHostBuffer};
use std::mem::ManuallyDrop;
use std::time::{Duration, Instant};

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

/// Every cache position's rope row on the card, the table every path's rope
/// launch reads by position: row `p`, [`HEAD`] f32 at `p · HEAD`, holds
/// `RopeTable::push`'s bits for position `p`, for each `p` below the cache's
/// rows — so the one check a position gets, below the cache's rows, keeps
/// the read inside the table. Built once at load.
pub(super) struct RopeRows {
    pub(super) table: DeviceBuffer<f32>,
    /// The host time the rows took at load.
    pub(super) build: Duration,
}

impl RopeRows {
    /// Rows `0..ctx` of `rope` (a [`HEAD`]-wide spec), one `push` per
    /// position in order, copied to the card. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        rope: &RopeTable,
        ctx: usize,
    ) -> Result<RopeRows, GpuError> {
        const WHAT: &str = "qwen3moe::RopeRows::new";
        let positions = u32::try_from(ctx).map_err(|_| {
            GpuError::shape(
                WHAT,
                format!("a cache of {ctx} rows: positions and live key counts are u32"),
            )
        })?;
        if rope.n_dims() != HEAD {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a rope table of {} values a position; a row is {HEAD}",
                    rope.n_dims()
                ),
            ));
        }
        let t0 = Instant::now();
        let mut host = Vec::with_capacity(ctx * HEAD);
        for pos in 0..positions {
            rope.push(pos, Direction::Forward, &mut host);
        }
        let build = t0.elapsed();
        Ok(RopeRows {
            table: DeviceBuffer::from_host(stream, &host)?,
            build,
        })
    }

    /// The positions the table holds: the cache's rows.
    pub(super) fn rows(&self) -> usize {
        self.table.len() / HEAD
    }
}

/// The shapes the arena is cut for, read from the file at load.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dims {
    pub(super) hidden: usize,
    pub(super) n_head: usize,
    pub(super) n_kv: usize,
    pub(super) ff: usize,
    pub(super) ctx: usize,
}

/// One layer's K and V planes, `[n_kv][ctx][HEAD]` f16 each.
pub(super) struct KvPlanes {
    pub(super) k: DeviceBuffer<u16>,
    pub(super) v: DeviceBuffer<u16>,
}

impl KvPlanes {
    pub(super) fn new(stream: &CudaStream, d: &Dims) -> Result<KvPlanes, GpuError> {
        let n = d.n_kv * d.ctx * HEAD;
        Ok(KvPlanes {
            k: DeviceBuffer::zeroed(stream, n)?,
            v: DeviceBuffer::zeroed(stream, n)?,
        })
    }

    pub(super) fn bytes(&self) -> usize {
        self.k.num_bytes() + self.v.num_bytes()
    }
}

/// The arena, in chain order: every intermediate of one layer for up to
/// `rows` tokens, token-major, shared by all layers (they run in turn). The
/// decode step's arena has one row, the prompt prefill's
/// [`super::router::MAX_TOKENS`]; a pass of `m <= rows` tokens uses the
/// first `m` rows and the `m`-column activations.
pub(super) struct Arena {
    pub(super) dims: Dims,
    pub(super) rows: usize,
    /// The layer's input residual (the embedding rows for layer 0), and its
    /// output: the combine writes the next layer's input here.
    pub(super) x: DeviceBuffer<f32>,
    /// Row `t` of `x`.
    pub(super) x_rows: Vec<ManuallyDrop<DeviceBuffer<f32>>>,
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
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) q: ManuallyDrop<DeviceBuffer<f32>>,
    pub(super) k: ManuallyDrop<DeviceBuffer<f32>>,
    pub(super) v: ManuallyDrop<DeviceBuffer<f32>>,
    /// A Q6_K value projection's output at more than one token: `q6k_gemv`
    /// writes it row-major, and a copy puts it into `v` token-major. `None`
    /// on a one-row arena, where the gemv writes `v` itself.
    pub(super) v_cols: Option<DeviceBuffer<f32>>,
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The attention rows, `n_head · HEAD` per token.
    pub(super) attn: DeviceBuffer<f32>,
    pub(super) act_attn: Vec<Q8Act>,
    /// The FFN's input residual: `x` plus the attention output.
    pub(super) ffn_inp: DeviceBuffer<f32>,
    /// q8_1 of the FFN-normed rows, the experts' gate·up input. At more
    /// than one row the router reads their f32 twin `normed`; at one row the
    /// router's launch writes these and keeps the normed row to itself.
    pub(super) act_ffn: Vec<Q8Act>,
    /// The router's ids, token-major `N_USED` per token, are the down's
    /// selector: slot `t · N_USED + j` of a pass is token `t`'s slot `j`.
    pub(super) route: RouterOut,
    /// The selected experts' SwiGLU rows, per token slot-major `N_USED · ff`.
    pub(super) h: DeviceBuffer<f32>,
    /// q8_1 of `h`, one column per slot: `act_h[m − 1]` holds the `m ·
    /// N_USED` columns of `m` tokens, the down's input.
    pub(super) act_h: Vec<Q8Act>,
    /// The down outputs, per token slot-major `N_USED · hidden`.
    pub(super) down: DeviceBuffer<f32>,
}

/// The decode step's input record — its position and its token — in one
/// [`Inbox`], so a step's refresh is one fill and one copy, and the windows
/// the captured graph reads.
pub(super) struct StepParams {
    /// Non-owning windows into the inbox's device words: the token (the
    /// embedding's one id) and the position word.
    token: ManuallyDrop<DeviceBuffer<u32>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    inbox: Inbox,
}

/// What a unit's first launch reads: its ids (the unit's row count is their
/// length), the word holding the first position of the input record they
/// are a window of, and where the window starts in that record — row `t` of
/// the unit is position `pos0 + first + t`.
pub(super) struct Io<'a> {
    pub(super) ids: &'a DeviceBuffer<u32>,
    pub(super) pos0: &'a DeviceBuffer<u32>,
    pub(super) first: usize,
}

impl StepParams {
    /// Words of the record: the position, then the token.
    const WORDS: usize = IN_IDS + 1;

    /// A zeroed record (token 0 at position 0). Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<StepParams, GpuError> {
        let inbox = Inbox::new(stream, Self::WORDS)?;
        // SAFETY: each window is one word of the inbox's `WORDS` device words
        // (`IN_POS0`, `IN_IDS` < `WORDS`), and the inbox moves into the struct
        // beside them (a move of the handle, not of the allocation), where it
        // outlives them.
        let (token, pos0) = unsafe {
            (
                param_view::<u32>(inbox.dev(), IN_IDS, 1),
                param_view::<u32>(inbox.dev(), IN_POS0, 1),
            )
        };
        Ok(StepParams { token, pos0, inbox })
    }

    /// Write `token` at position `pos` and enqueue its copy — the step's
    /// refresh. Asynchronous: the step's launches behind it read it.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        token: u32,
        pos: u32,
    ) -> Result<(), GpuError> {
        put_input(self.inbox.host_mut()?, &[token], pos)?;
        self.inbox.upload(stream, Self::WORDS)
    }

    /// The step's input, as its first launch reads it.
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            ids: &self.token,
            pos0: &self.pos0,
            first: 0,
        }
    }

    /// Device bytes of the record.
    pub(super) fn bytes(&self) -> usize {
        self.inbox.bytes()
    }
}

/// A non-owning window of `len` `T` over `parent`, from element `off` of the
/// parent's u32 grid.
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
) -> ManuallyDrop<DeviceBuffer<T>> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<u32>()) as u64;
    // SAFETY: the range is the caller's contract above, inside `parent`.
    unsafe { window(ptr, len, parent.context()) }
}

/// A non-owning window of `len` f32 over `parent` from element `off`.
///
/// # Safety
///
/// `off + len <= parent.len()`, and `parent` must outlive the window unmoved
/// in memory (a captured graph bakes the address in).
pub(super) unsafe fn f32_view(
    parent: &DeviceBuffer<f32>,
    off: usize,
    len: usize,
) -> ManuallyDrop<DeviceBuffer<f32>> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<f32>()) as u64;
    // SAFETY: the range is the caller's contract above, inside `parent`.
    unsafe { window(ptr, len, parent.context()) }
}

impl Arena {
    /// Allocate the arena for `d` and up to `rows` tokens. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims, rows: usize) -> Result<Arena, GpuError> {
        let q_len = d.n_head * HEAD;
        let kv_len = d.n_kv * HEAD;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        let acts = |k: usize| {
            (1..=rows)
                .map(|m| Q8Act::with_k(stream, m, k))
                .collect::<Result<Vec<_>, _>>()
        };
        let x = f(rows * d.hidden)?;
        let qkv = f(rows * (q_len + 2 * kv_len))?;
        let route = RouterOut::with_tokens(stream, rows)?;
        // SAFETY: every window below lies inside its parent — row `t < rows`
        // of `x` (`hidden` values), and the three blocks that tile `qkv`
        // (`rows · (q_len + 2·kv_len)`) — and each parent moves into the
        // arena beside its windows (a move of the handle, not of the
        // allocation), where it outlives them.
        let (x_rows, q, k, v) = unsafe {
            (
                (0..rows)
                    .map(|t| f32_view(&x, t * d.hidden, d.hidden))
                    .collect(),
                f32_view(&qkv, 0, rows * q_len),
                f32_view(&qkv, rows * q_len, rows * kv_len),
                f32_view(&qkv, rows * (q_len + kv_len), rows * kv_len),
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
            v_cols: (rows > 1).then(|| f(rows * kv_len)).transpose()?,
            part_v: f(partials_v_len(rows, d.n_head, d.ctx))?,
            part_ms: f(partials_ms_len(rows, d.n_head, d.ctx))?,
            attn: f(rows * q_len)?,
            act_attn: acts(q_len)?,
            ffn_inp: f(rows * d.hidden)?,
            act_ffn: acts(d.hidden)?,
            route,
            h: f(rows * N_USED * d.ff)?,
            act_h: (1..=rows)
                .map(|m| Q8Act::with_slots(stream, m * N_USED, d.ff))
                .collect::<Result<Vec<_>, _>>()?,
            down: f(rows * N_USED * d.hidden)?,
            dims: d,
            rows,
        })
    }

    /// Where the key and value blocks start in `qkv`.
    pub(super) fn qkv_offsets(&self) -> (usize, usize) {
        let (q_len, kv_len) = (self.dims.n_head * HEAD, self.dims.n_kv * HEAD);
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
            + self.route.bytes()
            + acts.map(act_bytes).sum::<usize>()
    }
}

/// Device bytes of one quantized activation's planes.
fn act_bytes(a: &Q8Act) -> usize {
    a.q3.num_bytes() + a.q4.num_bytes() + a.q6.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes()
}
