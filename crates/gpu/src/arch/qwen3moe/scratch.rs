//! The qwen3moe chain's resident arenas: every intermediate of one layer for
//! the decode step's one row or the prefill's several, shared by all layers
//! (they run in turn), the per-step parameter image the captured graph
//! reads, and the per-layer K/V planes. Both arenas are allocated once at
//! load; nothing here is allocated per step or per prompt.

use super::experts::GroupTickets;
use super::router::{N_USED, RouterOut};
use crate::GpuError;
use crate::flash_gqa::{HEAD, partials_ms_len, partials_v_len};
use crate::tensor::{Q8Act, window};
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;

/// Element offsets into [`StepParams::buf`]: the three u32 first, then
/// the rope table (one position, [`HEAD`] f32 as bits).
pub(super) const SP_TOKEN: usize = 0;
pub(super) const SP_POS: usize = 1;
pub(super) const SP_N_KEYS: usize = 2;
pub(super) const SP_CS: usize = 3;

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
    /// The one-token gate·up's ticket counts, one per 128-value group of a
    /// token's `h`.
    pub(super) gate_up_tickets: GroupTickets,
    /// The down outputs, per token slot-major `N_USED · hidden`.
    pub(super) down: DeviceBuffer<f32>,
}

/// The decode step's parameters in one allocation, laid out by the `SP_*`
/// offsets, so a step's refresh is one host-to-device copy; the captured
/// graph reads the windows.
pub(super) struct StepParams {
    pub(super) buf: DeviceBuffer<u32>,
    pub(super) host: Vec<u32>,
    /// Non-owning windows into `buf`, one per kernel argument.
    pub(super) token: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) pos: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) n_keys: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) cs: ManuallyDrop<DeviceBuffer<f32>>,
}

/// The parameter windows one pass reads, one entry per token: the ids the
/// embedding gathers (its row count is the window's length), the cache rows
/// they land in, their live key counts and their rope rows.
pub(super) struct Io<'a> {
    pub(super) tokens: &'a DeviceBuffer<u32>,
    pub(super) pos: &'a DeviceBuffer<u32>,
    pub(super) n_keys: &'a DeviceBuffer<u32>,
    pub(super) cs: &'a DeviceBuffer<f32>,
}

impl StepParams {
    /// Position 0's image: token 0, pos 0, one live key, a zeroed table.
    /// Load-time only.
    pub(super) fn new(stream: &CudaStream) -> Result<StepParams, GpuError> {
        let mut host = vec![0u32; SP_CS + HEAD];
        host[SP_N_KEYS] = 1;
        let buf = DeviceBuffer::from_host(stream, &host)?;
        // SAFETY: each window is inside `buf` by the `SP_*` layout (SP_CS +
        // HEAD elements), every offset is a u32 multiple, and `buf` moves
        // into the struct beside them (a move of the handle, not of the
        // allocation), where it outlives them.
        let (token, pos, n_keys, cs) = unsafe {
            (
                param_view::<u32>(&buf, SP_TOKEN, 1),
                param_view::<u32>(&buf, SP_POS, 1),
                param_view::<u32>(&buf, SP_N_KEYS, 1),
                param_view::<f32>(&buf, SP_CS, HEAD),
            )
        };
        Ok(StepParams {
            buf,
            host,
            token,
            pos,
            n_keys,
            cs,
        })
    }

    /// The windows, as a pass reads them.
    pub(super) fn io(&self) -> Io<'_> {
        Io {
            tokens: &self.token,
            pos: &self.pos,
            n_keys: &self.n_keys,
            cs: &self.cs,
        }
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
            gate_up_tickets: GroupTickets::new(stream, (N_USED * d.ff).div_ceil(128))?,
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
            + self.v_cols.as_ref().map_or(0, DeviceBuffer::num_bytes)
            + self.route.bytes()
            + self.gate_up_tickets.bytes()
            + acts.map(act_bytes).sum::<usize>()
    }
}

/// Device bytes of one quantized activation's planes.
fn act_bytes(a: &Q8Act) -> usize {
    a.q3.num_bytes() + a.q4.num_bytes() + a.q6.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes()
}
