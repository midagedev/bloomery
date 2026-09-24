//! The qwen3moe chain's resident arena: every intermediate of one layer at
//! m = 1, shared by all layers (they run in turn), the per-step parameter
//! image the captured graph reads, and the per-layer K/V planes. Allocated
//! once at load; nothing here is allocated per step.

use super::router::{N_USED, RouterOut};
use crate::GpuError;
use crate::flash_gqa::{HEAD, partials_ms_len, partials_v_len};
use crate::tensor::{Q8Act, window};
use cuda_core::{CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;

/// Element offsets into [`Arena::step_params`]: the three u32 first, then
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

/// The arena, in chain order.
pub(super) struct Arena {
    pub(super) dims: Dims,
    /// The layer's input residual (the embedding row for layer 0).
    pub(super) x: DeviceBuffer<f32>,
    pub(super) normed: DeviceBuffer<f32>,
    /// q8_1 of the attention-normed row: q, k and v all read it.
    pub(super) act_x: Q8Act,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) part_v: DeviceBuffer<f32>,
    pub(super) part_ms: DeviceBuffer<f32>,
    /// The attention rows, `n_head · HEAD`.
    pub(super) attn: DeviceBuffer<f32>,
    pub(super) act_attn: Q8Act,
    pub(super) attn_out: DeviceBuffer<f32>,
    /// The FFN's input residual.
    pub(super) ffn_inp: DeviceBuffer<f32>,
    /// q8_1 of the FFN-normed row: the router reads its f32 twin `normed`,
    /// the experts' gate·up this.
    pub(super) act_ffn: Q8Act,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) route: RouterOut,
    /// The selected experts' SwiGLU rows, slot-major `N_USED · ff`.
    pub(super) h: DeviceBuffer<f32>,
    /// q8_1 of `h`, one column per slot.
    pub(super) act_h: Q8Act,
    /// The down outputs, slot-major `N_USED · hidden`.
    pub(super) down: DeviceBuffer<f32>,
    pub(super) l_out: DeviceBuffer<f32>,
    /// Every per-step parameter in one allocation, laid out by the `SP_*`
    /// offsets, so a step's refresh is one host-to-device copy.
    pub(super) step_params: DeviceBuffer<u32>,
    pub(super) params_host: Vec<u32>,
    /// Non-owning windows into `step_params`, one per kernel argument.
    pub(super) token_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) pos_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) n_keys_buf: ManuallyDrop<DeviceBuffer<u32>>,
    pub(super) cs_buf: ManuallyDrop<DeviceBuffer<f32>>,
}

/// A non-owning window of `len` `T` over `parent`, from element `off` of the
/// parent's u32 grid.
///
/// # Safety
///
/// `off + len · size_of::<T>() / 4` must be within `parent`, `T` four-byte
/// aligned, and `parent` must outlive the window unmoved in memory (a
/// captured graph bakes the address in).
unsafe fn param_view<T>(
    parent: &DeviceBuffer<u32>,
    off: usize,
    len: usize,
) -> ManuallyDrop<DeviceBuffer<T>> {
    let ptr = parent.cu_deviceptr() + (off * size_of::<u32>()) as u64;
    // SAFETY: the range is the caller's contract above, inside `parent`.
    unsafe { window(ptr, len, parent.context()) }
}

impl Arena {
    /// Allocate the arena for `d`. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims) -> Result<Arena, GpuError> {
        let q_len = d.n_head * HEAD;
        let kv_len = d.n_kv * HEAD;
        // Position 0's image: token 0, pos 0, one live key, a zeroed table.
        let mut params_host = vec![0u32; SP_CS + HEAD];
        params_host[SP_N_KEYS] = 1;
        let step_params = DeviceBuffer::from_host(stream, &params_host)?;
        // SAFETY: each window is inside `step_params` by the `SP_*` layout
        // (SP_CS + HEAD elements), every offset is a u32 multiple, and
        // `step_params` moves into the arena beside them (a move of the
        // handle, not of the allocation), where it outlives them.
        let (token_buf, pos_buf, n_keys_buf, cs_buf) = unsafe {
            (
                param_view::<u32>(&step_params, SP_TOKEN, 1),
                param_view::<u32>(&step_params, SP_POS, 1),
                param_view::<u32>(&step_params, SP_N_KEYS, 1),
                param_view::<f32>(&step_params, SP_CS, HEAD),
            )
        };
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(Arena {
            x: f(d.hidden)?,
            normed: f(d.hidden)?,
            act_x: Q8Act::with_k(stream, 1, d.hidden)?,
            q: f(q_len)?,
            k: f(kv_len)?,
            v: f(kv_len)?,
            part_v: f(partials_v_len(d.n_head, d.ctx))?,
            part_ms: f(partials_ms_len(d.n_head, d.ctx))?,
            attn: f(q_len)?,
            act_attn: Q8Act::with_k(stream, 1, q_len)?,
            attn_out: f(d.hidden)?,
            ffn_inp: f(d.hidden)?,
            act_ffn: Q8Act::with_k(stream, 1, d.hidden)?,
            logits: f(super::router::N_EXPERT)?,
            route: RouterOut::new(stream)?,
            h: f(N_USED * d.ff)?,
            act_h: Q8Act::with_k(stream, N_USED, d.ff)?,
            down: f(N_USED * d.hidden)?,
            l_out: f(d.hidden)?,
            step_params,
            params_host,
            token_buf,
            pos_buf,
            n_keys_buf,
            cs_buf,
            dims: d,
        })
    }

    /// Device bytes of the arena (the planes are counted by their owner; the
    /// parameter windows once, through their parent).
    pub(super) fn bytes(&self) -> usize {
        let bufs = [
            &self.x,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.part_v,
            &self.part_ms,
            &self.attn,
            &self.attn_out,
            &self.ffn_inp,
            &self.logits,
            &self.route.probs,
            &self.route.weights,
            &self.h,
            &self.down,
            &self.l_out,
        ];
        bufs.iter().map(|b| b.num_bytes()).sum::<usize>()
            + self.route.ids.num_bytes()
            + self.step_params.num_bytes()
            + [&self.act_x, &self.act_attn, &self.act_ffn, &self.act_h]
                .iter()
                .map(|a| act_bytes(a))
                .sum::<usize>()
    }
}

/// Device bytes of one quantized activation's planes.
fn act_bytes(a: &Q8Act) -> usize {
    a.q3.num_bytes() + a.q4.num_bytes() + a.q6.num_bytes() + a.s8.num_bytes() + a.d8.num_bytes()
}
