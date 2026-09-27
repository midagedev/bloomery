//! Qwen3.8's own element kernels, the ones no other body launches: the Q8_0
//! embedding row, the attention output under its sigmoid gate in f32, the
//! selector's raw key append, and the shared expert's gated sum.
//!
//! - [`q38_kernels::embed_rows_q8_0`] — one thread per value: row `ids[t]`
//!   of the Q8_0 table in the q8f32 planes (`crate::weights::q8_0_planes`),
//!   each value `f32(code) · f32(d)` as ggml's `dequantize_row_q8_0` makes it
//!   (one product of an 8-bit and an 11-bit significand: exact), and row
//!   `t`'s position `pos0[0] + first + t` and live key count one more.
//! - [`q38_kernels::gqa_out_gate_f32`] — one thread per value: `attn ·
//!   σ(gate)`, the gate at `[q | gate]`'s head stride, `σ` the Qwen3.6 gated
//!   quantizer's (`route_core::sigmoid`); the output projection reads the f32
//!   product (its weights are Q8_0 with f32 activations).
//! - [`q38_kernels::qsa_key_append`] — one thread per value: the indexer's
//!   raw key, `DIM` values a row of the `f32_gemv` output (row-major `[DIM]
//!   [m]`), rounded once to f16 into the raw plane at the row's position.
//! - [`q38_kernels::q38_shared_add`] — one thread per value: `hsum + sh · w`,
//!   the shared expert's output times its gate weight (the router's last slot)
//!   rounded, then added to the host tier's routed sum and rounded — ggml's
//!   MUL then ADD, each an explicit round-to-nearest intrinsic so the device
//!   build does not contract them into one fused multiply-add.
//!
//! No silent failure: an embedding id past the table raises
//! [`FaultSite::TokenId`] and writes a NaN row; a position at or past the
//! raw plane raises [`FaultSite::CachePos`] and writes nothing; a raw key that
//! is not finite after its f16 rounding raises [`FaultSite::PoolSelect`]; an
//! input or a result of the two f32 products that is not finite raises
//! [`FaultSite::F32Product`]. Every value is written as computed.

use crate::fault::{FaultSink, FaultSite};
use crate::flash::{f32_to_f16_bits, half_bits_to_f32};
use crate::route_core::sigmoid;
use crate::tensor::DeviceTensor;
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values of an attention head: the gate kernel's head, whose `[q | gate]`
/// rows are twice as wide.
pub const HEAD: usize = 256;
/// Values of an indexer key.
pub const DIM: usize = crate::qsa::DIM;
/// Threads per block of every kernel here.
const THREADS: u32 = 256;
const _: () = assert!(HEAD == 256 && DIM == 128);

/// `v` as a position or live key count word: `v` while it fits a u32, else
/// `u32::MAX` — a value past every cache, which the rope and the flash
/// refuse, where a wrapped one would name a plausible row.
#[inline(always)]
fn position_word(v: usize) -> u32 {
    if v > u32::MAX as usize {
        u32::MAX
    } else {
        v as u32
    }
}

#[cuda_module]
mod q38_kernels {
    use super::*;

    /// Dequantize `ids.len()` rows of the Q8_0 table (`qs` `8·k32` words and
    /// `d` `k32` f16 scales a row, `32·k32` values) into `y`, token-major,
    /// one thread per value, and give each row its position and live key
    /// count (module doc). An id past the table's `n_rows` rows raises
    /// [`FaultSite::TokenId`], reads no table word and writes a NaN row; the
    /// positions do not depend on the ids.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= 8 * k32 * n_rows,
            d.len() >= k32 * n_rows,
            y.len() >= 32 * k32 * ids.len(),
            pos0.len() >= 1,
            pos.len() >= ids.len(),
            n_keys.len() >= ids.len()
        )
    )]
    pub fn embed_rows_q8_0(
        qs: &[u32],
        d: &[u16],
        ids: &[u32],
        pos0: &[u32],
        first: u32,
        n_rows: u32,
        k32: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut pos: DisjointSlice<u32>,
        mut n_keys: DisjointSlice<u32>,
    ) {
        let i = thread::index_1d().get();
        if i < ids.len() {
            // SAFETY: pos0 holds a word by the launch contract.
            let p = unsafe { *pos0.get_unchecked(0) } as usize + first as usize + i;
            // SAFETY: i < ids.len() <= pos.len(), n_keys.len() by the launch
            // contract; thread i owns entry i of both.
            unsafe {
                *pos.get_unchecked_mut(i) = position_word(p);
                *n_keys.get_unchecked_mut(i) = position_word(p + 1);
            }
        }
        let width = 32 * k32 as usize;
        if i >= ids.len() * width {
            return;
        }
        let t = i / width;
        let v = i % width;
        // SAFETY: t < ids.len() by the guard.
        let id = unsafe { *ids.get_unchecked(t) };
        let val = if id < n_rows {
            let row = id as usize;
            // SAFETY: id < n_rows, so word row·8·k32 + v/4 and scale
            // row·k32 + v/32 lie inside qs and d by the launch contract.
            let (word, bits) = unsafe {
                (
                    *qs.get_unchecked(row * 8 * k32 as usize + v / 4),
                    *d.get_unchecked(row * k32 as usize + v / 32),
                )
            };
            let code = (word >> (8 * (v % 4))) as u8 as i8;
            code as f32 * half_bits_to_f32(bits)
        } else {
            if v == 0 {
                fault.raise(FaultSite::TokenId);
            }
            f32::NAN
        };
        // SAFETY: i < ids.len()·width <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = val;
        }
    }

    /// `y = attn · σ(gate)` over `m` tokens of `n_head` heads of [`HEAD`]
    /// values: value `v` of head `h` of token `t` at `(t·n_head + h)·HEAD +
    /// v`, its gate at `(t·n_head + h)·2·HEAD + HEAD + v` of the query
    /// projection's `[q | gate]` rows. A non-finite input or product raises
    /// [`FaultSite::F32Product`].
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            attn.len() >= 256 * n_head * m,
            qg.len() >= 512 * n_head * m,
            y.len() >= 256 * n_head * m
        )
    )]
    pub fn gqa_out_gate_f32(
        attn: &[f32],
        qg: &[f32],
        n_head: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= HEAD * n_head as usize * m as usize {
            return;
        }
        let (hv, v) = (i / HEAD, i % HEAD);
        // SAFETY: i < HEAD·n_head·m <= attn.len(), and hv < n_head·m puts
        // hv·2·HEAD + HEAD + v below 2·HEAD·n_head·m <= qg.len(), by the
        // launch contract.
        let (a, g) = unsafe {
            (
                *attn.get_unchecked(i),
                *qg.get_unchecked(hv * 2 * HEAD + HEAD + v),
            )
        };
        let out = a * sigmoid(g);
        if !(a.is_finite() & g.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < HEAD·n_head·m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }

    /// Row `c` of `m` indexer keys — value `r` at `kr[r·m + c]`, the f32
    /// gemv's row-major output — rounded once to f16 into the raw plane's
    /// row `pos[c]`, [`DIM`] values a row. A position at or past `ctx`
    /// raises [`FaultSite::CachePos`] and writes nothing; a value that is not
    /// finite after its rounding raises [`FaultSite::PoolSelect`] and is
    /// written.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (kr.len() >= 128 * m, pos.len() >= m, raw.len() >= 128 * ctx)
    )]
    pub fn qsa_key_append(
        kr: &[f32],
        pos: &[u32],
        m: u32,
        ctx: u32,
        fault: FaultSink,
        mut raw: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        if i >= DIM * m as usize {
            return;
        }
        let (c, r) = (i / DIM, i % DIM);
        // SAFETY: c < m, so c < pos.len() and r·m + c < DIM·m <= kr.len(), by
        // the launch contract.
        let (p, x) = unsafe { (*pos.get_unchecked(c), *kr.get_unchecked(r * m as usize + c)) };
        if p >= ctx {
            if r == 0 {
                fault.raise(FaultSite::CachePos);
            }
            return;
        }
        let bits = f32_to_f16_bits(x);
        if !half_bits_to_f32(bits).is_finite() {
            fault.raise(FaultSite::PoolSelect);
        }
        // SAFETY: p < ctx, so p·DIM + r < DIM·ctx <= raw.len() by the launch
        // contract; positions of one launch are distinct (`pos[0] + c`), so
        // thread (c, r) is the word's only writer.
        unsafe {
            *raw.get_unchecked_mut(p as usize * DIM + r) = bits;
        }
    }

    /// `y = hsum + sh · w[t·slots + slot]` over `m` tokens of `n` values,
    /// token `t`'s weight the router's slot `slot`: the product rounded, then
    /// the sum rounded (module doc). A non-finite input or result raises
    /// [`FaultSite::F32Product`].
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            slots >= slot + 1,
            hsum.len() >= n * m,
            sh.len() >= n * m,
            w.len() >= slots * m,
            y.len() >= n * m
        )
    )]
    pub fn q38_shared_add(
        hsum: &[f32],
        sh: &[f32],
        w: &[f32],
        slot: u32,
        slots: u32,
        n: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i >= n as usize * m as usize {
            return;
        }
        let t = i / n as usize;
        // SAFETY: i < n·m bounds hsum and sh; t < m and slot < slots put
        // t·slots + slot below slots·m <= w.len(); by the launch contract.
        let (h, s, wt) = unsafe {
            (
                *hsum.get_unchecked(i),
                *sh.get_unchecked(i),
                *w.get_unchecked(t * slots as usize + slot as usize),
            )
        };
        let out = add_rn_f32(h, mul_rn_f32(s, wt));
        if !(h.is_finite() & s.is_finite() & wt.is_finite() & out.is_finite()) {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: i < n·m <= y.len() by the launch contract.
        unsafe {
            *y.get_unchecked_mut(i) = out;
        }
    }
}

/// [`Q38Kernels::enqueue_embed_rows`]'s arguments: the Q8_0 table's planes,
/// one unit's ids, the word holding the first position of the input the ids
/// are a window of and the window's offset in it, and the three outputs —
/// the rows, each row's position and its live key count.
pub struct EmbedQ8Args<'a> {
    pub qs: &'a DeviceTensor<u32>,
    pub d: &'a DeviceTensor<u16>,
    pub ids: &'a DeviceBuffer<u32>,
    pub pos0: &'a DeviceBuffer<u32>,
    pub first: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub pos: &'a mut DeviceBuffer<u32>,
    pub n_keys: &'a mut DeviceBuffer<u32>,
}

/// [`Q38Kernels::enqueue_shared_add`]'s arguments: the host tier's routed
/// sums and the shared expert's outputs (`[m][n]` each), the router's
/// weights (`slots` a token) and the shared expert's slot among them, and
/// the output.
pub struct SharedAddArgs<'a> {
    pub hsum: &'a DeviceBuffer<f32>,
    pub sh: &'a DeviceBuffer<f32>,
    pub w: &'a DeviceBuffer<f32>,
    pub slot: usize,
    pub slots: usize,
    pub n: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_out_gate`]'s arguments: the attention output and
/// the `[q | gate]` rows it is gated by, `m` tokens of `n_head` heads, and
/// the output.
pub struct OutGateArgs<'a> {
    pub attn: &'a DeviceBuffer<f32>,
    pub qg: &'a DeviceBuffer<f32>,
    pub n_head: usize,
    pub m: usize,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
}

/// [`Q38Kernels::enqueue_key_append`]'s arguments: the f32 gemv's `[DIM][m]`
/// raw keys and the `m` rows' positions, the plane's `ctx` rows, and the
/// layer's raw plane.
pub struct KeyAppendArgs<'a> {
    pub kr: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub m: usize,
    pub ctx: usize,
    pub fault: FaultSink,
    pub raw: &'a mut DeviceBuffer<u16>,
}

/// A buffer shorter than a launch reads or writes, named.
fn short(what: &'static str, lens: &[(&str, usize, usize)]) -> Result<(), GpuError> {
    match lens.iter().find(|(_, got, need)| got < need) {
        Some((name, got, need)) => Err(GpuError::shape(
            what,
            format!("{name}.len() {got} < {need}"),
        )),
        None => Ok(()),
    }
}

/// A launch of `n` threads in blocks of [`THREADS`].
fn grid(what: &'static str, n: usize) -> Result<LaunchConfig1D, GpuError> {
    let blocks = launch_u32(what, "grid", n.div_ceil(THREADS as usize))?;
    Ok(LaunchConfig1D::new(blocks, THREADS, 0))
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct Q38Kernels {
    module: q38_kernels::LoadedModule,
}

impl Q38Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Q38Kernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launchers check its launch contracts.
        let module = unsafe { q38_kernels::load(ctx)? };
        Ok(Q38Kernels { module })
    }

    /// Enqueue the Q8_0 embedding lookup of one unit of rows ([`EmbedQ8Args`],
    /// module doc): `k = 32 · d.cols()` values a row, `qs.cols()` its
    /// `8 · d.cols()` words. The rows are `gguf::quant::dequant_row`'s bits.
    /// One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_embed_rows(
        &self,
        stream: &CudaStream,
        a: EmbedQ8Args<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_embed_rows";
        let (qs, d) = (a.qs, a.d);
        let k32 = d.cols();
        if qs.rows() == 0 || qs.rows() != d.rows() || k32 == 0 || qs.cols() != 8 * k32 {
            return Err(GpuError::shape(
                what,
                format!(
                    "Q8_0 planes qs {}x{} and d {}x{}: equal rows, 8 words a scale",
                    qs.rows(),
                    qs.cols(),
                    d.rows(),
                    d.cols()
                ),
            ));
        }
        let n = a.ids.len();
        if n == 0 {
            return Err(GpuError::shape(what, "empty ids"));
        }
        short(
            what,
            &[
                ("y", a.y.len(), 32 * k32 * n),
                ("pos0", a.pos0.len(), 1),
                ("pos", a.pos.len(), n),
                ("n_keys", a.n_keys.len(), n),
            ],
        )?;
        let cfg = grid(what, 32 * k32 * n)?;
        let first = launch_u32(what, "first", a.first)?;
        let n_rows = launch_u32(what, "rows", qs.rows())?;
        let k32 = launch_u32(what, "k32", k32)?;
        let prep = self.module.prepare_embed_rows_q8_0(cfg)?;
        self.module.embed_rows_q8_0(
            stream,
            &prep,
            qs.buf(),
            d.buf(),
            a.ids,
            a.pos0,
            first,
            n_rows,
            k32,
            a.fault,
            a.y,
            a.pos,
            a.n_keys,
        )?;
        Ok(())
    }

    /// Enqueue `y = attn · σ(gate)` over `m` tokens of `n_head` heads of
    /// [`HEAD`] ([`OutGateArgs`], module doc): `attn` and `y`
    /// `m·n_head·HEAD`, `qg` the `[q | gate]` rows, `m·n_head·2·HEAD`. One
    /// launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_out_gate(
        &self,
        stream: &CudaStream,
        a: OutGateArgs<'_>,
    ) -> Result<(), GpuError> {
        let OutGateArgs {
            attn,
            qg,
            n_head,
            m,
            fault,
            y,
        } = a;
        let what = "q38::enqueue_out_gate";
        let n = HEAD * n_head * m;
        if n == 0 {
            return Err(GpuError::shape(what, "no head or no token"));
        }
        short(
            what,
            &[
                ("attn", attn.len(), n),
                ("qg", qg.len(), 2 * n),
                ("y", y.len(), n),
            ],
        )?;
        let cfg = grid(what, n)?;
        let prep = self.module.prepare_gqa_out_gate_f32(cfg)?;
        self.module.gqa_out_gate_f32(
            stream,
            &prep,
            attn,
            qg,
            launch_u32(what, "n_head", n_head)?,
            launch_u32(what, "m", m)?,
            fault,
            y,
        )?;
        Ok(())
    }

    /// Enqueue the raw key append of `m` rows ([`KeyAppendArgs`], module
    /// doc): `kr` the f32 gemv's `[DIM][m]` output, `pos` the rows'
    /// positions, `raw` the layer's `[ctx][DIM]` f16 plane. One launch.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_key_append(
        &self,
        stream: &CudaStream,
        a: KeyAppendArgs<'_>,
    ) -> Result<(), GpuError> {
        let KeyAppendArgs {
            kr,
            pos,
            m,
            ctx,
            fault,
            raw,
        } = a;
        let what = "q38::enqueue_key_append";
        if m == 0 || ctx == 0 {
            return Err(GpuError::shape(
                what,
                format!("need m and ctx >= 1, got m={m} ctx={ctx}"),
            ));
        }
        short(
            what,
            &[
                ("kr", kr.len(), DIM * m),
                ("pos", pos.len(), m),
                ("raw", raw.len(), DIM * ctx),
            ],
        )?;
        let cfg = grid(what, DIM * m)?;
        let prep = self.module.prepare_qsa_key_append(cfg)?;
        self.module.qsa_key_append(
            stream,
            &prep,
            kr,
            pos,
            launch_u32(what, "m", m)?,
            launch_u32(what, "ctx", ctx)?,
            fault,
            raw,
        )?;
        Ok(())
    }

    /// Enqueue the shared expert's gated sum ([`SharedAddArgs`], module
    /// doc). One launch. Asynchronous, allocation-free, capturable.
    pub fn enqueue_shared_add(
        &self,
        stream: &CudaStream,
        a: SharedAddArgs<'_>,
    ) -> Result<(), GpuError> {
        let what = "q38::enqueue_shared_add";
        let nm = a.n * a.m;
        if nm == 0 || a.slot >= a.slots {
            return Err(GpuError::shape(
                what,
                format!(
                    "{} values of {} tokens, slot {} of {}",
                    a.n, a.m, a.slot, a.slots
                ),
            ));
        }
        short(
            what,
            &[
                ("hsum", a.hsum.len(), nm),
                ("sh", a.sh.len(), nm),
                ("w", a.w.len(), a.slots * a.m),
                ("y", a.y.len(), nm),
            ],
        )?;
        let cfg = grid(what, nm)?;
        let prep = self.module.prepare_q38_shared_add(cfg)?;
        self.module.q38_shared_add(
            stream,
            &prep,
            a.hsum,
            a.sh,
            a.w,
            launch_u32(what, "slot", a.slot)?,
            launch_u32(what, "slots", a.slots)?,
            launch_u32(what, "n", a.n)?,
            launch_u32(what, "m", a.m)?,
            a.fault,
            a.y,
        )?;
        Ok(())
    }
}
