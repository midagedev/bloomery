//! V4.1's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! where a routed expert's bytes come from when the machine moves it onto the
//! stage card, where they go, and whether the host can serve an expert the
//! card gives up ([`Ds41Swap`], the model's [`SwapSource`]).
//!
//! **Parts.** An expert is three parts in stack order: gate and up (Q3_K,
//! `ff` rows of `n_embd` values each) and down (Q4_K, `n_embd` rows of `ff`).
//! On the card each part of slot `s` is the file's bytes of that expert at
//! byte `s · part` of its layer's stack, which the placed load uploads as
//! file bytes in slot order.
//!
//! **Sources.** The host reads its routed gates and ups from the r8 sidecar
//! when the load reads one (`BLOOMERY_R8`), so those are the bytes the load's
//! host set holds, and the down from the source file. A staged gate or up is
//! then the r8 row-lane layout, not Q3_K: once its bytes are in the slot, the
//! copy stream copies them to a scratch part and [`R8Kernels`] writes the
//! Q3_K bytes back into the slot (`ds41_r8_q3k`, the inverse of
//! `qdot::repack_q3k_r8`, bit for bit `qdot::unpack_q3k_r8`). The host
//! alternative, `qdot::unpack_q3k_r8` on the staging thread, is a scalar
//! bit gather of about 6.7k operations per 110-byte block, ~6·10⁸ for the
//! 92,160 blocks of a gate and an up [derived]: several steps' worth a flip.
//! On the card the scratch copy and the unpack are two passes over the part
//! at the card's copy rate, on the copy stream in the host leg's window.
//! Under `BLOOMERY_R8=off` every part is the source's bytes and nothing is
//! converted.
//!
//! **Host residency.** The host serves an expert from resident pages when
//! every byte it reads for it — the sidecar's (or source's) gate and up and
//! the source's down — lies in the load's host set and is in the page cache
//! now ([`HostSet::serves`], `mincore`): the set says what the load read in
//! and locked, the page cache what a step would fault on. A victim outside
//! the set is not host-resident whatever the page cache holds, so the
//! machine refuses its flip by name. The load puts each layer's churn pool —
//! its stage card experts past the pinned ones ([`ChurnPool`]) — in the set.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use bloomery_gpu::host::swap::{MachineCfg, Piece, Residency, SwapSource, Transform};
use bloomery_gpu::hybrid::SlotMap;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{GpuError, launch_u32, window};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, IntoResult, LaunchConfig1D, sys};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::Split;
use model::arch::deepseek41::names;
use model::placement::churn::ChurnPool;
use model::placement::host_lock::{HostFile, HostSet, expert_run};
use model::placement::{ModelTensor, Plan};
use model::r8file::R8Pair;

use crate::chain::ffn::CardStacks;

/// Passes from the boundary that makes a flip to the one it lands at. The
/// victims are host-resident (the churn pool), so a flip waits on no NVMe
/// read, only on its staging and its copy: a planning pass makes at most
/// `cap` = 24 flips, each one expert's memcpy into the pinned ring on one
/// thread and one H2D copy, which together stay under four passes' wall
/// [derived]; the staging runs in the host leg's wait window, about a
/// third of a step, and a late copy only makes the engine stream wait at the
/// landing.
pub const LIVE_DELAY: u64 = 4;

/// The bound on every host wait of the machine. Every boundary follows a
/// pass's readback, so the engine stream has drained and a wait is for the
/// staging thread and the copy stream alone: at most a planning pass's 24
/// experts, far inside this bound.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// Threads per block of the unpack: one output word each.
const UNPACK_THREADS: u32 = 256;

/// `sizeof(block_q3_K)` and one r8 group super-block (eight rows' blocks).
const Q3K_BLOCK: usize = 110;
const R8_ROWS: usize = 8;
const R8_BLOCK: usize = R8_ROWS * Q3K_BLOCK;
/// Where a group super-block's scales and code pairs start, and a pair's
/// bytes (`qdot::repack_q3k_r8` has the layout).
const R8_SCALES: usize = 16;
const R8_CODES: usize = R8_SCALES + 96;
const R8_PAIR: usize = 96;

/// Byte `o` of `src`, little-endian words.
///
/// SAFETY: `o / 4 < src.len()`.
#[inline(always)]
unsafe fn byte_at(src: &[u32], o: usize) -> u32 {
    // SAFETY: the caller's bound.
    (unsafe { *src.get_unchecked(o / 4) } >> (8 * (o % 4))) & 0xFF
}

/// Code `u = value + 4` of value `i` of row `r`, off the group super-block
/// at byte `blk` (`qdot`'s `r8_code`).
///
/// SAFETY: `blk + R8_BLOCK <= 4 · src.len()`.
#[inline(always)]
unsafe fn r8_code(src: &[u32], blk: usize, r: usize, i: usize) -> u32 {
    let (j, t) = (i / 16, i % 16);
    let (p, f) = (j / 2, 4 * (j % 2) + t / 4);
    let at = blk + R8_CODES + R8_PAIR * p + 4 * r + t % 4;
    // SAFETY: at + 64 < blk + R8_BLOCK by the layout, inside src.
    let (a, b, c) = unsafe {
        (
            byte_at(src, at),
            byte_at(src, at + 32),
            byte_at(src, at + 64),
        )
    };
    match f {
        0 => a & 7,
        1 => (a >> 3) & 7,
        2 => (a >> 6) | (((c >> 6) & 1) << 2),
        3 => b & 7,
        4 => (b >> 3) & 7,
        5 => (b >> 6) | (((c >> 7) & 1) << 2),
        6 => c & 7,
        _ => (c >> 3) & 7,
    }
}

/// The six-bit scale `s + 32` of sub-block `j` of row `r`, off the group
/// super-block at byte `blk` (`qdot`'s `r8_scale` plus 32).
///
/// SAFETY: `blk + R8_BLOCK <= 4 · src.len()`.
#[inline(always)]
unsafe fn r8_scale6(src: &[u32], blk: usize, r: usize, j: usize) -> u32 {
    let p = j / 2;
    let (q, b) = (p / 2, 16 * (p % 2) + 2 * r + j % 2);
    // SAFETY: both bytes lie in the super-block's scale bytes 16..112.
    let (lo, hi) = unsafe {
        (
            byte_at(src, blk + R8_SCALES + 32 * (q / 2) + b),
            byte_at(src, blk + R8_SCALES + 64 + b),
        )
    };
    ((lo >> (4 * (q % 2))) & 0xF) | (((hi >> (2 * q)) & 3) << 4)
}

/// Byte `b` of row `r`'s Q3_K block, off the group super-block at byte `blk`:
/// `hmask` (0..32), `qs` (32..96), the scales (96..108) and `d` (108..110),
/// as `qdot`'s `q3k_put_codes` and `q3k_put_scales6` write them.
///
/// SAFETY: `blk + R8_BLOCK <= 4 · src.len()`, `r < 8`, `b < 110`.
#[inline(always)]
unsafe fn q3k_byte(src: &[u32], blk: usize, r: usize, b: usize) -> u32 {
    // SAFETY: every call's `i < 256` and the caller's bounds on `blk`, `r`.
    let code = |i: usize| unsafe { r8_code(src, blk, r, i) };
    // SAFETY: every call's `j < 16` and the caller's bounds on `blk`, `r`.
    let s = |j: usize| unsafe { r8_scale6(src, blk, r, j) };
    if b < 32 {
        let mut high = 0u32;
        let mut hf = 0usize;
        while hf < 8 {
            let (h, f) = (hf / 4, hf % 4);
            high |= (code(128 * h + 32 * f + b) >> 2) << (4 * h + f);
            hf += 1;
        }
        high
    } else if b < 96 {
        let (h, c) = ((b - 32) / 32, (b - 32) % 32);
        let mut low = 0u32;
        let mut f = 0usize;
        while f < 4 {
            low |= (code(128 * h + 32 * f + c) & 3) << (2 * f);
            f += 1;
        }
        low
    } else if b < 100 {
        let k = b - 96;
        (s(k) & 0xF) | ((s(8 + k) & 0xF) << 4)
    } else if b < 104 {
        let k = b - 100;
        (s(4 + k) & 0xF) | ((s(12 + k) & 0xF) << 4)
    } else if b < 108 {
        let k = b - 104;
        (s(k) >> 4) | ((s(4 + k) >> 4) << 2) | ((s(8 + k) >> 4) << 4) | ((s(12 + k) >> 4) << 6)
    } else {
        // SAFETY: byte 2r + 1 < 16 of the super-block.
        unsafe { byte_at(src, blk + 2 * r + (b - 108)) }
    }
}

#[cuda_module]
mod r8_kernels {
    use super::*;

    /// `dst` = Q3_K rows of `src`, the r8 row-lane layout of rows of `nb`
    /// super-blocks in groups of eight: thread `i` writes word `i` of
    /// `dst`, each of its bytes read off the group super-block it lies in.
    /// Integer bit moves only, so the output is `qdot::unpack_q3k_r8`'s bit
    /// for bit.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (src.len() >= words, dst.len() >= words, nb >= 1)
    )]
    pub fn ds41_r8_q3k(src: &[u32], nb: u32, words: u32, mut dst: DisjointSlice<u32>) {
        let i = thread::index_1d().get();
        if i >= words as usize {
            return;
        }
        let row_bytes = nb as usize * Q3K_BLOCK;
        let group = R8_ROWS * row_bytes;
        let mut w = 0u32;
        let mut q = 0usize;
        while q < 4 {
            let o = 4 * i + q;
            let (g, ob) = (o / group, o % group);
            let (r, x) = (ob / row_bytes, ob % row_bytes);
            let blk = g * group + (x / Q3K_BLOCK) * R8_BLOCK;
            // SAFETY: `words` is a whole number of groups (the host's
            // check), so the group super-block holding byte o ends inside
            // src's `words` words; r < 8 and x % 110 < 110.
            w |= unsafe { q3k_byte(src, blk, r, x % Q3K_BLOCK) } << (8 * q);
            q += 1;
        }
        // SAFETY: i < words <= dst.len(); thread i is dst[i]'s only writer.
        unsafe {
            *dst.get_unchecked_mut(i) = w;
        }
    }
}

/// The loaded unpack module, and its scratch part on the stage card: the
/// copy stream copies a staged part there and unpacks it back into its slot.
/// Used on the machine's copy stream alone, one part at a time.
pub struct R8Kernels {
    module: r8_kernels::LoadedModule,
    scratch: DeviceBuffer<u32>,
    ctx: Arc<CudaContext>,
}

impl R8Kernels {
    /// Load the module into `ctx` with a scratch part of `words` words.
    /// Load-time only.
    pub fn load(
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
        words: usize,
    ) -> Result<R8Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { r8_kernels::load(ctx)? };
        Ok(R8Kernels {
            module,
            scratch: DeviceBuffer::zeroed(stream, words)?,
            ctx: Arc::clone(ctx),
        })
    }

    /// Enqueue on `stream`: the `words` words at `at` into the scratch, then
    /// their Q3_K rows (rows of `nb` super-blocks) back at `at`. Refused
    /// before anything is enqueued: a part past the scratch, or not a whole
    /// number of 8-row groups.
    pub fn enqueue_in_place(
        &self,
        stream: &CudaStream,
        at: sys::CUdeviceptr,
        words: usize,
        nb: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "R8Kernels::enqueue_in_place";
        let group_words = R8_ROWS * nb * Q3K_BLOCK / 4;
        if nb == 0 || words > self.scratch.len() || !words.is_multiple_of(group_words) {
            return Err(shape(
                WHAT,
                format!(
                    "{words} words of rows of {nb} super-blocks (groups of {group_words} words) \
                     through a scratch of {}",
                    self.scratch.len()
                ),
            ));
        }
        // SAFETY: `at` is a slot of a stage stack (the source's `dest`), which
        // holds `words` words and stays allocated for the machine's life; the
        // scratch holds at least `words`; both are this context's, and only
        // this stream touches either while the copy runs.
        let rc = unsafe {
            sys::cuMemcpyDtoDAsync_v2(
                self.scratch.cu_deviceptr(),
                at,
                4 * words,
                stream.cu_stream(),
            )
        };
        rc.result().map_err(|source| GpuError::Driver {
            op: Some("cuMemcpyDtoDAsync_v2 (r8 scratch)"),
            source,
        })?;
        // SAFETY: the same slot, `words` u32 of a live allocation of this
        // context, aligned (a slot starts at a whole word).
        let mut dst = unsafe { window::<u32>(at, words, &self.ctx) };
        let grid = launch_u32(WHAT, "grid", words.div_ceil(UNPACK_THREADS as usize))?;
        let (nb32, words32) = (
            launch_u32(WHAT, "nb", nb)?,
            launch_u32(WHAT, "words", words)?,
        );
        let prep = self
            .module
            .prepare_ds41_r8_q3k(LaunchConfig1D::new(grid, UNPACK_THREADS, 0))?;
        self.module
            .ds41_r8_q3k(stream, &prep, &self.scratch, nb32, words32, &mut dst)?;
        Ok(())
    }
}

fn shape(what: &'static str, detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what,
        detail: detail.into(),
    }
}

/// One layer's parts: the file tensors, the stage stacks' base addresses and
/// slots, and each part's bytes.
struct LayerParts {
    tensors: [ModelTensor; 3],
    base: [sys::CUdeviceptr; 3],
    slots: usize,
}

/// V4.1's [`SwapSource`] over one placed load's stage card.
pub struct Ds41Swap {
    pair: R8Pair,
    set: HostSet,
    experts: u64,
    first: usize,
    layers: Vec<Option<LayerParts>>,
    parts: [usize; 3],
    /// Rows of a gate or up part, and super-blocks of each of its rows.
    ff: usize,
    nb: usize,
    /// The unpack, when the host's gates and ups are the sidecar's.
    unpack: Option<R8Kernels>,
}

// SAFETY: the stack addresses are plain device pointers into allocations the
// model keeps in place for the machine's life; the kernels and the scratch
// are used from the one thread that drives the machine (`convert`), and
// everything the staging thread reads (`source`, `prepare_victim`) is the
// split's and the sidecar's read-only mappings and the set.
unsafe impl Send for Ds41Swap {}
// SAFETY: as for `Send`: no call mutates shared state.
unsafe impl Sync for Ds41Swap {}

impl Ds41Swap {
    /// The source of `plan`'s card `card`, whose layers `layers` hold their
    /// routed stacks in `w`: the split `file` and the host reading `r8` the
    /// load took (the same sidecar open as the host set's), and `set`, the
    /// host set the load read in. Refused by name: a set the load did not
    /// populate (`populated`), a layer whose stacks are not the plan's
    /// experts, parts not whole words. Load-time only.
    #[allow(
        clippy::too_many_arguments,
        reason = "the load's plan, card, file, host reading, set, stacks and context (rust-quality R8)"
    )]
    pub fn new(
        plan: &Plan<'_>,
        layers: Range<usize>,
        file: Arc<Split>,
        r8: bool,
        set: HostSet,
        populated: bool,
        w: &Weights,
        ctx: &Arc<CudaContext>,
        stream: &CudaStream,
    ) -> Result<Ds41Swap, GpuError> {
        const WHAT: &str = "Ds41Swap::new";
        if !populated {
            return Err(shape(
                WHAT,
                "adaptive residency on a load whose host set was not read in \
                 (BLOOMERY_HOST_POPULATE=0): no expert would be host-resident",
            ));
        }
        let pair = R8Pair::at_load(file, r8).map_err(|e| GpuError::plan(WHAT, e))?;
        let experts = plan.model.experts;
        let find = |name: String| {
            plan.model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .cloned()
                .ok_or(GpuError::Tensor {
                    what: WHAT,
                    name,
                    need: "a routed stack of the plan",
                })
        };
        let mut out = Vec::with_capacity(layers.len());
        let mut parts: Option<[usize; 3]> = None;
        for l in layers.clone() {
            let Some(s) = CardStacks::of(w, l)? else {
                out.push(None);
                continue;
            };
            let tensors = [
                find(names::ffn_gate_exps(l))?,
                find(names::ffn_up_exps(l))?,
                find(names::ffn_down_exps(l))?,
            ];
            let per: [usize; 3] = std::array::from_fn(|i| {
                usize::try_from(tensors[i].file_bytes / experts.max(1)).unwrap_or(0)
            });
            let stacks = [s.gate, s.up, s.down];
            let slots = s.down.rows() / tensors[2].dims[1].max(1) as usize;
            for (i, st) in stacks.iter().enumerate() {
                if per[i] == 0 || !per[i].is_multiple_of(4) || st.buf().len() * 4 < slots * per[i] {
                    return Err(shape(
                        WHAT,
                        format!(
                            "layer {l} part {i}: {} bytes an expert, a stack of {} words for {slots} \
                             slots",
                            per[i],
                            st.buf().len()
                        ),
                    ));
                }
            }
            match parts {
                Some(p) if p != per => {
                    return Err(shape(
                        WHAT,
                        format!("layer {l}: parts of {per:?} bytes, earlier layers {p:?}"),
                    ));
                }
                _ => parts = Some(per),
            }
            out.push(Some(LayerParts {
                tensors,
                base: stacks.map(|st| st.buf().cu_deviceptr()),
                slots,
            }));
        }
        let parts = parts.ok_or_else(|| shape(WHAT, "no layer of the card holds routed stacks"))?;
        let t0 = out
            .iter()
            .flatten()
            .next()
            .map(|p| p.tensors[0].clone())
            .ok_or_else(|| shape(WHAT, "no layer of the card holds routed stacks"))?;
        let (n_embd, ff) = (t0.dims[0] as usize, t0.dims[1] as usize);
        let nb = n_embd / 256;
        let unpack = match pair.r8().sidecar() {
            Some(_) => Some(R8Kernels::load(ctx, stream, parts[0].max(parts[1]) / 4)?),
            None => None,
        };
        Ok(Ds41Swap {
            pair,
            set,
            experts,
            first: layers.start,
            layers: out,
            parts,
            ff,
            nb,
            unpack,
        })
    }

    /// The host reads the gates and ups from the r8 sidecar, so a flip
    /// unpacks them on the card ([`R8Kernels`]).
    pub fn unpacks(&self) -> bool {
        self.unpack.is_some()
    }

    /// Part `part` of layer `layer`'s expert `id` as a static load uploads
    /// it to a card slot: the source file's bytes (gate and up Q3_K, down
    /// Q4_K), whatever the host reads.
    pub fn card_bytes(&self, layer: usize, id: u32, part: usize) -> Result<&[u8], GpuError> {
        const WHAT: &str = "Ds41Swap::card_bytes";
        let t = &self.layer(layer, WHAT)?.tensors[part.min(2)];
        let per = *self
            .parts
            .get(part)
            .ok_or_else(|| shape(WHAT, format!("part {part} of an expert of three")))?;
        let split = self.pair.source().split();
        let (s, info) = split.find(&t.name).ok_or(GpuError::Tensor {
            what: WHAT,
            name: t.name.clone(),
            need: "a routed stack of the split",
        })?;
        let whole = split
            .shard(s)
            .ok_or_else(|| shape(WHAT, format!("shard {s} of the split")))?
            .data(info)
            .map_err(|e| GpuError::plan(WHAT, e))?;
        let at = id as usize * per;
        whole
            .get(at..at + per)
            .ok_or_else(|| shape(WHAT, format!("layer {layer} expert {id}: past {}", t.name)))
    }

    fn layer(&self, layer: usize, what: &'static str) -> Result<&LayerParts, GpuError> {
        layer
            .checked_sub(self.first)
            .and_then(|i| self.layers.get(i))
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                shape(
                    what,
                    format!("layer {layer} holds no routed stack on the stage card"),
                )
            })
    }

    /// Where the host reads part `part` of layer `layer`'s expert `id`: the
    /// sidecar's run for a gate or an up when the load reads one, else the
    /// source's.
    fn run(
        &self,
        layer: usize,
        id: u32,
        part: usize,
        what: &'static str,
    ) -> Result<(HostFile, Range<u64>), GpuError> {
        let t = &self.layer(layer, what)?.tensors[part];
        expert_run(self.pair.source(), t, self.experts, id, part < 2)
            .map_err(|e| GpuError::plan(what, e))
    }
}

impl SwapSource for Ds41Swap {
    fn part_bytes(&self) -> &[usize] {
        &self.parts
    }

    fn source(&self, layer: usize, id: u32, part: usize) -> Result<Piece<'_>, GpuError> {
        const WHAT: &str = "Ds41Swap::source";
        let t = &self.layer(layer, WHAT)?.tensors[part];
        let per = *self
            .parts
            .get(part)
            .ok_or_else(|| shape(WHAT, format!("part {part} of an expert of three")))?;
        let src = self.pair.source();
        let whole = match src.sidecar().filter(|_| part < 2) {
            Some(side) => side.data(&t.name).map_err(|e| GpuError::plan(WHAT, e))?,
            None => {
                let split = src.split();
                let (s, info) = split.find(&t.name).ok_or(GpuError::Tensor {
                    what: WHAT,
                    name: t.name.clone(),
                    need: "a routed stack of the split",
                })?;
                split
                    .shard(s)
                    .ok_or_else(|| shape(WHAT, format!("shard {s} of the split")))?
                    .data(info)
                    .map_err(|e| GpuError::plan(WHAT, e))?
            }
        };
        let at = id as usize * per;
        let bytes = whole.get(at..at + per).ok_or_else(|| {
            shape(
                WHAT,
                format!(
                    "layer {layer} expert {id}: past the {} bytes of {}",
                    whole.len(),
                    t.name
                ),
            )
        })?;
        Ok(Piece {
            bytes,
            transform: Transform::Identity,
        })
    }

    fn dest(&self, layer: usize, part: usize, slot: u32) -> Result<sys::CUdeviceptr, GpuError> {
        const WHAT: &str = "Ds41Swap::dest";
        let p = self.layer(layer, WHAT)?;
        if slot as usize >= p.slots || part >= 3 {
            return Err(shape(
                WHAT,
                format!(
                    "layer {layer} part {part} slot {slot}: {} slots of 3 parts",
                    p.slots
                ),
            ));
        }
        Ok(p.base[part] + (slot as usize * self.parts[part]) as u64)
    }

    /// A gate or an up staged from the sidecar is its r8 layout: unpacked in
    /// place into Q3_K ([`R8Kernels::enqueue_in_place`]).
    fn convert(
        &self,
        _layer: usize,
        part: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        match (&self.unpack, part) {
            (Some(k), 0 | 1) => {
                let words = self.parts[part] / 4;
                debug_assert_eq!(self.parts[part], self.ff * self.nb * Q3K_BLOCK);
                k.enqueue_in_place(stream, dst, words, self.nb)
            }
            _ => Ok(()),
        }
    }

    fn prepare_victim(&self, layer: usize, id: u32) -> Result<(), GpuError> {
        const WHAT: &str = "Ds41Swap::prepare_victim";
        for part in 0..3 {
            let (file, at) = self.run(layer, id, part, WHAT)?;
            self.set
                .populate_run(self.pair.source(), &file, &at)
                .map_err(|e| GpuError::plan(WHAT, e))?;
        }
        Ok(())
    }

    fn host_resident(&self, layer: usize, id: u32) -> Result<bool, GpuError> {
        const WHAT: &str = "Ds41Swap::host_resident";
        for part in 0..3 {
            let (file, at) = self.run(layer, id, part, WHAT)?;
            let serves = self
                .set
                .serves(self.pair.source(), &file, &at)
                .map_err(|e| GpuError::plan(WHAT, e))?;
            if !serves {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Nothing: every stage card expert the machine can move is the churn
    /// pool's, which the load's host set holds for the model's life, and
    /// the host never reads the rest of a card expert's bytes (the source's
    /// gate and up under the sidecar), which the load already released. The
    /// reset lets go of no byte the host set held, so it reports 0.
    fn release_host(&self, _layer: usize, _id: u32) -> Result<u64, GpuError> {
        Ok(0)
    }
}

/// The machine's shape for a V4.1 load under `residency` over `map`: the
/// `mid` rule at [`LIVE_DELAY`], `pinned` of it per layer (a layer's card
/// holds fewer is the machine's refusal), `top_k` ids a row, passes of up to
/// `max_rows` rows. `None` for `off`.
#[must_use]
pub fn machine_cfg(
    residency: Residency,
    map: &SlotMap,
    top_k: usize,
    max_rows: usize,
) -> Option<MachineCfg> {
    let (params, pinned) = residency.params(LIVE_DELAY)?;
    Some(MachineCfg {
        params,
        pinned: vec![pinned; map.layers().len()],
        top_k,
        max_rows,
        deadline: DEADLINE,
    })
}

/// The churn pool a V4.1 load under `residency` holds in its host set: card
/// `card`'s experts past the pinned ones ([`ChurnPool::of`]); `None` for
/// `off`.
pub fn churn(
    plan: &Plan<'_>,
    card: usize,
    residency: Residency,
) -> Result<Option<ChurnPool>, GpuError> {
    match residency {
        Residency::Off => Ok(None),
        Residency::Mid { pinned, .. } => ChurnPool::of(plan, card, pinned)
            .map(Some)
            .map_err(|e| GpuError::plan("deepseek41 residency churn pool", e)),
    }
}
