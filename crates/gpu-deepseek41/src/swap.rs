//! V4.1's side of adaptive expert residency ([`bloomery_gpu::host::swap`]):
//! the model's [`FileStacks`] for the common file source
//! (`bloomery_gpu::host::swap_source::FileSwap`) — [`Ds41Stacks`], its gate,
//! up and down and the r8 unpack of a staged gate or up — with the load's
//! churn pool ([`churn`]) and the machine's live delay and deadline.
//!
//! **Parts.** An expert is three parts in stack order: gate and up (Q3_K,
//! `ff` rows of `n_embd` values each) and down (Q4_K, `n_embd` rows of `ff`).
//! On the card each part of slot `s` is the file's bytes of that expert at
//! byte `s · part` of its layer's stack, which the placed load uploads as
//! file bytes in slot order.
//!
//! **The unpack.** The host reads its routed gates and ups from the r8
//! sidecar when the load reads one (`BLOOMERY_R8`), so those are the bytes
//! the load's host set holds, and the down from the source file. A staged
//! gate or up is then the r8 row-lane layout, not Q3_K: once its bytes are in
//! the slot, [`R8Kernels`] turns them into the Q3_K bytes in place on the
//! copy stream (`ds41_r8_q3k_groups`, the inverse of `qdot::repack_q3k_r8`,
//! bit for bit `qdot::unpack_q3k_r8`). A group of eight rows is the same byte
//! range in both layouts, so one block reads its group into shared memory and
//! writes it back as Q3_K rows: no scratch part and no device-to-device
//! copy. The host alternative, `qdot::unpack_q3k_r8` on the staging thread,
//! is a scalar bit gather of about 6.7k operations per 110-byte block, ~6·10⁸
//! for the 92,160 blocks of a gate and an up [derived]: several steps' worth
//! a flip. Under `BLOOMERY_R8=off` every part is the source's bytes and
//! nothing is converted.

use std::sync::Arc;
use std::time::Duration;

use bloomery_gpu::host::swap::Residency;
use bloomery_gpu::host::swap_source::{Convert, FileStacks};
use bloomery_gpu::{GpuError, launch_u32, window};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D, sys};
use cuda_device::vector::U32x4;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use gguf::quant::GgmlType;
use model::arch::deepseek41::names;
use model::placement::Plan;
use model::placement::churn::ChurnPool;

/// Passes from the boundary that makes a flip to the one it lands at. The
/// victims are host-resident (the churn pool), so a flip waits on no NVMe
/// read, only on its staging and its copy: a planning pass makes at most
/// the rule's `cap` flips ([`runtime::swaprule::SwapParams::mid`]), each one
/// expert's memcpy into the pinned ring on one thread and one H2D copy,
/// which together stay under four passes' wall
/// [derived]; the staging runs in the host leg's wait window, about a
/// third of a step, and a late copy only makes the engine stream wait at the
/// landing.
pub const LIVE_DELAY: u64 = 4;

/// The bound on every host wait of the machine. Every boundary follows a
/// pass's readback, so the engine stream has drained and a wait is for the
/// staging thread and the copy stream alone: at most a planning pass's
/// `cap` experts ([`runtime::swaprule::SwapParams::mid`]), far inside this
/// bound.
pub const DEADLINE: Duration = Duration::from_secs(30);

/// Threads per block of the reference unpack (`ds41_r8_q3k`): one output
/// word each.
const UNPACK_THREADS: u32 = 256;

/// Threads per block of the group unpack (`ds41_r8_q3k_groups`).
const GROUP_THREADS: u32 = 256;
const GT: usize = GROUP_THREADS as usize;

/// Blocks of the group unpack at most, each walking the groups `b, b +
/// gridDim, …`: half the A6000's 84 SMs, so a flip's unpack holds at most
/// half the stage card's SMs beside the engine stream. Its cost to the
/// engine is then its DRAM bytes, which this does not change.
const GROUP_BLOCKS: usize = 42;

/// Super-blocks a row holds at most for the group unpack: V4.1's 5120-value
/// rows, a shared group of 880 · 20 = 17,600 bytes.
const R8_NB_MAX: usize = 20;

/// `sizeof(block_q3_K)` and one r8 group super-block (eight rows' blocks).
const Q3K_BLOCK: usize = 110;
const R8_ROWS: usize = 8;
const R8_BLOCK: usize = R8_ROWS * Q3K_BLOCK;
/// Where a group super-block's scales and code pairs start, and a pair's
/// bytes (`qdot::repack_q3k_r8` has the layout).
const R8_SCALES: usize = 16;
const R8_CODES: usize = R8_SCALES + 96;
const R8_PAIR: usize = 96;

/// The same layout in words: a group super-block, where its scale vectors
/// and its code pairs start, and a pair.
const R8_BLOCK_WORDS: usize = R8_BLOCK / 4;
const R8_SCALE_WORDS: usize = R8_SCALES / 4;
const R8_CODE_WORDS: usize = R8_CODES / 4;
const R8_PAIR_WORDS: usize = R8_PAIR / 4;
/// Tasks of one group super-block: each row's eight code fields.
const R8_TASKS: usize = R8_ROWS * 8;
/// The shared group's words at [`R8_NB_MAX`].
const R8_GROUP_WORDS_MAX: usize = R8_BLOCK_WORDS * R8_NB_MAX;

/// Byte `o` of `src`, little-endian words.
///
/// # Safety
///
/// `o / 4 < src.len()`.
#[inline(always)]
unsafe fn byte_at(src: &[u32], o: usize) -> u32 {
    // SAFETY: the caller's bound.
    (unsafe { *src.get_unchecked(o / 4) } >> (8 * (o % 4))) & 0xFF
}

/// Code `u = value + 4` of value `i` of row `r`, off the group super-block
/// at byte `blk` (`qdot`'s `r8_code`).
///
/// # Safety
///
/// `blk + R8_BLOCK <= 4 · src.len()`.
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
/// # Safety
///
/// `blk + R8_BLOCK <= 4 · src.len()`.
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
/// # Safety
///
/// `blk + R8_BLOCK <= 4 · src.len()`, `r < 8`, `b < 110`.
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

/// Eight four-bit entries, entry `f` at bits `4f`.
const fn nibbles(v: [u32; 8]) -> u32 {
    let mut out = 0;
    let mut f = 0;
    while f < 8 {
        out |= v[f] << (4 * f);
        f += 1;
    }
    out
}

/// Code field `f` of a pair (`W_f` of `qdot::repack_q3k_r8`'s layout): the
/// pair word its two low bits are in (0 = A, 1 = B, 2 = C) and their shift,
/// and the word and shift of its high bit.
const LO_WORD: u32 = nibbles([0, 0, 0, 1, 1, 1, 2, 2]);
const LO_SHIFT: u32 = nibbles([0, 3, 6, 0, 3, 6, 0, 3]);
const HI_WORD: u32 = nibbles([0, 0, 2, 1, 1, 2, 2, 2]);
const HI_SHIFT: u32 = nibbles([2, 5, 6, 2, 5, 7, 2, 5]);

#[inline(always)]
fn nibble(table: u32, f: usize) -> u32 {
    (table >> (4 * f)) & 0xF
}

/// Task `t` of a group (`t < 64 · nb`): super-block `t / 64`, row
/// `t / 8 % 8`, code field `t % 8`.
#[inline(always)]
fn r8_task(t: usize) -> (usize, usize, usize) {
    (t >> 6, (t >> 3) & 7, t & 7)
}

/// The byte row `r`'s Q3_K block `sb` starts at in its group, rows of `nb`
/// super-blocks.
#[inline(always)]
fn q3k_block_at(r: usize, sb: usize, nb: usize) -> usize {
    Q3K_BLOCK * (nb * r + sb)
}

/// Scale word `w` (bytes `96 + 4w ..`) of row `r`'s Q3_K block, off the
/// group super-block at word `blk` of `s`. Byte `k` of a gathered vector is
/// its byte `16 (k / 2) + 2r + k % 2`: sub-block `4q + k` of the row in
/// `L0 = X_0 | X_1 << 4`, `L1 = X_2 | X_3 << 4` and the high pairs `X_q >> 4`
/// at bits `2q`.
///
/// # Safety
///
/// `s` holds words `blk .. blk + 28`, `r < 8`.
#[inline(always)]
unsafe fn r8_scale_word(s: *const u32, blk: usize, r: usize, w: usize) -> u32 {
    let at = blk + R8_SCALE_WORDS + r / 2;
    let sh = 16 * (r as u32 % 2);
    // SAFETY: words `at + v + 4 <= blk + 27` for `v <= 16`.
    let gather = |v: usize| unsafe {
        ((*s.add(at + v) >> sh) & 0xFFFF) | ((*s.add(at + v + 4) >> sh) << 16)
    };
    match w {
        0 => (gather(0) & 0x0F0F_0F0F) | ((gather(8) & 0x0F0F_0F0F) << 4),
        1 => ((gather(0) >> 4) & 0x0F0F_0F0F) | (gather(8) & 0xF0F0_F0F0),
        _ => gather(16),
    }
}

/// What task (`sb`, `r`, `f`) writes of row `r`'s Q3_K block `sb`, off the
/// group in `s`: `(byte in the block, word, bytes)` for hmask word `f`, `qs`
/// words `8 + f` and `16 + f`, and scale word `f` (`f < 3`), `d` (`f = 3`,
/// two bytes) or nothing (`f > 3`, zero bytes). Value `128h + 32f' + c` of
/// the row is byte `c % 4` of field `c / 4` in pair `4h + f'` (`qdot`'s
/// `r8_code`), so hmask byte `c` and `qs` byte `32h + c` gather field
/// `c / 4` of the pairs, four bytes a word, and the hmask bit of pair `p`
/// is bit `p`.
///
/// # Safety
///
/// `s` holds the group's super-block `sb` (words `220 sb ..
/// 220 (sb + 1)`), `r < 8`, `f < 8`.
#[inline(always)]
unsafe fn r8_task_words(s: *const u32, sb: usize, r: usize, f: usize) -> [(usize, u32, usize); 4] {
    let blk = R8_BLOCK_WORDS * sb;
    let code = blk + R8_CODE_WORDS + r;
    let (lo, ls) = (code + 8 * nibble(LO_WORD, f) as usize, nibble(LO_SHIFT, f));
    let (hi, hs) = (code + 8 * nibble(HI_WORD, f) as usize, nibble(HI_SHIFT, f));
    let (mut q0, mut q1, mut h) = (0u32, 0u32, 0u32);
    let mut p = 0;
    while p < 8 {
        // SAFETY: the last pair's C word is `blk + 28 + r + 16 + 168 <
        // blk + 220`.
        let (x, y) = unsafe {
            (
                *s.add(lo + R8_PAIR_WORDS * p) >> ls,
                *s.add(hi + R8_PAIR_WORDS * p) >> hs,
            )
        };
        let low = (x & 0x0303_0303) << (2 * (p % 4));
        if p < 4 {
            q0 |= low;
        } else {
            q1 |= low;
        }
        h |= (y & 0x0101_0101) << p;
        p += 1;
    }
    let last = if f < 3 {
        // SAFETY: the scale vectors are words `blk + 4 .. blk + 28`.
        (96 + 4 * f, unsafe { r8_scale_word(s, blk, r, f) }, 4)
    } else if f == 3 {
        // SAFETY: the rows' `d` are words `blk .. blk + 4`.
        let d = unsafe { *s.add(blk + r / 2) } >> (16 * (r as u32 % 2));
        (108, d & 0xFFFF, 2)
    } else {
        (0, 0, 0)
    };
    [
        (4 * f, h, 4),
        (32 + 4 * f, q0, 4),
        (64 + 4 * f, q1, 4),
        last,
    ]
}

/// The low `n` bytes (4, 2 or 0) of `w` at byte `o` of `base`, `o` even: one
/// word store where `o` is a whole word, else halves.
///
/// # Safety
///
/// bytes `o .. o + n` lie in the allocation `base` points into, and
/// this thread is their only writer.
#[inline(always)]
unsafe fn put(base: *mut u32, o: usize, w: u32, n: usize) {
    let half = base.cast::<u16>();
    // SAFETY: the caller's bounds; `o` even keeps every half aligned.
    unsafe {
        if n == 4 && o.is_multiple_of(4) {
            *base.add(o / 4) = w;
        } else if n == 4 {
            *half.add(o / 2) = w as u16;
            *half.add(o / 2 + 1) = (w >> 16) as u16;
        } else if n == 2 {
            *half.add(o / 2) = w as u16;
        }
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

    /// `part` = its Q3_K rows in place, where it holds the r8 row-lane layout
    /// of `groups` groups of eight rows of `nb` super-blocks. A group is the
    /// same byte range in both layouts: block `b` takes the groups `b, b +
    /// gridDim, …`, reads one into shared memory, and its threads write the
    /// group's Q3_K bytes back, task by task ([`r8_task_words`]), so no block
    /// reads bytes another writes. Integer bit moves only: bit for bit
    /// `ds41_r8_q3k` and `qdot::unpack_q3k_r8`.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (nb >= 1, nb <= R8_NB_MAX, part.len() >= groups * nb * R8_BLOCK_WORDS)
    )]
    pub fn ds41_r8_q3k_groups(groups: u32, nb: u32, mut part: DisjointSlice<u32>) {
        static mut GROUP: SharedArray<u32, R8_GROUP_WORDS_MAX, 16> = SharedArray::UNINIT;
        let tid = thread::threadIdx_x() as usize;
        let nb = nb as usize;
        let (words, tasks) = (R8_BLOCK_WORDS * nb, R8_TASKS * nb);
        let vecs = words / 4;
        // SAFETY: GROUP is this block's own shared allocation, 16-byte
        // aligned, and holds `words <= R8_GROUP_WORDS_MAX` (nb <= R8_NB_MAX).
        let s = unsafe { SharedArray::as_raw_mut_ptr(&raw mut GROUP) };
        let sv = s.cast::<U32x4>();
        let base = part.as_mut_ptr();
        let mut g = thread::blockIdx_x() as usize;
        while g < groups as usize {
            let at = g * words;
            // SAFETY: at + words <= groups · nb · 220 <= part.len() (the
            // contract). The host passes a 16-byte-aligned part, and a
            // group's 880 · nb bytes are a multiple of 16.
            let gv = unsafe { base.add(at) }.cast::<U32x4>().cast_const();
            let mut v = tid;
            // Four 16-byte loads in flight before their shared stores.
            while v + 3 * GT < vecs {
                // SAFETY: the four vectors lie below `vecs` in the group and
                // in the shared group; thread `tid` alone writes them.
                unsafe {
                    let x0 = *gv.add(v);
                    let x1 = *gv.add(v + GT);
                    let x2 = *gv.add(v + 2 * GT);
                    let x3 = *gv.add(v + 3 * GT);
                    *sv.add(v) = x0;
                    *sv.add(v + GT) = x1;
                    *sv.add(v + 2 * GT) = x2;
                    *sv.add(v + 3 * GT) = x3;
                }
                v += 4 * GT;
            }
            while v < vecs {
                // SAFETY: as above, one vector.
                unsafe { *sv.add(v) = *gv.add(v) };
                v += GT;
            }
            thread::sync_threads();
            let mut t = tid;
            while t < tasks {
                let (sb, r, f) = r8_task(t);
                let blk = 4 * at + q3k_block_at(r, sb, nb);
                // SAFETY: sb < nb, r < 8, f < 8, and the shared group holds
                // the group's nb super-blocks, written before the barrier.
                let [a, b, c, d] = unsafe { r8_task_words(s, sb, r, f) };
                // SAFETY: each piece lies in row r's block sb of group g,
                // inside the part; the tasks' pieces tile the group's bytes
                // once (`tasks_tile_the_group`), and every read of the group
                // came before the barrier.
                unsafe {
                    put(base, blk + a.0, a.1, a.2);
                    put(base, blk + b.0, b.1, b.2);
                    put(base, blk + c.0, c.1, c.2);
                    put(base, blk + d.0, d.1, d.2);
                }
                t += GT;
            }
            // The next group's loads overwrite the shared group.
            thread::sync_threads();
            g += thread::gridDim_x() as usize;
        }
    }
}

/// The loaded unpack module, used on the machine's copy stream.
pub struct R8Kernels {
    module: r8_kernels::LoadedModule,
    ctx: Arc<CudaContext>,
}

impl R8Kernels {
    /// Load the module into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<R8Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the
        // module above; the launcher checks its launch contract.
        let module = unsafe { r8_kernels::load(ctx)? };
        Ok(R8Kernels {
            module,
            ctx: Arc::clone(ctx),
        })
    }

    /// Enqueue on `stream` the Q3_K rows (rows of `nb` super-blocks) of the
    /// `words` words at `at`, in place of the r8 row-lane layout they hold
    /// (`ds41_r8_q3k_groups`). Refused before anything is enqueued: `at` off
    /// a 16-byte boundary, rows of more than [`R8_NB_MAX`] super-blocks, or
    /// not a whole number of 8-row groups.
    pub fn enqueue_in_place(
        &self,
        stream: &CudaStream,
        at: sys::CUdeviceptr,
        words: usize,
        nb: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "R8Kernels::enqueue_in_place";
        let group_words = R8_BLOCK_WORDS * nb;
        if nb == 0
            || nb > R8_NB_MAX
            || words == 0
            || !words.is_multiple_of(group_words)
            || !at.is_multiple_of(16)
        {
            return Err(shape(
                WHAT,
                format!(
                    "{words} words at {at:#x} of rows of {nb} super-blocks: a group is \
                     {group_words} words, at most {R8_NB_MAX} super-blocks a row, at a \
                     16-byte boundary"
                ),
            ));
        }
        let groups = words / group_words;
        // SAFETY: `at` is a slot of a stage stack (the source's `dest`), which
        // holds `words` words and stays allocated until the copy stream has
        // drained, of this context; only this stream touches it while the
        // unpack runs.
        let mut part = unsafe { window::<u32>(at, words, &self.ctx) };
        let grid = launch_u32(WHAT, "grid", groups.min(GROUP_BLOCKS))?;
        let (groups32, nb32) = (
            launch_u32(WHAT, "groups", groups)?,
            launch_u32(WHAT, "nb", nb)?,
        );
        let prep =
            self.module
                .prepare_ds41_r8_q3k_groups(LaunchConfig1D::new(grid, GROUP_THREADS, 0))?;
        self.module
            .ds41_r8_q3k_groups(stream, &prep, groups32, nb32, &mut part)?;
        Ok(())
    }

    /// Enqueue on `stream` the reference unpack (`ds41_r8_q3k`, one thread an
    /// output word): `dst` = the Q3_K rows (rows of `nb` super-blocks) of
    /// `src`, the r8 row-lane layout. The gate holds the group unpack to it;
    /// the engine never runs it. Refused before anything is enqueued: a
    /// `src` that is not a whole number of 8-row groups, a shorter `dst`.
    pub fn enqueue_reference(
        &self,
        stream: &CudaStream,
        src: &DeviceBuffer<u32>,
        dst: &mut DeviceBuffer<u32>,
        nb: usize,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "R8Kernels::enqueue_reference";
        let (words, group_words) = (src.len(), R8_BLOCK_WORDS * nb);
        if nb == 0 || words == 0 || !words.is_multiple_of(group_words) || dst.len() < words {
            return Err(shape(
                WHAT,
                format!(
                    "{words} words of rows of {nb} super-blocks (groups of {group_words} \
                     words) into {}",
                    dst.len()
                ),
            ));
        }
        let grid = launch_u32(WHAT, "grid", words.div_ceil(UNPACK_THREADS as usize))?;
        let (nb32, words32) = (
            launch_u32(WHAT, "nb", nb)?,
            launch_u32(WHAT, "words", words)?,
        );
        let prep = self
            .module
            .prepare_ds41_r8_q3k(LaunchConfig1D::new(grid, UNPACK_THREADS, 0))?;
        self.module
            .ds41_r8_q3k(stream, &prep, src, nb32, words32, dst)?;
        Ok(())
    }
}

/// The unpack's shape of a gate or an up of `ff` rows of `n_embd`, else
/// refused by name: whole super-blocks of 256 a row, and gate and up parts
/// (`parts[0]`, `parts[1]`) of exactly those rows as Q3_K.
fn unpack_shape(n_embd: usize, ff: usize, parts: &[usize]) -> Result<(), GpuError> {
    let q3k = ff * (n_embd / 256) * Q3K_BLOCK;
    if n_embd.is_multiple_of(256) && parts.first() == Some(&q3k) && parts.get(1) == Some(&q3k) {
        return Ok(());
    }
    Err(shape(
        "Ds41Stacks::open",
        format!(
            "the unpack of gates and ups of {ff} rows of {n_embd}: whole super-blocks of 256 and \
             parts of {q3k} bytes, the stacks hold {parts:?}"
        ),
    ))
}

fn shape(what: &'static str, detail: impl Into<String>) -> GpuError {
    GpuError::Shape {
        what,
        detail: detail.into(),
    }
}

/// The K-quant types of V4.1's three stacks, in stack order: the Q3_K gate
/// and up and the Q4_K down its kernels read.
const TYPES: [GgmlType; 3] = [GgmlType::Q3_K, GgmlType::Q3_K, GgmlType::Q4_K];

/// V4.1's [`FileStacks`] for the common file source: each layer's gate, up
/// and down by name, and under the r8 sidecar the unpack of a staged gate or
/// up into its slot's Q3_K bytes ([`R8Unpack`]).
pub struct Ds41Stacks;

impl FileStacks for Ds41Stacks {
    fn names(&self, layer: usize) -> Vec<String> {
        vec![
            names::ffn_gate_exps(layer),
            names::ffn_up_exps(layer),
            names::ffn_down_exps(layer),
        ]
    }

    fn types(&self) -> &'static [GgmlType] {
        &TYPES
    }

    fn open(
        &self,
        dims: &[u64],
        parts: &[usize],
        sidecar: bool,
        ctx: &Arc<CudaContext>,
    ) -> Result<Option<Arc<dyn Convert>>, GpuError> {
        if !sidecar {
            return Ok(None);
        }
        let (n_embd, ff) = (dims[0] as usize, dims[1] as usize);
        unpack_shape(n_embd, ff, parts)?;
        let nb = n_embd / 256;
        let kernels = R8Kernels::load(ctx)?;
        Ok(Some(Arc::new(R8Unpack {
            kernels,
            parts: parts.to_vec(),
            nb,
        })))
    }
}

/// The r8 unpack as a [`Convert`]: a gate or an up staged from the sidecar
/// holds the r8 row-lane layout, unpacked in place into Q3_K
/// ([`R8Kernels::enqueue_in_place`]).
struct R8Unpack {
    kernels: R8Kernels,
    parts: Vec<usize>,
    /// Super-blocks of each row of a gate or up part.
    nb: usize,
}

impl Convert for R8Unpack {
    fn convert(
        &self,
        part: usize,
        dst: sys::CUdeviceptr,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        match part {
            0 | 1 => self
                .kernels
                .enqueue_in_place(stream, dst, self.parts[part] / 4, self.nb),
            _ => Ok(()),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` words of a xorshift stream from `seed`.
    fn words(n: usize, seed: u64) -> Vec<u32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 16) as u32
            })
            .collect()
    }

    /// One group of rows of `nb` super-blocks through the group unpack's
    /// tasks, each piece's bytes little-endian as `put` stores them, and how
    /// many pieces wrote each byte.
    fn by_tasks(group: &[u32], nb: usize) -> (Vec<u8>, Vec<u32>) {
        let bytes = R8_BLOCK * nb;
        let (mut out, mut writes) = (vec![0u8; bytes], vec![0u32; bytes]);
        for t in 0..R8_TASKS * nb {
            let (sb, r, f) = r8_task(t);
            let blk = q3k_block_at(r, sb, nb);
            // SAFETY: `group` holds the group's nb super-blocks; sb < nb,
            // r < 8, f < 8.
            for (o, w, n) in unsafe { r8_task_words(group.as_ptr(), sb, r, f) } {
                assert!(
                    o % 2 == 0 && matches!(n, 0 | 2 | 4),
                    "task {t}: piece at {o} of {n}"
                );
                for (k, b) in w.to_le_bytes()[..n].iter().enumerate() {
                    out[blk + o + k] = *b;
                    writes[blk + o + k] += 1;
                }
            }
        }
        (out, writes)
    }

    /// The same group through the reference unpack's per-byte rule
    /// (`q3k_byte`, what `ds41_r8_q3k` runs).
    fn by_bytes(group: &[u32], nb: usize) -> Vec<u8> {
        let row_bytes = Q3K_BLOCK * nb;
        (0..R8_BLOCK * nb)
            .map(|o| {
                let (r, x) = (o / row_bytes, o % row_bytes);
                // SAFETY: the group super-block at byte (x / 110) · 880
                // lies in `group`; r < 8, x % 110 < 110.
                unsafe { q3k_byte(group, (x / Q3K_BLOCK) * R8_BLOCK, r, x % Q3K_BLOCK) as u8 }
            })
            .collect()
    }

    #[test]
    fn the_unpack_shape_is_whole_q3k_rows() {
        let (n_embd, ff) = (4096, 2048);
        let q3k = ff * (n_embd / 256) * Q3K_BLOCK;
        unpack_shape(n_embd, ff, &[q3k, q3k, 7]).expect("whole Q3_K rows");
        for (n, parts, why) in [
            (n_embd + 128, [q3k, q3k, 7], "a row past whole super-blocks"),
            (n_embd, [q3k + 4, q3k, 7], "a gate part of another size"),
            (n_embd, [q3k, q3k - 4, 7], "an up part of another size"),
        ] {
            assert!(unpack_shape(n, ff, &parts).is_err(), "{why} is refused");
        }
    }

    #[test]
    fn tasks_tile_the_group() {
        for nb in 1..=R8_NB_MAX {
            let (_, writes) = by_tasks(&vec![0; R8_BLOCK_WORDS * nb], nb);
            if let Some(o) = writes.iter().position(|&n| n != 1) {
                panic!("nb {nb}: byte {o} of the group written {} times", writes[o]);
            }
        }
    }

    #[test]
    fn tasks_equal_the_reference_bytes() {
        for nb in 1..=R8_NB_MAX {
            for seed in 0..4u64 {
                let group = words(
                    R8_BLOCK_WORDS * nb,
                    0x9E37_79B9_7F4A_7C15 ^ (seed << 8) ^ nb as u64,
                );
                let (got, _) = by_tasks(&group, nb);
                let want = by_bytes(&group, nb);
                if let Some(o) = (0..got.len()).find(|&o| got[o] != want[o]) {
                    panic!(
                        "nb {nb} seed {seed}: byte {o} (row {}, block byte {}) is {:#04x}, the \
                         reference's {:#04x}",
                        o / (Q3K_BLOCK * nb),
                        o % Q3K_BLOCK,
                        got[o],
                        want[o]
                    );
                }
            }
        }
    }
}
