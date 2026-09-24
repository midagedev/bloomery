//! The bf16 tensor-core GEMM: `C[M, N] = A[M, K] · B[N, K]ᵀ (+ bias[N])`, bf16 inputs, f32
//! accumulation on `mma.sync.m16n8k16`, bf16 output, with the epilogues the encoder's linear
//! layers need. `B` is a linear layer's weight as the file stores it (one row per output), so no
//! operand is transposed on the host.
//!
//! Tiling: one block of [`THREADS`] threads computes a [`TILE_M`] × [`TILE_N`] tile of `C`; its four
//! warps each own a 32 × 32 quarter (2 m16 × 4 n8 fragments, 32 f32 accumulators per lane). The K
//! axis is walked in [`TILE_K`]-value steps: the block stages the A and B tiles into shared memory
//! as u32 words (two bf16 each), rows padded to [`ROW_WORDS`] words so each `ldmatrix` phase reads
//! eight rows across all 32 banks, then each warp loads its fragments with `ldmatrix.x4` and
//! issues two k16 steps. Values past `K` (a `K` that is not a multiple of [`TILE_K`], the patch
//! embedding's 588) and rows past `M` stage as zero, which adds nothing to any sum.
//!
//! Numeric rule, per output `(i, j)`:
//! 1. `acc = Σ_k A[i,k]·B[j,k]` in f32: the products of two bf16 are exact in f32; the sum runs
//!    in `K/16` tensor-core steps, ascending `k`, each adding a 16-product partial into the f32
//!    accumulator. The instruction's order inside a partial is the hardware's — so the host rule
//!    ([`gemm_ref`]) takes the exact sum (f64) and the gate bounds the kernel against it by the
//!    accumulation-order bound [`order_bound`].
//! 2. `v = acc + bias[j]` (one f32 add) when a bias is given.
//! 3. `y = bf16(v)`, round to nearest even — the output rounding of torch's bf16 linear.
//! 4. Epilogue on `y`, each op rounded on its own as torch's separate kernel rounds it:
//!    [`Epilogue::Gelu`] `y = bf16(gelu_erf(y))`; [`Epilogue::Residual`] `y = bf16(y + r[i,j])`
//!    with `r` a bf16 tensor of `C`'s shape (the block's residual stream).
//!
//! Step 4 applied to the bf16 `y` of step 3 is what makes an epilogue exact: a GEMM with an
//! epilogue returns bit for bit the epilogue of the same GEMM without it ([`epilogue_ref`] on
//! the no-epilogue output), which the gate checks.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, launch_bounds, launch_contract};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Output rows one block computes.
pub const TILE_M: usize = 64;
/// Output columns one block computes; `N` must be a multiple of it.
pub const TILE_N: usize = 64;
/// K values one staging step covers: two `mma.sync` k16 steps.
pub const TILE_K: usize = 32;
/// Threads per block: four warps in a 2 × 2 grid of 32 × 32 quarters.
pub const THREADS: usize = 128;
const THREADS_U32: u32 = THREADS as u32;
/// u32 words per staged tile row: the [`TILE_K`] values as bf16 pairs, then four words of pad.
/// The row stride is an odd multiple of 16 bytes, so the eight 16-byte rows one `ldmatrix`
/// phase reads land in eight distinct bank quads.
pub const ROW_WORDS: usize = TILE_K / 2 + 4;
/// Words of one staged tile (A or B).
const TILE_WORDS: usize = TILE_M * ROW_WORDS;
/// Words each thread stages per tile per step.
const STAGE_PER_THREAD: usize = TILE_M * (TILE_K / 2) / THREADS;

const _: () = assert!(TILE_M == TILE_N);
const _: () = assert!(TILE_K == 32 && TILE_M == 64 && THREADS == 128);
const _: () = assert!((ROW_WORDS * 4).is_multiple_of(16) && ((ROW_WORDS * 4) / 16) % 2 == 1);
const _: () = assert!(STAGE_PER_THREAD * THREADS == TILE_M * (TILE_K / 2));

/// `M_SQRT1_2` as the f32 torch's GELU multiplies by.
const SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// What happens to the bf16 output after the rounding of the product (step 4 of the rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// `C = bf16(A·Bᵀ + bias)`.
    None,
    /// `C = bf16(gelu_erf(bf16(A·Bᵀ + bias)))` — the aligner's `F.gelu(w1(x))`.
    Gelu,
    /// `C = bf16(bf16(A·Bᵀ + bias) + R)` — a block's `x + branch(x)`.
    Residual,
}

// ----------------------------------------------------------------- device math

#[device]
unsafe extern "C" {
    /// libdevice's `erff`, the function torch's CUDA GELU calls (`c10::cuda::compat::erf` on
    /// float). Rust has no `erf` intrinsic to reach it through.
    fn __nv_erff(x: f32) -> f32;
}

/// torch's exact GELU on one f32, in its op order: `x · 0.5 · (1 + erf(x · √½))`, every op
/// rounded on its own (the explicit `_rn` forms are never contracted).
#[inline(always)]
fn gelu_erf_dev(x: f32) -> f32 {
    // SAFETY: `__nv_erff` is a pure libdevice function defined for every f32.
    let e = unsafe { __nv_erff(mul_rn_f32(x, SQRT1_2)) };
    mul_rn_f32(mul_rn_f32(x, 0.5), add_rn_f32(1.0, e))
}

// ---------------------------------------------------------------- kernels

#[cuda_module]
mod gemm_kernels {
    use super::*;

    /// `C = A·Bᵀ` with the module rule's bias and epilogue: `a` is `m` rows of `k` bf16, `b` is
    /// `n` rows of `k` bf16, `c` (and `resid`, when read) `m` rows of `n` bf16, all row-major.
    /// `use_bias` 1 adds `bias[j]` (f32) before the rounding; `gelu` 1 and `use_resid` 1 select
    /// the epilogues (the launcher passes at most one). Block `b` computes tile row
    /// `b / (n / TILE_N)`, tile column `b % (n / TILE_N)`. `n` a multiple of [`TILE_N`] and `k`
    /// even and non-zero (host-checked).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            a.len() >= m * k,
            b.len() >= n * k,
            bias.len() >= n * use_bias,
            resid.len() >= m * n * use_resid,
            c.len() >= m * n
        )
    )]
    pub fn vis_gemm_bf16(
        a: &[u16],
        b: &[u16],
        bias: &[f32],
        resid: &[u16],
        m: u32,
        n: u32,
        k: u32,
        use_bias: u32,
        gelu: u32,
        use_resid: u32,
        mut c: DisjointSlice<u16>,
    ) {
        static mut AS: SharedArray<u32, TILE_WORDS> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, TILE_WORDS> = SharedArray::UNINIT;

        let tid = cuda_device::thread::threadIdx_x() as usize;
        let blk = cuda_device::thread::blockIdx_x() as usize;
        let rows = m as usize;
        let cols = n as usize;
        let kw = (k / 2) as usize; // u32 words per operand row
        let tiles_n = cols / TILE_N;
        let tm = blk / tiles_n;
        let tn = blk - tm * tiles_n;
        if tm * TILE_M >= rows {
            return; // block-uniform: no barrier and no warp collective is skipped
        }
        // SAFETY: block-shared, TILE_WORDS words each, written only by the staging loop between
        // the barriers that publish it.
        let (as_, bs_) = unsafe {
            (
                SharedArray::as_raw_mut_ptr(&raw mut AS),
                SharedArray::as_raw_mut_ptr(&raw mut BS),
            )
        };
        let aw = a.as_ptr().cast::<u32>();
        let bw = b.as_ptr().cast::<u32>();

        let lane = cuda_device::warp::lane_id() as usize;
        let wid = tid / 32;
        let wm = wid / 2; // this warp's 32-row half of the tile
        let wn = wid % 2; // and its 32-column half
        // `ldmatrix.x4` lane roles. A: row `lane % 16` of an m16 fragment at k half `lane / 16`.
        // B: row `(lane % 8) + 8 * (lane / 16)` of an n16 pair at k half `(lane / 8) % 2`.
        let a_row = lane % 16;
        let a_half = lane / 16;
        let b_row = (lane % 8) + 8 * (lane / 16);
        let b_half = (lane / 8) % 2;

        let mut acc = [[0.0f32; 4]; 8];

        let k_steps = kw.div_ceil(TILE_K / 2);
        let mut ks = 0usize;
        while ks < k_steps {
            let w0 = ks * (TILE_K / 2);
            // The previous step's fragment reads are done before this step's staging writes.
            cuda_device::thread::sync_threads();
            let mut ra = [0u32; STAGE_PER_THREAD];
            let mut rb = [0u32; STAGE_PER_THREAD];
            let mut s = 0usize;
            while s < STAGE_PER_THREAD {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (TILE_K / 2);
                let w = i - r * (TILE_K / 2);
                let gw = w0 + w;
                let ar = tm * TILE_M + r;
                if ar < rows && gw < kw {
                    // SAFETY: ar < m and gw < k / 2, so the word is inside row `ar` of `a`,
                    // whose m·k bf16 the launch contract bounds; device buffers are 4-byte
                    // aligned and k is even, so the word read is aligned.
                    ra[s] = unsafe { *aw.add(ar * kw + gw) };
                }
                let br = tn * TILE_N + r;
                if gw < kw {
                    // SAFETY: br < n (tiles cover n exactly) and gw < k / 2 bound the word
                    // inside `b`'s n·k bf16 (launch contract), aligned as above.
                    rb[s] = unsafe { *bw.add(br * kw + gw) };
                }
                s += 1;
            }
            let mut s = 0usize;
            while s < STAGE_PER_THREAD {
                cuda_device::thread::__unroll_config::<0>();
                let i = tid + s * THREADS;
                let r = i / (TILE_K / 2);
                let w = i - r * (TILE_K / 2);
                // SAFETY: r < TILE_M and w < TILE_K / 2 < ROW_WORDS bound both stores inside
                // their tile; each (r, w) has one owner thread.
                unsafe {
                    *as_.add(r * ROW_WORDS + w) = ra[s];
                    *bs_.add(r * ROW_WORDS + w) = rb[s];
                }
                s += 1;
            }
            cuda_device::thread::sync_threads();

            let mut kk = 0usize;
            while kk < TILE_K / 16 {
                cuda_device::thread::__unroll_config::<0>();
                let mut af = [[0u32; 4]; 2];
                let mut mi = 0usize;
                while mi < 2 {
                    cuda_device::thread::__unroll_config::<0>();
                    let row = wm * 32 + mi * 16 + a_row;
                    // SAFETY: row < TILE_M and word kk·8 + a_half·4 + 4 <= TILE_K / 2 keep the
                    // 16-byte row inside the A tile, published by the barrier above.
                    let p = unsafe { as_.add(row * ROW_WORDS + kk * 8 + a_half * 4) };
                    // SAFETY: every lane of the warp reaches this load with the same
                    // qualifiers and an aligned address inside the A tile.
                    af[mi] = unsafe {
                        cuda_device::wmma::ldmatrix_x4_shared_u32(
                            cuda_device::shared::cvta_generic_to_shared_u32(
                                p.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    mi += 1;
                }
                let mut nj = 0usize;
                while nj < 2 {
                    cuda_device::thread::__unroll_config::<0>();
                    let row = wn * 32 + nj * 16 + b_row;
                    // SAFETY: row < TILE_N and the same word bound keep the row inside the B
                    // tile, published by the barrier above.
                    let p = unsafe { bs_.add(row * ROW_WORDS + kk * 8 + b_half * 4) };
                    // SAFETY: every lane of the warp reaches this load with the same
                    // qualifiers and an aligned address inside the B tile.
                    let bf = unsafe {
                        cuda_device::wmma::ldmatrix_x4_shared_u32(
                            cuda_device::shared::cvta_generic_to_shared_u32(
                                p.cast_const().cast::<u8>(),
                            ),
                        )
                    };
                    let mut mi = 0usize;
                    while mi < 2 {
                        cuda_device::thread::__unroll_config::<0>();
                        // SAFETY: the whole warp issues these `mma.sync` (no lane-dependent
                        // branch encloses them) with the fragments it loaded.
                        unsafe {
                            acc[mi * 4 + nj * 2] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                                acc[mi * 4 + nj * 2],
                                af[mi],
                                [bf[0], bf[1]],
                            );
                            acc[mi * 4 + nj * 2 + 1] = cuda_device::wmma::mma_m16n8k16_f32_bf16(
                                acc[mi * 4 + nj * 2 + 1],
                                af[mi],
                                [bf[2], bf[3]],
                            );
                        }
                        mi += 1;
                    }
                    nj += 1;
                }
                kk += 1;
            }
            ks += 1;
        }

        // Epilogue: accumulator register j of fragment (mi, ni) is row `group + 8·(j / 2)`,
        // column `2·(lane % 4) + j % 2` of that fragment.
        let group = lane / 4;
        let t4 = lane % 4;
        let mut f = 0usize;
        while f < 8 {
            cuda_device::thread::__unroll_config::<0>();
            let mi = f / 4;
            let ni = f % 4;
            let mut j = 0usize;
            while j < 4 {
                cuda_device::thread::__unroll_config::<0>();
                let row = tm * TILE_M + wm * 32 + mi * 16 + group + 8 * (j / 2);
                let col = tn * TILE_N + wn * 32 + ni * 8 + 2 * t4 + (j % 2);
                if row < rows {
                    let mut v = acc[f][j];
                    if use_bias != 0 {
                        // SAFETY: col < n and use_bias = 1, so bias.len() >= n (contract).
                        v = add_rn_f32(v, unsafe { *bias.get_unchecked(col) });
                    }
                    let mut y = f32_to_bf16_rne(v);
                    if gelu != 0 {
                        y = f32_to_bf16_rne(gelu_erf_dev(bf16_to_f32(y)));
                    }
                    if use_resid != 0 {
                        // SAFETY: row < m, col < n and use_resid = 1, so the index is below
                        // m·n <= resid.len() (contract).
                        let r = unsafe { *resid.get_unchecked(row * cols + col) };
                        y = f32_to_bf16_rne(add_rn_f32(bf16_to_f32(y), bf16_to_f32(r)));
                    }
                    // SAFETY: row < m and col < n bound the index below m·n <= c.len(); each
                    // (row, col) of the tile has exactly one owner lane.
                    unsafe {
                        *c.get_unchecked_mut(row * cols + col) = y;
                    }
                }
                j += 1;
            }
            f += 1;
        }
    }
}

// -------------------------------------------------------------- launchers

/// One GEMM launch: `c = a · bᵀ` over `m` rows, `n` columns and `k` values, with an optional f32
/// bias of `n` and the epilogue (`Residual` reads `resid`, `m · n` bf16).
pub struct GemmArgs<'a> {
    pub a: &'a DeviceBuffer<u16>,
    pub b: &'a DeviceBuffer<u16>,
    pub bias: Option<&'a DeviceBuffer<f32>>,
    pub epilogue: Epilogue,
    pub resid: Option<&'a DeviceBuffer<u16>>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub c: &'a mut DeviceBuffer<u16>,
}

/// The loaded GEMM module. Owns no stream: each enqueue takes the caller's stream.
pub struct GemmKernels {
    module: gemm_kernels::LoadedModule,
    /// A one-element stand-in for the bias and residual operands a launch does not read: the
    /// contract asks their length only times the flag that selects them.
    dummy_f32: DeviceBuffer<f32>,
    dummy_u16: DeviceBuffer<u16>,
}

impl GemmKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>, stream: &CudaStream) -> Result<GemmKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { gemm_kernels::load(ctx)? };
        Ok(GemmKernels {
            module,
            dummy_f32: DeviceBuffer::zeroed(stream, 1)?,
            dummy_u16: DeviceBuffer::zeroed(stream, 1)?,
        })
    }

    /// Enqueue one GEMM ([`GemmArgs`]). `n` a multiple of [`TILE_N`], `k` even and non-zero, `m`
    /// non-zero; a `Residual` epilogue needs `resid`, and a `resid` without it is refused.
    /// Asynchronous, allocation-free.
    pub fn enqueue(&self, stream: &CudaStream, args: GemmArgs<'_>) -> Result<(), GpuError> {
        let what = "GemmKernels::enqueue";
        let GemmArgs {
            a,
            b,
            bias,
            epilogue,
            resid,
            m,
            n,
            k,
            c,
        } = args;
        if m == 0 || n == 0 || !n.is_multiple_of(TILE_N) || k == 0 || !k.is_multiple_of(2) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "need m >= 1, n a positive multiple of {TILE_N}, k even and >= 2; got m={m} n={n} k={k}"
                ),
            });
        }
        let short = |name: &str, len: usize, need: usize| {
            (len < need).then(|| format!("{name}.len() {len} < {need}"))
        };
        let problems: Vec<String> = [
            short("a", a.len(), m * k),
            short("b", b.len(), n * k),
            short("c", c.len(), m * n),
            bias.and_then(|x| short("bias", x.len(), n)),
            resid.and_then(|x| short("resid", x.len(), m * n)),
        ]
        .into_iter()
        .flatten()
        .collect();
        if !problems.is_empty() {
            return Err(GpuError::Shape {
                what,
                detail: problems.join(", "),
            });
        }
        let (gelu, resid) = match (epilogue, resid) {
            (Epilogue::Residual, Some(r)) => (0, Some(r)),
            (Epilogue::Residual, None) => {
                return Err(GpuError::Shape {
                    what,
                    detail: "a Residual epilogue needs `resid`".into(),
                });
            }
            (_, Some(_)) => {
                return Err(GpuError::Shape {
                    what,
                    detail: format!("`resid` given to a {epilogue:?} epilogue, which reads none"),
                });
            }
            (Epilogue::Gelu, None) => (1, None),
            (Epilogue::None, None) => (0, None),
        };
        let grid = launch_u32(what, "grid", m.div_ceil(TILE_M) * (n / TILE_N))?;
        let prep = self
            .module
            .prepare_vis_gemm_bf16(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.vis_gemm_bf16(
            stream,
            &prep,
            a,
            b,
            bias.unwrap_or(&self.dummy_f32),
            resid.unwrap_or(&self.dummy_u16),
            launch_u32(what, "m", m)?,
            launch_u32(what, "n", n)?,
            launch_u32(what, "k", k)?,
            u32::from(bias.is_some()),
            gelu,
            u32::from(resid.is_some()),
            c,
        )?;
        Ok(())
    }
}

// ------------------------------------------------------------ host rules

/// torch's exact GELU of one f32 on the host: `x · 0.5 · (1 + erf(x · √½))`, each op in f32 as
/// the kernel rounds it, `erf` from [`erf_f64`] rounded to f32 (the device's `erff` differs from
/// that by at most a few ulps, which the bf16 rounding after it almost always absorbs).
#[must_use]
pub fn gelu_erf(x: f32) -> f32 {
    let e = erf_f64(f64::from(x * SQRT1_2)) as f32;
    (x * 0.5) * (1.0 + e)
}

/// `erf` in f64: the Maclaurin series below 2.5, the continued fraction of `erfc` above. Both
/// converge to well under an f32 ulp over the whole range an f32 `erf` resolves (|x| < 4; past
/// it `erf` is ±1 in f32).
#[must_use]
pub fn erf_f64(x: f64) -> f64 {
    let a = x.abs();
    let v = if a == 0.0 {
        0.0
    } else if a < 2.5 {
        // erf(a) = 2/√π · Σ (-1)^n a^(2n+1) / (n! (2n+1))
        let a2 = a * a;
        let mut term = a; // (-1)^n a^(2n+1) / n!
        let mut sum = a;
        let mut n = 0.0f64;
        loop {
            n += 1.0;
            term *= -a2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-18 * sum.abs() {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else if a < 6.0 {
        // erfc(a) = exp(-a²)/√π · 1/(a + (1/2)/(a + 1/(a + (3/2)/(a + …)))), evaluated from 60 down.
        let mut t = a;
        let mut i = 60.0f64;
        while i >= 1.0 {
            t = a + (i / 2.0) / t;
            i -= 1.0;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * t)
    } else {
        1.0
    };
    v.copysign(x)
}

/// Step 4 of the rule on the host: the epilogue applied to a no-epilogue output `y` (bf16
/// bits), with `r` the residual value when the epilogue reads one.
#[must_use]
pub fn epilogue_ref(y: u16, epilogue: Epilogue, r: u16) -> u16 {
    match epilogue {
        Epilogue::None => y,
        Epilogue::Gelu => crate::f32_bf16(gelu_erf(crate::bf16_f32(y))),
        Epilogue::Residual => crate::f32_bf16(crate::bf16_f32(y) + crate::bf16_f32(r)),
    }
}

/// The exact value of output `(i, j)` before the rounding: `Σ_k a[i,k]·b[j,k] + bias[j]` in f64
/// (the products are exact and a sum of at most a few thousand of them loses nothing an f32
/// result can see), and `Σ_k |a[i,k]·b[j,k]| + |bias[j]|`, the magnitude [`order_bound`] scales.
#[must_use]
pub fn gemm_ref(
    a: &[u16],
    b: &[u16],
    bias: Option<&[f32]>,
    k: usize,
    i: usize,
    j: usize,
) -> (f64, f64) {
    let (ar, br) = (&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k]);
    let (mut s, mut mag) = (0.0f64, 0.0f64);
    for (&x, &w) in ar.iter().zip(br) {
        let p = f64::from(crate::bf16_f32(x)) * f64::from(crate::bf16_f32(w));
        s += p;
        mag += p.abs();
    }
    if let Some(bias) = bias {
        let v = f64::from(bias[j]);
        s += v;
        mag += v.abs();
    }
    (s, mag)
}

/// The widest distance an f32 accumulation of `k` products (and one bias add) can put between
/// its result and the exact sum, per unit of `Σ|terms|`: `(k + 1) · 2⁻²³` — one f32 rounding of
/// up to a full ulp (the tensor core's internal truncation included) per term. The standard
/// `γ_n` bound, doubled for truncation; it holds for every summation order, so it holds for the
/// kernel's.
#[must_use]
pub fn order_bound(k: usize) -> f64 {
    (k as f64 + 1.0) * f64::powi(2.0, -23)
}

/// Whether bf16 `got` is a rounding of some value within `bound` of `exact`: the bf16 values
/// `round(exact − bound)` through `round(exact + bound)` are the outputs an order-only difference
/// can produce.
#[must_use]
pub fn within_order(got: u16, exact: f64, bound: f64) -> bool {
    let lo = crate::bf16_f32(round_f64_bf16(exact - bound));
    let hi = crate::bf16_f32(round_f64_bf16(exact + bound));
    let g = crate::bf16_f32(got);
    lo <= g && g <= hi
}

/// f64 to bf16 bits through f32 (two roundings; the f32 step is exact for every value these
/// bounds produce to within the order bound itself, which is far wider than one f32 ulp).
fn round_f64_bf16(x: f64) -> u16 {
    crate::f32_bf16(x as f32)
}
