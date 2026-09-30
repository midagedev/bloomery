//! Write views for device kernels: the launch facts a kernel's accesses
//! lean on, stored once at entry, so the accesses themselves are safe code.
//!
//! A kernel's bounds live in its `#[launch_contract]`: the generated safe
//! launcher evaluates every `requires` relation on the host before it
//! enqueues (cuda-macros `contract.rs`, `generate_requires_checks` — only
//! the `*_unchecked` launchers skip them, and they are `unsafe fn`), and
//! `prepare_*` refuses a block shape other than the contract's exact one
//! (cuda-core `BlockRequirement::Exact`). What no launcher can check is
//! grid-uniformity — that a shape argument holds the same value in every
//! thread of the grid. Kernel parameters read unmodified do; values safe
//! device code computes per block do not, which is why no type can make a
//! per-block-shaped write row sound on its own. So each kernel's view set
//! has one `unsafe fn` constructor, called once at kernel entry, whose
//! `# Safety` section names exactly three things: the `requires` clauses
//! the launcher already checked, the grid-uniformity the kernel's
//! parameters give, and the block shape the contract pins. Everything
//! after that entry is safe code, and the accessors take no caller index:
//! the strided pass walks its own counter, whose `it < end` compare is the
//! kernel's own loop condition, and a rope pair is minted from the
//! thread's own [`ThreadIndex`] — a hardware coordinate safe code cannot
//! forge. The views read no hardware state of their own.
//!
//! [`ThreadIndex`]: cuda_device::thread::ThreadIndex
//!
//! The shared-memory reduce stays outside this file: "every slot was
//! written before the barrier" is a fact about other threads' control
//! flow, which no per-thread type can carry, so those accesses keep their
//! `unsafe` and their `// SAFETY:` where they stand.

use core::marker::PhantomData;

use cuda_device::DisjointSlice;
use cuda_device::thread::{Index1D, ThreadIndex};

/// The block-strided apply pass of a norm-shaped kernel: `gain` is the
/// row-global read (one value per column of the row), `x` and `y` are the
/// calling block's own row at `base`, and the pass strides `step` from the
/// calling thread's own index up to the row's `k` values.
/// [`next_cell`](Apply::next_cell) is the accessor; it takes no index.
#[must_use]
pub(crate) struct Apply<'g, 'x, 's, 'a, T> {
    gain: *const T,
    x: *const T,
    y: *mut T,
    /// The row `x` and `y` are viewed at: both address `base + it`, `it`
    /// running below `end`.
    base: usize,
    it: usize,
    step: usize,
    end: usize,
    _gain: PhantomData<&'g [T]>,
    _x: PhantomData<&'x [T]>,
    _y: PhantomData<&'s mut DisjointSlice<'a, T>>,
}

impl<'g, 'x, 's, 'a, T> Apply<'g, 'x, 's, 'a, T> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this pass that means `gain.len() >= k`,
    ///     `x.len() >= k * m` and `y.len() >= k * m` for the `k` and `m`
    ///     this launch passed.
    /// (b) `base`, `k` and `step` are values every thread of the grid
    ///     computes identically from the kernel's parameters (`k` the
    ///     parameter itself; `base = blockIdx * k` with `blockIdx < m`, the
    ///     kernel's block-uniform token guard having returned already), and
    ///     `it` is the calling thread's own `threadIdx_x`.
    /// (c) The launch's block is the contract's exact 1-D `(step, 1, 1)`
    ///     (`BlockRequirement::Exact`, cuda-core — `prepare_*` refuses any
    ///     other shape), so `it < step`.
    #[inline(always)]
    pub(crate) unsafe fn new(
        gain: &'g [T],
        x: &'x [T],
        y: &'s mut DisjointSlice<'a, T>,
        base: usize,
        k: usize,
        it: usize,
        step: usize,
    ) -> Apply<'g, 'x, 's, 'a, T> {
        Apply {
            gain: gain.as_ptr(),
            x: x.as_ptr(),
            y: y.as_mut_ptr(),
            base,
            it,
            step,
            end: k,
            _gain: PhantomData,
            _x: PhantomData,
            _y: PhantomData,
        }
    }

    /// The next `(gain[it], x[base + it], &mut y[base + it])` of the pass,
    /// `None` once the row is exhausted. The `it < end` compare is the loop
    /// condition the kernel's own pass had; no other check runs per step.
    ///
    /// Sound for every `Apply` safe code can name: an `Apply` can only be
    /// built by [`new`](Self::new), and the state that constructor fixes —
    /// under its contract (a) `it < end` stays inside `gain`, (b)
    /// `base + it` (with `it < end = k`, `base` a row start below `m * k`)
    /// stays inside `x` and `y`, and (c) this thread's cells are its alone:
    /// no other thread of the block shares its `it` (distinct `threadIdx_x`
    /// below `step`, one residue class of the stride each) and no other
    /// block's pass names this row (`base` uniform, one row per block). The
    /// `&mut` borrows this `Apply`, so one cell reference is live per thread
    /// at a time. The counter cannot wrap: `it < end` and `it = it + step`
    /// stay below `end + step`, and `end <= len` of a real buffer leaves
    /// that far under `usize::MAX`.
    #[inline(always)]
    pub(crate) fn next_cell(&mut self) -> Option<(T, T, &mut T)>
    where
        T: Copy,
    {
        let it = self.it;
        if it < self.end {
            self.it = it + self.step;
            // SAFETY: `it < end`, and the constructing `new` proved every
            // buffer covers `end` values from this row's `base`.
            Some(unsafe {
                (
                    *self.gain.add(it),
                    *self.x.add(self.base + it),
                    &mut *self.y.add(self.base + it),
                )
            })
        } else {
            None
        }
    }
}

/// The rope launch's three buffers as one shape: `src` and `dst` hold
/// `m * n_vec * nd` values in the `[n_dims, n_vec, m]` column layout, `cs`
/// holds `m * nd` cache values, one thread per (column, pair) of the
/// `npairs = nd / 2` pairs each column covers. [`own`](Rope::own) mints
/// the calling thread's pair from its own launch index; it takes no caller
/// index.
#[must_use]
pub(crate) struct Rope<'b, 's, 'a, T> {
    src: *const T,
    cs: *const T,
    dst: *mut T,
    n_vec: usize,
    nd: usize,
    npairs: usize,
    total: usize,
    _reads: PhantomData<&'b [T]>,
    _dst: PhantomData<&'s mut DisjointSlice<'a, T>>,
}

impl<'b, 's, 'a, T> Rope<'b, 's, 'a, T> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch — the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For rope that means
    ///     `src.len() >= m * n_vec * nd`, `cs.len() >= m * nd` and
    ///     `dst.len() >= m * n_vec * nd` for the shape this launch passed.
    /// (b) `nd`, `n_vec` and `m` are the kernel's parameters read
    ///     unmodified — every thread of the grid holds the same values, and
    ///     the host refused the empty and odd shapes (`n_vec >= 1`, `m >= 1`,
    ///     `nd` even and at least 2) — so the flat-index decomposition below
    ///     is the same map in every thread, which is what makes two
    ///     threads' pairs distinct cells.
    /// (c) The launch is the contract's 1-D domain with its exact block, so
    ///     the [`ThreadIndex`] a thread carries is unique in the launch
    ///     (cuda-device's index-space invariant).
    ///
    /// [`ThreadIndex`]: cuda_device::thread::ThreadIndex
    #[inline(always)]
    pub(crate) unsafe fn new(
        src: &'b [T],
        cs: &'b [T],
        dst: &'s mut DisjointSlice<'a, T>,
        nd: u32,
        n_vec: u32,
        m: u32,
    ) -> Rope<'b, 's, 'a, T> {
        let npairs = (nd >> 1) as usize;
        let total = m as usize * n_vec as usize * npairs;
        Rope {
            src: src.as_ptr(),
            cs: cs.as_ptr(),
            dst: dst.as_mut_ptr(),
            n_vec: n_vec as usize,
            nd: nd as usize,
            npairs,
            total,
            _reads: PhantomData,
            _dst: PhantomData,
        }
    }

    /// The pair this launch thread owns, `None` for a thread past the last
    /// pair — the kernel's own `i >= total` guard, same polarity.
    ///
    /// Sound for every `Rope` safe code can name: a `Rope` can only be
    /// built by [`new`](Self::new), whose contract proved `src` and `dst`
    /// cover `m * n_vec * nd` values and `cs` `m * nd`. A
    /// [`ThreadIndex`] cannot be forged by safe code — only the launch
    /// machinery mints it from hardware coordinates — and it is unique per
    /// thread of the launch, so two threads' `own` calls decompose
    /// different flat indices: `i ↦ (col, p)` is a bijection onto
    /// `[0, m * n_vec) × [0, npairs)`, and `(col, p) ↦ col·nd + 2p` is
    /// injective (`2p < nd`), so the pairs' cells — and their `+1`s — are
    /// disjoint writes inside `dst`, and the read offsets stay below the
    /// extents the contract proved (`2p + 1 <= nd - 1` for any `nd >= 2`).
    /// The pair borrows this `Rope` exclusively, so one is live per thread.
    ///
    /// [`ThreadIndex`]: cuda_device::thread::ThreadIndex
    #[inline(always)]
    pub(crate) fn own(
        &mut self,
        idx: ThreadIndex<'_, Index1D>,
    ) -> Option<RopePair<'_, 'b, 's, 'a, T>> {
        let i = idx.get();
        let (npairs, n_vec, nd) = (self.npairs, self.n_vec, self.nd);
        if i >= self.total {
            return None;
        }
        let col = i / npairs;
        let d = (i % npairs) * 2;
        let t = col / n_vec;
        Some(RopePair {
            base: col * nd + d,
            cache: t * nd + d,
            rope: self,
        })
    }
}

/// One launch thread's rope pair: the two `src` values and the two cache
/// values to read, the two `dst` cells to write — offsets resolved when
/// [`Rope::own`] minted the pair. Built only there.
///
/// [`Rope::own`]: Rope::own
#[must_use]
pub(crate) struct RopePair<'v, 'b, 's, 'a, T> {
    rope: &'v mut Rope<'b, 's, 'a, T>,
    /// The pair's first cell in `src`/`dst` (`col * nd + 2p`); `+ 1` is the
    /// second. Both below `m * n_vec * nd` by `own`'s decomposition.
    base: usize,
    /// The pair's first cache cell (`t * nd + 2p`); `+ 1` is the second.
    /// Both below `m * nd`.
    cache: usize,
}

impl<'v, 'b, 's, 'a, T> RopePair<'v, 'b, 's, 'a, T> {
    /// The pair's four inputs: `(src0, src1, cos, sin)`.
    ///
    /// Sound: the offsets were resolved by the minting [`Rope::own`] from
    /// this thread's own launch index below `total`, a decomposition the
    /// `Rope`'s constructor contract keeps below the extents it proved, so
    /// all four loads are inside their buffers.
    ///
    /// [`Rope::own`]: Rope::own
    #[inline(always)]
    pub(crate) fn load(&self) -> (T, T, T, T)
    where
        T: Copy,
    {
        // SAFETY: `base` and `cache` were proven in bounds by the minting
        // `own` from a launch index below `total`; `+ 1` is part of that
        // proof.
        unsafe {
            (
                *self.rope.src.add(self.base),
                *self.rope.src.add(self.base + 1),
                *self.rope.cs.add(self.cache),
                *self.rope.cs.add(self.cache + 1),
            )
        }
    }

    /// Write the pair's two outputs to `dst`.
    ///
    /// Sound: the cells were proven inside `dst` at mint, and the minting
    /// index is unique per thread of the launch, so no other thread's pair
    /// resolves to either cell; the pair holds its `Rope` exclusively, so
    /// at most one pair per thread is live.
    #[inline(always)]
    pub(crate) fn store(&mut self, y0: T, y1: T) {
        // SAFETY: `base` and `base + 1` were proven inside `dst` by the
        // minting `own`, and this thread alone owns the pair's cells.
        unsafe {
            *self.rope.dst.add(self.base) = y0;
            *self.rope.dst.add(self.base + 1) = y1;
        }
    }
}
