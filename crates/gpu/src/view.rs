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
/// [`map`](Apply::map) runs the pass; it takes no index.
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

    /// Run the pass: `y[base + it] = f(gain[it], x[base + it])` for every
    /// `it` of this thread's stride below the row's `k` values. The loop is
    /// the kernel's own (`it < end`, then `it += step`); no other check runs
    /// per step. Taking `f` rather than handing out `Option<&mut T>` keeps
    /// the loop free of a per-step `Some`/`None` test and keeps the store
    /// addressed from `y` itself, so the backend keeps its global state space.
    ///
    /// Sound for every `Apply` safe code can name: an `Apply` can only be
    /// built by [`new`](Self::new), and the state that constructor fixes —
    /// under its contract (a) `it < end` stays inside `gain`, (b)
    /// `base + it` (with `it < end = k`, `base` a row start below `m * k`)
    /// stays inside `x` and `y`, and (c) this thread's cells are its alone:
    /// no other thread of the block shares its `it` (distinct `threadIdx_x`
    /// below `step`, one residue class of the stride each) and no other
    /// block's pass names this row (`base` uniform, one row per block). `f`
    /// receives values, never a reference into a buffer, and the pass
    /// consumes the `Apply`, so it runs once. The counter cannot wrap:
    /// `it < end` and `it = it + step` stay below `end + step`, and
    /// `end <= len` of a real buffer leaves that far under `usize::MAX`.
    #[inline(always)]
    pub(crate) fn map(self, mut f: impl FnMut(T, T) -> T)
    where
        T: Copy,
    {
        let mut it = self.it;
        while it < self.end {
            // SAFETY: `it < end`, and the constructing `new` proved every
            // buffer covers `end` values from this row's `base`; the cell
            // `base + it` is this thread's alone (the doc above).
            unsafe {
                let g = *self.gain.add(it);
                let v = *self.x.add(self.base + it);
                *self.y.add(self.base + it) = f(g, v);
            }
            it += self.step;
        }
    }
}

/// The rope launch's three buffers as one shape: `src` and `dst` hold
/// `m * n_vec * nd` values in the `[n_dims, n_vec, m]` column layout, `cs`
/// holds `m * nd` cache values, one thread per (column, pair) of the
/// `npairs = nd / 2` pairs each column covers. [`apply`](Rope::apply)
/// rotates the calling thread's pair, found from its own launch index; it
/// takes no caller index.
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

    /// Rotate this launch thread's pair: `(y0, y1) = f(src0, src1, cos, sin)`
    /// written to `dst` at the pair's two cells, nothing for a thread past
    /// the last pair (the kernel's own `i >= total` guard, same polarity).
    /// The view does the guard, the loads and the stores and hands `f`
    /// values only: a pair handed back as a value would carry the buffer
    /// pointers through the `Some`/`None` merge, where the backend loses
    /// their global state space and the accesses turn generic.
    ///
    /// Sound for every `Rope` safe code can name: a `Rope` can only be
    /// built by [`new`](Self::new), whose contract proved `src` and `dst`
    /// cover `m * n_vec * nd` values and `cs` `m * nd`. A
    /// [`ThreadIndex`] cannot be forged by safe code — only the launch
    /// machinery mints it from hardware coordinates — and it is unique per
    /// thread of the launch, so two threads' calls decompose different flat
    /// indices: `i ↦ (col, p)` is a bijection onto `[0, m * n_vec) ×
    /// [0, npairs)`, and `(col, p) ↦ col·nd + 2p` is injective (`2p < nd`),
    /// so the pairs' cells — and their `+1`s — are disjoint writes inside
    /// `dst`, and the read offsets stay below the extents the contract
    /// proved (`2p + 1 <= nd - 1` for any `nd >= 2`). The call consumes the
    /// `Rope`, so it runs once per thread.
    ///
    /// [`ThreadIndex`]: cuda_device::thread::ThreadIndex
    #[inline(always)]
    pub(crate) fn apply(self, idx: ThreadIndex<'_, Index1D>, f: impl FnOnce(T, T, T, T) -> (T, T))
    where
        T: Copy,
    {
        let i = idx.get();
        let (npairs, n_vec, nd) = (self.npairs, self.n_vec, self.nd);
        if i < self.total {
            let col = i / npairs;
            let d = (i % npairs) * 2;
            let t = col / n_vec;
            let base = col * nd + d;
            let cache = t * nd + d;
            // SAFETY: `base`, `cache` and their `+ 1`s are inside the buffers
            // the constructing `new` proved (the doc above), and `dst`'s two
            // cells are this thread's alone.
            unsafe {
                let (y0, y1) = f(
                    *self.src.add(base),
                    *self.src.add(base + 1),
                    *self.cs.add(cache),
                    *self.cs.add(cache + 1),
                );
                *self.dst.add(base) = y0;
                *self.dst.add(base + 1) = y1;
            }
        }
    }
}
