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
//! kernel's own loop condition.
//!
//! A view runs the access pattern itself and hands the kernel's closure
//! values, never a reference or a pointer. An accessor that returned
//! `Option<&mut T>`, or a struct carrying a buffer pointer, would put a
//! `Some`/`None` test in every step, cost the loop its strength reduction,
//! and let the backend lose the pointer's global state space at the merge;
//! a struct holding the address of a kernel-local struct keeps that struct
//! in a local depot. A kernel whose converted form has more instructions, a
//! longer loop body or a generic access where it had a global one keeps its
//! `get_unchecked` form and its `// SAFETY:`.
//!
//! The shared-memory reduce stays outside this file: "every slot was
//! written before the barrier" is a fact about other threads' control
//! flow, which no per-thread type can carry, so those accesses keep their
//! `unsafe` and their `// SAFETY:` where they stand.

use core::marker::PhantomData;

use cuda_device::DisjointSlice;

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
