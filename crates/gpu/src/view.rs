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

/// The calling block's own row of a one-block-per-row kernel: the row's
/// count, read once at entry from `counts[row]`, and the row's one output
/// word `out[STRIDE · row + OFFSET]`, which thread 0 of the block writes.
/// [`count`](RowWord::count) returns the value read and
/// [`first`](RowWord::first) runs the write; neither takes an index.
#[must_use]
pub(crate) struct RowWord<'s, 'a, const STRIDE: usize, const OFFSET: usize> {
    count: u32,
    out: *mut u32,
    row: usize,
    tid: usize,
    _out: PhantomData<&'s mut DisjointSlice<'a, u32>>,
}

impl<'s, 'a, const STRIDE: usize, const OFFSET: usize> RowWord<'s, 'a, STRIDE, OFFSET> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this row that means
    ///     `counts.len() >= m` and `out.len() >= STRIDE * m` for the `m` this
    ///     launch passed.
    /// (b) `row` is the calling block's `blockIdx_x`, below `m` (the
    ///     kernel's block-uniform guard on its parameter `m` has returned
    ///     already), so every thread of the block holds the same `row` and
    ///     no two blocks hold the same one; `tid` is the calling thread's own
    ///     `threadIdx_x`.
    /// (c) The launch's block is the contract's exact 1-D block
    ///     (`BlockRequirement::Exact`, cuda-core — `prepare_*` refuses any
    ///     other shape), so exactly one thread of the block has `tid == 0`.
    #[inline(always)]
    pub(crate) unsafe fn new(
        counts: &[u32],
        row: usize,
        out: &'s mut DisjointSlice<'a, u32>,
        tid: usize,
    ) -> RowWord<'s, 'a, STRIDE, OFFSET> {
        const { assert!(OFFSET < STRIDE) };
        RowWord {
            // SAFETY: row < m <= counts.len() by (a) and (b).
            count: unsafe { *counts.as_ptr().add(row) },
            out: out.as_mut_ptr(),
            row,
            tid,
            _out: PhantomData,
        }
    }

    /// The row's count, as read at construction.
    #[inline(always)]
    #[must_use]
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Thread 0 of the block writes `f()` into the row's output word; every
    /// other thread neither calls `f` nor writes. The compare is the
    /// kernel's own `tid == 0`, and the store is addressed from `out` itself
    /// at `STRIDE · row + OFFSET`, so the backend keeps its global state
    /// space.
    ///
    /// Sound for every `RowWord` safe code can name: a `RowWord` can only be
    /// built by [`new`](Self::new), and under its contract (a) and (b) the
    /// word `STRIDE · row + OFFSET` (`OFFSET < STRIDE`, checked at compile
    /// time, and `row < m`) is below `STRIDE · m <= out.len()`; (b) gives
    /// each block its own `row`, and `STRIDE · row + OFFSET` is distinct for
    /// distinct rows, so no other block writes this word; (c) leaves one
    /// thread of the block with `tid == 0`, so no other thread of the block
    /// writes it. The view borrows `out` mutably for its life, so nothing
    /// else in the kernel reaches the buffer meanwhile; `first` consumes the
    /// view, so a thread writes through it once, and `f` returns a value,
    /// never a reference into a buffer.
    #[inline(always)]
    pub(crate) fn first(self, f: impl FnOnce() -> u32) {
        if self.tid == 0 {
            // SAFETY: the word is inside `out` and this thread's alone (the
            // doc above).
            unsafe { *self.out.add(STRIDE * self.row + OFFSET) = f() };
        }
    }
}

// ---- unsafew1e

/// The calling thread's own value of a flat one-thread-per-value launch:
/// input `x[i]` and output `y[i]`, `i` the thread's flat index.
/// [`map`](Elem::map) runs the access; it takes no index.
#[must_use]
pub(crate) struct Elem<'x, 's, 'a, X, Y> {
    x: *const X,
    y: *mut Y,
    i: usize,
    _x: PhantomData<&'x [X]>,
    _y: PhantomData<&'s mut DisjointSlice<'a, Y>>,
}

impl<'x, 's, 'a, X, Y> Elem<'x, 's, 'a, X, Y> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this value that means `x.len() >= n`
    ///     and `y.len() >= n` for the `n` this launch passed.
    /// (b) `n` is a value every thread of the grid computes identically from
    ///     the kernel's parameters, and `i` is the calling thread's own flat
    ///     index — its `index_1d`, or that less a grid-uniform offset —
    ///     below `n`, the kernel's own `i >= n` guard having returned
    ///     already.
    /// (c) The launch is `domain = 1` (the launcher takes a
    ///     `LaunchConfig1D`, a 1-D grid) with the contract's exact 1-D block
    ///     (`BlockRequirement::Exact`, cuda-core — `prepare_*` refuses any
    ///     other shape), so `index_1d = blockIdx_x · blockDim_x +
    ///     threadIdx_x` is distinct for every thread of the grid (cuda-device
    ///     `thread.rs`, `index_1d`: the X-only formula is unique for a 1-D
    ///     launch).
    #[inline(always)]
    pub(crate) unsafe fn new(
        x: &'x [X],
        y: &'s mut DisjointSlice<'a, Y>,
        i: usize,
    ) -> Elem<'x, 's, 'a, X, Y> {
        Elem {
            x: x.as_ptr(),
            y: y.as_mut_ptr(),
            i,
            _x: PhantomData,
            _y: PhantomData,
        }
    }

    /// `y[i] = f(x[i])`: one load, `f`, one store, and no other check.
    /// Taking `f` rather than handing out a reference keeps the store
    /// addressed from `y` itself, so the backend keeps its global state
    /// space.
    ///
    /// Sound for every `Elem` safe code can name: an `Elem` can only be
    /// built by [`new`](Self::new), and under its contract (a) and (b)
    /// `i < n` is inside `x` and `y`; (b) and (c) give each thread of the
    /// grid its own `i`, so no other thread writes `y[i]`. The view borrows
    /// `y` mutably for its life, so nothing else in the kernel reaches the
    /// buffer meanwhile; `map` consumes the view, so a thread writes through
    /// it once, and `f` receives and returns values, never a reference into
    /// a buffer.
    #[inline(always)]
    pub(crate) fn map(self, f: impl FnOnce(X) -> Y)
    where
        X: Copy,
        Y: Copy,
    {
        // SAFETY: `i < n`, inside both buffers, and `y[i]` is this thread's
        // alone (the doc above).
        unsafe {
            let v = *self.x.add(self.i);
            *self.y.add(self.i) = f(v);
        }
    }
}

/// [`Elem`] with two inputs: `a[i]` and `b[i]` in, `y[i]` out.
/// [`map`](Elem2::map) runs the access; it takes no index.
#[must_use]
pub(crate) struct Elem2<'x, 's, 'a, A, B, Y> {
    a: *const A,
    b: *const B,
    y: *mut Y,
    i: usize,
    _a: PhantomData<&'x [A]>,
    _b: PhantomData<&'x [B]>,
    _y: PhantomData<&'s mut DisjointSlice<'a, Y>>,
}

impl<'x, 's, 'a, A, B, Y> Elem2<'x, 's, 'a, A, B, Y> {
    /// # Safety
    ///
    /// [`Elem::new`]'s three clauses, with (a) reading `a.len() >= n`,
    /// `b.len() >= n` and `y.len() >= n`.
    #[inline(always)]
    pub(crate) unsafe fn new(
        a: &'x [A],
        b: &'x [B],
        y: &'s mut DisjointSlice<'a, Y>,
        i: usize,
    ) -> Elem2<'x, 's, 'a, A, B, Y> {
        Elem2 {
            a: a.as_ptr(),
            b: b.as_ptr(),
            y: y.as_mut_ptr(),
            i,
            _a: PhantomData,
            _b: PhantomData,
            _y: PhantomData,
        }
    }

    /// `y[i] = f(a[i], b[i])`: the two loads in that order, `f`, one store,
    /// and no other check.
    ///
    /// Sound for every `Elem2` safe code can name, as [`Elem::map`] is: only
    /// [`new`](Self::new) builds one, its contract puts `i` inside all three
    /// buffers and gives `y[i]` to this thread alone, the view holds `y`'s
    /// mutable borrow, `map` consumes it, and `f` sees values only.
    #[inline(always)]
    pub(crate) fn map(self, f: impl FnOnce(A, B) -> Y)
    where
        A: Copy,
        B: Copy,
        Y: Copy,
    {
        // SAFETY: `i < n`, inside all three buffers, and `y[i]` is this
        // thread's alone (the doc above).
        unsafe {
            let (va, vb) = (*self.a.add(self.i), *self.b.add(self.i));
            *self.y.add(self.i) = f(va, vb);
        }
    }
}

/// The calling thread's own value pair of an interleaved rope launch: flat
/// index `i` is pair `d / 2` of column `col = i / npairs` (`d = 2 · (i %
/// npairs)`), whose values are `src[col · nd + d]` and the next, turned by
/// token `t = col / n_vec`'s cache pair `cs[t · nd + d]` and the next, into
/// `dst` at the values' own places. [`map`](RopePair::map) runs the access;
/// it takes no index.
#[must_use]
pub(crate) struct RopePair<'x, 's, 'a> {
    src: *const f32,
    cs: *const f32,
    dst: *mut f32,
    /// `col · nd + d`: the pair's first value in `src` and `dst`.
    at: usize,
    /// `t · nd + d`: the pair's cosine in `cs`.
    ct: usize,
    _src: PhantomData<&'x [f32]>,
    _cs: PhantomData<&'x [f32]>,
    _dst: PhantomData<&'s mut DisjointSlice<'a, f32>>,
}

impl<'x, 's, 'a> RopePair<'x, 's, 'a> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this pair that means
    ///     `src.len() >= m · n_vec · nd`, `cs.len() >= m · nd` and
    ///     `dst.len() >= m · n_vec · nd` for the `m`, `n_vec` and `nd` this
    ///     launch passed.
    /// (b) `nd`, `n_vec` and `npairs = nd / 2` are values every thread of the
    ///     grid computes identically from the kernel's parameters, and `i` is
    ///     the calling thread's own `index_1d`, below `m · n_vec · npairs`
    ///     (the kernel's own guard having returned already).
    /// (c) The launch is `domain = 1` (a `LaunchConfig1D` grid) with the
    ///     contract's exact 1-D block (`BlockRequirement::Exact`, cuda-core),
    ///     so `index_1d` is distinct for every thread of the grid (cuda-device
    ///     `thread.rs`, `index_1d`).
    #[inline(always)]
    pub(crate) unsafe fn new(
        src: &'x [f32],
        cs: &'x [f32],
        dst: &'s mut DisjointSlice<'a, f32>,
        i: usize,
        npairs: usize,
        nd: usize,
        n_vec: usize,
    ) -> RopePair<'x, 's, 'a> {
        let col = i / npairs;
        let d = (i % npairs) * 2;
        let t = col / n_vec;
        RopePair {
            src: src.as_ptr(),
            cs: cs.as_ptr(),
            dst: dst.as_mut_ptr(),
            at: col * nd + d,
            ct: t * nd + d,
            _src: PhantomData,
            _cs: PhantomData,
            _dst: PhantomData,
        }
    }

    /// `(dst[at], dst[at + 1]) = f(src[at], src[at + 1], cs[ct], cs[ct + 1])`:
    /// the four loads in that order, `f`, the two stores, and no other
    /// check.
    ///
    /// Sound for every `RopePair` safe code can name: only
    /// [`new`](Self::new) builds one, and under its contract `i < m · n_vec ·
    /// npairs` puts `col < m · n_vec` and `d + 1 <= 2 · npairs − 1 <= nd − 1`,
    /// so `at + 1 < m · n_vec · nd`, inside `src` and `dst`, and `t < m`, so
    /// `ct + 1 < m · nd`, inside `cs` (no parity of `nd` is assumed: an odd
    /// one leaves its last value unpaired). Distinct `i` name distinct
    /// `(col, d)`, and the pairs `(d, d + 1)` of a column do not overlap, so
    /// with (c) no other thread writes this pair's two values. The view
    /// holds `dst`'s mutable borrow, `map` consumes it, and `f` sees values
    /// only.
    #[inline(always)]
    pub(crate) fn map(self, f: impl FnOnce(f32, f32, f32, f32) -> (f32, f32)) {
        // SAFETY: both pairs are inside their buffers and the `dst` pair is
        // this thread's alone (the doc above).
        unsafe {
            let (x0, x1, c, s) = (
                *self.src.add(self.at),
                *self.src.add(self.at + 1),
                *self.cs.add(self.ct),
                *self.cs.add(self.ct + 1),
            );
            let (y0, y1) = f(x0, x1, c, s);
            *self.dst.add(self.at) = y0;
            *self.dst.add(self.at + 1) = y1;
        }
    }
}

/// The calling thread's own value of a stream-combine launch over `[m][S]
/// [hid]` streams: flat index `i` is value `v = i % hid` of stream
/// `cs = i / hid` (column `c = cs / S`), which becomes `f(wgt[cs], y[c ·
/// hid + v], res[i])` in place. [`map`](StreamFma::map) runs the access; it
/// takes no index.
#[must_use]
pub(crate) struct StreamFma<'x, 's, 'a, const S: usize> {
    y: *const f32,
    wgt: *const f32,
    res: *mut f32,
    i: usize,
    cs: usize,
    /// `c · hid + v`: the value's input in `y`.
    yi: usize,
    _y: PhantomData<&'x [f32]>,
    _wgt: PhantomData<&'x [f32]>,
    _res: PhantomData<&'s mut DisjointSlice<'a, f32>>,
}

impl<'x, 's, 'a, const S: usize> StreamFma<'x, 's, 'a, S> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this value that means
    ///     `y.len() >= m · hid`, `wgt.len() >= m · S` and
    ///     `res.len() >= m · S · hid` for the `m` and `hid` this launch
    ///     passed.
    /// (b) `hid` is the kernel's parameter, the same in every thread of the
    ///     grid, and `i` is the calling thread's own `index_1d`, below
    ///     `m · S · hid` (the kernel's own guard having returned already, so
    ///     `hid > 0`).
    /// (c) The launch is `domain = 1` (a `LaunchConfig1D` grid) with the
    ///     contract's exact 1-D block (`BlockRequirement::Exact`, cuda-core),
    ///     so `index_1d` is distinct for every thread of the grid (cuda-device
    ///     `thread.rs`, `index_1d`).
    #[inline(always)]
    pub(crate) unsafe fn new(
        y: &'x [f32],
        wgt: &'x [f32],
        res: &'s mut DisjointSlice<'a, f32>,
        i: usize,
        hid: usize,
    ) -> StreamFma<'x, 's, 'a, S> {
        let (cs, v) = (i / hid, i % hid);
        let c = cs / S;
        StreamFma {
            y: y.as_ptr(),
            wgt: wgt.as_ptr(),
            res: res.as_mut_ptr(),
            i,
            cs,
            yi: c * hid + v,
            _y: PhantomData,
            _wgt: PhantomData,
            _res: PhantomData,
        }
    }

    /// `res[i] = f(wgt[cs], y[c · hid + v], res[i])`, returning the value
    /// stored: the three loads in that order, `f`, one store, and no other
    /// check.
    ///
    /// Sound for every `StreamFma` safe code can name: only
    /// [`new`](Self::new) builds one, and under its contract `i < m · S ·
    /// hid` is inside `res`, `cs = i / hid < m · S` inside `wgt`, and
    /// `c = cs / S < m` with `v < hid` puts `c · hid + v < m · hid` inside
    /// `y`; (c) gives `res[i]` to this thread alone, the only cell it writes.
    /// The view holds `res`'s mutable borrow, `map` consumes it, and `f`
    /// sees and returns values only.
    #[inline(always)]
    pub(crate) fn map(self, f: impl FnOnce(f32, f32, f32) -> f32) -> f32 {
        // SAFETY: the three cells are inside their buffers and `res[i]` is
        // this thread's alone (the doc above).
        unsafe {
            let r = self.res.add(self.i);
            *r = f(*self.wgt.add(self.cs), *self.y.add(self.yi), *r);
            *r
        }
    }
}

/// One block's stream of a stream-norm launch over `[m][S][hid]` streams:
/// the block is stream `s` of column `c`, and the calling thread strides
/// `step` from its own index over the stream's `hid` values. Built with the
/// stream's combine weight read once (`wgt[c · S + s]` under `combine == 1`).
/// [`combine_fold`](StreamNorm::combine_fold) runs the in-place pass over
/// `res` (a combine through `y`, a copy of `y`, or neither, as the flags
/// say) and folds each value; [`apply`](StreamNorm::apply) runs the store
/// pass into `xn` with the stream's gains. Neither takes an index.
#[must_use]
pub(crate) struct StreamNorm<'x, 's, 'a, 't, 'b, const S: usize> {
    y: *const f32,
    gamma: *const f32,
    res: *mut f32,
    xn: *mut f32,
    /// `(c · S + s) · hid`: the stream in `res` and `xn`.
    base: usize,
    /// `c · hid`: the column in `y`.
    ybase: usize,
    /// `s · hid`: the stream's gains in `gamma`.
    gbase: usize,
    it: usize,
    step: usize,
    end: usize,
    combine: u32,
    init: u32,
    w: f32,
    _y: PhantomData<&'x [f32]>,
    _gamma: PhantomData<&'x [f32]>,
    _res: PhantomData<&'s mut DisjointSlice<'a, f32>>,
    _xn: PhantomData<&'t mut DisjointSlice<'b, f32>>,
}

impl<'x, 's, 'a, 't, 'b, const S: usize> StreamNorm<'x, 's, 'a, 't, 'b, S> {
    /// # Safety
    ///
    /// The caller is a kernel, and every clause is one of its launch facts:
    ///
    /// (a) The kernel's `requires` clauses held at launch: the generated
    ///     safe launcher evaluated them on the host before enqueueing and
    ///     refused the launch otherwise (`generate_requires_checks`,
    ///     cuda-macros `contract.rs`; the `*_unchecked` launchers that skip
    ///     them are `unsafe fn`). For this stream that means
    ///     `combine + init <= 1`, `y.len() >= (combine + init) · m · hid`,
    ///     `wgt.len() >= combine · m · S`, `gamma.len() >= S · hid`,
    ///     `res.len() >= m · S · hid` and `xn.len() >= m · S · hid` for the
    ///     `m`, `hid`, `combine` and `init` this launch passed — each side
    ///     evaluated in `u64` with checked arithmetic, so the flags are 0 or
    ///     1 and at most one is 1.
    /// (b) `hid`, `combine`, `init` and `step` are values every thread of the
    ///     grid holds identically (kernel parameters and the block width);
    ///     `c` and `s` are the calling block's `blockIdx_x / S` and
    ///     `blockIdx_x % S`, with `c < m` (the kernel's block-uniform column
    ///     guard having returned already), so every thread of the block holds
    ///     the same pair and no two blocks hold the same one; `it` is the
    ///     calling thread's own `threadIdx_x`.
    /// (c) The launch's block is the contract's exact 1-D `(step, 1, 1)`
    ///     (`BlockRequirement::Exact`, cuda-core — `prepare_*` refuses any
    ///     other shape), so `it < step`.
    #[allow(
        clippy::too_many_arguments,
        reason = "one constructor carries the kernel's whole view set (crate::view module doc)"
    )]
    #[inline(always)]
    pub(crate) unsafe fn new(
        y: &'x [f32],
        wgt: &[f32],
        gamma: &'x [f32],
        res: &'s mut DisjointSlice<'a, f32>,
        xn: &'t mut DisjointSlice<'b, f32>,
        (c, s): (usize, usize),
        (combine, init): (u32, u32),
        hid: usize,
        it: usize,
        step: usize,
    ) -> StreamNorm<'x, 's, 'a, 't, 'b, S> {
        let w = if combine == 1 {
            // SAFETY: c·S + s < m·S <= wgt.len() under `combine == 1` by (a)
            // and (b).
            unsafe { *wgt.as_ptr().add(c * S + s) }
        } else {
            0.0
        };
        StreamNorm {
            y: y.as_ptr(),
            gamma: gamma.as_ptr(),
            res: res.as_mut_ptr(),
            xn: xn.as_mut_ptr(),
            base: (c * S + s) * hid,
            ybase: c * hid,
            gbase: s * hid,
            it,
            step,
            end: hid,
            combine,
            init,
            w,
            _y: PhantomData,
            _gamma: PhantomData,
            _res: PhantomData,
            _xn: PhantomData,
        }
    }

    /// The in-place pass: for every value `i` of this thread's stride, `res
    /// [base + i]` becomes `upd(w, y[ybase + i], res[base + i])` under
    /// `combine == 1`, `y[ybase + i]` under `init == 1`, and stays as it is
    /// otherwise (`y` is then not read); then `acc = fold(acc, res[base +
    /// i])`. Returns the last `acc`. The loop is the kernel's own (`i <
    /// end`, then `i += step`).
    ///
    /// Sound for every `StreamNorm` safe code can name: only
    /// [`new`](Self::new) builds one, and under its contract `i < end = hid`
    /// puts `base + i < (c · S + s + 1) · hid <= m · S · hid`, inside `res`,
    /// and, when `y` is read (`combine == 1` or `init == 1`, so
    /// `combine + init = 1`), `ybase + i < m · hid <= y.len()`. This thread
    /// alone names `res[base + i]`: no other block holds `(c, s)`, and no
    /// other thread of the block shares `i`'s residue class of the stride
    /// (distinct `threadIdx_x` below `step`). The pass writes only those
    /// cells; running it twice changes values, never which cells. `upd` and
    /// `fold` see and return values only.
    #[inline(always)]
    pub(crate) fn combine_fold<A>(
        &mut self,
        mut acc: A,
        upd: impl Fn(f32, f32, f32) -> f32,
        mut fold: impl FnMut(A, f32) -> A,
    ) -> A {
        let mut i = self.it;
        while i < self.end {
            // SAFETY: the cells are inside their buffers and `res[base + i]`
            // is this thread's alone (the doc above).
            let x = unsafe {
                let r = self.res.add(self.base + i);
                if self.combine == 1 {
                    *r = upd(self.w, *self.y.add(self.ybase + i), *r);
                } else if self.init == 1 {
                    *r = *self.y.add(self.ybase + i);
                }
                *r
            };
            acc = fold(acc, x);
            i += self.step;
        }
        acc
    }

    /// The store pass: `xn[base + i] = f(res[base + i], gamma[gbase + i])`
    /// for every value `i` of this thread's stride, the loads in that order.
    /// The loop is the kernel's own; the pass consumes the view.
    ///
    /// Sound for every `StreamNorm` safe code can name, as
    /// [`combine_fold`](Self::combine_fold) is: `base + i` is inside `res`
    /// and `xn` and this thread's alone, and `gbase + i < (s + 1) · hid <=
    /// S · hid <= gamma.len()`. The `res` cell read is one this thread wrote
    /// or left, never another thread's, so no ordering with other threads is
    /// needed. `f` sees values only.
    #[inline(always)]
    pub(crate) fn apply(self, mut f: impl FnMut(f32, f32) -> f32) {
        let mut i = self.it;
        while i < self.end {
            // SAFETY: the cells are inside their buffers and `xn[base + i]`
            // is this thread's alone (the doc above).
            unsafe {
                let x = *self.res.add(self.base + i);
                *self.xn.add(self.base + i) = f(x, *self.gamma.add(self.gbase + i));
            }
            i += self.step;
        }
    }
}
