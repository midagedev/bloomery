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
