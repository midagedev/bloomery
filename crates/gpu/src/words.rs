//! The page words' operations of one card ([`WordOps`]): the counters a card
//! and the host tier hand each other over a host-mapped page — waits, writes,
//! system barriers and adds — one batch of stream memory operations a site,
//! one graph node when captured ([`mem_batch`]). A driver that refuses the
//! atomic-reduction stream memory operation gets the batch's adds from a
//! kernel instead ([`word_add`]): one thread of system-scope reductions, the
//! waits, writes and barriers staying memory operations. Which side a card
//! takes is decided once, where the owner of its words opens
//! (`host::word_ops`).

use crate::GpuError;
use crate::graph::{MappedHost, add_of, is_barrier, mem_batch, op_barrier_sys};
use cuda_core::{CudaContext, CudaStream, LaunchConfig1D, sys};
use cuda_device::{kernel, launch_bounds, launch_contract, ptx_asm, thread};
use cuda_host::cuda_module;
use std::fmt;
use std::sync::Arc;

/// Adds one [`word_add`] launch carries: a stage go's (each tier's go
/// counter, the generation and the sequence) and a stage wait's take-backs
/// (the host's counter and each tier's) fit one launch.
pub const WORD_ADDS: usize = 16;

/// Memory operations [`WordOps::batch`] holds before it enqueues them: more
/// than any site's batch, so a site's segment is one batch node.
const SEGMENT: usize = 32;

/// The adds of one [`word_add`] launch, its grid-constant parameter: `n`
/// adds, `by[i]` to the u32 at device address `at[i]`, in order. Built on
/// the host by [`WordAdds::push`] alone.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct WordAdds {
    at: [u64; WORD_ADDS],
    by: [u32; WORD_ADDS],
    n: u32,
}

impl WordAdds {
    /// No adds.
    const NONE: WordAdds = WordAdds {
        at: [0; WORD_ADDS],
        by: [0; WORD_ADDS],
        n: 0,
    };

    /// Adds held.
    fn len(&self) -> usize {
        self.n as usize
    }

    /// Hold one more add, `by` to the u32 at `at`; the caller enqueues the
    /// table once it holds [`WORD_ADDS`].
    fn push(&mut self, at: sys::CUdeviceptr, by: u32) {
        let i = self.len();
        self.at[i] = at;
        self.by[i] = by;
        self.n += 1;
    }

    /// Device code: each add in order, a system-scope reduction ordered
    /// after every access of this thread before it (release), so an
    /// observer that sees an add sees the adds before it, as a batch of
    /// reduction operations runs them. Held past [`WORD_ADDS`] adds, the
    /// first [`WORD_ADDS`].
    #[inline(always)]
    fn run(&self) {
        let n = self.len().min(WORD_ADDS);
        let mut i = 0;
        while i < n {
            // SAFETY: i < n <= WORD_ADDS indexes both arrays. Each address is
            // a live 4-aligned u32 of memory the card reaches in its global
            // window (mapped host memory or a device allocation), for the
            // launch and every replay of a graph that captured it (the
            // launcher's contract); the generic-to-global conversion is exact
            // for such an address.
            unsafe {
                ptx_asm!(
                    "{ .reg .b64 g;\n\t\
                       cvta.to.global.u64 g, %0;\n\t\
                       red.release.sys.global.add.u32 [g], %1; }",
                    in("l") *self.at.get_unchecked(i),
                    in("r") *self.by.get_unchecked(i),
                    clobber("memory")
                );
            }
            i += 1;
        }
    }
}

#[cuda_module]
mod word_kernels {
    use super::*;

    /// The adds of `adds`, in order, by one thread ([`WordAdds::run`]).
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn word_add(#[grid_constant] adds: &WordAdds) {
        if thread::index_1d().get() == 0 {
            adds.run();
        }
    }
}

/// How a card adds to its page words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordSide {
    /// The atomic-reduction stream memory operation.
    Memop,
    /// [`word_add`].
    Kernel,
}

impl WordSide {
    /// The lever's word for the side.
    #[must_use]
    pub fn word(self) -> &'static str {
        match self {
            WordSide::Memop => "memop",
            WordSide::Kernel => "kernel",
        }
    }
}

/// The page words' operations of one card's streams, opened once where the
/// owner of its words opens (`host::word_ops`): the side, the driver
/// version the side was chosen under, and on the kernel side the loaded
/// [`word_add`]. Its display is the load's report, `word_ops=kernel
/// driver=<cuDriverGetVersion>`.
pub struct WordOps {
    side: WordSide,
    driver: i32,
    kernels: Option<word_kernels::LoadedModule>,
    /// The probe's scratch word when a probe chose the side: freed with the
    /// owner, at its teardown, not in the middle of a load.
    _probe: Option<MappedHost>,
}

impl WordOps {
    /// The operations of `side` on `ctx`, chosen under driver version
    /// `driver`; `probe` is the scratch word a probe used, kept until the
    /// owner goes. Loads [`word_add`] on the kernel side. Load-time only.
    pub(crate) fn open(
        ctx: &Arc<CudaContext>,
        side: WordSide,
        driver: i32,
        probe: Option<MappedHost>,
    ) -> Result<WordOps, GpuError> {
        let kernels = match side {
            WordSide::Memop => None,
            // SAFETY: this crate owns the embedded device bundle produced for
            // the module above; its launcher checks the launch contract.
            WordSide::Kernel => Some(unsafe { crate::shared_module!(word_kernels, ctx)? }),
        };
        Ok(WordOps {
            side,
            driver,
            kernels,
            _probe: probe,
        })
    }

    /// The side the card takes.
    #[must_use]
    pub fn side(&self) -> WordSide {
        self.side
    }

    /// The driver version (`cuDriverGetVersion`) the side was chosen under.
    #[must_use]
    pub fn driver(&self) -> i32 {
        self.driver
    }

    /// Enqueue `ops` on `stream`. On the memop side, one batch of stream
    /// memory operations ([`mem_batch`]): one graph node when captured. On
    /// the kernel side, every add goes to [`word_add`] and every other
    /// operation stays a memory operation, in the batch's order but for one
    /// move: a run of adds a system barrier follows, and then another add,
    /// is enqueued after that barrier, together with the adds after it. An
    /// add only moves later, past a barrier alone, so every order the batch
    /// held still holds; a go `[barrier, write, adds, barrier, adds]` is one
    /// batch node and one launch, as `[barrier, write, barrier, adds]` is.
    /// Capturable, allocation-free.
    ///
    /// # Safety
    ///
    /// [`mem_batch`]'s, for every operation, the adds included: each
    /// address points at a 4-aligned u32 of memory the card reaches in
    /// `stream`'s context, allocated and mapped until the batch has run, and
    /// for a captured batch until every launch of the graph has run.
    pub unsafe fn batch(
        &self,
        stream: &CudaStream,
        ops: &mut [sys::CUstreamBatchMemOpParams],
        what: &'static str,
    ) -> Result<(), GpuError> {
        let Some(kernels) = &self.kernels else {
            // SAFETY: the caller's contract is mem_batch's.
            return unsafe { mem_batch(stream, ops, what) };
        };
        let mut seg = Segments {
            kernels,
            stream,
            what,
            mem: [op_barrier_sys(); SEGMENT],
            m: 0,
            adds: WordAdds::NONE,
            held: 0,
        };
        for op in ops.iter() {
            match add_of(op, what)? {
                Some((at, by)) => {
                    seg.lift_held()?;
                    if seg.adds.len() == WORD_ADDS {
                        seg.flush()?;
                    }
                    seg.adds.push(at, by);
                }
                None if is_barrier(op) && seg.adds.len() > 0 => seg.held += 1,
                None => {
                    if seg.adds.len() > 0 {
                        seg.flush()?;
                        seg.lift_held()?;
                    }
                    seg.mem_push(*op)?;
                }
            }
        }
        seg.flush()?;
        seg.lift_held()?;
        seg.flush()
    }
}

impl fmt::Display for WordOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "word_ops={} driver={}", self.side.word(), self.driver)
    }
}

/// [`WordOps::batch`]'s kernel side as it walks a batch: the memory
/// operations held, which go before the adds held, and the barriers that
/// came after those adds, held until the next operation says where they go.
struct Segments<'a> {
    kernels: &'a word_kernels::LoadedModule,
    stream: &'a CudaStream,
    what: &'static str,
    mem: [sys::CUstreamBatchMemOpParams; SEGMENT],
    m: usize,
    adds: WordAdds,
    held: usize,
}

impl Segments<'_> {
    /// Hold one memory operation, enqueueing those held first when full.
    fn mem_push(&mut self, op: sys::CUstreamBatchMemOpParams) -> Result<(), GpuError> {
        if self.m == SEGMENT {
            self.flush_mem()?;
        }
        self.mem[self.m] = op;
        self.m += 1;
        Ok(())
    }

    /// The held barriers into the memory operations, ahead of the adds held.
    fn lift_held(&mut self) -> Result<(), GpuError> {
        while self.held > 0 {
            self.held -= 1;
            self.mem_push(op_barrier_sys())?;
        }
        Ok(())
    }

    /// Enqueue the memory operations held as one batch.
    fn flush_mem(&mut self) -> Result<(), GpuError> {
        if self.m > 0 {
            let m = std::mem::take(&mut self.m);
            // SAFETY: each held operation is one of the batch `WordOps::batch`
            // was given, whose addresses are its caller's contract.
            unsafe { mem_batch(self.stream, &mut self.mem[..m], self.what)? };
        }
        Ok(())
    }

    /// Enqueue the memory operations held, then the adds held as one launch.
    fn flush(&mut self) -> Result<(), GpuError> {
        self.flush_mem()?;
        if self.adds.len() > 0 {
            let adds = std::mem::replace(&mut self.adds, WordAdds::NONE);
            let prep = self
                .kernels
                .prepare_word_add(LaunchConfig1D::new(1, 32, 0))?;
            // SAFETY: the table holds only adds of the batch `WordOps::batch`
            // was given, whose addresses — live 4-aligned u32s the card
            // reaches, for the launch and every replay of a graph that
            // captured it — are its caller's contract; a captured launch
            // carries the table by value, as a captured batch carries its
            // operations.
            unsafe { self.kernels.word_add(self.stream, &prep, adds)? };
        }
        Ok(())
    }
}
