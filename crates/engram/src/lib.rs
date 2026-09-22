//! engram — the NVMe-resident lookup table of DeepSeek-V4.1-Flash.
//!
//! **This crate is the IO path, not the model path.** It answers one question:
//! can NVMe serve 48 scattered row reads inside a decode step? It does not know
//! which rows a token wants — that derivation (the rolling hash over the last
//! three tokens, with its constants in the GGUF metadata) belongs to the model
//! side and plugs in at [`Site::rows_into`], whose only input is a slice of row
//! ids. [`SeededRows`] stands in for it so the access pattern is reproducible.
//!
//! The table is `blk.1.engram_embd.weight` and `blk.14.engram_embd.weight`:
//! two Q8_0 tensors of 256 values per row — 8 blocks × 34 B = **272 B a row** —
//! and ~384 M rows each, 194.55 GiB together. It cannot live in RAM beside the
//! weights, and caching it is not the design: each site is mapped and read
//! where it lies.
//!
//! Four shapes the caller has to know about:
//!
//! * **Rows are borrowed from the mapping.** [`Site::row`] returns a slice into
//!   the mapped file; nothing is copied and nothing is allocated per row.
//! * **The mappings are `MADV_RANDOM`.** Without it the kernel reads 128 KiB
//!   around each fault until its miss heuristic gives up; with it every fault
//!   is one page from the first. `MADV_WILLNEED` is unaffected — that path does
//!   not consult the VMA flags — so [`Site::prefetch`] still works.
//! * **Prefetch is a cold-path lever and a warm-path tax.** It costs one
//!   syscall per row whether or not the page is resident, so a caller that
//!   expects hits should not pay it.
//! * **Borrowing moves the faults onto whoever reads.** A consumer that reads
//!   through the mapping takes one minor fault per row no matter who advised
//!   the pages, and the engine's step is its calling thread's serial time. So
//!   [`Site::copy_rows`] and [`prefetch::Prefetcher`] exist: the helper thread
//!   advises, faults and copies, and the step thread gets one memcpy.
//!
//! The strict reader ([`gguf::Gguf`]) refuses these shards: the engine has no
//! `GgmlType` for Q8_0-as-a-weight yet. So the header comes from
//! [`gguf::inventory_of`], which parses headers only, and this crate owns the
//! mapping.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use gguf::{LoadError, RawTensorInfo, inventory_of};
use memmap2::{Advice, Mmap, UncheckedAdvice};

pub mod prefetch;

/// ggml type id of Q8_0, the type both engram tables carry.
const GGML_TYPE_Q8_0: u32 = 8;
/// Q8_0 packs 32 values into 34 bytes: one `f16` scale and 32 `i8`.
const Q8_0_BLOCK: u64 = 32;
const Q8_0_BLOCK_BYTES: u64 = 34;

/// The tensors this crate opens. Same order as the block index.
const SITE_NAMES: [&str; 2] = ["blk.1.engram_embd.weight", "blk.14.engram_embd.weight"];

#[derive(Debug, thiserror::Error)]
pub enum EngramError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("gguf header: {0}")]
    Header(#[from] LoadError),
    #[error("no engram tensor in the shards given; looked for {0}")]
    NoSites(String),
    #[error("{name}: ggml type {ty} is not Q8_0 ({GGML_TYPE_Q8_0})")]
    NotQ8_0 { name: String, ty: u32 },
    #[error("{name}: expected a 2-D tensor, header says dims {dims:?}")]
    NotATable { name: String, dims: Vec<u64> },
    #[error("{name}: row length {ne0} is not a multiple of the Q8_0 block ({Q8_0_BLOCK})")]
    UnalignedRow { name: String, ne0: u64 },
    #[error(
        "{name}: {rows} rows x {row_bytes} B = {product} B does not tile the {nbytes} B the header states"
    )]
    Untiled {
        name: String,
        rows: u64,
        row_bytes: u64,
        product: u64,
        nbytes: u64,
    },
    #[error("{name}: data at {at} + {nbytes} B runs past the {len} B file {path}")]
    PastEndOfFile {
        name: String,
        path: PathBuf,
        at: u64,
        nbytes: u64,
        len: u64,
    },
    #[error("{name}: copying {rows} rows wants a {want} B buffer, got {got} B")]
    OutBufferSize {
        name: String,
        rows: usize,
        want: usize,
        got: usize,
    },
    #[error("{name}: row {id} is past the {rows} rows the header states")]
    RowOutOfRange { name: String, id: u32, rows: u64 },
    #[error("prefetcher: {0}")]
    Prefetch(&'static str),
}

/// A byte count as `posix_fadvise` takes it. The mapping is far under
/// `off_t::MAX`, so this cannot fail on any host that mapped the file.
fn fadvise_off(n: usize) -> libc::off_t {
    libc::off_t::try_from(n).expect("a span inside a mapping fits an off_t")
}

/// One engram table: a Q8_0 row store mapped where it lies in its shard.
pub struct Site {
    name: String,
    path: PathBuf,
    // Both are kept for the mapping's lifetime. `file` is also the handle the
    // page-cache advice (`posix_fadvise`) and any independent read need.
    file: File,
    map: Mmap,
    /// Byte offset of row 0. The mapping starts at file offset 0, so this is
    /// the file offset too — which is what makes [`Site::file_offset`] usable
    /// by a reader that never maps the file.
    base: u64,
    row_bytes: u64,
    rows: u64,
    /// `sysconf(_SC_PAGESIZE)`, read once: the unit `page_span` rounds to.
    page: u64,
}

impl Site {
    fn open(
        path: &Path,
        data_base: u64,
        file_len: u64,
        t: &RawTensorInfo,
    ) -> Result<Site, EngramError> {
        if t.type_id != GGML_TYPE_Q8_0 {
            return Err(EngramError::NotQ8_0 {
                name: t.name.clone(),
                ty: t.type_id,
            });
        }
        let [ne0, ne1] = t.dims[..] else {
            return Err(EngramError::NotATable {
                name: t.name.clone(),
                dims: t.dims.clone(),
            });
        };
        if !ne0.is_multiple_of(Q8_0_BLOCK) {
            return Err(EngramError::UnalignedRow {
                name: t.name.clone(),
                ne0,
            });
        }
        let row_bytes = ne0 / Q8_0_BLOCK * Q8_0_BLOCK_BYTES;
        // The header's own byte count must be exactly `rows` whole rows: this is
        // what says the row stride is the one the ids are multiplied by.
        let nbytes = t.nbytes.unwrap_or(0);
        let product = ne1 * row_bytes;
        if product != nbytes {
            return Err(EngramError::Untiled {
                name: t.name.clone(),
                rows: ne1,
                row_bytes,
                product,
                nbytes,
            });
        }
        let base = data_base + t.offset;
        if base + nbytes > file_len {
            return Err(EngramError::PastEndOfFile {
                name: t.name.clone(),
                path: path.to_path_buf(),
                at: base,
                nbytes,
                len: file_len,
            });
        }

        // SAFETY: `sysconf` reads a static system value and touches no memory of
        // ours; a negative return means the name is unknown, which _SC_PAGESIZE
        // never is on the platforms this crate builds for.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;

        let file = File::open(path)?;
        // SAFETY: the file is opened read-only and nothing maps it writable; a
        // concurrent truncation would surface as SIGBUS, the same contract
        // `gguf::Gguf::open_backed` and ik_llama.cpp's own loader accept.
        let map = unsafe { memmap2::MmapOptions::new().map(&file)? };
        // Random: `do_sync_mmap_readahead` returns early on VM_RAND_READ, so each
        // fault reads one page instead of 128 KiB around it. Every row of this
        // table is somewhere else; read-around would be 32x the bytes for nothing.
        map.advise(Advice::Random)?;

        Ok(Site {
            name: t.name.clone(),
            path: path.to_path_buf(),
            file,
            map,
            base,
            row_bytes,
            rows: ne1,
            page,
        })
    }

    /// The tensor's name, e.g. `blk.1.engram_embd.weight`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The shard this table lives in.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Rows in the table.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Bytes in one row (272 for a 256-value Q8_0 row).
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    /// Offset of `id`'s first byte **in the shard file**, for a reader that
    /// wants the same bytes without the mapping. Not bounds-checked: this is
    /// arithmetic, [`Site::row`] is the checked accessor.
    pub fn file_offset(&self, id: u32) -> u64 {
        self.base + u64::from(id) * self.row_bytes
    }

    /// One row, borrowed from the mapping. No copy, no allocation, no lock.
    ///
    /// The bound is a compare and a branch; a wrong id must be an error rather
    /// than a slice into the next tensor.
    pub fn row(&self, id: u32) -> Result<&[u8], EngramError> {
        if u64::from(id) >= self.rows {
            return Err(EngramError::RowOutOfRange {
                name: self.name.clone(),
                id,
                rows: self.rows,
            });
        }
        let at = self.file_offset(id) as usize;
        Ok(&self.map[at..at + self.row_bytes as usize])
    }

    /// Every row of `ids`, in order, appended to a caller-owned vector.
    ///
    /// This is the seam the model side plugs into: it hands its row ids and
    /// gets back borrowed rows. `out` is cleared first and reused across calls,
    /// so a steady step allocates nothing.
    pub fn rows_into<'a>(
        &'a self,
        ids: &[u32],
        out: &mut Vec<&'a [u8]>,
    ) -> Result<(), EngramError> {
        out.clear();
        for &id in ids {
            out.push(self.row(id)?);
        }
        Ok(())
    }

    /// Start the reads for `ids` and return without waiting.
    ///
    /// One `madvise(WILLNEED)` per row, which is how ik's own port does it: the
    /// kernel submits each row's page(s) and the later touch finds them in
    /// flight. That turns 48 serial faults into one batch at the drive's queue
    /// depth. The cost is one syscall per row, paid whether or not the page is
    /// already resident.
    pub fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        for &id in ids {
            if u64::from(id) >= self.rows {
                return Err(EngramError::RowOutOfRange {
                    name: self.name.clone(),
                    id,
                    rows: self.rows,
                });
            }
            // `advise_range` rounds the start down to a page and the kernel
            // rounds the length up, so a row that straddles advises both pages.
            self.map.advise_range(
                Advice::WillNeed,
                self.file_offset(id) as usize,
                self.row_bytes as usize,
            )?;
        }
        Ok(())
    }

    /// Fault `ids`' pages in and return only when they are resident.
    ///
    /// `MADV_POPULATE_READ` walks the range in the kernel, so the later touch
    /// takes no user-mode trap per page. It is **synchronous**: under
    /// `MADV_RANDOM` a cold page is one device round trip and this call waits
    /// for each of them in turn. The order that makes it cheap is
    /// [`Site::prefetch`] first — `WILLNEED` submits every row's read at once —
    /// and `populate` after, which then waits on reads already in flight.
    /// Calling it alone on cold rows pays the device latency once per row.
    pub fn populate(&self, ids: &[u32]) -> Result<(), EngramError> {
        for &id in ids {
            if u64::from(id) >= self.rows {
                return Err(EngramError::RowOutOfRange {
                    name: self.name.clone(),
                    id,
                    rows: self.rows,
                });
            }
            let (first, span) = self.page_span(self.file_offset(id), self.row_bytes);
            self.map.advise_range(Advice::PopulateRead, first, span)?;
        }
        Ok(())
    }

    /// Every row of `ids`, in order, **copied** into `out`.
    ///
    /// `out.len()` must be exactly `ids.len() * row_bytes()`. The reads go
    /// through the mapping, so the faults land on the calling thread — which is
    /// the point: [`prefetch::Prefetcher`] calls this on its helper so the step
    /// thread never touches the mapping.
    pub fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        let stride = self.row_bytes as usize;
        let want = ids.len() * stride;
        if out.len() != want {
            return Err(EngramError::OutBufferSize {
                name: self.name.clone(),
                rows: ids.len(),
                want,
                got: out.len(),
            });
        }
        for (&id, slot) in ids.iter().zip(out.chunks_exact_mut(stride)) {
            slot.copy_from_slice(self.row(id)?);
        }
        Ok(())
    }

    /// Drop `ids`' pages from this process and from the page cache, so the next
    /// read of them is a real device read again.
    ///
    /// Order is load-bearing: `posix_fadvise(DONTNEED)` skips a folio that is
    /// still mapped, so the mapping's page-table entries go first.
    pub fn evict_rows(&self, ids: &[u32]) -> Result<(), EngramError> {
        for &id in ids {
            if u64::from(id) >= self.rows {
                return Err(EngramError::RowOutOfRange {
                    name: self.name.clone(),
                    id,
                    rows: self.rows,
                });
            }
            self.evict_range(self.file_offset(id), self.row_bytes)?;
        }
        Ok(())
    }

    /// [`Site::evict_rows`] for the whole table. A full reset: the kernel walks
    /// the page cache over ~104 GB, so this takes as long as there is of this
    /// table cached.
    ///
    /// This drops the table's pages for **every** process reading that file, not
    /// only this one — it is the same class of act as dropping the whole page
    /// cache, narrowed to one file. Narrow it further with [`Site::evict_rows`]
    /// whenever the rows about to be read are known.
    pub fn evict(&self) -> Result<(), EngramError> {
        self.evict_range(self.base, self.rows * self.row_bytes)
    }

    /// `at..at + len` grown outward to whole pages, clamped to the mapping.
    ///
    /// `posix_fadvise(DONTNEED)` rounds its range **inward** — the start up to
    /// a page, the end down to one — so that it never drops a page the caller
    /// named only part of. A 272 B row names no whole page, so the un-grown
    /// call invalidates nothing and the "cold" read that follows is a page-
    /// cache hit. Growing to whole pages is what makes eviction do anything,
    /// and every page in the range holds rows of this table and nothing else,
    /// so the whole page is the right unit. The advice paths round the same way
    /// for the same reason: a row that straddles names both its pages.
    fn page_span(&self, at: u64, len: u64) -> (usize, usize) {
        let first = at / self.page * self.page;
        let end = (at + len)
            .div_ceil(self.page)
            .min((self.map.len() as u64).div_ceil(self.page))
            * self.page;
        (first as usize, (end - first) as usize)
    }

    fn evict_range(&self, at: u64, len: u64) -> Result<(), EngramError> {
        let (first, span) = self.page_span(at, len);

        // SAFETY: `map` is a read-only `MAP_SHARED` view of a file. MADV_DONTNEED
        // on such a mapping drops the page-table entries only — the next touch
        // re-faults the same file bytes. memmap2 calls it unchecked because on an
        // anonymous mapping it would zero the range instead; this mapping is never
        // anonymous.
        unsafe {
            self.map
                .unchecked_advise_range(UncheckedAdvice::DontNeed, first, span)?;
        }
        // SAFETY: the descriptor is this file's own and `self` outlives the call.
        let rc = unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                fadvise_off(first),
                fadvise_off(span),
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if rc != 0 {
            return Err(EngramError::Io(std::io::Error::from_raw_os_error(rc)));
        }
        Ok(())
    }
}

/// The engram table of one model: one [`Site`] per engram layer.
pub struct Engram {
    sites: Vec<Site>,
}

impl Engram {
    /// Open every engram table found in `shards`. Headers only — the tensor
    /// bytes are never touched, so a 444 GiB split opens in the time nine
    /// header parses take.
    pub fn open<P: AsRef<Path>>(
        shards: impl IntoIterator<Item = P>,
    ) -> Result<Engram, EngramError> {
        let mut sites = Vec::new();
        for shard in shards {
            let path = shard.as_ref();
            let inv = inventory_of(path)?;
            for name in SITE_NAMES {
                let Some(t) = inv.tensors.iter().find(|t| t.name == name) else {
                    continue;
                };
                sites.push(Site::open(path, inv.data_base, inv.file_len, t)?);
            }
        }
        if sites.is_empty() {
            return Err(EngramError::NoSites(SITE_NAMES.join(", ")));
        }
        Ok(Engram { sites })
    }

    /// Open the engram tables of the split set in `dir` (every `*.gguf` in it,
    /// in name order).
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Engram, EngramError> {
        let mut shards: Vec<PathBuf> = std::fs::read_dir(dir.as_ref())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        shards.sort();
        Engram::open(shards)
    }

    /// The tables, in the order the shards named them.
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// Bytes in the whole table, both sites summed.
    pub fn bytes(&self) -> u64 {
        self.sites.iter().map(|s| s.rows * s.row_bytes).sum()
    }
}

/// Faults this process has taken, from `getrusage(RUSAGE_SELF)`.
///
/// One syscall, no allocation — cheap enough to sample around a batch.
///
/// Read the pair together, because a prefetched read is not a major fault: the
/// advise put the folio in the page cache with its read in flight, and the
/// later touch waits on it and is counted **minor**. Major faults alone would
/// say a prefetched arm did no IO at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    pub major: u64,
    pub minor: u64,
}

/// glibc `bits/resource.h` defines `RUSAGE_THREAD` as 1; the `libc` crate
/// exposes it for uclibc and emscripten but not for linux-gnu.
const RUSAGE_THREAD: libc::c_int = 1;

/// This process's fault counts right now.
pub fn faults() -> Faults {
    faults_of(libc::RUSAGE_SELF)
}

/// The **calling thread's** fault counts right now.
///
/// This is the counter a helper-thread design needs: the process total cannot
/// say which thread paid for a fault, and the whole claim of moving the read
/// off the step thread is that the step thread stops taking them. Sample it on
/// the thread it describes — a thread cannot ask for another's.
pub fn faults_thread() -> Faults {
    faults_of(RUSAGE_THREAD)
}

fn faults_of(who: libc::c_int) -> Faults {
    // SAFETY: `usage` is a live, writable `rusage` for the duration of the call
    // and `getrusage` fills it or leaves it untouched.
    let usage = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        libc::getrusage(who, &mut usage);
        usage
    };
    Faults {
        major: usage.ru_majflt.max(0) as u64,
        minor: usage.ru_minflt.max(0) as u64,
    }
}

/// Bytes this process actually fetched from the block layer (`/proc/self/io`
/// `read_bytes`), readahead it submitted through `madvise` included.
///
/// **Diagnostic, not a hot-path counter**: it opens and reads a file, which
/// costs about as much as a whole warm token. Sample it at the ends of a run.
///
/// This is the only counter that proves a prefetched arm did any IO — see
/// [`Faults`].
pub fn read_bytes() -> Result<u64, std::io::Error> {
    let text = std::fs::read_to_string("/proc/self/io")?;
    text.lines()
        .find_map(|l| l.strip_prefix("read_bytes:"))
        .and_then(|v| v.trim().parse().ok())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "/proc/self/io has no read_bytes line",
            )
        })
}

/// What a run served. Nothing in the lookup path writes one — the caller folds
/// its own batches in, so an unprofiled path takes no lock, no atomic and no
/// per-row store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    pub rows: u64,
    pub bytes: u64,
    pub nanos: u64,
    pub major_faults: u64,
    pub minor_faults: u64,
}

impl Counters {
    /// Fold in one batch's rows and the caller's own elapsed time.
    pub fn add_batch(&mut self, rows: u64, bytes: u64, nanos: u64) {
        self.rows += rows;
        self.bytes += bytes;
        self.nanos += nanos;
    }

    /// Fold in the faults taken between two [`faults`] samples.
    pub fn add_faults(&mut self, before: Faults, after: Faults) {
        self.major_faults += after.major.saturating_sub(before.major);
        self.minor_faults += after.minor.saturating_sub(before.minor);
    }
}

/// Reproducible row ids, standing in for the model's derivation.
///
/// **The real thing is not here.** In V4.1 a site's 24 ids per token come from
/// a rolling hash over the last three tokens (3 orders × 8 heads) whose
/// constants live in the GGUF metadata (`deepseek41.engram.{multipliers,
/// primes, offsets, token_map, pad_id}`); the 24 buckets partition the table
/// into disjoint prime-sized intervals, so one token's rows are spread over the
/// whole table. That derivation is the model side's, and it meets this crate at
/// [`Site::rows_into`] — a slice of ids is the entire interface.
///
/// What this reproduces is the **access pattern**: uniform over the table, the
/// same sequence for the same seed. What it does not reproduce is **reuse** —
/// the real hash keys on n-grams, so common unigrams and bigrams re-hit rows
/// that are still resident. A uniform draw over 384 M rows is the zero-reuse
/// floor; real decode sits between it and an all-resident table.
pub struct SeededRows {
    state: u64,
}

impl SeededRows {
    pub fn new(seed: u64) -> SeededRows {
        SeededRows { state: seed }
    }

    /// `n` ids uniform in `0..rows`, appended to a cleared caller-owned vector.
    pub fn next_into(&mut self, rows: u64, n: usize, out: &mut Vec<u32>) {
        out.clear();
        for _ in 0..n {
            out.push((self.next_u64() % rows) as u32);
        }
    }

    /// splitmix64 — a fixed sequence per seed, so a run is repeatable and two
    /// arms can be handed the same rows.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
