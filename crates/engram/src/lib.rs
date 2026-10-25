//! engram — the NVMe-resident lookup table of DeepSeek-V4.1-Flash.
//!
//! **This crate is the IO path first.** It answers one question: can NVMe serve
//! 48 scattered row reads inside a decode step? Which rows a token wants is the
//! [`hash`] module's, built from the GGUF metadata's own constants; the two meet
//! at [`Site::rows_into`], whose only input is a slice of row ids. What
//! measures these reads — a uniform row draw, a DRAM row cache, a reuse
//! simulator and the `engram-rate` and `engram-reuse` bins — is the
//! `engram-lab` crate, built on this one; the engine uses none of it.
//!
//! The tables are the `engram_embd` weights of the blocks the metadata names
//! (blk.1 and blk.14 in V4.1-Flash): ~384 M rows of 256 values each, in the
//! file's block type — one Q3_K block of **110 B a row** in the public
//! `Q3_K_M` file (84.6 GB together). They cannot live in RAM beside the
//! weights, and each site is mapped and read where it lies. A row's bytes are
//! the file's; decoding them is the
//! reader's, so the row type is carried ([`Site::type_id`]) and not
//! interpreted here.
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
//!   syscall per distinct merged page run whether or not its pages are
//!   resident, so a caller that expects hits should not pay it.
//! * **Borrowing moves the faults onto whoever reads.** A consumer that reads
//!   through the mapping takes one minor fault per row no matter who advised
//!   the pages, and the engine's step is its calling thread's serial time. So
//!   [`Site::copy_rows`] and [`prefetch::Prefetcher`] exist: the helper thread
//!   advises, faults and copies, and the step thread gets one memcpy.
//!
//! The same row reads serve a row-gathered table another model's load maps
//! through its [`gguf::Split`] ([`rows::SplitTable`]: Qwen3.8's PLE table
//! when its plan leaves it on the NVMe tier, or a routed stack of experts,
//! each expert a row), and a [`prefetch::Prefetcher`] reads either
//! ([`RowTables`]).
//!
//! The header comes from [`gguf::inventory_of`], which parses headers only and
//! never populates a mapping, and this crate maps each shard itself so the
//! mapping can be `MADV_RANDOM`. The strict reader ([`gguf::Gguf`]) sizes
//! the same types; this crate does not go through it, since the header and
//! its own mapping are all it needs.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use gguf::{Inventory, LoadError, RawTensorInfo, ggml_type_info, inventory_of};
use memmap2::{Advice, Mmap, UncheckedAdvice};

pub mod direct;
pub mod hash;
pub mod prefetch;
pub mod rows;

pub use direct::DIRECT_ALIGN;
pub use direct::open_direct;
pub use hash::Hash;

/// The tensor one engram site lives in. The block index is metadata, so this is
/// the only place the name's shape is written down.
fn site_name(layer_id: u32) -> String {
    format!("blk.{layer_id}.engram_embd.weight")
}

/// Every `*.gguf` of a split set, in name order.
fn shards_in(dir: &Path) -> Result<Vec<PathBuf>, EngramError> {
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|source| EngramError::ShardDir {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    Ok(shards)
}

#[derive(Debug, thiserror::Error)]
pub enum EngramError {
    /// An IO error with no file behind it: spawning the prefetch helper
    /// thread is the only producer. File and mapping errors carry their
    /// context in [`EngramError::SiteIo`] and [`EngramError::ShardDir`].
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{name} in {path}: {op}: {source}")]
    SiteIo {
        name: String,
        path: PathBuf,
        op: &'static str,
        source: std::io::Error,
    },
    #[error("listing shards in {path}: {source}")]
    ShardDir {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("gguf header: {0}")]
    Header(#[from] LoadError),
    #[error("no shard carries the engram metadata; looked for {0}")]
    NoHashMetadata(String),
    #[error("the metadata names an engram site at {name}, and no shard carries that tensor")]
    MissingSite { name: String },
    #[error("{0}")]
    Fact(#[from] gguf::v41_facts::FactError),
    #[error("gguf metadata has no key {key}")]
    MissingKey { key: String },
    #[error("{key}: the file's value is not {want}")]
    KeyType { key: String, want: &'static str },
    #[error("{key}: expected {want} values, the file has {got}")]
    KeyLength {
        key: String,
        want: usize,
        got: usize,
    },
    #[error("{key}: entry {at} is not {what}")]
    KeyValue {
        key: String,
        at: usize,
        what: &'static str,
    },
    #[error("engram site {site} is past the {sites} the metadata names")]
    SiteOutOfRange { site: usize, sites: usize },
    #[error("the context window wants {want} mapped tokens, got {got}")]
    WindowSize { want: usize, got: usize },
    #[error("one site's row ids want a {want}-id buffer, got {got}")]
    RowBufferSize { want: usize, got: usize },
    #[error("{name}: ggml type {ty} has no block size in ggml's size table")]
    UnsizedType { name: String, ty: u32 },
    #[error("{name}: expected {want}, header says dims {dims:?}")]
    NotATable {
        name: String,
        dims: Vec<u64>,
        want: &'static str,
    },
    #[error("{path}: opening for direct reads: {source}")]
    OpenDirect {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(
        "{path}: the direct-read probe of its first block: {source}: the mount does not take \
         O_DIRECT reads"
    )]
    DirectProbe {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: a direct read of {len} B at {at}: {source}")]
    DirectRead {
        path: PathBuf,
        at: u64,
        len: usize,
        source: std::io::Error,
    },
    #[error(
        "{path}: a direct read of {len} B at {at}: the buffer, the offset and the length must \
         each be a multiple of {DIRECT_ALIGN}"
    )]
    DirectUnaligned { path: PathBuf, at: u64, len: usize },
    #[error("the anonymous mapping of {bytes} B: {source}")]
    AnonMap {
        bytes: usize,
        source: std::io::Error,
    },
    #[error("returning the anonymous mapping's {len} B at {at}: {source}")]
    AnonMapAdvise {
        at: usize,
        len: usize,
        source: std::io::Error,
    },
    #[error("{name}: row length {ne0} is not a multiple of its type's block ({block})")]
    UnalignedRow { name: String, ne0: u64, block: u64 },
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
    /// A table held as runs ([`rows::RunTable`]) of no rows, of empty rows, or of more rows
    /// than a `u32` row id names.
    #[error(
        "{name}: {rows} rows of {row_bytes} B: a run table holds 1 to 2^32 rows of at least one byte"
    )]
    RunRows {
        name: String,
        rows: u64,
        row_bytes: u64,
    },
    #[error("{name}: data at {base}+{off} + {nbytes} B runs past the {len} B file {path}")]
    PastEndOfFile {
        name: String,
        path: PathBuf,
        base: u64,
        off: u64,
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
    #[error("no shard of the split carries the row-gathered table {name}")]
    MissingTable { name: String },
    #[error(
        "token {token} is past the {map} entries of the engram token map: not a token of this model"
    )]
    TokenPastMap { token: u32, map: usize },
    #[error(
        "the engram token map has {map} entries and the model's vocabulary {vocab} tokens; the map is indexed by token id, so the two must be equal"
    )]
    VocabMismatch { map: usize, vocab: usize },
    #[error("prefetcher: {0}")]
    Prefetch(&'static str),
}

/// A byte count as `posix_fadvise` takes it. The mapping is far under
/// `off_t::MAX`, so this cannot fail on any host that mapped the file.
fn fadvise_off(n: usize) -> libc::off_t {
    libc::off_t::try_from(n).expect("a span inside a mapping fits an off_t")
}

/// One engram table: a row store mapped where it lies in its shard.
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
    /// The rows' ggml type id.
    type_id: u32,
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
        let Some((_, block, block_bytes)) = ggml_type_info(t.type_id) else {
            return Err(EngramError::UnsizedType {
                name: t.name.clone(),
                ty: t.type_id,
            });
        };
        let [ne0, ne1] = t.dims[..] else {
            return Err(EngramError::NotATable {
                name: t.name.clone(),
                dims: t.dims.clone(),
                want: "a 2-D table",
            });
        };
        if !ne0.is_multiple_of(block) {
            return Err(EngramError::UnalignedRow {
                name: t.name.clone(),
                ne0,
                block,
            });
        }
        let row_bytes = ne0 / block * block_bytes;
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
        let base = data_base
            .checked_add(t.offset)
            .filter(|b| b.checked_add(nbytes).is_some_and(|end| end <= file_len))
            .ok_or_else(|| EngramError::PastEndOfFile {
                name: t.name.clone(),
                path: path.to_path_buf(),
                base: data_base,
                off: t.offset,
                nbytes,
                len: file_len,
            })?;

        let page = page_size() as u64;

        let io = |op: &'static str| {
            move |source| EngramError::SiteIo {
                name: t.name.clone(),
                path: path.to_path_buf(),
                op,
                source,
            }
        };
        let file = File::open(path).map_err(io("open"))?;
        // SAFETY: the file is opened read-only and nothing maps it writable; a
        // concurrent truncation would surface as SIGBUS, the same contract
        // `gguf::Gguf::open_backed` and ik_llama.cpp's own loader accept.
        let map = unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(io("mmap"))?;
        // Random: `do_sync_mmap_readahead` returns early on VM_RAND_READ, so each
        // fault reads one page instead of 128 KiB around it. Every row of this
        // table is somewhere else; read-around would be 32x the bytes for nothing.
        map.advise(Advice::Random).map_err(io("madvise(RANDOM)"))?;

        Ok(Site {
            name: t.name.clone(),
            path: path.to_path_buf(),
            file,
            map,
            base,
            type_id: t.type_id,
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

    /// The rows' ggml type id, as the header states it.
    pub fn type_id(&self) -> u32 {
        self.type_id
    }

    /// Bytes in one row: 110 for a 256-value Q3_K row, 272 for a Q8_0 one.
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
    /// One `madvise(WILLNEED)` per distinct page run. Repeated and neighbouring
    /// rows share advice; the later touch finds their reads in flight.
    pub fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.view().prefetch(ids)
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
        self.view().populate(ids)
    }

    /// How many rows of `ids` have every page in the page cache right now
    /// (`mincore`, one call per row). A diagnostic: it answers "would this row
    /// have been a device read", and costs a syscall per row whether or not.
    ///
    /// `mincore` reports the page cache of a file mapping only to a caller that
    /// owns the file or could write it (root on the box does); anyone else
    /// sees only the pages this process has mapped, so a cached-but-unmapped
    /// row reads as not resident.
    pub fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError> {
        self.view().resident_rows(ids)
    }

    /// Every row of `ids`, in order, **copied** into `out`.
    ///
    /// `out.len()` must be exactly `ids.len() * row_bytes()`. The reads go
    /// through the mapping, so the faults land on the calling thread — which is
    /// the point: [`prefetch::Prefetcher`] calls this on its helper so the step
    /// thread never touches the mapping.
    pub fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        self.view().copy_rows(ids, out)
    }

    /// The table's rows in the mapping, as the row reads see them.
    fn view(&self) -> TableBytes<'_> {
        let at = self.base as usize;
        TableBytes {
            name: &self.name,
            path: &self.path,
            bytes: &self.map[at..at + (self.rows * self.row_bytes) as usize],
            row_bytes: self.row_bytes as usize,
            page: self.page as usize,
        }
    }

    /// Drop `ids`' pages from this process and from the page cache, so the next
    /// read of them is a real device read again: one eviction per run of
    /// adjacent rows (`adjacent_runs`), over the whole pages that hold it.
    /// An id past the table is refused by name before anything is dropped.
    ///
    /// Order is load-bearing: `posix_fadvise(DONTNEED)` skips a folio that is
    /// still mapped, so the mapping's page-table entries go first.
    pub fn evict_rows(&self, ids: &[u32]) -> Result<(), EngramError> {
        let runs = adjacent_runs(ids);
        let last = runs.last().map(|&(_, last)| last);
        if let Some(id) = last.filter(|&id| u64::from(id) >= self.rows) {
            return Err(EngramError::RowOutOfRange {
                name: self.name.clone(),
                id,
                rows: self.rows,
            });
        }
        for (first, last) in runs {
            self.evict_range(
                self.file_offset(first),
                (u64::from(last) - u64::from(first) + 1) * self.row_bytes,
            )?;
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
    /// named only part of. A row of a few hundred bytes names no whole page,
    /// so the un-grown call invalidates nothing and the "cold" read that
    /// follows is a page-cache hit. Growing to whole pages is what makes eviction do anything,
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

    /// `source` from `op` on this table, with the table's name and file.
    fn io_err(&self, op: &'static str, source: std::io::Error) -> EngramError {
        EngramError::SiteIo {
            name: self.name.clone(),
            path: self.path.clone(),
            op,
            source,
        }
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
                .unchecked_advise_range(UncheckedAdvice::DontNeed, first, span)
                .map_err(|e| self.io_err("madvise(DONTNEED)", e))?;
        }
        fadvise_dontneed(&self.file, first, span)
            .map_err(|e| self.io_err("posix_fadvise(DONTNEED)", e))
    }
}

/// `posix_fadvise(DONTNEED)` over `at .. at + len` of `file`: the page-cache
/// half of an eviction, after the mapping's own entries are dropped.
fn fadvise_dontneed(file: &File, at: usize, len: usize) -> Result<(), std::io::Error> {
    os::fadvise_dontneed(file.as_raw_fd(), fadvise_off(at), fadvise_off(len))
        .map_err(std::io::Error::from)
}

/// `ids` as runs of adjacent rows, each `(first, last)` inclusive, sorted and
/// without repeats: `[8, 3, 0, 2]` is `(0, 0), (2, 3), (8, 8)`.
///
/// The unit an eviction drops. `posix_fadvise(DONTNEED)` drops only the
/// folios that lie whole inside its range, and a large folio can hold several
/// rows, so a call per row keeps every folio that crosses a row boundary. A
/// call per run keeps only a folio that crosses one of the run's two edges: a
/// page the run shares with a row or bytes outside it may stay, with the rest
/// of its folio.
fn adjacent_runs(ids: &[u32]) -> Vec<(u32, u32)> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for id in sorted {
        match runs.last_mut() {
            Some((_, last)) if id - *last == 1 => *last = id,
            _ => runs.push((id, id)),
        }
    }
    runs
}

/// Pages one `mincore` call covers.
const MINCORE_PAGES: usize = 8;

/// `sysconf(_SC_PAGESIZE)`: the unit a row's pages round to.
fn page_size() -> usize {
    // SAFETY: `sysconf` reads a static system value and touches no memory of
    // ours; a negative return means the name is unknown, which _SC_PAGESIZE
    // never is on the platforms this crate builds for.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize
}

/// One row-gathered table's bytes in a file mapping and the row reads over
/// them — the advice, the residency query and the copy — for a [`Site`]'s
/// own mapping and a [`rows::SplitTable`]'s split mapping alike. `bytes`
/// lies inside a mapping that starts on a page boundary (every `mmap`
/// does), so the whole pages a row's span rounds out to are the mapping's.
struct TableBytes<'a> {
    name: &'a str,
    /// The file the mapping reads, for the refusals.
    path: &'a Path,
    bytes: &'a [u8],
    row_bytes: usize,
    /// `sysconf(_SC_PAGESIZE)`: the unit a row's pages round to.
    page: usize,
}

impl TableBytes<'_> {
    /// Rows in the table.
    fn rows(&self) -> u64 {
        (self.bytes.len() / self.row_bytes) as u64
    }

    /// Row `id`'s bytes; refused by name past the table, so a wrong id never
    /// slices into the next tensor. The id is checked before its offset is
    /// computed, so a product that wraps cannot land on another row.
    fn row(&self, id: u32) -> Result<&[u8], EngramError> {
        if u64::from(id) >= self.rows() {
            return Err(EngramError::RowOutOfRange {
                name: self.name.to_owned(),
                id,
                rows: self.rows(),
            });
        }
        let at = id as usize * self.row_bytes;
        Ok(&self.bytes[at..at + self.row_bytes])
    }

    /// The whole pages that hold row `id`: the row's start rounded down to a
    /// page and its end up — a row that straddles a boundary names both.
    fn pages(&self, id: u32) -> Result<(usize, usize), EngramError> {
        self.run_pages(id, id)
    }

    /// The whole pages that hold rows `first ..= last`, rounded out as
    /// [`TableBytes::pages`] rounds one row.
    fn run_pages(&self, first: u32, last: u32) -> Result<(usize, usize), EngramError> {
        let start = self.row(first)?.as_ptr() as usize;
        let end = self.row(last)?.as_ptr() as usize + self.row_bytes;
        let first = start / self.page * self.page;
        let end = end.div_ceil(self.page) * self.page;
        Ok((first, end - first))
    }

    /// The distinct page runs of `ids`, sorted and joined at page boundaries.
    fn page_runs(&self, ids: &[u32]) -> Result<Vec<(usize, usize)>, EngramError> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (first, last) in adjacent_runs(ids) {
            let (at, len) = self.run_pages(first, last)?;
            match spans.last_mut() {
                Some((start, size)) if at <= *start + *size => {
                    *size = (*start + *size).max(at + len) - *start;
                }
                _ => spans.push((at, len)),
            }
        }
        Ok(spans)
    }

    /// One `madvise(advice)` per distinct page run, `op` naming a refusal.
    fn advise(
        &self,
        ids: &[u32],
        advice: libc::c_int,
        op: &'static str,
    ) -> Result<(), EngramError> {
        for (first, len) in self.page_runs(ids)? {
            self.advise_pages(first, len, advice, op)?;
        }
        Ok(())
    }

    /// `madvise(advice)` over the whole pages that hold the table: once at a
    /// load ([`rows::SplitTable::open`]).
    fn advise_all(&self, advice: libc::c_int, op: &'static str) -> Result<(), EngramError> {
        let start = self.bytes.as_ptr() as usize;
        let first = start / self.page * self.page;
        let end = (start + self.bytes.len()).div_ceil(self.page) * self.page;
        self.advise_pages(first, end - first, advice, op)
    }

    /// One `madvise(advice)` over `first .. first + len`, whole pages of the
    /// mapping `bytes` lies in.
    fn advise_pages(
        &self,
        first: usize,
        len: usize,
        advice: libc::c_int,
        op: &'static str,
    ) -> Result<(), EngramError> {
        #[cfg(test)]
        use crate::tests::recording_madvise as madvise;
        #[cfg(not(test))]
        use libc::madvise;

        // SAFETY: every caller supplies whole pages inside the file mapping
        // `bytes` lies in; madvise reads and writes no memory of ours
        // through them. The one advice that can change what a read returns
        // is DONTNEED on an anonymous mapping, which zeroes the range; every
        // mapping this type covers is a read-only file mapping (a site's own,
        // and a split's, whose shards `gguf::Split::open` maps from their
        // files and nothing else), where it drops the page-table entries and
        // the next touch reads the file's bytes again. A run table's
        // (`rows::RunTable`) maps its files read-only too.
        let rc = unsafe { madvise(first as *mut libc::c_void, len, advice) };
        if rc == 0 {
            Ok(())
        } else {
            Err(self.io(op, std::io::Error::last_os_error()))
        }
    }

    /// [`Site::prefetch`]'s read: `WILLNEED` over the distinct page runs.
    fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.advise(ids, libc::MADV_WILLNEED, "madvise(WILLNEED)")
    }

    /// [`Site::populate`]'s read: `POPULATE_READ` over the distinct page runs.
    fn populate(&self, ids: &[u32]) -> Result<(), EngramError> {
        let page = u64::try_from(self.page).expect("a page size fits a u64");
        for (first, len) in self.page_runs(ids)? {
            let addr = first as *mut libc::c_void;
            // SAFETY: page_runs joins only overlapping or adjacent whole
            // pages inside the same file mapping.
            unsafe { os::populate_span(addr, len, page, || os::advise_populate(addr, len)) }
                .map_err(|e| self.io("madvise(POPULATE_READ)", e.into()))?;
        }
        Ok(())
    }

    /// [`Site::resident_rows`]' count: the rows of `ids` whose pages `mincore`
    /// reports all resident, a call per row of at most eight pages each.
    fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError> {
        let mut resident = 0u64;
        for &id in ids {
            let (first, span) = self.pages(id)?;
            let mut all = true;
            let mut at = first;
            while all && at < first + span {
                let len = (first + span - at).min(MINCORE_PAGES * self.page);
                let vec = self.mincore_chunk(at, len)?;
                all = vec[..len.div_ceil(self.page)].iter().all(|b| b & 1 == 1);
                at += len;
            }
            resident += u64::from(all);
        }
        Ok(resident)
    }

    /// The pages `mincore` reports resident among those `ids`' rows span, and
    /// the pages they span: a row's whole pages, counted once for each row
    /// that names them, so two neighbours' shared page counts twice. The
    /// count a drop is checked against (nothing resident) and a read (every
    /// page).
    fn resident_pages(&self, ids: &[u32]) -> Result<(u64, u64), EngramError> {
        let (mut resident, mut total) = (0u64, 0u64);
        for &id in ids {
            let (first, span) = self.pages(id)?;
            let mut at = first;
            while at < first + span {
                let len = (first + span - at).min(MINCORE_PAGES * self.page);
                let vec = self.mincore_chunk(at, len)?;
                let pages = len.div_ceil(self.page);
                resident += vec[..pages].iter().filter(|b| *b & 1 == 1).count() as u64;
                total += pages as u64;
                at += len;
            }
        }
        Ok((resident, total))
    }

    /// `mincore` over `at .. at + len`, a page-aligned span of at most
    /// [`MINCORE_PAGES`] pages inside the mapping: one byte a page, bit 0 set
    /// when the page is in the page cache.
    fn mincore_chunk(&self, at: usize, len: usize) -> Result<[u8; MINCORE_PAGES], EngramError> {
        let mut vec = [0u8; MINCORE_PAGES];
        // `at .. at +len` is page-aligned and inside the mapping (`pages`
        // rounds a row out to the mapping's own pages), and `vec` holds one
        // byte for each of the at most `MINCORE_PAGES` pages of that range.
        // SAFETY: the caller's contract above.
        unsafe { os::mincore(at as *mut libc::c_void, len, &mut vec) }
            .map_err(|e| self.io("mincore", e.into()))?;
        Ok(vec)
    }

    /// [`Site::copy_rows`]' copy: every row of `ids`, in order, into `out`
    /// of exactly `ids.len()` rows.
    fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        let want = ids.len() * self.row_bytes;
        if out.len() != want {
            return Err(EngramError::OutBufferSize {
                name: self.name.to_owned(),
                rows: ids.len(),
                want,
                got: out.len(),
            });
        }
        for (&id, slot) in ids.iter().zip(out.chunks_exact_mut(self.row_bytes)) {
            slot.copy_from_slice(self.row(id)?);
        }
        Ok(())
    }

    fn io(&self, op: &'static str, source: std::io::Error) -> EngramError {
        EngramError::SiteIo {
            name: self.name.to_owned(),
            path: self.path.to_path_buf(),
            op,
            source,
        }
    }
}

/// A row-gathered table a [`prefetch::Prefetcher`] reads: fixed-size rows by
/// id, advised, populated, classified and copied ([`Site`], an engram site
/// in its own mapping; [`rows::SplitTable`], a table in its load's split).
pub trait RowTable: Send + Sync {
    /// Bytes in one row.
    fn row_bytes(&self) -> u64;
    /// `WILLNEED` over `ids`' pages ([`Site::prefetch`]).
    fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError>;
    /// `POPULATE_READ` over `ids`' pages ([`Site::populate`]).
    fn populate(&self, ids: &[u32]) -> Result<(), EngramError>;
    /// The rows of `ids` already resident ([`Site::resident_rows`]).
    fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError>;
    /// `ids`' rows copied into `out` ([`Site::copy_rows`]).
    fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError>;
}

impl RowTable for Site {
    fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.view().prefetch(ids)
    }

    fn populate(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.view().populate(ids)
    }

    fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError> {
        self.view().resident_rows(ids)
    }

    fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        self.view().copy_rows(ids, out)
    }
}

/// The tables one [`prefetch::Prefetcher`] reads, in the order its jobs
/// name them: an [`Engram`]'s sites, or one [`rows::SplitTable`].
pub trait RowTables: Send + Sync {
    /// How many tables a job names.
    fn count(&self) -> usize;
    /// Table `i`; `None` past [`RowTables::count`].
    fn table(&self, i: usize) -> Option<&dyn RowTable>;
}

impl<T: RowTable> RowTables for Vec<T> {
    fn count(&self) -> usize {
        self.len()
    }

    fn table(&self, i: usize) -> Option<&dyn RowTable> {
        self.get(i).map(|t| t as &dyn RowTable)
    }
}

impl RowTables for Engram {
    fn count(&self) -> usize {
        self.sites.len()
    }

    fn table(&self, i: usize) -> Option<&dyn RowTable> {
        self.sites.get(i).map(|s| s as &dyn RowTable)
    }
}

/// The engram table of one model: the hash and one [`Site`] per engram layer.
pub struct Engram {
    hash: Hash,
    sites: Vec<Site>,
}

impl Engram {
    /// Open every engram table the metadata names. Headers only — the tensor
    /// bytes are never touched, so a 444 GiB split opens in the time nine
    /// header parses take.
    ///
    /// The sites come back in `layer_ids` order, not in shard order, so
    /// `sites()[e]` and the hash's site `e` are the same site by construction.
    /// Every site the metadata names must be present: a set missing one is an
    /// error, because the caller's `e` would otherwise silently mean a
    /// different table.
    pub fn open<P: AsRef<Path>>(
        shards: impl IntoIterator<Item = P>,
    ) -> Result<Engram, EngramError> {
        let paths: Vec<PathBuf> = shards
            .into_iter()
            .map(|p| p.as_ref().to_path_buf())
            .collect();
        let mut invs: Vec<Inventory> = Vec::with_capacity(paths.len());
        for path in &paths {
            invs.push(inventory_of(path)?);
        }
        let hash = Hash::from_shards(&invs)?;

        let mut slots: Vec<Option<Site>> = (0..hash.sites()).map(|_| None).collect();
        for (path, inv) in paths.iter().zip(&invs) {
            for (e, slot) in slots.iter_mut().enumerate() {
                let name = site_name(hash.layer_ids()[e]);
                let Some(t) = inv.tensors.iter().find(|t| t.name == name) else {
                    continue;
                };
                *slot = Some(Site::open(path, inv.data_base, inv.file_len, t)?);
            }
        }

        let mut sites = Vec::with_capacity(slots.len());
        for (e, slot) in slots.into_iter().enumerate() {
            sites.push(slot.ok_or_else(|| EngramError::MissingSite {
                name: site_name(hash.layer_ids()[e]),
            })?);
        }
        Ok(Engram { hash, sites })
    }

    /// Open the engram tables of the split set in `dir` (every `*.gguf` in it,
    /// in name order).
    pub fn open_dir(dir: impl AsRef<Path>) -> Result<Engram, EngramError> {
        Engram::open(shards_in(dir.as_ref())?)
    }

    /// The tables, in `layer_ids` order.
    pub fn sites(&self) -> &[Site] {
        &self.sites
    }

    /// The row derivation, with this model's own constants.
    pub fn hash(&self) -> &Hash {
        &self.hash
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
/// the thread it describes — a thread cannot ask for another's. Linux only: a
/// kernel without `RUSAGE_THREAD` (macOS) refuses it, and that panics by name.
pub fn faults_thread() -> Faults {
    faults_of(RUSAGE_THREAD)
}

fn faults_of(who: libc::c_int) -> Faults {
    // SAFETY: `usage` is a live, writable `rusage` for the duration of the call
    // and `getrusage` fills it or leaves it untouched.
    let (rc, usage) = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        let rc = libc::getrusage(who, &mut usage);
        (rc, usage)
    };
    assert_eq!(
        rc,
        0,
        "getrusage({who}) refused: {}",
        std::io::Error::last_os_error()
    );
    Faults {
        major: usage.ru_majflt.max(0) as u64,
        minor: usage.ru_minflt.max(0) as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::{EngramError, RawTensorInfo, Site, adjacent_runs};

    std::thread_local! {
        static MADVISE_CALLS: std::cell::RefCell<Vec<(usize, usize, libc::c_int)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// # Safety
    /// `addr` and `len` must name whole pages inside a read-only file mapping.
    pub(super) unsafe fn recording_madvise(
        addr: *mut libc::c_void,
        len: usize,
        advice: libc::c_int,
    ) -> libc::c_int {
        // SAFETY: the caller supplies whole pages in a read-only file mapping.
        let rc = unsafe { libc::madvise(addr, len, advice) };
        if rc == 0 {
            MADVISE_CALLS.with_borrow_mut(|calls| calls.push((addr as usize, len, advice)));
        }
        rc
    }

    // PIN(2026-10-23): this syscall test subsumes page_runs_advise_each_requested_page_once.
    #[test]
    fn prefetch_advises_the_requested_page_runs() {
        let page = super::page_size();
        let path = std::env::temp_dir().join(format!("engram-advice-{}", std::process::id()));
        for row_bytes in [12, page, page + 12] {
            for offset in [0, 3] {
                let base = page + offset;
                let len = base + row_bytes * 32 + page;
                std::fs::File::create(&path)
                    .unwrap()
                    .set_len(len as u64)
                    .unwrap();
                let tensor = RawTensorInfo {
                    name: "blk.1.engram_embd.weight".to_string(),
                    dims: vec![(row_bytes / 4) as u64, 32],
                    type_id: 0,
                    offset: offset as u64,
                    nbytes: Some((row_bytes * 32) as u64),
                };
                let site = Site::open(&path, page as u64, len as u64, &tensor).unwrap();
                let ids = [31, 15, 2, 0, 2, 1, 8, 9];
                let mut expected = std::collections::BTreeSet::new();
                for &id in &ids {
                    let start = site.map.as_ptr() as usize + base + id as usize * row_bytes;
                    expected.extend(start / page..=(start + row_bytes - 1) / page);
                }
                let expected: Vec<_> = expected.into_iter().collect();
                for prefetch in [Site::prefetch, <Site as super::RowTable>::prefetch] {
                    MADVISE_CALLS.with_borrow_mut(Vec::clear);
                    prefetch(&site, &ids).unwrap();
                    let calls = MADVISE_CALLS.with_borrow_mut(std::mem::take);
                    assert!(
                        !calls.is_empty(),
                        "prefetch must advise the requested pages"
                    );
                    assert!(
                        calls.iter().all(|&(at, len, advice)| {
                            at.is_multiple_of(page)
                                && len > 0
                                && len.is_multiple_of(page)
                                && advice == libc::MADV_WILLNEED
                        }),
                        "prefetch must use WILLNEED over whole pages"
                    );
                    let actual: Vec<_> = calls
                        .iter()
                        .flat_map(|&(at, len, _)| at / page..(at + len) / page)
                        .collect();
                    assert_eq!(
                        actual, expected,
                        "prefetch must advise exactly the row pages"
                    );
                    assert!(calls.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0));
                    prefetch(&site, &[]).unwrap();
                    assert!(MADVISE_CALLS.with_borrow(Vec::is_empty));
                    assert!(matches!(
                        prefetch(&site, &[0, 32]),
                        Err(EngramError::RowOutOfRange { .. })
                    ));
                    assert!(MADVISE_CALLS.with_borrow(Vec::is_empty));
                }
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// A table whose start or end passes u64 is past the end of its file, named
    /// with the data base and the header's offset: an F32 [4, 1] table in an
    /// 80-byte file whose data starts at 64, at an offset that wraps the start
    /// to the last header byte, and at one that wraps the end to 8.
    #[test]
    fn a_table_whose_range_passes_u64_is_past_the_end() {
        let (data_base, len) = (64, 80);
        let path = std::env::temp_dir().join(format!("engram-wrap-{}", std::process::id()));
        std::fs::write(&path, [0u8; 80]).unwrap();
        for offset in [u64::MAX, u64::MAX - data_base - 7] {
            let t = RawTensorInfo {
                name: "blk.1.engram_embd.weight".to_string(),
                dims: vec![4, 1],
                type_id: 0,
                offset,
                nbytes: Some(16),
            };
            match Site::open(&path, data_base, len, &t) {
                Err(EngramError::PastEndOfFile { base, off, .. }) => {
                    assert_eq!((base, off), (data_base, offset));
                }
                other => panic!(
                    "offset {offset} must be past the end, got {:?}",
                    other.map(|_| ())
                ),
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    /// Runs come out sorted, without repeats, adjacent ids joined, and the
    /// last id a `u32` holds joins its neighbour without an overflow.
    #[test]
    fn adjacent_runs_sorts_dedups_and_joins() {
        assert_eq!(adjacent_runs(&[]), []);
        assert_eq!(adjacent_runs(&[5, 5]), [(5, 5)]);
        assert_eq!(adjacent_runs(&[8, 3, 0, 2]), [(0, 0), (2, 3), (8, 8)]);
        assert_eq!(
            adjacent_runs(&[u32::MAX - 1, u32::MAX]),
            [(u32::MAX - 1, u32::MAX)]
        );
    }
}
