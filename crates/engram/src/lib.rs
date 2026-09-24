//! engram — the NVMe-resident lookup table of DeepSeek-V4.1-Flash.
//!
//! **This crate is the IO path first.** It answers one question: can NVMe serve
//! 48 scattered row reads inside a decode step? Which rows a token wants is the
//! [`hash`] module's, built from the GGUF metadata's own constants; the two meet
//! at [`Site::rows_into`], whose only input is a slice of row ids.
//! [`SeededRows`] stands in for the hash when the point is the access pattern
//! alone — a uniform draw is the zero-reuse floor. A real stream re-asks for
//! rows, and [`cache::RowCache`] keeps them in DRAM so that only its misses
//! reach the disk; [`reuse::Lru`] is the simulator its hits are held to.
//!
//! The tables are the `engram_embd` weights of the blocks the metadata names
//! (blk.1 and blk.14 in V4.1-Flash): ~384 M rows of 256 values each, in the
//! file's block type — one Q3_K block of **110 B a row** in the public
//! `Q3_K_M` file (84.6 GB together), eight Q8_0 blocks of 272 B in the mixed
//! one. They cannot live in RAM beside the weights, and each site is mapped
//! and read where it lies. A row's bytes are the file's; decoding them is the
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
//!   syscall per row whether or not the page is resident, so a caller that
//!   expects hits should not pay it.
//! * **Borrowing moves the faults onto whoever reads.** A consumer that reads
//!   through the mapping takes one minor fault per row no matter who advised
//!   the pages, and the engine's step is its calling thread's serial time. So
//!   [`Site::copy_rows`] and [`prefetch::Prefetcher`] exist: the helper thread
//!   advises, faults and copies, and the step thread gets one memcpy.
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

pub mod cache;
pub mod hash;
pub mod prefetch;
pub mod reuse;

pub use hash::{Context, Hash};

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
    #[error("{name}: expected a 2-D tensor, header says dims {dims:?}")]
    NotATable { name: String, dims: Vec<u64> },
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
    #[error("row cache: {0}")]
    Cache(&'static str),
    #[error("reuse simulator: {0}")]
    Reuse(&'static str),
    #[error("{path}: {source}")]
    IdsFile {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: '{text}': {source}")]
    TokenId {
        path: PathBuf,
        line: usize,
        text: String,
        source: std::num::ParseIntError,
    },
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
            self.map
                .advise_range(
                    Advice::WillNeed,
                    self.file_offset(id) as usize,
                    self.row_bytes as usize,
                )
                .map_err(|e| self.io_err("madvise(WILLNEED)", e))?;
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
            self.map
                .advise_range(Advice::PopulateRead, first, span)
                .map_err(|e| self.io_err("madvise(POPULATE_READ)", e))?;
        }
        Ok(())
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
        const CHUNK: usize = 8;
        let mut resident = 0u64;
        for &id in ids {
            if u64::from(id) >= self.rows {
                return Err(EngramError::RowOutOfRange {
                    name: self.name.clone(),
                    id,
                    rows: self.rows,
                });
            }
            let (first, span) = self.page_span(self.file_offset(id), self.row_bytes);
            let page = self.page as usize;
            let mut all = true;
            let mut at = first;
            while all && at < first + span {
                let len = (first + span - at).min(CHUNK * page);
                let mut vec = [0u8; CHUNK];
                // SAFETY: `at..at + len` is page-aligned and inside the live
                // mapping (`page_span` clamps to it), and `vec` holds one byte
                // for each of the at most `CHUNK` pages of that range.
                let rc = unsafe {
                    libc::mincore(
                        self.map.as_ptr().add(at).cast_mut().cast(),
                        len,
                        vec.as_mut_ptr(),
                    )
                };
                if rc != 0 {
                    return Err(self.io_err("mincore", std::io::Error::last_os_error()));
                }
                all = vec[..len.div_ceil(page)].iter().all(|b| b & 1 == 1);
                at += len;
            }
            resident += u64::from(all);
        }
        Ok(resident)
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
            return Err(self.io_err(
                "posix_fadvise(DONTNEED)",
                std::io::Error::from_raw_os_error(rc),
            ));
        }
        Ok(())
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
/// **The real derivation is [`Hash`]**, and an arm that wants the rows a real
/// token stream asks for uses it. What this reproduces instead is the **access
/// pattern**: uniform over the table, the same sequence for the same seed, with
/// no allocation and no metadata.
///
/// What it does not reproduce is **reuse** — the real hash keys on n-grams, so
/// common bigrams re-hit rows that are still resident. A uniform draw over
/// 384 M rows is the zero-reuse floor; real decode sits between it and an
/// all-resident table, and `engram-reuse` is where that distance is measured.
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
