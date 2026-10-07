//! A row-gathered table read where its load's split maps it.
//!
//! [`Site`](crate::Site) maps its shard itself. A table whose model side
//! opened the file through a [`Split`] — Qwen3.8's PLE table — is read
//! inside that split's own mapping instead: the host-resident arm of the same
//! table reads that mapping with no advice at all, and one mapping keeps one
//! set of page-table entries for the file whichever tier the plan named.
//! [`SplitTable`] is the table as a [`RowTable`] — the same advice,
//! residency query and copy as a site's ([`crate::TableBytes`]) — so a
//! [`Prefetcher`](crate::prefetch::Prefetcher) reads it.
//!
//! A routed expert stack is the same read over a 3-D tensor: its rows are the
//! experts, one `[ne0, ne1]` slab each, so a stack on the NVMe tier is
//! advised, populated, classified, copied and evicted by expert id.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use gguf::{Gguf, Split, TensorInfo};

use crate::{EngramError, RowTable, RowTables, TableBytes, fadvise_dontneed, page_size};

/// The advice a table's pages take when it opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageAdvice {
    /// `MADV_RANDOM`: a fault reads its own page and nothing around it. For a
    /// table read a few rows at a time, wherever the ids fall (PLE).
    Random,
    /// `MADV_NORMAL`: the kernel's own read-around. For a table read in runs
    /// a whole row long, which a `WILLNEED` or a populate then reads in one
    /// request (a routed expert stack).
    Normal,
}

impl PageAdvice {
    /// The `madvise` advice and the name a refusal gives it.
    fn madvise(self) -> (libc::c_int, &'static str) {
        match self {
            PageAdvice::Random => (libc::MADV_RANDOM, "madvise(RANDOM)"),
            PageAdvice::Normal => (libc::MADV_NORMAL, "madvise(NORMAL)"),
        }
    }
}

/// The pages `mincore` found in the page cache among the pages some rows span
/// ([`SplitTable::resident_pages`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageCount {
    pub resident: u64,
    pub total: u64,
}

/// One row-gathered table of a [`Split`], read from the NVMe tier through
/// the split's own mapping.
pub struct SplitTable {
    split: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    /// The shard file, for the refusals and the eviction's descriptor.
    path: PathBuf,
    row_bytes: u64,
    rows: u64,
    /// `sysconf(_SC_PAGESIZE)`, read once.
    page: usize,
    /// The shard opened for `posix_fadvise`, on the first eviction: a table
    /// that is never evicted holds no descriptor.
    file: OnceLock<File>,
}

impl SplitTable {
    /// [`SplitTable::open_with`] under [`PageAdvice::Random`]: the table read
    /// a few rows at a time (Qwen3.8's PLE table), a fault reading its own
    /// page and not the 128 KiB around it.
    pub fn open(split: Arc<Split>, name: &str) -> Result<SplitTable, EngramError> {
        SplitTable::open_with(split, name, PageAdvice::Random)
    }

    /// The table `name` of `split`: a 2-D table, its rows the second dim, or
    /// a 3-D stack of experts, its rows the third (one `[ne0, ne1]` slab
    /// each); whole blocks a row and rows tiling the bytes the header states,
    /// each refused by name otherwise, and any other rank refused as
    /// [`EngramError::NotATable`]. Then `advice` over its pages in the
    /// split's mapping. Load-time only, and only for a table the plan left on
    /// the NVMe tier: a table the host set holds is read through the mapping
    /// as it is.
    pub fn open_with(
        split: Arc<Split>,
        name: &str,
        advice: PageAdvice,
    ) -> Result<SplitTable, EngramError> {
        let missing = || EngramError::MissingTable {
            name: name.to_owned(),
        };
        let (shard, info) = split
            .find(name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(missing)?;
        let path = split.shard_path(shard).ok_or_else(missing)?.to_path_buf();
        let (ne0, slab, rows) = match info.dims[..] {
            [ne0, ne1] => (ne0, 1, ne1),
            [ne0, ne1, ne2] => (ne0, ne1, ne2),
            _ => {
                return Err(EngramError::NotATable {
                    name: name.to_owned(),
                    dims: info.dims.clone(),
                    want: "a 2-D table or a 3-D stack of experts",
                });
            }
        };
        let (Some(block), Some(block_bytes)) = (info.ty.blck_size(), info.ty.type_size()) else {
            return Err(EngramError::UnsizedType {
                name: name.to_owned(),
                ty: info.ty.as_u32(),
            });
        };
        if !ne0.is_multiple_of(block) {
            return Err(EngramError::UnalignedRow {
                name: name.to_owned(),
                ne0,
                block,
            });
        }
        let row_bytes = ne0 / block * block_bytes * slab;
        if rows * row_bytes != info.nbytes {
            return Err(EngramError::Untiled {
                name: name.to_owned(),
                rows,
                row_bytes,
                product: rows * row_bytes,
                nbytes: info.nbytes,
            });
        }
        let page = page_size();
        let table = SplitTable {
            split,
            shard,
            info,
            path,
            row_bytes,
            rows,
            page,
            file: OnceLock::new(),
        };
        let (flag, op) = advice.madvise();
        table.view()?.advise_all(flag, op)?;
        Ok(table)
    }

    /// Rows in the table: the experts of a stack.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The shard file the table's bytes are in.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The page the table's rows round out to, in bytes.
    pub fn page_bytes(&self) -> u64 {
        self.page as u64
    }

    /// Row `id`'s place in the shard file: its first byte's offset and its
    /// length. For a reader that reads the same bytes without the mapping.
    pub fn file_range(&self, id: u32) -> Result<(u64, u64), EngramError> {
        self.view()?.row(id)?;
        let at = self.shard()?.data_base() + self.info.offset + u64::from(id) * self.row_bytes;
        Ok((at, self.row_bytes))
    }

    /// The pages `mincore` finds in the page cache among those `ids`' rows
    /// span, and the pages they span ([`PageCount`]). A row spans the whole
    /// pages that hold it, so two neighbours' shared page counts once for
    /// each. `mincore` reports the page cache of a file mapping to the file's
    /// owner or a caller that could write it (root on the box); anyone else
    /// reads pages this process mapped only, and a count of 0 from them says
    /// nothing about the cache.
    pub fn resident_pages(&self, ids: &[u32]) -> Result<PageCount, EngramError> {
        let (resident, total) = self.view()?.resident_pages(ids)?;
        Ok(PageCount { resident, total })
    }

    /// Drop `ids`' pages from this process and from the page cache, so the
    /// next read of them is a device read again: the whole pages of each row,
    /// the mapping's entries first (`posix_fadvise(DONTNEED)` skips a folio
    /// that is still mapped), then the cache. A cold path: the shard is
    /// opened for it on the first call. A page shared with a neighbouring
    /// row, or held mapped or locked by another process, may stay.
    pub fn evict_rows(&self, ids: &[u32]) -> Result<(), EngramError> {
        let view = self.view()?;
        let file = self.file()?;
        let base = self.shard()?.mapping().as_ptr() as usize;
        for &id in ids {
            let (first, span) = view.pages(id)?;
            view.advise_pages(first, span, libc::MADV_DONTNEED, "madvise(DONTNEED)")?;
            fadvise_dontneed(file, first - base, span)
                .map_err(|e| view.io("posix_fadvise(DONTNEED)", e))?;
        }
        Ok(())
    }

    /// The shard's reader, whose mapping holds the table.
    fn shard(&self) -> Result<&Gguf, EngramError> {
        self.split
            .shard(self.shard)
            .ok_or_else(|| EngramError::MissingTable {
                name: self.info.name.clone(),
            })
    }

    /// The shard file, opened once for the cache advice.
    fn file(&self) -> Result<&File, EngramError> {
        if let Some(f) = self.file.get() {
            return Ok(f);
        }
        let f = File::open(&self.path).map_err(|source| EngramError::SiteIo {
            name: self.info.name.clone(),
            path: self.path.clone(),
            op: "open",
            source,
        })?;
        Ok(self.file.get_or_init(|| f))
    }

    /// The table's bytes in the split's mapping, as the row reads see them.
    fn view(&self) -> Result<TableBytes<'_>, EngramError> {
        let bytes = self.shard()?.data(&self.info)?;
        Ok(TableBytes {
            name: &self.info.name,
            path: &self.path,
            bytes,
            row_bytes: self.row_bytes as usize,
            page: self.page,
        })
    }
}

impl RowTable for SplitTable {
    fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.view()?.prefetch(ids)
    }

    fn populate(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.view()?.populate(ids)
    }

    fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError> {
        self.view()?.resident_rows(ids)
    }

    fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        self.view()?.copy_rows(ids, out)
    }
}

impl RowTables for SplitTable {
    fn count(&self) -> usize {
        1
    }

    fn table(&self, i: usize) -> Option<&dyn RowTable> {
        (i == 0).then_some(self as &dyn RowTable)
    }
}
