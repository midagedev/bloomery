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
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use gguf::{Gguf, Split, TensorInfo};

use crate::{
    EngramError, RowTable, RowTables, TableBytes, adjacent_runs, fadvise_dontneed, page_size,
};

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
    /// next read of them is a device read again: one eviction per run of
    /// adjacent rows over the whole pages that hold it, the mapping's entries
    /// first (`posix_fadvise(DONTNEED)` skips a folio that is still mapped),
    /// then the cache. An id past the table is refused by name before
    /// anything is dropped. A cold path: the shard is opened for it on the
    /// first call. A page a run shares with a row or bytes outside it may
    /// stay, with the rest of its folio, as may a page another process holds
    /// mapped or locked.
    pub fn evict_rows(&self, ids: &[u32]) -> Result<(), EngramError> {
        let view = self.view()?;
        let spans = adjacent_runs(ids)
            .into_iter()
            .map(|(first, last)| view.run_pages(first, last))
            .collect::<Result<Vec<_>, _>>()?;
        let file = self.file()?;
        let base = self.shard()?.mapping().as_ptr() as usize;
        for (first, span) in spans {
            view.advise_pages(first, span, libc::MADV_DONTNEED, "madvise(DONTNEED)")?;
            fadvise_dontneed(file, first - base, span)
                .map_err(|e| view.io("posix_fadvise(DONTNEED)", e))?;
        }
        Ok(())
    }

    /// Drop this split's mapping entries and page cache for the whole shard,
    /// including folios that cross a table or row boundary. Pages another
    /// process maps or locks may stay resident.
    pub fn evict_shard(&self) -> Result<(), EngramError> {
        let view = self.view()?;
        let file = self.file()?;
        let map = self.shard()?.mapping();
        view.advise_pages(
            map.as_ptr() as usize,
            map.len().div_ceil(self.page) * self.page,
            libc::MADV_DONTNEED,
            "madvise(DONTNEED)",
        )?;
        fadvise_dontneed(file, 0, 0).map_err(|e| view.io("posix_fadvise(DONTNEED)", e))
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

/// A row-gathered table held as runs of whole rows in files of its own, in row order (a
/// checkpoint folder's PLE table, each part a tensor of the folder's shards), read from the
/// NVMe tier as [`SplitTable`] reads a split's table: one read-only mapping a file, and the
/// same advice, residency query and copy over each run's bytes.
pub struct RunTable {
    name: String,
    /// Each file's path and mapping, in the order the runs first name them.
    files: Vec<(PathBuf, memmap2::Mmap)>,
    /// Each run's file, its first byte in that file, and the table's first row past it.
    runs: Vec<(usize, usize, u64)>,
    row_bytes: u64,
    page: usize,
}

impl RunTable {
    /// The table `name` of `row_bytes`-byte rows held as `runs`, in row order: each a file and
    /// the byte range of whole rows it holds there. Each file is mapped once, read-only, under
    /// `advice`. A table of no rows, of empty rows or past `u32` row ids, a run that is not whole
    /// rows and a run past its file's end are refused by name before any advice.
    pub fn open(
        name: &str,
        row_bytes: u64,
        runs: &[(PathBuf, Range<u64>)],
        advice: PageAdvice,
    ) -> Result<RunTable, EngramError> {
        let rows_err = |rows: u64| EngramError::RunRows {
            name: name.to_owned(),
            rows,
            row_bytes,
        };
        if row_bytes == 0 {
            return Err(rows_err(0));
        }
        let mut files: Vec<(PathBuf, memmap2::Mmap)> = Vec::new();
        let mut placed = Vec::with_capacity(runs.len());
        let mut end = 0u64;
        for (path, at) in runs {
            let file = match files.iter().position(|(p, _)| p == path) {
                Some(file) => file,
                None => {
                    let io = |op, source| EngramError::SiteIo {
                        name: name.to_owned(),
                        path: path.clone(),
                        op,
                        source,
                    };
                    let f = File::open(path).map_err(|e| io("open", e))?;
                    // SAFETY: read-only, and nothing maps the file writable: the contract
                    // `gguf::Gguf::open_backed` accepts for a split's shards.
                    let map = unsafe { memmap2::Mmap::map(&f) }.map_err(|e| io("mmap", e))?;
                    files.push((path.clone(), map));
                    files.len() - 1
                }
            };
            let len = files[file].1.len() as u64;
            let nbytes = at.end.saturating_sub(at.start);
            if at.end > len || at.start > at.end {
                return Err(EngramError::PastEndOfFile {
                    name: name.to_owned(),
                    path: path.clone(),
                    base: 0,
                    off: at.start,
                    nbytes,
                    len,
                });
            }
            if !nbytes.is_multiple_of(row_bytes) {
                return Err(EngramError::Untiled {
                    name: name.to_owned(),
                    rows: nbytes / row_bytes,
                    row_bytes,
                    product: nbytes / row_bytes * row_bytes,
                    nbytes,
                });
            }
            // Inside a mapping of the file, so inside usize.
            let start = at.start as usize;
            end += nbytes / row_bytes;
            placed.push((file, start, end));
        }
        if end == 0 || u32::try_from(end - 1).is_err() {
            return Err(rows_err(end));
        }
        let table = RunTable {
            name: name.to_owned(),
            files,
            runs: placed,
            row_bytes,
            page: page_size(),
        };
        let (flag, op) = advice.madvise();
        for i in 0..table.runs.len() {
            table.view(i).advise_all(flag, op)?;
        }
        Ok(table)
    }

    /// Rows in the table.
    pub fn rows(&self) -> u64 {
        self.runs.last().map_or(0, |r| r.2)
    }

    /// Run `i`'s bytes as the row reads see them.
    fn view(&self, i: usize) -> TableBytes<'_> {
        let (file, start, end) = self.runs[i];
        let first = i.checked_sub(1).map_or(0, |p| self.runs[p].2);
        let len = ((end - first) * self.row_bytes) as usize;
        let (path, map) = &self.files[file];
        TableBytes {
            name: &self.name,
            path,
            bytes: &map[start..start + len],
            row_bytes: self.row_bytes as usize,
            page: self.page,
        }
    }

    /// `ids` cut into stretches that lie in one run, in order: each run, the stretch's ids as
    /// that run's own rows, and where the stretch starts in `ids`. An id past the table is
    /// refused by name.
    fn stretches(&self, ids: &[u32]) -> Result<Vec<(usize, Vec<u32>, usize)>, EngramError> {
        let mut out: Vec<(usize, Vec<u32>, usize)> = Vec::new();
        for (at, &id) in ids.iter().enumerate() {
            let run = self.runs.partition_point(|r| r.2 <= u64::from(id));
            if run == self.runs.len() {
                return Err(EngramError::RowOutOfRange {
                    name: self.name.clone(),
                    id,
                    rows: self.rows(),
                });
            }
            let first = run.checked_sub(1).map_or(0, |p| self.runs[p].2);
            // Below the run's end, which `open` held inside u32.
            let own = (u64::from(id) - first) as u32;
            match out.last_mut() {
                Some((r, own_ids, _)) if *r == run => own_ids.push(own),
                _ => out.push((run, vec![own], at)),
            }
        }
        Ok(out)
    }

    /// An order-independent `op` over each file's rows. Repeated ids stay repeated.
    fn each(
        &self,
        ids: &[u32],
        mut op: impl FnMut(TableBytes<'_>, &[u32]) -> Result<(), EngramError>,
    ) -> Result<(), EngramError> {
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        for (run, own, _) in self.stretches(&sorted)? {
            op(self.view(run), &own)?;
        }
        Ok(())
    }
}

impl RowTable for RunTable {
    fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    fn prefetch(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.each(ids, |v, own| v.prefetch(own))
    }

    fn populate(&self, ids: &[u32]) -> Result<(), EngramError> {
        self.each(ids, |v, own| v.populate(own))
    }

    fn resident_rows(&self, ids: &[u32]) -> Result<u64, EngramError> {
        let mut n = 0;
        self.each(ids, |v, own| {
            n += v.resident_rows(own)?;
            Ok(())
        })?;
        Ok(n)
    }

    fn copy_rows(&self, ids: &[u32], out: &mut [u8]) -> Result<(), EngramError> {
        let rb = self.row_bytes as usize;
        if out.len() != ids.len() * rb {
            return Err(EngramError::OutBufferSize {
                name: self.name.clone(),
                rows: ids.len(),
                want: ids.len() * rb,
                got: out.len(),
            });
        }
        for (run, own, at) in self.stretches(ids)? {
            let span = at * rb..(at + own.len()) * rb;
            self.view(run).copy_rows(&own, &mut out[span])?;
        }
        Ok(())
    }
}

impl RowTables for RunTable {
    fn count(&self) -> usize {
        1
    }

    fn table(&self, i: usize) -> Option<&dyn RowTable> {
        (i == 0).then_some(self as &dyn RowTable)
    }
}

#[cfg(test)]
mod tests {
    use super::{PageAdvice, RunTable};
    use crate::{EngramError, RowTable};

    /// Rows read across runs in two files come back in the ids' order, each run's own; an id
    /// past the table, a run past its file and a run of part of a row are refused by name.
    #[test]
    fn a_run_table_reads_each_row_from_its_run() {
        let dir = std::env::temp_dir().join(format!("engram-runs-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a"), dir.join("b"));
        // Rows of 4 bytes, every byte the row's id; file a holds rows 0..2 after 3 bytes of
        // something else, file b rows 2..5 and then rows 5..6 after one row's gap.
        let row = |id: u8| [id; 4];
        let mut fa = vec![9u8; 3];
        fa.extend(row(0).iter().chain(&row(1)));
        let mut fb: Vec<u8> = [2, 3, 4].iter().flat_map(|&i| row(i)).collect();
        fb.extend([7u8; 4].iter().chain(&row(5)));
        std::fs::write(&a, &fa).unwrap();
        std::fs::write(&b, &fb).unwrap();
        let run = |path: &std::path::Path, offset: u64, rows: u64| {
            (path.to_path_buf(), offset..offset + rows * 4)
        };
        let runs = [run(&a, 3, 2), run(&b, 0, 3), run(&b, 16, 1)];
        let t = RunTable::open("t", 4, &runs, PageAdvice::Random).unwrap();
        assert_eq!(t.rows(), 6);
        let ids = [5, 0, 1, 2, 4, 3, 1];
        let mut out = vec![0u8; ids.len() * 4];
        t.copy_rows(&ids, &mut out).unwrap();
        let want: Vec<u8> = ids.iter().flat_map(|&i| row(i as u8)).collect();
        assert_eq!(out, want);
        t.prefetch(&ids).unwrap();
        let mut groups = Vec::new();
        t.each(&ids, |_, rows| {
            groups.push(rows.to_vec());
            Ok(())
        })
        .unwrap();
        assert_eq!(groups, [vec![0, 1, 1], vec![0, 1, 2], vec![0]]);
        assert!(matches!(
            t.copy_rows(&[6], &mut [0; 4]),
            Err(EngramError::RowOutOfRange { id: 6, rows: 6, .. })
        ));
        let past = RunTable::open("t", 4, &[run(&b, 16, 2)], PageAdvice::Random);
        assert!(matches!(past, Err(EngramError::PastEndOfFile { .. })));
        let torn = RunTable::open("t", 4, &[(a, 3..9)], PageAdvice::Random);
        assert!(matches!(torn, Err(EngramError::Untiled { .. })));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
