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

use std::path::PathBuf;
use std::sync::Arc;

use gguf::{Split, TensorInfo};

use crate::{EngramError, RowTable, RowTables, TableBytes, page_size};

/// One row-gathered table of a [`Split`], read from the NVMe tier through
/// the split's own mapping.
pub struct SplitTable {
    split: Arc<Split>,
    shard: usize,
    info: TensorInfo,
    /// The shard file, for the refusals.
    path: PathBuf,
    row_bytes: u64,
    /// `sysconf(_SC_PAGESIZE)`, read once.
    page: usize,
}

impl SplitTable {
    /// The 2-D table `name` of `split`: whole blocks a row, its rows tiling
    /// the bytes the header states, each refused by name otherwise; then
    /// `MADV_RANDOM` over its pages in the split's mapping, so a fault there
    /// reads its own page, not the 128 KiB around it. Load-time only, and
    /// only for a table the plan left on the NVMe tier: a table the host set
    /// holds is read through the mapping as it is.
    pub fn open(split: Arc<Split>, name: &str) -> Result<SplitTable, EngramError> {
        let missing = || EngramError::MissingTable {
            name: name.to_owned(),
        };
        let (shard, info) = split
            .find(name)
            .map(|(s, t)| (s, t.clone()))
            .ok_or_else(missing)?;
        let path = split.shard_path(shard).ok_or_else(missing)?.to_path_buf();
        let [ne0, ne1] = info.dims[..] else {
            return Err(EngramError::NotATable {
                name: name.to_owned(),
                dims: info.dims.clone(),
            });
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
        let row_bytes = ne0 / block * block_bytes;
        if ne1 * row_bytes != info.nbytes {
            return Err(EngramError::Untiled {
                name: name.to_owned(),
                rows: ne1,
                row_bytes,
                product: ne1 * row_bytes,
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
            page,
        };
        table
            .view()?
            .advise_all(libc::MADV_RANDOM, "madvise(RANDOM)")?;
        Ok(table)
    }

    /// The table's bytes in the split's mapping, as the row reads see them.
    fn view(&self) -> Result<TableBytes<'_>, EngramError> {
        let bytes = self
            .split
            .shard(self.shard)
            .ok_or_else(|| EngramError::MissingTable {
                name: self.info.name.clone(),
            })?
            .data(&self.info)?;
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
