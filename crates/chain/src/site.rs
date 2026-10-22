//! A weight site's type: the launch family a dense projection of a file's type runs, read once at
//! load from the file's header ([`file_site`], before any byte is uploaded). A type no launch here
//! reads is refused by name, with the site.
//!
//! Which activation form a site reads is [`SiteTy::reads`]: a body quantizes a unit's rows into
//! exactly the forms its sites read. The launches themselves are a backend's.

use crate::ChainError;
use gguf::Split;
use gguf::card::GemmEntry;
use gguf::quant::GgmlType;
use model::placement::kernels::{self, SiteAct, SiteEntry};
use std::fmt;

/// The launch family a site's file type selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteTy {
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    /// Q5_0, read in blocks of 32 values; wide, the kernel table's GEMM of its own.
    Q5_0,
    Q8_0,
    F32,
    /// NVFP4 in its two planes and the tensor's scale; no wide GEMM.
    Nvfp4,
    /// bloomery's MXFP8 in its two planes, read on the f32 rows, narrow and wide.
    Mxfp8,
    /// bloomery's FP8_B128 in the same two planes, read as MXFP8 is.
    Fp8B128,
    /// BF16 in its words, read on the f32 rows, narrow and wide.
    Bf16,
}

impl SiteTy {
    /// Every family, the range of [`SiteTy::of_ggml`].
    pub const ALL: &'static [SiteTy] = &[
        SiteTy::Q3K,
        SiteTy::Q4K,
        SiteTy::Q5K,
        SiteTy::Q6K,
        SiteTy::Q5_0,
        SiteTy::Q8_0,
        SiteTy::F32,
        SiteTy::Nvfp4,
        SiteTy::Mxfp8,
        SiteTy::Fp8B128,
        SiteTy::Bf16,
    ];

    /// The family of a file tensor of type `ty`, the one that launches the
    /// kernel table's dense entry for it (`kernels::dense`); a type with no
    /// dense entry that loads widened to f32 (F16) is the f32 site of its
    /// widened rows; `None` for a type no launch here reads.
    #[must_use]
    pub fn of_ggml(ty: GgmlType) -> Option<SiteTy> {
        match kernels::dense(ty) {
            Some(entry) => SiteTy::of_entry(entry),
            None if gguf::card::CardFormat::of(ty) == Some(gguf::card::CardFormat::Bf16AsF32) => {
                Some(SiteTy::F32)
            }
            None => None,
        }
    }

    /// The family that launches the kernel table's dense entry `entry`;
    /// `None` for an entry no family here launches.
    #[must_use]
    pub fn of_entry(entry: SiteEntry) -> Option<SiteTy> {
        SiteTy::ALL.iter().copied().find(|s| s.entry() == entry)
    }

    /// The kernel table's dense entry this family launches: the one map
    /// between the two names.
    const fn entry(self) -> SiteEntry {
        match self {
            SiteTy::Q3K => SiteEntry::Q3k,
            SiteTy::Q4K => SiteEntry::Q4k,
            SiteTy::Q5K => SiteEntry::Q5k,
            SiteTy::Q6K => SiteEntry::Q6k,
            SiteTy::Q5_0 => SiteEntry::Q5_0,
            SiteTy::Q8_0 => SiteEntry::Q8_0,
            SiteTy::F32 => SiteEntry::F32,
            SiteTy::Nvfp4 => SiteEntry::Nvfp4,
            SiteTy::Mxfp8 => SiteEntry::Mxfp8,
            SiteTy::Fp8B128 => SiteEntry::Fp8B128,
            SiteTy::Bf16 => SiteEntry::Bf16,
        }
    }

    /// The file type a site of this family holds: the type whose row names
    /// its entry (`SiteEntry::ty`).
    #[must_use]
    pub fn ggml(self) -> GgmlType {
        self.entry().ty()
    }

    /// Whether the type is one of the four K-quants: the kernel table runs
    /// its stacks wide through the K-quant grouped GEMM (`kernels::gemm`).
    #[must_use]
    pub fn kquant(self) -> bool {
        matches!(
            kernels::gemm(self.ggml()),
            Some(GemmEntry::Q3k | GemmEntry::Q4k | GemmEntry::Q5k | GemmEntry::Q6k)
        )
    }

    /// The activation form the site's launch reads (`SiteEntry::reads`, the kernel table's fact).
    #[must_use]
    pub const fn reads(self) -> SiteAct {
        self.entry().reads()
    }
}

/// The family's file type, its ggml name in upper case (`Q4_K`, `NVFP4`).
impl fmt::Display for SiteTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.ggml().to_string().to_ascii_uppercase())
    }
}

/// The type of site `name` in `file`, a matrix of `rows` rows (every expert's
/// of a stack) of `k` values, read from its header entry before any upload:
/// a type outside `allowed`, another shape or a missing tensor is refused by
/// name with the site.
pub fn file_site(
    file: &Split,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, ChainError> {
    let (_, t) = file
        .find(name)
        .ok_or_else(|| ChainError::tensor(what, name, "in the file"))?;
    let got_k = t.dims.first().copied().unwrap_or(0);
    let got_rows: u64 = t.dims.iter().skip(1).product();
    if usize::try_from(got_k).ok() != Some(k) || usize::try_from(got_rows).ok() != Some(rows) {
        return Err(ChainError::shape(
            what,
            format!(
                "{name} is {:?} in the file; the site takes {rows} rows of {k}",
                t.dims
            ),
        ));
    }
    match SiteTy::of_ggml(t.ty).filter(|ty| allowed.contains(ty)) {
        Some(ty) => Ok(ty),
        None => Err(ChainError::shape(
            what,
            format!(
                "{name} is {} in the file; this site launches {}",
                t.ty,
                list(allowed)
            ),
        )),
    }
}

/// `allowed` as a list of names, the one a refusal of a site of the wrong type names.
pub fn list(allowed: &[SiteTy]) -> String {
    allowed
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::SiteTy;

    /// The kernel table's K-quant GEMM entries name exactly the four K-quant
    /// families: a table row that moves a type in or out of them moves
    /// [`SiteTy::kquant`] with it, and this names the change.
    #[test]
    fn the_kquant_sites_are_the_four_kquants() {
        let kq: Vec<SiteTy> = SiteTy::ALL.iter().copied().filter(|s| s.kquant()).collect();
        assert_eq!(kq, [SiteTy::Q3K, SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K]);
    }
}
