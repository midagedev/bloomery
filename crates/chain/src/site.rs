//! A weight site's type: the launch family a dense projection of a file's type runs, read once at
//! load from the file's header ([`file_site`], before any byte is uploaded). A type no launch here
//! reads is refused by name, with the site. The card's families come from the kernel table; a
//! backend's own ([`BACKEND_SITES`]) only a backend's list admits.
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
    /// NVFP4 in its two planes and the tensor's scale; its gemv reads the q8_1 rows, its wide GEMM
    /// the q8 blocks of 32.
    Nvfp4,
    /// bloomery's MXFP8 in its two planes, read on the f32 rows, narrow and wide.
    Mxfp8,
    /// bloomery's FP8_B128 in the same two planes, read as MXFP8 is.
    Fp8B128,
    /// BF16 in its words, read on the f32 rows, narrow and wide.
    Bf16,
    /// MLX's affine groups — 4-bit (`MLX_A4`) and 8-bit (`MLX_A8`), their F16
    /// pairs (`MLX_A4H`, `MLX_A8H`): a family no card launch reads, a
    /// backend's mat-vec (Metal's). Outside the card kernel table, so
    /// [`SiteTy::ALL`] does not hold it, no card list names it and its dense
    /// entry is none; a backend whose launch reads it names it in the list
    /// [`file_site`] takes.
    MlxA4,
    MlxA8,
    MlxA4h,
    MlxA8h,
    /// IQ4_XS in its original blocks, over q8_1 columns, narrow and wide.
    Iq4Xs,
}

impl SiteTy {
    /// Every family the card launches, the range of [`SiteTy::of_ggml`]. A
    /// family outside the card kernel table ([`BACKEND_SITES`]) is not here,
    /// so no card list holds it; a backend's list alone admits it.
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
        SiteTy::Iq4Xs,
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
        SiteTy::ALL
            .iter()
            .copied()
            .find(|s| s.entry() == Some(entry))
    }

    /// The kernel table's dense entry this family launches: the one map
    /// between the two names; `None` for a family outside the card kernel
    /// table, whose [`BACKEND_SITES`] row answers for it.
    const fn entry(self) -> Option<SiteEntry> {
        match self {
            SiteTy::Q3K => Some(SiteEntry::Q3k),
            SiteTy::Q4K => Some(SiteEntry::Q4k),
            SiteTy::Q5K => Some(SiteEntry::Q5k),
            SiteTy::Q6K => Some(SiteEntry::Q6k),
            SiteTy::Q5_0 => Some(SiteEntry::Q5_0),
            SiteTy::Q8_0 => Some(SiteEntry::Q8_0),
            SiteTy::F32 => Some(SiteEntry::F32),
            SiteTy::Nvfp4 => Some(SiteEntry::Nvfp4),
            SiteTy::Mxfp8 => Some(SiteEntry::Mxfp8),
            SiteTy::Fp8B128 => Some(SiteEntry::Fp8B128),
            SiteTy::Bf16 => Some(SiteEntry::Bf16),
            SiteTy::Iq4Xs => Some(SiteEntry::Iq4xs),
            SiteTy::MlxA4 | SiteTy::MlxA8 | SiteTy::MlxA4h | SiteTy::MlxA8h => None,
        }
    }

    /// The file type a site of this family holds: the type whose row names
    /// its entry (`SiteEntry::ty`), or, for a family outside the card kernel
    /// table, its [`BACKEND_SITES`] row's.
    #[must_use]
    pub fn ggml(self) -> GgmlType {
        match self.entry() {
            Some(entry) => entry.ty(),
            None => self.row().0,
        }
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

    /// The activation form the site's launch reads (`SiteEntry::reads`, the
    /// kernel table's fact), or the backend row's for a family outside it.
    #[must_use]
    pub fn reads(self) -> SiteAct {
        match self.entry() {
            Some(entry) => entry.reads(),
            None => self.row().2,
        }
    }

    /// This family's [`BACKEND_SITES`] row; a card family has none. A family
    /// outside the card kernel table whose table holds no row is a named
    /// panic, not a guessed type.
    fn row(self) -> (GgmlType, SiteTy, SiteAct) {
        BACKEND_SITES
            .iter()
            .copied()
            .find(|r| r.1 == self)
            .expect("a site family outside the card kernel table has no BACKEND_SITES row")
    }
}

/// The site families outside the card kernel table: each a file type no card
/// launch reads, with the family a backend's launch reads it at and the
/// activation form that launch reads. One row a type; [`file_site`] alone
/// reads the table, so a card's `allowed` — which holds no family here —
/// refuses the type by name exactly as it did before the row, and
/// [`SiteTy::of_ggml`], the card table's map, names each of the types none.
pub const BACKEND_SITES: &[(GgmlType, SiteTy, SiteAct)] = &[
    (GgmlType::MLX_A4, SiteTy::MlxA4, SiteAct::F32),
    (GgmlType::MLX_A8, SiteTy::MlxA8, SiteAct::F32),
    (GgmlType::MLX_A4H, SiteTy::MlxA4h, SiteAct::F32),
    (GgmlType::MLX_A8H, SiteTy::MlxA8h, SiteAct::F32),
];

/// The family's file type, its ggml name in upper case (`Q4_K`, `NVFP4`).
impl fmt::Display for SiteTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.ggml().to_string().to_ascii_uppercase())
    }
}

/// The header entry of `name` in `file`, a matrix of `rows` rows of `k`
/// values: a missing tensor or another shape is refused by name.
pub(crate) fn shaped<'a>(
    file: &'a Split,
    what: &'static str,
    name: &str,
    (rows, k): (usize, usize),
) -> Result<&'a gguf::TensorInfo, ChainError> {
    let (_, t) = file
        .find(name)
        .ok_or_else(|| ChainError::tensor(what, name, "in the file"))?;
    let got_k = t.dims.first().copied().unwrap_or(0);
    let got_rows: u64 = t.dims.iter().skip(1).product();
    if usize::try_from(got_k).ok() != Some(k) || usize::try_from(got_rows).ok() != Some(rows) {
        return Err(ChainError::shape(
            what,
            format!(
                "{name} is {:?} in the file; it takes {rows} rows of {k}",
                t.dims
            ),
        ));
    }
    Ok(t)
}

/// The type of site `name` in `file`, a matrix of `rows` rows (every expert's
/// of a stack) of `k` values, read from its header entry before any upload:
/// a type outside `allowed`, another shape or a missing tensor is refused by
/// name with the site. The family comes from [`SiteTy::of_ggml`] or, for a
/// type no card launch reads, [`BACKEND_SITES`], so a backend whose list
/// names such a family reads it while a card's list refuses it by name.
pub fn file_site(
    file: &Split,
    what: &'static str,
    name: &str,
    shape: (usize, usize),
    allowed: &[SiteTy],
) -> Result<SiteTy, ChainError> {
    let t = shaped(file, what, name, shape)?;
    match family_of(t.ty).filter(|ty| allowed.contains(ty)) {
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

/// The family of a file tensor of type `ty`: [`SiteTy::of_ggml`]'s, or its
/// [`BACKEND_SITES`] row's for a type no card launch reads. The one map a
/// site's family comes through.
fn family_of(ty: GgmlType) -> Option<SiteTy> {
    SiteTy::of_ggml(ty).or_else(|| BACKEND_SITES.iter().find(|r| r.0 == ty).map(|r| r.1))
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
    use super::{BACKEND_SITES, GgmlType, SiteTy, family_of};

    /// The kernel table's K-quant GEMM entries name exactly the four K-quant
    /// families: a table row that moves a type in or out of them moves
    /// [`SiteTy::kquant`] with it, and this names the change.
    #[test]
    fn the_kquant_sites_are_the_four_kquants() {
        let kq: Vec<SiteTy> = SiteTy::ALL.iter().copied().filter(|s| s.kquant()).collect();
        assert_eq!(kq, [SiteTy::Q3K, SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K]);
    }

    /// Each new MLX row maps both ways: its file type reaches the family
    /// through [`family_of`] (the map [`file_site`] reads), and the family's
    /// file type is the type again. One test a row, so a row that moves or
    /// loses its type is named.
    #[test]
    fn mlx_a8_is_its_row_both_ways() {
        assert_eq!(family_of(GgmlType::MLX_A8), Some(SiteTy::MlxA8));
        assert_eq!(SiteTy::MlxA8.ggml(), GgmlType::MLX_A8);
    }

    #[test]
    fn mlx_a4h_is_its_row_both_ways() {
        assert_eq!(family_of(GgmlType::MLX_A4H), Some(SiteTy::MlxA4h));
        assert_eq!(SiteTy::MlxA4h.ggml(), GgmlType::MLX_A4H);
    }

    #[test]
    fn mlx_a8h_is_its_row_both_ways() {
        assert_eq!(family_of(GgmlType::MLX_A8H), Some(SiteTy::MlxA8h));
        assert_eq!(SiteTy::MlxA8h.ggml(), GgmlType::MLX_A8H);
    }

    /// A family outside the card kernel table is its row alone: no card list
    /// holds it ([`SiteTy::ALL`]), it maps no dense entry, its file type and
    /// activation form are its row's, and [`SiteTy::of_ggml`] — the card
    /// table's map — names its type none, so a card `allowed` refuses the
    /// type by name exactly as before the row existed.
    #[test]
    fn a_backend_family_is_its_row_alone() {
        for &(ty, family, act) in BACKEND_SITES {
            assert!(family.entry().is_none(), "{family} has a card entry");
            assert!(!SiteTy::ALL.contains(&family), "{family} is a card list's");
            assert_eq!(family.ggml(), ty, "{family}");
            assert_eq!(family.to_string(), ty.to_string().to_ascii_uppercase());
            assert_eq!(family.reads(), act, "{family}");
            assert!(!family.kquant(), "{family}");
            assert_eq!(SiteTy::of_ggml(ty), None, "{ty} has a card family");
        }
    }
}
