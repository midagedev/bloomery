//! The row's d1k reference set, opened through its family's check
//! (`refset::arch::<arch>::fixture::IK`). The self-bit clauses read its
//! prompt ids. A set dumped from another file, another generation of the
//! fixture or another ik build is refused by name.
//!
//! The set's numeric tensors were the GGUF oracle.
//! PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified

use refset::ik::RefManifest;

/// The row's d1k set, checked against its family.
pub fn d1k() -> Result<RefManifest, String> {
    let row = super::row::current();
    RefManifest::open(&row.ik.path(row.d1k), row.ik).map_err(|e| format!("{}: {e}", row.d1k))
}
