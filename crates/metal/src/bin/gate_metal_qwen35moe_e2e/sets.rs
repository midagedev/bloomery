//! The row's prompt ids: the first ids of its d1k reference set (opened
//! through its family's check, `refset::arch::<arch>::fixture::IK`; a set
//! dumped from another file, another generation of the fixture or another ik
//! build is refused by name), or the ids a row of another vocabulary
//! generates. The self-bit clauses read them.
//!
//! The set's numeric tensors were the GGUF oracle.
//! PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified

use refset::ik::RefManifest;

use super::row::Prompt;

/// The row's d1k set, checked against its family.
fn d1k(ik: &'static refset::family::Family, set: &str) -> Result<RefManifest, String> {
    RefManifest::open(&ik.path(set), ik).map_err(|e| format!("{set}: {e}"))
}

/// The multiplier and increment of the generator's 64-bit linear congruence
/// (Knuth's MMIX constants): the ids it draws are a function of the row's
/// numbers alone.
const MUL: u64 = 6_364_136_223_846_793_005;
const INC: u64 = 1_442_695_040_888_963_407;

/// `n` ids: `first`, then each the high bits of the next state of a linear
/// congruence from `first`, mapped into `lo..hi`.
fn generated(n: usize, first: u32, lo: u32, hi: u32) -> Result<Vec<u32>, String> {
    if lo >= hi {
        return Err(format!("the generated prompt's range {lo}..{hi} is empty"));
    }
    let span = u64::from(hi - lo);
    let mut state = u64::from(first);
    let mut ids = Vec::with_capacity(n);
    ids.push(first);
    while ids.len() < n {
        state = state.wrapping_mul(MUL).wrapping_add(INC);
        let id = u64::from(lo) + (state >> 33) % span;
        ids.push(u32::try_from(id).map_err(|_| "a generated id does not fit u32".to_string())?);
    }
    ids.truncate(n);
    Ok(ids)
}

/// The ids of a prompt of `n` ids for the current row.
pub fn prompt_ids(n: usize) -> Result<Vec<u32>, String> {
    match &super::row::current().prompt {
        Prompt::D1k { ik, set } => {
            let man = d1k(ik, set)?;
            let (_, _, prompt) = man
                .step()
                .map_err(|e| format!("the d1k set's tokens: {e}"))?;
            prompt
                .get(..n)
                .map(<[u32]>::to_vec)
                .ok_or_else(|| format!("the d1k set holds {} prompt ids, not {n}", prompt.len()))
        }
        Prompt::Generated { first, lo, hi } => generated(n, *first, *lo, *hi),
    }
}
