//! Reading the device code a gate binary carries. `cargo oxide` embeds the
//! PTX text of the whole `bloomery-gpu` bundle in the executable's `.oxart`
//! ELF section, so a kernel's compiled shape is a byte scan of
//! `/proc/self/exe` — no device, no clock, and the same answer the backend
//! would report.
//!
//! Two root causes were found this way and both are now asserted from here:
//! a per-lane accumulator array spilled to a local depot (round-trip per
//! multiply-add), and a launch geometry that left one warp resident for a
//! whole row. `gate_p5::no_local_depot` asserts the depot counts of the
//! flash kernels; `gate_p4::norm_geometry` asserts `rms_norm`/`norm_quant`
//! and `argmax` against the block width their host side launches.
//!
//! Spelling matters more than it looks: PTX writes a fused multiply-add as
//! `fma.rn.f32` (and `fma.rm.f32`), never `fma.f32`, so [`Counts::fma`]
//! counts the `fma.` prefix. A needle that matches nothing returns zero and
//! reads exactly like a clean kernel.

/// Byte offset of `needle` in `hay`.
#[must_use]
pub fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Occurrences of `needle` in `hay`, overlapping included.
#[must_use]
pub fn count(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || needle.len() > hay.len() {
        return 0;
    }
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

/// The marker every kernel's PTX declaration opens with.
const ENTRY: &[u8] = b".visible .entry ";

/// The PTX body of one `.visible .entry`: from just past its `name(` to the
/// next entry's declaration, or to the end of the bundle for the last one.
/// `None` when the bundle holds no entry of that name.
#[must_use]
pub fn body<'a>(blob: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let head = format!(".visible .entry {name}(");
    let start = find(blob, head.as_bytes())? + head.len();
    let rest = &blob[start..];
    Some(match find(rest, ENTRY) {
        Some(end) => &rest[..end],
        None => rest,
    })
}

/// Every entry name in the bundle, in the order the text declares them.
#[must_use]
pub fn entries(blob: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while let Some(off) = find(&blob[at..], ENTRY) {
        let name_at = at + off + ENTRY.len();
        let name: Vec<u8> = blob[name_at..]
            .iter()
            .copied()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'$')
            .collect();
        at = name_at;
        match String::from_utf8(name) {
            Ok(n) if !n.is_empty() => out.push(n),
            _ => {}
        }
    }
    out
}

/// The `x` of an entry body's `.reqntid x, y, z` — the block width the
/// device code was compiled for. `None` when the entry declares none.
#[must_use]
pub fn reqntid(body: &[u8]) -> Option<usize> {
    let at = find(body, b".reqntid ")? + b".reqntid ".len();
    let digits: Vec<u8> = body[at..]
        .iter()
        .copied()
        .take_while(u8::is_ascii_digit)
        .collect();
    String::from_utf8(digits).ok()?.parse().ok()
}

/// What one entry's compiled body carries. `depot` is the register spill
/// the backend names `__local_depot<n>`; `ld_local`/`st_local` are the
/// traffic it costs. `fma` and `cvt_f16` are shape, not verdict: they say
/// whether the body multiplies-and-adds in one instruction and whether it
/// widens `f16` in hardware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts {
    pub name: String,
    pub reqntid: Option<usize>,
    pub depot: bool,
    pub ld_local: usize,
    pub st_local: usize,
    pub fma: usize,
    pub cvt_f16: usize,
}

/// Read one entry's counts out of the bundle.
#[must_use]
pub fn counts(blob: &[u8], name: &str) -> Option<Counts> {
    let b = body(blob, name)?;
    Some(Counts {
        name: name.to_string(),
        reqntid: reqntid(b),
        depot: find(b, b"__local_depot").is_some(),
        ld_local: count(b, b"ld.local"),
        st_local: count(b, b"st.local"),
        fma: count(b, b"fma."),
        cvt_f16: count(b, b"cvt.f32.f16"),
    })
}

/// Every entry of the bundle, sorted so the ones carrying a local depot come
/// first and names ascend inside each group — the order a reader wants,
/// because a depot is the defect and the rest is context.
#[must_use]
pub fn scan(blob: &[u8]) -> Vec<Counts> {
    let mut names = entries(blob);
    names.sort();
    names.dedup();
    let mut out: Vec<Counts> = names.iter().filter_map(|n| counts(blob, n)).collect();
    out.sort_by(|a, b| b.depot.cmp(&a.depot).then_with(|| a.name.cmp(&b.name)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bundle shaped like the real one: three entries, the middle one
    /// spilling, the last one with no `.reqntid` at all.
    const BLOB: &[u8] = b"//
.visible .entry alpha(
  .param .u64 alpha_param_0
)
.reqntid 256, 1, 1
{
  fma.rn.f32 %f1, %f2, %f3, %f4;
  cvt.f32.f16 %f5, %rs1;
  cvt.f32.f16 %f6, %rs2;
  ret;
}
.visible .entry beta(
  .param .u64 beta_param_0
)
.reqntid 32, 1, 1
{
  .local .align 4 .b8 __local_depot1[96];
  st.local.b32 [%rd1], %r1;
  st.local.b32 [%rd2], %r2;
  ld.local.b32 %r3, [%rd1];
  fma.rm.f32 %f1, %f2, %f3, %f4;
  ret;
}
.visible .entry gamma(
  .param .u64 gamma_param_0
)
{
  ret;
}
";

    #[test]
    fn lists_every_entry_in_declaration_order() {
        assert_eq!(entries(BLOB), ["alpha", "beta", "gamma"]);
    }

    /// An entry's body must stop at the next entry, or the counts of the
    /// whole tail land on the first kernel scanned.
    #[test]
    fn body_stops_at_the_next_entry() {
        let a = body(BLOB, "alpha").expect("alpha");
        assert!(find(a, b"__local_depot").is_none());
        assert_eq!(count(a, b"ld.local"), 0);
        let b = body(BLOB, "beta").expect("beta");
        assert!(find(b, b"gamma").is_none());
        assert_eq!(body(BLOB, "delta"), None);
    }

    #[test]
    fn reads_the_block_width_and_its_absence() {
        assert_eq!(reqntid(body(BLOB, "alpha").unwrap()), Some(256));
        assert_eq!(reqntid(body(BLOB, "beta").unwrap()), Some(32));
        assert_eq!(reqntid(body(BLOB, "gamma").unwrap()), None);
    }

    /// The counters, including the spelling trap: `fma.f32` matches nothing
    /// in real PTX, so the prefix is what the counter uses.
    #[test]
    fn counts_the_shapes_of_one_entry() {
        let a = counts(BLOB, "alpha").expect("alpha");
        assert_eq!(a.reqntid, Some(256));
        assert!(!a.depot);
        assert_eq!((a.ld_local, a.st_local), (0, 0));
        assert_eq!((a.fma, a.cvt_f16), (1, 2));
        let b = counts(BLOB, "beta").expect("beta");
        assert!(b.depot);
        assert_eq!((b.ld_local, b.st_local, b.fma), (1, 2, 1));
        assert_eq!(count(body(BLOB, "alpha").unwrap(), b"fma.f32"), 0);
    }

    /// Depot first, then name — the tool's whole reading order.
    #[test]
    fn scan_puts_the_spilling_entries_first() {
        let names: Vec<String> = scan(BLOB).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["beta", "alpha", "gamma"]);
    }

    #[test]
    fn find_and_count_survive_degenerate_needles() {
        assert_eq!(find(b"abc", b""), None);
        assert_eq!(find(b"abc", b"abcd"), None);
        assert_eq!(count(b"abc", b"abcd"), 0);
        assert_eq!(count(b"aaaa", b"aa"), 3);
    }
}
