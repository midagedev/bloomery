//! Reading the device code a gate binary carries. `cargo oxide` embeds the
//! PTX text of the whole `bloomery-gpu` bundle in the executable's `.oxart`
//! ELF section, so a kernel's compiled shape is a byte scan of that text —
//! no device, no clock, and the same answer the backend would report.
//!
//! The section is an oxide-artifacts container: a header, the payload and
//! entry records, each payload, then the entry-symbol table. Only its own
//! parser knows where a payload ends. A payload is stored with its length
//! and padded to 8 bytes, so no NUL is promised after the PTX text: a
//! payload of 8k bytes runs straight into the container's next record (for
//! the last payload, the symbol table). So every reader here takes
//! [`Module`]s, which only that parser's output yields ([`modules`] of
//! [`section_bundles`] or [`current_exe_bundles`]); raw section or
//! executable bytes do not type-check.
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

use crate::GateError;
use oxide_artifacts::{ArtifactPayloadKind, OwnedArtifactBundle};

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

/// One PTX module: the PTX payload of one `.oxart` bundle, exactly as long
/// as the container records it. Only [`modules`] makes one.
#[derive(Clone, Copy, Debug)]
pub struct Module<'a> {
    bundle: &'a str,
    text: &'a [u8],
}

impl<'a> Module<'a> {
    /// The name of the bundle the payload belongs to.
    #[must_use]
    pub fn bundle(&self) -> &'a str {
        self.bundle
    }

    /// The PTX text: the payload's bytes and nothing past them.
    #[must_use]
    pub fn text(&self) -> &'a [u8] {
        self.text
    }
}

/// The bundles of an `.oxart` section's bytes (what `objcopy -O binary
/// --only-section=.oxart` writes), cut by oxide-artifacts' parser. An error
/// when the parser rejects the container.
pub fn section_bundles(section: &[u8]) -> Result<Vec<OwnedArtifactBundle>, GateError> {
    let bundles = oxide_artifacts::parse_artifact_section(section)
        .map_err(|e| format!("the .oxart container does not parse: {e}"))?;
    Ok(bundles.into_iter().map(Into::into).collect())
}

/// The `.oxart` bundles of the running executable, read by oxide-artifacts'
/// object reader — the call cuda-core's module loader makes on the same
/// file, so a gate scans the text the driver is handed.
pub fn current_exe_bundles() -> Result<Vec<OwnedArtifactBundle>, GateError> {
    let exe = std::env::current_exe()?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("read {}: {e}", exe.display()))?;
    oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
        .map_err(|e| format!("{}: the .oxart section does not parse: {e}", exe.display()).into())
}

/// The PTX modules of `bundles`: one per bundle that carries a PTX payload,
/// in bundle order.
#[must_use]
pub fn modules(bundles: &[OwnedArtifactBundle]) -> Vec<Module<'_>> {
    bundles
        .iter()
        .filter_map(|b| {
            let text = b.payload(ArtifactPayloadKind::Ptx)?;
            Some(Module {
                bundle: &b.name,
                text,
            })
        })
        .collect()
}

/// The marker every kernel's PTX declaration opens with.
const ENTRY: &[u8] = b".visible .entry ";

/// The PTX body of one `.visible .entry`: from just past its `name(` to the
/// next entry's declaration, or to the end of its module for the last one.
/// `None` when no module declares an entry of that name.
#[must_use]
pub fn body<'a>(modules: &[Module<'a>], name: &str) -> Option<&'a [u8]> {
    let head = format!(".visible .entry {name}(");
    modules.iter().find_map(|m| {
        let start = find(m.text, head.as_bytes())? + head.len();
        let rest = &m.text[start..];
        Some(match find(rest, ENTRY) {
            Some(end) => &rest[..end],
            None => rest,
        })
    })
}

/// Every entry name, module by module, in the order the text declares them.
#[must_use]
pub fn entries(modules: &[Module<'_>]) -> Vec<String> {
    let mut out = Vec::new();
    for m in modules {
        let mut at = 0usize;
        while let Some(off) = find(&m.text[at..], ENTRY) {
            let name_at = at + off + ENTRY.len();
            let name: Vec<u8> = m.text[name_at..]
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

/// Read one entry's counts out of the modules.
#[must_use]
pub fn counts(modules: &[Module<'_>], name: &str) -> Option<Counts> {
    let b = body(modules, name)?;
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

/// Every entry of the modules, sorted so the ones carrying a local depot
/// come first and names ascend inside each group — the order a reader
/// wants, because a depot is the defect and the rest is context.
#[must_use]
pub fn scan(modules: &[Module<'_>]) -> Vec<Counts> {
    let mut names = entries(modules);
    names.sort();
    names.dedup();
    let mut out: Vec<Counts> = names.iter().filter_map(|n| counts(modules, n)).collect();
    out.sort_by(|a, b| b.depot.cmp(&a.depot).then_with(|| a.name.cmp(&b.name)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_artifacts::{
        ArtifactBundleSpec, ArtifactEntryKind, ArtifactEntrySpec, ArtifactPayloadSpec,
        build_artifact_blob,
    };

    /// A module shaped like the real one: three entries, the middle one
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

    /// `BLOB` as the one module of a bundle, as the text readers see it.
    const ONE: [Module<'static>; 1] = [Module {
        bundle: "test",
        text: BLOB,
    }];

    #[test]
    fn lists_every_entry_in_declaration_order() {
        assert_eq!(entries(&ONE), ["alpha", "beta", "gamma"]);
    }

    /// An entry's body must stop at the next entry, or the counts of the
    /// whole tail land on the first kernel scanned.
    #[test]
    fn body_stops_at_the_next_entry() {
        let a = body(&ONE, "alpha").expect("alpha");
        assert!(find(a, b"__local_depot").is_none());
        assert_eq!(count(a, b"ld.local"), 0);
        let b = body(&ONE, "beta").expect("beta");
        assert!(find(b, b"gamma").is_none());
        assert_eq!(body(&ONE, "delta"), None);
    }

    #[test]
    fn reads_the_block_width_and_its_absence() {
        assert_eq!(reqntid(body(&ONE, "alpha").unwrap()), Some(256));
        assert_eq!(reqntid(body(&ONE, "beta").unwrap()), Some(32));
        assert_eq!(reqntid(body(&ONE, "gamma").unwrap()), None);
    }

    /// The counters, including the spelling trap: `fma.f32` matches nothing
    /// in real PTX, so the prefix is what the counter uses.
    #[test]
    fn counts_the_shapes_of_one_entry() {
        let a = counts(&ONE, "alpha").expect("alpha");
        assert_eq!(a.reqntid, Some(256));
        assert!(!a.depot);
        assert_eq!((a.ld_local, a.st_local), (0, 0));
        assert_eq!((a.fma, a.cvt_f16), (1, 2));
        let b = counts(&ONE, "beta").expect("beta");
        assert!(b.depot);
        assert_eq!((b.ld_local, b.st_local, b.fma), (1, 2, 1));
        assert_eq!(count(body(&ONE, "alpha").unwrap(), b"fma.f32"), 0);
    }

    /// Depot first, then name — the tool's whole reading order.
    #[test]
    fn scan_puts_the_spilling_entries_first() {
        let names: Vec<String> = scan(&ONE).into_iter().map(|c| c.name).collect();
        assert_eq!(names, ["beta", "alpha", "gamma"]);
    }

    #[test]
    fn find_and_count_survive_degenerate_needles() {
        assert_eq!(find(b"abc", b""), None);
        assert_eq!(find(b"abc", b"abcd"), None);
        assert_eq!(count(b"abc", b"abcd"), 0);
        assert_eq!(count(b"aaaa", b"aa"), 3);
    }

    /// A PTX payload whose length is a multiple of 8, holding two entries.
    const PTX8: &[u8] = b"//
// Two entries whose text is a multiple of 8 bytes long, so the container
// pads nothing after it.
//
.version 8.7
.target sm_86
.address_size 64

.visible .entry alpha(
  .param .u64 alpha_param_0
)
.reqntid 256, 1, 1
{
  fma.rn.f32 %f1, %f2, %f3, %f4;
  ret;
}
.visible .entry beta(
  .param .u64 beta_param_0
)
.reqntid 32, 1, 1
{
  st.local.b32 [%rd1], %r10;
  ret;
}
";

    /// One bundle's blob with `ptx` as its only payload, written by the
    /// container's own writer.
    fn bundle_blob(name: &str, ptx: &[u8], symbols: &[&str]) -> Vec<u8> {
        let mut spec = ArtifactBundleSpec::new(name, "sm_86").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "bloomery_gpu.ptx", ptx),
        );
        for &s in symbols {
            spec = spec.with_entry(ArtifactEntrySpec::new(s, ArtifactEntryKind::Kernel));
        }
        build_artifact_blob(&spec).expect("a valid bundle spec")
    }

    /// The container stores a payload with its length and pads it to 8
    /// bytes only, so a payload of 8k bytes is followed directly by the
    /// next record — here the entry-symbol table. The last entry's body
    /// must end where its payload ends.
    #[test]
    fn last_body_ends_where_its_payload_ends() {
        assert_eq!(PTX8.len() % 8, 0, "the case needs a payload of 8k bytes");
        let blob = bundle_blob("bloomery-gpu", PTX8, &["alpha", "beta"]);
        let end = find(&blob, PTX8).expect("the payload sits in the blob") + PTX8.len();
        assert_ne!(
            blob[end], 0,
            "a NUL follows the payload: not the case under test"
        );
        let bundles = section_bundles(&blob).expect("the container parses");
        let mods = modules(&bundles);
        assert_eq!(mods.len(), 1);
        assert_eq!((mods[0].bundle(), mods[0].text()), ("bloomery-gpu", PTX8));
        assert_eq!(entries(&mods), ["alpha", "beta"]);
        let b = body(&mods, "beta").expect("beta");
        assert!(
            PTX8.ends_with(b),
            "the last body runs past its payload, ending {:?}",
            String::from_utf8_lossy(&b[b.len().saturating_sub(24)..])
        );
        assert_eq!(b.as_ptr_range().end, mods[0].text().as_ptr_range().end);
    }

    /// Bundles concatenate in one section; a body never runs from one
    /// bundle's module into the next.
    #[test]
    fn each_bundle_is_its_own_module() {
        let gamma: &[u8] =
            b".version 8.7\n.target sm_86\n\n.visible .entry gamma(\n)\n{\n  ret;\n}\n";
        let mut section = bundle_blob("first", PTX8, &["alpha", "beta"]);
        section.extend_from_slice(&bundle_blob("second", gamma, &["gamma"]));
        let bundles = section_bundles(&section).expect("both blobs parse");
        let mods = modules(&bundles);
        let names: Vec<&str> = mods.iter().map(Module::bundle).collect();
        assert_eq!(names, ["first", "second"]);
        assert_eq!(entries(&mods), ["alpha", "beta", "gamma"]);
        let b = body(&mods, "beta").expect("beta");
        assert!(PTX8.ends_with(b));
        assert_eq!(reqntid(body(&mods, "gamma").expect("gamma")), None);
    }

    /// A section the parser rejects is an error, never an empty list; a
    /// section of zero padding alone is a container with no bundle.
    #[test]
    fn a_rejected_container_is_an_error_not_an_empty_scan() {
        let blob = bundle_blob("bloomery-gpu", PTX8, &["alpha", "beta"]);
        assert!(section_bundles(&blob[..blob.len() - 8]).is_err());
        assert!(section_bundles(b"not an oxide-artifacts container at all").is_err());
        let empty = section_bundles(&[0u8; 32]).expect("zero padding parses");
        assert!(modules(&empty).is_empty());
    }
}
