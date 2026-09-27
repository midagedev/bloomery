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
//! whole row. `gate_p5` asserts the depot counts of the flash kernels through
//! [`crate::no_local_depot`]; `gate_p4::norm_geometry` asserts `rms_norm`/`norm_quant`
//! and `gate_p4::argmax_geometry` the `argmax_fault` pair against the block width
//! their host side launches;
//! `gate_p6` asserts the router gemvs' multiply-add floor and the Q3_K
//! entries' hardware f16 convert. Each walks its entries through
//! [`crate::ptx_shapes`].
//!
//! Spelling matters more than it looks: PTX writes a fused multiply-add as
//! `fma.rn.f32` (and `fma.rm.f32`), never `fma.f32`, so [`Counts::fma`]
//! counts the `fma.` prefix. A needle that matches nothing returns zero and
//! reads exactly like a clean kernel.
//!
//! The counts see the instructions an entry carries, not their order or
//! their operands: predicated selects rewritten into branches, or a
//! `stacksave` appearing, can leave every count equal. [`normalize`] is the
//! entry's whole instruction stream with only the names the backend picks
//! freely taken out, so two bodies that differ in anything else normalize
//! differently; `tools/ptx-scan.sh` prints the md5 of each entry's
//! normalized text, under the method name [`DIGEST_METHOD`]. The rules,
//! applied to the entry's [`body`] line by line:
//!
//! | input                                  | normalized                 |
//! |----------------------------------------|----------------------------|
//! | `// …` to the end of the line          | removed                    |
//! | a run of spaces and tabs               | one space; none at the ends|
//! | a line left empty                      | removed                    |
//! | `%r12` `%rd7` `%f3` `%p1` (a register) | `%r` `%rd` `%f` `%p`       |
//! | `%r<143>` (a register declaration)     | `%r<N>`                    |
//! | `$L__BB54_3` (a block label)           | `$L`                       |
//! | `__local_depot2`                       | `__local_depot`            |
//! | a generated module symbol (below)      | `<stem><<signature>>#k`    |
//! | the entry's own name, `<name>_param_3` | `ENTRY`, `ENTRY_param_3`   |
//! | anything else                          | kept byte for byte         |
//!
//! A generated module symbol is a whole token that starts with one of
//! [`GENERATED_STEMS`] (`__shared_mem_`, `__device_global_`,
//! `__dynamic_smem_`), whatever follows the stem: the backend picks the rest
//! of the name — a crate hash, a number, the kernel's name — and numbers and
//! orders the module's declarations freely. Such a token becomes its stem,
//! the signature of its declaration in the entry's module ([`Decls`]), and
//! `#k`, where k counts the entry's distinct generated symbols in order of
//! first appearance. The signature is the declaration with its name taken
//! out: linkage, state space, alignment, element type, dimensions, and
//! `= #<hex>`, the FNV-1a 64 of its initializer, when it has one. So the
//! digest moves when the entry's instructions move, when a symbol it names
//! changes its declaration, or when two of its references come to name one
//! symbol or one splits into two; not when only names or numbering move. A
//! generated token its module does not declare, or declares twice, is an
//! error, never a guess. Every other symbol (`_$_str`, a named global) is
//! kept byte for byte.
//!
//! Registers are the classes `%r %rd %rs %f %fd %p %h %hh %rq`; a special
//! register (`%tid.x`, `%clock64`) is kept. Opcodes with their type
//! suffixes, immediates (`0fFF800000`), operand order and instruction order
//! are kept. A name is replaced only as a whole token, so `add.s64` and
//! `alpha2` are not touched while normalizing `alpha`. What the digest does
//! not see: which register or label is which (a permuted operand pair of two
//! `%r` operands, or a branch retargeted to another label), and the body of
//! a device function the entry calls, which lies outside its [`body`].

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

    /// A module saved as a file of its own (the `mod<N>.ptx` `oxart_ptx`
    /// writes): the file's bytes are the payload, with nothing past them.
    #[must_use]
    pub fn saved(bundle: &'a str, text: &'a [u8]) -> Module<'a> {
        Module { bundle, text }
    }
}

/// The bundles of an `.oxart` section's bytes (what `objcopy -O binary
/// --only-section=.oxart` writes), cut by oxide-artifacts' parser. An error
/// when the parser rejects the container.
pub fn section_bundles(section: &[u8]) -> Result<Vec<OwnedArtifactBundle>, GateError> {
    Ok(oxide_artifacts::parse_artifact_section(section)
        .map_err(|e| format!("the .oxart container does not parse: {e}"))?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// The `.oxart` bundles of the running executable, read by oxide-artifacts'
/// object reader — the call cuda-core's module loader makes on the same
/// file, so a gate scans the text the driver is handed.
pub fn current_exe_bundles() -> Result<Vec<OwnedArtifactBundle>, GateError> {
    let exe = std::env::current_exe()?;
    let bytes = std::fs::read(&exe).map_err(|e| format!("read {}: {e}", exe.display()))?;
    Ok(
        oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
            .map_err(|e| format!("{}: the .oxart section does not parse: {e}", exe.display()))?,
    )
}

/// Err when a bundle carries a Cubin as well as a PTX payload. cuda-core's
/// loader takes the Cubin first and never hands the driver that PTX, so a
/// scan of it would describe code that does not run. [`modules`] refuses
/// through here, and it is the only way to a [`Module`].
fn ptx_is_what_loads(bundles: &[OwnedArtifactBundle]) -> Result<(), GateError> {
    let shadowed: Vec<&str> = bundles
        .iter()
        .filter(|b| {
            b.payload(ArtifactPayloadKind::Ptx).is_some()
                && b.payload(ArtifactPayloadKind::Cubin).is_some()
        })
        .map(|b| b.name.as_str())
        .collect();
    if shadowed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "bundle(s) {shadowed:?} carry a Cubin beside the PTX: the loader runs the Cubin, \
             so their PTX is not the device code"
        )
        .into())
    }
}

/// The PTX modules of `bundles` ([`section_bundles`], [`current_exe_bundles`]):
/// one per bundle that carries a PTX payload, in bundle order. An error when
/// a bundle carries a Cubin beside its PTX ([`ptx_is_what_loads`]).
pub fn modules(bundles: &[OwnedArtifactBundle]) -> Result<Vec<Module<'_>>, GateError> {
    ptx_is_what_loads(bundles)?;
    Ok(bundles
        .iter()
        .filter_map(|b| {
            let text = b.payload(ArtifactPayloadKind::Ptx)?;
            Some(Module {
                bundle: &b.name,
                text,
            })
        })
        .collect())
}

/// The marker every kernel's PTX declaration opens with.
const ENTRY: &[u8] = b".visible .entry ";

/// Offset of the line that holds the first `directive` (`.entry`, `.func`)
/// of `text`, standing as its own token: whitespace or the text's start
/// before it, whitespace or `(` after it — so `.callprototype`, a comment's
/// `.funcs` or a name that contains the word does not match.
fn directive_line(text: &[u8], directive: &[u8]) -> Option<usize> {
    let mut at = 0;
    while let Some(off) = find(&text[at..], directive) {
        let i = at + off;
        let before = i == 0 || matches!(text[i - 1], b' ' | b'\t' | b'\n');
        let after = matches!(
            text.get(i + directive.len()),
            Some(b' ' | b'\t' | b'\n' | b'(')
        );
        if before && after {
            return Some(
                text[..i]
                    .iter()
                    .rposition(|&c| c == b'\n')
                    .map_or(0, |n| n + 1),
            );
        }
        at = i + directive.len();
    }
    None
}

/// The PTX body of one `.visible .entry`: from just past its `name(` to the
/// line of the next declaration — the next entry, or a `.func` (a device
/// function's definition or prototype, whose instructions are not the
/// entry's) — or to the end of its module. `None` when no module declares
/// an entry of that name.
#[must_use]
pub fn body<'a>(modules: &[Module<'a>], name: &str) -> Option<&'a [u8]> {
    located(modules, name).map(|(_, b)| b)
}

/// [`body`], with the index of the module that holds it.
fn located<'a>(modules: &[Module<'a>], name: &str) -> Option<(usize, &'a [u8])> {
    let head = format!(".visible .entry {name}(");
    modules.iter().enumerate().find_map(|(i, m)| {
        let start = find(m.text, head.as_bytes())? + head.len();
        let rest = &m.text[start..];
        let end = directive_line(rest, b".entry").unwrap_or(rest.len());
        let end = directive_line(&rest[..end], b".func").unwrap_or(end);
        Some((i, &rest[..end]))
    })
}

/// `call` instructions in `body`: the mnemonic as its own token (`call`,
/// `call.uni`, predicated or not), not `.callprototype` or the `// callseq`
/// comments around a call. An entry that calls has its callee's work
/// outside its [`body`], so every count read from that body leaves it out.
#[must_use]
pub fn calls(body: &[u8]) -> usize {
    let mut n = 0;
    let mut at = 0;
    while let Some(off) = find(&body[at..], b"call") {
        let i = at + off;
        let before = i == 0 || matches!(body[i - 1], b' ' | b'\t' | b'\n' | b'{' | b';');
        let after = matches!(body.get(i + 4), Some(b'.' | b' ' | b'\t'));
        n += usize::from(before && after);
        at = i + 4;
    }
    n
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
/// widens `f16` in hardware. `calls` counts the body's `call` instructions:
/// when it is not zero, every other count here is the entry's own body
/// without its callees'.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counts {
    pub name: String,
    pub reqntid: Option<usize>,
    pub depot: bool,
    pub ld_local: usize,
    pub st_local: usize,
    pub fma: usize,
    pub cvt_f16: usize,
    pub calls: usize,
}

impl Counts {
    /// The counts of entry `name` whose [`body`] is `b`.
    #[must_use]
    pub fn of(name: &str, b: &[u8]) -> Counts {
        Counts {
            name: name.to_string(),
            reqntid: reqntid(b),
            depot: find(b, b"__local_depot").is_some(),
            ld_local: count(b, b"ld.local"),
            st_local: count(b, b"st.local"),
            fma: count(b, b"fma."),
            cvt_f16: count(b, b"cvt.f32.f16"),
            calls: calls(b),
        }
    }
}

/// Read one entry's counts out of the modules.
#[must_use]
pub fn counts(modules: &[Module<'_>], name: &str) -> Option<Counts> {
    body(modules, name).map(|b| Counts::of(name, b))
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

/// The register classes the backend numbers: `%<class><n>` and the
/// declaration `%<class><<count>>`.
const REG_CLASSES: &[&[u8]] = &[b"r", b"rd", b"rs", b"f", b"fd", b"p", b"h", b"hh", b"rq"];

/// Entry-local generated symbols that carry a number with no meaning of its own.
const NUMBERED_STEMS: &[&[u8]] = &[b"__local_depot"];

/// The stems of the module symbols the backend generates and names freely
/// (the table in this module's comment): a token that starts with one is
/// replaced by its declaration's signature.
pub const GENERATED_STEMS: &[&[u8]] = &[b"__shared_mem_", b"__device_global_", b"__dynamic_smem_"];

/// The name of the rule set [`normalize`] applies, which `tools/ptx-scan.sh`
/// prints over its digest block. Two digests compare only under one method:
/// a rule change that can move the digest of unchanged code renames it.
pub const DIGEST_METHOD: &str = "decl1";

/// The state spaces a module-scope declaration can open with, after its
/// linkage.
const DECL_OPENERS: &[&[u8]] = &[
    b".visible",
    b".extern",
    b".weak",
    b".common",
    b".shared",
    b".global",
    b".const",
];

/// A character of a PTX identifier (or of the alphanumeric run of a number).
fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_' || c == b'$'
}

/// Whether `tok`, a whole identifier token, is a generated module symbol.
fn generated(tok: &[u8]) -> bool {
    GENERATED_STEMS.iter().any(|s| tok.starts_with(s))
}

/// Whitespace runs to one space, none at the ends.
fn collapse(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for word in text
        .split(|c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
        .filter(|w| !w.is_empty())
    {
        if !out.is_empty() {
            out.push(b' ');
        }
        out.extend_from_slice(word);
    }
    out
}

/// FNV-1a, 64 bits: the initializer digest of a signature.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The generated module symbols a module declares, each with its signature
/// (the table in this module's comment). Only [`Decls::of`] makes one.
#[derive(Debug, Clone, Default)]
pub struct Decls(std::collections::HashMap<Vec<u8>, Vec<u8>>);

impl Decls {
    /// The declarations of every generated symbol in `module`: each line
    /// that opens with a linkage or a state space and names a generated
    /// symbol. An error when such a line is not one whole declaration this
    /// reader parses, or when a symbol is declared twice.
    pub fn of(module: &Module<'_>) -> Result<Decls, GateError> {
        let mut map = std::collections::HashMap::new();
        for raw in module.text.split(|&c| c == b'\n') {
            let Some((name, sig)) =
                declaration(raw).map_err(|e| format!("module {}: {e}", module.bundle))?
            else {
                continue;
            };
            if map.insert(name.clone(), sig).is_some() {
                return Err(format!(
                    "module {} declares {} twice",
                    module.bundle,
                    String::from_utf8_lossy(&name)
                )
                .into());
            }
        }
        Ok(Decls(map))
    }

    /// The signature of generated symbol `name`, or `None` when the module
    /// does not declare it.
    #[must_use]
    pub fn signature(&self, name: &[u8]) -> Option<&[u8]> {
        self.0.get(name).map(Vec::as_slice)
    }
}

/// A generated symbol's name and the signature of its declaration.
type Declared = (Vec<u8>, Vec<u8>);

/// The name and signature of the generated symbol `raw` declares; `None`
/// for a line that declares none (an instruction, another symbol). An error
/// for a line that opens like a declaration and names a generated symbol,
/// but is not `<directives> <name>[dims] [= init];` on one line.
fn declaration(raw: &[u8]) -> Result<Option<Declared>, String> {
    let code = find(raw, b"//").map_or(raw, |i| &raw[..i]).trim_ascii();
    let opens = DECL_OPENERS
        .iter()
        .any(|o| code.starts_with(o) && matches!(code.get(o.len()), Some(b' ' | b'\t')));
    if !opens {
        return Ok(None);
    }
    let mut at = 0;
    let (start, end) = loop {
        let Some(off) = code[at..].iter().position(|&c| is_ident(c)) else {
            return Ok(None);
        };
        let i = at + off;
        let end = i + code[i..].iter().take_while(|&&b| is_ident(b)).count();
        let whole = i == 0 || !is_ident(code[i - 1]);
        if whole && generated(&code[i..end]) {
            break (i, end);
        }
        at = end;
    };
    let name = code[start..end].to_vec();
    let refuse = |why: &str| {
        Err(format!(
            "declaration of {} {why}: {}",
            String::from_utf8_lossy(&name),
            String::from_utf8_lossy(code)
        ))
    };
    let head = &code[..start];
    let words_ok = head
        .split(|c| matches!(c, b' ' | b'\t'))
        .filter(|w| !w.is_empty())
        .all(|w| w.starts_with(b".") || w.iter().all(u8::is_ascii_digit));
    let spaced = [b".shared".as_slice(), b".global", b".const"]
        .iter()
        .any(|s| head.split(|c| matches!(c, b' ' | b'\t')).any(|w| w == *s));
    if !words_ok || !spaced {
        return refuse("does not open with directives naming a state space");
    }
    let Some(tail) = code[end..].strip_suffix(b";") else {
        return refuse("does not end on its line with `;`");
    };
    let (dims, init) = match tail.iter().position(|&c| c == b'=') {
        Some(eq) => (&tail[..eq], Some(tail[eq + 1..].trim_ascii())),
        None => (tail, None),
    };
    let dims = collapse(dims);
    let dims_ok = dims.is_empty()
        || dims.split(|&c| c == b']').all(|d| {
            d.is_empty()
                || d.strip_prefix(b"[")
                    .is_some_and(|n| n.iter().all(u8::is_ascii_digit))
        });
    if !dims_ok || init.is_some_and(<[u8]>::is_empty) {
        return refuse("has dimensions or an initializer this reader does not parse");
    }
    let mut sig = collapse(head);
    if !dims.is_empty() {
        sig.push(b' ');
        sig.extend_from_slice(&dims);
    }
    if let Some(init) = init {
        sig.extend_from_slice(format!(" = #{:016x}", fnv1a64(&collapse(init))).as_bytes());
    }
    Ok(Some((name, sig)))
}

/// Entry `name`'s normalized body ([`normalize`]) read out of `modules`,
/// with the declarations of the module that holds it; `decls` is
/// [`module_decls`] of the same `modules`. An error when no module declares
/// the entry, or when [`normalize`] refuses it.
pub fn normalized(
    modules: &[Module<'_>],
    decls: &[Decls],
    name: &str,
) -> Result<Vec<u8>, GateError> {
    let (i, b) =
        located(modules, name).ok_or_else(|| format!("no module declares entry {name}"))?;
    let d = decls
        .get(i)
        .ok_or_else(|| format!("no declarations read for module {i} (entry {name})"))?;
    normalize(d, name, b)
}

/// [`Decls::of`] of each module, in order.
pub fn module_decls(modules: &[Module<'_>]) -> Result<Vec<Decls>, GateError> {
    modules.iter().map(Decls::of).collect()
}

/// The text of entry `name`'s [`body`] with the freely picked names taken
/// out — the table in this module's comment — against `decls`, the
/// declarations of its module. Every kept line ends in `\n`. An error when
/// the body names a generated symbol `decls` does not hold.
pub fn normalize(decls: &Decls, name: &str, body: &[u8]) -> Result<Vec<u8>, GateError> {
    let mut n = Normalizer {
        decls,
        name: name.as_bytes(),
        seen: Vec::new(),
    };
    let mut out = Vec::with_capacity(body.len());
    for raw in body.split(|&c| c == b'\n') {
        let code = find(raw, b"//").map_or(raw, |i| &raw[..i]);
        let start = out.len();
        n.line(code, &mut out)?;
        if out.len() > start {
            out.push(b'\n');
        }
    }
    Ok(out)
}

/// One entry's normalization state: its module's declarations, its name,
/// and its generated symbols in order of first appearance.
struct Normalizer<'d> {
    decls: &'d Decls,
    name: &'d [u8],
    seen: Vec<Vec<u8>>,
}

impl Normalizer<'_> {
    /// One line with its comment already cut off: whitespace runs to one
    /// space, registers through [`register`], tokens through [`Self::symbol`].
    fn line(&mut self, line: &[u8], out: &mut Vec<u8>) -> Result<(), GateError> {
        let start = out.len();
        let mut space = false;
        let mut i = 0;
        while i < line.len() {
            let c = line[i];
            if matches!(c, b' ' | b'\t' | b'\r') {
                space = true;
                i += 1;
                continue;
            }
            if space && out.len() > start {
                out.push(b' ');
            }
            space = false;
            if c == b'%' {
                i = register(line, i, out);
            } else if is_ident(c) && (i == 0 || !is_ident(line[i - 1])) {
                let end = i + line[i..].iter().take_while(|&&b| is_ident(b)).count();
                self.symbol(&line[i..end], out)?;
                i = end;
            } else {
                out.push(c);
                i += 1;
            }
        }
        Ok(())
    }

    /// One whole identifier token: a block label, a numbered local symbol,
    /// a generated module symbol or the entry's own name is replaced; any
    /// other token is kept.
    fn symbol(&mut self, tok: &[u8], out: &mut Vec<u8>) -> Result<(), GateError> {
        let numbered = |rest: &[u8]| !rest.is_empty() && rest.iter().all(u8::is_ascii_digit);
        if let Some(rest) = tok.strip_prefix(b"$L__BB".as_slice())
            && rest.split(|&c| c == b'_').all(numbered)
        {
            out.extend_from_slice(b"$L");
            return Ok(());
        }
        for stem in NUMBERED_STEMS {
            if tok.strip_prefix(*stem).is_some_and(numbered) {
                out.extend_from_slice(stem);
                return Ok(());
            }
        }
        if let Some(stem) = GENERATED_STEMS.iter().find(|s| tok.starts_with(s)) {
            let sig = self.decls.signature(tok).ok_or_else(|| {
                format!(
                    "entry {} names {}, which its module does not declare",
                    String::from_utf8_lossy(self.name),
                    String::from_utf8_lossy(tok)
                )
            })?;
            let k = match self.seen.iter().position(|s| s == tok) {
                Some(p) => p + 1,
                None => {
                    self.seen.push(tok.to_vec());
                    self.seen.len()
                }
            };
            out.extend_from_slice(stem);
            out.push(b'<');
            out.extend_from_slice(sig);
            out.extend_from_slice(format!(">#{k}").as_bytes());
            return Ok(());
        }
        let name = self.name;
        if tok == name {
            out.extend_from_slice(b"ENTRY");
        } else if let Some(rest) = tok.strip_prefix(name)
            && rest.starts_with(b"_param_")
        {
            out.extend_from_slice(b"ENTRY");
            out.extend_from_slice(rest);
        } else {
            out.extend_from_slice(tok);
        }
        Ok(())
    }
}

/// The `%` at `line[at]`: a numbered register of a [`REG_CLASSES`] class
/// becomes its class, a declaration's count becomes `N`, anything else
/// (a special register) is kept. Returns the offset past what was read.
fn register(line: &[u8], at: usize, out: &mut Vec<u8>) -> usize {
    let letters = line[at + 1..]
        .iter()
        .take_while(|b| b.is_ascii_lowercase())
        .count();
    let class_end = at + 1 + letters;
    out.extend_from_slice(&line[at..class_end]);
    if !REG_CLASSES.contains(&&line[at + 1..class_end]) {
        return class_end;
    }
    let digits = line[class_end..]
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .count();
    let after = class_end + digits;
    if digits > 0 && line.get(after).is_none_or(|&c| !is_ident(c)) {
        return after;
    }
    if line.get(class_end) == Some(&b'<') {
        let count = line[class_end + 1..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        let close = class_end + 1 + count;
        if count > 0 && line.get(close) == Some(&b'>') {
            out.extend_from_slice(b"<N>");
            return close + 1;
        }
    }
    class_end
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

    /// An entry that calls a device function, the function's definition
    /// between it and the next entry, and an entry with a predicated call.
    /// The function carries every instruction a gate reads — two
    /// multiply-adds, an f16 widen, a `clz`, local traffic — so a body that
    /// ran on into it would hand all of them to `alpha`.
    const WITH_FUNC: &[u8] = b"//
.visible .entry alpha(
  .param .u64 alpha_param_0
)
.reqntid 256, 1, 1
{
  prototype_0 : .callprototype (.param .b32 _) _ (.param .b32 _);
  fma.rn.f32 %f1, %f2, %f3, %f4;
  { // callseq 0, 0
  .param .b32 retval0;
  call.uni (retval0), helper, (param0);
  } // callseq 0
  ret;
}
.func  (.param .b32 func_retval0) helper(
  .param .b32 helper_param_0
)
{
  fma.rn.f32 %f1, %f2, %f3, %f4;
  fma.rn.f32 %f5, %f6, %f7, %f8;
  cvt.f32.f16 %f9, %rs1;
  clz.b32 %r1, %r2;
  st.local.b32 [%rd1], %r1;
  ret;
}
.visible .entry beta(
  .param .u64 beta_param_0
)
{
  @%p1 call.uni helper, ();
  ret;
}
";

    /// A body ends at a `.func` as it ends at the next entry: the device
    /// function's instructions are its own, not the entry's before it.
    #[test]
    fn body_stops_at_a_device_function() {
        let mods = [Module {
            bundle: "test",
            text: WITH_FUNC,
        }];
        assert_eq!(entries(&mods), ["alpha", "beta"]);
        let a = body(&mods, "alpha").expect("alpha");
        assert!(
            find(a, b"helper(").is_none(),
            "alpha's body runs into the function: {:?}",
            String::from_utf8_lossy(a)
        );
        let c = counts(&mods, "alpha").expect("alpha");
        assert_eq!((c.fma, c.cvt_f16, c.st_local), (1, 0, 0));
        assert_eq!(count(a, b"clz."), 0);
        let b = counts(&mods, "beta").expect("beta");
        assert_eq!((b.fma, b.cvt_f16), (0, 0));
    }

    /// A call is counted once per instruction — plain, `.uni` or predicated
    /// — and never for the prototype or the callseq comments around it.
    #[test]
    fn counts_the_calls_an_entry_makes() {
        let mods = [Module {
            bundle: "test",
            text: WITH_FUNC,
        }];
        assert_eq!(counts(&mods, "alpha").expect("alpha").calls, 1);
        assert_eq!(counts(&mods, "beta").expect("beta").calls, 1);
        assert_eq!(counts(&ONE, "alpha").expect("alpha").calls, 0);
        assert_eq!(calls(b"call foo, ();\n\tcall.uni bar, ();"), 2);
        assert_eq!(calls(b"// callseq 1\n.callprototype\n_Z6recallv"), 0);
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
        let mods = modules(&bundles).expect("no Cubin shadows the PTX");
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
        let mods = modules(&bundles).expect("no Cubin shadows the PTX");
        let names: Vec<&str> = mods.iter().map(Module::bundle).collect();
        assert_eq!(names, ["first", "second"]);
        assert_eq!(entries(&mods), ["alpha", "beta", "gamma"]);
        let b = body(&mods, "beta").expect("beta");
        assert!(PTX8.ends_with(b));
        assert_eq!(reqntid(body(&mods, "gamma").expect("gamma")), None);
    }

    /// A bundle that carries a Cubin beside its PTX is refused where the
    /// modules are cut: the loader runs the Cubin, so the PTX is not the
    /// device code. A Cubin-only bundle is no PTX module at all, and is not an
    /// error.
    #[test]
    fn a_bundle_whose_ptx_a_cubin_shadows_is_refused() {
        let cubin: &[u8] = b"\x7fELF not a real cubin";
        let both = build_artifact_blob(
            &ArtifactBundleSpec::new("shadowed", "sm_86")
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ptx,
                    "bloomery_gpu.ptx",
                    PTX8,
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Cubin,
                    "bloomery_gpu.cubin",
                    cubin,
                )),
        )
        .expect("a valid bundle spec");
        let bundles = section_bundles(&both).expect("the container parses");
        let e = modules(&bundles).expect_err("a PTX shadowed by a Cubin");
        assert!(e.to_string().contains("shadowed"), "{e}");
        let only = build_artifact_blob(&ArtifactBundleSpec::new("cubin", "sm_86").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Cubin, "bloomery_gpu.cubin", cubin),
        ))
        .expect("a valid bundle spec");
        let bundles = section_bundles(&only).expect("a Cubin-only bundle");
        assert!(modules(&bundles).expect("no PTX to shadow").is_empty());
    }

    /// A section the parser rejects is an error, never an empty list; a
    /// section of zero padding alone is a container with no bundle.
    #[test]
    fn a_rejected_container_is_an_error_not_an_empty_scan() {
        let blob = bundle_blob("bloomery-gpu", PTX8, &["alpha", "beta"]);
        assert!(section_bundles(&blob[..blob.len() - 8]).is_err());
        assert!(section_bundles(b"not an oxide-artifacts container at all").is_err());
        let empty = section_bundles(&[0u8; 32]).expect("zero padding parses");
        assert!(modules(&empty).expect("no bundle to refuse").is_empty());
    }

    /// A 20-line entry body shaped like the backend's: parameters, register
    /// declarations, a predicated branch, labels, shared memory, a special
    /// register and a hex-float immediate.
    const NORM_A: &[u8] = b"
	.param .u64 .ptr .align 4 alpha_param_0,
	.param .u32 alpha_param_1
)
.reqntid 32, 1, 1                       // @alpha
{
	.reg .pred 	%p<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<5>;

// %bb.0:                               // %entry
	ld.param.b64 	%rd1, [alpha_param_0];
	mov.u32 	%r1, %tid.x;
	setp.ne.b32 	%p1, %r1, 0;
	@%p1 bra 	$L__BB3_2;
	mov.b32 	%r2, 0fFF800000;
	st.shared.b32 	[__shared_mem_4], %r2;
$L__BB3_2:                              // %exit
	add.s64 	%rd2, %rd1, 4;
	ret;
}
";

    /// `NORM_A` as another build emits it: every register, label and shared
    /// symbol renumbered, the entry renamed, comments and spacing changed.
    const NORM_A_RENUMBERED: &[u8] = b"
    .param .u64 .ptr .align 4 beta_param_0,
    .param .u32 beta_param_1
)
.reqntid 32, 1, 1
{
    .reg .pred %p<4>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<5>;
    ld.param.b64 %rd7, [beta_param_0];    // renamed
    mov.u32 %r5, %tid.x;

    setp.ne.b32 %p3, %r5, 0;
    @%p3 bra $L__BB11_7;
    mov.b32 %r6, 0fFF800000;
    st.shared.b32 [__shared_mem_19], %r6;
$L__BB11_7:
    add.s64 %rd8, %rd7, 4;
    ret;
}
";

    /// The declarations of a module whose only generated symbols are
    /// `lines`, one declaration a line.
    fn decls_of(lines: &str) -> Decls {
        Decls::of(&Module::saved("test", lines.as_bytes())).expect("the declarations parse")
    }

    /// `normalize` against `decls`, which must accept the body.
    fn norm(decls: &Decls, name: &str, body: &[u8]) -> Vec<u8> {
        normalize(decls, name, body).expect("every generated symbol is declared")
    }

    /// Renumbering and renaming leave the normalized text — and so its md5
    /// — unchanged; the text is the stream with only the numbers gone.
    #[test]
    fn normalize_ignores_renumbering() {
        let a = norm(
            &decls_of(".visible .shared .align 4 .b8 __shared_mem_4[4];"),
            "alpha",
            NORM_A,
        );
        let b = norm(
            &decls_of(".visible .shared .align 4 .b8 __shared_mem_19[4];"),
            "beta",
            NORM_A_RENUMBERED,
        );
        assert_eq!(a, b);
        let text = String::from_utf8(a).expect("ascii");
        assert_eq!(text.lines().count(), 18, "{text}");
        for kept in [
            "ld.param.b64 %rd, [ENTRY_param_0];",
            "mov.u32 %r, %tid.x;",
            "@%p bra $L;",
            "mov.b32 %r, 0fFF800000;",
            "st.shared.b32 [__shared_mem_<.visible .shared .align 4 .b8 [4]>#1], %r;",
            "$L:",
            "add.s64 %rd, %rd, 4;",
            ".reg .b64 %rd<N>;",
            ".reqntid 32, 1, 1",
        ] {
            assert!(
                text.lines().any(|l| l == kept),
                "no line {kept:?} in\n{text}"
            );
        }
    }

    /// Any change that is not a number the backend picks freely changes the
    /// normalized text: an opcode, an immediate, operand order, instruction
    /// order, a special register, the block width.
    #[test]
    fn normalize_sees_every_other_change() {
        let d = decls_of(".visible .shared .align 4 .b8 __shared_mem_4[4];");
        let a = norm(&d, "alpha", NORM_A);
        let src = std::str::from_utf8(NORM_A).expect("ascii");
        for (from, to) in [
            ("setp.ne.b32", "setp.eq.b32"),
            ("add.s64", "add.u64"),
            ("0fFF800000", "0f7F800000"),
            ("%rd1, 4;", "%rd1, 8;"),
            ("%r1, 0;", "0, %r1;"),
            ("%tid.x", "%tid.y"),
            (".reqntid 32", ".reqntid 64"),
            ("alpha_param_0]", "alpha_param_1]"),
            ("\tret;", "\texit;"),
        ] {
            assert!(src.contains(from), "{from:?} is not in the body");
            let changed = norm(&d, "alpha", src.replacen(from, to, 1).as_bytes());
            assert_ne!(a, changed, "{from:?} -> {to:?} normalized the same");
        }
        let swapped = src.replacen(
            "\tmov.u32 \t%r1, %tid.x;\n\tsetp.ne.b32 \t%p1, %r1, 0;\n",
            "\tsetp.ne.b32 \t%p1, %r1, 0;\n\tmov.u32 \t%r1, %tid.x;\n",
            1,
        );
        assert_ne!(swapped, src, "the swap must apply");
        assert_ne!(a, norm(&d, "alpha", swapped.as_bytes()));
    }

    /// Only whole tokens are rewritten: a longer name that starts with the
    /// entry's, a special register that ends in digits, and a register class
    /// outside the list stay as they are.
    #[test]
    fn normalize_rewrites_whole_tokens_only() {
        let none = Decls::default();
        let n = norm(
            &none,
            "alpha",
            b"call alpha2, (alpha_x);\nmov.u64 %rd1, %clock64;\n%q3;\n",
        );
        assert_eq!(
            std::str::from_utf8(&n).expect("ascii"),
            "call alpha2, (alpha_x);\nmov.u64 %rd, %clock64;\n%q3;\n"
        );
        assert_eq!(norm(&none, "alpha", b"  // only a comment\n\n\t\n"), b"");
        assert_eq!(
            norm(&none, "a", b"$L__BB1_x: __local_depot7 __local_depotx\n"),
            b"$L__BB1_x: __local_depot __local_depotx\n"
        );
        assert_eq!(
            norm(
                &decls_of(".visible .global .align 8 .b8 __device_global_3;"),
                "a",
                b"ld.global.nc.u64 %rd1, [__device_global_3+8], __device_globalx;\n"
            ),
            b"ld.global.nc.u64 %rd, [__device_global_<.visible .global .align 8 .b8>#1+8], \
              __device_globalx;\n"
        );
    }

    /// A module like the backend's, its generated names spelled by `names`:
    /// the device global, three shared buffers (`s2` and `s3` of one
    /// signature, `s5` another) and the dynamic shared symbol, in that
    /// order; `order` is the order the declarations are written in. `alpha`
    /// names the global, `s2`, `s3` and the dynamic symbol; `beta` names `s5`.
    fn module(names: [&str; 5], order: [usize; 5]) -> String {
        let [g, s2, s3, s5, dy] = names;
        let decls = [
            format!(".visible .global .align 4 .b8 {g}[8] = {{1, 2, 3, 4, 5, 6, 7, 8}};"),
            format!(".visible .shared .align 4 .b8 {s2}[256];"),
            format!(".visible .shared .align 4 .b8 {s3}[256];"),
            format!(".visible .shared .align 4 .b8 {s5}[64];"),
            format!(".extern .shared .align 16 .b8 {dy}[];"),
        ];
        let decls: Vec<&str> = order.iter().map(|&i| decls[i].as_str()).collect();
        format!(
            ".version 8.7\n.target sm_86\n.address_size 64\n\n{}\n\
             .global .align 1 .b8 _$_str[4] = {{97, 98, 99, 0}};\n\n\
             .visible .entry alpha(\n\t.param .u64 alpha_param_0\n)\n{{\n\
             \tld.global.nc.u32 \t%r1, [{g}+4];\n\
             \tst.shared.b32 \t[{s2}], %r1;\n\
             \tst.shared.b32 \t[{s3}+4], %r1;\n\
             \tld.shared.b32 \t%r2, [{dy}];\n\
             \tmov.u64 \t%rd1, _$_str;\n\
             \tret;\n}}\n\
             .visible .entry beta(\n)\n{{\n\
             \tld.shared.b32 \t%r1, [{s5}];\n\
             \tret;\n}}\n",
            decls.join("\n")
        )
    }

    const PLAIN: [&str; 5] = [
        "__device_global_4",
        "__shared_mem_2",
        "__shared_mem_3",
        "__shared_mem_5",
        "__dynamic_smem_alpha",
    ];
    const IN_ORDER: [usize; 5] = [0, 1, 2, 3, 4];

    /// Each entry's normalized body, in entry order, or the first refusal.
    fn digests(text: &str) -> Result<Vec<Vec<u8>>, GateError> {
        let mods = [Module::saved("test", text.as_bytes())];
        let decls = module_decls(&mods)?;
        entries(&mods)
            .iter()
            .map(|e| normalized(&mods, &decls, e))
            .collect()
    }

    /// The backend spells a generated name with a crate hash between the
    /// stem and the number (or the kernel's name); the hash is a name, so
    /// the hashed module normalizes like the plain one.
    #[test]
    fn a_hashed_generated_name_normalizes_like_the_plain_one() {
        let hashed = [
            "__device_global_0123456789abcdef_4",
            "__shared_mem_0123456789abcdef_2",
            "__shared_mem_0123456789abcdef_3",
            "__shared_mem_0123456789abcdef_5",
            "__dynamic_smem_0123456789abcdef_alpha",
        ];
        let plain = digests(&module(PLAIN, IN_ORDER)).expect("plain");
        assert_eq!(plain, digests(&module(hashed, IN_ORDER)).expect("hashed"));
        let alpha = String::from_utf8(plain[0].clone()).expect("ascii");
        for kept in [
            "ld.global.nc.u32 %r, [__device_global_<.visible .global .align 4 .b8 [8] = \
             #187158eaeba8f101>#1+4];",
            "st.shared.b32 [__shared_mem_<.visible .shared .align 4 .b8 [256]>#2], %r;",
            "st.shared.b32 [__shared_mem_<.visible .shared .align 4 .b8 [256]>#3+4], %r;",
            "ld.shared.b32 %r, [__dynamic_smem_<.extern .shared .align 16 .b8 []>#4];",
            "mov.u64 %rd, _$_str;",
        ] {
            assert!(
                alpha.lines().any(|l| l == kept),
                "no line {kept:?} in\n{alpha}"
            );
        }
    }

    /// Declarations renumbered and written in another order — each symbol
    /// keeping its own declaration — leave every entry's digest in place.
    #[test]
    fn two_modules_that_differ_only_by_renumbering_give_equal_entry_digests() {
        let renumbered = [
            "__device_global_0",
            "__shared_mem_9",
            "__shared_mem_1",
            "__shared_mem_2",
            "__dynamic_smem_alpha",
        ];
        assert_eq!(
            digests(&module(PLAIN, IN_ORDER)).expect("base"),
            digests(&module(renumbered, [4, 3, 0, 2, 1])).expect("renumbered")
        );
    }

    /// A referenced symbol's declaration that changes size moves the digest
    /// of the entry that names it, and of no other entry.
    #[test]
    fn a_changed_declaration_size_moves_the_digest() {
        let base = digests(&module(PLAIN, IN_ORDER)).expect("base");
        let text = module(PLAIN, IN_ORDER);
        for (from, to, moved) in [
            ("__shared_mem_5[64]", "__shared_mem_5[128]", [false, true]),
            ("__shared_mem_3[256]", "__shared_mem_3[512]", [true, false]),
            (
                ".align 4 .b8 __shared_mem_2",
                ".align 8 .b8 __shared_mem_2",
                [true, false],
            ),
            (
                "{1, 2, 3, 4, 5, 6, 7, 8}",
                "{1, 2, 3, 4, 5, 6, 7, 9}",
                [true, false],
            ),
        ] {
            assert!(text.contains(from), "{from:?} is not in the module");
            let changed = digests(&text.replacen(from, to, 1)).expect("changed");
            for (e, m) in moved.into_iter().enumerate() {
                assert_eq!(base[e] != changed[e], m, "{from:?} -> {to:?}, entry {e}");
            }
        }
    }

    /// Two declarations swapped under their references — each reference now
    /// lands on a symbol of another size — move the digest of every entry
    /// that names either; the same swap made in the references too is a
    /// renaming and moves nothing.
    #[test]
    fn a_reference_moved_onto_a_declaration_of_another_size_moves_the_digest() {
        let text = module(PLAIN, IN_ORDER);
        let base = digests(&text).expect("base");
        let decl_swapped = text
            .replacen(".b8 __shared_mem_3[256]", ".b8 __shared_mem_X[256]", 1)
            .replacen(".b8 __shared_mem_5[64]", ".b8 __shared_mem_3[64]", 1)
            .replacen(".b8 __shared_mem_X[256]", ".b8 __shared_mem_5[256]", 1);
        let changed = digests(&decl_swapped).expect("swapped");
        assert_ne!(base[0], changed[0]);
        assert_ne!(base[1], changed[1]);
        let renamed = [
            "__device_global_4",
            "__shared_mem_2",
            "__shared_mem_5",
            "__shared_mem_3",
            "__dynamic_smem_alpha",
        ];
        assert_eq!(base, digests(&module(renamed, IN_ORDER)).expect("renamed"));
    }

    /// Two references to two symbols of one signature that come to name one
    /// symbol move the digest: the order of first appearance tells them apart.
    #[test]
    fn two_references_merged_into_one_symbol_move_the_digest() {
        let base = digests(&module(PLAIN, IN_ORDER)).expect("base");
        let text = module(PLAIN, IN_ORDER).replacen("[__shared_mem_3+4]", "[__shared_mem_2+4]", 1);
        let merged = digests(&text).expect("merged");
        assert_ne!(base[0], merged[0]);
        assert_eq!(base[1], merged[1]);
    }

    /// A symbol that is not generated is kept byte for byte, name included.
    #[test]
    fn a_symbol_that_is_not_generated_is_kept_by_name() {
        let base = digests(&module(PLAIN, IN_ORDER)).expect("base");
        let text = module(PLAIN, IN_ORDER).replace("_$_str", "_$_str_$_2");
        assert_ne!(base[0], digests(&text).expect("renamed")[0]);
    }

    /// A generated symbol its module does not declare, one declared twice,
    /// and a declaration this reader does not parse are named errors, never
    /// a digest.
    #[test]
    fn an_undeclared_or_unreadable_generated_symbol_is_a_named_error() {
        let text = module(PLAIN, IN_ORDER);
        let away = text.replacen(".b8 __shared_mem_5[64]", ".b8 __shared_mem_6[64]", 1);
        let e = digests(&away).expect_err("beta names an undeclared symbol");
        assert!(
            e.to_string().contains("entry beta names __shared_mem_5"),
            "{e}"
        );
        let twice = format!("{text}.visible .shared .align 4 .b8 __shared_mem_5[64];\n");
        let e = digests(&twice).expect_err("declared twice");
        assert!(
            e.to_string().contains("declares __shared_mem_5 twice"),
            "{e}"
        );
        for bad in [
            ".visible .global .align 4 .b8 __device_global_7[8] = {1, 2,",
            ".visible .align 4 .b8 __shared_mem_7[8];",
            ".visible .shared .align 4 .b8 __shared_mem_7[x];",
        ] {
            let e = digests(&format!("{bad}\n{text}")).expect_err(bad);
            assert!(
                e.to_string().contains("__shared_mem_7")
                    || e.to_string().contains("__device_global_7"),
                "{e}"
            );
        }
    }
}
