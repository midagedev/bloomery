//! `metal_scan` — the Metal move proof's pipeline hash and the two compares
//! (`just metal-compare`, owner `tools/metal-compare.sh`).
//!
//!     metal_scan [--instance NAME] [--out FILE] [--keep DIR]
//!     metal_scan --compare BASE.tsv HEAD.tsv
//!     metal_scan --compare-trace BASE.txt HEAD.txt
//!
//! The scan builds every compute pipeline the engine builds, on this Mac's
//! device, from both libraries (the simdgroup program and the tensor-op GEMMs'
//! program), exactly as `Kernels::from_library` builds them: the function
//! names and function-constant values are the engine's own, recorded as the
//! engine's constructors make them (`bloomery_metal::scan`), never a list
//! kept here. Each is serialized into a binary archive of its own
//! (`MTLBinaryArchive`: the device compiler's output for this device) and the
//! archive's bytes are hashed. Hashing archive bytes gives each entry its own
//! hash even though every kernel is one translation unit: the archive holds
//! the machine code of the one function, not the library's text.
//!
//! Three things in an archive are not the function's: the library's identity.
//! The device compiler names a function in its archive by the content hash of
//! the library it came from, so a function with no function constants carries
//! that hash in every other entry of the library: the 64-hex-digit module
//! signature (six times: in the `alias:<signature>#<entry>` names), the
//! module's UUID (the `UUID` record of the AIR header and the copy in the
//! GPU-code slice) and the GPU slice's Mach-O `LC_UUID`. One changed kernel
//! moves them in all of those entries and in no other bytes. The scan zeroes
//! exactly those fields (`stripped_bytes` says how many bytes) and then
//! proves nothing else depends on the library: it builds both libraries a
//! second time with one kernel appended that no pipeline uses
//! (`bloomery_scan_calibration`), scans that too, and requires the stripped
//! archives of the two builds to be byte-equal, entry by entry, or ends with
//! a named error.
//!
//! The TSV has a `#` header block naming what the hash depends on (the OS
//! build, the GPU, the language versions, the compile options, whether the
//! library is the build's metallib or the run-time source, and the sources'
//! own digests), then one row a pipeline: library, entry, function constants,
//! whether the engine builds it on this device, the archive's SHA-256 and
//! size, the SHA-256 of the archive's GPU-code slice alone (the machine code
//! the pipeline runs; the archive also carries the function's AIR), the
//! identity bytes stripped, three numbers the device compiler fixes in the
//! built pipeline (most threads a group, SIMD width, static threadgroup
//! bytes), and `exec_sha256`: the SHA-256 over the GPU slice's Mach-O sections
//! other than `__metallib`, which holds the function's AIR (the IR the device
//! compiler read, not code the device runs). Each section enters as its
//! segment, name and length, then its bytes, in the order the slice lists
//! them; the slice's load commands do not enter (their sizes follow
//! `__metallib`'s). The names the hash covered are the `exec_sections`
//! header. `text_sha256` is the SHA-256 of the bytes of `__TEXT,__text` in the
//! Mach-O 64 that the slice's `__compute` section holds: the instructions the
//! GPU runs, without the compiler's link and statistics notes
//! (`__GPU_LD_MD`, `__GPU_STATS_MD`) beside them, which can move while the
//! instructions do not. A slice that is no 64-bit Mach-O, a section that lies
//! past its container, no `__compute` section (or two), a `__compute` section
//! that is no 64-bit Mach-O and one with no `__TEXT,__text` (or two) is a
//! named error. A pipeline that cannot be built
//! or serialized is a named error and exit 1, and so is a library that is the
//! build's metallib: the calibration compiles the run-time source, so it must
//! be what runs.
//!
//! `--instance NAME` scans one registered instance's kernels (default: the
//! build's host instance); a name that is not registered is exit 64, naming
//! the registered ones.
//!
//! `--keep DIR` keeps each serialized archive as `DIR/<lib>-<entry>-<constants>.bin`, for
//! reading what differs between two runs.
//!
//! `--compare` prints, for two scans, the pipelines added, removed and
//! changed (with the columns that moved), the ones relinked and the count
//! unchanged. A row whose archive moved (`sha256`, `code_sha256`,
//! `archive_bytes`, `stripped_bytes`) while its `exec_sha256` and every other
//! column stayed is `relinked`: the AIR moved and the device code is the
//! same; it counts as unchanged for the exit code and for the drift rule
//! below, which judges the rows whose device code moved. A scan of format v1
//! has no `exec_sha256`: a compare with one on either side is judged on
//! `code_sha256`, as before, and says so. A scan of format v2 has no
//! `text_sha256`: a compare with one on either side says so in a note and
//! the rows say nothing of it. With the column on both sides, a `changed` row
//! whose `text_sha256` is equal says the device instructions are the same, and
//! one whose values differ lists the column. That is a diagnostic: equal
//! instructions with other metadata can still be other register or resource
//! use, so the row stays `changed` and no exit code or count moves. The
//! `archive` header documents the columns, so two scans of different formats
//! are compared without it. Exit 0
//! when nothing existing moved (rows only added are listed), 3 an existing
//! row's hash or number moved, 4 a row was removed, 5 both, 2 two scans of
//! different environments or a file that is no scan, 7 every row of the base
//! moved and none changed a number or more than an alignment unit of size (the
//! device compiler's session state moved between the two scans, not a kernel:
//! scan again), 64 usage.
//! `--compare-trace` compares two encode traces
//! (`generate_metal --encode-trace`) and prints the first differing event:
//! exit 0 identical, 6 identical up to buffer numbering, 3 they differ, 2 a
//! file that is no trace of a format version this reads, or holds a line it
//! cannot read (a `refused` line, which a trace carries when a buffer had no
//! place in the numbering, is one). A binding names its buffer by allocation
//! order, so a change that allocates one more buffer early shifts every later
//! id and moves every later line. When the lines differ, each side's buffer
//! ids are renumbered by first use (the first buffer a binding names is
//! buffer 0, the next new one buffer 1) and the traces compared again with
//! every other field as text: equal renumbered traces bind their slots in the
//! same pattern under a bijection of the two sides' buffers (exit 6, the
//! pairs printed), anything else is a real difference (exit 3, the first
//! differing event printed with both sides' raw lines). Exit 6 is a
//! diagnostic, not a proof: a trace names a buffer by its allocation order
//! only, so two buffers of one length whose roles are swapped at every use
//! renumber the same. Two traces of the same format version with equal lines
//! are identical. A v1 trace numbered the buffers of the weight load in the
//! order its threads made them, so a v1 and a v2 trace are never identical:
//! they are compared renumbered, with a note, and equal lines are exit 6 at
//! best.

use std::process::ExitCode;

const USAGE: &str = "\
metal_scan [--instance NAME] [--out FILE] [--keep DIR]
metal_scan --compare BASE.tsv HEAD.tsv
metal_scan --compare-trace BASE.txt HEAD.txt";

/// The scan's format, the first line of a TSV: the columns below.
const SCAN_VERSION: &str = "# metal_scan v3";

/// The format before `text_sha256`: the columns but the last. `--compare` still reads it.
const SCAN_VERSION_V2: &str = "# metal_scan v2";

/// The format before `exec_sha256`: the columns but the last two. `--compare` still reads it.
const SCAN_VERSION_V1: &str = "# metal_scan v1";

/// The scan's columns.
const COLUMNS: [&str; 13] = [
    "lib",
    "entry",
    "constants",
    "live",
    "sha256",
    "code_sha256",
    "archive_bytes",
    "stripped_bytes",
    "max_threads",
    "width",
    "tg_static",
    "exec_sha256",
    "text_sha256",
];

/// The header keys two scans must hold equal to be compared: what the
/// hash depends on. The sources' digests are not among them (they are what
/// the compare reports through the rows).
const ENVIRONMENT: [&str; 8] = [
    "os",
    "gpu",
    "architecture",
    "language_version",
    "compile_options",
    "library_path",
    "archive",
    "gemm_path",
];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--compare", base, head] => ExitCode::from(compare::pipelines(base, head)),
        ["--compare-trace", base, head] => ExitCode::from(compare::traces(base, head)),
        ["--help"] => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        _ => scan(&args),
    }
}

#[cfg(target_os = "macos")]
fn scan(args: &[String]) -> ExitCode {
    let mut out = None;
    let mut keep = None;
    let mut instance = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--out" => &mut out,
            "--keep" => &mut keep,
            "--instance" => &mut instance,
            other => {
                eprintln!("metal_scan: {other} is not a flag of this tool\n{USAGE}");
                return ExitCode::from(64);
            }
        };
        match it.next() {
            Some(v) => *slot = Some(v.clone()),
            None => {
                eprintln!("metal_scan: {a} needs a value\n{USAGE}");
                return ExitCode::from(64);
            }
        }
    }
    let inst = match instance.as_deref().map(mac::instance_named) {
        None => &bloomery_metal::instance::HOST,
        Some(Ok(inst)) => inst,
        Some(Err(e)) => {
            eprintln!("metal_scan: {e}\n{USAGE}");
            return ExitCode::from(64);
        }
    };
    let (out, keep) = (
        out.map(std::path::PathBuf::from),
        keep.map(std::path::PathBuf::from),
    );
    match mac::run(inst, out.as_deref(), keep.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("metal_scan: {e}");
            ExitCode::from(1)
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn scan(_args: &[String]) -> ExitCode {
    eprintln!("metal_scan: refused: the scan runs on macOS only");
    ExitCode::from(78)
}

/// The GPU-code slice of an archive, read as the 64-bit little-endian Mach-O it
/// is: its load commands and its sections, by hand.
#[cfg(target_os = "macos")]
mod macho {
    use gguf::digest::{sha256_hex, sha256_hex_of};

    /// `LC_SEGMENT_64`: a segment and the `section_64` entries after it.
    const LC_SEGMENT_64: u32 = 0x19;
    /// `LC_UUID`: the slice's identity, which the scan zeroes.
    pub(super) const LC_UUID: u32 = 0x1b;
    /// The segment command's bytes before its sections, and a `section_64`'s.
    const SEGMENT_HEADER: usize = 72;
    const SECTION_HEADER: usize = 80;
    /// A section of the function's AIR, which the device does not run.
    const AIR_SECTION: &str = "__metallib";
    /// The section of the device's machine code: the one a slice must hold. It
    /// holds a Mach-O 64 of its own, the device binary.
    const CODE_SECTION: &str = "__compute";
    /// The device binary's section of instructions.
    const TEXT_SEGMENT: &str = "__TEXT";
    const TEXT_SECTION: &str = "__text";
    /// What a refusal calls the two Mach-O 64 files the scan reads.
    pub(super) const GPU_SLICE: &str = "the GPU slice";
    const DEVICE_BINARY: &str = "the __compute section";
    /// A section's type (the low byte of its flags) that holds no bytes of the file.
    const ZEROFILL: [u32; 3] = [0x1, 0xc, 0x12];

    /// One load command: its id, its offset in the slice and its size.
    pub(super) struct LoadCommand {
        pub(super) cmd: u32,
        pub(super) at: usize,
        pub(super) size: usize,
    }

    /// The little-endian word at `at` of `slice`, which `what` names in a refusal.
    fn word(what: &str, slice: &[u8], at: usize) -> Result<u32, String> {
        slice
            .get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .ok_or_else(|| format!("{what} ends inside its header at byte {at}"))
    }

    /// The load commands of a 64-bit Mach-O `slice` (which `what` names in a
    /// refusal), in file order, each inside the slice.
    pub(super) fn load_commands(what: &str, slice: &[u8]) -> Result<Vec<LoadCommand>, String> {
        if word(what, slice, 0)? != 0xfeed_facf {
            return Err(format!(
                "{what}'s magic is {:#x}, not a 64-bit Mach-O's",
                word(what, slice, 0)?
            ));
        }
        let mut commands = Vec::new();
        let mut at = 32;
        for _ in 0..word(what, slice, 16)? {
            let (cmd, size) = (word(what, slice, at)?, word(what, slice, at + 4)? as usize);
            if size < 8 {
                return Err(format!("a Mach-O load command of {size} bytes at {at}"));
            }
            if size > slice.len() - at {
                return Err(format!(
                    "the Mach-O load command at {at} of {size} bytes runs past {what}'s {} bytes",
                    slice.len()
                ));
            }
            commands.push(LoadCommand { cmd, at, size });
            at += size;
        }
        Ok(commands)
    }

    /// One section of a slice: `segment,section` and its bytes.
    pub(super) struct Section<'a> {
        pub(super) segment: String,
        pub(super) name: String,
        pub(super) bytes: &'a [u8],
    }

    /// A 16-byte NUL-padded name of a segment or section.
    fn name_of(field: &[u8]) -> Result<String, String> {
        let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
        String::from_utf8(field[..end].to_vec()).map_err(|_| {
            format!("a Mach-O segment or section name that is not UTF-8: {field:02x?}")
        })
    }

    /// The sections of every `LC_SEGMENT_64` of `slice` (which `what` names in
    /// a refusal), in the order the commands list them, each with the bytes its
    /// offset and size name.
    pub(super) fn sections<'a>(what: &str, slice: &'a [u8]) -> Result<Vec<Section<'a>>, String> {
        let mut found = Vec::new();
        for lc in load_commands(what, slice)? {
            if lc.cmd != LC_SEGMENT_64 {
                continue;
            }
            if lc.size < SEGMENT_HEADER {
                return Err(format!(
                    "{what}: the segment command at {} is {} bytes, short of its {SEGMENT_HEADER}-byte header",
                    lc.at, lc.size
                ));
            }
            let count = word(what, slice, lc.at + 64)? as usize;
            if lc.size < SEGMENT_HEADER + count * SECTION_HEADER {
                return Err(format!(
                    "{what}: the segment command at {} of {} bytes cannot hold its {count} sections",
                    lc.at, lc.size
                ));
            }
            for n in 0..count {
                let at = lc.at + SEGMENT_HEADER + n * SECTION_HEADER;
                let (name, segment) = (
                    name_of(&slice[at..at + 16])?,
                    name_of(&slice[at + 16..at + 32])?,
                );
                let size = u64::from_le_bytes(slice[at + 40..at + 48].try_into().expect("8 bytes"));
                let (offset, flags) = (
                    word(what, slice, at + 48)? as usize,
                    word(what, slice, at + 64)?,
                );
                if ZEROFILL.contains(&(flags & 0xff)) {
                    return Err(format!(
                        "{what}: the section {segment},{name} is zero-fill: it holds no bytes to hash"
                    ));
                }
                let bytes = usize::try_from(size)
                    .ok()
                    .and_then(|size| slice.get(offset..offset.checked_add(size)?))
                    .ok_or_else(|| {
                        format!(
                            "the section {segment},{name} ({size} bytes at {offset}) lies past {what}'s {} bytes",
                            slice.len()
                        )
                    })?;
                found.push(Section {
                    segment,
                    name,
                    bytes,
                });
            }
        }
        Ok(found)
    }

    /// What the device code of a slice hashes to.
    pub(super) struct Exec {
        /// The SHA-256 over the covered sections.
        pub(super) sha256: String,
        /// The covered sections, `segment,section`, in the order hashed.
        pub(super) sections: Vec<String>,
    }

    /// `segment,section` of each of `all`, space-separated; `none` for no section.
    fn listed(all: &[Section<'_>]) -> String {
        if all.is_empty() {
            return "none".to_string();
        }
        let names: Vec<String> = all
            .iter()
            .map(|s| format!("{},{}", s.segment, s.name))
            .collect();
        names.join(" ")
    }

    /// The one `__compute` section of a slice's `all` sections: nothing would
    /// be proven by a slice with none, and which of two is the device's is
    /// not for the scan to guess.
    fn code_section<'a, 'b>(all: &'b [Section<'a>]) -> Result<&'b Section<'a>, String> {
        let mut found = all.iter().filter(|s| s.name == CODE_SECTION);
        match (found.next(), found.next()) {
            (Some(code), None) => Ok(code),
            (None, _) => Err(format!(
                "{GPU_SLICE} holds no {CODE_SECTION} section (its sections: {})",
                listed(all)
            )),
            (Some(_), Some(_)) => Err(format!(
                "{GPU_SLICE} holds more than one {CODE_SECTION} section (its sections: {})",
                listed(all)
            )),
        }
    }

    /// The scan's `exec_sha256` of `slice`: its sections but the AIR's, each as
    /// `segment,section,length\n` and its bytes. A slice with no machine-code
    /// section is refused: nothing would be proven.
    pub(super) fn exec(slice: &[u8]) -> Result<Exec, String> {
        let all = sections(GPU_SLICE, slice)?;
        code_section(&all)?;
        let covered: Vec<&Section<'_>> = all.iter().filter(|s| s.name != AIR_SECTION).collect();
        let names: Vec<String> = covered
            .iter()
            .map(|s| format!("{},{}", s.segment, s.name))
            .collect();
        let heads: Vec<String> = covered
            .iter()
            .zip(&names)
            .map(|(s, name)| format!("{name},{}\n", s.bytes.len()))
            .collect();
        let parts = covered
            .iter()
            .zip(&heads)
            .flat_map(|(s, head)| [head.as_bytes(), s.bytes]);
        Ok(Exec {
            sha256: sha256_hex_of(parts),
            sections: names,
        })
    }

    /// The scan's `text_sha256` of `slice`: the SHA-256 of the bytes of
    /// `__TEXT,__text` in the Mach-O 64 that its `__compute` section holds.
    /// The compiler's link and statistics notes beside the instructions are
    /// not in it. A `__compute` section that is no Mach-O 64, or has no
    /// `__TEXT,__text` or two, is refused.
    pub(super) fn text(slice: &[u8]) -> Result<String, String> {
        let all = sections(GPU_SLICE, slice)?;
        let binary = sections(DEVICE_BINARY, code_section(&all)?.bytes)?;
        let mut found = binary
            .iter()
            .filter(|s| s.segment == TEXT_SEGMENT && s.name == TEXT_SECTION);
        match (found.next(), found.next()) {
            (Some(text), None) => Ok(sha256_hex(text.bytes)),
            (None, _) => Err(format!(
                "{DEVICE_BINARY} holds no {TEXT_SEGMENT},{TEXT_SECTION} section (its sections: {})",
                listed(&binary)
            )),
            (Some(_), Some(_)) => Err(format!(
                "{DEVICE_BINARY} holds more than one {TEXT_SEGMENT},{TEXT_SECTION} section (its sections: {})",
                listed(&binary)
            )),
        }
    }

    /// A slice of `sections` (name, bytes) in one `__TEXT` segment, then an
    /// `LC_UUID` of sixteen `uuid` bytes: the layout the device compiler writes.
    #[cfg(test)]
    pub(super) fn built(uuid: u8, sections: &[(&str, &[u8])]) -> Vec<u8> {
        let sections: Vec<(&str, &str, &[u8])> = sections
            .iter()
            .map(|(name, bytes)| (TEXT_SEGMENT, *name, *bytes))
            .collect();
        built_in(uuid, &sections)
    }

    /// `built` with each section's segment (segment, name, bytes) named.
    #[cfg(test)]
    pub(super) fn built_in(uuid: u8, sections: &[(&str, &str, &[u8])]) -> Vec<u8> {
        let segment = SEGMENT_HEADER + sections.len() * SECTION_HEADER;
        let first = 32 + segment + 24;
        let mut out = Vec::new();
        for w in [
            0xfeed_facf_u32,
            0x100_0013,
            0x212,
            0,
            2,
            (segment + 24) as u32,
            0,
            0,
        ] {
            out.extend(w.to_le_bytes());
        }
        out.extend(LC_SEGMENT_64.to_le_bytes());
        out.extend((segment as u32).to_le_bytes());
        out.extend(name16("__TEXT"));
        for w in [0u64, 0, first as u64, 0] {
            out.extend(w.to_le_bytes());
        }
        for w in [7u32, 5, sections.len() as u32, 0] {
            out.extend(w.to_le_bytes());
        }
        let mut offset = first;
        for (owner, name, bytes) in sections {
            out.extend(name16(name));
            out.extend(name16(owner));
            out.extend(0u64.to_le_bytes());
            out.extend((bytes.len() as u64).to_le_bytes());
            out.extend((offset as u32).to_le_bytes());
            out.extend([0u8; 28]);
            offset += bytes.len();
        }
        out.extend(LC_UUID.to_le_bytes());
        out.extend(24u32.to_le_bytes());
        out.extend([uuid; 16]);
        for (_, _, bytes) in sections {
            out.extend(*bytes);
        }
        out
    }

    #[cfg(test)]
    fn name16(name: &str) -> [u8; 16] {
        let mut field = [0u8; 16];
        field[..name.len()].copy_from_slice(name.as_bytes());
        field
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        const SECTIONS: [(&str, &[u8]); 3] = [
            ("__reflection", b"reflection"),
            ("__compute", b"machine code"),
            ("__metallib", b"the function's AIR"),
        ];

        fn exec_of(slice: &[u8]) -> Exec {
            exec(slice).expect("a slice of the device compiler's layout")
        }

        #[test]
        fn the_exec_hash_follows_the_device_code_sections_only() {
            let base = exec_of(&built(1, &SECTIONS));
            assert_eq!(base.sections, ["__TEXT,__reflection", "__TEXT,__compute"]);
            let with = |section: usize, bytes: &'static [u8]| {
                let mut sections = SECTIONS;
                sections[section].1 = bytes;
                exec_of(&built(1, &sections)).sha256
            };
            // The machine code or its reflection moving moves the hash.
            assert_ne!(with(1, b"machine codf"), base.sha256);
            assert_ne!(with(1, b"machine code, longer"), base.sha256);
            assert_ne!(with(0, b"reflectioN"), base.sha256);
            // The AIR moving, in its bytes or its size, and the slice's identity do not.
            assert_eq!(with(2, b"another AIR, a longer one"), base.sha256);
            assert_eq!(exec_of(&built(9, &SECTIONS)).sha256, base.sha256);
            // A byte moved from one section to its neighbour is a different program.
            let split = |at: usize| {
                let bytes = b"reflectionmachine code";
                exec_of(&built(
                    1,
                    &[("__reflection", &bytes[..at]), ("__compute", &bytes[at..])],
                ))
                .sha256
            };
            assert_ne!(split(10), split(11));
        }

        #[test]
        fn the_load_commands_are_listed_in_file_order() {
            let slice = built(1, &SECTIONS);
            let commands = load_commands(GPU_SLICE, &slice).expect("load commands");
            let seen: Vec<(u32, usize, usize)> =
                commands.iter().map(|c| (c.cmd, c.at, c.size)).collect();
            let segment = SEGMENT_HEADER + 3 * SECTION_HEADER;
            assert_eq!(
                seen,
                [(LC_SEGMENT_64, 32, segment), (LC_UUID, 32 + segment, 24)]
            );
        }

        #[test]
        fn a_slice_it_cannot_read_is_refused_by_name() {
            let slice = built(1, &SECTIONS);
            let refused = |bytes: &[u8]| exec(bytes).err().expect("the slice is refused");
            // Not a 64-bit Mach-O: another magic, and the archive's fat container.
            let mut other = slice.clone();
            other[..4].copy_from_slice(&0xcafe_babe_u32.to_le_bytes());
            assert!(
                refused(&other).contains("not a 64-bit Mach-O"),
                "{}",
                refused(&other)
            );
            assert!(refused(&[0xca, 0xfe]).contains("ends inside its header"));
            // A section past the slice's end.
            let cut = &slice[..slice.len() - 1];
            let err = refused(cut);
            assert!(
                err.contains("__TEXT,__metallib") && err.contains("lies past the GPU slice"),
                "{err}"
            );
            // A load command past the slice's end.
            let err = refused(&slice[..100]);
            assert!(err.contains("runs past the GPU slice"), "{err}");
            // No machine code section, and two.
            let err = refused(&built(1, &[SECTIONS[0], SECTIONS[2]]));
            assert!(
                err.contains("no __compute section")
                    && err.contains("__TEXT,__reflection __TEXT,__metallib"),
                "{err}"
            );
            let err = refused(&built(1, &[]));
            assert!(
                err.contains("no __compute section (its sections: none)"),
                "{err}"
            );
            let err = refused(&built(1, &[SECTIONS[1], SECTIONS[1]]));
            assert!(
                err.contains("more than one __compute section")
                    && err.contains("__TEXT,__compute __TEXT,__compute"),
                "{err}"
            );
            // A section with no bytes in the file.
            let mut zerofill = slice.clone();
            // The flags of the second section header, `__compute`.
            let flags = 32 + SEGMENT_HEADER + SECTION_HEADER + 64;
            zerofill[flags..flags + 4].copy_from_slice(&1u32.to_le_bytes());
            let err = refused(&zerofill);
            assert!(err.contains("__TEXT,__compute is zero-fill"), "{err}");
            // A segment command too small for its sections.
            let mut small = slice.clone();
            small[32 + 64..32 + 68].copy_from_slice(&9u32.to_le_bytes());
            let err = refused(&small);
            assert!(err.contains("cannot hold its 9 sections"), "{err}");
        }

        const CODE: &[u8] = b"the instructions";
        const LINK: &[u8] = b"link notes";

        /// A device binary as the compiler writes one: the instructions in
        /// `__TEXT,__text`, its notes in `__GPU_*` sections beside them.
        fn binary(code: &[u8], link: &[u8]) -> Vec<u8> {
            built_in(
                0,
                &[
                    (TEXT_SEGMENT, TEXT_SECTION, code),
                    ("__GPU_METADATA", "__compute", b"metadata"),
                    ("__GPU_LD_MD", "__compute", link),
                    ("__GPU_STATS_MD", "__compute", b"statistics"),
                ],
            )
        }

        /// A slice whose `__compute` section holds `device` and whose AIR is `air`.
        fn slice_of(device: &[u8], air: &[u8]) -> Vec<u8> {
            built(
                1,
                &[
                    ("__reflection", b"reflection"),
                    ("__compute", device),
                    ("__metallib", air),
                ],
            )
        }

        fn text_of(slice: &[u8]) -> String {
            text(slice).expect("a slice of the device compiler's layout")
        }

        #[test]
        fn the_text_hash_follows_the_instructions_only() {
            let hash = |code: &[u8], link: &[u8]| text_of(&slice_of(&binary(code, link), b"air"));
            let base = hash(CODE, LINK);
            // The compiler's notes beside the instructions moving, in a byte or in length, does not move it.
            assert_eq!(hash(CODE, b"link notez"), base);
            assert_eq!(hash(CODE, b"other link notes, a longer one"), base);
            // The instructions moving, in a byte or in length, does.
            assert_ne!(hash(b"the instructionS", LINK), base);
            assert_ne!(hash(b"the instructions, more", LINK), base);
            assert_eq!(base, sha256_hex(CODE));
            // Neither does the AIR, nor the slice's identity.
            assert_eq!(
                text_of(&slice_of(&binary(CODE, LINK), b"another AIR")),
                base
            );
            let other_uuid = built(
                9,
                &[
                    ("__reflection", b"reflection"),
                    ("__compute", &binary(CODE, LINK)),
                    ("__metallib", b"air"),
                ],
            );
            assert_eq!(text_of(&other_uuid), base);
            // The exec hash covers the notes: the move the text hash sees past.
            let exec_with = |link: &[u8]| exec_of(&slice_of(&binary(CODE, link), b"air")).sha256;
            assert_ne!(exec_with(LINK), exec_with(b"other link notes"));
        }

        #[test]
        fn a_device_binary_it_cannot_read_is_refused_by_name() {
            let refused = |slice: &[u8]| text(slice).expect_err("the slice is refused");
            // A __compute section that is no 64-bit Mach-O, and one too short to say.
            let err = refused(&slice_of(b"machine code", b"air"));
            assert!(
                err.contains("the __compute section's magic is")
                    && err.contains("not a 64-bit Mach-O"),
                "{err}"
            );
            let err = refused(&slice_of(b"ma", b"air"));
            assert!(
                err.contains("the __compute section ends inside its header"),
                "{err}"
            );
            // No __TEXT,__text: none at all, the name in another segment, and two.
            let err = refused(&slice_of(
                &built_in(0, &[("__GPU_LD_MD", "__compute", LINK)]),
                b"air",
            ));
            assert!(
                err.contains("the __compute section holds no __TEXT,__text section")
                    && err.contains("(its sections: __GPU_LD_MD,__compute)"),
                "{err}"
            );
            let err = refused(&slice_of(
                &built_in(0, &[("__DATA", TEXT_SECTION, CODE)]),
                b"air",
            ));
            assert!(err.contains("holds no __TEXT,__text section"), "{err}");
            let twice = built_in(
                0,
                &[
                    (TEXT_SEGMENT, TEXT_SECTION, CODE),
                    (TEXT_SEGMENT, TEXT_SECTION, LINK),
                ],
            );
            let err = refused(&slice_of(&twice, b"air"));
            assert!(
                err.contains("the __compute section holds more than one __TEXT,__text section"),
                "{err}"
            );
            // A section past the device binary's end, and a load command past it.
            let device = binary(CODE, LINK);
            let err = refused(&slice_of(&device[..device.len() - 1], b"air"));
            assert!(
                err.contains("__GPU_STATS_MD,__compute")
                    && err.contains("lies past the __compute section's"),
                "{err}"
            );
            let err = refused(&slice_of(&device[..100], b"air"));
            assert!(
                err.contains("runs past the __compute section's 100 bytes"),
                "{err}"
            );
            // The slice itself: no __compute section, and a section past the slice.
            let err = refused(&built(1, &[SECTIONS[0], SECTIONS[2]]));
            assert!(
                err.contains("the GPU slice holds no __compute section"),
                "{err}"
            );
            let slice = slice_of(&device, b"air");
            let err = refused(&slice[..slice.len() - 1]);
            assert!(err.contains("lies past the GPU slice"), "{err}");
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;

    use bloomery_metal::device::MetalDevice;
    use bloomery_metal::instance::{Instance, REGISTERED};
    use bloomery_metal::kernels::Kernels;
    use bloomery_metal::library::{GemmPath, Library, LibraryPath, sources_of};
    use bloomery_metal::scan::{self, Built};
    use gguf::digest::sha256_hex;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSProcessInfo, NSString, NSURL};
    use objc2_metal::{
        MTLBinaryArchive, MTLBinaryArchiveDescriptor, MTLComputePipelineDescriptor, MTLDevice,
    };

    use super::{SCAN_VERSION, macho};

    /// A kernel appended to a library's source for the calibration: no
    /// pipeline of the engine is this one.
    const CALIBRATION: &str = "
// scan calibration
kernel void bloomery_scan_calibration(device float * x [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    x[i] = 0.0f;
}
";

    /// (library, entry, constants): a pipeline's identity in the scan.
    type Key = (&'static str, String, String);

    /// One pipeline of a scan: what was built and its archive, as serialized.
    struct Entry {
        built: Built,
        live: bool,
        archive: Vec<u8>,
    }

    /// The function constants as the rows print them, `u<index>=<value>` joined by commas.
    fn constants_of(built: &Built) -> String {
        if built.constants.is_empty() {
            return "-".to_string();
        }
        built
            .constants
            .iter()
            .map(|(index, value)| format!("u{index}={value}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The pipelines `kernels` built, in `lib` terms: each is told apart by the
    /// library object it came from.
    fn tagged(
        built: Vec<Built>,
        main: &Library,
        tensor: &Library,
    ) -> Result<Vec<(&'static str, Built)>, String> {
        built
            .into_iter()
            .map(|b| {
                let lib = if b.library == main.scan_id() {
                    "main"
                } else if b.library == tensor.scan_id() {
                    "tensor"
                } else {
                    return Err(format!(
                        "{}: built from a library the scan did not make",
                        b.entry
                    ));
                };
                Ok((lib, b))
            })
            .collect()
    }

    /// The GPU-code slice of an archive: the archive is a fat container of
    /// big-endian words (magic, slice count, then per slice cputype,
    /// cpusubtype, offset, size, alignment), one slice the function's AIR
    /// and one the device's machine code. Its byte range.
    fn code_slice(archive: &[u8]) -> Result<core::ops::Range<usize>, String> {
        const MAGIC: u32 = 0xcbfe_babe;
        const AIR: u32 = 0x0100_0017;
        let word = |at: usize| -> Result<u32, String> {
            archive
                .get(at..at + 4)
                .map(|w| u32::from_be_bytes([w[0], w[1], w[2], w[3]]))
                .ok_or_else(|| format!("the archive ends inside its header at byte {at}"))
        };
        if word(0)? != MAGIC {
            return Err(format!(
                "the archive's magic is {:#x}, not {MAGIC:#x}",
                word(0)?
            ));
        }
        let mut code = None;
        for slice in 0..word(4)? as usize {
            let at = 8 + 20 * slice;
            if word(at)? == AIR {
                continue;
            }
            let (offset, size) = (word(at + 8)? as usize, word(at + 12)? as usize);
            if offset + size > archive.len() {
                return Err(format!(
                    "slice {slice} lies past the archive's {} bytes",
                    archive.len()
                ));
            }
            if code.replace(offset..offset + size).is_some() {
                return Err("the archive holds more than one non-AIR slice".to_string());
            }
        }
        code.ok_or_else(|| "the archive holds no GPU-code slice".to_string())
    }

    /// The device compiler's archive of one pipeline: serialized to a file of
    /// `scratch`, read back.
    fn archive_of(
        device: &ProtocolObject<dyn MTLDevice>,
        built: &Built,
        scratch: &Path,
    ) -> Result<Vec<u8>, String> {
        let descriptor = MTLBinaryArchiveDescriptor::new();
        let archive = device
            .newBinaryArchiveWithDescriptor_error(&descriptor)
            .map_err(|e| format!("the empty archive: {}", e.localizedDescription()))?;
        let pipeline = MTLComputePipelineDescriptor::new();
        pipeline.setComputeFunction(Some(&built.function));
        archive
            .addComputePipelineFunctionsWithDescriptor_error(&pipeline)
            .map_err(|e| format!("added to an archive: {}", e.localizedDescription()))?;
        let path = scratch.join("archive.bin");
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        archive
            .serializeToURL_error(&url)
            .map_err(|e| format!("serialized: {}", e.localizedDescription()))?;
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(bytes)
    }

    /// Every pipeline of both GEMM programs from `main` and `tensor`, the
    /// lazily built ones built, each serialized. The engine's own set on this
    /// device is one of the two programs' sets.
    fn archives(
        device: &MetalDevice,
        main: &Library,
        tensor: &Library,
        scratch: &Path,
    ) -> Result<BTreeMap<Key, Entry>, String> {
        let live_path = device.library().gemm_path();
        scan::start();
        let simdgroup =
            Kernels::from_library(&main.with_simdgroup_gemms()).map_err(|e| e.to_string())?;
        simdgroup.build_deferred().map_err(|e| e.to_string())?;
        let simdgroup_set = tagged(scan::take(), main, tensor)?;
        scan::start();
        Kernels::from_library(&main.with_tensor_gemms(tensor.clone()))
            .map_err(|e| e.to_string())?;
        let tensor_set = tagged(scan::take(), main, tensor)?;

        let mut entries: BTreeMap<Key, Entry> = BTreeMap::new();
        for (set, is_live) in [
            (simdgroup_set, live_path == GemmPath::Simdgroup),
            (tensor_set, live_path == GemmPath::TensorOps),
        ] {
            for (lib, built) in set {
                let key = (lib, built.entry.clone(), constants_of(&built));
                if let Some(entry) = entries.get_mut(&key) {
                    entry.live |= is_live;
                    continue;
                }
                let archive = archive_of(device.metal(), &built, scratch)
                    .map_err(|e| format!("{} {} {}: {e}", key.0, key.1, key.2))?;
                entries.insert(
                    key,
                    Entry {
                        built,
                        live: is_live,
                        archive,
                    },
                );
            }
        }
        Ok(entries)
    }

    /// Zero `len` bytes at `at`.
    fn zero(archive: &mut [u8], at: usize, len: usize) {
        archive[at..at + len].fill(0);
    }

    /// The archive with the library's identity zeroed (see the module's
    /// notes), and the bytes that covered.
    fn strip(archive: &[u8]) -> Result<(Vec<u8>, usize), String> {
        let mut out = archive.to_vec();
        let mut covered = 0;
        // The module signature: a maximal run of exactly 64 uppercase hex digits.
        let hex = |b: u8| b.is_ascii_digit() || (b'A'..=b'F').contains(&b);
        let mut i = 0;
        while i < out.len() {
            if !hex(out[i]) {
                i += 1;
                continue;
            }
            let run = out[i..].iter().take_while(|b| hex(**b)).count();
            if run == 64 {
                zero(&mut out, i, 64);
                covered += 64;
            }
            i += run;
        }
        // The module UUID: every `UUID` record (tag, 16 in a 16-bit word, the
        // 16 bytes), and every other copy of those bytes.
        let mut uuids: Vec<[u8; 16]> = Vec::new();
        for at in 0..out.len().saturating_sub(22) {
            if &out[at..at + 6] == b"UUID\x10\x00" {
                let mut uuid = [0u8; 16];
                uuid.copy_from_slice(&out[at + 6..at + 22]);
                if !uuids.contains(&uuid) {
                    uuids.push(uuid);
                }
            }
        }
        for uuid in uuids.iter().filter(|u| **u != [0u8; 16]) {
            let mut at = 0;
            while at + 16 <= out.len() {
                if out[at..at + 16] == uuid[..] {
                    zero(&mut out, at, 16);
                    covered += 16;
                    at += 16;
                } else {
                    at += 1;
                }
            }
        }
        // The GPU slice's Mach-O LC_UUID.
        let code = code_slice(&out)?;
        let lc_uuids: Vec<usize> = macho::load_commands(macho::GPU_SLICE, &out[code.clone()])?
            .iter()
            .filter(|lc| lc.cmd == macho::LC_UUID && lc.size >= 24)
            .map(|lc| code.start + lc.at + 8)
            .collect();
        for at in lc_uuids {
            zero(&mut out, at, 16);
            covered += 16;
        }
        Ok((out, covered))
    }

    /// `stripped` and `twin` (both stripped) are byte-equal: nothing else of
    /// the archive depends on the rest of the library.
    fn same_but_identity(stripped: &[u8], twin: &[u8]) -> Result<(), String> {
        if stripped.len() != twin.len() {
            return Err(format!(
                "the archive is {} bytes and its calibration twin {}",
                stripped.len(),
                twin.len()
            ));
        }
        let differ: Vec<usize> = (0..stripped.len())
            .filter(|&i| stripped[i] != twin[i])
            .collect();
        if differ.is_empty() {
            return Ok(());
        }
        Err(format!(
            "{} bytes differ from the calibration twin after the identity is stripped, the first at byte {}: something else of the archive depends on the rest of the library",
            differ.len(),
            differ[0]
        ))
    }

    /// The registered instance named `name`, or the refusal naming every
    /// registered one.
    pub(super) fn instance_named(name: &str) -> Result<&'static Instance, String> {
        REGISTERED
            .iter()
            .find(|inst| inst.name == name)
            .ok_or_else(|| {
                let names: Vec<&str> = REGISTERED.iter().map(|inst| inst.name).collect();
                format!(
                    "{name} is not a registered instance (registered: {})",
                    names.join(", ")
                )
            })
    }

    pub(super) fn run(
        inst: &'static Instance,
        out: Option<&Path>,
        keep: Option<&Path>,
    ) -> Result<(), String> {
        let device = MetalDevice::open(inst).map_err(|e| e.to_string())?;
        let (source, tensor_source) = sources_of(inst).map_err(|e| e.to_string())?;
        if device.library().path() != LibraryPath::Source {
            return Err(
                "the library is the build's metallib (the Metal Toolchain is installed): the calibration compiles the run-time source, so the scan would not be of what runs"
                    .to_string(),
            );
        }
        let scratch = std::env::temp_dir().join(format!("metal_scan-{}", std::process::id()));
        std::fs::create_dir_all(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
        if let Some(dir) = keep {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let main = device.library().with_simdgroup_gemms();
        let tensor = Library::tensor_from_source_text(&device, tensor_source)
            .map_err(|e| format!("the tensor-op library: {e}"))?;
        let real = archives(&device, &main, &tensor, &scratch)?;
        let main_c = Library::from_source_text(&device, &format!("{source}{CALIBRATION}"))
            .map_err(|e| format!("the calibration library: {e}"))?;
        let tensor_c =
            Library::tensor_from_source_text(&device, &format!("{tensor_source}{CALIBRATION}"))
                .map_err(|e| format!("the calibration tensor-op library: {e}"))?;
        let twin = archives(&device, &main_c, &tensor_c, &scratch)?;
        std::fs::remove_dir(&scratch).map_err(|e| format!("{}: {e}", scratch.display()))?;
        if real.len() != twin.len() {
            return Err(format!(
                "the scan has {} pipelines and the calibration build {}",
                real.len(),
                twin.len()
            ));
        }

        let mut rows = String::new();
        let mut exec_sections = BTreeSet::new();
        for (key, entry) in &real {
            let name = format!("{} {} {}", key.0, key.1, key.2);
            let other = twin
                .get(key)
                .ok_or_else(|| format!("{name}: the calibration build has no such pipeline"))?;
            if let Some(dir) = keep {
                let file = dir.join(format!("{}-{}-{}.bin", key.0, key.1, key.2));
                std::fs::write(&file, &entry.archive)
                    .map_err(|e| format!("{}: {e}", file.display()))?;
            }
            let (masked, stripped) = strip(&entry.archive).map_err(|e| format!("{name}: {e}"))?;
            let (twin_masked, _) =
                strip(&other.archive).map_err(|e| format!("{name} (calibration): {e}"))?;
            same_but_identity(&masked, &twin_masked).map_err(|e| format!("{name}: {e}"))?;
            let code = code_slice(&masked).map_err(|e| format!("{name}: {e}"))?;
            let exec = macho::exec(&masked[code.clone()]).map_err(|e| format!("{name}: {e}"))?;
            let text = macho::text(&masked[code.clone()]).map_err(|e| format!("{name}: {e}"))?;
            exec_sections.extend(exec.sections);
            rows.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                key.0,
                key.1,
                key.2,
                if entry.live { "y" } else { "n" },
                sha256_hex(&masked),
                sha256_hex(&masked[code]),
                masked.len(),
                stripped,
                entry.built.max_threads,
                entry.built.execution_width,
                entry.built.static_threadgroup_bytes,
                exec.sha256,
                text,
            ));
        }

        let units = device.library().matrix_units();
        let mut text = String::new();
        text.push_str(SCAN_VERSION);
        text.push('\n');
        let mut head = |key: &str, value: String| text.push_str(&format!("# {key}\t{value}\n"));
        head(
            "os",
            NSProcessInfo::processInfo()
                .operatingSystemVersionString()
                .to_string(),
        );
        head("gpu", device.metal().name().to_string());
        head("architecture", units.architecture);
        // The languages and options `Library::source_of` compiles with.
        head("language_version", "main=3.2 tensor=4.0".to_string());
        head(
            "compile_options",
            "mathMode=safe fpFunctions=precise".to_string(),
        );
        head("library_path", device.library().path().name().to_string());
        head("gemm_path", device.library().gemm_path().name().to_string());
        head(
            "archive",
            "MTLBinaryArchive of one function (descriptor defaults), serializeToURL; the library's identity is zeroed (the 64-hex module signature, the module UUID, the GPU slice's LC_UUID: stripped_bytes) and checked against a build of the library with one unused kernel appended; sha256 of the rest, code_sha256 of its GPU-code slice, exec_sha256 of that slice's Mach-O sections but __metallib (the function's AIR), each as segment,section,length and its bytes in the order the slice lists them, the sections hashed named in exec_sections, text_sha256 of the bytes of __TEXT,__text in the Mach-O 64 that slice's __compute section holds (the instructions the GPU runs, without the compiler's link and statistics notes beside them)"
                .to_string(),
        );
        head(
            "exec_sections",
            exec_sections.into_iter().collect::<Vec<_>>().join(" "),
        );
        head("instance", inst.name.to_string());
        head("source_sha256_main", sha256_hex(source.as_bytes()));
        head("source_sha256_tensor", sha256_hex(tensor_source.as_bytes()));
        text.push_str(&format!("{}\n", super::COLUMNS.join("\t")));
        text.push_str(&rows);
        match out {
            Some(path) => {
                std::fs::write(path, &text).map_err(|e| format!("{}: {e}", path.display()))?;
            }
            None => print!("{text}"),
        }
        Ok(())
    }

    #[cfg(test)]
    mod instance_tests {
        use super::*;

        #[test]
        fn an_instance_is_named_by_the_registry() {
            for inst in REGISTERED {
                assert_eq!(instance_named(inst.name), Ok(inst));
            }
            let refused = instance_named("no_such_instance").expect_err("no such instance");
            for inst in REGISTERED {
                assert!(refused.contains(inst.name), "{refused}");
            }
        }
    }
}

/// The two compares: text in, a table out. Pure, so they run on any OS.
mod compare {
    use std::collections::{BTreeMap, HashMap};

    use super::{COLUMNS, ENVIRONMENT, SCAN_VERSION, SCAN_VERSION_V1, SCAN_VERSION_V2};

    /// The trace format's first line (`bloomery_metal::trace::VERSION`, which
    /// exists only in an encode-trace build).
    const TRACE_VERSION: &str = "# bloomery encode trace v2";

    /// The first line of the older format, read for the compare of a trace
    /// made before the move: its buffer numbers followed thread scheduling.
    const TRACE_VERSION_V1: &str = "# bloomery encode trace v1";

    /// The bytes an archive may change size by between two sessions of the
    /// device compiler (one 16-byte alignment unit, twice) and still be the
    /// same function.
    const DRIFT_ALIGN: i64 = 32;

    /// The places of the columns in a row.
    const LIVE: usize = 3;
    const SHA: usize = 4;
    const CODE: usize = 5;
    const SIZE: usize = 6;
    const STRIPPED: usize = 7;
    const EXEC: usize = 11;
    const TEXT: usize = 12;

    /// The columns a row may move in while it stays `relinked`: the archive
    /// as the device compiler wrote it, whose AIR the device does not run.
    const ARCHIVE_COLUMNS: [usize; 4] = [SHA, CODE, SIZE, STRIPPED];

    /// A row that moved, as `table` lists it.
    struct Changed {
        key: String,
        /// The columns that moved, `name a -> b`.
        moved: Vec<String>,
        /// The archive moved and the GPU-code slice did not.
        air_only: bool,
        /// Both scans have `text_sha256` and it is equal.
        text_equal: bool,
        /// No number moved and the size moved by no more than an alignment unit.
        hash_only: bool,
    }

    /// One scan: its header and its rows keyed by (lib, entry, constants).
    struct Scan {
        header: BTreeMap<String, String>,
        rows: BTreeMap<String, Vec<String>>,
        /// The columns the scan has: all of them (format v3), all but `text_sha256`
        /// (v2) or all but that and `exec_sha256` (v1).
        width: usize,
    }

    impl Scan {
        /// The scan has the column at `column`.
        fn has(&self, column: usize) -> bool {
            self.width > column
        }
    }

    fn read(path: &str) -> Result<Scan, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        let columns = match lines.next() {
            Some(SCAN_VERSION) => &COLUMNS[..],
            Some(SCAN_VERSION_V2) => &COLUMNS[..TEXT],
            Some(SCAN_VERSION_V1) => &COLUMNS[..EXEC],
            _ => {
                return Err(format!(
                    "{path}: not a {SCAN_VERSION}, {SCAN_VERSION_V2} or {SCAN_VERSION_V1} scan"
                ));
            }
        };
        let mut header = BTreeMap::new();
        let mut rows = BTreeMap::new();
        let mut has_columns = false;
        for line in lines {
            if let Some(rest) = line.strip_prefix("# ") {
                let (key, value) = rest
                    .split_once('\t')
                    .ok_or_else(|| format!("{path}: header line without a tab: {line}"))?;
                header.insert(key.to_string(), value.to_string());
            } else if !has_columns {
                if line.split('\t').ne(columns.iter().copied()) {
                    return Err(format!(
                        "{path}: the column line is not {columns:?}: {line}"
                    ));
                }
                has_columns = true;
            } else {
                let cells: Vec<String> = line.split('\t').map(str::to_string).collect();
                if cells.len() != columns.len() {
                    return Err(format!("{path}: a row of {} cells: {line}", cells.len()));
                }
                let key = format!("{} {} {}", cells[0], cells[1], cells[2]);
                if rows.insert(key.clone(), cells).is_some() {
                    return Err(format!("{path}: the row {key} twice"));
                }
            }
        }
        if !has_columns {
            return Err(format!("{path}: no column line"));
        }
        for key in ENVIRONMENT {
            if header.get(key).is_none_or(String::is_empty) {
                return Err(format!("{path}: missing environment header {key}"));
            }
        }
        for (key, row) in &rows {
            if !row[SIZE].parse::<i64>().is_ok_and(|n| n >= 0) {
                return Err(format!(
                    "{path}: {key}: invalid archive_bytes {}",
                    row[SIZE]
                ));
            }
            for hash in [EXEC, TEXT].into_iter().filter(|&c| c < columns.len()) {
                if row[hash].is_empty() {
                    return Err(format!("{path}: {key}: empty {}", COLUMNS[hash]));
                }
            }
        }
        Ok(Scan {
            header,
            rows,
            width: columns.len(),
        })
    }

    /// The table of two scans; the exit code is `ptx-compare`'s.
    pub(super) fn pipelines(base: &str, head: &str) -> u8 {
        let (a, b) = match (read(base), read(head)) {
            (Ok(a), Ok(b)) => (a, b),
            (a, b) => {
                for e in [a.err(), b.err()].into_iter().flatten() {
                    eprintln!("metal_scan: {e}");
                }
                return 2;
            }
        };
        // `archive` documents the columns, so scans of two formats differ in it by
        // construction; the columns they share are made the same way.
        let formats_differ = a.width != b.width;
        let differing: Vec<String> = ENVIRONMENT
            .iter()
            .filter(|&&k| !(formats_differ && k == "archive"))
            .filter(|k| a.header.get(**k) != b.header.get(**k))
            .map(|k| format!("{k}: {:?} against {:?}", a.header.get(*k), b.header.get(*k)))
            .collect();
        if !differing.is_empty() {
            eprintln!(
                "metal_scan: the scans are of different environments, so their hashes do not compare:"
            );
            for line in &differing {
                eprintln!("  {line}");
            }
            return 2;
        }
        let (lines, rc) = table(&a, &b);
        for line in lines {
            println!("{line}");
        }
        rc
    }

    /// The table of two scans of one environment, as lines, and the exit code.
    fn table(a: &Scan, b: &Scan) -> (Vec<String>, u8) {
        // `exec_sha256` judges a row only when both sides have it, and
        // `text_sha256` is compared only when both sides have it.
        let judged = a.has(EXEC) && b.has(EXEC);
        let texted = a.has(TEXT) && b.has(TEXT);
        let columns = match (judged, texted) {
            (_, true) => TEXT + 1,
            (true, false) => EXEC + 1,
            (false, false) => EXEC,
        };
        let (mut added, mut removed, mut changed, mut relinked, mut same) =
            (vec![], vec![], Vec::<Changed>::new(), vec![], 0usize);
        for (key, row) in &a.rows {
            match b.rows.get(key) {
                None => removed.push(key.clone()),
                Some(other) => {
                    let moved: Vec<usize> =
                        (LIVE..columns).filter(|&c| row[c] != other[c]).collect();
                    if moved.is_empty() {
                        same += 1;
                    } else if judged
                        && row[EXEC] == other[EXEC]
                        && moved.iter().all(|c| ARCHIVE_COLUMNS.contains(c))
                    {
                        relinked.push(key.clone());
                    } else {
                        let moved_text: Vec<String> = moved
                            .iter()
                            .map(|&c| format!("{} {} -> {}", COLUMNS[c], row[c], other[c]))
                            .collect();
                        let air_only = row[SHA] != other[SHA] && row[CODE] == other[CODE];
                        let size = |r: &[String]| {
                            r[SIZE]
                                .parse::<i64>()
                                .expect("read validated archive_bytes")
                        };
                        let hash_only = row[LIVE] == other[LIVE]
                            && row[STRIPPED..EXEC] == other[STRIPPED..EXEC]
                            && (size(row) - size(other)).abs() <= DRIFT_ALIGN;
                        changed.push(Changed {
                            key: key.clone(),
                            moved: moved_text,
                            air_only,
                            text_equal: texted && row[TEXT] == other[TEXT],
                            hash_only,
                        });
                    }
                }
            }
        }
        for key in b.rows.keys() {
            if !a.rows.contains_key(key) {
                added.push(key.clone());
            }
        }
        let mut lines = vec![format!(
            "metal-scan compare: base {} rows, head {} rows",
            a.rows.len(),
            b.rows.len()
        )];
        let source = |k: &str| a.header.get(k) == b.header.get(k);
        lines.push(format!(
            "sources: main {}, tensor {}",
            if source("source_sha256_main") {
                "same text"
            } else {
                "text moved"
            },
            if source("source_sha256_tensor") {
                "same text"
            } else {
                "text moved"
            },
        ));
        let has = |scan: &Scan, column: usize| if scan.has(column) { "yes" } else { "no" };
        if !judged {
            lines.push(format!(
                "note: changed is judged on code_sha256 because a side has no exec_sha256 (base has it: {}, head has it: {})",
                has(a, EXEC),
                has(b, EXEC)
            ));
        }
        if !texted {
            lines.push(format!(
                "note: the device instructions are not compared because a side has no text_sha256 (base has it: {}, head has it: {})",
                has(a, TEXT),
                has(b, TEXT)
            ));
        }
        for key in &added {
            lines.push(format!("added    {key}"));
        }
        for key in &removed {
            lines.push(format!("removed  {key}"));
        }
        for row in &changed {
            let same_text = if row.text_equal {
                "; the device instructions are the same (text_sha256 equal)"
            } else {
                ""
            };
            let note = if row.air_only {
                " (the archive moved, the GPU code is equal)"
            } else {
                ""
            };
            lines.push(format!(
                "changed  {}: {}{same_text}{note}",
                row.key,
                row.moved.join("; ")
            ));
        }
        for key in &relinked {
            lines.push(format!(
                "relinked {key}: the device code is the same (exec_sha256 equal); the AIR moved"
            ));
        }
        lines.push(format!(
            "summary: added={} removed={} changed={} same={same} relinked={}",
            added.len(),
            removed.len(),
            changed.len(),
            relinked.len()
        ));
        // Every row moved in its device code and none changed a number or more than an alignment unit of size: the hashes alone moved.
        let hashes_only = !a.rows.is_empty()
            && changed.len() == a.rows.len()
            && changed.iter().all(|row| row.hash_only);
        if hashes_only {
            lines.push(format!(
                "drift: every row's hash moved and none changed a number or more than {DRIFT_ALIGN} bytes of size: the device compiler's session state moved between the scans, not a kernel; scan again"
            ));
            return (lines, 7);
        }
        let rc = match (changed.is_empty(), removed.is_empty()) {
            (true, true) => 0,
            (false, true) => 3,
            (true, false) => 4,
            (false, false) => 5,
        };
        (lines, rc)
    }

    /// The events of a trace and its format's first line: its lines but the
    /// `#` ones.
    fn events(path: &str) -> Result<(&'static str, Vec<String>), String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        let first = lines.next();
        let version = [TRACE_VERSION, TRACE_VERSION_V1]
            .into_iter()
            .find(|v| first == Some(v))
            .ok_or_else(|| format!("{path}: not a {TRACE_VERSION} or {TRACE_VERSION_V1} trace"))?;
        Ok((
            version,
            lines
                .filter(|l| !l.starts_with('#'))
                .map(str::to_string)
                .collect(),
        ))
    }

    /// A buffer as a binding names it: `#n` for the nth buffer the engine
    /// made, `#?n` for a buffer bound that it did not make (numbered by first
    /// sight, in a numbering of its own).
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct BufferId {
        stranger: bool,
        n: u64,
    }

    impl BufferId {
        fn show(self) -> String {
            format!("#{}{}", if self.stranger { "?" } else { "" }, self.n)
        }
    }

    /// A trace with each event's buffer ids renumbered by first use.
    struct Renumbered {
        lines: Vec<String>,
        /// The ids in the order a binding first named them: buffer `k` of the
        /// renumbered lines is `first_use[k]`.
        first_use: Vec<BufferId>,
    }

    fn digits(s: &str) -> bool {
        !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
    }

    /// A binding token `b<slot>=<value>` of a dispatch line. `None` for a
    /// `bytes:` value, whose text is a value and never an id; for a buffer
    /// window `#n+offset/length` the token up to its id (`b0=`), the id and
    /// the window (`+offset/length`). Anything else is an error.
    fn binding(token: &str) -> Result<Option<(&str, BufferId, &str)>, String> {
        let bad = || {
            format!("a binding that is no `bN=#n+offset/length` and no `bN=bytes:...`: {token:?}")
        };
        let (slot, value) = token.split_once('=').ok_or_else(bad)?;
        if !slot.strip_prefix('b').is_some_and(digits) {
            return Err(bad());
        }
        if value.starts_with("bytes:") {
            return Ok(None);
        }
        let id = value.strip_prefix('#').ok_or_else(bad)?;
        let (stranger, id) = id
            .strip_prefix('?')
            .map_or((false, id), |rest| (true, rest));
        let (id, window) = id.split_at(id.find('+').ok_or_else(bad)?);
        let (offset, length) = window[1..].split_once('/').ok_or_else(bad)?;
        // An id that is not canonical decimal would parse equal to another text.
        let canonical = digits(id) && (id == "0" || !id.starts_with('0'));
        if !(canonical && digits(offset) && digits(length)) {
            return Err(bad());
        }
        let n = id.parse().map_err(|_| bad())?;
        Ok(Some((
            &token[..slot.len() + 1],
            BufferId { stranger, n },
            window,
        )))
    }

    /// The events with each side's buffer ids renumbered by first use. Only a
    /// dispatch line carries buffer ids (after its `tpg=` field); every other
    /// field stays as text. An event of a kind this does not know, or a
    /// binding it cannot read, is an error: it might carry an id.
    fn renumber(events: &[String]) -> Result<Renumbered, String> {
        let mut index: HashMap<BufferId, usize> = HashMap::new();
        let mut first_use = Vec::new();
        let mut lines = Vec::with_capacity(events.len());
        for (at, line) in events.iter().enumerate() {
            match line.split(' ').next() {
                Some("cb" | "enc" | "barrier" | "host") => {
                    lines.push(line.clone());
                    continue;
                }
                Some("refused") => {
                    return Err(format!(
                        "event {at}: the trace found a buffer no place in the numbering, so its ids are no names: {line}"
                    ));
                }
                Some("dispatch") => {}
                kind => {
                    return Err(format!(
                        "event {at}: an event of kind {kind:?} that this compare does not know (cb, enc, dispatch, barrier, host): {line}"
                    ));
                }
            }
            let tokens: Vec<&str> = line.split(' ').collect();
            let first = tokens
                .iter()
                .position(|t| t.starts_with("tpg="))
                .map(|p| p + 1)
                .ok_or_else(|| format!("event {at}: a dispatch with no tpg= field: {line}"))?;
            let mut out: Vec<String> = tokens[..first].iter().map(|t| (*t).to_string()).collect();
            for token in &tokens[first..] {
                match binding(token).map_err(|e| format!("event {at}: {e} in {line}"))? {
                    None => out.push((*token).to_string()),
                    Some((head, id, window)) => {
                        let k = *index.entry(id).or_insert_with(|| {
                            first_use.push(id);
                            first_use.len() - 1
                        });
                        let mark = if id.stranger { "?" } else { "" };
                        out.push(format!("{head}#{mark}{k}{window}"));
                    }
                }
            }
            lines.push(out.join(" "));
        }
        Ok(Renumbered { lines, first_use })
    }

    /// A format version's name in a note: the last word of its first line.
    fn version_tag(version: &str) -> &str {
        version.rsplit(' ').next().unwrap_or(version)
    }

    /// The first position where two event lists differ, a list ending early
    /// differing where it ends.
    fn first_difference(a: &[String], b: &[String]) -> Option<usize> {
        match a.iter().zip(b).position(|(x, y)| x != y) {
            None if a.len() == b.len() => None,
            None => Some(a.len().min(b.len())),
            at => at,
        }
    }

    /// The most buffer pairs the renumbered verdict lists.
    const PAIRS_SHOWN: usize = 8;

    /// Two traces: rc 0 the events are equal and the traces number buffers
    /// alike (one format version), 6 equal once each side's buffer ids are
    /// renumbered by first use, 3 they differ (the first differing event, in
    /// that comparison, printed with the events before it), 2 a file that is
    /// no trace or holds a line this cannot read.
    pub(super) fn traces(base: &str, head: &str) -> u8 {
        let ((va, a), (vb, b)) = match (events(base), events(head)) {
            (Ok(a), Ok(b)) => (a, b),
            (a, b) => {
                for e in [a.err(), b.err()].into_iter().flatten() {
                    eprintln!("metal_scan: {e}");
                }
                return 2;
            }
        };
        let dispatches = |v: &[String]| v.iter().filter(|l| l.starts_with("dispatch ")).count();
        let buffers = |v: &[String]| v.iter().filter(|l| l.ends_with(" begin")).count();
        println!(
            "trace compare: base {} events ({} dispatches, {} command buffers), head {} events ({} dispatches, {} command buffers)",
            a.len(),
            dispatches(&a),
            buffers(&a),
            b.len(),
            dispatches(&b),
            buffers(&b)
        );
        // Equal lines of two formats are not one program: the weights' ids
        // follow thread scheduling in one and job order in the other.
        let alike = va == vb;
        if !alike {
            println!(
                "note: base is a {} trace and head a {} trace: they number buffers differently, so the events are compared renumbered and are never identical",
                version_tag(va),
                version_tag(vb)
            );
        }
        // Every line is read before any verdict, so two equal traces with a
        // line this cannot read are refused, not identical.
        let (ra, rb) = match (renumber(&a), renumber(&b)) {
            (Ok(ra), Ok(rb)) => (ra, rb),
            (ra, rb) => {
                for (path, e) in [(base, ra.err()), (head, rb.err())] {
                    if let Some(e) = e {
                        eprintln!("metal_scan: {path}: {e}");
                    }
                }
                return 2;
            }
        };
        let raw_at = first_difference(&a, &b);
        if raw_at.is_none() && alike {
            println!("identical");
            return 0;
        }
        let Some(at) = first_difference(&ra.lines, &rb.lines) else {
            let moved: Vec<String> = ra
                .first_use
                .iter()
                .zip(&rb.first_use)
                .filter(|(x, y)| x != y)
                .map(|(x, y)| format!("base {} -> head {}", x.show(), y.show()))
                .collect();
            let more = if moved.len() > PAIRS_SHOWN {
                ", ..."
            } else {
                ""
            };
            let pairs = if moved.is_empty() {
                "the lines are equal as written".to_string()
            } else {
                format!("{}{more}", moved[..moved.len().min(PAIRS_SHOWN)].join(", "))
            };
            println!(
                "identical up to buffer numbering: {} of {} buffers renumbered ({pairs})",
                moved.len(),
                ra.first_use.len(),
            );
            return 6;
        };
        println!("first difference at event {at}:");
        if let Some(raw_at) = raw_at.filter(|&raw_at| raw_at != at) {
            println!(
                "  (the raw lines first differ at event {raw_at}; buffer ids are compared renumbered by first use)"
            );
        }
        for line in &a[at.saturating_sub(3)..at] {
            println!("   {line}");
        }
        println!(
            "  base: {}",
            a.get(at).map_or("(the trace ends)", String::as_str)
        );
        println!(
            "  head: {}",
            b.get(at).map_or("(the trace ends)", String::as_str)
        );
        if let (Some(x), Some(y)) = (a.get(at), b.get(at)) {
            println!("  differing fields:");
            let fields = |raw: &str, renumbered: &str| -> Vec<(String, String)> {
                raw.split(' ')
                    .zip(renumbered.split(' '))
                    .map(|(r, n)| (r.to_string(), n.to_string()))
                    .collect()
            };
            let (xs, ys) = (fields(x, &ra.lines[at]), fields(y, &rb.lines[at]));
            // A buffer id whose text is the same on both sides can still be a different buffer: the renumbered field is shown beside it.
            let show = |f: Option<&(String, String)>| {
                f.map_or_else(
                    || "-".to_string(),
                    |(raw, n)| {
                        if raw == n {
                            raw.clone()
                        } else {
                            format!("{raw} (renumbered {n})")
                        }
                    },
                )
            };
            for i in 0..xs.len().max(ys.len()) {
                let (p, q) = (xs.get(i), ys.get(i));
                if p.map(|f| &f.1) != q.map(|f| &f.1) {
                    println!("    base {} | head {}", show(p), show(q));
                }
            }
        }
        3
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A scan of `version` with `columns` whose header holds `headers` as (key, value).
        fn scan_text_in(
            version: &str,
            columns: &[&str],
            headers: &[(&str, &str)],
            rows: &[&str],
        ) -> String {
            let mut text = format!("{version}\n");
            for (k, v) in headers {
                text.push_str(&format!("# {k}\t{v}\n"));
            }
            text.push_str(&columns.join("\t"));
            text.push('\n');
            for r in rows {
                text.push_str(r);
                text.push('\n');
            }
            text
        }

        /// A scan of the format before `exec_sha256` (rows of 11 cells).
        fn scan_text_of(headers: &[(&str, &str)], rows: &[&str]) -> String {
            scan_text_in(SCAN_VERSION_V1, &COLUMNS[..EXEC], headers, rows)
        }

        fn scan_text(rows: &[&str]) -> String {
            let headers: Vec<(&str, &str)> = ENVIRONMENT.iter().map(|k| (*k, "v")).collect();
            scan_text_of(&headers, rows)
        }

        /// A scan of the format before `text_sha256` (rows of 12 cells).
        fn scan_text_v2(rows: &[&str]) -> String {
            let headers: Vec<(&str, &str)> = ENVIRONMENT.iter().map(|k| (*k, "v")).collect();
            scan_text_in(SCAN_VERSION_V2, &COLUMNS[..TEXT], &headers, rows)
        }

        /// A scan of the current format (rows of 13 cells).
        fn scan_text_v3(rows: &[&str]) -> String {
            let headers: Vec<(&str, &str)> = ENVIRONMENT.iter().map(|k| (*k, "v")).collect();
            scan_text_in(SCAN_VERSION, &COLUMNS, &headers, rows)
        }

        fn write(name: &str, text: &str) -> String {
            // Each call its own file: the tests run on several threads.
            static CALLS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
            let n = CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("metal_scan_test-{}-{n}-{name}", std::process::id()));
            std::fs::write(&path, text).expect("test file");
            path.to_string_lossy().into_owned()
        }

        /// The exit code of comparing two scan texts.
        fn rc_of(a: &str, b: &str) -> u8 {
            let (x, y) = (write("a", a), write("b", b));
            let code = pipelines(&x, &y);
            let _ = std::fs::remove_file(&x);
            let _ = std::fs::remove_file(&y);
            code
        }

        fn rc(a: &[&str], b: &[&str]) -> u8 {
            rc_of(&scan_text(a), &scan_text(b))
        }

        const P: &str = "main\tp\t-\ty\taa\tdd\t10\t0\t1024\t32\t0";
        const P_MOVED: &str = "main\tp\t-\ty\tbb\tee\t10\t0\t1024\t32\t0";
        const Q: &str = "main\tq\t-\ty\tcc\tff\t10\t0\t1024\t32\t0";

        #[test]
        fn a_uniform_hash_move_is_drift_not_a_kernel() {
            // Every row's hash moved, nothing else: the compiler's state, rc 7.
            let q_moved = "main\tq\t-\ty\tgg\thh\t10\t0\t1024\t32\t0";
            assert_eq!(rc(&[P, Q], &[P_MOVED, q_moved]), 7);
            // One row of two moved: a kernel.
            assert_eq!(rc(&[P, Q], &[P_MOVED, Q]), 3);
            // An alignment unit of size beside the hashes is still drift; a bigger archive is a kernel.
            let p_aligned = "main\tp\t-\ty\tbb\tee\t26\t0\t1024\t32\t0";
            assert_eq!(rc(&[P, Q], &[p_aligned, q_moved]), 7);
            let p_bigger = "main\tp\t-\ty\tbb\tee\t110\t0\t1024\t32\t0";
            assert_eq!(rc(&[P, Q], &[p_bigger, q_moved]), 3);
            // A single row that moved is every row of a scan of one.
            assert_eq!(rc(&[P], &[P_MOVED]), 7);
        }

        #[test]
        fn the_exit_codes_are_ptx_compares() {
            assert_eq!(rc(&[P], &[P]), 0);
            assert_eq!(
                rc(&[P], &[P, Q]),
                0,
                "a row only added moves nothing existing"
            );
            assert_eq!(rc(&[P, Q], &[P]), 4);
            assert_eq!(rc(&[P, Q], &[P_MOVED]), 5);
        }

        #[test]
        fn a_scan_missing_an_environment_value_is_refused() {
            // Two scans with the same key absent compare equal on it, so each is refused alone.
            for key in ENVIRONMENT {
                let without: Vec<(&str, &str)> = ENVIRONMENT
                    .iter()
                    .filter(|k| **k != key)
                    .map(|k| (*k, "v"))
                    .collect();
                let empty: Vec<(&str, &str)> = ENVIRONMENT
                    .iter()
                    .map(|k| (*k, if *k == key { "" } else { "v" }))
                    .collect();
                for headers in [&without, &empty] {
                    let text = scan_text_of(headers, &[P]);
                    assert_eq!(rc_of(&text, &text), 2, "{key}");
                    let path = write("read", &text);
                    let err = read(&path)
                        .err()
                        .expect("a scan without the value is refused");
                    let _ = std::fs::remove_file(&path);
                    assert!(
                        err.contains(&format!("missing environment header {key}")),
                        "{err}"
                    );
                }
            }
        }

        #[test]
        fn a_row_with_a_bad_archive_size_is_refused() {
            // One row, the hashes moved, every other column equal: bad sizes must not compare as one default.
            let row = |sha: &str, code: &str, size: &str| {
                format!("main\tp\t-\ty\t{sha}\t{code}\t{size}\t0\t1024\t32\t0")
            };
            for (x, y) in [("x", "y"), ("-1", "-1"), ("", "1.5")] {
                let (a, b) = (row("aa", "dd", x), row("bb", "ee", y));
                assert_eq!(rc(&[&a], &[&b]), 2, "{x:?} against {y:?}");
                let path = write("read", &scan_text(&[&a]));
                let err = read(&path).err().expect("a bad archive_bytes is refused");
                let _ = std::fs::remove_file(&path);
                assert!(err.contains("invalid archive_bytes"), "{err}");
            }
        }

        /// The table and exit code of comparing two scan texts.
        fn table_of(a: &str, b: &str) -> (String, u8) {
            let (x, y) = (write("a", a), write("b", b));
            let scans = (read(&x), read(&y));
            let _ = std::fs::remove_file(&x);
            let _ = std::fs::remove_file(&y);
            let (a, b) = (scans.0.expect("base scan"), scans.1.expect("head scan"));
            let (lines, rc) = table(&a, &b);
            (lines.join("\n"), rc)
        }

        /// Rows of the format before `text_sha256`: `p` and `q` with their `exec_sha256` last.
        const P2: &str = "main\tp\t-\ty\taa\tdd\t10\t0\t1024\t32\t0\tx1";
        const Q2: &str = "main\tq\t-\ty\tcc\tff\t10\t0\t1024\t32\t0\ty1";
        /// `p` with its archive moved (hashes, an alignment unit of size) and its device code the same.
        const P2_AIR: &str = "main\tp\t-\ty\tbb\tee\t26\t0\t1024\t32\t0\tx1";
        /// `p` with its device code moved.
        const P2_EXEC: &str = "main\tp\t-\ty\tbb\tee\t10\t0\t1024\t32\t0\tx2";
        /// Rows of the current format: those above with their `text_sha256` last.
        const P3: &str = "main\tp\t-\ty\taa\tdd\t10\t0\t1024\t32\t0\tx1\tt1";
        const Q3: &str = "main\tq\t-\ty\tcc\tff\t10\t0\t1024\t32\t0\ty1\tu1";
        /// `p` with its archive moved and its device code the same.
        const P3_AIR: &str = "main\tp\t-\ty\tbb\tee\t26\t0\t1024\t32\t0\tx1\tt1";
        /// `p` with its device code moved and its instructions the same.
        const P3_NOTES: &str = "main\tp\t-\ty\tbb\tee\t10\t0\t1024\t32\t0\tx2\tt1";
        /// `p` with its device code and its instructions moved.
        const P3_TEXT: &str = "main\tp\t-\ty\tbb\tee\t10\t0\t1024\t32\t0\tx2\tt2";

        #[test]
        fn the_column_indexes_name_their_columns() {
            for (at, name) in [
                (LIVE, "live"),
                (SHA, "sha256"),
                (CODE, "code_sha256"),
                (SIZE, "archive_bytes"),
                (STRIPPED, "stripped_bytes"),
                (EXEC, "exec_sha256"),
                (TEXT, "text_sha256"),
            ] {
                assert_eq!(COLUMNS[at], name);
            }
            assert_eq!(COLUMNS.len(), TEXT + 1, "text_sha256 is the last column");
            assert_eq!(TEXT, EXEC + 1, "text_sha256 follows exec_sha256");
        }

        #[test]
        fn a_row_whose_archive_moved_and_device_code_did_not_is_relinked() {
            let (table, rc) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[P2_AIR, Q2]));
            assert_eq!(rc, 0, "{table}");
            assert!(
                table.contains(
                    "relinked main p -: the device code is the same (exec_sha256 equal); the AIR moved"
                ),
                "{table}"
            );
            assert!(!table.contains("changed  "), "{table}");
            assert!(table.contains("changed=0 same=1 relinked=1"), "{table}");
            // Every row relinked is no drift (rc 7 is the hashes of changed device code moving alone).
            assert_eq!(rc_of(&scan_text_v2(&[P2]), &scan_text_v2(&[P2_AIR])), 0);
            // A number the device compiler fixed, or the live flag, moving beside the archive is a change.
            let threads = "main\tp\t-\ty\tbb\tee\t26\t0\t512\t32\t0\tx1";
            let (table, rc) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[threads, Q2]));
            assert_eq!(rc, 3, "{table}");
            assert!(table.contains("changed  main p -: "), "{table}");
            let dead = "main\tp\t-\tn\tbb\tee\t26\t0\t1024\t32\t0\tx1";
            assert_eq!(
                rc_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[dead, Q2])),
                3
            );
        }

        #[test]
        fn a_row_whose_device_code_moved_is_changed() {
            let (table, rc) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[P2_EXEC, Q2]));
            assert_eq!(rc, 3, "{table}");
            assert!(
                table.contains(
                    "changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee; exec_sha256 x1 -> x2"
                ),
                "{table}"
            );
            assert!(!table.contains("relinked "), "{table}");
            // The drift rule judges the rows whose device code moved: all of them, by hash alone, is drift.
            assert_eq!(rc_of(&scan_text_v2(&[P2]), &scan_text_v2(&[P2_EXEC])), 7);
            let q2_exec = "main\tq\t-\ty\tgg\thh\t10\t0\t1024\t32\t0\ty2";
            assert_eq!(
                rc_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[P2_EXEC, q2_exec])),
                7
            );
            // One row relinked and one changed is a kernel, not drift.
            assert_eq!(
                rc_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[P2_AIR, q2_exec])),
                3
            );
        }

        #[test]
        fn a_side_without_exec_sha256_is_judged_on_code_sha256() {
            let note = "changed is judged on code_sha256 because a side has no exec_sha256";
            // The base is a v1 scan: p's archive moved, so it is changed as before, with the note.
            let (table, rc) = table_of(&scan_text(&[P, Q]), &scan_text_v2(&[P2_AIR, Q2]));
            assert_eq!(rc, 3, "{table}");
            assert!(table.contains(note), "{table}");
            assert!(
                table.contains("(base has it: no, head has it: yes)"),
                "{table}"
            );
            assert!(
                table.contains("changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee"),
                "{table}"
            );
            assert!(!table.contains("relinked "), "{table}");
            // The head a v1 scan: the same.
            let (table, rc) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text(&[P_MOVED, Q]));
            assert_eq!(rc, 3, "{table}");
            assert!(table.contains(note), "{table}");
            // Two scans with the column say nothing of it.
            let (table, _) = table_of(&scan_text_v3(&[P3]), &scan_text_v3(&[P3]));
            assert!(!table.contains("note:"), "{table}");
        }

        #[test]
        fn a_scan_that_misstates_its_format_is_refused() {
            let headers: Vec<(&str, &str)> = ENVIRONMENT.iter().map(|k| (*k, "v")).collect();
            let refused = |text: &str| {
                let path = write("read", text);
                let err = read(&path).err().expect("the scan is refused");
                let _ = std::fs::remove_file(&path);
                err
            };
            // Each version with the columns of another.
            for (version, columns) in [
                (SCAN_VERSION, &COLUMNS[..EXEC]),
                (SCAN_VERSION, &COLUMNS[..TEXT]),
                (SCAN_VERSION_V2, &COLUMNS[..]),
                (SCAN_VERSION_V2, &COLUMNS[..EXEC]),
                (SCAN_VERSION_V1, &COLUMNS[..]),
                (SCAN_VERSION_V1, &COLUMNS[..TEXT]),
            ] {
                let err = refused(&scan_text_in(version, columns, &headers, &[P]));
                assert!(err.contains("the column line is not"), "{err}");
            }
            let err = refused(&scan_text_in("# metal_scan v4", &COLUMNS, &headers, &[P3]));
            assert!(
                err.contains("not a # metal_scan v3, # metal_scan v2 or # metal_scan v1 scan"),
                "{err}"
            );
            // A hash cell that is empty is no hash.
            let empty = "main\tp\t-\ty\taa\tdd\t10\t0\t1024\t32\t0\t";
            let err = refused(&scan_text_v2(&[empty]));
            assert!(err.contains("empty exec_sha256"), "{err}");
            let err = refused(&scan_text_v3(&[&format!("{P2}\t")]));
            assert!(err.contains("empty text_sha256"), "{err}");
        }

        /// A scan row of a pipeline whose archive is `slice`, as the scan writes one.
        #[cfg(target_os = "macos")]
        fn row_of(slice: &[u8]) -> String {
            use gguf::digest::sha256_hex;
            let exec = crate::macho::exec(slice).expect("a slice the scan reads");
            format!(
                "main\tp\t-\ty\t{}\t{}\t{}\t0\t1024\t32\t0\t{}",
                sha256_hex(slice),
                sha256_hex(slice),
                slice.len(),
                exec.sha256
            )
        }

        #[cfg(target_os = "macos")]
        #[test]
        fn a_move_of_only_the_air_section_is_relinked_and_one_of_the_code_is_not() {
            use crate::macho::built;
            let slice = |air: &'static [u8], code: &'static [u8]| {
                built(
                    1,
                    &[
                        ("__reflection", b"reflection"),
                        ("__compute", code),
                        ("__metallib", air),
                    ],
                )
            };
            let base = scan_text_v2(&[&row_of(&slice(b"air", b"code"))]);
            let moved_air = scan_text_v2(&[&row_of(&slice(b"another air", b"code"))]);
            let (table, rc) = table_of(&base, &moved_air);
            assert_eq!(rc, 0, "{table}");
            assert!(table.contains("relinked main p -: "), "{table}");
            let moved_code = scan_text_v2(&[&row_of(&slice(b"air", b"cod3"))]);
            let (table, rc) = table_of(&base, &moved_code);
            assert_eq!(rc, 7, "{table}");
            assert!(table.contains("changed  main p -: "), "{table}");
        }

        /// The first line of `table` that starts a `changed` row.
        fn changed_line(table: &str) -> &str {
            table
                .lines()
                .find(|l| l.starts_with("changed  "))
                .unwrap_or_else(|| panic!("no changed row: {table}"))
        }

        #[test]
        fn a_changed_row_with_equal_instructions_says_so_and_stays_changed() {
            let (table, rc) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v3(&[P3_NOTES, Q3]));
            assert_eq!(rc, 3, "{table}");
            assert_eq!(
                changed_line(&table),
                "changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee; exec_sha256 x1 -> x2; the device instructions are the same (text_sha256 equal)"
            );
            assert!(table.contains("changed=1 same=1 relinked=0"), "{table}");
            assert!(!table.contains("note:"), "{table}");
            // Equal instructions beside a number the device compiler fixed: still a change.
            let threads = "main\tp\t-\ty\taa\tdd\t10\t0\t512\t32\t0\tx1\tt1";
            let (table, rc) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v3(&[threads, Q3]));
            assert_eq!(rc, 3, "{table}");
            assert!(
                changed_line(&table).contains("text_sha256 equal"),
                "{table}"
            );
        }

        #[test]
        fn a_changed_row_with_other_instructions_lists_the_column() {
            let (table, rc) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v3(&[P3_TEXT, Q3]));
            assert_eq!(rc, 3, "{table}");
            assert_eq!(
                changed_line(&table),
                "changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee; exec_sha256 x1 -> x2; text_sha256 t1 -> t2"
            );
            assert!(!table.contains("instructions are the same"), "{table}");
            assert!(!table.contains("note:"), "{table}");
        }

        #[test]
        fn the_column_moves_no_exit_code_and_no_count() {
            // The same rows with and without the column: one rc and one summary line.
            let summary = |table: &str| {
                table
                    .lines()
                    .find(|l| l.starts_with("summary:"))
                    .expect("a summary line")
                    .to_string()
            };
            for (v2, v3) in [
                ([P2, Q2], [P3, Q3]),
                ([P2_EXEC, Q2], [P3_NOTES, Q3]),
                ([P2_EXEC, Q2], [P3_TEXT, Q3]),
                ([P2_AIR, Q2], [P3_AIR, Q3]),
            ] {
                let (t2, rc2) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&v2));
                let (t3, rc3) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v3(&v3));
                assert_eq!((rc2, summary(&t2)), (rc3, summary(&t3)), "{t2}\n{t3}");
            }
            // A relinked row stays relinked, with no word of instructions.
            let (table, rc) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v3(&[P3_AIR, Q3]));
            assert_eq!(rc, 0, "{table}");
            assert!(table.contains("relinked main p -: "), "{table}");
            assert!(!table.contains("instructions are the same"), "{table}");
            // Every row's hash moving alone is drift, whatever the instructions did.
            assert_eq!(rc_of(&scan_text_v3(&[P3]), &scan_text_v3(&[P3_NOTES])), 7);
            assert_eq!(rc_of(&scan_text_v3(&[P3]), &scan_text_v3(&[P3_TEXT])), 7);
        }

        #[test]
        fn a_side_without_text_sha256_says_nothing_on_the_rows() {
            let note =
                "note: the device instructions are not compared because a side has no text_sha256";
            let rows =
                "changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee; exec_sha256 x1 -> x2";
            // A v2 base, and a v2 head.
            let (table, rc) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v3(&[P3_NOTES, Q3]));
            assert_eq!(rc, 3, "{table}");
            assert!(table.contains(note), "{table}");
            assert!(
                table.contains("(base has it: no, head has it: yes)"),
                "{table}"
            );
            assert_eq!(changed_line(&table), rows);
            let (table, rc) = table_of(&scan_text_v3(&[P3, Q3]), &scan_text_v2(&[P2_EXEC, Q2]));
            assert_eq!(rc, 3, "{table}");
            assert!(
                table.contains("(base has it: yes, head has it: no)"),
                "{table}"
            );
            assert_eq!(changed_line(&table), rows);
            // A v1 base has neither column: both notes, and the row is judged on code_sha256.
            let (table, rc) = table_of(&scan_text(&[P, Q]), &scan_text_v3(&[P3_NOTES, Q3]));
            assert_eq!(rc, 3, "{table}");
            assert!(table.contains(note), "{table}");
            assert!(
                table.contains("because a side has no exec_sha256"),
                "{table}"
            );
            assert_eq!(
                changed_line(&table),
                "changed  main p -: sha256 aa -> bb; code_sha256 dd -> ee"
            );
            // Two v2 scans: the same note, and no suffix.
            let (table, _) = table_of(&scan_text_v2(&[P2, Q2]), &scan_text_v2(&[P2_EXEC, Q2]));
            assert!(
                table.contains("(base has it: no, head has it: no)"),
                "{table}"
            );
            assert_eq!(changed_line(&table), rows);
        }

        #[test]
        fn scans_of_two_formats_are_compared_though_their_archive_headers_differ() {
            let scan = |version: &str, columns: &[&str], key: &str, value: &str, rows: &[&str]| {
                let headers: Vec<(&str, &str)> = ENVIRONMENT
                    .iter()
                    .map(|k| (*k, if *k == key { value } else { "v" }))
                    .collect();
                scan_text_in(version, columns, &headers, rows)
            };
            let v2 = |archive: &str| {
                scan(
                    SCAN_VERSION_V2,
                    &COLUMNS[..TEXT],
                    "archive",
                    archive,
                    &[P2, Q2],
                )
            };
            let v3 =
                |key: &str, value: &str| scan(SCAN_VERSION, &COLUMNS, key, value, &[P3_NOTES, Q3]);
            // The columns' own documentation differs between formats: compared all the same.
            assert_eq!(
                rc_of(&v2("two columns documented less"), &v3("archive", "more")),
                3
            );
            // The same format with another archive header, and any other header across formats, is refused.
            assert_eq!(rc_of(&v3("archive", "more"), &v3("archive", "another")), 2);
            assert_eq!(rc_of(&v2("a"), &v3("os", "another")), 2);
        }

        /// A scan row of a pipeline whose archive is `slice`, with the text column.
        #[cfg(target_os = "macos")]
        fn row_of_v3(slice: &[u8]) -> String {
            let text = crate::macho::text(slice).expect("a slice the scan reads");
            format!("{}\t{text}", row_of(slice))
        }

        #[cfg(target_os = "macos")]
        #[test]
        fn a_move_of_only_the_compilers_notes_leaves_the_instructions_the_same() {
            use crate::macho::{built, built_in};
            let scan = |notes: &[u8], code: &[u8]| {
                let device = built_in(
                    0,
                    &[
                        ("__TEXT", "__text", code),
                        ("__GPU_LD_MD", "__compute", notes),
                    ],
                );
                let slice = built(
                    1,
                    &[
                        ("__reflection", b"reflection"),
                        ("__compute", &device),
                        ("__metallib", b"air"),
                    ],
                );
                scan_text_v3(&[&row_of_v3(&slice), Q3])
            };
            let base = scan(b"notes", b"code");
            let (table, rc) = table_of(&base, &scan(b"other notes", b"code"));
            assert_eq!(rc, 3, "{table}");
            assert!(
                changed_line(&table)
                    .ends_with("; the device instructions are the same (text_sha256 equal)"),
                "{table}"
            );
            let (table, rc) = table_of(&base, &scan(b"notes", b"cod3"));
            assert_eq!(rc, 3, "{table}");
            assert!(changed_line(&table).contains("; text_sha256 "), "{table}");
            assert!(!table.contains("instructions are the same"), "{table}");
        }

        /// A program's events: one dispatch for each of `binds` between a
        /// command buffer's begin and commit.
        fn program(binds: &[&str]) -> Vec<String> {
            let mut events: Vec<String> = ["cb 0 begin", "enc 0.0 open concurrent"]
                .map(String::from)
                .into();
            for (k, bind) in binds.iter().enumerate() {
                events.push(format!(
                    "dispatch 0.0.{k} q4k_matvec max=512 width=32 tg_static=0 grid=4x1 tpg=64 {bind}"
                ));
                events.push("barrier 0.0".to_string());
            }
            events.extend(["enc 0.0 end", "cb 0 commit", "host sync"].map(String::from));
            events
        }

        /// The exit code of comparing two traces made of `a` and `b`, of the
        /// format versions `versions` (base, head).
        fn trace_rc_as(versions: [&str; 2], a: &[String], b: &[String]) -> u8 {
            let text =
                |version: &str, events: &[String]| format!("{version}\n{}\n", events.join("\n"));
            let (x, y) = (
                write("a", &text(versions[0], a)),
                write("b", &text(versions[1], b)),
            );
            let code = traces(&x, &y);
            let _ = std::fs::remove_file(&x);
            let _ = std::fs::remove_file(&y);
            code
        }

        /// The exit code of comparing two traces of the current format.
        fn trace_rc(a: &[String], b: &[String]) -> u8 {
            trace_rc_as([TRACE_VERSION; 2], a, b)
        }

        #[test]
        fn equal_traces_are_identical() {
            let t = program(&[
                "b0=#1+0/64 b1=#2+0/64 b2=bytes:0800",
                "b0=#2+0/64 b1=#1+0/64",
            ]);
            assert_eq!(trace_rc(&t, &t), 0);
        }

        /// Equal lines under two format versions are not one program: v1 numbered
        /// the weights' buffers in the order threads made them. They are compared
        /// renumbered (6 when that is equal, 3 when not), never 0, whichever side
        /// is the older. Mutant: the version ignored (0).
        #[test]
        fn equal_lines_of_two_format_versions_are_never_identical() {
            let t = program(&["b0=#1+0/64 b1=#2+0/64", "b0=#2+0/64 b1=#1+0/64"]);
            for versions in [
                [TRACE_VERSION_V1, TRACE_VERSION],
                [TRACE_VERSION, TRACE_VERSION_V1],
            ] {
                assert_eq!(trace_rc_as(versions, &t, &t), 6, "{versions:?}");
                // Two traces of the old format are compared as they were.
                let shifted = program(&["b0=#2+0/64 b1=#3+0/64", "b0=#3+0/64 b1=#2+0/64"]);
                assert_eq!(trace_rc_as(versions, &t, &shifted), 6, "{versions:?}");
                let other = program(&["b0=#1+0/64 b1=#2+0/64", "b0=#1+0/64 b1=#2+0/64"]);
                assert_eq!(trace_rc_as(versions, &t, &other), 3, "{versions:?}");
            }
            assert_eq!(trace_rc_as([TRACE_VERSION_V1; 2], &t, &t), 0);
            // A format this does not know is refused, not read as the nearest one.
            let later = "# bloomery encode trace v3";
            assert_eq!(trace_rc_as([later, TRACE_VERSION], &t, &t), 2);
            assert_eq!(trace_rc_as([TRACE_VERSION, later], &t, &t), 2);
        }

        /// A trace that says a buffer had no place in the numbering is refused
        /// by name: its ids cannot be read, equal or not.
        #[test]
        fn a_trace_that_refused_a_buffer_is_refused() {
            let mut t = program(&["b0=#1+0/64"]);
            t.insert(0, "refused a buffer made by no job: 1".to_string());
            assert_eq!(trace_rc(&t, &t), 2);
            assert!(
                renumber(&t).err().is_some_and(
                    |e| e.contains("no place in the numbering") && e.contains("refused")
                )
            );
        }

        /// The format version the engine writes is the one this reads.
        #[cfg(all(target_os = "macos", feature = "encode-trace"))]
        #[test]
        fn the_engines_trace_version_is_the_one_read() {
            assert_eq!(bloomery_metal::trace::VERSION, TRACE_VERSION);
        }

        #[test]
        fn a_trace_with_its_ids_shifted_is_the_same_program() {
            let base = program(&["b0=#1+0/64 b1=#2+0/64", "b0=#2+0/64 b1=#3+0/64"]);
            // Every id one higher.
            let head = program(&["b0=#2+0/64 b1=#3+0/64", "b0=#3+0/64 b1=#4+0/64"]);
            assert_eq!(trace_rc(&base, &head), 6);
            // The shift begins after the first dispatch, as one more buffer made early does
            // (a trace numbers buffers in the order one thread would make them).
            let head = program(&["b0=#1+0/64 b1=#3+0/64", "b0=#3+0/64 b1=#4+0/64"]);
            assert_eq!(trace_rc(&base, &head), 6);
            let (a, b) = (renumber(&base).unwrap(), renumber(&head).unwrap());
            assert_eq!(a.lines, b.lines);
            let ids = |r: &Renumbered| r.first_use.iter().map(|i| i.n).collect::<Vec<_>>();
            assert_eq!((ids(&a), ids(&b)), (vec![1, 2, 3], vec![1, 3, 4]));
        }

        #[test]
        fn two_buffers_swapped_in_one_dispatch_are_a_difference() {
            let base = program(&["b0=#1+0/64 b1=#2+0/64", "b0=#1+0/64 b1=#2+0/64"]);
            let head = program(&["b0=#1+0/64 b1=#2+0/64", "b0=#2+0/64 b1=#1+0/64"]);
            assert_eq!(trace_rc(&base, &head), 3);
        }

        #[test]
        fn a_numbering_that_is_no_bijection_is_a_difference() {
            // One base buffer where the head binds two, and two where the head binds one.
            let one = program(&["b0=#5+0/64", "b0=#5+0/64"]);
            let two = program(&["b0=#5+0/64", "b0=#6+0/64"]);
            assert_eq!(trace_rc(&one, &two), 3);
            assert_eq!(trace_rc(&two, &one), 3);
        }

        #[test]
        fn a_field_that_is_no_id_is_compared_as_text() {
            let base = program(&["b0=#1+0/64 b1=bytes:0800"]);
            let changes = [
                ("#1+0/64", "#1+8/64"),
                ("#1+0/64", "#1+0/65"),
                ("q4k_matvec", "q4k_matvec<2>"),
                ("grid=4x1", "grid=5x1"),
                ("tpg=64", "tpg=128"),
                ("tg_static=0", "tg_static=16"),
                ("bytes:0800", "bytes:0801"),
            ];
            for (from, to) in changes {
                let moved: Vec<String> = base.iter().map(|l| l.replace(from, to)).collect();
                assert_ne!(moved, base, "{from}");
                assert_eq!(trace_rc(&base, &moved), 3, "{from} -> {to}");
                // The numbering moved too: the difference stays.
                let shifted: Vec<String> = moved.iter().map(|l| l.replace("#1+", "#2+")).collect();
                assert_eq!(trace_rc(&base, &shifted), 3, "{from} -> {to}, shifted");
            }
        }

        #[test]
        fn a_bytes_payload_is_never_renumbered() {
            let base = program(&["b0=#1+0/64 b1=bytes:#1#2"]);
            let head = program(&["b0=#7+0/64 b1=bytes:#1#2"]);
            assert_eq!(trace_rc(&base, &head), 6);
            let renumbered = renumber(&head).unwrap();
            assert!(
                renumbered
                    .lines
                    .iter()
                    .any(|l| l.ends_with("b0=#0+0/64 b1=bytes:#1#2"))
            );
            assert_eq!(renumbered.first_use.len(), 1);
            // The same numbering and a payload that differs only in its `#` text is a difference.
            let other = program(&["b0=#1+0/64 b1=bytes:#2#1"]);
            assert_eq!(trace_rc(&base, &other), 3);
        }

        #[test]
        fn a_buffer_the_engine_did_not_make_has_a_numbering_of_its_own() {
            let base = program(&["b0=#?0+0/64 b1=#3+0/64", "b0=#?1+0/64"]);
            let head = program(&["b0=#?1+0/64 b1=#4+0/64", "b0=#?2+0/64"]);
            assert_eq!(trace_rc(&base, &head), 6);
            let made = program(&["b0=#0+0/64 b1=#3+0/64", "b0=#1+0/64"]);
            assert_eq!(trace_rc(&base, &made), 3);
        }

        #[test]
        fn a_trace_that_ends_early_is_a_difference() {
            let base = program(&["b0=#1+0/64", "b0=#2+0/64"]);
            // The renumbered lines of the head are a prefix of the base's, the head stopping after its second dispatch's barrier.
            let mut head = program(&["b0=#2+0/64", "b0=#3+0/64"]);
            head.truncate(base.len() - 3);
            assert_eq!(head.len(), 6);
            assert_eq!(trace_rc(&base, &head), 3);
            assert_eq!(trace_rc(&head, &base), 3);
        }

        #[test]
        fn a_line_it_cannot_read_is_refused_by_name() {
            let base = program(&["b0=#1+0/64"]);
            let unreadable = [
                "b0=#x+0/64",
                "b0=#1+0",
                "b0=#1/64",
                "b0=#01+0/64",
                "b0=1+0/64",
                "b0=junk",
                "b0",
                "x0=#1+0/64",
                "#1+0/64",
            ];
            for bind in unreadable {
                let head = program(&[bind]);
                assert_eq!(trace_rc(&base, &head), 2, "{bind}");
                assert_eq!(trace_rc(&head, &head), 2, "{bind}");
                assert!(
                    renumber(&head)
                        .err()
                        .is_some_and(|e| e.contains("a binding")),
                    "{bind}"
                );
            }
            let mut head = base.clone();
            head[2] = head[2].replace(" tpg=64", "");
            assert!(renumber(&head).err().is_some_and(|e| e.contains("no tpg=")));
            let mut head = base.clone();
            head.push("frobnicate 0".to_string());
            assert!(
                renumber(&head)
                    .err()
                    .is_some_and(|e| e.contains("does not know"))
            );
            assert_eq!(trace_rc(&base, &head), 2);
        }
    }
}
