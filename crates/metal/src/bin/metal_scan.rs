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
//! identity bytes stripped, and three numbers the device compiler fixes in the
//! built pipeline (most threads a group, SIMD width, static threadgroup
//! bytes). A pipeline that cannot be built or serialized is a named error and
//! exit 1, and so is a library that is the build's metallib: the calibration
//! compiles the run-time source, so it must be what runs.
//!
//! `--instance NAME` scans one registered instance's kernels (default: the
//! build's host instance); a name that is not registered is exit 64, naming
//! the registered ones.
//!
//! `--keep DIR` keeps each serialized archive as `DIR/<lib>-<entry>-<constants>.bin`, for
//! reading what differs between two runs.
//!
//! `--compare` prints, for two scans, the pipelines added, removed and
//! changed (with the columns that moved) and the count unchanged. Exit 0
//! when nothing existing moved (rows only added are listed), 3 an existing
//! row's hash or number moved, 4 a row was removed, 5 both, 2 two scans of
//! different environments or a file that is no scan, 7 every row of the base
//! moved and none changed a number or more than an alignment unit of size (the
//! device compiler's session state moved between the two scans, not a kernel:
//! scan again), 64 usage.
//! `--compare-trace` compares two encode traces
//! (`generate_metal --encode-trace`) and prints the first differing event:
//! exit 0 identical, 3 they differ, 2 a file that is no trace or two format
//! versions.

use std::process::ExitCode;

const USAGE: &str = "\
metal_scan [--instance NAME] [--out FILE] [--keep DIR]
metal_scan --compare BASE.tsv HEAD.tsv
metal_scan --compare-trace BASE.txt HEAD.txt";

/// The scan's format, the first line of a TSV.
const SCAN_VERSION: &str = "# metal_scan v1";

/// The scan's columns.
const COLUMNS: [&str; 11] = [
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

#[cfg(target_os = "macos")]
mod mac {
    use std::collections::BTreeMap;
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

    use super::SCAN_VERSION;

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
        let mut lc_uuids = Vec::new();
        {
            let slice = &out[code.clone()];
            let word = |at: usize| -> Result<u32, String> {
                slice
                    .get(at..at + 4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .ok_or_else(|| format!("the GPU slice ends inside its header at byte {at}"))
            };
            if word(0)? != 0xfeed_facf {
                return Err(format!(
                    "the GPU slice's magic is {:#x}, not a 64-bit Mach-O's",
                    word(0)?
                ));
            }
            let mut at = 32;
            for _ in 0..word(16)? {
                let (cmd, size) = (word(at)?, word(at + 4)? as usize);
                if size < 8 {
                    return Err(format!("a Mach-O load command of {size} bytes at {at}"));
                }
                if cmd == 0x1b && size >= 24 {
                    lc_uuids.push(code.start + at + 8);
                }
                at += size;
            }
        }
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
            rows.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
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
            "MTLBinaryArchive of one function (descriptor defaults), serializeToURL; the library's identity is zeroed (the 64-hex module signature, the module UUID, the GPU slice's LC_UUID: stripped_bytes) and checked against a build of the library with one unused kernel appended; sha256 of the rest, code_sha256 of its GPU-code slice"
                .to_string(),
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
    use std::collections::BTreeMap;

    use super::{COLUMNS, ENVIRONMENT, SCAN_VERSION};

    /// The trace format's first line (`bloomery_metal::trace::VERSION`, which
    /// exists only in an encode-trace build).
    const TRACE_VERSION: &str = "# bloomery encode trace v1";

    /// The bytes an archive may change size by between two sessions of the
    /// device compiler (one 16-byte alignment unit, twice) and still be the
    /// same function.
    const DRIFT_ALIGN: i64 = 32;

    /// One scan: its header and its rows keyed by (lib, entry, constants).
    struct Scan {
        header: BTreeMap<String, String>,
        rows: BTreeMap<String, Vec<String>>,
    }

    fn read(path: &str) -> Result<Scan, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        if lines.next() != Some(SCAN_VERSION) {
            return Err(format!("{path}: not a {SCAN_VERSION} scan"));
        }
        let mut header = BTreeMap::new();
        let mut rows = BTreeMap::new();
        let mut columns = false;
        for line in lines {
            if let Some(rest) = line.strip_prefix("# ") {
                let (key, value) = rest
                    .split_once('\t')
                    .ok_or_else(|| format!("{path}: header line without a tab: {line}"))?;
                header.insert(key.to_string(), value.to_string());
            } else if !columns {
                if line.split('\t').ne(COLUMNS) {
                    return Err(format!(
                        "{path}: the column line is not {COLUMNS:?}: {line}"
                    ));
                }
                columns = true;
            } else {
                let cells: Vec<String> = line.split('\t').map(str::to_string).collect();
                if cells.len() != COLUMNS.len() {
                    return Err(format!("{path}: a row of {} cells: {line}", cells.len()));
                }
                let key = format!("{} {} {}", cells[0], cells[1], cells[2]);
                if rows.insert(key.clone(), cells).is_some() {
                    return Err(format!("{path}: the row {key} twice"));
                }
            }
        }
        if !columns {
            return Err(format!("{path}: no column line"));
        }
        for key in ENVIRONMENT {
            if header.get(key).is_none_or(String::is_empty) {
                return Err(format!("{path}: missing environment header {key}"));
            }
        }
        for (key, row) in &rows {
            if !row[6].parse::<i64>().is_ok_and(|n| n >= 0) {
                return Err(format!("{path}: {key}: invalid archive_bytes {}", row[6]));
            }
        }
        Ok(Scan { header, rows })
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
        let differing: Vec<String> = ENVIRONMENT
            .iter()
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
        let (mut added, mut removed, mut changed, mut same) = (vec![], vec![], vec![], 0usize);
        for (key, row) in &a.rows {
            match b.rows.get(key) {
                None => removed.push(key.clone()),
                Some(other) => {
                    let moved: Vec<String> = COLUMNS
                        .iter()
                        .zip(row.iter().zip(other))
                        .skip(3)
                        .filter(|(_, (x, y))| x != y)
                        .map(|(c, (x, y))| format!("{c} {x} -> {y}"))
                        .collect();
                    if moved.is_empty() {
                        same += 1;
                    } else {
                        let air_only = row[4] != other[4] && row[5] == other[5];
                        let size = |r: &[String]| {
                            r[6].parse::<i64>().expect("read validated archive_bytes")
                        };
                        let hash_only = row[3] == other[3]
                            && row[7..] == other[7..]
                            && (size(row) - size(other)).abs() <= DRIFT_ALIGN;
                        changed.push((key.clone(), moved, air_only, hash_only));
                    }
                }
            }
        }
        for key in b.rows.keys() {
            if !a.rows.contains_key(key) {
                added.push(key.clone());
            }
        }
        println!(
            "metal-scan compare: base {} rows, head {} rows",
            a.rows.len(),
            b.rows.len()
        );
        let source = |k: &str| a.header.get(k) == b.header.get(k);
        println!(
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
        );
        for key in &added {
            println!("added    {key}");
        }
        for key in &removed {
            println!("removed  {key}");
        }
        for (key, moved, air_only, _) in &changed {
            let note = if *air_only {
                " (the archive moved, the GPU code is equal)"
            } else {
                ""
            };
            println!("changed  {key}: {}{note}", moved.join("; "));
        }
        println!(
            "summary: added={} removed={} changed={} same={same}",
            added.len(),
            removed.len(),
            changed.len()
        );
        // Every row moved and no row changed a number or more than an alignment unit of size: the hashes alone moved.
        let hashes_only = !a.rows.is_empty()
            && changed.len() == a.rows.len()
            && changed.iter().all(|(_, _, _, hash_only)| *hash_only);
        if hashes_only {
            println!(
                "drift: every row's hash moved and none changed a number or more than {DRIFT_ALIGN} bytes of size: the device compiler's session state moved between the scans, not a kernel; scan again"
            );
            return 7;
        }
        match (changed.is_empty(), removed.is_empty()) {
            (true, true) => 0,
            (false, true) => 3,
            (true, false) => 4,
            (false, false) => 5,
        }
    }

    /// The events of a trace: its lines but the `#` ones.
    fn events(path: &str) -> Result<Vec<String>, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut lines = text.lines();
        if lines.next() != Some(TRACE_VERSION) {
            return Err(format!("{path}: not a {TRACE_VERSION} trace"));
        }
        Ok(lines
            .filter(|l| !l.starts_with('#'))
            .map(str::to_string)
            .collect())
    }

    /// The first differing event of two traces, with the events before it.
    pub(super) fn traces(base: &str, head: &str) -> u8 {
        let (a, b) = match (events(base), events(head)) {
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
        let at = a.iter().zip(&b).position(|(x, y)| x != y);
        let at = match (at, a.len() == b.len()) {
            (Some(at), _) => at,
            (None, true) => {
                println!("identical");
                return 0;
            }
            (None, false) => a.len().min(b.len()),
        };
        println!("first difference at event {at}:");
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
            let (xs, ys): (Vec<_>, Vec<_>) = (x.split(' ').collect(), y.split(' ').collect());
            for i in 0..xs.len().max(ys.len()) {
                let (p, q) = (xs.get(i), ys.get(i));
                if p != q {
                    println!(
                        "    base {} | head {}",
                        p.unwrap_or(&"-"),
                        q.unwrap_or(&"-")
                    );
                }
            }
        }
        3
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A scan whose header holds `headers` as (key, value).
        fn scan_text_of(headers: &[(&str, &str)], rows: &[&str]) -> String {
            let mut text = format!("{SCAN_VERSION}\n");
            for (k, v) in headers {
                text.push_str(&format!("# {k}\t{v}\n"));
            }
            text.push_str(&COLUMNS.join("\t"));
            text.push('\n');
            for r in rows {
                text.push_str(r);
                text.push('\n');
            }
            text
        }

        fn scan_text(rows: &[&str]) -> String {
            let headers: Vec<(&str, &str)> = ENVIRONMENT.iter().map(|k| (*k, "v")).collect();
            scan_text_of(&headers, rows)
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
    }
}
