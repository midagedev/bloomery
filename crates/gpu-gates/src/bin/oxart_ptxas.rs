//! Compile the PTX payloads of an `.oxart` section into cubins offline:
//! `oxart_ptxas <section.bin> <out.bin>` reads the bytes `objcopy -O binary
//! --only-section=.oxart` wrote, hands each bundle's PTX to the CUDA toolkit's
//! `ptxas -arch=<the bundle's target>` and writes `out.bin` as a section of
//! the same bundles in the same order, each with its name, target, compile
//! options and entries and the cubin as its one payload — the section shape
//! `oxart_jit --cubin` writes from the driver's link on a card. Before a
//! bundle is packed, every entry its PTX names must be a function symbol of
//! the cubin's ELF, refused by bundle and entry name otherwise. It prints one
//! `cubin <bundle> ptx=<bytes> cubin=<bytes> seconds=<s>` line a bundle, after
//! one `# ptxas …` line naming the toolkit that compiled them.
//!
//! No card and no gate lock: it runs on the build host, where
//! `tools/release/build.sh` builds the GB10 archive cross
//! (`BLOOMERY_RELEASE_TARGET=aarch64`). Host-only like `oxart_ptx`: the
//! package's device dependency sits behind the `gpu` feature, so plain
//! `cargo build --release -p bloomery-gpu-gates --bin oxart_ptxas` builds it
//! with no device crate.

use bloomery_gpu_gates::{GateError, exit_with, ptx};
use std::path::PathBuf;

const USAGE: &str = "usage: oxart_ptxas <section.bin> <out.bin>";

fn main() -> std::process::ExitCode {
    exit_with("oxart_ptxas", run())
}

fn run() -> Result<(), GateError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (section, out) = match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [section, out] => (section, out),
        _ => return Err(USAGE.into()),
    };
    let bytes = std::fs::read(section).map_err(|e| format!("read {section}: {e}"))?;
    let packed = pack(&bytes, "ptxas")?;
    std::fs::write(out, packed).map_err(|e| format!("write {out}: {e}"))
}

/// The section's bundles with each PTX payload replaced by ptxas's cubin of
/// it, as one section blob: the same bundles in the same order, the cubin each
/// one's only payload. One `cubin …` line a bundle, after the ptxas version.
fn pack(section: &[u8], ptxas: &str) -> Result<Vec<u8>, GateError> {
    let bundles = ptx::section_bundles(section)?;
    let modules = ptx::modules(&bundles)?;
    if modules.is_empty() {
        return Err("no PTX module in the section".into());
    }
    if modules.len() != bundles.len() {
        return Err("a bundle carries no PTX to compile".into());
    }
    println!("# ptxas {}", ptxas_version(ptxas)?);
    let tmp = TempDir::new()?;
    let mut written = Vec::new();
    for (n, (b, m)) in (1_usize..).zip(bundles.iter().zip(&modules)) {
        written.extend(cubin_bundle(b, m, ptxas, &tmp, n)?);
    }
    Ok(written)
}

/// Bundle `b` with its PTX module `m` replaced by ptxas's cubin of it, as a
/// container blob: the same name, target, compile options and entries, the
/// cubin its one payload. Every entry `m` names is checked against the
/// cubin's ELF symbols first.
fn cubin_bundle(
    b: &oxide_artifacts::OwnedArtifactBundle,
    m: &ptx::Module<'_>,
    ptxas: &str,
    tmp: &TempDir,
    n: usize,
) -> Result<Vec<u8>, GateError> {
    use oxide_artifacts::{
        ArtifactBundleSpec, ArtifactEntrySpec, ArtifactPayloadKind, ArtifactPayloadSpec,
    };
    let start = std::time::Instant::now();
    let ptx = tmp.path.join(format!("mod{n}.ptx"));
    std::fs::write(&ptx, m.text()).map_err(|e| format!("write {}: {e}", ptx.display()))?;
    let cubin_path = tmp.path.join(format!("mod{n}.cubin"));
    let compiled = std::process::Command::new(ptxas)
        .arg(format!("-arch={}", b.target))
        .arg("-o")
        .arg(&cubin_path)
        .arg(&ptx)
        .output()
        .map_err(|e| format!("bundle {}: run {ptxas}: {e}", b.name))?;
    if !compiled.status.success() {
        return Err(format!(
            "bundle {}: ptxas -arch={} refused its PTX ({}):\n{}",
            b.name,
            b.target,
            compiled.status,
            String::from_utf8_lossy(&compiled.stderr).trim_end()
        )
        .into());
    }
    let cubin =
        std::fs::read(&cubin_path).map_err(|e| format!("read {}: {e}", cubin_path.display()))?;
    check_entries(&b.name, &ptx::entries(std::slice::from_ref(m)), &cubin)?;
    let payload = format!("{}.cubin", b.name);
    let mut spec = ArtifactBundleSpec::new(&b.name, &b.target)
        .with_compile_options(b.compile_options)
        .with_payload(ArtifactPayloadSpec::new(
            ArtifactPayloadKind::Cubin,
            &payload,
            &cubin,
        ));
    for e in &b.entries {
        let entry = ArtifactEntrySpec::new(&e.symbol, e.kind);
        spec = spec.with_entry(match e.metadata {
            Some(meta) => entry.with_metadata(meta),
            None => entry,
        });
    }
    let blob = oxide_artifacts::build_artifact_blob(&spec)
        .map_err(|e| format!("bundle {}: {e}", b.name))?;
    println!(
        "cubin {} ptx={} cubin={} seconds={:.1}",
        b.name,
        m.text().len(),
        cubin.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(blob)
}

/// Every entry `names` must be a function symbol of `cubin`'s ELF, refused by
/// bundle and entry name otherwise — the offline stand-in for the
/// load-and-resolve check `oxart_jit --cubin` runs on a card.
fn check_entries(bundle: &str, names: &[String], cubin: &[u8]) -> Result<(), GateError> {
    let funcs = elf_func_names(cubin).ok_or_else(|| {
        format!(
            "bundle {bundle}: its cubin is not a little-endian ELF64 object with a symbol table"
        )
    })?;
    for name in names {
        if !funcs.contains(name) {
            return Err(format!(
                "bundle {bundle} entry {name}: not a function symbol of its cubin"
            )
            .into());
        }
    }
    Ok(())
}

/// The names of every function symbol (`STT_FUNC`) of a little-endian ELF64
/// object's symbol table — the shape ptxas writes a cubin in — or `None` when
/// the bytes are not an object a symbol table can be read from.
fn elf_func_names(bytes: &[u8]) -> Option<Vec<String>> {
    if bytes.get(..4) != Some(b"\x7fELF") || bytes.get(4) != Some(&2) || bytes.get(5) != Some(&1) {
        return None;
    }
    let u16r = |at: usize| Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?));
    let u32r = |at: usize| Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?));
    let u64r = |at: usize| Some(u64::from_le_bytes(bytes.get(at..at + 8)?.try_into().ok()?));
    let shoff = u64r(0x28)? as usize;
    let shentsize = u16r(0x3e)? as usize;
    let shnum = u16r(0x40)? as usize;
    if shoff == 0 || shentsize < 64 || shnum == 0 {
        return None;
    }
    let header = |i: usize| shoff.checked_add(i.checked_mul(shentsize)?)?;
    if header(shnum - 1)? + shentsize > bytes.len() {
        return None;
    }
    let mut names = Vec::new();
    let mut symtab = false;
    for i in 0..shnum {
        let sh = header(i)?;
        if u32r(sh + 4)? != 2 {
            continue; // SHT_SYMTAB alone is walked
        }
        symtab = true;
        let offset = u64r(sh + 0x18)? as usize;
        let size = u64r(sh + 0x20)? as usize;
        let link = u32r(sh + 0x28)? as usize;
        let entsize = u64r(sh + 0x38)? as usize;
        if entsize < 24 || offset.checked_add(size)? > bytes.len() || link >= shnum {
            return None;
        }
        let strh = header(link)?;
        let str_off = u64r(strh + 0x18)? as usize;
        let str_size = u64r(strh + 0x20)? as usize;
        if str_off.checked_add(str_size)? > bytes.len() {
            return None;
        }
        let strtab = &bytes[str_off..str_off + str_size];
        for k in 0..size / entsize {
            let sym = offset.checked_add(k.checked_mul(entsize)?)?;
            let name_at = u32r(sym)? as usize;
            if *bytes.get(sym + 4)? & 0xf != 2 {
                continue; // STT_FUNC alone counts
            }
            let end = strtab.get(name_at..)?.iter().position(|&c| c == 0)?;
            names.push(String::from_utf8_lossy(&strtab[name_at..name_at + end]).into_owned());
        }
    }
    symtab.then_some(names)
}

/// ptxas's own version line, the one that names the toolkit release.
fn ptxas_version(ptxas: &str) -> Result<String, GateError> {
    let out = std::process::Command::new(ptxas)
        .arg("--version")
        .output()
        .map_err(|e| format!("run {ptxas} --version: {e} (is the CUDA toolkit on PATH?)"))?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
    let line = lines
        .find(|l| l.contains("Cuda compilation tools"))
        .or_else(|| lines.next());
    line.map(|l| l.strip_prefix("# ").unwrap_or(l).to_string())
        .ok_or_else(|| format!("{ptxas} --version printed no version line").into())
}

/// A directory under the system temp dir removed when the pack ends, holding
/// each module's PTX and ptxas's cubin of it while they are handed over.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, GateError> {
        let path = std::env::temp_dir().join(format!("oxart_ptxas.{}", std::process::id()));
        std::fs::create_dir_all(&path).map_err(|e| format!("mkdir {}: {e}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_artifacts::ArtifactEntryKind;

    /// A module ptxas accepts: one visible entry whose body is a bare `ret;`.
    const MOD: &[u8] =
        b".version 8.7\n.target sm_86\n\n.visible .entry fix_dot(\n)\n{\n  ret;\n}\n";

    /// One bundle's section: the module as its PTX payload, one kernel entry record.
    fn section() -> Vec<u8> {
        use oxide_artifacts::{
            ArtifactBundleSpec, ArtifactEntrySpec, ArtifactPayloadKind, ArtifactPayloadSpec,
            build_artifact_blob,
        };
        let spec = ArtifactBundleSpec::new("fix", "sm_86")
            .with_payload(ArtifactPayloadSpec::new(
                ArtifactPayloadKind::Ptx,
                "bloomery_gpu.ptx",
                MOD,
            ))
            .with_entry(ArtifactEntrySpec::new("fix_dot", ArtifactEntryKind::Kernel));
        build_artifact_blob(&spec).expect("a valid bundle spec")
    }

    /// The fixture module through ptxas, packed into the section shape
    /// `oxart_jit --cubin` writes: the same bundle name, target and entries,
    /// the cubin the one payload, its entry a function symbol of the cubin's
    /// ELF. Skipped by name where ptxas is absent: the check needs the CUDA
    /// toolkit on the machine it runs on.
    #[test]
    #[ignore]
    fn hw_ptxas_packs_a_module_with_its_entries_as_func_symbols() {
        let Ok(version) = std::process::Command::new("ptxas")
            .arg("--version")
            .output()
        else {
            eprintln!("skip: ptxas is not on PATH, so no cubin can be compiled here");
            return;
        };
        assert!(
            version.status.success(),
            "ptxas --version failed: {}",
            version.status
        );
        let packed = pack(&section(), "ptxas").expect("ptxas compiles the fixture module");
        let bundles = ptx::section_bundles(&packed).expect("the packed section parses");
        assert_eq!(bundles.len(), 1, "one bundle in, one bundle out");
        let b = &bundles[0];
        assert_eq!(
            (&b.name, &b.target),
            (&"fix".to_string(), &"sm_86".to_string()),
            "the packed bundle keeps the PTX bundle's name and target"
        );
        let cubin = b
            .payload(oxide_artifacts::ArtifactPayloadKind::Cubin)
            .expect("the cubin payload");
        assert!(b.entry("fix_dot").is_some(), "the entry record is carried");
        assert!(
            elf_func_names(cubin).is_some_and(|names| names.contains(&"fix_dot".to_string())),
            "the entry is a function symbol of the cubin ptxas wrote"
        );
    }

    /// An entry the cubin's ELF does not name is refused by bundle and entry
    /// name, and a cubin that is no little-endian ELF64 object names the
    /// shape. Pure: the ELF bytes are built here.
    #[test]
    fn an_entry_missing_from_the_cubin_is_refused_by_name() {
        let cubin = elf64_symtab(&["present"]);
        let e = check_entries("fix", &["absent".to_string()], &cubin)
            .expect_err("the entry is not in the cubin")
            .to_string();
        assert!(e.contains("bundle fix entry absent"), "{e}");
        assert!(check_entries("fix", &["present".to_string()], &cubin).is_ok());
        let e = check_entries("fix", &["present".to_string()], b"not an elf")
            .expect_err("not an ELF")
            .to_string();
        assert!(e.contains("not a little-endian ELF64"), "{e}");
    }

    /// A little-endian ELF64 object: a null section, a symbol table of one
    /// global function symbol a name, and its string table.
    fn elf64_symtab(names: &[&str]) -> Vec<u8> {
        let mut strtab = vec![0u8];
        let mut syms = vec![0u8; 24]; // the null symbol
        for n in names {
            let at = strtab.len() as u32;
            strtab.extend_from_slice(n.as_bytes());
            strtab.push(0);
            syms.extend_from_slice(&at.to_le_bytes()); // st_name
            syms.push(0x12); // STB_GLOBAL << 4 | STT_FUNC
            syms.push(0); // st_other
            syms.extend_from_slice(&1u16.to_le_bytes()); // st_shndx
            syms.extend_from_slice(&0u64.to_le_bytes()); // st_value
            syms.extend_from_slice(&0u64.to_le_bytes()); // st_size
        }
        let mut out = vec![0u8; 64]; // the ELF header
        out[..4].copy_from_slice(b"\x7fELF");
        out[4] = 2; // ELFCLASS64
        out[5] = 1; // little-endian
        out[6] = 1; // EV_CURRENT
        let sym_off = out.len();
        out.extend_from_slice(&syms);
        let str_off = out.len();
        out.extend_from_slice(&strtab);
        let sh_off = out.len();
        out[0x28..0x30].copy_from_slice(&(sh_off as u64).to_le_bytes()); // e_shoff
        out[0x38..0x3a].copy_from_slice(&64u16.to_le_bytes()); // e_ehsize
        out[0x3e..0x40].copy_from_slice(&64u16.to_le_bytes()); // e_shentsize
        out[0x40..0x42].copy_from_slice(&3u16.to_le_bytes()); // e_shnum
        let shdr =
            |out: &mut Vec<u8>, ty: u32, offset: usize, size: usize, link: u32, entsize: u64| {
                let base = out.len();
                out.resize(base + 64, 0);
                out[base + 4..base + 8].copy_from_slice(&ty.to_le_bytes()); // sh_type
                out[base + 0x18..base + 0x20].copy_from_slice(&(offset as u64).to_le_bytes()); // sh_offset
                out[base + 0x20..base + 0x28].copy_from_slice(&(size as u64).to_le_bytes()); // sh_size
                out[base + 0x28..base + 0x2c].copy_from_slice(&link.to_le_bytes()); // sh_link
                out[base + 0x38..base + 0x40].copy_from_slice(&entsize.to_le_bytes()); // sh_entsize
            };
        shdr(&mut out, 0, 0, 0, 0, 0); // the null section
        shdr(&mut out, 2, sym_off, syms.len(), 2, 24); // SHT_SYMTAB, linked to section 2
        shdr(&mut out, 3, str_off, strtab.len(), 0, 0); // SHT_STRTAB
        out
    }
}
