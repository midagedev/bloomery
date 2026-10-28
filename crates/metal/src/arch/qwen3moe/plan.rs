//! `generate_metal --plan`: what a file asks of this Mac, read from its header
//! and tensor table alone.
//!
//! The refusals are [`super::check`]'s, the open's own, every one rather than
//! the first. The plan adds what the open does not print: the bytes the open
//! would make buffers of, beside the room the device reports
//! (`recommendedMaxWorkingSetSize`, a device query) and the host's free memory
//! now (`vm_stat`: other processes included, which the census's count of this
//! process's own buffers is not). The two room numbers are printed side by
//! side and judge nothing: the census's rule is the open's, and this plan
//! states what each says.
//!
//! A GGUF file whose data section is cut short (a header fetched by a range
//! read) is read from its header and tensor table alone ([`open_plan_header`]);
//! the open of such a file stays a named error. A checkpoint folder is read
//! through the model file's own header.

use std::fmt;
use std::path::Path;

use bloomery_device::Census;
use gguf::Split;
use gguf::write::{Layout as FileLayout, TensorDecl};
use model_hf::ModelFile;
use model_hf::model_file::Error as FileError;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use super::check::{Bytes, Item, Outcome, Tally};
use crate::census::MetalCensus;

/// The exit code of a header the plan cannot read: a named error, never an
/// empty plan.
pub const EXIT_UNREADABLE: u8 = 64;

/// The exit code of a file the open would refuse.
pub const EXIT_REFUSED: u8 = 3;

/// How much of the file's data section the file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Data {
    /// The file the open opens: every tensor's bytes lie in it.
    Complete,
    /// A header fetched alone: the file is `file_len` bytes, its tensors end
    /// at `need`.
    Cut { file_len: u64, need: u64 },
}

/// The model file a plan reads.
pub struct Header {
    pub file: ModelFile,
    pub data: Data,
}

impl Header {
    /// The header and tensor table the plan is read from.
    #[must_use]
    pub fn split(&self) -> &Split {
        self.file.header()
    }
}

/// Why the plan has no header to read, with the code it ends with
/// ([`OpenError::code`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpenError {
    /// The open refuses the file: a folder no map reads or whose config or
    /// tensors a map refuses, or a GGUF whose header reads and that the reader
    /// still refuses whole. The text is the open's own refusal.
    Refused(String),
    /// The header cannot be read: no GGUF, a table cut short, a tensor type no
    /// table sizes, or the first of several shards cut.
    Unreadable(String),
}

impl OpenError {
    /// The exit code of `generate_metal --plan` for this error.
    #[must_use]
    pub fn code(&self) -> u8 {
        match self {
            OpenError::Refused(_) => EXIT_REFUSED,
            OpenError::Unreadable(_) => EXIT_UNREADABLE,
        }
    }
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpenError::Refused(why) | OpenError::Unreadable(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for OpenError {}

/// [`open_plan_header`] with its error as text, for a caller that only prints
/// it.
pub fn open_header(path: &Path) -> Result<Header, String> {
    open_plan_header(path).map_err(|e| e.to_string())
}

/// The file at `path` as the plan reads it: the open's own model file when the
/// file is whole, else a GGUF's header and tensor table alone. A folder the
/// open refuses, or a GGUF that is whole and still refused by the reader, is
/// [`OpenError::Refused`]; a header the reader cannot take is
/// [`OpenError::Unreadable`].
pub fn open_plan_header(path: &Path) -> Result<Header, OpenError> {
    let refused = match ModelFile::open(path) {
        Ok(file) => {
            return Ok(Header {
                file,
                data: Data::Complete,
            });
        }
        Err(FileError::Gguf { source, .. }) => source,
        Err(e) => return Err(OpenError::Refused(e.to_string())),
    };
    let inv = gguf::inventory_of(path).map_err(|e| {
        OpenError::Unreadable(format!(
            "{}: {refused}; the header alone: {e}",
            path.display()
        ))
    })?;
    let mut end = 0u64;
    let mut decls = Vec::with_capacity(inv.tensors.len());
    for t in &inv.tensors {
        let nbytes = t.nbytes.ok_or_else(|| {
            OpenError::Unreadable(format!(
                "{}: tensor {:?} has the type id {}, which no table sizes",
                path.display(),
                t.name,
                t.type_id
            ))
        })?;
        end = end.max(t.offset.saturating_add(nbytes));
        decls.push(TensorDecl {
            name: t.name.clone(),
            dims: t.dims.clone(),
            type_id: t.type_id,
            nbytes,
        });
    }
    let need = inv.data_base.saturating_add(end);
    if inv.file_len >= need {
        return Err(OpenError::Refused(format!("{}: {refused}", path.display())));
    }
    if let Some(count) = inv.value("split.count").and_then(gguf::Value::as_unsigned)
        && count > 1
    {
        return Err(OpenError::Unreadable(format!(
            "{}: the first of {count} shards, cut short: a plan reads one file's header",
            path.display()
        )));
    }
    let layout = FileLayout::new(&inv.meta, decls).map_err(|e| {
        OpenError::Unreadable(format!(
            "{}: the header does not lay out: {e}",
            path.display()
        ))
    })?;
    let split = Split::header_only(path.to_path_buf(), &layout).map_err(|e| {
        OpenError::Unreadable(format!("{}: the header-only split: {e}", path.display()))
    })?;
    Ok(Header {
        file: ModelFile::Gguf(split),
        data: Data::Cut {
            file_len: inv.file_len,
            need,
        },
    })
}

/// The device's room, from the driver and no allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRoom {
    pub name: String,
    /// `recommendedMaxWorkingSetSize`.
    pub recommended: u64,
    /// `currentAllocatedSize` less nothing: what this process holds (none yet).
    pub allocated: u64,
}

/// The default device's room, or the reason there is none.
pub fn device_room() -> Result<DeviceRoom, String> {
    let default = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| "this machine has no Metal device".to_owned())?;
    let id = default.registryID().to_le_bytes();
    let infos = MetalCensus::census().map_err(|e| e.to_string())?;
    let info = infos
        .into_iter()
        .find(|i| i.uuid[..8] == id)
        .ok_or_else(|| "the default device is not in the driver's device list".to_owned())?;
    Ok(DeviceRoom {
        name: info.name,
        recommended: info.total_bytes,
        allocated: MetalCensus::allocated_of(&default).map_err(|e| e.to_string())?,
    })
}

/// The host's memory now, in bytes, by the pages `vm_stat` counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemory {
    pub free: u64,
    pub inactive: u64,
    pub speculative: u64,
    pub purgeable: u64,
}

impl HostMemory {
    /// What `vm_stat` prints now.
    pub fn probe() -> Result<HostMemory, String> {
        let out = std::process::Command::new("vm_stat")
            .output()
            .map_err(|e| format!("vm_stat: {e}"))?;
        if !out.status.success() {
            return Err(format!("vm_stat: {}", out.status));
        }
        HostMemory::parse(&String::from_utf8_lossy(&out.stdout))
    }

    /// The counts in `vm_stat`'s text, each a page count times the page size
    /// its first line names; a missing line is a named error.
    pub fn parse(text: &str) -> Result<HostMemory, String> {
        let page: u64 = text
            .lines()
            .next()
            .and_then(|l| l.split("page size of ").nth(1))
            .and_then(|l| l.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .ok_or("vm_stat: the first line names no page size")?;
        let pages = |key: &str| -> Result<u64, String> {
            text.lines()
                .find_map(|l| l.strip_prefix(key)?.strip_prefix(':'))
                .and_then(|n| n.trim().trim_end_matches('.').parse::<u64>().ok())
                .map(|n| n * page)
                .ok_or_else(|| format!("vm_stat: no `{key}` count"))
        };
        Ok(HostMemory {
            free: pages("Pages free")?,
            inactive: pages("Pages inactive")?,
            speculative: pages("Pages speculative")?,
            purgeable: pages("Pages purgeable")?,
        })
    }

    /// The free pages and the ones the OS gives back on demand: derived, a
    /// sum of the four counts.
    #[must_use]
    pub fn available(&self) -> u64 {
        self.free + self.inactive + self.speculative + self.purgeable
    }
}

/// The text of a refusal with the numbers that differ between its repeats
/// replaced: a layer's index and a tensor's block index, so a file's thirty
/// layers of one refusal are one kind.
#[must_use]
pub fn kind_of(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while !rest.is_empty() {
        let boundary = out
            .chars()
            .last()
            .is_none_or(|c| !c.is_ascii_alphanumeric());
        let hit = ["blk.", "layer "]
            .iter()
            .find(|p| boundary && rest.starts_with(**p))
            .filter(|p| rest[p.len()..].starts_with(|c: char| c.is_ascii_digit()));
        match hit {
            Some(p) => {
                out.push_str(p);
                out.push('N');
                rest = rest[p.len()..].trim_start_matches(|c: char| c.is_ascii_digit());
            }
            None => {
                let c = rest.chars().next().expect("rest is not empty");
                out.push(c);
                rest = &rest[c.len_utf8()..];
            }
        }
    }
    out
}

/// The items grouped by [`kind_of`] their text, in the order of first
/// appearance, each with its count.
#[must_use]
pub fn kinds(items: &[Item]) -> Vec<(String, usize)> {
    let mut out: Vec<(String, usize)> = Vec::new();
    for item in items {
        let kind = kind_of(&item.error.to_string());
        match out.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, n)) => *n += 1,
            None => out.push((kind, 1)),
        }
    }
    out
}

fn tally(t: &Tally) -> String {
    format!(
        "buffers={} requested={} paged={}",
        t.buffers, t.requested, t.paged
    )
}

/// What the plan printed and the code it ends with.
pub struct Printed {
    pub lines: Vec<String>,
    pub code: u8,
}

/// The plan's lines for `outcome`, read at a cache of `ctx` rows from the file
/// at `path` (`data`: how much of it is there), beside the room.
#[must_use]
pub fn render(
    path: &Path,
    data: Data,
    ctx: usize,
    outcome: &Outcome,
    device: &Result<DeviceRoom, String>,
    host: &Result<HostMemory, String>,
) -> Printed {
    let mut lines = vec![format!("plan: file={}", path.display())];
    match data {
        Data::Complete => lines.push("plan: data=complete".to_owned()),
        Data::Cut { file_len, need } => lines.push(format!(
            "plan: data=cut file_bytes={file_len} tensors_end_bytes={need} (header and tensor \
             table alone: no weight byte is read; the open of this file is a named error)"
        )),
    }
    if let Some(s) = &outcome.spec {
        lines.push(format!(
            "plan: arch={} layers={} (delta {}, conv {}, attention {}) hidden={} vocab={} \
             experts={} picks={} trained_ctx={}",
            s.arch,
            s.layers,
            s.delta,
            s.conv,
            s.attention,
            s.hidden,
            s.vocab,
            s.experts,
            s.picks,
            s.ctx_train
        ));
    }
    lines.push(format!("plan: ctx={ctx}"));
    match device {
        Ok(d) => lines.push(format!(
            "plan: device={:?} recommended_max_working_set_bytes={} allocated_bytes={}",
            d.name, d.recommended, d.allocated
        )),
        Err(e) => lines.push(format!("plan: device=unavailable ({e})")),
    }
    match host {
        Ok(h) => lines.push(format!(
            "plan: host free_bytes={} inactive_bytes={} speculative_bytes={} \
             purgeable_bytes={} available_bytes={} (vm_stat now; available is the four summed)",
            h.free,
            h.inactive,
            h.speculative,
            h.purgeable,
            h.available()
        )),
        Err(e) => lines.push(format!("plan: host=unavailable ({e})")),
    }
    if let Some(checked) = &outcome.checked {
        match checked.bytes() {
            Ok(b) => lines.extend(bytes_lines(&b, device)),
            Err(e) => lines.push(format!("plan: bytes=unavailable ({e})")),
        }
    }
    for item in &outcome.items {
        match item.layer {
            Some(l) => lines.push(format!("plan: refused layer={l}: {}", item.error)),
            None => lines.push(format!("plan: refused: {}", item.error)),
        }
    }
    let by_kind = kinds(&outcome.items);
    for (kind, n) in &by_kind {
        lines.push(format!("plan: kind x{n}: {kind}"));
    }
    let code = if outcome.items.is_empty() {
        lines.push("plan: opens".to_owned());
        0
    } else {
        lines.push(format!("plan: refused, {} items", outcome.items.len()));
        EXIT_REFUSED
    };
    Printed { lines, code }
}

fn bytes_lines(b: &Bytes, device: &Result<DeviceRoom, String>) -> Vec<String> {
    let total = b.total();
    let mut lines = vec![
        format!("plan: bytes weights {}", tally(&b.weights)),
        format!("plan: bytes kv {}", tally(&b.kv)),
        format!("plan: bytes state {}", tally(&b.state)),
        format!("plan: bytes scratch {}", tally(&b.scratch)),
        format!("plan: bytes rows {}", tally(&b.rows)),
        format!(
            "plan: bytes total {} (requested: the bytes asked of the driver, exact; paged: each \
             buffer rounded up to a 16 KiB page, an upper bound; the driver's count lies \
             between)",
            tally(&total)
        ),
    ];
    if let Ok(d) = device {
        let over = i128::from(total.paged) - i128::from(d.recommended);
        lines.push(format!(
            "plan: room paged_minus_recommended_bytes={over} (negative: the upper bound is under \
             the device's working set; the census's rule is the open's, not this line)"
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    const VM_STAT: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   230224.
Pages active:                                 666358.
Pages inactive:                               628846.
Pages speculative:                             49004.
Pages throttled:                                   0.
Pages wired down:                             206965.
Pages purgeable:                                3987.
";

    #[test]
    fn vm_stat_counts_are_pages_times_the_page_size() {
        let h = HostMemory::parse(VM_STAT).expect("the text reads");
        assert_eq!(h.free, 230_224 * 16_384);
        assert_eq!(h.inactive, 628_846 * 16_384);
        assert_eq!(h.speculative, 49_004 * 16_384);
        assert_eq!(h.purgeable, 3_987 * 16_384);
        assert_eq!(h.available(), (230_224 + 628_846 + 49_004 + 3_987) * 16_384);
    }

    #[test]
    fn a_vm_stat_without_a_count_is_a_named_error() {
        let cut = VM_STAT.replace("Pages purgeable:", "Pages gone:");
        assert_eq!(
            HostMemory::parse(&cut),
            Err("vm_stat: no `Pages purgeable` count".to_owned())
        );
        assert!(HostMemory::parse("not vm_stat").is_err());
    }

    /// A directory is read as the checkpoint folder it is: one the folder open refuses is that open's
    /// refusal by name, not a GGUF header that failed.
    #[test]
    fn a_directory_that_is_no_checkpoint_folder_is_the_folder_opens_refusal() {
        let dir =
            std::env::temp_dir().join(format!("bloomery-metal-plan-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the directory is made");
        let refused = open_header(&dir).err();
        std::fs::remove_dir_all(&dir).expect("the directory is removed");
        let e = refused.expect("an empty directory is no checkpoint folder");
        assert!(
            e.starts_with(&format!("checkpoint folder {}: ", dir.display())),
            "{e}"
        );
    }

    /// A folder the open refuses is [`OpenError::Refused`], which ends the plan with 3; a header
    /// cut inside its own table is [`OpenError::Unreadable`], which ends it with 64.
    #[test]
    fn a_refused_folder_ends_the_plan_with_3_and_an_unreadable_header_with_64() {
        let tmp = std::env::temp_dir();
        let pid = std::process::id();
        let dir = tmp.join(format!("bloomery-metal-plan-refused-{pid}"));
        std::fs::create_dir_all(&dir).expect("the directory is made");
        std::fs::write(dir.join("config.json"), r#"{"model_type": "nope"}"#)
            .expect("the config is written");
        let folder = open_plan_header(&dir).err();
        std::fs::remove_dir_all(&dir).expect("the directory is removed");
        let folder = folder.expect("a model_type no map reads is refused");
        assert!(matches!(folder, OpenError::Refused(_)), "{folder:?}");
        assert_eq!(folder.code(), EXIT_REFUSED);
        assert!(
            folder
                .to_string()
                .contains("model_type \"nope\" has no folder map"),
            "{folder}"
        );

        let file = tmp.join(format!("bloomery-metal-plan-cut-{pid}.gguf"));
        let mut cut = b"GGUF".to_vec();
        cut.extend(3u32.to_le_bytes());
        cut.extend(1u64.to_le_bytes());
        cut.extend(0u64.to_le_bytes());
        std::fs::write(&file, cut).expect("the file is written");
        let header = open_plan_header(&file).err();
        std::fs::remove_file(&file).expect("the file is removed");
        let header = header.expect("a header cut inside its tensor table is no plan");
        assert!(matches!(header, OpenError::Unreadable(_)), "{header:?}");
        assert_eq!(header.code(), EXIT_UNREADABLE);
    }

    #[test]
    fn a_layers_repeats_of_a_refusal_are_one_kind() {
        assert_eq!(
            kind_of("the plan: blk.12.attn_qkv.weight is q8_0; layer 3: x blk.0.y"),
            "the plan: blk.N.attn_qkv.weight is q8_0; layer N: x blk.N.y"
        );
        // A word that only ends in the prefix, and a prefix with no number,
        // are text.
        assert_eq!(kind_of("ablk.1 layer x"), "ablk.1 layer x");
    }
}
