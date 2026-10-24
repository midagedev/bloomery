//! A Hugging Face checkpoint folder as a load reads it ([`Checkpoint`], a `model::part::Source`):
//! each tensor of the program whole or the part a rank holds, by the load's rule
//! ([`model::part::Take`]). The map from the checkpoint's tensors to the program's is the
//! model's ([`Made`]); the checkpoint knows no model and no rank count. Every part is planned from
//! headers alone before any value is read.
//!
//! A folder's tensor is normalized into its card type (`hf::modelopt`), each weight and block
//! scale cut first, so a part reads only its own stored values; a tensor the model moves into its
//! layout after normalizing is made whole, moved, and then cut.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use engram::EngramError;
use engram::direct::{DIRECT_ALIGN, DirectFile, aligned_span};
use gguf::{GgmlType, TensorInfo};
use hf::modelopt::{self, Normal, Use};
use hf::quant_config::{Placed, Plan};
use model::part::{self, Part, Planned, Source, Take};
use model::placement::tp::{TpError, part_of};
use safetensors::{Dtype, Folder, Raw};

/// Why a part was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Part(#[from] part::Error),
    #[error(transparent)]
    Normal(#[from] modelopt::Error),
    /// A model's move refused the tensor; the reason names it.
    #[error("{0}")]
    Step(String),
    /// A stored tensor's file pages could not be released.
    #[error(transparent)]
    Release(#[from] model::placement::PlacementError),
    #[error(transparent)]
    Tensors(#[from] safetensors::Error),
    #[error(transparent)]
    Direct(#[from] engram::EngramError),
}

impl From<TpError> for Error {
    fn from(e: TpError) -> Error {
        Error::Part(e.into())
    }
}

fn refuse(name: &str, what: impl Into<String>) -> Error {
    part::refuse(name, what).into()
}

/// A model's move of a normalized tensor into its program's layout.
pub type Step = Box<dyn Fn(&mut Normal) -> Result<(), String> + Send + Sync>;

/// How a checkpoint folder makes one tensor of the program, as the model's map gives it.
pub struct Made {
    /// The checkpoint tensors: one, or a routed stack's experts in expert order.
    pub sources: Vec<String>,
    /// How its role reads it, which with its format picks its card type
    /// ([`modelopt::type_of`]).
    pub role: Use,
    /// ggml's dims, the fastest axis first.
    pub dims: Vec<u64>,
    /// The move into the program's layout after normalizing, if the model has one.
    pub step: Option<Step>,
}

/// A checkpoint folder as a load reads it: its tensors, its quantization plan, and the tensors
/// the model's map makes of them, by the program's names.
pub struct Checkpoint {
    folder: Folder,
    plan: Plan,
    made: BTreeMap<String, Made>,
}

impl Checkpoint {
    /// The tensors `made` makes of `folder`'s under `plan`; a source the plan does not place, or
    /// a tensor of no sources, is refused by name.
    pub fn new(
        folder: Folder,
        plan: Plan,
        made: BTreeMap<String, Made>,
    ) -> Result<Checkpoint, Error> {
        for (name, m) in &made {
            if m.sources.is_empty() {
                return Err(refuse(name, "is made of no checkpoint tensor"));
            }
            if let Some(s) = m.sources.iter().find(|s| !plan.placed.contains_key(*s)) {
                let why = plan
                    .refused
                    .get(s)
                    .map_or("is not in the checkpoint", String::as_str);
                return Err(refuse(name, format!("its source {s} {why}")));
            }
        }
        Ok(Checkpoint { folder, plan, made })
    }

    /// The tensors the map makes, by name.
    #[cfg(test)]
    pub(crate) fn made(&self) -> &BTreeMap<String, Made> {
        &self.made
    }

    fn placed<'s>(&'s self, m: &'s Made) -> Vec<(&'s str, &'s Placed)> {
        // `new` held every source placed.
        m.sources
            .iter()
            .filter_map(|s| Some((s.as_str(), self.plan.placed.get(s)?)))
            .collect()
    }

    /// The card type `name`, made as `m`, is normalized into.
    fn card_type(&self, name: &str, m: &Made) -> Result<GgmlType, Error> {
        let format = modelopt::format_of(&self.placed(m))?;
        modelopt::type_of(format, m.role)
            .map(GgmlType::from_u32)
            .ok_or_else(|| {
                refuse(
                    name,
                    format!("{format} read as {:?} has no card type", m.role),
                )
            })
    }
}

impl Source for Checkpoint {
    type Error = Error;

    /// Every tensor the map makes; a part off a stored tensor's values or scale blocks is also
    /// refused by name before any value is read.
    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Error> {
        let mut out = Vec::new();
        for (name, m) in &self.made {
            let whole = whole(name, self.card_type(name, m)?, &m.dims)?;
            let Some(at) = part::take(&rule, &whole)? else {
                continue;
            };
            let (info, _) = part_of(&whole, at.clone())?;
            // A moved tensor, or a whole row table, is checked whole as it is made in read.
            if m.step.is_none() && (at.is_some() || m.role != Use::Rows) {
                let cut = at.clone().unwrap_or((0, 0..m.dims[0]));
                modelopt::check_part(
                    |n| self.folder.raw(n),
                    &self.placed(m),
                    m.role,
                    &m.dims,
                    cut,
                )?;
            }
            out.push(Planned { at, info });
        }
        Ok(out)
    }

    /// A tensor its format refuses, or that does not normalize into the planned type and bytes,
    /// is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Error> {
        let name = p.info.name.as_str();
        let m = self
            .made
            .get(name)
            .ok_or_else(|| refuse(name, "is not one the map makes"))?;
        let placed = self.placed(m);
        let ty = self.card_type(name, m)?;
        // A whole tensor's stored bytes are read into memory of the read's own ([`read_whole`]);
        // a part reads only its rows through the mapping.
        let mut read = HashMap::new();
        if p.at.is_none() || m.step.is_some() {
            for n in stored(&placed) {
                if let Some(w) = read_whole(&self.folder, n)? {
                    read.insert(n, w);
                }
            }
        }
        let get = |n: &str| match read.get(n) {
            Some(w) => Ok(w.raw()),
            None => self.folder.raw(n),
        };
        let n = match (&p.at, &m.step) {
            (Some(at), None) => {
                modelopt::normalize_part(get, &placed, m.role, &m.dims, at.clone())?
            }
            _ => {
                let mut n = modelopt::normalize(get, &placed, m.role)?;
                if let Some(step) = &m.step {
                    step(&mut n).map_err(Error::Step)?;
                }
                n
            }
        };
        let whole = whole(name, ty, &m.dims)?;
        let want = if m.step.is_some() { &whole } else { &p.info };
        let count = |d: &[u64]| d.iter().try_fold(1u64, |a, &v| a.checked_mul(v));
        if GgmlType::from_u32(n.type_id) != ty
            || count(&n.dims) != count(&want.dims)
            || n.bytes.len() as u64 != want.nbytes
        {
            return Err(refuse(
                name,
                format!(
                    "normalized into {} {:?} ({} bytes), not the planned {} {:?} ({} bytes)",
                    GgmlType::from_u32(n.type_id),
                    n.dims,
                    n.bytes.len(),
                    ty,
                    want.dims,
                    want.nbytes
                ),
            ));
        }
        let bytes = match (&p.at, &m.step) {
            (Some(_), Some(_)) => {
                let (_, runs) = part_of(&whole, p.at.clone())?;
                runs.into_iter()
                    .flat_map(|r| &n.bytes[r])
                    .copied()
                    .collect()
            }
            _ => n.bytes,
        };
        Ok(Part::made(
            p.info.clone(),
            bytes,
            ty.reads_scale().then_some(n.scale),
        ))
    }

    /// Every stored tensor the part was read from, its values and its companions (scales, the
    /// packed shape), each the whole pages inside its bytes: a read normalizes into bytes of its
    /// own ([`Part::made`]), so no later reader of the part takes them from the file.
    fn release(&self, p: &Planned) -> Result<(), Error> {
        let name = p.info.name.as_str();
        let m = self
            .made
            .get(name)
            .ok_or_else(|| refuse(name, "is not one the map makes"))?;
        for stored in stored(&self.placed(m)) {
            let (path, map, at) = self.folder.mapped(stored)?;
            // SAFETY: `Folder::mapped` gives the shard file's own read-only mapping whole.
            unsafe { model::placement::host_lock::release_file_pages(&path, map, at) }?;
        }
        Ok(())
    }
}

/// A stored tensor read whole into memory of its own ([`read_whole`]).
struct Whole {
    dtype: Dtype,
    shape: Vec<usize>,
    buf: Vec<u8>,
    /// The tensor's bytes in `buf`.
    at: Range<usize>,
}

impl Whole {
    fn raw(&self) -> Raw<'_> {
        Raw {
            dtype: self.dtype,
            shape: &self.shape,
            bytes: &self.buf[self.at.clone()],
        }
    }
}

/// Stored tensor `name` read whole from its shard past the page cache (engram's direct reader)
/// into new memory advised onto huge pages, so the read takes the drive's rate, faults once each
/// 2 MiB and leaves the page cache as it was. `None` where the shard cannot be opened for direct
/// reads or its mount refuses them ([`DirectFile::open`]): the caller reads the tensor through
/// the mapping.
fn read_whole(folder: &Folder, name: &str) -> Result<Option<Whole>, Error> {
    let (path, range) = folder.file_range(name)?;
    let file = match DirectFile::open(&path) {
        Ok(file) => file,
        Err(EngramError::OpenDirect { .. } | EngramError::DirectProbe { .. }) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let raw = folder.raw(name)?;
    let (start, span, need) = aligned_span(range.start, range.end - range.start);
    let mut buf = vec![0u8; span + DIRECT_ALIGN];
    os::advise_huge_within(&mut buf);
    let lead = buf.as_ptr().align_offset(DIRECT_ALIGN);
    if file.read_span(&mut buf[lead..lead + span], start)? < need {
        return Err(refuse(
            name,
            format!("{} ends inside its bytes", path.display()),
        ));
    }
    let skip = need - raw.bytes.len();
    Ok(Some(Whole {
        dtype: raw.dtype,
        shape: raw.shape.to_vec(),
        buf,
        at: lead + skip..lead + need,
    }))
}

/// The stored tensors a tensor placed as `placed` is made of: each source, then the companions
/// its place names (scales, the packed shape).
fn stored<'s>(placed: &[(&'s str, &'s Placed)]) -> impl Iterator<Item = &'s str> {
    placed.iter().flat_map(|&(source, placed)| {
        std::iter::once(source).chain(placed.companions.iter().map(String::as_str))
    })
}

/// The whole tensor `name` of `ty` and `dims` as a header entry; dims that are not whole blocks
/// of the type, or bytes past u64, are refused by name.
fn whole(name: &str, ty: GgmlType, dims: &[u64]) -> Result<TensorInfo, Error> {
    let nbytes = ty
        .blck_size()
        .zip(ty.type_size())
        .filter(|&(b, _)| dims.first().is_some_and(|d0| b > 0 && d0.is_multiple_of(b)))
        .and_then(|(b, s)| gguf::tensor_bytes(b, s, dims))
        .ok_or_else(|| {
            refuse(
                name,
                format!("{ty} {dims:?} are not whole blocks of the type"),
            )
        })?;
    Ok(TensorInfo {
        name: name.to_string(),
        dims: dims.to_vec(),
        ty,
        offset: 0,
        nbytes,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gguf::Split;
    use gguf::write::{Layout, TensorDecl, Writer};
    use hf::modelopt::Plain;
    use hf::quant_config::Format;
    use model::placement::tp::Cut;

    use crate::test_support::temp_dir;

    /// One stored tensor: name, dtype, shape, bytes.
    type Stored = (String, &'static str, Vec<usize>, Vec<u8>);

    /// `n` bytes, each `f(i)`.
    fn bytes(n: usize, f: impl Fn(usize) -> u8) -> Vec<u8> {
        (0..n).map(f).collect()
    }

    /// The checkpoint every case reads: an NVFP4 linear, a routed stack of two NVFP4 experts
    /// with `csf_32x2_v1` scales, MXFP8, FP8 with one scale a 128 by 128 block, BF16, an F32
    /// vector, and a BF16 matrix the map transposes after normalizing. Values are finite and
    /// every scale byte a scale.
    fn checkpoint(dir: &std::path::Path) -> Checkpoint {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let e4m3 = |i: usize| (i * 7 % 0x7e) as u8;
        let mut t: Vec<Stored> = vec![
            (
                "nv.w".into(),
                "U8",
                vec![8, 128],
                bytes(1024, |i| (i * 37) as u8),
            ),
            (
                "nv.s".into(),
                "F8_E4M3",
                vec![8, 16],
                bytes(128, |i| 0x30 + (i % 13) as u8),
            ),
            ("nv.t".into(), "F32", vec![], f32s(&[0.5])),
            ("rows.w".into(), "F8_E4M3", vec![3, 2], vec![0x38; 6]),
            ("rows.s".into(), "F32", vec![], f32s(&[0.5])),
            ("mx.w".into(), "F8_E4M3", vec![8, 512], bytes(4096, e4m3)),
            (
                "mx.s".into(),
                "U8",
                vec![8, 16],
                bytes(128, |i| 120 + (i % 9) as u8),
            ),
            ("fb.w".into(), "F8_E4M3", vec![256, 256], bytes(65536, e4m3)),
            (
                "fb.s".into(),
                "F32",
                vec![2, 2],
                f32s(&[0.5, 0.25, 2.0, 1.0]),
            ),
            (
                "bf.w".into(),
                "BF16",
                vec![4, 64],
                bytes(512, |i| if i % 2 == 0 { i as u8 } else { 0x3f }),
            ),
            (
                "v.w".into(),
                "F32",
                vec![64],
                f32s(&(0..64).map(|i| i as f32).collect::<Vec<_>>()),
            ),
            (
                "tr.w".into(),
                "BF16",
                vec![8, 4],
                bytes(64, |i| if i % 2 == 0 { i as u8 } else { 0x40 }),
            ),
        ];
        for e in 0..2 {
            let p = format!("cs.{e}");
            t.push((
                format!("{p}.w"),
                "U8",
                vec![32, 128],
                bytes(4096, |i| (i * 11 + e) as u8),
            ));
            t.push((
                format!("{p}.s.csf32_bases"),
                "U8",
                vec![32],
                bytes(32, |r| 0x30 + (r % 8) as u8),
            ));
            t.push((
                format!("{p}.s.csf32_codes"),
                "U8",
                vec![256],
                bytes(256, |i| (i * 29 + e) as u8),
            ));
            t.push((
                format!("{p}.s.csf32_page_bases"),
                "U32",
                vec![1],
                vec![0; 4],
            ));
            t.push((format!("{p}.s.csf32_offsets"), "U16", vec![9], vec![0; 18]));
            t.push((format!("{p}.s.csf32_exceptions"), "U16", vec![0], vec![]));
            t.push((
                format!("{p}.t"),
                "F32",
                vec![],
                f32s(&[0.25 * (e + 1) as f32]),
            ));
        }
        let entries: Vec<_> = t
            .iter()
            .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        std::fs::write(dir.join("model.safetensors"), safetensors::write(&entries)).unwrap();
        let map: BTreeMap<_, _> = t
            .iter()
            .map(|(n, ..)| (n.as_str(), "model.safetensors"))
            .collect();
        std::fs::write(
            dir.join(Folder::INDEX),
            serde_json::json!({ "weight_map": map }).to_string(),
        )
        .unwrap();
        let place = |format, companions: &[&str]| Placed {
            format,
            companions: companions.iter().map(|c| c.to_string()).collect(),
        };
        let mut placed = BTreeMap::from([
            ("nv.w".to_string(), place(Format::Nvfp4, &["nv.s", "nv.t"])),
            ("rows.w".into(), place(Format::Fp8Tensor, &["rows.s"])),
            ("mx.w".into(), place(Format::Mxfp8, &["mx.s"])),
            (
                "fb.w".into(),
                place(
                    Format::Fp8Block {
                        rows: 128,
                        cols: 128,
                    },
                    &["fb.s"],
                ),
            ),
            ("bf.w".into(), place(Format::Bf16, &[])),
            ("v.w".into(), place(Format::F32, &[])),
            ("tr.w".into(), place(Format::Bf16, &[])),
        ]);
        for e in 0..2 {
            let parts = ["bases", "codes", "page_bases", "offsets", "exceptions"]
                .map(|p| format!("cs.{e}.s.csf32_{p}"));
            let mut companions: Vec<&str> = parts.iter().map(String::as_str).collect();
            let t = format!("cs.{e}.t");
            companions.push(&t);
            placed.insert(format!("cs.{e}.w"), place(Format::Nvfp4Csf32, &companions));
        }
        let plan = Plan {
            placed,
            ..Plan::default()
        };
        let matrix = Use::Matrix(Plain::F32);
        let made = |sources: &[&str], role, dims: &[u64], step: Option<Step>| Made {
            sources: sources.iter().map(|s| s.to_string()).collect(),
            role,
            dims: dims.to_vec(),
            step,
        };
        let transpose: Step = Box::new(|n: &mut Normal| {
            let [c, r] = n.dims[..] else {
                return Err("not 2-D".into());
            };
            let (c, r) = (c as usize, r as usize);
            let at = |i: usize, j: usize| &n.bytes[(i * c + j) * 2..][..2];
            n.bytes = (0..c)
                .flat_map(|j| (0..r).flat_map(move |i| at(i, j).to_vec()))
                .collect();
            n.dims = vec![r as u64, c as u64];
            Ok(())
        });
        let bf = Use::Matrix(Plain::Bf16);
        let made = BTreeMap::from([
            (
                "nv.weight".to_string(),
                made(&["nv.w"], matrix, &[256, 8], None),
            ),
            (
                "nv-one.weight".into(),
                made(&["nv.w"], matrix, &[256, 8, 1], None),
            ),
            (
                "cs.weight".into(),
                made(&["cs.0.w", "cs.1.w"], matrix, &[256, 32, 2], None),
            ),
            (
                "rows.weight".into(),
                made(&["rows.w"], Use::Rows, &[2, 3], None),
            ),
            ("mx.weight".into(), made(&["mx.w"], matrix, &[512, 8], None)),
            (
                "fb.weight".into(),
                made(&["fb.w"], matrix, &[256, 256], None),
            ),
            ("bf.weight".into(), made(&["bf.w"], bf, &[64, 4], None)),
            (
                "v.weight".into(),
                made(&["v.w"], Use::Matrix(Plain::F32), &[64], None),
            ),
            (
                "tr.weight".into(),
                made(&["tr.w"], bf, &[8, 4], Some(transpose)),
            ),
        ]);
        Checkpoint::new(Folder::open(dir).unwrap(), plan, made).unwrap()
    }

    /// The GGUF file of the same model as `src`, written at `path`: each tensor read whole, in its
    /// card type, with its `.scale` beside it.
    pub(crate) fn gguf_of(src: &Checkpoint, path: &std::path::Path) -> Split {
        let planned = src.plan(|_, _, _| Ok::<_, String>(Take::Whole)).unwrap();
        let mut decls = Vec::new();
        let mut data = Vec::new();
        for p in &planned {
            let part = src.read(p).unwrap();
            decls.push(TensorDecl {
                name: p.info.name.clone(),
                dims: p.info.dims.clone(),
                type_id: p.info.ty.as_u32(),
                nbytes: p.info.nbytes,
            });
            data.push((p.info.name.clone(), part.spans().concat()));
            if let Some(scale) = part.scale {
                let name = p.info.name.replace(".weight", ".scale");
                let b: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
                decls.push(TensorDecl {
                    name: name.clone(),
                    dims: vec![scale.len() as u64],
                    type_id: GgmlType::F32.as_u32(),
                    nbytes: b.len() as u64,
                });
                data.push((name, b));
            }
        }
        let mut w = Writer::new(
            std::io::BufWriter::new(std::fs::File::create(path).unwrap()),
            Layout::new(&[], decls).unwrap(),
        )
        .unwrap();
        for (name, b) in &data {
            w.tensor(name, b).unwrap();
        }
        w.finish().unwrap();
        Split::open(path).unwrap()
    }

    /// The part of tensor `name` rank `rank` of `n` holds cut along `dim`, every other tensor
    /// skipped: whole at one rank, else `n` parts of whole blocks of the card type.
    fn rule(
        name: &'static str,
        dim: usize,
        rank: usize,
        n: usize,
    ) -> impl Fn(&str, GgmlType, &[u64]) -> Result<Take, String> {
        move |t, ty, dims| {
            if t != name {
                return Ok(Take::Skip);
            }
            if n == 1 {
                return Ok(Take::Whole);
            }
            let blocks = [ty.blck_size().unwrap_or(1), 1];
            let at = Cut { dim, unit: 1 }
                .span(t, dims, blocks, rank, n)
                .map_err(|e| e.to_string())?;
            Ok(Take::Part(dim, at))
        }
    }

    /// A part as both sources must give it: name, card type, dims, bytes and scales.
    type Read = (String, GgmlType, Vec<u64>, Vec<u8>, Option<Vec<f32>>);

    /// The one part a plan of `rule` reads.
    fn read_one(
        src: &impl Source,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, String>,
    ) -> Result<Read, String> {
        let planned = src.plan(rule).map_err(|e| e.to_string())?;
        let [p] = planned.as_slice() else {
            panic!("{} parts planned", planned.len())
        };
        let part = src.read(p).map_err(|e| e.to_string())?;
        let i = &part.info;
        assert_eq!(
            part.spans().iter().map(|s| s.len() as u64).sum::<u64>(),
            i.nbytes
        );
        Ok((
            i.name.clone(),
            i.ty,
            i.dims.clone(),
            part.spans().concat(),
            part.scale,
        ))
    }

    #[test]
    fn a_row_table_is_read_whole_but_not_cut() {
        let dir = temp_dir("rows");
        let c = checkpoint(&dir);
        let f = &c;
        let whole = read_one(f, rule("rows.weight", 0, 0, 1)).unwrap();
        assert_eq!(whole.1, GgmlType::F8E4M3_ROWS);
        assert_eq!(whole.2, [2, 3]);
        assert_eq!(whole.3, [0x38; 6]);
        assert_eq!(whole.4, Some(vec![0.5]));
        let file = gguf_of(f, &dir.join("model.gguf"));
        assert_eq!(
            read_one(&file, rule("rows.weight", 0, 0, 1)).unwrap(),
            whole
        );
        for dim in [0, 1] {
            let error = f
                .plan(|name, _, _| {
                    Ok::<_, String>(if name == "rows.weight" {
                        Take::Part(dim, 0..1)
                    } else {
                        Take::Skip
                    })
                })
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("rows.w") && error.contains("row table"),
                "{error}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The same model as a GGUF file and as a checkpoint folder gives each rank the same part, in
    /// the same card type and bytes, with the same tensor scales: at one rank every tensor
    /// whole, uncut; at two and four ranks along every dim of each format, a stack's experts and
    /// a tensor the map moves. A part off the card type's blocks is refused by both sources;
    /// one that splits FP8's 128-row scale blocks by the folder alone, naming its scale, in its
    /// plan. A part read again after its file pages' release gives the same bytes.
    #[test]
    fn a_folder_and_its_gguf_give_every_rank_the_same_part() {
        let dir = temp_dir("same");
        let c = checkpoint(&dir);
        let file = gguf_of(&c, &dir.join("model.gguf"));
        let (g, f) = (&file, &c);
        fn names(s: &impl Source) -> Vec<String> {
            let mut v: Vec<String> = s
                .plan(|_, _, _| Ok::<_, String>(Take::Whole))
                .unwrap()
                .into_iter()
                .map(|p| p.info.name)
                .collect();
            v.sort();
            v
        }
        assert_eq!(names(g), names(f));
        let cases: [(&str, usize); 8] = [
            ("nv.weight", 2),
            ("nv-one.weight", 2),
            ("cs.weight", 3),
            ("mx.weight", 2),
            ("fb.weight", 2),
            ("bf.weight", 2),
            ("v.weight", 1),
            ("tr.weight", 2),
        ];
        let one = read_one(f, rule("nv-one.weight", 0, 0, 1)).unwrap();
        let expert = read_one(f, |name, _, _| {
            Ok::<_, String>(if name == "nv-one.weight" {
                Take::Part(2, 0..1)
            } else {
                Take::Skip
            })
        })
        .unwrap();
        assert_eq!(expert, one);
        let (mut same, mut refused) = (0, Vec::new());
        for (name, rank_dims) in cases {
            let whole = read_one(g, rule(name, 0, 0, 1)).unwrap();
            assert_eq!(
                read_one(f, rule(name, 0, 0, 1)).unwrap(),
                whole,
                "{name} whole"
            );
            for n in [2, 4] {
                for dim in 0..rank_dims {
                    let mut parts = Vec::new();
                    for r in 0..n {
                        let (want, got) = (
                            read_one(g, rule(name, dim, r, n)),
                            read_one(f, rule(name, dim, r, n)),
                        );
                        match (want, got) {
                            (Ok(want), Ok(got)) => {
                                assert_eq!(got, want, "{name} dim {dim} rank {r}/{n}");
                                parts.push(got.3);
                                same += 1;
                            }
                            (Err(_), Err(e)) => refused.push((name, dim, n, "both", e)),
                            (Ok(_), Err(e)) => refused.push((name, dim, n, "folder", e)),
                            (Err(e), Ok(_)) => {
                                panic!("{name} dim {dim} {r}/{n}: the file refused {e}")
                            }
                        }
                    }
                    if parts.len() == n && dim + 1 == whole.2.len() {
                        assert_eq!(parts.concat(), whole.3, "{name} dim {dim} over {n}");
                    }
                }
            }
        }
        refused.sort();
        refused.dedup_by(|a, b| (a.0, a.1, a.2, a.3) == (b.0, b.1, b.2, b.3));
        for r in &refused {
            println!("REFUSED {} dim {} N={} by {}: {}", r.0, r.1, r.2, r.3, r.4);
        }
        let by: Vec<_> = refused.iter().map(|r| (r.0, r.1, r.2, r.3)).collect();
        assert_eq!(
            by,
            [
                ("cs.weight", 2, 4, "both"),
                ("fb.weight", 0, 4, "both"),
                ("fb.weight", 1, 4, "folder"),
            ]
        );
        let fb = &refused.iter().find(|r| r.3 == "folder").unwrap().4;
        assert!(
            fb.contains("fb.s") && fb.contains("128-value scale blocks"),
            "{fb}"
        );
        let e = f.plan(rule("fb.weight", 1, 1, 4)).unwrap_err().to_string();
        assert!(
            e.contains("fb.s") && e.contains("128-value scale blocks"),
            "{e}"
        );
        println!("SAME {same} rank parts");
        for (name, _) in cases {
            let before = read_one(f, rule(name, 0, 0, 1)).unwrap();
            for p in f.plan(rule(name, 0, 0, 1)).unwrap() {
                f.release(&p).unwrap();
            }
            let after = read_one(f, rule(name, 0, 0, 1)).unwrap();
            assert_eq!(after, before, "{name} read again after its pages' release");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Before any value is read, a source the plan does not place, a refusal of the rule and a
    /// part off its card type's blocks are refused by name.
    #[test]
    fn entry_refusals_name_the_tensor_before_a_read() {
        let dir = temp_dir("refuse");
        let c = checkpoint(&dir);
        let f = &c;
        let e = f
            .plan(|t, _, _| {
                if t == "bf.weight" {
                    Err("no rule")
                } else {
                    Ok(Take::Skip)
                }
            })
            .unwrap_err()
            .to_string();
        assert_eq!(e, "bf.weight: no rule");
        let e = f.plan(|t, _, _| {
            Ok::<_, String>(if t == "nv.weight" {
                Take::Part(0, 0..32)
            } else {
                Take::Skip
            })
        });
        assert!(e.unwrap_err().to_string().contains("nv.weight"));
        let made = BTreeMap::from([(
            "x.weight".to_string(),
            Made {
                sources: vec!["absent.w".into()],
                role: Use::Matrix(Plain::F32),
                dims: vec![4],
                step: None,
            },
        )]);
        let e = Checkpoint::new(Folder::open(&dir).unwrap(), Plan::default(), made)
            .err()
            .unwrap();
        assert_eq!(
            e.to_string(),
            "x.weight: its source absent.w is not in the checkpoint"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
