//! A Hugging Face checkpoint folder as a load reads it ([`Checkpoint`], a `model::part::Source`):
//! each tensor of the program whole or the part a rank holds, by the load's rule
//! ([`model::part::Take`]). The map from the checkpoint's tensors to the program's is the
//! model's ([`Made`]); the checkpoint knows no model and no rank count. Every part is planned from
//! headers alone before any value is read.
//!
//! A folder's tensor is normalized into its card type (`hf::modelopt`), each weight and block
//! scale cut first, so a part reads only its own stored values: a stack's experts, along its
//! experts or by a list, are the experts' own sources normalized in the part's order. A tensor
//! the model moves into its layout after normalizing is made whole, moved, and then cut.

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::path::PathBuf;

use engram::EngramError;
use engram::direct::{DIRECT_ALIGN, DirectFile, aligned_span};
use gguf::{GgmlType, TensorInfo};
use hf::modelopt::{self, Normal, Plain, Stack, Use};
use hf::quant_config::{Placed, Plan};
use model::part::{self, At, Part, Planned, Source, Take};
use model::placement::ExpertList;
use model::placement::tp::TpError;
use safetensors::{Dtype, Folder, Raw};

use crate::folder::Constant;
use crate::folder::scales::Decoded;

/// The most stored bytes one direct read of a stack's sources holds, unless a single source is
/// larger ([`Checkpoint::read_stack`]).
const STACK_READ_MAX: u64 = 32 << 20;

/// The checkpoint tensors a tensor is made of, each with its place in the plan, in expert order.
type Sources<'s> = Vec<(&'s str, &'s Placed)>;

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
    #[error("{0}")]
    Release(#[from] model::placement::PlacementError),
    #[error(transparent)]
    Tensors(#[from] safetensors::Error),
    #[error(transparent)]
    Direct(#[from] engram::EngramError),
    /// A direct-read helper could not be placed.
    #[error(transparent)]
    Helper(#[from] threads::helper::HelperError),
    /// The scale store could not take a folder's block scales read back from their compressed
    /// form; the reason names its file.
    #[error("{0}")]
    Scales(String),
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

/// A checkpoint folder as a load reads it: its tensors, its quantization plan, the tensors the
/// model's map makes of them, and the ones it gives of its own, by the program's names.
pub struct Checkpoint {
    folder: Folder,
    plan: Plan,
    made: BTreeMap<String, Made>,
    /// The tensors no folder tensor holds ([`Constant`]), made whole and cut as a moved tensor is.
    constants: BTreeMap<String, Constant>,
    /// The block scales a located header reads from the scale store, decoded on the first
    /// [`Source::make_in_place`].
    scales: Option<Decoded>,
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
        Ok(Checkpoint {
            folder,
            plan,
            made,
            constants: BTreeMap::new(),
            scales: None,
        })
    }

    /// This checkpoint with the tensors `constants` of the program that no folder tensor holds.
    #[must_use]
    pub fn with_constants(mut self, constants: BTreeMap<String, Constant>) -> Checkpoint {
        self.constants = constants;
        self
    }

    /// This checkpoint with the scale store's file its located header reads the block scales
    /// of its compressed stacks from, once [`Source::make_in_place`] has written them.
    #[must_use]
    pub(crate) fn with_scales(mut self, scales: Option<Decoded>) -> Checkpoint {
        self.scales = scales;
        self
    }

    /// The tensors the map makes, by name.
    #[cfg(test)]
    pub(crate) fn made(&self) -> &BTreeMap<String, Made> {
        &self.made
    }

    /// The tensors no folder tensor holds, by name.
    #[cfg(test)]
    pub(crate) fn constants(&self) -> &BTreeMap<String, Constant> {
        &self.constants
    }

    fn placed<'s>(&'s self, m: &'s Made) -> Sources<'s> {
        // `new` held every source placed.
        m.sources
            .iter()
            .filter_map(|s| Some((s.as_str(), self.plan.placed.get(s)?)))
            .collect()
    }

    /// The sources of the experts `list` names of the stack `name`, made as `m`, in the list's
    /// order; an expert past its sources is refused by name.
    fn listed<'s>(
        &'s self,
        name: &str,
        m: &'s Made,
        list: &ExpertList,
    ) -> Result<Sources<'s>, Error> {
        let placed = self.placed(m);
        list.ids()
            .iter()
            .map(|&e| {
                placed.get(e as usize).copied().ok_or_else(|| {
                    refuse(
                        name,
                        format!("expert {e} is past the stack's {} sources", placed.len()),
                    )
                })
            })
            .collect()
    }

    /// The sources part `p` of `name`, made as `m`, reads whole: every source of a whole tensor,
    /// a stack's experts along its experts or by a list; `None` for a part that cuts its sources'
    /// rows and for a tensor the model moves after normalizing.
    fn whole_sources<'s>(
        &'s self,
        name: &str,
        m: &'s Made,
        p: &Planned,
    ) -> Result<Option<Sources<'s>>, Error> {
        let placed = self.placed(m);
        match (&p.at, &m.step) {
            (None, None) => Ok(Some(placed)),
            (Some(At::Experts(list)), None) => self.listed(name, m, list).map(Some),
            (Some(At::Along(2, r)), None)
                if m.dims.len() == 3 && placed.len() as u64 == m.dims[2] =>
            {
                let r = usize::try_from(r.start)
                    .ok()
                    .zip(usize::try_from(r.end).ok());
                r.and_then(|(a, b)| placed.get(a..b))
                    .map(|s| Some(s.to_vec()))
                    .ok_or_else(|| refuse(name, format!("experts {:?} of its sources", p.at)))
            }
            _ => Ok(None),
        }
    }

    /// The sources part `p` of `name`, made as `m`, reads its values from: those it reads whole
    /// ([`Checkpoint::whole_sources`]), every source for any other part and for a tensor the
    /// model moves after normalizing, which is made whole.
    fn read_from<'s>(&'s self, name: &str, m: &'s Made, p: &Planned) -> Result<Sources<'s>, Error> {
        Ok(self
            .whole_sources(name, m, p)?
            .unwrap_or_else(|| self.placed(m)))
    }

    /// The sources part `p` of `name`, made as `m`, streams in bounded direct reads
    /// ([`Checkpoint::read_stack`]), with the stack they are written into: the whole tensor or
    /// whole experts of its stack, when its type writes them one at a time ([`Stack::of`]);
    /// `None` for any other tensor and for a part that cuts rows.
    fn streamed<'s>(
        &'s self,
        name: &str,
        m: &'s Made,
        p: &Planned,
    ) -> Result<Option<(Sources<'s>, Stack)>, Error> {
        let Some(sources) = self.whole_sources(name, m, p)? else {
            return Ok(None);
        };
        let format = modelopt::format_of(&sources)?;
        Ok(Stack::of(format, m.role, sources.len()).map(|stack| (sources, stack)))
    }

    /// Bound each direct read to a run of sources of at most `limit` stored bytes, allowing a
    /// single larger source whole; each run with the bytes its read holds, every stored tensor
    /// with its two alignment blocks.
    fn batches(
        &self,
        placed: &[(&str, &Placed)],
        limit: u64,
    ) -> Result<Vec<(Range<usize>, u64)>, Error> {
        let mut runs = Vec::new();
        let (mut first, mut bytes) = (0, 0u64);
        for (i, source) in placed.iter().enumerate() {
            let mut held = 0u64;
            for n in stored(std::slice::from_ref(source)) {
                let (_, range) = self.folder.file_range(n)?;
                held = held
                    .checked_add(range.end - range.start)
                    .and_then(|n| n.checked_add((2 * DIRECT_ALIGN) as u64))
                    .ok_or_else(|| refuse(source.0, "stored read bytes pass u64"))?;
            }
            if i > first && bytes.saturating_add(held) > limit {
                runs.push((first..i, bytes));
                first = i;
                bytes = 0;
            }
            bytes = bytes
                .checked_add(held)
                .ok_or_else(|| refuse(source.0, "stored read bytes pass u64"))?;
        }
        if first < placed.len() {
            runs.push((first..placed.len(), bytes));
        }
        Ok(runs)
    }

    /// Read one bounded input batch at a time into the stack's single output allocation.
    fn read_stack(
        &self,
        placed: &[(&str, &Placed)],
        mut stack: Stack,
        limit: u64,
    ) -> Result<Normal, Error> {
        for (run, _) in self.batches(placed, limit)? {
            let sources = &placed[run];
            let read = Wholes::read(&self.folder, stored(sources))?;
            for &source in sources {
                stack.push(|n| self.raw(read.as_ref(), n), source)?;
            }
        }
        Ok(stack.finish()?)
    }

    /// Prefer the direct batch's bytes, using the mapping when direct reads are unsupported.
    fn raw<'a>(
        &'a self,
        read: Option<&'a Wholes<'_>>,
        name: &str,
    ) -> Result<safetensors::Raw<'a>, safetensors::Error> {
        match read.and_then(|w| w.raw(name)) {
            Some(raw) => Ok(raw),
            None => self.folder.raw(name),
        }
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
            let (info, _) = part::part_at(&whole, at.as_ref())?;
            // A moved tensor, or a whole row table, is checked whole as it is made in read; a
            // stack's listed experts each as the cut along its experts that reads it alone.
            if m.step.is_none() && (at.is_some() || m.role != Use::Rows) {
                let cuts = match &at {
                    None => vec![(0, 0..m.dims[0])],
                    Some(At::Along(dim, r)) => vec![(*dim, r.clone())],
                    Some(At::Experts(list)) => list
                        .ids()
                        .iter()
                        .map(|&e| (2, u64::from(e)..u64::from(e) + 1))
                        .collect(),
                };
                for cut in cuts {
                    modelopt::check_part(
                        |n| self.folder.raw(n),
                        &self.placed(m),
                        m.role,
                        &m.dims,
                        cut,
                    )?;
                }
            }
            out.push(Planned { at, info });
        }
        for (name, c) in &self.constants {
            let whole = whole(name, GgmlType::F32, &c.dims)?;
            let Some(at) = part::take(&rule, &whole)? else {
                continue;
            };
            let (info, _) = part::part_at(&whole, at.as_ref())?;
            out.push(Planned { at, info });
        }
        Ok(out)
    }

    fn read_bytes(&self, p: &Planned) -> Result<u64, Error> {
        let name = &p.info.name;
        if let Some(c) = self.constants.get(name) {
            return whole(name, GgmlType::F32, &c.dims)?
                .nbytes
                .checked_add(p.info.nbytes)
                .ok_or_else(|| refuse(name, "normalized read bytes pass u64"));
        }
        let m = self
            .made
            .get(name)
            .ok_or_else(|| refuse(name, "is not one the map makes"))?;
        let placed = self.placed(m);
        let ty = self.card_type(name, m)?;
        let whole = whole(name, ty, &m.dims)?;
        let cut = p.at.is_some() && m.step.is_none();
        let format = modelopt::format_of(&placed)?;
        let compressed = matches!(
            format,
            hf::quant_config::Format::Nvfp4Csf | hf::quant_config::Format::Nvfp4Csf32
        );
        let plain = matches!(
            format,
            hf::quant_config::Format::Bf16
                | hf::quant_config::Format::F16
                | hf::quant_config::Format::F32
        );
        // One BF16 matrix read whole is copied as stored (`modelopt::normalize`): no values widen.
        let widened = plain
            && !(matches!(format, hf::quant_config::Format::Bf16)
                && m.role == Use::Matrix(Plain::Bf16)
                && m.sources.len() == 1
                && !cut
                && m.step.is_none());
        let len = |n: &str| -> Result<u64, Error> {
            let (_, range) = self.folder.file_range(n)?;
            range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| refuse(name, "stored read bytes pass u64"))
        };
        let sum = |a: u64, b: u64| {
            a.checked_add(b)
                .ok_or_else(|| refuse(name, "read bytes pass u64"))
        };
        // A direct read's buffer is aligned within one more block, and opening a shard for direct
        // reads probes it with two ([`DirectFile::open`]), one shard at a time.
        let direct = (3 * DIRECT_ALIGN) as u64;
        if let Some((sources, stack)) = self.streamed(name, m, p)? {
            // One bounded batch of stored inputs, the current source's scratch (decoded or
            // widened scales), and the part's single output allocation coexist.
            let mut peak = 0;
            for (run, held) in self.batches(&sources, STACK_READ_MAX)? {
                let mut scratch = 0;
                for &source in &sources[run] {
                    scratch = scratch.max(stack.scratch(|n| self.folder.raw(n), source)?);
                }
                peak = peak.max(sum(sum(direct, held)?, scratch)?);
            }
            let scales = (sources.len() as u64)
                .checked_mul(8)
                .ok_or_else(|| refuse(name, "read bytes pass u64"))?;
            return sum(peak, sum(p.info.nbytes, scales)?);
        }

        // Direct inputs, or growing cut buffers; decoded companion words and widened plain
        // values coexist with them. A step may shrink the tensor, so its pre-step output is
        // bounded from the stored bytes too. Output capacity can grow to twice its length.
        let mut bytes = direct;
        for n in stored(&placed) {
            let companion = !m.sources.iter().any(|source| source == n);
            let factor = if cut { 2u64 } else { 1 }
                + u64::from(compressed) * 2
                + u64::from(m.step.is_some()) * 3
                + u64::from(widened || companion) * 4;
            bytes = len(n)?
                .checked_mul(factor)
                .and_then(|n| n.checked_add((2 * DIRECT_ALIGN) as u64))
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(|| refuse(name, "stored read bytes pass u64"))?;
        }
        let normal = if cut { p.info.nbytes } else { whole.nbytes };
        let moved = if m.step.is_some() { whole.nbytes } else { 0 };
        normal
            .checked_mul(2)
            .and_then(|n| n.checked_add(moved))
            .and_then(|n| n.checked_add(p.info.nbytes))
            .and_then(|n| n.checked_add(if compressed { whole.nbytes } else { 0 }))
            .and_then(|n| n.checked_add((m.sources.len() as u64).checked_mul(8)?))
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| refuse(name, "normalized read bytes pass u64"))
    }

    /// A tensor its format refuses, or that does not normalize into the planned type and bytes,
    /// is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Error> {
        let name = p.info.name.as_str();
        if let Some(c) = self.constants.get(name) {
            let whole = whole(name, GgmlType::F32, &c.dims)?;
            let all = c.bytes();
            let bytes: Vec<u8> = match &p.at {
                None => all,
                Some(at) => {
                    let (_, runs) = part::part_at(&whole, Some(at))?;
                    runs.into_iter().flat_map(|r| &all[r]).copied().collect()
                }
            };
            if bytes.len() as u64 != p.info.nbytes {
                return Err(refuse(
                    name,
                    format!(
                        "made {} bytes, not the planned {}",
                        bytes.len(),
                        p.info.nbytes
                    ),
                ));
            }
            return Ok(Part::made(p.info.clone(), bytes, None));
        }
        let m = self
            .made
            .get(name)
            .ok_or_else(|| refuse(name, "is not one the map makes"))?;
        let placed = self.placed(m);
        let ty = self.card_type(name, m)?;
        // A whole tensor's stored bytes are read into memory of the read's own ([`Wholes::read`]),
        // a stack's in bounded batches; a part that cuts rows reads only its rows through the
        // mapping.
        let streamed = self.streamed(name, m, p)?;
        let read = match streamed.is_none() && (p.at.is_none() || m.step.is_some()) {
            true => Wholes::read(&self.folder, stored(&placed))?,
            false => None,
        };
        let get = |n: &str| self.raw(read.as_ref(), n);
        let n = if let Some((sources, stack)) = streamed {
            self.read_stack(&sources, stack, STACK_READ_MAX)?
        } else {
            match (&p.at, &m.step) {
                (Some(At::Along(dim, r)), None) => {
                    modelopt::normalize_part(get, &placed, m.role, &m.dims, (*dim, r.clone()))?
                }
                (Some(At::Experts(list)), None) => {
                    modelopt::normalize(get, &self.listed(name, m, list)?, m.role)?
                }
                _ => {
                    let mut n = modelopt::normalize(get, &placed, m.role)?;
                    if let Some(step) = &m.step {
                        step(&mut n).map_err(Error::Step)?;
                    }
                    n
                }
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
        let (bytes, scale) = match (&p.at, &m.step) {
            (Some(at), Some(_)) => {
                let (_, runs) = part::part_at(&whole, Some(at))?;
                let bytes = runs
                    .into_iter()
                    .flat_map(|r| &n.bytes[r])
                    .copied()
                    .collect();
                let scale = if ty.reads_scale() {
                    part::scales_at(name, &whole.dims, n.scale, Some(at))?
                } else {
                    n.scale
                };
                (bytes, scale)
            }
            _ => (n.bytes, n.scale),
        };
        Ok(Part::made(
            p.info.clone(),
            bytes,
            ty.reads_scale().then_some(scale),
        ))
    }

    /// Every stored tensor the part was read from, its values and its companions (scales, the
    /// packed shape), each the whole pages inside its bytes: a read normalizes into bytes of its
    /// own ([`Part::made`]), so no later reader of the part takes them from the file. A part of
    /// a stack's experts releases those experts' alone: the rest stay where a host or another
    /// part reads them.
    fn release(&self, p: &Planned) -> Result<(), Error> {
        let name = p.info.name.as_str();
        if self.constants.contains_key(name) {
            return Ok(());
        }
        let m = self
            .made
            .get(name)
            .ok_or_else(|| refuse(name, "is not one the map makes"))?;
        for (source, placed) in self.read_from(name, m, p)? {
            for stored in
                std::iter::once(source).chain(placed.companions.iter().map(String::as_str))
            {
                let (path, map, at) = self.folder.mapped(stored)?;
                // SAFETY: `Folder::mapped` gives the shard file's own read-only mapping whole.
                unsafe { model::placement::host_lock::release_file_pages(&path, map, at) }?;
            }
        }
        Ok(())
    }

    /// The block scales of the stacks stored with them compressed, decoded into the scale
    /// store's file once ([`Decoded::fill`]).
    fn make_in_place(&self) -> Result<(), Error> {
        self.scales
            .as_ref()
            .map_or(Ok(()), |s| s.fill(&self.folder))
    }
}

/// Stored tensors whose aligned spans touch in one shard, read as one ([`Wholes::read`]).
struct Extent<'n> {
    path: PathBuf,
    /// The aligned file offset and length.
    start: u64,
    len: usize,
    /// Each tensor's name and file bytes.
    held: Vec<(&'n str, Range<u64>)>,
}

impl Extent<'_> {
    fn end(&self) -> u64 {
        self.start + self.len as u64
    }
}

/// The stored tensors a part reads whole, in one buffer of their own ([`Wholes::read`]).
struct Wholes<'n> {
    buf: Vec<u8>,
    /// Each tensor's dtype, shape and bytes in `buf`.
    at: HashMap<&'n str, (Dtype, Vec<usize>, Range<usize>)>,
}

impl<'n> Wholes<'n> {
    /// Stored tensors `names` read whole from their shards past the page cache (engram's direct
    /// reader) into one new buffer advised onto huge pages, so the reads take the drive's rate,
    /// fault once each 2 MiB and leave the page cache as it was. A routed stack is thousands of
    /// stored tensors, so each shard is opened, and its direct reads proved, once, and tensors
    /// whose aligned spans touch in a shard are read as one extent. A stack's experts are small
    /// extents apart in their shards, and one synchronous read of each in turn keeps too little
    /// of the drive busy, so a shard's extents are read over [`READ_LANES`] threads at once
    /// ([`read_extents`]). The extents run shard by shard, so a read holds one shard open at a
    /// time, however many shards the part spans. `None` only when a shard's filesystem refuses
    /// direct reads as invalid or unsupported ([`DirectFile::open`]); other I/O errors propagate,
    /// the first extent's first.
    fn read(folder: &Folder, names: impl Iterator<Item = &'n str>) -> Result<Option<Self>, Error> {
        let mut spans = names
            .map(|name| Ok((folder.file_range(name)?, name)))
            .collect::<Result<Vec<_>, Error>>()?;
        spans.sort_by(|((p, r), _), ((q, s), _)| (p, r.start).cmp(&(q, s.start)));
        let mut extents: Vec<Extent<'n>> = Vec::new();
        for ((path, range), name) in spans {
            let (start, span, _) = aligned_span(range.start, range.end - range.start);
            let end = start + span as u64;
            match extents.last_mut() {
                Some(e) if e.path == path && start <= e.end() => {
                    e.len = (end.max(e.end()) - e.start) as usize;
                    e.held.push((name, range));
                }
                _ => extents.push(Extent {
                    path,
                    start,
                    len: span,
                    held: vec![(name, range)],
                }),
            }
        }
        let mut size = 0usize;
        for e in &extents {
            size = size
                .checked_add(e.len)
                .ok_or_else(|| refuse(e.held[0].0, "the part's direct read bytes pass usize"))?;
        }
        let mut buf = vec![0u8; size + DIRECT_ALIGN];
        os::advise_huge_within(&mut buf);
        let first = buf.as_ptr().align_offset(DIRECT_ALIGN);
        let mut got = Vec::with_capacity(extents.len());
        let mut rest = &mut buf[first..];
        for shard in extents.chunk_by(|a, b| a.path == b.path) {
            let file = match DirectFile::open(&shard[0].path) {
                Ok(file) => file,
                Err(
                    EngramError::OpenDirect { source, .. }
                    | EngramError::DirectProbe { source, .. },
                ) if matches!(
                    source.kind(),
                    std::io::ErrorKind::InvalidInput | std::io::ErrorKind::Unsupported
                ) =>
                {
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            };
            let mut windows = Vec::with_capacity(shard.len());
            for e in shard {
                let (window, after) = std::mem::take(&mut rest).split_at_mut(e.len);
                windows.push((e.start, window));
                rest = after;
            }
            got.extend(read_extents(&file, windows)?);
        }
        let mut slot = first;
        let mut at = HashMap::new();
        for (
            Extent {
                path,
                start,
                len,
                held,
            },
            got,
        ) in extents.into_iter().zip(got)
        {
            for (name, range) in held {
                if (got as u64) < range.end - start {
                    return Err(refuse(
                        name,
                        format!("{} ends inside its bytes", path.display()),
                    ));
                }
                let raw = folder.raw(name)?;
                let from = slot + (range.start - start) as usize;
                at.insert(
                    name,
                    (raw.dtype, raw.shape.to_vec(), from..from + raw.bytes.len()),
                );
            }
            slot += len;
        }
        Ok(Some(Wholes { buf, at }))
    }

    /// Stored tensor `name` as read, `None` for one this read does not hold.
    fn raw(&self, name: &str) -> Option<Raw<'_>> {
        self.at.get(name).map(|(dtype, shape, at)| Raw {
            dtype: *dtype,
            shape,
            bytes: &self.buf[at.clone()],
        })
    }
}

/// The direct reads [`Wholes::read`] keeps in flight in a shard at once, its own thread among
/// them.
const READ_LANES: usize = 8;

/// Each window filled from `file` at its offset, the bytes placed per window in order; the
/// windows are dealt over up to [`READ_LANES`] threads, each reading its own in turn. The first
/// window's error in order is returned.
fn read_extents(file: &DirectFile, windows: Vec<(u64, &mut [u8])>) -> Result<Vec<usize>, Error> {
    let lanes = READ_LANES.min(windows.len()).max(1);
    let mut dealt: Vec<Vec<(usize, u64, &mut [u8])>> = (0..lanes).map(|_| Vec::new()).collect();
    for (i, (at, window)) in windows.into_iter().enumerate() {
        dealt[i % lanes].push((i, at, window));
    }
    let read = |lane: Vec<(usize, u64, &mut [u8])>| {
        lane.into_iter()
            .map(|(i, at, window)| (i, file.read_span(window, at)))
            .collect::<Vec<_>>()
    };
    let mut done: Vec<Vec<_>> = (0..lanes).map(|_| Vec::new()).collect();
    std::thread::scope(|scope| {
        let mut dealt = dealt.into_iter();
        let own = dealt.next().unwrap_or_default();
        let (own_done, others_done) = done.split_first_mut().expect("at least one read lane");
        let mut others = Vec::new();
        for (lane, result) in dealt.zip(others_done) {
            let handle = std::thread::Builder::new()
                .name("part-read".to_owned())
                .spawn_scoped(scope, move || {
                    threads::helper::float_caller()?;
                    *result = read(lane);
                    Ok::<_, threads::helper::HelperError>(())
                })
                .map_err(threads::helper::HelperError::Spawn)?;
            others.push(handle);
        }
        *own_done = read(own);
        for lane in others {
            lane.join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e))?;
        }
        Ok::<_, Error>(())
    })?;
    let mut done: Vec<_> = done.into_iter().flatten().collect();
    done.sort_by_key(|(i, _)| *i);
    done.into_iter()
        .map(|(_, got)| got.map_err(Error::from))
        .collect()
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
    use std::alloc::{GlobalAlloc, System};
    use std::cell::Cell;

    use crate::test_support::temp_dir;

    /// One stored tensor: name, dtype, shape, bytes.
    type Stored = (String, &'static str, Vec<usize>, Vec<u8>);

    /// `n` bytes, each `f(i)`.
    fn bytes(n: usize, f: impl Fn(usize) -> u8) -> Vec<u8> {
        (0..n).map(f).collect()
    }

    /// The checkpoint every case reads: an NVFP4 linear, routed stacks of two NVFP4 experts with
    /// `csf_32x2_v1` scales and with `csf` scales (the latter's decoded scales larger than the
    /// alignment of its stored tensors' direct reads), MXFP8, FP8 with one scale a 128 by 128
    /// block, BF16, an F32 vector, and a BF16 matrix the map transposes after normalizing. Values
    /// are finite and every scale byte a scale.
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
        for e in 0..2 {
            let p = format!("cf.{e}");
            t.push((
                format!("{p}.w"),
                "U8",
                vec![1024, 512],
                bytes(1024 * 512, |i| (i * 13 + e) as u8),
            ));
            t.push((
                format!("{p}.s.nvfp4_csf_fixed"),
                "U8",
                vec![64, 528],
                vec![0x31; 64 * 528],
            ));
            t.push((
                format!("{p}.s.nvfp4_csf_exceptions"),
                "U32",
                vec![0],
                vec![],
            ));
            t.push((format!("{p}.t"), "F32", vec![], f32s(&[0.5])));
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
            let p = format!("cf.{e}");
            let companions =
                ["s.nvfp4_csf_fixed", "s.nvfp4_csf_exceptions", "t"].map(|c| format!("{p}.{c}"));
            let companions: Vec<&str> = companions.iter().map(String::as_str).collect();
            placed.insert(format!("{p}.w"), place(Format::Nvfp4Csf, &companions));
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
                "cf.weight".into(),
                made(&["cf.0.w", "cf.1.w"], matrix, &[1024, 1024, 2], None),
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

    /// A stack `ex.weight` of `experts` experts of `rows` rows in `format`, each expert in a shard
    /// of its own, read as F32 where the format is unquantized, and the 2-D `lone.weight` in
    /// `model.safetensors`. A compressed-tensors NVFP4 expert is codes, block scales, a divisor
    /// and an input scale (as MiniMax-M3.1 stores them) of 128 values a row; an FP8 expert of
    /// 128 by 128 scale blocks is 128 values a row and its block scales; a BF16 expert is 64
    /// values a row.
    fn stack(dir: &std::path::Path, experts: usize, rows: usize, format: Format) -> Checkpoint {
        let f32s = |v: f32| v.to_le_bytes().to_vec();
        let mut shards: Vec<(String, Vec<Stored>)> = vec![(
            "model.safetensors".into(),
            vec![
                (
                    "lone.w".into(),
                    "U8",
                    vec![8, 64],
                    bytes(512, |i| (i * 5) as u8),
                ),
                (
                    "lone.s".into(),
                    "F8_E4M3",
                    vec![8, 8],
                    bytes(64, |i| 0x38 + (i % 5) as u8),
                ),
                ("lone.g".into(), "F32", vec![], f32s(2.0)),
                ("lone.i".into(), "F32", vec![], f32s(1.0)),
            ],
        )];
        let k = if format == Format::Bf16 { 64 } else { 128 };
        for e in 0..experts {
            let t: Vec<Stored> = match format {
                Format::Nvfp4Ct => vec![
                    (
                        format!("ex.{e}.w"),
                        "U8",
                        vec![rows, 64],
                        bytes(rows * 64, |i| (i * 13 + 41 * e) as u8),
                    ),
                    (
                        format!("ex.{e}.s"),
                        "F8_E4M3",
                        vec![rows, 8],
                        bytes(rows * 8, |i| 0x30 + ((i + 3 * e) % 11) as u8),
                    ),
                    (
                        format!("ex.{e}.g"),
                        "F32",
                        vec![],
                        f32s(4.0 * (e + 1) as f32),
                    ),
                    (format!("ex.{e}.i"), "F32", vec![], f32s(1.0)),
                ],
                Format::Fp8Block { .. } => vec![
                    (
                        format!("ex.{e}.w"),
                        "F8_E4M3",
                        vec![rows, 128],
                        bytes(rows * 128, |i| ((i * 7 + e) % 0x7e) as u8),
                    ),
                    (
                        format!("ex.{e}.s"),
                        "F32",
                        vec![rows.div_ceil(128), 1],
                        (0..rows.div_ceil(128))
                            .flat_map(|b| f32s(0.25 * (1 + (b + e) % 4) as f32))
                            .collect(),
                    ),
                ],
                _ => vec![(
                    format!("ex.{e}.w"),
                    "BF16",
                    vec![rows, 64],
                    bytes(
                        rows * 128,
                        |i| if i % 2 == 0 { (i + e) as u8 } else { 0x3f },
                    ),
                )],
            };
            shards.push((format!("ex-{e}.safetensors"), t));
        }
        let mut map = BTreeMap::new();
        for (file, t) in &shards {
            let entries: Vec<_> = t
                .iter()
                .map(|(n, d, s, b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
                .collect();
            std::fs::write(dir.join(file), safetensors::write(&entries)).unwrap();
            map.extend(t.iter().map(|(n, ..)| (n.as_str(), file.as_str())));
        }
        std::fs::write(
            dir.join(Folder::INDEX),
            serde_json::json!({ "weight_map": map }).to_string(),
        )
        .unwrap();
        let ct = |stem: &str| Placed {
            format: Format::Nvfp4Ct,
            companions: ["s", "g", "i"]
                .iter()
                .map(|c| format!("{stem}.{c}"))
                .collect(),
        };
        let mut placed = BTreeMap::from([("lone.w".to_string(), ct("lone"))]);
        for e in 0..experts {
            let stem = format!("ex.{e}");
            let place = match format {
                Format::Nvfp4Ct => ct(&stem),
                Format::Fp8Block { .. } => Placed {
                    format,
                    companions: vec![format!("{stem}.s")],
                },
                _ => Placed {
                    format,
                    companions: vec![],
                },
            };
            placed.insert(format!("{stem}.w"), place);
        }
        let plan = Plan {
            placed,
            ..Plan::default()
        };
        let made = |sources: Vec<String>, dims: &[u64]| Made {
            sources,
            role: Use::Matrix(Plain::F32),
            dims: dims.to_vec(),
            step: None,
        };
        let made = BTreeMap::from([
            (
                "ex.weight".to_string(),
                made(
                    (0..experts).map(|e| format!("ex.{e}.w")).collect(),
                    &[k, rows as u64, experts as u64],
                ),
            ),
            ("lone.weight".into(), made(vec!["lone.w".into()], &[128, 8])),
        ]);
        Checkpoint::new(Folder::open(dir).unwrap(), plan, made).unwrap()
    }

    /// The experts a list names of a stack are read by the folder and by its GGUF as the same
    /// part: each expert's bytes and tensor scale, in the list's order, the experts the list
    /// leaves out unread; one expert is the stack's cut along its experts. A list on a tensor
    /// that is no stack, and one naming an expert past the stack, are refused by name by both
    /// sources before a value is read. Releasing the listed part releases the listed experts'
    /// sources alone. A listed part reads its experts' stored bytes past the page cache, giving
    /// the bytes and scales the mapping gives and holding no more than its read bytes count.
    /// Mutants (`split-mutants/`): the folder's list read in reverse, the GGUF's listed scales
    /// left whole, the release of a listed part over every source. FAIL-first: a listed part read
    /// through the mapping holds its read bytes' count less its stored bytes.
    #[test]
    fn a_stack_s_listed_experts_are_the_same_part_from_the_folder_and_its_gguf() {
        let dir = temp_dir("listed");
        let f = stack(&dir, 4, 32, Format::Nvfp4Ct);
        let g = gguf_of(&f, &dir.join("model.gguf"));
        let list = |ids: &[u32], n: u64| ExpertList::new(ids.to_vec(), n).unwrap();
        let listed = |name: &'static str, l: ExpertList| {
            move |t: &str, _: GgmlType, _: &[u64]| {
                Ok::<_, String>(if t == name {
                    Take::Experts(l.clone())
                } else {
                    Take::Skip
                })
            }
        };
        let one = |e: u64| read_one(&f, rule_part("ex.weight", 2, e..e + 1)).unwrap();
        for ids in [&[0, 2, 3][..], &[1], &[0, 1, 2, 3], &[3]] {
            let got = read_one(&f, listed("ex.weight", list(ids, 4))).unwrap();
            assert_eq!(
                read_one(&g, listed("ex.weight", list(ids, 4))).unwrap(),
                got,
                "{ids:?}: the GGUF's part"
            );
            assert_eq!(got.2, [128, 32, ids.len() as u64], "{ids:?}");
            let each: Vec<Read> = ids.iter().map(|&e| one(u64::from(e))).collect();
            assert_eq!(
                got.3,
                each.iter().flat_map(|r| r.3.clone()).collect::<Vec<u8>>(),
                "{ids:?}: the experts' bytes in the list's order"
            );
            assert_eq!(
                got.4,
                Some(each.iter().flat_map(|r| r.4.clone().unwrap()).collect()),
                "{ids:?}: the experts' tensor scales in the list's order"
            );
        }
        let scales = read_one(&f, listed("ex.weight", list(&[1, 3], 4)))
            .unwrap()
            .4;
        assert_eq!(scales, Some(vec![1.0 / 8.0, 1.0 / 16.0]));
        for (name, l, says) in [
            ("lone.weight", list(&[0], 4), "not a stack of experts"),
            (
                "ex.weight",
                list(&[1, 4], 5),
                "expert 4 is past the stack's 4",
            ),
        ] {
            for (by, e) in [
                (
                    "folder",
                    f.plan(listed(name, l.clone())).err().map(|e| e.to_string()),
                ),
                (
                    "gguf",
                    g.plan(listed(name, l.clone())).err().map(|e| e.to_string()),
                ),
            ] {
                let e = e.unwrap_or_else(|| panic!("{by} planned {name} at {l:?}"));
                assert!(e.contains(name) && e.contains(says), "{by}: {e}");
            }
        }
        let [p] = &f.plan(listed("ex.weight", list(&[0, 2], 4))).unwrap()[..] else {
            panic!("one part planned")
        };
        let m = &f.made()["ex.weight"];
        let from = f.read_from("ex.weight", m, p).unwrap();
        let names: Vec<&str> = from.iter().map(|&(s, _)| s).collect();
        assert_eq!(names, ["ex.0.w", "ex.2.w"]);
        let mapped = modelopt::normalize(|n| f.folder.raw(n), &from, m.role).unwrap();
        let read_bytes = f.read_bytes(p).unwrap();
        let (held, got) = peak(|| {
            let part = f.read(p).unwrap();
            (part.spans().concat(), part.scale)
        });
        assert_eq!(
            got,
            (mapped.bytes, Some(mapped.scale)),
            "the mapping's part"
        );
        let (shard, _) = f.folder.file_range("ex.0.w").unwrap();
        if DirectFile::open(&shard).is_ok() {
            let align = (2 * stored(&from).count() as u64 + 3) * DIRECT_ALIGN as u64;
            assert!(
                read_bytes >= held && read_bytes - held <= align,
                "the listed part holds {held}, its read bytes count {read_bytes}"
            );
        }
        f.release(p).unwrap();
        let again = read_one(&f, listed("ex.weight", list(&[0, 2], 4))).unwrap();
        assert_eq!(
            again.3,
            [one(0).3, one(2).3].concat(),
            "read again after release"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A whole stack of 64 experts, one a shard, is read past the page cache with one shard open
    /// at a time: the test runs again in a process of its own under a soft open-file limit of 32,
    /// and that process reads the stack. Mutant: each visited shard held open until the part is
    /// read fails the read with too many open files.
    #[test]
    fn a_whole_stack_over_more_shards_than_open_files_reads() {
        const NAME: &str = "part::tests::a_whole_stack_over_more_shards_than_open_files_reads";
        // The limited process's own argument: a filter no test's name is.
        const LIMITED: &str = "a-process-under-the-open-file-limit";
        if !std::env::args().any(|a| a == LIMITED) {
            let out = std::process::Command::new("sh")
                .args([
                    "-c",
                    r#"ulimit -S -n 32 && exec "$0" --exact "$1" "$2" --test-threads=1"#,
                ])
                .arg(std::env::current_exe().unwrap())
                .args([NAME, LIMITED])
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let dir = temp_dir("many-shards");
        let c = stack(&dir, 64, 32, Format::Nvfp4Ct);
        let placed = c.placed(&c.made["ex.weight"]);
        if Wholes::read(&c.folder, stored(&placed)).unwrap().is_none() {
            eprintln!("direct reads unsupported; descriptor-limit check skipped");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Windows dealt over the read lanes come back in their own order, each holding its offset's
    /// bytes, with more windows than lanes and one past the file's end placing only what is there.
    #[test]
    fn extents_read_over_lanes_land_in_their_windows() {
        let dir = temp_dir("lanes");
        let path = dir.join("f.bin");
        let bytes: Vec<u8> = (0..64 * 1024u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = DirectFile::open_buffered(&path).unwrap();
        let n = 3 * READ_LANES + 1;
        let mut buf = vec![0u8; n * 1024];
        let mut windows = Vec::new();
        let mut rest = &mut buf[..];
        for i in 0..n {
            let (w, after) = std::mem::take(&mut rest).split_at_mut(1024);
            windows.push(((i * 2048) as u64, w));
            rest = after;
        }
        let last = 64 * 1024 - 100;
        windows[n - 1].0 = last as u64;
        let got = read_extents(&file, windows).unwrap();
        assert!(
            threads::helper::helpers()
                .iter()
                .all(|h| h.name != "part-read"),
            "short-lived part readers must not enter the permanent helper registry"
        );
        assert_eq!(got[..n - 1], vec![1024; n - 1][..]);
        assert_eq!(
            got[n - 1],
            100,
            "the window past the end places what is there"
        );
        for i in 0..n - 1 {
            assert_eq!(
                buf[i * 1024..][..1024],
                bytes[i * 2048..][..1024],
                "window {i}"
            );
        }
        assert_eq!(buf[(n - 1) * 1024..][..100], bytes[last..]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The part along `dim` at `at` of tensor `name`, every other skipped.
    fn rule_part(
        name: &'static str,
        dim: usize,
        at: std::ops::Range<u64>,
    ) -> impl Fn(&str, GgmlType, &[u64]) -> Result<Take, String> {
        move |t, _, _| {
            Ok(if t == name {
                Take::Part(dim, at.clone())
            } else {
                Take::Skip
            })
        }
    }

    thread_local! {
        // Const-initialized and destructor-free, so the allocator reads them without recursing.
        static HELD: Cell<i64> = const { Cell::new(0) };
        static PEAK: Cell<i64> = const { Cell::new(0) };
    }

    /// The global allocator of this crate's tests: [`System`], counting the
    /// bytes each thread holds ([`HELD`]) and the most it has held ([`PEAK`]).
    struct Counting;

    fn count(bytes: i64) {
        let held = HELD.get() + bytes;
        HELD.set(held);
        PEAK.set(PEAK.get().max(held));
    }

    // SAFETY: both methods forward to `System` with the caller's arguments unchanged; the trait's
    // reallocation and zeroing go through them.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: std::alloc::Layout) -> *mut u8 {
            count(l.size() as i64);
            // SAFETY: the caller meets `alloc`'s contract for `l`, and `System` receives `l`
            // unchanged.
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: std::alloc::Layout) {
            count(-(l.size() as i64));
            // SAFETY: `p` came from `alloc` above with layout `l`, so from `System`.
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static COUNTING: Counting = Counting;

    /// The most bytes the calling thread holds while `f` runs beyond what it held before, and
    /// `f`'s value.
    fn peak<T>(f: impl FnOnce() -> T) -> (u64, T) {
        let before = HELD.get();
        PEAK.set(before);
        let out = f();
        ((PEAK.get() - before) as u64, out)
    }

    /// Each part read whole holds at most its read bytes on the reading thread, counted by the
    /// allocator: stored inputs, decoded and widened scales, scratch and the result. An NVFP4
    /// matrix's read bytes pass that peak by at most two alignment blocks a stored tensor and the
    /// three of the direct buffer and its shard's probe, where its shard takes direct reads. A
    /// stack over the production read limit is read and counted in runs of two of its experts.
    ///
    /// FAIL-first: the `csf` stack's read bytes without its decoded scales, and the BF16 matrix's
    /// without its direct buffer's alignment and its shard's open probe, fall under their peaks;
    /// an NVFP4 stack's read bytes by the general bound (its weights thrice, its output four
    /// times) pass its peak by more than the alignment.
    #[test]
    fn read_peaks_cover_the_owned_inputs_and_results() {
        let dir = temp_dir("read-peak");
        let mut c = checkpoint(&dir);
        let shrinking = c.made.get_mut("mx.weight").unwrap();
        shrinking.dims[1] = 1;
        shrinking.step = Some(Box::new(|n| {
            let row = n.bytes.len() / n.dims[1] as usize;
            n.bytes.truncate(row);
            n.dims[1] = 1;
            Ok(())
        }));
        let (shard, _) = c.folder.file_range("cf.0.w").unwrap();
        let direct = DirectFile::open(&shard).is_ok();
        let planned = c.plan(|_, _, _| Ok::<_, String>(Take::Whole)).unwrap();
        for p in &planned {
            let name = &p.info.name;
            let read_bytes = c.read_bytes(p).unwrap();
            if name == "cf.weight" {
                let placed = c.placed(&c.made[name]);
                let input: u64 = stored(&placed)
                    .map(|n| {
                        c.folder.raw(n).unwrap().bytes.len() as u64 + (2 * DIRECT_ALIGN) as u64
                    })
                    .sum();
                let decoded = c.folder.raw(placed[0].0).unwrap().bytes.len() as u64 / 8;
                let words: u64 = placed[0]
                    .1
                    .companions
                    .iter()
                    .map(|n| {
                        let raw = c.folder.raw(n).unwrap();
                        if raw.dtype.size() > 1 {
                            (raw.bytes.len() / raw.dtype.size() * 4) as u64
                        } else {
                            0
                        }
                    })
                    .sum();
                let bound =
                    p.info.nbytes + input + decoded + words + (3 * DIRECT_ALIGN) as u64 + 16;
                assert!(
                    read_bytes <= bound,
                    "{name}: {read_bytes} > one expert's scale bound {bound}"
                );
                // A run holds its experts while their stored bytes fit the limit, a run past it
                // flushes before the next expert, and an expert alone over it is read whole.
                let cf_stack = || Stack::of(Format::Nvfp4Csf, Use::Matrix(Plain::F32), 2).unwrap();
                let batches = |limit| c.batches(&placed, limit).unwrap();
                assert_eq!(batches(input), [(0..2, input)]);
                let runs = |limit| {
                    batches(limit)
                        .into_iter()
                        .map(|(r, _)| r)
                        .collect::<Vec<_>>()
                };
                assert_eq!(runs(input - 1), [0..1, 1..2]);
                assert_eq!(runs(1), [0..1, 1..2]);
                let (whole_peak, whole) =
                    peak(|| c.read_stack(&placed, cf_stack(), input).unwrap());
                let (split_peak, split) = peak(|| c.read_stack(&placed, cf_stack(), 1).unwrap());
                assert_eq!(
                    split, whole,
                    "bounded reads keep every expert's bytes and scales"
                );
                if direct {
                    assert!(
                        split_peak < whole_peak,
                        "split read peak {split_peak} must be below whole read peak {whole_peak}"
                    );
                }
            }
            let (held, part) = peak(|| c.read(p).unwrap());
            drop(part);
            assert!(read_bytes >= held, "{name}: {read_bytes} < {held}");
            if p.info.ty == GgmlType::NVFP4 && direct {
                let stored = stored(&c.placed(&c.made[name])).count() as u64;
                let align = (2 * stored + 3) * DIRECT_ALIGN as u64;
                assert!(
                    read_bytes - held <= align,
                    "{name}: {read_bytes} passes {held} by more than {align}"
                );
            }
        }
        // A BF16 matrix read whole holds its stored copy and its output, none widened.
        let bf = planned.iter().find(|p| p.info.name == "bf.weight").unwrap();
        let s = bf.info.nbytes;
        assert_eq!(
            c.read_bytes(bf).unwrap(),
            4 * s + 5 * DIRECT_ALIGN as u64 + 8
        );
        let p = planned.iter().find(|p| p.info.name == "v.weight").unwrap();
        c.made.get_mut("v.weight").unwrap().dims = vec![u64::MAX, 2];
        assert!(
            c.read_bytes(p)
                .unwrap_err()
                .to_string()
                .contains("v.weight")
        );
        std::fs::remove_dir_all(&dir).unwrap();

        // Three experts of which two fit the production read limit, in each format a stack is
        // written in one source at a time: the read holds two experts' stored bytes at most, and
        // its read bytes count the larger first run, not the last.
        let fp8 = Format::Fp8Block {
            rows: 128,
            cols: 128,
        };
        for (format, rows) in [
            (Format::Nvfp4Ct, 160 << 10),
            (fp8, 96 << 10),
            (Format::Bf16, 96 << 10),
        ] {
            let dir = temp_dir("read-peak-batches");
            let c = stack(&dir, 3, rows, format);
            let planned = c
                .plan(|t: &str, _: GgmlType, _: &[u64]| {
                    Ok::<_, String>(if t == "ex.weight" {
                        Take::Whole
                    } else {
                        Take::Skip
                    })
                })
                .unwrap();
            let p = &planned[0];
            let placed = c.placed(&c.made["ex.weight"]);
            let expert: u64 = stored(&placed[..1])
                .map(|n| c.folder.raw(n).unwrap().bytes.len() as u64 + (2 * DIRECT_ALIGN) as u64)
                .sum();
            assert!(2 * expert <= STACK_READ_MAX && 3 * expert > STACK_READ_MAX);
            let read_bytes = c.read_bytes(p).unwrap();
            assert!(
                (p.info.nbytes + 2 * expert..p.info.nbytes + 3 * expert).contains(&read_bytes),
                "{format}: {read_bytes} does not count two of three experts of {expert} bytes"
            );
            let (held, part) = peak(|| c.read(p).unwrap());
            let mapped =
                modelopt::normalize(|n| c.folder.raw(n), &placed, Use::Matrix(Plain::F32)).unwrap();
            assert_eq!(
                part.spans().concat(),
                mapped.bytes,
                "{format}: the mapping's stack"
            );
            drop(part);
            assert!(read_bytes >= held, "{format}: {read_bytes} < {held}");
            std::fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn a_mapped_shard_does_not_hide_a_direct_open_error() {
        let dir = temp_dir("direct-error");
        let c = checkpoint(&dir);
        let (path, _) = c.folder.file_range("bf.w").unwrap();
        std::fs::rename(&path, path.with_extension("moved")).unwrap();
        let parts = c.plan(rule("bf.weight", 0, 0, 1)).unwrap();
        assert_eq!(parts.len(), 1);
        for p in parts {
            match c.read(&p) {
                Err(Error::Direct(EngramError::OpenDirect {
                    path: found,
                    source,
                })) => {
                    assert_eq!(found, path);
                    assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
                }
                Err(e) => panic!("unexpected error: {e}"),
                Ok(_) => panic!("the mapping hid the missing shard"),
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
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
