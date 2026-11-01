//! The tensors a load reads, each whole or the part a rank holds, from a source ([`Source`]): a
//! GGUF file here, a Hugging Face checkpoint folder in `model_hf::part`. The load gives one rule
//! ([`Source::plan`]): for a tensor's name, card type and dims, whether it is read and which part
//! ([`Take`]). A source knows no model and no rank count. Every part is planned from headers
//! alone before any value is read; [`Source::read`] then gives the part's bytes in its card type
//! and the tensor scales it reads.
//!
//! A GGUF file holds its tensors in their card types: a part is runs of its rows' bytes
//! ([`part_at`]), and a scaled tensor's scales are the `.scale` beside it, one an expert along a
//! cut of a stack's experts or of the experts a list names ([`scales_at`]).

use std::ops::Range;
use std::path::PathBuf;

use gguf::{GgmlType, Split, TensorInfo};

use crate::placement::ExpertList;
use crate::placement::tp::{TpError, part_of};

/// Why a part was refused.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{name}: {what}")]
    Tensor { name: String, what: String },
    #[error(transparent)]
    Cut(#[from] TpError),
    #[error(transparent)]
    Load(#[from] gguf::LoadError),
}

/// The refusal of tensor `name` for `what`.
pub fn refuse(name: &str, what: impl Into<String>) -> Error {
    Error::Tensor {
        name: name.to_string(),
        what: what.into(),
    }
}

/// The indices of a tensor a part holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum At {
    /// The indices `.1` along dim `.0`.
    Along(usize, Range<u64>),
    /// The experts the list names of a stack (`[k, rows, n_expert]`), in the list's order: a
    /// card's slots, expert `ids()[s]` in slot `s`.
    Experts(ExpertList),
}

/// What a load reads of a tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Take {
    Skip,
    Whole,
    /// The indices `.1` along dim `.0`.
    Part(usize, Range<u64>),
    /// The experts of a stack the list names, in its order ([`At::Experts`]).
    Experts(ExpertList),
}

/// What `rule` takes of tensor `t`: `None` skipped, else the indices it reads (`None` the whole
/// tensor); a refusal of `rule` names the tensor.
pub fn take<E: std::fmt::Display>(
    rule: &impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    t: &TensorInfo,
) -> Result<Option<Option<At>>, Error> {
    Ok(
        match rule(&t.name, t.ty, &t.dims).map_err(|e| refuse(&t.name, e.to_string()))? {
            Take::Skip => None,
            Take::Whole => Some(None),
            Take::Part(dim, at) => Some(Some(At::Along(dim, at))),
            Take::Experts(list) => Some(Some(At::Experts(list))),
        },
    )
}

/// The part of the tensor `t` (a header entry of its whole bytes) at `at`, the whole tensor for
/// `None`: the part's header entry and the byte ranges of its rows in the tensor's bytes, in the
/// part's order. Indices along a dim are [`part_of`]'s; a stack's listed experts are each
/// expert's run, in the list's order. A list of no expert, a list of a tensor that is not a stack
/// of experts and an expert past the stack's are refused by name, besides [`part_of`]'s
/// refusals.
pub fn part_at(t: &TensorInfo, at: Option<&At>) -> Result<(TensorInfo, Vec<Range<usize>>), Error> {
    let list = match at {
        None => return Ok(part_of(t, None)?),
        Some(At::Along(dim, r)) => return Ok(part_of(t, Some((*dim, r.clone())))?),
        Some(At::Experts(list)) => list,
    };
    let &[_, _, n] = t.dims.as_slice() else {
        return Err(refuse(
            &t.name,
            format!(
                "{:?} is not a stack of experts, which a list of experts is taken of",
                t.dims
            ),
        ));
    };
    if list.is_empty() {
        return Err(refuse(&t.name, "a list of no expert takes nothing"));
    }
    let (mut runs, mut nbytes) = (Vec::with_capacity(list.ids().len()), 0u64);
    for &e in list.ids() {
        let e = u64::from(e);
        if e >= n {
            return Err(refuse(
                &t.name,
                format!("expert {e} is past the stack's {n}"),
            ));
        }
        let (one, run) = part_of(t, Some((2, e..e + 1)))?;
        nbytes += one.nbytes;
        runs.extend(run);
    }
    let mut dims = t.dims.clone();
    dims[2] = list.len();
    Ok((
        TensorInfo {
            name: t.name.clone(),
            dims,
            ty: t.ty,
            offset: t.offset,
            nbytes,
        },
        runs,
    ))
}

/// The tensor scales of the part at `at` of tensor `name` of `dims`, of the tensor's scales `all`
/// (one, or one an expert of a stack): a stack's experts' own in the part's order, along its
/// experts or by a list; `all` for any other part. An expert past `all` is refused by name.
pub fn scales_at(
    name: &str,
    dims: &[u64],
    all: Vec<f32>,
    at: Option<&At>,
) -> Result<Vec<f32>, Error> {
    let pick = |e: u64| {
        usize::try_from(e)
            .ok()
            .and_then(|e| all.get(e).copied())
            .ok_or_else(|| {
                refuse(
                    name,
                    format!("expert {e} is past its {} tensor scales", all.len()),
                )
            })
    };
    match at {
        Some(At::Along(2, r)) if dims.len() == 3 => r.clone().map(pick).collect(),
        Some(At::Experts(list)) => list.ids().iter().map(|&e| pick(u64::from(e))).collect(),
        _ => Ok(all),
    }
}

/// One part a load reads ([`Source::plan`]): the indices it holds (`None` the whole tensor) and
/// its header entry (name, card type, dims and bytes).
#[derive(Clone, Debug)]
pub struct Planned {
    pub at: Option<At>,
    pub info: TensorInfo,
}

/// One read's leading parts, bounded by normalized payload and owned host room. An oversized
/// first part runs alone; the caller checks its returned peak before allocating it.
/// `costs` holds each part's owned read peak and serial packing peak. Reads coexist;
/// packing holds only one part's buffers at a time.
pub fn read_ahead(
    planned: &[Planned],
    costs: &[(u64, u64)],
    room: u64,
) -> Result<(usize, u64), Error> {
    const PAYLOAD: u64 = 8 << 30;
    if planned.len() != costs.len() {
        return Err(refuse(
            "read-ahead",
            "the parts and their costs differ in length",
        ));
    }
    let (mut n, mut bytes, mut payload) = (0, 0u64, 0u64);
    let (mut reads, mut packing) = (0u64, 0u64);
    for (p, &(read, pack)) in planned.iter().zip(costs) {
        let next_reads = reads
            .checked_add(read)
            .ok_or_else(|| refuse(&p.info.name, "read-ahead bytes pass u64"))?;
        let next_packing = packing.max(pack);
        let next = next_reads
            .checked_add(next_packing)
            .ok_or_else(|| refuse(&p.info.name, "read-ahead bytes pass u64"))?;
        let more = payload
            .checked_add(p.info.nbytes)
            .ok_or_else(|| refuse(&p.info.name, "read-ahead payload passes u64"))?;
        if n > 0 && (more > PAYLOAD || next > room) {
            break;
        }
        (reads, packing, bytes, payload) = (next_reads, next_packing, next, more);
        n += 1;
        if payload > PAYLOAD || bytes > room {
            break;
        }
    }
    Ok((n, bytes))
}

/// A part's bytes in its card type ([`Source::read`]), with its header entry and the tensor
/// scales its type reads.
pub struct Part<'s> {
    pub info: TensorInfo,
    bytes: Bytes<'s>,
    pub scale: Option<Vec<f32>>,
}

enum Bytes<'s> {
    /// Runs of a mapped file's rows.
    File(Vec<&'s [u8]>),
    /// Bytes the source made in memory of its own.
    Made(Vec<u8>),
}

impl Part<'_> {
    /// A part whose bytes `bytes` the source made in memory of its own.
    #[must_use]
    pub fn made(info: TensorInfo, bytes: Vec<u8>, scale: Option<Vec<f32>>) -> Part<'static> {
        Part {
            info,
            bytes: Bytes::Made(bytes),
            scale,
        }
    }

    /// The part's rows' bytes, runs in row order.
    #[must_use]
    pub fn spans(&self) -> Vec<&[u8]> {
        match &self.bytes {
            Bytes::File(runs) => runs.clone(),
            Bytes::Made(bytes) => vec![bytes],
        }
    }
}

/// One plane of a row-gathered table, in row order: each run holds whole
/// rows of `row_bytes` bytes where the source stores them.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredPlane {
    pub row_bytes: u64,
    pub runs: Vec<(PathBuf, Range<u64>)>,
}

/// A row-gathered table where its source stores it: F8 values, or NVFP4
/// codes with a separate plane of block scales, at one tensor `scale`.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRows {
    pub values: StoredPlane,
    pub block_scales: Option<StoredPlane>,
    pub scale: f32,
}

/// Where a load reads its tensors.
pub trait Source {
    /// Why the source refused a part.
    type Error: std::error::Error;

    /// Every tensor of the source `rule` takes, each at the part it takes ([`Take`]) given the
    /// tensor's name, card type and dims, planned from headers alone: a refusal of `rule` and a
    /// part off the tensor's blocks or past its dims are refused by name before any value is
    /// read.
    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Self::Error>;

    /// An upper bound on the owned payload bytes a read of `p` holds, including its result,
    /// normalization scratch, stored inputs and a contiguous copy for a packed upload.
    /// Borrowed file pages are not owned bytes.
    fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error>;

    /// The bytes of the part `p` planned ([`Source::plan`]) and the tensor scales its type reads;
    /// a tensor no longer in the source is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Self::Error>;

    /// Drop the file pages the source read part `p` from, once a load has copied its bytes and no
    /// later reader takes them from the file. A source whose loads release their own pages (a
    /// split's placed load, `PageDrop`) drops none here.
    fn release(&self, _p: &Planned) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Write what the host reads of the source where a located header holds it but the source
    /// keeps in another form (a checkpoint folder's block scales stored compressed), before any
    /// host reader takes it. A source that keeps every such byte as the header holds it writes
    /// nothing.
    fn make_in_place(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A source whose plan considers only the names `keep` selects. Reads, page release and the
/// in-place writes use the underlying source, so several model components can share its stored
/// tensors.
pub struct Filtered<'s, S, F> {
    pub source: &'s S,
    pub keep: F,
}

impl<S: Source, F: Fn(&str) -> bool> Source for Filtered<'_, S, F> {
    type Error = S::Error;

    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Self::Error> {
        self.source.plan(|name, ty, dims| {
            if (self.keep)(name) {
                rule(name, ty, dims)
            } else {
                Ok(Take::Skip)
            }
        })
    }

    fn read_bytes(&self, p: &Planned) -> Result<u64, Self::Error> {
        self.source.read_bytes(p)
    }

    fn read(&self, p: &Planned) -> Result<Part<'_>, Self::Error> {
        self.source.read(p)
    }

    fn release(&self, p: &Planned) -> Result<(), Self::Error> {
        self.source.release(p)
    }

    fn make_in_place(&self) -> Result<(), Self::Error> {
        self.source.make_in_place()
    }
}

impl Source for Split {
    type Error = Error;

    /// A `.scale` beside a scaled tensor is read with it, not on its own.
    fn plan<E: std::fmt::Display>(
        &self,
        rule: impl Fn(&str, GgmlType, &[u64]) -> Result<Take, E>,
    ) -> Result<Vec<Planned>, Error> {
        let mut out = Vec::new();
        for (_, t) in self.iter_tensors() {
            if gguf::scaled_weight(self, &t.name).is_some() {
                continue;
            }
            if let Some(at) = take(&rule, t)? {
                let (info, _) = part_at(t, at.as_ref())?;
                out.push(Planned { at, info });
            }
        }
        Ok(out)
    }

    fn read_bytes(&self, p: &Planned) -> Result<u64, Error> {
        let name = &p.info.name;
        let (_, t) = self
            .find(name)
            .ok_or_else(|| refuse(name, "is not in the file"))?;
        // A noncontiguous rank cut needs a contiguous copy when its upload packs planes.
        // tensor_scales may also hold the whole scale vector beside a rank's cut of it.
        let scales = if t.ty.reads_scale() {
            t.dims
                .last()
                .copied()
                .unwrap_or(1)
                .checked_mul(8)
                .ok_or_else(|| refuse(name, "scale read bytes pass u64"))?
        } else {
            0
        };
        p.info
            .nbytes
            .checked_add(scales)
            .ok_or_else(|| refuse(name, "row copy and scale read bytes pass u64"))
    }

    /// A scale the file lacks or does not hold as its type reads it is refused by name.
    fn read(&self, p: &Planned) -> Result<Part<'_>, Error> {
        let name = p.info.name.as_str();
        let (shard, t) = self
            .find(name)
            .ok_or_else(|| refuse(name, "is not in the file"))?;
        let data = self
            .shard(shard)
            .ok_or_else(|| refuse(name, format!("names shard {shard}")))?
            .data(t)?;
        let (info, runs) = part_at(t, p.at.as_ref())?;
        let find = |n: &str| {
            let (s, t) = self.find(n)?;
            Some((self.shard(s)?, t))
        };
        let scale = gguf::tensor_scales(t, find)?
            .map(|s| scales_at(name, &t.dims, s, p.at.as_ref()))
            .transpose()?;
        Ok(Part {
            info,
            bytes: Bytes::File(runs.into_iter().map(|r| &data[r]).collect()),
            scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gguf::write::{Layout, TensorDecl};

    #[test]
    fn a_read_takes_the_parts_that_fit_and_at_least_one() {
        let part = |nbytes| Planned {
            at: None,
            info: TensorInfo {
                name: "p".into(),
                dims: vec![1],
                ty: GgmlType::F32,
                offset: 0,
                nbytes,
            },
        };
        let gib = 1u64 << 30;
        let parts = [part(3 * gib), part(5 * gib), part(4)];
        assert_eq!(
            read_ahead(&parts, &[(6 * gib, 0), (10 * gib, 0), (8, 0)], 16 * gib).unwrap(),
            (2, 16 * gib)
        );
        assert_eq!(
            read_ahead(&parts, &[(6 * gib, 0), (10 * gib, 0), (8, 0)], 16 * gib - 1).unwrap(),
            (1, 6 * gib)
        );
        assert_eq!(
            read_ahead(&[part(9 * gib), part(4)], &[(9 * gib, 0), (4, 0)], 20 * gib).unwrap(),
            (1, 9 * gib)
        );
        assert_eq!(
            read_ahead(&[part(gib)], &[(gib, gib)], gib).unwrap(),
            (1, 2 * gib)
        );
        assert_eq!(read_ahead(&[], &[], 0).unwrap(), (0, 0));
        let small = vec![part(32 << 20); 256];
        let costs = vec![(128 << 20, 64 << 20); 256];
        assert_eq!(
            read_ahead(&small, &costs, (1024 - 96) << 20).unwrap(),
            (6, 832 << 20)
        );
        assert_eq!(
            read_ahead(&small, &costs, 48 * gib).unwrap(),
            (256, 32 * gib + (64 << 20))
        );
        assert!(read_ahead(&parts, &[], gib).is_err());
        let small = [part(4), part(4), part(4)];
        let costs = [(1, 2), (1, 8), (1, 4)];
        assert_eq!(read_ahead(&small, &costs, 10).unwrap(), (2, 10));
        assert_eq!(read_ahead(&small, &costs, 9).unwrap(), (1, 3));
        assert!(read_ahead(&small[..2], &[(1, 0), (u64::MAX, 0)], u64::MAX).is_err());
        assert!(read_ahead(&small[..1], &[(1, u64::MAX)], u64::MAX).is_err());
        assert!(read_ahead(&[part(4), part(u64::MAX)], &[(1, 0), (1, 0)], u64::MAX).is_err());
    }

    #[test]
    fn shared_parts_are_planned_only_for_the_selected_component() {
        let names = ["target.weight", "draft.weight"];
        let layout = Layout::new(
            &[],
            names
                .iter()
                .map(|name| TensorDecl {
                    name: (*name).to_string(),
                    dims: vec![1],
                    type_id: GgmlType::F32.as_u32(),
                    nbytes: 4,
                })
                .collect(),
        )
        .unwrap();
        let source = Split::header_only(PathBuf::from("/synthetic/shared-parts"), &layout).unwrap();
        for wanted in names {
            let selected = Filtered {
                source: &source,
                keep: |name: &str| name == wanted,
            };
            let parts = selected
                .plan(|_, _, _| Ok::<_, std::convert::Infallible>(Take::Whole))
                .unwrap();
            assert_eq!(source.read_bytes(&parts[0]).unwrap(), 4);
            assert_eq!(selected.read_bytes(&parts[0]).unwrap(), 4);
            assert_eq!(
                parts
                    .iter()
                    .map(|p| p.info.name.as_str())
                    .collect::<Vec<_>>(),
                vec![wanted]
            );
        }
    }
}
