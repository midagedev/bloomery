//! A checkpoint folder as the input of `bloomery-quantize` (`--hf DIR`): the split `prepare` would write of the
//! folder, read without writing it. The folder is planned as `prepare` plans it (the same map, the same
//! `--arch` and `--drop`, the same shards of `--hf-shard-gib`, [`crate::shards::plan_shards`]), so the header keys, the
//! tensors, their order, their shard and every byte of each are the intermediate set's. A tensor's bytes are made when
//! the quantizer asks for them, from the stored tensors of the folder alone: a stack of experts reads the experts of
//! the run asked for and no others, so an expert batch costs its own bytes.
//!
//! A tensor the split would write with a `.scale` beside it, or as a row table, is refused by name: the K3 map makes
//! none, and the quantizer reads whole rows of one type.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

use gguf::{GgmlType, TensorInfo, Value};
use hf::modelopt::{self, Use};
use hf::quant_config::Plan;
use quantize::cli::HfInput;
use quantize::source::Source;
use quantize::{Error, Result};
use safetensors::Folder;

use crate::arch::{self, MapUse};
use crate::folder::{self, FolderMap, Make};
use crate::part;
use crate::select::Select;
use crate::shards::plan_shards;

fn refused(what: impl std::fmt::Display) -> Error {
    Error::Refused(what.to_string())
}

/// The folder `input.dir`, opened as the split `prepare` writes of it with `input`'s architecture word, drops and
/// shard size; the quantizer's [`Source`].
///
/// # Errors
///
/// A folder no map reads, a refusal of the map or of its plan, and a tensor this source does not make, by name.
pub fn open(input: &HfInput) -> Result<Box<dyn Source + Send>> {
    let model_type = folder::model_type(&input.dir).map_err(refused)?;
    let options = folder::Options {
        arch: input.arch.clone(),
        select: Select::of(input.drop.clone(), vec![]).map_err(refused)?,
    };
    arch::with_split_map(&model_type, Open { input, options }).ok_or_else(|| {
        refused(format!(
            "{}: model_type {model_type:?} has no split map; --hf reads {:?}",
            input.dir.display(),
            arch::split_model_types()
        ))
    })?
}

struct Open<'a> {
    input: &'a HfInput,
    options: folder::Options,
}

impl MapUse for Open<'_> {
    type Out = Result<Box<dyn Source + Send>>;

    fn with<M: FolderMap>(self) -> Self::Out {
        let dir = &self.input.dir;
        let folder::Read {
            checkpoint,
            folder,
            plan,
            planned,
            kvs,
            ..
        } = folder::read_with::<M>(dir, false, &self.options)
            .map_err(|e| refused(format!("{}: {e}", dir.display())))?;
        let mut entries = HashMap::new();
        for p in &planned {
            let name = &p.decl.name;
            let entry = match &p.make {
                Make::Tensor(m) if m.role != Use::Rows => Entry::Stored {
                    sources: m.sources.clone(),
                    role: m.role,
                    mover: checkpoint.mover(name, &m.steps),
                    stack: m.dims.len() == 3 && m.sources.len() as u64 == m.dims[2],
                },
                Make::Constant(c) => Entry::Made(c.bytes()),
                Make::Tensor(_) => {
                    return Err(refused(format!("{name}: a row table is not read by --hf")));
                }
                Make::Scale { of } => {
                    return Err(refused(format!(
                        "{name}: the tensor scale beside {of} is not read by --hf"
                    )));
                }
            };
            entries.insert(name.clone(), entry);
        }
        let limit = self.input.shard_gib << 30;
        let shards =
            plan_shards(planned, &kvs, limit).map_err(|e| refused(format!("--hf: {e}")))?;
        let mut tensors = Vec::new();
        for (no, (shard, _)) in shards.iter().enumerate() {
            let mut offset = 0;
            for p in shard {
                let d = &p.decl;
                tensors.push((
                    no,
                    TensorInfo {
                        name: d.name.clone(),
                        dims: d.dims.clone(),
                        ty: GgmlType::from_u32(d.type_id),
                        offset,
                        nbytes: d.nbytes,
                    },
                ));
                offset += d.nbytes;
            }
        }
        Ok(Box::new(HfSource {
            label: dir.display().to_string(),
            folder,
            plan,
            kvs,
            shards: shards.len(),
            tensors,
            entries,
        }))
    }
}

/// How a tensor of the split is made.
enum Entry {
    /// Of folder tensors: one, or a stack's experts in expert order.
    Stored {
        sources: Vec<String>,
        role: Use,
        /// The move the map makes of the normalized tensor, if any.
        mover: Option<part::Step>,
        /// Whether the sources are the experts of a stack of three dims.
        stack: bool,
    },
    /// Values the map gives, as the F32 tensor's bytes.
    Made(Vec<u8>),
}

struct HfSource {
    label: String,
    folder: Folder,
    plan: Plan,
    kvs: Vec<(String, Value)>,
    shards: usize,
    tensors: Vec<(usize, TensorInfo)>,
    entries: HashMap<String, Entry>,
}

impl HfSource {
    /// `sources` normalized as the split's writer does, held to the declared type and bytes.
    fn make(
        &self,
        info: &TensorInfo,
        sources: &[String],
        role: Use,
        mover: Option<&part::Step>,
        want: u64,
    ) -> Result<Vec<u8>> {
        let name = &info.name;
        let placed = sources
            .iter()
            .map(|s| {
                let at = self.plan.placed.get(s).ok_or_else(|| {
                    refused(format!("{name}: its source {s} is not placed by the plan"))
                })?;
                Ok((s.as_str(), at))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut n = modelopt::normalize(|s| self.folder.raw(s), &placed, role)
            .map_err(|e| refused(format!("{name}: {e}")))?;
        if let Some(step) = mover {
            step(&mut n).map_err(|e| refused(format!("{name}: {e}")))?;
        }
        if GgmlType::from_u32(n.type_id) != info.ty || n.bytes.len() as u64 != want {
            return Err(refused(format!(
                "{name}: made {} bytes of type {}, the split declares {want} of {:?}",
                n.bytes.len(),
                n.type_id,
                info.ty
            )));
        }
        Ok(n.bytes)
    }
}

impl Source for HfSource {
    fn label(&self) -> String {
        format!("{} (checkpoint folder)", self.label)
    }

    fn keys(&self) -> &[(String, Value)] {
        &self.kvs
    }

    fn shard_count(&self) -> usize {
        self.shards
    }

    fn tensors(&self) -> Vec<(usize, TensorInfo)> {
        self.tensors.clone()
    }

    fn read(&self, name: &str, bytes: Range<u64>) -> Result<Cow<'_, [u8]>> {
        let (_, info) = self
            .tensors
            .iter()
            .find(|(_, t)| t.name == name)
            .ok_or_else(|| refused(format!("{name}: not a tensor of the folder's split")))?;
        if bytes.start > bytes.end || bytes.end > info.nbytes {
            return Err(refused(format!(
                "{name}: bytes {}..{} of a tensor of {}",
                bytes.start, bytes.end, info.nbytes
            )));
        }
        let entry = self.entries.get(name).expect("every tensor has its entry");
        let whole = bytes == (0..info.nbytes);
        match entry {
            Entry::Made(all) => Ok(Cow::Owned(
                all[bytes.start as usize..bytes.end as usize].to_vec(),
            )),
            Entry::Stored {
                sources,
                role,
                mover: None,
                stack: true,
            } if !whole => {
                let n_exp = sources.len() as u64;
                let eb = info.nbytes / n_exp;
                if eb * n_exp != info.nbytes
                    || !bytes.start.is_multiple_of(eb)
                    || !bytes.end.is_multiple_of(eb)
                {
                    return Err(refused(format!(
                        "{name}: bytes {}..{} are not whole experts of {eb} bytes",
                        bytes.start, bytes.end
                    )));
                }
                let (e0, e1) = ((bytes.start / eb) as usize, (bytes.end / eb) as usize);
                let want = bytes.end - bytes.start;
                Ok(Cow::Owned(self.make(
                    info,
                    &sources[e0..e1],
                    *role,
                    None,
                    want,
                )?))
            }
            Entry::Stored {
                sources,
                role,
                mover,
                ..
            } => {
                let all = self.make(info, sources, *role, mover.as_ref(), info.nbytes)?;
                if whole {
                    Ok(Cow::Owned(all))
                } else {
                    Ok(Cow::Owned(
                        all[bytes.start as usize..bytes.end as usize].to_vec(),
                    ))
                }
            }
        }
    }
}
