//! The plan: what becomes of every tensor, decided before anything is written. [`Plan::build`] orders the tensors
//! as llama-quantize's driver walks them, types them ([`crate::resolve`]), checks that each conversion can be made
//! (a source the host reads, the importance llama-quantize needs, `--allow-requantize`), lays out every output
//! shard's header ([`crate::meta`]) and, with `--llama-plan`, holds all of it against llama-quantize's own
//! `--dry-run` plan. [`Plan::render`] prints it, and `--plan` stops there.

use crate::cli::Args;
use crate::imatrix::Imatrix;
use crate::resolve::{self, Model, Options, TensorRef, layer_of};
use crate::rules::Rules;
use crate::scan::{SOURCE_TYPES, row_bytes};
use crate::{Error, Result};
use gguf::iq_quant::IqKind;
use gguf::kquant::KType;
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// llama-quantize's `--dry-run` rows by tensor name: type, dims as it spells them, bytes.
#[derive(Debug, Default)]
pub struct LlamaPlan(HashMap<String, (String, String, u64)>);

impl LlamaPlan {
    /// Read the plan step's TSV (`what` names the file in a refusal).
    pub fn parse(text: &str, what: &str) -> Result<LlamaPlan> {
        let mut rows = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            let bytes = f.get(3).and_then(|b| b.parse::<u64>().ok());
            let (4, Some(bytes)) = (f.len(), bytes) else {
                return Err(Error::refused(format!(
                    "{what} line {}: not name TAB type TAB dims TAB bytes",
                    i + 1
                )));
            };
            if rows
                .insert(
                    f[0].to_string(),
                    (f[1].to_uppercase(), f[2].to_string(), bytes),
                )
                .is_some()
            {
                return Err(Error::refused(format!("{what}: {} is planned twice", f[0])));
            }
        }
        Ok(LlamaPlan(rows))
    }
}

/// `dims` as the plan spells them: `x`-joined, trailing 1s dropped.
#[must_use]
pub fn plan_dims(dims: &[u64]) -> String {
    let mut d: Vec<String> = dims.iter().map(u64::to_string).collect();
    while d.len() > 1 && d.last().is_some_and(|x| x == "1") {
        d.pop();
    }
    d.join("x")
}

/// What quantizes a target type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    /// The card or the host rule's IQ search.
    Iq(IqKind),
    /// The host's K-quant rules.
    K(KType),
    /// A widening to f32.
    F32,
}

impl Target {
    /// The writer of `ty`, when this quantizer has one.
    #[must_use]
    pub fn of(ty: GgmlType) -> Option<Target> {
        match ty {
            GgmlType::IQ1_S => Some(Target::Iq(IqKind::Iq1S)),
            GgmlType::IQ2_XXS => Some(Target::Iq(IqKind::Iq2Xxs)),
            GgmlType::F32 => Some(Target::F32),
            t => KType::of(t).map(Target::K),
        }
    }
}

/// What becomes of one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// Written byte for byte.
    Copy,
    /// Converted to the item's `dst` by the target.
    Convert(Target),
}

/// One tensor of the plan.
#[derive(Debug, Clone)]
pub struct Item {
    pub name: String,
    pub dims: Vec<u64>,
    pub src: GgmlType,
    pub dst: GgmlType,
    /// The input shard it is read from.
    pub shard_in: usize,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub fate: Fate,
    /// Whether the importance matrix has an entry the quantizer uses.
    pub importance: bool,
}

/// The GGML type an IQ kind writes.
#[must_use]
pub fn target_type(k: IqKind) -> GgmlType {
    match k {
        IqKind::Iq1S => GgmlType::IQ1_S,
        IqKind::Iq2Xxs => GgmlType::IQ2_XXS,
    }
}

/// The card's rates, in weights per second: the search kernel's rows in `docs/research/iq-quant-card.md` (derived
/// terms of `--plan`, never a measurement of this run; `tools/ref/quant-measure.sh` is the measurement).
const CARD_WEIGHTS_PER_S: [(IqKind, f64); 2] = [(IqKind::Iq1S, 1.09e9), (IqKind::Iq2Xxs, 8.3e8)];
/// The drive rates `--plan` derives its I/O terms from, bytes per second.
const READ_BPS: f64 = 1.6e9;
const WRITE_BPS: f64 = 1.5e9;
/// The host rule's rate on one thread, weights per second, per target: measured by `gate-quantize-oracle`'s rate
/// line on the box (docs/quantize.md), carried here as the plan's derived term.
const HOST_WEIGHTS_PER_S_PER_THREAD: f64 = 8.0e6;

/// The derived wall's terms, seconds.
#[derive(Debug, Clone, Copy)]
pub struct Wall {
    pub card: f64,
    pub host: f64,
    pub read: f64,
    pub write: f64,
    pub total: f64,
}

/// The whole plan.
#[derive(Debug)]
pub struct Plan {
    /// Every tensor in output order.
    pub items: Vec<Item>,
    /// Per output shard, the indices of its items.
    pub shards: Vec<Vec<usize>>,
    /// Per output shard, the layout of its file.
    pub layouts: Vec<Layout>,
    /// The output files, shard by shard.
    pub outs: Vec<PathBuf>,
    /// Weights quantized by an IQ target, per kind.
    pub iq_weights: [(IqKind, f64); 2],
    /// Weights converted by the host's K rules and widenings.
    pub host_weights: f64,
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// The output's file bytes, headers and padding included.
    pub out_file_bytes: u64,
}

/// The output names: `PREFIX-0000i-of-0000n.gguf` under `--keep-split`, else the path as given.
fn out_names(a: &Args, n_split: usize) -> Vec<PathBuf> {
    if a.keep_split {
        (0..n_split)
            .map(|i| PathBuf::from(format!("{}-{:05}-of-{n_split:05}.gguf", a.out, i + 1)))
            .collect()
    } else {
        vec![PathBuf::from(&a.out)]
    }
}

/// Whether `ty` is one of the quantized types (`ggml_is_quantized`).
fn is_quantized(ty: GgmlType) -> bool {
    !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

/// The source types the host reads for a K or f32 target.
fn readable(ty: GgmlType) -> bool {
    SOURCE_TYPES.contains(&ty) || ty.has_dequant()
}

impl Plan {
    /// Decide every tensor of `set`; no file is written and none but the inputs' headers is read.
    pub fn build(
        set: &gguf::Split,
        rules: &Rules,
        llama: Option<&LlamaPlan>,
        im: Option<&Imatrix>,
        a: &Args,
    ) -> Result<Plan> {
        let model = Model::read(set)?;
        if let Some(v) = set.value(gguf::GENERAL_ALIGNMENT)
            && gguf::write::alignment_value(v).map_err(|e| Error::refused(e.to_string()))? != 32
        {
            return Err(Error::refused(
                "general.alignment is not 32: llama-quantize pads every tensor to 32 whatever the header says",
            ));
        }
        // The order ggml's driver walks: the loader's map (by layer, then name), or under --keep-split the files'.
        let mut ts: Vec<(usize, &gguf::TensorInfo)> = set.iter_tensors().collect();
        if a.keep_split {
            ts.sort_by_key(|(s, t)| (*s, t.offset));
        } else {
            ts.sort_by(|(_, x), (_, y)| {
                (layer_of(&x.name).unwrap_or(-1), &x.name)
                    .cmp(&(layer_of(&y.name).unwrap_or(-1), &y.name))
            });
        }
        let refs: Vec<TensorRef<'_>> = ts
            .iter()
            .map(|(_, t)| TensorRef {
                name: &t.name,
                dims: &t.dims,
                ty: t.ty,
            })
            .collect();
        let opts = Options {
            ftype: a.ftype,
            output_tensor_type: a.output_tensor_type,
            token_embedding_type: a.token_embedding_type,
        };
        let dst = resolve::resolve(&refs, rules, &model, &opts)?;
        let n_split = if a.keep_split { set.shard_count() } else { 1 };
        let mut items = Vec::with_capacity(ts.len());
        let mut shards = vec![Vec::new(); n_split];
        for (i, ((shard, t), dst)) in ts.iter().zip(dst).enumerate() {
            let item = decide(t, *shard, dst, a, im)?;
            shards[if a.keep_split { *shard } else { 0 }].push(i);
            items.push(item);
        }
        if let Some(lp) = llama {
            cross_check(&items, lp)?;
        }
        let outs = out_names(a, n_split);
        let kvs = crate::meta::output_kvs(set, a, im, items.len(), n_split);
        let mut p = Plan {
            items,
            shards,
            layouts: Vec::new(),
            outs,
            iq_weights: [(IqKind::Iq1S, 0.0), (IqKind::Iq2Xxs, 0.0)],
            host_weights: 0.0,
            bytes_in: 0,
            bytes_out: 0,
            out_file_bytes: 0,
        };
        for (s, kv) in kvs.iter().enumerate() {
            let decls = p.shards[s]
                .iter()
                .map(|&i| {
                    let it = &p.items[i];
                    TensorDecl {
                        name: it.name.clone(),
                        dims: ggml_dims(&it.dims),
                        type_id: it.dst.as_u32(),
                        nbytes: it.bytes_out,
                    }
                })
                .collect();
            let layout = Layout::new(kv, decls).map_err(|e| Error::refused(e.to_string()))?;
            p.out_file_bytes += layout.file_len();
            p.layouts.push(layout);
        }
        for it in &p.items {
            let w = it.dims.iter().product::<u64>() as f64;
            match it.fate {
                Fate::Convert(Target::Iq(k)) => {
                    p.iq_weights[usize::from(k == IqKind::Iq2Xxs)].1 += w;
                    p.bytes_in += it.bytes_in;
                    p.bytes_out += it.bytes_out;
                }
                Fate::Convert(_) => {
                    p.host_weights += w;
                    p.bytes_in += it.bytes_in;
                    p.bytes_out += it.bytes_out;
                }
                Fate::Copy => {}
            }
        }
        Ok(p)
    }

    /// The number of tensors converted.
    #[must_use]
    pub fn converted(&self) -> usize {
        self.items
            .iter()
            .filter(|i| matches!(i.fate, Fate::Convert(_)))
            .count()
    }

    /// The wall `--plan` derives: the card's search of the IQ stacks (host cores when `cpu`), the host rules on the
    /// rest (under the card's shadow when the card runs), the reads and the write. `None` with nothing converted.
    #[must_use]
    pub fn derived_wall(&self, cpu: bool, threads: usize) -> Option<Wall> {
        if self.converted() == 0 {
            return None;
        }
        let host_rate = HOST_WEIGHTS_PER_S_PER_THREAD * threads as f64;
        let iq: f64 = self.iq_weights.iter().map(|w| w.1).sum();
        let card: f64 = if cpu {
            0.0
        } else {
            CARD_WEIGHTS_PER_S
                .iter()
                .zip(&self.iq_weights)
                .map(|(r, w)| w.1 / r.1)
                .sum()
        };
        let host = (self.host_weights + if cpu { iq } else { 0.0 }) / host_rate;
        let copied: u64 = self
            .items
            .iter()
            .filter(|i| i.fate == Fate::Copy)
            .map(|i| i.bytes_in)
            .sum();
        let read = (self.bytes_in + copied) as f64 / READ_BPS;
        let write = self.out_file_bytes as f64 / WRITE_BPS;
        let compute = if cpu { host } else { card.max(host) };
        Some(Wall {
            card,
            host,
            read,
            write,
            total: compute.max(read).max(write),
        })
    }

    /// What `--plan` prints; `free` is the free bytes of the output's drive, when it could be read.
    #[must_use]
    pub fn render(&self, a: &Args, free: Option<u64>, threads: usize) -> String {
        let mut s = String::new();
        let gb = |b: u64| b as f64 / 1e9;
        let _ = writeln!(
            s,
            "bloomery-quantize --plan: {} output shard(s), {} tensors, file type {:?}; source type, target type, bytes in -> out, fate",
            self.shards.len(),
            self.items.len(),
            a.ftype
        );
        for item in &self.items {
            let fate = match item.fate {
                Fate::Convert(Target::Iq(_)) if a.cpu => "converted on the host",
                Fate::Convert(Target::Iq(_)) => "converted on the card",
                Fate::Convert(_) => "converted on the host",
                Fate::Copy => "copied",
            };
            let _ = writeln!(
                s,
                "  {} {:?} -> {:?} [{}] {} -> {} B: {fate}",
                item.name,
                item.src,
                item.dst,
                plan_dims(&item.dims),
                item.bytes_in,
                item.bytes_out
            );
        }
        let _ = writeln!(
            s,
            "converted: {} tensors, IQ1_S {:.3e} + IQ2_XXS {:.3e} weights on {}, {:.3e} weights on the host; {:.2} GB in -> {:.2} GB out; the rest copied",
            self.converted(),
            self.iq_weights[0].1,
            self.iq_weights[1].1,
            if a.cpu { "the host" } else { "the card" },
            self.host_weights,
            gb(self.bytes_in),
            gb(self.bytes_out)
        );
        match self.derived_wall(a.cpu, threads) {
            Some(w) => {
                let _ = writeln!(
                    s,
                    "wall [derived: card {:.0} s, host rules {:.0} s on {threads} threads at {:.1e} weights/s/thread (under the card's), reads {:.0} s at 1.6 GB/s, write {:.0} s at 1.5 GB/s, the largest of them]: {:.0} s; the measurement is the run's own",
                    w.card, w.host, HOST_WEIGHTS_PER_S_PER_THREAD, w.read, w.write, w.total
                );
            }
            None => {
                let _ = writeln!(s, "wall: nothing is converted");
            }
        }
        let _ = writeln!(
            s,
            "room: the output needs {:.2} GB under {}{}",
            gb(self.out_file_bytes),
            a.out_dir().display(),
            match free {
                Some(f) if f >= self.out_file_bytes => format!(" ({:.2} GB free)", gb(f)),
                Some(f) => format!(" ({:.2} GB free: NOT ENOUGH, a run is refused)", gb(f)),
                None => " (free space not readable)".to_string(),
            }
        );
        s
    }
}

/// `ggml_n_dims`: the dims a tensor header lists, trailing 1s dropped.
#[must_use]
pub fn ggml_dims(dims: &[u64]) -> Vec<u64> {
    let n = dims.iter().rposition(|&d| d != 1).map_or(1, |p| p + 1);
    dims[..n].to_vec()
}

/// The plan's row of one tensor.
fn decide(
    t: &gguf::TensorInfo,
    shard_in: usize,
    dst: GgmlType,
    a: &Args,
    im: Option<&Imatrix>,
) -> Result<Item> {
    let ne0 = t.dims[0] as usize;
    let rows: u64 = t.dims[1..].iter().product();
    let base = |fate: Fate, bytes_out: u64, importance: bool| Item {
        name: t.name.clone(),
        dims: t.dims.clone(),
        src: t.ty,
        dst,
        shard_in,
        bytes_in: t.nbytes,
        bytes_out,
        fate,
        importance,
    };
    let is_embd_or_out = t.name == "token_embd.weight"
        || t.name == "per_layer_token_embd.weight"
        || t.name == "output.weight";
    let needs_importance = matches!(dst, GgmlType::IQ1_S | GgmlType::IQ2_XXS) && !is_embd_or_out;
    if needs_importance && im.is_none() && dst != t.ty && !a.plan {
        return Err(Error::refused(format!(
            "{}: {dst:?} needs an importance matrix (--imatrix): llama-quantize refuses it without one (--plan alone prints without)",
            t.name
        )));
    }
    if dst == t.ty {
        return Ok(base(Fate::Copy, t.nbytes, false));
    }
    let target = Target::of(dst).ok_or_else(|| {
        Error::refused(format!(
            "{}: target type {dst:?} is not one this quantizer writes",
            t.name
        ))
    })?;
    if is_quantized(t.ty) && !a.allow_requantize {
        return Err(Error::refused(format!(
            "{}: requantizing from type {:?} is disabled (--allow-requantize)",
            t.name, t.ty
        )));
    }
    let src_ok = match target {
        Target::Iq(_) => SOURCE_TYPES.contains(&t.ty),
        _ => readable(t.ty),
    };
    if !src_ok {
        return Err(Error::refused(format!(
            "{} is {:?}: the source types read here for {dst:?} do not include it",
            t.name, t.ty
        )));
    }
    row_bytes(t.ty, ne0).ok_or_else(|| {
        Error::refused(format!(
            "{}: {:?} rows of {ne0} values are not read",
            t.name, t.ty
        ))
    })?;
    let n_exp = t.dims[2..].iter().product::<u64>();
    let importance = match im.and_then(|m| m.len_of(&t.name)) {
        None => false,
        Some(n) if n == t.dims[0] * n_exp => true,
        Some(_) if t.name == "token_embd.weight" || t.name == "per_layer_token_embd.weight" => {
            false
        }
        Some(n) => {
            return Err(Error::refused(format!(
                "imatrix {}: {n} values, the tensor needs {} ({} x {n_exp} experts)",
                t.name,
                t.dims[0] * n_exp,
                t.dims[0]
            )));
        }
    };
    if needs_importance && !importance && !a.plan {
        return Err(Error::refused(format!(
            "Missing importance matrix for tensor {} in a very low-bit quantization",
            t.name
        )));
    }
    let row = row_bytes(dst, ne0)
        .ok_or_else(|| Error::refused(format!("{}: {dst:?} rows of {ne0} values", t.name)))?;
    Ok(base(Fate::Convert(target), rows * row as u64, importance))
}

/// The plan's size column is MiB printed to two decimals for a converted tensor and three for a copied one, so the
/// table's bytes are that print read back: the two sizes are the same when they print the same.
fn same_mib(plan_bytes: u64, bytes: u64, fate: Fate) -> bool {
    let mib = |b: u64| b as f64 / 1_048_576.0;
    match fate {
        Fate::Convert(_) => format!("{:.2}", mib(plan_bytes)) == format!("{:.2}", mib(bytes)),
        Fate::Copy => format!("{:.3}", mib(plan_bytes)) == format!("{:.3}", mib(bytes)),
    }
}

/// Every tensor's type, dims and sizes against llama-quantize's own plan, in either direction.
fn cross_check(items: &[Item], lp: &LlamaPlan) -> Result<()> {
    if lp.0.len() != items.len() {
        let missing: Vec<&String> =
            lp.0.keys()
                .filter(|k| !items.iter().any(|i| &i.name == *k))
                .take(3)
                .collect();
        return Err(Error::refused(format!(
            "the llama plan names {} tensors, the set holds {}; plan rows the set does not hold: {missing:?}",
            lp.0.len(),
            items.len()
        )));
    }
    for it in items {
        let Some((lty, ldims, lbytes)) = lp.0.get(&it.name) else {
            return Err(Error::refused(format!(
                "the llama plan has no row for {}",
                it.name
            )));
        };
        let ours = format!("{:?}", it.dst).to_uppercase();
        if *lty != ours
            || *ldims != plan_dims(&it.dims)
            || !same_mib(*lbytes, it.bytes_out, it.fate)
        {
            return Err(Error::refused(format!(
                "{}: llama-quantize's plan says {lty} [{ldims}] {lbytes} B, this plan {ours} [{}] {} B: the two must agree",
                it.name,
                plan_dims(&it.dims),
                it.bytes_out
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_are_spelled_as_the_plan_spells_them() {
        assert_eq!(plan_dims(&[512, 4, 3]), "512x4x3");
        assert_eq!(plan_dims(&[256, 3, 1]), "256x3");
        assert_eq!(plan_dims(&[8, 1]), "8");
        assert_eq!(ggml_dims(&[256, 3, 1]), vec![256, 3]);
        assert_eq!(ggml_dims(&[8, 1, 1]), vec![8]);
        assert_eq!(ggml_dims(&[4, 1, 256]), vec![4, 1, 256]);
    }

    #[test]
    fn the_llama_plan_reads_four_columns_and_refuses_the_rest() {
        let p = LlamaPlan::parse("a\tiq1_s\t4x2\t10\nb\tF32\t8\t32\n", "plan.tsv").unwrap();
        assert_eq!(p.0["a"], ("IQ1_S".to_string(), "4x2".to_string(), 10));
        for bad in ["a\tx\t1\n", "a\tx\t1\tmany\n"] {
            assert!(
                LlamaPlan::parse(bad, "plan.tsv")
                    .unwrap_err()
                    .to_string()
                    .contains("plan.tsv line 1")
            );
        }
        assert!(
            LlamaPlan::parse("a\tx\t1\t2\na\ty\t1\t2\n", "p")
                .unwrap_err()
                .to_string()
                .contains("planned twice")
        );
    }

    #[test]
    fn a_plan_row_that_differs_is_refused_in_either_direction() {
        let item = |name: &str, dst| Item {
            name: name.into(),
            dims: vec![256, 2],
            src: GgmlType::F32,
            dst,
            shard_in: 0,
            bytes_in: 2048,
            bytes_out: 288,
            fate: Fate::Copy,
            importance: false,
        };
        let items = [item("a", GgmlType::Q4_K)];
        let ok = LlamaPlan::parse("a\tQ4_K\t256x2\t288\n", "p").unwrap();
        assert!(cross_check(&items, &ok).is_ok());
        for bad in [
            "a\tQ5_K\t256x2\t288\n",
            "a\tQ4_K\t256x2\t2880000\n",
            "a\tQ4_K\t256x3\t288\n",
            "a\tQ4_K\t256x2\t288\nb\tQ4_K\t1\t1\n",
            "b\tQ4_K\t256x2\t288\n",
        ] {
            let lp = LlamaPlan::parse(bad, "p").unwrap();
            assert!(cross_check(&items, &lp).is_err(), "{bad}");
        }
    }
}
