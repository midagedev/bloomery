//! The plan: what becomes of every tensor, decided before anything is written. [`Plan::build`] refuses a set it
//! cannot quantize as ggml would (by tensor name), [`Plan::render`] prints it — source and target type, bytes in
//! and out, quantized here or copied, the derived wall, the room the output needs and the `llama-quantize`
//! command to run next — and `--plan` stops there.
//!
//! What is quantized here: a routed-expert stack (`*_exps.weight`, three or more dims) that the rules type IQ1_S
//! or IQ2_XXS, from F32, F16, BF16 or MXFP4. Everything else is copied, and llama-quantize quantizes the trunk.
//! `--llama-plan` (llama-quantize's own `--dry-run` resolution, `name TAB type TAB dims TAB bytes`) is an optional
//! cross-check: a routed stack the plan and the rules type differently is refused, in either direction.

use crate::cli::Args;
use crate::scan::{SOURCE_TYPES, row_bytes};
use crate::{Error, Result, rules::Rules};
use gguf::iq_quant::{IqKind, imatrix_weights};
use gguf::quant::GgmlType;
use gguf::write::{Layout, TensorDecl};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The rows of llama-quantize's own plan, by tensor name: its type and its dims as it spells them.
#[derive(Debug, Default)]
pub struct LlamaPlan(HashMap<String, (String, String)>);

impl LlamaPlan {
    /// Read the plan step's TSV (`what` names the file in a refusal).
    pub fn parse(text: &str, what: &str) -> Result<LlamaPlan> {
        let mut rows = HashMap::new();
        for (i, line) in text.lines().enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 4 {
                return Err(Error::refused(format!(
                    "{what} line {}: not name TAB type TAB dims TAB bytes",
                    i + 1
                )));
            }
            if rows
                .insert(f[0].to_string(), (f[1].to_uppercase(), f[2].to_string()))
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

/// What becomes of one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fate {
    /// Written byte for byte.
    Copy,
    /// Quantized here to the kind.
    Quantize(IqKind),
}

/// One tensor of the plan.
#[derive(Debug, Clone)]
pub struct Item {
    pub name: String,
    pub dims: Vec<u64>,
    pub src: GgmlType,
    /// The type of the bytes this tool writes.
    pub dst: GgmlType,
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub fate: Fate,
    /// Why a tensor is copied, or the type llama-quantize will give it.
    pub note: String,
}

/// The GGML type a kind writes.
#[must_use]
pub fn target_type(k: IqKind) -> GgmlType {
    match k {
        IqKind::Iq1S => GgmlType::IQ1_S,
        IqKind::Iq2Xxs => GgmlType::IQ2_XXS,
    }
}

fn kind_of(rule: &str) -> Option<IqKind> {
    match rule {
        "IQ1_S" => Some(IqKind::Iq1S),
        "IQ2_XXS" => Some(IqKind::Iq2Xxs),
        _ => None,
    }
}

/// The card's rates, in weights per second: the search kernel's rows in `docs/research/iq-quant-card.md` (derived
/// terms of `--plan`, never a measurement of this run; `tools/ref/quant-measure.sh` is the measurement).
const CARD_WEIGHTS_PER_S: [(IqKind, f64); 2] = [(IqKind::Iq1S, 1.09e9), (IqKind::Iq2Xxs, 8.3e8)];
/// The drive rates `--plan` derives its I/O terms from, bytes per second.
const READ_BPS: f64 = 1.6e9;
const WRITE_BPS: f64 = 1.5e9;

/// The derived wall's terms, seconds.
#[derive(Debug, Clone, Copy)]
pub struct Wall {
    pub card: f64,
    pub read: f64,
    pub write: f64,
    pub total: f64,
}

/// The whole plan.
#[derive(Debug)]
pub struct Plan {
    /// Per shard, its tensors in file order.
    pub shards: Vec<Vec<Item>>,
    /// Per shard, the layout of its output file.
    pub layouts: Vec<Layout>,
    /// The output files, shard by shard.
    pub outs: Vec<PathBuf>,
    pub quantized: usize,
    pub weights: [(IqKind, f64); 2],
    pub bytes_in: u64,
    pub bytes_out: u64,
    /// The output's file bytes, headers and padding included.
    pub out_file_bytes: u64,
}

/// Tensor `t`'s importance, `ne0` per expert, as llama-quantize computes it from the imatrix; an entry missing,
/// unusable or of another length is refused.
pub fn importance(im: &gguf::Gguf, t: &gguf::TensorInfo) -> Result<Vec<f32>> {
    let qw = imatrix_weights(im, &t.name).map_err(|e| Error::refused(e.to_string()))?;
    let (ne0, n_exp) = (t.dims[0], t.dims[2..].iter().product::<u64>());
    if qw.len() as u64 != ne0 * n_exp {
        return Err(Error::refused(format!(
            "imatrix {}: {} values, the tensor needs {} ({ne0} x {n_exp} experts)",
            t.name,
            qw.len(),
            ne0 * n_exp
        )));
    }
    Ok(qw)
}

/// The output names: `PREFIX.gguf` for a set of one file, `PREFIX-0000i-of-0000n.gguf` for a split set.
fn out_names(set: &gguf::Split, prefix: &str) -> Result<Vec<PathBuf>> {
    let n = set.shard_count();
    match set.shard(0).and_then(|g| g.value("split.count")) {
        None if n == 1 => Ok(vec![PathBuf::from(format!("{prefix}.gguf"))]),
        Some(c) if c.as_unsigned() == Some(n as u64) => Ok((0..n)
            .map(|i| PathBuf::from(format!("{prefix}-{:05}-of-{n:05}.gguf", i + 1)))
            .collect()),
        c => Err(Error::refused(format!(
            "the set opens as {n} files and its split.count is {c:?}: a companion file is not written here"
        ))),
    }
}

impl Plan {
    /// Decide every tensor of `set`; no file is written and none but the inputs is read.
    pub fn build(
        set: &gguf::Split,
        rules: &Rules,
        llama: Option<&LlamaPlan>,
        im: &gguf::Gguf,
        only: Option<IqKind>,
        out_prefix: &str,
    ) -> Result<Plan> {
        if let Some(lp) = llama
            && lp.0.len() != set.tensor_count()
        {
            let missing: Vec<&String> =
                lp.0.keys()
                    .filter(|k| set.find(k).is_none())
                    .take(3)
                    .collect();
            return Err(Error::refused(format!(
                "the llama plan names {} tensors, the set holds {}; plan rows the set does not hold: {missing:?}",
                lp.0.len(),
                set.tensor_count()
            )));
        }
        let mut p = Plan {
            shards: Vec::new(),
            layouts: Vec::new(),
            outs: out_names(set, out_prefix)?,
            quantized: 0,
            weights: [(IqKind::Iq1S, 0.0), (IqKind::Iq2Xxs, 0.0)],
            bytes_in: 0,
            bytes_out: 0,
            out_file_bytes: 0,
        };
        for i in 0..set.shard_count() {
            let shard = set.shard(i).expect("i < shard_count");
            let mut items = Vec::new();
            let mut decls = Vec::new();
            for t in shard.iter_tensors() {
                let item = decide(t, rules, llama, im, only)?;
                if let Fate::Quantize(k) = item.fate {
                    p.quantized += 1;
                    let w = (t.dims.iter().product::<u64>()) as f64;
                    p.weights[usize::from(k == IqKind::Iq2Xxs)].1 += w;
                    p.bytes_in += item.bytes_in;
                    p.bytes_out += item.bytes_out;
                }
                decls.push(TensorDecl {
                    name: item.name.clone(),
                    dims: item.dims.clone(),
                    type_id: item.dst.as_u32(),
                    nbytes: item.bytes_out,
                });
                items.push(item);
            }
            let layout =
                Layout::new(shard.keys(), decls).map_err(|e| Error::refused(e.to_string()))?;
            p.out_file_bytes += layout.file_len();
            p.layouts.push(layout);
            p.shards.push(items);
        }
        Ok(p)
    }

    /// The wall `--plan` derives for the card path: per tensor the read of its source under the card's time, then
    /// the output written after it; the copied tensors' read and write add. `None` with no quantized tensor.
    #[must_use]
    pub fn derived_wall(&self) -> Option<Wall> {
        if self.quantized == 0 {
            return None;
        }
        let card: f64 = CARD_WEIGHTS_PER_S
            .iter()
            .zip(&self.weights)
            .map(|(r, w)| w.1 / r.1)
            .sum();
        let copied: u64 = self
            .shards
            .iter()
            .flatten()
            .filter(|i| i.fate == Fate::Copy)
            .map(|i| i.bytes_in)
            .sum();
        let read = (self.bytes_in + copied) as f64 / READ_BPS;
        let write = self.out_file_bytes as f64 / WRITE_BPS;
        Some(Wall {
            card,
            read,
            write,
            total: card.max(self.bytes_in as f64 / READ_BPS) + copied as f64 / READ_BPS + write,
        })
    }

    /// What `--plan` prints; `free` is the free bytes of the output's drive, when it could be read.
    #[must_use]
    pub fn render(&self, a: &Args, free: Option<u64>) -> String {
        let mut s = String::new();
        let gb = |b: u64| b as f64 / 1e9;
        let n_tensors: usize = self.shards.iter().map(Vec::len).sum();
        let _ = writeln!(
            s,
            "bloomery-quantize --plan: {} shard(s), {n_tensors} tensors; source type, target type, bytes in -> out, fate",
            self.shards.len()
        );
        for item in self.shards.iter().flatten() {
            let fate = match item.fate {
                Fate::Quantize(_) => "quantized here".to_string(),
                Fate::Copy => format!("copied ({})", item.note),
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
        let weights = self.weights[0].1 + self.weights[1].1;
        let _ = writeln!(
            s,
            "quantized here: {} tensors, {weights:.3e} weights (IQ1_S {:.3e}, IQ2_XXS {:.3e}), {:.2} GB in -> {:.2} GB out; the rest copied",
            self.quantized,
            self.weights[0].1,
            self.weights[1].1,
            gb(self.bytes_in),
            gb(self.bytes_out)
        );
        match (a.cpu, self.derived_wall()) {
            (true, _) => {
                let _ = writeln!(
                    s,
                    "wall: not derived for --cpu (no host rate has been measured)"
                );
            }
            (false, Some(w)) => {
                let _ = writeln!(
                    s,
                    "wall [derived: card {:.0} s at the docs' kernel rates, source read {:.0} s at 1.6 GB/s under it, output write {:.0} s at 1.5 GB/s after it]: {:.0} s; the measurement is tools/ref/quant-measure.sh",
                    w.card, w.read, w.write, w.total
                );
            }
            (false, None) => {
                let _ = writeln!(s, "wall: nothing is quantized here");
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
        let split = if self.shards.len() > 1 {
            " --keep-split"
        } else {
            ""
        };
        let _ = writeln!(
            s,
            "next: llama-quantize --allow-requantize --imatrix {} --tensor-type-file {}{split} <the same flags as the quantize run this replaces: --output-tensor-type, --token-embedding-type> {} <final output> <ftype>",
            a.imatrix.display(),
            a.rules.display(),
            self.outs[0].display()
        );
        s
    }
}

/// The plan's row of one tensor.
fn decide(
    t: &gguf::TensorInfo,
    rules: &Rules,
    llama: Option<&LlamaPlan>,
    im: &gguf::Gguf,
    only: Option<IqKind>,
) -> Result<Item> {
    let rule = rules.type_of(&t.name);
    let routed = t.name.ends_with("_exps.weight") && t.dims.len() >= 3;
    if routed && let Some(lp) = llama {
        cross_check(t, rule, lp)?;
    }
    let copy = |note: String| Item {
        name: t.name.clone(),
        dims: t.dims.clone(),
        src: t.ty,
        dst: t.ty,
        bytes_in: t.nbytes,
        bytes_out: t.nbytes,
        fate: Fate::Copy,
        note,
    };
    let Some(kind) = rule.and_then(kind_of) else {
        return Ok(copy(match rule {
            Some(r) => format!("llama-quantize types it {r}"),
            None => "no rule: llama-quantize's own".to_string(),
        }));
    };
    if !routed {
        return Ok(copy(format!(
            "rule {kind:?} on a tensor that is no routed-expert stack: llama-quantize's"
        )));
    }
    if only.is_some_and(|o| o != kind) {
        return Ok(copy(format!("--type leaves {kind:?} to llama-quantize")));
    }
    let target = target_type(kind);
    if t.ty == target {
        return Ok(copy(format!("already {target:?}")));
    }
    if !SOURCE_TYPES.contains(&t.ty) {
        return Err(Error::refused(format!(
            "{} is {:?}: the oracle covers F32, F16, BF16 and MXFP4 sources, and a type it does not cover is not quantized here",
            t.name, t.ty
        )));
    }
    let ne0 = t.dims[0] as usize;
    if !ne0.is_multiple_of(256) {
        return Err(Error::refused(format!(
            "{}: ne0 {ne0} is not a multiple of 256",
            t.name
        )));
    }
    row_bytes(t.ty, ne0).ok_or_else(|| {
        Error::refused(format!(
            "{}: {:?} rows of {ne0} values are not read",
            t.name, t.ty
        ))
    })?;
    importance(im, t)?;
    let rows: u64 = t.dims[1..].iter().product();
    Ok(Item {
        name: t.name.clone(),
        dims: t.dims.clone(),
        src: t.ty,
        dst: target,
        bytes_in: t.nbytes,
        bytes_out: rows * (ne0 / 256 * kind.block_bytes()) as u64,
        fate: Fate::Quantize(kind),
        note: String::new(),
    })
}

/// A routed stack's rules type against llama-quantize's own plan.
fn cross_check(t: &gguf::TensorInfo, rule: Option<&str>, lp: &LlamaPlan) -> Result<()> {
    let Some((lty, ldims)) = lp.0.get(&t.name) else {
        return Err(Error::refused(format!(
            "the llama plan has no row for {}",
            t.name
        )));
    };
    if *ldims != plan_dims(&t.dims) {
        return Err(Error::refused(format!(
            "{}: the llama plan's dims {ldims}, the set's {}",
            t.name,
            plan_dims(&t.dims)
        )));
    }
    let iq = |s: &str| kind_of(s).is_some();
    let rule_iq = rule.filter(|r| iq(r));
    let agree = match rule_iq {
        Some(r) => r == lty,
        // a stack the rules leave alone must not be one llama-quantize sends to an IQ type, unless it is one already
        None => !iq(lty) || *lty == format!("{:?}", t.ty).to_uppercase(),
    };
    if agree {
        Ok(())
    } else {
        Err(Error::refused(format!(
            "{}: the rules type it {}, llama-quantize's own plan {lty}: the two must agree (a rule for every routed stack)",
            t.name,
            rule.unwrap_or("nothing")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_are_spelled_as_the_plan_spells_them() {
        assert_eq!(plan_dims(&[512, 4, 3]), "512x4x3");
        assert_eq!(plan_dims(&[256, 3, 1]), "256x3");
        assert_eq!(plan_dims(&[8, 1]), "8");
    }

    #[test]
    fn the_llama_plan_reads_four_columns_and_refuses_the_rest() {
        let p = LlamaPlan::parse("a\tiq1_s\t4x2\t10\nb\tF32\t8\t32\n", "plan.tsv").unwrap();
        assert_eq!(p.0["a"], ("IQ1_S".to_string(), "4x2".to_string()));
        assert!(
            LlamaPlan::parse("a\tx\t1\n", "plan.tsv")
                .unwrap_err()
                .to_string()
                .contains("plan.tsv line 1")
        );
        assert!(
            LlamaPlan::parse("a\tx\t1\t2\na\ty\t1\t2\n", "p")
                .unwrap_err()
                .to_string()
                .contains("planned twice")
        );
    }
}
