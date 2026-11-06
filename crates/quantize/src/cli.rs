//! The arguments and the usage text of `bloomery-quantize`.

use crate::resolve::{Ftype, parse_type};
use crate::{Error, Result};
use gguf::quant::GgmlType;
use std::path::PathBuf;

/// What `--help` prints.
pub const USAGE: &str = "\
bloomery-quantize: llama-quantize's job, byte for byte, in one pass: the routed-expert stacks on the card (or the
host's cores), every other tensor on the host's cores, the metadata and the shard layout as llama-quantize writes
them (docs/quantize.md).

usage:
  bloomery-quantize [options] <in first shard> <out> <ftype> [nthreads]
  bloomery-quantize --help | --version

  <in first shard>       the first file of the input set
  <out>                  the output file; with --keep-split its prefix (a trailing .gguf is dropped) and the shards
                         are PREFIX-0000i-of-0000n.gguf; an existing file is refused, never overwritten
  <ftype>                Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ2_XXS or IQ1_S
  [nthreads]             accepted for llama-quantize's command line and ignored: the host rules run on the resident
                         pool, one thread a core
  --imatrix FILE         the GGUF importance matrix llama-imatrix wrote (IQ1_S and IQ2_XXS need it)
  --tensor-type-file FILE
                         llama-quantize's type rules (regex=type tokens)
  --output-tensor-type T the type of output.weight
  --token-embedding-type T
                         the type of token_embd.weight
  --allow-requantize     quantize a tensor whose source is already a quantized type (MXFP4 included)
  --keep-split           write one output shard per input shard, as llama-quantize does
  --plan                 print every tensor's source and target type, bytes in and out, the derived wall and the
                         room needed; write nothing
  --llama-plan TSV       llama-quantize's own --dry-run plan (name TAB type TAB dims TAB bytes): every tensor's type,
                         dims and bytes must equal this plan's, or the run is refused
  --check N              run N rows of every card-quantized tensor through the host rule as well and stop by name
                         (tensor, expert, row, byte) on the first byte that differs
  --cpu                  run every tensor on the host's cores; open no card
  --card N               the card to use, by CUDA ordinal (default: the first)
  --batch N              experts per card call (default 32)

Types are written for F32, Q4_K, Q5_K, Q6_K, Q8_0, IQ1_S and IQ2_XXS and the architecture kimi-k3; anything else is
refused by name.

exit: 0 done; 1 a refusal, named (a source row that is not finite, an imatrix entry that is unusable, a type this
quantizer does not write, too little room, a --check or --llama-plan difference); 64 a usage error; 66 a missing
input file.";

/// The arguments of a run.
#[derive(Debug, Clone)]
pub struct Args {
    pub first: PathBuf,
    pub out: String,
    pub ftype: Ftype,
    pub imatrix: Option<String>,
    pub rules: Option<PathBuf>,
    pub output_tensor_type: Option<GgmlType>,
    pub token_embedding_type: Option<GgmlType>,
    pub allow_requantize: bool,
    pub keep_split: bool,
    pub llama_plan: Option<PathBuf>,
    pub plan: bool,
    pub check: usize,
    pub cpu: bool,
    pub card: Option<usize>,
    pub batch: usize,
}

/// What the command line asks for.
#[derive(Debug)]
pub enum Parsed {
    Help,
    Version,
    Run(Args),
}

impl Args {
    /// The directory the output goes under: the output path's parent, `.` for a bare name.
    #[must_use]
    pub fn out_dir(&self) -> PathBuf {
        match std::path::Path::new(&self.out).parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        }
    }
}

fn usage(what: impl std::fmt::Display) -> Error {
    Error::Usage(format!("{what}; bloomery-quantize --help prints the usage"))
}

/// Read the arguments after the program name.
pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Parsed> {
    let mut it = args.into_iter();
    let mut pos = Vec::new();
    let (mut imatrix, mut rules, mut llama_plan) = (None, None, None);
    let (mut out_t, mut emb_t) = (None, None);
    let (mut allow, mut keep, mut plan, mut cpu) = (false, false, false, false);
    let (mut check, mut card, mut batch) = (0usize, None, 32usize);
    while let Some(a) = it.next() {
        let value = |it: &mut dyn Iterator<Item = String>| {
            it.next().ok_or_else(|| usage(format!("{a} needs a value")))
        };
        let ty = |s: String| parse_type(&s).map_err(|e| usage(format!("{a}: {e}")));
        match a.as_str() {
            "--help" | "-h" => return Ok(Parsed::Help),
            "--version" => return Ok(Parsed::Version),
            "--imatrix" => imatrix = Some(value(&mut it)?),
            "--tensor-type-file" => rules = Some(PathBuf::from(value(&mut it)?)),
            "--llama-plan" => llama_plan = Some(PathBuf::from(value(&mut it)?)),
            "--output-tensor-type" => out_t = Some(ty(value(&mut it)?)?),
            "--token-embedding-type" => emb_t = Some(ty(value(&mut it)?)?),
            "--allow-requantize" => allow = true,
            "--keep-split" => keep = true,
            "--plan" => plan = true,
            "--cpu" => cpu = true,
            "--check" => {
                check = value(&mut it)?
                    .parse()
                    .map_err(|_| usage("--check takes a row count"))?;
            }
            "--card" => {
                card = Some(
                    value(&mut it)?
                        .parse()
                        .map_err(|_| usage("--card takes a CUDA ordinal"))?,
                );
            }
            "--batch" => {
                batch = value(&mut it)?
                    .parse()
                    .map_err(|_| usage("--batch takes an expert count"))?;
                if batch == 0 {
                    return Err(usage("--batch 0"));
                }
            }
            o if o.starts_with('-') => return Err(usage(format!("unknown argument {o}"))),
            _ => pos.push(a),
        }
    }
    if pos.len() == 4 && pos[3].parse::<usize>().is_ok() {
        pos.pop();
    }
    let [first, out, ftype] = <[String; 3]>::try_from(pos).map_err(|p| {
        usage(format!(
            "three positional arguments (<in first shard> <out> <ftype>), got {}",
            p.len()
        ))
    })?;
    if cpu && card.is_some() {
        return Err(usage("--cpu opens no card: it takes no --card"));
    }
    if cpu && check > 0 {
        return Err(usage(
            "--check compares the card with the host rule: --cpu is the host rule",
        ));
    }
    let out = match (keep, out.strip_suffix(".gguf")) {
        (true, Some(p)) => p.to_string(),
        _ => out,
    };
    Ok(Parsed::Run(Args {
        first: PathBuf::from(first),
        out,
        ftype: Ftype::parse(&ftype)?,
        imatrix,
        rules,
        output_tensor_type: out_t,
        token_embedding_type: emb_t,
        allow_requantize: allow,
        keep_split: keep,
        llama_plan,
        plan,
        check,
        cpu,
        card,
        batch,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: &[&str]) -> Result<Parsed> {
        parse(a.iter().map(|s| (*s).to_string()))
    }

    #[test]
    fn a_full_command_line() {
        let Parsed::Run(a) = run(&[
            "in.gguf",
            "out/p.gguf",
            "q4_k_m",
            "--imatrix",
            "im.gguf",
            "--tensor-type-file",
            "r.txt",
            "--output-tensor-type",
            "q6_k",
            "--token-embedding-type",
            "Q4_K",
            "--allow-requantize",
            "--keep-split",
            "--check",
            "4",
            "--card",
            "1",
            "--batch",
            "8",
            "--plan",
        ])
        .unwrap() else {
            panic!("a run")
        };
        assert_eq!(
            (a.check, a.card, a.batch, a.plan, a.ftype),
            (4, Some(1), 8, true, Ftype::Q4KM)
        );
        assert_eq!(a.out, "out/p", "--keep-split drops a trailing .gguf");
        assert_eq!(a.out_dir(), PathBuf::from("out"));
        assert_eq!(a.output_tensor_type, Some(GgmlType::Q6_K));
        assert_eq!(a.token_embedding_type, Some(GgmlType::Q4_K));
        assert!(a.allow_requantize && a.keep_split);
    }

    #[test]
    fn refusals_are_usage_errors() {
        for bad in [
            &["in", "out"][..],
            &["in", "out", "q4_k_m", "--cpu", "--card", "0"],
            &["in", "out", "q4_k_m", "--cpu", "--check", "2"],
            &["in", "out", "q2_k"],
            &["in", "out", "q4_k_m", "--output-tensor-type", "q3_k"],
            &["in", "out", "q4_k_m", "--batch", "0"],
            &["in", "out", "q4_k_m", "--bogus"],
            &["in", "out", "q4_k_m", "--imatrix"],
        ] {
            let e = run(bad).unwrap_err();
            assert_eq!(e.status(), 64, "{bad:?}: {e}");
        }
        assert!(matches!(run(&["--help"]).unwrap(), Parsed::Help));
        assert!(matches!(run(&["x", "--version"]).unwrap(), Parsed::Version));
    }
}
