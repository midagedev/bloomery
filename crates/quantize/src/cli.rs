//! The arguments and the usage text of `bloomery-quantize`.

use crate::{Error, Result};
use gguf::iq_quant::IqKind;
use std::path::PathBuf;

/// What `--help` prints.
pub const USAGE: &str = "\
bloomery-quantize: IQ1_S and IQ2_XXS quantization of a GGUF set's routed-expert stacks, byte-identical to ggml's,
on the card or on the host's cores; llama-quantize then copies them and quantizes the trunk (docs/quantize.md).

usage:
  bloomery-quantize <in first shard> <out prefix> --imatrix FILE --tensor-type-file FILE [options]
  bloomery-quantize --help | --version

  <in first shard>       the first file of the input set (MXFP4, F16, BF16 or F32 expert stacks)
  <out prefix>           the output set: PREFIX.gguf, or PREFIX-0000i-of-0000n.gguf for a split set; an
                         existing file is refused, never overwritten
  --imatrix FILE         the GGUF importance matrix llama-imatrix wrote (required)
  --tensor-type-file FILE
                         llama-quantize's type rules (regex=type tokens); the stacks they type IQ1_S or IQ2_XXS
                         are quantized here (required)
  --type IQ1_S|IQ2_XXS   quantize only the stacks the rules type this; llama-quantize does the others
  --plan                 print every tensor's source and target type, bytes in and out, quantized-here or
                         copied, the derived wall, the room needed and the llama-quantize command to run next;
                         write nothing
  --check N              run N rows of every quantized tensor through the host rule as well and stop by name
                         (tensor, expert, row, byte) on the first byte that differs
  --cpu                  run the host rule on all cores; open no card
  --card N               the card to use, by CUDA ordinal (default: the first)
  --batch N              experts per card call (default 32)
  --llama-plan TSV       llama-quantize's own --dry-run plan (name TAB type TAB dims TAB bytes): cross-checked
                         against the rules for every routed stack

exit: 0 done; 1 a refusal, named (a source row that is not finite, an imatrix entry that is unusable, a type no
oracle covers, too little room, a --check difference); 64 a usage error; 66 a missing input file.";

/// The arguments of a run.
#[derive(Debug, Clone)]
pub struct Args {
    pub first: PathBuf,
    pub out: String,
    pub imatrix: PathBuf,
    pub rules: PathBuf,
    pub llama_plan: Option<PathBuf>,
    pub only: Option<IqKind>,
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
    /// The directory the output set goes under: the prefix's parent, `.` for a bare prefix.
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
    let (mut pos, mut imatrix, mut rules, mut llama_plan) = (Vec::new(), None, None, None);
    let (mut only, mut plan, mut check, mut cpu, mut card, mut batch) =
        (None, false, 0usize, false, None, 32usize);
    while let Some(a) = it.next() {
        let value = |it: &mut dyn Iterator<Item = String>| {
            it.next().ok_or_else(|| usage(format!("{a} needs a value")))
        };
        match a.as_str() {
            "--help" | "-h" => return Ok(Parsed::Help),
            "--version" => return Ok(Parsed::Version),
            "--imatrix" => imatrix = Some(PathBuf::from(value(&mut it)?)),
            "--tensor-type-file" => rules = Some(PathBuf::from(value(&mut it)?)),
            "--llama-plan" => llama_plan = Some(PathBuf::from(value(&mut it)?)),
            "--type" => {
                only = Some(match value(&mut it)?.to_ascii_uppercase().as_str() {
                    "IQ1_S" => IqKind::Iq1S,
                    "IQ2_XXS" => IqKind::Iq2Xxs,
                    o => return Err(usage(format!("--type {o}: IQ1_S or IQ2_XXS"))),
                });
            }
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
    let [first, out] = <[String; 2]>::try_from(pos).map_err(|p| {
        usage(format!(
            "two positional arguments (<in first shard> <out prefix>), got {}",
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
    Ok(Parsed::Run(Args {
        first: PathBuf::from(first),
        out,
        imatrix: imatrix.ok_or_else(|| usage("--imatrix is required"))?,
        rules: rules.ok_or_else(|| usage("--tensor-type-file is required"))?,
        llama_plan,
        only,
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
            "out/p",
            "--imatrix",
            "im.gguf",
            "--tensor-type-file",
            "r.txt",
            "--type",
            "iq1_s",
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
            (a.check, a.card, a.batch, a.plan, a.only),
            (4, Some(1), 8, true, Some(IqKind::Iq1S))
        );
        assert_eq!(a.out_dir(), PathBuf::from("out"));
    }

    #[test]
    fn refusals_are_usage_errors() {
        for bad in [
            &["in", "out", "--imatrix", "i"][..],
            &["in", "--imatrix", "i", "--tensor-type-file", "r"],
            &[
                "in",
                "out",
                "--imatrix",
                "i",
                "--tensor-type-file",
                "r",
                "--cpu",
                "--card",
                "0",
            ],
            &[
                "in",
                "out",
                "--imatrix",
                "i",
                "--tensor-type-file",
                "r",
                "--cpu",
                "--check",
                "2",
            ],
            &[
                "in",
                "out",
                "--imatrix",
                "i",
                "--tensor-type-file",
                "r",
                "--type",
                "Q4_K",
            ],
            &[
                "in",
                "out",
                "--imatrix",
                "i",
                "--tensor-type-file",
                "r",
                "--batch",
                "0",
            ],
            &[
                "in",
                "out",
                "--imatrix",
                "i",
                "--tensor-type-file",
                "r",
                "--bogus",
            ],
            &["in", "out", "--imatrix"],
        ] {
            let e = run(bad).unwrap_err();
            assert_eq!(e.status(), 64, "{bad:?}: {e}");
        }
        assert!(matches!(run(&["--help"]).unwrap(), Parsed::Help));
        assert!(matches!(run(&["x", "--version"]).unwrap(), Parsed::Version));
    }
}
