//! `clef_hidden` — Clef's backbone (a Qwen3.5 dense `qwen35` file) as a
//! prompt-only pass: every position's final-norm hidden state out, no head.
//!
//!     clef_hidden --model <gguf> --ids <file> --out <file>
//!                 [--count N] [--ctx C] [--ubatch U] [--warm W]
//!                 [--row-ids <file> --rows-out <file>]
//!
//! The ids file holds one decimal id a line; `--count` takes its first N
//! (default every id). The prompt runs through `Qwen35moeModel::prefill_hidden`
//! on the auto path (GEMM ubatches of up to U ids, a tail of at most eight one
//! pass), and `--out` gets the hidden states: f32 little-endian, row-major
//! `[positions][width]`, row `t` the final-norm state after position `t`
//! (llama.cpp's `result_norm`, HF's `last_hidden_state`). Defaults: C the
//! prompt's length, U 4096 clipped to C, W 0. Each of the W warm calls
//! runs the same prompt from a `reset` and is discarded; the recorded call
//! follows from a `reset` too. The `clef hidden` record names the positions,
//! the width, the bytes written and the recorded call's wall, from the ids'
//! upload to the last row's readback (a functional figure, not a timed
//! measurement: no lease).
//!
//! `--row-ids` names a second ids file: its ids' rows of `output.weight`,
//! dequantized on the host (`model::arch::qwen35moe::output_rows`), go to `--rows-out`
//! in the same layout, `[ids][width]`, and a `clef rows` record follows.
//! A flag given twice takes its last value; an unknown flag, a missing
//! value, or one of the row flags without the other is refused by name.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("clef_hidden: built without the `gpu` feature; see `just clef-hidden`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::record::at_main("clef_hidden", bloomery_gpu_gates::record::CLEF_HIDDEN_BIN);
    bloomery_gpu_gates::exit_with("clef_hidden", run::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen35_open.rs"]
mod qwen35_open;

#[cfg(feature = "gpu")]
mod run {
    use super::qwen35_open;
    use bloomery_gpu::arch::qwen3moe::PrefillPath;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu_gates::GateError;
    use bloomery_gpu_gates::record::{CLEF_HIDDEN, CLEF_ROWS, Record};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    struct Args {
        model: PathBuf,
        ids: PathBuf,
        out: PathBuf,
        count: Option<usize>,
        ctx: Option<usize>,
        ubatch: usize,
        warm: usize,
        rows: Option<(PathBuf, PathBuf)>,
    }

    fn parse() -> Result<Args, GateError> {
        let mut a = std::env::args().skip(1);
        let (mut model, mut ids, mut out) = (None, None, None);
        let (mut count, mut ctx, mut ubatch, mut warm) = (None, None, UBATCH, 0usize);
        let (mut row_ids, mut rows_out) = (None, None);
        let num = |flag: &str, v: String| -> Result<usize, GateError> {
            match v.parse::<usize>() {
                Ok(n) if n > 0 || flag == "--warm" => Ok(n),
                _ => Err(format!("{flag} takes a whole number of at least 1, not {v:?}").into()),
            }
        };
        while let Some(flag) = a.next() {
            let value = a
                .next()
                .ok_or_else(|| format!("{flag}: a value is due after it"))?;
            match flag.as_str() {
                "--model" => model = Some(PathBuf::from(value)),
                "--ids" => ids = Some(PathBuf::from(value)),
                "--out" => out = Some(PathBuf::from(value)),
                "--count" => count = Some(num(&flag, value)?),
                "--ctx" => ctx = Some(num(&flag, value)?),
                "--ubatch" => ubatch = num(&flag, value)?,
                "--warm" => warm = num(&flag, value)?,
                "--row-ids" => row_ids = Some(PathBuf::from(value)),
                "--rows-out" => rows_out = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown flag {flag:?}").into()),
            }
        }
        let rows = match (row_ids, rows_out) {
            (Some(i), Some(o)) => Some((i, o)),
            (None, None) => None,
            _ => return Err("--row-ids and --rows-out go together".into()),
        };
        let need = |v: Option<PathBuf>, flag: &str| {
            v.ok_or_else(|| GateError::from(format!("{flag} is required")))
        };
        Ok(Args {
            model: need(model, "--model")?,
            ids: need(ids, "--ids")?,
            out: need(out, "--out")?,
            count,
            ctx,
            ubatch,
            warm,
            rows,
        })
    }

    /// `values` as f32 little-endian into a new file at `path`; the byte count.
    fn write_f32(path: &Path, values: &[f32]) -> Result<u64, GateError> {
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?,
        );
        for v in values {
            f.write_all(&v.to_le_bytes())?;
        }
        f.flush().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(u64::try_from(values.len() * 4)?)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let a = parse()?;
        let ids = qwen35_open::read_ids(&a.ids, a.count)?;
        let ctx = a.ctx.unwrap_or(ids.len());
        let (mut engine, split) = qwen35_open::open(&a.model, ctx, a.ubatch.min(ctx))?;
        for _ in 0..a.warm {
            engine.reset()?;
            engine.prefill_hidden(&ids, PrefillPath::Auto)?;
        }
        engine.reset()?;
        let t0 = Instant::now();
        let hidden = engine.prefill_hidden(&ids, PrefillPath::Auto)?;
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let width = hidden.len() / ids.len();
        let bytes = write_f32(&a.out, &hidden)?;
        Record::new(&CLEF_HIDDEN)
            .u("n", u64::try_from(ids.len())?)
            .u("width", u64::try_from(width)?)
            .u("bytes", bytes)
            .f("ms", ms)
            .f("tok/s", ids.len() as f64 / (ms / 1e3))
            .print();
        if let Some((row_ids, rows_out)) = &a.rows {
            let ids = qwen35_open::read_ids(row_ids, None)?;
            let gguf = split.shard(0).ok_or("the model file has no shard 0")?;
            let rows = model::arch::qwen35moe::output_rows(gguf, &ids)?;
            let bytes = write_f32(rows_out, &rows.data)?;
            Record::new(&CLEF_ROWS)
                .u("ids", u64::try_from(ids.len())?)
                .u("width", u64::try_from(rows.ne0)?)
                .u("bytes", bytes)
                .print();
        }
        Ok(())
    }
}
