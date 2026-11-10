//! What every clause file of the kernels' gate shares: the layer, the fault
//! expectations, the band harness of the mat-vec families and the refusal
//! loop. A clause file owns its kernel's launches and its case table; this
//! module owns what would otherwise be a copy per file.

use std::process::Command;

use bloomery_device::conformance::{Clause, Verdict, why};
use bloomery_device::fault::FaultSite;
use bloomery_device::{Buffer, Device, DeviceError, Elem};
use bloomery_metal::MetalBuffer;
use bloomery_metal::encode::ComputeEncoder;
use bloomery_metal::kernels::Quant;

use super::fixture::Fixture;
use super::oracle::{self, Kind};
use super::{Harness, Launch, ModeSpec, SENTINEL_BITS, flag};

/// The layer the clauses launch in: a model layer, so a raise is told apart
/// from a launch that names none. Nothing reads the number; every clause
/// clears the fault word before it launches.
pub const LAYER: u32 = 5;

/// A tensor a clause reads: its name, the type, the columns and the rows the
/// clause takes of it.
#[derive(Clone, Copy)]
pub struct Tensor {
    pub name: &'static str,
    pub quant: Quant,
    pub cols: usize,
    pub rows: usize,
}

impl Tensor {
    pub const fn q4k(name: &'static str, cols: usize, rows: usize) -> Self {
        Self {
            name,
            quant: Quant::Q4K,
            cols,
            rows,
        }
    }

    pub const fn q6k(name: &'static str, cols: usize, rows: usize) -> Self {
        Self {
            name,
            quant: Quant::Q6K,
            cols,
            rows,
        }
    }

    pub const fn a4(name: &'static str, cols: usize, rows: usize) -> Self {
        Self {
            name,
            quant: Quant::A4,
            cols,
            rows,
        }
    }

    pub fn bytes<'f>(&self, fixture: &'f Fixture) -> Result<&'f [u8], String> {
        fixture.rows(self.name, self.quant.ggml(), self.cols, self.rows)
    }

    pub fn label(&self) -> String {
        format!("{} {}x{}", self.name, self.rows, self.cols)
    }
}

/// The gain a norm clause uses for a tensor of `cols` columns: the fixture's
/// own where it has one of that width, else a positive synthetic one.
pub fn gain_for(
    fixture: &Fixture,
    name: Option<&str>,
    cols: usize,
    seed: u64,
) -> Result<Vec<f32>, String> {
    match name {
        Some(name) => fixture.f32s(name, cols),
        None => Ok(oracle::activations(cols, seed, 0.5)
            .iter()
            .map(|g| 1.0 + g.abs())
            .collect()),
    }
}

/// Run `launch` and hold the output to the oracle's band: the worst ratio of
/// the error to the band.
pub fn run_checked(h: &Harness<'_>, launch: &Launch<'_>) -> Result<f64, String> {
    let (effective, kind, base): (Vec<f64>, Kind, Option<&[f32]>) = match &launch.mode {
        ModeSpec::Store => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Store,
            None,
        ),
        ModeSpec::Norm { gain, eps, .. } => {
            (oracle::rms_norm(launch.x, gain, *eps), Kind::Norm, None)
        }
        ModeSpec::Residual => (
            launch.x.iter().map(|&v| f64::from(v)).collect(),
            Kind::Residual,
            Some(launch.y0),
        ),
    };
    let reference = oracle::reference(
        launch.quant,
        launch.weights,
        launch.rows,
        launch.cols,
        &effective,
    )?;
    let got = h.matvec(launch)?;
    if let Some(fault) = h.fault()? {
        return Err(format!("the launch raised {fault}"));
    }
    let band = oracle::band(
        launch.quant,
        launch.cols / launch.quant.values_per_block(),
        kind,
    );
    oracle::check(&got, &reference, band, kind, base)
}

pub fn sentinel_rows(rows: usize) -> Vec<f32> {
    vec![f32::from_bits(SENTINEL_BITS); rows]
}

/// The note a mat-vec clause prints: each tensor's worst error over its band.
pub fn noted(parts: Vec<String>) -> Clause {
    Ok(Verdict::Noted(format!(
        "worst error over its band: {}",
        parts.join(", ")
    )))
}

/// The fault word names `site` in [`LAYER`], and only that site.
pub fn expect_fault(h: &Harness<'_>, what: &str, site: FaultSite) -> Result<(), String> {
    match h.fault()? {
        Some(f)
            if f.layer == LAYER && f.code == site as u32 && f.sites == 1u32 << (site as u32) =>
        {
            Ok(())
        }
        Some(f) => Err(format!(
            "{what} raised {f}, not {} in layer {LAYER}",
            site.name()
        )),
        None => Err(format!("{what} raised nothing, not {}", site.name())),
    }
}

/// The fault word is clean.
pub fn expect_clean(h: &Harness<'_>, what: &str) -> Result<(), String> {
    if let Some(fault) = h.fault()? {
        return Err(format!("{what} raised {fault}"));
    }
    Ok(())
}

/// Two f32 slices hold the same words.
pub fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits())
}

/// The first index where `got` and `want` differ, with both words.
pub fn first_diff<T: PartialEq + Copy>(got: &[T], want: &[T]) -> Option<(usize, T, T)> {
    got.iter()
        .zip(want)
        .enumerate()
        .find_map(|(i, (g, w))| (*g != *w).then_some((i, *g, *w)))
}

/// An error over its band: 0 for no error where the band is 0, infinity for
/// any error there.
pub fn ratio(diff: f64, band: f64) -> f64 {
    if band > 0.0 {
        diff / band
    } else if diff == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// The rows of `rows` laid `stride` apart, the words between them `pad`.
pub fn strided(rows: &[Vec<f32>], stride: usize, pad: f32) -> Vec<f32> {
    let mut out = vec![pad; rows.len() * stride];
    for (r, row) in rows.iter().enumerate() {
        out[r * stride..r * stride + row.len()].copy_from_slice(row);
    }
    out
}

/// The input the head-normed mat-vec dots with: the gated numerator times the
/// scale of the head each column belongs to.
pub fn head_scaled(
    g: &[f32],
    partials: &[f32],
    heads: usize,
    cols_per_head: usize,
    partials_per_head: usize,
    eps: f32,
) -> Vec<f64> {
    let scales: Vec<f64> = (0..heads)
        .map(|head| {
            let head_partials: Vec<f64> = partials
                [head * partials_per_head..(head + 1) * partials_per_head]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            oracle::head_scale(&head_partials, cols_per_head, f64::from(eps))
        })
        .collect();
    g.iter()
        .enumerate()
        .map(|(c, &v)| f64::from(v) * scales[c / cols_per_head])
        .collect()
}

/// One command buffer with the launch encoded by `launch`, submitted; the
/// `DeviceError` is kept, so a refusal is told apart by name.
fn encode(
    h: &Harness<'_>,
    launch: impl FnOnce(&mut ComputeEncoder<'_>) -> Result<(), DeviceError>,
) -> Result<(), DeviceError> {
    let mut recording = h.stream.begin()?;
    {
        let mut encoder = recording.compute_encoder()?;
        launch(&mut encoder)?;
        encoder.end();
    }
    h.stream.submit(recording)
}

/// One refusal case: the name for the failure, the words the refusal must
/// name, a word it must not name, and the launch with one argument bad.
pub struct BadArg<'a, B> {
    pub what: &'static str,
    pub needle: &'static str,
    /// A second word the refusal must name, when it names two things.
    pub also: &'static str,
    /// A word the refusal must not name: an argument check that masks another.
    pub mask: &'static str,
    pub edit: Edit<'a, B>,
}

/// The launch with one argument bad.
pub type Edit<'a, B> =
    Box<dyn FnOnce(&mut B, &mut ComputeEncoder<'_>) -> Result<(), DeviceError> + 'a>;

impl<'a, B> BadArg<'a, B> {
    pub fn new(what: &'static str, needle: &'static str, edit: Edit<'a, B>) -> Self {
        Self {
            what,
            needle,
            also: "",
            mask: "",
            edit,
        }
    }

    /// The refusal names `also` beside the needle.
    pub fn naming(mut self, also: &'static str) -> Self {
        self.also = also;
        self
    }

    /// The refusal does not name `mask`: another argument's check ran first.
    pub fn masking(mut self, mask: &'static str) -> Self {
        self.mask = mask;
        self
    }
}

/// The launch of one case, as an error: refused by `kernel` by name, with the
/// needle in the detail.
fn expect_refusal(
    refused: Result<(), DeviceError>,
    kernel: &'static str,
    case: (&str, &str, &str, &str),
) -> Result<(), String> {
    let (what, needle, also, mask) = case;
    match refused {
        Ok(()) => Err(format!("{what} was accepted")),
        Err(DeviceError::Range { what: got, detail }) => {
            if got != kernel {
                return Err(format!(
                    "{what} was refused as {got}, not {kernel}: {detail}"
                ));
            }
            if !detail.contains(needle) {
                return Err(format!(
                    "{what} was refused without naming {needle}: {detail}"
                ));
            }
            if !also.is_empty() && !detail.contains(also) {
                return Err(format!(
                    "{what} was refused without naming {also}: {detail}"
                ));
            }
            if !mask.is_empty() && detail.contains(mask) {
                return Err(format!(
                    "{what} named {mask}, another argument's check: {detail}"
                ));
            }
            Ok(())
        }
        Err(other) => Err(format!("{what} was refused, not as a named range: {other}")),
    }
}

/// Every case is refused by name and writes nothing: after each, `unchanged`
/// reads the outputs back. Returns the case count. The well-formed launch is
/// [`accepted_writes`], run once after this.
pub fn refuse_launches<B>(
    h: &Harness<'_>,
    kernel: &'static str,
    bufs: &mut B,
    cases: Vec<BadArg<'_, B>>,
    unchanged: impl Fn(&Harness<'_>, &B) -> Result<(), String>,
) -> Result<usize, String> {
    let count = cases.len();
    for case in cases {
        let BadArg {
            what,
            needle,
            also,
            mask,
            edit,
        } = case;
        let refused = encode(h, |encoder| edit(bufs, encoder));
        expect_refusal(refused, kernel, (what, needle, also, mask))?;
        unchanged(h, bufs).map_err(|e| format!("{what}: {e}"))?;
    }
    Ok(count)
}

/// The well-formed launch is accepted, and `wrote` sees some output move: a
/// table whose every launch is refused proves nothing about the kernel.
pub fn accepted_writes<B>(
    h: &Harness<'_>,
    bufs: &mut B,
    launch: Edit<'_, B>,
    wrote: impl FnOnce(&Harness<'_>, &B) -> Result<bool, String>,
) -> Result<(), String> {
    encode(h, |encoder| launch(bufs, encoder))
        .map_err(|e| format!("the well-formed launch was refused: {e}"))?;
    let w = wrote(h, bufs)?;
    if !w {
        return Err("an accepted launch wrote nothing".to_string());
    }
    Ok(())
}

/// Every word of `buffer` still passes `keep`.
pub fn still<T: Elem + Default>(
    h: &Harness<'_>,
    buffer: &MetalBuffer<T>,
    keep: impl Fn(&T) -> bool,
) -> Result<(), String> {
    if moved(h, buffer, |v| !keep(v))? {
        return Err("a refused launch wrote".to_string());
    }
    Ok(())
}

/// Whether some word of `buffer` passes `keep`: what a well-formed launch
/// does to its outputs.
pub fn moved<T: Elem + Default>(
    h: &Harness<'_>,
    buffer: &MetalBuffer<T>,
    keep: impl Fn(&T) -> bool,
) -> Result<bool, String> {
    let mut back = vec![T::default(); buffer.len().get()];
    h.device
        .read(h.stream, buffer.whole().map_err(why)?, &mut back)
        .map_err(why)?;
    Ok(back.iter().any(keep))
}

/// What a child run of the gate under Metal's shader validation said on
/// stderr.
struct Validated {
    /// Whether the validation layer announced itself.
    started: bool,
    /// How many accesses it reported.
    reports: usize,
    /// The first report's lines, for a clause that fails on it.
    first: String,
    /// The child's own verdict and last words.
    exit: Result<(), String>,
}

/// The gate's own binary, run with `child` and a `--mutant` when one is named,
/// with Metal's shader validation on (its reports go to stderr).
fn under_validation(child: &[&str], mutant: Option<&str>) -> Result<Validated, String> {
    let exe = std::env::current_exe().map_err(|e| format!("the gate's own path: {e}"))?;
    let mut run = Command::new(exe);
    run.args(child);
    if let Some(name) = mutant {
        run.args(["--mutant", name]);
    }
    run.env("MTL_SHADER_VALIDATION", "1")
        .env("MTL_SHADER_VALIDATION_GLOBAL_MEMORY", "1")
        .env("MTL_SHADER_VALIDATION_REPORT_TO_STDERR", "1")
        .env("MTL_SHADER_VALIDATION_ENABLE_VERBOSE_LOGGING", "1");
    let out = run
        .output()
        .map_err(|e| format!("the child gate did not start: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let (reports, first) = validation_reports(&stderr);
    Ok(Validated {
        started: stderr.contains("Metal GPU Validation Enabled"),
        reports,
        first,
        exit: if out.status.success() {
            Ok(())
        } else {
            Err(format!(
                "the child exited {}: {}",
                out.status,
                stdout.lines().last().unwrap_or("no output")
            ))
        },
    })
}

/// The accesses outside a buffer that a shader validation stderr reports, and
/// the first report's first two lines. The layer writes a report in one of two
/// forms: a block between `=====` lines that names "Invalid Metal usage", or,
/// where it writes no such block (macOS 26.5), a paragraph its line
/// "Invalid device load …" or "Invalid device store …" opens.
fn validation_reports(stderr: &str) -> (usize, String) {
    let two = |lines: &mut dyn Iterator<Item = &str>| {
        lines
            .filter(|l| !l.trim().is_empty())
            .take(2)
            .collect::<Vec<_>>()
            .join(" / ")
    };
    let blocks: Vec<&str> = stderr
        .split("=====")
        .filter(|block| block.contains("Invalid Metal usage"))
        .collect();
    if let Some(block) = blocks.first() {
        return (blocks.len(), two(&mut block.lines()));
    }
    let lines: Vec<&str> = stderr.lines().collect();
    let opens: Vec<usize> = (0..lines.len())
        .filter(|&i| lines[i].starts_with("Invalid device "))
        .collect();
    let first = opens
        .first()
        .map(|&i| two(&mut lines[i..].iter().copied()))
        .unwrap_or_default();
    (opens.len(), first)
}

/// The launches the gate's `child` run makes read nothing outside their
/// buffers. Only Metal's shader validation reports such a read, and it must be
/// on when the process starts, so the launches run in child gates. The
/// control, the child with the `control` mutant (one that reads past its
/// weights), must be reported, or the validation checked nothing and the
/// clause is red; then the child with the parent's own `--mutant`, if any,
/// must report nothing and exit clean. `launches` names them in the verdict.
/// Returns the control's report count.
pub fn reads_inside(
    child: &[&str],
    args: &[String],
    control: &str,
    launches: &str,
) -> Result<usize, String> {
    let checked = under_validation(child, Some(control))?;
    if !checked.started || checked.reports == 0 {
        return Err(format!(
            "shader validation {} for the control, a kernel that reads past its weights: nothing \
             was checked",
            if checked.started {
                "reported nothing"
            } else {
                "did not start"
            }
        ));
    }
    let own = under_validation(child, flag(args, "--mutant"))?;
    if !own.started {
        return Err("shader validation did not start: nothing was checked".to_string());
    }
    if own.reports > 0 {
        return Err(format!(
            "{} accesses outside a buffer reported in the {launches}, the first: {}",
            own.reports, own.first
        ));
    }
    own.exit?;
    Ok(checked.reports)
}

#[cfg(test)]
mod tests {
    use super::validation_reports;

    /// Both forms of the layer's report are counted, the first named by its
    /// two opening lines, and a stderr that reports nothing counts none.
    #[test]
    fn validation_reports_reads_both_forms() {
        let blocks = "Metal GPU Validation Enabled\n=====\nInvalid Metal usage\nread past a \
                      buffer\n=====\n=====\nInvalid Metal usage\nanother\n=====\n";
        assert_eq!(
            validation_reports(blocks),
            (2, "Invalid Metal usage / read past a buffer".to_string())
        );
        let paragraphs = "2026-11-03 21:23:39.864 gate[4899:51860] Metal GPU Validation \
                          Enabled\nInvalid device load at offset 4678, executing kernel \
                          function: \"q4k_matvec\" \nbuffer: <unnamed>, length:1476, \
                          resident:Read Write\n\t* frame #0: k4_scale_words()\n\n\nInvalid \
                          device store at offset 64, executing kernel function: \"x\" \n\
                          buffer: <unnamed>, length:8, resident:Read Write\n";
        assert_eq!(
            validation_reports(paragraphs),
            (
                2,
                "Invalid device load at offset 4678, executing kernel function: \
                 \"q4k_matvec\"  / buffer: <unnamed>, length:1476, resident:Read Write"
                    .to_string()
            )
        );
        assert_eq!(
            validation_reports("Metal GPU Validation Enabled\nclause row_tail: PASS\n"),
            (0, String::new())
        );
    }
}
