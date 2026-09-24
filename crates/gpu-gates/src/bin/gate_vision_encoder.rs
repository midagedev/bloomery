//! GPU gate for the V4.1 image encoder on the card (`bloomery_gpu_vision`): the whole chain from
//! patches to aligner rows, on every image of the vision oracle set, tap by tap against the
//! official `vision.py` run by torch (`just dump-ref-vision`), and every kernel of block 0 against
//! its host rule.
//!
//! Per image, in chain order (so the first failing line names the earliest op):
//! 1. The patches of our preprocessing (`vision::preprocess` on the committed PNG) equal the
//!    set's bit for bit — the V1 gate's claim, re-read here because every later tap starts there.
//! 2. Every tap the set holds for the image — `embed`, the full-tap image's block-0 ops, the
//!    blocks, `vit`, the aligner's hidden rows and its output — against the reference within the
//!    tap's pinned band: `max|Δ| / max|ref|`, `rms(Δ) / rms(ref)` and the fraction of values that
//!    differ at all (the unit of the set's `# probe` rows, which count the reference's own
//!    distance from the exact rounding).
//! 3. Full-tap image only, block 0 op by op against the host rules of `bloomery_gpu_vision`, each
//!    on our own tapped inputs: the norms, the RoPE and the unfold bit for bit; every GEMM inside
//!    its accumulation-order bound around the exact product (`gemm_bf16::within_order`), the
//!    attention inside its f32 bound around the exact attention (`attn::attn_bound`); the
//!    residual epilogues bit for bit against a second, epilogue-free launch of the same GEMM; the
//!    GELU epilogue and the SiLU gate within one bf16 step of their host rule, the count of
//!    differing values pinned (the device's `erff`/`expf` against the host's).
//! 4. The plain encode run twice is bit-identical to itself and to the tapped run, and takes
//!    `Encoder::launches` launches.
//!
//! The bands of the patch embedding, block 0 and every teacher-forced tap are measurements of this
//! build against this set, pinned per tap name as the widest over the images. A free-running tap
//! after block 0 is held to a multiple of the reference's distance from itself under one-ulp
//! noise (the set's `# sensitivity` rows), since the network amplifies a last-bit difference.

#[cfg(not(feature = "vision"))]
fn main() {
    eprintln!(
        "gate_vision_encoder: built without the `vision` feature; see `just gate-gpu-vision`."
    );
    std::process::exit(2);
}

#[cfg(feature = "vision")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_vision_encoder", gate::run())
}

#[cfg(feature = "vision")]
mod gate {
    use std::collections::{HashMap, HashSet};
    use std::path::{Path, PathBuf};

    use bloomery_gpu_gates::{GateError, checks_failed, data_dir, verdict};
    use bloomery_gpu_vision::aligner::unfold_ref;
    use bloomery_gpu_vision::attn::{HEAD_DIM, QkvLayout, attn_bound, attn_ref};
    use bloomery_gpu_vision::encoder::{Encoder, TapSink};
    use bloomery_gpu_vision::gemm_bf16::{
        Epilogue, epilogue_ref, gemm_ref, order_bound, within_order,
    };
    use bloomery_gpu_vision::mlp::silu_mul_ref;
    use bloomery_gpu_vision::norm::rms_norm_ref;
    use bloomery_gpu_vision::rope2d::{RopeTable, rope_ref};
    use bloomery_gpu_vision::{bf16_f32, f32_bf16};
    use cuda_core::CudaContext;
    use gguf::Gguf;
    use vision::arch::deepseek41v::names;
    use vision::{Rgb8, preprocess};

    const NAME: &str = "gate_vision_encoder";
    /// The checkpoint revision every set must name.
    const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
    /// The image whose every block and block-0 ops the set holds.
    const FULL: &str = "grad-448";
    /// Threads the host rules run on.
    const HOST_THREADS: usize = 8;
    /// A free-running tap after block 0 passes when `rms(ours − ref) / rms(ref)` is at most this
    /// multiple of its [`ruler`] row's `rms_rel` in the set's `# sensitivity` rows: the reference
    /// run against itself with a one-ulp flip in the fraction of its patch embedding ours differs
    /// in.
    /// PIN(2026-09-24): the smallest integer all six images pass (widest ratio 1.94, vit of checker-1036); the network amplifies a 1-ulp difference, so the reference's own self-distance is the ruler.
    const SENSITIVITY_K: f64 = 2.0;

    /// One tap's band: `max|Δ|/max|ref|`, `rms(Δ)/rms(ref)` and the fraction of values that
    /// differ.
    struct Pin {
        tap: &'static str,
        max_rel: f64,
        rms_rel: f64,
        differ: f64,
    }

    /// Per tap name, the widest band over the set's images, for the patch embedding, block 0 and
    /// its ops, and every `forced.<tap>`: that tap computed from the reference's own input to it
    /// (one block, the final norm, or the aligner alone). These and the op rules are the
    /// contract; the free-running taps after block 0 are held by [`SENSITIVITY_K`] instead.
    /// PIN(2026-09-24): measured on the 3090 against set deepseek41v, each value the widest over its images rounded up at three significant digits (differ at five decimals).
    const PINS: &[Pin] = &[
        Pin {
            tap: "embed",
            max_rel: 0.00371,
            rms_rel: 2.4e-05,
            differ: 0.00016,
        },
        Pin {
            tap: "blk0.norm1",
            max_rel: 0.000136,
            rms_rel: 8.14e-06,
            differ: 0.00015,
        },
        Pin {
            tap: "blk0.qkv",
            max_rel: 0.00224,
            rms_rel: 2.29e-05,
            differ: 0.00080,
        },
        Pin {
            tap: "blk0.qrot",
            max_rel: 0.00198,
            rms_rel: 3.47e-05,
            differ: 0.00023,
        },
        Pin {
            tap: "blk0.krot",
            max_rel: 0.00213,
            rms_rel: 2.16e-05,
            differ: 0.00011,
        },
        Pin {
            tap: "blk0.sdpa",
            max_rel: 0.00267,
            rms_rel: 0.000143,
            differ: 0.00394,
        },
        Pin {
            tap: "blk0.attn",
            max_rel: 0.00457,
            rms_rel: 0.000172,
            differ: 0.02424,
        },
        Pin {
            tap: "blk0.norm2",
            max_rel: 0.00155,
            rms_rel: 0.000245,
            differ: 0.00932,
        },
        Pin {
            tap: "blk0.w1",
            max_rel: 0.00527,
            rms_rel: 0.000544,
            differ: 0.07716,
        },
        Pin {
            tap: "blk0.act",
            max_rel: 0.00358,
            rms_rel: 0.00055,
            differ: 0.12918,
        },
        Pin {
            tap: "blk0.mlp",
            max_rel: 0.00559,
            rms_rel: 0.000869,
            differ: 0.12620,
        },
        Pin {
            tap: "blk0",
            max_rel: 0.0112,
            rms_rel: 0.00241,
            differ: 0.21305,
        },
        Pin {
            tap: "forced.blk0",
            max_rel: 0.0112,
            rms_rel: 0.0024,
            differ: 0.20566,
        },
        Pin {
            tap: "forced.blk1",
            max_rel: 0.00758,
            rms_rel: 0.00172,
            differ: 0.16267,
        },
        Pin {
            tap: "forced.blk2",
            max_rel: 0.00426,
            rms_rel: 0.000898,
            differ: 0.08925,
        },
        Pin {
            tap: "forced.blk3",
            max_rel: 0.00447,
            rms_rel: 0.000892,
            differ: 0.08227,
        },
        Pin {
            tap: "forced.blk4",
            max_rel: 0.0069,
            rms_rel: 0.00105,
            differ: 0.10695,
        },
        Pin {
            tap: "forced.blk5",
            max_rel: 0.00685,
            rms_rel: 0.000982,
            differ: 0.09442,
        },
        Pin {
            tap: "forced.blk6",
            max_rel: 0.00626,
            rms_rel: 0.00105,
            differ: 0.10250,
        },
        Pin {
            tap: "forced.blk7",
            max_rel: 0.00663,
            rms_rel: 0.0011,
            differ: 0.10717,
        },
        Pin {
            tap: "forced.blk8",
            max_rel: 0.00618,
            rms_rel: 0.00121,
            differ: 0.11712,
        },
        Pin {
            tap: "forced.blk9",
            max_rel: 0.00296,
            rms_rel: 0.00126,
            differ: 0.12766,
        },
        Pin {
            tap: "forced.blk10",
            max_rel: 0.00345,
            rms_rel: 0.00155,
            differ: 0.17017,
        },
        Pin {
            tap: "forced.blk11",
            max_rel: 0.00251,
            rms_rel: 0.00144,
            differ: 0.13884,
        },
        Pin {
            tap: "forced.blk12",
            max_rel: 0.00106,
            rms_rel: 0.000481,
            differ: 0.14788,
        },
        Pin {
            tap: "forced.blk13",
            max_rel: 3.3e-05,
            rms_rel: 0.000223,
            differ: 0.14466,
        },
        Pin {
            tap: "forced.blk14",
            max_rel: 3.21e-05,
            rms_rel: 0.000199,
            differ: 0.12457,
        },
        Pin {
            tap: "forced.blk15",
            max_rel: 6.41e-05,
            rms_rel: 0.00019,
            differ: 0.11141,
        },
        Pin {
            tap: "forced.blk16",
            max_rel: 3.21e-05,
            rms_rel: 0.000185,
            differ: 0.10092,
        },
        Pin {
            tap: "forced.blk17",
            max_rel: 3.21e-05,
            rms_rel: 0.000176,
            differ: 0.09071,
        },
        Pin {
            tap: "forced.blk18",
            max_rel: 1.61e-05,
            rms_rel: 0.000168,
            differ: 0.08871,
        },
        Pin {
            tap: "forced.blk19",
            max_rel: 3.21e-05,
            rms_rel: 0.000163,
            differ: 0.08083,
        },
        Pin {
            tap: "forced.blk20",
            max_rel: 3.21e-05,
            rms_rel: 0.000179,
            differ: 0.08843,
        },
        Pin {
            tap: "forced.blk21",
            max_rel: 3.21e-05,
            rms_rel: 0.00017,
            differ: 0.08368,
        },
        Pin {
            tap: "forced.blk22",
            max_rel: 3.21e-05,
            rms_rel: 0.000189,
            differ: 0.09939,
        },
        Pin {
            tap: "forced.blk23",
            max_rel: 3.19e-05,
            rms_rel: 0.000193,
            differ: 0.09787,
        },
        Pin {
            tap: "forced.blk24",
            max_rel: 3.18e-05,
            rms_rel: 0.000188,
            differ: 0.08553,
        },
        Pin {
            tap: "forced.blk25",
            max_rel: 3.18e-05,
            rms_rel: 0.0002,
            differ: 0.09100,
        },
        Pin {
            tap: "forced.blk26",
            max_rel: 3.17e-05,
            rms_rel: 0.000239,
            differ: 0.10853,
        },
        Pin {
            tap: "forced.blk27",
            max_rel: 3.17e-05,
            rms_rel: 0.00024,
            differ: 0.09962,
        },
        Pin {
            tap: "forced.blk28",
            max_rel: 3.17e-05,
            rms_rel: 0.000266,
            differ: 0.09681,
        },
        Pin {
            tap: "forced.blk29",
            max_rel: 3.17e-05,
            rms_rel: 0.00027,
            differ: 0.07798,
        },
        Pin {
            tap: "forced.blk30",
            max_rel: 3.17e-05,
            rms_rel: 0.000276,
            differ: 0.05540,
        },
        Pin {
            tap: "forced.blk31",
            max_rel: 0.00268,
            rms_rel: 0.000404,
            differ: 0.07074,
        },
        Pin {
            tap: "forced.vit",
            max_rel: 0.00108,
            rms_rel: 2.07e-05,
            differ: 0.00002,
        },
        Pin {
            tap: "forced.aligner",
            max_rel: 0.00532,
            rms_rel: 0.0032,
            differ: 0.48929,
        },
    ];

    /// Values of the GELU epilogue and of the SiLU gate that may differ (by one bf16 step at most)
    /// from their host rule, per full-tap image: the device's `erff`/`expf` against the host's.
    /// PIN(2026-09-24): measured 0 of 865280 (GELU) and 0 of 4283136 (SiLU) on grad-448.
    const GELU_DIFFER_PIN: usize = 0;
    /// Values of the [−8, 8] GELU sweep that may differ from the host rule (each within one bf16
    /// step or `|x|·2⁻²³`): where `1 + erf(x/√2)` cancels, one ulp of the device's `erff` is many
    /// steps of the tiny result.
    /// PIN(2026-09-24): measured 4 of 33026, at x = −4.25, −4.375, −4.4375, −4.90625.
    const GELU_SWEEP_PIN: usize = 4;
    const SILU_DIFFER_PIN: usize = 0;

    // ------------------------------------------------------------ the set

    struct Img {
        name: String,
        n_vit_h: usize,
        n_vit_w: usize,
    }

    impl Img {
        fn stem(&self) -> &str {
            self.name.strip_suffix(".png").unwrap_or(&self.name)
        }
    }

    struct Set {
        dir: PathBuf,
        mmproj: PathBuf,
        images: Vec<Img>,
        /// file name -> (rows, cols) for every bf16 file.
        files: HashMap<String, (usize, usize)>,
        /// tap -> `rms_rel` of its `# sensitivity` row.
        sensitivity: HashMap<String, f64>,
    }

    fn read_set() -> Result<Set, GateError> {
        let set = std::env::var("BLOOMERY_VISION_SET").unwrap_or_else(|_| "deepseek41v".into());
        let dir = data_dir().join("ref-vision").join(set);
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv"))
            .map_err(|e| format!("{}: {e} — run: just dump-ref-vision", dir.display()))?;
        if !text.lines().any(|l| l.starts_with("# complete\t")) {
            return Err(format!("{}: no # complete trailer", dir.display()).into());
        }
        let ckpt = text
            .lines()
            .find(|l| l.starts_with("# checkpoint\t"))
            .ok_or("MANIFEST has no # checkpoint line")?;
        if !ckpt.contains(REVISION) {
            return Err(format!("stale set: {ckpt:?} is not revision {REVISION}").into());
        }
        let columns = text
            .lines()
            .find(|l| l.starts_with("# sensitivity columns\t"))
            .ok_or("MANIFEST has no # sensitivity columns line — run: just dump-ref-vision")?;
        if !columns.starts_with("# sensitivity columns\ttap max_rel rms_rel differ\t") {
            return Err(format!("unknown sensitivity columns: {columns:?}").into());
        }
        let (mut mmproj, mut images, mut files) = (None, Vec::new(), HashMap::new());
        let mut sensitivity = HashMap::new();
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            match f[0] {
                "# mmproj" => mmproj = Some(PathBuf::from(f[1])),
                "# sensitivity" => {
                    sensitivity.insert(f[1].to_string(), f[3].parse::<f64>()?);
                }
                "image" => images.push(Img {
                    name: f[1].to_string(),
                    n_vit_h: f[7].parse()?,
                    n_vit_w: f[8].parse()?,
                }),
                "file" if f[3] == "bf16" => {
                    let dims: Vec<usize> =
                        f[4].split('x').map(str::parse).collect::<Result<_, _>>()?;
                    if let [r, c] = dims[..] {
                        files.insert(f[1].to_string(), (r, c));
                    }
                }
                _ => {}
            }
        }
        // The full-tap image first: its op-level checks name a defect before any block does.
        images.sort_by_key(|i| (i.stem() != FULL, i.name.clone()));
        Ok(Set {
            dir,
            mmproj: mmproj.ok_or("MANIFEST has no # mmproj line")?,
            images,
            files,
            sensitivity,
        })
    }

    fn read_bits(dir: &Path, name: &str) -> Result<Vec<u16>, GateError> {
        let b = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        Ok(b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect())
    }

    // ------------------------------------------------------------ taps

    /// Collects the named tensors of one encode.
    struct Collect {
        want: HashSet<String>,
        got: HashMap<String, Vec<u16>>,
    }

    impl TapSink for Collect {
        fn wants(&self, name: &str) -> bool {
            self.want.contains(name)
        }
        fn take(&mut self, name: &str, _cols: usize, bits: Vec<u16>) {
            self.got.insert(name.to_string(), bits);
        }
    }

    /// The oracle taps of `stem`, in chain order.
    fn oracle_taps(set: &Set, stem: &str) -> Vec<String> {
        let mut t = vec!["embed".to_string()];
        for op in [
            "norm1", "qkv", "qrot", "krot", "sdpa", "attn", "norm2", "w1", "act", "mlp",
        ] {
            t.push(format!("blk0.{op}"));
        }
        t.extend((0..32).map(|b| format!("blk{b}")));
        t.extend(["vit", "aligner.w1", "aligner.h"].map(String::from));
        t.into_iter()
            .filter(|n| set.files.contains_key(&format!("{stem}.{n}.bf16")))
            .collect()
    }

    // ------------------------------------------------------------ bands

    /// The `# sensitivity` row a free-running tap is measured against: its own for `blk1` ..
    /// `blk31`, `vit` and `aligner`; the aligner's for the aligner's inner taps, which the rows do
    /// not list. `None` for the taps held by an absolute [`Pin`].
    fn ruler(tap: &str) -> Option<&str> {
        match tap {
            "vit" | "aligner" => Some(tap),
            "aligner.w1" | "aligner.h" => Some("aligner"),
            _ => tap
                .strip_prefix("blk")
                .and_then(|b| b.parse::<usize>().ok())
                .filter(|&b| b >= 1)
                .map(|_| tap),
        }
    }

    #[derive(Clone, Copy)]
    struct Stats {
        max_rel: f64,
        rms_rel: f64,
        differ: f64,
    }

    fn stats(ours: &[u16], refv: &[u16]) -> Stats {
        let (mut max_d, mut max_r, mut sd, mut sr, mut differ) = (0.0f64, 0.0f64, 0.0, 0.0, 0usize);
        for (&a, &b) in ours.iter().zip(refv) {
            let (x, y) = (f64::from(bf16_f32(a)), f64::from(bf16_f32(b)));
            let d = (x - y).abs();
            max_d = max_d.max(d);
            max_r = max_r.max(y.abs());
            sd += d * d;
            sr += y * y;
            differ += usize::from(a != b);
        }
        Stats {
            max_rel: max_d / max_r,
            rms_rel: (sd / sr).sqrt(),
            differ: differ as f64 / refv.len() as f64,
        }
    }

    /// Compare one tap with the reference and its pin, or with [`SENSITIVITY_K`] times its
    /// [`ruler`] row; prints one line.
    fn band_check(
        sens: &HashMap<String, f64>,
        stem: &str,
        tap: &str,
        ours: &[u16],
        refv: &[u16],
    ) -> bool {
        if ours.len() != refv.len() {
            println!(
                "tap {stem:<18} {tap:<12} FAIL: {} values, the set has {}",
                ours.len(),
                refv.len()
            );
            return false;
        }
        let s = stats(ours, refv);
        let pin = PINS.iter().find(|p| p.tap == tap);
        let (pass, band) = match (ruler(tap), pin) {
            (Some(r), None) => match sens.get(r) {
                Some(&u) => (
                    s.rms_rel <= SENSITIVITY_K * u,
                    format!(
                        "rms / sensitivity({r}) {:.2} (at most {SENSITIVITY_K})",
                        s.rms_rel / u
                    ),
                ),
                None => (false, format!("no # sensitivity row {r}")),
            },
            (Some(_), Some(_)) => (false, "both a pin and a sensitivity ruler".to_string()),
            (None, Some(p)) => (
                s.max_rel <= p.max_rel && s.rms_rel <= p.rms_rel && s.differ <= p.differ,
                format!("band {:.2e} {:.2e} {:.4}", p.max_rel, p.rms_rel, p.differ),
            ),
            (None, None) => (false, "no pin".to_string()),
        };
        println!(
            "tap {stem:<18} {tap:<12} max|rel| {:.3e}  rms rel {:.3e}  differ {:.5}  {band}  {}",
            s.max_rel,
            s.rms_rel,
            s.differ,
            verdict(pass)
        );
        pass
    }

    /// The `k` rows of a `cols`-wide tap whose difference from the reference is largest (row
    /// rms), with the reference row's rms and largest value — where a band's width comes from.
    fn worst_rows(
        stem: &str,
        tap: &str,
        ours: &[u16],
        refv: &[u16],
        cols: usize,
        n_w: usize,
        k: usize,
    ) {
        let mut rows: Vec<(f64, f64, f64, usize)> = ours
            .chunks_exact(cols)
            .zip(refv.chunks_exact(cols))
            .enumerate()
            .map(|(i, (a, b))| {
                let (mut sd, mut sr, mut mr) = (0.0f64, 0.0f64, 0.0f64);
                for (&x, &y) in a.iter().zip(b) {
                    let (x, y) = (f64::from(bf16_f32(x)), f64::from(bf16_f32(y)));
                    sd += (x - y) * (x - y);
                    sr += y * y;
                    mr = mr.max(y.abs());
                }
                ((sd / cols as f64).sqrt(), (sr / cols as f64).sqrt(), mr, i)
            })
            .collect();
        let all_rms = {
            let (sr, n): (f64, usize) = rows
                .iter()
                .fold((0.0, 0), |(s, n), r| (s + r.1 * r.1, n + 1));
            (sr / n as f64).sqrt()
        };
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let list: Vec<String> = rows
            .iter()
            .take(k)
            .map(|&(d, r, m, i)| {
                format!(
                    "row {i} (h {}, w {}): rms Δ {d:.3e}, ref rms {r:.3e}, ref max {m:.3e}",
                    i / n_w,
                    i % n_w
                )
            })
            .collect();
        println!(
            "  worst {stem} {tap} (all rows: ref rms {all_rms:.3e}): {}",
            list.join("; ")
        );
    }

    // ------------------------------------------------------------ host rules

    /// `f(row)` for every row on [`HOST_THREADS`] threads, summed.
    fn par_rows<T: Send + Default + std::ops::AddAssign>(
        rows: usize,
        f: impl Fn(usize) -> T + Sync,
    ) -> T {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..HOST_THREADS)
                .map(|t| {
                    let f = &f;
                    s.spawn(move || {
                        let mut acc = T::default();
                        let mut r = t;
                        while r < rows {
                            acc += f(r);
                            r += HOST_THREADS;
                        }
                        acc
                    })
                })
                .collect();
            let mut total = T::default();
            for h in hs {
                total += h.join().expect("host rule thread");
            }
            total
        })
    }

    #[derive(Default, Clone, Copy)]
    struct Count {
        outside: usize,
        differ: usize,
    }

    impl std::ops::AddAssign for Count {
        fn add_assign(&mut self, o: Count) {
            self.outside += o.outside;
            self.differ += o.differ;
        }
    }

    /// One GEMM of the chain against [`gemm_ref`]: every output inside the accumulation-order
    /// bound, and the count that differ from the exact rounding.
    #[allow(
        clippy::too_many_arguments,
        reason = "one GEMM's operands and shape, as the kernel takes them"
    )]
    fn gemm_check(
        name: &str,
        a: &[u16],
        b: &[u16],
        bias: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        got: &[u16],
    ) -> bool {
        let g = order_bound(k);
        let c = par_rows(m, |i| {
            let mut c = Count::default();
            for j in 0..n {
                let (exact, mag) = gemm_ref(a, b, bias, k, i, j);
                let y = got[i * n + j];
                c.outside += usize::from(!within_order(y, exact, g * mag));
                c.differ += usize::from(y != f32_bf16(exact as f32));
            }
            c
        });
        let pass = c.outside == 0;
        println!(
            "rule gemm {name:<12} m {m} n {n} k {k}: outside the order bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
            c.outside,
            m * n,
            c.differ,
            c.differ as f64 / (m * n) as f64,
            verdict(pass)
        );
        pass
    }

    /// A bit-exact rule: `got` equals `want` everywhere.
    fn exact_check(name: &str, got: &[u16], want: &[u16]) -> bool {
        let differ =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        let pass = differ == 0;
        println!(
            "rule {name:<17} bit-exact: {differ} of {} differ  {}",
            want.len(),
            verdict(pass)
        );
        pass
    }

    /// A rule the device's libm differs from by ulps: at most `pin` values differ, none by more
    /// than one bf16 step.
    fn near_check(name: &str, got: &[u16], want: &[u16], pin: usize) -> bool {
        let (mut differ, mut far) = (0usize, 0usize);
        for (&a, &b) in got.iter().zip(want) {
            if a != b {
                differ += 1;
                far += usize::from(a.abs_diff(b) > 1 || (a ^ b) & 0x8000 != 0);
            }
        }
        let pass = got.len() == want.len() && differ <= pin && far == 0;
        println!(
            "rule {name:<17} {differ} of {} differ (pin {pin}), {far} by more than one step  {}",
            want.len(),
            verdict(pass)
        );
        pass
    }

    /// The GELU epilogue on every normal bf16 value in [−8, 8]: a GEMM of `K = 2` whose A rows
    /// are `(x, 0)` and whose B row 0 is `(1, 0)` puts each `x` exactly into column 0 of its row,
    /// so the epilogue sees every input once; column 0 against the host rule, `pin` values allowed
    /// to differ by one step (the device's `erff` against the host's erf). A no-epilogue launch
    /// of the same operands must return every normal `x` itself; the zero and subnormal inputs
    /// are counted, not gated.
    fn gelu_sweep(
        ctx: &std::sync::Arc<CudaContext>,
        stream: &cuda_core::CudaStream,
    ) -> Result<bool, GateError> {
        use bloomery_gpu_vision::gemm_bf16::{GemmArgs, GemmKernels, TILE_N};
        use cuda_core::DeviceBuffer;
        let xs: Vec<u16> = (0..=0x4100u16).chain(0x8000..=0xC100).collect();
        let m = xs.len();
        let a: Vec<u16> = xs.iter().flat_map(|&x| [x, 0]).collect();
        let mut b = vec![0u16; TILE_N * 2];
        b[0] = f32_bf16(1.0);
        let gemm = GemmKernels::load(ctx, stream)?;
        let (da, db) = (
            DeviceBuffer::from_host(stream, &a)?,
            DeviceBuffer::from_host(stream, &b)?,
        );
        let run = |epilogue: Epilogue| -> Result<Vec<u16>, GateError> {
            let mut c = DeviceBuffer::zeroed(stream, m * TILE_N)?;
            gemm.enqueue(
                stream,
                GemmArgs {
                    a: &da,
                    b: &db,
                    bias: None,
                    epilogue,
                    resid: None,
                    m,
                    n: TILE_N,
                    k: 2,
                    c: &mut c,
                },
            )?;
            stream.synchronize()?;
            Ok(c.to_host_vec(stream)?
                .as_chunks::<TILE_N>()
                .0
                .iter()
                .map(|r| r[0])
                .collect())
        };
        let plain = run(Epilogue::None)?;
        let gelu = run(Epilogue::Gelu)?;
        // By class: a zero or subnormal input (exponent bits 0) against a normal one; −0 comes
        // back +0 because the accumulator starts at +0 and −0 + +0 is +0.
        let tiny = |x: u16| x & 0x7F80 == 0;
        let (mut bad_normal, mut bad_tiny) = (0usize, 0usize);
        for (&x, &y) in xs.iter().zip(&plain) {
            if x != y {
                if tiny(x) {
                    bad_tiny += 1;
                } else {
                    bad_normal += 1;
                }
            }
        }
        let n_tiny = xs.iter().filter(|&&x| tiny(x)).count();
        let identity = bad_normal == 0;
        println!(
            "rule gemm identity       {} normal values through K = 2: {bad_normal} do not come back as themselves; zero or subnormal inputs: {bad_tiny} of {n_tiny} do not (not gated)  {}",
            m - n_tiny,
            verdict(identity)
        );
        let (normal_x, normal_gelu): (Vec<u16>, Vec<u16>) = xs
            .iter()
            .zip(&gelu)
            .filter(|(x, _)| !tiny(**x))
            .map(|(&x, &y)| (x, y))
            .unzip();
        // A value may differ from the host rule where the device's `erff` and the host's erf
        // round apart: by one bf16 step, or — where `1 + erf` cancels — by at most
        // `|x| · 0.5 · 2 ulp(1) = |x| · 2⁻²³` in value, whatever that is in steps.
        let (mut differ, mut outside, mut shown) = (0usize, 0usize, Vec::new());
        for (&x, &y) in normal_x.iter().zip(&normal_gelu) {
            let want = epilogue_ref(x, Epilogue::Gelu, 0);
            if y == want {
                continue;
            }
            differ += 1;
            let (xv, yv, wv) = (
                f64::from(bf16_f32(x)),
                f64::from(bf16_f32(y)),
                f64::from(bf16_f32(want)),
            );
            let step = y.abs_diff(want) == 1 && (y ^ want) & 0x8000 == 0;
            let far = !step && (yv - wv).abs() > xv.abs() * f64::powi(2.0, -23);
            outside += usize::from(far);
            if shown.len() < 6 {
                shown.push(format!("x {xv:e}: {yv:e} vs {wv:e}"));
            }
        }
        let pass = differ <= GELU_SWEEP_PIN && outside == 0;
        println!(
            "rule gelu sweep [-8, 8]  {differ} of {} differ (pin {GELU_SWEEP_PIN}), {outside} outside one step or |x|·2⁻²³  {}{}",
            normal_x.len(),
            verdict(pass),
            if shown.is_empty() {
                String::new()
            } else {
                format!("  [{}]", shown.join("; "))
            }
        );
        Ok(identity & pass)
    }

    /// The host copies of the weights block 0 and the aligner read.
    struct HostW {
        patch_w: Vec<u16>,
        patch_b: Vec<f32>,
        ln1: Vec<f32>,
        qkv_w: Vec<u16>,
        qkv_b: Vec<f32>,
        o_w: Vec<u16>,
        o_b: Vec<f32>,
        ln2: Vec<f32>,
        w13: Vec<u16>,
        w2: Vec<u16>,
        post_ln: Vec<f32>,
        mm1_w: Vec<u16>,
        mm1_b: Vec<f32>,
        mm2_w: Vec<u16>,
        mm2_b: Vec<f32>,
    }

    fn host_w(file: &Gguf) -> Result<HostW, GateError> {
        let raw = |n: String| -> Result<&[u8], GateError> {
            let t = file
                .find(&n)
                .ok_or_else(|| format!("{n} not in the encoder file"))?;
            Ok(file.data(t)?)
        };
        let b16 = |n: String| -> Result<Vec<u16>, GateError> {
            Ok(raw(n)?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect())
        };
        let f32s = |n: String| -> Result<Vec<f32>, GateError> {
            Ok(raw(n)?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect())
        };
        let mut w13 = b16(names::ffn_gate(0))?;
        w13.extend(b16(names::ffn_up(0))?);
        Ok(HostW {
            patch_w: b16(names::patch_embd_weight())?,
            patch_b: f32s(names::patch_embd_bias())?,
            ln1: f32s(names::ln1(0))?,
            qkv_w: b16(names::attn_qkv_weight(0))?,
            qkv_b: f32s(names::attn_qkv_bias(0))?,
            o_w: b16(names::attn_out_weight(0))?,
            o_b: f32s(names::attn_out_bias(0))?,
            ln2: f32s(names::ln2(0))?,
            w13,
            w2: b16(names::ffn_down(0))?,
            post_ln: f32s(names::post_ln())?,
            mm1_w: b16(names::mm1_weight())?,
            mm1_b: f32s(names::mm1_bias())?,
            mm2_w: b16(names::mm2_weight())?,
            mm2_b: f32s(names::mm2_bias())?,
        })
    }

    fn norm_rows(x: &[u16], g: &[f32], eps: f32) -> Vec<u16> {
        x.chunks_exact(g.len())
            .flat_map(|r| rms_norm_ref(r, g, eps))
            .collect()
    }

    /// Block 0 and the aligner of the full-tap image, op by op, on our own tapped inputs.
    fn host_rules(
        enc: &Encoder,
        w: &HostW,
        t: &HashMap<String, Vec<u16>>,
        patches: &[u16],
        img: &Img,
        out: &[u16],
    ) -> Result<bool, GateError> {
        let hp = enc.hparams();
        let (dim, ff, n) = (hp.dim, hp.ff, img.n_vit_h * img.n_vit_w);
        let get = |k: &str| t.get(k).ok_or_else(|| format!("tap {k} was not taken"));
        let (embed, norm1, qkv) = (get("embed")?, get("blk0.norm1")?, get("blk0.qkv")?);
        let (qrot, krot, sdpa) = (get("blk0.qrot")?, get("blk0.krot")?, get("blk0.sdpa")?);
        let (attn_side, resid1, norm2) =
            (get("blk0.attn")?, get("blk0.resid1")?, get("blk0.norm2")?);
        let (w1, act, mlp_side, blk0) = (
            get("blk0.w1")?,
            get("blk0.act")?,
            get("blk0.mlp")?,
            get("blk0")?,
        );
        let (blk_last, vit, al_x) = (
            get(&format!("blk{}", hp.n_layer - 1))?,
            get("vit")?,
            get("aligner.x")?,
        );
        let (al_w1, al_h) = (get("aligner.w1")?, get("aligner.h")?);
        let k_patch = 3 * hp.patch * hp.patch;
        let mut ok = true;

        ok &= gemm_check(
            "patch_embed",
            patches,
            &w.patch_w,
            Some(&w.patch_b),
            n,
            dim,
            k_patch,
            embed,
        );
        ok &= exact_check("blk0.norm1", norm1, &norm_rows(embed, &w.ln1, hp.eps));
        ok &= gemm_check(
            "blk0.wqkv",
            norm1,
            &w.qkv_w,
            Some(&w.qkv_b),
            n,
            3 * dim,
            dim,
            qkv,
        );
        let table = RopeTable::new(img.n_vit_h, img.n_vit_w, hp.rope_theta);
        let mut rot = qkv.clone();
        rope_ref(&mut rot, 3 * dim, 0, hp.n_head, &table);
        rope_ref(&mut rot, 3 * dim, dim, hp.n_head, &table);
        let cols = |x: &[u16], c0: usize| -> Vec<u16> {
            x.chunks_exact(3 * dim)
                .flat_map(|r| r[c0..c0 + dim].iter().copied())
                .collect()
        };
        ok &= exact_check("blk0.rope q", qrot, &cols(&rot, 0));
        ok &= exact_check("blk0.rope k", krot, &cols(&rot, dim));
        // The attention's input is the kernel chain's own: our rotated q and k, and v, so the
        // attention rule judges the attention kernel alone.
        let mut ours_rot = qkv.clone();
        for (i, row) in ours_rot.chunks_exact_mut(3 * dim).enumerate() {
            row[..dim].copy_from_slice(&qrot[i * dim..(i + 1) * dim]);
            row[dim..2 * dim].copy_from_slice(&krot[i * dim..(i + 1) * dim]);
        }
        // The attention on our own rotated q, k (and the untouched v).
        let lay = QkvLayout {
            row_width: 3 * dim,
            q0: 0,
            k0: dim,
            v0: 2 * dim,
            n_heads: hp.n_head,
        };
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let c = par_rows(n, |i| {
            let mut c = Count::default();
            for h in 0..hp.n_head {
                let (o, mag, logit) = attn_ref(&ours_rot, lay, n, scale, h, i);
                for d in 0..HEAD_DIM {
                    let y = sdpa[i * dim + h * HEAD_DIM + d];
                    let b = attn_bound(o[d], mag[d], logit, n);
                    c.outside += usize::from(!within_order(y, o[d], b));
                    c.differ += usize::from(y != f32_bf16(o[d] as f32));
                }
            }
            c
        });
        let pass = c.outside == 0;
        println!(
            "rule attn blk0.sdpa     n {n}: outside the f32 bound {} of {}, differ from the exact rounding {} ({:.5})  {}",
            c.outside,
            n * dim,
            c.differ,
            c.differ as f64 / (n * dim) as f64,
            verdict(pass)
        );
        ok &= pass;

        ok &= gemm_check(
            "blk0.wo",
            sdpa,
            &w.o_w,
            Some(&w.o_b),
            n,
            dim,
            dim,
            attn_side,
        );
        let want: Vec<u16> = attn_side
            .iter()
            .zip(embed)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0.resid1 epi", resid1, &want);
        ok &= exact_check("blk0.norm2", norm2, &norm_rows(resid1, &w.ln2, hp.eps));
        ok &= gemm_check("blk0.w1", norm2, &w.w13, None, n, 2 * ff, dim, w1);
        let want: Vec<u16> = w1
            .chunks_exact(2 * ff)
            .flat_map(|r| (0..ff).map(move |j| silu_mul_ref(r[j], r[ff + j])))
            .collect();
        ok &= near_check("blk0.silu_mul", act, &want, SILU_DIFFER_PIN);
        ok &= gemm_check("blk0.w2", act, &w.w2, None, n, dim, ff, mlp_side);
        let want: Vec<u16> = mlp_side
            .iter()
            .zip(resid1)
            .map(|(&y, &r)| epilogue_ref(y, Epilogue::Residual, r))
            .collect();
        ok &= exact_check("blk0 resid2 epi", blk0, &want);
        ok &= exact_check("post_ln", vit, &norm_rows(blk_last, &w.post_ln, hp.eps));
        let r = hp.downsample;
        ok &= exact_check(
            "aligner unfold",
            al_x,
            &unfold_ref(vit, img.n_vit_h, img.n_vit_w, dim, r),
        );
        let n_llm = img.n_vit_h.div_ceil(r) * img.n_vit_w.div_ceil(r);
        let k1 = dim * r * r;
        ok &= gemm_check(
            "aligner.w1",
            al_x,
            &w.mm1_w,
            Some(&w.mm1_b),
            n_llm,
            hp.out_dim,
            k1,
            al_w1,
        );
        let want: Vec<u16> = al_w1
            .iter()
            .map(|&y| epilogue_ref(y, Epilogue::Gelu, 0))
            .collect();
        ok &= near_check("aligner gelu epi", al_h, &want, GELU_DIFFER_PIN);
        ok &= gemm_check(
            "aligner.w2",
            al_h,
            &w.mm2_w,
            Some(&w.mm2_b),
            n_llm,
            hp.out_dim,
            hp.out_dim,
            out,
        );
        Ok(ok)
    }

    // ------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let set = read_set()?;
        let file = Gguf::open(&set.mmproj)?;
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let enc = Encoder::load(&ctx, &stream, &file)?;
        let hp = enc.hparams().clone();
        let want_launches = Encoder::launches(hp.n_layer);
        println!(
            "load: {} blocks, dim {}, heads {}, ff {}, out {}; weights on the card {} B; {} launches per image",
            hp.n_layer,
            hp.dim,
            hp.n_head,
            hp.ff,
            hp.out_dim,
            enc.weight_bytes(),
            want_launches
        );
        let hw = host_w(&file)?;
        let mut ok = gelu_sweep(&ctx, &stream)?;
        let images_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/ref/vision/images");
        let mut widest: HashMap<String, Stats> = HashMap::new();

        for img in &set.images {
            let stem = img.stem().to_string();
            let png = std::fs::read(images_dir.join(&img.name))
                .map_err(|e| format!("{}: {e}", img.name))?;
            let patches = preprocess(&Rgb8::from_png(&png)?, &hp.grid())?;
            let want_p = read_bits(&set.dir, &format!("{stem}.patches.bf16"))?;
            let p_ok = patches.bf16 == want_p
                && (patches.n_vit_h, patches.n_vit_w) == (img.n_vit_h, img.n_vit_w);
            println!(
                "image {stem}: {}x{} patches, {} aligner rows; patches equal the set's: {}",
                img.n_vit_h,
                img.n_vit_w,
                img.n_vit_h.div_ceil(hp.downsample) * img.n_vit_w.div_ceil(hp.downsample),
                verdict(p_ok)
            );
            ok &= p_ok;

            let taps = oracle_taps(&set, &stem);
            let mut want: HashSet<String> = taps.iter().cloned().collect();
            if stem == FULL {
                want.extend(["blk0.resid1", "aligner.x"].map(String::from));
                want.insert(format!("blk{}", hp.n_layer - 1));
            }
            let mut sink = Collect {
                want,
                got: HashMap::new(),
            };
            let tapped = enc.encode_tapped(&stream, &patches, &mut sink)?;
            stream.synchronize()?;
            let out = tapped.rows.buf().to_host_vec(&stream)?;

            for tap in taps.iter().map(String::as_str).chain(["aligner"]) {
                let ours = if tap == "aligner" {
                    &out
                } else {
                    sink.got
                        .get(tap)
                        .ok_or_else(|| format!("tap {tap} was not taken"))?
                };
                let refv = read_bits(&set.dir, &format!("{stem}.{tap}.bf16"))?;
                ok &= band_check(&set.sensitivity, &stem, tap, ours, &refv);
                if ours.len() == refv.len() {
                    let s = stats(ours, &refv);
                    let e = widest.entry(tap.to_string()).or_insert(s);
                    e.max_rel = e.max_rel.max(s.max_rel);
                    e.rms_rel = e.rms_rel.max(s.rms_rel);
                    e.differ = e.differ.max(s.differ);
                }
            }
            if stem == FULL {
                for tap in ["blk11", "blk12", "blk31", "vit"] {
                    if let Some(ours) = sink.got.get(tap) {
                        let refv = read_bits(&set.dir, &format!("{stem}.{tap}.bf16"))?;
                        worst_rows(&stem, tap, ours, &refv, hp.dim, img.n_vit_w, 4);
                    }
                }
            }

            // Teacher-forced: each block the set holds both the input and the output of, run
            // alone from the reference's input; then the tail from the reference's last block and
            // the aligner from the reference's final norm.
            let grid = (img.n_vit_h, img.n_vit_w);
            for b in 0..hp.n_layer {
                let input = if b == 0 {
                    "embed".to_string()
                } else {
                    format!("blk{}", b - 1)
                };
                let (fi, fo) = (
                    format!("{stem}.{input}.bf16"),
                    format!("{stem}.blk{b}.bf16"),
                );
                if !(set.files.contains_key(&fi) && set.files.contains_key(&fo)) {
                    continue;
                }
                let got = enc.forced_block(&stream, grid, b, &read_bits(&set.dir, &fi)?)?;
                ok &= band_check(
                    &set.sensitivity,
                    &stem,
                    &format!("forced.blk{b}"),
                    &got,
                    &read_bits(&set.dir, &fo)?,
                );
            }
            let last = format!("{stem}.blk{}.bf16", hp.n_layer - 1);
            if set.files.contains_key(&last) {
                let (vit, _) =
                    enc.forced_tail(&stream, grid, &read_bits(&set.dir, &last)?, false)?;
                ok &= band_check(
                    &set.sensitivity,
                    &stem,
                    "forced.vit",
                    &vit,
                    &read_bits(&set.dir, &format!("{stem}.vit.bf16"))?,
                );
            }
            let (_, rows) = enc.forced_tail(
                &stream,
                grid,
                &read_bits(&set.dir, &format!("{stem}.vit.bf16"))?,
                true,
            )?;
            ok &= band_check(
                &set.sensitivity,
                &stem,
                "forced.aligner",
                &rows,
                &read_bits(&set.dir, &format!("{stem}.aligner.bf16"))?,
            );

            if stem == FULL {
                ok &= host_rules(&enc, &hw, &sink.got, &patches.bf16, img, &out)?;
            }

            let a = enc.encode(&stream, &patches)?;
            let b = enc.encode(&stream, &patches)?;
            stream.synchronize()?;
            let (va, vb) = (
                a.rows.buf().to_host_vec(&stream)?,
                b.rows.buf().to_host_vec(&stream)?,
            );
            let twice = va == vb && va == out;
            let launches = a.launches == want_launches && tapped.launches == want_launches;
            println!(
                "image {stem}: eager twice bit-identical and equal to the tapped run: {}; launches {} (want {want_launches}): {}",
                verdict(twice),
                a.launches,
                verdict(launches)
            );
            ok &= twice && launches;
        }

        println!("widest per tap (the pin table's measured values; a ratio for a ruled tap):");
        let mut names: Vec<&String> = widest.keys().collect();
        names.sort();
        for t in names {
            let s = widest[t];
            match ruler(t).and_then(|r| set.sensitivity.get(r).map(|&u| (r, u))) {
                Some((r, u)) => println!(
                    "  {t}: rms / sensitivity({r}) {:.2} (rms rel {:.2e})",
                    s.rms_rel / u,
                    s.rms_rel
                ),
                None => println!(
                    "  Pin {{ tap: \"{t}\", max_rel: {:.2e}, rms_rel: {:.2e}, differ: {:.5} }},",
                    s.max_rel, s.rms_rel, s.differ
                ),
            }
        }
        if ok {
            println!("{NAME}: PASS");
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
