//! GPU gate for DeepSeek-V4.1's compressor and index key — B4 op blocks D
//! and E (`docs/research/v41-b4-plan-report.md` §1-D, §1-E) — against ik's
//! CPU dump. A source layer is one whose compressor weights exist (three of
//! ratio 2 with a gate weight, one of ratio 1 without); each gives, per set:
//! - `gemv` sites: the state projections `wkv_c · x` and `wgate_c · x`
//!   (q3_K, K = 5120) and the index key's projection of the pre-rope row
//!   (q3_K, K = 512) — our q3_K gemv on q8_1 activations where ik's CPU build
//!   quantizes them to q8_K;
//! - a `comp` site: `ds41_comp_pool` (ratio 2) or `ds41_comp_rows` (ratio 1)
//!   — pooling, norm, the pre-rope row, the tail rope, the f16 cache row —
//!   and the ring's persist;
//! - an `idx` site: `ds41_index_key` — norm, tail rope, Hadamard, the f16
//!   key row.
//!
//! The kernels read the step's integers from the buffer `CompGeom::pack`
//! fills from the host plan (`model::arch::deepseek41::plan`) at the set's
//! position — the captured graph's form — not from the set's integer rows
//! (`gate-ds41-plan` proves the two equal). Each kernel's inputs are the
//! dump's own inputs to that op.
//!
//! Three layers per site (the B4 gate form):
//! 1. the kernel against this binary's transcription of our rule, and a
//!    rerun — bit-identical, except where the host cannot transcribe the
//!    kernel (`KERNEL_BAND`): the pooled rows of ratio 2, whose weights come
//!    from the device `expf` (`__nv_expf`), and the gemv, against the exact
//!    dot of its own q8_1 values; the caches are filled with a sentinel
//!    first, so a row written where it should not be, or not written, shows;
//! 2. ik's rule simulated here against the dump — the semantics: which rows
//!    pool (the plan's reads on the dump's own concatenated sources), that a
//!    ring slot is read only after a token was kept in it, the identity of a
//!    ratio-1 pooling, the norm, the rope, where the row lands in the cache
//!    (every other row unchanged), what the ring keeps, and that the index
//!    key projects the pre-rope row;
//! 3. the kernel against the dump in a per-value band derived from the two
//!    rules' difference: `over` counts the values outside it, beside
//!    `ik_rel` and, off the gemv, the max ulp.
//!
//! Sets: the 5-token prefill (every read inside the batch) and the decode
//! steps [`STEP_SETS`] at 4 (even: no ratio-2 group completes, the persist
//! alone runs and nothing is written), 301 and 1,025 (a group completes and
//! pools the slot the previous step kept). A step set's ring is the f32
//! input of the ring's shape its first toucher takes, by file order
//! (`docs/oracle.md`, 「상태는 input 행이다」).

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_comp: built without the `deepseek41` feature; see `just gate-gpu-ds41-comp`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_comp", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::HashMap;
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::path::Path;

    use bloomery_gpu::{DeviceTensor, Gpu, Q8Act};
    use bloomery_gpu_deepseek41::compress::{
        self, CompGeom, CompressKernels, PoolArgs, RowsArgs, StepInts,
    };
    use bloomery_gpu_deepseek41::index_key::{self, HT_SCALE, IndexKeyArgs, IndexKeyKernels};
    use bloomery_gpu_deepseek41::rope::{Direction, RopeSpec, RopeTable, ggml_rope_cache};
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, RefManifest, RefRow, bits_equal, bytes_to_words, checks_failed,
        max_rel_err, ref_dir_named, ref_model_path, ref_tensor_logical_in, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits, half_to_f32};
    use gguf::{Split, Value};
    use model::arch::Arch;
    use model::arch::deepseek41::plan::{Planner, StepPlan, StreamStep};

    /// The decode-step sets: one step after a prefill run under the dumped
    /// schedule, the `unfused` ones with the indexer's scores as nodes — the
    /// same compressor in another graph — and `d1n` at 301 with the file's
    /// top-k.
    pub const STEP_SETS: [&str; 6] = [
        "ref_deepseek41_step4_every_node",
        "ref_deepseek41_d1_every_node",
        "ref_deepseek41_d1_unfused_every_node",
        "ref_deepseek41_d1n_every_node",
        "ref_deepseek41_d2_every_node",
        "ref_deepseek41_d2_unfused_every_node",
    ];

    /// The port's names of the compressed streams, in the planner's order.
    const STREAMS: [&str; 2] = ["csa", "hca"];

    /// f32's unit roundoff, 2^-24.
    const U: f32 = f32::EPSILON / 2.0;

    /// What the gate fills a cache with before a launch: an f16 NaN, which
    /// `f32_to_f16_bits` never writes (it rounds a NaN to infinity), so a
    /// row still holding it was not written.
    const SENTINEL: u16 = 0xffff;

    /// What the gate fills an f32 buffer with that a kernel must not read: a
    /// NaN, so a read of it poisons every value it feeds (and the harness's
    /// comparisons refuse a non-finite value).
    const POISON: f32 = f32::NAN;

    /// Relative distance of the two `expf`s: the device's is within 2 ulp
    /// of `e^x` (CUDA's documented bound), glibc's within 0.502 ulp, and one
    /// ulp is at most `2u` of the value — together under `6u`.
    const EXP_REL: f32 = 6.0 * U;

    /// Units of `u · A` two DS4_COMP results of `r` rows may differ by, `A`
    /// the largest `|kv|` pooled into the value. The weights differ by
    /// [`EXP_REL`] each, and `∂y/∂w_k = (kv_k − y)/Σw`, so `|Δy| <= 6u ·
    /// max|kv_k − y| <= 12u·A`; each side rounds its `r` fused multiply-adds
    /// (`<= r·u·Σw·A`), its sum (`<= (r − 1)u·Σw`, a relative `(r − 1)u` on
    /// `|y| <= A`) and its division (`u·A`) — `2r·u·A` a side.
    fn pool_units(r: usize) -> f32 {
        2.0 * (EXP_REL / U) + 4.0 * r as f32
    }

    /// Units of `u · |n|` two norms of the same 512-value row differ by, ours
    /// against ik's. Our sum of squares rounds at most 11 times (four per
    /// thread, five butterfly levels, two tree levels) on non-negative
    /// terms, ik's twice (each square, then the f64 mean to f32): the means
    /// differ by `13u` of themselves. `+ eps` adds the same positive value
    /// and rounds each side (`15u`), the square root halves that and rounds
    /// each (`9.5u`), the reciprocal (`11.5u`), `· gain` (`13.5u`) and
    /// `· y` (`15.5u`) round each side once more.
    const NORM512_UNITS: f32 = 15.5;

    /// Units of `u · ‖out‖₂` an index key may differ by, ours against ik's.
    /// The norms of the same 128 values differ by `14.5u` of each value (the
    /// 512-value chain with nine roundings in our sum: four per lane, five
    /// butterfly levels). A tail pair's turn keeps the pair's 2-norm and
    /// rounds two products and a sum per value on each side (`4√2 u` of the
    /// pair). The transform is `σ·H` with `σ√128` within an ulp of 1: it maps
    /// the input difference with its 2-norm, each of its seven stages rounds
    /// every value once on each side (`7√128·u` of the input's 2-norm a side,
    /// `14u` of the output's together), and the final multiply once more
    /// (`2u`). A value moves by at most the 2-norm of the whole difference.
    const IDX_L2_UNITS: f32 = 14.5 + 4.0 * std::f32::consts::SQRT_2 + 14.0 + 2.0;

    /// `γ(n) = n·u / (1 − n·u)`: the relative bound of a sum whose every
    /// term passes through at most `n` roundings, on the sum of the terms'
    /// magnitudes.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * f64::from(U);
        nu / (1.0 - nu)
    }

    // ------------------------------------------------------------ inputs

    /// The file's rope and norm constants.
    struct Meta {
        n_dims: usize,
        yarn: RopeSpec,
        eps: f32,
    }

    impl Meta {
        fn read(split: &Split) -> Result<Meta, GateError> {
            let want = Arch::Deepseek41.name();
            if split.architecture() != Some(want) {
                return Err(format!(
                    "the model file is {:?}, want {want} — run through `just gate-gpu-ds41-comp`",
                    split.architecture()
                )
                .into());
            }
            let f = |s: &str| -> Result<f32, GateError> {
                split
                    .arch_get_f32(s)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
            };
            let u = |s: &str| -> Result<u64, GateError> {
                split
                    .arch_get_u64(s)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)).into())
            };
            let scaling = split.arch_get_str("rope.scaling.type");
            if scaling != Some("yarn") {
                return Err(format!("rope.scaling.type is {scaling:?}, want \"yarn\"").into());
            }
            let key = split.arch_key("attention.compress_ratios");
            if !matches!(split.value(&key), Some(Value::Array(_))) {
                return Err(format!("metadata {key} is absent or not an array").into());
            }
            let n_dims = usize::try_from(u("rope.dimension_count")?)?;
            Ok(Meta {
                n_dims,
                yarn: RopeSpec::yarn(
                    f("attention.compress_rope_freq_base")?,
                    f("rope.scaling.factor")?,
                    i32::try_from(u("rope.scaling.original_context_length")?)?,
                    f("rope.scaling.yarn_beta_fast")?,
                    f("rope.scaling.yarn_beta_slow")?,
                    n_dims,
                ),
                eps: f("attention.layer_norm_rms_epsilon")?,
            })
        }
    }

    /// The header lines a set's step is read from.
    struct Header {
        /// `# tokens`: the whole sequence.
        tokens: Vec<u32>,
        /// `# decode_pos`: the step's position, in a decode-step set.
        decode_pos: Option<u32>,
        /// `-c` in `# flags`: the port's context.
        ctx: u64,
    }

    impl Header {
        fn read(dir: &Path) -> Result<Header, GateError> {
            let path = dir.join("MANIFEST.tsv");
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let bad = |line: &str, e: std::num::ParseIntError| -> GateError {
                format!("{}: {line:?}: {e}", path.display()).into()
            };
            let (mut tokens, mut decode_pos, mut ctx) = (Vec::new(), None, None);
            for line in text.lines().take_while(|l| l.starts_with('#')) {
                if let Some(v) = line.strip_prefix("# tokens\t") {
                    tokens = v
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|e| bad(line, e))?;
                } else if let Some(v) = line.strip_prefix("# decode_pos\t") {
                    decode_pos = Some(v.parse().map_err(|e| bad(line, e))?);
                } else if let Some(v) = line.strip_prefix("# flags\t") {
                    let mut args = v.split_whitespace();
                    if args.by_ref().any(|a| a == "-c") {
                        ctx = Some(
                            args.next()
                                .unwrap_or("")
                                .parse()
                                .map_err(|e| bad(line, e))?,
                        );
                    }
                }
            }
            let ctx = ctx.ok_or_else(|| format!("{}: no -c in # flags", path.display()))?;
            if tokens.is_empty() {
                return Err(format!("{} has no # tokens line", path.display()).into());
            }
            Ok(Header {
                tokens,
                decode_pos,
                ctx,
            })
        }

        /// The step the set holds: its first position, its tokens, the tokens
        /// before it.
        fn step(&self) -> Result<(u32, &[u32], &[u32]), GateError> {
            match self.decode_pos {
                None => Ok((0, &self.tokens[..], &[][..])),
                Some(p) if p as usize + 1 == self.tokens.len() => {
                    let at = p as usize;
                    Ok((p, &self.tokens[at..], &self.tokens[..at]))
                }
                Some(p) => Err(format!(
                    "# decode_pos {p} is not the last of the {} tokens",
                    self.tokens.len()
                )
                .into()),
            }
        }
    }

    /// Each input row's first toucher, from the manifest's file order: the
    /// dumper writes a state input right before the node that first touches
    /// it, which may read it (a `CONCAT`) or write into it (a `SET_ROWS`
    /// destination, in no source column).
    struct FirstTouch(HashMap<(String, u32), (String, u32)>);

    impl FirstTouch {
        fn read(dir: &Path) -> Result<FirstTouch, GateError> {
            let path = dir.join("MANIFEST.tsv");
            let text = std::fs::read_to_string(&path)
                .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            let mut pending = Vec::new();
            let mut map = HashMap::new();
            for line in text.lines() {
                let f: Vec<&str> = line.split('\t').collect();
                let occ = || -> Result<u32, GateError> {
                    f.get(2)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| format!("{}: {line:?}", path.display()).into())
                };
                match f.first() {
                    Some(&"input") => pending.push((f[1].to_string(), occ()?)),
                    Some(&"tensor") => {
                        let by = (f[1].to_string(), occ()?);
                        for key in pending.drain(..) {
                            map.insert(key, by.clone());
                        }
                    }
                    _ => {}
                }
            }
            Ok(FirstTouch(map))
        }

        /// The input rows node `name`/`occ` touches first — a state and the
        /// integer rows that index it can share a first toucher.
        fn inputs_of<'m>(&self, man: &'m RefManifest, name: &str, occ: u32) -> Vec<&'m RefRow> {
            man.inputs
                .iter()
                .filter(|r| {
                    self.0
                        .get(&(r.name.clone(), r.occurrence))
                        .is_some_and(|(n, o)| n == name && *o == occ)
                })
                .collect()
        }
    }

    /// A set by name under the data directory, its `# arch` line checked.
    fn open_set(name: &str) -> Result<RefManifest, GateError> {
        let man = RefManifest::read(&ref_dir_named(name))?;
        let want = Arch::Deepseek41.name();
        if man.arch.as_deref() != Some(want) {
            return Err(format!("{name}: # arch is {:?}, want {want}", man.arch).into());
        }
        Ok(man)
    }

    /// The tensor row at or after nothing, before `at`, named `name`: the
    /// last one, with its index.
    fn last_before<'a>(
        man: &'a RefManifest,
        at: usize,
        name: &str,
    ) -> Result<(usize, &'a RefRow), GateError> {
        man.tensors[..at]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.name == name)
            .ok_or_else(|| format!("no node {name:?} before manifest row {at}").into())
    }

    /// The tensor row `name`/`occ` of op `op`, with its index.
    fn node<'a>(
        man: &'a RefManifest,
        name: &str,
        occ: u32,
        op: &str,
    ) -> Result<(usize, &'a RefRow), GateError> {
        man.tensors
            .iter()
            .enumerate()
            .find(|(_, r)| r.name == name && r.occurrence == occ)
            .filter(|(_, r)| r.op == op)
            .ok_or_else(|| format!("no {op} node {name}/{occ}").into())
    }

    /// The cache view the dump shows right before the `SET_ROWS` at `at`: the
    /// last f16 VIEW of the same shape before it — the cache before the write.
    fn cache_before(man: &RefManifest, at: usize) -> Result<&RefRow, GateError> {
        let write = &man.tensors[at];
        man.tensors[..at]
            .iter()
            .rev()
            .find(|r| r.op == "VIEW" && r.ty == "f16" && r.ne == write.ne)
            .ok_or_else(|| format!("no f16 view shaped like {} before it", write.name).into())
    }

    /// Rows `0 .. rows` of an f16 row as f16 bits: the dump widened each half
    /// to f32 exactly, so rounding back recovers ik's bits, and a value that
    /// does not come back is an error.
    fn f16_all(man: &RefManifest, row: &RefRow) -> Result<Vec<u16>, GateError> {
        if row.ty != "f16" || row.bytes != 4 * row.count() {
            return Err(format!(
                "{}/{} is {} with {} bytes for {} values, want f16 widened to f32",
                row.name,
                row.occurrence,
                row.ty,
                row.bytes,
                row.count()
            )
            .into());
        }
        let path = man.dir.join(row.file_name());
        let raw = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() as u64 != row.bytes {
            return Err(format!(
                "{} is {} bytes, the row says {}",
                path.display(),
                raw.len(),
                row.bytes
            )
            .into());
        }
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| {
                let v = f32::from_le_bytes(*c);
                let h = f32_to_f16_bits(v);
                if half_to_f32(h).to_bits() == v.to_bits() {
                    Ok(h)
                } else {
                    Err(format!("{} holds {v}, not a widened f16", path.display()).into())
                }
            })
            .collect()
    }

    /// An F32 tensor of the model file, `want` values.
    fn split_f32(split: &Split, name: &str, want: usize) -> Result<Vec<f32>, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        if t.ty != GgmlType::F32 || t.dims.iter().product::<u64>() != want as u64 {
            return Err(format!("{name} is {:?} {:?}, want F32 x {want}", t.ty, t.dims).into());
        }
        let g = split.shard(s).ok_or("shard index out of range")?;
        Ok(g.data(t)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    // ------------------------------------------------------ host rules

    /// ik's DS4_COMP type 1 on the host (`ggml.c`, its CPU build's
    /// contraction of `res += w·kv`) — and our kernel's pooling op for op,
    /// save the device `expf`: per value the C `MAX` from `−∞` over the
    /// sources' scores, then `w = expf(s − max)`, `sum += w`,
    /// `res = fma(w, kv, res)` in source order, `y = res / sum`.
    fn ds4_comp(kv: &[&[f32]], score: &[&[f32]]) -> Vec<f32> {
        (0..kv[0].len())
            .map(|i| {
                let mx = score
                    .iter()
                    .fold(f32::NEG_INFINITY, |m, s| if m > s[i] { m } else { s[i] });
                let (sum, res) =
                    kv.iter()
                        .zip(score)
                        .fold((0.0f32, 0.0f32), |(sum, res), (k, s)| {
                            let w = (s[i] - mx).exp();
                            (sum + w, w.mul_add(k[i], res))
                        });
                res / sum
            })
            .collect()
    }

    /// The butterfly `warp::reduce_sum_f32` runs over 32 lane values (xor
    /// 16, 8, 4, 2, 1, own + partner): lane 0's result.
    fn butterfly(lanes: &[f32]) -> f32 {
        let mut v = [0.0f32; 32];
        v.copy_from_slice(lanes);
        for off in [16, 8, 4, 2, 1] {
            let prev = v;
            for (l, s) in v.iter_mut().enumerate() {
                *s = prev[l] + prev[l ^ off];
            }
        }
        v[0]
    }

    /// Our norm of one row as the kernels compute it: each thread's (lane's)
    /// four values squared and summed by fused multiply-adds from `+0`, the
    /// warp butterfly, the warp sums as `(w0 + w1) + (w2 + w3)` (512 values:
    /// four warps; 128: one), `1 / sqrt(sum / n + eps)` (written out here,
    /// not `elem::rms_scale`, which the kernels inline), then
    /// `(scale · gain) · x`.
    fn norm_ours(x: &[f32], gain: &[f32], eps: f32) -> Result<Vec<f32>, GateError> {
        let part: Vec<f32> = x
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| c.iter().fold(0.0f32, |a, &v| v.mul_add(v, a)))
            .collect();
        let warps: Vec<f32> = part.chunks(32).map(butterfly).collect();
        let sum = match warps.as_slice() {
            [w] => *w,
            [w0, w1, w2, w3] => (w0 + w1) + (w2 + w3),
            w => return Err(format!("a row of {} values: {} warps", x.len(), w.len()).into()),
        };
        let mean = sum / x.len() as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        Ok(x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect())
    }

    /// ik's `FUSED_RMS_NORM` of one row: f32 squares summed serially in f64,
    /// the mean rounded to f32, `1/sqrtf(mean + eps)`, then `(scale · c) · x`.
    fn norm_ik(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
        let sum = x.iter().fold(0.0f64, |a, &v| a + f64::from(v * v));
        let mean = (sum / x.len() as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect()
    }

    /// The tail rope of one row: its last `cs.len()` values turned in pairs
    /// by `cs` (`[cos, sin, …]`), every product and sum rounded on its own —
    /// the kernels' `rope_pair_rn` and ik's unfused rotation alike.
    fn rotate(x: &[f32], cs: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        let tail = y.len() - cs.len();
        for (p, &[c, s]) in y[tail..]
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(cs.as_chunks::<2>().0)
        {
            let [x0, x1] = *p;
            *p = [x0 * c - x1 * s, x0 * s + x1 * c];
        }
        y
    }

    /// ik's `fast_ht` (`iqk_cpu_ops.cpp`): butterflies at `h = 1, 2, 4, …`,
    /// `x + y` and `x − y`, the scale multiplied by `0.707106781f` (the f32
    /// `FRAC_1_SQRT_2`) per stage, then every value by it.
    fn fast_ht(v: &mut [f32]) -> f32 {
        let n = v.len();
        let mut scale = 1.0f32;
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (x, y) = (v[j], v[j + h]);
                    v[j] = x + y;
                    v[j + h] = x - y;
                }
            }
            scale *= FRAC_1_SQRT_2;
            h <<= 1;
        }
        for x in v.iter_mut() {
            *x *= scale;
        }
        scale
    }

    /// The YaRN table of each position, `n_dims` values apiece: ours
    /// ([`RopeTable::push`]) or ik's (ggml's recipe filling a `ne0`-value
    /// head's cache, of which the tail reads the first `n_dims`).
    fn tables(meta: &Meta, pos: &[u32], ik_ne0: Option<usize>) -> Result<Vec<Vec<f32>>, GateError> {
        let ours = RopeTable::new(&meta.yarn)?;
        Ok(pos
            .iter()
            .map(|&p| match ik_ne0 {
                None => {
                    let mut cs = Vec::with_capacity(meta.n_dims);
                    ours.push(p, Direction::Forward, &mut cs);
                    cs
                }
                Some(ne0) => ggml_rope_cache(&meta.yarn, p, ne0, Direction::Forward)
                    .into_iter()
                    .take(meta.n_dims)
                    .collect(),
            })
            .collect())
    }

    /// Our q8_1 activation (`cores::q8_quad`) of `k`-value columns: per
    /// 128-value block `d = amax/127` (1 for an all-zero block), `q =
    /// round(x/d)` half away from zero, clamped to ±127 — each value's exact
    /// `q·d`.
    fn q8_1_exact(x: &[f32]) -> Vec<f64> {
        x.chunks(128)
            .flat_map(|b| {
                let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                b.iter()
                    .map(move |&v| f64::from((v / d).round().clamp(-127.0, 127.0)) * f64::from(d))
            })
            .collect()
    }

    /// ik's q8_K activation as its AVX2 build quantizes it
    /// (`iqk_quantize_row_q8_K`): per 256-value block `d = max|x| / 127`,
    /// `q = rne(x · (127 / max|x|))` (the product rounded to f32 first), all
    /// zero for an all-zero block — each value's exact `q·d`.
    fn q8k_exact(x: &[f32]) -> Vec<f64> {
        x.chunks(256)
            .flat_map(|b| {
                let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                b.iter().map(move |&v| {
                    let q = (id * v).round_ties_even().clamp(-128.0, 127.0);
                    f64::from(q) * f64::from(d)
                })
            })
            .collect()
    }

    /// A q3_K row as ik's AVX2 dot sees it: per weight the sub-block's
    /// `|d · (scale − 32)|` and its 3-bit value `q ∈ [−4, 3]`
    /// (`gguf::quant`'s `dequant_q3_k` walk), so the weight is `±that · q`
    /// and the dot adds `(q + 4)` and `−4` parts separately.
    fn q3k_fields(row: &[u8]) -> Vec<(f32, i8)> {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let mut out = Vec::with_capacity(row.len() / 110 * 256);
        for blk in row.as_chunks::<110>().0 {
            let (hm, qs) = (&blk[0..32], &blk[32..96]);
            let d_all = half_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
            let mut aux = [0u32; 4];
            for (i, a) in aux.iter_mut().take(3).enumerate() {
                let s = &blk[96 + 4 * i..100 + 4 * i];
                *a = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
            }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let scales: [i8; 16] = core::array::from_fn(|i| aux[i / 4].to_le_bytes()[i % 4] as i8);
            let (mut is, mut m) = (0usize, 1u8);
            for half in 0..2 {
                let q = &qs[32 * half..32 * half + 32];
                let mut shift = 0u32;
                for _field in 0..4 {
                    for half16 in 0..2 {
                        let dl = (d_all * (i32::from(scales[is]) - 32) as f32).abs();
                        is += 1;
                        for l in 0..16 {
                            let qv = ((q[16 * half16 + l] >> shift) & 3) as i8;
                            let hv = if hm[16 * half16 + l] & m != 0 { 0 } else { 4 };
                            out.push((dl, qv - hv));
                        }
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        out
    }

    // -------------------------------------------------------- measures

    /// Ulps between two finite f32, through integers ordered like the
    /// floats they encode.
    fn ulps(a: f32, b: f32) -> u32 {
        let key = |v: f32| {
            let b = v.to_bits().cast_signed();
            if b < 0 { i32::MIN - b } else { b }
        };
        key(a).abs_diff(key(b))
    }

    fn max_ulps(a: &[f32], b: &[f32]) -> u32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| ulps(x, y))
            .max()
            .unwrap_or(0)
    }

    fn same_bits(a: &[f32], b: &[f32]) -> usize {
        a.iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count()
    }

    /// Values of `a` off `b` by more than their band, and the largest
    /// `|a − b| / band`.
    fn over(a: &[f32], b: &[f32], band: &[f64]) -> (usize, f64) {
        a.iter()
            .zip(b)
            .zip(band)
            .fold((0, 0.0f64), |(n, w), ((&x, &y), &bd)| {
                let d = (f64::from(x) - f64::from(y)).abs();
                (
                    n + usize::from(d > bd),
                    w.max(d / bd.max(f64::MIN_POSITIVE)),
                )
            })
    }

    fn list(v: &[impl ToString]) -> String {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    fn norm2(v: &[f32]) -> f64 {
        v.iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt()
    }

    // ------------------------------------------------------------ sites

    /// A q3_K weight on the device and on the host.
    struct Weight {
        k: usize,
        rows: usize,
        dev: DeviceTensor<u32>,
        /// The dequantized rows, exact in f32, row-major.
        deq: Vec<f32>,
        /// Per weight, as ik's dot splits it ([`q3k_fields`]).
        fields: Vec<(f32, i8)>,
    }

    /// What every site reads besides its set.
    struct Cx {
        gpu: Gpu,
        comp: CompressKernels,
        idx: IndexKeyKernels,
        split: Split,
        meta: Meta,
        weights: HashMap<String, Weight>,
        sites: u32,
        failed: u32,
    }

    impl Cx {
        fn tally(&mut self, pass: bool) {
            self.sites += 1;
            self.failed += u32::from(!pass);
        }

        /// The q3_K weight `name`, uploaded and decoded once.
        fn weight(&mut self, name: &str) -> Result<&Weight, GateError> {
            if !self.weights.contains_key(name) {
                let (s, t) = self
                    .split
                    .find(name)
                    .ok_or_else(|| format!("{name} is not in the model file"))?;
                let &[k, rows] = t.dims.as_slice() else {
                    return Err(format!("{name} has dims {:?}, want [K, rows]", t.dims).into());
                };
                let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?);
                if t.ty != GgmlType::Q3_K || !k.is_multiple_of(512) {
                    return Err(format!(
                        "{name} is {:?} K={k}, want q3_K with K a multiple of 512",
                        t.ty
                    )
                    .into());
                }
                let bytes = self
                    .split
                    .shard(s)
                    .ok_or("shard index out of range")?
                    .data(t)?;
                let row_bytes = k / 256 * 110;
                let mut deq = vec![0.0f32; rows * k];
                for (r, out) in deq.chunks_mut(k).enumerate() {
                    dequant_row(
                        GgmlType::Q3_K,
                        &bytes[r * row_bytes..(r + 1) * row_bytes],
                        out,
                    )?;
                }
                let fields = q3k_fields(bytes);
                let exact = deq
                    .iter()
                    .zip(&fields)
                    .all(|(&w, &(s, q))| w.abs().to_bits() == (s * f32::from(q)).abs().to_bits());
                if !exact {
                    return Err(
                        format!("{name}: the q3_K field walk disagrees with dequant_row").into(),
                    );
                }
                let words = bytes_to_words(bytes);
                let dev = DeviceTensor::upload(self.gpu.stream(), &words, rows, row_bytes / 4)?;
                self.weights.insert(
                    name.to_string(),
                    Weight {
                        k,
                        rows,
                        dev,
                        deq,
                        fields,
                    },
                );
            }
            self.weights
                .get(name)
                .ok_or_else(|| format!("{name} was just inserted").into())
        }
    }

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        let meta = Meta::read(&split)?;
        let gpu = Gpu::new()?;
        let comp = CompressKernels::load(gpu.context())?;
        let idx = IndexKeyKernels::load(gpu.context())?;
        println!(
            "gate_deepseek41_comp: device {} — n_dims {} eps {:e}; yarn rope {:?}; ht_scale {:e}",
            gpu.device_name()?,
            meta.n_dims,
            meta.eps,
            meta.yarn,
            HT_SCALE
        );
        let mut cx = Cx {
            gpu,
            comp,
            idx,
            split,
            meta,
            weights: HashMap::new(),
            sites: 0,
            failed: 0,
        };
        let cpu = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = vec![(cpu.set_name(Set::Cpu)?.to_string(), cpu.open(Set::Cpu)?)];
        for name in STEP_SETS {
            sets.push((name.to_string(), open_set(name)?));
        }
        for (label, man) in &sets {
            gate_set(&mut cx, label, man)?;
        }
        let pass = cx.failed == 0;
        println!(
            "gate_deepseek41_comp: {} sites across {} sets, {} failed — {}",
            cx.sites,
            sets.len(),
            cx.failed,
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// One source layer's rows in a set, found from its kv projection.
    struct Source<'a> {
        layer: usize,
        tag: &'static str,
        /// The step's first position.
        pos0: usize,
        st: &'a StreamStep,
        geom: CompGeom,
        /// The projections' input, `attn_norm-L`.
        x_row: &'a RefRow,
        kv_row: &'a RefRow,
        score_row: Option<&'a RefRow>,
        /// The ring before the step (a step set's state inputs; the prefill
        /// dumps none).
        ring: Option<(&'a RefRow, &'a RefRow)>,
        /// The ring after the step (`SET_ROWS`), ratio above 1.
        ring_after: Option<(&'a RefRow, &'a RefRow)>,
    }

    /// Plan the set's step and gate every source layer in it.
    fn gate_set(cx: &mut Cx, label: &str, man: &RefManifest) -> Result<(), GateError> {
        let head = Header::read(&man.dir)?;
        let (pos0, tokens, before) = head.step()?;
        let planner = Planner::from_file(&cx.split, head.ctx)?;
        if planner.stream_ratios().len() != STREAMS.len() {
            return Err(format!("the file has streams {:?}", planner.stream_ratios()).into());
        }
        let mut plan = StepPlan::default();
        planner.plan_into(tokens, pos0, before, &mut plan)?;
        let touch = FirstTouch::read(&man.dir)?;
        let mut layers = Vec::new();
        for (at, r) in man.tensors.iter().enumerate() {
            let layer = r
                .src0
                .as_deref()
                .and_then(|w| w.strip_prefix("blk."))
                .and_then(|w| w.strip_suffix(".attn_compressor_kv.weight"))
                .and_then(|l| l.parse::<usize>().ok());
            if let (Some(l), "MUL_MAT") = (layer, r.op.as_str()) {
                layers.push((at, l));
            }
        }
        println!(
            "set {label}: {} (build {}) — a step of {} at position {pos0}, context {}; source layers {}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            plan.len(),
            head.ctx,
            list(&layers.iter().map(|&(_, l)| l).collect::<Vec<_>>())
        );
        for (at, layer) in layers {
            let s = planner
                .layer_stream(layer)
                .ok_or_else(|| format!("layer {layer} has a compressor and no stream"))?;
            let st = &plan.streams[s];
            let tag = STREAMS[s];
            let r = st.ratio as usize;
            let kv_row = &man.tensors[at];
            let score_row = man.tensors.iter().find(|t| {
                t.op == "MUL_MAT"
                    && t.src0.as_deref()
                        == Some(&format!("blk.{layer}.attn_compressor_gate.weight"))
            });
            if (r > 1) != score_row.is_some() {
                return Err(format!(
                    "layer {layer}: ratio {r} and a gate projection {}",
                    score_row.is_some()
                )
                .into());
            }
            let (_, x_row) = last_before(man, at, kv_row.src1.as_deref().ok_or("no src1")?)?;
            let ring_after = if r > 1 {
                let (_, k) = node(
                    man,
                    &format!("{tag}_k_state_persist-{layer}"),
                    0,
                    "SET_ROWS",
                )?;
                let (_, s) = node(
                    man,
                    &format!("{tag}_score_state_persist-{layer}"),
                    0,
                    "SET_ROWS",
                )?;
                Some((k, s))
            } else {
                None
            };
            let ring = match (ring_after, score_row) {
                (Some((k_after, s_after)), Some(score)) => {
                    // The ring is the f32 input of its shape the source CONCAT
                    // touches first, or the persist when no group completes.
                    let ring_ne = [compress::WIDTH as u64, r as u64, 1, 1];
                    let is_ring = |i: &&RefRow| i.ty == "f32" && i.ne == ring_ne;
                    let first = |concat: &str, src1: &str, persist: &RefRow| {
                        let by_concat = man.tensors.iter().find(|t| {
                            t.op == "CONCAT" && t.name == concat && t.src1.as_deref() == Some(src1)
                        });
                        by_concat
                            .and_then(|c| {
                                touch
                                    .inputs_of(man, &c.name, c.occurrence)
                                    .into_iter()
                                    .find(is_ring)
                            })
                            .or_else(|| {
                                touch
                                    .inputs_of(man, &persist.name, persist.occurrence)
                                    .into_iter()
                                    .find(is_ring)
                            })
                    };
                    match (
                        first(&format!("{tag}_source_kv"), &kv_row.name, k_after),
                        first(&format!("{tag}_source_score"), &score.name, s_after),
                    ) {
                        (Some(k), Some(s)) => Some((k, s)),
                        (None, None) => None,
                        _ => {
                            return Err(
                                format!("layer {layer}: one ring input without the other").into()
                            );
                        }
                    }
                }
                _ => None,
            };
            let t = plan.len();
            let geom = CompGeom {
                ratio: r,
                max_groups: t.div_ceil(r),
                tokens: t,
                rows: usize::try_from(head.ctx)?.div_ceil(r),
            };
            let src = Source {
                layer,
                tag,
                pos0: pos0 as usize,
                st,
                geom,
                x_row,
                kv_row,
                score_row,
                ring,
                ring_after,
            };
            gate_source(cx, label, man, &src)?;
        }
        Ok(())
    }

    /// Every site of one source layer.
    fn gate_source(
        cx: &mut Cx,
        label: &str,
        man: &RefManifest,
        src: &Source,
    ) -> Result<(), GateError> {
        let x = ref_tensor_logical_in(&man.dir, src.x_row)?;
        let pass = gemv_site(cx, label, man, src.kv_row, &x, None)?;
        cx.tally(pass);
        let kv = ref_tensor_logical_in(&man.dir, src.kv_row)?;
        let score = match src.score_row {
            Some(row) => {
                let pass = gemv_site(cx, label, man, row, &x, None)?;
                cx.tally(pass);
                ref_tensor_logical_in(&man.dir, row)?
            }
            None => Vec::new(),
        };
        let pass = comp_site(cx, label, man, src, &kv, &score)?;
        cx.tally(pass);
        let pass = if src.st.groups() > 0 {
            idx_site(cx, label, man, src)?
        } else {
            idx_idle(cx, label, src)?
        };
        cx.tally(pass);
        Ok(())
    }

    /// One q3_K projection: `row` (a MUL_MAT of weight `row.src0`) of `x`,
    /// `m` token-major columns of K. With `alt`, an input the dump must not
    /// be the projection of, in the columns flagged.
    ///
    /// Layer 1 is our gemv against the exact dot of its own q8_1 values, in
    /// `KERNEL_BAND`. Layer 2 is ik's dot of its q8_K values (exact in f64)
    /// against the dump, per value within `γ(2·n_sb + 4)` of the magnitudes
    /// its AVX2 kernel sums: per super-block the `(q + 4)` part and the `−4`
    /// part each enter a lane accumulator by one fused multiply-add after one
    /// rounded scale product, and eight lanes meet in a three-level sum.
    /// Layer 3 is our gemv against the dump, per value within the two
    /// activations' distances to `x` weighted by `|w|` (each within half its
    /// own step of `x`), plus the other two layers' bands.
    fn gemv_site(
        cx: &mut Cx,
        label: &str,
        man: &RefManifest,
        row: &RefRow,
        x: &[f32],
        alt: Option<(&[f32], &[bool])>,
    ) -> Result<bool, GateError> {
        let wname = row.src0.clone().ok_or("a MUL_MAT row without src0")?;
        cx.weight(&wname)?;
        let w = cx.weights.get(&wname).ok_or("the weight was just loaded")?;
        let (k, rows) = (w.k, w.rows);
        let m = x.len() / k;
        if m == 0 || m > 8 || x.len() != m * k || row.ne != [rows as u64, m as u64, 1, 1] {
            return Err(format!(
                "{}: {:?} from {} values of K={k}",
                row.name,
                row.ne,
                x.len()
            )
            .into());
        }
        let want = ref_tensor_logical_in(&man.dir, row)?;

        let stream = cx.gpu.stream();
        let x_dev = DeviceBuffer::from_host(stream, x)?;
        let mut runs = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut act = Q8Act::with_k(stream, m, k)?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, rows * m)?;
            cx.gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
            cx.gpu.enqueue_gemv_q3k(&w.dev, &act, &mut y)?;
            stream.synchronize()?;
            let by_row = y.to_host_vec(stream)?;
            let mut tok = vec![0.0f32; rows * m];
            for (j, r) in by_row.chunks(m).enumerate() {
                for (t, &v) in r.iter().enumerate() {
                    tok[t * rows + j] = v;
                }
            }
            runs.push(tok);
        }
        let got = &runs[0];
        let rerun = bits_equal(got, &runs[1]);

        let xo = q8_1_exact(x);
        let xi = q8k_exact(x);
        let n_sb = k / 256;
        let (mut ours, mut sim, mut b2, mut quant) = (
            Vec::with_capacity(rows * m),
            Vec::with_capacity(rows * m),
            Vec::with_capacity(rows * m),
            Vec::with_capacity(rows * m),
        );
        for t in 0..m {
            let col = t * k..(t + 1) * k;
            for j in 0..rows {
                let (wr, fr) = (&w.deq[j * k..(j + 1) * k], &w.fields[j * k..(j + 1) * k]);
                let (mut so, mut si, mut mag, mut qd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (((&wv, &(sc, q)), (&a, &b)), &xv) in wr
                    .iter()
                    .zip(fr)
                    .zip(xo[col.clone()].iter().zip(&xi[col.clone()]))
                    .zip(&x[col.clone()])
                {
                    let wv = f64::from(wv);
                    so += wv * a;
                    si += wv * b;
                    mag += f64::from(sc) * b.abs() * (f64::from(q + 4) + 4.0);
                    qd += wv.abs() * ((a - f64::from(xv)).abs() + (b - f64::from(xv)).abs());
                }
                ours.push(so as f32);
                sim.push(si);
                b2.push(gamma(2 * n_sb + 4) * mag);
                quant.push(qd);
            }
        }
        let l1 = max_rel_err(got, &ours)?;
        let n = rows * m;
        let (sim_over, sim_worst) = over64(&want, &sim, &b2);
        let big = ours.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let b3: Vec<f64> = quant
            .iter()
            .zip(&b2)
            .map(|(&q, &b)| q + b + f64::from(KERNEL_BAND) * f64::from(big))
            .collect();
        let got64: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
        let (k_over, k_worst) = over64(&want, &got64, &b3);
        let ik_rel = max_rel_err(got, &want)?;

        // The input the dump must not come from: ik's dot of it misses the
        // dump by more than layer 2's band somewhere in every flagged column.
        let mut alt_note = String::new();
        let mut alt_ok = true;
        if let Some((xa, cols)) = alt {
            let xa_i = q8k_exact(xa);
            let big_want = want.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let mut worst_rel = 0.0f64;
            for (t, &flag) in cols.iter().enumerate().take(m) {
                let col = t * k..(t + 1) * k;
                let mut explained = true;
                for j in 0..rows {
                    let (wr, fr) = (&w.deq[j * k..(j + 1) * k], &w.fields[j * k..(j + 1) * k]);
                    let (mut sa, mut mag) = (0.0f64, 0.0f64);
                    for ((&wv, &(sc, q)), &b) in wr.iter().zip(fr).zip(&xa_i[col.clone()]) {
                        sa += f64::from(wv) * b;
                        mag += f64::from(sc) * b.abs() * (f64::from(q + 4) + 4.0);
                    }
                    let d = (f64::from(want[t * rows + j]) - sa).abs();
                    explained &= d <= gamma(2 * n_sb + 4) * mag;
                    if flag {
                        worst_rel = worst_rel.max(d / f64::from(big_want));
                    }
                }
                alt_ok &= !(flag && explained);
            }
            alt_note = format!(" roped_input_rel={worst_rel:.3e} roped_input_rejected={alt_ok}");
        }

        let pass = l1 <= KERNEL_BAND && rerun && sim_over == 0 && k_over == 0 && alt_ok;
        println!(
            "gemv set={label} site={}/{} weight={wname} k={k} rows={rows} m={m} ours_rel={l1:.3e} \
             (band {KERNEL_BAND:e}) bit_identical_rerun={rerun} ik_sim_over={sim_over}/{n} \
             ik_sim_worst={sim_worst:.3} ik_rel={ik_rel:.3e} over={k_over}/{n} \
             worst={k_worst:.3}{alt_note} {}",
            row.name,
            row.occurrence,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Values of `a` off `b` by more than their band, and the largest
    /// `|a − b| / band`.
    fn over64(a: &[f32], b: &[f64], band: &[f64]) -> (usize, f64) {
        a.iter()
            .zip(b)
            .zip(band)
            .fold((0, 0.0f64), |(n, w), ((&x, &y), &bd)| {
                let d = (f64::from(x) - y).abs();
                (
                    n + usize::from(d > bd),
                    w.max(d / bd.max(f64::MIN_POSITIVE)),
                )
            })
    }

    /// One compressor launch's results, read back.
    struct CompOut {
        pre: Vec<f32>,
        cache: Vec<u16>,
        ring_kv: Vec<f32>,
        ring_score: Vec<f32>,
    }

    /// The inputs of one compressor launch, uploaded once for both runs.
    struct CompIn<'a> {
        words: &'a [u32],
        kv: &'a [f32],
        score: &'a [f32],
        gain: &'a [f32],
        cs: &'a [f32],
        ring: (&'a [f32], &'a [f32]),
    }

    /// One launch into fresh outputs: the ring from `inp.ring`, the pre-rope
    /// rows poisoned, the cache filled with the sentinel.
    fn comp_run(cx: &Cx, geom: CompGeom, inp: &CompIn) -> Result<CompOut, GateError> {
        let stream = cx.gpu.stream();
        let width = compress::WIDTH;
        let step = DeviceBuffer::from_host(stream, inp.words)?;
        let kv = DeviceBuffer::from_host(stream, inp.kv)?;
        let gain = DeviceBuffer::from_host(stream, inp.gain)?;
        let cs = DeviceBuffer::from_host(stream, inp.cs)?;
        let mut pre = DeviceBuffer::from_host(stream, &vec![POISON; geom.max_groups * width])?;
        let mut cache =
            DeviceTensor::upload(stream, &vec![SENTINEL; geom.rows * width], geom.rows, width)?;
        let (ring_kv, ring_score) = if geom.ratio > 1 {
            let score = DeviceBuffer::from_host(stream, inp.score)?;
            let mut ring_kv = DeviceBuffer::from_host(stream, inp.ring.0)?;
            let mut ring_score = DeviceBuffer::from_host(stream, inp.ring.1)?;
            cx.comp.enqueue_pool(
                stream,
                PoolArgs {
                    geom,
                    step: &step,
                    kv: &kv,
                    score: &score,
                    gain: &gain,
                    cs: &cs,
                    eps: cx.meta.eps,
                    n_dims: cx.meta.n_dims,
                    ring_kv: &mut ring_kv,
                    ring_score: &mut ring_score,
                    pre: &mut pre,
                    cache: &mut cache,
                },
            )?;
            stream.synchronize()?;
            (
                ring_kv.to_host_vec(stream)?,
                ring_score.to_host_vec(stream)?,
            )
        } else {
            cx.comp.enqueue_rows(
                stream,
                RowsArgs {
                    geom,
                    step: &step,
                    kv: &kv,
                    gain: &gain,
                    cs: &cs,
                    eps: cx.meta.eps,
                    n_dims: cx.meta.n_dims,
                    pre: &mut pre,
                    cache: &mut cache,
                },
            )?;
            stream.synchronize()?;
            (Vec::new(), Vec::new())
        };
        Ok(CompOut {
            pre: pre.to_host_vec(stream)?,
            cache: cache.buf().to_host_vec(stream)?,
            ring_kv,
            ring_score,
        })
    }

    /// The step buffer of a source's stream, packed from the plan.
    fn step_words(src: &Source) -> Result<Vec<u32>, GateError> {
        let st = src.st;
        let mut words = vec![0u32; src.geom.words()];
        src.geom.pack(
            &StepInts {
                write_row: &st.state_write,
                read: &st.state_read,
                persist_src: &st.persist_src,
                persist_dst: &st.persist_dst,
            },
            &mut words,
        )?;
        Ok(words)
    }

    /// Our rope tables for the groups the step completes, poisoned past them
    /// (a kernel reads only a completed group's table).
    fn step_tables(cx: &Cx, src: &Source) -> Result<(Vec<Vec<f32>>, Vec<f32>), GateError> {
        let nd = cx.meta.n_dims;
        let ours = tables(&cx.meta, &src.st.write_pos, None)?;
        let mut cs = vec![POISON; src.geom.max_groups * nd];
        for (g, t) in ours.iter().enumerate() {
            cs[g * nd..(g + 1) * nd].copy_from_slice(t);
        }
        Ok((ours, cs))
    }

    /// The ring with the plan's kept tokens copied in: `slots` rows of
    /// `WIDTH`, slot `persist_dst[j]` taking token `persist_src[j]` of `rows`.
    fn persisted(ring: &[f32], rows: &[f32], st: &StreamStep) -> Vec<f32> {
        let w = compress::WIDTH;
        let mut out = ring.to_vec();
        for (&s, &d) in st.persist_src.iter().zip(&st.persist_dst) {
            let (s, d) = (s as usize, d as usize);
            out[d * w..(d + 1) * w].copy_from_slice(&rows[s * w..(s + 1) * w]);
        }
        out
    }

    /// The compressor at one source layer: the kernel's pre-rope rows, cache
    /// rows and ring against our rule, ik's rule against the dump's chain
    /// (`<stream>_state_compress-L` occurrences 0–2, the cache write, the
    /// ring's `SET_ROWS`), and the kernel against the dump.
    fn comp_site(
        cx: &Cx,
        label: &str,
        man: &RefManifest,
        src: &Source,
        kv: &[f32],
        score: &[f32],
    ) -> Result<bool, GateError> {
        let (geom, st, layer, tag) = (src.geom, src.st, src.layer, src.tag);
        let (r, width, eps) = (geom.ratio, compress::WIDTH, cx.meta.eps);
        let groups = st.groups();
        if kv.len() != geom.tokens * width || (r > 1 && score.len() != kv.len()) {
            return Err(format!(
                "layer {layer}: {} projection values for {} tokens",
                kv.len(),
                geom.tokens
            )
            .into());
        }
        let gain = split_f32(
            &cx.split,
            &format!("blk.{layer}.attn_compressor_norm.weight"),
            width,
        )?;
        let words = step_words(src)?;
        let (ours_tab, cs) = step_tables(cx, src)?;
        let (ring_kv0, ring_sc0) = match src.ring {
            Some((k, s)) => (
                ref_tensor_logical_in(&man.dir, k)?,
                ref_tensor_logical_in(&man.dir, s)?,
            ),
            None => (vec![POISON; r * width], vec![POISON; r * width]),
        };
        if ring_kv0.len() != r * width || ring_sc0.len() != r * width {
            return Err(format!(
                "layer {layer}: rings of {} and {} values for ratio {r}",
                ring_kv0.len(),
                ring_sc0.len()
            )
            .into());
        }
        let inp = CompIn {
            words: &words,
            kv,
            score,
            gain: &gain,
            cs: &cs,
            ring: (&ring_kv0, &ring_sc0),
        };
        let out = comp_run(cx, geom, &inp)?;
        let out2 = comp_run(cx, geom, &inp)?;
        let rerun = bits_equal(&out.pre, &out2.pre)
            && out.cache == out2.cache
            && bits_equal(&out.ring_kv, &out2.ring_kv)
            && bits_equal(&out.ring_score, &out2.ring_score);

        // Layer 1: our rule on the host, from the same sources.
        let row_of = |buf: &[f32], i: usize| buf[i * width..(i + 1) * width].to_vec();
        let source = |s: usize| -> (Vec<f32>, Vec<f32>) {
            if s < r {
                (row_of(&ring_kv0, s), row_of(&ring_sc0, s))
            } else if r > 1 {
                (row_of(kv, s - r), row_of(score, s - r))
            } else {
                (row_of(kv, s - r), Vec::new())
            }
        };
        let mut host_pre = Vec::with_capacity(groups * width);
        let mut spread = Vec::with_capacity(groups * width);
        for g in 0..groups {
            let reads = &st.state_read[g * r..(g + 1) * r];
            let rows: Vec<(Vec<f32>, Vec<f32>)> =
                reads.iter().map(|&s| source(s as usize)).collect();
            let y = if r == 1 {
                rows[0].0.clone()
            } else {
                let k: Vec<&[f32]> = rows.iter().map(|(k, _)| k.as_slice()).collect();
                let s: Vec<&[f32]> = rows.iter().map(|(_, s)| s.as_slice()).collect();
                ds4_comp(&k, &s)
            };
            spread.extend(
                (0..width).map(|i| rows.iter().fold(0.0f32, |a, (k, _)| a.max(k[i].abs()))),
            );
            host_pre.extend(norm_ours(&y, &gain, eps)?);
        }
        let got_pre = &out.pre[..groups * width];
        let (pre_l1, pre_ok) = if groups == 0 {
            (0.0, true)
        } else if r == 1 {
            (
                max_rel_err(got_pre, &host_pre)?,
                bits_equal(got_pre, &host_pre),
            )
        } else {
            let e = max_rel_err(got_pre, &host_pre)?;
            (e, e <= KERNEL_BAND)
        };
        let pre_same = same_bits(got_pre, &host_pre);
        let pre_ulp = max_ulps(got_pre, &host_pre);
        // Each written row is the f16 of the kernel's own pre-rope row turned
        // by our table; every other row keeps the sentinel.
        let written: Vec<usize> = st.state_write.iter().map(|&w| w as usize).collect();
        let mut rows_exact = true;
        for (g, &w) in written.iter().enumerate() {
            let turned = rotate(&out.pre[g * width..(g + 1) * width], &ours_tab[g]);
            let bits: Vec<u16> = turned.iter().map(|&v| f32_to_f16_bits(v)).collect();
            rows_exact &= out.cache[w * width..(w + 1) * width] == bits[..];
        }
        let untouched = (0..geom.rows).filter(|w| !written.contains(w)).all(|w| {
            out.cache[w * width..(w + 1) * width]
                .iter()
                .all(|&h| h == SENTINEL)
        });
        let ring_exact = r == 1
            || (bits_equal(&out.ring_kv, &persisted(&ring_kv0, kv, st))
                && bits_equal(&out.ring_score, &persisted(&ring_sc0, score, st)));
        let rule_ok = pre_ok && rows_exact && untouched && ring_exact && rerun;

        // Layer 2: ik's rule against the dump's chain.
        let chain = node(man, &format!("{tag}_state_compress-{layer}"), 0, "DS4_COMP");
        let mut notes = Vec::new();
        let mut sim_ok = true;
        let mut dump_ok = true;
        if groups == 0 {
            // ik builds no pooling for a step that completes no group.
            sim_ok &= chain.is_err();
            notes.push(format!("no_pool_node={}", chain.is_err()));
        } else {
            let (at_pool, pooled) = chain?;
            let (_, normed) = node(man, &pooled.name, 1, "FUSED_RMS_NORM")?;
            let (_, roped) = node(man, &pooled.name, 2, "ROPE")?;
            let (_, src_kv) = last_before(man, at_pool, pooled.src0.as_deref().ok_or("no src0")?)?;
            let (_, src_sc) = last_before(man, at_pool, pooled.src1.as_deref().ok_or("no src1")?)?;
            let (at_write, write) = node(man, &format!("{tag}_k_write-{layer}"), 0, "SET_ROWS")?;
            let before = cache_before(man, at_write)?;
            let d_pool = ref_tensor_logical_in(&man.dir, pooled)?;
            let d_norm = ref_tensor_logical_in(&man.dir, normed)?;
            let d_rope = ref_tensor_logical_in(&man.dir, roped)?;
            let d_skv = ref_tensor_logical_in(&man.dir, src_kv)?;
            let d_ssc = ref_tensor_logical_in(&man.dir, src_sc)?;
            if d_pool.len() != groups * width || d_skv.len() != (r + geom.tokens) * width {
                return Err(format!(
                    "layer {layer}: the dump pools {} values from {} for {groups} group(s)",
                    d_pool.len(),
                    d_skv.len()
                )
                .into());
            }
            // The concatenated sources are the ring (where the set dumps it)
            // then this step's projections.
            let sources_are = bits_equal(&d_skv[r * width..], kv)
                && (r == 1 || bits_equal(&d_ssc[r * width..], score))
                && src.ring.is_none_or(|_| {
                    bits_equal(&d_skv[..r * width], &ring_kv0)
                        && bits_equal(&d_ssc[..r * width], &ring_sc0)
                });
            let mut sim_pool = Vec::with_capacity(groups * width);
            for g in 0..groups {
                let reads = &st.state_read[g * r..(g + 1) * r];
                let k: Vec<&[f32]> = reads
                    .iter()
                    .map(|&s| &d_skv[s as usize * width..][..width])
                    .collect();
                let s: Vec<&[f32]> = reads
                    .iter()
                    .map(|&s| &d_ssc[s as usize * width..][..width])
                    .collect();
                sim_pool.extend(ds4_comp(&k, &s));
            }
            let pool_same = same_bits(&sim_pool, &d_pool);
            // A ring slot is read only after a real token was kept in it:
            // the slot holds position `q = write_pos + k` of an earlier
            // step, the last of its residue before this one (`q + r >=
            // pos0`), and the dump's ring row there is not ik's zero start.
            let (mut ring_reads, mut kept_first) = (0usize, true);
            for (g, reads) in st.state_read.chunks(r).enumerate() {
                for (k, &s) in reads.iter().enumerate() {
                    let s = s as usize;
                    if s < r {
                        let q = st.write_pos[g] as usize + k;
                        let real = d_skv[s * width..(s + 1) * width].iter().any(|&v| v != 0.0);
                        ring_reads += 1;
                        kept_first &= q < src.pos0 && q % r == s && q + r >= src.pos0 && real;
                    }
                }
            }
            // A ratio-1 group is its token's projection, bit for bit, and
            // never reads the ring.
            let identity = r > 1
                || (st.state_read.iter().all(|&s| s as usize >= r) && {
                    let rows: Vec<f32> = st
                        .state_read
                        .iter()
                        .flat_map(|&s| kv[(s as usize - r) * width..][..width].iter().copied())
                        .collect();
                    bits_equal(&d_pool, &rows)
                });
            let mut sim_norm = Vec::with_capacity(groups * width);
            let mut sim_rope = Vec::with_capacity(groups * width);
            let ik_tab = tables(&cx.meta, &st.write_pos, Some(width))?;
            for g in 0..groups {
                sim_norm.extend(norm_ik(&d_pool[g * width..(g + 1) * width], &gain, eps));
                sim_rope.extend(rotate(&d_norm[g * width..(g + 1) * width], &ik_tab[g]));
            }
            let norm_same = same_bits(&sim_norm, &d_norm);
            let rope_same = same_bits(&sim_rope, &d_rope);
            let ik_after = f16_all(man, write)?;
            let ik_before = f16_all(man, before)?;
            let lands = ik_after.len() == geom.rows * width
                && ik_before.len() == ik_after.len()
                && (0..geom.rows).all(|w| {
                    let (a, b) = (
                        &ik_after[w * width..(w + 1) * width],
                        &ik_before[w * width..(w + 1) * width],
                    );
                    match written.iter().position(|&x| x == w) {
                        Some(g) => a
                            .iter()
                            .zip(&d_rope[g * width..(g + 1) * width])
                            .all(|(&h, &v)| h == f32_to_f16_bits(v)),
                        None => a == b,
                    }
                });
            let n = groups * width;
            sim_ok &= sources_are
                && pool_same == n
                && identity
                && kept_first
                && norm_same == n
                && rope_same == n
                && lands;
            notes.push(format!(
                "ik_sources_are_ring_and_batch={sources_are} ring_reads={ring_reads} \
                 ring_read_after_kept={kept_first} ik_sim_pool_same={pool_same}/{n} \
                 ratio1_identity={identity} ik_sim_norm_same={norm_same}/{n} \
                 ik_sim_rope_same={rope_same}/{n} write_lands_only_on_rows={lands}"
            ));

            // Layer 3: the pre-rope rows against ik's, per value within the
            // pooling's and the norm's bands; the cache rows equal wherever
            // the turned f32 rows are.
            let mut band = Vec::with_capacity(n);
            for g in 0..groups {
                let y = &d_pool[g * width..(g + 1) * width];
                let nrm = &d_norm[g * width..(g + 1) * width];
                let sum = y.iter().fold(0.0f64, |a, &v| a + f64::from(v * v));
                let scale = 1.0 / ((sum / width as f64) as f32 + eps).sqrt();
                let dy: Vec<f64> = spread[g * width..(g + 1) * width]
                    .iter()
                    .map(|&a| {
                        if r == 1 {
                            0.0
                        } else {
                            f64::from(pool_units(r) * U) * f64::from(a)
                        }
                    })
                    .collect();
                let rho = dy.iter().map(|d| d * d).sum::<f64>().sqrt() / norm2(y);
                for i in 0..width {
                    band.push(
                        f64::from((scale * gain[i]).abs()) * dy[i]
                            + (f64::from(NORM512_UNITS * U) + rho) * f64::from(nrm[i].abs()),
                    );
                }
            }
            let (k_over, k_worst) = over(got_pre, &d_norm, &band);
            let ik_rel = max_rel_err(got_pre, &d_norm)?;
            let ulp = max_ulps(got_pre, &d_norm);
            let (mut f16_same, mut on_equal) = (0usize, true);
            for (g, &w) in written.iter().enumerate() {
                let turned = rotate(&out.pre[g * width..(g + 1) * width], &ours_tab[g]);
                for i in 0..width {
                    let (a, b) = (out.cache[w * width + i], ik_after[w * width + i]);
                    f16_same += usize::from(a == b);
                    on_equal &= turned[i].to_bits() != d_rope[g * width + i].to_bits() || a == b;
                }
            }
            dump_ok &= k_over == 0 && on_equal;
            notes.push(format!(
                "ik_rel={ik_rel:.3e} over={k_over}/{n} worst={k_worst:.3} max_ulp={ulp} \
                 f16_same={f16_same}/{n} f16_equal_on_equal_f32={on_equal}"
            ));
        }
        // The ring after the step: ik's against the ring before with the
        // plan's kept tokens (layer 2), ours against ik's (layer 3).
        if let Some((k_after, s_after)) = src.ring_after {
            let d_k = ref_tensor_logical_in(&man.dir, k_after)?;
            let d_s = ref_tensor_logical_in(&man.dir, s_after)?;
            let kept: Vec<usize> = st.persist_dst.iter().map(|&d| d as usize).collect();
            let known = |slot: usize| src.ring.is_some() || kept.contains(&slot);
            let slots_eq = |a: &[f32], b: &[f32]| {
                (0..r).filter(|&s| known(s)).all(|s| {
                    bits_equal(
                        &a[s * width..(s + 1) * width],
                        &b[s * width..(s + 1) * width],
                    )
                })
            };
            let ik_keeps = slots_eq(&d_k, &persisted(&ring_kv0, kv, st))
                && slots_eq(&d_s, &persisted(&ring_sc0, score, st));
            let ours_keeps = slots_eq(&out.ring_kv, &d_k) && slots_eq(&out.ring_score, &d_s);
            sim_ok &= ik_keeps;
            dump_ok &= ours_keeps;
            notes.push(format!(
                "kept={}->{} ik_ring_after_is_persist={ik_keeps} ring_after_equals_ik={ours_keeps}",
                list(&st.persist_src),
                list(&kept)
            ));
        }
        let pass = rule_ok && sim_ok && dump_ok;
        println!(
            "comp set={label} layer={layer} stream={tag} ratio={r} groups={groups}/{} rows={} pos={} \
             write_rows={} reads={} pre_ours_rel={pre_l1:.3e} ({}) pre_same={pre_same}/{} \
             pre_max_ulp={pre_ulp} cache_rows_are_rope_of_pre={rows_exact} \
             unwritten_rows_sentinel={untouched} ring_exact={ring_exact} bit_identical_rerun={rerun} {} {}",
            geom.max_groups,
            geom.rows,
            list(&st.write_pos),
            list(&written),
            list(&st.state_read),
            if groups == 0 {
                "no group"
            } else if r == 1 {
                "bit-identical: no expf"
            } else {
                "band 1e-5: device expf"
            },
            groups * width,
            notes.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The index-key chain the dump holds for a source layer: the HADAMARD
    /// `lid_k_new-L` back through its rope, reshape, norm and projection,
    /// and the key cache's write.
    struct IdxChain<'a> {
        mm: &'a RefRow,
        norm: &'a RefRow,
        rope: &'a RefRow,
        hada: &'a RefRow,
        write: &'a RefRow,
        before: &'a RefRow,
    }

    impl<'a> IdxChain<'a> {
        fn read(man: &'a RefManifest, layer: usize, tag: &str) -> Result<IdxChain<'a>, GateError> {
            let (at_h, hada) = node(man, &format!("lid_k_new-{layer}"), 0, "HADAMARD")?;
            let (at_r, rope) = last_before(man, at_h, hada.src0.as_deref().ok_or("no src0")?)?;
            let (at_s, shaped) = last_before(man, at_r, rope.src0.as_deref().ok_or("no src0")?)?;
            let (at_n, norm) = last_before(man, at_s, shaped.src0.as_deref().ok_or("no src0")?)?;
            let (_, mm) = last_before(man, at_n, norm.src0.as_deref().ok_or("no src0")?)?;
            let (at_w, write) = node(man, &format!("lid_k_write-{layer}"), 0, "SET_ROWS")?;
            let before = cache_before(man, at_w)?;
            let want_mm = (
                format!("blk.{layer}.indexer.attn_k.weight"),
                format!("{tag}_state_compress-{layer}"),
            );
            if rope.op != "ROPE"
                || norm.op != "FUSED_RMS_NORM"
                || norm.src1.as_deref() != Some(&format!("blk.{layer}.indexer.k_norm.weight"))
                || mm.op != "MUL_MAT"
                || (mm.src0.as_deref(), mm.src1.as_deref())
                    != (Some(want_mm.0.as_str()), Some(want_mm.1.as_str()))
            {
                return Err(format!(
                    "layer {layer}: the key chain is {} <- {} <- {} ({:?} x {:?})",
                    rope.op, norm.op, mm.op, mm.src0, mm.src1
                )
                .into());
            }
            Ok(IdxChain {
                mm,
                norm,
                rope,
                hada,
                write,
                before,
            })
        }
    }

    /// One index-key launch into a fresh cache of sentinels, read back.
    fn idx_run(
        cx: &Cx,
        geom: CompGeom,
        words: &[u32],
        k: &[f32],
        gain: &[f32],
        cs: &[f32],
    ) -> Result<Vec<u16>, GateError> {
        let stream = cx.gpu.stream();
        let w = index_key::WIDTH;
        let step = DeviceBuffer::from_host(stream, words)?;
        let k = DeviceBuffer::from_host(stream, k)?;
        let gain = DeviceBuffer::from_host(stream, gain)?;
        let cs = DeviceBuffer::from_host(stream, cs)?;
        let mut cache = DeviceTensor::upload(stream, &vec![SENTINEL; geom.rows * w], geom.rows, w)?;
        cx.idx.enqueue_index_key(
            stream,
            IndexKeyArgs {
                geom,
                step: &step,
                k: &k,
                gain: &gain,
                cs: &cs,
                eps: cx.meta.eps,
                n_dims: cx.meta.n_dims,
                cache: &mut cache,
            },
        )?;
        stream.synchronize()?;
        Ok(cache.buf().to_host_vec(stream)?)
    }

    /// The index key at a source layer whose step completes a group: its
    /// projection (a `gemv` site fed the pre-rope row; the dump must not be
    /// the projection of the turned row), then the kernel against our rule,
    /// ik's rule against the dump's chain, and the kernel against the dump.
    fn idx_site(
        cx: &mut Cx,
        label: &str,
        man: &RefManifest,
        src: &Source,
    ) -> Result<bool, GateError> {
        let (geom, st, layer, tag) = (src.geom, src.st, src.layer, src.tag);
        let (w, eps) = (index_key::WIDTH, cx.meta.eps);
        let groups = st.groups();
        let ch = IdxChain::read(man, layer, tag)?;
        let (_, pre_row) = node(
            man,
            &format!("{tag}_state_compress-{layer}"),
            1,
            "FUSED_RMS_NORM",
        )?;
        let (_, roped_row) = node(man, &format!("{tag}_state_compress-{layer}"), 2, "ROPE")?;
        let pre = ref_tensor_logical_in(&man.dir, pre_row)?;
        let roped = ref_tensor_logical_in(&man.dir, roped_row)?;
        let moved: Vec<bool> = st.write_pos.iter().map(|&p| p > 0).collect();
        let mm_pass = gemv_site(cx, label, man, ch.mm, &pre, Some((&roped, &moved)))?;
        cx.tally(mm_pass);

        let d_mm = ref_tensor_logical_in(&man.dir, ch.mm)?;
        let d_norm = ref_tensor_logical_in(&man.dir, ch.norm)?;
        let d_rope = ref_tensor_logical_in(&man.dir, ch.rope)?;
        let d_hada = ref_tensor_logical_in(&man.dir, ch.hada)?;
        let n = groups * w;
        if [d_mm.len(), d_norm.len(), d_rope.len(), d_hada.len()] != [n; 4] {
            return Err(format!(
                "layer {layer}: the key chain holds {} values for {groups} group(s)",
                d_mm.len()
            )
            .into());
        }
        let gain = split_f32(&cx.split, &format!("blk.{layer}.indexer.k_norm.weight"), w)?;
        let words = step_words(src)?;
        let (ours_tab, cs) = step_tables(cx, src)?;
        let mut k = vec![POISON; geom.max_groups * w];
        k[..n].copy_from_slice(&d_mm);
        let got = idx_run(cx, geom, &words, &k, &gain, &cs)?;
        let got2 = idx_run(cx, geom, &words, &k, &gain, &cs)?;
        let rerun = got == got2;

        // Layer 1: our rule on the host, bit for bit in f16.
        let written: Vec<usize> = st.state_write.iter().map(|&x| x as usize).collect();
        let mut host = Vec::with_capacity(n);
        for g in 0..groups {
            let normed = norm_ours(&d_mm[g * w..(g + 1) * w], &gain, eps)?;
            let mut v = rotate(&normed, &ours_tab[g]);
            fast_ht(&mut v);
            host.extend(v);
        }
        let host_f16: Vec<u16> = host.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let rows_exact = written
            .iter()
            .enumerate()
            .all(|(g, &r)| got[r * w..(r + 1) * w] == host_f16[g * w..(g + 1) * w]);
        let untouched = (0..geom.rows)
            .filter(|r| !written.contains(r))
            .all(|r| got[r * w..(r + 1) * w].iter().all(|&h| h == SENTINEL));
        let rule_ok = rows_exact && untouched && rerun;

        // Layer 2: ik's norm, rope, transform and write against the dump.
        let ik_tab = tables(&cx.meta, &st.write_pos, Some(w))?;
        let (mut sim_norm, mut sim_rope, mut sim_hada) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let mut scale_is = true;
        for g in 0..groups {
            sim_norm.extend(norm_ik(&d_mm[g * w..(g + 1) * w], &gain, eps));
            sim_rope.extend(rotate(&d_norm[g * w..(g + 1) * w], &ik_tab[g]));
            let mut v = d_rope[g * w..(g + 1) * w].to_vec();
            scale_is &= fast_ht(&mut v).to_bits() == HT_SCALE.to_bits();
            sim_hada.extend(v);
        }
        let norm_same = same_bits(&sim_norm, &d_norm);
        let rope_same = same_bits(&sim_rope, &d_rope);
        let hada_same = same_bits(&sim_hada, &d_hada);
        let ik_after = f16_all(man, ch.write)?;
        let ik_before = f16_all(man, ch.before)?;
        let lands = ik_after.len() == geom.rows * w
            && ik_before.len() == ik_after.len()
            && (0..geom.rows).all(|r| {
                let (a, b) = (
                    &ik_after[r * w..(r + 1) * w],
                    &ik_before[r * w..(r + 1) * w],
                );
                match written.iter().position(|&x| x == r) {
                    Some(g) => a
                        .iter()
                        .zip(&d_hada[g * w..(g + 1) * w])
                        .all(|(&h, &v)| h == f32_to_f16_bits(v)),
                    None => a == b,
                }
            });
        let sim_ok = norm_same == n && rope_same == n && hada_same == n && lands && scale_is;

        // Layer 3: our f32 key (the kernel's, by layer 1) against ik's, per
        // value within the band of the row's 2-norm; the f16 rows equal
        // wherever the f32 values are.
        let mut band = Vec::with_capacity(n);
        for g in 0..groups {
            let b = f64::from(IDX_L2_UNITS * U) * norm2(&d_hada[g * w..(g + 1) * w]);
            band.extend(std::iter::repeat_n(b, w));
        }
        let (k_over, k_worst) = over(&host, &d_hada, &band);
        let ik_rel = max_rel_err(&host, &d_hada)?;
        let ulp = max_ulps(&host, &d_hada);
        let (mut f16_same, mut on_equal) = (0usize, true);
        for (g, &r) in written.iter().enumerate() {
            for i in 0..w {
                let (a, b) = (got[r * w + i], ik_after[r * w + i]);
                f16_same += usize::from(a == b);
                on_equal &= host[g * w + i].to_bits() != d_hada[g * w + i].to_bits() || a == b;
            }
        }
        let dump_ok = k_over == 0 && on_equal;

        let pass = rule_ok && sim_ok && dump_ok;
        println!(
            "idx set={label} layer={layer} stream={tag} groups={groups}/{} rows={} pos={} write_rows={} \
             bit_exact_host={rows_exact} unwritten_rows_sentinel={untouched} bit_identical_rerun={rerun} \
             ik_sim_norm_same={norm_same}/{n} ik_sim_rope_same={rope_same}/{n} ik_sim_hadamard_same={hada_same}/{n} \
             ht_scale_is_fast_ht={scale_is} write_lands_only_on_rows={lands} ik_rel={ik_rel:.3e} \
             over={k_over}/{n} worst={k_worst:.3} max_ulp={ulp} f16_same={f16_same}/{n} \
             f16_equal_on_equal_f32={on_equal} {}",
            geom.max_groups,
            geom.rows,
            list(&st.write_pos),
            list(&written),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The index key at a step that completes no group: the same launch, a
    /// poisoned key, and no row written — the dump builds no key either.
    fn idx_idle(cx: &Cx, label: &str, src: &Source) -> Result<bool, GateError> {
        let (geom, w) = (src.geom, index_key::WIDTH);
        let words = step_words(src)?;
        let (_, cs) = step_tables(cx, src)?;
        let k = vec![POISON; geom.max_groups * w];
        let gain = vec![POISON; w];
        let got = idx_run(cx, geom, &words, &k, &gain, &cs)?;
        let untouched = got.iter().all(|&h| h == SENTINEL);
        println!(
            "idx set={label} layer={} stream={} groups=0/{} rows={} unwritten_rows_sentinel={untouched} {}",
            src.layer,
            src.tag,
            geom.max_groups,
            geom.rows,
            verdict(untouched)
        );
        Ok(untouched)
    }
}
