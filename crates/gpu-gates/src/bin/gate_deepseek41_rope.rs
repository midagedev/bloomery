//! GPU gate for DeepSeek-V4.1's rope family — B4 op blocks B and C
//! (`docs/research/v41-b4-plan-report.md` §1-B, §1-C) — against ik's CPU
//! dump. Every ROPE and ROPE_BACK row of every set below is a site of
//! `ds41_rope_tail`, which turns the row's input in place: `kv_rope-L`,
//! `q_rope-L`, the attention output's inverse `attn-L`, the pooled
//! compressed rows (`csa_state_compress-L`/`hca_state_compress-L`
//! occurrence 2), the index keys (` (reshaped) (view)`) and, in the decode
//! steps, the indexer query `indexer_q-L`. Every layer's latent K/V row is a
//! site of `ds41_kv_norm_rope_append`: the chain from the `kv_b-L`
//! projection through `kv_norm-L` and `kv_rope-L` to the row
//! `dsv4_raw_k_write-L` writes into ik's cache.
//!
//! Three layers per site (the B4 gate form):
//! 1. the kernel against this binary's transcription of our rule —
//!    bit-identical — and a rerun, bit-identical;
//! 2. ik's rule simulated here against the dump: the table from
//!    `rope::ggml_rope_cache` (ggml's recipe as ik's CPU build compiles it),
//!    the turn unfused, the values before the tail untouched; the K/V norm
//!    summed in f64 serially; the append rounded to nearest even. It proves
//!    the semantics — tail offset, base per site, positions (`inp_pos`, or
//!    the group start `dsv4_*_write_pos` for pooled rows and index keys),
//!    ROPE_BACK's sign, the view the append reads — bit for bit;
//! 3. the kernel against the dump, in the band derived from the difference
//!    between the two rules (at each constant).
//!
//! Sets: the 5-token prefill (`Set::Cpu`, its index-key chain included) and
//! the decode steps [`STEP_SETS`] — T = 1 at positions 4, 301 and 1,025,
//! where the ring slot wraps. Positions past those are the host unit test's
//! (`bloomery_gpu_deepseek41::rope`, `just gate-gpu-ds41-lib`). A node's
//! input is found through its `src0`/`src1` column: the last row of that
//! name before the reader, manifest order being execution order.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_rope: built without the `deepseek41` feature; see `just gate-gpu-ds41-rope`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_rope", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::elem::{RMS_THREADS, RMS_WARPS, rms_scale, rms_warp_tree};
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_deepseek41::rope::{
        Direction, KvAppendArgs, RopeKernels, RopeSpec, RopeTable, TailShape, ggml_rope_cache,
    };
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RefRow, RowKind, bits_equal, checks_failed, find_int_row,
        max_rel_err, ref_dir_named, ref_ints_of_in, ref_model_path, ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{GgmlType, f32_to_f16_bits, half_to_f32};
    use gguf::{Split, Value};
    use model::arch::Arch;

    /// The decode-step sets: one step after a prefill run under the dumped
    /// schedule (the `-every-node` variants of `tools/ref/models/deepseek41.sh`),
    /// the `unfused` ones with the indexer's scores and top-k as nodes of their
    /// own — the same rope sites in another graph — and `d1n` at 301 with the
    /// file's top-k, where no indexer query is turned.
    pub const STEP_SETS: [&str; 6] = [
        "ref_deepseek41_step4_every_node",
        "ref_deepseek41_d1_every_node",
        "ref_deepseek41_d1_unfused_every_node",
        "ref_deepseek41_d1n_every_node",
        "ref_deepseek41_d2_every_node",
        "ref_deepseek41_d2_unfused_every_node",
    ];

    /// f32's unit roundoff, 2^-24.
    const U: f32 = f32::EPSILON / 2.0;

    /// Band for the K/V row against ik's `kv_rope-L`, as `max|Δ| / M` with
    /// `M` the largest `|value|` ik wrote. The two rules differ in the norm's
    /// sum alone: ours is the f32 tree (per thread one rounded square and one
    /// fused multiply-add, five butterfly levels, three tree levels — at most
    /// 10 roundings on a sum of non-negative terms; the division by the
    /// width, a power of two, is exact), ik's the f64 sum of f32 squares
    /// rounded once to f32 (2 roundings): the means differ by at most 12u of
    /// themselves. `+ eps` adds the same positive value and rounds each side
    /// (14u); the square root halves that and rounds each (9u); the
    /// reciprocal (11u), `· gain` (13u) and `· x` (15u) round each side once
    /// more, so a value before the tail differs by at most 15u of itself.
    /// A tail pair then turns: each product rounds on both sides (17u of
    /// `|y0·c|`, `|y1·s|`) and each sum once on both (2u of the result), and
    /// `|y0·c| + |y1·s| <= |(y0, y1)|` because `c² + s² = 1` — the norm of
    /// the turned pair, at most `√2·M`. A value moves by at most
    /// `(17·√2 + 2)·u < 27u` of `M`.
    const KV_BAND: f32 = 27.0 * U;

    /// Band for a tail rope against the dump, in ulps per value. Our table
    /// is ik's recipe on the same libm and both turns are unfused, so the
    /// derived distance is 0; the allowance is the one ulp a trig or `powf`
    /// from another libm would move — layer 2 prints whether ours does.
    const ROPE_ULP_BAND: u32 = 1;

    /// What the gate fills the ring with before an append: an f16 NaN, which
    /// `f32_to_f16_bits` never writes (it rounds a NaN to infinity), so a
    /// slot still holding it was not written.
    const SENTINEL: u16 = 0xffff;

    /// The file's rope and norm constants, from its metadata.
    struct Meta {
        n_dims: usize,
        window: RopeSpec,
        yarn: RopeSpec,
        /// `attention.compress_ratios`, one per layer.
        ratios: Vec<u64>,
        eps: f32,
        /// Rows of a layer's raw window ring: `attention.sliding_window`.
        ring: usize,
    }

    impl Meta {
        fn read(split: &Split) -> Result<Meta, GateError> {
            let want = Arch::Deepseek41.name();
            if split.architecture() != Some(want) {
                return Err(format!(
                    "the model file is {:?}, want {want} — run through `just gate-gpu-ds41-rope`, \
                     which picks the deepseek41 profile",
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
            let n_dims = usize::try_from(u("rope.dimension_count")?)?;
            let n_ctx_orig = i32::try_from(u("rope.scaling.original_context_length")?)?;
            let key = split.arch_key("attention.compress_ratios");
            let Some(Value::Array(list)) = split.value(&key) else {
                return Err(format!("metadata {key} is absent or not an array").into());
            };
            let ratios = list
                .iter()
                .map(|v| v.as_unsigned().ok_or_else(|| format!("{key} holds {v:?}")))
                .collect::<Result<Vec<u64>, String>>()?;
            Ok(Meta {
                n_dims,
                window: RopeSpec::window(f("rope.freq_base")?, n_dims),
                yarn: RopeSpec::yarn(
                    f("attention.compress_rope_freq_base")?,
                    f("rope.scaling.factor")?,
                    n_ctx_orig,
                    f("rope.scaling.yarn_beta_fast")?,
                    f("rope.scaling.yarn_beta_slow")?,
                    n_dims,
                ),
                ratios,
                eps: f("attention.layer_norm_rms_epsilon")?,
                ring: usize::try_from(u("attention.sliding_window")?)?,
            })
        }

        /// The rope of layer `l`'s heads: ik's `use_compress_rope` is
        /// `compress_ratios[l] != 0`.
        fn head_spec(&self, l: usize) -> Result<(&RopeSpec, &'static str), GateError> {
            match self.ratios.get(l) {
                Some(0) => Ok((&self.window, "window")),
                Some(_) => Ok((&self.yarn, "yarn")),
                None => Err(format!("layer {l} has no compress ratio").into()),
            }
        }
    }

    /// What a rope row is, from its op and name. A head site carries its
    /// layer, which picks its base; the others are YaRN at every layer.
    #[derive(Clone, Copy, Debug)]
    enum Site {
        Kv(usize),
        Q(usize),
        Back(usize),
        Pooled,
        IndexKey,
        IndexerQ,
    }

    impl Site {
        /// `None` for a row that is no rope; an error for a rope row this
        /// gate does not know — a new site is classified, never skipped.
        fn of(row: &RefRow) -> Result<Option<Site>, GateError> {
            let layer = |p: &str| {
                row.name
                    .strip_prefix(p)
                    .and_then(|l| l.parse::<usize>().ok())
            };
            let site = match row.op.as_str() {
                "ROPE_BACK" => layer("attn-").map(Site::Back),
                "ROPE" => layer("kv_rope-")
                    .map(Site::Kv)
                    .or_else(|| layer("q_rope-").map(Site::Q))
                    .or_else(|| {
                        layer("csa_state_compress-")
                            .or_else(|| layer("hca_state_compress-"))
                            .map(|_| Site::Pooled)
                    })
                    .or_else(|| layer("indexer_q-").map(|_| Site::IndexerQ))
                    .or_else(|| (row.name == " (reshaped) (view)").then_some(Site::IndexKey)),
                _ => return Ok(None),
            };
            site.map(Some).ok_or_else(|| {
                format!(
                    "{} row {:?}/{} is no rope site this gate knows",
                    row.op, row.name, row.occurrence
                )
                .into()
            })
        }

        fn kind(self) -> &'static str {
            match self {
                Site::Kv(_) => "kv",
                Site::Q(_) => "q",
                Site::Back(_) => "back",
                Site::Pooled => "pooled",
                Site::IndexKey => "index_key",
                Site::IndexerQ => "indexer_q",
            }
        }

        fn dir(self) -> Direction {
            match self {
                Site::Back(_) => Direction::Back,
                _ => Direction::Forward,
            }
        }

        /// The site's rope: the layer's for the heads, YaRN for the pooled
        /// rows, the index keys and the indexer query (`build_deepseek4.cpp`).
        fn spec(self, meta: &Meta) -> Result<(&RopeSpec, &'static str), GateError> {
            match self {
                Site::Kv(l) | Site::Q(l) | Site::Back(l) => meta.head_spec(l),
                Site::Pooled | Site::IndexKey | Site::IndexerQ => Ok((&meta.yarn, "yarn")),
            }
        }
    }

    /// What every site reads besides its set.
    struct Cx<'a> {
        k: &'a RopeKernels,
        stream: &'a CudaStream,
        meta: &'a Meta,
        split: &'a Split,
    }

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        let meta = Meta::read(&split)?;
        let gpu = Gpu::new()?;
        let k = RopeKernels::load(gpu.context())?;
        let cx = Cx {
            k: &k,
            stream: gpu.stream(),
            meta: &meta,
            split: &split,
        };
        println!(
            "gate_deepseek41_rope: device {} — n_dims {} ring {} eps {:e}; window rope {:?}; yarn rope {:?}",
            gpu.device_name()?,
            meta.n_dims,
            meta.ring,
            meta.eps,
            meta.window,
            meta.yarn
        );

        let cpu = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = vec![(cpu.set_name(Set::Cpu)?, cpu.open(Set::Cpu)?)];
        for name in STEP_SETS {
            sets.push((name, open_set(name)?));
        }

        let (mut ropes, mut kvs, mut failed) = (0u32, 0u32, 0u32);
        for (label, man) in &sets {
            let (r0, k0) = (ropes, kvs);
            for (at, row) in man.tensors.iter().enumerate() {
                if let Some(site) = Site::of(row)? {
                    ropes += 1;
                    failed += u32::from(!rope_site(&cx, label, man, at, site)?);
                } else if row.op == "SET_ROWS" && row.name.starts_with("dsv4_raw_k_write-") {
                    kvs += 1;
                    failed += u32::from(!kv_site(&cx, label, man, at)?);
                }
            }
            println!(
                "set {label}: {} (build {}) — {} rope sites, {} K/V rows",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                ropes - r0,
                kvs - k0
            );
        }
        let pass = failed == 0;
        println!(
            "gate_deepseek41_rope: {ropes} rope sites and {kvs} K/V rows across {} sets, {failed} failed — {}",
            sets.len(),
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
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

    /// The node a reader at manifest index `at` reads as `name`: the last
    /// tensor row of that name before it, with its index.
    fn src_row<'a>(
        man: &'a RefManifest,
        at: usize,
        name: Option<&str>,
    ) -> Result<(usize, &'a RefRow), GateError> {
        let name = name.ok_or("a reader row has no src column")?;
        man.tensors[..at]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, r)| r.name == name)
            .ok_or_else(|| format!("no node {name:?} before manifest row {at}").into())
    }

    /// A graph input's integers — positions, cache rows — from its exact
    /// twin, as the u32 the kernels take.
    fn input_u32(man: &RefManifest, name: Option<&str>) -> Result<Vec<u32>, GateError> {
        let name = name.ok_or("a reader row has no src1 column")?;
        let row = find_int_row(man, name, 0, RowKind::Input, Layout::Flat)?;
        ref_ints_of_in(&man.dir, row)?
            .into_iter()
            .map(|v| u32::try_from(v).map_err(|_| format!("{name} holds {v}").into()))
            .collect()
    }

    /// Rows `idx` of an f16 cache row as f16 bits, row by row: the dump
    /// widened each half to f32 exactly, so rounding back recovers ik's
    /// bits, and a value that does not come back is an error.
    fn f16_rows(man: &RefManifest, row: &RefRow, idx: &[u32]) -> Result<Vec<u16>, GateError> {
        let width = row.ne[0] as usize;
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
            let n = raw.len();
            return Err(format!(
                "{} is {n} bytes, the row says {}",
                path.display(),
                row.bytes
            )
            .into());
        }
        let vals = raw.as_chunks::<4>().0;
        let mut out = Vec::with_capacity(idx.len() * width);
        for &r in idx {
            let r = r as usize;
            let cells = vals.get(r * width..(r + 1) * width).ok_or_else(|| {
                format!(
                    "{}: row {r} is past its {} values",
                    path.display(),
                    vals.len()
                )
            })?;
            for c in cells {
                let v = f32::from_le_bytes(*c);
                let h = f32_to_f16_bits(v);
                if half_to_f32(h).to_bits() != v.to_bits() {
                    return Err(format!("{} holds {v}, not a widened f16", path.display()).into());
                }
                out.push(h);
            }
        }
        Ok(out)
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

    /// The engine's tables: `m` of `n_dims`, token `t`'s at `t·n_dims`.
    fn tables(table: &RopeTable, pos: &[u32], dir: Direction) -> Vec<f32> {
        let mut cs = Vec::with_capacity(pos.len() * table.n_dims());
        for &p in pos {
            table.push(p, dir, &mut cs);
        }
        cs
    }

    /// ik's tables: ggml's recipe filling a `ne0`-value head's cache, of
    /// which the tail rope reads the first `n_dims`.
    fn ik_tables(spec: &RopeSpec, pos: &[u32], ne0: usize, dir: Direction) -> Vec<f32> {
        pos.iter()
            .flat_map(|&p| {
                ggml_rope_cache(spec, p, ne0, dir)
                    .into_iter()
                    .take(spec.n_dims)
            })
            .collect()
    }

    /// The tail rope on the host, heads of `width` with `n_vec` per token:
    /// each pair turned with every product and sum rounded on its own — the
    /// kernel's `rope_pair_rn` (host f32 `*`, `-`, `+` are exactly
    /// `mul.rn`/`add.rn`) and ik's unfused rotation alike.
    fn rotate(x: &[f32], cs: &[f32], width: usize, nd: usize, n_vec: usize) -> Vec<f32> {
        let mut y = x.to_vec();
        for (r, head) in y.chunks_mut(width).enumerate() {
            let t = r / n_vec;
            let tail = head[width - nd..].as_chunks_mut::<2>().0;
            let tab = cs[t * nd..(t + 1) * nd].as_chunks::<2>().0;
            for (p, &[c, s]) in tail.iter_mut().zip(tab) {
                let [x0, x1] = *p;
                *p = [x0 * c - x1 * s, x0 * s + x1 * c];
            }
        }
        y
    }

    /// Ulps between two finite f32, through integers ordered like the
    /// floats they encode (`-0.0` and `+0.0` the same point).
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

    fn list(v: &[impl ToString]) -> String {
        v.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    }

    /// One tail-rope site: the row at manifest index `at` of `man`.
    fn rope_site(
        cx: &Cx,
        label: &str,
        man: &RefManifest,
        at: usize,
        site: Site,
    ) -> Result<bool, GateError> {
        let row = &man.tensors[at];
        let [width, n_vec, m, one] = row.ne.map(|n| n as usize);
        let nd = cx.meta.n_dims;
        if one != 1 || width < nd {
            return Err(format!(
                "{}/{} is {:?}, want [width >= {nd}, heads, tokens, 1]",
                row.name, row.occurrence, row.ne
            )
            .into());
        }
        let (_, src) = src_row(man, at, row.src0.as_deref())?;
        if src.count() != row.count() {
            return Err(format!(
                "{} reads {} of {} values, it has {}",
                row.name,
                src.name,
                src.count(),
                row.count()
            )
            .into());
        }
        let x = ref_tensor_logical_in(&man.dir, src)?;
        let want = ref_tensor_logical_in(&man.dir, row)?;
        let pos = input_u32(man, row.src1.as_deref())?;
        if pos.len() != m {
            return Err(format!("{}: {} positions for {m} tokens", row.name, pos.len()).into());
        }
        let (spec, base) = site.spec(cx.meta)?;
        let dir = site.dir();

        // Layer 1: the kernel against our rule on the host, and a rerun.
        let cs = tables(&RopeTable::new(spec)?, &pos, dir);
        let host = rotate(&x, &cs, width, nd, n_vec);
        let cs_dev = DeviceBuffer::from_host(cx.stream, &cs)?;
        let shape = TailShape {
            width,
            n_dims: nd,
            n_vec,
            m,
        };
        let mut runs = Vec::with_capacity(2);
        for _ in 0..2 {
            let mut dev = DeviceBuffer::from_host(cx.stream, &x)?;
            cx.k.enqueue_rope_tail(cx.stream, &mut dev, &cs_dev, shape)?;
            cx.stream.synchronize()?;
            runs.push(dev.to_host_vec(cx.stream)?);
        }
        let y = &runs[0];
        let exact = bits_equal(y, &host);
        let rerun = bits_equal(y, &runs[1]);

        // Layer 2: ik's rule against the dump — its tables, and the head it
        // leaves alone.
        let ik_cs = ik_tables(spec, &pos, width, dir);
        let table_is_recipe = bits_equal(&cs, &ik_cs);
        let sim = rotate(&x, &ik_cs, width, nd, n_vec);
        let sim_same = same_bits(&sim, &want);
        let sim_ulps = max_ulps(&sim, &want);
        let head_kept = x
            .chunks(width)
            .zip(want.chunks(width))
            .all(|(a, b)| bits_equal(&a[..width - nd], &b[..width - nd]));

        // Layer 3: the kernel against the dump.
        let same = same_bits(y, &want);
        let ulp = max_ulps(y, &want);
        let ik_rel = max_rel_err(y, &want)?;

        let pass = exact
            && rerun
            && table_is_recipe
            && sim_ulps <= ROPE_ULP_BAND
            && head_kept
            && ulp <= ROPE_ULP_BAND;
        let name = match site {
            Site::IndexKey => format!(
                "{}/{}->{}",
                row.name.trim().replace(' ', "_"),
                row.occurrence,
                consumer(man, at)
            ),
            _ => format!("{}/{}", row.name, row.occurrence),
        };
        let n = want.len();
        println!(
            "rope set={label} site={name} kind={} base={base} dir={dir:?} width={width} heads={n_vec} m={m} \
             pos={} table_is_recipe={table_is_recipe} bit_exact_host={exact} bit_identical_rerun={rerun} \
             ik_sim_same={sim_same}/{n} ik_sim_max_ulp={sim_ulps} head_kept={head_kept} same={same}/{n} \
             max_ulp={ulp} (band {ROPE_ULP_BAND}) ik_rel={ik_rel:.3e} {}",
            site.kind(),
            list(&pos),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The first node after `at` that reads row `at` — for an index key, the
    /// Hadamard `lid_k_new-L` that names its layer.
    fn consumer(man: &RefManifest, at: usize) -> String {
        let name = &man.tensors[at].name;
        man.tensors[at + 1..]
            .iter()
            .find(|r| r.src0.as_deref() == Some(name.as_str()))
            .map_or_else(|| "-".to_string(), |r| r.name.clone())
    }

    /// Our norm of one row on the host, as the device compiles
    /// `elem::rms_norm`'s body: per thread `acc = fma(v, v, acc)` over values
    /// `tid, tid + RMS_THREADS, …` (cuda-oxide contracts `acc += v * v`; the
    /// core itself run on the host would round twice), the warp butterfly
    /// (xor 16, 8, 4, 2, 1, `own + partner`), `rms_warp_tree`, `rms_scale`,
    /// then `(scale · gain) · x`.
    fn norm_ours(x: &[f32], gain: &[f32], eps: f32) -> Result<Vec<f32>, GateError> {
        let mut part = [0.0f32; RMS_THREADS];
        for (tid, p) in part.iter_mut().enumerate() {
            *p = x
                .iter()
                .skip(tid)
                .step_by(RMS_THREADS)
                .fold(0.0f32, |acc, &v| v.mul_add(v, acc));
        }
        let mut warps = [0.0f32; RMS_WARPS];
        for (w, lanes) in warps.iter_mut().zip(part.as_chunks::<32>().0) {
            let mut v = *lanes;
            for off in [16, 8, 4, 2, 1] {
                let prev = v;
                for (l, s) in v.iter_mut().enumerate() {
                    *s = prev[l] + prev[l ^ off];
                }
            }
            *w = v[0];
        }
        let k = u32::try_from(x.len())?;
        let scale = rms_scale(rms_warp_tree(warps), k, eps);
        Ok(x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect())
    }

    /// ik's `FUSED_RMS_NORM` of one row (`ggml_compute_forward_fused_rms_norm_f32`):
    /// f32 squares summed serially in f64, the mean rounded to f32, then
    /// `1/sqrtf(mean + eps)` and `(scale · c) · x`.
    fn norm_ik(x: &[f32], gain: &[f32], eps: f32) -> Vec<f32> {
        let sum = x.iter().fold(0.0f64, |a, &v| a + f64::from(v * v));
        let mean = (sum / x.len() as f64) as f32;
        let scale = 1.0 / (mean + eps).sqrt();
        x.iter().zip(gain).map(|(&v, &g)| (scale * g) * v).collect()
    }

    /// One layer's K/V chain in a set, read back from the manifest from the
    /// cache write at index `at`: `dsv4_raw_k_write-L` ← `kv_rope-L (view)`
    /// ← `kv_rope-L` (ROPE) ← `kv_norm-L (reshaped)` ← `kv_norm-L`
    /// (FUSED_RMS_NORM, gain in `src1`) ← its input, the `kv_b-L` projection.
    struct KvChain<'a> {
        layer: usize,
        width: usize,
        m: usize,
        write: &'a RefRow,
        view: &'a RefRow,
        rope: &'a RefRow,
        norm: &'a RefRow,
        x: &'a RefRow,
    }

    impl<'a> KvChain<'a> {
        fn read(man: &'a RefManifest, at: usize) -> Result<KvChain<'a>, GateError> {
            let write = &man.tensors[at];
            let (v_at, view) = src_row(man, at, write.src0.as_deref())?;
            let (r_at, rope) = src_row(man, v_at, view.src0.as_deref())?;
            let (s_at, shaped) = src_row(man, r_at, rope.src0.as_deref())?;
            let (n_at, norm) = src_row(man, s_at, shaped.src0.as_deref())?;
            let (_, x) = src_row(man, n_at, norm.src0.as_deref())?;
            let layer = rope
                .name
                .strip_prefix("kv_rope-")
                .and_then(|l| l.parse().ok())
                .ok_or_else(|| {
                    format!("{} writes {}, not a kv_rope-L view", write.name, view.name)
                })?;
            if rope.op != "ROPE" || norm.op != "FUSED_RMS_NORM" {
                return Err(format!(
                    "{}: the chain is {} <- {}, want ROPE <- FUSED_RMS_NORM",
                    write.name, rope.op, norm.op
                )
                .into());
            }
            let (width, m) = (rope.ne[0] as usize, rope.ne[2] as usize);
            if rope.ne[1] != 1
                || rope.ne[3] != 1
                || x.count() != rope.count()
                || view.count() != rope.count()
                || write.ne[0] != rope.ne[0]
            {
                return Err(format!(
                    "{}: kv_rope {:?}, its view {:?}, the norm's input {:?}, the cache {:?} — \
                     want one {width}-value head per token",
                    write.name, rope.ne, view.ne, x.ne, write.ne
                )
                .into());
            }
            Ok(KvChain {
                layer,
                width,
                m,
                write,
                view,
                rope,
                norm,
                x,
            })
        }
    }

    /// One layer's latent K/V row: the chain ending at manifest index `at`.
    fn kv_site(cx: &Cx, label: &str, man: &RefManifest, at: usize) -> Result<bool, GateError> {
        let ch = KvChain::read(man, at)?;
        let (width, m, nd, window) = (ch.width, ch.m, cx.meta.n_dims, cx.meta.ring);
        let gain_name = ch
            .norm
            .src1
            .as_deref()
            .ok_or("the norm has no gain column")?;
        let gain = split_f32(cx.split, gain_name, width)?;
        let x = ref_tensor_logical_in(&man.dir, ch.x)?;
        let norm_dump = ref_tensor_logical_in(&man.dir, ch.norm)?;
        let rope_dump = ref_tensor_logical_in(&man.dir, ch.rope)?;
        let view_dump = ref_tensor_logical_in(&man.dir, ch.view)?;
        let pos = input_u32(man, ch.rope.src1.as_deref())?;
        let idxs = input_u32(man, ch.write.src1.as_deref())?;
        let ik_rows = f16_rows(man, ch.write, &idxs)?;
        if pos.len() != m || idxs.len() != m || m > window {
            return Err(format!(
                "{}: {} positions, {} cache rows for {m} tokens, ring {window}",
                ch.write.name,
                pos.len(),
                idxs.len()
            )
            .into());
        }
        let (spec, base) = cx.meta.head_spec(ch.layer)?;
        let cs = tables(&RopeTable::new(spec)?, &pos, Direction::Forward);
        let slots: Vec<usize> = pos.iter().map(|&p| p as usize % window).collect();

        // Layer 1: the kernel against our rule on the host — the f32 rows,
        // every ring slot (unwritten ones keep the sentinel) — and a rerun.
        let mut host = Vec::with_capacity(m * width);
        for row in x.chunks(width) {
            host.extend(norm_ours(row, &gain, cx.meta.eps)?);
        }
        let host = rotate(&host, &cs, width, nd, 1);
        let host_f16: Vec<u16> = host.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let dev = KvInputs {
            kv: DeviceBuffer::from_host(cx.stream, &x)?,
            gain: DeviceBuffer::from_host(cx.stream, &gain)?,
            cs: DeviceBuffer::from_host(cx.stream, &cs)?,
            pos: DeviceBuffer::from_host(cx.stream, &pos)?,
        };
        let (out, ring) = kv_run(cx, &dev, m, width, window)?;
        let (out2, ring2) = kv_run(cx, &dev, m, width, window)?;
        let exact = bits_equal(&out, &host);
        let ring_exact = (0..window).all(|s| {
            let got = &ring[s * width..(s + 1) * width];
            match slots.iter().position(|&x| x == s) {
                Some(t) => got == &host_f16[t * width..(t + 1) * width],
                None => got.iter().all(|&h| h == SENTINEL),
            }
        });
        let rerun = bits_equal(&out, &out2) && ring == ring2;

        // Layer 2: ik's rule against the dump — the f64 norm; the append
        // reads the rope's output and rounds it to nearest even. Its rope is
        // the kv_rope-L rope site.
        let mut sim_norm = Vec::with_capacity(m * width);
        for row in x.chunks(width) {
            sim_norm.extend(norm_ik(row, &gain, cx.meta.eps));
        }
        let n = m * width;
        let norm_same = same_bits(&sim_norm, &norm_dump);
        let view_is_rope = bits_equal(&view_dump, &rope_dump);
        let sim_rows: Vec<u16> = view_dump.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let append_same = sim_rows
            .iter()
            .zip(&ik_rows)
            .filter(|(a, b)| a == b)
            .count();

        // Layer 3: the kernel against the dump — the f32 rows in the norm's
        // band, and our ring row of each position equal to ik's cache row of
        // that position wherever the two f32 rows agree.
        let ik_rel = max_rel_err(&out, &rope_dump)?;
        let f32_same = same_bits(&out, &rope_dump);
        let our_rows: Vec<u16> = slots
            .iter()
            .flat_map(|&s| ring[s * width..(s + 1) * width].iter().copied())
            .collect();
        let f16_same = our_rows
            .iter()
            .zip(&ik_rows)
            .filter(|(a, b)| a == b)
            .count();
        let append_on_equal = out
            .iter()
            .zip(&rope_dump)
            .zip(our_rows.iter().zip(&ik_rows))
            .all(|((a, b), (c, d))| a.to_bits() != b.to_bits() || c == d);

        let pass = exact
            && ring_exact
            && rerun
            && norm_same == n
            && view_is_rope
            && append_same == n
            && ik_rel <= KV_BAND
            && append_on_equal;
        println!(
            "kv set={label} layer={} base={base} width={width} m={m} pos={} slots={} ik_rows={} \
             bit_exact_host={exact} ring_exact={ring_exact} bit_identical_rerun={rerun} \
             ik_sim_norm_same={norm_same}/{n} view_is_rope={view_is_rope} ik_sim_append_same={append_same}/{n} \
             ik_rel={ik_rel:.3e} (band {KV_BAND:.3e}) f32_same={f32_same}/{n} f16_same={f16_same}/{n} \
             append_on_equal_inputs={append_on_equal} {}",
            ch.layer,
            list(&pos),
            list(&slots),
            list(&idxs),
            verdict(pass)
        );
        Ok(pass)
    }

    /// One K/V site's device inputs, uploaded once for both runs.
    struct KvInputs {
        kv: DeviceBuffer<f32>,
        gain: DeviceBuffer<f32>,
        cs: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
    }

    /// One launch of the K/V kernel into a fresh ring of sentinels: the f32
    /// rows and the whole ring, read back.
    fn kv_run(
        cx: &Cx,
        dev: &KvInputs,
        m: usize,
        width: usize,
        window: usize,
    ) -> Result<(Vec<f32>, Vec<u16>), GateError> {
        let mut out = DeviceBuffer::<f32>::zeroed(cx.stream, m * width)?;
        let mut cache =
            DeviceTensor::upload(cx.stream, &vec![SENTINEL; window * width], window, width)?;
        cx.k.enqueue_kv_norm_rope_append(
            cx.stream,
            KvAppendArgs {
                kv: &dev.kv,
                gain: &dev.gain,
                cs: &dev.cs,
                pos: &dev.pos,
                eps: cx.meta.eps,
                n_dims: cx.meta.n_dims,
                m,
                out: &mut out,
                cache: &mut cache,
            },
        )?;
        cx.stream.synchronize()?;
        Ok((
            out.to_host_vec(cx.stream)?,
            cache.buf().to_host_vec(cx.stream)?,
        ))
    }
}
