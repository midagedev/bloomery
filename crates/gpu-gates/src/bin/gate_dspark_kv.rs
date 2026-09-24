//! GPU gate for the DSpark draft's load and its feature-to-KV graph
//! (`bloomery_gpu_deepseek41::draft::{load, kv}`), on the 3090.
//!
//! 1. **Load.** The draft file (`$BLOOMERY_DSPARK_MODEL`) and the target's
//!    borrowed tensors (`$BLOOMERY_REF_MODEL`) go on the card, each tensor in
//!    `dspark::card_format`'s format. Every group's buffer bytes equal the
//!    inventory's card bytes, the borrowed head's equal its Q6_K file bytes;
//!    the `cuMemGetInfo` drop across the load is printed next to the sum and
//!    next to the allocator model's figure.
//! 2. **Feature-to-KV against the dsref set** (`code64_n32_w3`, blocks 0–3:
//!    block 0's graph is the prompt's 64 positions, the others append each
//!    round's committed rows). Fed the set's own inputs — the feature rows,
//!    each row's rope position (`pos_ctx`) and ring row (`rows`) — our
//!    `main_x` must sit within the bound ik's q8_1 activation rule allows
//!    around `dflash_kv_fused_target`, and each layer's appended rows within
//!    the bound around ik's f16 ring rows, both when the layers read ik's
//!    `main_x` and when they read ours (the engine's chain, whose bound
//!    carries the measured `main_x` gap through `attn_kv`). The bounds are
//!    derived per value from the inputs (the band section below).
//! 3. **The rings.** Every row a feature-to-KV graph has written so far equals
//!    in our ring, bit for bit, what the append left there; ik's rows are
//!    compared as above in the ring the block pass reads (`graph block`) and in
//!    the one the next append reads (`graph kv`), and ik's own two views are
//!    compared with each other (the set's consistency).
//!
//! Band. ik's CUDA matmul of a Q8_0 weight quantizes its f32 activation to
//! q8_1 (32-value blocks, `d = amax/127`), ours keeps it f32; so a row's gap
//! is at most `Σ_b (d_b/2)(1 + 256u)·Σ_{i∈b}|w_i|` plus both sides' f32
//! accumulation, `2γ_K·Σ_b amax_b·Σ_{i∈b}|w_i|`, plus `Σ_j |w_j|·e_j` for an
//! input already off by `e`. An RMS norm carries a row's bound `E` to
//! `|g_i|·s·(E_i + |y_i|·Σ_j|y_j|E_j / (Σ_j y_j² + K·eps))` (first order) plus
//! `2(γ_K + 4u)|z_i|`; the tail rope adds the pair partner's bound and
//! `(|x_0| + |x_1|)·(8u·pos + 8u)` (both tables' angle rounding); ik stores f16,
//! so its value is at most half an f16 ulp from its f32.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_dspark_kv: built without the `deepseek41` feature; see `just gate-gpu-dspark-kv`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_dspark_kv", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use bloomery_gpu::Gpu;
    use bloomery_gpu_deepseek41::draft::kv::{DraftRings, KvAppend, KvReadback};
    use bloomery_gpu_deepseek41::draft::load::DraftWeights;
    use bloomery_gpu_gates::{
        GateError, checks_failed, data_dir, dump_stem, ref_model_path, verdict,
    };
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, f32_to_f16_bits, half_to_f32};
    use model::arch::dspark::{self, Borrow, DraftHparams, Group, names};

    const NAME: &str = "gate_dspark_kv";
    const DSREF_SET: &str = "code64_n32_w3";
    /// The ik tree every V4.1 oracle set must name in its `# build` line.
    const IK_BUILD: &str = "db517b69";
    /// Blocks whose feature-to-KV graph is checked.
    const BLOCKS: [i32; 4] = [0, 1, 2, 3];
    const U: f64 = 1.0 / (1u64 << 24) as f64;

    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    // ------------------------------------------------------------- the set

    struct Row {
        kind: String,
        name: String,
        occ: u32,
        ne: [usize; 4],
        block: i32,
        graph: String,
    }

    struct Set {
        dir: PathBuf,
        rows: Vec<Row>,
    }

    fn read_set(dir: &Path) -> Result<Set, GateError> {
        let text = std::fs::read_to_string(dir.join("MANIFEST.tsv"))
            .map_err(|e| format!("{}/MANIFEST.tsv: {e}", dir.display()))?;
        let build = text
            .lines()
            .find_map(|l| l.strip_prefix("# build\t"))
            .unwrap_or("");
        if !build.contains(IK_BUILD) {
            return Err(format!(
                "{}: # build {build:?} does not name {IK_BUILD} — stale set",
                dir.display()
            )
            .into());
        }
        let mut rows = Vec::new();
        for l in text.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = l.split('\t').collect();
            if matches!(f[0], "tensor" | "input") && f.len() == 19 {
                let u = |i: usize| f[i].parse::<usize>().unwrap_or(0);
                rows.push(Row {
                    kind: f[0].to_string(),
                    name: f[1].to_string(),
                    occ: f[2].parse()?,
                    ne: [u(4), u(5), u(6), u(7)],
                    block: f[15].parse()?,
                    graph: f[18].to_string(),
                });
            }
        }
        Ok(Set {
            dir: dir.to_path_buf(),
            rows,
        })
    }

    impl Set {
        fn find(&self, block: i32, graph: &str, name: &str) -> Result<&Row, GateError> {
            self.rows
                .iter()
                .find(|r| r.block == block && r.graph == graph && r.name == name && r.occ == 0)
                .ok_or_else(|| {
                    format!("the set has no {graph} row {name:?} in block {block}").into()
                })
        }

        fn path(&self, r: &Row, ext: &str) -> PathBuf {
            let kv = if r.graph == "kv" { "kv." } else { "" };
            let input = if r.kind == "input" { ".input" } else { "" };
            self.dir.join(format!(
                "b{}.{kv}{}.{}{input}.{ext}",
                r.block,
                dump_stem(&r.name),
                r.occ
            ))
        }

        fn f32s(&self, r: &Row) -> Result<Vec<f32>, GateError> {
            let p = self.path(r, "f32");
            let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            let v: Vec<f32> = b
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let want: usize = r.ne.iter().product();
            if v.len() != want {
                return Err(format!(
                    "{}: {} values, manifest ne {:?}",
                    p.display(),
                    v.len(),
                    r.ne
                )
                .into());
            }
            Ok(v)
        }

        fn i32s(&self, r: &Row) -> Result<Vec<i32>, GateError> {
            let p = self.path(r, "i32");
            let b = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            Ok(b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect())
        }
    }

    // ------------------------------------------------------------ the load

    /// placement.rs's allocator model: an allocation of a granule or more
    /// takes whole granules; a smaller one its size rounded up to 512 from
    /// the first shared granule with room, else a new one.
    fn heap_bytes(buffers: &[u64], granule: u64) -> u64 {
        let (mut taken, mut left) = (0u64, Vec::<u64>::new());
        for &b in buffers {
            if b >= granule {
                taken += b.div_ceil(granule) * granule;
                continue;
            }
            let room = b.next_multiple_of(512).min(granule);
            match left.iter_mut().find(|l| **l >= room) {
                Some(l) => *l -= room,
                None => {
                    left.push(granule - room);
                    taken += granule;
                }
            }
        }
        taken
    }

    fn load(
        gpu: &Gpu,
        draft: &Split,
        target: &Split,
        ok: &mut bool,
    ) -> Result<DraftWeights, GateError> {
        let hp = DraftHparams::read(draft)?;
        let inv = dspark::inventory(draft, &hp, target)?;
        let groups = [
            Group::Attention,
            Group::HyperConnection,
            Group::Router,
            Group::SharedExpert,
            Group::RoutedExperts,
            Group::Fc,
            Group::Markov,
            Group::Confidence,
            Group::OutputNorm,
        ];
        let mut predicted: BTreeMap<String, u64> = BTreeMap::new();
        for g in groups {
            predicted.insert(
                format!("{g:?}"),
                inv.card_bytes(g).ok_or("a group has no card bytes")?,
            );
        }
        for b in &inv.borrowed {
            let f = b
                .card_format()
                .ok_or("a borrowed tensor has no card format")?;
            let rows = match b.borrow {
                Borrow::Copied => b.dims[1],
                Borrow::RowSource => 1,
            };
            let bytes = f
                .resident_bytes(b.ty, b.dims[0], rows)
                .ok_or("no resident size")?;
            predicted.insert(format!("target {}", b.name), bytes);
        }
        let total_pred: u64 = predicted.values().sum();
        for (k, v) in &predicted {
            println!("{NAME}: predict | {k} | {v} B");
        }
        println!(
            "{NAME}: predict | total | {total_pred} B (inventory card bytes; the mask row is the one resident row source)"
        );

        let (free0, _) = gpu.mem_info()?;
        let w = DraftWeights::load(gpu.stream(), draft, target)?;
        gpu.stream().synchronize()?;
        let (free1, total) = gpu.mem_info()?;

        let mut got: BTreeMap<String, u64> = BTreeMap::new();
        let mut buffers = Vec::new();
        println!("{NAME}: table | tensor | group | card format | buffers B");
        for t in w.table() {
            let key = match t.group {
                Some(g) => format!("{g:?}"),
                None => format!("target {}", t.name),
            };
            *got.entry(key.clone()).or_default() += t.buffers.iter().sum::<u64>();
            buffers.extend_from_slice(&t.buffers);
            println!(
                "{NAME}: table | {} | {key} | {} | {:?}",
                t.name, t.format, t.buffers
            );
        }
        let same = got == predicted;
        println!(
            "{NAME}: load | per-group buffer bytes equal the prediction: {same} {}",
            verdict(same)
        );
        if !same {
            for (k, v) in &got {
                println!(
                    "{NAME}: load | {k} | loaded {v} predicted {:?}",
                    predicted.get(k)
                );
            }
        }
        *ok &= same;
        let sum: u64 = buffers.iter().sum();
        let granule = gpu.allocation_granularity()? as u64;
        let model = heap_bytes(&buffers, granule);
        let delta = (free0 - free1) as u64;
        println!(
            "{NAME}: load | {} buffers, sum {sum} B; allocator model (granule {granule}) {model} B; \
             cuMemGetInfo free {free0} -> {free1} of {total}: delta {delta} B ({:+} vs sum, {:+} vs model) — measured, not pinned",
            buffers.len(),
            delta as i64 - sum as i64,
            delta as i64 - model as i64
        );
        Ok(w)
    }

    // ------------------------------------------------------------ the bands

    /// A Q8_0 weight on the host: dequantized rows and, per row and 32-value
    /// block, `Σ|w|`.
    struct Q8Host {
        k: usize,
        rows: usize,
        w: Vec<f32>,
        abs_blocks: Vec<f64>,
    }

    fn q8_host(split: &Split, name: &str) -> Result<Q8Host, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the draft"))?;
        if t.ty != GgmlType::Q8_0 {
            return Err(format!("{name} is {}, not Q8_0", t.ty).into());
        }
        let (k, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        let bytes = split.shard(s).ok_or("no shard")?.data(t)?;
        let mut w = vec![0.0f32; k * rows];
        dequant_row(GgmlType::Q8_0, bytes, &mut w)?;
        let abs_blocks = w
            .as_chunks::<32>()
            .0
            .iter()
            .map(|b| b.iter().map(|x| f64::from(x.abs())).sum())
            .collect();
        Ok(Q8Host {
            k,
            rows,
            w,
            abs_blocks,
        })
    }

    fn f32_tensor(split: &Split, name: &str) -> Result<Vec<f32>, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the draft"))?;
        let bytes = split.shard(s).ok_or("no shard")?.data(t)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// Per output row of `m` for one activation `x` (the one ik quantizes),
    /// optionally already off by `e_in`: the q8_1 bound, both sides'
    /// accumulation, and the input's error carried through `|w|`.
    fn gemv_bound(m: &Q8Host, x: &[f32], e_in: Option<&[f64]>) -> Vec<f64> {
        let nb = m.k / 32;
        let amax: Vec<f64> = x
            .as_chunks::<32>()
            .0
            .iter()
            .enumerate()
            .map(|(b, c)| {
                let a = c.iter().map(|v| f64::from(v.abs())).fold(0.0, f64::max);
                let e = e_in.map_or(0.0, |e| {
                    e[32 * b..32 * b + 32].iter().copied().fold(0.0, f64::max)
                });
                a + e
            })
            .collect();
        let acc = 2.0 * gamma(m.k);
        (0..m.rows)
            .map(|r| {
                let a = &m.abs_blocks[r * nb..(r + 1) * nb];
                let mut q = 0.0;
                let mut s = 0.0;
                for b in 0..nb {
                    q += (amax[b] / 127.0 / 2.0) * (1.0 + 256.0 * U) * a[b];
                    s += amax[b] * a[b];
                }
                let prop = e_in.map_or(0.0, |e| {
                    m.w[r * m.k..(r + 1) * m.k]
                        .iter()
                        .zip(e)
                        .map(|(w, e)| f64::from(w.abs()) * e)
                        .sum()
                });
                q + acc * s + prop
            })
            .collect()
    }

    /// `g·s·y`, the RMS norm of `y` in f64 rounded once: the row before the
    /// turn, whose magnitudes the bounds read.
    fn normed(y: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
        let sq: f64 = y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let s = 1.0 / (sq / y.len() as f64 + f64::from(eps)).sqrt();
        y.iter()
            .zip(g)
            .map(|(v, g)| (f64::from(*v) * f64::from(*g) * s) as f32)
            .collect()
    }

    /// An RMS norm's bound on `z = g·s·y` from `y`'s bound `e`.
    fn norm_bound(y: &[f32], e: &[f64], g: &[f32], eps: f32, z: &[f32]) -> Vec<f64> {
        let k = y.len();
        let sq: f64 = y.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let d = sq + k as f64 * f64::from(eps);
        let s = 1.0 / (sq / k as f64 + f64::from(eps)).sqrt();
        let s1: f64 = y.iter().zip(e).map(|(v, e)| f64::from(v.abs()) * e).sum();
        let round = 2.0 * (gamma(k) + 4.0 * U);
        (0..k)
            .map(|i| {
                f64::from(g[i].abs()) * s * (e[i] + f64::from(y[i].abs()) * s1 / d)
                    + round * f64::from(z[i].abs())
            })
            .collect()
    }

    /// The tail rope's effect on a normed row's bound (`z` before the turn).
    fn rope_bound(e: &mut [f64], z: &[f32], rope_dims: usize, pos: i32) {
        let head = e.len() - rope_dims;
        let angle = 8.0 * U * f64::from(pos.max(0)) + 8.0 * U;
        for p in (head..e.len()).step_by(2) {
            let b = e[p] + e[p + 1] + (f64::from(z[p].abs()) + f64::from(z[p + 1].abs())) * angle;
            e[p] = b;
            e[p + 1] = b;
        }
    }

    /// Half an f16 ulp at magnitude `a`.
    fn half_ulp16(a: f64) -> f64 {
        let ex = if a < f64::from(2.0f32.powi(-14)) {
            -14
        } else {
            a.log2().floor() as i32
        };
        2.0f64.powi(ex - 10) / 2.0
    }

    /// `(worst |ours − ik| / bound, max |ours − ik|)` over the values.
    fn worst(ours: &[f32], ik: &[f32], bound: &[f64]) -> (f64, f64) {
        let mut r = (0.0f64, 0.0f64);
        for ((a, b), e) in ours.iter().zip(ik).zip(bound) {
            let d = (f64::from(*a) - f64::from(*b)).abs();
            r.0 = r.0.max(if *e > 0.0 {
                d / e
            } else if d > 0.0 {
                f64::INFINITY
            } else {
                0.0
            });
            r.1 = r.1.max(d);
        }
        r
    }

    // ---------------------------------------------------------- the blocks

    struct Host {
        fc: Q8Host,
        fc_gain: Vec<f32>,
        kv: Vec<Q8Host>,
        kv_gain: Vec<Vec<f32>>,
    }

    /// Per layer, the bound on every appended row value against ik's f16 ring
    /// row, from `main_x` (`main`, off by `e_main` from ik's) through the
    /// layer's `attn_kv`, norm and rope.
    fn ring_bounds(
        h: &Host,
        hp: &DraftHparams,
        main: &[f32],
        e_main: Option<&[f64]>,
        rb: &KvReadback,
        pos: &[i32],
    ) -> Vec<Vec<f64>> {
        let (ne, hd) = (hp.n_embd, hp.head_dim);
        (0..hp.n_layer)
            .map(|l| {
                let mut all = Vec::with_capacity(pos.len() * hd);
                for (t, &p) in pos.iter().enumerate() {
                    let x = &main[t * ne..(t + 1) * ne];
                    let e_in = e_main.map(|e| &e[t * ne..(t + 1) * ne]);
                    let e_kv = gemv_bound(&h.kv[l], x, e_in);
                    let y = &rb.kv[l][t * hd..(t + 1) * hd];
                    let z = normed(y, &h.kv_gain[l], hp.rms_eps);
                    let mut e = norm_bound(y, &e_kv, &h.kv_gain[l], hp.rms_eps, &z);
                    rope_bound(&mut e, &z, hp.rope_dims, p);
                    let out = &rb.out[l][t * hd..(t + 1) * hd];
                    for (i, v) in e.iter_mut().enumerate() {
                        *v += half_ulp16(f64::from(out[i].abs()) + *v);
                    }
                    all.extend(e);
                }
                all
            })
            .collect()
    }

    fn widen(ring: &[u16]) -> Vec<f32> {
        ring.iter().map(|&h| half_to_f32(h)).collect()
    }

    struct Cx<'a> {
        gpu: &'a Gpu,
        w: &'a DraftWeights,
        hp: &'a DraftHparams,
        host: &'a Host,
        set: &'a Set,
        ok: bool,
    }

    /// One block's graph through our append, both ways; returns the ring
    /// rows it wrote (ik's row index and its rope position).
    fn check_block(
        cx: &mut Cx<'_>,
        b: i32,
        kv: &mut KvAppend,
        rings: &mut DraftRings,
        iso: &mut DraftRings,
    ) -> Result<Vec<(usize, i32)>, GateError> {
        let (set, hp, s) = (cx.set, cx.hp, cx.gpu.stream());
        let feats = set.f32s(set.find(b, "kv", "CUDA0#dflash_kv_input_target_features#0")?)?;
        let pos = set.i32s(set.find(b, "kv", "CUDA0#dflash_kv_input_pos_ctx#0")?)?;
        let rows = set.i32s(set.find(b, "kv", "CUDA0#dflash_kv_input_rows#0")?)?;
        let fused = set.f32s(set.find(b, "kv", "dflash_kv_fused_target")?)?;
        let n = pos.len();
        let consecutive = |v: &[i32]| v.windows(2).all(|w| w[1] == w[0] + 1);
        if rows.len() != n || !consecutive(&pos) || !consecutive(&rows) || rows[0] < 0 || pos[0] < 0
        {
            return Err(
                format!("block {b}: pos {pos:?} rows {rows:?} are not consecutive runs").into(),
            );
        }
        println!(
            "{NAME}: block {b}: {n} rows, ring rows {}..={}, rope positions {}..={}",
            rows[0],
            rows[n - 1],
            pos[0],
            pos[n - 1]
        );

        // The engine's chain: features -> main_x -> every layer.
        kv.stage(s, &feats, rows[0] as u32, pos[0] as u32)?;
        kv.enqueue(cx.gpu, cx.w, rings)?;
        s.synchronize()?;
        let rb = kv.to_host(s)?;
        let ne = hp.n_embd;
        let mut e_main = Vec::with_capacity(n * ne);
        for t in 0..n {
            let x = &feats[t * kv.features()..(t + 1) * kv.features()];
            let e_y = gemv_bound(&cx.host.fc, x, None);
            e_main.extend(norm_bound(
                &rb.fc[t * ne..(t + 1) * ne],
                &e_y,
                &cx.host.fc_gain,
                hp.rms_eps,
                &rb.main[t * ne..(t + 1) * ne],
            ));
        }
        let (ratio, maxd) = worst(&rb.main, &fused, &e_main);
        let pass = ratio <= 1.0;
        println!(
            "{NAME}: block {b} main_x vs dflash_kv_fused_target: max|diff| {maxd:e}, worst |diff|/bound {ratio:.3e} {}",
            verdict(pass)
        );
        cx.ok &= pass;

        // ik's f16 ring rows as the block pass reads them.
        let ik_rows: Vec<Vec<f32>> = (0..hp.n_layer)
            .map(|l| set.f32s(set.find(b, "block", &format!("dflash_k_ctx_cache_{l}"))?))
            .collect::<Result<_, GateError>>()?;
        let hd = hp.head_dim;
        let pick = |v: &[f32]| -> Vec<f32> {
            rows.iter()
                .flat_map(|&r| v[r as usize * hd..(r as usize + 1) * hd].to_vec())
                .collect()
        };
        // The chain carries the main_x gap it has, measured, not its bound.
        let gap: Vec<f64> = rb
            .main
            .iter()
            .zip(&fused)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .collect();
        let chain = ring_bounds(cx.host, hp, &rb.main, Some(&gap), &rb, &pos);
        for l in 0..hp.n_layer {
            let ik = pick(&ik_rows[l]);
            let (ratio, maxd) = worst(&rb.out[l], &ik, &chain[l]);
            let pass = ratio <= 1.0;
            println!(
                "{NAME}: block {b} layer {l} chain rows vs ik ring: max|diff| {maxd:e}, worst |diff|/bound {ratio:.3e} {}",
                verdict(pass)
            );
            cx.ok &= pass;
        }

        // The layers alone, on ik's main_x.
        kv.set_main(s, &fused)?;
        kv.enqueue_layers(cx.gpu, cx.w, iso)?;
        s.synchronize()?;
        let rbi = kv.to_host(s)?;
        let alone = ring_bounds(cx.host, hp, &fused, None, &rbi, &pos);
        for l in 0..hp.n_layer {
            let ik = pick(&ik_rows[l]);
            let (ratio, maxd) = worst(&rbi.out[l], &ik, &alone[l]);
            let f16_same = (0..n * hd)
                .filter(|&i| f32_to_f16_bits(rbi.out[l][i]) == f32_to_f16_bits(ik[i]))
                .count();
            let pass = ratio <= 1.0;
            println!(
                "{NAME}: block {b} layer {l} alone (ik main_x) vs ik ring: max|diff| {maxd:e}, worst |diff|/bound {ratio:.3e}, \
                 f16 equal {f16_same}/{} {}",
                n * hd,
                verdict(pass)
            );
            cx.ok &= pass;
        }
        Ok(rows
            .iter()
            .zip(&pos)
            .map(|(&r, &p)| (r as usize, p))
            .collect())
    }

    /// The rows the graphs wrote so far: in our ring bit for bit what the
    /// appends left; ik's `block` view of block `b` and `kv` view of block
    /// `b + 1` agree with each other there (the set's consistency), K with V.
    fn check_rings(
        cx: &mut Cx<'_>,
        b: i32,
        written: &BTreeMap<usize, (i32, Vec<Vec<u16>>)>,
        rings: &DraftRings,
    ) -> Result<(), GateError> {
        let (set, hp, s) = (cx.set, cx.hp, cx.gpu.stream());
        let hd = hp.head_dim;
        for l in 0..hp.n_layer {
            let ours = rings.ring(l).ok_or("a ring")?.buf().to_host_vec(s)?;
            let mut own = 0usize;
            for (&r, (_, rows)) in written {
                let slot = r % hp.window;
                own += usize::from(ours[slot * hd..(slot + 1) * hd] == rows[l][..]);
            }
            let block_k = set.f32s(set.find(b, "block", &format!("dflash_k_ctx_cache_{l}"))?)?;
            let block_v = set.f32s(set.find(b, "block", &format!("dflash_v_ctx_cache_{l}"))?)?;
            let next = set
                .find(b + 1, "kv", &format!("dflash_k_ctx_cache_{l}"))
                .and_then(|r| set.f32s(r));
            let (mut kv_same, mut next_same, mut ours_ik) = (0usize, 0usize, 0usize);
            for &r in written.keys() {
                let at = r * hd..(r + 1) * hd;
                kv_same += usize::from(bits(&block_k[at.clone()]) == bits(&block_v[at.clone()]));
                if let Ok(nx) = &next {
                    next_same += usize::from(bits(&block_k[at.clone()]) == bits(&nx[at.clone()]));
                }
                let slot = r % hp.window;
                ours_ik += usize::from(widen(&ours[slot * hd..(slot + 1) * hd]) == block_k[at]);
            }
            let w = written.len();
            let next_note = match &next {
                Ok(_) => format!("ik block {b} = ik kv {} on {next_same}/{w}", b + 1),
                Err(_) => format!("no kv graph in block {}", b + 1),
            };
            let pass = own == w && kv_same == w && (next.is_err() || next_same == w);
            println!(
                "{NAME}: rings after block {b} layer {l}: {w} written rows; ours = our appends {own}/{w}; ik K = V {kv_same}/{w}; \
                 {next_note}; ours = ik bit for bit {ours_ik}/{w} (printed) {}",
                verdict(pass)
            );
            cx.ok &= pass;
        }
        Ok(())
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    pub fn run() -> Result<(), GateError> {
        let path = std::env::var_os("BLOOMERY_DSPARK_MODEL")
            .filter(|p| !p.is_empty())
            .map(PathBuf::from)
            .ok_or("BLOOMERY_DSPARK_MODEL unset — run through `just gate-gpu-dspark-kv`")?;
        let draft = Split::open(&path)?;
        let target = Split::open(ref_model_path()?)?;
        let gpu = Gpu::new()?;
        println!("{NAME}: card {}", gpu.device_name()?);
        let mut ok = true;
        let w = load(&gpu, &draft, &target, &mut ok)?;
        let hp = w.hp().clone();

        let host = Host {
            fc: q8_host(&draft, &names::fc())?,
            fc_gain: f32_tensor(&draft, &names::enc_output_norm())?,
            kv: (0..hp.n_layer)
                .map(|l| q8_host(&draft, &names::attn_kv(l)))
                .collect::<Result<_, _>>()?,
            kv_gain: (0..hp.n_layer)
                .map(|l| f32_tensor(&draft, &names::attn_kv_a_norm(l)))
                .collect::<Result<_, _>>()?,
        };
        let set = read_set(&data_dir().join("ref-draft").join(DSREF_SET))?;
        let mut kv = KvAppend::new(&gpu, &hp)?;
        let mut rings = DraftRings::new(gpu.stream(), &hp)?;
        let mut iso = DraftRings::new(gpu.stream(), &hp)?;
        println!(
            "{NAME}: rings {} B, append buffers {} B, launches per append of 1/4/64 rows {}/{}/{}",
            rings.device_bytes(),
            kv.device_bytes(),
            kv.launches(1),
            kv.launches(4),
            kv.launches(64)
        );
        let mut cx = Cx {
            gpu: &gpu,
            w: &w,
            hp: &hp,
            host: &host,
            set: &set,
            ok,
        };
        let mut written: BTreeMap<usize, (i32, Vec<Vec<u16>>)> = BTreeMap::new();
        for b in BLOCKS {
            let rows = check_block(&mut cx, b, &mut kv, &mut rings, &mut iso)?;
            let s = gpu.stream();
            let now: Vec<Vec<u16>> = (0..hp.n_layer)
                .map(|l| Ok(rings.ring(l).ok_or("a ring")?.buf().to_host_vec(s)?))
                .collect::<Result<_, GateError>>()?;
            for (r, p) in rows {
                let slot = r % hp.window;
                let per: Vec<Vec<u16>> = now
                    .iter()
                    .map(|ring| ring[slot * hp.head_dim..(slot + 1) * hp.head_dim].to_vec())
                    .collect();
                written.insert(r, (p, per));
            }
            check_rings(&mut cx, b, &written, &rings)?;
        }
        if cx.ok {
            println!(
                "PASSED: {NAME} — the draft on the card at its predicted bytes; main_x and every layer's appended rows within \
                 the q8_1 bound of ik's, chained and alone, blocks 0-3; the rings hold every append"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
