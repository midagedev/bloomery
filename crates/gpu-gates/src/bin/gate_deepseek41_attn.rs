//! GPU gate for the V4.1 attention (`bloomery_gpu_deepseek41::attn`): the
//! tensor-core segment pass, over a compressed prefix and over the rows the
//! indexer selects, and the merge with each head's sink — at every layer of
//! the V4.1 oracle sets, the five-token batch and the six decode steps.
//!
//! The caches are ours, built from the dump's own operands. `fattn-L`'s key
//! operand is ik's raw rows, followed on a layer with a stream by the
//! compressed rows it attends: a prefix of the stream, or the rows the
//! indexer's list gathers, in list order. ik's raw cache is linear —
//! operand row `r` holds position `r + offset`, the offset being where the
//! last token's visible run ends — so each visible position `p` goes to our
//! ring slot `p % W`, and ik's own write of each token's K must sit at that
//! token's position. A prefix goes to the front of our compressed buffer; a
//! selected row goes to its stream row, and the kernel reads it through
//! ik's list. Every other row of every buffer is the f16 NaN: a row read
//! past a token's counts, or a stream row the list does not name, reaches
//! the output as NaN, which `max_rel_err` refuses. The counts are the
//! engine's rule — `min(pos + 1, W)` ring rows and `(pos + 1) / ratio`
//! compressed rows, at most the list's length where the indexer selects —
//! and the dump's mask must name exactly those keys. Where the dump reads
//! the stream's state, each compressed row is proven the state's row — the
//! list's entry, or the prefix index — bit for bit, but the one row this
//! step's compressor writes.
//!
//! ik attends by one of two CPU paths, and the dump says which. The generic
//! flash attention rounds the query to f16 and keeps the value sum in f16.
//! The iqk kernels, taken where the graph builds their key index
//! (`mask_to_idx-L`), keep the query in f32 and sum in f32. On their T = 1
//! index branch the builds in [`T1_SINK_DEFECT_BUILDS`] hand each thread's
//! run of heads the sinks of the first heads (`iqk_flash_attn.cpp`, the
//! `neq1 == 1` branch passes `sinks` without the run's offset), so on such a
//! layer of such a set the dump is the attention of those sinks, and the
//! layer's rules and launches take them ([`ik_index_sinks`]); every other
//! build folds each head's own sink there. Three distances per layer, all
//! asserted:
//! - (i) ik's arithmetic simulated here over ik's key order vs `fattn-L`.
//!   On a generic layer, bit for bit: the f16 dot's fixed lane tree, the
//!   value sum rounded to f16 at every key, glibc's `expf`, the sink folded
//!   after the keys. On an iqk layer, the f32 query's softmax in f64 within
//!   `IQK_SIM_BAND` — the f32 lane order and `v_expf` are not transcribed.
//!   This proves the key set, the scale and the sinks, and on a generic
//!   layer the key order.
//! - (ii) the kernel vs our rule — the f16 query, exact f16 products summed
//!   in f64, the logit rounded to f32 and scaled in f32, the softmax and the
//!   value sum in f64 with the sink in the denominator — within
//!   `KERNEL_BAND`, not bit for bit: the kernel sums the products on the
//!   tensor cores and the softmax and the values in f32, each in its own
//!   order. The rule is written out here; it calls nothing the kernel runs.
//! - (iii) the kernel vs `fattn-L` (`ik_rel`), within the band of the path
//!   ik took, each derived from the distance between our rule and ik's on
//!   that path; beside it the values off by more than that band of their
//!   own magnitude, for the kernel (`ik_off`) and for our rule
//!   (`rule_off`).
//!
//! Plus: a rerun bit-identical; each token of the batch run alone (T = 1)
//! bit-identical to its rows of the batch launch; one captured graph per
//! compressed source replayed with its inputs rewritten between replays —
//! per token of the batch with the query and the counts, and over lists and
//! counts with the selected rows — bit-identical to eager runs (the grid
//! comes from the buffer heights and the list's length, never from the
//! counts or the lists); a depth case per compressed source that the sets
//! cannot reach — a full ring and compressed keys over many segments, the
//! last one partial, and for the selected rows list entries past the count
//! that name NaN rows — against our rule with the model's sinks; the same
//! selected list with one visible entry naming the row just past the stream,
//! which the kernel must refuse by raising its fault (`FaultSite::AttnSel`),
//! not read as a plausible row; and every entry compiled with no local
//! depot. Every
//! partial and output buffer is NaN before each launch, so a slot a kernel
//! fails to write is caught too.
//!
//! Ring ⧺ staging, "batch = sequence": synthetic rows from fixed seeds,
//! batches of 1, 127, 128, 129 and 300 tokens whose first position is 0, 5,
//! 127, 128 or 300 — before, at and across the wrap of the model's window —
//! each with no compressed rows, a stream prefix and a list. One staged
//! launch over ring ⧺ staging plus the commit against one ring-only launch
//! per token in order, each writing its row into the ring first: every
//! token's output bit for bit and the ring after the commit byte for byte;
//! the ring's slots past its positions and the staging rows past the batch
//! are NaN. One captured staged launch and commit replayed at other first
//! positions against eager runs; and each shape the staged enqueue and the
//! commit refuse, refused by name before any launch.
//!
//! Outside these sets: ik's iqk arithmetic below `IQK_SIM_BAND`.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_attn: built without the `deepseek41` feature; see `just gate-gpu-ds41-attn`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_attn", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu, GpuError, LAYER_NONE};
    use bloomery_gpu_deepseek41::attn::{
        self, AttnArgs, AttnKernels, CommitArgs, LATENT, SelectedRows, Staged,
    };
    use bloomery_gpu_gates::oracle::{Set, for_arch};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, Layout, RefManifest, RefRow, RowKind, activations, bits_equal,
        checks_failed, mask_bits_in, max_rel_err, no_local_depot, open_split, ref_ints,
        ref_model_path, ref_tensor_logical_in, same_bits, verdict, widened_f16_rows_in,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{GgmlType, f32_to_f16_bits, half_to_f32};
    use gguf::{Split, Value};
    use model::arch::Arch;

    /// (i) on a generic layer: our transcription of ik's arithmetic against
    /// ik's own output. The transcription reproduces the sets bit for bit
    /// (each layer's line counts the equal values); the asserted band is the
    /// B4 plan's rather than bit identity because the simulation calls this
    /// host's `expf`, and a libm that moved by an ulp since the sets were
    /// dumped would otherwise redden a correct key set.
    const IK_SIM_BAND: f32 = 2e-7;
    /// (i) on an iqk layer: the f32 query's softmax in f64 against ik's f32
    /// sums and `v_expf`. The band is 2.5 times the largest such distance
    /// over the iqk layers of the sets.
    // PIN(2026-09-23): the largest was 1.98e-6 (d2_unfused layer 6) over
    // the 190 iqk layers, every one on the T = 1 index branch.
    const IQK_SIM_BAND: f32 = 4.95e-6;
    /// (iii) on a generic layer: ik keeps the value sum in f16, rounded at
    /// every key, and we keep it in f32, so the kernel's distance to the
    /// dump is that rule distance plus the kernel's own distance to our
    /// rule, (ii), plus the simulation's, (i). The band is 2.5 times the
    /// largest rule distance over the batch set's layers, the factor the B4
    /// plan derived its band with.
    // PIN(2026-09-23): the largest rule distance over the 40 layers was
    // 1.96e-3 (layer 22, ratio 1, up to ten keys a token), with (i) at 0
    // and (ii) at most 2.25e-6 on every layer. The plan's 2e-3 was 2.5
    // times the distance of layers 0 and 2 alone.
    // PIN(2026-09-23): the decode steps' generic layers, up to 128 keys,
    // reach 2.32e-3 (d1n layer 0) with (i) at 0 — inside the band, which
    // stays; 2.5 times that would be 5.8e-3.
    const IK_BAND: f32 = 4.9e-3;
    /// (iii) on an iqk layer: ik's query is f32 and ours the f16 the tensor
    /// cores take, so the kernel's distance to the dump is that rule
    /// distance, plus (ii), plus (i). The band is 2.5 times the largest rule
    /// distance over the iqk layers of the sets.
    // PIN(2026-09-23): the largest rule distance was 3.87e-4 (d2_unfused
    // layer 14) over the 190 iqk layers.
    const IQK_BAND: f32 = 9.675e-4;
    /// The f16 NaN every cache row we did not fill holds.
    const NAN_F16: u16 = 0x7E00;
    /// The f16 bits of a mask cell the query sees, `0.0` ([`mask_bits_in`]).
    const SEEN: u16 = 0x0000;
    /// The depth cases: compressed keys visible, and the compressed source's
    /// capacity — nine segments, the eighth partial, the ninth neutral. The
    /// prefix case's stream holds `DEPTH_COMP_ROWS` rows; the selected-row
    /// case's list holds `DEPTH_SEL_STRIDE` entries over a stream of
    /// `DEPTH_COMP_ROWS` rows.
    const DEPTH_COMP_VISIBLE: usize = 8 * attn::SEG_KEYS - 7;
    const DEPTH_COMP_ROWS: usize = 9 * attn::SEG_KEYS;
    const DEPTH_SEL_STRIDE: usize = 8 * attn::SEG_KEYS + 5;
    /// The visible entry the past-stream case plants: it names row
    /// `DEPTH_COMP_ROWS`, the first row past the stream.
    const DEPTH_PAST_ENTRY: usize = 100;
    /// The ik builds, by `# build` (the part before any `-`), whose T = 1
    /// index branch hands each thread's run of heads the sinks of the first
    /// heads. A set from any other build is held to each head's own sink, so
    /// an unlisted build with that branch reddens its iqk layers instead of
    /// passing them.
    const T1_SINK_DEFECT_BUILDS: &[&str] = &["c10fbbcc", "49ef19d0"];

    /// What a whole set shares: its shape, positions and the model's
    /// attention parameters.
    struct SetInfo {
        /// The set's name in the verdict lines.
        label: String,
        tokens: usize,
        heads: usize,
        /// `inp_pos`: each token's position.
        pos: Vec<usize>,
        /// `attention.sliding_window`: our ring's height.
        window: usize,
        /// `attention.compress_ratios`, per layer; 0 on a layer without a
        /// stream.
        ratios: Vec<u64>,
        scale: f32,
        /// ik's threads, `-t` on the set's `# flags` line.
        threads: Option<usize>,
        /// The set's build is one of [`T1_SINK_DEFECT_BUILDS`].
        t1_first_sinks: bool,
    }

    /// Which of ik's CPU flash attention paths a layer took.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum IkPath {
        /// `ggml_compute_forward_flash_attn_ext_f16`.
        Generic,
        /// The iqk kernels over the key index, on a branch that folds each
        /// head's own sink.
        Iqk,
        /// The iqk kernels on the index path's T = 1 branch, where a head
        /// folds its own sink — or, in a set of a
        /// [`T1_SINK_DEFECT_BUILDS`] build, the sink [`ik_index_sinks`]
        /// gives it.
        IqkT1,
    }

    impl IkPath {
        fn name(self) -> &'static str {
            match self {
                IkPath::Generic => "generic",
                IkPath::Iqk => "iqk",
                IkPath::IqkT1 => "iqk-t1",
            }
        }
    }

    /// One layer's inputs, our caches and ik's output.
    struct LayerCase {
        layer: usize,
        /// `raw`, `csa` or `hca`: which key operand ik attended over.
        source: &'static str,
        ratio: u64,
        path: IkPath,
        /// `q_rope-L`, `[token][head][LATENT]` f32.
        q: Vec<f32>,
        /// Our window ring, `window` rows of f16 bits.
        ring: Vec<u16>,
        /// Our compressed buffer, f16 bits: a prefix at its front, or the
        /// selected rows at their stream rows; `None` without a stream.
        comp: Option<Vec<u16>>,
        /// The indexer's list, `stride` entries per token, and the stride;
        /// `None` where the compressed rows are a prefix.
        sel: Option<(Vec<u32>, usize)>,
        /// Per token: ring rows, compressed rows visible.
        vis: Vec<u32>,
        /// Per token, the ring slots in ik's key order: by ascending
        /// position.
        ik_slots: Vec<Vec<usize>>,
        /// `blk.L.attn_sinks.weight`: the model's sink per head.
        model_sinks: Vec<f32>,
        /// The sink per head ik folded, which this layer's rules and
        /// launches take: the model's, or on [`IkPath::IqkT1`] in a set of
        /// a [`T1_SINK_DEFECT_BUILDS`] build the index branch's.
        sinks: Vec<f32>,
        /// `fattn-L`, `[token][head][LATENT]` f32.
        dump: Vec<f32>,
        /// The dump's mask and ik's K writes against our keys: the first
        /// difference.
        mask: Result<(), String>,
        /// The compressed rows against the stream state the dump reads:
        /// what was proven, or the first difference; `None` where the dump
        /// reads no state.
        state: Option<Result<String, String>>,
    }

    impl LayerCase {
        /// The launch inputs of tokens `t0 .. t0 + n` of this layer.
        fn inputs(&self, set: &SetInfo, t0: usize, n: usize) -> Inputs<'_> {
            let per_token = set.heads * LATENT;
            Inputs {
                q: &self.q[t0 * per_token..(t0 + n) * per_token],
                ring: &self.ring,
                comp: self.comp.as_deref(),
                sel: self
                    .sel
                    .as_ref()
                    .map(|(s, k)| (&s[t0 * k..(t0 + n) * k], *k)),
                vis: &self.vis[2 * t0..2 * (t0 + n)],
                sinks: &self.sinks,
                tokens: n,
                window: set.window,
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let model = ref_model_path()?;
        let split = open_split(Arch::Deepseek41, "gate-gpu-ds41-attn")?;
        let oracle = for_arch(Arch::Deepseek41)?;
        let mut sets = vec![oracle.open(Set::Cpu)?];
        for &name in oracle.step_sets {
            sets.push(oracle.open_named(name)?);
        }
        let infos = sets
            .iter()
            .map(|man| set_info(man, &split))
            .collect::<Result<Vec<_>, _>>()?;
        println!("gate_deepseek41_attn: model {}", model.display());

        let gpu = Gpu::new()?;
        let kernels = AttnKernels::load(gpu.context())?;
        let stream = gpu.stream();
        // Every entry compiles with no local depot: a spilled accumulator
        // array changes no output and no band, only the time.
        let mut ok = no_local_depot(&[
            "ds41_attn_seg",
            "ds41_attn_seg_sel",
            "ds41_attn_seg_stage",
            "ds41_attn_seg_sel_stage",
            "ds41_attn_merge",
            "ds41_ring_commit",
        ])?;
        let mut max = Maxima::default();
        let mut tally = [0usize; 3];
        // The prefix graph replays a batch layer that reads both sources, at
        // the smallest ratio: the most compressed rows per token. The
        // selected-row graph replays the selecting layer with the longest
        // list.
        let mut graph_layer: Option<(usize, LayerCase, Vec<Vec<f32>>)> = None;
        let mut sel_layer: Option<(usize, LayerCase)> = None;
        for (s, (man, set)) in sets.iter().zip(&infos).enumerate() {
            println!(
                "gate_deepseek41_attn: set {} ({}, build {}), tokens {} at positions {:?}, heads {}, \
                 window {}, scale {:e}, ik threads {:?}, T = 1 index branch sinks {}",
                set.label,
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                set.tokens,
                set.pos,
                set.heads,
                set.window,
                set.scale,
                set.threads,
                if set.t1_first_sinks {
                    "the first heads'"
                } else {
                    "each head's own"
                }
            );
            for l in 0..set.ratios.len() {
                let case = layer_case(man, &split, set, l)?;
                let (pass, t1) = check_layer(
                    &kernels,
                    stream,
                    gpu.unlabelled_sink(),
                    set,
                    &case,
                    &mut max,
                )?;
                ok &= pass;
                tally[0] += 1;
                tally[1] += usize::from(case.path != IkPath::Generic);
                tally[2] += usize::from(case.sel.is_some());
                if set.tokens > 1 {
                    if case.ratio > 0
                        && graph_layer
                            .as_ref()
                            .is_none_or(|(_, g, _)| case.ratio < g.ratio)
                    {
                        graph_layer = Some((s, case, t1));
                    }
                } else if let Some((_, k)) = &case.sel
                    && sel_layer
                        .as_ref()
                        .is_none_or(|(_, c)| c.sel.as_ref().is_some_and(|(_, kc)| k > kc))
                {
                    sel_layer = Some((s, case));
                }
            }
        }
        let Some((gs, case, t1)) = graph_layer else {
            return Err("no layer of a batch set reads a compressed stream".into());
        };
        ok &= graph_replay(&gpu, &kernels, &infos[gs], &case, &t1)?;
        ok &= depth_case(
            &kernels,
            stream,
            gpu.unlabelled_sink(),
            &infos[gs],
            &case,
            &mut max,
        )?;
        let Some((ss, sel_case)) = sel_layer else {
            return Err("no layer of a decode-step set reads selected rows".into());
        };
        ok &= sel_graph_replay(&gpu, &kernels, &infos[ss], &sel_case)?;
        ok &= sel_depth_case(
            &kernels,
            stream,
            gpu.unlabelled_sink(),
            &infos[ss],
            &sel_case,
            &mut max,
        )?;
        ok &= sel_past_stream_case(&kernels, &gpu, &infos[ss], &sel_case)?;
        ok &= stage_cases(
            &kernels,
            stream,
            gpu.unlabelled_sink(),
            &infos[gs],
            &case.model_sinks,
        )?;
        ok &= stage_graph_replay(&gpu, &kernels, &infos[gs], &case.model_sinks)?;
        ok &= stage_refusals(&kernels, stream, gpu.unlabelled_sink(), &infos[gs])?;

        let (g, i) = (&max.generic, &max.iqk);
        println!(
            "gate_deepseek41_attn: {} sets, {} layer cases ({} iqk, {} selecting) — kernel_rel max \
             {:.2e} (band {KERNEL_BAND:e}); generic: ik_sim_rel max {:.2e} (band {IK_SIM_BAND:e}), \
             rule_rel max {:.2e} at {}, ik_rel max {:.2e} (band {IK_BAND:e}); iqk: ik_sim_rel max \
             {:.2e} (band {IQK_SIM_BAND:e}), rule_rel max {:.2e} at {}, ik_rel max {:.2e} (band \
             {IQK_BAND:e}) — {}",
            sets.len(),
            tally[0],
            tally[1],
            tally[2],
            max.kernel,
            g.sim,
            g.rule,
            g.rule_at,
            g.ik,
            i.sim,
            i.rule,
            i.rule_at,
            i.ik,
            verdict(ok)
        );
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// The largest distances over one path's layers; the path's (iii) band
    /// is derived from `rule`.
    #[derive(Default)]
    struct PathMax {
        sim: f32,
        rule: f32,
        rule_at: String,
        ik: f32,
    }

    /// The largest distance of each kind, printed on the final line.
    #[derive(Default)]
    struct Maxima {
        kernel: f32,
        generic: PathMax,
        iqk: PathMax,
    }

    // ------------------------------------------------------------ the sets

    fn set_info(man: &RefManifest, split: &Split) -> Result<SetInfo, GateError> {
        let pos = ref_ints(man, "inp_pos", 0, RowKind::Input, Layout::Flat)?
            .into_iter()
            .map(|x| usize::try_from(x).map_err(|_| format!("inp_pos: {x} is negative")))
            .collect::<Result<Vec<_>, _>>()?;
        let tokens = pos.len();
        if tokens == 0 {
            return Err(format!("{}: inp_pos holds no token", man.dir.display()).into());
        }
        let window = split
            .arch_get_u64("attention.sliding_window")
            .ok_or("the model has no attention.sliding_window")?;
        let window = usize::try_from(window)?;
        let key = split.arch_key("attention.compress_ratios");
        let Some(Value::Array(ratios)) = split.value(&key) else {
            return Err(format!("{key} is absent or not an array").into());
        };
        let ratios: Vec<u64> = ratios
            .iter()
            .map(|v| {
                v.as_unsigned()
                    .ok_or_else(|| format!("{key}: {v:?} is not unsigned"))
            })
            .collect::<Result<_, _>>()?;
        // The backbone's layers are the set's; the array also covers the
        // layers past them that the set does not run.
        let layers = (0..)
            .take_while(|l| {
                man.first_named(RowKind::Tensor, &format!("fattn-{l}"))
                    .is_some()
            })
            .count();
        if layers == 0 || layers > ratios.len() {
            return Err(format!(
                "{}: {layers} fattn layers, {key} {} ratios",
                man.dir.display(),
                ratios.len()
            )
            .into());
        }
        let heads = man.tensor("fattn-0", 0)?.ne[1];
        let dir = man.dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let label = match dir.strip_prefix("ref_deepseek41_") {
            Some(step) => step.strip_suffix("_every_node").unwrap_or(step).to_string(),
            None => "batch".to_string(),
        };
        Ok(SetInfo {
            label,
            tokens,
            heads: usize::try_from(heads)?,
            pos,
            window,
            ratios: ratios[..layers].to_vec(),
            scale: 1.0 / (LATENT as f32).sqrt(),
            threads: man.header.threads.map(usize::try_from).transpose()?,
            t1_first_sinks: man
                .build
                .as_deref()
                .is_some_and(|b| T1_SINK_DEFECT_BUILDS.contains(&b.split('-').next().unwrap_or(b))),
        })
    }

    /// A row's source `src` names: `src0` or `src1`.
    fn src(row: &RefRow, which: u8) -> Result<&str, GateError> {
        let s = if which == 0 { &row.src0 } else { &row.src1 };
        s.as_deref()
            .filter(|n| *n != "-")
            .ok_or_else(|| format!("{} names no src{which}", row.name).into())
    }

    /// Row `r` of `LATENT`-value f16 rows.
    fn row_of(bits: &[u16], r: usize) -> &[u16] {
        &bits[r * LATENT..(r + 1) * LATENT]
    }

    /// Rows `a` of `x` and `b` of `y` carry the same f16 bits.
    fn same_row(x: &[u16], a: usize, y: &[u16], b: usize) -> bool {
        row_of(x, a) == row_of(y, b)
    }

    // ------------------------------------------------------------ one layer

    /// One layer's case, every operand proven from the manifest first.
    fn layer_case(
        man: &RefManifest,
        split: &Split,
        set: &SetInfo,
        l: usize,
    ) -> Result<LayerCase, GateError> {
        let (tokens, heads, lat) = (set.tokens as u64, set.heads as u64, LATENT as u64);
        let fattn = man.tensor(&format!("fattn-{l}"), 0)?;
        fattn.expect("fattn", "f32", [lat, heads, tokens, 1], "FLASH_ATTN_EXT")?;
        let q_row = query_row(man, fattn, set, l)?;
        let k = key_rows(man, fattn, set, l)?;
        let (n_kv, n_raw) = (
            usize::try_from(k.keys.ne[2])?,
            usize::try_from(k.raw.ne[2])?,
        );
        let keys = widened_f16_rows_in(&man.dir, k.keys)?;
        let mask = mask_bits_in(&man.dir, k.mask)?;
        let ring = ring_of(set, &keys, &mask, n_raw, n_kv, l)?;
        let writes = raw_writes(man, set, &keys, ring.offset, l)?;
        let c = compressed(man, set, k.comp, &keys, n_raw, n_kv, l)?;
        let vis: Vec<u32> = ring
            .counts
            .iter()
            .zip(&c.counts)
            .flat_map(|(&w, &n)| [w, n])
            .map(u32::try_from)
            .collect::<Result<_, _>>()?;
        let mask_ok = mask_matches(&mask, n_kv, n_raw, set, &ring, &c.counts);
        let path = match k.index_width {
            None => IkPath::Generic,
            Some(w) if set.tokens == 1 && w < n_kv => IkPath::IqkT1,
            Some(_) => IkPath::Iqk,
        };
        let model_sinks = sinks_of(split, l, set.heads)?;
        let sinks = if path == IkPath::IqkT1 && set.t1_first_sinks {
            let threads = set.threads.ok_or_else(|| {
                format!("layer {l}: ik's T = 1 index branch, but no -t in # flags")
            })?;
            ik_index_sinks(&model_sinks, threads)
        } else {
            model_sinks.clone()
        };
        Ok(LayerCase {
            layer: l,
            source: k.source,
            ratio: set.ratios[l],
            path,
            q: ref_tensor_logical_in(&man.dir, q_row)?,
            ring: ring.rows,
            comp: c.comp,
            sel: c.sel,
            vis,
            ik_slots: ring.ik_slots,
            model_sinks,
            sinks,
            dump: ref_tensor_logical_in(&man.dir, fattn)?,
            mask: mask_ok.and(writes),
            state: c.state,
        })
    }

    /// `fattn-L`'s query: `q_rope-L` whole, through a view and a permute.
    fn query_row<'a>(
        man: &'a RefManifest,
        fattn: &RefRow,
        set: &SetInfo,
        l: usize,
    ) -> Result<&'a RefRow, GateError> {
        let (tokens, heads, lat) = (set.tokens as u64, set.heads as u64, LATENT as u64);
        let q_name = format!("q_rope-{l}");
        let q_view = format!("{q_name} (view)");
        if fattn.src0.as_deref() != Some(format!("{q_view} (permuted)").as_str()) {
            return Err(
                format!("fattn-{l} reads {:?}, want {q_view} (permuted)", fattn.src0).into(),
            );
        }
        let q_row = man.tensor(&q_name, 0)?;
        q_row.expect("q", "f32", [lat, heads, tokens, 1], "ROPE")?;
        let v_row = man.tensor(&q_view, 0)?;
        v_row.expect("q view", "f32", q_row.ne, "VIEW")?;
        if v_row.src0.as_deref() != Some(q_name.as_str()) || v_row.sum != q_row.sum {
            return Err(format!("{q_view} is not the whole of {q_name}").into());
        }
        Ok(q_row)
    }

    /// The rows `fattn-L` attends over, each proven by type, shape and
    /// source.
    struct KeyRows<'a> {
        /// `raw`, `csa` or `hca`.
        source: &'static str,
        /// The key operand: ik's raw rows, ⧺ the compressed rows on a layer
        /// with a stream.
        keys: &'a RefRow,
        /// ik's raw rows: its cache through a view, cropped by a second
        /// where the cache outgrows the window.
        raw: &'a RefRow,
        /// The compressed rows; `None` without a stream.
        comp: Option<&'a RefRow>,
        mask: &'a RefRow,
        /// The key index's width where the graph builds one from this
        /// layer's mask — the mark of ik's iqk path.
        index_width: Option<usize>,
    }

    fn key_rows<'a>(
        man: &'a RefManifest,
        fattn: &RefRow,
        set: &SetInfo,
        l: usize,
    ) -> Result<KeyRows<'a>, GateError> {
        let lat = LATENT as u64;
        let k_perm = src(fattn, 1)?;
        let k_name = k_perm
            .strip_suffix(" (permuted)")
            .ok_or_else(|| format!("fattn-{l} reads keys {k_perm}, not a permute"))?;
        let keys = man.tensor(k_name, 0)?;
        let n_kv = keys.ne[2];
        keys.expect("keys", "f16", [lat, 1, n_kv, 1], "in")?;
        let kp_row = man.tensor(k_perm, 0)?;
        kp_row.expect("keys permuted", "f16", [lat, n_kv, 1, 1], "PERMUTE")?;
        if kp_row.src0.as_deref() != Some(k_name) || kp_row.sum != keys.sum {
            return Err(format!("{k_perm} is not the whole of {k_name}").into());
        }
        let raw_names = [format!("raw_k-{l}"), format!("raw_k-{l} (view)")];
        let is_raw = |n: &str| raw_names.iter().any(|r| r == n);
        let (source, raw, comp, mask) = if is_raw(k_name) {
            ("raw", keys, None, raw_mask(man, l)?)
        } else {
            let source = if k_name == format!("csa_k_all-{l}") {
                "csa"
            } else if k_name == format!("hca_k_all-{l}") {
                "hca"
            } else {
                return Err(
                    format!("fattn-{l} reads keys {k_name}, not a known key operand").into(),
                );
            };
            let raw_part = src(keys, 0)?;
            if keys.op != "CONCAT" || !is_raw(raw_part) {
                return Err(format!("{k_name} is not ik's raw rows ⧺ the compressed rows").into());
            }
            (
                source,
                man.tensor(raw_part, 0)?,
                Some(man.tensor(src(keys, 1)?, 0)?),
                man.tensor(&format!("{source}_kq_mask-{l}"), 0)?,
            )
        };
        raw.expect("raw keys", "f16", [lat, 1, raw.ne[2], 1], "VIEW")?;
        let ratio = set.ratios[l];
        if (ratio == 0) != (source == "raw") {
            return Err(format!("layer {l}: ratio {ratio} but ik attends over {k_name}").into());
        }
        mask.expect("mask", "f16", [n_kv, mask.ne[1], 1, 1], "in")?;
        if mask.ne[1] < set.tokens as u64 {
            return Err(format!(
                "{} has {} query rows for {} tokens",
                mask.name, mask.ne[1], set.tokens
            )
            .into());
        }
        let m2i_name = format!("mask_to_idx-{l}");
        let index_width = match man.first_named(RowKind::Tensor, &m2i_name) {
            None => None,
            Some(m) if m.src0.as_deref() == Some(mask.name.as_str()) => {
                Some(usize::try_from(m.ne[0])?)
            }
            Some(m) => {
                return Err(format!(
                    "{m2i_name} indexes {:?}, not the mask {}",
                    m.src0, mask.name
                )
                .into());
            }
        };
        Ok(KeyRows {
            source,
            keys,
            raw,
            comp,
            mask,
            index_width,
        })
    }

    /// The raw-window mask ik reads on a layer without a stream: the input
    /// every layer's mask starts from — named after the last layer's
    /// callback, so found by its prefix — cropped where the window is.
    fn raw_mask(man: &RefManifest, l: usize) -> Result<&RefRow, GateError> {
        let cropped = format!("dsv4_raw_mask_padded-{l} (view) (cont)");
        if let Some(r) = man.find(RowKind::Tensor, &cropped, 0) {
            return Ok(r);
        }
        man.only_with_prefix(RowKind::Input, "dsv4_raw_mask_padded")
    }

    /// Our window ring and where its rows came from.
    struct Ring {
        /// `window` rows of f16 bits: position `p` at slot `p % W`, NaN
        /// elsewhere.
        rows: Vec<u16>,
        /// Per token, the ring rows it sees: `min(pos + 1, W)`.
        counts: Vec<usize>,
        /// Per token, the ring slots in ik's key order: by ascending
        /// position.
        ik_slots: Vec<Vec<usize>>,
        /// Operand row `r` holds position `r + offset`.
        offset: usize,
    }

    /// The ring from ik's raw rows (`keys`, f16 bits, of which the first
    /// `n_raw` rows are raw): the offset is where the last token's visible
    /// run ends in the dump's mask (`mask`, f16 bits).
    fn ring_of(
        set: &SetInfo,
        keys: &[u16],
        mask: &[u16],
        n_raw: usize,
        n_kv: usize,
        l: usize,
    ) -> Result<Ring, GateError> {
        let w = set.window;
        let counts: Vec<usize> = set.pos.iter().map(|&p| (p + 1).min(w)).collect();
        let last = set.tokens - 1;
        let end = (0..n_raw)
            .rev()
            .find(|&i| mask[last * n_kv + i] == SEEN)
            .ok_or_else(|| format!("layer {l}: the last token sees no raw row"))?;
        let offset = set.pos[last].checked_sub(end).ok_or_else(|| {
            format!(
                "layer {l}: raw row {end} visible at position {}",
                set.pos[last]
            )
        })?;
        let mut rows = vec![NAN_F16; w * LATENT];
        let mut slot_pos = vec![None; w];
        let mut ik_slots = Vec::with_capacity(set.tokens);
        for (t, &p) in set.pos.iter().enumerate() {
            let first = p + 1 - counts[t];
            for q in first..=p {
                let slot = q % w;
                let r = q
                    .checked_sub(offset)
                    .filter(|&r| r < n_raw)
                    .ok_or_else(|| {
                        format!(
                            "layer {l}: position {q} is no raw row (offset {offset}, {n_raw} rows)"
                        )
                    })?;
                match slot_pos[slot] {
                    Some(have) if have != q => {
                        return Err(format!(
                            "layer {l}: positions {have} and {q} share ring slot {slot} — the set's \
                             batch must not wrap the ring"
                        )
                        .into());
                    }
                    Some(_) => {}
                    None => {
                        slot_pos[slot] = Some(q);
                        rows[slot * LATENT..(slot + 1) * LATENT].copy_from_slice(row_of(keys, r));
                    }
                }
            }
            ik_slots.push((first..=p).map(|q| q % w).collect());
        }
        Ok(Ring {
            rows,
            counts,
            ik_slots,
            offset,
        })
    }

    /// ik's write of each token's K: the `SET_ROWS` source row, rounded to
    /// the cache's f16, is the operand row at that token's position.
    fn raw_writes(
        man: &RefManifest,
        set: &SetInfo,
        keys: &[u16],
        offset: usize,
        l: usize,
    ) -> Result<Result<(), String>, GateError> {
        let write = man.tensor(&format!("dsv4_raw_k_write-{l}"), 0)?;
        let k_src = man.tensor(src(write, 0)?, 0)?;
        k_src.expect(
            "K written",
            "f32",
            [LATENT as u64, set.tokens as u64, 1, 1],
            "in",
        )?;
        let written = ref_tensor_logical_in(&man.dir, k_src)?;
        Ok(set.pos.iter().enumerate().try_for_each(|(t, &p)| {
            let r = p - offset;
            let same = written[t * LATENT..(t + 1) * LATENT]
                .iter()
                .zip(row_of(keys, r))
                .all(|(&x, &k)| f32_to_f16_bits(x) == k);
            if same {
                Ok(())
            } else {
                Err(format!("token {t}: {} is not raw row {r}", k_src.name))
            }
        }))
    }

    /// A layer's compressed source.
    struct Compressed {
        /// Our buffer: a prefix at its front, or the selected rows at their
        /// stream rows; `None` without a stream.
        comp: Option<Vec<u16>>,
        /// The list and its length where the indexer selects.
        sel: Option<(Vec<u32>, usize)>,
        /// Per token, the compressed keys visible: the engine's rule.
        counts: Vec<usize>,
        /// The rows checked against the stream's state, where the dump
        /// reads it.
        state: Option<Result<String, String>>,
    }

    /// Layer `l`'s compressed rows after ik's `n_raw` raw ones in `keys`
    /// (f16 bits): gathered by the indexer's list, or a prefix.
    fn compressed(
        man: &RefManifest,
        set: &SetInfo,
        comp_row: Option<&RefRow>,
        keys: &[u16],
        n_raw: usize,
        n_kv: usize,
        l: usize,
    ) -> Result<Compressed, GateError> {
        let Some(c) = comp_row else {
            return Ok(Compressed {
                comp: None,
                sel: None,
                counts: vec![0; set.tokens],
                state: None,
            });
        };
        let ratio = usize::try_from(set.ratios[l])?;
        let n_comp = |p: usize| (p + 1).checked_div(ratio).unwrap_or(0);
        if c.op == "RESHAPE" && man.tensor(src(c, 0)?, 0)?.op == "GET_ROWS" {
            return selected_rows(man, set, c, keys, n_raw, n_kv, n_comp(set.pos[0]));
        }
        let lat = LATENT as u64;
        let n_c = c.ne[2];
        c.expect("compressed keys", "f16", [lat, 1, n_c, 1], "in")?;
        let n_c = usize::try_from(n_c)?;
        let counts: Vec<usize> = set.pos.iter().map(|&p| n_comp(p)).collect();
        let held = counts.iter().copied().max().unwrap_or(0);
        if n_raw + n_c != n_kv || held > n_c {
            return Err(format!(
                "layer {l}: {held} compressed rows seen, {} holds {n_c} after {n_raw} raw of {n_kv}",
                c.name
            )
            .into());
        }
        // Two segments past the rows seen: a partly visible one and a wholly
        // neutral one.
        let height = (held.div_ceil(attn::SEG_KEYS) + 1) * attn::SEG_KEYS;
        let mut comp = vec![NAN_F16; height * LATENT];
        for j in 0..held {
            comp[j * LATENT..(j + 1) * LATENT].copy_from_slice(row_of(keys, n_raw + j));
        }
        // A prefix read from the stream's state: operand row n_raw + j is
        // state row j, but the row this step writes.
        let state = match src(c, 0) {
            Ok(s) if c.op == "VIEW" && man.first_named(RowKind::Input, s).is_some() => {
                let st = man.input(s, 0)?;
                st.expect("stream state", "f16", [lat, st.ne[1], 1, 1], "in")?;
                if usize::try_from(st.ne[1])? < held {
                    return Err(format!("{s} holds {} rows, {held} seen", st.ne[1]).into());
                }
                let sv = widened_f16_rows_in(&man.dir, st)?;
                let differ: Vec<usize> = (0..held)
                    .filter(|&j| !same_row(keys, n_raw + j, &sv, j))
                    .collect();
                Some(state_verdict(s, &differ, n_comp(set.pos[set.tokens - 1])))
            }
            _ => None,
        };
        Ok(Compressed {
            comp: Some(comp),
            sel: None,
            counts,
            state,
        })
    }

    /// The rows the indexer selects, read from the dump of one decode step:
    /// the gathered rows (`GET_ROWS` of the stream's state by the list)
    /// reshaped into the key operand after the raw rows. Our stream is the
    /// state's height, NaN but the rows the visible entries name, each ik's
    /// gathered row; the visible count is the engine's rule, the stream's
    /// rows at most the list's length.
    fn selected_rows(
        man: &RefManifest,
        set: &SetInfo,
        c: &RefRow,
        keys: &[u16],
        n_raw: usize,
        n_kv: usize,
        n_comp: usize,
    ) -> Result<Compressed, GateError> {
        let lat = LATENT as u64;
        let g = man.tensor(src(c, 0)?, 0)?;
        let k = g.ne[1];
        g.expect("gathered rows", "f16", [lat, k, 1, 1], "GET_ROWS")?;
        c.expect("selected keys", "f16", [lat, 1, k, 1], "RESHAPE")?;
        if c.sum != g.sum {
            return Err(format!("{} is not the whole of {}", c.name, g.name).into());
        }
        if set.tokens != 1 {
            return Err(format!(
                "{}: a selecting set is one decode step, this one has {} tokens",
                g.name, set.tokens
            )
            .into());
        }
        let k = usize::try_from(k)?;
        if n_raw + k != n_kv {
            return Err(format!("{}: {k} rows after {n_raw} raw of {n_kv}", c.name).into());
        }
        let state_name = src(g, 0)?;
        let st = man.input(state_name, 0)?;
        let n_state = st.ne[1];
        st.expect("stream state", "f16", [lat, n_state, 1, 1], "in")?;
        let n_state = usize::try_from(n_state)?;
        let list_name = src(g, 1)?;
        let list: Vec<usize> = ref_ints(man, list_name, 0, RowKind::Tensor, Layout::Flat)?
            .into_iter()
            .map(|x| {
                usize::try_from(x)
                    .ok()
                    .filter(|&r| r < n_state)
                    .ok_or_else(|| format!("{list_name}: entry {x} is not a row of {state_name}"))
            })
            .collect::<Result<_, _>>()?;
        if list.len() != k {
            return Err(format!(
                "{list_name} holds {} entries, {} gathers {k}",
                list.len(),
                g.name
            )
            .into());
        }
        let gathered = widened_f16_rows_in(&man.dir, g)?;
        if keys[n_raw * LATENT..] != gathered[..] {
            return Err(format!(
                "the key operand's rows after the raw ones are not {}",
                g.name
            )
            .into());
        }
        let sv = widened_f16_rows_in(&man.dir, st)?;
        let differ: Vec<usize> = (0..k)
            .filter(|&j| !same_row(&sv, list[j], &gathered, j))
            .map(|j| list[j])
            .collect();
        let count = n_comp.min(k);
        let mut comp = vec![NAN_F16; n_state * LATENT];
        for (j, &r) in list.iter().enumerate().take(count) {
            let row = row_of(&gathered, j);
            let dst = &mut comp[r * LATENT..(r + 1) * LATENT];
            if dst[0] != NAN_F16 && dst != row {
                return Err(format!("{list_name} names row {r} twice with different rows").into());
            }
            dst.copy_from_slice(row);
        }
        let sel = list
            .iter()
            .map(|&r| u32::try_from(r))
            .collect::<Result<_, _>>()?;
        Ok(Compressed {
            comp: Some(comp),
            sel: Some((sel, k)),
            counts: vec![count],
            state: Some(state_verdict(state_name, &differ, n_comp)),
        })
    }

    /// The compressed rows against the stream's state: every row equal but
    /// the ones listed in `differ`, which must all be the row this step's
    /// compressor writes, `n_comp - 1`.
    fn state_verdict(state: &str, differ: &[usize], n_comp: usize) -> Result<String, String> {
        let written = n_comp.checked_sub(1);
        match differ.iter().find(|&&r| Some(r) != written) {
            None if differ.is_empty() => Ok(format!("{state}:exact")),
            None => Ok(format!("{state}:exact-but-written-{}", differ[0])),
            Some(r) => Err(format!("{state} row {r} is not the key read")),
        }
    }

    /// The dump's mask (f16 bits, every cell 0 or −inf — [`mask_bits_in`]):
    /// token `t` sees exactly its raw run and its first compressed keys.
    fn mask_matches(
        mask: &[u16],
        n_kv: usize,
        n_raw: usize,
        set: &SetInfo,
        ring: &Ring,
        counts: &[usize],
    ) -> Result<(), String> {
        (0..set.tokens).try_for_each(|t| {
            let row = &mask[t * n_kv..(t + 1) * n_kv];
            let dump: Vec<usize> = (0..n_kv).filter(|&i| row[i] == SEEN).collect();
            let hi = set.pos[t] - ring.offset;
            let ours: Vec<usize> = (hi + 1 - ring.counts[t]..=hi)
                .chain((0..counts[t]).map(|j| n_raw + j))
                .collect();
            if dump == ours {
                Ok(())
            } else {
                Err(format!(
                    "token {t}: the dump sees keys {dump:?}, we see {ours:?}"
                ))
            }
        })
    }

    /// `blk.L.attn_sinks.weight`: one f32 logit per head.
    fn sinks_of(split: &Split, l: usize, heads: usize) -> Result<Vec<f32>, GateError> {
        let name = format!("blk.{l}.attn_sinks.weight");
        let (shard, info) = split
            .find(&name)
            .ok_or_else(|| format!("{name} is not in the model"))?;
        if info.ty != GgmlType::F32 || info.dims != [heads as u64] {
            return Err(format!(
                "{name} is {:?} {:?}, want F32 [{heads}]",
                info.ty, info.dims
            )
            .into());
        }
        let bytes = split.shard(shard).ok_or("shard out of range")?.data(info)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// The sink each head folds on ik's T = 1 index branch
    /// (`iqk_flash_attn.cpp`, `neq1 == 1`): the heads are split over the
    /// threads in runs of `n = ceil(heads / threads)`, one fewer from
    /// thread `heads - threads * (n - 1)` on, and each thread hands its run
    /// the sinks from the first head on — head `first + j` of a run folds
    /// `sinks[j]`.
    fn ik_index_sinks(sinks: &[f32], threads: usize) -> Vec<f32> {
        let heads = sinks.len();
        let per = heads.div_ceil(threads);
        let mut out = sinks.to_vec();
        for ith in 0..threads {
            let (mut n, mut first) = (per, ith * per);
            if per * threads > heads {
                let mid = heads - threads * (per - 1);
                if ith >= mid {
                    n -= 1;
                    first = mid * per + (ith - mid) * n;
                }
            }
            let n = n.min(heads.saturating_sub(first));
            out[first..first + n].copy_from_slice(&sinks[..n]);
        }
        out
    }

    // ------------------------------------------------------------ host rules

    /// Keys token `t` sees, widened: the ring rows at `slots`, then its
    /// `count` compressed rows — a prefix of the stream, or its list's first
    /// entries, each clamped to the stream as the kernel reads it.
    fn keys_of<'a>(
        ring: &'a [f32],
        slots: &[usize],
        comp: Option<&'a [f32]>,
        sel: Option<(&[u32], usize)>,
        t: usize,
        count: usize,
    ) -> Vec<&'a [f32]> {
        let mut keys: Vec<&[f32]> = slots
            .iter()
            .map(|&s| &ring[s * LATENT..(s + 1) * LATENT])
            .collect();
        if let Some(c) = comp {
            let height = c.len() / LATENT;
            for j in 0..count {
                let r = sel.map_or(j, |(s, k)| s[t * k + j] as usize);
                assert!(
                    r < height,
                    "keys_of: entry {r} names no row of a stream of {height}"
                );
                keys.push(&c[r * LATENT..(r + 1) * LATENT]);
            }
        }
        keys
    }

    /// Every token's keys in the kernel's walk order: ring slots in order,
    /// then the compressed keys.
    fn walk_keys<'a>(
        ring: &'a [f32],
        comp: Option<&'a [f32]>,
        sel: Option<(&[u32], usize)>,
        vis: &[u32],
    ) -> Vec<Vec<&'a [f32]>> {
        vis.chunks(2)
            .enumerate()
            .map(|(t, c)| {
                let slots: Vec<usize> = (0..c[0] as usize).collect();
                keys_of(ring, &slots, comp, sel, t, c[1] as usize)
            })
            .collect()
    }

    fn widen(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| half_to_f32(b)).collect()
    }

    /// A query row as the f16 values the tensor cores and ik's generic path
    /// see.
    fn query_f16(q: &[f32]) -> Vec<f32> {
        q.iter().map(|&v| half_to_f32(f32_to_f16_bits(v))).collect()
    }

    /// ik's f16 dot on AVX2 (`ggml_vec_dot_f16`, F16 step 32 over four
    /// eight-lane FMA accumulators), and its reduction: accumulators 0 += 2,
    /// 1 += 3, 0 += 1, then the two 128-bit halves, then two horizontal adds.
    fn ik_dot(k: &[f32], q: &[f32]) -> f32 {
        let mut sum = [[0.0f32; 8]; 4];
        for i in (0..LATENT).step_by(32) {
            for (j, acc) in sum.iter_mut().enumerate() {
                for (lane, a) in acc.iter_mut().enumerate() {
                    let e = i + j * 8 + lane;
                    *a = k[e].mul_add(q[e], *a);
                }
            }
        }
        for lane in 0..8 {
            sum[0][lane] += sum[2][lane];
            sum[1][lane] += sum[3][lane];
        }
        for lane in 0..8 {
            sum[0][lane] += sum[1][lane];
        }
        let t: [f32; 4] = std::array::from_fn(|i| sum[0][i] + sum[0][i + 4]);
        (t[0] + t[1]) + (t[2] + t[3])
    }

    /// ik's generic CPU flash attention for one query row with F16 keys and
    /// values (`ggml_compute_forward_flash_attn_ext_f16`): the query rounded
    /// to f16, the online softmax over the keys in order with the value sum
    /// in f16 — scaled (`ggml_vec_scale_f16`) on a new max, the key added by
    /// `ggml_vec_mad_f16`, each rounded to f16 — the sum's update
    /// `S = S·ms + vs` rounded twice, a product and then a sum (ik's build
    /// leaves it uncontracted, unlike the vector FMAs), then the sink folded
    /// into the f32 value sum and `S` the same way, then `1/S`.
    fn ik_rule(q: &[f32], keys: &[&[f32]], scale: f32, sink: f32) -> Vec<f32> {
        let q16 = query_f16(q);
        let mut acc = vec![0u16; LATENT];
        let (mut s_sum, mut m) = (0.0f32, f32::NEG_INFINITY);
        for k in keys {
            let s = ik_dot(k, &q16) * scale;
            let (ms, vs) = if s > m {
                let ms = (m - s).exp();
                m = s;
                for a in &mut acc {
                    *a = f32_to_f16_bits(half_to_f32(*a) * ms);
                }
                (ms, 1.0)
            } else {
                (1.0, (s - m).exp())
            };
            for (a, &x) in acc.iter_mut().zip(*k) {
                *a = f32_to_f16_bits(x.mul_add(vs, half_to_f32(*a)));
            }
            s_sum = s_sum * ms + vs;
        }
        let mut out = widen(&acc);
        let (ms, vs) = if sink > m {
            let ms = (m - sink).exp();
            for o in &mut out {
                *o *= ms;
            }
            (ms, 1.0)
        } else {
            (1.0, (sink - m).exp())
        };
        s_sum = s_sum * ms + vs;
        let inv = if s_sum == 0.0 { 0.0 } else { 1.0 / s_sum };
        for o in &mut out {
            *o *= inv;
        }
        out
    }

    /// The exact softmax for one query row as given: each logit the exact
    /// sum of the products rounded to f32 and scaled in f32, then the
    /// softmax and the value sum in f64 with the sink one more logit in the
    /// denominator. Over the f32 query it stands for ik's iqk kernels (the
    /// f32 query times the f16 keys in `iqk_gemm_default_floats`, f32 sums,
    /// `v_expf`), whose f32 lane order and `exp` it does not transcribe.
    fn exact_rule(q: &[f32], keys: &[&[f32]], scale: f32, sink: f32) -> Vec<f32> {
        let logits: Vec<f64> = keys
            .iter()
            .map(|k| {
                let dot: f64 = k
                    .iter()
                    .zip(q)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                f64::from((dot as f32) * scale)
            })
            .collect();
        let sink = f64::from(sink);
        let mx = logits.iter().copied().fold(sink, f64::max);
        let mut denom = (sink - mx).exp();
        let mut num = vec![0.0f64; LATENT];
        for (k, &s) in keys.iter().zip(&logits) {
            let w = (s - mx).exp();
            denom += w;
            for (n, &v) in num.iter_mut().zip(*k) {
                *n += w * f64::from(v);
            }
        }
        num.iter().map(|&n| (n / denom) as f32).collect()
    }

    /// Our rule for one query row: the query rounded to f16, the tensor
    /// cores' input, then [`exact_rule`].
    fn our_rule(q: &[f32], keys: &[&[f32]], scale: f32, sink: f32) -> Vec<f32> {
        exact_rule(&query_f16(q), keys, scale, sink)
    }

    /// A rule for one query row: the f32 query, its keys in order, the
    /// scale and the head's sink.
    type RowRule = fn(&[f32], &[&[f32]], f32, f32) -> Vec<f32>;

    /// A rule over every query row of a launch: `q` holds `keys.len()`
    /// tokens of `heads` rows, and row `t * heads + h` sees token `t`'s keys
    /// with head `h`'s sink.
    fn rule_all(
        rule: RowRule,
        q: &[f32],
        keys: &[Vec<&[f32]>],
        sinks: &[f32],
        scale: f32,
    ) -> Vec<f32> {
        let heads = sinks.len();
        let mut out = Vec::with_capacity(q.len());
        for (t, keys) in keys.iter().enumerate() {
            for (h, &sink) in sinks.iter().enumerate() {
                let row = (t * heads + h) * LATENT;
                out.extend(rule(&q[row..row + LATENT], keys, scale, sink));
            }
        }
        out
    }

    // ------------------------------------------------------------ launches

    /// One launch's host inputs.
    #[derive(Clone, Copy)]
    struct Inputs<'a> {
        q: &'a [f32],
        ring: &'a [u16],
        comp: Option<&'a [u16]>,
        /// The lists and their stride, when the compressed rows are
        /// selected.
        sel: Option<(&'a [u32], usize)>,
        vis: &'a [u32],
        sinks: &'a [f32],
        tokens: usize,
        window: usize,
    }

    /// A launch's device buffers: inputs uploaded once, partials and output
    /// refilled with NaN before every launch.
    struct Launch {
        q: DeviceBuffer<f32>,
        window: DeviceTensor<u16>,
        comp: Option<DeviceTensor<u16>>,
        sel: Option<(DeviceBuffer<u32>, usize)>,
        vis: DeviceBuffer<u32>,
        sinks: DeviceBuffer<f32>,
        tokens: usize,
        heads: usize,
        part_v: DeviceBuffer<f32>,
        part_ms: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
        /// Where the selected-row entry raises a list entry past the stream.
        fault: FaultSink,
    }

    impl Launch {
        fn new(stream: &CudaStream, i: &Inputs<'_>, fault: FaultSink) -> Result<Launch, GateError> {
            let heads = i.sinks.len();
            let comp = i
                .comp
                .map(|c| DeviceTensor::upload(stream, c, c.len() / LATENT, LATENT))
                .transpose()?;
            let comp_keys = match i.sel {
                Some((_, k)) => k,
                None => comp.as_ref().map_or(0, DeviceTensor::rows),
            };
            let sel = i
                .sel
                .map(|(s, k)| DeviceBuffer::from_host(stream, s).map(|b| (b, k)))
                .transpose()?;
            let segs = attn::segments(i.window, comp_keys);
            let rows = i.tokens * heads;
            Ok(Launch {
                q: DeviceBuffer::from_host(stream, i.q)?,
                window: DeviceTensor::upload(stream, i.ring, i.window, LATENT)?,
                comp,
                sel,
                vis: DeviceBuffer::from_host(stream, i.vis)?,
                sinks: DeviceBuffer::from_host(stream, i.sinks)?,
                tokens: i.tokens,
                heads,
                part_v: DeviceBuffer::from_host(
                    stream,
                    &vec![f32::NAN; attn::partials_v_len(rows, segs)],
                )?,
                part_ms: DeviceBuffer::from_host(
                    stream,
                    &vec![f32::NAN; attn::partials_ms_len(rows, segs)],
                )?,
                y: DeviceBuffer::from_host(stream, &vec![f32::NAN; rows * LATENT])?,
                fault,
            })
        }

        /// NaN into the partials and the output (synchronous copies).
        fn poison(&mut self, stream: &CudaStream) -> Result<(), GateError> {
            self.part_v
                .copy_from_host(stream, &vec![f32::NAN; self.part_v.len()])?;
            self.part_ms
                .copy_from_host(stream, &vec![f32::NAN; self.part_ms.len()])?;
            self.y
                .copy_from_host(stream, &vec![f32::NAN; self.y.len()])?;
            Ok(())
        }

        /// The launch's arguments over its buffers.
        fn args(&mut self, scale: f32) -> AttnArgs<'_> {
            AttnArgs {
                q: &self.q,
                window: &self.window,
                compressed: self.comp.as_ref(),
                selected: self.sel.as_ref().map(|(rows, stride)| SelectedRows {
                    rows,
                    stride: *stride,
                }),
                vis: &self.vis,
                sinks: &self.sinks,
                scale,
                tokens: self.tokens,
                heads: self.heads,
                part_v: &mut self.part_v,
                part_ms: &mut self.part_ms,
                y: &mut self.y,
                fault: self.fault,
            }
        }

        fn enqueue(
            &mut self,
            kernels: &AttnKernels,
            stream: &CudaStream,
            scale: f32,
        ) -> Result<(), GpuError> {
            kernels.enqueue(stream, self.args(scale))
        }

        /// One eager launch on NaN-filled partials and output, read back.
        fn run(
            &mut self,
            kernels: &AttnKernels,
            stream: &CudaStream,
            scale: f32,
        ) -> Result<Vec<f32>, GateError> {
            self.poison(stream)?;
            self.enqueue(kernels, stream, scale)?;
            stream.synchronize()?;
            Ok(self.y.to_host_vec(stream)?)
        }
    }

    /// Values of `y` farther from `d` than `band` times `d`'s own magnitude:
    /// the per-value view that `max_rel_err`, divided by the largest
    /// magnitude, does not give.
    fn off_count(y: &[f32], d: &[f32], band: f32) -> usize {
        y.iter()
            .zip(d)
            .filter(|(a, b)| (**a - **b).abs() > band * b.abs())
            .count()
    }

    /// Our rule over the kernel's walk order, and ik's rule — the one of
    /// the path ik took — over ik's key order: the ring rows by ascending
    /// position, then the compressed keys as the kernel walks them.
    fn host_rules(set: &SetInfo, case: &LayerCase) -> (Vec<f32>, Vec<f32>) {
        let ring = widen(&case.ring);
        let comp = case.comp.as_deref().map(widen);
        let sel = case.sel.as_ref().map(|(s, k)| (&s[..], *k));
        let ours = rule_all(
            our_rule,
            &case.q,
            &walk_keys(&ring, comp.as_deref(), sel, &case.vis),
            &case.sinks,
            set.scale,
        );
        let ik_keys: Vec<Vec<&[f32]>> = case
            .ik_slots
            .iter()
            .enumerate()
            .map(|(t, slots)| {
                keys_of(
                    &ring,
                    slots,
                    comp.as_deref(),
                    sel,
                    t,
                    case.vis[2 * t + 1] as usize,
                )
            })
            .collect();
        let ik_sim_rule: RowRule = match case.path {
            IkPath::Generic => ik_rule,
            IkPath::Iqk | IkPath::IqkT1 => exact_rule,
        };
        let ik_sim = rule_all(ik_sim_rule, &case.q, &ik_keys, &case.sinks, set.scale);
        (ours, ik_sim)
    }

    /// Each token of a batch launched alone: whether every one is its rows
    /// of the batch output `y` bit for bit, and the T = 1 outputs; nothing
    /// for a one-token set.
    fn t1_runs(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        case: &LayerCase,
        y: &[f32],
    ) -> Result<(bool, Vec<Vec<f32>>), GateError> {
        let per_token = set.heads * LATENT;
        let mut t1 = Vec::with_capacity(set.tokens);
        let mut same = true;
        if set.tokens > 1 {
            for t in 0..set.tokens {
                let y1 = Launch::new(stream, &case.inputs(set, t, 1), fault)?
                    .run(kernels, stream, set.scale)?;
                same &= bits_equal(&y1, &y[t * per_token..(t + 1) * per_token]);
                t1.push(y1);
            }
        }
        Ok((same, t1))
    }

    /// One layer: the three distances, the rerun and, for a batch, the
    /// T = 1 launches. Returns the verdict and the T = 1 outputs, token by
    /// token.
    fn check_layer(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        case: &LayerCase,
        max: &mut Maxima,
    ) -> Result<(bool, Vec<Vec<f32>>), GateError> {
        let (ours, ik_sim) = host_rules(set, case);
        let (sim_band, band) = match case.path {
            IkPath::Generic => (IK_SIM_BAND, IK_BAND),
            IkPath::Iqk | IkPath::IqkT1 => (IQK_SIM_BAND, IQK_BAND),
        };
        let mut launch = Launch::new(stream, &case.inputs(set, 0, set.tokens), fault)?;
        let y = launch.run(kernels, stream, set.scale)?;
        let rerun = bits_equal(&y, &launch.run(kernels, stream, set.scale)?);
        let (t1_same, t1) = t1_runs(kernels, stream, fault, set, case, &y)?;

        let sim_rel = max_rel_err(&ik_sim, &case.dump)?;
        let rule_rel = max_rel_err(&ours, &ik_sim)?;
        let kernel_rel = max_rel_err(&y, &ours)?;
        let ik_rel = max_rel_err(&y, &case.dump)?;
        let path_max = if case.path == IkPath::Generic {
            &mut max.generic
        } else {
            &mut max.iqk
        };
        path_max.sim = path_max.sim.max(sim_rel);
        if rule_rel > path_max.rule {
            path_max.rule = rule_rel;
            path_max.rule_at = format!("{} layer {}", set.label, case.layer);
        }
        path_max.ik = path_max.ik.max(ik_rel);
        max.kernel = max.kernel.max(kernel_rel);
        let state_ok = case.state.as_ref().is_none_or(Result::is_ok);
        let pass = case.mask.is_ok()
            && state_ok
            && sim_rel <= sim_band
            && kernel_rel <= KERNEL_BAND
            && rerun
            && t1_same
            && ik_rel <= band;
        let counts = |k: usize| {
            case.vis
                .chunks(2)
                .map(|c| c[k].to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let n = case.dump.len();
        println!(
            "attn set={} layer={} src={} ratio={} keys={} ik={} window_rows={} comp_rows={} mask={} \
             state={} ik_sim_rel={sim_rel:.2e} ik_sim_bits={}/{n} rule_rel={rule_rel:.2e} \
             kernel_rel={kernel_rel:.2e} rerun={} t1={} ik_rel={ik_rel:.2e} ik_off={}/{n} \
             rule_off={}/{n} {}",
            set.label,
            case.layer,
            case.source,
            case.ratio,
            match &case.sel {
                None => "prefix".to_string(),
                Some((_, k)) => format!("sel/{k}"),
            },
            case.path.name(),
            counts(0),
            counts(1),
            match &case.mask {
                Ok(()) => "exact".to_string(),
                Err(e) => format!("MISMATCH({e})"),
            },
            match &case.state {
                None => "-".to_string(),
                Some(Ok(s)) => s.clone(),
                Some(Err(e)) => format!("MISMATCH({e})"),
            },
            same_bits(&ik_sim, &case.dump),
            if rerun { "bit-identical" } else { "DIFFERS" },
            match (set.tokens > 1, t1_same) {
                (false, _) => "-",
                (true, true) => "bit-identical",
                (true, false) => "DIFFERS",
            },
            off_count(&y, &case.dump, band),
            off_count(&ours, &case.dump, band),
            verdict(pass)
        );
        Ok((pass, t1))
    }

    /// One graph captured with a T = 1 launch of `case`'s layer and
    /// replayed for every token, the query and the counts rewritten in
    /// place between replays: each replay bit-identical to that token's
    /// eager T = 1 run. A grid taken from the counts would freeze the
    /// captured token's segments.
    fn graph_replay(
        gpu: &Gpu,
        kernels: &AttnKernels,
        set: &SetInfo,
        case: &LayerCase,
        t1: &[Vec<f32>],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let per_token = set.heads * LATENT;
        let mut launch = Launch::new(stream, &case.inputs(set, 0, 1), gpu.unlabelled_sink())?;
        let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
        let mut same = 0;
        for (t, want) in t1.iter().enumerate() {
            launch
                .q
                .copy_from_host(stream, &case.q[t * per_token..(t + 1) * per_token])?;
            launch
                .vis
                .copy_from_host(stream, &case.vis[2 * t..2 * t + 2])?;
            launch.poison(stream)?;
            graph.launch(stream)?;
            stream.synchronize()?;
            same += usize::from(bits_equal(&launch.y.to_host_vec(stream)?, want));
        }
        let pass = same == t1.len();
        println!(
            "graph set={} layer={} keys=prefix nodes={} replays={} bit-identical={same}/{} {}",
            set.label,
            case.layer,
            graph.node_count(),
            t1.len(),
            t1.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// One graph captured with a selecting layer's launch and replayed over
    /// other lists and counts — the list reversed and rotated, the count cut
    /// to part of a segment, to one key and to none — the list and the
    /// counts rewritten in place between replays: each replay bit-identical
    /// to an eager launch of the same inputs. A grid taken from the counts
    /// or the list would freeze the captured selection.
    fn sel_graph_replay(
        gpu: &Gpu,
        kernels: &AttnKernels,
        set: &SetInfo,
        case: &LayerCase,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (list, stride) = case
            .sel
            .as_ref()
            .ok_or("the replayed layer selects no rows")?;
        let count = case.vis[1] as usize;
        let cut = count
            .checked_sub(37)
            .ok_or("the replayed layer sees under 38 selected rows")?;
        let mut reversed = list.clone();
        reversed[..count].reverse();
        let mut rotated = list.clone();
        rotated[..count].rotate_left(count / 3);
        let variants = [
            (list.clone(), count),
            (reversed, count),
            (list.clone(), cut),
            (rotated, 1),
            (list.clone(), 0),
        ];
        let base = case.inputs(set, 0, 1);
        let mut launch = Launch::new(stream, &base, gpu.unlabelled_sink())?;
        let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
        let mut same = 0;
        for (l, c) in &variants {
            let vis = [case.vis[0], u32::try_from(*c)?];
            let want = Launch::new(
                stream,
                &Inputs {
                    sel: Some((l, *stride)),
                    vis: &vis,
                    ..base
                },
                gpu.unlabelled_sink(),
            )?
            .run(kernels, stream, set.scale)?;
            let (sel_buf, _) = launch
                .sel
                .as_mut()
                .ok_or("the captured launch holds no list")?;
            sel_buf.copy_from_host(stream, l)?;
            launch.vis.copy_from_host(stream, &vis)?;
            launch.poison(stream)?;
            graph.launch(stream)?;
            stream.synchronize()?;
            same += usize::from(bits_equal(&launch.y.to_host_vec(stream)?, &want));
        }
        let pass = same == variants.len();
        println!(
            "graph set={} layer={} keys=sel/{stride} nodes={} replays={} counts={} bit-identical={same}/{} {}",
            set.label,
            case.layer,
            graph.node_count(),
            variants.len(),
            variants
                .iter()
                .map(|(_, c)| c.to_string())
                .collect::<Vec<_>>()
                .join(","),
            variants.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// The query rows of the last token of `case` and a full ring of
    /// synthetic rows: the depth cases' shared inputs.
    fn depth_ring<'a>(set: &SetInfo, case: &'a LayerCase) -> (&'a [f32], Vec<u16>) {
        let per_token = set.heads * LATENT;
        let q = &case.q[(set.tokens - 1) * per_token..];
        let ring = activations(LATENT, set.window, 7)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        (q, ring)
    }

    /// The shape the sets cannot reach with a prefix: one token seeing a
    /// full ring and a compressed prefix of many segments, the last partial
    /// and one more wholly past the count — synthetic rows, `case`'s last
    /// query and the model's sinks, against our rule, plus a rerun.
    fn depth_case(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        case: &LayerCase,
        max: &mut Maxima,
    ) -> Result<bool, GateError> {
        let w = set.window;
        let (q, ring) = depth_ring(set, case);
        let mut comp = vec![NAN_F16; DEPTH_COMP_ROWS * LATENT];
        for (c, v) in comp
            .iter_mut()
            .zip(activations(LATENT, DEPTH_COMP_VISIBLE, 11))
        {
            *c = f32_to_f16_bits(v);
        }
        let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
        let (ring_w, comp_w) = (widen(&ring), widen(&comp));
        let ours = rule_all(
            our_rule,
            q,
            &walk_keys(&ring_w, Some(&comp_w), None, &vis),
            &case.model_sinks,
            set.scale,
        );
        let inputs = Inputs {
            q,
            ring: &ring,
            comp: Some(&comp),
            sel: None,
            vis: &vis,
            sinks: &case.model_sinks,
            tokens: 1,
            window: w,
        };
        let mut launch = Launch::new(stream, &inputs, fault)?;
        let y = launch.run(kernels, stream, set.scale)?;
        let rerun = bits_equal(&y, &launch.run(kernels, stream, set.scale)?);
        let kernel_rel = max_rel_err(&y, &ours)?;
        max.kernel = max.kernel.max(kernel_rel);
        let pass = kernel_rel <= KERNEL_BAND && rerun;
        println!(
            "depth keys=prefix window_rows={w}/{w} comp_rows={DEPTH_COMP_VISIBLE}/{DEPTH_COMP_ROWS} \
             segments={} kernel_rel={kernel_rel:.2e} rerun={} {}",
            attn::segments(w, DEPTH_COMP_ROWS),
            if rerun { "bit-identical" } else { "DIFFERS" },
            verdict(pass)
        );
        Ok(pass)
    }

    /// The shape the sets cannot reach with selected rows: one token seeing
    /// a full ring and a list of many segments, its visible entries a
    /// scrambled set of stream rows, the last visible segment partial and
    /// one more wholly past the count; the entries past the count name rows
    /// no visible entry names, which hold NaN. Synthetic rows, `case`'s last
    /// query and the model's sinks, against our rule, plus a rerun.
    ///
    /// PIN(2026-09-24): no silent failure — a list entry past the stream is
    /// refused, not clamped; this case kept its numeric pin with every entry
    /// in the stream, and the planted entry moved to [`sel_past_stream_case`].
    fn sel_depth_case(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        case: &LayerCase,
        max: &mut Maxima,
    ) -> Result<bool, GateError> {
        let w = set.window;
        let (q, ring) = depth_ring(set, case);
        let (list, comp) = depth_list(None)?;
        let rows = DEPTH_COMP_ROWS;
        let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
        let (ring_w, comp_w) = (widen(&ring), widen(&comp));
        let sel = Some((&list[..], DEPTH_SEL_STRIDE));
        let ours = rule_all(
            our_rule,
            q,
            &walk_keys(&ring_w, Some(&comp_w), sel, &vis),
            &case.model_sinks,
            set.scale,
        );
        let inputs = Inputs {
            q,
            ring: &ring,
            comp: Some(&comp),
            sel,
            vis: &vis,
            sinks: &case.model_sinks,
            tokens: 1,
            window: w,
        };
        let mut launch = Launch::new(stream, &inputs, fault)?;
        let y = launch.run(kernels, stream, set.scale)?;
        let rerun = bits_equal(&y, &launch.run(kernels, stream, set.scale)?);
        let kernel_rel = max_rel_err(&y, &ours)?;
        max.kernel = max.kernel.max(kernel_rel);
        let pass = kernel_rel <= KERNEL_BAND && rerun;
        println!(
            "depth keys=sel/{DEPTH_SEL_STRIDE} window_rows={w}/{w} comp_rows={DEPTH_COMP_VISIBLE}/{rows} \
             unused_entries={} past_stream_entries=0 segments={} kernel_rel={kernel_rel:.2e} rerun={} {}",
            DEPTH_SEL_STRIDE - DEPTH_COMP_VISIBLE,
            attn::segments(w, DEPTH_SEL_STRIDE),
            if rerun { "bit-identical" } else { "DIFFERS" },
            verdict(pass)
        );
        Ok(pass)
    }

    /// The depth case's list — a bijection of the stream's rows but the
    /// last, scrambled, visible entry [`DEPTH_PAST_ENTRY`] replaced by
    /// `planted` when given — and its stream: the rows the visible entries
    /// name hold synthetic values, every other row NaN.
    fn depth_list(planted: Option<u32>) -> Result<(Vec<u32>, Vec<u16>), GateError> {
        let rows = DEPTH_COMP_ROWS;
        let mut list: Vec<u32> = (0..DEPTH_SEL_STRIDE)
            .map(|j| u32::try_from((j * 7919 + 13) % (rows - 1)))
            .collect::<Result<_, _>>()?;
        if let Some(r) = planted {
            list[DEPTH_PAST_ENTRY] = r;
        }
        let data: Vec<u16> = activations(LATENT, rows, 11)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let mut comp = vec![NAN_F16; rows * LATENT];
        for &r in &list[..DEPTH_COMP_VISIBLE] {
            let r = r as usize;
            if r < rows {
                comp[r * LATENT..(r + 1) * LATENT]
                    .copy_from_slice(&data[r * LATENT..(r + 1) * LATENT]);
            }
        }
        Ok((list, comp))
    }

    /// The depth case's list with visible entry [`DEPTH_PAST_ENTRY`] naming
    /// row `DEPTH_COMP_ROWS` — the first row past the stream. The kernel
    /// must raise [`FaultSite::AttnSel`] on the launch's sink, and the word,
    /// read back and cleared, must name exactly that. The output is printed
    /// against the host's rule with the entry read as the stream's last row
    /// (what the kernel used to do), over a stream whose every row holds
    /// values — evidence, not a contract: the step that reads this fault
    /// back is refused.
    fn sel_past_stream_case(
        kernels: &AttnKernels,
        gpu: &Gpu,
        set: &SetInfo,
        case: &LayerCase,
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let w = set.window;
        let (q, ring) = depth_ring(set, case);
        let rows = DEPTH_COMP_ROWS;
        let (list, _) = depth_list(Some(u32::try_from(rows)?))?;
        let vis = [u32::try_from(w)?, u32::try_from(DEPTH_COMP_VISIBLE)?];
        let mut clamped = list.clone();
        clamped[DEPTH_PAST_ENTRY] = u32::try_from(rows - 1)?;
        // Every row holds values here, so whichever row a kernel reads for the
        // planted entry, the output stays finite and comparable.
        let comp_clamped: Vec<u16> = activations(LATENT, rows, 11)
            .into_iter()
            .map(f32_to_f16_bits)
            .collect();
        let (ring_w, comp_w) = (widen(&ring), widen(&comp_clamped));
        let last_row = rule_all(
            our_rule,
            q,
            &walk_keys(
                &ring_w,
                Some(&comp_w),
                Some((&clamped[..], DEPTH_SEL_STRIDE)),
                &vis,
            ),
            &case.model_sinks,
            set.scale,
        );
        let inputs = Inputs {
            q,
            ring: &ring,
            comp: Some(&comp_clamped),
            sel: Some((&list[..], DEPTH_SEL_STRIDE)),
            vis: &vis,
            sinks: &case.model_sinks,
            tokens: 1,
            window: w,
        };
        gpu.clear_fault()?;
        let y =
            Launch::new(stream, &inputs, gpu.unlabelled_sink())?.run(kernels, stream, set.scale)?;
        let got = gpu.take_fault()?;
        let want = Fault {
            layer: LAYER_NONE,
            code: FaultSite::AttnSel as u32,
        };
        let pass = got == Some(want);
        println!(
            "depth keys=sel/{DEPTH_SEL_STRIDE} past_stream_entry={rows} (comp_rows {rows}) \
             rel_to_last_row_rule={:.2e} want=attn_sel got={} {}",
            max_rel_err(&y, &last_row)?,
            got.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------------ ring ⧺ staging

    /// The staged cases' batch lengths and first positions: every pair.
    const STAGE_TOKENS: [usize; 5] = [1, 127, 128, 129, 300];
    const STAGE_BASES: [usize; 5] = [0, 5, 127, 128, 300];
    /// The staged cases' compressed stream: its ratio, its height — every
    /// case's the same, so one captured graph replays a case of any first
    /// position — and the list's stride.
    const STAGE_RATIO: usize = 4;
    const STAGE_COMP_ROWS: usize = 4 * attn::SEG_KEYS;
    const STAGE_SEL_STRIDE: usize = 64;
    /// Staging rows past the batch's, NaN.
    const STAGE_SPARE: usize = 2;
    /// The batch length the staged graph replays at every first position.
    const STAGE_REPLAY_TOKENS: usize = 129;

    /// The compressed rows a staged case reads besides its window.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum StageKeys {
        Window,
        Prefix,
        Sel,
    }

    impl StageKeys {
        const ALL: [StageKeys; 3] = [StageKeys::Window, StageKeys::Prefix, StageKeys::Sel];

        fn name(self) -> String {
            match self {
                StageKeys::Window => "window".to_string(),
                StageKeys::Prefix => "prefix".to_string(),
                StageKeys::Sel => format!("sel/{STAGE_SEL_STRIDE}"),
            }
        }
    }

    /// One staged case's host inputs: `tokens` tokens at positions `base ..`
    /// over the model's window, every position's latent row synthetic.
    struct StageCase {
        tokens: usize,
        base: usize,
        keys: StageKeys,
        window: usize,
        heads: usize,
        /// `[token][head][LATENT]` f32.
        q: Vec<f32>,
        /// Position `p`'s latent row, f16 bits, for `p < base + tokens`.
        rows: Vec<u16>,
        /// The ring before the batch: the last `min(base, window)` positions
        /// below `base` at their slots, NaN in every other slot.
        ring: Vec<u16>,
        /// The batch's rows, row `j` position `base + j`, then
        /// [`STAGE_SPARE`] NaN rows.
        staging: Vec<u16>,
        /// [`STAGE_COMP_ROWS`] stream rows: those below the batch's last
        /// compressed count hold values, the rest NaN.
        stream: Option<Vec<u16>>,
        /// `tokens` lists of [`STAGE_SEL_STRIDE`] entries: the visible ones
        /// a scramble of the token's visible rows, the rest naming NaN rows.
        sel: Option<Vec<u32>>,
        /// Per token: `min(pos + 1, window)` window keys, then its compressed
        /// rows.
        vis: Vec<u32>,
    }

    impl StageCase {
        fn new(
            set: &SetInfo,
            tokens: usize,
            base: usize,
            keys: StageKeys,
        ) -> Result<StageCase, GateError> {
            let (w, heads, end) = (set.window, set.heads, base + tokens);
            let f16 = |v: Vec<f32>| -> Vec<u16> { v.into_iter().map(f32_to_f16_bits).collect() };
            let rows = f16(activations(LATENT, end, 17));
            let mut ring = vec![NAN_F16; w * LATENT];
            for p in base.saturating_sub(w)..base {
                let s = p % w;
                ring[s * LATENT..(s + 1) * LATENT].copy_from_slice(row_of(&rows, p));
            }
            let mut staging = vec![NAN_F16; (tokens + STAGE_SPARE) * LATENT];
            staging[..tokens * LATENT].copy_from_slice(&rows[base * LATENT..end * LATENT]);
            let visible = |t: usize| (base + t + 1) / STAGE_RATIO;
            let written = visible(tokens - 1);
            if written >= STAGE_COMP_ROWS {
                return Err(format!(
                    "a staged case of {tokens} tokens from {base}: {written} compressed rows, the \
                     stream holds {STAGE_COMP_ROWS} with at least one NaN row"
                )
                .into());
            }
            let stream = (keys != StageKeys::Window).then(|| {
                let mut s = vec![NAN_F16; STAGE_COMP_ROWS * LATENT];
                s[..written * LATENT].copy_from_slice(&f16(activations(LATENT, written, 29)));
                s
            });
            let sel = (keys == StageKeys::Sel)
                .then(|| {
                    (0..tokens)
                        .flat_map(|t| {
                            let n = visible(t);
                            let live = n.min(STAGE_SEL_STRIDE);
                            // 7919 is prime and n below it, so the live
                            // entries are distinct rows.
                            (0..STAGE_SEL_STRIDE).map(move |j| {
                                if j < live {
                                    (j * 7919 + 13 + t) % n
                                } else {
                                    written + j % (STAGE_COMP_ROWS - written)
                                }
                            })
                        })
                        .map(u32::try_from)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            let vis = (0..tokens)
                .flat_map(|t| {
                    let comp = match keys {
                        StageKeys::Window => 0,
                        StageKeys::Prefix => visible(t),
                        StageKeys::Sel => visible(t).min(STAGE_SEL_STRIDE),
                    };
                    [(base + t + 1).min(w), comp]
                })
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(StageCase {
                tokens,
                base,
                keys,
                window: w,
                heads,
                q: activations(LATENT, tokens * heads, 23),
                rows,
                ring,
                staging,
                stream,
                sel,
                vis,
            })
        }

        /// The launch inputs of tokens `t0 .. t0 + n` over `ring`.
        fn inputs<'a>(
            &'a self,
            sinks: &'a [f32],
            ring: &'a [u16],
            t0: usize,
            n: usize,
        ) -> Inputs<'a> {
            let per_token = self.heads * LATENT;
            let k = STAGE_SEL_STRIDE;
            Inputs {
                q: &self.q[t0 * per_token..(t0 + n) * per_token],
                ring,
                comp: self.stream.as_deref(),
                sel: self.sel.as_ref().map(|s| (&s[t0 * k..(t0 + n) * k], k)),
                vis: &self.vis[2 * t0..2 * (t0 + n)],
                sinks,
                tokens: n,
                window: self.window,
            }
        }
    }

    /// A staged launch's device buffers: a [`Launch`] of the whole batch over
    /// the ring before it, the staging rows and the batch's first position.
    struct StageLaunch {
        launch: Launch,
        staging: DeviceTensor<u16>,
        base: DeviceBuffer<u32>,
    }

    impl StageLaunch {
        fn new(
            stream: &CudaStream,
            c: &StageCase,
            sinks: &[f32],
            fault: FaultSink,
        ) -> Result<StageLaunch, GateError> {
            Ok(StageLaunch {
                launch: Launch::new(stream, &c.inputs(sinks, &c.ring, 0, c.tokens), fault)?,
                staging: DeviceTensor::upload(
                    stream,
                    &c.staging,
                    c.staging.len() / LATENT,
                    LATENT,
                )?,
                base: DeviceBuffer::from_host(stream, &[u32::try_from(c.base)?])?,
            })
        }

        /// The staged attention, then the commit.
        fn enqueue(
            &mut self,
            kernels: &AttnKernels,
            stream: &CudaStream,
            scale: f32,
        ) -> Result<(), GpuError> {
            let staged = Staged {
                rows: &self.staging,
                base: &self.base,
            };
            kernels.enqueue_staged(stream, self.launch.args(scale), staged)?;
            kernels.enqueue_commit(
                stream,
                CommitArgs {
                    staged,
                    tokens: self.launch.tokens,
                    ring: &mut self.launch.window,
                },
            )
        }

        /// One eager run on NaN-filled partials and output: the outputs and
        /// the ring after the commit.
        fn run(
            &mut self,
            kernels: &AttnKernels,
            stream: &CudaStream,
            scale: f32,
        ) -> Result<(Vec<f32>, Vec<u16>), GateError> {
            self.launch.poison(stream)?;
            self.enqueue(kernels, stream, scale)?;
            stream.synchronize()?;
            Ok((
                self.launch.y.to_host_vec(stream)?,
                self.launch.window.buf().to_host_vec(stream)?,
            ))
        }

        /// `c`'s inputs into these buffers, of the same shapes: every buffer
        /// the launches read.
        fn load(&mut self, stream: &CudaStream, c: &StageCase) -> Result<(), GateError> {
            let l = &mut self.launch;
            l.q.copy_from_host(stream, &c.q)?;
            l.vis.copy_from_host(stream, &c.vis)?;
            l.window.buf_mut().copy_from_host(stream, &c.ring)?;
            if let (Some(buf), Some(s)) = (l.comp.as_mut(), c.stream.as_ref()) {
                buf.buf_mut().copy_from_host(stream, s)?;
            }
            if let (Some((buf, _)), Some(s)) = (l.sel.as_mut(), c.sel.as_ref()) {
                buf.copy_from_host(stream, s)?;
            }
            self.staging.buf_mut().copy_from_host(stream, &c.staging)?;
            self.base
                .copy_from_host(stream, &[u32::try_from(c.base)?])?;
            Ok(())
        }
    }

    /// `c` token by token, the decode step's rule: each token writes its
    /// row into its ring slot, then one ring-only launch. Every token's
    /// output in order, and the ring the last token leaves.
    fn stage_sequence(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        sinks: &[f32],
        c: &StageCase,
    ) -> Result<(Vec<f32>, Vec<u16>), GateError> {
        let (w, per_token, k) = (c.window, c.heads * LATENT, STAGE_SEL_STRIDE);
        let mut ring = c.ring.clone();
        let mut launch = Launch::new(stream, &c.inputs(sinks, &ring, 0, 1), fault)?;
        let mut y = Vec::with_capacity(c.tokens * per_token);
        for t in 0..c.tokens {
            let (pos, s) = (c.base + t, (c.base + t) % w);
            ring[s * LATENT..(s + 1) * LATENT].copy_from_slice(row_of(&c.rows, pos));
            launch.window.buf_mut().copy_from_host(stream, &ring)?;
            launch
                .q
                .copy_from_host(stream, &c.q[t * per_token..(t + 1) * per_token])?;
            launch
                .vis
                .copy_from_host(stream, &c.vis[2 * t..2 * t + 2])?;
            if let (Some((buf, _)), Some(sel)) = (launch.sel.as_mut(), c.sel.as_ref()) {
                buf.copy_from_host(stream, &sel[t * k..(t + 1) * k])?;
            }
            y.extend(launch.run(kernels, stream, set.scale)?);
        }
        Ok((y, launch.window.buf().to_host_vec(stream)?))
    }

    /// One staged case: the staged launch and the commit against
    /// [`stage_sequence`] — every token's output bit for bit and the ring
    /// byte for byte, the reference finite (it reads no NaN row).
    fn stage_case(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        sinks: &[f32],
        c: &StageCase,
    ) -> Result<bool, GateError> {
        let per_token = c.heads * LATENT;
        let (want, want_ring) = stage_sequence(kernels, stream, fault, set, sinks, c)?;
        let (y, ring) =
            StageLaunch::new(stream, c, sinks, fault)?.run(kernels, stream, set.scale)?;
        let same = (0..c.tokens)
            .filter(|&t| {
                let r = t * per_token..(t + 1) * per_token;
                bits_equal(&y[r.clone()], &want[r])
            })
            .count();
        let finite = want.iter().all(|v| v.is_finite());
        let ring_same = ring == want_ring;
        let pass = same == c.tokens && ring_same && finite;
        println!(
            "stage keys={} tokens={} base={} window={}: outputs bit-identical={same}/{} ring={} \
             reference {} {}",
            c.keys.name(),
            c.tokens,
            c.base,
            c.window,
            c.tokens,
            if ring_same { "identical" } else { "DIFFERS" },
            if finite { "finite" } else { "NOT FINITE" },
            verdict(pass)
        );
        Ok(pass)
    }

    /// Every staged case: each compressed source, batch length and first
    /// position.
    fn stage_cases(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
        sinks: &[f32],
    ) -> Result<bool, GateError> {
        let (mut n, mut passed) = (0usize, 0usize);
        for keys in StageKeys::ALL {
            for &tokens in &STAGE_TOKENS {
                for &base in &STAGE_BASES {
                    let c = StageCase::new(set, tokens, base, keys)?;
                    n += 1;
                    passed += usize::from(stage_case(kernels, stream, fault, set, sinks, &c)?);
                }
            }
        }
        let pass = passed == n;
        println!("stage cases={n} pass={passed} {}", verdict(pass));
        Ok(pass)
    }

    /// One graph captured with a staged launch and its commit at the first
    /// of [`STAGE_BASES`] and replayed at each, every input rewritten in
    /// place between replays: outputs and ring bit-identical to an eager
    /// run of the same case. A grid or a slot taken from the first position
    /// would freeze the captured one.
    fn stage_graph_replay(
        gpu: &Gpu,
        kernels: &AttnKernels,
        set: &SetInfo,
        sinks: &[f32],
    ) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let t = STAGE_REPLAY_TOKENS;
        let cases = STAGE_BASES
            .iter()
            .map(|&b| StageCase::new(set, t, b, StageKeys::Sel))
            .collect::<Result<Vec<_>, _>>()?;
        let mut launch = StageLaunch::new(stream, &cases[0], sinks, gpu.unlabelled_sink())?;
        let graph = gpu.capture(|_s| launch.enqueue(kernels, stream, set.scale))?;
        let mut same = 0;
        for c in &cases {
            let (want, want_ring) = StageLaunch::new(stream, c, sinks, gpu.unlabelled_sink())?
                .run(kernels, stream, set.scale)?;
            launch.load(stream, c)?;
            launch.launch.poison(stream)?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let y = launch.launch.y.to_host_vec(stream)?;
            let ring = launch.launch.window.buf().to_host_vec(stream)?;
            same += usize::from(bits_equal(&y, &want) && ring == want_ring);
        }
        let pass = same == cases.len();
        println!(
            "graph keys={} staged tokens={t} bases={} nodes={} bit-identical={same}/{} {}",
            StageKeys::Sel.name(),
            STAGE_BASES
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            graph.node_count(),
            cases.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Each shape the staged enqueue and the commit refuse: the call returns
    /// the shape error whose text names the cause, and nothing launches —
    /// the output stays the NaN it was filled with and the ring as uploaded.
    fn stage_refusals(
        kernels: &AttnKernels,
        stream: &CudaStream,
        fault: FaultSink,
        set: &SetInfo,
    ) -> Result<bool, GateError> {
        let c = StageCase::new(set, 2, 5, StageKeys::Window)?;
        let sinks = vec![0.0f32; set.heads];
        let mut l = StageLaunch::new(stream, &c, &sinks, fault)?;
        l.launch.poison(stream)?;
        let short = DeviceTensor::<u16>::zeroed(stream, 1, LATENT)?;
        let narrow = DeviceTensor::<u16>::zeroed(stream, 2, LATENT / 2)?;
        let no_word = DeviceBuffer::<u32>::zeroed(stream, 0)?;
        let mut no_ring = DeviceTensor::<u16>::zeroed(stream, 0, LATENT)?;
        let scale = set.scale;
        let StageLaunch {
            launch,
            staging,
            base,
        } = &mut l;
        let (staging, base) = (&*staging, &*base);
        let mut got: Vec<(&str, &str, Result<(), GpuError>)> = vec![
            (
                "staged: staging rows fewer than the tokens",
                "staging holds 1 rows, the batch's 2 tokens",
                kernels.enqueue_staged(stream, launch.args(scale), Staged { rows: &short, base }),
            ),
            (
                "staged: staging rows not LATENT wide",
                "staging rows are 256 wide",
                kernels.enqueue_staged(
                    stream,
                    launch.args(scale),
                    Staged {
                        rows: &narrow,
                        base,
                    },
                ),
            ),
            (
                "staged: a base buffer of no word",
                "the base buffer holds no word",
                kernels.enqueue_staged(
                    stream,
                    launch.args(scale),
                    Staged {
                        rows: staging,
                        base: &no_word,
                    },
                ),
            ),
        ];
        let mut a = launch.args(scale);
        a.window = &no_ring;
        got.push((
            "staged: a ring of no rows",
            "over a ring of no rows",
            kernels.enqueue_staged(
                stream,
                a,
                Staged {
                    rows: staging,
                    base,
                },
            ),
        ));
        /// One refused commit: its name, the text its error must hold, the
        /// rows and base it stages, its tokens, and whether it commits into
        /// the case's ring (else into a ring of no rows).
        struct Commit<'a> {
            name: &'static str,
            want: &'static str,
            rows: &'a DeviceTensor<u16>,
            base: &'a DeviceBuffer<u32>,
            tokens: usize,
            own_ring: bool,
        }
        let commits = [
            Commit {
                name: "commit: no token",
                want: "a batch of 0 tokens",
                rows: staging,
                base,
                tokens: 0,
                own_ring: true,
            },
            Commit {
                name: "commit: a ring of no rows",
                want: "into a ring of 0 rows",
                rows: staging,
                base,
                tokens: 2,
                own_ring: false,
            },
            Commit {
                name: "commit: staging rows fewer than the tokens",
                want: "staging holds 1 rows, the batch's 2 tokens",
                rows: &short,
                base,
                tokens: 2,
                own_ring: true,
            },
            Commit {
                name: "commit: staging rows not LATENT wide",
                want: "staging rows are 256 wide",
                rows: &narrow,
                base,
                tokens: 2,
                own_ring: true,
            },
            Commit {
                name: "commit: a base buffer of no word",
                want: "the base buffer holds no word",
                rows: staging,
                base: &no_word,
                tokens: 2,
                own_ring: true,
            },
        ];
        for Commit {
            name,
            want,
            rows,
            base,
            tokens,
            own_ring,
        } in commits
        {
            let ring = if own_ring {
                &mut launch.window
            } else {
                &mut no_ring
            };
            let r = kernels.enqueue_commit(
                stream,
                CommitArgs {
                    staged: Staged { rows, base },
                    tokens,
                    ring,
                },
            );
            got.push((name, want, r));
        }
        let mut pass = true;
        for (name, want, r) in &got {
            let (ok, text) = match r {
                Err(GpuError::Shape { detail, .. }) => (detail.contains(want), detail.clone()),
                Err(e) => (false, format!("not a shape error: {e}")),
                Ok(()) => (false, "accepted".to_string()),
            };
            pass &= ok;
            println!("refuse {name}: \"{text}\" names \"{want}\" {}", verdict(ok));
        }
        stream.synchronize()?;
        let quiet = launch.y.to_host_vec(stream)?.iter().all(|v| v.is_nan())
            && launch.window.buf().to_host_vec(stream)? == c.ring;
        println!(
            "refuse: {} refusals, output and ring untouched {}",
            got.len(),
            verdict(quiet)
        );
        Ok(pass && quiet)
    }
}
