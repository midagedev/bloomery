//! GPU gate for the V4.1 glue piece (`bloomery_gpu_deepseek41::chain::glue`),
//! chain G1 on the decode-step sets step4 and d1n (T = 1). The piece runs as
//! the step runs it: from the step image `params.rs` builds out of the host
//! plan and the piece's own host half (`StepRows`), and from the streams and
//! HC_PRE results each boundary reads, injected from the set.
//!
//! - (i) Host row ids: `StepRows`' ids per engram site equal the set's int
//!   input `engram_rows-L`, integer for integer. Host only.
//! - (iv) The embedding broadcast: every stream equals `hc_init` and layer 0's
//!   attention input equals `hc_attn_pre-0`, bit for bit. Its premise — the
//!   first `pre` (`hc_pre_init`) is one-hot on stream 0, so that fold is the
//!   row — is checked first and printed.
//! - (ii) Each engram site, the streams `l_out-(L−1)` and the previous ffn's
//!   HC_PRE result injected: the rows dequantized from the image equal
//!   `engram_embd-L` bit for bit; the gated streams sit within the band
//!   b4engram's chain check derives against `engram_out-L` ([`bands`] with
//!   the projection's bound, both copied from `gate_deepseek41_engram`); the
//!   fold sits within that band carried through the fold against
//!   `hc_attn_pre-L`.
//! - (iii) The head end, `l_out-(last)` and the last ffn's HC_PRE result
//!   injected: `hc_out` bit for bit (our fold is ik's rule on the same
//!   inputs); each logit's gap from `result_output` against its prediction
//!   from both sides' exact values — our q8_1 activations (128 values a
//!   block) of our norm and ik's q8_2 activations (32 a block) of its own —
//!   within both sides' f32 accumulation bounds, and the prediction within
//!   the band the two activation rules and the two norms allow
//!   ([`check_head`]); the argmax equals the set's (ties to the lowest id),
//!   both margins printed.
//! - Structure: every glue launch of one step captured as one graph. Per set,
//!   the set's inputs are written into the captured buffers, the outputs
//!   cleared, and the graph replayed: bit-identical to that set's eager run.
//!   The node count equals the prediction, `1 + 3·sites + 2·sites + 5`, all
//!   kernels.
//!
//! Distances a pin does not own (the projection's, the gates', the norm's)
//! print as diagnostics.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_chain_glue: built without the `deepseek41` feature; see `just gate-gpu-ds41-chain-glue`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_chain_glue", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeSet;

    use bloomery_gpu::head::Head;
    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_deepseek41::chain::glue::{EngramStep, Glue, StepRows};
    use bloomery_gpu_deepseek41::engram_gate::{CLAMP_MIN, PER_THREAD, ROW, inv_sqrt_row};
    use bloomery_gpu_deepseek41::hc::{HC_MIX, HC_STREAMS};
    use bloomery_gpu_deepseek41::params::{ImageDims, ImageLayout, StepImage, rope_specs};
    use bloomery_gpu_gates::ik_q8_2::{self, QK};
    use bloomery_gpu_gates::oracle;
    use bloomery_gpu_gates::oracle::deepseek41::{D1N, STEP4};
    use bloomery_gpu_gates::{
        GateError, Layout, RefManifest, RefRow, RowKind, bits_equal, checks_failed, max_rel_err,
        ref_ints, ref_model_path, ref_tensor_logical_in, ref_tensor_of_in, verdict,
    };
    use cuda_core::{DeviceBuffer, sys};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::arch::deepseek41::plan::{Planner, StepPlan};

    /// The sets this gate reads: the decode steps without a selection.
    const SETS: [&str; 2] = [STEP4, D1N];

    /// f32's unit roundoff, 2^-24.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// f64's unit roundoff, 2^-53.
    const U64: f64 = f64::EPSILON / 2.0;

    /// `gamma_n = n u / (1 - n u)`: the relative bound on a result that went
    /// through `n` f32 roundings in sequence.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// `max_rel_err`, with a non-finite value (or a length that does not
    /// match) read as an infinite distance: a pin then prints its line and
    /// fails instead of aborting the run.
    fn rel(y: &[f32], y_ref: &[f32]) -> f64 {
        max_rel_err(y, y_ref).map_or(f64::INFINITY, f64::from)
    }

    /// `gap / bound`, 0 for no gap and infinite for a gap that is not a
    /// number or a bound of zero.
    fn ratio(gap: f64, bound: f64) -> f64 {
        if gap == 0.0 {
            0.0
        } else if gap.is_finite() && bound > 0.0 {
            gap / bound
        } else {
            f64::INFINITY
        }
    }

    // --------------------------------------------------------------- the run

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        let hp = Hparams::read(&split)?;
        if hp.n_embd != ROW || hp.hc.streams != HC_STREAMS {
            return Err(format!(
                "a model of rows of {} in {} streams; the kernels take {ROW} and {HC_STREAMS}",
                hp.n_embd, hp.hc.streams
            )
            .into());
        }
        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let sites = &hp.engram.layer_ids;
        let mut keep: BTreeSet<String> = [names::output(), names::output_norm()].into();
        for &l in sites {
            keep.extend([names::engram_wkv(l), names::engram_k(l), names::engram_q(l)]);
        }
        let w = Weights::load_where(stream, &split, |name| keep.contains(name))?;
        let resident = w.names().count();
        if resident != keep.len() {
            return Err(
                format!("{resident} tensors resident, the gate keeps {}", keep.len()).into(),
            );
        }
        let mut head = Head::new(&gpu, &w, hp.rms_eps)?;
        println!(
            "gate_deepseek41_chain_glue: device {} — n_embd {} streams {} layers {} engram sites \
             {sites:?} vocab {} resident {resident} tensors",
            gpu.device_name()?,
            hp.n_embd,
            hp.hc.streams,
            hp.n_layer,
            head.n_vocab()
        );

        let table = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = Vec::with_capacity(SETS.len());
        for name in SETS {
            let man = table.open_named(name)?;
            let (pos0, tokens, before) = man.step()?;
            let ctx = man
                .header
                .ctx
                .unwrap_or((before.len() + tokens.len()) as u64);
            let planner = Planner::from_file(&split, &hp, ctx)?;
            let mut plan = StepPlan::default();
            planner.plan_into(tokens, pos0, before, &mut plan)?;
            if plan.len() != 1 {
                return Err(format!(
                    "{name}: a step of {} tokens, the piece runs one",
                    plan.len()
                )
                .into());
            }
            println!(
                "set {name}: {} (build {}) pos {pos0} token {} ctx {ctx}",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                plan.tokens[0]
            );
            sets.push(Set {
                name,
                man,
                planner,
                plan,
            });
        }

        let mut rows = StepRows::open(&split, &hp, 1)?;
        let first = sets.first().ok_or("no set")?;
        let layout = ImageLayout::new(ImageDims::of(&hp, &first.planner, 1, rows.row_bytes()))?;
        let (window, yarn) = rope_specs(&hp)?;
        let mut image = StepImage::new(layout, &window, &yarn)?;
        let mut glue = Glue::new(&gpu, &hp, image.layout())?;
        let site_w = sites
            .iter()
            .map(|&l| SiteW::load(&split, l))
            .collect::<Result<Vec<_>, _>>()?;
        println!(
            "glue: scratch {} B, image {} words",
            glue.device_bytes(),
            image.layout().words()
        );

        let n = hp.n_embd;
        let mut bufs = Bufs {
            params: DeviceBuffer::zeroed(stream, image.layout().words())?,
            streams: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            input: DeviceBuffer::zeroed(stream, n)?,
            sites: sites
                .iter()
                .map(|&layer| {
                    Ok(SiteBufs {
                        layer,
                        streams: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
                        pre: DeviceBuffer::zeroed(stream, HC_MIX)?,
                        out: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
                        input: DeviceBuffer::zeroed(stream, n)?,
                    })
                })
                .collect::<Result<Vec<_>, GpuError>>()?,
            head_streams: DeviceBuffer::zeroed(stream, HC_STREAMS * n)?,
            head_pre: DeviceBuffer::zeroed(stream, HC_MIX)?,
        };

        let mut pass = true;
        let mut runs = Vec::with_capacity(sets.len());
        for set in &sets {
            rows.fill(&split, &set.plan)?;
            pass &= check_ids(set, &hp, &rows)?;
            image.build(&set.plan, rows.embd(), rows.engram())?;
            let inputs = Inputs::read(set, &hp, image.words())?;
            inputs.write(&gpu, &mut bufs)?;
            enqueue_step(&gpu, &w, &mut glue, &mut head, &mut bufs)?;
            stream.synchronize()?;
            let outs = Outs::read(&gpu, &glue, &mut head, &bufs)?;
            pass &= check_embed(set, &outs)?;
            for (s, w) in site_w.iter().enumerate() {
                pass &= check_site(set, &hp, w, &inputs.sites[s], &outs.sites[s])?;
            }
            pass &= check_head(set, &split, &inputs, &outs)?;
            runs.push((inputs, outs));
        }

        let graph = gpu.capture(|_| enqueue_step(&gpu, &w, &mut glue, &mut head, &mut bufs))?;
        let kinds = graph.nodes()?;
        let kernels = kinds
            .iter()
            .filter(|k| k.kind == sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL)
            .count();
        // The prediction: the embedding broadcast; per site the rows, `engram_wkv`
        // and the key norm, then the gate and the fold; the hc_out fold and the
        // head's norm, q8_1, Q6_K gemv and argmax.
        let want = 1 + 3 * sites.len() + 2 * sites.len() + 5;
        for (set, (inputs, eager)) in sets.iter().zip(&runs) {
            inputs.write(&gpu, &mut bufs)?;
            bufs.clear(&gpu, &mut head)?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let replay = Outs::read(&gpu, &glue, &mut head, &bufs)?;
            let same = replay.same(eager);
            let ok = same && graph.node_count() == want && kernels == want;
            println!(
                "graph set={} nodes={} kernels={kernels} memops={} want={want} \
                 replay_bit_identical_to_eager={same} {}",
                set.name,
                graph.node_count(),
                kinds.len() - kernels,
                verdict(ok)
            );
            pass &= ok;
        }

        if pass {
            println!(
                "PASSED: gate_deepseek41_chain_glue — host row ids exact, the embedding broadcast \
                 bit for bit, each engram site's streams and fold inside the band b4engram's pins \
                 propagate, hc_out bit for bit and the logits inside their predicted gap with the \
                 set's argmax, the captured step replaying bit-identically at its node count"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// One decode-step set with its host plan.
    struct Set {
        name: &'static str,
        man: RefManifest,
        planner: Planner,
        plan: StepPlan,
    }

    // ----------------------------------------------------- the step's buffers

    /// Every buffer the captured step reads or writes besides the piece's
    /// own scratch and the head's.
    struct Bufs {
        params: DeviceBuffer<u32>,
        streams: DeviceBuffer<f32>,
        input: DeviceBuffer<f32>,
        sites: Vec<SiteBufs>,
        head_streams: DeviceBuffer<f32>,
        head_pre: DeviceBuffer<f32>,
    }

    /// An engram step's inputs and outputs.
    struct SiteBufs {
        layer: usize,
        streams: DeviceBuffer<f32>,
        pre: DeviceBuffer<f32>,
        out: DeviceBuffer<f32>,
        input: DeviceBuffer<f32>,
    }

    impl Bufs {
        /// Zero every output a step writes, the head's input included.
        fn clear(&mut self, gpu: &Gpu, head: &mut Head) -> Result<(), GateError> {
            let stream = gpu.stream();
            let mut outs = vec![&mut self.streams, &mut self.input, head.input_mut()];
            for s in &mut self.sites {
                outs.push(&mut s.out);
                outs.push(&mut s.input);
            }
            for b in outs {
                let zeros = vec![0.0f32; b.len()];
                b.copy_from_host(stream, &zeros)?;
            }
            Ok(())
        }
    }

    /// The glue launches of one step, in step order.
    fn enqueue_step(
        gpu: &Gpu,
        w: &Weights,
        glue: &mut Glue,
        head: &mut Head,
        b: &mut Bufs,
    ) -> Result<(), GpuError> {
        glue.enqueue_embed(gpu, &b.params, &mut b.streams, &mut b.input)?;
        for s in &b.sites {
            glue.enqueue_engram_kv_at(gpu, w, &b.params, s.layer)?;
        }
        for s in &mut b.sites {
            let step = EngramStep {
                streams: &s.streams,
                pre: &s.pre,
                out: &mut s.out,
                input: &mut s.input,
            };
            glue.enqueue_engram(gpu, w, s.layer, step)?;
        }
        glue.enqueue_head(gpu, w, &b.head_streams, &b.head_pre, head)
    }

    // ------------------------------------------------------- the set's inputs

    /// What a set gives the step: the image, and per boundary the streams and
    /// HC_PRE result it reads.
    struct Inputs {
        words: Vec<u32>,
        /// Per engram site: `l_out-(L−1)` and layer L−1's ffn HC_PRE result.
        sites: Vec<(Vec<f32>, Vec<f32>)>,
        head_streams: Vec<f32>,
        head_pre: Vec<f32>,
    }

    impl Inputs {
        fn read(set: &Set, hp: &Hparams, words: &[u32]) -> Result<Inputs, GateError> {
            let man = &set.man;
            let mut sites = Vec::with_capacity(hp.engram.layer_ids.len());
            for &l in &hp.engram.layer_ids {
                let prev = l.checked_sub(1).ok_or("an engram site at layer 0")?;
                sites.push((streams_after(man, prev)?, ffn_pre(man, prev)?));
            }
            let last = hp.n_layer - 1;
            Ok(Inputs {
                words: words.to_vec(),
                sites,
                head_streams: streams_after(man, last)?,
                head_pre: ffn_pre(man, last)?,
            })
        }

        fn write(&self, gpu: &Gpu, b: &mut Bufs) -> Result<(), GateError> {
            let stream = gpu.stream();
            b.params.copy_from_host(stream, &self.words)?;
            for (bs, (streams, pre)) in b.sites.iter_mut().zip(&self.sites) {
                bs.streams.copy_from_host(stream, streams)?;
                bs.pre.copy_from_host(stream, pre)?;
            }
            b.head_streams.copy_from_host(stream, &self.head_streams)?;
            b.head_pre.copy_from_host(stream, &self.head_pre)?;
            Ok(())
        }
    }

    fn src0(r: &RefRow) -> &str {
        r.src0.as_deref().unwrap_or("")
    }

    /// `l_out-l`: the streams layer `l`'s ffn HC_POST leaves, `[n, 4]`.
    fn streams_after(man: &RefManifest, l: usize) -> Result<Vec<f32>, GateError> {
        let name = format!("l_out-{l}");
        let r = man.tensor(&name, 0)?;
        r.expect(
            &name,
            "f32",
            [ROW as u64, HC_STREAMS as u64, 1, 1],
            "HC_POST",
        )?;
        ref_tensor_of_in(&man.dir, r)
    }

    /// Layer `l`'s ffn HC_PRE result, found by the mixes and the scale it
    /// reads. At T = 1 ik's layout (`[pre S T][post S T][comb S S T]`) is
    /// the kernels' (`[pre, post, comb]`).
    fn ffn_pre(man: &RefManifest, l: usize) -> Result<Vec<f32>, GateError> {
        let mixes = format!("hc_pre_mixes-{l}");
        let scale = format!("blk.{l}.hc_ffn_scale.weight");
        let node = man
            .tensors
            .iter()
            .find(|r| {
                r.op == "HC_PRE" && src0(r) == mixes && r.src1.as_deref() == Some(scale.as_str())
            })
            .ok_or_else(|| format!("no HC_PRE node reads {mixes} with {scale}"))?;
        node.expect(&node.name, "f32", [HC_MIX as u64, 1, 1, 1], "HC_PRE")?;
        ref_tensor_of_in(&man.dir, node)
    }

    // ------------------------------------------------------ the step's outputs

    /// What one step wrote, read back.
    struct Outs {
        streams: Vec<f32>,
        input: Vec<f32>,
        sites: Vec<SiteOut>,
        hc_out: Vec<f32>,
        normed: Vec<f32>,
        logits: Vec<f32>,
        token: u32,
    }

    struct SiteOut {
        rows: Vec<f32>,
        kv: Vec<f32>,
        gate: Vec<f32>,
        out: Vec<f32>,
        fold: Vec<f32>,
    }

    impl Outs {
        fn read(gpu: &Gpu, glue: &Glue, head: &mut Head, b: &Bufs) -> Result<Outs, GateError> {
            let stream = gpu.stream();
            let mut sites = Vec::with_capacity(b.sites.len());
            for s in &b.sites {
                let sb = glue
                    .site_buffers(s.layer)
                    .ok_or_else(|| format!("the piece has no engram site at layer {}", s.layer))?;
                sites.push(SiteOut {
                    rows: sb.rows.to_host_vec(stream)?,
                    kv: sb.kv.to_host_vec(stream)?,
                    gate: sb.gate.to_host_vec(stream)?,
                    out: s.out.to_host_vec(stream)?,
                    fold: s.input.to_host_vec(stream)?,
                });
            }
            Ok(Outs {
                streams: b.streams.to_host_vec(stream)?,
                input: b.input.to_host_vec(stream)?,
                sites,
                hc_out: head.input_mut().to_host_vec(stream)?,
                normed: head.normed_to_host(gpu)?,
                logits: head.logits_to_host(gpu)?,
                token: head.token(gpu)?,
            })
        }

        /// Every value bit-identical to `o`'s.
        fn same(&self, o: &Outs) -> bool {
            let sites = self.sites.len() == o.sites.len()
                && self.sites.iter().zip(&o.sites).all(|(a, b)| {
                    bits_equal(&a.rows, &b.rows)
                        && bits_equal(&a.kv, &b.kv)
                        && bits_equal(&a.gate, &b.gate)
                        && bits_equal(&a.out, &b.out)
                        && bits_equal(&a.fold, &b.fold)
                });
            sites
                && bits_equal(&self.streams, &o.streams)
                && bits_equal(&self.input, &o.input)
                && bits_equal(&self.hc_out, &o.hc_out)
                && bits_equal(&self.normed, &o.normed)
                && bits_equal(&self.logits, &o.logits)
                && self.token == o.token
        }
    }

    // --------------------------------------------------------- (i) the row ids

    /// Each site's host row ids against the set's `engram_rows-L`.
    fn check_ids(set: &Set, hp: &Hparams, rows: &StepRows) -> Result<bool, GateError> {
        let mut all = true;
        for (s, &l) in hp.engram.layer_ids.iter().enumerate() {
            let want = ref_ints(
                &set.man,
                &format!("engram_rows-{l}"),
                0,
                RowKind::Input,
                Layout::Flat,
            )?;
            let got: Vec<i64> = rows.ids(0, s).iter().map(|&v| i64::from(v)).collect();
            let same = got == want;
            println!(
                "ids set={} L={l}: {} host row ids, first {:?}, equal to engram_rows-{l}: {same} {}",
                set.name,
                got.len(),
                &got[..got.len().min(4)],
                verdict(same)
            );
            all &= same;
        }
        Ok(all)
    }

    // ------------------------------------------------- (iv) the embedding

    /// The streams and layer 0's input against the dump, and the premise.
    fn check_embed(set: &Set, o: &Outs) -> Result<bool, GateError> {
        let man = &set.man;
        let n = ROW as u64;
        let load = |name: &str, ne: [u64; 4], op: &str| -> Result<Vec<f32>, GateError> {
            let r = man.tensor(name, 0)?;
            r.expect(name, "f32", ne, op)?;
            ref_tensor_of_in(&man.dir, r)
        };
        let init = load("hc_init", [n, HC_STREAMS as u64, 1, 1], "REPEAT")?;
        let embd = load("inp_embd", [n, 1, 1, 1], "GET_ROWS")?;
        let fold0 = man.tensor("hc_attn_pre-0", 0)?;
        fold0.expect("hc_attn_pre-0", "f32", [n, 1, 1, 1], "MUL_MULTI_ADD")?;
        if src0(fold0) != "hc_init" {
            return Err(format!("hc_attn_pre-0 folds {}, not hc_init", src0(fold0)).into());
        }
        let fold0 = ref_tensor_of_in(&man.dir, fold0)?;
        let pre = ref_tensor_of_in(&man.dir, man.tensor("hc_pre_init", 0)?)?;
        let one_hot = pre.len() == HC_STREAMS
            && pre
                .iter()
                .enumerate()
                .all(|(i, &p)| p.to_bits() == if i == 0 { 1.0f32 } else { 0.0 }.to_bits());
        let row_is_fold = bits_equal(&embd, &fold0);
        println!(
            "premise set={}: hc_pre_init={pre:?} one_hot_on_stream_0={one_hot} \
             hc_attn_pre-0==inp_embd_bit={row_is_fold} {}",
            set.name,
            verdict(one_hot && row_is_fold)
        );
        let streams = bits_equal(&o.streams, &init);
        let input = bits_equal(&o.input, &fold0);
        let pass = one_hot && row_is_fold && streams && input;
        println!(
            "embed set={}: streams==hc_init_bit={streams} input==hc_attn_pre-0_bit={input} {}",
            set.name,
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------ (ii) the engram sites

    /// Nodes of one engram site, the row gather to `engram_out-L`
    /// (`gate_deepseek41_engram`'s walk).
    const CHAIN: usize = 35;
    /// Each chain node's op, in manifest order.
    const OPS: [&str; CHAIN] = [
        "GET_ROWS", "RESHAPE", "MUL_MAT", "VIEW", "CONT", "RESHAPE", "REPEAT", "VIEW", "CONT",
        "RESHAPE", "RMS_NORM", "RESHAPE", "GET_ROWS", "RESHAPE", "MUL", "RESHAPE", "CONT",
        "RESHAPE", "RMS_NORM", "RESHAPE", "GET_ROWS", "RESHAPE", "MUL", "RESHAPE", "MUL",
        "SUM_ROWS", "SCALE", "SGN", "ABS", "CLAMP", "SQRT", "MUL", "SIGMOID", "MUL", "ADD",
    ];
    // Positions in the chain of the nodes the bands read.
    const EMBD: usize = 1;
    const KV: usize = 2;
    const KN: usize = 14;
    const QN: usize = 22;
    const PROD: usize = 24;
    const SUM: usize = 25;
    const SCALED: usize = 26;
    const SIGNED: usize = 31;
    const GATE: usize = 32;
    const OUT: usize = 34;

    /// The dumped values of one site the checks and the bands read.
    struct SiteDump {
        embd: Vec<f32>,
        kv: Vec<f32>,
        kn: Vec<f32>,
        qn: Vec<f32>,
        prod: Vec<f32>,
        sum: Vec<f32>,
        scaled: Vec<f32>,
        signed: Vec<f32>,
        gate: Vec<f32>,
        out: Vec<f32>,
        fold: Vec<f32>,
    }

    /// Layer `l`'s engram chain in `man`: the [`CHAIN`] rows ending at
    /// `engram_out-l`, each op checked and the named ones by name, and the
    /// fold of its output, `hc_attn_pre-l`.
    fn site_dump(man: &RefManifest, l: usize) -> Result<SiteDump, GateError> {
        let out_name = format!("engram_out-{l}");
        let (e, _) = man.tensor_at(&out_name, 0)?;
        let start = (e + 1)
            .checked_sub(CHAIN)
            .ok_or_else(|| format!("{out_name} sits at manifest row {e}, before a whole chain"))?;
        let rows = &man.tensors[start..=e];
        let named = [
            (EMBD, format!("engram_embd-{l}")),
            (KV, format!("engram_kv-{l}")),
            (GATE, format!("engram_gate-{l}")),
            (OUT, out_name.clone()),
        ];
        for (i, row) in rows.iter().enumerate() {
            let name_ok = named
                .iter()
                .find(|(at, _)| *at == i)
                .is_none_or(|(_, want)| &row.name == want);
            if row.op != OPS[i] || !name_ok {
                return Err(format!(
                    "{out_name} chain row {i}: {}/{} {}, want {}",
                    row.name, row.occurrence, row.op, OPS[i]
                )
                .into());
            }
        }
        let fold = man.tensor(&format!("hc_attn_pre-{l}"), 0)?;
        fold.expect(&fold.name, "f32", [ROW as u64, 1, 1, 1], "MUL_MULTI_ADD")?;
        if src0(fold) != out_name {
            return Err(format!("hc_attn_pre-{l} folds {}, not {out_name}", src0(fold)).into());
        }
        let ld = |i: usize| ref_tensor_logical_in(&man.dir, &rows[i]);
        Ok(SiteDump {
            embd: ld(EMBD)?,
            kv: ld(KV)?,
            kn: ld(KN)?,
            qn: ld(QN)?,
            prod: ld(PROD)?,
            sum: ld(SUM)?,
            scaled: ld(SCALED)?,
            signed: ld(SIGNED)?,
            gate: ld(GATE)?,
            out: ld(OUT)?,
            fold: ref_tensor_of_in(&man.dir, fold)?,
        })
    }

    /// One engram site's file weights on the host: the projection's bytes
    /// and the key gains widened to f32.
    struct SiteW {
        l: usize,
        wkv: Vec<u8>,
        gk: Vec<f32>,
    }

    /// A file tensor's bytes and type.
    fn file_bytes<'a>(split: &'a Split, name: &str) -> Result<(&'a [u8], GgmlType), GateError> {
        let (s, info) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the file"))?;
        let shard = split.shard(s).ok_or("shard missing")?;
        Ok((shard.data(info)?, info.ty))
    }

    impl SiteW {
        fn load(split: &Split, l: usize) -> Result<SiteW, GateError> {
            let (wkv, ty) = file_bytes(split, &names::engram_wkv(l))?;
            if ty != GgmlType::Q8_0 {
                return Err(format!("blk.{l}.engram_wkv.weight is {ty}, want q8_0").into());
            }
            let (gk_bytes, ty) = file_bytes(split, &names::engram_k(l))?;
            if ty != GgmlType::BF16 {
                return Err(format!("blk.{l}.engram_k.weight is {ty}, want bf16").into());
            }
            let mut gk = vec![0.0f32; HC_STREAMS * ROW];
            dequant_row(GgmlType::BF16, gk_bytes, &mut gk)?;
            Ok(SiteW {
                l,
                wkv: wkv.to_vec(),
                gk,
            })
        }
    }

    /// Site `w.l` of `set`: the rows, the gated streams and the fold.
    fn check_site(
        set: &Set,
        hp: &Hparams,
        w: &SiteW,
        input: &(Vec<f32>, Vec<f32>),
        o: &SiteOut,
    ) -> Result<bool, GateError> {
        let (hc, n) = (HC_STREAMS, ROW);
        let d = site_dump(&set.man, w.l)?;
        let rows_bit = bits_equal(&o.rows, &d.embd);

        // The band b4engram's chain check derives: how far our kv sits from
        // ik's, the two exact values' distance plus both sides' accumulation
        // bounds and the f64 sums' own error, carried through the gate.
        let k_in = d.embd.len();
        let p = project(&w.wkv, (hc + 1) * n, &d.embd);
        let f64_sum = k_in as f64 * U64;
        let bk: Vec<f64> = (0..(hc + 1) * n)
            .map(|r| {
                (p.ours[r] - p.ik[r]).abs()
                    + (ik_gemv_rel(k_in) + f64_sum) * p.abs_ik[r]
                    + (our_gemv_rel(k_in) + f64_sum) * p.abs_ours[r]
            })
            .collect();
        let b = bands(&d, &w.gk, hp.rms_eps, hc, &bk[..hc * n], &bk[hc * n..]);
        let out_rel = rel(&o.out, &d.out);
        let out_ok = out_rel <= b.out;

        // The fold: both sides fold by the same rule and the same `pre`, so
        // the gap is the streams' gap through `pre` plus each side's four
        // roundings.
        let pre = &input.1[..hc];
        let bo = b.out * d.out.iter().fold(0.0f64, |a, &v| a.max(f64::from(v).abs()));
        let pre_abs: f64 = pre.iter().map(|&v| f64::from(v).abs()).sum();
        let mut fold_ratio = 0.0f64;
        for i in 0..n {
            let mag = |s: &[f32]| -> f64 {
                (0..hc)
                    .map(|j| (f64::from(pre[j]) * f64::from(s[j * n + i])).abs())
                    .sum()
            };
            let bound = bo * pre_abs + gamma(4) * (mag(&d.out) + mag(&o.out));
            let gap = (f64::from(o.fold[i]) - f64::from(d.fold[i])).abs();
            fold_ratio = fold_ratio.max(ratio(gap, bound));
        }
        let fold_ok = fold_ratio <= 1.0;

        let (kv_rel, gate_rel) = (rel(&o.kv, &d.kv), rel(&o.gate, &d.gate));
        let pass = rows_bit && out_ok && fold_ok;
        println!(
            "engram set={} L={}: rows==engram_embd_bit={rows_bit} out ik_rel={out_rel:.3e} \
             (band {:.3e}) fold max|gap|/bound={fold_ratio:.3e} — diag kv_rel={kv_rel:.3e} \
             gate_rel={gate_rel:.3e} (band {:.3e}) {}",
            set.name,
            w.l,
            b.out,
            b.gate,
            verdict(pass)
        );
        Ok(pass)
    }

    /// Roundings on our side of a norm's sum of squares: [`PER_THREAD`] fused
    /// multiply-adds per thread, five butterfly levels and three
    /// `rms_warp_tree` levels.
    const OUR_SUM_ROUNDINGS: f64 = PER_THREAD as f64 + 5.0 + 3.0;

    /// How far our norm scale and ik's can sit apart on the same row,
    /// relative to the scale (`gate_deepseek41_engram`'s derivation).
    fn scale_rel() -> f64 {
        let means = OUR_SUM_ROUNDINGS * U + U + ROW as f64 * U64 + 2.0 * U;
        (means + 2.0 * U) / 2.0 + 2.0 * U + 2.0 * U
    }

    /// The exponential's distance between the two rules, relative to it
    /// (`gate_deepseek41_engram`'s derivation).
    const EXP_REL: f64 = (1.0 + 1.0 / 268_435_456.0 + 2.0 * 0.502) * U;

    /// The bands one site's gate step carries, each `max|Δ| / M` with `M`
    /// the largest `|value|` ik wrote.
    struct Bands {
        gate: f64,
        out: f64,
    }

    /// `gate_deepseek41_engram`'s `bands` at T = 1 for the outputs this
    /// piece hands on: from ik's own values (the dump), first order in each
    /// difference with the cross terms kept, `pk`/`pv` the bound on how far
    /// our key and value rows sit from ik's. See that gate for each line's
    /// derivation.
    fn bands(d: &SiteDump, gk: &[f32], eps: f32, hc: usize, pk: &[f64], pv: &[f64]) -> Bands {
        let (key, value) = (&d.kv[..hc * ROW], &d.kv[hc * ROW..(hc + 1) * ROW]);
        let (sr, c) = (scale_rel(), f64::from(inv_sqrt_row()));
        let (mut max_gate, mut max_out, mut mag_gate, mut mag_out) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for s in 0..hc {
            let r = s * ROW..(s + 1) * ROW;
            let (k, kn, qn) = (&key[r.clone()], &d.kn[r.clone()], &d.qn[r.clone()]);
            let (p, out) = (&d.prod[r.clone()], &d.out[r]);
            let (g, bk) = (&gk[s * ROW..(s + 1) * ROW], &pk[s * ROW..(s + 1) * ROW]);
            let ssq = k
                .iter()
                .fold(0.0f64, |a, &x| a + f64::from(x) * f64::from(x));
            let sc = 1.0 / (ssq / ROW as f64 + f64::from(eps)).sqrt();
            let eps_m = k.iter().zip(bk).fold(0.0f64, |a, (&x, &e)| {
                a + 2.0 * f64::from(x).abs() * e + e * e
            }) / ssq;
            let rho = if eps_m < 1.0 {
                eps_m / (2.0 * (1.0 - eps_m).powf(1.5))
            } else {
                f64::INFINITY
            };
            let (mut sum_dp, mut sum_p) = (0.0f64, 0.0f64);
            for (i, &kv_) in k.iter().enumerate() {
                let knv = f64::from(kn[i]).abs();
                let (qv, pv_) = (f64::from(qn[i]).abs(), f64::from(p[i]).abs());
                let moved = bk[i] * sc * (1.0 + rho) + f64::from(kv_).abs() * sc * rho;
                let dk = f64::from(g[i]).abs() * moved + (sr + 4.0 * U) * knv;
                let dq = (sr + 4.0 * U) * qv;
                sum_dp += dk * qv + knv * dq + dk * dq + 2.0 * U * pv_;
                sum_p += pv_;
            }
            let dot = f64::from(d.sum[s]).abs();
            let ddot = sum_dp + 2.0 * ROW as f64 * U64 * sum_p + 2.0 * U * dot;
            let sv = f64::from(d.scaled[s]).abs();
            let ds = c * ddot + 2.0 * U * sv;
            let m = f64::from(d.signed[s]).abs();
            let dm = if sv - ds > f64::from(CLAMP_MIN) {
                ds / (2.0 * (sv - ds).sqrt()) + 2.0 * U * m
            } else {
                2.0 * (sv + ds + f64::from(CLAMP_MIN)).sqrt()
            };
            let near0 = (m - dm).max(0.0);
            let sig = 1.0 / (1.0 + (-near0).exp());
            let gv = f64::from(d.gate[s]);
            let dg = sig * (1.0 - sig) * dm + EXP_REL * gv * (1.0 - gv) + 4.0 * U * gv;
            max_gate = max_gate.max(dg);
            mag_gate = mag_gate.max(gv.abs());
            for ((&vi, &bvi), &o) in value.iter().zip(pv).zip(out) {
                let (vi, o) = (f64::from(vi).abs(), f64::from(o).abs());
                max_out = max_out.max(bvi * (gv + dg) + vi * dg + 2.0 * U * vi * gv + 2.0 * U * o);
                mag_out = mag_out.max(o);
            }
        }
        Bands {
            gate: max_gate / mag_gate,
            out: max_out / mag_out,
        }
    }

    /// The projection of one token's gathered rows by `engram_wkv` (file
    /// bytes), per output row: the f64 dot with the f32 activations (our
    /// rule's exact value), with ik's q8_2 activations (ik's exact value),
    /// and the magnitudes the accumulation bounds scale
    /// (`gate_deepseek41_engram`'s `Proj`).
    struct Proj {
        ours: Vec<f64>,
        ik: Vec<f64>,
        abs_ours: Vec<f64>,
        abs_ik: Vec<f64>,
    }

    /// [`Proj`] of `x` through the q8_0 weight `w` of `rows` rows, the rows
    /// split across the host's threads.
    fn project(w: &[u8], rows: usize, x: &[f32]) -> Proj {
        let rb = x.len() / QK * 34;
        let (qx, dx) = ik_q8_2::quantize(x);
        let mut p = Proj {
            ours: vec![0.0; rows],
            ik: vec![0.0; rows],
            abs_ours: vec![0.0; rows],
            abs_ik: vec![0.0; rows],
        };
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let chunk = rows.div_ceil(threads);
        std::thread::scope(|sc| {
            let parts = p
                .ours
                .chunks_mut(chunk)
                .zip(p.ik.chunks_mut(chunk))
                .zip(p.abs_ours.chunks_mut(chunk))
                .zip(p.abs_ik.chunks_mut(chunk))
                .enumerate();
            for (ci, (((ours, ik), ao), ai)) in parts {
                let (qx, dx) = (&qx, &dx);
                sc.spawn(move || {
                    let outs = ours.iter_mut().zip(ik).zip(ao).zip(ai);
                    for (j, (((o, i), a), b)) in outs.enumerate() {
                        let row = &w[(ci * chunk + j) * rb..(ci * chunk + j + 1) * rb];
                        let (mut so, mut si, mut sao, mut sai) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        let blocks = row.as_chunks::<34>().0.iter().zip(x.as_chunks::<QK>().0);
                        for (bi, (blk, xb)) in blocks.enumerate() {
                            let qb = Q8Block::from_bytes(blk);
                            let dw = half_to_f32(qb.d);
                            for (&qw, &xv) in qb.q.iter().zip(xb) {
                                let t = f64::from(f32::from(qw) * dw) * f64::from(xv);
                                so += t;
                                sao += t.abs();
                            }
                            let isum = ik_q8_2::block_sum(&qb, &qx[bi * QK..(bi + 1) * QK]);
                            let t = f64::from(dw) * f64::from(dx[bi]) * f64::from(isum);
                            si += t;
                            sai += t.abs();
                        }
                        (*o, *i, *a, *b) = (so, si, sao, sai);
                    }
                });
            }
        });
        p
    }

    /// ik's f32 projection's distance from its exact value, relative to
    /// `Σ_b|d_w·d_x·isum_b|` (`gate_deepseek41_engram`'s).
    fn ik_gemv_rel(k: usize) -> f64 {
        (k / QK + 4) as f64 * U
    }

    /// Our `q8_0_gemv`'s at m = 1, relative to `Σ|w·x|`
    /// (`gate_deepseek41_engram`'s).
    fn our_gemv_rel(k: usize) -> f64 {
        (k / 32 + 5) as f64 * U
    }

    // ------------------------------------------------------- (iii) the head

    /// q8_1 of one 128-value block by the crate quantizer's rule
    /// (`gate_deepseek41_hc`'s transcription, pinned there against the
    /// quantizer's bytes): the codes and `d = amax/127`, 1 for an all-zero
    /// block.
    fn q8_block(x: &[f32]) -> ([i32; 128], f32) {
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let mut q = [0i32; 128];
        for (qi, &v) in q.iter_mut().zip(x) {
            *qi = (v / d).round().clamp(-127.0, 127.0) as i32;
        }
        (q, d)
    }

    /// Per logit, in exact arithmetic: our projection minus ik's (`pred`),
    /// the band that difference must sit in (`qband`), and each side's
    /// magnitude `Σ|w·x̂|`, the scale of its f32 rounding.
    struct LogitPrediction {
        pred: Vec<f64>,
        qband: Vec<f64>,
        ours: Vec<f64>,
        ik: Vec<f64>,
    }

    /// [`LogitPrediction`] of the Q6_K weight `w` (`k` values a row) against
    /// our activations `xo` and ik's `xd`, both as the exact values their
    /// codes and scales stand for, with `e` the bound on how far the two sit
    /// apart per value; the rows split across the host's threads.
    fn predict_logits(
        w: &[u8],
        k: usize,
        xo: &[f64],
        xd: &[f64],
        e: &[f64],
    ) -> Result<LogitPrediction, GateError> {
        let rb = k / 256 * 210;
        if rb == 0 || !w.len().is_multiple_of(rb) {
            return Err(format!("output.weight holds {} bytes, not rows of {rb}", w.len()).into());
        }
        let rows = w.len() / rb;
        let mut p = LogitPrediction {
            pred: vec![0.0; rows],
            qband: vec![0.0; rows],
            ours: vec![0.0; rows],
            ik: vec![0.0; rows],
        };
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
        let chunk = rows.div_ceil(threads);
        let failed = std::thread::scope(|sc| {
            let parts = p
                .pred
                .chunks_mut(chunk)
                .zip(p.qband.chunks_mut(chunk))
                .zip(p.ours.chunks_mut(chunk))
                .zip(p.ik.chunks_mut(chunk))
                .enumerate();
            let handles: Vec<_> = parts
                .map(|(ci, (((pred, qband), ours), ik))| {
                    sc.spawn(move || -> Result<(), String> {
                        let mut row = vec![0.0f32; k];
                        let outs = pred.iter_mut().zip(qband).zip(ours).zip(ik);
                        for (j, (((pr, qb), o), i)) in outs.enumerate() {
                            let r = ci * chunk + j;
                            dequant_row(GgmlType::Q6_K, &w[r * rb..(r + 1) * rb], &mut row)
                                .map_err(|e| format!("output.weight row {r}: {e}"))?;
                            let (mut dp, mut sq, mut so, mut si) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                            for (((&wv, &a), &b), &ej) in row.iter().zip(xo).zip(xd).zip(e) {
                                let wv = f64::from(wv);
                                dp += wv * (a - b);
                                sq += wv.abs() * ej;
                                so += (wv * a).abs();
                                si += (wv * b).abs();
                            }
                            (*pr, *qb, *o, *i) = (dp, sq, so, si);
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .filter_map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("a thread panicked".into()))
                        .err()
                })
                .next()
        });
        match failed {
            Some(e) => Err(e.into()),
            None => Ok(p),
        }
    }

    /// The index of the largest value, the lowest on ties — the kernel's
    /// rule — with the largest and the runner-up values.
    fn top2(v: &[f32]) -> (usize, f32, f32) {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        let second = v
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != best)
            .fold(f32::NEG_INFINITY, |a, (_, &x)| a.max(x));
        (best, v[best], second)
    }

    /// The head end of `set`: `hc_out`, the logits, and the argmax.
    ///
    /// The logits: each side's value is its exact projection of its own
    /// activations plus its f32 rounding. The exact values' difference
    /// `pred = Σ_j w_j·(x̂o_j − x̂d_j)` is computed here from both sides'
    /// codes; with `x̂o − x̂d = (x̂o − xo) + (xo − xd) + (xd − x̂d)` it lies
    /// within `qband = Σ_j |w_j|·(|x̂o_j − xo_j| + |xd_j − x̂d_j| + (scale_rel
    /// + 4u)·|xd_j|)`, the last term the two norms' distance on the same
    /// input (`hc_out` bit for bit). The pin: `|pred| <= qband` and the gap
    /// minus `pred` within both sides' rounding.
    fn check_head(set: &Set, split: &Split, inputs: &Inputs, o: &Outs) -> Result<bool, GateError> {
        let man = &set.man;
        let n = ROW as u64;
        let load = |name: &str, ne: [u64; 4], op: &str| -> Result<Vec<f32>, GateError> {
            let r = man.tensor(name, 0)?;
            r.expect(name, "f32", ne, op)?;
            ref_tensor_of_in(&man.dir, r)
        };
        let hc_out = load("hc_out", [n, 1, 1, 1], "MUL_MULTI_ADD")?;
        let norm = load("result_norm", [n, 1, 1, 1], "in")?;
        let vocab = o.logits.len() as u64;
        let logits = load("result_output", [vocab, 1, 1, 1], "MUL_MAT")?;
        let hc_bit = bits_equal(&o.hc_out, &hc_out);
        // The head's input is the fold of `inputs.head_streams` by
        // `inputs.head_pre`; its norm's distance prints as a diagnostic.
        let pre = &inputs.head_pre[..HC_STREAMS];
        let norm_rel = rel(&o.normed, &norm);

        let name = names::output();
        let (w, ty) = file_bytes(split, &name)?;
        if ty != GgmlType::Q6_K {
            return Err(format!("{name} is {ty}, want q6_K").into());
        }
        let k = o.normed.len();
        let mut xo = vec![0.0f64; k];
        for (xb, ob) in o
            .normed
            .as_chunks::<128>()
            .0
            .iter()
            .zip(xo.as_chunks_mut::<128>().0)
        {
            let (q, d) = q8_block(xb);
            for (x, &qi) in ob.iter_mut().zip(&q) {
                *x = f64::from(d) * f64::from(qi);
            }
        }
        let (qd, dd) = ik_q8_2::quantize(&norm);
        let xd: Vec<f64> = qd
            .iter()
            .enumerate()
            .map(|(j, &q)| f64::from(dd[j / QK]) * f64::from(q))
            .collect();
        let norm_band = scale_rel() + 4.0 * U;
        let e: Vec<f64> = (0..k)
            .map(|j| {
                let (x_o, x_d) = (f64::from(o.normed[j]), f64::from(norm[j]));
                (xo[j] - x_o).abs() + (x_d - xd[j]).abs() + norm_band * x_d.abs()
            })
            .collect();
        let p = predict_logits(w, k, &xo, &xd, &e)?;
        if p.pred.len() != o.logits.len() {
            return Err(format!(
                "{name} has {} rows, the head wrote {} logits",
                p.pred.len(),
                o.logits.len()
            )
            .into());
        }
        // Each side's f32 sum of k/16 sub-block terms in any order rounds at
        // most k/16 − 1 times, and each term at most three times on its way
        // in (the scales' two products, the product with the integer sum);
        // one more for the margin. The f64 prediction adds `2k·2^-53` of the
        // magnitudes.
        let sum_rel = (k / 16 + 4) as f64 * U;
        let (mut worst, mut worst_at, mut qworst) = (0.0f64, 0usize, 0.0f64);
        let (mut pred_max, mut gap_max) = (0.0f64, 0.0f64);
        for (i, ((&yo, &yd), (((&pr, &qb), &ao), &ad))) in o
            .logits
            .iter()
            .zip(&logits)
            .zip(p.pred.iter().zip(&p.qband).zip(&p.ours).zip(&p.ik))
            .enumerate()
        {
            let slack = 2.0 * k as f64 * U64 * (ao + ad);
            let gap = f64::from(yo) - f64::from(yd);
            let r = ratio((gap - pr).abs(), sum_rel * (ao + ad) + slack);
            if r > worst {
                (worst, worst_at) = (r, i);
            }
            qworst = qworst.max(ratio(pr.abs(), qb + slack));
            pred_max = pred_max.max(pr.abs());
            gap_max = gap_max.max(gap.abs());
        }
        let logits_ok = worst <= 1.0 && qworst <= 1.0;
        let logits_rel = rel(&o.logits, &logits);
        let (ours_top, ours_1, ours_2) = top2(&o.logits);
        let (ik_top, ik_1, ik_2) = top2(&logits);
        let top_ok = o.token as usize == ik_top && ours_top == ik_top;
        let pass = hc_bit && logits_ok && top_ok;
        println!(
            "head set={}: hc_out_bit={hc_bit} logits max|gap−pred|/bound={worst:.3e} (at {worst_at}) \
             max|pred|/qband={qworst:.3e} top1 ours={} host={ours_top} set={ik_top} margin \
             ours={:.4} set={:.4} — diag \
             pre={pre:?} result_norm rel={norm_rel:.3e} logits rel={logits_rel:.3e} \
             max|gap|={gap_max:.3e} max|pred|={pred_max:.3e} {}",
            set.name,
            o.token,
            ours_1 - ours_2,
            ik_1 - ik_2,
            verdict(pass)
        );
        Ok(pass)
    }
}
