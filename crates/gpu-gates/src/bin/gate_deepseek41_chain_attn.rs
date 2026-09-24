//! GPU gate for the V4.1 attention piece (`bloomery_gpu_deepseek41::chain::attn`,
//! B5 phase 1) against ik's CPU dump: G1 of the B5 plan cut to the piece
//! (`docs/research/v41-b5-plan-report.md` §4.1). It pins the composition;
//! the B4 op gates pin each op against ik.
//!
//! Sets: `step4` (position 4, where no ratio-2 group completes) and `d1n`
//! (position 301, the file's top-k, where no layer selects). For every layer
//! of both, the gate injects the piece's inputs from the set — the streams
//! and the folded input at the attention's start, the window ring (the last
//! window of ik's raw cache, slot `cell % window`), the compressed rows the
//! layer attends, and on a compressor's layer that compressor's ring —
//! uploads the step image `params.rs` builds from the host plan at the set's
//! position, runs the piece, and pins:
//!
//! - **words**, per set: the words the piece's gather left — the window
//!   length, each stream's visible count and `CompGeom` words
//!   (`CompGeom::pack` of the plan), the rope tables — are the plan's;
//! - **streams and fold**: every value of the new streams and of the ffn's
//!   fold within [`Z`] deviations of ik's, the deviation derived per value
//!   (below), after a rounding floor;
//! - **ring**: slot `pos % window` holds the step's latent row within `Z`
//!   deviations of the row ik wrote, every other slot bit-identical to the
//!   injection;
//! - **compressor** (its layer): where the step completes a group, the
//!   plan's row of the compressed cache and of the index-key cache within
//!   `Z` deviations of ik's; the ring's kept slot within `Z` deviations of
//!   ik's kept projection; every other row and slot bit-identical to the
//!   injection;
//! - **structure**: the layer's launches captured into a graph, replayed at
//!   both sets' inputs (the image and every buffer rewritten between
//!   replays) bit-identical to the eager runs, and the graph's node count the
//!   layer kind's launches plus the step's gather;
//! - **rule**, on a layer whose projections are Q3_K: ik's q8_K × Q3_K dot
//!   (`act_rule`) on ik's own inputs gives each projection's output in the
//!   dump bit for bit — the codes the deviation reads ik's rounding from.
//!
//! The five projections (q_a, q_b, kv, wo_a, wo_b) are Q8_0 or Q3_K — the
//! file's format picks each side's rule; any other format is refused at load
//! with the tensor named.
//!
//! The deviation. The piece's ops and ik's differ by rule, not by
//! composition: ik quantizes the activations of every quantized matmul
//! (q8_2 blocks of 32 under Q8_0 weights, q8_K blocks of 256 under q3_K),
//! ours read f32 under Q8_0 and q8_1 blocks of 128 under q3_K, and the
//! attention's own rule differs by the B4 attention gate's band. Those are
//! the rule differences the B4 gates pinned, in the form that composes: the
//! rounding error of every quantized input — exact for ik's q8_2
//! (`ik_q8_2`, the sign fold's wrap included) and for ik's q8_K at a Q3_K
//! projection (`act_rule::Rule::input_var`, read from ik's own input), `d²/12`
//! a value for our q8_1 and for the q8_K of HC_PRE and the compressor, whose
//! inputs differ between the sides — taken as independent across values and carried to
//! the outputs to first order: exactly through every linear map (`Σ W²·var`),
//! through a norm by its gain over its RMS, through a rope as the rotation
//! of the variances, through the attention by its Jacobian at ik's queries,
//! keys and sinks, and through HC_PRE by the Jacobian of its rule
//! (transcribed in f64) at ik's mixes, under the mixes' covariance. The
//! B4 gates' worst-case bands, carried through five contractions, would
//! allow the size of the values themselves; a composition fault — a wrong
//! buffer, table, count, stream or launch — moves values by their own size,
//! hundreds of deviations.
//!
//! Diagnostics, printed only: per layer, the node-by-node distances of the
//! piece's intermediate buffers to ik's nodes, and the sub-layer output's
//! predicted and measured spread.
//!
//! FAIL-first: `BLOOMERY_CHAIN_ATTN_MUTATE=<name>` breaks one thing a pin
//! guards — `swap-fold` (the body reads stream 0 as its fold), `wrong-rows`
//! (a reading layer gets its rows shifted by one), `zero-vis` (every
//! stream's visible count in the image reads 0), `slot-shift` (the image's
//! position reads one on), `no-group` (every stream's group count reads 0),
//! `stale-image` (a replay runs on the other set's image), `extra-launch`
//! (the capture records the gather twice). `BLOOMERY_CHAIN_ATTN_LAYERS=a,b,…`
//! runs those layers only.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_chain_attn: built without the `deepseek41` feature; see `just gate-gpu-ds41-chain-attn`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_chain_attn", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeSet;

    use bloomery_gpu::weights::Weights;
    use bloomery_gpu::{DeviceTensor, Gpu};
    use bloomery_gpu_deepseek41::body::STEP_TOKENS;
    use bloomery_gpu_deepseek41::chain::attn::{
        AttnChain, AttnIo, AttnTaps, Compressed, Selection, SourceIo, WordsLayout,
    };
    use bloomery_gpu_deepseek41::compress::StepInts;
    use bloomery_gpu_deepseek41::hc::HC_MIX;
    use bloomery_gpu_deepseek41::index_key::HT_SCALE;
    use bloomery_gpu_deepseek41::params::{ImageDims, ImageLayout, StepImage, Table, rope_specs};
    use bloomery_gpu_gates::act_rule::{self, Q3K_BYTES, QK_K, Rule, q8_var};
    use bloomery_gpu_gates::ik_q8_2::{self, QK};
    use bloomery_gpu_gates::oracle::deepseek41::{D1N, STEP4};
    use bloomery_gpu_gates::oracle::for_arch;
    use bloomery_gpu_gates::{
        GateError, RefManifest, RefRow, checks_failed, max_rel_err, ref_model_path,
        ref_tensor_logical_in, ref_tensor_of_in, split_f32, verdict, widened_f16_rows_in,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::arch::deepseek41::plan::{Planner, StepPlan};

    /// f32's unit roundoff.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// The pin, in derived deviations. Over one layer's 20,480 stream values
    /// a normal spread passes 5.9 deviations once in a thousand layers; the
    /// factor 1.5 on top is the model's slack — first order, independent
    /// roundings, the attention band read as a spread.
    const Z: f64 = 9.0;
    /// f16 NaN: what the gate fills every cache row and slot it does not
    /// inject, so a read of one shows.
    const NAN16: u16 = 0x7e00;
    /// The B4 attention gate's bands (`gate_deepseek41_attn`'s `IK_BAND`,
    /// `IQK_BAND`): our attention against ik's on the same queries and keys,
    /// the largest distance over a set relative to its largest value, on
    /// ik's generic path and on its iqk path.
    const ATTN_BAND_GENERIC: f64 = 4.9e-3;
    const ATTN_BAND_IQK: f64 = 9.675e-4;
    /// A band over the attention's 32,768 values read as a spread: their
    /// largest sits near this many deviations of a normal spread.
    const BAND_SIGMAS: f64 = 4.5;
    /// Host threads for the deviation's sums.
    const THREADS: usize = 16;
    /// The port's names of the compressed streams, in the planner's order.
    const STREAMS: [&str; 2] = ["csa", "hca"];

    // ------------------------------------------------------------ mutations

    /// One thing a pin guards, broken on purpose (FAIL-first).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Mutation {
        SwapFold,
        WrongRows,
        ZeroVis,
        SlotShift,
        NoGroup,
        StaleImage,
        ExtraLaunch,
    }

    impl Mutation {
        const ALL: [(&'static str, Mutation); 7] = [
            ("swap-fold", Mutation::SwapFold),
            ("wrong-rows", Mutation::WrongRows),
            ("zero-vis", Mutation::ZeroVis),
            ("slot-shift", Mutation::SlotShift),
            ("no-group", Mutation::NoGroup),
            ("stale-image", Mutation::StaleImage),
            ("extra-launch", Mutation::ExtraLaunch),
        ];

        fn from_env() -> Result<Option<Mutation>, GateError> {
            let Ok(v) = std::env::var("BLOOMERY_CHAIN_ATTN_MUTATE") else {
                return Ok(None);
            };
            Mutation::ALL
                .iter()
                .find(|(n, _)| *n == v)
                .map(|&(_, m)| Some(m))
                .ok_or_else(|| {
                    let known: Vec<_> = Mutation::ALL.iter().map(|(n, _)| *n).collect();
                    format!("BLOOMERY_CHAIN_ATTN_MUTATE={v}: not one of {known:?}").into()
                })
        }
    }

    /// The layers `BLOOMERY_CHAIN_ATTN_LAYERS` names, or all of them.
    fn layer_filter(n_layer: usize) -> Result<Vec<usize>, GateError> {
        let Ok(v) = std::env::var("BLOOMERY_CHAIN_ATTN_LAYERS") else {
            return Ok((0..n_layer).collect());
        };
        let set = v
            .split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<BTreeSet<_>, _>>()
            .map_err(|e| format!("BLOOMERY_CHAIN_ATTN_LAYERS={v}: {e}"))?;
        if set.iter().any(|&l| l >= n_layer) {
            return Err(format!("BLOOMERY_CHAIN_ATTN_LAYERS={v}: the file has {n_layer}").into());
        }
        Ok(set.into_iter().collect())
    }

    // ---------------------------------------------------------- layer kinds

    /// A layer as the gate reads it: the plan's stream, the file's
    /// compressor, whether it runs the indexer, and which of its projections
    /// read a q8_1 input.
    #[derive(Clone, Copy, Debug)]
    struct Kind {
        stream: Option<usize>,
        /// It owns its stream's compressor: whether that pools with scores
        /// (above ratio 1), whether it owns index keys.
        source: Option<(bool, bool)>,
        indexer: bool,
        quant: Quant,
    }

    /// Where a layer's projections read q8_1 (a K-quant weight,
    /// `Dense::reads_q8_1`): q_a or kv (the norm leaves the normed input's
    /// q8_1 form, which an indexer without a compressor then reuses), wo_a
    /// (its groups' input quantized first) and wo_b (wo_a's output quantized
    /// first).
    #[derive(Clone, Copy, Debug, Default)]
    struct Quant {
        normed: bool,
        heads: bool,
        wo_a: bool,
    }

    impl Quant {
        fn of(split: &Split, l: usize) -> Result<Quant, GateError> {
            let q8_1 = |name: String| -> Result<bool, GateError> {
                let (_, t) = split
                    .find(&name)
                    .ok_or_else(|| format!("{name} is not in the model file"))?;
                Ok(matches!(t.ty, GgmlType::Q3_K | GgmlType::Q4_K))
            };
            Ok(Quant {
                normed: q8_1(names::attn_q_a(l))? || q8_1(names::attn_kv(l))?,
                heads: q8_1(names::attn_output_a(l))?,
                wo_a: q8_1(names::attn_output_b(l))?,
            })
        }
    }

    impl Kind {
        fn of(hp: &Hparams, planner: &Planner, split: &Split, l: usize) -> Result<Kind, GateError> {
            let kind = &hp.layers[l];
            Ok(Kind {
                stream: planner.layer_stream(l),
                source: kind.compressor.map(|c| (c.gated, kind.index_keys)),
                indexer: kind.indexer,
                quant: Quant::of(split, l)?,
            })
        }

        fn name(&self) -> String {
            match (self.stream, self.source) {
                (None, _) => "window".to_string(),
                (Some(s), None) => format!("{}-reader", STREAMS[s]),
                (Some(s), Some(_)) => format!("{}-source", STREAMS[s]),
            }
        }

        /// The piece's launches for a layer of this kind (the module doc of
        /// `chain::attn`): fourteen on every layer — HC_PRE, the norm, q_a,
        /// its norm, q_b, the q rope, kv, the K/V append, the attention's two,
        /// the inverse rope, wo_a, wo_b, HC_POST — plus the q8_1 of wo_a's
        /// input and of wo_b's where those read one (a norm that leaves a
        /// q8_1 form stays one launch); on a compressor's layer its
        /// projections, the row launch and the index key's three, and on an
        /// indexer layer its two projections, the score and top-k passes and,
        /// without a compressor and without the norm's q8_1 form, the q8_1 of
        /// the normed input.
        fn launches(&self) -> usize {
            14 + usize::from(self.quant.heads)
                + usize::from(self.quant.wo_a)
                + self.source.map_or(0, |(gated, keys)| {
                    1 + usize::from(gated) + 1 + 3 * usize::from(keys)
                })
                + if self.indexer {
                    4 + usize::from(self.source.is_none() && !self.quant.normed)
                } else {
                    0
                }
        }
    }

    // --------------------------------------------------------- host weights

    /// A Q8_0 weight's blocks, row-major.
    struct HostQ8 {
        blocks: Vec<Q8Block>,
        k: usize,
        rows: usize,
    }

    fn host_q8(split: &Split, name: &str) -> Result<HostQ8, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        let &[k, rows] = t.dims.as_slice() else {
            return Err(format!("{name} has dims {:?}, want [K, rows]", t.dims).into());
        };
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?);
        if t.ty != GgmlType::Q8_0 || !k.is_multiple_of(QK) {
            return Err(format!("{name} is {:?} K={k}, want Q8_0", t.ty).into());
        }
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        let blocks: Vec<Q8Block> = bytes
            .as_chunks::<34>()
            .0
            .iter()
            .map(Q8Block::from_bytes)
            .collect();
        if blocks.len() != rows * k / QK {
            return Err(format!("{name}: {} blocks, want {}", blocks.len(), rows * k / QK).into());
        }
        Ok(HostQ8 { blocks, k, rows })
    }

    /// A Q3_K weight's raw rows, for ik's dot and, dequantized row by row,
    /// the deviation.
    struct HostQ3k {
        bytes: Vec<u8>,
        k: usize,
        rows: usize,
    }

    impl HostQ3k {
        fn row(&self, r: usize) -> &[u8] {
            let n = self.k / QK_K * Q3K_BYTES;
            &self.bytes[r * n..(r + 1) * n]
        }
    }

    /// A projection's weight in the file's format: the rule of each side
    /// follows from it.
    enum HostW {
        Q8(HostQ8),
        Q3k(HostQ3k),
    }

    impl HostW {
        fn rows(&self) -> usize {
            match self {
                HostW::Q8(w) => w.rows,
                HostW::Q3k(w) => w.rows,
            }
        }
    }

    /// Projection weight `name`: Q8_0 or Q3_K, the two formats whose rules
    /// the gate simulates; any other is refused with the tensor named.
    fn host_w(split: &Split, name: &str) -> Result<HostW, GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        match t.ty {
            GgmlType::Q8_0 => return Ok(HostW::Q8(host_q8(split, name)?)),
            GgmlType::Q3_K => {}
            ty => {
                return Err(format!(
                    "{name} is {ty:?} {:?}, want Q8_0 or Q3_K: the gate has no rule for it",
                    t.dims
                )
                .into());
            }
        }
        let &[k, rows] = t.dims.as_slice() else {
            return Err(format!("{name} has dims {:?}, want [K, rows]", t.dims).into());
        };
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?);
        if !k.is_multiple_of(QK_K) {
            return Err(format!("{name} is Q3_K K={k}, not whole super-blocks").into());
        }
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        if bytes.len() != rows * k / QK_K * Q3K_BYTES {
            return Err(format!("{name}: {} bytes for {rows} rows of {k}", bytes.len()).into());
        }
        Ok(HostW::Q3k(HostQ3k {
            bytes: bytes.to_vec(),
            k,
            rows,
        }))
    }

    /// A q3_K weight dequantized, row-major, with its row width.
    fn host_q3k(split: &Split, name: &str) -> Result<(Vec<f32>, usize), GateError> {
        let (s, t) = split
            .find(name)
            .ok_or_else(|| format!("{name} is not in the model file"))?;
        let &[k, rows] = t.dims.as_slice() else {
            return Err(format!("{name} has dims {:?}, want [K, rows]", t.dims).into());
        };
        let (k, rows) = (usize::try_from(k)?, usize::try_from(rows)?);
        if t.ty != GgmlType::Q3_K || !k.is_multiple_of(256) {
            return Err(format!("{name} is {:?} K={k}, want q3_K", t.ty).into());
        }
        let bytes = split.shard(s).ok_or("shard index out of range")?.data(t)?;
        let row_bytes = k / 256 * 110;
        let mut w = vec![0.0f32; rows * k];
        for (r, out) in w.chunks_mut(k).enumerate() {
            dequant_row(
                GgmlType::Q3_K,
                &bytes[r * row_bytes..(r + 1) * row_bytes],
                out,
            )?;
        }
        Ok((w, k))
    }

    /// One layer's weights on the host, for the deviation.
    struct HostWeights {
        q_a: HostW,
        q_b: HostW,
        kv: HostW,
        out_a: HostW,
        out_b: HostW,
        q_a_norm: Vec<f32>,
        kv_norm: Vec<f32>,
        sinks: Vec<f32>,
        hc_fn: Vec<f32>,
        hc_scale: Vec<f32>,
        hc_base: Vec<f32>,
        comp: Option<HostComp>,
    }

    /// A compressor's weights on the host.
    struct HostComp {
        kv: Vec<f32>,
        gate: Option<Vec<f32>>,
        norm: Vec<f32>,
        /// The index key's projection and norm.
        key: Option<(Vec<f32>, Vec<f32>)>,
    }

    impl HostWeights {
        fn load(split: &Split, hp: &Hparams, l: usize, kind: &Kind) -> Result<Self, GateError> {
            let comp = match kind.source {
                None => None,
                Some((gated, keys)) => Some(HostComp {
                    kv: host_q3k(split, &names::attn_compressor_kv(l))?.0,
                    gate: gated
                        .then(|| host_q3k(split, &names::attn_compressor_gate(l)).map(|w| w.0))
                        .transpose()?,
                    norm: split_f32(split, &names::attn_compressor_norm(l), hp.head_dim)?,
                    key: keys
                        .then(|| -> Result<_, GateError> {
                            Ok((
                                host_q3k(split, &names::indexer_attn_k(l))?.0,
                                split_f32(split, &names::indexer_k_norm(l), hp.indexer.head_dim)?,
                            ))
                        })
                        .transpose()?,
                }),
            };
            Ok(HostWeights {
                q_a: host_w(split, &names::attn_q_a(l))?,
                q_b: host_w(split, &names::attn_q_b(l))?,
                kv: host_w(split, &names::attn_kv(l))?,
                out_a: host_w(split, &names::attn_output_a(l))?,
                out_b: host_w(split, &names::attn_output_b(l))?,
                q_a_norm: split_f32(split, &names::attn_q_a_norm(l), hp.q_lora_rank)?,
                kv_norm: split_f32(split, &names::attn_kv_a_norm(l), hp.head_dim)?,
                sinks: split_f32(split, &names::attn_sinks(l), hp.n_head)?,
                hc_fn: host_q3k(split, &names::hc_attn_fn(l))?.0,
                hc_scale: split_f32(split, &names::hc_attn_scale(l), 3)?,
                hc_base: split_f32(split, &names::hc_attn_base(l), HC_MIX)?,
                comp,
            })
        }
    }

    /// The names of layer `l`'s attention tensors: what the gate makes
    /// resident for it.
    fn resident_names(l: usize, kind: &Kind) -> BTreeSet<String> {
        let mut set: BTreeSet<String> = [
            names::hc_attn_fn(l),
            names::hc_attn_scale(l),
            names::hc_attn_base(l),
            names::attn_norm(l),
            names::attn_q_a(l),
            names::attn_q_a_norm(l),
            names::attn_q_b(l),
            names::attn_kv(l),
            names::attn_kv_a_norm(l),
            names::attn_sinks(l),
            names::attn_output_a(l),
            names::attn_output_b(l),
        ]
        .into_iter()
        .collect();
        if let Some((gated, keys)) = kind.source {
            set.insert(names::attn_compressor_kv(l));
            set.insert(names::attn_compressor_norm(l));
            if gated {
                set.insert(names::attn_compressor_gate(l));
            }
            if keys {
                set.insert(names::indexer_attn_k(l));
                set.insert(names::indexer_k_norm(l));
            }
        }
        if kind.indexer {
            set.insert(names::indexer_attn_q_b(l));
            set.insert(names::indexer_proj(l));
        }
        set
    }

    /// Bytes of one engram table row: `body.rs`'s rule, which is private
    /// there — every site's rows the same size.
    fn engram_row_bytes(split: &Split, hp: &Hparams) -> Result<usize, GateError> {
        let mut bytes = None;
        for &l in &hp.engram.layer_ids {
            let name = names::engram_embd(l);
            let (_, t) = split
                .find(&name)
                .ok_or_else(|| format!("{name} is not in the file"))?;
            let rows: u64 = t.dims.iter().skip(1).product();
            if rows == 0 || !t.nbytes.is_multiple_of(rows) {
                return Err(format!("{name}: {:?} dims, {} bytes", t.dims, t.nbytes).into());
            }
            let row = t.nbytes / rows;
            if bytes.is_some_and(|b| b != row) {
                return Err(
                    format!("{name} has rows of {row} bytes, another site's differ").into(),
                );
            }
            bytes = Some(row);
        }
        Ok(usize::try_from(
            bytes.ok_or("the model has no engram site")?,
        )?)
    }

    // ----------------------------------------------------------------- sets

    /// One set's step: its plan, the image built from it, and the words the
    /// gate uploads (the image, or a mutation of it).
    struct SetRun {
        label: &'static str,
        man: RefManifest,
        plan: StepPlan,
        image: StepImage,
        upload: Vec<u32>,
    }

    impl SetRun {
        fn pos(&self) -> usize {
            self.plan.pos[0] as usize
        }

        /// The window the step's token sees: `min(pos + 1, window)` rows.
        fn window_len(&self) -> usize {
            (self.plan.pos[0] - self.plan.raw_first[0] + 1) as usize
        }

        /// The image's rope table `t` at the step's token, as f32.
        fn table(&self, t: Table) -> Result<Vec<f32>, GateError> {
            let view = self.image.layout().view(self.image.words())?;
            Ok(view
                .table(0, t)
                .iter()
                .map(|&w| f32::from_bits(w))
                .collect())
        }

        /// The compressor's rope table of stream `s`: its row table above
        /// ratio 1, the token's YaRN table at ratio 1.
        fn row_table(&self, s: usize) -> Result<Vec<f32>, GateError> {
            let view = self.image.layout().view(self.image.words())?;
            Ok(match view.row_table(s, 0) {
                Some(t) => t.iter().map(|&w| f32::from_bits(w)).collect(),
                None => self.table(Table::YarnForward)?,
            })
        }
    }

    /// The image's words, `mutation`'s change applied: the offsets come from
    /// an image whose every word holds its own index.
    fn mutate_image(
        layout: &ImageLayout,
        words: &[u32],
        mutation: Option<Mutation>,
    ) -> Result<Vec<u32>, GateError> {
        let mut out = words.to_vec();
        let probe: Vec<u32> = (0..u32::try_from(layout.words())?).collect();
        let at = layout.view(&probe)?;
        match mutation {
            Some(Mutation::SlotShift) => out[at.token(0).pos as usize] += 1,
            Some(Mutation::ZeroVis) => {
                for s in 0..layout.streams().len() {
                    out[at.stream(s).n_visible[0] as usize] = 0;
                }
            }
            Some(Mutation::NoGroup) => {
                for s in 0..layout.streams().len() {
                    out[at.stream(s).groups as usize] = 0;
                }
            }
            _ => {}
        }
        Ok(out)
    }

    /// The words the piece's gather must leave for `set`, at `words`'
    /// offsets, as u32: the plan's counts, `CompGeom::pack` of its streams,
    /// the image's tables. `None` where the copy holds nothing.
    fn expected_words(words: &WordsLayout, set: &SetRun) -> Result<Vec<Option<u32>>, GateError> {
        let mut want = vec![None; words.len];
        let view = set.image.layout().view(set.image.words())?;
        let len = u32::try_from(set.window_len())?;
        want[words.pos] = Some(set.plan.pos[0]);
        want[words.window_vis] = Some(len);
        want[words.window_vis + 1] = Some(0);
        for (&at, t) in words.tables.iter().zip(Table::ALL) {
            for (i, &w) in view.table(0, t).iter().enumerate() {
                want[at + i] = Some(w);
            }
        }
        for (s, (sw, st)) in words.streams.iter().zip(&set.plan.streams).enumerate() {
            want[sw.vis] = Some(len);
            want[sw.vis + 1] = Some(st.n_visible[0]);
            let mut packed = vec![0u32; sw.geom.words()];
            sw.geom.pack(
                &StepInts {
                    write_row: &st.state_write,
                    read: &st.state_read,
                    persist_src: &st.persist_src,
                    persist_dst: &st.persist_dst,
                },
                &mut packed,
            )?;
            for (i, &w) in packed.iter().enumerate() {
                want[sw.step + i] = Some(w);
            }
            if sw.geom.ratio > 1 {
                let t = view
                    .row_table(s, 0)
                    .ok_or_else(|| format!("stream {s} has no row table"))?;
                for (i, &w) in t.iter().enumerate() {
                    want[sw.cs + i] = Some(w);
                }
            }
        }
        Ok(want)
    }

    // ------------------------------------------------------------- dump rows

    /// Tensor row `name`/0, made by `op`, with its position.
    fn node<'m>(
        man: &'m RefManifest,
        name: &str,
        op: &str,
    ) -> Result<(usize, &'m RefRow), GateError> {
        let (at, r) = man.tensor_at(name, 0)?;
        if r.op != op {
            return Err(format!("{}: {name} is {}, want {op}", man.dir.display(), r.op).into());
        }
        Ok((at, r))
    }

    /// The first tensor row made by `op` whose `src0` is `src0` (a weight's
    /// name: ik names some projections after the stream, not the weight).
    fn by_src0<'m>(man: &'m RefManifest, op: &str, src0: &str) -> Option<(usize, &'m RefRow)> {
        man.tensors
            .iter()
            .enumerate()
            .find(|(_, r)| r.op == op && r.src0.as_deref() == Some(src0))
    }

    fn f32s(man: &RefManifest, row: &RefRow) -> Result<Vec<f32>, GateError> {
        ref_tensor_logical_in(&man.dir, row)
    }

    fn f16s(man: &RefManifest, row: &RefRow) -> Result<Vec<u16>, GateError> {
        widened_f16_rows_in(&man.dir, row)
    }

    fn widen(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&h| half_to_f32(h)).collect()
    }

    /// One layer of one set: what the gate injects, what it compares with,
    /// and the points the deviation is taken at.
    struct Case {
        // Injected.
        s_in: Vec<f32>,
        x_in: Vec<f32>,
        ring: Vec<u16>,
        rows: Option<Vec<u16>>,
        keys: Option<Vec<u16>>,
        comp_ring: Option<(Vec<f32>, Vec<f32>)>,
        /// The compressed rows the step's token sees: the identity list's
        /// length.
        n_vis: usize,
        // Compared with.
        s_ref: Vec<f32>,
        x_ref: Vec<f32>,
        /// The row ik wrote at the step's cell.
        ring_row: Vec<u16>,
        /// The group the step completes: its row, ik's compressed row and
        /// index key.
        written: Option<(usize, Vec<u16>, Option<Vec<u16>>)>,
        /// ik's compressor ring after the step, and the slots it kept.
        comp_ring_ref: Option<(Vec<f32>, Vec<f32>, Vec<u32>)>,
        // ik's nodes.
        hc: Vec<f32>,
        mixes: Vec<f32>,
        hc_normed: Vec<f32>,
        attn_norm: Vec<f32>,
        qr: Vec<f32>,
        qr_norm: Vec<f32>,
        /// q_b's output, where the set has it: what ik's rule on a Q3_K
        /// q_b is checked against.
        q_b: Option<Vec<f32>>,
        q_rope: Vec<f32>,
        kv_b: Vec<f32>,
        kv_rope: Vec<f32>,
        fattn: Vec<f32>,
        attn: Vec<f32>,
        wo_a: Vec<f32>,
        out: Vec<f32>,
        /// The keys ik's attention attended, the window's cells then the
        /// stream's rows, as f32.
        keys_lin: Vec<f32>,
        window_len: usize,
        iqk: bool,
        /// The layer's rope tables at the step's token and, on a
        /// compressor's layer, the compressor's.
        fwd: Vec<f32>,
        back: Vec<f32>,
        comp: Option<CompCase>,
    }

    /// A compressor's nodes in one set.
    struct CompCase {
        /// The pooled group's sources (ring slots, then this step's), with
        /// their scores, and the pooled row before and after its norm.
        pooled: Option<Pooled>,
        cs: Vec<f32>,
    }

    struct Pooled {
        kv: Vec<Vec<f32>>,
        score: Vec<Vec<f32>>,
        /// Which of the sources are this step's projections.
        fresh: Vec<bool>,
        y: Vec<f32>,
        normed: Vec<f32>,
        /// The index key's projection and the key before its f16 rounding.
        key_proj: Option<Vec<f32>>,
    }

    /// The case of layer `l` in `set`.
    #[allow(
        clippy::too_many_lines,
        reason = "one dump walk: every row the case reads, in manifest order"
    )]
    fn load_case(
        set: &SetRun,
        hp: &Hparams,
        l: usize,
        kind: &Kind,
        mutation: Option<Mutation>,
    ) -> Result<Case, GateError> {
        let man = &set.man;
        let width = hp.head_dim;
        let name = |stem: &str| format!("{stem}-{l}");

        // HC_PRE and the streams it read.
        let hc_scale = names::hc_attn_scale(l);
        let (at_hc, hc_row) = man
            .tensors
            .iter()
            .enumerate()
            .find(|(_, r)| r.op == "HC_PRE" && r.src1.as_deref() == Some(hc_scale.as_str()))
            .ok_or_else(|| format!("no HC_PRE of layer {l}'s attention"))?;
        let (at_mix, mix_row) = man.last_before(at_hc, hc_row.src0.as_deref())?;
        let (at_hn, hn_row) = man.last_before(at_mix, mix_row.src1.as_deref())?;
        let (_, s_row) = man.last_before(at_hn, hn_row.src0.as_deref())?;
        let (at_an, an_row) = node(man, &name("attn_norm"), "FUSED_RMS_NORM")?;
        let (_, x_row) = man.last_before(at_an, an_row.src0.as_deref())?;
        let s_in = f32s(man, s_row)?;
        let mut x_in = f32s(man, x_row)?;
        if s_in.len() != 4 * hp.n_embd || x_in.len() != hp.n_embd {
            return Err(format!(
                "layer {l}: streams {} and fold {} values",
                s_in.len(),
                x_in.len()
            )
            .into());
        }
        if mutation == Some(Mutation::SwapFold) {
            x_in.copy_from_slice(&s_in[..hp.n_embd]);
        }

        // The window: ik's raw cache before and after the step's write.
        let before = f16s(man, man.input(&format!("cache_k_l{l}"), 0)?)?;
        let after = f16s(man, node(man, &name("dsv4_raw_k_write"), "SET_ROWS")?.1)?;
        let (pos, len) = (set.pos(), set.window_len());
        let ring_rows = hp.window.min(before.len() / width);
        let first = pos + 1 - len;
        let mut ring = vec![NAN16; ring_rows * width];
        for c in first..pos {
            let slot = c % ring_rows;
            ring[slot * width..(slot + 1) * width]
                .copy_from_slice(&before[c * width..(c + 1) * width]);
        }
        let ring_row = after[pos * width..(pos + 1) * width].to_vec();
        let mut keys_lin = widen(&after[first * width..(pos + 1) * width]);

        // The stream's rows: the ones ik attended (its view after the source
        // layer wrote), and the source layer's own before its write.
        let st = kind.stream.map(|s| &set.plan.streams[s]);
        let mut written = None;
        let rows = match (kind.stream, st) {
            (Some(s), Some(st)) => {
                let tag = STREAMS[s];
                let (_, view) = node(man, &format!("{tag}_k-{l}"), "VIEW")?;
                let attended = f16s(man, view)?;
                let n_vis = st.n_visible[0] as usize;
                keys_lin.extend(widen(&attended[..n_vis * width]));
                let mut rows = vec![NAN16; rows_of(set, st.ratio as usize) * width];
                match kind.source {
                    None => {
                        let shift = usize::from(mutation == Some(Mutation::WrongRows));
                        for r in 0..n_vis {
                            let from = (r + shift) * width;
                            if from + width <= attended.len().min(rows.len()) {
                                rows[r * width..(r + 1) * width]
                                    .copy_from_slice(&attended[from..from + width]);
                            }
                        }
                    }
                    Some((_, keys)) => {
                        let leaf = view.src0.as_deref().ok_or("the rows view has no src0")?;
                        let pre = f16s(man, man.input(leaf, 0)?)?;
                        let row = st.state_write.first().map(|&w| w as usize);
                        for r in (0..n_vis).filter(|&r| Some(r) != row) {
                            rows[r * width..(r + 1) * width]
                                .copy_from_slice(&pre[r * width..(r + 1) * width]);
                        }
                        if let Some(w) = row {
                            let (_, wr) = node(man, &format!("{tag}_k_write-{l}"), "SET_ROWS")?;
                            let ik_rows = f16s(man, wr)?;
                            let key = if keys {
                                let (_, kw) = node(man, &name("lid_k_write"), "SET_ROWS")?;
                                let ik_keys = f16s(man, kw)?;
                                let kd = hp.indexer.head_dim;
                                Some(ik_keys[w * kd..(w + 1) * kd].to_vec())
                            } else {
                                None
                            };
                            written = Some((w, ik_rows[w * width..(w + 1) * width].to_vec(), key));
                        }
                    }
                }
                Some(rows)
            }
            _ => None,
        };
        let keys = match kind.source {
            Some((_, true)) => {
                let s = kind.stream.ok_or("a compressor without a stream")?;
                let n = rows_of(set, set.plan.streams[s].ratio as usize);
                Some(vec![NAN16; n * hp.indexer.head_dim])
            }
            _ => None,
        };

        // The compressor: this step's projections, its ring, its group.
        let (comp, comp_ring, comp_ring_ref) = match (kind.source, kind.stream) {
            (Some((gated, has_keys)), Some(s)) => load_comp(
                set,
                hp,
                l,
                s,
                gated,
                has_keys,
                written.as_ref().map(|w| w.0),
            )?,
            _ => (None, None, None),
        };

        let (_, fattn) = node(man, &name("fattn"), "FLASH_ATTN_EXT")?;
        let (_, attn) = node(man, &name("attn"), "ROPE_BACK")?;
        let (_, wo_a) = node(man, &name("attn_wo_a"), "MUL_MAT")?;
        let (_, out) = node(man, &name("attn_out"), "MUL_MAT")?;
        let (fwd, back) = match kind.stream {
            Some(_) => (Table::YarnForward, Table::YarnBack),
            None => (Table::WindowForward, Table::WindowBack),
        };
        Ok(Case {
            s_in,
            x_in,
            ring,
            rows,
            keys,
            comp_ring,
            s_ref: f32s(man, node(man, &name("hc_attn_post"), "HC_POST")?.1)?,
            x_ref: f32s(man, node(man, &name("hc_ffn_pre"), "MUL_MULTI_ADD")?.1)?,
            ring_row,
            written,
            comp_ring_ref,
            hc: f32s(man, hc_row)?,
            mixes: f32s(man, mix_row)?,
            hc_normed: f32s(man, hn_row)?,
            attn_norm: f32s(man, an_row)?,
            qr: f32s(man, node(man, &name("qr"), "MUL_MAT")?.1)?,
            qr_norm: f32s(man, node(man, &name("qr_norm"), "FUSED_RMS_NORM")?.1)?,
            q_b: node(man, &name("q_b"), "MUL_MAT")
                .ok()
                .map(|(_, r)| f32s(man, r))
                .transpose()?,
            q_rope: f32s(man, node(man, &name("q_rope"), "ROPE")?.1)?,
            kv_b: f32s(man, node(man, &name("kv_b"), "MUL_MAT")?.1)?,
            kv_rope: f32s(man, node(man, &name("kv_rope"), "ROPE")?.1)?,
            fattn: f32s(man, fattn)?,
            attn: f32s(man, attn)?,
            wo_a: f32s(man, wo_a)?,
            out: f32s(man, out)?,
            keys_lin,
            window_len: len,
            n_vis: st.map_or(0, |st| st.n_visible[0] as usize),
            iqk: man.tensor(&name("mask_to_idx"), 0).is_ok(),
            fwd: set.table(fwd)?,
            back: set.table(back)?,
            comp,
        })
    }

    /// Rows of a stream's caches at the set's context: what the gate
    /// allocates.
    fn rows_of(set: &SetRun, ratio: usize) -> usize {
        (set.man.header.ctx.unwrap_or(0) as usize).div_ceil(ratio)
    }

    type CompParts = (
        Option<CompCase>,
        Option<(Vec<f32>, Vec<f32>)>,
        Option<(Vec<f32>, Vec<f32>, Vec<u32>)>,
    );

    /// A compressor layer's nodes: its projections, its ring before and
    /// after the step (above ratio 1), and the group it completes at `row`.
    fn load_comp(
        set: &SetRun,
        hp: &Hparams,
        l: usize,
        s: usize,
        gated: bool,
        has_keys: bool,
        row: Option<usize>,
    ) -> Result<CompParts, GateError> {
        let man = &set.man;
        let tag = STREAMS[s];
        let st = &set.plan.streams[s];
        let r = st.ratio as usize;
        let width = hp.head_dim;
        let (at_kv, kv_row) = by_src0(man, "MUL_MAT", &names::attn_compressor_kv(l))
            .ok_or_else(|| format!("layer {l}: no compressor kv projection"))?;
        let kv = f32s(man, kv_row)?;
        let score_node = gated
            .then(|| {
                by_src0(man, "MUL_MAT", &names::attn_compressor_gate(l))
                    .ok_or_else(|| format!("layer {l}: no compressor gate projection"))
            })
            .transpose()?;
        let score = score_node.map(|(_, row)| f32s(man, row)).transpose()?;

        // The ring above ratio 1: the f32 input of its shape the sources'
        // CONCAT touches first, or the persist when no group completes.
        let (ring, ring_ref) = match score_node {
            Some((at_sc, sc_row)) if r > 1 => {
                let ring_ne = [width as u64, r as u64, 1, 1];
                let (at_pk, pk) = node(man, &format!("{tag}_k_state_persist-{l}"), "SET_ROWS")?;
                let (at_ps, ps) = node(man, &format!("{tag}_score_state_persist-{l}"), "SET_ROWS")?;
                let first = |concat: &str, src1: &str, persist: usize| {
                    let by_concat = man.tensors.iter().position(|t| {
                        t.op == "CONCAT" && t.name == concat && t.src1.as_deref() == Some(src1)
                    });
                    by_concat
                        .and_then(|c| {
                            man.first_touched_by(c)
                                .into_iter()
                                .find(|i| i.ty == "f32" && i.ne == ring_ne)
                        })
                        .or_else(|| {
                            man.first_touched_by(persist)
                                .into_iter()
                                .find(|i| i.ty == "f32" && i.ne == ring_ne)
                        })
                };
                let _ = (at_kv, at_sc);
                let v = first(&format!("{tag}_source_kv"), &kv_row.name, at_pk)
                    .ok_or_else(|| format!("layer {l}: no ring of values"))?;
                let sc = first(&format!("{tag}_source_score"), &sc_row.name, at_ps)
                    .ok_or_else(|| format!("layer {l}: no ring of scores"))?;
                let ring = (
                    ref_tensor_of_in(&man.dir, v)?,
                    ref_tensor_of_in(&man.dir, sc)?,
                );
                let ring_ref = (f32s(man, pk)?, f32s(man, ps)?, st.persist_dst.clone());
                (Some(ring), Some(ring_ref))
            }
            _ => (None, None),
        };

        let pooled = match row {
            None => None,
            Some(_) => {
                let (_, y_row) = node(man, &format!("{tag}_state_compress-{l}"), "DS4_COMP")?;
                let normed = man.tensor(&format!("{tag}_state_compress-{l}"), 1)?;
                if normed.op != "FUSED_RMS_NORM" {
                    return Err(format!("layer {l}: the pooled row's norm is {}", normed.op).into());
                }
                let mut src_kv = Vec::with_capacity(r);
                let mut src_score = Vec::with_capacity(r);
                let mut fresh = Vec::with_capacity(r);
                for &read in &st.state_read[..r] {
                    let read = read as usize;
                    match (&ring, read < r) {
                        (Some((v, sc)), true) => {
                            src_kv.push(v[read * width..(read + 1) * width].to_vec());
                            src_score.push(sc[read * width..(read + 1) * width].to_vec());
                            fresh.push(false);
                        }
                        (_, false) => {
                            src_kv.push(kv.clone());
                            src_score.push(score.clone().unwrap_or_default());
                            fresh.push(true);
                        }
                        (None, true) => {
                            return Err(format!("layer {l}: a ring read at ratio {r}").into());
                        }
                    }
                }
                let key_proj = has_keys
                    .then(|| -> Result<_, GateError> {
                        let (_, row) = by_src0(man, "MUL_MAT", &names::indexer_attn_k(l))
                            .ok_or_else(|| format!("layer {l}: no index key projection"))?;
                        f32s(man, row)
                    })
                    .transpose()?;
                Some(Pooled {
                    kv: src_kv,
                    score: src_score,
                    fresh,
                    y: f32s(man, y_row)?,
                    normed: f32s(man, normed)?,
                    key_proj,
                })
            }
        };
        let cs = set.row_table(s)?;
        Ok((Some(CompCase { pooled, cs }), ring, ring_ref))
    }

    // ------------------------------------------------------------ deviation

    /// ik's q8_2 rounding error of each value of `x`: under a positive
    /// weight code, and under a negative one — they differ only where the
    /// sign fold wraps an activation code of −128.
    fn q8_2_err(x: &[f32]) -> Vec<[f64; 2]> {
        let (q, d) = ik_q8_2::quantize(x);
        x.iter()
            .enumerate()
            .map(|(c, &v)| {
                let dx = f64::from(d[c / QK]);
                let pos = f64::from(v) - f64::from(q[c]) * dx;
                let neg = f64::from(v) + f64::from(ik_q8_2::folded(-1, q[c])) * dx;
                [pos, neg]
            })
            .collect()
    }

    fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    /// `Σ_c W_rc² (var_c + err_c²)` per row of a Q8_0 weight; row `r` reads
    /// the input window `r / rows_per_window` (`w.k` values each).
    fn var_q8(w: &HostQ8, var: &[f64], err: &[[f64; 2]], rows_per_window: usize) -> Vec<f64> {
        let nb = w.k / QK;
        let mut out = vec![0.0f64; w.rows];
        let chunk = w.rows.div_ceil(THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let r = c * chunk + i;
                        let x0 = r / rows_per_window * w.k;
                        let mut acc = 0.0f64;
                        for (b, blk) in w.blocks[r * nb..(r + 1) * nb].iter().enumerate() {
                            let d = f64::from(half_to_f32(blk.d));
                            let mut sum = 0.0f64;
                            for (j, &q) in blk.q.iter().enumerate() {
                                let x = x0 + b * QK + j;
                                let e = err[x][usize::from(q < 0)];
                                sum += f64::from(q) * f64::from(q) * (var[x] + e * e);
                            }
                            acc += d * d * sum;
                        }
                        *o = acc;
                    }
                });
            }
        });
        out
    }

    /// `Σ_c W_rc² (var_c + in_c)` per row of a Q3_K weight, each row
    /// dequantized as `dequant_row` does it; row `r` reads the input window
    /// `r / rows_per_window` (`w.k` values each).
    fn var_q3k(w: &HostQ3k, var: &[f64], input: &[f64], rows_per_window: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; w.rows];
        let chunk = w.rows.div_ceil(THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    let mut row = vec![0.0f32; w.k];
                    for (i, o) in part.iter_mut().enumerate() {
                        let r = c * chunk + i;
                        let x0 = r / rows_per_window * w.k;
                        dequant_row(GgmlType::Q3_K, w.row(r), &mut row)
                            .expect("a Q3_K row of whole super-blocks decodes");
                        *o = row
                            .iter()
                            .enumerate()
                            .map(|(j, &v)| f64::from(v).powi(2) * (var[x0 + j] + input[x0 + j]))
                            .sum();
                    }
                });
            }
        });
        out
    }

    /// A projection's output variances from its input's: `var` carried from
    /// upstream, `x` ik's input (the dump's node), each side's rounding of
    /// it by the weight's rule — Q8_0: ik's q8_2 exactly under each weight
    /// code's sign, ours f32; Q3_K: [`Rule::input_var`].
    fn var_w(w: &HostW, var: &[f64], x: &[f32], rows_per_window: usize) -> Vec<f64> {
        match w {
            HostW::Q8(w) => var_q8(w, var, &q8_2_err(x), rows_per_window),
            HostW::Q3k(w) => {
                let rule = Rule::of(GgmlType::Q3_K).expect("Q3_K has a rule");
                var_q3k(w, var, &rule.input_var(x), rows_per_window)
            }
        }
    }

    /// `Σ_c W_rc² var_c` per row of a dense weight, `k` values a row.
    fn var_dense(w: &[f32], k: usize, var: &[f64]) -> Vec<f64> {
        w.chunks(k)
            .map(|row| {
                row.iter()
                    .zip(var)
                    .map(|(&x, v)| f64::from(x) * f64::from(x) * v)
                    .sum()
            })
            .collect()
    }

    /// A norm's first order: each variance times `(gain / rms(x))²`.
    fn norm_var(var: &[f64], x: &[f32], gain: &[f32], eps: f32) -> Vec<f64> {
        let ms = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64;
        let s = 1.0 / (ms + f64::from(eps)).sqrt();
        var.iter()
            .zip(gain)
            .map(|(v, &g)| v * (s * f64::from(g)).powi(2))
            .collect()
    }

    /// A tail rope of the variances: each pair of every `width`-value head
    /// mixes by `c²` and `s²` of `table` (`n_dims` values, cos/sin pairs).
    fn rope_var(var: &mut [f64], width: usize, table: &[f32]) {
        let nd = table.len();
        for head in var.chunks_mut(width) {
            let tail = &mut head[width - nd..];
            for (pair, cs) in tail.chunks_mut(2).zip(table.chunks(2)) {
                let (c2, s2) = (f64::from(cs[0]).powi(2), f64::from(cs[1]).powi(2));
                let (a, b) = (pair[0], pair[1]);
                pair[0] = c2 * a + s2 * b;
                pair[1] = s2 * a + c2 * b;
            }
        }
    }

    /// The variance two independent f16 roundings add near the value of f16
    /// bits `h`: `ulp² / 6`.
    fn f16_var(h: u16) -> f64 {
        let exp = i32::from((h >> 10) & 0x1f).max(1);
        let ulp = 2f64.powi(exp - 25);
        ulp * ulp / 6.0
    }

    /// The attention linearized at ik's queries, keys and sinks.
    struct AttnLin<'a> {
        q: &'a [f32],
        keys: &'a [f32],
        sinks: &'a [f32],
        scale: f32,
        var_q: &'a [f64],
        /// Keys of the step's own writing: their index among `keys` and the
        /// variance of each of their values.
        fresh: Vec<(usize, &'a [f64])>,
        /// Each output value's variance from the op's own rule.
        rule: f64,
    }

    /// The variance of each attention output value (heads × width, before
    /// the inverse rope). With `p` the softmax weights (the sink in the
    /// denominator), `o = Σ p_i k_i` and `c_i = k_i − o`, a query error
    /// `δq` moves `o` by `scale · Σ_i p_i (k_i·δq) c_i`, so value `d` has
    /// variance `scale² Σ_j var_q[j] (Σ_i p_i k_ij c_id)²`; a fresh key `n`
    /// moves it by `p_n (δk_n + scale (q·δk_n) c_n)`.
    fn attn_var(a: &AttnLin<'_>) -> Vec<f64> {
        let w = a.keys.len() / (a.keys.len() / a.q.len().max(1)).max(1);
        let heads = a.sinks.len();
        let width = a.q.len() / heads;
        let n = a.keys.len() / width;
        let _ = w;
        let mut out = vec![0.0f64; heads * width];
        let chunk = heads.div_ceil(THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk * width).enumerate() {
                s.spawn(move || {
                    for (hh, var) in part.chunks_mut(width).enumerate() {
                        let h = c * chunk + hh;
                        head_var(a, h, width, n, var);
                    }
                });
            }
        });
        out
    }

    fn head_var(a: &AttnLin<'_>, h: usize, width: usize, n: usize, var: &mut [f64]) {
        let q: Vec<f64> = a.q[h * width..(h + 1) * width]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let vq = &a.var_q[h * width..(h + 1) * width];
        let scale = f64::from(a.scale);
        let key = |i: usize| &a.keys[i * width..(i + 1) * width];
        let logits: Vec<f64> = (0..n)
            .map(|i| {
                scale
                    * key(i)
                        .iter()
                        .zip(&q)
                        .map(|(&k, q)| f64::from(k) * q)
                        .sum::<f64>()
            })
            .collect();
        let sink = f64::from(a.sinks[h]);
        let m = logits.iter().fold(sink, |m, &s| m.max(s));
        let e: Vec<f64> = logits.iter().map(|&s| (s - m).exp()).collect();
        let z = (sink - m).exp() + e.iter().sum::<f64>();
        let p: Vec<f64> = e.iter().map(|&x| x / z).collect();
        let mut o = vec![0.0f64; width];
        for (i, &pi) in p.iter().enumerate() {
            for (od, &k) in o.iter_mut().zip(key(i)) {
                *od += pi * f64::from(k);
            }
        }
        let c: Vec<f64> = (0..n)
            .flat_map(|i| key(i).iter().zip(&o).map(|(&k, od)| f64::from(k) - od))
            .collect();
        var.fill(a.rule);
        let mut r = vec![0.0f64; width];
        for (j, &v) in vq.iter().enumerate() {
            if v == 0.0 {
                continue;
            }
            r.fill(0.0);
            for (i, &pi) in p.iter().enumerate() {
                let pk = pi * f64::from(key(i)[j]);
                for (rd, cd) in r.iter_mut().zip(&c[i * width..(i + 1) * width]) {
                    *rd += pk * cd;
                }
            }
            for (vd, rd) in var.iter_mut().zip(&r) {
                *vd += scale * scale * v * rd * rd;
            }
        }
        for &(idx, u) in &a.fresh {
            let pn = p[idx];
            let qu: f64 = q.iter().zip(u).map(|(qj, uj)| qj * qj * uj).sum();
            for (d, vd) in var.iter_mut().enumerate() {
                let ad = scale * c[idx * width + d];
                let own = u[d] * (1.0 + ad * q[d]).powi(2);
                let rest = ad * ad * (qu - q[d] * q[d] * u[d]);
                *vd += pn * pn * (own + rest);
            }
        }
    }

    /// HC_PRE's rule in f64 (`hc.rs`, `hc_pre_lane`): `pre`, `post`, then
    /// `comb` row-major, from the 24 mixes.
    fn hc_pre_rule(mix: &[f64], sc: &[f32], base: &[f32], eps: f64, iters: usize) -> [f64; 24] {
        let v = |l: usize| {
            let s = sc[if l < 4 {
                0
            } else if l < 8 {
                1
            } else {
                2
            }];
            mix[l] * f64::from(s) + f64::from(base[l])
        };
        let mut out = [0.0f64; 24];
        for i in 0..4 {
            out[i] = 1.0 / (1.0 + (-v(i)).exp()) + eps;
            out[4 + i] = 2.0 / (1.0 + (-v(4 + i)).exp());
        }
        let mut m = [[0.0f64; 4]; 4];
        for (r, row) in m.iter_mut().enumerate() {
            let x: [f64; 4] = std::array::from_fn(|c| v(8 + 4 * r + c));
            let mx = x
                .iter()
                .skip(1)
                .fold(x[0], |a, &b| if a > b { a } else { b });
            let e = x.map(|t| (t - mx).exp());
            let sum = e.iter().fold(0.0, |a, &b| a + b);
            for (mc, ec) in row.iter_mut().zip(e) {
                *mc = ec / sum + eps;
            }
        }
        let cols = |m: &mut [[f64; 4]; 4]| {
            for c in 0..4 {
                let s = (0..4).fold(eps, |a, r| a + m[r][c]);
                for row in m.iter_mut() {
                    row[c] /= s;
                }
            }
        };
        cols(&mut m);
        for _ in 1..iters {
            for row in &mut m {
                let s = row.iter().fold(eps, |a, &b| a + b);
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            cols(&mut m);
        }
        for (r, row) in m.iter().enumerate() {
            out[8 + 4 * r..12 + 4 * r].copy_from_slice(row);
        }
        out
    }

    /// The covariance of HC_PRE's 24 outputs from the mixes' rounding: both
    /// sides quantize the streams — ours raw, q8_1 per 128, its gemv scaled
    /// by the RMS after; ik's normalized, q8_K per 256 — so each mix moves by
    /// `W·e`, and the outputs by the rule's Jacobian at ik's mixes.
    fn hc_cov(c: &Case, hw: &HostWeights, hp: &Hparams) -> [[f64; 24]; 24] {
        let k = c.s_in.len();
        let ms = c.s_in.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / k as f64;
        let s2 = 1.0 / (ms + f64::from(hp.rms_eps));
        let var = add(
            &q8_var(&c.s_in, 128)
                .iter()
                .map(|v| v * s2)
                .collect::<Vec<_>>(),
            &q8_var(&c.hc_normed, 256),
        );
        let mut sig = [[0.0f64; 24]; 24];
        for r in 0..24 {
            for t in r..24 {
                let (a, b) = (&hw.hc_fn[r * k..(r + 1) * k], &hw.hc_fn[t * k..(t + 1) * k]);
                let v: f64 = a
                    .iter()
                    .zip(b)
                    .zip(&var)
                    .map(|((&x, &y), v)| f64::from(x) * f64::from(y) * v)
                    .sum();
                sig[r][t] = v;
                sig[t][r] = v;
            }
        }
        let mix: Vec<f64> = c.mixes.iter().map(|&v| f64::from(v)).collect();
        let (eps, iters) = (f64::from(hp.hc.eps), hp.hc.sinkhorn_iters);
        let mut jac = [[0.0f64; 24]; 24];
        for r in 0..24 {
            let h = sig[r][r].sqrt().max(1e-9 * mix[r].abs().max(1.0));
            let (mut up, mut dn) = (mix.clone(), mix.clone());
            up[r] += h;
            dn[r] -= h;
            let (fu, fd) = (
                hc_pre_rule(&up, &hw.hc_scale, &hw.hc_base, eps, iters),
                hc_pre_rule(&dn, &hw.hc_scale, &hw.hc_base, eps, iters),
            );
            for o in 0..24 {
                jac[o][r] = (fu[o] - fd[o]) / (2.0 * h);
            }
        }
        let mut cov = [[0.0f64; 24]; 24];
        for a in 0..24 {
            for b in 0..24 {
                let mut v = 0.0;
                for r in 0..24 {
                    for t in 0..24 {
                        v += jac[a][r] * sig[r][t] * jac[b][t];
                    }
                }
                cov[a][b] = v;
            }
        }
        cov
    }

    /// Every output's derived variance, and the rounding floors.
    struct Sigma {
        /// The sub-layer output's (wo_b's) variances.
        out: Vec<f64>,
        streams: Vec<f64>,
        streams_floor: Vec<f64>,
        fold: Vec<f64>,
        fold_floor: Vec<f64>,
        ring: Vec<f64>,
        row: Option<Vec<f64>>,
        key: Option<Vec<f64>>,
        /// The compressor's projections: what its ring keeps.
        comp_kv: Option<Vec<f64>>,
        comp_score: Option<Vec<f64>>,
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the model's stages in path order, each a few lines"
    )]
    fn sigma(c: &Case, hw: &HostWeights, hp: &Hparams, kind: &Kind) -> Sigma {
        let (width, eps, n) = (hp.head_dim, hp.rms_eps, hp.n_embd);
        let zeros = vec![0.0f64; n];

        // The query.
        let v_qa = var_w(&hw.q_a, &zeros, &c.attn_norm, hw.q_a.rows());
        let v_qn = norm_var(&v_qa, &c.qr, &hw.q_a_norm, eps);
        let mut v_q = var_w(&hw.q_b, &v_qn, &c.qr_norm, hw.q_b.rows());
        rope_var(&mut v_q, width, &c.fwd);

        // The step's latent row.
        let v_kv = var_w(&hw.kv, &zeros, &c.attn_norm, hw.kv.rows());
        let mut ring = norm_var(&v_kv, &c.kv_b, &hw.kv_norm, eps);
        rope_var(&mut ring, width, &c.fwd);
        for (v, &h) in ring.iter_mut().zip(&c.ring_row) {
            *v += f16_var(h);
        }

        // The compressor's projections, row and key.
        let (mut comp_kv, mut comp_score, mut row, mut key) = (None, None, None, None);
        if let (Some(cc), Some(cw)) = (&c.comp, &hw.comp) {
            let v_x = add(&q8_var(&c.attn_norm, 128), &q8_var(&c.attn_norm, 256));
            let v_kv = var_dense(&cw.kv, n, &v_x);
            let v_sc = cw.gate.as_ref().map(|g| var_dense(g, n, &v_x));
            if let (Some(pl), Some((_, ik_row, ik_key))) = (&cc.pooled, &c.written) {
                let mut v_y = vec![0.0f64; width];
                if pl.kv.len() == 1 {
                    v_y.clone_from(&v_kv);
                } else {
                    for d in 0..width {
                        let mx = pl
                            .score
                            .iter()
                            .fold(f64::NEG_INFINITY, |m, s| m.max(f64::from(s[d])));
                        let e: Vec<f64> = pl
                            .score
                            .iter()
                            .map(|s| (f64::from(s[d]) - mx).exp())
                            .collect();
                        let z: f64 = e.iter().sum();
                        for (i, fresh) in pl.fresh.iter().enumerate() {
                            if !fresh {
                                continue;
                            }
                            let p = e[i] / z;
                            let spread = p * (f64::from(pl.kv[i][d]) - f64::from(pl.y[d]));
                            v_y[d] += p * p * v_kv[d]
                                + spread * spread * v_sc.as_ref().map_or(0.0, |v| v[d]);
                        }
                    }
                }
                let v_pre = norm_var(&v_y, &pl.y, &cw.norm, eps);
                let mut v_row = v_pre.clone();
                rope_var(&mut v_row, width, &cc.cs);
                for (v, &h) in v_row.iter_mut().zip(ik_row) {
                    *v += f16_var(h);
                }
                row = Some(v_row);
                if let (Some((proj, knorm)), Some(k_ik), Some(kp)) = (&cw.key, ik_key, &pl.key_proj)
                {
                    let v_in = add(
                        &add(&v_pre, &q8_var(&pl.normed, 128)),
                        &q8_var(&pl.normed, 256),
                    );
                    let kd = hp.indexer.head_dim;
                    let v_k = var_dense(proj, width, &v_in);
                    let mut v_kn = norm_var(&v_k, kp, knorm, eps);
                    rope_var(&mut v_kn, kd, &cc.cs);
                    let hada = f64::from(HT_SCALE).powi(2) * v_kn.iter().sum::<f64>();
                    key = Some(k_ik.iter().map(|&h| hada + f16_var(h)).collect());
                }
            }
            comp_kv = Some(v_kv);
            comp_score = v_sc;
        }

        // The attention, its inverse rope, wo_a and wo_b.
        let mut fresh = vec![(c.window_len - 1, &ring[..])];
        if let (Some((w, ..)), Some(v_row)) = (&c.written, &row) {
            fresh.push((c.window_len + w, &v_row[..]));
        }
        let band = if c.iqk {
            ATTN_BAND_IQK
        } else {
            ATTN_BAND_GENERIC
        };
        let fmax = c
            .fattn
            .iter()
            .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        let rule = (band * fmax / BAND_SIGMAS).powi(2);
        let mut v_o = attn_var(&AttnLin {
            q: &c.q_rope,
            keys: &c.keys_lin,
            sinks: &hw.sinks,
            scale: 1.0 / (width as f32).sqrt(),
            var_q: &v_q,
            fresh,
            rule,
        });
        rope_var(&mut v_o, width, &c.back);
        let v_u = var_w(&hw.out_a, &v_o, &c.attn, hp.o_lora_rank);
        let out = var_w(&hw.out_b, &v_u, &c.wo_a, hw.out_b.rows());

        // HC_POST and the fold, with HC_PRE's covariance.
        let cov = hc_cov(c, hw, hp);
        let (pre, post, comb) = (&c.hc[..4], &c.hc[4..8], &c.hc[8..24]);
        let mut streams = vec![0.0f64; 4 * n];
        let mut streams_floor = vec![0.0f64; 4 * n];
        let mut fold = vec![0.0f64; n];
        let mut fold_floor = vec![0.0f64; n];
        let pp: f64 = pre
            .iter()
            .zip(post)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        for d in 0..n {
            let o = f64::from(c.out[d]);
            let s_in = |j: usize| f64::from(c.s_in[j * n + d]);
            let mut b = [0.0f64; 24];
            for i in 0..4 {
                let mut a = [0.0f64; 24];
                a[4 + i] = o;
                for j in 0..4 {
                    a[8 + 4 * j + i] = s_in(j);
                }
                let (pi, qi) = (f64::from(pre[i]), f64::from(post[i]));
                streams[i * n + d] = qi * qi * out[d] + quad(&a, &cov);
                streams_floor[i * n + d] = 8.0
                    * U
                    * ((qi * o).abs()
                        + (0..4)
                            .map(|j| (f64::from(comb[4 * j + i]) * s_in(j)).abs())
                            .sum::<f64>());
                b[i] = f64::from(c.s_ref[i * n + d]);
                b[4 + i] = pi * o;
                for j in 0..4 {
                    b[8 + 4 * j + i] = pi * s_in(j);
                }
                fold_floor[d] += pi.abs() * streams_floor[i * n + d]
                    + 8.0 * U * (pi * f64::from(c.s_ref[i * n + d])).abs();
            }
            fold[d] = pp * pp * out[d] + quad(&b, &cov);
        }
        let _ = kind;
        Sigma {
            out,
            streams,
            streams_floor,
            fold,
            fold_floor,
            ring,
            row,
            key,
            comp_kv,
            comp_score,
        }
    }

    /// `aᵀ C a`.
    fn quad(a: &[f64; 24], c: &[[f64; 24]; 24]) -> f64 {
        let mut v = 0.0;
        for (i, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            for (j, &aj) in a.iter().enumerate() {
                v += ai * c[i][j] * aj;
            }
        }
        v
    }

    /// The largest deviation of `got` from `want` in units of each value's
    /// derived deviation after its floor, where it is, and the RMS of the
    /// same ratio.
    #[derive(Clone, Copy, Debug, Default)]
    struct Zs {
        max: f64,
        at: usize,
        rms: f64,
    }

    fn zs(got: &[f32], want: &[f32], var: &[f64], floor: Option<&[f64]>) -> Zs {
        let mut z = Zs::default();
        let mut sum = 0.0;
        for (i, ((&g, &w), &v)) in got.iter().zip(want).zip(var).enumerate() {
            let dist = (f64::from(g) - f64::from(w)).abs() - floor.map_or(0.0, |f| f[i]);
            let zi = if g.is_nan() || w.is_nan() {
                f64::INFINITY
            } else if dist <= 0.0 {
                0.0
            } else if v > 0.0 {
                dist / v.sqrt()
            } else {
                f64::INFINITY
            };
            sum += zi * zi;
            if zi > z.max || i == 0 {
                z.max = zi;
                z.at = i;
            }
        }
        z.rms = (sum / got.len().max(1) as f64).sqrt();
        z
    }

    // --------------------------------------------------------------- buffers

    /// The buffers the gate lends the piece: one of each for every layer, so
    /// a captured layer replays at the same addresses.
    struct Bufs {
        streams_in: DeviceBuffer<f32>,
        fold_in: DeviceBuffer<f32>,
        streams_out: DeviceBuffer<f32>,
        fold_out: DeviceBuffer<f32>,
        ring: DeviceTensor<u16>,
        /// The ring's shadow, which the append writes and this gate does not
        /// read.
        shadow: DeviceTensor<u16>,
        /// Per stream: its rows, its index keys, above ratio 1 its ring, and
        /// the list its layers read the rows through.
        rows: Vec<DeviceTensor<u16>>,
        keys: Vec<DeviceTensor<u16>>,
        comp_ring: Vec<Option<(DeviceTensor<f32>, DeviceTensor<f32>)>>,
        lists: Vec<DeviceBuffer<u32>>,
        image: DeviceBuffer<u32>,
    }

    impl Bufs {
        fn new(
            stream: &CudaStream,
            hp: &Hparams,
            planner: &Planner,
            image_words: usize,
            list_len: usize,
        ) -> Result<Bufs, GateError> {
            let ctx = planner.ctx_max() as usize;
            let mut rows = Vec::new();
            let mut keys = Vec::new();
            let mut comp_ring = Vec::new();
            let mut lists = Vec::new();
            for &r in planner.stream_ratios() {
                lists.push(DeviceBuffer::zeroed(stream, STEP_TOKENS * list_len)?);
                let (r, n) = (r as usize, ctx.div_ceil(r as usize));
                rows.push(DeviceTensor::zeroed(stream, n, hp.head_dim)?);
                keys.push(DeviceTensor::zeroed(stream, n, hp.indexer.head_dim)?);
                comp_ring.push(if r > 1 {
                    Some((
                        DeviceTensor::zeroed(stream, r, hp.head_dim)?,
                        DeviceTensor::zeroed(stream, r, hp.head_dim)?,
                    ))
                } else {
                    None
                });
            }
            Ok(Bufs {
                streams_in: DeviceBuffer::zeroed(stream, 4 * hp.n_embd)?,
                fold_in: DeviceBuffer::zeroed(stream, hp.n_embd)?,
                streams_out: DeviceBuffer::zeroed(stream, 4 * hp.n_embd)?,
                fold_out: DeviceBuffer::zeroed(stream, hp.n_embd)?,
                ring: DeviceTensor::zeroed(stream, ctx.min(hp.window), hp.head_dim)?,
                shadow: DeviceTensor::zeroed(stream, ctx, hp.head_dim)?,
                rows,
                keys,
                comp_ring,
                lists,
                image: DeviceBuffer::zeroed(stream, image_words)?,
            })
        }

        fn io(&mut self, kind: &Kind) -> AttnIo<'_> {
            let Bufs {
                streams_in,
                fold_in,
                streams_out,
                fold_out,
                ring,
                shadow,
                rows,
                keys,
                comp_ring,
                lists,
                ..
            } = self;
            let Some(s) = kind.stream else {
                return AttnIo {
                    streams_in,
                    fold_in,
                    streams_out,
                    fold_out,
                    ring,
                    shadow,
                    compressed: Compressed::None,
                    selection: Selection::None,
                };
            };
            let list = &mut lists[s];
            let (compressed, selection) = match kind.source {
                None => (
                    Compressed::Read(&rows[s]),
                    if kind.indexer {
                        Selection::Run {
                            keys: Some(&keys[s]),
                            list,
                        }
                    } else {
                        Selection::Read(list)
                    },
                ),
                Some((_, has_keys)) => (
                    Compressed::Source(SourceIo {
                        rows: &mut rows[s],
                        keys: if has_keys { Some(&mut keys[s]) } else { None },
                        ring: comp_ring[s].as_mut().map(|(v, sc)| (v, sc)),
                    }),
                    if kind.indexer {
                        Selection::Run { keys: None, list }
                    } else {
                        Selection::Read(list)
                    },
                ),
            };
            AttnIo {
                streams_in,
                fold_in,
                streams_out,
                fold_out,
                ring,
                shadow,
                compressed,
                selection,
            }
        }

        /// Write `c`'s inputs and `words` into the buffers a layer of `kind`
        /// reads.
        fn inject(
            &mut self,
            stream: &CudaStream,
            kind: &Kind,
            c: &Case,
            words: &[u32],
        ) -> Result<(), GateError> {
            self.streams_in.copy_from_host(stream, &c.s_in)?;
            self.fold_in.copy_from_host(stream, &c.x_in)?;
            self.ring.buf_mut().copy_from_host(stream, &c.ring)?;
            self.image.copy_from_host(stream, words)?;
            if let Some(s) = kind.stream {
                // The selection of the phase's sets, where no layer selects:
                // the identity over the visible rows. An indexer layer
                // writes its own over it.
                let mut list = vec![u32::MAX; self.lists[s].len()];
                for (i, e) in list.iter_mut().take(c.n_vis).enumerate() {
                    *e = u32::try_from(i)?;
                }
                self.lists[s].copy_from_host(stream, &list)?;
            }
            if let (Some(s), Some(rows)) = (kind.stream, &c.rows) {
                self.rows[s].buf_mut().copy_from_host(stream, rows)?;
                if let Some(keys) = &c.keys {
                    self.keys[s].buf_mut().copy_from_host(stream, keys)?;
                }
                if let (Some((v, sc)), Some((cv, csc))) = (&mut self.comp_ring[s], &c.comp_ring) {
                    v.buf_mut().copy_from_host(stream, cv)?;
                    sc.buf_mut().copy_from_host(stream, csc)?;
                }
            }
            Ok(())
        }

        fn read(&self, stream: &CudaStream, kind: &Kind) -> Result<Outputs, GateError> {
            let s = kind.stream;
            Ok(Outputs {
                streams: self.streams_out.to_host_vec(stream)?,
                fold: self.fold_out.to_host_vec(stream)?,
                ring: self.ring.buf().to_host_vec(stream)?,
                rows: s
                    .map(|s| self.rows[s].buf().to_host_vec(stream))
                    .transpose()?,
                keys: (kind.source.is_some_and(|(_, k)| k))
                    .then(|| s.map(|s| self.keys[s].buf().to_host_vec(stream)))
                    .flatten()
                    .transpose()?,
                comp_ring: match (s, kind.source) {
                    (Some(s), Some(_)) => self.comp_ring[s]
                        .as_ref()
                        .map(|(v, sc)| -> Result<_, GateError> {
                            Ok((v.buf().to_host_vec(stream)?, sc.buf().to_host_vec(stream)?))
                        })
                        .transpose()?,
                    _ => None,
                },
            })
        }
    }

    /// What a run left in the buffers the piece writes (and the rows a
    /// reading layer only reads).
    struct Outputs {
        streams: Vec<f32>,
        fold: Vec<f32>,
        ring: Vec<u16>,
        rows: Option<Vec<u16>>,
        keys: Option<Vec<u16>>,
        comp_ring: Option<(Vec<f32>, Vec<f32>)>,
    }

    impl Outputs {
        /// Bit for bit the same.
        fn same(&self, o: &Outputs) -> bool {
            let f = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            f(&self.streams, &o.streams)
                && f(&self.fold, &o.fold)
                && self.ring == o.ring
                && self.rows == o.rows
                && self.keys == o.keys
                && match (&self.comp_ring, &o.comp_ring) {
                    (Some((a, b)), Some((c, d))) => f(a, c) && f(b, d),
                    (None, None) => true,
                    _ => false,
                }
        }
    }

    // -------------------------------------------------------------- the run

    /// What the layers read besides their own weights.
    struct Cx<'a> {
        gpu: &'a Gpu,
        split: &'a Split,
        hp: &'a Hparams,
        planner: &'a Planner,
        mutation: Option<Mutation>,
    }

    /// Tallies per pin.
    #[derive(Default)]
    struct Tally {
        runs: usize,
        failed: Vec<String>,
    }

    impl Tally {
        fn check(&mut self, pass: bool, what: String) {
            self.runs += 1;
            if !pass {
                self.failed.push(what);
            }
        }
    }

    pub fn run() -> Result<(), GateError> {
        let mutation = Mutation::from_env()?;
        let model = ref_model_path()?;
        let split = Split::open(&model).map_err(|e| format!("open {}: {e}", model.display()))?;
        if split.architecture() != Some(Arch::Deepseek41.name()) {
            return Err(format!(
                "{} is not a {} model",
                model.display(),
                Arch::Deepseek41.name()
            )
            .into());
        }
        let hp = Hparams::read(&split)?;
        let oracle = for_arch(Arch::Deepseek41)?;
        let mans = [
            ("step4", oracle.open_named(STEP4)?),
            ("d1n", oracle.open_named(D1N)?),
        ];
        let ctx = mans
            .iter()
            .map(|(_, m)| m.header.ctx)
            .collect::<BTreeSet<_>>();
        let ctx = match ctx.into_iter().collect::<Vec<_>>().as_slice() {
            [Some(c)] => *c,
            other => return Err(format!("the sets' contexts {other:?}: one is needed").into()),
        };
        let planner = Planner::from_file(&split, &hp, ctx)?;
        let layout = ImageLayout::new(ImageDims::of(
            &hp,
            &planner,
            STEP_TOKENS,
            engram_row_bytes(&split, &hp)?,
        ))?;
        let (window, yarn) = rope_specs(&hp)?;
        let mut sets = Vec::new();
        for (label, man) in mans {
            let (pos0, tokens, before) = man.step()?;
            let mut plan = StepPlan::default();
            planner.plan_into(tokens, pos0, before, &mut plan)?;
            let mut image = StepImage::new(layout.clone(), &window, &yarn)?;
            let embd = vec![0u8; STEP_TOKENS * layout.dims().embd_bytes];
            let engram = vec![0u8; STEP_TOKENS * layout.dims().engram_bytes];
            image.build(&plan, &embd, &engram)?;
            let upload = mutate_image(&layout, image.words(), mutation)?;
            println!(
                "set {label}: {} (build {}) — position {pos0}, context {ctx}, window {} rows; \
                 streams {:?} visible {:?} completing {:?}",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-"),
                plan.pos[0] - plan.raw_first[0] + 1,
                planner.stream_ratios(),
                plan.streams
                    .iter()
                    .map(|s| s.n_visible[0])
                    .collect::<Vec<_>>(),
                plan.streams
                    .iter()
                    .map(|s| s.state_write.clone())
                    .collect::<Vec<_>>(),
            );
            sets.push(SetRun {
                label,
                man,
                plan,
                image,
                upload,
            });
        }

        let gpu = Gpu::new()?;
        let stream = gpu.stream();
        let mut piece = AttnChain::new(&gpu, &hp, 0..hp.n_layer, &layout, &planner)?;
        let mut bufs = Bufs::new(stream, &hp, &planner, layout.words(), piece.list_len())?;
        println!(
            "gate_deepseek41_chain_attn: device {}; model {}; piece scratch {} bytes, {} step \
             words; mutation {mutation:?}; pin Z = {Z}",
            gpu.device_name()?,
            model.display(),
            piece.device_bytes(),
            piece.words_layout().len,
        );

        let mut tally = Tally::default();
        for set in &sets {
            words_check(&gpu, &mut piece, &mut bufs, set, &mut tally)?;
        }
        let cx = Cx {
            gpu: &gpu,
            split: &split,
            hp: &hp,
            planner: &planner,
            mutation,
        };
        for l in layer_filter(hp.n_layer)? {
            gate_layer(&cx, &mut piece, &mut bufs, &sets, l, &mut tally)?;
        }
        let pass = tally.failed.is_empty();
        println!(
            "gate_deepseek41_chain_attn: {} checks, {} failed{} — {}",
            tally.runs,
            tally.failed.len(),
            if pass {
                String::new()
            } else {
                format!(": {}", tally.failed.join("; "))
            },
            verdict(pass)
        );
        if pass { Ok(()) } else { Err(checks_failed()) }
    }

    /// The words pin of one set: gather its image and read the piece's copy
    /// back against the plan's words.
    fn words_check(
        gpu: &Gpu,
        piece: &mut AttnChain,
        bufs: &mut Bufs,
        set: &SetRun,
        tally: &mut Tally,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        bufs.image.copy_from_host(stream, &set.upload)?;
        piece.enqueue_step(gpu, &bufs.image)?;
        stream.synchronize()?;
        let got = piece.words().to_host_vec(stream)?;
        let want = expected_words(piece.words_layout(), set)?;
        let wrong: Vec<usize> = want
            .iter()
            .enumerate()
            .filter(|&(i, w)| w.is_some_and(|w| got[i].to_bits() != w))
            .map(|(i, _)| i)
            .collect();
        let pass = wrong.is_empty();
        println!(
            "words set={} pos={}: {} of {} words the plan's{} — {}",
            set.label,
            set.pos(),
            want.iter().flatten().count() - wrong.len(),
            want.iter().flatten().count(),
            wrong.first().map_or(String::new(), |&i| format!(
                "; first off at word {i}: {} for {}",
                got[i].to_bits(),
                want[i].unwrap_or(0)
            )),
            verdict(pass)
        );
        tally.check(pass, format!("words {}", set.label));
        Ok(())
    }

    /// Every pin of layer `l`: both sets eager, then the capture and its
    /// replays.
    fn gate_layer(
        cx: &Cx<'_>,
        piece: &mut AttnChain,
        bufs: &mut Bufs,
        sets: &[SetRun],
        l: usize,
        tally: &mut Tally,
    ) -> Result<(), GateError> {
        let (gpu, hp) = (cx.gpu, cx.hp);
        let stream = gpu.stream();
        let kind = Kind::of(hp, cx.planner, cx.split, l)?;
        let keep = resident_names(l, &kind);
        let w = Weights::load_where(stream, cx.split, |n| keep.contains(n))?;
        let hw = HostWeights::load(cx.split, hp, l, &kind)?;
        let mut eager = Vec::with_capacity(sets.len());
        let mut cases = Vec::with_capacity(sets.len());
        for set in sets {
            let c = load_case(set, hp, l, &kind, cx.mutation)?;
            bufs.inject(stream, &kind, &c, &set.upload)?;
            piece.enqueue_step(gpu, &bufs.image)?;
            piece.enqueue_layer(gpu, &w, l, bufs.io(&kind))?;
            stream.synchronize()?;
            let got = bufs.read(stream, &kind)?;
            let sig = sigma(&c, &hw, hp, &kind);
            let worst = pins(cx, set, l, &kind, &c, &got, &sig, tally)?;
            rule_check(&c, &hw, hp, set, l, worst, tally)?;
            nodes(stream, &piece.taps(), &c, &sig, set, l)?;
            eager.push(got);
            cases.push(c);
        }

        // The capture, then a replay at each set's inputs.
        let extra = cx.mutation == Some(Mutation::ExtraLaunch);
        let graph = gpu.capture(|_| {
            piece.enqueue_step(gpu, &bufs.image)?;
            if extra {
                piece.enqueue_step(gpu, &bufs.image)?;
            }
            piece.enqueue_layer(gpu, &w, l, bufs.io(&kind))
        })?;
        let want = 1 + kind.launches();
        let mut same = Vec::with_capacity(sets.len());
        for (i, (set, c)) in sets.iter().zip(&cases).enumerate() {
            let stale = cx.mutation == Some(Mutation::StaleImage) && i > 0;
            let words = if stale { &sets[0].upload } else { &set.upload };
            bufs.inject(stream, &kind, c, words)?;
            graph.launch(stream)?;
            stream.synchronize()?;
            same.push(bufs.read(stream, &kind)?.same(&eager[i]));
        }
        let pass = graph.node_count() == want && same.iter().all(|&s| s);
        println!(
            "replay layer={l} kind={} nodes={} predicted={want} replay_equals_eager={} — {}",
            kind.name(),
            graph.node_count(),
            sets.iter()
                .zip(&same)
                .map(|(s, &e)| format!("{}:{e}", s.label))
                .collect::<Vec<_>>()
                .join(","),
            verdict(pass)
        );
        tally.check(pass, format!("layer {l} replay"));
        Ok(())
    }

    /// The value pins and the bit-identical ones of one layer in one set;
    /// returns the largest deviation any value pin measured.
    #[allow(
        clippy::too_many_arguments,
        reason = "one layer's case, outputs and deviation, and where to count them"
    )]
    fn pins(
        cx: &Cx<'_>,
        set: &SetRun,
        l: usize,
        kind: &Kind,
        c: &Case,
        got: &Outputs,
        sig: &Sigma,
        tally: &mut Tally,
    ) -> Result<f64, GateError> {
        let hp = cx.hp;
        let width = hp.head_dim;
        let n = hp.n_embd;
        let tag = format!("layer {l} {}", set.label);
        let zs_s = zs(
            &got.streams,
            &c.s_ref,
            &sig.streams,
            Some(&sig.streams_floor),
        );
        let zs_x = zs(&got.fold, &c.x_ref, &sig.fold, Some(&sig.fold_floor));
        let streams_ok = zs_s.max <= Z;
        let fold_ok = zs_x.max <= Z;
        let mut worst = zs_s.max.max(zs_x.max);

        // The ring: the step's slot against ik's row, the rest as injected.
        let rows = c.ring.len() / width;
        let slot = set.pos() % rows;
        let ours = widen(&got.ring[slot * width..(slot + 1) * width]);
        let zs_r = zs(&ours, &widen(&c.ring_row), &sig.ring, None);
        let ring_same = (0..rows)
            .filter(|&r| r != slot)
            .all(|r| got.ring[r * width..(r + 1) * width] == c.ring[r * width..(r + 1) * width]);
        let ring_ok = zs_r.max <= Z && ring_same;
        worst = worst.max(zs_r.max);

        let mut line = format!(
            "chain {tag} kind={} pos={} window={}: streams z_max={:.2} at [{}][{}] z_rms={:.2} {} | \
             fold z_max={:.2} at [{}] z_rms={:.2} {} | ring slot {slot} z_max={:.2} others_as_injected={ring_same} {}",
            kind.name(),
            set.pos(),
            c.window_len,
            zs_s.max,
            zs_s.at / n,
            zs_s.at % n,
            zs_s.rms,
            verdict(streams_ok),
            zs_x.max,
            zs_x.at,
            zs_x.rms,
            verdict(fold_ok),
            zs_r.max,
            verdict(ring_ok),
        );
        tally.check(streams_ok, format!("{tag} streams"));
        tally.check(fold_ok, format!("{tag} fold"));
        tally.check(ring_ok, format!("{tag} ring"));

        // The compressor's caches and ring.
        if let (Some(s), Some((_, has_keys))) = (kind.stream, kind.source) {
            let rows_in = c.rows.as_ref().ok_or("a source layer without rows")?;
            let rows_out = got.rows.as_ref().ok_or("no rows read back")?;
            let written = c.written.as_ref().map(|w| w.0);
            let cache_same = |a: &[u16], b: &[u16], w: usize| {
                (0..a.len() / w)
                    .filter(|&r| Some(r) != written)
                    .all(|r| a[r * w..(r + 1) * w] == b[r * w..(r + 1) * w])
            };
            let rows_same = cache_same(rows_out, rows_in, width);
            let mut ok = rows_same;
            let mut desc = format!(" | rows others_as_injected={rows_same}");
            if let (Some((w, ik_row, ik_key)), Some(v_row)) = (&c.written, &sig.row) {
                let z = zs(
                    &widen(&rows_out[w * width..(w + 1) * width]),
                    &widen(ik_row),
                    v_row,
                    None,
                );
                ok &= z.max <= Z;
                worst = worst.max(z.max);
                desc += &format!(" row {w} z_max={:.2}", z.max);
                if let (true, Some(ik_key), Some(v_key), Some(keys_out)) =
                    (has_keys, ik_key, &sig.key, &got.keys)
                {
                    let kd = hp.indexer.head_dim;
                    let zk = zs(
                        &widen(&keys_out[w * kd..(w + 1) * kd]),
                        &widen(ik_key),
                        v_key,
                        None,
                    );
                    ok &= zk.max <= Z;
                    worst = worst.max(zk.max);
                    desc += &format!(" key z_max={:.2}", zk.max);
                }
            } else if c.written.is_some() {
                ok = false;
                desc += " row: no derived deviation";
            }
            if let (true, Some(keys_in), Some(keys_out)) = (has_keys, &c.keys, &got.keys) {
                let same = cache_same(keys_out, keys_in, hp.indexer.head_dim);
                ok &= same;
                desc += &format!(" keys others_as_injected={same}");
            }
            if let (Some((ik_v, ik_s, kept)), Some((v, sc)), Some((in_v, in_s))) =
                (&c.comp_ring_ref, &got.comp_ring, &c.comp_ring)
            {
                let r = v.len() / width;
                let mut zmax = 0.0f64;
                let mut others = true;
                for slot in 0..r {
                    let span = slot * width..(slot + 1) * width;
                    if kept.contains(&u32::try_from(slot)?) {
                        let floor_v: Vec<f64> = ik_v[span.clone()]
                            .iter()
                            .map(|&x| 4.0 * U * f64::from(x).abs())
                            .collect();
                        let floor_s: Vec<f64> = ik_s[span.clone()]
                            .iter()
                            .map(|&x| 4.0 * U * f64::from(x).abs())
                            .collect();
                        let zv = sig.comp_kv.as_ref().map_or(Zs::default(), |var| {
                            zs(&v[span.clone()], &ik_v[span.clone()], var, Some(&floor_v))
                        });
                        let zc = sig.comp_score.as_ref().map_or(Zs::default(), |var| {
                            zs(&sc[span.clone()], &ik_s[span.clone()], var, Some(&floor_s))
                        });
                        zmax = zmax.max(zv.max).max(zc.max);
                    } else {
                        others &= v[span.clone()]
                            .iter()
                            .zip(&in_v[span.clone()])
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                            && sc[span.clone()]
                                .iter()
                                .zip(&in_s[span])
                                .all(|(a, b)| a.to_bits() == b.to_bits());
                    }
                }
                ok &= zmax <= Z && others;
                worst = worst.max(zmax);
                desc += &format!(" ring kept {kept:?} z_max={zmax:.2} others_as_injected={others}");
            }
            line += &format!("{desc} {}", verdict(ok));
            tally.check(ok, format!("{tag} {} compressor", STREAMS[s]));
        }
        println!("{line}");
        Ok(worst)
    }

    /// One projection as [`rule_check`] reads it: its name, weight, ik's
    /// input, ik's output where the set has it, and rows per input window.
    type Site<'a> = (&'a str, &'a HostW, &'a [f32], Option<&'a [f32]>, usize);

    /// ik's rule on the layer's Q3_K projections against the dump, bit for
    /// bit: every output ik's dot ([`act_rule::dot_q3k`]) of the weight row
    /// with the q8_K blocks of ik's own input — the codes the deviation's ik
    /// term is read from. A layer with no Q3_K projection prints nothing.
    fn rule_check(
        c: &Case,
        hw: &HostWeights,
        hp: &Hparams,
        set: &SetRun,
        l: usize,
        worst: f64,
        tally: &mut Tally,
    ) -> Result<(), GateError> {
        let sites: [Site<'_>; 5] = [
            ("q_a", &hw.q_a, &c.attn_norm, Some(&c.qr), hw.q_a.rows()),
            ("kv", &hw.kv, &c.attn_norm, Some(&c.kv_b), hw.kv.rows()),
            ("q_b", &hw.q_b, &c.qr_norm, c.q_b.as_deref(), hw.q_b.rows()),
            ("wo_a", &hw.out_a, &c.attn, Some(&c.wo_a), hp.o_lora_rank),
            ("wo_b", &hw.out_b, &c.wo_a, Some(&c.out), hw.out_b.rows()),
        ];
        let mut parts = Vec::new();
        let mut pass = true;
        for (tag, w, x, want, rows_per_window) in sites {
            let HostW::Q3k(w) = w else { continue };
            let want =
                want.ok_or_else(|| format!("layer {l}: no {tag} node in set {}", set.label))?;
            let got = ik_q3k_rows(w, x, rows_per_window);
            let same = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            pass &= got.len() == want.len() && same == want.len();
            parts.push(format!("{tag} {same}/{}", want.len()));
        }
        if parts.is_empty() {
            return Ok(());
        }
        println!(
            "    rule layer={l} set={}: ik's q8_K x Q3_K dot on ik's inputs, bit-identical \
             outputs {}; largest value-pin z/Z {:.3} — {}",
            set.label,
            parts.join(" "),
            worst / Z,
            verdict(pass)
        );
        tally.check(pass, format!("layer {l} {} ik rule", set.label));
        Ok(())
    }

    /// ik's output of every row of `w`, row `r` against window
    /// `r / rows_per_window` of `x` (`w.k` values each): its q8_K blocks,
    /// then [`act_rule::dot_q3k`].
    fn ik_q3k_rows(w: &HostQ3k, x: &[f32], rows_per_window: usize) -> Vec<f32> {
        let a = act_rule::quantize_q8_k(x);
        let (a, sb) = (&a, w.k / QK_K);
        let mut out = vec![0.0f32; w.rows];
        let chunk = w.rows.div_ceil(THREADS);
        std::thread::scope(|s| {
            for (c, part) in out.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, o) in part.iter_mut().enumerate() {
                        let r = c * chunk + i;
                        *o = act_rule::dot_q3k(w.row(r), a, r / rows_per_window * sb);
                    }
                });
            }
        });
        out
    }

    /// The node-by-node distances and the output's predicted and measured
    /// spread: printed, never pinned.
    fn nodes(
        stream: &CudaStream,
        t: &AttnTaps<'_>,
        c: &Case,
        sig: &Sigma,
        set: &SetRun,
        l: usize,
    ) -> Result<(), GateError> {
        // A broken run leaves non-finite values; the pins already said so.
        let dist = |buf: &DeviceBuffer<f32>, want: &[f32]| -> Result<String, GateError> {
            let got = buf.to_host_vec(stream)?;
            Ok(max_rel_err(&got[..want.len()], want)
                .map_or_else(|_| "non-finite".to_string(), |e| format!("{e:.2e}")))
        };
        let out = t.out.to_host_vec(stream)?;
        let rms = |v: &mut dyn Iterator<Item = f64>| {
            let (s, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x * x, n + 1));
            (s / n.max(1) as f64).sqrt()
        };
        let rms_out = rms(&mut c.out.iter().map(|&v| f64::from(v)));
        let meas = rms(&mut out
            .iter()
            .zip(&c.out)
            .map(|(&a, &b)| f64::from(a) - f64::from(b)));
        let pred = rms(&mut sig.out.iter().map(|v| v.sqrt()));
        println!(
            "    nodes layer={l} set={}: hc {} normed {} q_a {} q_a_norm {} q {} \
             kv {} kv_row {} y {} wo_a {} out {}; out spread/rms predicted \
             {:.3e} measured {:.3e}{}",
            set.label,
            dist(t.hc, &c.hc)?,
            dist(t.normed, &c.attn_norm)?,
            dist(t.q_a, &c.qr)?,
            dist(t.q_a_normed, &c.qr_norm)?,
            dist(t.q, &c.q_rope)?,
            dist(t.kv, &c.kv_b)?,
            dist(t.kv_row, &c.kv_rope)?,
            dist(t.y, &c.attn)?,
            dist(t.wo_a, &c.wo_a)?,
            dist(t.out, &c.out)?,
            pred / rms_out,
            meas / rms_out,
            if c.iqk { " (ik iqk path)" } else { "" },
        );
        Ok(())
    }
}
