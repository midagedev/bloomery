//! GPU gate for DeepSeek-V4.1's engram gate — B4 op block L
//! (`docs/research/v41-b4-plan-report.md` §1-L) — against ik's CPU dump:
//! `ds41_engram_key_norm` (the key side) and `ds41_engram_gate` (the query
//! side and the stream update) at every engram layer (`engram.layer_ids`) of
//! every set — the 5-token prefill (T = 5) and the table's decode-step sets
//! (`step_sets`, T = 1).
//!
//! A site is one layer of one set. Its [`CHAIN`] nodes, from the gather of
//! the looked-up rows to `engram_out-L`, are consecutive in the manifest
//! (manifest order is execution order); the gate checks their ops, names and
//! shapes and reads them by position — the anonymous views' `src` columns
//! name no occurrence, so a name walk cannot tell the key from the query.
//! Per site, three layers (the B4 gate form):
//! 1. ik's rule simulated here against the dump, each node from its own
//!    dumped inputs, then chained from `engram_kv-L` and `l_out-(L−1)` alone:
//!    the gather at the exact integer ids, both gains, both norms (squares
//!    summed in f64), the products, SUM_ROWS (f64), SCALE, sgn/abs/clamp/sqrt,
//!    the sigmoid (`1/(1 + expf(−x))` per value, libm's `expf`), the
//!    broadcast value times the gate and the add — bit for bit; and
//!    `engram_wkv` on ik's q8_2 activations (`bloomery_gpu_gates::ik_q8_2`)
//!    against `engram_kv-L` within ik's f32 accumulation bound. This proves the semantics: the key and
//!    value views, the gains per stream, the broadcast, and which activation
//!    rule ik's projection ran.
//! 2. The kernels on the dump's `engram_kv-L` and `l_out-(L−1)`, T tokens in
//!    one launch: against this binary's transcription of our rule,
//!    bit-identical, and rerun bit-identically; against the dump (`ik_rel`)
//!    within the band the two rules' difference derives ([`bands`]).
//! 3. The chain at the decode shape, token by token: the dump's looked-up
//!    rows through `q8_0_gemv` (f32 activations) into both kernels — the gemv
//!    against its f64 reference within `KERNEL_BAND`, the kernels against the
//!    transcription on the gemv's own output, bit-identical, and against the
//!    dump within the band that ik's q8_2 activations derive through the
//!    score. The chain captured as one graph replays bit-identically to the
//!    eager run.
//!
//! The file's types pick the rules (`bloomery_gpu_gates::act_rule`): the
//! table's rows are Q8_0 or Q3_K, dequantized either way, and so are the
//! two gains (bf16 or Q3_K, widened to f32 as the card's load widens them); a Q8_0
//! `engram_wkv` runs the rule above, a Q3_K one ik's q8_K activations and its
//! AVX2 Q3_K dot — bit for bit against `engram_kv-L` in layer 1 — and in the
//! chain the rows' q8_1 form and `q3k_gemv` (`dense::DenseKernels`), within
//! `KERNEL_BAND` of the exact dot of our q8_1 values, a four-node graph, its
//! kv's distance from ik's from `act_rule::q3k_shared_input_bound`. Any
//! other type is refused at load with the tensor named.
//!
//! Apart from the sites, the host half's two row paths ([`gate::helper_rows`]):
//! the engram rows the step's helper thread reads and the ones the calling
//! thread reads itself are the same bytes under the same ids, step by step
//! over a synthetic token stream of `HELPER_STEPS` steps — each step's rows
//! are the ones it waited for, not the other buffer of the helper's pair.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_engram: built without the `deepseek41` feature; see `just gate-gpu-ds41-engram`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_engram", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::elem::{RMS_THREADS, RMS_WARPS, rms_scale, rms_warp_tree};
    use bloomery_gpu::weights::{DevWeight, upload_file_tensor};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act};
    use bloomery_gpu_deepseek41::chain::glue::{RowsLevers, StepRows};
    use bloomery_gpu_deepseek41::dense::{Dense, DenseKernels};
    use bloomery_gpu_deepseek41::engram_gate::{
        CLAMP_MIN, EngramGateKernels, GateArgs, KeyNormArgs, PER_THREAD, ROW, inv_sqrt_row,
    };
    use bloomery_gpu_gates::act_rule::{self, Q3kRow, Q3kWeight};
    use bloomery_gpu_gates::ik_norm;
    use bloomery_gpu_gates::ik_q8_2::{self, QK};
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::rounding::{U, U64, butterfly};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, Layout, RefManifest, RowKind, bits_equal, checks_failed,
        max_rel_err, open_split, ref_ints, ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};
    use gguf::{Split, TensorInfo, Value};
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::plan::{Planner, StepPlan};

    /// Nodes of one engram site, the row gather to `engram_out-L`.
    const CHAIN: usize = 35;
    /// Each chain node's op, in manifest order (`ds4_build_engram`, visited
    /// from `engram_out` source 0 first: the value's branch, then the key's,
    /// then the query's).
    const OPS: [&str; CHAIN] = [
        "GET_ROWS", "RESHAPE", "MUL_MAT", "VIEW", "CONT", "RESHAPE", "REPEAT", "VIEW", "CONT",
        "RESHAPE", "RMS_NORM", "RESHAPE", "GET_ROWS", "RESHAPE", "MUL", "RESHAPE", "CONT",
        "RESHAPE", "RMS_NORM", "RESHAPE", "GET_ROWS", "RESHAPE", "MUL", "RESHAPE", "MUL",
        "SUM_ROWS", "SCALE", "SGN", "ABS", "CLAMP", "SQRT", "MUL", "SIGMOID", "MUL", "ADD",
    ];
    // Positions in the chain of the nodes the gate reads.
    const GATHER: usize = 0;
    const EMBD: usize = 1;
    const KV: usize = 2;
    const VALUE_VIEW: usize = 3;
    const VALUE_REP: usize = 6;
    const KEY_VIEW: usize = 7;
    const K_NORM: usize = 10;
    const K_GAIN: usize = 12;
    const KN: usize = 14;
    const Q_CONT: usize = 16;
    const Q_NORM: usize = 18;
    const Q_GAIN: usize = 20;
    const QN: usize = 22;
    const PROD: usize = 24;
    const SUM: usize = 25;
    const SCALED: usize = 26;
    const SGN: usize = 27;
    const ABS: usize = 28;
    const CLAMPED: usize = 29;
    const ROOT: usize = 30;
    const SIGNED: usize = 31;
    const GATE: usize = 32;
    const VG: usize = 33;
    const OUT: usize = 34;
    /// The chain's CONTs and RESHAPEs, each with the node whose values it
    /// carries unchanged.
    const COPIES: [(usize, usize); 11] = [
        (4, VALUE_VIEW),
        (5, VALUE_VIEW),
        (8, KEY_VIEW),
        (9, KEY_VIEW),
        (11, K_NORM),
        (13, K_GAIN),
        (15, KN),
        (17, Q_CONT),
        (19, Q_NORM),
        (21, Q_GAIN),
        (23, QN),
    ];

    /// Roundings on our side of a norm's sum of squares: [`PER_THREAD`] fused
    /// multiply-adds per thread, five butterfly levels and three
    /// `rms_warp_tree` levels. The terms are non-negative, so the sum is
    /// within this many `u` of itself.
    const OUR_SUM_ROUNDINGS: f64 = PER_THREAD as f64 + 5.0 + 3.0;

    /// How far our norm scale and ik's can sit apart on the same row,
    /// relative to the scale. Sums: ours within [`OUR_SUM_ROUNDINGS`]·u; ik's
    /// rounds each square once (u) and sums in f64 (`ROW`·2^-53). Means: each
    /// side rounds once more (2u; ours divides in f32, ik casts its f64
    /// quotient). `+ eps` rounds each side (2u). The square root halves the
    /// relative distance and rounds each side (2u); the reciprocal rounds
    /// each side (2u).
    fn scale_rel() -> f64 {
        let means = OUR_SUM_ROUNDINGS * U + U + ROW as f64 * U64 + 2.0 * U;
        (means + 2.0 * U) / 2.0 + 2.0 * U + 2.0 * U
    }

    /// The exponential's distance between the two rules, relative to it: ours
    /// is the device's f64 `exp` (within one f64 ulp, 2^-28 of u) rounded
    /// once (u); ik's is glibc's `expf`, within 0.502 ulp (its source's
    /// bound), and an ulp is at most 2u.
    const EXP_REL: f64 = (1.0 + 1.0 / 268_435_456.0 + 2.0 * 0.502) * U;

    // ------------------------------------------------------------ metadata

    /// The file's engram constants.
    struct Meta {
        eps: f32,
        hc: usize,
        layers: Vec<usize>,
        /// Values in one table row (`engram.key_length`).
        key_len: usize,
        /// Rows gathered per token: `(max_ngram_size − 1)·head_count`.
        n_cols: usize,
    }

    impl Meta {
        fn read(split: &Split) -> Result<Meta, GateError> {
            let u = |s: &str| -> Result<usize, GateError> {
                let v = split
                    .arch_get_u64(s)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(s)))?;
                Ok(usize::try_from(v)?)
            };
            let n_embd = u("embedding_length")?;
            if n_embd != ROW {
                return Err(
                    format!("embedding_length {n_embd}, the kernels hold rows of {ROW}").into(),
                );
            }
            let key = split.arch_key("engram.layer_ids");
            let Some(Value::Array(ids)) = split.value(&key) else {
                return Err(format!("metadata {key} missing or not an array").into());
            };
            let layers = ids
                .iter()
                .map(|v| {
                    v.as_unsigned()
                        .and_then(|l| usize::try_from(l).ok())
                        .ok_or_else(|| GateError::from(format!("{key} holds {v:?}")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let max_ngram = u("engram.max_ngram_size")?;
            if max_ngram < 2 {
                return Err(format!("engram.max_ngram_size {max_ngram}").into());
            }
            Ok(Meta {
                eps: split
                    .arch_get_f32("attention.layer_norm_rms_epsilon")
                    .ok_or("metadata attention.layer_norm_rms_epsilon missing")?,
                hc: u("hyper_connection.count")?,
                layers,
                key_len: u("engram.key_length")?,
                n_cols: (max_ngram - 1) * u("engram.head_count")?,
            })
        }

        /// Values one token's gathered rows hold, the projection's input width.
        fn k_in(&self) -> usize {
            self.key_len * self.n_cols
        }
    }

    /// One engram layer's weights: the projection in the file's type
    /// ([`Wkv`]), the row table's file bytes and type, and both gains widened
    /// to f32 on both sides.
    struct LayerW<'a> {
        l: usize,
        wkv: Wkv<'a>,
        table: &'a [u8],
        table_ty: GgmlType,
        gk: Vec<f32>,
        gq: Vec<f32>,
        gk_dev: DeviceBuffer<f32>,
        gq_dev: DeviceBuffer<f32>,
    }

    /// `engram_wkv` twice, in the file's type: resident as the loader packs
    /// it (`upload_file_tensor`) and on the host as both rules read it.
    enum Wkv<'a> {
        /// The q8f32 planes, and the file's bytes for [`project`].
        Q8 {
            qs: DeviceTensor<u32>,
            d: DeviceTensor<u16>,
            bytes: &'a [u8],
        },
        /// The K-quant words, and the host form of `act_rule`'s Q3_K walks.
        Q3 {
            dev: DeviceTensor<u32>,
            host: Q3kWeight,
        },
    }

    impl Wkv<'_> {
        /// Launches of the projection at m = 1: the gemv, after the rows'
        /// q8_1 form under a K-quant.
        fn launches(&self) -> usize {
            match self {
                Wkv::Q8 { .. } => 1,
                Wkv::Q3 { .. } => 2,
            }
        }
    }

    /// Bytes of one table row of `key_len` values in `ty`: Q8_0 or Q3_K,
    /// the two types the card reads rows of.
    fn table_row_bytes(ty: GgmlType, key_len: usize) -> Option<usize> {
        match ty {
            GgmlType::Q8_0 if key_len.is_multiple_of(QK) => Some(key_len / QK * 34),
            GgmlType::Q3_K if key_len.is_multiple_of(act_rule::QK_K) => {
                Some(key_len / act_rule::QK_K * act_rule::Q3K_BYTES)
            }
            _ => None,
        }
    }

    impl<'a> LayerW<'a> {
        fn load(
            split: &'a Split,
            stream: &CudaStream,
            meta: &Meta,
            l: usize,
        ) -> Result<LayerW<'a>, GateError> {
            let kv_rows = (meta.hc + 1) * ROW;
            let name = format!("blk.{l}.engram_wkv.weight");
            let (s, wkv_info) = tensor(split, &name)?;
            if !matches!(wkv_info.ty, GgmlType::Q8_0 | GgmlType::Q3_K)
                || wkv_info.dims != [meta.k_in() as u64, kv_rows as u64]
            {
                return Err(format!(
                    "{name} is {} {:?}, want q8_0 or q3_K [{}, {kv_rows}]",
                    wkv_info.ty,
                    wkv_info.dims,
                    meta.k_in()
                )
                .into());
            }
            let shard = split.shard(s).ok_or("wkv shard missing")?;
            let bytes = shard.data(wkv_info)?;
            let wkv = match upload_file_tensor(stream, shard, wkv_info)? {
                DevWeight::Q8_0 { qs, d, .. } if wkv_info.ty == GgmlType::Q8_0 => {
                    Wkv::Q8 { qs, d, bytes }
                }
                DevWeight::KQuant { w, .. } if wkv_info.ty == GgmlType::Q3_K => Wkv::Q3 {
                    dev: w,
                    host: Q3kWeight::new(bytes, meta.k_in(), kv_rows)
                        .map_err(|e| format!("{name}: {e}"))?,
                },
                _ => return Err(format!("{name}: the loader packed it in another form").into()),
            };
            let (ts, t_info) = tensor(split, &format!("blk.{l}.engram_embd.weight"))?;
            if table_row_bytes(t_info.ty, meta.key_len).is_none()
                || t_info.dims.first() != Some(&(meta.key_len as u64))
            {
                return Err(format!(
                    "blk.{l}.engram_embd.weight is {} {:?}, want q8_0 or q3_K [{}, ..]",
                    t_info.ty, t_info.dims, meta.key_len
                )
                .into());
            }
            let table = split.shard(ts).ok_or("table shard missing")?.data(t_info)?;
            // Both gains widened to f32 as the card's load widens them (bf16
            // exactly, a K-quant by `dequant_row`); layer 1 pins them against
            // ik's GET_ROWS of the gain.
            let gain = |which: &str| -> Result<Vec<f32>, GateError> {
                let name = format!("blk.{l}.engram_{which}.weight");
                let (gs, info) = tensor(split, &name)?;
                if !matches!(info.ty, GgmlType::BF16 | GgmlType::Q3_K)
                    || info.dims != [ROW as u64, meta.hc as u64]
                {
                    return Err(format!(
                        "{name} is {} {:?}, want bf16 or q3_K [{ROW}, {}]",
                        info.ty, info.dims, meta.hc
                    )
                    .into());
                }
                let mut g = vec![0.0f32; ROW * meta.hc];
                let bytes = split.shard(gs).ok_or("gain shard missing")?.data(info)?;
                dequant_row(info.ty, bytes, &mut g)?;
                Ok(g)
            };
            let (gk, gq) = (gain("k")?, gain("q")?);
            Ok(LayerW {
                l,
                wkv,
                table,
                table_ty: t_info.ty,
                gk_dev: DeviceBuffer::from_host(stream, &gk)?,
                gq_dev: DeviceBuffer::from_host(stream, &gq)?,
                gk,
                gq,
            })
        }
    }

    fn tensor<'a>(split: &'a Split, name: &str) -> Result<(usize, &'a TensorInfo), GateError> {
        split
            .find(name)
            .ok_or_else(|| format!("tensor {name} not in the model").into())
    }

    // -------------------------------------------------------------- run

    pub fn run() -> Result<(), GateError> {
        let split = open_split(Arch::Deepseek41, "gate-gpu-ds41-engram")?;
        let meta = Meta::read(&split)?;
        let gpu = Gpu::new()?;
        let eg = EngramGateKernels::load(gpu.context())?;
        let dense = DenseKernels::load(gpu.context())?;
        let stream = gpu.stream();
        println!(
            "gate_deepseek41_engram: device {} — hc {} eps {:e} layers {:?} rows/token {} x {} values",
            gpu.device_name()?,
            meta.hc,
            meta.eps,
            meta.layers,
            meta.n_cols,
            meta.key_len
        );
        let weights = meta
            .layers
            .iter()
            .map(|&l| LayerW::load(&split, stream, &meta, l))
            .collect::<Result<Vec<_>, _>>()?;

        let cpu = oracle::for_arch(Arch::Deepseek41)?;
        let mut sets = vec![(cpu.set_name(Set::Cpu)?, cpu.open(Set::Cpu)?)];
        for &name in cpu.step_sets {
            sets.push((name, cpu.open_named(name)?));
        }
        let cx = Cx {
            gpu: &gpu,
            eg: &eg,
            dense: &dense,
            meta: &meta,
        };
        let (mut sites, mut failed) = (0u32, 0u32);
        for (label, man) in &sets {
            let found = man
                .tensors
                .iter()
                .filter(|r| r.op == "ADD" && r.name.starts_with("engram_out-"))
                .count();
            if found != weights.len() {
                return Err(format!(
                    "{label}: {found} engram_out rows, the file has {} engram layers",
                    weights.len()
                )
                .into());
            }
            println!(
                "set {label}: {} (build {})",
                man.dir.display(),
                man.build.as_deref().unwrap_or("-")
            );
            for w in &weights {
                sites += 1;
                failed += u32::from(!site(&cx, label, man, w)?);
            }
        }
        let rows_pass = helper_rows(&split)?;
        let pass = failed == 0 && rows_pass;
        println!(
            "gate_deepseek41_engram: {sites} sites across {} sets, {failed} failed; host rows \
             helper = direct {} — {}",
            sets.len(),
            verdict(rows_pass),
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// Steps of the synthetic stream [`helper_rows`] walks.
    const HELPER_STEPS: usize = 256;

    /// The host half's rows two ways over one synthetic token stream: a
    /// [`StepRows`] whose helper thread reads the engram rows (keeping stats,
    /// so the classifying path runs too) and one that reads them on this
    /// thread, both filled from the same plan at every step; per step the
    /// ids of every site and the engram bytes must be equal. The stream
    /// repeats earlier tokens often, so n-grams recur and rows come back warm,
    /// and draws the rest over the whole vocabulary.
    pub(super) fn helper_rows(split: &Split) -> Result<bool, GateError> {
        let hp = Hparams::read(split)?;
        let planner = Planner::from_file(split, &hp, HELPER_STEPS as u64)?;
        let mut helper = StepRows::open_with(
            split,
            &hp,
            1,
            RowsLevers {
                helper: true,
                stats: true,
            },
        )?;
        let mut direct = StepRows::open_with(
            split,
            &hp,
            1,
            RowsLevers {
                helper: false,
                stats: false,
            },
        )?;
        let sites = hp.engram.layer_ids.len();
        let mut plan = StepPlan::default();
        let mut history: Vec<u32> = Vec::with_capacity(HELPER_STEPS);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let (mut bad_bytes, mut bad_ids, mut first_bad) = (0usize, 0usize, None);
        for pos in 0..HELPER_STEPS {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let draw = (state >> 33) as usize;
            let token = match history.len() {
                n if n > 0 && draw.is_multiple_of(3) => history[draw / 3 % n],
                _ => (draw % hp.n_vocab) as u32,
            };
            planner.plan_into(&[token], pos as u32, &history, &mut plan)?;
            helper.fill(split, &plan)?;
            direct.fill(split, &plan)?;
            let ids_eq = (0..sites).all(|s| helper.ids(0, s) == direct.ids(0, s));
            let bytes_eq = helper.engram() == direct.engram() && helper.embd() == direct.embd();
            bad_ids += usize::from(!ids_eq);
            bad_bytes += usize::from(!bytes_eq);
            if (!ids_eq || !bytes_eq) && first_bad.is_none() {
                first_bad = Some(pos);
            }
            history.push(token);
        }
        let st = helper.engram_stats();
        let pass =
            helper.helper_cpu().is_some() && bad_ids == 0 && bad_bytes == 0 && st.direct == 0;
        println!(
            "host rows: {HELPER_STEPS} steps x {sites} sites, helper {} vs direct: ids differ at {bad_ids} \
             steps, bytes at {bad_bytes} (first {first_bad:?}); helper rows warm {} cold {} \
             direct {}: {}",
            match helper.helper_cpu() {
                None => "off".to_string(),
                Some(None) => "floating".to_string(),
                Some(Some(c)) => format!("on cpu{c}"),
            },
            st.warm,
            st.cold,
            st.direct,
            verdict(pass)
        );
        Ok(pass)
    }

    /// What every site shares.
    struct Cx<'a> {
        gpu: &'a Gpu,
        eg: &'a EngramGateKernels,
        dense: &'a DenseKernels,
        meta: &'a Meta,
    }

    // ------------------------------------------------------------ the site

    /// Every dumped value of one site the checks read, in ggml's layouts.
    struct SiteData {
        t: usize,
        gather: Vec<f32>,
        embd: Vec<f32>,
        kv: Vec<f32>,
        value_view: Vec<f32>,
        value_rep: Vec<f32>,
        key_view: Vec<f32>,
        k_norm: Vec<f32>,
        k_gain: Vec<f32>,
        kn: Vec<f32>,
        x: Vec<f32>,
        q_cont: Vec<f32>,
        q_norm: Vec<f32>,
        q_gain: Vec<f32>,
        qn: Vec<f32>,
        prod: Vec<f32>,
        sum: Vec<f32>,
        scaled: Vec<f32>,
        sgn: Vec<f32>,
        abs: Vec<f32>,
        clamped: Vec<f32>,
        root: Vec<f32>,
        signed: Vec<f32>,
        gate: Vec<f32>,
        vg: Vec<f32>,
        out: Vec<f32>,
        ids: Vec<usize>,
        /// The [`COPIES`] rows' values, in its order.
        copies: Vec<Vec<f32>>,
    }

    impl SiteData {
        /// The values of chain node `i`, for the nodes [`COPIES`] names as
        /// sources.
        fn node(&self, i: usize) -> &[f32] {
            match i {
                VALUE_VIEW => &self.value_view,
                KEY_VIEW => &self.key_view,
                K_NORM => &self.k_norm,
                K_GAIN => &self.k_gain,
                KN => &self.kn,
                Q_CONT => &self.q_cont,
                Q_NORM => &self.q_norm,
                Q_GAIN => &self.q_gain,
                QN => &self.qn,
                _ => &[],
            }
        }
    }

    /// The chain of layer `l` in `man`: the [`CHAIN`] tensor rows ending at
    /// `engram_out-l`, each op, name and shape checked, and the rows it reads
    /// loaded.
    fn site_data(man: &RefManifest, meta: &Meta, l: usize) -> Result<SiteData, GateError> {
        let out_name = format!("engram_out-{l}");
        let (e, _) = man.tensor_at(&out_name, 0)?;
        if e + 1 < CHAIN {
            return Err(
                format!("{out_name} sits at manifest row {e}, before a whole chain").into(),
            );
        }
        let rows = &man.tensors[e + 1 - CHAIN..=e];
        let t = usize::try_from(rows[KV].ne[1])?;
        let (n, hc) = (ROW as u64, meta.hc as u64);
        let (tt, kc) = (t as u64, meta.k_in() as u64);
        let prev = format!("l_out-{}", l.checked_sub(1).ok_or("an engram layer 0")?);
        let row_shape = [n, hc, tt, 1];
        let flat = [n * hc, tt, 1, 1];
        let gain_shape = [n, hc, 1, 1];
        let gain_flat = [n * hc, 1, 1, 1];
        let scalar = [1, hc, tt, 1];
        let reshaped = || Some(" (reshaped)".to_string());
        let want: [(Option<String>, [u64; 4]); CHAIN] = [
            (None, [meta.key_len as u64, meta.n_cols as u64 * tt, 1, 1]),
            (Some(format!("engram_embd-{l}")), [kc, tt, 1, 1]),
            (Some(format!("engram_kv-{l}")), [n * (hc + 1), tt, 1, 1]),
            (Some(format!("engram_kv-{l} (view)")), [n, tt, 1, 1]),
            (Some(format!("engram_kv-{l} (view) (cont)")), [n, tt, 1, 1]),
            (
                Some(format!("engram_kv-{l} (view) (cont) (reshaped)")),
                [n, 1, tt, 1],
            ),
            (None, row_shape),
            (Some(format!("engram_kv-{l} (view)")), flat),
            (Some(format!("engram_kv-{l} (view) (cont)")), flat),
            (
                Some(format!("engram_kv-{l} (view) (cont) (reshaped)")),
                row_shape,
            ),
            (None, row_shape),
            (reshaped(), flat),
            (None, gain_shape),
            (reshaped(), gain_flat),
            (None, flat),
            (reshaped(), row_shape),
            (Some(format!("{prev} (cont)")), row_shape),
            (Some(format!("{prev} (cont) (reshaped)")), row_shape),
            (None, row_shape),
            (reshaped(), flat),
            (None, gain_shape),
            (reshaped(), gain_flat),
            (None, flat),
            (reshaped(), row_shape),
            (None, row_shape),
            (None, scalar),
            (None, scalar),
            (None, scalar),
            (None, scalar),
            (Some(" (view)".to_string()), scalar),
            (None, scalar),
            (None, scalar),
            (Some(format!("engram_gate-{l}")), scalar),
            (None, row_shape),
            (Some(out_name.clone()), row_shape),
        ];
        for (i, (row, (name, ne))) in rows.iter().zip(&want).enumerate() {
            if row.op != OPS[i] || row.ne != *ne || name.as_ref().is_some_and(|w| &row.name != w) {
                return Err(format!(
                    "{out_name} chain row {i}: {}/{} {} {:?}, want {} {:?}{}",
                    row.name,
                    row.occurrence,
                    row.op,
                    row.ne,
                    OPS[i],
                    ne,
                    name.as_ref()
                        .map_or(String::new(), |w| format!(" named {w:?}"))
                )
                .into());
            }
        }
        // The nodes' sources, and the value view before the key view.
        let srcs = [
            (GATHER, format!("blk.{l}.engram_embd.weight")),
            (KV, format!("blk.{l}.engram_wkv.weight")),
            (K_GAIN, format!("blk.{l}.engram_k.weight")),
            (Q_GAIN, format!("blk.{l}.engram_q.weight")),
            (Q_CONT, prev.clone()),
            (OUT, prev.clone()),
        ];
        for (i, want) in srcs {
            if rows[i].src0.as_deref() != Some(want.as_str()) {
                return Err(format!(
                    "{out_name} chain row {i} ({}): src0 {:?}, want {want:?}",
                    rows[i].name, rows[i].src0
                )
                .into());
            }
        }
        if rows[VALUE_VIEW].occurrence + 1 != rows[KEY_VIEW].occurrence {
            return Err(format!("{out_name}: the value view does not precede the key view").into());
        }
        let ld = |i: usize| ref_tensor_logical_in(&man.dir, &rows[i]);
        let (_, x_row) = man.last_before(e + 1 - CHAIN, Some(&prev))?;
        x_row.expect(&prev, "f32", row_shape, "in")?;
        let ints = |name: &str| ref_ints(man, name, 0, RowKind::Input, Layout::Flat);
        for which in ["k", "q"] {
            let ids = ints(&format!("engram_{which}_ids-{l}"))?;
            if ids != (0..hc as i64).collect::<Vec<_>>() {
                return Err(format!(
                    "engram_{which}_ids-{l} is {ids:?}, want every stream in order"
                )
                .into());
            }
        }
        let ids = ints(&format!("engram_rows-{l}"))?
            .into_iter()
            .map(|v| {
                usize::try_from(v)
                    .map_err(|_| GateError::from(format!("engram_rows-{l} holds {v}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids.len() != meta.n_cols * t {
            return Err(format!(
                "engram_rows-{l} holds {} ids, want {}",
                ids.len(),
                meta.n_cols * t
            )
            .into());
        }
        Ok(SiteData {
            t,
            gather: ld(GATHER)?,
            embd: ld(EMBD)?,
            kv: ld(KV)?,
            value_view: ld(VALUE_VIEW)?,
            value_rep: ld(VALUE_REP)?,
            key_view: ld(KEY_VIEW)?,
            k_norm: ld(K_NORM)?,
            k_gain: ld(K_GAIN)?,
            kn: ld(KN)?,
            x: ref_tensor_logical_in(&man.dir, x_row)?,
            q_cont: ld(Q_CONT)?,
            q_norm: ld(Q_NORM)?,
            q_gain: ld(Q_GAIN)?,
            qn: ld(QN)?,
            prod: ld(PROD)?,
            sum: ld(SUM)?,
            scaled: ld(SCALED)?,
            sgn: ld(SGN)?,
            abs: ld(ABS)?,
            clamped: ld(CLAMPED)?,
            root: ld(ROOT)?,
            signed: ld(SIGNED)?,
            gate: ld(GATE)?,
            vg: ld(VG)?,
            out: ld(OUT)?,
            ids,
            copies: COPIES
                .iter()
                .map(|&(i, _)| ld(i))
                .collect::<Result<_, _>>()?,
        })
    }

    /// One site: layer `w.l` of set `label`. Prints its verdict lines and
    /// returns whether every check passed.
    fn site(cx: &Cx, label: &str, man: &RefManifest, w: &LayerW) -> Result<bool, GateError> {
        let d = site_data(man, cx.meta, w.l)?;
        let k_in = cx.meta.k_in();
        let projs: Vec<TokProj> = d
            .embd
            .chunks_exact(k_in)
            .map(|x| TokProj::of(&w.wkv, (cx.meta.hc + 1) * ROW, x))
            .collect();
        let ik = ik_checks(&d, w, cx.meta, &projs)?;
        let op = op_checks(cx, &d, w)?;
        let mut chain = true;
        for (t, p) in projs.iter().enumerate() {
            chain &= chain_checks(cx, label, &d, w, t, p)?;
        }
        let pass = ik && op && chain;
        println!(
            "site {label} L={} T={}: ik_sim {} op {} chain {} — {}",
            w.l,
            d.t,
            verdict(ik),
            verdict(op),
            verdict(chain),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------- 1. ik's rule vs the dump

    /// Values of `a` bit-identical to `b`, as `hits/len`, and whether all are.
    fn hits(a: &[f32], b: &[f32]) -> (String, bool) {
        let n = a
            .iter()
            .zip(b)
            .filter(|(x, y)| x.to_bits() == y.to_bits())
            .count();
        (
            format!("{n}/{}", b.len()),
            a.len() == b.len() && n == a.len(),
        )
    }

    /// ik's RMS_NORM of each `ROW`-value row of `x`: each value times the
    /// row's [`ik_norm::scale`].
    fn ik_rms(x: &[f32], eps: f32) -> Vec<f32> {
        let mut y = vec![0.0f32; x.len()];
        for (xr, yr) in x
            .as_chunks::<ROW>()
            .0
            .iter()
            .zip(y.as_chunks_mut::<ROW>().0)
        {
            let scale = ik_norm::scale(xr, eps);
            for (o, &v) in yr.iter_mut().zip(xr) {
                *o = v * scale;
            }
        }
        y
    }

    /// `a[i] · b[i mod b.len()]` — ggml's MUL with `b` broadcast over rows.
    fn mul_bcast(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter()
            .enumerate()
            .map(|(i, &v)| v * b[i % b.len()])
            .collect()
    }

    /// ik's SUM_ROWS: each row's values summed in f64 serially, rounded once.
    fn ik_sum_rows(x: &[f32], width: usize) -> Vec<f32> {
        x.chunks_exact(width)
            .map(|r| r.iter().fold(0.0f64, |a, &v| a + f64::from(v)) as f32)
            .collect()
    }

    /// ggml's `sgn`: 1, −1, or 0 for zero and NaN.
    fn ggml_sgn(v: f32) -> f32 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    /// ggml's `clamp`, the C macros `MAX(MIN(v, hi), lo)`.
    fn ggml_clamp(v: f32, lo: f32, hi: f32) -> f32 {
        let m = if v < hi { v } else { hi };
        if m > lo { m } else { lo }
    }

    /// ik's SIGMOID, `ggml_vec_sigmoid_f32` row by row: `1.f / (1.f +
    /// expf(-x))` per value, libm's `expf`.
    fn ik_sigmoid(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    /// A site's `kv` (`[(hc+1)·ROW, t]`) as the key rows `[ROW, hc, t]` and
    /// the value rows `[ROW, t]`.
    fn views(kv: &[f32], hc: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut key = Vec::with_capacity(hc * ROW * t);
        let mut value = Vec::with_capacity(ROW * t);
        for tok in kv.chunks_exact((hc + 1) * ROW).take(t) {
            key.extend_from_slice(&tok[..hc * ROW]);
            value.extend_from_slice(&tok[hc * ROW..]);
        }
        (key, value)
    }

    /// ik's graph from a site's `kv` and `x` to the gate and `engram_out`,
    /// every node by the rules above.
    fn ik_chain(kv: &[f32], x: &[f32], w: &LayerW, meta: &Meta, t: usize) -> (Vec<f32>, Vec<f32>) {
        let hc = meta.hc;
        let (key, value) = views(kv, hc, t);
        let kn = mul_bcast(&ik_rms(&key, meta.eps), &w.gk);
        let qn = mul_bcast(&ik_rms(x, meta.eps), &w.gq);
        let prod: Vec<f32> = kn.iter().zip(&qn).map(|(&a, &b)| a * b).collect();
        let inv = inv_sqrt_row();
        let signed: Vec<f32> = ik_sum_rows(&prod, ROW)
            .iter()
            .map(|&s| {
                let s = s * inv;
                ggml_sgn(s) * ggml_clamp(s.abs(), CLAMP_MIN, f32::INFINITY).sqrt()
            })
            .collect();
        let gate = ik_sigmoid(&signed);
        let out = x
            .iter()
            .enumerate()
            .map(|(i, &xv)| xv + value[i / (hc * ROW) * ROW + i % ROW] * gate[i / ROW])
            .collect();
        (gate, out)
    }

    /// Layer 1: ik's rule against the dump, node by node and chained, and the
    /// projection by ik's rule for the weight's type against `engram_kv`:
    /// under Q8_0 its q8_2 activations within ik's f32 accumulation bound,
    /// under Q3_K its q8_K activations and AVX2 dot bit for bit.
    fn ik_checks(
        d: &SiteData,
        w: &LayerW,
        meta: &Meta,
        projs: &[TokProj],
    ) -> Result<bool, GateError> {
        let (hc, t) = (meta.hc, d.t);
        let mut line = Vec::new();
        let mut all = true;
        let mut check = |what: &str, got: &[f32], want: &[f32]| {
            let (s, ok) = hits(got, want);
            line.push(format!("{what}={s}"));
            all &= ok;
        };
        // The gather: each id's table row, dequantized.
        let row_bytes = table_row_bytes(w.table_ty, meta.key_len)
            .ok_or_else(|| format!("blk.{}.engram_embd.weight: no row size", w.l))?;
        let mut gathered = vec![0.0f32; d.ids.len() * meta.key_len];
        for (&id, out) in d.ids.iter().zip(gathered.chunks_exact_mut(meta.key_len)) {
            let b = w
                .table
                .get(id * row_bytes..(id + 1) * row_bytes)
                .ok_or_else(|| format!("row id {id} is past blk.{}.engram_embd.weight", w.l))?;
            dequant_row(w.table_ty, b, out)?;
        }
        check("gather", &gathered, &d.gather);
        check("embd", &d.embd, &d.gather);
        let (key, value) = views(&d.kv, hc, t);
        check("key_view", &key, &d.key_view);
        check("value_view", &value, &d.value_view);
        let rep: Vec<f32> = (0..t * hc * ROW)
            .map(|i| value[i / (hc * ROW) * ROW + i % ROW])
            .collect();
        check("value_rep", &rep, &d.value_rep);
        check("k_gain", &w.gk, &d.k_gain);
        check("q_gain", &w.gq, &d.q_gain);
        check("k_norm", &ik_rms(&d.key_view, meta.eps), &d.k_norm);
        check("kn", &mul_bcast(&d.k_norm, &d.k_gain), &d.kn);
        check("q_cont", &d.x, &d.q_cont);
        check("q_norm", &ik_rms(&d.q_cont, meta.eps), &d.q_norm);
        check("qn", &mul_bcast(&d.q_norm, &d.q_gain), &d.qn);
        let prod: Vec<f32> = d.kn.iter().zip(&d.qn).map(|(&a, &b)| a * b).collect();
        check("prod", &prod, &d.prod);
        check("sum", &ik_sum_rows(&d.prod, ROW), &d.sum);
        let inv = inv_sqrt_row();
        let each = |x: &[f32], f: &dyn Fn(f32) -> f32| x.iter().map(|&v| f(v)).collect::<Vec<_>>();
        check("scale", &each(&d.sum, &|s| s * inv), &d.scaled);
        check("sgn", &each(&d.scaled, &ggml_sgn), &d.sgn);
        check("abs", &each(&d.scaled, &f32::abs), &d.abs);
        check(
            "clamp",
            &each(&d.abs, &|a| ggml_clamp(a, CLAMP_MIN, f32::INFINITY)),
            &d.clamped,
        );
        check("sqrt", &each(&d.clamped, &f32::sqrt), &d.root);
        let signed: Vec<f32> = d.sgn.iter().zip(&d.root).map(|(&s, &r)| s * r).collect();
        check("signed", &signed, &d.signed);
        check("gate", &ik_sigmoid(&d.signed), &d.gate);
        let vg: Vec<f32> = d
            .value_rep
            .iter()
            .enumerate()
            .map(|(i, &v)| v * d.gate[i / ROW])
            .collect();
        check("vg", &vg, &d.vg);
        let out: Vec<f32> = d.x.iter().zip(&d.vg).map(|(&a, &b)| a + b).collect();
        check("out", &out, &d.out);
        let (cg, co) = ik_chain(&d.kv, &d.x, w, meta, t);
        check("chained_gate", &cg, &d.gate);
        check("chained_out", &co, &d.out);
        let copies = COPIES
            .iter()
            .zip(&d.copies)
            .filter(|&(&(_, src), vals)| bits_equal(vals, d.node(src)))
            .count();
        line.push(format!("copies={copies}/{}", COPIES.len()));
        all &= copies == COPIES.len();

        // engram_wkv by ik's rule, token by token.
        let (wkv, wkv_ok) = ik_wkv(projs, &d.kv, (hc + 1) * ROW, meta.k_in());
        all &= wkv_ok;
        println!(
            "ik_sim L={} T={t}: {} {wkv} {}",
            w.l,
            line.join(" "),
            verdict(all)
        );
        Ok(all)
    }

    /// Each token's `engram_kv` row (`width` values of `kv`) against ik's
    /// projection of the token's rows: under Q8_0 within [`ik_gemv_rel`] of
    /// the q8_2 dot's exact value, under Q3_K equal to [`dot_q3k`]'s value
    /// bit for bit. The printed field and whether it holds.
    ///
    /// [`dot_q3k`]: act_rule::dot_q3k
    fn ik_wkv(projs: &[TokProj], kv: &[f32], width: usize, k_in: usize) -> (String, bool) {
        let (mut worst, mut same, mut n, mut q3) = (0.0f64, 0usize, 0usize, false);
        for (p, dumped) in projs.iter().zip(kv.chunks_exact(width)) {
            match p {
                TokProj::Q8(p) => {
                    for ((&got, &sim), &mag) in dumped.iter().zip(&p.ik).zip(&p.abs_ik) {
                        let dist = (f64::from(got) - sim).abs();
                        let bound = ik_gemv_rel(k_in) * mag;
                        let ratio = if dist == 0.0 {
                            0.0
                        } else if bound > 0.0 {
                            dist / bound
                        } else {
                            f64::INFINITY
                        };
                        worst = worst.max(ratio);
                    }
                }
                TokProj::Q3 { ik, .. } => {
                    q3 = true;
                    n += dumped.len();
                    same += dumped
                        .iter()
                        .zip(ik)
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count();
                }
            }
        }
        if q3 {
            (format!("wkv_q3k_dot={same}/{n}"), same == n)
        } else {
            (
                format!("wkv_q8_2 max|kv-sim|/bound={worst:.3}"),
                worst <= 1.0,
            )
        }
    }

    // ------------------------------------------------- our rule, transcribed

    /// Thread `tid`'s values of a row, `tid + RMS_THREADS·c` for `c`
    /// ascending — the order every per-thread sum below takes.
    fn thread_values(tid: usize) -> impl Iterator<Item = usize> {
        (0..PER_THREAD).map(move |c| tid + RMS_THREADS * c)
    }

    /// Our norm scale of one row, as the kernels take it: per thread its
    /// values' squares by fused multiply-adds in order, the butterfly per
    /// warp, `rms_warp_tree`, `rms_scale`.
    fn our_scale(row: &[f32], eps: f32) -> f32 {
        let part: Vec<f32> = (0..RMS_THREADS)
            .map(|tid| thread_values(tid).fold(0.0f32, |acc, i| row[i].mul_add(row[i], acc)))
            .collect();
        let mut sums = [0.0f32; RMS_WARPS];
        for (s, lanes) in sums.iter_mut().zip(part.as_chunks::<32>().0) {
            *s = butterfly(*lanes);
        }
        rms_scale(rms_warp_tree(sums), ROW as u32, eps)
    }

    /// Our key side: `kn[(t·hc + s)·ROW + d] = (key·scale)·gain`.
    fn our_key_norm(kv: &[f32], gk: &[f32], eps: f32, hc: usize, m: usize) -> Vec<f32> {
        let (key, _) = views(kv, hc, m);
        let mut kn = vec![0.0f32; key.len()];
        let rows = key
            .as_chunks::<ROW>()
            .0
            .iter()
            .zip(kn.as_chunks_mut::<ROW>().0);
        for (b, (kr, out)) in rows.enumerate() {
            let scale = our_scale(kr, eps);
            let g = &gk[b % hc * ROW..(b % hc + 1) * ROW];
            for ((o, &v), &gv) in out.iter_mut().zip(kr).zip(g) {
                *o = (v * scale) * gv;
            }
        }
        kn
    }

    /// Our gate of one stream from the f64 dot — `gate_of` in the module.
    fn our_gate(dot: f64) -> f32 {
        let s = (dot as f32) * inv_sqrt_row();
        let m = ggml_sgn(s) * ggml_clamp(s.abs(), CLAMP_MIN, f32::INFINITY).sqrt();
        let e = f64::from(-m).exp() as f32;
        1.0 / (1.0 + e)
    }

    /// Our query side and update, `x`, `kn` and `out` `[ROW, hc, m]`: per
    /// (token, stream) the f64 dot — per thread its products
    /// `kn·((x·scale)·gain)` rounded to f32 and added in order, then warp 0
    /// lane `l` adds slots `l, l + 32, …, l + 224` in order, then the f64
    /// butterfly — the gate, and `x + value·gate`.
    fn our_gate_update(
        x: &[f32],
        kn: &[f32],
        kv: &[f32],
        w: &LayerW,
        eps: f32,
        hc: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let m = x.len() / (hc * ROW);
        let (_, value) = views(kv, hc, m);
        let mut out = vec![0.0f32; x.len()];
        let mut gate = vec![0.0f32; hc * m];
        let rows = x
            .as_chunks::<ROW>()
            .0
            .iter()
            .zip(kn.as_chunks::<ROW>().0)
            .zip(out.as_chunks_mut::<ROW>().0)
            .zip(gate.iter_mut());
        for (b, (((xr, kr), or), gslot)) in rows.enumerate() {
            let g = &w.gq[b % hc * ROW..(b % hc + 1) * ROW];
            let vr = &value[b / hc * ROW..(b / hc + 1) * ROW];
            let scale = our_scale(xr, eps);
            let dots: Vec<f64> = (0..RMS_THREADS)
                .map(|tid| {
                    thread_values(tid).fold(0.0f64, |acc, i| {
                        acc + f64::from(kr[i] * ((xr[i] * scale) * g[i]))
                    })
                })
                .collect();
            let mut lanes = [0.0f64; 32];
            for (l, v) in lanes.iter_mut().enumerate() {
                *v = (1..RMS_WARPS).fold(dots[l], |acc, wp| acc + dots[l + 32 * wp]);
            }
            let gv = our_gate(butterfly(lanes));
            *gslot = gv;
            for ((o, &xv), &vv) in or.iter_mut().zip(xr).zip(vr) {
                *o = xv + vv * gv;
            }
        }
        (out, gate)
    }

    // ---------------------------------------------------------- the bands

    /// One value per compared output: a largest distance, a largest
    /// magnitude, or their ratio, the band.
    struct Bands {
        kn: f64,
        gate: f64,
        out: f64,
    }

    /// The bands of tokens `toks` of a site for our rule against ik's, each
    /// `max|Δ| / M` with `M` the largest `|value|` ik wrote, from ik's own
    /// values (the dump) — first order in each difference, with the cross
    /// terms kept. `pert` bounds, per value, how far our `kv` rows sit from
    /// ik's: `(key [hc·ROW], value [ROW])` of one token, absent when both
    /// rules read the same `kv`.
    ///
    /// Per (token, stream), with `u` f32's unit roundoff:
    /// - key norm: our scale and ik's differ by [`scale_rel`] of the scale on
    ///   the same row. A key moved by `Bk` moves the mean square by at most
    ///   `ε = Σ(2|k|·Bk + Bk²)/Σk²` of itself and the scale by at most
    ///   `ρ = ε/(2(1 − ε)^1.5)` (the mean value theorem on `(1+ε)^−½`). So
    ///   `|Δkn| <= |gain|·(Bk·sc·(1 + ρ) + |k|·sc·ρ) + (scale_rel + 4u)·|kn|`,
    ///   the 4u the two products' roundings on each side;
    /// - query: `|Δqn| <= (scale_rel + 4u)·|qn|`;
    /// - product: `|Δp| <= |Δkn|·|qn| + |kn|·|Δqn| + |Δkn|·|Δqn| + 2u·|p|`;
    /// - dot: the sum of those, plus both f64 sums' own error (`ROW`·2^-53
    ///   of `Σ|p|` each) and one f32 rounding on each side (2u of `|dot|`);
    /// - `s = dot·(1/√ROW)`: `c·Δdot + 2u·|s|`;
    /// - `m`: `Δs/(2√(|s| − Δs)) + 2u·|m|` while the clamp and the sign
    ///   cannot move, else `2√(|s| + Δs + 1e-6)`;
    /// - gate: `σ'·Δm` with `σ'` the sigmoid's slope at the point of `[m −
    ///   Δm, m + Δm]` nearest 0, plus the exponentials' distance
    ///   ([`EXP_REL`], through `dg/de = −g²` and `g²·e = g(1−g)`) and the
    ///   add's and division's roundings on each side (4u of `g`);
    /// - `out = x + v·g`: `Bv·(g + Δg) + |v|·Δg + 2u·|v·g| + 2u·|out|`.
    fn bands(
        d: &SiteData,
        w: &LayerW,
        meta: &Meta,
        toks: std::ops::Range<usize>,
        pert: Option<(&[f64], &[f64])>,
    ) -> Bands {
        let hc = meta.hc;
        let (key, value) = views(&d.kv, hc, d.t);
        let (sr, c) = (scale_rel(), f64::from(inv_sqrt_row()));
        let zero = vec![0.0f64; hc * ROW];
        let (pk, pv) = pert.unwrap_or((&zero[..], &zero[..ROW]));
        let mut max = Bands {
            kn: 0.0,
            gate: 0.0,
            out: 0.0,
        };
        let mut mag = Bands {
            kn: 0.0,
            gate: 0.0,
            out: 0.0,
        };
        for tok in toks {
            let v = &value[tok * ROW..(tok + 1) * ROW];
            for s in 0..hc {
                let b = tok * hc + s;
                let r = b * ROW..(b + 1) * ROW;
                let (k, kn, qn) = (&key[r.clone()], &d.kn[r.clone()], &d.qn[r.clone()]);
                let (p, out) = (&d.prod[r.clone()], &d.out[r]);
                let (gk, bk) = (&w.gk[s * ROW..(s + 1) * ROW], &pk[s * ROW..(s + 1) * ROW]);
                let ssq = k
                    .iter()
                    .fold(0.0f64, |a, &x| a + f64::from(x) * f64::from(x));
                let sc = 1.0 / (ssq / ROW as f64 + f64::from(meta.eps)).sqrt();
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
                    let dk = f64::from(gk[i]).abs() * moved + (sr + 4.0 * U) * knv;
                    let dq = (sr + 4.0 * U) * qv;
                    sum_dp += dk * qv + knv * dq + dk * dq + 2.0 * U * pv_;
                    sum_p += pv_;
                    max.kn = max.kn.max(dk);
                    mag.kn = mag.kn.max(knv);
                }
                let dot = f64::from(d.sum[b]).abs();
                let ddot = sum_dp + 2.0 * ROW as f64 * U64 * sum_p + 2.0 * U * dot;
                let sv = f64::from(d.scaled[b]).abs();
                let ds = c * ddot + 2.0 * U * sv;
                let m = f64::from(d.signed[b]).abs();
                let dm = if sv - ds > f64::from(CLAMP_MIN) {
                    ds / (2.0 * (sv - ds).sqrt()) + 2.0 * U * m
                } else {
                    2.0 * (sv + ds + f64::from(CLAMP_MIN)).sqrt()
                };
                let near0 = (m - dm).max(0.0);
                let sig = 1.0 / (1.0 + (-near0).exp());
                let g = f64::from(d.gate[b]);
                let dg = sig * (1.0 - sig) * dm + EXP_REL * g * (1.0 - g) + 4.0 * U * g;
                max.gate = max.gate.max(dg);
                mag.gate = mag.gate.max(g.abs());
                for ((&vi, &bvi), &o) in v.iter().zip(pv).zip(out) {
                    let (vi, o) = (f64::from(vi).abs(), f64::from(o).abs());
                    max.out = max
                        .out
                        .max(bvi * (g + dg) + vi * dg + 2.0 * U * vi * g + 2.0 * U * o);
                    mag.out = mag.out.max(o);
                }
            }
        }
        Bands {
            kn: max.kn / mag.kn,
            gate: max.gate / mag.gate,
            out: max.out / mag.out,
        }
    }

    // ------------------------------------------------ 2. the kernels, dump in

    /// The kernels on the site's dumped `kv` and streams, all T tokens in one
    /// launch each.
    fn op_checks(cx: &Cx, d: &SiteData, w: &LayerW) -> Result<bool, GateError> {
        let (hc, m, eps) = (cx.meta.hc, d.t, cx.meta.eps);
        let stream = cx.gpu.stream();
        let kv = DeviceBuffer::from_host(stream, &d.kv)?;
        let x = DeviceBuffer::from_host(stream, &d.x)?;
        let run = || -> Result<[Vec<f32>; 3], GateError> {
            let mut kn = DeviceBuffer::<f32>::zeroed(stream, hc * m * ROW)?;
            let mut out = DeviceBuffer::<f32>::zeroed(stream, hc * m * ROW)?;
            let mut gate = DeviceBuffer::<f32>::zeroed(stream, hc * m)?;
            let key = KeyNormArgs {
                kv: &kv,
                gain: &w.gk_dev,
                eps,
                hc,
                m,
                kn: &mut kn,
            };
            cx.eg.enqueue_key_norm(stream, key)?;
            let query = GateArgs {
                x: &x,
                kn: &kn,
                kv: &kv,
                gain: &w.gq_dev,
                eps,
                hc,
                m,
                out: &mut out,
                gate: &mut gate,
            };
            cx.eg.enqueue_gate(stream, query)?;
            stream.synchronize()?;
            Ok([
                kn.to_host_vec(stream)?,
                gate.to_host_vec(stream)?,
                out.to_host_vec(stream)?,
            ])
        };
        let got = run()?;
        let again = run()?;
        let rerun = got.iter().zip(&again).all(|(a, b)| bits_equal(a, b));
        let kn_ref = our_key_norm(&d.kv, &w.gk, eps, hc, m);
        let (out_ref, gate_ref) = our_gate_update(&d.x, &kn_ref, &d.kv, w, eps, hc);
        let exact = [
            bits_equal(&got[0], &kn_ref),
            bits_equal(&got[1], &gate_ref),
            bits_equal(&got[2], &out_ref),
        ];
        let b = bands(d, w, cx.meta, 0..m, None);
        let rel = [
            max_rel_err(&got[0], &d.kn)?,
            max_rel_err(&got[1], &d.gate)?,
            max_rel_err(&got[2], &d.out)?,
        ];
        // A band that is not a number passes nothing.
        let within = [b.kn, b.gate, b.out]
            .iter()
            .zip(rel)
            .all(|(&band, r)| band.is_finite() && f64::from(r) <= band);
        let pass = exact.iter().all(|&e| e) && rerun && within;
        println!(
            "op L={} T={m}: bit_exact_host kn={} gate={} out={} bit_identical_rerun={rerun} \
             ik_rel kn={:.3e} (band {:.3e}) gate={:.3e} (band {:.3e}) out={:.3e} (band {:.3e}) {}",
            w.l,
            exact[0],
            exact[1],
            exact[2],
            rel[0],
            b.kn,
            rel[1],
            b.gate,
            rel[2],
            b.out,
            verdict(pass)
        );
        Ok(pass)
    }

    // --------------------------------------------------- 3. the chain, m = 1

    /// The projection of one token's gathered rows by `engram_wkv` (file
    /// bytes), per output row: the f64 dot with the f32 activations (our
    /// rule's exact value), with ik's q8_2 activations (the integer block
    /// dots times both scales — ik's exact value), and the magnitudes
    /// `Σ|w·x|` and `Σ_b|d_w·d_x·isum_b|` the accumulation bounds scale.
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

    /// One token's projection by `engram_wkv` on the host, by the weight's
    /// type.
    enum TokProj {
        /// [`Proj`].
        Q8(Proj),
        /// Per row, both rules' exact values and ik's dot band
        /// (`act_rule::q3k_rows`), and ik's dot itself
        /// (`act_rule::ik_q3k_rows`).
        Q3 { rows: Vec<Q3kRow>, ik: Vec<f32> },
    }

    impl TokProj {
        /// The projection of `x`, one token's gathered rows, onto `rows`
        /// rows.
        fn of(w: &Wkv, rows: usize, x: &[f32]) -> TokProj {
            match w {
                Wkv::Q8 { bytes, .. } => TokProj::Q8(project(bytes, rows, x)),
                Wkv::Q3 { host, .. } => TokProj::Q3 {
                    rows: act_rule::q3k_rows(host, x, rows),
                    ik: act_rule::ik_q3k_rows(host, x, rows),
                },
            }
        }

        /// Our rule's exact value of each row, rounded to f32: the gemv's
        /// reference.
        fn ours(&self) -> Vec<f32> {
            match self {
                TokProj::Q8(p) => p.ours.iter().map(|&v| v as f32).collect(),
                TokProj::Q3 { rows, .. } => rows.iter().map(|r| r.ours as f32).collect(),
            }
        }

        /// Per row, how far our kv may sit from ik's, `k_in` values a row.
        /// Q8_0: the two exact values' distance plus both sides'
        /// accumulation bounds and the f64 sums' own error. Q3_K:
        /// `act_rule::q3k_shared_input_bound`, whose kernel term the gemv's
        /// `KERNEL_BAND` pin proves on the same run.
        fn gap_bound(&self, k_in: usize) -> Vec<f64> {
            match self {
                TokProj::Q8(p) => {
                    let f64_sum = k_in as f64 * U64;
                    (0..p.ours.len())
                        .map(|r| {
                            (p.ours[r] - p.ik[r]).abs()
                                + (ik_gemv_rel(k_in) + f64_sum) * p.abs_ik[r]
                                + (our_gemv_rel(k_in) + f64_sum) * p.abs_ours[r]
                        })
                        .collect()
                }
                TokProj::Q3 { rows, .. } => act_rule::q3k_shared_input_bound(rows, k_in),
            }
        }
    }

    /// How far ik's f32 projection sits from its exact value, relative to
    /// `Σ_b|d_w·d_x·isum_b|`: each block term rounds at most twice (the
    /// scales' product, then the integer sum's), and a sum of `k/32` terms in
    /// any order rounds at most `k/32 − 1` times; two more for the margin.
    fn ik_gemv_rel(k: usize) -> f64 {
        (k / QK + 4) as f64 * U
    }

    /// How far our `q8_0_gemv` at m = 1 sits from its exact value, relative
    /// to `Σ|w·x|`: each lane runs one fused multiply-add per value along
    /// its `k/32` values (the weight `q·d` is exact in f32), then five
    /// butterfly levels.
    fn our_gemv_rel(k: usize) -> f64 {
        (k / 32 + 5) as f64 * U
    }

    /// The chain for token `t` of the site at the decode shape: the dump's
    /// gathered rows through `q8_0_gemv` and both kernels; token 0 also
    /// captures the three launches as one graph and replays it.
    fn chain_checks(
        cx: &Cx,
        label: &str,
        d: &SiteData,
        w: &LayerW,
        t: usize,
        p: &TokProj,
    ) -> Result<bool, GateError> {
        let (hc, eps, k_in) = (cx.meta.hc, cx.meta.eps, cx.meta.k_in());
        let stream = cx.gpu.stream();
        let emb = &d.embd[t * k_in..(t + 1) * k_in];
        let xs = &d.x[t * hc * ROW..(t + 1) * hc * ROW];
        let emb_dev = DeviceBuffer::from_host(stream, emb)?;
        let x_dev = DeviceBuffer::from_host(stream, xs)?;
        // The resident projection, with the rows' q8_1 scratch under a
        // K-quant, allocated before the capture.
        enum Gemv<'w> {
            Q8 {
                qs: &'w DeviceTensor<u32>,
                d: &'w DeviceTensor<u16>,
            },
            Q3 {
                weight: &'w DeviceTensor<u32>,
                act: Box<Q8Act>,
            },
        }
        let mut gemv = match &w.wkv {
            Wkv::Q8 { qs, d, .. } => Gemv::Q8 { qs, d },
            Wkv::Q3 { dev, .. } => Gemv::Q3 {
                weight: dev,
                act: Box::new(Q8Act::with_k(stream, 1, k_in)?),
            },
        };
        // kv, kn, out, gate.
        let mut enqueue =
            |s: &CudaStream, bufs: &mut [DeviceBuffer<f32>; 4]| -> Result<(), GpuError> {
                let [kv, kn, out, gate] = bufs;
                match &mut gemv {
                    Gemv::Q8 { qs, d } => cx
                        .gpu
                        .q8f32()
                        .enqueue_q8_0_gemv(s, qs, d, &emb_dev, 1, kv)?,
                    Gemv::Q3 { weight, act } => {
                        cx.gpu.enqueue_quantize_q8_1(&emb_dev, act)?;
                        cx.dense
                            .enqueue(cx.gpu, Dense::Q3K(weight), &emb_dev, Some(&**act), kv)?;
                    }
                }
                let key = KeyNormArgs {
                    kv: &*kv,
                    gain: &w.gk_dev,
                    eps,
                    hc,
                    m: 1,
                    kn: &mut *kn,
                };
                cx.eg.enqueue_key_norm(s, key)?;
                let query = GateArgs {
                    x: &x_dev,
                    kn: &*kn,
                    kv: &*kv,
                    gain: &w.gq_dev,
                    eps,
                    hc,
                    m: 1,
                    out,
                    gate,
                };
                cx.eg.enqueue_gate(s, query)
            };
        let buffers = || -> Result<[DeviceBuffer<f32>; 4], GateError> {
            Ok([
                DeviceBuffer::<f32>::zeroed(stream, (hc + 1) * ROW)?,
                DeviceBuffer::<f32>::zeroed(stream, hc * ROW)?,
                DeviceBuffer::<f32>::zeroed(stream, hc * ROW)?,
                DeviceBuffer::<f32>::zeroed(stream, hc)?,
            ])
        };
        let host = |bufs: &[DeviceBuffer<f32>; 4]| -> Result<Vec<Vec<f32>>, GateError> {
            bufs.iter().map(|b| Ok(b.to_host_vec(stream)?)).collect()
        };
        let mut eager = buffers()?;
        enqueue(stream, &mut eager)?;
        stream.synchronize()?;
        let got = host(&eager)?;
        let (kv_h, kn_h, out_h, gate_h) = (&got[0], &got[1], &got[2], &got[3]);

        let kv_ref = p.ours();
        let gemv_rel = max_rel_err(kv_h, &kv_ref)?;
        let kn_ref = our_key_norm(kv_h, &w.gk, eps, hc, 1);
        let (out_ref, gate_ref) = our_gate_update(xs, &kn_ref, kv_h, w, eps, hc);
        let exact = bits_equal(kn_h, &kn_ref)
            && bits_equal(gate_h, &gate_ref)
            && bits_equal(out_h, &out_ref);

        // How far our kv sits from ik's.
        let bk = p.gap_bound(k_in);
        let b = bands(
            d,
            w,
            cx.meta,
            t..t + 1,
            Some((&bk[..hc * ROW], &bk[hc * ROW..])),
        );
        let rows = t * hc * ROW..(t + 1) * hc * ROW;
        let rel = [
            max_rel_err(kn_h, &d.kn[rows.clone()])?,
            max_rel_err(gate_h, &d.gate[t * hc..(t + 1) * hc])?,
            max_rel_err(out_h, &d.out[rows])?,
        ];
        // A band that is not a number passes nothing.
        let within = [b.kn, b.gate, b.out]
            .iter()
            .zip(rel)
            .all(|(&band, r)| band.is_finite() && f64::from(r) <= band);
        let mut pass = gemv_rel <= KERNEL_BAND && exact && within;
        let mut replay = String::new();
        if t == 0 {
            let mut graphed = buffers()?;
            let g = cx.gpu.capture(|s| enqueue(s, &mut graphed))?;
            g.launch(stream)?;
            stream.synchronize()?;
            let same = host(&graphed)?
                .iter()
                .zip(&got)
                .all(|(a, b)| bits_equal(a, b));
            // The gemv (after the rows' q8_1 form under a K-quant), the key
            // norm and the gate.
            pass &= same && g.node_count() == w.wkv.launches() + 2;
            replay = format!(
                " graph_nodes={} replay_bit_identical={same}",
                g.node_count()
            );
        }
        println!(
            "chain {label} L={} t={t}: gemv max_rel_err={gemv_rel:.3e} (band {KERNEL_BAND:.0e}) \
             bit_exact_host={exact} ik_rel kn={:.3e} (band {:.3e}) gate={:.3e} (band {:.3e}) \
             out={:.3e} (band {:.3e}){replay} {}",
            w.l,
            rel[0],
            b.kn,
            rel[1],
            b.gate,
            rel[2],
            b.out,
            verdict(pass)
        );
        Ok(pass)
    }
}
