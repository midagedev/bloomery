//! GPU gate for the assembled V4.1 decode step (`bloomery_gpu_deepseek41::body`,
//! B5 phases 1 and 2: the indexer's selection wired in), on the gate placement
//! (`workstation::plan_gate`: the 3090 runs every layer and the head, the host
//! tier the experts the plan leaves off the card). The model is opened through
//! the engine's own entry (`body::open`); what each mode pins is the chain's
//! contract — the pieces' and the ops' values are their own gates'.
//!
//! - `--structure` (G2): the step captured by the engine (`capture_step`)
//!   holds, as kernels, the launches the pieces report — per layer the
//!   attention's by its kind and the MoE sub-layer's by its card experts, the
//!   glue's per step and the attention's gather — two memory-operation
//!   batches per layer, and the engram rows' arrival: one flag wait and the
//!   one copy after it, in the shadow of the layer before the first engram
//!   site, ahead of every reader of the rows; nothing else. A replay of the
//!   captured step on the
//!   step4 set's injected state, and on the d1 and d2 sets' (every indexer
//!   layer selecting; d2 at the file's own `top_k`), is bit-identical to an
//!   eager run of it (the logits and every cache, compressed row, index key
//!   and compressor state). And each
//!   layer's host-leg shadow — the kernels between its go and its wait, read
//!   in stream order off a capture of the engine's chain — holds exactly the
//!   MoE piece's own shadow work and, on the layer before each engram site,
//!   that site's token-only work; none of that work runs outside a shadow.
//!   And the engine refuses a step it cannot answer (`bloomery_gpu::fault`):
//!   after a three-token prompt, layer 0's first window-ring row is set to
//!   NaN, so the next step's layer-0 attention reads a NaN key. The engine's
//!   `step` must return the fault at layer 0 instead of a token, the step
//!   after it must refuse as poisoned naming the same fault, and a `reset`
//!   must clear both. The host tier refused the NaN at layer 0 and released
//!   the stream: its words at the fault are the release's arithmetic (the
//!   card's generation one go per layer past the host's services, row 0's
//!   counter the release less one per layer, row 1's the release), and the
//!   reset lifts that poison (one reset counted, the poison kept as refused
//!   at layer 0, the words at rest). The eight steps after it — the prompt
//!   one token a step, then greedy — are each served by the host once per
//!   layer the gate plan places, with the words at rest after every one (a
//!   counter left at the release would let every wait pass with no host
//!   service), and answer, bit for bit, the tokens the same steps answered
//!   from a reset before the fault. And a fault is the step's error even
//!   when the step then fails for another reason: with a NaN in the output
//!   norm's gain, a step of two tokens whose second id is one past the
//!   vocabulary runs the first token's whole chain — the head's q8_1
//!   quantizer raises `quant_column` at the output head — and then refuses
//!   the second id's embedding row on the host, before its launch. The step
//!   must return exactly that fault and stand one position on, poisoned by
//!   it; with the gain put back, the next step must refuse as poisoned
//!   rather than read the word back as a fault of its own; and after a
//!   reset the prompt answers its clean token again.
//! - `--sets` (G1): the decode-step sets `step4` (position 4) and `d1n`
//!   (position 301, no selection). The state a set's prefill left — each
//!   layer's window ring (the last window of ik's raw cache), each
//!   compressor's rows before the step and, above ratio 1, its ring — is
//!   written into the body's buffers, the set's history seated, and one step
//!   run eagerly; each sub-layer's streams are read where it wrote them. Pins:
//!   the engram row ids exact; the router ids exact, unless a pair of experts
//!   sits within its selection band where the order is decided (the MoE chain
//!   gate's `tie_margin`) while the layer's input is inside its envelope; each
//!   sub-layer's streams inside the envelope below; `result_output`'s argmax
//!   and its logits inside the envelope carried one step on.
//! - `--select` (G1, phase 2): the same on the sets whose streams select,
//!   `d1` (position 301, ik's `top_k` overridden to 64, which the body takes
//!   through `set_indexer_top_k`) and `d2` (position 1,025, the file's
//!   `top_k`); the state includes each key source's index keys. Per indexer
//!   layer whose stream selects, three pins: the list the attention read is
//!   the exact top-k of the layer's own scores (ties to the lower row, the op
//!   gate's `exact_top`); the indexer's query and weights sit inside their
//!   envelope (the fold's carried distance plus ik's quantizer terms, as the
//!   streams' model has them); and the list's symmetric difference with ik's
//!   `lid_top_k-L` holds only rows inside the tie band — the op gate's rule
//!   (`gate_deepseek41_index.rs` `ids_vs_rule`: a row is left out when `|s_t −
//!   s_k| ≤ β_t + β_k`), with `β` the score's spread under those input
//!   distances ([`score_band`]). Where the lists differ, what the swap alone
//!   does to the attention output (ik's queries, keys and sinks over our list
//!   and over ik's) joins that layer's and its readers' envelope terms.
//!   These pins hold past a routing tie that takes the step off ik's path,
//!   where the streams' turn into diagnostics: the exact top-k and each
//!   reader's list are our own step's, and the bands against ik's nodes
//!   are conditioned on the measured distance of the fold the layer reads,
//!   not on a distance carried along ik's path.
//! - `--greedy PROMPT` (G3): 64 tokens greedy from `tools/ref/prompts.tsv`
//!   row PROMPT after a reset, on the engine's own step, against ik's
//!   (`tools/ref/ik-greedy.sh`, which stops at EOS): at the first position
//!   where the ids differ, our margin must be below
//!   [`GREEDY_MARGIN`](bloomery_gpu_gates::GREEDY_MARGIN). Also
//!   printed: the allocator calls and the calling thread's page faults per
//!   steady step.
//! - `--ppl TAG` (G4): ik's KL-divergence base file `$BLOOMERY_DATA/ikppl/
//!   TAG.kld`, chunk by chunk from a reset, one step per id; at every scored
//!   position the paired difference `d = NLL_ours − NLL_ik`, KL(ik‖ours), the
//!   top-1 agreement and both margins. Red is Δ_PPL = e^mean(d) − 1 above
//!   [`PPL_RED`].
//!
//! The envelope (G1). Each sub-layer adds to the streams its own rule
//! difference from ik's, and carries on the one it read. A rule difference is
//! the rounding of the activations ik quantizes and ours does not, or
//! quantizes otherwise — the relative variance `v` of a quantizer's rounding
//! over the vector it reads, exact for ik's q8_2 (`ik_q8_2`), `d²/12` a value
//! for the q8_K and q8_1 blocks (`d = amax/127`) — carried through each weight
//! as the relative variance of its output (a weight that mixes its inputs
//! evenly: `Σ W²·var / |Wx|²` → `Σ var / Σ x²`). Per attention sub-layer:
//! q_a and kv over the normed input, q_b over its norm, wo_a over the
//! attention output, wo_b over wo_a's (ik's q8_2 each, ours f32 under Q8_0),
//! the compressor's q3_K over the normed input on its layer (ik's q8_K and our
//! q8_1 both), and the attention's own rule — the B4 attention band read as a
//! spread, as the attention piece's gate reads it
//! (`gate_deepseek41_chain_attn.rs`, `ATTN_BAND_*` and `BAND_SIGMAS`). Per MoE
//! sub-layer: the routed gate·up's q3_K over the normed input (both rules),
//! the down's q4_K over h (ik's q8_2, our q8_1), the shared expert's two Q8_0
//! (ik's q8_2). Per engram step: `engram_wkv` over the gathered rows (ik's
//! q8_2). The body's term enters the streams at the sub-layer's share `g =
//! |post|·|y| / |streams|`; its HC_PRE's mixes (q3_K over the normed streams,
//! ik's q8_K and our q8_1) move them through `post` (`g` again) and through
//! `comb`, whose rows sum to one, by the streams' spread about their mean
//! (`κ`). So a seam's own term is `s² = v_body·g² + v_mix·(κ² + g²)`, and the
//! pin is local: `r_k ≤ Z·sqrt(r_{k−1}² + s_k²)`, `r` the measured relative
//! distance of the streams — what the sub-layer read, carried on, plus its
//! own. A wiring fault — a dropped or swapped sub-layer, a wrong stream, an
//! engram step at the wrong layer, the wrong collapse — moves the streams by a
//! sizable fraction of themselves at one seam. Past the first layer whose
//! routed set is a near tie away from ik's the step follows a path the dump
//! does not hold, so the pins stop there and the lines after it print only;
//! what the swap alone does to the streams prints beside the flip.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_step: built without the `deepseek41` feature; see `just gate-gpu-ds41-step`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_step", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_shadow.rs"]
mod shadow;

#[cfg(feature = "deepseek41")]
mod gate {
    use crate::shadow;
    use std::alloc::{GlobalAlloc, Layout as AllocLayout, System};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    use bloomery_gpu::head::Head;
    use bloomery_gpu::hybrid::{HostExperts, PoisonKind, RELEASE};
    use bloomery_gpu::model::ChainBody;
    use bloomery_gpu::weights::{DevWeight, Weights};
    use bloomery_gpu::{FLAG_WAIT_OPS, Fault, FaultSite, Gpu, GpuError, LAYER_HEAD};
    use bloomery_gpu_deepseek41::body::{self, Body, Deepseek41Model, Seam};
    use bloomery_gpu_deepseek41::chain::attn::AttnChain;
    use bloomery_gpu_deepseek41::chain::ffn::{Ds41Host, FfnPiece};
    use bloomery_gpu_deepseek41::chain::glue::Glue;
    use bloomery_gpu_deepseek41::indexer::{HEAD_DIM as KEY_DIM, HEADS as KEY_HEADS};
    use bloomery_gpu_deepseek41::router::{N_EXPERT, N_USED};
    use bloomery_gpu_gates::ik_q8_2::{self, QK};
    use bloomery_gpu_gates::kld::KldBase;
    use bloomery_gpu_gates::nodes::{Captured, StepNode, capture_order, count_kinds};
    use bloomery_gpu_gates::oracle::deepseek41::{D1, D1N, D2, STEP4};
    use bloomery_gpu_gates::oracle::for_arch;
    use bloomery_gpu_gates::{
        GREEDY_MARGIN, GateError, Layout, NAN_F16, RefManifest, RefRow, RowKind, checks_failed,
        data_dir, patch_bytes, ref_ints, ref_tensor_logical_in, ref_tensor_of_in, split_f32,
        topk_ids_logical_within, verdict, widened_f16_rows_in,
    };
    use cuda_core::sys;
    use gguf::quant::half_to_f32;
    use gguf::{GgmlType, Split};
    use model::Tensor2;
    use model::arch::Arch;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::names;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::deepseek41::plan::{Planner, StepPlan};
    use model::placement::workstation;

    // ------------------------------------------------------------ allocator

    /// Counts allocator calls, for the per-step allocation line.
    struct Counting;

    static ALLOCS: AtomicU64 = AtomicU64::new(0);

    // SAFETY: every call forwards to `System` with the caller's own
    // arguments; the counter is a relaxed atomic that allocates nothing.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: AllocLayout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller's layout, passed through unchanged.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: AllocLayout) {
            // SAFETY: `ptr` came from `alloc` above with this layout.
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: AllocLayout) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: the caller's layout, passed through unchanged.
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: AllocLayout, new_size: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: `ptr` came from this allocator with `layout`; the new
            // size is the caller's.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: Counting = Counting;

    // ------------------------------------------------------------ constants

    /// The decode-step sets of phase 1.
    const SETS: [&str; 2] = [STEP4, D1N];
    /// The decode-step sets whose streams select (phase 2).
    const SELECT_SETS: [&str; 2] = [D1, D2];
    /// f32's unit roundoff.
    const U: f64 = f32::EPSILON as f64 / 2.0;
    /// The score band's width in derived deviations: one row's score against
    /// a normal spread, as the attention gates read a band (`BAND_SIGMAS`).
    const Z_TIE: f64 = 4.5;
    /// Our tensor-core score's distance to its exact dot, in units of `U`
    /// times the products' magnitude (`gate_deepseek41_index.rs`
    /// `TC_UNITS`).
    const TC_UNITS: f64 = 288.0;
    /// The serving context: the gate placement's, and the body's caches'.
    const CTX_MAX: u64 = workstation::CTX_MAX;
    /// The envelope's pin on a relative distance over a whole vector of
    /// streams: the model's slack, the factor the attention piece's gate puts
    /// on its own first-order model (`gate_deepseek41_chain_attn.rs:109-114`,
    /// "the factor 1.5 on top is the model's slack"). Its per-value 5.9
    /// deviations do not apply here: a norm over 20,480 independent roundings
    /// has no spread of its own to speak of, only the model's error.
    const Z: f64 = 1.5;
    /// The B4 attention gate's bands, generic and iqk path, and their reading
    /// as a spread — the attention piece's gate's constants
    /// (`gate_deepseek41_chain_attn.rs:118-126`).
    const ATTN_BAND_GENERIC: f64 = 4.9e-3;
    const ATTN_BAND_IQK: f64 = 9.675e-4;
    const BAND_SIGMAS: f64 = 4.5;
    /// The port's names of the compressed streams, in the planner's order.
    const STREAMS: [&str; 2] = ["csa", "hca"];
    /// G3: tokens generated; a first difference is judged by
    /// [`GREEDY_MARGIN`].
    const GREEDY: usize = 64;
    /// G4: red above this Δ_PPL [derived, plan.md].
    const PPL_RED: f64 = 0.015;
    /// Selection-band terms the MoE chain gate's `tie_margin` takes
    /// (`gate_deepseek41_chain_ffn.rs`): the device's `expf`/`logf` ulps and
    /// the host's.
    const DEVICE_EXP_ULPS: f64 = 2.5;
    const DEVICE_LOG_ULPS: f64 = 1.0;
    const HOST_EXP_ULPS: f64 = 1.0;
    const HOST_LOG_ULPS: f64 = 1.0;

    struct Args {
        structure: bool,
        sets: bool,
        select: bool,
        greedy: Option<u32>,
        ppl: Option<String>,
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_step [--structure] [--sets] [--select] \
             [--greedy PROMPT] [--ppl TAG]";
        let mut a = Args {
            structure: false,
            sets: false,
            select: false,
            greedy: None,
            ppl: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--structure" => a.structure = true,
                "--sets" => a.sets = true,
                "--select" => a.select = true,
                "--greedy" => {
                    let id = it
                        .next()
                        .ok_or_else(|| format!("--greedy needs a prompt id: {USAGE}"))?;
                    a.greedy = Some(id.parse()?);
                }
                "--ppl" => {
                    a.ppl = Some(
                        it.next()
                            .ok_or_else(|| format!("--ppl needs a tag: {USAGE}"))?,
                    );
                }
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        if !(a.structure || a.sets || a.select || a.greedy.is_some() || a.ppl.is_some()) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    // ------------------------------------------------------------------ run

    pub fn run() -> Result<(), GateError> {
        let args = parse_args()?;
        let path = workstation::model_v41();
        let split = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let hp = Hparams::read(&split)?;
        let inputs = PlanInputs::read(&split)?;
        let machine = workstation::plan_gate(inputs.model.layers);
        let plan = inputs.plan(&machine, CTX_MAX)?;
        let planned = &plan.cards[0];
        println!(
            "gate plan of {path}: {} layers, ctx_max {}, scratch {} B, headroom {} B",
            hp.n_layer, plan.ctx_max, planned.scratch_bytes, planned.headroom_bytes
        );

        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        let mut m = body::open(file, workstation::plan_gate, CTX_MAX as usize)?;
        let load_s = t.elapsed().as_secs_f64();
        let mut head = {
            let (gpu, w, body) = m.body_parts("gate_deepseek41_step")?;
            let [a, f, g] = body.piece_bytes();
            let (free, total) = gpu.mem_info()?;
            println!(
                "load: {} B resident in {load_s:.1} s (runtime value); pieces' scratch attn {a} ffn \
                 {f} glue {g} = {} B against the plan's scratch {} B; card {} B free of {total}",
                body.resident_bytes(),
                a + f + g,
                planned.scratch_bytes,
                free
            );
            Head::new(gpu, w, hp.rms_eps)?
        };
        println!("load: engine resident {} B", m.resident_bytes());

        let mut pass = true;
        if args.structure {
            pass &= structure(&mut m, &mut head, &split, &hp)?;
            pass &= fault_case(&mut m, plan.n_l.len())?;
            pass &= fault_behind_error_case(&mut m, &hp)?;
        }
        if args.sets {
            pass &= sets(&mut m, &mut head, &split, &hp, &SETS)?;
        }
        if args.select {
            pass &= sets(&mut m, &mut head, &split, &hp, &SELECT_SETS)?;
        }
        if let Some(prompt) = args.greedy {
            pass &= greedy(&mut m, &split, prompt)?;
        }
        if let Some(tag) = &args.ppl {
            pass &= ppl(&mut m, &hp, tag)?;
        }
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: gate_deepseek41_step");
        Ok(())
    }

    // ----------------------------------------------------------- the fault

    /// The fault case's prompt, fed one token a step.
    const FAULT_PROMPT: [u32; 3] = [1, 2, 3];
    /// Steps of a fault run: the prompt's, then greedy ones.
    const FAULT_STEPS: usize = 8;

    /// [`FAULT_STEPS`] steps from where the model stands: per step its token,
    /// the host tier's services and whether the tier's words were at rest
    /// after it; the first error ends the run.
    #[derive(Default)]
    struct FaultRun {
        tokens: Vec<u32>,
        served: Vec<u64>,
        rest: Vec<bool>,
        error: Option<String>,
    }

    impl FaultRun {
        /// Every step ran, served `services` layers and left the words at
        /// rest.
        fn served_all(&self, services: usize) -> bool {
            self.error.is_none()
                && self.tokens.len() == FAULT_STEPS
                && self.served.len() == FAULT_STEPS
                && self.served.iter().all(|&s| s == services as u64)
                && self.rest.iter().all(|&r| r)
        }
    }

    fn fault_run(m: &mut Deepseek41Model) -> Result<FaultRun, GateError> {
        const WHAT: &str = "gate_deepseek41_step fault_run";
        let mut run = FaultRun::default();
        let mut next = 0u32;
        for i in 0..FAULT_STEPS {
            let t = FAULT_PROMPT.get(i).copied().unwrap_or(next);
            let s0 = m.body(WHAT)?.hybrid().stats().served;
            let r = m.step(&[t]);
            let h = m.body(WHAT)?.hybrid();
            run.served.push(h.stats().served - s0);
            run.rest.push(h.words().at_rest());
            match r {
                Ok(tok) => {
                    run.tokens.push(tok);
                    next = tok;
                }
                Err(e) => {
                    run.error = Some(e.to_string());
                    break;
                }
            }
        }
        Ok(run)
    }

    /// The engine's step on a condemned state (the module doc's fault case).
    /// `services` is the host services a step makes: the gate plan puts every
    /// layer on the one card, and each layer's FFN piece hands its routed
    /// slots to the host tier once a step — one service per layer the plan
    /// places.
    fn fault_case(m: &mut Deepseek41Model, services: usize) -> Result<bool, GateError> {
        const WHAT: &str = "gate_deepseek41_step fault";
        m.reset()?;
        let fresh = fault_run(m)?;
        m.reset()?;
        let tok = m.step(&FAULT_PROMPT)?;
        {
            let (gpu, _, body) = m.body_parts(WHAT)?;
            let stream = gpu.stream();
            let ring = body
                .state_mut(0)
                .ok_or("the body does not run layer 0")?
                .ring;
            let width = ring.cols();
            let mut host = ring.buf().to_host_vec(stream)?;
            host[..width].fill(NAN_F16);
            ring.buf_mut().copy_from_host(stream, &host)?;
        }
        let first = m.step(&[tok]);
        let poisoned = m.poisoned();
        let at_fault = m.body(WHAT)?.hybrid().words();
        let second = m.step(&[tok]);
        let resets0 = m.body(WHAT)?.hybrid().stats().resets;
        let reset = m.reset();
        let unpoisoned = m.poisoned().is_none();
        let h = m.body(WHAT)?.hybrid();
        let (st, after_reset) = (h.stats(), h.words());
        let poison = h
            .last_poison()
            .map_or_else(|| "none".to_string(), ToString::to_string);
        let after = fault_run(m)?;
        let faulted = match &first {
            Err(GpuError::Fault { fault, .. }) => Some(*fault),
            _ => None,
        };
        let refused = match (&second, faulted) {
            (Err(GpuError::Poisoned { fault, .. }), Some(f)) => *fault == f,
            _ => false,
        };
        let describe = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        // The refusal at layer 0 serves nothing and releases both rows'
        // counters; every one of the step's waits then drains through row
        // 0's, each taking one back, and the card makes every go.
        let n = u32::try_from(services)?;
        let released = at_fault.generation.wrapping_sub(at_fault.served) == n
            && at_fault.counters[..at_fault.rows]
                .iter()
                .enumerate()
                .all(|(r, &c)| c == if r == 0 { RELEASE - n } else { RELEASE });
        let pass = faulted.is_some_and(|f| f.layer == 0 && f.site().is_some())
            && poisoned == faulted
            && refused
            && released;
        println!(
            "fault nan_key layer=0: step {}; next {}; words at the fault {at_fault} (predicted: \
             generation {services} past served, counters [RELEASE - {services}, RELEASE] with \
             RELEASE = {RELEASE}) {released} {}",
            describe(&first),
            describe(&second),
            verdict(pass)
        );
        let lifted = reset.is_ok()
            && st.resets == resets0 + 1
            && st
                .last_poison
                .is_some_and(|p| p.kind == PoisonKind::Refused && p.layer == 0)
            && after_reset.at_rest()
            && after_reset.generation == at_fault.generation
            && unpoisoned;
        let served = fresh.served_all(services) && after.served_all(services);
        let same = after.tokens == fresh.tokens && after.tokens.get(2) == Some(&tok);
        let reset_pass = lifted && served && same;
        println!(
            "fault reset: {} resets={} last poison \"{poison}\" lifted {lifted}, words after it \
             {after_reset}; {FAULT_STEPS} steps after it served {:?} with the words at rest {:?}{} \
             (want {services} each: the gate plan's {services} layers on the one card, one host \
             service each; the fresh run {:?} {:?}) {served}; tokens {:?} = the fresh run's {:?} \
             bit for bit, token {tok} after the prompt {same} {}",
            match &reset {
                Ok(()) => "Ok".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            st.resets,
            after.served,
            after.rest,
            after
                .error
                .as_ref()
                .map_or_else(String::new, |e| format!(", then error \"{e}\"")),
            fresh.served,
            fresh.rest,
            after.tokens,
            fresh.tokens,
            verdict(reset_pass)
        );
        Ok(pass && reset_pass)
    }

    /// The output norm's gain, whose first value the fault-behind-an-error
    /// case sets to NaN and puts back.
    const HEAD_GAIN: &str = "output_norm.weight";

    /// Write `bytes` over the first value of [`HEAD_GAIN`] and return the
    /// bytes it replaced.
    fn patch_head_gain(m: &mut Deepseek41Model, bytes: [u8; 4]) -> Result<[u8; 4], GateError> {
        let (gpu, w, _) = m.body_parts("gate_deepseek41_step patch_head_gain")?;
        let Some(DevWeight::F32 { w: gain, .. }) = w.get(HEAD_GAIN) else {
            return Err(format!("{HEAD_GAIN} is not resident as F32").into());
        };
        patch_bytes(gpu.stream(), gain.buf(), 0, bytes)
    }

    /// A step that faults and then fails for another reason (the module
    /// doc's second fault case). `hp.n_vocab` is one past the embedding
    /// table's rows, which the engine's step rows check against it.
    fn fault_behind_error_case(m: &mut Deepseek41Model, hp: &Hparams) -> Result<bool, GateError> {
        let bad = u32::try_from(hp.n_vocab)?;
        let want = Fault::at(LAYER_HEAD, FaultSite::QuantColumn);
        m.reset()?;
        let clean = m.step(&FAULT_PROMPT)?;
        m.reset()?;
        let old = patch_head_gain(m, f32::NAN.to_le_bytes())?;
        let first = m.step(&[FAULT_PROMPT[0], bad]);
        let (at, poisoned) = (m.pos(), m.poisoned());
        patch_head_gain(m, old)?;
        let next = m.step(&[FAULT_PROMPT[1]]);
        let reset = m.reset();
        let unpoisoned = m.poisoned().is_none();
        let again = m.step(&FAULT_PROMPT);
        m.reset()?;
        let describe = |r: &Result<u32, GpuError>| match r {
            Ok(t) => format!("token {t}"),
            Err(e) => format!("error \"{e}\""),
        };
        let named = matches!(&first, Err(GpuError::Fault { fault, .. }) if *fault == want);
        let refused = matches!(&next, Err(GpuError::Poisoned { fault, .. }) if *fault == want);
        let cleared = reset.is_ok() && unpoisoned && matches!(&again, Ok(t) if *t == clean);
        let pass = named && at == 1 && poisoned == Some(want) && refused && cleared;
        println!(
            "fault behind an error: NaN in {HEAD_GAIN}[0], a step of [{}, {bad}] (want the fault \
             {want} at the first token's head, then the second id refused before its launch): \
             step {} at position {at} (want 1), poisoned {}; next step with the gain put back {} \
             (want refused as poisoned); reset {}, the prompt again {} (clean token {clean}) {}",
            FAULT_PROMPT[0],
            describe(&first),
            poisoned.map_or_else(|| "none".to_string(), |f| f.to_string()),
            describe(&next),
            match &reset {
                Ok(()) => "Ok".to_string(),
                Err(e) => format!("error \"{e}\""),
            },
            describe(&again),
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- the set's state

    /// One decode-step set: its manifest, the step, the host plan of it and
    /// the indexer's `top_k` ik ran with.
    struct Set {
        name: &'static str,
        man: RefManifest,
        pos: u32,
        token: u32,
        before: Vec<u32>,
        plan: StepPlan,
        top_k: usize,
    }

    /// The top-k override in a set's `# flags` — ik's `--override-kv
    /// <arch>.attention.indexer.top_k=int:<n>` — if there is one
    /// (`gate_deepseek41_index.rs` `top_k_override`).
    fn top_k_override(flags: &str) -> Result<Option<usize>, GateError> {
        let key = format!("{}.attention.indexer.top_k=int:", Arch::Deepseek41.name());
        let mut it = flags.split_whitespace();
        let mut found = None;
        while let Some(tok) = it.next() {
            if tok == "--override-kv" {
                let kv = it.next().ok_or("# flags: --override-kv without a value")?;
                if let Some(v) = kv.strip_prefix(key.as_str()) {
                    found = Some(v.parse::<usize>()?);
                }
            }
        }
        Ok(found)
    }

    fn open_set(split: &Split, hp: &Hparams, name: &'static str) -> Result<Set, GateError> {
        let man = for_arch(Arch::Deepseek41)?.open_named(name)?;
        let (pos, tokens, before) = man.step()?;
        let (tokens, before) = (tokens.to_vec(), before.to_vec());
        let ctx = man
            .header
            .ctx
            .unwrap_or((before.len() + tokens.len()) as u64);
        let planner = Planner::from_file(split, hp, ctx)?;
        let mut plan = StepPlan::default();
        planner.plan_into(&tokens, pos, &before, &mut plan)?;
        if tokens.len() != 1 {
            return Err(format!("{name}: a step of {} tokens", tokens.len()).into());
        }
        let top_k = match man.header.flags.as_deref() {
            Some(flags) => top_k_override(flags)?,
            None => None,
        };
        println!(
            "set {name}: {} (build {}) position {pos} token {} ctx {ctx}{}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            tokens[0],
            top_k.map_or(String::new(), |k| format!(" top_k {k} (overridden)"))
        );
        Ok(Set {
            name,
            man,
            pos,
            token: tokens[0],
            before,
            plan,
            top_k: top_k.unwrap_or(hp.indexer.top_k),
        })
    }

    fn f32s(man: &RefManifest, row: &RefRow) -> Result<Vec<f32>, GateError> {
        ref_tensor_logical_in(&man.dir, row)
    }

    fn named(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        f32s(man, man.tensor(name, 0)?)
    }

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

    /// One layer's state before the step, in the body's layout.
    struct LayerState {
        ring: Vec<u16>,
        rows: Option<Vec<u16>>,
        keys: Option<Vec<u16>>,
        comp: Option<(Vec<f32>, Vec<f32>)>,
    }

    /// A cache of compressed rows or index keys as the set holds it: ik's
    /// cache after its write this step (`write`, a `SET_ROWS`) or, where the
    /// step writes none, the leaf the view `view` reads, before the step;
    /// `None` when the set has neither. The rows the step does not write are
    /// the state before it either way.
    fn cache_rows(
        man: &RefManifest,
        write: &str,
        view: &str,
    ) -> Result<Option<Vec<u16>>, GateError> {
        if let Ok((_, w)) = node(man, write, "SET_ROWS") {
            return Ok(Some(widened_f16_rows_in(&man.dir, w)?));
        }
        match node(man, view, "VIEW") {
            Ok((_, v)) => {
                let leaf = v.src0.as_deref().ok_or("a cache view has no src0")?;
                Ok(Some(widened_f16_rows_in(&man.dir, man.input(leaf, 0)?)?))
            }
            Err(_) => Ok(None),
        }
    }

    /// `n_vis` rows of `width` from `from` into a cache of `cap` rows, every
    /// other row — and the step's written row `written` — f16 NaN.
    fn seat_rows(
        from: &[u16],
        width: usize,
        cap: usize,
        n_vis: usize,
        written: Option<usize>,
    ) -> Vec<u16> {
        let mut rows = vec![NAN_F16; cap * width];
        for r in (0..n_vis).filter(|&r| Some(r) != written) {
            rows[r * width..(r + 1) * width].copy_from_slice(&from[r * width..(r + 1) * width]);
        }
        rows
    }

    /// Every layer's state the set's prefill left: the window ring from
    /// ik's raw cache, a compressor's rows before its write this step, and
    /// above ratio 1 the compressor's ring (the attention piece's gate's
    /// lookups, `gate_deepseek41_chain_attn.rs` `load_case`/`load_comp`).
    fn set_state(set: &Set, hp: &Hparams) -> Result<Vec<LayerState>, GateError> {
        let man = &set.man;
        let width = hp.head_dim;
        let pos = set.pos as usize;
        let len = (set.plan.pos[0] - set.plan.raw_first[0] + 1) as usize;
        let ring_rows = hp.window.min(CTX_MAX as usize);
        let first = pos + 1 - len;
        let mut out = Vec::with_capacity(hp.n_layer);
        for (l, kind) in hp.layers.iter().enumerate() {
            let before = widened_f16_rows_in(&man.dir, man.input(&format!("cache_k_l{l}"), 0)?)?;
            let mut ring = vec![NAN_F16; ring_rows * width];
            for c in first..pos {
                let slot = c % ring_rows;
                ring[slot * width..(slot + 1) * width]
                    .copy_from_slice(&before[c * width..(c + 1) * width]);
            }
            let (rows, keys, comp) = match (kind.stream, kind.compressor) {
                (Some(st), Some(c)) if st.kv_source == l => {
                    let s = stream_index(&set.plan, st.ratio)?;
                    let tag = STREAMS[s];
                    let pst = &set.plan.streams[s];
                    // Where the step selects, `{tag}_k-L` is the gathered
                    // rows, not a view of the cache.
                    let pre = match node(man, &format!("{tag}_k-{l}"), "VIEW") {
                        Ok((_, view)) => {
                            let leaf = view.src0.as_deref().ok_or("the rows view has no src0")?;
                            widened_f16_rows_in(&man.dir, man.input(leaf, 0)?)?
                        }
                        Err(_) => cache_rows(man, &format!("{tag}_k_write-{l}"), "-")?
                            .ok_or_else(|| format!("layer {l}: no cache of compressed rows"))?,
                    };
                    let n_vis = pst.n_visible[0] as usize;
                    let written = pst.state_write.first().map(|&w| w as usize);
                    let cap = (CTX_MAX as usize).div_ceil(st.ratio as usize);
                    let rows = seat_rows(&pre, width, cap, n_vis, written);
                    let keys = if kind.index_keys {
                        cache_rows(man, &format!("lid_k_write-{l}"), &format!("lid_k-{l}"))?
                            .map(|k| seat_rows(&k, hp.indexer.head_dim, cap, n_vis, written))
                    } else {
                        None
                    };
                    let comp = if c.gated && st.ratio > 1 {
                        Some(comp_ring(man, l, tag, st.ratio as usize, width)?)
                    } else {
                        None
                    };
                    (Some(rows), keys, comp)
                }
                _ => (None, None, None),
            };
            out.push(LayerState {
                ring,
                rows,
                keys,
                comp,
            });
        }
        Ok(out)
    }

    /// The plan's stream of ratio `ratio`.
    fn stream_index(plan: &StepPlan, ratio: u32) -> Result<usize, GateError> {
        plan.streams
            .iter()
            .position(|s| s.ratio == ratio)
            .ok_or_else(|| format!("the plan has no stream of ratio {ratio}").into())
    }

    /// A ratio-`r` compressor's ring before the step: the f32 input of its
    /// shape the sources' CONCAT touches first, or the persist's when no
    /// group completes.
    fn comp_ring(
        man: &RefManifest,
        l: usize,
        tag: &str,
        r: usize,
        width: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let ring_ne = [width as u64, r as u64, 1, 1];
        let by_src0 = |src0: &str| {
            man.tensors
                .iter()
                .find(|t| t.op == "MUL_MAT" && t.src0.as_deref() == Some(src0))
        };
        let kv = by_src0(&names::attn_compressor_kv(l))
            .ok_or_else(|| format!("layer {l}: no compressor kv projection"))?;
        let sc = by_src0(&names::attn_compressor_gate(l))
            .ok_or_else(|| format!("layer {l}: no compressor gate projection"))?;
        let (at_pk, _) = node(man, &format!("{tag}_k_state_persist-{l}"), "SET_ROWS")?;
        let (at_ps, _) = node(man, &format!("{tag}_score_state_persist-{l}"), "SET_ROWS")?;
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
        let v = first(&format!("{tag}_source_kv"), &kv.name, at_pk)
            .ok_or_else(|| format!("layer {l}: no ring of values"))?;
        let s = first(&format!("{tag}_source_score"), &sc.name, at_ps)
            .ok_or_else(|| format!("layer {l}: no ring of scores"))?;
        Ok((
            ref_tensor_of_in(&man.dir, v)?,
            ref_tensor_of_in(&man.dir, s)?,
        ))
    }

    /// Reset the body, write `state` into its buffers and seat the set's
    /// history; the next step is the set's.
    fn inject(
        gpu: &Gpu,
        body: &mut Body,
        set: &Set,
        state: &[LayerState],
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        body.reset(gpu)?;
        for (l, s) in state.iter().enumerate() {
            let b = body
                .state_mut(l)
                .ok_or_else(|| format!("the body does not run layer {l}"))?;
            b.ring.buf_mut().copy_from_host(stream, &s.ring)?;
            match (&s.rows, b.rows) {
                (Some(h), Some(d)) => d.buf_mut().copy_from_host(stream, h)?,
                (None, _) => {}
                (Some(_), None) => return Err(format!("layer {l}: the body holds no rows").into()),
            }
            match (&s.keys, b.keys) {
                (Some(h), Some(d)) => d.buf_mut().copy_from_host(stream, h)?,
                (None, _) => {}
                (Some(_), None) => {
                    return Err(format!("layer {l}: the body holds no index keys").into());
                }
            }
            match (&s.comp, b.values, b.scores) {
                (Some((v, sc)), Some(dv), Some(ds)) => {
                    dv.buf_mut().copy_from_host(stream, v)?;
                    ds.buf_mut().copy_from_host(stream, sc)?;
                }
                (None, _, _) => {}
                _ => return Err(format!("layer {l}: the body holds no compressor ring").into()),
            }
        }
        body.set_history(&set.before)?;
        stream.synchronize()?;
        Ok(())
    }

    /// The set's step on the injected state: its host half, the image's
    /// upload, then the chain with `observe` shown each seam.
    fn run_step(
        gpu: &Gpu,
        w: &bloomery_gpu::weights::Weights,
        body: &mut Body,
        head: &mut Head,
        set: &Set,
        observe: &mut dyn FnMut(&Gpu, Seam<'_>) -> Result<(), GpuError>,
    ) -> Result<(), GateError> {
        let input = body.decode_input(set.token, set.pos)?;
        body.refresh(gpu.stream(), &input)?;
        body.enqueue_observed(gpu, w, head, observe)?;
        gpu.stream().synchronize()?;
        Ok(())
    }

    // ------------------------------------------------------------ structure

    /// What the step leaves that a replay must reproduce: the logits and
    /// every layer's caches and compressor state.
    #[derive(PartialEq)]
    struct Outputs {
        logits: Vec<u32>,
        state: Vec<Vec<u16>>,
        comp: Vec<Vec<u32>>,
    }

    fn outputs(
        gpu: &Gpu,
        body: &mut Body,
        head: &Head,
        layers: usize,
    ) -> Result<Outputs, GateError> {
        let stream = gpu.stream();
        let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<u32>>();
        let mut state = Vec::new();
        let mut comp = Vec::new();
        for l in 0..layers {
            let b = body
                .state_mut(l)
                .ok_or_else(|| format!("the body does not run layer {l}"))?;
            state.push(b.ring.buf().to_host_vec(stream)?);
            for t in [b.rows, b.keys].into_iter().flatten() {
                state.push(t.buf().to_host_vec(stream)?);
            }
            for t in [b.values, b.scores].into_iter().flatten() {
                comp.push(bits(t.buf().to_host_vec(stream)?));
            }
        }
        Ok(Outputs {
            logits: bits(head.logits_to_host(gpu)?),
            state,
            comp,
        })
    }

    /// The step's predicted kernels: per layer the attention's (14, and on a
    /// compressor's layer its kv projection, its gate and pool above ratio 1
    /// or its row at ratio 1, and three for index keys; on an indexer layer
    /// its two projections and the score and top-k passes, and without a
    /// compressor the q8_1 of the normed input its weights read unless q_a
    /// or kv, being K-quants, already had the norm leave it; one more for the
    /// q8_1 of the heads when wo_a is a K-quant and one for wo_a's when wo_b
    /// is a q3_K/q4_K) and the MoE sub-layer's (its own count: ten with card
    /// experts, seven without, one more for a q8_1 shared down projection);
    /// the glue's (the broadcast, three and two per engram site and one more
    /// for the looked-up rows' q8_1 when wkv is a K-quant, the collapse and
    /// the head's four) and the gather — less one launch for each projection
    /// a Q3_K row join folds into another's (`join_projections`). The types
    /// come from `split`. Printed as the table G2 pins.
    fn predicted(split: &Split, hp: &Hparams, body: &Body) -> Result<(usize, usize), GateError> {
        let ty = |name: String| {
            split
                .find(&name)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| format!("{name} is not in the file"))
        };
        let kquant = |t: GgmlType| {
            matches!(
                t,
                GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
            )
        };
        let q8_1 = |t: GgmlType| matches!(t, GgmlType::Q3_K | GgmlType::Q4_K);
        let mut kinds: Vec<(String, usize, usize, usize)> = Vec::new();
        let mut attn_total = 0;
        let mut ffn_total = 0;
        for (l, k) in hp.layers.iter().enumerate() {
            let own = k
                .compressor
                .filter(|_| k.stream.is_some_and(|s| s.kv_source == l));
            let normed_q8_1 = q8_1(ty(names::attn_q_a(l))?) || q8_1(ty(names::attn_kv(l))?);
            // PIN(2026-09-24): `join_projections` folds each group of Q3_K projections of one
            // activation into one launch (ds41dense): kv and the indexer's weights into q_a's,
            // the indexer's query into q_b's, a gated compressor's gate into its kv.
            let q3k = |name: String| ty(name).map(|t| t == GgmlType::Q3_K);
            let proj_q3k = !k.indexer || q3k(names::indexer_proj(l))?;
            let join_qkv = q3k(names::attn_q_a(l))? && q3k(names::attn_kv(l))? && proj_q3k;
            let join_query =
                k.indexer && q3k(names::attn_q_b(l))? && q3k(names::indexer_attn_q_b(l))?;
            let join_kv_gate = own.is_some_and(|c| c.gated)
                && q3k(names::attn_compressor_kv(l))?
                && q3k(names::attn_compressor_gate(l))?;
            let folded = usize::from(join_qkv) * (1 + usize::from(k.indexer))
                + usize::from(join_query)
                + usize::from(join_kv_gate);
            let attn = 14 - folded
                + own.map_or(0, |c| 1 + if c.gated { 2 } else { 1 })
                + if k.index_keys { 3 } else { 0 }
                + match (k.indexer, own) {
                    (false, _) => 0,
                    (true, Some(_)) => 4,
                    (true, None) if normed_q8_1 => 4,
                    (true, None) => 5,
                }
                + usize::from(kquant(ty(names::attn_output_a(l))?))
                + usize::from(q8_1(ty(names::attn_output_b(l))?));
            let ffn = body
                .ffn_launches(l)
                .ok_or_else(|| format!("the ffn piece does not run layer {l}"))?;
            attn_total += attn;
            ffn_total += ffn;
            let name = format!(
                "{}{}{}{}",
                match (k.stream, own) {
                    (None, _) => "window".to_string(),
                    (Some(s), Some(_)) => format!("source-r{}", s.ratio),
                    (Some(s), None) => format!("reader-r{}", s.ratio),
                },
                if k.indexer { "+indexer" } else { "" },
                if k.engram.is_some() { "+engram" } else { "" },
                if ffn > 7 { "+card" } else { "+host-only" }
            );
            match kinds.iter_mut().find(|e| e.0 == name) {
                Some(e) => e.3 += 1,
                None => kinds.push((name, attn, ffn, 1)),
            }
        }
        let sites = hp.engram()?.layer_ids.len();
        let mut wkv_q8_1 = 0;
        for &l in &hp.engram()?.layer_ids {
            wkv_q8_1 += usize::from(q8_1(ty(names::engram_wkv(l))?));
        }
        let glue = 1 + 3 * sites + wkv_q8_1 + 2 * sites + 1 + 4;
        for (name, a, f, n) in &kinds {
            println!(
                "predict kind={name} layers={n} attn_kernels={a} ffn_kernels={f} memops=2 \
                 nodes_per_layer={}",
                a + f + 2
            );
        }
        let kernels = 1 + attn_total + ffn_total + glue;
        // PIN(2026-09-25): the engram rows reach the card after the launch — one flag wait and
        // one copy before the first site's token-only work (`RowsArrival`), where the step had
        // two batches per layer and no copy.
        let memops = 2 * hp.n_layer + ARRIVALS;
        println!(
            "predict step: gather 1 + attn {attn_total} + ffn {ffn_total} + glue {glue} = {kernels} \
             kernels, {memops} memops (go and wait per layer, {ARRIVALS} rows wait), {ARRIVALS} \
             memcpy (the rows' arrival), 0 other"
        );
        Ok((kernels, memops))
    }

    fn structure(
        m: &mut Deepseek41Model,
        head: &mut Head,
        split: &Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        let (want_k, want_m) = {
            let (_, _, body) = m.body_parts("structure")?;
            predicted(split, hp, body)?
        };
        let nodes = m.capture_step()?;
        let list = m.step_graph_nodes()?;
        let ([kernels, memops, memcpy], other) = count_kinds(
            &list,
            [
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
            ],
        );
        let ok_nodes = kernels == want_k && memops == want_m && memcpy == ARRIVALS && other == 0;
        println!(
            "graph: nodes={nodes} kernels={kernels} memops={memops} memcpy={memcpy} other={other} \
             predicted kernels={want_k} memops={want_m} memcpy={ARRIVALS}: {}",
            verdict(ok_nodes)
        );

        let mut all_same = true;
        for name in [STEP4, D1, D2] {
            let set = open_set(split, hp, name)?;
            let state = set_state(&set, hp)?;
            let (gpu, w, body) = m.body_parts("structure")?;
            body.set_indexer_top_k(gpu, set.top_k)?;
            inject(gpu, body, &set, &state)?;
            run_step(gpu, w, body, head, &set, &mut |_, _| Ok(()))?;
            let eager = outputs(gpu, body, head, hp.n_layer)?;
            let graph = gpu.capture(|_| body.enqueue_chain(gpu, w, head))?;
            inject(gpu, body, &set, &state)?;
            let input = body.decode_input(set.token, set.pos)?;
            body.refresh(gpu.stream(), &input)?;
            graph.launch(gpu.stream())?;
            body.serve_replay()?;
            gpu.stream().synchronize()?;
            let replay = outputs(gpu, body, head, hp.n_layer)?;
            let same = replay == eager;
            println!(
                "graph set={} replay_bit_identical_to_eager={same} (logits, {} state buffers, {} \
                 compressor buffers): {}",
                set.name,
                eager.state.len(),
                eager.comp.len(),
                verdict(same)
            );
            drop(graph);
            body.set_indexer_top_k(gpu, hp.indexer.top_k)?;
            all_same &= same;
        }

        let (ok_topo, ok_shadow, ok_arrival) = {
            let (gpu, w, body) = m.body_parts("structure")?;
            let graph = step_graph(gpu, w, body, head)?;
            let ok_topo = hc_branch(&graph, &body.layers());
            let ok_arrival = rows_arrival(&graph, body, split, hp)?;
            (
                ok_topo,
                shadow_sets(&graph.nodes, body, split, hp)?,
                ok_arrival,
            )
        };

        let (gpu, _, body) = m.body_parts("structure")?;
        piece_modules(gpu, body, hp, split)?;
        Ok(ok_nodes && all_same && ok_topo && ok_shadow && ok_arrival)
    }

    /// Flag waits a step holds: the engram rows' arrival, one per step.
    const ARRIVALS: usize = 1;

    /// Whether node `n` is a go or a wait batch of the host tier — a memop
    /// batch other than the rows' flag wait.
    fn host_batch(n: &StepNode) -> bool {
        matches!(n, StepNode::Memop(c) if *c != FLAG_WAIT_OPS)
    }

    /// The engram rows reach the card after the launch, and before anything
    /// reads them: the step holds one flag wait ([`FLAG_WAIT_OPS`] ops), the
    /// one copy depends on it alone, and the first site's token-only work
    /// (its rows' decode first) depends on the copy — all in the host-leg
    /// shadow of the layer before that site, off the critical path; no rows
    /// decode runs before the copy.
    fn rows_arrival(
        g: &Captured,
        body: &Body,
        split: &Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        let layers = body.layers();
        let at = regions(&g.nodes, &layers);
        let waits: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| g.nodes[i] == StepNode::Memop(FLAG_WAIT_OPS))
            .collect();
        let copies: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| matches!(&g.nodes[i], StepNode::Other(k) if k == "memcpy"))
            .collect();
        let first = *hp
            .engram()?
            .layer_ids
            .iter()
            .min()
            .ok_or("the file has no engram site")?;
        let decode = shadow::engram_kv_kernels(split, first)?[0];
        let decodes: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| {
                g.kernel(i)
                    .is_some_and(|k| k.starts_with("ds41_glue_engram_rows"))
            })
            .collect();
        let (ok, how) = match (waits.as_slice(), copies.as_slice()) {
            (&[w], &[c]) => {
                let next = c + 1;
                let ok_edges = g.deps[c] == [w] && g.deps[next] == [c];
                let ok_next = g.kernel(next) == Some(decode);
                let ok_where = at[w] == Region::Batch
                    && at[c] == Region::Shadow(first - 1)
                    && at[next] == Region::Shadow(first - 1);
                let ok_before = decodes.first().is_some_and(|&d| d > c);
                let how = format!(
                    "wait node {w} (depends on [{}]), copy node {c} (depends on [{}]), then node \
                     {next} [{}] (depends on [{}]); copy in {:?}, want the shadow of layer {} (the \
                     layer before site {first}); first rows decode at node {}",
                    labels(g, &g.deps[w]),
                    labels(g, &g.deps[c]),
                    label(g, next),
                    labels(g, &g.deps[next]),
                    at[c],
                    first - 1,
                    decodes
                        .first()
                        .map_or_else(|| "none".to_string(), ToString::to_string),
                );
                (ok_edges && ok_next && ok_where && ok_before, how)
            }
            _ => (
                false,
                format!(
                    "{} flag waits at [{}], {} copies at [{}], want one of each",
                    waits.len(),
                    bloomery_gpu_gates::comma_list(&waits),
                    copies.len(),
                    bloomery_gpu_gates::comma_list(&copies)
                ),
            ),
        };
        println!("arrival: engram rows {how}: {}", verdict(ok));
        Ok(ok)
    }

    // --------------------------------------------------------- shadow sets

    /// Where a node of the step sits against the layers' go and wait
    /// batches.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Region {
        /// Before the first layer's go.
        Head,
        /// Between layer `l`'s go and its wait: the host leg's shadow.
        Shadow(usize),
        /// After layer `l`'s wait, before the next go (or the step's end).
        After(usize),
        /// A go or a wait batch itself.
        Batch,
    }

    /// The step's nodes in enqueue order and its edges: the engine's own
    /// chain captured on its stream (the host tier sees a capture and serves
    /// nothing), read back by [`capture_order`] and never instantiated.
    fn step_graph(
        gpu: &Gpu,
        w: &Weights,
        body: &mut Body,
        head: &mut Head,
    ) -> Result<Captured, GateError> {
        capture_order(gpu.stream(), || body.enqueue_chain(gpu, w, head))
    }

    /// A node's name as the topology lines print it.
    fn label(g: &Captured, i: usize) -> String {
        match &g.nodes[i] {
            StepNode::Kernel(k) => k.clone(),
            StepNode::Memop(c) => format!("memop({c})"),
            StepNode::Other(k) => k.clone(),
        }
    }

    fn labels(g: &Captured, ids: &[usize]) -> String {
        ids.iter()
            .map(|&i| label(g, i))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The attention sub-layer's `ds41_hc_pre` runs on a branch beside the
    /// sub-layer's body, and nothing else in the step does. Per layer, the
    /// one `ds41_hc_pre` between the previous layer's wait (the step's head
    /// for the first layer) and the layer's go is the attention's (the MoE
    /// sub-layer's opens the go's shadow). It has one dependency, which the
    /// body's first kernel shares and depends on alone, and one dependent, a
    /// `ds41_hc_post` before the go with two dependencies. The step holds
    /// exactly one such fork and one such join per layer and no node has
    /// more than two edges on a side.
    fn hc_branch(g: &Captured, layers: &std::ops::Range<usize>) -> bool {
        const PRE: &str = "ds41_hc_pre";
        const POST: &str = "ds41_hc_post";
        let mem: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| host_batch(&g.nodes[i]))
            .collect();
        if mem.len() != 2 * layers.len() {
            println!(
                "topology: {} memop batches for {} layers: FAIL",
                mem.len(),
                layers.len()
            );
            return false;
        }
        let mut ok = true;
        for (k, l) in layers.clone().enumerate() {
            let start = if k == 0 { 0 } else { mem[2 * k - 1] + 1 };
            let pres: Vec<usize> = (start..mem[2 * k])
                .filter(|&i| g.kernel(i) == Some(PRE))
                .collect();
            let &[a] = pres.as_slice() else {
                println!(
                    "topology layer={l}: {} {PRE} between the previous wait and the go, want 1 \
                     (the attention's): FAIL",
                    pres.len()
                );
                ok = false;
                continue;
            };
            let deps = &g.deps[a];
            let after = &g.dependents[a];
            let fork = deps.first().copied();
            let body_first = fork.and_then(|p| match g.dependents[p].as_slice() {
                &[x, b] if deps.len() == 1 && x == a => Some(b),
                _ => None,
            });
            let post = match after.as_slice() {
                &[h] if g.kernel(h) == Some(POST) => Some(h),
                _ => None,
            };
            let ok_body = body_first.zip(fork).is_some_and(|(b, p)| g.deps[b] == [p]);
            let ok_post = post.is_some_and(|h| g.deps[h].len() == 2 && h < mem[2 * k]);
            let good = ok_body && ok_post;
            println!(
                "topology layer={l} attention {PRE} (node {a}) depends on [{}], is depended on by \
                 [{}]; body's first kernel [{}] depends on [{}]; {POST} depends on [{}]: {}",
                labels(g, deps),
                labels(g, after),
                body_first.map_or_else(|| "none beside it".to_string(), |b| label(g, b)),
                body_first.map_or_else(String::new, |b| labels(g, &g.deps[b])),
                post.map_or_else(|| "none".to_string(), |h| labels(g, &g.deps[h])),
                verdict(good)
            );
            ok &= good;
        }
        let forks = g.dependents.iter().filter(|d| d.len() > 1).count();
        let joins = g.deps.iter().filter(|d| d.len() > 1).count();
        let wide = g
            .deps
            .iter()
            .chain(&g.dependents)
            .filter(|d| d.len() > 2)
            .count();
        let ok_count = forks == layers.len() && joins == layers.len() && wide == 0;
        println!(
            "topology: {} nodes, {forks} forks and {joins} joins (one each per attention \
             sub-layer: {}), {wide} nodes with more than two edges on a side: {}",
            g.nodes.len(),
            layers.len(),
            verdict(ok_count)
        );
        ok && ok_count
    }

    /// Each node's region: the host tier's batch `2i` is layer `layers[i]`'s
    /// go and `2i + 1` its wait; the rows' flag wait is a batch that opens no
    /// region.
    fn regions(nodes: &[StepNode], layers: &std::ops::Range<usize>) -> Vec<Region> {
        let mut seen = 0usize;
        nodes
            .iter()
            .map(|n| {
                if matches!(n, StepNode::Memop(_)) {
                    seen += usize::from(host_batch(n));
                    return Region::Batch;
                }
                match seen {
                    0 => Region::Head,
                    k if k % 2 == 1 => Region::Shadow(layers.start + k / 2),
                    k => Region::After(layers.start + k / 2 - 1),
                }
            })
            .collect()
    }

    /// Q1: the kernels between each layer's go and its wait — the host leg's
    /// shadow — are exactly the table's, in order: the MoE piece's own
    /// ([`shadow::shadow_kernels`], with card experts or without) and, on
    /// the layer before each engram site, that site's token-only work
    /// ([`shadow::engram_kv_kernels`]), which reads the step image alone and is first read
    /// by the site layer's engram step. Also: no engram site's token-only
    /// work runs outside a shadow. Needs the overlap lever on (the default):
    /// off, each wait sits right after its go and every shadow is empty.
    fn shadow_sets(
        nodes: &[StepNode],
        body: &Body,
        split: &Split,
        hp: &Hparams,
    ) -> Result<bool, GateError> {
        if !bloomery_gpu::hybrid::levers()?.overlap {
            return Err(
                "the shadow-set check needs BLOOMERY_HYBRID_OVERLAP unset or 1: with the \
                        lever off each wait sits right after its go and no layer has a shadow"
                    .into(),
            );
        }
        let layers = body.layers();
        let at = regions(nodes, &layers);
        let memops: Vec<u32> = nodes
            .iter()
            .filter_map(|n| match n {
                StepNode::Memop(c) if host_batch(n) => Some(*c),
                _ => None,
            })
            .collect();
        let others: Vec<&str> = nodes
            .iter()
            .filter_map(|n| match n {
                StepNode::Other(k) => Some(k.as_str()),
                _ => None,
            })
            .collect();
        // The rows' arrival copy is the one other node ([`rows_arrival`]).
        let ok_shape = memops.len() == 2 * layers.len() && others == ["memcpy"; ARRIVALS];
        println!(
            "shadow: {} nodes in enqueue order, {} memop batches (ops: go {:?}, wait {:?}), \
             other [{}]: {}",
            nodes.len(),
            memops.len(),
            memops
                .iter()
                .step_by(2)
                .collect::<std::collections::BTreeSet<_>>(),
            memops
                .iter()
                .skip(1)
                .step_by(2)
                .collect::<std::collections::BTreeSet<_>>(),
            others.join(","),
            verdict(ok_shape)
        );
        if !ok_shape {
            return Ok(false);
        }

        let sites = &hp.engram()?.layer_ids;
        let mut kv_lists: Vec<Vec<&str>> = Vec::new();
        for &site in sites {
            let k = shadow::engram_kv_kernels(split, site)?;
            if !kv_lists.contains(&k) {
                kv_lists.push(k);
            }
        }
        let mut table: Vec<(Vec<&str>, Vec<usize>, bool)> = Vec::new();
        let mut ok_table = true;
        for l in layers.clone() {
            let launches = body
                .ffn_launches(l)
                .ok_or_else(|| format!("the ffn piece does not run layer {l}"))?;
            let mut want = shadow::shadow_kernels(split, l, launches > 7)?;
            if sites.contains(&(l + 1)) {
                want.extend(shadow::engram_kv_kernels(split, l + 1)?);
            }
            let got: Vec<&str> = nodes
                .iter()
                .zip(&at)
                .filter(|&(_, r)| *r == Region::Shadow(l))
                .filter_map(|(n, _)| match n {
                    StepNode::Kernel(k) => Some(k.as_str()),
                    _ => None,
                })
                .collect();
            let same = got == want;
            if !same {
                println!(
                    "shadow layer={l} got=[{}] want=[{}]: FAIL",
                    got.join(","),
                    want.join(",")
                );
            }
            ok_table &= same;
            match table.iter_mut().find(|e| e.0 == want) {
                Some(e) => {
                    e.1.push(l);
                    e.2 &= same;
                }
                None => table.push((want, vec![l], same)),
            }
        }
        for (want, ls, same) in &table {
            println!(
                "shadow set=[{}] layers={}: {}",
                want.join(","),
                bloomery_gpu_gates::comma_list(ls),
                verdict(*same)
            );
        }

        // Each engram site's token-only work, found as the three kernels in a
        // row, and where it runs.
        let names: Vec<Option<&str>> = nodes
            .iter()
            .map(|n| match n {
                StepNode::Kernel(k) => Some(k.as_str()),
                _ => None,
            })
            .collect();
        let mut found = Vec::new();
        let mut kv_len = 0;
        for kv in &kv_lists {
            kv_len = kv.len();
            for i in 0..names.len().saturating_sub(kv.len() - 1) {
                if kv
                    .iter()
                    .zip(&names[i..])
                    .all(|(want, got)| *got == Some(*want))
                {
                    found.push(at[i]);
                }
            }
        }
        let outside = found
            .iter()
            .filter(|r| !matches!(r, Region::Shadow(_)))
            .count();
        let ok_engram = found.len() == sites.len() && outside == 0;
        println!(
            "shadow engram token-only work: {} of {} sites found, at [{}]; kernels outside every \
             shadow: {}: {}",
            found.len(),
            sites.len(),
            found
                .iter()
                .map(|r| match r {
                    Region::Head => "before the first go (critical path)".to_string(),
                    Region::Shadow(l) => format!("layer {l}'s shadow"),
                    Region::After(l) => format!("after layer {l}'s wait (critical path)"),
                    Region::Batch => "a memop batch".to_string(),
                })
                .collect::<Vec<_>>()
                .join("; "),
            outside * kv_len,
            verdict(ok_engram)
        );
        Ok(ok_table && ok_engram)
    }

    /// The device memory a second set of pieces takes beyond its scratch:
    /// what each piece's own kernel modules cost the card, since the pieces
    /// load theirs each. Printed, not pinned.
    fn piece_modules(gpu: &Gpu, body: &Body, hp: &Hparams, split: &Split) -> Result<(), GateError> {
        let planner = Planner::from_file(split, hp, CTX_MAX)?;
        let free = || -> Result<i128, GateError> { Ok(gpu.mem_info()?.0 as i128) };
        let f0 = free()?;
        let attn = AttnChain::new(gpu, hp, body.layers(), body.image().layout(), &planner)?;
        let f1 = free()?;
        let ffn = FfnPiece::new(gpu, hp, body.slot_map())?;
        let f2 = free()?;
        let glue = Glue::new(gpu, hp, body.image().layout())?;
        let f3 = free()?;
        let (a, f, g) = (
            attn.device_bytes() as i128,
            ffn.device_bytes() as i128,
            glue.device_bytes() as i128,
        );
        println!(
            "modules: a second attn piece took {} B ({a} scratch, {} beyond), ffn {} B ({f}, {}), \
             glue {} B ({g}, {}) — the beyond is each piece's loaded modules and allocation rounding",
            f0 - f1,
            f0 - f1 - a,
            f1 - f2,
            f1 - f2 - f,
            f2 - f3,
            f2 - f3 - g
        );
        Ok(())
    }

    // ----------------------------------------------------------------- sets

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Stage {
        Attn,
        Engram,
        Ffn,
    }

    /// What one seam left, read back.
    struct SeamOut {
        stage: Stage,
        layer: usize,
        streams: Vec<f32>,
        fold: Option<Vec<f32>>,
        router: Option<RouterHost>,
        /// At an attention seam of a stream's layer: the list its attention
        /// read, every entry.
        list: Option<Vec<u32>>,
        /// At an indexer layer's attention seam: the indexer's taps.
        index: Option<IndexHost>,
    }

    /// An indexer layer's query after its transform, scaled weights and
    /// scores.
    struct IndexHost {
        q: Vec<f32>,
        w: Vec<f32>,
        scores: Vec<f32>,
    }

    struct RouterHost {
        ids: Vec<u32>,
        weights: Vec<f32>,
        logits: Vec<f32>,
        probs: Vec<f32>,
    }

    /// A reader of every seam into `out`; `indexer[l]` says layer `l` runs
    /// the indexer, whose taps its attention seam reads.
    fn collect<'a>(
        out: &'a mut Vec<SeamOut>,
        indexer: &'a [bool],
    ) -> impl FnMut(&Gpu, Seam<'_>) -> Result<(), GpuError> + 'a {
        move |gpu: &Gpu, seam: Seam<'_>| {
            let stream = gpu.stream();
            stream.synchronize()?;
            let o = match seam {
                Seam::Attn {
                    layer,
                    streams,
                    fold,
                    taps,
                    list,
                } => SeamOut {
                    stage: Stage::Attn,
                    layer,
                    streams: streams.to_host_vec(stream)?,
                    fold: Some(fold.to_host_vec(stream)?),
                    router: None,
                    list: list.map(|l| l.to_host_vec(stream)).transpose()?,
                    index: match (indexer.get(layer), taps.select) {
                        (Some(true), Some(t)) => Some(IndexHost {
                            q: t.q.to_host_vec(stream)?,
                            w: t.w.to_host_vec(stream)?,
                            scores: t.scores.to_host_vec(stream)?,
                        }),
                        _ => None,
                    },
                },
                Seam::Engram {
                    layer,
                    streams,
                    fold,
                } => SeamOut {
                    stage: Stage::Engram,
                    layer,
                    streams: streams.to_host_vec(stream)?,
                    fold: Some(fold.to_host_vec(stream)?),
                    router: None,
                    list: None,
                    index: None,
                },
                Seam::Ffn {
                    layer,
                    streams,
                    fold,
                    taps,
                } => SeamOut {
                    stage: Stage::Ffn,
                    layer,
                    streams: streams.to_host_vec(stream)?,
                    fold: fold.map(|f| f.to_host_vec(stream)).transpose()?,
                    router: Some(RouterHost {
                        ids: taps.router.ids.to_host_vec(stream)?,
                        weights: taps.router.weights.to_host_vec(stream)?,
                        logits: taps.router.logits.to_host_vec(stream)?,
                        probs: taps.router.probs.to_host_vec(stream)?,
                    }),
                    list: None,
                    index: None,
                },
            };
            out.push(o);
            Ok(())
        }
    }

    fn sets(
        m: &mut Deepseek41Model,
        head: &mut Head,
        split: &Split,
        hp: &Hparams,
        names: &[&'static str],
    ) -> Result<bool, GateError> {
        let mut pass = true;
        let mut worst = 0.0f64;
        let indexer: Vec<bool> = hp.layers.iter().map(|k| k.indexer).collect();
        for &name in names {
            let set = open_set(split, hp, name)?;
            let state = set_state(&set, hp)?;
            let (gpu, w, body) = m.body_parts("sets")?;
            body.set_indexer_top_k(gpu, set.top_k)?;
            inject(gpu, body, &set, &state)?;
            let mut seams = Vec::with_capacity(3 * hp.n_layer);
            run_step(gpu, w, body, head, &set, &mut collect(&mut seams, &indexer))?;
            body.set_indexer_top_k(gpu, hp.indexer.top_k)?;
            pass &= check_ids(&set, hp, body)?;
            let (ok, ratio, acc) = check_seams(&set, split, hp, &seams)?;
            pass &= ok;
            worst = worst.max(ratio);
            pass &= check_head(&set, &head.logits_to_host(gpu)?, acc)?;
        }
        println!("sets: largest measured/bound ratio pinned over both sets {worst:.3} (pin {Z})");
        Ok(pass)
    }

    /// Each engram site's host row ids against the set's `engram_rows-L`.
    fn check_ids(set: &Set, hp: &Hparams, body: &Body) -> Result<bool, GateError> {
        let mut all = true;
        for (s, &l) in hp.engram()?.layer_ids.iter().enumerate() {
            let want = ref_ints(
                &set.man,
                &format!("engram_rows-{l}"),
                0,
                RowKind::Input,
                Layout::Flat,
            )?;
            let got: Vec<i64> = body
                .step_rows()
                .ids(0, s)
                .iter()
                .map(|&v| i64::from(v))
                .collect();
            let same = got == want;
            println!(
                "ids set={} L={l}: {} host row ids equal to engram_rows-{l}: {same} {}",
                set.name,
                got.len(),
                verdict(same)
            );
            all &= same;
        }
        Ok(all)
    }

    /// Relative variance of ik's q8_2 rounding of `x`, exact.
    fn v_q8_2(x: &[f32]) -> f64 {
        let (q, d) = ik_q8_2::quantize(x);
        let (mut e, mut s) = (0.0f64, 0.0f64);
        for (c, &v) in x.iter().enumerate() {
            let r = f64::from(v) - f64::from(q[c]) * f64::from(d[c / QK]);
            e += r * r;
            s += f64::from(v) * f64::from(v);
        }
        if s > 0.0 { e / s } else { 0.0 }
    }

    /// Relative variance of a q8 quantizer with blocks of `block` values,
    /// `d = amax/127`, `d²/12` a value.
    fn v_q8(x: &[f32], block: usize) -> f64 {
        let (mut e, mut s) = (0.0f64, 0.0f64);
        for b in x.chunks(block) {
            let amax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = f64::from(amax) / 127.0;
            e += b.len() as f64 * d * d / 12.0;
            s += b.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>();
        }
        if s > 0.0 { e / s } else { 0.0 }
    }

    fn sumsq(x: &[f32]) -> f64 {
        x.iter().map(|&v| f64::from(v) * f64::from(v)).sum()
    }

    /// `|a − b| / |b|` over the whole vector.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        let d: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum();
        let n = sumsq(b);
        if a.len() != b.len() {
            f64::INFINITY
        } else if n > 0.0 {
            (d / n).sqrt()
        } else if d == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    }

    /// The HC_PRE a sub-layer runs, found by its scale: its result (`pre`,
    /// `post`, `comb`) and the normed streams its q3_K mix reads.
    fn hc_pre(man: &RefManifest, scale: &str) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let (at_hc, hc_row) = man
            .tensors
            .iter()
            .enumerate()
            .find(|(_, r)| r.op == "HC_PRE" && r.src1.as_deref() == Some(scale))
            .ok_or_else(|| format!("no HC_PRE reads {scale}"))?;
        let (at_mix, mix_row) = man.last_before(at_hc, hc_row.src0.as_deref())?;
        let (_, hn_row) = man.last_before(at_mix, mix_row.src1.as_deref())?;
        Ok((f32s(man, hc_row)?, f32s(man, hn_row)?))
    }

    /// Relative spread of the four streams about their mean, `κ² = Σ_j
    /// |s_j − s̄|² / Σ_j |s_j|²`: HC_POST's mix has rows that sum to one, so
    /// an error in it moves the streams only by their differences.
    fn spread2(s: &[f32]) -> f64 {
        let n = s.len() / 4;
        let mut d = 0.0f64;
        for i in 0..n {
            let v = [s[i], s[i + n], s[i + 2 * n], s[i + 3 * n]].map(f64::from);
            let m = v.iter().sum::<f64>() / 4.0;
            d += v.iter().map(|x| (x - m) * (x - m)).sum::<f64>();
        }
        d / sumsq(s)
    }

    /// A sub-layer's own term of the envelope, `(v_body·g², v_mix·(κ² +
    /// g²))`, from ik's nodes (module doc): the body's rule differences
    /// entering the streams at the sub-layer's share `g`, and its mixes'
    /// through `comb` (the streams' spread `κ`) and `post` (`g`). `v_swap` is
    /// the relative variance a list swap adds at the attention's output.
    fn term(
        set: &Set,
        hp: &Hparams,
        stage: Stage,
        l: usize,
        v_swap: f64,
    ) -> Result<(f64, f64), GateError> {
        let man = &set.man;
        let n = |stem: &str| format!("{stem}-{l}");
        let g2 = |post: &[f32], y: &[f32], s: &[f32]| {
            let p: f64 = post.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            p * sumsq(y) / sumsq(s)
        };
        let mix = |hn: &[f32]| v_q8(hn, 256) + v_q8(hn, 128);
        match stage {
            Stage::Attn => {
                let (hc, hn) = hc_pre(man, &names::hc_attn_scale(l))?;
                let x = named(man, &n("attn_norm"))?;
                let mut v = 2.0 * v_q8_2(&x)
                    + v_q8_2(&named(man, &n("qr_norm"))?)
                    + v_q8_2(&named(man, &n("attn"))?)
                    + v_q8_2(&named(man, &n("attn_wo_a"))?);
                let k = &hp.layers[l];
                if k.compressor.is_some() && k.stream.is_some_and(|s| s.kv_source == l) {
                    v += v_q8(&x, 256) + v_q8(&x, 128);
                }
                let fattn = named(man, &n("fattn"))?;
                let band = if man.tensor(&n("mask_to_idx"), 0).is_ok() {
                    ATTN_BAND_IQK
                } else {
                    ATTN_BAND_GENERIC
                };
                let amax = fattn.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
                let rms = (sumsq(&fattn) / fattn.len() as f64).sqrt();
                v += (band * f64::from(amax) / (BAND_SIGMAS * rms)).powi(2) + v_swap;
                let y = named(man, &n("attn_out"))?;
                let s = named(man, &n("hc_attn_post"))?;
                let g = g2(&hc[4..8], &y, &s);
                Ok((v * g, mix(&hn) * (spread2(&hn) + g)))
            }
            Stage::Ffn => {
                let (hc, hn) = hc_pre(man, &names::hc_ffn_scale(l))?;
                let x = named(man, &n("ffn_norm"))?;
                let h = named(man, &n("ffn_moe_gate_par"))?;
                let v = v_q8(&x, 256)
                    + v_q8(&x, 128)
                    + v_q8_2(&h)
                    + v_q8(&h, 128)
                    + v_q8_2(&x)
                    + v_q8_2(&named(man, &n("ffn_up_gate"))?);
                let y = named(man, &n("ffn_out"))?;
                let s = named(man, &n("l_out"))?;
                let g = g2(&hc[4..8], &y, &s);
                Ok((v * g, mix(&hn) * (spread2(&hn) + g)))
            }
            Stage::Engram => {
                let rows = named(man, &n("engram_embd"))?;
                let v = v_q8_2(&rows);
                let out = named(man, &n("engram_out"))?;
                let before = named(man, &format!("l_out-{}", l - 1))?;
                let add: Vec<f32> = out.iter().zip(&before).map(|(a, b)| a - b).collect();
                Ok((v * sumsq(&add) / sumsq(&out), 0.0))
            }
        }
    }

    /// ik's streams and fold where the seam wrote ours. After a MoE
    /// sub-layer ik folds when the next attention's input reads `l_out-L`;
    /// into an engram layer and after the last layer it does not.
    fn seam_ref(
        man: &RefManifest,
        stage: Stage,
        l: usize,
    ) -> Result<(Vec<f32>, Option<Vec<f32>>), GateError> {
        Ok(match stage {
            Stage::Attn => (
                named(man, &format!("hc_attn_post-{l}"))?,
                Some(named(man, &format!("hc_ffn_pre-{l}"))?),
            ),
            Stage::Engram => (
                named(man, &format!("engram_out-{l}"))?,
                Some(named(man, &format!("hc_attn_pre-{l}"))?),
            ),
            Stage::Ffn => {
                let l_out = format!("l_out-{l}");
                let fold = match man.tensor(&format!("hc_attn_pre-{}", l + 1), 0) {
                    Ok(r) if r.src0.as_deref() == Some(l_out.as_str()) => Some(f32s(man, r)?),
                    _ => None,
                };
                (named(man, &l_out)?, fold)
            }
        })
    }

    /// Every seam against ik's, in step order. The pin at seam `k` is
    /// local: `r_k ≤ Z·sqrt(r_{k−1}² + s_k²)` — the relative distance the
    /// streams carried in, measured at the seam before, and this sub-layer's
    /// own term (`s_k² = v_body·g² + v_mix·(κ² + g²)`, [`term`]). The pins
    /// hold up to the first MoE sub-layer whose routed set differs from ik's
    /// by a near tie: from there the step follows another, legitimate path
    /// the dump does not hold, and every later line — the head's too — is a
    /// diagnostic. Returns the pass, the largest ratio pinned, and the
    /// carried distance at the head (`None` past a divergence).
    fn check_seams(
        set: &Set,
        split: &Split,
        hp: &Hparams,
        seams: &[SeamOut],
    ) -> Result<(bool, f64, Option<f64>), GateError> {
        let want = hp.n_layer * 2 + hp.engram()?.layer_ids.len();
        let mut pass = seams.len() == want;
        println!(
            "seams set={}: {} seams, want {want}: {}",
            set.name,
            seams.len(),
            verdict(pass)
        );
        let (mut r_prev, mut worst) = (0.0f64, 0.0f64);
        let mut diverged: Option<usize> = None;
        let mut input_ok = true;
        // The relative distance of the fold the next sub-layer reads.
        let mut fold_prev = 0.0f64;
        let mut sel = SelState::default();
        for o in seams {
            let (s_ref, f_ref) = seam_ref(&set.man, o.stage, o.layer)?;
            let mut v_swap = 0.0;
            if o.stage == Stage::Attn
                && let Some((own, vs_ik, v)) =
                    check_selection(set, split, hp, o, fold_prev, &mut sel)?
            {
                pass &= own && vs_ik;
                v_swap = v;
            }
            if let (Some(a), Some(b)) = (&o.fold, &f_ref) {
                fold_prev = rel(a, b);
            }
            let (body_term, mix_term) = term(set, hp, o.stage, o.layer, v_swap)?;
            let r = rel(&o.streams, &s_ref);
            let bound = (r_prev * r_prev + body_term + mix_term).sqrt();
            let ratio = if bound > 0.0 {
                r / bound
            } else {
                f64::INFINITY
            };
            let form_ok = o.fold.is_some() == f_ref.is_some();
            let fold = match (&o.fold, &f_ref) {
                (Some(a), Some(b)) => format!("{:.3e}", rel(a, b)),
                (None, None) => "-".to_string(),
                _ => "form-differs".to_string(),
            };
            let flip = match &o.router {
                Some(rt) if diverged.is_none() => {
                    let (ok, set_differs) = check_router(set, split, hp, o.layer, rt, input_ok)?;
                    pass &= ok;
                    set_differs
                }
                _ => false,
            };
            if flip {
                diverged = Some(o.layer);
            }
            let ok = r.is_finite() && ratio <= Z;
            let pinned = diverged.is_none();
            let tag = if pinned {
                verdict(ok && form_ok)
            } else if flip {
                "diverges here (near tie): diagnostic from here on"
            } else {
                "diagnostic"
            };
            println!(
                "envelope set={} L={} {:?} streams_rel={r:.3e} carried={r_prev:.3e} bound={bound:.3e} \
                 ratio={ratio:.3} (body {:.3e} mix {:.3e}) fold_rel={fold}: {tag}",
                set.name,
                o.layer,
                o.stage,
                body_term.sqrt(),
                mix_term.sqrt()
            );
            if pinned {
                pass &= ok && form_ok;
                worst = worst.max(ratio);
            } else {
                pass &= form_ok;
            }
            if o.stage == Stage::Attn {
                input_ok = ok;
            }
            r_prev = r;
        }
        if let Some(l) = diverged {
            println!(
                "set {}: the step leaves ik's path at layer {l}'s routing (a near tie); the seams \
                 after it and the head are diagnostics",
                set.name
            );
        }
        Ok((pass, worst, diverged.is_none().then_some(r_prev)))
    }

    // ------------------------------------------------------------ selection

    /// What the selection pins carry from one attention seam to the next,
    /// per plan stream: the list its indexer layer wrote, and the relative
    /// spread of the index key its key source wrote this step.
    #[derive(Default)]
    struct SelState {
        lists: Vec<Option<Vec<u32>>>,
        key_rho: Vec<Option<f64>>,
    }

    impl SelState {
        fn slot<T>(v: &mut Vec<Option<T>>, s: usize) -> &mut Option<T> {
            if v.len() <= s {
                v.resize_with(s + 1, || None);
            }
            &mut v[s]
        }
    }

    /// `γ(n) = n·u / (1 − n·u)`.
    fn gamma(n: usize) -> f64 {
        let nu = n as f64 * U;
        nu / (1.0 - nu)
    }

    /// The order-preserving key of a score (the kernels' `order_key`).
    fn key_of(v: f32) -> u32 {
        let b = v.to_bits();
        let b = if b == 0x8000_0000 { 0 } else { b };
        if b & 0x8000_0000 != 0 {
            !b
        } else {
            b | 0x8000_0000
        }
    }

    /// The `k` rows of `scores` with the largest keys, ties to the lower row,
    /// ascending: the list the top-k pass writes (`gate_deepseek41_index.rs`
    /// `exact_top`).
    fn exact_top(scores: &[f32], k: usize) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            key_of(scores[b as usize])
                .cmp(&key_of(scores[a as usize]))
                .then(a.cmp(&b))
        });
        let mut top = idx[..k].to_vec();
        top.sort_unstable();
        top
    }

    /// ik's exact scores of rows `0 .. n` and the band each of ours may sit
    /// from it [derived]: `s_t = Σ_h w_h · relu(q_h·k_t)` in f64 at ik's query,
    /// scaled weights and keys; `β_t = Z_TIE·σ_t + ε_t`. `σ_t` is first order
    /// in three independent input errors — each weight off by `ρ_w·rms(w)`,
    /// each query value by `ρ_q·rms(q)` (so `q_h·k_t` by `ρ_q·rms(q)·|k_t|`),
    /// and on the row `written` this step, each key value by `ρ_k·rms(k_t)` —
    /// through the heads whose dot is positive. `ε_t` is the two sides'
    /// roundings: ik's 128-long f32 dots and 32-head sum (`γ(128)`, `γ(32)`)
    /// and our tensor-core dot ([`TC_UNITS`]) over the products' magnitude.
    #[allow(
        clippy::too_many_arguments,
        reason = "one band: ik's three inputs, the three input spreads and the written row"
    )]
    fn score_band(
        q: &[f32],
        w: &[f32],
        keys: &[u16],
        n: usize,
        rho_q: f64,
        rho_w: f64,
        rho_k: f64,
        written: Option<usize>,
    ) -> (Vec<f64>, Vec<f64>) {
        let rms = |v: &[f32]| (sumsq(v) / v.len() as f64).sqrt();
        let (dq, dw) = (rho_q * rms(q), rho_w * rms(w));
        let mut score = Vec::with_capacity(n);
        let mut beta = Vec::with_capacity(n);
        for t in 0..n {
            let k: Vec<f64> = keys[t * KEY_DIM..(t + 1) * KEY_DIM]
                .iter()
                .map(|&b| f64::from(half_to_f32(b)))
                .collect();
            let kn = k.iter().map(|v| v * v).sum::<f64>().sqrt();
            let k_rms = kn / (KEY_DIM as f64).sqrt();
            let (mut sc, mut var, mut mag, mut sum) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (h, &wh) in w.iter().enumerate().take(KEY_HEADS) {
                let qh = &q[h * KEY_DIM..(h + 1) * KEY_DIM];
                let (mut d, mut m, mut qn) = (0.0f64, 0.0f64, 0.0f64);
                for (&a, &b) in qh.iter().zip(&k) {
                    let a = f64::from(a);
                    d += a * b;
                    m += (a * b).abs();
                    qn += a * a;
                }
                let wh = f64::from(wh);
                let relu = d.max(0.0);
                sc += wh * relu;
                var += (dw * relu).powi(2);
                if d > 0.0 {
                    var += (wh * dq * kn).powi(2);
                    if Some(t) == written {
                        var += (wh * rho_k * k_rms * qn.sqrt()).powi(2);
                    }
                }
                mag += wh.abs() * m;
                sum += wh.abs() * relu;
            }
            score.push(sc);
            beta.push(
                Z_TIE * var.sqrt() + (gamma(KEY_DIM) + TC_UNITS * U) * mag + gamma(KEY_HEADS) * sum,
            );
        }
        (score, beta)
    }

    /// Our list against ik's under the tie rule of `gate_deepseek41_index.rs`
    /// (`ids_vs_rule`): `s_k` is ik's `k`-th score (ties to the lower row),
    /// and a row of the symmetric difference passes only when `|s_t − s_k| ≤
    /// β_t + β_k`. Returns the rows of the difference, those outside the
    /// band (a row past `n` is always outside), and the rows the band holds.
    fn list_vs_ik(
        ours: &[u32],
        ik: &[u32],
        score: &[f64],
        beta: &[f64],
        k: usize,
    ) -> (usize, usize, usize) {
        let mut order: Vec<usize> = (0..score.len()).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        let kth = order[k - 1];
        let (sk, bk) = (score[kth], beta[kth]);
        let near = |t: usize| (score[t] - sk).abs() <= beta[t] + bk;
        let (mut diff, mut off) = (0usize, 0usize);
        for (a, b) in [(ours, ik), (ik, ours)] {
            for &t in a.iter().filter(|t| !b.contains(t)) {
                diff += 1;
                let t = t as usize;
                off += usize::from(t >= score.len() || !near(t));
            }
        }
        let band = (0..score.len()).filter(|&t| near(t)).count();
        (diff, off, band)
    }

    /// What swapping ik's list for ours alone does to layer `l`'s attention
    /// output: `|y(ours) − y(ik)| / |y(ik)|`, each `y` the attention in f64
    /// at ik's roped queries, its window (`dsv4_raw_k_write-L`, the step's
    /// window rows), the stream's rows the list names (the source layer's
    /// cache after the step's write) and the file's sinks.
    fn swap_attention(
        set: &Set,
        split: &Split,
        hp: &Hparams,
        l: usize,
        s: usize,
        ours: &[u32],
        ik: &[u32],
    ) -> Result<f64, GateError> {
        let man = &set.man;
        let width = hp.head_dim;
        let st = hp.layers[l]
            .stream
            .ok_or("a selecting layer attends no stream")?;
        let q = named(man, &format!("q_rope-{l}"))?;
        let (_, raw) = node(man, &format!("dsv4_raw_k_write-{l}"), "SET_ROWS")?;
        let raw = widened_f16_rows_in(&man.dir, raw)?;
        let rows = cache_rows(
            man,
            &format!("{}_k_write-{}", STREAMS[s], st.kv_source),
            "-",
        )?
        .ok_or_else(|| format!("layer {l}: no cache of its stream's rows after the step"))?;
        let sinks = split_f32(split, &names::attn_sinks(l), hp.n_head)?;
        let pos = set.pos as usize;
        let len = (set.plan.pos[0] - set.plan.raw_first[0] + 1) as usize;
        let widen = |bits: &[u16]| -> Vec<f64> {
            bits.iter().map(|&b| f64::from(half_to_f32(b))).collect()
        };
        let window: Vec<Vec<f64>> = (pos + 1 - len..=pos)
            .map(|c| widen(&raw[c * width..(c + 1) * width]))
            .collect();
        let scale = 1.0 / (width as f64).sqrt();
        let attend = |list: &[u32]| -> Vec<f64> {
            let keys: Vec<Vec<f64>> = window
                .iter()
                .cloned()
                .chain(list.iter().map(|&r| {
                    let r = r as usize;
                    widen(&rows[r * width..(r + 1) * width])
                }))
                .collect();
            let mut y = vec![0.0f64; hp.n_head * width];
            for h in 0..hp.n_head {
                let qh: Vec<f64> = q[h * width..(h + 1) * width]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                let logits: Vec<f64> = keys
                    .iter()
                    .map(|k| scale * qh.iter().zip(k).map(|(a, b)| a * b).sum::<f64>())
                    .collect();
                let sink = f64::from(sinks[h]);
                let mx = logits.iter().fold(sink, |a, &b| a.max(b));
                let p: Vec<f64> = logits.iter().map(|&x| (x - mx).exp()).collect();
                let denom = p.iter().sum::<f64>() + (sink - mx).exp();
                let yh = &mut y[h * width..(h + 1) * width];
                for (pi, k) in p.iter().zip(&keys) {
                    for (o, v) in yh.iter_mut().zip(k) {
                        *o += pi * v;
                    }
                }
                for o in yh.iter_mut() {
                    *o /= denom;
                }
            }
            y
        };
        let (a, b) = (attend(ours), attend(ik));
        let d: f64 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        let n: f64 = b.iter().map(|y| y * y).sum();
        Ok((d / n).sqrt())
    }

    /// The selection at layer `l`'s attention seam in a set where its stream
    /// selects (`n_vis > top_k`; `None` elsewhere). An indexer layer pins its
    /// list as the exact top-k of its own scores, its query and weights
    /// inside their envelope (`Z·sqrt(r_fold² + v)`, `v` ik's quantizers over
    /// the inputs: q8_2 of `attn_norm` for q_a and of `qr_norm` for the
    /// query's projection, q8_K against our q8_1 of `attn_norm` for the
    /// weights'), and the list against ik's `lid_top_k-L` under
    /// [`list_vs_ik`] with [`score_band`]; a layer that reads another's list
    /// pins that it read that list. Returns the passes of the pins on our
    /// own step (the exact top-k, the list read) and of those against ik's
    /// nodes and, where our list and ik's differ, the swap's relative
    /// variance at the attention output.
    fn check_selection(
        set: &Set,
        split: &Split,
        hp: &Hparams,
        o: &SeamOut,
        r_fold: f64,
        state: &mut SelState,
    ) -> Result<Option<(bool, bool, f64)>, GateError> {
        let l = o.layer;
        let Some(st) = hp.layers[l].stream else {
            return Ok(None);
        };
        let s = stream_index(&set.plan, st.ratio)?;
        let n_vis = set.plan.streams[s].n_visible[0] as usize;
        let k = set.top_k;
        if n_vis <= k {
            return Ok(None);
        }
        let man = &set.man;
        let list = o
            .list
            .as_ref()
            .filter(|v| v.len() >= k)
            .ok_or_else(|| format!("layer {l}: its seam holds no list of {k}"))?;
        let ours = &list[..k];
        let ts = st.topk_source;
        let ik: Vec<u32> = ref_ints(
            man,
            &format!("lid_top_k-{ts}"),
            0,
            RowKind::Tensor,
            Layout::Flat,
        )?
        .into_iter()
        .map(u32::try_from)
        .collect::<Result<_, _>>()?;
        let (own, vs_ik, what) = if hp.layers[l].indexer {
            let ix = o
                .index
                .as_ref()
                .ok_or_else(|| format!("layer {l}: its seam holds no indexer taps"))?;
            let exact = ix.scores.len() >= n_vis && ours == exact_top(&ix.scores[..n_vis], k);
            let x = named(man, &format!("attn_norm-{l}"))?;
            let qr = named(man, &format!("qr_norm-{l}"))?;
            let ik_q = named(man, &format!("lid_q_hadamard-{l}"))?;
            let w_name = format!("lid_weights-{l}");
            let ik_w = f32s(
                man,
                man.tensors
                    .iter()
                    .find(|r| r.op == "SCALE" && r.src0.as_deref() == Some(w_name.as_str()))
                    .ok_or_else(|| format!("no SCALE of {w_name}"))?,
            )?;
            let rho_q = Z * (r_fold * r_fold + v_q8_2(&x) + v_q8_2(&qr)).sqrt();
            let rho_w = Z * (r_fold * r_fold + v_q8(&x, 256) + v_q8(&x, 128)).sqrt();
            let (q_rel, w_rel) = (rel(&ix.q, &ik_q), rel(&ix.w, &ik_w));
            let inputs = q_rel <= rho_q && w_rel <= rho_w;
            if st.index_key_source == l {
                *SelState::slot(&mut state.key_rho, s) = Some(std::f64::consts::SQRT_2 * rho_w);
            }
            let rho_k = state
                .key_rho
                .get(s)
                .copied()
                .flatten()
                .unwrap_or(std::f64::consts::SQRT_2 * rho_w);
            let (_, kv) = node(man, &format!("lid_k-{l}"), "VIEW")?;
            let keys = widened_f16_rows_in(&man.dir, kv)?;
            if keys.len() < n_vis * KEY_DIM {
                return Err(format!("lid_k-{l} holds fewer than {n_vis} keys").into());
            }
            let written = set.plan.streams[s].state_write.first().map(|&w| w as usize);
            let (score, beta) =
                score_band(&ik_q, &ik_w, &keys, n_vis, rho_q, rho_w, rho_k, written);
            let (diff, off, band) = list_vs_ik(ours, &ik, &score, &beta, k);
            *SelState::slot(&mut state.lists, s) = Some(ours.to_vec());
            (
                exact,
                inputs && off == 0,
                format!(
                    "exact_top={exact} q_rel={q_rel:.3e} (bound {rho_q:.3e}) w_rel={w_rel:.3e} \
                     (bound {rho_w:.3e}) symdiff={diff} outside_tie_band={off} tie_band_rows={band}"
                ),
            )
        } else {
            let same = state.lists.get(s).and_then(Option::as_deref) == Some(ours);
            (same, true, format!("reads layer {ts}'s list: {same}"))
        };
        let differs = {
            let (mut a, mut b) = (ours.to_vec(), ik.clone());
            a.sort_unstable();
            b.sort_unstable();
            a != b
        };
        let v_swap = if differs {
            swap_attention(set, split, hp, l, s, ours, &ik)?.powi(2)
        } else {
            0.0
        };
        println!(
            "select set={} L={l} n_vis={n_vis} top_k={k} {what} swap_attn_rel={:.3e}: own {}, \
             against ik {}",
            set.name,
            v_swap.sqrt(),
            verdict(own),
            verdict(vs_ik)
        );
        Ok(Some((own, vs_ik, v_swap)))
    }

    // -------------------------------------------------------------- routing

    /// One f32 ulp at `v`.
    fn ulp32(v: f64) -> f64 {
        let a = v.abs() as f32;
        if a < f32::MIN_POSITIVE {
            return f64::from(f32::from_bits(1));
        }
        f64::from(f32::from_bits(a.to_bits() & 0x7f80_0000)) * f64::from(f32::EPSILON)
    }

    /// The exact `sqrt(ln(1 + e^x))`.
    fn softplus64(x: f32) -> f64 {
        let x = f64::from(x);
        if x > 20.0 {
            x.sqrt()
        } else {
            (1.0 + x.exp()).ln().sqrt()
        }
    }

    /// How far one side's f32 score can sit from the exact score at `x`
    /// (the MoE chain gate's `softplus_err`).
    fn softplus_err(x: f32, e_ulps: f64, l_ulps: f64) -> f64 {
        let p = softplus64(x);
        if x > 20.0 {
            return 0.5 * ulp32(p);
        }
        let e = f64::from(x).exp();
        let y = 1.0 + e;
        let sp = y.ln();
        let dy = e_ulps * ulp32(e) + 0.5 * ulp32(y);
        let dsp = dy / (y - dy) + l_ulps * ulp32(sp);
        dsp / p + 0.5 * ulp32(p)
    }

    /// The MoE chain gate's `tie_margin`: the smallest gap over band sum
    /// between an expert of ik's first six and one after it (above 1:
    /// decided).
    fn tie_margin(
        ik_logits: &[f32],
        ik_sel: &[f32],
        logits: &[f32],
        probs: &[f32],
        bias: &[f32],
    ) -> f64 {
        let band: Vec<f64> = (0..N_EXPERT)
            .map(|e| {
                let (lo, lk) = (logits[e], ik_logits[e]);
                let sel_o = f64::from(probs[e] + bias[e]);
                (softplus64(lo) - softplus64(lk)).abs()
                    + softplus_err(lo, DEVICE_EXP_ULPS, DEVICE_LOG_ULPS)
                    + softplus_err(lk, HOST_EXP_ULPS, HOST_LOG_ULPS)
                    + 0.5 * ulp32(sel_o)
                    + 0.5 * ulp32(f64::from(ik_sel[e]))
            })
            .collect();
        let mut order: Vec<usize> = (0..N_EXPERT).collect();
        order.sort_by(|&a, &b| {
            (ik_sel[b] + 0.0)
                .total_cmp(&(ik_sel[a] + 0.0))
                .then(b.cmp(&a))
        });
        let mut worst = f64::INFINITY;
        for p in 0..N_USED {
            let a = order[p];
            for &b in &order[p + 1..] {
                let gap = f64::from(ik_sel[a]) - f64::from(ik_sel[b]);
                worst = worst.min(gap / (band[a] + band[b]));
            }
        }
        worst
    }

    /// Layer `l`'s routed ids against ik's `ffn_moe_topk-L`, in order. A
    /// difference passes only as a near tie (`tie_margin` at most 1) while
    /// the layer's attention output is inside its envelope. Returns the pass
    /// and whether the two sets of experts differ — not only their order,
    /// which moves nothing but a sum's order. Where they differ, what the
    /// swap alone does to the streams prints: the two sides' weighted host
    /// sums of the experts only one side picked, at ik's normed input, carried
    /// through `post` (the ffn HC_PRE's) against ik's `l_out-L`.
    fn check_router(
        set: &Set,
        split: &Split,
        hp: &Hparams,
        l: usize,
        rt: &RouterHost,
        input_ok: bool,
    ) -> Result<(bool, bool), GateError> {
        let man = &set.man;
        let topk = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ik: Vec<u32> = topk_ids_logical_within(man, topk, u32::try_from(N_EXPERT)?)?
            .iter()
            .map(|&i| i.cast_unsigned())
            .collect();
        if rt.ids == ik {
            return Ok((true, false));
        }
        let (mut a, mut b) = (rt.ids.clone(), ik.clone());
        a.sort_unstable();
        b.sort_unstable();
        let set_differs = a != b;
        let ik_logits = named(man, &format!("ffn_moe_logits-{l}"))?;
        let ik_sel = named(man, &format!("ffn_moe_probs_biased-{l}"))?;
        let bias = split_f32(split, &names::exp_probs_b(l), N_EXPERT)?;
        let margin = tie_margin(&ik_logits, &ik_sel, &rt.logits, &rt.probs, &bias);
        let waived = margin <= 1.0 && input_ok;
        let swap = if set_differs {
            format!("{:.3e}", swap_effect(set, split, hp, l, rt, &ik)?)
        } else {
            "-".to_string()
        };
        println!(
            "router set={} L={l} ids={:?} ik_ids={ik:?} set_differs={set_differs} \
             tie_margin={margin:.3e} input_in_envelope={input_ok} swap_streams_rel={swap}: {}",
            set.name,
            rt.ids,
            if waived { "near tie: waived" } else { "FAIL" }
        );
        Ok((waived, set_differs))
    }

    /// `|post| · |Σ_{ours∖ik} w°·E(x) − Σ_{ik∖ours} wᵏ·E(x)| / |l_out-L|` at
    /// ik's normed input `x`, the experts computed by the host tier's own
    /// code over the file.
    fn swap_effect(
        set: &Set,
        split: &Split,
        hp: &Hparams,
        l: usize,
        rt: &RouterHost,
        ik: &[u32],
    ) -> Result<f64, GateError> {
        let man = &set.man;
        let ik_w = named(man, &format!("ffn_moe_weights_scaled-{l}"))?;
        let x = named(man, &format!("ffn_norm-{l}"))?;
        let (hc, _) = hc_pre(man, &names::hc_ffn_scale(l))?;
        let s = named(man, &format!("l_out-{l}"))?;
        let first = split.shard_path(0).ok_or("the file has no shard 0")?;
        let mut host = Ds41Host::build(Split::open(first)?, hp, l..l + 1)?;
        let xt = Tensor2 {
            ne0: x.len(),
            ne1: 1,
            data: x,
        };
        let only = |a: &[u32], w: &[f32], b: &[u32]| -> Vec<(u32, f32)> {
            a.iter()
                .zip(w)
                .filter(|(e, _)| !b.contains(e))
                .map(|(&e, &wv)| (e, wv))
                .collect()
        };
        let mut ours = vec![0.0f32; hp.n_embd];
        let mut theirs = vec![0.0f32; hp.n_embd];
        host.experts_into(l, &xt, &only(&rt.ids, &rt.weights, ik), &mut ours)?;
        host.experts_into(l, &xt, &only(ik, &ik_w, &rt.ids), &mut theirs)?;
        let d: Vec<f32> = ours.iter().zip(&theirs).map(|(a, b)| a - b).collect();
        let p: f64 = hc[4..8].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        Ok((p * sumsq(&d) / sumsq(&s)).sqrt())
    }

    /// The top two of `v` by (value desc, id asc): ids and values.
    fn top2(v: &[f32]) -> (usize, f32, usize, f32) {
        let ahead = |a: usize, b: usize| v[a] > v[b] || (v[a] == v[b] && a < b);
        let (mut best, mut second) = (0usize, usize::MAX);
        for i in 1..v.len() {
            if ahead(i, best) {
                second = best;
                best = i;
            } else if second == usize::MAX || ahead(i, second) {
                second = i;
            }
        }
        (best, v[best], second, v[second])
    }

    /// The head end: the argmax equals `result_output`'s, and the logits sit
    /// inside the envelope carried one step on — the last streams' collapse
    /// keeps their relative error (`acc`, the envelope's variance there), and
    /// the head's Q6_K reads our norm's q8_1 per 128 against ik's q8_2.
    fn check_head(set: &Set, logits: &[f32], carried: Option<f64>) -> Result<bool, GateError> {
        let want = named(&set.man, "result_output")?;
        let norm = named(&set.man, "result_norm")?;
        let r_in = carried.unwrap_or(f64::NAN);
        let bound = (r_in * r_in + v_q8(&norm, 128) + v_q8_2(&norm)).sqrt();
        let r = rel(logits, &want);
        let (a, av, b, bv) = top2(logits);
        let (ka, kav, kb, kbv) = top2(&want);
        let ok = a == ka && r.is_finite() && r <= Z * bound;
        println!(
            "head set={}: top1 {a} (margin {:.4} over {b}) ik top1 {ka} (margin {:.4} over {kb}), \
             logits_rel={r:.3e} carried={r_in:.3e} bound={bound:.3e} ratio={:.3}: {}",
            set.name,
            av - bv,
            kav - kbv,
            r / bound,
            if carried.is_some() {
                verdict(ok)
            } else {
                "diagnostic (the step left ik's path)"
            }
        );
        Ok(ok || carried.is_none())
    }

    // --------------------------------------------------------------- greedy

    /// ik's greedy row for prompt 0 (`tools/ref/ik-greedy.sh`): the prompt's
    /// ids and the generated ids with their margins.
    struct IkGreedy {
        prompt: Vec<u32>,
        ids: Vec<u32>,
        margins: Vec<f32>,
    }

    fn read_ik_greedy(p: u32) -> Result<IkGreedy, GateError> {
        let dir = data_dir().join("greedy-ds41");
        let list = |s: &str| -> Result<Vec<String>, GateError> {
            Ok(s.split(',').map(|v| v.trim().to_string()).collect())
        };
        let prompt_path = dir.join(format!("prompt{p}.tsv"));
        let prompt_text = std::fs::read_to_string(&prompt_path).map_err(|e| {
            format!(
                "{}: {e} — run just ik-greedy-ds41 {p}",
                prompt_path.display()
            )
        })?;
        let row = prompt_text
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or("no prompt row")?;
        let f: Vec<&str> = row.split('\t').collect();
        let prompt = list(f.get(2).ok_or("prompt row has no ids")?)?
            .iter()
            .map(|v| v.parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;
        let out_path = dir.join(format!("greedy-ik-cpu-{GREEDY}-p{p}.tsv"));
        let out = std::fs::read_to_string(&out_path)
            .map_err(|e| format!("{}: {e} — run just ik-greedy-ds41 {p}", out_path.display()))?;
        let row = out
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or("no greedy row")?;
        let f: Vec<&str> = row.split('\t').collect();
        let ids = list(f.get(5).ok_or("greedy row has no gen_ids")?)?
            .iter()
            .map(|v| v.parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?;
        let margins = list(f.get(6).ok_or("greedy row has no gen_margins")?)?
            .iter()
            .map(|v| v.parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(IkGreedy {
            prompt,
            ids,
            margins,
        })
    }

    /// The file's token strings, for printing the two continuations.
    fn vocab(split: &Split) -> Vec<String> {
        match split.value("tokenizer.ggml.tokens") {
            Some(gguf::Value::Array(v)) => v
                .iter()
                .map(|t| match t {
                    gguf::Value::String(s) => s.clone(),
                    _ => "?".to_string(),
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    fn text(vocab: &[String], ids: &[u32]) -> String {
        ids.iter()
            .map(|&i| vocab.get(i as usize).map_or("?", String::as_str))
            .collect::<String>()
            .replace('Ġ', " ")
            .replace('Ċ', "\\n")
    }

    fn vmstat(key: &str) -> u64 {
        std::fs::read_to_string("/proc/vmstat")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix(key)?.trim().parse().ok())
            })
            .unwrap_or(0)
    }

    fn greedy(m: &mut Deepseek41Model, split: &Split, p: u32) -> Result<bool, GateError> {
        let ik = read_ik_greedy(p)?;
        let eos = bloomery_gpu_gates::eos_token_id(split)?;
        let vocab = vocab(split);
        m.reset()?;
        let t = Instant::now();
        let mut ids = Vec::with_capacity(GREEDY);
        let mut margins = Vec::with_capacity(GREEDY);
        let mut tok = m.step(&ik.prompt)?;
        let (mut allocs, mut minor, mut major, mut steady) = (0u64, 0u64, 0u64, 0u64);
        let pgmaj0 = vmstat("pgmajfault ");
        for i in 0..GREEDY {
            let logits = m.logits()?;
            let (a, av, _, bv) = top2(&logits);
            if a != tok as usize {
                return Err(
                    format!("step {i}: the engine picked {tok}, its logits' top is {a}").into(),
                );
            }
            ids.push(tok);
            margins.push(av - bv);
            if i + 1 == GREEDY {
                break;
            }
            let (a0, f0) = (ALLOCS.load(Ordering::Relaxed), engram::faults_thread());
            tok = m.step(&[tok])?;
            let (a1, f1) = (ALLOCS.load(Ordering::Relaxed), engram::faults_thread());
            if i >= 8 {
                allocs += a1 - a0;
                minor += f1.minor - f0.minor;
                major += f1.major - f0.major;
                steady += 1;
            }
        }
        let wall = t.elapsed().as_secs_f64();
        let pgmaj = vmstat("pgmajfault ") - pgmaj0;
        // ik stops at its EOS and records it: the comparison runs over ik's ids.
        let ik_stopped = ik.ids.len() < GREEDY && ik.ids.last() == Some(&eos);
        let first = ids.iter().zip(&ik.ids).position(|(a, b)| a != b);
        let ok = match first {
            None => ik.ids.len() == GREEDY || ik_stopped,
            Some(p) => margins[p] < GREEDY_MARGIN,
        };
        println!(
            "greedy prompt ids {:?} ({:?})",
            ik.prompt,
            text(&vocab, &ik.prompt)
        );
        println!("greedy ours {:?}", ids);
        println!("greedy ik   {:?}", ik.ids);
        println!("greedy ours text {:?}", text(&vocab, &ids));
        println!("greedy ik   text {:?}", text(&vocab, &ik.ids));
        match first {
            None => println!(
                "greedy prompt {p}: ik's {} ids equal ours{}; margins ours min {:.3}, ik min {:.3}; \
                 {GREEDY} steps in {wall:.1} s (runtime value): {}",
                ik.ids.len(),
                if ik_stopped {
                    " (ik stopped at EOS)"
                } else {
                    ""
                },
                margins[..ik.ids.len().min(GREEDY)]
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min),
                ik.margins.iter().copied().fold(f32::INFINITY, f32::min),
                verdict(ok)
            ),
            Some(at) => println!(
                "greedy prompt {p}: first difference at {at}: ours {} (margin {:.4}) ik {} (margin \
                 {:.4}); pass rule our margin < {GREEDY_MARGIN}: {}",
                ids[at],
                margins[at],
                ik.ids[at],
                ik.margins.get(at).copied().unwrap_or(f32::NAN),
                verdict(ok)
            ),
        }
        println!(
            "steady steps {steady}: allocs_per_step={:.2} minor_faults_per_step={:.1} \
             major_faults_per_step={:.2} (calling thread); pgmajfault over the run {pgmaj} \
             (machine-wide)",
            allocs as f64 / steady.max(1) as f64,
            minor as f64 / steady.max(1) as f64,
            major as f64 / steady.max(1) as f64
        );
        Ok(ok)
    }

    // ------------------------------------------------------------------ ppl

    fn ppl(m: &mut Deepseek41Model, hp: &Hparams, tag: &str) -> Result<bool, GateError> {
        let path: PathBuf = data_dir().join("ikppl").join(format!("{tag}.kld"));
        let base = KldBase::open(&path, hp.n_vocab)?;
        println!(
            "ppl base {}: ctx {} chunks {} scored per chunk {} from {}",
            path.display(),
            base.n_ctx(),
            base.n_chunk(),
            base.scored_per_chunk(),
            base.first_scored()
        );
        let t = Instant::now();
        let (mut n, mut sd, mut sd2) = (0usize, 0.0f64, 0.0f64);
        let (mut nll_o, mut nll_i, mut kld, mut kld2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut same_top, mut dm2) = (0usize, 0.0f64);
        let mut lq = vec![0.0f64; hp.n_vocab];
        for c in 0..base.n_chunk() {
            let toks = base
                .chunk_tokens(c)
                .ok_or_else(|| format!("chunk {c} has no ids"))?
                .to_vec();
            m.reset()?;
            for pos in 0..base.n_ctx() - 1 {
                m.step(&toks[pos..=pos])?;
                let Some(rec) = base.record(c, pos) else {
                    continue;
                };
                let logits = m.logits()?;
                let mx = logits.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
                let lse = f64::from(mx)
                    + logits
                        .iter()
                        .map(|&v| (f64::from(v) - f64::from(mx)).exp())
                        .sum::<f64>()
                        .ln();
                for (q, &v) in lq.iter_mut().zip(&logits) {
                    *q = f64::from(v) - lse;
                }
                let no = -lq[rec.next as usize];
                let ni = rec.nll();
                let d = no - ni;
                n += 1;
                sd += d;
                sd2 += d * d;
                nll_o += no;
                nll_i += ni;
                let mut k = 0.0f64;
                let (mut top_i, mut best_i, mut second_i) =
                    (0usize, f64::NEG_INFINITY, f64::NEG_INFINITY);
                for (i, lp) in rec.log_probs().enumerate() {
                    let lp = f64::from(lp);
                    k += lp.exp() * (lp - lq[i]);
                    if lp > best_i {
                        second_i = best_i;
                        (top_i, best_i) = (i, lp);
                    } else if lp > second_i {
                        second_i = lp;
                    }
                }
                kld += k;
                kld2 += k * k;
                let (a, av, _, bv) = top2(&logits);
                same_top += usize::from(a == top_i);
                let dm = f64::from(av - bv) - (best_i - second_i);
                dm2 += dm * dm;
            }
            let nf = n as f64;
            println!(
                "ppl chunk {c}: {n} positions, mean d {:+.5}, ppl ours {:.4} ik {:.4}, {:.0} s",
                sd / nf,
                (nll_o / nf).exp(),
                (nll_i / nf).exp(),
                t.elapsed().as_secs_f64()
            );
        }
        let nf = n as f64;
        let mean = sd / nf;
        let var = (sd2 / nf - mean * mean).max(0.0) * nf / (nf - 1.0);
        let se = (var / nf).sqrt();
        let dppl = mean.exp() - 1.0;
        let kmean = kld / nf;
        let kse = ((kld2 / nf - kmean * kmean).max(0.0) / (nf - 1.0)).sqrt();
        let ok = dppl <= PPL_RED;
        println!(
            "ppl: {n} positions; PPL ours {:.4} ik {:.4}; d = NLL_ours − NLL_ik mean {mean:+.5} ± \
             {se:.5} (SE), sd(d) {:.4}; Δ_PPL {:+.3} % (red above {:+.1} %); KLD(ik‖ours) {kmean:.5} \
             ± {kse:.5}; same top {:.2} %; σ_rel (rms of margin ours − ik) {:.4}; {:.0} s: {}",
            (nll_o / nf).exp(),
            (nll_i / nf).exp(),
            var.sqrt(),
            100.0 * dppl,
            100.0 * PPL_RED,
            100.0 * same_top as f64 / nf,
            (dm2 / nf).sqrt(),
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        Ok(ok)
    }
}
