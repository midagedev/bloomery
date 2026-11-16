//! The qwen35 family's next-token (MTP) draft gate: a load opened with the
//! file's own next-token layer (`GpuModel::<Body35>::open_mtp`) runs one draft
//! step through Body35's own layer path (`Mtp35`).
//!
//! What is asserted, on two real files that carry the layer, each named by
//! its lever and pinned by its architecture and its size in bytes (the first
//! also by its `general.name`), each through one load and every clause below:
//! the unsloth Qwen3.6 `UD-Q4_K_M` (`BLOOMERY_MTP_MODEL`; `blk.40`: full
//! attention and a routed MoE with the trunk's shapes) and lmstudio's
//! Qwen3.8-27B `Q4_K_M` (`BLOOMERY_MTP_DENSE_MODEL`; `blk.64`: full attention
//! and the trunk's dense SwiGLU), each beside
//! `nextn.{eh_proj, enorm, hnorm, shared_head_norm}`:
//! - (q) refusals by name at open, before any upload: for the Qwen3.6 file the
//!   plain twin (`BLOOMERY_REF_MODEL`: no `nextn` tensor, no next-token layer)
//!   fails with the in-file rule's reason, not a missing-tensor error from the
//!   loader; for both, a media load. No file of the 27B lacks the layer: the
//!   model crate's tests hold its absent, stripped, unshaped and FFN-less
//!   tails.
//! - (d) the draft's resident card bytes are the figure derived from the
//!   file's tensor table: the layer's own tensors (a bf16 one, as the Qwen3.6
//!   router and shared expert's gate, widened to f32 on the card, every other
//!   type as the file stores it), one f32 stream and one dense K/V store at
//!   the open's context; the head's output matrix and the embedding are the target's own device
//!   buffers at the addresses the draft recorded, and the draft's weights
//!   hold neither.
//! - (j) the input join (`mtp_input` at one stream) is the two RMS norms of
//!   ik's MTP graph, packed `[e_n | h_n]`, within the rounding band of its
//!   operation depth against a host f64 computation; at four streams the
//!   embedding half is the one-stream pack's bits in every stream's slot; a
//!   stream count other than 1 or 4 is refused by name.
//! - (h) the draft's token is the host argmax (the first index on a tie) of
//!   the logits the card wrote, on every step of (s).
//! - (r) refusals by name on the live draft: a position past its K/V rows, a
//!   token past the vocabulary and a hidden row too short leave the load as
//!   it was, and the step they interrupted repeats to the same token. A
//!   second resident sequence (`add_slots(2)`) is refused by name and the
//!   load keeps its one slot.
//! - (s) the draft's acceptance against the target: a 64-id prompt of the file's
//!   prose corpus (`prose_ids`; both files share the vocabulary) through the target,
//!   one step at a time for its own next tokens and in one pass for its
//!   final-norm rows; the draft is then fed each position in order — the
//!   prompt's next id, the position and the target's row of it — so its K/V
//!   rows cover the prompt, and its token for each of the last 16 positions
//!   is compared with the token the target itself takes after that next id.
//!   Under 30 % is a wiring error (the join order, a wrong `h`, a wrong
//!   gain), and the clause is red. The draft's own K/V rows weigh little
//!   at this prompt length: a draft attending to one row stays above it. The
//!   wiring floor does not hold the dense layer's feed-forward block (a draft
//!   with a wrong gain there clears it), so the 27B's agreement over every
//!   position is held by a pin ([`DENSE_AGREE`]).
//! - (w) the store walk and the step row, after (s): the draft's K/V rows
//!   below (s)'s last position are overwritten by steps of other ids and
//!   rows, which moves that position's logits; one store walk
//!   (`WalkMode::Store`, all those rows at once through the wide arm's
//!   GEMMs) of (s)'s feed over those positions leaves the draft's last
//!   logits bit for bit (no head runs) and the stored keys and values within
//!   `key_band` and `value_band` of the eager steps' (s)-stored rows, which
//!   the GEMMs' q8 activations move where the eager gemv reads f32 rows;
//!   store walks of three rows over the same positions, after an overwrite,
//!   store the same bits as the one walk (a row's stored bits are its inputs'
//!   alone). After two plain target steps, a chain at position 1 that reads
//!   the target head's row on the card (`MtpArena::Step`) gives the token and
//!   logits, bit for bit, of the same step over that row read back to the
//!   host.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_mtp", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(dead_code, reason = "this gate uses only the shared argmax and rel")]
mod e2e;

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::{
        Mtp35, MtpArena, Open35, PrefillPath, Qwen35moeModel, addresses, mtp_chain, mtp_walk,
    };
    use bloomery_gpu::draft::{DraftHead, Feed, Hidden, WalkBounds, WalkMode};
    use bloomery_gpu::kv_format::KvQ8;
    use bloomery_gpu::mtp::{MtpInputArgs, MtpKernels};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::rounding::{U, gamma, q8_32_rel};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, open_split, prose_ids, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::half_to_f32;
    use model::arch::Arch;
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::mtp::{
        EH_PROJ, ENORM, HNORM, InFile, SHARED_HEAD_NORM, STEP_BYTES, in_file, weight_bytes,
    };
    use model::arch::qwen35moe::spec;
    use runtime::stores::dense_kv_bytes;

    const WHAT: &str = "gate_qwen35moe_mtp";

    /// The levers naming the two files.
    const MOE_LEVER: &str = "BLOOMERY_MTP_MODEL";
    const DENSE_LEVER: &str = "BLOOMERY_MTP_DENSE_MODEL";

    /// A file the gate runs on, named by `lever` and pinned by its
    /// architecture, its size in bytes and its `general.name` when it has a
    /// pinned one.
    struct Pin {
        lever: &'static str,
        arch: &'static str,
        name: Option<&'static str>,
        bytes: u64,
        /// The directory of `$BLOOMERY_DATA` whose `corpus-prose.ids` is the prompt.
        prose: &'static str,
        /// The width of one hidden row: the file's embedding length.
        hidden: usize,
        /// The architecture of the file's own twin without the next-token
        /// layer, when one exists on the box (clause (q)'s first line).
        plain: Option<Arch>,
        /// A bf16 tensor sits in the layer, so (d)'s figure carries a widening.
        widened: bool,
        /// A floor on the positions of (s) at which the draft's token is the
        /// target's own next token, where a pin holds the feed-forward block.
        agree: Option<usize>,
        /// The load's tensor-core decode flash: the file's head ratio decides it (a
        /// six-heads-a-key-head file reads the scalar pairs' pass, which refuses `mma`).
        mma: bool,
    }

    /// unsloth's Qwen3.6 `UD-Q4_K_M` with its next-token layer (routed MoE).
    const MOE: Pin = Pin {
        lever: MOE_LEVER,
        arch: "qwen35moe",
        name: Some("Qwen3.6-35B-A3B"),
        bytes: 22_663_387_424,
        prose: "qwen4exp",
        hidden: 2048,
        plain: Some(Arch::Qwen35moe),
        widened: true,
        agree: None,
        mma: true,
    };

    /// lmstudio's Qwen3.8-27B `Q4_K_M` with its next-token layer (dense
    /// SwiGLU); the size is `tools/ref/dense-logits.py` `TARGETS`' pin.
    const DENSE: Pin = Pin {
        lever: DENSE_LEVER,
        arch: "qwen35",
        name: None,
        bytes: 16_810_714_336,
        prose: "qwen35",
        hidden: 5120,
        plain: None,
        widened: false,
        agree: Some(DENSE_AGREE),
        mma: false,
    };

    /// Cache rows of the load and the prompt's ubatch: the prompt and the
    /// draft's rows over it, with room.
    const CTX: usize = 256;
    const UBATCH: usize = 64;

    /// (s)'s prompt, and how many of its last positions are compared.
    const PROMPT: usize = 64;
    const LAST: usize = 16;

    /// The wiring floor of (s), a share of the positions.
    const FLOOR: f64 = 0.30;

    /// PIN(2026-11-16): the positions of (s)'s 63 at which the dense draft's token
    /// is the target's own next token. The 27B's draft agrees at 46; one whose
    /// feed-forward input is normed by another block's gain agrees at 38 and still
    /// clears [`FLOOR`] on the last 16 positions (7 of 16), so the wiring floor does
    /// not hold the feed-forward block. The pin is the midpoint, 42, a bug catcher
    /// like the forced_exact count pin: the walk is greedy and bit-exact, so the
    /// count is one number (46 on the A6000 and on the 3090), and the margin is 4
    /// positions each way. A change to the float order of the draft's kernels
    /// moves the count: it re-pins here, with a new PIN line.
    const DENSE_AGREE: usize = 42;

    /// Rows of the second store walks of (w): a width no caller walks, so the
    /// first walk's bits equal theirs only when a row's stored bits are its
    /// inputs' alone, not those of the walk it lands in.
    const RUN: usize = 3;

    /// PIN(2026-11-04): a store walk's values against an eager walk's over the
    /// same rows, a position's row at a time. The store walk's projections
    /// read 32-value q8 blocks where the eager walk's Q8_0 gemv reads the f32
    /// rows: `eh_proj`'s output moves by at most `rounding::q8_32_rel` of its
    /// RMS, the attention norm moves a unit vector by at most twice its
    /// input's relative distance, and the value's own projection adds one
    /// rounding of its own; the three add (the triangle inequality, as the
    /// qwen4exp gate's `value_band` adds its two); each side's f16 store one
    /// rounding of 2^-11. 3 · 1.2858e-2 + 2 · 2^-11 = 3.96e-2.
    fn value_band() -> f64 {
        3.0 * q8_32_rel() + 2.0 * 2f64.powi(-11)
    }

    /// PIN(2026-11-04): a store walk's keys against an eager walk's: the
    /// value's chain, then each head's RMS norm, which moves a unit vector by
    /// at most twice its input's relative distance, and the rope, a rotation;
    /// each side's f16 store one rounding. 2 · 3 · 1.2858e-2 + 2 · 2^-11 =
    /// 7.81e-2.
    fn key_band() -> f64 {
        6.0 * q8_32_rel() + 2.0 * 2f64.powi(-11)
    }

    fn open_at(o: Open35, file: Split) -> Result<Qwen35moeModel, GpuError> {
        Qwen35moeModel::open_mtp(Gpu::new()?, file, o)
    }

    fn opts(media: bool, mma: bool) -> Open35 {
        Open35 {
            ctx: CTX,
            mma,
            ubatch: UBATCH,
            kv: KvQ8::F16,
            media,
        }
    }

    /// The path the lever naming `p`'s file holds.
    fn lever_value(p: &Pin) -> Result<String, std::env::VarError> {
        if p.lever == DENSE_LEVER {
            std::env::var(DENSE_LEVER)
        } else {
            std::env::var(MOE_LEVER)
        }
    }

    /// The pinned file: architecture `p.arch`, `general.name` `p.name` when
    /// pinned, one shard of `p.bytes`.
    fn pinned(p: &Pin) -> Result<Split, GateError> {
        let lever = p.lever;
        let path = lever_value(p).map_err(|_| {
            format!(
                "{lever} unset: it names the {} file with its next-token layer",
                p.arch
            )
        })?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        if file.architecture() != Some(p.arch) {
            return Err(format!("{path} is {:?}, not {}", file.architecture(), p.arch).into());
        }
        if let Some(want) = p.name {
            let name = file.value("general.name").and_then(|v| v.as_str());
            if name != Some(want) {
                return Err(format!("{path}: general.name is {name:?}, not {want:?}").into());
            }
        }
        let shard = file.shard_path(0).ok_or("the file has no first shard")?;
        let bytes = std::fs::metadata(shard)
            .map_err(|e| format!("{}: {e}", shard.display()))?
            .len();
        if bytes != p.bytes {
            return Err(format!("{path} is {bytes} bytes, not {}", p.bytes).into());
        }
        Ok(file)
    }

    /// The width of the live draft's hidden row.
    fn hidden_of(m: &Qwen35moeModel) -> Result<usize, GateError> {
        Ok(m.body(WHAT)?
            .mtp()
            .ok_or("the load opened no next-token layer")?
            .hidden())
    }

    /// The error opening `file` with the draft gives, `None` when it opened.
    fn open_error(file: Split, o: Open35) -> Option<String> {
        open_at(o, file).err().map(|e| e.to_string())
    }

    /// (q): the refusals at open, before the card is touched.
    fn refusals_at_open(p: &Pin) -> Result<bool, GateError> {
        let mut none_ok = true;
        if let Some(arch) = p.plain {
            let plain = open_split(arch, "gate-gpu-qwen35moe-mtp")?;
            let r = open_error(plain, opts(false, p.mma));
            let text = r.as_deref().unwrap_or("opened");
            none_ok = r
                .as_deref()
                .is_some_and(|e| e.contains("the file has no in-file draft"))
                && r.as_deref()
                    .is_some_and(|e| !e.contains("is not in the file"));
            println!(
                "(q) the file with no next-token layer: {text} {}",
                verdict(none_ok)
            );
        }
        let r = open_error(pinned(p)?, opts(true, p.mma));
        let text = r.as_deref().unwrap_or("opened");
        let media_ok = r.as_deref().is_some_and(|e| e.contains("a media load"));
        println!("(q) a media load: {text} {}", verdict(media_ok));
        Ok(none_ok && media_ok)
    }

    /// (d): the draft's resident bytes against the plan's figure, and the
    /// borrowed matrices at the target's addresses.
    fn load_clause(m: &Qwen35moeModel, file: &Split, p: &Pin) -> Result<bool, GateError> {
        let Some(mtp) = m.body(WHAT)?.mtp() else {
            return Err("the load opened no next-token layer".into());
        };
        let InFile::Draft(draft) = in_file(file).map_err(|e| format!("in_file: {e}"))? else {
            return Err("the pinned file reads as no in-file draft".into());
        };
        let read = spec::read(file).map_err(|e| format!("spec::read: {e}"))?;
        let Mixer::Gqa(g) = &draft.layer.mixer else {
            return Err("the in-file draft's mixer is not full attention".into());
        };
        // [derived] The layer's card weights are the tail block's tensors
        // as the file stores them, a bf16 one widened to f32
        // (`DevWeight::F32`): the one derivation the serve seat's whole-fit
        // verdict prices the draft with (`mtp::weight_bytes`). The embedding
        // and the head are the target's and carry no layer index.
        let own = weight_bytes(&read.tensors.tensors, mtp.index());
        let widened = own
            - read
                .tensors
                .tensors
                .iter()
                .filter(|t| t.layer == Some(mtp.index()))
                .map(|t| t.file_bytes)
                .sum::<u64>();
        let kv = dense_kv_bytes(g.kv_heads as usize, g.head_dim as usize, CTX);
        let stream = (p.hidden * 4) as u64;
        let want = own + stream + kv;
        let got = mtp.resident_bytes() as u64;
        // The verdict's reserve for the step beside the layer's own bytes
        // (`mtp::STEP_BYTES`) holds the engine's scratch.
        let scratch = mtp.scratch_bytes() as u64;
        let bytes_ok = got == want && scratch <= STEP_BYTES;
        println!(
            "(d) draft resident {got} = derived {want} ({}): layer {own} (bf16 widened \
             {widened}) + stream {stream} + K/V {kv}; weights {}; step scratch {scratch} within \
             the verdict's {STEP_BYTES} {}",
            got as i64 - want as i64,
            mtp.weights().resident_bytes(),
            verdict(bytes_ok)
        );
        let mut ok = bytes_ok && (widened > 0 || !p.widened);
        for b in mtp.borrowed_planes() {
            let target = m.weights().get(&b.name).and_then(addresses);
            let same = target == Some(b.planes);
            let held = mtp.weights().get(&b.name).is_some();
            ok &= same && !held;
            println!(
                "(d) {} read at {:x?}, the target's at {target:x?}, the draft's own copy {} {}",
                b.name,
                b.planes,
                if held { "present" } else { "absent" },
                verdict(same && !held)
            );
        }
        let stems =
            [EH_PROJ, ENORM, HNORM, SHARED_HEAD_NORM].map(|s| format!("blk.{}.{s}", mtp.index()));
        let resident = stems.iter().all(|n| mtp.weights().get(n).is_some());
        ok &= resident;
        println!("(d) the four nextn stems resident {}", verdict(resident));
        let gain = &stems[3];
        let gain_ok = mtp.head_gain() == gain;
        ok &= gain_ok;
        println!(
            "(d) the head norms by {} {}",
            mtp.head_gain(),
            verdict(gain_ok)
        );
        let stream = m.gpu().stream();
        let own = mtp.weights().f32_buf(WHAT, gain)?.to_host_vec(stream)?;
        let trunk = m
            .weights()
            .f32_buf(WHAT, "output_norm.weight")?
            .to_host_vec(stream)?;
        let differ = own
            .iter()
            .zip(&trunk)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let norm = |x: &[f32]| x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
        let delta: Vec<f32> = own.iter().zip(&trunk).map(|(a, b)| a - b).collect();
        println!(
            "(d) the layer's gain against output_norm.weight: {differ}/{} values differ, \
             |gain - output_norm| / |output_norm| = {:.3e} (printed)",
            own.len(),
            norm(&delta) / norm(&trunk)
        );
        Ok(ok)
    }

    /// The deterministic values of (j): a xorshift stream mapped to
    /// `[-1, 1)` times `scale`.
    fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = bloomery_gpu_gates::GateRng(seed);
        (0..n).map(|_| (2.0 * rng.unit() - 1.0) * scale).collect()
    }

    /// `(x · r) · γ` with `r = 1/√(Σx²/n + eps)` over `x`, in f64.
    fn rms_ref(x: &[f32], gain: &[f32], eps: f32) -> Vec<f64> {
        let n = x.len() as f64;
        let ss: f64 = x.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let r = 1.0 / (ss / n + f64::from(eps)).sqrt();
        x.iter()
            .zip(gain)
            .map(|(&v, &g)| f64::from(v) * r * f64::from(g))
            .collect()
    }

    /// One launch of the join over `e` and `h` (`streams` streams of
    /// `hidden`) at `streams`: the pack, or the launch's error.
    fn join(
        (gpu, k, hidden): (&Gpu, &MtpKernels, usize),
        (e, h): (&[f32], &[f32]),
        (enorm, hnorm): (&[f32], &[f32]),
        streams: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let stream = gpu.stream();
        let eb = DeviceBuffer::from_host(stream, e)?;
        let hb = DeviceBuffer::from_host(stream, h)?;
        let en = DeviceBuffer::from_host(stream, enorm)?;
        let hn = DeviceBuffer::from_host(stream, hnorm)?;
        let mut out = DeviceBuffer::<f32>::zeroed(stream, 2 * 4 * hidden)?;
        k.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &eb,
                h: &hb,
                enorm: &en,
                hnorm: &hn,
                hidden,
                eps: 1e-6,
                m: 1,
                streams,
                fault: gpu.unlabelled_sink(),
                out: &mut out,
            },
        )?;
        Ok(out.to_host_vec(stream)?)
    }

    /// (j): the join at one stream against the host, the four-stream pack's
    /// embedding half against the one-stream pack, and the refused counts.
    fn join_clause(gpu: &Gpu, hidden: usize) -> Result<bool, GateError> {
        let k = MtpKernels::load(gpu.context())?;
        let at = (gpu, &k, hidden);
        let (e, enorm) = (values(hidden, 1, 3.0), values(hidden, 2, 1.0));
        let (h4, hnorm4) = (values(4 * hidden, 3, 5.0), values(4 * hidden, 4, 1.0));
        let one = join(at, (&e, &h4[..hidden]), (&enorm, &hnorm4[..hidden]), 1)?;
        // [derived] A value is `x · r · γ`: r goes through the thread's
        // `hidden/256` fused terms, the butterfly's five adds and the eight
        // warps' seven, 20 roundings in the sum, half of them in r, then the
        // add of eps, the root, the reciprocal, and the two products.
        let band = 0.5 * gamma(hidden / 256 + 5 + 7) + 5.0 * U;
        let (re, rh) = (
            rms_ref(&e, &enorm, 1e-6),
            rms_ref(&h4[..hidden], &hnorm4[..hidden], 1e-6),
        );
        let worst = |got: &[f32], want: &[f64]| {
            got.iter()
                .zip(want)
                .map(|(&g, &w)| (f64::from(g) - w).abs() / w.abs().max(1e-30))
                .fold(0.0f64, f64::max)
        };
        let (we, wh) = (
            worst(&one[..hidden], &re),
            worst(&one[hidden..2 * hidden], &rh),
        );
        let one_ok = one[..2 * hidden].iter().all(|v| v.is_finite()) && we <= band && wh <= band;
        println!(
            "(j) one stream: e half worst {we:.3e}, h half {wh:.3e} of band {band:.3e} {}",
            verdict(one_ok)
        );
        let four = join(at, (&e, &h4), (&enorm, &hnorm4), 4)?;
        let slots_ok = (0..4).all(|s| {
            four[s * 2 * hidden..s * 2 * hidden + hidden]
                .iter()
                .zip(&one[..hidden])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        println!(
            "(j) four streams: the embedding half is the one-stream pack's bits in all four \
             slots {}",
            verdict(slots_ok)
        );
        let mut refused = true;
        for streams in [0usize, 2, 3, 5] {
            let r = join(at, (&e, &h4), (&enorm, &hnorm4), streams)
                .err()
                .map(|e| e.to_string());
            let named = r
                .as_deref()
                .is_some_and(|e| e.contains(&format!("{streams} streams: the pack takes 1 or 4")));
            refused &= named;
            println!(
                "(j) {streams} streams: {} {}",
                r.as_deref().unwrap_or("launched"),
                verdict(named)
            );
        }
        Ok(one_ok && slots_ok && refused)
    }

    /// What (s) and (h) read from the target and the draft.
    struct Run {
        /// The draft's token after each fed position, from position 0.
        drafted: Vec<u32>,
        /// Steps whose token is the host argmax of the card's logits.
        argmax_ok: usize,
        /// Every logit of the last eager step, the store restoration's reference.
        logits: Vec<f32>,
    }

    /// The target's own next token after each prompt id (a plain step per
    /// id), then its final-norm row of each position (one pass).
    fn target_run(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let next = ids
            .iter()
            .map(|&t| m.step(&[t]))
            .collect::<Result<Vec<_>, _>>()?;
        m.reset()?;
        let rows = m.prefill_hidden(ids, PrefillPath::Pass)?;
        Ok((next, rows))
    }

    /// Feed the draft every position `i` of `ids` but the last in order: id
    /// `i + 1`, position `i`, the target's row `rows[i]`.
    fn draft_run(m: &mut Qwen35moeModel, ids: &[u32], rows: &[f32]) -> Result<Run, GateError> {
        let stream = m.gpu().stream_handle();
        let hidden = hidden_of(m)?;
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, hidden)?;
        let (mut drafted, mut argmax_ok, mut logits) = (Vec::new(), 0, Vec::new());
        for i in 0..ids.len() - 1 {
            h.copy_from_host(&stream, &rows[i * hidden..(i + 1) * hidden])?;
            let token = m.mtp_draft(ids[i + 1], i, &h)?;
            logits = draft_logits(m)?;
            let at = super::e2e::argmax(&logits) as usize;
            argmax_ok += usize::from(at == token as usize);
            drafted.push(token);
        }
        Ok(Run {
            drafted,
            argmax_ok,
            logits,
        })
    }

    fn draft_logits(m: &Qwen35moeModel) -> Result<Vec<f32>, GateError> {
        let mtp = m
            .body(WHAT)?
            .mtp()
            .ok_or("the load opened no next-token layer")?;
        Ok(mtp.logits_to_host(m.gpu())?)
    }

    /// (h) and (s) on the run, and (r) after it.
    fn step_clauses(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        agree: Option<usize>,
    ) -> Result<bool, GateError> {
        let (next, rows) = target_run(m, ids)?;
        let run = draft_run(m, ids, &rows)?;
        let steps = run.drafted.len();
        let h_ok = run.argmax_ok == steps;
        println!(
            "(h) the draft's token is the host argmax of its logits on {}/{steps} steps {}",
            run.argmax_ok,
            verdict(h_ok)
        );
        // The draft at position `i` follows id `i + 1`; the target's own next
        // token after that id is `next[i + 1]`.
        let from = steps - LAST;
        let hits = (from..steps)
            .filter(|&i| run.drafted[i] == next[i + 1])
            .count();
        let text = (from..steps)
            .filter(|&i| ids.get(i + 2) == Some(&run.drafted[i]))
            .count();
        let all = (0..steps)
            .filter(|&i| run.drafted[i] == next[i + 1])
            .count();
        let rate = hits as f64 / LAST as f64;
        let wired = rate >= FLOOR;
        println!(
            "(s) the draft's token is the target's own next token at {hits}/{LAST} of the last \
             positions = {:.1} % ({all}/{steps} over every position, {text}/{LAST} the \
             prompt's next id; floor {:.0} %) {}",
            100.0 * rate,
            100.0 * FLOOR,
            verdict(wired)
        );
        let agreed = agree.is_none_or(|n| all >= n);
        if let Some(n) = agree {
            println!(
                "(s) the draft agrees with the target at {all}/{steps} positions; pin {n} {}",
                verdict(agreed)
            );
        }
        let at = super::e2e::argmax(&run.logits) as usize;
        println!(
            "(s) the last draft step's largest logit {:.6} at token {}",
            run.logits[at], at
        );
        let r_ok = refusals(m, ids, &rows, run.drafted[steps - 1])?;
        let w_ok = store_clause(m, ids, &rows, &run)?;
        Ok(h_ok && r_ok && wired && agreed && w_ok)
    }

    /// The draft's step `t` at position `pos` over the host row `row`, and its
    /// logits.
    fn host_step(
        m: &mut Qwen35moeModel,
        (t, pos): (u32, usize),
        row: &[f32],
    ) -> Result<(u32, Vec<f32>), GateError> {
        let stream = m.gpu().stream_handle();
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, hidden_of(m)?)?;
        h.copy_from_host(&stream, row)?;
        let token = m.mtp_draft(t, pos, &h)?;
        Ok((token, draft_logits(m)?))
    }

    /// The draft's stored keys and values at positions `0..n`, position-major.
    fn store_of(m: &Qwen35moeModel, n: usize) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let d = m
            .body(WHAT)?
            .mtp()
            .ok_or("the load opened no next-token layer")?;
        Ok(d.store_host(m.gpu().stream(), n)?)
    }

    /// The largest relative distance over the positions below `n` of the keys,
    /// then of the values, in `a` from those in `b` (two [`store_of`] reads of
    /// `n` positions), a position's row at a time.
    fn store_rel(a: &(Vec<u16>, Vec<u16>), b: &(Vec<u16>, Vec<u16>), n: usize) -> (f64, f64) {
        let row = a.0.len() / n.max(1);
        let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| half_to_f32(x)).collect() };
        let worst = |x: &[u16], y: &[u16]| {
            (0..n)
                .map(|p| {
                    let r = p * row..(p + 1) * row;
                    super::e2e::rel(&f(&x[r.clone()]), &f(&y[r]))
                })
                .fold(0.0f64, f64::max)
        };
        (worst(&a.0, &b.0), worst(&a.1, &b.1))
    }

    /// Steps of other ids over positions `0..last`, so no stored row is (s)'s.
    fn overwrite(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        rows: &[f32],
        last: usize,
    ) -> Result<(), GateError> {
        let hidden = hidden_of(m)?;
        let row = |i: usize| &rows[i * hidden..(i + 1) * hidden];
        for i in 0..last {
            host_step(m, (ids[i + 2], i), row(i + 1))?;
        }
        Ok(())
    }

    /// Store walks of (s)'s feed over positions `0..last`, `run` rows a walk.
    fn store_walks(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        rows: &[f32],
        (last, run): (usize, usize),
    ) -> Result<(), GateError> {
        let hidden = hidden_of(m)?;
        for pos0 in (0..last).step_by(run) {
            let end = (pos0 + run).min(last);
            let feed = Feed {
                tokens: &ids[pos0 + 1..end + 1],
                pos0: u32::try_from(pos0)?,
                hidden: Hidden::Host(&rows[pos0 * hidden..end * hidden]),
                media: None,
            };
            mtp_walk(m, feed, DraftHead::Full, WalkMode::Store)?;
        }
        Ok(())
    }

    /// (w) on (s)'s run.
    fn store_clause(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        rows: &[f32],
        run: &Run,
    ) -> Result<bool, GateError> {
        let last = run.drafted.len() - 1;
        let hidden = hidden_of(m)?;
        let row = |i: usize| &rows[i * hidden..(i + 1) * hidden];
        let peak_bits = |l: &[f32]| {
            let at = super::e2e::argmax(l) as usize;
            (at, l[at].to_bits())
        };
        let fed = peak_bits(&run.logits);
        // (s)'s eager steps stored positions `0..last`; read them before the
        // overwrite.
        let eager = store_of(m, last)?;
        overwrite(m, ids, rows, last)?;
        let (_, moved) = host_step(m, (ids[last + 1], last), row(last))?;
        let sensitive = !bits_equal(&moved, &run.logits);
        let before = draft_logits(m)?;
        let one = <Mtp35 as WalkBounds>::STORE_ROWS;
        store_walks(m, ids, rows, (last, one))?;
        let kept = bits_equal(&draft_logits(m)?, &before);
        let whole = store_of(m, last)?;
        let (dk, dv) = store_rel(&whole, &eager, last);
        let (bk, bv) = (key_band(), value_band());
        let band_ok = dk <= bk && dv <= bv;
        overwrite(m, ids, rows, last)?;
        store_walks(m, ids, rows, (last, RUN))?;
        let apart = store_of(m, last)?;
        let alone = apart == whole;
        let (token, logits) = host_step(m, (ids[last + 1], last), row(last))?;
        let finite = logits.iter().all(|v| v.is_finite())
            && token as usize == super::e2e::argmax(&logits) as usize;
        println!(
            "(w) other rows below position {last} move its logits: peak {:?} against \
             (s)'s {fed:?} {}",
            peak_bits(&moved),
            verdict(sensitive)
        );
        println!(
            "(w) one store walk of {last} rows (the walk takes {one}) leaves the draft's logits \
             bit for bit {}",
            verdict(kept)
        );
        println!(
            "(w) that walk's stored keys within {dk:.3e} (band {bk:.3e}) and values within \
             {dv:.3e} (band {bv:.3e}) of (s)'s eager rows {}",
            verdict(band_ok)
        );
        println!(
            "(w) store walks of {RUN} rows over the same positions, after an overwrite, store the \
             one walk's keys and values bit for bit {}",
            verdict(alone)
        );
        println!(
            "(w) the step at position {last} after them: token {token} ((s): {}), finite logits \
             {}",
            run.drafted[last],
            verdict(finite)
        );
        m.reset()?;
        m.step(&ids[..1])?;
        m.step(&ids[1..2])?;
        let normed = m.head_normed()?;
        let (want, want_logits) = host_step(m, (ids[2], 1), &normed)?;
        let mut out = [0u32; 1];
        let feed = Feed {
            tokens: &ids[2..3],
            pos0: 1,
            hidden: Hidden::Target {
                walk: MtpArena::Step,
                first: 0,
            },
            media: None,
        };
        mtp_chain(m, feed, 0, DraftHead::Full, WalkMode::Eager, &mut out, None)?;
        let on_card = out[0] == want && bits_equal(&draft_logits(m)?, &want_logits);
        println!(
            "(w) the step row on the card: token {} (read back: {want}), logits bit for bit {}",
            out[0],
            verdict(on_card)
        );
        Ok(sensitive && kept && band_ok && alone && finite && on_card)
    }

    /// (r): the refusals on the live draft, then the last step again.
    fn refusals(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        rows: &[f32],
        last: u32,
    ) -> Result<bool, GateError> {
        let stream = m.gpu().stream_handle();
        let i = ids.len() - 2;
        let hidden = hidden_of(m)?;
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, hidden)?;
        h.copy_from_host(&stream, &rows[i * hidden..(i + 1) * hidden])?;
        let short = DeviceBuffer::<f32>::zeroed(&stream, hidden - 1)?;
        let vocab = m.n_vocab()?;
        let mut ok = true;
        let cases: [(&str, Result<u32, GpuError>, &str); 3] = [
            (
                "a position past the K/V rows",
                m.mtp_draft(ids[i + 1], CTX, &h),
                "on a draft of",
            ),
            (
                "a token past the vocabulary",
                m.mtp_draft(u32::try_from(vocab)?, i, &h),
                "past the vocabulary",
            ),
            (
                "a hidden row too short",
                m.mtp_draft(ids[i + 1], i, &short),
                "a hidden row of",
            ),
        ];
        for (what, r, want) in cases {
            let text = r.as_ref().err().map(ToString::to_string);
            let named = text.as_deref().is_some_and(|e| e.contains(want));
            ok &= named;
            println!(
                "(r) {what}: {} {}",
                text.as_deref().unwrap_or("accepted"),
                verdict(named)
            );
        }
        let again = m.mtp_draft(ids[i + 1], i, &h)?;
        let same = again == last;
        ok &= same;
        println!(
            "(r) the step the refusals interrupted repeats: token {again} (before: {last}) {}",
            verdict(same)
        );
        let extra = m.add_slots(2);
        let text = extra.as_ref().err().map(ToString::to_string);
        let named = text
            .as_deref()
            .is_some_and(|e| e.contains("one resident sequence"));
        let kept = m.slots() == 1;
        ok &= named && kept;
        println!(
            "(r) a second resident sequence: {} slots {} {}",
            text.as_deref().unwrap_or("accepted"),
            m.slots(),
            verdict(named && kept)
        );
        Ok(ok)
    }

    /// Every clause on one pinned file, one load.
    fn clauses(p: &Pin) -> Result<bool, GateError> {
        let mut ok = refusals_at_open(p)?;
        let ids = prose_ids(p.prose, PROMPT)?;
        let mut m = open_at(opts(false, p.mma), pinned(p)?)?;
        println!(
            "load {}: resident_bytes={} ctx={CTX} ubatch={UBATCH} prompt={PROMPT} hidden={}",
            p.arch,
            m.resident_bytes(),
            p.hidden
        );
        if hidden_of(&m)? != p.hidden {
            return Err(format!(
                "the draft's hidden row is {} values, the pin says {}",
                hidden_of(&m)?,
                p.hidden
            )
            .into());
        }
        ok &= load_clause(&m, &pinned(p)?, p)?;
        ok &= join_clause(m.gpu(), p.hidden)?;
        ok &= step_clauses(&mut m, &ids, p.agree)?;
        println!("{WHAT} {}: {}", p.arch, verdict(ok));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let moe = clauses(&MOE)?;
        let dense = clauses(&DENSE)?;
        let ok = moe && dense;
        println!("{WHAT}: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
