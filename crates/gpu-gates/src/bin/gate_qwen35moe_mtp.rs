//! The Qwen3.6 next-token (MTP) draft gate: a load opened with the file's
//! own next-token layer (`GpuModel::<Body35>::open_mtp`) runs one draft step
//! through Body35's own layer path (`Mtp35`).
//!
//! What is asserted, on the real unsloth `UD-Q4_K_M` file that carries the
//! layer (`blk.40`: full attention and a routed MoE with the trunk's shapes,
//! beside `nextn.{eh_proj, enorm, hnorm, shared_head_norm}`), named by
//! `BLOOMERY_MTP_MODEL` and pinned by its `general.name` and its size:
//! - (q) refusals by name at open, before any upload: the plain Qwen3.6 file
//!   (`BLOOMERY_REF_MODEL`: no `nextn` tensor, no next-token layer) fails with the in-file rule's
//!   reason, not a missing-tensor error from the loader, and a media load.
//! - (d) the draft's resident card bytes are the figure derived from the
//!   file's tensor table: the layer's own tensors (a bf16 one, the router and
//!   the shared expert's gate, widened to f32 on the card, every other type as
//!   the file stores it), one f32 stream and one dense K/V store at the open's
//!   context; the head's output matrix and the embedding are the target's own device
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
//! - (s) the draft's acceptance against the target: a 64-id prompt of the shared prose
//!   corpus (`prose_ids`, Qwen3.8's ids: the same vocabulary) through the target,
//!   one step at a time for its own next tokens and in one pass for its
//!   final-norm rows; the draft is then fed each position in order — the
//!   prompt's next id, the position and the target's row of it — so its K/V
//!   rows cover the prompt, and its token for each of the last 16 positions
//!   is compared with the token the target itself takes after that next id.
//!   Under 30 % is a wiring error (the join order, a wrong `h`, a wrong
//!   gain), and the clause is red. The draft's own K/V rows weigh little
//!   at this prompt length: a draft attending to one row stays above it.
//! - (w) the store walk and the step row, after (s): the draft's K/V rows
//!   below (s)'s last position are overwritten by steps of other ids and
//!   rows, which moves that position's logits; store walks
//!   (`WalkMode::Store`, two rows a walk) of (s)'s feed over those positions
//!   leave the draft's last logits bit for bit (no head runs), and the eager
//!   step at the last position after them is (s)'s last step bit for bit
//!   (the store holds the eager walk's keys and values). After two plain
//!   target steps, a chain at position 1 that reads the target head's row on
//!   the card (`MtpArena::Step`) gives the token and logits, bit for bit, of
//!   the same step over that row read back to the host.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_mtp", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/e2e.rs"]
#[allow(dead_code, reason = "this gate uses only the shared argmax")]
mod e2e;

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::arch::qwen3moe::{
        KvQ8, Mtp35, MtpArena, Open35, PrefillPath, Qwen35moeModel, addresses, mtp_chain, mtp_walk,
    };
    use bloomery_gpu::draft::{DraftHead, Feed, Hidden, WalkBounds, WalkMode};
    use bloomery_gpu::mtp::{MtpInputArgs, MtpKernels};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::rounding::{U, gamma};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, open_split, prose_ids, verdict,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use gguf::quant::GgmlType;
    use model::arch::Arch;
    use model::arch::models::Mixer;
    use model::arch::qwen35moe::mtp::{EH_PROJ, ENORM, HNORM, InFile, SHARED_HEAD_NORM, in_file};
    use model::arch::qwen35moe::spec;
    use runtime::stores::dense_kv_bytes;

    const WHAT: &str = "gate_qwen35moe_mtp";

    /// The lever naming the file the gate runs on: unsloth's `UD-Q4_K_M` with
    /// its next-token layer. Pinned by `general.name` and by its size in bytes.
    const DRAFT_LEVER: &str = "BLOOMERY_MTP_MODEL";
    const DRAFT_NAME: &str = "Qwen3.6-35B-A3B";
    const DRAFT_BYTES: u64 = 22_663_387_424;

    /// Cache rows of the load and the prompt's ubatch: the prompt and the
    /// draft's rows over it, with room.
    const CTX: usize = 256;
    const UBATCH: usize = 64;

    /// (s)'s prompt, and how many of its last positions are compared.
    const PROMPT: usize = 64;
    const LAST: usize = 16;

    /// The wiring floor of (s), a share of the positions.
    const FLOOR: f64 = 0.30;

    /// (j)'s embedding width and head-row count: the file's.
    const HIDDEN: usize = 2048;

    fn open_at(o: Open35, file: Split) -> Result<Qwen35moeModel, GpuError> {
        Qwen35moeModel::open_mtp(Gpu::new()?, file, o)
    }

    fn opts(media: bool) -> Open35 {
        Open35 {
            ctx: CTX,
            mma: true,
            ubatch: UBATCH,
            kv: KvQ8::F16,
            media,
        }
    }

    /// The pinned file: qwen35moe, `general.name` [`DRAFT_NAME`], one shard
    /// of [`DRAFT_BYTES`].
    fn pinned() -> Result<Split, GateError> {
        let path = std::env::var(DRAFT_LEVER).map_err(|_| {
            format!("{DRAFT_LEVER} unset: it names the {DRAFT_NAME} file with its next-token layer")
        })?;
        let file = Split::open(&path).map_err(|e| format!("open {path}: {e}"))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!("{path} is {:?}, not qwen35moe", file.architecture()).into());
        }
        let name = file.value("general.name").and_then(|v| v.as_str());
        if name != Some(DRAFT_NAME) {
            return Err(format!("{path}: general.name is {name:?}, not {DRAFT_NAME:?}").into());
        }
        let shard = file.shard_path(0).ok_or("the file has no first shard")?;
        let bytes = std::fs::metadata(shard)
            .map_err(|e| format!("{}: {e}", shard.display()))?
            .len();
        if bytes != DRAFT_BYTES {
            return Err(format!("{path} is {bytes} bytes, not {DRAFT_BYTES}").into());
        }
        Ok(file)
    }

    /// The error opening `file` with the draft gives, `None` when it opened.
    fn open_error(file: Split, o: Open35) -> Option<String> {
        open_at(o, file).err().map(|e| e.to_string())
    }

    /// (q): the refusals at open, before the card is touched.
    fn refusals_at_open() -> Result<bool, GateError> {
        let plain = open_split(Arch::Qwen35moe, "gate-gpu-qwen35moe-mtp")?;
        let r = open_error(plain, opts(false));
        let text = r.as_deref().unwrap_or("opened");
        let none_ok = r
            .as_deref()
            .is_some_and(|e| e.contains("the file has no in-file draft"))
            && r.as_deref()
                .is_some_and(|e| !e.contains("is not in the file"));
        println!(
            "(q) the file with no next-token layer: {text} {}",
            verdict(none_ok)
        );
        let r = open_error(pinned()?, opts(true));
        let text = r.as_deref().unwrap_or("opened");
        let media_ok = r.as_deref().is_some_and(|e| e.contains("a media load"));
        println!("(q) a media load: {text} {}", verdict(media_ok));
        Ok(none_ok && media_ok)
    }

    /// (d): the draft's resident bytes against the plan's figure, and the
    /// borrowed matrices at the target's addresses.
    fn load_clause(m: &Qwen35moeModel, file: &Split) -> Result<bool, GateError> {
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
        // [derived] A bf16 tensor is decoded to f32 on the card
        // (`DevWeight::F32`): two bytes a value become four; every other
        // type is resident as the file stores it. The embedding and the head
        // are the target's and carry no layer index.
        let (mut own, mut widened) = (0u64, 0u64);
        for t in read
            .tensors
            .tensors
            .iter()
            .filter(|t| t.layer == Some(mtp.index()))
        {
            if t.ty == GgmlType::BF16 {
                let wide = 4 * t.dims.iter().product::<u64>();
                widened += wide - t.file_bytes;
                own += wide;
            } else {
                own += t.file_bytes;
            }
        }
        let kv = dense_kv_bytes(g.kv_heads as usize, g.head_dim as usize, CTX);
        let stream = (HIDDEN * 4) as u64;
        let want = own + stream + kv;
        let got = mtp.resident_bytes() as u64;
        let bytes_ok = got == want;
        println!(
            "(d) draft resident {got} = derived {want} ({}): layer {own} (bf16 widened \
             {widened}) + stream {stream} + K/V {kv}; weights {}; step scratch {} {}",
            got as i64 - want as i64,
            mtp.weights().resident_bytes(),
            mtp.scratch_bytes(),
            verdict(bytes_ok)
        );
        let mut ok = bytes_ok && widened > 0;
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
    /// [`HIDDEN`]) at `streams`: the pack, or the launch's error.
    fn join(
        gpu: &Gpu,
        k: &MtpKernels,
        (e, h): (&[f32], &[f32]),
        (enorm, hnorm): (&[f32], &[f32]),
        streams: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let stream = gpu.stream();
        let eb = DeviceBuffer::from_host(stream, e)?;
        let hb = DeviceBuffer::from_host(stream, h)?;
        let en = DeviceBuffer::from_host(stream, enorm)?;
        let hn = DeviceBuffer::from_host(stream, hnorm)?;
        let mut out = DeviceBuffer::<f32>::zeroed(stream, 2 * 4 * HIDDEN)?;
        k.enqueue_mtp_input(
            stream,
            MtpInputArgs {
                e: &eb,
                h: &hb,
                enorm: &en,
                hnorm: &hn,
                hidden: HIDDEN,
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
    fn join_clause(gpu: &Gpu) -> Result<bool, GateError> {
        let k = MtpKernels::load(gpu.context())?;
        let (e, enorm) = (values(HIDDEN, 1, 3.0), values(HIDDEN, 2, 1.0));
        let (h4, hnorm4) = (values(4 * HIDDEN, 3, 5.0), values(4 * HIDDEN, 4, 1.0));
        let one = join(gpu, &k, (&e, &h4[..HIDDEN]), (&enorm, &hnorm4[..HIDDEN]), 1)?;
        // [derived] A value is `x · r · γ`: r goes through the thread's
        // `HIDDEN/256` fused terms, the butterfly's five adds and the eight
        // warps' seven, 20 roundings in the sum, half of them in r, then the
        // add of eps, the root, the reciprocal, and the two products.
        let band = 0.5 * gamma(HIDDEN / 256 + 5 + 7) + 5.0 * U;
        let (re, rh) = (
            rms_ref(&e, &enorm, 1e-6),
            rms_ref(&h4[..HIDDEN], &hnorm4[..HIDDEN], 1e-6),
        );
        let worst = |got: &[f32], want: &[f64]| {
            got.iter()
                .zip(want)
                .map(|(&g, &w)| (f64::from(g) - w).abs() / w.abs().max(1e-30))
                .fold(0.0f64, f64::max)
        };
        let (we, wh) = (
            worst(&one[..HIDDEN], &re),
            worst(&one[HIDDEN..2 * HIDDEN], &rh),
        );
        let one_ok = one[..2 * HIDDEN].iter().all(|v| v.is_finite()) && we <= band && wh <= band;
        println!(
            "(j) one stream: e half worst {we:.3e}, h half {wh:.3e} of band {band:.3e} {}",
            verdict(one_ok)
        );
        let four = join(gpu, &k, (&e, &h4), (&enorm, &hnorm4), 4)?;
        let slots_ok = (0..4).all(|s| {
            four[s * 2 * HIDDEN..s * 2 * HIDDEN + HIDDEN]
                .iter()
                .zip(&one[..HIDDEN])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
        println!(
            "(j) four streams: the embedding half is the one-stream pack's bits in all four \
             slots {}",
            verdict(slots_ok)
        );
        let mut refused = true;
        for streams in [0usize, 2, 3, 5] {
            let r = join(gpu, &k, (&e, &h4), (&enorm, &hnorm4), streams)
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
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, HIDDEN)?;
        let (mut drafted, mut argmax_ok, mut logits) = (Vec::new(), 0, Vec::new());
        for i in 0..ids.len() - 1 {
            h.copy_from_host(&stream, &rows[i * HIDDEN..(i + 1) * HIDDEN])?;
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
    fn step_clauses(m: &mut Qwen35moeModel, ids: &[u32]) -> Result<bool, GateError> {
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
        let at = super::e2e::argmax(&run.logits) as usize;
        println!(
            "(s) the last draft step's largest logit {:.6} at token {}",
            run.logits[at], at
        );
        let r_ok = refusals(m, ids, &rows, run.drafted[steps - 1])?;
        let w_ok = store_clause(m, ids, &rows, &run)?;
        Ok(h_ok && r_ok && wired && w_ok)
    }

    /// The draft's step `t` at position `pos` over the host row `row`, and its
    /// logits.
    fn host_step(
        m: &mut Qwen35moeModel,
        (t, pos): (u32, usize),
        row: &[f32],
    ) -> Result<(u32, Vec<f32>), GateError> {
        let stream = m.gpu().stream_handle();
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, HIDDEN)?;
        h.copy_from_host(&stream, row)?;
        let token = m.mtp_draft(t, pos, &h)?;
        Ok((token, draft_logits(m)?))
    }

    /// (w) on (s)'s run.
    fn store_clause(
        m: &mut Qwen35moeModel,
        ids: &[u32],
        rows: &[f32],
        run: &Run,
    ) -> Result<bool, GateError> {
        let last = run.drafted.len() - 1;
        let row = |i: usize| &rows[i * HIDDEN..(i + 1) * HIDDEN];
        let peak_bits = |l: &[f32]| {
            let at = super::e2e::argmax(l) as usize;
            (at, l[at].to_bits())
        };
        let fed = peak_bits(&run.logits);
        for i in 0..last {
            host_step(m, (ids[i + 2], i), row(i + 1))?;
        }
        let (_, moved) = host_step(m, (ids[last + 1], last), row(last))?;
        let sensitive = !bits_equal(&moved, &run.logits);
        let before = draft_logits(m)?;
        for pos0 in (0..last).step_by(<Mtp35 as WalkBounds>::STORE_ROWS) {
            let end = (pos0 + <Mtp35 as WalkBounds>::STORE_ROWS).min(last);
            let feed = Feed {
                tokens: &ids[pos0 + 1..end + 1],
                pos0: u32::try_from(pos0)?,
                hidden: Hidden::Host(&rows[pos0 * HIDDEN..end * HIDDEN]),
                media: None,
            };
            mtp_walk(m, feed, DraftHead::Full, WalkMode::Store)?;
        }
        let kept = bits_equal(&draft_logits(m)?, &before);
        let (token, logits) = host_step(m, (ids[last + 1], last), row(last))?;
        let same = token == run.drafted[last] && bits_equal(&logits, &run.logits);
        println!(
            "(w) other rows below position {last} move its logits: peak {:?} against \
             (s)'s {fed:?} {}",
            peak_bits(&moved),
            verdict(sensitive)
        );
        println!(
            "(w) store walks over positions 0..{last} leave the draft's logits bit for bit {}",
            verdict(kept)
        );
        println!(
            "(w) the step at position {last} after them: token {token} (s): {}, logits \
             bit for bit {}",
            run.drafted[last],
            verdict(same)
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
        Ok(sensitive && kept && same && on_card)
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
        let mut h = DeviceBuffer::<f32>::zeroed(&stream, HIDDEN)?;
        h.copy_from_host(&stream, &rows[i * HIDDEN..(i + 1) * HIDDEN])?;
        let short = DeviceBuffer::<f32>::zeroed(&stream, HIDDEN - 1)?;
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

    pub fn run() -> Result<(), GateError> {
        let mut ok = refusals_at_open()?;
        let file = pinned()?;
        let ids = prose_ids("qwen4exp", PROMPT)?;
        let mut m = open_at(opts(false), file)?;
        println!(
            "load: resident_bytes={} ctx={CTX} ubatch={UBATCH} prompt={PROMPT}",
            m.resident_bytes()
        );
        let file = pinned()?;
        ok &= load_clause(&m, &file)?;
        ok &= join_clause(m.gpu())?;
        ok &= step_clauses(&mut m, &ids)?;
        println!("{WHAT}: {}", verdict(ok));
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
