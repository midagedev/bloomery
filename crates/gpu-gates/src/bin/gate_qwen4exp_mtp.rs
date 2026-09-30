//! The Qwen3.8-Flash-Next MTP draft program's gate: the target opened by
//! its placement on the gate card with the shared draft file beside it
//! (`Body38::open_placed_mtp`, the head a list of every sixth id), the
//! draft's walks (`GpuModel::mtp_draft`) held against ik's MTP draft set
//! (refset family `mtp-qwen4exp`) and against themselves.
//!
//! What is asserted:
//! - (e) teacher-forced, every graph of the set replayed in ik's order —
//!   the warmup, then each block's graphs — each row's token, position and
//!   target hidden row (ik's `inp_tokens`, `inp_pos`, `inp_mtp_states`) fed
//!   in runs of up to eight rows: the streams `eh_proj` writes within
//!   [`EH_BAND`] of ik's `mtp_eh_proj-48`, a row at a time.
//! - (l) each row's streams after the layer within [`L_OUT_BAND`] of ik's
//!   `l_out-48`, but on a row whose routed set is not ik's: such a flip is
//!   allowed only when each exchanged pair's gap in ik's router logits lies
//!   within our two logits' error, that error within [`flip_cap`] and the
//!   gap within [`margin_cap`] of the router input's band
//!   ([`ROUTER_BAND`]); its row's streams and logits are then printed, not
//!   held.
//! - (h) the full head's logits of each row ik computes one for (its
//!   `inp_out_ids`) within [`LOGITS_BAND`] of ik's `result_output`, and the
//!   row's draft token the argmax of our logits, first index on a tie; it
//!   equals ik's argmax wherever ik's top-2 margin clears
//!   [`margin_cap`] at [`LOGITS_BAND`], and is ik's runner-up otherwise.
//! - (r) the row-list head against the full head over one walk's rows: each
//!   list row's logits the full head's row of its id, bit for bit; each
//!   row's token the id of the list's argmax; each head's probability the
//!   host's softmax maximum over its rows within the sum's rounding bound.
//! - (t) the pairing: after the set's prompt through the target by passes,
//!   the target's streams at position `q` lie within [`FREE_BAND`] of ik's
//!   warmup hidden row at position `q + 1` — ik's MTP row `p` reads the
//!   target's hidden of `p − 1` — and that shift reads closer than 0 or 2;
//!   a draft walk fed the target's streams in place equals one fed them from
//!   the host, bit for bit.
//! - (g) a captured walk equals the eager walk over the same rows, bit for
//!   bit — tokens, probabilities, streams, logits — at 1 to 4 rows with the
//!   list head and at 1 row with the full head, each capture of
//!   [`GRAPH_NODES`] nodes; two own rows (`MtpFeed::Own`) in a row, captured
//!   then replayed, equal them eager, and the first equals a host walk of
//!   the previous walk's last token and streams.
//! - (f) the walk's refusals by name — its own row before any walk and
//!   after a reset, no rows, nine rows eager, five captured, rows past the
//!   store, a walk past the positions the store holds for the sequence, a
//!   token past the vocabulary, a hidden slice of the wrong length, a capture
//!   with the taps armed — and a NaN in a hidden row raised on the fault
//!   word as the draft layer's `hc_mix`, the walk's error.
//! - (w) the windows end to end, against the plain run at the set's prompt
//!   and D3K's (depth 3,000 after it): under `app::arch::qwen4exp`'s `Draft`
//!   impl the drafted greedy ids are the plain run's for 64 tokens, with a
//!   rejected row among the windows — else the rollback path never ran and
//!   the clause is red — and the live stores after the run the plain run's
//!   at the same position; `commit(k)` is `k` steps for every k, a
//!   scripted draft keeping each exactly through the same verify and commit,
//!   at the deep prompt its rejected row completing a pool leaving the
//!   pooled planes the plain run's; and the lane word planted on a lane no
//!   call wrote makes the next drafted pass raise the delta stamp fault, by
//!   name, and poisons the model.
//!
//! (l) and (h) are held on at least one row each: a run that excuses every
//! row as a flip fails.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen4exp_mtp: built without the `gpu` feature; see `just gate-gpu-qwen4exp-mtp`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_mtp", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use std::time::Instant;

    use bloomery_gpu::GpuError;
    use bloomery_gpu::arch::qwen3moe::{
        Body38, MTP_GRAPH_ROWS, MTP_ROWS, MtpDraft, MtpFeed, MtpHead, MtpHidden, MtpMode, Prompt38,
        Qwen38Model, Store38Host, TargetRows,
    };
    use bloomery_gpu::fault::FaultSite;
    use bloomery_gpu_gates::flip::{self, Flip};
    use bloomery_gpu_gates::rounding::{U, gamma, q8_32_rel};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, verdict};
    use gguf::Split;
    use model::arch::models::HeadRows;
    use model::arch::qwen35moe::place::{MtpInputs, PlanInputs, machine, vocab_sha256};
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::IK;
    use refset::arch::qwen4exp::MODEL;
    use refset::arch::qwen4exp::mtp::{DRAFT, MTP, MTP_SET};
    use refset::ik::Layout;
    use refset::mtpref::{Graph, MtpSet};
    use runtime::{Advance as _, Committed, Draft, PassSink, TapNeed, Target as _};

    /// Cache rows: the e2e gate's, and the load gate's.
    const CTX: u64 = 3072;
    /// The head's rows: every sixth id, 0 to 245,754, the load gate's list.
    const LIST_ROWS: u32 = 40_960;
    /// The draft layer's index in the draft file: ik names its nodes by it.
    const LAYER: usize = 48;
    /// Streams of a hidden row, and the values of one.
    const STREAMS: usize = 4;
    const HIDDEN: usize = 2560;
    const WIDE: usize = STREAMS * HIDDEN;
    /// ggml's IMROPE position streams per row.
    const POS_SECTIONS: usize = 4;
    /// The router's experts and routed picks.
    const N_EXPERT: usize = 512;
    const N_USED: usize = 10;

    /// PIN(2026-09-30): the streams `eh_proj` writes against ik's on the same
    /// inputs. Both sides compute the two norms in f32 (a few roundings
    /// each); the projection is the one term: ik quantizes its 5,120-value
    /// input to 32-value q8 blocks, ours reads it in f32, and a q8
    /// activation moves a projection's output by at most
    /// `rounding::q8_32_rel` = 1.2858e-2 of its RMS (the largest crest).
    fn eh_band() -> f64 {
        q8_32_rel()
    }

    /// PIN(2026-09-30): the layer's output against ik's on the same inputs,
    /// the error model of the e2e gate's `gemm_band` counted over the
    /// draft's projections in series whose outputs join the streams, where
    /// ik's side reads q8 activations and ours f32: `eh_proj`, the
    /// attention's input (q, k and v read one quantized row) and output
    /// projections, the routed experts' gate·up and down (ours
    /// `q8_0_gemv_sel_f32` over f32 rows, ik's CPU `mul_mat_id` over q8
    /// rows) and the shared expert's gate·up and down: seven terms of
    /// [`q8_32_rel`] each, independent, √7 · 1.2858e-2 = 3.40e-2. The mixes'
    /// down and up reach the streams through σ (slope at most ¼) and add
    /// under 4 %, as the e2e derivation states; the f16 store both sides'
    /// attention reads rounds at 2^-11, under 1/25 of a term.
    fn l_out_band() -> f64 {
        7f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-09-30): the head's logits against ik's: [`l_out_band`]'s
    /// seven terms and the output projection's own, √8 · 1.2858e-2 =
    /// 3.64e-2 (the head mix's down and up as above).
    fn logits_band() -> f64 {
        8f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-09-30): the router's input against ik's: `eh_proj` and the
    /// attention's input and output projections, √3 · 1.2858e-2 = 2.23e-2.
    fn router_band() -> f64 {
        3f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-09-30): a captured walk's nodes at 1 to 4 rows, derived from
    /// the walk's launch list (`mtp38`'s module doc): the embedding and the
    /// pack (2); `eh_proj` over the `4·m` packed columns in runs of eight
    /// (1, 1, 2, 2); the attention site — its mix (3), q, k, v, the norm and
    /// append, the flash's segment pass and merge, the gate, the output
    /// projection (11); the feed-forward site — its mix (3), the router, the
    /// places, the routed gate, up, SwiGLU and down, the slots' sum, the
    /// shared expert's gate, up, SwiGLU and down, the gated sum (15); the
    /// head — its mix (3), the projection, the argmax (5); the two copies of
    /// the last row (2). 35 + ⌈4m/8⌉.
    const GRAPH_NODES: [usize; 4] = [36, 36, 37, 37];

    /// PIN(2026-09-30): [provisional — backlog] the target's last-layer
    /// streams after the prompt by passes against ik's hidden rows, the e2e
    /// gate's `FREE_BAND` borrowed: its derivation is GLM's (√45 ·
    /// 1.415e-2) carried to this model's 48 layers, not a Qwen3.8 forced
    /// arm, which neither gate has; a forced arm's worst layer replaces it.
    const FREE_BAND: f64 = 0.10;

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch, so
    /// neither passes a band.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The RMS of `v`.
    fn rms(v: &[f32]) -> f64 {
        (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
    }

    /// The deviation of `v` about its mean: the spread a ranking reads.
    fn spread(v: &[f32]) -> f64 {
        let n = v.len().max(1) as f64;
        let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        (v.iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt()
    }

    /// The most error two of a row's values may carry at an excused flip:
    /// each value moves by about `band` times the row's RMS (a dot of an
    /// input carrying `band` of relative error), three deviations each —
    /// the e2e gate's `flip_cap`.
    fn flip_cap(band: f64, v: &[f32]) -> f64 {
        6.0 * band * rms(v)
    }

    /// The widest gap between two of a row's values the errors can cross:
    /// `band` times the row's spread (a common offset moves no rank), three
    /// deviations each — the e2e gate's `margin_cap`.
    fn margin_cap(band: f64, v: &[f32]) -> f64 {
        6.0 * band * spread(v)
    }

    /// The first index of the largest value, and the largest value of the
    /// rest: the head's tie rule.
    fn top2(v: &[f32]) -> (usize, usize) {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        let mut second = usize::from(best == 0);
        for (i, &x) in v.iter().enumerate() {
            if i != best && x > v[second] {
                second = i;
            }
        }
        (best, second)
    }

    /// `1 / Σ exp(v_i − max)` in f64: the largest softmax probability.
    fn p_max(v: &[f32]) -> f64 {
        let top = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        1.0 / v
            .iter()
            .map(|&x| (f64::from(x) - f64::from(top)).exp())
            .sum::<f64>()
    }

    /// The rounding bound on our probability over `n` rows: each thread's
    /// share of `⌈n/1024⌉` terms in order, the butterfly's five, the 32 warp
    /// slots in order, each term's `exp` within two ulps, and the division.
    fn p_bound(n: usize) -> f64 {
        gamma(n.div_ceil(1024) + 5 + 32 + 1) + 4.0 * U
    }

    /// Column `c` of a `[row][m]` logits readback.
    fn column(v: &[f32], m: usize, c: usize) -> Vec<f32> {
        v.iter().skip(c).step_by(m).copied().collect()
    }

    /// One graph of the set: its block, label and rows, ik's side.
    struct IkGraph {
        block: i32,
        graph: Graph,
        tokens: Vec<u32>,
        pos: Vec<u32>,
        states: Vec<f32>,
        out_ids: Vec<usize>,
        eh: Vec<f32>,
        l_out: Vec<f32>,
        logits: Vec<f32>,
        router: Vec<f32>,
        topk: Vec<i32>,
    }

    impl IkGraph {
        fn read(set: &MtpSet, block: i32, graph: Graph) -> Result<IkGraph, GateError> {
            let find = |name: &str| set.find(block, graph, name, 0);
            let ints = |name: &str| -> Result<Vec<i32>, GateError> {
                Ok(set.i32s(find(name)?, Layout::Flat)?)
            };
            let f32s =
                |name: &str| -> Result<Vec<f32>, GateError> { Ok(set.logical_f32s(find(name)?)?) };
            let unsigned = |name: &str| -> Result<Vec<u32>, GateError> {
                ints(name)?
                    .into_iter()
                    .map(|v| u32::try_from(v).map_err(|_| format!("{name} holds {v}").into()))
                    .collect()
            };
            let tokens = unsigned("inp_tokens")?;
            let n = tokens.len();
            // ggml's multi-section positions (IMROPE): four streams of n, the
            // first the text position; a text row's second and third equal it.
            let sections = unsigned("inp_pos")?;
            if sections.len() != POS_SECTIONS * n
                || sections[n..3 * n].chunks(n).any(|s| s != &sections[..n])
            {
                return Err(format!(
                    "block {block} {}: inp_pos holds {} values, not {POS_SECTIONS} sections of \
                     {n} with the second and third the first's",
                    graph.as_str(),
                    sections.len()
                )
                .into());
            }
            let pos = sections[..n].to_vec();
            let out_ids = unsigned("inp_out_ids")?
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();
            let topk_row = find(&format!("ffn_moe_topk-{LAYER}"))?;
            let topk = set.i32s(topk_row, Layout::Logical)?;
            let g = IkGraph {
                block,
                graph,
                states: f32s("inp_mtp_states")?,
                eh: f32s(&format!("mtp_eh_proj-{LAYER}"))?,
                l_out: f32s(&format!("l_out-{LAYER}"))?,
                logits: f32s("result_output")?,
                router: f32s(&format!("ffn_moe_logits-{LAYER}"))?,
                topk,
                tokens,
                pos,
                out_ids,
            };
            let shapes = [
                ("inp_pos", g.pos.len(), n),
                ("inp_mtp_states", g.states.len(), n * WIDE),
                ("mtp_eh_proj", g.eh.len(), n * WIDE),
                ("l_out", g.l_out.len(), n * WIDE),
                ("ffn_moe_logits", g.router.len(), n * N_EXPERT),
                ("ffn_moe_topk", g.topk.len(), n * N_USED),
            ];
            if let Some((what, got, want)) = shapes.iter().find(|(_, got, want)| got != want) {
                return Err(format!(
                    "block {block} {}: {what} holds {got} values, {n} rows take {want}",
                    graph.as_str()
                )
                .into());
            }
            if g.pos.windows(2).any(|w| w[1] != w[0] + 1)
                || g.out_ids.iter().any(|&o| o >= n)
                || g.topk.iter().any(|&e| !(0..N_EXPERT as i32).contains(&e))
            {
                return Err(format!(
                    "block {block} {}: positions not consecutive, an output row past {n}, or a \
                     routed id past {N_EXPERT}",
                    graph.as_str()
                )
                .into());
            }
            Ok(g)
        }

        fn label(&self) -> String {
            format!("block {} {}", self.block, self.graph.as_str())
        }
    }

    /// The set's graphs in the order ik computed them: the warmup, then each
    /// verified block's graphs in file order.
    fn graphs_of(set: &MtpSet) -> Vec<(i32, Graph)> {
        let mut order: Vec<(i32, Graph)> = Vec::new();
        for r in &set.rows {
            if !order.contains(&(r.block, r.graph)) {
                order.push((r.block, r.graph));
            }
        }
        order
    }

    /// A walk's readback: the draft, the streams and the logits.
    struct Walk {
        draft: MtpDraft,
        l_out: Vec<f32>,
        logits: Vec<f32>,
        rows: usize,
    }

    fn walk(
        m: &mut Qwen38Model,
        feed: MtpFeed<'_>,
        head: MtpHead,
        mode: MtpMode,
    ) -> Result<Walk, GateError> {
        let draft = m.mtp_draft(feed, head, mode)?;
        let l_out = m.mtp_l_out()?;
        let (logits, rows) = m.mtp_logits()?;
        Ok(Walk {
            draft,
            l_out,
            logits,
            rows,
        })
    }

    /// Two walks' readbacks, bit for bit.
    fn same_bits(a: &Walk, b: &Walk) -> bool {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        a.draft.tokens == b.draft.tokens
            && bits(&a.draft.p) == bits(&b.draft.p)
            && bits(&a.l_out) == bits(&b.l_out)
            && bits(&a.logits) == bits(&b.logits)
    }

    /// The worst of each clause over the replay, and the counts.
    #[derive(Default)]
    struct Replay {
        eh_worst: f64,
        eh_fail: usize,
        l_out_worst: f64,
        l_out_fail: usize,
        logits_worst: f64,
        logits_fail: usize,
        rows: usize,
        heads: usize,
        flips: usize,
        flips_bad: usize,
        argmax_same: usize,
        l_out_held: usize,
        heads_held: usize,
        argmax_tie: usize,
        argmax_bad: usize,
        kernel_bad: usize,
    }

    /// Replay graph `g` in runs of up to [`MTP_ROWS`] rows, the full head,
    /// the taps armed, into `r`.
    fn replay(m: &mut Qwen38Model, g: &IkGraph, r: &mut Replay) -> Result<(), GateError> {
        let n = g.tokens.len();
        let vocab = m.body("replay")?.vocab();
        if g.logits.len() != g.out_ids.len() * vocab {
            return Err(format!(
                "{}: result_output holds {} values, {} output rows of {vocab}",
                g.label(),
                g.logits.len(),
                g.out_ids.len()
            )
            .into());
        }
        let mut c0 = 0;
        while c0 < n {
            let rows = MTP_ROWS.min(n - c0);
            let w = walk(
                m,
                MtpFeed::Rows {
                    tokens: &g.tokens[c0..c0 + rows],
                    pos0: g.pos[c0],
                    hidden: MtpHidden::Host(&g.states[c0 * WIDE..(c0 + rows) * WIDE]),
                },
                MtpHead::Full,
                MtpMode::Eager,
            )?;
            let taps = m.mtp_taps()?;
            let mut flipped = vec![false; rows];
            for t in 0..rows {
                let at = c0 + t;
                let span = at * WIDE..(at + 1) * WIDE;
                let e = rel(&taps.eh[t * WIDE..(t + 1) * WIDE], &g.eh[span.clone()]);
                r.eh_worst = r.eh_worst.max(e);
                if e > eh_band() {
                    r.eh_fail += 1;
                    println!("(e) {} row {at}: eh_proj {e:.3e} FAIL", g.label());
                }
                let ours_ids: Vec<u32> =
                    taps.ids[t * taps.slots_row..t * taps.slots_row + N_USED].to_vec();
                let ours_v = &taps.logits[t * taps.logits_row..t * taps.logits_row + N_EXPERT];
                let ik_v = &g.router[at * N_EXPERT..(at + 1) * N_EXPERT];
                let ik_ids = &g.topk[at * N_USED..(at + 1) * N_USED];
                if let Some(f) = Flip::between(
                    (LAYER, at),
                    (&ours_ids, ours_v),
                    (ik_ids, ik_v),
                    flip::margin(ik_v, ik_ids),
                ) {
                    flipped[t] = true;
                    r.flips += 1;
                    let cap = flip_cap(router_band(), ik_v);
                    let gap_cap = margin_cap(router_band(), ik_v);
                    let ok = f.allowed(cap) && f.pairs.iter().all(|&(_, _, gap, _)| gap <= gap_cap);
                    if !ok {
                        r.flips_bad += 1;
                    }
                    println!(
                        "(l) {} {} (gap cap {gap_cap:.3e}) {}",
                        g.label(),
                        f.line("mtp", cap),
                        verdict(ok)
                    );
                }
                let l = rel(&w.l_out[t * WIDE..(t + 1) * WIDE], &g.l_out[span]);
                if flipped[t] {
                    println!(
                        "(l) {} row {at}: l_out {l:.3e} on a flip, not held",
                        g.label()
                    );
                } else {
                    r.l_out_held += 1;
                    r.l_out_worst = r.l_out_worst.max(l);
                    if l > l_out_band() {
                        r.l_out_fail += 1;
                        println!("(l) {} row {at}: l_out {l:.3e} FAIL", g.label());
                    }
                }
            }
            r.rows += rows;
            for c in 0..rows {
                let ours = column(&w.logits, rows, c);
                let (top, _) = top2(&ours);
                if w.rows != vocab || w.draft.tokens[c] as usize != top {
                    r.kernel_bad += 1;
                    println!(
                        "(h) {} row {}: the draft names {}, our logits' argmax {top} over {} rows \
                         FAIL",
                        g.label(),
                        c0 + c,
                        w.draft.tokens[c],
                        w.rows
                    );
                }
            }
            for (k, &o) in g.out_ids.iter().enumerate() {
                if !(c0..c0 + rows).contains(&o) {
                    continue;
                }
                let c = o - c0;
                r.heads += 1;
                let ours = column(&w.logits, rows, c);
                let ik = &g.logits[k * vocab..(k + 1) * vocab];
                let d = rel(&ours, ik);
                let (top, _) = top2(&ours);
                let (ik_top, ik_2) = top2(ik);
                let margin = f64::from(ik[ik_top]) - f64::from(ik[ik_2]);
                let clears = margin > margin_cap(logits_band(), ik);
                if flipped[c] {
                    println!(
                        "(h) {} row {o}: logits {d:.3e}, argmax {top} ik {ik_top} on a flip, not \
                         held",
                        g.label()
                    );
                    continue;
                }
                r.heads_held += 1;
                r.logits_worst = r.logits_worst.max(d);
                if d > logits_band() {
                    r.logits_fail += 1;
                    println!("(h) {} row {o}: logits {d:.3e} FAIL", g.label());
                }
                if top == ik_top {
                    r.argmax_same += 1;
                } else if !clears && top == ik_2 {
                    r.argmax_tie += 1;
                    println!(
                        "(h) {} row {o}: argmax {top}, ik's {ik_top} by {margin:.3e} under the cap \
                         {:.3e}: ik's runner-up, a tie",
                        g.label(),
                        margin_cap(logits_band(), ik)
                    );
                } else {
                    r.argmax_bad += 1;
                    println!(
                        "(h) {} row {o}: argmax {top}, ik's {ik_top} (runner-up {ik_2}) by \
                         {margin:.3e}, cap {:.3e} FAIL",
                        g.label(),
                        margin_cap(logits_band(), ik)
                    );
                }
            }
            c0 += rows;
        }
        Ok(())
    }

    /// (r): the list head against the full head over the rows of `g`'s last
    /// run.
    fn list_head(m: &mut Qwen38Model, g: &IkGraph, ids: &[u32]) -> Result<bool, GateError> {
        let n = g.tokens.len();
        let rows = MTP_ROWS.min(n);
        let c0 = n - rows;
        let feed = MtpFeed::Rows {
            tokens: &g.tokens[c0..],
            pos0: g.pos[c0],
            hidden: MtpHidden::Host(&g.states[c0 * WIDE..]),
        };
        let full = walk(m, feed, MtpHead::Full, MtpMode::Eager)?;
        let list = walk(m, feed, MtpHead::Rows, MtpMode::Eager)?;
        let mut ok = list.rows == ids.len();
        let mut differ = 0usize;
        let (mut p_worst, mut p_bad) = (0.0f64, 0usize);
        for c in 0..rows {
            let lc = column(&list.logits, rows, c);
            let fc = column(&full.logits, rows, c);
            differ += lc
                .iter()
                .zip(ids)
                .filter(|&(v, &id)| v.to_bits() != fc[id as usize].to_bits())
                .count();
            let (top, _) = top2(&lc);
            let token_ok = list.draft.tokens[c] == ids[top];
            ok &= token_ok;
            if !token_ok {
                println!(
                    "(r) row {c}: the list head names {}, its argmax row {top} is id {} FAIL",
                    list.draft.tokens[c], ids[top]
                );
            }
            for (w, v) in [(&list, &lc), (&full, &fc)] {
                let want = p_max(v);
                let e = (f64::from(w.draft.p[c]) - want).abs() / want;
                p_worst = p_worst.max(e / p_bound(v.len()));
                if e > p_bound(v.len()) {
                    p_bad += 1;
                    println!(
                        "(r) row {c}: p {} against the host's {want:.9} over {} rows, {e:.3e} past \
                         {:.3e} FAIL",
                        w.draft.p[c],
                        v.len(),
                        p_bound(v.len())
                    );
                }
            }
        }
        ok &= differ == 0 && p_bad == 0;
        println!(
            "(r) {} rows of {} ({}): list logits = the full head's rows of their ids, {differ} \
             differ; tokens mapped; p within its bound (worst {p_worst:.2} of it) {}",
            rows,
            g.label(),
            list.rows,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (t): the target's streams after the prompt against ik's warmup hidden
    /// rows, and a draft walk fed them in place against one fed them from the
    /// host.
    fn pairing(m: &mut Qwen38Model, prompt: &[u32], warm: &IkGraph) -> Result<bool, GateError> {
        m.prompt38(prompt, Prompt38::Pass)?;
        let n = prompt.len();
        let rows = (n - 1) % MTP_ROWS + 1;
        let first = n - rows;
        let ours = m.target_streams(TargetRows::Pass, rows)?;
        let at = |pos: usize| warm.pos.iter().position(|&p| p as usize == pos);
        let mut by_shift = Vec::new();
        for s in 0..3usize {
            let mut worst = 0.0f64;
            let mut pairs = 0;
            for t in 0..rows {
                if let Some(r) = at(first + t + s) {
                    worst = worst.max(rel(
                        &ours[t * WIDE..(t + 1) * WIDE],
                        &warm.states[r * WIDE..(r + 1) * WIDE],
                    ));
                    pairs += 1;
                }
            }
            println!(
                "(t) the target's streams at q against ik's MTP hidden at q + {s}: {pairs} rows, \
                 worst {worst:.3e}"
            );
            by_shift.push(if pairs > 0 { worst } else { f64::INFINITY });
        }
        let shift_ok =
            by_shift[1] <= FREE_BAND && by_shift[1] < by_shift[0] && by_shift[1] < by_shift[2];
        println!(
            "(t) ik's row p reads the target's hidden of p − 1, within {FREE_BAND:.2}: {}",
            verdict(shift_ok)
        );
        let tokens = &prompt[first..];
        let pos0 = u32::try_from(first)?;
        let host = walk(
            m,
            MtpFeed::Rows {
                tokens,
                pos0,
                hidden: MtpHidden::Host(&ours),
            },
            MtpHead::Rows,
            MtpMode::Eager,
        )?;
        let place = walk(
            m,
            MtpFeed::Rows {
                tokens,
                pos0,
                hidden: MtpHidden::Target {
                    walk: TargetRows::Pass,
                    first: 0,
                },
            },
            MtpHead::Rows,
            MtpMode::Eager,
        )?;
        let same = same_bits(&host, &place);
        println!(
            "(t) {rows} rows fed the target's streams in place = from the host, bit for bit {}",
            verdict(same)
        );
        Ok(shift_ok && same)
    }

    /// (g): captured walks against eager ones.
    fn graphs(m: &mut Qwen38Model, g: &IkGraph) -> Result<bool, GateError> {
        m.set_mtp_taps(false)?;
        let mut ok = true;
        let rows_of = |k: usize| MtpFeed::Rows {
            tokens: &g.tokens[..k],
            pos0: g.pos[0],
            hidden: MtpHidden::Host(&g.states[..k * WIDE]),
        };
        let mut shapes: Vec<(usize, MtpHead)> =
            (1..=MTP_GRAPH_ROWS).map(|k| (k, MtpHead::Rows)).collect();
        shapes.push((1, MtpHead::Full));
        for (k, head) in shapes {
            let e = walk(m, rows_of(k), head, MtpMode::Eager)?;
            let c = walk(m, rows_of(k), head, MtpMode::Graph)?;
            let again = walk(m, rows_of(k), head, MtpMode::Graph)?;
            let same = same_bits(&e, &c) && same_bits(&e, &again);
            ok &= same;
            println!(
                "(g) {k} rows, {head:?} head: captured = eager, twice {}",
                verdict(same)
            );
        }
        let pos1 = g.pos[0] + 1;
        let own = |p: u32| MtpFeed::Own { pos0: p };
        let x = walk(m, rows_of(1), MtpHead::Rows, MtpMode::Eager)?;
        let own_e = [
            walk(m, own(pos1), MtpHead::Rows, MtpMode::Eager)?,
            walk(m, own(pos1 + 1), MtpHead::Rows, MtpMode::Eager)?,
        ];
        walk(m, rows_of(1), MtpHead::Rows, MtpMode::Eager)?;
        let own_g = [
            walk(m, own(pos1), MtpHead::Rows, MtpMode::Graph)?,
            walk(m, own(pos1 + 1), MtpHead::Rows, MtpMode::Graph)?,
        ];
        walk(m, rows_of(1), MtpHead::Rows, MtpMode::Eager)?;
        let host = walk(
            m,
            MtpFeed::Rows {
                tokens: &x.draft.tokens,
                pos0: pos1,
                hidden: MtpHidden::Host(&x.l_out),
            },
            MtpHead::Rows,
            MtpMode::Eager,
        )?;
        let chain_ok = same_bits(&own_e[0], &own_g[0]) && same_bits(&own_e[1], &own_g[1]);
        let host_ok = same_bits(&own_e[0], &host);
        ok &= chain_ok && host_ok;
        println!(
            "(g) two own rows, captured then replayed = eager {}; the first = a host walk of the \
             last token and streams {}",
            verdict(chain_ok),
            verdict(host_ok)
        );
        let nodes = m
            .body("graphs")?
            .mtp()
            .map(|d| d.graph_nodes())
            .unwrap_or_default();
        for (rows, own, head, n) in &nodes {
            let want = GRAPH_NODES.get(rows - 1).copied().unwrap_or(0);
            let fits = *n == want;
            ok &= fits;
            println!(
                "(g) capture rows={rows} own={own} head={head:?}: {n} nodes (want {want}) {}",
                verdict(fits)
            );
        }
        let all = nodes.len() == MTP_GRAPH_ROWS + 2;
        ok &= all;
        println!(
            "(g) {} captures (want {}) {}",
            nodes.len(),
            MTP_GRAPH_ROWS + 2,
            verdict(all)
        );
        Ok(ok)
    }

    /// Whether `r` is a refusal whose message holds `want`.
    fn refused<T>(what: &str, r: Result<T, GpuError>, want: &str) -> bool {
        let got = r.err().map_or("ran".to_string(), |e| e.to_string());
        let ok = got.contains(want);
        println!("(f) {what}: {got} {}", verdict(ok));
        ok
    }

    /// (f): the refusals, then the NaN.
    fn refusals(m: &mut Qwen38Model, g: &IkGraph) -> Result<bool, GateError> {
        let (vocab, ctx) = {
            let b = m.body("refusals")?;
            (b.vocab(), b.mtp().map_or(0, |d| d.ctx()))
        };
        let tokens: Vec<u32> = g.tokens.iter().copied().cycle().take(9).collect();
        let states: Vec<f32> = g.states.iter().copied().cycle().take(9 * WIDE).collect();
        let feed = |k: usize, pos0: u32| MtpFeed::Rows {
            tokens: &tokens[..k],
            pos0,
            hidden: MtpHidden::Host(&states[..k * WIDE]),
        };
        let (h, e) = (MtpHead::Rows, MtpMode::Eager);
        let mut ok = true;
        ok &= refused("no rows", m.mtp_draft(feed(0, 0), h, e), "a walk of 0 rows");
        ok &= refused(
            "nine rows",
            m.mtp_draft(feed(9, 0), h, e),
            "a walk of 9 rows",
        );
        ok &= refused(
            "five captured rows",
            m.mtp_draft(feed(5, 0), h, MtpMode::Graph),
            "a walk of 5 rows",
        );
        let last = u32::try_from(ctx - 1)?;
        ok &= refused(
            "rows past the store",
            m.mtp_draft(feed(2, last), h, e),
            "in a store of",
        );
        let past = [u32::try_from(vocab)?];
        ok &= refused(
            "a token past the vocabulary",
            m.mtp_draft(
                MtpFeed::Rows {
                    tokens: &past,
                    pos0: 0,
                    hidden: MtpHidden::Host(&states[..WIDE]),
                },
                h,
                e,
            ),
            "past the vocabulary",
        );
        ok &= refused(
            "a short hidden slice",
            m.mtp_draft(
                MtpFeed::Rows {
                    tokens: &tokens[..2],
                    pos0: 0,
                    hidden: MtpHidden::Host(&states[..WIDE]),
                },
                h,
                e,
            ),
            "hidden values for 2 rows",
        );
        m.set_mtp_taps(true)?;
        ok &= refused(
            "a capture with the taps armed",
            m.mtp_draft(feed(1, 0), h, MtpMode::Graph),
            "the taps off for a captured walk",
        );
        m.set_mtp_taps(false)?;
        ok &= refused(
            "a walk past the positions the store holds",
            m.mtp_draft(feed(1, last), h, e),
            "the draft's store holds this sequence's positions below",
        );
        let mut nan = states[..WIDE].to_vec();
        nan[7] = f32::NAN;
        let r = m.mtp_draft(
            MtpFeed::Rows {
                tokens: &tokens[..1],
                pos0: 0,
                hidden: MtpHidden::Host(&nan),
            },
            h,
            e,
        );
        let raised = match &r {
            Err(GpuError::Fault { fault, .. }) => {
                fault.layer as usize == LAYER && fault.sites & (1 << FaultSite::HcMix as u32) != 0
            }
            _ => false,
        };
        println!(
            "(f) a NaN in a hidden row: {} {}",
            r.err().map_or("ran".to_string(), |e| e.to_string()),
            verdict(raised)
        );
        m.reset()?;
        ok &= refused(
            "the draft's own row after a reset",
            m.mtp_draft(MtpFeed::Own { pos0: 0 }, h, e),
            "a walk before the draft's own row",
        );
        Ok(ok && raised)
    }

    /// D3K's prefill, the e2e family's deep prompt (the set's `# tokens`).
    fn deep_prompt() -> Result<Vec<u32>, GateError> {
        let man = RefManifest::open(&data_dir().join(refset::arch::qwen4exp::D3K), &IK)?;
        let (_, _, prefill) = man.step()?;
        Ok(prefill.to_vec())
    }

    /// One plain run: `ids` prefilled by `path`, `n` tokens generated one
    /// step each, every step's stores captured. Its tokens and its stores at
    /// each position are the drafted runs' reference.
    struct Plain {
        tokens: Vec<u32>,
        stores: Vec<(Vec<Store38Host>, Vec<f32>)>,
    }

    fn plain_run(
        m: Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        n: usize,
    ) -> Result<(Qwen38Model, Plain), GateError> {
        let mut m = m;
        m.reset()?;
        let mut tokens = vec![m.prompt38(ids, path)?];
        let mut stores = Vec::with_capacity(n);
        for _ in 1..n {
            let t = m.step(&[tokens[tokens.len() - 1]])?;
            tokens.push(t);
            let (gpu, _, b) = m.body_parts("plain")?;
            stores.push(b.stores_host(gpu)?);
        }
        Ok((m, Plain { tokens, stores }))
    }

    /// The store comparison's shape: the e2e gate's `(v)` rule in one bool —
    /// the committed delta lane, the conv ring's eight slots before the
    /// count, the K/V planes' and raw keys' rows below it, the pools
    /// complete at the count (a pool a rejected row completed is read only
    /// at a count that completes it again), the PLE ring's fourteen slots
    /// before it.
    fn live_same(
        a: &(Vec<Store38Host>, Vec<f32>),
        b: &(Vec<Store38Host>, Vec<f32>),
        pos: usize,
    ) -> bool {
        const N_KV: usize = 2;
        const HEAD: usize = 256;
        const IDX_DIM: usize = 128;
        const POOL: usize = 4;
        const CONV_RING: usize = 11;
        const PLE_RING: usize = 17;
        let conv_ch = 2 * 16 * 128 + 48 * 128;
        let slots =
            |ring: usize, back: usize| (pos.saturating_sub(back)..pos).map(move |p| p % ring);
        let ring_same = |x: &[f32], y: &[f32], ring: usize, width: usize, back: usize| {
            x.len() == ring * width
                && y.len() == ring * width
                && slots(ring, back).all(|s| {
                    x[s * width..(s + 1) * width]
                        .iter()
                        .zip(&y[s * width..(s + 1) * width])
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                })
        };
        let rows_same = |x: &[u16], y: &[u16], heads: usize, width: usize| {
            let ctx = x.len() / (heads * width);
            x.len() == y.len()
                && x.len() == heads * ctx * width
                && pos <= ctx
                && (0..heads).all(|h| {
                    let at = h * ctx * width;
                    x[at..at + pos * width] == y[at..at + pos * width]
                })
        };
        let layers = a.0.len() == b.0.len()
            && a.0.iter().zip(&b.0).all(|(x, y)| match (x, y) {
                (
                    Store38Host::Rec { state: s, ring: r },
                    Store38Host::Rec {
                        state: s2,
                        ring: r2,
                    },
                ) => {
                    s.len() == s2.len()
                        && s.iter().zip(s2).all(|(u, v)| u.to_bits() == v.to_bits())
                        && ring_same(r, r2, CONV_RING, conv_ch, 8)
                }
                (
                    Store38Host::Qsa { k, v, raw, pooled },
                    Store38Host::Qsa {
                        k: k2,
                        v: v2,
                        raw: raw2,
                        pooled: pooled2,
                    },
                ) => {
                    let live = pos / POOL * IDX_DIM;
                    rows_same(k, k2, N_KV, HEAD)
                        && rows_same(v, v2, N_KV, HEAD)
                        && rows_same(raw, raw2, 1, IDX_DIM)
                        && pooled.len() == pooled2.len()
                        && live <= pooled.len()
                        && pooled[..live] == pooled2[..live]
                }
                _ => false,
            });
        layers && ring_same(&a.1, &b.1, PLE_RING, STREAMS * HIDDEN, 14)
    }

    /// One drafted pass's kept rows, for the histogram and the rejection.
    #[derive(Default)]
    struct Kept {
        rows: Vec<usize>,
        proposed: Vec<bool>,
        /// The generation's end position: the target's after the last pass.
        pos: u32,
    }

    impl PassSink<app::Session<Body38>> for Kept {
        type Error = GateError;

        fn begin(&mut self, _: &app::Session<Body38>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            t: &app::Session<Body38>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            self.rows.push(c.kept);
            self.proposed.push(c.proposed);
            self.pos = t.pos();
            Ok(())
        }
    }

    /// [`PassSink`] that hears nothing.
    struct QuietSink;

    impl PassSink<app::Session<Body38>> for QuietSink {
        type Error = GateError;

        fn begin(&mut self, _: &app::Session<Body38>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &app::Session<Body38>,
            _: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            Ok(())
        }
    }

    /// The rows-log that hears nothing.
    struct Quiet;

    impl app::RowsLog for Quiet {
        fn capture_rows(&mut self, _: usize, _: usize) -> Result<(), app::SessionError> {
            Ok(())
        }
    }

    /// The drafted session over the model: the draft opened, the verify
    /// widths captured.
    fn drafted_session(
        m: Qwen38Model,
        cfg: app::arch::qwen3moe::Q38Cfg,
        ctx: u32,
    ) -> Result<(app::Session<Body38>, app::arch::qwen3moe::Drafted38), GateError> {
        let mut s = app::Session::from_model(m, ctx);
        let draft = app::arch::qwen3moe::MtpDraft::open(s.model(), cfg)?;
        let spec = s.with_draft::<app::arch::qwen3moe::MtpDraft, 4>(draft, &mut Quiet)?;
        Ok((s, spec))
    }

    /// A draft that proposes the plain run's own next tokens, its `keep`th
    /// id one the target rejects: every window keeps exactly `keep` rows
    /// through the same verify and commit the MTP draft drives.
    struct Scripted<'a> {
        plain: &'a [u32],
        keep: usize,
        vocab: u32,
        at: usize,
    }

    impl Draft<app::Session<Body38>> for Scripted<'_> {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::Final;

        fn begin(
            &mut self,
            _: &app::Session<Body38>,
            _: &[u32],
            _: u32,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }

        fn propose(
            &mut self,
            _: &mut app::Session<Body38>,
            _: u32,
            out: &mut [u32],
        ) -> Result<usize, app::SessionError> {
            // The plain run's next tokens up to the keep − 1st, then ids it
            // never emits: the verify keeps exactly `keep` rows.
            let wrong = |want: u32| (want + 1_000) % self.vocab;
            for (i, o) in out.iter_mut().enumerate() {
                let want = self.plain.get(self.at + 1 + i).copied().unwrap_or(0);
                *o = if i < self.keep - 1 { want } else { wrong(want) };
            }
            self.at += self.keep;
            Ok(Self::WIDTH)
        }

        fn accept(
            &mut self,
            _: &mut app::Session<Body38>,
            _: &[u32],
            _: &[u32],
            _: usize,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }

        fn stepped(
            &mut self,
            _: &mut app::Session<Body38>,
            _: u32,
            _: u32,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }
    }

    /// One scripted window keeping exactly `keep` rows: the kept tokens are
    /// the plain run's, and the live stores after the commit equal its at
    /// the same position — `commit(k)` is `k` steps. The model moves in and
    /// out. The label names the prompt in the clause's line.
    fn scripted_window(
        m: Qwen38Model,
        plain: &Plain,
        ids: &[u32],
        keep: usize,
        ctx: u32,
        label: &str,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let mut s = app::Session::from_model(m, ctx);
        let vocab = u32::try_from(s.model().body("scripted")?.vocab())?;
        let mut spec = s.with_draft::<Scripted, 4>(
            Scripted {
                plain: &plain.tokens,
                keep,
                vocab,
                at: 0,
            },
            &mut Quiet,
        )?;
        let first = spec.prompt(&mut s, ids)?;
        let out = runtime::generate(
            &mut s,
            &mut spec,
            ids,
            first,
            &runtime::Stop::new(keep + 1, ctx)?,
            &mut QuietSink,
        )?;
        let mut m = s.into_model();
        let at = m.pos() as usize;
        let want = &plain.tokens[..=keep];
        let tokens_ok = out.tokens == want;
        let (gpu, _, b) = m.body_parts("scripted")?;
        let stores_ok = live_same(&b.stores_host(gpu)?, &plain.stores[keep - 1], at);
        let ok = tokens_ok && stores_ok;
        println!(
            "(w) {label}: a window that keeps {keep} -> {} tokens = the plain run's {}, the \
             live stores after the commit its at position {at} {}",
            out.tokens.len(),
            verdict(tokens_ok),
            verdict(stores_ok)
        );
        Ok((m, ok))
    }

    /// The drafted session's cfg, its prompt path `path`.
    fn step_cfg(path: Prompt38) -> app::arch::qwen3moe::Q38Cfg {
        app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: MtpMode::Graph,
        }
    }

    /// (w): the windows end to end, against the plain run — at the set's
    /// prompt and D3K's (depth 3,000 after it): the drafted greedy ids are
    /// the plain run's for 64 tokens, with a rejected row among the windows
    /// (else the rollback path never ran, and the clause is red) and the
    /// live stores after the run the plain run's at the same position;
    /// `commit(k)` is `k` steps for every k, a scripted draft keeping each
    /// exactly through the same verify and commit, at the deep prompt its
    /// second row completing a pool; and the lane word planted on a lane no
    /// call wrote makes the next drafted pass raise the delta stamp fault,
    /// by name, and poisons the model.
    fn windows(
        m: Qwen38Model,
        prompt: &[u32],
        deep: &[u32],
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 64;
        let ctx = m.body("windows")?.ctx() as u32;
        let mut ok = true;

        // The plain reference: every step's live stores, four tokens past
        // the drafted loop's count (its last pass may keep rows past it).
        let (m, shallow) = plain_run(m, prompt, Prompt38::Auto, N + 4)?;
        let (mut m, out, kept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, prompt)?;
            let mut k = Kept::default();
            let out = runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            (s.into_model(), out, k)
        };
        let ids_ok = out.tokens[..N] == shallow.tokens[..N];
        ok &= ids_ok;
        println!(
            "(w) depth {}: the drafted run's {N} ids = the plain run's {}",
            prompt.len(),
            verdict(ids_ok)
        );
        let rejections: usize = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(k, p)| **p && **k < 4)
            .count();
        ok &= rejections > 0;
        println!(
            "(w) depth {}: {} windows kept {:?}, {rejections} rejected a row {}",
            prompt.len(),
            kept.rows.len(),
            kept.rows,
            verdict(rejections > 0)
        );
        let at = (kept.pos as usize - prompt.len()).min(N + 3);
        let (gpu, _, b) = m.body_parts("windows")?;
        let stores_ok = live_same(
            &b.stores_host(gpu)?,
            &shallow.stores[at - 1],
            kept.pos as usize,
        );
        ok &= stores_ok;
        println!(
            "(w) depth {}: the live stores after the drafted run = the plain run's at position \
             {} {}",
            prompt.len(),
            kept.pos,
            verdict(stores_ok)
        );

        // commit(k) is k steps, every k, through the rule's own driving.
        for keep in 1..=4usize {
            let (model, k_ok) = scripted_window(m, &shallow, prompt, keep, ctx, "shallow")?;
            m = model;
            ok &= k_ok;
        }

        // The deep prompt: the drafted ids and stores, and a rejection whose
        // row completes a pool (the scripted window's second row, at position
        // 3,003, completes pool 750).
        let (m, deep_plain) = plain_run(m, deep, Prompt38::Auto, 4)?;
        let (mut m, out, kept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, deep)?;
            let mut k = Kept::default();
            let out = runtime::generate(
                &mut s,
                &mut spec,
                deep,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            (s.into_model(), out, k)
        };
        let ids_ok = out.tokens[..4] == deep_plain.tokens[..4];
        ok &= ids_ok;
        println!(
            "(w) depth {}: the drafted run's first 4 ids = the plain run's {}",
            deep.len(),
            verdict(ids_ok)
        );
        let rejections: usize = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(k, p)| **p && **k < 4)
            .count();
        ok &= rejections > 0;
        println!(
            "(w) depth {}: {} windows kept {:?}, {rejections} rejected a row {}",
            deep.len(),
            kept.rows.len(),
            kept.rows,
            verdict(rejections > 0)
        );
        let at = (kept.pos as usize - deep.len()).min(3);
        let (gpu, _, b) = m.body_parts("windows")?;
        let stores_ok = live_same(
            &b.stores_host(gpu)?,
            &deep_plain.stores[at - 1],
            kept.pos as usize,
        );
        ok &= stores_ok;
        println!(
            "(w) depth {}: the live stores after the drafted run = the plain run's at position \
             {} {}",
            deep.len(),
            kept.pos,
            verdict(stores_ok)
        );
        let (model, p_ok) = scripted_window(
            m,
            &deep_plain,
            deep,
            2,
            ctx,
            "deep, its rejected row completing a pool",
        )?;
        m = model;
        ok &= p_ok;

        // The lane word planted on a lane no call wrote: the next drafted
        // pass raises the delta stamp fault by name and poisons the model.
        m.reset()?;
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Step), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        {
            let (gpu, _, b) = s.model_mut().body_parts("plant")?;
            b.plant_lane(gpu, 2)?;
        }
        let r = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(2, ctx)?,
            &mut QuietSink,
        );
        let fault_ok = matches!(&r, Err(e) if e.to_string().contains("delta_stamp"));
        ok &= fault_ok;
        println!(
            "(w) the lane word planted on lane 2, never written -> {} {}",
            match &r {
                Ok(o) => format!("accepted, {} tokens", o.tokens.len()),
                Err(e) => e.to_string(),
            },
            verdict(fault_ok)
        );
        let mut m = s.into_model();
        m.reset()?;
        Ok((m, ok))
    }

    pub(super) fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[])?;
        let dir = MTP.path(MTP_SET);
        let set = MtpSet::open(&dir, &MTP)?;
        let prompt = set
            .tokens
            .clone()
            .ok_or_else(|| format!("{}: no # tokens line", dir.display()))?;
        let order = graphs_of(&set);
        println!(
            "(o) {}: {} graphs over {} blocks, prompt {} ids, family {} PASS",
            dir.display(),
            order.len(),
            set.blocks().len(),
            prompt.len(),
            MTP.name
        );
        println!(
            "bands: eh {:.4e} l_out {:.4e} logits {:.4e} router {:.4e} free {FREE_BAND:.2}",
            eh_band(),
            l_out_band(),
            logits_band(),
            router_band()
        );
        let open = |p: &str| Split::open(p).map_err(|e| format!("open {p}: {e}"));
        let (file, draft) = (open(MODEL)?, open(DRAFT)?);
        let t = Instant::now();
        let inputs = PlanInputs::describe(&file)?;
        let ids: Vec<u32> = (0..LIST_ROWS).map(|i| i * 6).collect();
        let rows = HeadRows::List {
            ids: ids.clone().into(),
            digest: vocab_sha256(&file)?,
        };
        let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX)?)?;
        let machine = machine(RTX_3090, inputs.spec.layers.len(), u64::try_from(ub)?);
        let plan = inputs.plan_mtp(&machine, CTX, &PlanLevers::from_levers(&levers)?, &mtp)?;
        let mut m =
            Body38::open_placed_mtp(file, &plan, &inputs, 0, levers.host(), ub, &draft, &mtp)?;
        let arena = m.body("mtp")?.mtp().map_or(0, |d| d.arena_bytes());
        println!(
            "load card={} ctx_max={CTX} in {:.1} s; the draft program's arena {arena} bytes \
             (runtime value, outside the plan)",
            RTX_3090.name,
            t.elapsed().as_secs_f64()
        );
        let mut ok = refused(
            "the draft's own row before any walk",
            m.mtp_draft(MtpFeed::Own { pos0: 0 }, MtpHead::Rows, MtpMode::Eager),
            "a walk before the draft's own row",
        );

        m.set_mtp_taps(true)?;
        let mut r = Replay::default();
        let mut last = None;
        let mut warm = None;
        for &(b, g) in &order {
            let ig = IkGraph::read(&set, b, g)?;
            replay(&mut m, &ig, &mut r)?;
            if g == Graph::Warmup {
                warm = Some(ig);
            } else {
                last = Some(ig);
            }
        }
        let warm = warm.ok_or("the set has no warmup graph")?;
        let last = last.ok_or("the set has no block graph")?;
        let e_ok = r.eh_fail == 0 && r.rows > 0;
        println!(
            "(e) {} rows: eh_proj worst {:.3e} (band {:.3e}), {} past {}",
            r.rows,
            r.eh_worst,
            eh_band(),
            r.eh_fail,
            verdict(e_ok)
        );
        let l_ok = r.l_out_fail == 0 && r.flips_bad == 0 && r.l_out_held > 0;
        println!(
            "(l) l_out on {} rows held, {} on flips: worst {:.3e} (band {:.3e}), {} past; {} \
             flips not allowed {}",
            r.l_out_held,
            r.flips,
            r.l_out_worst,
            l_out_band(),
            r.l_out_fail,
            r.flips_bad,
            verdict(l_ok)
        );
        let h_ok = r.logits_fail == 0 && r.argmax_bad == 0 && r.kernel_bad == 0 && r.heads_held > 0;
        println!(
            "(h) {} head rows, {} held: logits worst {:.3e} (band {:.3e}), {} past; argmax = \
             ik's {}, ties {}, wrong {}; the draft's token = our argmax but {} {}",
            r.heads,
            r.heads_held,
            r.logits_worst,
            logits_band(),
            r.logits_fail,
            r.argmax_same,
            r.argmax_tie,
            r.argmax_bad,
            r.kernel_bad,
            verdict(h_ok)
        );
        ok &= e_ok && l_ok && h_ok;
        let map = m
            .body("list")?
            .mtp()
            .and_then(|d| d.head_map())
            .map(|(map, _)| map.to_host_vec(m.gpu().stream()))
            .transpose()?
            .ok_or("the load opened no row list")?;
        ok &= map == ids;
        ok &= list_head(&mut m, &last, &map)?;
        m.set_mtp_taps(false)?;
        ok &= pairing(&mut m, &prompt, &warm)?;
        ok &= graphs(&mut m, &warm)?;
        ok &= refusals(&mut m, &warm)?;
        let (_, w_ok) = windows(m, &prompt, &deep_prompt()?)?;
        ok &= w_ok;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
