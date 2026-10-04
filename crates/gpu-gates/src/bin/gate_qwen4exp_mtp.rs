//! The Qwen3.8-Flash-Next MTP draft program's gate: the target opened by
//! its placement on the gate card with the shared draft file beside it
//! (`Body38::open_placed_mtp`, the head a list of every sixth id), the
//! draft's load held to its plan, and the draft's walks
//! (`GpuModel::mtp_draft`) held against ik's MTP draft set (refset family
//! `mtp-qwen4exp`) and against themselves.
//!
//! What is asserted:
//! - (d) right after the load, before any walk: the draft is open: its
//!   weights, its store and the row map hold the bytes its plan
//!   (`PlanInputs::plan_mtp`) gives; the two matrices it borrows
//!   (`token_embd`, `output`) are the target's own buffers, at the target's
//!   addresses — nothing copied, the list's rows among them; the program's
//!   arena is the plan's; the map holds one word a vocabulary id, its first
//!   the list's ids and the rest 0.
//! - (u) before the load: the head `BLOOMERY_MTP_HEAD_ROWS` unset picks on
//!   this target (`head_list::head_rows_of`) is the shipped list — its
//!   65,536 rows, its first line naming the target tokenizer's digest, the
//!   pick `shipped` — and `full` is the full head.
//! - (q) `Mtp38::open` refuses by name, beside the loaded model: a draft
//!   read as carrying its own matrices, and a plan whose row map is not the
//!   one the load makes (after the draft's uploads, which it frees).
//! - (e) teacher-forced, every graph of the set replayed in ik's order —
//!   the warmup, then each block's graphs — each row's token, position and
//!   target hidden row (ik's `inp_tokens`, `inp_pos`, `inp_mtp_states`) fed
//!   in runs of up to eight rows: the streams `eh_proj` writes within
//!   [`eh_band`] of ik's `mtp_eh_proj-48`, a row at a time.
//! - (l) each row's routed set is ik's (`ffn_moe_topk-48`), but where it is
//!   not: such a flip is allowed only when each exchanged pair's gap in
//!   ik's router logits lies within our two logits' error, that error
//!   within [`flip_cap`] and the gap within [`margin_cap`] of the router
//!   input's band ([`router_band`]). The layer's output's distance from
//!   ik's `l_out-48` on the other rows is printed, not held: our f32
//!   activations against ik's q8 ones move each projection by its own
//!   rounding term, which the node-local pins (n) hold instead.
//! - (h) each row's draft token the argmax of our full head's logits, first
//!   index on a tie; on each row ik computes a head for (its `inp_out_ids`)
//!   that argmax is ik's wherever ik's top-2 margin clears [`margin_cap`] at
//!   [`logits_band`], and ik's runner-up otherwise. The logits' distance
//!   from ik's `result_output` is printed, not held.
//! - (n) the node-local pins, on every row: each node the walk taps against
//!   the host's f64 simulation of our own rule on that node's inputs as the
//!   walk read them back, within the bound of the node's own rounding —
//!   `eh_proj` from the fed token and hidden row (the input pack, then the
//!   projection), the attention site's mix, `AttnOut`, the attention's
//!   combine, the feed-forward site's mix, the routed slots' downs and
//!   their weighted sum, the shared expert's down, the gated sum (bit for
//!   bit), `l_out` (the head site's combine), the head site's mix and, on
//!   ik's head rows, the full head's logits. A dot's bound is `γ(k/32 +
//!   5)·Σ|w·x|` ([`dot_depth`]), a mix's the hyper-connection gate's
//!   `MIXED_BAND`, a combine's its weight's `LO_BAND` and one rounding.
//! - (r) the row-list head (`q8_0_gemv_ids` and `_mcol`, the list's rows of
//!   `output` read in place) against the full head over one walk's rows and
//!   over a walk of its last row alone: each list row's logits the full
//!   head's row of its id, bit for bit; each
//!   row's token the id of the list's argmax; each head's probability the
//!   host's softmax maximum over its rows within the sum's rounding bound.
//! - (t) the pairing: after the set's prompt through the target by passes,
//!   the target's streams at position `q` lie within [`FREE_BAND`] of ik's
//!   warmup hidden row at position `q + 1` — ik's MTP row `p` reads the
//!   target's hidden of `p − 1` — and that shift reads closer than 0 or 2;
//!   a draft walk from the position after the prompt's last unit's first,
//!   its rows reading the unit's arena rows in place, equals one fed them
//!   from the host, bit for bit (a walk error here is a FAIL line, not the
//!   gate's end).
//! - (g) a captured walk equals the eager walk over the same rows, bit for
//!   bit — tokens, probabilities, streams, logits — at 1 to 4 rows with the
//!   list head and at 1 row with the full head, each capture of
//!   [`GRAPH_NODES`] nodes; two own rows (`MtpFeed::Own`) in a row, captured
//!   then replayed, equal them eager, and the first equals a host walk of
//!   the previous walk's last token and streams; a window's chain
//!   (`GpuModel::mtp_chain`: a refresh and two own walks, one readback)
//!   reads back each walk's last id and probability, bit for bit.
//! - (f) the walk's refusals by name — its own row before any walk and
//!   after a reset, no rows, nine rows eager, five captured, rows past the
//!   store, a walk past the positions the store holds for the sequence, a
//!   token past the vocabulary, a hidden slice of the wrong length, a capture
//!   with the taps armed; a store walk (`MtpMode::Store`) past its
//!   `MTP_STORE_ROWS` rows, read back or chained, and the own row and the
//!   streams after one — and a NaN in a
//!   hidden row raised on the fault word as the draft layer's `hc_mix`, the
//!   walk's error; and a walk that reads the target's arenas by position
//!   (`MtpHidden::Target`): after the prompt by passes, one that reads the
//!   position before the Pass arena's first, one past its last row and one
//!   that starts its rows at the wrong arena row are each refused by name
//!   with the positions the arena holds, and the walk the arena holds runs.
//! - (s) the store walks: over a prompt's rows fed the target's hidden rows
//!   from the host (position 0 beside a zero row, each later position beside
//!   the row of the one before it), store walks in runs of three leave each
//!   position's keys within `key_band` and its values within `value_band`
//!   of what eager walks over the same rows store — the store walk's
//!   projections read q8 activations of 32 values, the eager walk's the f32
//!   rows.
//! - (w) the windows end to end: first the drafted session's prompt call
//!   walks the draft as (s)'s store walks: its warmup, which walks the store
//!   alone in walks of its own width, leaves every prompt position's keys
//!   and values those store walks stored, bit for bit, over a store other
//!   ids overwrote first — a row's stored bits are its own inputs' alone —
//!   and the anchor walk after them is bit for bit the same; then
//!   against the plain run at the set's prompt
//!   and D3K's (depth 3,000 after it): under `app::arch::qwen4exp`'s `Draft`
//!   impl the drafted greedy ids are the plain run's for 64 tokens, with a
//!   rejected row among the windows — else the rollback path never ran and
//!   the clause is red — and the live stores after the run the plain run's
//!   at the same position; at the set's prompt the draft keeps no window
//!   unset, and the same run keeping its windows (`BLOOMERY_MTP_WINDOWS`'s
//!   `MtpDraft::keep_windows`) reads back the same ids through the same
//!   passes, one window a drafted pass, each window's kept count its pass's
//!   kept rows less one and its kept ids the run's tokens after its position;
//!   `commit(k)` is `k` steps for every k, a
//!   scripted draft keeping each exactly through the same verify and commit,
//!   at the deep prompt its rejected row completing a pool leaving the
//!   pooled planes the plain run's; and the lane word planted on a lane no
//!   call wrote makes the next drafted pass raise the delta stamp fault, by
//!   name, and poisons the model. Last, prompt calls that continue the held
//!   sequence: the draft joins at the call's start and drafts with the plain
//!   run's ids, and restarted beside the held sequence it skips the next
//!   call by name, proposing nothing, the ids still the plain run's.
//! - (p) a partial accept's records: the commit of a verify that kept fewer
//!   rows than it ran cuts the draft's records to the kept rows — after a
//!   drafted run the draft store's count stands at the model's position, a
//!   snapshot there holds exactly the accepted rows, a store walked past the
//!   position (an anchor row) makes the snapshot refused by name, and after
//!   a scripted window that keeps 2 of its 4 rows the pass arena's rejected
//!   row is refused by name for a walk that reads it (the uncut record
//!   would let the walk run).
//! - (x) a resume of a state whose draft side the load does not take (a
//!   state of a load without the draft) is refused by name with the model
//!   untouched: its stores, position and draft store unchanged.
//! - (y) two drafted sequences resident over a plan of two
//!   (`PlanInputs::plan_mtp_with_slots`, `Body38::open_placed_mtp_slots`),
//!   in the server's order (`worker.rs`'s `start` → `genloop`'s `prompt`):
//!   each request's prompt call and its first plain step run together
//!   before any other slot is selected, then the drafted passes interleave
//!   with a select between every pass, the app-side draft state parked per
//!   slot around each select exactly as a seat will (`MtpDraft::park`
//!   before a select away, `unpark` after a select back): each slot's ids,
//!   draft proposals and acceptances and draft store equal its window's
//!   single-sequence reference, run on the load's own sequence before any
//!   slot exists — references the slots' exchange cannot have shaped. The
//!   windows diverge from position 0 on (the deep prompt read past the
//!   set's own whole length), so no per-sequence state the two share by
//!   content can pass for one the slots exchanged.
//!
//! (l) and (h) hold at least one row each off a flip: a run that excuses
//! every row as a flip fails.

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
        Body38, MTP_GRAPH_ROWS, MTP_ROWS, MTP_STORE_ROWS, Mtp38, MtpDraft, MtpFeed, MtpHead,
        MtpHidden, MtpMode, MtpNode, MtpTaps, Prompt38, Qwen38Model, Store38Host, TargetRows,
    };
    use bloomery_gpu::fault::FaultSite;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::flip::{self, Flip};
    use bloomery_gpu_gates::rounding::{U, gamma, q8_32_rel};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, data_dir, verdict};
    use gguf::Split;
    use gguf::quant::half_to_f32;
    use model::arch::models::{Borrows, HeadRows, MtpSource};
    use model::arch::qwen35moe::head_list::{HeadWhy, SHIPPED, head_rows_of};
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, MtpPlan, PlanInputs, machine, vocab_sha256,
    };
    use model::fileio::hex;
    use model::placement::PlanLevers;
    use model::placement::workstation::RTX_3090;
    use refset::arch::qwen4exp::IK;
    use refset::arch::qwen4exp::MODEL;
    use refset::arch::qwen4exp::mtp::{DRAFT, MTP, MTP_SET};
    use refset::ik::Layout;
    use refset::mtpref::{Graph, MtpSet};
    use runtime::hc_gated::{Geometry, LO_BAND, MIXED_BAND, MixWeights, mix_ref};
    use runtime::{Advance, Committed, Draft, PassSink, TapNeed, Target, Want};

    /// Cache rows: the e2e gate's.
    const CTX: u64 = 3072;
    /// The head's rows: every sixth id, 0 to 245,754.
    const LIST_ROWS: u32 = 40_960;
    /// The shipped list's rows ([`SHIPPED`]'s first line).
    const SHIPPED_ROWS: usize = 65_536;
    /// The draft layer's index in the draft file: ik names its nodes by it.
    const LAYER: usize = 48;
    /// Streams of a hidden row, and the values of one.
    const STREAMS: usize = 4;
    const HIDDEN: usize = 2560;
    const WIDE: usize = STREAMS * HIDDEN;
    /// The hyper-connection mix's rank.
    const RANK: usize = 320;
    /// ggml's IMROPE position streams per row.
    const POS_SECTIONS: usize = 4;
    /// The router's experts and routed picks, and an expert's width.
    const N_EXPERT: usize = 512;
    const N_USED: usize = 10;
    const FF: usize = 640;

    /// PIN(2026-09-30): the streams `eh_proj` writes against ik's on the same
    /// inputs. Both sides compute the two norms in f32 (a few roundings
    /// each); the projection is the one term: ik quantizes its 5,120-value
    /// input to 32-value q8 blocks, ours reads it in f32, and a q8
    /// activation moves a projection's output by at most
    /// `rounding::q8_32_rel` = 1.2858e-2 of its RMS (the largest crest).
    fn eh_band() -> f64 {
        q8_32_rel()
    }

    /// PIN(2026-09-30): the band of the head's logits against ik's that
    /// (h)'s tie cap reads: the error model of the e2e gate's `gemm_band`
    /// over the draft's projections in series, where ik's side reads q8
    /// activations and ours f32 — `eh_proj`, the attention's input and
    /// output projections, the routed experts' gate·up and down, the shared
    /// expert's gate·up and down, and the head's projection: eight terms of
    /// [`q8_32_rel`], √8 · 1.2858e-2 = 3.64e-2.
    fn logits_band() -> f64 {
        8f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-09-30): the router's input against ik's: `eh_proj` and the
    /// attention's input and output projections, √3 · 1.2858e-2 = 2.23e-2.
    fn router_band() -> f64 {
        3f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-10-01): a store walk's values against an eager walk's over
    /// the same rows, a position's row at a time: the store walk's
    /// projections read 32-value q8 activations where the eager walk reads
    /// the f32 rows — `eh_proj`, then the value's own projection, two
    /// roundings of at most `rounding::q8_32_rel` each in series, which add
    /// (the triangle inequality; a quadrature sum assumes the two independent,
    /// and the clean run's worst position already sits past √2 of it); the
    /// attention site's mix reaches that input through σ, whose slope is at
    /// most ¼, and adds under 4 % of it (the e2e gate's `gemm_band` rule);
    /// each side's f16 store one rounding of 2^−11. 2.772e-2.
    fn value_band() -> f64 {
        2.0 * q8_32_rel() * 1.04 + 2.0 * 2f64.powi(-11)
    }

    /// PIN(2026-10-01): a store walk's keys against an eager walk's: the
    /// value's two projections in series, then each head's RMS norm, which
    /// moves a unit vector by at most twice its input's relative distance,
    /// and the rope, a rotation; each side's f16 store one rounding. 5.446e-2.
    fn key_band() -> f64 {
        2.0 * 2.0 * q8_32_rel() * 1.04 + 2.0 * 2f64.powi(-11)
    }

    /// Rows a host store walk of (s) and (w) takes: a width no caller walks,
    /// so the call's warmup equals it only when a row's stored bits are a
    /// function of that row's inputs alone, not of the walk it lands in.
    const HOST_STORE_RUN: usize = 3;

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

    /// A Q8_0 matrix's rows on the host, `k` values a row.
    struct HostQ8<'a> {
        bytes: &'a [u8],
        k: usize,
        rows: usize,
    }

    impl<'a> HostQ8<'a> {
        fn open(split: &'a Split, name: &str) -> Result<HostQ8<'a>, GateError> {
            let (sh, t) = split
                .find(name)
                .ok_or_else(|| format!("no tensor {name}"))?;
            if t.ty != gguf::quant::GgmlType::Q8_0 || !(2..=3).contains(&t.dims.len()) {
                return Err(format!(
                    "{name} is {:?} {:?}, not a Q8_0 matrix or stack",
                    t.ty, t.dims
                )
                .into());
            }
            let g = split
                .shard(sh)
                .ok_or_else(|| format!("{name}: shard {sh}"))?;
            let bytes = g.data(t)?;
            let k = usize::try_from(t.dims[0])?;
            let rows = t.dims[1..]
                .iter()
                .try_fold(1usize, |a, &d| usize::try_from(d).map(|d| a * d))?;
            Ok(HostQ8 { bytes, k, rows })
        }

        /// Row `r`'s 34-byte blocks.
        fn blocks(&self, r: usize) -> &[[u8; 34]] {
            let nb = self.k / 32;
            self.bytes[r * nb * 34..(r + 1) * nb * 34]
                .as_chunks::<34>()
                .0
        }

        /// Row `r` dequantized: each value `q·d`, exact in f32.
        fn row(&self, r: usize) -> Vec<f32> {
            let mut v = Vec::with_capacity(self.k);
            for blk in self.blocks(r) {
                let d = gguf::quant::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                v.extend(blk[2..].iter().map(|&q| f32::from(q as i8) * d));
            }
            v
        }

        /// Every row dequantized, row after row.
        fn dequant(&self) -> Vec<f32> {
            (0..self.rows).flat_map(|r| self.row(r)).collect()
        }

        /// Row `r` against `x` in f64 over the exact weight values: `Σ w·x`
        /// and `Σ |w·x|`.
        fn dot_abs(&self, r: usize, x: &[f64]) -> (f64, f64) {
            let (mut acc, mut abs) = (0.0f64, 0.0f64);
            for (b, blk) in self.blocks(r).iter().enumerate() {
                let d = f64::from(gguf::quant::half_to_f32(u16::from_le_bytes([
                    blk[0], blk[1],
                ])));
                for (i, &q) in blk[2..].iter().enumerate() {
                    let t = d * f64::from(q as i8) * x[b * 32 + i];
                    acc += t;
                    abs += t.abs();
                }
            }
            (acc, abs)
        }

        /// Rows `rows` against `x` ([`HostQ8::dot_abs`]), split over the
        /// host's cores.
        fn apply_abs(&self, x: &[f64], rows: std::ops::Range<usize>) -> (Vec<f64>, Vec<f64>) {
            let n = rows.len();
            let lanes = std::thread::available_parallelism().map_or(1, |p| p.get());
            let chunk = n.div_ceil(lanes).max(1);
            let mut out = vec![(0.0f64, 0.0f64); n];
            std::thread::scope(|s| {
                for (c, part) in out.chunks_mut(chunk).enumerate() {
                    let first = rows.start + c * chunk;
                    s.spawn(move || {
                        for (i, o) in part.iter_mut().enumerate() {
                            *o = self.dot_abs(first + i, x);
                        }
                    });
                }
            });
            out.into_iter().unzip()
        }
    }

    /// PIN(2026-09-30): the rounding path of one output of the q8f32 Q8_0
    /// dot over `k` values (`q8_0_lane_partial_1col`, which the gemv, the
    /// token-major mcol gemv and the selecting gemv all run): a lane's `k/32`
    /// fused multiply-adds — its `k/128` words, four values a word, `q·d`
    /// exact in f32 — then the butterfly's five adds, so `|ŷ − Σ w·x| ≤
    /// γ(k/32 + 5)·Σ|w·x|` (Higham's bound for a sum of that depth).
    /// Measured on the set's 127 rows: the worst distance at 0.009 of its
    /// row's bound (the shared expert's down, 1.405e-7).
    fn dot_depth(k: usize) -> usize {
        k / 32 + 5
    }

    /// PIN(2026-09-30): the input pack's values against the host's, relative
    /// (`mtp_input`): the hidden row's sum of squares over its 10,240 values
    /// is 40 fused multiply-adds a thread, the butterfly's 5 and the 7 warp
    /// sums, every term positive: γ(52) of the sum (the embedding's 2,560
    /// values take γ(22), inside it); the root halves that share, the
    /// division by the count, `+ eps`, the root and the reciprocal add 4u,
    /// and `(x·r)·γ` two more: ½γ(52) + 6u. Measured on the set's 127
    /// rows: `eh_proj` over the pack at worst 2.240e-7, 0.001 of its bound.
    fn pack_rel() -> f64 {
        0.5 * gamma(52) + 6.0 * U
    }

    /// One hyper-connection site's weights on the host, dequantized, in
    /// `runtime::hc_gated::MixWeights`' layouts.
    struct HostSite {
        gamma: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Option<Vec<f32>>,
    }

    impl HostSite {
        /// The site `sub` of the draft layer (`attn`, `ffn`), or the head's
        /// (`nextn.hc_head`, no inject).
        fn open(draft: &Split, stem: &str, inject: bool) -> Result<HostSite, GateError> {
            let b = |s: &str| format!("blk.{LAYER}.{stem}_{s}.weight");
            let q = |s: &str| HostQ8::open(draft, &b(s)).map(|m| m.dequant());
            Ok(HostSite {
                gamma: bloomery_gpu_gates::split_f32(draft, &b("norm"), WIDE)?,
                down: q("down")?,
                up: q("up")?,
                inject: if inject { Some(q("inject")?) } else { None },
            })
        }

        fn weights(&self) -> MixWeights<'_> {
            MixWeights {
                gamma: &self.gamma,
                down: &self.down,
                up: &self.up,
                inject: self.inject.as_deref(),
            }
        }
    }

    /// The draft layer's weights the node-local pins read, the target's
    /// embedding and head, and the model's norm epsilon.
    struct Host<'a> {
        embd: HostQ8<'a>,
        enorm: Vec<f32>,
        hnorm: Vec<f32>,
        eh: HostQ8<'a>,
        out: HostQ8<'a>,
        down_e: HostQ8<'a>,
        down_s: HostQ8<'a>,
        head: HostQ8<'a>,
        attn: HostSite,
        ffn: HostSite,
        head_site: HostSite,
        geo: Geometry,
        eps: f32,
    }

    impl<'a> Host<'a> {
        fn open(file: &'a Split, draft: &'a Split, eps: f32) -> Result<Host<'a>, GateError> {
            let o = |stem: &str| HostQ8::open(draft, &format!("blk.{LAYER}.{stem}.weight"));
            let f = |stem: &str, n: usize| {
                bloomery_gpu_gates::split_f32(draft, &format!("blk.{LAYER}.{stem}.weight"), n)
            };
            Ok(Host {
                embd: HostQ8::open(file, "token_embd.weight")?,
                enorm: f("nextn.enorm", HIDDEN)?,
                hnorm: f("nextn.hnorm", WIDE)?,
                eh: o("nextn.eh_proj")?,
                out: o("attn_output")?,
                down_e: o("ffn_down_exps")?,
                down_s: o("ffn_down_shexp")?,
                head: HostQ8::open(file, "output.weight")?,
                attn: HostSite::open(draft, "hc_attn", true)?,
                ffn: HostSite::open(draft, "hc_ffn", true)?,
                head_site: HostSite::open(draft, "nextn.hc_head", false)?,
                geo: Geometry::new(STREAMS as u32, RANK as u32, HIDDEN as u32)
                    .map_err(|e| format!("the draft's hyper-connection shape: {e}"))?,
                eps,
            })
        }
    }

    /// PIN(2026-09-30): a hyper-connection mix against `mix_ref` on its
    /// input as read back, `max|ours − ref| / max|ref|`: the hyper-connection
    /// gate's `MIXED_BAND`, whose derivation takes the up's gain on random
    /// inputs. Measured on this set's 127 rows: AttnIn at worst 0.023 of it,
    /// FfnIn 0.021, HeadIn 0.007.
    fn mix_band() -> f64 {
        f64::from(MIXED_BAND)
    }

    /// A node's index in the taps ([`MtpNode::ALL`]'s order).
    fn node_at(n: MtpNode) -> usize {
        MtpNode::ALL
            .iter()
            .position(|&m| m == n)
            .expect("ALL lists every node")
    }

    /// One node-local pin's record over the replay: each row's distance
    /// from the host's simulation of our rule on the node's inputs as the
    /// walk read them back, against its bound.
    struct Pin {
        what: &'static str,
        rule: &'static str,
        rows: usize,
        past: usize,
        worst: f64,
        /// The largest distance over its row's bound.
        ratio: f64,
        first_past: Option<String>,
    }

    impl Pin {
        fn new(what: &'static str, rule: &'static str) -> Pin {
            Pin {
                what,
                rule,
                rows: 0,
                past: 0,
                worst: 0.0,
                ratio: 0.0,
                first_past: None,
            }
        }

        /// A row at distance `d` against its bound `band` (0: bit for bit).
        fn hold(&mut self, d: f64, band: f64, label: &dyn Fn() -> String) {
            self.rows += 1;
            self.worst = self.worst.max(d);
            let r = if band > 0.0 {
                d / band
            } else if d == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            self.ratio = self.ratio.max(if r.is_nan() { f64::INFINITY } else { r });
            // A NaN distance is past every bound.
            if d.is_nan() || d > band {
                self.past += 1;
                if self.first_past.is_none() {
                    self.first_past = Some(format!("{} ({d:.3e}, bound {band:.3e})", label()));
                }
            }
        }

        fn ok(&self) -> bool {
            self.past == 0 && self.rows > 0
        }

        fn line(&self) -> String {
            format!(
                "(n) {} = {}: {} rows, worst {:.3e}, worst over its bound {:.3}, {} past{} {}",
                self.what,
                self.rule,
                self.rows,
                self.worst,
                self.ratio,
                self.past,
                self.first_past
                    .as_ref()
                    .map_or(String::new(), |f| format!(", first {f}")),
                verdict(self.ok())
            )
        }
    }

    /// The node-local pins, in the walk's order.
    struct Pins {
        eh: Pin,
        attn_in: Pin,
        attn_out: Pin,
        attn_combined: Pin,
        ffn_in: Pin,
        routed: Pin,
        shared: Pin,
        ffn_out: Pin,
        l_out: Pin,
        head_in: Pin,
        logits: Pin,
    }

    impl Pins {
        fn new() -> Pins {
            Pins {
                eh: Pin::new(
                    "eh_proj",
                    "the input pack of the token's embedding row and the fed hidden row, \
                     then the projection",
                ),
                attn_in: Pin::new("AttnIn", "the attention site's mix of eh_proj's streams"),
                attn_out: Pin::new("AttnOut", "the output projection of AttnGated"),
                attn_combined: Pin::new(
                    "AttnCombined",
                    "eh_proj's streams plus the attention site's weights times AttnOut",
                ),
                ffn_in: Pin::new("FfnIn", "the feed-forward site's mix of AttnCombined"),
                routed: Pin::new(
                    "Routed",
                    "the routed slots' downs of their SwiGLU rows, weighted, in slot order",
                ),
                shared: Pin::new("Shared", "the shared expert's down of its SwiGLU row"),
                ffn_out: Pin::new("FfnOut", "Routed plus Shared times its gate, bit for bit"),
                l_out: Pin::new(
                    "l_out",
                    "AttnCombined plus the feed-forward site's weights times FfnOut",
                ),
                head_in: Pin::new("HeadIn", "the head site's mix of l_out"),
                logits: Pin::new("logits", "the full head's projection of HeadIn"),
            }
        }

        fn all(&self) -> [&Pin; 11] {
            [
                &self.eh,
                &self.attn_in,
                &self.attn_out,
                &self.attn_combined,
                &self.ffn_in,
                &self.routed,
                &self.shared,
                &self.ffn_out,
                &self.l_out,
                &self.head_in,
                &self.logits,
            ]
        }
    }

    /// `‖ours − y‖ / ‖y‖` and `‖Σ|w·x|‖ / ‖y‖`, the host's `y` and
    /// absolute sums in f64: a dot's bound is `γ(depth)` times the second.
    fn dot_pin(ours: &[f32], y: &[f64], abs: &[f64]) -> (f64, f64) {
        let den = y
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let num = ours
            .iter()
            .zip(y)
            .map(|(&o, &v)| (f64::from(o) - v).powi(2))
            .sum::<f64>()
            .sqrt();
        let a = abs.iter().map(|v| v * v).sum::<f64>().sqrt();
        (nan_inf(num / den), a / den)
    }

    /// A NaN distance as infinite, so no band passes it.
    fn nan_inf(d: f64) -> f64 {
        if d.is_nan() { f64::INFINITY } else { d }
    }

    /// `max|ours − ref| / max|ref|`: the mix bands' measure
    /// (`runtime::hc_gated::MIXED_BAND`).
    fn max_rel(ours: &[f32], r: &[f64]) -> f64 {
        let den = r
            .iter()
            .fold(0.0f64, |a, v| a.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let num = ours
            .iter()
            .zip(r)
            .fold(0.0f64, |a, (&o, &v)| a.max((f64::from(o) - v).abs()));
        nan_inf(num / den)
            + if ours.len() == r.len() {
                0.0
            } else {
                f64::INFINITY
            }
    }

    /// A combine against the host's (`res + wgt·y` per stream, `wgt` the
    /// site's in f64): `‖ours − host‖ / ‖host‖` and its bound over it.
    ///
    /// PIN(2026-09-30): the card's weight is within `LO_BAND` of the largest
    /// of the site's (`runtime::hc_gated::LO_BAND`'s derivation), so a value
    /// moves by at most that times `|y|`, and the fused multiply-add rounds
    /// once: per value `LO_BAND·max|wgt|·|y| + u·|host|`. Measured on the
    /// set's 127 rows: AttnCombined at worst 0.011 of its bound, l_out 0.018.
    fn combine_pin(ours: &[f32], res: &[f32], y: &[f32], wgt: &[f64]) -> (f64, f64) {
        let top = wgt.iter().fold(0.0f64, |a, w| a.max(w.abs()));
        let (mut num, mut den, mut bound) = (0.0f64, 0.0f64, 0.0f64);
        for (s, &w) in wgt.iter().enumerate() {
            let span = s * HIDDEN..(s + 1) * HIDDEN;
            for ((&o, &r), &yi) in ours[span.clone()].iter().zip(&res[span]).zip(y) {
                let h = f64::from(r) + w * f64::from(yi);
                num += (f64::from(o) - h).powi(2);
                den += h * h;
                bound += (f64::from(LO_BAND) * top * f64::from(yi).abs() + U * h.abs()).powi(2);
            }
        }
        let den = den.sqrt().max(f64::MIN_POSITIVE);
        (nan_inf(num.sqrt() / den), bound.sqrt() / den)
    }

    /// The node-local pins of row `t` of a walk of graph `g` at its row
    /// `at` ([`Pins`]): each node against the host's simulation of our rule
    /// on the node's inputs as the walk read them back — the eh_proj pin's
    /// inputs are the fed token and hidden row themselves.
    #[allow(
        clippy::too_many_arguments,
        reason = "one row of one walk: its taps, streams, logits and the host's weights"
    )]
    fn pin_row(
        h: &Host<'_>,
        g: &IkGraph,
        at: usize,
        taps: &MtpTaps,
        t: usize,
        l_out: &[f32],
        p: &mut Pins,
    ) {
        let label = || format!("{} row {at}", g.label());
        let node = |n: MtpNode| {
            let wd = n.width();
            &taps.nodes[node_at(n)].1[t * wd..(t + 1) * wd]
        };
        let f64s = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();

        // eh_proj: the pack `[e·r_e·enorm | h_s·r_h·hnorm_s]` a stream, then
        // the projection of its 5,120 values.
        let e = h.embd.row(g.tokens[at] as usize);
        let hid = &g.states[at * WIDE..(at + 1) * WIDE];
        let inv_rms = |v: &[f32]| {
            1.0 / (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len() as f64
                + f64::from(h.eps))
            .sqrt()
        };
        let (re, rh) = (inv_rms(&e), inv_rms(hid));
        let (mut y, mut abs) = (Vec::with_capacity(WIDE), Vec::with_capacity(WIDE));
        for s in 0..STREAMS {
            let mut pack = Vec::with_capacity(2 * HIDDEN);
            pack.extend((0..HIDDEN).map(|i| f64::from(e[i]) * re * f64::from(h.enorm[i])));
            pack.extend((0..HIDDEN).map(|i| {
                let j = s * HIDDEN + i;
                f64::from(hid[j]) * rh * f64::from(h.hnorm[j])
            }));
            let (ys, a) = h.eh.apply_abs(&pack, 0..HIDDEN);
            y.extend(ys);
            abs.extend(a);
        }
        let eh = &taps.eh[t * WIDE..(t + 1) * WIDE];
        let (d, a) = dot_pin(eh, &y, &abs);
        // The pack's own error reaches the output through |W|: at most
        // pack_rel of Σ|w·x|, beside the projection's γ.
        let g = gamma(dot_depth(2 * HIDDEN));
        p.eh.hold(d, (g + pack_rel() * (1.0 + g)) * a, &label);

        // The attention site: its mix of eh_proj's streams, the output
        // projection, the combine.
        let attn = mix_ref(h.geo, h.attn.weights(), eh, h.eps);
        p.attn_in.hold(
            max_rel(node(MtpNode::AttnIn), &attn.mixed),
            mix_band(),
            &label,
        );
        let gated = f64s(node(MtpNode::AttnGated));
        let (y, abs) = h.out.apply_abs(&gated, 0..HIDDEN);
        let (d, a) = dot_pin(node(MtpNode::AttnOut), &y, &abs);
        p.attn_out
            .hold(d, gamma(dot_depth(gated.len())) * a, &label);
        let wgt = attn.wgt.as_deref().unwrap_or(&[]);
        let (d, b) = combine_pin(node(MtpNode::AttnCombined), eh, node(MtpNode::AttnOut), wgt);
        p.attn_combined.hold(d, b, &label);

        // The feed-forward site: its mix, the routed slots, the shared
        // expert, their gated sum.
        let ffn = mix_ref(h.geo, h.ffn.weights(), node(MtpNode::AttnCombined), h.eps);
        p.ffn_in.hold(
            max_rel(node(MtpNode::FfnIn), &ffn.mixed),
            mix_band(),
            &label,
        );
        let slots = taps.slots_row;
        // PIN(2026-09-30): the routed sum's bound per value — each slot's
        // down (γ(640/32 + 5) of Σ|w·x|, `dot_depth`) through its weight,
        // and `q38_card_acc`'s ten fused multiply-adds in slot order, γ(10)
        // of Σ|w·down|. Measured on the set's 127 rows: at worst 0.006 of it.
        let (mut acc, mut bound) = (vec![0.0f64; HIDDEN], vec![0.0f64; HIDDEN]);
        for j in 0..N_USED {
            let e = taps.ids[t * slots + j] as usize;
            let w = f64::from(taps.weights[t * slots + j]);
            let x = f64s(&taps.routed_h[(t * N_USED + j) * FF..(t * N_USED + j + 1) * FF]);
            let (dj, aj) = h.down_e.apply_abs(&x, e * HIDDEN..(e + 1) * HIDDEN);
            for i in 0..HIDDEN {
                acc[i] += w * dj[i];
                // The down's own bound through the weight, and the ten
                // slots' fused multiply-adds (depth 10) over |w·down|.
                bound[i] += w.abs() * (gamma(dot_depth(FF)) * aj[i] + gamma(N_USED) * dj[i].abs());
            }
        }
        let routed = node(MtpNode::Routed);
        let den = acc
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let num = routed
            .iter()
            .zip(&acc)
            .map(|(&o, &v)| (f64::from(o) - v).powi(2))
            .sum::<f64>()
            .sqrt();
        let b = bound.iter().map(|v| v * v).sum::<f64>().sqrt();
        p.routed.hold(nan_inf(num / den), b / den, &label);
        let x = f64s(&taps.shared_h[t * FF..(t + 1) * FF]);
        let (y, abs) = h.down_s.apply_abs(&x, 0..HIDDEN);
        let (d, a) = dot_pin(node(MtpNode::Shared), &y, &abs);
        p.shared.hold(d, gamma(dot_depth(FF)) * a, &label);
        // `q38_shared_add`: the product rounded, then the sum rounded.
        let gate = taps.weights[t * slots + N_USED];
        let differ = routed
            .iter()
            .zip(node(MtpNode::Shared))
            .zip(node(MtpNode::FfnOut))
            .filter(|&((r, s), o)| (*r + *s * gate).to_bits() != o.to_bits())
            .count();
        p.ffn_out.hold(differ as f64, 0.0, &label);

        // The layer's output: the head site's combine.
        let wgt = ffn.wgt.as_deref().unwrap_or(&[]);
        let (d, b) = combine_pin(
            l_out,
            node(MtpNode::AttnCombined),
            node(MtpNode::FfnOut),
            wgt,
        );
        p.l_out.hold(d, b, &label);
        let head = mix_ref(h.geo, h.head_site.weights(), l_out, h.eps);
        p.head_in.hold(
            max_rel(node(MtpNode::HeadIn), &head.mixed),
            mix_band(),
            &label,
        );
    }

    /// The logits pin of one head row: our logits `ours` against the full
    /// head's projection of our `HeadIn` row `x`.
    fn pin_logits(h: &Host<'_>, x: &[f32], ours: &[f32], p: &mut Pin, label: &dyn Fn() -> String) {
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let (y, abs) = h.head.apply_abs(&x, 0..h.head.rows);
        let (d, a) = dot_pin(ours, &y, &abs);
        p.hold(d, gamma(dot_depth(HIDDEN)) * a, label);
    }

    /// The worst of each clause over the replay, and the counts.
    struct Replay {
        eh_worst: f64,
        eh_fail: usize,
        /// The layer's output's and the logits' distance from ik's: printed
        /// inside (l) and (h), not held.
        l_out_worst: f64,
        logits_worst: f64,
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
        /// The held head rows' smallest top-2 margin of ik's logits, over
        /// its tie cap, and the row: how near (h) came to a tie.
        margin_min: Option<(f64, f64, String)>,
        pins: Pins,
    }

    impl Replay {
        fn new() -> Replay {
            Replay {
                eh_worst: 0.0,
                eh_fail: 0,
                l_out_worst: 0.0,
                logits_worst: 0.0,
                rows: 0,
                heads: 0,
                flips: 0,
                flips_bad: 0,
                argmax_same: 0,
                l_out_held: 0,
                heads_held: 0,
                argmax_tie: 0,
                argmax_bad: 0,
                kernel_bad: 0,
                margin_min: None,
                pins: Pins::new(),
            }
        }
    }

    /// Replay graph `g` in runs of up to [`MTP_ROWS`] rows, the full head,
    /// the taps armed, into `r`.
    fn replay(
        m: &mut Qwen38Model,
        g: &IkGraph,
        host: &Host<'_>,
        r: &mut Replay,
    ) -> Result<(), GateError> {
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
                let l_out = &w.l_out[t * WIDE..(t + 1) * WIDE];
                pin_row(host, g, at, &taps, t, l_out, &mut r.pins);
                if !flipped[t] {
                    r.l_out_held += 1;
                    r.l_out_worst = r.l_out_worst.max(rel(l_out, &g.l_out[span]));
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
                let head_in = &taps.nodes[node_at(MtpNode::HeadIn)].1[c * HIDDEN..(c + 1) * HIDDEN];
                pin_logits(host, head_in, &ours, &mut r.pins.logits, &|| {
                    format!("{} row {o}", g.label())
                });
                let ik = &g.logits[k * vocab..(k + 1) * vocab];
                let (top, _) = top2(&ours);
                let (ik_top, ik_2) = top2(ik);
                let margin = f64::from(ik[ik_top]) - f64::from(ik[ik_2]);
                let clears = margin > margin_cap(logits_band(), ik);
                if flipped[c] {
                    println!(
                        "(h) {} row {o}: argmax {top} ik {ik_top} on a flip, not held",
                        g.label()
                    );
                    continue;
                }
                r.heads_held += 1;
                r.logits_worst = r.logits_worst.max(rel(&ours, ik));
                let cap = margin_cap(logits_band(), ik);
                if r.margin_min
                    .as_ref()
                    .is_none_or(|(_, q, _)| margin / cap < *q)
                {
                    r.margin_min = Some((margin, margin / cap, format!("{} row {o}", g.label())));
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
    /// run, as one walk of all of them (the m-column entry) and as one walk
    /// of its last row (the one-column entry).
    fn list_head(m: &mut Qwen38Model, g: &IkGraph, ids: &[u32]) -> Result<bool, GateError> {
        let n = g.tokens.len();
        let mut all = true;
        for rows in [MTP_ROWS.min(n), 1] {
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
                            "(r) row {c}: p {} against the host's {want:.9} over {} rows, {e:.3e} \
                             past {:.3e} FAIL",
                            w.draft.p[c],
                            v.len(),
                            p_bound(v.len())
                        );
                    }
                }
            }
            ok &= differ == 0 && p_bad == 0;
            println!(
                "(r) a walk of {rows} rows of {} ({}): list logits = the full head's rows of their \
                 ids, {differ} differ; tokens mapped; p within its bound (worst {p_worst:.2} of it) \
                 {}",
                g.label(),
                list.rows,
                verdict(ok)
            );
            all &= ok;
        }
        Ok(all)
    }

    /// (t): the target's streams after the prompt against ik's warmup hidden
    /// rows, and a draft walk fed them in place against one fed them from the
    /// host.
    fn pairing(m: &mut Qwen38Model, prompt: &[u32], warm: &IkGraph) -> Result<bool, GateError> {
        let next = m.prompt38(prompt, Prompt38::Pass)?;
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
        // The walk starts one past the arena's first position — each of its
        // rows at q reads the hidden row at q − 1, the arena's rows 0..rows
        // (the prompt's last unit's rows, `first` on) — and runs one row past
        // the prompt's end, the prompt's own next id its last token.
        let mut tokens = prompt[first + 1..].to_vec();
        tokens.push(next);
        let pos0 = u32::try_from(first + 1)?;
        let host = walk(
            m,
            MtpFeed::Rows {
                tokens: &tokens,
                pos0,
                hidden: MtpHidden::Host(&ours),
            },
            MtpHead::Rows,
            MtpMode::Eager,
        );
        let place = walk(
            m,
            MtpFeed::Rows {
                tokens: &tokens,
                pos0,
                hidden: MtpHidden::Target {
                    walk: TargetRows::Pass,
                    first: 0,
                },
            },
            MtpHead::Rows,
            MtpMode::Eager,
        );
        let same = match (&host, &place) {
            (Ok(h), Ok(p)) => same_bits(h, p),
            _ => false,
        };
        println!(
            "(t) {rows} rows from {pos0} fed the target's streams in place = from the host, bit \
             for bit {}{}{}",
            verdict(same),
            host.err().map_or(String::new(), |e| format!(" ({e})")),
            place.err().map_or(String::new(), |e| format!(" ({e})")),
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
        // A window's chain — a refresh of two rows, then two own walks, one
        // readback — against the same walks one by one: each walk's last id
        // and probability, bit for bit.
        let chain = m.mtp_chain(rows_of(2), 2, MtpHead::Rows, MtpMode::Graph)?;
        let one = [
            walk(m, rows_of(2), MtpHead::Rows, MtpMode::Eager)?,
            walk(m, own(g.pos[0] + 2), MtpHead::Rows, MtpMode::Eager)?,
            walk(m, own(g.pos[0] + 3), MtpHead::Rows, MtpMode::Eager)?,
        ];
        let want: Vec<(u32, u32)> = one
            .iter()
            .map(|w| {
                let last = w.draft.tokens.len() - 1;
                (w.draft.tokens[last], w.draft.p[last].to_bits())
            })
            .collect();
        let got: Vec<(u32, u32)> = chain
            .tokens
            .iter()
            .zip(&chain.p)
            .map(|(&t, p)| (t, p.to_bits()))
            .collect();
        let chain_walks = got == want;
        ok &= chain_walks;
        println!(
            "(g) a chain of two rows and two own walks = its walks one by one: ids {:?} {}",
            chain.tokens,
            verdict(chain_walks)
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
        let over = MTP_STORE_ROWS + 1;
        let tokens: Vec<u32> = g.tokens.iter().copied().cycle().take(over).collect();
        let states: Vec<f32> = g.states.iter().copied().cycle().take(over * WIDE).collect();
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
        ok &= refused(
            "a store walk past its rows",
            m.mtp_walk(feed(over, 0), h, MtpMode::Store),
            &format!("a walk of {over} rows"),
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
        // A store walk writes the store alone: nothing of it is read back,
        // and it leaves no own row for the next walk.
        let store = MtpMode::Store;
        ok &= refused(
            "a store walk read back",
            m.mtp_draft(feed(1, 0), h, store),
            "a walk with a head to read back",
        );
        ok &= refused(
            "a chain of store walks",
            m.mtp_chain(feed(1, 0), 1, h, store),
            "a chain's walks with a head",
        );
        m.mtp_walk(feed(1, 0), h, store)?;
        ok &= refused(
            "the draft's own row after a store walk",
            m.mtp_draft(MtpFeed::Own { pos0: 1 }, h, e),
            "a walk before the draft's own row",
        );
        ok &= refused(
            "the streams of a store walk",
            m.mtp_l_out(),
            "a walk to read",
        );
        Ok(ok && raised)
    }

    /// (f): a walk that reads the target's arenas by position: after the
    /// set's prompt by passes (the Pass arena holding its last unit's rows),
    /// with the draft's store warmed over the prompt and its next position,
    /// a walk that reads the position before the arena's first, one that
    /// reads past its last row, and one that starts its rows at the wrong
    /// arena row are each refused by name with the positions the arena
    /// holds, and the walk the arena does hold — its rows from the position
    /// after the arena's first, the prompt's own next id its last token —
    /// runs.
    fn arena_holds(m: &mut Qwen38Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        let next = m.prompt38(prompt, Prompt38::Pass)?;
        let n = u32::try_from(prompt.len())?;
        let rows = (prompt.len() - 1) % MTP_ROWS + 1;
        let first = n - rows as u32;
        // The draft's store must hold every walk's start: warm it over the
        // prompt's rows and the position after them (store walks, the prompt
        // call's warmup shape).
        let zeros = vec![0.0f32; MTP_STORE_ROWS * WIDE];
        m.mtp_walk(
            MtpFeed::Rows {
                tokens: &prompt[..MTP_STORE_ROWS.min(prompt.len())],
                pos0: 0,
                hidden: MtpHidden::Host(&zeros[..MTP_STORE_ROWS.min(prompt.len()) * WIDE]),
            },
            MtpHead::Rows,
            MtpMode::Store,
        )?;
        m.mtp_walk(
            MtpFeed::Rows {
                tokens: &prompt[..1],
                pos0: n,
                hidden: MtpHidden::Host(&zeros[..WIDE]),
            },
            MtpHead::Rows,
            MtpMode::Store,
        )?;
        fn feed(tokens: &[u32], pos0: u32, at: usize) -> MtpFeed<'_> {
            MtpFeed::Rows {
                tokens,
                pos0,
                hidden: MtpHidden::Target {
                    walk: TargetRows::Pass,
                    first: at,
                },
            }
        }
        let (h, e) = (MtpHead::Rows, MtpMode::Eager);
        let mut ok = refused(
            "the Pass arena's rows read as the position before their first",
            m.mtp_draft(feed(&prompt[..1], first, 0), h, e),
            &format!(
                "as positions {}..{first} for a walk from {first}: the arena holds positions \
                 {first}..{n}",
                first - 1
            ),
        );
        ok &= refused(
            "the Pass arena's rows read past their last",
            m.mtp_draft(feed(&prompt[..1], n + 1, 0), h, e),
            &format!(
                "as positions {n}..{} for a walk from {}: the arena holds positions {first}..{n}",
                n + 1,
                n + 1
            ),
        );
        ok &= refused(
            "the Pass arena's rows read as other positions",
            m.mtp_draft(feed(&prompt[..1], first + 1, 1), h, e),
            &format!(
                "rows 1..2 of the Pass arena as positions {first}..{} for a walk from {}",
                first + 1,
                first + 1
            ),
        );
        let mut tokens = prompt[first as usize + 1..].to_vec();
        tokens.push(next);
        let runs = m.mtp_draft(
            MtpFeed::Rows {
                tokens: &tokens,
                pos0: first + 1,
                hidden: MtpHidden::Target {
                    walk: TargetRows::Pass,
                    first: 0,
                },
            },
            h,
            e,
        );
        let runs_ok = runs.is_ok();
        println!(
            "(f) the walk the Pass arena holds, its {rows} rows from {}: {} {}",
            first + 1,
            match &runs {
                Ok(w) => format!("ran, {} tokens", w.tokens.len()),
                Err(err) => err.to_string(),
            },
            verdict(runs_ok)
        );
        Ok(ok && runs_ok)
    }

    /// (p): a partial accept's records and snapshot. The commit of a verify
    /// that kept fewer rows than it ran cuts the draft's records to the kept
    /// rows: after a drafted run the store's count stands at the model's
    /// position, a snapshot there holds exactly the accepted rows, and once
    /// the store walks past the position (an anchor row) the snapshot is
    /// refused by name; after a scripted window that keeps 2 of its 4 rows
    /// the pass arena's rejected row is a position the arena no longer
    /// holds — a walk that reads it is refused by name, where the uncut
    /// record would let it run.
    fn kept_state(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 6;
        const KEEP: usize = 2;
        let ctx = m.body("kept state")?.ctx() as u32;
        // A drafted run: the model stands after its last commit.
        let (mut m, kept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, prompt)?;
            let mut k = Kept::default();
            runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            (s.into_model(), k)
        };
        let at = m.pos() as usize;
        let held = m.body("kept state")?.mtp().map_or(0, |d| d.held());
        let cut = held == at;
        println!(
            "(p) after the drafted run's {} windows at {at}: the draft store holds {held} \
             positions {}",
            kept.rows.len(),
            verdict(cut)
        );
        let snap = m.seq_save();
        let snap_ok = matches!(&snap, Ok(s) if s.positions() as usize == at);
        println!(
            "(p) a snapshot at {at} holds exactly the accepted rows: {} {}",
            match &snap {
                Ok(s) => format!("{} positions, {} bytes", s.positions(), s.bytes()),
                Err(e) => e.to_string(),
            },
            verdict(snap_ok)
        );
        // The store walked past the position — an anchor row: the state
        // carries the rows below the position only.
        let zeros = vec![0.0f32; WIDE];
        m.mtp_walk(
            MtpFeed::Rows {
                tokens: &prompt[..1],
                pos0: u32::try_from(at)?,
                hidden: MtpHidden::Host(&zeros),
            },
            MtpHead::Rows,
            MtpMode::Store,
        )?;
        let mut ok = cut && snap_ok;
        ok &= refused(
            "a snapshot of a store past the position",
            m.seq_save(),
            &format!(
                "a state at {at} of a draft store that holds {} positions",
                at + 1
            ),
        );
        // A scripted window that keeps `KEEP` of its 4 rows: the arena's
        // rows `KEEP` on are the rejected tail the commit takes back. The
        // scripted drafter walks no draft, so the store is warmed over the
        // positions first (store walks of other ids).
        let (m, plain) = plain_run(m, prompt, Prompt38::Auto, KEEP + 2)?;
        let (mut m, w_ok) = scripted_window(m, &plain, prompt, KEEP, ctx, "the partial accept")?;
        let at = m.pos() as usize;
        let pos0 = at - KEEP;
        let zeros = vec![0.0f32; MTP_STORE_ROWS * WIDE];
        let vocab = u32::try_from(m.body("kept state")?.vocab())?;
        let ids: Vec<u32> = (0..at + 2)
            .map(|i| (prompt[i % prompt.len()] + 1) % vocab)
            .collect();
        for (i, run) in ids[..at].chunks(MTP_STORE_ROWS).enumerate() {
            m.mtp_walk(
                MtpFeed::Rows {
                    tokens: run,
                    pos0: u32::try_from(i * MTP_STORE_ROWS)?,
                    hidden: MtpHidden::Host(&zeros[..run.len() * WIDE]),
                },
                MtpHead::Rows,
                MtpMode::Store,
            )?;
        }
        m.mtp_walk(
            MtpFeed::Rows {
                tokens: &ids[at..at + 1],
                pos0: u32::try_from(at)?,
                hidden: MtpHidden::Host(&zeros[..WIDE]),
            },
            MtpHead::Rows,
            MtpMode::Store,
        )?;
        ok &= w_ok;
        ok &= refused(
            "the pass arena's rejected row",
            m.mtp_draft(
                MtpFeed::Rows {
                    tokens: &ids[..1],
                    pos0: u32::try_from(at + 1)?,
                    hidden: MtpHidden::Target {
                        walk: TargetRows::Pass,
                        first: KEEP,
                    },
                },
                MtpHead::Rows,
                MtpMode::Eager,
            ),
            &format!(
                "rows {KEEP}..{} of the Pass arena as positions {at}..{} for a walk from {}: the \
                 arena holds positions {pos0}..{at}",
                KEEP + 1,
                at + 1,
                at + 1
            ),
        );
        Ok((m, ok))
    }

    /// (x): a resume refused before any copy. A state of a load without the
    /// draft (its own short prompt run) put back on this load is refused by
    /// name — the store layouts differ — with the model untouched: its
    /// stores still the empty model's bit for bit, its position 0, the
    /// draft's store empty.
    fn resume_refused(
        m: Qwen38Model,
        inputs: &PlanInputs,
        levers: &bloomery_levers::Levers,
        ub: usize,
        prompt: &[u32],
    ) -> Result<(Qwen38Model, bool), GateError> {
        let machine = machine(RTX_3090, inputs.spec.layers.len(), u64::try_from(ub)?);
        let plan_levers = PlanLevers::from_levers(levers)?;
        let plain_plan = inputs.plan(&machine, CTX, &plan_levers)?;
        let mut b = Body38::open_placed(
            Split::open(MODEL)?,
            &plain_plan,
            inputs,
            0,
            levers.host(),
            ub,
        )?;
        // Other ids over the same length: the state's rows differ from
        // whatever this model's stores hold, so a copy that should not
        // happen shows.
        let vocab = u32::try_from(b.body("resume refused")?.vocab())?;
        let ids: Vec<u32> = prompt.iter().map(|&t| (t + 1) % vocab).collect();
        b.prompt38(&ids, Prompt38::Pass)?;
        let without = b.seq_save()?;
        drop(b);
        let mut m = m;
        m.reset()?;
        let read = |m: &mut Qwen38Model| -> Result<(Vec<Store38Host>, Vec<f32>), GateError> {
            let (gpu, _, body) = m.body_parts("resume refused")?;
            Ok(body.stores_host(gpu)?)
        };
        let before = read(&mut m)?;
        let r = m.seq_resume(&without);
        let after = read(&mut m)?;
        let pos = m.pos();
        let held = m.body("resume refused")?.mtp().map_or(0, |d| d.held());
        let stores_same = before.0.len() == after.0.len()
            && before.0.iter().zip(&after.0).all(|(a, b)| a.same_bits(b));
        let ring_same = before.1.len() == after.1.len()
            && before
                .1
                .iter()
                .zip(&after.1)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let named = matches!(&r, Err(e) if e.to_string().contains(
            "a state without the draft's side put back on a load with the draft"
        ));
        let untouched = stores_same && ring_same && pos == 0 && held == 0;
        let ok = named && untouched;
        println!(
            "(x) a state without the draft's side put back on a load with the draft: {} — the \
             model untouched: stores {}, position {pos}, draft store {held} {}",
            match &r {
                Ok(()) => "resumed".to_string(),
                Err(e) => e.to_string(),
            },
            if stores_same && ring_same {
                "same".to_string()
            } else {
                "changed".to_string()
            },
            verdict(ok)
        );
        Ok((m, ok))
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

    /// A run's live stores as the body reads them back, and its PLE ring.
    type Stores = (Vec<Store38Host>, Vec<f32>);

    /// One plain run of `n` tokens from a reset, as [`plain_run`]'s, its
    /// stores read once at its end.
    fn plain_to(
        m: Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        n: usize,
    ) -> Result<(Qwen38Model, Vec<u32>, Stores), GateError> {
        let mut m = m;
        m.reset()?;
        let mut tokens = vec![m.prompt38(ids, path)?];
        for _ in 1..n {
            let t = m.step(&[tokens[tokens.len() - 1]])?;
            tokens.push(t);
        }
        let stores = {
            let (gpu, _, b) = m.body_parts("plain")?;
            b.stores_host(gpu)?
        };
        Ok((m, tokens, stores))
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

    /// The drafted session over the model from a reset — the plain run it
    /// is held against starts at position 0 — the draft opened, the verify
    /// widths captured.
    fn drafted_session(
        m: Qwen38Model,
        cfg: app::arch::qwen3moe::Q38Cfg,
        ctx: u32,
    ) -> Result<(app::Session<Body38>, app::arch::qwen3moe::Drafted38), GateError> {
        let mut m = m;
        m.reset()?;
        let mut s = app::Session::from_model(m, ctx);
        let draft = app::mtp::MtpDraft::open(s.model(), cfg.prompt, cfg.draft)?;
        let spec = s.with_draft::<app::mtp::MtpDraft<Body38>, 4>(draft, &mut Quiet)?;
        Ok((s, spec))
    }

    /// The windows the draft keeps ([`app::mtp::MtpDraft::keep_windows`],
    /// `BLOOMERY_MTP_WINDOWS`): `unkept` windows came back from the drafted
    /// run `off` (its ids and passes), which kept none, and must be 0; the
    /// same run keeping its windows reads back the same ids through the same
    /// passes, one window a drafted pass, each kept one row past the ids its
    /// window says the target kept, those ids the run's own tokens at the
    /// positions after the window's, each probability in (0, 1].
    fn kept_windows(
        m: Qwen38Model,
        prompt: &[u32],
        ctx: u32,
        off: (&[u32], &Kept),
        unkept: usize,
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 64;
        let (off_tokens, off_kept) = off;
        let (m, out, kept, windows) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            spec.draft_mut().keep_windows();
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
            let w = spec.draft_mut().take_windows();
            (s.into_model(), out, k, w)
        };
        let same = out.tokens == off_tokens
            && kept.rows == off_kept.rows
            && kept.proposed == off_kept.proposed;
        let drafted: Vec<usize> = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(_, p)| **p)
            .map(|(k, _)| *k)
            .collect();
        let p0 = prompt.len();
        let mut bad = Vec::new();
        if drafted.len() != windows.len() {
            bad.push(format!(
                "{} windows of {} drafted passes",
                windows.len(),
                drafted.len()
            ));
        }
        for (i, (&k, w)) in drafted.iter().zip(&windows).enumerate() {
            let at = w.pos as usize + 1 - p0;
            let emitted = out.tokens.get(at..at + w.accepted);
            let p_ok = w.p.iter().all(|&p| p > 0.0 && p <= 1.0);
            if w.accepted + 1 != k
                || w.ids.len() != w.p.len()
                || w.ids.is_empty()
                || !p_ok
                || emitted.is_some_and(|e| e != &w.ids[..w.accepted])
            {
                bad.push(format!(
                    "window {i}: pass kept {k}, window {:?} p {:?} accepted {}, emitted {emitted:?}",
                    w.ids, w.p, w.accepted
                ));
            }
        }
        let ok = unkept == 0 && same && bad.is_empty();
        println!(
            "(w) depth {}: unset the draft keeps {unkept} windows; kept, the same ids and passes \
             {} and {} windows of {} drafted passes, each its pass's kept rows and ids{} {}",
            prompt.len(),
            verdict(same),
            windows.len(),
            drafted.len(),
            if bad.is_empty() {
                String::new()
            } else {
                format!(" — {}", bad.join("; "))
            },
            verdict(ok)
        );
        Ok((m, ok))
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
    /// out, from a reset as the plain run's. The label names the prompt in
    /// the clause's line.
    fn scripted_window(
        m: Qwen38Model,
        plain: &Plain,
        ids: &[u32],
        keep: usize,
        ctx: u32,
        label: &str,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let mut m = m;
        m.reset()?;
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

    /// (s) and (w)'s first clause: the draft's walks over a prompt call are
    /// the walks the module doc of `app::arch::qwen3moe` names. From a reset,
    /// the prompt by passes with every unit's hidden rows read to the host,
    /// then the draft walked from the host as the prompt call walks it —
    /// position 0 with a zero hidden row, each unit's rows from its second
    /// position on (the last unit's last excepted) each beside the hidden row
    /// of the position before it — eagerly in runs of eight (every launch,
    /// the head's included), then over the same rows as store walks
    /// (`MtpMode::Store`) in runs of [`HOST_STORE_RUN`]: (s) each position's
    /// keys and values the store walks leave within [`key_band`] and
    /// [`value_band`] of the eager walks'. Then one anchor walk at the
    /// prompt's end; against the drafted session's prompt call by the same
    /// passes, whose warmup walks the store alone: the store's keys and values
    /// at every prompt position bit for bit the host's store walks', then the
    /// same anchor walk's token, streams and logits bit for bit — the anchor's
    /// attention reads every key the walks before it wrote. Between the two,
    /// eager walks of other ids over the same positions overwrite every
    /// stored row, so a position the prompt call does not write reads as
    /// another sequence's, not as the host run's.
    fn prompt_walks(
        m: Qwen38Model,
        prompt: &[u32],
        path: Prompt38,
        ctx: u32,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let n = prompt.len();
        let mut m = m;
        m.reset()?;
        let mut streams = vec![0.0f32; n * WIDE];
        let mut units: Vec<(usize, usize)> = Vec::new();
        {
            let mut sink = |mm: &mut Qwen38Model,
                            walk: TargetRows,
                            first: u32,
                            rows: usize|
             -> Result<(), GpuError> {
                let f = first as usize;
                let v = mm.target_streams(walk, rows)?;
                streams[f * WIDE..(f + rows) * WIDE].copy_from_slice(&v);
                units.push((f, rows));
                Ok(())
            };
            m.prompt38_with(prompt, path, Some(&mut sink))?;
        }
        let (h, e) = (MtpHead::Rows, MtpMode::Eager);
        let zeros = vec![0.0f32; WIDE];
        // Position 0 beside a zero row, then each unit's rows from its second
        // position on, each beside the hidden row of the position before it,
        // in walks of `mode` of `run` rows.
        let walk_host = |m: &mut Qwen38Model, mode: MtpMode, run: usize| -> Result<(), GateError> {
            m.mtp_walk(
                MtpFeed::Rows {
                    tokens: &prompt[..1],
                    pos0: 0,
                    hidden: MtpHidden::Host(&zeros),
                },
                h,
                mode,
            )?;
            for &(f, rows) in &units {
                let end = (f + rows + 1).min(n);
                for (i, r) in prompt[f + 1..end].chunks(run).enumerate() {
                    let q = f + 1 + i * run;
                    m.mtp_walk(
                        MtpFeed::Rows {
                            tokens: r,
                            pos0: u32::try_from(q)?,
                            hidden: MtpHidden::Host(
                                &streams[(q - 1) * WIDE..(q - 1 + r.len()) * WIDE],
                            ),
                        },
                        h,
                        mode,
                    )?;
                }
            }
            Ok(())
        };
        // Other ids over the same positions, from position 0, eagerly: every
        // row an earlier walk stored is overwritten.
        let vocab = u32::try_from(m.body("prompt_walks")?.vocab())?;
        let other: Vec<u32> = prompt.iter().map(|&t| (t + 1) % vocab).collect();
        let overwrite = |m: &mut Qwen38Model| -> Result<(), GateError> {
            m.mtp_walk(
                MtpFeed::Rows {
                    tokens: &other[..1],
                    pos0: 0,
                    hidden: MtpHidden::Host(&zeros),
                },
                h,
                e,
            )?;
            for (i, run) in other[1..].chunks(MTP_ROWS).enumerate() {
                let q = 1 + i * MTP_ROWS;
                m.mtp_walk(
                    MtpFeed::Rows {
                        tokens: run,
                        pos0: u32::try_from(q)?,
                        hidden: MtpHidden::Host(
                            &streams[(q - 1) * WIDE..(q - 1 + run.len()) * WIDE],
                        ),
                    },
                    h,
                    e,
                )?;
            }
            Ok(())
        };
        walk_host(&mut m, e, MTP_ROWS)?;
        let eager_store = store_of(&m, n)?;
        overwrite(&mut m)?;
        walk_host(&mut m, MtpMode::Store, HOST_STORE_RUN)?;
        let host_store = store_of(&m, n)?;
        let (dk, dv) = store_rel(&host_store, &eager_store, n);
        let (bk, bv) = (key_band(), value_band());
        let band_ok = dk <= bk && dv <= bv;
        println!(
            "(s) the store walks ({path:?}, runs of {HOST_STORE_RUN}) over a store other ids \
             overwrote, against the eager walks over the same {n} positions: keys within \
             {dk:.3e} (band {bk:.3e}), values within {dv:.3e} (band {bv:.3e}) {}",
            verdict(band_ok)
        );
        let anchor = |m: &mut Qwen38Model| {
            walk(
                m,
                MtpFeed::Rows {
                    tokens: &prompt[..1],
                    pos0: u32::try_from(n).expect("a prompt fits u32"),
                    hidden: MtpHidden::Host(&streams[(n - 1) * WIDE..]),
                },
                h,
                e,
            )
        };
        let host = anchor(&mut m)?;
        overwrite(&mut m)?;
        let overwritten = same_rows(&host_store, &store_of(&m, n)?, n)
            .iter()
            .filter(|&&same| !same)
            .count();
        let cfg = app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: bloomery_gpu::model::StepMode::Eager,
        };
        let (mut s, mut spec) = drafted_session(m, cfg, ctx)?;
        spec.prompt(&mut s, prompt)?;
        let mut m = s.into_model();
        let same_at = same_rows(&host_store, &store_of(&m, n)?, n);
        let stored = same_at.iter().filter(|&&same| same).count();
        let store_ok = overwritten == n && stored == n;
        let first_off = same_at.iter().position(|&same| !same);
        let call = anchor(&mut m)?;
        let same = same_bits(&host, &call);
        println!(
            "(w) the prompt call ({path:?}) walks the draft over {n} positions in {} units as the \
             store walks fed from the host: other ids overwrote {overwritten} of {n} stored rows, \
             then the call's warmup stored {} of {n} as the host's (first other at {first_off:?}) \
             {}; the anchor's token {} (host {}), streams and logits bit for bit {}",
            units.len(),
            stored,
            verdict(store_ok),
            call.draft.tokens[0],
            host.draft.tokens[0],
            verdict(same)
        );
        Ok((m, band_ok && store_ok && same))
    }

    /// The draft's stored keys and values at positions `0..n`, position-major.
    fn store_of(m: &Qwen38Model, n: usize) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let d = m.body("store")?.mtp().ok_or("the load opened no draft")?;
        Ok(d.store_host(m.gpu().stream(), n)?)
    }

    /// The largest relative distance over the positions below `n` of the
    /// keys, then of the values, in `a` from those in `b` (two [`store_of`]
    /// reads of `n` positions), a position's row at a time.
    fn store_rel(a: &(Vec<u16>, Vec<u16>), b: &(Vec<u16>, Vec<u16>), n: usize) -> (f64, f64) {
        let row = a.0.len() / n.max(1);
        let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| half_to_f32(x)).collect() };
        let worst = |x: &[u16], y: &[u16]| {
            (0..n)
                .map(|p| {
                    let r = p * row..(p + 1) * row;
                    rel(&f(&x[r.clone()]), &f(&y[r]))
                })
                .fold(0.0f64, f64::max)
        };
        (worst(&a.0, &b.0), worst(&a.1, &b.1))
    }

    /// Whether each position below `n` holds bit for bit the same keys and
    /// values in `a` and `b` (two [`store_of`] reads of `n` positions).
    fn same_rows(a: &(Vec<u16>, Vec<u16>), b: &(Vec<u16>, Vec<u16>), n: usize) -> Vec<bool> {
        let row = a.0.len() / n.max(1);
        (0..n)
            .map(|p| {
                let r = p * row..(p + 1) * row;
                a.0[r.clone()] == b.0[r.clone()] && a.1[r.clone()] == b.1[r]
            })
            .collect()
    }

    /// The drafted session's cfg, its prompt path `path`.
    fn step_cfg(path: Prompt38) -> app::arch::qwen3moe::Q38Cfg {
        app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: bloomery_gpu::model::StepMode::Graph,
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
        let (m, auto_ok) = prompt_walks(m, prompt, Prompt38::Auto, ctx)?;
        let (m, pass_ok) = prompt_walks(m, prompt, Prompt38::Pass, ctx)?;
        let mut ok = pass_ok && auto_ok;

        // The plain reference: every step's live stores, four tokens past
        // the drafted loop's count (its last pass may keep rows past it).
        let (m, shallow) = plain_run(m, prompt, Prompt38::Auto, N + 4)?;
        let (mut m, out, kept, unkept) = {
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
            let unkept = spec.draft_mut().take_windows();
            (s.into_model(), out, k, unkept)
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

        let (model, kept_ok) = kept_windows(m, prompt, ctx, (&out.tokens, &kept), unkept.len())?;
        m = model;
        ok &= kept_ok;

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
        let drafted = {
            let (gpu, _, b) = m.body_parts("windows")?;
            b.stores_host(gpu)?
        };
        // The plain run to the drafted run's end, its stores read there.
        let n_plain = kept.pos as usize - deep.len() + 1;
        let (m2, plain_tokens, plain_stores) = plain_to(m, deep, Prompt38::Auto, n_plain)?;
        let mut m = m2;
        let ids_ok = out.tokens[..N] == plain_tokens[..N];
        ok &= ids_ok;
        println!(
            "(w) depth {}: the drafted run's {N} ids = the plain run's {}",
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
        let stores_ok = live_same(&drafted, &plain_stores, kept.pos as usize);
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

    /// (w) prompt calls that continue the held sequence: after the drafted
    /// windows of `prompt`, three ids the model did not generate fed by the
    /// rule's prompt call, then windows. The draft joins at the call's start
    /// (the rows the last window left waiting walked, the call's first id at
    /// their last position) and drafts, and the ids are the plain run's —
    /// the same prompt, the drafted run's fed ids stepped, the same three ids
    /// by the same prompt path. Then a join the draft cannot make: restarted
    /// beside a target that holds the sequence, it skips the next call by
    /// name and proposes nothing, the ids still the plain run's.
    fn continued(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 8;
        let ctx = m.body("continued")?.ctx() as u32;
        let ext: Vec<u32> = prompt.iter().copied().skip(1).take(3).collect();
        let p0 = prompt.len();
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        let stop = runtime::Stop::new(N, ctx)?;
        let out1 = runtime::generate(&mut s, &mut spec, prompt, first, &stop, &mut QuietSink)?;
        let p1 = s.pos() as usize;
        let fed1 = out1.tokens[..p1 - p0].to_vec();
        let own = out1.tokens[p1 - p0];
        let first2 = spec.prompt(&mut s, &ext)?;
        let join = spec.draft_mut().take_joined();
        let mut k2 = Kept::default();
        let out2 = runtime::generate(&mut s, &mut spec, &ext, first2, &stop, &mut k2)?;
        let p2 = s.pos() as usize;
        let fed2 = out2.tokens[..p2 - p1 - ext.len()].to_vec();
        spec.draft_mut().restart();
        let first3 = spec.prompt(&mut s, &ext)?;
        let skip = spec.draft_mut().take_joined();
        let mut k3 = Kept::default();
        let out3 = runtime::generate(&mut s, &mut spec, &ext, first3, &stop, &mut k3)?;
        let mut m = s.into_model();

        // The plain run: the same prompt calls, the drafted run's fed ids
        // stepped between them.
        m.reset()?;
        let plain = |m: &mut Qwen38Model, ids: &[u32]| -> Result<Vec<u32>, GateError> {
            let mut t = vec![m.prompt38(ids, Prompt38::Auto)?];
            for _ in 1..N {
                t.push(m.step(&[t[t.len() - 1]])?);
            }
            Ok(t)
        };
        let _ = m.prompt38(prompt, Prompt38::Auto)?;
        for &x in &fed1 {
            m.step(&[x])?;
        }
        let plain2 = plain(&mut m, &ext)?;
        m.reset()?;
        let _ = m.prompt38(prompt, Prompt38::Auto)?;
        for &x in fed1.iter().chain(&ext).chain(&fed2) {
            m.step(&[x])?;
        }
        let plain3 = plain(&mut m, &ext)?;
        m.reset()?;

        let joined_ok =
            join.is_some_and(|j| j.start as usize == p1 && j.caught_up >= 1 && j.skipped.is_none());
        let drafted_ok = k2.proposed.iter().any(|&p| p);
        let ids2_ok = out2.tokens[..N] == plain2[..];
        let skip_ok =
            skip.is_some_and(|j| j.start as usize == p2 && j.caught_up == 0 && j.skipped.is_some());
        let none_ok = !k3.proposed.iter().any(|&p| p);
        let ids3_ok = out3.tokens[..N] == plain3[..];
        println!(
            "(w) a prompt call of {ext:?} continuing at {p1} (the model's own next id {own}): \
             joined {join:?} {}, windows kept {:?} drafted {}, its {N} ids = the plain run's {}",
            verdict(joined_ok),
            k2.rows,
            verdict(drafted_ok),
            verdict(ids2_ok)
        );
        println!(
            "(w) the draft restarted, a prompt call continuing at {p2}: joined {skip:?} {}, \
             {} passes proposed nothing {}, its {N} ids = the plain run's {}",
            verdict(skip_ok),
            k3.rows.len(),
            verdict(none_ok),
            verdict(ids3_ok)
        );
        let ok = joined_ok && drafted_ok && ids2_ok && skip_ok && none_ok && ids3_ok;
        Ok((m, ok))
    }

    /// A Q8_0 weight's two planes' device addresses.
    fn planes(dw: Option<&DevWeight>) -> Option<[u64; 2]> {
        match dw {
            Some(DevWeight::Q8_0 { qs, d, .. }) => {
                Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()])
            }
            _ => None,
        }
    }

    /// (d) the head's map holds one word a vocabulary id, its first `ids`
    /// and the rest 0: the head reads the list's rows of the target's
    /// `output` through it, in place.
    fn head_rows(m: &Qwen38Model, mtp: &Mtp38, ids: &[u32]) -> Result<bool, GateError> {
        let stream = m.gpu().stream();
        let Some((map, n)) = mtp.head_map() else {
            println!("(d) the draft has no head map FAIL");
            return Ok(false);
        };
        let map = map.to_host_vec(stream)?;
        let vocab = m.weights().get("output.weight").map_or(0, |dw| match dw {
            DevWeight::Q8_0 { d, .. } => d.rows(),
            _ => 0,
        });
        let ok = n == ids.len()
            && map.len() == vocab
            && map.get(..n) == Some(ids)
            && map[n..].iter().all(|&w| w == 0);
        println!(
            "(d) head map: {} words (output's {vocab} rows), {n} rows, the list's ids then 0 {} \
             (want {LIST_ROWS})",
            map.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// (d) and (q), on the model as the load left it: no walk has run.
    fn draft_load(
        m: &Qwen38Model,
        plan: MtpPlan<'_>,
        inputs: &PlanInputs,
        draft: &Split,
        mtp: &MtpInputs,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        let target = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let d = &plan.draft.cards[0];
        let body = m.body("mtp load")?;
        let Some(mtp38) = body.mtp() else {
            return Err("the load opened no MTP draft".into());
        };
        let want = plan.draft_resident_bytes() + d.kv_bytes + plan.map_bytes;
        let got = mtp38.resident_bytes() as u64;
        let mut ok = got == want;
        println!(
            "(d) draft resident {got} (want {want}: weights {} + store {} + map {}) {}",
            plan.draft_resident_bytes(),
            d.kv_bytes,
            plan.map_bytes,
            verdict(got == want)
        );
        for b in mtp38.borrowed_planes() {
            let target_planes = planes(m.weights().get(&b.name));
            let same = target_planes == Some(b.planes);
            ok &= same;
            println!(
                "(d) {} read at {:x?}, the target's at {target_planes:x?} {}",
                b.name,
                b.planes,
                verdict(same)
            );
        }
        let arena = mtp38.arena_bytes() as u64;
        let arena_ok = arena == plan.arena_bytes;
        ok &= arena_ok;
        println!(
            "(d) the draft program's arena {arena} (the plan's {}) {}",
            plan.arena_bytes,
            verdict(arena_ok)
        );
        let borrowed = mtp38.borrowed(m.weights()).is_ok();
        ok &= borrowed;
        println!("(d) the walk's borrow resolves: {}", verdict(borrowed));
        let rows_ok = head_rows(m, mtp38, ids)?;
        ok &= rows_ok;
        let mut own = MtpInputs::read(draft, &target, inputs, HeadRows::Full)?;
        if let MtpSource::File { borrows, .. } = &mut own.draft.source {
            *borrows = Borrows {
                embedding: false,
                head: false,
            };
        }
        let r1 = Mtp38::open(m.gpu(), m.weights(), draft, &own, &plan, false, 1)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r1_ok = r1.contains("the load borrows the target's token_embd and output");
        ok &= r1_ok;
        println!("(q) a draft with its own matrices: {r1} {}", verdict(r1_ok));
        let mut other = plan;
        other.map_bytes += 4;
        let r2 = Mtp38::open(m.gpu(), m.weights(), draft, mtp, &other, false, 1)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r2_ok = r2.contains("resident weights, store and row map hold");
        ok &= r2_ok;
        println!("(q) a plan of another row map: {r2} {}", verdict(r2_ok));
        Ok(ok)
    }

    /// (u) the head a drafted load of this target picks with
    /// `BLOOMERY_MTP_HEAD_ROWS` unset is the shipped list: its rows, its
    /// first line naming the target tokenizer's digest, the pick's reason
    /// `shipped`; `full` set is the full head.
    fn shipped_head(file: &Split, vocab: u32) -> Result<bool, GateError> {
        let want = vocab_sha256(file)?;
        let first = SHIPPED.lines().next().unwrap_or_default();
        let line_ok = first.contains(&format!(" rows={SHIPPED_ROWS} "))
            && first.ends_with(&format!(" vocab_sha256={}", hex(&want)));
        let pick = head_rows_of(None, file, vocab)?;
        let rows_ok = matches!(&pick.rows, HeadRows::List { ids, digest }
            if ids.len() == SHIPPED_ROWS && *digest == want);
        let unset_ok = line_ok && rows_ok && pick.why == HeadWhy::Shipped;
        println!(
            "(u) unset: head {} of {} rows from {} ({}); the list's first line {first:?}, the \
             target's tokenizer {} {}",
            pick.head_word(),
            pick.rows_of(vocab),
            pick.why.from_word(),
            pick.why,
            hex(&want),
            verdict(unset_ok)
        );
        let full = head_rows_of(Some(bloomery_levers::MtpHead::Full), file, vocab)?;
        let full_ok = full.rows == HeadRows::Full && full.why == HeadWhy::SetFull;
        println!(
            "(u) full: head {} of {} rows from {} ({}) {}",
            full.head_word(),
            full.rows_of(vocab),
            full.why.from_word(),
            full.why,
            verdict(full_ok)
        );
        Ok(unset_ok && full_ok)
    }

    // ------------------------------------------------------ (y) two slots

    /// The drafted session's speculative over this body.
    type Spec38 = runtime::Speculative<app::mtp::MtpDraft<Body38>, 4>;

    /// The server's first plain step after a request's prompt call
    /// (`serve_seats::drafted`'s step): the rows the last call left waiting
    /// walked, the step read, then the step told to the draft.
    fn plain_step(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        last: u32,
    ) -> Result<u32, GateError> {
        spec.draft_mut().before_step(s, last)?;
        let next = Target::step(s, last, Want::Argmax)?.argmax();
        Draft::stepped(spec.draft_mut(), s, last, next)?;
        Ok(next)
    }

    /// Make `to` the slot every later call acts on, exactly as a seat
    /// switches: the live slot's draft state parked before the select away,
    /// the target's own put back after the select back, the parked states
    /// kept one a slot.
    fn switch(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        parked: &mut [Option<app::mtp::Parked<TargetRows>>; 2],
        to: usize,
    ) -> Result<(), GateError> {
        let from = s.model().selected();
        let live = spec.draft().park();
        s.model_mut().select_slot(to)?;
        if let Some(back) = parked[to].take() {
            spec.draft_mut().unpark(&back);
        }
        parked[from] = Some(live);
        Ok(())
    }

    /// One drafted request's start as the server runs it (`worker.rs`'s
    /// `start` → `genloop`'s `prompt`): the prompt call, then its first
    /// plain step, together — no select between them. Returns the ids so
    /// far (the prompt's argmax, then the first step's).
    fn begin_request(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        ids: &[u32],
    ) -> Result<Vec<u32>, GateError> {
        let first = Advance::prompt(spec, s, ids)?;
        Ok(vec![first, plain_step(s, spec, first)?])
    }

    /// One drafted request as the server runs it when nothing switches
    /// beneath it (the alone run): the start of [`begin_request`], then
    /// `rounds` drafted passes — every id the passes kept, and every pass's
    /// proposal and kept rows.
    fn drive(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        ids: &[u32],
        rounds: usize,
    ) -> Result<(Vec<u32>, Vec<Committed>), GateError> {
        let mut out = begin_request(s, spec, ids)?;
        let mut passes = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            passes.push(one_pass(s, spec, &mut out)?);
        }
        Ok((out, passes))
    }

    /// One drafted pass from `out`'s last id, its kept ids appended.
    fn one_pass(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        out: &mut Vec<u32>,
    ) -> Result<Committed, GateError> {
        let last = *out.last().ok_or("no token")?;
        let mut kept = Vec::new();
        let c = Advance::pass(spec, s, last, &mut kept)?;
        out.append(&mut kept);
        Ok(c)
    }

    /// The selected slot's draft store, read back at the count its own walks
    /// hold: the rows the sequence's walks wrote.
    fn slot_store(s: &app::Session<Body38>) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let m = s.model();
        let n = m
            .body("slots")?
            .mtp()
            .ok_or("the load opened no draft")?
            .held();
        store_of(m, n)
    }

    /// (y) (module doc): two drafted sequences over one plan of two, in the
    /// server's order, against each window's single-sequence reference.
    fn slots_windows(
        inputs: &PlanInputs,
        mtp: &MtpInputs,
        levers: &bloomery_levers::Levers,
        ub: usize,
        a: &[u32],
        b: &[u32],
    ) -> Result<bool, GateError> {
        const ROUNDS: usize = 8;
        let ctx = u32::try_from(CTX)?;
        let (file, draft_file) = (
            Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?,
            Split::open(DRAFT).map_err(|e| format!("open {DRAFT}: {e}"))?,
        );
        let machine = machine(RTX_3090, inputs.spec.layers.len(), u64::try_from(ub)?);
        let plan = inputs.plan_mtp_with_slots(
            &machine,
            CTX,
            &PlanLevers::from_levers(levers)?,
            mtp,
            Experts::Host,
            2,
        )?;
        let mut m = Body38::open_placed_mtp_slots(
            file,
            &plan,
            inputs,
            0,
            levers.host(),
            ub,
            &draft_file,
            mtp,
            2,
        )?;
        m.set_mode(bloomery_gpu::model::StepMode::Graph);
        let mut s = app::Session::from_model(m, ctx);
        let mut spec = s.with_draft::<app::mtp::MtpDraft<Body38>, 4>(
            app::mtp::MtpDraft::open(
                s.model(),
                Prompt38::Auto,
                bloomery_gpu::model::StepMode::Graph,
            )?,
            &mut Quiet,
        )?;
        // The references, each on the load's own single sequence before any
        // slot exists: what the server's one-request runs produce, so the
        // two-slot run is judged against runs the slots' exchange cannot
        // have shaped.
        let (a_ids, a_passes, a_store) = {
            s.reset()?;
            spec.draft_mut().restart();
            let run = drive(&mut s, &mut spec, a, ROUNDS)?;
            (run.0, run.1, slot_store(&s)?)
        };
        let (b_ids, b_passes, b_store) = {
            s.reset()?;
            spec.draft_mut().restart();
            let run = drive(&mut s, &mut spec, b, ROUNDS)?;
            (run.0, run.1, slot_store(&s)?)
        };
        s.model_mut().add_slots(2)?;
        // Together, in the server's order: slot 0's prompt call and first
        // plain step run together, then slot 1's, then the drafted passes
        // interleave — a select between every pass. Each slot's request
        // starts from its own reset.
        let mut parked = [None, None];
        s.reset()?;
        spec.draft_mut().restart();
        let mut t_a = begin_request(&mut s, &mut spec, a)?;
        switch(&mut s, &mut spec, &mut parked, 1)?;
        s.reset()?;
        spec.draft_mut().restart();
        let mut t_b = begin_request(&mut s, &mut spec, b)?;
        let (mut p_a, mut p_b) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            switch(&mut s, &mut spec, &mut parked, 0)?;
            p_a.push(one_pass(&mut s, &mut spec, &mut t_a)?);
            switch(&mut s, &mut spec, &mut parked, 1)?;
            p_b.push(one_pass(&mut s, &mut spec, &mut t_b)?);
        }
        let ids_ok = t_a == a_ids && t_b == b_ids;
        let passes_ok = p_a == a_passes && p_b == b_passes;
        // Each slot's draft store, read back at its own held count: the rows
        // its walks wrote, bit for bit its single-sequence reference's — the
        // ids and the passes' kept counts alone cannot see a store another
        // slot's walks filled (a proposal the target refuses every row of
        // leaves them as it found them).
        let stores_ok = slot_store(&s)? == b_store && {
            switch(&mut s, &mut spec, &mut parked, 0)?;
            slot_store(&s)? == a_store
        };
        println!(
            "(y) two slots in the server's order: {} drafted passes a slot, a select between \
             every pass; each slot's ids {} its single-sequence reference's, its proposals and \
             acceptances {}, its draft store {} {}",
            ROUNDS,
            verdict(ids_ok),
            verdict(passes_ok),
            verdict(stores_ok),
            verdict(ids_ok && passes_ok && stores_ok)
        );
        Ok(ids_ok && passes_ok && stores_ok)
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
            "bands: eh {:.4e} router {:.4e} the argmax cap's {:.4e} free {FREE_BAND:.2}; the \
             node-local pins: γ(k/32 + 5) a dot, the pack ½γ(52) + 6u = {:.3e}, the mixes \
             MIXED_BAND {:.1e}, the combines' weights LO_BAND {:.1e}",
            eh_band(),
            router_band(),
            logits_band(),
            pack_rel(),
            MIXED_BAND,
            LO_BAND
        );
        let open = |p: &str| Split::open(p).map_err(|e| format!("open {p}: {e}"));
        let (file, draft) = (open(MODEL)?, open(DRAFT)?);
        let t = Instant::now();
        let inputs = PlanInputs::describe(&file)?;
        let shipped_ok = shipped_head(&file, inputs.spec.vocab)?;
        let ids: Vec<u32> = (0..LIST_ROWS).map(|i| i * 6).collect();
        let rows = HeadRows::List {
            ids: ids.clone().into(),
            digest: vocab_sha256(&file)?,
        };
        let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX)?)?;
        let machine = machine(RTX_3090, inputs.spec.layers.len(), u64::try_from(ub)?);
        let plan = inputs.plan_mtp(&machine, CTX, &PlanLevers::from_levers(&levers)?, &mtp)?;
        let d = &plan.draft.cards[0];
        println!(
            "plan card={} ctx_max={CTX} draft dense={} experts={} rounding={} kv={} map={} \
             arena={} headroom={}",
            RTX_3090.name,
            d.dense_bytes,
            d.expert_bytes,
            d.rounding_bytes,
            d.kv_bytes,
            plan.map_bytes,
            plan.arena_bytes,
            plan.headroom_bytes
        );
        let mut m =
            Body38::open_placed_mtp(file, &plan, &inputs, 0, levers.host(), ub, &draft, &mtp)?;
        let arena = m.body("mtp")?.mtp().map_or(0, |d| d.arena_bytes());
        println!(
            "load card={} ctx_max={CTX} resident_bytes={} in {:.1} s; the draft program's arena \
             {arena} bytes (runtime value, outside the plan)",
            RTX_3090.name,
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        let mut ok = shipped_ok;
        ok &= draft_load(&m, plan, &inputs, &draft, &mtp, &ids)?;
        ok &= refused(
            "the draft's own row before any walk",
            m.mtp_draft(MtpFeed::Own { pos0: 0 }, MtpHead::Rows, MtpMode::Eager),
            "a walk before the draft's own row",
        );

        let head_file = open(MODEL)?;
        let host = Host::open(&head_file, &draft, inputs.hp.rms_eps)?;
        m.set_mtp_taps(true)?;
        let mut r = Replay::new();
        let mut last = None;
        let mut warm = None;
        for &(b, g) in &order {
            let ig = IkGraph::read(&set, b, g)?;
            replay(&mut m, &ig, &host, &mut r)?;
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
        let l_ok = r.flips_bad == 0 && r.l_out_held > 0;
        println!(
            "(l) {} rows, {} on flips, {} flips not allowed; l_out's distance from ik's on the {} \
             rows held {:.3e} (printed, not held: the rounding of ik's q8 activations) {}",
            r.rows,
            r.flips,
            r.flips_bad,
            r.l_out_held,
            r.l_out_worst,
            verdict(l_ok)
        );
        let h_ok = r.argmax_bad == 0 && r.kernel_bad == 0 && r.heads_held > 0;
        println!(
            "(h) {} head rows, {} held: argmax = ik's {}, ties {}, wrong {}; ik's smallest top-2 \
             margin {}; the draft's token = our argmax but {}; the logits' distance from ik's \
             {:.3e} (printed, not held) {}",
            r.heads,
            r.heads_held,
            r.argmax_same,
            r.argmax_tie,
            r.argmax_bad,
            r.margin_min
                .as_ref()
                .map_or("-".to_string(), |(m, q, at)| format!(
                    "{m:.3e}, {q:.2} of its tie cap, at {at}"
                )),
            r.kernel_bad,
            r.logits_worst,
            verdict(h_ok)
        );
        let mut n_ok = true;
        for p in r.pins.all() {
            println!("{}", p.line());
            n_ok &= p.ok();
        }
        ok &= e_ok && l_ok && h_ok && n_ok;
        let map = m
            .body("list")?
            .mtp()
            .and_then(|d| d.head_map())
            .map(|(map, n)| -> Result<Vec<u32>, GateError> {
                let mut v = map.to_host_vec(m.gpu().stream())?;
                v.truncate(n);
                Ok(v)
            })
            .transpose()?
            .ok_or("the load opened no row list")?;
        ok &= map == ids;
        ok &= list_head(&mut m, &last, &map)?;
        m.set_mtp_taps(false)?;
        ok &= pairing(&mut m, &prompt, &warm)?;
        ok &= graphs(&mut m, &warm)?;
        ok &= refusals(&mut m, &warm)?;
        let (m, w_ok) = windows(m, &prompt, &deep_prompt()?)?;
        ok &= w_ok;
        let (mut m, c_ok) = continued(m, &prompt)?;
        ok &= c_ok;
        ok &= arena_holds(&mut m, &prompt)?;
        let (model, p_ok) = kept_state(m, &prompt)?;
        ok &= p_ok;
        let (model, x_ok) = resume_refused(model, &inputs, &levers, ub, &prompt)?;
        ok &= x_ok;
        drop(model);
        // Window B diverges from window A from position 0 on: the corpus's
        // deep window read past A's whole length, so no per-sequence state
        // the two share by content can pass for one the slots exchanged.
        let deep = deep_prompt()?;
        ok &= slots_windows(&inputs, &mtp, &levers, ub, &prompt, &deep[1000..])?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
