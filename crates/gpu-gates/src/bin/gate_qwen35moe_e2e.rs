//! The Qwen3.6-35B-A3B end-to-end gate: the whole chain — 30 gated-delta-rule
//! layers, 10 gated GQA layers at head 256, 40 MoE blocks with the shared
//! expert as a ninth slot, the head and the argmax — on one card, against
//! ik's CPU oracle sets (`refset::arch::qwen35moe`: the 5-token batch set,
//! the step after a 4-token prefill, the step after a 1,024-token prefill),
//! all read in one process over one load.
//!
//! What is asserted:
//! - (s) structure: the captured decode step holds [`NODES_DECODE`] nodes,
//!   all kernels; a captured pass of `m` rows holds [`NODES_PASS`] `+ 4m`
//!   (`m` memcpys of a row into its head, the rest kernels), at `m` = 5 and
//!   8; each count also equals the body's own launch count
//!   (`Body35::pass_launches`) plus its heads. The layer stores' bytes equal
//!   their derivation from the header ([`store_bytes`]). The flash
//!   launches' grids: the segment pass `n_kv · SEGMENTS` blocks an
//!   attention layer and the merge `n_head`, one of each an attention
//!   layer, whatever the cache height the load allocated.
//! - (p) one layer body: the batch set's five tokens as five decode steps
//!   in graph mode, as one pass of five rows (`step_rows`) and as five eager
//!   steps, each from zero stores written by the gate, then as five graph
//!   steps from `reset` alone after the eager run's state, leave the same
//!   per-position logits and tokens and every layer's store (K/V planes,
//!   recurrent state, conv ring) bit for bit. No reduction differs between
//!   the decode step and the pass: every launch of a pass computes each row
//!   the way its one-row launch does (the FFN norm inside the router at one
//!   row is `norm_quant` + the router bit for bit, gate_qwen35moe_moe's norm
//!   clause; the flash and rope rows are their one-row launches',
//!   gate_qwen35moe_attn; the q4_K projections run the one-column body;
//!   `q6k_gemv` keeps one accumulator chain per column whatever the column
//!   count). Every other clause starts from the gate's zero stores, so a
//!   `reset` that keeps state reddens the last line of this clause alone.
//! - (c) free-running on the batch set: the five decode steps' every layer
//!   output against ik's `l_out-L` within [`FREE_BAND`], and the last
//!   position's argmax equal to ik's `result_output` argmax.
//! - (f) teacher-forced, per layer, on the batch set as one pass of five
//!   rows from a reset: each layer run alone on ik's own input rows
//!   (`inp_embd` for layer 0, `l_out-(L−1)` after), its mixer's taps against
//!   ik's, and its FFN half run alone on ik's own FFN input against ik's MoE
//!   taps; every tap's relative error over the relative distance between
//!   the two sides' 8-bit activations of the input that reaches it
//!   ([`quant_gap`]) within [`RATIO_BAND`]; the routed ids equal as sets
//!   except where ik's own margin between its eighth pick and the best
//!   other expert lies inside our logits' measured error (counted,
//!   printed); the decay's underflow sites (exp(g) = 0) equal.
//! - (t) the decode step after each step set's prefill: the prefill's state
//!   loaded from the set's inputs (`cache_s_lL` — conv then ssm —,
//!   `cache_k_lL`, `cache_v_lL`) into every layer's store, the model stood at
//!   the set's position, then (t1) the step free-running: every layer
//!   output within [`FREE_BAND`], the argmax equal to ik's, every delta
//!   layer's new state within [`RATIO_BAND`] of the input gap; and (t2) each
//!   layer teacher-forced at one row on its reloaded store, its taps as (f).
//! - (p′) the ubatch arena's gemv arm: the batch set's five tokens through
//!   the prompt call (`prefill_with`) from zero stores — on the pass path
//!   (one unit of five rows) and on the GEMM path (a ubatch of five, which
//!   is no more than `GEMV_COLS` and so the same walk) — leave (p)'s five
//!   decode steps' last token, last logits (the last unit's last row
//!   through the head, `Tail::Last`) and every store bit for bit.
//! - (u) the wide arm against the gemv arm: the 1,024 prompt ids of the
//!   step-1,024 set prefilled on the pass path (128 units of eight, bit for
//!   bit the one-token path) and as one ubatch of 1,024 (the wide arm), each
//!   from zero stores: layer 0's recurrent state and conv ring within
//!   [`U_L0_REL`] of the pass run's; every layer's store distance and the
//!   prefill's last logits printed, and (d) prints each layer's store
//!   distance between the two runs beside its distances from ik.
//! - (d) both prefills against ik's state after its 1,024-token prompt batch
//!   (the set's `cache_s_lL`, `cache_k_lL`, `cache_v_lL`): each layer's
//!   store distance from ik's, the ubatch run's over the pass run's, within
//!   [`PROMPT_RATIO`]; then the step at position 1,024 free-running from each
//!   run's own state: each layer's output distance from ik's `l_out`, and the
//!   logits' from ik's `result_output`, the ubatch run's over the pass run's
//!   within [`STEP_RATIO`]; the three argmaxes printed.
//! - (w) the ubatch size moves no bit: the same 1,024 ids on the GEMM path at
//!   each size of [`W_SIZES`] (ubatches of 512 x 2; 1,000 and 24; 100 x 10
//!   and 24), and at [`U_GATE`] in two calls cut at [`W_CUT`], leave the last
//!   token, the last logits and every store of the one-ubatch run bit for
//!   bit. Every unit of these runs is wide: a unit of at most `GEMV_COLS`
//!   rows is the gemv arm, a band away.
//! - (k) checkpoints, on a second load at [`KCTX`] (the marks' positions
//!   need the room [`CTX`] does not hold) with the marks armed and an lcg
//!   prompt of [`KP`] ids on the seat's schedule
//!   (`PrefillPath::Wide`): a prompt call takes its checkpoints at the
//!   load's ubatch multiples inside it and [`KP`] (the spacing pinned in
//!   the clause), and one checkpoint's bytes and slots are their derivation
//!   (each delta layer's state of `N_V · HEAD_V²` f32 and conv ring of
//!   `RING_ROWS · C`, the host budget's share); the call runs exactly
//!   ⌈[`KP`]/ub⌉ walks, one a unit of its own plan, the re-fed tail after a
//!   cut one walk; a cut to an ask between the last inner mark and [`KP`]
//!   keeps it, one below that keeps the mark under it, one below the first
//!   keeps nothing (`no-checkpoint`), each with its code, and a cut to a
//!   non-checkpoint is refused by name; a cut to the last inner mark and
//!   the re-fed tail leave the token, the last logits and every store bit
//!   for bit the uncut call's (the call takes its points back, the mark
//!   held); a reset drops every point, and the captured step after it holds
//!   the flash launches at this load's height with (s)'s grids — the
//!   engine-level two-height pin, no second load of its own. FAIL-first
//!   mutants, each red on its
//!   line: the spacing put back at the flat 512 it was (the points and the
//!   walk count red); the restore omitted (the cut a Stay: the re-feed runs
//!   from the fed state, the stores and logits red); `kept` answering the
//!   ask instead of the checkpoint (the neighbours red, and the refused cut
//!   not refused); `reset` keeping the checkpoints (the dropped-points line
//!   red).
//! - (n) resident slots (`add_slots`), on their own load with the
//!   checkpoints armed, as the seat arms them: the slot harness's contracts
//!   (`slots_gate`: H1 interleave, H3 bytes, H4 reset, H6 refusals, H7
//!   captures) over two lcg streams, A of [`SLOT_A`] ids and B of
//!   [`SLOT_B`] — each past the load's first mark (its ubatch, [`U_GATE`])
//!   by a run the mark keeps — prompted by the session's schedule (the wide
//!   walk), [`SLOT_STEPS`] steps a stream, each slot's state digest every
//!   attention layer's K/V rows below its position and every delta layer's
//!   state and ring, and `seq_bytes` held to [`store_bytes`]. Between the
//!   harness's halves, each slot's checkpoints are its own: after the
//!   interleave slot 0 holds its points at the mark and its prompt's end,
//!   slot 1 at the mark and its own end; slot 1 cut back to the mark, slot
//!   0 — selected before slot 1 runs again, so a cut that waited in the
//!   wrong checkpoints is carried out on it — holds its points and digest
//!   and its next ids are its solo run's continuation; slot 1, the rest of
//!   B re-fed from the mark and stepped as far as the interleave stepped
//!   it, stands with the interleave's last id, digest, points and position;
//!   a reset of slot 1 leaves slot 0's points standing.
//! - (r) refusals: a ubatch size of 0 or past `UBATCH` is refused by name
//!   with the size and the resident bytes kept; a prompt past the cache is
//!   refused by name before any launch, the position kept.
//! - (v) the q8_0 cache arm (`BLOOMERY_QWEN3_KV=q8_0`, the seat's
//!   `--cache-type-k q8_0`), on its own models: the store bytes' derived
//!   drop (the attention planes' two-plane layout against f16, the delta
//!   layers' stores f32 either way) carrying the resident bytes' delta
//!   exactly, and 96 of the oracle set's ids stepped, prefilled on the
//!   pass path and replayed through the captured step graph leaving the
//!   same last logits bit for bit and the same argmax; the distance to the
//!   f16 run prints as the quantization diagnostic.
//! - (o) the placed load: the file planned on device 0 under a card budget
//!   of [`PLACED_BUDGET`] (`shared/qwen3moe_place.rs`, the CLI's and the
//!   seat's planner), which must leave routed experts both on the card and
//!   on the host tier, then loaded by that plan (`open_placed`, the shared
//!   expert its own one-expert stacks on the card): (o1) the batch set's
//!   five tokens as five graph steps and as five eager steps from zero
//!   stores leave the same logits, tokens and stores bit for bit, and so
//!   does the prompt call on the pass path (one unit of five rows through
//!   the host tier's batch port) against the fifth step — each row of a
//!   unit runs every card launch and every host dot the way its one-row
//!   step does; (o2) (c) on it, the same [`FREE_BAND`] — a host expert runs
//!   ik's own 8-bit rule (Q8_K activations) where the card runs q8_1, so no
//!   layer's error grows; (o3) a GEMM prompt and a ubatch resize refused by
//!   name, the ubatch kept; (o4) the host tier served slots on both of its
//!   ports. `--placed-only` runs (o) alone.
//!
//! (m) the memory guard, `--memguard-only` (no model loaded): the census
//!   free reading, the quiet-card whole decision, the held-card refusal by
//!   name and the placed decision, and the live reading once the hold drops
//!   — `gate_qwen3moe_e2e`'s (m) is the clause's full form.
//!
//! Tap maps (ik → ours). Delta layer: `qkv_mixed` → the q·k·v projection;
//! `z` → the gate projection; `beta_in`, `alpha` → β's and α's raw
//! projections; `g_in` → `ln` of our decay (`exp(g)`); the v channels of
//! `conv_output_silu` → the conv's v channels; `q_fused`, `k_fused` (the
//! L2 norm, a PERMUTE's src0 where the set names the permute) → the conv's
//! normed q (over `Q_SCALE`) and k; `attn_output` → the delta output;
//! `new_state` (`[k][v]` per head) → our state (`[v][k]`), transposed;
//! `new_conv_states_cont` (`[C][3]`) → the ring's last three positions;
//! `attn_out_norm` → the gated norm's output; `linear_attn_out` → the
//! residual update. Attention layer: `Qaux` → the `[q | gate]` rows;
//! `Qcur_normed` and `Qcur_roped` → our queries past and inside the 64
//! turned values; `Kcur_*` likewise; `Vcur`; `fa` → the flash output;
//! `qkv_gated` → our flash output times the sigmoid of our gate (a host
//! product: the kernel folds it into the quantizer); `attn_out` → the
//! residual update. MoE: `ffn_moe_logits` → the router's first 256 logits;
//! `ffn_moe_topk` → the first eight slots' ids; `ffn_moe_weights_norm` →
//! their weights, matched by id; `shared_expert_gate_sigmoid` → the ninth
//! slot's weight; `l_out` → the layer output.
//!
//! Known differences, named, not banded away: ik clamps nothing on these
//! sets (q35oracle Q1); layer 37's decay underflows to 0 at one site in 960,
//! and ours must underflow at the same sites; ik combines `(routed + resid)
//! + shexp_gated`, ours `((Σ8 w·d) + w8·d_sh) + resid` — one or two f32
//! roundings of the update, some 1e-7 of it, five orders under the band of
//! `l_out`, which is the only tap past that add.
//!
//! The chain runs the tensor-core flash pass, the engine's.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen35moe_e2e: built without the `gpu` feature; see `just gate-gpu-qwen35moe-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen35moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate plans and opens a qwen35moe file only; the other half serves the CLI and the seat"
)]
mod q3place;

#[cfg(feature = "gpu")]
#[path = "shared/flash_grid.rs"]
mod flash_grid;

#[cfg(feature = "gpu")]
mod gate {
    use super::flash_grid::flash_grids;
    use super::q3place::{self, PlaceQ3};
    use bloomery_gpu::NodeInfo;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::arch::qwen3moe::{
        Body35, Delta35Run, Gqa35Run, KvQ8, LayerKind35, Mixer35Run, Open35, PrefillPath,
        Qwen35moeModel, StoreHost,
    };
    use bloomery_gpu::linear::{Q_SCALE, RING_ROWS, expf_ik};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::route_core::sigmoid;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::slots_gate::{self, Derived, SlotsAdapter};
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, data_dir,
        ik_q8_2, q8_1_dequant, ref_ints, ref_tensor_logical_in, split_f32, topk_ids_logical_within,
        verdict, widened_f16_rows_in,
    };
    use bloomery_levers::HostCfg;
    use cuda_core::sys;
    use gguf::Split;
    use gguf::quant::half_to_f32;
    use model::arch::models::Mixer;
    use model::arch::models::shape::{AttnShape, MoeShape};
    use model::placement::PlanLevers;
    use refset::arch::qwen35moe::{BATCH, D1K, IK, MODEL, STEP4};
    use std::time::Instant;

    /// Cache rows: the 1,024-token set's step at position 1,024, with room.
    const CTX: usize = 1088;

    /// The ubatch size the gate opens with and every clause but (w) runs at:
    /// the step-1,024 set's prompt as one ubatch, and an arena of 1,024 rows
    /// (about 0.36 GB at some 350 KB a row, derived from the arena's
    /// buffers), which fits a gate lane on either card.
    const U_GATE: usize = 1024;

    /// (w)'s other ubatch sizes, and where its two-call run cuts the prompt.
    const W_SIZES: [usize; 3] = [512, 1000, 100];
    const W_CUT: usize = 24;

    /// PIN(2026-09-27): (u)'s band on layer 0's recurrent state and conv ring
    /// against the pass run's (`‖ubatch − pass‖ / ‖pass‖`). Layer 0's input is
    /// the same bits on both paths (the embedding rows; `rms_norm` then the
    /// GEMM quantizer write the bytes `norm_quant` writes, gate_qwen3moe's
    /// premise for the same kernels), so only its four projections differ,
    /// each within its sum's rounding of the exact dot of the same codes:
    /// about 1.2e-6 of `Σ |terms|` at K = 2,048, some four times a value, so
    /// Δ ≤ 1e-5 of a row (gate_qwen3moe's GEMM_L0_REL note). The ring holds
    /// the q·k·v projection's rows (Δ). The state is f32 throughout — no f16
    /// step turns Δ into an ulp, as the K/V rows' band had to — and each
    /// token's update is a product of its normed k, its v, β and the decay,
    /// each carrying Δ, under a map that does not expand (unit k, β and the
    /// decay at most 1); over 1,024 tokens whose updates partly cancel the
    /// sum's relative error is a few Δ. Predicted 1e-6 to 3e-5; a wiring
    /// fault in layer 0 (β read from α's weights, a token's position off)
    /// reads an error of order one.
    const U_L0_REL: f64 = 1e-4;

    /// PIN(2026-09-27): (d)'s bound on the step at position 1,024 free-running
    /// from each run's own state — each layer's output distance from ik's and
    /// the logits' — the ubatch run's over the pass run's. Derivation: past
    /// layer 0 the two runs' first difference (Δ, [`U_L0_REL`]) grows at every
    /// q8_1 re-quantization by code flips until the ubatch run is another q8_1
    /// realization of the same rule (gate_qwen3moe's GEMM_SPREAD_RATIO note),
    /// statistically the pass run's twin against ik; one step from each state
    /// reads about 1. A wiring fault — another layer's weight, a token off by
    /// one, a stale route row — adds an error of order one to distances of
    /// 1e-2 to 0.3, a ratio above 3 on every layer it reaches.
    const STEP_RATIO: f64 = 1.5;

    /// PIN(2026-09-28): (d)'s bound on a layer's store distance from ik (a
    /// recurrent state and the conv ring's rows ik keeps, or the K and V rows,
    /// the larger of the two), the ubatch run's over the pass run's. Raised from
    /// 1.5 [잠정 — 백로그]: 1.5 assumed the two runs' own distance d_wp is at
    /// most the pass run's distance from ik d_pi, so the ratio is at most
    /// √(1 + (d_wp/d_pi)²) ≤ √2; over 1,024 positions the recurrent stores
    /// integrate the flips and d_wp/d_pi reads 0.52, 1.03, 1.19, 1.41 at layers
    /// 8, 16, 24, 32, each measured ratio (0.97, 1.14, 1.30, 1.54) at or under
    /// that bound, the worst 1.72 at layer 29. 2.5 holds d_wp up to 2.3 d_pi;
    /// the wiring faults above still read past 3. A band derived from the flip
    /// amplification replaces this pin.
    const PROMPT_RATIO: f64 = 2.5;

    /// The file's shape, as the header states it (q35design §1): what the
    /// derivations below are written against.
    const HIDDEN: usize = 2048;
    const N_LAYER: usize = 40;
    const N_ATTN: usize = 10;
    const N_DELTA: usize = 30;
    /// Delta layers whose `attn_qkv` is Q6_K, and attention layers whose
    /// `attn_v` is (3, 7, 19, 31, 35, 39).
    const Q6_QKV: usize = 14;
    const Q6_V: usize = 6;
    const C: usize = 8192;
    const N_V: usize = 32;
    const HEAD_V: usize = 128;
    const Q_ROWS: usize = 8192;
    const KV_ROW: usize = 512;
    const N_HEAD: usize = 16;
    const N_KV: usize = 2;
    const HEAD: usize = 256;
    const ROT: usize = 64;
    const EPS: f64 = 1e-6;

    /// PIN(2026-09-27): the captured decode step's node count, derived before
    /// the chain was built: the embedding row; each delta layer's 8 mixer
    /// launches (norm+quant, the two projection launches — a Q6_K `attn_qkv`
    /// in its own gemv, `attn_gate`·β·α in one — conv, delta, gated norm,
    /// q8_1, `ssm_out` with the residual) and 5 FFN launches (the norm with
    /// the gated router, gate·up over the joined stacks, q8_1 of the nine
    /// slots, the down `_sel`, the combine); each attention layer's 7 mixer
    /// launches (norm+quant, q·k·v, rope-256, flash segment pass and merge,
    /// the gated q8_1, the output projection with the residual), one more
    /// on the 6 layers whose `attn_v` is Q6_K, and 5 FFN; then the head's
    /// three: 1 + 30·13 + 10·12 + 6 + 3.
    const NODES_DECODE: usize = 520;

    /// PIN(2026-10-05): a captured pass of `m >= 2` rows without its heads,
    /// derived the same way at more than one row: each delta layer 8 + 5
    /// (the norm with the gated router in one launch, as at one row), one
    /// more on the 14 whose `attn_qkv` is Q6_K (its token-major copy); each
    /// attention layer 7 + 5, two more on the 6 Q6_K `attn_v` layers (gemv
    /// and copy): 1 + 30·13 + 14 + 10·12 + 12. Each row's head adds a copy
    /// of the row's residual and three launches.
    const NODES_PASS: usize = 537;

    const _: () = assert!(
        NODES_DECODE == 1 + N_DELTA * 13 + N_ATTN * 12 + Q6_V + 3
            && NODES_PASS == 1 + N_DELTA * 13 + Q6_QKV + N_ATTN * 12 + 2 * Q6_V
            && N_DELTA + N_ATTN == N_LAYER
    );

    /// PIN(2026-09-27): the teacher-forced bound on a tap's error ratio — its
    /// relative error over the relative distance between the two sides'
    /// 8-bit activations of the input that reaches it (ours q8_1 per 128
    /// values, ik's q8_2 per 32: [`quant_gap`]; two inputs in quadrature).
    /// Derivation, the qwen3moe gate's: to first order a linear map carries
    /// its input's relative perturbation unchanged, so a projection reads
    /// about 1; the kernels are held to their host rules bit for bit or
    /// within γ(n) of them (gate_linear, gate_qwen35moe_attn,
    /// gate_qwen35moe_moe), terms some 1e-6 of the ~1e-2 gap, so the gap is
    /// the whole prediction. Predicted per class: the projections, β/α, z,
    /// the conv's v channels and the normed q/k about 1 (median), at most 3;
    /// the delta output and the state, a sum over the call's tokens of
    /// per-token errors the decay and β (both at most 1) do not amplify,
    /// at most √5 ≈ 2.2 times a token's, so at most 5; the flash output at
    /// most 3 (the softmax sharpens a score error, qwen3moe measured 2.8);
    /// the FFN at most 6 (qwen3moe measured 6.1 on a layer with one dominant
    /// channel). A wiring fault — another layer's weight, a head or a token
    /// off by one, the state transposed — reads an error of order one over a
    /// gap of about 2e-2, a ratio above 40.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-27): the free-running bound on a layer output's relative
    /// distance from ik's. Derivation, the qwen3moe gate's: the forced
    /// arm's per-layer errors, up to about 4e-2 of a layer's update, whose
    /// norm is of the residual's order, added in quadrature over 40 layers,
    /// independent: √40 · 4e-2 ≈ 0.25. The step sets' one step starts from
    /// ik's own state, so the same composition bounds it.
    const FREE_BAND: f64 = 0.26;

    /// The layer stores' bytes at [`CTX`] rows, derived from the header:
    /// each attention layer's K and V planes, `2 · n_kv · ctx · 256` f16;
    /// each delta layer's state, `32 · 128 · 128` f32 (one lane), and conv
    /// ring, `RING_ROWS · 8192` f32.
    fn store_bytes() -> usize {
        N_ATTN * 2 * N_KV * CTX * HEAD * 2 + N_DELTA * (N_V * HEAD_V * HEAD_V + RING_ROWS * C) * 4
    }

    /// `‖a − b‖ / ‖base‖` over the values, in f64.
    fn rel(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += base(i).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// `rel` of our last `ik.len()` values against ik's, on ik's own norm:
    /// a tap ik keeps only the output token's rows of (the last layer past
    /// its attention) meets our last rows. Infinite when ik holds more
    /// values than ours, and a NaN reads infinite, so neither passes a band.
    fn rel_to(ours: &[f32], ik: &[f32]) -> f64 {
        let Some(tail) = ours.len().checked_sub(ik.len()).map(|s| &ours[s..]) else {
            return f64::INFINITY;
        };
        worse(0.0, rel(tail, ik, |i| f64::from(ik[i])))
    }

    /// The larger of two errors, a NaN counting as infinite: `f64::max`
    /// would drop it.
    fn worse(a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            f64::INFINITY
        } else {
            a.max(b)
        }
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32).
    fn quant_gap(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel(&o, &i, |j| f64::from(x[j]))
    }

    /// The RMS-normed rows of `x` (`k` values each) times `gain`, in f64
    /// then rounded: the input a norm+quant launch quantizes, near enough
    /// for its gap.
    fn normed(x: &[f32], gain: &[f32], k: usize) -> Vec<f32> {
        x.chunks(k)
            .flat_map(|r| {
                let ms = r.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / k as f64;
                let s = 1.0 / (ms + EPS).sqrt();
                r.iter()
                    .zip(gain)
                    .map(move |(&v, &g)| (f64::from(v) * s * f64::from(g)) as f32)
            })
            .collect()
    }

    /// The first index of the largest value, as the head's argmax breaks ties.
    fn argmax(v: &[f32]) -> u32 {
        let best = v
            .iter()
            .enumerate()
            .fold(0usize, |b, (i, &x)| if x > v[b] { i } else { b });
        best as u32
    }

    /// A set's tap `name` in its logical order.
    fn tap(man: &RefManifest, name: &str) -> Result<Vec<f32>, GateError> {
        Ok(ref_tensor_logical_in(&man.dir, man.tensor(name, 0)?)?)
    }

    /// The node a set dumps as `name`, or when that row only relabels
    /// another (a permute, reshape, view or copy), the node it relabels,
    /// through its `src0`, with that row's `ne`: the step sets name the
    /// permute of the L2 norm's output and leave the output itself unnamed,
    /// the batch set names the output.
    fn tap_or_src(man: &RefManifest, name: &str) -> Result<(Vec<f32>, [u64; 4]), GateError> {
        const RELABEL: [&str; 5] = ["PERMUTE", "RESHAPE", "VIEW", "CONT", "TRANSPOSE"];
        let (mut at, mut row) = man.tensor_at(name, 0)?;
        for _ in 0..4 {
            if !RELABEL.contains(&row.op.as_str()) {
                return Ok((ref_tensor_logical_in(&man.dir, row)?, row.ne));
            }
            (at, row) = man.last_before(at, row.src0.as_deref())?;
        }
        Err(format!("{name}: four relabelling rows and no node under them").into())
    }

    /// The last `n` rows of `k` values of `v`, the rows a tap kept (the
    /// last layer keeps only the output token's rows past its attention):
    /// `None` when `v` holds fewer.
    fn last_rows(v: &[f32], k: usize, n: usize) -> Option<&[f32]> {
        v.len().checked_sub(n * k).map(|s| &v[s..])
    }

    /// Worst ratio and its tap, over a layer's taps.
    #[derive(Default)]
    struct Worst {
        ratio: f64,
        tap: String,
        lines: Vec<String>,
    }

    impl Worst {
        fn add(&mut self, tap: &str, e: f64, gap: f64) {
            let r = worse(0.0, e / gap.max(f64::MIN_POSITIVE));
            self.lines.push(format!("{tap} rel={e:.3e} ratio={r:.2}"));
            if self.tap.is_empty() || r > self.ratio {
                self.ratio = r;
                self.tap = tap.to_string();
            }
        }
    }

    fn open(ctx: usize, kv: KvQ8) -> Result<Qwen35moeModel, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        if file.architecture() != Some("qwen35moe") {
            return Err(format!("{MODEL} is {:?}, not qwen35moe", file.architecture()).into());
        }
        let t = Instant::now();
        let m = Qwen35moeModel::open(
            Gpu::new()?,
            file,
            Open35 {
                ctx,
                mma: true,
                ubatch: U_GATE,
                kv,
            },
        )?;
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={} ubatch={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            m.body("gate_qwen35moe_e2e")?.flash_mma(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    // ------------------------------------------------------ (s) structure

    /// The flash launches' grids in a captured step (module doc, (s) and
    /// (k)): the eight-head `_256` entries, one pack a key head, at one row,
    /// one segment pass and one merge an attention layer ([`flash_grids`]).
    fn flash_grid_ok(nodes: &[NodeInfo]) -> Result<(bool, String), GateError> {
        let names = ["gqa_flash_seg_mma_256", "gqa_flash_merge_256"];
        flash_grids(nodes, (1, N_KV, N_HEAD), N_ATTN, names)
    }

    fn structure(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let body = m.body("structure")?;
        let kinds = body.kinds();
        let n_attn = kinds
            .iter()
            .filter(|k| **k == LayerKind35::Attention)
            .count();
        let (launch_1, launch_m) = (body.pass_launches(1), body.pass_launches(2));
        let (stores, want_stores) = (body.store_bytes(), store_bytes());
        let mut ok = n_attn == N_ATTN && kinds.len() == N_LAYER && stores == want_stores;
        let attn_at: Vec<usize> = (0..kinds.len())
            .filter(|&l| kinds[l] == LayerKind35::Attention)
            .collect();
        println!(
            "structure layers={} attention at {attn_at:?} ({n_attn}, want {N_ATTN}); store bytes \
             {stores} (want {want_stores}, derived) {}",
            kinds.len(),
            verdict(ok)
        );
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memcpy = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY;
        let nodes = m.capture_step()?;
        let ([k, c], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memcpy]);
        let pass = nodes == NODES_DECODE && k == nodes && launch_1 + 3 == nodes;
        println!(
            "structure decode graph_nodes={nodes} (want {NODES_DECODE}; the body counts {} + 3) \
             kernel={k} memcpy={c} other={other} {}",
            launch_1,
            verdict(pass)
        );
        ok &= pass;
        let (flash_ok, got) = flash_grid_ok(&m.step_graph_nodes()?)?;
        println!("structure flash grid {got} {}", verdict(flash_ok));
        ok &= flash_ok;
        for (rows, nodes, list) in [
            (5usize, m.capture_rows::<5>()?, m.rows_graph_nodes::<5>()?),
            (8, m.capture_rows::<8>()?, m.rows_graph_nodes::<8>()?),
        ] {
            let ([k, c], other) = count_kinds(&list, [kernel, memcpy]);
            let want = NODES_PASS + 4 * rows;
            let pass =
                nodes == want && c == rows && k == nodes - rows && launch_m + 4 * rows == nodes;
            println!(
                "structure pass m={rows} graph_nodes={nodes} (want {want}; the body counts {} + 4·{rows}) \
                 kernel={k} memcpy={c} (want {rows}) other={other} {}",
                launch_m,
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// The file's attention and delta shapes, from its spec, against the
    /// constants the derivations above are written with: a file of another
    /// shape fails here by name rather than under a slicing written for this
    /// one.
    fn file_shape() -> Result<bool, GateError> {
        let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let spec = model::arch::qwen35moe::spec::read(&split)
            .map_err(|e| format!("{MODEL}: {e}"))?
            .spec;
        let c = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
        let (mut ok, mut seen) = (true, Vec::new());
        for (l, layer) in spec.layers.iter().enumerate() {
            let (got, want) = match &layer.mixer {
                Mixer::Gqa(g) => {
                    let a = AttnShape::of(g);
                    let got = [a.n_head, a.n_kv, a.head, g.rope.dims];
                    (("gqa", got), [c(N_HEAD), c(N_KV), c(HEAD), c(ROT)])
                }
                Mixer::DeltaRule(d) => {
                    let got = [d.v_heads, d.d, (2 * d.k_heads + d.v_heads) * d.d, 0];
                    (("delta", got), [c(N_V), c(HEAD_V), c(C), 0])
                }
                Mixer::Latent(_) => (("latent", [0; 4]), [u32::MAX; 4]),
            };
            let pass = got.1 == want;
            ok &= pass;
            if !pass || !seen.contains(&got) {
                println!(
                    "shape layer={l} {} {:?} (the derivations' {want:?}) {}",
                    got.0,
                    got.1,
                    verdict(pass)
                );
                seen.push(got);
            }
        }
        let rows_ok = Q_ROWS == 2 * N_HEAD * HEAD && KV_ROW == N_KV * HEAD;
        ok &= rows_ok;
        println!(
            "shape q rows {Q_ROWS} = 2·{N_HEAD}·{HEAD}, kv row {KV_ROW} = {N_KV}·{HEAD} {}",
            verdict(rows_ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (p) one layer body

    /// Every layer's store, read back.
    fn stores(m: &mut Qwen35moeModel) -> Result<Vec<StoreHost>, GateError> {
        let n = m.layers().len();
        let (gpu, _, body) = m.body_parts("stores")?;
        Ok((0..n)
            .map(|l| body.store(gpu, l))
            .collect::<Result<_, GpuError>>()?)
    }

    fn same_store(a: &StoreHost, b: &StoreHost) -> bool {
        match (a, b) {
            (StoreHost::Kv { k: ka, v: va }, StoreHost::Kv { k: kb, v: vb }) => {
                ka == kb && va == vb
            }
            (
                StoreHost::Rec {
                    state: sa,
                    ring: ra,
                },
                StoreHost::Rec {
                    state: sb,
                    ring: rb,
                },
            ) => bits_equal(sa, sb) && bits_equal(ra, rb),
            _ => false,
        }
    }

    /// What a run of the batch tokens left: each position's logits and
    /// token, and every layer's store.
    struct PathRun {
        logits: Vec<Vec<f32>>,
        tokens: Vec<u32>,
        stores: Vec<StoreHost>,
    }

    /// `reset`, then every layer's store written to zeros through
    /// `set_store`: a start from zero state that does not rest on `reset`'s
    /// own clearing, so only the clause that tests `reset` reads it.
    fn fresh(m: &mut Qwen35moeModel) -> Result<(), GateError> {
        m.reset()?;
        let n = m.layers().len();
        let (gpu, _, body) = m.body_parts("fresh")?;
        for l in 0..n {
            let zero = match body.store(gpu, l)? {
                StoreHost::Kv { k, v } => StoreHost::Kv {
                    k: vec![0; k.len()],
                    v: vec![0; v.len()],
                },
                StoreHost::Rec { state, ring } => StoreHost::Rec {
                    state: vec![0.0; state.len()],
                    ring: vec![0.0; ring.len()],
                },
            };
            body.set_store(gpu, l, &zero)?;
        }
        Ok(())
    }

    /// The batch tokens as decode steps from position 0: from [`fresh`]'s
    /// zero stores, or (`by_reset`) from `reset` alone.
    fn decode_run(
        m: &mut Qwen35moeModel,
        toks: &[u32],
        by_reset: bool,
    ) -> Result<PathRun, GateError> {
        if by_reset {
            m.reset()?;
        } else {
            fresh(m)?;
        }
        let (mut logits, mut tokens) = (Vec::new(), Vec::new());
        for &t in toks {
            tokens.push(m.step(&[t])?);
            logits.push(m.logits()?);
        }
        Ok(PathRun {
            logits,
            tokens,
            stores: stores(m)?,
        })
    }

    fn same_run(label: &str, got: &PathRun, want: &PathRun) -> bool {
        let logits = got.logits.len() == want.logits.len()
            && got
                .logits
                .iter()
                .zip(&want.logits)
                .all(|(a, b)| bits_equal(a, b));
        let stores_same = got.stores.len() == want.stores.len()
            && got
                .stores
                .iter()
                .zip(&want.stores)
                .all(|(a, b)| same_store(a, b));
        let differ: Vec<usize> = (0..got.stores.len().min(want.stores.len()))
            .filter(|&l| !same_store(&got.stores[l], &want.stores[l]))
            .collect();
        let ok = logits && stores_same && got.tokens == want.tokens;
        println!(
            "paths {label}: tokens {:?} vs {:?}, every position's logits bit-identical={logits}, \
             every store bit-identical={stores_same} (layers differing {differ:?}) {}",
            got.tokens,
            want.tokens,
            verdict(ok)
        );
        ok
    }

    fn paths(m: &mut Qwen35moeModel, toks: &[u32]) -> Result<(bool, PathRun), GateError> {
        let toks5: [u32; 5] = toks
            .try_into()
            .map_err(|_| format!("the batch set holds {} tokens, want 5", toks.len()))?;
        m.set_mode(StepMode::Graph);
        let decode = decode_run(m, toks, false)?;
        // One pass of five rows.
        fresh(m)?;
        let tokens = m.step_rows::<5>(toks5)?.to_vec();
        let pass = PathRun {
            logits: m.rows_logits::<5>()?.to_vec(),
            tokens,
            stores: stores(m)?,
        };
        let mut ok = same_run("pass m=5 vs five decode steps (graph)", &pass, &decode);
        m.set_mode(StepMode::Eager);
        let eager = decode_run(m, toks, false)?;
        ok &= same_run("five eager steps vs five graph replays", &eager, &decode);
        m.set_mode(StepMode::Graph);
        // The only run that starts from `reset` alone, after the eager run
        // left its state: `reset` must clear it.
        let again = decode_run(m, toks, true)?;
        ok &= same_run("after reset alone, the same five steps", &again, &decode);
        Ok((ok, decode))
    }

    // --------------------------------------------------- (c) free-running

    fn free(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        toks: &[u32],
        decode: &PathRun,
    ) -> Result<bool, GateError> {
        let n = m.layers().len();
        let ik_out: Vec<Vec<f32>> = (0..n)
            .map(|l| tap(man, &format!("l_out-{l}")))
            .collect::<Result<_, _>>()?;
        let ik_logits = tap(man, "result_output")?;
        let vocab = m.body("free")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output holds no row")?;
        m.set_layer_taps(true)?;
        fresh(m)?;
        let t_n = toks.len();
        let mut per_layer = vec![0.0f64; n];
        for (t, &tok) in toks.iter().enumerate() {
            m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for ((worst, ik), got) in per_layer.iter_mut().zip(&ik_out).zip(&taps) {
                let kept = ik.len() / HIDDEN;
                let Some(i) = (t + kept).checked_sub(t_n) else {
                    continue;
                };
                *worst = worse(*worst, rel_to(got, &ik[i * HIDDEN..(i + 1) * HIDDEN]));
            }
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ours = decode.logits.last().ok_or("no decode logits")?;
        let (top, ik_top) = (argmax(ours), argmax(ik_last));
        let worst = per_layer.iter().copied().fold(0.0f64, worse);
        for (l, &e) in per_layer.iter().enumerate() {
            if l % 8 == 0 || l == n - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        let ok = worst <= FREE_BAND && top == ik_top;
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); last \
             position argmax ours={top} ik={ik_top} logits_rel={:.3e} (printed) {}",
            rel_to(ours, ik_last),
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------- (f)/(t2) forced taps

    /// The file's gains a gap is computed through.
    struct Gains {
        attn_norm: Vec<Vec<f32>>,
        ffn_norm: Vec<Vec<f32>>,
        /// Each layer's routed mixture as the file's spec states it: the
        /// taps slice the router's rows and slots by it, the engine's source.
        moe: Vec<MoeShape>,
    }

    impl Gains {
        fn attn(&self, l: usize) -> Result<&[f32], GateError> {
            Ok(self
                .attn_norm
                .get(l)
                .map(Vec::as_slice)
                .ok_or("a layer past the file's gains")?)
        }

        fn read(n: usize) -> Result<Gains, GateError> {
            let split = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
            let each = |stem: &str| -> Result<Vec<Vec<f32>>, GateError> {
                (0..n)
                    .map(|l| split_f32(&split, &format!("blk.{l}.{stem}"), HIDDEN))
                    .collect()
            };
            let spec = model::arch::qwen35moe::spec::read(&split)
                .map_err(|e| format!("{MODEL}: {e}"))?
                .spec;
            let moe = spec
                .layers
                .iter()
                .take(n)
                .enumerate()
                .map(|(l, s)| {
                    s.moe()
                        .map(MoeShape::of)
                        .ok_or_else(|| format!("layer {l}: no routed FFN in the file's spec"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if moe.len() != n {
                return Err(format!("the file's spec has {} layers, want {n}", moe.len()).into());
            }
            Ok(Gains {
                attn_norm: each("attn_norm.weight")?,
                ffn_norm: each("post_attention_norm.weight")?,
                moe,
            })
        }
    }

    /// Our `[t][h][d]` values of `h_n` heads of `d_n` against ik's `ik`
    /// laid out by `ne`: `[t][h][d]` (`ne = [d, h, t]`) or `[h][t][d]` (`ne
    /// = [d, t, h]`, a permuted norm's output), each ik value times `scale`.
    fn heads_rel(
        ours: &[f32],
        ik: &[f32],
        ne: [u64; 4],
        (t_n, h_n, d_n): (usize, usize, usize),
        scale: f32,
    ) -> Result<f64, GateError> {
        let ne = [ne[0] as usize, ne[1] as usize, ne[2] as usize];
        let head_major = ne == [d_n, t_n, h_n] && t_n > 1;
        if !(head_major || ne == [d_n, h_n, t_n] || ik.len() == t_n * h_n * d_n) {
            return Err(format!("a head tap of ne {ne:?} for {t_n} x {h_n} x {d_n}").into());
        }
        let mut want = vec![0.0f32; t_n * h_n * d_n];
        for t in 0..t_n {
            for h in 0..h_n {
                for d in 0..d_n {
                    let src = if head_major {
                        (h * t_n + t) * d_n + d
                    } else {
                        (t * h_n + h) * d_n + d
                    };
                    want[(t * h_n + h) * d_n + d] = ik[src] * scale;
                }
            }
        }
        Ok(rel_to(ours, &want))
    }

    /// Values `lo..hi` of every `width`-value head of `v`.
    fn head_part(v: &[f32], width: usize, lo: usize, hi: usize) -> Vec<f32> {
        v.chunks(width).flat_map(|h| h[lo..hi].to_vec()).collect()
    }

    /// A delta layer's taps against ik's: the worst ratio; the decay's
    /// underflow sites must be ik's.
    #[allow(
        clippy::too_many_arguments,
        reason = "a layer's run, its inputs' gaps, its store and ik's set, as the tap map reads them"
    )]
    fn delta_taps(
        man: &RefManifest,
        l: usize,
        r: &Delta35Run,
        store: &StoreHost,
        (t_n, pos_end): (usize, usize),
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<bool, GateError> {
        let gap_y = quant_gap(&r.y, N_V * HEAD_V);
        w.add(
            "qkv_mixed",
            rel_to(&r.x, &tap(man, &format!("qkv_mixed-{l}"))?),
            gap_in,
        );
        w.add("z", rel_to(&r.z, &tap(man, &format!("z-{l}"))?), gap_in);
        w.add(
            "beta_in",
            rel_to(&r.b, &tap(man, &format!("beta_in-{l}"))?),
            gap_in,
        );
        w.add(
            "alpha",
            rel_to(&r.a, &tap(man, &format!("alpha-{l}"))?),
            gap_in,
        );
        // ik dumps g, not exp(g) (the delta op takes g): our g is the log of
        // our decay where it did not underflow, and the underflow sites are
        // those where ik's exp — `expf_ik`, the same algorithm — gives 0.
        let g = tap(man, &format!("g_in-{l}"))?;
        let ik_decay: Vec<f32> = g.iter().map(|&x| expf_ik(x)).collect();
        let under = |v: &[f32]| -> Vec<usize> { (0..v.len()).filter(|&i| v[i] == 0.0).collect() };
        let (ours_u, ik_u) = (under(&r.decay), under(&ik_decay));
        let live: Vec<usize> = (0..g.len().min(r.decay.len()))
            .filter(|&i| r.decay[i] > 0.0)
            .collect();
        let g_ours: Vec<f32> = live
            .iter()
            .map(|&i| f64::from(r.decay[i]).ln() as f32)
            .collect();
        let g_ik: Vec<f32> = live.iter().map(|&i| g[i]).collect();
        w.add(
            "g_in (the log of our decay)",
            rel_to(&g_ours, &g_ik),
            gap_in,
        );
        let silu = tap(man, &format!("conv_output_silu-{l}"))?;
        let v_of =
            |x: &[f32]| -> Vec<f32> { x.chunks(C).flat_map(|t| t[C / 2..].to_vec()).collect() };
        w.add(
            "conv_output_silu (v)",
            rel_to(&v_of(&r.conv), &v_of(&silu)),
            gap_in,
        );
        let (q, q_ne) = tap_or_src(man, &format!("q_fused-{l}"))?;
        let (k, k_ne) = tap_or_src(man, &format!("k_fused-{l}"))?;
        let q_ours: Vec<f32> = r.conv.chunks(C).flat_map(|t| t[..C / 4].to_vec()).collect();
        let k_ours: Vec<f32> = r
            .conv
            .chunks(C)
            .flat_map(|t| t[C / 4..C / 2].to_vec())
            .collect();
        w.add(
            "q_fused",
            heads_rel(&q_ours, &q, q_ne, (t_n, 16, HEAD_V), Q_SCALE)?,
            gap_in,
        );
        w.add(
            "k_fused",
            heads_rel(&k_ours, &k, k_ne, (t_n, 16, HEAD_V), 1.0)?,
            gap_in,
        );
        w.add(
            "attn_output",
            rel_to(&r.o, &tap(man, &format!("attn_output-{l}"))?),
            gap_in,
        );
        w.add(
            "attn_out_norm",
            rel_to(&r.y, &tap(man, &format!("attn_out_norm-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let ik_upd = tap(man, &format!("linear_attn_out-{l}"))?;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        w.add(
            "linear_attn_out",
            rel_to(&got, &ik_upd),
            gap_in.hypot(gap_y),
        );
        let StoreHost::Rec { state, ring } = store else {
            return Err(format!("layer {l} is a delta layer with another store").into());
        };
        let ik_state = tap(man, &format!("new_state-{l}"))?;
        let mut t = vec![0.0f32; state.len()];
        for h in 0..N_V {
            for kk in 0..HEAD_V {
                for v in 0..HEAD_V {
                    t[(h * HEAD_V + v) * HEAD_V + kk] = ik_state[(h * HEAD_V + kk) * HEAD_V + v];
                }
            }
        }
        w.add("new_state (transposed)", rel_to(state, &t), gap_in);
        let ik_conv = tap(man, &format!("new_conv_states_cont-{l}"))?;
        let mut ours_conv = vec![0.0f32; 3 * C];
        for j in 0..3 {
            let p = pos_end - 3 + j;
            let slot = p % RING_ROWS;
            for ch in 0..C {
                ours_conv[ch * 3 + j] = ring[slot * C + ch];
            }
        }
        w.add("new_conv_states_cont", rel_to(&ours_conv, &ik_conv), gap_in);
        let ok = ours_u == ik_u;
        if !ours_u.is_empty() || !ik_u.is_empty() {
            println!(
                "forced layer={l} decay underflow sites ours {ours_u:?} ik {ik_u:?} {}",
                verdict(ok)
            );
        }
        Ok(ok)
    }

    /// An attention layer's taps against ik's.
    fn attn_taps(
        man: &RefManifest,
        l: usize,
        r: &Gqa35Run,
        t_n: usize,
        gap_in: f64,
        upd: (&[f32], &[f32]),
        w: &mut Worst,
    ) -> Result<(), GateError> {
        w.add(
            "Qaux",
            rel_to(&r.qg, &tap(man, &format!("Qaux-{l}"))?),
            gap_in,
        );
        let q_normed = tap(man, &format!("Qcur_normed-{l}"))?;
        let q_roped = tap(man, &format!("Qcur_roped-{l}"))?;
        let k_normed = tap(man, &format!("Kcur_normed-{l}"))?;
        let k_roped = tap(man, &format!("Kcur_roped-{l}"))?;
        for (name, ours, ik, lo, hi) in [
            ("Qcur_normed (past the turn)", &r.q, &q_normed, ROT, HEAD),
            ("Qcur_roped (the turn)", &r.q, &q_roped, 0, ROT),
            ("Kcur_normed (past the turn)", &r.k, &k_normed, ROT, HEAD),
            ("Kcur_roped (the turn)", &r.k, &k_roped, 0, ROT),
        ] {
            w.add(
                name,
                rel_to(&head_part(ours, HEAD, lo, hi), &head_part(ik, HEAD, lo, hi)),
                gap_in,
            );
        }
        w.add(
            "Vcur",
            rel_to(&r.v, &tap(man, &format!("Vcur-{l}"))?),
            gap_in,
        );
        w.add("fa", rel_to(&r.fa, &tap(man, &format!("fa-{l}"))?), gap_in);
        let gated: Vec<f32> = (0..t_n * N_HEAD * HEAD)
            .map(|i| {
                let (t, h, d) = (i / (N_HEAD * HEAD), (i / HEAD) % N_HEAD, i % HEAD);
                r.fa[i] * sigmoid(r.qg[t * Q_ROWS + h * 2 * HEAD + HEAD + d])
            })
            .collect();
        let gap_g = quant_gap(&gated, N_HEAD * HEAD);
        w.add(
            "qkv_gated (host product)",
            rel_to(&gated, &tap(man, &format!("qkv_gated-{l}"))?),
            gap_in,
        );
        let (ours_upd, x_in) = upd;
        let got: Vec<f32> = ours_upd.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
        let ik_upd = tap(man, &format!("attn_out-{l}"))?;
        w.add("attn_out", rel_to(&got, &ik_upd), gap_in.hypot(gap_g));
        Ok(())
    }

    /// The FFN half on ik's own FFN input against ik's MoE taps, over the
    /// rows ik kept (`ffn_in` may hold more: the last layer keeps only the
    /// output token's rows past its attention): the router's 257 logits
    /// (`ffn_moe_logits` and `shared_expert_gate`), the ids as sets — a
    /// token whose sets differ is a flip, allowed only where ik's own margin
    /// between its eighth pick and the best expert it left lies inside
    /// twice our logits' error at that token, counted and printed — the
    /// weights matched by id with the shared expert's (ik's sigmoid tap, or
    /// the sigmoid of ik's logit where a one-token graph fused it away), and
    /// the layer output's update on the tokens without a flip.
    fn moe_taps(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        l: usize,
        ffn_in: &[f32],
        gains: &Gains,
        w: &mut Worst,
    ) -> Result<(bool, usize), GateError> {
        let shape = gains.moe[l];
        if !shape.rule.gated {
            return Err(format!("layer {l}: {shape:?} has no gated shared expert").into());
        }
        let (n_expert, n_used) = (shape.experts as usize, shape.top_k as usize);
        let slots = n_used + 1;
        let logits = tap(man, &format!("ffn_moe_logits-{l}"))?;
        let gate = tap(man, &format!("shared_expert_gate-{l}"))?;
        let t_n = gate.len();
        if t_n == 0 || logits.len() != t_n * n_expert {
            return Err(format!(
                "layer {l}: {} router logits for {t_n} gate logits",
                logits.len()
            )
            .into());
        }
        let ffn_in = last_rows(ffn_in, HIDDEN, t_n)
            .ok_or_else(|| format!("layer {l}: ik routes {t_n} rows, the input holds fewer"))?;
        let f = m.ffn_rows(l, ffn_in)?;
        let gap = quant_gap(&normed(ffn_in, &gains.ffn_norm[l], HIDDEN), HIDDEN);
        let ik_logits: Vec<f32> = (0..t_n)
            .flat_map(|t| {
                let mut row = logits[t * n_expert..(t + 1) * n_expert].to_vec();
                row.push(gate[t]);
                row
            })
            .collect();
        w.add(
            "ffn_moe_logits + shared_expert_gate",
            rel_to(&f.logits, &ik_logits),
            gap,
        );
        let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
        let ids = topk_ids_logical_within(man, trow, shape.experts)?;
        let wn = tap(man, &format!("ffn_moe_weights_norm-{l}"))?;
        if ids.len() != t_n * n_used || wn.len() != t_n * n_used {
            return Err(format!(
                "layer {l}: {} ids and {} weights for {t_n} tokens",
                ids.len(),
                wn.len()
            )
            .into());
        }
        let ik_sg: Vec<f32> = match man.tensor(&format!("shared_expert_gate_sigmoid-{l}"), 0) {
            Ok(row) => ref_tensor_logical_in(&man.dir, row)?,
            Err(_) => gate.iter().map(|&g| sigmoid(g)).collect(),
        };
        if ik_sg.len() != t_n {
            return Err(format!("layer {l}: {} shared gates for {t_n} tokens", ik_sg.len()).into());
        }
        let (mut ok, mut flipped) = (true, Vec::new());
        let (mut w_ours, mut w_ik) = (Vec::new(), Vec::new());
        let row = n_expert + 1;
        for (t, &sg) in ik_sg.iter().enumerate() {
            let ours = &f.ids[t * slots..t * slots + n_used];
            let theirs = &ids[t * n_used..(t + 1) * n_used];
            if !ours.iter().all(|&e| theirs.contains(&(e as i32))) {
                let lg = &ik_logits[t * row..t * row + n_expert];
                let min_in = theirs
                    .iter()
                    .map(|&e| lg[e as usize])
                    .fold(f32::INFINITY, f32::min);
                let max_out = (0..n_expert)
                    .filter(|&e| !theirs.contains(&(e as i32)))
                    .map(|e| lg[e])
                    .fold(f32::NEG_INFINITY, f32::max);
                let err = f.logits[t * row..t * row + n_expert]
                    .iter()
                    .zip(lg)
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let tie = min_in - max_out <= 2.0 * err;
                ok &= tie;
                flipped.push(t);
                println!(
                    "forced layer={l} token={t}: ids ours {ours:?} ik {theirs:?}, ik margin {:.3e}, \
                     our logits' error {err:.3e}: {}",
                    min_in - max_out,
                    if tie { "a near tie (counted)" } else { "FAIL" }
                );
                continue;
            }
            for (s, &e) in ours.iter().enumerate() {
                let j = theirs
                    .iter()
                    .position(|&x| x == e as i32)
                    .ok_or("an id the set test found")?;
                w_ours.push(f.weights[t * slots + s]);
                w_ik.push(wn[t * n_used + j]);
            }
            w_ours.push(f.weights[t * slots + n_used]);
            w_ik.push(sg);
        }
        if !w_ik.is_empty() {
            w.add(
                "ffn_moe_weights_norm + shared_expert_gate_sigmoid",
                rel_to(&w_ours, &w_ik),
                gap,
            );
        }
        let out = tap(man, &format!("l_out-{l}"))?;
        let ik_out = last_rows(&out, HIDDEN, t_n).ok_or("l_out rows")?;
        let (mut got, mut want) = (Vec::new(), Vec::new());
        for t in (0..t_n).filter(|t| !flipped.contains(t)) {
            let r = t * HIDDEN..(t + 1) * HIDDEN;
            got.extend(
                f.l_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r.clone()])
                    .map(|(&o, &i)| o - i),
            );
            want.extend(
                ik_out[r.clone()]
                    .iter()
                    .zip(&ffn_in[r])
                    .map(|(&o, &i)| o - i),
            );
        }
        if !want.is_empty() {
            w.add("l_out (update)", rel_to(&got, &want), gap);
        }
        Ok((ok, flipped.len()))
    }

    /// Teacher-forced taps on every layer of `man` at `t_n` rows from
    /// position `pos`, each layer's store first set by `load` (a reset's
    /// zero store, or the set's prefill state).
    fn forced(
        m: &mut Qwen35moeModel,
        man: &RefManifest,
        label: &str,
        (pos, t_n): (u32, usize),
        gains: &Gains,
        load: &dyn Fn(&mut Qwen35moeModel, usize) -> Result<(), GateError>,
    ) -> Result<bool, GateError> {
        let n = m.layers().len();
        let kinds = m.body("forced")?.kinds();
        let embd = tap(man, "inp_embd")?;
        let mut ok = true;
        let (mut worst, mut flips_all) = (0.0f64, 0usize);
        for (l, &kind) in kinds.iter().enumerate() {
            let x_all = if l == 0 {
                embd.clone()
            } else {
                tap(man, &format!("l_out-{}", l - 1))?
            };
            let x_in = last_rows(&x_all, HIDDEN, t_n).ok_or("a layer input row is missing")?;
            load(m, l)?;
            let run = m.layer_rows(l, x_in, pos)?;
            let gap_in = quant_gap(&normed(x_in, gains.attn(l)?, HIDDEN), HIDDEN);
            let mut w = Worst::default();
            let body_ok = match (&run.mixer, kind) {
                (Mixer35Run::Delta(r), LayerKind35::Delta) => {
                    let store = {
                        let (gpu, _, body) = m.body_parts("forced")?;
                        body.store(gpu, l)?
                    };
                    delta_taps(
                        man,
                        l,
                        r,
                        &store,
                        (t_n, pos as usize + t_n),
                        gap_in,
                        (&run.ffn_inp, x_in),
                        &mut w,
                    )?
                }
                (Mixer35Run::Attention(r), LayerKind35::Attention) => {
                    attn_taps(man, l, r, t_n, gap_in, (&run.ffn_inp, x_in), &mut w)?;
                    true
                }
                _ => return Err(format!("layer {l}: a run of another kind than the layer").into()),
            };
            // The FFN half alone on ik's FFN input: the layer input plus
            // ik's mixer update, over the rows ik kept.
            let upd_name = match kind {
                LayerKind35::Delta => format!("linear_attn_out-{l}"),
                LayerKind35::Attention => format!("attn_out-{l}"),
            };
            let ik_upd = tap(man, &upd_name)?;
            let kept = ik_upd.len() / HIDDEN;
            let x_kept = last_rows(x_in, HIDDEN, kept).ok_or("rows")?;
            let ffn_in: Vec<f32> = ik_upd.iter().zip(x_kept).map(|(&a, &x)| a + x).collect();
            let (moe_ok, flips) = moe_taps(m, man, l, &ffn_in, gains, &mut w)?;
            let pass = body_ok && moe_ok && w.ratio <= RATIO_BAND;
            println!(
                "forced {label} layer={l} {:?}: worst ratio {:.2} at {} (band {RATIO_BAND}); \
                 flip sites {flips} {}",
                kind,
                w.ratio,
                w.tap,
                verdict(pass)
            );
            if !pass || l % 8 == 0 {
                for line in &w.lines {
                    println!("  forced {label} layer={l} {line}");
                }
            }
            worst = worse(worst, w.ratio);
            flips_all += flips;
            ok &= pass;
        }
        println!(
            "forced {label}: {n} layers x {t_n} rows from position {pos}; worst ratio {worst:.2} \
             (band {RATIO_BAND}); flip sites {flips_all} (near ties, counted) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (t) the step sets

    /// Layer `l`'s store as a step set's prefill left it: a delta layer's
    /// state (ik's `[k][v]` per head, transposed into ours) and conv ring
    /// (ik's `[C][3]` window of the three positions before `p`, into slots
    /// `(p − 3 + j) mod RING_ROWS`), or an attention layer's K and V rows
    /// `0..p`. ik lays a plane out as rows of 512 in three shapes: `[512,
    /// cells]`, K's `[256, 2·cells]` (a cell's two heads adjacent) and, with
    /// flash attention, V's flat `[512·cells]`; without flash attention V is
    /// transposed, `[cells, 512]`. A flat plane from a set with no
    /// `FLASH_ATTN_EXT` at the layer is refused by name.
    fn prefill_store(
        man: &RefManifest,
        kind: LayerKind35,
        l: usize,
        p: usize,
    ) -> Result<StoreHost, GateError> {
        match kind {
            LayerKind35::Delta => {
                let s = ref_tensor_logical_in(&man.dir, man.input(&format!("cache_s_l{l}"), 0)?)?;
                let conv_len = 3 * C;
                if s.len() != conv_len + N_V * HEAD_V * HEAD_V {
                    return Err(format!("cache_s_l{l} holds {} values", s.len()).into());
                }
                let (conv, ssm) = s.split_at(conv_len);
                let mut state = vec![0.0f32; ssm.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            state[(h * HEAD_V + v) * HEAD_V + k] =
                                ssm[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let mut ring = vec![0.0f32; RING_ROWS * C];
                for j in 0..3 {
                    let Some(pos) = (p + j).checked_sub(3) else {
                        continue;
                    };
                    let slot = pos % RING_ROWS;
                    for ch in 0..C {
                        ring[slot * C + ch] = conv[ch * 3 + j];
                    }
                }
                Ok(StoreHost::Rec { state, ring })
            }
            LayerKind35::Attention => {
                let flash = man
                    .tensor(&format!("fa-{l}"), 0)
                    .is_ok_and(|t| t.op == "FLASH_ATTN_EXT");
                let plane = |name: String| -> Result<Vec<u16>, GateError> {
                    let row = man.input(&name, 0)?;
                    let bits = widened_f16_rows_in(&man.dir, row)?;
                    let (ne0, ne1) = (row.ne[0] as usize, row.ne[1] as usize);
                    let flat = ne1 == 1 && ne0.is_multiple_of(KV_ROW);
                    let rows = (ne0 == KV_ROW && ne1 >= p)
                        || (ne0 == HEAD && ne1.is_multiple_of(N_KV) && ne1 / N_KV >= p)
                        || (flat && flash && ne0 / KV_ROW >= p);
                    let transposed = ne1 == KV_ROW && ne0 >= p;
                    if !rows && !transposed {
                        return Err(format!(
                            "{name} is {:?} (flash attention at the layer: {flash}), \
                             not rows of {KV_ROW} holding {p} positions",
                            row.ne
                        )
                        .into());
                    }
                    let mut out = vec![0u16; N_KV * CTX * HEAD];
                    for r in 0..p {
                        for h in 0..N_KV {
                            for d in 0..HEAD {
                                let c = h * HEAD + d;
                                let src = if rows { r * KV_ROW + c } else { c * ne0 + r };
                                out[(h * CTX + r) * HEAD + d] = bits[src];
                            }
                        }
                    }
                    Ok(out)
                };
                Ok(StoreHost::Kv {
                    k: plane(format!("cache_k_l{l}"))?,
                    v: plane(format!("cache_v_l{l}"))?,
                })
            }
        }
    }

    fn step_set(m: &mut Qwen35moeModel, name: &str, gains: &Gains) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(name), &IK)?;
        let (p, tail, before) = man.step()?;
        let token = *tail.first().ok_or("the step set holds no token")?;
        let pu = p as usize;
        let kinds = m.body("step_set")?.kinds();
        let loaded: Vec<StoreHost> = kinds
            .iter()
            .enumerate()
            .map(|(l, &k)| prefill_store(&man, k, l, pu))
            .collect::<Result<_, _>>()?;
        println!(
            "step {name}: position {p}, token {token} after {} tokens; every layer's store loaded \
             from the set's inputs",
            before.len()
        );
        let load_all = |m: &mut Qwen35moeModel| -> Result<(), GateError> {
            fresh(m)?;
            m.seed_depth(pu)?;
            let (gpu, _, body) = m.body_parts("step_set")?;
            for (l, s) in loaded.iter().enumerate() {
                body.set_store(gpu, l, s)?;
            }
            Ok(())
        };
        // (t1) the step, free-running from the loaded state.
        load_all(m)?;
        m.set_layer_taps(true)?;
        m.set_mode(StepMode::Eager);
        let top = m.step(&[token])?;
        let taps = m.layer_taps()?;
        let logits = m.logits()?;
        let after = stores(m)?;
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        let ik_logits = tap(&man, "result_output")?;
        let vocab = m.body("step_set")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output")?;
        let ik_top = argmax(ik_last);
        let (mut worst, mut worst_state) = (0.0f64, 0.0f64);
        for (l, got) in taps.iter().enumerate() {
            let want = tap(&man, &format!("l_out-{l}"))?;
            let want = last_rows(&want, HIDDEN, 1).ok_or("l_out")?;
            let e = rel_to(got, want);
            worst = worse(worst, e);
            if let (LayerKind35::Delta, StoreHost::Rec { state, .. }) = (kinds[l], &after[l]) {
                let ik_state = tap(&man, &format!("new_state-{l}"))?;
                let mut t = vec![0.0f32; state.len()];
                for h in 0..N_V {
                    for k in 0..HEAD_V {
                        for v in 0..HEAD_V {
                            t[(h * HEAD_V + v) * HEAD_V + k] =
                                ik_state[(h * HEAD_V + k) * HEAD_V + v];
                        }
                    }
                }
                let x_in = if l == 0 {
                    tap(&man, "inp_embd")?
                } else {
                    tap(&man, &format!("l_out-{}", l - 1))?
                };
                let x_in = last_rows(&x_in, HIDDEN, 1).ok_or("input")?;
                let gap = quant_gap(&normed(x_in, &gains.attn_norm[l], HIDDEN), HIDDEN);
                worst_state = worse(worst_state, rel_to(state, &t) / gap.max(f64::MIN_POSITIVE));
            }
            if l % 8 == 0 || l == taps.len() - 1 || e > FREE_BAND {
                println!("step {name} free layer={l} l_out_rel={e:.3e}");
            }
        }
        let mut ok = worst <= FREE_BAND && top == ik_top && worst_state <= RATIO_BAND;
        println!(
            "step {name} free: worst l_out_rel={worst:.3e} (band {FREE_BAND:.2}); worst new_state \
             ratio {worst_state:.2} (band {RATIO_BAND}); argmax ours={top} ik={ik_top} \
             logits_rel={:.3e} (printed) {}",
            rel_to(&logits, ik_last),
            verdict(ok)
        );
        // (t2) each layer teacher-forced at one row on its reloaded store.
        let reload = |m: &mut Qwen35moeModel, l: usize| -> Result<(), GateError> {
            let (gpu, _, body) = m.body_parts("step_set")?;
            body.set_store(gpu, l, &loaded[l])?;
            Ok(())
        };
        load_all(m)?;
        m.set_mode(StepMode::Eager);
        ok &= forced(m, &man, name, (p, 1), gains, &reload)?;
        m.set_mode(StepMode::Graph);
        Ok(ok)
    }

    // ------------------------------------- (p′) the ubatch arena's gemv arm

    fn gemv_arm(m: &mut Qwen35moeModel, toks: &[u32], decode: &PathRun) -> Result<bool, GateError> {
        let (want_tok, want_logits) = decode
            .tokens
            .last()
            .zip(decode.logits.last())
            .ok_or("no decode run")?;
        let mut ok = true;
        for path in [PrefillPath::Pass, PrefillPath::Gemm] {
            fresh(m)?;
            let plan = m.prefill_plan(toks.len(), path)?;
            let tok = m.prefill_with(toks, path)?;
            let logits = m.logits()?;
            let got = stores(m)?;
            let differ: Vec<usize> = (0..got.len())
                .filter(|&l| !decode.stores.get(l).is_some_and(|w| same_store(&got[l], w)))
                .collect();
            let same_logits = bits_equal(&logits, want_logits);
            let pass = tok == *want_tok
                && same_logits
                && differ.is_empty()
                && got.len() == decode.stores.len();
            println!(
                "gemv arm {path:?} plan={plan}: token {tok} vs the fifth step's {want_tok}, last \
                 logits bit-identical={same_logits}, every store bit-identical (layers differing \
                 {differ:?}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ------------------------------------------- (u)/(d) the wide arm

    /// A prompt prefilled from zero stores, and the step after it.
    struct PromptRun {
        next: u32,
        logits: Vec<f32>,
        stores: Vec<StoreHost>,
        step_top: u32,
        step_taps: Vec<Vec<f32>>,
        step_logits: Vec<f32>,
    }

    fn prompt_run(
        m: &mut Qwen35moeModel,
        prompt: &[u32],
        path: PrefillPath,
        token: u32,
    ) -> Result<PromptRun, GateError> {
        fresh(m)?;
        println!(
            "prompt {path:?}: {} ids, plan={}",
            prompt.len(),
            m.prefill_plan(prompt.len(), path)?
        );
        let next = m.prefill_with(prompt, path)?;
        let logits = m.logits()?;
        let stores = stores(m)?;
        m.set_layer_taps(true)?;
        m.set_mode(StepMode::Eager);
        let step_top = m.step(&[token])?;
        let step_taps = m.layer_taps()?;
        let step_logits = m.logits()?;
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        Ok(PromptRun {
            next,
            logits,
            stores,
            step_top,
            step_taps,
            step_logits,
        })
    }

    /// The ring slots of the three positions before `p`, the rows ik's conv
    /// state holds.
    fn ik_slots(p: usize) -> Vec<usize> {
        (1..=3).rev().map(|d| (p - d) % RING_ROWS).collect()
    }

    /// The rows `slots` of a `[RING_ROWS][C]` ring.
    fn ring_rows(ring: &[f32], slots: &[usize]) -> Vec<f32> {
        slots
            .iter()
            .flat_map(|&s| ring[s * C..(s + 1) * C].to_vec())
            .collect()
    }

    /// `(a, b)` of a layer's store against `base`'s, each `rel_to`: the
    /// recurrent state and the ring's rows `slots`, or the K and the V rows.
    fn store_rel(ours: &StoreHost, base: &StoreHost, slots: &[usize]) -> (f64, f64) {
        let f16 = |v: &[u16]| -> Vec<f32> { v.iter().map(|&b| half_to_f32(b)).collect() };
        match (ours, base) {
            (StoreHost::Rec { state, ring }, StoreHost::Rec { state: s, ring: r }) => (
                rel_to(state, s),
                rel_to(&ring_rows(ring, slots), &ring_rows(r, slots)),
            ),
            (StoreHost::Kv { k, v }, StoreHost::Kv { k: kb, v: vb }) => {
                (rel_to(&f16(k), &f16(kb)), rel_to(&f16(v), &f16(vb)))
            }
            _ => (f64::INFINITY, f64::INFINITY),
        }
    }

    fn wide_arm(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (p, tail, prompt) = man.step()?;
        let token = *tail.first().ok_or("the step set holds no token")?;
        let pu = p as usize;
        if prompt.len() != pu || m.ubatch()? != U_GATE {
            return Err(format!(
                "the step set's prompt is {} ids before position {p}, the model's ubatch {}; \
                 the clause runs {U_GATE} ids as one ubatch",
                prompt.len(),
                m.ubatch()?
            )
            .into());
        }
        let pass = prompt_run(m, prompt, PrefillPath::Pass, token)?;
        let ub = prompt_run(m, prompt, PrefillPath::Gemm, token)?;
        let kinds = m.body("wide_arm")?.kinds();
        let slots: Vec<usize> = (0..RING_ROWS).collect();
        // (u)
        let mut ok = true;
        for (l, (a, b)) in ub.stores.iter().zip(&pass.stores).enumerate() {
            let (x, y) = store_rel(a, b, &slots);
            if l == 0 {
                let pass0 = kinds[0] == LayerKind35::Delta && worse(x, y) <= U_L0_REL;
                println!(
                    "wide layer=0 {:?} state_rel={x:.3e} ring_rel={y:.3e} against the pass run \
                     (band {U_L0_REL:.0e}) {}",
                    kinds[0],
                    verdict(pass0)
                );
                ok &= pass0;
            } else {
                println!(
                    "wide layer={l} {:?} rel={x:.3e} / {y:.3e} (printed)",
                    kinds[l]
                );
            }
        }
        println!(
            "wide prefill: token ours ubatch={} pass={}, last logits_rel={:.3e} (printed)",
            ub.next,
            pass.next,
            rel_to(&ub.logits, &pass.logits)
        );
        // (d)
        let ik_slot = ik_slots(pu);
        let mut worst = (0.0f64, 0usize);
        for (l, &kind) in kinds.iter().enumerate() {
            let ik = prefill_store(&man, kind, l, pu)?;
            let (ua, ub_) = store_rel(&ub.stores[l], &ik, &ik_slot);
            let (pa, pb) = store_rel(&pass.stores[l], &ik, &ik_slot);
            let (wa, wb) = store_rel(&ub.stores[l], &pass.stores[l], &ik_slot);
            let d_pi = worse(pa, pb).max(f64::MIN_POSITIVE);
            let r = worse(0.0, worse(ua, ub_) / d_pi);
            if r > worst.0 {
                worst = (r, l);
            }
            let rw = worse(wa, wb) / d_pi;
            println!(
                "prompt store layer={l} {kind:?} from ik: ubatch {:.3e} pass {:.3e} ratio {r:.2}; \
                 ubatch from pass {:.3e}, independent-error bound {:.2} (printed)",
                worse(ua, ub_),
                worse(pa, pb),
                worse(wa, wb),
                (1.0 + rw * rw).sqrt()
            );
        }
        let pass_d = worst.0 <= PROMPT_RATIO;
        println!(
            "prompt stores: worst ratio {:.2} at layer {} (band {PROMPT_RATIO}) {}",
            worst.0,
            worst.1,
            verdict(pass_d)
        );
        ok &= pass_d;
        let mut worst_out = (0.0f64, 0usize);
        for (l, (a, b)) in ub.step_taps.iter().zip(&pass.step_taps).enumerate() {
            let want = tap(&man, &format!("l_out-{l}"))?;
            let want = last_rows(&want, HIDDEN, 1).ok_or("l_out")?;
            let r = worse(
                0.0,
                rel_to(a, want) / rel_to(b, want).max(f64::MIN_POSITIVE),
            );
            if r > worst_out.0 {
                worst_out = (r, l);
            }
        }
        let ik_logits = tap(&man, "result_output")?;
        let vocab = m.body("wide_arm")?.vocab();
        let ik_last = last_rows(&ik_logits, vocab, 1).ok_or("result_output")?;
        let (lu, lp) = (
            rel_to(&ub.step_logits, ik_last),
            rel_to(&pass.step_logits, ik_last),
        );
        let r_logits = worse(0.0, lu / lp.max(f64::MIN_POSITIVE));
        let pass_s = worst_out.0 <= STEP_RATIO && r_logits <= STEP_RATIO;
        println!(
            "prompt step {p}: worst l_out ratio {:.2} at layer {}, logits from ik ubatch {lu:.3e} \
             pass {lp:.3e} ratio {r_logits:.2} (band {STEP_RATIO}); argmax ubatch={} pass={} \
             ik={} (printed) {}",
            worst_out.0,
            worst_out.1,
            ub.step_top,
            pass.step_top,
            argmax(ik_last),
            verdict(pass_s)
        );
        Ok(ok && pass_s)
    }

    // ------------------------------------------ (w) the ubatch size

    fn ubatch_bits(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let man = RefManifest::open(&data_dir().join(D1K), &IK)?;
        let (_, _, prompt) = man.step()?;
        let run = |m: &mut Qwen35moeModel,
                   u: usize,
                   cut: Option<usize>|
         -> Result<(u32, Vec<f32>, Vec<StoreHost>), GateError> {
            m.set_ubatch(u)?;
            fresh(m)?;
            let next = match cut {
                None => m.prefill_with(prompt, PrefillPath::Gemm)?,
                Some(c) => {
                    m.prefill_with(&prompt[..c], PrefillPath::Gemm)?;
                    m.prefill_with(&prompt[c..], PrefillPath::Gemm)?
                }
            };
            Ok((next, m.logits()?, stores(m)?))
        };
        let (tok0, logits0, stores0) = run(m, U_GATE, None)?;
        let arms = W_SIZES
            .iter()
            .map(|&u| (u, None))
            .chain(std::iter::once((U_GATE, Some(W_CUT))));
        let mut ok = true;
        for (u, cut) in arms {
            let (tok, logits, st) = run(m, u, cut)?;
            let differ: Vec<usize> = (0..st.len())
                .filter(|&l| !stores0.get(l).is_some_and(|w| same_store(&st[l], w)))
                .collect();
            let same_logits = bits_equal(&logits, &logits0);
            let pass = tok == tok0 && same_logits && differ.is_empty();
            let how = match cut {
                None => format!("plan={}", m.prefill_plan(prompt.len(), PrefillPath::Gemm)?),
                Some(c) => format!("two calls of {c} and {} ids", prompt.len() - c),
            };
            println!(
                "ubatch {u} {how}: token {tok} vs {tok0}, last logits bit-identical={same_logits}, \
                 every store bit-identical (layers differing {differ:?}) {}",
                verdict(pass)
            );
            ok &= pass;
        }
        m.set_ubatch(U_GATE)?;
        Ok(ok)
    }

    // ---------------------------------------------- (k) the checkpoints

    /// (k)'s prompt: past an inner mark of the load's ubatch with a run
    /// left after it, and short of the next whole window.
    const KP: usize = 4200;

    /// (k)'s cache: the prompt, and the marks' positions, with room.
    const KCTX: usize = 4352;

    /// `n` ids of an lcg over the vocabulary: every id a row of the
    /// embedding, the routing spread over the experts.
    fn lcg_ids(n: usize, vocab: usize) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((x >> 33) % vocab as u64) as u32
            })
            .collect()
    }

    /// (k) (module doc): the checkpoints of a marked prompt call, the keep
    /// rule around them, the refused non-checkpoint cut, and the re-feed
    /// after a cut bit for bit. The model is dropped before the clause
    /// returns.
    fn checkpoints() -> Result<bool, GateError> {
        let mut m = open(KCTX, KvQ8::F16)?;
        {
            let (_, _, body) = m.body_parts("gate_qwen35moe_e2e")?;
            body.set_checkpoints(true);
        }
        let vocab = m.body("gate_qwen35moe_e2e")?.vocab();
        let ids = lcg_ids(KP, vocab);
        let mut ok = true;

        // One checkpoint's bytes and slots against their derivation: each
        // delta layer's state (N_V heads of HEAD_V x HEAD_V f32) and conv
        // ring (RING_ROWS rows of C), and the host budget's share of them.
        let c = m.body("gate_qwen35moe_e2e")?.checkpoints();
        let want_bytes = N_DELTA * (N_V * HEAD_V * HEAD_V + RING_ROWS * C) * 4;
        let points_ok = c.bytes() == want_bytes as u64 && c.capacity() >= 9;
        println!(
            "checkpoints: one copy {} bytes (want {want_bytes}, derived: a delta layer's state \
             {} f32 and ring {} f32), {} slots of one copy the host budget holds {}",
            c.bytes(),
            N_V * HEAD_V * HEAD_V,
            RING_ROWS * C,
            c.capacity(),
            verdict(points_ok)
        );
        ok &= points_ok;

        // The fresh call: its marks, the walks it ran, and what a cut keeps
        // around them.
        fresh(&mut m)?;
        let plan = m.prefill_plan(KP, PrefillPath::Wide)?;
        let tok0 = m.prefill_with(&ids, PrefillPath::Wide)?;
        let (logits0, stores0) = (m.logits()?, stores(&mut m)?);
        // PIN(2026-10-04): the marks' spacing is the ubatch size the load was
        // asked for — the walk's most positions, the sibling body's rule —
        // re-derived here from the model's own size, where it was GLM's flat
        // 512: a mark every 512 rows cut a seat prompt of one ubatch walk's
        // rows into a walk each, every walk re-reading the expert weights
        // ([derived] pp@4096 at the seat down 27–29 %). At the load's size
        // the marks are its multiples: the points, the keep below an ask
        // (the multiple under it) and the walks follow — a run between
        // marks is one whole ubatch, the tail one unit, so the call runs
        // exactly ⌈P/ub⌉ walks, one more only where a mark falls inside a
        // walk (a continuation's first, or the borrowed tail).
        let ub = u32::try_from(m.ubatch()?).expect("a ubatch the load validated, at most UBATCH");
        let mark_below = |ask: u32| ub * (ask / ub);
        let want_pts: Vec<u32> = (1..)
            .map(|k| ub * k)
            .take_while(|&p| p < KP as u32)
            .chain([KP as u32])
            .collect();
        let pts = m.body("gate_qwen35moe_e2e")?.checkpoints().positions();
        let walks = m.body("gate_qwen35moe_e2e")?.prompt_walks();
        let (k_above, k_below, k_low, k_here) = (
            m.body("gate_qwen35moe_e2e")?.kept(4150, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(4095, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(100, KP as u32),
            m.body("gate_qwen35moe_e2e")?.kept(KP as u32, KP as u32),
        );
        let marks_ok = pts == want_pts
            && (k_above.at, k_above.why.code()) == (mark_below(4150), "checkpoint")
            && (k_below.at, k_below.why.code()) == (mark_below(4095), "checkpoint")
            && (k_low.at, k_low.why.code()) == (mark_below(100), "no-checkpoint")
            && (k_here.at, k_here.why.code()) == (KP as u32, "current")
            && plan.ubatch_tokens() == KP;
        let walks_ok = walks == (KP as u32).div_ceil(ub) as usize;
        println!(
            "checkpoints: a prompt of {KP} ids (plan {plan}) took its points at {pts:?} (want \
             {want_pts:?}, the ubatch {ub}'s multiples); kept(4150) = {k_above}; kept(4095) = \
             {k_below}; kept(100) = {k_low}; kept({KP}) = {k_here}; {walks} walks (want \
             {}, the plan's own units) {}",
            (KP as u32).div_ceil(ub),
            verdict(marks_ok && walks_ok)
        );
        ok &= marks_ok && walks_ok;

        // A cut to a non-checkpoint is refused by name, the points kept.
        let refused = match m.rollback(4150) {
            Err(e) => e.to_string(),
            Ok(()) => "accepted".to_owned(),
        };
        let cut_ok = refused.contains("no checkpoint restores it")
            && m.body("gate_qwen35moe_e2e")?.checkpoints().positions() == want_pts;
        println!(
            "checkpoints: a cut to 4150: {refused}; the points stand {} ",
            verdict(cut_ok)
        );
        ok &= cut_ok;

        // The cut and the re-fed tail: the same bits the uncut call left.
        let cut_at = mark_below(4150);
        m.rollback(cut_at)?;
        let tail = &ids[cut_at as usize..];
        let tok = m.prefill_with(tail, PrefillPath::Wide)?;
        let (logits, st) = (m.logits()?, stores(&mut m)?);
        let differ: Vec<usize> = (0..st.len())
            .filter(|&l| !stores0.get(l).is_some_and(|w| same_store(&st[l], w)))
            .collect();
        let same_logits = bits_equal(&logits, &logits0);
        let refed_walks = m.body("gate_qwen35moe_e2e")?.prompt_walks();
        let refed_ok = tok == tok0
            && same_logits
            && differ.is_empty()
            && st.len() == stores0.len()
            && m.body("gate_qwen35moe_e2e")?.checkpoints().positions() == want_pts
            // A cut lands on a mark, so the re-fed call's only run is the
            // tail itself: one walk of its own plan.
            && refed_walks == (tail.len() as u32).div_ceil(ub) as usize;
        println!(
            "checkpoints: cut to {cut_at} and {} ids re-fed: token {tok} vs {tok0}, last logits \
             bit-identical={same_logits}, every store bit-identical (layers differing {differ:?}), \
             the points back at {want_pts:?}, {refed_walks} walk {}",
            tail.len(),
            verdict(refed_ok)
        );
        ok &= refed_ok;

        // A reset drops every point.
        fresh(&mut m)?;
        let dropped = m
            .body("gate_qwen35moe_e2e")?
            .checkpoints()
            .positions()
            .is_empty();
        println!(
            "checkpoints: a reset drops every point {}",
            verdict(dropped)
        );
        // The flash grid at this load's height (4,352 rows, not the (s)
        // clause's 1,088): the same SEGMENTS-sized segment pass and head
        // merge as (s) — the engine-level two-height pin, on a capture that
        // runs after the reset and so changes nothing above.
        m.capture_step()?;
        let (flash_ok, got) = flash_grid_ok(&m.step_graph_nodes()?)?;
        println!(
            "checkpoints flash grid at ctx {KCTX}: {got} {}",
            verdict(flash_ok)
        );
        Ok(ok && dropped && flash_ok)
    }

    // ------------------------------------------------ (n) resident slots

    /// (n)'s prompts, A and B: each past the load's first mark ([`U_GATE`])
    /// by a run the mark keeps (`marked` drops a mark with a run of fewer
    /// than `WIDE_FROM` = `GEMV_COLS + 1` = 9 rows beside it), of different
    /// lengths so the slots stand at different positions, each with its
    /// steps inside [`CTX`].
    const SLOT_A: usize = 1040;
    const SLOT_B: usize = 1056;

    /// Greedy steps a stream takes past its prompt in the interleave, and
    /// of one continuation check: each solo run steps `SLOT_STEPS + 2 ·
    /// SLOT_TAIL` (`slots_gate`), inside [`CTX`] for B.
    const SLOT_STEPS: usize = 16;
    const SLOT_TAIL: usize = 8;

    const _: () = assert!(
        SLOT_A >= U_GATE + 9
            && SLOT_B >= U_GATE + 9
            && SLOT_A + SLOT_STEPS + 2 * SLOT_TAIL <= CTX
            && SLOT_B + SLOT_STEPS + 2 * SLOT_TAIL <= CTX
    );

    /// Qwen3.6's adapter: the whole-card load at [`CTX`] with the
    /// checkpoints armed, and two lcg prompts over the vocabulary.
    struct Q35Slots {
        /// Stream 0's ids (A), then stream 1's (B).
        ids: Vec<u32>,
    }

    impl Q35Slots {
        fn new(vocab: usize) -> Q35Slots {
            Q35Slots {
                ids: lcg_ids(SLOT_A + SLOT_B, vocab),
            }
        }

        /// Stream `stream`'s ids, else refused by name.
        fn ids(&self, stream: usize) -> Result<&[u32], GateError> {
            let (a, b) = self.ids.split_at(SLOT_A);
            match stream {
                0 => Ok(a),
                1 => Ok(b),
                s => Err(format!("stream {s} of the adapter's two").into()),
            }
        }
    }

    impl SlotsAdapter for Q35Slots {
        type Body = Body35;

        const STEPS: usize = SLOT_STEPS;
        const TAIL: usize = SLOT_TAIL;

        /// The whole-card load at [`CTX`], the checkpoints armed as the seat
        /// arms them. It is the same load at every slot count: its fit
        /// check counts no slot, and each sequence's stores are allocated
        /// by `add_slots`.
        fn open(&self, _slots: usize) -> Result<Qwen35moeModel, GateError> {
            let mut m = open(CTX, KvQ8::F16)?;
            m.body_parts("slots")?.2.set_checkpoints(true);
            Ok(m)
        }

        /// `reset`: every row `state_hash` reads is written by a call from
        /// it.
        fn rewind(&self, m: &mut Qwen35moeModel) -> Result<(), GateError> {
            Ok(m.reset()?)
        }

        /// The session's schedule (`app::Prompt for Body35`): the wide walk
        /// from `GEMM_FROM` ids on, passes below.
        fn prompt(&self, m: &mut Qwen35moeModel, stream: usize) -> Result<u32, GateError> {
            Ok(<Body35 as app::Prompt>::prompt(m, self.ids(stream)?)?)
        }

        /// Every attention layer's K and V rows below the model's position —
        /// rows past it are no state: the flash never reads them, and a slot
        /// rewound after a longer sequence still holds that sequence's there —
        /// and every delta layer's whole state and conv ring.
        fn state_hash(&self, m: &mut Qwen35moeModel) -> Result<u64, GateError> {
            let body = m.body("slots")?;
            let rows = usize::try_from(m.pos())? * HEAD;
            let plane = body.ctx_rows() * HEAD;
            let mut h = Fnv1a64::default();
            for l in m.layers() {
                h = match body.store(m.gpu(), l)? {
                    StoreHost::Kv { k, v } => {
                        k.chunks(plane)
                            .chain(v.chunks(plane))
                            .try_fold(h, |h, head| {
                                let below =
                                    head.get(..rows).ok_or("a position past the cache rows")?;
                                Ok::<_, GateError>(
                                    below.iter().fold(h, |h, w| h.bytes(&w.to_le_bytes())),
                                )
                            })?
                    }
                    StoreHost::Rec { state, ring } => h.f32s(&state).f32s(&ring),
                };
            }
            Ok(h.value())
        }

        /// The stores' derivation from the header ([`store_bytes`]): each
        /// attention layer's K and V planes, each delta layer's state and
        /// conv ring.
        fn seq_bytes_derived(&self, _m: &Qwen35moeModel) -> Result<Derived, GateError> {
            Ok(Derived {
                bytes: store_bytes(),
                terms: format!(
                    "{N_ATTN} x K/V planes of 2 x {N_KV} x {CTX} x {HEAD} f16 + {N_DELTA} x (state \
                     {N_V} x {HEAD_V} x {HEAD_V} + ring {RING_ROWS} x {C}) f32"
                ),
            })
        }
    }

    /// The selected slot's checkpoints' positions.
    fn points(m: &Qwen35moeModel) -> Result<Vec<u32>, GateError> {
        Ok(m.body("slots")?.checkpoints().positions())
    }

    /// `n` greedy steps on the selected slot, appended to `ids` (its last id
    /// the first step's input).
    fn greedy(m: &mut Qwen35moeModel, ids: &mut Vec<u32>, n: usize) -> Result<(), GateError> {
        for _ in 0..n {
            let last = *ids.last().ok_or("no token")?;
            ids.push(m.step(&[last])?);
        }
        Ok(())
    }

    /// (n) (module doc): the harness's contracts, and between its halves
    /// each slot's checkpoints its own under a cut and a reset of the other
    /// slot. The clause's model is dropped with its second sequence.
    fn slots_two(a: &Q35Slots) -> Result<bool, GateError> {
        let mut s = slots_gate::interleave(a)?;
        let mark = u32::try_from(U_GATE)?;
        let ends = [u32::try_from(SLOT_A)?, u32::try_from(SLOT_B)?];
        let b_last = s.last(1)?;
        let m = s.model();
        // Each slot's points as its prompt in the interleave took them, and
        // where each stands.
        m.select_slot(0)?;
        let a_held = (a.state_hash(m)?, points(m)?);
        m.select_slot(1)?;
        let b_held = (a.state_hash(m)?, points(m)?, m.pos());
        let taken = a_held.1 == [mark, ends[0]] && b_held.1 == [mark, ends[1]];
        println!(
            "slots points: slot 0 {:?}, slot 1 {:?} (want [{mark}, {}] and [{mark}, {}]: the \
             mark and each prompt's end) {}",
            a_held.1,
            b_held.1,
            ends[0],
            ends[1],
            verdict(taken)
        );
        // Slot 1 cut back to the mark; slot 0 runs before slot 1 does again.
        m.rollback(mark)?;
        m.select_slot(0)?;
        let a_now = (a.state_hash(m)?, points(m)?);
        let cont = s.continues(0)?;
        let held_ok = a_now == a_held && cont;
        println!(
            "slots cut: slot 1 cut back to {mark}; slot 0's points {:?} and state digest {} as it \
             stood, its next {SLOT_TAIL} ids {} its solo run's continuation {}",
            a_now.1,
            if a_now.0 == a_held.0 {
                "equal"
            } else {
                "differ from"
            },
            if cont { "equal" } else { "differ from" },
            verdict(held_ok)
        );
        // Slot 1: the rest of B from the mark, then the interleave's steps.
        let m = s.model();
        m.select_slot(1)?;
        let rest = a
            .ids(1)?
            .get(U_GATE..)
            .ok_or("stream B shorter than the first mark")?;
        let mut ids = vec![<Body35 as app::Prompt>::prompt(m, rest)?];
        greedy(m, &mut ids, SLOT_STEPS)?;
        let b_now = (a.state_hash(m)?, points(m)?, m.pos());
        let last = ids.last().copied();
        let refed_ok = last == Some(b_last) && b_now == b_held;
        println!(
            "slots cut: slot 1 re-fed {} ids from {mark} and stepped {SLOT_STEPS}: last id {last:?} \
             vs {b_last}, state digest equal={}, points {:?} vs {:?}, position {} vs {} {}",
            rest.len(),
            b_now.0 == b_held.0,
            b_now.1,
            b_held.1,
            b_now.2,
            b_held.2,
            verdict(refed_ok)
        );
        // A reset of slot 1 leaves slot 0's points.
        a.rewind(m)?;
        m.select_slot(0)?;
        let kept = points(m)?;
        let kept_ok = kept == a_held.1;
        println!(
            "slots cut: after slot 1's reset slot 0's points {kept:?} (want {:?}) {}",
            a_held.1,
            verdict(kept_ok)
        );
        let harness_ok = s.finish()?;
        Ok(taken && held_ok && refed_ok && kept_ok && harness_ok)
    }

    // ---------------------------------------------------- (r) refusals

    fn refusals(m: &mut Qwen35moeModel) -> Result<bool, GateError> {
        let (u0, bytes0) = (m.ubatch()?, m.resident_bytes());
        let mut ok = true;
        for u in [0, UBATCH + 1] {
            let e = m.set_ubatch(u).err();
            let kept = m.ubatch()? == u0 && m.resident_bytes() == bytes0;
            let pass = e.is_some() && kept;
            println!(
                "refuse ubatch {u}: {} (size {u0} and resident bytes kept={kept}) {}",
                e.map_or("accepted".to_string(), |e| e.to_string()),
                verdict(pass)
            );
            ok &= pass;
        }
        fresh(m)?;
        let long = vec![0u32; CTX + 1];
        let e = m.prefill_with(&long, PrefillPath::Auto).err();
        let pass = e.is_some() && m.pos() == 0;
        println!(
            "refuse a prompt of {} ids on a {CTX}-row cache: {} (position {}) {}",
            long.len(),
            e.map_or("accepted".to_string(), |e| e.to_string()),
            m.pos(),
            verdict(pass)
        );
        Ok(ok && pass)
    }

    // ---------------------------------------------- (o) the placed load

    /// (o)'s card budget: a 12 GiB card's, under which the file's routed
    /// experts split between the card and the host tier.
    const PLACED_BUDGET: u64 = 12 << 30;

    /// What the memguard clause leaves free of the card while it holds the
    /// rest: well under the trunk's smallest term, so the refusal is
    /// certain.
    const MEMGUARD_LEFT: usize = 512 << 20;

    /// The memory guard on this file (`gate_qwen3moe_e2e`'s (m) is the
    /// clause's full form; this one holds the qwen35moe shapes of the same
    /// planner): the quiet-card decision is the whole-card load, holding all
    /// but [`MEMGUARD_LEFT`] of the free bytes refuses the plan on device 0
    /// by name and makes the unset decision the placed one, and the hold
    /// dropped restores the whole-card decision — the reading is live.
    /// FAIL-first: a census that reads the total as the free bytes (mutant:
    /// `raw_device_free_bytes` returning `cuDeviceTotalMem`'s figure) turns
    /// every arm red.
    fn memguard() -> Result<bool, GateError> {
        use bloomery_gpu_gates::gpu_census::census;
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let d0 = census()?
            .into_iter()
            .find(|d| d.ordinal == 0)
            .ok_or("the census reads no device 0")?;
        println!(
            "memguard census: cuda0 {} free {} B",
            d0.name, d0.free_bytes
        );
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
        };
        let whole = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Whole
        );
        println!("memguard quiet: whole decision {whole} {}", verdict(whole));
        let gpu = bloomery_gpu::Gpu::new()?;
        let (free, _) = gpu.mem_info()?;
        let hold = free
            .checked_sub(MEMGUARD_LEFT)
            .ok_or("the card had less free than the clause leaves")?;
        let held = cuda_core::DeviceBuffer::<u8>::zeroed(gpu.stream(), hold)?;
        let refusal = match q3place::PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?
            .plan(CTX, &PlanLevers::default())
        {
            Ok(_) => String::new(),
            Err(e) => e.to_string(),
        };
        let refused = refusal.contains("the plan's dense trunk alone needs");
        println!("memguard held: refusal {refused} {refusal}");
        let placed = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Placed(_)
        );
        println!(
            "memguard held: placed decision {placed} {}",
            verdict(placed)
        );
        drop(held);
        drop(gpu);
        let whole_again = matches!(
            q3place::unplaced_qwen35(&file, &o)?,
            q3place::Unplaced::Whole
        );
        println!(
            "memguard released: whole decision {whole_again} {}",
            verdict(whole_again)
        );
        Ok(whole && refused && placed && whole_again)
    }

    /// (o) (module doc): the model is dropped before the clause returns.
    fn placed(man: &RefManifest, toks: &[u32], host: HostCfg) -> Result<bool, GateError> {
        let file = Split::open(MODEL).map_err(|e| format!("open {MODEL}: {e}"))?;
        let o = Open35 {
            ctx: CTX,
            mma: true,
            ubatch: U_GATE,
            kv: KvQ8::F16,
        };
        let q = PlaceQ3::qwen35(&file, Place::parse("cuda0")?, o)?;
        let levers = PlanLevers {
            card_budget_bytes: Some(PLACED_BUDGET),
        };
        let plan = q.plan(CTX, &levers)?;
        q.record(&plan, None).print();
        let split = plan.host.experts > 0 && plan.cards[0].experts > 0;
        println!(
            "placed plan: card_experts={} host_experts={} (both above 0) {}",
            plan.cards[0].experts,
            plan.host.experts,
            verdict(split)
        );
        if !split {
            return Ok(false);
        }
        let t = Instant::now();
        let mut m = GpuModel::<Body35>::open_placed(file, &plan, o, host)?;
        let counts = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .card_counts();
        println!(
            "placed load resident_bytes={} ubatch={} card experts per layer {counts:?} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        m.set_mode(StepMode::Graph);
        println!("placed capture graph_nodes={}", m.capture_step()?);
        // (o1)
        let decode = decode_run(&mut m, toks, false)?;
        m.set_mode(StepMode::Eager);
        let eager = decode_run(&mut m, toks, false)?;
        let mut ok = same_run(
            "placed: five eager steps vs five graph replays",
            &eager,
            &decode,
        );
        // (o2)
        ok &= free(&mut m, man, toks, &decode)?;
        m.set_mode(StepMode::Graph);
        fresh(&mut m)?;
        let plan_p = m.prefill_plan(toks.len(), PrefillPath::Auto)?;
        let tok = m.prefill_with(toks, PrefillPath::Auto)?;
        let call = PathRun {
            logits: vec![m.logits()?],
            tokens: vec![tok],
            stores: stores(&mut m)?,
        };
        let PathRun {
            mut logits,
            mut tokens,
            stores: last_stores,
        } = decode;
        let fifth = PathRun {
            logits: logits.pop().into_iter().collect(),
            tokens: tokens.pop().into_iter().collect(),
            stores: last_stores,
        };
        ok &= same_run(
            &format!("placed: the prompt call ({plan_p}) vs the fifth graph step"),
            &call,
            &fifth,
        );
        // (o3)
        let u0 = m.ubatch()?;
        let gemm = m.prefill_plan(toks.len(), PrefillPath::Gemm).err();
        let resize = m.set_ubatch(U_GATE).err();
        let refused = gemm.is_some() && resize.is_some() && m.ubatch()? == u0;
        println!(
            "placed refusals: a GEMM prompt: {}; ubatches of {U_GATE}: {} (ubatch {u0} kept) {}",
            gemm.map_or("accepted".to_string(), |e| e.to_string()),
            resize.map_or("accepted".to_string(), |e| e.to_string()),
            verdict(refused)
        );
        // (o4)
        let stats = m
            .body("gate_qwen35moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .hybrid()
            .stats();
        let served = stats.host_slots > 0 && stats.batch_host_slots > 0;
        println!(
            "placed host tier: step host_slots={} batch host_slots={} (both above 0) {}",
            stats.host_slots,
            stats.batch_host_slots,
            verdict(served)
        );
        Ok(ok && refused && served)
    }

    /// The batch set's manifest and its five tokens.
    fn batch_set() -> Result<(RefManifest, Vec<u32>), GateError> {
        let man = RefManifest::open(&data_dir().join(BATCH), &IK)?;
        let toks: Vec<u32> = ref_ints(&man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
            .iter()
            .map(|&i| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        Ok((man, toks))
    }

    /// The q8_0 cache arm (`BLOOMERY_QWEN3_KV=q8_0`, the seat's
    /// `--cache-type-k q8_0`), on its own models dropped before the clause
    /// returns: the attention layers' planes hold the two-plane layout (the
    /// store bytes drop, and the resident bytes drop by exactly the stores'
    /// own delta; the delta layers' stores are f32 either way), and the
    /// oracle set's first ids stepped, the same ids prefilled on the pass
    /// path and the same steps through the captured graph all leave the
    /// same last logits bit for bit and the same argmax — the arm's own
    /// consistency, the (p) and (r) contracts on the q8 path. The distance
    /// to the f16 run's logits prints as the quantization diagnostic
    /// (`store`'s f16 readback refuses a q8_0 cache by name, so the bits
    /// clause holds the logits and tokens). Asking the q8 model for the
    /// scalar pass is refused by name at the call, and its next step still
    /// runs.
    fn q8_cache() -> Result<bool, GateError> {
        const Q8_N: usize = 96;
        let (_, toks) = batch_set()?;
        let ids: Vec<u32> = toks.iter().take(Q8_N).copied().collect();
        let mut f16 = open(CTX, KvQ8::F16)?;
        let (f16_resident, f16_store) = (f16.resident_bytes(), f16.body("q8")?.store_bytes());
        f16.reset()?;
        let mut f16_tok = 0;
        for &id in &ids {
            f16_tok = f16.step(&[id])?;
        }
        let f16_logits = f16.logits()?;
        drop(f16);
        let mut m = open(CTX, KvQ8::Q8)?;
        let (resident, store) = (m.resident_bytes(), m.body("q8")?.store_bytes());
        let bytes_ok = f16_store > store
            && f16_resident.saturating_sub(resident) == f16_store.saturating_sub(store);
        println!(
            "q8 load: resident_bytes={resident} (f16 {f16_resident}), store bytes {store} of \
             f16's {f16_store} — the residents differ by the stores' own delta {}",
            f16_store.saturating_sub(store)
        );
        let mut ok = bytes_ok;
        m.reset()?;
        let mut step_tok = 0;
        for &id in &ids {
            step_tok = m.step(&[id])?;
        }
        let logits = m.logits()?;
        // The quantization diagnostic against the f16 run: the argmax's
        // agreement and the relative distance, printed.
        let num: f64 = logits
            .iter()
            .zip(&f16_logits)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let den: f64 = f16_logits.iter().map(|b| f64::from(*b).powi(2)).sum();
        println!(
            "q8 steps: {Q8_N} ids; argmax ours={step_tok} f16={f16_tok} (printed); logits \
             rel dist {:.3e} (diagnostic)",
            if den > 0.0 { (num / den).sqrt() } else { 0.0 }
        );
        // The pass prefill of the same ids leaves the stepped run's bits.
        m.reset()?;
        let pass_tok = m.prefill_with(&ids, PrefillPath::Pass)?;
        let pass_logits = m.logits()?;
        let bits = pass_tok == step_tok && logits == pass_logits;
        println!(
            "q8 bits: pass prefill == {Q8_N} steps, last logits bit for bit {}",
            verdict(bits)
        );
        ok &= bits;
        // The graph replay: the same steps through the captured step graph.
        m.reset()?;
        m.set_mode(StepMode::Graph);
        println!("q8 capture graph_nodes={}", m.capture_step()?);
        m.reset()?;
        let mut last = 0;
        for &id in &ids {
            last = m.step(&[id])?;
        }
        let graph_logits = m.logits()?;
        let replay = last == step_tok && logits == graph_logits;
        println!(
            "q8 replay: {Q8_N} captured steps == the eager bits {}",
            verdict(replay)
        );
        ok &= replay;
        // The scalar pass on a q8_0 cache: refused by name at the call that
        // asks for it, in `q8_unserved`'s words, and the model's pass kept —
        // the next eager step runs.
        m.set_mode(StepMode::Eager);
        let named = match m.set_flash_mma(false) {
            Err(GpuError::Shape {
                what: "qwen35moe::set_flash_mma",
                detail,
            }) => detail.contains(
                "a q8_0 cache's flash runs the eight-head layout's tensor-core pass only; asked \
                 for the scalar pass",
            ),
            _ => false,
        };
        let kept = m.reset().is_ok() && m.step(&[ids[0]]).is_ok();
        println!(
            "q8 scalar refusal: set_flash_mma(false) refused by name at the call {named}, the \
             next step runs {kept} {}",
            verdict(named && kept)
        );
        ok &= named && kept;
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        let host = bloomery_levers::at_main(&[])?.host();
        if std::env::args().any(|a| a == "--placed-only") {
            let (man, toks) = batch_set()?;
            let ok = placed(&man, &toks, host)?;
            println!("gate_qwen35moe_e2e --placed-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if std::env::args().any(|a| a == "--memguard-only") {
            let ok = memguard()?;
            println!("gate_qwen35moe_e2e --memguard-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        let mut m = open(CTX, KvQ8::F16)?;
        let mut ok = file_shape()?;
        ok &= structure(&mut m)?;
        let (man, toks) = batch_set()?;
        let (p_ok, decode) = paths(&mut m, &toks)?;
        ok &= p_ok;
        ok &= free(&mut m, &man, &toks, &decode)?;
        let gains = Gains::read(m.layers().len())?;
        m.set_mode(StepMode::Eager);
        fresh(&mut m)?;
        let zero = |_: &mut Qwen35moeModel, _: usize| -> Result<(), GateError> { Ok(()) };
        ok &= forced(&mut m, &man, BATCH, (0, toks.len()), &gains, &zero)?;
        m.set_mode(StepMode::Graph);
        for set in [STEP4, D1K] {
            ok &= step_set(&mut m, set, &gains)?;
        }
        ok &= gemv_arm(&mut m, &toks, &decode)?;
        ok &= wide_arm(&mut m)?;
        ok &= ubatch_bits(&mut m)?;
        ok &= refusals(&mut m)?;
        let slots = Q35Slots::new(m.body("gate_qwen35moe_e2e")?.vocab());
        drop(m);
        ok &= checkpoints()?;
        ok &= slots_two(&slots)?;
        ok &= placed(&man, &toks, host)?;
        ok &= q8_cache()?;
        println!("gate_qwen35moe_e2e: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
