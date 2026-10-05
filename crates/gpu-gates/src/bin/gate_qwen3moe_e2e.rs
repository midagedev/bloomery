//! The qwen3moe end-to-end gate: the whole chain — 48 layers, the head and
//! the argmax — on one card, against ik's CPU oracle and its greedy answers.
//!
//! Every prompt runs from position 0 (`GpuModel::reset`), its tokens fed one
//! position at a time through the decode path — the greedy reference was
//! written the same way (`argmax_ref --step-prefill`) — and then again
//! through the prompt prefill (`Qwen3moeModel::prefill`): on the pass path,
//! which must leave the same answer bit for bit, and on the GEMM ubatches,
//! which must stay inside a derived band.
//!
//! What is asserted:
//! - (s) structure: the captured step holds [`NODES_CHAIN`] nodes, none of
//!   them a memcpy (the combine writes the next layer's input in place) or
//!   a host node; the captured prefill pass holds [`NODES_PASS_1`] nodes at
//!   one token and [`NODES_PASS_M`] at every count from two to `MAX_TOKENS`,
//!   all of them kernels. The flash launches' grids, in the step and in
//!   every prefill pass: the segment pass `m · n_kv · SEGMENTS` blocks a
//!   layer whatever the cache height, the merge `m · n_head`, one of each
//!   a layer.
//! - (k) the clear: right after (s), before any token has run, three arms
//!   through the session (`app::Session::arms`), each after the first from
//!   `Session::clear` — a GEMM ubatch arm (the prose's first [`CLEAR_A`] ids
//!   and greedy steps), a pass arm (ids after those, at most `MAX_TOKENS`,
//!   and steps), and the GEMM arm again. The first arm stands in for a fresh
//!   process, so the third must equal it in every field that is not a time:
//!   the position it starts at, the plan, the tokens, the last logits and
//!   every layer's K/V rows below its end (FNV-1a 64 each), and the ubatch
//!   image it wrote. Each arm's plan is refused by name unless it takes the
//!   path the clause names. (s) has read its node counts before it.
//! - (f) teacher-forced, per layer: each layer run alone on ik's own input
//!   row (`inp_embd` for layer 0, `l_out-(L−1)` after) at each of the
//!   oracle's five positions. The attention half against ik's `attn_out-L`
//!   (`‖(ffn_inp − x_in) − attn_out‖ / ‖attn_out‖`), and the FFN half run
//!   alone on ik's own FFN input `x_in + attn_out` against ik's
//!   `routed_out-L`, measured on the magnitude of the routed sum's terms
//!   (`Σ_s |w_s · down_s|`, which the eight terms' cancellation cannot
//!   shrink). Each half's error over the gap between the two sides' 8-bit
//!   inputs to it must stay within [`RATIO_BAND`]. A site where the router
//!   picks an expert ik does not is counted and printed, not judged.
//! - (c) free-running: the five tokens through the whole chain, every
//!   layer's `l_out` against ik's, `‖ours − ik‖ / ‖ik‖` within
//!   [`FREE_BAND`]; the last position's logits against ik's
//!   `result_output`: the same argmax, the relative distance printed.
//! - (d) the tap dump (`shared/qwen3moe_taps.rs`, `generate_qwen3moe
//!   --dump-taps`): [`TAP_SEQS`] windows of the prose, [`TAP_PROMPT`] ids
//!   each and [`TAP_GEN`] greedy steps, dumped into a fresh directory and
//!   read back. Every position's row of every tapped layer equals, bit for
//!   bit, that layer's `layer_taps` copy after an independent eager step of
//!   the file's id at that position (the copies (c) holds to ik's `l_out`,
//!   the last layer's before the output norm); and each ids file is the
//!   prompt, then the tokens of the plain run — graph mode, the prompt on the
//!   pass path, `TAP_GEN − 1` steps — at the same cache height.
//! - (g) greedy: prompts `0..PROMPTS` of `tools/ref/prompts.tsv` under this
//!   model's tokenizer (`just ik-greedy-qwen3moe` writes them and ik's
//!   continuations), [`GEN`] tokens each in graph mode: no prompt diverges
//!   from ik at a position where ik's own top1-top2 margin is at or above
//!   [`MARGIN_FLOOR`] — `gpu_gates::prompts::compare_greedy`'s classes, the
//!   V2-Lite gate's criterion; later divergences and near ties are printed.
//! - (r) graph replay equals the eager body: the same prompts in eager mode
//!   give the same tokens, and the last step's logits are bit-identical.
//! - (p) prefill equals the one-token path: the same prompts prefilled
//!   (each fits one pass, so the default path takes the pass), then the same
//!   greedy steps, give the same tokens and bit-identical last logits; so do
//!   the prompts' concatenation on the pass path (several passes) against its
//!   own one-token run, whole and in two calls split at [`SPLIT`], the second
//!   starting at a position that is not a pass boundary. These run through
//!   the captured passes.
//! - (q) a replayed prefill pass equals the eager one: for every pass size
//!   `m`, the concatenation's first `MAX_TOKENS + m` ids (a full pass, then
//!   one of `m`) prefilled on the pass path in graph mode and in eager mode,
//!   each from a reset over cache rows seeded with a pattern (`seed_depth`),
//!   leave the same K/V rows bit for bit, none of them still the pattern, and
//!   the same last logits bit for bit.
//! - (u) the GEMM prefill against the one-token path, on the first
//!   [`LONG`] ids of the prose file `corpus-prose.ids` (its digest checked),
//!   at the load's ubatch size (the default, 4096, clipped to the cache: each
//!   prompt one ubatch). Layer 0's K and V rows within [`GEMM_L0_REL`] of the
//!   one-token path's; each later layer's distance within
//!   [`GEMM_SPREAD_RATIO`] times the one-token path's own distance between
//!   its two flash arithmetics (the same prompt stepped eagerly on the other
//!   flash pass, `set_flash_mma`); the last logits within the same ratio of
//!   that run's; the greedy continuation after the prefill judged as (g)
//!   against the one-token path's own continuation and margins. And the 1,300 ids
//!   prefilled in two calls cut at [`LONG_SPLIT`] (every unit a ubatch, the
//!   second call's first position off every ubatch boundary) leave the whole
//!   prefill's K/V rows and last logits bit for bit: a token's GEMM-path
//!   values do not depend on the ubatch it lands in.
//! - (w) the ubatch size moves no bit: a model opened with [`CTX_W`] cache
//!   rows prefills the prose's first [`W_LONG`] ids at each size of
//!   [`W_SIZES`] (ubatches of 512 x 8; 1,000 x 4 and 96; one of 4,096 —
//!   32,768 routed slots in one table) and every run leaves the K/V rows,
//!   the last logits and the token of the 4,096 run bit for bit; so do one
//!   id more at 512 and at 4,096 (each ending in a one-id pass), and the
//!   4,096 ids at 4,096 prefilled in two calls cut at [`W_SPLIT`]. Sizes
//!   outside `1..=UBATCH` are refused with the size kept.
//! - (t) the rope table: every row of the table all three paths read — the
//!   cache's [`CTX`] positions, 0 and the last among them — holds
//!   `RopeTable::push`'s bits for its position. Each token reads the table
//!   row of the position it is appended at (the rope kernel's one `pos`
//!   word does both), so where the rows land shows which rows were read:
//!   from a reset over cache rows seeded with the pattern, the first
//!   [`T_FIRST`] prose ids and the next [`T_NEXT`] on the GEMM path in
//!   ubatches of [`T_UB`], the next [`T_PASS`] on the pass path and
//!   [`T_STEPS`] one-token steps leave every row below their end written in
//!   every layer's K and V planes and every row past it still the pattern.
//!   The ubatch size is put back and the model reset after, so the clauses
//!   after (t) run as they would without it.
//! - (x) a fault names its layer: layer [`FAULT_LAYER`]'s FFN half run alone
//!   on a finite row with one NaN raises inside that layer's launches, and
//!   the next step returns `GpuError::Fault` with that layer (the site is the
//!   smallest code among the layer's raises, printed), the model poisoned;
//!   `reset` leaves the word clean and the next step a token.
//!
//! - (v) the q8_0 cache arm (`BLOOMERY_QWEN3_KV=q8_0`, the seat's
//!   `--cache-type-k q8_0`): its own models, each dropped before the clause
//!   returns — the plan's KV term exactly 17/32 of the f16 term's
//!   (census-free), {`Q8_IDS`} prose ids stepped, prefilled on the pass path
//!   and replayed through the captured step graph leaving the same planes
//!   and logits bit for bit, and row 0's dequantized values within each
//!   32-value block's own quantization step of the f16 model's row 0 (bounds
//!   derived at runtime, no measured band); the later rows' compounding
//!   distance prints as a diagnostic. The auto context search under the two
//!   formats and the resident bytes' drop are `gate_qwen3_serve`'s
//!   (`q8_ctx_never_below_the_f16_answer`, `q8_resident_drops_by_the_planes`).
//! - (o) the placed load: the file planned on device 0 under a card budget
//!   of [`PLACED_BUDGET`] (`shared/qwen3moe_place.rs`, the CLI's and the
//!   seat's planner), which must leave routed experts both on the card and
//!   on the host tier, then loaded by that plan (`open_placed`): (c) and (g)
//!   again on it — every layer's `l_out` within [`FREE_BAND`] of ik's and
//!   the argmax ik's; no greedy prompt diverging from ik where ik's margin
//!   is at or above [`MARGIN_FLOOR`]; graph = eager; the pass prefill = the
//!   one-token path, bit for bit (every unit a walk through the host tier's
//!   batch port) — and the host tier served slots on both of its ports. The
//!   bands are the card's: a host expert runs ik's own 8-bit rule (Q8_K
//!   activations) where the card runs q8_1, so no layer's error grows.
//!   `--placed-only` runs (o) alone.
//!
//! - (m) the memory guard, `--memguard-only` (the card as the runner pins
//!   it, nothing loaded): the census reads device 0's free bytes (at most
//!   its usable bytes); with the card quiet the `--place`-unset load is
//!   today's whole-card one (`q3place::unplaced_qwen3` answers `Whole`, no
//!   plan record); holding all but [`MEMGUARD_LEFT`] of the free bytes in
//!   one device buffer, the plan on device 0 is refused by name with every
//!   term — the card, its free and usable bytes, the dense trunk, KV,
//!   context, scratch, reserves and margin — under no budget and under
//!   `BLOOMERY_CARD_BUDGET` above the free reading (the free term binds
//!   under it), and the unset load's decision is the placed plan on `a`'s
//!   card, its record naming `whole_does_not_fit`; the hold dropped, both
//!   are gone (the reading is live). FAIL-first: a census that reads the
//!   total as the free bytes (mutant: `raw_device_free_bytes` returning
//!   `cuDeviceTotalMem`'s figure) plans where the refusal is expected and
//!   answers `Whole` where the placed decision is — every arm red.
//!
//! - (n) two resident slots (`add_slots`): prompts A and B — two distinct
//!   windows of the prose, [`N_IDS`] ids each, the pass path — [`GEN_W`]
//!   greedy steps a slot. Alone, the load's one sequence: each prompt's ids,
//!   last logits and every layer's K/V rows hashed (FNV-1a 64). Together, a
//!   second sequence resident: `resident_bytes` grows by exactly one
//!   `seq_bytes`, the derived per-sequence KV bytes; the two slots stepped
//!   token-interleaved leave each the bits its alone run left. A reset of
//!   slot 1 between runs does not move slot 0: its next [`N_STEPS`] ids are
//!   its alone run's continuation. A fault raised on one slot refuses the
//!   other's steps naming the slot, and only the faulting slot's own reset
//!   lifts it; the select refusals name themselves; and both slots hold
//!   captures after the interleave, none after a `set_mode` round trip.
//!
//! - (j) one pass of two slots (`GpuModel::step_slots`): prompts A and B
//!   ([`J_A`], [`J_B`], distinct lengths, the pass path) prefilled each in
//!   its own slot, then [`J_ROUNDS`] passes of a row a slot each feeding
//!   back its argmax, a pass of two rows on slot 0 beside one on slot 1,
//!   and a pass of slot 1 alone. (j1) every row's id and logits, and each
//!   slot's position and K/V rows, bit for bit the slot's solo run (slot 0
//!   alone from a reset, the prompt then a step a token) — in graph mode
//!   and again in eager mode. (j2) the captured pass of slots 0 and 1 at a
//!   row each holds the two-row pass of slot 0's nodes plus one slot's
//!   ([`J_SLOT_LAYER_NODES`] a layer and its embedding), one memcpy a row
//!   in each; the one-row pass and the difference from it print, with the
//!   terms the difference holds. (j3) a pass of nine rows, a slot twice, no
//!   slot, a slot out of range, a slot of no token and rows past the cache
//!   refused by name, no position moved. (j4) a fault planted ahead of a
//!   pass of both slots ((x)'s plant) poisons both: each slot and the pass
//!   refused naming the set, slot 0's reset leaves slot 1's standing, both
//!   resets lift it, and the next pass is the solo runs' first step bit for
//!   bit. The last clause on the main model: the model it leaves is
//!   dropped, its second sequence's planes with it. `--one-pass-only` runs
//!   the load and (j).
//!
//! `--gemm-only` runs the load, (u), (t) and (w), `--ubatch-only` (w) alone,
//! `--rope-only` the load and (t), `--fault-only` the load and (x),
//! `--taps-only` the load and (d).
//!
//! The chain runs the tensor-core flash pass, the engine's; the scalar pass
//! runs only eagerly as (u)'s ruler (`set_flash_mma`), and the kernel itself
//! is `gate_qwen3moe_flash`'s. The `load` line names the pass.
//!
//! `--dump DIR` also writes each greedy prompt's tokens (u32 LE) and last
//! logits (f32 LE) of the graph, the eager and the prefilled pass as raw
//! files under DIR (`p{i}-{graph,eager,prefill}.{tokens,logits}`), and each
//! (u) prompt's GEMM prefill — every layer's K/V rows (u16 LE, layer after
//! layer as `kv_rows` returns them), the last logits and the greedy tokens
//! after it (`u{n}-gemm.{kv,logits,tokens}`) — and the same three of its
//! one-token run (`u{n}-step.{kv,logits,tokens}`), for a byte comparison
//! across builds (`md5sum DIR/*`). With `--gemm-only` only the (u) files
//! are written.
//!
//! `--ppl TAG [--placed]` instead scores the chain — with `--placed` (o)'s
//! placed load — against ik's KL-divergence base file
//! `$BLOOMERY_DATA/ikppl/TAG.kld` (`tools/ref/ik-ppl.sh --kld-base`): chunk by
//! chunk from a reset, one step per id, at every scored position our NLL,
//! ik's, KL(ik‖ours) and the top-1 agreement. Printed, not judged.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_qwen3moe_e2e: built without the `gpu` feature; see `just gate-gpu-qwen3moe-e2e`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen3moe_e2e", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_taps.rs"]
mod taps;

#[cfg(feature = "gpu")]
#[path = "shared/flash_grid.rs"]
mod flash_grid;

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_place.rs"]
#[allow(
    dead_code,
    reason = "the gate plans and opens a qwen3moe file only; the other half serves the CLI and the seat"
)]
mod q3place;

#[cfg(feature = "gpu")]
mod gate {
    use super::flash_grid::flash_grids;
    use super::q3place::{self, PlaceQ3};
    use super::taps;
    use app::Session;
    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu::arch::qwen3moe::ubatch::UBATCH;
    use bloomery_gpu::arch::qwen3moe::{Body, KvQ8, KvQ8Host, PrefillPath};
    use bloomery_gpu::flash_gqa::HEAD;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::{Gpu, GpuError, GpuModel, Qwen3moeModel, Slots};
    use bloomery_gpu_gates::generate::Place;
    use bloomery_gpu_gates::kld::{KldBase, PplModel, score_ppl};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::prompts::{
        GreedyClass, GreedyRow, PromptRow, compare_greedy, read_greedy, read_prompts,
    };
    use bloomery_gpu_gates::{
        Fnv1a64, GateError, Layout, RefManifest, RowKind, bits_equal, checks_failed, data_dir,
        ik_q8_2, open_split, q8_1_dequant, ref_ints, ref_model_path, ref_tensor_logical_in,
        topk_ids_logical_within, verdict,
    };
    use bloomery_levers::HostCfg;
    use cuda_core::sys;
    use model::arch::Arch;
    use model::arch::qwen3moe::hparams::Hparams;
    use model::placement::PlanLevers;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    /// Generated tokens per prompt: the greedy files' width.
    const GEN: usize = 32;

    /// Where the split prefill of the concatenated prompts cuts them: inside
    /// the first pass, so the second call starts off a pass boundary.
    const SPLIT: usize = 5;

    /// Prompts of the greedy arm: rows `0..PROMPTS` of `tools/ref/prompts.tsv`.
    const PROMPTS: usize = 8;

    /// Cache rows: the longest of [`LONG`] plus `GEN`, with room (the
    /// oracle's five tokens and the prompts' concatenation fit far below).
    const CTX: usize = 1344;

    /// The GEMM clause's prompt lengths: at the default ubatch size (4,096,
    /// its arena clipped to the [`CTX`]-row cache) each is one ubatch.
    const LONG: [usize; 2] = [1025, 1300];

    /// Where the split GEMM prefill of the longer prompt cuts it: off every
    /// ubatch boundary, so both calls end in a partial ubatch.
    const LONG_SPLIT: usize = 700;

    /// The prose the GEMM clause reads, and its digest as
    /// `tools/ref/models/qwen3moe.sh` pins it.
    const PROSE: &str = "corpus-prose.ids";
    const PROSE_SHA256: &str = "9444bc5b4e2a7c5f4caac4f1aa7b6ef8e945b114fdecac52aca36a1de4e76cf6";

    /// PIN(2026-09-25): the GEMM prefill's band on layer 0's K and V rows
    /// against the one-token path's (`‖ours − one-token‖ / ‖one-token‖`).
    /// Layer 0's input is the same bits on both paths (the embedding rows;
    /// `rms_norm` then the quantizer, the bytes `norm_quant` writes), so only
    /// the k and v projections differ, each within its own sum's rounding of
    /// the exact dot of the same codes: the GEMM's `γ(nb + 4) · Σ_b (|D_b| +
    /// |M_b|)` (gate_gemm), the gemv's lane partials and warp tree the same
    /// order, about 1.2e-6 of `Σ_b |terms|` each at K = 2048, which is about
    /// √16 = 4 times a value's size for blocks of either sign: Δ ≤ 1e-5 of the
    /// row. The head norm and the rope carry that relative distance; the f16
    /// rounding then moves a value by one ulp (at most 2^-10 of it) with
    /// probability Δ/ulp, so the rows' distance is at most √(Δ · ulp) ≈
    /// √(1e-5 · 1e-3) = 1e-4. A value much smaller than its row can move by
    /// several of its own ulps under the same absolute Δ, so the count of
    /// moved values and their largest move in ulp are printed, not judged.
    const GEMM_L0_REL: f64 = 1e-4;

    /// PIN(2026-09-25): the GEMM prefill's K/V distance from the one-token
    /// path at every layer past 0, over the one-token path's own distance
    /// between its two gated flash arithmetics on the same prompt (the
    /// tensor-core pass's f16 scores against the scalar pass's f32), each
    /// layer's `max(K, V)` over `max(K, V)`; and the same ratio of the last
    /// logits' relative distances. Derivation: past layer 0 the two
    /// paths' first difference (the products' sum order, about 1e-6 of the
    /// terms, [`GEMM_L0_REL`]'s note) grows at every q8_1 re-quantization by
    /// code flips — a value crosses a rounding boundary with probability
    /// δ/d8 and then moves by d8, so δ' ≈ √(δ · d8/σ) with d8/σ ≈ 2.3e-2 —
    /// until, within two or three layers, the two paths round like two
    /// independent roundings of one rule. From there the distance is the
    /// model's own amplification of rounding-sized differences, which no
    /// isotropic model gives (an isotropic `√(l + 1) · 4e-2`, the forced
    /// arm's per-layer term composed as [`FREE_BAND`] is, undercounts layers
    /// 34-37, whose V moves 0.2-0.3 under either perturbation). So the band
    /// is a ratio against a same-class perturbation measured in the same
    /// process, the teacher-forced arm's method: the flash arithmetics'
    /// distance starts later (the scores are first rounded in layer 0's
    /// attention) and smaller per layer. The GEMM path's attention is the
    /// prefill flash in either arm (`flash_gqa_prefill`: the tensor-core
    /// scores, f16 weights), so from layer 0's attention on its difference
    /// also carries that arithmetic's against the arm's decode flash. A
    /// correct GEMM path's K/V rows read below 1.9 on these prompts under
    /// either flash pass, highest on layers 34-37, and its logits ratio below
    /// 2.8 (highest in the scalar arm, where the prefill flash's f16
    /// arithmetic stands against the scalar decode flash); a wiring fault — a table built from another layer's or token's
    /// ids, weights one slot off, positions off by one — reads an error of
    /// order one against a spread of 1e-3 to 2.5e-1, a K/V ratio above 3.7
    /// on every layer it reaches. The logits ratio alone does not separate
    /// every fault (positions off by one leave it near 1); the K/V rows do.
    const GEMM_SPREAD_RATIO: f64 = 3.0;

    /// PIN(2026-09-25): the captured step's node count, derived before the
    /// chain was built: the embedding row, 12 nodes per layer (attention
    /// norm+quant, q·k·v, QK-norm+rope+append, flash segment pass, flash
    /// merge, q8_1 of the attention rows, attn_output with the residual, the
    /// FFN norm with the router gemv and the routing, gate·up·SwiGLU, q8_1 of
    /// the SwiGLU rows, down, combine into the next layer's input), one more
    /// on each of the 24 layers whose value projection is Q6_K (its own
    /// gemv), and the head's three (norm, q8_1, the Q6_K gemv with the argmax
    /// folded in): 1 + 24·12 + 24·13 + 3. Was 508: the two q8_1 launches
    /// were folded into attn_output and gate·up, and decode ran slower —
    /// every attn_output block re-quantized the attention row, every gate·up
    /// block fenced and drew a ticket (rig-log `#qwen3fuse-regression-nsys`).
    const NODES_CHAIN: usize = 604;

    /// Memcpy nodes in the captured step: the residual crosses no layer
    /// boundary as a copy.
    const MEMCPY_CHAIN: usize = 0;

    /// PIN(2026-09-25): the captured prefill pass's node count at one token,
    /// derived from the pass's launches before the graphs were built: the
    /// decode step's chain without its head, 1 + 24·12 + 24·13. Was 505,
    /// with the two q8_1 launches of [`NODES_CHAIN`]'s note folded.
    const NODES_PASS_1: usize = 601;

    /// PIN(2026-09-25): the captured prefill pass's node count at every `m`
    /// from two to `MAX_TOKENS`, derived the same way: the embedding rows,
    /// then per layer the attention half's 7 (norm+quant, q·k·v,
    /// QK-norm+rope+append, flash segment pass, flash merge, the attention
    /// rows' q8_1, attn_output with the residual) plus 2 on each of the 24
    /// layers whose value projection is Q6_K (its gemv, the token-major
    /// copy), and the FFN half's 6 (norm+quant, the m-token router,
    /// gate·up·SwiGLU, one q8_1 over every token's slots, one down `_sel`
    /// over every token's slots, the combine): 1 + 24·13 + 24·15. Every
    /// launch covers all of the pass's tokens, so the count does not grow
    /// with m. The head after the last pass runs eager.
    const NODES_PASS_M: usize = 673;

    /// PIN(2026-09-24): the teacher-forced bound on each half's error ratio
    /// — its relative error over the relative distance between the two
    /// sides' 8-bit inputs to that half (`quant_gap`: ours q8_1 per 128
    /// values against ik's q8_2 per 32, at the attention's q/k/v and
    /// attn_output inputs, the FFN's gate·up and down inputs, in
    /// quadrature). To first order a linear map carries its input's relative
    /// perturbation to its output unchanged, so a correct chain reads about
    /// 1; measured on this set: median about 1, worst 2.8 in attention (the
    /// softmax sharpens a score error) and 6.1 in the FFN (a layer whose
    /// input has one dominant channel, where the isotropic prediction
    /// undercounts). The kernel gates bound each kernel exactly; this band
    /// only has to separate that class from a wiring fault, which reads an
    /// error of order one over a gap of about 2e-2.
    const RATIO_BAND: f64 = 10.0;

    /// PIN(2026-09-24): the free-running bound on a layer output's relative
    /// distance. Derivation: the forced arm's per-layer errors (up to about
    /// 4e-2 of a layer's update, whose norm is of the residual's order)
    /// added in quadrature over 48 layers, independent: √48 · 4e-2 ≈ 0.28.
    const FREE_BAND: f64 = 0.28;

    /// PIN(2026-09-21): the V2-Lite gate's floor — a first difference where
    /// the reference's own top1-top2 margin is below this is a near-tie
    /// re-lottery, not a fault.
    const MARGIN_FLOOR: f32 = 0.5;

    fn rel(a: &[f32], b: &[f32], base: impl Fn(usize) -> f64) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += base(i).powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// ik's `name` rows, `k` values each.
    fn rows(man: &RefManifest, name: &str, k: usize) -> Result<Vec<Vec<f32>>, GateError> {
        let row = man.tensor(name, 0)?;
        let v = ref_tensor_logical_in(&man.dir, row)?;
        if row.ne[0] as usize != k || v.len() % k != 0 {
            return Err(format!("{name} is {:?}, want rows of {k}", row.ne).into());
        }
        Ok(v.chunks(k).map(<[f32]>::to_vec).collect())
    }

    fn open(ctx: usize, mode: StepMode, kv: KvQ8) -> Result<Qwen3moeModel, GateError> {
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let t = Instant::now();
        let mut m = Qwen3moeModel::open(Gpu::new()?, file, Qwen3moeModel::lever_opts(ctx, kv)?)?;
        m.set_mode(mode);
        let mma = m.body("gate_qwen3moe_e2e")?.flash_mma();
        println!(
            "load resident_bytes={} ctx={ctx} layers={} flash_mma={mma} ubatch={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            m.layers().len(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        Ok(m)
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        let host = bloomery_levers::at_main(&[])?.host();
        let args: Vec<String> = std::env::args().collect();
        if let Some(i) = args.iter().position(|a| a == "--ppl") {
            let tag = args.get(i + 1).ok_or("--ppl needs a tag")?;
            let placed = args.iter().any(|a| a == "--placed").then_some(host);
            return ppl(tag, placed);
        }
        let dump = match args.iter().position(|a| a == "--dump") {
            Some(i) => Some(PathBuf::from(
                args.get(i + 1).ok_or("--dump needs a directory")?,
            )),
            None => None,
        };
        if args.iter().any(|a| a == "--placed-only") {
            let ok = placed(host)?;
            println!("gate_qwen3moe_e2e --placed-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--memguard-only") {
            let ok = memguard()?;
            println!("gate_qwen3moe_e2e --memguard-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--ubatch-only") {
            let ok = ubatch_sizes()?;
            println!("gate_qwen3moe_e2e --ubatch-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        let mut m = open(CTX, StepMode::Graph, KvQ8::F16)?;
        if args.iter().any(|a| a == "--fault-only") {
            let ok = fault_layer(&mut m)?;
            println!("gate_qwen3moe_e2e --fault-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--rope-only") {
            let ok = rope_table(&mut m, CTX)?;
            println!("gate_qwen3moe_e2e --rope-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--taps-only") {
            let ok = tap_dump(&mut m)?;
            println!("gate_qwen3moe_e2e --taps-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--one-pass-only") {
            let ok = one_pass_two_slots(&mut m)?;
            println!("gate_qwen3moe_e2e --one-pass-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--q8-only") {
            drop(m);
            let ok = q8_cache()?;
            println!("gate_qwen3moe_e2e --q8-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--gemm-only") {
            let mut ok = gemm_prefill(&mut m, dump.as_deref())?;
            ok &= rope_table(&mut m, CTX)?;
            drop(m);
            ok &= ubatch_sizes()?;
            println!("gate_qwen3moe_e2e --gemm-only: {}", verdict(ok));
            if !ok {
                return Err(checks_failed());
            }
            return Ok(());
        }
        let mut ok = true;
        ok &= structure(&mut m)?;
        let mut s = Session::from_model(m, u32::try_from(CTX)?);
        ok &= clear(&mut s)?;
        let m = s.model_mut();
        let o = oracle::for_arch(Arch::Qwen3moe)?;
        let man = o.open(Set::Cpu)?;
        ok &= forced(m, &man)?;
        ok &= free(m, &man)?;
        ok &= tap_dump(m)?;
        ok &= greedy(m, dump.as_deref(), true)?;
        ok &= rope_table(m, CTX)?;
        ok &= fault_layer(m)?;
        ok &= slots_two(m)?;
        ok &= one_pass_two_slots(m)?;
        drop(s);
        ok &= ubatch_sizes()?;
        ok &= placed(host)?;
        ok &= q8_cache()?;
        println!("gate_qwen3moe_e2e: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ---------------------------------------------------------- (k) the clear

    /// The clear clause's arms, as (first id, ids, steps) of the prose: A one
    /// GEMM ubatch, B one pass after A's ids.
    const CLEAR_A: (usize, usize, usize) = (0, 600, 4);
    const CLEAR_B: (usize, usize, usize) = (600, 5, 4);

    /// What one arm of (k) leaves: every field but its time.
    struct ArmOut {
        /// The position the arm starts at: 0 after a clear.
        pos0: u32,
        /// The prefill's plan, as its units print.
        plan: String,
        /// The prefill's argmax, then each step's.
        tokens: Vec<u32>,
        /// FNV-1a 64 of the last logits' bits, and of every layer's K/V rows
        /// below the arm's end.
        logits: u64,
        kv: u64,
        /// The last ubatch image written: its tokens and bytes.
        image: Option<(usize, usize)>,
    }

    /// (k) (module doc).
    fn clear(s: &mut Session<Body>) -> Result<bool, GateError> {
        let (a, b) = (CLEAR_A, CLEAR_B);
        let prose = prose((a.0 + a.1).max(b.0 + b.1))?;
        let arms = [a, b, a];
        let want = ["gemm", "prefill", "gemm"];
        for (arm, kind) in arms.iter().zip(want) {
            let plan = s.model().prefill_plan(arm.1, PrefillPath::Auto)?;
            if plan.kind() != kind {
                return Err(format!(
                    "(k): {} ids plan {plan} ({}); the clause wants the {kind} path",
                    arm.1,
                    plan.kind()
                )
                .into());
            }
        }
        let t = Instant::now();
        let mut outs: Vec<ArmOut> = Vec::with_capacity(arms.len());
        s.arms(&arms, |s, i, &(first, n, steps)| {
            let t = Instant::now();
            let m = s.model_mut();
            let pos0 = m.pos();
            let plan = m.prefill_plan(n, PrefillPath::Auto)?.to_string();
            let mut tokens = vec![m.prefill_with(&prose[first..first + n], PrefillPath::Auto)?];
            for _ in 0..steps {
                let last = *tokens.last().ok_or("no token")?;
                tokens.push(m.step(&[last])?);
            }
            let logits = Fnv1a64::default().f32s(&m.logits()?).value();
            let end = usize::try_from(m.pos())?;
            let kv = m
                .kv_rows(end)?
                .iter()
                .flatten()
                .fold(Fnv1a64::default(), |h, v| h.bytes(&v.to_le_bytes()))
                .value();
            let image = m.ubatch_prologue()?.last.map(|w| (w.tokens, w.bytes));
            println!(
                "clear arm {i} of {}: prose ids {first}..{} ({plan}) + {steps} steps from \
                 position {pos0}: tokens {tokens:?} logits fnv64 {logits:016x} kv fnv64 \
                 {kv:016x} image {image:?} | {:.2} s (runtime value)",
                arms.len(),
                first + n,
                t.elapsed().as_secs_f64()
            );
            outs.push(ArmOut {
                pos0,
                plan,
                tokens,
                logits,
                kv,
                image,
            });
            Ok::<(), GateError>(())
        })
        .map_err(|f| Box::new(f) as GateError)?;
        let [first, _, again] = outs.as_slice() else {
            return Err(format!("(k): {} arms of 3 ran", outs.len()).into());
        };
        let off: Vec<&str> = [
            ("pos0", again.pos0 == first.pos0),
            ("plan", again.plan == first.plan),
            ("tokens", again.tokens == first.tokens),
            ("logits", again.logits == first.logits),
            ("kv", again.kv == first.kv),
            ("image", again.image == first.image),
        ]
        .into_iter()
        .filter_map(|(name, same)| (!same).then_some(name))
        .collect();
        let ok = off.is_empty();
        println!(
            "clear: arm 2 (the GEMM arm after the pass arm and the clear) against arm 0 (the \
             GEMM arm right after the load): {}; {:.2} s (runtime value) {}",
            if ok {
                "every field but the time equal".to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            t.elapsed().as_secs_f64(),
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (x) a fault's layer

    /// The layer clause (x) plants its fault in.
    const FAULT_LAYER: usize = 13;

    /// Raise the fault word in layer [`FAULT_LAYER`]: its FFN half run alone
    /// on a finite row with one NaN. Nothing reads the word back, so the next
    /// call that does is the one it poisons.
    fn plant_fault(m: &mut Qwen3moeModel) -> Result<(), GateError> {
        let hidden = m.body("gate_qwen3moe_e2e")?.hparams().n_embd;
        let mut x = vec![0.25f32; hidden];
        x[5] = f32::NAN;
        m.step_ffn(FAULT_LAYER, &x)?;
        Ok(())
    }

    /// (x) (module doc).
    fn fault_layer(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        m.reset()?;
        let before = m.gpu().fault()?;
        plant_fault(m)?;
        let raised = m.gpu().fault()?;
        let step = m.step(&[1]);
        let poisoned = m.poisoned();
        m.reset()?;
        let after = m.gpu().fault()?;
        let clean_step = m.step(&[1]).is_ok();
        m.reset()?;
        let layer = u32::try_from(FAULT_LAYER)?;
        let named = match &step {
            Err(GpuError::Fault { fault, .. }) => {
                Some(*fault) == raised && fault.layer == layer && fault.site().is_some()
            }
            _ => false,
        };
        let pass = before.is_none() && named && poisoned == raised && after.is_none() && clean_step;
        println!(
            "fault layer={FAULT_LAYER} ffn input with a NaN: word before {before:?}, the next step {} (want a \
             fault at layer {FAULT_LAYER}), poisoned {poisoned:?}, after reset {after:?} and a clean step \
             {clean_step} {}",
            match &step {
                Ok(t) => format!("returned token {t}"),
                Err(e) => format!("returned \"{e}\""),
            },
            verdict(pass)
        );
        Ok(pass)
    }

    // ------------------------------------------------------- (s) structure

    fn structure(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        let nodes = m.capture_step()?;
        let ([kernel, memcpy, memset, host], other) = count_kinds(
            &m.step_graph_nodes()?,
            [
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMSET,
                sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_HOST,
            ],
        );
        let mut pass = nodes == NODES_CHAIN && memcpy == MEMCPY_CHAIN && host == 0;
        println!(
            "structure graph_nodes={nodes} (want {NODES_CHAIN}) kernel={kernel} memcpy={memcpy} \
             (want {MEMCPY_CHAIN}) memset={memset} host={host} (want 0) other={other} {}",
            verdict(pass)
        );
        // The flash launches' grids, whatever the cache height: the segment
        // pass `m · n_kv · SEGMENTS` blocks a layer, the merge `m · n_head`,
        // one of each a layer; the line prints each kind's distinct grids and
        // node count as the graph holds them. The entry names print when they
        // are not the tensor-core ones the load picks.
        let hp = m.body("gate_qwen3moe_e2e")?.hparams();
        let (n_kv, n_head, n_layer) = (hp.n_head_kv, hp.n_head, hp.n_layer);
        let names = ["gqa_flash_seg_mma", "gqa_flash_merge"];
        let (flash_ok, got) =
            flash_grids(&m.step_graph_nodes()?, (1, n_kv, n_head), n_layer, names)?;
        pass &= flash_ok;
        println!("structure flash grid m=1 {got} {}", verdict(flash_ok));
        let t = Instant::now();
        let counts = m.capture_prefill()?;
        println!(
            "structure prefill: {} passes captured in {:.1} ms (runtime value)",
            counts.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        for (i, &nodes) in counts.iter().enumerate() {
            let rows = i + 1;
            let want = if rows == 1 {
                NODES_PASS_1
            } else {
                NODES_PASS_M
            };
            let list = m.prefill_graph_nodes(rows)?;
            let ([kernel], other) =
                count_kinds(&list, [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL]);
            let ok = nodes == want && kernel == nodes;
            println!(
                "structure prefill m={rows} graph_nodes={nodes} (want {want}) kernel={kernel} \
                 other={other} (want 0) {}",
                verdict(ok)
            );
            pass &= ok;
            let (flash_ok, got) = flash_grids(&list, (rows, n_kv, n_head), n_layer, names)?;
            println!(
                "structure prefill flash grid m={rows} {got} {}",
                verdict(flash_ok)
            );
            pass &= flash_ok;
        }
        Ok(pass)
    }

    // ------------------------------------------------------ (f) forced

    /// ik's rows of `name` at every layer, and the offset of its first row
    /// in the prefill's positions (the last layer keeps only the output
    /// token's rows past its attention).
    fn per_layer(
        man: &RefManifest,
        stem: &str,
        n_layer: usize,
        k: usize,
    ) -> Result<Vec<Vec<Vec<f32>>>, GateError> {
        (0..n_layer)
            .map(|l| rows(man, &format!("{stem}-{l}"), k))
            .collect()
    }

    /// The row of `v` (a layer's rows) for prefill position `t` of `t_n`, if
    /// the layer kept it.
    fn at(v: &[Vec<f32>], t: usize, t_n: usize) -> Option<&[f32]> {
        t.checked_sub(t_n - v.len()).map(|i| v[i].as_slice())
    }

    /// `‖x̂_ours − x̂_ik‖ / ‖x‖` over the `k`-value rows of `x`: how far apart
    /// the two sides' 8-bit activations of the same input sit (ours q8_1 per
    /// 128 values, ik q8_2 per 32).
    fn quant_gap(x: &[f32], k: usize) -> f64 {
        let o = q8_1_dequant(x, k, x.len() / k);
        let i = ik_q8_2::reconstruct(x);
        rel(&o, &i, |j| f64::from(x[j]))
    }

    fn forced(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("forced")?.hparams().clone();
        let (h, n_layer, n_exp) = (hp.n_embd, hp.n_layer, hp.experts.n_expert as u32);
        let embd = rows(man, "inp_embd", h)?;
        let t_n = embd.len();
        let ik_out = per_layer(man, "l_out", n_layer, h)?;
        let ik_attn = per_layer(man, "attn_out", n_layer, h)?;
        let ik_routed = per_layer(man, "routed_out", n_layer, h)?;
        let ik_anorm = per_layer(man, "attn_norm", n_layer, h)?;
        let ik_fnorm = per_layer(man, "ffn_inp_normed", n_layer, h)?;
        let q_len = hp.n_head * hp.head_dim;
        let ik_fa: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| rows(man, &format!("fa-{l} (reshaped)"), q_len))
            .collect::<Result<_, _>>()?;
        let ff = hp.experts.ff;
        let ik_par: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let slots = rows(man, &format!("ffn_moe_gate_par-{l}"), ff)?;
                Ok(slots
                    .chunks(hp.experts.n_used)
                    .map(<[Vec<f32>]>::concat)
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        // The routed sum's terms: each slot's down output times its weight,
        // summed in magnitude — the scale the FFN's error is measured on,
        // since the eight terms can cancel in `routed_out`.
        let n_used = hp.experts.n_used;
        let ik_mag: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|l| {
                let down = rows(man, &format!("ffn_moe_down-{l}"), h)?;
                let w = ref_tensor_logical_in(
                    &man.dir,
                    man.tensor(&format!("ffn_moe_weights_norm-{l}"), 0)?,
                )?;
                Ok(down
                    .chunks(n_used)
                    .zip(w.chunks(n_used))
                    .map(|(d, w)| {
                        (0..h)
                            .map(|i| (0..n_used).map(|s| (w[s] * d[s][i]).abs()).sum::<f32>())
                            .collect()
                    })
                    .collect())
            })
            .collect::<Result<_, GateError>>()?;
        let mut ik_ids = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let trow = man.tensor(&format!("ffn_moe_topk-{l}"), 0)?;
            let ids = topk_ids_logical_within(man, trow, n_exp)?;
            let n_used = hp.experts.n_used;
            ik_ids.push(ids.chunks(n_used).map(<[i32]>::to_vec).collect::<Vec<_>>());
        }
        m.reset()?;
        let mut ok = true;
        let (mut w_up, mut w_attn, mut w_ffn) = (0.0f64, 0.0f64, 0.0f64);
        let (mut w_attn_r, mut w_ffn_r) = (0.0f64, 0.0f64);
        let (mut flips, mut sites) = (0usize, 0usize);
        for l in 0..n_layer {
            let (mut l_up, mut l_attn, mut l_ffn, mut l_flip) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut l_attn_r, mut l_attn_p, mut l_ffn_r, mut l_ffn_p) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for t in 0..t_n {
                let x_in: &[f32] = if l == 0 {
                    &embd[t]
                } else {
                    at(&ik_out[l - 1], t, t_n).ok_or("a layer input row is missing")?
                };
                let run = m.step_layer(l, x_in, u32::try_from(t)?)?;
                if let Some(a) = at(&ik_attn[l], t, t_n) {
                    let got: Vec<f32> =
                        run.ffn_inp.iter().zip(x_in).map(|(&f, &x)| f - x).collect();
                    let e = rel(&got, a, |i| f64::from(a[i]));
                    let g_in = at(&ik_anorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                    let g_fa = at(&ik_fa[l], t, t_n).map_or(0.0, |x| quant_gap(x, q_len));
                    let pred = g_in.hypot(g_fa);
                    l_attn = l_attn.max(e);
                    l_attn_r = l_attn_r.max(e / pred.max(f64::MIN_POSITIVE));
                    l_attn_p = l_attn_p.max(pred);
                }
                let (Some(want), Some(a), Some(r)) = (
                    at(&ik_out[l], t, t_n),
                    at(&ik_attn[l], t, t_n),
                    at(&ik_routed[l], t, t_n),
                ) else {
                    continue;
                };
                l_up = l_up.max(rel(&run.l_out, want, |i| {
                    f64::from(want[i]) - f64::from(x_in[i])
                }));
                // The FFN half alone, on ik's own FFN input.
                let ik_inp: Vec<f32> = a.iter().zip(x_in).map(|(&a, &x)| a + x).collect();
                let f = m.step_ffn(l, &ik_inp)?;
                let got: Vec<f32> = f.l_out.iter().zip(&ik_inp).map(|(&o, &i)| o - i).collect();
                let theirs = &ik_ids[l][ik_ids[l].len() - (t_n - t)];
                let flipped = f
                    .ids
                    .iter()
                    .filter(|&&e| !theirs.contains(&(e as i32)))
                    .count();
                sites += 1;
                let mag = at(&ik_mag[l], t, t_n).ok_or("a routed magnitude row is missing")?;
                let e = rel(&got, r, |i| f64::from(mag[i]));
                let g_in = at(&ik_fnorm[l], t, t_n).map_or(0.0, |x| quant_gap(x, h));
                let g_par = at(&ik_par[l], t, t_n).map_or(0.0, |x| quant_gap(x, ff));
                let pred = g_in.hypot(g_par);
                l_ffn_p = l_ffn_p.max(pred);
                if flipped == 0 {
                    l_ffn_r = l_ffn_r.max(e / pred.max(f64::MIN_POSITIVE));
                }
                if flipped > 0 {
                    flips += 1;
                    l_flip += 1;
                    println!(
                        "forced layer={l} pos={t}: {flipped} router id(s) differ from ik's, ffn_rel={e:.3e} (printed)"
                    );
                } else {
                    l_ffn = l_ffn.max(e);
                }
            }
            println!(
                "forced layer={l} attn_rel={l_attn:.3e} (quant gap {l_attn_p:.3e}, worst ratio {l_attn_r:.2}) \
                 ffn_rel={l_ffn:.3e} (quant gap {l_ffn_p:.3e}, worst ratio {l_ffn_r:.2}) \
                 update_rel={l_up:.3e} flip_sites={l_flip}"
            );
            w_up = w_up.max(l_up);
            w_attn = w_attn.max(l_attn);
            w_ffn = w_ffn.max(l_ffn);
            w_attn_r = w_attn_r.max(l_attn_r);
            w_ffn_r = w_ffn_r.max(l_ffn_r);
            if l_attn_r > RATIO_BAND || l_ffn_r > RATIO_BAND {
                ok = false;
                println!("forced layer={l}: an error ratio passes {RATIO_BAND} FAIL");
            }
        }
        println!(
            "forced: {n_layer} layers x {t_n} positions; attention half worst rel {w_attn:.3e}, ratio \
             {w_attn_r:.2}; FFN half on ik's input worst rel {w_ffn:.3e}, ratio {w_ffn_r:.2} over {} \
             sites without a routing difference (band ratio {RATIO_BAND}); {flips} sites route an \
             expert ik does not (printed); whole-layer update worst {w_up:.3e} (printed) {}",
            sites - flips,
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------------- (c) free-running

    fn free(m: &mut Qwen3moeModel, man: &RefManifest) -> Result<bool, GateError> {
        let hp = m.body("free")?.hparams().clone();
        let (h, n_layer) = (hp.n_embd, hp.n_layer);
        let toks: Vec<u32> = ref_ints(man, "inp_tokens", 0, RowKind::Input, Layout::Flat)?
            .iter()
            .map(|&i| u32::try_from(i))
            .collect::<Result<_, _>>()?;
        let mut ik_out = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            ik_out.push(rows(man, &format!("l_out-{l}"), h)?);
        }
        let ik_logits = rows(man, "result_output", hp.n_vocab)?;
        m.set_layer_taps(true)?;
        m.reset()?;
        let t_n = toks.len();
        let mut per_layer = vec![0.0f64; n_layer];
        let mut last = 0u32;
        for (t, &tok) in toks.iter().enumerate() {
            last = m.step(&[tok])?;
            let taps = m.layer_taps()?;
            for l in 0..n_layer {
                let off = t_n - ik_out[l].len();
                let Some(t_ik) = t.checked_sub(off) else {
                    continue;
                };
                let want = &ik_out[l][t_ik];
                per_layer[l] = per_layer[l].max(rel(&taps[l], want, |i| f64::from(want[i])));
            }
        }
        let logits = m.logits()?;
        m.set_layer_taps(false)?;
        let want = ik_logits.last().ok_or("result_output has no row")?;
        let ik_top = argmax(want);
        let lrel = rel(&logits, want, |i| f64::from(want[i]));
        let worst = per_layer.iter().copied().fold(0.0f64, f64::max);
        let ok = worst <= FREE_BAND && last == ik_top;
        for (l, &e) in per_layer.iter().enumerate() {
            if l % 8 == 0 || l == n_layer - 1 || e > FREE_BAND {
                println!("free layer={l} l_out_rel={e:.3e}");
            }
        }
        println!(
            "free: {t_n} tokens {toks:?}, worst l_out_rel={worst:.3e} (band {FREE_BAND:.0e}); last \
             position argmax ours={last} ik={ik_top} logits_rel={lrel:.3e} (printed) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ----------------------------------------------- (m) the memory guard

    /// What (m) leaves free of the card while it holds the rest: well under
    /// the trunk's smallest term, so the refusal is certain.
    const MEMGUARD_LEFT: usize = 512 << 20;

    /// (m) (module doc), no model loaded: the census's free reading, the
    /// quiet-card whole decision, the held-card refusal with every term
    /// under no budget and under a budget above free, the placed decision,
    /// the auto record's why and `card_free`, and the live reading once the
    /// hold is dropped.
    fn memguard() -> Result<bool, GateError> {
        use bloomery_gpu_gates::gpu_census::census;
        use model::placement::workstation::census_usable;
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let d0 = census()?
            .into_iter()
            .find(|d| d.ordinal == 0)
            .ok_or("the census reads no device 0")?;
        let usable = census_usable(d0.total_bytes);
        println!(
            "memguard census: cuda0 {} free {} B of usable {usable} B",
            d0.name, d0.free_bytes
        );
        let mut ok = d0.free_bytes <= usable;
        // The quiet card: the unset load is today's whole-card one.
        let whole = matches!(
            q3place::unplaced_qwen3(&file, CTX, KvQ8::F16)?,
            q3place::Unplaced::Whole
        );
        println!("memguard quiet: whole decision {whole} {}", verdict(whole));
        ok &= whole;
        // The auto record's shape, planned explicitly under (o)'s budget:
        // its why word and the free bytes it names.
        let q = PlaceQ3::qwen3(&file, Place::A.on_host()?, CTX, KvQ8::F16)?;
        let plan = q.plan(
            CTX,
            &PlanLevers {
                card_budget_bytes: Some(PLACED_BUDGET),
            },
        )?;
        let line = q.record(&plan, Some(q3place::WHY_NOT_WHOLE)).line();
        let record_ok = line.contains("why=whole_does_not_fit") && line.contains("card_free=");
        println!("memguard record: {record_ok} {line}");
        ok &= record_ok;
        // Held: all but MEMGUARD_LEFT of the free bytes in one buffer on
        // this context (our own pid, which the holder list never names).
        let gpu = Gpu::new()?;
        let (free, _) = gpu.mem_info()?;
        let hold = free
            .checked_sub(MEMGUARD_LEFT)
            .ok_or("the card had less free than the clause leaves")?;
        let held = cuda_core::DeviceBuffer::<u8>::zeroed(gpu.stream(), hold)?;
        let refusal = |levers: &PlanLevers| -> Result<String, GateError> {
            let q = PlaceQ3::qwen3(&file, Place::parse("cuda0")?, CTX, KvQ8::F16)?;
            Ok(q.plan(CTX, levers)
                .map_or_else(|e| e.to_string(), |_| String::new()))
        };
        let mut terms = true;
        let budgets = [
            ("no budget", PlanLevers::default()),
            (
                "a budget above free",
                PlanLevers {
                    card_budget_bytes: Some(usable),
                },
            ),
        ];
        for (what, levers) in budgets {
            let text = refusal(&levers)?;
            for part in [
                "the device had",
                " B free of its usable",
                " B at plan time",
                "the plan's dense trunk alone needs",
                " B = dense ",
                " B (the allocator's granules) + KV ",
                " B + context ",
                " B + scratch ",
                " B + reserves ",
                " B + margin ",
            ] {
                terms &= text.contains(part);
                if !text.contains(part) {
                    println!("memguard refusal ({what}): {part:?} missing: {text}");
                }
            }
        }
        println!("memguard held: refusal with every term {terms}");
        ok &= terms;
        // The unset load's decision under the hold: the placed plan.
        let placed = matches!(
            q3place::unplaced_qwen3(&file, CTX, KvQ8::F16)?,
            q3place::Unplaced::Placed(_)
        );
        println!(
            "memguard held: placed decision {placed} {}",
            verdict(placed)
        );
        ok &= placed;
        drop(held);
        drop(gpu);
        // The hold gone, the decision is the whole-card one again: the
        // reading is live, not a constant.
        let whole = matches!(
            q3place::unplaced_qwen3(&file, CTX, KvQ8::F16)?,
            q3place::Unplaced::Whole
        );
        println!(
            "memguard released: whole decision {whole} {}",
            verdict(whole)
        );
        ok &= whole;
        Ok(ok)
    }

    // -------------------------------------------------- (o) the placed load

    /// (o)'s card budget: a 12 GiB card's, under which the file's routed
    /// experts split between the card and the host tier.
    const PLACED_BUDGET: u64 = 12 << 30;
    /// The file planned on device 0 at `ctx` positions under
    /// [`PLACED_BUDGET`], its `plan` record and the split line printed, and
    /// loaded by that plan with its host set as `host` asks; `None` when the
    /// plan leaves no routed expert on the card or none on the host.
    fn open_placed(ctx: usize, host: HostCfg) -> Result<Option<Qwen3moeModel>, GateError> {
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let q = PlaceQ3::qwen3(&file, Place::parse("cuda0")?, ctx, KvQ8::F16)?;
        let levers = PlanLevers {
            card_budget_bytes: Some(PLACED_BUDGET),
        };
        let plan = q.plan(ctx, &levers)?;
        q.record(&plan, None).print();
        let split = plan.host.experts > 0 && plan.cards[0].experts > 0;
        println!(
            "placed plan: card_experts={} host_experts={} (both above 0) {}",
            plan.cards[0].experts,
            plan.host.experts,
            verdict(split)
        );
        if !split {
            return Ok(None);
        }
        let opts = Qwen3moeModel::lever_opts(ctx, KvQ8::F16)?;
        Ok(Some(GpuModel::<Body>::open_placed(
            file, &plan, opts, host,
        )?))
    }

    /// (o) (module doc): the model is dropped before the clause returns.
    fn placed(host: HostCfg) -> Result<bool, GateError> {
        let t = Instant::now();
        let Some(mut m) = open_placed(CTX, host)? else {
            return Ok(false);
        };
        let counts = m
            .body("gate_qwen3moe_e2e")?
            .placed()
            .ok_or("an open_placed load with no placed side")?
            .card_counts();
        println!(
            "placed load resident_bytes={} card experts per layer {counts:?} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        m.set_mode(StepMode::Graph);
        println!("placed capture graph_nodes={}", m.capture_step()?);
        let o = oracle::for_arch(Arch::Qwen3moe)?;
        let man = o.open(Set::Cpu)?;
        let mut ok = free(&mut m, &man)?;
        ok &= greedy(&mut m, None, false)?;
        let stats = m
            .body("gate_qwen3moe_e2e")?
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
        Ok(ok && served)
    }

    // ------------------------------------------ (v) the q8_0 cache arm

    /// The stepped rows the (v) planes clause compares: row 0 of every
    /// layer's K and V, the one row whose values no cache read touched (each
    /// later row's input carries the attention over the rows before it, the
    /// two formats' own error growing from there — printed, not judged).
    const Q8_IDS: usize = 96;

    /// (v) (module doc): the cache lever's `q8_0` arm, on its own models
    /// (each dropped before the clause returns). The budget relation is
    /// host-only and census-free: the plan's KV term at q8_0 is the f16
    /// term's 17/32 exactly (17/16 B a value against 2). The q8 model runs
    /// the prose's first ids: eager steps, the pass prefill of the same ids
    /// and the graph replay all leave the same planes and logits bit for
    /// bit (the arm's own consistency, the (p) and (r) contracts on the q8
    /// path), and row 0's dequantized values sit within each 32-value
    /// block's own quantization step of the f16 model's row 0 (the two
    /// formats round the same f32 append values; a block's `d/2` plus the
    /// f16 rounding, derived at runtime — no measured band). Later rows'
    /// distance to the f16 run prints as the compounding diagnostic. Asking
    /// the q8 model for the scalar pass is refused by name at the call, and
    /// its next step still runs.
    fn q8_cache() -> Result<bool, GateError> {
        let file = open_split(Arch::Qwen3moe, "gate-gpu-qwen3moe-e2e")?;
        let mut ok = q8_budget(&file)?;
        drop(file);
        let ids = prose(Q8_IDS)?;
        let (f16_row0, hp) = {
            let mut m = open(CTX, StepMode::Eager, KvQ8::F16)?;
            let hp = m.body("q8")?.hparams().clone();
            m.reset()?;
            for &id in &ids {
                m.step(&[id])?;
            }
            (m.kv_rows(1)?, hp)
        };
        let mut m = open(CTX, StepMode::Eager, KvQ8::Q8)?;
        m.reset()?;
        let mut step_tok = 0;
        for &id in &ids {
            step_tok = m.step(&[id])?;
        }
        let stepped = m.kv_q8_rows(Q8_IDS)?;
        let logits = Fnv1a64::default().f32s(&m.logits()?).value();
        // The pass prefill of the same ids leaves the stepped run's bits.
        m.reset()?;
        let pass_tok = m.prefill_with(&ids, PrefillPath::Pass)?;
        let pass = m.kv_q8_rows(Q8_IDS)?;
        let pass_logits = Fnv1a64::default().f32s(&m.logits()?).value();
        let bits = pass_tok == step_tok && pass == stepped && logits == pass_logits;
        println!(
            "q8 bits: pass prefill == {Q8_IDS} steps, planes and logits bit for bit {}",
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
        let graph = m.kv_q8_rows(Q8_IDS)?;
        let graph_logits = Fnv1a64::default().f32s(&m.logits()?).value();
        let replay = last == step_tok && graph == stepped && logits == graph_logits;
        println!(
            "q8 replay: {Q8_IDS} captured steps == the eager bits {}",
            verdict(replay)
        );
        ok &= replay;
        // The scalar pass on a q8_0 cache: refused by name at the call that
        // asks for it, in `q8_unserved`'s words, and the model's pass kept —
        // the next eager step runs.
        m.set_mode(StepMode::Eager);
        let named = match m.set_flash_mma(false) {
            Err(GpuError::Shape {
                what: "qwen3moe::set_flash_mma",
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
        // Row 0 against the f16 model's row 0, block by block.
        ok &= q8_row0(&stepped, &f16_row0, &hp)?;
        Ok(ok)
    }

    /// (v)'s budget relation, host-only: the plan's KV term under the two
    /// formats, through qwen3moe's own `KvLayout` and the format's wiring
    /// into the plan (`PlaceQ3::qwen3`) — no pure test reaches either;
    /// `qwen35moe::place`'s `q8_layer_bytes` holds the other family's
    /// layout.
    fn q8_budget(file: &gguf::Split) -> Result<bool, GateError> {
        let levers = PlanLevers::default();
        let f16_kv = PlaceQ3::qwen3(file, Place::parse("cuda0")?, CTX, KvQ8::F16)?
            .plan(CTX, &levers)?
            .cards[0]
            .kv_bytes;
        let q8_kv = PlaceQ3::qwen3(file, Place::parse("cuda0")?, CTX, KvQ8::Q8)?
            .plan(CTX, &levers)?
            .cards[0]
            .kv_bytes;
        let term = q8_kv * 32 == f16_kv * 17;
        println!(
            "q8 budget: plan kv_bytes f16 {f16_kv} q8_0 {q8_kv} (17/32 exactly) {}",
            verdict(term)
        );
        Ok(term)
    }

    /// (v)'s row-0 clause: layer 0's first K and V row (the one row whose
    /// values no cache read touched — every later layer's input carries the
    /// attention over the rows before it, the two formats' own error
    /// compounding from layer 1 on), dequantized from the q8_0 planes and
    /// within each 32-value block's own step of the f16 model's row: the
    /// codes' `d/2` nearest radius, the scale's own f16 rounding on every
    /// code unit, the f16 side's `2^-11` relative rounding on top. The
    /// later layers' worst ratio to the same bound prints as the
    /// compounding diagnostic.
    fn q8_row0(q8: &[KvQ8Host], f16: &[Vec<u16>], hp: &Hparams) -> Result<bool, GateError> {
        const WHAT: &str = "q8_row0";
        let (n_kv, head, rows) = (hp.n_head_kv, hp.head_dim, Q8_IDS);
        if q8.len() != f16.len() || q8.len() != hp.n_layer {
            return Err(format!(
                "{WHAT}: {} layers of K/V rows, {} and {}",
                hp.n_layer,
                q8.len(),
                f16.len()
            )
            .into());
        }
        // The bound each value carries: the codes are taken against the f32
        // scale (id = 127/amax) but dequantized by the scale's own f16 bits,
        // so |deq − v| <= d/2 + 127·d·2^-11 (nearest's d/2, the scale's f16
        // rounding on every code unit), and the f16 side rounds v relatively
        // by 2^-11 with |v| <= |f16|·(1 + 2^-10):
        // |q8 − f16| <= d·(1/2 + 127·2^-11) + (2^-11 + 2^-21)·|f16|.
        let q8_round = 0.5f32 + 127.0 * 2.0f32.powi(-11);
        let f16_round = 2.0f32.powi(-11) + 2.0f32.powi(-21);
        let mut worst = 0.0f32;
        let mut worst_ratio = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        let mut bad = String::new();
        for (l, (q, f)) in q8.iter().zip(f16).enumerate() {
            for (side, (codes, scales)) in [(0, (&q.kq, &q.kd)), (1, (&q.vq, &q.vd))] {
                let (cstride, sstride) = (rows * head / 4, rows * head / 32);
                for h in 0..n_kv {
                    let cs = &codes[h * cstride..][..head / 4];
                    let ds = &scales[h * sstride..][..head / 32];
                    let frow = &f[(side * n_kv + h) * head..][..head];
                    for c in 0..head {
                        let d = gguf::quant::half_to_f32(ds[c / 32]);
                        let code = ((cs[c / 4] >> (8 * (c % 4))) & 0xff) as u8 as i8;
                        let v = f32::from(code) * d;
                        let f = gguf::quant::half_to_f32(frow[c]);
                        let e = (v - f).abs();
                        let bound = d * q8_round + f16_round * f.abs();
                        if e > worst {
                            worst = e;
                            worst_at = (l, side);
                        }
                        worst_ratio = worst_ratio.max(e / bound);
                        if l == 0 && e > bound && bad.is_empty() {
                            bad = format!(
                                "layer {l} side {side} head {h} value {c}: |{v:.6e} − {f:.6e}| \
                                 = {e:.3e} past the block's bound {bound:.3e} (d = {d:.6e})"
                            );
                        }
                    }
                }
            }
        }
        println!(
            "q8 row0: layer 0's first K and V row within its blocks' own bounds (worst |Δ| \
             {worst:.3e} at layer {} side {}); all layers' worst ratio to the same bound \
             {worst_ratio:.2} (the compounding past layer 0, diagnostic)",
            worst_at.0, worst_at.1
        );
        match bad.is_empty() {
            true => Ok(true),
            false => Err(format!("{WHAT}: {bad}").into()),
        }
    }

    // --------------------------------------------------- (d) the tap dump

    /// The tap clause's windows of the prose, their prompt ids and greedy
    /// steps: two passes' worth of prompt (a pass of `MAX_TOKENS`, then a
    /// shorter one) on the plain run's pass path.
    const TAP_SEQS: usize = 2;
    const TAP_PROMPT: usize = 12;
    const TAP_GEN: usize = 6;

    /// (d) (module doc). Leaves the taps off and the model in graph mode; the
    /// dump directory is removed on a pass and kept, named, on a failure.
    fn tap_dump(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        let hidden = m.body("gate_qwen3moe_e2e")?.hparams().n_embd;
        prose(TAP_PROMPT)?;
        let src = data_dir().join("qwen3moe").join(PROSE);
        let seqs = taps::windows(&src, TAP_SEQS, TAP_PROMPT)?;
        let dir =
            std::env::temp_dir().join(format!("gate_qwen3moe_e2e-taps-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        let mut dump = taps::Dump::create(&dir, &ref_model_path()?, hidden)?;
        for s in &seqs {
            dump.seq(m, s, TAP_GEN)?;
        }
        let rows = taps::read_manifest(&dir, hidden)?;
        let mut ok = rows.len() == seqs.len();
        let width = taps::TAPS.len() * hidden;
        for (row, s) in rows.iter().zip(&seqs) {
            let ids: Vec<u32> = std::fs::read(dir.join(&row.ids))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            let file: Vec<f32> = std::fs::read(dir.join(&row.taps))?
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            m.set_layer_taps(true)?;
            m.reset()?;
            let mut differing = Vec::new();
            for (p, &id) in ids.iter().enumerate() {
                m.step(&[id])?;
                let all = m.layer_taps()?;
                for (j, &l) in taps::TAPS.iter().enumerate() {
                    let got = &file[p * width + j * hidden..][..hidden];
                    if !bits_equal(got, &all[l]) {
                        differing.push((p, l));
                    }
                }
            }
            m.set_layer_taps(false)?;
            m.set_mode(StepMode::Graph);
            m.reset()?;
            let mut plain = vec![m.prefill_with(&s.prompt, PrefillPath::Pass)?];
            for _ in 1..TAP_GEN {
                let last = plain[plain.len() - 1];
                plain.push(m.step(&[last])?);
            }
            let (head, tail) = ids.split_at(row.n_prompt.min(ids.len()));
            let ids_ok = row.n_prompt == TAP_PROMPT
                && row.n_total == TAP_PROMPT + TAP_GEN
                && head == s.prompt.as_slice()
                && tail == plain.as_slice();
            let taps_ok = differing.is_empty();
            ok &= ids_ok && taps_ok;
            println!(
                "taps seq={} offset={} positions={}: rows of layers {:?} against the layer taps bit \
                 for bit, {} of {} differ{} {}; ids the prompt, then the plain run's {:?} {}",
                row.seq,
                s.offset,
                row.n_total,
                taps::TAPS,
                differing.len(),
                row.n_total * taps::TAPS.len(),
                differing.first().map_or(String::new(), |(p, l)| format!(
                    " (first at position {p}, layer {l})"
                )),
                verdict(taps_ok),
                plain,
                verdict(ids_ok)
            );
        }
        m.set_layer_taps(false)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        if ok {
            std::fs::remove_dir_all(&dir)?;
        } else {
            println!("taps: the dump is kept in {}", dir.display());
        }
        println!(
            "taps: {} sequences of {TAP_PROMPT} prose ids and {TAP_GEN} greedy steps dumped and read back {}",
            rows.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    fn argmax(v: &[f32]) -> u32 {
        let mut best = 0usize;
        for i in 1..v.len() {
            if v[i] > v[best] {
                best = i;
            }
        }
        best as u32
    }

    // ------------------------------------------------ (g) greedy, (r) replay

    fn greedy_dir() -> PathBuf {
        data_dir().join("qwen3moe").join("greedy")
    }

    /// Our greedy continuation of `ids`: `GEN` tokens, the first the
    /// prompt's own next token, and the last step's logits.
    fn continue_greedy(
        m: &mut Qwen3moeModel,
        ids: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let mut out = Vec::with_capacity(GEN);
        let mut next = m.step(ids)?;
        out.push(next);
        for _ in 1..GEN {
            next = m.step(&[next])?;
            out.push(next);
        }
        Ok((out, m.logits()?))
    }

    /// One prompt's greedy tokens and last logits as raw little-endian files
    /// under `dir` (`--dump`).
    fn dump(
        dir: &Path,
        tag: &str,
        p: usize,
        toks: &[u32],
        logits: &[f32],
    ) -> Result<(), GateError> {
        std::fs::create_dir_all(dir)?;
        let t: Vec<u8> = toks.iter().flat_map(|v| v.to_le_bytes()).collect();
        let l: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("p{p}-{tag}.tokens")), t)?;
        std::fs::write(dir.join(format!("p{p}-{tag}.logits")), l)?;
        Ok(())
    }

    /// (g), (r), (p) and (q) over the prompts, and (u) when `gemm` (a placed
    /// load runs no GEMM ubatch).
    fn greedy(
        m: &mut Qwen3moeModel,
        dump_dir: Option<&Path>,
        gemm: bool,
    ) -> Result<bool, GateError> {
        let dir = greedy_dir();
        let mut prompts = Vec::new();
        let mut reference = Vec::new();
        for p in 0..PROMPTS {
            let pr = read_prompts(&dir.join(format!("prompt{p}.tsv")))?;
            let gr = read_greedy(&dir.join(format!("greedy-ik-cpu-{GEN}-p{p}.tsv")))?;
            let (Some(pr), Some(gr)) = (pr.into_iter().next(), gr.into_iter().next()) else {
                return Err(format!("{}: prompt {p} has no row", dir.display()).into());
            };
            if gr.n_tokens != pr.tokens.len() {
                return Err(format!(
                    "prompt {p}: {} ids, the greedy file's row says {}",
                    pr.tokens.len(),
                    gr.n_tokens
                )
                .into());
            }
            prompts.push(pr);
            reference.push(gr);
        }
        let mut graph = Vec::with_capacity(PROMPTS);
        let mut graph_logits = Vec::with_capacity(PROMPTS);
        m.set_mode(StepMode::Graph);
        for (i, p) in prompts.iter().enumerate() {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            if let Some(dir) = dump_dir {
                dump(dir, "graph", i, &toks, &logits)?;
            }
            graph.push(toks);
            graph_logits.push(logits);
        }
        m.set_mode(StepMode::Eager);
        let mut replay_ok = true;
        for (i, p) in prompts.iter().enumerate() {
            let (toks, logits) = continue_greedy(m, &p.tokens)?;
            if let Some(dir) = dump_dir {
                dump(dir, "eager", i, &toks, &logits)?;
            }
            let same_t = toks == graph[i];
            let same_l = logits
                .iter()
                .zip(&graph_logits[i])
                .all(|(a, b)| a.to_bits() == b.to_bits());
            if !(same_t && same_l) {
                replay_ok = false;
                println!(
                    "replay prompt {}: tokens_equal={same_t} logits_bit_equal={same_l} FAIL",
                    p.id
                );
            }
        }
        m.set_mode(StepMode::Graph);
        println!(
            "replay: {PROMPTS} prompts x {GEN} tokens, graph = eager tokens and last logits bit for bit {}",
            verdict(replay_ok)
        );
        let rep = compare_greedy(&graph, &reference, MARGIN_FLOOR);
        for (r, p) in rep.rows.iter().zip(&prompts) {
            println!(
                "greedy prompt {} ({} ids): {:?} first_diff={:?} ik_margin={:?} ours={}/{} tokens",
                r.id,
                p.tokens.len(),
                r.class,
                r.first_diff,
                r.ref_margin,
                r.n_ours,
                r.n_ref
            );
        }
        let greedy_ok = rep.rows.iter().all(|r| r.class != GreedyClass::Diverged);
        println!(
            "greedy: identical={} near_tie={} diverged={} (want 0, margin floor {MARGIN_FLOOR}) {}",
            rep.n_identical,
            rep.n_near_tie,
            rep.n_diverged,
            verdict(greedy_ok)
        );
        let prefill_ok = prefilled(m, &prompts, &graph, &graph_logits, dump_dir)?;
        let passes_ok = prefill_replay(m, &prompts)?;
        let gemm_ok = !gemm || gemm_prefill(m, dump_dir)?;
        Ok(replay_ok && greedy_ok && prefill_ok && passes_ok && gemm_ok)
    }

    // ------------------------------------------------------------ (p) prefill

    /// Our greedy continuation of `ids` prefilled by `path` — whole, or in
    /// two calls cut at `cut` — then stepped: `GEN` tokens and the last
    /// step's logits.
    fn continue_prefilled(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        cut: Option<usize>,
        path: PrefillPath,
    ) -> Result<(Vec<u32>, Vec<f32>), GateError> {
        m.reset()?;
        let mut out = Vec::with_capacity(GEN);
        let mut next = match cut {
            Some(c) => {
                m.prefill_with(&ids[..c], path)?;
                m.prefill_with(&ids[c..], path)?
            }
            None => m.prefill_with(ids, path)?,
        };
        out.push(next);
        for _ in 1..GEN {
            next = m.step(&[next])?;
            out.push(next);
        }
        Ok((out, m.logits()?))
    }

    fn prefilled(
        m: &mut Qwen3moeModel,
        prompts: &[PromptRow],
        graph: &[Vec<u32>],
        graph_logits: &[Vec<f32>],
        dump_dir: Option<&Path>,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let mut ok = true;
        let mut passes = 0usize;
        let same = |toks: &[u32], logits: &[f32], i: usize| {
            toks == graph[i]
                && logits.len() == graph_logits[i].len()
                && logits
                    .iter()
                    .zip(&graph_logits[i])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        for (i, p) in prompts.iter().enumerate() {
            let plan = m.prefill_plan(p.tokens.len(), PrefillPath::Auto)?;
            if plan.kind() != "prefill" {
                return Err(format!(
                    "prompt {} ({} ids) plans {plan}; this clause is the pass path's",
                    p.id,
                    p.tokens.len()
                )
                .into());
            }
            let (toks, logits) = continue_prefilled(m, &p.tokens, None, PrefillPath::Auto)?;
            if let Some(dir) = dump_dir {
                dump(dir, "prefill", i, &toks, &logits)?;
            }
            passes += Qwen3moeModel::prefill_passes(p.tokens.len());
            if !same(&toks, &logits, i) {
                ok = false;
                println!(
                    "prefill prompt {} ({} ids): tokens or last logits differ from the one-token path FAIL",
                    p.id,
                    p.tokens.len()
                );
            }
        }
        // Every prompt fits one pass, so the multi-pass walk runs on their
        // concatenation, against its own one-token run.
        let long: Vec<u32> = prompts
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect();
        let (step_toks, step_logits) = continue_greedy(m, &long)?;
        let same_long = |(toks, logits): &(Vec<u32>, Vec<f32>)| {
            *toks == step_toks
                && logits.len() == step_logits.len()
                && logits
                    .iter()
                    .zip(&step_logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        let whole_ok = same_long(&continue_prefilled(m, &long, None, PrefillPath::Pass)?);
        let split_ok = same_long(&continue_prefilled(
            m,
            &long,
            Some(SPLIT),
            PrefillPath::Pass,
        )?);
        println!(
            "prefill: {PROMPTS} prompts in {passes} passes of up to {MAX_TOKENS} positions, then \
             {GEN}-token greedy steps: tokens and last logits = the one-token path's bit for bit \
             {}; their {}-id concatenation in {} passes: {}; prefilled as {SPLIT} + {} ids: {}",
            verdict(ok),
            long.len(),
            Qwen3moeModel::prefill_passes(long.len()),
            verdict(whole_ok),
            long.len() - SPLIT,
            verdict(split_ok)
        );
        Ok(ok && whole_ok && split_ok)
    }

    // ------------------------------------------- (q) prefill replay = eager

    /// What one prefill leaves: every layer's K/V rows over its positions,
    /// the last logits and the token.
    struct PrefillRun {
        kv: Vec<Vec<u16>>,
        logits: Vec<f32>,
        token: u32,
    }

    /// One prefill of `ids` in `mode`, from a reset over cache rows seeded
    /// with the pattern.
    fn prefill_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        mode: StepMode,
    ) -> Result<PrefillRun, GateError> {
        m.set_mode(mode);
        m.seed_depth(ids.len())?;
        m.reset()?;
        let token = m.prefill_with(ids, PrefillPath::Pass)?;
        Ok(PrefillRun {
            kv: m.kv_rows(ids.len())?,
            logits: m.logits()?,
            token,
        })
    }

    /// Rows of `a` (per layer, [`HEAD`]-value rows) that differ from `b`'s.
    fn rows_differing(a: &[Vec<u16>], b: &[Vec<u16>]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                x.chunks(HEAD)
                    .zip(y.chunks(HEAD))
                    .filter(|(r, s)| r != s)
                    .count()
            })
            .sum()
    }

    /// Rows of `a` (per layer, [`HEAD`]-value rows) equal to `b`'s.
    fn rows_same(a: &[Vec<u16>], b: &[Vec<u16>]) -> usize {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                x.chunks(HEAD)
                    .zip(y.chunks(HEAD))
                    .filter(|(r, s)| r == s)
                    .count()
            })
            .sum()
    }

    fn prefill_replay(m: &mut Qwen3moeModel, prompts: &[PromptRow]) -> Result<bool, GateError> {
        let long: Vec<u32> = prompts
            .iter()
            .flat_map(|p| p.tokens.iter().copied())
            .collect();
        if long.len() < 2 * MAX_TOKENS {
            return Err(format!(
                "the prompts' concatenation has {} ids; the replay clause takes {}",
                long.len(),
                2 * MAX_TOKENS
            )
            .into());
        }
        let mut ok = true;
        for rows in 1..=MAX_TOKENS {
            let ids = &long[..MAX_TOKENS + rows];
            m.seed_depth(ids.len())?;
            let seeded = m.kv_rows(ids.len())?;
            let g = prefill_run(m, ids, StepMode::Graph)?;
            let e = prefill_run(m, ids, StepMode::Eager)?;
            let kv_diff = rows_differing(&g.kv, &e.kv);
            let stale = rows_same(&g.kv, &seeded);
            let logits_same = g.logits.len() == e.logits.len()
                && g.logits
                    .iter()
                    .zip(&e.logits)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
            let pass = kv_diff == 0 && stale == 0 && logits_same && g.token == e.token;
            println!(
                "prefill replay m={rows}: {} ids in passes of {MAX_TOKENS} + {rows}: K/V rows \
                 differing graph vs eager {kv_diff} (want 0), still the seeded pattern {stale} \
                 (want 0), last logits bit-equal {logits_same}, token graph={} eager={} {}",
                ids.len(),
                g.token,
                e.token,
                verdict(pass)
            );
            ok &= pass;
        }
        m.set_mode(StepMode::Graph);
        println!(
            "prefill replay: every pass size's replay = its eager twin (K/V rows and last logits bit \
             for bit) {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // --------------------------------------------- (u) GEMM prefill band

    /// The first `n` ids of the prose file, after its digest is checked.
    fn prose(n: usize) -> Result<Vec<u32>, GateError> {
        let path = data_dir().join("qwen3moe").join(PROSE);
        let out = std::process::Command::new("sha256sum")
            .arg(&path)
            .output()?;
        let digest = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() || digest.split_whitespace().next() != Some(PROSE_SHA256) {
            return Err(format!(
                "{}: sha256 {:?}, want {PROSE_SHA256}",
                path.display(),
                digest.trim()
            )
            .into());
        }
        let ids: Vec<u32> = std::fs::read_to_string(&path)?
            .lines()
            .take(n)
            .map(|l| l.trim().parse::<u32>())
            .collect::<Result<_, _>>()?;
        if ids.len() < n {
            return Err(format!("{} holds {} ids, want {n}", path.display(), ids.len()).into());
        }
        Ok(ids)
    }

    /// Top-1 minus top-2 of `v`.
    fn margin(v: &[f32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &x in v {
            if x > a {
                b = a;
                a = x;
            } else if x > b {
                b = x;
            }
        }
        a - b
    }

    /// What one prefill of a long prompt leaves: every layer's K/V rows, the
    /// last logits, and the greedy continuation after it with each
    /// generated position's top1-top2 margin.
    struct LongRun {
        kv: Vec<Vec<u16>>,
        logits: Vec<f32>,
        tokens: Vec<u32>,
        margins: Vec<f32>,
    }

    /// `ids` from a reset, fed one step per token (`None`) or prefilled by
    /// the path (`Some`; cut into two calls at `cut`), then `GEN − 1`
    /// greedy steps.
    fn long_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        path: Option<PrefillPath>,
        cut: Option<usize>,
    ) -> Result<LongRun, GateError> {
        m.reset()?;
        let first = match (path, cut) {
            (None, _) => m.step(ids)?,
            (Some(p), Some(c)) => {
                m.prefill_with(&ids[..c], p)?;
                m.prefill_with(&ids[c..], p)?
            }
            (Some(p), None) => m.prefill_with(ids, p)?,
        };
        let kv = m.kv_rows(ids.len())?;
        let logits = m.logits()?;
        let mut tokens = vec![first];
        let mut margins = vec![margin(&logits)];
        for _ in 1..GEN {
            let t = m.step(&[*tokens.last().ok_or("no token")?])?;
            tokens.push(t);
            margins.push(margin(&m.logits()?));
        }
        Ok(LongRun {
            kv,
            logits,
            tokens,
            margins,
        })
    }

    /// An f16's bits as a monotone integer, so two values' distance in ulp
    /// is the difference (both zeros at 0).
    fn f16_ord(b: u16) -> i32 {
        let mag = i32::from(b & 0x7fff);
        if b & 0x8000 != 0 { -mag } else { mag }
    }

    /// `‖a − b‖ / ‖b‖` over f16 bits.
    fn rel16(a: &[u16], b: &[u16]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (
                f64::from(gguf::quant::half_to_f32(x)),
                f64::from(gguf::quant::half_to_f32(y)),
            );
            num += (x - y).powi(2);
            den += y.powi(2);
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    /// One (u) run of a prompt of `n` ids as raw little-endian files under
    /// `dir` named by `tag` (`--dump`): every layer's K/V rows in `kv_rows`
    /// order, the last logits, the greedy tokens after it.
    fn dump_long(dir: &Path, n: usize, tag: &str, run: &LongRun) -> Result<(), GateError> {
        std::fs::create_dir_all(dir)?;
        let kv: Vec<u8> = run
            .kv
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let l: Vec<u8> = run.logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let t: Vec<u8> = run.tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join(format!("u{n}-{tag}.kv")), kv)?;
        std::fs::write(dir.join(format!("u{n}-{tag}.logits")), l)?;
        std::fs::write(dir.join(format!("u{n}-{tag}.tokens")), t)?;
        Ok(())
    }

    fn gemm_prefill(m: &mut Qwen3moeModel, dump_dir: Option<&Path>) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let longest = LONG.iter().copied().max().ok_or("no long prompt")?;
        let prose = prose(longest)?;
        let mut ok = true;
        let mut whole_1300 = None;
        for &n in &LONG {
            let ids = &prose[..n];
            let plan = m.prefill_plan(n, PrefillPath::Auto)?;
            let one = long_run(m, ids, None, None)?;
            let t0 = Instant::now();
            let gemm = long_run(m, ids, Some(PrefillPath::Auto), None)?;
            let wall = t0.elapsed().as_secs_f64();
            if let Some(dir) = dump_dir {
                dump_long(dir, n, "step", &one)?;
                dump_long(dir, n, "gemm", &gemm)?;
            }
            // The ruler: the same prompt on the other flash pass, eager.
            let mma = m.body("gemm_prefill")?.flash_mma();
            m.set_mode(StepMode::Eager);
            m.set_flash_mma(!mma)?;
            let alt = long_run(m, ids, None, None)?;
            m.set_flash_mma(mma)?;
            m.set_mode(StepMode::Graph);
            // Layer 0's moves in ulp, printed.
            let (mut l0_max, mut l0_moved) = (0u32, 0usize);
            for (&x, &y) in gemm.kv[0].iter().zip(&one.kv[0]) {
                let d = (f16_ord(x) - f16_ord(y)).unsigned_abs();
                l0_max = l0_max.max(d);
                l0_moved += usize::from(d != 0);
            }
            println!(
                "gemm n={n} plan={plan} layer=0 kv_values={} moved={l0_moved} max_ulp={l0_max} \
                 (printed)",
                one.kv[0].len()
            );
            let mut kv_ok = true;
            // (layer, distance over its band) of the layer closest to its band.
            let mut worst = (0usize, 0.0f64);
            for (l, ((g, o), a)) in gemm.kv.iter().zip(&one.kv).zip(&alt.kv).enumerate() {
                let half = o.len() / 2;
                let (rk, rv) = (rel16(&g[..half], &o[..half]), rel16(&g[half..], &o[half..]));
                let (sk, sv) = (rel16(&a[..half], &o[..half]), rel16(&a[half..], &o[half..]));
                let (d, band) = if l == 0 {
                    (rk.max(rv), GEMM_L0_REL)
                } else {
                    (
                        rk.max(rv) / sk.max(sv).max(f64::MIN_POSITIVE),
                        GEMM_SPREAD_RATIO,
                    )
                };
                let pass = d <= band;
                if d / band > worst.1 {
                    worst = (l, d / band);
                }
                if l % 8 == 0 || l == gemm.kv.len() - 1 || (32..40).contains(&l) || !pass {
                    let judged = if l == 0 { "rel" } else { "ratio" };
                    println!(
                        "gemm n={n} layer={l} k_rel={rk:.3e} v_rel={rv:.3e} spread k={sk:.3e} \
                         v={sv:.3e} {judged}={d:.3e} band={band:.3e} {}",
                        verdict(pass)
                    );
                }
                kv_ok &= pass;
            }
            let lrel = rel(&gemm.logits, &one.logits, |i| f64::from(one.logits[i]));
            let lspread = rel(&alt.logits, &one.logits, |i| f64::from(one.logits[i]));
            let lratio = lrel / lspread.max(f64::MIN_POSITIVE);
            let logits_ok = lratio <= GEMM_SPREAD_RATIO;
            let reference = GreedyRow {
                id: n,
                n_tokens: n,
                argmax: one.tokens[0],
                top5: Vec::new(),
                gen_ids: one.tokens.clone(),
                gen_margins: one.margins.clone(),
            };
            let rep = compare_greedy(
                std::slice::from_ref(&gemm.tokens),
                &[reference],
                MARGIN_FLOOR,
            );
            let r = &rep.rows[0];
            let greedy_ok = r.class != GreedyClass::Diverged;
            println!(
                "gemm n={n}: K/V rows within their bands {} (closest: layer {} at {:.2} of its \
                 band); last logits rel {lrel:.3e} against the flash arithmetics' own \
                 {lspread:.3e}, ratio {lratio:.3} (band {GEMM_SPREAD_RATIO}) {}; greedy {:?} \
                 first_diff={:?} one_token_margin={:?} {}; the GEMM prefill and its {GEN} tokens \
                 {wall:.2} s (runtime value)",
                verdict(kv_ok),
                worst.0,
                worst.1,
                verdict(logits_ok),
                r.class,
                r.first_diff,
                r.ref_margin,
                verdict(greedy_ok)
            );
            ok &= kv_ok && logits_ok && greedy_ok;
            if n == LONG[1] {
                whole_1300 = Some(gemm);
            }
        }
        let whole = whole_1300.ok_or("the 1,300-id run is missing")?;
        let n = LONG[1];
        let ids = &prose[..n];
        let cut_plans = (
            m.prefill_plan(LONG_SPLIT, PrefillPath::Auto)?,
            m.prefill_plan(n - LONG_SPLIT, PrefillPath::Auto)?,
        );
        let split = long_run(m, ids, Some(PrefillPath::Auto), Some(LONG_SPLIT))?;
        let kv_diff = rows_differing(&split.kv, &whole.kv);
        let logits_same = bits_equal(&split.logits, &whole.logits);
        let pass = kv_diff == 0 && logits_same && split.tokens == whole.tokens;
        println!(
            "gemm split n={n} as {LONG_SPLIT} ({}) + {} ({}): K/V rows differing from the whole \
             prefill {kv_diff} (want 0), last logits bit-equal {logits_same}, continuation equal {} {}",
            cut_plans.0,
            n - LONG_SPLIT,
            cut_plans.1,
            split.tokens == whole.tokens,
            verdict(pass)
        );
        ok &= pass;
        println!(
            "gemm prefill: {LONG:?} prose ids against the one-token path, K/V in band (layer 0 \
             {GEMM_L0_REL:e}, later layers and the logits {GEMM_SPREAD_RATIO} x the flash \
             arithmetics' spread), greedy not diverged, split = whole {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------------- (t) the rope table

    /// The ubatch size (t) prefills at: small, so that its prompts cross
    /// ubatch boundaries far below the cache's height.
    const T_UB: usize = 64;

    /// (t)'s calls, in order from a reset: the first `T_FIRST` prose ids and
    /// the next `T_NEXT` on the GEMM path (at [`T_UB`] two and three
    /// ubatches, the last of each short, the second call's first position
    /// off every ubatch boundary), the next `T_PASS` on the pass path (a full
    /// pass and a short one), then `T_STEPS` one-token steps.
    const T_FIRST: usize = 100;
    const T_NEXT: usize = 150;
    const T_PASS: usize = MAX_TOKENS + 5;
    const T_STEPS: usize = 3;

    /// The positions of `got`'s [`HEAD`]-value rows, the first at position
    /// `first`, whose bits are not `RopeTable::push`'s at that position.
    fn rope_rows_differing(
        rope: &RopeTable,
        got: &[f32],
        first: usize,
    ) -> Result<Vec<usize>, GateError> {
        let mut want = Vec::with_capacity(HEAD);
        let mut bad = Vec::new();
        for (i, row) in got.chunks(HEAD).enumerate() {
            want.clear();
            rope.push(u32::try_from(first + i)?, Direction::Forward, &mut want);
            if !bits_equal(row, &want) {
                bad.push(first + i);
            }
        }
        Ok(bad)
    }

    /// Of `kv`'s rows (`kv_rows(ctx)` order: per layer, K then V, each head's
    /// `ctx` rows), those below `end` still holding `seeded`'s bits and those
    /// at or past it no longer holding them; `None` when the two differ in
    /// shape.
    fn rows_by_end(
        kv: &[Vec<u16>],
        seeded: &[Vec<u16>],
        ctx: usize,
        end: usize,
    ) -> Option<(usize, usize)> {
        let shaped = kv.len() == seeded.len()
            && kv
                .iter()
                .zip(seeded)
                .all(|(a, b)| a.len() == b.len() && a.len() % (ctx * HEAD) == 0);
        if !shaped {
            return None;
        }
        let (mut stale, mut stray) = (0, 0);
        for (a, b) in kv.iter().zip(seeded) {
            for (i, (x, y)) in a.chunks(HEAD).zip(b.chunks(HEAD)).enumerate() {
                if i % ctx < end {
                    stale += usize::from(x == y);
                } else {
                    stray += usize::from(x != y);
                }
            }
        }
        Some((stale, stray))
    }

    /// (t) (module doc), on a model of `ctx` cache rows.
    fn rope_table(m: &mut Qwen3moeModel, ctx: usize) -> Result<bool, GateError> {
        let hp = m.body("rope_table")?.hparams().clone();
        let rope = RopeTable::new(&RopeSpec::window(hp.rope.base, hp.rope.dims))?;
        let table = m.rope_rows(0..ctx)?;
        let bad = rope_rows_differing(&rope, &table, 0)?;
        let table_ok = table.len() == ctx * HEAD && bad.is_empty();
        println!(
            "rope table rows 0..{ctx} (position 0 to the last, {}): {} not RopeTable::push's bits \
             (want 0){} {}",
            ctx - 1,
            bad.len(),
            bad.first()
                .map_or(String::new(), |p| format!(", the first at position {p}")),
            verdict(table_ok)
        );
        let end = T_FIRST + T_NEXT + T_PASS + T_STEPS;
        let prose = prose(end)?;
        let kept = m.ubatch()?;
        m.set_ubatch(T_UB)?;
        m.seed_depth(ctx - 1)?;
        m.reset()?;
        let seeded = m.kv_rows(ctx)?;
        let mut calls = Vec::new();
        let mut at = 0;
        for (n, path) in [
            (T_FIRST, Some(PrefillPath::Gemm)),
            (T_NEXT, Some(PrefillPath::Gemm)),
            (T_PASS, Some(PrefillPath::Pass)),
            (T_STEPS, None),
        ] {
            let ids = &prose[at..at + n];
            match path {
                Some(p) => {
                    calls.push(format!("{at}: {}", m.prefill_plan(n, p)?));
                    m.prefill_with(ids, p)?;
                }
                None => {
                    calls.push(format!("{at}: {n} steps"));
                    m.step(ids)?;
                }
            }
            at += n;
        }
        let pos = m.pos() as usize;
        let kv = m.kv_rows(ctx)?;
        m.set_ubatch(kept)?;
        m.reset()?;
        let (stale, stray) = rows_by_end(&kv, &seeded, ctx, end)
            .ok_or("kv_rows returned another shape after the calls")?;
        let rows_ok = pos == end && stale == 0 && stray == 0;
        println!(
            "rope rows by position: calls from a reset over seeded rows [{}]: position {pos} (want \
             {end}), rows below it still the seeded pattern {stale} (want 0), rows at or past it \
             not the seeded pattern {stray} (want 0) {}",
            calls.join("; "),
            verdict(rows_ok)
        );
        let ok = table_ok && rows_ok;
        println!(
            "rope: the table holds RopeTable::push's rows, and the ubatch, the pass and the step \
             each write their tokens' rows at their own positions and no other {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------ (w) ubatch-size invariance

    /// Cache rows of the model (w) opens: its longest prompt, with room
    /// past it (`seed_depth` keeps a row free for a step).
    const CTX_W: usize = W_LONG + 64;

    /// (w)'s prompt: a whole number of the largest ubatch.
    const W_LONG: usize = UBATCH;

    /// The ubatch sizes (w) compares: the size before the load-time lever,
    /// an odd one that cuts the prompt with a ragged last ubatch, the largest.
    const W_SIZES: [usize; 3] = [512, 1000, UBATCH];

    /// Where (w)'s split prefill at the largest size cuts the prompt: off
    /// every boundary of [`W_SIZES`].
    const W_SPLIT: usize = 2500;

    /// `ids` prefilled by the default path from a reset over cache rows
    /// seeded with the pattern (in two calls cut at `cut`), at the model's
    /// ubatch size: the K/V rows, the last logits, the token.
    fn ub_run(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        cut: Option<usize>,
    ) -> Result<PrefillRun, GateError> {
        m.seed_depth(ids.len())?;
        m.reset()?;
        let token = match cut {
            Some(c) => {
                m.prefill_with(&ids[..c], PrefillPath::Auto)?;
                m.prefill_with(&ids[c..], PrefillPath::Auto)?
            }
            None => m.prefill_with(ids, PrefillPath::Auto)?,
        };
        Ok(PrefillRun {
            kv: m.kv_rows(ids.len())?,
            logits: m.logits()?,
            token,
        })
    }

    /// Whether `run` left what `want` did, bit for bit; one line under `label`.
    fn same_run(label: &str, run: &PrefillRun, want: &PrefillRun) -> bool {
        let kv_diff = rows_differing(&run.kv, &want.kv);
        let logits_same = bits_equal(&run.logits, &want.logits);
        let pass = kv_diff == 0 && logits_same && run.token == want.token;
        println!(
            "ubatch {label}: K/V rows differing from the {UBATCH}-token ubatch {kv_diff} (want 0), \
             last logits bit-equal {logits_same}, token {} (want {}) {}",
            run.token,
            want.token,
            verdict(pass)
        );
        pass
    }

    /// (w) (module doc), on its own model: the caller drops any other first,
    /// so the two never share the card.
    fn ubatch_sizes() -> Result<bool, GateError> {
        let mut m = open(CTX_W, StepMode::Graph, KvQ8::F16)?;
        let prose = prose(W_LONG + 1)?;
        let (ids, ids1) = (&prose[..W_LONG], &prose[..=W_LONG]);
        let mut ok = true;
        m.set_ubatch(UBATCH)?;
        let plan = |m: &Qwen3moeModel, n: usize| m.prefill_plan(n, PrefillPath::Auto);
        // The references, each checked to have written every row it covers
        // (none still the seeded pattern), so an equal run is not two runs
        // that wrote nothing.
        let mut refs = Vec::with_capacity(2);
        for x in [ids, ids1] {
            m.seed_depth(x.len())?;
            let seeded = m.kv_rows(x.len())?;
            let run = ub_run(&mut m, x, None)?;
            let stale = rows_same(&run.kv, &seeded);
            let pass = stale == 0;
            println!(
                "ubatch size={UBATCH} n={} plan={} resident_bytes={} (a reference): K/V rows still \
                 the seeded pattern {stale} (want 0) {}",
                x.len(),
                plan(&m, x.len())?,
                m.resident_bytes(),
                verdict(pass)
            );
            ok &= pass;
            refs.push(run);
        }
        let (whole, whole1) = (&refs[0], &refs[1]);
        let split = ub_run(&mut m, ids, Some(W_SPLIT))?;
        ok &= same_run(
            &format!(
                "size={UBATCH} n={W_LONG} split as {W_SPLIT} ({}) + {} ({})",
                plan(&m, W_SPLIT)?,
                W_LONG - W_SPLIT,
                plan(&m, W_LONG - W_SPLIT)?
            ),
            &split,
            whole,
        );
        drop(split);
        for &u in &W_SIZES[..W_SIZES.len() - 1] {
            m.set_ubatch(u)?;
            let run = ub_run(&mut m, ids, None)?;
            ok &= same_run(
                &format!(
                    "size={u} n={W_LONG} plan={} resident_bytes={}",
                    plan(&m, W_LONG)?,
                    m.resident_bytes()
                ),
                &run,
                whole,
            );
            drop(run);
            if u == W_SIZES[0] {
                let run = ub_run(&mut m, ids1, None)?;
                ok &= same_run(
                    &format!("size={u} n={} plan={}", W_LONG + 1, plan(&m, W_LONG + 1)?),
                    &run,
                    whole1,
                );
            }
        }
        // Sizes past the range: refused, the size kept.
        let kept = m.ubatch()?;
        let refused = [0, UBATCH + 1].iter().all(|&u| m.set_ubatch(u).is_err());
        let still = m.ubatch()? == kept;
        let pass = refused && still;
        println!(
            "ubatch refusals: sizes 0 and {} refused {refused}, size kept at {kept} {still} {}",
            UBATCH + 1,
            verdict(pass)
        );
        ok &= pass;
        println!(
            "ubatch: the prompt's K/V rows, last logits and token at sizes {W_SIZES:?} and split = \
             whole at {UBATCH}, bit for bit {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ------------------------------------------- (n) two resident slots

    /// The slot clause's prompt width: two distinct windows of the prose
    /// this wide, every pass of each [`MAX_TOKENS`] tokens on the pass path.
    const N_IDS: usize = 64;

    /// The slot clause's prompt windows of the prose: A and B, distinct.
    const N_A: usize = 64;
    const N_B: usize = 700;

    /// Interleaved steps a slot of (n): each slot replays its own captured
    /// chain against its own planes while the other's state sits between
    /// every pair of its steps.
    const GEN_W: usize = 48;

    /// The steps the reset-isolation arm of (n) runs on each slot.
    const N_STEPS: usize = 8;

    /// Prompt `ids` prefilled on the pass path — the model standing wherever
    /// it stands, so the caller resets — then `steps` greedy steps; the ids
    /// cover the prefill's argmax and every step's.
    fn feed_and_step(
        m: &mut Qwen3moeModel,
        ids: &[u32],
        steps: usize,
    ) -> Result<Vec<u32>, GateError> {
        let mut out = vec![m.prefill_with(ids, PrefillPath::Pass)?];
        step_more(m, &mut out, steps)?;
        Ok(out)
    }

    /// `steps` more greedy steps on the model as it stands, appended to
    /// `out` (its last id the first step's input).
    fn step_more(m: &mut Qwen3moeModel, out: &mut Vec<u32>, steps: usize) -> Result<(), GateError> {
        for _ in 0..steps {
            let last = *out.last().ok_or("no token")?;
            out.push(m.step(&[last])?);
        }
        Ok(())
    }

    /// FNV-1a 64 of every layer's K/V rows below `end` — the (k) clause's
    /// hash of the same readback.
    fn kv_fnv(m: &mut Qwen3moeModel, end: usize) -> Result<u64, GateError> {
        Ok(m.kv_rows(end)?
            .iter()
            .flatten()
            .fold(Fnv1a64::default(), |h, v| h.bytes(&v.to_le_bytes()))
            .value())
    }

    /// (n) (module doc): two resident slots, interleaved.
    fn slots_two(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        let prose = prose(N_B + N_IDS)?;
        let (a, b) = (&prose[N_A..N_A + N_IDS], &prose[N_B..N_B + N_IDS]);
        // Alone, the load's one sequence: A runs GEN_W steps (what the
        // interleave compares against) and then N_STEPS more (the
        // continuation the isolation arm compares against); B runs GEN_W.
        m.reset()?;
        let mut a_alone = feed_and_step(m, a, GEN_W)?;
        let a_logits = Fnv1a64::default().f32s(&m.logits()?).value();
        let a_kv = kv_fnv(m, usize::try_from(m.pos())?)?;
        step_more(m, &mut a_alone, N_STEPS)?;
        let a_cont = a_alone[1 + GEN_W..].to_vec();
        m.reset()?;
        let b_alone = feed_and_step(m, b, GEN_W)?;
        let b_logits = Fnv1a64::default().f32s(&m.logits()?).value();
        let b_kv = kv_fnv(m, usize::try_from(m.pos())?)?;
        // Together: a second sequence resident, its bytes counted against
        // the one owner of the per-sequence KV term.
        let before = m.resident_bytes();
        m.add_slots(2)?;
        let seq_bytes = m.body("slots")?.seq_bytes();
        let grown = m.resident_bytes() - before;
        let hp = m.body("slots")?.hparams().clone();
        let derived = m.layers().len()
            * CTX
            * usize::try_from(runtime::stores::kv_row_bytes(hp.n_head_kv, hp.head_dim))?;
        let bytes_ok = grown == seq_bytes && seq_bytes == derived;
        println!(
            "slots bytes: the second sequence grew resident_bytes by {grown}, seq_bytes \
             {seq_bytes}, the derived {} layers x {CTX} rows x kv_row_bytes({}, {}) = {derived} {}",
            m.layers().len(),
            hp.n_head_kv,
            hp.head_dim,
            verdict(bytes_ok)
        );
        // The interleave: each slot prefilled from its own reset, then a
        // step of slot 0 and a step of slot 1 a round. Each slot's last
        // logits are hashed inside that round: the head is the model's one,
        // so a step of the other slot's replaces what it holds.
        m.select_slot(0)?;
        m.reset()?;
        let mut a_ids = vec![m.prefill_with(a, PrefillPath::Pass)?];
        m.select_slot(1)?;
        m.reset()?;
        let mut b_ids = vec![m.prefill_with(b, PrefillPath::Pass)?];
        let (mut a_last, mut b_last) = (0u64, 0u64);
        for r in 0..GEN_W {
            let last = r + 1 == GEN_W;
            m.select_slot(0)?;
            let t = *a_ids.last().ok_or("no token")?;
            a_ids.push(m.step(&[t])?);
            if last {
                a_last = Fnv1a64::default().f32s(&m.logits()?).value();
            }
            m.select_slot(1)?;
            let t = *b_ids.last().ok_or("no token")?;
            b_ids.push(m.step(&[t])?);
            if last {
                b_last = Fnv1a64::default().f32s(&m.logits()?).value();
            }
        }
        m.select_slot(0)?;
        let a_got = (a_last, kv_fnv(m, usize::try_from(m.pos())?)?);
        m.select_slot(1)?;
        let b_got = (b_last, kv_fnv(m, usize::try_from(m.pos())?)?);
        let off: Vec<&str> = [
            ("slot 0 ids", a_ids == a_alone[..1 + GEN_W]),
            ("slot 0 logits", a_got.0 == a_logits),
            ("slot 0 kv", a_got.1 == a_kv),
            ("slot 1 ids", b_ids == b_alone),
            ("slot 1 logits", b_got.0 == b_logits),
            ("slot 1 kv", b_got.1 == b_kv),
        ]
        .into_iter()
        .filter_map(|(name, same)| (!same).then_some(name))
        .collect();
        let together_ok = off.is_empty();
        println!(
            "slots interleave: {} interleaved steps a slot against its alone run: {} {}",
            GEN_W,
            if together_ok {
                "every slot's ids, logits and K/V rows bit for bit".to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(together_ok)
        );
        // Captures per slot: both hold their own after the interleave.
        m.select_slot(0)?;
        let cap0 = m.has_capture();
        m.select_slot(1)?;
        let cap1 = m.has_capture();
        let caps = cap0 && cap1;
        // Reset isolation: slot 1 rewinds and runs its prompt again; slot
        // 0's next ids are its alone run's continuation.
        m.select_slot(1)?;
        m.reset()?;
        feed_and_step(m, b, N_STEPS)?;
        m.select_slot(0)?;
        let mut a_tail = vec![*a_ids.last().ok_or("no token")?];
        step_more(m, &mut a_tail, N_STEPS)?;
        let iso_ok = a_tail[1..] == a_cont[..];
        println!(
            "slots reset isolation: slot 1 reset and re-prompted, slot 0's next {N_STEPS} ids {} \
             its alone run's continuation {}",
            if iso_ok { "equal" } else { "differ from" },
            verdict(iso_ok)
        );
        // A fault on one slot: the other's steps refuse naming the slot, and
        // only the faulting slot's reset lifts it ((x)'s plant).
        m.select_slot(1)?;
        m.reset()?;
        plant_fault(m)?;
        let read = m.step(&[1]);
        let poisoned = m.poisoned();
        m.select_slot(0)?;
        let named = match m.step(&[1]) {
            Err(GpuError::Shape { detail, .. }) => detail.contains("slot 1"),
            _ => false,
        };
        m.reset()?;
        let stands = m.step(&[1]).is_err();
        m.select_slot(1)?;
        m.reset()?;
        let clean = m.step(&[1]).is_ok();
        m.reset()?;
        let fault_ok = read.is_err() && poisoned.is_some() && named && stands && clean;
        println!(
            "slots fault: the faulting slot's step {}, slot 0's step refused naming slot 1 \
             {named}, after slot 0's reset still refused {stands}, after slot 1's own reset a \
             clean step {clean} {}",
            match &read {
                Ok(t) => format!("returned token {t}"),
                Err(e) => format!("returned \"{e}\""),
            },
            verdict(fault_ok)
        );
        // The select refusals name themselves.
        let sel_named = match m.select_slot(2) {
            Err(GpuError::Shape {
                what: "GpuModel::select_slot",
                detail,
            }) => detail.contains("serves 0..2"),
            _ => false,
        };
        let add_named = match m.add_slots(1) {
            Err(GpuError::Shape {
                what: "GpuModel::add_slots",
                detail,
            }) => detail.contains("already serves 2"),
            _ => false,
        };
        let refused_ok = sel_named && add_named;
        println!(
            "slots refusals: select 2 of two slots named {sel_named}, add_slots(1) after \
             add_slots(2) named {add_named} {}",
            verdict(refused_ok)
        );
        // A mode round trip leaves no capture on either slot.
        m.set_mode(StepMode::Eager);
        m.set_mode(StepMode::Graph);
        m.select_slot(0)?;
        let gone0 = !m.has_capture();
        m.select_slot(1)?;
        let gone1 = !m.has_capture();
        let gone_ok = gone0 && gone1;
        println!(
            "slots captures: both slots held captures after the interleave {caps}; none after a \
             set_mode round trip ({gone0}, {gone1}) {}",
            verdict(caps && gone_ok)
        );
        let ok = bytes_ok && together_ok && caps && iso_ok && fault_ok && refused_ok && gone_ok;
        println!("slots: {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------------------- (j) one pass of two slots

    /// (j)'s prompts: windows of the prose (first id, ids) of distinct
    /// lengths, so the two slots stand at different positions.
    const J_A: (usize, usize) = (64, 64);
    const J_B: (usize, usize) = (700, 37);

    /// (j)'s passes of one row a slot.
    const J_ROUNDS: usize = 16;

    /// PIN(2026-10-05): the nodes a pass of several slots adds per layer for
    /// each busy slot past the first, at the same rows, derived before the
    /// pass was built: the launches bound to one sequence's planes run once
    /// a slot over its row window — the head norm and rope with the cache
    /// append, the flash's segment pass and its merge (`slot_pass`: the
    /// flash runs whole per slot) — 3; every other launch covers all of the
    /// pass's rows. With its embedding from its own record, a slot adds
    /// `1 + 3 · n_layer` ([`slot_nodes`]).
    const J_SLOT_LAYER_NODES: usize = 3;

    /// The nodes a busy slot adds to a pass of several slots at the same
    /// rows ([`J_SLOT_LAYER_NODES`]).
    fn slot_nodes(n_layer: usize) -> usize {
        1 + J_SLOT_LAYER_NODES * n_layer
    }

    /// A slot's run of (j): the ids — its prompt's argmax, then each
    /// position's — each step's last-logits hash, the position it ends at
    /// and its K/V rows' hash there.
    #[derive(PartialEq)]
    struct JRun {
        ids: Vec<u32>,
        logits: Vec<u64>,
        pos: u32,
        kv: u64,
    }

    /// Slot 0 alone from a reset — the state a fresh process stands in,
    /// (k) — on `prompt`, then `steps` steps each feeding its argmax.
    fn j_solo(m: &mut Qwen3moeModel, prompt: &[u32], steps: usize) -> Result<JRun, GateError> {
        m.select_slot(0)?;
        m.reset()?;
        let mut ids = vec![m.prefill_with(prompt, PrefillPath::Pass)?];
        let mut logits = Vec::with_capacity(steps);
        for t in 0..steps {
            ids.push(m.step(&[ids[t]])?);
            logits.push(Fnv1a64::default().f32s(&m.logits()?).value());
        }
        let pos = m.pos();
        Ok(JRun {
            ids,
            logits,
            pos,
            kv: kv_fnv(m, usize::try_from(pos)?)?,
        })
    }

    /// The two slots' runs in passes of several slots: A in slot 0 and B
    /// in slot 1 prefilled each from its own reset, then [`J_ROUNDS`] passes
    /// of one row a slot each feeding back its argmax, a pass of three rows
    /// (slot 0's last argmax and the next id of `a_next` — a pass cannot feed
    /// its own row back — beside slot 1's last argmax), and a pass of slot 1
    /// alone. Every pass's rows' logits hashed in row order.
    fn j_passes(
        m: &mut Qwen3moeModel,
        (a, b): (&[u32], &[u32]),
        a_next: &[u32],
    ) -> Result<(JRun, JRun), GateError> {
        m.select_slot(0)?;
        m.reset()?;
        let mut ga = vec![m.prefill_with(a, PrefillPath::Pass)?];
        m.select_slot(1)?;
        m.reset()?;
        let mut gb = vec![m.prefill_with(b, PrefillPath::Pass)?];
        let (mut la, mut lb) = (Vec::new(), Vec::new());
        let hash = |m: &Qwen3moeModel| -> Result<Vec<u64>, GateError> {
            Ok(m.slots_logits()?
                .iter()
                .map(|l| Fnv1a64::default().f32s(l).value())
                .collect())
        };
        for t in 0..J_ROUNDS {
            let out = m.step_slots(&[(0, &[ga[t]]), (1, &[gb[t]])])?;
            let h = hash(m)?;
            let (&[ia, ib], &[ha, hb]) = (&out.ids[..], &h[..]) else {
                return Err(format!("(j): a pass of two rows gave {:?}", out.ids).into());
            };
            ga.push(ia);
            gb.push(ib);
            la.push(ha);
            lb.push(hb);
        }
        let next = *a_next
            .get(J_ROUNDS + 1)
            .ok_or("(j): A's solo run is short")?;
        let out = m.step_slots(&[(0, &[ga[J_ROUNDS], next][..]), (1, &[gb[J_ROUNDS]][..])])?;
        let h = hash(m)?;
        let (&[ia0, ia1, ib], &[ha0, ha1, hb]) = (&out.ids[..], &h[..]) else {
            return Err(format!("(j): a pass of three rows gave {:?}", out.ids).into());
        };
        ga.extend([ia0, ia1]);
        gb.push(ib);
        la.extend([ha0, ha1]);
        lb.push(hb);
        let out = m.step_slots(&[(1, &[gb[J_ROUNDS + 1]])])?;
        let h = hash(m)?;
        let (&[ib], &[hb]) = (&out.ids[..], &h[..]) else {
            return Err(format!("(j): a pass of one row gave {:?}", out.ids).into());
        };
        gb.push(ib);
        lb.push(hb);
        let mut runs = Vec::with_capacity(2);
        for (slot, ids, logits) in [(0, ga, la), (1, gb, lb)] {
            m.select_slot(slot)?;
            let pos = m.pos();
            runs.push(JRun {
                ids,
                logits,
                pos,
                kv: kv_fnv(m, usize::try_from(pos)?)?,
            });
        }
        let [ra, rb]: [JRun; 2] = runs.try_into().map_err(|_| "(j): two runs")?;
        Ok((ra, rb))
    }

    /// The fields of `got` that differ from `want`, by name.
    fn j_off(slot: usize, got: &JRun, want: &JRun) -> Vec<String> {
        [
            ("ids", got.ids == want.ids),
            ("logits", got.logits == want.logits),
            ("position", got.pos == want.pos),
            ("kv", got.kv == want.kv),
        ]
        .into_iter()
        .filter(|&(_, same)| !same)
        .map(|(name, _)| format!("slot {slot} {name}"))
        .collect()
    }

    /// `r`, a call that must be refused by `what` with a detail holding
    /// `phrase`.
    fn refused<T>(r: Result<T, GpuError>, what: &str, phrase: &str) -> bool {
        match r {
            Err(GpuError::Shape { what: w, detail }) => w == what && detail.contains(phrase),
            _ => false,
        }
    }

    /// (j) (module doc): one pass of two slots. The last clause on the main
    /// model: the model it leaves is dropped, its second sequence's planes
    /// with it.
    fn one_pass_two_slots(m: &mut Qwen3moeModel) -> Result<bool, GateError> {
        const WHAT: &str = "GpuModel::step_slots";
        m.set_mode(StepMode::Graph);
        m.add_slots(2)?;
        let prose = prose(J_B.0 + J_B.1)?;
        let (a, b) = (&prose[J_A.0..J_A.0 + J_A.1], &prose[J_B.0..J_B.0 + J_B.1]);
        // (j1) bits: each slot's ids, logits, position and K/V rows against
        // its solo run, in graph mode and again in eager mode.
        let steps = J_ROUNDS + 2;
        let (sa, sb) = (j_solo(m, a, steps)?, j_solo(m, b, steps)?);
        let mut bits_ok = true;
        for mode in [StepMode::Graph, StepMode::Eager] {
            m.set_mode(mode);
            // A pass that fails is this arm's failure, named here: the
            // clauses after it still run (and refuse while it poisons).
            let off = match j_passes(m, (a, b), &sa.ids) {
                Ok((ra, rb)) => {
                    let mut off = j_off(0, &ra, &sa);
                    off.extend(j_off(1, &rb, &sb));
                    off
                }
                Err(e) => vec![format!("a pass that failed ({e})")],
            };
            let ok = off.is_empty();
            println!(
                "one pass bits {mode:?}: {J_ROUNDS} passes of slots 0 and 1 at one row a slot, a \
                 pass of 2 + 1 rows and one of slot 1 alone, from positions {} and {}: {} {}",
                a.len(),
                b.len(),
                if ok {
                    format!(
                        "every row's id and logits, each slot's position ({}, {}) and K/V rows \
                         bit for bit its solo run's",
                        sa.pos, sb.pos
                    )
                } else {
                    format!("differs in {}", off.join(", "))
                },
                verdict(ok)
            );
            bits_ok &= ok;
        }
        m.set_mode(StepMode::Graph);
        // (j2) nodes: the pass of slots 0 and 1 at a row each against the
        // two-row pass of slot 0, one busy slot's launches apart; each pass
        // copies one row a head.
        let n_layer = m.body("one pass")?.hparams().n_layer;
        let two = m.capture_slots(&[(0, 1), (1, 1)])?;
        let two_row = m.capture_slots(&[(0, 2)])?;
        let one_row = m.capture_slots(&[(0, 1)])?;
        let mut memcpy_ok = true;
        for (key, rows) in [
            (&[(0, 1), (1, 1)][..], 2),
            (&[(0, 2)][..], 2),
            (&[(0, 1)][..], 1),
        ] {
            let ([memcpy], _) = count_kinds(
                &m.slots_graph_nodes(key)?,
                [sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_MEMCPY],
            );
            memcpy_ok &= memcpy == rows;
        }
        let per_slot = slot_nodes(n_layer);
        let nodes_ok = two == two_row + per_slot && memcpy_ok;
        // Against the one-row pass, the two-slot pass also holds the m = 1 to
        // m >= 2 step of the row-wise launches and the second row's head: its
        // three launches (the step's head) and its copy.
        let head = NODES_CHAIN - NODES_PASS_1 + 1;
        let from_one = per_slot + (NODES_PASS_M - NODES_PASS_1) + head;
        println!(
            "one pass nodes: slots 0 and 1 at a row each {two} = the two-row pass of slot 0 \
             {two_row} + a slot's {per_slot} (want {}); against the one-row pass {one_row}: +{} \
             (derived +{from_one}: a slot's {per_slot}, the m>=2 launch step {}, a row's head \
             {head}); one memcpy a row in each {memcpy_ok} {}",
            two_row + per_slot,
            two.wrapping_sub(one_row),
            NODES_PASS_M - NODES_PASS_1,
            verdict(nodes_ok)
        );
        // (j3) refusals by name, nothing moved.
        let at = |m: &mut Qwen3moeModel| -> Result<[u32; 2], GateError> {
            m.select_slot(1)?;
            let p1 = m.pos();
            m.select_slot(0)?;
            Ok([m.pos(), p1])
        };
        let before = at(m)?;
        let ids5 = [1u32; 5];
        let ids4 = [1u32; 4];
        let refusals = [
            (
                "nine rows",
                refused(
                    m.step_slots(&[(0, &ids5[..]), (1, &ids4[..])]),
                    WHAT,
                    "9 rows in one pass",
                ),
            ),
            (
                "a slot twice",
                refused(m.step_slots(&[(0, &[1]), (0, &[1])]), WHAT, "slot 0 twice"),
            ),
            (
                "no slot",
                refused(m.step_slots(&[]), WHAT, "a pass of no slot"),
            ),
            (
                "slot out of range",
                refused(
                    m.step_slots(&[(2, &[1])]),
                    WHAT,
                    "slot 2 of a model that serves 0..2",
                ),
            ),
            (
                "a slot of no token",
                refused(m.step_slots(&[(1, &[])]), WHAT, "slot 1 with no token"),
            ),
        ];
        let still = at(m)? == before;
        m.seed_depth(CTX - 4)?;
        let past = refused(
            m.step_slots(&[(0, &[1; 8])]),
            WHAT,
            &format!(
                "slot 0's 8 rows from position {} pass the resident cache's {CTX} rows",
                CTX - 4
            ),
        );
        m.reset()?;
        let missed: Vec<&str> = refusals
            .iter()
            .filter_map(|&(name, ok)| (!ok).then_some(name))
            .chain((!past).then_some("past the cache"))
            .collect();
        let refused_ok = missed.is_empty() && still;
        println!(
            "one pass refusals: nine rows, a slot twice, no slot, a slot out of range, a slot of \
             no token and rows past the cache each refused by name{}; positions unmoved {still} {}",
            if missed.is_empty() {
                String::new()
            } else {
                format!(" — except {}", missed.join(", "))
            },
            verdict(refused_ok)
        );
        // (j4) a fault in a pass poisons both slots; each slot's reset takes
        // it off the set, and only both lift it.
        m.select_slot(0)?;
        m.reset()?;
        m.prefill_with(a, PrefillPath::Pass)?;
        m.select_slot(1)?;
        m.reset()?;
        m.prefill_with(b, PrefillPath::Pass)?;
        plant_fault(m)?;
        let read = m.step_slots(&[(0, &[sa.ids[0]]), (1, &[sb.ids[0]])]);
        let raised = matches!(read, Err(GpuError::Fault { .. })) && m.poisoned().is_some();
        let set = "slots 0 and 1 are poisoned";
        m.select_slot(0)?;
        let named0 = refused(m.step(&[1]), "GpuModel::step", set);
        m.select_slot(1)?;
        let named1 = refused(m.step(&[1]), "GpuModel::step", set);
        let named_pass = refused(m.step_slots(&[(0, &[1]), (1, &[1])]), WHAT, set);
        m.select_slot(0)?;
        m.reset()?;
        let stands = refused(m.step(&[1]), "GpuModel::step", "slot 1 is poisoned");
        m.select_slot(1)?;
        m.reset()?;
        let lifted = m.poisoned().is_none();
        // Each slot from its own reset again, so this check stands apart from
        // the refusals before it.
        m.select_slot(0)?;
        m.reset()?;
        let a0 = m.prefill_with(a, PrefillPath::Pass)?;
        m.select_slot(1)?;
        m.reset()?;
        let b0 = m.prefill_with(b, PrefillPath::Pass)?;
        let again = m.step_slots(&[(0, &[a0]), (1, &[b0])])?;
        let again_logits: Vec<u64> = m
            .slots_logits()?
            .iter()
            .map(|l| Fnv1a64::default().f32s(l).value())
            .collect();
        let solo_again = [a0, b0] == [sa.ids[0], sb.ids[0]]
            && again.ids == [sa.ids[1], sb.ids[1]]
            && again_logits == [sa.logits[0], sb.logits[0]];
        m.select_slot(0)?;
        m.reset()?;
        m.select_slot(1)?;
        m.reset()?;
        let fault_ok = raised && named0 && named1 && named_pass && stands && lifted && solo_again;
        println!(
            "one pass fault: the pass after a planted fault {}; both slots and the pass refused \
             naming the set ({named0}, {named1}, {named_pass}), after slot 0's reset still \
             refused naming slot 1 {stands}, after both resets lifted {lifted}, the next pass \
             bit for bit the solo runs' first step {solo_again} {}",
            match &read {
                Ok(o) => format!("returned ids {:?}", o.ids),
                Err(e) => format!("returned \"{e}\""),
            },
            verdict(fault_ok)
        );
        let ok = bits_ok && nodes_ok && refused_ok && fault_ok;
        println!("one pass: {}", verdict(ok));
        Ok(ok)
    }

    // ------------------------------------------------------------- ppl

    /// The chain as the PPL scorer's model.
    struct Scored<'a>(&'a mut Qwen3moeModel);

    impl PplModel for Scored<'_> {
        fn reset(&mut self) -> Result<(), GateError> {
            Ok(self.0.reset()?)
        }

        fn step(&mut self, token: u32) -> Result<(), GateError> {
            self.0.step(&[token])?;
            Ok(())
        }

        fn logits(&mut self) -> Result<Vec<f32>, GateError> {
            Ok(self.0.logits()?)
        }
    }

    /// `--ppl TAG [--placed]` (module doc): on the whole-card load, or with
    /// `--placed` on (o)'s placed load at the base's context.
    fn ppl(tag: &str, placed: Option<HostCfg>) -> Result<(), GateError> {
        let path: PathBuf = data_dir().join("ikppl").join(format!("{tag}.kld"));
        let base = KldBase::open_own_vocab(&path)?;
        println!(
            "ppl base {}: ctx {} chunks {} scored per chunk {} from {}",
            path.display(),
            base.n_ctx(),
            base.n_chunk(),
            base.scored_per_chunk(),
            base.first_scored()
        );
        let mut m = match placed {
            None => open(base.n_ctx(), StepMode::Graph, KvQ8::F16)?,
            Some(host) => {
                let mut m = open_placed(base.n_ctx(), host)?
                    .ok_or("--placed: the plan leaves no routed expert on one side")?;
                m.set_mode(StepMode::Graph);
                m
            }
        };
        let n_vocab = m.body("ppl")?.hparams().n_vocab;
        if base.n_vocab() != n_vocab {
            return Err(format!(
                "{} holds {} vocabulary entries per record, the model {n_vocab}",
                path.display(),
                base.n_vocab()
            )
            .into());
        }
        let s = score_ppl(&base, &mut Scored(&mut m))?;
        let nf = s.n as f64;
        println!(
            "ppl: {} positions; PPL ours {:.4} ik {:.4}; d = NLL_ours − NLL_ik mean {:+.5} ± \
             {:.5} (SE), sd(d) {:.4}; Δ_PPL {:+.3} %; KLD(ik‖ours) {:.5} ± {:.5}; same top \
             {:.2} %; σ_rel (rms of margin ours − ik) {:.4}; {:.0} s (printed, not judged)",
            s.n,
            s.ppl_ours,
            s.ppl_ik,
            s.mean_d,
            s.se_d,
            s.sd_d,
            100.0 * s.dppl(),
            s.kld,
            s.kld_se,
            100.0 * s.same_top as f64 / nf,
            s.sigma_rel,
            s.secs
        );
        Ok(())
    }
}
